# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""Stage3 rolling pipeline with explicit execution modes.

One class serves four rollouts that share the whole rolling body and differ only in what they keep:
``GENERATOR_DMD_TRAINING`` (gradients on the selected windows, DMD backward per window),
``FAKE_SCORE_TRAINING`` (same window selection, no gradients), ``SAMPLING`` (distilled inference: no
window is selected, every chunk drains to sigma=0) and ``I2V_ANCHOR_GENERATION`` (runs just far enough to
pop the first clean chunk and returns its first frame). One shared loop is what keeps distilled inference
and the training rollout numerically the same rollout.
"""

import dataclasses
from contextlib import contextmanager
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from movieroll.commons.streaming_config import RollingStageType
from movieroll.models.transformers.modules.context_noise import apply_context_noise
from movieroll.models.transformers.modules.chunking import check_and_get_chunk_sizes
from movieroll.models.transformers.modules.rolling_queue import RollingQueue
from movieroll.pipelines.movieroll_pipeline import MovieRollStreamingPipeline
from movieroll.pipelines.rolling_dmd_runtime import (
    DMDWindowBackwardHandler,
    RollingContext,
    RollingMode,
    RollingResult,
    RollingState,
    RollingTrainingObjective,
    WindowPrediction,
    WindowSelectionPlan,
    build_denoising_range,
    build_window_selection_plan,
    splice_window_predictions,
)
from movieroll.schedulers.scheduling_flow_match_streaming import (
    FlowMatchStreamingScheduler,
    StreamingSchedulerWrapper,
)
from movieroll.utils.communications import bcast_in_group, is_group_rank0
from movieroll.commons.parallel_states import get_sequence_parallel_state, nccl_info

# The distilled ladder has one entry per window slot (len(denoising_step_list) == window_size), so every rolling
# round is a single denoising step.
STEPS_PER_WINDOW = 1


class MovieRollDMDPipeline(MovieRollStreamingPipeline):
    """Stage3 rolling DMD, fake-score rollout, sampling, and i2v bootstrap."""

    _SHARD_DECISION_SEED_DOMAIN = 19_337_771
    _UNIFORM_NOISE_SEED_DOMAIN = 29_337_773
    _UNIFORM_DECISION_SEED_DOMAIN = 29_337_781

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rollout_sync_group = None
        self._sync_draw_counter = 0
        self._sync_group_id = 0
        self._sync_draw_base_seed = 42

    # ---- Synchronized draws and the rollout lifecycle ----
    # Any decision that changes the SHAPE of a rollout (which window is supervised, in what order) must come
    # out identical on every rank of an FSDP shard group, or the ranks desynchronise and the collectives
    # deadlock. So the draw happens on the group's rank 0 from a shard-specific generator and the result is
    # broadcast, instead of every rank drawing from a shared seed.
    def set_sync_group(self, group) -> None:
        """Inject the FSDP shard group used for rollout-shape-affecting decisions."""

        if dist.is_initialized() and dist.get_world_size() > 1 and group is None:
            raise ValueError("distributed rolling execution requires an explicit shard group")
        self._rollout_sync_group = group
        self._sync_group_id = (
            int(dist.get_process_group_ranks(group)[0])
            if dist.is_initialized() and group is not None
            else 0
        )

    def _sync_group(self):
        if (
            dist.is_initialized()
            and dist.get_world_size() > 1
            and self._rollout_sync_group is None
        ):
            raise RuntimeError(
                "distributed rolling execution requires set_sync_group() "
                "before any synchronized draw"
            )
        return self._rollout_sync_group

    @contextmanager
    def rollout_lifecycle(self):
        """Always release per-rollout queue/cache state, including on callback errors."""

        try:
            yield
        finally:
            progress = getattr(self, "_active_rollout_progress_bar", None)
            if progress is not None:
                progress.close()
                self._active_rollout_progress_bar = None
            queue = getattr(self, "rolling_queue", None)
            if queue is not None:
                queue.clear()
            manager = getattr(self.transformer, "kv_cache_manager", None)
            if manager is not None:
                manager.clear_kv_cache()

    def _broadcast_in_sync_group(self, tensor: torch.Tensor) -> torch.Tensor:
        return bcast_in_group(tensor, self._sync_group())

    def reset_sync_random(self, generator: Optional[torch.Generator]) -> None:
        """Start a new source-only decision stream for one rollout."""

        self._sync_draw_base_seed = (
            int(generator.initial_seed()) if generator is not None else 42
        )
        self._sync_draw_counter = 0

    def _make_sync_source_generator(self) -> Optional[torch.Generator]:
        """Return a shard-specific RNG only on this shard group's rank 0."""

        self._sync_draw_counter += 1
        if dist.is_initialized() and not is_group_rank0(self._sync_group()):
            return None
        seed = (
            self._sync_draw_base_seed
            + self._SHARD_DECISION_SEED_DOMAIN
            + self._sync_group_id * 1_000_003
            + self._sync_draw_counter * 10_007
        )
        return torch.Generator(device="cpu").manual_seed(seed)

    def draw_synced_int(self, high: int, *, device: torch.device) -> int:
        """Draw on the shard-group leader and explicitly broadcast the result."""

        if high <= 0:
            raise ValueError(f"high must be > 0, got {high}")
        if high == 1:
            return 0
        generator = self._make_sync_source_generator()
        if dist.is_initialized():
            value = torch.empty(1, device=device, dtype=torch.long)
            if generator is not None:
                value.copy_(
                    torch.randint(
                        0, high, (1,), generator=generator, dtype=torch.long
                    ).to(device)
                )
            self._broadcast_in_sync_group(value)
        else:
            value = torch.randint(
                0, high, (1,), generator=generator, dtype=torch.long
            )
        return int(value.item())

    def draw_synced_permutation(
        self, length: int, *, device: torch.device
    ) -> torch.Tensor:
        """Draw on the shard-group leader and broadcast the permutation."""

        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        generator = self._make_sync_source_generator()
        if dist.is_initialized():
            value = torch.empty(length, device=device, dtype=torch.long)
            if generator is not None:
                value.copy_(
                    torch.randperm(
                        length, generator=generator, dtype=torch.long
                    ).to(device)
                )
            self._broadcast_in_sync_group(value)
        else:
            value = torch.randperm(
                length, generator=generator, dtype=torch.long
            )
        return value

    # ------------------------------------------------------------------
    # Rollout planning and the execute() entry points
    # ------------------------------------------------------------------
    def configure_uniform_timestep_regularization(
        self,
        *,
        probability: float,
        candidate_step_indices: list[int],
    ) -> None:
        """Configure the loss-only uniform-timestep training branch."""

        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("uniform regularization probability must be in [0, 1]")
        candidates = tuple(int(index) for index in candidate_step_indices)
        if not candidates or any(index <= 0 for index in candidates):
            raise ValueError(
                "uniform regularization requires nonzero step indices"
            )
        if len(set(candidates)) != len(candidates):
            raise ValueError("uniform regularization step indices must be unique")
        self._uniform_reg_enabled = True
        self._uniform_reg_probability = float(probability)
        self._uniform_reg_step_indices = candidates
        self._uniform_noise_counter = 0
        self._uniform_decision_counter = 0

    def _draw_uniform_decision(self, high: int, *, device: torch.device) -> int:
        """Draw a uniform-branch decision without touching the rolling RNG stream.

        Shares the shard-group broadcast of ``draw_synced_int`` but keeps its own counter and seed
        domain, so enabling the uniform branch never shifts the seeds of later recompute draws.
        """

        if high <= 0:
            raise ValueError(f"high must be > 0, got {high}")
        if high == 1:
            return 0
        self._uniform_decision_counter += 1
        seed = (
            self._sync_draw_base_seed
            + self._UNIFORM_DECISION_SEED_DOMAIN
            + self._sync_group_id * 1_000_003
            + self._uniform_decision_counter * 10_007
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if dist.is_initialized():
            value = torch.empty(1, device=device, dtype=torch.long)
            if is_group_rank0(self._sync_group()):
                value.copy_(
                    torch.randint(
                        0, high, (1,), generator=generator, dtype=torch.long
                    ).to(device)
                )
            self._broadcast_in_sync_group(value)
        else:
            value = torch.randint(
                0, high, (1,), generator=generator, dtype=torch.long
            )
        return int(value.item())

    def _choose_training_objective(
        self, *, mode: RollingMode, device: torch.device
    ) -> tuple[RollingTrainingObjective, Optional[int]]:
        if mode not in (
            RollingMode.GENERATOR_DMD_TRAINING,
            RollingMode.FAKE_SCORE_TRAINING,
        ) or not getattr(self, "_uniform_reg_enabled", False):
            return RollingTrainingObjective.ROLLING_DMD, None

        probability = float(self._uniform_reg_probability)
        use_uniform = probability >= 1.0
        if 0.0 < probability < 1.0:
            scale = 1_000_000
            use_uniform = (
                self._draw_uniform_decision(scale, device=device)
                < round(probability * scale)
            )
        if not use_uniform:
            return RollingTrainingObjective.ROLLING_DMD, None

        candidates = self._uniform_reg_step_indices
        choice = self._draw_uniform_decision(len(candidates), device=device)
        return RollingTrainingObjective.UNIFORM_TIMESTEP_DMD, candidates[choice]

    def _select_training_recompute_chunks(self, output_latents, event_split_indices):
        """Recompute selection of the training rollouts: sink + recent_k plus randomly drawn middle chunks.

        Keeps chunk 0 and the recent_k most recent chunks, and adds mid_min..mid_max middle chunks drawn
        once per shard group and broadcast; event boundaries are ignored. SAMPLING instead uses the base
        pipeline's deterministic `_select_recompute_chunks`, and I2V_ANCHOR_GENERATION returns at the
        first DECODE round, before any recompute decision.
        """

        del event_split_indices
        recent_k = int(self.streaming_config.kv_recompute_recent_k)
        total = len(output_latents)
        indices = {0}
        indices.update(range(max(0, total - recent_k), total))
        middle = [
            index
            for index in range(1, max(1, total - recent_k))
            if index not in indices
        ]
        if middle:
            high = min(int(getattr(self, "_recompute_mid_max", 5)), len(middle))
            low = min(int(getattr(self, "_recompute_mid_min", 1)), high)
            device = self.transformer.device
            count = low + self.draw_synced_int(
                high - low + 1, device=device
            )
            permutation = self.draw_synced_permutation(
                len(middle), device=device
            )
            indices.update(
                middle[int(permutation[index].item())]
                for index in range(count)
            )
        return sorted(indices)

    def _compute_rollout_event_split_indices(
        self,
        num_chunks: int,
        chunk_sizes: List[int],
        event_prompts: Optional[List[str]],
        event_timestamps: Optional[List[List[float]]],
        rank: int = 0,
    ):
        """Event splits of a Stage3 rollout: the base even split without timestamps (sampling), otherwise the
        training rollout's pixel-frame boundaries, each mapped to the nearest chunk boundary."""

        if event_timestamps is None or event_prompts is None or len(event_prompts) < 2:
            split_indices, kept = self._compute_event_split_indices(num_chunks, event_prompts, rank=-1)
        else:
            if len(chunk_sizes) != num_chunks:
                raise ValueError(
                    "event split chunk count mismatch: "
                    f"num_chunks={num_chunks}, chunk_sizes={len(chunk_sizes)}"
                )
            if len(event_timestamps) != len(event_prompts):
                raise ValueError(
                    "event timestamp count must match event prompts: "
                    f"{len(event_timestamps)} != {len(event_prompts)}"
                )

            # A split can only fall between latent chunks: every representable latent boundary is mapped
            # back to its exact 1-based pixel frame, then the requested boundary snaps to the nearest one.
            # An exact tie prefers the later boundary, so a half-chunk event stays with the earlier prompt.
            candidates = []
            latent_prefix = 0
            for split_index, chunk_size in enumerate(chunk_sizes[:-1], start=1):
                latent_prefix += int(chunk_size)
                pixel_frame = (latent_prefix - 1) * 4 + 1
                candidates.append((pixel_frame, split_index))
            if not candidates:
                raise ValueError("multi-event Stage3 rollout requires at least two chunks")

            split_indices = []
            for event_index in range(1, len(event_prompts)):
                requested = float(event_timestamps[event_index][0])
                _, split_index = min(
                    candidates,
                    key=lambda item: (
                        abs(float(item[0]) - requested),
                        -float(item[0]),
                    ),
                )
                split_indices.append(split_index)

            boundaries = [0] + split_indices + [num_chunks]
            kept = [
                event_index
                for event_index in range(len(boundaries) - 1)
                if boundaries[event_index] < boundaries[event_index + 1]
            ]
            split_indices = [
                split_indices[event_index - 1]
                for event_index in kept[1:]
            ]

        if rank == 0 and event_prompts is not None and len(event_prompts) >= 2:
            mode = "pixel-frame" if event_timestamps is not None else "even"
            print(
                f"[event split] {mode} mode: {len(event_prompts)} events, "
                f"split at chunks {split_indices}/{num_chunks}"
            )
        return split_indices, kept

    def _assert_cached_output_identity(self, state: RollingState) -> None:
        """Check that tracked output-chunk identities match the real KV depth."""

        manager = getattr(self.transformer, "kv_cache_manager", None)
        if manager is None or not hasattr(
            manager, "get_cached_cond_latents_num_frames"
        ):
            raise RuntimeError(
                "Stage3 transformer must expose KV cache frame accounting"
            )
        tracked = state.cached_output_chunk_indices
        if tracked != sorted(set(tracked)):
            raise RuntimeError(
                f"cached output chunk identities are not unique/increasing: {tracked}"
            )
        if any(index < 0 or index >= len(state.finalized_chunks) for index in tracked):
            raise RuntimeError(
                "cached output chunk identity points outside finalized chunks"
            )
        expected_frames = sum(
            int(state.finalized_chunks[index].shape[2]) for index in tracked
        )
        actual_frames = int(manager.get_cached_cond_latents_num_frames())
        if actual_frames != expected_frames:
            raise RuntimeError(
                "KV cache/output identity mismatch: "
                f"tracked={tracked}, expected_frames={expected_frames}, "
                f"cached_frames={actual_frames}"
            )

    def _build_training_window_plan(
        self, *, num_chunks: int, window_size: int, device: torch.device
    ) -> WindowSelectionPlan:
        """Draw the shard-synchronized phase and build the fixed Stage3 plan."""
        start_phase = self.draw_synced_int(window_size, device=device)
        return build_window_selection_plan(
            num_chunks=num_chunks,
            window_size=window_size,
            start_phase=start_phase,
        )

    def execute(
        self,
        *,
        mode: RollingMode,
        dmd_backward_handler: Optional[DMDWindowBackwardHandler] = None,
        teacher_unconditional_condition: Optional[tuple] = None,
        cond_latents: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> RollingResult:
        """Execute one explicit Stage3 rolling mode."""
        if not isinstance(mode, RollingMode):
            raise TypeError(f"mode must be RollingMode, got {type(mode)!r}")
        if mode is RollingMode.I2V_ANCHOR_GENERATION:
            raise ValueError(
                "I2V_ANCHOR_GENERATION is internal to execute_i2v_self_anchored, which runs it without "
                "gradients and detaches the anchor"
            )
        if mode is RollingMode.GENERATOR_DMD_TRAINING and dmd_backward_handler is None:
            raise ValueError("generator DMD training requires dmd_backward_handler")
        if (
            mode is RollingMode.GENERATOR_DMD_TRAINING
            and teacher_unconditional_condition is None
        ):
            raise ValueError(
                "generator DMD training requires a teacher unconditional condition"
            )
        if mode is not RollingMode.GENERATOR_DMD_TRAINING and dmd_backward_handler is not None:
            raise ValueError(f"{mode.value} must not receive a DMD backward handler")
        if (
            mode is not RollingMode.GENERATOR_DMD_TRAINING
            and teacher_unconditional_condition is not None
        ):
            raise ValueError(
                f"{mode.value} must not receive a teacher unconditional condition"
            )
        with self.rollout_lifecycle():
            return self._execute_rollout(
                mode=mode,
                dmd_backward_handler=dmd_backward_handler,
                teacher_unconditional_condition=teacher_unconditional_condition,
                cond_latents=cond_latents,
                **kwargs,
            )

    def execute_i2v_self_anchored(
        self,
        *,
        mode: RollingMode,
        dmd_backward_handler: Optional[DMDWindowBackwardHandler] = None,
        **kwargs,
    ) -> RollingResult:
        """Generate one detached anchor frame, then run the independent i2v rollout."""
        if mode is RollingMode.I2V_ANCHOR_GENERATION:
            raise ValueError(
                "execute_i2v_self_anchored requires a final training or sampling mode"
            )
        rollout_kwargs = dict(kwargs)
        teacher_unconditional_condition = rollout_kwargs.pop(
            "teacher_unconditional_condition", None
        )
        anchor_kwargs = dict(rollout_kwargs)
        anchor_kwargs["multitask_type"] = "t2v"
        anchor_kwargs["event_prompts"] = None
        anchor_kwargs["event_timestamps"] = None
        with self.rollout_lifecycle(), torch.no_grad():
            anchor = self._execute_rollout(
                mode=RollingMode.I2V_ANCHOR_GENERATION,
                dmd_backward_handler=None,
                teacher_unconditional_condition=None,
                cond_latents=None,
                **anchor_kwargs,
            ).detach()
        rollout_kwargs["multitask_type"] = "i2v"
        return self.execute(
            mode=mode,
            dmd_backward_handler=dmd_backward_handler,
            teacher_unconditional_condition=teacher_unconditional_condition,
            cond_latents=anchor,
            **rollout_kwargs,
        )

    def encode_text_condition(
        self,
        prompt: Union[str, List[str]],
    ) -> tuple:
        """Encode one conditional-only text context for DMD score evaluation.

        Generator rolling never performs classifier-free guidance, so the frozen real-score teacher's
        unconditional context is prepared by the Stage3 trainer through this boundary instead.
        """

        self._use_cfg = False
        condition = self._encode_single_event_prompt(
            prompt,
            device=self.transformer.device,
            negative_prompt=None,
        )
        return condition

    def _prepare_rollout_context(
        self,
        *,
        mode: RollingMode,
        all_event_conditions: list[tuple],
        chunk_sizes: list[int],
        timesteps: torch.Tensor,
        event_index_by_chunk: list[int],
        generated_num_chunks: int,
        window_size: int,
        device: torch.device,
    ) -> tuple[RollingContext, Optional[WindowSelectionPlan]]:
        """Build the immutable execution context for one rolling trajectory."""

        selection_plan: Optional[WindowSelectionPlan] = None
        if mode in (
            RollingMode.GENERATOR_DMD_TRAINING,
            RollingMode.FAKE_SCORE_TRAINING,
        ):
            selection_plan = self._build_training_window_plan(
                num_chunks=generated_num_chunks,
                window_size=window_size,
                device=device,
            )
            selected = selection_plan.selected_round_indices
        elif mode in (RollingMode.SAMPLING, RollingMode.I2V_ANCHOR_GENERATION):
            # Sampling drains every real chunk to sigma=0 and returns those finalized latents directly;
            # a direct-x0 window would overwrite the clean tail with mixed-timestep predictions. Anchor
            # generation returns at the first DECODE pop and needs no window either.
            selected = ()
        else:
            raise ValueError(f"unsupported rolling mode: {mode!r}")

        training_objective, uniform_step_index = (
            self._choose_training_objective(mode=mode, device=device)
        )
        if (
            uniform_step_index is not None
            and uniform_step_index >= len(timesteps)
        ):
            raise ValueError(
                "uniform regularization step index exceeds scheduler ladder: "
                f"{uniform_step_index} >= {len(timesteps)}"
            )

        return (
            RollingContext(
                mode=mode,
                event_conditions=tuple(all_event_conditions),
                chunk_sizes=tuple(chunk_sizes),
                timesteps=timesteps,
                event_index_by_chunk=tuple(event_index_by_chunk),
                selected_round_indices=frozenset(selected),
                training_objective=training_objective,
                uniform_step_index=uniform_step_index,
            ),
            selection_plan,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def _execute_rollout(
        self,
        *,
        mode: RollingMode,
        dmd_backward_handler: Optional[DMDWindowBackwardHandler],
        teacher_unconditional_condition: Optional[tuple],
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        denoising_step_list: List[int],
        cond_latents: Optional[torch.Tensor] = None,
        context_noise: float = 0.0,
        # Multi-event (multi-shot): with at least two prompts the prompt switches per round; None or a
        # single element means one prompt for the whole rollout.
        event_prompts: Optional[List[str]] = None,
        event_timestamps: Optional[List[List[float]]] = None,
        generator: Optional[torch.Generator] = None,
        multitask_type: Optional[str] = None,
    ) -> RollingResult:
        """Execute one rolling trajectory and return typed state."""
        training_mode = mode in (
            RollingMode.GENERATOR_DMD_TRAINING,
            RollingMode.FAKE_SCORE_TRAINING,
        )
        if training_mode and (
            event_prompts is None or len(event_prompts) not in (1, 2)
        ):
            raise ValueError(
                f"{mode.value} requires exactly 1 or 2 event prompts (the batch decision "
                "stream draws one switch frame); "
                f"got {0 if event_prompts is None else len(event_prompts)}"
            )
        if mode is RollingMode.SAMPLING and (
            event_prompts is None or len(event_prompts) < 1
        ):
            raise ValueError("sampling requires at least one ordered event prompt")
        if event_prompts is not None and any(
            not isinstance(item, str) or not item.strip()
            for item in event_prompts
        ):
            raise ValueError("event_prompts must contain only non-empty strings")
        if mode is RollingMode.SAMPLING and event_timestamps is not None:
            raise ValueError(
                "Stage3 sampling uses even event splits and must not receive event_timestamps"
            )
        if training_mode and len(event_prompts) == 1 and event_timestamps is not None:
            raise ValueError("single-event training must not pass event_timestamps")
        if training_mode and len(event_prompts) == 2 and (
            event_timestamps is None or len(event_timestamps) != 2
        ):
            raise ValueError("dual-event training requires two event timestamp ranges")
        if training_mode and len(event_prompts) == 2:
            if any(len(frame_range) != 2 for frame_range in event_timestamps):
                raise ValueError("each event timestamp range must be [start, end]")
            first, second = event_timestamps
            if (
                float(first[0]) != 0.0
                or float(first[1]) != float(second[0])
                or float(second[1]) != float(video_length)
                or not 0.0 < float(first[1]) < float(video_length)
            ):
                raise ValueError(
                    "dual-event timestamps must be contiguous and cover the "
                    f"full pixel timeline, got {event_timestamps}"
                )
        task_type = multitask_type or "t2v"
        if task_type not in {"t2v", "i2v"}:
            raise ValueError(f"unsupported Stage3 multitask_type: {task_type!r}")
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError(
                f"Stage3 height/width must be positive multiples of 16, got {height}x{width}"
            )
        if video_length <= 0 or (video_length - 1) % 4:
            raise ValueError(
                "Stage3 video_length must be positive and satisfy 4n+1 pixel "
                f"frames, got {video_length}"
            )
        if task_type == "i2v" and cond_latents is None:
            raise ValueError(
                "i2v rollout requires an anchor: the input image latent (SAMPLING) or "
                "execute_i2v_self_anchored() (training)"
            )
        if task_type == "t2v" and cond_latents is not None:
            raise ValueError("t2v rollout must not receive cond_latents")

        target_dtype = self.target_dtype
        device = self.transformer.device
        autocast_enabled = target_dtype != torch.float32

        if self.streaming_config is None:
            raise RuntimeError("Stage3 rolling execution requires streaming_config")
        num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        if not 0.0 <= float(context_noise) < 1.0:
            raise ValueError(
                "context_noise must be a normalized sigma ratio in [0, 1), "
                f"got {context_noise}"
            )
        if not denoising_step_list:
            raise ValueError("denoising_step_list must not be empty")
        if any(
            denoising_step_list[index] <= denoising_step_list[index + 1]
            for index in range(len(denoising_step_list) - 1)
        ):
            raise ValueError("denoising_step_list must be strictly descending")
        if any(
            float(step) <= 0 or float(step) > num_train_timesteps
            for step in denoising_step_list
        ):
            raise ValueError(
                "Stage3 denoising steps must lie in "
                f"(0, {num_train_timesteps}], got {denoising_step_list}"
            )
        window_size = int(self.streaming_config.window_size)
        if window_size != len(denoising_step_list):
            # Both directions matter: the loop only emits step indices 0..window_size-1, so a longer
            # ladder silently abandons its tail and leaves the sample under-denoised, while a shorter one
            # is an IndexError at `timesteps[step_index]`.
            raise ValueError(
                f"streaming window size must match len(denoising_step_list); got {window_size} and "
                f"{len(denoising_step_list)}. The rollout runs one denoising step per window slot, so a "
                f"longer ladder would leave its last {max(len(denoising_step_list) - window_size, 0)} "
                f"step(s) unused and return an under-denoised sample, and a shorter one would index past "
                f"its end"
            )
        # Stage3 is conditional-only and this entry point bypasses the base __call__, so the inherited CFG
        # state is pinned by hand: do_classifier_free_guidance is a read-only property over _use_cfg.
        self._use_cfg = False

        # ---- 1. window / step count (one denoising step per window slot: window == num_denoising_steps) ----
        num_inference_steps = window_size * STEPS_PER_WINDOW
        self.reset_sync_random(generator)

        # ---- 2. encode prompts -> all_event_conds ----
        # Multi-event: encode every event separately; a single event falls back to one prompt.
        rank = dist.get_rank() if dist.is_initialized() else 0
        use_multi_event = event_prompts is not None and len(event_prompts) >= 2

        def _encode_one(p):
            return self.encode_text_condition(p)

        all_event_conds = [
            _encode_one(event_prompt)
            for event_prompt in (event_prompts or [prompt])
        ]
        prompt_embeds = all_event_conds[0][0]

        # ---- 3. chunk_sizes / timesteps / scheduler_wrapper / rolling_queue ----
        chunk_size = self.streaming_config.chunk_size
        latent_length = (video_length - 1) // 4 + 1
        tt = latent_length // self.transformer.patch_size[0]
        # The chunk layout follows the task type (see modules/chunking.py):
        #   t2v (mode='default') -> [1+chunk_size, chunk_size, ...] = [5,4,4,...]
        #   i2v (mode='i2v')     -> [1,            chunk_size, ...] = [1,4,4,...]
        # 4n+1 frames leave one frame over, which is either swallowed by the first chunk (t2v => 5) or
        # becomes a reference-frame chunk of its own (i2v => 1). SFT trains both layouts.
        chunk_mode = "i2v" if task_type == "i2v" else "default"
        # tt must satisfy tt % chunk_size == 1: otherwise the first chunk is no longer 5 (t2v) / 1 (i2v)
        # and the differentiable window holds half as many frames, so this raises instead of falling back.
        if tt % chunk_size != 1:
            raise ValueError(
                f"latent frame count tt={tt} must satisfy tt % chunk_size({chunk_size}) == 1 (i.e. 4n+1 pixel "
                f"frames), otherwise the first chunk is not {1 + chunk_size} (t2v) / 1 (i2v)."
                f" video_length={video_length}, latent_length={latent_length}"
            )
        chunk_sizes_in_patches = check_and_get_chunk_sizes(
            tt, chunk_size=chunk_size,
            mode=chunk_mode,
        )
        chunk_sizes = [cs * self.transformer.patch_size[0] for cs in chunk_sizes_in_patches]
        num_chunks = len(chunk_sizes)
        expected_first = 1 if chunk_mode == "i2v" else 1 + chunk_size
        if (
            chunk_sizes[0] != expected_first
            or not set(chunk_sizes[1:]) <= {chunk_size}
        ):
            raise RuntimeError(
                f"chunk layout does not match the contract: mode={chunk_mode} "
                f"expected first chunk={expected_first} and {chunk_size} for the rest, "
                f"got {chunk_sizes[:3]}... ({num_chunks} chunks in total)"
            )

        # ---- 3b. multi-event: map the events onto chunks and build chunk_to_event ----
        # chunk_to_event[c] = the event segment this chunk belongs to (0-based, after subsetting to kept events).
        if use_multi_event:
            event_split_indices, kept_event_indices = self._compute_rollout_event_split_indices(
                num_chunks, chunk_sizes, event_prompts, event_timestamps, rank=rank,
            )
            if len(kept_event_indices) != len(all_event_conds):
                raise ValueError(
                    "Stage3 event mapping produced an empty segment; every supplied "
                    "event must retain at least one chunk. "
                    f"events={len(all_event_conds)}, kept={kept_event_indices}, "
                    f"splits={event_split_indices}, num_chunks={num_chunks}"
                )
            if len(event_split_indices) != len(all_event_conds) - 1:
                raise RuntimeError(
                    "event split count does not match encoded event conditions: "
                    f"splits={event_split_indices}, events={len(all_event_conds)}"
                )
            chunk_to_event = [0] * num_chunks
            for ev_idx, boundary_chunk in enumerate(event_split_indices):
                for ci in range(boundary_chunk, num_chunks):
                    chunk_to_event[ci] = ev_idx + 1
        else:
            event_split_indices = []
            chunk_to_event = [0] * num_chunks


        # Drive the streaming scheduler with the few-step custom ladder (denoising_step_list).
        if not isinstance(self.scheduler, FlowMatchStreamingScheduler):
            raise TypeError(
                "MovieRoll rolling DMD requires FlowMatchStreamingScheduler "
                "with tensor step-index support"
            )
        self.scheduler.set_custom_timesteps(denoising_step_list, device=device)
        timesteps = self.scheduler.timesteps
        if len(timesteps) != window_size:
            raise RuntimeError(
                "scheduler changed the Stage3 denoising ladder length: "
                f"expected {window_size}, got {len(timesteps)}"
            )

        self.scheduler_wrapper = StreamingSchedulerWrapper(
            self.streaming_config, self.scheduler, window_size, num_inference_steps,
        )
        self.rolling_queue = RollingQueue()

        # ---- 4. Prepare immutable selection and mutable loop state ----
        output_chunk_sizes = tuple(chunk_sizes)
        output_event_split_indices = tuple(event_split_indices)
        generated_num_chunks = num_chunks - (1 if cond_latents is not None else 0)
        if generated_num_chunks <= 0:
            raise ValueError("rolling execution must contain at least one generated chunk")
        total_rounds = generated_num_chunks + window_size

        context, selection_plan = self._prepare_rollout_context(
            mode=mode,
            all_event_conditions=all_event_conds,
            chunk_sizes=chunk_sizes,
            timesteps=timesteps,
            event_index_by_chunk=chunk_to_event,
            generated_num_chunks=generated_num_chunks,
            window_size=window_size,
            device=device,
        )

        state = RollingState(
            kv_frame_position=0,
            output_frame_count=0,
            num_decoded_chunks=0,
            finalized_chunks=[],
            cached_output_chunk_indices=[],
            window_predictions=[],
        )

        use_sp = get_sequence_parallel_state()
        context_noise_fn = (
            partial(
                apply_context_noise,
                sigma_max=float(context_noise),
                force_equal_sigma=True,
                generator=generator,
            )
            if context_noise > 0
            else None
        )

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        def _sample_noisy_chunk(temporal_size: int) -> torch.Tensor:
            shape = (
                prompt_embeds.shape[0],
                self.transformer.config.in_channels,
                temporal_size,
                latent_height,
                latent_width,
            )
            if use_sp:
                sp_group = nccl_info.sp_group
                if is_group_rank0(sp_group):
                    noise = randn_tensor(
                        shape,
                        generator=generator,
                        device=device,
                        dtype=target_dtype,
                    )
                else:
                    noise = torch.empty(shape, device=device, dtype=target_dtype)
                bcast_in_group(noise, sp_group)
            else:
                noise = randn_tensor(
                    shape, generator=generator, device=device, dtype=target_dtype
                )
            return noise

        if mode is RollingMode.GENERATOR_DMD_TRAINING:
            # WORLD-wide K is validated before any generator forward, and before the anchor KV encode.
            dmd_backward_handler.on_rollout_start(
                num_windows=len(context.selected_round_indices)
            )

        # ---- i2v anchor: the input image's latent at inference, a self-generated frame in training ----
        # It arrives as cond_latents (in training, from a short t2v-layout rollout the caller runs first),
        # is pre-encoded into the KV with the i2v layout (chunk_sizes[0] == 1), appended to finalized_chunks
        # and dropped from chunk_sizes. The handover cannot happen mid-loop: the queue's noisy chunks carry a
        # noise ladder grown from round 0, so a clear() would desynchronise the queue length from the step
        # indices get_step_index returns, which expand_step_index asserts on during DECODE.
        if cond_latents is not None:
            if chunk_mode != "i2v":
                raise ValueError(
                    "passing cond_latents requires the i2v chunk layout (first chunk = 1); "
                    f"multitask_type={multitask_type!r} implies chunk_mode={chunk_mode!r}"
                )
            if cond_latents.shape[2] != 1 or chunk_sizes[0] != 1:
                raise ValueError(
                    "the i2v anchor must be exactly 1 latent frame, "
                    f"got cond_latents={cond_latents.shape[2]}, "
                    f"chunk_sizes[0]={chunk_sizes[0]}"
                )
            anchor = cond_latents.detach().to(device=device, dtype=target_dtype)
            state.finalized_chunks.append(anchor)
            with torch.no_grad():
                # The anchor shares the context-noise helper with the later history and is encoded at timestep 0.
                state.kv_frame_position = self._encode_clean_chunk_kv_with_context_noise(
                    anchor,
                    kv_frame_position=state.kv_frame_position,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                    context_noise_fn=context_noise_fn,
                    use_sp=use_sp,
                )
            state.output_frame_count += anchor.shape[2]
            state.cached_output_chunk_indices.append(0)
            self._assert_cached_output_identity(state)
            # The anchor takes no further part in denoising, so drop it from chunk_sizes.
            chunk_sizes = chunk_sizes[1:]
            num_chunks = len(chunk_sizes)
            # The loop indexes generated chunks, while finalized_chunks and event
            # split metadata remain in full-output coordinates (including anchor).
            chunk_to_event = chunk_to_event[1:]
            if rank == 0:
                print(f"[i2v] anchor pre-encoded into the KV cache (1 latent frame); {num_chunks} chunks "
                      f"left to generate, total_rounds={total_rounds}")

            context = dataclasses.replace(
                context,
                chunk_sizes=tuple(chunk_sizes),
                event_index_by_chunk=tuple(chunk_to_event),
            )

        outer_pbar = tqdm(total=num_chunks, desc="MovieRoll Rolling DMD", disable=rank != 0)
        self._active_rollout_progress_bar = outer_pbar

        for round_idx in range(total_rounds):
            is_prefill = round_idx < window_size
            rolling_stage = RollingStageType.PREFILL if is_prefill else RollingStageType.DECODE

            # DECODE: first pop the head clean chunk as output, then push a new noisy chunk
            if not is_prefill:
                is_global_last = state.num_decoded_chunks == num_chunks - 1
                clean_chunk = self.rolling_queue.pop_clean_chunk()
                state.finalized_chunks.append(clean_chunk)
                if mode is RollingMode.I2V_ANCHOR_GENERATION:
                    anchor = clean_chunk[:, :, :1].detach().clone()
                    return anchor
                if is_global_last:
                    # No later noisy window can consume this chunk. Match the shared
                    # inference pipeline and avoid one unnecessary full-model KV encode.
                    state.output_frame_count += clean_chunk.shape[2]
                    state.num_decoded_chunks += 1
                    outer_pbar.update(1)
                    break
                # kv_recompute fires in two cases, as in the base __call__: the cached-chunk count crosses
                # the threshold, or an event switch. Both plain encode and recompute run the context-noise
                # helper first and encode at timestep 0. An event switch forces the recompute through even
                # when nothing can be dropped, so the new prompt never attends an uncompressed history.
                event_switch = bool(output_event_split_indices) and (
                    len(state.finalized_chunks) in output_event_split_indices
                )
                do_recompute = (
                    getattr(self.streaming_config, "kv_recompute_enabled", False)
                    and hasattr(self.transformer, "kv_cache_manager")
                    and (
                        event_switch
                        or self.transformer.kv_cache_manager.should_recompute(
                            self.streaming_config.kv_recompute_threshold)
                    )
                )
                if do_recompute:
                    # Sampling keeps the base pipeline's deterministic inference rule; the training rollouts draw
                    # their middle chunks at random.
                    if mode is RollingMode.SAMPLING:
                        selected = self._select_recompute_chunks(
                            state.finalized_chunks, list(output_event_split_indices)
                        )
                    else:
                        selected = self._select_training_recompute_chunks(
                            state.finalized_chunks, list(output_event_split_indices)
                        )
                    # When the threshold triggered it and there is nothing to gain, fall back to a plain encode
                    # and save one forward.
                    if len(selected) >= len(state.finalized_chunks) and not event_switch:
                        do_recompute = False
                with torch.no_grad():
                    if do_recompute:
                        state.kv_frame_position = self._kv_recompute(
                            output_latents=[c.detach() for c in state.finalized_chunks],
                                        target_dtype=target_dtype,
                            autocast_enabled=autocast_enabled,
                            context_noise_fn=context_noise_fn,
                            use_sp=use_sp,
                            trigger_reason="event_switch" if event_switch else "threshold",
                            selected_indices=selected,
                        )
                        # The cache now holds exactly the `selected` chunks and their kv_cache_frame_indices
                        # have been rebuilt compactly, so their identity is recorded wholesale.
                        state.cached_output_chunk_indices[:] = sorted(selected)
                    else:
                        state.kv_frame_position = self._encode_clean_chunk_kv_with_context_noise(
                            clean_chunk.detach(),
                            kv_frame_position=state.kv_frame_position,
                            target_dtype=target_dtype,
                            autocast_enabled=autocast_enabled,
                            context_noise_fn=context_noise_fn,
                            use_sp=use_sp,
                        )
                        # what was just encoded into the cache is exactly the chunk that was just popped
                        state.cached_output_chunk_indices.append(
                            len(state.finalized_chunks) - 1
                        )
                    self._assert_cached_output_identity(state)
                # output_frame_count is independent of kv_recompute: generated_latents is always the ordered
                # concatenation of EVERY popped chunk, so it must be accumulated unconditionally.
                state.output_frame_count += clean_chunk.shape[2]
                state.num_decoded_chunks += 1
                outer_pbar.update(1)

            # append a new noisy chunk
            chunk_temporal_size = (
                context.chunk_sizes[round_idx]
                if round_idx < num_chunks
                else context.chunk_sizes[-1]
            )
            self.rolling_queue.add_new_noisy_chunk(_sample_noisy_chunk(chunk_temporal_size))

            # Multi-event: pick the prompt of the event the currently cleanest chunk belongs to (same rule as
            # the base pipeline).
            cleanest_chunk_idx = min(state.num_decoded_chunks, num_chunks - 1)
            cur_event_idx = context.event_index_by_chunk[cleanest_chunk_idx]
            round_event_cond = context.event_conditions[cur_event_idx]

            is_selected_round = round_idx in context.selected_round_indices
            use_uniform_loss = (
                context.training_objective
                is RollingTrainingObjective.UNIFORM_TIMESTEP_DMD
            )
            should_grad = (
                mode is RollingMode.GENERATOR_DMD_TRAINING
                and is_selected_round
                and not use_uniform_loss
            )

            predicted_clean_window = self._run_window_round(
                round_idx=round_idx,
                rolling_stage=rolling_stage,
                steps_per_window=STEPS_PER_WINDOW,
                event_cond=round_event_cond,
                kv_frame_position=state.kv_frame_position,
                timesteps=context.timesteps,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                should_grad=should_grad,
                compute_predicted_clean_window=is_selected_round,
            )

            if is_selected_round and predicted_clean_window is not None:
                _n_win_chunks = self.rolling_queue.get_num_chunks_in_window()
                if use_uniform_loss:
                    if predicted_clean_window.requires_grad:
                        raise RuntimeError(
                            "uniform loss source must be detached"
                        )
                    uniform_step_index = context.uniform_step_index
                    if uniform_step_index is None:
                        raise RuntimeError(
                            "uniform loss is missing its scheduler step"
                        )
                    loss_prediction = self._run_uniform_timestep_regularizer(
                        x0_source=predicted_clean_window,
                        uniform_step_index=uniform_step_index,
                        event_cond=round_event_cond,
                        kv_frame_position=state.kv_frame_position,
                        timesteps=context.timesteps,
                        target_dtype=target_dtype,
                        autocast_enabled=autocast_enabled,
                        should_grad=(
                            mode is RollingMode.GENERATOR_DMD_TRAINING
                        ),
                        use_sp=use_sp,
                    )
                    step_idx_list = [uniform_step_index] * _n_win_chunks
                else:
                    loss_prediction = predicted_clean_window
                    step_idx_list = self.scheduler_wrapper.get_step_index(
                        round_idx * STEPS_PER_WINDOW
                    ).tolist()[:_n_win_chunks]
                if not step_idx_list:
                    raise RuntimeError(
                        f"round {round_idx}: step_idx_list is empty after truncation "
                        f"(num_chunks_in_window={_n_win_chunks}) -- the window holds no chunk yet produced a "
                        "predicted_clean_window, so rolling_queue and the denoising range have gone out of sync."
                    )
                denoising_range = build_denoising_range(
                    step_indices=step_idx_list,
                    timesteps=context.timesteps,
                )
                prediction = WindowPrediction(
                    round_index=round_idx,
                    output_start_frame=int(state.output_frame_count),
                    predicted_clean_latents=loss_prediction.detach(),
                    event_index=cur_event_idx,
                    denoising_range=denoising_range,
                    dmd_backward_completed=False,
                )
                if mode is RollingMode.GENERATOR_DMD_TRAINING:
                    dmd_backward_handler.on_window(
                        window_index=len(state.window_predictions),
                        expected_num_windows=len(context.selected_round_indices),
                        predicted_clean_window=loss_prediction,
                        output_start_frame=int(state.output_frame_count),
                        history_chunks=tuple(state.finalized_chunks),
                        previous_window_predictions=tuple(state.window_predictions),
                        prompt_condition=round_event_cond,
                        teacher_unconditional_condition=teacher_unconditional_condition,
                        denoising_range=denoising_range,
                    )
                    prediction.dmd_backward_completed = True
                    # Release the window's graph before the next window's forward: these two names are not
                    # rebound until the top of the next iteration, so without this the previous window's
                    # activations stay alive through it.
                    del loss_prediction, predicted_clean_window
                state.window_predictions.append(prediction)

        # ---- 6. Finalize typed result ----
        if not state.finalized_chunks:
            raise RuntimeError("rolling loop produced no decoded chunks")
        base_latents = torch.cat(state.finalized_chunks, dim=2)
        if mode is RollingMode.SAMPLING:
            if state.window_predictions:
                raise RuntimeError(
                    "sampling produced unexpected training window predictions"
                )
            generated_latents = base_latents
            window_prediction_mask = None
        else:
            generated_latents, window_prediction_mask = splice_window_predictions(
                base_latents, state.window_predictions
            )
        expected_frames = sum(output_chunk_sizes)
        if generated_latents.shape[2] != expected_frames:
            raise RuntimeError(
                "rolling output length mismatch: "
                f"expected {expected_frames}, got {generated_latents.shape[2]}"
            )
        if mode in (RollingMode.GENERATOR_DMD_TRAINING, RollingMode.FAKE_SCORE_TRAINING):
            expected = selection_plan.num_windows if selection_plan is not None else 0
            if len(state.window_predictions) != expected:
                raise RuntimeError(
                    f"selected-window count mismatch: expected {expected}, "
                    f"got {len(state.window_predictions)}"
                )
        result = RollingResult(
            generated_latents=generated_latents,
            window_predictions=tuple(state.window_predictions),
            window_prediction_mask=window_prediction_mask,
            event_conditions=context.event_conditions,
            chunk_sizes=output_chunk_sizes,
            dmd_backward_completed=(mode is RollingMode.GENERATOR_DMD_TRAINING),
            event_split_indices=output_event_split_indices,
            start_phase=selection_plan.start_phase if selection_plan is not None else None,
            selected_round_indices=tuple(
                prediction.round_index for prediction in state.window_predictions
            ),
            training_objective=context.training_objective,
            uniform_step_index=context.uniform_step_index,
        )
        result.validate_for_mode(mode)
        return result

    def _encode_clean_chunk_kv_with_context_noise(
        self,
        clean_latents: torch.Tensor,
        *,
        kv_frame_position: int,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        context_noise_fn: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        use_sp: bool = False,
    ) -> int:
        """Add the optional context noise, then encode KV at timestep zero."""

        with torch.no_grad():
            kv_latents = self._add_context_noise(
                clean_latents.detach(), context_noise_fn, use_sp
            ).detach()
            return self._encode_clean_chunk_kv(
                kv_latents,
                current_frame_index=kv_frame_position,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
            )

    # ------------------------------------------------------------------
    # Stateless streaming-window forward shared by RF and the loss regularizer.
    # ------------------------------------------------------------------
    def _forward_window(
        self,
        *,
        latents: torch.Tensor,
        noisy_t: torch.Tensor,
        event_cond: Tuple,
        kv_frame_position: int,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        should_grad: bool,
        compute_x0: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        prompt_embeds, prompt_mask, prompt_embeds_2, extra_kwargs = event_cond
        grad_context = torch.enable_grad() if should_grad else torch.no_grad()

        with grad_context:
            gt_latents_concat, mask_concat = self.prepare_latent_concat(latents)
            latent_model_input = self.prepare_latent_model_input(
                latents,
                gt_latents_concat,
                mask_concat,
                do_classifier_free_guidance=False,
            )
            batch_size = latent_model_input.shape[0]
            t_expand = noisy_t.unsqueeze(0).repeat(batch_size, 1)

            with torch.autocast(
                device_type=latents.device.type,
                dtype=target_dtype,
                enabled=autocast_enabled and latents.device.type == "cuda",
            ):
                noise_pred = self.transformer(
                    latent_model_input,
                    t_expand,
                    prompt_embeds,
                    prompt_mask,
                    return_dict=False,
                    extra_kwargs=extra_kwargs,
                    return_kv=False,
                    timestep_txt=t_expand[:, :1],
                    cond_chunk_sizes=[],
                    noisy_chunk_sizes=[latents.shape[2]],
                    current_frame_index=kv_frame_position,
                    training=False,
                )[0]

            if noise_pred.shape[2] != latents.shape[2]:
                raise RuntimeError(
                    f"noise_pred frames {noise_pred.shape[2]} != "
                    f"window {latents.shape[2]}"
                )
            predicted_x0 = (
                self.scheduler.velocity_to_x0(
                    model_output=noise_pred,
                    timestep=t_expand,
                    sample=latents,
                )
                if compute_x0
                else None
            )
        return noise_pred, predicted_x0

    # ------------------------------------------------------------------
    # One round = one full-window denoise (steps_per_window steps), gradients on demand, returns the window x0
    # ------------------------------------------------------------------
    def _run_window_round(
        self,
        *,
        round_idx: int,
        rolling_stage,
        steps_per_window: int,
        event_cond: Tuple,
        kv_frame_position: int,
        timesteps: torch.Tensor,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        should_grad: bool,
        compute_predicted_clean_window: bool,
    ) -> Optional[torch.Tensor]:
        """Advance one window in the inference attention regime."""
        predicted_clean_window: Optional[torch.Tensor] = None

        for step in range(steps_per_window):
            local_step = round_idx * steps_per_window + step
            latents = self.rolling_queue.get_current_latents()
            step_index = self.scheduler_wrapper.get_step_index(local_step)
            step_index_expanded = self.expand_step_index(step_index, rolling_stage)
            noisy_t = timesteps[step_index_expanded]
            is_last_step = step == steps_per_window - 1
            noise_pred, predicted_x0 = self._forward_window(
                latents=latents,
                noisy_t=noisy_t,
                event_cond=event_cond,
                kv_frame_position=kv_frame_position,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                should_grad=should_grad,
                compute_x0=compute_predicted_clean_window and is_last_step,
            )
            if predicted_x0 is not None:
                predicted_clean_window = predicted_x0

            # The loss-only prediction must never feed the rollout state.
            with torch.no_grad():
                latents = self.scheduler_wrapper.step(noise_pred, step_index_expanded, latents)
                self.rolling_queue.update_with_new_latents(latents.detach())

        return predicted_clean_window

    def _run_uniform_timestep_regularizer(
        self,
        *,
        x0_source: torch.Tensor,
        uniform_step_index: int,
        event_cond: Tuple,
        kv_frame_position: int,
        timesteps: torch.Tensor,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        should_grad: bool,
        use_sp: bool,
    ) -> torch.Tensor:
        """Produce a loss-only x0 without mutating queue, scheduler, or KV."""

        source = x0_source.detach()
        self._uniform_noise_counter += 1
        if use_sp:
            sp_group = nccl_info.sp_group
            if is_group_rank0(sp_group):
                group_id = int(dist.get_process_group_ranks(sp_group)[0])
                generator = torch.Generator(device="cpu").manual_seed(
                    42
                    + self._UNIFORM_NOISE_SEED_DOMAIN
                    + group_id * 1_000_003
                    + self._uniform_noise_counter * 10_007
                )
                noise = randn_tensor(
                    source.shape,
                    generator=generator,
                    device=source.device,
                    dtype=source.dtype,
                )
            else:
                noise = torch.empty_like(source)
            bcast_in_group(noise, sp_group)
        else:
            group_id = dist.get_rank() if dist.is_initialized() else 0
            generator = torch.Generator(device="cpu").manual_seed(
                42
                + self._UNIFORM_NOISE_SEED_DOMAIN
                + group_id * 1_000_003
                + self._uniform_noise_counter * 10_007
            )
            noise = randn_tensor(
                source.shape,
                generator=generator,
                device=source.device,
                dtype=source.dtype,
            )

        uniform_t = timesteps[uniform_step_index].to(
            device=source.device, dtype=torch.float32
        )
        sigma = uniform_t / float(self.scheduler.config.num_train_timesteps)
        with torch.no_grad():
            noisy_latents = (1.0 - sigma) * source + sigma * noise
        noisy_t = uniform_t.expand(source.shape[2])

        _, predicted_x0 = self._forward_window(
            latents=noisy_latents,
            noisy_t=noisy_t,
            event_cond=event_cond,
            kv_frame_position=kv_frame_position,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            should_grad=should_grad,
            compute_x0=True,
        )
        if predicted_x0 is None:
            raise RuntimeError("uniform-timestep forward did not produce x0")
        return predicted_x0
