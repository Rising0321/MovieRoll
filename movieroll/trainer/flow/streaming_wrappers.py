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

"""Training-time wrappers that turn one clip into a clean history plus a noisy rolling window.

These replace the flow transport's hook points so that one training step looks like one step of rolling
inference: `StreamingSplitLatentsWrapper` draws the task form and splits the clip into a clean prefix and a
noisy tail, `StreamingTimestepsWrapper` puts each chunk of a window on its own rung of a timestep ladder,
`StreamingMergeLatentsWrapper` assembles the model input, and the text / loss wrappers put the matching
per-window text in place and reduce the loss (the per-chunk losses are for logging only).

Two invariants: every decision that changes the SHAPE of a step (task form, split index, window sizes, which
chunks are dropped) is drawn on the SP group's first rank and broadcast, because disagreeing ranks desynchronise
the collectives; and the number of draws per step is identical on every rank, because each decision's seed comes
from a module-level counter that every rank increments in lockstep.
"""
import weakref
from typing import List, Optional, Tuple

import torch

from movieroll.commons.streaming_config import StreamingConfig
from movieroll.models.transformers.modules.context_noise import apply_context_noise
from movieroll.models.transformers.modules.chunking import (
    check_and_get_chunk_sizes,
    check_and_get_window_sizes,
    get_window_wrapped_chunk_sizes,
)
from movieroll.models.transformers.modules.streaming_utils import safe_get_rank
from movieroll.commons.parallel_states import get_sequence_parallel_state, nccl_info
from movieroll.trainer.flow.utils import mean_flat


# Random interval split, then a logit-normal percentile: one percentile per sample is shared by every chunk
# of the window, so the chunk timesteps come out monotone in the sorted pivots.
def sample_rolling_timesteps(
    batch_size: int,
    t0: float,
    t1: float,
    chunk_sizes: List[int],
    without_rolling: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    current_window_size = len(chunk_sizes)

    if current_window_size > 1:
        pivots = torch.sort(
            torch.rand(
                batch_size,
                current_window_size - 1,
                generator=generator,
            ) * (t1 - t0) + t0,
            dim=1,
        )[0]
        boundaries = torch.cat([
            torch.full((batch_size, 1), t0),
            pivots,
            torch.full((batch_size, 1), t1),
        ], dim=1)
    else:
        boundaries = torch.tensor([t0, t1]).unsqueeze(0).expand(batch_size, -1)
    percent = torch.sigmoid(
        torch.randn(batch_size, generator=generator)
    )

    if without_rolling:
        timesteps = percent * (boundaries[:, 1] - boundaries[:, 0]) + boundaries[:, 0]
        return timesteps.unsqueeze(-1).expand(-1, sum(chunk_sizes))

    timesteps = torch.zeros((batch_size, sum(chunk_sizes)))
    frame_offset = 0
    for chunk_idx in range(current_window_size):
        chunk_size = chunk_sizes[chunk_idx]
        boundary_lo = boundaries[:, chunk_idx]
        boundary_hi = boundaries[:, chunk_idx + 1]
        chunk_timesteps = percent * (boundary_hi - boundary_lo) + boundary_lo  # [B]
        timesteps[:, frame_offset:frame_offset + chunk_size] = chunk_timesteps.unsqueeze(-1).expand(-1, chunk_size)
        frame_offset += chunk_size
    return timesteps



def _is_sp_rank0():
    if not (
        torch.distributed.is_initialized()
        and get_sequence_parallel_state()
    ):
        return True
    return torch.distributed.get_rank(nccl_info.sp_group) == 0


# Seed arithmetic of the per-decision generators: the SP group agrees only because every rank derives the
# same seed from these values, and their VALUES must not change: they decide which draw each rank's decision stream produces.
_SP_SOURCE_RANDOM_COUNTER = 0
_SP_SOURCE_RANDOM_DOMAIN = 31_337_783
_SP_SOURCE_RANDOM_BASE_SEED = 42
_SP_SOURCE_RANDOM_GROUP_STRIDE = 1_000_003
_SP_SOURCE_RANDOM_COUNTER_STRIDE = 10_007

# Upper bound on the probability that a window shares one timestep across all of its chunks instead of sitting
# on a ladder. The per-window probability is min(1 / num_windows, this).
_MAX_NON_ROLLING_PROB = 0.5

# One-shot guard for the window-tiling hint below: training mixes clip durations, so a per-step message
# would flood the log.
_WINDOW_TILING_HINT_SHOWN = False


def _make_sp_source_generator(device):
    """Return a group-specific RNG only on the current SP group's rank 0."""

    global _SP_SOURCE_RANDOM_COUNTER
    _SP_SOURCE_RANDOM_COUNTER += 1
    has_sp = (
        torch.distributed.is_initialized()
        and get_sequence_parallel_state()
    )
    group = nccl_info.sp_group if has_sp else None
    if group is not None and not _is_sp_rank0():
        return None
    group_id = (
        int(torch.distributed.get_process_group_ranks(group)[0])
        if group is not None
        else (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
    )
    seed = (
        _SP_SOURCE_RANDOM_BASE_SEED
        + _SP_SOURCE_RANDOM_DOMAIN
        + group_id * _SP_SOURCE_RANDOM_GROUP_STRIDE
        + _SP_SOURCE_RANDOM_COUNTER * _SP_SOURCE_RANDOM_COUNTER_STRIDE
    )
    return torch.Generator(device=device).manual_seed(seed)


class StreamingWrapper:
    """Descriptor that replaces a method with the streaming implementation selected by `instance.streaming_config`.

    `streaming_func(streaming_config, bound_method)` builds the replacement; it is cached per instance and per
    `id(streaming_config)`, so a config must not be mutated or replaced after the first attribute access.
    """
    def __init__(self, func):
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        # instance -> {config id: function}; weak-keyed so the cache does not keep instances alive
        self._cache = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        if instance is None:
            return self

        streaming_config = getattr(instance, 'streaming_config', None)

        if instance not in self._cache:
            self._cache[instance] = {}

        instance_cache = self._cache[instance]

        config_key = id(streaming_config) if streaming_config is not None else None

        if config_key not in instance_cache:
            wrapped = self.streaming_func(streaming_config, self.func.__get__(instance, owner))
            instance_cache[config_key] = wrapped

        return instance_cache[config_key]

    def streaming_func(self, streaming_config: StreamingConfig, func):
        raise NotImplementedError


class StreamingTimestepsWrapper(StreamingWrapper):
    """
    Sample the per-frame timesteps of the noisy part by splitting [t0, t1] along the current window layout.
    With probability min(1 / num_windows, _MAX_NON_ROLLING_PROB) a window instead shares one timestep across
    all of its chunks, which is the non-rolling case the model also has to handle.
    """
    def streaming_func(self, streaming_config: Optional[StreamingConfig], func):
        timesteps_func = func
        if streaming_config is None:
            return timesteps_func

        def wrapped_timesteps_func(batch_size, t0, t1, chunk_sizes=None, window_sizes=None):
            assert chunk_sizes is not None and len(chunk_sizes) > 0, "chunk_sizes must be provided"
            assert window_sizes is not None and len(window_sizes) > 0, "window_sizes must be provided"
            is_nested = chunk_sizes and isinstance(chunk_sizes[0], list)
            assert is_nested, "chunk_sizes must be a nested per-window list"
            assert len(window_sizes) == len(chunk_sizes), "window_sizes length must be equal to chunk_sizes length"

            mask = None
            total_frames = sum(sum(sub_chunk_sizes) for sub_chunk_sizes in chunk_sizes)
            num_windows = len(window_sizes)
            non_rolling_prob = min(1 / num_windows, _MAX_NON_ROLLING_PROB)
            timesteps = torch.zeros((batch_size, total_frames))
            generator = _make_sp_source_generator("cpu")
            source_rank = _is_sp_rank0()
            current_index = 0
            for sub_chunk_sizes, sub_window_sizes in zip(chunk_sizes, window_sizes):
                assert sub_window_sizes == len(sub_chunk_sizes), "sub_window_sizes must be equal to len(sub_chunk_sizes)"
                sub_total = sum(sub_chunk_sizes)
                # With some probability the window is non-rolling: every chunk shares one timestep.
                # Only SP rank 0 draws the source value; `t` is broadcast explicitly further downstream.
                if source_rank:
                    if generator is None:
                        raise RuntimeError(
                            "SP source rank is missing its random generator"
                        )
                    without_rolling = bool(
                        torch.rand((), generator=generator).item()
                        < non_rolling_prob
                    )
                    sub_timesteps = sample_rolling_timesteps(
                        batch_size,
                        t0,
                        t1,
                        sub_chunk_sizes,
                        without_rolling=without_rolling,
                        generator=generator,
                    )
                else:
                    sub_timesteps = torch.zeros(batch_size, sub_total)
                timesteps[:, current_index:current_index + sub_total] = sub_timesteps
                current_index += sub_total
            return timesteps, mask

        return wrapped_timesteps_func

class StreamingMultitaskManager:
    """Draw the task form of a training step (t2v / i2v) and keep the SP group in agreement.

    Call order: sample_task_type, then check_and_get_chunk_sizes with
    mode='i2v' if task_type == 'i2v' else 'default', then get_split_index.
    """
    def __init__(self, streaming_config: StreamingConfig):
        self.streaming_config = streaming_config
        self.t2v_prob = streaming_config.t2v_prob

        if safe_get_rank() == 0:
            print(f"StreamingMultitaskManager initialized with t2v_prob: {self.t2v_prob} "
                  f"(i2v_prob: {1.0 - self.t2v_prob})")

    def sample_task_type(self, device="cuda") -> str:
        """Draw the task form and broadcast it, so every rank of the SP group lays the clip out the same way.

        Must be called before check_and_get_chunk_sizes: i2v means mode='i2v' with first_chunk=1 (a single
        reference frame), t2v means mode='default' with first_chunk=1+chunk_size.
        """
        # One Bernoulli draw: t2v with probability t2v_prob, i2v otherwise. Encoded as an int32 flag so the
        # SP group can agree on it with a single broadcast.
        is_i2v = torch.zeros(1, dtype=torch.int32, device=device)
        generator = _make_sp_source_generator(device)
        if _is_sp_rank0():
            if generator is None:
                raise RuntimeError("SP source rank is missing its random generator")
            # Write IN PLACE: rebinding would leave the source rank with a different dtype from the other
            # ranks, and the broadcast below would then mismatch the receive buffer.
            is_i2v[:] = int(
                float(torch.rand((), device=device, generator=generator).item()) >= self.t2v_prob
            )

        if get_sequence_parallel_state():
            sp_group = nccl_info.sp_group
            torch.distributed.broadcast(is_i2v, group_src=0, group=sp_group)
        return "i2v" if is_i2v.item() else "t2v"

    def get_split_index(self, task_type: str, num_chunks: int, device="cuda") -> int:
        """
        Compute the split index from the already-decided task form and the actual chunk count.

        Deterministic given the task form, so no draw happens here; the broadcast stays because the SP
        group must agree on the layout even when a rank derived it independently.
        """
        if num_chunks == 1:
            return 0

        ref_split_index = torch.tensor([0], dtype=torch.int32, device=device)
        if task_type == "t2v":
            pass
        # i2v lays the clip out as [1, chunk_size, ...]: the single reference frame is its own chunk, and it
        # is the clean prefix.
        elif task_type == "i2v":
            ref_split_index[:] = 1
        else:
            raise ValueError(f"unknown task form {task_type!r}: expected 't2v' or 'i2v'")

        if get_sequence_parallel_state():
            sp_group = nccl_info.sp_group
            torch.distributed.broadcast(ref_split_index, group_src=0, group=sp_group)
        return ref_split_index.item()


class StreamingSplitLatentsWrapper(StreamingWrapper):
    """
    Split the clip into a clean cond prefix and a noisy tail, and compute the chunk / window layout.
    """
    def streaming_func(self, streaming_config: Optional[StreamingConfig], func):
        split_latents_func = func
        if streaming_config is None:
            return split_latents_func

        multitask_manager = StreamingMultitaskManager(streaming_config)
        chunk_size = streaming_config.chunk_size
        window_size = streaming_config.window_size

        # window_size == 1 means chunk-by-chunk training
        if safe_get_rank() == 0:
            print(
                f"""
                StreamingSplitLatentsWrapper initialized with:
                chunk_size: {chunk_size}
                window_size: {window_size}
                """
            )

        def wrapped_split_latents(latents):
            num_frames = latents.shape[2]
            device = latents.device

            # Draw the task form first: it decides the chunk layout.
            task_type = multitask_manager.sample_task_type(device)
            mode = 'i2v' if task_type == 'i2v' else 'default'

            chunk_sizes = check_and_get_chunk_sizes(num_frames, chunk_size=chunk_size, mode=mode)
            num_chunks = len(chunk_sizes)

            split_index = multitask_manager.get_split_index(task_type, num_chunks, device)
            cond_chunk_sizes = chunk_sizes[:split_index]
            noisy_chunk_sizes = chunk_sizes[split_index:]
            noisy_start_index = sum(chunk_sizes[:split_index])
            num_noisy_chunks = len(noisy_chunk_sizes)

            window_generator = _make_sp_source_generator("cpu")
            if _is_sp_rank0():
                if window_generator is None:
                    raise RuntimeError(
                        "SP source rank is missing its random generator"
                    )
                # The grid offset of this step, drawn here rather than inside check_and_get_window_sizes
                # because the whole SP group has to lay the clip out the same way, which the broadcast below
                # is what makes true. Same convention as Stage-3's start_phase: window_phase p puts p + 1
                # chunks in the first window, so over its window_size values every chunk visits every rung
                # of the timestep ladder.
                window_phase = int(
                    torch.randint(0, window_size, (), generator=window_generator).item()
                )
                window_sizes = check_and_get_window_sizes(
                    num_noisy_chunks,
                    window_size=window_size,
                    window_phase=window_phase,
                )
            else:
                window_sizes = []

            _window_sizes = [window_sizes]
            if get_sequence_parallel_state():
                sp_group = nccl_info.sp_group
                torch.distributed.broadcast_object_list(_window_sizes, group_src=0, group=sp_group)
            window_sizes = _window_sizes[0]

            global _WINDOW_TILING_HINT_SHOWN
            if not _WINDOW_TILING_HINT_SHOWN and safe_get_rank() == 0:
                _WINDOW_TILING_HINT_SHOWN = True
                print(
                    f"[window-tiling] first window = window_phase + 1 chunks, every later window = "
                    f"{window_size}; a trailing remainder that cannot fill a whole window is dropped. With "
                    f"window_phase drawn uniformly from 0..{window_size - 1}, a step therefore drops between "
                    f"0 and {window_size - 1} chunks off the end of the clip, so the last {window_size - 1} "
                    f"chunks are supervised less often than the rest. Choosing a clip whose chunk count is a "
                    f"multiple of {window_size} removes the drop for the full-window phase only; it is not "
                    f"required, and mixed-duration training will not satisfy it on every step. This step: "
                    f"{num_noisy_chunks} noisy chunks -> windows {window_sizes}. Shown once per run."
                )

            num_noisy_chunks = sum(window_sizes)
            noisy_chunk_sizes = noisy_chunk_sizes[:num_noisy_chunks]

            cond_latents_num_frames = sum(cond_chunk_sizes)
            noisy_latents_num_frames = sum(noisy_chunk_sizes)
            # The overlap frames are the chunks of the first window_sizes[:-1] windows; attention uses them
            # as the causal prefix.
            overlap_chunk_sizes = noisy_chunk_sizes[:sum(window_sizes[:-1])]
            overlap_latents_num_frames = sum(overlap_chunk_sizes)

            noisy_chunk_sizes = get_window_wrapped_chunk_sizes(noisy_chunk_sizes, window_sizes)

            if cond_latents_num_frames > 0:
                cond_latents = latents[:, :, :cond_latents_num_frames + overlap_latents_num_frames].clone()
                noisy_latents = latents[:, :, cond_latents_num_frames:cond_latents_num_frames + noisy_latents_num_frames].clone()
            elif overlap_latents_num_frames > 0:
                cond_latents = latents[:, :, :overlap_latents_num_frames].clone()
                noisy_latents = latents[:, :, :noisy_latents_num_frames].clone()
            else:
                cond_latents = None
                noisy_latents = latents[:, :, :noisy_latents_num_frames].clone()

            # the cond chunk sizes include the overlap frames
            cond_latents_num_frames = cond_latents.shape[2] if cond_latents is not None else 0
            cond_chunk_sizes = cond_chunk_sizes + overlap_chunk_sizes
            noisy_latents_num_frames = noisy_latents.shape[2]

            assert cond_latents_num_frames == sum(cond_chunk_sizes), f"cond_latents_num_frames {cond_latents_num_frames} != sum(cond_chunk_sizes) {sum(cond_chunk_sizes)}"

            return cond_latents, noisy_latents, \
                cond_chunk_sizes, noisy_chunk_sizes, window_sizes, \
                cond_latents_num_frames, noisy_latents_num_frames, noisy_start_index

        return wrapped_split_latents


class StreamingMergeLatentsWrapper(StreamingWrapper):
    def streaming_func(self, streaming_config: Optional[StreamingConfig], func):
        merge_latents_func = func
        if streaming_config is None:
            return merge_latents_func

        _context_noise = None
        if streaming_config.context_noise_enabled:
            sigma_max = streaming_config.context_noise_sigma_max
            sigma_min = streaming_config.context_noise_sigma_min
            def _context_noise(latents, chunk_sizes, generator=None):
                return apply_context_noise(
                    latents,
                    sigma_max=sigma_max,
                    sigma_min=sigma_min,
                    chunk_sizes=chunk_sizes,
                    generator=generator,
                )
            if safe_get_rank() == 0:
                print(f"""[context-noise] apply_context_noise initialized:
                sigma_max: {sigma_max}, sigma_min: {sigma_min}""")

        def wrapped_merge_latents(cond_latents, xt, cond_chunk_sizes):
            """
            Assemble the model input from the clean history and the noisy window, adding context noise to
            the history first with probability context_noise_prob.

            cond_latents [B, C, T_cond, H, W] or None, xt [B, C, T_noisy, H, W] ->
            merged [B, C, T_cond+T_noisy, H, W], t_cond [B, T_cond] (cond frames are always 0).
            """
            _device = xt.device
            has_cond = cond_latents is not None and len(cond_chunk_sizes) > 0
            _aug_enabled = _context_noise is not None
            p_aug = streaming_config.context_noise_prob if _aug_enabled else 0.0

            if has_cond and p_aug > 0:
                generator = _make_sp_source_generator(_device)
                if _is_sp_rank0():
                    if generator is None:
                        raise RuntimeError(
                            "SP source rank is missing its random generator"
                        )
                    do_aug = torch.tensor(
                        [
                            int(
                                torch.rand(
                                    (),
                                    device=_device,
                                    generator=generator,
                                ).item()
                                < p_aug
                            )
                        ],
                        dtype=torch.int32, device=_device,
                    )
                else:
                    do_aug = torch.empty(
                        1, dtype=torch.int32, device=_device
                    )
                if torch.distributed.is_initialized() and get_sequence_parallel_state():
                    torch.distributed.broadcast(do_aug, group_src=0, group=nccl_info.sp_group)

                if do_aug.item() == 1:
                    if torch.distributed.is_initialized() and get_sequence_parallel_state():
                        if _is_sp_rank0():
                            cond_latents = _context_noise(
                                cond_latents,
                                chunk_sizes=cond_chunk_sizes,
                                generator=generator,
                            ).contiguous()
                        else:
                            cond_latents = torch.empty_like(cond_latents)
                        torch.distributed.broadcast(
                            cond_latents, group_src=0, group=nccl_info.sp_group,
                        )
                    else:
                        cond_latents = _context_noise(
                            cond_latents,
                            chunk_sizes=cond_chunk_sizes,
                            generator=generator,
                        ).contiguous()

            t_cond = None
            if cond_latents is not None:
                t_cond = torch.zeros(
                    (cond_latents.shape[0], cond_latents.shape[2]),
                    device=cond_latents.device, dtype=cond_latents.dtype,
                )
                merged = torch.cat([cond_latents, xt], dim=2)
            else:
                merged = xt

            return merged, t_cond

        return wrapped_merge_latents


class StreamingLossWrapper(StreamingWrapper):
    def streaming_func(self, streaming_config: Optional[StreamingConfig], func):
        loss_func = func
        if streaming_config is None:
            return loss_func

        def _flatten_chunk_sizes(noisy_chunk_sizes):
            if noisy_chunk_sizes and isinstance(noisy_chunk_sizes[0], list):
                flat = []
                for sub in noisy_chunk_sizes:
                    flat.extend(sub)
                return flat
            return list(noisy_chunk_sizes)

        def compute_per_chunk_losses(model_output, ut, noisy_chunk_sizes):
            """Per-window, per-chunk loss as {name: float}, for logging only."""
            is_nested = noisy_chunk_sizes and isinstance(noisy_chunk_sizes[0], list)
            windows = noisy_chunk_sizes if is_nested else [noisy_chunk_sizes]

            flat_chunk_sizes = _flatten_chunk_sizes(noisy_chunk_sizes)
            all_chunk_losses = []
            offset = 0
            for cs in flat_chunk_sizes:
                chunk_out = model_output[:, :, offset:offset + cs]
                chunk_ut = ut[:, :, offset:offset + cs]
                chunk_loss = mean_flat((chunk_out.float() - chunk_ut.float()) ** 2)
                all_chunk_losses.append(chunk_loss)
                offset += cs
            all_chunk_losses = torch.stack(all_chunk_losses, dim=1)

            raw_chunk_losses = all_chunk_losses.detach().mean(dim=0).tolist()
            chunk_losses_dict = {}
            if is_nested:
                flat_idx = 0
                for window_idx, chunk_sizes in enumerate(windows):
                    for chunk_idx in range(len(chunk_sizes)):
                        chunk_losses_dict[f"window_{window_idx}_chunk_{chunk_idx}"] = raw_chunk_losses[flat_idx]
                        flat_idx += 1
            else:
                for chunk_idx in range(len(noisy_chunk_sizes)):
                    chunk_losses_dict[f"chunk_{chunk_idx}"] = raw_chunk_losses[chunk_idx]

            return chunk_losses_dict

        def wrapped_compute_loss(model_output, ut, noisy_chunk_sizes=None):
            loss_dict = {}
            loss = loss_func(model_output, ut)['loss']

            if noisy_chunk_sizes is not None:
                per_chunk_losses = compute_per_chunk_losses(model_output, ut, noisy_chunk_sizes)
                loss_dict['per_chunk_losses'] = per_chunk_losses

            loss_dict['loss'] = loss
            return loss_dict

        return wrapped_compute_loss


class StreamingTimestepTxtWrapper(StreamingWrapper):
    """
    timestep_txt: the timestep of each window's FIRST frame -- the one closest to the history, i.e. the least
    noisy -- used as that window's representative time for text modulation.
    """
    def streaming_func(self, streaming_config: Optional[StreamingConfig], func):
        base_func = func
        if streaming_config is None:
            return base_func

        def wrapped_compute_timestep_txt(t, cond_latents_num_frames, noisy_chunk_sizes):
            """
            t [B, cond_frames + noisy_frames] -> [B, num_windows], one text timestep per window.
            noisy_chunk_sizes is nested (the chunks of each window) or flat (frames per window).
            """
            is_nested = noisy_chunk_sizes and isinstance(noisy_chunk_sizes[0], list)
            if is_nested:
                window_frame_counts = [sum(sub) for sub in noisy_chunk_sizes]
            else:
                window_frame_counts = list(noisy_chunk_sizes)

            timestep_txt_list = []
            offset = cond_latents_num_frames
            for wf in window_frame_counts:
                timestep_txt_list.append(t[:, offset:offset + 1])
                offset += wf
            return torch.cat(timestep_txt_list, dim=1)

        return wrapped_compute_timestep_txt


def _find_event_for_window(window_start, window_end, event_stamp, num_caps):
    """Return 0-based event index whose latent range best overlaps the window.

    Falls back to nearest-center event when no overlap exists.
    Returns -1 only when *num_caps* == 0.
    """
    best, best_ov = -1, 0
    for i in range(num_caps):
        s, e = event_stamp[i, 0].item(), event_stamp[i, 1].item()
        if s < 0:
            continue
        ov = max(0, min(window_end, e) - max(window_start, s))
        if ov > best_ov:
            best_ov, best = ov, i
    if best < 0 and num_caps > 0:
        mid = (window_start + window_end) / 2.0
        best_d = float('inf')
        for i in range(num_caps):
            s, e = event_stamp[i, 0].item(), event_stamp[i, 1].item()
            if s < 0:
                continue
            d = abs((s + e) / 2.0 - mid)
            if d < best_d:
                best_d, best = d, i
    return best


def _build_per_window_text_4d(
    streaming_text_states: torch.Tensor,
    streaming_text_mask_enc: torch.Tensor,
    num_windows: int,
    cap_types: list,
    event_stamp: torch.Tensor,
    num_captions: list,
    noisy_start_index: int = 0,
    noisy_chunk_sizes=None,
    null_text_states: torch.Tensor = None,
    null_text_mask_enc: torch.Tensor = None,
    per_window_null_prob: float = 0.05,
    per_sample_null_prob: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick each window's caption and build the per-window text tensors.

    An "event" caption makes every window take the event overlapping its frame range, "structure" repeats
    slot 0. A sample-level null draw nulls all windows of a sample, otherwise windows are dropped singly.

    ``streaming_text_states`` [B, NC, S, D] and ``event_stamp`` [B, NC, 2] (latent frame ranges) ->
    ``text_4d`` [B, num_windows, S, D], ``mask_3d`` [B, num_windows, S], ``null_drop_mask``
    [B, num_windows] bool (True = null-dropped) or None when null dropout is disabled.
    """
    B = streaming_text_states.shape[0]

    window_offsets = []
    if noisy_chunk_sizes is not None:
        is_nested = noisy_chunk_sizes and isinstance(noisy_chunk_sizes[0], list)
        off = noisy_start_index
        if is_nested:
            for wc in noisy_chunk_sizes:
                window_offsets.append((off, off + sum(wc)))
                off += sum(wc)
        else:
            for cs in noisy_chunk_sizes:
                window_offsets.append((off, off + cs))
                off += cs

    per_b_text, per_b_mask = [], []
    for b in range(B):
        ct = cap_types[b] if cap_types and b < len(cap_types) else "structure"
        nc = num_captions[b] if num_captions and b < len(num_captions) else 1

        if ct == "event" and nc > 1:
            rows_t, rows_m = [], []
            for i in range(num_windows):
                ei = 0
                if i < len(window_offsets):
                    ei = _find_event_for_window(
                        window_offsets[i][0], window_offsets[i][1],
                        event_stamp[b], nc,
                    )
                    if ei < 0:
                        ei = 0
                rows_t.append(streaming_text_states[b, ei])
                rows_m.append(streaming_text_mask_enc[b, ei])
            per_b_text.append(rows_t)
            per_b_mask.append(rows_m)
        else:
            # structure / single-event -> repeat slot 0
            per_b_text.append(
                [streaming_text_states[b, 0] for _ in range(num_windows)]
            )
            per_b_mask.append(
                [streaming_text_mask_enc[b, 0] for _ in range(num_windows)]
            )

    null_drop_mask = None
    if (per_sample_null_prob > 0 or per_window_null_prob > 0) and null_text_states is not None:
        device = streaming_text_states.device
        conditional_window_null_prob = (
            per_window_null_prob / (1.0 - per_sample_null_prob)
            if per_sample_null_prob < 1.0 else 0.0
        )
        generator = _make_sp_source_generator(device)
        if _is_sp_rank0():
            if generator is None:
                raise RuntimeError("SP source rank is missing its random generator")
            sample_drop_mask = (
                torch.rand(
                    B, 1, device=device, generator=generator
                ) < per_sample_null_prob
                if per_sample_null_prob > 0 else
                torch.zeros(B, 1, dtype=torch.bool, device=device)
            )
            window_drop_mask = (
                torch.rand(
                    B,
                    num_windows,
                    device=device,
                    generator=generator,
                )
                < conditional_window_null_prob
                if conditional_window_null_prob > 0 else
                torch.zeros(
                    B, num_windows, dtype=torch.bool, device=device
                )
            )
            null_drop_mask = sample_drop_mask | (
                ~sample_drop_mask & window_drop_mask
            )
        else:
            null_drop_mask = torch.empty(
                B, num_windows, dtype=torch.bool, device=device
            )
        if get_sequence_parallel_state():
            torch.distributed.broadcast(null_drop_mask, group_src=0, group=nccl_info.sp_group)

        for b in range(B):
            for i in range(num_windows):
                if null_drop_mask[b, i]:
                    per_b_text[b][i] = null_text_states[b]
                    per_b_mask[b][i] = null_text_mask_enc[b]

    text_4d = torch.stack([torch.stack(rows) for rows in per_b_text]).contiguous()
    mask_3d = torch.stack([torch.stack(rows) for rows in per_b_mask]).contiguous()
    return text_4d, mask_3d, null_drop_mask


class StreamingTextSelectionWrapper(StreamingWrapper):
    """Give every noisy window the text of the event it falls in.

    Turns the per-event ``streaming_text_states [B, ME, S, D]`` into ``[B, num_windows, S, D]`` using
    ``cap_types`` and ``event_stamp``, and applies the two-stage null dropout.
    """
    def streaming_func(self, streaming_config: Optional[StreamingConfig], func):
        if streaming_config is None:
            return func

        def wrapped_build_chunk_text(model_kwargs, cond_chunk_sizes, noisy_chunk_sizes, noisy_start_index):
            num_noisy = len(noisy_chunk_sizes)
            if num_noisy == 0:
                return

            extra = model_kwargs.get("extra_kwargs", {})

            text_4d, mask_3d, null_drop_mask = _build_per_window_text_4d(
                extra["streaming_text_states"],
                extra["streaming_text_mask"],
                num_noisy,
                extra.get("cap_types", []),
                extra["event_stamp"],
                extra.get("num_captions", []),
                noisy_start_index=noisy_start_index,
                noisy_chunk_sizes=noisy_chunk_sizes,
                null_text_states=extra.get("null_text_states"),
                null_text_mask_enc=extra.get("null_text_mask_enc"),
                per_window_null_prob=streaming_config.per_window_null_prob,
                per_sample_null_prob=streaming_config.per_sample_null_prob,
            )

            # Glyph embeddings describe the whole clip, so they can only follow a structure caption: with an
            # event caption there is no way to say which event the quoted text belongs to, and a
            # null-dropped window must not keep its glyph either.
            cap_types = extra.get("cap_types", [])
            byt5_st = extra["byt5_text_states"]
            byt5_mk = extra["byt5_text_mask"]
            B = byt5_st.shape[0]
            rows_st, rows_mk = [], []
            for b in range(B):
                ct = cap_types[b] if b < len(cap_types) else "structure"
                if ct == "structure":
                    row_st = byt5_st[b:b+1].expand(num_noisy, -1, -1).clone()
                    row_mk = byt5_mk[b:b+1].expand(num_noisy, -1).clone()
                    if null_drop_mask is not None:
                        for i in range(num_noisy):
                            if null_drop_mask[b, i]:
                                row_st[i] = 0
                                row_mk[i] = 0
                    rows_st.append(row_st)
                    rows_mk.append(row_mk)
                else:
                    rows_st.append(torch.zeros(num_noisy, *byt5_st.shape[1:],
                                               dtype=byt5_st.dtype, device=byt5_st.device))
                    rows_mk.append(torch.zeros(num_noisy, byt5_mk.shape[1],
                                               dtype=byt5_mk.dtype, device=byt5_mk.device))
            extra["byt5_text_states"] = torch.stack(rows_st)
            extra["byt5_text_mask"] = torch.stack(rows_mk)

            model_kwargs["text_states"] = text_4d
            model_kwargs["encoder_attention_mask"] = mask_3d

        return wrapped_build_chunk_text
