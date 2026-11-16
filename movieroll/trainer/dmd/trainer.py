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

"""Orchestrator for Stage3 MovieRoll rolling DMD training."""

import gc
import os
import math
import logging
from typing import Dict
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from movieroll.optim.muon import Muon
from movieroll.trainer.ema import FSDPEMAModel
from movieroll.trainer.dmd.sampling import DMDSampler
from movieroll.trainer.dmd.setup import build_dmd_runtime
from movieroll.trainer.dmd.data import (
    DMDDataModule,
    DMDTrainingPhase,
    phase_spec_from_config,
)
from movieroll.trainer.dmd.checkpoint import (
    DMDCheckpointManager,
    DMDTrainingProgress,
)

logger = logging.getLogger(__name__)

class DMDTrainer:

    def __init__(self, config: DictConfig, rank: int, world_size: int, device: torch.device):
        """Build the Stage3 runtime (models, data, optimizers, EMA, checkpoints) on this rank."""
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.is_main = (rank == 0)

        self.current_phase = DMDTrainingPhase.SHORT_SINGLE_EVENT
        self.phase_step = 0
        self.global_step = 0
        # FSDP2's MixedPrecisionPolicy steps the optimizer on the SHARDED parameter in its ORIGINAL
        # dtype, so the model must still be fp32 when it is wrapped: forward/backward precision comes
        # from torch.autocast and FSDP's reduce_dtype stays fp32. bf16 resolves only 3.9e-3 relative
        # while Muon's update is dw/w ~ 1e-4, so stepping in bf16 would round `w + dw` back to `w` and
        # accumulate the error in the momentum buffer as well.
        self.master_weight_type = str(config.master_weight_type).lower()
        # Parameter/weight dtype (the precision the optimizer actually steps in)
        self.param_dtype = torch.float32 if self.master_weight_type == "fp32" else torch.bfloat16
        # Compute dtype (used by autocast): with mixed_precision=true forward/backward run in bf16
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.grad_accum_steps = int(config.gradient_accumulation_steps)

        self._setup_logdir()
        self._init_logging()

        runtime = build_dmd_runtime(
            config=self.config,
            world_size=self.world_size,
            device=self.device,
            parameter_dtype=self.param_dtype,
            compute_dtype=self.dtype,
            logger=logger,
        )
        self.parallel_context = runtime.parallel_context
        self.fsdp_dp_mesh = runtime.fsdp_dp_mesh
        self.dp_rank = runtime.parallel_context.dp_rank
        self.dp_degree = runtime.parallel_context.dp_degree
        self.vae = runtime.vae
        self.generator = runtime.generator
        self.critic = runtime.critic
        self.dmd_model = runtime.dmd_model
        self.data = DMDDataModule(
            config=self.config, device=self.device,
            parallel_context=self.parallel_context, logger=logger,
        )
        self._init_optimizers()
        self._init_ema()
        self._pending_runtime_state = None
        self.checkpoints = DMDCheckpointManager(
            config=self.config, rank=self.rank, world_size=self.world_size,
            generator=self.generator, critic=self.critic,
            optimizer_generator=self.optimizer_gen, optimizer_critic=self.optimizer_critic,
            fsdp_dp_mesh=self.fsdp_dp_mesh,
            get_ema=lambda: self.generator_ema,
            set_ema=lambda ema: setattr(self, "generator_ema", ema),
            get_runtime_state=self._checkpoint_runtime_state,
            set_runtime_state=self._receive_runtime_state,
        )
        if config.get("resume_checkpoint", None):
            self._restore_training_progress(
                self.checkpoints.load(config.resume_checkpoint)
            )
        # Order is load-bearing: `load` only parks the runtime state, `configure_phase` builds the loader
        # and sets `data.phase`, and only then can `_apply_pending_runtime_state` decide whether to replay
        # the data cursor. Swapped, it either loses the cursor or replays it into a loader that is not built.
        self.data.configure_phase(self.current_phase)
        self._apply_pending_runtime_state()

        logger.info(
            "DMDTrainer initialized on rank %d/%d",
            rank,
            world_size,
        )

    def _phase_max_steps(self, phase: DMDTrainingPhase) -> int:
        return phase_spec_from_config(self.config, phase).max_steps

    def _checkpoint_runtime_state(self) -> dict:
        # The pipeline's `_uniform_noise_counter` / `_uniform_decision_counter` are NOT saved: a resumed
        # run restarts those two streams from zero, so a resume consistency check must not compare them.
        state = {
            "dmd_noise_step_counter": int(
                self.dmd_model._noise_step_counter
            ),
            "data": self.data.state_dict(),
        }
        # Only written away from the default, so a released-recipe checkpoint keeps the metadata it has.
        if self.dmd_model.config.score_mode != "t2v":
            state["score_mode"] = self.dmd_model.config.score_mode
        return state

    def _receive_runtime_state(self, state: dict) -> None:
        self._pending_runtime_state = dict(state)

    def _apply_pending_runtime_state(self) -> None:
        state = getattr(self, "_pending_runtime_state", None)
        if state is None:
            return
        # The critic's weights were trained against one conditioning convention, vision_in included;
        # resuming under the other one would silently change the distribution it scores.
        saved_mode = str(state.get("score_mode", "t2v"))
        if saved_mode != self.dmd_model.config.score_mode:
            raise RuntimeError(
                f"checkpoint was trained with dmd.score_mode={saved_mode!r} but the config says "
                f"{self.dmd_model.config.score_mode!r}; resume with the config the run was started with"
            )
        self.dmd_model._noise_step_counter = int(
            state.get("dmd_noise_step_counter", 0)
        )
        data_state = state.get("data")
        if data_state is not None:
            self.data.load_state_dict(
                dict(data_state),
                restore_cursor=(self.data.phase is self.current_phase),
            )
        self._pending_runtime_state = None

    def _restore_training_progress(self, progress: DMDTrainingProgress) -> None:
        max_steps = self._phase_max_steps(progress.phase)
        if progress.phase_step > max_steps:
            raise RuntimeError(
                f"checkpoint phase_step={progress.phase_step} exceeds "
                f"{progress.phase.value}.max_steps={max_steps}"
            )
        if progress.phase_complete and progress.phase_step != max_steps:
            raise RuntimeError(
                "phase_complete checkpoint must be at the phase boundary: "
                f"{progress.phase.value} {progress.phase_step}/{max_steps}"
            )
        if (
            progress.phase is DMDTrainingPhase.SHORT_SINGLE_EVENT
            and progress.phase_complete
        ):
            self.current_phase = DMDTrainingPhase.LONG_DUAL_EVENT
            self.phase_step = 0
        else:
            self.current_phase = progress.phase
            self.phase_step = progress.phase_step
        self.global_step = progress.global_step
        logger.info(
            "Restored Stage3 progress: checkpoint_phase=%s phase_step=%d "
            "global_step=%d complete=%s -> active_phase=%s active_phase_step=%d",
            progress.phase.value,
            progress.phase_step,
            progress.global_step,
            progress.phase_complete,
            self.current_phase.value,
            self.phase_step,
        )

    def _setup_logdir(self):
        from datetime import datetime

        base_logdir = self.config.logdir
        exp_name = str(self.config.exp_name)

        resume_checkpoint = self.config.get("resume_checkpoint", None)
        if resume_checkpoint:
            # The layout is written by `save_multi_model_checkpoint` / `save_checkpoint_with_optim`, which
            # both join <output_dir>/checkpoints/step-<N>; this parser has to agree with them. So the experiment
            # directory to resume into is TWO levels up: tensorboard / logs / samples stay where the
            # interrupted run put them. The `checkpoints` name is matched explicitly
            # instead of taking `parents[1]` blindly, so a path from elsewhere is reported, not
            # reinterpreted.
            checkpoint_path = Path(resume_checkpoint)
            if checkpoint_path.exists():
                if checkpoint_path.parent.name == "checkpoints":
                    timestamped_logdir = str(checkpoint_path.parent.parent)
                else:
                    raise ValueError(
                        "resume_checkpoint must be a step directory directly below "
                        f"a 'checkpoints' directory, got {checkpoint_path}"
                    )
                if self.is_main:
                    logger.info(f"Resuming training - using existing experiment directory: {timestamped_logdir}")
            else:
                # A resume_checkpoint pointing nowhere MUST fail: creating a new directory and carrying
                # on would retrain from the initial weights at step 0 behind a single warning line.
                raise FileNotFoundError(
                    f"resume_checkpoint path does not exist: {resume_checkpoint}\n"
                    "Continuing would retrain from the initial weights at step 0, so this fails here.")
        else:
            if self.is_main:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                timestamped_logdir = os.path.join(base_logdir, f"{exp_name}_{timestamp}")
            else:
                timestamped_logdir = None

        if dist.is_initialized():
            timestamped_logdir_list = [timestamped_logdir]
            dist.broadcast_object_list(timestamped_logdir_list, src=0)
            timestamped_logdir = timestamped_logdir_list[0]

        self.config.logdir = timestamped_logdir
        if self.is_main:
            logger.info(f"Experiment directory: {timestamped_logdir}")

    def _build_muon(self, model, *, lr, adamw_betas):
        """Build one Muon optimizer: matrices (`p.ndim >= 2`) take the Muon update, the rest AdamW.

        The split does not filter on `requires_grad`, which is correct only because the generator and
        fake_score are fully unfrozen; freezing anything would put frozen matrices into Muon's group.
        """
        muon_params = [p for _, p in model.named_parameters() if p.ndim >= 2]
        adamw_params = [p for _, p in model.named_parameters() if p.ndim < 2]
        return Muon(
            lr=float(lr),
            wd=float(self.config.optimizer.weight_decay),
            muon_params=muon_params,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adamw_params=adamw_params,
            # Muon defaults adamw_betas to (0.95, 0.95); Stage3 wants AdamW's own defaults for the
            # ndim < 2 parameters, so they are passed explicitly. tuple() also turns the YAML's
            # ListConfig into a native type: the betas reach param_groups, which distributed
            # checkpointing refuses to hold config objects in.
            adamw_betas=tuple(adamw_betas),
            adamw_eps=float(self.config.optimizer.adamw_eps),
        )

    def _init_optimizers(self):
        logger.info("Initializing optimizers...")
        self.optimizer_gen = self._build_muon(
            self.dmd_model.generator,
            lr=self.config.optimizer.lr,
            adamw_betas=self.config.optimizer.adamw_betas,
        )
        self.optimizer_critic = self._build_muon(
            self.dmd_model.fake_score,
            lr=self.config.optimizer.lr_critic,
            adamw_betas=self.config.optimizer.adamw_betas_critic,
        )
        logger.info(
            "Using Muon optimizers: generator lr=%s, fake_score lr=%s",
            self.config.optimizer.lr,
            self.config.optimizer.lr_critic,
        )

    def _init_ema(self):
        """EMA for the generator; with resume_checkpoint the shadow comes from the checkpoint instead."""
        self.ema_weight = float(self.config.ema_weight)
        self.ema_start_step = int(self.config.ema_start_step)
        self.generator_ema = None

        # EMA is unconditional: the validator keeps ema_weight in the open interval (0, 1).
        logger.info(f"EMA configured with weight={self.ema_weight}, start_step={self.ema_start_step}")
        if self.config.get("resume_checkpoint", None):
            logger.info("EMA shadow will be restored from the checkpoint (resume mode)")
        elif self.global_step >= self.ema_start_step:
            logger.info(f"Creating EMA for generator (step {self.global_step} >= start_step {self.ema_start_step})")
            self.generator_ema = FSDPEMAModel(self.generator, decay=self.ema_weight)

    def _maybe_create_ema(self):
        if self.generator_ema is None and self.global_step >= self.ema_start_step:
            logger.info(f"Creating EMA for generator at step {self.global_step}")
            self.generator_ema = FSDPEMAModel(self.generator, decay=self.ema_weight)

    def _update_ema(self):
        if self.generator_ema is not None:
            self.generator_ema.update(self.generator)

    def _init_logging(self):
        """Create the log directory and attach the per-rank (and, on rank 0, the combined) file handlers."""
        if self.is_main:
            os.makedirs(self.config.logdir, exist_ok=True)

        if dist.is_initialized():
            dist.barrier()

        log_dir = os.path.join(self.config.logdir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"rank_{self.rank:03d}.log")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            f'[%(asctime)s][Rank {self.rank}][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        if self.is_main:
            combined_log_file = os.path.join(log_dir, "main.log")
            combined_handler = logging.FileHandler(combined_log_file, mode='a')
            combined_handler.setLevel(logging.INFO)
            combined_handler.setFormatter(file_formatter)
            logger.addHandler(combined_handler)
            logger.info(f"Experiment logdir: {self.config.logdir}")
            logger.info(f"Per-rank logs saved to: {log_dir}/rank_*.log")

        self.use_tensorboard = not self.config.get("disable_tensorboard", False)
        self.writer = None
        if self.is_main and self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise ImportError(
                    "tensorboard is not installed; install it or set `disable_tensorboard: true` in the config"
                ) from error

            tb_log_dir = os.path.join(self.config.logdir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)

            self.writer = SummaryWriter(
                log_dir=tb_log_dir,
                comment=str(self.config.exp_name),
            )
            logger.info(f"TensorBoard initialized: {tb_log_dir}")

    def _build_sampler(self) -> DMDSampler:
        return DMDSampler(
            config=self.config,
            pipeline=self.dmd_model.generator_pipeline,
            generator_model=self.generator,
            vae=self.vae,
            ema=self.generator_ema,
            dp_rank=self.dp_rank,
            dp_degree=self.dp_degree,
            device=self.device,
            logger_=logger,
        )

    def generate_samples(self):
        self._build_sampler().run(
            phase=self.current_phase,
            step=self.global_step,
            num_samples=int(self.config.sampling.num_samples),
            output_subdir="samples",
        )

    def generate_samples_val(self):
        cap = int(self.config.sampling.val_num_samples)
        self._build_sampler().run(
            phase=self.current_phase,
            step=self.global_step,
            num_samples=None if cap < 0 else cap,
            output_subdir="samples_val",
        )

    @staticmethod
    def _grad_norm_value(gn) -> float:
        """The value clip_grad_norm_ returned, reduced to the GLOBAL norm as a scalar.

        Under FSDP2/DTensor it is a DTensor placed as `_NormPartial(sum, 2)`, this rank's partial sum of
        squares before the all-reduce, so `.item()` alone reads one shard and under-reports;
        `.full_tensor()` performs the missing reduce. Clipping itself already uses the true global norm,
        so this affects only the logged value and the isfinite check below.
        """
        if hasattr(gn, "full_tensor"):
            return float(gn.full_tensor().item())
        return float(gn.item()) if torch.is_tensor(gn) else float(gn)

    def _set_fsdp_gradient_sync(self, model: torch.nn.Module, requires_sync: bool):
        """FSDP2 gradient-sync control: skip the all-reduce on intermediate accumulation micro-steps.

        The call must drill through ConcatModelWrapper, which forwards parameters/train/eval but not
        `set_requires_gradient_sync`. A silent miss would let every accumulation micro-step
        reduce-scatter into the sharded .grad, reducing the gradient accum_steps times over and
        multiplying the critic's effective learning rate, so a miss raises.
        """
        if self.world_size <= 1:
            return
        target = model
        if not hasattr(target, 'set_requires_gradient_sync'):
            inner = getattr(model, 'model', None)   # ConcatModelWrapper.model
            if inner is not None and hasattr(inner, 'set_requires_gradient_sync'):
                target = inner
        if not hasattr(target, 'set_requires_gradient_sync'):
            raise RuntimeError(
                f"_set_fsdp_gradient_sync: {type(model).__name__} has no "
                "set_requires_gradient_sync (and no inner .model that has it either). "
                "Skipping silently would let gradient accumulation reduce the gradients "
                "accum_steps times over, so this fails here.")
        target.set_requires_gradient_sync(requires_sync, recurse=True)

    def _prepare_update_batch(self):
        """Prepare the typed inputs shared by generator and fake-score updates."""

        spec = self.data.next_batch()
        if spec.phase is not self.current_phase:
            raise RuntimeError(
                "data/trainer phase mismatch: "
                f"data={spec.phase.value}, trainer={self.current_phase.value}"
            )
        self.dmd_model.multitask_type = spec.task_type
        return spec, {
            "prompts": spec.prompts,
            "num_frames": spec.num_frames,
            "height": spec.height,
            "width": spec.width,
            "event_prompts": spec.event_prompts,
            "event_timestamps": (
                [list(frame_range) for frame_range in spec.event_frame_ranges]
                if spec.event_frame_ranges is not None
                else None
            ),
        }

    def _any_rank(self, value: bool) -> bool:
        if not dist.is_initialized():
            return bool(value)
        flag = torch.tensor(
            [int(bool(value))],
            device=self.device,
            dtype=torch.long,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def train_generator_step(self) -> Dict[str, float]:
        """Run one optimizer update with immediate per-window DMD backward (CuLT-DMD).

        One `zero_grad`, then each window backpropagates as soon as it is predicted, and a single
        clip + step closes the rollout, so the whole rollout runs with one fixed set of weights.
        """

        self.optimizer_gen.zero_grad(set_to_none=True)

        accum_steps = self.grad_accum_steps
        total_loss = 0.0
        combined_log_dict = {}

        for micro_idx in range(accum_steps):
            is_last = (micro_idx == accum_steps - 1)

            # FSDP2: skip gradient sync on every micro-step but the last
            if accum_steps > 1:
                self._set_fsdp_gradient_sync(self.dmd_model.generator, requires_sync=is_last)

            spec, loss_kwargs = self._prepare_update_batch()

            if micro_idx == 0 and self.is_main and self.global_step % 100 == 0:
                logger.info(
                    f"[Step {self.global_step}] gen: {spec.num_frames} latent frames, "
                    f"{spec.height}x{spec.width}, task={spec.task_type}, "
                    f"events={len(spec.event_prompts)}"
                )

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.device.type == "cuda" and self.dtype != torch.float32,
            ):
                loss_value, log_dict = self.dmd_model.compute_dmd_loss_per_window(
                    backward_fn=lambda tensor: tensor.backward(),
                    loss_scale=1.0 / accum_steps,
                    **loss_kwargs,
                )
            total_loss += loss_value / accum_steps
            for key, value in log_dict.items():
                if isinstance(value, (int, float)):
                    combined_log_dict[key] = (
                        combined_log_dict.get(key, 0.0) + value / accum_steps
                    )
                else:
                    combined_log_dict[key] = value
            del log_dict

        grad_norm_gen = torch.nn.utils.clip_grad_norm_(
            self.dmd_model.generator.parameters(),
            float(self.config.max_grad_norm_generator),
            foreach=True,  # More efficient for FSDP
        )

        # The step is SKIPPED on a non-finite grad_norm: clip_grad_norm_ does not scrub NaN away, and a
        # NaN that reaches the master weights is burnt into the EMA shadow permanently by _update_ema.
        grad_norm = self._grad_norm_value(grad_norm_gen)
        local_nonfinite = not math.isfinite(grad_norm)
        skipped = self._any_rank(local_nonfinite)
        if skipped:
            logger.error(
                f"[Step {self.global_step}] generator grad_norm={grad_norm}; "
                f"non-finite gradient detected on {'this' if local_nonfinite else 'a peer'} "
                "rank, skipping optimizer/EMA update on every rank"
            )
            self.optimizer_gen.zero_grad(set_to_none=True)
        else:
            self.optimizer_gen.step()
            self._update_ema()

        combined_log_dict.update({
            "generator_loss": total_loss,
            "generator_grad_norm": grad_norm,
            "generator_step_skipped": float(skipped),
            "num_frames": spec.num_frames,
        })

        return combined_log_dict

    def train_critic_step(self) -> Dict[str, float]:
        """Run one fake-score optimizer update with immediate per-window critic backward."""

        self.optimizer_critic.zero_grad(set_to_none=True)

        accum_steps = self.grad_accum_steps
        total_loss = 0.0
        combined_log_dict = {}

        for micro_idx in range(accum_steps):
            is_last = (micro_idx == accum_steps - 1)

            if accum_steps > 1:
                self._set_fsdp_gradient_sync(self.dmd_model.fake_score, requires_sync=is_last)

            spec, loss_kwargs = self._prepare_update_batch()

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.device.type == "cuda" and self.dtype != torch.float32,
            ):
                loss_value, log_dict = self.dmd_model.compute_critic_loss_per_window(
                    backward_fn=lambda tensor: tensor.backward(),
                    loss_scale=1.0 / accum_steps,
                    **loss_kwargs,
                )
            total_loss += loss_value / accum_steps
            for key, value in log_dict.items():
                if isinstance(value, (int, float)):
                    combined_log_dict[key] = (
                        combined_log_dict.get(key, 0.0) + value / accum_steps
                    )
                else:
                    combined_log_dict[key] = value
            del log_dict

        grad_norm_critic = torch.nn.utils.clip_grad_norm_(
            self.dmd_model.fake_score.parameters(),
            float(self.config.max_grad_norm_critic),
            foreach=True,
        )

        grad_norm = self._grad_norm_value(grad_norm_critic)
        local_nonfinite = not math.isfinite(grad_norm)
        critic_skipped = self._any_rank(local_nonfinite)
        if critic_skipped:
            logger.error(
                f"[Step {self.global_step}] critic grad_norm={grad_norm}; "
                f"non-finite gradient detected on {'this' if local_nonfinite else 'a peer'} "
                "rank, skipping optimizer update on every rank"
            )
            self.optimizer_critic.zero_grad(set_to_none=True)
        else:
            self.optimizer_critic.step()

        combined_log_dict.update({
            "critic_loss": total_loss,
            "critic_grad_norm": grad_norm,
            "critic_step_skipped": float(critic_skipped),
            "num_frames": spec.num_frames,
        })

        return combined_log_dict

    def save_checkpoint(self, *, phase_complete: bool = False):
        return self.checkpoints.save(
            DMDTrainingProgress(
                phase=self.current_phase,
                phase_step=self.phase_step,
                global_step=self.global_step,
                phase_complete=phase_complete,
            )
        )

    def train(self):
        """Run the fixed short-single-event then long-dual-event schedule."""
        logger.info(
            "Starting Stage3 training at phase=%s phase_step=%d global_step=%d",
            self.current_phase.value,
            self.phase_step,
            self.global_step,
        )
        ratio = int(self.config.dfake_gen_update_ratio)
        sample_interval = int(self.config.sample_interval)
        val_interval = int(self.config.val_interval)
        try:
            if bool(self.config.get("sample_at_start", False)):
                self.generate_samples()
            phases = DMDTrainingPhase.ordered()
            for phase in phases[phases.index(self.current_phase):]:
                if phase is not self.current_phase:
                    self.current_phase = phase
                    self.phase_step = 0
                phase_spec = self.data.configure_phase(self.current_phase)
                expected_k = (
                    (phase_spec.latent_frames - 1)
                    // int(self.config.streaming_config.streaming_chunk_size)
                    // int(self.config.streaming_config.streaming_window_size)
                )
                logger.info(
                    "Entering phase=%s phase_step=%d/%d global_step=%d "
                    "frames=%d latent/%d pixel events=%d K=%d",
                    self.current_phase.value,
                    self.phase_step,
                    phase_spec.max_steps,
                    self.global_step,
                    phase_spec.latent_frames,
                    phase_spec.video_length,
                    phase_spec.event_count,
                    expected_k,
                )

                if self.phase_step == phase_spec.max_steps and (
                    self.current_phase is DMDTrainingPhase.LONG_DUAL_EVENT
                ):
                    logger.info(
                        "Final Stage3 phase is already complete at global_step=%d",
                        self.global_step,
                    )
                    break

                while self.phase_step < phase_spec.max_steps:
                    train_generator = self.global_step % ratio == 0
                    if train_generator:
                        generator_log = self.train_generator_step()
                        if self.is_main:
                            logger.info(
                                "Global %d phase=%s %d/%d [GEN] "
                                "loss=%.6f grad=%.6f dmd_grad=%.6f windows=%s "
                                "objective=%s task=%s",
                                self.global_step,
                                self.current_phase.value,
                                self.phase_step,
                                phase_spec.max_steps,
                                generator_log["generator_loss"],
                                generator_log["generator_grad_norm"],
                                generator_log.get("dmd_gradient_norm", 0.0),
                                generator_log.get("num_grad_windows_supervised"),
                                (
                                    f"uniform@{generator_log.get('uniform_step_index', -1)}"
                                    if generator_log.get(
                                        "rolling_objective_uniform", 0.0
                                    )
                                    else "rf"
                                ),
                                generator_log.get(
                                    "task_type",
                                    self.dmd_model.multitask_type,
                                ),
                            )
                        if self.writer is not None:
                            for key, tag in (
                                ("generator_loss", "generator/loss"),
                                ("generator_grad_norm", "generator/grad_norm"),
                                ("dmd_gradient_norm", "generator/dmd_grad_norm"),
                                ("num_grad_windows_supervised", "generator/num_windows"),
                                ("generator_step_skipped", "generator/step_skipped"),
                                ("dmd_windows_total_frames", "generator/supervised_frames"),
                                ("rolling_objective_uniform", "generator/objective_uniform"),
                                ("uniform_step_index", "generator/uniform_step_index"),
                                ("uniform_shifted_timestep", "generator/uniform_shifted_timestep"),
                                ("reencoded_window_count", "generator/reencoded_windows"),
                                ("identity_window_count", "generator/identity_windows"),
                                ("vae_local_decode_seconds", "generator/vae_local_decode_seconds"),
                                ("vae_boundary_encode_seconds", "generator/vae_boundary_encode_seconds"),
                                ("vae_local_decoded_latent_frames", "generator/vae_local_decoded_latent_frames"),
                                ("latent_context_cache_peak_bytes", "generator/latent_context_cache_peak_bytes"),
                            ):
                                if key in generator_log:
                                    self.writer.add_scalar(tag, generator_log[key], self.global_step)
                        del generator_log

                    critic_log = self.train_critic_step()
                    if self.is_main:
                        logger.info(
                            "Global %d phase=%s %d/%d [CRITIC] "
                            "loss=%.6f grad=%.6f windows=%s objective=%s",
                            self.global_step,
                            self.current_phase.value,
                            self.phase_step,
                            phase_spec.max_steps,
                            critic_log["critic_loss"],
                            critic_log["critic_grad_norm"],
                            critic_log.get("num_critic_windows_supervised"),
                            (
                                "uniform@%s"
                                % critic_log.get("critic_uniform_step_index", -1)
                                if critic_log.get("critic_objective_uniform", 0.0)
                                else "rf"
                            ),
                        )
                    if self.writer is not None:
                        for key, tag in (
                            ("critic_loss", "critic/loss"),
                            ("critic_grad_norm", "critic/grad_norm"),
                            ("num_critic_windows_supervised", "critic/num_windows"),
                            ("critic_step_skipped", "critic/step_skipped"),
                            ("critic_timestep_shifted", "critic/timestep_shifted"),
                            ("critic_objective_uniform", "critic/objective_uniform"),
                            ("critic_uniform_step_index", "critic/uniform_step_index"),
                            ("critic_uniform_shifted_timestep", "critic/uniform_shifted_timestep"),
                            ("reencoded_window_count", "critic/reencoded_windows"),
                            ("identity_window_count", "critic/identity_windows"),
                            ("vae_local_decode_seconds", "critic/vae_local_decode_seconds"),
                            ("vae_boundary_encode_seconds", "critic/vae_boundary_encode_seconds"),
                            ("vae_local_decoded_latent_frames", "critic/vae_local_decoded_latent_frames"),
                            ("latent_context_cache_peak_bytes", "critic/latent_context_cache_peak_bytes"),
                        ):
                            if key in critic_log:
                                self.writer.add_scalar(tag, critic_log[key], self.global_step)

                    self.phase_step += 1
                    self.global_step += 1
                    self._maybe_create_ema()
                    if (
                        self.phase_step < phase_spec.max_steps
                        and self.phase_step % int(self.config.save_interval) == 0
                    ):
                        self.save_checkpoint()
                    if sample_interval > 0 and self.phase_step % sample_interval == 0:
                        self.generate_samples()
                    if val_interval > 0 and self.phase_step % val_interval == 0:
                        self.generate_samples_val()
                    if self.global_step % int(self.config.gc_interval) == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                phase_checkpoint = self.save_checkpoint(phase_complete=True)
                verified = self.checkpoints.inspect_synchronized(
                    phase_checkpoint
                )
                if (
                    verified.phase is not self.current_phase
                    or not verified.phase_complete
                    or verified.phase_step != phase_spec.max_steps
                    or verified.global_step != self.global_step
                ):
                    raise RuntimeError(
                        "phase-complete checkpoint verification failed: "
                        f"saved={verified}, expected phase={self.current_phase.value} "
                        f"phase_step={phase_spec.max_steps} global_step={self.global_step}"
                    )
                if self.current_phase is DMDTrainingPhase.SHORT_SINGLE_EVENT:
                    # Phase 2 starts from the persisted Phase-1 artifact rather than the in-memory
                    # objects that wrote it, so model, optimizer, EMA, metadata and COMPLETE are all
                    # validated at the transition boundary.
                    restored = self.checkpoints.load(phase_checkpoint)
                    if restored != verified:
                        raise RuntimeError(
                            "Phase-1 checkpoint load disagrees with inspection: "
                            f"inspect={verified}, load={restored}"
                        )
                    # Order is load-bearing: _restore_training_progress promotes current_phase
                    # SHORT -> LONG while data.phase is still SHORT, so _apply_pending_runtime_state
                    # sees restore_cursor False and restores only the decision counter -- the LONG phase
                    # starts with a fresh data cursor. Swapped, it would replay the SHORT cursor.
                    self._restore_training_progress(restored)
                    self._apply_pending_runtime_state()
        finally:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
        logger.info("Training completed")
