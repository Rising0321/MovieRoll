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

"""Checkpoint lifecycle for the two-phase Stage3 DMD run.

``COMPLETE`` is the final write of a successful checkpoint, so a directory
without it is incomplete. ``checkpoint_metadata.json`` also names the optimizer
whose state the ``{generator,critic}_optimizer`` payloads hold
(``optimizer_impl``); only a checkpoint written by this trainer's optimizer is
resumed. The ``{name}_model`` / ``{name}_optimizer`` directory layout and the
``step`` metadata key are shared with ``movieroll.utils.checkpoint``'s
multi-model save/load helpers and must stay compatible with them.
"""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

from movieroll.trainer.dmd.data import DMDTrainingPhase
from movieroll.trainer.dmd.data import phase_spec_from_config
from movieroll.utils.torch_utils import count_failed_ranks

logger = logging.getLogger(__name__)

# The optimizer this trainer builds, as recorded in checkpoint_metadata.json.
OPTIMIZER_IMPL = "muon"


@dataclass(frozen=True)
class DMDTrainingProgress:
    phase: DMDTrainingPhase
    phase_step: int
    global_step: int
    phase_complete: bool = False

    def __post_init__(self) -> None:
        if self.phase_step < 0 or self.global_step < 0:
            raise ValueError("phase_step and global_step must be non-negative")


class DMDCheckpointManager:
    def __init__(
        self,
        *,
        config,
        rank: int,
        world_size: int,
        generator,
        critic,
        optimizer_generator,
        optimizer_critic,
        fsdp_dp_mesh=None,
        get_ema: Callable[[], object | None],
        set_ema: Callable[[object], None],
        get_runtime_state: Callable[[], dict] | None = None,
        set_runtime_state: Callable[[dict], None] | None = None,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.generator = generator
        self.critic = critic
        self.optimizer_generator = optimizer_generator
        self.optimizer_critic = optimizer_critic
        self.fsdp_dp_mesh = fsdp_dp_mesh
        self.get_ema = get_ema
        self.set_ema = set_ema
        self.get_runtime_state = get_runtime_state
        self.set_runtime_state = set_runtime_state

    def checkpoint_dir(self, global_step: int) -> Path:
        return (
            Path(self.config.logdir)
            / "checkpoints"
            / f"step-{int(global_step)}"
        )

    def _mesh_shape(self) -> list[int] | None:
        mesh = getattr(self.fsdp_dp_mesh, "mesh", None)
        shape = getattr(mesh, "shape", None)
        return list(shape) if shape is not None else None

    def _run_synchronized_stage(self, action: Callable[[], object], *, context: str):
        """Run one checkpoint stage and make every rank fail together."""

        # Without a process group the stage runs bare: its own exception propagates unwrapped.
        if not dist.is_initialized():
            return action()

        result = None
        local_error = None
        try:
            result = action()
        except (KeyboardInterrupt, SystemExit):
            # A deliberate interruption propagates at once instead of entering the all-reduce below: a
            # Ctrl-C or an exit that reaches only some ranks would otherwise block them here for the
            # whole NCCL timeout before the job could come down.
            raise
        except BaseException as exc:
            # BaseException rather than Exception: `CheckpointException`, the class the DCP load path is most
            # likely to raise, does NOT derive from Exception, and under `except Exception` the failing rank
            # unwinds past the all-reduce below and leaves every peer blocked in it until the NCCL timeout.
            # Nothing here inspects the type, so the wide catch only changes how many ranks learn about it.
            local_error = exc

        sync_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        failure_count = count_failed_ranks(local_error is not None, sync_device)
        if failure_count:
            if local_error is not None:
                raise RuntimeError(
                    f"{context} failed on {failure_count} rank(s)"
                ) from local_error
            raise RuntimeError(
                f"{context} failed on {failure_count} peer rank(s)"
            )
        return result

    def _validate_progress(self, progress: DMDTrainingProgress) -> None:
        phase_max = phase_spec_from_config(self.config, progress.phase).max_steps
        if progress.phase_step > phase_max:
            raise RuntimeError(
                f"checkpoint phase_step={progress.phase_step} exceeds "
                f"{progress.phase.value}.max_steps={phase_max}"
            )
        if progress.phase_complete and progress.phase_step != phase_max:
            raise RuntimeError(
                "phase_complete checkpoint is not at its phase boundary: "
                f"{progress.phase.value} {progress.phase_step}/{phase_max}"
            )
        phase1_steps = phase_spec_from_config(
            self.config, DMDTrainingPhase.SHORT_SINGLE_EVENT
        ).max_steps
        expected_global_step = progress.phase_step
        if progress.phase is DMDTrainingPhase.LONG_DUAL_EVENT:
            expected_global_step += phase1_steps
        if progress.global_step != expected_global_step:
            raise RuntimeError(
                "checkpoint phase/global step mismatch: "
                f"phase={progress.phase.value}, phase_step={progress.phase_step}, "
                f"global_step={progress.global_step}, expected={expected_global_step}"
            )

    def inspect(self, checkpoint_path: str | Path) -> DMDTrainingProgress:
        path = Path(checkpoint_path)
        if not path.is_dir():
            raise NotADirectoryError(f"resume checkpoint must be a directory: {path}")
        if not (path / "COMPLETE").exists() and not bool(
            self.config.get("allow_incomplete_checkpoint", False)
        ):
            raise FileNotFoundError(f"resume checkpoint is incomplete (missing COMPLETE): {path}")

        metadata_path = path / "checkpoint_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"resume checkpoint is missing metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        # Named explicitly, so a metadata file without `phase_name` does not raise a bare KeyError below.
        if "phase_name" not in metadata:
            raise RuntimeError(
                f"{metadata_path} predates the two-phase schema (no `phase_name`) "
                "and cannot be resumed."
            )

        saved_world_size = metadata.get("world_size")
        if saved_world_size is not None and int(saved_world_size) != self.world_size:
            raise RuntimeError(
                "checkpoint world size mismatch: "
                f"saved={saved_world_size}, current={self.world_size}"
            )
        saved_mesh_shape = metadata.get("fsdp_mesh_shape")
        current_mesh_shape = self._mesh_shape()
        if (
            saved_mesh_shape is not None
            and current_mesh_shape is not None
            and list(saved_mesh_shape) != current_mesh_shape
        ):
            raise RuntimeError(
                "checkpoint FSDP mesh mismatch: "
                f"saved={saved_mesh_shape}, current={current_mesh_shape}"
            )

        progress = DMDTrainingProgress(
            phase=DMDTrainingPhase(metadata["phase_name"]),
            phase_step=int(metadata["phase_step"]),
            global_step=int(metadata.get("global_step", metadata["step"])),
            phase_complete=bool(metadata.get("phase_complete", False)),
        )
        if int(metadata["step"]) != progress.global_step:
            raise RuntimeError(
                "checkpoint step/global_step mismatch: "
                f"step={metadata['step']}, global_step={progress.global_step}"
            )
        self._validate_progress(progress)
        return progress

    def inspect_synchronized(
        self, checkpoint_path: str | Path
    ) -> DMDTrainingProgress:
        return self._run_synchronized_stage(
            lambda: self.inspect(checkpoint_path),
            context="Stage3 checkpoint inspection",
        )

    def _validate_load_payload(
        self, path: Path, progress: DMDTrainingProgress
    ) -> None:
        """Reject incomplete model/optimizer/EMA payloads before mutating live state."""

        for model_name in ("generator", "critic"):
            metadata_path = path / f"{model_name}_metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("model_name") not in (None, model_name):
                    raise RuntimeError(
                        f"{metadata_path} describes model {metadata.get('model_name')!r}, "
                        f"expected {model_name!r}"
                    )
                if int(metadata.get("step", progress.global_step)) != progress.global_step:
                    raise RuntimeError(
                        f"{metadata_path} step disagrees with checkpoint progress"
                    )

            # Every payload is a sharded (DCP) directory, so the layout is checked unconditionally.
            required = (
                path / f"{model_name}_model",
                path / f"{model_name}_optimizer",
            )
            missing = [str(item) for item in required if not item.exists()]
            if missing:
                raise FileNotFoundError(
                    f"resume checkpoint is missing {model_name} payload: {missing}"
                )
            empty = [
                str(item)
                for item in required
                if not (item / ".metadata").is_file()
            ]
            if empty:
                raise FileNotFoundError(
                    f"resume checkpoint has incomplete {model_name} DCP payload: {empty}"
                )

        if progress.global_step >= int(self.config.ema_start_step):
            ema_path = path / f"generator_ema_rank{self.rank}.pt"
            if not ema_path.is_file():
                raise FileNotFoundError(
                    f"resume checkpoint is missing EMA shard: {ema_path}"
                )

    def _check_optimizer_impl(self, path: Path) -> None:
        """Refuse a checkpoint whose optimizer state was not written by this trainer's optimizer."""
        metadata_path = path / "checkpoint_metadata.json"
        saved = json.loads(metadata_path.read_text(encoding="utf-8")).get("optimizer_impl")
        if saved != OPTIMIZER_IMPL:
            raise RuntimeError(
                f"{metadata_path} records optimizer_impl={saved!r}: only checkpoints written by this "
                f"trainer (optimizer_impl={OPTIMIZER_IMPL!r}) can be resumed."
            )

    def load(self, checkpoint_path: str | Path) -> DMDTrainingProgress:
        """Restore a Stage-3 step directory into the live models and return its training progress.

        Collective: every rank runs each stage (metadata check, payload check, generator / critic weights and
        optimizers, the EMA shards once ema_start_step was reached, runtime_state) and the stages agree on
        failure.
        """
        from movieroll.utils.checkpoint import load_multi_model_checkpoint

        path = Path(checkpoint_path)
        progress = self.inspect_synchronized(path)
        self._run_synchronized_stage(
            lambda: self._validate_load_payload(path, progress),
            context="Stage3 checkpoint payload validation",
        )
        self._run_synchronized_stage(
            lambda: self._check_optimizer_impl(path),
            context="Stage3 checkpoint optimizer format check",
        )
        optimizers_dict = {
            "generator": self.optimizer_generator,
            "critic": self.optimizer_critic,
        }
        loaded_step = int(
            self._run_synchronized_stage(
                lambda: load_multi_model_checkpoint(
                    models_dict={"generator": self.generator, "critic": self.critic},
                    optimizers_dict=optimizers_dict,
                    checkpoint_dir=str(path),
                    strict=True,
                ),
                context="Stage3 model/optimizer checkpoint load",
            )
        )
        if loaded_step != progress.global_step:
            raise RuntimeError(
                "checkpoint metadata disagrees about saved step: "
                f"loader={loaded_step}, expected={progress.global_step}"
            )

        saved_training_step = progress.global_step
        def _load_ema() -> None:
            if saved_training_step >= int(self.config.ema_start_step):
                from movieroll.trainer.ema import FSDPEMAModel

                ema_path = path / f"generator_ema_rank{self.rank}.pt"
                ema = self.get_ema()
                if ema is None:
                    ema = FSDPEMAModel(
                        self.generator, decay=float(self.config.ema_weight)
                    )
                ema.load_state_dict(
                    torch.load(
                        ema_path, map_location="cpu", weights_only=True
                    )
                )
                self.set_ema(ema)
        self._run_synchronized_stage(
            _load_ema,
            context="Stage3 EMA checkpoint load",
        )

        def _load_runtime_state() -> None:
            if self.set_runtime_state is None:
                return
            metadata = json.loads(
                (path / "checkpoint_metadata.json").read_text(encoding="utf-8")
            )
            runtime_state = metadata.get("runtime_state")
            if runtime_state is not None:
                self.set_runtime_state(dict(runtime_state))
        self._run_synchronized_stage(
            _load_runtime_state,
            context="Stage3 runtime state load",
        )
        return progress

    def _export_generator(self, checkpoint_dir: Path) -> None:
        """Export the full generator state using the configured master dtype."""

        from safetensors.torch import save_file

        export_dtype = (
            torch.float32
            if str(self.config.master_weight_type).lower() == "fp32"
            else torch.bfloat16
        )
        model_state = self.generator.state_dict()
        cpu_state = {}
        for name, value in model_state.items():
            tensor = value.full_tensor() if hasattr(value, "full_tensor") else value
            cpu_state[name] = tensor.detach().to(dtype=export_dtype).cpu().clone()
        del model_state
        if self.rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_file(cpu_state, str(checkpoint_dir / "generator.safetensors"))
        del cpu_state

    def _export_generator_ema(self, checkpoint_dir: Path, ema) -> None:
        """Gather and export the full EMA generator state."""

        from safetensors.torch import save_file

        cpu_state = ema.gather_full_state_dict()
        if self.rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_file(cpu_state, str(checkpoint_dir / "generator_ema.safetensors"))
        del cpu_state

    def save(self, progress: DMDTrainingProgress) -> Path:
        """Write the step directory for `progress` and return it; COMPLETE is written last.

        Collective. Writes the generator / critic model and optimizer shards, the per-rank EMA shards, the
        merged generator.safetensors / generator_ema.safetensors exports and the metadata files (including the
        optimizer_impl marker that load checks).
        """
        from movieroll.utils.checkpoint import save_multi_model_checkpoint

        self._validate_progress(progress)
        checkpoint_dir = self.checkpoint_dir(progress.global_step)
        def _prepare_directory() -> None:
            if self.rank == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                # Never leave a stale completion marker in a directory being rewritten.
                (checkpoint_dir / "COMPLETE").unlink(missing_ok=True)
        self._run_synchronized_stage(
            _prepare_directory,
            context="Stage3 checkpoint directory preparation",
        )

        self._run_synchronized_stage(
            lambda: save_multi_model_checkpoint(
                models_dict={"generator": self.generator, "critic": self.critic},
                optimizers_dict={
                    "generator": self.optimizer_generator,
                    "critic": self.optimizer_critic,
                },
                rank=self.rank,
                output_dir=self.config.logdir,
                step=progress.global_step,
                clear_cache_between_models=True,
                fsdp_dp_mesh=self.fsdp_dp_mesh,
            ),
            context="Stage3 model/optimizer checkpoint save",
        )
        ema = self.get_ema()
        def _save_ema_shard() -> None:
            if ema is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    ema.state_dict(),
                    checkpoint_dir / f"generator_ema_rank{self.rank}.pt",
                )
        self._run_synchronized_stage(
            _save_ema_shard,
            context="Stage3 EMA shard save",
        )

        self._run_synchronized_stage(
            lambda: self._export_generator(checkpoint_dir),
            context="Stage3 generator export",
        )
        if ema is not None:
            self._run_synchronized_stage(
                lambda: self._export_generator_ema(checkpoint_dir, ema),
                context="Stage3 EMA generator export",
            )

        def _publish_metadata() -> None:
            if self.rank != 0:
                return
            metadata_path = checkpoint_dir / "checkpoint_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(
                {
                    "step": progress.global_step,
                    "phase_name": progress.phase.value,
                    "phase_step": progress.phase_step,
                    "global_step": progress.global_step,
                    "phase_complete": progress.phase_complete,
                    "world_size": self.world_size,
                    "fsdp_mesh_shape": self._mesh_shape(),
                    "optimizer_impl": OPTIMIZER_IMPL,
                }
            )
            if self.get_runtime_state is not None:
                metadata["runtime_state"] = self.get_runtime_state()
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # This sentinel must be the final filesystem write of the checkpoint.
            (checkpoint_dir / "COMPLETE").write_text(
                "phase_name={phase} phase_step={phase_step} "
                "global_step={global_step} phase_complete={complete}\n".format(
                    phase=progress.phase.value,
                    phase_step=progress.phase_step,
                    global_step=progress.global_step,
                    complete=int(progress.phase_complete),
                ),
                encoding="utf-8",
            )
        self._run_synchronized_stage(
            _publish_metadata,
            context="Stage3 checkpoint metadata publication",
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return checkpoint_dir
