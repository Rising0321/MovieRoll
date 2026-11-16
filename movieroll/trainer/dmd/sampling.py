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

"""Validation sampling for Stage3 rolling DMD."""

from __future__ import annotations

import csv
import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from movieroll.models.autoencoders.streaming_vae import decode_latent_chunks_streaming
from movieroll.pipelines.rolling_dmd_runtime import RollingMode, ordered_event_columns
from movieroll.trainer.dmd.data import DMDTrainingPhase
from movieroll.utils.file_utils import save_videos_grid
from movieroll.utils.image_condition import encode_input_image
from movieroll.utils.torch_utils import PRECISION_TO_TYPE, count_failed_ranks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingRow:
    """One benchmark row: its ordered event prompts, and the input image an ACS run conditions on."""

    events: list[str]
    image: Optional[Path]

class DMDSampler:
    """Load ordered event prompts, apply EMA safely, and run production sampling."""

    def __init__(
        self,
        *,
        config,
        pipeline,
        generator_model,
        vae,
        ema,
        dp_rank: int,
        dp_degree: int,
        device: torch.device,
        logger_=None,
    ):
        self.config = config
        self.pipeline = pipeline
        self.generator_model = generator_model
        self.vae = vae
        self.ema = ema
        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.device = device
        self.logger = logger_ or logger

    @staticmethod
    def _is_sp_leader() -> bool:
        """Return whether this rank owns output I/O for its SP replica."""

        from movieroll.commons.parallel_states import nccl_info

        return int(getattr(nccl_info, "rank_within_spgroup", 0)) == 0

    def _synchronize_error(
        self, error: Exception | None, *, context: str
    ) -> Exception | None:
        failure_count = count_failed_ranks(error is not None, self.device)
        if not failure_count:
            return None
        if error is not None:
            return error
        return RuntimeError(
            f"{context} failed on {failure_count} peer rank(s)"
        )

    def _decode_and_save(
        self,
        latents: torch.Tensor,
        save_path: str | Path,
    ) -> None:
        """Decode one latent video with the exact T-cache streaming path."""

        vae_dtype = PRECISION_TO_TYPE[str(self.config.vae_precision)]
        latents = latents.to(self.device, dtype=vae_dtype)
        scaling_factor = getattr(self.vae.config, "scaling_factor", None)
        if scaling_factor is None or float(scaling_factor) == 0.0:
            raise RuntimeError(
                "Stage3 sampling requires a non-zero VAE scaling_factor"
            )
        latents = latents / float(scaling_factor)
        shift_factor = getattr(self.vae.config, "shift_factor", None)
        if shift_factor is not None:
            latents = latents + float(shift_factor)

        if not callable(getattr(self.vae, "clear_cache", None)):
            raise TypeError("Stage3 sampling requires the streaming VAE")

        max_chunk_size = self.config.dmd.get(
            "vae_stream_decode_max_chunk_size", 4
        )
        latent_chunks = (
            (latents,)
            if max_chunk_size is None
            else latents.split(int(max_chunk_size), dim=2)
        )
        decoded_chunks = decode_latent_chunks_streaming(
            self.vae, latent_chunks, vae_dtype=vae_dtype, device=self.device
        )

        video = torch.cat(decoded_chunks, dim=2)
        video = (video / 2 + 0.5).clamp(0, 1).cpu().float()
        save_videos_grid(video, str(save_path), rescale=False, n_rows=1, fps=24)
        del video, decoded_chunks

    def _resolve_csv(self) -> Path:
        path = Path(self.config.sampling.test_event_csv)
        if not path.is_file():
            raise FileNotFoundError(f"sampling event CSV does not exist: {path}")
        return path

    def _is_acs(self) -> bool:
        """Whether this run scores with ACS, which makes its samples image-to-video."""

        return str(self.config.dmd.get("score_mode", "t2v")) == "acs"

    def load_event_rows(self) -> tuple[Path, list[SamplingRow]]:
        """The benchmark CSV named by `sampling.test_event_csv`, read through its ordered event columns.

        An empty cell is an error, not a skipped row: silently losing prompts would change what
        validation measures. An `input_image` cell names an image relative to the CSV's directory and is
        required of every row under ACS, which `validate_stage3_config` has already checked.
        """
        path = self._resolve_csv()
        rows: list[SamplingRow] = []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = ordered_event_columns(tuple(reader.fieldnames or ()))
            for row_index, row in enumerate(reader):
                events = [(row.get(column) or "").strip() for column in columns]
                if any(not event for event in events):
                    raise ValueError(
                        f"{path}: row {row_index + 2} has an empty ordered event column"
                    )
                image = (row.get("input_image") or "").strip()
                rows.append(
                    SamplingRow(
                        events=events,
                        image=(path.parent / image) if image else None,
                    )
                )
        if not rows:
            raise ValueError(f"sampling event CSV has no usable rows: {path}")
        missing = [str(row.events[0]) for row in rows if row.image is None]
        if self._is_acs() and missing:
            raise ValueError(
                f"{path}: dmd.score_mode='acs' samples image-to-video, so every row needs an "
                f"input_image; {len(missing)} row(s) have none"
            )
        return path, rows

    def run(
        self,
        *,
        phase: DMDTrainingPhase,
        step: int,
        num_samples: Optional[int],
        output_subdir: str,
    ) -> None:
        preflight_error = None
        source = None
        local_rows: list[SamplingRow] = []
        row_count = 0
        start = 0
        try:
            source, rows = self.load_event_rows()
            if num_samples is not None:
                rows = rows[: min(int(num_samples), len(rows))]
            if not rows:
                raise ValueError("sampling requires at least one event row")
            row_count = len(rows)
            per_rank = max(1, (row_count + self.dp_degree - 1) // self.dp_degree)
            padded = rows + [rows[0]] * (per_rank * self.dp_degree - row_count)
            start = self.dp_rank * per_rank
            local_rows = padded[start:start + per_rank]
            output_dir = (
                Path(self.config.logdir) / output_subdir / f"step_{step:06d}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            preflight_error = exc
        synchronized = self._synchronize_error(
            preflight_error, context="Stage3 sampling preflight"
        )
        if synchronized is not None:
            raise synchronized

        sampling = self.config.sampling
        was_training = bool(self.generator_model.training)
        ema_applied = False
        setup_error = None
        try:
            self.generator_model.eval()
            if self.ema is not None:
                self.ema.store(self.generator_model)
                ema_applied = True
                self.ema.copy_to(self.generator_model)
        except Exception as exc:
            setup_error = exc
        synchronized = self._synchronize_error(
            setup_error, context="Stage3 sampling EMA setup"
        )
        if synchronized is not None:
            restore_error = None
            try:
                if ema_applied:
                    self.ema.restore(self.generator_model)
                self.generator_model.train(was_training)
            except Exception as exc:
                restore_error = exc
            restore_failure = self._synchronize_error(
                restore_error, context="Stage3 sampling setup rollback"
            )
            if restore_failure is not None and setup_error is None:
                raise restore_failure
            raise synchronized

        run_error: Exception | None = None
        try:
            for local_index, row in enumerate(local_rows):
                events = row.events
                global_index = start + local_index
                local_error = None
                result = None
                latents = None
                cond_latents = None
                try:
                    generator = torch.Generator(device="cpu").manual_seed(
                        int(sampling.seed) + global_index
                    )
                    # Sampling is inference: an ACS run trains with context noise and samples without it,
                    # so the sigma comes from its own key; absent or null, it is the training value.
                    sampling_context_noise = self.config.dmd.get("sampling_context_noise")
                    context_noise = float(
                        self.config.dmd.context_noise
                        if sampling_context_noise is None
                        else sampling_context_noise
                    )
                    # ACS distils image-to-video, so its samples condition on the row's real image;
                    # otherwise training-time sampling is text-to-video.
                    if self._is_acs():
                        cond_latents = encode_input_image(
                            self.vae,
                            self.config.vae_precision,
                            self.device,
                            row.image,
                            int(sampling.height),
                            int(sampling.width),
                        )
                    with torch.no_grad():
                        result = self.pipeline.execute(
                            mode=RollingMode.SAMPLING,
                            prompt=events[0],
                            event_prompts=events,
                            event_timestamps=None,
                            height=int(sampling.height),
                            width=int(sampling.width),
                            video_length=int(sampling.video_length),
                            denoising_step_list=list(self.config.dmd.denoising_step_list),
                            context_noise=context_noise,
                            generator=generator,
                            multitask_type="i2v" if cond_latents is not None else "t2v",
                            cond_latents=cond_latents,
                        )
                    # Every latent channel is content: the generator's in/out_channels are both 32, so
                    # no channel slice is needed here.
                    latents = result.generated_latents
                    if global_index < row_count and self._is_sp_leader():
                        video_name = f"video_prompt{global_index:04d}.mp4"
                        if bool(sampling.decode_to_video):
                            self._decode_and_save(latents, output_dir / video_name)
                        if bool(sampling.save_latents):
                            torch.save(
                                latents.cpu(),
                                output_dir / f"latents_prompt{global_index:04d}.pt",
                            )
                        sidecar = {
                            "step": step,
                            "phase_name": phase.value,
                            "prompt_idx": global_index,
                            "video": video_name if bool(sampling.decode_to_video) else None,
                            "event_csv": str(source),
                            "event_prompts": events,
                            "multitask_type": "i2v" if cond_latents is not None else "t2v",
                            "input_image": None if row.image is None else str(row.image),
                            "switch_mode": "even_split",
                            "split_chunk_indices": list(result.event_split_indices),
                            "split_pixel_frames": list(result.event_split_pixel_frames),
                            "selected_round_indices": list(result.selected_round_indices),
                            "num_window_predictions": len(result.window_predictions),
                            "generated_latent_shape": list(latents.shape),
                            "video_length_pixel": int(sampling.video_length),
                            "resolution": [int(sampling.height), int(sampling.width)],
                            "seed": int(sampling.seed) + global_index,
                        }
                        (
                            output_dir / f"events_prompt{global_index:04d}.json"
                        ).write_text(
                            json.dumps(sidecar, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                except Exception as exc:
                    local_error = exc
                finally:
                    del latents, result, cond_latents
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                run_error = self._synchronize_error(
                    local_error,
                    context=f"Stage3 sampling sample {global_index}",
                )
                if run_error is not None:
                    break
        finally:
            # Restore live training weights before any cleanup that might fail.
            restore_error = None
            try:
                if ema_applied:
                    self.ema.restore(self.generator_model)
                self.generator_model.train(was_training)
            except Exception as exc:
                restore_error = exc
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        restore_error = self._synchronize_error(
            restore_error, context="Stage3 sampling EMA restore"
        )
        if restore_error is not None:
            raise restore_error
        if run_error is not None:
            raise run_error
