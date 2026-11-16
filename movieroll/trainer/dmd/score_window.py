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

"""Loss-only causal-VAE re-encoding for Stage3 score windows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import time
from typing import Iterator, Optional

import numpy as np
import torch


class ScoreWindowPurpose(str, Enum):
    GENERATOR_DMD = "generator_dmd"
    CRITIC = "critic"


@dataclass(frozen=True)
class ReEncodedScoreWindow:
    latents: torch.Tensor
    supervision_mask: Optional[torch.Tensor]
    score_start: int
    reencoded: bool


def _tensor_storage_bytes(value) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if is_dataclass(value):
        return sum(
            _tensor_storage_bytes(getattr(value, field.name))
            for field in fields(value)
        )
    if isinstance(value, dict):
        return sum(_tensor_storage_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_tensor_storage_bytes(item) for item in value)
    return 0


class CausalVaeScoreWindowReEncoder:
    """Re-encode score starts from a sink plus four recent latents.

    The object is deliberately not an ``nn.Module``: it shares the frozen VAE
    already owned by the generator pipeline and must not register a second
    module/state-dict path.
    """

    def __init__(
        self,
        vae,
    ) -> None:
        # Re-encoding is always on, so the VAE capability check below is unconditional.
        for name in ("decode", "encode", "clear_cache"):
            if not callable(getattr(vae, name, None)):
                raise TypeError(
                    f"score-window re-encoding requires VAE.{name}()"
                )

        self.vae = vae
        self.recent_latent_frames = 4
        self._active = False
        self._appended_latent_frames = 0
        self._required_latent_indices: Optional[set[int]] = None
        self._sink_latent: Optional[torch.Tensor] = None
        self._latent_cache: dict[int, torch.Tensor] = {}
        self._boundary_image_latents: dict[int, torch.Tensor] = {}
        self._stats: dict[str, float] = {}
        self._last_stats: dict[str, float] = {}

    @property
    def appended_latent_frames(self) -> int:
        return int(self._appended_latent_frames)

    @contextmanager
    def rollout(
        self, *, retain_score_starts: Optional[set[int]] = None
    ) -> Iterator["CausalVaeScoreWindowReEncoder"]:
        if self._active:
            raise RuntimeError("score-window re-encoder contexts must not be nested")

        retain = (
            None
            if retain_score_starts is None
            else {int(index) for index in retain_score_starts}
        )
        if retain is not None and any(index <= 0 for index in retain):
            raise ValueError(
                "retained score starts must contain only positive global indices"
            )
        required_indices = None
        if retain is not None:
            required_indices = {0}
            for score_start in retain:
                required_indices.update(
                    range(
                        max(1, score_start - self.recent_latent_frames + 1),
                        score_start + 1,
                    )
                )

        self.vae.clear_cache()
        self._active = True
        self._appended_latent_frames = 0
        self._required_latent_indices = required_indices
        self._sink_latent = None
        self._latent_cache.clear()
        self._boundary_image_latents.clear()
        self._stats = {
            "reencoded_window_count": 0.0,
            "identity_window_count": 0.0,
            "vae_local_decode_seconds": 0.0,
            "vae_boundary_encode_seconds": 0.0,
            "vae_local_decoded_latent_frames": 0.0,
            "latent_context_cache_peak_bytes": 0.0,
        }
        try:
            yield self
        finally:
            self._stats["latent_context_cache_bytes"] = float(
                _tensor_storage_bytes(self._sink_latent)
                + _tensor_storage_bytes(self._latent_cache)
            )
            self._last_stats = dict(self._stats)
            self._sink_latent = None
            self._latent_cache.clear()
            self._boundary_image_latents.clear()
            self._required_latent_indices = None
            self._appended_latent_frames = 0
            self._active = False
            self.vae.clear_cache()

    def stats(self) -> dict[str, float]:
        source = self._stats if self._active else self._last_stats
        return dict(source)

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                "score-window re-encoder operation requires an active rollout context"
            )

    def _vae_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        # The real VAE always has parameters; report a wrong VAE object the same way the
        # constructor does instead of silently returning the score window's own device/dtype.
        parameter = next(self.vae.parameters(), None)
        if parameter is None:
            raise RuntimeError(
                "score-window re-encoding requires a VAE with parameters"
            )
        return parameter.device, parameter.dtype

    def _inverse_scale(self, latents: torch.Tensor) -> torch.Tensor:
        scaling = getattr(self.vae.config, "scaling_factor", None)
        if scaling is None or float(scaling) == 0.0:
            raise RuntimeError("VAE scaling_factor must be non-zero")
        native = latents / float(scaling)
        shift = getattr(self.vae.config, "shift_factor", None)
        if shift is not None:
            native = native + float(shift)
        return native

    def _apply_scale(self, latents: torch.Tensor) -> torch.Tensor:
        scaling = getattr(self.vae.config, "scaling_factor", None)
        if scaling is None or float(scaling) == 0.0:
            raise RuntimeError("VAE scaling_factor must be non-zero")
        shift = getattr(self.vae.config, "shift_factor", None)
        if shift is not None:
            latents = latents - float(shift)
        return latents * float(scaling)

    def append_latents(
        self,
        *,
        latents: torch.Tensor,
        global_start: int,
    ) -> None:
        self._require_active()
        if latents.ndim != 5 or int(latents.shape[2]) <= 0:
            raise ValueError(
                "append_latents expects non-empty [B,C,T,H,W], got "
                f"{tuple(latents.shape)}"
            )
        if latents.requires_grad:
            raise RuntimeError("loss-side VAE may consume only detached latents")
        start = int(global_start)
        if start != self._appended_latent_frames:
            raise RuntimeError(
                "score-window latent coordinates are not contiguous: "
                f"global_start={start}, appended={self._appended_latent_frames}"
            )

        total_frames = int(latents.shape[2])
        for offset in range(total_frames):
            global_index = start + offset
            required = self._required_latent_indices
            if (
                global_index != 0
                and required is not None
                and global_index not in required
            ):
                continue
            latent = (
                latents[:, :, offset : offset + 1]
                .detach()
                .contiguous()
            )
            if global_index == 0:
                self._sink_latent = latent
                continue
            self._latent_cache[global_index] = latent
        self._appended_latent_frames += total_frames

        cache_bytes = (
            _tensor_storage_bytes(self._sink_latent)
            + _tensor_storage_bytes(self._latent_cache)
        )
        self._stats["latent_context_cache_peak_bytes"] = max(
            self._stats["latent_context_cache_peak_bytes"],
            float(cache_bytes),
        )

    def _sink_recent_context(self, score_start: int) -> tuple[torch.Tensor, int]:
        if self._sink_latent is None:
            raise RuntimeError("sink latent 0 is unavailable")
        # The sink plus the recent_latent_frames latents ending at score_start, clamped at index 1
        # because index 0 IS the sink.
        first_recent_index = max(
            1, score_start - self.recent_latent_frames + 1
        )
        pieces = [self._sink_latent]
        pieces.extend(
            self._latent_cache[index]
            for index in range(first_recent_index, score_start + 1)
        )
        target_local_index = score_start - first_recent_index + 1
        device, dtype = self._vae_device_dtype()
        context = torch.cat(pieces, dim=2).to(
            device=device, dtype=dtype
        )
        return context, target_local_index

    @contextmanager
    def _decode_without_tiling(self):
        spatial = bool(getattr(self.vae, "use_spatial_tiling", False))
        temporal = bool(getattr(self.vae, "use_temporal_tiling", False))
        if spatial:
            self.vae.disable_spatial_tiling()
        if temporal:
            self.vae.disable_temporal_tiling()
        try:
            yield
        finally:
            if spatial:
                self.vae.enable_spatial_tiling()
            if temporal:
                self.vae.enable_temporal_tiling()

    def decode_condition_frame(self, latent: torch.Tensor) -> np.ndarray:
        """One condition latent [B, C, 1, H, W] -> the SigLIP input, uint8 [B, H, W, 3].

        ACS (anchor-conditioned scoring) feeds the anchor to SigLIP exactly as i2v training did: decode the
        single latent frame and map it to 0..255. Like the boundary re-encode, tiling is off and the
        streaming cache is bypassed, so the result does not depend on the rollout's decode history.
        """
        self._require_active()
        if latent.ndim != 5 or int(latent.shape[2]) != 1:
            raise ValueError(
                f"condition latent must be [B,C,1,H,W], got {tuple(latent.shape)}"
            )
        device, dtype = self._vae_device_dtype()
        native = self._inverse_scale(latent.to(device=device, dtype=dtype))
        started = time.perf_counter()
        with self._decode_without_tiling(), torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            decoded = self.vae.decode(
                native, return_dict=False, streaming_mode=False
            )[0]
        self._stats["vae_local_decode_seconds"] += time.perf_counter() - started
        if int(decoded.shape[1]) != 3 or int(decoded.shape[2]) != 1:
            raise RuntimeError(
                f"condition frame decode must give one RGB frame, got {tuple(decoded.shape)}"
            )
        frame = ((decoded[:, :, 0].float() / 2 + 0.5).clamp(0, 1) * 255).round()
        return frame.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    def _get_boundary_image_latent(
        self, score_start: int, raw_score_window: torch.Tensor
    ) -> torch.Tensor:
        cached = self._boundary_image_latents.get(score_start)
        if cached is not None:
            return cached.to(
                device=raw_score_window.device,
                dtype=raw_score_window.dtype,
            )

        try:
            context, target_local_index = self._sink_recent_context(score_start)
        except KeyError as exc:
            raise RuntimeError(
                "missing sink+recent latent for score window: "
                f"score_start={score_start}, "
                f"appended={self._appended_latent_frames}, "
                f"cached={tuple(self._latent_cache.keys())}"
            ) from exc
        device, dtype = self._vae_device_dtype()
        native = self._inverse_scale(context)
        started = time.perf_counter()
        with self._decode_without_tiling(), torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            decoded = self.vae.decode(
                native, return_dict=False, streaming_mode=False
            )[0]
        self._stats["vae_local_decode_seconds"] += (
            time.perf_counter() - started
        )
        self._stats["vae_local_decoded_latent_frames"] += float(
            context.shape[2]
        )
        pixel_index = target_local_index * 4
        if pixel_index >= int(decoded.shape[2]):
            raise RuntimeError(
                "sink+recent decode output is too short: "
                f"pixel_index={pixel_index}, frames={decoded.shape[2]}"
            )
        pixel = decoded[:, :, pixel_index : pixel_index + 1]
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda" and dtype != torch.float32,
        ):
            image_latent = self.vae.encode(
                pixel, streaming_mode=False
            ).latent_dist.mode()
        self._stats["vae_boundary_encode_seconds"] += (
            time.perf_counter() - started
        )
        image_latent = self._apply_scale(image_latent).detach().to(
            device=raw_score_window.device,
            dtype=raw_score_window.dtype,
        )
        expected = (
            raw_score_window.shape[0],
            raw_score_window.shape[1],
            1,
            raw_score_window.shape[3],
            raw_score_window.shape[4],
        )
        if tuple(image_latent.shape) != expected:
            raise RuntimeError(
                "re-encoded boundary latent shape mismatch: "
                f"got {tuple(image_latent.shape)}, expected {expected}"
            )

        self._boundary_image_latents[score_start] = image_latent
        del context, native, decoded, pixel
        return image_latent

    def reencode(
        self,
        *,
        purpose: ScoreWindowPurpose,
        raw_score_window: torch.Tensor,
        score_start: int,
        supervision_mask: Optional[torch.Tensor],
    ) -> ReEncodedScoreWindow:
        if not isinstance(purpose, ScoreWindowPurpose):
            purpose = ScoreWindowPurpose(purpose)
        if raw_score_window.ndim != 5 or int(raw_score_window.shape[2]) <= 0:
            raise ValueError(
                "raw_score_window must be non-empty [B,C,T,H,W], got "
                f"{tuple(raw_score_window.shape)}"
            )
        start = int(score_start)
        if start < 0:
            raise ValueError(f"score_start must be non-negative, got {start}")
        self._require_active()
        if start == 0:
            self._stats["identity_window_count"] += 1.0
            return ReEncodedScoreWindow(
                latents=raw_score_window,
                supervision_mask=supervision_mask,
                score_start=start,
                reencoded=False,
            )

        if (
            purpose is ScoreWindowPurpose.GENERATOR_DMD
            and supervision_mask is None
        ):
            raise ValueError("generator re-encoding requires a supervision mask")
        if purpose is ScoreWindowPurpose.CRITIC and supervision_mask is not None:
            raise ValueError("critic re-encoding must not receive a supervision mask")
        if (
            supervision_mask is not None
            and supervision_mask.shape != raw_score_window.shape
        ):
            raise ValueError(
                "generator supervision mask must match the raw score window: "
                f"{tuple(supervision_mask.shape)} != "
                f"{tuple(raw_score_window.shape)}"
            )

        image_latent = self._get_boundary_image_latent(
            start, raw_score_window
        )
        score_window = torch.cat(
            (image_latent, raw_score_window[:, :, 1:]), dim=2
        )
        if score_window.shape != raw_score_window.shape:
            raise RuntimeError(
                "re-encoded score window changed shape: "
                f"{tuple(raw_score_window.shape)} -> {tuple(score_window.shape)}"
            )

        mask = supervision_mask
        if mask is not None:
            mask = mask.clone()
            mask[:, :, :1] = False
        self._stats["reencoded_window_count"] += 1.0
        return ReEncodedScoreWindow(
            latents=score_window,
            supervision_mask=mask,
            score_start=start,
            reencoded=True,
        )
