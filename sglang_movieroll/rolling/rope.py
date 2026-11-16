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

"""Rotary embedding of the rolling window.

Every token sits at a frame position of a fixed ``(max_num_latent_frames, H, W)`` grid. At inference the cached
history is re-packed from frame 0, so a window's queries take the rows behind the cache and its keys the cache
rows followed by its own; under ring parallelism a rank holds a contiguous block of each chunk, which is why
rows are asked for as ``(start, count)`` segments rather than as one range.

The table builder is `movieroll.models.transformers.modules.posemb_layers.get_nd_rotary_pos_embed` ported
(axis split, theta, interleaved pairs). SGLang's rotary kernel takes one value per rotated *pair*, so the rows
are handed out with the pair-repeated columns dropped; the table matches the `movieroll` original.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

RopeRows = Tuple[torch.Tensor, torch.Tensor]
Segment = Tuple[int, int]


def _axis_rows(
    dim: int, positions: torch.Tensor, theta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real-valued RoPE rows of one axis: ``(cos, sin)``, each ``[S, dim]`` with every frequency repeated per pair."""
    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device)[
                : dim // 2
            ]
            / dim
        )
    )
    freqs = torch.outer(positions, freqs)
    return freqs.cos().repeat_interleave(2, dim=1), freqs.sin().repeat_interleave(
        2, dim=1
    )


def _meshgrid(sizes: Sequence[int], device: torch.device) -> torch.Tensor:
    axes = [
        torch.linspace(0, n, n + 1, dtype=torch.float32, device=device)[:n]
        for n in sizes
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)


def nd_rotary_rows(
    rope_dim_list: Sequence[int],
    sizes: Sequence[int],
    theta: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` of the whole position grid, each ``[prod(sizes), sum(rope_dim_list)]``."""
    grid = _meshgrid(sizes, device)
    cos_parts, sin_parts = [], []
    for axis, dim in enumerate(rope_dim_list):
        cos, sin = _axis_rows(dim, grid[axis].reshape(-1), theta)
        cos_parts.append(cos)
        sin_parts.append(sin)
    return torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1)


class RopeTable:
    """Cos / sin rows (one column per rotated pair) of the position grid, built once per (H, W, device)."""

    def __init__(
        self, rope_dim_list: Sequence[int], theta: float, max_num_latent_frames: int
    ):
        self.rope_dim_list = list(rope_dim_list)
        self.theta = float(theta)
        self.max_num_latent_frames = int(max_num_latent_frames)
        self._tables: dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def table(
        self, height: int, width: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (height, width, str(device))
        if key not in self._tables:
            cos, sin = nd_rotary_rows(
                self.rope_dim_list,
                (self.max_num_latent_frames, height, width),
                self.theta,
                device,
            )
            # MovieRoll repeats every frequency for both elements of its pair; one column per pair is what
            # SGLang's rotary kernel reads.
            self._tables[key] = (cos[:, ::2].contiguous(), sin[:, ::2].contiguous())
        return self._tables[key]

    def rows(
        self, height: int, width: int, device: torch.device, segments: Sequence[Segment]
    ) -> RopeRows:
        """The rows of ``segments`` (each ``(first token, count)``) concatenated in order."""
        cos, sin = self.table(height, width, device)
        wanted = [(start, count) for start, count in segments if count > 0]
        if not wanted:
            return cos[:0], sin[:0]
        end = max(start + count for start, count in wanted)
        if end > cos.shape[0]:
            raise ValueError(
                f"token position {end} exceeds the position grid of {cos.shape[0]} tokens, i.e. "
                f"max_num_latent_frames={self.max_num_latent_frames}"
            )
        if len(wanted) == 1:
            start, count = wanted[0]
            return cos[start : start + count], sin[start : start + count]
        return (
            torch.cat([cos[start : start + count] for start, count in wanted]),
            torch.cat([sin[start : start + count] for start, count in wanted]),
        )


def chunk_spans(chunk_frames: Sequence[int], frame_tokens: int) -> List[Segment]:
    """``(first token, token count)`` of every cached chunk in the re-packed history."""
    spans: List[Segment] = []
    start = 0
    for frames in chunk_frames:
        spans.append((start, frames * frame_tokens))
        start += frames * frame_tokens
    return spans
