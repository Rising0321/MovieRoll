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

"""Row (H) sharding of the streaming VAE decode: the spatial-parallel half of PS-VAE (Parallel Streaming VAE).

Ported from `movieroll.models.autoencoders.row_shard`. Each rank of a process group owns a contiguous band of
latent rows. Spatial 3x3 convolutions exchange a ``pad``-row halo with the neighbouring ranks; at the global
top / bottom edges the halo is the replicate padding the single-device path would have applied, so sharding does
not change the function computed. Ops that act per position (RMSNorm, 1x1 convolutions, the DCAE pixel shuffle)
need no communication, and every rank's temporal cache only holds its own band. After the last band-local op an
all-gather hands every rank the complete decoded chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


def split_rows(num_rows: int, world_size: int) -> Tuple[int, ...]:
    """Contiguous, as-even-as-possible split; the first ``num_rows % world_size`` ranks get one extra row."""
    if num_rows < world_size:
        raise ValueError(
            f"cannot shard {num_rows} latent rows over {world_size} ranks: every rank needs at least one row"
        )
    base, extra = divmod(num_rows, world_size)
    return tuple(base + 1 if r < extra else base for r in range(world_size))


@dataclass(frozen=True)
class RowShard:
    group: "dist.ProcessGroup"
    rank: int
    global_ranks: Tuple[int, ...]
    latent_rows: Tuple[int, ...]

    @classmethod
    def create(
        cls, group: Optional["dist.ProcessGroup"], num_latent_rows: int
    ) -> "RowShard":
        if not dist.is_initialized():
            raise RuntimeError(
                "parallel VAE decode requires an initialized torch.distributed process group"
            )
        group = dist.group.WORLD if group is None else group
        rank = dist.get_rank(group)
        if rank < 0:
            raise RuntimeError(
                "this process is not a member of the VAE parallel decode group"
            )
        global_ranks = tuple(dist.get_process_group_ranks(group))
        return cls(
            group, rank, global_ranks, split_rows(num_latent_rows, len(global_ranks))
        )

    @property
    def world_size(self) -> int:
        return len(self.global_ranks)

    def local_rows(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's band of a tensor that every rank holds at latent resolution."""
        if x.shape[-2] != sum(self.latent_rows):
            raise RuntimeError(
                f"expected {sum(self.latent_rows)} latent rows to shard, got {x.shape[-2]}"
            )
        start = sum(self.latent_rows[: self.rank])
        return x[..., start : start + self.latent_rows[self.rank], :].contiguous()

    def exchange_halo(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        """``x`` with ``pad`` rows of the neighbouring bands (or replicate rows at the global edges) on each side."""
        if pad == 0:
            return x
        if x.shape[-2] < pad:
            raise RuntimeError(
                f"halo of {pad} rows exceeds the local band of {x.shape[-2]} rows"
            )
        halo_shape = (*x.shape[:-2], pad, x.shape[-1])
        ops = []
        top = bottom = None
        if self.rank > 0:
            peer = self.global_ranks[self.rank - 1]
            top = x.new_empty(halo_shape)
            ops.append(
                dist.P2POp(dist.isend, x[..., :pad, :].contiguous(), peer, self.group)
            )
            ops.append(dist.P2POp(dist.irecv, top, peer, self.group))
        if self.rank < self.world_size - 1:
            peer = self.global_ranks[self.rank + 1]
            bottom = x.new_empty(halo_shape)
            ops.append(
                dist.P2POp(dist.isend, x[..., -pad:, :].contiguous(), peer, self.group)
            )
            ops.append(dist.P2POp(dist.irecv, bottom, peer, self.group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        if top is None:
            top = x[..., :1, :].expand(halo_shape)
        if bottom is None:
            bottom = x[..., -1:, :].expand(halo_shape)
        return torch.cat([top, x, bottom], dim=-2)

    def gather_rows(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather every rank's band (at any resolution that is an integer multiple of the latent rows)."""
        local = self.latent_rows[self.rank]
        scale, remainder = divmod(x.shape[-2], local)
        if remainder:
            raise RuntimeError(
                f"{x.shape[-2]} rows is not a multiple of the {local} local latent rows"
            )
        max_rows = max(self.latent_rows) * scale
        if x.shape[-2] < max_rows:
            x = F.pad(x, (0, 0, 0, max_rows - x.shape[-2]))
        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(self.world_size)]
        dist.all_gather(parts, x, group=self.group)
        return torch.cat(
            [p[..., : n * scale, :] for p, n in zip(parts, self.latent_rows)], dim=-2
        )
