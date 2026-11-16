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

"""Where a request's work sits across the GPUs: SGLang's parallel groups, seen from MovieRoll's rolling window.

SGLang gives a worker four groups (`runtime/distributed/parallel_state.py`): tensor parallel, sequence parallel
(itself a Ulysses x Ring product), classifier-free-guidance parallel and data parallel. This module is the one
place that reads them, so the DiT and the stages never call SGLang's parallel state directly.

What each means for MovieRoll:

* **tensor parallel** (``--tp-size``) shards attention heads and the feed-forward width inside SGLang's own
  parallel linear layers, which all-reduce by themselves; nothing here has to move data. The only thing this
  module does is report ``tp_size`` so attention can divide the head count.
* **Ulysses** (``--ulysses-degree``) shards the image tokens of a window. Around attention each rank trades its
  token shard for a head shard (one all-to-all) and trades back, so attention sees the whole window with
  ``heads / (tp * ulysses)`` heads.
* **Ring** (``--ring-degree``) shards the post-Ulysses sequence once more, and attention is completed by
  rotating the keys and values around the ring group with a log-sum-exp merge (see
  `sglang_movieroll.runtime.layers.attention.cached_joint_attention`). Because SGLang builds the Ulysses
  subgroup out of consecutive sequence-parallel ranks (`set_seq_parallel_pg_by_sp_groups`), the block a ring
  rank holds after the all-to-all is a **contiguous** slice of the window, which is what makes the rotary rows
  and the KV cache slicing tractable.
* **classifier-free-guidance parallel** (``--enable-cfg-parallel``) puts the two guidance branches of the base
  chain on two ranks; see `sglang_movieroll.runtime.pipelines_core.stages.model_specific_stages.movieroll`.
* **data parallel** is refused by SGLang itself for diffusion (`ServerArgs._validate_parallelism`).

With every degree at 1 each method here is the identity, so the single-GPU path carries no communication code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _pad_tokens(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[1] == length:
        return x
    if x.shape[1] > length:
        return x[:, :length]
    pad = x.new_zeros((x.shape[0], length - x.shape[1], *x.shape[2:]))
    return torch.cat([x, pad], dim=1)


@dataclass(frozen=True)
class Block:
    """A contiguous slice ``[start, start + length)`` of a padded sequence, of which ``valid`` tokens are real."""

    start: int
    length: int  # the padded length every rank's block has (what the ring rotates)
    valid: int  # how many of them are real tokens

    @property
    def end(self) -> int:
        return self.start + self.valid


class ParallelLayout:
    """The parallel degrees of this worker, and the token bookkeeping that follows from them."""

    def __init__(
        self,
        tp_size: int = 1,
        ulysses_size: int = 1,
        ulysses_rank: int = 0,
        ring_size: int = 1,
        ring_rank: int = 0,
        sp_rank: int = 0,
    ):
        self.tp_size = tp_size
        self.ulysses_size = ulysses_size
        self.ulysses_rank = ulysses_rank
        self.ring_size = ring_size
        self.ring_rank = ring_rank
        self.sp_rank = sp_rank

    @classmethod
    def from_framework(cls) -> "ParallelLayout":
        """The groups SGLang set up for this worker (all degrees 1 when it runs on one GPU)."""
        from sglang.multimodal_gen.runtime.distributed import (
            get_tp_world_size,
            model_parallel_is_initialized,
        )
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_ring_parallel_rank,
            get_ring_parallel_world_size,
            get_sp_parallel_rank,
            get_ulysses_parallel_rank,
            get_ulysses_parallel_world_size,
        )

        if not model_parallel_is_initialized():
            return cls()
        return cls(
            tp_size=get_tp_world_size(),
            ulysses_size=get_ulysses_parallel_world_size(),
            ulysses_rank=get_ulysses_parallel_rank(),
            ring_size=get_ring_parallel_world_size(),
            ring_rank=get_ring_parallel_rank(),
            sp_rank=get_sp_parallel_rank(),
        )

    # ------------------------------------------------------------------ degrees
    @property
    def sp_size(self) -> int:
        """Sequence-parallel degree: SGLang requires ``sp_degree == ulysses_degree * ring_degree``."""
        return self.ulysses_size * self.ring_size

    @property
    def sharded(self) -> bool:
        return self.sp_size > 1

    @property
    def ringed(self) -> bool:
        return self.ring_size > 1

    def check_heads(self, heads: int) -> int:
        """Heads left on this rank after tensor and Ulysses parallelism; refuses a split that does not divide."""
        per_rank = self.tp_size * self.ulysses_size
        if heads % per_rank:
            raise ValueError(
                f"MovieRoll has {heads} attention heads, which --tp-size {self.tp_size} x --ulysses-degree "
                f"{self.ulysses_size} = {per_rank} does not divide; pick degrees whose product divides {heads}"
            )
        return heads // per_rank

    # ------------------------------------------------------------------ the token axis, before attention
    # A window of N tokens is split over the sp_size ranks in equal shards of ceil(N / sp_size) tokens, the last
    # shard zero-padded (the first chunk of a video is 5 latent frames, so N does not always divide). The padding
    # lives only between `shard` and the all-to-all, and between the all-to-all and `gather`: attention never
    # sees it.
    def padded(self, tokens: int) -> int:
        return _ceil_div(tokens, self.sp_size) * self.sp_size

    def local_range(self, tokens: int) -> tuple[int, int]:
        """The ``[start, end)`` token positions of this rank's pre-attention shard in the padded window."""
        per_rank = self.padded(tokens) // self.sp_size
        return self.sp_rank * per_rank, (self.sp_rank + 1) * per_rank

    # ------------------------------------------------------------------ the token axis, inside attention
    def ring_block(self, tokens: int) -> Block:
        """The slice of the window this rank holds once the Ulysses all-to-all has run.

        The Ulysses subgroup of a ring rank is made of consecutive sequence-parallel ranks, so after the
        all-to-all this rank holds the contiguous block number ``ring_rank`` of the padded window. Only part of
        the last block is real.
        """
        length = self.padded(tokens) // self.ring_size
        start = self.ring_rank * length
        return Block(
            start=start, length=length, valid=max(0, min(length, tokens - start))
        )

    def ring_blocks(self, tokens: int) -> List[Block]:
        """Every ring rank's block of a window, in ring-rank order."""
        length = self.padded(tokens) // self.ring_size
        return [
            Block(
                start=r * length,
                length=length,
                valid=max(0, min(length, tokens - r * length)),
            )
            for r in range(self.ring_size)
        ]

    def stored_tokens(self, tokens: int) -> int:
        """How many of a chunk's ``tokens`` this rank keeps in the KV cache (all of them without ring)."""
        return tokens if not self.ringed else self.ring_block(tokens).valid

    # ------------------------------------------------------------------ collectives
    def shard(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's slice of ``x`` ``[B, N, ...]`` along the token axis (zero-padded to the shard size)."""
        if not self.sharded:
            return x
        start, end = self.local_range(x.shape[1])
        return _pad_tokens(x, self.padded(x.shape[1]))[:, start:end].contiguous()

    def gather(self, x: torch.Tensor, tokens: int) -> torch.Tensor:
        """Inverse of `shard`: every rank's ``[B, ceil(N / sp), ...]`` -> ``[B, N, ...]``."""
        if not self.sharded:
            return x
        from sglang.multimodal_gen.runtime.distributed import (
            sequence_model_parallel_all_gather,
        )

        return sequence_model_parallel_all_gather(x.contiguous(), dim=1)[:, :tokens]

    def tokens_to_heads(self, t: torch.Tensor, tokens: int) -> torch.Tensor:
        """``[B, shard, H, D]`` -> this rank's ring block of the window with its own heads, ``[B, valid, H / u, D]``.

        The exchange runs over the **Ulysses** subgroup, not over the whole sequence-parallel group: with both
        degrees above 1 the two differ, and an all-to-all over the larger group would scatter the heads
        ``ulysses * ring`` ways. sglang's own Ulysses primitives take exactly this layout (heads at dim 2).
        """
        if self.ulysses_size > 1:
            from sglang.multimodal_gen.runtime.layers.usp import _usp_input_all_to_all

            t = _usp_input_all_to_all(t.contiguous(), head_dim=2)
        if not self.sharded:
            return t
        return t[:, : self.ring_block(tokens).valid].contiguous()

    def heads_to_tokens(self, t: torch.Tensor, tokens: int) -> torch.Tensor:
        """Inverse of `tokens_to_heads`: ``[B, valid, H / u, D]`` -> ``[B, shard, H, D]``."""
        if not self.sharded:
            return t
        t = _pad_tokens(t, self.ring_block(tokens).length)
        if self.ulysses_size > 1:
            from sglang.multimodal_gen.runtime.layers.usp import _usp_output_all_to_all

            t = _usp_output_all_to_all(t.contiguous(), head_dim=2)
        return t

    def head_slice(self, t: torch.Tensor) -> torch.Tensor:
        """This rank's Ulysses heads of a replicated ``[B, S, H, D]`` tensor (the text stream)."""
        if self.ulysses_size <= 1:
            return t
        per_rank = t.shape[2] // self.ulysses_size
        return t[
            :, :, self.ulysses_rank * per_rank : (self.ulysses_rank + 1) * per_rank
        ].contiguous()

    def gather_heads(self, t: torch.Tensor) -> torch.Tensor:
        """``[B, S, H / u, D]`` of every Ulysses rank -> the replicated ``[B, S, H, D]``."""
        if self.ulysses_size <= 1:
            return t
        parts = [torch.empty_like(t) for _ in range(self.ulysses_size)]
        torch.distributed.all_gather(parts, t.contiguous(), group=self.ulysses_group())
        return torch.cat(parts, dim=2)

    def ring_all_gather(self, t: torch.Tensor, blocks: List[Block]) -> torch.Tensor:
        """Every ring rank's block of a sequence -> the whole sequence, ``[B, sum(valid), ...]`` in rank order.

        The blocks have different valid lengths, so they travel padded to a common width and are trimmed on
        arrival. Used by KV recompute, which needs the whole prefix at once; the rolling steps never do this.
        """
        if not self.ringed:
            return t
        width = max(block.length for block in blocks)
        padded = _pad_tokens(t, width).contiguous()
        parts = [torch.empty_like(padded) for _ in blocks]
        torch.distributed.all_gather(parts, padded, group=self.ring_group())
        return torch.cat(
            [part[:, : block.valid] for part, block in zip(parts, blocks)], dim=1
        )

    # ------------------------------------------------------------------ groups
    def ulysses_group(self):
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_group,
        )

        return get_sp_group().ulysses_group

    def ring_group(self):
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_group,
        )

        return get_sp_group().ring_group


def guidance_group() -> Optional[tuple[int, int]]:
    """``(rank, world size)`` of SGLang's classifier-free-guidance group, or None when there is none."""
    from sglang.multimodal_gen.runtime.distributed import model_parallel_is_initialized
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_classifier_free_guidance_rank,
        get_classifier_free_guidance_world_size,
    )

    if not model_parallel_is_initialized():
        return None
    world = get_classifier_free_guidance_world_size()
    return None if world == 1 else (get_classifier_free_guidance_rank(), world)
