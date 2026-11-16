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
#
# ==============================================================================
# Portions adapted from vLLM-Omni v0.32.0 (commit 5917730a): the ring loop of `ring_attention` follows the one of
# vllm_omni/diffusion/attention/backends/ring_flash_attn.py and ring_pytorch_attn.py (RingComm send / recv / commit / wait
# from vllm_omni/diffusion/distributed/comm.py, update_out_and_lse from vllm_omni/diffusion/attention/backends/ring/
# ring_utils.py), itself adapted from https://github.com/feifeibear/long-context-attention
# ==============================================================================
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2024, Jiarui Fang.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Sequence parallelism of the rolling window over vLLM-Omni's sequence-parallel groups.

vLLM-Omni builds one sequence-parallel (SP) group of ``P = ulysses_degree * ring_degree * allgather_degree`` ranks, split
into Ulysses sub-groups and either Ring or AllGather-KV sub-groups. This module maps MovieRoll's tokens onto them:

* a sequence of ``n`` image tokens is cut into ``P`` equal shards of ``ceil(n / P)`` tokens (the last ones zero-padded); SP
  rank ``s`` holds shard ``s`` for everything outside attention (projections, modulation, MLPs);
* around attention, the Ulysses all-to-all (vLLM-Omni's `all_to_all_4D`) trades the shards of a Ulysses group for heads:
  each rank then holds a *block* -- the shards of its Ulysses group -- with ``heads / ulysses_degree`` heads;
* the keys / values of the other blocks are reached either by AllGather-KV (gathered once: the history is kept whole on
  every rank) or by Ring attention (passed around the ring group: each rank keeps only its block of the history);
* with neither (pure Ulysses) a block is the whole sequence.

The padding exists only between `shard` and the all-to-all and in the blocks; it is removed (by index, so its position
does not matter) before any key reaches a softmax. Positions are tracked explicitly, so the RoPE row of every token is
known whatever block it ends up in. With one SP rank every method is the identity.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


class SequenceLayout:
    """The SP group of this rank and the block structure of its Ulysses / Ring / AllGather-KV sub-groups."""

    def __init__(self, sp_group=None, ulysses_group=None, kv_group=None, kv_mode: Optional[str] = None):
        if kv_mode not in (None, "ring", "allgather"):
            raise ValueError(f"kv_mode must be None, 'ring' or 'allgather', got {kv_mode!r}")
        self.sp_group = sp_group
        self.world = 1 if sp_group is None else dist.get_world_size(sp_group)
        self.rank = 0 if sp_group is None else dist.get_rank(sp_group)
        self.ulysses_group = ulysses_group if self.world > 1 else None
        self.ulysses_world = 1 if self.ulysses_group is None else dist.get_world_size(ulysses_group)
        self.ulysses_rank = 0 if self.ulysses_group is None else dist.get_rank(ulysses_group)
        self.kv_group = kv_group if self.world > 1 and kv_mode is not None else None
        self.kv_mode = kv_mode if self.kv_group is not None else None
        self.kv_world = 1 if self.kv_group is None else dist.get_world_size(kv_group)
        self.kv_rank = 0 if self.kv_group is None else dist.get_rank(kv_group)

        # block_shards[s]: the shards (SP ranks, in all-to-all order) whose tokens SP rank s holds inside attention.
        # kv_members: the SP ranks of this rank's ring / all-gather group, in that group's rank order.
        if self.world == 1:
            self.block_shards = {0: [0]}
            self.kv_members = [0]
            return
        sp_ranks = dist.get_process_group_ranks(sp_group)
        position = {global_rank: index for index, global_rank in enumerate(sp_ranks)}
        mine = [position[r] for r in dist.get_process_group_ranks(self.ulysses_group)] if self.ulysses_group else [self.rank]
        everyone: list = [None] * self.world
        dist.all_gather_object(everyone, mine, group=sp_group)
        self.block_shards = {s: list(shards) for s, shards in enumerate(everyone)}
        self.kv_members = [position[r] for r in dist.get_process_group_ranks(self.kv_group)] if self.kv_group else [self.rank]
        covered = sorted(shard for member in self.kv_members for shard in self.block_shards[member])
        if covered != list(range(self.world)):
            raise RuntimeError(f"the blocks of the KV group do not cover the sequence: {covered}")

    @classmethod
    def from_framework(cls) -> "SequenceLayout":
        """The groups vLLM-Omni set up for this worker (the identity layout when it runs without sequence parallelism)."""
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        try:
            group = get_sp_group()
        except AssertionError:
            return cls()
        if group.world_size == 1:
            return cls()
        if group.ring_world_size > 1:
            return cls(group.device_group, group.ulysses_group, group.ring_group, "ring")
        if group.allgather_world_size > 1:
            return cls(group.device_group, group.ulysses_group, group.allgather_group, "allgather")
        return cls(group.device_group, group.ulysses_group)

    @property
    def enabled(self) -> bool:
        return self.world > 1

    @property
    def ring(self) -> bool:
        return self.kv_mode == "ring"

    # ------------------------------------------------------------------ positions
    def shard_size(self, tokens: int) -> int:
        return -(-tokens // self.world)

    def local_range(self, tokens: int) -> tuple[int, int]:
        """The ``[start, end)`` positions of this rank's shard (``end`` may pass ``tokens``: padding)."""
        size = self.shard_size(tokens)
        return self.rank * size, (self.rank + 1) * size

    def block_positions(self, tokens: int, device, member: Optional[int] = None) -> torch.Tensor:
        """Sequence positions of the tokens of SP rank ``member``'s attention block (padding has positions >= tokens)."""
        size = self.shard_size(tokens)
        shards = self.block_shards[self.rank if member is None else member]
        return torch.cat([torch.arange(s * size, (s + 1) * size, device=device) for s in shards])

    def valid_block_index(self, tokens: int, device, member: Optional[int] = None) -> torch.Tensor:
        """Indices into a block of its real (non-padding) tokens."""
        return torch.nonzero(self.block_positions(tokens, device, member) < tokens).flatten()

    # ------------------------------------------------------------------ token axis (outside attention)
    @staticmethod
    def _pad_tokens(x: torch.Tensor, length: int) -> torch.Tensor:
        if x.shape[1] == length:
            return x
        return torch.cat([x, x.new_zeros((x.shape[0], length - x.shape[1], *x.shape[2:]))], dim=1)

    def shard(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's shard of a full sequence ``[B, N, ...]``."""
        if not self.enabled:
            return x
        start, end = self.local_range(x.shape[1])
        return self._pad_tokens(x, self.shard_size(x.shape[1]) * self.world)[:, start:end].contiguous()

    def gather(self, x: torch.Tensor, tokens: int) -> torch.Tensor:
        """Inverse of `shard`: the shards of all SP ranks -> ``[B, N, ...]``."""
        if not self.enabled:
            return x
        parts = [torch.empty_like(x) for _ in range(self.world)]
        dist.all_gather(parts, x.contiguous(), group=self.sp_group)
        return torch.cat(parts, dim=1)[:, :tokens]

    # ------------------------------------------------------------------ Ulysses: shards <-> heads
    def to_heads(self, t: torch.Tensor) -> torch.Tensor:
        """``[B, shard, H, D]`` -> this rank's block ``[B, ulysses * shard, H / ulysses, D]``."""
        if self.ulysses_world == 1:
            return t
        from vllm_omni.diffusion.distributed.comm import all_to_all_4D

        return all_to_all_4D(t.contiguous(), scatter_idx=2, gather_idx=1, group=self.ulysses_group)

    def to_tokens(self, t: torch.Tensor) -> torch.Tensor:
        """Inverse of `to_heads`."""
        if self.ulysses_world == 1:
            return t
        from vllm_omni.diffusion.distributed.comm import all_to_all_4D

        return all_to_all_4D(t.contiguous(), scatter_idx=1, gather_idx=2, group=self.ulysses_group)

    def head_slice(self, t: torch.Tensor) -> torch.Tensor:
        """This rank's heads of a tensor every rank holds whole (the text stream)."""
        if self.ulysses_world == 1:
            return t
        per_rank = t.shape[2] // self.ulysses_world
        return t[:, :, self.ulysses_rank * per_rank:(self.ulysses_rank + 1) * per_rank].contiguous()

    def gather_heads(self, t: torch.Tensor) -> torch.Tensor:
        if self.ulysses_world == 1:
            return t
        parts = [torch.empty_like(t) for _ in range(self.ulysses_world)]
        dist.all_gather(parts, t.contiguous(), group=self.ulysses_group)
        return torch.cat(parts, dim=2)

    # ------------------------------------------------------------------ blocks -> the whole sequence
    def assemble(self, block: torch.Tensor, tokens: int) -> torch.Tensor:
        """The whole sequence ``[B, N, h, D]`` (position order) from this rank's block, gathering over the KV group."""
        if not self.enabled:
            return block[:, :tokens]
        device = block.device
        if self.kv_group is None:
            parts, positions = [block], [self.block_positions(tokens, device)]
        else:
            parts = [torch.empty_like(block) for _ in range(self.kv_world)]
            dist.all_gather(parts, block.contiguous(), group=self.kv_group)
            positions = [self.block_positions(tokens, device, member) for member in self.kv_members]
        whole = torch.cat(parts, dim=1)
        position = torch.cat(positions)
        keep = torch.nonzero(position < tokens).flatten()
        order = keep[torch.argsort(position[keep])]
        return whole.index_select(1, order)

    def chunk_positions(self, chunk_tokens: int, device) -> torch.Tensor:
        """Positions (inside one cached chunk) of the tokens this rank keeps of it: its block's real tokens under Ring,
        every token otherwise."""
        if not self.ring:
            return torch.arange(chunk_tokens, device=device)
        positions = self.block_positions(chunk_tokens, device)
        return positions[positions < chunk_tokens]

    def member_chunk_tokens(self, chunk_tokens: int, member: int) -> int:
        """How many tokens of a chunk the ring member ``member`` keeps."""
        size = self.shard_size(chunk_tokens)
        return sum(max(0, min(size, chunk_tokens - s * size)) for s in self.block_shards[member])


def _kernel(q: torch.Tensor):
    """A dense attention kernel that also returns the log-sum-exp ``[B, H, S]`` (vLLM-Omni's ring kernels)."""
    from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType, select_flash_attn_impl

    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        try:
            return select_flash_attn_impl(AttnType.VLLM_FA, stage="fwd-only")
        except (ImportError, RuntimeError, AssertionError):
            pass
    return select_flash_attn_impl(AttnType.TORCH, stage="fwd-only")


def ring_attention(
    layout: SequenceLayout,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_index: list[torch.Tensor],
    joint_k: Optional[torch.Tensor] = None,
    joint_v: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Non-causal attention of the local queries over the key / value blocks of every ring member.

    ``k`` / ``v`` are this rank's block, padded to the length every member uses; ``valid_index[j]`` selects the real
    tokens of ring member ``j``'s block (in ring-group rank order). ``joint_k`` / ``joint_v`` (the text, replicated on
    every rank) are attended once. The partial results are merged with vLLM-Omni's log-sum-exp update.
    """
    from vllm_omni.diffusion.attention.backends.ring.ring_utils import update_out_and_lse
    from vllm_omni.diffusion.distributed.comm import RingComm

    kernel = _kernel(q)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    comm = RingComm(layout.kv_group)
    world, rank = comm.world_size, comm.rank
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = lse = None
    for step in range(world):
        if step + 1 != world:
            next_k, next_v = comm.send_recv(k), comm.send_recv(v)
            comm.commit()
        index = valid_index[(rank - step) % world]
        step_k, step_v = k.index_select(1, index), v.index_select(1, index)
        if step == 0 and joint_k is not None:
            step_k, step_v = torch.cat([step_k, joint_k], dim=1), torch.cat([step_v, joint_v], dim=1)
        if step_k.shape[1] > 0:
            block_out, block_lse = kernel(q, step_k, step_v, dropout_p=0.0, softmax_scale=scale, causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse, lse_layout="bhs")
        if step + 1 != world:
            comm.wait()
            k, v = next_k, next_v
    return out.to(q.dtype)
