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
# Modified from DeepSpeed
# ==============================================================================
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
#
# DeepSpeed Team

"""Sequence-parallel collectives (all-to-all / all-gather with autograd) and in-group broadcast helpers."""

from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch.nn import functional as F

from movieroll.commons.parallel_states import nccl_info


# Only pads the sequence dimension (dim 1).
def pad_on_where(input: torch.Tensor, group: dist.ProcessGroup):
    seq_world_size = dist.get_world_size(group)

    local_length = torch.tensor([input.shape[1]], dtype=torch.int32, device=input.device)
    total_lengths = [torch.empty(1, dtype=torch.int32, device=input.device) for _ in range(seq_world_size)]
    dist.all_gather(total_lengths, local_length, group=group)
    total_lengths = [length.item() for length in total_lengths]

    pad_on_rank_first = False
    pad_on_rank_last = False
    gap = 0

    if total_lengths[0] > total_lengths[-1]:
        pad_on_rank_last = True
        gap = total_lengths[0] - total_lengths[-1]
    elif total_lengths[0] < total_lengths[-1]:
        pad_on_rank_first = True
        gap = total_lengths[-1] - total_lengths[0]

    return gap, total_lengths, pad_on_rank_first, pad_on_rank_last

def get_padding_info(padding_info_dict, input, group):
    # Without an explicit padding_info_dict the padding is computed from the actual shard lengths.
    if padding_info_dict is None:
        gap, total_lengths, pad_on_rank_first, pad_on_rank_last = pad_on_where(input, group)
    else:
        gap = padding_info_dict['padding']
        total_lengths = padding_info_dict['total_lengths']
        pad_on_rank_first = padding_info_dict['pad_on_rank_first']
        pad_on_rank_last = padding_info_dict['pad_on_rank_last']
    return gap, total_lengths, pad_on_rank_first, pad_on_rank_last


def _all_to_all_4D(
    input: torch.Tensor, scatter_idx: int = 2, gather_idx: int = 1, group: dist.ProcessGroup = None,
    padding_info_dict: dict = None,
) -> torch.Tensor:
    """all-to-all for QKV: reshard `input` from `scatter_idx` onto `gather_idx`.

    2 / 1 turns (bs, seqlen/P, hc, hs) into (bs, seqlen, hc/P, hs) and is the only branch that handles the
    padding info; 1 / 2 is its inverse.
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        gap, seq_lens, pad_on_rank_first, pad_on_rank_last = get_padding_info(padding_info_dict, input, group)
        # padding on the last rank appends at the end of the sequence
        if gap > 0 and pad_on_rank_last and dist.get_group_rank(group, dist.get_rank()) == seq_world_size - 1:
            assert input.shape[1] == seq_lens[-1]
            input = F.pad(input, (0, 0, 0, 0, 0, gap))
        # padding on rank 0 prepends at the start of the sequence
        elif gap > 0 and pad_on_rank_first and dist.get_group_rank(group, dist.get_rank()) == 0:
            assert input.shape[1] == seq_lens[0]
            input = F.pad(input, (0, 0, 0, 0, gap, 0))

        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size

        # the number of heads must be divisible by the sp size
        if hc % seq_world_size != 0:
            raise ValueError(f"Invalid Head size: {hc}, which should be divisible by spsize {seq_world_size}")
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        if gap > 0:
            # padded at the end of the sequence (last rank): trim from the back
            if pad_on_rank_last:
                output = output[:, :-gap]
            # padded at the start of the sequence (rank 0): trim from the front
            elif pad_on_rank_first:
                output = output[:, gap:]

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape

        hc = shard_hc * seq_world_size
        padding, seq_lens, pad_on_rank_first, pad_on_rank_last = get_padding_info(padding_info_dict, input, group)
        # no explicit padding: derive it from the input seqlen
        if padding <= 0 and seqlen % seq_world_size != 0:
            new_seqlen = (seqlen // seq_world_size + 1) * seq_world_size
            gap = new_seqlen - seqlen
            # with scatter_idx == 1 every rank has the same input seqlen;
            # without an explicit padding side, pad on the last rank
            if not pad_on_rank_last and not pad_on_rank_first:
                pad_on_rank_last = True
            # pad at the end of the sequence (last rank)
            if pad_on_rank_last:
                input = F.pad(input, (0, 0, 0, 0, 0, gap))
            # pad at the start of the sequence (rank 0)
            elif pad_on_rank_first:
                input = F.pad(input, (0, 0, 0, 0, gap, 0))
            bs, seqlen, shard_hc, hs = input.shape
        # padding > 0 (e.g. windowed SP): seqlen may already be divisible by the sp size, so only the explicit
        # padding restores the original sequence length
        elif padding > 0:
            assert pad_on_rank_last or pad_on_rank_first, "pad_on_rank_last or pad_on_rank_first must be True when padding > 0"
            if pad_on_rank_last:
                input = F.pad(input, (0, 0, 0, 0, 0, padding))
            elif pad_on_rank_first:
                input = F.pad(input, (0, 0, 0, 0, padding, 0))
            bs, seqlen, shard_hc, hs = input.shape
            gap = padding
        else:
            gap = 0

        assert seqlen % seq_world_size == 0

        shard_seqlen = seqlen // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
        if gap > 0 and pad_on_rank_last and dist.get_group_rank(group, dist.get_rank()) == seq_world_size - 1:
            output = output[:, :-gap]
        elif gap > 0 and pad_on_rank_first and dist.get_group_rank(group, dist.get_rank()) == 0:
            output = output[:, gap:]

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        padding_info_dict: dict,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.padding_info_dict = padding_info_dict

        return _all_to_all_4D(
            input, scatter_idx, gather_idx, group=group,
            padding_info_dict=padding_info_dict
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        return (
            None, # group
            _all_to_all_4D(
                grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.group,
                padding_info_dict=ctx.padding_info_dict
            ), # input
            None, # scatter_idx
            None, # gather_idx
            None, # padding_info_dict
        )


def all_to_all_4D(
    input_: torch.Tensor, group: dist.ProcessGroup, scatter_dim: int = 2, gather_dim: int = 1,
    padding_info_dict: dict = None,
):
    return SeqAllToAll4D.apply(
        group, input_, scatter_dim, gather_dim,
        padding_info_dict
    )


class _AllGather(torch.autograd.Function):
    """All-gather along `dim` with autograd support; the shards may have different lengths."""

    @staticmethod
    def forward(ctx, input_, dim, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)
        input_size = list(input_.size())

        sizes = [None] * world_size
        dist.all_gather_object(sizes, input_.shape, group)

        ctx.input_size = input_size[dim]

        tensor_list = [torch.empty(sizes[i], dtype=input_.dtype, device=input_.device) for i in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        world_size = dist.get_world_size(group)
        global_rank = nccl_info.global_rank
        rank = dist.get_group_rank(group, global_rank)
        dim = ctx.dim
        input_size = ctx.input_size

        sizes = [None] * world_size
        dist.all_gather_object(sizes, input_size, group=group)

        grad_output = grad_output.contiguous()
        m = max(sizes)
        grad_chunks = []
        for seg in torch.split(grad_output, sizes, dim=dim):
            pad_len = m - seg.size(dim)
            if pad_len > 0:
                pad_shape = list(seg.shape)
                pad_shape[dim] = pad_len
                seg = torch.cat([seg, seg.new_zeros(pad_shape)], dim=dim)
            grad_chunks.append(seg.contiguous())
        grad_input = torch.empty_like(grad_chunks[rank])
        dist.reduce_scatter(grad_input, grad_chunks, op=dist.ReduceOp.SUM, group=group)
        grad_input = grad_input.narrow(dim, 0, sizes[rank])

        return grad_input.contiguous(), None, None


def all_gather(input_: torch.Tensor, dim: int = 1, group=None):
    """All-gather `input_` [B, H, S, D] and return the concatenation along `dim`."""
    return _AllGather.apply(input_, dim, group)


def is_group_rank0(group=None) -> bool:
    """Whether this rank is rank 0 inside ``group``.

    For source-only random draws followed by :func:`bcast_in_group`. It must not be used to skip the
    broadcast itself: every group member still has to enter the collective.
    """

    if not dist.is_initialized():
        return True
    return dist.get_rank(group) == 0


def bcast_in_group(t: torch.Tensor, group=None) -> torch.Tensor:
    """Broadcast ``t`` in place from rank 0 *of* ``group`` and return it.

    ``src`` is a GLOBAL rank, so ``dist.broadcast(t, src=0)`` would broadcast from outside any group that
    does not contain global rank 0 and silently hand every member somebody else's random draw;
    ``group_src=0`` means group rank 0 and needs torch >= 2.6 (see requirements.txt). ``group=None`` broadcasts over WORLD, and without an initialized
    process group ``t`` is returned unchanged.
    """
    if not dist.is_initialized():
        return t
    if group is None:
        dist.broadcast(t, src=0)
        return t
    dist.broadcast(t, group_src=0, group=group)
    return t
