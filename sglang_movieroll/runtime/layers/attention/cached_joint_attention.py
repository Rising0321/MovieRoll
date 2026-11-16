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

"""Attention over a key set that is partly sharded over the ring group and partly replicated.

MovieRoll's image queries attend to ``[clean history, the window, the text]`` in one softmax. Two of those three
live in different places once SGLang's ring degree is above 1: the history and the window are sharded over the
ring group (each rank holds a contiguous block, see `ParallelLayout.ring_block`), while the text is replicated
on every rank. A ring pass must therefore see each sharded key exactly once and each text key exactly once --
handing the replicated text to every rank would count it ``ring_degree`` times and change the softmax.

That is why the ring is written here instead of taken from SGLang's `runtime/layers/usp.py::ring_attn`, which
wraps PyTorch's `_templated_ring_attention` and has no notion of a replicated block or of keys without
matching queries:

* `attend_with_lse` runs one attention and also returns the log-sum-exp of its scores (SGLang's FlashAttention
  and SageAttention implementations return it; for torch SDPA, which does not, the scores are evaluated
  explicitly in fp32. A server never takes that path: SGLang allows the ring only with FlashAttention or
  SageAttention, so it is there for a model built directly with SDPA, in fp32 on a small scale);
* `merge` combines two partial attentions over disjoint key sets into the attention over their union;
* `ring_attend` rotates the sharded keys and values around the ring group, merging as it goes, and merges the
  replicated block in once at the end.

Attention over a set of keys does not depend on their order, so the ring never has to restore the global token
order: the blocks may arrive in any order and a cached chunk may be stored as per-rank slices. Only the queries
keep their place.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from sglang_movieroll.runtime.distributed.layout import Block

Attended = Tuple[torch.Tensor, torch.Tensor]


def _lse_as_tokens(
    lse: torch.Tensor, batch: int, tokens: int, heads: int
) -> torch.Tensor:
    """A FlashAttention log-sum-exp, whatever layout it comes in, as ``[B, S, H]``."""
    if lse.shape == (batch, tokens, heads):
        return lse
    if lse.shape == (batch, heads, tokens):
        return lse.transpose(1, 2)
    raise RuntimeError(
        f"unexpected softmax_lse shape {tuple(lse.shape)} for q [{batch}, {tokens}, {heads}, *]"
    )


def returns_lse(attn) -> bool:
    """Whether this `LocalAttention` resolved to a backend that returns a log-sum-exp (FlashAttention and
    SageAttention both do, in natural log)."""
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    return getattr(attn, "backend", None) in (
        AttentionBackendEnum.FA,
        AttentionBackendEnum.SAGE_ATTN,
    )


def attend_with_lse(
    attn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Attended:
    """``(out, lse)`` of one attention; ``out`` ``[B, S, H, D]`` and ``lse`` ``[B, S, H]`` (natural log).

    ``attn`` is SGLang's `LocalAttention`. Its FlashAttention and SageAttention backends return the log-sum-exp
    themselves; for torch SDPA (the one backend with fp32) the scores are evaluated explicitly. The two agree to
    float rounding.
    """
    if returns_lse(attn):
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        # how many values the backend returns depends on its FlashAttention version; the first two are the
        # output and the log-sum-exp in every one of them
        attended = attn.attn_impl.forward(
            q, k, v, get_forward_context().attn_metadata, return_softmax_lse=True
        )
        out, lse = attended[0], attended[1]
        return out, _lse_as_tokens(lse, q.shape[0], q.shape[1], q.shape[2])

    scale = attn.softmax_scale
    scores = torch.einsum("bshd,bthd->bhst", q.float(), k.float()) * scale
    lse = torch.logsumexp(scores, dim=-1)  # [B, H, S]
    out = torch.einsum("bhst,bthd->bshd", torch.softmax(scores, dim=-1), v.float())
    return out.to(q.dtype), lse.transpose(1, 2)


def merge(first: Optional[Attended], second: Optional[Attended]) -> Attended:
    """Attention over the union of two disjoint key sets, from the two partial ``(out, lse)`` pairs.

    ``out = (o1 w1 + o2 w2) / (w1 + w2)`` with ``wi = exp(lse_i - max)``: the softmax denominators are added,
    which is what attending to both key sets at once would have done. Either side may be None (no keys).
    """
    if first is None:
        return second
    if second is None:
        return first
    out_a, lse_a = first
    out_b, lse_b = second
    top = torch.maximum(lse_a, lse_b)
    weight_a = (lse_a - top).exp()
    weight_b = (lse_b - top).exp()
    total = weight_a + weight_b
    out = (
        out_a.float() * weight_a.unsqueeze(-1) + out_b.float() * weight_b.unsqueeze(-1)
    ) / total.unsqueeze(-1)
    return out.to(out_a.dtype), top + total.log()


def ring_attend(
    attn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    blocks: List[Block],
    ring_rank: int,
    group,
    replicated: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Attention of this rank's queries over every ring rank's keys, plus a replicated key block counted once.

    ``k`` / ``v`` are this rank's own keys ``[B, valid, H, D]``, ``blocks[r].valid`` says how many keys rank
    ``r`` holds (every rank can work that out, so no shape has to be communicated), and the keys travel in
    buffers padded to ``blocks[r].length``; the padding is sliced off before the kernel, so it never enters a
    softmax.
    """
    world = len(blocks)
    if world == 1:
        merged = merge(
            attend_with_lse(attn, q, k, v) if k.shape[1] else None,
            attend_with_lse(attn, q, *replicated) if replicated is not None else None,
        )
        if merged is None:
            raise RuntimeError("attention over an empty key set")
        return merged[0]

    width = max(block.length for block in blocks)
    payload = torch.zeros(
        (2, q.shape[0], width, k.shape[2], k.shape[3]), dtype=k.dtype, device=k.device
    )
    payload[0, :, : k.shape[1]] = k
    payload[1, :, : v.shape[1]] = v
    send_to = (ring_rank + 1) % world
    recv_from = (ring_rank - 1) % world

    merged: Optional[Attended] = None
    for step in range(world):
        owner = (ring_rank - step) % world
        incoming = torch.empty_like(payload) if step < world - 1 else None
        requests = []
        if incoming is not None:
            # both ranks of a pair post the same two operations, so a plain isend / irecv pair cannot deadlock
            requests = dist.batch_isend_irecv(
                [
                    dist.P2POp(
                        dist.isend,
                        payload.contiguous(),
                        dist.get_global_rank(group, send_to),
                        group,
                    ),
                    dist.P2POp(
                        dist.irecv,
                        incoming,
                        dist.get_global_rank(group, recv_from),
                        group,
                    ),
                ]
            )
        valid = blocks[owner].valid
        if valid:
            merged = merge(
                merged,
                attend_with_lse(attn, q, payload[0, :, :valid], payload[1, :, :valid]),
            )
        for request in requests:
            request.wait()
        if incoming is not None:
            payload = incoming

    if replicated is not None:
        merged = merge(merged, attend_with_lse(attn, q, *replicated))
    if merged is None:
        raise RuntimeError("attention over an empty key set")
    return merged[0]
