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

"""Triton fused RoPE kernel, a faster `apply_rotary_emb` with autograd backward (training and inference).

Fuses rotate_half, the cos / sin products, the sum and the dtype cast into one kernel.

Layout (sequence-first, as used by flash attention):
  x:       [B, S, H, D]  - bf16 / fp16 / fp32
  cos/sin: [S, D]        - RoPE frequencies after repeat_interleave(2)
  out:     [B, S, H, D]  - same dtype as x

Pair-wise formula (cos / sin are repeat_interleaved, cos[2i] == cos[2i+1]):
  forward:  out[2i]   = x[2i]*cos[2i]   - x[2i+1]*sin[2i]
            out[2i+1] = x[2i+1]*cos[2i+1] + x[2i]*sin[2i+1]

  backward: dx[2i]   = dout[2i]*cos[2i]   + dout[2i+1]*sin[2i]
            dx[2i+1] = dout[2i+1]*cos[2i+1] - dout[2i]*sin[2i+1]

Unified as: out[d] = x[d]*cos[d] + sign(d)*x[d^1]*sin[d]
  partner(d) = d ^ 1 (XOR swap)
  forward sign:  -1 (even) / +1 (odd)
  backward sign: +1 (even) / -1 (odd), i.e. flipped

Key assumption: cos / sin went through repeat_interleave(2), i.e. cos[2i] == cos[2i+1]. `get_1d_rotary_pos_embed`
guarantees this invariant, which is what lets forward and backward share one kernel (in general the backward needs
sin[partner] rather than sin[d]; repeat_interleave makes them equal).
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


# ── Triton kernel shared by forward and backward; CONJUGATE selects the sign ──

@triton.jit
def _fused_rope_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Out_ptr,
    stride_xb,     # x.stride(0)
    stride_xs,     # x.stride(1)
    stride_xh,     # x.stride(2)
    stride_cs,     # cos.stride(0)
    seqlen,
    nheads,
    HEADDIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CONJUGATE: tl.constexpr,   # False = forward, True = backward (sign flipped)
):
    pid = tl.program_id(0)

    h = pid % nheads
    tmp = pid // nheads
    s = tmp % seqlen
    b = tmp // seqlen

    x_off = b * stride_xb + s * stride_xs + h * stride_xh
    c_off = s * stride_cs

    d = tl.arange(0, BLOCK_D)
    mask = d < HEADDIM

    x = tl.load(X_ptr + x_off + d, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(Cos_ptr + c_off + d, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(Sin_ptr + c_off + d, mask=mask, other=0.0).to(tl.float32)

    partner = d ^ 1
    partner_mask = partner < HEADDIM
    x_partner = tl.load(X_ptr + x_off + partner, mask=mask & partner_mask, other=0.0).to(tl.float32)

    if CONJUGATE:
        sign = tl.where((d & 1) == 0, 1.0, -1.0)
    else:
        sign = tl.where((d & 1) == 0, -1.0, 1.0)

    out = x * cos + (sign * x_partner) * sin
    tl.store(Out_ptr + x_off + d, out, mask=mask)


def _launch_rope_kernel(x, cos, sin, conjugate=False):
    """Launch the RoPE kernel."""
    B, S, H, D = x.shape
    out = torch.empty_like(x)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B * S * H,)
    _fused_rope_kernel[grid](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2),
        cos.stride(0),
        S, H,
        HEADDIM=D,
        BLOCK_D=BLOCK_D,
        CONJUGATE=conjugate,
        num_warps=4 if D >= 128 else 2,
        num_stages=1,
    )
    return out


# ── autograd Function: gradient support for training ──

class _FusedRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        # x: [B, S, H, D], cos/sin: [S, D]
        ctx.save_for_backward(cos, sin)
        return _launch_rope_kernel(x, cos, sin, conjugate=False)

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = _launch_rope_kernel(grad_output, cos, sin, conjugate=True)
        return grad_input, None, None



def fused_apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Triton replacement of `apply_rotary_emb` (training and inference).

    Args:
        x: [B, S, H, D], bf16 / fp16 / fp32
        freqs_cis: (cos, sin), each [S, D] (after repeat_interleave)

    Returns:
        [B, S, H, D], same dtype as x
    """
    cos, sin = freqs_cis
    assert x.ndim == 4, f"Expected x.ndim == 4 [B, S, H, D], got {x.ndim}"
    B, S, H, D = x.shape
    assert D % 2 == 0, f"Expected even head dim for RoPE, got D={D}"
    assert cos.shape == (S, D) and sin.shape == (S, D), \
        f"Expected cos/sin shape [{S}, {D}], got cos={cos.shape}, sin={sin.shape}"

    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    return _FusedRoPEFunc.apply(x, cos, sin)
