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

"""Triton fused RMSNorm kernel with autograd backward (training and inference).

Math:
  rstd  = rsqrt(mean(x^2, dim=-1) + eps)
  out   = x * rstd * weight

Backward:
  dx = rstd * (dy * w  -  x_hat * mean(dy * w * x_hat, dim=-1))
  dw = sum_over_rows(dy * x * rstd)

Layout:  x [*batch, D],  weight [D],  out [*batch, D]
"""

import torch
import triton
import triton.language as tl


def _get_block_size(num_cols: int, element_size: int) -> int:
    max_fused_size = 65536 // element_size
    block_size = min(max_fused_size, triton.next_power_of_2(num_cols))
    if num_cols > block_size:
        raise RuntimeError(
            f"RMSNorm does not support feature dim >= 64KB per row: "
            f"N={num_cols}, element_size={element_size}, block_size={block_size}"
        )
    return block_size


def _num_warps(block_size: int) -> int:
    return min(max(block_size // 256, 1), 8)


# ────────────────────────── Forward kernel ──────────────────────────

@triton.jit
def _rms_norm_fwd_kernel(
    X_ptr, W_ptr, Out_ptr, Rstd_ptr,
    stride_x_row,
    N,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    off = row * stride_x_row
    d = tl.arange(0, BLOCK_D)
    mask = d < N

    x = tl.load(X_ptr + off + d, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + d, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    out = x * rstd * w

    tl.store(Rstd_ptr + row, rstd)
    tl.store(Out_ptr + off + d, out, mask=mask)


# ────────────────────────── Backward dx kernel ──────────────────────────

@triton.jit
def _rms_norm_bwd_dx_kernel(
    DY_ptr, X_ptr, W_ptr, Rstd_ptr, DX_ptr,
    stride_x_row,
    N,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    off = row * stride_x_row
    d = tl.arange(0, BLOCK_D)
    mask = d < N

    dy = tl.load(DY_ptr + off + d, mask=mask, other=0.0).to(tl.float32)
    x  = tl.load(X_ptr  + off + d, mask=mask, other=0.0).to(tl.float32)
    w  = tl.load(W_ptr + d, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd_ptr + row).to(tl.float32)

    x_hat = x * rstd
    dy_w = dy * w
    c = tl.sum(dy_w * x_hat, axis=0) / N
    dx = rstd * (dy_w - x_hat * c)

    tl.store(DX_ptr + off + d, dx, mask=mask)


# ────────────────────────── Backward dw kernel ──────────────────────────

@triton.jit
def _rms_norm_bwd_dw_kernel(
    DY_ptr, X_ptr, Rstd_ptr, DW_ptr,
    stride_x_row,
    N, num_rows,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Each program accumulates BLOCK_ROWS rows locally, then atomic_adds into dw."""
    pid = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    mask_d = d < N

    row_start = pid * BLOCK_ROWS
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for i in range(BLOCK_ROWS):
        row = row_start + i
        if row < num_rows:
            off = row * stride_x_row
            dy = tl.load(DY_ptr + off + d, mask=mask_d, other=0.0).to(tl.float32)
            x  = tl.load(X_ptr  + off + d, mask=mask_d, other=0.0).to(tl.float32)
            rstd = tl.load(Rstd_ptr + row).to(tl.float32)
            acc += dy * x * rstd

    tl.atomic_add(DW_ptr + d, acc, mask=mask_d)


# ────────────────────────── Python wrappers ──────────────────────────

def _rms_norm_forward(x, weight, eps):
    x_flat = x.reshape(-1, x.shape[-1])
    num_rows, N = x_flat.shape
    block_size = _get_block_size(N, x.element_size())
    out_flat = torch.empty_like(x_flat)
    rstd = torch.empty(num_rows, dtype=torch.float32, device=x.device)

    _rms_norm_fwd_kernel[(num_rows,)](
        x_flat, weight, out_flat, rstd,
        x_flat.stride(0), N, eps,
        BLOCK_D=block_size,
        num_warps=_num_warps(block_size),
        num_stages=1,
    )
    return out_flat.reshape(x.shape), rstd


def _rms_norm_backward(dy, x, weight, rstd):
    dy_flat = dy.reshape(-1, dy.shape[-1]).contiguous()
    x_flat  = x.reshape(-1, x.shape[-1]).contiguous()
    num_rows, N = x_flat.shape
    block_size = _get_block_size(N, x.element_size())
    nw = _num_warps(block_size)

    # dx
    dx_flat = torch.empty_like(x_flat)
    _rms_norm_bwd_dx_kernel[(num_rows,)](
        dy_flat, x_flat, weight, rstd, dx_flat,
        x_flat.stride(0), N,
        BLOCK_D=block_size, num_warps=nw, num_stages=1,
    )

    # dw: local accumulation + atomic_add
    dw = torch.zeros(N, dtype=torch.float32, device=x.device)
    BLOCK_ROWS = 128
    num_blocks = triton.cdiv(num_rows, BLOCK_ROWS)
    _rms_norm_bwd_dw_kernel[(num_blocks,)](
        dy_flat, x_flat, rstd, dw,
        x_flat.stride(0), N, num_rows,
        BLOCK_D=block_size, BLOCK_ROWS=BLOCK_ROWS,
        num_warps=nw, num_stages=1,
    )
    return dx_flat.reshape(dy.shape), dw.to(weight.dtype)


# ────────────────────────── autograd.Function ──────────────────────────

class _FusedRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        out, rstd = _rms_norm_forward(x, weight, eps)
        ctx.save_for_backward(x, weight, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rstd = ctx.saved_tensors
        dx, dw = _rms_norm_backward(grad_output.contiguous(), x, weight, rstd)
        return dx, dw, None


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """RMSNorm over the last dim of x (any leading shape); statistics and scaling in fp32, output in x.dtype."""
    return _FusedRMSNormFunc.apply(x.contiguous(), weight, eps)
