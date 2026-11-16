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

from typing import List, Tuple, Union

import torch


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2, device="cuda"):
    """n-D meshgrid taking (num), (start, stop) with step 1, or (start, stop, num).

    Each argument is an int or an n-tuple stacked in dim order. Returns [dim, *num] float32 on `device`.
    """
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32, device=device)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

    return grid


#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/model.py#L80


def reshape_for_broadcast(
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    x: torch.Tensor,
):
    """Reshape real-valued (cos, sin), each [S, D], to broadcast against x [B, S, H, D] (sequence on dim 1)."""
    ndim = x.ndim
    assert 0 <= 1 < ndim

    assert freqs_cis[0].shape == (
        x.shape[1],
        x.shape[-1],
    ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)


def rotate_half(x):
    x_real, x_imag = (
        x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    )  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply RoPE to x [B, S, H, D] with interleaved-pair (cos, sin), each [S, D], in fp32 then back to x.dtype.

    PyTorch equivalent of the triton `fused_apply_rotary_emb`, used on CPU or with triton disabled.
    """
    cos, sin = reshape_for_broadcast(freqs_cis, x)  # [S, D]
    # real * cos - imag * sin
    # imag * cos + real * sin
    x_out = (x.float() * cos + rotate_half(x.float()) * sin).type_as(x)

    return x_out


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
    device="cuda",
):
    """RoPE for tokens with n-d structure: one rope per axis, `rope_dim_list[i]` dims each.

    sum(rope_dim_list) must equal the attention head_dim. `start, *args` are as in `get_meshgrid_nd`.
    Returns (cos, sin), each [prod(grid), sum(rope_dim_list)], real-valued and interleaved per pair.
    """

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list), device=device
    )  # [3, W, H, D] / [2, W, H]

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
            device=device,
        )  # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
    sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
    return cos, sin


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
    device="cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real-valued RoPE frequencies for `dim`, at positions `pos` ([S] or a count).

    Returns (cos, sin), each [S, D]. Every frequency is repeated for both elements of its pair
    (repeat_interleave(2)); the fused triton RoPE relies on that to share one kernel with its backward.
    """
    if isinstance(pos, int):
        pos = torch.arange(pos, dtype=torch.float32, device=device)

    # NTK-style rescale: stretches the embeddings to a longer sequence length without fine-tuning
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim)
    )  # [D/2]
    freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
    return freqs_cos, freqs_sin
