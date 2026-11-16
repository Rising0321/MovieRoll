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

from typing import Callable

import torch
import torch.nn as nn

from movieroll.commons.parallel_states import get_sequence_parallel_state, nccl_info


def _torch_chunk_range(total_len: int, chunks: int, rank: int):
    """Return the slice range that matches torch.chunk(...)[rank]."""
    chunk_size = (total_len + chunks - 1) // chunks
    start = rank * chunk_size
    end = min(start + chunk_size, total_len)
    if start >= total_len:
        raise ValueError(f"rank {rank} is out of range for length {total_len} split into {chunks} chunks")
    return start, end


def _expand_frame_values_local(values: torch.Tensor, frame_tokens: int, sp_size: int, sp_rank: int):
    """Expand [B, T, C] frame-wise values only for the local SP shard."""
    if values.shape[1] == 0:
        return values[:, :0, :]
    total_tokens = values.shape[1] * frame_tokens
    start, end = _torch_chunk_range(total_tokens, sp_size, sp_rank)
    if start == end:
        return values[:, :0, :]
    token_to_frame = torch.arange(start, end, device=values.device) // frame_tokens
    return values.index_select(1, token_to_frame)


def _expand_frame_values_for_sp(values: torch.Tensor, frame_tokens: int, cond_latents_num_frames=None):
    """Expand [B, T, C] frame-wise values for the current SP rank only."""
    sp_size = nccl_info.sp_size
    sp_rank = nccl_info.rank_within_spgroup
    if cond_latents_num_frames is not None and cond_latents_num_frames > 0:
        cond_values = values[:, :cond_latents_num_frames, :]
        noisy_values = values[:, cond_latents_num_frames:, :]
        cond_local = _expand_frame_values_local(cond_values, frame_tokens, sp_size, sp_rank)
        noisy_local = _expand_frame_values_local(noisy_values, frame_tokens, sp_size, sp_rank)
        return torch.cat([cond_local, noisy_local], dim=1)
    return _expand_frame_values_local(values, frame_tokens, sp_size, sp_rank)

class ModulateDiT(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(
            hidden_size, factor * hidden_size, bias=True, **factory_kwargs
        )
        # zero init: the block starts as the identity
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_out = self.linear(self.act(x))

        return x_out


def modulate(x, shift=None, scale=None, thw=None, cond_latents_num_frames=None, num_chunks=None):
    """Modulate by shift and scale: out = x * (1 + scale) + shift.

    Layouts of shift / scale: frame-wise [B, T, C] for image tokens (`thw` given), broadcast over the H*W tokens
    of each frame of x [B, T*H*W, C] -- under sequence parallelism only the local shard is expanded, with cond
    and noisy frames sharded separately when `cond_latents_num_frames > 0`; [B, num_chunks, C] for text tokens
    (`num_chunks` given), per-chunk over equal-length chunks of x; otherwise [B, C] or broadcastable as is.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale[:, None, :])
    elif scale is None:
        return x + shift[:, None, :]
    else:
        if thw is not None and len(thw) == 3:
            _, H, W = thw
            B, L, C = x.shape
            frame_tokens = H * W
            if get_sequence_parallel_state():
                scale_seq = _expand_frame_values_for_sp(scale, frame_tokens, cond_latents_num_frames)
                shift_seq = _expand_frame_values_for_sp(shift, frame_tokens, cond_latents_num_frames)
                assert scale_seq.shape[1] == L, f'scale_seq.shape[1]: {scale_seq.shape[1]}, L: {L}'
                assert shift_seq.shape[1] == L, f'shift_seq.shape[1]: {shift_seq.shape[1]}, L: {L}'
                return x * (1 + scale_seq) + shift_seq
            T = scale.shape[1]
            assert T * frame_tokens == L, f'T * H * W: {T * frame_tokens}, L: {L}'
            x_5d = x.reshape(B, T, H, W, C)
            scale_5d = scale[:, :, None, None, :]
            shift_5d = shift[:, :, None, None, :]
            return (x_5d * (1 + scale_5d) + shift_5d).reshape(B, L, C)

        elif num_chunks is not None and scale.shape[1] == num_chunks:
            B, C = x.shape[0], x.shape[-1]
            # [B, num_chunks, per_chunk_size, C]
            x = x.view(B, num_chunks, -1, C)
            x = x * (1 + scale[:, :, None, :]) + shift[:, :, None, :]
            return x.view(B, -1, C)
        if scale.ndim == 2 and shift.ndim == 2:
            scale = scale[:, None, :]
            shift = shift[:, None, :]
        return x * (1 + scale) + shift


def apply_gate(x, gate=None, thw=None, cond_latents_num_frames=None, num_chunks=None):
    """Apply gate: out = x * gate; gate layouts as in `modulate`."""
    if gate is None:
        return x

    if thw is not None and len(thw) == 3:
        _, H, W = thw
        B, L, C = x.shape
        frame_tokens = H * W
        if get_sequence_parallel_state():
            gate_seq = _expand_frame_values_for_sp(gate, frame_tokens, cond_latents_num_frames)
            assert gate_seq.shape[1] == L, f'gate_seq.shape[1]: {gate_seq.shape[1]}, L: {L}'
            return x * gate_seq
        T = gate.shape[1]
        assert T * frame_tokens == L, f'T * H * W: {T * frame_tokens}, L: {L}'
        x_5d = x.reshape(B, T, H, W, C)
        gate_5d = gate[:, :, None, None, :]
        return (x_5d * gate_5d).reshape(B, L, C)

    elif num_chunks is not None and gate.shape[1] == num_chunks:
        B, L, C = x.shape
        x = x.view(B, num_chunks, -1, C)
        g = gate[:, :, None, :]
        return (x * g).view(B, -1, C)

    if gate.ndim == 2:
        gate = gate[:, None, :]
    return x * gate
