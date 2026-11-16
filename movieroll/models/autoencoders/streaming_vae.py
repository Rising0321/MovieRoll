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

"""Causal RMSNorm 3D VAE with T-cache streaming encode/decode, whose decoder is the paper's PS-VAE.

The non-streaming (``streaming_mode=False``) path matches ``causal_vae.py`` exactly: same parameters,
same numerics, same ``state_dict`` keys. On top of it sits an exact T-cache streaming path --
``encode(chunk, streaming_mode=True)`` / ``decode(chunk, streaming_mode=True)`` / ``clear_cache()``,
with ``decode_latent_chunks_streaming`` running one such stream over a whole video.

Streaming is inference-only (``torch.no_grad()`` + ``eval()``), holds one encode and one decode stream
per instance, and bypasses spatial/temporal tiling. Streaming decode can also be sharded over the latent
rows of a process group (``vae.enable_parallel_decode(group)``, see ``row_shard.py``), which computes the
same function as the single-device stream up to FP16 kernel rounding.

PS-VAE (Parallel Streaming VAE) is those two halves of the decode used together. Stateful chunk decoding: the
decoder takes latent chunks in order and carries its convolution caches, attention K/V caches and temporal
up-sampling first-frame flags in a ``DecoderStreamState``. Spatial parallelism: every rank runs ``conv_in`` and
the middle block on the whole chunk, then keeps one contiguous band of latent rows through the up-sampling blocks
and the output head (halo rows are exchanged for the convolutions, see ``row_shard.py``), and an all-gather hands
the complete decoded chunk to every rank.

Reference: [FLUX] https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch import Tensor, nn

from movieroll.models.autoencoders.causal_vae import (
    Conv3d,
    DecoderOutput,
    RMS_norm,
    prepare_causal_attention_mask,
    swish,
)
from movieroll.models.autoencoders.row_shard import RowShard
from movieroll.models.autoencoders.vae_loader import DiagonalGaussianDistribution

# Default latent-frame budget for EXACT streaming attention: up to this many latent frames the
# bottleneck attention keeps the full history, beyond it it becomes LOCAL, a sliding window over the most
# recent this-many frames. That window is a genuine approximation, since the bottleneck attention is
# near-uniform over history. 361 latent frames is ~1 minute at 24 fps with 4x temporal compression
# (1 + (60*24 - 1)/4), ~2.4 GB of fp16 K/V at 30x54 latent and 1024 channels. None means always exact.
DEFAULT_STREAMING_EXACT_ATTENTION_FRAMES = 361


# ---------------------------------------------------------------------------
# Streaming state (structured, non-persistent; never enters ``state_dict``)
# ---------------------------------------------------------------------------
@dataclass
class ConvTCache:
    """Last ``kernel_t - 1`` input frames of one CausalConv3d."""

    frames: Optional[torch.Tensor] = None


@dataclass
class AttentionTCache:
    """Projected K/V history of one AttnBlock (exact = full history)."""

    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    num_frames: int = 0


@dataclass
class TemporalStreamState:
    conv_caches: Dict[str, ConvTCache]
    attention_caches: Dict[str, AttentionTCache]
    first_frame_done: Dict[str, bool]
    input_frames_seen: int
    batch_size: int
    input_height: int
    input_width: int
    dtype: torch.dtype
    device: torch.device
    max_attention_cache_frames: Optional[int] = None


@dataclass
class EncoderStreamState(TemporalStreamState):
    pass


@dataclass
class DecoderStreamState(TemporalStreamState):
    row_shard: Optional[RowShard] = None


# ---------------------------------------------------------------------------
# One-time "streaming ignores tiling" warning (process global)
# ---------------------------------------------------------------------------
_STREAMING_TILING_WARNING_EMITTED = False


def _warn_streaming_ignores_tiling_once() -> None:
    global _STREAMING_TILING_WARNING_EMITTED
    if not _STREAMING_TILING_WARNING_EMITTED:
        warnings.warn(
            "VAE streaming_mode=True uses T-cache and ignores "
            "spatial/temporal tiling.",
            UserWarning,
            stacklevel=2,
        )
        _STREAMING_TILING_WARNING_EMITTED = True


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode='replicate',
        disable_causal=False,
        **kwargs
    ):
        super().__init__()

        self.pad_mode = pad_mode
        if disable_causal:
            padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2)
        else:
            padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0)  # W, H, T
        self.time_causal_padding = padding

        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        self._streaming_contract_validated = False

    def _validate_streaming_contract(self) -> None:
        if self.time_causal_padding[5] != 0:
            raise RuntimeError(
                "streaming requires causal temporal padding"
            )
        self._streaming_contract_validated = True

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)

    def forward_stream(
        self, x: Tensor, cache: ConvTCache, shard: Optional[RowShard] = None
    ) -> Tuple[Tensor, ConvTCache]:
        """Streaming forward: prepend cached input frames, emit new cache.

        Numerically identical to ``forward`` on the concatenation of all chunks.
        With ``shard`` the H padding is the neighbouring ranks' rows, and the cache
        keeps the haloed frames so history never has to be exchanged again.
        """
        if not self._streaming_contract_validated:
            self._validate_streaming_contract()

        padding = list(self.time_causal_padding)
        if shard is not None:
            x = shard.exchange_halo(x, padding[2])
            padding[2] = padding[3] = 0

        kt_minus_1 = padding[4]  # kernel_t - 1
        history = cache.frames
        if history is not None:
            if history.shape[0] != x.shape[0] or history.shape[1] != x.shape[1] \
                    or history.shape[3] != x.shape[3] or history.shape[4] != x.shape[4]:
                raise RuntimeError(
                    f"conv T-cache shape {tuple(history.shape)} incompatible with input {tuple(x.shape)}"
                )
            if history.dtype != x.dtype or history.device != x.device:
                raise RuntimeError("conv T-cache dtype/device mismatch with input")
            if history.shape[2] > kt_minus_1:
                raise RuntimeError("conv T-cache holds more frames than kernel_t - 1")
            combined = torch.cat([history, x], dim=2)
        else:
            combined = x

        # .contiguous() forces a small standalone copy: caching a slice *view* of
        # `combined` would otherwise pin the whole (large) conv-input tensor alive.
        new_history = combined[:, :, -kt_minus_1:].detach().contiguous() if kt_minus_1 > 0 else None

        if history is not None:
            padding[4] -= history.shape[2]
        conv_input = F.pad(combined, padding, mode=self.pad_mode)
        y = self.conv(conv_input)
        return y, ConvTCache(frames=new_history)


def prepare_streaming_attention_mask(
    n_frame_cur: int, past_tokens: int, n_hw: int, dtype, device
):
    """Rectangular mask ``[n_frame_cur*n_hw, past_tokens + n_frame_cur*n_hw]``.

    All ``past_tokens`` (strictly earlier frames) are visible to every current
    query; the current-vs-current block is causal per frame. Equivalent to the
    corresponding rows of ``prepare_causal_attention_mask`` over the full stream.
    """
    cur_tokens = n_frame_cur * n_hw
    total = past_tokens + cur_tokens
    mask = torch.zeros((cur_tokens, total), dtype=dtype, device=device)
    frame_ids = torch.arange(n_frame_cur, device=device).repeat_interleave(n_hw)
    disallow = frame_ids[None, :] > frame_ids[:, None]  # key frame later than query frame
    cur_block = mask[:, past_tokens:]
    cur_block.masked_fill_(disallow, float("-inf"))
    return mask


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = RMS_norm(in_channels, images=False)

        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        attention_mask = prepare_causal_attention_mask(f, h * w, h_.dtype, h_.device, batch_size=b)
        h_ = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.unsqueeze(1))

        return rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))

    def forward_stream(
        self, x: Tensor, cache: AttentionTCache, max_frames: Optional[int] = None
    ) -> Tuple[Tensor, AttentionTCache]:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        hw = h * w
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k_cur = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v_cur = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()

        if cache.key is not None:
            if cache.key.shape[-1] != c or cache.key.shape[-2] % hw != 0:
                raise RuntimeError("attention T-cache channel/HW mismatch with input")
            past_tokens = cache.key.shape[-2]
            all_k = torch.cat([cache.key, k_cur], dim=-2)
            all_v = torch.cat([cache.value, v_cur], dim=-2)
        else:
            past_tokens = 0
            all_k = k_cur
            all_v = v_cur

        mask = prepare_streaming_attention_mask(f, past_tokens, hw, x.dtype, x.device)
        out = nn.functional.scaled_dot_product_attention(
            q, all_k, all_v, attn_mask=mask.unsqueeze(0).unsqueeze(0)
        )
        out = rearrange(out, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

        if max_frames is None:
            new_key = all_k.detach()
            new_value = all_v.detach()
            new_num = cache.num_frames + f
        else:
            # keep only the most recent `max_frames` frames; .contiguous() copies the
            # slice so the dropped-frame storage of `all_k`/`all_v` is actually freed.
            keep = max_frames * hw
            new_key = all_k[:, :, -keep:, :].detach().contiguous()
            new_value = all_v[:, :, -keep:, :].detach().contiguous()
            new_num = min(cache.num_frames + f, max_frames)

        return x + self.proj_out(out), AttentionTCache(key=new_key, value=new_value, num_frames=new_num)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = RMS_norm(in_channels, images=False)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = RMS_norm(out_channels, images=False)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h

    def forward_stream(
        self, x: Tensor, state: TemporalStreamState, prefix: str, shard: Optional[RowShard] = None
    ) -> Tensor:
        h = self.norm1(x)
        h = swish(h)
        key1 = f"{prefix}.conv1"
        h, state.conv_caches[key1] = self.conv1.forward_stream(h, state.conv_caches[key1], shard)

        h = self.norm2(h)
        h = swish(h)
        key2 = f"{prefix}.conv2"
        h, state.conv_caches[key2] = self.conv2.forward_stream(h, state.conv_caches[key2], shard)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)  # 1x1x1 conv, per-position, no T-cache needed
        return x + h


class DownsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = CausalConv3d(in_channels, out_channels // factor, kernel_size=3)

        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        if self.add_temporal_downsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            h_first = torch.cat([h_first, h_first], dim=1)
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)
            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            B, C, T, H, W = x_first.shape
            x_first = x_first.view(B, h.shape[1], self.group_size // 2, T, H, W).mean(dim=2)

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(x_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = x_next.shape
            x_next = x_next.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape
            shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)

        return h + shortcut

    def _downsample_first(self, h_first_src: Tensor, x_first_src: Tensor):
        h_first = rearrange(h_first_src, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
        h_first = torch.cat([h_first, h_first], dim=1)
        x_first = rearrange(x_first_src, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
        B, C, T, H, W = x_first.shape
        x_first = x_first.view(B, h_first.shape[1], self.group_size // 2, T, H, W).mean(dim=2)
        return h_first, x_first

    def _downsample_rest(self, h_rest_src: Tensor, x_rest_src: Tensor):
        h_next = rearrange(h_rest_src, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=2, r2=2, r3=2)
        x_next = rearrange(x_rest_src, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=2, r2=2, r3=2)
        B, C, T, H, W = x_next.shape
        x_next = x_next.view(B, h_next.shape[1], self.group_size, T, H, W).mean(dim=2)
        return h_next, x_next

    def forward_stream(self, x: Tensor, state: TemporalStreamState, prefix: str) -> Tensor:
        conv_key = f"{prefix}.conv"
        h, state.conv_caches[conv_key] = self.conv.forward_stream(x, state.conv_caches[conv_key])

        if not self.add_temporal_downsample:
            # spatial-only: r1 == 1 keeps T; no first-frame phase.
            h_out = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=1, r2=2, r3=2)
            sc = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=1, r2=2, r3=2)
            B, C, T, H, W = sc.shape
            sc = sc.view(B, h_out.shape[1], self.group_size, T, H, W).mean(dim=2)
            return h_out + sc

        first_done = state.first_frame_done[prefix]
        h_parts, sc_parts = [], []
        if not first_done:
            hf, xf = self._downsample_first(h[:, :, :1], x[:, :, :1])
            h_parts.append(hf)
            sc_parts.append(xf)
            h_rest, x_rest = h[:, :, 1:], x[:, :, 1:]
            state.first_frame_done[prefix] = True
        else:
            h_rest, x_rest = h, x

        if h_rest.shape[2] > 0:
            if h_rest.shape[2] % 2 != 0:
                raise RuntimeError(
                    f"{prefix}: temporal downsample needs an even number of non-first frames, got {h_rest.shape[2]}"
                )
            hn, xn = self._downsample_rest(h_rest, x_rest)
            h_parts.append(hn)
            sc_parts.append(xn)

        h_out = h_parts[0] if len(h_parts) == 1 else torch.cat(h_parts, dim=2)
        sc = sc_parts[0] if len(sc_parts) == 1 else torch.cat(sc_parts, dim=2)
        return h_out + sc


class UpsampleDCAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = CausalConv3d(in_channels, out_channels * factor, kernel_size=3)

        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        if self.add_temporal_upsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            h_first = h_first[:, :h_first.shape[1]//2]

            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            x_first = x_first.repeat_interleave(repeats=self.repeats // 2, dim=1)

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(x_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            x_next = x_next.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = torch.cat([x_first, x_next], dim=2)

        else:
            h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        return h + shortcut

    def _upsample_first(self, h_first_src: Tensor, x_first_src: Tensor):
        h_first = rearrange(h_first_src, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
        h_first = h_first[:, :h_first.shape[1] // 2]
        x_first = rearrange(x_first_src, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
        x_first = x_first.repeat_interleave(repeats=self.repeats // 2, dim=1)
        return h_first, x_first

    def _upsample_rest(self, h_rest_src: Tensor, x_rest_src: Tensor):
        h_next = rearrange(h_rest_src, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=2, r2=2, r3=2)
        x_next = rearrange(x_rest_src, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=2, r2=2, r3=2)
        x_next = x_next.repeat_interleave(repeats=self.repeats, dim=1)
        return h_next, x_next

    def forward_stream(
        self, x: Tensor, state: TemporalStreamState, prefix: str, shard: Optional[RowShard] = None
    ) -> Tensor:
        conv_key = f"{prefix}.conv"
        h, state.conv_caches[conv_key] = self.conv.forward_stream(x, state.conv_caches[conv_key], shard)

        if not self.add_temporal_upsample:
            # spatial-only: r1 == 1 keeps T; no first-frame phase.
            h_out = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=1, r2=2, r3=2)
            sc = x.repeat_interleave(repeats=self.repeats, dim=1)
            sc = rearrange(sc, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=1, r2=2, r3=2)
            return h_out + sc

        first_done = state.first_frame_done[prefix]
        h_parts, sc_parts = [], []
        if not first_done:
            hf, xf = self._upsample_first(h[:, :, :1], x[:, :, :1])
            h_parts.append(hf)
            sc_parts.append(xf)
            h_rest, x_rest = h[:, :, 1:], x[:, :, 1:]
            state.first_frame_done[prefix] = True
        else:
            h_rest, x_rest = h, x

        if h_rest.shape[2] > 0:
            hn, xn = self._upsample_rest(h_rest, x_rest)
            h_parts.append(hn)
            sc_parts.append(xn)

        h_out = h_parts[0] if len(h_parts) == 1 else torch.cat(h_parts, dim=2)
        sc = sc_parts[0] if len(sc_parts) == 1 else torch.cat(sc_parts, dim=2)
        return h_out + sc


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(i_level >= np.log2(ffactor_spatial // ffactor_temporal))
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = DownsampleDCAE(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = RMS_norm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, 2 * z_channels, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h += shortcut
        return h

    def forward_stream(self, x: Tensor, state: EncoderStreamState) -> Tensor:
        # downsampling
        h, state.conv_caches["conv_in"] = self.conv_in.forward_stream(x, state.conv_caches["conv_in"])
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block].forward_stream(h, state, f"down.{i_level}.block.{i_block}")
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample.forward_stream(h, state, f"down.{i_level}.downsample")

        # middle
        h = self.mid.block_1.forward_stream(h, state, "mid.block_1")
        h, state.attention_caches["mid.attn_1"] = self.mid.attn_1.forward_stream(
            h, state.attention_caches["mid.attn_1"], state.max_attention_cache_frames
        )
        h = self.mid.block_2.forward_stream(h, state, "mid.block_2")

        # end
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
        h = self.norm_out(h)
        h = swish(h)
        h, state.conv_caches["conv_out"] = self.conv_out.forward_stream(h, state.conv_caches["conv_out"])
        h = h + shortcut
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # z to block_in
        block_in = block_out_channels[0]
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = UpsampleDCAE(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        # end
        self.norm_out = RMS_norm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3)

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        repeats = self.block_out_channels[0] // (self.z_channels)
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

    def forward_stream(self, z: Tensor, state: DecoderStreamState) -> Tensor:
        # z to block_in
        repeats = self.block_out_channels[0] // (self.z_channels)
        hc, state.conv_caches["conv_in"] = self.conv_in.forward_stream(z, state.conv_caches["conv_in"])
        h = hc + z.repeat_interleave(repeats=repeats, dim=1)

        # middle
        h = self.mid.block_1.forward_stream(h, state, "mid.block_1")
        h, state.attention_caches["mid.attn_1"] = self.mid.attn_1.forward_stream(
            h, state.attention_caches["mid.attn_1"], state.max_attention_cache_frames
        )
        h = self.mid.block_2.forward_stream(h, state, "mid.block_2")

        # The bottleneck attention is global over H*W, so everything above runs on the
        # full latent on every rank; from here on each rank only holds its row band.
        shard = state.row_shard
        if shard is not None:
            h = shard.local_rows(h)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block].forward_stream(h, state, f"up.{i_level}.block.{i_block}", shard)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample.forward_stream(h, state, f"up.{i_level}.upsample", shard)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h, state.conv_caches["conv_out"] = self.conv_out.forward_stream(h, state.conv_caches["conv_out"], shard)
        return h


def _collect_stream_keys(module: nn.Module):
    """Validate streaming-static contracts and collect stable cache keys."""
    conv_names, attn_names, phase_names = [], [], []
    for name, m in module.named_modules():
        if isinstance(m, CausalConv3d):
            try:
                m._validate_streaming_contract()
            except RuntimeError as exc:
                raise RuntimeError(f"{name}: {exc}") from exc
            conv_names.append(name)
        elif isinstance(m, AttnBlock):
            attn_names.append(name)
        elif isinstance(m, DownsampleDCAE) and m.add_temporal_downsample:
            phase_names.append(name)
        elif isinstance(m, UpsampleDCAE) and m.add_temporal_upsample:
            phase_names.append(name)
    return tuple(conv_names), tuple(attn_names), tuple(phase_names)


class AutoencoderKLConv3D(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float = None,
        shift_factor: Optional[float] = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            downsample_match_channel=downsample_match_channel,
        )
        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(reversed(block_out_channels)),
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            upsample_match_channel=upsample_match_channel,
        )
        # Module topology cannot change after construction, so the cache keys are validated and
        # collected once instead of per stream.
        self._encoder_stream_keys = _collect_stream_keys(self.encoder)
        self._decoder_stream_keys = _collect_stream_keys(self.decoder)

        self.use_slicing = False
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.25

        # ---- streaming state (plain attributes; never enter state_dict) ----
        # Attention history budget in LATENT frames, the same for encode and decode: at or below it the
        # attention is exact, above it local over the most recent N frames, and None is always exact.
        # configure_streaming_attention(...) sets it in seconds+fps or turns it off.
        self.streaming_max_attention_cache_frames: Optional[int] = DEFAULT_STREAMING_EXACT_ATTENTION_FRAMES
        self._streaming_encode_state: Optional[EncoderStreamState] = None
        self._streaming_decode_state: Optional[DecoderStreamState] = None
        self._parallel_decode_group: Optional[dist.ProcessGroup] = None

    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        self.enable_spatial_tiling(use_tiling)

    def disable_tiling(self):
        self.disable_spatial_tiling()

    def enable_slicing(self):
        self.use_slicing = True

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (x / blend_extent)
        return b

    def spatial_tiled_encode(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))  # 256 * (1 - 0.25) = 192
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        moments = torch.cat(result_rows, dim=-2)
        return moments

    def temporal_tiled_encode(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))  # 64 * (1 - 0.25) = 48
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_latent_min_tsize - blend_extent

        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i: i + self.tile_sample_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size):
                tile = self.spatial_tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, :t_limit + 1, :, :])
        moments = torch.cat(result_row, dim=-3)
        return moments

    def spatial_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)  # 256 * 0.25 = 64
        row_limit = self.tile_sample_min_size - blend_extent  # 256 - 64 = 192

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[:, :, :, i: i + self.tile_latent_min_size, j: j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=-2)
        return dec

    def temporal_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)  # 64 * 0.25 = 16
        t_limit = self.tile_sample_min_tsize - blend_extent  # 64 - 16 = 48
        assert 0 < overlap_size < self.tile_latent_min_tsize

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i: i + self.tile_latent_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size):
                decoded = self.spatial_tiled_decode(tile)
            else:
                decoded = self.decoder(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, :t_limit + 1, :, :])
        dec = torch.cat(result_row, dim=-3)
        return dec

    def encode(self, x: Tensor, return_dict: bool = True, *, streaming_mode: bool = False):
        if streaming_mode:
            return self._encode_stream_public(x, return_dict)

        def _encode(x):
            if self.use_temporal_tiling and x.shape[-3] > self.tile_sample_min_tsize:
                return self.temporal_tiled_encode(x)
            if self.use_spatial_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
                return self.spatial_tiled_encode(x)
            return self.encoder(x)

        assert len(x.shape) == 5  # (B, C, T, H, W)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [_encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = _encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: Tensor, return_dict: bool = True, *, streaming_mode: bool = False):
        if streaming_mode:
            return self._decode_stream_public(z, return_dict)

        def _decode(z):
            if self.use_temporal_tiling and z.shape[-3] > self.tile_latent_min_tsize:
                return self.temporal_tiled_decode(z)
            if self.use_spatial_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
                return self.spatial_tiled_decode(z)
            return self.decoder(z)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    # -------------------------- streaming API --------------------------
    def clear_cache(self) -> None:
        """Release both the encode and decode streaming state."""
        self._streaming_encode_state = None
        self._streaming_decode_state = None

    def configure_streaming_attention(
        self,
        *,
        latent_frames: Optional[int] = None,
        seconds: Optional[float] = None,
        fps: Optional[float] = None,
        exact: bool = False,
    ) -> "AutoencoderKLConv3D":
        """Set the exact-attention budget, in latent frames, for both encode and decode streams.

        ``exact=True`` never switches to local attention; ``latent_frames=N`` sets N directly;
        ``seconds=S, fps=F`` sets N = round(1 + (S * F - 1) / ffactor_temporal), so 60 s at 24 fps is 361.
        The value is read when a stream is created, so call this before the first chunk.
        """
        if exact:
            self.streaming_max_attention_cache_frames = None
            return self
        if latent_frames is not None:
            n = int(latent_frames)
        elif seconds is not None:
            if fps is None:
                raise ValueError("configure_streaming_attention(seconds=...) also requires fps=")
            # pixel frames -> latent frames: the first pixel frame is a standalone latent,
            # then every ffactor_temporal pixels -> 1 latent, i.e. 1 + (pixels - 1)/ft.
            pixels = float(seconds) * float(fps)
            n = int(round(1.0 + (pixels - 1.0) / self.ffactor_temporal))
        else:
            raise ValueError("pass exact=True, latent_frames=N, or seconds=S with fps=F")
        if n <= 0:
            raise ValueError(f"attention budget must be a positive number of latent frames, got {n}")
        self.streaming_max_attention_cache_frames = n
        return self

    @property
    def parallel_decode_group(self) -> Optional[dist.ProcessGroup]:
        """Group that ``decode(..., streaming_mode=True)`` is sharded over, or None for single-device."""
        return self._parallel_decode_group

    def enable_parallel_decode(self, group: Optional[dist.ProcessGroup] = None) -> "AutoencoderKLConv3D":
        """Shard streaming decode over the latent rows of ``group`` (None = WORLD).

        The group is independent of the one the DiT uses. Every rank of ``group``
        must call ``decode(..., streaming_mode=True)`` in lockstep with identical
        latent chunks, and each one gets the full decoded chunk back. Encode and
        ``streaming_mode=False`` decode stay single-device. A one-rank group is the
        same as no parallel decode.
        """
        self._require_no_decode_stream("changing the parallel decode group")
        if not dist.is_initialized():
            raise RuntimeError("enable_parallel_decode requires an initialized torch.distributed process group")
        group = dist.group.WORLD if group is None else group
        if dist.get_rank(group) < 0:
            raise RuntimeError("this process is not a member of the requested VAE parallel decode group")
        self._parallel_decode_group = group if dist.get_world_size(group) > 1 else None
        return self

    def disable_parallel_decode(self) -> "AutoencoderKLConv3D":
        self._require_no_decode_stream("disabling parallel decode")
        self._parallel_decode_group = None
        return self

    def _require_no_decode_stream(self, action: str) -> None:
        if self._streaming_decode_state is not None:
            raise RuntimeError(f"call clear_cache() to end the active decode stream before {action}")

    def _assert_streaming_ok(self) -> None:
        if torch.is_grad_enabled():
            raise RuntimeError("VAE streaming requires torch.no_grad()")
        if self.training:
            raise RuntimeError("VAE streaming requires eval() mode")

    def _warn_if_tiling(self) -> None:
        if self.use_spatial_tiling or self.use_temporal_tiling:
            _warn_streaming_ignores_tiling_once()

    def _create_stream_state(self, stream_keys, x, cls):
        conv_names, attn_names, phase_names = stream_keys
        max_attention_cache_frames = self.streaming_max_attention_cache_frames
        if (
            max_attention_cache_frames is not None
            and max_attention_cache_frames <= 0
        ):
            raise ValueError(
                "max_attention_cache_frames must be None or a positive integer"
            )
        B, C, T, H, W = x.shape
        return cls(
            conv_caches={n: ConvTCache() for n in conv_names},
            attention_caches={n: AttentionTCache() for n in attn_names},
            first_frame_done=dict.fromkeys(phase_names, False),
            input_frames_seen=0,
            batch_size=B,
            input_height=H,
            input_width=W,
            dtype=x.dtype,
            device=x.device,
            max_attention_cache_frames=max_attention_cache_frames,
        )

    @staticmethod
    def _validate_chunk_metadata(x, state, kind: str) -> None:
        if x.ndim != 5:
            raise ValueError(f"streaming {kind} expects a 5D (B,C,T,H,W) tensor, got {tuple(x.shape)}")
        B, C, T, H, W = x.shape
        if T == 0:
            raise ValueError(f"streaming {kind} received an empty chunk (T=0)")
        if B != state.batch_size or H != state.input_height or W != state.input_width:
            raise RuntimeError(
                f"streaming {kind} metadata changed: got (B={B},H={H},W={W}) but stream is "
                f"(B={state.batch_size},H={state.input_height},W={state.input_width}); call clear_cache() first"
            )
        if x.dtype != state.dtype:
            raise RuntimeError(f"streaming {kind} dtype changed: {x.dtype} vs stream {state.dtype}")
        if x.device != state.device:
            raise RuntimeError(f"streaming {kind} device changed: {x.device} vs stream {state.device}")
    def _encode_stream_public(self, x: Tensor, return_dict: bool):
        self._assert_streaming_ok()

        state = self._streaming_encode_state
        if state is None:
            if x.ndim != 5:
                raise ValueError(f"streaming encode expects a 5D tensor, got {tuple(x.shape)}")
            self._warn_if_tiling()
            state = self._create_stream_state(
                self._encoder_stream_keys, x, EncoderStreamState
            )
            self._streaming_encode_state = state

        self._validate_chunk_metadata(x, state, "encode")

        T_in = x.shape[2]
        ft = self.ffactor_temporal
        is_first = state.input_frames_seen == 0
        if is_first:
            if (T_in - 1) % ft != 0:
                raise ValueError(f"first streaming encode chunk must have T = 1 + {ft}n, got T={T_in}")
            expected = (T_in - 1) // ft + 1
        else:
            if T_in % ft != 0:
                raise ValueError(f"continuation streaming encode chunk must have T = {ft}n, got T={T_in}")
            expected = T_in // ft

        moments = self.encoder.forward_stream(x, state)
        if moments.shape[2] != expected:
            raise RuntimeError(
                f"streaming encode produced {moments.shape[2]} latent frames, expected {expected} (T_in={T_in})"
            )
        state.input_frames_seen += T_in

        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode_stream_public(self, z: Tensor, return_dict: bool):
        self._assert_streaming_ok()

        state = self._streaming_decode_state
        if state is None:
            if z.ndim != 5:
                raise ValueError(f"streaming decode expects a 5D tensor, got {tuple(z.shape)}")
            self._warn_if_tiling()
            state = self._create_stream_state(
                self._decoder_stream_keys, z, DecoderStreamState
            )
            if self._parallel_decode_group is not None:
                state.row_shard = RowShard.create(self._parallel_decode_group, z.shape[-2])
            self._streaming_decode_state = state

        self._validate_chunk_metadata(z, state, "decode")

        T_in = z.shape[2]
        ft = self.ffactor_temporal
        is_first = state.input_frames_seen == 0
        expected = (T_in - 1) * ft + 1 if is_first else T_in * ft

        decoded = self.decoder.forward_stream(z, state)
        if state.row_shard is not None:
            decoded = state.row_shard.gather_rows(decoded)
        if decoded.shape[2] != expected:
            raise RuntimeError(
                f"streaming decode produced {decoded.shape[2]} pixel frames, expected {expected} (T_in={T_in})"
            )
        state.input_frames_seen += T_in

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)


def decode_latent_chunks_streaming(
    vae, latent_chunks, *, vae_dtype: torch.dtype, device: torch.device
) -> list[Tensor]:
    """Streaming-decode ``latent_chunks`` in order and return the pixel chunks on the CPU.

    The T-cache is cleared before the first chunk and again afterwards, including when a decode raises, so
    the next stream starts clean. The latents arrive already unscaled, cast to ``vae_dtype`` on ``device``.
    """
    decoded_chunks = []
    vae.clear_cache()
    try:
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=vae_dtype,
            enabled=device.type == "cuda" and vae_dtype != torch.float32,
        ):
            for latent_chunk in latent_chunks:
                pixel_chunk = vae.decode(
                    latent_chunk,
                    return_dict=False,
                    streaming_mode=True,
                )[0]
                decoded_chunks.append(pixel_chunk.cpu())
                del pixel_chunk
    finally:
        vae.clear_cache()
    return decoded_chunks
