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
# The streaming forwards (`_stream_conv`, `_stream_attention`, `_stream_resnet`, `_stream_upsample`,
# `stream_decode_chunk`) follow the decoder modules of diffusers v0.37.0,
# src/diffusers/models/autoencoders/autoencoder_kl_hunyuanvideo15.py, re-expressed over their own modules and weights.
#
# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

"""The HunyuanVideo-1.5 VAE as an SGLang component, decoding in causal temporal chunks (PS-VAE).

The modules are diffusers' (`AutoencoderKLHunyuanVideo15`'s encoder and decoder, which compute MovieRoll's VAE
to float rounding), wrapped in the class
SGLang's loader and `DecodingStage` expect (`ParallelTiledVAE`: ``_encode`` / ``_decode``, a scaling factor,
compression ratios, tiling switches). The class is registered under the diffusers class name, so the ``vae/``
folder of a model directory is an ordinary diffusers component.

What is MovieRoll's own, and ported here from `movieroll.models.autoencoders.streaming_vae`, is the decode:

* **temporal streaming**: the decoder takes the latent chunks of a video in order and carries its convolution
  caches (the last ``kernel_t - 1`` input frames of every causal convolution), the key / value history of the
  bottleneck attention and the "first frame seen" flag of every temporal up-sampler. Decoding a video in chunks
  this way computes the same function as decoding it in one piece, which is what makes it a memory optimisation
  rather than a different model; it is also what lets a 30 s video be decoded at all on one GPU.
* **row parallelism** (``--vae-sp``): every rank runs the convolution into the bottleneck and the middle block
  on the whole chunk, then keeps one contiguous band of latent rows through the up-sampling blocks and the
  output head (halo rows are exchanged for the spatial convolutions, see `row_shard`), and an all-gather hands
  the complete decoded chunk to every rank.

The streaming driver reads the diffusers modules' parameters and runs the same math with the caches added; it
does not modify diffusers, and the one-shot path is diffusers' own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import (
    HunyuanVideo15AttnBlock,
    HunyuanVideo15CausalConv3d,
    HunyuanVideo15Decoder3D,
    HunyuanVideo15Encoder3D,
    HunyuanVideo15Upsample,
)
from sglang.multimodal_gen.configs.models.vaes.base import VAEConfig
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from sglang_movieroll.runtime.models.vaes.row_shard import RowShard

logger = init_logger(__name__)


@dataclass
class ConvCache:
    """Last ``kernel_t - 1`` input frames of one causal convolution."""

    frames: Optional[torch.Tensor] = None


@dataclass
class AttnCache:
    """Projected key / value history of the bottleneck attention."""

    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    num_frames: int = 0


@dataclass
class DecodeStream:
    conv: Dict[str, ConvCache] = field(default_factory=dict)
    attn: Dict[str, AttnCache] = field(default_factory=dict)
    first_frame_done: Dict[str, bool] = field(default_factory=dict)
    frames_seen: int = 0
    row_shard: Optional[RowShard] = None
    max_attention_frames: Optional[int] = None


def _stream_keys(decoder: torch.nn.Module):
    """Stable cache keys of every causal convolution, bottleneck attention and temporal up-sampler."""
    conv, attn, phase = [], [], []
    for name, module in decoder.named_modules():
        if isinstance(module, HunyuanVideo15CausalConv3d):
            conv.append(name)
        elif isinstance(module, HunyuanVideo15AttnBlock):
            attn.append(name)
        elif (
            isinstance(module, HunyuanVideo15Upsample) and module.add_temporal_upsample
        ):
            phase.append(name)
    return conv, attn, phase


def _stream_conv(
    module: HunyuanVideo15CausalConv3d,
    x: torch.Tensor,
    cache: ConvCache,
    shard: Optional[RowShard],
):
    """A causal convolution over one chunk: the cached input frames are prepended instead of padded."""
    padding = list(module.time_causal_padding)
    if padding[5] != 0:
        raise RuntimeError("streaming requires causal temporal padding")
    if shard is not None:
        x = shard.exchange_halo(x, padding[2])
        padding[2] = padding[3] = 0
    kernel_t_minus_1 = padding[4]
    history = cache.frames
    if history is not None:
        combined = torch.cat([history, x], dim=2)
        padding[4] -= history.shape[2]
    else:
        combined = x
    # .contiguous() forces a small standalone copy: caching a slice view of `combined` would otherwise pin the
    # whole convolution input alive
    new_history = (
        combined[:, :, -kernel_t_minus_1:].detach().contiguous()
        if kernel_t_minus_1 > 0
        else None
    )
    y = module.conv(F.pad(combined, padding, mode=module.pad_mode))
    return y, ConvCache(frames=new_history)


def _streaming_attention_mask(frames: int, past_tokens: int, hw: int, dtype, device):
    """Every strictly earlier frame is visible to every current query; the current block is causal per frame."""
    current = frames * hw
    mask = torch.zeros((current, past_tokens + current), dtype=dtype, device=device)
    frame_ids = torch.arange(frames, device=device).repeat_interleave(hw)
    mask[:, past_tokens:].masked_fill_(
        frame_ids[None, :] > frame_ids[:, None], float("-inf")
    )
    return mask


def _stream_attention(
    module: HunyuanVideo15AttnBlock,
    x: torch.Tensor,
    cache: AttnCache,
    max_frames: Optional[int],
):
    """The bottleneck attention over one chunk, attending to the key / value history of the earlier chunks."""
    identity = x
    h = module.norm(x)
    query, key, value = module.to_q(h), module.to_k(h), module.to_v(h)
    batch, channels, frames, height, width = query.shape
    hw = height * width

    def flat(t):
        return (
            t.reshape(batch, channels, frames * hw)
            .permute(0, 2, 1)
            .unsqueeze(1)
            .contiguous()
        )

    query, key_cur, value_cur = flat(query), flat(key), flat(value)
    if cache.key is not None:
        past_tokens = cache.key.shape[-2]
        all_key = torch.cat([cache.key, key_cur], dim=-2)
        all_value = torch.cat([cache.value, value_cur], dim=-2)
    else:
        past_tokens = 0
        all_key, all_value = key_cur, value_cur

    mask = _streaming_attention_mask(frames, past_tokens, hw, x.dtype, x.device)
    out = F.scaled_dot_product_attention(
        query, all_key, all_value, attn_mask=mask.unsqueeze(0).unsqueeze(0)
    )
    out = (
        out.squeeze(1)
        .reshape(batch, frames, height, width, channels)
        .permute(0, 4, 1, 2, 3)
    )

    if max_frames is None:
        new_key, new_value = all_key.detach(), all_value.detach()
        new_num = cache.num_frames + frames
    else:
        keep = max_frames * hw
        new_key = all_key[:, :, -keep:, :].detach().contiguous()
        new_value = all_value[:, :, -keep:, :].detach().contiguous()
        new_num = min(cache.num_frames + frames, max_frames)
    return identity + module.proj_out(out), AttnCache(
        key=new_key, value=new_value, num_frames=new_num
    )


def _stream_resnet(
    module, x: torch.Tensor, state: DecodeStream, prefix: str, shard: Optional[RowShard]
):
    h = module.nonlinearity(module.norm1(x))
    h, state.conv[f"{prefix}.conv1"] = _stream_conv(
        module.conv1, h, state.conv[f"{prefix}.conv1"], shard
    )
    h = module.nonlinearity(module.norm2(h))
    h, state.conv[f"{prefix}.conv2"] = _stream_conv(
        module.conv2, h, state.conv[f"{prefix}.conv2"], shard
    )
    residual = (
        module.conv_shortcut(x) if module.conv_shortcut is not None else x
    )  # 1x1x1, no cache needed
    return h + residual


def _stream_upsample(
    module: HunyuanVideo15Upsample,
    x: torch.Tensor,
    state: DecodeStream,
    prefix: str,
    shard: Optional[RowShard],
):
    """A DCAE up-sampler over one chunk: the very first frame of the stream is up-sampled on its own."""
    key = f"{prefix}.conv"
    h, state.conv[key] = _stream_conv(module.conv, x, state.conv[key], shard)
    rearrange = module._dcae_upsample_rearrange

    if not module.add_temporal_upsample:
        out = rearrange(h, r1=1, r2=2, r3=2)
        shortcut = rearrange(
            x.repeat_interleave(repeats=module.repeats, dim=1), r1=1, r2=2, r3=2
        )
        return out + shortcut

    parts, shortcuts = [], []
    if not state.first_frame_done[prefix]:
        first = rearrange(h[:, :, :1], r1=1, r2=2, r3=2)
        parts.append(first[:, : first.shape[1] // 2])
        shortcuts.append(
            rearrange(x[:, :, :1], r1=1, r2=2, r3=2).repeat_interleave(
                module.repeats // 2, dim=1
            )
        )
        h_rest, x_rest = h[:, :, 1:], x[:, :, 1:]
        state.first_frame_done[prefix] = True
    else:
        h_rest, x_rest = h, x

    if h_rest.shape[2] > 0:
        parts.append(rearrange(h_rest, r1=2, r2=2, r3=2))
        shortcuts.append(
            rearrange(x_rest, r1=2, r2=2, r3=2).repeat_interleave(module.repeats, dim=1)
        )

    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)
    shortcut = shortcuts[0] if len(shortcuts) == 1 else torch.cat(shortcuts, dim=2)
    return out + shortcut


def stream_decode_chunk(
    decoder: HunyuanVideo15Decoder3D, z: torch.Tensor, state: DecodeStream
) -> torch.Tensor:
    """Decode one latent chunk, carrying ``state`` from the previous chunks of the same video."""
    h, state.conv["conv_in"] = _stream_conv(
        decoder.conv_in, z, state.conv["conv_in"], None
    )
    h = h + z.repeat_interleave(repeats=decoder.repeat, dim=1)

    mid = decoder.mid_block
    h = _stream_resnet(mid.resnets[0], h, state, "mid_block.resnets.0", None)
    for index, attention in enumerate(mid.attentions):
        if attention is not None:
            key = f"mid_block.attentions.{index}"
            h, state.attn[key] = _stream_attention(
                attention, h, state.attn[key], state.max_attention_frames
            )
        h = _stream_resnet(
            mid.resnets[index + 1], h, state, f"mid_block.resnets.{index + 1}", None
        )

    # the bottleneck attention is global over the rows, so everything above runs on the whole latent on every
    # rank; from here on each rank only holds its band
    shard = state.row_shard
    if shard is not None:
        h = shard.local_rows(h)

    for level, up_block in enumerate(decoder.up_blocks):
        for index, resnet in enumerate(up_block.resnets):
            h = _stream_resnet(
                resnet, h, state, f"up_blocks.{level}.resnets.{index}", shard
            )
        if up_block.upsamplers is not None:
            for index, upsampler in enumerate(up_block.upsamplers):
                h = _stream_upsample(
                    upsampler, h, state, f"up_blocks.{level}.upsamplers.{index}", shard
                )

    h = decoder.conv_act(decoder.norm_out(h))
    h, state.conv["conv_out"] = _stream_conv(
        decoder.conv_out, h, state.conv["conv_out"], shard
    )
    return h


class AutoencoderKLHunyuanVideo15(ParallelTiledVAE):
    """The HunyuanVideo-1.5 VAE: diffusers' modules, SGLang's component contract, MovieRoll's streaming decode."""

    def __init__(self, config: VAEConfig, **kwargs) -> None:
        super().__init__(config)
        arch = config.arch_config
        block_out_channels = tuple(arch.block_out_channels)
        self.encoder = HunyuanVideo15Encoder3D(
            in_channels=arch.in_channels,
            out_channels=arch.latent_channels * 2,
            block_out_channels=block_out_channels,
            layers_per_block=arch.layers_per_block,
            temporal_compression_ratio=arch.temporal_compression_ratio,
            spatial_compression_ratio=arch.spatial_compression_ratio,
            downsample_match_channel=arch.downsample_match_channel,
        )
        self.decoder = HunyuanVideo15Decoder3D(
            in_channels=arch.latent_channels,
            out_channels=arch.out_channels,
            block_out_channels=tuple(reversed(block_out_channels)),
            layers_per_block=arch.layers_per_block,
            temporal_compression_ratio=arch.temporal_compression_ratio,
            spatial_compression_ratio=arch.spatial_compression_ratio,
            upsample_match_channel=arch.upsample_match_channel,
        )
        # how many latent frames one streaming decode step takes, and the group its rows are sharded over
        self.stream_chunk_frames: int = int(getattr(arch, "stream_chunk_frames", 4))
        self.max_attention_frames: Optional[int] = getattr(
            arch, "max_attention_frames", None
        )
        self.parallel_decode_group = None
        self._conv_keys, self._attn_keys, self._phase_keys = _stream_keys(self.decoder)

    def enable_parallel_decode(self, group) -> "AutoencoderKLHunyuanVideo15":
        """Shard the streaming decode over the latent rows of ``group`` (one rank means no sharding)."""
        import torch.distributed as dist

        self.parallel_decode_group = (
            group if (group is not None and dist.get_world_size(group) > 1) else None
        )
        return self

    # ------------------------------------------------------------------ the SGLang component contract
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a whole video by streaming over chunks of ``stream_chunk_frames`` latent frames."""
        chunk = self.stream_chunk_frames
        pieces = (z,) if chunk <= 0 else z.split(chunk, dim=2)
        state = DecodeStream(
            conv={name: ConvCache() for name in self._conv_keys},
            attn={name: AttnCache() for name in self._attn_keys},
            first_frame_done=dict.fromkeys(self._phase_keys, False),
            max_attention_frames=self.max_attention_frames,
        )
        if self.parallel_decode_group is not None:
            state.row_shard = RowShard.create(self.parallel_decode_group, z.shape[-2])
        decoded: List[torch.Tensor] = []
        for piece in pieces:
            out = stream_decode_chunk(self.decoder, piece, state)
            if state.row_shard is not None:
                out = state.row_shard.gather_rows(out)
            expected = (
                (piece.shape[2] - 1) * self.temporal_compression_ratio + 1
                if state.frames_seen == 0
                else piece.shape[2] * self.temporal_compression_ratio
            )
            if out.shape[2] != expected:
                raise RuntimeError(
                    f"streaming decode produced {out.shape[2]} pixel frames, expected {expected} "
                    f"(latent frames {piece.shape[2]}, already decoded {state.frames_seen})"
                )
            state.frames_seen += piece.shape[2]
            decoded.append(out)
        return torch.cat(decoded, dim=2)


EntryClass = AutoencoderKLHunyuanVideo15
