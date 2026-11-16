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
# Modified from diffusers v0.40.0, src/diffusers/models/autoencoders/autoencoder_kl_hunyuanvideo15.py
# (the decoder's module structure, re-expressed as streaming forwards over its own modules and weights)
# ==============================================================================
# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming (T-cache) decode of the HunyuanVideo-1.5 VAE, over diffusers' `AutoencoderKLHunyuanVideo15`.

The VAE itself -- weights, encoder, module classes -- is diffusers' (the model directory's ``vae/`` is in diffusers' layout).
diffusers decodes a whole clip at once, whose mid-block attention builds a dense ``(T*h*w)^2`` causal mask (gigabytes at 81
frames, hundreds at 721), or tiles it, which blends tiles and changes the output. MovieRoll decodes chunk by chunk with
temporal caches instead -- the last ``kernel_t - 1`` input frames of every causal conv, the projected K/V of the mid-block
attention, the first-frame flag of every temporal up-sampler -- which computes the same function as a whole-clip decode
(the algorithm of MovieRoll's own `streaming_vae.py`). This module is that algorithm, written against diffusers'
modules (their norms, activations, convolutions and parameters).

Optional row parallelism (MovieRoll's PS-VAE): every rank of a group decodes one band of latent rows, exchanging a halo
for the 3x3 convolutions, and an all-gather returns the whole chunk to every rank. It computes the same function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

# up to this many latent frames the mid-block attention keeps the whole history (exact); past it, the most recent ones
# (MovieRoll's DEFAULT_STREAMING_EXACT_ATTENTION_FRAMES: ~1 minute at 24 fps)
EXACT_ATTENTION_FRAMES = 361


# --------------------------------------------------------------------------- row sharding (PS-VAE)
def split_rows(num_rows: int, world_size: int) -> tuple[int, ...]:
    """Contiguous, as-even-as-possible split; the first ``num_rows % world_size`` ranks get one extra row."""
    if num_rows < world_size:
        raise ValueError(f"cannot shard {num_rows} latent rows over {world_size} ranks: every rank needs one row")
    base, extra = divmod(num_rows, world_size)
    return tuple(base + 1 if r < extra else base for r in range(world_size))


@dataclass(frozen=True)
class RowShard:
    group: dist.ProcessGroup
    rank: int
    global_ranks: tuple[int, ...]
    latent_rows: tuple[int, ...]

    @classmethod
    def create(cls, group: dist.ProcessGroup, num_latent_rows: int) -> "RowShard":
        rank = dist.get_rank(group)
        if rank < 0:
            raise RuntimeError("this process is not a member of the VAE parallel decode group")
        global_ranks = tuple(dist.get_process_group_ranks(group))
        return cls(group, rank, global_ranks, split_rows(num_latent_rows, len(global_ranks)))

    @property
    def world_size(self) -> int:
        return len(self.global_ranks)

    def local_rows(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != sum(self.latent_rows):
            raise RuntimeError(f"expected {sum(self.latent_rows)} latent rows to shard, got {x.shape[-2]}")
        start = sum(self.latent_rows[: self.rank])
        return x[..., start:start + self.latent_rows[self.rank], :].contiguous()

    def exchange_halo(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        """``x`` with ``pad`` rows of the neighbouring bands (replicate rows at the global edges) on each side."""
        if pad == 0:
            return x
        if x.shape[-2] < pad:
            raise RuntimeError(f"halo of {pad} rows exceeds the local band of {x.shape[-2]} rows")
        halo_shape = (*x.shape[:-2], pad, x.shape[-1])
        ops, top, bottom = [], None, None
        if self.rank > 0:
            peer = self.global_ranks[self.rank - 1]
            top = x.new_empty(halo_shape)
            ops.append(dist.P2POp(dist.isend, x[..., :pad, :].contiguous(), peer, self.group))
            ops.append(dist.P2POp(dist.irecv, top, peer, self.group))
        if self.rank < self.world_size - 1:
            peer = self.global_ranks[self.rank + 1]
            bottom = x.new_empty(halo_shape)
            ops.append(dist.P2POp(dist.isend, x[..., -pad:, :].contiguous(), peer, self.group))
            ops.append(dist.P2POp(dist.irecv, bottom, peer, self.group))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        if top is None:
            top = x[..., :1, :].expand(halo_shape)
        if bottom is None:
            bottom = x[..., -1:, :].expand(halo_shape)
        return torch.cat([top, x, bottom], dim=-2)

    def gather_rows(self, x: torch.Tensor) -> torch.Tensor:
        local = self.latent_rows[self.rank]
        scale, remainder = divmod(x.shape[-2], local)
        if remainder:
            raise RuntimeError(f"{x.shape[-2]} rows is not a multiple of the {local} local latent rows")
        max_rows = max(self.latent_rows) * scale
        if x.shape[-2] < max_rows:
            x = F.pad(x, (0, 0, 0, max_rows - x.shape[-2]))
        x = x.contiguous()
        parts = [torch.empty_like(x) for _ in range(self.world_size)]
        dist.all_gather(parts, x, group=self.group)
        return torch.cat([p[..., :n * scale, :] for p, n in zip(parts, self.latent_rows)], dim=-2)


# --------------------------------------------------------------------------- the stream
@dataclass
class _Stream:
    conv: dict = field(default_factory=dict)            # conv module id -> last kernel_t - 1 input frames
    attn: dict = field(default_factory=dict)            # attention module id -> (key, value, frames)
    first_done: set = field(default_factory=set)        # temporal up-samplers that emitted their first frame
    frames_seen: int = 0
    shard: Optional[RowShard] = None
    banded: bool = False                                # this rank holds only its band of rows (past the mid block)


class StreamingDecoder:
    """Chunked causal decode of a diffusers `AutoencoderKLHunyuanVideo15` (``vae``) with MovieRoll's T-caches."""

    def __init__(self, vae, exact_attention_frames: Optional[int] = EXACT_ATTENTION_FRAMES):
        self.vae = vae
        self.exact_attention_frames = exact_attention_frames
        self.parallel_group: Optional[dist.ProcessGroup] = None

    @property
    def temporal(self) -> int:
        return int(self.vae.config.temporal_compression_ratio)

    def enable_row_parallel(self, group: Optional[dist.ProcessGroup]) -> None:
        """Decode a band of latent rows on every rank of ``group`` (None or a one-rank group: no parallelism)."""
        self.parallel_group = group if group is not None and dist.get_world_size(group) > 1 else None

    # ------------------------------------------------------------------ modules
    def _conv(self, module, x: torch.Tensor, s: _Stream) -> torch.Tensor:
        padding = list(module.time_causal_padding)
        if s.banded:
            x = s.shard.exchange_halo(x, padding[2])
            padding[2] = padding[3] = 0
        kt_minus_1 = padding[4]
        history = s.conv.get(id(module))
        combined = x if history is None else torch.cat([history, x], dim=2)
        s.conv[id(module)] = combined[:, :, -kt_minus_1:].detach().contiguous() if kt_minus_1 > 0 else None
        if history is not None:
            padding[4] -= history.shape[2]
        return module.conv(F.pad(combined, padding, mode=module.pad_mode))

    def _resnet(self, module, x: torch.Tensor, s: _Stream) -> torch.Tensor:
        h = self._conv(module.conv1, module.nonlinearity(module.norm1(x)), s)
        h = self._conv(module.conv2, module.nonlinearity(module.norm2(h)), s)
        if module.conv_shortcut is not None:
            x = module.conv_shortcut(x)          # 1x1x1 conv, per position, no T-cache needed
        return x + h

    def _attention(self, module, x: torch.Tensor, s: _Stream) -> torch.Tensor:
        h_ = module.norm(x)
        q, k, v = module.to_q(h_), module.to_k(h_), module.to_v(h_)
        b, c, f, h, w = q.shape
        hw = h * w

        def tokens(t):
            return t.reshape(b, c, f * hw).permute(0, 2, 1).unsqueeze(1).contiguous()

        q, k_cur, v_cur = tokens(q), tokens(k), tokens(v)
        cached = s.attn.get(id(module))
        if cached is not None:
            past_k, past_v, past_frames = cached
            past_tokens = past_k.shape[-2]
            all_k, all_v = torch.cat([past_k, k_cur], dim=-2), torch.cat([past_v, v_cur], dim=-2)
        else:
            past_tokens, past_frames = 0, 0
            all_k, all_v = k_cur, v_cur
        # every past frame is visible; the current frames are causal per frame
        mask = torch.zeros((f * hw, past_tokens + f * hw), dtype=x.dtype, device=x.device)
        frame_ids = torch.arange(f, device=x.device).repeat_interleave(hw)
        mask[:, past_tokens:].masked_fill_(frame_ids[None, :] > frame_ids[:, None], float("-inf"))
        out = F.scaled_dot_product_attention(q, all_k, all_v, attn_mask=mask[None, None])
        out = out.squeeze(1).permute(0, 2, 1).reshape(b, c, f, h, w)
        limit = self.exact_attention_frames
        if limit is None:
            s.attn[id(module)] = (all_k.detach(), all_v.detach(), past_frames + f)
        else:
            keep = limit * hw
            s.attn[id(module)] = (all_k[:, :, -keep:].detach().contiguous(), all_v[:, :, -keep:].detach().contiguous(),
                                  min(past_frames + f, limit))
        return x + module.proj_out(out)

    def _upsample(self, module, x: torch.Tensor, s: _Stream) -> torch.Tensor:
        rearrange = module._dcae_upsample_rearrange
        h = self._conv(module.conv, x, s)
        if not module.add_temporal_upsample:
            shortcut = rearrange(x.repeat_interleave(repeats=module.repeats, dim=1), r1=1, r2=2, r3=2)
            return rearrange(h, r1=1, r2=2, r3=2) + shortcut
        h_parts, x_parts = [], []
        if id(module) not in s.first_done:
            h_first = rearrange(h[:, :, :1], r1=1, r2=2, r3=2)
            h_parts.append(h_first[:, : h_first.shape[1] // 2])
            x_parts.append(rearrange(x[:, :, :1], r1=1, r2=2, r3=2).repeat_interleave(repeats=module.repeats // 2, dim=1))
            h, x = h[:, :, 1:], x[:, :, 1:]
            s.first_done.add(id(module))
        if h.shape[2] > 0:
            h_parts.append(rearrange(h, r1=2, r2=2, r3=2))
            x_parts.append(rearrange(x, r1=2, r2=2, r3=2).repeat_interleave(repeats=module.repeats, dim=1))
        h_out = h_parts[0] if len(h_parts) == 1 else torch.cat(h_parts, dim=2)
        x_out = x_parts[0] if len(x_parts) == 1 else torch.cat(x_parts, dim=2)
        return h_out + x_out

    def _decode_chunk(self, z: torch.Tensor, s: _Stream) -> torch.Tensor:
        decoder = self.vae.decoder
        s.banded = False
        h = self._conv(decoder.conv_in, z, s) + z.repeat_interleave(repeats=decoder.repeat, dim=1)
        mid = decoder.mid_block
        h = self._resnet(mid.resnets[0], h, s)
        for attention, resnet in zip(mid.attentions, mid.resnets[1:]):
            if attention is not None:
                h = self._attention(attention, h, s)
            h = self._resnet(resnet, h, s)
        # the bottleneck attention is global over h*w, so everything above runs on the whole latent on every rank; from
        # here on a rank of a row-parallel decode only holds its band
        if s.shard is not None:
            h = s.shard.local_rows(h)
            s.banded = True
        for block in decoder.up_blocks:
            for resnet in block.resnets:
                h = self._resnet(resnet, h, s)
            if block.upsamplers is not None:
                for upsampler in block.upsamplers:
                    h = self._upsample(upsampler, h, s)
        h = decoder.conv_act(decoder.norm_out(h))
        h = self._conv(decoder.conv_out, h, s)
        if s.shard is not None:
            h = s.shard.gather_rows(h)
        return h

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, max_chunk_frames: int, dtype: torch.dtype, device: torch.device,
               on_chunk=None) -> torch.Tensor:
        """Decode scaled latents ``[1, C, T, h, w]`` chunk by chunk; returns pixels ``[1, 3, 4(T-1)+1, H, W]`` in [-1, 1]
        on the CPU, in the VAE's dtype. ``on_chunk(pixels)`` sees every decoded chunk."""
        vae = self.vae
        z = latents.to(device=device, dtype=dtype) / float(vae.config.scaling_factor)
        shift = getattr(vae.config, "shift_factor", None)
        if shift is not None:
            z = z + float(shift)
        pieces = (z,) if max_chunk_frames <= 0 else z.split(max_chunk_frames, dim=2)
        stream = _Stream()
        if self.parallel_group is not None:
            stream.shard = RowShard.create(self.parallel_group, z.shape[-2])
        decoded = []
        for piece in pieces:
            first = stream.frames_seen == 0
            pixels = self._decode_chunk(piece, stream)
            expected = (piece.shape[2] - 1) * self.temporal + 1 if first else piece.shape[2] * self.temporal
            if pixels.shape[2] != expected:
                raise RuntimeError(f"streaming decode produced {pixels.shape[2]} pixel frames, expected {expected}")
            stream.frames_seen += piece.shape[2]
            if on_chunk is not None:
                on_chunk(pixels)
            decoded.append(pixels.cpu())
            del pixels
        return torch.cat(decoded, dim=2)
