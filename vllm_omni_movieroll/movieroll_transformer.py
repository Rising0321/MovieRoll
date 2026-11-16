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
# Portions adapted from vLLM-Omni v0.32.0 (commit 5917730a),
# vllm_omni/diffusion/models/hunyuan_video/hunyuan_video_15_transformer.py
# (the q/k/v projection and normalisation of `HunyuanVideo15Attention.forward`, the stacked-shard loop of `load_weights`)
# ==============================================================================
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""MovieRoll's streaming MMDiT, built from vLLM-Omni's HunyuanVideo-1.5 transformer parts.

The backbone is HunyuanVideo-1.5's: patch embedding, text token refiner, byT5 projection, timestep embedding, 54 double-stream
blocks whose attention is `HunyuanVideo15Attention` (tensor-parallel q/k/v and output projections, q/k RMSNorm, the shared
rotary op), and a final projection. All of those come from `vllm_omni.diffusion.models.hunyuan_video`. MovieRoll adds:

* per-frame AdaLN: every latent frame has its own timestep, so the shift / scale / gate of the image stream are broadcast
  over the tokens of their own frame (the frames of a rolling window sit at different noise levels);
* a clean-history KV cache: the image queries of a window also attend to the cached keys / values (pre-RoPE, re-positioned
  on every use), while the text queries do not see the history;
* KV recompute: the selected chunks are re-encoded at timestep 0 with block-wise causal attention into a fresh cache;
* HieraKV: the first ``full_history_block_num`` blocks keep the whole history, the others only the sink and recent chunks.

The model has one entry point, ``forward(..., mode=...)`` with ``mode`` in ``denoise`` / ``encode`` / ``recompute``, and
calls every block as ``block(...)``: FSDP2 (HSDP), offload and compile hooks therefore see every use. Sequence parallelism
is model-owned (`parallel.SequenceLayout`, like vLLM-Omni's window-SP models): the attention layers are created with
``skip_sequence_parallel=True`` and the exchanges happen around the cache. Batch size is 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp_utils import is_transformer_block_module
from vllm_omni.diffusion.models.flux.flux_transformer import FeedForward
from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
    HunyuanVideo15Attention,
    HunyuanVideo15ByT5TextProjection,
    HunyuanVideo15PatchEmbed,
    HunyuanVideo15RotaryPosEmbed,
    HunyuanVideo15TimeEmbedding,
    HunyuanVideo15TokenRefiner,
)

from vllm_omni_movieroll.kv_cache import KVCache
from vllm_omni_movieroll.parallel import SequenceLayout, ring_attention
from vllm_omni_movieroll.rolling import TextTokens
from vllm_omni_movieroll.weights import FusedShardTracker, map_name

logger = init_logger(__name__)

MODES = ("denoise", "encode", "recompute")


class MovieRollTimeEmbedding(HunyuanVideo15TimeEmbedding):
    """HunyuanVideo-1.5's timestep embedding for per-frame fp32 timesteps: the sinusoid in fp32, cast to the MLP's dtype.

    (HunyuanVideo-1.5 casts the sinusoid to the timestep's dtype, which with fp32 per-frame timesteps and bf16 weights is a
    dtype mismatch, and bf16 timesteps would round the timestep itself.)
    """

    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim, use_meanflow=False)

    def forward(self, timestep: torch.Tensor, timestep_r: Optional[torch.Tensor] = None) -> torch.Tensor:
        projected = self.time_proj(timestep.float().flatten())
        return self.timestep_embedder(projected.to(self.timestep_embedder.linear_1.weight.dtype))


class _Modulation(nn.Module):
    """``SiLU -> Linear``: the modulation of an AdaLN (the ``norm1.linear`` of HunyuanVideo-1.5's ``AdaLayerNormZero``),
    applied per latent frame."""

    def __init__(self, dim: int, chunks: int):
        super().__init__()
        self.linear = nn.Linear(dim, chunks * dim)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.linear(F.silu(emb))


def _layer_norm(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6)


class FrameLayout:
    """Where the image tokens of a window sit: ``frame_tokens`` per latent frame, this rank holding ``[start, end)``.

    Without sequence parallelism the range is the whole window and per-frame values broadcast through a reshape; with it
    each local token looks up the value of its own frame.
    """

    def __init__(self, frames: int, frame_tokens: int, layout: SequenceLayout):
        self.frames = frames
        self.frame_tokens = frame_tokens
        self.tokens = frames * frame_tokens
        self.layout = layout
        self.start, self.end = layout.local_range(self.tokens) if layout.enabled else (0, self.tokens)

    def expand(self, values: torch.Tensor) -> torch.Tensor:
        """``[B, T, C]`` per-frame values -> ``[B, local tokens, C]``, in the dtype of ``values``."""
        if not self.layout.enabled:
            return values.repeat_interleave(self.frame_tokens, dim=1)
        positions = torch.arange(self.start, self.end, device=values.device)
        # shard padding (past the last token) takes the last frame's values; its output is discarded
        return values.index_select(1, (positions // self.frame_tokens).clamp(max=self.frames - 1))


def _modulate_frames(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, layout: FrameLayout) -> torch.Tensor:
    """``LayerNorm(x) * (1 + scale) + shift`` with the shift / scale of each token's own frame."""
    return _layer_norm(x) * (1 + layout.expand(scale)) + layout.expand(shift)


def _gate_frames(x: torch.Tensor, gate: torch.Tensor, layout: FrameLayout) -> torch.Tensor:
    return x * layout.expand(gate)


def _modulate_text(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The text stream has one timestep for the whole prompt: ``vec_txt`` is ``[B, 1, C]``."""
    return _layer_norm(x) * (1 + scale) + shift


class RopeTable:
    """Half-dim cos / sin rows of the ``(max_num_latent_frames, h, w)`` position grid (HunyuanVideo-1.5's rotary embedding).

    At inference the cached history is re-packed from frame 0, so positions are always a dense prefix of that grid: the
    window's rows sit right behind the cache. Built once per (h, w, device).
    """

    def __init__(self, rope_axes_dim, theta: float, max_num_latent_frames: int):
        self.embed = HunyuanVideo15RotaryPosEmbed(1, 1, list(rope_axes_dim), theta)
        self.max_num_latent_frames = int(max_num_latent_frames)
        self._tables: dict = {}

    def table(self, height: int, width: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (height, width, str(device))
        if key not in self._tables:
            grid = torch.empty((1, 0, self.max_num_latent_frames, height, width), device=device)
            self._tables[key] = self.embed(grid)
        return self._tables[key]


def _rows(table: tuple[torch.Tensor, torch.Tensor], positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return table[0].index_select(0, positions), table[1].index_select(0, positions)


@dataclass
class AttentionInputs:
    """What one block's attention needs besides the streams: the window geometry, the cached history, the RoPE rows."""

    tokens: int                                              # image tokens of the window (all ranks)
    rope_block: tuple[torch.Tensor, torch.Tensor]            # rows of this rank's attention block (padding clamped)
    rope_window: tuple[torch.Tensor, torch.Tensor]           # rows of the whole window, in order
    history: Optional[tuple[torch.Tensor, torch.Tensor]] = None   # cached pre-RoPE key / value kept on this rank
    rope_history: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ring_history_tokens: Optional[list[int]] = None          # Ring: history tokens kept by every ring member
    chunk_tokens: Optional[list[int]] = None                 # KV recompute: tokens of every chunk of the sequence


class MovieRollAttention(HunyuanVideo15Attention):
    """HunyuanVideo-1.5's dual-stream attention whose image queries also see the cached history.

    Image queries attend to ``[history, window, text]`` (role ``movieroll.video``), text queries to ``[window, text]``
    (role ``movieroll.text``), and a KV recompute attends block-wise causally (role ``movieroll.recompute``). Dense calls
    go through the backend vLLM-Omni selected for the role; Ring attention through `parallel.ring_attention`. RoPE is
    applied to fp32 q / k (MovieRoll's precision) with the parent's shared rotary op.
    """

    def __init__(self, dim: int, heads: int, head_dim: int, layout: SequenceLayout, eps: float = 1e-6, quant_config=None,
                 prefix: str = ""):
        super().__init__(query_dim=dim, heads=heads, dim_head=head_dim, bias=True, added_kv_proj_dim=dim, out_dim=dim,
                         eps=eps, quant_config=quant_config, prefix=prefix)
        self.layout = layout
        local_heads = self.to_qkv.num_heads            # after tensor parallelism
        if local_heads % layout.ulysses_world:
            raise ValueError(
                f"{local_heads} heads per tensor-parallel rank cannot be split over ulysses_degree={layout.ulysses_world}: "
                f"the {heads} heads must divide by tensor_parallel_size * ulysses_degree"
            )
        heads_here = local_heads // layout.ulysses_world       # after the Ulysses all-to-all
        common = dict(num_heads=heads_here, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False,
                      num_kv_heads=heads_here, qkv_layout="BSND", skip_sequence_parallel=True, allow_fp32_fallback=True,
                      role_category="self")
        self.attn = Attention(role="movieroll.video", prefix=f"{prefix}.attn", **common)
        self.attn_text = Attention(role="movieroll.text", prefix=f"{prefix}.attn_text", **common)
        self.attn_recompute = Attention(role="movieroll.recompute", prefix=f"{prefix}.attn_recompute", **common)

    def _rope(self, x: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        cos, sin = cos_sin
        if cos.shape[0] != x.shape[1]:
            raise ValueError(f"RoPE rows {tuple(cos.shape)} do not match the tokens of x {tuple(x.shape)}")
        return self.rope(x, cos.to(x.dtype), sin.to(x.dtype))

    def _project(self, linear, norm_q, norm_k, x: torch.Tensor):
        qkv, _ = linear(x.contiguous())
        size = linear.num_heads * self.head_dim
        q, k, v = qkv.split([size, size, size], dim=-1)
        q, k, v = (t.unflatten(-1, (linear.num_heads, self.head_dim)) for t in (q, k, v))
        return norm_q(q), norm_k(k), v

    def _project_image(self, x: torch.Tensor):
        """Image q / k / v of this rank's attention block: ``[1, block tokens, heads / (tp * ulysses), head_dim]``."""
        return tuple(self.layout.to_heads(t) for t in self._project(self.to_qkv, self.norm_q, self.norm_k, x))

    def _project_text(self, x: torch.Tensor):
        return tuple(self.layout.head_slice(t) for t in self._project(self.add_kv_proj, self.norm_added_q, self.norm_added_k, x))

    def _out_image(self, attended: torch.Tensor) -> torch.Tensor:
        return self.to_out[0](self.layout.to_tokens(attended).flatten(2))

    def _out_text(self, attended: torch.Tensor) -> torch.Tensor:
        return self.to_add_out(self.layout.gather_heads(attended).flatten(2))

    def forward(self, img_modulated: torch.Tensor, txt_modulated: Optional[torch.Tensor], inputs: AttentionInputs):
        """Returns ``(img_out, txt_out, kv)``; ``kv`` is what this rank caches of the window: ``(key, value, positions)``
        before RoPE (``positions`` is None when the rank keeps every token)."""
        layout, n = self.layout, inputs.tokens
        device = img_modulated.device
        q, k_block, v_block = self._project_image(img_modulated)
        q = self._rope(q, inputs.rope_block)
        text = self._project_text(txt_modulated) if txt_modulated is not None else None

        if layout.ring:
            mine = layout.valid_block_index(n, device)
            kv = (k_block.index_select(1, mine), v_block.index_select(1, mine), layout.block_positions(n, device)[mine])
            k_rope = self._rope(k_block, inputs.rope_block)
            history_tokens = inputs.ring_history_tokens
            longest = max(history_tokens)
            if inputs.history is not None and longest > 0:
                # every member's history is padded to the longest so the ring passes equal shapes; the padding is
                # dropped again by the members' index lists
                h_k = self._rope(inputs.history[0].to(k_block.dtype), inputs.rope_history)
                h_v = inputs.history[1].to(v_block.dtype)
                h_k, h_v = SequenceLayout._pad_tokens(h_k, longest), SequenceLayout._pad_tokens(h_v, longest)
                k_all, v_all = torch.cat([h_k, k_rope], dim=1), torch.cat([h_v, v_block], dim=1)
            else:
                k_all, v_all, longest = k_rope, v_block, 0
            window_index = [layout.valid_block_index(n, device, member) for member in layout.kv_members]
            all_index = [
                torch.cat([torch.arange(history_tokens[j], device=device), longest + window_index[j]])
                for j in range(len(window_index))
            ]
            joint = (text[1], text[2]) if text is not None else (None, None)
            img = ring_attention(layout, q, k_all, v_all, all_index, *joint)
            txt_out = None
            if text is not None:
                txt_out = self._out_text(ring_attention(layout, text[0], k_rope, v_block, window_index, text[1], text[2]))
            return self._out_image(img), txt_out, kv

        k_pre, v_window = layout.assemble(k_block, n), layout.assemble(v_block, n)
        k_window = self._rope(k_pre, inputs.rope_window)
        if inputs.history is not None:
            k_all = torch.cat([self._rope(inputs.history[0].to(k_window.dtype), inputs.rope_history), k_window], dim=1)
            v_all = torch.cat([inputs.history[1].to(v_window.dtype), v_window], dim=1)
        else:
            k_all, v_all = k_window, v_window
        txt_out = None
        if text is None:
            img = self.attn(q, k_all, v_all, None)
        else:
            tq, tk, tv = text
            img = self.attn(q, torch.cat([k_all, tk], dim=1), torch.cat([v_all, tv], dim=1), None)
            # the text does not see the history: its keys are the window's and the text's own
            txt_out = self._out_text(
                self.attn_text(tq, torch.cat([k_window, tk], dim=1), torch.cat([v_window, tv], dim=1), None)
            )
        return self._out_image(img), txt_out, (k_pre, v_window, None)

    def forward_causal(self, img_modulated: torch.Tensor, inputs: AttentionInputs):
        """Block-wise causal attention over consecutive chunks (KV recompute): a token of chunk i attends to chunks 0..i.

        Returns ``(img_out, (key, value))`` with the key / value of the whole sequence before RoPE, in order (gathered
        over the Ring / AllGather-KV group: a recompute is rare and short).
        """
        layout, n = self.layout, inputs.tokens
        q, k_block, v_block = self._project_image(img_modulated)
        q = self._rope(q, inputs.rope_block)
        k_pre, v_all = layout.assemble(k_block, n), layout.assemble(v_block, n)
        k_all = self._rope(k_pre, inputs.rope_window)
        positions = layout.block_positions(n, q.device)
        out = torch.zeros_like(q)
        start = 0
        for tokens in inputs.chunk_tokens:
            end = start + tokens
            rows = torch.nonzero((positions >= start) & (positions < end)).flatten()
            if rows.numel():
                out[:, rows] = self.attn_recompute(q.index_select(1, rows), k_all[:, :end], v_all[:, :end], None).to(out.dtype)
            start = end
        return self._out_image(out), (k_pre, v_all)


class MovieRollTransformerBlock(nn.Module):
    """One double-stream block with per-frame AdaLN on the image stream (HunyuanVideo-1.5 parameter names).

    ``forward`` serves the three uses -- denoise a window, encode a clean chunk into the cache, the causal pass of KV
    recompute -- selected by ``mode``.
    """

    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: float, layout: SequenceLayout, quant_config=None,
                 prefix: str = ""):
        super().__init__()
        self.norm1 = _Modulation(dim, 6)
        self.norm1_context = _Modulation(dim, 6)
        self.attn = MovieRollAttention(dim, heads, head_dim, layout, quant_config=quant_config, prefix=f"{prefix}.attn")
        self.ff = FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, quant_config=quant_config, prefix=f"{prefix}.ff")
        self.ff_context = FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, quant_config=quant_config,
                                      prefix=f"{prefix}.ff_context")

    def _image_tail(self, x, attn_out, mods, layout):
        # chunk order of the modulation: shift1, scale1, gate1, shift2, scale2, gate2
        _, _, gate1, shift2, scale2, gate2 = mods
        x = x + _gate_frames(attn_out, gate1, layout)
        return x + _gate_frames(self.ff(_modulate_frames(x, shift2, scale2, layout)), gate2, layout)

    def forward(self, mode: str, x, vec, frames: FrameLayout, inputs: AttentionInputs, txt=None, vec_txt=None):
        """``denoise`` -> ``(x, txt)``; ``encode`` -> ``(x, kv)``; ``recompute`` -> ``(x, (key, value))``."""
        mods = self.norm1(vec).chunk(6, dim=-1)
        img_modulated = _modulate_frames(x, mods[0], mods[1], frames)
        if mode == "recompute":
            img_attn, kv = self.attn.forward_causal(img_modulated, inputs)
            return self._image_tail(x, img_attn, mods, frames), kv
        if mode == "encode":
            img_attn, _, kv = self.attn(img_modulated, None, inputs)
            return self._image_tail(x, img_attn, mods, frames), kv
        if mode != "denoise":
            raise ValueError(f"unknown block mode {mode!r}")
        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = self.norm1_context(vec_txt).chunk(6, dim=-1)
        img_attn, txt_attn, _ = self.attn(img_modulated, _modulate_text(txt, txt_shift1, txt_scale1), inputs)
        x = self._image_tail(x, img_attn, mods, frames)
        txt = txt + txt_attn * txt_gate1
        txt = txt + self.ff_context(_modulate_text(txt, txt_shift2, txt_scale2)) * txt_gate2
        return x, txt


class _FinalNorm(nn.Module):
    """MovieRoll's final AdaLN: per-frame ``(shift, scale)`` from ``SiLU -> Linear`` (note the order: shift first)."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor, vec: torch.Tensor, layout: FrameLayout) -> torch.Tensor:
        shift, scale = self.linear(F.silu(vec)).chunk(2, dim=-1)
        return _modulate_frames(x, shift, scale, layout)


class MovieRollTransformer3DModel(nn.Module):
    """MovieRoll's streaming backbone (HunyuanVideo-1.5 MMDiT + per-frame AdaLN + clean-history KV cache)."""

    _repeated_blocks = ["MovieRollTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]
    _hsdp_shard_conditions = [is_transformer_block_module]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "add_kv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
    }

    def __init__(
        self,
        od_config: Optional[OmniDiffusionConfig] = None,
        in_channels: int = 32,
        out_channels: int = 32,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 54,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        patch_size_t: int = 1,
        text_embed_dim: int = 3584,
        text_embed_2_dim: int = 1472,
        rope_theta: float = 256.0,
        rope_axes_dim: tuple[int, ...] = (16, 56, 56),
        max_num_latent_frames: int = 120,
        full_history_block_num: int = 6,
        sink_chunks: int = 1,
        quant_config=None,
        sequence_layout: Optional[SequenceLayout] = None,
    ):
        super().__init__()
        if patch_size != 1 or patch_size_t != 1:
            raise ValueError("MovieRoll uses 1x1x1 patches")
        if not 0 <= full_history_block_num <= num_layers:
            raise ValueError(f"full_history_block_num must be in [0, {num_layers}], got {full_history_block_num}")
        self.od_config = od_config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.inner_dim = num_attention_heads * attention_head_dim
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.full_history_block_num = full_history_block_num
        self.sink_chunks = sink_chunks
        self.sink_recent_blocks = list(range(full_history_block_num, num_layers))
        self.rope = RopeTable(rope_axes_dim, rope_theta, max_num_latent_frames)
        # the sequence-parallel groups of the worker this model lives in (the identity without sequence parallelism)
        self.layout = sequence_layout if sequence_layout is not None else SequenceLayout.from_framework()

        dim = self.inner_dim
        # the model input is [latents, zeros (the unused conditioning copy), zeros (mask)]
        self.x_embedder = HunyuanVideo15PatchEmbed((1, 1, 1), 2 * in_channels + 1, dim)
        self.context_embedder = HunyuanVideo15TokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )
        self.context_embedder_2 = HunyuanVideo15ByT5TextProjection(text_embed_2_dim, 2048, dim)
        self.time_embed = MovieRollTimeEmbedding(dim)
        self.cond_type_embed = nn.Embedding(3, dim)
        self.transformer_blocks = nn.ModuleList(
            [
                MovieRollTransformerBlock(dim, num_attention_heads, attention_head_dim, mlp_ratio, self.layout,
                                          quant_config=quant_config, prefix=f"transformer_blocks.{i}")
                for i in range(num_layers)
            ]
        )
        self.norm_out = _FinalNorm(dim)
        self.proj_out = nn.Linear(dim, out_channels)

    # ------------------------------------------------------------------ helpers
    @property
    def dtype(self) -> torch.dtype:
        return self.proj_out.weight.dtype

    def new_cache(self, latent_height: int, latent_width: int, dtype: Optional[torch.dtype] = None) -> KVCache:
        return KVCache(self.num_layers, latent_height * latent_width, dtype or self.dtype)

    def _embed_image(self, hidden_states: torch.Tensor):
        if hidden_states.shape[0] != 1:
            raise ValueError(f"batch size 1 only, got {hidden_states.shape[0]}")
        if hidden_states.shape[1] != 2 * self.in_channels + 1:
            raise ValueError(f"expected {2 * self.in_channels + 1} input channels, got {hidden_states.shape[1]}")
        _, _, frames, height, width = hidden_states.shape
        return self.x_embedder(hidden_states.to(self.dtype)), frames, height * width

    def _text_stream(self, text: TextTokens, timestep_txt: torch.Tensor):
        """Refine the prompt for this timestep and pack ``[byT5 glyph tokens, prompt tokens]`` (valid tokens only, so the
        refiner's pooled context, a mean when no mask is given, is the mean over the prompt), each with its type
        embedding."""
        timestep_txt = timestep_txt.float().flatten()[:1]
        qwen = self.context_embedder(text.qwen.to(self.dtype), timestep_txt, None)
        qwen = qwen + self.cond_type_embed(torch.zeros(qwen.shape[:2], dtype=torch.long, device=qwen.device))
        pieces = [qwen]
        if text.byt5 is not None and text.byt5.shape[1] > 0:
            glyph = self.context_embedder_2(text.byt5.to(self.dtype))
            glyph = glyph + self.cond_type_embed(torch.ones(glyph.shape[:2], dtype=torch.long, device=glyph.device))
            pieces.insert(0, glyph)
        return torch.cat(pieces, dim=1), self.time_embed(timestep_txt)[:, None, :]

    def _unpatchify(self, x: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
        # patches are 1x1x1, so this is a reshape of [B, T*H*W, C] to [B, C, T, H, W]
        return x.view(x.shape[0], frames, height, width, -1).permute(0, 4, 1, 2, 3).contiguous()

    def _inputs(self, index: int, cache: Optional[KVCache], tokens: int, height: int, width: int, device,
                chunk_tokens: Optional[list[int]] = None) -> AttentionInputs:
        """The attention inputs of block ``index`` for a window of ``tokens`` tokens behind its cache (or none)."""
        layout = self.layout
        table = self.rope.table(height, width, device)
        offset = cache.total_tokens(index) if cache is not None else 0
        if offset + tokens > table[0].shape[0]:
            raise ValueError(
                f"{offset + tokens} tokens ({offset} cached + {tokens} in the window) exceed the position grid of "
                f"{table[0].shape[0]} tokens, i.e. max_num_latent_frames={self.rope.max_num_latent_frames}"
            )
        block = layout.block_positions(tokens, device).clamp(max=tokens - 1) + offset
        inputs = AttentionInputs(
            tokens=tokens,
            rope_block=_rows(table, block),
            rope_window=(table[0][offset:offset + tokens], table[1][offset:offset + tokens]),
            chunk_tokens=chunk_tokens,
        )
        history = cache.get(index) if cache is not None else None
        if history is not None:
            positions = cache.positions(index)
            inputs.history = history
            inputs.rope_history = (table[0][:offset], table[1][:offset]) if positions is None else _rows(table, positions)
        if layout.ring:
            frames = cache.chunk_frames(index) if cache is not None else []
            inputs.ring_history_tokens = [
                sum(layout.member_chunk_tokens(f * height * width, member) for f in frames) for member in layout.kv_members
            ]
        return inputs

    # --------------------------------------------------------------- the single entry point
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        *,
        mode: str = "denoise",
        text: Optional[TextTokens] = None,
        timestep_txt: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        chunk_frames: Optional[list[int]] = None,
    ) -> Optional[torch.Tensor]:
        """``denoise``: velocity of a noisy window ``[1, 2C+1, T, H, W]`` at per-frame ``timestep`` ``[1, T]`` given the
        cache and the text; ``encode``: append a clean chunk to the cache (timestep 0, no text); ``recompute``: rebuild the
        cache from the selected chunks concatenated in ``hidden_states`` (``chunk_frames``). Returns None but for denoise."""
        if mode == "denoise":
            return self._denoise(hidden_states, timestep, text, timestep_txt, cache)
        if mode == "encode":
            self._encode(hidden_states, cache)
            return None
        if mode == "recompute":
            self._recompute(hidden_states, chunk_frames, cache)
            return None
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    def _denoise(self, hidden_states, timestep, text, timestep_txt, cache) -> torch.Tensor:
        x, frames, frame_tokens = self._embed_image(hidden_states)
        height, width = hidden_states.shape[-2:]
        if timestep is None or timestep.shape != (1, frames):
            raise ValueError(f"timestep must be [1, {frames}] (one per latent frame), got "
                             f"{None if timestep is None else tuple(timestep.shape)}")
        if timestep_txt is None:
            timestep_txt = timestep[:, :1]
        layout = FrameLayout(frames, frame_tokens, self.layout)
        vec = self.time_embed(timestep).view(1, frames, -1)
        txt, vec_txt = self._text_stream(text, timestep_txt)
        x = self.layout.shard(x)
        for index, block in enumerate(self.transformer_blocks):
            inputs = self._inputs(index, cache, layout.tokens, height, width, x.device)
            x, txt = block("denoise", x, vec, layout, inputs, txt=txt, vec_txt=vec_txt)
        out = self.layout.gather(self.proj_out(self.norm_out(x, vec, layout)), layout.tokens)
        return self._unpatchify(out, frames, height, width)

    def _store(self, cache: KVCache, index: int, kv, frames: int) -> None:
        key, value, positions = kv
        if positions is not None:
            # positions of the kept tokens inside the window, which here is exactly one chunk
            cache.append(index, key, value, frames, positions)
        else:
            cache.append(index, key, value, frames)

    def _encode(self, hidden_states: torch.Tensor, cache: KVCache) -> None:
        x, frames, frame_tokens = self._embed_image(hidden_states)
        height, width = hidden_states.shape[-2:]
        layout = FrameLayout(frames, frame_tokens, self.layout)
        vec = self.time_embed(torch.zeros(frames, device=x.device)).view(1, frames, -1)
        for index in self.sink_recent_blocks:
            # the new chunk of a sink-recent block sees the sink only; the stale recent chunk leaves first
            cache.trim_to_sink(index, self.sink_chunks)
        x = self.layout.shard(x)
        for index, block in enumerate(self.transformer_blocks):
            inputs = self._inputs(index, cache, layout.tokens, height, width, x.device)
            x, kv = block("encode", x, vec, layout, inputs)
            self._store(cache, index, kv, frames)
        cache.record_chunk(frames)

    def _recompute(self, hidden_states: torch.Tensor, chunk_frames: list[int], cache: KVCache) -> None:
        """The full-history blocks re-encode every selected chunk with block-wise causal attention; the sink-recent
        blocks keep only the sink (first) and the recent (last) chunk of that same pass, re-encoded on their own."""
        x, frames, frame_tokens = self._embed_image(hidden_states)
        height, width = hidden_states.shape[-2:]
        if chunk_frames is None or sum(chunk_frames) != frames:
            raise ValueError(f"chunk_frames {chunk_frames} do not add up to the {frames} frames given")
        cache.clear()
        device = x.device

        def causal_pass(x_full, blocks, chunk_frames):
            """Run ``blocks`` over the chunks of the full sequence ``x_full``; returns the full sequence afterwards."""
            blocks = list(blocks)
            if not blocks:
                return x_full
            total = sum(chunk_frames)
            layout = FrameLayout(total, frame_tokens, self.layout)
            chunk_tokens = [f * frame_tokens for f in chunk_frames]
            vec = self.time_embed(torch.zeros(total, device=device)).view(1, total, -1)
            # under Ring a rank keeps, of every chunk, the tokens its block holds when that chunk is encoded on its own
            starts = [sum(chunk_tokens[:c]) for c in range(len(chunk_tokens))]
            kept = [self.layout.chunk_positions(t, device) for t in chunk_tokens]
            x_local = self.layout.shard(x_full)
            for index in blocks:
                inputs = self._inputs(index, None, layout.tokens, height, width, device, chunk_tokens=chunk_tokens)
                x_local, (key, value) = self.transformer_blocks[index]("recompute", x_local, vec, layout, inputs)
                if self.layout.ring:
                    rows = torch.cat([s + p for s, p in zip(starts, kept)])
                    cache.replace(index, key.index_select(1, rows), value.index_select(1, rows), chunk_frames, kept)
                else:
                    cache.replace(index, key, value, chunk_frames)
            return self.layout.gather(x_local, layout.tokens)

        x = causal_pass(x, range(self.full_history_block_num), chunk_frames)
        if self.sink_recent_blocks:
            first, last = chunk_frames[0] * frame_tokens, chunk_frames[-1] * frame_tokens
            if len(chunk_frames) == 1:
                kept, kept_frames = x[:, :first], [chunk_frames[0]]
            else:
                kept, kept_frames = torch.cat([x[:, :first], x[:, -last:]], dim=1), [chunk_frames[0], chunk_frames[-1]]
            causal_pass(kept.contiguous(), self.sink_recent_blocks, kept_frames)
        cache.record_recompute(chunk_frames)

    # --------------------------------------------------------------------- weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load a MovieRoll DiT checkpoint (its own tensor names, renamed by `weights.map_name`)."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        tracker = FusedShardTracker()
        for name, tensor in weights:
            for mapped in map_name(name):
                if mapped.name not in params:
                    raise KeyError(f"checkpoint key {name!r} maps to {mapped.name!r}, which the model does not have")
                part = tensor if mapped.part is None else tensor.chunk(3, dim=0)[mapped.part]
                param = params[mapped.name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                if mapped.shard_id is not None:
                    loader(param, part, mapped.shard_id)
                else:
                    loader(param, part)
                tracker.add(mapped)
                loaded.add(mapped.name)
        tracker.check()
        return loaded
