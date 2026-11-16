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
# The block layout (parameter names, the fused `*_attn_qkv`, the fused norm / residual / gate ops and the order
# they are applied in) follows sglang's own `runtime/models/dits/hunyuanvideo.py` at v0.5.10.post1, which is
# itself adapted from FastVideo. That file's notice:
#
#   # Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
#   # SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

"""MovieRoll's streaming MMDiT, written the way SGLang Diffusion writes a DiT.

The backbone is HunyuanVideo-1.5's: a patch embedding, a text token refiner, a byT5 projection, double-stream
blocks and a final projection. SGLang already has this family (`HunyuanVideoTransformer3DModel`: same parameter
names, same fused ``*_attn_qkv``), so this module is built from SGLang's own pieces -- `QKVParallelLinear` /
`RowParallelLinear` / `MLP` / `ModulateProjection` (which shard themselves under ``--tp-size``),
`TimestepEmbedder`, `PatchEmbed`, `RMSNorm` with `apply_qk_norm`, the fused `LayerNormScaleShift` /
`ScaleResidualLayerNormScaleShift` / `MulAdd`, the token refiner of the Hunyuan DiT, `LocalAttention` (the
attention backend selector) and the rotary kernel -- and adds what MovieRoll has on top:

* per-frame AdaLN: ``vec`` holds one timestep embedding per latent frame, so the shift / scale / gate of the
  image stream are the ones of each token's own frame (the frames of a rolling window sit at different noise
  levels). Where the fused kernels take a per-frame ``[B, F, 1, C]`` argument they get one;
* a clean-history KV cache: the image queries of a window also attend to the cached keys / values (pre-RoPE,
  re-positioned on every use), while the text queries do not see the history;
* HieraKV: the first ``full_history_block_num`` blocks keep every chunk in their cache, the others only the
  sink and the most recent chunk;
* KV recompute: the selected chunks are re-encoded at timestep 0 with block-wise causal attention.

**Everything goes through a ``forward``.** The three things a block does (denoise a window, encode a clean
chunk, re-encode for KV recompute) are one ``forward`` with a ``mode``, and the model's own three entry points
are likewise one ``forward`` with a ``mode``. That is not a style choice: FSDP2 all-gathers a sharded module's
parameters in the pre-forward hook of ``__call__``, and SGLang's layer-wise offload stages weights in a forward
pre-hook, so a side method would run against sharded or placeholder weights. For the same reason the blocks are
visited in increasing index order in every mode.

Parallelism is in `sglang_movieroll.runtime.distributed.layout` (tensor / Ulysses / ring) and in
`sglang_movieroll.runtime.layers.attention.cached_joint_attention` (the ring's log-sum-exp merge); this module
only asks them where the tokens are. Batch size is 1: a request is one video, and classifier-free guidance is
two forwards (on one rank, or one each under ``--enable-cfg-parallel``).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    RMSNorm,
    ScaleResidualLayerNormScaleShift,
    apply_qk_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import _apply_rotary_emb
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    ModulateProjection,
    PatchEmbed,
    TimestepEmbedder,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo import SingleTokenRefiner
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin

from sglang_movieroll.configs.models.dits.movieroll import MovieRollDitConfig
from sglang_movieroll.rolling.kv_cache import KVCache
from sglang_movieroll.rolling.rope import RopeRows, RopeTable, Segment, chunk_spans
from sglang_movieroll.runtime.distributed.layout import Block, ParallelLayout
from sglang_movieroll.runtime.layers.attention.cached_joint_attention import ring_attend

DENOISE, ENCODE, RECOMPUTE = "denoise", "encode", "recompute"

# The fused CuTe norm kernels take a per-frame [B, F, 1, C] argument, but only for a width that is a multiple of
# 256 and at most 8192 on CUDA; everywhere else the values are expanded to one per token, which the same layers
# accept as [B, L, C].
_FUSED_WIDTH_MULTIPLE = 256
_FUSED_WIDTH_MAX = 8192


def _rope(x: torch.Tensor, rows: RopeRows) -> torch.Tensor:
    """Interleaved-pair RoPE of ``x`` ``[B, S, H, D]`` with cos / sin ``[S, D / 2]`` (SGLang's rotary kernel)."""
    cos, sin = rows
    if cos.shape != (x.shape[1], x.shape[-1] // 2):
        raise ValueError(
            f"RoPE rows {tuple(cos.shape)} do not match the tokens of x {tuple(x.shape)}"
        )
    if x.shape[1] == 0:
        return x
    return _apply_rotary_emb(x, cos, sin, is_neox_style=False)


class FrameLayout:
    """Where the image tokens of a window sit: ``frame_tokens`` per latent frame, this rank holding a slice.

    This is the layout *before* attention (the sequence-parallel token shard), which is what the per-frame
    modulation and the residual stream use. It also decides how a per-frame value ``[B, T, C]`` is handed to the
    block's modulation: as the per-frame ``[B, T_local, 1, C]`` the fused kernels take when the shard covers
    whole frames and the width suits them, otherwise expanded to one value per local token.
    """

    def __init__(self, frames: int, frame_tokens: int, parallel: ParallelLayout):
        self.frames = frames
        self.frame_tokens = frame_tokens
        self.tokens = frames * frame_tokens
        self.parallel = parallel
        self.start, self.end = (
            parallel.local_range(self.tokens) if parallel.sharded else (0, self.tokens)
        )
        self.local_tokens = self.end - self.start
        self.frame_aligned = (
            self.start % frame_tokens == 0 and self.local_tokens % frame_tokens == 0
        )
        self._frame_index: Optional[torch.Tensor] = None
        self._token_index: Optional[torch.Tensor] = None

    @staticmethod
    def fused_per_frame(width: int, device: torch.device) -> bool:
        """Whether the fused kernels take the per-frame form for this width on this device."""
        return (
            device.type == "cuda"
            and width % _FUSED_WIDTH_MULTIPLE == 0
            and width <= _FUSED_WIDTH_MAX
        )

    def frame_index(self, device: torch.device) -> torch.Tensor:
        """Frame of every local frame slot; the slots past the last real frame repeat it (their output is dropped)."""
        if self._frame_index is None or self._frame_index.device != device:
            first = self.start // self.frame_tokens
            count = self.local_tokens // self.frame_tokens
            self._frame_index = torch.arange(first, first + count, device=device).clamp(
                max=self.frames - 1
            )
        return self._frame_index

    def token_index(self, device: torch.device) -> torch.Tensor:
        """Frame of every local token (the fallback form, when the shard does not cover whole frames)."""
        if self._token_index is None or self._token_index.device != device:
            positions = torch.arange(self.start, self.end, device=device)
            self._token_index = (positions // self.frame_tokens).clamp(
                max=self.frames - 1
            )
        return self._token_index

    def per_frame(self, values: torch.Tensor) -> torch.Tensor:
        """``[B, T, C]`` per-frame values -> the argument shape the fused norm / gate layers take here."""
        if not self.parallel.sharded and self.fused_per_frame(
            values.shape[-1], values.device
        ):
            return values.unsqueeze(2)
        if self.frame_aligned and self.fused_per_frame(values.shape[-1], values.device):
            return values.index_select(1, self.frame_index(values.device)).unsqueeze(2)
        return values.index_select(1, self.token_index(values.device))


class MovieRollDoubleStreamBlock(nn.Module):
    """One double-stream block with per-frame AdaLN on the image stream and a cache-aware attention.

    Image queries attend to ``[clean history, the window, the text]``, text queries to ``[the window, the
    text]``; both go through the attention backend SGLang selected. Under tensor parallelism the heads are split
    by SGLang's parallel linear layers; under Ulysses the image tokens are traded for heads around attention;
    under ring parallelism the keys are rotated around the ring group, with the text counted exactly once.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
        parallel: ParallelLayout,
        supported_attention_backends: set,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.parallel = parallel
        self.head_dim = hidden_size // num_attention_heads
        self.local_heads = parallel.check_heads(num_attention_heads)
        mlp_hidden = int(hidden_size * mlp_ratio)

        for stream in ("img", "txt"):
            setattr(
                self,
                f"{stream}_mod",
                ModulateProjection(
                    hidden_size,
                    factor=6,
                    act_layer="silu",
                    prefix=f"{prefix}.{stream}_mod",
                ),
            )
            # the fused norm / residual / gate kernels of SGLang's own double-stream block
            setattr(
                self,
                f"{stream}_attn_norm",
                LayerNormScaleShift(hidden_size, elementwise_affine=False),
            )
            setattr(
                self,
                f"{stream}_attn_residual_mlp_norm",
                ScaleResidualLayerNormScaleShift(hidden_size, elementwise_affine=False),
            )
            setattr(self, f"{stream}_mlp_residual", MulAdd())
            # the checkpoint stores one fused [3 * hidden, hidden] tensor; SGLang's loader splits it into the
            # q / k / v shards of this tensor-parallel rank
            setattr(
                self,
                f"{stream}_attn_qkv",
                QKVParallelLinear(
                    hidden_size=hidden_size,
                    head_size=self.head_dim,
                    total_num_heads=num_attention_heads,
                    bias=True,
                    prefix=f"{prefix}.{stream}_attn_qkv",
                    quant_config=quant_config,
                ),
            )
            setattr(self, f"{stream}_attn_q_norm", RMSNorm(self.head_dim, eps=1e-6))
            setattr(self, f"{stream}_attn_k_norm", RMSNorm(self.head_dim, eps=1e-6))
            setattr(
                self,
                f"{stream}_attn_proj",
                RowParallelLinear(
                    hidden_size,
                    hidden_size,
                    bias=True,
                    input_is_parallel=True,
                    prefix=f"{prefix}.{stream}_attn_proj",
                    quant_config=quant_config,
                ),
            )
            setattr(
                self,
                f"{stream}_mlp",
                MLP(
                    hidden_size,
                    mlp_hidden,
                    bias=True,
                    act_type="gelu_pytorch_tanh",
                    prefix=f"{prefix}.{stream}_mlp",
                    quant_config=quant_config,
                ),
            )

        self.attn = LocalAttention(
            num_heads=self.local_heads,  # after tensor parallelism and the Ulysses all-to-all
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    # ------------------------------------------------------------------ projections
    def _qkv(self, stream: str, x: torch.Tensor):
        qkv, _ = getattr(self, f"{stream}_attn_qkv")(x)
        heads = qkv.shape[-1] // (3 * self.head_dim)
        q, k, v = qkv.unflatten(-1, (3, heads, self.head_dim)).unbind(2)
        q, k = apply_qk_norm(
            q.contiguous(),
            k.contiguous(),
            getattr(self, f"{stream}_attn_q_norm"),
            getattr(self, f"{stream}_attn_k_norm"),
            self.head_dim,
        )
        return q, k, v

    def _project_image(self, x: torch.Tensor, tokens: int):
        """Image q / k / v of this rank's attention block, ``[1, block tokens, heads / (tp * ulysses), head_dim]``."""
        q, k, v = self._qkv("img", x)
        return tuple(self.parallel.tokens_to_heads(t, tokens) for t in (q, k, v))

    def _project_text(self, x: torch.Tensor):
        q, k, v = self._qkv("txt", x)
        return tuple(self.parallel.head_slice(t) for t in (q, k, v))

    def _out_image(self, attended: torch.Tensor, tokens: int) -> torch.Tensor:
        out, _ = self.img_attn_proj(
            self.parallel.heads_to_tokens(attended, tokens).flatten(2)
        )
        return out

    def _attend(self, q, k, v, key_blocks, replicated):
        """Attention of ``q`` over this rank's keys plus a replicated block, over the ring group if there is one."""
        if not self.parallel.ringed:
            if replicated is not None:
                k = torch.cat([k, replicated[0]], dim=1)
                v = torch.cat([v, replicated[1]], dim=1)
            return self.attn(q, k, v)
        return ring_attend(
            self.attn,
            q,
            k,
            v,
            key_blocks,
            self.parallel.ring_rank,
            self.parallel.ring_group(),
            replicated=replicated,
        )

    def _attend_causal(self, img_modulated, rope_q, rope_k, chunk_tokens: List[int]):
        """Block-wise causal attention over consecutive chunks (KV recompute): chunk i attends to chunks 0..i."""
        total = sum(chunk_tokens)
        q, key, value = self._project_image(img_modulated, total)
        if self.parallel.ringed:
            # the causal pass needs every key of the prefix, which the ring shards; gather it once (recompute is rare)
            blocks = self.parallel.ring_blocks(total)
            key = self.parallel.ring_all_gather(key, blocks)
            value = self.parallel.ring_all_gather(value, blocks)
        q = _rope(q, rope_q)
        keyed = _rope(key, rope_k)
        block = self.parallel.ring_block(total)
        outputs, start = [], 0
        for tokens in chunk_tokens:
            end = start + tokens
            low, high = max(block.start, start), min(block.start + block.valid, end)
            if high > low:
                outputs.append(
                    self.attn(
                        q[:, low - block.start : high - block.start],
                        keyed[:, :end],
                        value[:, :end],
                    )
                )
            start = end
        attended = torch.cat(outputs, dim=1) if outputs else q[:, :0]
        return attended, (key, value)

    # ------------------------------------------------------------------ the one forward
    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        *,
        mode: str,
        layout: FrameLayout,
        rope_q: RopeRows,
        rope_k: RopeRows,
        txt: Optional[torch.Tensor] = None,
        vec_txt: Optional[torch.Tensor] = None,
        history: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_blocks: Optional[List[Block]] = None,
        window_blocks: Optional[List[Block]] = None,
        chunk_tokens: Optional[List[int]] = None,
    ):
        """``denoise`` -> ``(image, text)``; ``encode`` and ``recompute`` -> ``(image, (key, value))``, pre-RoPE.

        Under ring parallelism the key / value a ``recompute`` returns is the whole re-encoded sequence (the ring
        group gathered it for the block-causal pass), so the caller slices each chunk's share out of it; without
        ring the local tensor already is the whole sequence and the same slicing is the identity.
        """
        # chunk order of the modulation: shift1, scale1, gate1, shift2, scale2, gate2
        shift1, scale1, gate1, shift2, scale2, gate2 = self.img_mod(vec).chunk(
            6, dim=-1
        )
        shift1, scale1 = layout.per_frame(shift1), layout.per_frame(scale1)
        gate1, shift2, scale2, gate2 = (
            layout.per_frame(t) for t in (gate1, shift2, scale2, gate2)
        )
        img_modulated = self.img_attn_norm(x, shift1, scale1)

        if mode == RECOMPUTE:
            attended, keys = self._attend_causal(
                img_modulated, rope_q, rope_k, chunk_tokens
            )
            return (
                self._image_tail(x, attended, (gate1, shift2, scale2, gate2), layout),
                keys,
            )

        q, k, v = self._project_image(img_modulated, layout.tokens)
        keys = (k, v)  # the window's own key / value, before RoPE
        if history is not None:
            k = torch.cat([history[0].to(k.dtype), k], dim=1)
            v = torch.cat([history[1].to(v.dtype), v], dim=1)
        q = _rope(q, rope_q)
        k = _rope(k, rope_k)

        if mode == ENCODE:
            attended = self._attend(q, k, v, key_blocks, replicated=None)
            return (
                self._image_tail(x, attended, (gate1, shift2, scale2, gate2), layout),
                keys,
            )

        if mode != DENOISE:
            raise ValueError(
                f"unknown mode {mode!r}, expected one of {(DENOISE, ENCODE, RECOMPUTE)}"
            )

        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = (
            self.txt_mod(vec_txt).chunk(6, dim=-1)
        )
        text_q, text_k, text_v = self._project_text(
            self.txt_attn_norm(txt, txt_shift1, txt_scale1)
        )
        attended = self._attend(q, k, v, key_blocks, replicated=(text_k, text_v))
        # the text does not see the history: its keys are the window's and the text's own
        window_k, window_v = (
            k[:, k.shape[1] - keys[0].shape[1] :],
            v[:, v.shape[1] - keys[1].shape[1] :],
        )
        text_out = self._attend(
            text_q, window_k, window_v, window_blocks, replicated=(text_k, text_v)
        )
        text_out, _ = self.txt_attn_proj(
            self.parallel.gather_heads(text_out).flatten(2)
        )

        x = self._image_tail(x, attended, (gate1, shift2, scale2, gate2), layout)
        txt_mlp_input, txt_residual = self.txt_attn_residual_mlp_norm(
            txt, text_out, txt_gate1, txt_shift2, txt_scale2
        )
        txt = self.txt_mlp_residual(
            self.txt_mlp(txt_mlp_input), txt_gate2, txt_residual
        )
        return x, txt

    def _image_tail(self, x, attended, mods, layout: FrameLayout):
        gate1, shift2, scale2, gate2 = mods
        mlp_input, residual = self.img_attn_residual_mlp_norm(
            x, self._out_image(attended, layout.tokens), gate1, shift2, scale2
        )
        return self.img_mlp_residual(self.img_mlp(mlp_input), gate2, residual)


class ByT5Mapper(nn.Module):
    """Project byT5 glyph-text embeddings to the transformer width: LayerNorm -> fc1 -> GELU -> fc2 -> GELU -> fc3."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, prefix: str = ""):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = ReplicatedLinear(
            in_dim, hidden_dim, bias=True, prefix=f"{prefix}.fc1"
        )
        self.fc2 = ReplicatedLinear(
            hidden_dim, hidden_dim, bias=True, prefix=f"{prefix}.fc2"
        )
        self.fc3 = ReplicatedLinear(
            hidden_dim, out_dim, bias=True, prefix=f"{prefix}.fc3"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(self.layernorm(x.to(self.layernorm.weight.dtype)))
        x, _ = self.fc2(F.gelu(x))
        x, _ = self.fc3(F.gelu(x))
        return x


class MovieRollFinalLayer(nn.Module):
    """MovieRoll's final layer: per-frame ``(shift, scale)`` AdaLN (note the order, unlike diffusers') and a linear."""

    def __init__(self, hidden_size: int, out_channels: int, prefix: str = ""):
        super().__init__()
        self.adaLN_modulation = ModulateProjection(
            hidden_size, factor=2, act_layer="silu", prefix=f"{prefix}.adaLN_modulation"
        )
        self.norm_final = LayerNormScaleShift(hidden_size, elementwise_affine=False)
        # 32 output channels do not divide by a tensor-parallel degree, and this is one matrix per video:
        # keep it replicated
        self.linear = ReplicatedLinear(
            hidden_size, out_channels, bias=True, prefix=f"{prefix}.linear"
        )

    def forward(
        self, x: torch.Tensor, vec: torch.Tensor, layout: FrameLayout
    ) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        out, _ = self.linear(
            self.norm_final(x, layout.per_frame(shift), layout.per_frame(scale))
        )
        return out


class MovieRollTransformer3DModel(BaseDiT, OffloadableDiTMixin):
    """MovieRoll's streaming backbone (HunyuanVideo-1.5 MMDiT + per-frame AdaLN + clean-history KV cache)."""

    _fsdp_shard_conditions = MovieRollDitConfig()._fsdp_shard_conditions
    _compile_conditions = MovieRollDitConfig()._compile_conditions
    _supported_attention_backends = MovieRollDitConfig()._supported_attention_backends
    param_names_mapping = MovieRollDitConfig().param_names_mapping
    reverse_param_names_mapping = MovieRollDitConfig().reverse_param_names_mapping
    lora_param_names_mapping = MovieRollDitConfig().lora_param_names_mapping

    def __init__(
        self,
        config: MovieRollDitConfig,
        hf_config: dict[str, Any],
        quant_config: Optional[QuantizationConfig] = None,
        parallel: Optional[ParallelLayout] = None,
    ):
        super().__init__(config=config, hf_config=hf_config)
        if tuple(config.patch_size) != (1, 1, 1):
            raise ValueError("MovieRoll uses 1x1x1 patches")
        if not 0 <= config.full_history_block_num <= config.num_layers:
            raise ValueError(
                f"full_history_block_num must be in [0, {config.num_layers}], got {config.full_history_block_num}"
            )
        if sum(config.rope_axes_dim) != config.attention_head_dim:
            raise ValueError(
                f"rope_axes_dim {tuple(config.rope_axes_dim)} must add up to the head dimension {config.attention_head_dim}"
            )
        self.patch_size = list(config.patch_size)
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents
        self.full_history_block_num = config.full_history_block_num
        self.sink_chunks = config.sink_chunks
        self.sink_recent_blocks = list(
            range(config.full_history_block_num, config.num_layers)
        )
        self.rope = RopeTable(
            config.rope_axes_dim, config.rope_theta, config.max_num_latent_frames
        )
        # the parallel groups of the worker this model lives in (all degrees 1 without them)
        self.parallel = (
            parallel if parallel is not None else ParallelLayout.from_framework()
        )
        self.parallel.check_heads(config.num_attention_heads)

        dim = self.hidden_size
        prefix = config.prefix
        # the model input is [latents, zeros (the unused conditioning copy), zeros (mask)]
        self.img_in = PatchEmbed(
            self.patch_size, 2 * self.in_channels + 1, dim, prefix=f"{prefix}.img_in"
        )
        self.txt_in = SingleTokenRefiner(
            config.text_embed_dim,
            dim,
            config.num_attention_heads,
            depth=config.num_refiner_layers,
            prefix=f"{prefix}.txt_in",
        )
        self.byt5_in = ByT5Mapper(
            config.text_embed_2_dim,
            config.byt5_hidden_dim,
            dim,
            prefix=f"{prefix}.byt5_in",
        )
        self.time_in = TimestepEmbedder(
            dim, act_layer="silu", prefix=f"{prefix}.time_in"
        )
        # 0: the prompt's tokens, 1: the glyph tokens
        self.cond_type_embedding = nn.Embedding(3, dim)
        self.double_blocks = nn.ModuleList(
            [
                MovieRollDoubleStreamBlock(
                    dim,
                    config.num_attention_heads,
                    config.mlp_ratio,
                    self.parallel,
                    self._supported_attention_backends,
                    prefix=f"{prefix}.double_blocks.{i}",
                    quant_config=quant_config,
                )
                for i in range(config.num_layers)
            ]
        )
        self.final_layer = MovieRollFinalLayer(
            dim, config.out_channels, prefix=f"{prefix}.final_layer"
        )
        self.__post_init__()
        self.layer_names = ["double_blocks"]  # what SGLang's layer-wise offload stages

    # ------------------------------------------------------------------ helpers
    @property
    def dtype(self) -> torch.dtype:
        return self.final_layer.linear.weight.dtype

    def new_cache(
        self, latent_height: int, latent_width: int, dtype: Optional[torch.dtype] = None
    ) -> KVCache:
        return KVCache(
            self.num_layers,
            latent_height * latent_width,
            dtype or self.dtype,
            stored_tokens=self.parallel.stored_tokens,
        )

    def _embed_image(self, hidden_states: torch.Tensor):
        if hidden_states.shape[0] != 1:
            raise ValueError(f"batch size 1 only, got {hidden_states.shape[0]}")
        if hidden_states.shape[1] != 2 * self.in_channels + 1:
            raise ValueError(
                f"expected {2 * self.in_channels + 1} input channels, got {hidden_states.shape[1]}"
            )
        _, _, frames, height, width = hidden_states.shape
        return self.img_in(hidden_states.to(self.dtype)), frames, height * width

    def _text_stream(self, encoder_hidden_states, timestep_txt: torch.Tensor):
        """Refine the prompt for this timestep and pack ``[byT5 glyph tokens, prompt tokens]`` (valid tokens only)."""
        qwen, byt5 = encoder_hidden_states[0], (
            encoder_hidden_states[1] if len(encoder_hidden_states) > 1 else None
        )
        timestep_txt = timestep_txt.flatten()[:1]
        # each stream is tagged with its source: 0 for the prompt's tokens, 1 for the glyph tokens
        txt = self.txt_in(qwen.to(self.dtype), timestep_txt)
        txt = txt + self.cond_type_embedding(
            torch.zeros(txt.shape[:2], dtype=torch.long, device=txt.device)
        )
        pieces = [txt]
        if byt5 is not None and byt5.shape[1] > 0:
            glyph = self.byt5_in(byt5)
            glyph = glyph + self.cond_type_embedding(
                torch.ones(glyph.shape[:2], dtype=torch.long, device=glyph.device)
            )
            pieces.insert(0, glyph)
        return torch.cat(pieces, dim=1), self.time_in(timestep_txt)[:, None, :]

    def _frame_vec(self, timesteps: torch.Tensor) -> torch.Tensor:
        """``[T]`` per-frame timesteps -> ``[1, T, hidden]`` embeddings."""
        return self.time_in(timesteps.flatten()).view(1, -1, self.hidden_size)

    @staticmethod
    def _unpatchify(
        x: torch.Tensor, frames: int, height: int, width: int
    ) -> torch.Tensor:
        # patches are 1x1x1, so this is a reshape of [B, T*H*W, C] to [B, C, T, H, W]
        return (
            x.view(x.shape[0], frames, height, width, -1)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    # ------------------------------------------------------------------ where the tokens of the cache and the window are
    def _window_segments(
        self, chunk_frames: List[int], frame_tokens: int, window_tokens: int
    ):
        """Rotary segments of the queries and of the keys, and the per-ring-rank key counts.

        The keys are, in this order, each cached chunk's share of this rank followed by the window's; the queries
        are the window's. Without ring parallelism a rank's share is the whole thing, so this is the plain
        ``[0, cache)`` + ``[cache, cache + window)`` of a single GPU.
        """
        spans = chunk_spans(chunk_frames, frame_tokens)
        cache_tokens = sum(count for _, count in spans)
        key_segments: List[Segment] = []
        counts = [0] * self.parallel.ring_size
        for start, count in spans:
            block = self.parallel.ring_block(count)
            key_segments.append((start + block.start, block.valid))
            for rank, other in enumerate(self.parallel.ring_blocks(count)):
                counts[rank] += other.valid
        window = self.parallel.ring_block(window_tokens)
        window_counts = [
            other.valid for other in self.parallel.ring_blocks(window_tokens)
        ]
        key_segments.append((cache_tokens + window.start, window.valid))
        query_segments: List[Segment] = [(cache_tokens + window.start, window.valid)]
        width = max((a + b for a, b in zip(counts, window_counts)), default=0)
        key_blocks = [Block(0, width, a + b) for a, b in zip(counts, window_counts)]
        window_blocks = [
            Block(0, max(window_counts) if window_counts else 0, count)
            for count in window_counts
        ]
        return query_segments, key_segments, key_blocks, window_blocks

    def _slice_per_chunk(
        self, key: torch.Tensor, value: torch.Tensor, chunk_tokens: List[int]
    ):
        """This rank's share of every chunk, out of the whole re-encoded sequence (the identity without ring)."""
        if not self.parallel.ringed:
            return key, value
        keys, values, start = [], [], 0
        for tokens in chunk_tokens:
            block = self.parallel.ring_block(tokens)
            keys.append(key[:, start + block.start : start + block.start + block.valid])
            values.append(
                value[:, start + block.start : start + block.start + block.valid]
            )
            start += tokens
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)

    # ------------------------------------------------------------------ the one forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        timestep: Optional[torch.Tensor] = None,
        encoder_hidden_states_image=None,
        guidance=None,
        *,
        mode: str = DENOISE,
        timestep_txt: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        chunk_frames: Optional[List[int]] = None,
        **kwargs,
    ):
        """One of three things, by ``mode`` (see the module docstring on why they share a forward).

        * ``denoise``: the velocity of a noisy window ``[1, 2C+1, T, H, W]`` given the cache.
          ``encoder_hidden_states`` is ``[qwen, byt5 or None]`` (valid tokens only), ``timestep`` ``[1, T]`` one
          per latent frame, ``timestep_txt`` ``[1, 1]`` the prompt's. The image-conditioning and guidance
          arguments of SGLang's DiT signature do not apply (the image-to-video frame enters through the cache,
          guidance is the sampling loop's) and are not used.
        * ``encode``: append the clean chunk ``hidden_states`` to ``kv_cache``, at timestep 0 and without text.
        * ``recompute``: rebuild ``kv_cache`` from the chunks concatenated in ``hidden_states``
          (``chunk_frames`` gives their latent frame counts); returns the number of frames now cached.
        """
        if mode == DENOISE:
            return self._denoise(
                hidden_states, encoder_hidden_states, timestep, timestep_txt, kv_cache
            )
        if mode == ENCODE:
            return self._encode(hidden_states, kv_cache)
        if mode == RECOMPUTE:
            return self._recompute(hidden_states, chunk_frames, kv_cache)
        raise ValueError(
            f"unknown mode {mode!r}, expected one of {(DENOISE, ENCODE, RECOMPUTE)}"
        )

    def _denoise(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_txt,
        kv_cache: KVCache,
    ):
        x, frames, frame_tokens = self._embed_image(hidden_states)
        height, width = hidden_states.shape[-2:]
        if timestep.shape != (1, frames):
            raise ValueError(
                f"timestep must be [1, {frames}] (one per latent frame), got {tuple(timestep.shape)}"
            )
        layout = FrameLayout(frames, frame_tokens, self.parallel)
        x = self.parallel.shard(x)
        vec = self._frame_vec(timestep)
        txt, vec_txt = self._text_stream(encoder_hidden_states, timestep_txt)
        for index, block in enumerate(self.double_blocks):
            query_rows, key_rows, key_blocks, window_blocks = self._window_segments(
                kv_cache.chunk_frames(index), frame_tokens, layout.tokens
            )
            x, txt = block(
                x,
                vec,
                mode=DENOISE,
                layout=layout,
                txt=txt,
                vec_txt=vec_txt,
                rope_q=self.rope.rows(height, width, x.device, query_rows),
                rope_k=self.rope.rows(height, width, x.device, key_rows),
                history=kv_cache.get(index),
                key_blocks=key_blocks,
                window_blocks=window_blocks,
            )
        x = self.parallel.gather(self.final_layer(x, vec, layout), layout.tokens)
        return self._unpatchify(x, frames, height, width)

    @torch.no_grad()
    def _encode(self, hidden_states, kv_cache: KVCache) -> None:
        x, frames, frame_tokens = self._embed_image(hidden_states)
        height, width = hidden_states.shape[-2:]
        layout = FrameLayout(frames, frame_tokens, self.parallel)
        x = self.parallel.shard(x)
        vec = self._frame_vec(torch.zeros(frames, device=x.device))
        for index in self.sink_recent_blocks:
            # the new chunk of a sink-recent block sees the sink only; the stale recent chunk leaves first
            kv_cache.trim_to_sink(index, self.sink_chunks)
        for index, block in enumerate(self.double_blocks):
            query_rows, key_rows, key_blocks, _ = self._window_segments(
                kv_cache.chunk_frames(index), frame_tokens, layout.tokens
            )
            x, (key, value) = block(
                x,
                vec,
                mode=ENCODE,
                layout=layout,
                rope_q=self.rope.rows(height, width, x.device, query_rows),
                rope_k=self.rope.rows(height, width, x.device, key_rows),
                history=kv_cache.get(index),
                key_blocks=key_blocks,
            )
            kv_cache.append(index, key, value, frames)

    @torch.no_grad()
    def _recompute(
        self, hidden_states, chunk_frames: List[int], kv_cache: KVCache
    ) -> int:
        """The full-history blocks re-encode every selected chunk with block-wise causal attention; the
        sink-recent blocks keep only the sink (first) and the recent (last) chunk of that same pass.
        """
        x, frames, frame_tokens = self._embed_image(hidden_states)
        height, width = hidden_states.shape[-2:]
        if sum(chunk_frames) != frames:
            raise ValueError(
                f"chunk_frames {chunk_frames} do not add up to the {frames} frames given"
            )
        kv_cache.clear()

        def causal_pass(x, blocks, chunk_frames):
            """Run ``blocks`` over the chunks (``x`` is the full sequence); returns the full sequence."""
            total = sum(chunk_frames)
            layout = FrameLayout(total, frame_tokens, self.parallel)
            chunk_tokens = [n * frame_tokens for n in chunk_frames]
            vec = self._frame_vec(torch.zeros(total, device=x.device))
            block = self.parallel.ring_block(layout.tokens)
            rope_q = self.rope.rows(
                height, width, x.device, [(block.start, block.valid)]
            )
            rope_k = self.rope.rows(height, width, x.device, [(0, layout.tokens)])
            x = self.parallel.shard(x)
            for index in blocks:
                x, (key, value) = self.double_blocks[index](
                    x,
                    vec,
                    mode=RECOMPUTE,
                    layout=layout,
                    rope_q=rope_q,
                    rope_k=rope_k,
                    chunk_tokens=chunk_tokens,
                )
                kv_cache.replace(
                    index,
                    *self._slice_per_chunk(key, value, chunk_tokens),
                    chunk_frames,
                )
            return self.parallel.gather(x, layout.tokens)

        x = causal_pass(x, range(self.full_history_block_num), chunk_frames)
        if self.sink_recent_blocks:
            first, last = (
                chunk_frames[0] * frame_tokens,
                chunk_frames[-1] * frame_tokens,
            )
            if len(chunk_frames) == 1:
                kept, kept_frames = x[:, :first], [chunk_frames[0]]
            else:
                kept, kept_frames = torch.cat([x[:, :first], x[:, -last:]], dim=1), [
                    chunk_frames[0],
                    chunk_frames[-1],
                ]
            causal_pass(kept.contiguous(), self.sink_recent_blocks, kept_frames)
        return frames


EntryClass = MovieRollTransformer3DModel
