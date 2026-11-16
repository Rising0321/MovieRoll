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

"""Streaming MMDiT backbone of MovieRoll.

The model denoises a rolling window of noisy latent chunks conditioned on a clean history. HieraKV
(hierarchical KV-cache compression) splits the double-stream blocks into two groups, which must attend the
same way in training as at inference:

  * full-history blocks [0, full_history_block_num): every noisy chunk attends over the whole clean history;
  * sink-recent blocks [full_history_block_num, depth): every noisy window attends only over the sink chunk
    plus its own most recent clean chunk, which bounds the KV cache.

Training passes the history in as input tokens (cond_chunk_sizes) and denoises every window in one forward;
inference keeps it in the KV cache, one chunk per call.
"""

from typing import Dict, List, Optional, Tuple, Union

from einops import rearrange, repeat

import torch
import torch.nn as nn

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from movieroll.commons.streaming_config import StreamingConfig
from movieroll.models.transformers.modules.activation_layers import get_activation_layer
from movieroll.models.transformers.modules.attention import parallel_attention
from movieroll.models.transformers.modules.byt5_mapper import ByT5Mapper
from movieroll.models.transformers.modules.embed_layers import PatchEmbed, TimestepEmbedder, reorder_txt_token
from movieroll.models.transformers.modules.kv_cache import KVCacheManager
from movieroll.models.transformers.modules.mlp_layers import MLP, FinalLayer, unpatchify
from movieroll.models.transformers.modules.modulate_layers import ModulateDiT, apply_gate, modulate
from movieroll.models.transformers.modules.norm_layers import get_norm_layer
from movieroll.models.transformers.modules.rope import RoPEResampler
from movieroll.models.transformers.modules.streaming_utils import safe_get_rank
from movieroll.models.transformers.modules.token_refiner import SingleTokenRefiner
from movieroll.utils.communications import all_gather, get_padding_info
from movieroll.commons.parallel_states import get_sequence_parallel_state, nccl_info


def _sp_shard(x, sp_size, sp_rank, group):
    """This rank's slice of `x` along the token dim, and the padding info the all-to-alls need to undo it.

    Callers run `check_img_sp_compatibility` themselves, before sharding.
    """
    x = torch.chunk(x, sp_size, dim=1)[sp_rank]
    padding, total_lengths, pad_on_rank_first, pad_on_rank_last = get_padding_info(None, x, group)
    return x, {
        'padding': padding,
        'total_lengths': total_lengths,
        'pad_on_rank_first': pad_on_rank_first,
        'pad_on_rank_last': pad_on_rank_last,
    }


class MMDoubleStreamBlock(nn.Module):
    """Double-stream block of the streaming backbone, with fused `*_attn_qkv` projections.

    Three forward routes share the projections: `forward_training` (all windows at once, the *_parallel_*
    kernels), `forward_inference` (one chunk, history from the KV cache) and `forward_kv_recompute` (img-only
    block-wise causal (FBCA) re-encoding, one block per chunk, that returns the pre-RoPE KV).
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.heads_num = heads_num

        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = qk_norm_layer(head_dim, eps=1e-6, **factory_kwargs)
        self.img_attn_k_norm = qk_norm_layer(head_dim, eps=1e-6, **factory_kwargs)
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = qk_norm_layer(head_dim, eps=1e-6, **factory_kwargs)
        self.txt_attn_k_norm = qk_norm_layer(head_dim, eps=1e-6, **factory_kwargs)
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        """Concatenate separate `*_attn_q/k/v` weights into the fused `*_attn_qkv` this block expects.

        Checkpoints that store the three projections under separate keys load unchanged.
        """
        for modal in ('img', 'txt'):
            for suffix in ('weight', 'bias'):
                q_key = f'{prefix}{modal}_attn_q.{suffix}'
                k_key = f'{prefix}{modal}_attn_k.{suffix}'
                v_key = f'{prefix}{modal}_attn_v.{suffix}'
                qkv_key = f'{prefix}{modal}_attn_qkv.{suffix}'
                if q_key in state_dict and k_key in state_dict and v_key in state_dict:
                    state_dict[qkv_key] = torch.cat([
                        state_dict.pop(q_key),
                        state_dict.pop(k_key),
                        state_dict.pop(v_key),
                    ], dim=0)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward_training(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vec_txt: torch.Tensor,
        freqs_cis_q: tuple = None,
        freqs_cis_k: tuple = None,
        text_mask=None,
        attn_param=None,
        attn_mode=None,
        cond_img_seq_len=None,
        cond_latents_num_frames=None,
        cond_padding_info_dict=None,
        img_padding_info_dict=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training-regime forward: the clean history and every noisy window in one pass.

        `img` is [cond tokens | noisy tokens], the first `cond_img_seq_len` being the clean history; `txt` is the
        per-window text [B, num_windows * S, D]. `vec` modulates frame by frame, `vec_txt` window by window.
        Returns (img, txt).
        """
        assert cond_img_seq_len is not None and cond_latents_num_frames is not None, "cond_img_seq_len and cond_latents_num_frames are None"

        cond_chunk_sizes = attn_param['cond_chunk_sizes']
        noisy_chunk_sizes = attn_param['noisy_chunk_sizes']
        num_noisy_chunks = len(noisy_chunk_sizes)

        thw = attn_param['thw']

        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec_txt).chunk(6, dim=-1)

        img_modulated = modulate(
            self.img_norm1(img), shift=img_mod1_shift, scale=img_mod1_scale, thw=thw,
            cond_latents_num_frames=cond_latents_num_frames
        )

        img_q, img_k, img_v = self.img_attn_qkv(img_modulated).unflatten(-1, (3, self.heads_num, -1)).unbind(2)

        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # RoPE is applied inside parallel_attention

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale, num_chunks=num_noisy_chunks
        )

        txt_q, txt_k, txt_v = self.txt_attn_qkv(txt_modulated).unflatten(-1, (3, self.heads_num, -1)).unbind(2)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # cond_img and img are sequence-parallel sharded separately, so parallel_attention gathers them separately
        cond_img_q = cond_img_k = cond_img_v = None
        if cond_img_seq_len > 0:
            cond_img_q, img_q = img_q[:, :cond_img_seq_len], img_q[:, cond_img_seq_len:]
            cond_img_k, img_k = img_k[:, :cond_img_seq_len], img_k[:, cond_img_seq_len:]
            cond_img_v, img_v = img_v[:, :cond_img_seq_len], img_v[:, cond_img_seq_len:]

        attn, _ = parallel_attention(
            (cond_img_q, img_q, txt_q),
            (cond_img_k, img_k, txt_k),
            (cond_img_v, img_v, txt_v),
            attn_mode=attn_mode,
            text_mask=text_mask,
            freqs_cis_q=freqs_cis_q,
            freqs_cis_k=freqs_cis_k,
            cond_padding_info_dict=cond_padding_info_dict,
            img_padding_info_dict=img_padding_info_dict,
            cond_chunk_sizes=cond_chunk_sizes,
            noisy_chunk_sizes=noisy_chunk_sizes,
            sink_chunks=attn_param['sink_chunks'],
            window_has_sink=attn_param['window_has_sink'],
            window_has_recent=attn_param['window_has_recent'],
        )

        # The img length is derived from the txt length: under sequence parallelism img is sharded, txt is not.
        txt_seq_len = txt_q.shape[1]
        total_seq_len = attn.shape[1]
        img_seq_len = total_seq_len - txt_seq_len
        img_attn, txt_attn = attn[:, :img_seq_len].contiguous(), attn[:, img_seq_len:].contiguous()

        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate, thw=thw,
                               cond_latents_num_frames=cond_latents_num_frames)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale, thw=thw,
                    cond_latents_num_frames=cond_latents_num_frames
                )
            ),
            gate=img_mod2_gate, thw=thw,
            cond_latents_num_frames=cond_latents_num_frames,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate,
                               num_chunks=num_noisy_chunks)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale, num_chunks=num_noisy_chunks)),
            gate=txt_mod2_gate,
            num_chunks=num_noisy_chunks,
        )

        return img, txt

    def forward_inference(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vec_txt: torch.Tensor,
        freqs_cis_q: tuple = None,
        freqs_cis_k: tuple = None,
        text_mask=None,
        attn_param=None,
        attn_mode=None,
        return_kv=True,
        kv_cache=None,
        cond_padding_info_dict=None,
        img_padding_info_dict=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-regime forward of one window, the clean history coming from `kv_cache`.

        With `return_kv=True` the call only encodes `img` for the cache: the text stream is skipped and
        (img, None, (k, v)) is returned for the caller to store. Otherwise the window is denoised with its text
        and (img, txt) is returned.
        """

        need_txt_compute = not return_kv

        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)

        if need_txt_compute:
            assert vec_txt.shape[1] == 1, "vec_txt.shape[1] must be 1"
            (
                txt_mod1_shift,
                txt_mod1_scale,
                txt_mod1_gate,
                txt_mod2_shift,
                txt_mod2_scale,
                txt_mod2_gate,
            ) = self.txt_mod(vec_txt).chunk(6, dim=-1)
        thw = attn_param['thw']
        img_modulated = modulate(
            self.img_norm1(img), shift=img_mod1_shift, scale=img_mod1_scale, thw=thw,
        )

        img_q, img_k, img_v = self.img_attn_qkv(img_modulated).unflatten(-1, (3, self.heads_num, -1)).unbind(2)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        if need_txt_compute:
            txt_modulated = self.txt_norm1(txt)
            txt_modulated = modulate(
                txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale, num_chunks=1
            )

            txt_q, txt_k, txt_v = self.txt_attn_qkv(txt_modulated).unflatten(-1, (3, self.heads_num, -1)).unbind(2)
            txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
            txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)
        else:
            txt_q = None
            txt_k = None
            txt_v = None

        attn, kv_cache_return = parallel_attention(
            (None, img_q, txt_q),
            (None, img_k, txt_k),
            (None, img_v, txt_v),
            attn_mode=attn_mode,
            text_mask=text_mask,
            freqs_cis_q=freqs_cis_q,
            freqs_cis_k=freqs_cis_k,
            kv_cache=kv_cache,
            return_kv=return_kv,
            need_txt_compute=need_txt_compute,
            cond_padding_info_dict=cond_padding_info_dict,
            img_padding_info_dict=img_padding_info_dict,
        )

        txt_seq_len = txt_q.shape[1] if txt_q is not None else 0
        total_seq_len = attn.shape[1]
        img_seq_len = total_seq_len - txt_seq_len
        img_attn, txt_attn = attn[:, :img_seq_len].contiguous(), attn[:, img_seq_len:].contiguous()
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate, thw=thw)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale, thw=thw,
                )
            ),
            gate=img_mod2_gate, thw=thw,
        )

        if need_txt_compute:
            txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate,
                                   num_chunks=1)
            txt = txt + apply_gate(
                self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale, num_chunks=1)),
                gate=txt_mod2_gate,
                num_chunks=1,
            )
        else:
            txt = None

        if return_kv:
            return img, txt, kv_cache_return
        else:
            return img, txt

    def forward_kv_recompute(
        self,
        img: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis_q: tuple = None,
        freqs_cis_k: tuple = None,
        attn_param=None,
        noisy_chunk_sizes=None,
        img_padding_info_dict=None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """KV recompute block forward: img-only block-wise causal (FBCA) attention, returns the pre-RoPE KV."""
        thw = attn_param['thw']

        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)

        img_modulated = modulate(
            self.img_norm1(img), shift=img_mod1_shift, scale=img_mod1_scale, thw=thw,
        )

        img_q, img_k, img_v = self.img_attn_qkv(img_modulated).unflatten(-1, (3, self.heads_num, -1)).unbind(2)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        attn, kv_cache_return = parallel_attention(
            (None, img_q, None),
            (None, img_k, None),
            (None, img_v, None),
            attn_mode="kv_recompute_causal",
            freqs_cis_q=freqs_cis_q,
            freqs_cis_k=freqs_cis_k,
            return_kv=True,
            need_txt_compute=False,
            noisy_chunk_sizes=noisy_chunk_sizes,
            img_padding_info_dict=img_padding_info_dict,
        )

        img = img + apply_gate(self.img_attn_proj(attn), gate=img_mod1_gate, thw=thw)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale, thw=thw,
                )
            ),
            gate=img_mod2_gate, thw=thw,
        )
        return img, kv_cache_return

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        vec_txt: torch.Tensor,
        freqs_cis_q: tuple = None,
        freqs_cis_k: tuple = None,
        text_mask=None,
        attn_param=None,
        training=True,
        attn_mode=None,
        return_kv=True,
        kv_cache=None,
        cond_img_seq_len=None,
        cond_latents_num_frames=None,
        cond_padding_info_dict=None,
        img_padding_info_dict=None,
        kv_recompute=False,
        noisy_chunk_sizes=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # KV recompute goes through the block's __call__/forward so the FSDP all-gather hook fires
        if kv_recompute:
            return self.forward_kv_recompute(
                img,
                vec,
                freqs_cis_q,
                freqs_cis_k,
                attn_param=attn_param,
                noisy_chunk_sizes=noisy_chunk_sizes,
                img_padding_info_dict=img_padding_info_dict,
            )
        if training:
            output = self.forward_training(
                img,
                txt,
                vec,
                vec_txt,
                freqs_cis_q,
                freqs_cis_k,
                text_mask,
                attn_param,
                attn_mode=attn_mode,
                cond_img_seq_len=cond_img_seq_len,
                cond_latents_num_frames=cond_latents_num_frames,
                cond_padding_info_dict=cond_padding_info_dict,
                img_padding_info_dict=img_padding_info_dict,
            )
        else:
            output = self.forward_inference(
                img,
                txt,
                vec,
                vec_txt,
                freqs_cis_q,
                freqs_cis_k,
                text_mask,
                attn_param,
                attn_mode=attn_mode,
                return_kv=return_kv,
                kv_cache=kv_cache,
                cond_padding_info_dict=cond_padding_info_dict,
                img_padding_info_dict=img_padding_info_dict,
            )
        return output


class MovieRollDiffusionTransformer(ModelMixin, ConfigMixin):

    ignore_for_config = ["streaming_config"]

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm_type: str = "rms",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        text_states_dim: int = 4096,
        rope_theta: int = 256,
        streaming_config: StreamingConfig = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta

        self.mm_double_blocks_depth = mm_double_blocks_depth
        self.streaming_config = streaming_config
        if self.streaming_config is None:
            raise ValueError("streaming_config is required for the streaming transformer")
        # HieraKV: blocks [0, full_history_block_num) see the full history, the rest sink + recent only.
        self.full_history_block_num = self.streaming_config.full_history_block_num
        assert 0 <= self.full_history_block_num <= self.mm_double_blocks_depth, \
            "full_history_block_num must be less than or equal to the number of double blocks"
        self.sink_chunks = self.streaming_config.sink_chunks
        self.recent_chunks = self.streaming_config.recent_chunks
        self.sink_recent_block_indices = list(range(self.full_history_block_num, self.mm_double_blocks_depth))
        if safe_get_rank() == 0:
            print(f"[HieraKV] full_history_block_num={self.full_history_block_num}, "
                  f"sink_recent_block_indices={self.sink_recent_block_indices}, sink_chunks={self.sink_chunks}, "
                  f"recent_chunks={self.recent_chunks}")
        self.rope_resampler = RoPEResampler(streaming_config=self.streaming_config)
        self.kv_cache_manager = KVCacheManager(streaming_config=self.streaming_config)

        self.byt5_in = ByT5Mapper(
            in_dim=1472,
            out_dim=2048,
            hidden_dim=2048,
            out_dim1=hidden_size)

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        self.txt_in = SingleTokenRefiner(
            text_states_dim,
            hidden_size,
            heads_num,
            depth=2,
            **factory_kwargs,
        )

        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for d_block_index in range(mm_double_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        self.attn_param = {}
        # 0: qwen feature
        # 1: byt5 feature
        self.cond_type_embedding = nn.Embedding(3, self.hidden_size)
        self.cond_type_embedding.weight.data.fill_(0)

    def check_img_sp_compatibility(self, img, sp_size):
        if img.shape[1] % sp_size != 0:
            n_token = img.shape[1]
            if n_token <= (n_token // sp_size + 1) * (sp_size - 1):
                raise ValueError(f"Token {n_token} is too short for SP {sp_size}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        return_dict: bool = False,
        extra_kwargs=None,
        # streaming args
        timestep_txt=None,
        cond_chunk_sizes=None,
        noisy_chunk_sizes=None,
        current_frame_index=0,
        return_kv=False,
        kv_recompute=False,
        recompute_chunk_sizes=None,
        training: Optional[bool] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Predict the velocity of the noisy latent windows, conditioned on the clean history and the text.

        `hidden_states` is [B, 2 * in_channels + 1, T, H, W]: in training the clean history (cond_chunk_sizes
        latent frames per chunk) followed by the noisy windows, at inference the window alone. `timestep` is
        [B, T] per frame, `timestep_txt` [B, num_windows] per window. `text_states` / `encoder_attention_mask`
        are shared ([B, S, D] / [B, S]) or per window ([B, num_windows, S, D] / [B, num_windows, S]), and
        `extra_kwargs` carries the matching byt5 pair. `noisy_chunk_sizes` lists the latent frames of the noisy
        chunks, one sub-list per window -- in a flat list every entry is a window of its own.
        `current_frame_index` is the first latent frame of the window, feeding the inference RoPE and the
        KV-cache write positions.

        Returns (velocity of the noisy frames [B, out_channels, T_noisy, H, W], None), or (None, None) with
        `return_kv` -- inference only, since the training blocks write no KV and return (None, None)
        regardless -- or the new current_frame_index with `kv_recompute`.
        """
        assert not return_dict, "return_dict is not supported."


        # `training` picks the *_parallel_* kernels over chunk_inference_attention, and makes the clean history
        # come from the input tokens (cond_chunk_sizes) instead of the KV cache -- forward_training has no
        # kv_cache argument at all. Passing it explicitly decouples the regime from whether gradients are on,
        # which DMD needs: it supervises the student in the regime inference will run, with gradients on.
        if training is None:
            training = torch.is_grad_enabled()
        else:
            training = bool(training)
        # KV recompute entry: goes through the model's __call__/forward so the root FSDP all-gather hook fires
        if kv_recompute:
            return self._forward_kv_recompute(hidden_states, recompute_chunk_sizes)

        need_txt_compute = not return_kv

        device = hidden_states.device

        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = text_states
        bs, _, ot, oh, ow = x.shape

        t_txt = timestep_txt

        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        # thw goes to modulate / apply_gate and the final layer, never to attention; only H and W are used.
        self.attn_param['thw'] = [tt, th, tw]

        if cond_chunk_sizes is None:
            cond_chunk_sizes = []
        assert noisy_chunk_sizes is not None and len(noisy_chunk_sizes) > 0, "noisy_chunk_sizes must be provided"
        cond_latents_num_frames = sum(cond_chunk_sizes)
        # nested [[chunk1, chunk2, ...], [...], ...] -> total frames per window [W1, W2, W3]
        if isinstance(noisy_chunk_sizes[0], list):
            noisy_chunk_sizes = [sum(sub_chunk_sizes) for sub_chunk_sizes in noisy_chunk_sizes]
        regular_chunk_size = noisy_chunk_sizes[-1]
        noisy_latents_num_frames = sum(noisy_chunk_sizes)
        overlap_latents_num_frames = noisy_latents_num_frames - regular_chunk_size if training else 0

        # Frame counts of the physical chunks / windows, kept before the conversion to tokens:
        #   - cond_chunk_frames[0]   = frames of the sink physical chunk
        #   - cond_chunk_frames[-1]  = frames of a regular physical chunk (= recent frames)
        #   - noisy_window_frames[i] = total frames of window i (window_size physical chunks)
        cond_chunk_frames = list(cond_chunk_sizes)
        noisy_window_frames = list(noisy_chunk_sizes)

        # frames -> tokens: th * tw tokens per latent frame
        cond_chunk_tokens = [frames * th * tw for frames in cond_chunk_frames]
        noisy_window_tokens = [frames * th * tw for frames in noisy_window_frames]

        if not training and not self.kv_cache_manager.is_init:
            name_list = [f"double_block_{index+1}" for index in range(self.mm_double_blocks_depth)]
            self.kv_cache_manager.init_kv_cache(name_list, [th, tw])

        # Before encoding a new chunk, trim every sink-recent cache back to the sink (popping the stale recent)
        # so that chunk attends over the sink only, as training does when it drops the middle.
        if (not training) and return_kv and len(self.sink_recent_block_indices) > 0:
            for index in self.sink_recent_block_indices:
                self.kv_cache_manager.trim_to_sink(
                    f"double_block_{index+1}", self.sink_chunks
                )

        if training:
            kv_cache_frame_indices = None
        else:
            # at inference the cond frame count comes from the KV cache manager
            kv_cache_frame_indices = self.kv_cache_manager.get_kv_cache_frame_indices()
            cond_latents_num_frames = self.kv_cache_manager.get_cached_cond_latents_num_frames()
            if cond_latents_num_frames > current_frame_index:
                raise ValueError(f"cond_latents_num_frames {cond_latents_num_frames} should be less than or equal to current_frame_index {current_frame_index}")

        num_noisy_chunks = len(noisy_chunk_sizes)
        num_valid_frames = cond_latents_num_frames + noisy_latents_num_frames if training else noisy_latents_num_frames

        img = self.img_in(img)

        # The resampler must run BEFORE the KV cache is updated, i.e. with the old kv_cache_frame_indices.
        # For a full-history block the noisy freqs start at current_frame_index.
        full_history_freqs_cis_q, full_history_freqs_cis_k = self.rope_resampler(
            [th, tw],
            cond_latents_num_frames=cond_latents_num_frames,
            noisy_latents_num_frames=noisy_latents_num_frames,
            overlap_latents_num_frames=overlap_latents_num_frames,
            noisy_chunk_sizes_tokens=noisy_window_tokens,
            current_frame_index=current_frame_index,
            kv_cache_frame_indices=kv_cache_frame_indices,
            training=training,
            rope_dim_list=self.rope_dim_list,
            head_dim=self.hidden_size // self.heads_num,
            rope_theta=self.rope_theta,
            device=device,
        )
        noisy_freqs_cis_q, noisy_freqs_cis_k = full_history_freqs_cis_q, full_history_freqs_cis_k

        # Sink-recent blocks in training: `_sink_recent_window_layout` decides each window's cond visibility on
        # its own, so layouts like "first window has no cond, later windows have sink/recent" stay aligned with
        # the inference rollout window by window.
        sink_recent_window_has_sink = None
        sink_recent_window_has_recent = None
        sink_recent_frame_ranges = None
        if (len(self.sink_recent_block_indices) > 0
                and training and cond_latents_num_frames > 0):
            sink_recent_window_has_sink, sink_recent_window_has_recent, sink_recent_frame_ranges = (
                self._sink_recent_window_layout(
                    sink_frames=cond_chunk_frames[0],
                    recent_frames=cond_chunk_frames[-1],
                    noisy_window_frames=noisy_window_frames,
                    cond_latents_num_frames=cond_latents_num_frames,
                    overlap_latents_num_frames=overlap_latents_num_frames,
                )
            )

        # slot RoPE of the sink-recent blocks (per-window sink + recent)
        sink_recent_freqs_cis_q = sink_recent_freqs_cis_k = None
        if len(self.sink_recent_block_indices) > 0:
            if training:
                if cond_latents_num_frames > 0:
                    # the sink uses PHYSICAL chunk frames and noisy uses WHOLE window frames; the per-window
                    # flags decide where each window's self slot starts (0 / sink_f / sink_f+rec_f).
                    sink_recent_freqs_cis_q, sink_recent_freqs_cis_k = self.rope_resampler.resample_training_slot(
                        [th, tw],
                        sink_frames=cond_chunk_frames[0],
                        recent_frames=cond_chunk_frames[-1],
                        noisy_window_frames=noisy_window_frames,
                        window_has_sink=sink_recent_window_has_sink,
                        window_has_recent=sink_recent_window_has_recent,
                        rope_dim_list=self.rope_dim_list,
                        head_dim=self.hidden_size // self.heads_num,
                        rope_theta=self.rope_theta,
                        device=device,
                    )
                else:
                    # without any cond (pure noisy) the sink-recent blocks fall back to the noisy-only RoPE
                    sink_recent_freqs_cis_q, sink_recent_freqs_cis_k = noisy_freqs_cis_q, noisy_freqs_cis_k
            else:
                # inference: the densely packed RoPE of the sink-recent blocks' own (already trimmed) cache
                sink_recent_name = f"double_block_{self.full_history_block_num + 1}"
                sink_recent_cache_frame_indices = self.kv_cache_manager.get_kv_cache_frame_indices(sink_recent_name)
                sink_recent_cond_frames = self.kv_cache_manager.get_cached_cond_latents_num_frames(sink_recent_name)
                sink_recent_freqs_cis_q, sink_recent_freqs_cis_k = self.rope_resampler(
                    [th, tw],
                    cond_latents_num_frames=sink_recent_cond_frames,
                    noisy_latents_num_frames=noisy_latents_num_frames,
                    overlap_latents_num_frames=0,
                    noisy_chunk_sizes_tokens=noisy_window_tokens,
                    current_frame_index=current_frame_index,
                    kv_cache_frame_indices=sink_recent_cache_frame_indices,
                    training=False,
                    rope_dim_list=self.rope_dim_list,
                    head_dim=self.hidden_size // self.heads_num,
                    rope_theta=self.rope_theta,
                    device=device,
                )
        cond_img_seq_len = cond_latents_num_frames * th * tw
        img_padding_info_dict = None
        cond_padding_info_dict = None
        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            # Training shards cond img and noisy img separately, so every rank holds part of both; at inference
            # the history is in the KV cache instead.
            if training and cond_latents_num_frames > 0:
                cond_img, img = img[:, :cond_img_seq_len], img[:, cond_img_seq_len:]
                self.check_img_sp_compatibility(cond_img, sp_size)
                self.check_img_sp_compatibility(img, sp_size)
                cond_img, cond_padding_info_dict = _sp_shard(cond_img, sp_size, sp_rank, nccl_info.sp_group)
                cond_img_seq_len = cond_img.shape[1]   # the local shard length, not the full one
                img, img_padding_info_dict = _sp_shard(img, sp_size, sp_rank, nccl_info.sp_group)
                img = torch.cat([cond_img, img], dim=1)

            else:
                self.check_img_sp_compatibility(img, sp_size)
                img, img_padding_info_dict = _sp_shard(img, sp_size, sp_rank, nccl_info.sp_group)

        assert t.shape[1] == num_valid_frames, f"t.shape[1]: {t.shape[1]} != num_valid_frames: {num_valid_frames}"
        t = rearrange(t, 'b n -> (b n)')

        vec = self.time_in(t)

        if need_txt_compute:
            # In training text_states is 4D, one entry per window; at inference it is 3D and shared by every
            # window, so it is repeated. Either way txt ends up [B * num_noisy_chunks, text_len, dim].
            if txt.dim() == 4:
                assert txt.shape[1] == num_noisy_chunks, (
                    f"txt.shape[1]={txt.shape[1]} != num_noisy_chunks={num_noisy_chunks}"
                )
                txt = rearrange(txt, 'b n s d -> (b n) s d')
                text_mask = rearrange(text_mask, 'b n s -> (b n) s')
            else:
                txt = repeat(txt, 'b s d -> (b n) s d', n=num_noisy_chunks)
                text_mask = repeat(text_mask, 'b s -> (b n) s', n=num_noisy_chunks)
            # t_txt has one timestep per noisy chunk, t has one per frame (cond chunks included)
            assert t_txt.shape[1] == num_noisy_chunks, f"t_txt.shape[1]: {t_txt.shape[1]} != num_noisy_chunks: {num_noisy_chunks}"
            t_txt = rearrange(t_txt, 'b n -> (b n)')
            vec_txt = self.time_in(t_txt)

            txt = self.txt_in(txt, t_txt, text_mask)
            cond_emb = self.cond_type_embedding(torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long))
            txt = txt + cond_emb

            byt5_text_states = extra_kwargs["byt5_text_states"]
            byt5_text_mask = extra_kwargs["byt5_text_mask"]

            if byt5_text_states.dim() == 4:
                # [B, N, S_b, D_b]: flatten the window dim before projecting
                byt5_txt = self.byt5_in(rearrange(byt5_text_states, 'b n s d -> (b n) s d'))
                byt5_text_mask = rearrange(byt5_text_mask, 'b n s -> (b n) s')
            else:
                byt5_txt = self.byt5_in(byt5_text_states)
                byt5_txt = repeat(byt5_txt, 'b s d -> (b n) s d', n=num_noisy_chunks)
                byt5_text_mask = repeat(byt5_text_mask, 'b s -> (b n) s', n=num_noisy_chunks)

            cond_emb = self.cond_type_embedding(torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long))
            byt5_txt = byt5_txt + cond_emb

            txt, text_mask = reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask)

        vec = rearrange(vec, '(b n) d -> b n d', b=bs)  # [B, num_frames, D], one entry per frame

        if need_txt_compute:
            txt = rearrange(txt, '(b n) s d -> b (n s) d', b=bs)
            text_mask = rearrange(text_mask, '(b n) s -> b (n s)', b=bs)
            vec_txt = rearrange(vec_txt, '(b n) d -> b n d', b=bs)
        else:
            txt = text_mask = vec_txt = None

        # These attn_param entries hold TOKEN counts, and parallel_attention reads them under these exact key
        # names. The text modulation only uses len().
        self.attn_param['cond_chunk_sizes'] = cond_chunk_tokens
        self.attn_param['noisy_chunk_sizes'] = noisy_window_tokens
        self.attn_param['sink_chunks'] = self.sink_chunks
        # Per-window flags of the sink-recent blocks, reset here every forward so a stale list of a different
        # length cannot leak, and only filled when the middle is dropped (cond > 0). With cond == 0 they stay
        # None and the kernel fills them with all-True.
        self.attn_param['window_has_sink'] = None
        self.attn_param['window_has_recent'] = None

        full_history_training_attn_mode = 'full_history_parallel_attention'
        full_history_inference_attn_mode = 'chunk_inference_attention'
        sink_recent_training_attn_mode = 'sink_recent_parallel_attention'
        sink_recent_inference_attn_mode = 'chunk_inference_attention'

        # --------------------- Pass through DiT blocks ------------------------
        for index, block in enumerate(self.double_blocks):
            name = f"double_block_{index+1}"
            is_sink_recent_block = index in self.sink_recent_block_indices

            # full-history -> sink-recent boundary in training: drop the middle cond tokens and reset the slot
            # RoPE, keeping the sink (cond physical chunk 0, if any window uses it) plus, for every window with
            # window_has_recent, the single physical chunk right before it.
            # attn_param is rewritten here, half way through the loop, so each block must be called with a
            # snapshot of it: under activation checkpointing a block re-runs its forward during backward with
            # the arguments it was given, and the live dict would hold the post-drop layout by then -- and
            # likewise the entries written at the top of this forward are overwritten by the next forward.
            if (training and index == self.full_history_block_num
                    and cond_latents_num_frames > 0 and len(self.sink_recent_block_indices) > 0):
                img, vec, cond_img_seq_len, cond_padding_info_dict, cond_latents_num_frames = (
                    self._drop_middle_history(
                        img, vec, cond_img_seq_len, cond_padding_info_dict,
                        cond_latents_num_frames, th, tw,
                        frame_ranges=sink_recent_frame_ranges,
                    )
                )
                tt = cond_latents_num_frames + noisy_latents_num_frames
                self.attn_param['thw'] = [tt, th, tw]
                # The kept cond chunk sizes come straight from frame_ranges, so they match the post-drop cond
                # layout [sink?, recents...] that the per-window flags below describe.
                self.attn_param['cond_chunk_sizes'] = [
                    (fe - fs) * th * tw for fs, fe in sink_recent_frame_ranges
                ]
                self.attn_param['window_has_sink'] = sink_recent_window_has_sink
                self.attn_param['window_has_recent'] = sink_recent_window_has_recent

            if is_sink_recent_block:
                freqs_cis_q = sink_recent_freqs_cis_q
                freqs_cis_k = sink_recent_freqs_cis_k
                attn_mode = sink_recent_training_attn_mode if training else sink_recent_inference_attn_mode
            else:
                freqs_cis_q = full_history_freqs_cis_q
                freqs_cis_k = full_history_freqs_cis_k
                attn_mode = full_history_training_attn_mode if training else full_history_inference_attn_mode

            # None while the layer's cache holds no data; forward_training ignores the argument either way
            kv_cache = self.kv_cache_manager.get_kv_cache(name, current_batch_size=bs)

            double_block_output = block(
                img=img, txt=txt, vec=vec, vec_txt=vec_txt,
                freqs_cis_q=freqs_cis_q, freqs_cis_k=freqs_cis_k,
                text_mask=text_mask,
                attn_param=dict(self.attn_param),  # per-call snapshot, see above
                training=training,
                attn_mode=attn_mode,
                return_kv=return_kv,
                kv_cache=kv_cache,
                cond_img_seq_len=cond_img_seq_len,
                cond_latents_num_frames=cond_latents_num_frames,
                cond_padding_info_dict=cond_padding_info_dict,
                img_padding_info_dict=img_padding_info_dict,
            )

            if return_kv:
                img, txt, kv_cache = double_block_output
                self.kv_cache_manager.update_kv_cache(
                    name, kv_cache[0], kv_cache[1],
                    start_index=current_frame_index, end_index=current_frame_index+noisy_latents_num_frames
                )
            else:
                img, txt = double_block_output

        # return_kv only refreshes the KV cache, so the unpatchify below is not needed
        if return_kv:
            return (None, None)

        img_seq_len = img.shape[1]

        # Numerically a no-op for img, but load-bearing for autograd: concatenating txt keeps the last block's
        # text branch in the graph, so its parameters get zero grads instead of None.
        x = torch.cat((img, txt), 1) if txt is not None else img
        img = x[:, :img_seq_len, ...]

        # the final layer is an mlp, so truncating the cond part does not affect gradients
        if training and cond_latents_num_frames > 0:
            tt = noisy_latents_num_frames
            vec = vec[:, cond_latents_num_frames:]
            img = img[:, cond_img_seq_len:]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec, thw=[tt, th, tw])  # (N, T, patch_size ** 2 * out_channels)

        if get_sequence_parallel_state():
            img = all_gather(img, dim=1, group=nccl_info.sp_group)

        img = unpatchify(img, tt, th, tw, self.patch_size, self.unpatchify_channels)

        return (img, None)

    def _gather_full_seq(self, x_sp):
        """Restore an SP-sharded [B, S_local, H, D] back to the full [B, S_total, H, D].

        all_gather concatenates the real per-rank shapes in rank order, so ragged shards need no pad/trim.
        """
        if not get_sequence_parallel_state():
            return x_sp
        return all_gather(x_sp, dim=1, group=nccl_info.sp_group)

    def _sink_recent_window_layout(
        self, sink_frames, recent_frames, noisy_window_frames,
        cond_latents_num_frames, overlap_latents_num_frames,
    ):
        """Per-window cond visibility of the sink-recent blocks in training, and the cond frames it keeps.

        With win_start_w = (cond_frames - overlap_frames) + sum_{j<w} window_j_frames, each window is judged on
        its own (no monotonicity assumed), recent implying sink because recent lies past the sink region:
          window_has_recent[w] = win_start_w - recent_frames >= sink_frames
          window_has_sink[w]   = win_start_w >= sink_frames or window_has_recent[w]
        This is the single source of truth for that cond layout.

        Returns (window_has_sink, window_has_recent, frame_ranges): two List[bool] of length num_window, and the
        kept (start, end) frame ranges ordered like the cond sequence [sink?, recents...].
        """
        noisy_start_in_cond = cond_latents_num_frames - overlap_latents_num_frames
        window_has_sink = []
        window_has_recent = []
        recent_ranges = []
        win_start = noisy_start_in_cond
        for win_f in noisy_window_frames:
            has_recent = (win_start - recent_frames) >= sink_frames
            has_sink = (win_start >= sink_frames) or has_recent
            window_has_sink.append(has_sink)
            window_has_recent.append(has_recent)
            if has_recent:
                recent_ranges.append((win_start - recent_frames, win_start))
            win_start += win_f
        frame_ranges = ([(0, sink_frames)] if any(window_has_sink) else []) + recent_ranges
        return window_has_sink, window_has_recent, frame_ranges

    def _drop_middle_history(
        self, img, vec, cond_img_seq_len, cond_padding_info_dict,
        cond_latents_num_frames, th, tw, frame_ranges,
    ):
        """Keep only the cond frames in `frame_ranges` (from `_sink_recent_window_layout`), dropping the middle.

        Returns (new_img, new_vec, new_cond_img_seq_len, new_cond_padding_info_dict, kept_cond_frames).
        """
        assert len(frame_ranges) > 0, "[HieraKV] frame_ranges is empty (cond > 0 but no window uses the cond)"
        frame_tokens = th * tw
        kept_cond_frames = sum(fe - fs for fs, fe in frame_ranges)

        # img is SP-aware: gather the full cond, slice it, re-shard
        cond_img_sp = img[:, :cond_img_seq_len]
        noisy_img_sp = img[:, cond_img_seq_len:]
        full_cond = self._gather_full_seq(cond_img_sp)  # [B, full_cond_tokens, H, D]
        kept_cond_full = torch.cat(
            [full_cond[:, fs * frame_tokens: fe * frame_tokens] for fs, fe in frame_ranges], dim=1
        )

        new_cond_padding_info_dict = cond_padding_info_dict
        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            self.check_img_sp_compatibility(kept_cond_full, sp_size)
            kept_cond_local, new_cond_padding_info_dict = _sp_shard(
                kept_cond_full, sp_size, sp_rank, nccl_info.sp_group
            )
        else:
            kept_cond_local = kept_cond_full

        new_cond_img_seq_len = kept_cond_local.shape[1]
        new_img = torch.cat([kept_cond_local, noisy_img_sp], dim=1)

        # vec is frame-wise and never SP-sharded
        vec_cond = vec[:, :cond_latents_num_frames]
        vec_noisy = vec[:, cond_latents_num_frames:]
        kept_vec_cond = torch.cat(
            [vec_cond[:, fs:fe] for fs, fe in frame_ranges], dim=1
        )
        new_vec = torch.cat([kept_vec_cond, vec_noisy], dim=1)

        return new_img, new_vec, new_cond_img_seq_len, new_cond_padding_info_dict, kept_cond_frames

    @torch.no_grad()
    def _forward_kv_recompute(
        self,
        hidden_states: torch.Tensor,
        recompute_chunk_sizes: List[int],
    ) -> int:
        """Re-encode the selected chunks at timestep 0, without text, through block-wise causal (FBCA) attention, and
        replace every block's cached KV with the result.

        hidden_states is [B, C, T_total, H, W], the selected chunks concatenated along time, and
        recompute_chunk_sizes their latent frame counts. Returns the new current_frame_index, i.e. the total
        number of frames now cached.
        """
        assert self.kv_cache_manager.is_init, "kv_cache_manager must be initialized before recompute"

        # Clear every stale cache before rebuilding, keeping the initialized state. Both block groups must be
        # rebuilt from THIS hidden_states: context noise may have re-sampled the sink, so an old sink-recent
        # cache would correspond to a different version of the input than the full-history one.
        self.kv_cache_manager.soft_clear()

        device = hidden_states.device
        bs, _, ot, oh, ow = hidden_states.shape

        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        noisy_latents_num_frames = sum(recompute_chunk_sizes)
        assert noisy_latents_num_frames == tt, (
            f"sum(recompute_chunk_sizes)={noisy_latents_num_frames} != patchified frames tt={tt}"
        )
        # frames -> tokens
        chunk_sizes_tokens = [cs * th * tw for cs in recompute_chunk_sizes]

        img = self.img_in(hidden_states)

        # compact RoPE: kv_cache_frame_indices=[], positions start densely packed at 0
        freqs_cis_q, freqs_cis_k = self.rope_resampler(
            [th, tw],
            cond_latents_num_frames=0,
            noisy_latents_num_frames=noisy_latents_num_frames,
            overlap_latents_num_frames=0,
            noisy_chunk_sizes_tokens=chunk_sizes_tokens,
            current_frame_index=0,
            kv_cache_frame_indices=[],
            training=False,
            rope_dim_list=self.rope_dim_list,
            head_dim=self.hidden_size // self.heads_num,
            rope_theta=self.rope_theta,
            device=device,
        )

        zero_t = torch.zeros(bs * tt, device=device, dtype=hidden_states.dtype)
        vec = self.time_in(zero_t)
        vec = rearrange(vec, '(b n) d -> b n d', b=bs)

        attn_param = dict(self.attn_param)
        attn_param['thw'] = [tt, th, tw]
        attn_param['cond_chunk_sizes'] = []
        attn_param['noisy_chunk_sizes'] = chunk_sizes_tokens

        img_padding_info_dict = None
        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            self.check_img_sp_compatibility(img, sp_size)
            img, img_padding_info_dict = _sp_shard(img, sp_size, sp_rank, nccl_info.sp_group)

        # new frame indices (densely packed, one entry per selected chunk)
        new_frame_indices = []
        offset = 0
        for cs in recompute_chunk_sizes:
            new_frame_indices.append((offset, offset + cs))
            offset += cs

        # Rebuild the full-history caches [0, full_history_block_num) from the compressed full history.
        for index in range(self.full_history_block_num):
            block = self.double_blocks[index]
            name = f"double_block_{index+1}"

            # go through the block's __call__/forward(kv_recompute=True) so the FSDP all-gather hook fires
            img, kv_cache = block(
                img=img,
                txt=None,
                vec=vec,
                vec_txt=None,
                freqs_cis_q=freqs_cis_q,
                freqs_cis_k=freqs_cis_k,
                attn_param=attn_param,
                kv_recompute=True,
                noisy_chunk_sizes=chunk_sizes_tokens,
                img_padding_info_dict=img_padding_info_dict,
            )

            self.kv_cache_manager.replace_kv_cache(
                name, kv_cache[0], kv_cache[1], new_frame_indices
            )

        # The sink-recent blocks keep only the sink + recent of THIS pass through the full-history blocks, so
        # both block groups end up corresponding to exactly the same input.
        if len(self.sink_recent_block_indices) > 0:
            self._recompute_sink_recent_cache(
                post_full_history_img=img,
                recompute_chunk_sizes=recompute_chunk_sizes,
                chunk_sizes_tokens=chunk_sizes_tokens,
                th=th, tw=tw, bs=bs, device=device,
                new_total_frames=noisy_latents_num_frames,
            )

        return noisy_latents_num_frames

    @torch.no_grad()
    def _recompute_sink_recent_cache(
        self,
        post_full_history_img: torch.Tensor,
        recompute_chunk_sizes: List[int],
        chunk_sizes_tokens: List[int],
        th: int, tw: int, bs: int, device,
        new_total_frames: int,
    ):
        """Rebuild the sink-recent caches from this pass's hidden states after the full-history blocks."""
        sink_frames = recompute_chunk_sizes[0]
        sink_tokens = chunk_sizes_tokens[0]
        recent_frames = recompute_chunk_sizes[-1]
        recent_tokens = chunk_sizes_tokens[-1]

        # Under SP restore the full sequence first, then take the first and last physical chunk. The selected
        # chunks in between belong to the full-history blocks only.
        if get_sequence_parallel_state():
            img_full = all_gather(post_full_history_img, dim=1, group=nccl_info.sp_group)
        else:
            img_full = post_full_history_img

        if len(recompute_chunk_sizes) == 1:
            sink_recent_img = img_full[:, :sink_tokens].contiguous()
            sink_recent_chunk_frames = [sink_frames]
            sink_recent_chunk_tokens = [sink_tokens]
            sink_recent_frame_indices = [(0, sink_frames)]
        else:
            sink_recent_img = torch.cat(
                (
                    img_full[:, :sink_tokens],
                    img_full[:, -recent_tokens:],
                ),
                dim=1,
            ).contiguous()
            sink_recent_chunk_frames = [sink_frames, recent_frames]
            sink_recent_chunk_tokens = [sink_tokens, recent_tokens]
            sink_recent_frame_indices = [
                (0, sink_frames),
                (new_total_frames - recent_frames, new_total_frames),
            ]

        sink_recent_total_frames = sum(sink_recent_chunk_frames)
        sink_recent_freqs_cis_q, sink_recent_freqs_cis_k = self.rope_resampler(
            [th, tw],
            cond_latents_num_frames=0,
            noisy_latents_num_frames=sink_recent_total_frames,
            overlap_latents_num_frames=0,
            noisy_chunk_sizes_tokens=sink_recent_chunk_tokens,
            current_frame_index=0,
            kv_cache_frame_indices=[],
            training=False,
            rope_dim_list=self.rope_dim_list,
            head_dim=self.hidden_size // self.heads_num,
            rope_theta=self.rope_theta,
            device=device,
        )

        zero_t = torch.zeros(
            bs * sink_recent_total_frames, device=device, dtype=sink_recent_img.dtype
        )
        sink_recent_vec = rearrange(self.time_in(zero_t), '(b n) d -> b n d', b=bs)

        sink_recent_attn_param = dict(self.attn_param)
        sink_recent_attn_param['thw'] = [sink_recent_total_frames, th, tw]
        sink_recent_attn_param['cond_chunk_sizes'] = []
        sink_recent_attn_param['noisy_chunk_sizes'] = sink_recent_chunk_tokens

        img_padding_info_dict = None
        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            self.check_img_sp_compatibility(sink_recent_img, sp_size)
            sink_recent_img, img_padding_info_dict = _sp_shard(
                sink_recent_img, sp_size, sp_rank, nccl_info.sp_group
            )

        # Two-chunk causal attention is exactly: the sink sees itself, the recent sees sink + self.
        for index in self.sink_recent_block_indices:
            block = self.double_blocks[index]
            name = f"double_block_{index+1}"
            sink_recent_img, kv_cache = block(
                img=sink_recent_img,
                txt=None,
                vec=sink_recent_vec,
                vec_txt=None,
                freqs_cis_q=sink_recent_freqs_cis_q,
                freqs_cis_k=sink_recent_freqs_cis_k,
                attn_param=sink_recent_attn_param,
                kv_recompute=True,
                noisy_chunk_sizes=sink_recent_chunk_tokens,
                img_padding_info_dict=img_padding_info_dict,
            )
            self.kv_cache_manager.replace_kv_cache(
                name, kv_cache[0], kv_cache[1], sink_recent_frame_indices
            )

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"]
        return counts
