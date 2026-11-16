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

"""Non-streaming MMDiT backbone, used by Stage-3 DMD as the real score (teacher) and fake score (critic).

It sees the whole video at once (full bidirectional attention, no KV cache, no chunking), which is what makes it
a valid score model for the distribution-matching loss. The streaming generator lives in
``movieroll_transformer.py``.

The attention projections are SEPARATE q / k / v linears (``img_attn_q`` / ``img_attn_k`` / ``img_attn_v``),
matching the published 480p checkpoint this model is initialised from; the streaming model merges them into one
``*_attn_qkv`` at load time.
"""

from typing import Dict, List, Optional, Tuple, Union

from einops import rearrange
import torch
import torch.nn as nn

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from movieroll.models.transformers.modules.activation_layers import get_activation_layer
from movieroll.models.transformers.modules.attention import parallel_attention
from movieroll.models.transformers.modules.byt5_mapper import ByT5Mapper
from movieroll.models.transformers.modules.embed_layers import (
    PatchEmbed,
    TimestepEmbedder,
    VisionProjection,
    reorder_txt_token,
)
from movieroll.models.transformers.modules.mlp_layers import MLP, FinalLayer, unpatchify
from movieroll.models.transformers.modules.modulate_layers import ModulateDiT, apply_gate, modulate
from movieroll.models.transformers.modules.norm_layers import get_norm_layer
from movieroll.models.transformers.modules.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from movieroll.models.transformers.modules.token_refiner import SingleTokenRefiner
from movieroll.utils.communications import all_gather
from movieroll.commons.parallel_states import get_sequence_parallel_state, nccl_info


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attn_mode: str = None,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.heads_num = heads_num
        self.attn_mode = attn_mode

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

        self.img_attn_q = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.img_attn_k = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.img_attn_v = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
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

        self.txt_attn_q = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.txt_attn_k = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
        )

        self.txt_attn_v = nn.Linear(
            hidden_size, hidden_size * 1, bias=qkv_bias, **factory_kwargs
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

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        img_modulated = self.img_norm1(img)

        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )

        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)

        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)

        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Under SP both img_q/img_k and freqs_cis are this rank's shard (the top-level forward chunks img and
        # freqs together), so applying RoPE here, before the all_to_all, is self-consistent; freqs_cis is
        # therefore not forwarded to parallel_attention, which would apply it twice.
        if freqs_cis is not None:
            img_qq = apply_rotary_emb(img_q, freqs_cis)
            img_kk = apply_rotary_emb(img_k, freqs_cis)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_qq: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )

        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)

        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        attn, _ = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            attn_mode=self.attn_mode,
            text_mask=text_mask,
        )

        # Under SP the img part of attn is this rank's shard while the txt part is full length; derive the img
        # length from the txt length.
        txt_seq_len = txt_q.shape[1]
        img_seq_len = attn.shape[1] - txt_seq_len
        img_attn, txt_attn = attn[:, :img_seq_len].contiguous(), attn[:, img_seq_len:].contiguous()

        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )

        return img, txt


class MovieRollScoreTransformer(ModelMixin, ConfigMixin):
    """
    MovieRoll score transformer backbone (non-streaming MMDiT; Flux.1 and MMDiT 2403.03206).

    Inherits ModelMixin and ConfigMixin so that `@register_to_config` on `__init__` works; `trainer/dmd/setup.py`
    builds it and loads the HunyuanVideo-1.5 weights with `load_hunyuan_state_dict`.

    `attn_mode` is the attention backend of every block, one of {"flash3", "torch"} (the same math; torch
    runs it with SDPA).
    """

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
        attn_mode: str = "flash3",
        vision_states_dim: Optional[int] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.attn_mode = attn_mode

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
                    attn_mode=attn_mode,
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

        # Only built for ACS (anchor-conditioned scoring), which feeds the anchor frame's SigLIP tokens;
        # None means the checkpoint's vision_in weights are left unused, as in the released t2v recipe.
        self.vision_in = (
            None
            if vision_states_dim is None
            else VisionProjection(int(vision_states_dim), hidden_size, **factory_kwargs)
        )

        self.attn_param = {}
        # 0: qwen feature
        # 1: byt5 feature
        # 2: SigLIP image feature (ACS only)
        self.cond_type_embedding = nn.Embedding(3, self.hidden_size)
        self.cond_type_embedding.weight.data.fill_(0)

    def get_rotary_pos_embed(self, rope_sizes):
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        return_dict: bool = False,
        extra_kwargs=None,
        vision_states: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Bidirectional (non-streaming) forward of the DMD real / fake score model.

        hidden_states [B, 2 * in_channels + 1, T, H, W], timestep [B], Qwen text_states [B, S, D] with
        encoder_attention_mask [B, S], "byt5_text_states" / "byt5_text_mask" in extra_kwargs, and
        return_dict False -> (prediction [B, out_channels, T, H, W], None). Under sequence parallelism the
        image tokens are sharded inside and gathered back before the output.

        vision_states [B, tokens, vision_states_dim] are the SigLIP tokens of the i2v condition frame and
        require `vision_in`; they are merged into the text sequence like the byT5 tokens. ACS passes them
        only for the windows it scores as i2v (`movieroll/trainer/dmd/loss.py`), and the released t2v recipe
        never passes them at all.
        """
        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = text_states
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))

        img = self.img_in(img)
        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_spgroup
            if img.shape[1] % sp_size != 0:
                n_token = img.shape[1]
                assert n_token > (n_token // sp_size + 1) * (sp_size - 1), f'Token is too short for SP {sp_size}'

            img = torch.chunk(img, sp_size, dim=1)[sp_rank]
            freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

        vec = self.time_in(t)

        txt = self.txt_in(txt, t, text_mask)
        cond_emb = self.cond_type_embedding(torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long))
        txt = txt + cond_emb

        byt5_text_states = extra_kwargs["byt5_text_states"]
        byt5_text_mask = extra_kwargs["byt5_text_mask"]
        byt5_txt = self.byt5_in(byt5_text_states)

        cond_emb = self.cond_type_embedding(torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long))
        byt5_txt = byt5_txt + cond_emb

        txt, text_mask = reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask)

        if vision_states is not None:
            if self.vision_in is None:
                raise ValueError(
                    "vision_states need a score model built with vision_states_dim (ACS)"
                )
            vision_txt = self.vision_in(vision_states.to(dtype=txt.dtype))
            cond_emb = self.cond_type_embedding(
                torch.full(vision_txt.shape[:2], 2, device=vision_txt.device, dtype=torch.long)
            )
            vision_txt = vision_txt + cond_emb
            # Every image token is valid, so the merge only moves the text padding to the back.
            vision_mask = torch.ones(
                vision_txt.shape[:2], device=text_mask.device, dtype=text_mask.dtype
            )
            txt, text_mask = reorder_txt_token(vision_txt, txt, vision_mask, text_mask)

        freqs_cis = (freqs_cos, freqs_sin)
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, freqs_cis=freqs_cis, text_mask=text_mask)

        img_seq_len = img.shape[1]

        # Numerically a no-op for img, but load-bearing for autograd: concatenating txt keeps the last block's
        # text branch in the graph, so its parameters get zero grads instead of None. For the fake score this
        # reaches the optimizer: Muon's AdamW branch skips a None grad, weight decay and moments included.
        img = torch.cat((img, txt), 1)[:, :img_seq_len, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        if get_sequence_parallel_state():
            img = all_gather(img, dim=1, group=nccl_info.sp_group)
        img = unpatchify(img, tt, th, tw, self.patch_size, self.unpatchify_channels)
        assert not return_dict, "return_dict is not supported."
        return (img, None)
