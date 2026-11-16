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

import math

import torch
import torch.nn as nn

from movieroll.utils.helpers import to_2tuple


class PatchEmbed(nn.Module):
    """Video latent to patch-token embedding (Conv3d), flattened to [B, N, C].

    Input is the `2 * in_chans + 1` channel concat [latents, cond copy, mask], which fixes
    `img_in.proj.weight` to [embed_dim, 2 * in_chans + 1, *patch_size]. Only the first `in_chans` input
    channels get xavier init; the rest start at zero.

    Based on the impl in https://github.com/google-research/vision_transformer (copyright 2020 Ross Wightman).
    """

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        bias=True,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        orig_in_chans = in_chans
        in_chans = in_chans * 2 + 1

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **factory_kwargs,
        )

        nn.init.xavier_uniform_(self.proj.weight[:, :orig_in_chans].view(self.proj.weight[:, :orig_in_chans].size(0), -1))
        nn.init.zeros_(self.proj.weight[:, orig_in_chans:].view(self.proj.weight[:, orig_in_chans:].size(0), -1))

        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        x = self.norm(x)
        return x


class TextProjection(nn.Module):
    """
    Two-layer MLP (Linear, activation, Linear) that projects a text embedding to hidden_size.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs,
        )
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs,
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class VisionProjection(nn.Module):
    """SigLIP image tokens to hidden_size, for the Stage-3 ACS (anchor-conditioned scoring) score models.

    The generator never builds this; only the real / fake score models do, and only to consume the anchor
    frame's SigLIP tokens (see `movieroll/trainer/dmd/loss.py`). The layer order fixes the checkpoint's
    `vision_in.proj.{0,1,3,4}` keys.

    Adapted from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
    """

    def __init__(self, in_dim, out_dim, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim, **factory_kwargs),
            nn.Linear(in_dim, in_dim, **factory_kwargs),
            nn.GELU(),
            nn.Linear(in_dim, out_dim, **factory_kwargs),
            nn.LayerNorm(out_dim, **factory_kwargs),
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)


def reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask):
    """Pack each sample's valid tokens first: [byT5 valid, text valid, byT5 padding, text padding].

    The padding slots are zeroed. Returns the reordered tokens and their int64 mask (1 = valid).
    """
    reorder_txt = []
    reorder_mask = []

    for i in range(text_mask.shape[0]):
        byt5_text_mask_i = byt5_text_mask[i].bool()
        text_mask_i = text_mask[i].bool()

        byt5_txt_i = byt5_txt[i]
        txt_i = txt[i]
        # zero out the padding slots: block-mask attention is approximate, zeros keep the error down
        pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
        pad_text = torch.zeros_like(txt_i[~text_mask_i])
        reorder_txt_i = torch.cat([byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i],
                                   pad_byt5, pad_text], dim=0)
        reorder_mask_i = torch.cat([byt5_text_mask_i[byt5_text_mask_i], text_mask_i[text_mask_i],
                                    byt5_text_mask_i[~byt5_text_mask_i], text_mask_i[~text_mask_i]], dim=0)

        reorder_txt.append(reorder_txt_i)
        reorder_mask.append(reorder_mask_i)

    reorder_txt = torch.stack(reorder_txt)
    reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)

    return reorder_txt, reorder_mask


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embeddings: `t` is a 1-D (N,) tensor of possibly fractional indices -> (N, dim).

    `max_period` controls the minimum frequency.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, hidden_size, bias=True, **factory_kwargs
            ),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.zeros_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, t):
        t_freq = timestep_embedding(
            t, self.frequency_embedding_size, self.max_period
        ).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
