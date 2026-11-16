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
# Modified from timm
#   https://github.com/huggingface/pytorch-image-models/blob/648aaa41233ba83eb38faf5ba9d415d574823241/timm/layers/mlp.py#L13
# ==============================================================================
# Copyright 2019 Ross Wightman
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

import math

import torch
import torch.nn as nn

from movieroll.models.transformers.modules.modulate_layers import modulate
from movieroll.utils.helpers import to_2tuple

# The MLP is token-wise, so chunking the sequence does not change the result; it only bounds the peak memory
# of the activation temporaries ([B, chunk, 4D] instead of [B, full_seq, 4D]). This limit is 480p, 200 frames.
_MLP_MAX_CHUNK_TOKENS = 40 * 40 * 50

class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = nn.Linear

        self.fc1 = linear_layer(
            in_channels, hidden_channels, bias=bias[0], **factory_kwargs
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = nn.Identity()
        self.fc2 = linear_layer(
            hidden_channels, out_features, bias=bias[1], **factory_kwargs
        )
        self.drop2 = nn.Dropout(drop_probs[1])

    def _ffn(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len > _MLP_MAX_CHUNK_TOKENS:
            num_chunks = math.ceil(seq_len / _MLP_MAX_CHUNK_TOKENS)
            chunk_size = math.ceil(seq_len / num_chunks)
            out_list = []
            for i in range(0, seq_len, chunk_size):
                out_list.append(self._ffn(x[:, i:i + chunk_size]))
            return torch.cat(out_list, dim=1)
        return self._ffn(x)

class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(
        self, hidden_size, patch_size, out_channels, act_layer, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        # `patch_size` is a 3-sequence [T, H, W]. This Linear takes no factory_kwargs: it is built in the
        # default dtype / device and callers move the model afterwards.
        self.linear = nn.Linear(
            hidden_size,
            patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
            bias=True,
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c, thw=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale, thw=thw)
        x = self.linear(x)
        return x


def unpatchify(x, t, h, w, patch_size, channels):
    """
    x:    (N, t*h*w, prod(patch_size) * C)
    imgs: (N, C, t*pt, h*ph, w*pw)
    """
    c = channels
    pt, ph, pw = patch_size
    assert t * h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
    x = torch.einsum("nthwcopq->nctohpwq", x)
    imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

    return imgs
