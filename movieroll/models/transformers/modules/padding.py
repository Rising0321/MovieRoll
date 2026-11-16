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
# Adapted from flash-attention (BSD 3-Clause License)
#   https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
#   https://github.com/Dao-AILab/flash-attention/blob/main/hopper/padding.py
# ==============================================================================

"""Pack / unpack for prefix-padded batches.

The attention mask is a prefix mask: the first k tokens of every sequence are valid and the rest is padding (k may
differ between batch elements). Compared with the gather / index_put based `unpad_input` / `pad_input` of
flash-attention:

                   Forward               Backward
    unpad_input    gather (random read)  scatter (random)
    prefix_pack    narrow (sequential)   slice-assign
    pad_input      index_put (random)    indexing (random)
    prefix_unpack  slice-assign          narrow (sequential)

narrow is cache friendly (sequential memory access) and the backward needs no scatter_add. Both functions use the
`setup_context` autograd API (torch.compile friendly).
"""

import torch


class PrefixPack(torch.autograd.Function):
    """Pack [batch, seqlen, ...] into [total_valid, ...] using the per-sequence valid prefix lengths.

    Forward:  x[i, :vl[i]] for each i -> cat -> packed
    Backward: split grad -> written back into the prefixes of a zero tensor
    """

    @staticmethod
    def forward(hidden_states, valid_lens_cpu):
        parts = [hidden_states[i, :vl] for i, vl in enumerate(valid_lens_cpu)]
        return torch.cat(parts, dim=0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        hidden_states, valid_lens_cpu = inputs
        ctx.valid_lens_cpu = valid_lens_cpu
        ctx.shape = hidden_states.shape

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_zeros(ctx.shape)
        offset = 0
        for i, vl in enumerate(ctx.valid_lens_cpu):
            grad_input[i, :vl] = grad_output[offset:offset + vl]
            offset += vl
        return grad_input, None


class PrefixUnpack(torch.autograd.Function):
    """Unpack [total_valid, ...] back into [batch, seqlen, ...] with zero padding.

    Forward:  split packed -> written into the prefixes of a zero tensor
    Backward: take the prefix of every sequence -> cat
    """

    @staticmethod
    def forward(packed, valid_lens_cpu, batch, seqlen):
        other_shape = packed.shape[1:]
        output = packed.new_zeros(batch, seqlen, *other_shape)
        offset = 0
        for i, vl in enumerate(valid_lens_cpu):
            output[i, :vl] = packed[offset:offset + vl]
            offset += vl
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, valid_lens_cpu, _, _ = inputs
        ctx.valid_lens_cpu = valid_lens_cpu

    @staticmethod
    def backward(ctx, grad_output):
        parts = [grad_output[i, :vl] for i, vl in enumerate(ctx.valid_lens_cpu)]
        return torch.cat(parts, dim=0), None, None, None


prefix_pack   = PrefixPack.apply
prefix_unpack = PrefixUnpack.apply
