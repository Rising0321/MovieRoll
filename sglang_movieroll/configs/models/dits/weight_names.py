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

"""MovieRoll checkpoint key -> parameter of `MovieRollTransformer3DModel`, as SGLang's ``param_names_mapping``.

MovieRoll's backbone is HunyuanVideo-1.5's MMDiT and its checkpoint keeps Tencent's original tensor names, which
are also the names of SGLang's own `HunyuanVideoTransformer3DModel` (``double_blocks.N.img_attn_qkv``,
``img_mod.linear``, ``txt_in.*``). The DiT in this package uses SGLang's layers under those names, so most keys
need no rule at all; what differs is only where an SGLang layer names its sub-modules differently (an
``nn.Sequential``'s ``0`` / ``2`` against ``fc_in`` / ``fc_out``, and the token refiner's block list).

The fused ``*_attn_qkv`` projections stay one tensor: SGLang's `QKVParallelLinear` keeps a fused parameter and
its ``weight_loader`` splits a checkpoint tensor into the q / k / v shards of the tensor-parallel rank, so
nothing has to be split or merged by this table (which could not express it anyway: SGLang's mapper merges but
cannot split). Every rule here is a plain rename.

The table is the loader's own format (regex -> replacement, applied in order, an empty replacement drops the
key), kept free of framework imports so that it can be tested without SGLang. `map_name` applies it the way the
loader does.
"""

from __future__ import annotations

import re

_REFINER = r"txt_in.refiner_blocks.\1"
_OLD_REFINER = r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\."

PARAM_NAMES_MAPPING: dict[str, str] = {
    # the timestep embedders are `nn.Sequential(Linear, SiLU, Linear)` in the checkpoint, SGLang's
    # `TimestepEmbedder.mlp.fc_in / fc_out`
    r"^time_in\.mlp\.0\.(.*)$": r"time_in.mlp.fc_in.\1",
    r"^time_in\.mlp\.2\.(.*)$": r"time_in.mlp.fc_out.\1",
    r"^txt_in\.t_embedder\.mlp\.0\.(.*)$": r"txt_in.t_embedder.mlp.fc_in.\1",
    r"^txt_in\.t_embedder\.mlp\.2\.(.*)$": r"txt_in.t_embedder.mlp.fc_out.\1",
    # the text-context projection (two linears) is SGLang's `MLP`
    r"^txt_in\.c_embedder\.linear_1\.(.*)$": r"txt_in.c_embedder.fc_in.\1",
    r"^txt_in\.c_embedder\.linear_2\.(.*)$": r"txt_in.c_embedder.fc_out.\1",
    # the token refiner: `individual_token_refiner.blocks` is `refiner_blocks` in SGLang
    _OLD_REFINER
    + r"(norm1|norm2|self_attn_qkv|self_attn_proj)\.(.*)$": _REFINER
    + r".\2.\3",
    _OLD_REFINER + r"mlp\.fc1\.(.*)$": _REFINER + r".mlp.fc_in.\2",
    _OLD_REFINER + r"mlp\.fc2\.(.*)$": _REFINER + r".mlp.fc_out.\2",
    _OLD_REFINER
    + r"adaLN_modulation\.1\.(.*)$": _REFINER
    + r".adaLN_modulation.linear.\2",
    # the feed-forwards of the double-stream blocks are SGLang's `MLP` too
    r"^double_blocks\.(\d+)\.(img|txt)_mlp\.fc1\.(.*)$": r"double_blocks.\1.\2_mlp.fc_in.\3",
    r"^double_blocks\.(\d+)\.(img|txt)_mlp\.fc2\.(.*)$": r"double_blocks.\1.\2_mlp.fc_out.\3",
    # the final layer's modulation is `nn.Sequential(SiLU, Linear)`: SGLang's `ModulateProjection.linear`
    r"^final_layer\.adaLN_modulation\.1\.(.*)$": r"final_layer.adaLN_modulation.linear.\1",
    # the image-to-video vision tower of the HunyuanVideo-1.5 checkpoints; MovieRoll does not build it
    r"^vision_in\..*$": "",
}


def map_name(name: str) -> str:
    """Target parameter of a checkpoint key (``""`` when the key is dropped), by the first rule that matches."""
    for pattern, replacement in PARAM_NAMES_MAPPING.items():
        if re.match(pattern, name):
            return re.sub(pattern, replacement, name)
    return name
