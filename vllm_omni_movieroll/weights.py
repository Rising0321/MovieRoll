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

"""Weight names of MovieRoll's DiT checkpoint -> the parameters of `MovieRollTransformer3DModel`.

The same kind of table as vLLM-Omni's `vllm_omni/diffusion/models/causvid_wan/weights.py`.

MovieRoll's streaming backbone is the HunyuanVideo-1.5 MMDiT, so the transformer in this package uses the
HunyuanVideo-1.5 parameter names (the ones vLLM-Omni's own HunyuanVideo-1.5 transformer has) and the checkpoint is
renamed on the way in. Two kinds of rename:

* plain: one checkpoint tensor -> one parameter;
* fused qkv: ``*_attn_qkv`` is one ``[3 * width, ...]`` tensor which the target holds as q / k / v shards.

Pure python, no framework import, so the table also works on a machine without vLLM-Omni. `FusedShardTracker` makes the
load strict for the fused destinations: every q / k / v shard of a fused parameter must arrive.
"""

from __future__ import annotations

import regex as re
from dataclasses import dataclass
from typing import Optional

_REFINER = r"context_embedder.token_refiner.refiner_blocks.\1"

# (checkpoint pattern, replacement); the fused-qkv rules are listed in QKV_RULES below
PLAIN_RULES: list[tuple[str, str]] = [
    (r"^img_in\.proj\.(.*)$", r"x_embedder.proj.\1"),
    (r"^time_in\.mlp\.0\.(.*)$", r"time_embed.timestep_embedder.linear_1.\1"),
    (r"^time_in\.mlp\.2\.(.*)$", r"time_embed.timestep_embedder.linear_2.\1"),
    (r"^cond_type_embedding\.(.*)$", r"cond_type_embed.\1"),
    (r"^byt5_in\.layernorm\.(.*)$", r"context_embedder_2.norm.\1"),
    (r"^byt5_in\.fc1\.(.*)$", r"context_embedder_2.linear_1.\1"),
    (r"^byt5_in\.fc2\.(.*)$", r"context_embedder_2.linear_2.\1"),
    (r"^byt5_in\.fc3\.(.*)$", r"context_embedder_2.linear_3.\1"),
    (r"^txt_in\.input_embedder\.(.*)$", r"context_embedder.proj_in.\1"),
    (r"^txt_in\.t_embedder\.mlp\.0\.(.*)$", r"context_embedder.time_text_embed.timestep_embedder.linear_1.\1"),
    (r"^txt_in\.t_embedder\.mlp\.2\.(.*)$", r"context_embedder.time_text_embed.timestep_embedder.linear_2.\1"),
    (r"^txt_in\.c_embedder\.linear_1\.(.*)$", r"context_embedder.time_text_embed.text_embedder.linear_1.\1"),
    (r"^txt_in\.c_embedder\.linear_2\.(.*)$", r"context_embedder.time_text_embed.text_embedder.linear_2.\1"),
    (r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.norm1\.(.*)$", _REFINER + r".norm1.\2"),
    (r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.norm2\.(.*)$", _REFINER + r".norm2.\2"),
    (r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.self_attn_proj\.(.*)$", _REFINER + r".attn.to_out.0.\2"),
    (r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.mlp\.fc1\.(.*)$", _REFINER + r".ff.net.0.proj.\2"),
    (r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.mlp\.fc2\.(.*)$", _REFINER + r".ff.net.2.\2"),
    (r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.adaLN_modulation\.1\.(.*)$", _REFINER + r".norm_out.linear.\2"),
    (r"^double_blocks\.(\d+)\.img_mod\.linear\.(.*)$", r"transformer_blocks.\1.norm1.linear.\2"),
    (r"^double_blocks\.(\d+)\.txt_mod\.linear\.(.*)$", r"transformer_blocks.\1.norm1_context.linear.\2"),
    (r"^double_blocks\.(\d+)\.img_attn_q_norm\.(.*)$", r"transformer_blocks.\1.attn.norm_q.\2"),
    (r"^double_blocks\.(\d+)\.img_attn_k_norm\.(.*)$", r"transformer_blocks.\1.attn.norm_k.\2"),
    (r"^double_blocks\.(\d+)\.txt_attn_q_norm\.(.*)$", r"transformer_blocks.\1.attn.norm_added_q.\2"),
    (r"^double_blocks\.(\d+)\.txt_attn_k_norm\.(.*)$", r"transformer_blocks.\1.attn.norm_added_k.\2"),
    (r"^double_blocks\.(\d+)\.img_attn_proj\.(.*)$", r"transformer_blocks.\1.attn.to_out.0.\2"),
    (r"^double_blocks\.(\d+)\.txt_attn_proj\.(.*)$", r"transformer_blocks.\1.attn.to_add_out.\2"),
    (r"^double_blocks\.(\d+)\.img_mlp\.fc1\.(.*)$", r"transformer_blocks.\1.ff.net.0.proj.\2"),
    (r"^double_blocks\.(\d+)\.img_mlp\.fc2\.(.*)$", r"transformer_blocks.\1.ff.net.2.\2"),
    (r"^double_blocks\.(\d+)\.txt_mlp\.fc1\.(.*)$", r"transformer_blocks.\1.ff_context.net.0.proj.\2"),
    (r"^double_blocks\.(\d+)\.txt_mlp\.fc2\.(.*)$", r"transformer_blocks.\1.ff_context.net.2.\2"),
    # MovieRoll's final layer chunks its modulation as (shift, scale), and `norm_out` keeps that order
    (r"^final_layer\.adaLN_modulation\.1\.(.*)$", r"norm_out.linear.\1"),
    (r"^final_layer\.linear\.(.*)$", r"proj_out.\1"),
]

# (checkpoint pattern, target template with {shard}); `parallel` targets are fused QKVParallelLinear parameters loaded by
# shard id, the others are separate q / k / v parameters of a diffusers attention.
QKV_RULES: list[tuple[str, str, bool]] = [
    (r"^double_blocks\.(\d+)\.img_attn_qkv\.(weight|bias)$", r"transformer_blocks.\1.attn.to_qkv.\2", True),
    (r"^double_blocks\.(\d+)\.txt_attn_qkv\.(weight|bias)$", r"transformer_blocks.\1.attn.add_kv_proj.\2", True),
    (
        r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.self_attn_qkv\.(weight|bias)$",
        _REFINER + r".attn.to_{shard}.\2",
        False,
    ),
]

SHARD_IDS = ("q", "k", "v")

# checkpoint keys of the I2V vision tower, which MovieRoll does not build and this transformer does not either
IGNORED_PREFIXES = ("vision_in.",)


@dataclass(frozen=True)
class Mapped:
    """One target parameter fed by (a part of) a checkpoint tensor."""

    name: str
    shard_id: Optional[str] = None      # "q" / "k" / "v" for a fused QKVParallelLinear parameter
    part: Optional[int] = None          # which third of the checkpoint tensor (None = the whole tensor)


def map_name(name: str) -> list[Mapped]:
    """The target parameter(s) of a checkpoint key; ``[]`` for a key that is deliberately dropped, error if unknown."""
    if name.startswith(IGNORED_PREFIXES):
        return []
    for pattern, replacement in PLAIN_RULES:
        if re.match(pattern, name):
            return [Mapped(re.sub(pattern, replacement, name))]
    for pattern, template, parallel in QKV_RULES:
        if re.match(pattern, name):
            if parallel:
                target = re.sub(pattern, template, name)
                return [Mapped(target, shard_id=shard, part=index) for index, shard in enumerate(SHARD_IDS)]
            return [
                Mapped(re.sub(pattern, template.replace("{shard}", shard), name), part=index)
                for index, shard in enumerate(SHARD_IDS)
            ]
    raise KeyError(f"no mapping for MovieRoll checkpoint key {name!r}")


class FusedShardTracker:
    """Records which shards of every fused (q / k / v) destination were loaded; `check` raises if one is incomplete."""

    def __init__(self):
        self._seen: dict[str, set[str]] = {}

    def add(self, mapped: Mapped) -> None:
        if mapped.shard_id is not None:
            self._seen.setdefault(mapped.name, set()).add(mapped.shard_id)

    def check(self) -> None:
        incomplete = {name: sorted(set(SHARD_IDS) - seen) for name, seen in self._seen.items() if set(SHARD_IDS) - seen}
        if incomplete:
            raise RuntimeError(f"incomplete fused checkpoint parameters (missing shards): {incomplete}")


# MovieRoll's VAE checkpoint (its own `AutoencoderKLConv3D` names) -> diffusers' `AutoencoderKLHunyuanVideo15` (218 / 218
# keys, same shapes); applied once by the assemble script, which writes the model directory's `vae/` in diffusers' layout
VAE_RULES: list[tuple[str, str]] = [
    (r"^(encoder|decoder)\.down\.(\d+)\.block\.(\d+)\.", r"\1.down_blocks.\2.resnets.\3."),
    (r"^(encoder|decoder)\.down\.(\d+)\.downsample\.", r"\1.down_blocks.\2.downsamplers.0."),
    (r"^(encoder|decoder)\.up\.(\d+)\.block\.(\d+)\.", r"\1.up_blocks.\2.resnets.\3."),
    (r"^(encoder|decoder)\.up\.(\d+)\.upsample\.", r"\1.up_blocks.\2.upsamplers.0."),
    (r"^(encoder|decoder)\.mid\.block_1\.", r"\1.mid_block.resnets.0."),
    (r"^(encoder|decoder)\.mid\.block_2\.", r"\1.mid_block.resnets.1."),
    (r"^(encoder|decoder)\.mid\.attn_1\.(q|k|v)\.", r"\1.mid_block.attentions.0.to_\2."),
    (r"^(encoder|decoder)\.mid\.attn_1\.", r"\1.mid_block.attentions.0."),
    (r"\.nin_shortcut\.", r".conv_shortcut."),
]


def vae_key(name: str) -> str:
    """The diffusers name of a MovieRoll VAE key (keys the rules do not touch are already the same)."""
    for pattern, replacement in VAE_RULES[:-1]:
        if re.match(pattern, name):
            name = re.sub(pattern, replacement, name)
            break
    return re.sub(VAE_RULES[-1][0], VAE_RULES[-1][1], name)


def vae_config(movieroll_config: dict) -> dict:
    """diffusers `AutoencoderKLHunyuanVideo15` config of a MovieRoll VAE `config.json`."""
    c = movieroll_config
    if c.get("shift_factor") not in (None, 0, 0.0):
        raise ValueError(f"diffusers' HunyuanVideo-1.5 VAE has no shift_factor, the checkpoint has {c['shift_factor']}")
    return {
        "_class_name": "AutoencoderKLHunyuanVideo15",
        "_diffusers_version": "0.40.0",
        "in_channels": c["in_channels"],
        "out_channels": c["out_channels"],
        "latent_channels": c["latent_channels"],
        "block_out_channels": list(c["block_out_channels"]),
        "layers_per_block": c["layers_per_block"],
        "spatial_compression_ratio": c["ffactor_spatial"],
        "temporal_compression_ratio": c["ffactor_temporal"],
        "downsample_match_channel": c.get("downsample_match_channel", True),
        "upsample_match_channel": c.get("upsample_match_channel", True),
        "scaling_factor": c["scaling_factor"],
    }
