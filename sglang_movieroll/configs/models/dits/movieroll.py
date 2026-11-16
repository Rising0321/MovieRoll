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

"""Architecture config of MovieRoll's DiT in SGLang's format (the counterpart of `configs/models/dits/hunyuanvideo.py`).

The defaults are the released checkpoints' architecture; the model directory's ``transformer/config.json``
(written by `sglang_movieroll.tools.assemble_model_dir`) overrides them through SGLang's ``update_model_arch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

from sglang_movieroll.configs.models.dits.weight_names import PARAM_NAMES_MAPPING


def is_double_block(name: str, module) -> bool:
    """The double-stream blocks, by name: what FSDP shards and what SGLang's layer-wise offload stages."""
    return "double_blocks" in name and name.split(".")[-1].isdigit()


@dataclass
class MovieRollArchConfig(DiTArchConfig):
    # FSDP2 shards (and all-gathers in the forward pre-hook of) the double-stream blocks; every way of calling a
    # block goes through its forward, see `runtime/models/dits/movieroll.py`
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_double_block])
    _compile_conditions: list = field(default_factory=list)
    # FlashAttention (the default on Hopper), SageAttention 2 (8-bit, opt-in through --attention-backend sage_attn)
    # and torch's SDPA (the one with fp32; SGLang allows the ring only with the other two); all three take a dense key
    # set, which is all the blocks ask for
    _supported_attention_backends: set = field(
        default_factory=lambda: {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.SAGE_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
        }
    )

    param_names_mapping: dict = field(default_factory=lambda: dict(PARAM_NAMES_MAPPING))
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    patch_size: tuple[int, int, int] = (1, 1, 1)
    in_channels: int = (
        32  # VAE latent channels; the model input has 2 * in_channels + 1 (latents, zeros, mask)
    )
    out_channels: int = 32
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    num_layers: int = 54
    num_refiner_layers: int = 2
    mlp_ratio: float = 4.0
    text_embed_dim: int = 3584  # Qwen2.5-VL hidden size
    text_embed_2_dim: int = 1472  # byT5 glyph encoder
    byt5_hidden_dim: int = 2048
    rope_axes_dim: tuple[int, int, int] = (16, 56, 56)
    rope_theta: float = 256.0

    # streaming geometry (HieraKV): blocks [0, full_history_block_num) see the whole clean history, the others
    # the sink and the most recent chunk
    max_num_latent_frames: int = 120
    full_history_block_num: int = 6
    sink_chunks: int = 1

    exclude_lora_layers: list = field(
        default_factory=lambda: [
            "img_in",
            "txt_in",
            "time_in",
            "byt5_in",
            "final_layer",
        ]
    )

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class MovieRollDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=MovieRollArchConfig)

    prefix: str = "MovieRoll"
