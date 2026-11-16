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

"""Configs of MovieRoll's two text encoders in SGLang's format.

Encoder 0 is the Qwen2.5-VL text tower (SGLang's `Qwen2_5_VLForConditionalGeneration`), encoder 1 the byT5
glyph encoder (SGLang's `T5EncoderModel`). What the checkpoint was trained with and must not change: the video
chat template and the length its prefix is cropped to, the hidden layer two before the last, the right padding
to a fixed length, and byT5's 256-token window with its special tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import TextEncoderArchConfig
from sglang.multimodal_gen.configs.models.encoders.qwen_image import (
    Qwen2_5VLConfig,
    QwenImageArchConfig,
)
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig, T5Config

# Tokens of the chat prefix that are dropped from the hidden states (measured once with MovieRoll's own
# tokenizer and template).
QWEN_CROP_START = 108


@dataclass
class MovieRollQwenArchConfig(QwenImageArchConfig):
    text_len: int = 1000
    hidden_state_skip_layer: int = 2
    crop_start: int = QWEN_CROP_START

    def __post_init__(self) -> None:
        super().__post_init__()
        # the template's prefix is cropped off the hidden states, so the window has to hold it as well
        self.tokenizer_kwargs = {
            "truncation": True,
            "max_length": self.text_len + self.crop_start,
            "padding": "max_length",
            "return_tensors": "pt",
        }


@dataclass
class MovieRollQwenConfig(Qwen2_5VLConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=MovieRollQwenArchConfig)
    prefix: str = "qwen2_5_vl"


@dataclass
class MovieRollByT5ArchConfig(T5ArchConfig):
    # byT5-small with the Glyph-SDXL-v2 colour / font tokens; the rest comes from the component's config.json
    text_len: int = 256
    vocab_size: int = 1510
    d_model: int = 1472
    d_kv: int = 64
    d_ff: int = 3584
    num_layers: int = 12
    num_heads: int = 6
    feed_forward_proj: str = "gated-gelu"


@dataclass
class MovieRollByT5Config(T5Config):
    arch_config: TextEncoderArchConfig = field(default_factory=MovieRollByT5ArchConfig)
    prefix: str = "byt5"
