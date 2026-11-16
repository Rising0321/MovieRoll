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

"""Plain data the stages hand to the transformer; no framework import, so the sampling loop stays testable alone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class TextTokens:
    """The valid (unpadded) text conditioning of one prompt, as the encoders produced it.

    ``qwen`` is the Qwen2.5-VL hidden state ``[1, S, text_embed_dim]`` of the prompt tokens; ``byt5`` the glyph
    encoder's ``[1, S2, text_embed_2_dim]`` for the quoted texts, or None when the prompt quotes nothing.
    Padding is dropped before it gets here: the padded keys are masked in MovieRoll's attention anyway, so
    cropping is exact.
    """

    qwen: torch.Tensor
    byt5: Optional[torch.Tensor] = None

    def as_embeddings(self) -> List[Optional[torch.Tensor]]:
        """The per-encoder list SGLang's DiTs take as ``encoder_hidden_states``: ``[qwen, byt5 or None]``."""
        return [self.qwen, self.byt5]


@dataclass(frozen=True)
class EventCondition:
    """The text conditioning of one event: the prompt, and for guided sampling the negative prompt."""

    positive: TextTokens
    negative: Optional[TextTokens] = None
