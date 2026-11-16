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

"""The SigLIP tower the ACS score models read the anchor frame with.

ACS (anchor-conditioned scoring, `dmd.score_mode: acs`) scores the window that starts at the i2v anchor the
way the bidirectional model was trained for i2v: the anchor latent in the condition channels plus this
encoder's tokens of the same frame. Nothing else in MovieRoll uses it -- the generator has no vision tower
and inference conditions through the KV cache only.
"""

import numpy as np
import torch
from transformers import SiglipImageProcessor, SiglipVisionModel

from movieroll.utils.torch_utils import PRECISION_TO_TYPE


class SiglipImageEncoder:
    """Frozen SigLIP: uint8 RGB frames -> the image tokens a score model's `vision_in` consumes.

    `path` holds the two subdirectories the released i2v tower ships with, `image_encoder/` (weights) and
    `feature_extractor/` (preprocessing). Deliberately not an `nn.Module`: the encoder is frozen, outside
    FSDP, and must not add a second state-dict path to the training checkpoint.
    """

    def __init__(self, path, device, precision="fp16", logger=None):
        self.dtype = PRECISION_TO_TYPE[precision]
        self.device = device
        self.model = SiglipVisionModel.from_pretrained(path, subfolder="image_encoder")
        self.model = self.model.to(device=device, dtype=self.dtype).eval()
        self.model.requires_grad_(False)
        self.processor = SiglipImageProcessor.from_pretrained(
            path, subfolder="feature_extractor"
        )
        if logger is not None:
            logger.info(f"[acs] SigLIP image encoder loaded from {path} in {precision}")

    def encode(self, images: np.ndarray) -> torch.Tensor:
        """Encode uint8 frames [B, H, W, 3] into image tokens [B, tokens, dim].

        The processor does the resize / normalisation i2v training used, so the tokens match what the score
        models were trained to read.
        """
        if not isinstance(images, np.ndarray) or images.ndim != 4 or images.shape[3] != 3:
            raise ValueError(
                f"SigLIP input must be a uint8 [B, H, W, 3] array, got {type(images)} "
                f"{getattr(images, 'shape', None)}"
            )
        inputs = self.processor.preprocess(images=images, return_tensors="pt").to(
            device=self.device, dtype=self.dtype
        )
        with torch.no_grad():
            return self.model(**inputs).last_hidden_state
