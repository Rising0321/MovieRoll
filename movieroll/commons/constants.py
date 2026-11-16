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

"""Constants shared by the MovieRoll text encoder, training and inference entry points."""

# Computation scale factor, 1P = 1_000_000_000_000_000. Tensorboard displays the value in PetaFLOPS to avoid
# an overflow error when logging.
C_SCALE = 1_000_000_000_000_000

# Name of the only transformer configuration (MODEL_CONFIG key and `model_type` value of the training configs).
MODEL_TYPE = "MovieRoll-16164-MMDiT"

# The flow-matching timestep range: t = sigma * NUM_TRAIN_TIMESTEPS. The transformer's timestep embedding was trained
# on t in [0, NUM_TRAIN_TIMESTEPS], so every scheduler, loss and validator uses this one value.
NUM_TRAIN_TIMESTEPS = 1000

# Width of the SigLIP image tokens the ACS score models read (`dmd.score_mode: acs`). Fixed by the
# checkpoint's vision_in.proj weights, which take 1152 and give the hidden size.
SIGLIP_TOKEN_DIM = 1152

# Default byT5 / Glyph-SDXL-v2 locations, following the `ckpts/` layout the README's Download section sets up.
BYT5_FONT = "ckpts/text_encoder/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json"
BYT5_COLOR = "ckpts/text_encoder/Glyph-SDXL-v2/assets/color_idx.json"
BYT5_GOOGLE = "ckpts/text_encoder/byt5-small"
BYT5_CKPT = "ckpts/text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt"
