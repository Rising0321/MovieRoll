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

"""Model size table and the transformer factory.

`MODEL_CONFIG[model_type]` is expanded straight into a transformer constructor, so every key must be a real
constructor parameter of BOTH transformers: the streaming generator (`MovieRollDiffusionTransformer`) and the
non-streaming score model (`MovieRollScoreTransformer`), which Stage-3 DMD builds from the same entry.

Every value here is load-bearing for the published checkpoints; none of them may be dropped as a "default":
  * patch_size = [1, 1, 1]      — the default [1, 2, 2] changes final_layer's out-features and unpatchify;
  * in_channels = 32            — PatchEmbed takes 2 * in_channels + 1 = 65 channels (latents, cond, mask);
  * text_states_dim = 3584      — the text encoder's hidden size; the constructor default (4096) is wrong.
"""

from movieroll.commons.constants import MODEL_TYPE
from movieroll.models.transformers.movieroll_transformer import MovieRollDiffusionTransformer

MODEL_CONFIG = {
    MODEL_TYPE: {
        "in_channels": 32,
        "out_channels": 32,
        'mm_double_blocks_depth': 54,
        'rope_dim_list': [16, 56, 56],
        'hidden_size': 2048,
        'heads_num': 16,
        'mlp_width_ratio': 4,
        'patch_size': [1, 1, 1],
        'text_states_dim': 3584,
    },
}


def load_model(model_type, factor_kwargs):
    """Build the streaming transformer for `model_type`.

    Args:
        model_type: a key of MODEL_CONFIG.
        factor_kwargs: constructor arguments supplied by the caller rather than the size table, i.e. `device`,
            `dtype` and `streaming_config`.

    Returns:
        model (nn.Module): the MovieRoll streaming transformer.
    """
    if model_type not in MODEL_CONFIG:
        raise NotImplementedError(
            f"model_type={model_type} is not supported, expected one of {list(MODEL_CONFIG.keys())}"
        )
    return MovieRollDiffusionTransformer(
        **MODEL_CONFIG[model_type],
        **factor_kwargs,
    )
