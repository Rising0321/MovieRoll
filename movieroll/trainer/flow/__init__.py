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

# ==============================================================================
# Modified from SiT (https://github.com/willisma/SiT)
# ==============================================================================

"""Construction of the flow-matching transport used by the SFT trainer."""

from movieroll.trainer.flow.transport import Transport


def create_transport(*, video_shift, streaming_config):
    """Build the flow-matching transport.

    The path (linear), the prediction target (velocity), the loss weighting (none) and the timestep
    distribution (logit-normal) are fixed; only the timestep shift and the streaming geometry are arguments.
    """
    state = Transport(
        video_shift=video_shift,
        streaming_config=streaming_config,
    )

    return state
