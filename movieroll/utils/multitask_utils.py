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

import random

import torch


def get_multitask_mask(video_length):
    """Return the t2v frame mask (all zeros) and its task name, and consume this call's global draw.

    Every released chain feeds the transformer the t2v input layout (an i2v image enters through the KV cache
    instead), so the mask is always all-zero and the name always "t2v". A non-None mask would switch the
    transformer input to the 65-channel concat layout.

    The `random.random()` below is deliberately kept with no reader: exactly one global draw per call is part of
    the training / inference RNG stream, and every draw after it -- the transport's noise included -- is
    positioned against it. Removing it, moving it, or making it conditional shifts the stream, and training and
    sampling no longer reproduce the released runs.
    """
    random.random()
    return torch.zeros(video_length), "t2v"
