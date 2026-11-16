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

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Normalize over the last dim; the two paths are not bit-identical.

        The fused triton kernel (CUDA only) does fp32 math and outputs `x.dtype`; the PyTorch path normalizes
        in fp32, casts back to `x.dtype` and then multiplies by `weight`, so its output dtype follows
        type promotion.
        """
        if x.is_cuda:
            from movieroll.models.transformers.modules.triton_kernels import fused_rms_norm, is_triton_enabled
            if is_triton_enabled():
                return fused_rms_norm(x, self.weight, self.eps)
        output = self._norm(x.float()).type_as(x)
        output = output * self.weight
        return output


def get_norm_layer(norm_layer):
    if norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")
