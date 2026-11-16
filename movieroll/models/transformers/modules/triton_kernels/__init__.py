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

"""Triton fused kernels (drop-in replacements with PyTorch fallbacks).

  - fused_apply_rotary_emb : RoPE (rotate_half + cos / sin fused)
  - fused_rms_norm         : RMSNorm (fp32 statistics, weight, cast fused)

Switch:
  - is_triton_enabled()     : True when triton is installed and the kernels are enabled (default)
  - set_triton_enabled(val) : enable / disable the kernels at runtime (callers then use the PyTorch implementation)
  - TRITON_AVAILABLE        : whether triton and the kernels imported successfully
"""

_triton_enabled = True

try:
    from movieroll.models.transformers.modules.triton_kernels.fused_rope import fused_apply_rotary_emb
    from movieroll.models.transformers.modules.triton_kernels.fused_rmsnorm import fused_rms_norm
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    fused_apply_rotary_emb = None
    fused_rms_norm = None


def is_triton_enabled():
    """Whether the fused kernels are used: triton is available and `set_triton_enabled(False)` was not called."""
    return TRITON_AVAILABLE and _triton_enabled


def set_triton_enabled(val: bool):
    """Enable / disable the fused kernels at runtime."""
    global _triton_enabled
    _triton_enabled = val


__all__ = [
    "TRITON_AVAILABLE",
    "is_triton_enabled",
    "set_triton_enabled",
    "fused_apply_rotary_emb",
    "fused_rms_norm",
]
