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

"""Check a SageAttention build on this GPU: every kernel path against PyTorch SDPA on random bf16 inputs.

    python -m sglang_movieroll.tools.check_sageattention

A good build gives a relative error of about 0.04 on every path. The v2.2.0 tag's sm90 kernel (the one `sageattn`
picks on Hopper) gave over 1 on H20, and an 81-frame video generated with it came out as NaN.
"""

import sys

import sageattention as sa
import torch
import torch.nn.functional as F

PATHS = (
    "sageattn",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda_sm90",
    "sageattn_qk_int8_pv_fp16_triton",
)


def main() -> int:
    torch.manual_seed(0)
    worst = 0.0
    # a square attention, and a MovieRoll window attending to its history and text
    for q_len, kv_len in ((1024, 1024), (6480, 14872)):
        q = torch.randn(1, q_len, 16, 128, device="cuda", dtype=torch.bfloat16) * 3
        k = torch.randn(1, kv_len, 16, 128, device="cuda", dtype=torch.bfloat16) * 3
        v = torch.randn(1, kv_len, 16, 128, device="cuda", dtype=torch.bfloat16) * 5
        reference = F.scaled_dot_product_attention(
            *(t.transpose(1, 2) for t in (q, k, v))
        )
        reference = reference.transpose(1, 2).float()
        for name in PATHS:
            kernel = getattr(sa, name, None)
            if kernel is None:
                continue
            out = kernel(q, k, v, tensor_layout="NHD", is_causal=False).float()
            error = ((out - reference).norm() / reference.norm()).item()
            worst = max(worst, error if torch.isfinite(out).all() else float("inf"))
            print(f"{q_len}x{kv_len} {name}: relative error {error:.3e}")
    print("OK" if worst < 0.1 else "BROKEN: a kernel path disagrees with SDPA")
    return 0 if worst < 0.1 else 1


if __name__ == "__main__":
    sys.exit(main())
