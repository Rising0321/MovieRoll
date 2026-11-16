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
import torch.distributed as dist

PRECISION_TO_TYPE = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def all_reduce_sum(value_on_this_rank, device):
    value = torch.tensor(value_on_this_rank, device=device, dtype=torch.float32)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value.item()


def count_failed_ranks(failed, device) -> int:
    """How many ranks have `failed` set: a SUM all-reduce of a 0/1 flag, or the local flag with no group.

    Callers reduce their local failure and then raise on every rank when the count is non-zero, so no rank
    is left waiting in the next collective.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return int(failed)
    t = torch.tensor([int(failed)], device=device, dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())
