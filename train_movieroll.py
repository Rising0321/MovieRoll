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

"""MovieRoll streaming SFT entry point. Forwards to `movieroll.train`.

It has to be launched with torchrun, e.g.

    torchrun --nproc_per_node 8 train_movieroll.py \
        --config movieroll/configs/sft/stage1_single_event.yaml \
        --override output_dir=./outputs/run1
"""

from movieroll.train import main, parse_args
from movieroll.trainer.sft.config import load_sft_config

if __name__ == "__main__":
    args = parse_args()
    main(load_sft_config(args.config, args.override))
