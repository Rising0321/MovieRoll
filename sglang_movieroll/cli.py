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

"""``python -m sglang_movieroll.cli {generate|serve} --model-path <model dir> ...``: sglang's CLI, MovieRoll registered.

This is `sglang generate` / `sglang serve` unchanged (it hands over to ``sglang.cli.main``), started from a
module that registers MovieRoll first. The registration is at module level on purpose: sglang's workers are
spawned and re-import the launching module (see `sglang_movieroll.registry`).
"""

from sglang_movieroll.registry import register

register()


def main() -> None:
    from sglang.cli.main import main as sglang_main

    sglang_main()


if __name__ == "__main__":
    main()
