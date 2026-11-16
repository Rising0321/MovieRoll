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

"""Run one of vLLM-Omni's shared example scripts (``text_to_video.py``, ``image_to_video.py``) with MovieRoll's defaults.

The shared scripts ask `vllm_omni.model_extras` for a model's defaults (size, frame count, fps, steps, guidance) and for
whether it sizes the input image itself before they build `Omni`, and building `Omni` is where vLLM-Omni loads its
plugins. Started directly, a script therefore cannot see what this plugin declares (`video_defaults`) and falls back to
its Wan2.2 preset. This launcher loads the plugins first, with vLLM-Omni's own loader, and then runs the script unmodified
with the arguments that follow it::

    python vllm_omni_movieroll/examples/offline_inference/movieroll/run_example.py \\
        <vLLM-Omni v0.32.0 checkout>/examples/offline_inference/text_to_video/text_to_video.py \\
        --model ./movieroll-distilled --prompt "..." --num-frames 81 --output fox.mp4
"""

import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        raise SystemExit("usage: run_example.py <vLLM-Omni example script>.py [the script's arguments ...]")
    from vllm_omni.plugins import load_omni_general_plugins

    load_omni_general_plugins()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
