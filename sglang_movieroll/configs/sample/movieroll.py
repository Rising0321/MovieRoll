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

"""MovieRoll's `SamplingParams`: the request defaults of each chain, and the request validation.

SGLang builds a `SamplingParams` for every request (Python API, command line, HTTP) from the class registered
for the model; the defaults below are the released chains' (`sglang_movieroll.rolling.profiles`). The frame
count and the size default to a 30 s, 480x864 video, an image-to-video one included (its image is scaled to cover
that size and centre-cropped), as MovieRoll's command line does for a model folder.

A multi-event video travels as a small JSON object in the prompt (`sglang_movieroll.rolling.request`): on
sglang 0.5.10.post1 no public entry point carries a model-specific request field to the worker -- the Python
API and the CLI build the *base* `SamplingParams` from the user's keyword arguments, and the OpenAI video
endpoint has a fixed request model.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

from sglang_movieroll.rolling.profiles import Chain
from sglang_movieroll.rolling.request import (
    DEFAULT_SEED,
    DEFAULT_VIDEO_LENGTH,
    parse_request,
)

_BASE = Chain.of("base")
_DISTILLED = Chain.of("distilled")


@dataclass
class MovieRollSamplingParams(SamplingParams):
    """The base chain. ``prompt`` is plain text or the JSON form of `rolling.request.parse_prompt`."""

    chain: ClassVar[Chain] = _BASE

    num_frames: int = DEFAULT_VIDEO_LENGTH
    fps: int = 24
    seed: int = DEFAULT_SEED
    negative_prompt: str = ""
    num_inference_steps: int = _BASE.default_steps
    # guidance_scale keeps SGLang's default: MovieRoll ignores it (whether classifier-free guidance runs is the
    # model directory's ``use_cfg``, its per-frame scale AnnealCFG's), and the worker warns about any other value

    # SGLang rounds the frame count to a multiple of the GPU count by default; a MovieRoll video has exactly
    # 4n + 1 pixel frames and a whole number of chunks
    adjust_frames: bool = False

    def _validate_with_pipeline_config(self, pipeline_config) -> None:
        super()._validate_with_pipeline_config(pipeline_config)
        # the client-side check: a bad request fails here, with a readable error, before it reaches a worker.
        # The command line hands over a list of prompts (``--prompt`` takes several) and a prompt file leaves
        # the prompt unset: check the other fields with every prompt there is (or with a stand-in), the file's
        # lines are checked by the worker. An ignored guidance_scale is reported by the worker alone, which knows
        # the model directory's use_cfg.
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        for prompt in prompts or [None]:
            fields = {
                **vars(self),
                "prompt": "(prompt read later)" if prompt is None else prompt,
                "guidance_scale": None,
            }
            parse_request(
                SimpleNamespace(**fields),
                self.chain,
                anchored=self.image_path is not None,
            )


@dataclass
class MovieRollDistilledSamplingParams(MovieRollSamplingParams):
    """The distilled chain: a fixed five-step ladder and no classifier-free guidance."""

    chain: ClassVar[Chain] = _DISTILLED

    num_inference_steps: int = _DISTILLED.default_steps
