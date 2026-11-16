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

"""MovieRoll's entries in `vllm_omni.model_extras`: the defaults of vLLM-Omni's shared video examples and of `/v1/videos`.

vLLM-Omni keys these on the pipeline's architecture name, one entry per chain (`profiles.PIPELINE_CLASS`):

* ``video_generation_defaults_builder``: what a request that leaves them out gets -- 480 x 864, 721 frames at 24 fps, and
  the chain's step count (Base 50, Distilled its five-step ladder). Without it the shared examples fall back to their
  Wan2.2 presets (text-to-video 1280 x 720, 40 steps, guidance 4.0; image-to-video 16 fps, guidance 5.0, a size taken
  from the image). No guidance scale is declared (``None``) for either chain: MovieRoll ignores a request's
  ``guidance_scale`` (the directory's ``use_cfg`` and AnnealCFG decide), and with ``None`` the shared examples send
  none, so their requests carry no value to warn about; ``/v1/videos`` does not read this field;
* ``reference_image_size_resolver``: the pipeline sizes the image-to-video frame itself (scaled to cover the video,
  centre-cropped; `MovieRollPipeline.encode_image`), so the examples and the server hand it over as it is instead of
  resizing it to the video size, which would stretch it.

vLLM-Omni 0.32.0 declares these for its in-tree models in the table of `vllm_omni.model_extras.registry` and has no
call for an out-of-tree model, so `register` adds MovieRoll's two entries to that table (``_EXTRA_SPECS``), from the
plugin entry point like the rest of the registration. The values are the built-in chains': a directory's
``movieroll_overrides`` do not change them.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

from vllm_omni_movieroll.profiles import PIPELINE_CLASS, VAE_SPATIAL, get_profile
from vllm_omni_movieroll.request import DEFAULT_VIDEO_LENGTH


def _defaults(sampling_mode: str) -> VideoGenerationDefaults:
    profile = get_profile(sampling_mode)
    distilled = sampling_mode == "distilled"
    return VideoGenerationDefaults(
        width=int(profile["width"]),
        height=int(profile["height"]),
        num_frames=DEFAULT_VIDEO_LENGTH,
        num_inference_steps=len(profile["denoising_step_list"]) if distilled else int(profile["num_inference_steps"]),
        fps=float(profile["video_fps"]),
        output="movieroll_output.mp4",
        guidance_scale=None,
        dimension_multiple=VAE_SPATIAL,
        default_negative_prompt="",
    )


def _builder(sampling_mode: str):
    def build(extra_body: Optional[Mapping[str, Any]] = None) -> VideoGenerationDefaults:
        del extra_body   # nothing in a request changes MovieRoll's defaults
        return _defaults(sampling_mode)

    return build


def _owns_reference_image_size(*, model: Optional[str], revision: Optional[str] = None) -> bool:
    del model, revision
    return True


def register_model_extras() -> None:
    """Add MovieRoll's two entries to vLLM-Omni's model-extras table (idempotent; an entry already there is kept)."""
    from vllm_omni.model_extras import registry

    for sampling_mode, class_name in PIPELINE_CLASS.items():
        registry._EXTRA_SPECS.setdefault(class_name, {
            "video_generation_defaults_builder": _builder(sampling_mode),
            "reference_image_size_resolver": _owns_reference_image_size,
        })
