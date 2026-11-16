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

"""MovieRoll for vLLM-Omni: the ``MovieRollPipeline`` diffusion pipeline, registered as an out-of-tree model.

Importing the package is cheap; `register` is what vLLM-Omni's plugin loader calls (entry point
``vllm_omni.general_plugins: movieroll``) in every process. The pipeline, transformer and pre-process function are exported
lazily, as vLLM-Omni's in-tree model packages export theirs. The one pipeline class serves both chains, under the two
architecture names of `profiles.PIPELINE_CLASS` (``MovieRollPipeline`` for Base, ``MovieRollDistilledPipeline`` for
Distilled): a model directory's ``_class_name`` picks one, and vLLM-Omni resolves the chain's defaults by it.
"""

import logging

# vLLM's handler sits on the "vllm" logger; route this package's loggers through it, as vLLM-Omni does for its own
# ("vllm_omni.logger"), so that its info lines (request summary, PS-VAE) reach the engine log. vLLM-Omni v0.32.0 leaves
# that logger unconfigured in the process that runs the pipeline; `pipeline_movieroll.configure_process_logging` does it
logging.getLogger(__name__).parent = logging.getLogger("vllm")

PIPELINE_MODULE = "vllm_omni_movieroll.pipeline_movieroll"

__all__ = [
    "register",
    "MovieRollPipeline",
    "MovieRollTransformer3DModel",
    "get_movieroll_pre_process_func",
]


def register() -> None:
    """Make MovieRoll's two architectures known to vLLM-Omni (idempotent, safe in every process): the pipeline class
    through `register_diffusion_model`, the one-stage video topology of each (`topology`) through `register_pipeline`,
    which the video API needs to accept the model, and the defaults of the shared examples and of `/v1/videos`
    (`video_defaults`)."""
    try:
        from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline
        from vllm_omni.diffusion.registry import register_diffusion_model
    except ImportError:        # called through vLLM's plugin group in an environment without vLLM-Omni
        return

    from vllm_omni_movieroll.profiles import PIPELINE_CLASS
    from vllm_omni_movieroll.topology import MOVIEROLL_PIPELINES
    from vllm_omni_movieroll.video_defaults import register_model_extras

    for model_arch in PIPELINE_CLASS.values():
        register_diffusion_model(
            model_arch,
            PIPELINE_MODULE,
            "MovieRollPipeline",
            pre_process_func_name="get_movieroll_pre_process_func",
        )
    for topology in MOVIEROLL_PIPELINES:
        if OMNI_PIPELINES.get(topology.model_type) is not topology:
            register_pipeline(topology)
    register_model_extras()


def __getattr__(name):
    if name in ("MovieRollPipeline", "get_movieroll_pre_process_func"):
        from vllm_omni_movieroll import pipeline_movieroll

        return getattr(pipeline_movieroll, name)
    if name == "MovieRollTransformer3DModel":
        from vllm_omni_movieroll.movieroll_transformer import MovieRollTransformer3DModel

        return MovieRollTransformer3DModel
    raise AttributeError(f"module 'vllm_omni_movieroll' has no attribute {name!r}")
