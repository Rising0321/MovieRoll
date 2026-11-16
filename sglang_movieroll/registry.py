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

"""Making MovieRoll known to SGLang Diffusion 0.5.10.post1.

sglang finds its pipelines by scanning its own ``runtime.pipelines`` package; an out-of-tree package has no
discovery hook on this version (``SGLANG_EXTERNAL_MODEL_PACKAGE`` is read only by the LLM runtime). What its
registries do offer to outside code is the public `register_configs` (sampling-parameter and pipeline-config
classes per model family) and `ModelRegistry.register_model` (component classes); for pipeline classes there is
only the dicts that discovery fills. `register` uses those, in an order that matters:

1. MovieRoll's component classes (the DiT and the VAE) go to `ModelRegistry`;
2. sglang's own pipeline discovery runs (`registry._discover_and_register_pipelines`). It returns at once when
   the pipeline registry is already filled ("run only once"), so putting MovieRoll's pipelines in before it
   would leave every built-in pipeline unregistered, without an error;
3. MovieRoll's pipeline classes are added to the two dicts discovery fills (`registry._PIPELINE_REGISTRY` and
   `registry._PIPELINE_CONFIG_REGISTRY`), and its config classes go through `register_configs`.

Those three names are private to sglang; they are the only ones this package uses for its registration.

It must run in EVERY process that builds the pipeline. sglang starts its GPU workers with the ``spawn`` start
method (importing `sglang.multimodal_gen` sets it), so a worker is a fresh interpreter that re-imports the
launching script as ``__mp_main__``: call `register` at the top level of that script (outside
``if __name__ == "__main__"``), as `sglang_movieroll.cli` and the examples do. Discovery imports sglang's
kernels, which needs a visible GPU.
"""

from __future__ import annotations

import warnings

_REGISTERED = False


def _register_component_models() -> None:
    """The DiT and the VAE, by architecture name (the ``_class_name`` of their folders in the model directory)."""
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

    from sglang_movieroll.runtime.models.dits.movieroll import (
        MovieRollTransformer3DModel,
    )
    from sglang_movieroll.runtime.models.vaes.hunyuanvideo15 import (
        AutoencoderKLHunyuanVideo15,
    )

    ModelRegistry.register_model(
        "MovieRollTransformer3DModel", MovieRollTransformer3DModel
    )
    # the VAE keeps the diffusers class name, so ``vae/`` is a plain diffusers component; sglang 0.5.10.post1 has
    # no class of that name, but a later one might
    if "AutoencoderKLHunyuanVideo15" in ModelRegistry.get_supported_archs():
        warnings.warn(
            "sglang already registers AutoencoderKLHunyuanVideo15; MovieRoll's streaming VAE replaces it "
            "in this process",
            stacklevel=2,
        )
    ModelRegistry.register_model(
        "AutoencoderKLHunyuanVideo15", AutoencoderKLHunyuanVideo15
    )


def register() -> None:
    """Idempotent; safe to call from every process."""
    global _REGISTERED
    if _REGISTERED:
        return
    from sglang.multimodal_gen import registry

    from sglang_movieroll.configs.pipeline_configs.movieroll import (
        register as register_configs_for_chains,
    )
    from sglang_movieroll.runtime.pipelines.movieroll_pipeline import EntryClass

    _register_component_models()
    registry._discover_and_register_pipelines()
    for pipeline_cls in EntryClass:
        registry._PIPELINE_REGISTRY[pipeline_cls.pipeline_name] = pipeline_cls
        registry._PIPELINE_CONFIG_REGISTRY[pipeline_cls.pipeline_name] = (
            pipeline_cls.pipeline_config_cls,
            pipeline_cls.sampling_params_cls,
        )
    register_configs_for_chains()
    _REGISTERED = True
