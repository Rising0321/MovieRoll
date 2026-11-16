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

"""`MovieRollPipeline` and `MovieRollDistilledPipeline`: long-video generation as SGLang Diffusion pipelines.

Shaped like the pipelines SGLang ships (`ComposedPipelineBase` subclasses with ``pipeline_name``,
``_required_config_modules``, ``create_pipeline_stages``, an ``EntryClass``): every module of the model
directory is loaded by SGLang's own component loaders, and the stages are the standard ones with the narrow
MovieRoll subclasses of `stages/model_specific_stages/movieroll.py` in their places.

The two chains are two pipeline classes, as SGLang's Helios has two, because the request defaults differ (50
steps with guidance against five steps without) and SGLang resolves the defaults from the class: the
``model_index.json`` of a model directory names the one that goes with its checkpoint, and its
``movieroll_overrides`` hold what the MovieRoll model folder it was built from sets differently from that chain
(the image-to-video folder's context noise; `tools.assemble_model_dir`).
"""

from __future__ import annotations

import json
import os

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from sglang_movieroll.configs.pipeline_configs.movieroll import (
    MovieRollDistilledPipelineConfig,
    MovieRollPipelineConfig,
)
from sglang_movieroll.configs.sample.movieroll import (
    MovieRollDistilledSamplingParams,
    MovieRollSamplingParams,
)
from sglang_movieroll.runtime.pipelines_core.stages.model_specific_stages.movieroll import (
    MovieRollImageVAEEncodingStage,
    MovieRollInputValidationStage,
    MovieRollRollingDenoisingStage,
    MovieRollTextEncodingStage,
)

logger = init_logger(__name__)


class MovieRollPipeline(LoRAPipeline, ComposedPipelineBase):
    """The base chain: 50 steps with classifier-free guidance (the per-frame AnnealCFG schedule)."""

    pipeline_name = "MovieRollPipeline"
    pipeline_config_cls = MovieRollPipelineConfig
    sampling_params_cls = MovieRollSamplingParams
    is_video_pipeline = True

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "text_encoder_2",
        "tokenizer_2",
        "vae",
        "transformer",
        "scheduler",
    ]

    def load_modules(self, server_args: ServerArgs, loaded_modules=None):
        # what this pipeline cannot do is refused before a single weight is read
        self._check_config(server_args)
        self._refuse_unsupported(server_args)
        return super().load_modules(server_args, loaded_modules)

    def initialize_pipeline(self, server_args: ServerArgs):
        self._load_model_settings(server_args)
        self._maybe_enable_vae_parallel_decode(server_args)

    def _load_model_settings(self, server_args: ServerArgs) -> None:
        """The model directory's ``movieroll_overrides`` onto the pipeline config, where the stages read the chain.

        Read here, from the ``model_index.json`` of the directory being loaded, as SGLang's
        `ComposedPipelineBase.load_modules` reads a model index's ``boundary_ratio`` onto the pipeline config.
        """
        with open(os.path.join(self.model_path, "model_index.json")) as handle:
            overrides = json.load(handle).get("movieroll_overrides") or {}
        config = server_args.pipeline_config
        config.movieroll_overrides = dict(overrides)
        chain = config.chain  # checks the overrides
        if server_args.enable_cfg_parallel and not chain.use_cfg:
            raise ValueError(
                "--enable-cfg-parallel puts the two classifier-free-guidance branches on two ranks, and this "
                "model directory turns guidance off (use_cfg: false), so there is no second branch. Use "
                "--ulysses-degree / --ring-degree / --tp-size to spread the video over several GPUs."
            )
        logger.info(
            "%s chain, context noise %g%s",
            chain.sampling_mode,
            chain.context_noise_sigma,
            f", model directory settings {overrides}" if overrides else "",
        )

    def _check_config(self, server_args: ServerArgs) -> None:
        """The config SGLang resolved must be this chain's: a wrong one would silently change every number.

        Both the class and the chain are checked. The class alone is not enough: the distilled config is a
        subclass of the base one, so a base pipeline holding a distilled config passes `isinstance` while
        running five steps without guidance, and `sampling_mode` is exactly what decides that.
        """
        config = server_args.pipeline_config
        expected = self.pipeline_config_cls
        chain = getattr(config, "sampling_mode", None)
        if not isinstance(config, expected) or chain != expected.sampling_mode:
            raise ValueError(
                f"{self.pipeline_name} needs a {expected.__name__} ({expected.sampling_mode} chain), but "
                f"sglang resolved {type(config).__name__} ({chain} chain) for this model path. Its detectors "
                f"match on the path, so a directory whose name contains another model family's name can win; "
                f"pass --model-id MovieRoll-Base or --model-id MovieRoll-Distilled, or rename the directory."
            )

    def _refuse_unsupported(self, server_args: ServerArgs) -> None:
        """What SGLang offers that MovieRoll cannot do, refused at start-up rather than mid-video."""
        backend = server_args.attention_backend
        if backend == "sage_attn_3":
            raise ValueError(
                "--attention-backend sage_attn_3 is SageAttention3, whose FP4 kernels run on Blackwell GPUs "
                "only; on Hopper use --attention-backend sage_attn (SageAttention 2) or the default fa."
            )
        supported = {
            str(b)
            for b in server_args.pipeline_config.dit_config.arch_config._supported_attention_backends
        }
        if backend is not None and backend not in supported:
            # sglang's selector would fall back to another backend without saying so
            raise ValueError(
                f"--attention-backend {backend} is not supported by MovieRoll; use one of {sorted(supported)}."
            )
        if server_args.use_fsdp_inference and server_args.tp_size > 1:
            raise ValueError(
                "--use-fsdp-inference and --tp-size shard the same linear layers; sglang loads one or the "
                "other. Use FSDP alone, or tensor parallelism alone."
            )
        if (
            server_args.enable_cfg_parallel
            and server_args.pipeline_config.sampling_mode == "distilled"
        ):
            raise ValueError(
                "--enable-cfg-parallel puts the two classifier-free-guidance branches on two ranks, and the "
                "distilled chain has no second branch (its guidance is distilled into the weights). Use "
                "--ulysses-degree / --ring-degree / --tp-size to spread a distilled video over several GPUs."
            )
        config = server_args.pipeline_config
        if config.dit_precision != "bf16":
            raise ValueError(
                f"--dit-precision {config.dit_precision}: the MovieRoll transformer runs in bf16, the precision "
                "both released chains were evaluated in; it has never been run in another one."
            )
        if (
            server_args.transformer_weights_path
            or server_args.nunchaku_config is not None
        ):
            raise ValueError(
                "--transformer-weights-path / --enable-svdquant load a pre-quantized transformer (FP8, NVFP4, "
                "ModelSlim, SVDQuant), and no quantized MovieRoll checkpoint exists to load or to verify against; "
                "the transformer is loaded from the model directory, in bf16."
            )
        if config.vae_tiling:
            raise ValueError(
                "--vae-tiling blends overlapping spatial tiles, which changes the video; MovieRoll decodes in "
                "causal temporal chunks instead (exact), and --vae-sp spreads those over the GPUs."
            )
        if config.vae_slicing:
            raise ValueError(
                "--vae-slicing splits a batch for the decode, and a MovieRoll request is one video, decoded in "
                "causal temporal chunks; --vae-sp spreads the decode over the GPUs instead."
            )
        if server_args.comfyui_mode:
            raise ValueError(
                "--comfyui-mode drives one latent per denoising step through a pass-through scheduler; "
                "MovieRoll denoises a rolling window of chunks."
            )
        import sglang.multimodal_gen.envs as envs

        if getattr(envs, "SGLANG_CACHE_DIT_ENABLED", False):
            raise ValueError(
                "SGLANG_CACHE_DIT_ENABLED caches the residuals of one latent across consecutive denoising "
                "steps; a rolling window holds different chunks every round, so a cached residual would belong "
                "to other frames."
            )

    def set_lora(self, *args, **kwargs):
        """Refused wherever it is asked for (``--lora-path`` at start-up, ``/v1/set_lora`` later)."""
        raise ValueError(
            "MovieRoll has no LoRA adapter: --lora-path / /v1/set_lora are refused because there is no released "
            "MovieRoll LoRA to verify a merge against. The rename table a LoRA would need is the identity (the "
            "checkpoint's names are sglang's own), so this is a verification gap, not a missing mapping."
        )

    def _maybe_enable_vae_parallel_decode(self, server_args: ServerArgs) -> None:
        """``--vae-sp`` shards the streaming decode over the rows of the sequence-parallel group (PS-VAE)."""
        from sglang.multimodal_gen.runtime.distributed import (
            get_sp_group,
            get_sp_world_size,
        )

        if server_args.pipeline_config.vae_sp and get_sp_world_size() > 1:
            self.get_module("vae").enable_parallel_decode(get_sp_group().device_group)
            logger.info(
                "PS-VAE: row-parallel streaming decode over %d ranks",
                get_sp_world_size(),
            )

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(MovieRollInputValidationStage())
        self.add_stage(
            MovieRollTextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                ],
            )
        )
        self.add_stage(
            MovieRollImageVAEEncodingStage(
                vae=self.get_module("vae"), component_name="vae"
            )
        )
        self.add_standard_timestep_preparation_stage()
        self.add_stage(
            MovieRollRollingDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
            "denoising_stage",
        )
        self.add_stage(
            DecodingStage(
                vae=self.get_module("vae"), pipeline=self, component_name="vae"
            ),
            "decoding_stage",
        )


class MovieRollDistilledPipeline(MovieRollPipeline):
    """The distilled chain: five steps, no guidance."""

    pipeline_name = "MovieRollDistilledPipeline"
    pipeline_config_cls = MovieRollDistilledPipelineConfig
    sampling_params_cls = MovieRollDistilledSamplingParams


EntryClass = [MovieRollPipeline, MovieRollDistilledPipeline]
