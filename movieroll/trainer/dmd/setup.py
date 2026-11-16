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

"""Build the Stage-3 DMD models and their parallel runtime."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from movieroll.commons.constants import NUM_TRAIN_TIMESTEPS, SIGLIP_TOKEN_DIM
from movieroll.trainer.dmd.loss import MovieRollDMD, MovieRollDMDConfig
from movieroll.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from movieroll.schedulers.scheduling_flow_match_streaming import FlowMatchStreamingScheduler


@dataclass(frozen=True)
class DMDParallelContext:
    world_mesh: Any = None
    shard_group: Any = None
    sp_group: Any = None
    replicate_group: Any = None
    dp_rank: int = 0
    dp_degree: int = 1
    parallel_dims: Any = None


@dataclass(frozen=True)
class DMDRuntime:
    """What the Stage-3 trainer keeps from the setup.

    `generator` and `critic` are the FSDP-sharded transformers themselves, not the wrappers MovieRollDMD holds
    them in: the optimizers and the checkpoint manager are built on them, so their parameter names are the
    checkpoint keys. The text encoder, the teacher and the rollout pipeline stay alive through `dmd_model`.
    """
    vae: Any
    generator: Any
    critic: Any
    dmd_model: MovieRollDMD
    fsdp_dp_mesh: Any
    parallel_context: DMDParallelContext


def _describe_topology_problem(world_size: int, shard_world: int, sp_size: int):
    """Why a hybrid Stage-3 topology cannot run, or None if it can. Only the derived degrees matter.

    A function rather than two copies of the condition because the two callers in this module exist for
    different reasons -- `validate_runtime_topology` to fail before the 33 GB of weights are allocated, and
    `_DMDSetupBuilder` right where the mesh is built -- and they have to agree exactly. The three cases are
    the asserts the launch would otherwise hit:

    * `assert d >= 1` in ParallelDims._validate when the shard group is bigger than the world;
    * `assert parallel_dims.dp_replicate_enabled` (utils/fsdp.py, `fsdp_device_mesh`) when the replica count is exactly 1
      while the strategy is still hybrid -- `_apply_fsdp` rewrites hybrid to full only when the group IS the
      whole world, so a group in (world_size/2, world_size) reaches that assert, which has no message;
    * the product relation asserted in ParallelDims._validate on `replicas * sp_size`.

    Plus one that is specific to Stage 3: `_apply_fsdp` reads `world_mesh["dp_shard_sp"]` unconditionally, and
    `build_mesh` only creates that flattened name when `dp_shard > 1 or sp > 1`.
    Stage 1 and Stage 2 never ask for it, which is why this case is checked here and not in sft/ode.
    """
    if shard_world > world_size:
        return (f"shard group fsdp.shard_world_size={shard_world} is larger than world_size={world_size}")
    replicas = world_size // shard_world
    if world_size != shard_world and replicas < 2:
        return (f"hybrid needs at least 2 replicas, but fsdp.shard_world_size={shard_world} leaves "
                f"{replicas} at world_size={world_size}; use fsdp.sharding_strategy=full for a single "
                f"shard group, or lower fsdp.shard_world_size to at most {world_size // 2}")
    if world_size == shard_world:
        replicas = 1                                    # `_apply_fsdp` falls back to full sharding
    if world_size % (replicas * sp_size):
        return (f"fsdp.shard_world_size={shard_world} at world_size={world_size} gives {replicas} "
                f"replicas, and {replicas} * sp_size={sp_size} does not divide {world_size}, so there is "
                f"no whole number of shards per replica")
    if world_size // (replicas * sp_size) == 1 and sp_size == 1:
        return (f"fsdp.shard_world_size={shard_world} at world_size={world_size} leaves one rank per shard "
                f"group with no sequence parallelism, so the mesh has no 'dp_shard_sp' dimension for the "
                f"shard process group to be built from; raise fsdp.shard_world_size or sp_size")
    return None


def validate_runtime_topology(config, *, world_size: int, device: torch.device) -> None:
    """Validate process topology before allocating any large model."""

    if device.type != "cuda":
        raise RuntimeError("Stage3 DMD training requires CUDA")
    # One rank cannot work on either strategy: every mesh degree is then 1, so `build_mesh` creates no
    # 'dp_shard_sp' flattened name (built in ParallelDims.build_mesh) and `_apply_fsdp` reads it
    # unconditionally. The
    # failure is otherwise a bare `KeyError: Invalid mesh_dim_names ('dp_shard_sp',)`.
    if world_size < 2:
        raise ValueError(
            f"Stage3 DMD training needs at least 2 ranks, got world_size={world_size}: with one rank the "
            f"device mesh has no 'dp_shard_sp' dimension for the shard process group to be built from"
        )
    sp_size = int(config.sp_size)
    if sp_size <= 0 or world_size % sp_size:
        raise ValueError(
            f"world_size={world_size} must be divisible by sp_size={sp_size}"
        )
    sharding = str(config.fsdp.sharding_strategy)
    if sharding not in {"full", "hybrid"}:
        raise ValueError(f"unsupported FSDP sharding_strategy: {sharding!r}")
    # Hybrid sharding replicates whole shard groups of fsdp.shard_world_size ranks (the mesh built in
    # _DMDSetupBuilder; a single group falls back to full sharding). The group size reaches exactly one
    # expression -- `dp_replicate = self.world_size // shard_world` in `_apply_fsdp` -- a FLOOR division, so
    # `world_size % shard_world` is not a requirement: 8 // 3 = 2 replicas of 4 ranks builds the same mesh
    # as 4 does. The structural conditions are on the DERIVED degrees, and `_describe_topology_problem` states
    # them once for both callers.
    shard_world = int(config.fsdp.get("shard_world_size", 8))
    if sharding == "hybrid":
        problem = _describe_topology_problem(world_size, shard_world, sp_size)
        if problem is not None:
            raise ValueError(f"Stage3 {problem}")
    # ACS gives the critic's vision_in a gradient only on the windows it scores as i2v, and which window
    # that is follows each shard group's own start_phase draw. With several replicas the groups disagree, so
    # the HSDP all-reduce after a per-window backward would see different parameter sets and hang.
    replicas = 1 if sharding == "full" else world_size // shard_world
    if str(config.dmd.get("score_mode", "t2v")) == "acs" and replicas > 1:
        raise ValueError(
            "Stage3 dmd.score_mode='acs' needs a single FSDP shard group, but "
            f"fsdp.shard_world_size={shard_world} leaves {replicas} replicas at world_size={world_size}; "
            f"set fsdp.shard_world_size={world_size} (or run on {shard_world} ranks)"
        )


class _DMDSetupBuilder:
    """Private stateful builder used once by :func:`build_dmd_runtime`."""

    def __init__(
        self,
        *,
        config,
        world_size: int,
        device: torch.device,
        parameter_dtype: torch.dtype,
        compute_dtype: torch.dtype,
        logger: logging.Logger,
    ):
        self.config = config
        self.world_size = world_size
        self.device = device
        self.parameter_dtype = parameter_dtype
        self.compute_dtype = compute_dtype
        self.logger = logger

        self.parallel_dims = None
        self.world_mesh = None
        self.fsdp_dp_mesh = None
        self.sp_group = None
        self.shard_group = None
        self.replicate_group = None
        self.dp_rank = 0
        self.dp_degree = 1

    def _score_mode(self) -> str:
        """How the score models are conditioned: "t2v" (released) or "acs" (anchor-conditioned scoring)."""

        return str(self.config.dmd.get("score_mode", "t2v"))

    def build(self) -> DMDRuntime:
        vae = self._load_vae()
        text_encoder, byt5_kwarg = self._load_text_encoders()
        generator, real_score, fake_score, streaming_config = self._load_transformers()

        score_scheduler = FlowMatchDiscreteScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            shift=float(self.config.dmd.timestep_shift),
        )
        rollout_scheduler = FlowMatchStreamingScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            shift=float(self.config.dmd.timestep_shift),
        )

        from movieroll.pipelines.movieroll_dmd_pipeline import MovieRollDMDPipeline

        pipeline = MovieRollDMDPipeline(
            vae=vae,
            text_encoder=text_encoder,
            transformer=generator,
            scheduler=rollout_scheduler,
            precision="bf16" if bool(self.config.mixed_precision) else "fp32",
            vae_precision=str(self.config.vae_precision),
            prompt_format_font_path=self.config.multilingual_prompt_format_font_path,
            prompt_format_color_path=self.config.multilingual_prompt_format_color_path,
            byt5_kwarg=byt5_kwarg,
            device=self.device,
            is_eval=False,
            streaming_config=streaming_config,
            text_len=int(self.config.text_len),
        )
        pipeline._recompute_mid_min = int(
            self.config.dmd.recompute_mid_chunk_min
        )
        pipeline._recompute_mid_max = int(
            self.config.dmd.recompute_mid_chunk_max
        )
        # The node is mandatory in the Stage3 schema, so the sub-keys are read directly: the yaml is
        # the single source of truth for the probability and the candidate step indices.
        uniform_reg = self.config.dmd.uniform_timestep_regularization
        pipeline.configure_uniform_timestep_regularization(
            probability=float(uniform_reg.probability),
            candidate_step_indices=list(uniform_reg.candidate_step_indices),
        )
        pipeline.transformer.kv_cache_manager.verbose = False

        dmd_config = MovieRollDMDConfig(
            min_step_ratio=float(self.config.dmd.min_step_ratio),
            max_step_ratio=float(self.config.dmd.max_step_ratio),
            timestep_shift=float(self.config.dmd.timestep_shift),
            real_guidance_scale=float(self.config.dmd.real_guidance_scale),
            context_noise=float(self.config.dmd.context_noise),
            denoising_step_list=list(self.config.dmd.denoising_step_list),
            score_window_latent_frames=int(self.config.dmd.score_window_latent_frames),
            score_mode=self._score_mode(),
        )
        image_encoder = None
        if dmd_config.score_mode == "acs":
            # The SigLIP tower the bidirectional teacher / critic were trained to read for i2v: frozen, one
            # copy per rank (about 0.9 GB in fp16), outside FSDP.
            from movieroll.models.image_encoders import SiglipImageEncoder

            image_encoder = SiglipImageEncoder(
                path=str(self.config.siglip_path),
                device=self.device,
                logger=self.logger,
            )
            self.logger.info(
                "score_mode=acs: the window that starts at the i2v anchor is scored as i2v (anchor "
                "latent plus its SigLIP tokens, CFG uncond keeps the image); every later window is "
                "scored as t2v like the released recipe"
            )
        dmd_model = MovieRollDMD(
            generator_pipeline=pipeline,
            real_score=real_score,
            fake_score=fake_score,
            scheduler=score_scheduler,
            config=dmd_config,
            device=self.device,
            image_encoder=image_encoder,
        )

        # Build the semantic wrappers first, then shard the three underlying
        # transformers in place. Pipeline/model-adapter references stay valid
        # while the FSDP topology is established exactly once.
        if self.world_size > 1:
            self._apply_fsdp(
                generator=generator,
                real_score=real_score,
                fake_score=fake_score,
            )
        else:
            generator.to(self.device)
            real_score.to(self.device)
            fake_score.to(self.device)

        pipeline.set_sync_group(self.shard_group)
        dmd_model.set_dp_rank(self.dp_rank, self.dp_degree)
        dmd_model.set_random_sync_group(self.sp_group)

        generator.train()
        fake_score.train()
        real_score.eval()

        parallel = DMDParallelContext(
            world_mesh=self.world_mesh,
            shard_group=self.shard_group,
            sp_group=self.sp_group,
            replicate_group=self.replicate_group,
            dp_rank=self.dp_rank,
            dp_degree=self.dp_degree,
            parallel_dims=self.parallel_dims,
        )
        return DMDRuntime(
            vae=vae,
            generator=generator,
            critic=fake_score,
            dmd_model=dmd_model,
            fsdp_dp_mesh=self.fsdp_dp_mesh,
            parallel_context=parallel,
        )

    def _load_vae(self):
        from movieroll.models.autoencoders.vae_loader import load_vae

        vae, vae_path, spatial_ratio, temporal_ratio = load_vae(
            vae_precision=str(self.config.vae_precision),
            vae_path=str(self.config.vae_path),
            logger=self.logger,
            device=self.device,
            streaming=True,
        )
        vae.eval()
        vae.requires_grad_(False)
        # Streaming decode ignores tiling by contract; the single-frame
        # boundary encode for score-window re-encoding still benefits
        # from spatial tiling.
        vae.enable_tiling()
        self.logger.info(
            "VAE loaded from %s (spatial ratio=%s, temporal ratio=%s)",
            vae_path,
            spatial_ratio,
            temporal_ratio,
        )
        return vae

    def _load_text_encoders(self):
        from movieroll.models.text_encoders import PROMPT_TEMPLATE
        from movieroll.models.text_encoders.byT5 import load_glyph_byT5_v2
        from movieroll.utils.load import load_text_encoder

        text_device = (
            torch.device("cpu")
            if self.world_size > 1
            and bool(self.config.fsdp.text_encoder_cpu_offload)
            else self.device
        )
        text_encoder = load_text_encoder(
            str(self.config.text_encoder_path),
            int(self.config.text_len),
            PROMPT_TEMPLATE["li-dit-encode-image-json"],
            PROMPT_TEMPLATE["li-dit-encode-video-json"],
            text_device,
            text_encoder_precision=str(self.config.text_encoder_precision),
            logger=self.logger,
        )
        text_encoder.requires_grad_(False)
        text_encoder.eval()

        byt5_kwarg = load_glyph_byT5_v2(
            self.device,
            byT5_google_path=str(self.config.byt5_google_path),
            byT5_ckpt_path=str(self.config.byt5_ckpt_path),
            color_ann_path=str(self.config.multilingual_prompt_format_color_path),
            font_ann_path=str(self.config.multilingual_prompt_format_font_path),
            byt5_max_length=int(self.config.byt5_max_length),
        )
        byt5_kwarg["byt5_model"].eval()
        return text_encoder, byt5_kwarg

    def _load_transformers(self):
        from movieroll.commons.streaming_config import streaming_config_from_stage_block
        from movieroll.models.transformers.model_configs import MODEL_CONFIG, load_model
        from movieroll.models.transformers.score_transformer import MovieRollScoreTransformer
        from movieroll.utils.load import load_hunyuan_state_dict

        if self.config.model_type not in MODEL_CONFIG:
            raise ValueError(f"unknown model_type: {self.config.model_type!r}")
        architecture = dict(MODEL_CONFIG[self.config.model_type])
        streaming_config = streaming_config_from_stage_block(self.config.streaming_config)
        # ACS: only the score models build vision_in, to read the anchor's SigLIP tokens. The generator is
        # an ODE student without those weights and stays exactly as in the released recipe.
        score_vision_states_dim = (
            SIGLIP_TOKEN_DIM if self._score_mode() == "acs" else None
        )

        def create_model(
            checkpoint: str,
            *,
            generator: bool,
            frozen: bool,
            load_weights: bool,
        ):
            model_dtype = torch.bfloat16 if frozen else self.parameter_dtype
            if generator:
                # The streaming generator goes through the shipped factory, which already checks the
                # model_type and splats the size table; its attention is hard-coded per block group, so there
                # is no attn_mode to pass.
                model = load_model(
                    self.config.model_type,
                    {
                        "device": None,
                        "dtype": model_dtype,
                        "streaming_config": streaming_config,
                    },
                )
            else:
                model = MovieRollScoreTransformer(
                    **architecture,
                    attn_mode=str(self.config.attn_mode),
                    vision_states_dim=score_vision_states_dim,
                    device=None,
                    dtype=model_dtype,
                )
            if load_weights:
                # The shared loader owns the whole strict-load contract (file / .safetensors checks,
                # the one checkpoint-key whitelist, load_state_dict(strict=True)). No key-set
                # pre-check: the streaming blocks merge separate attn_q/k/v into attn_qkv inside their
                # load hook, so comparing key sets up front would report false mismatches.
                model = load_hunyuan_state_dict(
                    model, str(checkpoint), logger=self.logger
                )
            model = model.to(dtype=model_dtype)
            if frozen:
                model.eval()
                model.requires_grad_(False)
            return model

        resuming = bool(self.config.get("resume_checkpoint"))
        generator = create_model(
            str(self.config.generator_ckpt),
            generator=True,
            frozen=False,
            load_weights=not resuming,
        )
        real_score = create_model(
            str(self.config.real_score_ckpt),
            generator=False,
            frozen=True,
            load_weights=True,
        )
        fake_score = create_model(
            str(self.config.fake_score_ckpt),
            generator=False,
            frozen=False,
            load_weights=not resuming,
        )
        return generator, real_score, fake_score, streaming_config

    def _apply_fsdp(self, *, generator, real_score, fake_score) -> None:
        from movieroll.utils.fsdp import (
            apply_fsdp2,
            apply_fsdp_checkpointing,
            broadcast_from_first_replica,
            fsdp_device_mesh,
        )
        from movieroll.utils.load import get_no_split_modules
        from movieroll.commons.parallel_states import ParallelDims, initialize_sequence_parallel_state_v2

        requested_strategy = str(self.config.fsdp.sharding_strategy)
        effective_strategy = requested_strategy
        # Size of one FSDP shard group (dp_shard * sp). Large-memory nodes need a shard world bigger
        # than the single-node default of 8 to fit the Phase 2 activations; the extra nodes then
        # become dp_replicate instead.
        shard_world = int(self.config.fsdp.get("shard_world_size", 8))
        if shard_world < 1:
            raise ValueError(f"fsdp.shard_world_size must be >= 1, got {shard_world}")
        # Under `full` the next line pins dp_replicate to 1 and the group size is never read again, so there
        # is nothing about it to check on that branch -- it is inert, and any value gives the same mesh.
        if requested_strategy == "hybrid":
            problem = _describe_topology_problem(self.world_size, shard_world, int(self.config.sp_size))
            if problem is not None:
                raise ValueError(problem)
        dp_replicate = 1 if requested_strategy == "full" else self.world_size // shard_world
        if requested_strategy == "hybrid" and self.world_size == shard_world:
            effective_strategy = "full"
            self.logger.info(
                "Single shard group (%d ranks): using full sharding for the "
                "hybrid topology", shard_world,
            )

        parallel_dims = ParallelDims(
            dp_replicate=dp_replicate,
            dp_shard=-1,
            sp=int(self.config.sp_size),
            world_size=self.world_size,
        )
        world_mesh = parallel_dims.build_mesh(device_type="cuda")
        initialize_sequence_parallel_state_v2(parallel_dims, world_mesh)

        self.parallel_dims = parallel_dims
        self.world_mesh = world_mesh
        self.sp_group = (
            world_mesh["sp"].get_group() if parallel_dims.sp_enabled else None
        )
        self.shard_group = world_mesh["dp_shard_sp"].get_group()
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            self.dp_degree = dp_mesh.size()
            self.dp_rank = dp_mesh.get_local_rank()

        generator_blocks = get_no_split_modules(generator)
        self.fsdp_dp_mesh = fsdp_device_mesh(parallel_dims, world_mesh, effective_strategy)
        score_blocks = get_no_split_modules(real_score)
        param_dtype = self.compute_dtype
        reduce_dtype = torch.float32

        if bool(self.config.fsdp.generator_checkpointing):
            apply_fsdp_checkpointing(generator, generator_blocks, p=1)
        apply_fsdp2(
            model=generator,
            dp_mesh=self.fsdp_dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cpu_offload=bool(self.config.fsdp.generator_cpu_offload),
        )

        if bool(self.config.fsdp.real_score_checkpointing):
            apply_fsdp_checkpointing(real_score, score_blocks, p=1)
        apply_fsdp2(
            model=real_score,
            dp_mesh=self.fsdp_dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cpu_offload=bool(self.config.fsdp.real_score_cpu_offload),
        )

        if bool(self.config.fsdp.fake_score_checkpointing):
            apply_fsdp_checkpointing(fake_score, score_blocks, p=1)
        apply_fsdp2(
            model=fake_score,
            dp_mesh=self.fsdp_dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cpu_offload=bool(self.config.fsdp.fake_score_cpu_offload),
        )

        if (
            effective_strategy == "hybrid"
            and "dp_replicate" in world_mesh.mesh_dim_names
        ):
            self.replicate_group = world_mesh["dp_replicate"].get_group()
            broadcast_from_first_replica((generator, real_score, fake_score), self.replicate_group)
        dist.barrier()


def build_dmd_runtime(
    *,
    config,
    world_size: int,
    device: torch.device,
    parameter_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    logger: logging.Logger,
) -> DMDRuntime:
    """Build the only supported Stage3 model/parallel runtime."""

    validate_runtime_topology(config, world_size=world_size, device=device)
    return _DMDSetupBuilder(
        config=config,
        world_size=world_size,
        device=device,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        logger=logger,
    ).build()
