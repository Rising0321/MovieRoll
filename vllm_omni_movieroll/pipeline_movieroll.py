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
#
# ==============================================================================
# Portions adapted from vLLM-Omni v0.32.0 (commit 5917730a),
# vllm_omni/diffusion/models/hunyuan_video/pipeline_hunyuan_video_1_5.py
# (component loading from the model directory, `weights_sources`, the use of `CFGParallelMixin`)
# ==============================================================================
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""`MovieRollPipeline`: MovieRoll long-video generation as a vLLM-Omni diffusion pipeline.

Shaped like vLLM-Omni's HunyuanVideo-1.5 pipeline, whose components it reuses: the Qwen2.5-VL prompt encoding and the
byT5 glyph encoding (`HunyuanVideo15Pipeline._get_mllm_prompt_embeds` / `_get_byte5_prompt_embeds`), diffusers'
`AutoencoderKLHunyuanVideo15` and `FlowMatchEulerDiscreteScheduler` from the model directory,
the transformer's HunyuanVideo-1.5 layers (`movieroll_transformer`), `CFGParallelMixin` for the two guidance branches,
`ProgressBarMixin`, the profiler mixin and cooperative cancellation. What is MovieRoll's own is the body of `diffuse`:
instead of one denoising loop over a fixed latent, a window of noisy chunks rolls over the video (`rolling.run_rolling`),
keeping the clean history in a KV cache, and the video is decoded chunk by chunk (`streaming_vae`).

Component dtypes: the DiT in the pipeline dtype (bf16), Qwen2.5-VL fp16, byT5 fp32, VAE fp16.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Iterable
from typing import Any, ClassVar, Optional

import torch
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import ByT5Tokenizer, Qwen2_5_VLTextModel, Qwen2Tokenizer, T5EncoderModel
from vllm.config.logging import LoggingConfig
from vllm.logger import configure_logging_if_needed, init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.cancellation import check_request_cancellation
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import HunyuanVideo15Pipeline
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.utils.hf_utils import get_diffusion_model_index
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.platforms import current_omni_platform

from vllm_omni_movieroll.movieroll_transformer import MovieRollTransformer3DModel
from vllm_omni_movieroll.profiles import NUM_TRAIN_TIMESTEPS, MovieRollSettings, load_settings
from vllm_omni_movieroll.request import RollingRequest, parse_request
from vllm_omni_movieroll.rolling import (
    EventCondition,
    RollingProfile,
    TextTokens,
    chunk_layout,
    event_split_indices,
    run_rolling,
)
from vllm_omni_movieroll.scheduling_movieroll import RollingStepper, apply_context_noise, ladder
from vllm_omni_movieroll.streaming_vae import StreamingDecoder

logger = init_logger(__name__)

# the system message of MovieRoll's `li-dit-encode-video-json` prompt template (== HunyuanVideo-1.5's; the whitespace after
# each backslash continuation is part of it and moves the crop start)
# fmt: off
SYSTEM_MESSAGE = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."
# fmt: on
COMPONENTS = ["tokenizer", "text_encoder", "tokenizer_2", "text_encoder_2", "vae", "scheduler"]
# a MovieRoll checkpoint is evaluated with these; requests and deploy configs cannot reach the rest
TEXT_ENCODER_DTYPE = torch.float16
GLYPH_ENCODER_DTYPE = torch.float32
VAE_DTYPE = torch.float16

# the reasons behind every vLLM-Omni option MovieRoll refuses (also listed in the README)
REFUSALS = {
    "cache_dit": "Cache-DiT reuses block residuals across denoising calls; in a rolling window every Distilled call sees a "
                 "different window (one step per round), Base calls alternate with cache-encode / recompute passes over "
                 "other tokens, and per-frame timesteps break its residual-difference test, so no reuse is valid",
    "tea_cache": "TeaCache skips denoising calls from the change of the modulated input; with a rolling window and "
                 "per-frame timesteps consecutive calls are not the same latent, and there are no MovieRoll coefficients",
    "pipeline_parallel": "one MovieRoll request is a strictly sequential chain of window forwards (no micro-batches), so "
                         "pipeline stages would only add idle time; per-GPU memory relief is available through tensor "
                         "parallelism, HSDP and CPU / layerwise offload",
    "expert_parallel": "MovieRoll's transformer has no mixture-of-experts layers",
    "text_encoder_tp": "text_encoder_tp_size shards vLLM-Omni's own Qwen3-VL text encoder, not MovieRoll's Qwen2.5-VL",
    "ulysses_mode": "MovieRoll's sequence parallelism is model-owned (the exchanges happen around its KV cache) and needs "
                    "ulysses_mode='strict' (heads divisible by tensor_parallel_size * ulysses_degree)",
    "ring_allgather": "ring_degree and allgather_degree cannot be combined",
    "cfg_distilled": "the distilled chain has no classifier-free guidance, so there is no second branch for a CFG rank",
    "cfg_off": "this model directory turns classifier-free guidance off (use_cfg: false), so there is no second branch "
               "for a CFG rank",
    "cfg_size": "classifier-free guidance has two branches: cfg_parallel_size must be 1 or 2",
    "vae_patch": "MovieRoll decodes with an exact streaming (T-cache) decoder; its row-parallel decode spans every rank, "
                 "so vae_patch_parallel_size must be 1 or the world size, with vae_parallel_mode='tile' and no data "
                 "parallelism",
    "vae_tiling": "tiled VAE decoding blends tiles and changes the output; MovieRoll's streaming decoder is already "
                  "memory-bounded per chunk",
    "vae_slicing": "VAE slicing splits a batch, and a MovieRoll request is one video",
    "step_execution": "step execution, continuous batching and streaming output need the pipeline split into per-step "
                      "methods with all rolling state per request; not offered in this release",
    "dlo": "distributed layerwise offload is not validated for MovieRoll; use layerwise or model-level CPU offload",
    "lora": "no LoRA adapters exist for MovieRoll",
    "quantization": "online FP8 of MovieRoll's transformer changes the video too much (Distilled t2v 15.0 dB against "
                    "native, below the 20 dB bar; README, Features); run without --quantization",
    "paged_kv": "the scheduler-paged diffusion KV cache is HunyuanImage-3 specific; MovieRoll's KV cache is model-owned",
    "kv_cache_dtype": "quantized attention K/V (diffusion_kv_cache_dtype) is not validated for MovieRoll (lossy)",
    "request_batch": "one MovieRoll request is one video with its own KV cache, events and anchor; max_num_seqs must be 1",
}


def _refuse(key: str, detail: str = "") -> None:
    raise ValueError(f"MovieRoll: {detail + ': ' if detail else ''}{REFUSALS[key]}")


def check_supported_options(od_config: OmniDiffusionConfig, settings: MovieRollSettings, world_size: int) -> None:
    """Refuse, with the reason, every vLLM-Omni option MovieRoll cannot honour (before any weight is loaded)."""
    parallel = od_config.parallel_config
    if (getattr(od_config, "cache_backend", None) or "none") not in ("none", None):
        backend = od_config.cache_backend
        _refuse("tea_cache" if "tea" in str(backend) else "cache_dit", f"cache_backend={backend!r}")
    if int(getattr(parallel, "pipeline_parallel_size", 1) or 1) > 1:
        _refuse("pipeline_parallel", f"pipeline_parallel_size={parallel.pipeline_parallel_size}")
    if getattr(parallel, "enable_expert_parallel", False):
        _refuse("expert_parallel")
    if int(getattr(parallel, "text_encoder_tp_size", 1) or 1) > 1:
        _refuse("text_encoder_tp")
    if getattr(parallel, "ulysses_mode", "strict") != "strict" or getattr(parallel, "ulysses_a2a_permute", False):
        _refuse("ulysses_mode")
    if int(getattr(parallel, "ring_degree", 1)) > 1 and int(getattr(parallel, "allgather_degree", 1)) > 1:
        _refuse("ring_allgather")
    cfg = int(getattr(parallel, "cfg_parallel_size", 1) or 1)
    if cfg > 2:
        _refuse("cfg_size", f"cfg_parallel_size={cfg}")
    if cfg == 2 and not settings.use_cfg:
        _refuse("cfg_distilled" if settings.distilled else "cfg_off", "cfg_parallel_size=2")
    vae_parallel = int(getattr(parallel, "vae_patch_parallel_size", 1) or 1)
    data_parallel = int(getattr(parallel, "data_parallel_size", 1) or 1)
    if vae_parallel > 1 and (vae_parallel != world_size or getattr(parallel, "vae_parallel_mode", "tile") != "tile"
                             or data_parallel > 1):
        _refuse("vae_patch", f"vae_patch_parallel_size={vae_parallel}, world size {world_size}")
    if getattr(od_config, "vae_use_tiling", False):
        _refuse("vae_tiling", "vae_use_tiling")
    if getattr(od_config, "vae_use_slicing", False):
        _refuse("vae_slicing", "vae_use_slicing")
    if getattr(od_config, "step_execution", False) or getattr(od_config, "streaming_output", False):
        _refuse("step_execution")
    if int(getattr(od_config, "max_num_seqs", 1) or 1) > 1:
        _refuse("request_batch", f"max_num_seqs={od_config.max_num_seqs}")
    from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy

    if resolve_offload_strategy(od_config) is OffloadStrategy.DISTRIBUTED_LAYER_WISE:
        _refuse("dlo")
    if getattr(od_config, "lora_path", None):
        _refuse("lora")
    if getattr(od_config, "quantization_config", None) is not None:
        _refuse("quantization")
    if str(getattr(od_config, "diffusion_kv_mode", "")).lower().endswith("paged_scheduler"):
        _refuse("paged_kv")
    if getattr(od_config, "diffusion_kv_cache_dtype", None) not in (None, "auto", "float"):
        _refuse("kv_cache_dtype", f"diffusion_kv_cache_dtype={od_config.diffusion_kv_cache_dtype!r}")
    flow_shift = getattr(od_config, "flow_shift", None)
    shift = float(settings.profile["timestep_shift" if settings.distilled else "flow_shift"])
    if flow_shift is not None and float(flow_shift) != shift:
        raise ValueError(f"MovieRoll: flow_shift={flow_shift} is not this checkpoint's; its chain uses shift {shift}")


def _world_size() -> int:
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


TIME_LIMIT_VARIABLE = "VLLM_OMNI_DLO_DP_WAVE_TIMEOUT"


def compute_time_limit(od_config: OmniDiffusionConfig) -> Optional[float]:
    """The seconds vLLM-Omni allows one request's model call before it stops every worker, or None. Its multi-process
    and ray executors wait ``VLLM_OMNI_DLO_DP_WAVE_TIMEOUT`` (600 by default) for each `execute_model`; the in-process
    executor (offline on one GPU) has no limit. Not applied with data parallelism (its replicas run different requests)."""
    backend = od_config.distributed_executor_backend
    if backend is None:
        backend = "uni" if (od_config.num_gpus or 1) == 1 else "mp"
    if backend not in ("mp", "ray") or int(getattr(od_config.parallel_config, "data_parallel_size", 1) or 1) > 1:
        return None
    return float(os.environ.get(TIME_LIMIT_VARIABLE, 600.0))


class ComputeBudget:
    """Refuses a request that would outrun `compute_time_limit`, before the executor stops the workers.

    The projection is the elapsed time plus the median of the last units of the current phase (rolling chunks, then
    decoded chunks) times the units left, from the third unit of a phase on: the first ones carry the window filling up
    and compilation. Every rank of the request decides on the same projection, the maximum over ``group``."""

    def __init__(self, limit: Optional[float], group=None, device=None, clock=time.monotonic):
        self.limit, self.group, self.device, self.clock = limit, group, device, clock
        self.start = clock()
        self.begin(0)

    def begin(self, total: int) -> None:
        """A phase of ``total`` units."""
        self.total, self.done, self.durations, self.last = total, 0, [], self.clock()

    def tick(self, *_: Any) -> None:
        """One unit of the phase is done (arguments are ignored, so that it can be a progress callback)."""
        now = self.clock()
        self.done += 1
        self.durations.append(now - self.last)
        self.last = now
        if self.limit is None or self.done < 3:
            return
        recent = sorted(self.durations[-8:])
        projected = now - self.start + recent[len(recent) // 2] * (self.total - self.done)
        if self.group is not None:
            value = torch.tensor([projected], device=self.device)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX, group=self.group)
            projected = value.item()
        if projected > self.limit:
            need = 60 * math.ceil(1.3 * projected / 60)
            raise OmniClientError(
                f"MovieRoll: this request would compute about {projected:.0f} s, longer than vLLM-Omni allows one request "
                f"({TIME_LIMIT_VARIABLE}={self.limit:.0f} s, after which its executor stops every worker); set "
                f"{TIME_LIMIT_VARIABLE} to at least {need} before starting the engine, or ask for fewer frames")


def configure_process_logging(od_config: OmniDiffusionConfig) -> None:
    """Apply vLLM's logging configuration in the process that runs the pipeline, unless logging is set up there.

    vLLM configures its ``vllm`` logger (and so ``vllm_omni`` and this package, whose loggers hang below it) only where
    an entry point asks for it: ``vllm serve`` does in its API process, but vLLM-Omni v0.32.0's spawned diffusion
    workers (`WorkerProc.worker_main`) and the process of an offline `Omni` never do. There every INFO line, vLLM-Omni's
    own included, falls to the root logger's WARNING level, and warnings reach stderr bare, through Python's last-resort
    handler. vLLM-Omni has since added this same call to `WorkerProc.worker_main`: the stage's ``log_level`` unless
    ``VLLM_LOGGING_LEVEL`` is set, with ``VLLM_CONFIGURE_LOGGING`` and ``VLLM_LOGGING_CONFIG_PATH`` honoured as
    everywhere in vLLM. A process with any handler for vLLM's records, vLLM's or the user's, is left as it is."""
    if logging.getLogger("vllm").hasHandlers():
        return
    level = os.environ.get("VLLM_LOGGING_LEVEL", od_config.log_level).upper()
    configure_logging_if_needed(LoggingConfig(log_level=level))


def get_movieroll_pre_process_func(od_config: OmniDiffusionConfig):
    """Validate a request in the engine before it is queued, and fill in its size and frame count."""
    index = get_diffusion_model_index(od_config.model) or {}
    settings = load_settings(index, getattr(od_config, "model_config", None))

    def pre_process_func(request):
        if request.is_dummy_run():
            return request
        parsed = parse_request(request.prompt, request.sampling_params, settings)
        params = request.sampling_params
        if params.height is None and params.width is None:
            params.height, params.width = parsed.height, parsed.width
        params.num_frames = parsed.video_length
        return request

    return pre_process_func


class MovieRollPipeline(
    nn.Module, CFGParallelMixin, ProgressBarMixin, DiffusionPipelineProfilerMixin, SupportImageInput,
    SupportsComponentDiscovery,
):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "text_encoder_2"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _PROFILER_TARGETS: ClassVar[list[str]] = ["encode_prompt", "encode_image", "diffuse", "decode_latents"]
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"
    # a cancelled request stops between two windows of the rolling loop (`run_rolling`'s checkpoint)
    supports_request_cancellation: ClassVar[bool] = True
    # the engine's warm-up request becomes one 17-frame window (see `_warmup_request`)
    dummy_run_num_frames: ClassVar[int] = 17

    # HunyuanVideo-1.5's prompt encoding, reused as is (it reads the attributes set in __init__)
    _get_mllm_prompt_embeds = HunyuanVideo15Pipeline._get_mllm_prompt_embeds
    _get_byte5_prompt_embeds = HunyuanVideo15Pipeline._get_byte5_prompt_embeds

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        configure_process_logging(od_config)
        self.od_config = od_config
        self.device = get_local_device()
        model = od_config.model
        if not os.path.isdir(model):
            raise FileNotFoundError(f"{model} is not a MovieRoll model directory (see "
                                    "examples/offline_inference/movieroll/download_movieroll.py)")
        self.settings = load_settings(get_diffusion_model_index(model) or {}, od_config.model_config)
        check_supported_options(od_config, self.settings, _world_size())
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)
        local_files_only = True

        prefetch_subfolders(model, COMPONENTS, local_files_only=local_files_only)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = from_pretrained_with_prefetch(
            Qwen2_5_VLTextModel.from_pretrained, model, subfolder="text_encoder", prefetch_list=COMPONENTS,
            local_files_only=local_files_only, torch_dtype=TEXT_ENCODER_DTYPE,
        ).to(self.device).eval()
        self.tokenizer_2 = ByT5Tokenizer.from_pretrained(model, subfolder="tokenizer_2", local_files_only=local_files_only)
        # transformers' T5 encoder, the class MovieRoll's Glyph-ByT5 is: vLLM-Omni's tensor-parallel T5 encoder (the one
        # HunyuanVideo-1.5 uses) shards byt5-small's 6 heads over the tensor-parallel group, which would cap
        # tensor_parallel_size at 2 for a 0.2B encoder that gains nothing from sharding
        self.text_encoder_2 = from_pretrained_with_prefetch(
            T5EncoderModel.from_pretrained, model, subfolder="text_encoder_2", prefetch_list=COMPONENTS,
            local_files_only=local_files_only, torch_dtype=GLYPH_ENCODER_DTYPE,
        ).to(self.device).eval()
        self.vae = from_pretrained_with_prefetch(
            AutoencoderKLHunyuanVideo15.from_pretrained, model, subfolder="vae", prefetch_list=COMPONENTS,
            local_files_only=local_files_only, torch_dtype=VAE_DTYPE,
        ).to(self.device).eval()
        self.decoder = StreamingDecoder(self.vae)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler",
                                                                         local_files_only=local_files_only)

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, MovieRollTransformer3DModel)
        self.transformer = MovieRollTransformer3DModel(od_config=od_config, **transformer_kwargs)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model, subfolder="transformer", revision=None, prefix="transformer.", fall_back_to_pt=True,
            ),
        ]
        # the attributes HunyuanVideo-1.5's prompt encoding reads
        self.system_message = SYSTEM_MESSAGE
        self.prompt_template_encode_start_idx = 108
        self.tokenizer_max_length = int(self.settings.profile["text_len"])
        self.tokenizer_2_max_length = int(self.settings.profile["byt5_max_length"])

        if int(getattr(od_config.parallel_config, "vae_patch_parallel_size", 1) or 1) > 1:
            from vllm_omni.diffusion.distributed.parallel_state import get_world_group

            self.decoder.enable_row_parallel(get_world_group().device_group)
            logger.info("MovieRoll: row-parallel streaming VAE decode over %d ranks", _world_size())
        self.compute_limit = compute_time_limit(od_config)
        self.ranks_group = None
        if self.compute_limit is not None and _world_size() > 1:
            from vllm_omni.diffusion.distributed.parallel_state import get_world_group

            self.ranks_group = get_world_group().device_group
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=self._PROFILER_TARGETS,
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler,
        )

    # ------------------------------------------------------------------ conditioning
    @torch.no_grad()
    def _encode_one(self, prompt: str, glyph: bool = True) -> TextTokens:
        """The valid tokens of one prompt: Qwen2.5-VL hidden state and (when it quotes text) the byT5 glyph embedding."""
        embeds, mask = self._get_mllm_prompt_embeds([prompt], self.device, TEXT_ENCODER_DTYPE)
        valid = int(mask.sum().item())
        if valid <= 0 or not bool(mask[0, :valid].all()):
            raise RuntimeError("the text mask is not a prefix of valid tokens; right padding was expected")
        byt5 = None
        if glyph:
            glyph_embeds, glyph_mask = self._get_byte5_prompt_embeds([prompt], self.device, GLYPH_ENCODER_DTYPE)
            glyph_valid = int(glyph_mask.sum().item())
            byt5 = glyph_embeds[:, :glyph_valid] if glyph_valid > 0 else None
        return TextTokens(qwen=embeds[:, :valid], byt5=byt5)

    def encode_prompt(self, events: list[str], negative_prompt: Optional[str], guided: bool) -> list[EventCondition]:
        """One condition per event (each event encoded once); the guided chain's negative is the (default empty)
        negative prompt without glyph tokens, as MovieRoll's."""
        negative = self._encode_one(negative_prompt or "", glyph=False) if guided else None
        return [EventCondition(self._encode_one(text), negative) for text in events]

    @torch.no_grad()
    def encode_image(self, image, height: int, width: int) -> torch.Tensor:
        """The i2v anchor: the image scaled to cover, centre-cropped, VAE-encoded (posterior mode) to one latent frame."""
        from torchvision import transforms

        image = image.convert("RGB")
        scale = max(width / image.width, height / image.height)
        transform = transforms.Compose([
            transforms.Resize((int(round(image.height * scale)), int(round(image.width * scale))),
                              interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        pixels = transform(image).unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=self.vae.dtype)
        latents = self.vae.encode(pixels).latent_dist.mode()
        shift = getattr(self.vae.config, "shift_factor", None)
        if shift is not None:
            latents = latents - float(shift)
        return latents * float(self.vae.config.scaling_factor)

    # ------------------------------------------------------------------ denoising
    def _rolling_profile(self, request: RollingRequest) -> RollingProfile:
        p = self.settings.profile
        common = dict(
            chunk_size=self.settings.chunk_size, window_size=self.settings.window_size,
            kv_recompute_enabled=bool(p["kv_recompute_enabled"]), kv_recompute_threshold=int(p["kv_recompute_threshold"]),
            kv_recompute_recent_k=int(p["kv_recompute_recent_k"]),
        )
        if self.settings.distilled:
            return RollingProfile(mode="distilled", steps_per_window=1, force_recompute_at_event=True, noise_anchor=True,
                                  **common)
        return RollingProfile(mode="base", steps_per_window=request.num_inference_steps // self.settings.window_size,
                              guided=request.guided, cfg_min=float(p["cfg_min"]), cfg_max=float(p["cfg_max"]), **common)

    @staticmethod
    def model_input(latents: torch.Tensor) -> torch.Tensor:
        """``[latents, conditioning copy, mask]``: the 2C+1 input channels; at inference the last two are zeros (the i2v
        image reaches the model through the KV cache only)."""
        mask = torch.zeros((latents.shape[0], 1, *latents.shape[2:]), device=latents.device)
        return torch.cat([latents, torch.zeros_like(latents), mask], dim=1)

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        return self.transformer(mode="denoise", **kwargs)

    def _sampler(self, request: RollingRequest, latent_h: int, latent_w: int):
        """Ladder, per-frame stepper, noise and context-noise functions of the chain, seeded from the request (the same
        generator kinds and draw order as MovieRoll's CLI)."""
        p = self.settings.profile
        channels = self.transformer.in_channels
        timesteps, sigmas = ladder(
            self.scheduler, distilled=self.settings.distilled,
            num_inference_steps=request.num_inference_steps or 0, denoising_step_list=p.get("denoising_step_list", ()),
            device=self.device,
        )
        if self.settings.distilled:
            # a CPU generator: a device generator draws a different stream
            generator = torch.Generator(device="cpu").manual_seed(request.seed)

            def noise_fn(frames: int) -> torch.Tensor:
                return randn_tensor((1, channels, frames, latent_h, latent_w), generator=generator, device=self.device,
                                    dtype=self.dtype)

        else:
            generator = torch.Generator(device=self.device).manual_seed(request.seed)

            def noise_fn(frames: int) -> torch.Tensor:
                return torch.randn((1, channels, frames, latent_h, latent_w), dtype=self.dtype, device=self.device,
                                   generator=generator)

        sigma = self.settings.context_noise_sigma
        context_noise_fn = None
        if sigma > 0:
            def context_noise_fn(latents: torch.Tensor) -> torch.Tensor:
                return apply_context_noise(latents, sigma, generator)
        return timesteps, RollingStepper(sigmas), noise_fn, context_noise_fn

    def diffuse(self, request: RollingRequest, events: list[EventCondition], anchor: Optional[torch.Tensor],
                total: Optional[int] = None, on_chunk=None, budget: Optional[ComputeBudget] = None) -> torch.Tensor:
        """The rolling loop: all output latents ``[1, C, T, h, w]`` (anchor first for i2v)."""
        settings = self.settings
        anchored = anchor is not None
        chunk_sizes = chunk_layout(request.video_length, settings.chunk_size, anchored=anchored,
                                   strict=settings.distilled)
        splits, kept = event_split_indices(len(chunk_sizes), len(events))
        if len(kept) != len(events):
            if settings.distilled:
                raise ValueError("every event must keep at least one chunk; the video is too short for these events")
            logger.warning("events %s keep no chunk and are dropped", sorted(set(range(len(events))) - set(kept)))
        events = [events[i] for i in kept]
        profile = self._rolling_profile(request)
        latent_h, latent_w = request.height // 16, request.width // 16
        timesteps, stepper, noise_fn, context_noise_fn = self._sampler(request, latent_h, latent_w)
        cache = self.transformer.new_cache(latent_h, latent_w)
        guided = profile.guided

        def velocity_fn(latents, frame_timesteps, event, cfg_scale):
            hidden = self.model_input(latents)
            positive = {"hidden_states": hidden, "timestep": frame_timesteps, "text": event.positive,
                        "timestep_txt": frame_timesteps[:, :1], "cache": cache}
            negative = None
            if guided:
                negative = dict(positive, text=event.negative)
            return self.predict_noise_maybe_with_cfg(
                do_true_cfg=guided, true_cfg_scale=cfg_scale, positive_kwargs=positive, negative_kwargs=negative,
                cfg_normalize=False,
            )

        def step_fn(velocity, step_index, latents):
            return self.scheduler_step_maybe_with_cfg(velocity, step_index, latents, guided, per_request_scheduler=stepper)

        def encode_fn(latents):
            self.transformer(self.model_input(latents), mode="encode", cache=cache)

        def recompute_fn(latents, chunk_frames):
            self.transformer(self.model_input(latents), mode="recompute", cache=cache, chunk_frames=chunk_frames)

        total = total if total is not None else len(chunk_sizes) - (1 if anchored else 0)
        if budget is not None:
            budget.begin(total)
        try:
            with self.progress_bar(total=total) as bar:
                def progress(_done, _total):
                    bar.update(1)
                    if budget is not None:
                        budget.tick()

                result = run_rolling(
                    profile=profile, events=events, chunk_sizes=chunk_sizes, event_split_indices=splits,
                    timesteps=timesteps, velocity_fn=velocity_fn, step_fn=step_fn, encode_fn=encode_fn,
                    recompute_fn=recompute_fn, cache=cache, noise_fn=noise_fn, context_noise_fn=context_noise_fn,
                    anchor=anchor, on_chunk=on_chunk or progress,
                    checkpoint=check_request_cancellation,
                )
        finally:
            cache.clear()
        return result.latents

    def decode_latents(self, latents: torch.Tensor, budget: Optional[ComputeBudget] = None) -> torch.Tensor:
        """Streaming decode: pixels ``[1, 3, T, H, W]`` in [-1, 1] (CPU, the VAE's dtype)."""
        max_chunk_frames = int(self.settings.profile["vae_stream_decode_max_chunk_size"])
        if budget is not None:
            budget.begin(-(-latents.shape[2] // max(max_chunk_frames, 1)))
        return self.decoder.decode(latents, max_chunk_frames=max_chunk_frames, dtype=VAE_DTYPE, device=self.device,
                                   on_chunk=budget.tick if budget is not None else None)

    # ------------------------------------------------------------------ request
    def _parse(self, req: DiffusionRequestBatch) -> RollingRequest:
        if len(req.prompts) != 1:
            raise ValueError("MovieRoll generates one video per request")
        warmup = req.is_dummy_run()
        params = req.sampling_params
        if warmup:
            # the engine's warm-up request (512x512, 2 steps, a blank image when the model takes one): one 17-frame
            # window of the chain, the dummy request's steps and guidance_scale do not apply
            params.num_frames = self.dummy_run_num_frames
        request = parse_request(req.prompts[0], params, self.settings, warmup=warmup)
        for note in request.notes:
            logger.warning("MovieRoll request %s: %s", req.request_id, note)
        return request

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch, **kwargs: Any) -> DiffusionOutput:
        request = self._parse(req)
        # the executor's limit covers the whole model call
        budget = ComputeBudget(self.compute_limit, self.ranks_group, self.device)
        logger.info("MovieRoll request %s: mode=%s frames=%d size=%dx%d seed=%d events=%d i2v=%s cfg=%s "
                    "context_noise=%g", req.request_id, self.settings.sampling_mode, request.video_length,
                    request.height, request.width, request.seed, len(request.events), request.image is not None,
                    request.guided, self.settings.context_noise_sigma)
        guided = request.guided
        events = self.encode_prompt(request.events, request.negative_prompt, guided)
        anchor = self.encode_image(request.image, request.height, request.width) if request.image is not None else None
        latents = self.diffuse(request, events, anchor, budget=budget)
        profiled = getattr(self, "enable_diffusion_pipeline_profiler", False)
        if request.output_type == "latent":
            return DiffusionOutput(output=latents, stage_durations=self.stage_durations if profiled else None)
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        video = self.decode_latents(latents, budget=budget).contiguous()
        # read after the decode, so that its timing is in
        durations = self.stage_durations if profiled else None
        media = DiffusionMediaOutput(
            video=VideoMediaOutput(
                tensor=video,
                spec=VideoTensorSpec(
                    layout=VideoTensorLayout.BCTHW,
                    encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                    value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
                ),
            )
        )
        return DiffusionOutput(media=media, stage_durations=durations)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)


__all__ = ["MovieRollPipeline", "get_movieroll_pre_process_func", "check_supported_options", "NUM_TRAIN_TIMESTEPS"]
