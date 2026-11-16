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

"""MovieRoll's stages: narrow subclasses of SGLang's own, plus the rolling denoising loop.

The pipeline is SGLang's standard chain of stages, and each of these is the smallest change to one of them:

* `MovieRollInputValidationStage` validates a request with MovieRoll's rules, gives it the profile's size when it
  has none (an image-to-video one too, as the command line does for a model folder) and plans the chunks and event
  boundaries;
* `MovieRollTextEncodingStage` runs the standard encoders once per event, and the glyph encoder only for the
  events that quote text;
* `MovieRollImageVAEEncodingStage` encodes the one input frame into the anchor latent;
* `MovieRollRollingDenoisingStage` replaces the standard loop over one latent with the rolling window over the
  whole video (the standard `TimestepPreparationStage` before it and the standard `DecodingStage` after it are
  used unchanged).

Everything that is MovieRoll's own arithmetic lives in `sglang_movieroll.rolling`, which has no framework
import; these stages only move tensors between it and SGLang's `Req`.
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import ImageOps
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vision_utils import load_image
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

from sglang_movieroll.configs.pipeline_configs.movieroll import extract_glyph_texts
from sglang_movieroll.rolling.kv_cache import KVCache
from sglang_movieroll.rolling.loop import SerialGuidance, run_rolling
from sglang_movieroll.rolling.profiles import Chain
from sglang_movieroll.rolling.request import RollingJob, parse_request, plan_job
from sglang_movieroll.rolling.schedule import apply_context_noise
from sglang_movieroll.rolling.tokens import EventCondition, TextTokens
from sglang_movieroll.runtime.distributed.layout import guidance_group

logger = init_logger(__name__)

# sglang's scheduler client waits this long for the worker's answer and then abandons the request
# (`zmq.RCVTIMEO` in `runtime/scheduler_client.py`, on both the sync and the async path; the declared
# SGLANG_DIFFUSION_ENGINE_ITERATION_TIMEOUT_S is read nowhere). A video that would take longer is refused as soon as
# its pace is known, instead of being generated for an hour and a half and then thrown away.
CLIENT_TIMEOUT_S = 6000

JOB_KEY = "movieroll_job"  # Req.extra key of the planned request


def _chain_of(server_args: ServerArgs) -> Chain:
    return server_args.pipeline_config.chain


class MovieRollInputValidationStage(InputValidationStage):
    """Request -> a planned `RollingJob`; sets the size, the seed and the chain's noise generator."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        chain = _chain_of(server_args)
        image = None
        if batch.image_path is not None:
            path = (
                batch.image_path[0]
                if isinstance(batch.image_path, list)
                else batch.image_path
            )
            # the same image a phone stores sideways, upright, as MovieRoll's command line opens it
            image = ImageOps.exif_transpose(load_image(path)).convert("RGB")

        request = parse_request(batch, chain, image=image)
        job = plan_job(request, chain)
        batch.extra[JOB_KEY] = job
        batch.height, batch.width, batch.num_frames = (
            request.height,
            request.width,
            request.video_length,
        )
        batch.seed = request.seed
        batch.num_inference_steps = request.num_inference_steps
        # the chain's own device decides the random stream, whatever the request asked for
        batch.generator_device = server_args.pipeline_config.generator_device
        self._generate_seeds(batch, server_args)
        # the flag the standard stages verify against: the model directory's, whatever guidance_scale says
        batch.do_classifier_free_guidance = chain.use_cfg
        if batch.negative_prompt is None:
            # what MovieRoll's own pipeline encodes when a guided run is given no negative prompt
            batch.negative_prompt = ""
        batch.condition_image = request.image
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_not_none)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        return result


class MovieRollTextEncodingStage(TextEncodingStage):
    """The standard encoders, run once per event; the glyph encoder only where the event quotes text."""

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """The standard checks, except that the unconditional prompt may be empty.

        The standard stage demands a non-empty negative prompt whenever guidance is on, while MovieRoll's own
        guided path encodes the empty string through the same prompt template when no negative prompt is given
        (``encode_prompt``: ``uncond_tokens = [""] * batch_size``), which is what every Base checkpoint was
        evaluated with. So the field is checked as a string instead.
        """
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        result.add_check(
            "negative_prompt", batch.negative_prompt, V.string_or_list_strings
        )
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )
        result.add_check("prompt_embeds", batch.prompt_embeds, V.is_list)
        result.add_check(
            "negative_prompt_embeds", batch.negative_prompt_embeds, V.none_or_list
        )
        return result

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        job: RollingJob = batch.extra[JOB_KEY]
        guided = _chain_of(server_args).use_cfg

        negative: Optional[TextTokens] = None
        if guided:
            negative = TextTokens(
                qwen=self._encode_one(
                    job.request.negative_prompt, server_args, glyph=False
                )
            )

        job.events = []
        for text in job.event_texts:
            glyph = bool(extract_glyph_texts(text))
            job.events.append(
                EventCondition(
                    positive=TextTokens(
                        qwen=self._encode_one(text, server_args, glyph=False),
                        byt5=(
                            self._encode_one(text, server_args, glyph=True)
                            if glyph
                            else None
                        ),
                    ),
                    negative=negative,
                )
            )

        # the standard fields of the batch carry the first event, which is what a single-event request is
        first = job.events[0]
        batch.prompt_embeds = list(first.positive.as_embeddings())
        batch.negative_prompt_embeds = (
            list(negative.as_embeddings()) if negative is not None else []
        )
        batch.do_classifier_free_guidance = guided
        return batch

    def _encode_one(
        self, text: str, server_args: ServerArgs, *, glyph: bool
    ) -> torch.Tensor:
        """One encoder on one prompt, with the padding dropped (the padded keys are masked anyway)."""
        index = 1 if glyph else 0
        embeds, masks, _ = self.encode_text(
            text, server_args, encoder_index=[index], return_attention_mask=True
        )
        embed, mask = embeds[0], masks[0]
        if glyph:
            valid = int(mask.sum().item())
        else:
            # the prompt encoder's hidden state has the template's prefix cropped off, and so must its mask
            crop = server_args.pipeline_config.text_encoder_configs[
                0
            ].arch_config.crop_start
            mask = mask[:, crop:]
            valid = int(mask.sum().item())
            if valid <= 0 or not bool(mask[0, :valid].all()):
                raise RuntimeError(
                    "the text mask is not a prefix of valid tokens; right padding was expected"
                )
        return embed[:, :valid]

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt_embeds", batch.prompt_embeds, V.is_list)
        return result


class MovieRollImageVAEEncodingStage(ImageVAEEncodingStage):
    """The one input frame of an image-to-video request -> the anchor latent ``[1, C, 1, h, w]``."""

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        job: RollingJob = batch.extra[JOB_KEY]
        request = job.request
        if not request.anchored:
            return batch
        from torchvision import transforms

        self.load_model()
        image = request.image
        scale = max(request.width / image.width, request.height / image.height)
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (int(round(image.height * scale)), int(round(image.width * scale))),
                    interpolation=transforms.InterpolationMode.LANCZOS,
                ),
                transforms.CenterCrop((request.height, request.width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        pixels = transform(image).unsqueeze(0).unsqueeze(2)
        latents = self.vae.encode(
            pixels.to(get_local_torch_device(), next(self.vae.parameters()).dtype)
        ).mode()
        scaling, shift = server_args.pipeline_config.get_decode_scale_and_shift(
            latents.device, latents.dtype, self.vae
        )
        if shift is not None:
            latents = latents - float(shift)
        job.anchor = latents * float(scaling)
        batch.image_latent = job.anchor
        self.offload_model()
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()


def refuse_if_it_cannot_finish_in_time(done: int, total: int, elapsed: float) -> None:
    """Refuse a video the client would abandon, as soon as the pace is known.

    Checked once, at the second chunk: the first also pays for filling the window, and two chunks into a video
    is early enough that nothing is wasted. ``elapsed`` is the time the first ``done`` chunks took.
    """
    if done != min(2, total):
        return
    projected = elapsed / done * total
    if projected > CLIENT_TIMEOUT_S:
        raise ValueError(
            f"this video needs about {projected / 60:.0f} min of denoising, and sglang's client abandons a "
            f"request after {CLIENT_TIMEOUT_S / 60:.0f} min ({total} chunks at {projected / total:.1f} s each). "
            f"Spread it over more GPUs (--ulysses-degree, and --enable-cfg-parallel for the base chain), or "
            f"generate it in shorter pieces."
        )


def noise_functions(
    chain: Chain, generator: torch.Generator, latent_shape, device, dtype
):
    """The chain's ``noise_fn(frames)`` and ``context_noise_fn(latents)`` (None when it has no context noise).

    Both draw from the request's generator, on the generator's own device (the chain's: CPU for Distilled, GPU
    for Base) and in the order the rolling loop asks, which is what makes a video comparable with MovieRoll's
    command line at the same seed. ``latent_shape`` is ``(channels, latent height, latent width)``.
    """

    def noise_fn(frames: int) -> torch.Tensor:
        return randn_tensor(
            (1, latent_shape[0], frames, *latent_shape[1:]),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    sigma = chain.context_noise_sigma
    if sigma <= 0:
        return noise_fn, None

    def context_noise_fn(latents: torch.Tensor) -> torch.Tensor:
        return apply_context_noise(latents, sigma, generator=generator)

    return noise_fn, context_noise_fn


class TransformerDenoiser:
    """The `Denoiser` the rolling loop drives, over the DiT: adds the unused conditioning channels and calls it.

    The checkpoint's input has ``2 * C + 1`` channels, ``[latents, conditioning copy, mask]``; at inference the
    last two blocks are zeros (the image-to-video frame reaches the model through the KV cache only).
    """

    def __init__(self, transformer):
        self.transformer = transformer

    @staticmethod
    def _model_input(latents: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(
            (latents.shape[0], 1, *latents.shape[2:]),
            device=latents.device,
            dtype=latents.dtype,
        )
        return torch.cat([latents, torch.zeros_like(latents), mask], dim=1)

    def predict(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text: TextTokens,
        cache: KVCache,
    ) -> torch.Tensor:
        return self.transformer(
            self._model_input(latents),
            text.as_embeddings(),
            timesteps,
            timestep_txt=timesteps[:, :1],
            kv_cache=cache,
        )

    def encode(self, latents: torch.Tensor, cache: KVCache) -> None:
        self.transformer(self._model_input(latents), mode="encode", kv_cache=cache)

    def recompute(
        self, latents: torch.Tensor, chunk_frames: List[int], cache: KVCache
    ) -> None:
        self.transformer(
            self._model_input(latents),
            mode="recompute",
            chunk_frames=chunk_frames,
            kv_cache=cache,
        )


class CfgParallelGuidance:
    """One guidance branch per rank of SGLang's classifier-free-guidance group, combined with one all-reduce.

    Both ranks read the same clean-history KV cache -- the history is encoded without any text, so they build
    bitwise the same cache from the same chunks and neither has to send it. Only the two predictions have to
    meet, which is the all-reduce of the already-scaled branches that SGLang's own denoising stage does. The
    scale is the per-frame AnnealCFG one, computed identically on both ranks from the same timesteps.
    """

    def __init__(self, rank: int):
        self.rank = rank

    def combine(self, denoiser, latents, timesteps, event, cache, scale):
        from sglang.multimodal_gen.runtime.distributed.communication_op import (
            cfg_model_parallel_all_reduce,
        )

        if self.rank == 0:
            partial = scale * denoiser.predict(
                latents, timesteps, event.positive, cache
            )
        else:
            partial = (1 - scale) * denoiser.predict(
                latents, timesteps, event.negative, cache
            )
        return cfg_model_parallel_all_reduce(partial.contiguous())


class MovieRollRollingDenoisingStage(DenoisingStage):
    """The rolling window over the whole video, in the place of the standard denoising loop.

    The standard loop denoises one fixed latent over one timestep per step. MovieRoll's window holds several
    chunks at different points of the ladder, pops a clean chunk every round into a per-request KV cache,
    rebuilds that cache at a threshold and at event switches, switches the text at event boundaries and draws
    its noise chunk by chunk in the order the checkpoint was evaluated with. The stage keeps everything else
    the standard stage does: the progress bar, the per-step profiler, the guidance combination, the layer-wise
    offload release and the device placement of the DiT.
    """

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        chain = _chain_of(server_args)
        device = get_local_torch_device()
        transformer = self.transformer
        # under --dit-cpu-offload (sglang's default for a video model) the DiT waits on the CPU between requests
        self._manage_device_placement(transformer, None, server_args)
        job: RollingJob = batch.extra[JOB_KEY]
        generator = (
            batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        )
        latent_shape = (
            server_args.pipeline_config.dit_config.arch_config.in_channels,
            batch.height // 16,
            batch.width // 16,
        )
        noise_fn, context_noise_fn = noise_functions(
            chain, generator, latent_shape, device, transformer.dtype
        )
        profile = chain.rolling_profile(batch.num_inference_steps)
        cache = transformer.new_cache(batch.height // 16, batch.width // 16)

        # a guidance group exists only for a guided chain (`MovieRollPipeline` refuses it otherwise at start-up)
        group = guidance_group()
        guidance = SerialGuidance() if group is None else CfgParallelGuidance(group[0])

        self.log_info(
            "mode=%s frames=%d size=%dx%d seed=%d events=%d chunks=%d guided=%s context_noise=%g "
            "tp=%d ulysses=%d ring=%d cfg=%d",
            chain.sampling_mode,
            batch.num_frames,
            batch.height,
            batch.width,
            batch.seed,
            len(job.events),
            len(job.chunk_sizes),
            profile.guided,
            chain.context_noise_sigma,
            transformer.parallel.tp_size,
            transformer.parallel.ulysses_size,
            transformer.parallel.ring_size,
            1 if group is None else group[1],
        )

        # the anchor leaves the VAE stage at the VAE's precision and enters the DiT's KV cache, so it is cast here
        anchor = (
            None
            if job.anchor is None
            else job.anchor.to(device=device, dtype=transformer.dtype)
        )
        total_chunks = len(job.chunk_sizes) - (1 if anchor is not None else 0)
        started = time.monotonic()
        with self.progress_bar(total=total_chunks) as progress_bar:

            def on_chunk(done: int, total: int) -> None:
                if progress_bar is not None:
                    progress_bar.update()
                logger.info(
                    "[%s] chunk %d / %d denoised",
                    type(self).__name__,
                    done,
                    total,
                    main_process_only=True,
                )
                refuse_if_it_cannot_finish_in_time(
                    done, total, time.monotonic() - started
                )

            def step_scope(step: int):
                return StageProfiler(
                    f"denoising_step_{step}",
                    logger=logger,
                    metrics=batch.metrics,
                    perf_dump_path_provided=batch.perf_dump_path is not None,
                    record_as_step=True,
                )

            with set_forward_context(
                current_timestep=0, attn_metadata=None, forward_batch=batch
            ):
                result = run_rolling(
                    denoiser=TransformerDenoiser(transformer),
                    cache=cache,
                    profile=profile,
                    events=job.events,
                    chunk_sizes=job.chunk_sizes,
                    event_split_indices=job.event_split_indices,
                    sigmas=self.scheduler.sigmas,
                    timesteps=batch.timesteps,
                    noise_fn=noise_fn,
                    context_noise_fn=context_noise_fn,
                    anchor=anchor,
                    guidance=guidance,
                    on_chunk=on_chunk,
                    step_scope=None if batch.is_warmup else step_scope,
                )
        cache.clear()
        batch.latents = result.latents
        batch.raw_latent_shape = tuple(result.latents.shape)
        self._post_denoising_loop(
            batch=batch,
            latents=result.latents,
            trajectory_latents=[],
            trajectory_timesteps=[],
            server_args=server_args,
            is_warmup=batch.is_warmup,
        )
        # and off again afterwards, which is what gives the streaming decode of a long video its room
        self._manage_device_placement(None, transformer, server_args)
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, V.is_tensor)
        result.add_check("sigmas", batch.sigmas, V.is_list)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.is_list)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result
