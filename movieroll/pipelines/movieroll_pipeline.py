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
# Modified from diffusers
# ==============================================================================
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base streaming pipeline: rolling-window text-to-video generation.

One round pushes a fresh noisy chunk into the rolling window, denoises the whole window `steps_per_window`
times (each chunk at its own point of the ladder), then pops the oldest chunk, now clean, as output. The
popped chunk is re-encoded into the transformer's KV cache, and that cache -- not the window -- carries
long-range consistency.

Three mechanisms limit drift: context noise (the clean chunk is lightly re-noised before it enters the KV
cache), KV recompute (past a cache threshold, or at an event boundary, the kept chunks are re-encoded into a
compact cache) and AnnealCFG (a per-frame guidance scale, since one window's frames sit at different noise
levels).

Subclasses override individual steps rather than `__call__`, so `_run_inner_denoising_loop` and
`_pop_and_output_clean_chunk` are part of the contract.
"""

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from diffusers.utils import BaseOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from movieroll.commons.constants import NUM_TRAIN_TIMESTEPS
from movieroll.commons.streaming_config import RollingStageType, StreamingConfig
from movieroll.models.text_encoders import TextEncoder
from movieroll.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat
from movieroll.models.text_encoders.glyph_text import extract_glyph_texts
from movieroll.models.transformers.modules.context_noise import apply_context_noise
from movieroll.models.transformers.modules.chunking import check_and_get_chunk_sizes
from movieroll.models.transformers.modules.rolling_queue import RollingQueue
from movieroll.models.transformers.modules.streaming_utils import safe_get_rank
from movieroll.models.transformers.movieroll_transformer import MovieRollDiffusionTransformer
from movieroll.schedulers.scheduling_flow_match_streaming import (
    AnnealCFG,
    FlowMatchStreamingScheduler,
    StreamingSchedulerWrapper,
)
from movieroll.commons.parallel_states import get_sequence_parallel_state, nccl_info
from movieroll.utils.communications import bcast_in_group, is_group_rank0
from movieroll.utils.torch_utils import PRECISION_TO_TYPE

logger = logging.getLogger(__name__)

@dataclass
class MovieRollStreamingPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class MovieRollStreamingPipeline(DiffusionPipeline):
    r"""
    Pipeline for streaming text-to-video generation using MovieRoll.

    Args:
        scheduler ([`FlowMatchStreamingScheduler`]):
            A scheduler whose `step` accepts a per-frame tensor `step_index`.
        streaming_config ([`StreamingConfig`]):
            The rolling-window geometry and the HieraKV / context-noise / AnnealCFG settings. Required.
        precision, vae_precision (`str`):
            `PRECISION_TO_TYPE` names of the transformer forward and of the VAE decode. Deliberately
            without defaults, so a caller cannot silently run at the wrong precision.
        prompt_format_font_path, prompt_format_color_path (`str`):
            The font and color tables of the byT5 glyph prompt format.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _exclude_from_cpu_offload = ["transformer"]

    def __init__(
        self,
        vae,
        text_encoder: TextEncoder,
        transformer: MovieRollDiffusionTransformer,
        scheduler: FlowMatchStreamingScheduler,
        text_len: int = 1000,
        byt5_kwarg=None,
        device=None,
        is_eval=False,
        streaming_config: StreamingConfig = None,
        *,
        precision: str,
        vae_precision: str,
        prompt_format_font_path: str,
        prompt_format_color_path: str,
    ):
        super().__init__()

        self.byt5_max_length = byt5_kwarg["byt5_max_length"]
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            byt5_model=byt5_kwarg["byt5_model"],
            byt5_tokenizer=byt5_kwarg["byt5_tokenizer"],
        )
        self.prompt_format = MultilingualPromptFormat(
            prompt_format_font_path,
            prompt_format_color_path,
        )
        self.target_dtype = PRECISION_TO_TYPE[precision]
        self.vae_dtype = PRECISION_TO_TYPE[vae_precision]

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.text_len = text_len

        if is_eval:
            if self.vae is not None:
                self.vae = self.vae.to(device=device)
            if self.text_encoder is not None:
                self.text_encoder = self.text_encoder.to(device=device)
            if self.transformer is not None:
                self.transformer = self.transformer.to(device=device)
            if self.byt5_model is not None:
                self.byt5_model = self.byt5_model.to(device=device)

        self.streaming_config = streaming_config

    def encode_prompt(
        self,
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        r"""Encode the prompt, and with guidance the negative prompt, into text encoder hidden states."""
        text_encoder = self.text_encoder

        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # Always the video template: text2tokens / encode default to the image one.
        text_inputs = text_encoder.text2tokens(prompt, data_type="video", max_length=self.text_len)
        prompt_outputs = text_encoder.encode(text_inputs, data_type="video", device=device)
        prompt_embeds = prompt_outputs.hidden_state

        attention_mask = prompt_outputs.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Keep both casts: they are this function's only device / dtype contract, and inside a bf16 autocast
        # the cast below is a live fp32 -> fp16 conversion. The returned embeddings are views into the text
        # encoder's pre-crop hidden state, not copies, and no downstream step writes them in place.
        prompt_embeds_dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        negative_prompt_embeds = None
        negative_attention_mask = None
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type="video", max_length=self.text_len)
            negative_prompt_outputs = text_encoder.encode(uncond_input, data_type="video")
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

    def decode_latents(self, latents):
        """Decode latents to pixels in [0, 1], on the CPU in fp32."""
        latents = 1 / self.vae.config.scaling_factor * latents
        self.vae.enable_tiling()
        image = self.vae.decode(latents, return_dict=False)[0]
        self.vae.disable_tiling()
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().float()
        return image

    def check_inputs(self, prompt, height, width):
        # The divisor is the VAE's own spatial reduction, read off the loaded model rather than written here:
        # `_make_noise` sizes the latents with `height // self.vae_scale_factor`, so a size this check lets
        # through but the floor division truncates would silently give a video of a different size than asked.
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor} (this VAE's spatial "
                f"reduction) but are {height} and {width}."
            )

        if prompt is None:
            raise ValueError("Provide a `prompt`.")
        if not isinstance(prompt, (str, list)):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

    @property
    def do_classifier_free_guidance(self):
        return self._use_cfg

    @staticmethod
    def get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return byt5_text_inputs.input_ids, byt5_text_inputs.attention_mask

    def _process_single_byt5_prompt(self, prompt_text, device):
        """Encode one prompt's glyph texts with byT5; all-zero embeddings and mask when it has none."""
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=device)
        byt5_mask = torch.zeros((1, self.byt5_max_length), device=device, dtype=torch.int64)

        glyph_texts = extract_glyph_texts(prompt_text)

        if len(glyph_texts) > 0:
            text_styles = [{'color': None, 'font-family': None} for _ in range(len(glyph_texts))]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)

            text_ids, text_mask = self.get_byt5_text_tokens(
                self.byt5_tokenizer, self.byt5_max_length, formatted_text
            )
            text_ids = text_ids.to(device=device)
            text_mask = text_mask.to(device=device)

            byt5_outputs = self.byt5_model(text_ids, attention_mask=text_mask.float())
            byt5_embeddings = byt5_outputs[0]
            byt5_mask = text_mask

        return byt5_embeddings, byt5_mask

    def _prepare_byt5_embeddings(self, prompts, device):
        """byT5 embeddings and masks for the batch, concatenated ``[negative, positive]`` under guidance."""
        if isinstance(prompts, str):
            prompt_list = [prompts]
        elif isinstance(prompts, list):
            prompt_list = prompts
        else:
            raise ValueError("prompts must be str or list of str")

        positive_embeddings = []
        positive_masks = []
        negative_embeddings = []
        negative_masks = []

        for prompt in prompt_list:
            pos_emb, pos_mask = self._process_single_byt5_prompt(prompt, device)
            positive_embeddings.append(pos_emb)
            positive_masks.append(pos_mask)

            if self.do_classifier_free_guidance:
                neg_emb, neg_mask = self._process_single_byt5_prompt("", device)
                negative_embeddings.append(neg_emb)
                negative_masks.append(neg_mask)

        byt5_positive = torch.cat(positive_embeddings, dim=0)
        byt5_positive_mask = torch.cat(positive_masks, dim=0)

        if self.do_classifier_free_guidance:
            byt5_negative = torch.cat(negative_embeddings, dim=0)
            byt5_negative_mask = torch.cat(negative_masks, dim=0)

            byt5_embeddings = torch.cat([byt5_negative, byt5_positive], dim=0)
            byt5_masks = torch.cat([byt5_negative_mask, byt5_positive_mask], dim=0)
        else:
            byt5_embeddings = byt5_positive
            byt5_masks = byt5_positive_mask

        return {
            "byt5_text_states": byt5_embeddings,
            "byt5_text_mask": byt5_masks
        }

    def _check_streaming_scheduler(self):
        """Reject a scheduler without the tensor-aware rolling ``step()``; raises, so ``python -O`` keeps it."""
        if not isinstance(self.scheduler, FlowMatchStreamingScheduler):
            raise TypeError(
                "scheduler must be a FlowMatchStreamingScheduler, got "
                f"{type(self.scheduler)!r}"
            )

    def prepare_timesteps(self, num_inference_steps, device, timesteps):
        """Install the denoising ladder on the scheduler; returns ``(timesteps, num_inference_steps)``.

        Without an explicit ladder the scheduler builds a uniform ladder of ``num_inference_steps`` and shifts
        it. An explicit ladder goes through ``set_custom_timesteps``, which applies the same shift and drops a
        trailing 0, so its count is read back from the scheduler.
        """
        if timesteps is None:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            return self.scheduler.timesteps, num_inference_steps
        self.scheduler.set_custom_timesteps(list(timesteps), device=device)
        return self.scheduler.timesteps, len(self.scheduler.timesteps)

    def prepare_latent_model_input(
        self,
        latents,
        gt_latents_concat,
        mask_concat,
        do_classifier_free_guidance: bool = True,
    ):
        """Build the 65-channel model input: [latents(32), zeros(32), zeros(1)], duplicated for CFG.

        The channel contract comes from the weights (`img_in.proj` takes 2 * in_channels + 1); the middle
        block and the mask are always zero at inference.
        """
        latents_concat = torch.cat([latents, gt_latents_concat, mask_concat], dim=1)
        latent_model_input = torch.cat([latents_concat] * 2) if do_classifier_free_guidance else latents_concat
        return latent_model_input

    def prepare_latent_concat(self, latents):
        """Return the all-zero conditioning blocks of the 65-channel input contract.

        Both blocks are zeros for i2v too, whose input frame reaches the model through the KV cache only.
        """
        gt_latents_concat = torch.zeros_like(latents)
        mask_concat = torch.zeros(
            (gt_latents_concat.shape[0], 1, *gt_latents_concat.shape[2:]),
            device=latents.device,
        )
        return gt_latents_concat, mask_concat

    def _add_context_noise(
        self,
        latents: torch.Tensor,
        context_noise_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        use_sp: bool,
    ) -> torch.Tensor:
        """Context-noise latents on their way into the KV cache; unchanged when ``context_noise_fn`` is None.

        Under sequence parallelism only the SP group's rank 0 draws from the sample's generator -- the other
        ranks must not advance it -- and broadcasts the result, so every rank caches the same history.
        """
        if context_noise_fn is None:
            return latents
        if not use_sp:
            return context_noise_fn(latents)
        sp_group = nccl_info.sp_group
        if is_group_rank0(sp_group):
            latents = context_noise_fn(latents)
        else:
            latents = torch.empty_like(latents)
        return bcast_in_group(latents, sp_group)

    def _encode_clean_chunk_kv(
        self,
        clean_latents: torch.Tensor,
        current_frame_index: int,
        target_dtype: torch.dtype = torch.bfloat16,
        autocast_enabled: bool = True,
    ) -> int:

        gt_latents_clean, mask_clean = self.prepare_latent_concat(clean_latents)
        # the KV cache pass needs no CFG
        lmi_clean = self.prepare_latent_model_input(
            clean_latents, gt_latents_clean, mask_clean,
            do_classifier_free_guidance=False
        )
        B, T = lmi_clean.shape[0], lmi_clean.shape[2]
        zero_t = torch.zeros(B, T, device=lmi_clean.device, dtype=lmi_clean.dtype)
        zero_t_txt = torch.zeros(B, 1, device=lmi_clean.device, dtype=lmi_clean.dtype)
        with torch.autocast(
            device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
        ):
            self.transformer(
                lmi_clean,
                zero_t,
                None,        # text_states = None
                None,        # encoder_attention_mask = None
                return_dict=False,
                extra_kwargs=None,
                return_kv=True,
                timestep_txt=zero_t_txt,
                noisy_chunk_sizes=[T],
                current_frame_index=current_frame_index,
            )
        return current_frame_index + T

    def _select_recompute_chunks(
        self,
        output_latents: List[torch.Tensor],
        event_split_indices: List[int],
    ) -> List[int]:
        """Chunk indices that take part in a KV recompute (deduplicated, ascending).

        Rule: the first chunk, the middle chunk of every event, and the kv_recompute_recent_k most recent ones.
        """
        recent_k = self.streaming_config.kv_recompute_recent_k
        total = len(output_latents)
        indices = set()

        indices.add(0)

        # event_split_indices are boundaries over ALL chunks; a recompute triggered mid-generation only sees
        # the chunks decoded so far, so boundaries at or past `total` must be filtered out, otherwise the
        # boundary list stops being monotonic and the midpoint index goes out of range.
        valid_splits = sorted(s for s in event_split_indices if 0 < s < total)
        boundaries = [0] + valid_splits + [total]
        for i in range(len(boundaries) - 1):
            seg_start, seg_end = boundaries[i], boundaries[i + 1]
            if seg_start < seg_end:
                indices.add((seg_start + seg_end - 1) // 2)

        indices.update(range(max(0, total - recent_k), total))

        return sorted(indices)

    def _kv_recompute(
        self,
        output_latents: List[torch.Tensor],
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        context_noise_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_sp: bool = False,
        trigger_reason: str = "threshold",
        *,
        selected_indices: List[int],
    ) -> int:
        """Rebuild a compact KV cache and return the new current_frame_index."""
        selected_indices = list(selected_indices)
        if not selected_indices:
            raise ValueError("KV recompute requires at least one selected chunk")
        if selected_indices != sorted(set(selected_indices)):
            raise ValueError(
                "selected_indices must be unique and strictly increasing, "
                f"got {selected_indices}"
            )
        if selected_indices[0] < 0 or selected_indices[-1] >= len(output_latents):
            raise IndexError(
                "selected_indices contains an index outside output_latents: "
                f"indices={selected_indices}, num_chunks={len(output_latents)}"
            )
        selected = [output_latents[i] for i in selected_indices]
        recompute_chunk_sizes = [lat.shape[2] for lat in selected]

        new_latents = torch.cat(selected, dim=2)

        # as in a regular encode, context-noise whatever goes into the KV cache
        new_latents = self._add_context_noise(new_latents, context_noise_fn, use_sp)

        gt_latents, mask = self.prepare_latent_concat(new_latents)
        # the KV cache pass needs no CFG
        lmi = self.prepare_latent_model_input(
            new_latents, gt_latents, mask,
            do_classifier_free_guidance=False,
        )

        # The stale cache is cleared inside the transformer's kv_recompute forward: it drops every layer's
        # cached KV, then rebuilds the block groups from this same hidden_states.
        with torch.autocast(
            device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
        ):
            # go through the transformer's __call__/forward(kv_recompute=True) so the root FSDP hook fires
            new_current_frame_index = self.transformer(
                hidden_states=lmi,
                timestep=None,
                text_states=None,
                encoder_attention_mask=None,
                kv_recompute=True,
                recompute_chunk_sizes=recompute_chunk_sizes,
            )

        if self.transformer.kv_cache_manager.verbose and safe_get_rank() == 0:
            print(
                f"[kv_recompute] trigger={trigger_reason}; selected chunks {selected_indices} "
                f"(sizes={recompute_chunk_sizes}) from {len(output_latents)} decoded; "
                f"new current_frame_index={new_current_frame_index}"
            )
        return new_current_frame_index

    def expand_step_index(self, step_index: torch.Tensor, rolling_stage: RollingStageType) -> torch.Tensor:
        """Expand step_index from one entry per window chunk to one entry per latent frame.

        Expands by ``rolling_queue.chunk_lengths``, so a first chunk of a different size is handled too.
        """
        num_chunks_in_window = self.rolling_queue.get_num_chunks_in_window()
        if rolling_stage == RollingStageType.DECODE:
            # the window is full, so there is nothing to truncate
            assert len(step_index) == num_chunks_in_window, "step_index length should be equal to num_chunks_in_window in decode stage"
        else:
            step_index = step_index[:num_chunks_in_window]
        chunk_lengths = self.rolling_queue.chunk_lengths
        step_index_expanded = step_index.repeat_interleave(
            torch.tensor(chunk_lengths, device=step_index.device)
        )
        return step_index_expanded

    def _encode_single_event_prompt(
        self,
        prompt,
        *,
        device,
        negative_prompt,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], dict]:
        """Encode one event prompt into ``(prompt_embeds, prompt_mask, prompt_embeds_2, extra_kwargs)``.

        The embeddings are CFG-concatenated ``[neg, pos]`` under guidance. ``prompt_embeds_2`` is always None
        (MovieRoll has a single text encoder); the slot stays because callers unpack this tuple positionally.
        """
        pe, npe, pm, npm = self.encode_prompt(
            prompt,
            device,
            self.do_classifier_free_guidance,
            negative_prompt,
        )

        pe2 = None

        extra_kwargs = self._prepare_byt5_embeddings(prompt, device)

        if self.do_classifier_free_guidance:
            pe = torch.cat([npe, pe])
            if pm is not None:
                pm = torch.cat([npm, pm])

        return pe, pm, pe2, extra_kwargs

    @staticmethod
    def _compute_event_split_indices(
        num_chunks: int,
        event_prompts: Optional[List[str]],
        rank: int = 0,
    ) -> Tuple[List[int], List[int]]:
        """Return ``(split_indices, kept_event_indices)`` of an even split of the chunks across the events.

        ``split_indices[k]`` is the chunk index at which segment k+1 starts; ``kept_event_indices[k]`` indexes
        ``event_prompts`` for the prompt driving segment k. Events whose segment comes out empty are dropped
        from ``kept_event_indices``, so callers can subset ``all_event_conds`` in lockstep with
        ``split_indices``. The single-event / no-event case is ``([], [0])``.
        """
        if event_prompts is None or len(event_prompts) < 2:
            return [], [0]

        num_events = len(event_prompts)
        split_indices = [(num_chunks * ei) // num_events for ei in range(1, num_events)]
        if rank == 0:
            print(
                f"[event split] even mode: {num_events} events, "
                f"split at chunks {split_indices}/{num_chunks}"
            )

        # Drop events whose segment is empty (more events than chunks); otherwise the prompt-switch loop would
        # silently never use the corresponding prompt.
        boundaries = [0] + split_indices + [num_chunks]
        kept_event_indices: List[int] = []
        for ei in range(len(boundaries) - 1):
            if boundaries[ei] < boundaries[ei + 1]:
                kept_event_indices.append(ei)
            elif rank == 0:
                print(
                    f"[event split] WARNING: event {ei} has 0 chunks "
                    f"(boundary {boundaries[ei]} == {boundaries[ei + 1]}); "
                    "dropping its prompt."
                )
        deduped_splits = [split_indices[ev - 1] for ev in kept_event_indices[1:]]
        return deduped_splits, kept_event_indices

    def _pop_and_output_clean_chunk(
        self,
        *,
        is_global_last_chunk: bool,
        output_latents: List[torch.Tensor],
        context_noise_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        current_frame_index: int,
        use_sp: bool,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        outer_progress_bar: tqdm,
        event_split_indices: Optional[List[int]] = None,
    ) -> int:
        """Pop the oldest clean chunk, save it as output, and encode it into the KV cache.

        Context noise is added to every real output except the global last one, whose KV cache is never
        consumed. Callers must only invoke this with a real (non-padding) chunk at the head of the queue.
        """
        clean_latents = self.rolling_queue.pop_clean_chunk()
        output_latents.append(clean_latents.clone())

        # Event switch: the last chunk of an event has just been decoded (len(output_latents) now equals the
        # chunk index the next event starts at), so force a recompute to compress that event's cache.
        event_switch = (
            event_split_indices is not None
            and len(output_latents) in event_split_indices
        )

        # No later denoising consumes the globally last chunk's KV, so skip the encode / recompute entirely.
        if is_global_last_chunk:
            outer_progress_bar.update(1)
            return current_frame_index

        # KV recompute or a regular encode, never both: when the cache hits the threshold or an event switch
        # happens, rebuild it from output_latents (which already contains the chunk just popped); otherwise
        # encode the current chunk.
        do_recompute = getattr(
            self.streaming_config, "kv_recompute_enabled", False
        ) and (
            event_switch
            or self.transformer.kv_cache_manager.should_recompute(
                self.streaming_config.kv_recompute_threshold
            )
        )

        # Only compress when the recompute can actually drop chunks: if the selection covers all of them it
        # degenerates into re-encoding every chunk for no memory saved, so take the regular encode instead.
        # That happens for very short events, whose cache holds fewer chunks than the selection rule keeps.
        if do_recompute:
            selected = self._select_recompute_chunks(
                output_latents, event_split_indices or []
            )
            if len(selected) >= len(output_latents):
                do_recompute = False

        if do_recompute:
            current_frame_index = self._kv_recompute(
                output_latents=output_latents,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                context_noise_fn=context_noise_fn,
                use_sp=use_sp,
                trigger_reason="event_switch" if event_switch else "threshold",
                selected_indices=selected,
            )
        else:
            kv_latents = self._add_context_noise(clean_latents, context_noise_fn, use_sp)
            current_frame_index = self._encode_clean_chunk_kv(
                kv_latents,
                current_frame_index=current_frame_index,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
            )

        outer_progress_bar.update(1)
        return current_frame_index

    def _run_inner_denoising_loop(
        self,
        *,
        local_step_base: int,
        rolling_stage: RollingStageType,
        event_cond: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], dict],
        current_frame_index: int,
        timesteps: torch.Tensor,
        anneal_cfg: AnnealCFG,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        inner_progress_bar: tqdm,
        steps_per_window: int,
    ) -> None:
        """Run ``steps_per_window`` denoising steps on the current queue contents."""
        prompt_embeds, prompt_mask, prompt_embeds_2, extra_kwargs = event_cond

        latents = self.rolling_queue.get_current_latents()
        gt_latents_concat, mask_concat = self.prepare_latent_concat(latents)

        for step in range(steps_per_window):
            local_step = local_step_base + step
            latents = self.rolling_queue.get_current_latents()

            step_index = self.scheduler_wrapper.get_step_index(local_step)
            step_index_expanded = self.expand_step_index(step_index, rolling_stage)
            latent_model_input = self.prepare_latent_model_input(
                latents, gt_latents_concat, mask_concat,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            t_expand = timesteps[step_index_expanded].unsqueeze(0).repeat(
                latent_model_input.shape[0], 1
            )

            with torch.autocast(
                device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
            ):
                noise_pred = self.transformer(
                    latent_model_input,
                    t_expand,
                    prompt_embeds,
                    prompt_mask,
                    return_dict=False,
                    extra_kwargs=extra_kwargs,
                    return_kv=False,
                    timestep_txt=t_expand[:, :1],
                    noisy_chunk_sizes=[latents.shape[2]],
                    current_frame_index=current_frame_index,
                )[0]

            if self.do_classifier_free_guidance:
                cfg = self._compute_cfg(anneal_cfg, t_expand)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler_wrapper.step(
                noise_pred, step_index_expanded, latents
            )
            self.rolling_queue.update_with_new_latents(latents)
            inner_progress_bar.update(1)

    def _compute_cfg(
        self,
        anneal_cfg: AnnealCFG,
        t_expand: torch.Tensor,
    ) -> torch.Tensor:
        """Per-frame CFG coefficient for this step, interpolated from each frame's own timestep."""
        cfg = anneal_cfg.get_cfg(t_expand[0])
        return cfg.view(1, 1, -1, 1, 1)

    def _generate_strategy_prompt_switch(
        self,
        *,
        chunk_sizes: List[int],
        chunk_to_event: List[int],
        all_event_conds: List[Tuple],
        window_size: int,
        num_chunks: int,
        tail_chunk_size: int,
        current_frame_index: int,
        sample_noisy_chunk: Callable[[int], torch.Tensor],
        denoise_kwargs: dict,
        pop_kwargs: dict,
        rank: int,
    ) -> int:
        """One continuous rolling loop; swap the prompt at an event boundary.

        There is no drain / re-prefill between events, so a transition spans up to ``window_size`` output
        chunks denoised under mixed prompts.
        """
        if rank == 0 and len(all_event_conds) > 1:
            boundaries = [
                i for i in range(1, num_chunks) if chunk_to_event[i] != chunk_to_event[i - 1]
            ]
            print(
                f"[prompt switch] {len(all_event_conds)} events, "
                f"prompt switches at chunks {boundaries}"
            )

        steps_per_window = denoise_kwargs["steps_per_window"]
        outer_pbar = pop_kwargs["outer_progress_bar"]
        total_rounds = num_chunks + window_size
        num_decoded = 0
        prev_event_idx = -1

        for round_idx in range(total_rounds):
            is_prefill = round_idx < window_size
            rolling_stage = (
                RollingStageType.PREFILL if is_prefill else RollingStageType.DECODE
            )

            if not is_prefill:
                is_global_last = num_decoded == num_chunks - 1
                current_frame_index = self._pop_and_output_clean_chunk(
                    is_global_last_chunk=is_global_last,
                    current_frame_index=current_frame_index,
                    **pop_kwargs,
                )
                num_decoded += 1
                if num_decoded >= num_chunks:
                    self.rolling_queue.clear()
                    break

            cleanest_chunk_idx = min(num_decoded, num_chunks - 1)
            cur_event_idx = chunk_to_event[cleanest_chunk_idx]
            # log actual transitions only, not the no-op first round
            if (
                rank == 0
                and len(all_event_conds) > 1
                and cur_event_idx != prev_event_idx
            ):
                print(
                    f"[round {round_idx}] prompt switch -> "
                    f"event_{cur_event_idx + 1}/{len(all_event_conds)} "
                    f"(cleanest chunk #{cleanest_chunk_idx})"
                )
            prev_event_idx = cur_event_idx
            event_cond = all_event_conds[cur_event_idx]

            chunk_temporal_size = (
                chunk_sizes[round_idx] if round_idx < num_chunks else tail_chunk_size
            )
            self.rolling_queue.add_new_noisy_chunk(sample_noisy_chunk(chunk_temporal_size))

            num_in_window = self.rolling_queue.get_num_chunks_in_window()
            inner_pbar = tqdm(
                total=steps_per_window,
                desc=f"Round {round_idx} [{rolling_stage.name}], w={num_in_window}",
                disable=rank != 0,
            )
            outer_pbar.set_description(f"Outer Sampling with Stage {rolling_stage.name}")
            try:
                self._run_inner_denoising_loop(
                    local_step_base=round_idx * steps_per_window,
                    rolling_stage=rolling_stage,
                    event_cond=event_cond,
                    current_frame_index=current_frame_index,
                    inner_progress_bar=inner_pbar,
                    **denoise_kwargs,
                )
            finally:
                inner_pbar.close()

        return current_frame_index

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        use_cfg: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # NOT a list: it goes straight to `torch.randn(..., generator=...)`, which rejects one.
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pt",
        event_prompts: Optional[List[str]] = None,
        cond_latents: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        r"""
        Roll a denoising window over the chunks; returns one video in `MovieRollStreamingPipelineOutput`.

        Args:
            height / width:
                Must be divisible by the VAE's spatial reduction, since the latent size is that division.
            video_length:
                Pixel-space frame count, converted to latent frames and truncated down to whole chunks.
            num_inference_steps:
                Denoising steps per chunk; must be divisible by the window size, and is rounded down if not.
            use_cfg:
                Classifier-free guidance on or off; when on, the per-frame scale comes from AnnealCFG, the schedule
                between `cfg_min` / `cfg_max`.
            generator:
                Draws the per-chunk noise and the context noise. Under sequence parallelism only rank 0 of the
                SP group draws and broadcasts, so all ranks share one noise tensor.
            event_prompts:
                Multi-event prompts; the chunks are split evenly across the events and the prompt is swapped at
                each boundary. With these set, `event_prompts[0]` takes precedence over `prompt`.
            cond_latents:
                Image-to-video: the VAE latent of the input image, ``[1, C, 1, h, w]``. It is output chunk 0,
                written into the KV cache once at timestep 0 and never denoised; ``None`` is text-to-video.
            timesteps:
                An explicit denoising ladder instead of one derived from `num_inference_steps`.

        Several videos per prompt come from calling this repeatedly with ONE shared generator, so each pass
        consumes the next draws.
        """
        target_dtype = self.target_dtype

        self.check_inputs(prompt, height, width)
        # `num_videos_per_prompt` is refused rather than ignored, because **kwargs would otherwise swallow it
        # and hand back one video while the caller believed it had asked for several. The keyword's PRESENCE
        # is what raises, so callers pass nothing rather than passing 1.
        if "num_videos_per_prompt" in kwargs:
            raise ValueError(
                f"num_videos_per_prompt is not a parameter of this pipeline, got "
                f"{kwargs['num_videos_per_prompt']!r}: call it once per video with the same generator "
                "instead (see `--num-videos-per-prompt`)"
            )

        self._use_cfg = bool(use_cfg)

        autocast_enabled = target_dtype != torch.float32
        vae_dtype = self.vae_dtype
        vae_autocast_enabled = vae_dtype != torch.float32

        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        device = self.transformer.device
        rank = safe_get_rank()

        # Encode all event prompts -> all_event_conds
        has_event_prompts = event_prompts is not None and len(event_prompts) >= 2
        if (
            has_event_prompts
            and isinstance(prompt, str)
            and isinstance(event_prompts[0], str)
            and prompt != event_prompts[0]
            and rank == 0
        ):
            logger.warning(
                "`prompt` differs from `event_prompts[0]`; using "
                "`event_prompts[0]` for event 0 to match the multi-event "
                "configuration."
            )

        all_event_conds: List[Tuple] = [
            self._encode_single_event_prompt(
                event_prompt,
                device=device,
                negative_prompt=negative_prompt,
            )
            for event_prompt in (event_prompts if has_event_prompts else [prompt])
        ]

        if video_length is None:
            raise ValueError("streaming rolling inference requires an explicit video_length")
        chunk_size = self.streaming_config.chunk_size
        window_size = self.streaming_config.window_size
        latent_length = (video_length - 1) // 4 + 1
        # the latent frame count must be a whole number of chunks plus one
        latent_length = (latent_length - 1) // chunk_size * chunk_size + 1
        if rank == 0:
            print(f"reset latent length to a whole number of chunks: {latent_length}")

        if num_inference_steps % window_size != 0 and window_size > 1:
            adjusted_steps = (num_inference_steps // window_size) * window_size
            if adjusted_steps == 0:
                raise ValueError(
                    f"Cannot adjust num_inference_steps ({num_inference_steps}) to be divisible by "
                    f"window_size ({window_size}). Please adjust window_size or num_inference_steps."
                )
            logger.warning(
                f"num_inference_steps ({num_inference_steps}) is not divisible by window_size "
                f"({window_size}). Adjusting to {adjusted_steps}."
            )
            num_inference_steps = adjusted_steps

        steps_per_window = num_inference_steps // window_size
        if steps_per_window == 0:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) must be divisible by "
                f"window_size ({window_size})."
            )
        self._check_streaming_scheduler()
        self.scheduler_wrapper = StreamingSchedulerWrapper(
            self.streaming_config, self.scheduler, window_size, num_inference_steps,
        )
        self.rolling_queue = RollingQueue()

        tt = latent_length // self.transformer.patch_size[0]
        # t2v: [1 + chunk_size, chunk_size, ...]; i2v: [1, chunk_size, ...], the 1-frame chunk is the input image
        chunk_mode = "i2v" if cond_latents is not None else "default"
        chunk_sizes_in_patches = check_and_get_chunk_sizes(tt, chunk_size=chunk_size, mode=chunk_mode)
        chunk_sizes = [cs * self.transformer.patch_size[0] for cs in chunk_sizes_in_patches]

        timesteps, resolved_steps = self.prepare_timesteps(
            num_inference_steps, device, timesteps,
        )
        # `steps_per_window` and the scheduler wrapper above were both built from `num_inference_steps`, and
        # `get_step_index` indexes `timesteps` with values up to num_inference_steps - 1, so a ladder of any
        # other length would run the window off the end of it, or silently over a subset, instead of failing.
        if resolved_steps != num_inference_steps or len(timesteps) != num_inference_steps:
            raise ValueError(
                f"the denoising ladder has {len(timesteps)} steps (prepare_timesteps resolved "
                f"{resolved_steps}) but the rolling window was built for {num_inference_steps}"
            )
        if rank == 0:
            print(f"num_inference_steps: {num_inference_steps}")
            print(f"steps_per_window: {steps_per_window}")

        current_frame_index = 0
        output_latents: List[torch.Tensor] = []
        num_chunks = len(chunk_sizes)
        if rank == 0:
            print(f"chunk_sizes: {chunk_sizes}")

        # Per-event chunk splits, context noise, AnnealCFG
        event_split_indices, kept_event_indices = self._compute_event_split_indices(
            num_chunks, event_prompts, rank=rank,
        )
        # Drop conds for events whose segment ended up empty, so the surviving conds stay 1:1 with the
        # segments defined by event_split_indices.
        if len(kept_event_indices) != len(all_event_conds):
            all_event_conds = [all_event_conds[i] for i in kept_event_indices]
        # chunk_to_event[c] = the event segment output chunk c belongs to
        chunk_to_event = [0] * num_chunks
        for ev_idx, boundary_chunk in enumerate(event_split_indices):
            for ci in range(boundary_chunk, num_chunks):
                chunk_to_event[ci] = ev_idx + 1

        # i2v: the input frame is output chunk 0. It is written into the KV cache once, at timestep 0 and
        # without context noise, and is never denoised, so it leaves chunk_sizes. The event splits above are
        # in full-output-layout coordinates, which is what the pop and the recompute selection compare with.
        if cond_latents is not None:
            if cond_latents.shape[2] != chunk_sizes[0] or num_chunks < 2:
                raise ValueError(
                    "i2v needs a 1-frame cond_latents and at least one chunk to generate after it, got "
                    f"cond_latents frames={cond_latents.shape[2]}, chunk_sizes={chunk_sizes[:3]}... (n={num_chunks})"
                )
            output_latents.append(cond_latents)
            current_frame_index = self._encode_clean_chunk_kv(
                cond_latents,
                current_frame_index=current_frame_index,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
            )
            chunk_sizes = chunk_sizes[1:]
            chunk_to_event = chunk_to_event[1:]
            num_chunks = len(chunk_sizes)
            if rank == 0:
                print(f"[i2v] input frame pre-encoded into the KV cache; {num_chunks} chunks left to generate")

        context_noise_fn = None
        if self.streaming_config.context_noise_enabled:
            sigma_max = self.streaming_config.inference_context_noise

            # The history noise comes from the sample's own generator, the one its chunk noise is drawn from,
            # so a sample does not depend on the rank it runs on or on the samples before it.
            def context_noise_fn(latents: torch.Tensor) -> torch.Tensor:
                return apply_context_noise(
                    latents, sigma_max=sigma_max, force_equal_sigma=True, generator=generator,
                )
            if rank == 0:
                print("[context-noise] apply_context_noise enabled "
                      f"(sigma_max={sigma_max}, force_equal_sigma=True)")

        sc = self.streaming_config
        anneal_cfg = AnnealCFG(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            cfg_min=sc.cfg_min,
            cfg_max=sc.cfg_max,
        )
        if rank == 0:
            print(f"AnnealCFG enabled (cfg_min={sc.cfg_min}, cfg_max={sc.cfg_max})")

        num_channels_latents = self.transformer.config.in_channels
        latent_h = int(height) // self.vae_scale_factor
        latent_w = int(width) // self.vae_scale_factor
        use_sp = get_sequence_parallel_state()
        tail_chunk_size = chunk_sizes[-1] if chunk_sizes else chunk_size

        outer_progress_bar = tqdm(total=num_chunks, disable=rank != 0)

        def _make_noise(temporal_size: int) -> torch.Tensor:
            shape = (
                batch_size, num_channels_latents,
                temporal_size, latent_h, latent_w,
            )
            if use_sp:
                sp_group = nccl_info.sp_group
                if dist.get_rank(sp_group) == 0:
                    noise = torch.randn(
                        shape,
                        dtype=target_dtype,
                        device=device,
                        generator=generator,
                    )
                else:
                    noise = torch.empty(
                        shape, dtype=target_dtype, device=device
                    )
                torch.distributed.broadcast(
                    noise, group_src=0, group=sp_group,
                )
            else:
                noise = torch.randn(
                    shape,
                    dtype=target_dtype,
                    device=device,
                    generator=generator,
                )
            return noise

        denoise_kwargs = dict(
            timesteps=timesteps,
            anneal_cfg=anneal_cfg,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            steps_per_window=steps_per_window,
        )
        pop_kwargs = dict(
            output_latents=output_latents,
            context_noise_fn=context_noise_fn,
            use_sp=use_sp,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            outer_progress_bar=outer_progress_bar,
            event_split_indices=event_split_indices,
        )

        try:
            current_frame_index = self._generate_strategy_prompt_switch(
                chunk_sizes=chunk_sizes,
                chunk_to_event=chunk_to_event,
                all_event_conds=all_event_conds,
                window_size=window_size,
                num_chunks=num_chunks,
                tail_chunk_size=tail_chunk_size,
                current_frame_index=current_frame_index,
                sample_noisy_chunk=_make_noise,
                denoise_kwargs=denoise_kwargs,
                pop_kwargs=pop_kwargs,
                rank=rank,
            )
        finally:
            outer_progress_bar.close()

        # The DiT KV cache and the VAE T-cache are independent; free the former before decoding.
        self.transformer.kv_cache_manager.clear_kv_cache()
        self.rolling_queue.clear()

        # VAE decode
        if len(output_latents[0].shape) != 5:
            raise ValueError(
                f"Only support latents with shape (b, c, f, h, w), but got {output_latents[0].shape}."
            )
        latents = torch.cat(output_latents, dim=2)

        if output_type != "latent":
            if (
                hasattr(self.vae.config, "shift_factor")
                and self.vae.config.shift_factor
            ):
                latents = (
                    latents / self.vae.config.scaling_factor
                    + self.vae.config.shift_factor
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(
                device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
            ):
                use_t_cache = (
                    callable(getattr(self.vae, "clear_cache", None))
                    and hasattr(self.vae, "_streaming_decode_state")
                )
                if use_t_cache:
                    decoded_chunks = []
                    max_chunk_size = kwargs.get(
                        "vae_stream_decode_max_chunk_size", 4
                    )
                    if max_chunk_size is not None:
                        if (
                            isinstance(max_chunk_size, bool)
                            or not isinstance(max_chunk_size, int)
                        ):
                            raise TypeError(
                                "vae_stream_decode_max_chunk_size must be "
                                "None or an integer"
                            )
                        if max_chunk_size <= 0:
                            raise ValueError(
                                "vae_stream_decode_max_chunk_size must be "
                                "None or a positive integer"
                            )
                    latent_chunks = (
                        (latents,)
                        if max_chunk_size is None
                        else latents.split(int(max_chunk_size), dim=2)
                    )
                    self.vae.clear_cache()
                    try:
                        for latent_chunk in latent_chunks:
                            pixel_chunk = self.vae.decode(
                                latent_chunk,
                                return_dict=False,
                                streaming_mode=True,
                            )[0]
                            decoded_chunks.append(pixel_chunk.cpu())
                            del pixel_chunk
                    finally:
                        self.vae.clear_cache()
                    image = torch.cat(decoded_chunks, dim=2)
                    del decoded_chunks
                else:
                    # The non-streaming VAE keeps the spatial-tiling path.
                    if latent_length <= 31:
                        sample_size = 256
                    elif latent_length <= 61:
                        sample_size = 128
                    else:
                        sample_size = 64
                    # Restore the sizes that were found, not the config values: the VAE instance is shared
                    # with the trainer, and `decode_latents` only toggles tiling and reads whatever sizes are
                    # left on it, so residue here would make a later decode depend on this call's geometry.
                    prev_tile_sizes = (self.vae.tile_sample_min_size, self.vae.tile_latent_min_size)
                    try:
                        self.vae.tile_sample_min_size = sample_size
                        self.vae.tile_latent_min_size = sample_size // 16
                        self.vae.enable_tiling()
                        image = self.vae.decode(latents, return_dict=False)[0]
                    finally:
                        self.vae.disable_tiling()
                        self.vae.tile_sample_min_size, self.vae.tile_latent_min_size = prev_tile_sizes
        else:
            image = latents

        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().float()

        return MovieRollStreamingPipelineOutput(videos=image)
