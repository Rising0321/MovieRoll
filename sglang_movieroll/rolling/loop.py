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

"""The rolling-window sampling loop, for inference only.

One round pushes a fresh noisy chunk into the window, denoises the whole window (each chunk at its own point of
the ladder), then pops the oldest chunk, now clean, as output and encodes it into the KV cache; past a cache
threshold, or at an event boundary, the cache is rebuilt from a selection of the output chunks (KV recompute).

The loop knows nothing about the model or the framework: it drives a `Denoiser` (three calls: predict a
window's velocity, encode a clean chunk into the cache, rebuild the cache) and a `GuidanceCombiner` (one call:
the guided velocity of a window, serially or over a classifier-free-guidance group). The two released chains
share the loop and differ only in what `RollingProfile` spells out: Base runs several steps per round with
guidance, Distilled one step without; Distilled forces a recompute at an event switch even when nothing can be
dropped, and noises the image-to-video anchor like any other history.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, ContextManager, List, Optional, Protocol, Sequence

import torch

from sglang_movieroll.rolling.geometry import (
    event_index_per_chunk,
    select_recompute_chunks,
)
from sglang_movieroll.rolling.kv_cache import KVCache
from sglang_movieroll.rolling.profiles import NUM_TRAIN_TIMESTEPS, RollingProfile
from sglang_movieroll.rolling.queue import RollingQueue
from sglang_movieroll.rolling.schedule import AnnealCFG, WindowSchedule, flow_match_step
from sglang_movieroll.rolling.tokens import EventCondition, TextTokens


class Denoiser(Protocol):
    """What the loop needs from the model."""

    def predict(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text: TextTokens,
        cache: KVCache,
    ) -> torch.Tensor:
        """Velocity of the window ``latents`` ``[1, C, T, h, w]`` at per-frame ``timesteps`` ``[1, T]``."""

    def encode(self, latents: torch.Tensor, cache: KVCache) -> None:
        """Append a clean chunk to the cache."""

    def recompute(
        self, latents: torch.Tensor, chunk_frames: List[int], cache: KVCache
    ) -> None:
        """Rebuild the cache from the chunks concatenated in ``latents``."""


class GuidanceCombiner(Protocol):
    """How the two guidance branches of a window are produced and combined (serially, or one per rank)."""

    def combine(
        self,
        denoiser: Denoiser,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        event: EventCondition,
        cache: KVCache,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """``negative + scale * (positive - negative)``, however the two branches are obtained."""


class SerialGuidance:
    """Both branches on this rank: the single-GPU path and every topology without guidance parallelism."""

    def combine(self, denoiser, latents, timesteps, event, cache, scale):
        positive = denoiser.predict(latents, timesteps, event.positive, cache)
        negative = denoiser.predict(latents, timesteps, event.negative, cache)
        return negative + scale * (positive - negative)


@dataclass
class RollingResult:
    latents: torch.Tensor  # all output chunks concatenated, ``[1, C, T, h, w]``
    chunk_frames: List[int]
    recomputes: List[int] = field(
        default_factory=list
    )  # number of output chunks at each recompute


def _expand_step_index(step_index: torch.Tensor, queue: RollingQueue) -> torch.Tensor:
    """One step index per window chunk -> one per latent frame."""
    in_window = queue.get_num_chunks_in_window()
    step_index = step_index[:in_window]
    return step_index.repeat_interleave(
        torch.tensor(queue.chunk_lengths, device=step_index.device)
    )


def run_rolling(
    *,
    denoiser: Denoiser,
    cache: KVCache,
    profile: RollingProfile,
    events: Sequence[EventCondition],
    chunk_sizes: List[int],
    event_split_indices: Sequence[int],
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    noise_fn: Callable[[int], torch.Tensor],
    context_noise_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    anchor: Optional[torch.Tensor] = None,
    guidance: Optional[GuidanceCombiner] = None,
    on_chunk: Optional[Callable[[int, int], None]] = None,
    step_scope: Optional[Callable[[int], ContextManager]] = None,
) -> RollingResult:
    """Generate the output chunks.

    ``chunk_sizes`` is the full output layout (with the single-frame anchor chunk first when ``anchor`` is
    given), ``event_split_indices`` are boundaries in output-chunk coordinates, ``sigmas`` / ``timesteps`` are
    the ladder of the chain (the framework's scheduler produced them), ``noise_fn(frames)`` draws a fresh noisy
    chunk, ``on_chunk(done, total)`` reports every chunk that left the window and ``step_scope(step)`` wraps one
    denoising step (the stage puts its profiler there).
    """
    window_size = profile.window_size
    if len(timesteps) != profile.num_inference_steps:
        raise ValueError(
            f"the ladder has {len(timesteps)} steps but the profile runs {profile.num_inference_steps}"
        )
    schedule = WindowSchedule(window_size, profile.num_inference_steps)
    anneal = (
        AnnealCFG(NUM_TRAIN_TIMESTEPS, profile.cfg_min, profile.cfg_max)
        if profile.guided
        else None
    )
    combiner = guidance or SerialGuidance()
    queue = RollingQueue()
    finalized: List[torch.Tensor] = []
    result = RollingResult(latents=None, chunk_frames=[])

    chunk_to_event = event_index_per_chunk(len(chunk_sizes), event_split_indices)
    if anchor is not None:
        if chunk_sizes[0] != 1 or anchor.shape[2] != 1:
            raise ValueError(
                "the image-to-video anchor is one latent frame and the first chunk of the layout"
            )
        finalized.append(anchor)
        denoiser.encode(
            (
                context_noise_fn(anchor)
                if (profile.noise_anchor and context_noise_fn)
                else anchor
            ),
            cache,
        )
        chunk_sizes = chunk_sizes[1:]
        chunk_to_event = chunk_to_event[1:]
    num_chunks = len(chunk_sizes)
    if num_chunks <= 0:
        raise ValueError("nothing to generate")

    num_decoded = 0
    global_step = 0
    for round_index in range(num_chunks + window_size):
        if round_index >= window_size:
            is_last = num_decoded == num_chunks - 1
            clean = queue.pop_clean_chunk()
            finalized.append(clean)
            num_decoded += 1
            if on_chunk is not None:
                on_chunk(num_decoded, num_chunks)
            if is_last:
                break
            _refresh_cache(
                denoiser,
                cache,
                profile,
                finalized,
                clean,
                event_split_indices,
                context_noise_fn,
                result,
            )

        frames = (
            chunk_sizes[round_index] if round_index < num_chunks else chunk_sizes[-1]
        )
        queue.add_new_noisy_chunk(noise_fn(frames))

        event = events[chunk_to_event[min(num_decoded, num_chunks - 1)]]
        for _ in range(profile.steps_per_window):
            with step_scope(global_step) if step_scope is not None else nullcontext():
                latents = queue.get_current_latents()
                step_index = _expand_step_index(
                    schedule.get_step_index(global_step), queue
                )
                frame_timesteps = timesteps[step_index].unsqueeze(0)
                if anneal is None:
                    velocity = denoiser.predict(
                        latents, frame_timesteps, event.positive, cache
                    )
                else:
                    scale = anneal.get_cfg(frame_timesteps[0]).view(1, 1, -1, 1, 1)
                    velocity = combiner.combine(
                        denoiser, latents, frame_timesteps, event, cache, scale
                    )
                queue.update_with_new_latents(
                    flow_match_step(velocity, step_index, latents, sigmas)
                )
            global_step += 1

    result.latents = torch.cat(finalized, dim=2)
    result.chunk_frames = [chunk.shape[2] for chunk in finalized]
    return result


def _refresh_cache(
    denoiser,
    cache,
    profile,
    finalized,
    clean,
    event_split_indices,
    context_noise_fn,
    result,
) -> None:
    """Put the chunk that was just popped into the cache: a plain encode, or a rebuild at a threshold or switch."""
    event_switch = bool(event_split_indices) and len(finalized) in event_split_indices
    do_recompute = profile.kv_recompute_enabled and (
        event_switch or cache.should_recompute(profile.kv_recompute_threshold)
    )
    selected: List[int] = []
    if do_recompute:
        selected = select_recompute_chunks(
            len(finalized), event_split_indices, profile.kv_recompute_recent_k
        )
        # a recompute that drops nothing is a plain encode, except where the chain wants the switch compressed anyway
        if len(selected) >= len(finalized) and not (
            event_switch and profile.force_recompute_at_event
        ):
            do_recompute = False
    if do_recompute:
        history = torch.cat([finalized[i] for i in selected], dim=2)
        history = context_noise_fn(history) if context_noise_fn else history
        denoiser.recompute(history, [finalized[i].shape[2] for i in selected], cache)
        result.recomputes.append(len(finalized))
    else:
        denoiser.encode(context_noise_fn(clean) if context_noise_fn else clean, cache)
