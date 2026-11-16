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

"""The rolling-window sampling loop of MovieRoll, for inference only.

One round pushes a fresh noisy chunk into the rolling window, denoises the whole window (each chunk at its own point of the
timestep ladder), then pops the oldest chunk, now clean, as output and encodes it into the KV cache; past a cache threshold,
or at an event boundary, the cache is rebuilt from a selection of the output chunks (KV recompute).

The loop knows nothing about the model or the framework: the pipeline hands it callables (predict a window's velocity,
take a step, encode a clean chunk, rebuild the cache). The two released chains share the loop and differ in a few places
kept explicit in `RollingProfile`: Base runs ``num_inference_steps / window_size`` steps per round with classifier-free
guidance (AnnealCFG), Distilled one step per round without; Distilled forces a recompute at an event switch even when
nothing can be dropped and noises the i2v anchor like any other history, Base does neither.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from vllm_omni_movieroll.kv_cache import KVCache
from vllm_omni_movieroll.profiles import NUM_TRAIN_TIMESTEPS, VAE_TEMPORAL
from vllm_omni_movieroll.scheduling_movieroll import AnnealCFG, RollingStepIndex


class RollingStage(enum.Enum):
    PREFILL = 0     # the window is still filling up
    DECODE = 1      # every round pops one clean chunk


@dataclass(frozen=True)
class TextTokens:
    """The valid (unpadded) text conditioning of one prompt: Qwen2.5-VL ``[1, S, 3584]`` and byT5 ``[1, S2, 1472]`` (None
    when the prompt quotes nothing)."""

    qwen: torch.Tensor
    byt5: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class EventCondition:
    """The text conditioning of one event: the prompt, and for guided sampling the negative prompt."""

    positive: TextTokens
    negative: Optional[TextTokens] = None


@dataclass(frozen=True)
class RollingProfile:
    """What a chain fixes: the window geometry, the KV recompute policy and the per-chain differences."""

    mode: str                                  # "base" or "distilled"
    chunk_size: int = 4
    window_size: int = 5
    steps_per_window: int = 1
    kv_recompute_enabled: bool = True
    kv_recompute_threshold: int = 18
    kv_recompute_recent_k: int = 3
    guided: bool = False                       # classifier-free guidance with AnnealCFG
    cfg_min: float = 1.0
    cfg_max: float = 4.0
    force_recompute_at_event: bool = False     # Distilled: recompute at an event switch even if nothing is dropped
    noise_anchor: bool = False                 # Distilled: the i2v anchor goes through the context noise too

    def __post_init__(self):
        if self.mode not in ("base", "distilled"):
            raise ValueError(f"mode must be 'base' or 'distilled', got {self.mode!r}")
        if self.guided and self.mode != "base":
            raise ValueError("only the base chain is guided")

    @property
    def num_inference_steps(self) -> int:
        return self.window_size * self.steps_per_window


# --------------------------------------------------------------------------- chunk layout
def chunk_sizes_of(latent_length: int, chunk_size: int, anchored: bool) -> list[int]:
    """Split ``latent_length`` latent frames into chunks (MovieRoll's `check_and_get_chunk_sizes`).

    With one leftover frame the first chunk is ``1`` (i2v: the single anchor frame) or ``1 + chunk_size`` (t2v: the
    leading frame plus a whole chunk); without one all chunks are equal.
    """
    if latent_length % chunk_size not in (0, 1):
        raise ValueError(f"{latent_length} latent frames are not a multiple of chunk_size={chunk_size} (+ 1)")
    if anchored:
        first = 1 if latent_length % chunk_size == 1 else chunk_size
    else:
        first = 1 + chunk_size if latent_length % chunk_size == 1 else chunk_size
    first = min(first, latent_length)
    sizes = [first] + [chunk_size] * ((latent_length - first) // chunk_size)
    if sum(sizes) != latent_length:
        raise ValueError(f"chunk sizes {sizes} do not add up to {latent_length}")
    return sizes


def chunk_layout(video_length: int, chunk_size: int, *, anchored: bool, strict: bool) -> list[int]:
    """Latent frames of every chunk for ``video_length`` pixel frames.

    ``strict`` is the distilled rule (the latent length must already be a whole number of chunks plus one); the base chain
    rounds down to it. ``anchored`` is the i2v layout whose first chunk is the single input frame.
    """
    if video_length <= 0 or (video_length - 1) % VAE_TEMPORAL:
        raise ValueError(f"video_length must be positive and 4n+1 pixel frames, got {video_length}")
    latent_length = (video_length - 1) // VAE_TEMPORAL + 1
    if strict:
        if latent_length % chunk_size != 1:
            raise ValueError(
                f"video_length {video_length} gives {latent_length} latent frames, which is not 1 modulo chunk_size={chunk_size}"
            )
    else:
        latent_length = (latent_length - 1) // chunk_size * chunk_size + 1
    return chunk_sizes_of(latent_length, chunk_size, anchored)


def event_split_indices(num_chunks: int, num_events: int) -> tuple[list[int], list[int]]:
    """``(split_indices, kept_events)``: the chunk where every next event starts, and the events that keep a chunk.

    An even split; an event whose segment comes out empty is dropped (MovieRoll's
    `MovieRollStreamingPipeline._compute_event_split_indices`).
    """
    if num_events < 2:
        return [], [0]
    splits = [(num_chunks * event) // num_events for event in range(1, num_events)]
    boundaries = [0] + splits + [num_chunks]
    kept = [event for event in range(len(boundaries) - 1) if boundaries[event] < boundaries[event + 1]]
    return [splits[event - 1] for event in kept[1:]], kept


def select_recompute_chunks(num_chunks: int, event_split_indices: Sequence[int], recent_k: int) -> list[int]:
    """Output chunks a recompute keeps: the first, the middle of every event, and the ``recent_k`` most recent ones."""
    selected = {0}
    # boundaries at or past the chunks decoded so far do not exist yet
    splits = sorted(s for s in event_split_indices if 0 < s < num_chunks)
    boundaries = [0] + splits + [num_chunks]
    for start, end in zip(boundaries, boundaries[1:]):
        if start < end:
            selected.add((start + end - 1) // 2)
    selected.update(range(max(0, num_chunks - recent_k), num_chunks))
    return sorted(selected)


def event_index_per_chunk(num_chunks: int, event_split_indices: Sequence[int]) -> list[int]:
    chunk_to_event = [0] * num_chunks
    for event, boundary in enumerate(event_split_indices):
        for chunk in range(boundary, num_chunks):
            chunk_to_event[chunk] = event + 1
    return chunk_to_event


class RollingQueue:
    """The latents inside the rolling window: noisy chunks enter at the tail, the clean one leaves at the head."""

    def __init__(self):
        self.queue: Optional[torch.Tensor] = None
        self.chunk_lengths: list[int] = []

    def add_new_noisy_chunk(self, noisy_chunk: torch.Tensor) -> None:
        self.queue = noisy_chunk if self.queue is None else torch.cat([self.queue, noisy_chunk], dim=2)
        self.chunk_lengths.append(noisy_chunk.shape[2])

    def pop_clean_chunk(self) -> torch.Tensor:
        length = self.chunk_lengths.pop(0)
        clean, remaining = self.queue[:, :, :length], self.queue[:, :, length:]
        self.queue = remaining if remaining.shape[2] > 0 else None
        return clean

    def get_current_latents(self) -> Optional[torch.Tensor]:
        return self.queue

    def get_num_chunks_in_window(self) -> int:
        return len(self.chunk_lengths)

    def update_with_new_latents(self, new_latents: torch.Tensor) -> None:
        self.queue = new_latents


def _expand_step_index(step_index: torch.Tensor, queue: RollingQueue, stage: RollingStage) -> torch.Tensor:
    """One step index per window chunk -> one per latent frame."""
    in_window = queue.get_num_chunks_in_window()
    if stage == RollingStage.DECODE:
        if len(step_index) != in_window:
            raise RuntimeError(f"step_index has {len(step_index)} entries for {in_window} chunks in the decode stage")
    else:
        step_index = step_index[:in_window]
    return step_index.repeat_interleave(torch.tensor(queue.chunk_lengths, device=step_index.device))


@dataclass
class RollingResult:
    latents: Optional[torch.Tensor] = None     # all output chunks concatenated, ``[1, C, T, h, w]``
    chunk_frames: list[int] = field(default_factory=list)
    recomputes: list[int] = field(default_factory=list)   # number of output chunks at each recompute


VelocityFn = Callable[[torch.Tensor, torch.Tensor, EventCondition, Optional[torch.Tensor]], torch.Tensor]
StepFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def run_rolling(
    *,
    profile: RollingProfile,
    events: Sequence[EventCondition],
    chunk_sizes: list[int],
    event_split_indices: Sequence[int],
    timesteps: torch.Tensor,
    velocity_fn: VelocityFn,
    step_fn: StepFn,
    encode_fn: Callable[[torch.Tensor], None],
    recompute_fn: Callable[[torch.Tensor, list[int]], None],
    cache: KVCache,
    noise_fn: Callable[[int], torch.Tensor],
    context_noise_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    anchor: Optional[torch.Tensor] = None,
    on_chunk: Optional[Callable[[int, int], None]] = None,
    checkpoint: Optional[Callable[[], None]] = None,
) -> RollingResult:
    """Generate the output chunks.

    ``chunk_sizes`` is the full output layout (with the single-frame anchor chunk first when ``anchor`` is given),
    ``event_split_indices`` are boundaries in output-chunk coordinates, ``timesteps`` the chain's ladder (one per step),
    ``velocity_fn(latents, frame_timesteps, event, cfg_scale)`` the (guided) velocity of the window, ``step_fn(velocity,
    frame_step_index, latents)`` one Euler step per frame, ``noise_fn(frames)`` a fresh noisy chunk; ``checkpoint`` is
    called before every round and may raise to cancel.
    """
    window_size = profile.window_size
    if len(timesteps) != profile.num_inference_steps:
        raise ValueError(f"the ladder has {len(timesteps)} steps but the profile runs {profile.num_inference_steps}")
    step_indices = RollingStepIndex(window_size, profile.num_inference_steps)
    anneal = AnnealCFG(NUM_TRAIN_TIMESTEPS, profile.cfg_min, profile.cfg_max) if profile.guided else None
    queue = RollingQueue()
    finalized: list[torch.Tensor] = []
    result = RollingResult()

    chunk_to_event = event_index_per_chunk(len(chunk_sizes), event_split_indices)
    if anchor is not None:
        if chunk_sizes[0] != 1 or anchor.shape[2] != 1:
            raise ValueError("the i2v anchor is one latent frame and the first chunk of the layout")
        finalized.append(anchor)
        encode_fn(context_noise_fn(anchor) if (profile.noise_anchor and context_noise_fn) else anchor)
        chunk_sizes = chunk_sizes[1:]
        chunk_to_event = chunk_to_event[1:]
    num_chunks = len(chunk_sizes)
    if num_chunks <= 0:
        raise ValueError("nothing to generate")

    num_decoded = 0
    for round_idx in range(num_chunks + window_size):
        if checkpoint is not None:
            checkpoint()               # the cancellation point: raises to stop the generation between two windows
        stage = RollingStage.PREFILL if round_idx < window_size else RollingStage.DECODE

        if stage == RollingStage.DECODE:
            is_last = num_decoded == num_chunks - 1
            clean = queue.pop_clean_chunk()
            finalized.append(clean)
            num_decoded += 1
            if on_chunk is not None:
                on_chunk(num_decoded, num_chunks)
            if is_last:
                break
            _refresh_cache(cache, profile, finalized, clean, event_split_indices, context_noise_fn, encode_fn,
                           recompute_fn, result)

        frames = chunk_sizes[round_idx] if round_idx < num_chunks else chunk_sizes[-1]
        queue.add_new_noisy_chunk(noise_fn(frames))

        event = events[chunk_to_event[min(num_decoded, num_chunks - 1)]]
        for step in range(profile.steps_per_window):
            latents = queue.get_current_latents()
            step_index = _expand_step_index(
                step_indices.get_step_index(round_idx * profile.steps_per_window + step).to(timesteps.device), queue, stage
            )
            frame_timesteps = timesteps[step_index].unsqueeze(0)
            scale = None if anneal is None else anneal.get_cfg(frame_timesteps[0]).view(1, 1, -1, 1, 1)
            velocity = velocity_fn(latents, frame_timesteps, event, scale)
            queue.update_with_new_latents(step_fn(velocity, step_index, latents))

    result.latents = torch.cat(finalized, dim=2)
    result.chunk_frames = [chunk.shape[2] for chunk in finalized]
    return result


def _refresh_cache(cache, profile, finalized, clean, event_split_indices, context_noise_fn, encode_fn, recompute_fn,
                   result) -> None:
    """Put the chunk that was just popped into the cache: a plain encode, or a rebuild at a threshold / event switch."""
    event_switch = bool(event_split_indices) and len(finalized) in event_split_indices
    do_recompute = profile.kv_recompute_enabled and (event_switch or cache.should_recompute(profile.kv_recompute_threshold))
    if do_recompute:
        selected = select_recompute_chunks(len(finalized), event_split_indices, profile.kv_recompute_recent_k)
        # a recompute that drops nothing is a plain encode, except where the chain wants the event switch compressed anyway
        if len(selected) >= len(finalized) and not (event_switch and profile.force_recompute_at_event):
            do_recompute = False
    if do_recompute:
        history = torch.cat([finalized[i] for i in selected], dim=2)
        history = context_noise_fn(history) if context_noise_fn else history
        recompute_fn(history, [finalized[i].shape[2] for i in selected])
        result.recomputes.append(len(finalized))
    else:
        encode_fn(context_noise_fn(clean) if context_noise_fn else clean)
