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

"""Where a request's frames, chunks and events sit: the geometry of a rolling generation.

Ported from `movieroll`: `models.transformers.modules.chunking.check_and_get_chunk_sizes` (chunk layout),
`generate.py`'s `_check_video_length` / `_check_i2v_video_length` (request rules) and
`pipelines.movieroll_pipeline.MovieRollStreamingPipeline._compute_event_split_indices` (event boundaries). Every
function matches its `movieroll` original.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# The temporal compression the frame arithmetic is written for: latent_frames = (pixel_frames - 1) // 4 + 1.
TEMPORAL_COMPRESSION = 4


def latent_length(video_length: int) -> int:
    return (video_length - 1) // TEMPORAL_COMPRESSION + 1


def chunk_sizes_of(
    total_latent_frames: int, chunk_size: int, *, anchored: bool
) -> List[int]:
    """Latent frames of every chunk of a video of ``total_latent_frames`` frames.

    The first chunk is the one that differs: a single frame for an image-to-video layout (the input frame),
    otherwise the leading frame plus a full chunk so the first window has context. When the latent length is a
    whole number of chunks there is no leading frame and every chunk is equal.
    """
    if total_latent_frames % chunk_size not in (0, 1):
        raise ValueError(
            f"{total_latent_frames} latent frames is neither a multiple of chunk_size={chunk_size} nor one more"
        )
    if total_latent_frames % chunk_size == 1:
        first = 1 if anchored else 1 + chunk_size
    else:
        first = chunk_size
    first = min(first, total_latent_frames)
    sizes = [first] + [chunk_size] * ((total_latent_frames - first) // chunk_size)
    if sum(sizes) != total_latent_frames:
        raise ValueError(
            f"chunk sizes {sizes} do not add up to {total_latent_frames} latent frames"
        )
    return sizes


def chunk_layout(
    video_length: int, chunk_size: int, *, anchored: bool, strict: bool
) -> List[int]:
    """Chunk layout of ``video_length`` pixel frames.

    ``strict`` is the distilled rule (the latent length must already be a whole number of chunks plus one); the
    base chain rounds down to it, as its pipeline does.
    """
    if video_length <= 0 or (video_length - 1) % TEMPORAL_COMPRESSION:
        raise ValueError(
            f"video_length must be positive and 4n+1 pixel frames, got {video_length}"
        )
    total = latent_length(video_length)
    if strict:
        if total % chunk_size != 1:
            raise ValueError(
                f"video_length {video_length} gives {total} latent frames, which is not 1 modulo chunk_size={chunk_size}"
            )
    else:
        total = (total - 1) // chunk_size * chunk_size + 1
    return chunk_sizes_of(total, chunk_size, anchored=anchored)


def check_video_length(
    video_length: int, chunk_size: int, mode: str, anchored: bool
) -> None:
    """The frame-count rules of the command line, with the same messages.

    A count that is not 1 modulo the chunk size would silently produce a shorter video on the base chain and is
    refused outright on the distilled one; the distilled chain and every image-to-video run also need a whole
    first chunk.
    """
    if (
        isinstance(video_length, bool)
        or not isinstance(video_length, int)
        or video_length <= 0
    ):
        raise ValueError(
            f"video_length must be a positive number of pixel frames, got {video_length!r}"
        )
    if (video_length - 1) % TEMPORAL_COMPRESSION:
        # a duration from the OpenAI video endpoint lands here (fps * seconds is rarely 4n + 1)
        raise ValueError(
            f"video_length must be 4n+1 pixel frames, got {video_length}; the nearest valid frame counts "
            f"are {nearest_valid_video_lengths(video_length, chunk_size)}"
        )
    total = latent_length(video_length)
    if total % chunk_size != 1:
        raise ValueError(
            f"video_length {video_length} gives {total} latent frames, which is not 1 modulo chunk_size="
            f"{chunk_size}; the nearest valid frame counts are {nearest_valid_video_lengths(video_length, chunk_size)}"
        )
    if (mode == "distilled" or anchored) and total < 1 + chunk_size:
        raise ValueError(
            f"video_length {video_length} is too short: the first chunk needs {1 + chunk_size} latent frames "
            f"(shortest video: {4 * chunk_size + 1} frames)"
        )


def nearest_valid_video_lengths(video_length: int, chunk_size: int) -> List[int]:
    """The valid frame counts around ``video_length`` (what the error message offers the caller)."""
    total = latent_length(max(video_length, 1))
    num_chunks = max((total - 1) // chunk_size, 0)
    return [4 * (n * chunk_size) + 1 for n in (num_chunks, num_chunks + 1) if n >= 1]


def event_split_indices(
    num_chunks: int,
    event_prompts: Optional[Sequence[str]],
) -> Tuple[List[int], List[int]]:
    """``(split_indices, kept_event_indices)``: where each event starts, and which events keep a chunk.

    ``split_indices[k]`` is the chunk index at which segment k+1 starts; the chunks are split evenly across the
    events. Events whose segment comes out empty are dropped, so the prompts of the kept events line up with the
    boundaries. A single event is ``([], [0])``.
    """
    if event_prompts is None or len(event_prompts) < 2:
        return [], [0]

    num_events = len(event_prompts)
    splits = [(num_chunks * event) // num_events for event in range(1, num_events)]

    boundaries = [0] + splits + [num_chunks]
    kept: List[int] = [
        event
        for event in range(len(boundaries) - 1)
        if boundaries[event] < boundaries[event + 1]
    ]
    deduped = [splits[event - 1] for event in kept[1:]]
    return deduped, kept


def select_recompute_chunks(
    num_chunks: int, splits: Sequence[int], recent_k: int
) -> List[int]:
    """Output chunks a KV recompute keeps: the first, the middle of every event, and the ``recent_k`` newest."""
    selected = {0}
    # boundaries at or past the chunks decoded so far do not exist yet
    bounded = sorted(s for s in splits if 0 < s < num_chunks)
    boundaries = [0] + bounded + [num_chunks]
    for start, end in zip(boundaries, boundaries[1:]):
        if start < end:
            selected.add((start + end - 1) // 2)
    selected.update(range(max(0, num_chunks - recent_k), num_chunks))
    return sorted(selected)


def event_index_per_chunk(num_chunks: int, splits: Sequence[int]) -> List[int]:
    """Which event drives each output chunk."""
    chunk_to_event = [0] * num_chunks
    for event, boundary in enumerate(splits):
        for chunk in range(boundary, num_chunks):
            chunk_to_event[chunk] = event + 1
    return chunk_to_event
