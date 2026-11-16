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

"""
Chunk size generation utilities.

Split a video of `tt` latent frames into chunks of `chunk_size` frames.

`mode` sets the size of the first chunk when tt % chunk_size == 1:
  - 'i2v': first_chunk = 1
      the first chunk only holds the single reference frame
  - 'default': first_chunk = 1 + chunk_size
      the first chunk holds the leading frame plus a full chunk, so that the first window has enough context

When tt % chunk_size == 0 (no leftover frame) both modes use first_chunk = chunk_size, i.e. all chunks are equal.
"""

# Chunk sizes for a video length and chunk_size.
def check_and_get_chunk_sizes(tt, chunk_size=1, mode='default'):
    """Split `tt` latent frames into chunks.

    Args:
        tt: total number of latent frames
        chunk_size: regular chunk size; tt must satisfy tt % chunk_size in (0, 1)
        mode: 'i2v' or 'default', sets the first chunk size (see the module docstring)

    Returns:
        chunk_sizes: List[int], number of frames of every chunk
    """
    # frames should be multiple of chunk_size + 1 or multiple of chunk_size
    assert (tt % chunk_size == 0 or tt % chunk_size == 1), f"tt {tt} should be multiple of chunk_size {chunk_size} + 1 or multiple of chunk_size {chunk_size}"

    if mode == 'i2v':
        first_frame_chunk_size = 1 if tt % chunk_size == 1 else chunk_size
    else:
        first_frame_chunk_size = 1 + chunk_size if tt % chunk_size == 1 else chunk_size

    # Boundary guard: for very short inputs (e.g. a single frame) the first chunk may exceed tt.
    first_frame_chunk_size = min(first_frame_chunk_size, tt)
    chunk_sizes = [first_frame_chunk_size] + [chunk_size] * ((tt - first_frame_chunk_size) // chunk_size)

    # Verify total frames
    total_frames = sum(chunk_sizes)
    assert total_frames == tt, f"Chunk sizes {chunk_sizes} don't sum to {tt}"
    return chunk_sizes

def check_and_get_window_sizes(
    num_chunks,
    window_size=1,
    *,
    window_phase,
):
    """Split `num_chunks` chunks into training windows, with the grid offset by `window_phase`.

    The first window holds `window_phase + 1` chunks and every following window holds `window_size`; a
    trailing remainder that cannot fill a whole window is dropped. So the phase slides one fixed grid along
    the clip: phase 0 gives the shortest possible first window (one chunk) and phase `window_size - 1` a
    full one.

    This matches Stage-3's convention, where `start_phase` selects rounds `p, p + window_size, ...` so the
    first selected round falls in the PREFILL growth stage with exactly `start_phase + 1` chunks in flight.
    The same phase value means the same thing in both stages.

    Args:
        num_chunks: total number of chunks
        window_size: number of chunks per window
        window_phase: grid offset, 0 <= window_phase < window_size. The caller draws it, because the draw
            has to be synchronised across the sequence-parallel group.
    Returns:
        window_sizes: List[int], number of chunks of every window
    """
    assert window_size >= 1, f"window_size {window_size} must be >= 1"
    assert num_chunks >= 1, f"num_chunks {num_chunks} must be >= 1"
    assert 0 <= window_phase < window_size, \
        f"window_phase {window_phase} must satisfy 0 <= phase < window_size {window_size}"
    if window_size == 1:
        return [1] * num_chunks

    first_window_size = min(window_phase + 1, num_chunks)

    remaining = num_chunks - first_window_size
    if remaining == 0:
        return [first_window_size]

    return [first_window_size] + [window_size] * (remaining // window_size)

def get_window_wrapped_chunk_sizes(chunk_sizes, window_sizes):
    assert len(chunk_sizes) == sum(window_sizes), f"chunk_sizes length {len(chunk_sizes)} must be equal to sum of window_sizes {sum(window_sizes)}"
    window_wrapped_chunk_sizes = []
    current_index = 0
    for window_size in window_sizes:
        window_wrapped_chunk_sizes.append(chunk_sizes[current_index:current_index + window_size])
        current_index += window_size
    return window_wrapped_chunk_sizes
