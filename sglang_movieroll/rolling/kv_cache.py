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

"""Clean-history KV cache of one rolling generation.

The streaming blocks attend to the clean history through this cache. It belongs to the request -- the rolling
stage creates one per generation and hands it to the transformer -- so a worker serves one request after another
without them seeing each other's history.

Per block it keeps the pre-RoPE key / value of every cached chunk, concatenated along the token axis, and each
chunk's latent frame count. Keys are stored before RoPE on purpose: KV recompute and HieraKV re-pack the
history, so a chunk's position changes between uses.

How many of a chunk's tokens a rank stores depends on the parallel layout, which this package knows nothing
about: ``stored_tokens`` is handed in as a function of a chunk's frame count (the identity on one GPU and under
Ulysses, a ring rank's share under ring parallelism). The frame counts are therefore kept separately from the
stored token counts: the frame count says where a chunk sits in the re-packed history, the function says which
part of it is here.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch


class KVCache:
    """Per-block (key, value) of the clean chunks, ``[B, tokens, heads, head_dim]``, plus each chunk's frames."""

    def __init__(
        self,
        num_blocks: int,
        tokens_per_frame: int,
        dtype: torch.dtype = torch.bfloat16,
        stored_tokens: Optional[Callable[[int], int]] = None,
    ) -> None:
        if num_blocks <= 0 or tokens_per_frame <= 0:
            raise ValueError(
                f"num_blocks and tokens_per_frame must be positive, got {num_blocks}, {tokens_per_frame}"
            )
        self.tokens_per_frame = tokens_per_frame
        self.dtype = dtype
        self._stored_tokens = stored_tokens or (lambda tokens: tokens)
        self._key: List[Optional[torch.Tensor]] = [None] * num_blocks
        self._value: List[Optional[torch.Tensor]] = [None] * num_blocks
        self._chunk_frames: List[List[int]] = [[] for _ in range(num_blocks)]

    @property
    def num_blocks(self) -> int:
        return len(self._key)

    def stored_tokens_of(self, frames: int) -> int:
        """How many tokens of a chunk of ``frames`` latent frames this rank stores."""
        return self._stored_tokens(frames * self.tokens_per_frame)

    def get(self, block: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        key = self._key[block]
        return None if key is None else (key, self._value[block])

    def chunk_frames(self, block: int) -> List[int]:
        return list(self._chunk_frames[block])

    def append(
        self, block: int, key: torch.Tensor, value: torch.Tensor, frames: int
    ) -> None:
        """Add this rank's share of one chunk of ``frames`` latent frames behind the ones already cached."""
        self._check(key, value, [frames])
        key = key.detach().to(self.dtype).contiguous()
        value = value.detach().to(self.dtype).contiguous()
        previous = self.get(block)
        if previous is None:
            self._key[block], self._value[block] = key, value
        else:
            if (
                previous[0].shape[0] != key.shape[0]
                or previous[0].shape[2:] != key.shape[2:]
            ):
                raise ValueError(
                    f"block {block}: cached {tuple(previous[0].shape)} vs new {tuple(key.shape)}"
                )
            self._key[block] = torch.cat([previous[0], key], dim=1)
            self._value[block] = torch.cat([previous[1], value], dim=1)
        self._chunk_frames[block].append(frames)

    def trim_to_sink(self, block: int, sink_chunks: int) -> None:
        """Keep only the first ``sink_chunks`` chunks of a sink-recent block (the stale recent chunk is dropped)."""
        frames = self._chunk_frames[block]
        if len(frames) <= sink_chunks:
            return
        keep = sum(self.stored_tokens_of(count) for count in frames[:sink_chunks])
        self._key[block] = self._key[block][:, :keep].clone()
        self._value[block] = self._value[block][:, :keep].clone()
        self._chunk_frames[block] = frames[:sink_chunks]

    def replace(
        self,
        block: int,
        key: torch.Tensor,
        value: torch.Tensor,
        chunk_frames: List[int],
    ) -> None:
        """Set a block's cache wholesale (KV recompute)."""
        self._check(key, value, chunk_frames)
        self._key[block] = key.detach().to(self.dtype).contiguous()
        self._value[block] = value.detach().to(self.dtype).contiguous()
        self._chunk_frames[block] = list(chunk_frames)

    def _check(
        self, key: torch.Tensor, value: torch.Tensor, chunk_frames: List[int]
    ) -> None:
        if key.shape != value.shape:
            raise ValueError(
                f"key {tuple(key.shape)} and value {tuple(value.shape)} must have the same shape"
            )
        expected = sum(self.stored_tokens_of(frames) for frames in chunk_frames)
        if key.shape[1] != expected:
            raise ValueError(
                f"chunks of {chunk_frames} latent frame(s) hold {expected} token(s) on this rank, got {key.shape[1]}"
            )

    def clear(self) -> None:
        for block in range(self.num_blocks):
            self._key[block] = None
            self._value[block] = None
            self._chunk_frames[block] = []

    def should_recompute(self, threshold: int) -> bool:
        """Whether the number of cached chunks reached the KV recompute threshold (read off the first block)."""
        return len(self._chunk_frames[0]) >= threshold
