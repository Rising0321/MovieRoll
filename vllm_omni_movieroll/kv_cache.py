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

The streaming blocks attend to the clean history through this cache. It is owned by the request (the pipeline creates
one per generation and hands it to the transformer), not by the transformer, so a worker can serve several requests
without them seeing each other's history.

Per block it keeps the pre-RoPE key / value of every cached chunk, concatenated along the token axis, and the frame
count of every chunk. Keys are stored before RoPE on purpose: KV recompute and HieraKV re-pack the history, so a
chunk's position changes between uses. Under Ring attention a rank keeps only some tokens of each chunk (its block);
then the positions of those tokens inside their chunk are kept too, so the RoPE row of every cached token stays known.
"""

from __future__ import annotations

from typing import Optional

import torch


class KVCache:
    """Per-block (key, value) ``[B, tokens, heads, head_dim]`` of the clean chunks, each chunk's frame count and, when a rank
    keeps only part of a chunk, which positions of it."""

    def __init__(self, num_blocks: int, tokens_per_frame: int, dtype: torch.dtype = torch.bfloat16):
        if num_blocks <= 0 or tokens_per_frame <= 0:
            raise ValueError(f"num_blocks and tokens_per_frame must be positive, got {num_blocks}, {tokens_per_frame}")
        self.tokens_per_frame = tokens_per_frame
        self.dtype = dtype
        self._key: list[Optional[torch.Tensor]] = [None] * num_blocks
        self._value: list[Optional[torch.Tensor]] = [None] * num_blocks
        self._chunk_frames: list[list[int]] = [[] for _ in range(num_blocks)]
        # per chunk: positions (inside the chunk) of the tokens kept here, or None for all of them
        self._chunk_positions: list[list[Optional[torch.Tensor]]] = [[] for _ in range(num_blocks)]
        # the chunks of the full history, kept apart from the per-block lists so the recompute decision does not depend on
        # what any one block holds
        self._history: list[int] = []

    @property
    def num_blocks(self) -> int:
        return len(self._key)

    def get(self, block: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        key = self._key[block]
        return None if key is None else (key, self._value[block])

    def num_tokens(self, block: int) -> int:
        """Tokens kept on this rank."""
        key = self._key[block]
        return 0 if key is None else int(key.shape[1])

    def num_frames(self, block: int) -> int:
        return sum(self._chunk_frames[block])

    def total_tokens(self, block: int) -> int:
        """Tokens of the whole history (every rank's part), i.e. where the next window starts."""
        return self.num_frames(block) * self.tokens_per_frame

    def chunk_frames(self, block: int) -> list[int]:
        return list(self._chunk_frames[block])

    def _chunk_count(self, frames: int, positions: Optional[torch.Tensor]) -> int:
        return frames * self.tokens_per_frame if positions is None else int(positions.numel())

    def positions(self, block: int) -> Optional[torch.Tensor]:
        """History positions (in the packed history) of the tokens kept here, or None when every token is kept, in order."""
        chunks = self._chunk_positions[block]
        if all(p is None for p in chunks):
            return None
        out, offset = [], 0
        for frames, chunk in zip(self._chunk_frames[block], chunks):
            tokens = frames * self.tokens_per_frame
            out.append(offset + (chunk if chunk is not None else torch.arange(tokens, device=self._key[block].device)))
            offset += tokens
        return torch.cat(out)

    def append(self, block: int, key: torch.Tensor, value: torch.Tensor, frames: int,
               positions: Optional[torch.Tensor] = None) -> None:
        """Add one chunk of ``frames`` latent frames behind the cached ones (``positions``: which of its tokens these are)."""
        if key.shape != value.shape:
            raise ValueError(f"key {tuple(key.shape)} and value {tuple(value.shape)} must have the same shape")
        if key.shape[1] != self._chunk_count(frames, positions):
            raise ValueError(
                f"a chunk of {frames} frame(s) holds {self._chunk_count(frames, positions)} tokens here, got {key.shape[1]}"
            )
        key = key.detach().to(self.dtype).contiguous()
        value = value.detach().to(self.dtype).contiguous()
        previous = self.get(block)
        if previous is None:
            self._key[block], self._value[block] = key, value
        else:
            if previous[0].shape[0] != key.shape[0] or previous[0].shape[2:] != key.shape[2:]:
                raise ValueError(f"block {block}: cached {tuple(previous[0].shape)} vs new {tuple(key.shape)}")
            self._key[block] = torch.cat([previous[0], key], dim=1)
            self._value[block] = torch.cat([previous[1], value], dim=1)
        self._chunk_frames[block].append(frames)
        self._chunk_positions[block].append(positions)

    def trim_to_sink(self, block: int, sink_chunks: int) -> None:
        """Keep only the first ``sink_chunks`` chunks of a sink-recent block (the stale recent chunk is dropped)."""
        frames = self._chunk_frames[block]
        if len(frames) <= sink_chunks:
            return
        positions = self._chunk_positions[block]
        keep = sum(self._chunk_count(f, p) for f, p in zip(frames[:sink_chunks], positions[:sink_chunks]))
        self._key[block] = self._key[block][:, :keep].clone()
        self._value[block] = self._value[block][:, :keep].clone()
        self._chunk_frames[block] = frames[:sink_chunks]
        self._chunk_positions[block] = positions[:sink_chunks]

    def replace(self, block: int, key: torch.Tensor, value: torch.Tensor, chunk_frames: list[int],
                chunk_positions: Optional[list[Optional[torch.Tensor]]] = None) -> None:
        """Set a block's cache wholesale (KV recompute)."""
        chunk_positions = list(chunk_positions) if chunk_positions is not None else [None] * len(chunk_frames)
        if key.shape != value.shape:
            raise ValueError(f"key {tuple(key.shape)} and value {tuple(value.shape)} must have the same shape")
        expected = sum(self._chunk_count(f, p) for f, p in zip(chunk_frames, chunk_positions))
        if key.shape[1] != expected:
            raise ValueError(f"{key.shape[1]} tokens do not match chunks of {chunk_frames} frame(s) ({expected} expected)")
        self._key[block] = key.detach().to(self.dtype).contiguous()
        self._value[block] = value.detach().to(self.dtype).contiguous()
        self._chunk_frames[block] = list(chunk_frames)
        self._chunk_positions[block] = chunk_positions

    def record_chunk(self, frames: int) -> None:
        self._history.append(frames)

    def record_recompute(self, chunk_frames: list[int]) -> None:
        self._history = list(chunk_frames)

    def clear(self) -> None:
        self._history = []
        for block in range(self.num_blocks):
            self._key[block] = None
            self._value[block] = None
            self._chunk_frames[block] = []
            self._chunk_positions[block] = []

    def should_recompute(self, threshold: int) -> bool:
        """Whether the number of cached chunks reached the KV recompute threshold."""
        return len(self._history) >= threshold
