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

"""The FIFO of latents inside the rolling window (ported from `movieroll.models.transformers.modules.rolling_queue`).

While the window grows, noisy chunks are appended at the tail; in the steady state the oldest chunk, now clean,
is popped from the head as a fresh noisy one is appended.
"""

from __future__ import annotations

from typing import List, Optional

import torch


class RollingQueue:
    def __init__(self) -> None:
        self.queue: Optional[torch.Tensor] = None
        self.chunk_lengths: List[int] = []

    def add_new_noisy_chunk(self, noisy_chunk: torch.Tensor) -> None:
        """Append a noisy chunk ``[B, C, T, H, W]`` at the tail."""
        self.queue = (
            noisy_chunk
            if self.queue is None
            else torch.cat([self.queue, noisy_chunk], dim=2)
        )
        self.chunk_lengths.append(noisy_chunk.shape[2])

    def pop_clean_chunk(self) -> torch.Tensor:
        """Pop the oldest chunk from the head."""
        length = self.chunk_lengths[0]
        clean = self.queue[:, :, :length]
        remaining = self.queue[:, :, length:]
        self.queue = remaining if remaining.shape[2] > 0 else None
        self.chunk_lengths.pop(0)
        return clean

    def get_current_latents(self) -> Optional[torch.Tensor]:
        return self.queue

    def get_num_chunks_in_window(self) -> int:
        return len(self.chunk_lengths)

    def update_with_new_latents(self, new_latents: torch.Tensor) -> None:
        """Replace the queue content with new latents of the same chunk layout."""
        self.queue = new_latents
