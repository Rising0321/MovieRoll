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

"""Per-frame scheduling of a rolling window: which step each chunk is on, its guidance scale, its context noise.

The sigma ladder itself is the framework's (SGLang's `FlowMatchEulerDiscreteScheduler`, which produces the same
sigmas and timesteps as MovieRoll's `FlowMatchStreamingScheduler`, bitwise). What is MovieRoll's own, and
ported here, is everything that follows from a window whose chunks sit at different points of that ladder:

* `WindowSchedule`: the cyclically shifted index table that maps a global step onto one step index per chunk
  (`movieroll.schedulers.scheduling_flow_match_streaming.StreamingSchedulerWrapper`);
* `flow_match_step`: the Euler update that reads a different sigma per frame (that module's
  `FlowMatchStreamingScheduler.step`);
* `AnnealCFG`: the per-frame guidance scale, linear in the timestep (the same module's class);
* `apply_context_noise`: the re-noising of a clean chunk before it enters the KV cache
  (`movieroll.models.transformers.modules.context_noise`, inference branch: one sigma for every frame).

All four match the `movieroll` originals.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class WindowSchedule:
    """Which step of the ladder every chunk of the window is on, round by round."""

    def __init__(self, window_size: int, num_inference_steps: int):
        if num_inference_steps % window_size:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) must be a multiple of window_size ({window_size})"
            )
        self.window_size = window_size
        self.num_inference_steps = num_inference_steps
        self.steps_per_window = num_inference_steps // window_size
        rows = [
            torch.roll(torch.arange(num_inference_steps), w * self.steps_per_window)
            for w in range(window_size)
        ]
        self.step_indices = torch.stack(rows, dim=0)

    def get_step_index(self, global_step: int) -> torch.Tensor:
        """Per-chunk step index of the current window, shape ``[window_size]``."""
        step = global_step % self.num_inference_steps
        offset = (
            math.ceil(
                max(global_step - self.num_inference_steps + 1, 0)
                / self.steps_per_window
            )
            % self.window_size
        )
        return torch.cat(
            [self.step_indices[offset:, step], self.step_indices[:offset, step]]
        )


def flow_match_step(
    model_output: torch.Tensor,
    step_index: torch.Tensor,
    sample: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """One Euler step in fp32 with a per-frame sigma: ``prev = sample + velocity * (sigma[i + 1] - sigma[i])``.

    ``step_index`` holds one index per latent frame and is reshaped to ``[1, 1, T, 1, 1]`` so every frame of the
    window advances by its own sigma.
    """
    sample = sample.to(torch.float32)
    index = step_index.view(1, 1, -1, 1, 1)
    sigmas = sigmas.to(sample.device)
    flat = index.reshape(-1).long()
    sigma_cur = sigmas[flat].reshape(index.shape)
    sigma_next = sigmas[flat + 1].reshape(index.shape)
    return sample + model_output.float() * (sigma_next - sigma_cur).to(sample.device)


class AnnealCFG:
    """The per-frame guidance scale, annealed linearly in the timestep: ``cfg_min + (cfg_max - cfg_min) * t / T``.

    Interpolating in the timestep (a physical noise level) rather than in the step index is what keeps the scale
    meaningful when the frames of a window sit at different points of the ladder.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        cfg_min: float = 1.0,
        cfg_max: float = 4.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.cfg_min = cfg_min
        self.cfg_max = cfg_max

    def get_cfg(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Guidance scale of every frame, same shape as ``timesteps`` (the ladder's values, not step indices)."""
        ratio = (
            1.0
            - (self.num_train_timesteps - timesteps.float()) / self.num_train_timesteps
        )
        return self.cfg_min + (self.cfg_max - self.cfg_min) * ratio


def apply_context_noise(
    latents: torch.Tensor, sigma: float, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """``(1 - sigma) * latents + sigma * noise`` with one sigma for every frame (the inference branch).

    The noise is drawn on the generator's own device and moved to the latents', so the random stream of a
    request is the chain's (the distilled chain draws on the CPU, the base chain on the GPU).
    """
    device = torch.device(generator.device) if generator is not None else latents.device
    noise = torch.randn(
        latents.shape, device=device, dtype=latents.dtype, generator=generator
    ).to(latents.device)
    return ((1 - sigma) * latents + sigma * noise).contiguous()
