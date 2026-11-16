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

"""Scheduling of the rolling window: per-frame step indices and the per-frame CFG scale.

In a rolling window every frame (chunk) sits at a different point of the denoising ladder, so one global step
maps to a VECTOR of step indices. `StreamingSchedulerWrapper` owns that mapping (a cyclically shifted index
table) and `FlowMatchStreamingScheduler` consumes it: its `step` indexes `sigmas` element-wise.

`FlowMatchStreamingScheduler` is the one stepping scheduler of every rolling chain -- Base and distilled
inference, Stage-2 and Stage-3 training -- and the pipelines accept it by `isinstance`. DMD training scores with
its parent `FlowMatchDiscreteScheduler`, which has no `step`, so a score scheduler cannot be handed to a rollout.
"""

import math

import torch

from diffusers.schedulers.scheduling_utils import SchedulerMixin

from movieroll.commons.streaming_config import StreamingConfig
from movieroll.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler


def broadcast_step_index(step_index: torch.Tensor) -> torch.Tensor:
    """Reshape a 1-D per-frame step_index to [1, 1, T, 1, 1] so it broadcasts against [B, C, T, H, W]."""
    return step_index.view(1, 1, -1, 1, 1)


class FlowMatchStreamingScheduler(FlowMatchDiscreteScheduler):
    """Rolling-window stepping: `step_index` may be a tensor, and sigmas are differenced element-wise."""

    def step(
        self,
        model_output: torch.FloatTensor,
        step_index: torch.Tensor,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """One Euler step in fp32: ``prev = sample + model_output * (sigmas[i + 1] - sigmas[i])``.

        ``step_index`` is a tensor of per-frame indices already shaped to broadcast against ``[B, C, T, H, W]``
        (``StreamingSchedulerWrapper.step`` passes ``[1, 1, T, 1, 1]``), so every frame advances by its own
        sigma. The scheduler keeps no step counter; the caller owns the index.
        """
        sample = sample.to(torch.float32)
        sigmas = self.sigmas.to(sample.device)

        flat_idx = step_index.reshape(-1).long()
        sigma_cur = sigmas[flat_idx].reshape(step_index.shape)
        sigma_next = sigmas[flat_idx + 1].reshape(step_index.shape)
        dt = sigma_next - sigma_cur

        return sample + model_output.float() * dt.to(sample.device)


class StreamingSchedulerWrapper:
    """Scheduler wrapper for rolling-window denoising.

    Responsibilities:
    - own the cyclically shifted step_index table, one row per window slot;
    - map a global step count onto the per-frame step indices of the current window;
    - delegate the actual denoising to scheduler.step().
    """

    def __init__(
        self,
        streaming_config: StreamingConfig,
        scheduler: SchedulerMixin,
        window_size: int,
        num_inference_steps: int,
    ):
        self.scheduler = scheduler
        self.window_size = window_size
        self.num_inference_steps = num_inference_steps
        self.steps_per_window = num_inference_steps // window_size

        # cyclically shifted index table, shape [window_size, num_inference_steps]
        orig = torch.arange(num_inference_steps)
        # `torch.roll` reduces the shift modulo the length while the hand-rolled concat it replaces did not;
        # the two agree because `steps_per_window = num_inference_steps // window_size` above bounds every
        # shift by `(window_size - 1) * (num_inference_steps // window_size) < num_inference_steps`.
        rows = [torch.roll(orig, w * self.steps_per_window) for w in range(window_size)]
        self.step_indices: torch.Tensor = torch.stack(rows, dim=0)

    def get_step_index(self, global_step: int) -> torch.Tensor:
        """Per-frame step_index of the current window, shape [window_size]."""
        step_idx = global_step % self.num_inference_steps
        offset = math.ceil(
            max(global_step - self.num_inference_steps + 1, 0) / self.steps_per_window
        ) % self.window_size
        return torch.cat([
            self.step_indices[offset:, step_idx],
            self.step_indices[:offset, step_idx],
        ])

    def step(
        self,
        noise_pred: torch.Tensor,
        step_index: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Run one denoising step."""
        step_index = broadcast_step_index(step_index)
        return self.scheduler.step(noise_pred, step_index, latents)


class AnnealCFG:
    """AnnealCFG: the per-frame CFG scale, annealed linearly in the timestep.

    w(t) = cfg_min + (cfg_max - cfg_min) * t / num_train_timesteps, which is the paper's 1 + (w_max - 1) * t / 1000
    at the shipped cfg_min = 1.

    The scale runs from `cfg_max` in the high-noise part of the schedule down to `cfg_min` in the low-noise
    part. Interpolating in the timestep (a physical noise level) rather than in the step index is what keeps the
    scale meaningful when the frames of a rolling window sit at different points of the ladder.
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
        """CFG scale of every frame, same shape as ``timesteps``.

        Args:
            timesteps: per-frame shifted timestep (the scheduler's ladder values, not step indices), shape
                [num_frames] (already expanded to frame level).
        """
        t = timesteps.float()
        ratio = (self.num_train_timesteps - t) / self.num_train_timesteps
        ratio = 1.0 - ratio
        return self.cfg_min + (self.cfg_max - self.cfg_min) * ratio
