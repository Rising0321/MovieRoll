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

"""Scheduling of the rolling window: the timestep ladder, per-frame Euler steps, AnnealCFG and context noise.

The ladder itself is the one diffusers' `FlowMatchEulerDiscreteScheduler` computes from the model directory's
`scheduler/` (shift 5; the same sigmas as MovieRoll's own scheduler). What is MovieRoll's own and lives here:
in a rolling window every frame (chunk) sits at a different point of the ladder, so one global step maps to a VECTOR of
step indices (`RollingStepIndex`), and `RollingStepper` advances every frame by its own sigma.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch


def ladder(scheduler, *, distilled: bool, num_inference_steps: int, denoising_step_list: Sequence[int],
           device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(timesteps, sigmas)`` of a chain from a diffusers flow-match scheduler (its config carries the shift).

    Base: ``num_inference_steps`` uniform sigmas in (0, 1]; Distilled: the fixed ``denoising_step_list`` (timesteps in
    [0, 1000]). ``sigmas`` has the terminal 0 appended, as diffusers and MovieRoll both do.
    """
    train = int(scheduler.config.num_train_timesteps)
    if distilled:
        steps = [t for t in denoising_step_list if t != 0]
        sigmas = np.asarray(steps, dtype=np.float64) / train
    else:
        sigmas = np.linspace(1.0, 0.0, int(num_inference_steps) + 1)[:-1]
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    return scheduler.timesteps.to(torch.float32), scheduler.sigmas.to(torch.float32)


class RollingStepIndex:
    """The cyclically shifted step-index table of a rolling window (MovieRoll's `StreamingSchedulerWrapper`).

    Row ``w`` is the ladder shifted by ``w * steps_per_window``; ``get_step_index(g)`` is the per-chunk index of global
    step ``g``.
    """

    def __init__(self, window_size: int, num_inference_steps: int):
        self.window_size = window_size
        self.num_inference_steps = num_inference_steps
        self.steps_per_window = num_inference_steps // window_size
        order = torch.arange(num_inference_steps)
        self.table = torch.stack([torch.roll(order, w * self.steps_per_window) for w in range(window_size)], dim=0)

    def get_step_index(self, global_step: int) -> torch.Tensor:
        step = global_step % self.num_inference_steps
        offset = math.ceil(max(global_step - self.num_inference_steps + 1, 0) / self.steps_per_window) % self.window_size
        return torch.cat([self.table[offset:, step], self.table[:offset, step]])


class RollingStepper:
    """Per-frame Euler step ``x + v * (sigma[i + 1] - sigma[i])`` in fp32, ``i`` one index per latent frame.

    Shaped like a diffusers scheduler's ``step(model_output, timestep, sample, return_dict=False)`` so it can be the
    ``per_request_scheduler`` of `CFGParallelMixin.scheduler_step_maybe_with_cfg`; ``timestep`` is the per-frame step
    index (``[T]``), and the stepper keeps no counter: the caller owns the index.
    """

    def __init__(self, sigmas: torch.Tensor):
        self.sigmas = sigmas.to(torch.float32)

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, return_dict: bool = False,
             generator: Optional[torch.Generator] = None):
        step_index = timestep.view(1, 1, -1, 1, 1)
        sample = sample.to(torch.float32)
        sigmas = self.sigmas.to(sample.device)
        flat = step_index.reshape(-1).long()
        sigma = sigmas[flat].reshape(step_index.shape)
        sigma_next = sigmas[flat + 1].reshape(step_index.shape)
        prev = sample + model_output.float() * (sigma_next - sigma).to(sample.device)
        return (prev,)


class AnnealCFG:
    """The per-frame CFG scale, linear in the timestep: ``cfg_min + (cfg_max - cfg_min) * t / num_train_timesteps``."""

    def __init__(self, num_train_timesteps: int = 1000, cfg_min: float = 1.0, cfg_max: float = 4.0):
        self.num_train_timesteps = num_train_timesteps
        self.cfg_min = cfg_min
        self.cfg_max = cfg_max

    def get_cfg(self, timesteps: torch.Tensor) -> torch.Tensor:
        t = timesteps.float()
        ratio = (self.num_train_timesteps - t) / self.num_train_timesteps
        ratio = 1.0 - ratio
        return self.cfg_min + (self.cfg_max - self.cfg_min) * ratio


def apply_context_noise(latents: torch.Tensor, sigma: float, generator: Optional[torch.Generator]) -> torch.Tensor:
    """Context noise of inference: ``(1 - sigma) * clean + sigma * eps``, every frame with the same ``sigma``.

    ``eps`` is drawn on the generator's device (a CPU generator draws on the CPU), which keeps the random stream the
    one MovieRoll's own `apply_context_noise` consumes.
    """
    device = generator.device if generator is not None else latents.device
    noise = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=generator).to(latents.device)
    return ((1 - sigma) * latents + sigma * noise).contiguous()
