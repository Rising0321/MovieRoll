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
#
# ==============================================================================
# Modified from diffusers
# ==============================================================================
# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discrete flow-matching scheduler.

Holds the shifted timestep ladder, `set_custom_timesteps` for the distilled ladder and `velocity_to_x0`
for the DMD losses. The tensor-aware `step` the rolling window needs lives on the subclass
`FlowMatchStreamingScheduler` in `scheduling_flow_match_streaming.py`.
"""

from typing import List, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """Euler scheduler over a `shift`-warped sigma ladder."""

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

        self.sigmas = sigmas
        # the value fed to model
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)

    def set_custom_timesteps(self, timesteps: List[int], device: Union[str, torch.device] = None):
        """Set custom discrete timesteps for DMD-style inference.

        `timesteps` must be descending (e.g. [1000, 750, 500, 250]) and may or may not end with 0.
        """
        # Remove trailing 0 if present (will be added back)
        if timesteps[-1] == 0:
            timesteps = timesteps[:-1]

        self.num_inference_steps = len(timesteps)

        timesteps_tensor = torch.tensor(timesteps, dtype=torch.float32)

        sigmas = timesteps_tensor / self.config.num_train_timesteps

        # The shift must match set_timesteps and the shift used during training.
        if self.config.shift != 1.:
            sigmas = self.sd3_time_shift(sigmas)

        # sigma=0 at the end represents the clean sample.
        sigmas = torch.cat([sigmas, torch.zeros(1)])

        # The timesteps derive from the SHIFTED sigmas, as in set_timesteps.
        shifted_timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32)
        self.timesteps = shifted_timesteps.to(device=device) if device else shifted_timesteps
        self.sigmas = sigmas.to(device=device) if device else sigmas

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """Set the uniform discrete timesteps used for the diffusion chain; run before inference."""
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)

        if self.config.shift != 1.:
            sigmas = self.sd3_time_shift(sigmas)

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)

    def sd3_time_shift(self, t: torch.Tensor):
        return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)

    def velocity_to_x0(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Convert a velocity prediction to the clean sample: ``x0 = x_t - sigma * velocity``.

        The parameterisation is ``x_t = (1 - sigma) * x0 + sigma * noise`` with
        ``velocity = noise - x0``, which is why sigma is ``timestep / num_train_timesteps`` and no
        ``1 / (1 - sigma)`` factor appears.

        The branch is chosen by the SHAPE of `timestep`, never by its values: a [B, T] matching the
        sample's batch and frame count gives a per-frame sigma [B, 1, T, 1, 1]; anything else collapses to
        one sigma per sample, trailing-unsqueezed to the sample's rank.
        """
        # The incoming timestep is already shifted, so sigma = timestep / num_train_timesteps stays
        # consistent with the forward process; it is NOT shifted again here.
        num_frames = sample.shape[2] if sample.dim() == 5 else None
        if (
            timestep.dim() == 2
            and num_frames is not None
            and timestep.shape[1] == num_frames
            and timestep.shape[0] == sample.shape[0]
        ):
            # Frames of a rolling window sit at different noise levels, so x0 is solved per frame;
            # collapsing onto timestep[:, 0] would apply the first frame's sigma to all of them.
            sigma = timestep.float() / self.config.num_train_timesteps  # [B, T]
            sigma = sigma.to(device=sample.device)
            sigma = sigma.view(timestep.shape[0], 1, num_frames, 1, 1)
        else:
            if timestep.dim() == 2:
                timestep = timestep[:, 0]
            sigma = timestep.float() / self.config.num_train_timesteps
            sigma = sigma.to(device=sample.device)
            while len(sigma.shape) < len(sample.shape):
                sigma = sigma.unsqueeze(-1)

        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        pred_x0 = sample - sigma * model_output

        return pred_x0
