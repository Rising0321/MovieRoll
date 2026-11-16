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

"""Context noise on the clean history: noised_t = (1 - sigma_t) * clean_t + sigma_t * eps_t.

Training re-noises the clean history per chunk with a random sigma; inference re-noises the chunk that is about to
enter the KV cache with one fixed sigma. Both go through `apply_context_noise`; the difference is only how sigma is
drawn.

The generator argument is honoured where it is passed. SFT training passes its SP-source generator; the Base
pipeline and the DMD / Distilled rolling pipeline pass the
sample's noise generator, so an inference sample is reproducible from its own seed. With ``generator=None`` the
draw comes from the global RNG of the latents' device.
"""

from typing import List, Optional

import torch
from torch import Tensor


def _random_device(generator: Optional[torch.Generator], fallback) -> torch.device:
    return torch.device(generator.device) if generator is not None else torch.device(fallback)


def _normalize_sigma_range(sigma_min, sigma_max):
    """Make [min, max] a valid interval (min <= max). Both bounds are floats."""
    return min(sigma_min, sigma_max), sigma_max


def sample_context_noise_sigma(
    model_input: Tensor,
    batch_size: int,
    sigma_max: float,
    sigma_min: float = 0.0,
    *,
    chunk_sizes: List[int],
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Draw one sigma per chunk and expand it over that chunk's frames.

    `chunk_sizes` is required: the only caller that gets here is the training-time context noise, which always
    knows its chunk layout. Inference noises the whole history with a single sigma and never
    reaches this function.

    Returns:
        noise_sigma: already expanded to model_input's ndim, ready to broadcast.
    """
    sigma_min, sigma_max = _normalize_sigma_range(sigma_min, sigma_max)

    num_chunks = len(chunk_sizes)
    chunk_sizes_t = torch.tensor(chunk_sizes, device=model_input.device, dtype=torch.long)
    random_values = torch.rand(
        size=(batch_size, 1, num_chunks),
        device=_random_device(generator, model_input.device),
        dtype=model_input.dtype,
        generator=generator,
    ).to(model_input.device)
    noise_sigma = (
        random_values.repeat_interleave(chunk_sizes_t, dim=-1)
        * (sigma_max - sigma_min) + sigma_min
    )

    while noise_sigma.ndim < model_input.ndim:
        noise_sigma = noise_sigma.unsqueeze(-1)
    return noise_sigma


def apply_context_noise(
    latents: Tensor,
    sigma_max: float = 0.1,
    sigma_min: float = 0.0,
    force_equal_sigma: bool = False,
    chunk_sizes: List[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Context noise: noised_t = (1 - sigma_t) * clean_t + sigma_t * eps_t.

    How sigma is drawn:
      - force_equal_sigma=True : `sigma_max` is the one sigma of every frame (inference);
      - otherwise              : frames of a chunk share a sigma, chunks draw independently (training), which
                                 is why `chunk_sizes` is required on that path.
    """
    if force_equal_sigma:
        noise_sigma = sigma_max
    else:
        noise_sigma = sample_context_noise_sigma(
            model_input=latents,
            batch_size=latents.shape[0],
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            chunk_sizes=chunk_sizes,
            generator=generator,
        )

    noise = torch.randn(
        latents.shape,
        device=_random_device(generator, latents.device),
        dtype=latents.dtype,
        generator=generator,
    ).to(latents.device)
    return ((1 - noise_sigma) * latents + noise_sigma * noise).contiguous()
