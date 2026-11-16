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

"""Loader of the 16164 causal VAE (public ``config.json`` + ``diffusion_pytorch_model.safetensors`` layout) and the
diagonal Gaussian posterior shared by both VAE classes."""

import os
from typing import Optional

import torch
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file

from movieroll.utils.torch_utils import PRECISION_TO_TYPE

# The temporal compression ratio the frame arithmetic is written for. `(video_length - 1) // 4 + 1` and its
# inverse are spelled out as literal 4s across the entry points, the config validators and the pipelines, so
# the number cannot be read off the loaded VAE -- which is why the loader refuses a VAE that disagrees.
_ASSUMED_FFACTOR_TEMPORAL = 4


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor):
        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1  # (B, C, T, H ,W) / (B, C, H, W)
        else:
            raise NotImplementedError
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def mode(self) -> torch.Tensor:
        return self.mean


def load_vae(
    vae_precision: str = None,
    vae_path: str = "ckpts/vae",
    logger=None,
    device=None,
    streaming: bool = False,
):
    """Load the 16164 causal VAE from a directory with ``config.json`` and ``diffusion_pytorch_model.safetensors``.

    ``streaming=True`` builds the T-cache streaming class (inference and DMD), ``streaming=False`` the
    non-streaming one (SFT and ODE); both have identical ``state_dict`` keys and identical
    ``streaming_mode=False`` numerics. Returns
    ``(vae, vae_path, spatial_compression_ratio, time_compression_ratio)``.
    """
    config_path = os.path.join(vae_path, "config.json")
    vae_ckpt = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")
    for path in (config_path, vae_ckpt):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"VAE file not found: {path}")

    if logger is not None:
        logger.info(f"Loading 3D VAE model from: {vae_path}")

    # Imported here: both VAE modules import DiagonalGaussianDistribution from this module.
    if streaming:
        from movieroll.models.autoencoders.streaming_vae import AutoencoderKLConv3D as vae_cls
    else:
        from movieroll.models.autoencoders.causal_vae import AutoencoderKLConv3D as vae_cls

    config = vae_cls.load_config(vae_path)
    vae = vae_cls.from_config(config)

    ckpt = load_file(vae_ckpt, device=str(vae.device))
    vae.load_state_dict(ckpt, strict=True)

    spatial_compression_ratio = vae.config.ffactor_spatial
    time_compression_ratio = vae.config.ffactor_temporal

    # The temporal compression ratio is a weight-shape contract, and unlike the spatial one it is never read
    # back: the entry points spell the call `vae, _, _, _ = load_vae(...)` and the frame arithmetic hard-codes
    # the 4. A VAE with a different temporal factor -- which the ordinary `--vae-path` flag invites -- makes
    # every frame count silently wrong, so it is refused here, at the one place that has the number.
    if int(time_compression_ratio) != _ASSUMED_FFACTOR_TEMPORAL:
        raise ValueError(
            f"this VAE compresses time by {time_compression_ratio}, but the frame arithmetic is written "
            f"for {_ASSUMED_FFACTOR_TEMPORAL} (`latent_frames = (video_length - 1) // "
            f"{_ASSUMED_FFACTOR_TEMPORAL} + 1`, hard-coded in the entry points, the validators and the "
            f"pipelines). Every frame count would be wrong without anything failing, so this is refused "
            f"rather than warned about. Use a VAE with ffactor_temporal="
            f"{_ASSUMED_FFACTOR_TEMPORAL}: {vae_path}"
        )
    # The spatial factor IS read back, but from two different places: the pipelines derive it as
    # `2 ** (len(block_out_channels) - 1)` while `trainer/model_inputs.py` reads `config.ffactor_spatial`.
    # Only one direction of a mismatch is reachable: an `ffactor_spatial` LARGER than the bins imply already
    # trips this class's own `assert i_level < len(block_out_channels) - 1`, while a smaller one builds fine
    # and then the pipelines size latents by one rule and `trainer/model_inputs.py` tiles by the other.
    derived_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    if int(spatial_compression_ratio) != derived_spatial:
        raise ValueError(
            f"this VAE's ffactor_spatial ({spatial_compression_ratio}) disagrees with the "
            f"{derived_spatial} implied by its {len(vae.config.block_out_channels)} "
            f"`block_out_channels`. The pipelines size latents by the second and the training input "
            f"pipeline tiles by the first, so the two must match: {vae_path}"
        )

    if vae_precision is not None:
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    vae.requires_grad_(False)

    if logger is not None:
        logger.info(f"VAE to dtype: {vae.dtype}")

    if device is not None:
        vae = vae.to(device)

    vae.eval()

    return vae, vae_path, spatial_compression_ratio, time_compression_ratio
