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

"""The image-to-video condition: one input image to the anchor latent the rollout starts from.

Shared by inference (`movieroll/generate.py`) and the Stage-3 ACS samples
(`movieroll/trainer/dmd/sampling.py`) so both condition on a bit-identical anchor.
"""

import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from torchvision import transforms

from movieroll.commons.parallel_states import get_sequence_parallel_state, nccl_info
from movieroll.utils.torch_utils import PRECISION_TO_TYPE


def open_input_image(image_path):
    """The input image as RGB, rotated upright by its EXIF orientation (a phone photo is stored sideways)."""
    with Image.open(image_path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def encode_input_image(vae, vae_precision, device, image_path, height, width):
    """VAE-encode one input image into the 1-frame i2v condition, shape [1, C, 1, height / 16, width / 16].

    The image is scaled to cover height x width (LANCZOS), centre-cropped and mapped to [-1, 1]; the
    posterior mode is taken (no random draw) and scaled like the training latents. Under sequence
    parallelism the latent of sp rank 0 is broadcast so the whole group conditions on one tensor.
    """
    if not isinstance(device, torch.device):
        device = torch.device("cuda", int(device))
    image = open_input_image(image_path)
    scale = max(width / image.width, height / image.height)
    transform = transforms.Compose([
        transforms.Resize((int(round(image.height * scale)), int(round(image.width * scale))),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    pixels = transform(image).unsqueeze(0).unsqueeze(2).to(device)
    vae_dtype = PRECISION_TO_TYPE[str(vae_precision)]
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=vae_dtype,
        enabled=device.type == "cuda" and vae_dtype != torch.float32,
    ):
        latents = vae.encode(pixels).latent_dist.mode()
    shift_factor = getattr(vae.config, "shift_factor", None)
    if shift_factor is not None:
        latents = latents - float(shift_factor)
    latents = latents * float(vae.config.scaling_factor)
    if get_sequence_parallel_state():
        dist.broadcast(latents, group_src=0, group=nccl_info.sp_group)
    return latents
