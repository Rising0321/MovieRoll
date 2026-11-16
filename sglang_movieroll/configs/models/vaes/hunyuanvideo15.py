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

"""Config of the HunyuanVideo-1.5 VAE in SGLang's format.

The defaults are the released VAE's ``config.json`` (16x spatial / 4x temporal, 32 latent channels); the model
directory's ``vae/config.json`` overrides them through SGLang's ``update_model_arch``. Tiling is off: MovieRoll
decodes in causal temporal chunks, which is exact, where a spatial tiling would blend overlapping tiles and
change the video.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class HunyuanVideo15VAEArchConfig(VAEArchConfig):
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 32
    block_out_channels: tuple = (128, 256, 512, 1024, 1024)
    layers_per_block: int = 2
    spatial_compression_ratio: int = 16
    temporal_compression_ratio: int = 4
    downsample_match_channel: bool = True
    upsample_match_channel: bool = True
    scaling_factor: float = 1.03682

    # latent frames per streaming decode step (the released chains' `vae_stream_decode_max_chunk_size`), and the
    # exact-attention budget of the bottleneck attention (None keeps the whole history, which is exact)
    stream_chunk_frames: int = 4
    max_attention_frames: int | None = None


@dataclass
class HunyuanVideo15VAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=HunyuanVideo15VAEArchConfig)

    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    use_temporal_scaling_frames: bool = True

    def get_vae_scale_factor(self):
        return self.arch_config.spatial_compression_ratio

    def encode_sample_mode(self):
        # the image-to-video anchor is the posterior mode, not a draw (`generate.py`)
        return "argmax"
