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

"""Fixed sampling parameters of the two released inference chains.

These are not defaults to tune: they are the exact values the released checkpoints were evaluated with, so they
live here rather than on the command line. Only the knobs a user legitimately has to change (prompt source and
input images, frame count, resolution, output location, parallel topology, precision) are exposed by
`movieroll/generate.py`.

`COMMON` holds what both chains share -- the streaming geometry the checkpoints were trained with (chunk 4 /
window 5 / 6 full-history blocks / a 120-latent-frame RoPE budget) and the KV recompute policy. `BASE` is the
50-step CFG chain; `DISTILLED` is the few-step DMD chain, whose guidance is baked into the weights.

A released model directory ships these values in its `config.json` next to `model.safetensors`
(`load_model_dir`, read by `generate.py --model`); the files in `movieroll/configs/models/` are those configs.
"""

import json
import os

from movieroll.commons.constants import MODEL_TYPE

# Default resolution of both released chains (`generate.py --height / --width` override it).
_HEIGHT = 480
_WIDTH = 864

# An i2v run of a bare `--dit-weight` checkpoint leaves the resolution to the input image unless the command line
# sets it (`generate.py`): the image's aspect ratio picks a bucket. A model folder runs at its config's size.
# Every candidate holds about `base_size ** 2 / patch_size ** 2` latent patches, so the sizes trade height against
# width at a roughly constant token count.
I2V_BUCKET_BASE_SIZE = 640
# One latent patch per 16 pixels, the VAE's spatial downsample. Every bucket side is a multiple of it, so
# `_check_resolution`'s "positive multiple of 16" holds for a bucket by construction.
I2V_BUCKET_PATCH_SIZE = 16
# Widest and tallest bucket allowed, as a side ratio.
I2V_BUCKET_MAX_RATIO = 4.0

COMMON = {
    "model_type": MODEL_TYPE,
    "height": _HEIGHT,
    "width": _WIDTH,
    "video_fps": 24,
    # text conditioning
    "text_len": 1000,
    "byt5_max_length": 256,
    "prompt_template": "li-dit-encode-image-json",
    "prompt_template_video": "li-dit-encode-video-json",
    # streaming geometry (must match the checkpoint; see the transformer's RoPE resampler capacity)
    "chunk_size": 4,
    "window_size": 5,
    "max_num_latent_frames": 120,
    "full_history_block_num": 6,
    # KV recompute policy
    "kv_recompute_enabled": True,
    "kv_recompute_threshold": 18,
    "kv_recompute_recent_k": 3,
    # VAE streaming decode chunk, in latent frames. The two chains read it with OPPOSITE "decode in one piece"
    # sentinels: distilled (generate.py's own decode, via int()) takes <= 0 and cannot take
    # None, base (the pipeline __call__) takes None and otherwise needs a positive int. generate.parse_args
    # refuses a value the selected chain cannot use before any weight is loaded.
    "vae_stream_decode_max_chunk_size": 4,
}

BASE = {
    "num_inference_steps": 50,
    "flow_shift": 5.0,
    # classifier-free guidance on; its per-frame scale comes from AnnealCFG (cfg_max -> cfg_min)
    "use_cfg": True,
    "cfg_min": 1.0,
    "cfg_max": 4.0,
    "negative_prompt": None,
    # context noise on the clean history (its sigma: inference_context_noise)
    "context_noise_enabled": True,
    "inference_context_noise": 0.05,
}

DISTILLED = {
    "denoising_step_list": [1000, 800, 600, 400, 200],
    "timestep_shift": 5.0,
    # the distilled chain's context noise on the clean KV context
    "context_noise": 0.15,
}

SAMPLING_MODES = ("base", "distilled")


def get_profile(sampling_mode: str) -> dict:
    """Return COMMON merged with the chain-specific block for `sampling_mode`."""
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(
            f"unknown sampling_mode {sampling_mode!r}, expected one of {SAMPLING_MODES}"
        )
    profile = dict(COMMON)
    profile.update(BASE if sampling_mode == "base" else DISTILLED)
    return profile


# A released model directory: the transformer weights plus the settings they are sampled with.
MODEL_CONFIG_FILE = "config.json"
MODEL_WEIGHT_FILE = "model.safetensors"


def load_model_dir(model_dir: str) -> dict:
    """Read the `config.json` of a model directory and return its full sampling settings.

    The file holds `sampling_mode` and any `get_profile` values for that chain; a value it leaves out keeps the
    built-in profile's. An unknown key, a `model_type` this code cannot build, or a missing `model.safetensors`
    is an error here, before any weight is loaded.
    """
    config_path = os.path.join(model_dir, MODEL_CONFIG_FILE)
    weight_path = os.path.join(model_dir, MODEL_WEIGHT_FILE)
    for path in (config_path, weight_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"model directory {model_dir!r} has no {os.path.basename(path)}")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or config.get("sampling_mode") not in SAMPLING_MODES:
        raise ValueError(f"{config_path}: `sampling_mode` must be one of {SAMPLING_MODES}")
    settings = get_profile(config["sampling_mode"])
    unknown = sorted(set(config) - set(settings) - {"sampling_mode"})
    if unknown:
        raise ValueError(f"{config_path}: unknown keys {unknown}")
    if config.get("model_type", MODEL_TYPE) != MODEL_TYPE:
        raise ValueError(f"{config_path}: model_type {config['model_type']!r} is not {MODEL_TYPE!r}")
    settings.update(config)
    return settings
