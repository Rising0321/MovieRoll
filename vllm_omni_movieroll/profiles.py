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

"""The fixed numbers of MovieRoll's two released chains, and the model-directory settings that pick one.

These are the values the released checkpoints were evaluated with (the same numbers as MovieRoll's own
`commons/profiles.py`). A model directory holds one checkpoint, so it fixes the chain: ``_class_name`` in its
`model_index.json` is ``MovieRollPipeline`` (Base: 50 steps with classifier-free guidance) or ``MovieRollDistilledPipeline``
(Distilled: 5 steps, guidance baked into the weights), and ``sampling_mode`` says the same in MovieRoll's own words. The
directory's ``movieroll_overrides`` hold what its MovieRoll model folder sets differently from the built-in chain (the
image-to-video folder's context noise, for one; `read_model_folder`), and a deploy YAML may override the same short,
validated list through ``model_config``; requests cannot.

Pure Python: no framework import, so request validation can run anywhere.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

NUM_TRAIN_TIMESTEPS = 1000
VAE_SPATIAL = 16
VAE_TEMPORAL = 4

# MovieRoll-16164-MMDiT, the backbone of both checkpoints (HunyuanVideo-1.5's MMDiT sizes)
ARCH = {
    "in_channels": 32,
    "out_channels": 32,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 54,
    "num_refiner_layers": 2,
    "mlp_ratio": 4.0,
    "patch_size": 1,
    "patch_size_t": 1,
    "text_embed_dim": 3584,
    "text_embed_2_dim": 1472,
    "rope_theta": 256.0,
    "rope_axes_dim": [16, 56, 56],
}

COMMON = {
    "height": 480,
    "width": 864,
    "video_fps": 24,
    "text_len": 1000,
    "byt5_max_length": 256,
    "chunk_size": 4,
    "window_size": 5,
    "max_num_latent_frames": 120,
    "full_history_block_num": 6,
    "kv_recompute_enabled": True,
    "kv_recompute_threshold": 18,
    "kv_recompute_recent_k": 3,
    "vae_stream_decode_max_chunk_size": 4,
}

BASE = {
    "num_inference_steps": 50,
    "flow_shift": 5.0,
    # classifier-free guidance on (two branches per window); its per-frame scale is AnnealCFG's (cfg_max -> cfg_min)
    "use_cfg": True,
    "cfg_min": 1.0,
    "cfg_max": 4.0,
    "negative_prompt": None,
    "context_noise_enabled": True,
    "inference_context_noise": 0.05,
}

DISTILLED = {
    "denoising_step_list": [1000, 800, 600, 400, 200],
    "timestep_shift": 5.0,
    "context_noise": 0.15,
}

SAMPLING_MODES = ("base", "distilled")
# the `model_index.json` class of each chain; vLLM-Omni resolves a model's defaults by this name (`video_defaults`)
PIPELINE_CLASS = {"base": "MovieRollPipeline", "distilled": "MovieRollDistilledPipeline"}

# A MovieRoll model folder, what `movieroll/generate.py --model` takes: the weights and the sampling settings of one
# checkpoint. Its config.json also names the architecture and the prompt templates, which this package has built in.
MODEL_FOLDER_CONFIG = "config.json"
MODEL_FOLDER_WEIGHTS = "model.safetensors"
FOLDER_FIXED = {
    "model_type": "MovieRoll-16164-MMDiT",
    "prompt_template": "li-dit-encode-image-json",
    "prompt_template_video": "li-dit-encode-video-json",
}

# what a model directory or a deploy YAML may change (``movieroll_overrides``); everything else is the checkpoint's.
# The step count is not here: it is a request setting (``num_inference_steps``), filled in for every request.
OVERRIDABLE = frozenset({
    "kv_recompute_enabled", "kv_recompute_threshold", "kv_recompute_recent_k", "flow_shift", "use_cfg", "cfg_min",
    "cfg_max", "context_noise_enabled", "inference_context_noise", "denoising_step_list", "timestep_shift",
    "context_noise", "vae_stream_decode_max_chunk_size",
})


def get_profile(sampling_mode: str) -> dict:
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"unknown sampling_mode {sampling_mode!r}, expected one of {SAMPLING_MODES}")
    profile = dict(COMMON)
    profile.update(BASE if sampling_mode == "base" else DISTILLED)
    return profile


@dataclass(frozen=True)
class MovieRollSettings:
    """The chain a model directory runs and its numbers."""

    sampling_mode: str
    profile: dict

    @property
    def distilled(self) -> bool:
        return self.sampling_mode == "distilled"

    @property
    def use_cfg(self) -> bool:
        """Classifier-free guidance on: Base unless its model directory turns it off; Distilled never."""
        return bool(self.profile.get("use_cfg", False))

    @property
    def chunk_size(self) -> int:
        return int(self.profile["chunk_size"])

    @property
    def window_size(self) -> int:
        return int(self.profile["window_size"])

    @property
    def context_noise_sigma(self) -> float:
        """Sigma of the context noise put on the clean history before it enters the KV cache (0: none)."""
        p = self.profile
        if self.distilled:
            return float(p["context_noise"])
        return float(p["inference_context_noise"]) if p["context_noise_enabled"] else 0.0


def _apply_overrides(profile: dict, overrides: Optional[Mapping[str, Any]], source: str) -> None:
    if not overrides:
        return
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{source} must be a mapping, got {type(overrides).__name__}")
    unknown = set(overrides) - OVERRIDABLE
    if unknown:
        raise ValueError(f"{source}: unknown MovieRoll overrides {sorted(unknown)}; allowed: {sorted(OVERRIDABLE)}")
    profile.update(overrides)


def load_settings(model_index: Mapping[str, Any], model_config: Optional[Mapping[str, Any]] = None) -> MovieRollSettings:
    """Settings from a model directory's `model_index.json` (+ the deploy YAML's ``model_config``)."""
    mode = model_index.get("sampling_mode")
    if mode not in SAMPLING_MODES:
        raise ValueError(
            f"model_index.json must name the chain: sampling_mode in {SAMPLING_MODES}, got {mode!r} (build the directory "
            "with examples/offline_inference/movieroll/download_movieroll.py)"
        )
    class_name = model_index.get("_class_name")
    if class_name is not None and class_name != PIPELINE_CLASS[mode]:
        raise ValueError(
            f"model_index.json names the class {class_name!r} for the {mode} chain, whose class is {PIPELINE_CLASS[mode]!r}; "
            "rebuild the directory with examples/offline_inference/movieroll/download_movieroll.py"
        )
    profile = get_profile(mode)
    _apply_overrides(profile, model_index.get("movieroll_overrides"), "model_index.json movieroll_overrides")
    _apply_overrides(profile, (model_config or {}).get("movieroll_overrides"), "model_config.movieroll_overrides")
    if mode == "distilled" and "use_cfg" in profile:
        raise ValueError("use_cfg is a base-chain setting; the distilled chain has no classifier-free guidance (it is "
                         "distilled into the weights)")
    if not isinstance(profile.get("use_cfg", False), bool):
        raise ValueError(f"use_cfg must be true or false, got {profile['use_cfg']!r}")
    if mode == "distilled" and len(profile["denoising_step_list"]) != int(profile["window_size"]):
        raise ValueError(
            f"the distilled chain needs window_size == len(denoising_step_list), got {profile['window_size']} and "
            f"{len(profile['denoising_step_list'])}"
        )
    return MovieRollSettings(sampling_mode=mode, profile=profile)


def read_model_folder(folder: str) -> tuple[str, dict]:
    """The chain and the overrides of a MovieRoll model folder (``config.json`` + ``model.safetensors``).

    The folder is read by the rules of native MovieRoll's `load_model_dir`: ``sampling_mode`` names the chain, any other
    key must be a setting of that chain, and a key left out keeps the built-in value. Returned are the values that
    differ from the built-in chain; each must be one this package can honour (`OVERRIDABLE`), so a folder that changes
    the geometry or the text conditioning is refused here rather than run with the released values.
    """
    config_path = os.path.join(folder, MODEL_FOLDER_CONFIG)
    for path in (config_path, os.path.join(folder, MODEL_FOLDER_WEIGHTS)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{folder} is not a MovieRoll model folder: it has no {os.path.basename(path)}")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    mode = config.get("sampling_mode") if isinstance(config, dict) else None
    if mode not in SAMPLING_MODES:
        raise ValueError(f"{config_path}: sampling_mode must be one of {SAMPLING_MODES}, got {mode!r}")
    released = {**get_profile(mode), **FOLDER_FIXED}
    unknown = sorted(set(config) - set(released) - {"sampling_mode"})
    if unknown:
        raise ValueError(f"{config_path}: unknown keys {unknown}")
    overrides = {key: value for key, value in config.items() if key != "sampling_mode" and value != released[key]}
    fixed = sorted(set(overrides) - OVERRIDABLE)
    if fixed:
        hint = "; the step count is sent with each request instead" if "num_inference_steps" in fixed else ""
        raise ValueError(
            f"{config_path}: {fixed} differ from the released {mode} chain, and this package runs that chain's "
            f"values for them; only {sorted(OVERRIDABLE)} can differ{hint}"
        )
    load_settings({"sampling_mode": mode, "movieroll_overrides": overrides})   # the same checks as at load time
    return mode, overrides


def transformer_config(full_history_block_num: int = COMMON["full_history_block_num"]) -> dict:
    """`transformer/config.json` of a model directory: the architecture and its HieraKV split."""
    return {
        "_class_name": "MovieRollTransformer3DModel",
        **{k: v for k, v in ARCH.items()},
        "max_num_latent_frames": COMMON["max_num_latent_frames"],
        "full_history_block_num": full_history_block_num,
        "sink_chunks": 1,
    }
