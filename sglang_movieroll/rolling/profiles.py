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

"""The two released sampling chains as plain data (ported from `movieroll.commons.profiles`).

These are not defaults to tune: they are the values the released checkpoints were evaluated with. A model
directory holds one checkpoint, so it fixes the chain: Base (50 steps with classifier-free guidance, the
per-frame AnnealCFG schedule) or Distilled (a five-step ladder whose guidance is baked into the weights).
Every number matches `movieroll.commons.profiles.get_profile`.

A released MovieRoll model folder (what `movieroll/generate.py --model` takes) may set some of them differently
-- the image-to-video folder turns the context noise off. `read_model_folder` reads such a folder, and the
model directory built from it keeps the differences (``movieroll_overrides`` in its ``model_index.json``), which
`Chain.of` applies on top of the built-in chain.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

SAMPLING_MODES = ("base", "distilled")

# Streaming geometry and KV recompute policy, shared by both chains.
COMMON: dict[str, Any] = {
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

BASE: dict[str, Any] = {
    "num_inference_steps": 50,
    "flow_shift": 5.0,
    # classifier-free guidance on (two branches per window); its per-frame scale is AnnealCFG's (cfg_max -> cfg_min)
    "use_cfg": True,
    "cfg_min": 1.0,
    "cfg_max": 4.0,
    "context_noise_enabled": True,
    "inference_context_noise": 0.05,
}

DISTILLED: dict[str, Any] = {
    "denoising_step_list": [1000, 800, 600, 400, 200],
    "timestep_shift": 5.0,
    "context_noise": 0.15,
}

NUM_TRAIN_TIMESTEPS = 1000

# The settings a model directory may set differently from the built-in chain: the ones only MovieRoll's stages read,
# from the chain on the pipeline config, once the directory is loaded (the context noise, the KV recompute policy,
# whether classifier-free guidance runs and its AnnealCFG range). The request defaults (steps), the geometry and the
# scheduler are fixed when SGLang resolves the pipeline config, before the directory is read.
OVERRIDABLE = frozenset(
    {
        "context_noise",
        "context_noise_enabled",
        "inference_context_noise",
        "kv_recompute_enabled",
        "kv_recompute_threshold",
        "kv_recompute_recent_k",
        "use_cfg",
        "cfg_min",
        "cfg_max",
    }
)

# A MovieRoll model folder: the weights and the sampling settings of one checkpoint. Its config.json also names
# the architecture and the prompt templates, which this package has built in.
MODEL_FOLDER_CONFIG = "config.json"
MODEL_FOLDER_WEIGHTS = "model.safetensors"
FOLDER_FIXED: dict[str, Any] = {
    "model_type": "MovieRoll-16164-MMDiT",
    "prompt_template": "li-dit-encode-image-json",
    "prompt_template_video": "li-dit-encode-video-json",
    "negative_prompt": None,
}


def get_profile(sampling_mode: str) -> dict[str, Any]:
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(
            f"unknown sampling_mode {sampling_mode!r}, expected one of {SAMPLING_MODES}"
        )
    profile = dict(COMMON)
    profile.update(BASE if sampling_mode == "base" else DISTILLED)
    return profile


@dataclass(frozen=True)
class RollingProfile:
    """What a chain fixes for the sampling loop: the window geometry, the recompute policy, the differences."""

    mode: str  # "base" or "distilled"
    chunk_size: int = 4
    window_size: int = 5
    steps_per_window: int = 1
    kv_recompute_enabled: bool = True
    kv_recompute_threshold: int = 18
    kv_recompute_recent_k: int = 3
    guided: bool = False  # classifier-free guidance with the AnnealCFG schedule
    cfg_min: float = 1.0
    cfg_max: float = 4.0
    force_recompute_at_event: bool = (
        False  # Distilled: recompute at an event switch even if nothing is dropped
    )
    noise_anchor: bool = (
        False  # Distilled: the image-to-video anchor goes through the context noise too
    )

    def __post_init__(self):
        if self.mode not in SAMPLING_MODES:
            raise ValueError(f"mode must be one of {SAMPLING_MODES}, got {self.mode!r}")
        if self.guided and self.mode != "base":
            raise ValueError("only the base chain is guided")

    @property
    def num_inference_steps(self) -> int:
        return self.window_size * self.steps_per_window


def check_overrides(
    overrides: Optional[Mapping[str, Any]], source: str
) -> dict[str, Any]:
    if not overrides:
        return {}
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{source} must be a mapping, got {type(overrides).__name__}")
    unknown = sorted(set(overrides) - OVERRIDABLE)
    if unknown:
        raise ValueError(
            f"{source}: {unknown} cannot be changed for MovieRoll under SGLang; only {sorted(OVERRIDABLE)} can"
        )
    return dict(overrides)


@dataclass(frozen=True)
class Chain:
    """One released chain: its profile, its request defaults and the rolling profile it selects."""

    sampling_mode: str
    profile: dict[str, Any]

    @classmethod
    def of(
        cls, sampling_mode: str, overrides: Optional[Mapping[str, Any]] = None
    ) -> "Chain":
        """The chain with a model directory's ``movieroll_overrides`` applied (`OVERRIDABLE` only)."""
        profile = get_profile(sampling_mode)
        profile.update(check_overrides(overrides, "movieroll_overrides"))
        if sampling_mode == "distilled" and "use_cfg" in profile:
            raise ValueError(
                "movieroll_overrides: use_cfg is a base-chain setting; the distilled chain has no "
                "classifier-free guidance (it is distilled into the weights)"
            )
        if not isinstance(profile.get("use_cfg", False), bool):
            raise ValueError(
                f"movieroll_overrides: use_cfg must be true or false, got {profile['use_cfg']!r}"
            )
        return cls(sampling_mode, profile)

    @property
    def chunk_size(self) -> int:
        return int(self.profile["chunk_size"])

    @property
    def window_size(self) -> int:
        return int(self.profile["window_size"])

    @property
    def distilled(self) -> bool:
        return self.sampling_mode == "distilled"

    @property
    def default_steps(self) -> int:
        """Denoising steps: the length of the ladder (Distilled) or the profile's step count (Base)."""
        return (
            len(self.profile["denoising_step_list"])
            if self.distilled
            else int(self.profile["num_inference_steps"])
        )

    @property
    def use_cfg(self) -> bool:
        """Classifier-free guidance on: Base unless its model directory turns it off; Distilled never."""
        return bool(self.profile.get("use_cfg", False))

    @property
    def shift(self) -> float:
        """Sigma-ladder shift of the chain (the two chains spell it with different profile keys)."""
        return float(
            self.profile["timestep_shift"]
            if self.distilled
            else self.profile["flow_shift"]
        )

    @property
    def context_noise_sigma(self) -> float:
        """Sigma of the context noise put on a chunk before it enters the cache (0 turns it off)."""
        if self.distilled:
            return float(self.profile["context_noise"])
        return (
            float(self.profile["inference_context_noise"])
            if self.profile["context_noise_enabled"]
            else 0.0
        )

    def check_steps(self, steps: Optional[int]) -> int:
        """The step count of a request: the Distilled ladder is fixed, Base takes a multiple of the window size."""
        if steps is None or steps == self.default_steps:
            return self.default_steps
        if self.distilled:
            raise ValueError(
                f"the distilled chain has a fixed {self.default_steps}-step ladder "
                f"{self.profile['denoising_step_list']}; num_inference_steps={steps} is not supported"
            )
        if steps <= 0 or steps % self.window_size:
            raise ValueError(
                f"num_inference_steps ({steps}) must be a positive multiple of window_size ({self.window_size})"
            )
        return steps

    def overrides(self) -> dict[str, Any]:
        """What this chain sets differently from the built-in one."""
        released = get_profile(self.sampling_mode)
        return {k: v for k, v in self.profile.items() if v != released[k]}

    def rolling_profile(self, steps: Optional[int] = None) -> RollingProfile:
        p = self.profile
        common = dict(
            chunk_size=self.chunk_size,
            window_size=self.window_size,
            kv_recompute_enabled=bool(p["kv_recompute_enabled"]),
            kv_recompute_threshold=int(p["kv_recompute_threshold"]),
            kv_recompute_recent_k=int(p["kv_recompute_recent_k"]),
        )
        steps = self.check_steps(steps)
        if self.distilled:
            return RollingProfile(
                mode="distilled",
                steps_per_window=1,
                force_recompute_at_event=True,
                noise_anchor=True,
                **common,
            )
        return RollingProfile(
            mode="base",
            steps_per_window=steps // self.window_size,
            guided=self.use_cfg,
            cfg_min=float(p["cfg_min"]),
            cfg_max=float(p["cfg_max"]),
            **common,
        )


def read_model_folder(folder: str) -> tuple[str, dict[str, Any]]:
    """The chain and the overrides of a MovieRoll model folder (``config.json`` + ``model.safetensors``).

    The folder is read by the rules of native MovieRoll's `load_model_dir`: ``sampling_mode`` names the chain, any
    other key must be a setting of that chain, and a key left out keeps the built-in value. Returned are the
    values that differ from the built-in chain; each must be in `OVERRIDABLE`, so a folder that changes what this
    package runs at the released value is refused here instead of being run with that value.
    """
    config_path = os.path.join(folder, MODEL_FOLDER_CONFIG)
    for path in (config_path, os.path.join(folder, MODEL_FOLDER_WEIGHTS)):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{folder} is not a MovieRoll model folder: it has no {os.path.basename(path)}"
            )
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    mode = config.get("sampling_mode") if isinstance(config, dict) else None
    if mode not in SAMPLING_MODES:
        raise ValueError(
            f"{config_path}: sampling_mode must be one of {SAMPLING_MODES}, got {mode!r}"
        )
    released = {**get_profile(mode), **FOLDER_FIXED}
    unknown = sorted(set(config) - set(released) - {"sampling_mode"})
    if unknown:
        raise ValueError(f"{config_path}: unknown keys {unknown}")
    overrides = {
        key: value
        for key, value in config.items()
        if key != "sampling_mode" and value != released[key]
    }
    return mode, check_overrides(overrides, config_path)
