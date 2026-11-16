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

"""Central entry validation for the two supported Stage-2 rolling ODE workflows.

It runs before NCCL is initialised and before any weight is loaded, so a wrong key or a missing path
costs seconds rather than the twenty minutes it takes to shard the model. The two flows are cache-only
ODE training (plus its rolling validation) and teacher-only trajectory precompute. `infer_only` makes the
video-length checks read `sampling.video_length` instead of `video.num_frames` and lets `max_steps` be 0; it is
rejected in precompute mode. A key the validator does not know is an error, so a typo is reported by name.
"""

from __future__ import annotations

import csv
import math
import os
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

from movieroll.commons.constants import MODEL_TYPE, NUM_TRAIN_TIMESTEPS

ALLOWED_TOP_LEVEL_KEYS = {
    "byt5_ckpt_path",
    "byt5_google_path",
    "byt5_max_length",
    "data",
    "exp_name",
    "fsdp",
    "gc_interval",
    "infer_only",
    "logdir",
    "max_grad_norm_student",
    "max_steps",
    "mixed_precision",
    "model_type",
    "multilingual_prompt_format_color_path",
    "multilingual_prompt_format_font_path",
    "ode",
    "optimizer",
    "precompute_skip_student",
    "progress_report_interval",
    "resume_checkpoint",
    "sampling",
    "save_interval",
    "seed",
    "sp_size",
    "streaming_config",
    "student_ckpt",
    "teacher_ckpt",
    "text_encoder_path",
    "text_encoder_precision",
    "text_len",
    "vae_path",
    "vae_precision",
    "validate_at_start",
    "validate_on_save",
    "validation",
    "video",
    "video_fps",
}
# The optimizer is always Muon.
ALLOWED_OPTIMIZER_KEYS = {
    "adamw_betas",
    "adamw_eps",
    "lr",
    "weight_decay",
}
ALLOWED_FSDP_KEYS = {
    "shard_world_size",
    "sharding_strategy",
    "student_checkpointing",
    "student_cpu_offload",
    "teacher_checkpointing",
    "teacher_cpu_offload",
    "text_encoder_cpu_offload",
}
# Stage2 always trains text-to-video.
ALLOWED_ODE_KEYS = {
    "cache_only",
    "context_noise_sigma_max",
    "denoising_step_list",
    "max_num_noisy_windows",
    "teacher_num_inference_steps",
    "timestep_shift",
    "trajectory_cache_dir",
}
# One training resolution.
ALLOWED_VIDEO_KEYS = {"num_frames", "resolution"}
ALLOWED_DATA_KEYS = {"batch_size", "num_workers", "prompt_path"}
# No `seed` here: the sampling seed is derived from the top-level `seed` plus the step.
ALLOWED_SAMPLING_KEYS = {"use_cfg", "video_length"}
ALLOWED_VALIDATION_KEYS = {"n_sample_each_rank", "prompt_indices"}
ALLOWED_STREAMING_KEYS = {
    "streaming_chunk_size",
    "streaming_full_history_block_num",
    "streaming_kv_recompute_enabled",
    "streaming_kv_recompute_recent_k",
    "streaming_kv_recompute_threshold",
    "streaming_max_num_latent_frames",
    "streaming_window_size",
}

_PRECISIONS = {"fp16", "bf16", "fp32"}

# Number of double-stream blocks of the 16164 model (MODEL_CONFIG[MODEL_TYPE]["mm_double_blocks_depth"]);
# `streaming_full_history_block_num` counts the leading ones.
_BLOCK_DEPTH = 54
# Attention heads of that model (MODEL_CONFIG[MODEL_TYPE]["heads_num"]): sequence parallelism splits them across the sp group, so `sp_size`
# has to divide it.
_NUM_HEADS = 16


def _as_bool(value, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{label} must be a boolean, got {value!r}")


def _require_equal(errors: list[str], label: str, actual, expected) -> None:
    if actual != expected:
        errors.append(f"{label} must be {expected!r}, got {actual!r}")


# numpy seeds are uint32; `np.random.seed` raises above this.
_MAX_SEED = 2 ** 32 - 1


def _require_int(
    errors: list[str],
    label: str,
    value,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        errors.append(f"{label} must be an integer, got {value!r}")
        return None
    result = int(value)
    if minimum is not None and result < minimum:
        errors.append(f"{label} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        errors.append(f"{label} must be <= {maximum}, got {result}")
    return result


def _require_number(
    errors: list[str],
    label: str,
    value,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        errors.append(f"{label} must be numeric, got {value!r}")
        return None
    result = float(value)
    if not math.isfinite(result):
        errors.append(f"{label} must be finite, got {result}")
        return None
    if minimum is not None and (
        result < minimum or (not minimum_inclusive and result == minimum)
    ):
        op = ">=" if minimum_inclusive else ">"
        errors.append(f"{label} must be {op} {minimum}, got {result}")
    if maximum is not None and (
        result > maximum or (not maximum_inclusive and result == maximum)
    ):
        op = "<=" if maximum_inclusive else "<"
        errors.append(f"{label} must be {op} {maximum}, got {result}")
    return result


def _require_betas(errors: list[str], label: str, value) -> None:
    values = list(value) if value is not None else []
    if len(values) != 2:
        errors.append(f"{label} must contain exactly two values, got {values!r}")
        return
    for index, beta in enumerate(values):
        _require_number(
            errors,
            f"{label}[{index}]",
            beta,
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )


def _require_file(errors: list[str], label: str, value) -> Path | None:
    if not value:
        errors.append(f"{label} must be configured")
        return None
    path = Path(str(value))
    if not path.is_file():
        errors.append(f"{label} is not a file: {path}")
    return path


def _require_dir(errors: list[str], label: str, value) -> Path | None:
    if not value:
        errors.append(f"{label} must be configured")
        return None
    path = Path(str(value))
    if not path.is_dir():
        errors.append(f"{label} is not a directory: {path}")
    return path


def _validate_prompt_file(errors: list[str], value) -> Path | None:
    path = _require_file(errors, "Stage2 precompute prompt_file", value)
    if path is None or not path.is_file():
        return path
    try:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                fields = tuple(reader.fieldnames or ())
                if "prompt" not in fields:
                    errors.append(
                        f"Stage2 prompt CSV must contain a 'prompt' column: {path}"
                    )
                elif not any((row.get("prompt") or "").strip() for row in reader):
                    errors.append(f"Stage2 prompt CSV has no non-empty prompts: {path}")
        elif not any(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        ):
            errors.append(f"Stage2 prompt file has no non-empty prompts: {path}")
    except OSError as exc:
        errors.append(f"cannot read Stage2 prompt file {path}: {exc}")
    return path


def _require_keys(errors: list[str], mapping, keys: set[str], label: str) -> None:
    missing = sorted(keys - set(mapping.keys()))
    if missing:
        errors.append(f"{label} is missing required fields: {missing}")


def _reject_unknown(errors: list[str], mapping, allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping.keys()) - allowed)
    if unknown:
        errors.append(f"unknown {label} fields are not allowed: {unknown}")


def validate_stage2_config(
    config,
    *,
    mode: Literal["train", "precompute"],
    prompt_file: str | Path | None = None,
    num_shards: int = 1,
    shard_id: int = 0,
    sp_size: int | None = None,
) -> None:
    """Reject unsupported Stage2 configurations before NCCL/model allocation."""

    if mode not in {"train", "precompute"}:
        raise ValueError(f"unsupported Stage2 validation mode: {mode!r}")

    errors: list[str] = []
    required_sections = (
        "optimizer",
        "fsdp",
        "ode",
        "video",
        "data",
        "streaming_config",
        "sampling",
        "validation",
    )
    for name in required_sections:
        if name not in config:
            errors.append(f"missing Stage2 config section: {name}")
    if errors:
        raise ValueError("Invalid Stage2 ODE configuration:\n- " + "\n- ".join(errors))

    _reject_unknown(errors, config, ALLOWED_TOP_LEVEL_KEYS, "top-level Stage2")
    _reject_unknown(errors, config.optimizer, ALLOWED_OPTIMIZER_KEYS, "optimizer")
    _reject_unknown(errors, config.fsdp, ALLOWED_FSDP_KEYS, "fsdp")
    _reject_unknown(errors, config.ode, ALLOWED_ODE_KEYS, "ode")
    _reject_unknown(errors, config.video, ALLOWED_VIDEO_KEYS, "video")
    _reject_unknown(errors, config.data, ALLOWED_DATA_KEYS, "data")
    _reject_unknown(errors, config.sampling, ALLOWED_SAMPLING_KEYS, "sampling")
    _reject_unknown(
        errors, config.validation, ALLOWED_VALIDATION_KEYS, "validation"
    )
    _reject_unknown(
        errors,
        config.streaming_config,
        ALLOWED_STREAMING_KEYS,
        "streaming_config",
    )

    # Every key below is read by ATTRIBUTE somewhere, so a config missing one would die with OmegaConf's
    # ConfigAttributeError instead of being reported with the others. Keys the code reads through `.get`
    # with a real default stay OUT of this set: they are optional, and requiring them would reject
    # configurations that are legal.
    top_level_required = {
        "byt5_ckpt_path",
        "byt5_google_path",
        "byt5_max_length",
        "gc_interval",
        "logdir",
        "max_grad_norm_student",
        "max_steps",
        "mixed_precision",
        "model_type",
        "multilingual_prompt_format_color_path",
        "multilingual_prompt_format_font_path",
        "save_interval",
        "seed",
        "sp_size",
        "student_ckpt",
        "teacher_ckpt",
        "text_encoder_path",
        "text_encoder_precision",
        "text_len",
        "vae_path",
        "vae_precision",
    }
    if mode == "train":
        # only the train branch reads it; a precompute config legitimately omits it
        top_level_required.add("progress_report_interval")
    _require_keys(errors, config, top_level_required, "top-level Stage2 config")
    _require_keys(errors, config.optimizer, ALLOWED_OPTIMIZER_KEYS, "optimizer")
    _require_keys(
        errors,
        config.fsdp,
        {
            "sharding_strategy",
            "student_checkpointing",
            "teacher_checkpointing",
            "student_cpu_offload",
            "teacher_cpu_offload",
            "text_encoder_cpu_offload",
        },
        "fsdp",
    )
    _require_keys(errors, config.ode, ALLOWED_ODE_KEYS, "ode")
    _require_keys(errors, config.video, ALLOWED_VIDEO_KEYS, "video")
    _require_keys(errors, config.data, ALLOWED_DATA_KEYS, "data")
    _require_keys(errors, config.sampling, ALLOWED_SAMPLING_KEYS, "sampling")
    _require_keys(errors, config.validation, ALLOWED_VALIDATION_KEYS, "validation")
    # All seven keys are read unconditionally (three by attribute, four through checks that reject
    # None), so the section's allowed set is also its required set.
    _require_keys(errors, config.streaming_config, ALLOWED_STREAMING_KEYS, "streaming_config")
    if errors:
        raise ValueError("Invalid Stage2 ODE configuration:\n- " + "\n- ".join(errors))

    _require_equal(errors, "model_type", str(config.get("model_type")), MODEL_TYPE)
    try:
        _as_bool(config.get("mixed_precision"), label="mixed_precision")
    except ValueError as exc:
        errors.append(str(exc))
    _require_int(errors, "byt5_max_length", config.byt5_max_length, minimum=1)
    if str(config.vae_precision) not in _PRECISIONS:
        errors.append(
            f"vae_precision must be one of {sorted(_PRECISIONS)}, got {config.vae_precision!r}"
        )
    _require_int(errors, "text_len", config.text_len, minimum=1)
    if str(config.text_encoder_precision) not in _PRECISIONS:
        errors.append(
            "text_encoder_precision must be one of "
            f"{sorted(_PRECISIONS)}, got {config.text_encoder_precision!r}"
        )

    # Paths, checked before any weight is loaded.
    vae_dir = _require_dir(errors, "vae_path", config.get("vae_path"))
    if vae_dir is not None and vae_dir.is_dir():
        for name in ("config.json", "diffusion_pytorch_model.safetensors"):
            if not (vae_dir / name).is_file():
                errors.append(f"vae_path is missing {name}: {vae_dir}")
    _require_dir(errors, "text_encoder_path", config.get("text_encoder_path"))
    _require_dir(errors, "byt5_google_path", config.get("byt5_google_path"))
    for name in (
        "byt5_ckpt_path",
        "multilingual_prompt_format_font_path",
        "multilingual_prompt_format_color_path",
    ):
        _require_file(errors, name, config.get(name))

    streaming = config.streaming_config
    # A chunk holds at least 2 latent frames, not because the helper cannot express 1 --
    # `check_and_get_chunk_sizes(tt, chunk_size=1, mode=...)` returns [1]*tt for both modes -- but because
    # at chunk 1 the 'default' and 'i2v' layouts collapse onto the same [1,1,...], so the t2v/i2v multitask
    # split stops distinguishing anything, and because Stage 2's output initialises Stage 3, where chunk 1
    # is unsatisfiable for EVERY clip length (movieroll_dmd_pipeline.py requires tt % chunk_size == 1 and
    # tt % 1 is always 0): the run would produce weights no Stage-3 config can consume.
    chunk_size = _require_int(
        errors,
        "streaming_config.streaming_chunk_size",
        streaming.get("streaming_chunk_size"),
        minimum=2,
    )
    window_size = _require_int(
        errors,
        "streaming_config.streaming_window_size",
        streaming.get("streaming_window_size"),
        minimum=1,
    )
    full_history_block_num = _require_int(
        errors,
        "streaming_config.streaming_full_history_block_num",
        streaming.get("streaming_full_history_block_num"),
        minimum=0,
    )
    if full_history_block_num is not None and full_history_block_num > _BLOCK_DEPTH:
        errors.append(
            "streaming_config.streaming_full_history_block_num must be <= "
            f"{_BLOCK_DEPTH} (the number of double-stream blocks), got {full_history_block_num}"
        )
    # The frame-layout checks below divide by the chunk and window sizes.
    if chunk_size is None or window_size is None or chunk_size < 2 or window_size < 1:
        raise ValueError("Invalid Stage2 ODE configuration:\n- " + "\n- ".join(errors))
    max_latent_frames = _require_int(
        errors,
        "streaming_config.streaming_max_num_latent_frames",
        streaming.streaming_max_num_latent_frames,
        minimum=1,
    )
    recompute_threshold = _require_int(
        errors,
        "streaming_config.streaming_kv_recompute_threshold",
        streaming.streaming_kv_recompute_threshold,
        minimum=1,
    )
    kv_recompute_enabled = None
    try:
        kv_recompute_enabled = _as_bool(
            streaming.get("streaming_kv_recompute_enabled"),
            label="streaming_config.streaming_kv_recompute_enabled",
        )
    except ValueError as exc:
        errors.append(str(exc))
    # `StreamingConfig.validate` rejects these two once recompute is on, which is after NCCL init and the
    # model loads, so the same two rules are checked here: `recent_k` positive, and the recompute's peak
    # cache (threshold chunks plus one window, plus the 1-frame first-chunk remainder) inside the RoPE table.
    _require_int(
        errors,
        "streaming_config.streaming_kv_recompute_recent_k",
        streaming.streaming_kv_recompute_recent_k,
        minimum=1 if kv_recompute_enabled else 0,
    )
    if (
        kv_recompute_enabled
        and recompute_threshold is not None
        and max_latent_frames is not None
        and (recompute_threshold + window_size) * chunk_size + 1 > max_latent_frames
    ):
        errors.append(
            "streaming_config: KV recompute would exceed RoPE capacity: "
            f"(streaming_kv_recompute_threshold({recompute_threshold}) + streaming_window_size({window_size})) "
            f"* streaming_chunk_size({chunk_size}) + 1 > streaming_max_num_latent_frames({max_latent_frames})"
        )

    ode = config.ode
    _require_number(
        errors,
        "ode.timestep_shift",
        ode.timestep_shift,
        minimum=0.0,
        minimum_inclusive=False,
    )
    denoising_steps = list(ode.denoising_step_list)
    _require_equal(
        errors,
        "len(ode.denoising_step_list)",
        len(denoising_steps),
        window_size,
    )
    parsed_denoising_steps: list[int] = []
    for index, value in enumerate(denoising_steps):
        parsed = _require_int(
            errors,
            f"ode.denoising_step_list[{index}]",
            value,
            minimum=1,
        )
        if parsed is not None:
            if parsed > NUM_TRAIN_TIMESTEPS:
                errors.append(
                    f"ode.denoising_step_list[{index}] must be <= "
                    f"{NUM_TRAIN_TIMESTEPS}, got {parsed}"
                )
            parsed_denoising_steps.append(parsed)
    if len(parsed_denoising_steps) == len(denoising_steps) and any(
        parsed_denoising_steps[index] <= parsed_denoising_steps[index + 1]
        for index in range(len(parsed_denoising_steps) - 1)
    ):
        errors.append("ode.denoising_step_list must be strictly descending")
    teacher_steps = _require_int(
        errors,
        "ode.teacher_num_inference_steps",
        ode.teacher_num_inference_steps,
        minimum=1,
    )
    if teacher_steps is not None and teacher_steps % window_size:
        errors.append(
            "ode.teacher_num_inference_steps must be divisible by "
            "streaming_window_size"
        )
    _require_number(
        errors,
        "ode.context_noise_sigma_max",
        ode.context_noise_sigma_max,
        minimum=0.0,
        maximum=1.0,
    )
    _require_int(
        errors,
        "ode.max_num_noisy_windows",
        ode.max_num_noisy_windows,
        minimum=1,
    )

    video = config.video
    try:
        infer_only = _as_bool(
            config.get("infer_only", False), label="infer_only"
        )
    except ValueError as exc:
        errors.append(str(exc))
        infer_only = False
    video_length_raw = config.sampling.video_length if infer_only else video.num_frames
    video_length = _require_int(
        errors, "Stage2 video length", video_length_raw, minimum=1
    )
    if video_length is not None and (video_length - 1) % 4:
        errors.append(
            "Stage2 video length must be positive and map exactly through the "
            f"4x temporal VAE, got {video_length}"
        )
    # The latent frame count minus the first frame must divide into whole chunks.
    if video_length is not None and (video_length - 1) % 4 == 0:
        latent_frames = (video_length - 1) // 4 + 1
        if (latent_frames - 1) % chunk_size:
            errors.append(
                f"Stage2 video length {video_length} gives {latent_frames} latent frames, which is not "
                f"a whole number of chunks of {chunk_size}; half chunks are not supported"
            )
    resolution = list(video.resolution)
    if len(resolution) != 2:
        errors.append(f"video.resolution must be [height, width], got {resolution!r}")
    else:
        for dimension_index, value in enumerate(resolution):
            size = _require_int(
                errors,
                f"video.resolution[{dimension_index}]",
                value,
                minimum=1,
            )
            if size is not None and size % 16:
                errors.append(
                    f"video.resolution dimensions must be divisible by 16, got {resolution}"
                )

    _require_number(
        errors,
        "optimizer.lr",
        config.optimizer.lr,
        minimum=0.0,
        minimum_inclusive=False,
    )
    _require_number(
        errors,
        "optimizer.weight_decay",
        config.optimizer.weight_decay,
        minimum=0.0,
    )
    _require_betas(errors, "optimizer.adamw_betas", config.optimizer.adamw_betas)
    _require_number(
        errors,
        "optimizer.adamw_eps",
        config.optimizer.adamw_eps,
        minimum=0.0,
        minimum_inclusive=False,
    )
    _require_number(
        errors,
        "max_grad_norm_student",
        config.max_grad_norm_student,
        minimum=0.0,
        minimum_inclusive=False,
    )
    # `distill._run_stage2` seeds numpy with this, and numpy takes a uint32: `np.random.seed` raises at exactly
    # 2**32 and accepts 2**32 - 1. Without the bound a seed pasted from a 64-bit hash gets all the way past
    # the weight load before failing.
    _require_int(errors, "seed", config.seed, minimum=0, maximum=_MAX_SEED)
    if config.get("video_fps") is not None:
        _require_int(errors, "video_fps", config.video_fps, minimum=1)
    batch_size = _require_int(
        errors, "data.batch_size", config.data.batch_size, minimum=1
    )
    if batch_size is not None and batch_size != 1:
        errors.append(
            f"data.batch_size must be 1, got {batch_size}: the trajectory-cache collate passes exactly one "
            "cached rollout per step"
        )
    _require_int(errors, "data.num_workers", config.data.num_workers, minimum=0)
    _require_int(errors, "gc_interval", config.gc_interval, minimum=1)
    # Validation runs the Base pipeline: false samples without CFG (the shipped value: CFG was distilled into
    # the student), true guides with the streaming config's AnnealCFG.
    try:
        _as_bool(config.sampling.use_cfg, label="sampling.use_cfg")
    except ValueError as exc:
        errors.append(str(exc))
    sampling_video_length = _require_int(
        errors,
        "sampling.video_length",
        config.sampling.video_length,
        minimum=1,
    )
    if sampling_video_length is not None:
        if (sampling_video_length - 1) % 4:
            errors.append("sampling.video_length must satisfy the 4n+1 frame rule")
        elif ((sampling_video_length - 1) // 4) % chunk_size:
            errors.append(
                f"sampling.video_length {sampling_video_length} gives "
                f"{(sampling_video_length - 1) // 4 + 1} latent frames, which is not a whole number of "
                f"chunks of {chunk_size}, so the tail would be cropped and the clip would be shorter "
                f"than configured (trainer/flow/transport.py crops it)"
            )
    # Without KV recompute the cache keeps the whole history, so the RoPE table has to hold every latent frame
    # plus one rolling window (the same headroom rule as the SFT and Stage-3 validators).
    if kv_recompute_enabled is False and max_latent_frames is not None:
        for label, length in (
            ("Stage2 video length", video_length),
            ("sampling.video_length", sampling_video_length),
        ):
            if length is None or (length - 1) % 4:
                continue
            needed = (length - 1) // 4 + 1 + chunk_size * window_size
            if max_latent_frames < needed:
                errors.append(
                    f"{label} {length} needs streaming_config.streaming_max_num_latent_frames >= {needed} "
                    "(its latent frames + one rolling window) while streaming_kv_recompute_enabled is false"
                )
    prompt_indices = list(config.validation.prompt_indices)
    if not prompt_indices:
        errors.append("validation.prompt_indices must not be empty")
    for index, prompt_index in enumerate(prompt_indices):
        _require_int(
            errors,
            f"validation.prompt_indices[{index}]",
            prompt_index,
            minimum=0,
        )
    _require_int(
        errors,
        "validation.n_sample_each_rank",
        config.validation.n_sample_each_rank,
        minimum=1,
    )
    for name in (
        "student_checkpointing",
        "teacher_checkpointing",
        "student_cpu_offload",
        "teacher_cpu_offload",
        "text_encoder_cpu_offload",
    ):
        if not isinstance(config.fsdp.get(name), bool):
            errors.append(f"fsdp.{name} must be a boolean")
    for name in (
        "validate_at_start",
        "validate_on_save",
    ):
        if config.get(name) is not None:
            try:
                _as_bool(config.get(name), label=name)
            except ValueError as exc:
                errors.append(str(exc))

    configured_sp = _require_int(errors, "sp_size", config.sp_size, minimum=1)
    if sp_size is not None:
        # Range-checked, but NOT compared with `config.sp_size`: both callers assign
        # `config.sp_size = sp_size` and then pass that same value (precompute_ode.main does both, and the
        # CPU test the same way), so a mismatch is unreachable.
        _require_int(errors, "precompute sp_size", sp_size, minimum=1)
    if configured_sp is not None and _NUM_HEADS % configured_sp:
        errors.append(f"sp_size={configured_sp} must divide the model's {_NUM_HEADS} attention heads")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if configured_sp is not None and world_size % configured_sp:
        errors.append(
            f"WORLD_SIZE={world_size} must be divisible by sp_size={configured_sp}"
        )
    sharding = str(config.fsdp.sharding_strategy)
    if sharding not in {"full", "hybrid"}:
        errors.append(
            f"fsdp.sharding_strategy must be 'full' or 'hybrid', got {sharding!r}"
        )
    shard_world_size = _require_int(
        errors, "fsdp.shard_world_size", config.fsdp.get("shard_world_size", 8), minimum=1
    )
    # `fsdp.shard_world_size` reaches exactly one expression -- `dp_replicate = self.world_size //
    # shard_world_size` (ODERegressionMainTrainer._apply_fsdp_wrapping) -- and only under `hybrid`. The division is a FLOOR, so a
    # non-divisor is not an error anywhere downstream; it rounds the replica count down to a neighbouring
    # admissible one (WORLD_SIZE=8 with 3 gives 2 replicas of 4 ranks). What IS structural is what the
    # DERIVED degrees face in `ParallelDims` and `utils/fsdp.py`; see the same block in
    # trainer/sft/config.py for the three asserts.
    if (
        world_size > 1
        and sharding == "hybrid"
        and shard_world_size is not None
        and configured_sp is not None
        and world_size != shard_world_size
    ):
        replicas = world_size // shard_world_size
        if replicas < 2:
            errors.append(
                f"hybrid Stage2 needs at least 2 replicas, but fsdp.shard_world_size={shard_world_size} "
                f"leaves {replicas} at WORLD_SIZE={world_size}; use fsdp.sharding_strategy=full for a "
                f"single shard group, or lower fsdp.shard_world_size to at most {world_size // 2}"
            )
        elif world_size % (replicas * configured_sp):
            errors.append(
                f"fsdp.shard_world_size={shard_world_size} at WORLD_SIZE={world_size} gives {replicas} "
                f"replicas, and {replicas} * sp_size={configured_sp} does not divide {world_size}, so there "
                f"is no whole number of shards per replica"
            )

    cache_dir = (
        Path(str(ode.trajectory_cache_dir))
        if ode.get("trajectory_cache_dir")
        else None
    )
    if cache_dir is None:
        errors.append("ode.trajectory_cache_dir must be configured")

    if mode == "train":
        try:
            precompute_skip_student = _as_bool(
                config.get("precompute_skip_student", False),
                label="precompute_skip_student",
            )
        except ValueError as exc:
            errors.append(str(exc))
            precompute_skip_student = False
        try:
            cache_only = _as_bool(ode.cache_only, label="ode.cache_only")
        except ValueError as exc:
            errors.append(str(exc))
            cache_only = False
        if precompute_skip_student:
            errors.append("Stage2 training must not set precompute_skip_student=true")
        if not cache_only:
            errors.append(
                "Stage2 training supports only cache_only=true; use the precompute "
                "entry to create trajectories"
            )
        student_path = _require_file(errors, "student_ckpt", config.get("student_ckpt"))
        if student_path is not None and student_path.suffix != ".safetensors":
            errors.append(f"student_ckpt must be a .safetensors file, got {student_path}")
        if cache_dir is not None:
            if not cache_dir.is_dir():
                errors.append(
                    f"Stage2 trajectory cache is not a directory: {cache_dir}"
                )
            elif not any(cache_dir.glob("*.pt")):
                errors.append(
                    f"Stage2 trajectory cache contains no .pt files: {cache_dir}"
                )
        _require_int(
            errors,
            "max_steps",
            config.max_steps,
            minimum=0 if infer_only else 1,
        )
        _require_int(errors, "save_interval", config.save_interval, minimum=1)
        _require_int(
            errors,
            "progress_report_interval",
            config.progress_report_interval,
            minimum=1,
        )
        if config.get("resume_checkpoint"):
            resume = Path(str(config.resume_checkpoint))
            if not resume.is_dir():
                errors.append(f"resume_checkpoint is not a directory: {resume}")
            else:
                for filename in (
                    "student.safetensors",
                    "optimizer.pt",
                    "training_state.pt",
                ):
                    if not (resume / filename).is_file():
                        errors.append(
                            f"resume_checkpoint is missing {filename}: {resume}"
                        )
    else:
        if infer_only:
            errors.append("Stage2 precompute must not set infer_only=true")
        teacher_path = _require_file(errors, "teacher_ckpt", config.get("teacher_ckpt"))
        if teacher_path is not None and teacher_path.suffix != ".safetensors":
            errors.append(f"teacher_ckpt must be a .safetensors file, got {teacher_path}")
        try:
            cache_only = _as_bool(ode.cache_only, label="ode.cache_only")
        except ValueError as exc:
            errors.append(str(exc))
            cache_only = True
        try:
            precompute_skip_student = _as_bool(
                config.get("precompute_skip_student", False),
                label="precompute_skip_student",
            )
        except ValueError as exc:
            errors.append(str(exc))
            precompute_skip_student = False
        if cache_only:
            errors.append("Stage2 precompute requires ode.cache_only=false")
        if not precompute_skip_student:
            errors.append("Stage2 precompute requires precompute_skip_student=true")
        _validate_prompt_file(errors, prompt_file)
        validated_num_shards = _require_int(
            errors, "num_shards", num_shards, minimum=1
        )
        validated_shard_id = _require_int(
            errors, "shard_id", shard_id, minimum=0
        )
        if (
            validated_num_shards is not None
            and validated_shard_id is not None
            and validated_shard_id >= validated_num_shards
        ):
            errors.append(
                "shard_id must satisfy 0 <= shard_id < num_shards, got "
                f"{shard_id}/{num_shards}"
            )
        if cache_dir is not None and not cache_dir.parent.is_dir():
            errors.append(
                f"parent of Stage2 trajectory cache does not exist: {cache_dir.parent}"
            )

    if errors:
        raise ValueError("Invalid Stage2 ODE configuration:\n- " + "\n- ".join(errors))
