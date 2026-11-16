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

"""Entry-point validation for the two-phase Stage3 DMD configuration.

Stage 3 shards three 8.3B models across the mesh, so a configuration is rejected before NCCL is
initialised and before any weight is loaded. Every key a DMD run reads is checked -- types, legal ranges,
and the relations between keys (the ladder has one descending entry per window slot, the phase frame
counts agree with the latent arithmetic) -- and every error is accumulated so one pass reports them all.

An unknown key is an error, not a setting that silently does nothing, so a typo is reported by name.

The checkpoint and asset paths (`vae_path`, `text_encoder_path`, the byT5 paths) are spelled as in the
SFT config, so one `ckpts/` layout serves every stage.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path

from movieroll.commons.constants import MODEL_TYPE, NUM_TRAIN_TIMESTEPS
from movieroll.pipelines.rolling_dmd_runtime import ordered_event_columns
from movieroll.trainer.dmd.data import DMDTrainingPhase, phase_spec_from_config

__all__ = ["validate_stage3_config"]

# Number of double-stream blocks of the 16164 model (MODEL_CONFIG[MODEL_TYPE]["mm_double_blocks_depth"]);
# `streaming_full_history_block_num` counts the leading ones.
_BLOCK_DEPTH = 54
# Attention heads of that model (MODEL_CONFIG[MODEL_TYPE]["heads_num"]): sequence parallelism splits them across the sp group, so `sp_size`
# has to divide it.
_NUM_HEADS = 16


ALLOWED_TOP_LEVEL_KEYS = {
    "allow_incomplete_checkpoint",
    "attn_mode",
    "byt5_ckpt_path",
    "byt5_google_path",
    "byt5_max_length",
    "data",
    "dfake_gen_update_ratio",
    "disable_tensorboard",
    "dmd",
    "dmd_phases",
    "ema_start_step",
    "ema_weight",
    "exp_name",
    "fake_score_ckpt",
    "fsdp",
    "gc_interval",
    "generator_ckpt",
    "gradient_accumulation_steps",
    "logdir",
    "master_weight_type",
    "max_grad_norm_critic",
    "max_grad_norm_generator",
    "mixed_precision",
    "model_type",
    "multilingual_prompt_format_color_path",
    "multilingual_prompt_format_font_path",
    "optimizer",
    "real_score_ckpt",
    "resume_checkpoint",
    "sample_at_start",
    "sample_interval",
    "sampling",
    "save_interval",
    "siglip_path",
    "sp_size",
    "streaming_config",
    "text_encoder_path",
    "text_encoder_precision",
    "text_len",
    "vae_path",
    "vae_precision",
    "val_interval",
    "video",
}
ALLOWED_OPTIMIZER_KEYS = {
    "adamw_betas",
    "adamw_betas_critic",
    "adamw_eps",
    "lr",
    "lr_critic",
    "weight_decay",
}
ALLOWED_FSDP_KEYS = {
    "fake_score_checkpointing",
    "fake_score_cpu_offload",
    "generator_checkpointing",
    "generator_cpu_offload",
    "real_score_checkpointing",
    "real_score_cpu_offload",
    "shard_world_size",
    "sharding_strategy",
    "text_encoder_cpu_offload",
}
ALLOWED_STREAMING_KEYS = {
    "streaming_chunk_size",
    "streaming_full_history_block_num",
    "streaming_kv_recompute_enabled",
    "streaming_kv_recompute_recent_k",
    "streaming_kv_recompute_threshold",
    "streaming_max_num_latent_frames",
    "streaming_window_size",
}
ALLOWED_DMD_KEYS = {
    "context_noise",
    "denoising_step_list",
    "score_window_latent_frames",
    "max_step_ratio",
    "min_step_ratio",
    "real_guidance_scale",
    "recompute_mid_chunk_max",
    "recompute_mid_chunk_min",
    "sampling_context_noise",
    "score_mode",
    "timestep_shift",
    "uniform_timestep_regularization",
    "vae_stream_decode_max_chunk_size",
}
# How the real / fake score models are conditioned. "t2v" is the released recipe; "acs" is
# anchor-conditioned scoring, which scores the window starting at the i2v anchor the i2v way.
SCORE_MODES = ("t2v", "acs")
ALLOWED_UNIFORM_REGULARIZATION_KEYS = {
    "candidate_step_indices",
    "probability",
}
ALLOWED_DATA_KEYS = {"batch_size", "num_workers"}
ALLOWED_VIDEO_KEYS = {"resolution", "task_sampling"}
ALLOWED_TASK_KEYS = {"t2v_prob"}
ALLOWED_SAMPLING_KEYS = {
    "decode_to_video",
    "height",
    "num_samples",
    "save_latents",
    "seed",
    "test_event_csv",
    "val_num_samples",
    "video_length",
    "width",
}
# The keys of either phase: a phase takes one event (prompt_column) or two (event_columns + switch range).
ALLOWED_PHASE_KEYS = {
    "csv",
    "event_columns",
    "event_count",
    "latent_frames",
    "max_steps",
    "prompt_column",
    "switch_frame_max",
    "switch_frame_min",
    "video_length",
}
# The two event columns of a two-event CSV (rolling_dmd_runtime.ordered_event_columns at N = 2).
TWO_EVENT_COLUMNS = ("event_1", "event_2")


def _require_equal(errors: list[str], label: str, actual, expected) -> None:
    if actual != expected:
        errors.append(f"{label} must be {expected!r}, got {actual!r}")


def _require_bool(errors: list[str], label: str, value) -> bool | None:
    if not isinstance(value, bool):
        errors.append(f"{label} must be a boolean, got {value!r}")
        return None
    return value


def _require_int(
    errors: list[str],
    label: str,
    value,
    *,
    minimum: int | None = None,
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        errors.append(f"{label} must be an integer, got {value!r}")
        return None
    result = int(value)
    if minimum is not None and result < minimum:
        errors.append(f"{label} must be >= {minimum}, got {result}")
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


def _require_resolution(errors: list[str], label: str, value) -> None:
    values = list(value) if value is not None else []
    if len(values) != 2:
        errors.append(f"{label} must be [height, width], got {values!r}")
        return
    for index, item in enumerate(values):
        size = _require_int(errors, f"{label}[{index}]", item, minimum=1)
        if size is not None and size % 16:
            errors.append(f"{label}[{index}] must be divisible by 16, got {size}")


# The two path helpers accumulate instead of raising: this module collects every error and raises once.
def _require_file(errors: list[str], label: str, path) -> None:
    if not path:
        errors.append(f"{label} must be configured")
        return
    if not Path(str(path)).is_file():
        errors.append(f"{label} file not found: {path}")


def _require_directory(errors: list[str], label: str, path) -> None:
    if not path:
        errors.append(f"{label} must be configured")
        return
    if not Path(str(path)).is_dir():
        errors.append(f"{label} directory not found: {path}")


def _reject_unknown(
    errors: list[str], mapping, allowed: set[str], prefix: str
) -> None:
    unknown = sorted(set(mapping.keys()) - allowed)
    if unknown:
        errors.append(f"unknown {prefix} fields are not allowed: {unknown}")


def _validate_single_prompt_csv(
    path: Path,
    prompt_column: str,
    *,
    phase_label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{phase_label} prompt CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields.count(prompt_column) != 1:
            raise ValueError(
                f"{phase_label} CSV {path} must contain exactly one "
                f"{prompt_column!r} column"
            )
        row_count = 0
        for row_index, row in enumerate(reader, start=2):
            row_count += 1
            if not (row.get(prompt_column) or "").strip():
                raise ValueError(
                    f"{phase_label} CSV {path} row {row_index} has an empty prompt"
                )
        if row_count == 0:
            raise ValueError(f"{phase_label} CSV {path} contains no prompts")


def _validate_dual_event_csv(path: Path, event_columns: tuple[str, ...]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"two-event CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        event_fields = ordered_event_columns(fields)
        if event_fields != TWO_EVENT_COLUMNS:
            raise ValueError(
                f"two-event CSV {path} must expose only event_1/event_2, "
                f"got {event_fields}"
            )
        if tuple(event_columns) != TWO_EVENT_COLUMNS:
            raise ValueError(
                "two-event columns must be exactly ('event_1', 'event_2')"
            )
        if any(fields.count(column) != 1 for column in event_columns):
            raise ValueError(
                f"two-event CSV {path} must contain event_1 and event_2 exactly once"
            )
        usable = 0
        for row in reader:
            events = [(row.get(column) or "").strip() for column in event_columns]
            if all(events):
                usable += 1
        if usable == 0:
            raise ValueError(f"two-event CSV {path} has no complete event pairs")


def _validate_sampling_event_csv(path: Path, *, require_images: bool = False) -> None:
    """The sampling benchmark CSV: ordered event columns, plus an input image per row under ACS.

    `require_images` is set by `dmd.score_mode: acs`, whose samples are image-to-video: a missing or
    unreadable image would only surface when the first sample is rendered, steps into the run.
    """
    if not path.is_file():
        raise FileNotFoundError(f"sampling.test_event_csv does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = ordered_event_columns(tuple(reader.fieldnames or ()))
        if require_images and "input_image" not in (reader.fieldnames or ()):
            raise ValueError(
                f"{path}: dmd.score_mode='acs' samples image-to-video and needs an input_image column"
            )
        count = 0
        for row_index, row in enumerate(reader, start=2):
            events = [(row.get(column) or "").strip() for column in columns]
            if any(not event for event in events):
                raise ValueError(
                    f"{path}: row {row_index} has an empty ordered event column"
                )
            if require_images:
                image = (row.get("input_image") or "").strip()
                if not image:
                    raise ValueError(
                        f"{path}: row {row_index} has an empty input_image"
                    )
                if not (path.parent / image).is_file():
                    raise FileNotFoundError(
                        f"{path}: row {row_index} input_image not found: {path.parent / image}"
                    )
            count += 1
        if count == 0:
            raise ValueError(f"sampling.test_event_csv has no event rows: {path}")


def validate_stage3_config(config) -> None:
    """Validate immutable contracts plus types/ranges for tunable settings."""

    errors: list[str] = []
    _reject_unknown(errors, config, ALLOWED_TOP_LEVEL_KEYS, "top-level")
    _reject_unknown(errors, config.optimizer, ALLOWED_OPTIMIZER_KEYS, "optimizer")
    _reject_unknown(errors, config.fsdp, ALLOWED_FSDP_KEYS, "fsdp")
    _reject_unknown(
        errors, config.streaming_config, ALLOWED_STREAMING_KEYS, "streaming_config"
    )
    _reject_unknown(errors, config.dmd, ALLOWED_DMD_KEYS, "dmd")
    uniform_reg = config.dmd.get("uniform_timestep_regularization")
    # The node is required: `probability: 0.0` switches the regularizer off (the pipeline
    # short-circuits on 0 before it draws), so an absent node can only mean the key was forgotten.
    if uniform_reg is None:
        errors.append("dmd.uniform_timestep_regularization is required")
    else:
        _reject_unknown(
            errors,
            uniform_reg,
            ALLOWED_UNIFORM_REGULARIZATION_KEYS,
            "dmd.uniform_timestep_regularization",
        )
        _require_number(
            errors,
            "dmd.uniform_timestep_regularization.probability",
            uniform_reg.get("probability"),
            minimum=0.0,
            maximum=1.0,
        )
        candidates = uniform_reg.get("candidate_step_indices")
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or not candidates
        ):
            errors.append(
                "dmd.uniform_timestep_regularization.candidate_step_indices "
                "must be a non-empty list"
            )
        else:
            parsed_candidates = []
            for index, value in enumerate(candidates):
                # minimum=1 rejects index 0 on purpose: index 0 is the shifted timestep 1000
                # (sigma=1), and a uniform re-noise at sigma=1 erases the rolling x0_pred from the
                # regularizer input entirely.
                parsed = _require_int(
                    errors,
                    "dmd.uniform_timestep_regularization."
                    f"candidate_step_indices[{index}]",
                    value,
                    minimum=1,
                )
                if parsed is not None:
                    parsed_candidates.append(parsed)
            if len(set(parsed_candidates)) != len(parsed_candidates):
                errors.append(
                    "dmd.uniform_timestep_regularization."
                    "candidate_step_indices must not contain duplicates"
                )
            ladder_size = len(config.dmd.denoising_step_list)
            if any(index >= ladder_size for index in parsed_candidates):
                errors.append(
                    "dmd.uniform_timestep_regularization candidate step index "
                    f"must be < {ladder_size}"
                )
    _reject_unknown(errors, config.data, ALLOWED_DATA_KEYS, "data")
    _reject_unknown(errors, config.video, ALLOWED_VIDEO_KEYS, "video")
    _reject_unknown(
        errors,
        config.video.task_sampling,
        ALLOWED_TASK_KEYS,
        "video.task_sampling",
    )
    _reject_unknown(errors, config.sampling, ALLOWED_SAMPLING_KEYS, "sampling")

    # Identity keys get a type check only. Whether a resume directory is usable is decided later, by
    # movieroll/distill.py, DMDTrainer._setup_logdir and DMDCheckpointManager.load.
    if not isinstance(config.exp_name, str) or not config.exp_name.strip():
        errors.append(f"exp_name must be a non-empty string, got {config.exp_name!r}")
    if not isinstance(config.logdir, str) or not config.logdir.strip():
        errors.append(f"logdir must be a non-empty string, got {config.logdir!r}")
    resume_checkpoint = config.get("resume_checkpoint")
    if resume_checkpoint is not None and not isinstance(resume_checkpoint, str):
        errors.append(
            f"resume_checkpoint must be a string or null, got {resume_checkpoint!r}"
        )

    sp_size = _require_int(errors, "sp_size", config.sp_size, minimum=1)
    if sp_size is not None and _NUM_HEADS % sp_size:
        errors.append(f"sp_size={sp_size} must divide the model's {_NUM_HEADS} attention heads")
    sharding_strategy = str(config.fsdp.sharding_strategy)
    if sharding_strategy not in {"full", "hybrid"}:
        errors.append(
            "fsdp.sharding_strategy must be 'full' or 'hybrid', "
            f"got {sharding_strategy!r}"
        )
    _require_int(
        errors, "fsdp.shard_world_size", config.fsdp.get("shard_world_size", 8), minimum=1
    )
    # `shard_world_size` reaches one expression only, `dp_replicate = world_size // shard_world` under
    # `hybrid`, so no relation to `sp_size` holds here; the structural relations are on the derived
    # degrees and need the world size, so they live in `dmd/setup.py`.
    # sorted() so several bad fsdp booleans always report in the same order, as at every multi-error site.
    for name in sorted(ALLOWED_FSDP_KEYS - {"sharding_strategy", "shard_world_size"}):
        _require_bool(errors, f"fsdp.{name}", config.fsdp[name])
    # `mixed_precision` is the Stage-3 COMPUTE dtype (bfloat16 if set, else float32). It is NOT the
    # optimizer's master-weight dtype (`master_weight_type`) and NOT FSDP's own mixed-precision policy.
    # Only `true` runs: with `false` the compute dtype is fp32, FSDP2's root pre-forward casts the
    # activations up to fp32, and every kernel in `modules/streaming_attention.py` routes to
    # `flash_attn_func_v3` / `flash_attn_varlen_func_v3`, which take bf16/fp16 only and have no SDPA
    # fallback.
    if _require_bool(errors, "mixed_precision", config.mixed_precision) is False:
        errors.append(
            "mixed_precision must be true: it is the Stage-3 compute dtype, and false makes the whole "
            "forward fp32, which the FlashAttention kernels reject with 'FlashAttention only supports "
            "fp16, bf16, and fp8_e4m3 data type'. To keep fp32 OPTIMIZER state, leave this true and set "
            "`master_weight_type: fp32` instead -- that is the shipped arrangement"
        )
    if config.get("allow_incomplete_checkpoint") is not None:
        _require_bool(
            errors,
            "allow_incomplete_checkpoint",
            config.allow_incomplete_checkpoint,
        )
    if config.get("disable_tensorboard") is not None:
        _require_bool(errors, "disable_tensorboard", config.disable_tensorboard)
    if config.get("sample_at_start") is not None:
        _require_bool(errors, "sample_at_start", config.sample_at_start)
    _require_int(
        errors,
        "gradient_accumulation_steps",
        config.gradient_accumulation_steps,
        minimum=1,
    )
    _require_int(errors, "gc_interval", config.gc_interval, minimum=1)
    _require_number(
        errors,
        "max_grad_norm_generator",
        config.max_grad_norm_generator,
        minimum=0.0,
        minimum_inclusive=False,
    )
    _require_number(
        errors,
        "max_grad_norm_critic",
        config.max_grad_norm_critic,
        minimum=0.0,
        minimum_inclusive=False,
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
        "optimizer.lr_critic",
        config.optimizer.lr_critic,
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
    _require_betas(
        errors, "optimizer.adamw_betas_critic", config.optimizer.adamw_betas_critic
    )
    _require_number(
        errors,
        "optimizer.adamw_eps",
        config.optimizer.adamw_eps,
        minimum=0.0,
        minimum_inclusive=False,
    )
    _require_int(
        errors,
        "streaming_config.streaming_max_num_latent_frames",
        config.streaming_config.streaming_max_num_latent_frames,
        minimum=1,
    )

    # A chunk holds at least 2 latent frames: the rollout requires `tt % chunk_size == 1`, and `tt % 1`
    # is 0 for every tt, so chunk_size 1 admits no clip length at all on this chain.
    chunk_size = _require_int(
        errors,
        "streaming_config.streaming_chunk_size",
        config.streaming_config.streaming_chunk_size,
        minimum=2,
    )
    # A window holds at least 2 chunks: uniform-timestep regularization re-noises at a ladder index in
    # [1, window_size), which a 1-entry ladder does not have.
    window_size = _require_int(
        errors,
        "streaming_config.streaming_window_size",
        config.streaming_config.streaming_window_size,
        minimum=2,
    )
    full_history_block_num = _require_int(
        errors,
        "streaming_config.streaming_full_history_block_num",
        config.streaming_config.streaming_full_history_block_num,
        minimum=0,
    )
    if full_history_block_num is not None and full_history_block_num > _BLOCK_DEPTH:
        errors.append(
            "streaming_config.streaming_full_history_block_num must be <= "
            f"{_BLOCK_DEPTH} (the number of double-stream blocks), got {full_history_block_num}"
        )
    _require_bool(
        errors,
        "streaming_config.streaming_kv_recompute_enabled",
        config.streaming_config.streaming_kv_recompute_enabled,
    )
    recompute_threshold = _require_int(
        errors,
        "streaming_config.streaming_kv_recompute_threshold",
        config.streaming_config.streaming_kv_recompute_threshold,
        minimum=1,
    )
    # `StreamingConfig.validate` rejects these two once recompute is on, which is after NCCL init and the
    # model loads, so the same two rules are checked here: `recent_k` positive, and the recompute's peak
    # cache (threshold chunks plus one window, plus the 1-frame first-chunk remainder) inside the RoPE table.
    recompute_enabled = config.streaming_config.streaming_kv_recompute_enabled is True
    _require_int(
        errors,
        "streaming_config.streaming_kv_recompute_recent_k",
        config.streaming_config.streaming_kv_recompute_recent_k,
        minimum=1 if recompute_enabled else 0,
    )
    max_frames = config.streaming_config.streaming_max_num_latent_frames
    if (
        recompute_enabled
        and None not in (recompute_threshold, chunk_size, window_size)
        and isinstance(max_frames, int)
        and (recompute_threshold + window_size) * chunk_size + 1 > max_frames
    ):
        errors.append(
            "streaming_config: KV recompute would exceed RoPE capacity: "
            f"(streaming_kv_recompute_threshold({recompute_threshold}) + streaming_window_size({window_size})) "
            f"* streaming_chunk_size({chunk_size}) + 1 > streaming_max_num_latent_frames({max_frames})"
        )
    # The rollout ladder is shifted whenever the shift is not 1, but the loss side only inverts and applies
    # the shift above 1, so a shift below 1 would train on timesteps the rollout never visits.
    _require_number(
        errors,
        "dmd.timestep_shift",
        config.dmd.timestep_shift,
        minimum=1.0,
    )
    _require_number(
        errors,
        "dmd.real_guidance_scale",
        config.dmd.real_guidance_scale,
        minimum=0.0,
    )
    vae_stream_decode_max_chunk_size = config.dmd.get(
        "vae_stream_decode_max_chunk_size", 4
    )
    if vae_stream_decode_max_chunk_size is not None:
        _require_int(
            errors,
            "dmd.vae_stream_decode_max_chunk_size",
            vae_stream_decode_max_chunk_size,
            minimum=1,
        )
    min_step_ratio = _require_number(
        errors, "dmd.min_step_ratio", config.dmd.min_step_ratio, minimum=0.0, maximum=1.0
    )
    max_step_ratio = _require_number(
        errors, "dmd.max_step_ratio", config.dmd.max_step_ratio, minimum=0.0, maximum=1.0
    )
    if (
        min_step_ratio is not None
        and max_step_ratio is not None
        and min_step_ratio >= max_step_ratio
    ):
        errors.append(
            "dmd min/max step ratios must satisfy "
            f"0 <= min < max <= 1, got {min_step_ratio}, {max_step_ratio}"
        )
    _require_number(
        errors,
        "dmd.context_noise",
        config.dmd.context_noise,
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    # Training-time sampling is inference, so it may need a different sigma than training: the ACS weights
    # are sampled at 0. Unset means "the training value", i.e. the released behaviour.
    if config.dmd.get("sampling_context_noise") is not None:
        _require_number(
            errors,
            "dmd.sampling_context_noise",
            config.dmd.sampling_context_noise,
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
    score_mode = config.dmd.get("score_mode", "t2v")
    siglip_path = config.get("siglip_path")
    if score_mode not in SCORE_MODES:
        errors.append(
            f"dmd.score_mode must be one of {list(SCORE_MODES)}, got {score_mode!r}"
        )
    elif score_mode == "acs":
        # ACS conditions on the i2v anchor, which only an i2v rollout has. A t2v batch has no anchor, yet
        # its first scoring window would still start at frame 0, so the condition would silently become a
        # generated frame and change what the teacher scores.
        if config.video.task_sampling.t2v_prob != 0:
            errors.append(
                "dmd.score_mode='acs' requires video.task_sampling.t2v_prob=0 (i2v only), got "
                f"{config.video.task_sampling.t2v_prob!r}"
            )
        # The teacher and the critic must start from one initialization, or their difference is a systematic
        # part of the DMD gradient from step 0.
        real_ckpt, fake_ckpt = config.get("real_score_ckpt"), config.get("fake_score_ckpt")
        if str(real_ckpt) != str(fake_ckpt):
            errors.append(
                "dmd.score_mode='acs' requires real_score_ckpt == fake_score_ckpt, got "
                f"{real_ckpt!r} and {fake_ckpt!r}"
            )
        _require_directory(errors, "siglip_path", siglip_path)
        if siglip_path:
            for name in ("image_encoder", "feature_extractor"):
                _require_directory(
                    errors, f"siglip_path/{name}", Path(str(siglip_path)) / name
                )
    elif siglip_path is not None:
        errors.append("siglip_path is only used with dmd.score_mode='acs'")
    # The generator re-encodes a score window BEFORE the current rolling window is appended to the re-encoder,
    # so a score window that is not longer than the rolling window (window_size chunks) would need a latent
    # that has not been produced yet.
    _require_int(
        errors,
        "dmd.score_window_latent_frames",
        config.dmd.score_window_latent_frames,
        minimum=window_size * chunk_size + 1 if None not in (window_size, chunk_size) else 1,
    )
    recompute_min = _require_int(
        errors,
        "dmd.recompute_mid_chunk_min",
        config.dmd.recompute_mid_chunk_min,
        minimum=1,
    )
    recompute_max = _require_int(
        errors,
        "dmd.recompute_mid_chunk_max",
        config.dmd.recompute_mid_chunk_max,
        minimum=1,
    )
    if (
        recompute_min is not None
        and recompute_max is not None
        and recompute_max < recompute_min
    ):
        errors.append(
            "dmd recompute middle-chunk range must satisfy "
            f"1 <= min <= max, got {recompute_min}..{recompute_max}"
        )
    _require_int(
        errors,
        "dfake_gen_update_ratio",
        config.dfake_gen_update_ratio,
        minimum=1,
    )
    # fp16 is not offered: every non-fp32 master is built in bf16, so fp16 would silently train in bf16.
    master_weight_type = str(config.master_weight_type).lower()
    if master_weight_type not in ("fp32", "bf16"):
        errors.append(f"master_weight_type must be 'fp32' or 'bf16', got {master_weight_type!r}")
    batch_size = _require_int(errors, "data.batch_size", config.data.batch_size, minimum=1)
    if batch_size is not None and batch_size != 1:
        errors.append(
            f"data.batch_size must be 1, got {batch_size}: Stage-3 rollouts are single-prompt (one prompt "
            "per rank per step)"
        )
    _require_int(errors, "data.num_workers", config.data.num_workers, minimum=0)
    # The lower bound is exclusive: EMA is unconditional, so weight 0 would be an EMA that never moves
    # rather than "EMA off".
    _require_number(
        errors,
        "ema_weight",
        config.ema_weight,
        minimum=0.0,
        minimum_inclusive=False,
        maximum=1.0,
        maximum_inclusive=False,
    )
    _require_int(errors, "ema_start_step", config.ema_start_step, minimum=0)
    _require_equal(
        errors,
        "model_type",
        str(config.model_type),
        MODEL_TYPE,
    )
    # The two full-attention branches `parallel_attention` implements for the non-streaming real and
    # fake score models.
    if str(config.attn_mode) not in {"flash3", "torch"}:
        errors.append(
            "attn_mode must be one of {'flash3', 'torch'}, "
            f"got {config.attn_mode!r}"
        )
    _require_int(errors, "byt5_max_length", config.byt5_max_length, minimum=1)
    if str(config.vae_precision) not in {"fp16", "bf16", "fp32"}:
        errors.append(
            "vae_precision must be one of {'fp16', 'bf16', 'fp32'}, "
            f"got {config.vae_precision!r}"
        )
    _require_int(errors, "text_len", config.text_len, minimum=1)
    if str(config.text_encoder_precision) not in {"fp16", "bf16", "fp32"}:
        errors.append(
            "text_encoder_precision must be one of {'fp16', 'bf16', 'fp32'}, "
            f"got {config.text_encoder_precision!r}"
        )
    _require_number(
        errors,
        "video.task_sampling.t2v_prob",
        config.video.task_sampling.t2v_prob,
        minimum=0.0,
        maximum=1.0,
    )
    _require_int(errors, "save_interval", config.save_interval, minimum=1)
    _require_int(errors, "sample_interval", config.sample_interval, minimum=0)
    _require_int(errors, "val_interval", config.val_interval, minimum=0)
    _require_resolution(errors, "video.resolution", config.video.resolution)

    # The frame-layout checks below divide by the chunk and window sizes.
    if chunk_size is None or window_size is None or chunk_size < 2 or window_size < 2:
        raise ValueError("Invalid Stage3 DMD configuration:\n- " + "\n- ".join(errors))
    denoising = list(config.dmd.denoising_step_list)
    _require_equal(errors, "len(dmd.denoising_step_list)", len(denoising), window_size)
    denoising_values: list[int] = []
    for index, value in enumerate(denoising):
        parsed = _require_int(
            errors,
            f"dmd.denoising_step_list[{index}]",
            value,
            minimum=1,
        )
        if parsed is not None:
            if parsed > NUM_TRAIN_TIMESTEPS:
                errors.append(
                    f"dmd.denoising_step_list[{index}] must be <= "
                    f"{NUM_TRAIN_TIMESTEPS}, got {parsed}"
                )
            denoising_values.append(parsed)
    if len(denoising_values) == len(denoising) and any(
        denoising_values[index] <= denoising_values[index + 1]
        for index in range(len(denoising_values) - 1)
    ):
        errors.append("dmd.denoising_step_list must be strictly descending")

    # The schedule lives in `DMDTrainingPhase.ordered()`, not in this mapping: every runtime read is
    # `config.dmd_phases[phase.value]`, keyed by an enum member, so the mapping's KEY ORDER has no
    # consumer. Missing and misspelled phases are both checked by name, which is what the error has to
    # point at.
    scheduled_phase_names = [phase.value for phase in DMDTrainingPhase.ordered()]
    missing_phases = [name for name in scheduled_phase_names if name not in config.dmd_phases]
    if missing_phases:
        # Both entries are required even by a run that resumes straight into phase 2:
        # `DMDCheckpointManager._validate_progress` reads phase 1's `max_steps` on every resume to
        # recompute the expected global_step.
        errors.append(
            f"dmd_phases is missing the phase(s) {missing_phases}: the schedule is "
            f"{scheduled_phase_names} and every entry is read, including on a resume that skips phase 1"
        )
    # A misspelled phase key is NOT inert: the shipped YAML gives every phase complete defaults, so
    # `--override dmd_phases.long_dual_evemt.csv=...` would validate and then train the real
    # `long_dual_event` for its full budget against the wrong data.
    _reject_unknown(errors, config.dmd_phases, set(scheduled_phase_names), "dmd_phases")
    final_phase_name = scheduled_phase_names[-1]
    for phase in DMDTrainingPhase.ordered():
        if phase.value not in config.dmd_phases:
            continue                      # already reported above; the per-phase body would only add noise
        try:
            spec = phase_spec_from_config(config, phase)
        except Exception as exc:
            errors.append(f"{phase.value}: {exc}")
            continue
        _reject_unknown(
            errors,
            config.dmd_phases[phase.value],
            ALLOWED_PHASE_KEYS,
            f"dmd_phases.{phase.value}",
        )
        # max_steps 0 is a real "skip this phase" for the FINAL phase only: `train()` logs that the final
        # phase is complete and returns. An earlier phase cannot be 0 because its handover writes a
        # phase-complete checkpoint and loads it straight back, and a checkpoint saved before the first
        # `optimizer.step()` holds no Muon momentum, so the load raises on a missing momentum_buffer.
        _require_int(
            errors,
            f"dmd_phases.{phase.value}.max_steps",
            config.dmd_phases[phase.value].max_steps,
            minimum=0 if phase.value == final_phase_name else 1,
        )
        phase_video_length = _require_int(
            errors,
            f"dmd_phases.{phase.value}.video_length",
            config.dmd_phases[phase.value].video_length,
            minimum=1,
        )
        if phase_video_length is not None and (phase_video_length - 1) % 4:
            errors.append(
                f"dmd_phases.{phase.value}.video_length must satisfy the 4n+1 pixel-frame rule, "
                f"got {phase_video_length}"
            )
        _require_int(
            errors,
            f"dmd_phases.{phase.value}.latent_frames",
            config.dmd_phases[phase.value].latent_frames,
            minimum=1,
        )
        event_count = _require_int(
            errors,
            f"dmd_phases.{phase.value}.event_count",
            config.dmd_phases[phase.value].event_count,
            minimum=1,
        )
        if event_count is not None and event_count > 2:
            errors.append(
                f"dmd_phases.{phase.value}.event_count must be 1 or 2: the batch decision stream "
                f"draws one switch frame, so more than two events is not implemented; got {event_count}"
            )
        computed_latent_frames = (spec.video_length - 1) // 4 + 1
        _require_equal(
            errors,
            f"dmd_phases.{phase.value} pixel-to-latent mapping",
            computed_latent_frames,
            spec.latent_frames,
        )
        if (spec.latent_frames - 1) % chunk_size:
            errors.append(
                f"{phase.value} latent_frames - 1 must be a multiple of chunk_size {chunk_size}, "
                f"got {spec.latent_frames}"
            )
        num_chunks = (spec.latent_frames - 1) // chunk_size
        if num_chunks % window_size:
            errors.append(
                f"{phase.value} num_chunks must be divisible by rolling window size"
            )
        if num_chunks < window_size:
            errors.append(
                f"{phase.value} needs at least one rolling window: {num_chunks} chunks of "
                f"{chunk_size} latent frames < window size {window_size}"
            )
        if spec.event_count == 2 and num_chunks < 2:
            errors.append(f"{phase.value} has two events but only {num_chunks} chunk(s)")
        # Without KV recompute the cache keeps the whole history, so the RoPE table has to hold every latent
        # frame plus one rolling window (the same headroom rule as the SFT validator).
        if (
            config.streaming_config.get("streaming_kv_recompute_enabled") is False
            and isinstance(config.streaming_config.get("streaming_max_num_latent_frames"), int)
        ):
            needed = spec.latent_frames + chunk_size * window_size
            if config.streaming_config.streaming_max_num_latent_frames < needed:
                errors.append(
                    f"{phase.value} needs streaming_max_num_latent_frames >= {needed} ({spec.latent_frames} "
                    "latent frames + one rolling window) while streaming_kv_recompute_enabled is false"
                )
        try:
            if spec.event_count == 1:
                if not isinstance(spec.prompt_column, str) or not spec.prompt_column:
                    errors.append(
                        f"dmd_phases.{phase.value}.prompt_column must be a "
                        "non-empty string"
                    )
                _require_equal(
                    errors,
                    f"dmd_phases.{phase.value}.event_columns",
                    spec.event_columns,
                    (),
                )
                _require_equal(
                    errors,
                    f"dmd_phases.{phase.value}.switch_frame_min",
                    spec.switch_frame_min,
                    None,
                )
                _require_equal(
                    errors,
                    f"dmd_phases.{phase.value}.switch_frame_max",
                    spec.switch_frame_max,
                    None,
                )
                _validate_single_prompt_csv(
                    spec.csv_path,
                    spec.prompt_column or "",
                    phase_label=phase.value,
                )
            else:
                _require_equal(
                    errors,
                    f"dmd_phases.{phase.value}.prompt_column",
                    spec.prompt_column,
                    None,
                )
                _require_equal(
                    errors,
                    f"dmd_phases.{phase.value}.event_columns",
                    spec.event_columns,
                    TWO_EVENT_COLUMNS,
                )
                switch_min = _require_int(
                    errors,
                    f"dmd_phases.{phase.value}.switch_frame_min",
                    config.dmd_phases[phase.value].switch_frame_min,
                    minimum=1,
                )
                switch_max = _require_int(
                    errors,
                    f"dmd_phases.{phase.value}.switch_frame_max",
                    config.dmd_phases[phase.value].switch_frame_max,
                    minimum=1,
                )
                if (
                    switch_min is not None
                    and switch_max is not None
                    and (
                        switch_min >= switch_max
                        or switch_max >= spec.video_length
                    )
                ):
                    errors.append(
                        f"{phase.value} switch range must satisfy "
                        f"0 < min < max < {spec.video_length}, "
                        f"got {switch_min}..{switch_max}"
                    )
                _validate_dual_event_csv(spec.csv_path, spec.event_columns)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    sampling_video_length = _require_int(
        errors, "sampling.video_length", config.sampling.video_length, minimum=1
    )
    sampling_latent_frames = (
        (sampling_video_length - 1) // 4 + 1
        if sampling_video_length is not None
        else 0
    )
    if sampling_video_length is not None and (sampling_video_length - 1) % 4:
        errors.append("sampling.video_length must satisfy the 4n+1 pixel-frame rule")
    if sampling_latent_frames and (sampling_latent_frames - 1) % chunk_size:
        errors.append(
            "sampling.video_length must map to a 1-frame prefix followed by full chunks"
        )
    # Training-time sampling runs the same rollout, so the RoPE table has to hold its clip as well.
    if (
        sampling_latent_frames
        and config.streaming_config.get("streaming_kv_recompute_enabled") is False
        and isinstance(config.streaming_config.get("streaming_max_num_latent_frames"), int)
        and chunk_size is not None
        and window_size is not None
    ):
        needed = sampling_latent_frames + chunk_size * window_size
        if config.streaming_config.streaming_max_num_latent_frames < needed:
            errors.append(
                f"sampling.video_length needs streaming_max_num_latent_frames >= {needed} "
                f"({sampling_latent_frames} latent frames + one rolling window) while "
                "streaming_kv_recompute_enabled is false"
            )
    _require_resolution(
        errors,
        "sampling resolution",
        [config.sampling.height, config.sampling.width],
    )
    # No uint32 bound: `sampling.seed` only reaches `torch.Generator(...).manual_seed(seed + global_index)`,
    # which takes an int64.
    _require_int(errors, "sampling.seed", config.sampling.seed)
    _require_bool(
        errors, "sampling.decode_to_video", config.sampling.decode_to_video
    )
    _require_bool(errors, "sampling.save_latents", config.sampling.save_latents)
    _require_int(errors, "sampling.num_samples", config.sampling.num_samples, minimum=1)
    val_num_samples = _require_int(
        errors, "sampling.val_num_samples", config.sampling.val_num_samples
    )
    if val_num_samples is not None and (
        val_num_samples < -1 or val_num_samples == 0
    ):
        errors.append("sampling.val_num_samples must be positive or -1")

    for field in ("generator_ckpt", "real_score_ckpt", "fake_score_ckpt"):
        value = config.get(field)
        if not value or not Path(str(value)).exists():
            errors.append(f"{field} does not exist: {value}")

    # The `ckpts/` layout, checked in the same pass as the transformer checkpoints so one run reports
    # every missing asset. .get() rather than attribute access: a missing key must add to the aggregated
    # list instead of raising an omegaconf error.
    _require_directory(errors, "vae_path", config.get("vae_path"))
    if config.get("vae_path"):
        for name in ("config.json", "diffusion_pytorch_model.safetensors"):
            _require_file(errors, f"vae_path/{name}", Path(str(config.vae_path)) / name)
    _require_directory(errors, "text_encoder_path", config.get("text_encoder_path"))
    _require_directory(errors, "byt5_google_path", config.get("byt5_google_path"))
    # One font / color pair feeds both the glyph byT5 tokenizer and the sampler's MultilingualPromptFormat.
    for key in ("byt5_ckpt_path", "multilingual_prompt_format_font_path", "multilingual_prompt_format_color_path"):
        _require_file(errors, key, config.get(key))

    default_test_csv = Path(str(config.sampling.test_event_csv))
    try:
        _validate_sampling_event_csv(
            default_test_csv, require_images=score_mode == "acs"
        )
    except (OSError, ValueError) as exc:
        errors.append(str(exc))

    if errors:
        raise ValueError("Invalid Stage3 DMD configuration:\n- " + "\n- ".join(errors))
