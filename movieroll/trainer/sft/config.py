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

"""The SFT configuration: one dataclass, and a validator that rejects anything it does not know.

The fields are the set an SFT run reads. A key the dataclass does not define is an error, so a typo is
reported by name instead of being ignored: a silently dropped key is indistinguishable from one that worked,
and a run configured with a half-chunk length would train for days before anyone noticed.

The cross-key checks are the ones that are cheap here and expensive at 8-GPU scale: a frame count that is not
a whole number of chunks, a RoPE table too small for the window, an inference step count the rolling window
cannot divide, and a sampling prompt CSV that is missing or lacks the columns the trainer reads, which would
otherwise surface after 33 GB of weights have been loaded.
"""

import csv
import os
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, List, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

from movieroll.commons.constants import BYT5_CKPT, BYT5_COLOR, BYT5_FONT, BYT5_GOOGLE, MODEL_TYPE
from movieroll.commons.streaming_config import StreamingConfig

__all__ = ["SFTConfig", "load_sft_config", "validate_sft_config"]

# Number of double-stream blocks of the 16164 model; `streaming_full_history_block_num` indexes into them.
_BLOCK_DEPTH = 54
# Attention heads of that model (MODEL_CONFIG[MODEL_TYPE]["heads_num"]): sequence parallelism splits them across the sp group, so `sp_size`
# has to divide it.
_NUM_HEADS = 16

# The latent grid of the 16164 VAE.
_TEMPORAL_STRIDE = 4

_PRECISIONS = ("fp32", "bf16", "fp16")


@dataclass
class SFTConfig:
    """Every value an SFT run reads, with its default.

    Only the fields below may appear in the YAML. The two trailing `init=False` fields are filled in by the
    trainer at startup and are not configurable: `streaming_config` is read when the sampling pipeline is
    built and by `create_transport`, `local_seed` by the seeding and the online-encode generator.
    """

    # ---- identity and paths ----
    dit_weight: str
    output_dir: str
    data_root: str
    model_type: str = MODEL_TYPE
    vae_path: str = "ckpts/vae"
    text_encoder_path: str = "ckpts/text_encoder/llm"
    byt5_google_path: str = BYT5_GOOGLE
    byt5_ckpt_path: str = BYT5_CKPT
    multilingual_prompt_format_font_path: str = BYT5_FONT
    multilingual_prompt_format_color_path: str = BYT5_COLOR
    resume_checkpoint: Optional[str] = None

    # ---- shape ----
    height: int = 480
    width: int = 640
    sample_n_frames: int = 241
    batch_size: int = 1
    num_workers: int = 8
    prefetch_factor: int = 2

    # ---- topology and memory ----
    sp_size: int = 4
    fsdp_sharding_strategy: str = "hybrid"
    fsdp_sub_world_size: int = 16
    use_cpu_offload: bool = False
    gradient_checkpointing: bool = True
    selective_checkpointing: float = 1.0

    # ---- precision ----
    master_weight_type: str = "fp32"
    precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precision: str = "fp16"

    # ---- optimizer and schedule ----
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    gradient_accumulation_steps: int = 1
    max_train_steps: int = 1000000
    num_train_epochs: int = 100      # not a stop condition: the data reshuffles every epoch until max_train_steps
    global_seed: int = 930
    ema_decay: float = 0.9993
    flow_shift_video: float = 3.0

    # ---- streaming ----
    streaming_chunk_size: int = 4
    streaming_window_size: int = 5
    streaming_max_num_latent_frames: int = 120
    streaming_full_history_block_num: int = 6
    streaming_t2v_prob: float = 0.8
    streaming_caption_type_probs: Any = field(
        default_factory=lambda: {"event": 0.9, "structure": 0.1}
    )
    streaming_context_noise_enabled: bool = True
    streaming_context_noise_prob: float = 0.9
    streaming_context_noise_sigma_max: float = 0.15
    streaming_context_noise_sigma_min: float = 0.0
    streaming_inference_context_noise: float = 0.05
    streaming_per_sample_null_prob: float = 0.05
    streaming_per_window_null_prob: float = 0.05
    streaming_interactive_mode: bool = False

    # ---- text ----
    text_len: int = 1000
    byt5_max_length: int = 256
    byt5_uncond_p: float = 0.1

    # ---- training-time sampling ----
    sample_interval: int = 10000
    fixed_prompt_validation_every: int = 10000
    sample_prompt_csv: Optional[List[str]] = None
    infer_steps_video: int = 50
    infer_flow_shift_video: float = 5.0
    use_cfg: bool = True
    neg_prompt: Optional[str] = None
    video_fps: int = 24
    try_first_iter: bool = False

    # ---- run control ----
    checkpointing_steps: int = 2000
    log_interval: int = 1
    gc_interval: int = 200
    loss_spike_threshold: float = 2.0
    dry_run: bool = False
    wandb_project: Optional[str] = None
    wandb_exp_name: Optional[str] = None
    disable_tensorboard: bool = True

    # ---- filled in by the trainer, never by the YAML ----
    streaming_config: Optional[StreamingConfig] = field(default=None, init=False)
    local_seed: int = field(default=0, init=False)

    def build_streaming_config(self) -> StreamingConfig:
        """The `StreamingConfig` an SFT run trains with.

        The three KV-recompute fields are not passed: SFT runs with the dataclass defaults (off / 5 / 3).
        """
        return StreamingConfig(
            max_num_latent_frames=self.streaming_max_num_latent_frames,
            chunk_size=self.streaming_chunk_size,
            window_size=self.streaming_window_size,
            full_history_block_num=self.streaming_full_history_block_num,
            t2v_prob=self.streaming_t2v_prob,
            context_noise_enabled=self.streaming_context_noise_enabled,
            context_noise_prob=self.streaming_context_noise_prob,
            context_noise_sigma_max=self.streaming_context_noise_sigma_max,
            context_noise_sigma_min=self.streaming_context_noise_sigma_min,
            inference_context_noise=self.streaming_inference_context_noise,
            per_sample_null_prob=self.streaming_per_sample_null_prob,
            per_window_null_prob=self.streaming_per_window_null_prob,
        )


ALLOWED_KEYS = frozenset(f.name for f in fields(SFTConfig) if f.init)
REQUIRED_KEYS = frozenset(
    f.name for f in fields(SFTConfig)
    if f.init and f.default is MISSING and f.default_factory is MISSING
)
# Scalar defaults, for the checks that must see the value a run gets when the YAML leaves a key out.
_DEFAULTS = {f.name: f.default for f in fields(SFTConfig) if f.init and f.default is not MISSING}

# The files `resume_training` reads from a checkpoint directory: the live model resumes from the EMA weights.
_RESUME_FILES = ("ema_diffusion_pytorch_model.safetensors", "scalar_states.pkl")

_INT_KEYS = (
    "height", "width", "sample_n_frames", "batch_size", "num_workers", "prefetch_factor", "sp_size",
    "fsdp_sub_world_size", "lr_warmup_steps", "lr_num_cycles", "gradient_accumulation_steps",
    "max_train_steps", "num_train_epochs", "global_seed", "streaming_chunk_size", "streaming_window_size",
    "streaming_max_num_latent_frames", "streaming_full_history_block_num",
    "text_len", "byt5_max_length", "sample_interval",
    "fixed_prompt_validation_every", "infer_steps_video", "video_fps", "checkpointing_steps", "log_interval",
    "gc_interval",
)
_FLOAT_KEYS = (
    "selective_checkpointing", "learning_rate", "weight_decay", "max_grad_norm", "lr_power", "ema_decay",
    "flow_shift_video", "streaming_context_noise_prob", "streaming_context_noise_sigma_max",
    "streaming_context_noise_sigma_min", "streaming_inference_context_noise",
    "streaming_per_sample_null_prob", "streaming_per_window_null_prob", "byt5_uncond_p", "infer_flow_shift_video",
    "loss_spike_threshold", "streaming_t2v_prob",
)
_BOOL_KEYS = (
    "use_cpu_offload", "gradient_checkpointing", "streaming_context_noise_enabled",
    "streaming_interactive_mode", "try_first_iter", "dry_run", "disable_tensorboard", "use_cfg",
)
# numpy seeds are uint32; `accelerate.utils.set_seed` raises above this.
_MAX_SEED = 2 ** 32 - 1
# The diffusers scheduler names that work with the `get_scheduler` call in `train.main`. `piecewise_constant` is the
# seventh SchedulerType and is excluded on purpose: it reads `step_rules`, which that call does not pass.
_LR_SCHEDULERS = frozenset({
    "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup",
})
# Keys that must be > 0 rather than >= 0.
_POSITIVE_KEYS = (
    "height", "width", "sample_n_frames", "batch_size", "prefetch_factor", "sp_size", "fsdp_sub_world_size",
    "lr_num_cycles", "gradient_accumulation_steps", "max_train_steps", "num_train_epochs",
    "streaming_chunk_size", "streaming_window_size", "streaming_max_num_latent_frames", "text_len",
    "byt5_max_length", "fixed_prompt_validation_every", "infer_steps_video", "video_fps",
    "checkpointing_steps", "log_interval", "gc_interval",
    "learning_rate", "max_grad_norm", "lr_power", "flow_shift_video", "infer_flow_shift_video",
    "loss_spike_threshold",
)
# Keys that are a probability or a ratio.
_UNIT_INTERVAL_KEYS = (
    "ema_decay", "streaming_context_noise_prob", "streaming_context_noise_sigma_max",
    "streaming_context_noise_sigma_min", "streaming_inference_context_noise",
    "streaming_per_sample_null_prob", "streaming_per_window_null_prob", "byt5_uncond_p",
    "streaming_t2v_prob",
)

_PROB_DICT_KEYS = {
    "streaming_caption_type_probs": {"event", "structure"},
}


def _plain(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _require_number_value(cfg, key):
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"SFT config: `{key}` must be a number, got {value!r}")
    return value


def _require_bool_value(cfg, key):
    value = cfg[key]
    if not isinstance(value, bool):
        raise ValueError(f"SFT config: `{key}` must be true or false, got {value!r}")
    return value


def _value_or_default(cfg, key):
    """`cfg[key]`, or the default a run gets when the YAML leaves the key out (the `ckpts/` layout for paths).

    `key` is a field name written here in the validator, not user input, so a name missing from `_DEFAULTS`
    raises instead of yielding `None` and turning a check downstream into one that never fires.
    """
    if key in cfg:
        return cfg[key]
    return _DEFAULTS[key]


def _require_existing(path, key, *, directory):
    checker = os.path.isdir if directory else os.path.isfile
    what = "directory" if directory else "file"
    if not checker(str(path)):
        raise FileNotFoundError(f"SFT config: `{key}` {what} not found: {path}")


def _nearest_valid_frame_counts(chunk_size, around):
    """The two legal `sample_n_frames` around a rejected one, for the error message."""
    period = _TEMPORAL_STRIDE * chunk_size
    lower = ((around - 1) // period) * period + 1
    return max(lower, 1), lower + period


def _check_sample_prompt_csv(path):
    """Check a sampling prompt CSV in the format `movieroll.utils.prompt_csv` reads; return its row count.

    Separate from that parser so the error names the file and the line of a malformed row. Every row needs
    `index` and an integer `seed`, plus the prompt: the `event_<N>` columns when there is at least one,
    otherwise `prompt`. `global_content_prompt` / `global_quality_prompt` are read when both exist; no other
    column is read.
    """
    with open(str(path), encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        event_cols = [name for name in names if name.startswith("event_") and name[6:].isdigit()]
        missing = [name for name in ("index", "seed") if name not in names]
        if not event_cols and "prompt" not in names:
            missing.append("prompt (or event_<N> columns)")
        if missing:
            raise ValueError(
                f"SFT config: `sample_prompt_csv` {path} lacks the column(s) {missing}; its header is {names}"
            )
        if event_cols:
            global_cols = [name for name in ("global_content_prompt", "global_quality_prompt") if name in names]
            read_cols = ["index", "seed"] + event_cols + (global_cols if len(global_cols) == 2 else [])
        else:
            read_cols = ["index", "seed", "prompt"]

        rows = 0
        for row in reader:
            try:
                short = [name for name in read_cols if row[name] is None]
                if short:
                    raise ValueError(f"the row ends before the column(s) {short}")
                int(row["seed"])
            except (TypeError, ValueError, IndexError, KeyError) as error:
                raise ValueError(
                    f"SFT config: `sample_prompt_csv` {path} line {reader.line_num}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            rows += 1
    return rows


def validate_sft_config(cfg):
    """Check an SFT config in place; raise `ValueError` / `FileNotFoundError` on anything wrong.

    Runs before any weight is loaded, so a bad path or an illegal frame count costs seconds, not the twenty
    minutes it takes to shard 33 GB of parameters across the mesh.
    """
    unknown = sorted(set(cfg.keys()) - ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            f"SFT config: unknown key(s) {unknown}: a typo, or a setting that no longer exists (the keys are the "
            f"fields of SFTConfig)."
        )
    missing = sorted(REQUIRED_KEYS - set(cfg.keys()))
    if missing:
        raise ValueError(f"SFT config: missing required key(s) {missing}")

    for key in _INT_KEYS:
        if key in cfg:
            value = _require_number_value(cfg, key)
            if int(value) != value:
                raise ValueError(f"SFT config: `{key}` must be an integer, got {value!r}")
            cfg[key] = int(value)
    for key in _FLOAT_KEYS:
        if key in cfg:
            cfg[key] = float(_require_number_value(cfg, key))
    for key in _BOOL_KEYS:
        if key in cfg:
            _require_bool_value(cfg, key)
    for key in _POSITIVE_KEYS:
        if key in cfg and cfg[key] <= 0:
            raise ValueError(f"SFT config: `{key}` must be positive, got {cfg[key]!r}")
    for key in ("num_workers", "lr_warmup_steps", "global_seed", "sample_interval",
                "streaming_full_history_block_num", "weight_decay"):
        if key in cfg and cfg[key] < 0:
            raise ValueError(f"SFT config: `{key}` must not be negative, got {cfg[key]!r}")
    # `local_seed` is `global_seed + rank // sp_size` and reaches `accelerate.utils.set_seed`, which forwards
    # it to numpy's uint32 seed and raises at exactly 2**32. A `global_seed` inside the limit can still put
    # the last rank over it, and only that rank fails, which hangs rather than erroring cleanly.
    global_seed = _value_or_default(cfg, "global_seed")
    if global_seed > _MAX_SEED:
        raise ValueError(
            f"SFT config: `global_seed` must be at most {_MAX_SEED} (numpy seeds are uint32), "
            f"got {global_seed}"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    highest_rank_seed = global_seed + (world_size - 1) // _value_or_default(cfg, "sp_size")
    if highest_rank_seed > _MAX_SEED:
        raise ValueError(
            f"SFT config: `global_seed={global_seed}` is in range, but rank {world_size - 1} would use "
            f"{highest_rank_seed} (global_seed + rank // sp_size), which is above {_MAX_SEED}"
        )
    # `get_scheduler` is reached only after the weights are loaded and FSDP has wrapped them. Of diffusers'
    # seven SchedulerType names, `piecewise_constant` is not usable at that call site: it needs `step_rules`,
    # which train.py does not pass, and fails with an AttributeError on None.
    lr_scheduler = _value_or_default(cfg, "lr_scheduler")
    if lr_scheduler not in _LR_SCHEDULERS:
        raise ValueError(
            f"SFT config: `lr_scheduler` must be one of {sorted(_LR_SCHEDULERS)}, got {lr_scheduler!r}"
        )
    for key in _UNIT_INTERVAL_KEYS:
        if key in cfg and not 0.0 <= cfg[key] <= 1.0:
            raise ValueError(f"SFT config: `{key}` must be in [0, 1], got {cfg[key]!r}")

    if _value_or_default(cfg, "model_type") != MODEL_TYPE:
        raise ValueError(f"SFT config: `model_type` must be {MODEL_TYPE!r}, got {cfg['model_type']!r}")
    if _value_or_default(cfg, "fsdp_sharding_strategy") not in ("hybrid", "full"):
        raise ValueError(
            f"SFT config: `fsdp_sharding_strategy` must be 'hybrid' or 'full', "
            f"got {cfg['fsdp_sharding_strategy']!r}"
        )
    # The spelling check is shared by all four keys; their legal value sets are not, so `precision` and
    # `master_weight_type` each get their own rule below.
    for key in ("master_weight_type", "precision", "vae_precision", "text_encoder_precision"):
        if key in cfg and cfg[key] not in _PRECISIONS:
            raise ValueError(f"SFT config: `{key}` must be one of {_PRECISIONS}, got {cfg[key]!r}")
    # `precision` is not read by the training forward, whose bf16 autocast is hardcoded; its only reader is
    # the training-time sampling pipeline, where fp32 turns that autocast off and the FlashAttention kernels
    # then reject the fp32 q/k/v. So fp32 is rejected only when that pipeline is guaranteed to run, which
    # means replaying the real predicates: not `dry_run`, and either `try_first_iter` or `max_train_steps`
    # strictly greater than `sample_interval` (the loop exits on `update_steps >= max_train_steps` before the
    # sampling gate ever sees equality). A flat `sample_interval > 0` would reject runs that never sample.
    if _value_or_default(cfg, "precision") == "fp32":
        sample_interval = _value_or_default(cfg, "sample_interval")
        will_sample = (
            sample_interval > 0
            and not _value_or_default(cfg, "dry_run")
            and (
                _value_or_default(cfg, "try_first_iter")
                or _value_or_default(cfg, "max_train_steps") > sample_interval
            )
        )
        if will_sample:
            raise ValueError(
                "SFT config: `precision` must be 'bf16' or 'fp16' when training-time sampling runs, got "
                "'fp32': the training forward is bf16 either way (its autocast is hardcoded), but "
                "`precision` is the sampling pipeline's autocast dtype, and fp32 turns that autocast off, "
                "after which the FlashAttention kernels reject the fp32 q/k/v. Set `precision: bf16`, or "
                "turn sampling off with `sample_interval: 0`"
            )
    if _value_or_default(cfg, "master_weight_type") == "fp16":
        raise ValueError(
            "SFT config: `master_weight_type` must be fp32 or bf16, got 'fp16': every non-fp32 master is built in "
            "bf16, so fp16 would silently train in bf16"
        )
    if not 0.0 < _value_or_default(cfg, "selective_checkpointing") <= 1.0:
        raise ValueError(
            f"SFT config: `selective_checkpointing` must be in (0, 1], got {cfg['selective_checkpointing']!r}"
        )
    if not 0.0 < _value_or_default(cfg, "ema_decay") < 1.0:
        raise ValueError(f"SFT config: `ema_decay` must be in (0, 1), got {cfg['ema_decay']!r}")

    for key, expected in _PROB_DICT_KEYS.items():
        if key not in cfg:
            continue
        probs = _plain(cfg[key])
        if not isinstance(probs, dict) or set(probs) != expected:
            raise ValueError(f"SFT config: `{key}` must have exactly the keys {sorted(expected)}, got {probs!r}")
        for name, value in probs.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"SFT config: `{key}[{name}]` must be a non-negative number, got {value!r}")
        if sum(probs.values()) <= 0:
            raise ValueError(f"SFT config: `{key}` must not be all zero")
        # `streaming_caption_type_probs` is normalised by the dataset itself (`p_event / total_p`), so a
        # positive sum is all that is required.

    # ---- cross-key checks ----
    height, width = _value_or_default(cfg, "height"), _value_or_default(cfg, "width")
    for key, value in (("height", height), ("width", width)):
        if value % 16 != 0:
            raise ValueError(f"SFT config: `{key}` must be a multiple of 16 (the VAE's spatial stride), got {value}")

    num_frames = _value_or_default(cfg, "sample_n_frames")
    chunk_size = _value_or_default(cfg, "streaming_chunk_size")
    window_size = _value_or_default(cfg, "streaming_window_size")
    if (num_frames - 1) % _TEMPORAL_STRIDE != 0:
        lower, upper = _nearest_valid_frame_counts(chunk_size, num_frames)
        raise ValueError(
            f"SFT config: `sample_n_frames` must be of the form 4n+1 (the VAE encodes 4 pixel frames per "
            f"latent frame), got {num_frames}; try {lower} or {upper}"
        )
    latent_frames = (num_frames - 1) // _TEMPORAL_STRIDE + 1
    if (latent_frames - 1) % chunk_size != 0:
        lower, upper = _nearest_valid_frame_counts(chunk_size, num_frames)
        raise ValueError(
            f"SFT config: `sample_n_frames={num_frames}` gives {latent_frames} latent frames, which is not a "
            f"whole number of chunks of {chunk_size}, so `trainer/flow/transport.py` would crop the tail "
            f"and train on fewer frames than configured; try {lower} or {upper}"
        )
    needed_latent_frames = latent_frames + chunk_size * window_size
    max_latent_frames = _value_or_default(cfg, "streaming_max_num_latent_frames")
    if max_latent_frames < needed_latent_frames:
        raise ValueError(
            f"SFT config: `streaming_max_num_latent_frames` must be at least "
            f"{needed_latent_frames} ({latent_frames} latent frames + chunk_size * window_size of RoPE "
            f"headroom for the rolling window), got {max_latent_frames}"
        )
    if not 0 <= _value_or_default(cfg, "streaming_full_history_block_num") <= _BLOCK_DEPTH:
        raise ValueError(
            f"SFT config: `streaming_full_history_block_num` must be in [0, {_BLOCK_DEPTH}], "
            f"got {cfg['streaming_full_history_block_num']}"
        )
    infer_steps = _value_or_default(cfg, "infer_steps_video")
    if infer_steps % window_size != 0:
        raise ValueError(
            f"SFT config: `infer_steps_video` must be divisible by `streaming_window_size`: "
            f"{infer_steps} % {window_size} != 0"
        )
    if (_value_or_default(cfg, "streaming_context_noise_sigma_min")
            > _value_or_default(cfg, "streaming_context_noise_sigma_max")):
        raise ValueError(
            "SFT config: `streaming_context_noise_sigma_min` must not exceed "
            "`streaming_context_noise_sigma_max`"
        )
    if (_value_or_default(cfg, "streaming_per_sample_null_prob")
            + _value_or_default(cfg, "streaming_per_window_null_prob")) > 1.0:
        raise ValueError(
            "SFT config: `streaming_per_sample_null_prob` + `streaming_per_window_null_prob` must not exceed 1"
        )
    sub_world_size = _value_or_default(cfg, "fsdp_sub_world_size")
    sp_size = _value_or_default(cfg, "sp_size")
    sharding = _value_or_default(cfg, "fsdp_sharding_strategy")
    # `fsdp_sub_world_size` reaches one expression, `dp_replicate = world_size // fsdp_sub_world_size`, and
    # only under `hybrid`; the division floors, so a non-divisor behaves like the neighbouring divisor. The
    # checks below are about what the derived degrees hit: `assert d >= 1` (commons/parallel_states.py) when
    # the quotient is 0, the message-less `assert parallel_dims.dp_replicate_enabled` (utils/fsdp.py) when it
    # is exactly 1 while the strategy is still hybrid, and `dp_shard >= 1` on `replicas * sp_size`.
    if _NUM_HEADS % sp_size:
        raise ValueError(f"SFT config: `sp_size={sp_size}` must divide the model's {_NUM_HEADS} attention heads")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # `movieroll/__init__.py` fills WORLD_SIZE in with "1" when it is absent, so the value alone cannot tell a
    # real single-rank launch from "not launched at all", and validating against a world size of 1 nobody
    # asked for would reject every shipped config. torchrun also sets TORCHELASTIC_RUN_ID, which nothing in
    # this repo synthesizes, so that is the signal that the world size is real; without the second clause,
    # `torchrun --nproc_per_node=1` lands on the message-less `assert d >= 1`.
    if world_size > 1 or "TORCHELASTIC_RUN_ID" in os.environ:
        if world_size % sp_size:
            raise ValueError(
                f"SFT config: WORLD_SIZE={world_size} must be a multiple of `sp_size={sp_size}`: the "
                f"remaining ranks have to divide into whole data-parallel groups"
            )
        if sharding == "hybrid" and sub_world_size != world_size:
            replicas = world_size // sub_world_size
            if replicas < 2:
                raise ValueError(
                    f"SFT config: hybrid sharding needs at least 2 replicas, but "
                    f"`fsdp_sub_world_size={sub_world_size}` leaves {replicas} at WORLD_SIZE={world_size}. "
                    f"Use `fsdp_sharding_strategy=full` for a single shard group, or lower "
                    f"`fsdp_sub_world_size` to at most {world_size // 2}"
                )
            if world_size % (replicas * sp_size):
                raise ValueError(
                    f"SFT config: `fsdp_sub_world_size={sub_world_size}` at WORLD_SIZE={world_size} gives "
                    f"{replicas} replicas, and {replicas} * `sp_size={sp_size}` does not divide "
                    f"{world_size}, so there is no whole number of shards per replica"
                )
    if cfg.get("wandb_project") is not None and not cfg.get("wandb_exp_name"):
        raise ValueError("SFT config: `wandb_exp_name` is required when `wandb_project` is set")

    prompt_csv = _plain(cfg.get("sample_prompt_csv"))
    if _value_or_default(cfg, "sample_interval") > 0:
        if not prompt_csv:
            raise ValueError(
                "SFT config: `sample_prompt_csv` is required when `sample_interval > 0`; set "
                "`sample_interval: 0` to train without training-time sampling"
            )
        if isinstance(prompt_csv, str):
            raise ValueError("SFT config: `sample_prompt_csv` must be a list of paths, not a single string")

    # ---- paths, before any weight is loaded ----
    _require_existing(cfg["dit_weight"], "dit_weight", directory=False)
    if not str(cfg["dit_weight"]).endswith(".safetensors"):
        raise ValueError(f"SFT config: `dit_weight` must be a .safetensors file, got {cfg['dit_weight']!r}")
    # A path key the YAML leaves out is checked at the `ckpts/` default the run will then use.
    vae_path = _value_or_default(cfg, "vae_path")
    _require_existing(vae_path, "vae_path", directory=True)
    for name in ("config.json", "diffusion_pytorch_model.safetensors"):
        _require_existing(os.path.join(str(vae_path), name), f"vae_path/{name}", directory=False)
    _require_existing(_value_or_default(cfg, "text_encoder_path"), "text_encoder_path", directory=True)
    _require_existing(_value_or_default(cfg, "byt5_google_path"), "byt5_google_path", directory=True)
    for key in ("byt5_ckpt_path", "multilingual_prompt_format_font_path", "multilingual_prompt_format_color_path"):
        _require_existing(_value_or_default(cfg, key), key, directory=False)
    _require_existing(cfg["data_root"], "data_root", directory=True)
    _require_existing(os.path.join(str(cfg["data_root"]), "manifest.jsonl"), "data_root/manifest.jsonl",
                      directory=False)
    _require_existing(os.path.join(str(cfg["data_root"]), "latents"), "data_root/latents", directory=True)
    if cfg.get("resume_checkpoint") is not None:
        _require_existing(cfg["resume_checkpoint"], "resume_checkpoint", directory=True)
        for name in _RESUME_FILES:
            _require_existing(os.path.join(str(cfg["resume_checkpoint"]), name), f"resume_checkpoint/{name}",
                              directory=False)

    if _value_or_default(cfg, "sample_interval") > 0:
        rows = 0
        for csv_path in prompt_csv:
            _require_existing(csv_path, "sample_prompt_csv", directory=False)
            rows += _check_sample_prompt_csv(csv_path)
        if rows == 0:
            raise ValueError(f"SFT config: `sample_prompt_csv` {prompt_csv} holds no prompt rows")

    return cfg


def load_sft_config(config_path, overrides=None) -> SFTConfig:
    """Read a YAML config, validate it and freeze it into an `SFTConfig`."""
    if not os.path.isfile(str(config_path)):
        raise FileNotFoundError(f"config not found: {config_path}")
    cfg = OmegaConf.load(str(config_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    cfg = validate_sft_config(cfg)
    plain = OmegaConf.to_container(cfg, resolve=True)
    return SFTConfig(**plain)
