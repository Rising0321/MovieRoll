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

"""
Streaming configuration and enums.
"""
from dataclasses import dataclass
from enum import Enum, auto

import torch


class RollingStageType(Enum):
    # Prefill: the rolling window grows from 0 to window_size.
    PREFILL = auto()
    # Decode: steady-state sliding window; pop a clean chunk and push a noisy / filler chunk.
    DECODE = auto()

@dataclass
class StreamingConfig:
    """Hyper-parameters of the MovieRoll streaming architecture, shared by training and inference.

    The double-stream blocks attend to the clean history through HieraKV, in two block groups:
      * full-history blocks [0, full_history_block_num): every chunk sees the full clean history;
      * sink-recent blocks [full_history_block_num, depth): every chunk sees the sink chunk plus its most recent
        chunk.
    In parallel training the middle cond tokens are dropped at the full-history -> sink-recent boundary (keeping the
    sink and the recent chunk of every noisy window) and the slot RoPE is reset; at inference the KV cache of a
    sink-recent block holds at most sink_chunks + recent_chunks chunks.
    """
    kv_cache_save_dtype: torch.dtype = torch.bfloat16
    # RoPE resampling keeps every position below this many latent frames.
    max_num_latent_frames: int = 64
    chunk_size: int = 2
    window_size: int = 1
    # Probability that a training step is t2v; the rest of the mass is i2v. The two task forms differ only in
    # the chunk layout (t2v starts 1 + chunk_size, i2v starts with the single reference frame) and in whether
    # the first chunk is kept as the clean prefix. Named and polarised like Stage-3's `task_sampling.t2v_prob`.
    t2v_prob: float = 0.8
    full_history_block_num: int = 0
    # sink / recent must both be 1: only then do training and rolling inference see the same history (see
    # validate() for what goes wrong otherwise).
    sink_chunks: int = 1      # leading chunks kept forever as the sink
    recent_chunks: int = 1    # most recent chunks additionally visible to every noisy chunk
    # AnnealCFG of the Base pipeline (the per-frame CFG scale): linear in the timestep, cfg_max at high noise and
    # cfg_min at low noise.
    cfg_min: float = 1.0
    cfg_max: float = 4.0
    # Context noise: the clean history is re-noised, (1 - sigma) * clean + sigma * eps, before the model sees it.
    context_noise_enabled: bool = False
    context_noise_prob: float = 0.9          # training: per-sample probability of noising the history
    context_noise_sigma_max: float = 0.15    # training: upper bound of the per-chunk sigma
    context_noise_sigma_min: float = 0.0     # training: lower bound of the per-chunk sigma
    inference_context_noise: float = 0.05    # Base pipeline inference: the one sigma of the clean history

    # Null prompt dropout of the per-window text: a whole sample is dropped with per_sample_null_prob, then each
    # remaining window with per_window_null_prob / (1 - per_sample_null_prob), so the marginal null rate of a
    # window is per_sample_null_prob + per_window_null_prob.
    per_sample_null_prob: float = 0.05
    per_window_null_prob: float = 0.05

    # ---- KV recompute (inference) ----
    # When the KV cache reaches kv_recompute_threshold chunks, chunks are picked from the decoded latents (the first
    # one, the middle of every event and the kv_recompute_recent_k most recent ones), concatenated and re-encoded at
    # timestep 0 without text through block-wise causal attention (FBCA) into a fresh KV cache.
    kv_recompute_enabled: bool = False
    kv_recompute_threshold: int = 5    # number of cached chunks that triggers a recompute
    kv_recompute_recent_k: int = 3     # number of most recent chunks kept

    def __post_init__(self):
        self.validate()

    def validate(self):
        chunk_size = self.chunk_size
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")

        # HieraKV parameters; the upper bound (model depth) is checked when the transformer is built.
        if self.full_history_block_num < 0:
            raise ValueError(f"full_history_block_num must be non-negative, got {self.full_history_block_num}")
        if self.sink_chunks != 1:
            raise ValueError(
                f"sink_chunks must be 1, got {self.sink_chunks}: training and rolling inference only "
                "see the same history when both the sink and the recent window are one chunk. KV recompute "
                "rebuilds only chunk 0 as the sink, so a second sink chunk would be a stale recent chunk kept forever."
            )
        if self.recent_chunks != 1:
            raise ValueError(
                f"recent_chunks must be 1, got {self.recent_chunks}: training and rolling inference "
                "only see the same history when both the sink and the recent window are one chunk. More than one "
                "recent chunk turns the sink-recent blocks into a sliding window, which training does not "
                "reproduce, and keeps features of chunks that have already left the window inside the recent chunk."
            )

        # AnnealCFG parameters
        if self.cfg_min <= 0 or self.cfg_max <= 0:
            raise ValueError(
                f"cfg_min/cfg_max must be positive, got min={self.cfg_min}, max={self.cfg_max}"
            )

        for name in (
            "context_noise_sigma_max", "context_noise_sigma_min", "inference_context_noise",
            "per_sample_null_prob", "per_window_null_prob",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.context_noise_sigma_min > self.context_noise_sigma_max:
            raise ValueError(
                f"context_noise_sigma_min ({self.context_noise_sigma_min}) must not exceed "
                f"context_noise_sigma_max ({self.context_noise_sigma_max})"
            )
        if self.per_sample_null_prob + self.per_window_null_prob > 1:
            raise ValueError(
                f"per_sample_null_prob + per_window_null_prob must not exceed 1, got "
                f"{self.per_sample_null_prob} + {self.per_window_null_prob}"
            )
        # t2v_prob is a single Bernoulli parameter, so there is no distribution to normalise and nothing to
        # warn about; only the type and the range are checked.
        if isinstance(self.t2v_prob, bool) or not isinstance(self.t2v_prob, (int, float)):
            raise ValueError(f"t2v_prob must be a number, got {type(self.t2v_prob)}")
        if not 0 <= self.t2v_prob <= 1:
            raise ValueError(f"t2v_prob must be in [0, 1], got {self.t2v_prob}")

        # KV recompute parameters
        if self.kv_recompute_enabled:
            if self.kv_recompute_threshold <= 0:
                raise ValueError(f"kv_recompute_threshold must be positive, got {self.kv_recompute_threshold}")
            if self.kv_recompute_recent_k <= 0:
                raise ValueError(f"kv_recompute_recent_k must be positive, got {self.kv_recompute_recent_k}")
            # The recompute must fire before the KV cache outgrows max_num_latent_frames: in the round before
            # the cache reaches threshold chunks, the denoising window has to fit next to the cache. The cache
            # is one frame LARGER than threshold * chunk_size, because `check_and_get_chunk_sizes` hands the
            # first chunk 1 + chunk_size frames whenever the clip's latent length leaves
            # remainder 1. This validator sees neither the task nor the clip length, so the bound is the upper
            # bound over both modes and both remainders: passing it means every task and every length fits.
            peak_frames = (self.kv_recompute_threshold + self.window_size) * chunk_size + 1
            if peak_frames > self.max_num_latent_frames:
                raise ValueError(
                    f"KV recompute would exceed RoPE capacity: "
                    f"(kv_recompute_threshold({self.kv_recompute_threshold}) + window_size({self.window_size})) "
                    f"* chunk_size({chunk_size}) + 1 = {peak_frames} > "
                    f"max_num_latent_frames({self.max_num_latent_frames}). "
                    f"Decrease kv_recompute_threshold or increase max_num_latent_frames."
                )


def streaming_config_from_stage_block(raw):
    """The StreamingConfig of a Stage-2 / Stage-3 YAML's `streaming_config` block (its `streaming_*` keys).

    Shared by the ODE and the DMD trainer so the student and the generator are built with the same geometry. The
    SFT config builds its own (a different key set). `raw` is read by attribute, so this module stays free of the
    config library.
    """
    return StreamingConfig(
        chunk_size=int(raw.streaming_chunk_size),
        window_size=int(raw.streaming_window_size),
        max_num_latent_frames=int(raw.streaming_max_num_latent_frames),
        full_history_block_num=int(raw.streaming_full_history_block_num),
        kv_recompute_enabled=bool(raw.streaming_kv_recompute_enabled),
        kv_recompute_threshold=int(raw.streaming_kv_recompute_threshold),
        kv_recompute_recent_k=int(raw.streaming_kv_recompute_recent_k),
    )
