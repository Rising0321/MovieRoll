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

"""Stage-2 ODE regression: the student learns the teacher's rolling ODE trajectory.

Reference: CausVid (https://arxiv.org/abs/2412.07772) section 4.3.

Teacher and student share the same causal streaming architecture, which is what makes the trajectory
transferable. The teacher's 50-step rolling inference is cached offline, one entry per round
(`movieroll/scripts/precompute_ode.py`); the student replays one of them under its 5-step schedule as
window-wise teacher forcing: a pivot chunk is drawn, the clean history before it becomes the conditioning
prefix, and up to `max_num_noisy_windows` windows of the cached diagonal are denoised in one forward.
Every noisy window attends only CLEAN context (the overlap prefix plus the HieraKV sink and recent
chunks), the same context the rolling KV cache holds at inference.
"""

import gc
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import Dataset

from movieroll.commons.constants import NUM_TRAIN_TIMESTEPS
from movieroll.commons.streaming_config import streaming_config_from_stage_block
from movieroll.optim.muon import Muon
from movieroll.schedulers.scheduling_flow_match_streaming import FlowMatchStreamingScheduler
from movieroll.utils.communications import bcast_in_group, is_group_rank0
from movieroll.utils.parallel_sync import sync_object_for_parallel_training
from movieroll.utils.torch_utils import count_failed_ranks

logger = logging.getLogger(__name__)

# Seed arithmetic of the SP-source generator. The domain constant differs from the SFT stream in
# `trainer/flow/streaming_wrappers.py` so the two cannot collide; changing any value here changes every
# pivot / context-noise draw.
_SP_SOURCE_RANDOM_BASE_SEED = 42
_SP_SOURCE_RANDOM_DOMAIN = 23_337_779
_SP_SOURCE_RANDOM_GROUP_STRIDE = 1_000_003
_SP_SOURCE_RANDOM_COUNTER_STRIDE = 10_007


def _ode_cache_collate(batch):
    """Hand the single sample dict through unchanged (batch_size 1).

    A cache entry holds `rounds`, a list of dicts of tensors, which `default_collate` cannot handle.
    At module level rather than a lambda so it stays picklable with num_workers > 0.
    """
    return batch[0]


class TrajectoryDataset(Dataset):
    """The rolling ODE-init cache: one `{sample_idx:06d}.pt` file per video, holding

        {"prompt": str, "clean_video": [C, total, h, w] (every popped clean chunk, in order),
         "rounds": [{"window_input": [C, T, h, w], "window_timesteps": [T], "start_frame": int} ...],
         "seed": int, "chunk_sizes": [int], "warped_timesteps": [float], "height": int, "width": int}

    The key names must match `movieroll/scripts/precompute_ode.py`.
    """

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.files = sorted([
            f for f in os.listdir(cache_dir)
            if f.endswith('.pt') and '.tmp.' not in f
        ])
        logger.info(f"TrajectoryDataset (rolling): {len(self.files)} trajectories from {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.cache_dir, self.files[idx])
        data = torch.load(path, map_location='cpu', weights_only=False)
        return {
            "text": data.get("prompt", ""),
            "clean_video": data["clean_video"],
            "rounds": data["rounds"],
            "chunk_sizes": data.get("chunk_sizes"),
            "warped_timesteps": data.get("warped_timesteps"),
        }


@dataclass
class ODERegressionConfig:
    denoising_step_list: List[int]             # student N-step raw, e.g. [1000,800,600,400,200]
    # Context noise on the conditioning history: ONE sigma in [0, max] for the whole prefix, where SFT
    # draws one per chunk.
    context_noise_sigma_max: float
    # How many non-overlapping noisy windows one forward may carry.
    max_num_noisy_windows: int


class ODERegressionTrainer:
    """The loss itself: draw one cached rolling round, denoise it teacher-forced, regress the endpoint.

    The round is replayed as a diagonal full-window forward at guidance 1 and its clean endpoint is
    regressed with MSE.
    """

    def __init__(self, scheduler, config, device):
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.num_denoising_steps = len(config.denoising_step_list)
        # The student's N-step warped schedule, which the rolling loop steps through. velocity_to_x0
        # derives its sigma from the timestep it is given, not from this schedule.
        self.scheduler.set_custom_timesteps(config.denoising_step_list, device=device)
        self._source_random_counter = 0
        self._cache_hits = 0
        self.generator_pipeline = None   # the streaming pipeline (student), injected by the main trainer
        logger.info(
            f"[ODE-rolling] student {self.num_denoising_steps}-step warped="
            f"{self.scheduler.timesteps.tolist()}"
        )

    def set_generator_pipeline(self, pipeline):
        self.generator_pipeline = pipeline

    def _get_sp_group(self):
        """The sequence-parallel group, or None when sp_size == 1 (then every rank draws on its own)."""
        if not dist.is_initialized():
            return None
        from movieroll.commons.parallel_states import nccl_info

        if int(getattr(nccl_info, "sp_size", 1) or 1) <= 1:
            return None
        group = getattr(nccl_info, "sp_group", None)
        if group is None:
            raise RuntimeError("Stage2 SP is enabled but nccl_info.sp_group is missing")
        return group

    def _make_sp_source_generator(
        self, sp_group
    ) -> Optional[torch.Generator]:
        """Create an SP-group-specific RNG, on that group's rank 0 only.

        The counter is bumped on EVERY rank before the rank-0 early return, so the per-call seed stays in
        step across the group.
        """

        self._source_random_counter += 1
        if sp_group is not None and not is_group_rank0(sp_group):
            return None
        group_id = (
            int(dist.get_process_group_ranks(sp_group)[0])
            if sp_group is not None
            else (dist.get_rank() if dist.is_initialized() else 0)
        )
        seed = (
            _SP_SOURCE_RANDOM_BASE_SEED
            + _SP_SOURCE_RANDOM_DOMAIN
            + group_id * _SP_SOURCE_RANDOM_GROUP_STRIDE
            + self._source_random_counter * _SP_SOURCE_RANDOM_COUNTER_STRIDE
        )
        return torch.Generator(device="cpu").manual_seed(seed)

    def _build_event_cond(self, conditional_dict):
        extra = {}
        if "byt5_text_states" in conditional_dict:
            extra["byt5_text_states"] = conditional_dict["byt5_text_states"]
            extra["byt5_text_mask"] = conditional_dict["byt5_text_mask"]
        return (
            conditional_dict["encoder_hidden_states"],
            conditional_dict["encoder_attention_mask"],
            None,
            extra,
        )

    def compute_rolling_ode_loss(self, sample, conditional_dict):
        """Draw one rolling pivot from the cached sample, tile its windows, regress the clean endpoint.

        Cached sample: {clean_video: [C, total, h, w], rounds: [{window_input, window_timesteps,
        start_frame}], chunk_sizes}. The clean history before the pivot chunk is the lightly-noised
        conditioning prefix; the matching rounds' window_input, aligned by (start_frame, T), are the noisy
        windows, denoised in one differentiable forward at guidance 1 against the clean_video slice.
        """
        if self.generator_pipeline is None:
            raise RuntimeError(
                "generator_pipeline is not set (call set_generator_pipeline first)"
            )
        clean_video = sample["clean_video"]
        if not torch.is_tensor(clean_video):
            raise ValueError("sample['clean_video'] is missing (a rolling cache entry is required)")
        clean_video = clean_video.to(self.device, dtype=torch.bfloat16)
        if clean_video.dim() == 4:
            clean_video = clean_video.unsqueeze(0)          # [1, C, total, h, w]
        rounds = sample["rounds"]
        chunk_sizes = [int(cs) for cs in sample["chunk_sizes"]]
        num_chunks = len(chunk_sizes)
        ws = int(self.generator_pipeline.streaming_config.window_size)
        max_windows = int(self.config.max_num_noisy_windows)

        # Chunk start frames: frame_offsets[i] = sum(chunk_sizes[:i])
        frame_offsets = [0]
        for cs in chunk_sizes:
            frame_offsets.append(frame_offsets[-1] + cs)

        # (start_frame, expected frame count) -> round. The prefill rounds share start_frame 0 but have
        # different T (5/9/13/17/21) and are independent diagonal states, not prefixes of one another, so
        # they are all indexed by T. A second index keeps the fullest round per start frame as a fallback.
        round_by_start_T: Dict[tuple, Any] = {}
        round_by_start: Dict[int, Any] = {}
        for rd in rounds:
            sf = int(rd["start_frame"])
            T_rd = int(rd["window_input"].shape[1])   # [C, T, h, w] -> T is dim 1
            round_by_start_T.setdefault((sf, T_rd), rd)
            prev = round_by_start.get(sf)
            if prev is None or T_rd > int(prev["window_input"].shape[1]):
                round_by_start[sf] = rd
        # A cache rolled with another chunk or window size does not fail the lookups below -- they fall
        # back to the fullest round per start frame and train on windows of the wrong length -- so check
        # it here. The fullest prefill round (start frame 0) spans the first `ws` chunks.
        chunk_size = int(self.generator_pipeline.streaming_config.chunk_size)
        prefill_T = int(round_by_start[0]["window_input"].shape[1]) if 0 in round_by_start else None
        if any(cs != chunk_size for cs in chunk_sizes[1:]) or (
            num_chunks >= ws and prefill_T != frame_offsets[ws]
        ):
            raise RuntimeError(
                f"the ODE cache was not rolled with chunk_size {chunk_size} and window_size {ws}: its chunk "
                f"sizes are {chunk_sizes} and its fullest prefill round has {prefill_T} frames; precompute the "
                "cache again with this config"
            )

        sp_group = self._get_sp_group()
        source_generator = self._make_sp_source_generator(sp_group)
        # The first window is a real rolling round, drawn UNIFORMLY over all rounds: each round executes
        # once at inference, and the first window is the one carrying that round's real context (the later
        # windows are extra parallelism inference never has). round <-> (c, first_ws) is a bijection:
        #   prefill: c = 0,                first_ws = 1..ws   -> start_frame 0,        T = 5/9/13/17/21
        #   steady : c = 1..num_chunks-1,  first_ws = ws      -> start_frame off[c],   T = 20 (tail clipped)
        # Drawing c uniformly instead would under-train the prefill rounds by about 4x.
        n_prefill = min(ws, num_chunks) if ws > 1 else 1          # first-window sizes available at c = 0
        n_steady = max(0, num_chunks - 1)                          # c = 1..num_chunks-1
        if n_prefill + n_steady <= 0:
            raise RuntimeError("ODE cache contains no selectable rolling round")
        if sp_group is None or is_group_rank0(sp_group):
            if source_generator is None:
                raise RuntimeError("SP source rank is missing its random generator")
            r = int(
                torch.randint(
                    0,
                    n_prefill + n_steady,
                    (1,),
                    generator=source_generator,
                ).item()
            )
        else:
            r = 0
        if r < n_prefill:
            c, first_ws = 0, r + 1                             # prefill: first window is 1..ws chunks
        else:
            c, first_ws = r - n_prefill + 1, ws                # steady: full window
        c_max = max(0, num_chunks - 1)
        if sp_group is not None:
            # Synchronised inside the SP group only: its rank 0 draws, the others just receive.
            c_t = torch.tensor([c, first_ws], device=self.device, dtype=torch.long)
            bcast_in_group(c_t, sp_group)
            c, first_ws = int(c_t[0].item()), int(c_t[1].item())
        c = min(max(0, c), c_max)
        first_ws = min(max(1, first_ws), max(1, min(ws, num_chunks - c)))
        # Chunks per window: [first_ws, ws, ws, ...], up to max_windows. Only the FIRST window may be
        # short; the attention prefix length formula `first + (N-2)*regular` requires the rest to be equal.
        window_chunk_counts = [first_ws]
        used = first_ws
        while len(window_chunk_counts) < max_windows and (num_chunks - c - used) >= ws:
            window_chunk_counts.append(ws)
            used += ws
        K = len(window_chunk_counts)

        noisy_pieces, ts_pieces, target_pieces = [], [], []
        mask_pieces: List[Any] = []                           # whether each frame has a target
        noisy_chunk_sizes: List[List[int]] = []
        kept_window_spans: List[tuple] = []                   # the windows actually kept (cs_start, cs_end)
        total_frames = int(clean_video.shape[2])              # frames in clean_video (61)
        regular_cs = int(chunk_sizes[-1])                     # the regular chunk length (4)
        cs_cursor = c
        for i in range(K):
            cs_start = cs_cursor
            cs_end = min(cs_start + window_chunk_counts[i], num_chunks)
            cs_cursor = cs_end
            f0, f1 = frame_offsets[cs_start], frame_offsets[cs_end]
            sf = frame_offsets[cs_start]
            # Exact match on (start_frame, expected frames), else the fullest round at that start. The
            # tail pivots take the fallback: the cache holds a FULL window that runs past the end of the
            # video, because the teacher keeps rolling full windows while decoding, and those
            # out-of-range frames are real rollout content, not padding.
            rd_i = round_by_start_T.get((sf, f1 - f0)) or round_by_start.get(sf)
            if rd_i is None:                                 # no aligned round (should not happen): skip
                continue
            wi = rd_i["window_input"].to(self.device, dtype=torch.bfloat16)
            if wi.dim() == 4:
                wi = wi.unsqueeze(0)                         # [1, C, T_rd, h, w]
            wts = rd_i["window_timesteps"].to(self.device).float()   # [T_rd]
            # Full window in, target mask out: the window input is NOT clipped to the clean_video
            # boundary, because clipping would cut the high-noise entries and degenerate the diagonal of
            # the last rounds, and inference always rolls a full window while decoding. Only frames that
            # HAVE a target contribute to the loss.
            T_full = int(min(wi.shape[2], wts.shape[0]))
            wi = wi[:, :, :T_full]
            wts = wts[:T_full]
            valid = max(0, min(T_full, total_frames - f0))    # frames that land inside clean_video
            if valid <= 0:                                    # the whole window is out of range: skip
                continue
            tgt = clean_video[:, :, f0:f0 + valid]
            if valid < T_full:
                # Out-of-range frames have no clean target. The zero pad only makes target and prediction
                # the same length so one boolean index handles both; frame_mask is False there, so the
                # padded values never reach the loss.
                pad = torch.zeros(
                    tgt.shape[0], tgt.shape[1], T_full - valid, tgt.shape[3], tgt.shape[4],
                    device=tgt.device, dtype=tgt.dtype,
                )
                tgt = torch.cat([tgt, pad], dim=2)
            m = torch.zeros(T_full, device=self.device, dtype=torch.bool)
            m[:valid] = True
            # This window's chunk split has to cover all of T_full: real chunk_sizes inside the video, and
            # regular-sized chunks beyond it (which is how the teacher kept rolling).
            win_cs = list(chunk_sizes[cs_start:cs_end])
            acc = sum(win_cs)
            while acc < T_full:
                step_cs = min(regular_cs, T_full - acc)
                win_cs.append(step_cs)
                acc += step_cs
            if acc > T_full:                                  # already over inside the video (should not happen)
                win_cs, acc2 = [], 0
                for x in chunk_sizes[cs_start:cs_end]:
                    if acc2 >= T_full:
                        break
                    take = min(int(x), T_full - acc2)
                    win_cs.append(take)
                    acc2 += take
            noisy_pieces.append(wi)
            ts_pieces.append(wts)
            target_pieces.append(tgt)
            mask_pieces.append(m)
            noisy_chunk_sizes.append(win_cs)
            kept_window_spans.append((cs_start, cs_end))

        if not noisy_pieces:
            raise RuntimeError(
                f"no noisy window could be assembled (c={c}, K={K}, num_chunks={num_chunks})"
            )
        K = len(noisy_pieces)
        noisy_latents = torch.cat(noisy_pieces, dim=2)       # [1, C, T_noisy, h, w]
        noisy_timesteps = torch.cat(ts_pieces, dim=0)        # [T_noisy]
        target = torch.cat(target_pieces, dim=2).float()     # [1, C, T_noisy, h, w]
        loss_mask = torch.cat(mask_pieces, dim=0)            # [T_noisy] bool, True = frame has a target

        # cond = clean history [0..c-1] plus the clean overlap of W1..W_{K-1}; the last window WK has no
        # clean conditioning. The first window may be partial, so the overlap end must be the LAST
        # window's start, not c + (K-1)*ws.
        overlap_end_chunk = min(kept_window_spans[-1][0], num_chunks)
        cond_end_frame = frame_offsets[overlap_end_chunk]
        if cond_end_frame > 0:
            cond_latents = clean_video[:, :, :cond_end_frame]   # [1, C, T_cond, h, w]
            cond_chunk_sizes = chunk_sizes[:overlap_end_chunk]
            # Context noise: ONE sigma for the whole prefix, as (1-s)*clean + s*eps with
            # s in [0, context_noise_sigma_max]; the conditioning frames still declare timestep 0.
            # THE ORDER IS LOAD-BEARING: `torch.cat(target_pieces, dim=2).float()` above has already
            # copied the target out of clean_video, and the noising below is out-of-place, so neither the
            # target nor the cache entry is touched. Moving that cat below this block, or noising in
            # place, silently poisons the regression target -- no error, and the loss still falls.
            sigma_max = float(self.config.context_noise_sigma_max)
            if sigma_max > 0:
                if sp_group is None or is_group_rank0(sp_group):
                    if source_generator is None:
                        raise RuntimeError(
                            "SP source rank is missing its random generator"
                        )
                    s = (
                        torch.rand((1,), generator=source_generator).item()
                        * sigma_max
                    )
                    ctx_noise = torch.randn(
                        cond_latents.shape,
                        generator=source_generator,
                        dtype=torch.float32,
                    ).to(self.device, dtype=cond_latents.dtype)
                else:
                    s = 0.0
                    ctx_noise = torch.empty_like(cond_latents)
                if sp_group is not None:
                    s_t = torch.tensor(
                        [s], device=self.device, dtype=torch.float32
                    )
                    bcast_in_group(s_t, sp_group)
                    bcast_in_group(ctx_noise, sp_group)
                    s = float(s_t.item())
                cond_latents = (1.0 - s) * cond_latents + s * ctx_noise
        else:
            cond_latents = None
            cond_chunk_sizes = []

        event_cond = self._build_event_cond(conditional_dict)
        window_x0 = self.generator_pipeline.rolling_ode_teacher_forcing_forward(
            cond_latents=cond_latents,
            noisy_latents=noisy_latents,
            noisy_timesteps=noisy_timesteps,
            cond_chunk_sizes=cond_chunk_sizes,
            noisy_chunk_sizes=noisy_chunk_sizes,
            event_cond=event_cond,
        )
        pred = window_x0.float()
        valid_T = min(pred.shape[2], target.shape[2])
        pred, target = pred[:, :, :valid_T], target[:, :, :valid_T]
        frame_mask = loss_mask[:valid_T].bool()               # [T] True = frame has a clean target
        if pred.shape[1] != target.shape[1]:                 # safety: compare latent channels only
            cc = min(pred.shape[1], target.shape[1])
            pred, target = pred[:, :cc], target[:, :cc]
        # Only frames WITH a target contribute (the tail pivots' full windows run past the end of the
        # video). Boolean-index first, then mse_loss: multiplying by a mask instead would leave the
        # invalid frames' (pred-target)^2 in the graph, where one inf turns the whole loss into nan via
        # 0 * inf.
        if bool(frame_mask.all()):
            loss = F.mse_loss(pred, target, reduction="mean")
        else:
            loss = F.mse_loss(pred[:, :, frame_mask], target[:, :, frame_mask], reduction="mean")
        n_sup = int(frame_mask.sum().item())
        self._cache_hits += 1
        log_dict = {
            "ode_loss": loss.item(),
            "avg_timestep": float(noisy_timesteps.mean().item()),
            "window_x0_norm": float(pred.norm().item()),
            "target_norm": float(target.norm().item()),
            "pivot_c": c,
            "num_windows": K,
            # How many chunks the first window spans INSIDE clean_video; for the tail pivots the cached
            # full window runs past the end, so this reads below ws -- use T_noisy / T_supervised.
            "first_window_chunks": int(kept_window_spans[0][1] - kept_window_spans[0][0]),
            "num_chunks": num_chunks,
            "cond_frames": int(cond_end_frame),
            "T_noisy": int(noisy_latents.shape[2]),
            "T_supervised": n_sup,                           # frames with a target (<= T_noisy)
            "cache_hit": 1,
            "cache_hits_total": self._cache_hits,
            "cache_misses_total": 0,
        }
        return loss, log_dict


class ODERegressionMainTrainer:
    """The Stage-2 training job: models, data, optimizer, the loop, validation and checkpoints."""

    def __init__(self, config: DictConfig, rank: int, world_size: int, device: torch.device):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.is_main = (rank == 0)

        self.step = 0
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32

        self.dp_degree = None
        self.dp_rank = None
        self._last_saved_step = None

        self._setup_logdir()
        self._init_logging()

        self._init_models()
        teacher_only = bool(config.get("precompute_skip_student", False))
        # Teacher-only precompute builds no student, so there is nothing to feed and nothing to optimise.
        if not teacher_only:
            self._init_dataloader()
            self._init_optimizer()

        if config.get("resume_checkpoint", None):
            self._load_checkpoint(config.resume_checkpoint)
        # A cache too short for this topology's validation would otherwise only fail after the first save.
        if not teacher_only:
            self._check_validation_cache()

        logger.info(f"ODERegressionMainTrainer initialized on rank {rank}/{world_size}")

    def _setup_logdir(self):
        """Set up the log directory, timestamped unless this is a resume."""
        from datetime import datetime

        base_logdir = self.config.logdir
        exp_name = self.config.get("exp_name", "ode_regression")

        resume_checkpoint = self.config.get("resume_checkpoint", None)
        if resume_checkpoint:
            checkpoint_path = Path(resume_checkpoint)
            if not checkpoint_path.is_dir():
                raise FileNotFoundError(
                    f"Stage2 resume checkpoint does not exist: {checkpoint_path}"
                )
            if checkpoint_path.parent.name != "checkpoints":
                raise ValueError(
                    "Stage2 resume_checkpoint must be a step directory directly "
                    f"below checkpoints/, got {checkpoint_path}"
                )
            timestamped_logdir = str(checkpoint_path.parent.parent)
            if self.is_main:
                logger.info(
                    "Resuming training - using existing experiment directory: "
                    f"{timestamped_logdir}"
                )
        elif self.is_main:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_logdir = os.path.join(base_logdir, f"{exp_name}_{timestamp}")
        else:
            timestamped_logdir = None

        if dist.is_initialized():
            timestamped_logdir_list = [timestamped_logdir]
            dist.broadcast_object_list(timestamped_logdir_list, src=0)
            timestamped_logdir = timestamped_logdir_list[0]

        self.config.logdir = timestamped_logdir
        if self.is_main:
            logger.info(f"Experiment directory: {timestamped_logdir}")

    def _init_logging(self):
        if self.is_main:
            os.makedirs(self.config.logdir, exist_ok=True)
            os.makedirs(os.path.join(self.config.logdir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(self.config.logdir, "samples"), exist_ok=True)

    def _init_models(self):
        logger.info("Initializing models...")
        logger.info("Loading VAE...")
        self._load_vae()
        logger.info("Loading text encoders...")
        self._load_text_encoders()
        logger.info("Loading transformer models...")
        self._load_transformers()

        if self.world_size > 1:
            # Same FSDP split as the SFT trainer (hybrid: shard inside a node, replicate across) plus SP.
            logger.info("Applying FSDP2 (SFT-consistent hybrid shard) + SP...")
            self._apply_fsdp_wrapping()
        else:
            if self.student is not None:
                self.student.to(self.device)
            if self.teacher is not None:
                self.teacher.to(self.device)

        logger.info("Creating scheduler (FlowMatchStreamingScheduler for rolling)...")
        self.scheduler = FlowMatchStreamingScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            shift=self.config.ode.timestep_shift,
        )

        ode_config = ODERegressionConfig(
            denoising_step_list=list(self.config.ode.denoising_step_list),
            context_noise_sigma_max=self.config.ode.context_noise_sigma_max,
            max_num_noisy_windows=self.config.ode.max_num_noisy_windows,
        )

        self.ode_trainer = ODERegressionTrainer(
            scheduler=self.scheduler,
            config=ode_config,
            device=self.device,
        )

        logger.info("Rolling ODE Regression trainer created")

        # The rolling pipeline drives both the student's ODE forward and 5-step rolling sampling.
        self._create_sample_pipeline()
        self.ode_trainer.set_generator_pipeline(self.generator_pipeline)
        if self.is_main:
            logger.info(
                "[ODE-rolling] pivot/context randomness: "
                "SP group rank0 draws and explicitly broadcasts tensors"
            )

    def _create_sample_pipeline(self):
        """Build the Stage2 ODE rolling pipeline."""
        from movieroll.pipelines.movieroll_ode_pipeline import MovieRollODEPipeline

        pipeline_transformer = self.student if self.student is not None else self.teacher
        self.generator_pipeline = MovieRollODEPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            transformer=pipeline_transformer,
            scheduler=self.scheduler,
            precision="bf16",
            vae_precision=self.config.get("vae_precision", "fp16"),
            prompt_format_font_path=self.config.multilingual_prompt_format_font_path,
            prompt_format_color_path=self.config.multilingual_prompt_format_color_path,
            byt5_kwarg=self.byt5_kwarg,
            text_len=int(self.config.text_len),
            device=self.device,
            is_eval=True,
            streaming_config=self.streaming_config,
        )
        logger.info("Rolling pipeline (MovieRollODEPipeline) created")

    def _load_vae(self):
        """Load the VAE. Stage2 uses the NON-streaming class: validation takes latents from the pipeline
        and decodes them in `_decode_and_save_video`, which turns both tilings on."""
        from movieroll.models.autoencoders.vae_loader import load_vae

        vae_precision = self.config.get("vae_precision", "fp16")

        self.vae, vae_path, s_ratio, t_ratio = load_vae(
            vae_precision=vae_precision,
            vae_path=self.config.vae_path,
            logger=logger,
            device=self.device,
            streaming=False,
        )
        self.vae.eval()
        self.vae.requires_grad_(False)
        logger.info(f"VAE loaded from {vae_path}")
        logger.info(f"  Spatial ratio: {s_ratio}, Temporal ratio: {t_ratio}")

    def _load_text_encoders(self):
        """Load the text encoder and the glyph byT5 branch."""
        from movieroll.models.text_encoders import PROMPT_TEMPLATE
        from movieroll.models.text_encoders.byT5 import load_glyph_byT5_v2
        from movieroll.utils.load import load_text_encoder

        text_encoder_cpu_offload = self.config.fsdp.text_encoder_cpu_offload
        if self.world_size > 1 and text_encoder_cpu_offload:
            text_encoder_device = torch.device("cpu")
            if self.is_main:
                logger.info("Loading text encoders on CPU (for FSDP CPU offload)")
        else:
            text_encoder_device = self.device

        self.text_encoder = load_text_encoder(
            self.config.text_encoder_path,
            self.config.text_len,
            PROMPT_TEMPLATE["li-dit-encode-image-json"],
            PROMPT_TEMPLATE["li-dit-encode-video-json"],
            text_encoder_device,
            text_encoder_precision=self.config.get("text_encoder_precision", "fp16"),
            logger=logger,
        )
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        logger.info("Text encoder loaded")

        self.byt5_kwarg = load_glyph_byT5_v2(
            self.device,
            byT5_google_path=self.config.byt5_google_path,
            byT5_ckpt_path=self.config.byt5_ckpt_path,
            color_ann_path=self.config.multilingual_prompt_format_color_path,
            font_ann_path=self.config.multilingual_prompt_format_font_path,
            byt5_max_length=self.config.byt5_max_length,
        )
        self.byt5_kwarg["byt5_model"].eval()
        logger.info(f"ByT5 loaded with max_length={self.config.get('byt5_max_length', 256)}")

        logger.info("Text encoders loaded")

    def _load_transformers(self):
        """Load the transformer this run needs: the teacher alone when precompute skips the student,
        otherwise the student alone."""
        from movieroll.utils.load import load_transformer

        # The ODE student and the DMD generator are built with exactly the same streaming configuration,
        # so their weights stay interchangeable across the two stages.
        streaming_config = streaming_config_from_stage_block(self.config.streaming_config)
        self.streaming_config = streaming_config
        logger.info(
            f"ODE StreamingConfig: chunk={streaming_config.chunk_size}, "
            f"window={streaming_config.window_size}"
        )

        def create_streaming_model(checkpoint_path):
            """Build one streaming transformer and load its checkpoint strictly."""
            checkpoint_path = Path(str(checkpoint_path))
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            model = load_transformer(
                self.config.model_type,
                str(checkpoint_path),
                {"device": None, "dtype": self.dtype, "streaming_config": streaming_config},
                logger,
            )
            # Load-bearing: with patch_size [1,1,1] the FinalLayer list branch builds its Linear without
            # the factory kwargs, so this is what puts `final_layer.linear` in the model dtype.
            model = model.to(dtype=self.dtype)
            logger.info(f"Checkpoint loaded and converted to {self.dtype}")

            return model

        student_ckpt = self.config.student_ckpt
        teacher_ckpt = self.config.teacher_ckpt

        # Either entry mode needs exactly one large transformer: precompute uses the frozen teacher,
        # cache-only training uses the student.
        teacher_only = bool(self.config.get("precompute_skip_student", False))
        if teacher_only:
            self.student = None
            logger.info("Teacher-only precompute: student allocation skipped")
            logger.info(f"Teacher checkpoint: {teacher_ckpt}")
            self.teacher = create_streaming_model(teacher_ckpt)
            self.teacher.requires_grad_(False)
            self.teacher.eval()
            if self.is_main:
                param_count = sum(p.numel() for p in self.teacher.parameters()) / 1e9
                logger.info(f"  Teacher loaded (frozen), params: {param_count:.2f}B")
        else:
            logger.info(f"Student checkpoint: {student_ckpt}")
            self.student = create_streaming_model(student_ckpt)
            self.student.requires_grad_(True)
            self.student.train()
            if self.is_main:
                param_count = sum(p.numel() for p in self.student.parameters()) / 1e9
                logger.info(f"  Student loaded (trainable), params: {param_count:.2f}B")
            self.teacher = None
            logger.info("Cache-only Stage2 training: teacher allocation skipped")

    def _apply_fsdp_wrapping(self):
        """Resolve the sharding topology (including the hybrid -> full fallback), build the device mesh and the
        sequence-parallel state, then apply activation checkpointing and FSDP2."""
        from movieroll.utils.fsdp import apply_fsdp2, apply_fsdp_checkpointing, fsdp_device_mesh
        from movieroll.utils.load import get_no_split_modules
        from movieroll.commons.parallel_states import ParallelDims, initialize_sequence_parallel_state_v2

        sp_size = self.config.sp_size
        fsdp_sharding_strategy = self.config.fsdp.sharding_strategy
        shard_world_size = int(self.config.fsdp.get("shard_world_size", 8))

        if fsdp_sharding_strategy == 'full':
            dp_replicate = 1
        else:
            if fsdp_sharding_strategy != "hybrid":
                raise ValueError(
                    "Stage2 fsdp.sharding_strategy must be 'full' or 'hybrid', "
                    f"got {fsdp_sharding_strategy!r}"
                )
            dp_replicate = self.world_size // shard_world_size

        if self.world_size == shard_world_size and fsdp_sharding_strategy == 'hybrid':
            fsdp_sharding_strategy = "full"
            dp_replicate = 1
            if self.is_main:
                logger.info(
                    f"Single shard group ({shard_world_size} GPUs): falling back hybrid to full sharding"
                )

        parallel_dims = ParallelDims(
            dp_replicate=dp_replicate,
            dp_shard=-1,
            sp=sp_size,
            world_size=self.world_size,
        )

        world_mesh = parallel_dims.build_mesh(device_type="cuda")
        self.parallel_dims = parallel_dims

        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            self.dp_degree = dp_mesh.size()
            self.dp_rank = dp_mesh.get_local_rank()
        else:
            self.dp_degree = 1
            self.dp_rank = 0

        initialize_sequence_parallel_state_v2(parallel_dims, world_mesh)
        if self.is_main:
            logger.info("Streaming parallel states initialized")

        if self.is_main:
            logger.info(f"Parallelism: sharding={fsdp_sharding_strategy}, SP={sp_size}, DP={self.dp_degree}")

        reference_model = self.student if self.student is not None else self.teacher
        no_split_modules = get_no_split_modules(reference_model)
        dp_mesh = fsdp_device_mesh(parallel_dims, world_mesh, fsdp_sharding_strategy)
        param_dtype = torch.float32 if not self.config.mixed_precision else torch.bfloat16
        reduce_dtype = torch.float32

        # Activation checkpointing is applied BEFORE apply_fsdp2, so FSDP2 wraps the checkpoint wrapper
        # and the all-gather stays outside the recomputed region.
        if self.student is not None:
            if bool(self.config.fsdp.student_checkpointing):
                apply_fsdp_checkpointing(self.student, no_split_modules, p=1)
            apply_fsdp2(
                model=self.student,
                dp_mesh=dp_mesh,
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cpu_offload=bool(self.config.fsdp.student_cpu_offload),
            )
            if self.is_main:
                logger.info("  Student wrapped with FSDP2")

        if self.teacher is not None:
            no_split_modules_teacher = get_no_split_modules(self.teacher)
            if bool(self.config.fsdp.teacher_checkpointing):
                apply_fsdp_checkpointing(self.teacher, no_split_modules_teacher, p=1)
            apply_fsdp2(
                model=self.teacher,
                dp_mesh=dp_mesh,
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cpu_offload=bool(self.config.fsdp.teacher_cpu_offload),
            )
            if self.is_main:
                logger.info("  Teacher wrapped with FSDP2")

        if dist.is_initialized():
            dist.barrier()

    def _init_dataloader(self):
        data_config = self.config.data
        cache_only = self.config.ode.cache_only
        cache_dir = self.config.ode.trajectory_cache_dir

        if not cache_only or not cache_dir:
            raise ValueError(
                "Stage2 trainer supports only cache_only=true with "
                "ode.trajectory_cache_dir; use movieroll/scripts/precompute_ode.py "
                "to create the cache"
            )
        logger.info(f"Using TrajectoryDataset (cache_only): {cache_dir}")
        dataset = TrajectoryDataset(cache_dir)
        if len(dataset) == 0:
            raise ValueError(f"Stage2 trajectory cache is empty: {cache_dir}")

        from torch.utils.data import DataLoader
        collate_fn = _ode_cache_collate
        # DP sharding: each dp rank sees only its own slice, otherwise every rank shuffles independently
        # and two of them can draw the same entry in one step. Inside an SP group every rank must see the
        # SAME entry, because rolling_ode_teacher_forcing_forward is sequence-parallel -- hence the split
        # is by dp_rank / dp_degree and not by global rank. Both are already set by _apply_fsdp_wrapping:
        # they start as None, and the `or 1` below would silently turn an early call into no sharding.
        sampler = None
        dp_degree = int(getattr(self, "dp_degree", 1) or 1)
        dp_rank = int(getattr(self, "dp_rank", 0) or 0)
        if dp_degree > 1:
            from torch.utils.data import DistributedSampler
            sampler = DistributedSampler(
                dataset, num_replicas=dp_degree, rank=dp_rank,
                shuffle=True, seed=int(self.config.seed), drop_last=False,
            )
            logger.info(f"DistributedSampler: dp_rank={dp_rank}/{dp_degree} (one entry per SP group)")
        self.data_sampler = sampler
        self.dataloader = DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=data_config.num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        self.data_iter = iter(self.dataloader)

        logger.info(f"Dataloader created: {len(dataset)} samples, batch_size={data_config.batch_size}")

    def _check_validation_cache(self):
        """Fail up front when a validation pass of this run would read a missing trajectory cache entry.

        `_validate` hands every dp replica `n_sample_each_rank` entries of `validation.prompt_indices`
        round-robin, so which entries a run needs depends on the dp degree. Checked only when a validation
        will run, over the union of all dp replicas, and agreed across ranks so they stop together. The
        two validation defaults read here are copies of `train`'s and must stay in step with them.
        """
        max_steps = int(self.config.max_steps)
        save_interval = int(self.config.save_interval)
        infer_only = bool(self.config.get("infer_only", False))
        at_start = bool(self.config.get("validate_at_start", False))
        on_save = (
            bool(self.config.get("validate_on_save", True))
            and max_steps // save_interval > self.step // save_interval
        )
        if not (infer_only or at_start or on_save):
            return
        val_cfg = self.config.validation
        idx_list = [int(i) for i in val_cfg.prompt_indices]
        n_each = int(val_cfg.n_sample_each_rank)
        if not idx_list or n_each <= 0:
            return                                       # `_validate` raises its own error for these
        dp_degree = int(getattr(self, "dp_degree", 1) or 1)
        needed = sorted({
            idx_list[(dp_rank + slot * dp_degree) % len(idx_list)]
            for dp_rank in range(dp_degree)
            for slot in range(n_each)
        })
        cache_dir = self.config.ode.get("trajectory_cache_dir", None)
        files = set()
        if cache_dir and os.path.isdir(cache_dir):
            files = {f for f in os.listdir(cache_dir) if f.endswith(".pt") and ".tmp." not in f}
        missing = [idx for idx in needed if f"{idx:06d}.pt" not in files]

        failure_count = count_failed_ranks(bool(missing), self.device)
        if missing:
            if infer_only:
                switch = "; an infer_only run is nothing but this validation"
            else:
                on = [n for n, v in (("validate_at_start", at_start), ("validate_on_save", on_save)) if v]
                switch = f", or turn {' and '.join(on)} off"
            raise FileNotFoundError(
                f"Stage2 validation with {dp_degree} data-parallel rank(s) x {n_each} prompt(s) reads "
                f"{len(needed)} trajectory cache entries {needed}, but {len(missing)} of them are missing "
                f"from {cache_dir}: {missing}. Precompute at least the first {max(needed) + 1} prompts, "
                f"or shorten validation.prompt_indices{switch}."
            )
        if failure_count:
            raise FileNotFoundError(
                f"Stage2 validation cache check failed on {failure_count} peer rank(s)"
            )

    def _init_optimizer(self):
        optimizer_config = self.config.optimizer

        lr = optimizer_config.lr
        weight_decay = optimizer_config.weight_decay

        # Same construction as the SFT trainer. Muon defaults adamw_betas to (0.95, 0.95), so the AdamW
        # branch values are passed explicitly.
        # tuple() also turns the YAML ListConfig into a native type: the betas end up in param_groups,
        # and distributed checkpointing rejects config objects there.
        adamw_betas = tuple(optimizer_config.adamw_betas)
        adamw_eps = optimizer_config.adamw_eps

        params_to_optimize = [p for p in self.student.parameters() if p.requires_grad]
        muon_params = [p for p in params_to_optimize if p.ndim >= 2]
        adamw_params = [p for p in params_to_optimize if p.ndim < 2]
        self.optimizer = Muon(
            lr=lr,
            wd=weight_decay,
            muon_params=muon_params,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adamw_params=adamw_params,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        logger.info(
            f"Muon optimizer created: lr={lr}, weight_decay={weight_decay}, "
            f"adamw_betas={adamw_betas}, adamw_eps={adamw_eps}, "
            f"{len(muon_params)} muon params / {len(adamw_params)} adamw params"
        )

    def _load_checkpoint(self, checkpoint_path: str):
        """Load Stage2 model, optimizer, and step state."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Stage2 checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        student_ckpt_path = checkpoint_path / "student.safetensors"
        optimizer_ckpt_path = checkpoint_path / "optimizer.pt"
        state_ckpt_path = checkpoint_path / "training_state.pt"
        missing = [
            str(path)
            for path in (student_ckpt_path, optimizer_ckpt_path, state_ckpt_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Stage2 checkpoint payload is incomplete: {missing}"
            )

        training_state = torch.load(
            state_ckpt_path, map_location="cpu", weights_only=False
        )
        if "step" not in training_state:
            raise KeyError(f"Stage2 training state has no 'step': {state_ckpt_path}")
        format_version = training_state.get("checkpoint_format_version")
        optimizer_impl = training_state.get("optimizer_impl")
        if format_version != 2 or optimizer_impl != "muon":
            raise RuntimeError(
                f"Stage2 checkpoint {checkpoint_path} records checkpoint_format_version={format_version!r}, "
                f"optimizer_impl={optimizer_impl!r}: only checkpoints written by this trainer "
                "(version 2, optimizer 'muon') can be resumed."
            )
        if not (checkpoint_path / "COMPLETE").is_file():
            raise FileNotFoundError(
                f"Stage2 checkpoint is incomplete (missing COMPLETE): {checkpoint_path}"
            )
        saved_world_size = training_state.get("world_size")
        if saved_world_size is not None and int(saved_world_size) != self.world_size:
            raise RuntimeError(
                "Stage2 checkpoint world size mismatch: "
                f"saved={saved_world_size}, current={self.world_size}"
            )

        from safetensors.torch import load_file
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_state_dict,
        )
        # `_save_checkpoint` writes ONE unsharded full state dict to shared storage, so
        # every rank can read the same file. Loading on all ranks keeps this collective-free;
        # the rank-0 + `broadcast_from_rank0` variant deadlocks here, because only rank 0
        # holds tensors while every rank must enter the same NCCL broadcasts in the same order.
        student_state = load_file(str(student_ckpt_path))
        optimizer_state = torch.load(
            optimizer_ckpt_path, map_location="cpu", weights_only=False
        )
        if self.world_size > 1:
            from movieroll.utils.checkpoint import _materialize_optimizer_state

            # The per-parameter state has to be materialised first: Muon writes a `use_muon` marker at
            # construction, so torch's `_init_optim_state` sees a non-empty `optim.state`, returns early
            # and never runs the lr=0 step that allocates the momentum tensors. Without it,
            # `_load_optim_state_dict` asserts on the empty local state dict it infers the device from,
            # and the handed-over state dict has no momentum key -- which DCP ignores silently, zeroing
            # the momentum.
            _materialize_optimizer_state(self.optimizer)
            set_state_dict(
                self.student,
                self.optimizer,
                model_state_dict=student_state,
                optim_state_dict=optimizer_state,
                options=StateDictOptions(
                    # Every rank already holds the full state dict (read above), so no
                    # broadcast_from_rank0: each rank shards the full tensors locally.
                    full_state_dict=True,
                    strict=True,
                ),
            )
        else:
            # One rank: the student is not FSDP-wrapped, so the state dicts load back directly.
            # Optimizer.load_state_dict maps the state by parameter order, so it reads both formats a
            # one-rank save can have written: the plain state dict of a launch without a process group
            # and the FQN-keyed one of a one-process torchrun launch.
            self.student.load_state_dict(student_state, strict=True)
            self.optimizer.load_state_dict(optimizer_state)
        restored = sum(
            1
            for state in self.optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value) and value.dim() > 0
        )
        if restored == 0:
            # An empty state means the momentum was not restored, and that does not raise on its own.
            raise RuntimeError(
                "Stage2 resume: the optimizer state is empty, so the momentum was not restored. "
                "Check that _materialize_optimizer_state applies to this optimizer."
            )
        logger.info(f"Stage2 optimizer state restored: {restored} tensors")

        logger.info("Student model and optimizer loaded from checkpoint")
        self.step = int(training_state["step"])
        self._last_saved_step = self.step
        logger.info(f"Training state loaded: step={self.step}")

    def _save_checkpoint(self, step: int):
        """Save a complete full-state checkpoint; publish COMPLETE last."""
        from safetensors.torch import save_file

        checkpoint_dir = Path(self.config.logdir) / "checkpoints" / f"step-{step}"
        if self.is_main:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "COMPLETE").unlink(missing_ok=True)
        if dist.is_initialized():
            dist.barrier()

        if dist.is_initialized():
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_state_dict,
            )

            student_state, optimizer_state = get_state_dict(
                self.student,
                self.optimizer,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        else:
            student_state = {
                key: value.detach().cpu().clone()
                for key, value in self.student.state_dict().items()
            }
            optimizer_state = self.optimizer.state_dict()

        save_error = None
        if self.is_main:
            try:
                save_file(
                    {
                        key: value.detach().cpu().contiguous()
                        for key, value in student_state.items()
                    },
                    checkpoint_dir / "student.safetensors",
                )
                torch.save(optimizer_state, checkpoint_dir / "optimizer.pt")
                torch.save(
                    {
                        "step": int(step),
                        "checkpoint_format_version": 2,
                        # Which optimizer wrote the momentum; `_load_checkpoint` requires this value.
                        "optimizer_impl": "muon",
                        "world_size": int(self.world_size),
                        "config": self.config,
                    },
                    checkpoint_dir / "training_state.pt",
                )
                (checkpoint_dir / "COMPLETE").write_text(
                    f"step={int(step)}\n", encoding="utf-8"
                )
                logger.info(f"Checkpoint saved: {checkpoint_dir}")
            except Exception as exc:
                save_error = exc
        del student_state, optimizer_state

        failure_count = count_failed_ranks(save_error is not None, self.device)
        if failure_count:
            if save_error is not None:
                raise RuntimeError("Stage2 checkpoint save failed") from save_error
            raise RuntimeError(
                f"Stage2 checkpoint save failed on {failure_count} peer rank(s)"
            )

    def _get_next_batch(self) -> Dict[str, Any]:
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # DistributedSampler needs set_epoch every pass, or every epoch draws the same order.
            self._data_epoch = getattr(self, "_data_epoch", 0) + 1
            if getattr(self, "data_sampler", None) is not None:
                self.data_sampler.set_epoch(self._data_epoch)
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        if self.world_size > 1:
            batch = sync_object_for_parallel_training(
                batch,
                parallel_dims=getattr(self, 'parallel_dims', None)
            )

        return batch

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute one rolling ODE-init training step.

        batch is one cached rolling sample: {text, clean_video, rounds, chunk_sizes, warped_timesteps}.
        """
        import time
        t0 = time.time()

        self.optimizer.zero_grad()

        prompt = batch.get("text", batch.get("prompts", "A video"))
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if len(prompt) else "A video"

        if self.step == 0 and self.is_main:
            cv = batch.get("clean_video")
            logger.info(f"[Step 0] clean_video: {tuple(cv.shape) if torch.is_tensor(cv) else None}, "
                        f"#rounds={len(batch.get('rounds', []))}, prompt: {str(prompt)[:80]}")
            logger.info(f"[Step 0] student warped timesteps: {self.ode_trainer.scheduler.timesteps.tolist()}")

        # Positive prompt only: guidance is 1, so there is no unconditional branch -- CFG is already
        # distilled into the teacher's trajectory.
        t1 = time.time()
        with torch.no_grad():
            conditional_dict = self._encode_prompts_via_pipeline([prompt])
        t_encode = time.time() - t1

        t2 = time.time()
        loss, log_dict = self.ode_trainer.compute_rolling_ode_loss(
            sample=batch,
            conditional_dict=conditional_dict,
        )
        t_ode = time.time() - t2
        local_bad_loss = int(not bool(torch.isfinite(loss.detach()).all()))
        bad_loss_count = count_failed_ranks(local_bad_loss, self.device)
        if bad_loss_count:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"non-finite Stage2 ODE loss at step {self.step} on "
                f"{bad_loss_count} rank(s)"
            )

        t3 = time.time()
        loss.backward()
        t_backward = time.time() - t3

        max_grad_norm = self.config.max_grad_norm_student
        if max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), max_grad_norm
            )
            grad_norm_value = (
                float(grad_norm.full_tensor().item())
                if hasattr(grad_norm, "full_tensor")
                else float(grad_norm.item())
            )
            log_dict["grad_norm"] = grad_norm_value
            local_bad_grad = int(not math.isfinite(grad_norm_value))
            bad_grad_count = count_failed_ranks(local_bad_grad, self.device)
            if bad_grad_count:
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite Stage2 ODE gradient norm at step {self.step} "
                    f"on {bad_grad_count} rank(s)"
                )

        self.optimizer.step()
        t_total = time.time() - t0

        log_dict["time_encode"] = t_encode
        log_dict["time_ode"] = t_ode
        log_dict["time_backward"] = t_backward
        log_dict["time_total"] = t_total
        if self.step % self.config.get("progress_report_interval", 10) == 0:
            log_dict["gpu_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9

        return log_dict

    def _encode_prompts_via_pipeline(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode text prompts with the text encoder and the glyph branch.

        The glyph branch goes through the pipeline's own byT5 encoding, so quoted glyph text is read
        exactly as inference reads it.
        """
        conditional_dict = {}

        with torch.no_grad():
            text_tokens = self.text_encoder.text2tokens(prompts, data_type="video")
            text_outputs = self.text_encoder.encode(text_tokens, data_type="video", device=self.device)
            conditional_dict["encoder_hidden_states"] = text_outputs.hidden_state
            conditional_dict["encoder_attention_mask"] = text_outputs.attention_mask

        byt5_embeds, byt5_mask = self.generator_pipeline._process_single_byt5_prompt(prompts[0], self.device)
        conditional_dict["byt5_text_states"] = byt5_embeds
        conditional_dict["byt5_text_mask"] = byt5_mask

        return conditional_dict

    def train(self):
        logger.info("Starting training...")

        max_steps = self.config.max_steps
        save_interval = self.config.save_interval
        log_interval = self.config.get("progress_report_interval", 10)
        # One validation before training, so step 0 is the untrained reference for the later checkpoints.
        if self.config.get("validate_at_start", False):
            self._validate(0)
        elif self.is_main:
            logger.info("validate_at_start=false -> skipping the step-0 validation")
        while self.step < max_steps:
            batch = self._get_next_batch()
            log_dict = self.train_step(batch)

            if self.step % log_interval == 0 and self.is_main:
                loss_val = log_dict.get("ode_loss", 0)
                grad_val = log_dict.get("grad_norm", 0)
                t_val = log_dict.get("avg_timestep", 0)
                t_total = log_dict.get("time_total", 0)
                gpu_mem = log_dict.get("gpu_mem_gb", 0)
                x0_norm = log_dict.get("window_x0_norm", 0)
                cache_hit = log_dict.get("cache_hit", 0)
                cache_hits = log_dict.get("cache_hits_total", 0)
                cache_misses = log_dict.get("cache_misses_total", 0)

                logger.info(
                    f"[Step {self.step}/{max_steps}] "
                    f"loss={loss_val:.6f} grad={grad_val:.4f} t={t_val:.0f} "
                    f"x0_norm={x0_norm:.2f} | "
                    f"time={t_total:.1f}s mem={gpu_mem:.1f}GB "
                    f"cache={'HIT' if cache_hit else 'GEN'} "
                    f"hit/miss={cache_hits}/{cache_misses}"
                )

            # Save a checkpoint, and validate right after it (every rank runs one prompt in parallel)
            if (self.step + 1) % save_interval == 0:
                self._save_checkpoint(self.step + 1)
                self._last_saved_step = self.step + 1
                if self.config.get("validate_on_save", True):
                    self._validate(self.step + 1)

            if self.step % self.config.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()

            self.step += 1

        logger.info("Training completed!")

        # Avoid rewriting a multi-gigabyte checkpoint when max_steps lands
        # exactly on save_interval.
        if self._last_saved_step != self.step:
            self._save_checkpoint(self.step)
            self._last_saved_step = self.step

    def _validate(self, step: int):
        """One validation pass per checkpoint: each DP replica runs a different prompt.

        Every rank of an SP group must run the SAME prompt and shape and complete the forward together;
        only `rank_within_spgroup == 0` decodes and writes. Each DP replica gets exactly
        `validation.n_sample_each_rank` indices, strided by rank and wrapped around, so the replicas
        always agree on how many prompts they run, however few prompts there are.
        """
        val_cfg = self.config.validation
        idx_list = [int(i) for i in val_cfg.prompt_indices]
        if not idx_list:
            raise ValueError("Stage2 validation.prompt_indices must not be empty")

        dp_rank = int(getattr(self, "dp_rank", 0) or 0)
        dp_degree = int(getattr(self, "dp_degree", 1) or 1)
        # One sample is shared by every rank in an SP group, so validation work is partitioned by the
        # data-parallel coordinate, not by global rank.
        n_each = int(val_cfg.n_sample_each_rank)
        if n_each <= 0:
            raise ValueError("validation.n_sample_each_rank must be > 0")
        # Every rank in an FSDP shard group must execute the same number of forwards, which the
        # round-robin with wraparound guarantees even when len(idx_list) is not divisible by dp_degree.
        my_idx = [
            idx_list[(dp_rank + slot * dp_degree) % len(idx_list)]
            for slot in range(n_each)
        ]

        from movieroll.commons.parallel_states import nccl_info

        save_local = int(getattr(nccl_info, "rank_within_spgroup", 0)) == 0

        val_dir = os.path.join(self.config.logdir, "validation", f"step_{step}")
        if self.is_main:
            os.makedirs(val_dir, exist_ok=True)
            logger.info(
                f"[Val step={step}] {dp_degree} DP replicas running in parallel, "
                f"{len(idx_list)} prompts total ({n_each} per replica): {idx_list}"
            )
        if dist.is_initialized():
            dist.barrier()
        os.makedirs(val_dir, exist_ok=True)

        for cache_idx in my_idx:
            local_error = None
            try:
                self._run_inference(
                    step,
                    cache_index=int(cache_idx),
                    out_dir=val_dir,
                    tag=f"dp{dp_rank:04d}_idx{int(cache_idx):06d}",
                    save_local=save_local,
                )
            except Exception as exc:
                local_error = exc

            failure_count = count_failed_ranks(local_error is not None, self.device)
            if failure_count:
                if local_error is not None:
                    raise RuntimeError(
                        f"Stage2 validation failed on {failure_count} rank(s)"
                    ) from local_error
                raise RuntimeError(
                    f"Stage2 validation failed on {failure_count} peer rank(s)"
                )
        if self.is_main:
            logger.info(f"[Val step={step}] done -> {val_dir}")

    def _run_inference(self, step: int, cache_index: int, out_dir: str, tag: str,
                       save_local: bool):
        """One prompt's 5-step rolling inference (student, guidance 1), driven through the pipeline's
        standard call so the few-step rolling inference (prefill included) is the real one.

        Each rank runs its own prompt from cache entry `cache_index` and writes its own file; there is no
        barrier inside -- the barrier lives in `_validate`.
        """
        sample_dir = out_dir
        if save_local:
            os.makedirs(sample_dir, exist_ok=True)

        was_training = bool(self.student.training)
        self.student.eval()
        try:
            sampling = self.config.sampling
            prompt, height, width = self._get_train_sample_prompt(cache_index)
            video_length = int(sampling.video_length)
            denoising_step_list = list(self.config.ode.denoising_step_list)
            seed = self.config.seed + step

            self.generator_pipeline.transformer = self.student
            if save_local:
                logger.info(f"[Sample step={step}] {tag}: 5-step rolling (guidance=1, no CFG/uncond), "
                            f"prompt: {prompt[:70]}, {video_length}x{height}x{width}, "
                            f"steps={denoising_step_list}")
            # The base __call__'s _make_noise draws with torch.randn(device=cuda, generator=...), so the
            # generator must be a CUDA one. Only SP rank 0 holds the group-specific source generator;
            # the random tensors are broadcast by the pipeline itself.
            from movieroll.commons.parallel_states import nccl_info
            sp_size = int(getattr(nccl_info, "sp_size", 1) or 1)
            is_sp_rank0 = int(
                getattr(nccl_info, "rank_within_spgroup", 0)
            ) == 0
            group_id = (
                int(dist.get_process_group_ranks(nccl_info.sp_group)[0])
                if sp_size > 1
                else (dist.get_rank() if dist.is_initialized() else 0)
            )
            gen = (
                torch.Generator(device=self.device).manual_seed(
                    seed
                    + cache_index
                    + group_id * _SP_SOURCE_RANDOM_GROUP_STRIDE
                )
                if is_sp_rank0
                else None
            )
            with torch.no_grad():
                try:
                    # No negative_prompt on purpose: the ODE-init student is a single-condition model,
                    # with CFG distilled into the weights, and the shipped config sets use_cfg
                    # false, so there is no unconditional branch.
                    out = self.generator_pipeline(
                        prompt=prompt, height=height, width=width, video_length=video_length,
                        num_inference_steps=len(denoising_step_list),
                        timesteps=denoising_step_list,
                        use_cfg=bool(sampling.use_cfg),
                        output_type="latent", generator=gen,
                    )
                finally:
                    if hasattr(self.student, "kv_cache_manager"):
                        self.student.kv_cache_manager.clear_kv_cache()
            latents = out.videos if hasattr(out, "videos") else out

            if save_local:
                logger.info(f"  [{tag}] student latent: {tuple(latents.shape)}, "
                            f"mean={latents.float().mean():.4f}, std={latents.float().std():.4f}")
                self._decode_and_save_video(latents, os.path.join(sample_dir, f"{tag}.mp4"))
                with open(os.path.join(sample_dir, f"{tag}.prompt.txt"), "w") as f:
                    f.write(prompt)
        finally:
            self.student.train(was_training)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if save_local:
            logger.info(f"Sampling done at step {step} ({tag}), saved to {sample_dir}")

    def _get_train_sample_prompt(self, cache_index: int):
        """Take a real training prompt out of the ODE trajectory cache: (prompt, height, width).

        The resolution comes from cache entry `cache_index` itself, not from the config. Reading only the
        metadata still means a torch.load of the whole file, which is negligible next to the sampling.
        """
        cache_dir = self.config.ode.get("trajectory_cache_dir", None)
        if not cache_dir or not os.path.isdir(cache_dir):
            raise FileNotFoundError(
                f"Stage2 validation trajectory cache does not exist: {cache_dir}"
            )
        files = sorted(
            f
            for f in os.listdir(cache_dir)
            if f.endswith(".pt") and ".tmp." not in f
        )
        if not files:
            raise FileNotFoundError(
                f"Stage2 validation trajectory cache is empty: {cache_dir}"
            )
        fname = f"{cache_index:06d}.pt"
        if fname not in files:
            raise FileNotFoundError(
                f"validation cache entry does not exist: "
                f"{os.path.join(cache_dir, fname)}"
            )
        data = torch.load(
            os.path.join(cache_dir, fname),
            map_location="cpu",
            weights_only=False,
        )
        prompt = str(data.get("prompt", "") or "")
        if not prompt:
            raise ValueError(f"validation cache entry {fname} has no prompt")
        if "height" not in data or "width" not in data:
            raise KeyError(
                f"validation cache entry {fname} lacks height/width metadata"
            )
        h = int(data["height"])
        w = int(data["width"])
        logger.info(f"[Sample] training prompt {fname} ({h}x{w}), teacher reference video: "
                    f"{os.path.join(cache_dir, 'videos', fname.replace('.pt', '.mp4'))}")
        return prompt, h, w

    def _decode_and_save_video(self, latents: torch.Tensor, save_path: str):
        """VAE-decode the latents to an mp4 (called on the writing rank only)."""
        from movieroll.utils.file_utils import save_videos_grid
        from movieroll.utils.torch_utils import PRECISION_TO_TYPE

        vae = self.vae
        vae_precision = self.config.get("vae_precision", "fp16")
        # `load_vae` casts the whole VAE through this same map, so the latents have to use it too: a
        # two-way fp16/bf16 guess breaks on the fp32 VAE a `vae_precision: fp32` config builds.
        vae_dtype = PRECISION_TO_TYPE[str(vae_precision)]
        latents = latents.to(self.device, dtype=vae_dtype)

        logger.info(f"    Decoding: latents {latents.shape}, dtype={latents.dtype}, "
                    f"range=[{latents.min():.3f}, {latents.max():.3f}]")

        spatial_tiling_before = getattr(vae, "use_spatial_tiling", None)
        temporal_tiling_before = getattr(vae, "use_temporal_tiling", None)
        try:
            # Undo the scaling factor (and the shift) before decoding, the same way the pipeline does:
            # without it the VAE receives an un-normalised latent and the picture is wrong.
            if getattr(vae.config, "shift_factor", None):
                latents = latents / vae.config.scaling_factor + vae.config.shift_factor
            else:
                latents = latents / vae.config.scaling_factor
            with torch.no_grad():
                vae.enable_tiling()
                vae.enable_temporal_tiling()
                video = vae.decode(latents, return_dict=False)[0]
        finally:
            # Both take the flag as a bool, and the VAE only ever stores True / False in them.
            vae.enable_tiling(spatial_tiling_before)
            vae.enable_temporal_tiling(temporal_tiling_before)

        logger.info(f"    Decoded video: {video.shape}, range=[{video.min():.3f}, {video.max():.3f}]")

        video = (video / 2 + 0.5).clamp(0, 1).cpu().float()
        save_videos_grid(video, save_path, rescale=False, n_rows=1,
                         fps=int(self.config.get("video_fps", 24)))

        del video
        torch.cuda.empty_cache()

    def cleanup(self):
        if hasattr(self, "student"):
            del self.student
        if hasattr(self, "teacher"):
            del self.teacher
        del self.vae
        del self.text_encoder

        torch.cuda.empty_cache()
        gc.collect()

        logger.info("Cleanup completed")
