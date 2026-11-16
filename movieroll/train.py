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

"""MovieRoll streaming SFT: the sample contract, and the training entry point.

Training reads pre-encoded latents from one directory: `<root>/manifest.jsonl`, one JSON object per sample,
and `<root>/latents/<id>.pt`, a float32 Tensor [32, T_lat, H//16, W//16] already multiplied by the VAE's
scaling_factor, where T_lat = (num_frames - 1)//4 + 1 and (T_lat - 1) is a whole number of chunks.

Manifest fields: `id` names the latent file; `text` is the structure caption, and "" takes the
null-structure path; `events` is optional, [{"text", "start_sec", "end_sec"}, ...]; `duration_sec` is
required and positive when `events` is non-empty; `data_type` is optional and must be "video".

Nothing in the data path draws from a global RNG, so DataLoader workers cannot perturb the trainer's streams.
`__getitem__` does not tokenize either: the FSDP-sharded text encoder cannot live in a DataLoader worker, so
`movieroll/trainer/model_inputs.py` tokenizes and encodes in the main process.
"""

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from einops import rearrange
from torch.utils.data import DataLoader, Dataset, get_worker_info

from movieroll.commons.constants import BYT5_COLOR, BYT5_FONT, C_SCALE, NUM_TRAIN_TIMESTEPS
from movieroll.models.autoencoders.vae_loader import load_vae
from movieroll.models.text_encoders import PROMPT_TEMPLATE
from movieroll.models.text_encoders.byT5 import load_glyph_byT5_v2
from movieroll.models.text_encoders.byT5.format_prompt import MultilingualPromptFormat
from movieroll.models.text_encoders.glyph_text import extract_glyph_texts
from movieroll.optim.muon import Muon
from movieroll.pipelines.movieroll_pipeline import MovieRollStreamingPipeline
from movieroll.schedulers.scheduling_flow_match_streaming import FlowMatchStreamingScheduler
from movieroll.trainer.ema import FSDPEMAModel
from movieroll.trainer.flow import create_transport
from movieroll.trainer.model_inputs import prepare_model_inputs
from movieroll.trainer.sft.config import SFTConfig, load_sft_config
from movieroll.utils.checkpoint import resume_training, save_checkpoint_without_optim
from movieroll.utils.file_utils import save_videos_grid
from movieroll.utils.fsdp import (
    apply_fsdp2,
    apply_fsdp_checkpointing,
    broadcast_from_first_replica,
    fsdp_device_mesh,
)
from movieroll.utils.helpers import CycleStates, ScalarStates
from movieroll.utils.load import get_no_split_modules, load_text_encoder, load_transformer
from movieroll.utils.logging import setup_logger
from movieroll.utils.multitask_utils import get_multitask_mask
from movieroll.commons.parallel_states import (
    ParallelDims,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    init_nccl_process_group,
    initialize_sequence_parallel_state_v2,
    nccl_info,
)
from movieroll.utils.parallel_sync import sync_object_for_parallel_training, sync_random_states
from movieroll.utils.prompt_csv import read_prompt_csv
from movieroll.utils.torch_utils import PRECISION_TO_TYPE, all_reduce_sum

gc.set_threshold(7000, 100, 100)

check_min_version("0.31.0")

# The latent grid of the 16164 VAE: 4 pixel frames per latent frame, 16 pixels per latent cell.
TEMPORAL_STRIDE = 4
SPATIAL_STRIDE = 16
LATENT_CHANNELS = 32

# Multiplier that keeps the glyph-dropout seeds of different (base seed, dp rank) pairs apart. Worker seeds of
# one loader are consecutive integers, so a plain sum with the rank would collide.
_GLYPH_SEED_STRIDE = 1_000_003


def timestamp_to_latent_index(
    timestamp_sec: float,
    total_duration_sec: float,
    total_latent_frames: int,
    temporal_stride: int = TEMPORAL_STRIDE,
) -> int:
    """Map a timestamp to a latent frame index in [0, total_latent_frames].

    The upper bound is inclusive: it is the exclusive end of the last latent frame, the convention
    `event_stamp[..., 1]` uses. The first latent frame covers a single pixel frame, so the conversion goes
    through pixel frames rather than scaling the proportion directly.
    """
    if total_duration_sec <= 0 or total_latent_frames <= 0:
        return 0

    total_pixel_frames = (total_latent_frames - 1) * temporal_stride + 1
    proportion = min(max(timestamp_sec / total_duration_sec, 0.0), 1.0)
    pixel_idx = int(round(proportion * total_pixel_frames))
    pixel_idx = min(max(pixel_idx, 0), total_pixel_frames)

    if pixel_idx <= 0:
        return 0
    latent_idx = (pixel_idx - 1) // temporal_stride + 1
    return min(latent_idx, total_latent_frames)


def _normalize_streaming_events(
    events, total_duration_sec: float
) -> List[Tuple[float, float, str]]:
    """Clamp to [0, total_duration_sec], drop empty or non-positive events, sort by (start, end).

    Overlapping events are allowed, and the result may be empty.
    """
    if not isinstance(events, list) or total_duration_sec <= 0:
        return []

    normalized = []
    for event in events:
        if not isinstance(event, dict):
            continue
        text = str(event.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(event["start_sec"])
            end = float(event["end_sec"])
        except (TypeError, ValueError, KeyError):
            continue
        start = min(max(start, 0.0), total_duration_sec)
        end = min(max(end, 0.0), total_duration_sec)
        if end <= start:
            continue
        normalized.append((start, end, text))

    normalized.sort(key=lambda item: (item[0], item[1]))
    return normalized


class MovieRollJsonlDataset(Dataset):
    """Pre-encoded latents plus their captions, read from a manifest.jsonl directory.

    The caption type is derived from the sample index, so it is the same in every epoch and for any worker
    count; the glyph dropout is redrawn on every load from a local RNG (`_glyph_dropout_rng`).
    """

    def __init__(
        self,
        root,
        height: int,
        width: int,
        num_frames: int,  # PIXEL frames, of the form 4k+1; the latent count is self.latent_frames
        caption_type_probs: Dict[str, float],
        byt5_uncond_p: float,
        chunk_size: int,
        font_path: str = BYT5_FONT,
        color_path: str = BYT5_COLOR,
        seed: int = 0,
        dp_rank: int = 0,
    ):
        self.root = Path(root)
        manifest_path = self.root / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")

        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.caption_type_probs = dict(caption_type_probs)
        self.byt5_uncond_p = float(byt5_uncond_p)
        self.chunk_size = chunk_size
        self.seed = int(seed)
        self.dp_rank = int(dp_rank)
        # Built lazily, once per process that loads samples; see _glyph_dropout_rng.
        self._glyph_rng = None
        self._glyph_rng_owner = None

        # Two separate conditions: the VAE needs 4n+1 pixel frames, and the rolling window needs the resulting
        # latent frames to divide into whole chunks. Checking only the second lets e.g. 19 through, because the
        # floor division hides the remainder.
        if num_frames <= 0 or (num_frames - 1) % TEMPORAL_STRIDE != 0:
            raise ValueError(
                f"num_frames={num_frames} must be positive and of the form 4n+1"
            )
        self.latent_frames = (num_frames - 1) // TEMPORAL_STRIDE + 1
        if (self.latent_frames - 1) % chunk_size != 0:
            raise ValueError(
                f"num_frames={num_frames} gives {self.latent_frames} latent frames, which is not a whole "
                f"number of chunks of {chunk_size}; half chunks are not supported"
            )
        self.latent_shape = (
            LATENT_CHANNELS,
            self.latent_frames,
            height // SPATIAL_STRIDE,
            width // SPATIAL_STRIDE,
        )

        with open(manifest_path, encoding="utf-8-sig") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"manifest is empty: {manifest_path}")
        # The event stamps are fractions of duration_sec: without it every event is dropped and the sample
        # quietly trains on its structure caption. Checked for the whole manifest before the first step.
        for line_number, record in enumerate(self.records, start=1):
            if not isinstance(record.get("id"), str):
                raise ValueError(f"{manifest_path} record {line_number}: `id` must be a string, got {record.get('id')!r}")
            if not isinstance(record.get("text") or "", str):
                raise ValueError(f"{record['id']}: `text` must be a string or null, got {record['text']!r}")
            if not (isinstance(record.get("events"), list) and record["events"]):
                continue
            try:
                has_duration = float(record.get("duration_sec")) > 0
            except (TypeError, ValueError):
                has_duration = False
            if not has_duration:
                raise ValueError(
                    f"{record['id']}: `duration_sec` must be a positive number when `events` is given, "
                    f"got {record.get('duration_sec')!r}"
                )

        self.prompt_format = MultilingualPromptFormat(font_path, color_path)

    def __len__(self) -> int:
        return len(self.records)

    def _load_latents(self, record) -> torch.Tensor:
        sample_id = record["id"]
        path = self.root / "latents" / f"{sample_id}.pt"
        latents = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(latents.shape) != self.latent_shape:
            raise ValueError(
                f"{sample_id}: latents have shape {tuple(latents.shape)}, expected {self.latent_shape}"
            )
        if torch.isnan(latents).any():
            raise ValueError(f"{sample_id}: latents contain NaN")
        return latents.to(torch.float32)

    def _glyph_dropout_rng(self) -> random.Random:
        """The local RNG the glyph dropout draws from: one per DataLoader worker, or one in the main process.

        A worker seeds it from its own `seed`, the main process (num_workers=0) from the `seed` given at
        construction, and either is combined with the dp rank, because every dp rank seeds its loader alike and
        would otherwise blank the very same draws. The owner check re-seeds a forked worker, which inherits
        whatever RNG the parent had already built.
        """
        info = get_worker_info()
        owner = None if info is None else (info.id, info.seed)
        if self._glyph_rng is None or self._glyph_rng_owner != owner:
            base_seed = self.seed if info is None else info.seed
            self._glyph_rng = random.Random(base_seed * _GLYPH_SEED_STRIDE + self.dp_rank)
            self._glyph_rng_owner = owner
        return self._glyph_rng

    def _glyph_text(self, text: str) -> str:
        """Quoted text of this sample, formatted for byT5. Empty when the caption quotes nothing.

        One draw per call, all or nothing: with probability `byt5_uncond_p` EVERY quoted string is blanked,
        which still produces a `Text "". ` prompt and still runs the glyph encoder. A caption without quotes
        is the one case that returns "" and the one case that draws nothing.
        """
        quoted = extract_glyph_texts(text)
        if not quoted:
            return ""
        if self._glyph_dropout_rng().random() < self.byt5_uncond_p:
            quoted = ["" for _ in quoted]
        styles = [{"color": None, "font-family": None} for _ in quoted]
        return self.prompt_format.format_prompt(quoted, styles)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        data_type = record.get("data_type", "video")
        if data_type != "video":
            raise ValueError(f"{record['id']}: data_type must be 'video', got {data_type!r}")

        latents = self._load_latents(record)
        total_latent_frames = latents.shape[1]
        text = record.get("text") or ""     # a JSON null is the empty caption, not the string "None"

        sample = {
            "latents": latents,
            "byt5_text": self._glyph_text(text),
            "data_type": "video",
            "id": record["id"],
        }

        # A missing caption short-circuits: one empty caption over the whole clip, and NO caption-type draw.
        if not text:
            sample.update(
                captions=[""],
                cap_type="structure",
                event_stamp=torch.tensor([[0, total_latent_frames]], dtype=torch.long),
                num_captions=1,
            )
            return sample

        # duration_sec is only read alongside events, where __init__ has checked it; otherwise it may be absent
        # or null, and the result is [] either way.
        raw_events = record.get("events")
        events = (
            _normalize_streaming_events(raw_events, float(record["duration_sec"]))
            if isinstance(raw_events, list) and raw_events else []
        )
        has_event = len(events) > 0

        # Exactly one draw per sample, on every path that gets here.
        rng = random.Random(idx)
        p_event = float(self.caption_type_probs.get("event", 0.0))
        p_structure = float(self.caption_type_probs.get("structure", 1.0))
        if not has_event and p_event > 0:
            p_structure += p_event
            p_event = 0.0
        total_p = p_event + p_structure
        if total_p <= 0:
            p_event, p_structure, total_p = 0.0, 1.0, 1.0
        cap_type = "event" if rng.random() < (p_event / total_p) else "structure"

        if cap_type == "event" and has_event:
            captions = [description for _, _, description in events]
            duration_sec = float(record["duration_sec"])
            stamps = [
                [
                    timestamp_to_latent_index(start, duration_sec, total_latent_frames),
                    timestamp_to_latent_index(end, duration_sec, total_latent_frames),
                ]
                for start, end, _ in events
            ]
            event_stamp = torch.tensor(stamps, dtype=torch.long)
            # The clip is covered end to end regardless of where the events actually start and stop.
            event_stamp[0, 0] = 0
            event_stamp[len(captions) - 1, 1] = total_latent_frames
            sample.update(
                captions=captions,
                cap_type="event",
                event_stamp=event_stamp,
                num_captions=len(captions),
            )
        else:
            # Also the fall-through for "event was drawn but this sample has none".
            sample.update(
                captions=[text],
                cap_type="structure",
                event_stamp=torch.tensor([[0, total_latent_frames]], dtype=torch.long),
                num_captions=1,
            )
        return sample


def _pad_and_stack_varlen(tensors: Sequence[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Pad a list of [n, ...] tensors to the longest n and stack them."""
    max_n = max(tensor.shape[0] for tensor in tensors)
    padded = []
    for tensor in tensors:
        if tensor.shape[0] < max_n:
            filler = torch.full(
                (max_n - tensor.shape[0],) + tensor.shape[1:], pad_value, dtype=tensor.dtype
            )
            tensor = torch.cat([tensor, filler], dim=0)
        padded.append(tensor)
    return torch.stack(padded)


def streaming_collate_fn(batch: List[dict]) -> dict:
    """Collate samples whose caption count differs.

    `captions` is padded with None and `event_stamp` with -1, both so the absent slots stay distinguishable
    from legitimate values (an empty caption and latent index 0 are both meaningful). A batch carries either
    pre-encoded `latents` or `pixel_values` for online encoding, never a mix of the two.
    """
    data_types = [sample["data_type"] for sample in batch]
    if any(data_type != "video" for data_type in data_types):
        raise ValueError(f"every sample must be video, got {set(data_types)}")
    if all("latents" in sample for sample in batch):
        media_key = "latents"
    elif all("pixel_values" in sample and "latents" not in sample for sample in batch):
        media_key = "pixel_values"
    else:
        raise ValueError("every sample of a batch must carry `latents`, or every sample `pixel_values`")

    max_captions = max(sample["num_captions"] for sample in batch)
    captions: List[List[Optional[str]]] = [
        list(sample["captions"]) + [None] * (max_captions - sample["num_captions"])
        for sample in batch
    ]

    return {
        media_key: torch.stack([sample[media_key] for sample in batch]),
        "captions": captions,
        "num_captions": torch.tensor(
            [sample["num_captions"] for sample in batch], dtype=torch.long
        ),
        "cap_type": [sample["cap_type"] for sample in batch],
        "event_stamp": _pad_and_stack_varlen(
            [sample["event_stamp"] for sample in batch], pad_value=-1
        ),
        "byt5_text": [sample["byt5_text"] for sample in batch],
        "data_type": data_types,
        "id": [sample["id"] for sample in batch],
    }


class MovieRollDistributedSampler:
    """Distributed sampler that can resume mid-epoch: shuffle, drop_last, one sample per index.

    `start_index` skips the first N samples of THIS epoch's permutation -- globally, before the ranks
    interleave -- and the permutation comes from an explicit generator, so no global RNG is touched.
    """

    def __init__(
        self,
        dataset_length: int,
        num_replicas: int,
        rank: int,
        seed: int,
        start_index: int = 0,
    ):
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError(f"invalid rank {rank} for num_replicas {num_replicas}")
        self.dataset_length = dataset_length
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.start_index_expired = False
        self._start_index = 0
        self.start_index = start_index

    @property
    def start_index(self) -> int:
        return self._start_index

    @start_index.setter
    def start_index(self, value: int) -> None:
        self._start_index = int(value)
        self._recompute_sizes()

    def _recompute_sizes(self) -> None:
        self.num_samples = (self.dataset_length - self._start_index) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        assert not (self.start_index_expired and self._start_index != 0), (
            "start_index was already consumed; reset it to 0 before iterating again"
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_length, generator=generator).tolist()
        indices = indices[self._start_index:]
        indices = indices[: self.total_size]
        assert len(indices) == self.total_size, (len(indices), self.total_size)
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples, (len(indices), self.num_samples)
        self.start_index_expired = True
        return iter(indices)


def build_sft_dataloader(
    dataset,
    *,
    batch_size: int,
    dp_degree: int,
    dp_rank: int,
    global_seed: int,
    num_workers: int,
    prefetch_factor: int,
    start_index: int = 0,
):
    """DataLoader plus its sampler, both seeded explicitly.

    `generator` is not optional: without it every `iter()` draws its base seed from the global torch RNG, which
    would both move the trainer's own stream and make the worker seeds depend on how many random numbers the
    trainer happened to consume. Each worker's `seed`, and with it the dataset's glyph dropout, derives from
    that base seed.
    """
    sampler = MovieRollDistributedSampler(
        dataset_length=len(dataset),
        num_replicas=dp_degree,
        rank=dp_rank,
        seed=global_seed,
        start_index=start_index,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        collate_fn=streaming_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=None if num_workers == 0 else prefetch_factor,
        persistent_workers=num_workers > 0,
        generator=torch.Generator(device="cpu").manual_seed(int(global_seed)),
    )
    return loader, sampler


def iterate_epochs(loader, sampler, *, epoch: int = 0, start_index: int = 0):
    """Yield batches forever, rolling over to a fresh shuffle at the end of each epoch.

    Epoch N uses the permutation seeded `global_seed + N`, so every epoch really does reshuffle.
    """
    sampler.set_epoch(epoch)
    sampler.start_index = start_index
    iterator = iter(loader)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            sampler.start_index_expired = False
            sampler.start_index = 0
            iterator = iter(loader)
            batch = next(iterator)
        yield batch, epoch


# ======================================================================================================
# Trainer
# ======================================================================================================

# Videos generated per data-parallel rank at every sampling step.
_N_SAMPLES_PER_RANK = 2

# Seed offset of the online-encode generator from local_seed. set_seed(local_seed) also starts the global CUDA
# stream that the flow-matching noise x0 = randn_like(latents) is drawn from, with the same shape as the VAE
# posterior noise; without the offset the two would replay the same numbers.
_ENCODE_SEED_OFFSET = 1_000_003

# The single data key of the SFT run; `scalar_states.pkl` stores its consumed-sample counters under this key.
_DATA_KEY = "video"

# Prompt templates of the Qwen2.5-VL text encoder.
_PROMPT_TEMPLATE = "li-dit-encode-image-json"
_PROMPT_TEMPLATE_VIDEO = "li-dit-encode-video-json"


def sync_cuda_time():
    """Wall clock read after the whole world has caught up, so the logged step times are comparable.

    The barrier lines the ranks up, `torch.cuda.synchronize()` drains this rank's queued kernels, and only
    then is the clock read.
    """
    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    torch.cuda.synchronize()
    return time.time()


def _batch_size(batch: dict) -> int:
    """Samples in a collated batch, whichever of `latents` / `pixel_values` it carries."""
    return batch["latents" if "latents" in batch else "pixel_values"].shape[0]


def iterate_with_states(loader, sampler, scalar_states):
    """Yield batches forever, keeping the two counters a mid-epoch resume needs.

    `consumed_samples_per_dp` counts this epoch's samples on this data-parallel rank, `epoch_consumed_samples`
    counts them globally and is what the sampler's `start_index` is restored from; both restart when the epoch
    rolls over. The global counter is accumulated by the logging block, which has the all-gathered count.
    """
    epoch = int(scalar_states.epoch[_DATA_KEY])
    start_index = int(scalar_states.epoch_consumed_samples[_DATA_KEY])
    for batch, cur_epoch in iterate_epochs(loader, sampler, epoch=epoch, start_index=start_index):
        if cur_epoch != epoch:
            scalar_states.consumed_samples_per_dp[_DATA_KEY] = 0
            scalar_states.epoch_consumed_samples[_DATA_KEY] = 0
            scalar_states.epoch[_DATA_KEY] = cur_epoch
            epoch = cur_epoch
        scalar_states.consumed_samples_per_dp[_DATA_KEY] += _batch_size(batch)
        yield batch


def _assign_sampling_indices(total_samples: int, dp_degree: int, dp_rank: int):
    """The prompt-CSV rows this dp rank samples, and a flag per row that is True for a repeat not to be saved.

    Every pipeline call is a chain of collectives (text encoder and DiT are FSDP-sharded over the dp mesh), so
    every rank must make the same number of calls: min(_N_SAMPLES_PER_RANK, ceil(total_samples / dp_degree)).
    A rank holding fewer rows of its own tops up by repeating them, and a top-up is never written out. A
    rank past the last row borrows row `dp_rank % total_samples`, and that one IS written out -- without
    that fallback the top-up below divides by an empty list.
    """
    assigned = [i for i in range(total_samples) if i % dp_degree == dp_rank]
    assigned = assigned[:_N_SAMPLES_PER_RANK]
    if not assigned:
        assigned = [dp_rank % total_samples]
    per_rank = min(_N_SAMPLES_PER_RANK, -(-total_samples // dp_degree))
    top_up = [assigned[k % len(assigned)] for k in range(per_rank - len(assigned))]
    return assigned + top_up, [False] * len(assigned) + [True] * len(top_up)


def _reached_max_train_steps(update_steps: int, max_train_steps: int) -> bool:
    """The SFT loop's only exit, since the data iterator reshuffles forever.

    Checked after every update, once a checkpoint due at that step is written, and once right after a resume.
    `update_steps` is the same on every rank, and in every rank's entry of the scalar states, so all ranks
    agree.
    """
    return update_steps >= max_train_steps


@torch.no_grad()
def sample_validation(cfg, device, vae, text_encoder, model,
                      batch, global_step, logger, byt5_kwarg,
                      dp_rank, multitask_mask, mask_type,
                      parallel_dims, ema_sampling: bool = False,
                      interactive_mode: bool = False):
    """Generate a few videos from the current weights, either from a prompt CSV or from the batch's own prompt.

    `multitask_mask` and `mask_type` are parameters because only the fixed-prompt branch recomputes them; the
    batch-prompt branch reuses the values the training step drew. Only `mask_type` is read afterwards;
    `multitask_mask` is drawn and then unused, which keeps the draw itself in the RNG stream.
    """
    logger.info("---------------------------- Sample Stage --------------------------")
    model.eval()
    # gc before validation, so it does not run out of device memory
    gc.collect()
    torch.cuda.empty_cache()

    seeds = [cfg.global_seed]
    generator = torch.Generator(device=device).manual_seed(cfg.global_seed)

    base_images = batch.get("pixel_values")
    if base_images is not None:
        # The ground-truth half of the saved grid is written like the samples, in [0, 1]; pixel_values are [-1, 1].
        base_images = (base_images[:_N_SAMPLES_PER_RANK].float() / 2 + 0.5).clamp(0, 1).cpu()

    # Slot 0 is the caption the training step conditions on.
    prompts = [captions[0] for captions in batch["captions"]][:_N_SAMPLES_PER_RANK]
    videoid = batch.get("id") or ["unknown"] * _N_SAMPLES_PER_RANK
    videoid = videoid[:_N_SAMPLES_PER_RANK]
    data_type = "video"
    infer_flow_shift = cfg.infer_flow_shift_video

    mode_label = "interactive" if interactive_mode else "normal"

    scheduler = FlowMatchStreamingScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        shift=infer_flow_shift,
    )
    pipeline = MovieRollStreamingPipeline(
        vae=vae,
        text_encoder=text_encoder,
        transformer=model,
        scheduler=scheduler,
        precision=cfg.precision,
        vae_precision=cfg.vae_precision,
        prompt_format_font_path=cfg.multilingual_prompt_format_font_path,
        prompt_format_color_path=cfg.multilingual_prompt_format_color_path,
        byt5_kwarg=byt5_kwarg,
        text_len=cfg.text_len,
        streaming_config=cfg.streaming_config,
    )

    # decode vae to pixel_values
    if base_images is None:
        latents = batch["latents"][:_N_SAMPLES_PER_RANK].to(device)
        # Latents are stored fp32 while the VAE is built at `vae_precision`, and this decode path has no
        # autocast of its own -- so the cast belongs here.
        base_images = pipeline.decode_latents(
            latents.to(PRECISION_TO_TYPE[cfg.vae_precision])
        )  # b c t h w

    target_length, target_height, target_width = base_images.shape[-3:]

    use_fixed_prompt = False
    # Batch prompts are the same count on every rank, so only the CSV branch below ever pads.
    is_top_up = [False] * len(prompts)

    event_prompts_list = []

    if (global_step > 0 and global_step % cfg.fixed_prompt_validation_every == 0) or (
        cfg.try_first_iter and global_step == 0
    ):
        fixed_prompt_path = cfg.sample_prompt_csv
        if fixed_prompt_path is not None:
            prompts = []
            seeds = []
            indexs = []
            csv_file_names = []
            base_images = None
            use_fixed_prompt = True
            for pmt_path in fixed_prompt_path:
                table = read_prompt_csv(pmt_path)
                # table.input_images is dropped: SFT samples text-to-video only; an input_image column is ignored,
                # not rejected
                prompts.extend(table.prompts)
                seeds.extend(table.seeds)
                indexs.extend(table.indexs)
                csv_file_names.extend(table.csv_names)
                event_prompts_list.extend(table.event_prompts)

            total_samples = len(prompts)
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            assigned_indices, is_top_up = _assign_sampling_indices(total_samples, dp_degree, dp_rank)
            logger.info(f"[{mode_label}] Total inference samples: {total_samples}, dp_rank {dp_rank} "
                        f"assigned {len(assigned_indices)} samples: {assigned_indices} "
                        f"({sum(is_top_up)} repeated only to keep the ranks in step, not saved)")
            indexs = [indexs[i] for i in assigned_indices]
            prompts = [prompts[i] for i in assigned_indices]
            seeds = [seeds[i] for i in assigned_indices]
            csv_file_names = [csv_file_names[i] for i in assigned_indices]
            event_prompts_list = [event_prompts_list[i] for i in assigned_indices]
    infer_steps = cfg.infer_steps_video
    for sidx, prompt in enumerate(prompts):
        if use_fixed_prompt:
            sample_n_frames = cfg.sample_n_frames
            target_length = sample_n_frames
            latent_target_length = (sample_n_frames - 1) // TEMPORAL_STRIDE + 1
            target_height, target_width = cfg.height, cfg.width
            generator = torch.Generator(device=device).manual_seed(seeds[sidx])

            # t2v is the only input path here. The call reports the mask_type and consumes exactly one global
            # random.random() per sample, which is part of the RNG stream.
            multitask_mask, mask_type = get_multitask_mask(latent_target_length)

        assert mask_type == "t2v", f"only text-to-video sampling is supported, got {mask_type}"

        event_composed = event_prompts_list[sidx] if use_fixed_prompt and event_prompts_list else None
        _is_interactive = interactive_mode and event_composed is not None and len(event_composed) >= 2
        event_prompts_arg = list(event_composed) if _is_interactive else None

        print(f"Rank {dp_rank}, [{mode_label}] sample_validation [{sidx + 1}/{len(prompts)}] "
              f"Data type: {data_type} \t mask_type: {mask_type} \t target_length: {target_length}\t "
              f"target_height: {target_height}\t target_width: {target_width} \t "
              f"flow_shift: {infer_flow_shift} \t infer_steps: {infer_steps} \t "
              f"seed: {seeds[sidx] if use_fixed_prompt else seeds} \t prompt: {prompt}")
        samples = pipeline(prompt=prompt,
                           height=target_height,
                           width=target_width,
                           video_length=target_length,
                           num_inference_steps=infer_steps,
                           use_cfg=cfg.use_cfg,
                           negative_prompt=cfg.neg_prompt,
                           generator=generator,
                           output_type='pt',
                           event_prompts=event_prompts_arg,
                           )[0]

        # a top-up repeat was generated only to keep the collectives in step, so it is not written out
        if not is_top_up[sidx] and dp_rank < (total_samples if use_fixed_prompt else 128):
            sample = samples[0].unsqueeze(0)
            sub_samples = []
            if base_images is not None:
                gt_pixel_value = base_images[sidx][None, ...]
                sample = sample.cpu()
                gt_pixel_value = gt_pixel_value.cpu()
                sub_samples.append(sample)   # b c t h w, [0, 1]
                sub_samples.append(gt_pixel_value)
                sample_num = len(sub_samples)
                sub_samples = torch.concat(sub_samples)
            else:
                sub_samples.append(sample)
                sample_num = len(sub_samples)
                sub_samples = torch.concat(sub_samples)

            if use_fixed_prompt:
                # Save to folder named after csv file, video named by index
                folder_name, file_name = csv_file_names[sidx]
                ema_tag = 'ema' if ema_sampling else 'normal'
                save_dir = os.path.join(cfg.output_dir, "samples", ema_tag, f"{global_step:07d}",
                                        mode_label, folder_name, file_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{indexs[sidx]}.mp4")
            else:
                save_base_path = (
                    f"{cfg.output_dir}/samples/{'ema' if ema_sampling else 'normal'}/{global_step:07d}/"
                    f"rank{dp_rank}_{mask_type}_{target_length}x{target_height}x{target_width}-{videoid[sidx]}"
                )
                save_prompt_name = prompt[:100].replace('/', ' ')
                save_path = f"{save_base_path}-{save_prompt_name}.mp4"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)

            # One video per sp group: every rank of the group holds the same all_gathered sample and builds
            # this same path, so without the guard all sp_size of them write it at once. Everything below is
            # file I/O -- the decode happened inside the pipeline call above -- so a rank that skips it takes
            # part in no collective here.
            if nccl_info.rank_within_spgroup == 0:
                if target_length > 1:
                    try:
                        save_videos_grid(sub_samples, save_path, n_rows=sample_num, fps=cfg.video_fps)
                    except Exception:
                        save_prompt_name = prompt[:10].replace('/', ' ')
                        if use_fixed_prompt:
                            save_path = os.path.join(save_dir, f"{indexs[sidx]}_{save_prompt_name}.mp4")
                        else:
                            save_path = f"{save_base_path}-{save_prompt_name}.mp4"
                        save_videos_grid(sub_samples, save_path, n_rows=sample_num, fps=cfg.video_fps)
                else:
                    sample = rearrange(sub_samples, "b c f h w -> c h (b f w)")
                    try:
                        torchvision.utils.save_image(sample, os.path.splitext(save_path)[0] + '.png')
                    except Exception:
                        save_path = (
                            f"{cfg.output_dir}/samples/{'ema' if ema_sampling else 'normal'}/{global_step:07d}/"
                            f"rank{dp_rank}_{mask_type}_{target_length}x{target_height}x{target_width}-"
                            f"{videoid[sidx]}-{prompt[:10].replace('/', ' ')}.mp4"
                        )
                        torchvision.utils.save_image(sample, os.path.splitext(save_path)[0] + '.png')
                with open(os.path.splitext(save_path)[0] + '.txt', 'w') as f:
                    f.write(f"{mask_type}: {prompt}")
                print(f'Rank {dp_rank} saved sample [{sidx + 1}/{len(prompts)}] to {save_path}')

        if len(prompts) > 1:
            torch.cuda.empty_cache()

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    model.train()


def print_training_configuration(cfg, logger, model, world_size, local_rank, rank,
                                 dp_degree, dp_rank, video_total_batch_size, video_num,
                                 video_loader=None):
    params_count = model.params_count()

    logger.info("****************************** Running training ******************************")

    logger.info("=" * 80)
    logger.info("SYSTEM & HARDWARE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Number of GPUs                                   : {world_size}")
    logger.info(f"Local rank                                       : {local_rank}")
    logger.info(f"Global rank                                      : {rank}")
    logger.info(f"Device                                           : cuda:{local_rank}")
    logger.info(f"Master weight dtype                              : {model.parameters().__next__().dtype}")

    logger.info("-" * 80)
    logger.info("MODEL PARAMETERS & ARCHITECTURE")
    logger.info("-" * 80)
    for k, v in params_count.items():
        logger.info(f"Number of {k:<25}                     : {v:,}")
    total_params_b = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    logger.info(f"Total trainable parameters per FSDP shard       : {total_params_b:.3f}B")
    logger.info(f"Model type                                       : {cfg.model_type}")
    logger.info(f"DIT model path                                   : {cfg.dit_weight}")
    logger.info(f"Gradient checkpointing                           : {cfg.gradient_checkpointing}")
    logger.info(f"Text length                                      : {cfg.text_len}")
    logger.info(f"Selective checkpointing                          : {cfg.selective_checkpointing}")
    logger.info(f"EMA decay                                        : {cfg.ema_decay}")

    logger.info("-" * 80)
    logger.info("DISTRIBUTED TRAINING CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"World size                                       : {world_size}")
    logger.info(f"DP degree                                        : {dp_degree}")
    logger.info(f"DP rank                                          : {dp_rank}")
    logger.info(f"SP size                                          : {cfg.sp_size}")
    logger.info(f"FSDP sharding strategy                           : {cfg.fsdp_sharding_strategy}")
    logger.info(f"FSDP shard group size                            : {cfg.fsdp_sub_world_size}")
    logger.info(f"CPU offload                                      : {cfg.use_cpu_offload}")

    logger.info("-" * 80)
    logger.info("BATCH SIZE & DATA FLOW CONFIGURATION")
    logger.info("-" * 80)
    effective_batch_video = video_total_batch_size * cfg.gradient_accumulation_steps

    logger.info(f"Micro batch size                                 : {cfg.batch_size}")
    logger.info(f"Total batch size (all GPUs)                      : {video_total_batch_size} (video)")
    logger.info(f"Gradient accumulation steps                      : {cfg.gradient_accumulation_steps}")
    logger.info(f"Effective batch size (with grad accum)           : {effective_batch_video} (video)")

    logger.info("-" * 80)
    logger.info("DATASET & DATALOADER CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Data type                                        : {_DATA_KEY}")
    logger.info(f"Video examples                                   : {video_num:,}")
    logger.info(f"Video data path                                  : {cfg.data_root}")
    logger.info(f"Video size                                       : {cfg.height}x{cfg.width}")
    logger.info(f"Video frames                                     : {cfg.sample_n_frames}")

    if video_loader is not None:
        logger.info(f"Video dataloader batch size                      : {video_loader.batch_size}")
        logger.info(f"Video dataloader drop last                       : {video_loader.drop_last}")

    logger.info("-" * 80)
    logger.info("TRAINING SCHEDULE & STEPS CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Number of epochs                                 : {cfg.num_train_epochs}")
    logger.info(f"Max training steps (= optimizer updates)         : {cfg.max_train_steps}")

    total_micro_batch_steps = cfg.max_train_steps * cfg.gradient_accumulation_steps
    logger.info(f"Total forward/backward (micro-batch) steps       : {total_micro_batch_steps}")
    logger.info(f"Forward steps per optimizer update               : {cfg.gradient_accumulation_steps}")

    logger.info("-" * 80)
    logger.info("OPTIMIZATION CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Learning rate                                    : {cfg.learning_rate}")
    logger.info(f"Weight decay                                     : {cfg.weight_decay}")
    logger.info(f"Max gradient norm                                : {cfg.max_grad_norm}")
    logger.info(f"LR scheduler                                     : {cfg.lr_scheduler}")
    logger.info(f"LR warmup steps                                  : {cfg.lr_warmup_steps}")
    logger.info(f"LR num cycles                                    : {cfg.lr_num_cycles}")
    logger.info(f"LR power                                         : {cfg.lr_power}")

    logger.info("-" * 80)
    logger.info("LOGGING & CHECKPOINTING CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Output directory                                 : {cfg.output_dir}")
    logger.info(f"Checkpointing steps                              : {cfg.checkpointing_steps}")
    logger.info(f"Sample interval                                  : {cfg.sample_interval}")
    logger.info(f"Log interval                                     : {cfg.log_interval}")
    logger.info(f"Resume checkpoint                                : {cfg.resume_checkpoint}")
    logger.info(f"Dry run enabled                                  : {cfg.dry_run}")

    logger.info("-" * 80)
    logger.info("INFERENCE & VALIDATION CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Inference steps                                  : {cfg.infer_steps_video}")
    logger.info(f"CFG (AnnealCFG)                                  : {cfg.use_cfg}")
    logger.info(f"Inference flow shift                             : {cfg.infer_flow_shift_video}")
    logger.info(f"Negative prompt                                  : {cfg.neg_prompt}")
    logger.info(f"Sampling prompt CSV                              : {cfg.sample_prompt_csv}")
    logger.info(f"Fixed prompt interval                            : {cfg.fixed_prompt_validation_every}")

    logger.info("-" * 80)
    logger.info("SEEDS & REPRODUCIBILITY CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Global seed                                      : {cfg.global_seed}")
    logger.info(f"Local seed                                       : {cfg.local_seed}")

    logger.info("-" * 80)
    logger.info("ADDITIONAL CONFIGURATION: GLYPH BYT5 V2")
    logger.info("-" * 80)
    logger.info(f"ByT5 max length                                   : {cfg.byt5_max_length}")

    logger.info("-" * 80)
    logger.info("STREAMING CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Streaming config                                 : {cfg.streaming_config}")
    logger.info(f"Streaming max num latent frames                  : {cfg.streaming_max_num_latent_frames}")
    logger.info(f"Streaming chunk size                             : {cfg.streaming_chunk_size}")
    logger.info(f"Streaming window size                            : {cfg.streaming_window_size}")
    logger.info(f"Streaming t2v prob (rest is i2v)                 : {cfg.streaming_t2v_prob}")
    logger.info(f"Streaming full history block num (HieraKV)        : {cfg.streaming_full_history_block_num}")
    logger.info(f"Streaming caption type probs                      : {cfg.streaming_caption_type_probs}")
    logger.info(f"Streaming context noise enabled                   : {cfg.streaming_context_noise_enabled}")
    logger.info(f"Streaming context noise prob                      : {cfg.streaming_context_noise_prob}")
    logger.info(f"Streaming interactive mode                        : {cfg.streaming_interactive_mode}")

    logger.info("-" * 80)
    logger.info("PERFORMANCE ESTIMATES")
    logger.info("-" * 80)
    logger.info(f"Target throughput per update                     : {effective_batch_video} (video)")

    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)


def main(cfg: SFTConfig):

    # ============================= Setup ==============================
    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # sub_world_size controls the sharding granularity:
    #   8: shard within one node
    #  16: shard across two nodes
    #  32: shard across four nodes
    sub_world_size = cfg.fsdp_sub_world_size

    if cfg.fsdp_sharding_strategy == 'full':
        dp_replicate = 1
    else:
        assert cfg.fsdp_sharding_strategy == 'hybrid'
        dp_replicate = world_size // sub_world_size

    if world_size == sub_world_size and cfg.fsdp_sharding_strategy == 'hybrid':
        cfg.fsdp_sharding_strategy = "full"
        dp_replicate = 1
        print("node=1 fall back hybrid to full-shard")

    parallel_dims = ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=-1,
        sp=cfg.sp_size,
        world_size=world_size,
    )
    init_nccl_process_group()
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    initialize_sequence_parallel_state_v2(parallel_dims, world_mesh)

    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)

    # sync output_dir over all ranks
    obj_list = [cfg.output_dir]
    dist.broadcast_object_list(obj_list, src=0)
    cfg.output_dir = obj_list[0]

    logger = setup_logger(cfg.output_dir)
    tb_writer = None
    if rank == 0 and not cfg.disable_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError(
                "tensorboard is not installed; install it or set `disable_tensorboard: true` in the config"
            ) from error
        tb_dir = os.path.join(cfg.output_dir, "tb")
        tb_writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f"--> args {cfg}")
    logger.info(f"dp_degree, dp_rank: {dp_degree, dp_rank}")

    wandb_run = None
    wandb_module = None
    if cfg.wandb_project is not None and rank == 0:
        try:
            import wandb as wandb_module
        except ImportError as error:
            raise ImportError(
                "wandb is not installed; install it or set `wandb_project: null` in the config"
            ) from error
        wandb_run = wandb_module.init(
            project=cfg.wandb_project,
            name=cfg.wandb_exp_name,
            config=cfg,
            dir=cfg.output_dir
        )
        logger.info(f"--> wandb initialized, project: {cfg.wandb_project}, name: {cfg.wandb_exp_name}")

    # =========================== Build main model ===========================
    logger.info(f"--> loading model from {cfg.dit_weight}")
    # the master weight precision is a setting: fp32 or bf16
    factor_kwargs = {'device': device,
                     'dtype': torch.float32 if cfg.master_weight_type == "fp32" else torch.bfloat16}

    streaming_config = cfg.build_streaming_config()
    factor_kwargs['streaming_config'] = streaming_config
    # Read back by sample_validation and create_transport. Nothing may mutate a field of it after this point:
    # the streaming wrappers cache by id(streaming_config) and never invalidate.
    cfg.streaming_config = streaming_config

    model = load_transformer(
        cfg.model_type,
        cfg.dit_weight,
        factor_kwargs,
        logger,
    )

    model.train()

    # One seed per SP group (rank // sp_size is the group index), so the whole RNG stream -- the timesteps and
    # the noise alike -- is shared inside a group and differs only between groups.
    cfg.local_seed = cfg.global_seed + rank // cfg.sp_size
    set_seed(cfg.local_seed)
    random.seed(cfg.local_seed)
    print(f"dp_rank: {dp_rank}, rank: {rank},  local_seed: {cfg.local_seed}")

    # ============================= Build extra models =========================
    # The non-streaming VAE decodes the sampling output and encodes the training input when the Dataset
    # returns pixel_values.
    vae, _, _, _ = load_vae(
        cfg.vae_precision,
        cfg.vae_path,
        logger=logger,
        device=device,
        streaming=False,
    )
    # Online encoding of `pixel_values` samples the latent distribution. It draws from its own generator so the
    # global CUDA RNG the training step is written against is left alone, and its seed derives from local_seed
    # because the ranks of an sp group hold the same broadcast batch and must encode it to the same latents.
    encode_generator = torch.Generator(device=device).manual_seed(cfg.local_seed + _ENCODE_SEED_OFFSET)
    logger.info(f"--> loading text encoder from {cfg.text_encoder_path}")
    # The text encoder is FSDP-sharded below, so it is first loaded onto the cpu
    text_encoder = load_text_encoder(
        cfg.text_encoder_path,
        cfg.text_len,
        PROMPT_TEMPLATE[_PROMPT_TEMPLATE],
        PROMPT_TEMPLATE[_PROMPT_TEMPLATE_VIDEO],
        torch.device("cpu"),
        text_encoder_precision=cfg.text_encoder_precision,
        logger=logger,
    )
    denoiser = create_transport(video_shift=cfg.flow_shift_video, streaming_config=cfg.streaming_config)

    byt5_kwarg = load_glyph_byT5_v2(
        torch.device("cpu"),
        byT5_google_path=cfg.byt5_google_path,
        byT5_ckpt_path=cfg.byt5_ckpt_path,
        color_ann_path=cfg.multilingual_prompt_format_color_path,
        font_ann_path=cfg.multilingual_prompt_format_font_path,
        byt5_max_length=cfg.byt5_max_length,
    )

    logger.info(
        f"  Total training parameters = {sum(p.numel() for p in model.parameters()) / 1e6} M"
    )

    # ============================== resume ==============================
    init_steps = 0
    if cfg.resume_checkpoint:
        if not os.path.exists(cfg.resume_checkpoint):
            raise ValueError("Invalid resume_checkpoint path: %s" % cfg.resume_checkpoint)

        # EMA training is always on, so this reads ema_diffusion_pytorch_model.safetensors: the live model
        # resumes from the EMA weights.
        (model, _scalar_states_list) = resume_training(
            model,
            cfg.resume_checkpoint,
        )

        scalar_states = ScalarStates.from_pretrained(
            _scalar_states_list,
            rank=rank,
            world_size=world_size,
            default_rank0_ss=False,
        )
        print(f'--> resume training scalar states {scalar_states}')
        # The LR scheduler's last_epoch must be update_steps, not train_steps: lr_scheduler.step() runs once
        # per update step, so with grad_accum > 1 train_steps would put the LR at the wrong position.
        init_steps = scalar_states.update_steps  # resumed LR position (counted in update steps)

        if _reached_max_train_steps(scalar_states.update_steps, cfg.max_train_steps):
            logger.info(f"Resumed at update step {scalar_states.update_steps}, which already reaches "
                        f"max_train_steps={cfg.max_train_steps}; nothing left to train, stopping.")
            if get_sequence_parallel_state():
                destroy_sequence_parallel_group()
            return
    else:
        scalar_states = ScalarStates(lr=cfg.learning_rate)

    # ============================= Build FSDP =========================
    logger.info(
        f"--> Initializing FSDP with sharding strategy: {cfg.fsdp_sharding_strategy}"
    )
    no_split_modules = get_no_split_modules(model)
    fsdp_mesh = fsdp_device_mesh(parallel_dims, world_mesh, cfg.fsdp_sharding_strategy)
    logger.info(f"FSDP device mesh: {fsdp_mesh}")

    if cfg.gradient_checkpointing:
        apply_fsdp_checkpointing(
            model, no_split_modules, cfg.selective_checkpointing
        )

    apply_fsdp2(
        model=model,
        dp_mesh=fsdp_mesh,
        param_dtype=torch.float32 if cfg.master_weight_type == "fp32" else torch.bfloat16,
        reduce_dtype=torch.float32,
        cpu_offload=cfg.use_cpu_offload,
        reshard_after_forward_policy="default",
    )
    logger.info("--> model loaded")

    # hybrid with real replicas; a 1-D mesh holds distinct shards on every rank, and a single rank has no mesh
    if fsdp_mesh is not None and fsdp_mesh.ndim == 2:
        broadcast_from_first_replica([model], world_mesh['dp_replicate'].get_group())

    # ============================= Build FSDP EMA =========================
    logger.info("--> Initializing FSDP EMA")
    ema_model = FSDPEMAModel(
        model,
        decay=cfg.ema_decay,
    )
    logger.info("--> FSDP EMA initialized")

    # ============================FSDP for text encoder =========================
    logger.info("--> Initializing FSDP for text encoder")
    from torch.distributed.fsdp import fully_shard
    for layer in text_encoder.model.layers:
        fully_shard(layer, mesh=fsdp_mesh, reshard_after_forward=True)
    fully_shard(text_encoder.model, mesh=fsdp_mesh, reshard_after_forward=False)
    logger.info("--> FSDP for text encoder initialized")
    # ByT5 must not be FSDP-sharded: some ranks have glyph text to encode and others do not, and a rank that
    # skips the forward never joins the all-gather, so the job deadlocks.
    byt5_kwarg["byt5_model"] = byt5_kwarg["byt5_model"].to(device)

    # ============================= Optimizer =========================
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    muon_params = [p for p in params_to_optimize if p.ndim >= 2]
    adamw_params = [p for p in params_to_optimize if p.ndim < 2]
    logger.info(f"Muon optimizer: {len(muon_params)} muon params (ndim>=2), "
                f"{len(adamw_params)} adamw params (ndim<2), "
                f"total trainable = {len(params_to_optimize)}")
    optimizer = Muon(
        lr=cfg.learning_rate,
        wd=cfg.weight_decay,
        muon_params=muon_params,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=adamw_params,
        # Muon defaults these to (0.95, 0.95), while the ndim < 2 parameters train with AdamW's own defaults,
        # so they are passed explicitly.
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
    )

    if cfg.resume_checkpoint:
        for p in optimizer.param_groups:
            p["initial_lr"] = cfg.learning_rate

    logger.info(f"optimizer: {optimizer}")
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=cfg.max_train_steps,
        num_cycles=cfg.lr_num_cycles,
        power=cfg.lr_power,
        last_epoch=init_steps - 1,  # last_epoch is the step before the first update
    )

    # ============================== Load dataset ==============================
    video_dataset = MovieRollJsonlDataset(
        cfg.data_root,
        height=cfg.height,
        width=cfg.width,
        num_frames=cfg.sample_n_frames,
        caption_type_probs=dict(cfg.streaming_caption_type_probs),
        byt5_uncond_p=cfg.byt5_uncond_p,
        chunk_size=cfg.streaming_chunk_size,
        font_path=cfg.multilingual_prompt_format_font_path,
        color_path=cfg.multilingual_prompt_format_color_path,
        seed=cfg.global_seed,
        dp_rank=dp_rank,
    )
    video_loader, video_sampler = build_sft_dataloader(
        video_dataset,
        batch_size=cfg.batch_size,
        dp_degree=dp_degree,
        dp_rank=dp_rank,
        global_seed=cfg.global_seed,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        start_index=int(scalar_states.epoch_consumed_samples[_DATA_KEY]),
    )

    # ============================== Print Key Info. ==============================
    video_total_batch_size = cfg.batch_size * dp_degree
    video_num = len(video_dataset)

    params_count = model.params_count()
    print_training_configuration(cfg, logger, model, world_size, local_rank, rank,
                                 dp_degree, dp_rank, video_total_batch_size, video_num,
                                 video_loader)

    # ============================= Start training =============================
    for _ in range(cfg.num_train_epochs):
        cycle_states = CycleStates()
        data_iter = iterate_with_states(video_loader, video_sampler, scalar_states)
        optimizer.zero_grad()
        global_start_time = sync_cuda_time()
        # keeps the loss log comparable when gradient accumulation is on
        update_step_running_loss = 0.0
        last_step_loss = 0.0

        if cfg.dry_run:
            logger.info("=" * 80)
            logger.info("DRY RUN MODE ENABLED")
            logger.info("Will only iterate through data without training.")
            logger.info("=" * 80)

        while True:
            # `iterate_with_states` reshuffles forever: the only way out is the `return` below.
            start_time = sync_cuda_time()
            batch = next(data_iter)
            batch = sync_object_for_parallel_training(batch, parallel_dims=parallel_dims)  # syncing is cheap: <0.01s

            if cfg.dry_run:
                data_type = batch["data_type"][0]

                scalar_states.add(train_steps=1)
                is_update_step = scalar_states.train_steps % cfg.gradient_accumulation_steps == 0
                cycle_states.add(log_steps=1)

                if is_update_step:
                    scalar_states.add(update_steps=1)

                scalar_states = sync_object_for_parallel_training(scalar_states, parallel_dims=parallel_dims, force_object=True)  # syncing is cheap: <0.01s

                if is_update_step and scalar_states.update_steps % cfg.log_interval == 0:
                    end_time = sync_cuda_time()
                    sec_per_step = (end_time - global_start_time) / cycle_states.log_steps
                    logger.info(
                        f"[Dry Run] Progress: {scalar_states.update_steps}/{cfg.max_train_steps} | "
                        f"Step time: {sec_per_step:.2f}s | "
                        f"Data type: {data_type}"
                    )

                    cycle_states.reset_epoch_based_states()
                    global_start_time = sync_cuda_time()

                if scalar_states.update_steps >= cfg.max_train_steps:
                    logger.info("Dry run finished.")
                    if get_sequence_parallel_state():
                        destroy_sequence_parallel_group()
                    return

                continue

            if scalar_states.train_steps % cfg.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"Collect and empty cache (train_step #{scalar_states.train_steps}, "
                            f"update_step #{scalar_states.update_steps})")
                dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

            dataloader_time_ = sync_cuda_time()
            latents, model_kwargs, data_type, n_tokens, multitask_mask, mask_type = \
                prepare_model_inputs(batch, device, model, vae, text_encoder, byt5_kwarg, cfg.text_len,
                                     encode_generator=encode_generator)

            scalar_states = sync_object_for_parallel_training(scalar_states, parallel_dims=parallel_dims, force_object=True)  # syncing is cheap: < 0.01s
            # Sampling must fire only on the first micro-batch of an accumulation cycle: update_steps does
            # not change inside a cycle, so without the guard it would run grad_accum times per interval.
            is_first_micro_batch = (scalar_states.train_steps % cfg.gradient_accumulation_steps == 0)
            if cfg.sample_interval > 0 and is_first_micro_batch:
                if scalar_states.update_steps > 0 and (scalar_states.update_steps % cfg.sample_interval == 0) or (cfg.try_first_iter and scalar_states.update_steps == 0):
                    if not (cfg.try_first_iter and scalar_states.update_steps == 0):
                        ema_model.store(model)
                        ema_model.copy_to(model)
                        logger.info(f"--> Performing [EMA-normal] sample validation at step {scalar_states.update_steps}")
                        sample_validation(cfg, device, vae, text_encoder, model,
                                          batch, scalar_states.update_steps, logger, byt5_kwarg, dp_rank,
                                          multitask_mask, mask_type,
                                          parallel_dims=parallel_dims, ema_sampling=True, interactive_mode=False)
                        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
                        logger.info(f"--> [EMA-normal] sample validation finished at step {scalar_states.update_steps}")
                        if cfg.streaming_interactive_mode:
                            logger.info(f"--> Performing [EMA-interactive] sample validation at step {scalar_states.update_steps}")
                            sample_validation(cfg, device, vae, text_encoder, model,
                                              batch, scalar_states.update_steps, logger, byt5_kwarg, dp_rank,
                                              multitask_mask, mask_type,
                                              parallel_dims=parallel_dims, ema_sampling=True, interactive_mode=True)
                            dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
                            logger.info(f"--> [EMA-interactive] sample validation finished at step {scalar_states.update_steps}")
                        ema_model.restore(model)
                    if not cfg.streaming_interactive_mode:
                        logger.info(f"--> Performing [Normal] sample validation at step {scalar_states.update_steps}")
                        sample_validation(cfg, device, vae, text_encoder, model,
                                          batch, scalar_states.update_steps, logger, byt5_kwarg, dp_rank,
                                          multitask_mask, mask_type,
                                          parallel_dims=parallel_dims, ema_sampling=False, interactive_mode=False)
                        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
                        logger.info(f"--> [Normal] sample validation finished at step {scalar_states.update_steps}")
                    elif cfg.try_first_iter and scalar_states.update_steps == 0:
                        logger.info(f"--> Performing [Normal-interactive] sample validation at step {scalar_states.update_steps}")
                        sample_validation(cfg, device, vae, text_encoder, model,
                                          batch, scalar_states.update_steps, logger, byt5_kwarg, dp_rank,
                                          multitask_mask, mask_type,
                                          parallel_dims=parallel_dims, ema_sampling=False, interactive_mode=True)
                        dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
                        logger.info(f"--> [Normal-interactive] sample validation finished at step {scalar_states.update_steps}")

            cur_batch_size = latents.shape[0] / (world_size / dp_degree)
            data_time_ = sync_cuda_time()

            # broadcast a zero size tensor to indicate starting of step for profiling
            start_flag_tensor = torch.empty(0, device="cuda")
            if dist.is_initialized():
                dist.broadcast(start_flag_tensor, 0, async_op=True)

            with torch.autocast("cuda", torch.bfloat16):
                sync_random_states(parallel_dims)
                (
                    latents,
                    model_kwargs,
                    multitask_mask,
                    mask_type,
                ) = sync_object_for_parallel_training(
                    [
                        latents,
                        model_kwargs,
                        multitask_mask,
                        mask_type,
                    ],
                    parallel_dims=parallel_dims,
                )
                # parallel_dims is not optional: without it the timesteps drawn on the sp source rank are
                # never broadcast, and the other ranks keep training on t = 0.
                _, loss_dict = denoiser.training_losses(model, latents, model_kwargs, data_type=data_type,
                                                        multitask_mask=multitask_mask, mask_type=mask_type,
                                                        parallel_dims=parallel_dims)
            loss = 0
            if torch.isnan(loss_dict['loss']).any() or torch.isinf(loss_dict['loss']).any():
                print(f"train_step: {scalar_states.train_steps}, update_step: {scalar_states.update_steps}, "
                      f"Rank {rank}: Loss is NaN or Inf. id: {batch['id']}, "
                      f"Latent shape: {latents.shape}. Mean: {latents.mean()}. Max: {latents.max()}. Min: {latents.min()}")

            diffusion_loss = loss_dict['loss'].mean()
            loss += diffusion_loss

            per_chunk_losses_dict = loss_dict.get('per_chunk_losses', {})

            # defensive check
            sync_object_for_parallel_training(loss, parallel_dims=parallel_dims, debug_with_check=True)  # syncing is cheap: < 0.01s

            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps

            current_loss = loss.item()
            loss_threshold = cfg.loss_spike_threshold

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Rank {rank}: NaN/Inf loss detected at step {scalar_states.train_steps} "
                      f"update_step {scalar_states.update_steps}. "
                      f"Loss: {current_loss}, Data type: {data_type}, Batch size: {cur_batch_size}, "
                      f"id: {batch['id']}, Mask type: {mask_type}")

            elif current_loss > loss_threshold:
                t = loss_dict['t']
                print(f"Rank {rank}: High loss detected at step {scalar_states.train_steps} "
                      f"update_step {scalar_states.update_steps}. "
                      f"Loss: {current_loss:.6f} > {loss_threshold}, Data type: {data_type}, "
                      f"Batch size: {cur_batch_size} "
                      f"t: {t}, "
                      f"Mask type: {mask_type}, Diffusion loss: {diffusion_loss.item():.6f}")
                print(f"  Latent stats - Mean: {latents.mean():.6f}, Std: {latents.std():.6f}, "
                      f"Max: {latents.max():.6f}, Min: {latents.min():.6f}")

            forward_time_ = sync_cuda_time()

            scalar_states.add(train_steps=1)

            data_dtype = _DATA_KEY
            is_update_step = scalar_states.train_steps % cfg.gradient_accumulation_steps == 0

            # Accumulates this update step's already-scaled micro losses, so after G micros the sum is the
            # average raw loss. Reuses current_loss rather than a second .item() synchronisation.
            update_step_running_loss += current_loss

            loss.backward()
            if is_update_step:
                grad_norm_ = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.max_grad_norm,
                    foreach=True
                )

                # Under FSDP2 clip_grad_norm_ returns a DTensor; `.item()` alone reads this rank's partial sum.
                grad_norm = float(grad_norm_.full_tensor().item()) if hasattr(grad_norm_, "full_tensor") else grad_norm_.item()

                if torch.isnan(grad_norm_) or torch.isinf(grad_norm_):
                    print(f"Rank {rank}: NaN/Inf gradient norm detected at step {scalar_states.update_steps}. "
                          f"Grad norm: {grad_norm}, Loss: {current_loss:.6f}, Data type: {data_type}, "
                          f"Latent shape: {latents.shape}. Mean: {latents.mean()}. Max: {latents.max()}. Min: {latents.min()}")
                    # NaN/Inf grad: skip optimizer.step() and lr_scheduler.step(), so the LR does not drift
                    # while the model is not being updated.
                else:
                    optimizer.step()
                    lr_scheduler.step()

                optimizer.zero_grad()
                scalar_states.add(update_steps=1, current_run_update_steps=1)
                scalar_states.lr = optimizer.param_groups[0]["lr"]
                # logging only happens on an update step, so log_steps is only added there
                cycle_states.add(log_steps=1)
                ema_model.update(model)
                last_step_loss = update_step_running_loss
                update_step_running_loss = 0.0
            backward_time_ = sync_cuda_time()
            cycle_states.add(running_loss=current_loss)
            cycle_states.running_samples[data_dtype] += cur_batch_size
            cycle_states.running_tokens[data_dtype] += cur_batch_size * n_tokens
            cycle_states.add_streaming_loss(per_chunk_losses_dict)

            if is_update_step and scalar_states.update_steps % cfg.log_interval == 0:
                global_loss = all_reduce_sum(cycle_states.running_loss / cycle_states.log_steps, device) / world_size

                local_chunk_loss_keys = list(cycle_states.running_per_chunk_losses.keys())
                all_processes_chunk_loss_keys = [None for _ in range(world_size)]
                dist.all_gather_object(all_processes_chunk_loss_keys, local_chunk_loss_keys)
                all_chunk_loss_keys = sorted(set().union(*all_processes_chunk_loss_keys))

                global_per_chunk_losses = {}
                for key in all_chunk_loss_keys:
                    local_loss = cycle_states.running_per_chunk_losses.get(key, 0.0)
                    local_count = cycle_states.running_per_chunk_loss_count.get(key, 0)
                    total_loss = all_reduce_sum(local_loss, device)
                    total_count = all_reduce_sum(local_count, device)
                    if total_count > 0:
                        global_per_chunk_losses[key] = total_loss / total_count
                    else:
                        global_per_chunk_losses[key] = 0.0

                cum_samples, cum_tokens = {}, {}
                cum_samples[_DATA_KEY] = all_reduce_sum(cycle_states.running_samples[_DATA_KEY], device)
                cum_tokens[_DATA_KEY] = all_reduce_sum(cycle_states.running_tokens[_DATA_KEY], device)

                cum_samples_total = sum(cum_samples.values())
                cum_tokens_total = sum(cum_tokens.values())

                end_time = sync_cuda_time()
                samples_per_sec = cum_samples_total / (end_time - global_start_time)
                sec_per_step = (end_time - global_start_time) / cycle_states.log_steps
                steps_per_sec = cycle_states.log_steps / (end_time - global_start_time)

                scalar_states.consumed_samples_total[_DATA_KEY] += cum_samples[_DATA_KEY]
                scalar_states.epoch_consumed_samples[_DATA_KEY] += cum_samples[_DATA_KEY]
                scalar_states.consumed_tokens_total[_DATA_KEY] += cum_tokens[_DATA_KEY]
                scalar_states.add(
                    consumed_computations_attn=6 * params_count['attn+mlp'] * cum_tokens_total / C_SCALE,
                    consumed_computations_total=6 * params_count['total'] * cum_tokens_total / C_SCALE,
                )

                cycle_states.reset_epoch_based_states()
                global_start_time = sync_cuda_time()

            if rank == 0 and (is_update_step and scalar_states.update_steps % cfg.log_interval == 0):
                progress_info = {
                    "avg_loss": f"{global_loss:.8f}",
                    # the average raw loss per micro-batch of this update step
                    "step_loss": f"{last_step_loss:.8f}",
                    "grad_norm": f"{grad_norm:.10f}",
                    "step_time": f"{sec_per_step:.2f}s",
                    "steps_per_sec": f"{steps_per_sec:.2f}",
                    "samples_per_sec": f"{samples_per_sec:.2f}",
                    "dataloader_time": f"{dataloader_time_ - start_time:.2f}s",
                    "data_time": f"{data_time_ - dataloader_time_:.2f}s",
                    "forward_time": f"{forward_time_ - data_time_:.2f}s",
                    "backward_time": f"{backward_time_ - forward_time_:.2f}s",
                    "learning_rate": f"{lr_scheduler.get_last_lr()[0]:.8f}",
                    "current_batch_n_tokens": f"{n_tokens}",
                    "consumed_computations_attn": f"{scalar_states.consumed_computations_attn}",
                    "consumed_computations_total": f"{scalar_states.consumed_computations_total}",
                    "current_mask_type": mask_type,
                }
                for key in global_per_chunk_losses:
                    progress_info[f"avg_per_chunk_loss_{key}"] = f"{global_per_chunk_losses[key]:.8f}"
                progress_info[f"cum_samples_{_DATA_KEY}"] = f"{cum_samples[_DATA_KEY]}"
                progress_info[f"consumed_tokens_{_DATA_KEY}_total"] = f"{scalar_states.consumed_tokens_total[_DATA_KEY]}"
                progress_info[f"consumed_samples_{_DATA_KEY}_total"] = f"{scalar_states.consumed_samples_total[_DATA_KEY]}"
                progress_info[f"consumed_samples_{_DATA_KEY}_epoch"] = f"{scalar_states.epoch_consumed_samples[_DATA_KEY]}"
                progress_info[f"consumed_samples_{_DATA_KEY}_per_dp"] = f"{scalar_states.consumed_samples_per_dp[_DATA_KEY]}"

                logger.info(f"Progress: {scalar_states.update_steps}/{cfg.max_train_steps} | Details: {progress_info}")

                if wandb_run is not None:
                    wandb_logs = {
                        "avg_loss": global_loss,
                        "grad_norm": grad_norm,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "current_batch_n_tokens": n_tokens,
                    }
                    for key in global_per_chunk_losses:
                        wandb_logs[f"per_chunk_loss/{key}"] = global_per_chunk_losses[key]
                    wandb_module.log(wandb_logs, step=scalar_states.update_steps, commit=True)

                if tb_writer is not None:
                    tb_writer.add_scalar("Train/Steps/train_loss", global_loss, scalar_states.update_steps)
                    tb_writer.add_scalar("Train/Steps/step_loss", last_step_loss, scalar_states.update_steps)
                    tb_writer.add_scalar("Train/Steps/grad_norm", grad_norm, scalar_states.update_steps)
                    tb_writer.add_scalar(f"Train/Tokens/{_DATA_KEY}_train_loss", global_loss, scalar_states.consumed_tokens_total[_DATA_KEY])
                    tb_writer.add_scalar("Train/ComputationsAttn/train_loss", global_loss, scalar_states.consumed_computations_attn)
                    tb_writer.add_scalar("Train/ComputationsTotal/train_loss", global_loss, scalar_states.consumed_computations_total)

                    tb_writer.add_scalar("LR/learning_rate", lr_scheduler.get_last_lr()[0], scalar_states.update_steps)

                    tb_writer.add_scalar("Time/step_time", sec_per_step, scalar_states.update_steps)
                    tb_writer.add_scalar("Time/steps_per_sec", steps_per_sec, scalar_states.update_steps)
                    tb_writer.add_scalar("Time/samples_per_sec", int(samples_per_sec), scalar_states.update_steps)
                    tb_writer.add_scalar("Time/dataloader_time", dataloader_time_ - start_time, scalar_states.update_steps)
                    tb_writer.add_scalar("Time/data_time", data_time_ - dataloader_time_, scalar_states.update_steps)
                    tb_writer.add_scalar("Time/forward_time", forward_time_ - data_time_, scalar_states.update_steps)
                    tb_writer.add_scalar("Time/backward_time", backward_time_ - forward_time_, scalar_states.update_steps)

                    tb_writer.add_scalar(f"Data/{_DATA_KEY}_samples_total", scalar_states.consumed_samples_total[_DATA_KEY], scalar_states.update_steps)
                    tb_writer.add_scalar(f"Data/{_DATA_KEY}_samples_epoch", scalar_states.epoch_consumed_samples[_DATA_KEY], scalar_states.update_steps)

                    tb_writer.add_scalar(f"Data/{_DATA_KEY}_tokens_total", scalar_states.consumed_tokens_total[_DATA_KEY], scalar_states.update_steps)

            # The is_update_step guard is required: update_steps is the same for all grad_accum micro-batches
            # of a cycle, so without it one update step would save grad_accum checkpoints.
            if is_update_step and scalar_states.update_steps % cfg.checkpointing_steps == 0 and scalar_states.update_steps > 0:
                logger.info(f"--> save checkpoint at step {scalar_states.update_steps}, {cfg.output_dir}")
                save_checkpoint_without_optim(model, rank, cfg.output_dir, scalar_states, ema_model=ema_model)
                dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

            # The only way out of the loop, checked after any checkpoint due at this step was written.
            if _reached_max_train_steps(scalar_states.update_steps, cfg.max_train_steps):
                logger.info(f"Reached max_train_steps={cfg.max_train_steps}; stopping.")
                if get_sequence_parallel_state():
                    destroy_sequence_parallel_group()
                return


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="MovieRoll streaming SFT")
    parser.add_argument(
        "--config", type=str, required=True,
        help="path to a YAML config, e.g. movieroll/configs/sft/stage1_single_event.yaml",
    )
    parser.add_argument(
        "--override", type=str, action="append", default=[], metavar="KEY=VALUE",
        help="override a config key, e.g. --override learning_rate=2e-5 (repeatable)",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    _args = parse_args()
    main(load_sft_config(_args.config, _args.override))
