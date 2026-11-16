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

"""Phase-aware Stage3 DMD prompt data and synchronized batch decisions."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from movieroll.utils.communications import bcast_in_group, is_group_rank0

logger = logging.getLogger(__name__)


class DMDTrainingPhase(str, Enum):
    SHORT_SINGLE_EVENT = "short_single_event"
    LONG_DUAL_EVENT = "long_dual_event"

    @classmethod
    def ordered(cls) -> tuple["DMDTrainingPhase", ...]:
        return (cls.SHORT_SINGLE_EVENT, cls.LONG_DUAL_EVENT)


@dataclass(frozen=True)
class DMDPhaseSpec:
    phase: DMDTrainingPhase
    max_steps: int
    video_length: int
    latent_frames: int
    event_count: int
    csv_path: Path
    prompt_column: Optional[str] = None
    event_columns: tuple[str, ...] = ()
    switch_frame_min: Optional[int] = None
    switch_frame_max: Optional[int] = None


@dataclass(frozen=True)
class TrainBatchSpec:
    phase: DMDTrainingPhase
    prompts: list[str]
    event_prompts: list[str]
    event_frame_ranges: Optional[list[tuple[int, int]]]
    num_frames: int  # LATENT frames (= DMDPhaseSpec.latent_frames); video_length is the pixel count
    video_length: int
    height: int
    width: int
    task_type: str

    def __post_init__(self) -> None:
        if len(self.prompts) != 1:
            raise ValueError(
                f"Stage3 TrainBatchSpec requires one prompt (Stage-3 rollouts are single-prompt), "
                f"got {len(self.prompts)}"
            )
        if not self.prompts[0].strip():
            raise ValueError("Stage3 TrainBatchSpec prompt must be non-empty")
        if self.task_type not in {"t2v", "i2v"}:
            raise ValueError(f"unsupported Stage3 task_type: {self.task_type!r}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"Stage3 resolution must be positive, got {self.height}x{self.width}"
            )
        if self.video_length != (self.num_frames - 1) * 4 + 1:
            raise ValueError(
                "Stage3 pixel/latent frame mismatch: "
                f"{self.video_length} pixel vs {self.num_frames} latent"
            )

        if len(self.event_prompts) not in (1, 2):
            raise ValueError(
                "Stage3 batches have one or two events (the batch decision stream draws one "
                f"switch frame), got {len(self.event_prompts)}"
            )
        if any(not event.strip() for event in self.event_prompts):
            raise ValueError("Stage3 event prompts must be non-empty")
        if self.prompts[0] != self.event_prompts[0]:
            raise ValueError("primary prompt must equal event_prompts[0]")

        if len(self.event_prompts) == 1:
            if self.event_frame_ranges is not None:
                raise ValueError("single-event training must not define event_frame_ranges")
        else:
            ranges = self.event_frame_ranges
            if ranges is None or len(ranges) != 2:
                raise ValueError(
                    "two-event training requires exactly two event frame ranges"
                )
            first, second = ranges
            if (
                first[0] != 0
                or first[1] != second[0]
                or second[1] != self.video_length
                or not 0 < first[1] < self.video_length
            ):
                raise ValueError(
                    "two-event ranges must be contiguous and cover the full "
                    f"pixel timeline, got {ranges}"
                )


class EventPromptCsvDataset(Dataset):
    """Paired event-prompt dataset (multi-event / prompt-switch training).

    Reads a CSV whose listed columns give **consecutive events of the same scene** (default columns
    event_1 / event_2, the second one usually starting with "In a continuation of ..."). That is exactly
    the semantics multi-event training needs: a smooth transition between related events, and **not** a
    hard cut between two unrelated scenes.

    Every sample must really carry the configured number of non-empty events. The number of events
    decides how many times the per-event prompt is encoded and how many event-switch KV recomputes run,
    and both are collectives on the FSDP shard group: if some samples silently degrade to fewer events,
    the ranks of one shard group enqueue a different number of collectives and NCCL deadlocks. Incomplete
    rows are therefore dropped loudly at construction time instead of surfacing mid-training.

    Row order is the CSV order: the DistributedSampler owns the ordering, and a per-process python
    shuffle would map one sampler index to different prompts on each rank.
    """

    def __init__(
        self,
        csv_path: str,
        event_columns: list[str],
    ):
        cols = list(event_columns)
        if len(cols) < 2:
            raise ValueError(f"event_columns needs at least 2 columns, got {cols}")

        # CSV fields contain newlines (the style suffix is often written as "\n\nCinematic ... style."), so
        # the csv module has to do the quoting-aware parse; splitting by line would tear one prompt apart.
        # encoding="utf-8-sig": CSVs exported from Excel often carry a UTF-8 BOM, which would rename the
        # **first** column to a BOM-prefixed one ("\ufeffindex", "\ufeffevent_1") and break its lookup:
        # the event columns below, or the r.get("index") of the drop diagnostics. utf-8-sig behaves
        # identically on files without a BOM, so using it unconditionally is simply safer.
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            missing = [c for c in cols if c not in fields]
            if missing:
                raise ValueError(
                    f"{csv_path} is missing columns {missing}; actual columns: {fields}")
            # A duplicated column name must raise: `c not in fields` is equally false for a duplicated
            # column, so the missing-column check cannot see it, while csv.DictReader keeps only the
            # **last** same-named column -- a CSV that writes event_2 twice would silently train the
            # content of the later column, leaving no trace in the logs.
            dup = sorted({c for c in cols if fields.count(c) > 1})
            if dup:
                raise ValueError(
                    f"{csv_path} has duplicate column names {dup} (csv.DictReader keeps only the last "
                    f"one, so the wrong content is read silently); actual columns: {fields}")
            rows = list(reader)

        self.events: list[list[str]] = []
        dropped = []
        identical = 0
        for i, r in enumerate(rows):
            vals = [(r.get(c) or "").strip() for c in cols]
            if all(vals):
                # Two identical events = a switch where "nothing changed": the sample still has the full
                # number of events (so the collective counts stay aligned), but that step's switch
                # supervision is empty. Count and warn, do not drop -- dropping would change the sample
                # count, and such samples are harmless, just uninformative.
                if len(set(vals)) < len(vals):
                    identical += 1
                self.events.append(vals)
            else:
                # Row-number fallback: without an index column (or when the BOM renamed it), at least
                # report which row it was.
                dropped.append((i, r.get("index", i), [c for c, v in zip(cols, vals) if not v]))

        if not self.events:
            raise ValueError(f"{csv_path} has no row that carries all of {cols}")

        logger.info(f"Loaded {len(self.events)} paired-event samples from {csv_path} (columns={cols})")
        if dropped:
            # Say it out loud: silently dropping samples turns "I think I am training on 5000 rows" into
            # 4999 with nothing left to trace.
            logger.warning(f"dropped {len(dropped)} row(s) with an empty event column; "
                           f"first few (row_idx, index, empty_cols): {dropped[:3]}")
        if identical:
            logger.warning(f"{identical} row(s) have identical event texts -> those steps "
                           f"train a no-op 'switch' (kept, but they carry no switching signal)")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        ev = self.events[idx]
        return {
            # prompts: the first event -- used by the single-prompt path (and as the pipeline fallback)
            "prompts": ev[0],
            # event_prompts: consumed by the trainer's event extraction; switching needs >= 2
            "event_prompts": list(ev),
        }


class SinglePromptCsvDataset(Dataset):
    """Read a single-event prompt CSV for either training phase."""

    def __init__(self, csv_path: str | Path, prompt_column: str):
        self.csv_path = Path(csv_path)
        self.prompt_column = str(prompt_column)
        prompts: list[str] = []
        with self.csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if self.prompt_column not in fields:
                raise ValueError(
                    f"{self.csv_path} is missing prompt column {self.prompt_column!r}"
                )
            if fields.count(self.prompt_column) != 1:
                raise ValueError(
                    f"{self.csv_path} contains a duplicate {self.prompt_column!r} column"
                )
            for row_index, row in enumerate(reader, start=2):
                prompt = (row.get(self.prompt_column) or "").strip()
                if not prompt:
                    raise ValueError(
                        f"{self.csv_path}: row {row_index} has an empty "
                        f"{self.prompt_column!r} value"
                    )
                prompts.append(prompt)
        if not prompts:
            raise ValueError(f"{self.csv_path} contains no training prompts")
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        prompt = self.prompts[index]
        return {"prompts": prompt, "event_prompts": [prompt]}


def phase_spec_from_config(config, phase: DMDTrainingPhase) -> DMDPhaseSpec:
    raw = config.dmd_phases[phase.value]
    event_columns = tuple(raw.get("event_columns", ()) or ())
    return DMDPhaseSpec(
        phase=phase,
        max_steps=int(raw.max_steps),
        video_length=int(raw.video_length),
        latent_frames=int(raw.latent_frames),
        event_count=int(raw.event_count),
        csv_path=Path(str(raw.csv)),
        prompt_column=raw.get("prompt_column"),
        event_columns=event_columns,
        switch_frame_min=(
            int(raw.switch_frame_min) if raw.get("switch_frame_min") is not None else None
        ),
        switch_frame_max=(
            int(raw.switch_frame_max) if raw.get("switch_frame_max") is not None else None
        ),
    )


class DMDDataModule:
    """Own both fixed-length Stage3 phases and activate one dataset at a time."""

    def __init__(self, config, *, device, parallel_context, logger):
        self.config = config
        self.device = device
        self.parallel = parallel_context
        self.logger = logger
        self.phase: Optional[DMDTrainingPhase] = None
        self.phase_spec: Optional[DMDPhaseSpec] = None
        self.dataset = None
        self.sampler = None
        self.loader = None
        self._iterator = None
        self._epoch = 0
        self._batches_consumed_in_epoch = 0
        self._decision_counter = 0

    def configure_phase(self, phase: DMDTrainingPhase) -> DMDPhaseSpec:
        if not isinstance(phase, DMDTrainingPhase):
            phase = DMDTrainingPhase(phase)
        if phase is self.phase and self.phase_spec is not None and self._iterator is not None:
            return self.phase_spec
        spec = phase_spec_from_config(self.config, phase)

        if spec.event_count not in (1, 2):
            raise ValueError(
                f"{phase.value} event_count must be 1 or 2 (the batch decision stream draws one "
                f"switch frame), got {spec.event_count}"
            )

        if spec.event_count == 1:
            if not spec.prompt_column:
                raise ValueError(
                    f"{phase.value} single-event data requires prompt_column"
                )
            dataset = SinglePromptCsvDataset(spec.csv_path, spec.prompt_column)
        else:
            if len(spec.event_columns) != spec.event_count:
                raise ValueError(
                    f"{phase.value} expects {spec.event_count} event columns, "
                    f"got {spec.event_columns}"
                )
            dataset = EventPromptCsvDataset(
                csv_path=str(spec.csv_path),
                event_columns=list(spec.event_columns),
            )
            if any(len(events) != spec.event_count for events in dataset.events):
                raise ValueError(
                    f"every {phase.value} sample must contain {spec.event_count} events"
                )

        # Always use an epoch-addressable sampler, including dp_degree=1, so
        # mid-phase resume can reconstruct the exact batch cursor.
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=max(1, int(self.parallel.dp_degree)),
            rank=int(self.parallel.dp_rank),
            shuffle=True,
            drop_last=True,
        )

        self.phase = phase
        self.phase_spec = spec
        self.dataset = dataset
        self.sampler = sampler
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(self.config.data.batch_size),
            sampler=sampler,
            shuffle=False,
            num_workers=int(self.config.data.num_workers),
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            collate_fn=self._collate,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
        if len(self.loader) == 0:
            # Both drop_last flags would otherwise hand every replica an empty epoch, and the first next_batch()
            # would end the run on a bare StopIteration.
            dp_degree = max(1, int(self.parallel.dp_degree))
            batch_size = int(self.config.data.batch_size)
            raise ValueError(
                f"{spec.csv_path} has {len(dataset)} row(s), but {phase.value} needs at least "
                f"{dp_degree * batch_size}: one batch of {batch_size} for each of the {dp_degree} data-parallel "
                "replicas"
            )
        self._epoch = 0
        self._batches_consumed_in_epoch = 0
        self.sampler.set_epoch(self._epoch)
        self._iterator = iter(self.loader)
        self.logger.info(
            "Configured DMD phase %s: %d samples, %d latent/%d pixel frames, %d event(s)",
            phase.value,
            len(dataset),
            spec.latent_frames,
            spec.video_length,
            spec.event_count,
        )
        return spec

    @staticmethod
    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "prompts": [sample["prompts"] for sample in batch],
            "event_prompts": [sample["event_prompts"] for sample in batch],
        }

    def _next_raw_batch(self) -> dict[str, Any]:
        if self._iterator is None:
            raise RuntimeError("configure_phase() must be called before next_batch()")
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._epoch += 1
            self.sampler.set_epoch(self._epoch)
            self._iterator = iter(self.loader)
            self._batches_consumed_in_epoch = 0
            batch = next(self._iterator)
        self._batches_consumed_in_epoch += 1
        if self.parallel.parallel_dims is None:
            return batch
        from movieroll.utils.parallel_sync import sync_object_for_parallel_training

        return sync_object_for_parallel_training(
            batch, parallel_dims=self.parallel.parallel_dims
        )

    def state_dict(self) -> dict[str, int | str]:
        """Return the exact phase-local data and decision cursor."""

        return {
            "phase": self.phase.value if self.phase is not None else "",
            "epoch": int(self._epoch),
            "batches_consumed_in_epoch": int(
                self._batches_consumed_in_epoch
            ),
            "decision_counter": int(self._decision_counter),
        }

    def load_state_dict(
        self, state: dict, *, restore_cursor: bool = True
    ) -> None:
        """Restore decision RNG and the DataLoader cursor when phases match."""

        self._decision_counter = int(state.get("decision_counter", 0))
        if not restore_cursor:
            return
        if self.phase is None or self.loader is None or self.sampler is None:
            raise RuntimeError(
                "configure_phase() must run before restoring DMD data state"
            )
        saved_phase = str(state.get("phase", ""))
        if saved_phase != self.phase.value:
            # A phase-complete checkpoint resumes at the next phase with a
            # fresh data cursor, while retaining the global decision stream.
            return

        epoch = int(state.get("epoch", 0))
        consumed = int(state.get("batches_consumed_in_epoch", 0))
        if epoch < 0 or consumed < 0 or consumed > len(self.loader):
            raise ValueError(
                "invalid DMD data cursor: "
                f"epoch={epoch}, consumed={consumed}, "
                f"batches_per_epoch={len(self.loader)}"
            )
        self._epoch = epoch
        self._batches_consumed_in_epoch = 0
        self.sampler.set_epoch(epoch)
        self._iterator = iter(self.loader)
        for _ in range(consumed):
            try:
                next(self._iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "checkpoint DMD data cursor exceeds restored epoch"
                ) from exc
            self._batches_consumed_in_epoch += 1

    _DECISION_SEED_DOMAIN = 7_777_777

    def _draw_tensor(self, *, high: Optional[int], group) -> torch.Tensor:
        group_id = 0
        if torch.distributed.is_initialized() and group is not None:
            group_id = int(torch.distributed.get_process_group_ranks(group)[0])
        seed = (
            42
            + self._DECISION_SEED_DOMAIN
            + group_id * 1_000_003
            + self._decision_counter * 10_007
        )
        self._decision_counter += 1
        sync_in_group = torch.distributed.is_initialized() and group is not None
        if not sync_in_group or is_group_rank0(group):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            value = (
                torch.rand(1, generator=generator)
                if high is None
                else torch.randint(0, high, (1,), generator=generator)
            ).to(self.device)
        else:
            dtype = torch.float32 if high is None else torch.long
            value = torch.empty(1, device=self.device, dtype=dtype)
        if sync_in_group:
            bcast_in_group(value, group)
        return value

    def _draw_uniform(self, *, group) -> float:
        value = self._draw_tensor(high=None, group=group)
        return float(value.item())

    def _draw_int(self, low: int, high: int, *, group) -> int:
        if high <= low:
            raise ValueError(f"invalid integer draw range [{low}, {high})")
        value = self._draw_tensor(high=high - low, group=group)
        return low + int(value.item())

    def next_batch(self) -> TrainBatchSpec:
        """Return the next one-prompt batch of the active phase, with its batch-level decisions.

        The decisions come from one counter-seeded stream: each draw seeds a fresh CPU generator from
        ``_decision_counter`` and the drawing group's rank 0, advances the counter by one, and is
        broadcast from that rank 0. Per batch the stream advances by two (single event) or three
        (two events), in this order:

        1. an empty slot: the counter is advanced, nothing is drawn or broadcast, and the height / width come
           from ``video.resolution``;
        2. the task, t2v or i2v, drawn over the shard group;
        3. two events only: the switch frame in ``[switch_frame_min, switch_frame_max)``, drawn over the
           shard group.

        The counter is saved with the checkpoint. The empty slot keeps the task and switch-frame seeds of
        every batch where they have always been; removing it would shift every later draw and break the
        resume alignment of existing checkpoints.
        """
        if self.phase is None or self.phase_spec is None:
            raise RuntimeError("DMD data phase has not been configured")
        batch = self._next_raw_batch()
        prompts = list(batch["prompts"])
        if len(prompts) != 1:
            raise ValueError(
                f"Stage3 requires data.batch_size=1 (Stage-3 rollouts are single-prompt), got {len(prompts)}"
            )
        events = list(batch["event_prompts"][0])
        if len(events) != self.phase_spec.event_count or not all(events):
            raise ValueError(
                f"{self.phase.value} requires {self.phase_spec.event_count} non-empty "
                f"event(s), got {events}"
            )

        # The empty slot of the decision stream (see the docstring): the task and switch-frame seeds
        # below -- and therefore existing checkpoints -- stay where they are. No value is drawn and no
        # collective is entered, on every rank alike.
        self._decision_counter += 1
        height, width = self.config.video.resolution

        task = self.config.video.task_sampling
        # i2v adds an anchor rollout, so the full FSDP shard group must take the
        # same branch to keep model-forward collectives aligned.
        task_type = (
            "t2v"
            if self._draw_uniform(group=self.parallel.shard_group) < float(task.t2v_prob)
            else "i2v"
        )

        event_frame_ranges = None
        if self.phase_spec.event_count == 2:
            low = int(self.phase_spec.switch_frame_min)
            high = int(self.phase_spec.switch_frame_max)
            switch_frame = self._draw_int(
                low, high, group=self.parallel.shard_group
            )
            event_frame_ranges = [
                (0, switch_frame),
                (switch_frame, self.phase_spec.video_length),
            ]

        return TrainBatchSpec(
            phase=self.phase,
            prompts=prompts,
            event_prompts=events,
            event_frame_ranges=event_frame_ranges,
            num_frames=self.phase_spec.latent_frames,
            video_length=self.phase_spec.video_length,
            height=int(height),
            width=int(width),
            task_type=task_type,
        )
