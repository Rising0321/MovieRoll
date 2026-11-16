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
"""Typed runtime contracts between the rolling pipeline and the DMD loss adapter.

Deliberately holds no model, scheduler or distributed imports, so these contracts stay unit-testable
without a GPU; torch is imported only under ``TYPE_CHECKING`` and function-locally. Do not hoist it. Every dataclass validates itself in ``__post_init__``
because a malformed rolling plan does not fail loudly at run time - it silently supervises the wrong frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class RollingMode(str, Enum):
    """Purpose of one rolling execution."""

    GENERATOR_DMD_TRAINING = "generator_dmd_training"
    FAKE_SCORE_TRAINING = "fake_score_training"
    SAMPLING = "sampling"
    I2V_ANCHOR_GENERATION = "i2v_anchor_generation"


class RollingTrainingObjective(str, Enum):
    """Loss prediction used by one training rollout."""

    ROLLING_DMD = "rolling_dmd"
    UNIFORM_TIMESTEP_DMD = "uniform_timestep_dmd"


@dataclass(frozen=True)
class DenoisingRange:
    """The shifted scheduler interval traversed by one selected window."""

    step_indices: tuple[int, ...]
    start_timestep: float
    end_timestep: float

    def __post_init__(self) -> None:
        if not self.step_indices:
            raise ValueError("DenoisingRange.step_indices must not be empty")
        if any(index < 0 for index in self.step_indices):
            raise ValueError("DenoisingRange.step_indices must be non-negative")
        if self.end_timestep > self.start_timestep:
            raise ValueError(
                "DenoisingRange requires end_timestep <= start_timestep "
                f"(got {self.end_timestep} > {self.start_timestep})"
            )


def build_denoising_range(
    *,
    step_indices: Sequence[int],
    timesteps: Sequence[Any],
) -> DenoisingRange:
    """Build the score re-noising interval for one rolling window."""

    indices = tuple(int(index) for index in step_indices)
    if not indices:
        raise ValueError("step_indices must not be empty")
    if len(timesteps) == 0:
        raise ValueError("timesteps must not be empty")

    last_index = len(timesteps) - 1
    if any(index < 0 or index > last_index for index in indices):
        raise ValueError(
            f"step index outside scheduler range [0, {last_index}]: {indices}"
        )

    def _as_float(value: Any) -> float:
        return float(value.item()) if hasattr(value, "item") else float(value)

    noisiest_index = min(indices)
    cleanest_index = max(indices)
    return DenoisingRange(
        step_indices=indices,
        start_timestep=_as_float(timesteps[noisiest_index]),
        end_timestep=(
            0.0
            if cleanest_index == last_index
            else _as_float(timesteps[cleanest_index + 1])
        ),
    )


@dataclass(frozen=True)
class WindowSelectionPlan:
    """A non-overlapping tiling of real rolling rounds."""

    start_phase: int
    selected_round_indices: tuple[int, ...]
    num_windows: int

    def __post_init__(self) -> None:
        if self.start_phase < 0:
            raise ValueError(
                f"WindowSelectionPlan.start_phase must be non-negative, got {self.start_phase}"
            )
        if self.num_windows != len(self.selected_round_indices):
            raise ValueError(
                "num_windows must equal len(selected_round_indices): "
                f"{self.num_windows} != {len(self.selected_round_indices)}"
            )
        if tuple(sorted(set(self.selected_round_indices))) != self.selected_round_indices:
            raise ValueError(
                "selected_round_indices must be unique and strictly increasing"
            )
        if any(index < 0 for index in self.selected_round_indices):
            raise ValueError("selected_round_indices must be non-negative")


def build_window_selection_plan(
    *,
    num_chunks: int,
    window_size: int,
    start_phase: int,
) -> WindowSelectionPlan:
    """Select every ``window_size``-th round of ``range(num_chunks)``, offset by ``start_phase``.

    Includes the prefill growth window and excludes drain rounds. ``num_chunks`` must be divisible by
    ``window_size`` so every shard-group phase runs the same number of backwards under HSDP.
    """

    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be > 0, got {num_chunks}")
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if not 0 <= start_phase < window_size:
        raise ValueError(
            f"start_phase must satisfy 0 <= phase < {window_size}, got {start_phase}"
        )
    if num_chunks % window_size != 0:
        raise ValueError(
            "num_chunks must be divisible by window_size so every shard-group "
            f"phase has the same K; got {num_chunks} % {window_size}"
        )

    selected = tuple(
        round_index
        for round_index in range(num_chunks)
        if round_index % window_size == start_phase
    )
    return WindowSelectionPlan(
        start_phase=start_phase,
        selected_round_indices=selected,
        num_windows=len(selected),
    )


@dataclass
class WindowPrediction:
    """One selected rolling window and all metadata needed by DMD scoring."""

    round_index: int
    output_start_frame: int
    predicted_clean_latents: "torch.Tensor"
    event_index: int
    denoising_range: DenoisingRange
    dmd_backward_completed: bool

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError(f"round_index must be >= 0, got {self.round_index}")
        if self.output_start_frame < 0:
            raise ValueError(
                f"output_start_frame must be >= 0, got {self.output_start_frame}"
            )
        if self.event_index < 0:
            raise ValueError(f"event_index must be >= 0, got {self.event_index}")


@dataclass(frozen=True)
class RollingContext:
    """Immutable inputs shared by every round of one rollout."""

    mode: RollingMode
    event_conditions: tuple[Any, ...]
    chunk_sizes: tuple[int, ...]
    timesteps: "torch.Tensor"
    event_index_by_chunk: tuple[int, ...]
    selected_round_indices: frozenset[int]
    training_objective: RollingTrainingObjective = (
        RollingTrainingObjective.ROLLING_DMD
    )
    uniform_step_index: int | None = None

    def __post_init__(self) -> None:
        if not self.event_conditions:
            raise ValueError("event_conditions must not be empty")
        if not self.chunk_sizes or any(size <= 0 for size in self.chunk_sizes):
            raise ValueError(f"chunk_sizes must be positive, got {self.chunk_sizes}")
        if len(self.event_index_by_chunk) != len(self.chunk_sizes):
            raise ValueError(
                "event_index_by_chunk must align with chunk_sizes: "
                f"{len(self.event_index_by_chunk)} != {len(self.chunk_sizes)}"
            )
        if any(
            event_index < 0 or event_index >= len(self.event_conditions)
            for event_index in self.event_index_by_chunk
        ):
            raise ValueError(
                "event_index_by_chunk contains an index outside event_conditions"
            )
        if any(
            round_index < 0 or round_index >= len(self.chunk_sizes)
            for round_index in self.selected_round_indices
        ):
            raise ValueError(
                "selected_round_indices must refer to real generated rounds"
            )
        if self.training_objective is RollingTrainingObjective.UNIFORM_TIMESTEP_DMD:
            if self.mode not in (
                RollingMode.GENERATOR_DMD_TRAINING,
                RollingMode.FAKE_SCORE_TRAINING,
            ):
                raise ValueError(
                    "uniform-timestep objective is valid only for training rollouts"
                )
            if (
                self.uniform_step_index is None
                or not 0 < self.uniform_step_index < len(self.timesteps)
            ):
                raise ValueError(
                    "uniform-timestep objective requires a nonzero scheduler step "
                    f"index, got {self.uniform_step_index}"
                )
        elif self.uniform_step_index is not None:
            raise ValueError(
                "rolling objective must not carry a uniform_step_index"
            )


@dataclass
class RollingState:
    """Mutable bookkeeping for a rolling loop."""

    kv_frame_position: int
    output_frame_count: int
    num_decoded_chunks: int
    finalized_chunks: list["torch.Tensor"]
    cached_output_chunk_indices: list[int]
    window_predictions: list[WindowPrediction]

    def __post_init__(self) -> None:
        if self.kv_frame_position < 0 or self.output_frame_count < 0:
            raise ValueError("rolling frame positions must be non-negative")
        if self.num_decoded_chunks < 0:
            raise ValueError("num_decoded_chunks must be non-negative")


@dataclass(frozen=True)
class RollingResult:
    """Typed output of a complete Stage3 rolling execution."""

    generated_latents: "torch.Tensor"
    window_predictions: tuple[WindowPrediction, ...]
    window_prediction_mask: "torch.Tensor | None"
    event_conditions: tuple[Any, ...]
    chunk_sizes: tuple[int, ...]
    dmd_backward_completed: bool
    event_split_indices: tuple[int, ...] = ()
    start_phase: int | None = None
    selected_round_indices: tuple[int, ...] = ()
    training_objective: RollingTrainingObjective = (
        RollingTrainingObjective.ROLLING_DMD
    )
    uniform_step_index: int | None = None

    def validate_for_mode(
        self, mode: RollingMode, *, deep_tensor_checks: bool = False
    ) -> None:
        """Validate structure; optionally compare full device-tensor contents."""
        import torch

        if not isinstance(mode, RollingMode):
            raise TypeError(f"mode must be RollingMode, got {type(mode)!r}")
        if self.generated_latents.ndim != 5:
            raise RuntimeError(
                "generated_latents must be [B,C,T,H,W], got "
                f"{tuple(self.generated_latents.shape)}"
            )
        if self.generated_latents.requires_grad:
            raise RuntimeError("generated_latents stored in RollingResult must be detached")
        if mode is RollingMode.SAMPLING:
            if self.window_prediction_mask is not None:
                raise RuntimeError(
                    "sampling must not allocate a window_prediction_mask"
                )
            if (
                self.training_objective
                is not RollingTrainingObjective.ROLLING_DMD
                or self.uniform_step_index is not None
            ):
                raise RuntimeError(
                    "sampling must not use uniform-timestep regularization"
                )
        else:
            if self.window_prediction_mask is None:
                raise RuntimeError(
                    f"{mode.value} requires a window_prediction_mask"
                )
            if self.window_prediction_mask.shape != self.generated_latents.shape:
                raise RuntimeError(
                    "window_prediction_mask must align with generated_latents: "
                    f"{tuple(self.window_prediction_mask.shape)} != "
                    f"{tuple(self.generated_latents.shape)}"
                )
            if self.window_prediction_mask.dtype is not torch.bool:
                raise RuntimeError("window_prediction_mask must be boolean")
        if self.training_objective is RollingTrainingObjective.UNIFORM_TIMESTEP_DMD:
            if self.uniform_step_index is None or self.uniform_step_index <= 0:
                raise RuntimeError(
                    "uniform-timestep result requires a nonzero step index"
                )
        elif self.uniform_step_index is not None:
            raise RuntimeError(
                "rolling result must not carry a uniform step index"
            )
        if any(size <= 0 for size in self.chunk_sizes):
            raise RuntimeError(f"chunk_sizes must be positive: {self.chunk_sizes}")
        if sum(self.chunk_sizes) != int(self.generated_latents.shape[2]):
            raise RuntimeError(
                "chunk_sizes do not cover generated_latents: "
                f"sum={sum(self.chunk_sizes)}, frames={self.generated_latents.shape[2]}"
            )
        expected_mask = (
            torch.zeros_like(self.window_prediction_mask, dtype=torch.bool)
            if deep_tensor_checks and self.window_prediction_mask is not None
            else None
        )
        for prediction in self.window_predictions:
            if prediction.predicted_clean_latents.ndim != 5:
                raise RuntimeError(
                    "predicted_clean_latents must be [B,C,T,H,W], got "
                    f"{tuple(prediction.predicted_clean_latents.shape)}"
                )
            if (
                prediction.predicted_clean_latents.shape[:2]
                != self.generated_latents.shape[:2]
                or prediction.predicted_clean_latents.shape[3:]
                != self.generated_latents.shape[3:]
            ):
                raise RuntimeError(
                    "window prediction batch/channel/spatial shape does not match "
                    f"generated_latents: {tuple(prediction.predicted_clean_latents.shape)} "
                    f"vs {tuple(self.generated_latents.shape)}"
                )
            if prediction.event_index >= len(self.event_conditions):
                raise RuntimeError(
                    "window prediction references an unavailable event condition"
                )
            if prediction.predicted_clean_latents.requires_grad:
                raise RuntimeError("stored window predictions must be detached")
            if prediction.output_start_frame >= self.generated_latents.shape[2]:
                raise RuntimeError("window prediction starts outside generated_latents")
            window_frames = int(prediction.predicted_clean_latents.shape[2])
            if window_frames <= 0:
                raise RuntimeError("window prediction must contain at least one frame")
            window_end = prediction.output_start_frame + window_frames
            if window_end > self.generated_latents.shape[2]:
                raise RuntimeError(
                    "training/sampling window extends beyond generated_latents: "
                    f"{prediction.output_start_frame}:{window_end} > "
                    f"{self.generated_latents.shape[2]}"
                )
            if expected_mask is not None:
                expected_mask[
                    :, :, prediction.output_start_frame:window_end
                ] = True
        if expected_mask is not None and not torch.equal(
            self.window_prediction_mask, expected_mask
        ):
            raise RuntimeError(
                "window_prediction_mask does not equal the union of window predictions"
            )
        rounds = tuple(prediction.round_index for prediction in self.window_predictions)
        if rounds != self.selected_round_indices:
            raise RuntimeError(
                "selected_round_indices must exactly match window prediction order: "
                f"{self.selected_round_indices} != {rounds}"
            )
        if tuple(sorted(set(rounds))) != rounds:
            raise RuntimeError("selected round indices must be unique and increasing")
        if any(
            left.output_start_frame + int(left.predicted_clean_latents.shape[2])
            != right.output_start_frame
            for left, right in zip(
                self.window_predictions, self.window_predictions[1:]
            )
        ):
            raise RuntimeError(
                "selected training windows must form a contiguous non-overlapping tiling"
            )
        if tuple(sorted(set(self.event_split_indices))) != self.event_split_indices:
            raise RuntimeError("event split indices must be unique and increasing")
        if any(
            index <= 0 or index >= len(self.chunk_sizes)
            for index in self.event_split_indices
        ):
            raise RuntimeError("event split index lies outside chunk boundaries")
        if len(self.event_split_indices) != len(self.event_conditions) - 1:
            raise RuntimeError(
                "event split count must match event conditions: "
                f"splits={len(self.event_split_indices)}, "
                f"events={len(self.event_conditions)}"
            )
        count = len(self.window_predictions)
        if mode is RollingMode.GENERATOR_DMD_TRAINING:
            if count == 0 or self.start_phase is None or not self.dmd_backward_completed:
                raise RuntimeError(
                    "generator DMD training requires selected windows with completed backward"
                )
            if not all(p.dmd_backward_completed for p in self.window_predictions):
                raise RuntimeError("all generator window predictions must have completed backward")
        elif mode is RollingMode.FAKE_SCORE_TRAINING:
            if count == 0 or self.start_phase is None or self.dmd_backward_completed:
                raise RuntimeError(
                    "fake-score training requires detached windows without generator backward"
                )
            if any(p.dmd_backward_completed for p in self.window_predictions):
                raise RuntimeError(
                    "fake-score predictions must not report generator backward"
                )
        elif mode is RollingMode.SAMPLING:
            if (
                count != 0
                or self.start_phase is not None
                or self.dmd_backward_completed
                or any(p.dmd_backward_completed for p in self.window_predictions)
                or self.selected_round_indices
            ):
                raise RuntimeError(
                    "sampling must return only fully drained latents without "
                    "training window predictions"
                )
        elif mode is RollingMode.I2V_ANCHOR_GENERATION:
            raise RuntimeError("i2v anchor generation returns an anchor, not RollingResult")

    @property
    def event_split_pixel_frames(self) -> tuple[int, ...]:
        """Convert chunk-boundary indices to 1-based pixel-frame boundaries."""

        boundaries = []
        for chunk_index in self.event_split_indices:
            latent_boundary = sum(self.chunk_sizes[:chunk_index])
            boundaries.append((latent_boundary - 1) * 4 + 1)
        return tuple(boundaries)


class DMDWindowBackwardHandler(Protocol):
    """Synchronous callback used by generator DMD training (the per-window half of CuLT-DMD).

    `on_window` is called as soon as a window has been predicted, so its loss can be backpropagated and its
    graph freed before the next window is rolled out. Generator weights stay fixed for the whole rollout;
    gradients only accumulate.
    """

    def on_rollout_start(self, *, num_windows: int) -> None:
        ...

    def on_window(
        self,
        *,
        window_index: int,
        expected_num_windows: int,
        predicted_clean_window: "torch.Tensor",
        output_start_frame: int,
        history_chunks: tuple["torch.Tensor", ...],
        previous_window_predictions: tuple[WindowPrediction, ...],
        prompt_condition: Any,
        teacher_unconditional_condition: Any,
        denoising_range: DenoisingRange,
    ) -> None:
        ...


def splice_window_predictions(
    base_latents: "torch.Tensor",
    window_predictions: Sequence[WindowPrediction],
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Splice predictions in order and return the matching boolean coverage mask."""

    import torch

    generated = base_latents
    mask = torch.zeros_like(base_latents, dtype=torch.bool)
    total_frames = int(base_latents.shape[2])
    for prediction in window_predictions:
        start = prediction.output_start_frame
        if start >= total_frames:
            raise ValueError(
                f"window starts outside generated latents: {start} >= {total_frames}"
            )
        window = prediction.predicted_clean_latents
        if window.ndim != base_latents.ndim:
            raise ValueError(
                f"window rank {window.ndim} does not match base rank {base_latents.ndim}"
            )
        if window.shape[:2] != base_latents.shape[:2] or window.shape[3:] != base_latents.shape[3:]:
            raise ValueError(
                "window batch/channel/spatial shape does not match base latents: "
                f"{tuple(window.shape)} vs {tuple(base_latents.shape)}"
            )
        end = min(start + int(window.shape[2]), total_frames)
        length = end - start
        if length <= 0:
            raise ValueError(
                f"window has no in-range frames: start={start}, frames={window.shape[2]}"
            )
        generated = torch.cat(
            (generated[:, :, :start], window[:, :, :length], generated[:, :, end:]),
            dim=2,
        )
        mask[:, :, start:end] = True
    return generated, mask


def ordered_event_columns(fieldnames: Sequence[str]) -> tuple[str, ...]:
    """Return a contiguous numeric ``event_1..event_N`` column sequence."""

    numbered: dict[int, str] = {}
    malformed = []
    for name in fieldnames:
        if not name.startswith("event_"):
            continue
        suffix = name.removeprefix("event_")
        if not suffix.isdigit() or str(int(suffix)) != suffix:
            malformed.append(name)
            continue
        index = int(suffix)
        if index in numbered:
            raise ValueError(f"sampling CSV contains duplicate event column: {name}")
        numbered[index] = name
    if malformed:
        raise ValueError(f"sampling CSV contains malformed event columns: {malformed}")
    if not numbered:
        raise ValueError("sampling CSV must contain at least event_1")
    maximum = max(numbered)
    expected = set(range(1, maximum + 1))
    if set(numbered) != expected:
        raise ValueError(
            "sampling event columns must be contiguous from event_1; "
            f"found {sorted(numbered)}"
        )
    return tuple(numbered[index] for index in range(1, maximum + 1))
