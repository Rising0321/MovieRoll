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

"""A generation request, validated, in MovieRoll's terms. No framework import: the caller hands in the fields.

MovieRoll's one input the frameworks have no field for is the ordered list of event prompts of a long video.
It travels as a small JSON object in the prompt, the one channel that reaches the worker through every entry
point of sglang 0.5.10.post1 (Python API, command line and OpenAI video endpoint); the keys are the names
vLLM-Omni's adapter uses in its extra arguments::

    {"events": ["a fox trots through snow", "the fox lies down"]}

A plain string is event 1 of a single-event video. The video is split evenly across the events.

Everything is checked here, before any weight is touched, with the same rules and messages as
`movieroll/generate.py`; a field the framework offers that MovieRoll does not implement is refused rather than
ignored. Two fields are the exception, because SGLang's entry points fill them in for every model: the noise
device (the chain's own wins) and ``guidance_scale``. MovieRoll has no guidance strength to set per request: the
model directory decides whether classifier-free guidance runs (``use_cfg``), and the per-frame scale is AnnealCFG's
(``cfg_min`` / ``cfg_max``). A request's ``guidance_scale`` is ignored, with a warning when it is not SGLang's default.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import torch

from sglang_movieroll.rolling.geometry import (
    check_video_length,
    chunk_layout,
    event_split_indices,
)
from sglang_movieroll.rolling.profiles import Chain
from sglang_movieroll.rolling.tokens import EventCondition

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_LENGTH = 721
DEFAULT_SEED = 930
MAX_SEED = 2**32 - 1
# what a request that leaves guidance_scale out carries (SGLang's `SamplingParams` default); it is ignored silently
FRAMEWORK_GUIDANCE_SCALE = 1.0

# Sampling-parameter fields MovieRoll does not implement -> the value that means "not asked for".
UNSUPPORTED_FIELDS = {
    "num_outputs_per_prompt": 1,
    "true_cfg_scale": None,
    "guidance_rescale": 0.0,
    "cfg_normalization": 0.0,
    "boundary_ratio": None,
    "enable_teacache": False,
    "return_trajectory_latents": False,
    "return_trajectory_decoded": False,
}


@dataclass(frozen=True)
class RollingRequest:
    events: List[str]
    negative_prompt: str
    height: int
    width: int
    video_length: int
    seed: int
    image: Any = None  # the loaded input image of an image-to-video request
    num_inference_steps: int = 5
    fps: int = 24

    @property
    def anchored(self) -> bool:
        return self.image is not None


@dataclass
class RollingJob:
    """A request planned for the rolling loop, filled with its tensors as the stages run."""

    request: RollingRequest
    chunk_sizes: List[
        int
    ]  # latent frames of every output chunk (the anchor chunk first when anchored)
    event_split_indices: List[
        int
    ]  # boundaries between events, in output-chunk coordinates
    event_texts: List[str]  # the prompt of every kept event
    events: List[EventCondition] = field(
        default_factory=list
    )  # filled by the text-encoding stage
    anchor: Optional[torch.Tensor] = None  # filled by the image-encoding stage


def build_prompt(events: Sequence[str]) -> str:
    """The prompt string of a multi-event video (the JSON channel of the module docstring)."""
    return json.dumps({"events": list(events)}, ensure_ascii=False)


def _check_events(events: Any) -> List[str]:
    if not isinstance(events, list) or not events:
        raise ValueError("events must be a non-empty list of prompts")
    if any(not isinstance(e, str) or not e.strip() for e in events):
        raise ValueError("every event prompt must be a non-empty string")
    return list(events)


def parse_prompt(prompt: Any) -> List[str]:
    """The event prompts of a prompt string: plain text is one event, a leading ``{`` starts JSON."""
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
    if not prompt.lstrip().startswith("{"):
        return [prompt]
    try:
        body = json.loads(prompt)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"a prompt that starts with '{{' is read as MovieRoll's JSON form "
            f'{{"events": [...]}} and is not valid JSON ({error}); '
            f"start the prompt with another character to write plain text"
        ) from error
    if not isinstance(body, dict) or set(body) - {"events"}:
        raise ValueError(f"the JSON prompt takes only the key 'events', got {body!r}")
    return _check_events(body.get("events"))


def _check_resolution(height: Any, width: Any) -> None:
    for name, value in (("height", height), ("width", width)):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value % 16
        ):
            raise ValueError(f"{name} must be a positive multiple of 16, got {value!r}")


def parse_request(
    params: Any, chain: Chain, image: Any = None, *, anchored: Optional[bool] = None
) -> RollingRequest:
    """Validate ``params`` (a sampling-parameter object or the request that delegates to it) for ``chain``.

    ``image`` is the loaded input image of an image-to-video request. ``anchored=True`` without an image is the
    client-side check, before the image is loaded. A request without a size gets the profile's 480 x 864, an
    image-to-video one included, as MovieRoll's command line does for a model folder: the image is scaled to cover
    it and centre-cropped (`MovieRollImageVAEEncodingStage`).
    """
    anchored = image is not None if anchored is None else anchored
    for name, unset in UNSUPPORTED_FIELDS.items():
        value = getattr(params, name, unset)
        if value != unset:
            raise ValueError(f"{name}={value!r} is not supported by MovieRoll")
    device = getattr(params, "generator_device", None)
    wanted = "cpu" if chain.distilled else "cuda"
    if device is not None and device != wanted:
        # the OpenAI video schema injects generator_device="cuda" into every request, so this cannot be refused:
        # the chain's own device wins (a generator on the other device draws another stream than the one the
        # checkpoint was evaluated with)
        logger.warning(
            "generator_device=%r ignored: the %s chain draws its noise on '%s'",
            device,
            chain.sampling_mode,
            wanted,
        )

    events = parse_prompt(params.prompt)

    profile = chain.profile
    height, width = getattr(params, "height", None), getattr(params, "width", None)
    height = profile["height"] if height is None else height
    width = profile["width"] if width is None else width
    _check_resolution(height, width)

    video_length = getattr(params, "num_frames", None)
    video_length = DEFAULT_VIDEO_LENGTH if video_length is None else video_length
    check_video_length(video_length, chain.chunk_size, chain.sampling_mode, anchored)

    seed = getattr(params, "seed", None)
    if seed is None:
        seed = DEFAULT_SEED
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {MAX_SEED}], got {seed!r}")

    steps = chain.check_steps(getattr(params, "num_inference_steps", None))
    cfg = "on" if chain.use_cfg else "off"
    scale = getattr(params, "guidance_scale", None)
    if scale is not None and scale != FRAMEWORK_GUIDANCE_SCALE:
        logger.warning(
            "guidance_scale=%r ignored: the model directory decides whether classifier-free guidance runs "
            "(use_cfg; %s for this %s model), and its per-frame scale is AnnealCFG's (cfg_min / cfg_max)",
            scale,
            cfg,
            chain.sampling_mode,
        )
    negative = getattr(params, "negative_prompt", None) or ""
    if not chain.use_cfg and negative.strip():
        logger.warning(
            "classifier-free guidance is off for this %s model: the negative prompt is not used",
            chain.sampling_mode,
        )
    return RollingRequest(
        events=events,
        negative_prompt=negative.strip() if chain.use_cfg else "",
        height=int(height),
        width=int(width),
        video_length=int(video_length),
        seed=int(seed),
        image=image,
        num_inference_steps=steps,
        fps=int(getattr(params, "fps", None) or profile["video_fps"]),
    )


def plan_job(request: RollingRequest, chain: Chain) -> RollingJob:
    """Chunk layout and event boundaries of a request (what the rolling loop is told), before any model runs."""
    chunk_sizes = chunk_layout(
        request.video_length,
        chain.chunk_size,
        anchored=request.anchored,
        strict=chain.distilled,
    )
    splits, kept = event_split_indices(len(chunk_sizes), request.events)
    if len(kept) != len(request.events):
        dropped = sorted(set(range(len(request.events))) - set(kept))
        if chain.distilled:
            raise ValueError(
                "every event must keep at least one chunk; the video is too short for these events"
            )
        logger.warning("events %s keep no chunk and are dropped", dropped)
    return RollingJob(request, chunk_sizes, splits, [request.events[i] for i in kept])
