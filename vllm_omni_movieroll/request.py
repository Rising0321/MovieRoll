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

"""A generation request, validated, in MovieRoll's terms.

vLLM-Omni delivers a prompt (a string, or a dict with ``prompt`` / ``negative_prompt`` / ``multi_modal_data``) and
`OmniDiffusionSamplingParams`. MovieRoll's own inputs ride in ``extra_args`` (offline ``--extra-body`` of the shared examples,
``extra_params`` of ``/v1/videos``):

* ``events``: ordered event prompts of one video (two or more split it evenly and switch the prompt as it rolls on);
* ``video_length``: pixel frames, when ``num_frames`` cannot say it; ``duration`` (set by ``/v1/videos`` from ``seconds``).

The same function runs in the engine's pre-process (before a request is queued) and again in the pipeline. An invalid
request raises `OmniClientError` (a ``ValueError``): a 400 answer for that request, not a stage failure.

``guidance_scale`` is not one of MovieRoll's inputs: the model directory decides whether classifier-free guidance runs
(``use_cfg``), and the per-frame scale is AnnealCFG's (``cfg_min`` / ``cfg_max``). A request that sets it to anything
but vLLM-Omni's default of 1 gets a warning, and the value is not used. Likewise a negative prompt sent to a model
without guidance (Distilled, or Base with ``use_cfg: false``) gets a warning and is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# a ValueError answered with HTTP 400; unlike other errors it does not count towards the stage's failure circuit breaker
from vllm_omni.errors import OmniClientError

from vllm_omni_movieroll.profiles import VAE_SPATIAL, VAE_TEMPORAL, MovieRollSettings

DEFAULT_VIDEO_LENGTH = 721
# what vLLM-Omni puts in a request that leaves guidance_scale out (`OmniDiffusionRequest`); it is ignored silently
FRAMEWORK_GUIDANCE_SCALE = 1.0
MODEL_KEYS = frozenset({"events", "video_length", "duration"})
# keys the shared vLLM-Omni examples / the server put into extra_args for every model, accepted at these values only
GENERIC_DEFAULTS = {"sample_solver": "unipc", "lora_request": None}
# keys of the engine's start-up warm-up request
WARMUP_KEYS = frozenset({"cfg_text_scale", "cfg_img_scale"})
REFUSED_KEYS = {
    "aspect_ratio": "MovieRoll takes an explicit height and width (or its 480 x 864 default)",
    "short_edge": "MovieRoll takes an explicit height and width (or its 480 x 864 default)",
    "start_time_seconds": "MovieRoll always generates from the first frame",
    "generate_sound": "MovieRoll generates no audio",
    "sound_duration": "MovieRoll generates no audio",
}


@dataclass
class RollingRequest:
    events: list[str]
    negative_prompt: Optional[str]
    height: int
    width: int
    video_length: int
    seed: int
    image: Any = None                          # PIL image of the i2v condition
    num_inference_steps: Optional[int] = None  # Base only (None: the profile's)
    guided: bool = False                       # classifier-free guidance on: the directory's use_cfg (not in warm-up)
    fps: float = 24.0
    output_type: Optional[str] = None
    notes: list[str] = field(default_factory=list)   # one-line warnings for the log (e.g. ignored inputs)

    @property
    def prompt(self) -> str:
        return self.events[0]


def check_resolution(height: Any, width: Any) -> None:
    for name, value in (("height", height), ("width", width)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % VAE_SPATIAL:
            raise OmniClientError(f"{name} must be a positive multiple of {VAE_SPATIAL}, got {value!r}")


def check_video_length(video_length: Any, chunk_size: int, distilled: bool, anchored: bool) -> None:
    """The frame-count rules of `movieroll/generate.py` (`_check_video_length` / `_check_i2v_video_length`)."""
    if isinstance(video_length, bool) or not isinstance(video_length, int) or video_length <= 0 \
            or (video_length - 1) % VAE_TEMPORAL:
        raise OmniClientError(f"video_length must be positive and 4n+1 pixel frames, got {video_length!r}")
    latent_length = (video_length - 1) // VAE_TEMPORAL + 1
    if latent_length % chunk_size != 1:
        num_chunks = (latent_length - 1) // chunk_size
        nearest = [VAE_TEMPORAL * (n * chunk_size) + 1 for n in (num_chunks, num_chunks + 1) if n >= 1]
        raise OmniClientError(
            f"video_length {video_length} gives {latent_length} latent frames, which is not 1 modulo chunk_size="
            f"{chunk_size}; the nearest valid frame counts are {nearest}"
        )
    if (distilled or anchored) and latent_length < 1 + chunk_size:
        raise OmniClientError(
            f"video_length {video_length} is too short: the first chunk needs {1 + chunk_size} latent frames "
            f"(shortest video: {VAE_TEMPORAL * chunk_size + 1} frames)"
        )


def frames_for_duration(seconds: float, fps: float, chunk_size: int) -> int:
    """The valid frame count closest to ``seconds`` at ``fps`` (``seconds * fps + 1`` rounded to whole chunks)."""
    target = int(round(float(seconds) * float(fps))) + 1
    step = VAE_TEMPORAL * chunk_size
    chunks = max(1, int(round((target - 1) / step)))
    return chunks * step + 1


def _split_prompt(prompt: Any) -> tuple[Optional[str], Optional[str], Any]:
    if isinstance(prompt, str):
        return prompt, None, None
    if isinstance(prompt, dict):
        image = (prompt.get("multi_modal_data") or {}).get("image")
        if isinstance(image, (list, tuple)):
            if len(image) != 1:
                raise OmniClientError(f"image-to-video takes one input image, got {len(image)}")
            image = image[0]
        return prompt.get("prompt"), prompt.get("negative_prompt"), image
    raise TypeError(f"prompt must be a string or a dict, got {type(prompt).__name__}")


def _open_image(image: Any):
    if isinstance(image, str):
        from PIL import Image, ImageOps

        with Image.open(image) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    return image


def _seed_of(sampling_params: Any) -> int:
    seed = getattr(sampling_params, "seed", None)
    if seed is not None:
        return int(seed)
    generator = getattr(sampling_params, "generator", None)
    if isinstance(generator, (list, tuple)):
        if len(generator) != 1:
            raise OmniClientError("MovieRoll generates one video per request: give one generator")
        generator = generator[0]
    if generator is not None:
        return int(generator.initial_seed())
    raise OmniClientError("the request has neither a seed nor a generator")


def parse_request(prompt: Any, sampling_params: Any, settings: MovieRollSettings, *, warmup: bool = False) -> RollingRequest:
    """Validate a request with the CLI's rules and return it in MovieRoll's terms (raises `OmniClientError` with the reason)."""
    profile = settings.profile
    distilled = settings.distilled
    notes: list[str] = []
    text, negative, image = _split_prompt(prompt)
    image = _open_image(image)

    extra = dict(getattr(sampling_params, "extra_args", None) or {})
    for key, default in GENERIC_DEFAULTS.items():
        if key in extra:
            value = extra.pop(key)
            if value != default:
                raise OmniClientError(f"extra_args[{key!r}]={value!r} is not supported by MovieRoll (only {default!r})")
    extra.pop("lora_scale", None)
    if "flow_shift" in extra:
        shift = float(extra.pop("flow_shift"))
        expected = float(profile["timestep_shift"] if distilled else profile["flow_shift"])
        if shift != expected:
            raise OmniClientError(f"flow_shift={shift} is not this checkpoint's: MovieRoll's chain uses shift {expected}")
    for key, reason in REFUSED_KEYS.items():
        if key in extra:
            raise OmniClientError(f"extra_args[{key!r}] is not supported: {reason}")
    if warmup:
        for key in WARMUP_KEYS:
            extra.pop(key, None)
    unknown = set(extra) - MODEL_KEYS
    if unknown:
        raise OmniClientError(f"unknown extra_args {sorted(unknown)}; MovieRoll takes {sorted(MODEL_KEYS)}")

    events = extra.get("events")
    if events is None:
        events = [text]
    if isinstance(events, str) or not isinstance(events, Sequence) or not events:
        raise OmniClientError("events must be a non-empty list of prompts")
    events = list(events)
    if any(not isinstance(e, str) or not e.strip() for e in events):
        raise OmniClientError("every event prompt (and the prompt) must be a non-empty string")
    if text is not None and len(events) > 1 and text != events[0]:
        notes.append("the prompt differs from events[0]; the event list is used")

    height = getattr(sampling_params, "height", None)
    width = getattr(sampling_params, "width", None)
    if height is None and width is None:
        # the CLI's rule for a model folder, image-to-video included: the image is scaled to cover this size and
        # centre-cropped (`MovieRollPipeline.encode_image`)
        height, width = profile["height"], profile["width"]
    elif height is None or width is None:
        raise OmniClientError("give both height and width, or neither")
    check_resolution(height, width)

    fps_value = getattr(sampling_params, "fps", None)
    if isinstance(fps_value, (list, tuple)):
        raise OmniClientError("one fps per request")
    fps = float(fps_value) if fps_value else float(profile["video_fps"])
    frames = getattr(sampling_params, "num_frames", None)
    if "video_length" in extra:
        video_length = extra["video_length"]
    elif "duration" in extra and (not frames or frames <= 1 or frames in (DEFAULT_VIDEO_LENGTH,
                                                                           int(float(extra["duration"]) * fps))):
        # `/v1/videos` turns ``seconds`` into ``duration`` and fills ``num_frames`` with the frame count it would have
        # derived from it, or with MovieRoll's declared default (`video_defaults`); either one means "not given"
        video_length = frames_for_duration(extra["duration"], fps, settings.chunk_size)
    elif frames and frames > 1:
        video_length = int(frames)
    else:
        video_length = DEFAULT_VIDEO_LENGTH
    check_video_length(video_length, settings.chunk_size, distilled, image is not None)

    if int(getattr(sampling_params, "num_outputs_per_prompt", 1) or 1) != 1:
        raise OmniClientError("MovieRoll generates one video per request (num_outputs_per_prompt must be 1)")

    if negative and not settings.use_cfg:
        if not warmup:
            model = "distilled chain" if distilled else "base model directory (use_cfg: false)"
            notes.append(f"negative prompt ignored: this {model} runs no classifier-free guidance, so there is no "
                         "unconditional branch for it to condition")
        negative = None
    guidance = getattr(sampling_params, "guidance_scale", None)
    guidance_provided = bool(getattr(sampling_params, "guidance_scale_provided", False))
    if guidance_provided and guidance != FRAMEWORK_GUIDANCE_SCALE and not warmup:
        notes.append(
            f"guidance_scale={guidance} ignored: the model directory decides whether classifier-free guidance runs "
            f"(use_cfg; {'on' if settings.use_cfg else 'off'} for this {settings.sampling_mode} model), and its "
            "per-frame scale is AnnealCFG's (cfg_min / cfg_max)"
        )
    steps = getattr(sampling_params, "num_inference_steps", None)
    guided = settings.use_cfg and not warmup
    if distilled:
        ladder_steps = len(profile["denoising_step_list"])
        if steps not in (None, ladder_steps) and not warmup:
            notes.append(
                f"the distilled chain ignores num_inference_steps={steps}: it runs its fixed "
                f"{len(profile['denoising_step_list'])}-step ladder {list(profile['denoising_step_list'])}"
            )
        steps = None
    else:
        steps = int(steps) if steps is not None else int(profile["num_inference_steps"])
        if warmup:
            steps = settings.window_size
        if steps <= 0 or steps % settings.window_size:
            raise OmniClientError(f"num_inference_steps must be a positive multiple of window_size={settings.window_size}, "
                             f"got {steps}")

    return RollingRequest(
        events=events, negative_prompt=negative, height=int(height), width=int(width),
        video_length=int(video_length), seed=_seed_of(sampling_params), image=image, num_inference_steps=steps,
        guided=guided, fps=fps, output_type=getattr(sampling_params, "output_type", None), notes=notes,
    )
