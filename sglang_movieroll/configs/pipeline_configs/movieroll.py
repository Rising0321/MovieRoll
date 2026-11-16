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

"""MovieRoll's `PipelineConfig` for SGLang: the static model configuration and the callbacks the stages use.

One class per released chain, because a model directory holds one checkpoint: `MovieRollPipelineConfig` is the
base chain (50 steps with the per-frame AnnealCFG guidance), `MovieRollDistilledPipelineConfig` the distilled
one (a five-step ladder whose guidance is baked into the weights, and the noise drawn on the CPU because that
is the stream the checkpoint was evaluated with).

The text callbacks are the conditioning MovieRoll was trained with, expressed as the hooks SGLang's
`TextEncodingStage` calls: the video chat template and the crop of its prefix, the hidden layer two before the
last for the prompt encoder, and the glyph (byT5) prompt format for the second one.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)

from sglang_movieroll.configs.models.dits.movieroll import MovieRollDitConfig
from sglang_movieroll.configs.models.encoders.movieroll import (
    QWEN_CROP_START,
    MovieRollByT5Config,
    MovieRollQwenConfig,
)
from sglang_movieroll.configs.models.vaes.hunyuanvideo15 import HunyuanVideo15VAEConfig
from sglang_movieroll.rolling.profiles import (
    BASE,
    DISTILLED,
    NUM_TRAIN_TIMESTEPS,
    Chain,
)

# The video prompt template of the released checkpoints (`movieroll.models.text_encoders.PROMPT_TEMPLATE`,
# entry "li-dit-encode-video-json"), as the chat messages the tokenizer's template takes.
PROMPT_TEMPLATE_VIDEO = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video.",
    },
    {"role": "user", "content": "{}"},
]

# Quoted text picked up as glyph input: ASCII straight quotes and curly quotes
# (`movieroll.models.text_encoders.glyph_text`).
GLYPH_QUOTE_PATTERN = r"\"(.*?)\"|“(.*?)”"


def extract_glyph_texts(prompt: str) -> List[str]:
    """The quoted strings of ``prompt`` in order; repeats are dropped when there is more than one."""
    matches = re.findall(GLYPH_QUOTE_PATTERN, prompt)
    texts = [m[0] or m[1] for m in matches]
    return list(dict.fromkeys(texts)) if len(texts) > 1 else texts


def qwen_preprocess_text(prompt: str):
    """The prompt as the template's chat messages; an empty prompt becomes a single space, as MovieRoll does."""
    messages = [dict(message) for message in PROMPT_TEMPLATE_VIDEO]
    for message in messages:
        message["content"] = message["content"].format(prompt if prompt else " ")
    return messages


def byt5_preprocess_text(prompt: str) -> str:
    """The glyph prompt: ``Text "...". `` for every quoted text (styles are never set at inference)."""
    return "".join(f'Text "{text}". ' for text in extract_glyph_texts(prompt))


def qwen_postprocess_text(
    outputs: BaseEncoderOutput, text_inputs, hidden_state_skip_layer: int = 2
) -> torch.Tensor:
    """The hidden state two layers before the last, with the template's prefix cropped off."""
    if outputs.hidden_states is None:
        raise RuntimeError(
            "the prompt encoder returned no hidden states; MovieRoll reads the layer "
            f"{hidden_state_skip_layer} before the last"
        )
    hidden = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
    return hidden[:, QWEN_CROP_START:]


def byt5_postprocess_text(outputs: BaseEncoderOutput, text_inputs) -> torch.Tensor:
    return outputs.last_hidden_state


@dataclass
class MovieRollPipelineConfig(PipelineConfig):
    """The base chain."""

    task_type: ModelTaskType = ModelTaskType.TI2V
    # the input image is scaled to cover the video and centre-cropped by MovieRoll's own stage
    skip_input_image_preprocess: bool = True
    # guidance is the rolling stage's (per frame, annealed), not the standard denoising loop's
    should_use_guidance: bool = False
    enable_autocast: bool = False

    dit_config: DiTConfig = field(default_factory=MovieRollDitConfig)
    dit_precision: str = "bf16"
    vae_config: VAEConfig = field(default_factory=HunyuanVideo15VAEConfig)
    vae_precision: str = "fp16"
    vae_tiling: bool = False  # MovieRoll decodes in causal temporal chunks
    vae_sp: bool = True  # the row-parallel half of PS-VAE, when there is a group for it

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (MovieRollQwenConfig(), MovieRollByT5Config())
    )
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp16", "fp16")
    )
    preprocess_text_funcs: tuple[Callable[[str], Any] | None, ...] = field(
        default_factory=lambda: (qwen_preprocess_text, byt5_preprocess_text)
    )
    postprocess_text_funcs: tuple[Callable[..., torch.Tensor], ...] = field(
        default_factory=lambda: (qwen_postprocess_text, byt5_postprocess_text)
    )

    flow_shift: float | None = float(BASE["flow_shift"])
    generator_device: str | None = "cuda"
    sampling_mode: str = "base"
    # what the model directory sets differently from the built-in chain (``movieroll_overrides`` of its
    # model_index.json, put here when the pipeline loads: `MovieRollPipeline.initialize_pipeline`)
    movieroll_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def chain(self) -> Chain:
        """The chain the rolling stages run: the built-in one with the model directory's overrides."""
        return Chain.of(self.sampling_mode, self.movieroll_overrides)

    # ------------------------------------------------------------------ the timestep schedule
    def prepare_sigmas(self, sigmas, num_inference_steps):
        """The chain's own ladder, as the sigma list SGLang's scheduler takes.

        The base chain's ladder is ``linspace(1, 0, N + 1)`` without its last point, which the scheduler then
        shift-warps; the distilled chain's is its fixed timestep list divided by the training step count. Both
        produce the sigmas and timesteps of MovieRoll's own scheduler bitwise.
        """
        if sigmas is not None:
            return sigmas
        return torch.linspace(1, 0, num_inference_steps + 1)[:-1].tolist()

    # ------------------------------------------------------------------ text
    def tokenize_prompt(self, prompt: list, tokenizer, tok_kwargs) -> dict:
        """Chat-template tokenization for the prompt encoder, plain tokenization for the glyph encoder."""
        if prompt and isinstance(prompt[0], list):
            return tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **tok_kwargs,
            )
        return tokenizer(prompt, **tok_kwargs)

    # ------------------------------------------------------------------ denoising and decoding
    def post_denoising_loop(self, latents, batch):
        """The rolling stage returns the video's latents whole: no sequence padding or packing to undo."""
        return latents

    def post_decoding(self, frames, server_args):
        """The decoded video leaves the GPU: a long video is gigabytes, and the process that receives it (the
        HTTP server, or a `DiffGenerator` caller asking for the samples) shares the GPU with the worker.
        """
        return frames.cpu()

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def check_pipeline_config(self) -> None:
        # the base check requires --vae-tiling for --vae-sp, which is its own tiled decode; MovieRoll's parallel
        # decode is the streaming PS-VAE (and tiling is refused), so the rest of the check runs without that rule
        vae_sp, self.vae_sp = self.vae_sp, False
        super().check_pipeline_config()
        self.vae_sp = vae_sp


@dataclass
class MovieRollDistilledPipelineConfig(MovieRollPipelineConfig):
    """The distilled chain."""

    flow_shift: float | None = float(DISTILLED["timestep_shift"])
    # the distilled chain draws its noise on the CPU: a generator on another device draws another stream
    generator_device: str | None = "cpu"
    sampling_mode: str = "distilled"
    dmd_denoising_steps: Optional[List[int]] = field(
        default_factory=lambda: list(DISTILLED["denoising_step_list"])
    )

    def prepare_sigmas(self, sigmas, num_inference_steps):
        if sigmas is not None:
            return sigmas
        return [t / NUM_TRAIN_TIMESTEPS for t in self.dmd_denoising_steps]


def register() -> None:
    """Register both chains with SGLang's registry (the convention of `configs/pipeline_configs/{model}.py`)."""
    from sglang.multimodal_gen.registry import register_configs

    from sglang_movieroll.configs.sample.movieroll import (
        MovieRollDistilledSamplingParams,
        MovieRollSamplingParams,
    )

    register_configs(
        sampling_param_cls=MovieRollSamplingParams,
        pipeline_config_cls=MovieRollPipelineConfig,
        hf_model_paths=["MovieRoll/MovieRoll-Base"],
        model_detectors=[lambda name: name == "movierollpipeline"],
    )
    register_configs(
        sampling_param_cls=MovieRollDistilledSamplingParams,
        pipeline_config_cls=MovieRollDistilledPipelineConfig,
        hf_model_paths=["MovieRoll/MovieRoll-Distilled"],
        model_detectors=[lambda name: name == "movierolldistilledpipeline"],
    )
