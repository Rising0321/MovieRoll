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

"""Builders for the transformer and the text encoder.

Weight loading is strict: the checkpoint must exist, must be a .safetensors file, and must match the model
key for key. The single exception is `vision_in.*`, the I2V vision tower, which only the ACS score models
build (`dmd.score_mode: acs`): a model that does not build it has those keys dropped, and the drop is
logged. Anything else missing or unexpected raises.
"""

import os

from safetensors import safe_open

from movieroll.models.text_encoders import TextEncoder
from movieroll.models.transformers.model_configs import load_model

# Checkpoint keys MovieRoll builds only on request: the I2V vision tower, which the ACS score models use to
# read the anchor frame's SigLIP tokens. Dropped for every model that does not build them.
OPTIONAL_CHECKPOINT_KEY_PREFIXES = ("vision_in.",)


def _log(logger, message):
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def load_hunyuan_state_dict(model, dit_model_name_or_path, logger=None):
    """Strictly load a .safetensors checkpoint into `model`.

    Drops (and logs) only the OPTIONAL_CHECKPOINT_KEY_PREFIXES keys this `model` does not declare; every
    other missing or unexpected key raises.
    Never compare the key sets before the call: the streaming blocks merge separate attn_q / attn_k / attn_v
    into attn_qkv inside their `_load_from_state_dict` hook, so a separate-qkv checkpoint looks mismatched.
    strict=True checks key sets and shapes, never dtype, so the dtype the model was built with wins.
    `model` must not be FSDP-wrapped yet; wrapping first compares sharded shapes.
    """
    if not os.path.isfile(dit_model_name_or_path):
        raise FileNotFoundError(f"transformer checkpoint not found: {dit_model_name_or_path}")
    if not str(dit_model_name_or_path).endswith(".safetensors"):
        raise ValueError(
            f"transformer checkpoint must be a .safetensors file, got: {dit_model_name_or_path}"
        )

    state_dict = {}
    with safe_open(dit_model_name_or_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # Asking whether this model declares the optional keys is not the forbidden key-set pre-check: it
    # decides nothing about the merged qkv names, and strict=True still judges every other key.
    model_keys = set(model.state_dict())
    dropped = [
        k
        for k in state_dict
        if k.startswith(OPTIONAL_CHECKPOINT_KEY_PREFIXES) and k not in model_keys
    ]
    for key in dropped:
        state_dict.pop(key)
    if dropped:
        _log(logger, f"[load] dropped {len(dropped)} checkpoint keys unused by this model: {sorted(dropped)}")

    model.load_state_dict(state_dict, strict=True)
    return model


def load_transformer(
    model_type,
    dit_model_name_or_path,
    factor_kwargs=None,
    logger=None,
):
    transformer = load_model(model_type, factor_kwargs)
    transformer = load_hunyuan_state_dict(transformer, dit_model_name_or_path, logger=logger)
    return transformer


def load_text_encoder(
    text_encoder_path,
    max_length,
    prompt_template,
    prompt_template_video,
    device,
    text_encoder_precision="fp16",
    logger=None,
):
    """Build the Qwen2.5-VL text encoder with the values MovieRoll was trained with.

    TextEncoder exposes those values as parameters, so pinning them here stops an entry point from silently
    changing the conditioning (by turning `use_attention_mask` off, in particular). Both prompt templates are
    required: they carry the `crop_start` that trims the chat prefix off the hidden states.
    """
    text_encoder = TextEncoder(
        text_encoder_type="llm",
        text_encoder_path=text_encoder_path,
        max_length=max_length,
        text_encoder_precision=text_encoder_precision,
        tokenizer_path=text_encoder_path,
        prompt_template=prompt_template,
        prompt_template_video=prompt_template_video,
        hidden_state_skip_layer=2,
        # the values MovieRoll was trained with; do not expose them
        use_attention_mask=True,
        apply_final_norm=False,
        reproduce=False,
        logger=logger,
        device=device,
    )
    return text_encoder


def get_no_split_modules(transformer):
    from movieroll.models.transformers.score_transformer import MovieRollScoreTransformer
    from movieroll.models.transformers.movieroll_transformer import MovieRollDiffusionTransformer
    if isinstance(transformer, MovieRollScoreTransformer):
        from movieroll.models.transformers.score_transformer import MMDoubleStreamBlock
        return (MMDoubleStreamBlock,)
    if isinstance(transformer, MovieRollDiffusionTransformer):
        from movieroll.models.transformers.movieroll_transformer import MMDoubleStreamBlock
        return (MMDoubleStreamBlock,)
    else:
        raise ValueError(f"Unsupported transformer type: {type(transformer)}")
