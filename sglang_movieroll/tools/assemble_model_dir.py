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

"""Build the model directory sglang loads MovieRoll from, out of the released files.

sglang reads a diffusers-style directory: ``model_index.json`` names the pipeline class and every component
sits in its own folder, loaded by sglang's own component loaders. A released MovieRoll model folder
(``ckpts/movieroll/base``, ``distilled/t2v`` or ``distilled/i2v``: ``config.json`` + ``model.safetensors``, what
``movieroll/generate.py --model`` takes) and the shared components next to it are not stored that way, so this
tool lays the directory out once per model folder. The component paths default to the main README's ``ckpts/``
layout::

    python -m sglang_movieroll.tools.assemble_model_dir --model ckpts/movieroll/distilled/i2v \\
        --out ./movieroll-distilled-i2v

| entry | content |
|---|---|
| ``model_index.json`` | the pipeline class of the chain, its seven modules, and ``movieroll_overrides``: what the folder's ``config.json`` sets differently from the built-in chain (``{"context_noise": 0.0}`` for the image-to-video folder) |
| ``transformer/`` | ``config.json`` (the architecture) and a link to the folder's DiT checkpoint, whose own tensor names are kept (the loader renames them through ``param_names_mapping``) |
| ``text_encoder/``, ``tokenizer/`` | a link to the Qwen2.5-VL folder, and ``tokenizer`` a link to ``text_encoder`` |
| ``text_encoder_2/``, ``tokenizer_2/`` | **written once**: the byT5 glyph encoder as a standard `T5EncoderModel` (byT5-small plus the Glyph-SDXL-v2 weights and its colour / font tokens) and its tokenizer |
| ``vae/`` | **written once**: the released VAE under diffusers' `AutoencoderKLHunyuanVideo15` names (a pure rename, checked key by key) |
| ``scheduler/`` | the flow-match scheduler config of the chain |

Writing the two components is what lets sglang's own loaders read them; nothing else is converted, and the DiT
(the large file) is linked, not copied. ``--copy`` copies every link instead. The directory is built next to
``--out`` and moved there only when complete. ``--sampling-mode`` with ``--dit-weight`` builds from a bare
checkpoint instead, with the chain's built-in settings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Dict, Optional

PIPELINE_CLASS = {
    "base": "MovieRollPipeline",
    "distilled": "MovieRollDistilledPipeline",
}
DIFFUSERS_VERSION = "0.37.0"

# MovieRoll's VAE checkpoint -> diffusers' AutoencoderKLHunyuanVideo15 names (the shapes are identical, and every
# key and value was checked against MovieRoll's own VAE).
VAE_RULES = [
    (r"^(encoder|decoder)\.mid\.block_1\.", r"\1.mid_block.resnets.0."),
    (r"^(encoder|decoder)\.mid\.block_2\.", r"\1.mid_block.resnets.1."),
    (
        r"^(encoder|decoder)\.mid\.attn_1\.(q|k|v)\.",
        r"\1.mid_block.attentions.0.to_\2.",
    ),
    (r"^(encoder|decoder)\.mid\.attn_1\.", r"\1.mid_block.attentions.0."),
    (r"^encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2."),
    (r"^encoder\.down\.(\d+)\.downsample\.", r"encoder.down_blocks.\1.downsamplers.0."),
    (r"^decoder\.up\.(\d+)\.block\.(\d+)\.", r"decoder.up_blocks.\1.resnets.\2."),
    (r"^decoder\.up\.(\d+)\.upsample\.", r"decoder.up_blocks.\1.upsamplers.0."),
]


def rename_vae_key(key: str) -> str:
    for pattern, replacement in VAE_RULES:
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    return key


def transformer_config(sampling_mode: str) -> Dict:
    """``transformer/config.json``: the architecture of the released checkpoints and the HieraKV split."""
    from sglang_movieroll.rolling.profiles import get_profile

    profile = get_profile(sampling_mode)
    return {
        "_class_name": "MovieRollTransformer3DModel",
        "in_channels": 32,
        "out_channels": 32,
        "num_attention_heads": 16,
        "attention_head_dim": 128,
        "num_layers": 54,
        "num_refiner_layers": 2,
        "mlp_ratio": 4.0,
        "patch_size": [1, 1, 1],
        "text_embed_dim": 3584,
        "text_embed_2_dim": 1472,
        "byt5_hidden_dim": 2048,
        "rope_axes_dim": [16, 56, 56],
        "rope_theta": 256.0,
        "max_num_latent_frames": profile["max_num_latent_frames"],
        "full_history_block_num": profile["full_history_block_num"],
        "sink_chunks": 1,
    }


def scheduler_config(sampling_mode: str) -> Dict:
    from sglang_movieroll.rolling.profiles import Chain

    return {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "num_train_timesteps": 1000,
        "shift": Chain.of(sampling_mode).shift,
        "use_dynamic_shifting": False,
    }


def _place(source: str, target: str, copy: bool) -> None:
    source = os.path.abspath(source)
    if not os.path.exists(source):
        raise FileNotFoundError(source)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.lexists(target):
        raise FileExistsError(f"{target} already exists")
    if not copy:
        os.symlink(source, target)
    elif os.path.isdir(source):
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def write_vae(out: str, vae_path: str) -> None:
    """The released VAE under diffusers' names, with its config, and a check that nothing was lost."""
    from safetensors.torch import load_file, save_file

    source_config = json.load(open(os.path.join(vae_path, "config.json")))
    weights = load_file(os.path.join(vae_path, "diffusion_pytorch_model.safetensors"))
    renamed = {rename_vae_key(name): tensor for name, tensor in weights.items()}
    if len(renamed) != len(weights):
        raise RuntimeError("the VAE rename collapsed two keys into one")
    os.makedirs(os.path.join(out, "vae"))
    config = {
        "_class_name": "AutoencoderKLHunyuanVideo15",
        "_diffusers_version": DIFFUSERS_VERSION,
        "in_channels": source_config["in_channels"],
        "out_channels": source_config["out_channels"],
        "latent_channels": source_config["latent_channels"],
        "block_out_channels": source_config["block_out_channels"],
        "layers_per_block": source_config["layers_per_block"],
        "spatial_compression_ratio": source_config["ffactor_spatial"],
        "temporal_compression_ratio": source_config["ffactor_temporal"],
        "downsample_match_channel": source_config["downsample_match_channel"],
        "upsample_match_channel": source_config["upsample_match_channel"],
        "scaling_factor": source_config["scaling_factor"],
        "stream_chunk_frames": 4,
    }
    with open(os.path.join(out, "vae", "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    save_file(renamed, os.path.join(out, "vae", "diffusion_pytorch_model.safetensors"))


def write_glyph_encoder(
    out: str, google_path: str, ckpt_path: str, font_path: str, color_path: str
) -> None:
    """The byT5 glyph encoder as a standard `T5EncoderModel` component, with its tokenizer.

    byT5-small plus the colour and font tokens of Glyph-SDXL-v2 (which is what grows the vocabulary), with the
    Glyph weights loaded into it. The written folder is what sglang's text-encoder loader reads.
    """
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(google_path)
    font_dict = json.load(open(font_path))
    color_dict = json.load(open(color_path))
    extra = [f"<color-{i}>" for i in range(len(color_dict))]
    extra += [f"<{code[:2]}-font-{font_dict[code]}>" for code in font_dict]
    tokenizer.add_tokens(extra, special_tokens=True)

    config = T5EncoderModel.config_class.from_pretrained(google_path)
    config.vocab_size = len(tokenizer)
    model = T5EncoderModel(config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    remapped = {"shared.weight": state["embed_tokens.weight"]}
    remapped.update({f"encoder.{name}": tensor for name, tensor in state.items()})
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"the glyph encoder did not load cleanly: missing {missing}, unexpected {unexpected}"
        )
    model.save_pretrained(os.path.join(out, "text_encoder_2"))
    tokenizer.save_pretrained(os.path.join(out, "tokenizer_2"))


def build_model_dir(
    out: str,
    *,
    sampling_mode: str,
    dit_weight: str,
    vae_path: str,
    text_encoder_path: str,
    byt5_google_path: str,
    byt5_ckpt_path: str,
    font_path: str,
    color_path: str,
    overrides: Optional[Dict] = None,
    copy: bool = False,
) -> str:
    """Create ``out`` (it must not exist) and return its path.

    Everything is written into a sibling directory first and renamed to ``out`` at the end, so a failed
    conversion leaves no half-built directory in the way of the next attempt.
    """
    from sglang_movieroll.rolling.profiles import Chain

    if sampling_mode not in PIPELINE_CLASS:
        raise ValueError(
            f"sampling_mode must be one of {tuple(PIPELINE_CLASS)}, got {sampling_mode!r}"
        )
    if not dit_weight.endswith(".safetensors"):
        raise ValueError(
            f"the DiT checkpoint must be a .safetensors file, got {dit_weight}"
        )
    if os.path.exists(out):
        raise FileExistsError(f"{out} already exists")
    for source in (
        dit_weight,
        vae_path,
        text_encoder_path,
        byt5_google_path,
        byt5_ckpt_path,
        font_path,
        color_path,
    ):
        if not os.path.exists(source):
            raise FileNotFoundError(
                source
            )  # every source is checked before anything is written
    overrides = Chain.of(sampling_mode, overrides).overrides()  # checks them

    staging = os.path.join(
        os.path.dirname(os.path.abspath(out)),
        f".{os.path.basename(os.path.abspath(out))}.partial",
    )
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    try:
        _write_model_dir(
            staging,
            sampling_mode=sampling_mode,
            overrides=overrides,
            dit_weight=dit_weight,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
            byt5_google_path=byt5_google_path,
            byt5_ckpt_path=byt5_ckpt_path,
            font_path=font_path,
            color_path=color_path,
            copy=copy,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    os.rename(staging, out)
    return out


def _write_model_dir(
    out: str,
    *,
    sampling_mode: str,
    overrides: Dict,
    dit_weight: str,
    vae_path: str,
    text_encoder_path: str,
    byt5_google_path: str,
    byt5_ckpt_path: str,
    font_path: str,
    color_path: str,
    copy: bool,
) -> None:
    index = {
        "_class_name": PIPELINE_CLASS[sampling_mode],
        "_diffusers_version": DIFFUSERS_VERSION,
        "movieroll_overrides": overrides,
        "transformer": ["diffusers", "MovieRollTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLHunyuanVideo15"],
        "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
        "tokenizer": ["transformers", "AutoTokenizer"],
        "text_encoder_2": ["transformers", "T5EncoderModel"],
        "tokenizer_2": ["transformers", "AutoTokenizer"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    with open(os.path.join(out, "model_index.json"), "w") as handle:
        json.dump(index, handle, indent=2)

    os.makedirs(os.path.join(out, "transformer"))
    with open(os.path.join(out, "transformer", "config.json"), "w") as handle:
        json.dump(transformer_config(sampling_mode), handle, indent=2)
    _place(
        dit_weight,
        os.path.join(out, "transformer", "diffusion_pytorch_model.safetensors"),
        copy,
    )

    _place(text_encoder_path, os.path.join(out, "text_encoder"), copy)
    # the tokenizer files sit in the same Qwen2.5-VL folder: one link (or copy) of it serves both entries
    os.symlink("text_encoder", os.path.join(out, "tokenizer"))

    os.makedirs(os.path.join(out, "scheduler"))
    with open(os.path.join(out, "scheduler", "scheduler_config.json"), "w") as handle:
        json.dump(scheduler_config(sampling_mode), handle, indent=2)

    write_vae(out, vae_path)
    write_glyph_encoder(out, byt5_google_path, byt5_ckpt_path, font_path, color_path)


def main(argv: Optional[list[str]] = None) -> None:
    from sglang_movieroll.rolling.profiles import (
        MODEL_FOLDER_WEIGHTS,
        read_model_folder,
    )

    parser = argparse.ArgumentParser(
        description="Build an sglang model directory for MovieRoll"
    )
    parser.add_argument("--out", required=True, help="directory to create")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        help="a MovieRoll model folder (config.json + model.safetensors), e.g. ckpts/movieroll/distilled/i2v; "
        "its config.json gives the chain and its settings",
    )
    source.add_argument(
        "--dit-weight",
        help="a bare MovieRoll DiT .safetensors instead of --model (needs --sampling-mode; the chain's built-in "
        "settings)",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=tuple(PIPELINE_CLASS),
        help="the chain of --dit-weight (taken from --model otherwise)",
    )
    parser.add_argument("--vae-path", default="ckpts/vae")
    parser.add_argument("--text-encoder-path", default="ckpts/text_encoder/llm")
    parser.add_argument("--byt5-google-path", default="ckpts/text_encoder/byt5-small")
    parser.add_argument(
        "--byt5-ckpt-path",
        default="ckpts/text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt",
    )
    parser.add_argument(
        "--font-path",
        default="ckpts/text_encoder/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json",
    )
    parser.add_argument(
        "--color-path",
        default="ckpts/text_encoder/Glyph-SDXL-v2/assets/color_idx.json",
    )
    parser.add_argument(
        "--copy", action="store_true", help="copy the files instead of linking them"
    )
    args = parser.parse_args(argv)
    if args.model is not None:
        if args.sampling_mode is not None:
            parser.error("--sampling-mode comes from the config.json of --model")
        sampling_mode, overrides = read_model_folder(args.model)
        dit_weight = os.path.join(args.model, MODEL_FOLDER_WEIGHTS)
    else:
        if args.sampling_mode is None:
            parser.error("--dit-weight needs --sampling-mode")
        sampling_mode, overrides, dit_weight = args.sampling_mode, {}, args.dit_weight
    path = build_model_dir(
        args.out,
        sampling_mode=sampling_mode,
        dit_weight=dit_weight,
        vae_path=args.vae_path,
        text_encoder_path=args.text_encoder_path,
        byt5_google_path=args.byt5_google_path,
        byt5_ckpt_path=args.byt5_ckpt_path,
        font_path=args.font_path,
        color_path=args.color_path,
        overrides=overrides,
        copy=args.copy,
    )
    print(
        f"{sampling_mode} model directory written to {path}, settings {overrides or 'as released'}"
    )


if __name__ == "__main__":
    main()
