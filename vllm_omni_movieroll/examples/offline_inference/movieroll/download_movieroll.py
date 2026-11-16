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

"""Assemble the model directory vLLM-Omni loads MovieRoll from (vLLM-Omni's "download/setup script" for a native model).

A released MovieRoll model folder (``ckpts/movieroll/base``, ``distilled/t2v`` or ``distilled/i2v``: ``config.json`` +
``model.safetensors``, what ``movieroll/generate.py --model`` takes) is not a diffusers repository, and neither are the shared
components next to it (the VAE in MovieRoll's own key names, Qwen2.5-VL-7B-Instruct, google/byt5-small and the
Glyph-SDXL-v2 byT5 weights with its colour / font tables). This script lays out the HunyuanVideo-1.5 diffusers layout
vLLM-Omni's components read, and writes `model_index.json`:

    model_index.json        _class_name MovieRollPipeline (base) | MovieRollDistilledPipeline (distilled), sampling_mode,
                            movieroll_overrides (what the folder's config.json sets differently from the built-in chain)
    transformer/            config.json + the DiT checkpoint (linked; renamed at load time, 1353 / 1353 keys)
    text_encoder/, tokenizer/   Qwen2.5-VL-7B-Instruct (linked; read as Qwen2_5_VLTextModel / Qwen2Tokenizer)
    text_encoder_2/         converted once: byt5-small encoder with the Glyph-SDXL-v2 weights (T5 config, vocab with the
                            colour / font tokens), safetensors
    tokenizer_2/            converted once: the byT5 tokenizer with the colour / font special tokens
    vae/                    converted once: diffusers AutoencoderKLHunyuanVideo15 config + renamed safetensors (218 / 218)
    scheduler/              FlowMatchEulerDiscreteScheduler config (shift 5)

Conversions only rename / merge tensors (no dtype change) and are checked here: every DiT key must map, the VAE and the byT5
encoder must load strictly. The directory is built next to ``--out`` and moved there only when complete. Run it once per
model folder, inside the vLLM-Omni virtual environment (it loads the encoder and the VAE with its transformers and
diffusers); the component paths default to the main README's ``ckpts/`` layout::

    python vllm_omni_movieroll/examples/offline_inference/movieroll/download_movieroll.py \\
        --model ckpts/movieroll/distilled/i2v --out ./movieroll-distilled-i2v

``--sampling-mode`` with ``--dit <checkpoint.safetensors>`` builds from a bare checkpoint instead, with the chain's
built-in settings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from typing import Optional

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from vllm_omni_movieroll.profiles import (  # noqa: E402
    MODEL_FOLDER_WEIGHTS,
    PIPELINE_CLASS,
    SAMPLING_MODES,
    load_settings,
    read_model_folder,
    transformer_config,
)
from vllm_omni_movieroll.weights import map_name, vae_config, vae_key  # noqa: E402


def _safetensors_keys(path: str) -> list[str]:
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return list(header)


def _place(source: str, target: str, copy: bool) -> None:
    source = os.path.abspath(source)
    if not os.path.exists(source):
        raise FileNotFoundError(source)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if copy:
        (shutil.copytree if os.path.isdir(source) else shutil.copy2)(source, target)
    else:
        os.symlink(source, target)


def _write_json(path: str, value: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2)


def check_dit(path: str) -> int:
    """Every key of the DiT checkpoint maps onto `MovieRollTransformer3DModel` (raises otherwise); returns the key count."""
    keys = _safetensors_keys(path)
    for key in keys:
        map_name(key)
    return len(keys)


def build_text_encoder_2(byt5: str, glyph_ckpt: str, font_json: str, color_json: str, out: str) -> int:
    """byt5-small + the colour / font tokens + the Glyph-SDXL-v2 weights, saved as a T5 encoder and its tokenizer."""
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(byt5)
    with open(font_json) as handle:
        fonts = json.load(handle)
    with open(color_json) as handle:
        colors = json.load(handle)
    # MovieRoll's order: colours, then the multilingual fonts
    tokens = [f"<color-{i}>" for i in range(len(colors))] + [f"<{code[:2]}-font-{fonts[code]}>" for code in fonts]
    tokenizer.add_tokens(tokens, special_tokens=True)
    model = T5EncoderModel.from_pretrained(byt5, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    state = torch.load(glyph_ckpt, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(state, strict=True)
    model.save_pretrained(os.path.join(out, "text_encoder_2"), safe_serialization=True)
    tokenizer.save_pretrained(os.path.join(out, "tokenizer_2"))
    return len(tokenizer)


def build_vae(vae_dir: str, out: str) -> int:
    """MovieRoll's VAE in diffusers' `AutoencoderKLHunyuanVideo15` layout; loads strictly before it is written."""
    import torch
    from diffusers import AutoencoderKLHunyuanVideo15
    from safetensors.torch import load_file, save_file

    with open(os.path.join(vae_dir, "config.json")) as handle:
        config = vae_config(json.load(handle))
    state = {vae_key(k): v for k, v in load_file(os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")).items()}
    with torch.device("meta"):
        model = AutoencoderKLHunyuanVideo15(**{k: v for k, v in config.items() if not k.startswith("_")})
    missing = set(model.state_dict()) - set(state)
    unexpected = set(state) - set(model.state_dict())
    if missing or unexpected:
        raise RuntimeError(f"VAE conversion: missing {sorted(missing)[:5]}, unexpected {sorted(unexpected)[:5]}")
    for name, tensor in model.state_dict().items():
        if tuple(tensor.shape) != tuple(state[name].shape):
            raise RuntimeError(f"VAE conversion: {name} has shape {tuple(state[name].shape)}, expected {tuple(tensor.shape)}")
    os.makedirs(os.path.join(out, "vae"))
    _write_json(os.path.join(out, "vae", "config.json"), config)
    save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(out, "vae", "diffusion_pytorch_model.safetensors"))
    return len(state)


def build_model_dir(out: str, *, sampling_mode: str, dit: str, vae: str, text_encoder: str, byt5: str, glyph_ckpt: str,
                    font_json: str, color_json: str, overrides: Optional[dict] = None, copy: bool = False) -> dict:
    """Create ``out`` (it must not exist); returns a summary of what was linked / converted.

    Everything is written into a sibling directory first and renamed to ``out`` at the end, so a failed conversion leaves
    no half-built directory in the way of the next attempt.
    """
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"sampling_mode must be one of {SAMPLING_MODES}")
    if not dit.endswith(".safetensors"):
        raise ValueError(f"the DiT checkpoint must be a .safetensors file, got {dit}")
    if os.path.exists(out):
        raise FileExistsError(f"{out} already exists")
    for source in (dit, vae, text_encoder, byt5, glyph_ckpt, font_json, color_json):
        if not os.path.exists(source):
            raise FileNotFoundError(source)
    profile = load_settings({"sampling_mode": sampling_mode, "movieroll_overrides": overrides or {}}).profile
    summary = {"dit_keys": check_dit(dit), "movieroll_overrides": overrides or {}}
    staging = os.path.join(os.path.dirname(os.path.abspath(out)), f".{os.path.basename(os.path.abspath(out))}.partial")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    try:
        _write_json(os.path.join(staging, "model_index.json"), {
            "_class_name": PIPELINE_CLASS[sampling_mode],
            "_diffusers_version": "0.40.0",
            "sampling_mode": sampling_mode,
            "movieroll_overrides": overrides or {},
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Qwen2_5_VLTextModel"],
            "text_encoder_2": ["transformers", "T5EncoderModel"],
            "tokenizer": ["transformers", "Qwen2Tokenizer"],
            "tokenizer_2": ["transformers", "ByT5Tokenizer"],
            "transformer": ["vllm_omni_movieroll", "MovieRollTransformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLHunyuanVideo15"],
        })
        _write_json(os.path.join(staging, "transformer", "config.json"),
                    transformer_config(profile["full_history_block_num"]))
        _place(dit, os.path.join(staging, "transformer", "diffusion_pytorch_model.safetensors"), copy)
        _place(text_encoder, os.path.join(staging, "text_encoder"), copy)
        # the tokenizer files sit in the same Qwen2.5-VL folder: one copy of it serves both entries
        os.symlink("text_encoder", os.path.join(staging, "tokenizer"))
        summary["byt5_vocab"] = build_text_encoder_2(byt5, glyph_ckpt, font_json, color_json, staging)
        summary["vae_keys"] = build_vae(vae, staging)
        shift = float(profile["timestep_shift"] if sampling_mode == "distilled" else profile["flow_shift"])
        _write_json(os.path.join(staging, "scheduler", "scheduler_config.json"), {
            "_class_name": "FlowMatchEulerDiscreteScheduler", "_diffusers_version": "0.40.0",
            "num_train_timesteps": 1000, "shift": shift,
        })
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    os.rename(staging, out)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("::")[0].strip(), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", required=True, help="directory to create")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="a MovieRoll model folder (config.json + model.safetensors), e.g. "
                                        "ckpts/movieroll/distilled/i2v: its config.json gives the chain and its settings")
    source.add_argument("--dit", help="a bare MovieRoll DiT .safetensors instead of --model (needs --sampling-mode; the "
                                      "chain's built-in settings)")
    parser.add_argument("--sampling-mode", choices=SAMPLING_MODES, help="the chain of --dit (taken from --model otherwise)")
    parser.add_argument("--vae", default="ckpts/vae", help="MovieRoll's VAE folder (config.json + "
                                                           "diffusion_pytorch_model.safetensors)")
    parser.add_argument("--text-encoder", default="ckpts/text_encoder/llm", help="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--byt5", default="ckpts/text_encoder/byt5-small", help="google/byt5-small")
    parser.add_argument("--glyph-ckpt", default="ckpts/text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt",
                        help="Glyph-SDXL-v2's byT5 weights")
    parser.add_argument("--font-json", default="ckpts/text_encoder/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json",
                        help="Glyph-SDXL-v2's multilingual_10-lang_idx.json")
    parser.add_argument("--color-json", default="ckpts/text_encoder/Glyph-SDXL-v2/assets/color_idx.json",
                        help="Glyph-SDXL-v2's color_idx.json")
    parser.add_argument("--copy", action="store_true", help="copy the DiT and Qwen2.5-VL instead of linking them")
    args = parser.parse_args(argv)
    if args.model is not None:
        if args.sampling_mode is not None:
            parser.error("--sampling-mode comes from the config.json of --model")
        sampling_mode, overrides = read_model_folder(args.model)
        dit = os.path.join(args.model, MODEL_FOLDER_WEIGHTS)
    else:
        if args.sampling_mode is None:
            parser.error("--dit needs --sampling-mode")
        sampling_mode, overrides, dit = args.sampling_mode, {}, args.dit
    summary = build_model_dir(
        args.out, sampling_mode=sampling_mode, dit=dit, vae=args.vae, text_encoder=args.text_encoder, byt5=args.byt5,
        glyph_ckpt=args.glyph_ckpt, font_json=args.font_json, color_json=args.color_json, overrides=overrides,
        copy=args.copy,
    )
    print(f"{sampling_mode} model directory written to {args.out}: {summary}")


if __name__ == "__main__":
    main()
