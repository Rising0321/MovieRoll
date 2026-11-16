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

import json
import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from movieroll.commons.constants import BYT5_CKPT, BYT5_COLOR, BYT5_FONT, BYT5_GOOGLE


def create_byt5(
    byT5_google_path=BYT5_GOOGLE,
    byT5_ckpt_path=BYT5_CKPT,
    color_ann_path=BYT5_COLOR,
    font_ann_path=BYT5_FONT,
    byt5_max_length=256,
):
    byt5_config = dict(
        byt5_name=byT5_google_path,
        special_token=True,
        color_special_token=True,
        font_special_token=True,
        color_ann_path=color_ann_path,
        font_ann_path=font_ann_path,
        multilingual=True,
    )

    # device must be passed: load_byt5_and_byt5_tokenizer calls torch.device(device) unconditionally, so its
    # None default raises. "cpu" keeps the model where the caller expects it; load_glyph_byT5_v2 moves it.
    byt5_model, byt5_tokenizer = load_byt5_and_byt5_tokenizer(**byt5_config, device="cpu")

    if not os.path.exists(byT5_ckpt_path):
        raise FileNotFoundError(f"byT5 checkpoint not found: {byT5_ckpt_path}")
    byt5_state_dict = torch.load(byT5_ckpt_path, map_location="cpu", weights_only=True)
    byt5_model.load_state_dict(byt5_state_dict)

    return byt5_tokenizer, byt5_model, byt5_max_length


def load_glyph_byT5_v2(
    device,
    byT5_google_path=BYT5_GOOGLE,
    byT5_ckpt_path=BYT5_CKPT,
    color_ann_path=BYT5_COLOR,
    font_ann_path=BYT5_FONT,
    byt5_max_length=256,
):
    """Build the byT5 glyph encoder (frozen) and return it as the kwargs the pipelines expect."""
    byt5_tokenizer, byt5_model, byt5_max_length = create_byt5(
        byT5_google_path=byT5_google_path,
        byT5_ckpt_path=byT5_ckpt_path,
        color_ann_path=color_ann_path,
        font_ann_path=font_ann_path,
        byt5_max_length=byt5_max_length,
    )
    byt5_model = byt5_model.to(device=device)
    byt5_model.requires_grad_(False)

    byt5_kwarg = {"byt5_tokenizer": byt5_tokenizer,
                  "byt5_model": byt5_model,
                  "byt5_max_length": byt5_max_length}

    return byt5_kwarg


def add_special_token(
    tokenizer,
    text_encoder,
    add_color,
    add_font,
    color_ann_path,
    font_ann_path,
    multilingual=False,
):
    """Add the color/font special tokens to `tokenizer` and resize the encoder's embeddings to match."""
    with open(font_ann_path, 'r') as f:
        idx_font_dict = json.load(f)
    with open(color_ann_path, 'r') as f:
        idx_color_dict = json.load(f)

    if multilingual:
        font_token = [f'<{font_code[:2]}-font-{idx_font_dict[font_code]}>' for font_code in idx_font_dict]
    else:
        font_token = [f'<font-{i}>' for i in range(len(idx_font_dict))]
    color_token = [f'<color-{i}>' for i in range(len(idx_color_dict))]
    additional_special_tokens = []
    if add_color:
        additional_special_tokens += color_token
    if add_font:
        additional_special_tokens += font_token

    tokenizer.add_tokens(additional_special_tokens, special_tokens=True)
    # Set mean_resizing=False to avoid PyTorch LAPACK dependency
    text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def load_byt5_and_byt5_tokenizer(
    byt5_name='google/byt5-small',
    special_token=False,
    color_special_token=False,
    font_special_token=False,
    color_ann_path='assets/color_idx.json',
    font_ann_path='assets/font_idx_512.json',
    huggingface_cache_dir=None,
    multilingual=False,
    device=None,
):
    """Load the ByT5 encoder and tokenizer, adding the color/font special tokens when asked.

    Returns (byt5_text_encoder, byt5_tokenizer).
    """
    byt5_tokenizer = AutoTokenizer.from_pretrained(
        byt5_name,
        cache_dir=huggingface_cache_dir,
    )
    byt5_text_encoder = T5ForConditionalGeneration.from_pretrained(
        byt5_name,
        cache_dir=huggingface_cache_dir,
    ).get_encoder()

    if "cuda" not in str(device):
        device = torch.device(device)
    else:
        device = torch.device(device)
    byt5_text_encoder = byt5_text_encoder.to(device)

    if special_token:
        add_special_token(
            byt5_tokenizer,
            byt5_text_encoder,
            add_color=color_special_token,
            add_font=font_special_token,
            color_ann_path=color_ann_path,
            font_ann_path=font_ann_path,
            multilingual=multilingual,
        )
    return byt5_text_encoder, byt5_tokenizer


