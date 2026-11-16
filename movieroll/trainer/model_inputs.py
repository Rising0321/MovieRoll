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

"""Turn one collated batch into the arguments the transformer forward takes.

This is where the text actually gets encoded, and three things about how it happens are load-bearing:

* **every caption of the batch, plus the null caption, goes through ONE `encode()` call.** The text encoder is
  FSDP-sharded across the whole world, so each rank must call it the same number of times per step or the
  all-gathers inside it never line up and the job hangs. Batching also fixes the content of that forward, so
  splitting it would change the numbers even when it does not deadlock;
* **padded caption slots stay all-zero ids with an all-zero mask.** The batch arrives with `None` in those
  slots and they are simply never tokenized. Tokenizing `""` instead would not be equivalent: the template
  substitutes a space for an empty string, producing real template tokens with mask 1;
* **the task-form draw happens here, before the encode.** It consumes exactly one value from the global python
  RNG per micro-batch, which is the position the rest of the RNG ledger is written against.

The glyph encoder is not sharded, and it runs unconditionally -- including on an all-zero prompt, which is what
an empty `byt5_text` produces. The mask is what gates the glyph tokens downstream, not the embedding.
"""

import math
from typing import Optional

import torch

from movieroll.utils.multitask_utils import get_multitask_mask

# t2v is the only input path; the draw is kept because it is part of the RNG stream.


def _get_text_tokens(text_encoder, description, text_len):
    """Tokenize one caption. The sequence is `text_len` plus the template's crop_start prefix."""
    text_inputs = text_encoder.text2tokens(description, data_type="video", max_length=text_len)
    return text_inputs["input_ids"].squeeze(0), text_inputs["attention_mask"].squeeze(0)


def get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
    byt5_text_inputs = byt5_tokenizer(
        text_prompt,
        padding="max_length",
        max_length=byt5_max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    byt5_text_ids = byt5_text_inputs.input_ids
    byt5_text_mask = byt5_text_inputs.attention_mask

    return byt5_text_ids, byt5_text_mask


def encode_pixel_values(pixel_values, vae, encode_generator):
    """Encode [-1, 1] pixels to scaled latents: a batch of `pixel_values` here, a clip in `scripts/encode_latents.py`.

    The tile sizes are set from the VAE config here rather than inherited, so encoding is sized by the config
    and not by whatever a preceding decode happened to choose. These two lines state that locally; they are no
    longer repairing someone else's residue, because the one writer that departs from the config-derived sizes
    -- the Base pipeline's non-streaming decode branch, which sizes its tiles by the latent length -- now
    restores what it found. The VAE instance is shared between that decode and this encode within a process.
    """
    if encode_generator is None:
        raise ValueError(
            "encoding pixel_values requires an explicit generator; falling back to the global CUDA RNG "
            "would shift every later draw of the step"
        )
    if pixel_values.max() > 1.0 or pixel_values.min() < -1.0:
        raise ValueError(
            f"pixel_values must be in [-1, 1], got [{pixel_values.min().item()}, {pixel_values.max().item()}]"
        )
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(2)

    vae.tile_sample_min_size = vae.config.sample_size
    vae.tile_latent_min_size = vae.config.sample_size // vae.config.ffactor_spatial
    previous_slicing, previous_spatial = vae.use_slicing, vae.use_spatial_tiling
    vae.enable_slicing()
    vae.enable_tiling()
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            latents = vae.encode(pixel_values).latent_dist.sample(generator=encode_generator)
            if getattr(vae.config, "shift_factor", None):
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
            else:
                latents = latents * vae.config.scaling_factor
    finally:
        vae.use_slicing, vae.use_spatial_tiling = previous_slicing, previous_spatial
    return latents


def prepare_model_inputs(
    batch,
    device,
    model,
    vae,
    text_encoder,
    byt5_kwarg,
    text_len,
    encode_generator: Optional[torch.Generator] = None,
):
    """Build `(latents, model_kwargs, data_type, n_tokens, multitask_mask, mask_type)` for one micro-batch."""
    if "latents" in batch:
        latents = batch["latents"].to(device, non_blocking=True)
    elif "pixel_values" in batch:
        latents = encode_pixel_values(
            batch["pixel_values"].to(device, non_blocking=True), vae, encode_generator
        )
    else:
        raise KeyError("the batch must carry either 'latents' or 'pixel_values'")

    # One global random.random() per micro-batch, and it must happen BEFORE the text encode: the rest of the
    # step's RNG ledger is written against this position. A non-None mask switches the model input to the
    # 65-channel concat layout.
    multitask_mask, mask_type = get_multitask_mask(latents.shape[2])

    data_type = batch["data_type"]
    if isinstance(data_type, (list, tuple)):
        data_type = data_type[0]
    if data_type != "video":
        raise ValueError(f"data_type must be 'video', got {data_type!r}")

    captions = batch["captions"]
    num_captions = batch["num_captions"]
    if isinstance(num_captions, torch.Tensor):
        num_captions = num_captions.tolist()
    elif not isinstance(num_captions, list):
        num_captions = [num_captions] if num_captions is not None else []

    batch_size = len(captions)
    max_captions = max(num_captions)
    event_stamp = batch["event_stamp"]
    if max_captions != event_stamp.shape[1]:
        raise ValueError(
            f"num_captions max {max_captions} does not match event_stamp slots {event_stamp.shape[1]}"
        )

    # The null caption. It is both the per-window dropout target and the shape reference for the slots.
    null_ids_1, null_mask_1 = _get_text_tokens(text_encoder, "", text_len)
    sequence_length = null_ids_1.shape[-1]
    null_ids = null_ids_1.unsqueeze(0).repeat(batch_size, 1).to(device, non_blocking=True)
    null_mask_raw = null_mask_1.unsqueeze(0).repeat(batch_size, 1).to(device, non_blocking=True)

    # Pre-zeroed, so a padded slot is all-zero ids with an all-zero mask and is never tokenized.
    st_ids = torch.zeros(batch_size, max_captions, sequence_length, dtype=torch.long, device=device)
    st_mask = torch.zeros(batch_size, max_captions, sequence_length, dtype=torch.long, device=device)
    for b in range(batch_size):
        for i in range(num_captions[b]):
            caption = captions[b][i]
            if caption is None:
                raise ValueError(f"sample {b} has None at caption slot {i} but claims {num_captions[b]}")
            ids, mask = _get_text_tokens(text_encoder, caption, text_len)
            st_ids[b, i] = ids.to(device, non_blocking=True)
            st_mask[b, i] = mask.to(device, non_blocking=True)

    # One encode() call for every caption plus the null caption: the text encoder is FSDP-sharded over the whole
    # world, so the per-rank call count must match (see the module docstring)
    # Layout: [batch_size*max_captions captions, then batch_size null texts]
    caption_rows = batch_size * max_captions
    flat_ids = torch.cat([st_ids.reshape(caption_rows, sequence_length), null_ids], dim=0)
    flat_mask = torch.cat([st_mask.reshape(caption_rows, sequence_length), null_mask_raw], dim=0)
    all_out = text_encoder.encode(
        {"input_ids": flat_ids, "attention_mask": flat_mask},
        data_type=data_type,
    )
    S_enc = all_out.hidden_state.shape[1]

    # Split back
    caption_hidden_states = all_out.hidden_state[:caption_rows]   # [B*NC, S_enc, D]
    caption_attention_mask = all_out.attention_mask[:caption_rows]    # [B*NC, S_enc]
    null_hidden_states = all_out.hidden_state[caption_rows:]      # [B, S_enc, D]
    null_attention_mask = all_out.attention_mask[caption_rows:]             # [B, S_enc]

    streaming_text_states = caption_hidden_states.reshape(batch_size, max_captions, S_enc, -1)
    streaming_text_mask = caption_attention_mask.reshape(batch_size, max_captions, S_enc)
    text_states = streaming_text_states[:, 0]   # [B, S_enc, D]
    text_mask = streaming_text_mask[:, 0]        # [B, S_enc]

    # =================================== Patchified token count ===================================
    n_dim = len(latents.shape) - 2
    latents_size = list(latents.shape[2:])
    assert all(s % model.patch_size[i] == 0 for i, s in enumerate(latents_size)), \
        f"Latent size(last {n_dim} dimensions) should be divisible by patch size({model.patch_size}), " \
        f"but got {latents_size}."
    latent_size = [s // model.patch_size[i] for i, s in enumerate(latents_size)]
    n_tokens = math.prod(latent_size)

    model_kwargs = dict(
        text_states=text_states,                     # [B, S_enc, D]
        encoder_attention_mask=text_mask,            # [B, S_enc]
        return_dict=False,
    )

    # ============================== Streaming extra kwargs =======================================
    model_kwargs["extra_kwargs"] = {}
    model_kwargs["extra_kwargs"]["streaming_text_states"] = streaming_text_states  # [B, NC, S, D]
    model_kwargs["extra_kwargs"]["streaming_text_mask"] = streaming_text_mask      # [B, NC, S]
    model_kwargs["extra_kwargs"]["null_text_states"] = null_hidden_states                 # [B, S, D]
    model_kwargs["extra_kwargs"]["null_text_mask_enc"] = null_attention_mask                 # [B, S]
    model_kwargs["extra_kwargs"]["event_stamp"] = event_stamp.to(device, non_blocking=True)

    cap_types = batch.get("cap_type", [])
    if isinstance(cap_types, str):
        cap_types = [cap_types]
    model_kwargs["extra_kwargs"]["cap_types"] = cap_types
    model_kwargs["extra_kwargs"]["num_captions"] = num_captions

    # ========================================== byT5 ===========================================
    byt5_tokenizer = byt5_kwarg["byt5_tokenizer"]
    byt5_model = byt5_kwarg["byt5_model"]
    byt5_max_length = byt5_kwarg["byt5_max_length"]

    byt5_rows_ids, byt5_rows_mask = [], []
    for glyph_text in batch["byt5_text"]:
        if glyph_text:
            ids, mask = get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, glyph_text)
            byt5_rows_ids.append(ids.squeeze(0))
            byt5_rows_mask.append(mask.squeeze(0))
        else:
            # No quoted text: all-zero ids with an all-zero mask. The encoder still runs on them -- the mask
            # is what keeps the glyph tokens out of the attention downstream.
            byt5_rows_ids.append(torch.zeros(byt5_max_length, dtype=torch.int64))
            byt5_rows_mask.append(torch.zeros(byt5_max_length, dtype=torch.int64))
    byt5_text_ids = torch.stack(byt5_rows_ids).to(device, non_blocking=True)
    byt5_text_mask = torch.stack(byt5_rows_mask).to(device, non_blocking=True)

    byt5_prompt_embeds = byt5_model(byt5_text_ids, attention_mask=byt5_text_mask.float())
    byt5 = byt5_prompt_embeds[0]

    model_kwargs["extra_kwargs"]["byt5_text_states"] = byt5
    model_kwargs["extra_kwargs"]["byt5_text_mask"] = byt5_text_mask

    return latents, model_kwargs, data_type, n_tokens, multitask_mask, mask_type
