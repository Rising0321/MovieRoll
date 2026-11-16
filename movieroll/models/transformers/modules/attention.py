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

"""Attention dispatch shared by the transformer blocks.

``parallel_attention`` is the single entry of the double-stream blocks of both models: it runs the Ulysses
all-to-all, concatenates the KV cache, applies RoPE, then dispatches on ``attn_mode``:

  - ``torch`` / ``flash3``               : full attention (same math; torch runs it with SDPA)
  - ``full_history_parallel_attention``  : training regime of the full-history blocks
  - ``sink_recent_parallel_attention``   : training regime of the sink-recent blocks
  - ``chunk_inference_attention``        : inference regime (one chunk per call)
  - ``kv_recompute_causal``              : img-only block-wise causal (FBCA) recompute of the KV cache, a block per chunk

``attention`` is the FlashAttention-2 no-pad wrapper used by the token refiner.
"""

import torch
import torch.nn.functional as F

from movieroll.models.transformers.modules.streaming_attention import (
    full_history_parallel_attention, chunk_inference_attention,
    sink_recent_parallel_attention,
    block_wise_causal_attention_with_prefix,
)
from movieroll.models.transformers.modules.posemb_layers import apply_rotary_emb
from movieroll.utils.communications import all_gather, all_to_all_4D
from movieroll.utils.flash_attn_no_pad import (
    flash_attn,
    flash_attn_no_pad,
    flash_attn_no_pad_v3,
)
from movieroll.commons.parallel_states import get_sequence_parallel_state, nccl_info


def shrink_head(hidden_states):
    local_heads = hidden_states.shape[2] // nccl_info.sp_size
    return hidden_states.narrow(
        2, nccl_info.rank_within_spgroup * local_heads, local_heads
    )


def attention(
    q,
    k,
    v,
    drop_rate=0,
    attn_mask=None,
    causal=False,
):
    """FlashAttention-2 no-pad attention over q/k/v [B, S, H, D], returning [B, S, H*D].

    ``attn_mask`` is a [B, S] bool keep-mask (non-bool masks are cast); padded tokens are dropped before the kernel
    and restored afterwards.
    """

    qkv = torch.stack([q, k, v], dim=2)

    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()

    x = flash_attn_no_pad(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None)

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out

def parallel_attention(
    q, k, v,
    attn_mode=None,
    text_mask=None,
    freqs_cis_q=None, freqs_cis_k=None,
    kv_cache=None, return_kv=False,
    need_txt_compute=True,
    img_padding_info_dict=None,
    cond_padding_info_dict=None,
    cond_chunk_sizes=None,
    noisy_chunk_sizes=None,
    sink_chunks=1,
    window_has_sink=None,
    window_has_recent=None,
):
    """Attention of one double-stream block, dispatched by ``attn_mode``.

    q/k/v are each (cond, img, txt) or (img, txt) tuples of [B, S, H, D]. Under sequence parallelism the img
    and cond parts are all-to-all'd (sequence-sharded -> head-sharded) on the way in and back on the way out,
    while the text part is head-sharded with ``shrink_head`` and all-gathered afterwards.

    Order matters: the KV cached for ``return_kv`` is taken after the all-to-all but **before** RoPE, so a
    cached chunk can be re-used at a different position; RoPE comes after the cache is concatenated.

    Returns (hidden_states [B, S, H*D], kv_cache_return or None).
    """
    assert attn_mode is not None, "attn_mode is None"
    cond_query, query, encoder_query = q if len(q) == 3 else (None, q[0], q[1])
    cond_key, key, encoder_key = k if len(k) == 3 else (None, k[0], k[1])
    cond_value, value, encoder_value = v if len(v) == 3 else (None, v[0], v[1])

    cond_img_seq_len_query = cond_query.shape[1] if cond_query is not None else 0
    cond_img_seq_len_key = cond_key.shape[1] if cond_key is not None else 0

    if get_sequence_parallel_state():
        # batch_size, seq_len, attn_heads, head_dim
        if cond_query is not None and cond_img_seq_len_query > 0:
            cond_query = all_to_all_4D(
                cond_query, nccl_info.sp_group, scatter_dim=2, gather_dim=1,
                padding_info_dict=cond_padding_info_dict
            )
            cond_img_seq_len_query = cond_query.shape[1]
        if cond_key is not None and cond_img_seq_len_key > 0:
            cond_key = all_to_all_4D(
                cond_key, nccl_info.sp_group, scatter_dim=2, gather_dim=1,
                padding_info_dict=cond_padding_info_dict
            )
            cond_img_seq_len_key = cond_key.shape[1]
        if cond_value is not None and cond_img_seq_len_key > 0:
            cond_value = all_to_all_4D(
                cond_value, nccl_info.sp_group, scatter_dim=2, gather_dim=1,
                padding_info_dict=cond_padding_info_dict
            )
        if query is not None:
            query = all_to_all_4D(
                query, nccl_info.sp_group, scatter_dim=2, gather_dim=1,
                padding_info_dict=img_padding_info_dict
            )
        if key is not None and value is not None:
            key = all_to_all_4D(
                key, nccl_info.sp_group, scatter_dim=2, gather_dim=1,
                padding_info_dict=img_padding_info_dict
            )
            value = all_to_all_4D(
                value, nccl_info.sp_group, scatter_dim=2, gather_dim=1,
                padding_info_dict=img_padding_info_dict
            )

        if need_txt_compute:
            encoder_query = shrink_head(encoder_query)
            encoder_key = shrink_head(encoder_key)
            encoder_value = shrink_head(encoder_value)
        else:
            encoder_query = None
            encoder_key = None
            encoder_value = None

    if query is not None and cond_query is not None:
        query = torch.cat([cond_query, query], dim=1)
        cond_query = None
    if key is not None and cond_key is not None:
        key = torch.cat([cond_key, key], dim=1)
        cond_key = None
    if value is not None and cond_value is not None:
        value = torch.cat([cond_value, value], dim=1)
        cond_value = None

    if return_kv:
        # cache the post-all_to_all key/value from before RoPE, so RoPE can be re-applied at a new position
        kv_cache_return = (key.clone().detach(), value.clone().detach())

    if kv_cache is not None and len(kv_cache) == 2:
        key_cache, value_cache = kv_cache
        key_cache = key_cache.to(key.dtype)
        value_cache = value_cache.to(value.dtype)
        key = torch.cat([key_cache, key], dim=1)
        value = torch.cat([value_cache, value], dim=1)

    # RoPE comes after the KV cache is concatenated: only then is the KV complete
    if freqs_cis_q is not None and freqs_cis_k is not None:
        # Imported here, not at module level: triton_kernels imports triton eagerly, which must not happen in a
        # CPU-only process.
        from movieroll.models.transformers.modules.triton_kernels import is_triton_enabled
        if query.is_cuda and is_triton_enabled():
            from movieroll.models.transformers.modules.triton_kernels import fused_apply_rotary_emb
            query = fused_apply_rotary_emb(query, freqs_cis_q)
            key = fused_apply_rotary_emb(key, freqs_cis_k)
        else:
            query = apply_rotary_emb(query, freqs_cis_q)
            key = apply_rotary_emb(key, freqs_cis_k)

    # [b, s, h, d]
    sequence_length = query.size(1)
    key_sequence_length = key.size(1)
    encoder_sequence_length = encoder_query.size(1) if encoder_query is not None else 0

    if attn_mode == "torch":
        # Same math as the "flash3" branch, written with F.scaled_dot_product_attention:
        #   - text padding is masked on the key side, so no token attends to a padded text token;
        #   - a padded text query row gets the query itself at batch size 1, zeros otherwise, matching the flash
        #     wrappers (those positions are masked as keys, so no valid token ever reads them).
        # Unlike flash, the padded tokens still enter the matmuls here, so they must be finite.
        query_mask = None
        attn_mask = None
        if need_txt_compute:
            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)
            if text_mask is not None:
                query_mask = F.pad(text_mask.bool(), (sequence_length, 0), value=True)    # [B, Lq]
                key_mask = F.pad(text_mask.bool(), (key_sequence_length, 0), value=True)  # [B, Lk]
                attn_mask = key_mask[:, None, None, :]                                    # [B, 1, 1, Lk]

        # SDPA works on [B, H, L, D]
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        ).transpose(1, 2)

        if query_mask is not None:
            padded_rows = query if query.shape[0] == 1 else torch.zeros_like(query)
            hidden_states = torch.where(query_mask[:, :, None, None], hidden_states, padded_rows)

    # a 1-D mask is enough: concatenate img and txt and run one flash attention
    elif attn_mode == "flash3":
        # Only the score model reaches this mode: it always passes an (img, txt) pair with a text mask and never
        # a KV cache, so the text is always computed and q / kv have the same length. The assert names that
        # premise, since the KV-cache concatenation above is the one step that can lengthen the key alone.
        # One further premise cannot be checked here: at batch size 1 flash_attn_no_pad_v3 keeps the first
        # `valid_len` rows, so the valid text must be a PREFIX -- reorder_txt_token guarantees that.
        assert need_txt_compute and sequence_length == key_sequence_length, (
            f"flash3 is the score model's mode only: need_txt_compute={need_txt_compute}, "
            f"query length {sequence_length} vs key length {key_sequence_length}"
        )
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        # B, S, 3, H, D
        qkv = torch.stack([query, key, value], dim=2)
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
        hidden_states = flash_attn_no_pad_v3(qkv, attn_mask, causal=False, softmax_scale=None)
    elif attn_mode == "full_history_parallel_attention":
        if cond_img_seq_len_query > 0:
            cond_query, query = query[:, :cond_img_seq_len_query], query[:, cond_img_seq_len_query:]
        if cond_img_seq_len_key > 0:
            cond_key, key = key[:, :cond_img_seq_len_key], key[:, cond_img_seq_len_key:]
            cond_value, value = value[:, :cond_img_seq_len_key], value[:, cond_img_seq_len_key:]
        hidden_states = full_history_parallel_attention(
            query, key, value,
            encoder_query, encoder_key, encoder_value,
            cond_query, cond_key, cond_value,
            text_mask=text_mask,
            cond_chunk_sizes=cond_chunk_sizes,
            noisy_chunk_sizes=noisy_chunk_sizes,
        )
    elif attn_mode == "sink_recent_parallel_attention":
        if cond_img_seq_len_query > 0:
            cond_query, query = query[:, :cond_img_seq_len_query], query[:, cond_img_seq_len_query:]
        if cond_img_seq_len_key > 0:
            cond_key, key = key[:, :cond_img_seq_len_key], key[:, cond_img_seq_len_key:]
            cond_value, value = value[:, :cond_img_seq_len_key], value[:, cond_img_seq_len_key:]
        hidden_states = sink_recent_parallel_attention(
            query, key, value,
            encoder_query, encoder_key, encoder_value,
            cond_query, cond_key, cond_value,
            text_mask=text_mask,
            cond_chunk_sizes=cond_chunk_sizes,
            noisy_chunk_sizes=noisy_chunk_sizes,
            sink_chunks=sink_chunks,
            window_has_sink=window_has_sink,
            window_has_recent=window_has_recent,
        )
    elif attn_mode == "chunk_inference_attention":
        hidden_states = chunk_inference_attention(
            query, key, value,
            encoder_query, encoder_key, encoder_value,
            text_mask=text_mask,
        )
    elif attn_mode == "kv_recompute_causal":
        # KV recompute: img-only FBCA with one block per chunk, chunk_i attends to chunks 0..i.
        # noisy_chunk_sizes holds the per-chunk length in tokens (the first one may differ, the rest are equal).
        num_chunks = len(noisy_chunk_sizes)
        if num_chunks <= 1:
            hidden_states = flash_attn(query, key, value, causal=False)
        else:
            regular_chunk_size = noisy_chunk_sizes[-1]
            causal_prefix_key = key[:, :-regular_chunk_size]
            causal_prefix_value = value[:, :-regular_chunk_size]
            hidden_states = block_wise_causal_attention_with_prefix(
                query, key, value,
                causal_prefix_key=causal_prefix_key,
                causal_prefix_value=causal_prefix_value,
                block_sizes=noisy_chunk_sizes,
            )
    else:
        raise NotImplementedError

    if get_sequence_parallel_state():
        if need_txt_compute:
            hidden_states, encoder_hidden_states = hidden_states.split_with_sizes((sequence_length, encoder_sequence_length), dim=1)

        if cond_img_seq_len_query > 0:
            cond_hidden_states, hidden_states = hidden_states[:, :cond_img_seq_len_query], hidden_states[:, cond_img_seq_len_query:]
            cond_hidden_states = all_to_all_4D(
                cond_hidden_states, nccl_info.sp_group, scatter_dim=1, gather_dim=2,
                padding_info_dict=cond_padding_info_dict
            )

        hidden_states = all_to_all_4D(
            hidden_states, nccl_info.sp_group, scatter_dim=1, gather_dim=2,
            padding_info_dict=img_padding_info_dict
        )

        if cond_img_seq_len_query > 0:
            hidden_states = torch.cat([cond_hidden_states, hidden_states], dim=1)

        if need_txt_compute:
            encoder_hidden_states = all_gather(encoder_hidden_states, dim=2, group=nccl_info.sp_group)
            hidden_states = hidden_states.to(query.dtype)
            encoder_hidden_states = encoder_hidden_states.to(query.dtype)
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        else:
            hidden_states = hidden_states.to(query.dtype)

    b, s = hidden_states.shape[0], hidden_states.shape[1]
    hidden_states = hidden_states.reshape(b, s, -1)

    if return_kv:
        return hidden_states, kv_cache_return

    return hidden_states, None
