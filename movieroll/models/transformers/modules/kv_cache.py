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

"""
KV cache management utilities.
"""
import torch

from movieroll.commons.streaming_config import StreamingConfig
from movieroll.models.transformers.modules.streaming_utils import safe_get_rank

class KVCacheManager:
    """KV cache manager that keeps one merged tensor per layer.

    Storage:
      _merged_cache[name] = (merged_k, merged_v) | None
      kv_cache_frame_indices[name] = [(start, end), ...]

    `update_kv_cache` concatenates new chunks onto the merged tensors; nothing is ever popped (the cache keeps the
    full clean history). The sink-recent layers of HieraKV are trimmed back to the sink chunk with
    `trim_to_sink`, and KV recompute rebuilds the cache through `soft_clear` + `replace_kv_cache`. `get_kv_cache`
    returns the merged tensors as they are, or a zero-copy `expand` when one cached batch element serves several
    CFG branches; a cached batch size other than 1 falls back to `repeat`, which copies.
    """

    def __init__(self, streaming_config: StreamingConfig):
        self.kv_cache_save_dtype = streaming_config.kv_cache_save_dtype
        self._merged_cache = {}
        self.kv_cache_frame_indices = {}
        self.first_layer_name = None

        self._is_init = False
        self.verbose = True  # set to False by callers that want quiet logs (distilled sampling, DMD training)

    @property
    def is_init(self):
        return self._is_init

    def _frame_indices_to_tokens(self, frame_indices_entry):
        """Number of tokens of one (start, end) frame range."""
        start, end = frame_indices_entry
        return (end - start) * self.height * self.width

    def init_kv_cache(self, name_list, hw):
        self.first_layer_name = name_list[0]
        self._merged_cache = dict.fromkeys(name_list)
        self.kv_cache_frame_indices = {name: [] for name in name_list}
        self.height, self.width = hw
        self._is_init = True
        if self.verbose and safe_get_rank() == 0:
            print("KVCacheManager initialized")

    def clear_kv_cache(self):
        self.first_layer_name = None
        self._merged_cache = {}
        self.kv_cache_frame_indices = {}
        self.height, self.width = None, None
        self._is_init = False
        if self.verbose and safe_get_rank() == 0:
            print(f"KVCacheManager cleared, is_init: {self.is_init}, kv_cache_frame_indices: {self.kv_cache_frame_indices}")

    def _update(self, name, new_k, new_v, start_index, end_index):
        prev = self._merged_cache.get(name)
        if prev is not None:
            orig_shape = (prev[0].shape[0], prev[0].shape[2], prev[0].shape[3])
            new_shape = (new_k.shape[0], new_k.shape[2], new_k.shape[3])
            if orig_shape != new_shape:
                raise ValueError(f"KV cache dimension mismatch for {name}: orig_shape {orig_shape}, new_shape {new_shape}")
            self._merged_cache[name] = (
                torch.cat([prev[0], new_k], dim=1),
                torch.cat([prev[1], new_v], dim=1),
            )
        else:
            self._merged_cache[name] = (new_k, new_v)

        if name not in self.kv_cache_frame_indices:
            self.kv_cache_frame_indices[name] = []
        self.kv_cache_frame_indices[name].append((start_index, end_index))

    def trim_to_sink(self, name, num_sink_chunks):
        """HieraKV sink-recent blocks only: called *before* a new chunk is encoded, trims this layer's cache to its
        first num_sink_chunks chunks (i.e. pops the previous recent chunk).

        The new chunk then attends to the sink only, matching training where the middle cond chunks are dropped
        and a recent chunk "sees only the sink"; after encoding, update_kv_cache appends the current chunk, so the
        cache is back to {sink, recent} = num_sink_chunks + 1 chunks.
        """
        indices = self.kv_cache_frame_indices.get(name, [])
        if len(indices) <= num_sink_chunks:
            return
        keep_tokens = sum(self._frame_indices_to_tokens(idx) for idx in indices[:num_sink_chunks])
        self.kv_cache_frame_indices[name] = indices[:num_sink_chunks]
        merged = self._merged_cache.get(name)
        if merged is not None:
            k, v = merged
            self._merged_cache[name] = (
                k[:, :keep_tokens].clone(),
                v[:, :keep_tokens].clone(),
            )

    def update_kv_cache(self, name, new_k, new_v, start_index, end_index):
        assert self.is_init, "KVCacheManager is not initialized, call init_kv_cache first"

        # validate the inputs
        if new_k.shape != new_v.shape:
            raise ValueError(f"new_k shape {new_k.shape} must match new_v shape {new_v.shape}")

        new_k, new_v = new_k.contiguous(), new_v.contiguous()
        new_k, new_v = new_k.to(dtype=self.kv_cache_save_dtype), new_v.to(dtype=self.kv_cache_save_dtype)
        self._update(name, new_k, new_v, start_index, end_index)

        if self.verbose and name == self.first_layer_name and safe_get_rank() == 0:
            kv_cache_info = ""
            merged = self._merged_cache.get(name)
            if merged is not None:
                kv_cache_info = f"dtype: {merged[0].dtype}, device: {merged[0].device}"
            print(f"after update, kv_cache_frame_indices[{name}]: {self.kv_cache_frame_indices[name]}, "
                  f"{kv_cache_info}")

    def get_kv_cache(self, name, current_batch_size: int = 1):
        """Merged KV cache of a layer: the internal tensors, broadcast to `current_batch_size` if needed."""
        merged = self._merged_cache.get(name)
        if merged is None:
            return None

        k, v = merged

        # The cond chunks do not depend on the text, so one copy serves both CFG branches: usually a zero-copy
        # expand, but a non-unit cached batch size falls back to repeat(), which allocates.
        if k.shape[0] < current_batch_size:
            if k.shape[0] == 1:
                k = k.expand(current_batch_size, -1, -1, -1)
                v = v.expand(current_batch_size, -1, -1, -1)
            else:
                if current_batch_size % k.shape[0] != 0:
                    raise ValueError(f"current_batch_size {current_batch_size} must be divisible by cached batch size {k.shape[0]}")
                _repeat = current_batch_size // k.shape[0]
                k = k.repeat(_repeat, 1, 1, 1)
                v = v.repeat(_repeat, 1, 1, 1)
        return (k, v)

    def get_kv_cache_frame_indices(self, name=None):
        name = name or self.first_layer_name
        return self.kv_cache_frame_indices.get(name, [])

    def get_cached_cond_latents_num_frames(self, name=None):
        return sum(self.get_cached_cond_chunk_sizes(name))

    def get_cached_cond_chunk_sizes(self, name=None):
        cond_chunk_sizes = []
        name = name or self.first_layer_name
        if self.kv_cache_frame_indices.get(name, None) is not None and len(self.kv_cache_frame_indices[name]) > 0:
            for start_index, end_index in self.kv_cache_frame_indices[name]:
                cond_chunk_sizes.append(end_index - start_index)
        return cond_chunk_sizes

    def should_recompute(self, threshold: int) -> bool:
        """Whether the number of cached chunks reached the KV recompute threshold."""
        if not self._is_init or self.first_layer_name is None:
            return False
        return len(self.kv_cache_frame_indices.get(self.first_layer_name, [])) >= threshold

    def soft_clear(self):
        """Drop the cached data of every layer but keep the init state (_is_init / height / width /
        first_layer_name). KV recompute rebuilds the caches of both block groups together, so no layer is kept."""
        for name in self._merged_cache:
            self._merged_cache[name] = None
            self.kv_cache_frame_indices[name] = []
        if self.verbose and safe_get_rank() == 0:
            print(f"KVCacheManager soft_clear done, is_init: {self._is_init}")

    def replace_kv_cache(self, name, new_k, new_v, new_frame_indices):
        """Replace a layer's merged KV tensors and frame indices.

        new_k / new_v: [B, total_tokens, num_heads, head_dim]
        new_frame_indices: List[(start, end)], one frame range per chunk.
        """
        assert self.is_init, "KVCacheManager is not initialized, call init_kv_cache first"
        if new_k.shape != new_v.shape:
            raise ValueError(f"new_k shape {new_k.shape} must match new_v shape {new_v.shape}")
        new_k, new_v = new_k.contiguous(), new_v.contiguous()
        new_k = new_k.to(dtype=self.kv_cache_save_dtype)
        new_v = new_v.to(dtype=self.kv_cache_save_dtype)
        self._merged_cache[name] = (new_k, new_v)
        self.kv_cache_frame_indices[name] = list(new_frame_indices)
