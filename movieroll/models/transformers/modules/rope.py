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
RoPE resampling utilities.
"""
from collections import OrderedDict
from typing import Any, Hashable, List, Optional, Tuple

import torch

from movieroll.commons.streaming_config import StreamingConfig
from movieroll.models.transformers.modules.posemb_layers import get_nd_rotary_pos_embed
from movieroll.models.transformers.modules.streaming_utils import safe_get_rank

class SafeRoPECacheManager:
    """LRU cache of the (q, k) RoPE tables used at inference."""

    DEFAULT_MAX_CACHE_SIZE = 4

    def __init__(
        self,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
    ):
        self.max_cache_size = max_cache_size
        self.freq_cis_q_cache: OrderedDict[Hashable, Any] = OrderedDict()
        self.freq_cis_k_cache: OrderedDict[Hashable, Any] = OrderedDict()

    def is_exist(self, key: Hashable) -> bool:
        return key in self.freq_cis_q_cache and key in self.freq_cis_k_cache

    def get(self, key: Hashable) -> Any:
        try:
            freqs_cis_q = self.freq_cis_q_cache[key]
            freqs_cis_k = self.freq_cis_k_cache[key]
            self.freq_cis_q_cache.move_to_end(key, last=True)
            self.freq_cis_k_cache.move_to_end(key, last=True)
            return freqs_cis_q, freqs_cis_k
        except KeyError:
            return None, None

    def _evict_if_needed(self):
        while len(self.freq_cis_q_cache) > self.max_cache_size > 0:
            self.freq_cis_q_cache.popitem(last=False)
            self.freq_cis_k_cache.popitem(last=False)

    def set(self, key: Hashable, freqs_cis_q: Any, freqs_cis_k: Any):
        self.freq_cis_q_cache[key] = freqs_cis_q
        self.freq_cis_k_cache[key] = freqs_cis_k
        self.freq_cis_q_cache.move_to_end(key, last=True)
        self.freq_cis_k_cache.move_to_end(key, last=True)
        self._evict_if_needed()


class RoPEResampler:
    """RoPE tables of the streaming transformer on a fixed frame budget (`max_num_latent_frames`).

    Every token sits at the frame position it has during rolling inference, where the KV cache is re-packed
    from position 0; the parallel training layouts reproduce the same relative positions. Each call returns
    `((q_cos, q_sin), (k_cos, k_sin))`, rows ordered like the tokens.
    """
    def __init__(self, streaming_config: StreamingConfig):
        self.max_num_latent_frames = streaming_config.max_num_latent_frames

        self.current_rope_cis: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None)
        self.current_rope_sizes: Tuple[int, int, int] = (-1, -1, -1)

        if safe_get_rank() == 0:
            print(f"Initialized RoPE resampler with max_num_latent_frames: {self.max_num_latent_frames}")

        self.rope_cache_manager = SafeRoPECacheManager()

    def get_rotary_pos_embed(
        self,
        rope_sizes,
        rope_dim_list,
        head_dim,
        rope_theta,
        device="cuda",
    ):
        target_ndim = 3
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=rope_theta,
            theta_rescale_factor=1.0,
            interpolation_factor=1.0,
            device=device,
        )

        return freqs_cos, freqs_sin

    def _retrieval_rope_with_indices(
        self,
        hw: Tuple[int, int],
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        indices: List[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather the RoPE rows of the frame ranges [(start, end), ...] and concatenate them."""
        H, W = hw
        freqs_cos_list = []
        freqs_sin_list = []
        for start, end in indices:
            assert start < end, f"start {start} must be less than end {end}"
            freqs_cos_list.append(freqs_cos[start * H * W: end * H * W])
            freqs_sin_list.append(freqs_sin[start * H * W: end * H * W])
        return torch.cat(freqs_cos_list, dim=0), torch.cat(freqs_sin_list, dim=0)

    # ------------------------------------------------------------------
    # Training resample
    # ------------------------------------------------------------------
    def _resample_training(
        self,
        hw: Tuple[int, int],
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        cond_latents_num_frames: int,
        noisy_latents_num_frames: int,
        overlap_latents_num_frames: int,
        noisy_chunk_sizes_tokens: Optional[List[int]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """RoPE of the parallel training layout [cond, noisy] (q == k).

        With cond frames, cond takes freqs[0 : cond_num * H*W] and noisy takes
        freqs[(cond_num - overlap) * H*W : (cond_num - overlap + noisy_num) * H*W].
        Without cond frames (chunk_wise_reset), every noisy chunk restarts at freqs[0 : chunk_size_tokens].
        """
        H, W = hw

        if cond_latents_num_frames > 0:
            assert 0 <= overlap_latents_num_frames <= cond_latents_num_frames, (
                f"overlap_latents_num_frames ({overlap_latents_num_frames}) must be in "
                f"[0, cond_latents_num_frames={cond_latents_num_frames}]"
            )
            assert overlap_latents_num_frames <= noisy_latents_num_frames, (
                f"overlap_latents_num_frames ({overlap_latents_num_frames}) must be <= "
                f"noisy_latents_num_frames ({noisy_latents_num_frames})"
            )

            cond_start, cond_end = 0, cond_latents_num_frames
            noisy_start = cond_latents_num_frames - overlap_latents_num_frames
            noisy_end = noisy_start + noisy_latents_num_frames

            max_used_frame = max(cond_end, noisy_end)
            if max_used_frame * H * W > freqs_cos.shape[0]:
                raise ValueError(
                    f"Used frame range [0, {max_used_frame}) (cond_end={cond_end}, "
                    f"noisy_end={noisy_end}) exceeds freqs_cos length {freqs_cos.shape[0]} "
                    f"under H={H}, W={W}"
                )

            indices = []
            if cond_end > cond_start:
                indices.append((cond_start, cond_end))
            indices.append((noisy_start, noisy_end))

            freqs_cos_out, freqs_sin_out = self._retrieval_rope_with_indices(
                hw, freqs_cos, freqs_sin, indices
            )
            return (freqs_cos_out, freqs_sin_out), (freqs_cos_out, freqs_sin_out)

        # -- cond_latents_num_frames == 0 -> chunk_wise_reset --
        assert noisy_chunk_sizes_tokens is not None and len(noisy_chunk_sizes_tokens) > 0, (
            "noisy_chunk_sizes_tokens must be provided when cond_latents_num_frames == 0 "
            "(chunk_wise_reset requires per-chunk token sizes)"
        )
        assert sum(noisy_chunk_sizes_tokens) == noisy_latents_num_frames * H * W, (
            f"sum(noisy_chunk_sizes_tokens) ({sum(noisy_chunk_sizes_tokens)}) must equal "
            f"noisy_latents_num_frames * H * W ({noisy_latents_num_frames * H * W})"
        )

        parts_cos: List[torch.Tensor] = []
        parts_sin: List[torch.Tensor] = []
        for cs in noisy_chunk_sizes_tokens:
            if cs > freqs_cos.shape[0]:
                raise ValueError(
                    f"chunk_wise_reset: chunk_size_tokens={cs} exceeds freqs_cos length "
                    f"{freqs_cos.shape[0]}"
                )
            parts_cos.append(freqs_cos[:cs])
            parts_sin.append(freqs_sin[:cs])

        combined_cos = torch.cat(parts_cos, dim=0)
        combined_sin = torch.cat(parts_sin, dim=0)
        return (combined_cos, combined_sin), (combined_cos, combined_sin)

    def _resample_training_slot(
        self,
        hw: Tuple[int, int],
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        sink_frames: int,
        recent_frames: int,
        noisy_window_frames: List[int],
        window_has_sink: Optional[List[bool]] = None,
        window_has_recent: Optional[List[bool]] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Slot RoPE of the sink-recent blocks in parallel training, strictly per window.

        Slot frames: sink [0, sink_f), recent [sink_f, sink_f + rec_f), window i [s_w, s_w + win_i_f), where
        s_w is sink_f + rec_f, sink_f or 0 according to window_has_recent[w] / window_has_sink[w] -- the same
        view that window gets in the step-by-step inference rollout. Output rows follow the [cond, noisy]
        token layout with the middle cond tokens dropped:
        [cond(sink? + one recent per window with a recent chunk), noisy(one entry per window)].
        """
        sink_f = sink_frames
        rec_f = recent_frames
        N = len(noisy_window_frames)
        if window_has_sink is None:
            window_has_sink = [True] * N
        if window_has_recent is None:
            window_has_recent = [True] * N
        assert len(window_has_sink) == N and len(window_has_recent) == N

        def _slot_start(w: int) -> int:
            if window_has_recent[w]:
                return sink_f + rec_f
            if window_has_sink[w]:
                return sink_f
            return 0

        max_used_frame = 0
        for w, win_f in enumerate(noisy_window_frames):
            max_used_frame = max(max_used_frame, _slot_start(w) + win_f)
        if max_used_frame * hw[0] * hw[1] > freqs_cos.shape[0]:
            raise ValueError(
                f"slot RoPE used frame {max_used_frame} exceeds freqs length "
                f"{freqs_cos.shape[0] // (hw[0] * hw[1])}"
            )

        indices: List[Tuple[int, int]] = []
        # cond: the sink (if any window uses it) + one recent slot per window with a recent chunk
        if any(window_has_sink) and sink_f > 0:
            indices.append((0, sink_f))
        if rec_f > 0:
            for w in range(N):
                if window_has_recent[w]:
                    indices.append((sink_f, sink_f + rec_f))
        # noisy: the self part of every window, at the slot start its cond visibility implies
        for w, win_f in enumerate(noisy_window_frames):
            s = _slot_start(w)
            indices.append((s, s + win_f))

        freqs_cos_out, freqs_sin_out = self._retrieval_rope_with_indices(
            hw, freqs_cos, freqs_sin, indices
        )
        return (freqs_cos_out, freqs_sin_out), (freqs_cos_out, freqs_sin_out)

    def resample_training_slot(
        self,
        hw: Tuple[int, int],
        sink_frames: int,
        recent_frames: int,
        noisy_window_frames: List[int],
        window_has_sink: Optional[List[bool]] = None,
        window_has_recent: Optional[List[bool]] = None,
        rope_dim_list: Optional[List[int]] = None,
        head_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        device="cuda",
    ):
        """Build the base RoPE table and sample it by slot (sink-recent blocks in training)."""
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            (self.max_num_latent_frames, hw[0], hw[1]),
            rope_dim_list, head_dim, rope_theta, device=device,
        )
        return self._resample_training_slot(
            hw, freqs_cos, freqs_sin,
            sink_frames=sink_frames,
            recent_frames=recent_frames,
            noisy_window_frames=noisy_window_frames,
            window_has_sink=window_has_sink,
            window_has_recent=window_has_recent,
        )

    # ------------------------------------------------------------------
    # Inference resample
    # ------------------------------------------------------------------
    def _resample_inference(
        self,
        hw: Tuple[int, int],
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        noisy_latents_num_frames: int,
        kv_cache_frame_indices: List[Tuple[int, int]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Inference RoPE with the KV cache re-packed from position 0.

        q takes the freqs right after the end of the re-packed cache; k takes [cache_freqs, q_freqs].
        """
        H, W = hw

        resampled_kv_cache_frame_indices: List[Tuple[int, int]] = []
        offset = 0
        for start_index, end_index in kv_cache_frame_indices:
            resampled_kv_cache_frame_indices.append((offset, offset + end_index - start_index))
            offset += end_index - start_index
        cache_end_frame = offset

        q_start = cache_end_frame * H * W
        q_len = noisy_latents_num_frames * H * W
        q_end = q_start + q_len

        if q_end > freqs_cos.shape[0]:
            raise ValueError(
                f"Inference q_end {q_end} (cache_end_frame={cache_end_frame}, "
                f"noisy_latents_num_frames={noisy_latents_num_frames}, H={H}, W={W}) "
                f"exceeds freqs_cos length {freqs_cos.shape[0]}"
            )

        freq_cis_q_cos = freqs_cos[q_start:q_end]
        freq_cis_q_sin = freqs_sin[q_start:q_end]
        freq_cis_q = (freq_cis_q_cos, freq_cis_q_sin)

        if resampled_kv_cache_frame_indices:
            cache_cos, cache_sin = self._retrieval_rope_with_indices(
                hw, freqs_cos, freqs_sin, resampled_kv_cache_frame_indices
            )
            freq_cis_k = (
                torch.cat([cache_cos, freq_cis_q_cos], dim=0),
                torch.cat([cache_sin, freq_cis_q_sin], dim=0),
            )
        else:
            freq_cis_k = freq_cis_q
        return freq_cis_q, freq_cis_k

    def __call__(
        self,
        hw: Tuple[int, int],
        cond_latents_num_frames: int = 0,
        noisy_latents_num_frames: int = 0,
        overlap_latents_num_frames: int = 0,
        noisy_chunk_sizes_tokens: Optional[List[int]] = None,
        current_frame_index: int = 0,
        kv_cache_frame_indices: Optional[List[Tuple[int, int]]] = None,
        training: bool = True,
        rope_dim_list: Optional[List[int]] = None,
        head_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        device="cuda",
    ):
        H, W = hw

        if kv_cache_frame_indices is None:
            kv_cache_frame_indices = []

        if training:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(
                (self.max_num_latent_frames, H, W),
                rope_dim_list, head_dim, rope_theta,
                device=device,
            )
            return self._resample_training(
                hw, freqs_cos, freqs_sin,
                cond_latents_num_frames=cond_latents_num_frames,
                noisy_latents_num_frames=noisy_latents_num_frames,
                overlap_latents_num_frames=overlap_latents_num_frames,
                noisy_chunk_sizes_tokens=noisy_chunk_sizes_tokens,
            )

        # inference
        total_used_frames = cond_latents_num_frames + noisy_latents_num_frames
        if total_used_frames > self.max_num_latent_frames:
            raise ValueError(
                f"Total num of latent frames {total_used_frames} (cond: {cond_latents_num_frames}, "
                f"noisy: {noisy_latents_num_frames}) exceeds max_num_latent_frames "
                f"{self.max_num_latent_frames}"
            )

        cache_keys = (
            current_frame_index,
            tuple(kv_cache_frame_indices),
            cond_latents_num_frames,
            noisy_latents_num_frames,
            H, W,
        )
        if self.rope_cache_manager.is_exist(cache_keys):
            return self.rope_cache_manager.get(cache_keys)

        if (
            self.current_rope_cis != (None, None)
            and self.current_rope_sizes == (self.max_num_latent_frames, H, W)
        ):
            freqs_cos, freqs_sin = self.current_rope_cis
        else:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed(
                (self.max_num_latent_frames, H, W),
                rope_dim_list, head_dim, rope_theta,
                device=device,
            )
            self.current_rope_cis = (freqs_cos, freqs_sin)
            self.current_rope_sizes = (self.max_num_latent_frames, H, W)

        freq_cis_q, freq_cis_k = self._resample_inference(
            hw, freqs_cos, freqs_sin,
            noisy_latents_num_frames=noisy_latents_num_frames,
            kv_cache_frame_indices=kv_cache_frame_indices,
        )

        self.rope_cache_manager.set(cache_keys, freq_cis_q, freq_cis_k)
        return freq_cis_q, freq_cis_k
