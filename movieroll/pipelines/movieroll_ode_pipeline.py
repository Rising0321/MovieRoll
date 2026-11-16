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

"""Stage-2 ODE-init sibling of the Stage-3 DMD pipeline."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from movieroll.pipelines.movieroll_pipeline import MovieRollStreamingPipeline


class MovieRollODEPipeline(MovieRollStreamingPipeline):
    """Owns ODE trajectory capture, the ODE SFT forward, and Stage-2 rolling validation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ode_capture = False
        self._ode_rounds = []
        self._ode_clean_chunks = []

    # ODE-init capture: the base rolling inference runs unchanged; this subclass only snapshots the
    # diagonal window state per round. A caller sets self._ode_capture, clears self._ode_rounds, calls
    # the base __call__ with output_type='latent', and reads back the clean latents; self._ode_rounds
    # then holds every round's diagonal window, prefill windows included.
    # Snapshot timing: after the round adds its noisy chunk and pops its clean one, before denoising.
    def _run_inner_denoising_loop(self, *, local_step_base, rolling_stage,
                                  current_frame_index, timesteps, **rest):
        """Snapshot the diagonal window per round while capturing; otherwise pass through."""
        if self._ode_capture:
            win = self.rolling_queue.get_current_latents()          # [B, C, winT, h, w]
            step_index = self.scheduler_wrapper.get_step_index(local_step_base)
            sidx = self.expand_step_index(step_index, rolling_stage)
            self._ode_rounds.append({
                "window_input": win[0].detach().cpu(),               # [C, winT, h, w] diagonal state
                "window_timesteps": timesteps[sidx].detach().cpu(),  # [winT] per-frame warped timestep
                "start_frame": int(current_frame_index),             # global index of the window's first frame
            })
        return super()._run_inner_denoising_loop(
            local_step_base=local_step_base, rolling_stage=rolling_stage,
            current_frame_index=current_frame_index, timesteps=timesteps, **rest)

    def _pop_and_output_clean_chunk(self, **kw):
        """While capturing, keep the raw clean latent chunk that was just popped.

        It has to be taken here: the base __call__'s output_type normalisation would rescale the latent
        into [0, 1], and the regression target needs the raw one (std about 1).
        """
        output_latents = kw.get("output_latents")
        prev = len(output_latents) if output_latents is not None else 0
        ret = super()._pop_and_output_clean_chunk(**kw)
        if self._ode_capture and output_latents is not None and len(output_latents) > prev:
            self._ode_clean_chunks.append(output_latents[-1].detach().float().cpu())  # [B,C,ck,h,w]
        return ret

    def rolling_ode_teacher_forcing_forward(
        self,
        *,
        cond_latents: Optional[torch.Tensor],       # [B,C,T_cond,h,w] clean history + overlap, or None
        noisy_latents: torch.Tensor,                # [B,C,T_noisy,h,w] the ODE diagonal (W1..WK)
        noisy_timesteps: torch.Tensor,              # [T_noisy] per-frame warped diagonal (model time)
        cond_chunk_sizes: List[int],                # flat: history chunks + overlap (W1..W_{K-1})
        noisy_chunk_sizes: List[List[int]],         # nested: [W1_chunks, ..., WK_chunks]
        event_cond: Tuple,                          # (prompt_embeds, prompt_mask, prompt_embeds_2, extra)
    ) -> torch.Tensor:
        """One differentiable multi-window ODE-init forward, laid out like SFT teacher forcing.

        Every noisy window sees only clean context -- the clean overlap prefix (full-history blocks) plus
        the sink and recent chunks -- which is what the rolling KV cache holds at inference. Gradients are
        on, so conditioning enters as input tokens and the clean cond frames declare model time 0.
        """
        prompt_embeds, prompt_mask, prompt_embeds_2, extra_kwargs = event_cond
        target_dtype = self.target_dtype
        autocast_enabled = target_dtype != torch.float32
        device = noisy_latents.device
        B = noisy_latents.shape[0]

        window_frame_counts = [sum(sub) for sub in noisy_chunk_sizes]  # frames per window
        num_windows = len(noisy_chunk_sizes)
        T_noisy = noisy_latents.shape[2]
        if num_windows < 1:
            raise ValueError("noisy_chunk_sizes must hold at least one window")
        if sum(window_frame_counts) != T_noisy:
            raise ValueError(
                f"noisy window frames {sum(window_frame_counts)} != T_noisy {T_noisy}"
            )

        # merge [cond | noisy]; cond frames sit at t=0. Must match Transport.merge_latents_to_model_input,
        # which is what the SFT path uses.
        cond_frames = cond_latents.shape[2] if cond_latents is not None else 0
        merged = torch.cat([cond_latents, noisy_latents], dim=2) if cond_latents is not None else noisy_latents
        # the 65-channel input contract: [latent, zeros(C), zeros(1)]
        gt_concat, mask_concat = self.prepare_latent_concat(merged)
        lmi = self.prepare_latent_model_input(
            merged, gt_concat, mask_concat,
            do_classifier_free_guidance=False)

        # frame-wise timestep: cond frames 0 (clean), noisy frames the warped diagonal
        noisy_t = noisy_timesteps.to(device).float().unsqueeze(0).repeat(B, 1)  # [B, T_noisy]
        if cond_frames > 0:
            t_full = torch.cat(
                [torch.zeros(B, cond_frames, device=device, dtype=noisy_t.dtype), noisy_t], dim=1)
        else:
            t_full = noisy_t

        # timestep_txt: each window's first frame (the cleanest), as StreamingTimestepTxtWrapper does
        txt_ts, off = [], cond_frames
        for wf in window_frame_counts:
            txt_ts.append(t_full[:, off:off + 1])
            off += wf
        timestep_txt = torch.cat(txt_ts, dim=1)                # [B, num_windows]

        # per-window 4D text: the same prompt repeated for every window. The layout, including the
        # .contiguous() calls, must match Transport.build_chunk_text.
        text_4d = prompt_embeds.unsqueeze(1).expand(-1, num_windows, -1, -1).contiguous()
        mask_4d = (prompt_mask.unsqueeze(1).expand(-1, num_windows, -1).contiguous()
                   if prompt_mask is not None else None)
        _extra = dict(extra_kwargs) if extra_kwargs else {}
        if _extra.get("byt5_text_states") is not None:
            _extra["byt5_text_states"] = _extra["byt5_text_states"].unsqueeze(1).expand(
                -1, num_windows, -1, -1).contiguous()
            _extra["byt5_text_mask"] = _extra["byt5_text_mask"].unsqueeze(1).expand(
                -1, num_windows, -1).contiguous()

        with torch.autocast(
            device_type=device.type,
            dtype=target_dtype,
            enabled=autocast_enabled and device.type == "cuda",
        ):
            noise_pred = self.transformer(
                lmi, t_full, text_4d, mask_4d,
                return_dict=False, extra_kwargs=_extra, return_kv=False,
                timestep_txt=timestep_txt,
                cond_chunk_sizes=list(cond_chunk_sizes),
                noisy_chunk_sizes=[list(w) for w in noisy_chunk_sizes],
                current_frame_index=0,
            )[0]   # [B, C, T_noisy, h, w]: the model strips the cond frames and returns the windows

        # velocity -> x0 with per-frame sigma; the trainer assembles the target
        window_x0 = self.scheduler.velocity_to_x0(
            model_output=noise_pred, timestep=noisy_t, sample=noisy_latents)
        return window_x0
