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

# ==============================================================================
# Modified from SiT (https://github.com/willisma/SiT)
# ==============================================================================

"""Flow-matching transport, and the six hook points the streaming trainer replaces.

`training_losses` is one training step: interpolate between data and noise at a drawn timestep, ask the model
for the velocity, compare against the target. Six of its steps are overridable methods --
`split_latents_to_clean_and_noisy`, `merge_latents_to_model_input`, `compute_timestep_txt`, `build_chunk_text`,
`compute_loss`, `get_timesteps` -- which `movieroll/trainer/flow/streaming_wrappers.py` replaces so that one
training step instead looks like one step of rolling inference.

A streaming configuration is mandatory (`__init__` asserts one), so of the defaults only `compute_loss`'s body
runs: the streaming loss wrapper calls it for the reduction.
"""

import torch as th

from movieroll.commons.streaming_config import StreamingConfig
from movieroll.trainer.flow.streaming_wrappers import (
    StreamingLossWrapper,
    StreamingMergeLatentsWrapper,
    StreamingSplitLatentsWrapper,
    StreamingTextSelectionWrapper,
    StreamingTimestepTxtWrapper,
    StreamingTimestepsWrapper,
)
from movieroll.trainer.flow.path import ICPlan
from movieroll.trainer.flow.utils import mean_flat
from movieroll.utils.parallel_sync import sync_object_for_parallel_training

__all__ = ["Transport"]

# The model sees timesteps on a 0..1000 scale while the path works in [0, 1].
TRAINING_TIMESTEPS = 1000


class Transport:

    def __init__(self, *, video_shift, streaming_config: StreamingConfig = None):
        self.path_sampler = ICPlan()
        # timestep shift: http://arxiv.org/abs/2403.03206
        self.video_shift = video_shift
        assert streaming_config is not None, "streaming_config must be provided"
        # read by the StreamingWrapper descriptors; the attribute name is part of that contract
        self.streaming_config = streaming_config

    @StreamingTimestepsWrapper
    def get_timesteps(self, batch_size, t0, t1, **kwargs):
        """Default: one logit-normal timestep per sample, over the whole clip.

        The streaming wrapper instead gives every chunk of the rolling window its own timestep.
        """
        mask = None
        u = th.normal(mean=0.0, std=1.0, size=(batch_size,))
        t = 1 / (1 + th.exp(-u)) * (t1 - t0) + t0
        return t, mask

    @StreamingSplitLatentsWrapper
    def split_latents_to_clean_and_noisy(self, latents):
        """Split the latents [B, C, T, H, W] into a clean cond part and a noisy part; by default all noisy.

        Returns (cond_latents, noisy_latents, cond_chunk_sizes, noisy_chunk_sizes, window_sizes,
        cond_latents_num_frames, noisy_latents_num_frames, noisy_start_index).
        """
        T = latents.shape[2]
        return (None, latents, [], [[T]], [1], 0, T, 0)

    @StreamingMergeLatentsWrapper
    def merge_latents_to_model_input(self, cond_latents, xt, cond_chunk_sizes=None, **kwargs):
        """Assemble the model input from the clean cond latents and the noisy xt; default concatenates on time.

        Returns (merged_latents, t_cond), t_cond being the frame-wise timestep (sigma) of the cond frames or None.
        """
        if cond_latents is not None:
            return th.cat([cond_latents, xt], dim=2), None
        return xt, None

    @StreamingTimestepTxtWrapper
    def compute_timestep_txt(self, t, cond_latents_num_frames=0, noisy_chunk_sizes=None):
        """Timestep used for text modulation; None by default.

        The streaming wrapper takes the first frame of each window -- the one closest to the history, i.e. the
        least noisy.
        """
        return

    @StreamingTextSelectionWrapper
    def build_chunk_text(self, model_kwargs, cond_chunk_sizes, noisy_chunk_sizes, noisy_start_index=0):
        """Put the per-window text tensor, 4D ``[B, num_windows, S, D]``, into model_kwargs.

        The default is a no-op that leaves the text 3D.
        """

    @StreamingLossWrapper
    def compute_loss(self, model_output, ut, **kwargs):
        """Unweighted flow-matching loss: the per-sample mean of (model_output - ut) ** 2 in float32, {"loss": [B]}.

        The streaming wrapper adds "per_chunk_losses" (per window / chunk means, for logging only) when
        `noisy_chunk_sizes` is given.
        """
        loss = mean_flat((model_output.to(th.float32) - ut.to(th.float32)) ** 2)
        return dict(loss=loss)

    def sample(self, x1, chunk_sizes=None, window_sizes=None, *, data_type):
        """Sample the noise, the timesteps and the shifted time grid of one step.

        Args:
            x1: the noisy part of the clip, [B, C, T, H, W]
            chunk_sizes / window_sizes: the rolling layout, forwarded to get_timesteps
            data_type: must be "video"
        """
        assert data_type == "video", f"data_type must be 'video', got {data_type}"
        shift = self.video_shift

        x0 = th.randn_like(x1)

        # The linear path with a velocity target draws its timesteps from the whole [0, 1].
        t0, t1 = 0, 1
        t, mask = self.get_timesteps(
            batch_size=len(x1),
            t0=t0,
            t1=t1,
            chunk_sizes=chunk_sizes,
            window_sizes=window_sizes,
        )

        if shift != 1.:
            # SD3's timestep shift, t' = s*t / (1 + (s-1)*t)
            t = (shift * t) / (1 + (shift - 1) * t)

        t = t.to(device=x1.device, dtype=x1.dtype)
        return t, x0, x1

    def get_model_t(self, t):
        return t * TRAINING_TIMESTEPS

    def training_losses(self, model, x1, model_kwargs=None, *, data_type,
                        multitask_mask=None, mask_type=None, parallel_dims=None):
        """One training step of the score model: `model` predicts the velocity, `x1` is the clip [B, C, T, H, W].

        Returns (model_output, terms) where terms holds "loss" and "t".
        """
        if model_kwargs is None:
            model_kwargs = {}

        chunk_size = self.streaming_config.chunk_size
        # A partial chunk at the tail is dropped silently, leaving a whole number of chunks plus one frame.
        # The entry points (`_check_video_length` in movieroll/generate.py, the `sample_n_frames` rule in
        # trainer/sft/config.py) reject such lengths up front, so nobody trains on fewer frames unknowingly.
        x1 = x1[:, :, :((x1.shape[2] - 1) // chunk_size * chunk_size + 1)]

        # float32 so the interpolation below does not lose precision
        x1 = x1.to(dtype=th.float32)

        split_results = self.split_latents_to_clean_and_noisy(x1)
        cond_latents, noisy_latents, \
        cond_chunk_sizes, noisy_chunk_sizes, window_sizes, \
        cond_latents_num_frames, noisy_latents_num_frames, noisy_start_index = split_results

        t, noise, noisy_latents = self.sample(
            noisy_latents,
            chunk_sizes=noisy_chunk_sizes,
            window_sizes=window_sizes,
            data_type=data_type
        )

        # the drawn noise and timesteps must agree across the ranks of the SP group
        if parallel_dims is not None:
            (noise, t) = sync_object_for_parallel_training([noise, t], parallel_dims=parallel_dims)

        # the path is planned for the noisy part only
        t, xt, ut = self.path_sampler.plan(t, noise, noisy_latents)

        xt, t_cond = self.merge_latents_to_model_input(
            cond_latents, xt, cond_chunk_sizes=cond_chunk_sizes,
        )

        if t_cond is not None:
            t = th.cat([t_cond, t], dim=1)

        model_kwargs["cond_chunk_sizes"] = cond_chunk_sizes
        model_kwargs["noisy_chunk_sizes"] = noisy_chunk_sizes

        # mutates model_kwargs in place: the per-window text tensor
        self.build_chunk_text(model_kwargs, cond_chunk_sizes, noisy_chunk_sizes, noisy_start_index)

        timestep_txt = self.compute_timestep_txt(t, cond_latents_num_frames, noisy_chunk_sizes)
        if timestep_txt is not None:
            input_timestep_txt = self.get_model_t(timestep_txt)
            model_kwargs["timestep_txt"] = input_timestep_txt
        input_t = self.get_model_t(t)

        if multitask_mask is not None:
            assert mask_type == "t2v", f"because is_streaming_mode, mask_type must be t2v, but got {mask_type}"
            # The 65-channel input contract `img_in.proj` was trained on: [latents(32), zeros(32), zeros(1)].
            # A weight-shape contract, not I2V logic -- the middle block and the mask are always zero here.
            xt_concat = th.zeros_like(xt)
            mask_concat = th.zeros((xt.shape[0], 1, xt.shape[2], xt.shape[3], xt.shape[4]), device=xt.device)
            xt = th.concat([xt, xt_concat, mask_concat], dim=1)

        model_output = model(xt, input_t, **model_kwargs)[0]

        terms = {}

        loss_dict = self.compute_loss(
            model_output, ut,
            noisy_chunk_sizes=noisy_chunk_sizes,
        )
        terms.update(loss_dict)

        terms['t'] = t

        return model_output, terms
