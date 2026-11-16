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

"""Distribution matching losses for the MovieRoll rolling DMD generator."""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Optional, Dict, Any, Sequence
from dataclasses import dataclass

from diffusers.utils.torch_utils import randn_tensor
from movieroll.commons.constants import NUM_TRAIN_TIMESTEPS
from movieroll.pipelines.rolling_dmd_runtime import (
    DenoisingRange,
    RollingMode,
    RollingResult,
    RollingTrainingObjective,
    WindowPrediction,
)
from movieroll.trainer.dmd.score_window import (
    CausalVaeScoreWindowReEncoder,
    ScoreWindowPurpose,
)
from movieroll.utils.communications import bcast_in_group, is_group_rank0
from movieroll.utils.torch_utils import count_failed_ranks


@dataclass
class MovieRollDMDConfig:
    """Configuration of the rolling DMD loss path; every field is required.

    Ranges, including min_step_ratio < max_step_ratio, are checked by validate_stage3_config.
    """

    min_step_ratio: float
    max_step_ratio: float
    real_guidance_scale: float
    timestep_shift: float
    context_noise: float
    denoising_step_list: list[int]
    score_window_latent_frames: int
    # "t2v" = the released recipe: the score models never see the i2v condition. "acs" =
    # anchor-conditioned scoring: the window starting at the i2v anchor is scored the i2v way.
    score_mode: str


@dataclass(frozen=True)
class AnchorCondition:
    """The i2v condition of one scoring window: the anchor latent and its SigLIP tokens.

    The bidirectional model's i2v convention, which both training and inference build: every frame is
    noised with the same t, the concat channels are [x_t | condition latent on frame 0 only | mask 1 on
    frame 0 only], and the condition frame's image tokens come along. Under ACS the condition frame is the
    i2v anchor, i.e. latent frame 0 of the rollout.
    """

    latent: torch.Tensor  # [B, C, 1, H, W], the window's own scaled latents
    vision_states: torch.Tensor  # [B, tokens, dim], SigLIP of the decoded anchor frame

    def model_input(self, x_t: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x_t.shape
        if tuple(self.latent.shape) != (b, c, 1, h, w):
            raise ValueError(
                f"condition latent {tuple(self.latent.shape)} does not match window "
                f"{tuple(x_t.shape)}"
            )
        cond = torch.zeros_like(x_t)
        cond[:, :, :1] = self.latent.to(dtype=x_t.dtype)
        mask = torch.zeros((b, 1, t, h, w), device=x_t.device, dtype=x_t.dtype)
        mask[:, :, :1] = 1.0
        return torch.cat([x_t, cond, mask], dim=1)


class ConcatModelWrapper(torch.nn.Module):
    """Pad a score model's input up to the fixed concat channel count (32 content -> 65)."""

    def __init__(self, model, target_channels):
        super().__init__()
        self.model = model
        self.target_channels = target_channels

    def forward(self, hidden_states, timestep, anchor_condition=None, **kwargs):
        if anchor_condition is not None:
            # ACS: a real i2v input instead of the zero pad, plus the image tokens that go with it.
            hidden_states = anchor_condition.model_input(hidden_states)
            kwargs["vision_states"] = anchor_condition.vision_states

        if hidden_states is not None and hidden_states.shape[1] < self.target_channels:
            padding_channels = self.target_channels - hidden_states.shape[1]
            padding = torch.zeros(
                hidden_states.shape[0], padding_channels, *hidden_states.shape[2:],
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            hidden_states = torch.cat([hidden_states, padding], dim=1)
        elif (
            hidden_states is not None
            and hidden_states.shape[1] > self.target_channels
        ):
            raise ValueError(
                "score-model input has more channels than the fixed concat "
                f"contract: {hidden_states.shape[1]} > {self.target_channels}"
            )

        return self.model(hidden_states=hidden_states, timestep=timestep, **kwargs)


class MovieRollDMD(torch.nn.Module):
    """DMD training module for MovieRoll (generator + real/fake score)."""

    def __init__(
        self,
        generator_pipeline,
        real_score,
        fake_score,
        scheduler,
        config: MovieRollDMDConfig,
        device: torch.device,
        image_encoder=None,
    ):
        """real_score is the frozen teacher, fake_score the trainable critic.

        image_encoder is the frozen SigLIP tower, required by `score_mode: acs` and refused otherwise.
        """
        super().__init__()

        self.generator_pipeline = generator_pipeline
        self.real_score = real_score
        self.fake_score = fake_score
        self.scheduler = scheduler
        self.config = config
        self.device = device
        if (config.score_mode == "acs") != (image_encoder is not None):
            raise ValueError(
                "a SigLIP image encoder is required exactly when score_mode='acs', got "
                f"score_mode={config.score_mode!r}, "
                f"image_encoder={'set' if image_encoder is not None else 'None'}"
            )
        self.image_encoder = image_encoder

        self.min_step = int(config.min_step_ratio * NUM_TRAIN_TIMESTEPS)
        self.max_step = int(config.max_step_ratio * NUM_TRAIN_TIMESTEPS)

        self.real_score.requires_grad_(False)

        self.generator = self.generator_pipeline.transformer
        # set per batch by DMDTrainer from the data spec; read back for the log line
        self.multitask_type = None

        in_channels = 65

        self.fake_score = self._wrap_model_for_concat(self.fake_score, target_channels=in_channels)
        self.real_score = self._wrap_model_for_concat(self.real_score, target_channels=in_channels)

        if not hasattr(generator_pipeline, "execute"):
            raise TypeError("MovieRollDMD requires MovieRollDMDPipeline")
        self.score_window_reencoder = CausalVaeScoreWindowReEncoder(generator_pipeline.vae)

        self._noise_step_counter = 0
        self._base_seed = 42
        self._dp_rank = 0
        self._random_sync_group = None
        self._random_sync_group_id = 0

    def set_dp_rank(self, dp_rank: int, dp_degree: int = 1) -> None:
        """Inject the DP identity for independent source-rank random streams."""

        dp_rank = int(dp_rank)
        dp_degree = int(dp_degree)
        if dp_degree <= 0 or not 0 <= dp_rank < dp_degree:
            raise ValueError(
                f"invalid DP identity: rank={dp_rank}, degree={dp_degree}"
            )
        self._dp_rank = dp_rank

    def set_random_sync_group(self, group) -> None:
        """Inject the SP group used for explicit timestep/noise broadcasts."""

        self._random_sync_group = group
        self._random_sync_group_id = (
            int(dist.get_process_group_ranks(group)[0])
            if dist.is_available() and dist.is_initialized() and group is not None
            else self._dp_rank
        )

    def _make_source_generator(self) -> Optional[torch.Generator]:
        """Advance the reproducible source stream; broadcast provides sync."""

        self._noise_step_counter += 1
        if (
            dist.is_available()
            and dist.is_initialized()
            and self._random_sync_group is not None
            and not is_group_rank0(self._random_sync_group)
        ):
            return None
        seed = (
            self._base_seed
            + self._noise_step_counter
            + self._random_sync_group_id * 1_000_003
        )
        return torch.Generator(device="cpu").manual_seed(seed)

    def _draw_synced_randint(
        self,
        low: int,
        high: int,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Draw only on the SP leader, then broadcast the integer tensor."""

        generator = self._make_source_generator()
        group = self._random_sync_group
        sync_in_group = (
            dist.is_available()
            and dist.is_initialized()
            and group is not None
        )
        if not sync_in_group or is_group_rank0(group):
            if generator is None:
                raise RuntimeError("SP source rank is missing its random generator")
            value = torch.randint(
                low, high, shape, generator=generator, dtype=torch.long
            ).to(self.device)
        else:
            value = torch.empty(shape, device=self.device, dtype=torch.long)
        if sync_in_group:
            bcast_in_group(value, group)
        return value

    def _draw_synced_noise(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Draw only on the SP leader, then broadcast the full noise tensor."""

        generator = self._make_source_generator()
        group = self._random_sync_group
        sync_in_group = (
            dist.is_available()
            and dist.is_initialized()
            and group is not None
        )
        if not sync_in_group or is_group_rank0(group):
            if generator is None:
                raise RuntimeError("SP source rank is missing its random generator")
            noise = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            noise = torch.empty(shape, device=device, dtype=dtype)
        if sync_in_group:
            bcast_in_group(noise, group)
        return noise

    def _require_finite_world(self, tensor: torch.Tensor, *, label: str) -> None:
        """Fail all ranks together after matched backwards, before optimizer."""

        local_bad = int(not bool(torch.isfinite(tensor.detach()).all()))
        failure_count = count_failed_ranks(local_bad, self.device)
        if failure_count:
            location = "local rank" if local_bad else "peer rank"
            raise FloatingPointError(
                f"{label} is non-finite on {failure_count} rank(s); "
                f"detected on {location}"
            )


    def _wrap_model_for_concat(self, model, target_channels):
        return ConcatModelWrapper(model, target_channels)


    def _inverse_shift_timestep(self, shifted_t: float) -> float:
        """Invert the sigma shift back to the raw sigma.
        shift formula: s = shift * t / (1 + (shift-1) * t)
        inverse: t = s / (shift - (shift-1) * s)
        """
        shift = self.config.timestep_shift
        if shift <= 1.0 or shifted_t <= 0:
            return shifted_t
        return shifted_t / (shift - (shift - 1.0) * shifted_t)

    def _sample_and_renoise(
        self,
        clean_latent: torch.Tensor,
        min_t: float,
        batch_size: int,
        num_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Sample a timestep and renoise the clean latent """
        min_step = int(min_t * NUM_TRAIN_TIMESTEPS)
        # The upper bound is always the full range; min_t can itself reach 1.0, so the lower clamp
        # keeps low < high.
        max_step = max(NUM_TRAIN_TIMESTEPS, min_step + 1)

        timestep = self._draw_synced_randint(
            min_step, max_step, (batch_size, 1)
        ).repeat(1, num_frames)

        if self.config.timestep_shift > 1:
            timestep_float = timestep.float()
            timestep_float = self.config.timestep_shift * (timestep_float / NUM_TRAIN_TIMESTEPS) / \
                (1 + (self.config.timestep_shift - 1) * (timestep_float / NUM_TRAIN_TIMESTEPS)) * NUM_TRAIN_TIMESTEPS
            timestep = timestep_float.long()

        timestep = timestep.clamp(self.min_step, self.max_step)

        t = timestep.float() / NUM_TRAIN_TIMESTEPS  # [B, T]
        t_expanded = t.view(batch_size, 1, num_frames, 1, 1)

        noise = self._draw_synced_noise(
            tuple(clean_latent.shape),
            device=clean_latent.device,
            dtype=clean_latent.dtype,
        )

        # Linear-interpolation noising, with x_0 the CLEAN latent: x_t = t * noise + (1-t) * x_0
        # (the reverse-mode path further down in this file writes x0 for the noise instead)
        noisy_latent = t_expanded * noise + (1 - t_expanded) * clean_latent

        return noisy_latent, timestep

    def _compute_score(
        self,
        model: torch.nn.Module,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor,
        byt5_states: Optional[torch.Tensor] = None,
        byt5_mask: Optional[torch.Tensor] = None,
        anchor_condition: Optional[AnchorCondition] = None,
    ) -> torch.Tensor:
        """Run the score model and return its x0 (clean-latent) prediction, not the raw velocity."""
        extra_kwargs = {}
        if byt5_states is not None and byt5_mask is not None:
            extra_kwargs['byt5_text_states'] = byt5_states.to(dtype=torch.bfloat16)
            extra_kwargs['byt5_text_mask'] = byt5_mask
        # Passed only under ACS, so a t2v call is the same call it has always been.
        condition_kwargs = (
            {} if anchor_condition is None else {"anchor_condition": anchor_condition}
        )

        timestep_scalar = timestep[:, 0] if timestep.dim() == 2 else timestep

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            pred = model(
                hidden_states=noisy_latent,
                timestep=timestep_scalar,
                text_states=text_embeds,
                encoder_attention_mask=text_mask,
                extra_kwargs=extra_kwargs,
                **condition_kwargs,
            )

        if isinstance(pred, tuple):
            pred = pred[0]

        return self.scheduler.velocity_to_x0(
            model_output=pred,
            timestep=timestep,
            sample=noisy_latent,
        )

    def _compute_kl_grad(
        self,
        noisy_latent: torch.Tensor,
        estimated_clean: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_2: torch.Tensor,
        prompt_mask: torch.Tensor,
        negative_embeds: torch.Tensor,
        negative_embeds_2: torch.Tensor,
        negative_mask: torch.Tensor,
        byt5_text_states: Optional[torch.Tensor] = None,
        byt5_text_mask: Optional[torch.Tensor] = None,
        negative_byt5_text_states: Optional[torch.Tensor] = None,
        negative_byt5_text_mask: Optional[torch.Tensor] = None,
        anchor_condition: Optional[AnchorCondition] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute the original DMD direction: fake conditional minus real CFG.

        All three calls share one `anchor_condition`: fake and real must be conditioned alike, or the
        gradient picks up a "did it see the anchor" term. The unconditional branch of the CFG therefore
        drops only the text and keeps the image, which is also how this model is sampled.
        """
        alpha = self.config.real_guidance_scale
        with torch.no_grad():
            s_fake_cond = self._compute_score(
                self.fake_score, noisy_latent, timestep,
                prompt_embeds, prompt_mask,
                byt5_text_states, byt5_text_mask,
                anchor_condition=anchor_condition,
            )
            s_real_cond = self._compute_score(
                self.real_score, noisy_latent, timestep,
                prompt_embeds, prompt_mask,
                byt5_text_states, byt5_text_mask,
                anchor_condition=anchor_condition,
            )
            s_real_uncond = self._compute_score(
                self.real_score, noisy_latent, timestep,
                negative_embeds, negative_mask,
                negative_byt5_text_states, negative_byt5_text_mask,
                anchor_condition=anchor_condition,
            )
            s_real_cfg = s_real_uncond + alpha * (s_real_cond - s_real_uncond)
            grad = s_fake_cond - s_real_cfg
            log_dict = {"timestep": timestep.float().mean().detach()}

        with torch.no_grad():
            p_real = estimated_clean - s_real_cfg
        normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
        normalizer = normalizer.clamp_min(1e-4)
        grad = grad / normalizer

        # No per-window isfinite all-reduce here: the caller checks every scalar loss at once, after all
        # ranks have done the same number of backwards.

        log_dict["dmd_gradient_norm"] = torch.mean(torch.abs(grad)).detach()

        return grad, log_dict

    def _run_stage3_rollout(
        self,
        *,
        mode,
        prompts: list,
        num_frames: int,
        height: int,
        width: int,
        event_prompts: Optional[list] = None,
        event_timestamps: Optional[list] = None,
        dmd_backward_handler=None,
    ):
        if not isinstance(mode, RollingMode):
            raise TypeError(f"mode must be RollingMode, got {type(mode)!r}")
        video_length = (int(num_frames) - 1) * 4 + 1
        teacher_unconditional_condition = None
        if mode is RollingMode.GENERATOR_DMD_TRAINING:
            teacher_unconditional_condition = (
                self.generator_pipeline.encode_text_condition([""] * len(prompts))
            )

        kwargs = dict(
            prompt=prompts,
            height=height,
            width=width,
            video_length=video_length,
            denoising_step_list=list(self.config.denoising_step_list),
            context_noise=float(self.config.context_noise),
            multitask_type=self.multitask_type,
            event_prompts=event_prompts,
            event_timestamps=event_timestamps,
            generator=self._make_source_generator(),
            teacher_unconditional_condition=teacher_unconditional_condition,
        )
        if (self.multitask_type or "t2v") == "i2v":
            return self.generator_pipeline.execute_i2v_self_anchored(
                mode=mode,
                dmd_backward_handler=dmd_backward_handler,
                **kwargs,
            )
        return self.generator_pipeline.execute(
            mode=mode,
            dmd_backward_handler=dmd_backward_handler,
            **kwargs,
        )

    def _compute_dmd_loss_core(
        self,
        generated_video: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_2: torch.Tensor,
        prompt_mask: torch.Tensor,
        negative_embeds: torch.Tensor,
        negative_embeds_2: torch.Tensor,
        negative_mask: torch.Tensor,
        byt5_positive: torch.Tensor,
        byt5_mask_positive: torch.Tensor,
        byt5_negative: torch.Tensor,
        byt5_mask_negative: torch.Tensor,
        denoised_timestep_to: float,
        supervision_mask: Optional[torch.Tensor] = None,
        anchor_condition: Optional[AnchorCondition] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size = generated_video.shape[0]
        num_frames = generated_video.shape[2]

        # denoised_timestep_to is a shifted timestep and must be inverse-shifted back to the raw value, because
        # _sample_and_renoise applies a shift again internally.
        t_denoised = self._inverse_shift_timestep(
            denoised_timestep_to / NUM_TRAIN_TIMESTEPS
        )
        with torch.no_grad():
            min_t = t_denoised
            noisy_latent_dm, timestep_dm = self._sample_and_renoise(
                generated_video, min_t=min_t,
                batch_size=batch_size, num_frames=num_frames,
            )

        grad, log_dict = self._compute_kl_grad(
            noisy_latent=noisy_latent_dm.detach(),
            estimated_clean=generated_video.detach(),
            timestep=timestep_dm,
            prompt_embeds=prompt_embeds,
            prompt_embeds_2=prompt_embeds_2,
            prompt_mask=prompt_mask,
            negative_embeds=negative_embeds,
            negative_embeds_2=negative_embeds_2,
            negative_mask=negative_mask,
            byt5_text_states=byt5_positive,
            byt5_text_mask=byt5_mask_positive,
            negative_byt5_text_states=byt5_negative,
            negative_byt5_text_mask=byt5_mask_negative,
            anchor_condition=anchor_condition,
        )

        target = (generated_video.double() - grad.double()).detach()

        # The loss covers every latent channel: the generator's in/out_channels are both 32, so there are
        # no extra channels to exclude from the mean.
        gv = generated_video
        tgt = target

        if supervision_mask is not None:
            mask = self._align_supervision_mask(supervision_mask, gv)
            # Must "boolean-index first, then mse", never "multiply by the mask then divide by the
            # count": the two agree on finite values, but the latter keeps the masked-out frames in
            # the graph, where a single inf/nan becomes 0*inf=nan and poisons the whole loss and its
            # gradient.
            sel = gv.double()[mask]
            if sel.numel() == 0:
                # F.mse_loss on an empty selection returns NaN, so fail immediately.
                raise RuntimeError(
                    "DMD supervision_mask is all False: no frame is supervised and the loss "
                    "would be nan; check how the upstream mask is built.")
            loss = 0.5 * F.mse_loss(sel, tgt[mask], reduction="mean")
        else:
            loss = 0.5 * F.mse_loss(gv.double(), tgt, reduction="mean")

        return loss, log_dict

    @staticmethod
    def _align_supervision_mask(
        supervision_mask: torch.Tensor, ref: torch.Tensor
    ) -> torch.Tensor:
        """Align the supervision mask to ref's [B,C,T,H,W] layout and broadcast it.

        Stage3 produces [B,C,T,H,W]. A one-channel mask may be broadcast over C.
        """
        C = ref.shape[1]
        m = supervision_mask
        if m.shape == ref.shape:
            return m
        if m.dim() != 5 or m.shape[0] != ref.shape[0] or m.shape[2:] != ref.shape[2:]:
            raise ValueError(
                f"supervision mask shape {tuple(m.shape)} does not align with "
                f"prediction {tuple(ref.shape)}"
            )
        if m.shape[1] not in (1, C):
            raise ValueError(f"supervision mask channels must be 1 or {C}, got {m.shape[1]}")
        return m.expand_as(ref) if m.shape[1] == 1 else m

    def _compute_critic_loss_core(
        self,
        generated_video: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        byt5_text_states: torch.Tensor,
        byt5_text_mask: torch.Tensor,
        denoised_timestep_to: float,
        anchor_condition: Optional[AnchorCondition] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute full-window flow-matching loss for the fake-score model."""
        batch_size = generated_video.shape[0]

        model_kwargs = {
            'text_states': prompt_embeds.to(dtype=torch.bfloat16) if prompt_embeds is not None else None,
            'encoder_attention_mask': prompt_mask,
            'return_dict': False,
            'extra_kwargs': {
                'byt5_text_states': byt5_text_states.to(dtype=torch.bfloat16) if byt5_text_states is not None else None,
                'byt5_text_mask': byt5_text_mask,
            },
        }
        if anchor_condition is not None:
            # The critic must see the same condition the DMD gradient's fake branch saw.
            model_kwargs['anchor_condition'] = anchor_condition

        # denoised_timestep_to is a shifted timestep and must be inverse-shifted back to the raw
        # value, because a shift is applied again below. The lower bound follows the denoised range;
        # the upper bound is always the full range.
        min_timestep = int(self._inverse_shift_timestep(
            denoised_timestep_to / NUM_TRAIN_TIMESTEPS
        ) * NUM_TRAIN_TIMESTEPS)
        max_timestep = NUM_TRAIN_TIMESTEPS

        # Keep low < high in randint: under few-step rolling, when the exit chunk is close to clean,
        # inverse_shift(denoised_to) can approach max and make min >= max.
        min_timestep = max(0, min(min_timestep, NUM_TRAIN_TIMESTEPS - 1))
        if max_timestep <= min_timestep:
            max_timestep = min_timestep + 1

        critic_timestep = self._draw_synced_randint(
            min_timestep, max_timestep, (batch_size,)
        )

        # t in [0, 1]: sample first, then shift.
        t = critic_timestep.float() / NUM_TRAIN_TIMESTEPS
        # Same shift formula Transport.sample applies: t' = s*t / (1 + (s-1)*t). It is applied here
        # explicitly because the timestep is drawn here rather than inside the transport.
        shift = self.config.timestep_shift
        if shift > 1.0:
            t = (shift * t) / (1.0 + (shift - 1.0) * t)
        # Clamp to [min_step, max_step] exactly as `_sample_and_renoise` does on the DMD side, so the critic
        # never sees sigma close to 1 (pure noise, nothing to learn from) or close to 0 (the velocity
        # target explodes numerically).
        t = t.clamp(self.min_step / NUM_TRAIN_TIMESTEPS,
                    self.max_step / NUM_TRAIN_TIMESTEPS)

        # Only the SP leader samples; the full noise tensor is then broadcast explicitly.
        critic_noise = self._draw_synced_noise(
            tuple(generated_video.shape),
            device=generated_video.device,
            dtype=generated_video.dtype,
        )

        # Linear flow matching path (reverse mode): xt = (1-t)*x1 + t*x0
        t_expanded = t.view(batch_size, 1, 1, 1, 1).to(device=generated_video.device, dtype=generated_video.dtype)
        xt = (1.0 - t_expanded) * generated_video + t_expanded * critic_noise
        # Target velocity: ut = x0 - x1 = noise - clean
        ut = critic_noise - generated_video

        # Model timestep: t * 1000 (large = high noise, in line with sigma). Keep fp32: bf16 has only 8
        # significant bits, so the ulp in [512,1024) is 4 and 555.56 quantises to 556, which would make the
        # sigma reported to the model differ from the sigma used for noising. `_compute_score` does not cast
        # on the real/fake score path either; the two paths agree.
        model_t = (t * NUM_TRAIN_TIMESTEPS).to(device=self.device, dtype=torch.float32)

        # fake_score forward (ConcatModelWrapper pads 32ch -> 65ch automatically)
        model_output = self.fake_score(xt.to(dtype=torch.bfloat16), model_t, **model_kwargs)[0]

        loss = F.mse_loss(model_output.float(), ut.float(), reduction="mean")

        return loss, {
            "critic_timestep": critic_timestep.float().mean().detach(),
            "critic_timestep_shifted": (t * NUM_TRAIN_TIMESTEPS).mean().detach(),
        }

    def _anchor_condition(
        self, *, score_window: torch.Tensor, score_start: int, cache: dict
    ) -> Optional[AnchorCondition]:
        """The ACS condition of this scoring window, or None when it is scored as t2v.

        Only windows whose scoring crop starts at the i2v anchor are conditioned (`score_start == 0`, which
        is the first window and, depending on the rollout's start phase, the second). Later windows would
        have to be conditioned on generated frames, which lets the teacher follow their drift; they keep the
        released t2v call, zero-padded and without image tokens.

        The condition frame is the window's own latent frame 0. In an i2v rollout that is the anchor: a
        genuine single-frame latent, and `score_start == 0` is also the re-encoder's identity case, so there
        is nothing to re-encode. One decode plus one SigLIP call per rollout, broadcast so the SP group
        conditions on bit-identical tokens.
        """
        if self.config.score_mode != "acs":
            return None
        if (self.multitask_type or "t2v") != "i2v" or int(score_start) != 0:
            return None
        cached = cache.get("anchor")
        if cached is not None:
            return cached

        # clone: the cache keeps this one frame, not the whole window tensor
        latent = score_window[:, :, :1].detach().clone()
        with torch.no_grad():
            pixels = self.score_window_reencoder.decode_condition_frame(latent)
            # SigLIP runs in its own fp16 outside the trainer's bf16 autocast, as in i2v training.
            with torch.autocast(device_type=self.device.type, enabled=False):
                vision_states = self.image_encoder.encode(pixels)
        vision_states = vision_states.to(
            device=self.device, dtype=torch.bfloat16
        ).contiguous()
        group = self._random_sync_group
        if dist.is_available() and dist.is_initialized() and group is not None:
            bcast_in_group(vision_states, group)

        condition = AnchorCondition(latent=latent, vision_states=vision_states)
        cache["anchor"] = condition
        return condition

    def _assert_window_count_uniform(self, k: int):
        """Verify WORLD-wide K equality before the first per-window backward."""
        if k <= 0:
            raise RuntimeError(f"per-window training requires K > 0, got {k}")
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return
        k_eff = int(k)
        probe = torch.tensor([k_eff, -k_eff], device=self.device, dtype=torch.long)
        dist.all_reduce(probe, op=dist.ReduceOp.MAX)   # WORLD
        k_max, k_min = int(probe[0].item()), -int(probe[1].item())
        if k_max != k_min:
            raise RuntimeError(
                f"per-window backward count differs across ranks: local K={k_eff}, "
                f"world min={k_min} max={k_max}. Every window backward triggers the HSDP "
                f"dp_replicate all-reduce (cross-node), so K must be globally identical; "
                "MovieRoll DMD requires a fixed number of selected windows on every rank.")

    def compute_dmd_loss_per_window(
        self,
        prompts: list,
        num_frames: int,
        height: int,
        width: int,
        backward_fn,
        loss_scale: float = 1.0,
        event_prompts: Optional[list] = None,
        event_timestamps: Optional[list] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Backpropagate each selected DMD window before the next window forward (CuLT-DMD).

        `backward_fn` runs per window and releases that window's graph immediately, so a long rollout
        accumulates gradients at the memory cost of one window; `DMDTrainer` steps the optimizer once per
        rollout, with the generator's parameters fixed for the whole rollout.
        """
        state: Dict[str, Any] = {
            "num_windows": None,
            "count": 0,
            "total": 0.0,
            "total_frames": 0,
            "losses": [],
            "loss_tensors": [],
            "merged": {},
            # ACS: the anchor condition, computed once and reused by every anchor window of this rollout
            "condition_cache": {},
        }
        owner = self
        window_len = int(self.config.score_window_latent_frames)

        class Handler:
            def on_rollout_start(self, *, num_windows: int) -> None:
                k = int(num_windows)
                if k <= 0 or state["num_windows"] is not None:
                    raise RuntimeError(f"invalid DMD handler start: K={k}")
                owner._assert_window_count_uniform(k)
                state["num_windows"] = k

            def on_window(
                self,
                *,
                window_index: int,
                expected_num_windows: int,
                predicted_clean_window: torch.Tensor,
                output_start_frame: int,
                history_chunks: tuple[torch.Tensor, ...],
                previous_window_predictions: tuple[WindowPrediction, ...],
                prompt_condition,
                teacher_unconditional_condition,
                denoising_range: DenoisingRange,
            ) -> None:
                k = state["num_windows"]
                if (
                    k is None
                    or expected_num_windows != k
                    or not 0 <= window_index < k
                    or window_index != state["count"]
                ):
                    raise RuntimeError(
                        "DMD handler/window ordering mismatch: "
                        f"K={k}, expected={expected_num_windows}, "
                        f"index={window_index}, completed={state['count']}"
                    )
                if not predicted_clean_window.requires_grad:
                    raise RuntimeError(
                        "generator DMD callback received a detached prediction"
                    )
                if len(previous_window_predictions) != window_index or any(
                    prediction.predicted_clean_latents.requires_grad
                    or not prediction.dmd_backward_completed
                    for prediction in previous_window_predictions
                ):
                    raise RuntimeError(
                        "previous_window_predictions must contain exactly the "
                        "already-backpropagated detached windows"
                    )
                history_frames = sum(
                    int(chunk.shape[2]) for chunk in history_chunks
                )
                if history_frames != output_start_frame:
                    raise RuntimeError(
                        "DMD output/history coordinates diverged: "
                        f"output_start_frame={output_start_frame}, "
                        f"history_frames={history_frames}"
                    )
                teacher_window, supervision_mask = owner._assemble_teacher_window(
                    history_chunks=history_chunks,
                    previous_window_predictions=previous_window_predictions,
                    output_start_frame=output_start_frame,
                    predicted_clean_window=predicted_clean_window,
                    window_len=window_len,
                )
                history_needed = (
                    int(teacher_window.shape[2])
                    - int(predicted_clean_window.shape[2])
                )
                score_start = int(output_start_frame) - history_needed
                reencoded = owner.score_window_reencoder.reencode(
                    purpose=ScoreWindowPurpose.GENERATOR_DMD,
                    raw_score_window=teacher_window,
                    score_start=score_start,
                    supervision_mask=supervision_mask,
                )
                pe, pm, pe2, extra = prompt_condition
                neg_pe, neg_mask, neg_pe2, neg_extra = teacher_unconditional_condition
                anchor_condition = owner._anchor_condition(
                    score_window=reencoded.latents,
                    score_start=score_start,
                    cache=state["condition_cache"],
                )
                loss, log_dict = owner._compute_dmd_loss_core(
                    generated_video=reencoded.latents,
                    prompt_embeds=pe,
                    prompt_embeds_2=pe2,
                    prompt_mask=pm,
                    negative_embeds=neg_pe,
                    negative_embeds_2=neg_pe2,
                    negative_mask=neg_mask,
                    byt5_positive=extra.get("byt5_text_states"),
                    byt5_mask_positive=extra.get("byt5_text_mask"),
                    byt5_negative=neg_extra.get("byt5_text_states"),
                    byt5_mask_negative=neg_extra.get("byt5_text_mask"),
                    denoised_timestep_to=denoising_range.end_timestep,
                    supervision_mask=reencoded.supervision_mask,
                    anchor_condition=anchor_condition,
                )
                scaled_loss = loss * (loss_scale / k)
                backward_fn(scaled_loss)
                state["loss_tensors"].append(loss.detach().float())
                loss_value = float(loss.detach().float().item())
                state["count"] += 1
                state["total"] += loss_value
                state["total_frames"] += int(predicted_clean_window.shape[2])
                state["losses"].append(loss_value)
                merged = state["merged"]
                for key, value in log_dict.items():
                    if isinstance(value, (int, float)):
                        merged[key] = merged.get(key, 0.0) + float(value) / k
                    elif torch.is_tensor(value) and value.numel() == 1:
                        merged[key] = merged.get(key, 0.0) + float(value.item()) / k
                    else:
                        merged[key] = value.detach() if torch.is_tensor(value) else value
                del loss, log_dict, teacher_window, supervision_mask, scaled_loss, reencoded

                # Extend the re-encoder's loss-side latent history, which must stay contiguous.
                reencoder = owner.score_window_reencoder
                appended = reencoder.appended_latent_frames
                start = int(output_start_frame)
                if appended == 0 and start > 0:
                    if previous_window_predictions:
                        raise RuntimeError(
                            "loss-side VAE cannot bootstrap after previous "
                            "selected windows already exist"
                        )
                    if not history_chunks:
                        raise RuntimeError(
                            "loss-side VAE bootstrap is missing initial history"
                        )
                    initial_history = torch.cat(
                        [chunk.detach() for chunk in history_chunks], dim=2
                    )
                    if int(initial_history.shape[2]) != start:
                        raise RuntimeError(
                            "loss-side VAE bootstrap/history mismatch: "
                            f"history={initial_history.shape[2]}, start={start}"
                        )
                    reencoder.append_latents(
                        latents=initial_history,
                        global_start=0,
                    )
                    del initial_history
                    appended = reencoder.appended_latent_frames
                if appended != start:
                    raise RuntimeError(
                        "selected score windows no longer form contiguous "
                        "loss-side latent history: "
                        f"appended={appended}, current_start={start}"
                    )
                reencoder.append_latents(
                    latents=predicted_clean_window.detach(),
                    global_start=start,
                )

        with self.score_window_reencoder.rollout():
            result = self._run_stage3_rollout(
                mode=RollingMode.GENERATOR_DMD_TRAINING,
                prompts=prompts,
                num_frames=num_frames,
                height=height,
                width=width,
                event_prompts=event_prompts,
                event_timestamps=event_timestamps,
                dmd_backward_handler=Handler(),
            )
        k = state["num_windows"]
        count = int(state["count"] )
        if k is None or count != k or len(result.window_predictions) != k:
            raise RuntimeError(
                "immediate DMD backward count mismatch: "
                f"K={k}, callback={count}, predictions={len(result.window_predictions)}"
            )
        self._require_finite_world(
            torch.stack(state["loss_tensors"]),
            label="DMD per-window losses",
        )
        merged = state["merged"]
        merged["num_grad_windows_supervised"] = k
        merged["dmd_windows_total_frames"] = int(state["total_frames"] )
        merged["dmd_window_losses"] = tuple(state["losses"])
        merged["dmd_start_phase"] = result.start_phase
        merged["dmd_selected_rounds"] = result.selected_round_indices
        merged["rolling_objective_uniform"] = float(
            getattr(
                result,
                "training_objective",
                RollingTrainingObjective.ROLLING_DMD,
            )
            is RollingTrainingObjective.UNIFORM_TIMESTEP_DMD
        )
        merged["uniform_step_index"] = (
            -1
            if getattr(result, "uniform_step_index", None) is None
            else int(result.uniform_step_index)
        )
        merged["uniform_shifted_timestep"] = (
            -1.0
            if getattr(result, "uniform_step_index", None) is None
            else float(result.window_predictions[0].denoising_range.start_timestep)
        )
        merged.update(self.score_window_reencoder.stats())
        return float(state["total"]) / k, merged

    def _assemble_teacher_window(
        self,
        *,
        history_chunks: Sequence[torch.Tensor],
        previous_window_predictions: Sequence[WindowPrediction],
        output_start_frame: int,
        predicted_clean_window: torch.Tensor,
        window_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build detached history plus the current differentiable prediction."""
        start = int(output_start_frame)
        current_frames = int(predicted_clean_window.shape[2])
        history_needed = min(max(0, int(window_len) - current_frames), start)
        if history_needed == 0:
            return predicted_clean_window, torch.ones_like(
                predicted_clean_window, dtype=torch.bool
            )
        pieces = []
        remaining = history_needed
        for chunk in reversed(history_chunks):
            if remaining <= 0:
                break
            chunk_frames = int(chunk.shape[2])
            take = min(remaining, chunk_frames)
            pieces.append(chunk[:, :, chunk_frames - take:chunk_frames])
            remaining -= take
        if remaining:
            available = sum(int(chunk.shape[2]) for chunk in history_chunks)
            raise RuntimeError(
                f"DMD history underflow: start={start}, need={history_needed}, have={available}"
            )
        history = torch.cat(list(reversed(pieces)), dim=2).detach().clone()
        history_start = start - history_needed
        for prediction in previous_window_predictions:
            previous_start = prediction.output_start_frame
            previous_window = prediction.predicted_clean_latents
            previous_end = previous_start + int(previous_window.shape[2])
            overlap_start = max(history_start, previous_start)
            overlap_end = min(start, previous_end)
            if overlap_start < overlap_end:
                history[:, :, overlap_start - history_start:overlap_end - history_start] = (
                    previous_window[
                        :, :,
                        overlap_start - previous_start:overlap_end - previous_start,
                    ].detach()
                )
        teacher_window = torch.cat((history, predicted_clean_window), dim=2)
        mask = torch.zeros_like(teacher_window, dtype=torch.bool)
        mask[:, :, history_needed:] = True
        return teacher_window, mask

    def compute_critic_loss_per_window(
        self,
        prompts: list,
        num_frames: int,
        height: int,
        width: int,
        backward_fn,
        loss_scale: float = 1.0,
        event_prompts: Optional[list] = None,
        event_timestamps: Optional[list] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Train fake_score on every selected window from a no-grad generator rollout."""
        with torch.no_grad():
            result = self._run_stage3_rollout(
                mode=RollingMode.FAKE_SCORE_TRAINING,
                prompts=prompts,
                num_frames=num_frames,
                height=height,
                width=width,
                event_prompts=event_prompts,
                event_timestamps=event_timestamps,
            )
        windows = result.window_predictions
        self._assert_window_count_uniform(len(windows))
        if not windows:
            raise RuntimeError("fake-score rollout returned no selected windows")
        window_len = int(self.config.score_window_latent_frames)
        required_score_starts = set()
        for prediction in windows:
            start = int(prediction.output_start_frame)
            current_frames = int(prediction.predicted_clean_latents.shape[2])
            history_needed = min(max(0, window_len - current_frames), start)
            score_start = start - history_needed
            if score_start > 0:
                required_score_starts.add(score_start)

        total = 0.0
        window_losses: list[float] = []
        loss_tensors: list[torch.Tensor] = []
        merged: Dict[str, Any] = {}
        condition_cache: dict = {}  # ACS, as on the generator side
        with self.score_window_reencoder.rollout(
            retain_score_starts=required_score_starts
        ):
            self.score_window_reencoder.append_latents(
                latents=result.generated_latents.detach(),
                global_start=0,
            )

            for prediction in windows:
                teacher_window = self._slice_prediction_from_result(
                    result=result, prediction=prediction, window_len=window_len
                )
                history_needed = (
                    int(teacher_window.shape[2])
                    - int(prediction.predicted_clean_latents.shape[2])
                )
                score_start = int(prediction.output_start_frame) - history_needed
                reencoded = self.score_window_reencoder.reencode(
                    purpose=ScoreWindowPurpose.CRITIC,
                    raw_score_window=teacher_window,
                    score_start=score_start,
                    supervision_mask=None,
                )
                pe, pm, _, extra = result.event_conditions[prediction.event_index]
                loss, log_dict = self._compute_critic_loss_core(
                    generated_video=reencoded.latents,
                    prompt_embeds=pe,
                    prompt_mask=pm,
                    byt5_text_states=extra.get("byt5_text_states"),
                    byt5_text_mask=extra.get("byt5_text_mask"),
                    denoised_timestep_to=prediction.denoising_range.end_timestep,
                    anchor_condition=self._anchor_condition(
                        score_window=reencoded.latents,
                        score_start=score_start,
                        cache=condition_cache,
                    ),
                )
                backward_fn(loss * (loss_scale / len(windows)))
                loss_tensors.append(loss.detach().float())
                loss_value = float(loss.detach().float().item())
                total += loss_value
                window_losses.append(loss_value)
                for key, value in log_dict.items():
                    if isinstance(value, (int, float)):
                        merged[key] = (
                            merged.get(key, 0.0) + float(value) / len(windows)
                        )
                    elif torch.is_tensor(value) and value.numel() == 1:
                        merged[key] = (
                            merged.get(key, 0.0)
                            + float(value.item()) / len(windows)
                        )
                    else:
                        merged[key] = (
                            value.detach() if torch.is_tensor(value) else value
                        )
                del loss, log_dict, teacher_window, reencoded
        self._require_finite_world(
            torch.stack(loss_tensors),
            label="fake-score per-window losses",
        )
        merged["num_critic_windows_supervised"] = len(windows)
        merged["critic_window_losses"] = tuple(window_losses)
        merged["critic_start_phase"] = result.start_phase
        merged["critic_selected_rounds"] = result.selected_round_indices
        merged["critic_objective_uniform"] = float(
            getattr(
                result,
                "training_objective",
                RollingTrainingObjective.ROLLING_DMD,
            )
            is RollingTrainingObjective.UNIFORM_TIMESTEP_DMD
        )
        merged["critic_uniform_step_index"] = (
            -1
            if getattr(result, "uniform_step_index", None) is None
            else int(result.uniform_step_index)
        )
        merged["critic_uniform_shifted_timestep"] = (
            -1.0
            if getattr(result, "uniform_step_index", None) is None
            else float(result.window_predictions[0].denoising_range.start_timestep)
        )
        merged.update(self.score_window_reencoder.stats())
        return total / len(windows), merged

    @staticmethod
    def _slice_prediction_from_result(
        *, result: RollingResult, prediction: WindowPrediction, window_len: int
    ) -> torch.Tensor:
        """Build a native-size score-model window from a detached rollout result."""
        start = prediction.output_start_frame
        window = prediction.predicted_clean_latents
        history_needed = min(max(0, int(window_len) - int(window.shape[2])), start)
        if history_needed == 0:
            return window
        history = result.generated_latents[
            :, :, start - history_needed:start
        ].detach()
        return torch.cat((history, window), dim=2)
