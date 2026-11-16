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

"""MovieRoll inference entry point (`--sampling-mode base` / `distilled`).

The two modes differ only in the denoising protocol: `base` runs the 50-step CFG ladder through
`MovieRollStreamingPipeline.__call__`, `distilled` runs the few-step DMD rollout through
`MovieRollDMDPipeline.execute(mode=SAMPLING)`.

Both do text-to-video and image-to-video. For i2v the input image is VAE-encoded into one latent frame that
becomes the first output chunk: it is written into the KV cache at timestep 0 and every generated chunk
attends to it (chunk layout [1, chunk_size, ...]).

Sampling hyper-parameters come from the model folder's `config.json` (`--model`), or from the built-in profiles in
`movieroll/commons/profiles.py` for a bare `--dit-weight`; the command line exposes the prompt source (and input
images), the frame count, the resolution, the context noise, the output location, the parallel topology and
precision.
"""

import argparse
import gc
import os
import random

import torch
import torch.distributed as dist
import torchvision
from accelerate.utils import set_seed
from diffusers.utils import check_min_version

from movieroll.commons.constants import (
    BYT5_CKPT,
    BYT5_COLOR,
    BYT5_FONT,
    BYT5_GOOGLE,
    MODEL_TYPE,
    NUM_TRAIN_TIMESTEPS,
)
from movieroll.commons.profiles import (
    I2V_BUCKET_BASE_SIZE,
    I2V_BUCKET_MAX_RATIO,
    I2V_BUCKET_PATCH_SIZE,
    MODEL_WEIGHT_FILE,
    SAMPLING_MODES,
    get_profile,
    load_model_dir,
)
from movieroll.commons.streaming_config import StreamingConfig
from movieroll.models.autoencoders.streaming_vae import decode_latent_chunks_streaming
from movieroll.models.autoencoders.vae_loader import load_vae
from movieroll.models.text_encoders import PROMPT_TEMPLATE
from movieroll.models.text_encoders.byT5 import load_glyph_byT5_v2
from movieroll.models.transformers.model_configs import MODEL_CONFIG
from movieroll.models.transformers.modules.streaming_attention import ATTENTION_BACKENDS, set_attention_backend
from movieroll.pipelines.movieroll_dmd_pipeline import MovieRollDMDPipeline
from movieroll.pipelines.movieroll_pipeline import MovieRollStreamingPipeline
from movieroll.pipelines.rolling_dmd_runtime import RollingMode
from movieroll.schedulers.scheduling_flow_match_streaming import FlowMatchStreamingScheduler
from movieroll.utils.file_utils import save_videos_grid
from movieroll.utils.fsdp import apply_fsdp2, broadcast_from_first_replica, fsdp_device_mesh
from movieroll.utils.image_condition import encode_input_image, open_input_image
from movieroll.utils.load import load_text_encoder, load_transformer
from movieroll.utils.logging import setup_logger
from movieroll.utils.multitask_utils import get_multitask_mask
from movieroll.commons.parallel_states import (
    VAE_PARALLEL_MODES,
    ParallelDims,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    init_nccl_process_group,
    initialize_sequence_parallel_state_v2,
    initialize_vae_parallel_group,
    nccl_info,
)
from movieroll.utils.prompt_csv import read_prompt_csv
from movieroll.utils.torch_utils import PRECISION_TO_TYPE

gc.set_threshold(7000, 100, 100)

check_min_version("0.31.0")

# One VAE group per sp group for 8 GPUs at the default --sp-size 2.
VAE_PARALLEL_RANKS_EXAMPLE = "0,1;2,3;4,5;6,7"


def str_to_bool(value):
    """Parse true/false, 1/0, yes/no, on/off; None (a bare flag with no value) means True."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ('true', '1', 'yes', 'on'):
            return True
        if value in ('false', '0', 'no', 'off'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate long videos with MovieRoll")

    parser.add_argument(
        '--sampling-mode', type=str, default=None, choices=list(SAMPLING_MODES),
        help="base = 50-step CFG chain; distilled = few-step DMD chain. Taken from the config.json of --model; "
             "with --dit-weight it defaults to base."
    )

    # ---- prompt source: exactly one of the three ----
    parser.add_argument(
        '--prompt', type=str, default=None,
        help="A single prompt. Every dp rank generates it and only dp rank 0 writes the video."
    )
    parser.add_argument(
        '--prompt-list', type=str, nargs='+', default=None,
        help="Ordered event prompts of ONE video. With 2 or more entries the prompt is switched per event."
    )
    parser.add_argument(
        '--prompt-csv', type=str, nargs='+', default=None,
        help="CSV batch: columns index, seed and either prompt or event_1..event_N (the video is split evenly "
             "across the events), plus an optional input_image column for image-to-video (relative paths are "
             "taken from the CSV's directory). Optional global_content_prompt / global_quality_prompt columns are "
             "joined before and after every event text, but only when both are present (one alone is ignored) and "
             "never in a prompt-column CSV. Every row of one run must have the same number of events, and either "
             "every row has an input image or none does, or the run is refused before the weights load: each event "
             "is one text encode and each input image one extra transformer forward, collectives over all ranks. "
             "Keep the row count a multiple of the dp degree: rows are padded cyclically, and a padded duplicate "
             "writes the same {index}.mp4 as its original."
    )
    parser.add_argument(
        '--input-image', type=str, default=None,
        help="Input image of a --prompt / --prompt-list run (image-to-video): it is resized to cover "
             "--height x --width, centre-cropped and becomes the first frame. CSV runs use the input_image "
             "column instead. With --dit-weight and no --height / --width, the resolution follows the image's "
             "aspect ratio instead."
    )

    parser.add_argument(
        '--video-length', type=int, default=721,
        help="Number of pixel frames. Must be 4n+1 and leave a whole number of chunks; distilled and every "
             "image-to-video run additionally need at least one full first chunk (17 frames with the released "
             "chunk size)."
    )
    parser.add_argument('--height', type=int, default=None,
                        help="Output height in pixels, a multiple of 16; defaults to the model's 480 (config.json "
                             "or the built-in profile), with --model for image-to-video too. The released "
                             "distilled weights were trained at 480x864 and 864x480.")
    parser.add_argument('--width', type=int, default=None,
                        help="Output width in pixels, a multiple of 16; defaults to the profile's 864.")
    parser.add_argument('--num-videos-per-prompt', type=int, default=1,
                        help="Videos to generate per prompt. Each one gets its own noise: the prompt is run "
                             "this many times against one shared generator, so every pass consumes the next "
                             "draws and the videos differ. They are written as <name>_v0.mp4 ... "
                             "<name>_v{N-1}.mp4; at 1 the filename is unchanged. Cost is wall time, not "
                             "memory -- the videos are generated one after another, not as a batch. Batching "
                             "them inside the pipeline is not an option: the byT5 embeddings are not "
                             "replicated per video, so reorder_txt_token would index past the end of the byT5 "
                             "batch, and image-to-video keeps its conditioning latent at batch 1 while every "
                             "generated chunk would be batch N, which trips the KV cache's batch check.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Noise seed of --prompt / --prompt-list runs; defaults to --global-seed.")
    parser.add_argument('--global-seed', type=int, default=930,
                        help="Per-process seed base; each process uses global_seed + rank // sp_size.")
    parser.add_argument('--context-noise', type=float, default=None, metavar='SIGMA',
                        help="Sigma of the noise added to the clean history before it enters the KV cache, in "
                             "[0, 1); 0 turns it off. Defaults to the model's config.json (0.15 for distilled "
                             "text-to-video, 0 for distilled image-to-video, 0.05 for base).")
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--output-type', type=str, default='video', choices=['video', 'pt', 'latent'],
                        help="video = mp4; pt = decoded tensor as .pt; latent = undecoded latents as .pt "
                             "(distilled only)")

    # ---- weights ----
    parser.add_argument('--model', type=str, default=None,
                        help="A MovieRoll model directory (e.g. ckpts/movieroll/distilled/t2v) holding model.safetensors "
                             "and the config.json it is sampled with: the sampling mode and every sampling setting "
                             "come from that file, and --height / --width / --context-noise still override it.")
    parser.add_argument('--dit-weight', type=str, default=None,
                        help="A bare transformer checkpoint (.safetensors), instead of --model; its settings are then "
                             "the built-in profile of --sampling-mode.")
    parser.add_argument('--vae-path', type=str, default='ckpts/vae')
    parser.add_argument('--text-encoder-path', type=str, default='ckpts/text_encoder/llm')
    parser.add_argument('--byt5-google-path', type=str, default=BYT5_GOOGLE)
    parser.add_argument('--byt5-ckpt-path', type=str, default=BYT5_CKPT)
    parser.add_argument('--multilingual-prompt-format-font-path', type=str, default=BYT5_FONT)
    parser.add_argument('--multilingual-prompt-format-color-path', type=str, default=BYT5_COLOR)

    # ---- parallel topology ----
    parser.add_argument('--sp-size', type=int, default=2, help="Sequence-parallel group size.")
    parser.add_argument(
        '--shard-world', type=int, default=8,
        help="FSDP shard-group size (dp_shard * sp). Must divide world_size and be divisible by --sp-size. "
             "Ignored without FSDP."
    )
    parser.add_argument(
        '--fsdp-sharding-strategy', type=str, default='hybrid', choices=['hybrid', 'full', 'none'],
        help="hybrid / full: shard the transformer and the text encoder with FSDP2. none: no FSDP, every rank holds "
             "complete weights (a 96 GB GPU fits them). A single-process run never uses FSDP, whatever this says."
    )
    parser.add_argument('--use-cpu-offload', type=str_to_bool, nargs='?', const=True, default=False,
                        help="FSDP2 CPU offload of the transformer. Refused: it does not work for inference, because "
                             "the pipelines take their device from the offloaded (cpu) parameters, so cpu and cuda "
                             "tensors meet (byT5 glyph encode, noise generator or sp broadcast, transformer forward) "
                             "and the run would fail after the weights load.")
    parser.add_argument('--vae-parallel-mode', type=str, default='none', choices=list(VAE_PARALLEL_MODES),
                        help="PS-VAE: row-shard the streaming VAE decode across a group of ranks.")
    parser.add_argument('--vae-parallel-ranks', type=str, default=None,
                        help="Rank groups for --vae-parallel-mode custom: ';' between groups, ',' within a group, "
                             f"e.g. '{VAE_PARALLEL_RANKS_EXAMPLE}' for 8 GPUs at the default --sp-size 2. Every "
                             "group must lie inside one sp group, since its ranks decode the same latents.")

    # ---- attention kernel ----
    parser.add_argument('--attention-backend', type=str, default='fa3', choices=list(ATTENTION_BACKENDS),
                        help="Kernel of the transformer's streaming attention: fa3 (FlashAttention-3, the default) or "
                             "sage (SageAttention 2, INT8 / FP8 attention: faster, not bit-equal to fa3; needs the "
                             "`sageattention` package, see the README).")

    # ---- precision ----
    parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp16'],
                        help="autocast dtype of the transformer forward. The released chains run in bf16; fp16 "
                             "passes the kernels but is untested end to end. fp32 is not offered because it "
                             "cannot run: it turns the autocast off, and the streaming FlashAttention kernels "
                             "then reject the fp32 q/k/v (`FlashAttention only supports fp16, bf16, and "
                             "fp8_e4m3 data type`). bf16 masters are not an escape -- the token refiner's "
                             "promotion dies first on `mat1 and mat2 must have the same dtype`.")
    parser.add_argument('--master-weight-type', type=str, default=None, choices=['fp32', 'bf16'],
                        help="Parameter dtype of the loaded transformer. Default: bf16 without FSDP, fp32 with FSDP. "
                             "Inference has no optimizer to keep fp32 masters for and under bf16 autocast the matmuls "
                             "run in bf16 either way, so bf16 halves the weight memory (17 GB against 33 GB) and is what "
                             "lets a single 96 GB GPU hold the pipeline. FSDP2 needs one dtype per parameter group, and "
                             "the transformer builds a few small parameters (byt5_in.*, final_layer.linear.*, "
                             "cond_type_embedding) without the requested dtype, so bf16 masters cannot be sharded: "
                             "with FSDP the default is fp32 (the released multi-GPU runs), and asking for bf16 there "
                             "is an error. fp32 is also the training master-weight setting.")
    parser.add_argument('--vae-precision', type=str, default='fp16', choices=['fp32', 'bf16', 'fp16'])
    parser.add_argument('--text-encoder-precision', type=str, default='fp16', choices=['fp32', 'bf16', 'fp16'])

    args = parser.parse_args(argv)

    sources = [args.prompt is not None, args.prompt_list is not None, args.prompt_csv is not None]
    if sum(sources) != 1:
        raise ValueError("exactly one of --prompt / --prompt-list / --prompt-csv is required")
    if args.use_cpu_offload:
        raise ValueError("--use-cpu-offload does not work for inference (see its help); run without it")
    if args.prompt_csv is not None and args.input_image is not None:
        raise ValueError("--input-image is for --prompt / --prompt-list; a CSV run uses the input_image column")
    if (args.model is None) == (args.dit_weight is None):
        raise ValueError("exactly one of --model / --dit-weight is required")
    if args.model is not None:
        settings = load_model_dir(args.model)
        if args.sampling_mode not in (None, settings["sampling_mode"]):
            raise ValueError(f"--sampling-mode {args.sampling_mode} contradicts the {settings['sampling_mode']} "
                             f"model in {args.model}")
        args.dit_weight = os.path.join(args.model, MODEL_WEIGHT_FILE)
    else:
        mode = args.sampling_mode or "base"
        settings = dict(get_profile(mode), sampling_mode=mode)
    args.sampling_mode = settings["sampling_mode"]
    if args.sampling_mode == "base" and args.output_type == "latent":
        # The base pipeline rescales its output as if it were pixels, (x / 2 + 0.5).clamp(0, 1), on the
        # latent branch too, which destroys the latents; only distilled can export them.
        raise ValueError("--output-type latent is only supported with --sampling-mode distilled")
    if args.seed is None:
        args.seed = args.global_seed

    # The selected chain's sampling parameters -- the model directory's config.json, or the built-in profile --
    # are merged onto the namespace, so the pipelines (which read a handful of attributes off `args`) and the
    # logging below see one object. The resolution and the context noise are the values the command line may
    # override; the context noise lands on the chain's own knob (`_apply_context_noise`), not under its
    # command-line name.
    height, width = args.height, args.width
    context_noise = vars(args).pop("context_noise")
    for key, value in settings.items():
        setattr(args, key, value)
    if height is not None:
        args.height = height
    if width is not None:
        args.width = width
    if context_noise is not None:
        _apply_context_noise(args, context_noise)
    # A model folder fixes the resolution its weights were trained at (the released ones: 480x864), for i2v too:
    # the input image is scaled to cover it and centre-cropped. Only a bare --dit-weight run sizes an i2v row
    # from its input image. --height / --width win either way, and either flag counts: giving one side and
    # leaving the other to the profile is still an explicit request.
    args.i2v_size_from_image = args.model is None and height is None and width is None

    _check_resolution(args.height, args.width)
    _check_video_length(args.video_length, args.chunk_size, args.sampling_mode)
    _check_vae_stream_decode_max_chunk_size(args.vae_stream_decode_max_chunk_size, args.sampling_mode)
    _check_seeds(args.global_seed, args.seed, args.sp_size)
    return args


def _apply_context_noise(args, sigma):
    """Set the selected chain's context-noise sigma from `--context-noise`.

    The chains spell it differently. Distilled reads `context_noise`, and its DMD pipeline already skips the noise
    at 0. Base reads `inference_context_noise` behind `context_noise_enabled`, and draws the noise from the sample's
    generator even at sigma 0, which would shift the noise of every later chunk -- so 0 switches it off there.
    """
    if not 0.0 <= sigma < 1.0:
        raise ValueError(f"--context-noise must be in [0, 1), got {sigma}")
    if args.sampling_mode == "distilled":
        args.context_noise = sigma
    elif sigma > 0:
        args.inference_context_noise = sigma
    else:
        args.context_noise_enabled = False


# `accelerate.utils.set_seed` forwards to numpy, which takes a uint32: it accepts 2**32 - 1 and rejects 2**32.
_MAX_SEED = 2 ** 32 - 1


def _check_seeds(global_seed, seed, sp_size):
    """Reject seeds numpy will refuse, before the weights load rather than after.

    `main` uses `global_seed + rank // sp_size`, so a `--global-seed` inside the limit can still put the
    LAST rank over it, and that rank is the only one that fails -- a hang or a partial run rather than a
    clean error. WORLD_SIZE is read from the environment, not `args`, because the ranks are not up yet.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    highest_rank_seed = global_seed + (world_size - 1) // max(sp_size, 1)
    for name, value in (("--global-seed", global_seed), ("--seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_SEED:
            raise ValueError(f"{name} must be an integer in [0, {_MAX_SEED}], got {value!r}")
    if highest_rank_seed > _MAX_SEED:
        raise ValueError(
            f"--global-seed {global_seed} is in range, but rank {world_size - 1} would use "
            f"{highest_rank_seed} (global_seed + rank // sp_size), which is above {_MAX_SEED}; "
            f"lower --global-seed to at most {_MAX_SEED - (world_size - 1) // max(sp_size, 1)}"
        )


def _check_resolution(height, width):
    """The VAE downsamples by 16 in space, so both sides must be positive multiples of 16."""
    for name, value in (("--height", height), ("--width", width)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 16:
            raise ValueError(f"{name} must be a positive multiple of 16, got {value!r}")


def _aspect_bucket_sizes():
    """The (height, width) candidates an i2v run may be sized to, at a roughly constant patch count.

    Walks the patch grid from "one row, as wide as possible" towards square, keeping every shape whose side
    ratio is within `I2V_BUCKET_MAX_RATIO`; the product of the two patch counts stays at or just under
    `(base / patch) ** 2`, so every candidate costs about the same attention.
    """
    base, patch = I2V_BUCKET_BASE_SIZE, I2V_BUCKET_PATCH_SIZE
    num_patches = round((base / patch) ** 2)
    sizes = []
    tall, wide = num_patches, 1
    while tall > 0:
        if max(tall, wide) / min(tall, wide) <= I2V_BUCKET_MAX_RATIO:
            sizes.append((tall * patch, wide * patch))
        if (wide + 1) * tall <= num_patches:
            wide += 1
        else:
            tall -= 1
    return sizes


def _closest_aspect_bucket(image_height, image_width):
    """Size an i2v run from its input image: the candidate closest in aspect ratio, crop-side biased.

    The search is one-sided -- a portrait or square image only considers candidates no taller than itself, a
    landscape image only candidates no wider -- so scaling to cover and centre-cropping always trims the
    image's LONGER side and never has to invent pixels along the shorter one.
    """
    sizes = _aspect_bucket_sizes()
    image_ratio = float(image_height) / float(image_width)
    deltas = [(index, (h / w) - image_ratio) for index, (h, w) in enumerate(sizes)]
    if image_ratio >= 1:
        allowed = [pair for pair in deltas if pair[1] <= 0]
    else:
        allowed = [pair for pair in deltas if pair[1] >= 0]
    # Non-empty for any ratio: the candidate list spans both extremes, so the side being searched always
    # contains at least the most extreme candidate.
    return sizes[min(allowed, key=lambda pair: abs(pair[1]))[0]]


def _check_video_length(video_length, chunk_size, sampling_mode):
    """Reject frame counts that would silently produce a shorter video than the one asked for.

    `latent_length = (video_length - 1) // 4 + 1` floors, and `MovieRollStreamingPipeline.__call__` rounds
    down again to a whole number of chunks plus one, so on the base chain 18, 19 and 20 frames all give the
    same video as 17. On the distilled chain `latent_length % chunk_size != 1` is a hard stop in
    `MovieRollDMDPipeline._execute_rollout`, and its first chunk must be whole, so 4 * chunk_size + 1 is the
    shortest distilled video.
    """
    if video_length <= 0 or (video_length - 1) % 4:
        raise ValueError(f"--video-length must be positive and 4n+1 pixel frames, got {video_length}")
    latent_length = (video_length - 1) // 4 + 1
    if latent_length % chunk_size != 1:
        num_chunks = (latent_length - 1) // chunk_size
        nearest = [
            4 * (n * chunk_size + 1 - 1) + 1
            for n in (num_chunks, num_chunks + 1)
            if n >= 1
        ]
        raise ValueError(
            f"--video-length {video_length} gives {latent_length} latent frames, which is not 1 modulo "
            f"chunk_size={chunk_size}; the nearest valid frame counts are {nearest}"
        )
    if sampling_mode == "distilled" and latent_length < 1 + chunk_size:
        raise ValueError(
            f"--video-length {video_length} gives {latent_length} latent frame(s), but the distilled chain needs "
            f"a first chunk of {1 + chunk_size}; its shortest video is {4 * chunk_size + 1} frames"
        )


def _check_i2v_video_length(video_length, chunk_size):
    """Image-to-video needs at least one generated chunk after the 1-frame input chunk.

    The layout is [1, chunk_size, ...], so 4 * chunk_size + 1 pixel frames is the shortest i2v video.
    """
    latent_length = (video_length - 1) // 4 + 1
    if latent_length < 1 + chunk_size:
        raise ValueError(
            f"--video-length {video_length} gives {latent_length} latent frame(s); image-to-video needs the "
            f"input frame plus one chunk of {chunk_size}, so its shortest video is {4 * chunk_size + 1} frames"
        )


def _check_vae_stream_decode_max_chunk_size(value, sampling_mode):
    """Reject the decode-chunk values the selected chain's decode cannot use, before any weight loads.

    The chains read this profile value differently: distilled (`_streaming_decode_and_save`) takes whatever
    `int()` takes and decodes in one piece for <= 0, so None is unusable; base (the pipeline's `__call__`)
    decodes in one piece for None and otherwise needs a positive int. The check runs for every
    `--output-type`, so a latent export that never decodes still refuses a broken profile.
    """
    name = "vae_stream_decode_max_chunk_size"
    if sampling_mode == "distilled":
        if value is None:
            raise ValueError(f"{name}=None: the distilled chain decodes in one piece for <= 0 and cannot take None")
        try:
            int(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name}={value!r}: the distilled chain reads it with int()") from None
    elif sampling_mode == "base" and value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer or None on the base chain, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name}={value}: the base chain decodes in one piece for None and rejects <= 0")


def _streaming_decode_and_save(args, vae, device, latents, save_path):
    """T-cache streaming decode of the distilled branch.

    Undoes the VAE scale/shift and decodes in chunks of vae_stream_decode_max_chunk_size latent frames (<= 0
    decodes in one piece). ``save_path=None`` means this rank only takes part in the (parallel) decode and
    writes nothing.
    """

    # sample_validation's device is the int index returned by torch.cuda.current_device();
    # decode_latent_chunks_streaming's autocast reads device.type, so normalise it to a torch.device.
    if not isinstance(device, torch.device):
        device = torch.device("cuda", int(device))
    vae_dtype = PRECISION_TO_TYPE[str(args.vae_precision)]
    latents = latents.to(device, dtype=vae_dtype)
    scaling_factor = getattr(vae.config, "scaling_factor", None)
    if scaling_factor is None or float(scaling_factor) == 0.0:
        raise RuntimeError("streaming decode requires a non-zero VAE scaling_factor")
    latents = latents / float(scaling_factor)
    shift_factor = getattr(vae.config, "shift_factor", None)
    if shift_factor is not None:
        latents = latents + float(shift_factor)
    if not callable(getattr(vae, "clear_cache", None)):
        raise TypeError("streaming decode requires the streaming VAE")

    max_chunk_size = int(args.vae_stream_decode_max_chunk_size)
    latent_chunks = (
        (latents,)
        if max_chunk_size <= 0
        else latents.split(max_chunk_size, dim=2)
    )
    decoded_chunks = decode_latent_chunks_streaming(vae, latent_chunks, vae_dtype=vae_dtype, device=device)
    if save_path is None:
        return
    video = torch.cat(decoded_chunks, dim=2)
    video = (video / 2 + 0.5).clamp(0, 1).cpu().float()
    _save_sample(args, video, save_path)


def _save_sample(args, video, save_path, n_rows=1):
    """Write one decoded sample according to --output-type (`latent` never reaches here)."""
    # Replace the EXTENSION, never a substring: an output dir, csv name or index containing "mp4" would
    # otherwise be rewritten too and the file would land somewhere else.
    stem = os.path.splitext(str(save_path))[0]
    if args.output_type == "pt":
        torch.save(video, stem + ".pt")
        return
    if video.shape[2] > 1:
        save_videos_grid(video, str(save_path), rescale=False, n_rows=n_rows, fps=args.video_fps)
    else:
        # single-frame output: an mp4 of one frame is useless, write the frame instead
        torchvision.utils.save_image(video[0, :, 0], stem + ".png")


def _setup_vae_parallel_decode(args, vae, device, logger):
    """--vae-parallel-mode: PS-VAE decode, i.e. the streaming decode row-sharded across a VAE group.

    Every rank of a VAE group must decode the same latents, so each group has to live inside one sp group.
    The check runs on every rank before raising, otherwise one rank would exit while the others block
    forever in a later collective.
    """
    vae_group = initialize_vae_parallel_group(args.vae_parallel_mode, args.vae_parallel_ranks)
    if args.vae_parallel_mode == "none":
        return
    if not callable(getattr(vae, "enable_parallel_decode", None)):
        raise ValueError(
            f"--vae-parallel-mode {args.vae_parallel_mode} requires the streaming VAE"
        )
    if nccl_info.sp_size > 1:
        sp_ranks = set(dist.get_process_group_ranks(nccl_info.sp_group))
    else:
        sp_ranks = {dist.get_rank()}
    ok = torch.tensor(
        [int(vae_group is None or set(nccl_info.vae_group_ranks) <= sp_ranks)], device=device
    )
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if not ok.item():
        raise ValueError(
            f"--vae-parallel-ranks {args.vae_parallel_ranks!r}: every VAE group must live inside one sp group"
        )
    if vae_group is not None:
        vae.enable_parallel_decode(vae_group)
    logger.info(
        f"PS-VAE parallel decode: mode={args.vae_parallel_mode}, group ranks={nccl_info.vae_group_ranks}"
    )


def _set_rollout_sync_group(pipeline):
    """Give the distilled pipeline its rollout-shape sync group; returns whether one was set.

    Rollout-shape draws are synchronized across the ranks that roll out one sample, which here is one sp
    group: sp_group avoids coupling dp_shard peers that run different prompts into one draw stream.
    SAMPLING itself never draws, so with sp 1 (one rank per sample, sp_group None) no group is set --
    `set_sync_group` rejects None, and an unexpected synchronized draw still raises in `_sync_group`.
    """
    if dist.is_initialized() and dist.get_world_size() > 1 and nccl_info.sp_size > 1:
        pipeline.set_sync_group(nccl_info.sp_group)
        return True
    return False


def _check_input_images(args, input_images):
    """Refuse a run that mixes i2v and t2v rows, names a missing image, or is too short for i2v.

    An i2v sample runs one transformer forward more than a t2v one (the input frame's KV encode), and the dp
    ranks of a shard group run different rows, so mixed rows desynchronise the FSDP all-gathers.
    """
    with_image = [path is not None for path in input_images]
    if any(with_image) and not all(with_image):
        raise ValueError(
            f"{sum(with_image)} of {len(with_image)} rows have an input_image: either every row of a run is "
            "image-to-video or none is"
        )
    missing = [path for path in input_images if path is not None and not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} input image(s) not found, e.g. {missing[0]}")
    if any(with_image):
        _check_i2v_video_length(args.video_length, args.chunk_size)


def _load_prompt_inputs(args, logger, dp_degree, dp_rank):
    """Resolve the prompt source into this dp rank's share of the work.

    Returns ``(use_fixed_prompt, prompts, seeds, indexs, csv_file_names, event_prompts_list, input_images,
    total_samples)``; ``input_images`` holds each sample's input image
    path, or None for text-to-video. For --prompt / --prompt-list every dp rank gets the same single sample:
    the text encoder is FSDP-sharded over the whole world, so the ranks must perform the same number of
    encodes or the all-gathers never line up.
    """
    prompts = []
    seeds_list = []
    indexs = []
    csv_file_names = []
    event_prompts_list = []  # list of composed event prompt lists per sample
    input_images = []  # None (t2v) or the input image path per sample

    if args.prompt_csv is None:
        events = list(args.prompt_list) if args.prompt_list is not None else [args.prompt]
        prompts = [events[0]]
        seeds_list = [args.seed]
        indexs = ["0"]
        csv_file_names = [("default", "default")]
        event_prompts_list = [events]
        input_images = [args.input_image]
        _check_input_images(args, input_images)
        logger.info(f"Prompt input: {len(events)} event prompt(s), seed={args.seed}, "
                    f"input image={args.input_image}")
        return (False, prompts, seeds_list, indexs, csv_file_names, event_prompts_list, input_images, 1)

    for pmt_path in args.prompt_csv:
        table = read_prompt_csv(pmt_path)
        logger.info(f"CSV loaded: {pmt_path} ({len(table.prompts)} samples, "
                    f"events={'yes('+str(len(table.event_cols))+')' if table.event_cols else 'no'}, "
                    f"input_image={'yes' if table.has_input_image else 'no'})")

        prompts = prompts + table.prompts
        seeds_list = seeds_list + table.seeds
        indexs = indexs + table.indexs
        csv_file_names = csv_file_names + table.csv_names
        event_prompts_list = event_prompts_list + table.event_prompts
        input_images = input_images + table.input_images

    event_counts = sorted({len(events) for events in event_prompts_list})
    if len(event_counts) > 1:
        raise ValueError(
            f"the --prompt-csv rows have {event_counts} events: every row of a run must have the same number, "
            "since each event is one text encode over all ranks and mixed counts hang the run"
        )
    _check_input_images(args, input_images)
    logger.info(f"Total inference samples: {len(prompts)}")
    total_samples = len(prompts)
    # total_samples must be divisible by dp_degree or the FSDP hooks deadlock
    if total_samples % dp_degree != 0:
        padding_size = dp_degree - total_samples % dp_degree

        # padding_size can exceed the number of samples (e.g. one row with dp_degree 4), where
        # prompts[:padding_size] would leave some dp ranks with no sample: they run straight to the final
        # barrier while the ranks that do have samples wait forever in the world-level FSDP all-gather of
        # the text encoder. Repeating cyclically always adds exactly padding_size.
        def _pad_to_divisible(lst):
            lst.extend((lst * ((padding_size // len(lst)) + 1))[:padding_size])
        _pad_to_divisible(prompts)
        _pad_to_divisible(seeds_list)
        _pad_to_divisible(indexs)
        _pad_to_divisible(csv_file_names)
        _pad_to_divisible(event_prompts_list)
        _pad_to_divisible(input_images)
        total_samples = len(prompts)
    selected_idx = list(range(dp_rank, total_samples, dp_degree))
    indexs = [indexs[i] for i in selected_idx]
    prompts = [prompts[i] for i in selected_idx]
    seeds_list = [seeds_list[i] for i in selected_idx]
    csv_file_names = [csv_file_names[i] for i in selected_idx]
    event_prompts_list = [event_prompts_list[i] for i in selected_idx]
    input_images = [input_images[i] for i in selected_idx]
    return (True, prompts, seeds_list, indexs, csv_file_names, event_prompts_list, input_images, total_samples)


@torch.no_grad()
def sample_validation(args, device, vae, text_encoder, model,
                      logger, byt5_kwarg, dp_degree, dp_rank):
    logger.info("---------------------------- Sample Stage --------------------------")
    model.eval()
    # the pair exists to avoid an OOM at the start of sampling
    gc.collect()
    torch.cuda.empty_cache()

    _distilled = args.sampling_mode == "distilled"
    if _distilled:
        # Few-step DMD weights need the DMD pipeline plus the tensor-step scheduler; the step list, context
        # noise and timestep shift come from the distilled profile.
        denoising_step_list = list(args.denoising_step_list)
        context_noise = float(args.context_noise)
        scheduler = FlowMatchStreamingScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            shift=float(args.timestep_shift),
        )
        pipeline = MovieRollDMDPipeline(
            vae=vae,
            text_encoder=text_encoder,
            transformer=model,
            scheduler=scheduler,
            precision=args.precision,
            vae_precision=args.vae_precision,
            prompt_format_font_path=args.multilingual_prompt_format_font_path,
            prompt_format_color_path=args.multilingual_prompt_format_color_path,
            byt5_kwarg=byt5_kwarg,
            text_len=args.text_len,
            streaming_config=args.streaming_config,
            device=device,
            # is_eval=True would .to(device) the vae / text_encoder again, which is both redundant and
            # dangerous for an already fully_shard-ed text encoder.
            is_eval=False,
        )
        pipeline.transformer.kv_cache_manager.verbose = False
        sync_group = "sp_group" if _set_rollout_sync_group(pipeline) else "none"
        # The DMD rollout requires window_size == len(denoising_step_list); failing here points at the
        # configuration instead of at the rollout.
        if int(args.streaming_config.window_size) != len(denoising_step_list):
            raise ValueError(
                "distilled sampling requires window_size == len(denoising_step_list): "
                f"{int(args.streaming_config.window_size)} != {len(denoising_step_list)}"
            )
        logger.info(
            f"distilled sampling: denoising_step_list={denoising_step_list} "
            f"context_noise={context_noise} sync_group={sync_group}"
        )
    else:
        scheduler = FlowMatchStreamingScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            shift=args.flow_shift,
        )
        pipeline = MovieRollStreamingPipeline(
            vae=vae,
            text_encoder=text_encoder,
            transformer=model,
            scheduler=scheduler,
            precision=args.precision,
            vae_precision=args.vae_precision,
            prompt_format_font_path=args.multilingual_prompt_format_font_path,
            prompt_format_color_path=args.multilingual_prompt_format_color_path,
            byt5_kwarg=byt5_kwarg,
            text_len=args.text_len,
            streaming_config=args.streaming_config,
        )

    (use_fixed_prompt, prompts, seeds_list, indexs, csv_file_names, event_prompts_list, input_images,
     total_samples) = _load_prompt_inputs(args, logger, dp_degree, dp_rank)

    for sidx in range(len(prompts)):
        prompt = prompts[sidx]
        current_seed = seeds_list[sidx]
        index = indexs[sidx]
        csv_file_name = csv_file_names[sidx]
        event_composed = event_prompts_list[sidx]
        # With 2 or more events both chains split the video evenly into one segment per event and switch the
        # prompt at every boundary; the base call takes the events only then, the distilled call always.
        multi_event = len(event_composed) >= 2
        event_prompts_arg = list(event_composed) if multi_event else None
        data_type = "video"
        target_length = int(args.video_length)
        input_image = input_images[sidx]

        # Without a model folder, i2v is sized by its input image, so a portrait photo does not come back as a
        # landscape video with its top and bottom cropped away; a CSV may then mix aspect ratios because the
        # choice is made per row (see `parse_args`).
        if input_image is not None and args.i2v_size_from_image:
            probe = open_input_image(input_image)
            target_height, target_width = _closest_aspect_bucket(probe.height, probe.width)
        else:
            target_height, target_width = args.height, args.width

        # Consumes exactly one global random.random() per sample, i2v rows included, which keeps the RNG
        # stream in step. The mask itself is unused: the transformer input is the t2v layout in both tasks,
        # the i2v image enters through the KV cache only.
        _, mask_type = get_multitask_mask((target_length - 1) // 4 + 1)
        cond_latents = None
        if input_image is not None:
            cond_latents = encode_input_image(
                vae, args.vae_precision, device, input_image, target_height, target_width
            )
            mask_type = "i2v"

        if multi_event:
            evt_summary = ' | '.join(f"e{i+1}: {p[:60]}..." for i, p in enumerate(event_composed))
            print(f"dp_rank {dp_rank}, Multi-event ({len(event_composed)} events) | {evt_summary}")
        print(f"dp_rank {dp_rank}, sample_validation Data type: {data_type} \t mask_type: {mask_type} \t "
              f"target_length: {target_length}\t target_height: {target_height}\t "
              f"target_width: {target_width} \t seed: {current_seed} \t prompt: {prompt}"
              + (f" \t input_image: {input_image}" if input_image is not None else ""))

        if _distilled:
            # One cpu generator for the whole prompt, created OUTSIDE the per-video loop so each pass
            # consumes the next draws from it and the videos differ; re-seeding per pass would make them
            # identical. cpu, not device: a device generator produces a different random stream.
            rollout_generator = torch.Generator(device="cpu").manual_seed(current_seed)
            # a loop and not a batch: see --num-videos-per-prompt's help for why N cannot be batched inside
            for video_index in range(args.num_videos_per_prompt):
                result = pipeline.execute(
                    mode=RollingMode.SAMPLING,
                    prompt=prompt,
                    event_prompts=list(event_composed),
                    height=target_height,
                    width=target_width,
                    video_length=target_length,
                    denoising_step_list=denoising_step_list,
                    context_noise=context_noise,
                    generator=rollout_generator,
                    multitask_type=mask_type,
                    cond_latents=cond_latents,
                )
                # with i2v the input frame is output chunk 0, so the latents cover the whole video either way
                latents = result.generated_latents
                # The dp guard and the `_v{N}` naming below are duplicated in the base branch; change both
                # together. Only rank 0 of an sp group writes, and with --prompt / --prompt-list every dp rank
                # generates the same video, so only dp rank 0 keeps it. Skipping is group-uniform (a VAE
                # group lives inside one sp group, which lives inside one dp rank), so a skipping rank
                # never desynchronises the parallel decode collectives.
                produces_output = dp_rank < total_samples
                is_saver = produces_output and nccl_info.rank_within_spgroup == 0
                save_path = None
                if is_saver:
                    save_path = _build_save_path(args, use_fixed_prompt, csv_file_name, index, dp_rank, sidx)
                    if args.num_videos_per_prompt > 1:
                        stem, extension = os.path.splitext(str(save_path))
                        save_path = f"{stem}_v{video_index}{extension}"
                if args.output_type == "latent":
                    if is_saver:
                        torch.save(latents.cpu(), os.path.splitext(str(save_path))[0] + ".pt")
                        logger.info(f'dp_rank {dp_rank} saved distilled latents to {save_path}')
                elif produces_output and (
                    is_saver or getattr(vae, "parallel_decode_group", None) is not None
                ):
                    # Parallel VAE decode is a collective: every rank of a VAE group decodes together.
                    _streaming_decode_and_save(args, vae, device, latents, save_path)
                    if is_saver:
                        logger.info(f'dp_rank {dp_rank} saved distilled sample to {save_path}')
                del result, latents
                gc.collect()
                torch.cuda.empty_cache()
            del cond_latents
            continue

        # One generator for the whole prompt, created OUTSIDE the loop: each pass consumes the next draws
        # from it, so the videos differ. Re-seeding per pass would make them identical.
        generator = torch.Generator(device=device).manual_seed(current_seed)
        # a loop and not a batch: see --num-videos-per-prompt's help for why N cannot be batched inside
        for video_index in range(args.num_videos_per_prompt):
            samples = pipeline(prompt=prompt,
                               height=target_height,
                               width=target_width,
                               video_length=target_length,
                               num_inference_steps=args.num_inference_steps,
                               use_cfg=args.use_cfg,
                               negative_prompt=args.negative_prompt,
                               generator=generator,
                               output_type='pt',
                               event_prompts=event_prompts_arg,
                               cond_latents=cond_latents,
                               # __call__ reads this one out of **kwargs, so it has to be passed by name or
                               # the profile value is silently replaced by the in-pipeline default
                               vae_stream_decode_max_chunk_size=args.vae_stream_decode_max_chunk_size,
                               )[0]

            # write out only the rows this dp rank owns; rank 0 of the sp group writes (checked below)
            if dp_rank < total_samples:
                save_path = _build_save_path(args, use_fixed_prompt, csv_file_name, index, dp_rank, sidx)
                if args.num_videos_per_prompt > 1:
                    # Split on the extension, never by substring, and only when there is more than one
                    # video, so a single-video run keeps its plain name.
                    stem, extension = os.path.splitext(str(save_path))
                    save_path = f"{stem}_v{video_index}{extension}"

                # one video per sp group
                if nccl_info.rank_within_spgroup == 0:
                    _save_sample(args, samples[0].unsqueeze(0), save_path, n_rows=1)
                    logger.info(f'dp_rank {dp_rank} saved sample to {save_path}')
            del samples
            gc.collect()
            torch.cuda.empty_cache()

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def _build_save_path(args, use_fixed_prompt, csv_file_name, index, dp_rank, sidx):
    """Output path of one sample: samples/<csv dir>/<csv name>/<index>.mp4 for a CSV run."""
    if use_fixed_prompt:
        folder_name, file_name = csv_file_name
        save_dir = os.path.join(args.output_dir, "samples", folder_name, file_name)
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"{index}.mp4")
    save_dir = os.path.join(args.output_dir, "samples", "default")
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"rank{dp_rank}_sample_{sidx}.mp4")


def print_sample_configuration(args, logger, model, world_size, local_rank, rank, dp_degree, dp_rank):

    logger.info("****************************** Running sample ******************************")

    # ===============================================================================
    # System & Hardware Configuration
    # ===============================================================================
    logger.info("=" * 80)
    logger.info("SYSTEM & HARDWARE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Number of GPUs                                   : {world_size}")
    logger.info(f"Local rank                                       : {local_rank}")
    logger.info(f"Global rank                                      : {rank}")
    logger.info(f"Device                                           : cuda:{local_rank}")
    logger.info(f"Master weight dtype                              : {next(model.parameters()).dtype}")
    # ===============================================================================
    # Distributed Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("DISTRIBUTED CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"World size                                       : {world_size}")
    logger.info(f"DP degree                                        : {dp_degree}")
    logger.info(f"DP rank                                          : {dp_rank}")
    logger.info(f"SP size                                          : {args.sp_size}")
    logger.info(f"FSDP sharding strategy                           : {args.fsdp_sharding_strategy}")
    logger.info(f"CPU offload                                      : {args.use_cpu_offload}")
    # ===============================================================================
    # Inference & Validation Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("INFERENCE & VALIDATION CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Sampling mode                                    : {args.sampling_mode}")
    logger.info(f"Attention backend                                : {args.attention_backend}")
    logger.info(f"Video length / fps                               : {args.video_length} / {args.video_fps}")
    logger.info(f"Resolution                                       : {args.height}x{args.width}"
                + ("  (i2v rows size themselves from their input image)" if args.i2v_size_from_image else ""))
    if hasattr(args, 'use_cfg'):
        logger.info(f"CFG (AnnealCFG)                                  : {args.use_cfg}")
    if hasattr(args, 'num_inference_steps'):
        logger.info(f"Inference steps                                  : {args.num_inference_steps}")
    if hasattr(args, 'flow_shift'):
        logger.info(f"Inference flow shift                             : {args.flow_shift}")
    if hasattr(args, 'negative_prompt'):
        logger.info(f"Negative prompt                                  : {args.negative_prompt}")
    if hasattr(args, 'denoising_step_list'):
        logger.info(f"Denoising step list                              : {args.denoising_step_list}")
        logger.info(f"Context noise                                    : {args.context_noise}")
    logger.info(f"VAE path                                         : {args.vae_path}")

    # ===============================================================================
    # Seeds & Reproducibility Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("SEEDS & REPRODUCIBILITY CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Global seed                                      : {args.global_seed}")
    logger.info(f"Local seed                                       : {args.local_seed}")

    # ===============================================================================
    # Additional Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("ADDITIONAL CONFIGURATION: GLYPH BYT5 V2")
    logger.info("-" * 80)
    logger.info(f"ByT5 max length                                  : {args.byt5_max_length}")

    # ===============================================================================
    # Streaming Configuration
    # ===============================================================================
    logger.info("-" * 80)
    logger.info("STREAMING CONFIGURATION")
    logger.info("-" * 80)
    logger.info(f"Streaming config                                 : {args.streaming_config}")
    logger.info(f"Streaming max num latent frames                  : {args.max_num_latent_frames}")
    logger.info(f"Streaming chunk size                             : {args.chunk_size}")
    logger.info(f"Streaming window size                            : {args.window_size}")
    logger.info(f"Streaming full history block num                 : {args.full_history_block_num}")
    logger.info(f"Streaming kv recompute enabled                   : {args.kv_recompute_enabled}")
    logger.info(f"Streaming kv recompute threshold                 : {args.kv_recompute_threshold}")
    logger.info(f"Streaming kv recompute recent k                  : {args.kv_recompute_recent_k}")
    logger.info("=" * 80)
    logger.info("STARTING SAMPLE")
    logger.info("=" * 80)


def _resolve_master_weight_type(requested, use_fsdp):
    """The master-weight dtype of the run: the request, else bf16 without FSDP and fp32 with it.

    Refuses bf16 masters under FSDP up front instead of at the first forward: FSDP2 raises `FSDP expects uniform original
    parameter dtype` because the transformer keeps a few small parameters in float32 whatever dtype it is built with
    (`byt5_in.*`, `final_layer.linear.*`, `cond_type_embedding`; see `--master-weight-type`).
    """
    if requested is None:
        return "fp32" if use_fsdp else "bf16"
    if requested == "bf16" and use_fsdp:
        raise ValueError(
            "--master-weight-type bf16 cannot be sharded with FSDP (the transformer keeps a few parameters in float32 "
            "and FSDP2 needs one dtype per parameter group): use fp32 masters (the default with FSDP), or run without "
            "FSDP (a single process, or --fsdp-sharding-strategy none)"
        )
    return requested


def main(args):

    # ============================= Setup ==============================
    # Setup distributed environment
    torch.backends.cuda.matmul.allow_tf32 = True
    set_attention_backend(args.attention_backend)
    # Without torchrun, importing `movieroll` has set RANK / LOCAL_RANK / WORLD_SIZE / MASTER_* for one process.
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # FSDP is pointless for one process (it would shard over one rank), so a single-process run never uses it;
    # `--fsdp-sharding-strategy none` also drops it for several processes, each of which then holds the full weights.
    use_fsdp = world_size > 1 and args.fsdp_sharding_strategy != 'none'
    if not use_fsdp:
        args.fsdp_sharding_strategy = 'none'
    # FSDP shard-group size (dp_shard * sp). The default 8 is one node per group. Lower it for more dp
    # concurrency (it must stay divisible by sp_size); when world == shard, hybrid degrades to full and the
    # other nodes become dp_replicate.
    shard_world = int(args.shard_world)
    if use_fsdp:
        if shard_world < 1 or shard_world > world_size:
            raise ValueError(f"--shard-world must be in [1, {world_size}], got {shard_world}")
        if world_size % shard_world != 0:
            raise ValueError(f"world_size={world_size} must be divisible by --shard-world={shard_world}")
        if shard_world % args.sp_size != 0:
            raise ValueError(f"--shard-world={shard_world} must be divisible by --sp-size={args.sp_size}")
    elif world_size % args.sp_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by --sp-size={args.sp_size}")
    args.master_weight_type = _resolve_master_weight_type(args.master_weight_type, use_fsdp)
    num_heads = MODEL_CONFIG[MODEL_TYPE]["heads_num"]
    if num_heads % args.sp_size != 0:
        raise ValueError(f"--sp-size={args.sp_size} must divide the model's {num_heads} attention heads")

    # Fail before loading 33 GB of weights if the prompt CSV is missing.
    for pmt_path in (args.prompt_csv or []):
        if not os.path.isfile(pmt_path):
            raise FileNotFoundError(f"--prompt-csv not found: {pmt_path}")

    if not use_fsdp:
        # no sharding: every sp group holds a complete model, and the groups are the data-parallel replicas
        dp_replicate, dp_shard = world_size // args.sp_size, 1
    elif args.fsdp_sharding_strategy == 'full':
        dp_replicate, dp_shard = 1, -1
    else:
        assert args.fsdp_sharding_strategy == 'hybrid'
        dp_replicate, dp_shard = world_size // shard_world, -1

    if use_fsdp and world_size == shard_world and args.fsdp_sharding_strategy == 'hybrid':
        args.fsdp_sharding_strategy = "full"
        dp_replicate = 1
        print(f"world_size==shard_world({shard_world}) fall back hybrid to full-shard")

    parallel_dims = ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=dp_shard,
        sp=args.sp_size,
        world_size=world_size,
    )
    init_nccl_process_group()
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    # a device mesh over one rank has no dimension to name; nothing below needs it for a single process
    world_mesh = parallel_dims.build_mesh(device_type="cuda") if world_size > 1 else None
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    initialize_sequence_parallel_state_v2(parallel_dims, world_mesh)

    if rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # sync output_dir over all ranks
    if args.output_dir is not None:
        obj_list = [args.output_dir]
        dist.broadcast_object_list(obj_list, src=0)
        args.output_dir = obj_list[0]

    logger = setup_logger(args.output_dir)
    logger.info(f"--> args {args}")
    logger.info(f"dp_degree, dp_rank: {dp_degree, dp_rank}")
    # Parse the prompt inputs before the weights load, so a mixed i2v / t2v CSV, a missing input image or a
    # too short i2v video fails in seconds; sample_validation parses them again for the run.
    _load_prompt_inputs(args, logger, dp_degree, dp_rank)

    # =========================== Build main model ===========================
    logger.info(f"--> loading model from {args.dit_weight}")
    master_dtype = PRECISION_TO_TYPE[args.master_weight_type]
    factor_kwargs = {'device': device, 'dtype': master_dtype}

    streaming_config_kwargs = dict(
        max_num_latent_frames=args.max_num_latent_frames,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        full_history_block_num=args.full_history_block_num,
        kv_recompute_enabled=args.kv_recompute_enabled,
        kv_recompute_threshold=args.kv_recompute_threshold,
        kv_recompute_recent_k=args.kv_recompute_recent_k,
    )
    if args.sampling_mode == "base":
        # The distilled chain reads none of these: its guidance is distilled into the weights and its
        # context noise is the DMD rollout's own `context_noise`.
        streaming_config_kwargs.update(
            cfg_min=args.cfg_min,
            cfg_max=args.cfg_max,
            context_noise_enabled=args.context_noise_enabled,
            inference_context_noise=args.inference_context_noise,
        )
    streaming_config = StreamingConfig(**streaming_config_kwargs)
    factor_kwargs['streaming_config'] = streaming_config
    args.streaming_config = streaming_config

    model = load_transformer(
        args.model_type,
        args.dit_weight,
        factor_kwargs,
        logger,
    )

    model.eval()
    # One seed per SP group (rank // sp_size is the group index). No timestep is drawn at inference -- the
    # scheduler's fixed step list is used -- and each sample's noise comes from its own per-sample seed.
    args.local_seed = args.global_seed + rank // args.sp_size
    set_seed(args.local_seed)
    random.seed(args.local_seed)
    print(f"dp_rank: {dp_rank}, rank: {rank},  local_seed: {args.local_seed}")

    # ============================= Build extra models =========================
    vae, _, _, _ = load_vae(
        args.vae_precision,
        args.vae_path,
        logger=logger,
        device=device,
        streaming=True,
    )
    _setup_vae_parallel_decode(args, vae, device, logger)
    logger.info(f"--> loading text encoder from {args.text_encoder_path}")
    # the text encoder is FSDP-sharded below, so it is first loaded onto the cpu
    text_encoder = load_text_encoder(
        args.text_encoder_path,
        args.text_len,
        PROMPT_TEMPLATE[args.prompt_template],
        PROMPT_TEMPLATE[args.prompt_template_video],
        torch.device("cpu"),
        text_encoder_precision=args.text_encoder_precision,
        logger=logger,
    )

    byt5_kwarg = load_glyph_byT5_v2(
        torch.device("cpu"),
        byT5_google_path=args.byt5_google_path,
        byT5_ckpt_path=args.byt5_ckpt_path,
        color_ann_path=args.multilingual_prompt_format_color_path,
        font_ann_path=args.multilingual_prompt_format_font_path,
        byt5_max_length=args.byt5_max_length,
    )

    logger.info(
        f"  Total parameters = {sum(p.numel() for p in model.parameters()) / 1e6} M"
    )

    if use_fsdp:
        # ============================= Build FSDP =========================
        logger.info(
            f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_strategy}"
        )
        fsdp_mesh = fsdp_device_mesh(parallel_dims, world_mesh, args.fsdp_sharding_strategy)
        logger.info(f"FSDP device mesh: {fsdp_mesh}")

        apply_fsdp2(
            model=model,
            dp_mesh=fsdp_mesh,
            param_dtype=master_dtype,
            reduce_dtype=torch.float32,
            cpu_offload=args.use_cpu_offload,
            reshard_after_forward_policy="default",
            prefetch_depth=4,  # 4 at inference, 2 during training
        )
        logger.info("--> model loaded")

        if fsdp_mesh.ndim == 2:     # hybrid with real replicas; a 1-D mesh holds distinct shards on every rank
            broadcast_from_first_replica([model], world_mesh['dp_replicate'].get_group())

        # ============================FSDP for text encoder =========================
        # Sharding the text encoder too saves more than 10 GB of peak memory for about 0.03 s of data time.
        logger.info("--> Initializing FSDP for text encoder")
        from torch.distributed.fsdp import fully_shard
        for layer in text_encoder.model.layers:
            # mesh=None shards over the whole world
            fully_shard(layer, mesh=None)
        fully_shard(text_encoder.model, mesh=None)
        logger.info("--> FSDP for text encoder initialized")
    else:
        # `fully_shard` moved every parameter to the GPU, including the few the transformer builds on the cpu (its
        # `.device` is the first parameter's, and the pipelines take their device from it); without it the move is explicit.
        # The text encoder was loaded on the cpu.
        logger.info("--> no FSDP: every rank holds complete weights")
        model.to(device)
        text_encoder.model.to(device)
    byt5_kwarg["byt5_model"] = byt5_kwarg["byt5_model"].to(device)

    print_sample_configuration(args, logger, model, world_size, local_rank, rank,
                               dp_degree, dp_rank)

    # ============================= Start sample =============================
    sample_validation(args, device, vae, text_encoder, model,
                      logger, byt5_kwarg, dp_degree, dp_rank)
    dist.barrier(device_ids=[local_rank])

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    main(parse_args())
