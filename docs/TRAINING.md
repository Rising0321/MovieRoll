# MovieRoll Training Guide

[中文](./TRAINING_CN.md) · [Back to README](../README.md)

This guide covers data preparation and every training stage. First complete the [environment and checkpoint setup in the README](../README.md#setup). Run all commands below from the repository root.

## Training initialization checkpoint

In addition to the VAE, text encoders, and selected MovieRoll model described in the README, SFT initialization and the DMD score models use the HunyuanVideo-1.5 480p transformer weights:

```bash
hf download tencent/HunyuanVideo-1.5 --include 'transformer/480p_t2v/*' --local-dir ./ckpts
```

## 📦 Training Data

SFT trains on pre-encoded latents: a `manifest.jsonl` plus one latent file per clip. Cut your videos into clips, list
them in a manifest, and `encode_latents` takes care of the rest:

```
raw_clips/                                        movieroll_sft_stage1/
├── manifest.jsonl  (with a "video" field)  ──▶   ├── manifest.jsonl  (the same lines, without "video")
└── clips/0001.mp4 ...     encode_latents         └── latents/0001.pt ...
```

**Step 1: list your clips.** Each line of `manifest.jsonl` describes one clip:

```json
{"id": "0001", "video": "clips/0001.mp4", "text": "A chef flips a pancake in a sunlit kitchen."}
{"id": "0002", "video": "clips/0002.mp4", "text": "A day at the beach.", "duration_sec": 10.1, "events": [{"text": "Kids build a sandcastle.", "start_sec": 0.0, "end_sec": 5.0}, {"text": "A wave washes it away.", "start_sec": 5.0, "end_sec": 10.1}]}
```

| Field | Meaning |
|---|---|
| `id` | the clip's name; its latent is saved as `latents/<id>.pt` |
| `video` | path to the clip, relative to `raw_clips/` or absolute |
| `text` | the caption; an empty string trains the unconditional branch |
| `events` | optional list of `{text, start_sec, end_sec}` for multi-event clips |
| `duration_sec` | length of the clip in seconds; required when `events` is present. Event times are read as fractions of `duration_sec` and mapped onto the encoded first 241 frames, so for a longer clip give `duration_sec` and the event times for that first ~10 s, or cut the clip to that length first |

**Step 2: encode them.** The script runs on any number of GPUs:

```bash
torchrun --nproc_per_node=8 -m movieroll.scripts.encode_latents \
  --src ./data/raw_clips --dst ./data/movieroll_sft_stage1
```

Each clip is trimmed to its first 241 frames at 24 fps (a little over 10 seconds), scaled to cover 480×640,
center-cropped, and encoded with the same VAE code the trainer uses. A clip that yields fewer than 241 frames at
24 fps is rejected; a clip of exactly 10.0 seconds gives only 240, so cut clips slightly longer, e.g. 10.1 seconds.
The latents are float32 tensors of shape `[32, 61, 30, 40]`, already multiplied by the VAE scaling factor. The
defaults match the shipped SFT configs, and an interrupted run picks up where it stopped if you rerun it with
`--skip_existing`.

The distillation stages need prompts only, no videos:

| Stage | CSV columns |
|---|---|
| ODE initialization (`data.prompt_path`) | `prompt`, plus an optional `seed`; a `.txt` file with one prompt per line also works |
| CuLT-DMD, phase 1 / phase 2 | `prompt` / `event_1`, `event_2` |
| SFT sampling during training (`sample_prompt_csv`) | `index`, `seed`, and `prompt` or `event_1..event_N` |
| CuLT-DMD sampling during training (`sampling.test_event_csv`) | `event_1..event_N` (at least `event_1`; a `prompt` column is not read); ACS also needs `input_image` |
| Batch inference (`--prompt-csv`) | `index`, `seed`, and `prompt` or `event_1..event_N`, plus an optional `input_image` |

## 🎓 Training

MovieRoll is trained in four stages:

```
HunyuanVideo-1.5 480p T2V
  └─① Streaming SFT, single-event ──▶ MovieRoll-Base
       └─② Streaming SFT, multi-event ──▶ multi-event Base
            └─③ ODE initialization ──▶ few-step student
                 ├─④ CuLT-DMD ──▶ MovieRoll-Distilled (text-to-video)
                 └─④ CuLT-DMD with ACS ──▶ MovieRoll-Distilled (image-to-video)
```

The config file names number the stages differently: ① and ② are `sft/stage1_single_event.yaml` and
`sft/stage2_multi_event.yaml`, ③ is `stage2_ode.yaml`, and ④ is `stage3_dmd_rolling.yaml` (`stage3_dmd_rolling_acs.yaml`
for image-to-video).

Each stage has a launch script in `movieroll/scripts/launch/`. Set the fields shown below in the stage's config, then
run its script; all paths are checked before any weights are loaded. The scripts read the config path and the
launch topology from environment variables:

```bash
export NPROC=8                      # GPUs per node
export NNODES=1                     # multi-node: number of nodes
export NODE_RANK=0                  # multi-node: this node's index
export MASTER_ADDR=127.0.0.1        # multi-node: address of node 0
```

Each launcher selects its stage's config by default. To override it, set `CONFIG` for that command only, for example `CONFIG=movieroll/configs/sft/stage2_multi_event.yaml bash movieroll/scripts/launch/train_sft.sh`. Do not export one stage's `CONFIG` globally and reuse it for other stages.

### ①② Streaming SFT

Config: `movieroll/configs/sft/stage1_single_event.yaml` for ①, `movieroll/configs/sft/stage2_multi_event.yaml`
for ②. The fields you will most likely touch:

```yaml
dit_weight: ckpts/transformer/480p_t2v/diffusion_pytorch_model.safetensors   # initial weights; ② uses ckpts/movieroll/base/model.safetensors
data_root: ./data/movieroll_sft_stage1      # the encode_latents output (--dst)
output_dir: ./outputs/movieroll_sft_stage1  # checkpoints and samples
height: 480                                 # resolution and length, which must match the latents
width: 640
sample_n_frames: 241
sp_size: 2                                  # sequence-parallel group size (4 in ②)
fsdp_sharding_strategy: hybrid              # full for a single shard group, e.g. on fewer than 8 GPUs
fsdp_sub_world_size: 8                      # GPUs per FSDP shard group; must divide the GPU count (16 in the released runs)
learning_rate: 2.0e-5                       # 1.0e-5 in ②
max_train_steps: 1000000
checkpointing_steps: 2000
streaming_chunk_size: 4                     # latent frames per chunk
streaming_window_size: 5                    # chunks per rolling window
streaming_t2v_prob: 0.8                     # share of text-to-video samples; the rest train image-to-video
streaming_caption_type_probs: {event: 0.0, structure: 1.0}   # ② trains on event captions: {event: 0.9, structure: 0.1}
streaming_context_noise_sigma_max: 0.15     # strongest noise on the clean history
sample_interval: 2000                       # sample every N steps (10000 in ②); 0 turns sampling off
sample_prompt_csv:                          # prompts for those samples
  - ./data/prompts/movieroll_bench_streaming.csv
```

Launch:

```bash
bash movieroll/scripts/launch/train_sft.sh                                                      # stage ①
CONFIG=movieroll/configs/sft/stage2_multi_event.yaml bash movieroll/scripts/launch/train_sft.sh   # stage ②
```

> 💡 To check the setup first, set `dry_run: true` and `max_train_steps: 2`, which runs two steps without training.

### ③ ODE Initialization

Config: `movieroll/configs/stage2_ode.yaml`. The teacher first generates denoising trajectories, then the few-step
student learns to follow them.

```yaml
teacher_ckpt: ./outputs/movieroll_sft_stage2/checkpoints/step-6000/ema_diffusion_pytorch_model.safetensors  # EMA of ②
student_ckpt: ./outputs/movieroll_sft_stage2/checkpoints/step-6000/ema_diffusion_pytorch_model.safetensors  # starts as the teacher
data:
  prompt_path: ./data/ode_init/prompts.csv   # prompts the teacher generates from
ode:
  trajectory_cache_dir: ./data/ode_init/rolling_ode_traj_cache   # written by the precompute, read by training
  denoising_step_list: [1000, 800, 600, 400, 200]               # the student's 5 steps
video:
  num_frames: 241
  resolution: [480, 864]
logdir: ./outputs/movieroll_stage2_ode
max_steps: 1000
save_interval: 100
```

Launch:

```bash
bash movieroll/scripts/launch/precompute_ode.sh    # the teacher writes its trajectories
bash movieroll/scripts/launch/train_ode.sh         # the student learns them
```

The precompute splits across nodes differently: run one job per node with `SHARD_ID=<node index>` and
`NUM_SHARDS=<number of nodes>`, all writing into the same `trajectory_cache_dir`.

### ④ CuLT-DMD Distillation

Config: `movieroll/configs/stage3_dmd_rolling.yaml`. The student learns from long self-rollouts, with gradients
accumulated over the whole rollout, in two phases: 241-frame single-event videos, then 721-frame two-event videos.

```yaml
generator_ckpt: ./outputs/movieroll_stage2_ode/<run>/checkpoints/step-1000/student.safetensors  # the ③ student
real_score_ckpt: ckpts/transformer/480p_t2v/diffusion_pytorch_model.safetensors   # teacher score model
fake_score_ckpt: ckpts/transformer/480p_t2v/diffusion_pytorch_model.safetensors   # critic, starts from the same weights
dmd_phases:
  short_single_event:                       # phase 1
    max_steps: 600
    video_length: 241
    csv: ./data/movieroll/stage3_phase1_prompts.csv          # column: prompt
  long_dual_event:                          # phase 2
    max_steps: 600
    video_length: 721
    csv: ./data/movieroll/stage3_phase2_event_captions.csv   # columns: event_1, event_2
dmd:
  real_guidance_scale: 4.0                  # CFG of the teacher score
  context_noise: 0.15                       # noise on the clean history during rollouts
sampling:
  test_event_csv: ./data/movieroll/stage3_test_events.csv    # validation prompts
fsdp:
  shard_world_size: 8                       # GPUs per FSDP shard group; at most the GPU count (32 in the released runs)
sp_size: 4
logdir: ./outputs/movieroll_stage3_dmd
save_interval: 100
ema_start_step: 200                         # from here on, checkpoints also hold generator_ema.safetensors
```

Launch:

```bash
bash movieroll/scripts/launch/train_dmd.sh
```

To sample your own run, make a model folder from `movieroll/configs/models/distilled/t2v/config.json` and
`<logdir>/<exp_name>_<timestamp>/checkpoints/step-<N>/generator_ema.safetensors` renamed to `model.safetensors`, and
pass it to `--model`.

#### 🖼️ Image-to-Video Distillation (ACS)

Config: `movieroll/configs/stage3_dmd_rolling_acs.yaml`, the config above with these changes:

```yaml
dmd_phases:
  long_dual_event:
    max_steps: 900                 # phase 2 runs longer: 1500 steps in all
dmd:
  score_mode: acs                  # the window at the anchor frame is scored with the image as a condition
  sampling_context_noise: 0.0      # samples during training run without context noise
siglip_path: ckpts/siglip          # SigLIP image encoder of FLUX.1-Redux-dev
video:
  task_sampling:
    t2v_prob: 0.0                  # image-to-video only
sampling:
  test_event_csv: ./data/movieroll/stage3_test_events_i2v.csv   # needs an input_image column
fsdp:
  shard_world_size: 8              # must equal the total number of GPUs
```

Fill in `generator_ckpt` and the CSV paths as in ④, download the SigLIP encoder (FLUX.1-Redux-dev is a gated
repository, so accept its license on Hugging Face first), and launch:

```bash
hf download black-forest-labs/FLUX.1-Redux-dev --include 'image_encoder/*' 'feature_extractor/*' \
  --local-dir ./ckpts/siglip
CONFIG=movieroll/configs/stage3_dmd_rolling_acs.yaml bash movieroll/scripts/launch/train_dmd.sh
```

The released image-to-video model is `ckpts/movieroll/distilled/i2v`; sample it with `--input-image`. For your own
run, build the model folder as in ④, with `movieroll/configs/models/distilled/i2v/config.json`.
