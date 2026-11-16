# SGLang adapter — environment and model directory

One-off work. Both steps are needed before anything can run.

## 1. The virtual environment

SGLang pins a torch and a transformers that conflict with MovieRoll's, so the adapter needs its **own** venv.
It does not import the `movieroll` package at run time, which also means **no FlashAttention source build is
needed** for the adapter itself.

```bash
bash sglang_movieroll/tools/build_venv.sh ./venv-sglang python3.12
source ./venv-sglang/bin/activate
export PYTHONPATH=$PWD          # from the repository root; sglang_movieroll is not an installed package
```

Pinned: SGLang `0.5.10.post1` (`sglang[diffusion]`), Python 3.12, torch `2.9.1+cu128`. The script does not
install torch separately — SGLang pins 2.9.1 and the PyPI build of that version is the CUDA 12.8 one.
`0.5.10.post1` is the last release whose `sglang-kernel` is built for CUDA 12; 0.5.11 needs CUDA 13 and a newer
driver. The script's last act is `import torch, sglang, sglang_movieroll` as a self-check.

Add `--native-reference` as a third argument to also install what native `generate_movieroll.py` needs
(accelerate, peft>=0.18, omegaconf, loguru, imageio, imageio-ffmpeg, qwen-vl-utils, tiktoken, psutil, einops, ninja)
and build FlashAttention `v2.8.1` from source, FA2 plus the hopper FA3, for sm90 only. That exists for one purpose: comparing the
adapter against native **in the same venv** (see `run.md`).

### SageAttention (optional, ~1.4x on the denoise)

```bash
bash sglang_movieroll/tools/build_sageattention.sh ./venv-sglang
python -m sglang_movieroll.tools.check_sageattention       # must print OK
```

The script pins commit `d1a57a5` of `thu-ml/SageAttention` on `main` and builds for Hopper (sm90) by default; set
`TORCH_CUDA_ARCH_LIST` (e.g. `"8.0;9.0"`) before running it for other GPUs. **Do not use the `v2.2.0` tag**: its sm90
kernel is wrong on H20 — which is exactly the kernel `sageattn` selects on Hopper — and produces an all-NaN
video with no error. The checker compares five kernel paths against torch SDPA; a good build scores about 0.04
relative error, the threshold is 0.1, and the broken tag scored 1.25. Enable it per run with
`--attention-backend sage_attn`.

Supported attention backends are only `fa` (the Hopper default), `sage_attn` and `torch_sdpa`.

## 2. The model directory

SGLang loads diffusers-style directories with its own component loader, so each released MovieRoll model folder has
to be assembled into one. The tool takes the folder exactly as `movieroll/generate.py --model` does:

```bash
python -m sglang_movieroll.tools.assemble_model_dir --model ckpts/movieroll/distilled/t2v --out ./movieroll-distilled
python -m sglang_movieroll.tools.assemble_model_dir --model ckpts/movieroll/distilled/i2v --out ./movieroll-distilled-i2v
python -m sglang_movieroll.tools.assemble_model_dir --model ckpts/movieroll/base --out ./movieroll-base
```

The folder's `config.json` gives the chain (`sampling_mode`) and its settings; its `model.safetensors` is the DiT. The
shared components default to the main README's `ckpts/` layout (`--vae-path ckpts/vae`, `--text-encoder-path
ckpts/text_encoder/llm`, `--byt5-google-path ckpts/text_encoder/byt5-small`, `--byt5-ckpt-path`, `--font-path` and
`--color-path` under `ckpts/text_encoder/Glyph-SDXL-v2/`); pass the flags to point elsewhere. The older form,
`--dit-weight <file>.safetensors --sampling-mode base|distilled`, still works and uses the chain's built-in settings.

`--out` must not exist. Every source is checked for existence before anything is written, and the directory is built
under a hidden `.<name>.partial` next to `--out` and renamed only when complete, so a failed build leaves nothing in
the way. The default is to symlink the big files; `--copy` copies instead (Qwen2.5-VL once: `tokenizer/` is a link to
`text_encoder/`) — with symlinks, moving or deleting `ckpts/` later breaks the directory.

What it produces:

```
model_index.json      _class_name = MovieRollPipeline | MovieRollDistilledPipeline, seven module entries, and
                      movieroll_overrides: what the folder's config.json sets differently from the built-in chain
                      ({"context_noise": 0.0} for distilled/i2v, {} for the other two)
transformer/          config.json (54 layers, 16 heads x 128, rope_axes_dim [16,56,56], rope_theta 256,
                      text_embed_dim 3584, byt5_hidden_dim 2048, plus the streaming geometry) + the DiT
                      weights symlinked as diffusion_pytorch_model.safetensors (renamed at load time)
text_encoder/         -> the Qwen2.5-VL directory; tokenizer/ -> text_encoder/
text_encoder_2/ tokenizer_2/    written out: byT5-small + Glyph-SDXL-v2 weights and the colour/font tokens,
                                as a standard T5EncoderModel (load_state_dict must be clean)
vae/                  written out: MovieRoll's VAE keys renamed to diffusers AutoencoderKLHunyuanVideo15
                      (pure renaming, key count must match), config carries stream_chunk_frames = 4
scheduler/            FlowMatchEulerDiscreteScheduler, shift 5.0, num_train_timesteps 1000
```

A folder may differ from the built-in chain only in what MovieRoll's stages alone read, from the chain the pipeline
builds once the directory is loaded: `context_noise`, `context_noise_enabled`, `inference_context_noise`, the three
`kv_recompute_*` keys, `use_cfg` (base only: `false` runs one branch instead of two), `cfg_min`, `cfg_max`
(`rolling/profiles.py`, `OVERRIDABLE`). Anything else that differs — size, steps, shift, geometry — is refused by the
tool with the key named, because SGLang fixes those before the directory is read. A folder `config.json` still
carrying the old `guidance_scale` key is refused as an unknown key; replace it with `use_cfg`. When the model loads,
the worker logs `<chain> chain, context noise <sigma>` and the overrides it applied; check that line for an
image-to-video directory (`context noise 0`).

## 3. Where the numbers come from

`sglang_movieroll/rolling/profiles.py` is the single source of the built-in sampling parameters, and they mirror the
released MovieRoll values: 480x864, 24 fps, `chunk_size` 4, `window_size` 5, `max_num_latent_frames` 120,
`full_history_block_num` 6, KV recompute on with threshold 18 and recent_k 3, VAE stream decode chunk 4; base
adds 50 steps / flow shift 5.0 / `use_cfg` on, AnnealCFG range 1.0–4.0 / context noise 0.05; distilled adds the five-step
ladder, shift 5.0 and context noise 0.15.

These are the values the released checkpoints were evaluated with, not tunable defaults. The one released folder that
differs is `distilled/i2v`, whose context noise is 0.

The three config layers, for when something needs changing:

- `configs/models/` — component architecture (DiT dims, the weight-name mapping, the two text encoders, the
  VAE's compression factors). The model directory's `config.json` overrides these at load.
- `configs/pipeline_configs/` — the static pipeline config plus the stage callbacks (`prepare_sigmas`,
  `tokenize_prompt`, `post_decoding`, …), and `movieroll_overrides`, which the pipeline fills from the model
  directory when it loads. The distilled config differs from base in two places that matter:
  `generator_device="cpu"` (the distilled chain draws noise on the CPU — changing the device changes the random
  stream) and the five-step ladder. The shift is 5.0 in both.
- `configs/sample/` — request defaults and validation, including `adjust_frames=False` so SGLang does not round
  the frame count to a multiple of the GPU count. Requests are validated client-side, so a bad one fails with a
  readable error before it reaches a worker.
