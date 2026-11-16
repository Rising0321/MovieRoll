[中文文档](./README_CN.md)

<img src="../assets/movieroll-logo.png" alt="MovieRoll: And... Action!" width="800">

### × SGLang Diffusion

`sglang_movieroll` runs both MovieRoll models as native [SGLang Diffusion](https://github.com/sgl-project/sglang)
(`sglang.multimodal_gen`) pipelines, on one GPU or many: Distilled (5 steps, no CFG) and Base (50 steps with
AnnealCFG). Nothing in SGLang is patched. The package registers its pipelines with SGLang and then starts SGLang's own
CLI, server or `DiffGenerator`. It never imports the `movieroll` package at run time, so it doesn't need a
FlashAttention build either.

## ✨ Highlights

- **Every way of running SGLang**: the offline Python API, the `sglang` CLI and the HTTP server (`/v1/videos`).
- **Text-to-video, image-to-video and multi-event videos**, with the same request rules as `generate_movieroll.py`.
- **SGLang's parallelism**: Ulysses, ring, hybrid USP, tensor parallelism, CFG parallelism (Base) and FSDP. The VAE
  decode is split across GPUs with PS-VAE (`--vae-sp`).
- **Fast.** On 8 GPUs, the denoising of a one-minute Distilled video takes 136 s (99 s with SageAttention), against
  150 s in the original repository.

## 🛠️ Installation

SGLang needs its own virtual environment, because its pinned torch and transformers versions conflict with
MovieRoll's. We use SGLang 0.5.10.post1, the last release built for CUDA 12, with Python 3.12. From the repository
root:

```bash
bash sglang_movieroll/tools/build_venv.sh ./venv-sglang python3.12
source ./venv-sglang/bin/activate
export PYTHONPATH=$PWD        # so that `sglang_movieroll` can be imported
```

Optionally, add SageAttention for faster sampling. The script builds a known-good commit (the Hopper kernel of the
`v2.2.0` release is broken on H20) for Hopper (sm90); set `TORCH_CUDA_ARCH_LIST` before running it to target other
GPUs. The check prints `OK` or `BROKEN`:

```bash
bash sglang_movieroll/tools/build_sageattention.sh ./venv-sglang
python -m sglang_movieroll.tools.check_sageattention
```

## 🧱 Model Directory

SGLang loads diffusers-style model directories, so each MovieRoll model folder is assembled into one, once. Pass the
folder the way you would pass it to `generate_movieroll.py --model`: its `config.json` decides the chain (Base or
Distilled) and its sampling settings, its `model.safetensors` provides the weights. The shared components are taken
from the `ckpts/` layout of the [main README](../README.md#setup) unless you point elsewhere
(`--vae-path`, `--text-encoder-path`, ...). Large files are symlinked; add `--copy` to copy them instead.

```bash
python -m sglang_movieroll.tools.assemble_model_dir --model ckpts/movieroll/distilled/t2v --out ./movieroll-distilled
python -m sglang_movieroll.tools.assemble_model_dir --model ckpts/movieroll/distilled/i2v --out ./movieroll-distilled-i2v
python -m sglang_movieroll.tools.assemble_model_dir --model ckpts/movieroll/base --out ./movieroll-base
```

The image-to-video folder differs from the text-to-video one in its weights and in one setting: it adds no context
noise to the cached history. The directory built from it keeps that setting (`movieroll_overrides` in its
`model_index.json`), and the context noise in use is logged when the model loads. To build from a bare
checkpoint instead, pass `--dit-weight <file>.safetensors --sampling-mode base|distilled`; the chain's released
settings apply then.

Classifier-free guidance is a setting of the model, not of the request. Base runs it with AnnealCFG, whose
per-frame scale moves between `cfg_min` and `cfg_max`; Distilled has none. To turn it off for Base, or to change
that range, set `use_cfg: false` or `cfg_min` / `cfg_max` in the folder's `config.json` before assembling (or in
`movieroll_overrides` of an assembled directory). A request's `guidance_scale` is ignored.

## 🚀 Generate

On one GPU:

```bash
python -m sglang_movieroll.examples.offline_generate --model ./movieroll-distilled \
  --prompt "A red fox trots across a snowy clearing at dawn." --num-frames 81 --output-dir out
```

Common options:

| To get | Add |
|---|---|
| several scenes | `--event "First scene..." --event "Second scene..."`; the video is split evenly across the scenes |
| image-to-video | `--image ./first_frame.jpg`, best with `--model ./movieroll-distilled-i2v`. The video is 480×864 unless you set `--height` / `--width`; the image is scaled to cover it and centre-cropped |
| more GPUs | `--num-gpus 4 --ulysses-degree 4` |
| Base on two GPUs | `--model ./movieroll-base --num-gpus 2 --enable-cfg-parallel` (one GPU per guidance branch; refused for Distilled) |
| faster attention | `--attention-backend sage_attn` |
| a long video on one GPU | `--keep-offload` (keeps SGLang's CPU offload, which frees room for the VAE decode) |

If you write your own launcher, call `sglang_movieroll.register()` at the top level of the script. SGLang starts its
workers as spawned processes that re-import the script, so the call has to run in each of them.

### Serving

`python -m sglang_movieroll.cli` is the `sglang` command with MovieRoll registered, so `generate` and `serve` both
work:

```bash
python -m sglang_movieroll.cli serve --model-path ./movieroll-distilled --port 30000 --num-gpus 2
curl 127.0.0.1:30000/v1/videos -H 'Content-Type: application/json' \
  -d '{"prompt": "A red fox trots across a snowy clearing at dawn.", "num_frames": 81}'
# returns {"id": ...}; poll GET /v1/videos/<id> until it is completed, then download GET /v1/videos/<id>/content
```

`width` and `height` are optional and default to 480×864. An image-to-video request sends its image as
`input_reference`, either as a multipart upload or as a path in the JSON body.

Two things to know about HTTP requests:

- **Always send `num_frames`** (a count of the form `16k+1`). The endpoint turns a duration into `fps × seconds`,
  which is rarely a valid MovieRoll length; such a request gets a 400 that suggests the nearest valid counts.
- **Multi-event videos send their events in the prompt**, as JSON:
  `{"events": ["First scene...", "Second scene..."]}`. SGLang has no other channel that carries model-specific fields
  to the workers.

## ⚡ Parallelism

| Flag | What it does |
|---|---|
| `--ulysses-degree U` | splits the window's tokens across GPUs and trades them for attention heads |
| `--ring-degree R` | passes the keys around a ring and merges the partial attentions (SGLang allows it with the `fa` and `sage_attn` backends) |
| `--ulysses-degree U --ring-degree R` | both at once (SGLang's hybrid USP) |
| `--tp-size T` | splits the heads and feed-forward layers; `T × U` must divide the 16 heads |
| `--enable-cfg-parallel` | Base only: the positive and negative branches run on separate GPUs |
| `--use-fsdp-inference` | shards the transformer blocks and gathers each one as it runs; not together with `--tp-size` |
| `--vae-sp` | PS-VAE: splits the VAE decode across the sequence-parallel group; on by default when there is one. A flag of the `sglang_movieroll.cli` command only: the offline example has no such option and always leaves it on |

Measured on H20 with Distilled text-to-video at 81 frames (the multi-GPU rows):

| Layout | GPUs | Time | Speed-up |
|---|---|---|---|
| one GPU | 1 | 62.0 s | 1.00× |
| `--ulysses-degree 2` | 2 | 36.4 s | 1.70× |
| `--ulysses-degree 4` | 4 | 29.4 s | 2.11× |
| `--tp-size 2 --ulysses-degree 2` | 4 | 29.2 s | 2.12× |
| `--use-fsdp-inference` | 2 | 38.7 s | 1.60× |
| `--enable-cfg-parallel`, Base | 2 | 412.1 s (796.2 s on one GPU) | 1.93× |

PS-VAE also reduces the peak memory of the decode, from 56.3 GiB on one GPU to 30.3 GiB per GPU on two and 17.3 GiB on
four, which is what makes long videos fit.

## 🎞️ One Minute

A 60-second video (1441 frames at 24 fps, 480×864) on H20-3e. The times cover the rolling denoising loop only:

| Model | Layout | This adapter | Original repository |
|---|---|---|---|
| Distilled | 1 GPU | 736 s | 769 s |
| Distilled | `--ulysses-degree 8` | 136 s | 150 s |
| Distilled | `--ulysses-degree 8`, SageAttention | 99 s | – |
| Base | `--enable-cfg-parallel --ulysses-degree 4` (8 GPUs) | 1882 s | 2224 s |
| Base | the same, SageAttention | 1198 s | – |

The frames stay sharp for the whole minute. Distilled one-minute videos on one GPU need `--keep-offload`.

**The 100-minute limit.** SGLang 0.5.10.post1 abandons any request that runs longer than 100 minutes. A one-minute
Base video on one GPU would take about 4 hours, so use the 8-GPU layout above for it. The adapter doesn't wait for the
timeout: two chunks into the video, it estimates the total time and, if the request can't finish in time, refuses it
with that estimate.

## ✅ What Is Supported, and What Is Refused

Anything SGLang offers that MovieRoll can't honour is refused with a reason, either at start-up or when the request is
built, and nothing fails halfway through a video. Three request fields are ignored with a warning instead of
refused: `generator_device` (SGLang's video endpoint sends one with every request; each chain draws its noise on its
own device), `guidance_scale` when it isn't SGLang's default of 1 (whether guidance runs is the model's `use_cfg`, and
its strength is AnnealCFG's), and the negative prompt of a model without guidance (Distilled, or Base with
`use_cfg: false`).

**Supported:** one GPU; Ulysses, ring, hybrid USP, tensor parallelism, CFG parallelism (Base), FSDP / HSDP and
`--vae-sp`; CPU offload and layer-wise DiT offload; warmup; profiling and `--perf-dump-path`; the `fa`, `sage_attn` and
`torch_sdpa` attention backends (SGLang allows `--ring-degree` only with `fa` and `sage_attn`);
`--enable-torch-compile` (it works but is slower, because the KV cache grows every round and triggers recompiles);
serving through `/v1/videos`.

| Refused | Why |
|---|---|
| `--vae-tiling` | spatial tiles are blended, which changes the video; `--vae-sp` splits the decode across GPUs without tiles instead |
| `--vae-slicing` | it splits a batch, and a MovieRoll request is one video, already decoded chunk by chunk |
| Cache-DiT, TeaCache | they reuse residuals across steps of one latent, but each step of a rolling window holds different chunks |
| `sage_attn_3` | its FP4 kernels need Blackwell GPUs |
| sparse attention backends | lossy, and they assume full-sequence self-attention, while MovieRoll attends to a cache prefix |
| `--enable-cfg-parallel` with Distilled, or with a Base directory set to `use_cfg: false` | there is no second guidance branch to put on the other GPU |
| `--transformer-weights-path`, `--enable-svdquant` (FP8, NVFP4, ModelSlim, SVDQuant) | there is no quantized MovieRoll checkpoint to load or to verify against |
| `--dit-precision` other than `bf16` | both chains were evaluated with a bf16 transformer, and MovieRoll has never been run in another precision |
| LoRA | no MovieRoll LoRA exists to verify against |
| `comfyui_mode` (set by SGLang's ComfyUI integration) | it drives one latent per step, which a rolling window doesn't have |
| `num_outputs_per_prompt > 1`, `true_cfg_scale`, `guidance_rescale`, `cfg_normalization`, `boundary_ratio`, `return_trajectory_*` | not part of MovieRoll's sampler |

**Not available in SGLang 0.5.10.post1:** data parallelism, pipeline parallelism, cancelling a running video, and
progress reports over HTTP (the server logs each finished chunk instead). MovieRoll runs on SGLang's native backend
only: `--backend diffusers` has no MovieRoll pipeline to load.

## 📜 License

This folder is covered by MovieRoll's [license](../LICENSE). Code adapted from Apache-2.0 projects keeps its original
notice and is listed in [NOTICE](./NOTICE).
