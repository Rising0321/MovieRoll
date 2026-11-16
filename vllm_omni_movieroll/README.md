[中文文档](./README_CN.md)

<img src="../assets/movieroll-logo.png" alt="MovieRoll: And... Action!" width="800">

### × vLLM-Omni

`vllm_omni_movieroll` is a [vLLM-Omni](https://github.com/vllm-project/vllm-omni) plugin that runs both MovieRoll
models as a native diffusion pipeline, `MovieRollPipeline`, on one GPU or many: Distilled (5 steps, no CFG) and Base
(50 steps with AnnealCFG). It registers itself through vLLM-Omni's plugin entry points and patches nothing, and it
never imports the `movieroll` package at run time. It runs with vLLM-Omni's own defaults: regional `torch.compile`,
vLLM's parallel layers and the platform's attention backend.

## ✨ Highlights

- **Every way of running vLLM-Omni**: the offline `Omni` API, vLLM-Omni's own example scripts, and `vllm serve --omni`
  with `/v1/videos` (synchronous, asynchronous, and cancellable mid-video).
- **Text-to-video, image-to-video and multi-event videos**, with the same request rules as `generate_movieroll.py`.
- **vLLM-Omni's parallelism**: tensor parallelism, Ulysses, ring, AllGather-KV, CFG parallelism (Base), HSDP and CPU
  offload, plus PS-VAE to split the VAE decode across GPUs.
- **Fast.** On 8 GPUs, the denoising of a one-minute Distilled video takes 113 s (78 s with SageAttention), against
  150 s in the original repository.

## 🛠️ Installation

vLLM-Omni needs its own virtual environment, because its pinned torch and transformers versions conflict with
MovieRoll's. We use vLLM-Omni 0.32.0 and vLLM 0.32.0 with torch 2.13.0 (CUDA 12.9) and Python 3.12. The script also
installs this plugin in editable mode, which registers it with vLLM-Omni:

```bash
bash vllm_omni_movieroll/tools/build_venv.sh ./venv-vllm-omni python3.12
source ./venv-vllm-omni/bin/activate
```

Optionally, add SageAttention for faster sampling. The script builds a known-good commit (the Hopper kernel of the
`v2.2.0` release is broken on H20) for Hopper (sm90), and checks the build; set `TORCH_CUDA_ARCH_LIST` before running it
to target other GPUs:

```bash
bash vllm_omni_movieroll/tools/build_sageattention.sh ./venv-vllm-omni
```

## 🧱 Model Directory

vLLM-Omni loads diffusers-style model directories, so each MovieRoll model folder is assembled into one, once, inside
the vLLM-Omni environment. Pass the folder the way you would pass it to `generate_movieroll.py --model`: its
`config.json` decides the chain (Base or Distilled) and its sampling settings, its `model.safetensors` provides the
weights. The shared components are taken from the `ckpts/` layout of the
[main README](../README.md#setup) unless you point elsewhere (`--vae`, `--text-encoder`, ...). The
transformer and Qwen2.5-VL are symlinked (`--copy` copies them); the glyph encoder and the VAE are converted.

```bash
D=vllm_omni_movieroll/examples/offline_inference/movieroll
python $D/download_movieroll.py --model ckpts/movieroll/distilled/t2v --out ./movieroll-distilled
python $D/download_movieroll.py --model ckpts/movieroll/distilled/i2v --out ./movieroll-distilled-i2v
python $D/download_movieroll.py --model ckpts/movieroll/base --out ./movieroll-base
```

The image-to-video folder differs from the text-to-video one in its weights and in one setting: it adds no context
noise to the cached history. The directory built from it keeps that setting (`movieroll_overrides` in its
`model_index.json`, next to the class that names the chain: `MovieRollPipeline` for Base,
`MovieRollDistilledPipeline` for Distilled). To build from a bare checkpoint instead, pass
`--dit <file>.safetensors --sampling-mode base|distilled`; the chain's released settings apply then.

Classifier-free guidance is a setting of the model, not of the request. Base runs it with AnnealCFG, whose
per-frame scale moves between `cfg_min` and `cfg_max`; Distilled has none. To turn it off for Base, or to change
that range, set `use_cfg: false` or `cfg_min` / `cfg_max` in the folder's `config.json` before building (or in
`movieroll_overrides` of a built directory, or of a deploy YAML's `model_config`). A request's `guidance_scale` is
ignored.

## 🚀 Generate

With the offline Python API:

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="./movieroll-distilled")
out = omni.generate(
    {"prompt": "A red fox trots across a snowy clearing at dawn."},
    OmniDiffusionSamplingParams(num_frames=721, seed=930),
)
```

| To get | Set |
|---|---|
| several scenes | `extra_args={"events": ["First scene...", "Second scene..."]}`; the video is split evenly across the scenes. A scene left without a chunk (more scenes than chunks) is dropped with a warning on Base and is an error on Distilled |
| image-to-video | `"multi_modal_data": {"image": pil_image}` in the prompt dict (`input_reference` over HTTP), best with `./movieroll-distilled-i2v`. The video is 480×864 unless you set `height` / `width`; the image is scaled to cover it and centre-cropped |
| a different length or size | `num_frames` (`16k+1`, default 721), `height` / `width` (multiples of 16, default 480×864) |
| Base sampling settings | `num_inference_steps` (a multiple of 5, default 50), `negative_prompt` |
| SageAttention | `DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN` in the environment |

A `guidance_scale` other than vLLM-Omni's default of 1 is ignored with a warning, on either model: whether guidance runs
is the directory's `use_cfg`, and its strength is AnnealCFG's. Distilled always runs its fixed 5-step schedule and
ignores other `num_inference_steps` values with a warning. A model without guidance (Distilled, or Base with
`use_cfg: false`) also ignores a `negative_prompt` with a warning: it has no unconditional branch to use it on.

vLLM-Omni's shared example scripts, `text_to_video.py` and `image_to_video.py`, also work unmodified. They are not part
of the vLLM-Omni wheel: take them from `examples/offline_inference/` of a vLLM-Omni v0.32.0 source checkout. Start them
through `run_example.py`, which loads vLLM-Omni's plugins first and then runs the script with its own arguments:

```bash
python vllm_omni_movieroll/examples/offline_inference/movieroll/run_example.py \
  vllm-omni/examples/offline_inference/image_to_video/image_to_video.py \
  --model ./movieroll-distilled-i2v --image first_frame.jpg --prompt "..." --num-frames 81 --output eagle.mp4
```

The plugin declares MovieRoll's defaults to vLLM-Omni (480×864, 721 frames, 24 fps; 50 steps for Base, the 5-step
schedule for Distilled; no guidance scale, so the scripts send none) and that MovieRoll sizes the input image itself.
The scripts look both up before they create the engine, which is where vLLM-Omni loads plugins, so a script started
directly sees neither and falls back to its Wan2.2 presets: 1280×720, 40 steps and guidance 4.0 for text-to-video;
16 fps, 50 steps, guidance 5.0 and a size taken from the image for image-to-video, which also stretches the image to
any `--height` / `--width` you give it.

### Serving

```bash
MODEL=./movieroll-distilled bash vllm_omni_movieroll/examples/online_serving/movieroll/run_server.sh
curl -s http://127.0.0.1:8091/v1/videos/sync -F prompt="A red fox trots across a snowy clearing at dawn." \
  -F num_frames=81 -F seed=930 -o fox.mp4
```

A request without `size` gets 864x480 (width × height), one without `fps` 24 fps, and an image-to-video request's
`input_reference` reaches MovieRoll as it is, to be scaled to cover the video and centre-cropped. Several events go
in the `extra_params` form field as JSON, e.g. `-F 'extra_params={"events": ["...", "..."]}'`. vLLM-Omni draws a
random seed for a request that sends none, so pass `seed` to reproduce a video. `run_curl.sh`, in the same folder,
submits an asynchronous three-event job, polls it and downloads the video.
`DELETE /v1/videos/<id>` cancels a job at the next window. For SageAttention, add
`--diffusion-attention-backend SAGE_ATTN` to the server command.

## ⚡ Parallelism and PS-VAE

The `deploy/` folder has ready-made configs for common layouts: `movieroll_usp2.yaml`, `_usp4`, `_ring2`,
`_usp2_ring2`, `_tp2`, `_hsdp2` and `_base_cfg2_usp2`. You can also pass the flags directly: `--ulysses-degree`,
`--ring-degree` or `--allgather-degree`, `--tensor-parallel-size`, `--cfg-parallel-size 2` (Base with guidance on),
`--use-hsdp --hsdp-shard-size` and `--vae-patch-parallel-size`. The 16 attention heads must be divisible by the
tensor-parallel size times the Ulysses degree.

**PS-VAE.** `--vae-patch-parallel-size N`, with N equal to the number of GPUs, turns on MovieRoll's parallel streaming
decode: every GPU decodes a band of rows and the results are gathered. Unlike tiling, it computes the same decode as
a single GPU, up to floating-point rounding. vLLM-Omni still prints "VAE patch parallelism is NOT enabled for
MovieRollPipeline; ignoring" (or `MovieRollDistilledPipeline`); that refers to vLLM-Omni's own tiled decode, which
MovieRoll doesn't use, so the message is harmless.

**Environment.** Before any multi-GPU run and before any serving (the deploy YAMLs, the one-GPU one included, use the
multi-process executor), source `vllm_omni_movieroll/examples/movieroll_env.sh` (`run_server.sh` does this for you).
It sets two variables:

- `OMP_NUM_THREADS=16`. Otherwise vLLM sets it to the number of CPUs on the host, which slows down the per-frame
  video post-processing, by over a minute on a busy host.
- `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=14400`. Otherwise vLLM-Omni stops any request that runs longer than 600 s, and a
  one-minute Base video takes about 30 minutes even on 8 GPUs. If a request would still run past the limit, MovieRoll
  refuses it early with a 400 that names the value it needs, and the server keeps serving.

Measured on H20 at 81 frames, text-to-video (first request):

| Model | Layout | Time |
|---|---|---|
| Distilled | 1 GPU / Ulysses 2 / Ulysses 4 / TP 2 × Ulysses 2 | 63 s / 46 s / 35 s / 31 s |
| Base | 1 GPU / CFG 2 / CFG 2 × Ulysses 2 | 775 s / 412 s / 229 s |

PS-VAE lowers the decode's peak memory from 16.7 GB on one GPU to 8.5 GB per GPU on two and 4.6 GB on four.

## 🎞️ One Minute

A 60-second video (1441 frames at 24 fps, 480×864) on H20-3e. The times cover the rolling denoising loop only:

| Model | Layout | This plugin | Original repository |
|---|---|---|---|
| Distilled | 1 GPU | 711 s | 769 s |
| Distilled | Ulysses 8 + PS-VAE 8 | 113 s | 150 s |
| Distilled | the same, SageAttention | 78 s | – |
| Base | CFG 2 × Ulysses 4 + PS-VAE 8 | 1830 s | 2224 s |
| Base | the same, SageAttention | 1192 s | – |

The frames stay sharp to the end of the minute. Remember to source `movieroll_env.sh` for multi-GPU runs and for serving.

## ✅ Features

**Supported, and verified on the released weights:** Distilled and Base; text-to-video, image-to-video and multi-event;
`torch.compile` and `--enforce-eager`; tensor parallelism 2 and 4; Ulysses 2, 4 and 8; ring 2; AllGather-KV 2; CFG
parallelism 2 (Base); HSDP 2 and 4; PS-VAE; model-level and layer-wise CPU offload; the `FLASH_ATTN`, `TORCH_SDPA` and
`SAGE_ATTN` attention backends; 721-frame and 1441-frame videos; offline `Omni`, the shared example scripts and `vllm
serve --omni` (sync, async, cancellation); warm-up and the pipeline profiler.

Everything below is refused with a clear error before any weights are loaded:

| Refused | Why |
|---|---|
| FP8 (`--quantization fp8`) | on Distilled its video is only 15.0 dB (PSNR) from `generate_movieroll.py`'s, a visibly different video, and it is no faster |
| pipeline parallelism | a video is a strictly sequential chain of window passes, so the stages would only wait for each other |
| Cache-DiT, TeaCache | every call of the rolling window sees a different window with per-frame timesteps, so nothing can be reused |
| step execution, streaming output | would need the pipeline split into per-step methods; not in this release |
| LoRA | no MovieRoll LoRA exists |
| `SAGE_ATTN_3` | vLLM-Omni allows it on Blackwell GPUs only |
| VAE tiling and slicing | tiles are blended, which changes the video; the streaming decoder already keeps memory bounded |
| non-strict Ulysses (`ulysses_mode`), ring together with allgather, a `--vae-patch-parallel-size` other than 1 or the number of GPUs, a `flow_shift` other than the checkpoint's | each would change or break the rolling computation |
| distributed layer-wise offload, expert parallelism, `text_encoder_tp_size`, paged diffusion KV, quantized attention K/V, `max_num_seqs > 1` | not validated, or not applicable to MovieRoll (no MoE layers; one request is one video) |

vLLM-Omni's interactive streaming playback (`/v1/realtime/video` with `streaming_buffer_seconds`) is not available
either. It is a vLLM-Omni serving feature, not a MovieRoll setting, and vLLM-Omni itself refuses such a request,
because `MovieRollPipeline` is not an interactive chunked pipeline.

## 📜 License

This folder is covered by MovieRoll's [license](../LICENSE). Code adapted from Apache-2.0 projects (vLLM-Omni,
diffusers, ring attention) keeps its original notice and is listed in [NOTICE](./NOTICE).
