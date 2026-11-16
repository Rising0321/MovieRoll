# vLLM-Omni adapter — environment, model directory, environment variables

One-off work, plus two environment variables that have to be set correctly **every** time an engine starts.

## 1. The virtual environment

vLLM-Omni pins a torch and a transformers that conflict with MovieRoll's, so it needs its own venv.

```bash
bash vllm_omni_movieroll/tools/build_venv.sh ./venv-vllm-omni python3.12
source ./venv-vllm-omni/bin/activate
```

Pinned: vLLM-Omni `0.32.0`, vLLM `0.32.0`, torch `2.13.0` (CUDA 12.9), Python 3.12, plus torchvision 0.28.0 and
torchaudio 2.11.0.

The script does four unusual things, each for a reason worth knowing because they are where installs break:

- **torch comes from the cu129 index**, not PyPI: on PyPI, torch >= 2.11 is a CUDA 13 build and needs a newer
  driver than many clusters have. A CUDA 13 build installs fine and then reports `torch.cuda.is_available()`
  as False.
- **vLLM's `+cu129` release wheel is installed with `--no-deps`**, then its dependency list is exported from
  `importlib.metadata` and installed — **except `llguidance`**. vLLM pins 1.7.x, whose only Linux wheel needs
  glibc 2.31; `llguidance==1.6.0` is installed instead as a binary. The cost is one unsatisfied pin that
  `pip check` will report against vllm.
- **`torchcodec==0.16.0+cu129`** with `--no-deps`: the PyPI 0.17 is a CUDA 13 build that fails to load, and
  vLLM-Omni imports it.
- **`pip install --no-deps -e vllm_omni_movieroll`** — this editable install *is* the registration. The entry
  points come from installed package metadata, so putting the directory on `PYTHONPATH` does **not** make the
  plugin discoverable.

`--native-reference` as a third argument adds what native `generate_movieroll.py` needs and builds
FlashAttention v2.8.1 from source (FA2 + hopper FA3), for same-venv comparison.

### SageAttention (optional)

```bash
bash vllm_omni_movieroll/tools/build_sageattention.sh ./venv-vllm-omni
```

Pinned to commit `d1a57a5` of `thu-ml/SageAttention` on `main`, built for Hopper (sm90) by default (set
`TORCH_CUDA_ARCH_LIST`, e.g. `"8.0;9.0"`, before running the script for other GPUs) — **not the `v2.2.0` tag**, whose sm90 kernel is
wrong on H20 and produces a broken video with no error. The build runs `tools/check_sageattention.py` itself:
it compares five kernel paths against SDPA at two shapes (one of them the real shape of a MovieRoll window
attending over history plus text); about 0.04 relative error is good, the threshold is 0.1, and the broken tag
scored 1.25.

Enable it per run with `DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN` (offline) or
`--diffusion-attention-backend SAGE_ATTN` (serving); both are vLLM-Omni 0.32.0's own switches. `SAGE_ATTN_3` is
refused (Blackwell only).

## 2. The two environment variables

```bash
source vllm_omni_movieroll/examples/movieroll_env.sh
```

It sets exactly two values, both as `${VAR:-default}` so an existing value is kept:

| Variable | Value | Why |
|---|---|---|
| `OMP_NUM_THREADS` | 16 | unset, vLLM's multiprocess executor sets it to the host's full CPU count, which slows the engine's per-frame video post-processing — more than a minute on a busy machine |
| `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` | 14400 | vLLM-Omni's mp/ray executor waits only this many seconds per `execute_model` (default **600**) and on timeout **stops every worker**. A one-minute base video takes about 30 minutes even on 8 GPUs. |

**Source it before the engine starts.** The timeout is read twice: by vLLM-Omni's executor, once, when its module is
first imported as the first engine of the process starts (`multiproc_executor.py` / `ray_executor.py` keep it in a
module constant), and by MovieRoll's ComputeBudget when the pipeline is constructed. Exporting it later has no effect,
and changing `os.environ` before a second `Omni` in the same process only moves MovieRoll's projection, not the
executor's limit. `run_server.sh` sources it for you; offline runs and hand-built `Omni` objects do not.

## 3. The model directory

Built once per released MovieRoll model folder, inside the vLLM-Omni venv (the script loads the glyph encoder and
the VAE with that venv's transformers and diffusers). The script takes the folder exactly as
`movieroll/generate.py --model` does:

```bash
D=vllm_omni_movieroll/examples/offline_inference/movieroll
python $D/download_movieroll.py --model ckpts/movieroll/distilled/t2v --out ./movieroll-distilled
python $D/download_movieroll.py --model ckpts/movieroll/distilled/i2v --out ./movieroll-distilled-i2v
python $D/download_movieroll.py --model ckpts/movieroll/base --out ./movieroll-base
```

The folder's `config.json` gives the chain (`sampling_mode`) and its settings; its `model.safetensors` is the DiT. The
shared components default to the main README's `ckpts/` layout (`--vae ckpts/vae`, `--text-encoder
ckpts/text_encoder/llm`, `--byt5 ckpts/text_encoder/byt5-small`, `--glyph-ckpt`, `--font-json` and `--color-json`
under `ckpts/text_encoder/Glyph-SDXL-v2/`); pass the flags to point elsewhere. The older form,
`--dit <file>.safetensors --sampling-mode base|distilled`, still works and uses the chain's built-in settings.

`--out` must not exist. The directory is built under a hidden `.<name>.partial` next to `--out` and renamed only when
complete, so a failed conversion leaves nothing in the way of the next attempt. Symlinks by default, `--copy` to copy
(Qwen2.5-VL once: `tokenizer/` is a link to `text_encoder/`). One directory per folder, and the chain is fixed by it —
a request cannot switch chains.

```
model_index.json   _class_name = MovieRollPipeline (base) | MovieRollDistilledPipeline (distilled), sampling_mode,
                   movieroll_overrides = what the folder sets differently ({"context_noise": 0.0} for distilled/i2v)
transformer/       config.json + the DiT safetensors (symlinked; names mapped at load time)
text_encoder/      Qwen2.5-VL (symlinked); tokenizer/ -> text_encoder/
text_encoder_2/ tokenizer_2/  converted once: byt5-small + colour/font tokens + Glyph-SDXL-v2 weights
vae/               converted once: diffusers config + renamed safetensors
scheduler/         FlowMatchEulerDiscreteScheduler, shift 5
```

The two class names matter: vLLM-Omni looks a model's serving and example defaults up by that name, so the Base and
Distilled directories get their own (50 steps, or the five-step ladder). A directory built before
this split, with `MovieRollPipeline` on a distilled chain, is refused at load with a request to rebuild it.

Conversions only rename or merge tensors — no dtype changes — and validate on the spot: every DiT key must map,
the VAE must have no missing/unexpected keys and no shape change, and the byT5 encoder loads with `strict=True`. A
folder whose `config.json` differs from the built-in chain outside the whitelist below (size, geometry, text
templates) is refused with the key named.

### Overriding sampling parameters

`movieroll_overrides` is **not** a request field. It goes in `model_index.json` (the builder writes the folder's
differences there), or in a deploy YAML, in the stage entry next to `devices`:
`model_config: {movieroll_overrides: {kv_recompute_threshold: 12}}`. Whitelist: `kv_recompute_enabled`, `kv_recompute_threshold`, `kv_recompute_recent_k`,
`flow_shift`, `use_cfg`, `cfg_min`, `cfg_max`, `context_noise_enabled`, `inference_context_noise`,
`denoising_step_list`, `timestep_shift`, `context_noise`, `vae_stream_decode_max_chunk_size`. Anything else is an
error, and the distilled chain additionally requires `window_size == len(denoising_step_list)`.

Base guidance lives here, not in the request: `use_cfg` (base only, a boolean) turns classifier-free guidance on or
off — `false` runs one branch per window instead of two and makes `cfg_parallel_size: 2` a start-up error — and
`cfg_min` / `cfg_max` set AnnealCFG's per-frame strength. A request's `guidance_scale` is ignored. The step count is
not a directory setting: send `num_inference_steps` with the request (the server fills it in for every request, so a
directory value would never be used). A folder `config.json` that still carries `guidance_scale` is refused as an
unknown key; replace it with `use_cfg`.

Careful with `kv_recompute_*`: the position grid is `max_num_latent_frames = 120`, and raising the threshold can
push a window past it (the transformer raises and names that field).
