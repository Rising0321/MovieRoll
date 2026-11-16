# SGLang adapter — running it

## Registration, and the one rule about it

`sglang_movieroll.register()` is the only entry point into the adapter. It is idempotent, and it does three
things in a fixed order: register the component classes, then run SGLang's **own** pipeline discovery, then add
the MovieRoll pipelines and configs.

The order matters: SGLang's discovery returns immediately if the pipeline registry is non-empty, so registering
MovieRoll first would silently leave every built-in pipeline unregistered.

> **`register()` must be called at the top level of the launching script**, outside `if __name__ == "__main__"`.
> SGLang spawns its GPU workers; each one is a fresh interpreter that re-imports the launch script as
> `__mp_main__`, so every process has to register for itself. `cli.py` and `examples/offline_generate.py` both
> do it this way, before constructing `ServerArgs`.

## Offline generation

```bash
python -m sglang_movieroll.examples.offline_generate \
  --model ./movieroll-distilled \
  --prompt "..." \
  --num-frames 81 --output-dir outputs
```

Arguments:

| Group | Flags |
|---|---|
| model | `--model` (required, the assembled directory) |
| prompt | `--prompt` / `--prompt-file` / `--event` (repeatable) / `--event-file` (repeatable) / `--negative-prompt` (base only). All four empty is an error. |
| video | `--image` (i2v; use a directory built from `distilled/i2v`), `--num-frames` (default **721**), `--height`, `--width` (default 480x864, image-to-video included: the image is scaled to cover the size and centre-cropped), `--seed` (default 930) |
| parallel | `--num-gpus`, `--tp-size`, `--ulysses-degree`, `--ring-degree`, `--sp-degree`, `--enable-cfg-parallel`, `--use-fsdp-inference` |
| memory/speed | `--attention-backend`, `--enable-torch-compile`, `--dit-layerwise-offload`, `--warmup`, `--keep-offload` |
| output | `--output-dir` (default `outputs`), `--output-file-name`, `--dump-dir` (also saves the decoded `[1,3,T,H,W]` tensor in `[0,1]` as `.pt`), `--profile`, `--perf-dump-path`, `--master-port` |

Multi-event is passed by packing the events into JSON inside the prompt — SGLang has no other channel for
model-specific fields. A prompt whose first non-whitespace character is `{` is therefore parsed as MovieRoll JSON;
plain text must not begin with a brace.

Running several offline jobs on one machine needs a distinct `--master-port` each (the scheduler port is
`master_port + 1`).

## Serving

```bash
python -m sglang_movieroll.cli serve --model-path ./movieroll-distilled --port 30000 --num-gpus 2
```

`python -m sglang_movieroll.cli` is SGLang's own CLI with MovieRoll registered; the subcommands are SGLang's
(`generate`, `serve`). Note the flag is `--model-path` here and `--model` in the offline example.

```
POST /v1/videos                -> {"id": ...}   JSON or multipart; include num_frames
GET  /v1/videos/<id>           -> the job's status
GET  /v1/videos/<id>/content   -> the mp4, once the job is completed
```

`width` / `height` (or `size`) are optional and default to 480x864. Image-to-video sends the image as
`input_reference` (a multipart upload, or a path in a JSON body).

**Always send `num_frames` explicitly.** The endpoint otherwise derives it from a duration as `fps x seconds`,
which is almost never a legal frame count, and returns 400 with the nearest legal values. Legal counts are
`16k + 1`.

Multi-event over HTTP goes in the prompt as JSON: `{"events": [...]}`. The video is split evenly across the
events; the JSON takes no other key.

`generator_device` in the request is ignored with a warning — the chain's own device wins (CPU for distilled).
So is `guidance_scale` (when it is not SGLang's default of 1): whether classifier-free guidance runs is the model
directory's `use_cfg`, and its per-frame strength is AnnealCFG's `cfg_min` / `cfg_max`, also set in the directory.

## Frame counts and geometry

The same arithmetic as native MovieRoll, enforced by `check_video_length`:

```
latent_length = (num_frames - 1) // 4 + 1,   latent_length % 4 == 1   =>   num_frames = 16k + 1
legal, e.g.: 17, 33, 49, 81, 241, 721, 1441
```

The error message lists the nearest legal counts. Base additionally requires `num_inference_steps` to be a
positive multiple of `window_size` (5).

## Multi-GPU

| Mode | Flag | Notes |
|---|---|---|
| Tensor parallel | `--tp-size` | SGLang's own parallel linear layers |
| Ulysses | `--ulysses-degree` | token↔head all-to-all |
| Ring | `--ring-degree` | adapter's own log-sum-exp ring merge — SGLang's built-in ring does not understand the replicated text block or keys without a matching query. SGLang itself accepts the ring only with the `fa` or `sage_attn` backend (and picks `fa` when none is given) |
| Hybrid USP | both | requires `sp_degree == ulysses x ring` |
| CFG parallel | `--enable-cfg-parallel` | **base with guidance on only**: refused at start-up for a distilled directory, and once the weights are loaded for a base directory whose `use_cfg` is `false`. The two ranks' KV caches are bit-identical because the history carries no text, so no communication is needed. |
| FSDP | `--use-fsdp-inference` | shards the double blocks; **cannot be combined with `--tp-size > 1`** |
| PS-VAE | `--vae-sp` | default on when there is a sequence-parallel group; row-shards the decode. A `sglang_movieroll.cli` flag only: `offline_generate` has no such option and always leaves it on |

Constraint: 16 attention heads must be divisible by `tp_size x ulysses_degree`.

Not supported: data parallelism (SGLang itself refuses it for diffusion) and pipeline parallelism (absent in
this release).

Measured on H20 (the one-minute runs on H20-3e), distilled t2v 81 frames: 1 GPU 62.0 s; `--ulysses-degree 2` 36.4 s; `--ulysses-degree 4`
29.4 s; `--tp-size 2 --ulysses-degree 2` 29.2 s; `--use-fsdp-inference` on 2 GPUs 38.7 s. Base 81 frames:
1 GPU 796.2 s, `--enable-cfg-parallel` on 2 GPUs 412.1 s. One-minute video (1441 frames), denoise loop only:
distilled 8 GPUs with `--ulysses-degree 8` 136 s (99 s with SageAttention); base 8 GPUs with
`--enable-cfg-parallel --ulysses-degree 4` 1882 s (1198 s with SageAttention). PS-VAE decode peak: 56.3 GiB on
1 GPU, 30.3 GiB/GPU on 2, 17.3 GiB/GPU on 4.

Also run on the released weights at 81 frames, above the 20 dB bar against native but without published timings:
`--ring-degree 2`, `--ulysses-degree 2 --ring-degree 2`, `--ulysses-degree 4 --ring-degree 2`, `--ulysses-degree 8`,
`--tp-size 2`, and `--dit-layerwise-offload` on one GPU. Not measured: HSDP beyond FSDP over 2 ranks, and
`--dit-layerwise-offload` together with a parallel layout.

## Framework limits, and what to do about them

**The 100-minute client timeout.** SGLang's scheduler client abandons a request after `RCVTIMEO` expires
(`CLIENT_TIMEOUT_S = 6000`). Once, when the second chunk is done, the adapter extrapolates the total runtime from
the pace so far and **refuses** with an estimate and a suggestion if it exceeds the limit, rather than letting the
request die at minute 100. Consequence:
base at one minute on a single GPU — roughly four hours — is refused immediately. Add GPUs
(`--ulysses-degree`, plus `--enable-cfg-parallel` for base), or generate in segments.

**No cancellation and no HTTP progress** in this SGLang release. The server log prints one line per chunk
(`chunk d / d denoised`); offline runs also show a progress bar. For finer timing use `--profile` /
`--perf-dump-path`.

**One video per request.** `num_outputs_per_prompt > 1` is refused; batch size is fixed at 1.

**`--enable-torch-compile` works but is slower here**: the KV cache grows every round, so it recompiles
repeatedly.

**Refused at start-up**, before any weight is read: `sage_attn_3` and any attention backend outside `fa`,
`sage_attn`, `torch_sdpa` (sparse ones included), FSDP + TP, `--enable-cfg-parallel` on a distilled directory,
`--dit-precision` other than `bf16`, `--transformer-weights-path`
and `--enable-svdquant` (pre-quantized FP8 / NVFP4 / ModelSlim / SVDQuant weights; on Hopper SGLang itself already
rejects `--enable-svdquant`), `--vae-tiling`, `--vae-slicing`, `comfyui_mode` (set by SGLang's ComfyUI integration,
not a command-line flag), the cache-DiT env flag, and LoRA (`--lora-path`, and `/v1/set_lora` later). **Refused once
the weights are loaded**, before any request runs: `--enable-cfg-parallel` on a base directory with `use_cfg: false`
(its `model_index.json` is read in `initialize_pipeline`). **Refused when the request is parsed**: `num_outputs_per_prompt > 1`,
`true_cfg_scale`, `guidance_rescale`, `cfg_normalization`, `boundary_ratio`, `enable_teacache`,
`return_trajectory_*`.

Distilled-specific: `num_inference_steps` other than 5 is refused (its ladder is fixed). A negative prompt is ignored with a warning whenever the model runs without guidance (distilled, or base with
`use_cfg: false`). Base warns and drops events beyond the chunk count; distilled refuses.

## Comparing against native

The point of `--native-reference` in `build_venv.sh`. **Compare only against native running in the same venv**:
the bf16 CPU `randn` stream is not the same across torch versions (torch 2.13, the vLLM-Omni venv's, draws a
different stream from 2.7.1 and 2.9.1), so native in one venv can differ from native in another by far more than
any adapter bug would — different initial noise, different content.

```bash
# adapter
python -m sglang_movieroll.examples.offline_generate --model ./movieroll-distilled \
  --prompt "..." --num-frames 81 --seed 930 --dump-dir cmp/adapter
# native, same venv, same seed / prompt / size / frame count
python generate_movieroll.py --model ckpts/movieroll/distilled/t2v \
  --prompt "..." --video-length 81 --seed 930 --sp-size 1 \
  --output-type pt --output-dir cmp/native
```

For image-to-video, build the directory from `distilled/i2v`, pass `--image` to the adapter and `--model
ckpts/movieroll/distilled/i2v --input-image` to native, and give both the same size (480x864 is the adapter's
default; state it to native explicitly). Both write `[0, 1]` tensors. Compute PSNR/SSIM yourself — there is no
comparison script in the repository, and this release ships no tests. Getting the value range wrong is a classic
~6 dB error.

Reading the result: the practical acceptance bar for the adapters is **about 20 dB against same-venv native
plus a visual check**, with a negative control (different seed or content) clearly lower. The README publishes
only timings, so treat specific dB figures as context rather than contract. For **long** videos, PSNR against
native drops into the teens — that is trajectory divergence accumulated over dozens of chunks between two
correct implementations, not damage; judge those by eye.

## Failure modes

| Signal | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: sglang_movieroll` | not an installed package | `export PYTHONPATH=$PWD` from the repository root |
| works in the main process, workers cannot find the pipeline or the arch class | `register()` not at module top level | move it out of `main()` / `__main__` |
| every built-in SGLang pipeline missing | custom registration ran before SGLang's discovery | keep the order in `registry.py` |
| `video_length must be 4n+1 ...` / `not 1 modulo chunk_size=4`; HTTP 400 | frame count | use `16k+1`; always send `num_frames` over HTTP |
| `MovieRollPipeline needs a MovieRollPipelineConfig ... but sglang resolved ...` | the directory name collides with another model family | `--model-id MovieRoll-Base` / `MovieRoll-Distilled` (a `sglang_movieroll.cli` flag), or rename the directory (the only fix for `offline_generate`, which has no `--model-id`) |
| `this video needs about N min of denoising, and sglang's client abandons a request after 100 min` | the timeout guard | add GPUs, or segment the video |
| `--enable-cfg-parallel puts the two classifier-free-guidance branches on two ranks...` at start-up | `--enable-cfg-parallel` on a distilled directory, or on a base directory with `use_cfg: false` | drop the flag; use ulysses/ring/tp |
| `guidance_scale=... ignored: the model directory decides whether classifier-free guidance runs ...` (warning) | the request sent a `guidance_scale` | nothing to fix; to turn base guidance off or change its strength, set `use_cfg` / `cfg_min` / `cfg_max` in the folder's `config.json` and rebuild the directory |
| `Ring Attention is only supported for flash attention or sage attention backend` (from SGLang) | `torch_sdpa` with `--ring-degree` | `fa` or `sage_attn` with the ring |
| `--dit-precision fp16: the MovieRoll transformer runs in bf16 ...`, `... load a pre-quantized transformer ...`, `--vae-slicing splits a batch ...` | an unsupported precision, quantisation or VAE option | drop it |
| an image-to-video run whose load log says `distilled chain, context noise 0.15` | the directory was built from `distilled/t2v` (text-to-video weights and settings) | build one from `distilled/i2v`; its log says `context noise 0` |
| `... cannot be changed for MovieRoll under SGLang` from `assemble_model_dir` | the folder's config.json changes a setting SGLang fixes | only the context noise, KV recompute, `use_cfg` and AnnealCFG range can differ |
| `unknown keys ['guidance_scale']` from `assemble_model_dir` | a folder `config.json` written before `guidance_scale` became `use_cfg` | replace it with `"use_cfg": true` (base) |
| `MovieRoll has 16 attention heads, which --tp-size T x --ulysses-degree U = P does not divide` | topology | choose degrees whose product divides 16 |
| startup refusal with FSDP + TP | both shard the same layers | pick one |
| NaN or garbled video, `check_sageattention` prints `BROKEN` | the `v2.2.0` SageAttention tag | rebuild at the pinned commit; fall back to `fa` |
| VAE decode OOM on one GPU | whole-video decode | `--keep-offload` on 1 GPU; on several, PS-VAE is already on with a sequence-parallel group (raise `--ulysses-degree` / `--ring-degree`) |
| `every rank needs at least one row` | PS-VAE with more ranks than latent rows | 480p has only 30 rows |
| `FileExistsError` from `assemble_model_dir` | `--out` exists | it never overwrites; use a new name |
