# vLLM-Omni adapter — running it

## How it is discovered

The entry point is registered in **two** groups, both pointing at `vllm_omni_movieroll:register` —
`vllm_omni.general_plugins` and `vllm.general_plugins`. vLLM loads its general plugins while building the CLI
parser, before `vllm serve --omni` has parsed the stage topology, and the Omni group loads later; so the
topology has to be registrable from the vLLM group too. `register()` is idempotent and returns quietly if
vLLM-Omni is absent.

It registers, for each of the two architecture names (`MovieRollPipeline` for Base, `MovieRollDistilledPipeline`
for Distilled; a model directory's `_class_name` picks one), the diffusion model, the **stage pipeline topology**,
and MovieRoll's entry in vLLM-Omni's model-extras table (`video_defaults.py`). Without the topology a plugin model
falls back to a generic diffusion stage whose output type is `image`, and `/v1/videos` refuses with "No final video
output stage". The model-extras entries give the shared examples and `/v1/videos` MovieRoll's defaults (480x864, 721
frames, 24 fps, the chain's steps, and no guidance scale) and tell them to hand the input image over unresized. vLLM-Omni
0.32.0 has no public call for that table, so `register` adds the two entries to its `_EXTRA_SPECS` dict.

Entry points come from installed package metadata: `pip install -e` is what makes this work; `PYTHONPATH` alone
does not.

## Offline

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="./movieroll-distilled")
out = omni.generate({"prompt": "..."}, OmniDiffusionSamplingParams(num_frames=721, seed=930))
```

vLLM-Omni's own example scripts work unmodified:
`examples/offline_inference/text_to_video/text_to_video.py` and `.../image_to_video/image_to_video.py` from a
vLLM-Omni v0.32.0 source checkout (they are not in the wheel), with `--model --num-frames --seed --prompt
--extra-body '{"events": [...]}' --output` (and `--image` for image-to-video). **Start them through
`run_example.py`**:

```bash
python vllm_omni_movieroll/examples/offline_inference/movieroll/run_example.py \
  <vllm-omni>/examples/offline_inference/text_to_video/text_to_video.py --model ./movieroll-distilled ...
```

The scripts look up the model's defaults in vLLM-Omni's model-extras table before they create the engine, and the
engine is where plugins are loaded; `run_example.py` loads them first (vLLM-Omni's own `load_omni_general_plugins`)
and then runs the script. Through it, anything left out takes MovieRoll's defaults (480x864, 721 frames, 24 fps,
Base 50 steps, Distilled 5 steps, no `guidance_scale` sent), and `image_to_video.py` passes the image as it is, so MovieRoll
scales it to cover the video and centre-crops it, also with an explicit `--height`/`--width`. Started directly, a
script falls back to its Wan2.2 preset (text-to-video 1280x720, 40 steps, guidance 4.0, which MovieRoll ignores with a
warning; image-to-video 16 fps and the image stretched to the size).

Remember to `source vllm_omni_movieroll/examples/movieroll_env.sh` first if the engine will use the mp executor (see
`setup.md`).

## Serving

```bash
MODEL=./movieroll-distilled bash vllm_omni_movieroll/examples/online_serving/movieroll/run_server.sh
```

`PORT` defaults to 8091, `DEPLOY` to `deploy/movieroll_1gpu.yaml`; the script sources `movieroll_env.sh` and
then runs `vllm serve "$MODEL" --omni --port "$PORT" --deploy-config "$DEPLOY" "$@"`.

```
POST   /v1/videos            async, returns {"id": ...}
POST   /v1/videos/sync       synchronous, returns the mp4
GET    /v1/videos/{id}       status: queued | in_progress | completed | failed
GET    /v1/videos/{id}/content
DELETE /v1/videos/{id}       cancels at the next window boundary
```

Multipart fields: `prompt`, `num_frames`, `seed`, `extra_params={"events": [...]}`, optionally `size` (default
864x480) and `fps` (default 24); `input_reference` for image-to-video, which reaches MovieRoll unresized. `seconds`
becomes a `duration`, rounded to whole chunks. It sets the length unless the request also sends a `num_frames` other
than 721 or `seconds x fps`: the server fills in one of those two when `num_frames` is left out, so either value is
read as "not given", even when the request sends it explicitly. `guidance_scale` is ignored, with a warning unless it
is 1.

## Request fields

MovieRoll's own keys are exactly three, passed through `extra_args` (offline `--extra-body`, online
`extra_params`):

```
events, video_length, duration
```

Anything else there is an error that lists the allowed set — including `movieroll_overrides`, which belongs in
the model directory or the deploy YAML, not in a request (see `setup.md`).

Read from the sampling params: `height`, `width`, `num_frames`, `seed` or `generator`, `fps`,
`num_inference_steps`, `num_outputs_per_prompt`, `output_type`. The prompt dict accepts `prompt`, `negative_prompt`
and `multi_modal_data.image` (a path string is opened and exif-rotated). `guidance_scale` is not used: whether
classifier-free guidance runs is the model directory's `use_cfg`, and its per-frame strength is AnnealCFG's
`cfg_min` / `cfg_max` (`setup.md`, Overriding sampling parameters).

Validation, all of it client-facing (`OmniClientError` → HTTP 400, and it does not count against the stage's
failure breaker):

- `height`/`width`: positive multiples of 16, and **both or neither**
- `video_length`: `4n+1` with `latent_length % chunk_size == 1`, i.e. `16k+1`; the error prints the two nearest
  legal counts. Default 721. Distilled and image-to-video also need `latent >= 1 + chunk_size`, so at least 17
  frames.
- `duration` is rounded to whole chunks, and wins over `num_frames` only when that is unset, the declared default
  721, or `seconds x fps`
- `num_outputs_per_prompt` must be 1
- base: `num_inference_steps` must be a positive multiple of `window_size` (5)
- `guidance_scale`: ignored on both chains; a value other than 1 (vLLM-Omni's default) logs a warning
- no guidance (distilled, or base with `use_cfg: false`): a `negative_prompt` is dropped with a warning (there is
  no unconditional branch to use it on); the request runs
- distilled: other step counts only warn and the fixed five-step ladder is used (5 steps, what the declared defaults
  send, pass silently)
- two or more `events` split the video evenly; there is no per-event timing. With more events than chunks, an event
  left without a chunk is dropped with a warning on base, while distilled refuses the request with a plain
  `ValueError` raised in the pipeline (not an `OmniClientError`, so not a 400)
- no seed and no generator is an error
- no size gives the profile's 480x864, image-to-video included: the image is scaled to cover it and centre-cropped

Refused with a reason: `aspect_ratio`, `short_edge`, `start_time_seconds`, `generate_sound`, `sound_duration`.
`sample_solver` and `lora_request` must be at their defaults; `flow_shift` must equal the checkpoint's.

## The deploy YAMLs

All eight share `async_chunk: false`, `distributed_executor_backend: mp`, `dtype: bfloat16`, and a single stage
with `max_num_seqs: 1` and `vae_use_tiling: false` (MovieRoll's streaming decode is exact and chunk-bounded;
tiling would blend tiles). They differ only in `devices` and `parallel_config`:

| File | GPUs | Parallelism |
|---|---|---|
| `movieroll_1gpu.yaml` | 1 | none — the recommended layout on a 96 GB card, and `run_server.sh`'s default |
| `movieroll_tp2.yaml` | 2 | `tensor_parallel_size: 2` |
| `movieroll_usp2.yaml` | 2 | `ulysses_degree: 2` |
| `movieroll_usp4.yaml` | 4 | `ulysses_degree: 4` |
| `movieroll_ring2.yaml` | 2 | `ring_degree: 2` — each rank holds half the history cache |
| `movieroll_usp2_ring2.yaml` | 4 | `ulysses_degree: 2` + `ring_degree: 2` |
| `movieroll_hsdp2.yaml` | 2 | `use_hsdp: true`, `hsdp_shard_size: 2` |
| `movieroll_base_cfg2_usp2.yaml` | 4 | `cfg_parallel_size: 2` + `ulysses_degree: 2`, **base with `use_cfg` on only** |

Two things to notice: **no YAML enables PS-VAE** — add the field or pass
`--vae-patch-parallel-size` — and all eight use the `mp` executor, so the per-request timeout always applies.

## Parallelism

CLI (`vllm serve --omni`): `--ulysses-degree`, `--ring-degree` or `--allgather-degree`,
`--tensor-parallel-size`, `--cfg-parallel-size 2` (base with `use_cfg` on), `--use-hsdp --hsdp-shard-size`,
`--vae-patch-parallel-size`.

Structure: vLLM-Omni builds one sequence-parallel group of size
`P = ulysses_degree x ring_degree x allgather_degree`. Tokens are split P ways (zero-padded at the end); outside
attention each rank holds only its shard. Around attention, a Ulysses all-to-all switches to a head split, so
each rank holds `heads / ulysses_degree` heads of one block; other blocks' K/V arrive either by AllGather-KV
(every rank holds the whole history) or by Ring (each rank keeps only its own block's history).

Refused before any weight loads:

- 16 heads must be divisible by `tensor_parallel_size x ulysses_degree`
- `ulysses_mode` must be `strict`, `ulysses_a2a_permute` must be off (the SP here belongs to the model)
- `ring_degree > 1` and `allgather_degree > 1` cannot both be set
- `cfg_parallel_size` is 1 or 2; 2 without guidance (distilled, or base with `use_cfg: false`) is refused (no second
  branch to compute)
- `pipeline_parallel_size > 1`, `enable_expert_parallel`, `text_encoder_tp_size > 1`
- `vae_patch_parallel_size > 1` requires `== world_size`, `vae_parallel_mode == "tile"`, `data_parallel_size == 1`
- `vae_use_tiling`, `vae_use_slicing`, `step_execution`, `streaming_output`, `max_num_seqs > 1`, `lora_path`,
  `quantization_config`, distributed layer-wise offload, paged diffusion KV, quantised KV dtype
- `flow_shift` disagreeing with the checkpoint

**PS-VAE**: `--vae-patch-parallel-size N` with `N == world_size`. Each rank decodes a band of latent rows, 3x3
convolutions exchange a halo (replicate at the global edges, so the function is unchanged), then an all-gather.
The log line `MovieRoll: row-parallel streaming VAE decode over N ranks` confirms it is on. vLLM-Omni will
*also* print "VAE patch parallelism is NOT enabled for MovieRollPipeline; ignoring" — that refers to its own
tiled decode and can be ignored. Decode peak: 16.7 GB on 1 GPU, 8.5 GB/GPU on 2, 4.6 GB/GPU on 4.

Measured (H20, first request, 81 frames, t2v): distilled 1 GPU 63 s, Ulysses 2 46 s, Ulysses 4 35 s,
TP2 x Ulysses2 31 s; base 1 GPU 775 s, CFG2 412 s, CFG2 x Ulysses2 229 s. One minute (1441 frames, denoise loop
only): distilled 1 GPU 711 s (original repository 769 s); distilled Ulysses8 + PS-VAE8 113 s (original 150 s),
78 s with SageAttention; base CFG2 x Ulysses4 + PS-VAE8 1830 s (original 2224 s), 1192 s with SageAttention.

## Framework limits

**Per-request timeout** — applies whenever the executor is `mp`/`ray`, which covers every shipped deploy YAML. An
in-process executor (single-GPU offline) has no limit. See `setup.md`: set `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` before
the engine starts, or the executor kills **all** workers mid-request.

**ComputeBudget early refusal** — only with an `mp`/`ray` executor and `data_parallel_size == 1` (data-parallel
replicas run different requests, so the guard is off there while the executor's timeout still applies). From the
third unit of a phase onward (the first two include window fill and compilation), the adapter projects the total
from the median of the last units and refuses with HTTP 400 if it would exceed the limit, all-reducing the decision
so every rank agrees. The message states the value to set.
This is the good failure: a 400 with an explanation instead of the whole engine being torn down. The cost is
that a long request may be refused a few minutes in.

**Cancellation** happens between windows, so `DELETE` takes effect at the next boundary rather than instantly.

**One request, one video.** `max_num_seqs` must be 1 and `num_outputs_per_prompt` must be 1; the transformer
only supports batch 1.

**Unsupported, with reasons**: FP8 quantisation (measured 15.0 dB against native — below the acceptance bar and
not faster), pipeline parallelism (a video is a strictly sequential chain of windows), Cache-DiT / TeaCache
(every window is a different window with per-frame timesteps, so there is no reusable residual), step
execution, streaming output, LoRA, `SAGE_ATTN_3`, VAE tiling and slicing, distributed
layer-wise offload, expert parallelism, `text_encoder_tp_size`, paged diffusion KV, quantised attention K/V.
vLLM-Omni's interactive streaming playback (`/v1/realtime/video` with `streaming_buffer_seconds`) is a vLLM-Omni
serving feature, not a MovieRoll setting; vLLM-Omni itself refuses it per request ("playback feedback requires an
interactive chunked pipeline"), since `MovieRollPipeline` is not one.

**Warm-up** rewrites the request into a single 17-frame window (`dummy_run_num_frames = 17`), temporarily uses
`window_size` steps for base, ignores the dummy's steps and `guidance_scale`, and runs without guidance.

Output: `output_type == "latent"` returns latents; otherwise the pipeline hands vLLM-Omni `BCTHW` normalised float
media in **`[-1, 1]`**, which its post-processing maps (`output_type="pt"` from `Omni` is `[0, 1]`).

## Comparing against native

Build the venv with `--native-reference`, then run both **in that same venv** with the same seed, prompt, size
and frame count. A different venv is not a valid control: torch 2.13 changes the bf16 CPU `randn` stream, so
native-in-this-venv differs from native-in-torch-2.9 by far more than any adapter bug — different initial
noise, different content.

Mind the ranges: `Omni(...).generate` with `output_type="pt"` returns `[0, 1]`, like `generate_movieroll.py
--output-type pt`; the pipeline's own media (and a hand-written hook on it) is `[-1, 1]`. Check before computing PSNR;
getting it wrong is a ~6 dB error. For image-to-video, build the directory from `distilled/i2v`, give native
`--model ckpts/movieroll/distilled/i2v --input-image` and the same size on both sides (480x864 is the adapter's
default; state it to native explicitly). There is no comparison script in the repository and this release ships no
tests.

Reading the result: the practical bar is **about 20 dB against same-venv native plus a visual check**, with a
negative control (different seed or content) clearly lower. For reference, measured 81-frame values were
distilled 25.0 dB / SSIM 0.887, base 40.8 dB / 0.992, three-event distilled 19.7–21.9 dB, i2v 27.4 dB — against
a native-vs-native noise floor of 25–28 dB. For **long** videos PSNR falls into the teens: that is trajectory
divergence over dozens of chunks between two correct implementations, not damage. Judge those by eye.

## Failure modes

| Signal | Cause | Fix |
|---|---|---|
| unknown model architecture, or `/v1/videos` says "No final video output stage" | the plugin was not discovered, or the topology was not registered | `pip install --no-deps -e vllm_omni_movieroll` in that venv; `PYTHONPATH` is not enough; if `VLLM_PLUGINS` is set it must include this plugin |
| the request dies at 600 s and every worker stops | `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` not exported before the engine started | source `movieroll_env.sh` first; changing it in-process has no effect |
| HTTP 400 "this request would compute about X s … set VLLM_OMNI_DLO_DP_WAVE_TIMEOUT to at least N" | ComputeBudget projection | raise the variable and restart, or fewer frames / more GPUs |
| `video_length … is not 1 modulo chunk_size=4` / `too short: the first chunk needs 5 latent frames` | frame count | `16k+1`; at least 17 for distilled and i2v |
| `height/width must be a positive multiple of 16` / `give both height and width, or neither` | size | fix both |
| the shared example writes 1280x720 (text-to-video) or a stretched 16 fps video (image-to-video) | it was started directly, so it did not see the plugin's defaults | start it through `run_example.py` |
| `model_index.json names the class 'MovieRollPipeline' for the distilled chain ...` | a directory built before the two class names | rebuild it with `download_movieroll.py` |
| `config.json: [...] differ from the released distilled chain ...` | the folder changes a setting outside `OVERRIDABLE` | only the whitelisted settings can differ |
| `config.json: ['num_inference_steps'] differ from the released base chain ...` | the folder sets its own step count | send `num_inference_steps` with each request instead |
| `config.json: unknown keys ['guidance_scale']` | a folder written before `guidance_scale` became `use_cfg` | replace it with `"use_cfg": true` (base) |
| `guidance_scale=... ignored: the model directory decides whether classifier-free guidance runs ...` (warning) | the request sent a `guidance_scale` | nothing to fix; to turn base guidance off or change its strength, use `use_cfg` / `cfg_min` / `cfg_max` in `movieroll_overrides` |
| `negative prompt ignored: this ... runs no classifier-free guidance, so there is no unconditional branch for it to condition` (warning) | a `negative_prompt` sent to distilled, or to base with `use_cfg: false` | nothing to fix; for a negative prompt to count, use a base directory with `use_cfg: true` |
| `N heads per tensor-parallel rank cannot be split over ulysses_degree=…` | topology | 16 must be divisible by `tp x ulysses` |
| refusals for ring+allgather, CFG2 without guidance (distilled, or `use_cfg: false`), non-strict ulysses | unsupported combination | follow the message |
| `vae_patch_parallel_size=…, world size …` | PS-VAE misconfigured | it must equal world size, with `vae_parallel_mode: tile` and no data parallelism |
| "VAE patch parallelism is NOT enabled … ignoring" | vLLM-Omni's own tiled decode | ignore it; do not enable tiling (which is refused anyway) |
| `unknown extra_args [...]; MovieRoll takes ['duration', 'events', 'video_length']` | wrong key, often `movieroll_overrides` | overrides go in `model_index.json` or the deploy YAML |
| visibly broken video, no error | SageAttention built from the `v2.2.0` tag | rebuild at the pinned commit; run the checker |
| `import vllm_omni` raises "Could not load this library" | torchcodec 0.17 (CUDA 13) | `torchcodec==0.16.0+cu129` |
| pip stuck building llguidance from a Rust sdist | glibc 2.28 and no Rust toolchain | `llguidance==1.6.0` binary |
| `torch.cuda.is_available()` is False | a CUDA 13 torch build | reinstall from the cu129 index |
| raising `max_num_seqs`, or enabling FP8 / quantisation / Cache-DiT | unsupported by design | refusals carry their reason; follow the README's feature table |
