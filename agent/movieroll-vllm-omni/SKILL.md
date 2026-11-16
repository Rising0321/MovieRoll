---
name: movieroll-vllm-omni
description: Run MovieRoll through the vLLM-Omni adapter in `vllm_omni_movieroll/` — build its pinned virtual environment, assemble a model directory, set the two environment variables the engine depends on, run offline or serve `/v1/videos`, pick a deploy YAML and a parallel layout (tensor parallel, Ulysses, ring, CFG parallel, HSDP, PS-VAE), override sampling parameters safely, and compare against native MovieRoll in the same venv. Use it whenever someone wants MovieRoll under vLLM-Omni, hits plugin discovery or a request-timeout problem there, needs to choose or debug a deploy configuration, sees a 400 from the compute-budget guard, or asks how the adapter's output compares with the original repository.
---

# MovieRoll under vLLM-Omni

A Chinese version of this document is `SKILL_CN.md`; the `references/` files are English only.

| File | When to read it |
|---|---|
| `references/setup.md` | once per machine: the venv and why its pins are unusual, SageAttention, the model directory, `movieroll_overrides`, and the two environment variables |
| `references/run.md` | every run: discovery, offline and serving entry points, request fields, the eight deploy YAMLs, parallelism and its refusals, framework limits, comparing with native, failure table |

The adapter lives in `vllm_omni_movieroll/` (its own README is `vllm_omni_movieroll/README.md`). It is written
framework-native — vLLM layers, vLLM-Omni's loader and parallelism — and registers itself as a plugin.

## The three things that cause most lost time

1. **Discovery comes from installed package metadata.** The entry points only exist once
   `pip install --no-deps -e vllm_omni_movieroll` has run in that venv. Putting the directory on `PYTHONPATH`
   does nothing, and the symptom is an unknown model architecture or `/v1/videos` refusing with "No final video
   output stage".
2. **`VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` must be exported before the engine starts.** Its default is 600 seconds,
   and on timeout the executor stops *every* worker, not just the request. `source
   vllm_omni_movieroll/examples/movieroll_env.sh` first; setting it later has no effect, because vLLM-Omni's
   executor reads it once, when the first engine of the process starts (MovieRoll's compute-budget guard reads it
   again at pipeline construction). The same file also pins `OMP_NUM_THREADS`, without which post-processing crawls.
3. **vLLM-Omni needs its own venv**, and its pins are deliberate (cu129 torch, torchcodec 0.16, llguidance
   1.6). When comparing with native, compare only against native *inside the same venv* — a different torch
   changes the bf16 CPU random stream and therefore the initial noise.

## Procedure

1. **Classify the request.** First run on this machine → `references/setup.md`. Already set up →
   `references/run.md`. A failure → find the message in the failure table; nearly every refusal is deliberate
   and states its own fix.
2. **Confirm the venv, the editable install and the environment variables.** A successful
   `import vllm_omni_movieroll` is not enough (it also works from `PYTHONPATH`): `movieroll` must be listed by
   `importlib.metadata.entry_points(group="vllm_omni.general_plugins")` in that venv, and the timeout must be in
   the environment of the process that will start the engine.
3. **Confirm the model directory and the chain.** One directory is one chain; a request cannot switch. Each
   directory is built from one MovieRoll model folder (`download_movieroll.py --model ckpts/movieroll/base`,
   `distilled/t2v` or `distilled/i2v`); image-to-video belongs on the one built from `distilled/i2v`, which runs
   without context noise.
4. **Do the frame arithmetic**: `16k+1` (17, 81, 241, 721, 1441), at least 17 for distilled and i2v. Without a
   size, a video (image-to-video included) is 480x864. Start vLLM-Omni's shared example scripts through
   `run_example.py`; started directly they cannot see the plugin's defaults and use Wan2.2's preset.
5. **Pick a deploy YAML** from the table in `references/run.md`, or build a layout and check it:
   `tensor_parallel_size x ulysses_degree` must divide 16; ring and allgather are mutually exclusive; CFG
   parallel needs a base directory with guidance on (`use_cfg`); PS-VAE needs `vae_patch_parallel_size == world_size`. No shipped YAML enables PS-VAE —
   add it if you want it.
6. **Expect the compute-budget guard** on long requests: a 400 a few minutes in, with the value to set. That is
   the designed behaviour, and the answer is a higher timeout, fewer frames or more GPUs.
7. **Report** the command, where the output landed, and the real outcome.

## Hard rules

1. **Sampling parameters are the released evaluation values.** A model folder's own differences (the
   image-to-video folder's context noise) reach `movieroll_overrides` in `model_index.json` when the directory is
   built. Change anything else only on request, only through `movieroll_overrides` (whitelisted, in
   `model_index.json` or a deploy YAML — never in a request), and say what you changed. Base guidance is one of
   those settings: `use_cfg` turns it on or off, AnnealCFG's `cfg_min` / `cfg_max` set its strength, and a request's
   `guidance_scale` is ignored, as is a `negative_prompt` sent to a model without guidance (distilled, or
   `use_cfg: false`); both only log a warning. Raising `kv_recompute_threshold` can push a window past the
   120-frame position grid.
2. **Do not edit `movieroll/` to make the adapter work.**
3. **Prefer the framework's efficiency defaults.** Do not add code that emulates native MovieRoll's rounding to
   chase decibels.
4. **Do not work around a refusal.** Each one in `references/run.md` is either wrong for a rolling-window model
   (batch > 1, LoRA, TeaCache, pipeline parallelism, VAE tiling) or measured to be worse (FP8 at 15 dB).
5. **Mind the output range.** The pipeline hands vLLM-Omni `[-1, 1]` media; what `Omni` returns for
   `output_type="pt"` is already mapped to `[0, 1]`, like native `--output-type pt`. Check the range of what you
   compare before any metric.
6. This release ships no tests; the available checks are the request validation, a short real run, and a
   same-venv comparison with native.

## Working loop

```
- [ ] 1. Setup or run? Read the matching reference
- [ ] 2. Confirm venv, editable install, movieroll_env.sh, model directory
- [ ] 3. Frame count = 16k+1; topology divides 16 heads; PS-VAE == world size
- [ ] 4. Launch; read the first output for a refusal or a budget 400
- [ ] 5. Report command, artefacts, outcome
```
