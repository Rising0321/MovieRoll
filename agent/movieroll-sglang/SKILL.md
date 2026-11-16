---
name: movieroll-sglang
description: Run MovieRoll through the SGLang Diffusion adapter in `sglang_movieroll/` — build its pinned virtual environment, assemble a diffusers-style model directory from a released checkpoint, generate offline or serve over HTTP, choose a multi-GPU layout (tensor parallel, Ulysses, ring, CFG parallel, FSDP, PS-VAE), and compare the result against native MovieRoll in the same venv. Use it whenever someone wants MovieRoll under SGLang, hits a registration or model-directory problem there, needs to pick or debug an SGLang parallel layout, runs into the 100-minute client timeout or a refused request, or asks how the adapter's output compares with the original repository.
---

# MovieRoll under SGLang Diffusion

A Chinese version of this document is `SKILL_CN.md`; the `references/` files are English only.

| File | When to read it |
|---|---|
| `references/setup.md` | once per machine: the venv, SageAttention, assembling a model directory from a MovieRoll model folder, where the sampling numbers come from |
| `references/run.md` | every run: registration, offline and serving entry points, frame-count arithmetic, multi-GPU, framework limits, comparing with native, failure table |

The adapter lives in `sglang_movieroll/` (its own README is `sglang_movieroll/README.md`). It is written
framework-native: it uses SGLang's loader, layers and parallelism rather than wrapping MovieRoll's own
pipeline, and at run time it does **not** import the `movieroll` package.

## Before anything else

Two facts cause most of the lost time here:

1. **SGLang needs its own venv.** Its pinned torch and transformers conflict with MovieRoll's. Never install
   it into the environment you run native MovieRoll in — and when comparing outputs, compare only against
   native running *inside the same venv*, because a different torch version changes the bf16 CPU random stream
   and therefore the initial noise.
2. **`register()` must be called at the top level of the launching script.** SGLang spawns its GPU workers as
   fresh interpreters that re-import the script, so registration inside `main()` leaves the workers without the
   pipeline. `cli.py` and `examples/offline_generate.py` show the correct shape.

## Procedure

1. **Classify the request.** First run on this machine → `references/setup.md`. Already set up → straight to
   `references/run.md`. A failure → find the message in the failure table first; most refusals are deliberate
   and the text names the fix.
2. **Check the environment is the adapter's venv**, and that `PYTHONPATH` includes the repository root.
3. **Check the model directory exists and matches the chain** the user wants. One directory is one chain; a
   request cannot switch between base and distilled. Each directory is built from one MovieRoll model folder
   (`--model ckpts/movieroll/base`, `distilled/t2v` or `distilled/i2v`); image-to-video belongs on the directory
   built from `distilled/i2v`, which runs without context noise.
4. **Do the frame arithmetic**: `num_frames = 16k + 1` (17, 81, 241, 721, 1441). State the number you chose.
5. **Pick the layout** from the table in `references/run.md`, and sanity-check it: `tp_size x ulysses_degree`
   must divide 16; CFG parallel needs a base directory with guidance on (`use_cfg`); FSDP and TP are mutually
   exclusive.
6. **Estimate the runtime** before launching anything long. The adapter refuses requests it projects to exceed
   SGLang's 100-minute client timeout — that refusal is a feature, not a bug, and the answer is more GPUs or a
   segmented video, not a retry.
7. **Report** the command, where the output landed, and the real outcome.

## Hard rules

1. **The sampling parameters in `rolling/profiles.py` are the released evaluation values, not defaults to
   tune.** A model folder's `config.json` may differ from them only in the settings MovieRoll's stages read
   (context noise, KV recompute policy, `use_cfg` and the AnnealCFG range `cfg_min` / `cfg_max`); the builder
   carries those into the directory's `model_index.json`. Base guidance is a model setting: `use_cfg` turns it on
   or off, AnnealCFG sets its strength, and a request's `guidance_scale` is ignored. Change anything only on
   request, and say what you changed.
2. **Do not edit `movieroll/` to make the adapter work.** The adapter adapts; the package stays as it is.
3. **Keep the registration order in `registry.py`.** Component classes, then SGLang's own pipeline discovery,
   then the MovieRoll pipelines. Reversing it silently unregisters every built-in pipeline.
4. **Prefer SGLang's own efficiency defaults** (its layers, its attention, its loader). Do not add code that
   emulates native MovieRoll's rounding behaviour to chase decibels.
5. **Do not work around a refusal.** The refusals listed in `references/run.md` exist because the feature is
   wrong for a rolling-window model (LoRA, teacache, batch > 1, VAE tiling, …) or broken in this release.
6. **`--enable-torch-compile` is not a free win here** — the growing KV cache triggers repeated recompilation.
7. This release ships no tests; the available checks are the client-side request validation, a short real run,
   and a comparison against same-venv native.

## Working loop

```
- [ ] 1. Setup or run? Read the matching reference
- [ ] 2. Confirm the venv, PYTHONPATH and the model directory
- [ ] 3. Frame count = 16k+1; topology divides 16 heads
- [ ] 4. Project the runtime against the 100-minute limit
- [ ] 5. Launch; read the first output for a refusal
- [ ] 6. Report command, artefacts, outcome
```
