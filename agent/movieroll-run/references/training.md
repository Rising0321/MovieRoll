# Training reference

Four stages, each with a launch script in `movieroll/scripts/launch/`. Code is named by file, class and
function, never by line number.

```
HunyuanVideo-1.5 480p T2V
  └─① Streaming SFT, single-event ──▶ MovieRoll-Base
       └─② Streaming SFT, multi-event ──▶ multi-event Base
            └─③ ODE initialization ──▶ few-step student
                 └─④ CuLT-DMD ──▶ MovieRoll-Distilled
```

Each stage consumes the previous stage's checkpoint. ③ and ④ are driven by `movieroll/distill.py`
(`--mode stage2` / `--mode stage3`); ① and ② by `movieroll/train.py`.

## How every stage launches

The four scripts set the launch topology and the environment only; every training setting lives in the YAML.
Each one `cd`s to the repository root and exports `PYTHONPATH=$REPO` (torchrun puts the *script's* directory on
`sys.path`, not the cwd). Beyond that they differ.

`train_sft.sh` (①②), `train_ode.sh` (③ step 2) and `train_dmd.sh` (④) launch one torchrun per node and read:

```bash
export NPROC=8                 # GPUs per node
export NNODES=1                # multi-node: number of nodes
export NODE_RANK=0             # multi-node: this node's index
export MASTER_ADDR=127.0.0.1   # multi-node: address of node 0
```

Use command-scoped `CONFIG=path/to/stage.yaml bash <launcher>` assignments when overriding a stage's config. Leave `CONFIG` unset to use each launcher's own default; do not carry an exported config from one stage into another. `MASTER_PORT` also has a per-script default and only needs exporting when you override it.

`train_dmd.sh` also reads `LOG_DIR` (where its launcher log goes;
otherwise a forwarded `--override logdir=...`, else the config's `logdir`). These three also export
`PYTHONUNBUFFERED=1`, `TOKENIZERS_PARALLELISM=false` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(an exported value is kept).

`precompute_ode.sh` (③ step 1) is different: one job per node, never a multi-node torchrun, so it reads no
`NNODES` / `NODE_RANK` / `MASTER_ADDR`. Its knobs are `CONFIG`, `NPROC`, `SP_SIZE` (default 2, or 1 when `NPROC`
is 1 or odd), `SHARD_ID`, `NUM_SHARDS`, `CACHE_DIR` (overrides `ode.trajectory_cache_dir`), `MAX_SAMPLES`
(prompts per rank, 0 = no cap), `DECODE_VIDEO` (default 1) and `MASTER_PORT`; it exports only `PYTHONPATH`. See
③ for what `SHARD_ID` / `NUM_SHARDS` mean.

Cluster-specific tuning (`NCCL_SOCKET_IFNAME`, IB settings) is deliberately **not** set by any script — export it
in the launching environment.

`--override KEY=VALUE` (repeatable) is accepted by `movieroll/train.py` (①②) and by `movieroll/distill.py` in
both modes (③ training and ④). It is an OmegaConf dotlist merged before validation, so overridden keys are
validated too; the launch scripts forward extra arguments to it. `precompute_ode.py` does not take it — only its
own flags (`--cache_dir`, `--max_samples`, `--shard_id`, ...). Settings belong in the YAML; `--override` is for
tests and ad-hoc checks.

Resuming works the same way in every training stage: set `resume_checkpoint` in the stage's YAML to the
`checkpoints/step-<N>` directory the run wrote (the directory itself, not `checkpoints/`), and run the same launch
command again. The per-stage sections say what each checkpoint holds and what a resume restores.

Every stage validates its whole config **before** NCCL init and before any weight loads, and the validators
reject unknown keys by name. A typo costs seconds, not the twenty minutes it takes to shard 8.3 B parameters.

## Cross-stage consistency

These must agree across stages, and **the code does not check it** — each stage validates only its own config.
Getting one wrong does not raise; it trains something that will not sample correctly.

| Quantity | Released value | Where it appears |
|---|---|---|
| `chunk_size` | 4 | `streaming_chunk_size` in ①②, `streaming_config.streaming_chunk_size` in ③④, `chunk_size` in the model `config.json` |
| `window_size` | 5 | same, and must equal `len(denoising_step_list)` in ③④ |
| `full_history_block_num` | 6 | same |
| `max_num_latent_frames` | 120 | same |
| `denoising_step_list` | `[1000, 800, 600, 400, 200]` | ③ `ode.denoising_step_list`, ④ `dmd.denoising_step_list`, model `config.json` |
| `timestep_shift` | 5.0 | ③ `ode.timestep_shift`, ④ `dmd.timestep_shift`, model `config.json` |
| context-noise sigma | 0.15 | ③ `ode.context_noise_sigma_max`, ④ `dmd.context_noise`, distilled t2v `config.json` |
| `model_type` | `MovieRoll-16164-MMDiT` | every stage |
| VAE / text encoder / byT5 paths and precisions | `ckpts/...` | every stage |

Two more values are not cross-checked either. **Resolution**: ODE (`video.resolution` in `stage2_ode.yaml`),
CuLT-DMD (`video.resolution` and `sampling.height` / `sampling.width` in `stage3_dmd_rolling{,_acs}.yaml`) and
inference (the model `config.json`) all run at 480x864; SFT trains at 480x640 (`height` / `width` in ①②).
**KV-recompute threshold**: 18 in ③, ④ (`streaming_config.streaming_kv_recompute_threshold`) and the model
`config.json`. Keep them aligned when you edit one.

## ①② Streaming SFT

Config: `movieroll/configs/sft/stage1_single_event.yaml`, `stage2_multi_event.yaml`.
Entry: `train_movieroll.py` → `movieroll/train.py::main` with `load_sft_config`.

**Must be set for your environment**: `dit_weight` (must be `.safetensors`; ① the public 480p t2v transformer,
② the Base produced by ①), `data_root`, `output_dir`, `vae_path`, `text_encoder_path`, `byt5_google_path`,
`byt5_ckpt_path`, the two glyph JSONs, `sample_prompt_csv` (a **list**, not a string), `resume_checkpoint`.

**Must match the data and the checkpoint** — wrong values either raise or train the wrong thing:

```yaml
height: 480                                 # multiples of 16
width: 640
sample_n_frames: 241                        # 4n+1, and (T_lat - 1) % chunk == 0
streaming_chunk_size: 4
streaming_window_size: 5
streaming_max_num_latent_frames: 120        # >= T_lat + chunk*window  (61 + 20 = 81)
streaming_full_history_block_num: 6         # in [0, 54]
infer_steps_video: 50                       # must be divisible by streaming_window_size
model_type: MovieRoll-16164-MMDiT
```

**How the two stages differ.** Apart from the comments and the three per-stage paths (`dit_weight` — ② starts
from ①'s Base — `data_root` and `output_dir`), the shipped YAMLs differ in five fields: `learning_rate` (2.0e-5 →
1.0e-5), `sp_size` (2 → 4), `streaming_caption_type_probs` (`{event: 0.0, structure: 1.0}` →
`{event: 0.9, structure: 0.1}` — this is the actual single/multi-event switch), `sample_interval` and
`fixed_prompt_validation_every` (2000 → 10000).

**Data contract**:

```
<data_root>/
  manifest.jsonl        # one JSON object per line
  latents/<id>.pt       # float32 [32, T_lat, height//16, width//16], already VAE-scaled
```

| Field | Meaning |
|---|---|
| `id` | required, str; names the latent file |
| `text` | str or null; `""`/null takes the empty-caption short path (no caption-type draw) |
| `events` | optional `[{text, start_sec, end_sec}, ...]`; clipped, filtered and sorted by `_normalize_streaming_events` |
| `duration_sec` | **required and positive when `events` is present** — a hard error, because without it every event would be dropped and the sample would silently fall back to the structure caption |
| `data_type` | optional, must be `"video"` |

Latents are produced by `movieroll/scripts/encode_latents.py`, which trims each clip to its first `--num_frames`
at `--fps`, scales to cover `--height x --width`, centre-crops and encodes with the trainer's own VAE code. It
refuses `--dst == --src` (the output manifest would replace the one naming the videos), a manifest row without a
str `id` or non-empty `video`, a CPU-only machine, and a clip shorter than `--num_frames`. It writes
`<path>.partial` then `os.replace`, so `--skip_existing` can resume safely — but it does not validate existing
files' contents, and rank 0 rewrites the manifest either way.

The latent shape is checked in `MovieRollJsonlDataset._load_latents`, i.e. **when that sample is first drawn** —
a bad clip can surface hundreds of steps in. Keep `encode_latents`' geometry flags and the training YAML in sync.

**Topology**: `fsdp_sub_world_size` is the size of one FSDP shard group (8 = one node). `hybrid` with
`fsdp_sub_world_size == WORLD_SIZE` degrades to `full`. Constraints: `16 % sp_size == 0`,
`WORLD_SIZE % sp_size == 0`, and for real hybrid `replicas = WORLD_SIZE // fsdp_sub_world_size >= 2` with
`WORLD_SIZE % (replicas * sp_size) == 0`; the division floors, so the shard-group size need not divide
`WORLD_SIZE`. The head-count check always runs; the `WORLD_SIZE` checks only when `WORLD_SIZE > 1` or
`TORCHELASTIC_RUN_ID` is set, because importing `movieroll` fills in `WORLD_SIZE=1` — launching with plain
`python` skips them and hits a bare assert instead.

**What one step does** (`StreamingSplitLatentsWrapper`, `StreamingTimestepsWrapper`,
`StreamingMergeLatentsWrapper`, `StreamingTextSelectionWrapper` in `trainer/flow/streaming_wrappers.py`):

- Task draw: one Bernoulli on `streaming_t2v_prob` (0.8) → t2v or i2v; the layouts differ as in inference.
- Window tiling: `window_phase ~ U{0, window_size-1}`, drawn on sp rank 0 and broadcast; the first window holds
  `phase + 1` chunks, the rest `window_size`, and a trailing remainder is **dropped**. So the last
  `window_size - 1` chunks of a clip are supervised less often (logged once as `[window-tiling]`).
- Timesteps: with probability `min(1/num_windows, 0.5)` a window is "non-rolling" (all its chunks share one t),
  otherwise `sample_rolling_timesteps` builds a monotone ladder across the window.
- Context noise: with probability `context_noise_prob` (0.9), the clean history is re-noised per chunk with
  sigma in `[sigma_min, sigma_max]` = `[0, 0.15]`; `t_cond` stays 0.
- Text: per-window caption selection (event captions pick the event overlapping that window's frames), then
  two-level null dropout — sample-level `per_sample_null_prob`, then per-window conditional on surviving it, so
  the marginal null rate per window is the sum. Glyph (byT5) embeddings survive **only** for structure captions.
- All draws happen on sp rank 0 and are broadcast. Nothing may depend on the rank.

Autocast is hardcoded bf16 in the training loop, independent of `precision`.

**Checkpoint**: `<output_dir>/checkpoints/step-<N>/` holding `diffusion_pytorch_model.safetensors`,
`ema_diffusion_pytorch_model.safetensors`, `scalar_states.pkl`, `config.json`. **Optimizer state is not saved.**
`save_checkpoint_without_optim` must be called on every rank (it contains collectives); only rank 0 writes.

**Resume**: point `resume_checkpoint` at that directory. It loads the **EMA** weights, restores counters and the
in-epoch data cursor, and does **not** restore optimizer momentum — the loss curve jumps once on restart by design.

**`dry_run: true`** runs the data pipeline and the topology only: it fetches batches and syncs them, with no
forward, backward, sampling or checkpointing. Models are still loaded first, so it validates plumbing, not
load time. Use it with `max_train_steps: 2`.

## ③ ODE initialization

Config: `movieroll/configs/stage2_ode.yaml`. Two steps, two different launch topologies.

**Step 1, precompute** (`movieroll/scripts/launch/precompute_ode.sh` → `movieroll/scripts/precompute_ode.py`):
the frozen teacher runs a full rolling inference per prompt, and `MovieRollODEPipeline`'s hooks capture each
round's diagonal window plus the popped clean chunks. One file per prompt:

```
<ode.trajectory_cache_dir>/
  000000.pt            # prompt, seed, clean_video [C,total,h,w] (raw scale), rounds[], chunk_sizes, warped_timesteps, height, width
  videos/000000.mp4    # with --decode_video 1: the teacher's own video, for eyeballing
```

`rounds[i]` is `{window_input, window_timesteps, start_frame}`, prefill rounds included. Files are written as
`*.pt.tmp.<pid>` then `os.replace`d, and the video is written before the cache, so "a `.pt` exists" means the
entry is complete. The index is the prompt's **global** position in the filtered list — inserting or deleting
rows in the prompt file silently re-points existing cache files at different prompts.

Two topologies, and `SHARD_ID`/`NUM_SHARDS` mean different things in each:
- `NPROC=1`: one process per GPU (always `cuda:0`, distinguished by `CUDA_VISIBLE_DEVICES`), no collectives.
  `NUM_SHARDS` = number of such processes cluster-wide, `SHARD_ID` = this one's index. A cache hit skips the
  forward entirely. This is the fast way.
- `NPROC>1`: one torchrun job per node, FSDP-sharded teacher plus sequence parallelism, the ranks of an sp group
  in lockstep on one prompt. `NUM_SHARDS` = number of **jobs (nodes)**, `SHARD_ID` = this node's ordinal — not
  GPU or sp-group counts. The forward always runs (it is a collective); a cache hit only skips the rank-local
  write. `_check_job_shard_args` all-reduces to verify every rank of a job passes the same pair.

**Caches built with different `NPROC`/`SP_SIZE` are not bit-identical**: the per-prompt noise seed is
`seed + group_id * 1_000_003` where `group_id` is the sp group's first global rank, so which dp group a prompt
lands in changes its noise. Pick one topology per cache directory.

The launch script creates `$CACHE_DIR/videos` itself — with `--decode_video 0` nothing else would, and the first
`torch.save` would fail. Running the `.py` directly means creating it yourself, and exporting `PYTHONPATH`.

**Step 2, train** (`train_ode.sh` → `distill_movieroll.py --mode stage2`): reads **only** the cache directory
(prompts come from the cache entries, `data.prompt_path` is unused). One step is one `student` forward: a round
is drawn uniformly **over rounds, not chunks** (prefill rounds would otherwise be ~4x under-trained), up to
`ode.max_num_noisy_windows` windows are assembled in the WinTF layout, and the student regresses the cached
`clean_video`. Out-of-range tail frames are zero-padded and masked; the mask is applied by **boolean indexing
before** `mse_loss`, not by multiplication (a `0 * inf` would become NaN).

**Key fields**: `teacher_ckpt` (precompute only), `student_ckpt` (train only), `data.prompt_path` (precompute
only), `ode.trajectory_cache_dir`, `ode.cache_only` (must be `true` for train, `false` for precompute — the
script forces it), `ode.denoising_step_list` (length **must equal** `streaming_window_size`, strictly
descending), `ode.teacher_num_inference_steps` (must be divisible by `streaming_window_size`),
`ode.timestep_shift`, `ode.context_noise_sigma_max`, `video.num_frames`, `video.resolution` (both sides
multiples of 16), `max_steps`, `save_interval`, `validation.prompt_indices`, `sp_size`, `fsdp.*`. `data.batch_size`
must be 1.

**Prompt file**: `.csv` (opened `utf-8-sig`) needs a `prompt` column — the reader falls back to
`original_prompt`, but the validator only accepts `prompt`, so an `original_prompt`-only CSV fails validation. An
optional integer `seed` column; if absent or all-empty, the seed is `1234 + idx`. Any other suffix is treated as
one prompt per line.

**Checkpoint**: `<logdir>/<exp_name>_<timestamp>/checkpoints/step-<N>/` with `student.safetensors` (a full,
unsharded state dict — this is exactly what ④'s `generator_ckpt` points at), `optimizer.pt`, `training_state.pt`
and a `COMPLETE` sentinel written **last**.

**Resume**: `resume_checkpoint` must point at the `step-<N>` directory itself, directly below `checkpoints/`
(asserted). It must be resumed with the same GPU count: `training_state.pt` records `world_size`, the optimizer
state is re-sharded locally from a full state dict, and the data order plus every seed depend on `dp_degree` and
`group_id`.

A run-time cross-check worth knowing: `compute_rolling_ode_loss` compares the cache's own `chunk_sizes` against
the current `chunk_size`/`window_size` and raises `the ODE cache was not rolled with chunk_size ...` — change
the geometry and you must re-precompute, preferably into a fresh directory.

## ④ CuLT-DMD

Config: `movieroll/configs/stage3_dmd_rolling.yaml` (t2v), `stage3_dmd_rolling_acs.yaml` (i2v).
Launch: `train_dmd.sh` → `distill_movieroll.py --mode stage3`.

**One step** (`DMDTrainer.train`): `train_generator = global_step % dfake_gen_update_ratio == 0` — the ratio
*skips* generator steps rather than running an inner loop, so at ratio 3 and 1200 steps you get 400 generator
updates and 1200 fake-score updates. The critic step runs unconditionally.

A generator step is one long rollout: `zero_grad` once, then for each selected window the handler assembles a
score window (detached history + the live window), re-encodes its boundary frame through the VAE, computes the
DMD gradient from `s_fake - s_real_cfg`, takes `0.5 * mse` against `(x - grad).detach()` on the supervision mask,
**backwards immediately with weight `loss_scale / K`**, and frees the graph before the next window's forward.
`clip_grad_norm_` and `optimizer.step()` run once, after the whole rollout — so generator weights are fixed
throughout, and gradients only accumulate.

**K** is `(latent_frames - 1) // chunk_size // window_size`: 3 in phase 1, 9 in phase 2. Because
`num_chunks % window_size == 0` is enforced, K does not depend on the randomly drawn `start_phase`.

**Two phases**, fixed order, keys `short_single_event` then `long_dual_event`:

| | phase 1 | phase 2 |
|---|---|---|
| `max_steps` | 600 | 600 (900 in the acs config) |
| `video_length` / `latent_frames` | 241 / 61 | 721 / 181 |
| `event_count` | 1 | 2 |
| `csv` | `prompt` column | `event_1`, `event_2` columns |
| `switch_frame_min/max` | must be null | 129 / 593 |

At the phase boundary: a `phase_complete` checkpoint is written, verified, and (after phase 1) **read back
in**; `phase_step` resets to 0 and the dataloader is rebuilt with a fresh cursor. Weights, both Muon optimizer
states, the EMA shadow, `global_step` and the batch-decision counter are kept. The pipeline's
`_uniform_noise_counter` / `_uniform_decision_counter` are not checkpointed and restart at 0.

**Key fields**: `generator_ckpt` (③'s `student.safetensors`; **not** loaded when resuming),
`real_score_ckpt` (frozen teacher, always loaded), `fake_score_ckpt` (critic init; not loaded when resuming),
`dmd.real_guidance_scale` (4.0), `dmd.context_noise` (0.15), `dmd.score_mode` (`t2v` | `acs`),
`sampling.test_event_csv`, `fsdp.shard_world_size`, `sp_size`, `logdir`, `save_interval`, `ema_start_step`.
`data.batch_size` must be 1; `mixed_precision` must be `true`; `dmd.score_window_latent_frames` must be
`>= window_size * chunk_size + 1`; the `dmd.uniform_timestep_regularization` block is mandatory (set
`probability: 0.0` to disable it).

**Invariants the code enforces** — worth knowing because violating them hangs rather than fails:

| Invariant | Enforced in |
|---|---|
| Every rank has the same K | `_assert_window_count_uniform`: a WORLD `all_reduce(MAX)` on `[K, -K]`, called before any generator forward |
| Weights fixed during a rollout | structure: one `zero_grad` before, one `step` after |
| History and KV detached | `_assemble_teacher_window` (`.detach().clone()`), `WindowPrediction(... .detach())`, `rolling_queue.update_with_new_latents(latents.detach())` |
| Current window must be differentiable | `Handler.on_window` raises if not `requires_grad` |
| Each window backwards immediately and frees its graph | handler `backward_fn` + the pipeline's `del loss_prediction, predicted_clean_window` |
| Losses finite on all ranks, or all fail | `_require_finite_world` (SUM all_reduce) |
| Non-finite grad → every rank skips the step **and** the EMA update | `_any_rank` (MAX all_reduce); a NaN in the EMA shadow is permanent |
| Gradient accumulation does not all-reduce twice | `_set_fsdp_gradient_sync`, which raises rather than silently no-op |

**i2v (`score_mode: acs`)**: the score models gain a `vision_in` tower fed by a frozen SigLIP encoder
(`siglip_path` must contain `image_encoder/` and `feature_extractor/`); the generator is unchanged. The anchor
condition applies only to the window at frame 0, so later windows are still scored as t2v. Required together:
`video.task_sampling.t2v_prob: 0.0`, `real_score_ckpt == fake_score_ckpt` (string equality), and a single FSDP
shard group (`fsdp.shard_world_size == world_size`, or `fsdp.sharding_strategy: full`) — with more than one
shard group each group draws its own `start_phase`, so the per-window backwards would face different parameter
sets in the HSDP all-reduce and hang.

**CSV contracts**: phase 1 needs exactly one `prompt` column, no empty cells, at least one row. Phase 2 needs
exactly `event_1`, `event_2` (contiguous and complete, checked by `ordered_event_columns`); a row with an empty
event is dropped with a warning, all-empty raises. `sampling.test_event_csv` needs contiguous `event_1..event_N`
with no empty cells, plus — under `acs` — an `input_image` column whose paths exist relative to the CSV.
**Row count must be at least `dp_degree`** (times `data.batch_size`, which is 1): the sampler and the loader
both use `drop_last=True`, so fewer rows would give every rank an empty epoch. `DMDDataModule.configure_phase`
refuses that when it builds the phase's loader: `<csv> has N row(s), but <phase> needs at least M: one batch of 1
for each of the D data-parallel replicas`.

**Checkpoint**: `<logdir>/<exp_name>_<timestamp>/checkpoints/step-<N>/`, `N` being the global step
(`DMDTrainer._setup_logdir` makes the timestamped experiment directory, `DMDCheckpointManager` the step
directory) with DCP shard directories (`generator_model/`, `generator_optimizer/`, `critic_model/`,
`critic_optimizer/`), per-rank `generator_ema_rank<N>.pt`, the aggregated `generator.safetensors` and
`generator_ema.safetensors`, `checkpoint_metadata.json`, and `COMPLETE` written last.

**Resume**: point `resume_checkpoint` at the `step-<N>` directory; the run continues in its experiment directory,
two levels up. It must match the saved `world_size` and FSDP mesh shape (the EMA shadow is sharded per rank), the
optimizer impl must be `muon`, and `global_step` must equal `phase_step (+ phase1.max_steps)` — so **changing
phase 1's `max_steps` makes old checkpoints unresumable**.

**Multi-node: `logdir` must be on storage every node can read and write**, for any run, not only a resume. Each
rank writes its own shards and `generator_ema_rank<N>.pt` into the step directory, while rank 0 alone creates the
directory and writes `checkpoint_metadata.json` and `COMPLETE` (`DMDCheckpointManager.save`). At the end of phase
1 every rank then inspects and loads that checkpoint back (`inspect_synchronized`, `load` in
`movieroll/trainer/dmd/checkpoint.py`, called from `DMDTrainer.train`) before phase 2 starts, so a node-local
`logdir` breaks the phase-1 → phase-2 handover, and any later resume as well.

**Training-time samples** use `dmd.sampling_context_noise` as the context-noise sigma; absent or `null` means
the training value `dmd.context_noise`. The acs config sets it to 0.0, matching the `distilled/i2v` model folder.

**Turning the result into a model folder for inference**:

1. Copy `movieroll/configs/models/distilled/t2v/config.json` (or `.../i2v/config.json` for an acs run — they
   differ only in `context_noise`, 0.15 vs 0.0) into a new directory.
2. Copy the step directory's `generator_ema.safetensors` there as `model.safetensors` (use
   `generator.safetensors` before `ema_start_step`, or if you want the online weights).
3. Run inference with `--model <that directory>`.

`load_model_dir` validates key names against the profile but does **not** check that they match what you trained
with — if you changed the streaming geometry or the step list, edit this `config.json` to match.

## Failure modes

| Signal | Stage | Fix |
|---|---|---|
| `unknown key(s) [...]: a typo, or a setting that no longer exists` (①②) / `unknown ... fields are not allowed: [...]` (③④) | all | the validators whitelist keys; fix the name |
| `` `sample_n_frames` must be of the form 4n+1 ... try X or Y `` / `... latent frames, which is not a whole number of chunks of 4` | ①② | use a printed candidate, and re-encode latents to match |
| `Stage2 video length must be positive and map exactly through the 4x temporal VAE` | ③ | a `16n+1` frame count |
| `video_length must satisfy the 4n+1 pixel-frame rule` / `num_chunks must be divisible by rolling window size` / `pixel-to-latent mapping must be ...` | ④ | an `80n+1` frame count (whole windows of 5 chunks: 81, 161, 241, 721, ...), with `latent_frames = (video_length - 1) // 4 + 1` |
| `<id>: latents have shape (...), expected (...)` | ①② | `encode_latents` geometry must equal the training YAML |
| ``<id>: `duration_sec` must be a positive number when `events` is given`` | ①② | add it; without it events are dropped silently |
| `` `sample_prompt_csv` is required when `sample_interval > 0` `` / `` `sample_prompt_csv` must be a list of paths, not a single string `` | ①② | add the list, or `sample_interval: 0` |
| `` `precision` must be 'bf16' or 'fp16' when training-time sampling runs `` | ①② | `bf16`, or turn sampling off |
| `hybrid ... needs at least 2 replicas` | ①②③④ | `full`, or lower the shard-group size |
| `len(ode.denoising_step_list) must be 5` | ③ | equal to `streaming_window_size` |
| `the ODE cache was not rolled with chunk_size X and window_size Y` | ③ | re-precompute after a geometry change |
| `Stage2 validation ... but K of them are missing` | ③ | shorten `validation.prompt_indices`, or precompute more prompts |
| `per-window backward count differs across ranks` | ④ | all nodes must read the same YAML and CSV |
| `<csv> has N row(s), but <phase> needs at least M` | ④ | at least `dp_degree` rows in every phase CSV |
| `dmd.score_mode='acs' requires video.task_sampling.t2v_prob=0` / `dmd.score_mode='acs' needs a single FSDP shard group` | ④ acs | follow the message exactly |
| `checkpoint world size mismatch` (③④) / `checkpoint FSDP mesh mismatch` / `resume checkpoint is missing EMA shard` (④) | ③④ | resume with the identical topology, or restart from the exported safetensors |
| `resume_checkpoint must be a step directory directly below ...` | ③④ | point at `step-<N>` itself, inside `checkpoints/` |
| `checkpoint is incomplete (missing COMPLETE)` | ③④ | use the previous step directory |
| `checkpoint phase/global step mismatch` | ④ | phase 1's `max_steps` was changed |
| OOM entering `long_dual_event` | ④ | raise `fsdp.shard_world_size`, `sp_size: 4`, keep gradient checkpointing on, consider score-model CPU offload |
| `non-finite Stage2 ODE loss` / `non-finite Stage2 ODE gradient norm` | ③ | the run stops (③ never skips a step); lower lr, check `student_ckpt` |
| `... is non-finite on N rank(s)` | ④ | a non-finite loss stops the run on every rank; lower lr, check `generator_ckpt` really is ③'s student |
| `generator grad_norm=nan; non-finite gradient detected ... skipping optimizer/EMA update on every rank` (also `critic grad_norm=...`) | ④ | occasional skips are fine; persistent means lower lr, check `generator_ckpt` |
