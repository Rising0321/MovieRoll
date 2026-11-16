# Inference reference

Everything below is read off the code in `movieroll/`. Code is named by file, class and function, never by
line number.

## The two chains

One entry point, `movieroll/generate.py` (`main`, `parse_args`), serves both. The chain is chosen by the model
directory's `config.json`, not by the user.

| | `base` | `distilled` |
|---|---|---|
| Denoising | 50 steps per chunk, CFG | 5 steps, guidance baked into the weights |
| Pipeline | `MovieRollStreamingPipeline.__call__` | `MovieRollDMDPipeline.execute(mode=SAMPLING)` |
| Guidance | `use_cfg: true`; the per-frame scale is AnnealCFG's (`cfg_max` -> `cfg_min`) | none (`use_cfg` pinned to false) |
| Context noise | `inference_context_noise` 0.05 behind `context_noise_enabled` | `context_noise` 0.15 (t2v) / 0.0 (i2v) |
| Noise generator | `torch.Generator(device=device)` | `torch.Generator(device="cpu")` |
| `--output-type latent` | refused | allowed |
| Speed | slow, best quality | ~10x fewer transformer forwards |

The generator device matters: a device generator and a CPU generator produce different random streams, so the
two chains are not comparable sample-by-sample even at the same seed.

Both chains run the same method (the rolling loop, the KV cache, HieraKV, context noise, KV recompute and the
streaming VAE decode), but not all of it through the same code. `MovieRollDMDPipeline` subclasses
`MovieRollStreamingPipeline` and inherits `_encode_clean_chunk_kv`, `_kv_recompute`, `_select_recompute_chunks`
and `_add_context_noise` unchanged. It has its own rolling loop (`_execute_rollout`, where the base chain uses
`__call__` → `_generate_strategy_prompt_switch` / `_pop_and_output_clean_chunk`) and its own event mapping
(`_compute_rollout_event_split_indices`). The distilled video is decoded by `_streaming_decode_and_save` in `generate.py`
(through `decode_latent_chunks_streaming`), the base video at the end of `__call__`; both feed the same
streaming VAE chunk by chunk.

## One rolling round

This is the whole method in one place. `window_size` chunks are in flight; each sits at its own point of the
denoising ladder.

1. **(DECODE only) pop** the oldest chunk off `RollingQueue` — it is now clean — and append it to the output.
2. **Write it into the KV cache**: add context noise, then one transformer forward at timestep 0 with
   `return_kv=True` (`_encode_clean_chunk_kv`). This forward writes the cache and computes no velocity. Skipped
   for the globally last chunk, whose cache nobody would read.
   Instead of a plain encode, a **KV recompute** fires (with `kv_recompute_enabled`) when the cached-chunk count
   reaches `kv_recompute_threshold`, or at an event boundary: the kept chunks (first, the middle of each event, the
   `kv_recompute_recent_k` most recent) are concatenated and re-encoded at timestep 0 into a fresh compact cache
   (`_kv_recompute` → the transformer's `_forward_kv_recompute`). Both HieraKV block groups are rebuilt from the
   same forward. When the selection would keep every decoded chunk there is nothing to compress, and the chains
   differ there — see "Multi-event".
3. **Push** a fresh noisy chunk at the tail (`add_new_noisy_chunk`).
4. **Denoise the whole window** `steps_per_window` times (`num_inference_steps // window_size` for base, exactly
   1 for distilled). Each step is one transformer forward over the window, with a per-frame timestep vector; the
   clean history comes from the KV cache, not from the input tokens.
5. The prompt used is the one of the event the *cleanest* chunk belongs to, so a multi-event transition spans up
   to `window_size` chunks.

Total rounds: `num_chunks + window_size` (the window has to fill up before the first chunk can drain, and drain
after the last is pushed).

## Geometry arithmetic

Nearly every rejected run is this. Let `C = chunk_size` (4 in the released configs) and `W = window_size` (5).

```
latent_length = (video_length - 1) // 4 + 1        # the VAE compresses time 4x
latent_length % C == 1                             # required
=> video_length ∈ {4*(n*C + 1 - 1) + 1} = 16n + 1  # at C=4: 17, 81, 241, 721, 1441, ...
```

- `--video-length` must be `4n+1` **and** leave `latent_length ≡ 1 (mod C)`. `_check_video_length` raises with
  the two nearest valid frame counts.
- `distilled` and **every** i2v run additionally need a whole first chunk: `latent_length >= 1 + C`, i.e. at
  least `4C + 1 = 17` frames (`_check_video_length`, `_check_i2v_video_length`).
- Chunk layout comes from `check_and_get_chunk_sizes` in `modules/chunking.py`:
  - t2v (`mode='default'`): `[1 + C, C, C, ...]` = `[5, 4, 4, ...]`
  - i2v (`mode='i2v'`): `[1, C, C, ...]` — the lone first frame is the input image
- The base pipeline's `__call__` silently rounds `latent_length` down to a whole number of chunks plus one, so
  18/19/20 frames all give the same video as 17. `parse_args` refuses those up front instead.
- `--height` / `--width` must be positive multiples of 16 (`_check_resolution`); 16 is the VAE's spatial stride.
  The distilled chain was trained at 480x864 only.
- RoPE capacity: `max_num_latent_frames` (120) bounds `cached_frames + window_frames`. `StreamingConfig.validate`
  checks the worst case, `(kv_recompute_threshold + W) * C + 1 <= max_num_latent_frames`. With KV recompute off,
  the cache grows without bound and long videos will exceed the grid.

## Command line

`parse_args` in `movieroll/generate.py`. Exactly one prompt source and exactly one weight source are required.

**Weights** — one of:
- `--model <dir>`: a released model folder, `config.json` + `model.safetensors`. The config supplies
  `sampling_mode` and every sampling parameter (`load_model_dir` in `commons/profiles.py`). This is the normal way.
- `--dit-weight <file.safetensors>` plus `--sampling-mode {base,distilled}`: a bare checkpoint, settings then
  come from the built-in profile.

A `config.json` with an unknown key, a wrong `model_type`, or a missing `model.safetensors` is an error before
any weight loads.

**Prompt source** — exactly one of:
- `--prompt "..."`: one prompt. Every dp rank generates it; only dp rank 0 writes the file.
- `--prompt-list "..." "..."`: ordered events of **one** video.
- `--prompt-csv a.csv [b.csv]`: a batch. Columns `index`, `seed`, and either `prompt` or `event_1..event_N`
  plus an optional `input_image` column. Read by `utils/prompt_csv.py::read_prompt_csv`.

CSV rules that cause hangs rather than errors if broken, so they are checked up front: `main` calls
`_load_prompt_inputs` (which runs `_check_input_images`) once before any weight loads.
- Every row must have the same number of events, and either every row has an `input_image` or none does. Each
  event is one text encode and an i2v row runs one extra transformer forward, collectives over all ranks, and dp
  ranks run different rows, so mixed rows desynchronise the FSDP all-gathers. One CSV's header fixes the event
  count of its rows; several `--prompt-csv` files that disagree are refused.
- Row count should be a multiple of the dp degree. Rows are padded cyclically otherwise, and a padded duplicate
  overwrites its original's `{index}.mp4`.
- `global_content_prompt` / `global_quality_prompt` are joined around every event text, but only when **both**
  are present, and never in a `prompt`-column CSV.

**Shape and output**:

```
--video-length 721            # 16n+1; 241 = 10 s, 721 = 30 s, 1441 = 60 s at 24 fps
--height 480 --width 864      # multiples of 16; default the model's 480x864, i2v too (see Image-to-video)
--input-image first.jpg       # i2v for --prompt / --prompt-list; CSV runs use the input_image column
--num-videos-per-prompt 1     # sequential, one shared generator, so the videos differ; costs wall time
--seed 930                    # noise seed of --prompt / --prompt-list (CSV rows use their seed column); default --global-seed
--global-seed 930             # per-process base: rank uses global_seed + rank // sp_size
--context-noise 0.15          # overrides the model's sigma; 0 turns it off
--output-type video           # video (mp4) | pt (decoded tensor, [0,1]) | latent (distilled only)
--output-dir ./outputs        # required
```

`--output-type pt` writes the decoded tensor in `[0, 1]`, which is what a PSNR comparison should read. Getting
the range wrong is a classic ~6 dB error.

**Topology**:

```
--sp-size 8                   # explicit eight-GPU setting; parser default 2, use 1 on a single GPU
--shard-world 8               # FSDP shard-group size (dp_shard * sp); default 8, must be <= world_size and
                              # divide it, so lower it below 8 GPUs (or --fsdp-sharding-strategy none)
--fsdp-sharding-strategy hybrid   # hybrid | full | none
--vae-parallel-mode none      # none | sp | custom  (PS-VAE row-sharded decode)
--vae-parallel-ranks "0,1;2,3"    # only with --vae-parallel-mode custom
--use-cpu-offload false       # true is refused: it does not work for inference, see below
```

**Kernel and precision**:

```
--attention-backend fa3       # fa3 | sage (SageAttention 2: ~1.4x, not bit-equal)
--precision bf16              # bf16 | fp16; autocast dtype of the transformer forward
--master-weight-type          # fp32 | bf16; default bf16 without FSDP, fp32 with it
--vae-precision fp16
--text-encoder-precision fp16
```

## Topology rules

From `main` in `movieroll/generate.py` and `ParallelDims._validate` in `commons/parallel_states.py`:

- `sp_size` must divide the model's 16 attention heads → one of 1, 2, 4, 8, 16.
- With FSDP: `world_size % shard_world == 0` and `shard_world % sp_size == 0`.
- Without FSDP: `world_size % sp_size == 0`.
- A single-process run never uses FSDP, whatever `--fsdp-sharding-strategy` says. `none` also drops it for
  several processes, and every rank then holds complete weights (fits in 96 GB).
- `hybrid` with `world_size == shard_world` degrades to `full` automatically (logged).
- `dp_replicate * dp_shard * sp == world_size` is asserted.
- `--vae-parallel-mode`: every VAE group must lie inside one sp group, because its ranks decode the same latents.
  Checked on every rank before raising, so one rank cannot exit while the others block in a later collective.

## Precision rules

The one non-obvious interaction, in `_resolve_master_weight_type`:

- **Default**: bf16 masters without FSDP (halves weight memory, 17 GB vs 33 GB — this is what lets one 96 GB GPU
  hold the pipeline), fp32 masters with FSDP.
- **bf16 masters + FSDP is refused up front.** The transformer keeps a few small parameters in fp32 whatever
  dtype it is built with (`byt5_in.*`, `final_layer.linear.*`, `cond_type_embedding`), and FSDP2 needs one dtype
  per parameter group.
- `--precision fp32` is not offered at all: it turns autocast off and the streaming FlashAttention kernels reject
  fp32 q/k/v.
- `--use-cpu-offload true` is refused by `parse_args`, before anything loads. It cannot work for inference: the
  pipelines take their device from the offloaded (cpu) parameters, so cpu and cuda tensors would meet after the
  weights load.

## Image-to-video

The input image becomes output chunk 0. `encode_input_image` (`utils/image_condition.py`) scales the image to
cover the target size, centre-crops it and VAE-encodes it into one latent frame, which is written into the KV
cache once, at timestep 0, before the rolling loop. It is never denoised, and it leaves `chunk_sizes`. Whether
that write adds context noise depends on the chain:

- **base**: no. `MovieRollStreamingPipeline.__call__` writes the frame with a plain `_encode_clean_chunk_kv`,
  before the context-noise function even exists. It is not exempt for good, though: chunk 0 is in every KV
  recompute selection, and a recompute context-noises everything it re-encodes, input frame included.
- **distilled**: yes, whenever the chain's sigma is above 0. `MovieRollDMDPipeline._execute_rollout` writes the
  anchor through `_encode_clean_chunk_kv_with_context_noise`, the same helper as the later history. The
  `distilled/i2v` folder sets `context_noise: 0.0`, so the frame goes in clean; `distilled/t2v` with
  `--input-image` (sigma 0.15), or `--context-noise` above 0 on `distilled/i2v`, noises the input frame too.

Sizing: with `--model`, an i2v run uses the folder's `config.json` resolution (480x864 for the released
models) unless `--height` / `--width` is given; the image is scaled to cover that size and centre-cropped, so an
image of another aspect ratio loses its edges. Only a bare `--dit-weight` run with neither flag sizes each row
from its image: `_closest_aspect_bucket` picks the candidate closest in aspect ratio, one-sided so that
cover-and-centre-crop always trims the image's longer side and never invents pixels. The bucket list comes from
`_aspect_bucket_sizes` at a roughly constant patch count (`I2V_BUCKET_BASE_SIZE` 640, patch 16, max side ratio
4), and since the choice is per row, such a CSV may mix aspect ratios.

The released i2v model is `distilled/i2v`. Its `config.json` differs from `distilled/t2v`'s only in
`context_noise: 0.0`, but its weights are a separate distillation (the `score_mode: acs` recipe,
`stage3_dmd_rolling_acs.yaml`), not the t2v weights.

## Multi-event

`_compute_event_split_indices` splits the chunks evenly across the events, on both chains; inference takes no
event timestamps. The distilled chain reaches it through `_compute_rollout_event_split_indices`, whose
pixel-frame snapping serves only the two-event training rollouts (`MovieRollDMDPipeline.execute` refuses
timestamps in `SAMPLING` mode). Events whose segment comes out empty are dropped (the base chain logs a warning,
the distilled chain refuses the request), and the prompt conditions are subset in lockstep.

Both chains switch the prompt whenever a row has two or more events. One continuous rolling loop runs across the
boundary — the queue and the KV cache are kept — so a transition spans up to `window_size` output chunks
denoised under mixed prompts.

With `kv_recompute_enabled`, an event boundary also triggers a KV recompute, so the new prompt attends a
compressed history. The two chains part ways when the selection (first chunk, each event's middle, the recent
ones) would keep every decoded chunk, which happens for a short event early in the video:

- **base** (`_pop_and_output_clean_chunk`): nothing can be dropped, so it falls back to a plain encode of the
  popped chunk, at an event boundary as at the threshold.
- **distilled** (`_execute_rollout`): only a threshold trigger falls back; an event boundary forces the
  recompute through anyway.

## Output layout

`_build_save_path`: a CSV run writes `<output-dir>/samples/<csv dir>/<csv name>/<index>.mp4`; a
`--prompt` / `--prompt-list` run writes `<output-dir>/samples/default/rank<dp>_sample_<i>.mp4`. With
`--num-videos-per-prompt > 1` the stem gains `_v0`, `_v1`, …

Only rank 0 of an sp group writes, and with a single prompt only dp rank 0 keeps the video. Skipping is
group-uniform, so a skipping rank never desynchronises the parallel decode collectives.

## VAE decode

The streaming (T-cache) decode is exact, not tiled: `decode_latent_chunks_streaming` in
`models/autoencoders/streaming_vae.py` carries convolution caches, attention K/V caches and temporal
up-sampling flags across chunks. `vae_stream_decode_max_chunk_size` (4 latent frames) bounds the peak.

The two chains read that value with **opposite** "decode in one piece" sentinels, which is why
`_check_vae_stream_decode_max_chunk_size` validates it per chain before any weight loads: distilled decodes in
one piece for `<= 0` and cannot take `None`; base decodes in one piece for `None` and otherwise needs a positive
int.

PS-VAE (`--vae-parallel-mode sp`) row-shards that decode across a group: each rank keeps a contiguous band of
latent rows, 3x3 convolutions exchange a halo with their neighbours (replicate padding at the global edges, so
the function computed does not change), and an all-gather hands every rank the complete chunk
(`models/autoencoders/row_shard.py`). `split_rows` needs at least one row per rank — at 480p there are only 30
latent rows.

Beyond `DEFAULT_STREAMING_EXACT_ATTENTION_FRAMES` (361 latent frames, about one minute at 24 fps) the VAE's
bottleneck attention becomes a sliding window, which **is** an approximation; `configure_streaming_attention`
sets or disables that budget.

## Failure modes

| Signal | Cause | Fix |
|---|---|---|
| `--video-length must be positive and 4n+1 pixel frames, got N` | not a `4n+1` frame count | a `16n+1` frame count |
| `--video-length N gives M latent frames, which is not 1 modulo chunk_size=4` | frame count arithmetic | use one of the two frame counts the error prints |
| `the distilled chain needs a first chunk of 5; its shortest video is 17 frames` / `image-to-video needs the input frame plus one chunk of 4, so its shortest video is 17 frames` | too short for the distilled chain / for i2v | raise `--video-length` to >= 17 |
| `--height must be a positive multiple of 16` | resolution | round to 16 |
| `exactly one of --prompt / --prompt-list / --prompt-csv is required` | prompt source | pick one |
| `exactly one of --model / --dit-weight is required` | weight source | pick one |
| `--sampling-mode X contradicts the Y model in <dir>` | `--sampling-mode` vs `config.json` | drop the flag |
| `<dir>/config.json: unknown keys [...]` | a key that is neither `sampling_mode` nor a profile setting | remove it; `load_model_dir` accepts only the profile's keys |
| `--output-type latent is only supported with --sampling-mode distilled` | base rescales latents as if pixels | use `pt`, or the distilled chain |
| `K of N rows have an input_image: either every row of a run is image-to-video or none is` | mixed CSV | split into two runs |
| `the --prompt-csv rows have [...] events: every row of a run must have the same number` | CSV files with different event counts | one run per event count |
| `--sp-size=N must divide the model's 16 attention heads` | topology | 1, 2, 4, 8 or 16 |
| `--shard-world=N must be divisible by --sp-size=M` | topology | align the two |
| `--shard-world must be in [1, W], got N` / `world_size=W must be divisible by --shard-world=N` | FSDP on fewer GPUs than the default `--shard-world 8`, or a non-divisor | `--shard-world` a divisor of the GPU count, or `--fsdp-sharding-strategy none` |
| `world_size=W must be divisible by --sp-size=M` | topology without FSDP | an `--sp-size` that divides the GPU count |
| `--use-cpu-offload does not work for inference` | `--use-cpu-offload true` | drop the flag; spread memory with FSDP or `--vae-parallel-mode` instead |
| `--master-weight-type bf16 cannot be sharded with FSDP` | precision | drop the flag (fp32 is the FSDP default) or `--fsdp-sharding-strategy none` |
| `FlashAttention only supports fp16, bf16, and fp8_e4m3 data type` | tried fp32 | `--precision bf16` |
| hang after the weights load, no error | a rank-dependent branch or random draw (mixed CSV rows are refused before loading) | never make a code path depend on the rank |
| OOM in the VAE decode on one GPU | the decode is already streamed in chunks of `vae_stream_decode_max_chunk_size` (4) latent frames; what remains is one chunk's full-resolution activations and caches next to the resident weights | `--vae-parallel-mode sp` on several GPUs, which row-shards every chunk's decode |
| `Total num of latent frames N (cond: ..., noisy: ...) exceeds max_num_latent_frames 120` | KV recompute off | keep `kv_recompute_enabled: true` |
| `KV recompute would exceed RoPE capacity: (kv_recompute_threshold(T) + window_size(W)) * chunk_size(C) + 1 = P > max_num_latent_frames(M)` | `kv_recompute_threshold` raised in `config.json` (refused by `StreamingConfig.validate` before the weights load) | lower the threshold (18 in the released models) |
| video drifts or degrades late in a long run | context noise / recompute disabled | keep the model's own `context-noise`; do not turn recompute off |
