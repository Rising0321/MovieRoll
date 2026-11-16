# The portable logic of MovieRoll

Eleven units. Each one says what it does, where it lives here, what the host must already provide, the
invariants that must hold after the port (these fail silently, not loudly), and whether it can be taken alone.

Code is named by file, class and function, never by line number; file paths are relative to `movieroll/`.
**This repository is the source of truth** — when this document and the code disagree, the code is right and
this document should be fixed.

Dependency summary:

```
P1 streaming core ─┬─ P2 FBCA attention        (both are foundation: anything touching attention needs them)
                   ├─ P3 HieraKV
                   ├─ P4 KV recompute
                   ├─ P5 context noise
                   ├─ P7 rolling inference ─┬─ P9 rolling ODE init ── P10 CuLT-DMD
                   └─ P8 WinTF training ────┘
P6 AnnealCFG      (independent)
P11 PS-VAE        (independent)
```

That graph shows what *reinforces* what. It is **not** a list of things you must port together: most units go
over on their own. Read it together with the table below, which is the one to use when the user names a subset.

## Porting a single unit

Two kinds of dependency get conflated, and separating them is what makes a one-unit port possible:

- **Capability** — something the host must be able to do. If it already can, you port nothing extra. If it
  cannot, you either implement it in the host's own terms or bring over the corresponding part of P1/P2.
- **Semantics** — the unit will run either way, but it is pointless or harmful without its partner. P7 is the
  case to watch: a model never trained in the rolling regime will roll and look bad.

| Unit | Alone? | What the host must already be able to do | Rough size |
|---|---|---|---|
| P1 parts | yes, per part | nothing for the arithmetic and the queue | small each |
| P2 FBCA | yes | varlen attention for the fast path, or accept the masked fallback | medium |
| P3 HieraKV | yes | KV cache, per-block visibility, per-block RoPE | medium |
| P4 KV recompute | yes | KV cache, a timestep-0 encode path, access to decoded latents | medium |
| P5 context noise | yes | have a "clean history" whose timestep label you can pin at 0 | tiny |
| P6 AnnealCFG | yes | CFG, and per-frame timesteps | tiny |
| P7 rolling inference | code yes, meaning no | cache, per-frame scheduler, **a model trained in this regime** | medium |
| P8 WinTF training | yes | take history as input tokens; insert wrappers before the loss | large |
| P9 ODE init | no | P7 and P8 are real prerequisites | medium |
| P10 CuLT-DMD | no | P7 to roll the generator, a few-step start (P9 or the host's own), a DMD/score setup, controllable FSDP gradient sync | large |
| P10 ACS (i2v variant) | no | P10, plus a bidirectional teacher with its own image-to-video conditioning | medium |
| P11 PS-VAE | yes | a causal 3D VAE that can carry a cache | medium |

### Scenarios

What people actually ask for, and the smallest answer:

| "..." | Port | Note |
|---|---|---|
| my causal model's KV cache grows until it OOMs | **P3**, then P4 | no change to the training recipe |
| long videos drift, wash out or freeze | **P5** | the cheapest fix in the list; try it before anything structural |
| my multi-step causal sampling looks mediocre | **P6** | only helps if a window spans several noise levels |
| the VAE decode is the bottleneck / OOMs | **P11** | fully independent of the rest |
| I want to know whether this method is worth it | **P2 masked fallback + P7** | measures the gap on an untrained model; do not ship it |
| I already distil a few-step model; make long rollouts hold up | **P10**, started from that model instead of P9 | still needs P7; CuLT-DMD here is rolling only |
| I want image-to-video from the rolling model | the i2v layout of **P7**, trained by **P8**'s i2v draw | no new input: the image is chunk 0 of the clean history; add ACS (under P10) only when distilling an i2v model |
| I want the whole method | P1 + P2 → P5 → P8 → P7 → P3 → P4 → P9 → P10 | in that order; gate each one |

When the user names a unit, do this rather than widening the scope on your own:

1. Read that unit's **Standalone?** and **Add another unit only if** lines.
2. Check each required capability against the survey answers.
3. Only for capabilities the host genuinely lacks, propose the extra unit — and say what it costs.
4. If a unit is a *semantic* dependency rather than a capability one, say so plainly and let the user decide:
   "this will run without P8, and it will look worse; porting P8 is a training-side job."

---

## P1 — Streaming core

**What** The chunk/window layout, the rolling queue, the KV cache manager with compact frame indices, compact
RoPE, and the rule that training and inference are two regimes of one model.

**Here** `commons/streaming_config.py` (`StreamingConfig`), `models/transformers/modules/chunking.py`
(`check_and_get_chunk_sizes`, `check_and_get_window_sizes`), `models/transformers/modules/rolling_queue.py`
(`RollingQueue`), `models/transformers/modules/kv_cache.py` (`KVCacheManager`),
`models/transformers/modules/rope.py` (`RoPEResampler`).

**Host must provide** A DiT whose attention you can reach and modify; a flow-matching scheduler you can step
per frame; a VAE with a known temporal compression factor.

**Invariants**
- `latent_length % chunk_size == 1`. The leftover frame is either swallowed by the first chunk (t2v, first
  chunk `1 + C`) or becomes the reference-frame chunk (i2v, first chunk `1`).
- The KV cache is numbered **compactly from 0**, not by absolute frame index: a cached chunk's RoPE position is
  its position in the re-packed cache, and the current window sits immediately after the cache end
  (`RoPEResampler._resample_inference`). A host that uses absolute positions must be changed, or training and
  inference will see different relative distances.
- **The denoising forward never writes the cache.** Only the timestep-0 encode writes it, under `no_grad`. A
  host whose attention writes KV on every forward must have the two paths split.
- Nothing may depend on the rank: every shape-affecting draw happens on sequence-parallel rank 0 and is
  broadcast.

**Porting notes** Tensor layout differs between code bases (`[B, C, T, H, W]` here). Decide the layout once and
convert at the boundary, not per module. If the host's chunks are all equal (no `1 + C` first chunk), say so
explicitly and drop the leftover frame — but then i2v and the "first chunk is whole" checks change too.

**Standalone?** In parts, and that is often the right move. The chunk/window arithmetic
(`check_and_get_chunk_sizes`) and the rolling queue are pure functions with no dependency — port them first and
alone. The cache manager, the compact RoPE and the two-regime split each stand on their own as well; only
together do they make up "the whole core". Other units name the specific parts of it they need.

**Add another unit only if** — nothing. This *is* the base.

---

## P2 — FBCA, the streaming attention kernel

**What** Flash Block-wise Causal Attention. A *block* is one query unit that attends bidirectionally inside
itself, causally to the blocks before it, plus an optional shared prefix. A block is one chunk in the clean
history and in recompute, and one whole training window in the noisy branch.

**Here** `models/transformers/modules/streaming_attention.py`: `block_wise_causal_attention_with_prefix`
(image-only), `block_wise_causal_mm_attn_with_prefix` (with text), `full_history_parallel_attention`,
`sink_recent_parallel_attention`, `chunk_inference_attention`; dispatched by
`models/transformers/modules/attention.py::parallel_attention`.

**Host must provide** A varlen attention kernel (FlashAttention-2/3 here) or an equivalent; the ability to
replace a block's attention call.

**Invariants**
- Ragged visible-KV sets are packed with `gather_concat` into one varlen call — no materialised masks. Groups
  of blocks bound the packed KV so backward can free each group (`packed_kv_max_mem`).
- **Image queries see the clean history; text queries of a block see only that block's own `[img, txt]`.** The
  history is therefore text-independent, which is why one KV cache serves both CFG branches.
- In the full-history training regime, a noisy window's causal prefix is taken from the **clean copy at the
  tail of the cond tokens**, not from the noisy tokens. This is what keeps training windows independent of each
  other and makes them match the inference rollout.
- Valid text must be a **prefix** of the text tokens (the packing keeps the first `valid_len` rows).

**Porting notes** If the host is a cross-attention DiT whose self-attention contains no text tokens, port only
the image-only kernels and skip the `*_mm_*` variants, the text-query isolation and the text-mask packing. If
the host is an MMDiT, port both. Blocks of unequal size: here block 0 is handled by its own self-attention and
folded into the full prefix, which turns the rest into an equal-size problem (recursion depth 1).

**Standalone?** Yes, as a kernel: it is a function from (q, k, v, block sizes, prefixes) to an output,
testable without any of the rest.

**Degraded option worth knowing.** If you need the *visibility rule* and not the speed, implement it with an
explicit boolean mask and the host's ordinary attention. That is much simpler, it is correct, and it is exactly
the reference that gate G3 compares the packed kernel against. Start there when the port is exploratory, and
swap in the packed kernel once the method has proved itself.

**Add another unit only if the host lacks** a varlen attention kernel and you want the fast path — nothing to
port, but FlashAttention-class varlen attention has to be available.

---

## P3 — HieraKV

**What** Hierarchical KV compression. Transformer blocks `[0, full_history_block_num)` let every noisy chunk
attend over the whole clean history; the remaining blocks see only the sink chunk plus their own most recent
clean chunk. This is what bounds the cache.

**Here** `models/transformers/movieroll_transformer.py`: `sink_recent_block_indices`,
`_sink_recent_window_layout`, `_drop_middle_history`; `models/transformers/modules/kv_cache.py::trim_to_sink`;
`models/transformers/modules/rope.py::resample_training_slot`; the kernel side is `sink_recent_parallel_attention`.

**Host must provide** Per-block control of what the attention sees, and a way to give different blocks
different RoPE tables.

**Invariants**
- `sink_chunks == 1` and `recent_chunks == 1`. Anything else makes the sink-recent blocks a sliding window that
  training does not reproduce (`StreamingConfig.validate` refuses it and explains why).
- **The trim happens before the new chunk is encoded**, not after (`trim_to_sink` is called for the
  sink-recent blocks at the top of the forward when `return_kv` is set). Afterwards the cache is back to
  `{sink, recent}`.
- Slot RoPE in training: sink at slot 0, recent at slot 1, the window at slot 2 — the same relative distances
  the rollout gives it. `_sink_recent_window_layout` decides per window whether it has a sink and a recent
  (`has_recent ⇒ has_sink`), and that decision is the single source of truth shared by the RoPE and the kernel.
- The cond tokens dropped at the boundary are dropped from `img`, from `vec` and from the SP padding info
  together, and `attn_param` must be snapshotted per block call (activation checkpointing re-runs a block's
  forward in backward with the arguments it was given).

**Standalone?** Yes, when the host already has a KV cache and per-block control of visibility. For a host
that already generates causally but whose cache grows without bound, this is the highest-value single unit in
the list.

**Add another unit only if the host lacks** a KV cache → P1's cache manager; re-orderable positions → P1's
compact RoPE (the slot RoPE needs it); a block-causal attention path → P2, or P2's masked fallback.

---

## P4 — KV recompute

**What** When the cache reaches a threshold (or an event boundary is crossed), rebuild a compact cache from the
decoded latents instead of keeping growing: keep the first chunk, the middle of every event, and the
`kv_recompute_recent_k` most recent, concatenate them and re-encode at timestep 0 through FBCA.

**Here** `pipelines/movieroll_pipeline.py`: `_select_recompute_chunks`, `_kv_recompute`;
`models/transformers/movieroll_transformer.py`: `_forward_kv_recompute`, `_recompute_sink_recent_cache`;
`models/transformers/modules/kv_cache.py`: `soft_clear`, `replace_kv_cache`, `should_recompute`.

**Host must provide** P1's split encode path, and access to the decoded clean latents during the rollout.

**Invariants**
- **Both block groups are rebuilt from the same forward.** Context noise may have re-sampled the sink, so an
  old sink-recent cache would correspond to a different version of the input than the new full-history one.
  The sink-recent group takes the sink and the recent out of the hidden states *after* the full-history blocks.
- The rebuilt cache is numbered compactly from 0 (that is the point), and the returned value is the new
  `current_frame_index`.
- If the selection covers every chunk there is nothing to compress. The base pipeline
  (`_pop_and_output_clean_chunk`) then falls back to a plain encode whatever triggered it; the DMD pipeline
  (`MovieRollDMDPipeline._execute_rollout`) falls back only for a threshold trigger and forces an event switch
  through anyway.
- It must fire before the cache outgrows the RoPE grid: `(threshold + window) * chunk + 1 <= max_num_latent_frames`.

**Standalone?** Yes, on a host that has a cache and can run a timestep-0 encode. It pairs naturally with
P3 but does not require it — on a full-history-only model it still bounds the cache.

**Add another unit only if the host lacks** a write-only encode path → P1's two-regime split (this is the usual
blocker); and P3 only if you also want the sink-recent group rebuilt.

---

## P5 — Context noise

**What** The clean history is perturbed, `(1 - sigma) * clean + sigma * eps`, before the model sees it, while
its timestep label stays 0. Training draws sigma per chunk at random; inference uses one fixed sigma.

**Here** `models/transformers/modules/context_noise.py` (`apply_context_noise`, `sample_context_noise_sigma`);
called from the pipelines' `_add_context_noise` and from `trainer/flow/streaming_wrappers.py`.

**Host must provide** Nothing beyond P1.

**Invariants**
- The timestep label of the history stays **0** even though it is noised. The model learns to treat a slightly
  corrupted history as clean, which is what stops error accumulation.
- Inference draws the noise from the sample's own generator, so a sample does not depend on which rank it ran
  on or on the samples before it. Under sequence parallelism only sp rank 0 draws, and it broadcasts.
- Sigma 0 is not the same as "off" on the base chain: at 0 it would still consume a draw from the generator and
  shift every later chunk's noise, so 0 disables the mechanism instead (`generate.py::_apply_context_noise`).

**Standalone?** Yes, and it is the cheapest real win in the list — on the order of twenty lines. All it
needs is that the host has a notion of "clean history" and that you can keep that history's timestep label at 0
while perturbing its values.

**Add another unit only if** — nothing.

---

## P6 — AnnealCFG

**What** A per-frame CFG scale, linear in the timestep: `w(t) = cfg_min + (cfg_max - cfg_min) * t / 1000`.
High guidance where the window is noisy, low where it is nearly clean.

**Here** `schedulers/scheduling_flow_match_streaming.py::AnnealCFG`, applied in
`pipelines/movieroll_pipeline.py::_compute_cfg`.

**Host must provide** CFG at all, and the ability to multiply by a per-frame tensor rather than a scalar.

**Invariants** Interpolate in the **timestep** (a physical noise level), not in the step index — that is what
keeps the scale meaningful when the frames of one window sit at different points of the ladder. The scale is
broadcast as `[1, 1, T, 1, 1]`.

**Standalone?** Yes, fully. Needs CFG and a per-frame timestep, and multiplies by a `[1,1,T,1,1]` tensor
instead of a scalar.

**Add another unit only if** — nothing. Note that if every frame of the host's window shares one timestep, this
unit has nothing to do: it only matters when a window spans several noise levels.

---

## P7 — Rolling inference

**What** The loop: pop the oldest clean chunk, encode it into the cache, push a fresh noisy chunk, denoise the
whole window, repeat. `num_chunks + window_size` rounds. Multi-event switches the prompt at a boundary.

**Here** `pipelines/movieroll_pipeline.py::__call__` and `_generate_strategy_prompt_switch` (base, multi-step +
CFG); `pipelines/movieroll_dmd_pipeline.py::_execute_rollout` (few-step, one step per round, four modes);
`schedulers/scheduling_flow_match_streaming.py::StreamingSchedulerWrapper` for the per-frame step indices.

**Host must provide** A scheduler whose `step` accepts a per-frame index tensor, not a scalar step counter.

**Invariants**
- Every frame of the window is at its own point of the ladder, so one global step maps to a **vector** of step
  indices (`StreamingSchedulerWrapper`'s cyclically shifted table). The scheduler keeps no counter; the caller
  owns the index.
- Euler per frame: `prev = sample + v * (sigma[i+1] - sigma[i])`, computed in fp32.
- The prompt of a round is the one belonging to the **cleanest** chunk in the window, so a transition spans up
  to `window_size` chunks.
- The globally last chunk is not encoded into the cache — nobody would read it.

**Image-to-video** The input image is not a new kind of input: it is chunk 0 of the clean history, the anchor.

- Layout `[1, C, C, ...]` (`check_and_get_chunk_sizes(..., mode='i2v')`): P1's leftover frame becomes a chunk of
  its own instead of joining the first one, so the shortest i2v video is that frame plus one generated chunk
  (`generate.py::_check_i2v_video_length`).
- The image becomes one latent frame in `utils/image_condition.py::encode_input_image`: scaled to cover the target
  size, centre-cropped, mapped to `[-1, 1]`, VAE-encoded taking the posterior **mode** (no random draw), shifted
  and scaled like the training latents, and broadcast from sp rank 0.
- Before the first round it is written into the cache once, at timestep 0, by the same encode as any clean chunk.
  It is output chunk 0 and is never denoised, so the rounds count only the chunks after it
  (`MovieRollStreamingPipeline.__call__` and `MovieRollDMDPipeline._execute_rollout`, both on `cond_latents`).
- The transformer input stays the t2v layout: the condition block and the mask of the `2 * in_channels + 1`
  input are zeros in both tasks (`prepare_latent_concat`; `utils/multitask_utils.py::get_multitask_mask` always
  returns the t2v mask). The generator has no image encoder.
- Being chunk 0, the anchor is the HieraKV sink (one latent frame instead of `1 + C`), and every recompute
  selection keeps chunk 0 (`_select_recompute_chunks`, `_select_training_recompute_chunks`), so every block sees
  it for the whole video.
- Context noise: the base chain encodes the anchor without it, but a later recompute re-encodes it, noise
  included, together with the rest of the selection; the DMD chain sends it through the same helper as the later
  history (`_encode_clean_chunk_kv_with_context_noise`). The released distilled i2v weights are sampled at context
  noise 0 (`configs/models/distilled/i2v/config.json`), the distilled t2v ones at 0.15.
- An i2v sample runs one transformer forward more than a t2v one, so ranks that share FSDP collectives must agree
  on the task: a run is all-i2v or all-t2v (`generate.py::_check_input_images`), and Stage 3 draws the task once
  per shard group.

The host must provide, for i2v: a VAE that encodes a single image to one latent frame, and a model that saw the
`[1, C, ...]` layout in training (see P8).

**Standalone?** As code, yes, given a cache and a per-frame scheduler. **Semantically, no**: a model that
was not trained in this regime will roll, but the quality will be poor. Porting P7 alone is a legitimate way to
*measure* that gap, and a bad way to ship.

**Add another unit only if the host lacks** a per-frame `step()` → write the wrapper (small; see D2 in
`survey.md`); a split encode path → P1's two-regime split; training in this regime → P8, or P9/P10.

---

## P8 — WinTF training (window-wise teacher forcing)

**What** Train many rolling windows in one forward. Inside a window the noise increases chunk by chunk; the
condition is the ground-truth clean history; windows are isolated from each other.

**Here** `trainer/flow/streaming_wrappers.py` (the split/timestep/merge/text wrappers),
`models/transformers/movieroll_transformer.py::forward` with `cond_chunk_sizes` set and `training=True`,
`full_history_parallel_attention` on the kernel side.

**Host must provide** A trainer you can insert wrappers into, and a transformer forward that accepts the
history as input tokens.

**Invariants**
- The window grid is offset by an explicit phase: `phase ~ U{0, window_size-1}`, the first window holds
  `phase + 1` chunks, the rest are full, and a trailing remainder is dropped. The phase must be drawn on sp
  rank 0 and broadcast. The same phase convention is used by P10's `start_phase`, so the two stages mean the
  same thing by the same number.
- Windows must be **independent**: a window's causal prefix comes from the clean tail, never from another
  window's noisy tokens (see P2).
- The history's timestep label is 0 (with P5, even when noised).
- Text acts per window only; the history is never conditioned on text.

**Image-to-video** Trained by the same forward. Each step draws its task form on sp rank 0 and broadcasts it
(`StreamingMultitaskManager.sample_task_type`: i2v with probability `1 - streaming_t2v_prob`, 0.2 in the released
SFT configs). An i2v step lays the clip out `[1, C, ...]` and keeps that 1-frame chunk as the clean prefix
(`get_split_index` returns 1), so the clip's own first frame stands in for the anchor, at timestep 0 among the
cond tokens — where inference puts the image in the cache. A t2v step has no clean prefix. Context noise treats
the prefix like any other history.

**Porting notes** This is the module most entangled with the host's trainer. Decide early whether to reuse the
host's loss and data path (recommended) and insert only the layout wrappers, or to port the trainer too.

**Standalone?** Yes as a training recipe, if the host's transformer can take the history as input tokens
and you can insert layout wrappers ahead of the loss. It is the unit most entangled with the host's trainer, so
scope it deliberately.

**Add another unit only if the host lacks** window-isolated attention → P2. That one is not optional: without
the clean-tail prefix rule, training windows leak into each other and training stops matching the rollout.

---

## P9 — Rolling ODE initialization

**What** A frozen teacher runs the full rolling inference; each round's diagonal window is captured; the
few-step student regresses the teacher's clean output in the WinTF layout.

**Here** `pipelines/movieroll_ode_pipeline.py` (the two capture hooks and
`rolling_ode_teacher_forcing_forward`), `scripts/precompute_ode.py`, `trainer/ode/trainer.py`.

**Host must provide** P7 (to roll the teacher) and P8 (the layout the student is trained in).

**Invariants**
- The captured window is the teacher's **actual rollout content**, including windows that run past the end of
  the clip — do not clip them to the video boundary; pad and mask instead.
- Out-of-range frames are masked by **boolean indexing before the MSE**, not by multiplying a mask (a `0 * inf`
  becomes NaN).
- Round sampling is uniform **over rounds**, not over chunks, or the prefill rounds are badly under-trained.
- One student forward per step; the teacher does not run during student training.
- The trajectory cache is only bit-reproducible under the topology that produced it (the per-prompt seed
  includes the sp group's first rank).

**Standalone?** No — it needs something to roll the teacher with (P7) and the layout to regress in (P8).
It is also skippable: if the host already has a usable few-step initialization, go straight to P10.

**Add another unit only if** — P7 and P8 are genuine prerequisites here, not capability substitutions.

---

## P10 — CuLT-DMD

**What** Cumulative Long Tuning with distribution matching. One long self-rollout; every selected window's DMD
loss is backpropagated the moment the window exists; the whole rollout produces one optimizer step.

**Here** `trainer/dmd/loss.py` (`compute_dmd_loss_per_window`, `_assemble_teacher_window`,
`_compute_kl_grad`), `trainer/dmd/score_window.py`, `pipelines/movieroll_dmd_pipeline.py` (the
`GENERATOR_DMD_TRAINING` / `FAKE_SCORE_TRAINING` modes and the `DMDWindowBackwardHandler` protocol),
`pipelines/rolling_dmd_runtime.py` (the typed contracts, which validate themselves).

**Host must provide** A DMD or score-distillation setup (real score, fake score/critic) and an FSDP wrapper
whose gradient sync you can control.

**Invariants** — these are the ones that hang instead of failing:
- **Generator weights are fixed for the whole rollout**: one `zero_grad` before, one `clip` + `step` after.
- History and KV are **detached**; only the current window carries gradient.
- Each window backwards immediately with weight `loss_scale / K` and its graph is freed before the next
  window's forward.
- **Every rank must have the same K.** Checked with a WORLD `all_reduce(MAX)` on `[K, -K]` before any generator
  forward. `num_chunks % window_size == 0` is enforced so K does not depend on the drawn phase.
- A non-finite loss on any rank stops **every** rank together (`_require_finite_world`); a non-finite gradient
  norm on any rank makes **every** rank skip the step, including the EMA update (a NaN in the EMA shadow is
  permanent).
- Gradient sync is enabled only on the last micro-step, or gradients get all-reduced once per micro-step.

**Standalone?** No. CuLT-DMD in this repository is rolling only: `RollingMode` and
`RollingTrainingObjective` in `pipelines/rolling_dmd_runtime.py` have the rolling DMD objective plus the
uniform-timestep regularizer, both over a self-rollout, so it needs P7 and a good few-step start. That start can
be the host's own few-step model rather than P9, which is the subset to offer when the user wants "better
distillation" rather than "the whole method". A variant without rolling inference (block-wise DMD on top of P1/P2
alone) does not exist here; if the user wants one, it is a design the porter would have to work out and verify
from scratch, and it should be proposed as such.

**Add another unit only if the host lacks** a rolling few-step generator → P7; a few-step start → P9; a DMD /
score-distillation setup → that is a large job of its own and outside this list; controllable FSDP gradient
sync → required, see the invariants above.

### ACS — the image-to-video variant of P10

**What** Anchor-conditioned scoring (`dmd.score_mode: acs`), the recipe that distils the i2v weights. The
generator rolls i2v exactly as at inference (P7: the anchor in the cache, nothing else), and in training the
anchor is self-generated: a short t2v rollout without gradients yields one frame, detached, which then anchors
an independent i2v rollout (`RollingMode.I2V_ANCHOR_GENERATION` returns frame 0 of the first chunk it pops). Only
the scoring changes. A window whose scoring crop starts at latent frame 0 (`score_start == 0`: the first selected
window, and with some start phases the second) is scored the way the bidirectional teacher does i2v — the anchor
latent in the condition channels on frame 0, a mask of 1 on frame 0, and the anchor's SigLIP tokens merged into
the text sequence through `vision_in`. Every later window is scored as t2v, so the teacher is never conditioned on
generated frames.

**Here** `trainer/dmd/loss.py` (`AnchorCondition`, `ConcatModelWrapper`, `MovieRollDMD._anchor_condition`),
`trainer/dmd/score_window.py::CausalVaeScoreWindowReEncoder.decode_condition_frame`,
`models/image_encoders/__init__.py` (`SiglipImageEncoder`), `models/transformers/score_transformer.py`
(`vision_in`), `pipelines/movieroll_dmd_pipeline.py::MovieRollDMDPipeline.execute_i2v_self_anchored`; the checks
are in `trainer/dmd/config.py` and `trainer/dmd/setup.py`, the recipe is `configs/stage3_dmd_rolling_acs.yaml`.

**Host must provide** Everything P10 needs, plus a bidirectional teacher with its own i2v conditioning (here
condition-latent channels, a mask and an image encoder's tokens). ACS reuses the teacher's convention rather than
defining one. The generator needs nothing new.

**Invariants**
- The teacher and the critic start from one checkpoint (`real_score_ckpt == fake_score_ckpt`), or their
  difference is a systematic part of the DMD gradient from step 0.
- Fake-conditional, real-conditional and real-unconditional share one anchor condition: the CFG unconditional
  branch drops only the text and keeps the image. The critic's own flow-matching loss sees the same condition.
- The condition is built once per rollout — one single-frame VAE decode with tiling off and no streaming cache,
  one SigLIP call — and broadcast, so every rank of the group conditions on bit-identical tokens.
- i2v only (`video.task_sampling.t2v_prob: 0`): a t2v rollout has no anchor, and its first window would be
  conditioned on a generated frame.
- Only the score models build `vision_in`; the generator does not, and `vision_in.*` is the one optional
  checkpoint key (`utils/load.py`).
- `vision_in` gets a gradient only from windows scored with the image, and which window that is follows each
  shard group's own start-phase draw. ACS therefore needs a single FSDP shard group (`trainer/dmd/setup.py`
  refuses several replicas, whose all-reduce would see different parameter sets and hang), and the optimizer
  must accept a parameter without a gradient (`optim/muon.py` substitutes zeros).
- A checkpoint resumes only under the score mode it was trained with (`DMDTrainer._apply_pending_runtime_state`).
- Training keeps context noise 0.15 on the history, anchor included; samples and inference use 0
  (`dmd.sampling_context_noise`).

**Standalone?** No: it is P10 with a different scoring of one window. The released t2v recipe already trains
self-anchored i2v rollouts (`t2v_prob: 0.8`) and scores them as t2v; that is the fallback when the host's teacher
has no i2v conditioning.

**Add another unit only if the host lacks** P10 → port P10 first; a teacher that does i2v → ACS is out of reach,
use the fallback above.

---

## P11 — PS-VAE

**What** Parallel Streaming VAE decode: an exact T-cache stream over latent chunks, optionally row-sharded
across ranks.

**Here** `models/autoencoders/streaming_vae.py` (`forward_stream` on each module,
`DecoderStreamState`, `decode_latent_chunks_streaming`, `enable_parallel_decode`),
`models/autoencoders/row_shard.py` (`RowShard`, `split_rows`, `exchange_halo`, `gather_rows`).

**Host must provide** A causal 3D VAE whose convolutions and bottleneck attention you can give a cache to.

**Invariants**
- The streaming path must compute the **same function** as the whole-video path: convolution caches hold the
  last `kernel_t - 1` input frames, and the halo rows at the global edges are exactly the replicate padding the
  single-device path would have applied.
- Streaming is inference-only (`no_grad` + `eval`), one stream per instance, and it bypasses tiling.
- Beyond the exact-attention budget (361 latent frames by default, ~1 minute at 24 fps) the bottleneck
  attention becomes a sliding window — that **is** an approximation, and it is a deliberate choice.
- `split_rows` needs at least one latent row per rank (480p has only 30).

**Standalone?** Yes, completely independent of every other unit. Needs a causal 3D VAE whose
convolutions and bottleneck attention can be given a cache. Port it when decode is the bottleneck, which for
long videos it is.

**Add another unit only if** — nothing.
