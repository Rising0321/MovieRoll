# Surveying an unknown target repository

The target's structure is not known in advance, so the port starts with a survey. Answer every question below
**before** writing code, record the answers, and show them to the user — several of them change what gets
ported, not just how.

Work read-only in the target until the plan is agreed. Never write into this repository while porting out of it.

## How to find things

Generic starting points that work in most diffusion-video code bases:

```bash
# the denoising loop and the sampler
grep -rn "def step\|scheduler.step\|num_inference_steps\|sigmas" --include=*.py
# the attention implementation and whether a KV cache already exists
grep -rn "flash_attn\|scaled_dot_product_attention\|kv_cache\|past_key_value\|cache_k" --include=*.py
# positional encoding
grep -rn "rope\|rotary\|freqs_cis\|pos_embed" --include=*.py
# the transformer blocks and their count
grep -rn "class .*Block\|ModuleList\|depth\|num_layers" --include=*.py
# text conditioning
grep -rn "encode_prompt\|text_encoder\|cross_attn\|encoder_hidden_states" --include=*.py
# the VAE's compression factors
grep -rn "temporal_compression\|ffactor\|scale_factor\|downsample" --include=*.py
# parallelism
grep -rn "all_to_all\|sequence_parallel\|ulysses\|ring\|fully_shard\|FSDP" --include=*.py
# the training loop
grep -rn "def train\|backward()\|optimizer.step\|accelerator\|Trainer" --include=*.py
```

Then read the model's `forward` end to end once. Most of the questions below are answered by that one function.

## Question 0 — will the model be retrained?

Ask this first, because more follows from it than from anything in the list below:

- **Inference-side only** (an already-trained checkpoint nobody will retrain). Then the training-regime units
  are off the table, the G2 equivalence gate **cannot be run at all**, and correctness rests on G0b and G3
  (see `verification.md`). It also means P7's semantic dependency on P8 is a hard ceiling: rolling a model that
  never learned the regime will work and look worse, and no amount of inference-side work fixes that. Say so
  early rather than at the end.
- **Retraining or fine-tuning is possible.** Then the full stack is reachable, G2 becomes available, and the
  question is only how much of the recipe the user wants to adopt.

## The questions

### A. Architecture of the denoiser

| # | Question | Why it matters |
|---|---|---|
| A1 | Double-stream (text tokens inside self-attention, MMDiT-style) or single-stream with cross-attention? | Decides whether P2 needs the `*_mm_*` kernels and text-query isolation, or only the image-only kernels. A cross-attention host is *simpler*: port the image-only path and leave cross-attention alone. |
| A2 | Depth, head count, hidden size | `full_history_block_num` is a fraction of depth (6 of 54 here); `sp_size` must divide the head count |
| A3 | Temporal patch size | If it is not 1, chunk sizes are in patches and every frame↔token conversion gains a factor |
| A4 | Channel contract of the input projection | Here it is `2 * in_channels + 1` (latents, cond block, mask), a weight-shape contract of the checkpoint: the generator gets zeros in the two extra blocks in training and at inference, image-to-video included (the image enters through the history, see I1). A host without them needs nothing added; only the ACS score models (P10) fill them. |
| A5 | Is modulation per frame or per sample? | HieraKV drops cond frames mid-forward, so `vec` must be sliceable per frame alongside the tokens |

### B. Causality and the KV cache

| # | Question | Why it matters |
|---|---|---|
| B1 | Is there already a KV cache? Indexed by absolute frame or re-packed? | Absolute indexing must become compact, or training and inference diverge (P1) |
| B2 | Does every forward write the cache? | It must be split into a read-only denoising forward and a write-only timestep-0 encode. This is usually the single biggest change. |
| B3 | Is the attention already block/chunk-causal? | If yes, P2 may reduce to adding the prefix arguments; if it is fully bidirectional, the whole kernel comes over |
| B4 | Where is the cache stored and cleared? | P4 needs `soft_clear` + `replace`, P3 needs a pre-write trim |
| B5 | Does the cache depend on text? | It must not. If the host's self-attention mixes text into the cached KV, the CFG branches can no longer share one cache. |

### C. Positional encoding

| # | Question | Why it matters |
|---|---|---|
| C1 | RoPE axis order and per-axis dims | The resampler gathers rows by frame range; the axis layout decides the stride |
| C2 | Absolute positions, or is there a resampler? | P1 requires compact positions; a fused RoPE kernel must be checked for numerical equality after the change |
| C3 | Can different blocks get different RoPE tables? | P3's slot RoPE requires it |
| C4 | Is there a fixed maximum position grid? | That becomes `max_num_latent_frames`, and it is what bounds the recompute threshold |

### D. Scheduler

| # | Question | Why it matters |
|---|---|---|
| D1 | Flow matching or epsilon/DDPM? | The rolling Euler step and `velocity_to_x0` assume a flow-matching parameterisation |
| D2 | Does `step()` take a scalar index or can it take a per-frame tensor? | P7 cannot work with a scalar step counter; this almost always needs a small wrapper |
| D3 | Timestep range and shift convention | `sigma = t / num_train_timesteps` here; a host using `[0, 1]` needs the conversion done once, consistently |
| D4 | Does the host pick sigmas by table lookup / argmin? | Replace with direct computation, or per-frame sigmas will quantise |

### E. VAE

| # | Question | Why it matters |
|---|---|---|
| E1 | Temporal compression factor | The `latent_length = (frames - 1) // f + 1` formula and the whole `16n+1` arithmetic |
| E2 | Spatial downsample | The "multiple of 16" resolution rule becomes "multiple of that" |
| E3 | Causal in time? | P11 needs causal convolutions to carry a T-cache; a non-causal VAE cannot stream exactly |
| E4 | Latent channel count and scaling/shift factors | The input contract and the decode's unscaling |

### F. Parallelism

| # | Question | Why it matters |
|---|---|---|
| F1 | Is there sequence parallelism? Ulysses, ring, or none? | Every module that slices tokens (P3's drop-middle, P4's gather) must be SP-aware in the host's own idiom |
| F2 | Where does the all-to-all sit relative to QK-norm and RoPE? | The cached KV is taken after the all-to-all and **before** RoPE here, so a chunk can be reused at a new position |
| F3 | FSDP version, and can gradient sync be controlled per micro-step? | P10 requires `set_requires_gradient_sync` or an equivalent |
| F4 | Is there CFG parallelism? | Affects P6 and the cache-sharing argument |

### G. Training framework

| # | Question | Why it matters |
|---|---|---|
| G1 | Hand-written loop, `accelerate`, Lightning, something else? | Decides whether P8 inserts wrappers or needs a trainer port |
| G2 | Online VAE encoding or pre-encoded latents? | P8 works with either, but the data contract and the cost differ |
| G3 | Where is the loss computed, and can layout wrappers be inserted before it? | The recommended shape of the port |
| G4 | Does the host have a DMD / score-distillation setup already? | P10 reuses it if so; otherwise P10 is a much bigger job |

### H. Text conditioning

| # | Question | Why it matters |
|---|---|---|
| H1 | One text encoder or several? Any glyph/OCR branch? | Extra branches must be routed through the same per-window logic |
| H2 | Can text be made per-window? | P8 conditions each window on its own text |
| H3 | Does text reach the history? | It must not (see B5) |

### I. Image-to-video

| # | Question | Why it matters |
|---|---|---|
| I1 | Does the host do image-to-video, and how does the image get in (concatenated channels, image-encoder tokens, a clean first latent frame)? | The rolling generator here needs none of these: the image is a 1-frame chunk 0 of the clean history (P7, trained by P8). The host's own mechanism matters only for the teacher, and only if ACS is ported (P10). |
| I2 | Does the VAE encode one image to one latent frame? | That latent is the anchor (`[1, C, 1, h, w]` here), and E1's `4n+1` arithmetic is what leaves exactly one frame for it |

## Host classification, and what follows

**Cross-attention DiT (text outside self-attention).** The common case for recent video models. Port the
image-only FBCA kernels; skip the mm variants, the text-query isolation and the text-mask packing. Smaller job,
and the "history is text-independent" invariant holds for free.

**MMDiT (text inside self-attention).** Like this repository. Port both kernel families and keep the rule that
a block's text queries see only that block.

**Already block-causal with a KV cache** (a self-forcing-style model). The cheapest, most valuable subset is
P5 + P3 + P4 + P6: context noise, bounded cache, recompute and per-frame CFG, with no change to the training
recipe. Then P9/P10 if few-step distillation is wanted.

**Bidirectional, non-causal.** The full stack is needed, in order: P1 + P2, then P5, then P8, then P7, then
P3/P4, then P9, then P10. Expect the KV-cache split (B2) and the per-frame scheduler (D2) to be the two hardest
changes.

## Record the survey

Write the answers down as a table of `question → answer → consequence`, and keep it next to a decisions log,
because the answers are what later decisions are justified by. The record never goes into this repository (this
skill's own folder included — it is part of the source, which stays read-only). While the plan is not yet agreed,
keep it in a scratch directory of your own; once it is agreed, move it into the target next to the ported code if
the user wants it there. State explicitly which questions you could not answer and what you assumed; an
unrecorded assumption here turns into a silent numerical bug three modules later.
