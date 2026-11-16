# Verifying a port

## The thing to understand first

When you port into a different host, **there is no reference output to compare against.** The target has
different weights, a different VAE and a different text encoder, so comparing its video with MovieRoll's is
meaningless. PSNR against this repository is not a gate — it is not even a signal.

What replaces it is **self-consistency**: the ported code must be internally equivalent in the one way the
method depends on.

> Training and inference must be two regimes of **one** function. Every token must see the same set of KV
> entries and the same RoPE relative distances in both regimes.

This never raises. The shapes match, the loss goes down, the samples look plausible — and the model has learned
a distribution that deployment does not reproduce. G2 below is therefore the gate that matters most, and it is
worth building before the units that depend on it.

Two caveats, both common:

- **G2 needs both regimes.** A purely inference-side port into a model nobody will retrain cannot run it. There
  G0b (degenerate-configuration equivalence) and G3 (against an explicit-mask reference) carry the weight.
- **A unit ported alone still needs gates**, just fewer of them — see the per-unit table at the end.

## The gates

### G0 — Reversible switch

Every ported mechanism sits behind a config flag, defaulting to the host's original behaviour. With every flag
off, the host must produce **bit-identical** output to the commit you started from. Check this once per module;
it is what lets you bisect a regression later.

### G0b — Degenerate-configuration equivalence

Stronger than G0 and almost free, and it is the main gate for a unit ported **alone**. Every unit has a
parameter value at which it should reduce to the host's original behaviour. At that value the output must be
bit-identical with the flag *on*:

| Unit | Degenerate setting | Must equal |
|---|---|---|
| P3 HieraKV | `full_history_block_num = depth` (no sink-recent block) | the host's original full-history attention |
| P4 KV recompute | threshold above any reachable cache size | never recomputing |
| P5 context noise | sigma 0 | an unperturbed history (watch the generator: a draw at sigma 0 still shifts later noise, so 0 must *skip* the draw) |
| P6 AnnealCFG | `cfg_min == cfg_max` | the host's scalar CFG at that value |
| P11 PS-VAE | one rank, exact-attention budget disabled | the whole-video decode |
| P2 FBCA | one block, no prefix | the host's ordinary full attention |

A unit that fails G0b is wired in wrongly, and no amount of staring at the attention maths will show it.

### G1 — Shapes and contracts

Unit level, CPU or a tiny model, fp32. Cheap and fast: chunk/window arithmetic, the compact frame indices, the
cache's frame ranges after a trim and after a recompute, the step-index table, the mask that selects supervised
frames. Assert the *indices*, not just the shapes — an off-by-one chunk is invisible in a shape check. With
image-to-video, also assert the `[1, C, ...]` layout and that chunk 0, the anchor, is still in both block groups'
caches after every trim and every recompute.

### G2 — Training/inference equivalence (the one that matters)

**Prerequisite:** the host must have *both* regimes — a training forward that takes the history as input tokens
and an inference path that keeps it in a cache. A purely inference-side port into an already-trained model
cannot run G2 at all; use G0b plus G3 instead, and say in the record that G2 was not applicable and why.

Small model, **fp32**, no autocast, no sequence parallelism, one sample, a short clip (3–4 chunks). Identical
weights and identical input latents for both paths.

```
Path A (training regime): one forward with the clean history passed as cond tokens
                          (cond_chunk_sizes set, training=True)
Path B (inference regime): roll the same clip chunk by chunk — encode each clean chunk into
                          the KV cache at timestep 0, then denoise the window (training=False)

Assert: for the SAME window at the SAME timesteps, A's output == B's output
        within fp32 tolerance (here: max abs diff ~1e-5; anything above 1e-3 is a real bug)
```

Run it for each configuration that changes visibility:

- full-history blocks only (`full_history_block_num = depth`)
- sink-recent blocks only (`full_history_block_num = 0`)
- the mixed production split
- with and without context noise (P5 must not break equivalence: the same noised history must be fed to both
  paths, so inject a fixed noise tensor rather than drawing it)
- after a KV recompute (roll past the threshold, then compare the next window)
- the first window (prefill, partial window) and a steady-state window
- the i2v layout, if the port does image-to-video: path A with the 1-frame first chunk as the clean prefix,
  path B with that frame encoded into the cache before the first round

**When G2 fails, look in this order.** The failure is almost never in the attention maths.

1. **RoPE.** Print the frequency rows each path uses for the same token and diff them. Compact vs absolute
   positions, or a slot table that starts at the wrong offset, is the most common cause.
2. **The visible KV set.** Print, for one query block, the indices of the tokens it attends to in each path.
   Diff the sets. An extra or missing cond chunk, or a causal prefix taken from the noisy tokens instead of the
   clean tail, shows up here.
3. **The trim order.** For sink-recent blocks, the trim must happen before the new chunk is encoded, so the
   cache holds `{sink}` during the encode and `{sink, recent}` after.
4. **Which forward writes the cache.** If the denoising forward also writes, path B's cache contains entries
   path A never had.
5. Only then dtype and kernel differences.

### G3 — Numerical equivalence against a reference implementation

For a unit ported alone — especially into a model nobody intends to retrain — this is the **primary**
correctness gate, because it pins the visibility rule and the positions without needing a second regime.

Where a reference exists *inside the same host*, compare against it in fp32:

- a ported kernel vs a naive masked `scaled_dot_product_attention` implementing the same visibility rule
  (build the mask explicitly, run both, diff). This catches packing bugs that G2 cannot see because both of
  G2's paths use the same packer.
- the per-frame scheduler step vs a loop over frames calling the host's own scalar `step`.
- the streaming VAE decode vs the whole-video decode (P11 claims to compute the same function; prove it).

### G4 — End to end with real weights

Only meaningful after the model has been trained or distilled in the new host. Judge by eye, plus:

- does the video hold together past the point where the KV cache first recomputes?
- does it drift, saturate, or freeze late in a long rollout? That is usually context noise (P5) or the
  recompute selection (P4), not the attention.
- a long run and a short run must agree over their common prefix when the seed and the topology match.

Do not expect a long video to match any other implementation frame by frame. Trajectories diverge over
dozens of chunks even between two correct implementations.

### G5 — Distributed consistency

- Every rank of a sequence-parallel group must see identical inputs and draw identical random numbers. Add a
  checksum assertion at the entry point while porting, and keep it behind a debug flag afterwards.
- Multi-GPU vs single-GPU: compare outputs at the same seed. They will not be bit-identical in bf16, but they
  must be visually the same and close numerically; a large gap means a sharding or a draw-order bug.
- For P10: assert the per-rank window count K is equal across the world **before** the first generator forward.
  This is an all-reduce, not a local check.
- For ACS: every rank of the group must condition on bit-identical SigLIP tokens, and the run must have a single
  FSDP shard group; refuse other topologies before the weights load.

### G6 — Training stability

A short run (tens of steps) must show finite loss and finite gradient norm on every rank, and the skip counter
for non-finite gradients must stay near zero. Add the "all ranks skip together" rule from day one — a NaN that
reaches one rank's EMA shadow is permanent.

## Which gate per unit

| Unit | In a full port | Ported alone |
|---|---|---|
| P1 streaming core | G0, G1, G2 | G0, G1; add G2 once a second regime exists |
| P2 FBCA attention | G0, G1, G2, G3 (vs an explicit mask) | G0b, G1, G3 |
| P3 HieraKV | G1, G2 (all three block splits) | G0b, G1, G3, G4 |
| P4 KV recompute | G1, G2 (across a recompute), G4 | G0b, G1, G4 |
| P5 context noise | G1, G2 (fixed noise), G4 | G0b, G1, G4 |
| P6 AnnealCFG | G1, G4 | G0b, G1, G4 |
| P7 rolling inference | G1 (step-index table, i2v layout), G2, G4, G5 | G1; G2 if the host trains, else G3; plus G4, G5 |
| P8 WinTF training | G1 (window grid, phase, supervised-frame mask), G2, G5, G6 | G1, G2, G5, G6 (unchanged — it *is* the training side) |
| P9 rolling ODE init | G1 (cache contents), G6 | not ported alone |
| P10 CuLT-DMD | G1 (the typed contracts), G5 (K uniform), G6 | not ported without P7; same gates |
| P10 ACS (i2v variant) | G1 (only windows with `score_start == 0` get the anchor condition), G5 (identical tokens, one shard group), G6 | not ported without P10; same gates |
| P11 PS-VAE | G1 (cache frame counts, row split), G3 (vs whole-video decode), G5 | G0b, G1, G3, G5 |

G1 is never droppable: it is cheap and it catches the off-by-one chunk that every other gate hides.

## Keep a record

For each module, record: which gates ran, on what configuration, the numbers they produced, and anything left
open. A port is a long job and the gates are the only durable evidence that an earlier module is still correct
after a later one changed something.
