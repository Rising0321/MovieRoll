---
name: movieroll-port
description: Port MovieRoll's streaming long-video logic out of this repository into another code base whose structure is not known in advance — a different DiT, a different trainer, a different VAE. Holds the eleven portable units (streaming core, FBCA attention, HieraKV, KV recompute, context noise, AnnealCFG, rolling inference, WinTF training, rolling ODE init, CuLT-DMD, PS-VAE) with their invariants, a survey procedure for reading an unfamiliar target and mapping the units onto it, and a verification regime built around training/inference equivalence rather than comparing outputs with this repository. Use it whenever someone wants to move MovieRoll's rolling-window generation, bounded KV caching, context noise, image-to-video or few-step distillation into another model or repository, asks which parts are worth taking or what the minimum subset is, or needs to decide how deep a port should go.
---

# Porting MovieRoll into another code base

A Chinese version of this document is `SKILL_CN.md`; the `references/` files are English only.

This repository is the **source**, and it is read-only for the purposes of a port. The target can be anything:
another DiT, another framework, another training stack. Because the target is unknown, the work starts by
reading it, not by copying files.

| File | What it holds |
|---|---|
| `references/modules.md` | the eleven portable units: what each does, where it lives here, what the host must provide, the invariants, and — in "Porting a single unit" — the capability-vs-semantics table and the scenarios for taking just one or two; image-to-video is covered under P7, P8 and the ACS variant of P10 |
| `references/survey.md` | the questions to answer about an unfamiliar target, how to find the answers, and what each answer changes |
| `references/verification.md` | the gates, built around training/inference equivalence |

## The one idea to carry across

Everything else is detail around this:

> The model is **one** function used in two regimes. In training the clean history arrives as input tokens; at
> inference it lives in a KV cache. Every token must see the same set of KV entries and the same RoPE relative
> distances either way.

Break it and nothing raises. The shapes fit, the loss falls, the samples look fine, and the model has learned a
distribution that deployment does not reproduce. This is why `references/verification.md` makes equivalence the
primary gate and treats "compare with MovieRoll's output" as meaningless — the target has different weights, so
there is no reference video to compare against.

The exception worth knowing up front: an inference-only port cannot run that gate, because it has only one
regime. There, degenerate-configuration equivalence and an explicit-mask reference take its place.

## Procedure

### 1. Survey the target first

Work through `references/survey.md` read-only. Do not write into the target until the plan is agreed, and never
write into this repository. Record the answers as `question → answer → consequence` and show them to the user.

Start with that file's **Question 0 — will the model be retrained?** More follows from it than from anything
else: an inference-only port puts the training-side units out of reach, makes the G2 equivalence gate
impossible to run, and caps what P7 can achieve on its own.

The survey classifies the host, and the class decides the size of the job:

- **cross-attention DiT** — port only the image-only attention kernels; the "history is text-independent"
  invariant holds for free
- **MMDiT** — like this repository; port both kernel families
- **already block-causal with a KV cache** — the cheap, high-value subset is context noise + HieraKV +
  recompute + AnnealCFG, with no change to the training recipe
- **bidirectional, non-causal** — the full stack, and expect the KV-cache split and the per-frame scheduler to
  be the two hardest changes

### 2. Agree the subset with the user

**Most units port on their own.** Do not port everything by default, and do not widen the scope because the
dependency graph draws an arrow — that graph shows what reinforces what, not what must travel together. The
table and the scenarios in `references/modules.md` are what to use here.

When the user **names a unit or two**, follow the four steps at the end of that file's "Porting a single unit"
section:

1. read the unit's **Standalone?** and **Add another unit only if** lines;
2. check each required *capability* against your survey answers;
3. propose an extra unit only for a capability the host genuinely lacks, and say what it costs;
4. where the dependency is *semantic* rather than a capability — P7 without P8 being the case to watch — say so
   plainly and let the user decide, instead of quietly adding a training-side job to an inference-side request.

When the user describes a **goal** instead, use the scenarios table: an unbounded cache is P3 (+P4), drift is
P5 alone, mediocre multi-step sampling is P6, decode cost is P11, "is this worth it" is the P2 masked fallback
plus P7 as a measurement, "I already distil; make long rollouts hold up" is P10 started from the host's own
few-step model (it still needs P7; this repository has no block-wise CuLT-DMD). Image-to-video is not a unit of
its own: it is P7's i2v layout, trained by P8's i2v draw, plus ACS under P10 only when an i2v model is distilled.

Record the selection and the reasoning. A one-unit port is a legitimate, finished piece of work — it does not
have to grow into the full stack.

### 3. Port one unit at a time, in dependency order

For each unit: read its section in `references/modules.md`, find the host's insertion points from the survey,
write the code in the host's own idiom, keep it behind a flag (G0), then run that unit's gates before starting
the next one. A unit that has not passed its gates is not done, and stacking a second unit on top of an
unverified first one makes the failure impossible to localise.

With a single-unit port the order question disappears, but the gates do not: run that unit's row from the gate
table in `references/verification.md`, which lists a reduced set for units ported alone.

Write new code in its own directory or module inside the target rather than scattering edits, so the port stays
reviewable and reversible.

### 4. Verify

Follow `references/verification.md`. Build the G2 equivalence harness early — before the units that depend on
it — because it is the only check that catches the failure mode that matters.

### 5. Record

Keep a decisions log and a progress table outside this repository: in a scratch directory of your own until the
plan is agreed, then in the target next to the ported code if the user wants it there. Record which units are
selected, which gates passed with what numbers, what is still open, and every assumption you had to make because
the survey could not answer a question. An unrecorded assumption becomes a silent numerical bug several units
later.

## Hard rules

1. **Never write into this repository** while porting out of it.
2. **This repository's code is the source of truth.** Where `references/modules.md` and the code disagree, the
   code is right and the document should be corrected.
3. **Adapt, do not transplant.** Reuse the host's trainer, scheduler, VAE, text encoder and parallelism. Porting
   MovieRoll's framework alongside its logic produces two half-working stacks.
4. **Nothing may depend on the rank.** Every shape-affecting draw happens on sequence-parallel rank 0 and is
   broadcast. A rank-dependent branch or an extra draw on one rank does not error; it desynchronises the
   collectives and hangs.
5. **One unit, one gate run.** No stacking on unverified work.
6. **Stop and ask** when the survey cannot answer a question that changes the design. Do not pick a convention
   silently — record the question, propose a default, and get it confirmed.
7. **Do not widen the scope on your own.** If the user asked for one unit, deliver that unit. Extra units are
   proposed, with their cost, only for capabilities the host genuinely lacks — never because the dependency
   graph has an arrow, and never to make the port "complete".
8. **Do not refactor unrelated parts of the target**, and do not commit or push there unless asked.

## Working loop

```
- [ ] 1. Survey the target (references/survey.md), record answers, classify the host
- [ ] 2. Agree the unit subset: named units -> check capabilities, add nothing extra without saying why;
         a stated goal -> the scenarios table
- [ ] 3. Build the G2 equivalence harness if the subset includes a unit that needs it
- [ ] 4. For each unit in dependency order: read its section, insert behind a flag, run its gates
- [ ] 5. Record gates, numbers, open questions and assumptions
```
