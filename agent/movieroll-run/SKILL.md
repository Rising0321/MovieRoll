---
name: movieroll-run
description: Run MovieRoll in this repository — inference (the base 50-step chain or the distilled few-step chain, text-to-video or image-to-video, single or multi-event) and any one of the four training stages (streaming SFT single-event and multi-event, ODE initialization, CuLT-DMD distillation, including the acs image-to-video recipe). Turns a request like "generate a 30-second clip", "fine-tune on my own videos", "resume the distillation", "why was my run rejected" into the right config fields, the right launch command and the right pre-flight checks, using this repository's actual entry points. Use it whenever someone wants to generate a video with MovieRoll, start or resume a training stage, pick a frame count or a parallel topology, prepare SFT latents or distillation prompt CSVs, turn a training checkpoint into a model folder, or diagnose a run that was refused, hung or produced a shorter video than asked for.
---

# Running MovieRoll

A Chinese version of this document is `SKILL_CN.md`; the `references/` files are English only.

## What this repository contains

| Path | What it is |
|---|---|
| `generate_movieroll.py` → `movieroll/generate.py` | inference, both chains |
| `train_movieroll.py` → `movieroll/train.py` | streaming SFT, stages ① ② |
| `distill_movieroll.py` → `movieroll/distill.py` | `--mode stage2` = ③ ODE, `--mode stage3` = ④ CuLT-DMD |
| `movieroll/scripts/launch/*.sh` | the launcher for each training stage |
| `movieroll/configs/` | `sft/`, `stage2_ode.yaml`, `stage3_dmd_rolling{,_acs}.yaml`, `models/` |
| `movieroll/scripts/encode_latents.py` | raw clips → the pre-encoded latents SFT trains on |

Read `references/inference.md` before an inference task and `references/training.md` before a training task.
They hold the field-by-field contracts, the invariants and the failure tables; this file is the procedure.

## Step 1 — classify the request

Ask yourself which of these it is, and say so before acting:

- **Generate a video** → inference. Which chain? `distilled` unless the user wants best quality or 50-step
  behaviour, then `base`. Text-to-video or image-to-video? One prompt or several events?
- **Train** → which stage? "fine-tune on my videos" is ① (and ② if they have multi-event captions);
  "make it fast / few-step" is ③ then ④; "i2v distillation" is ④ with `score_mode: acs`.
- **Prepare data** → `encode_latents` for SFT, or a prompt CSV for ③ ④.
- **Package a result** → a training checkpoint into a `--model` folder.
- **Diagnose** → find the message in the failure table of the relevant reference first; most refusals are
  arithmetic and the error text already names the fix.

If the request is ambiguous in a way that changes the command (chain, stage, GPU count, frame count), ask.
Do not guess a training stage — the stages consume each other's checkpoints and running the wrong one wastes
hours.

## Step 2 — do the arithmetic before touching anything

Most failed runs are rejected by a validator for one of these. Work them out on paper first and state the
numbers in your reply.

```
chunk_size C = 4,  window_size W = 5          (the released geometry)

latent_length = (video_length - 1) // 4 + 1
required:  latent_length % C == 1   ⇒   video_length = 16n + 1
valid:     17, 81, 241 (10 s), 721 (30 s), 1441 (60 s)   at 24 fps
distilled and every i2v run also need  latent_length >= 1 + C   ⇒  >= 17 frames

height, width: positive multiples of 16
sp_size: divides 16 (the attention heads) ⇒ 1, 2, 4, 8, 16;  world_size % sp_size == 0
inference with FSDP (--shard-world):  world_size % shard_world == 0  and  shard_world % sp_size == 0
           (--shard-world defaults to 8: below 8 GPUs lower it, or --fsdp-sharding-strategy none)
training, hybrid sharding (shard group = fsdp_sub_world_size ①② / fsdp.shard_world_size ③④):
           replicas = world_size // shard_group >= 2  and  world_size % (replicas * sp_size) == 0
           (a floor division, so the shard group need not divide world_size;
            a shard group equal to world_size runs as full sharding)
training stage ④: at least 2 ranks; score_mode acs needs a single shard group
training stage ④: K = (latent_frames - 1) // C // W     (3 in phase 1, 9 in phase 2)
```

Then check the paths exist — `ckpts/vae`, the text encoders, the glyph JSONs, the checkpoint you are starting
from, the data root. Every entry point validates these before loading weights, but finding them yourself is
faster than a round trip.

## Step 3 — run it

**Inference.** Build the command from `references/inference.md`. Multi-GPU uses `torchrun`; one GPU uses plain
`python` with `--sp-size 1`:

```bash
torchrun --nproc_per_node=8 generate_movieroll.py \
  --model ckpts/movieroll/distilled/t2v --sp-size 8 \
  --prompt "..." --output-dir ./outputs
```

Image-to-video takes the image-to-video model folder and the first frame (`--input-image`, or an `input_image`
column in a CSV run); see "Image-to-video" in `references/inference.md`:

```bash
torchrun --nproc_per_node=8 generate_movieroll.py \
  --model ckpts/movieroll/distilled/i2v --sp-size 8 --input-image first_frame.jpg \
  --prompt "..." --output-dir ./outputs
```

**Training.** Edit the stage's YAML, then run its script. Settings belong in the YAML, not on the command line:

```bash
CONFIG=movieroll/configs/sft/stage1_single_event.yaml bash movieroll/scripts/launch/train_sft.sh
bash movieroll/scripts/launch/precompute_ode.sh    # ③ step 1
bash movieroll/scripts/launch/train_ode.sh         # ③ step 2
bash movieroll/scripts/launch/train_dmd.sh         # ④
CONFIG=movieroll/configs/stage3_dmd_rolling_acs.yaml bash movieroll/scripts/launch/train_dmd.sh   # ④ image-to-video (ACS)
```

Before a long training run, do a cheap dry pass: stage ① supports `dry_run: true` with
`max_train_steps: 2`, which exercises the data pipeline and the topology without training. For ③ ④, a config
error surfaces in seconds because validation precedes NCCL init — so just starting and reading the first
screen of output is itself the check.

## Step 4 — report honestly

State what actually happened: the command run, where the outputs landed (`references/inference.md` has the
path layout), and for training the step reached and the checkpoint written. If a run was refused, quote the
message. If you only validated the config and did not train, say that. Never describe a video as generated
without having seen the file appear.

## Hard rules

1. **The sampling parameters in `movieroll/configs/models/*/config.json` and `movieroll/commons/profiles.py` are not
   tunable defaults.** They are the values the released checkpoints were evaluated with. Change them only when
   the user asks, and say what you changed.
2. **Nothing may depend on the rank.** Every shape-affecting draw is made on sp rank 0 and broadcast. Adding a
   rank-dependent branch or an extra random draw on one rank does not error — it desynchronises the
   collectives and hangs. This applies to any code you add to training or to a pipeline.
3. **Geometry changes invalidate artefacts.** Changing `chunk_size`, `window_size`, the frame count or the
   resolution means re-encoding SFT latents and re-running the ODE precompute (into a fresh cache directory).
4. **Resume needs the identical topology** for ③ and ④: same GPU count, same `sp_size`, same FSDP layout. To
   change the topology, restart from the exported `safetensors` instead of resuming. A multi-node ④ run also
   needs its `logdir` on storage every node can read: between the two phases every rank loads the checkpoint
   back.
5. **Keep context noise, KV recompute and AnnealCFG on.** They are what make long videos hold together; turning
   them off to "simplify" a run changes the output quality, and KV recompute off will eventually exceed the RoPE
   grid.
6. **Do not invent flags or fields.** Every validator rejects unknown keys by name. If something seems missing,
   read the parser or the config dataclass rather than guessing a name.
7. This repository ships no tests. If the user asks for a test run, the checks available are the validators,
   `dry_run`, and a short real run.

## Working loop

```
- [ ] 1. Classify: inference or which training stage; state it
- [ ] 2. Read the matching reference file
- [ ] 3. Do the arithmetic (frames, geometry, topology) and check the paths
- [ ] 4. Show the user the command or the config diff before a long run
- [ ] 5. Launch; read the first output for a validator refusal
- [ ] 6. Report the command, the artefacts and the real outcome
```
