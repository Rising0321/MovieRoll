# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""Offline precompute of the rolling ODE trajectories that Stage 2 trains on (`cache_only: true`).

Per prompt the frozen teacher runs a full rolling inference (50 steps, window 5, AnnealCFG 4 -> 1) and only
the states the student's few-step schedule needs are kept: the diagonal window at every round boundary,
prefill windows included, plus the clean endpoints. Prompts are enough, no ground-truth video; each
trajectory is written to `{trajectory_cache_dir}/{sample_idx:06d}.pt`.

The teacher is FSDP2-sharded, so its forward is a collective and every rank must call it the same number of
times: the multi-GPU branch runs a fixed number of lockstep iterations, and a cache hit skips only the
rank-local disk write, never the forward. Run it as a module from the repository root (a script path puts
`movieroll/scripts` on sys.path instead) or through `movieroll/scripts/launch/precompute_ode.sh`:

    torchrun --nproc_per_node 8 -m movieroll.scripts.precompute_ode \
        --config movieroll/configs/stage2_ode.yaml --sp_size 2 --max_samples 0
"""
import argparse
import csv
import math
import os
import time as _time
from pathlib import Path

import torch
import torch.distributed as dist

from movieroll.commons.constants import NUM_TRAIN_TIMESTEPS
from movieroll.distill import init_distributed, load_config
from movieroll.trainer.ode.trainer import (
    _SP_SOURCE_RANDOM_GROUP_STRIDE,
    ODERegressionMainTrainer,
)
from movieroll.trainer.ode.config import validate_stage2_config


def _load_prompt_source(path):
    """Return (prompts, seeds): a .csv supplies both columns, a .txt one prompt per line and seeds None."""
    if str(path).lower().endswith(".csv"):
        prompts, seeds = [], []
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = (row.get("prompt") or row.get("original_prompt") or "").strip()
                if not p:
                    continue
                prompts.append(p)
                s = row.get("seed")
                seeds.append(int(s) if s not in (None, "") else None)
        if all(s is None for s in seeds):
            seeds = None
        return prompts, seeds
    with open(path, encoding="utf-8") as f:
        prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
    return prompts, None


def _clear_rollout_state(pipeline) -> None:
    """Clear partial queue/KV state after either success or failure."""

    queue = getattr(pipeline, "rolling_queue", None)
    if queue is not None:
        queue.clear()
    manager = getattr(pipeline.transformer, "kv_cache_manager", None)
    if manager is not None and manager.is_init:
        manager.clear_kv_cache()


def _check_job_shard_args(shard_id, num_shards, device) -> None:
    """Every rank of one torchrun job must carry the same --shard_id / --num_shards (collective).

    The flags index whole jobs; disagreeing ranks leave holes in the prompt split, and compute other
    prompts twice, with no error.
    """
    values = torch.tensor([int(shard_id), int(num_shards)], device=device, dtype=torch.long)
    low, high = values.clone(), values.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    if not torch.equal(low, high):
        raise ValueError(
            "under torchrun --shard_id / --num_shards index whole torchrun jobs and must be the same on every "
            f"rank of a job; got shard_id {int(low[0])}..{int(high[0])}, num_shards {int(low[1])}..{int(high[1])}. "
            "Launch one torchrun job per node with --shard_id <node> --num_shards <nodes>, or a single "
            "multi-node job with the defaults 0 / 1."
        )


def main():
    parser = argparse.ArgumentParser(description="Offline Rolling ODE trajectory precompute (Stage2)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="max prompts per rank; 0 = one pass over the whole prompt list")
    # The prompt split, per topology: a plain `python` launch (world_size=1, no FSDP, teacher replicated)
    # runs one process per GPU and takes the prompts with idx % num_shards == shard_id; under torchrun
    # (FSDP + SP lockstep) num_shards counts whole jobs, one per node, each spreading its share over its
    # own dp groups -- never GPUs or SP groups.
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="a prompt file, one per line (or a CSV); overrides config.data.prompt_path")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="plain python launch: total single-GPU processes; torchrun launch: total "
                             "independent torchrun jobs (nodes), not GPUs or SP groups")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="in [0, num_shards): this process's index (plain python launch) or this "
                             "torchrun job's ordinal, the same on all of its ranks (torchrun launch)")
    parser.add_argument("--cache_dir", type=str, default=None, help="override config.ode.trajectory_cache_dir")
    parser.add_argument("--max_per_shard", type=int, default=0, help="> 0 caps the prompts per shard")
    parser.add_argument("--sp_size", type=int, default=1,
                        help="> 1 runs the teacher sequence-parallel inside a torchrun job: the ranks of one SP "
                             "group share a prompt in lockstep; must divide the job's world size")
    parser.add_argument("--decode_video", type=int, default=1,
                        help="1 also VAE-decodes clean_video into <cache_dir>/videos/ as a visual reference")
    args = parser.parse_args()

    config = load_config(args.config, "stage2")
    config.ode.cache_only = False   # precompute generates online
    if args.cache_dir:
        config.ode.trajectory_cache_dir = args.cache_dir
    prompt_file = args.prompt_file or config.data.get("prompt_path")
    if not prompt_file:
        raise ValueError(
            "Stage2 precompute requires --prompt_file or config.data.prompt_path"
        )
    args.prompt_file = str(prompt_file)
    sp_size = max(1, int(args.sp_size))
    # Precompute reads this source directly; it builds no training dataset and no optimizer.
    config.data.prompt_path = args.prompt_file
    config.precompute_skip_student = True
    config.sp_size = sp_size
    # RANK / WORLD_SIZE are left alone: a plain ``python`` launch already has world_size=1, and erasing a
    # torchrun environment would make every worker rank 0, contending for cuda:0 and the same output files.

    validate_stage2_config(
        config,
        mode="precompute",
        prompt_file=args.prompt_file,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        sp_size=sp_size,
    )
    rank, world_size, device = init_distributed()
    if device.type != "cuda":
        raise RuntimeError("Stage2 precompute requires CUDA (teacher + flash-attn)")
    if world_size > 1:
        # Before the teacher is loaded: a mismatch would otherwise only show as missing cache entries.
        _check_job_shard_args(args.shard_id, args.num_shards, device)

    cache_dir = config.ode.get("trajectory_cache_dir", None)
    if not cache_dir:
        raise ValueError("config.ode.trajectory_cache_dir must be set (the precompute output directory)")

    # One resolution for every prompt.
    resolution = [int(v) for v in config.video.resolution]
    height, width = resolution[0], resolution[1]
    num_frames = int(config.video.get("num_frames", 125))                 # pixel frames
    teacher_steps = int(config.ode.get("teacher_num_inference_steps", 50))
    denoising_step_list = list(config.ode.denoising_step_list)
    window_size = int(config.streaming_config.streaming_window_size)

    tag = f"shard{args.shard_id}/{args.num_shards} sp{sp_size} r{rank}"
    if rank == 0:
        print("=" * 70)
        print(f"Rolling ODE Trajectory Precompute (Stage2) [{tag}]")
        print(f"  teacher_ckpt   = {config.get('teacher_ckpt')}")
        print(f"  teacher_steps  = {teacher_steps} (window={window_size}, "
              f"steps_per_window={teacher_steps // window_size})")
        print(f"  denoising_step = {denoising_step_list} ({len(denoising_step_list)} raw steps -> warped)")
        print(f"  resolution     = {height}x{width}, num_frames = {num_frames}")
        print(f"  cache_dir      = {cache_dir}, world_size = {world_size}, "
              f"sp_size = {sp_size}, decode_video = {bool(args.decode_video)}")
        print("=" * 70)

    trainer = ODERegressionMainTrainer(config=config, rank=rank, world_size=world_size, device=device)
    pipeline = trainer.generator_pipeline
    pipeline.transformer = trainer.teacher   # the TEACHER produces the trajectory, not the student
    # The teacher guides with the streaming config's AnnealCFG (cfg_max -> cfg_min).
    cfg_min, cfg_max = pipeline.streaming_config.cfg_min, pipeline.streaming_config.cfg_max
    if rank == 0:
        print(f"  anneal_cfg     = {cfg_max} -> {cfg_min}")

    from movieroll.commons.parallel_states import nccl_info as _nccl_info
    _sp = max(1, int(getattr(_nccl_info, "sp_size", 1)))
    is_leader = int(getattr(_nccl_info, "rank_within_spgroup", 0)) == 0
    video_dir = os.path.join(cache_dir, "videos")
    if is_leader and args.decode_video:
        os.makedirs(video_dir, exist_ok=True)

    # A scheduler of its own for the base __call__: its set_timesteps(teacher_steps) must not rewrite the
    # trainer's scheduler, which is shared and set to the student ladder. Same shift as the inference chain.
    from movieroll.schedulers.scheduling_flow_match_streaming import FlowMatchStreamingScheduler
    pipeline.scheduler = FlowMatchStreamingScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        shift=float(config.ode.get("timestep_shift", 5.0)),
    )
    if rank == 0:
        print(f"[precompute] scheduler -> FlowMatchStreamingScheduler(shift="
              f"{config.ode.get('timestep_shift', 5.0)}), sp={_sp}", flush=True)

    from movieroll.models.transformers.modules.chunking import check_and_get_chunk_sizes as _ck_chunks
    from movieroll.utils.file_utils import save_videos_grid as _save_videos_grid
    num_student = len(denoising_step_list)
    video_fps = int(config.get("video_fps", 24))

    def _run_one(idx, prompt, seed=None, force_forward=False, do_save=True):
        """One prompt through the base __call__, with the ODE states captured by the subclass hooks.

        output_type='pt' makes __call__ normalise its video to [0,1], so the RAW clean latent (the ODE
        target) comes from the _pop_and_output_clean_chunk hook and the noisy diagonal rounds from the
        _run_inner_denoising_loop hook. seed falls back to 1234 + idx. force_forward=True runs even on a
        cache hit (FSDP+SP lockstep); do_save gates the disk write to the leader rank.
        """
        cache_path = os.path.join(cache_dir, f"{idx:06d}.pt")
        video_path = os.path.join(video_dir, f"{idx:06d}.mp4")
        cached = Path(cache_path).is_file()
        cache_complete = cached and (
            not bool(args.decode_video) or Path(video_path).is_file()
        )
        if cache_complete and not force_forward:
            return "skip"
        _seed = int(seed) if seed is not None else (1234 + idx)
        # Only SP rank 0 holds the source RNG; the group stays in step because the pipeline broadcasts
        # the whole noise tensor.
        group_id = (
            int(dist.get_process_group_ranks(_nccl_info.sp_group)[0])
            if _sp > 1
            else rank
        )
        gen = (
            torch.Generator(device=device).manual_seed(
                _seed + group_id * _SP_SOURCE_RANDOM_GROUP_STRIDE
            )
            if is_leader
            else None
        )
        try:
            pipeline._ode_capture = True
            pipeline._ode_rounds = []
            pipeline._ode_clean_chunks = []
            with torch.no_grad():
                out = pipeline(
                    prompt=prompt, height=height, width=width, video_length=num_frames,
                    num_inference_steps=teacher_steps, use_cfg=True,
                    negative_prompt="", generator=gen,
                    output_type="pt" if args.decode_video else "latent",
                    event_prompts=None,
                )
            pipeline._ode_capture = False
            video = out.videos if hasattr(out, "videos") else out[0]   # [B, 3, T_pixel, H, W] in [0,1]
            rounds = list(pipeline._ode_rounds)
            clean_chunks = list(pipeline._ode_clean_chunks)
            clean_video = torch.cat(clean_chunks, dim=2)[0] if clean_chunks else None  # [C, total, h, w] raw latent
            if clean_video is None or not rounds:
                raise RuntimeError(
                    "ODE capture produced no clean video or no rolling rounds"
                )
            if args.decode_video and video is None:
                raise RuntimeError("ODE precompute requested video output but decode returned None")
            latent_len = (num_frames - 1) // 4 + 1
            tt = latent_len // pipeline.transformer.patch_size[0]
            csp = _ck_chunks(tt, chunk_size=pipeline.streaming_config.chunk_size, mode="default")
            chunk_sizes = [c * pipeline.transformer.patch_size[0] for c in csp]
            ts = pipeline.scheduler.timesteps
            stride = max(1, teacher_steps // num_student)
            warped_timesteps = [float(ts[i * stride].item()) for i in range(num_student) if i * stride < len(ts)]

            if not (do_save and is_leader) or cache_complete:
                return "skip" if cache_complete else "saved"   # non-leader ranks only join the forward
            cache_tmp = f"{cache_path}.tmp.{os.getpid()}"
            video_tmp = f"{video_path}.tmp.{os.getpid()}.mp4"
            try:
                torch.save({
                    "prompt": prompt, "seed": _seed, "clean_video": clean_video, "rounds": rounds,
                    "chunk_sizes": chunk_sizes, "warped_timesteps": warped_timesteps,
                    "height": height, "width": width,
                }, cache_tmp)
                # Every requested artifact is finished before the cache path is published: otherwise a
                # decode/save failure leaves a cache hit whose reference video stays missing on retries.
                if args.decode_video and video is not None:
                    vid = video if video.dim() == 5 else video.unsqueeze(0)   # [1,3,T,H,W]
                    _save_videos_grid(vid, video_tmp,
                                      rescale=False, n_rows=1, fps=video_fps)
                    os.replace(video_tmp, video_path)
                os.replace(cache_tmp, cache_path)
            except Exception:
                Path(cache_tmp).unlink(missing_ok=True)
                Path(video_tmp).unlink(missing_ok=True)
                # A video published before the cache commit is not a valid entry.
                if not Path(cache_path).exists():
                    Path(video_path).unlink(missing_ok=True)
                raise
            return "saved"
        except Exception as e:
            import traceback
            print(f"[precompute][{tag}] idx={idx} FAILED: {e}\n{traceback.format_exc()}")
            return "err"
        finally:
            pipeline._ode_capture = False
            _clear_rollout_state(pipeline)

    prompts, seeds = _load_prompt_source(args.prompt_file)
    if not prompts:
        raise ValueError(f"precompute prompt source has no usable rows: {args.prompt_file}")
    if rank == 0:
        print(f"[precompute] loaded {len(prompts)} prompts, "
              f"per-prompt seeds={'yes' if seeds else 'no(1234+idx)'}", flush=True)

    max_per_shard = (
        int(args.max_per_shard)
        if int(args.max_per_shard) > 0
        else int(args.max_samples)
    )
    if world_size == 1:
        # Single-GPU replicate (no dist, no lockstep): take the shard directly, a cache hit skips the
        # forward as well.
        my_idxs = list(range(args.shard_id, len(prompts), args.num_shards))
        if max_per_shard > 0:
            my_idxs = my_idxs[:max_per_shard]
        print(f"[precompute][{tag}] total_prompts={len(prompts)} my_prompts={len(my_idxs)}")
        n_saved = n_skip = n_err = 0
        t0 = _time.time()
        for k, idx in enumerate(my_idxs):
            r = _run_one(idx, prompts[idx], seed=(seeds[idx] if seeds else None), force_forward=False, do_save=True)
            n_saved += (r == "saved")
            n_skip += (r == "skip")
            n_err += (r == "err")
            if k % 2 == 0 or k == len(my_idxs) - 1:
                print(f"[precompute][{tag}] {k+1}/{len(my_idxs)} saved={n_saved} skip={n_skip} "
                      f"err={n_err} elapsed={_time.time()-t0:.0f}s", flush=True)
        print(f"[precompute][{tag}] FINISHED saved={n_saved} skip={n_skip} err={n_err}")
        if n_err:
            raise RuntimeError(
                f"precompute failed for {n_err} sample(s) in {tag}"
            )
        return

    # FSDP + SP lockstep: the ranks of an SP group share a prompt, different dp groups take different
    # prompts, and every rank runs the same number of forwards. SP is the innermost (contiguous) mesh
    # dimension, so the global group is node_ord * num_local_groups + rank // _sp. An FSDP shard spans a
    # node's SP groups, so the whole node stays in lockstep for ceil(N / num_global_groups) steps: the
    # forward always runs, and only a leader rank with a real index and no cache entry writes.
    num_local_groups = int(getattr(trainer, "dp_degree", 1) or 1)
    node_ord = int(args.shard_id)              # --shard_id: this job's ordinal (one job per node; 0 for one job)
    num_nodes = max(1, int(args.num_shards))   # --num_shards: the number of jobs (1 for one job)
    num_global_groups = num_nodes * num_local_groups
    my_group = node_ord * num_local_groups + int(trainer.dp_rank or 0)
    my_idxs = list(range(my_group, len(prompts), num_global_groups))
    max_group_len = math.ceil(len(prompts) / num_global_groups)
    num_steps = min(max_per_shard, max_group_len) if max_per_shard > 0 else max_group_len
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(f"[precompute] FSDP+SP: world={world_size} sp={_sp} node={node_ord}/{num_nodes} "
              f"local_groups={num_local_groups} global_groups={num_global_groups} "
              f"num_steps={num_steps} (about {len(my_idxs)} prompts for this group)", flush=True)
    n_saved = n_skip = n_err = 0
    t0 = _time.time()
    for step in range(num_steps):
        real = step < len(my_idxs)
        # padding: repeat the last index (always forward, never save)
        idx = my_idxs[step] if real else (my_idxs[-1] if my_idxs else 0)
        r = _run_one(idx, prompts[idx], seed=(seeds[idx] if seeds else None), force_forward=True, do_save=real)
        if real:
            n_saved += (r == "saved")
            n_skip += (r == "skip")
        n_err += (r == "err")
        step_errors = torch.tensor(
            [int(r == "err")], device=device, dtype=torch.long
        )
        dist.all_reduce(step_errors, op=dist.ReduceOp.SUM)
        if int(step_errors.item()) > 0:
            raise RuntimeError(
                f"precompute failed on {int(step_errors.item())} rank(s) "
                f"at lockstep iteration {step}"
            )
        if rank == 0 and (step % 2 == 0 or step == num_steps - 1):
            el = _time.time() - t0
            print(f"[precompute] step {step+1}/{num_steps} (leader group0) saved={n_saved} skip={n_skip} "
                  f"err={n_err} elapsed={el:.0f}s ~{el/max(1,step+1):.0f}s/step", flush=True)
    print(f"[precompute][{tag}] FINISHED saved={n_saved} skip={n_skip} err={n_err}")


if __name__ == "__main__":
    main()
