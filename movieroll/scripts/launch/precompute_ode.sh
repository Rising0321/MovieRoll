#!/bin/bash
# Stage-2 ODE trajectory precompute.
#
#   bash movieroll/scripts/launch/precompute_ode.sh [extra args forwarded to precompute_ode.py]
#
# Two topologies, both covered by the same script, and SHARD_ID / NUM_SHARDS mean different things in each:
#   NPROC=1  -> one process per GPU, the teacher replicated, no collectives. Launch it once per GPU with a
#               different CUDA_VISIBLE_DEVICES (the process always uses cuda:0); NUM_SHARDS = the number of
#               such processes across the cluster, SHARD_ID = this one's index in [0, NUM_SHARDS). This is
#               the fast way across a cluster.
#   NPROC>1  -> one torchrun job per node: FSDP-sharded teacher plus sequence parallelism, the ranks of an SP
#               group in lockstep on one prompt. NUM_SHARDS = the number of nodes (jobs), SHARD_ID = this
#               node's ordinal -- not GPU or SP-group counts: each job spreads its share over its own dp
#               groups. Every node must use the same NPROC and SP_SIZE, or the split leaves holes.
#
# The cache directory (the effective one: CACHE_DIR, a forwarded --cache_dir, or the config's) is created
# here rather than in the script: with --decode_video 0 nothing else would create it and the first torch.save
# would fail.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO"

CONFIG=${CONFIG:-movieroll/configs/stage2_ode.yaml}
NPROC=${NPROC:-8}
# SP_SIZE defaults to 2, which has to divide the
# job's world size (and fsdp.shard_world_size): a one-process shard has no SP group to split over, so NPROC=1
# (and an odd NPROC) falls back to 1. The SP size is part of the per-prompt seed (CSV seed + the job rank of
# the SP group's first rank * 1000003), so caches built with different NPROC / SP_SIZE are not bit-identical.
if [ -z "${SP_SIZE:-}" ]; then
    if [ "$NPROC" -gt 1 ] && [ $((NPROC % 2)) -eq 0 ]; then
        SP_SIZE=2
    else
        SP_SIZE=1
    fi
fi
SHARD_ID=${SHARD_ID:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
MAX_SAMPLES=${MAX_SAMPLES:-0}
DECODE_VIDEO=${DECODE_VIDEO:-1}
CACHE_DIR=${CACHE_DIR:-}

# torchrun puts the SCRIPT directory on sys.path, not the cwd, so the repo has to come in via
# PYTHONPATH (an environment setting, not a sys.path edit inside the package).
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(--config "$CONFIG" --sp_size "$SP_SIZE" --shard_id "$SHARD_ID"
      --num_shards "$NUM_SHARDS" --max_samples "$MAX_SAMPLES" --decode_video "$DECODE_VIDEO")
if [ -n "$CACHE_DIR" ]; then
    ARGS+=(--cache_dir "$CACHE_DIR")
fi

# argparse keeps the LAST occurrence of a flag and the forwarded arguments come after ARGS, so a --config or
# --cache_dir among them is the one the script will use.
EFFECTIVE_CONFIG=$CONFIG
FORWARDED=("$@")
for ((i = 0; i < ${#FORWARDED[@]}; i++)); do
    case "${FORWARDED[$i]}" in
        --config) EFFECTIVE_CONFIG=${FORWARDED[$((i + 1))]:-}; i=$((i + 1)) ;;
        --config=*) EFFECTIVE_CONFIG=${FORWARDED[$i]#--config=} ;;
        --cache_dir) CACHE_DIR=${FORWARDED[$((i + 1))]:-}; i=$((i + 1)) ;;
        --cache_dir=*) CACHE_DIR=${FORWARDED[$i]#--cache_dir=} ;;
    esac
done
if [ -z "$CACHE_DIR" ]; then
    # Usually the cache directory comes from the config -- read it from there, because the directory still has
    # to exist before the first torch.save and the script only creates the `videos/` subdirectory, and only
    # when --decode_video is on.
    CACHE_DIR=$(python -c "
import sys
from omegaconf import OmegaConf
print(OmegaConf.load(sys.argv[1]).ode.trajectory_cache_dir)
" "$EFFECTIVE_CONFIG")
fi
mkdir -p "$CACHE_DIR/videos"

echo "[precompute] config=$EFFECTIVE_CONFIG nproc=$NPROC sp=$SP_SIZE shard=$SHARD_ID/$NUM_SHARDS cache=$CACHE_DIR"
if [ "$NPROC" -le 1 ]; then
    exec python movieroll/scripts/precompute_ode.py "${ARGS[@]}" "$@"
else
    exec torchrun --nproc_per_node "$NPROC" --master_port "${MASTER_PORT:-29580}" \
        movieroll/scripts/precompute_ode.py "${ARGS[@]}" "$@"
fi
