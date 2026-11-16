#!/bin/bash
# Stage-3 rolling DMD distillation.
#
#   bash movieroll/scripts/launch/train_dmd.sh [extra args forwarded to distill_movieroll.py]
#
# Production schedule (movieroll/configs/stage3_dmd_rolling.yaml):
#   phase 1  short_single_event   600 steps, 241 pixel / 61 latent frames,  1 event,  K=3
#   phase 2  long_dual_event      600 steps, 721 pixel / 181 latent frames, 2 events, K=9
# with dfake_gen_update_ratio=3, i.e. 400 generator updates and 1200 fake_score updates in total.
#
# To resume, set `resume_checkpoint` in the config to the `step-<N>` directory ITSELF, not to its parent
# `checkpoints/`: that is where checkpoint_metadata.json lives, and the trainer derives the experiment directory
# two levels above it.
#
# The launcher log is <logdir>/train_dmd_node<NODE_RANK>_<timestamp>.log, where <logdir> is LOG_DIR if
# set, else the last `--override logdir=...` among the forwarded arguments, else the logdir of the config the
# trainer reads (a forwarded `--config` wins over CONFIG, as it does in distill.py's argparse).
#
# PYTORCH_CUDA_ALLOC_CONF defaults to expandable_segments:True (less fragmentation over a long run); an
# exported value is kept. Other cluster-specific tuning (NCCL_SOCKET_IFNAME, IB settings) is
# deliberately NOT set here: export it in the environment the launcher runs in.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO"

CONFIG=${CONFIG:-movieroll/configs/stage3_dmd_rolling.yaml}
NPROC=${NPROC:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
# Multi-node: give every node the same MASTER_ADDR. Use an IP, not a hostname, on clusters where all
# nodes report the same hostname.
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29531}

# torchrun puts the SCRIPT directory on sys.path, not the cwd, so the repo has to come in via
# PYTHONPATH (an environment setting, not a sys.path edit inside the package).
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ARGS=(--mode stage3 --config "$CONFIG")

# The log goes next to the run's outputs, never into the repository root. A forwarded
# `--override logdir=...` is what the trainer will use, so it wins over the config file; a forwarded
# `--config` replaces CONFIG for the same reason (argparse keeps the last one).
OVERRIDE_LOGDIR=
TRAINER_CONFIG=$CONFIG
prev=
for arg in "$@"; do
    case "$prev|$arg" in
        "--override|logdir="*) OVERRIDE_LOGDIR=${arg#logdir=} ;;
        "--config|"*) TRAINER_CONFIG=$arg ;;
    esac
    case "$arg" in
        --override=logdir=*) OVERRIDE_LOGDIR=${arg#--override=logdir=} ;;
        --config=*) TRAINER_CONFIG=${arg#--config=} ;;
    esac
    prev=$arg
done
if [ -z "${LOG_DIR:-}" ]; then
    if [ -n "$OVERRIDE_LOGDIR" ]; then
        LOG_DIR=$OVERRIDE_LOGDIR
    else
        LOG_DIR=$(python -c "
import sys
from omegaconf import OmegaConf
print(OmegaConf.load(sys.argv[1]).logdir)
" "$TRAINER_CONFIG")
    fi
fi
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/train_dmd_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "[train_dmd] config=$TRAINER_CONFIG nnodes=$NNODES node_rank=$NODE_RANK nproc=$NPROC"
echo "[train_dmd] log=$LOG"

exec torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC" \
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    distill_movieroll.py "${ARGS[@]}" "$@"
