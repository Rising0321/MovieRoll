#!/bin/bash
# Streaming SFT, stage 1 (single-event) or stage 2 (multi-event).
#
#   bash movieroll/scripts/launch/train_sft.sh [extra args forwarded to train_movieroll.py]
#   CONFIG=movieroll/configs/sft/stage2_multi_event.yaml bash movieroll/scripts/launch/train_sft.sh
#   NNODES=2 NODE_RANK=<0|1> MASTER_ADDR=<node 0 address> bash movieroll/scripts/launch/train_sft.sh
#
# Every training setting lives in the YAML (CONFIG, default stage 1); this script only sets the launch topology
# and the environment. The trainer writes its own log, <output_dir>/train.log.
#
# PYTORCH_CUDA_ALLOC_CONF defaults to expandable_segments:True (less fragmentation over a long run); an
# exported value is kept. Cluster-specific tuning (NCCL_SOCKET_IFNAME, IB settings) is deliberately NOT set
# here: export it in the environment the launcher runs in.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO"

CONFIG=${CONFIG:-movieroll/configs/sft/stage1_single_event.yaml}
NPROC=${NPROC:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29521}

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[train_sft] config=$CONFIG nnodes=$NNODES node_rank=$NODE_RANK nproc=$NPROC"
exec torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC" \
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    train_movieroll.py --config "$CONFIG" "$@"
