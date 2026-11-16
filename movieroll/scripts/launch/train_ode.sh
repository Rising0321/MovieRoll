#!/bin/bash
# Stage-2 ODE regression: the few-step student learns the teacher trajectories that precompute_ode.sh cached.
#
#   bash movieroll/scripts/launch/train_ode.sh [extra args forwarded to distill_movieroll.py]
#   NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node 0 address> bash movieroll/scripts/launch/train_ode.sh
#
# Every training setting lives in the YAML (CONFIG, default movieroll/configs/stage2_ode.yaml); this script only
# sets the launch topology and the environment. The trainer writes its own log under the config's logdir.
#
# PYTORCH_CUDA_ALLOC_CONF defaults to expandable_segments:True (less fragmentation over a long run); an
# exported value is kept. Cluster-specific tuning (NCCL_SOCKET_IFNAME, IB settings) is deliberately NOT set
# here: export it in the environment the launcher runs in.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO"

CONFIG=${CONFIG:-movieroll/configs/stage2_ode.yaml}
NPROC=${NPROC:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29541}

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[train_ode] config=$CONFIG nnodes=$NNODES node_rank=$NODE_RANK nproc=$NPROC"
exec torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC" \
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    distill_movieroll.py --mode stage2 --config "$CONFIG" "$@"
