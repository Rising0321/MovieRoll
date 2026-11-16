#!/bin/bash
# MovieRoll online serving (one GPU; see vllm_omni_movieroll/deploy/ for the multi-GPU layouts).
# MODEL is a directory assembled by examples/offline_inference/movieroll/download_movieroll.py.

MODEL="${MODEL:?set MODEL to an assembled MovieRoll model directory}"
PORT="${PORT:-8091}"
DEPLOY="${DEPLOY:-$(dirname "$0")/../../../deploy/movieroll_1gpu.yaml}"
# OMP_NUM_THREADS and VLLM_OMNI_DLO_DP_WAVE_TIMEOUT defaults (the deploy YAMLs use the multi-process executor)
. "$(dirname "$0")/../../movieroll_env.sh"

vllm serve "$MODEL" --omni --port "$PORT" --deploy-config "$DEPLOY" "$@"
