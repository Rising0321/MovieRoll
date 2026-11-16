#!/bin/bash
# Build the venv this adapter was verified in (SGLang 0.5.10.post1, torch 2.9.1+cu128) next to, not inside, any other
# environment.
#
#   bash build_venv.sh <venv dir> [python3.12] [--native-reference]
#
# --native-reference also installs what native MovieRoll (generate_movieroll.py) needs to run in this venv, for
# same-venv comparisons: its extra packages and FlashAttention-2 / -3 2.8.1 built from source for sm90 (the adapter
# itself needs neither).
set -euo pipefail
VENV=${1:?venv dir}; PY=${2:-python3.12}; NATIVE=${3:-}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
"$PY" -m venv "$VENV"
pip() { "$VENV/bin/pip" --disable-pip-version-check "$@"; }

# SGLang pins torch 2.9.1 itself; PyPI's 2.9.1 is the CUDA 12.8 build. 0.5.10.post1 is the last release whose
# sglang-kernel is a CUDA 12 build (0.5.11 needs CUDA 13, i.e. a driver recent enough for CUDA 13)
pip install "sglang[diffusion]==0.5.10.post1"

if [ "$NATIVE" = --native-reference ]; then
  pip install accelerate "peft>=0.18" omegaconf loguru imageio imageio-ffmpeg qwen-vl-utils tiktoken psutil einops ninja
  SRC=$(mktemp -d)/flash-attention
  git clone --depth 1 --branch v2.8.1 --recurse-submodules --shallow-submodules https://github.com/Dao-AILab/flash-attention.git "$SRC"
  (cd "$SRC" && FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 pip install --no-build-isolation --no-deps .)
  (cd "$SRC/hopper" && pip install --no-build-isolation --no-deps .)
fi
(cd "$HERE" && "$VENV/bin/python" -c "import torch, sglang, sglang_movieroll; print('torch', torch.__version__, 'sglang ok')")
