#!/bin/bash
# Build the venv this plugin was verified in (vLLM-Omni 0.32.0, torch 2.13.0+cu129) next to, not inside, any other
# environment.
#
#   bash build_venv.sh <venv dir> [python3.12] [--native-reference]
#
# --native-reference also installs what native MovieRoll (generate_movieroll.py) needs to run in this venv, for
# same-venv comparisons: its extra packages and FlashAttention-2 / -3 2.8.1 built from source for sm90.
set -euo pipefail
VENV=${1:?venv dir}; PY=${2:-python3.12}; NATIVE=${3:-}
HERE=$(cd "$(dirname "$0")/.." && pwd)
CU129=https://download.pytorch.org/whl/cu129
"$PY" -m venv "$VENV"
pip() { "$VENV/bin/pip" --disable-pip-version-check "$@"; }

# the CUDA 12.9 builds of torch (PyPI's torch >= 2.11 is the CUDA 13 build, which needs a driver recent enough for CUDA 13)
pip install --index-url "$CU129" --extra-index-url https://pypi.org/simple torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0
# vLLM 0.32.0's CUDA 12.9 wheel without its dependencies, then those minus llguidance: vLLM pins llguidance 1.7.x, whose
# only Linux wheel needs glibc 2.31 (hosts with glibc 2.28 have none); 1.6.0 serves vLLM's structured output, which a diffusion
# pipeline does not use
pip install --no-deps "https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
"$VENV/bin/python" - > "$VENV/vllm_deps.txt" <<'PYEOF'
import importlib.metadata as md, re
for req in md.distribution("vllm").requires or []:
    name = re.split(r"[ ;<>=!~\[(]", req, 1)[0].lower().replace("_", "-")
    if "extra ==" not in req and name != "llguidance":
        print(req)
PYEOF
pip install -r "$VENV/vllm_deps.txt"
pip install --no-deps --only-binary=:all: llguidance==1.6.0
pip install vllm-omni==0.32.0
# PyPI's torchcodec 0.17 is a CUDA 13 build that fails to load; vLLM-Omni imports it
pip install --no-deps --index-url "$CU129" torchcodec==0.16.0
pip install --no-deps -e "$HERE"

if [ "$NATIVE" = --native-reference ]; then
  pip install accelerate "peft>=0.18" omegaconf loguru imageio imageio-ffmpeg qwen-vl-utils tiktoken psutil einops ninja
  SRC=$(mktemp -d)/flash-attention
  git clone --depth 1 --branch v2.8.1 --recurse-submodules --shallow-submodules https://github.com/Dao-AILab/flash-attention.git "$SRC"
  (cd "$SRC" && FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 pip install --no-build-isolation --no-deps .)
  (cd "$SRC/hopper" && pip install --no-build-isolation --no-deps .)
fi
"$VENV/bin/python" -c "import torch, vllm_omni, vllm_omni_movieroll; print('torch', torch.__version__, 'vllm-omni ok')"
