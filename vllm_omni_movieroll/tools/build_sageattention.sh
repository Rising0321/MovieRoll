#!/bin/bash
# Build SageAttention from source into a venv, for vLLM-Omni's SAGE_ATTN backend. The default build targets Hopper (sm90);
# set TORCH_CUDA_ARCH_LIST (e.g. "8.0;9.0") for other GPUs.
#
#   bash build_sageattention.sh <venv dir>
#
# Commit d1a57a5 of SageAttention's main branch (package version 2.2.0), with the system CUDA 12.9 toolkit, against the
# venv's torch, no build isolation, no dependency changes. Not the v2.2.0 tag: on H20 its sm90 kernel (the one `sageattn`
# picks on Hopper) returns wrong output. Check a build with check_sageattention.py.
set -euo pipefail
VENV=${1:?venv dir}
COMMIT=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
SRC=$(mktemp -d)/SageAttention
git clone https://github.com/thu-ml/SageAttention.git "$SRC"
git -C "$SRC" checkout -q "$COMMIT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0} MAX_JOBS=${MAX_JOBS:-32}
export PATH=$CUDA_HOME/bin:$PATH
(cd "$SRC" && "$VENV/bin/pip" install --no-build-isolation --no-deps .)
"$VENV/bin/python" "$(dirname "$0")/check_sageattention.py"
