#!/usr/bin/env bash
# Build the Abundance-GS environment:   env/setup.sh [PREFIX]      (default: $ABGS_ENV_PREFIX or ~/envs/abundance_gs)
#  1. conda environment: python 3.10 + CUDA 12.9 compiler / runtime headers (for the package's JIT CUDA extensions)
#  2. pip packages pinned in env/requirements.lock (generated from env/requirements.txt), torch 2.9.1+cu126
#  3. tiny-cuda-nn torch bindings built from the pinned upstream commit, with the RTC headers next to the module
#  4. the torchvision weights of the LPIPS metric cached in the build directory (TORCH_HOME)
#  5. the package itself (`pip install -e .`, console script `abgs`) and its CUDA extensions (`abgs build`, compiled
#     through env/env.sh, i.e. with the conda g++ where the system g++ is older than 11)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${1:-${ABGS_ENV_PREFIX:-$HOME/envs/abundance_gs}}"
BUILD="${ABGS_BUILD_DIR:-$HOME/.cache/abundance_gs/build}"
TCNN_COMMIT=0109538c37ac0bf613f2bac8de6cda48352feca7
CONDA="${CONDA_EXE:-conda}"

if [ ! -x "$PREFIX/bin/python" ]; then
  "$CONDA" create -y -p "$PREFIX" -c nvidia -c conda-forge --override-channels \
      python=3.10.20 cuda-nvcc=12.9.86 cuda-cudart-dev=12.9.79 cuda-cccl=12.9.27 cuda-nvrtc-dev=12.9 cuda-driver-dev=12.9 git
fi
PY="$PREFIX/bin/python"
LOCK="$HERE/requirements.lock"; [ -f "$LOCK" ] || LOCK="$HERE/requirements.txt"
"$PY" -m pip install --no-cache-dir -r "$LOCK"
"$PY" -m pip check || true

export CUDA_HOME="$PREFIX" PATH="$PREFIX/bin:$PATH" TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-86}"
NV="$PREFIX/lib/python3.10/site-packages/nvidia"
T="$PREFIX/targets/x86_64-linux"
export CPATH="$T/include:$NV/cusparse/include:$NV/cublas/include:$NV/cusolver/include:$NV/cufft/include:$NV/curand/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$PREFIX/lib:$T/lib:$T/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
if ! "$PY" -c 'import tinycudann' 2>/dev/null; then
  SRC="$BUILD/src/tiny-cuda-nn"
  if [ ! -d "$SRC" ]; then
    git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git "$SRC"
    git -C "$SRC" checkout "$TCNN_COMMIT" && git -C "$SRC" submodule update --init --recursive
  fi
  (cd "$SRC/bindings/torch" && "$PY" -m pip install --no-build-isolation --no-cache-dir .)
fi
# RTC headers next to the module (tiny-cuda-nn checks them at runtime; JIT fusion itself is disabled, nht/tcnn.py):
# its own headers from the checkout, the CUDA fp16 / vector headers from this environment's toolkit
TC="$("$PY" -c 'import importlib.util as u, os; print(os.path.dirname(u.find_spec("tinycudann").origin))')"
SRC="$BUILD/src/tiny-cuda-nn"
mkdir -p "$TC/rtc/include"
[ -f "$TC/rtc/include/tiny-cuda-nn/ministd.h" ] || cp -r "$SRC/include/tiny-cuda-nn" "$SRC/dependencies/pcg32" "$TC/rtc/include/"
INC="$PREFIX/targets/x86_64-linux/include"
for h in cuda_fp16.h cuda_fp16.hpp vector_functions.h vector_functions.hpp vector_types.h; do
  [ -f "$TC/rtc/include/$h" ] || cp "$INC/$h" "$TC/rtc/include/"
done
# LPIPS backbones (torchvision) for the source-field evaluation
TORCH_HOME="$BUILD/torch_home" "$PY" -c 'import torchvision; torchvision.models.vgg16(weights="DEFAULT"); torchvision.models.alexnet(weights="DEFAULT")'
"$PY" -m pip install --no-deps --no-build-isolation -e "$HERE/.."
ABGS_ENV_PREFIX="$PREFIX" ABGS_BUILD_DIR="$BUILD" "$PREFIX/bin/abgs" build
echo "Abundance-GS environment ready: $PREFIX  (set ABGS_ENV_PREFIX if it is not the default)"
