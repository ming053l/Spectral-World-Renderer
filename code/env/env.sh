# Environment of every Abundance-GS stage (sourced by abgs.py and env/run.sh), built by env/setup.sh:
# a conda prefix with python 3.10, the CUDA 12.9 compiler and the pinned pip packages (torch 2.9.1+cu126, tiny-cuda-nn,
# torchmetrics, ...). Nothing is put on PYTHONPATH; all code comes from this folder.
export ABGS_ENV_PREFIX="${ABGS_ENV_PREFIX:-$HOME/envs/abundance_gs}"
export ABGS_PYTHON="$ABGS_ENV_PREFIX/bin/python"
export ABGS_BUILD_DIR="${ABGS_BUILD_DIR:-$HOME/.cache/abundance_gs/build}"
export CUDA_HOME="$ABGS_ENV_PREFIX"                         # nvcc 12.9 for the JIT-built CUDA extensions
export TORCH_HOME="$ABGS_BUILD_DIR/torch_home"              # LPIPS backbone weights, cached by env/setup.sh
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-86}"
export MAX_JOBS="${MAX_JOBS:-4}"
export GSPLAT_NVCC_SPLIT_COMPILE="${GSPLAT_NVCC_SPLIT_COMPILE:-8}"
export PATH="$ABGS_ENV_PREFIX/bin:$PATH"
# CUDA library headers shipped with the torch wheel (the conda toolkit has compiler, runtime and CCCL only)
NV="$ABGS_ENV_PREFIX/lib/python3.10/site-packages/nvidia"
T="$ABGS_ENV_PREFIX/targets/x86_64-linux"
export CPATH="$T/include:$NV/cusparse/include:$NV/cublas/include:$NV/cusolver/include:$NV/cufft/include:$NV/curand/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$ABGS_ENV_PREFIX/lib:$T/lib:$T/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
# gsplat needs C++20: on hosts whose system g++ is older than 11 (e.g. Ubuntu 20.04, gcc 9), build the JIT extensions
# with the conda-forge g++ that cuda-nvcc installs into this environment (host compiler of nvcc as well)
if [ -z "${CXX:-}" ] && [ -x "$ABGS_ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++" ] \
   && [ "$(g++ -dumpversion 2>/dev/null | cut -d. -f1)" -lt 11 ] 2>/dev/null; then
  export CC="$ABGS_ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc" CXX="$ABGS_ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export NVCC_PREPEND_FLAGS="-ccbin $CXX${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"
fi
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NUMPY_MADVISE_HUGEPAGE=0
if [ ! -x "$ABGS_PYTHON" ]; then echo "Abundance-GS environment missing at $ABGS_ENV_PREFIX: run env/setup.sh" >&2; fi
