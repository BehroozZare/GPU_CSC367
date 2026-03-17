#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=csc367-compute
#SBATCH --job-name build
#SBATCH --gres=gpu
#SBATCH --output=%x_%j.out

set -euo pipefail

WORK_DIR="$HOME/GPU_CSC367"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
BUILD_DIR="${WORK_DIR}/build"

export PATH="/usr/local/cuda/bin:${PATH}"

mkdir -p "${WORK_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${PROJECT_DIR}" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86
make -j"$(nproc)"

echo "Build complete. Executables in ${BUILD_DIR}:"
ls -1 warp_example stride_example cpu_example 2>/dev/null
