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

module load cuda/12.5 2>/dev/null || true

mkdir -p "${WORK_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${PROJECT_DIR}" -DCUDA_ARCHITECTURES="86"
make -j"$(nproc)"

echo "Build complete. Executables in ${BUILD_DIR}:"
ls -1 warp_example stride_example cpu_example 2>/dev/null
