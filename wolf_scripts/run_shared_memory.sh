#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=csc367-compute
#SBATCH --job-name shared_memory
#SBATCH --gres=gpu
#SBATCH --output=%x_%j.out

set -euo pipefail

MATRIX_SIZE="${1:-4096}"

WORK_DIR="$HOME/GPU_CSC367"
BUILD_DIR="${WORK_DIR}/build"
OUTPUT_CSV="${WORK_DIR}/output/shared_memory_transpose.csv"
EXECUTABLE="${BUILD_DIR}/shared_memory_no_conflict"

if [[ ! -x "${EXECUTABLE}" ]]; then
  echo "Error: executable 'shared_memory_no_conflict' not found in '${BUILD_DIR}'."
  echo "Run the build script first: sbatch build.sh"
  exit 1
fi

mkdir -p "${WORK_DIR}/output"
rm -f "${OUTPUT_CSV}"

echo "Running shared_memory_no_conflict with matrix_size=${MATRIX_SIZE}..."
"${EXECUTABLE}" "${MATRIX_SIZE}"
echo
echo "Results written to ${OUTPUT_CSV}"
