#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=csc367-compute
#SBATCH --job-name profiling
#SBATCH --gres=gpu
#SBATCH --output=%x_%j.out

set -euo pipefail

MATRIX_SIZE="${1:-4096}"

WORK_DIR="$HOME/GPU_CSC367"
BUILD_DIR="${WORK_DIR}/build"
EXECUTABLE="${BUILD_DIR}/shared_memory_no_conflict"
REPORT_DIR="${WORK_DIR}/prof_reports"
NCU="/usr/local/cuda-12.5/bin/ncu"

if [[ ! -x "${EXECUTABLE}" ]]; then
  echo "Error: executable 'shared_memory_no_conflict' not found in '${BUILD_DIR}'."
  echo "Run the build script first: sbatch build.sh"
  exit 1
fi

mkdir -p "${REPORT_DIR}"

${NCU} \
  --set roofline \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section SourceCounters \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis_Tables \
  --export "${REPORT_DIR}/shared_memory_report_${MATRIX_SIZE}.ncu-rep" \
  "${EXECUTABLE}" "${MATRIX_SIZE}"
