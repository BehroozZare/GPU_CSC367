#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=csc367-compute
#SBATCH --job-name profiling
#SBATCH --gres=gpu
#SBATCH --output=%x_%j.out

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch $0 <tuning_parameter> <block_size>"
  exit 1
fi

TP="$1"
BLOCK_SIZE="$2"

WORK_DIR="$HOME/GPU_CSC367"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
BUILD_DIR="${WORK_DIR}/build"
EXECUTABLE="${BUILD_DIR}/warp_example"
REPORT_DIR="${WORK_DIR}/prof_reports"
NCU="/usr/local/cuda-12.5/bin/ncu"

if [[ ! -x "${EXECUTABLE}" ]]; then
  echo "Error: executable 'warp_example' not found in '${BUILD_DIR}'."
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
  --export "${REPORT_DIR}/wrap_report_${TP}_${BLOCK_SIZE}.ncu-rep" \
  "${EXECUTABLE}" "${TP}" "${BLOCK_SIZE}"
