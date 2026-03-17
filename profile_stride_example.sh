#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <stride> <block_size>"
  exit 1
fi

STRIDE="$1"
BLOCK_SIZE="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
EXECUTABLE="${BUILD_DIR}/stride_example"
REPORT_DIR="${SCRIPT_DIR}/prof_reports"

if [[ ! -x "${EXECUTABLE}" ]]; then
  echo "Error: executable 'stride_example' not found in '${BUILD_DIR}'."
  echo "Make sure you have already built the project before running this script."
  exit 1
fi

mkdir -p "${REPORT_DIR}"

sudo /usr/local/cuda/bin/ncu \
  --set roofline \
  --section MemoryWorkloadAnalysis\
  --section WarpStateStats\
  --section SourceCounters\
  --section LaunchStats\
  --section Occupancy\
  --section MemoryWorkloadAnalysis_Tables\
  --export "${REPORT_DIR}/stride_report_${STRIDE}_${BLOCK_SIZE}.ncu-rep" \
  "${EXECUTABLE}" "${STRIDE}" "${BLOCK_SIZE}"
