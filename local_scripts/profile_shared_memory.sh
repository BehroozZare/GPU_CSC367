#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
EXECUTABLE="${BUILD_DIR}/shared_memory_no_conflict"
REPORT_DIR="${SCRIPT_DIR}/../prof_reports"
MATRIX_SIZE="${1:-4096}"

if [[ ! -x "${EXECUTABLE}" ]]; then
  echo "Error: executable 'shared_memory_no_conflict' not found in '${BUILD_DIR}'."
  echo "Make sure you have already built the project before running this script."
  exit 1
fi

mkdir -p "${REPORT_DIR}"

sudo /usr/local/cuda/bin/ncu \
  --set roofline \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section SourceCounters \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis_Tables \
  --export "${REPORT_DIR}/shared_memory_report_${MATRIX_SIZE}.ncu-rep" \
  "${EXECUTABLE}" "${MATRIX_SIZE}"
