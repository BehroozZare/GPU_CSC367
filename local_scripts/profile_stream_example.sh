#!/usr/bin/env bash

set -euo pipefail

ITERATIONS="${1:-10000}"
TOOL="${2:-nsys}"       # "nsys" for timeline, "ncu" for kernel metrics
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
EXECUTABLE="${BUILD_DIR}/stream_example"
REPORT_DIR="${SCRIPT_DIR}/../prof_reports"

if [[ ! -x "${EXECUTABLE}" ]]; then
  echo "Error: executable 'stream_example' not found in '${BUILD_DIR}'."
  echo "Make sure you have already built the project before running this script."
  exit 1
fi

mkdir -p "${REPORT_DIR}"

if [[ "${TOOL}" == "ncu" ]]; then
  echo "Profiling stream_example with iterations=${ITERATIONS} using Nsight Compute..."
  sudo /usr/local/cuda/bin/ncu \
    --set roofline \
    --section MemoryWorkloadAnalysis \
    --section WarpStateStats \
    --section SourceCounters \
    --section LaunchStats \
    --section Occupancy \
    --section MemoryWorkloadAnalysis_Tables \
    --export "${REPORT_DIR}/stream_example_ncu_${ITERATIONS}.ncu-rep" \
    "${EXECUTABLE}" "${ITERATIONS}"
  echo
  echo "Report saved to: ${REPORT_DIR}/stream_example_ncu_${ITERATIONS}.ncu-rep"
  echo "Open in Nsight Compute GUI:  ncu-ui ${REPORT_DIR}/stream_example_ncu_${ITERATIONS}.ncu-rep"
else
  echo "Profiling stream_example with iterations=${ITERATIONS} using Nsight Systems..."
  sudo /usr/local/cuda/bin/nsys profile \
    --stats=true \
    --force-overwrite=true \
    --output "${REPORT_DIR}/stream_example_${ITERATIONS}" \
    "${EXECUTABLE}" "${ITERATIONS}"
  echo
  echo "Report saved to: ${REPORT_DIR}/stream_example_${ITERATIONS}.nsys-rep"
  echo "Open in Nsight Systems GUI:  nsys-ui ${REPORT_DIR}/stream_example_${ITERATIONS}.nsys-rep"
fi

echo
echo "Usage: $0 [iterations] [nsys|ncu]"
echo "  nsys  -- timeline view (stream concurrency, kernel overlap)"
echo "  ncu   -- per-kernel metrics (occupancy, roofline, warp stats)"
