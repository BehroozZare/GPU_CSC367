#!/usr/bin/env bash

set -euo pipefail

NUM_STREAMS="${1:-4}"
BLUR_RADIUS="${2:-7}"
TOOL="${3:-nsys}"       # "nsys" for timeline, "ncu" for kernel metrics
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
EXECUTABLE="${BUILD_DIR}/stream_image_blur_example"
INPUT_PGM="${SCRIPT_DIR}/../input/test.pgm"
REPORT_DIR="${SCRIPT_DIR}/../prof_reports"

if [[ ! -x "${EXECUTABLE}" ]]; then
  echo "Error: executable 'stream_image_blur_example' not found in '${BUILD_DIR}'."
  echo "Make sure you have already built the project before running this script."
  exit 1
fi

if [[ ! -f "${INPUT_PGM}" ]]; then
  echo "Error: input image '${INPUT_PGM}' not found."
  echo "Run:  cd python_scripts && python prepare_image.py prepare ../input/test.jpg --size 4096 -o ../input/test.pgm"
  exit 1
fi

mkdir -p "${REPORT_DIR}"

TAG="s${NUM_STREAMS}_r${BLUR_RADIUS}"

if [[ "${TOOL}" == "ncu" ]]; then
  echo "Profiling stream_image_blur_example (${NUM_STREAMS} streams, R=${BLUR_RADIUS}) using Nsight Compute..."
  sudo /usr/local/cuda/bin/ncu \
    --set roofline \
    --section MemoryWorkloadAnalysis \
    --section WarpStateStats \
    --section SourceCounters \
    --section LaunchStats \
    --section Occupancy \
    --section MemoryWorkloadAnalysis_Tables \
    --export "${REPORT_DIR}/stream_image_blur_ncu_${TAG}.ncu-rep" \
    "${EXECUTABLE}" "${INPUT_PGM}" "${NUM_STREAMS}" "${BLUR_RADIUS}"
  echo
  echo "Report saved to: ${REPORT_DIR}/stream_image_blur_ncu_${TAG}.ncu-rep"
  echo "Open in Nsight Compute GUI:  ncu-ui ${REPORT_DIR}/stream_image_blur_ncu_${TAG}.ncu-rep"
else
  echo "Profiling stream_image_blur_example (${NUM_STREAMS} streams, R=${BLUR_RADIUS}) using Nsight Systems..."
  sudo /usr/local/cuda/bin/nsys profile \
    --stats=true \
    --force-overwrite=true \
    --output "${REPORT_DIR}/stream_image_blur_${TAG}" \
    "${EXECUTABLE}" "${INPUT_PGM}" "${NUM_STREAMS}" "${BLUR_RADIUS}"
  echo
  echo "Report saved to: ${REPORT_DIR}/stream_image_blur_${TAG}.nsys-rep"
  echo "Open in Nsight Systems GUI:  nsys-ui ${REPORT_DIR}/stream_image_blur_${TAG}.nsys-rep"
fi

echo
echo "Usage: $0 [num_streams] [blur_radius] [nsys|ncu]"
echo "  nsys  -- timeline view (H2D / kernel / D2H overlap across streams)"
echo "  ncu   -- per-kernel metrics (occupancy, roofline, warp stats)"
echo
echo "Tip: compare 1 stream vs 4 streams side-by-side:"
echo "  $0 1 7 nsys    # serial baseline"
echo "  $0 4 7 nsys    # pipelined with 4 streams"
