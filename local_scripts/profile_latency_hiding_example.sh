#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <chain_length> <threads_per_block>"
  exit 1
fi

CHAIN_LENGTH="$1"
THREADS="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
EXECUTABLE="${BUILD_DIR}/latency_hiding_example"
REPORT_DIR="${SCRIPT_DIR}/../prof_reports"

if [[ ! -x "${EXECUTABLE}" ]]; then
  echo "Error: executable 'latency_hiding_example' not found in '${BUILD_DIR}'."
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
  --export "${REPORT_DIR}/latency_hiding_report_${CHAIN_LENGTH}_${THREADS}.ncu-rep" \
  "${EXECUTABLE}" "${CHAIN_LENGTH}" "${THREADS}"
