#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${SCRIPT_DIR}/../build"
OUTPUT_CSV="${SCRIPT_DIR}/../output/shared_memory_transpose.csv"
EXECUTABLE_NAME="shared_memory_basic"
MATRIX_SIZE="${1:-4096}"

if [[ ! -x "${BUILD_DIR}/${EXECUTABLE_NAME}" ]]; then
  echo "Error: executable '${EXECUTABLE_NAME}' not found in '${BUILD_DIR}'."
  echo "Make sure you have already built the project (e.g., via CMake) before running this script."
  exit 1
fi

rm -f "${OUTPUT_CSV}"

cd "${BUILD_DIR}"

echo "Running ${EXECUTABLE_NAME} with matrix_size=${MATRIX_SIZE}..."
"./${EXECUTABLE_NAME}" "${MATRIX_SIZE}"
echo
echo "Results written to ${OUTPUT_CSV}"
