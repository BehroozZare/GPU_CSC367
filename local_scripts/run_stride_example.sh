#!/usr/bin/env bash

set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default build directory (adjust if you use a different one)
BUILD_DIR="${SCRIPT_DIR}/../build"
EXECUTABLE_NAME="stride_example"

STRIDES=(1 2 4 8 16 32 64 128 256 512 1024 2048)
BLOCK_SIZES=(256)

if [[ ! -x "${BUILD_DIR}/${EXECUTABLE_NAME}" ]]; then
  echo "Error: executable '${EXECUTABLE_NAME}' not found in '${BUILD_DIR}'."
  echo "Make sure you have already built the project (e.g., via CMake) before running this script."
  exit 1
fi

cd "${BUILD_DIR}"

for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
  for STRIDE in "${STRIDES[@]}"; do
    echo "Running ${EXECUTABLE_NAME} with stride=${STRIDE}, block_size=${BLOCK_SIZE}..."
    "./${EXECUTABLE_NAME}" "${STRIDE}" "${BLOCK_SIZE}"
    echo
  done
done
