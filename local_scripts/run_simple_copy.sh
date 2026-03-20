#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${SCRIPT_DIR}/../build"
EXECUTABLE_NAME="simple_copy"

BLOCK_SIZES=(256)

if [[ ! -x "${BUILD_DIR}/${EXECUTABLE_NAME}" ]]; then
  echo "Error: executable '${EXECUTABLE_NAME}' not found in '${BUILD_DIR}'."
  echo "Make sure you have already built the project (e.g., via CMake) before running this script."
  exit 1
fi

cd "${BUILD_DIR}"

for BLOCK_SIZE in "${BLOCK_SIZES[@]}"; do
  echo "Running ${EXECUTABLE_NAME} with block_size=${BLOCK_SIZE}..."
  "./${EXECUTABLE_NAME}" "${BLOCK_SIZE}"
  echo
done
