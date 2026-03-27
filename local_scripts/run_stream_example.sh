#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${SCRIPT_DIR}/../build"
EXECUTABLE_NAME="stream_example"

ITERATIONS=(5000 10000 20000 50000)

if [[ ! -x "${BUILD_DIR}/${EXECUTABLE_NAME}" ]]; then
  echo "Error: executable '${EXECUTABLE_NAME}' not found in '${BUILD_DIR}'."
  echo "Make sure you have already built the project (e.g., via CMake) before running this script."
  exit 1
fi

cd "${BUILD_DIR}"

for ITER in "${ITERATIONS[@]}"; do
  echo "Running ${EXECUTABLE_NAME} with iterations=${ITER}..."
  "./${EXECUTABLE_NAME}" "${ITER}"
  echo
done
