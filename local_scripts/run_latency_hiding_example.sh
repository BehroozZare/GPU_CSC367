#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${SCRIPT_DIR}/../build"
EXECUTABLE_NAME="latency_hiding_example"

CHAIN_LENGTH=10
THREAD_SIZES=(32 64 96 128 160 192 224 256)

if [[ ! -x "${BUILD_DIR}/${EXECUTABLE_NAME}" ]]; then
  echo "Error: executable '${EXECUTABLE_NAME}' not found in '${BUILD_DIR}'."
  echo "Make sure you have already built the project (e.g., via CMake) before running this script."
  exit 1
fi

cd "${BUILD_DIR}"

for THREADS in "${THREAD_SIZES[@]}"; do
  echo "Running ${EXECUTABLE_NAME} with chain_length=${CHAIN_LENGTH}, threads_per_block=${THREADS}..."
  "./${EXECUTABLE_NAME}" "${CHAIN_LENGTH}" "${THREADS}"
  echo
done
