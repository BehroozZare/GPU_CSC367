#!/usr/bin/env bash

set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default build directory (adjust if you use a different one)
BUILD_DIR="${SCRIPT_DIR}/build"
EXECUTABLE_NAME="cpu_example"

# List of tuning parameters to test
TUNING_PARAMS=(1 4 8 16 24 32 48 64 80 96)

if [[ ! -x "${BUILD_DIR}/${EXECUTABLE_NAME}" ]]; then
  echo "Error: executable '${EXECUTABLE_NAME}' not found in '${BUILD_DIR}'."
  echo "Make sure you have already built the project (e.g., via CMake) before running this script."
  exit 1
fi

cd "${BUILD_DIR}"

for TP in "${TUNING_PARAMS[@]}"; do
  echo "Running ${EXECUTABLE_NAME} with tuning_parameter=${TP}..."
  "./${EXECUTABLE_NAME}" "${TP}"
  echo
done

