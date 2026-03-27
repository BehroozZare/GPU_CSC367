#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${SCRIPT_DIR}/../build"
EXECUTABLE_NAME="stream_image_blur_example"
INPUT_PGM="${SCRIPT_DIR}/../input/test.pgm"

BLUR_RADIUS=7
NUM_STREAMS=(1 2 4 8)

if [[ ! -x "${BUILD_DIR}/${EXECUTABLE_NAME}" ]]; then
  echo "Error: executable '${EXECUTABLE_NAME}' not found in '${BUILD_DIR}'."
  echo "Make sure you have already built the project (e.g., via CMake) before running this script."
  exit 1
fi

if [[ ! -f "${INPUT_PGM}" ]]; then
  echo "Error: input image '${INPUT_PGM}' not found."
  echo "Run:  cd python_scripts && python prepare_image.py ../input/test.jpg --size 4096 -o ../input/test.pgm"
  exit 1
fi

cd "${BUILD_DIR}"

for S in "${NUM_STREAMS[@]}"; do
  echo "Running ${EXECUTABLE_NAME} with num_streams=${S}, blur_radius=${BLUR_RADIUS}..."
  "./${EXECUTABLE_NAME}" "${INPUT_PGM}" "${S}" "${BLUR_RADIUS}"
  echo
done

echo "Converting blurred output to PNG for slides..."
python3 "${SCRIPT_DIR}/../python_scripts/prepare_image.py" topng \
  "${SCRIPT_DIR}/../output/blurred.pgm" \
  -o "${SCRIPT_DIR}/../output/blurred.png"

python3 "${SCRIPT_DIR}/../python_scripts/prepare_image.py" topng \
  "${INPUT_PGM}" \
  -o "${SCRIPT_DIR}/../output/original.png"

echo "Slide-ready images:"
echo "  output/original.png  (before blur)"
echo "  output/blurred.png   (after blur, R=${BLUR_RADIUS})"
