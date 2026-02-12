#!/bin/bash
set -euo pipefail

export GIT_DESCRIBE_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "0.0.0")
export GIT_DESCRIBE_NUMBER=$(git describe --tags --long 2>/dev/null | sed 's/.*-\([0-9]*\)-.*/\1/' || echo "0")
export GIT_DESCRIBE_HASH=$(git describe --tags --long 2>/dev/null | sed 's/.*-g\(.*\)/\1/' || echo "unknown")

echo "Building version: ${GIT_DESCRIBE_TAG} (${GIT_DESCRIBE_NUMBER}_${GIT_DESCRIBE_HASH})"

OUTPUT_DIR="${1:-${HOME}/outdir}"
mkdir -p "${OUTPUT_DIR}"

rattler-build build --recipe recipe/ \
  --channel ga-fdp \
  --channel conda-forge \
  --output-dir "${OUTPUT_DIR}"
