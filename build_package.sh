#!/bin/bash
set -euo pipefail

export PKG_VERSION=$(pixi run python print_version.py)

echo "Building version: ${PKG_VERSION}"

OUTPUT_DIR="${1:-${HOME}/outdir}"
mkdir -p "${OUTPUT_DIR}"

CHANNEL_FLAGS="--channel ga-fdp --channel conda-forge"
if [ -n "${EXTRA_CHANNEL:-}" ]; then
    CHANNEL_FLAGS="--channel ${EXTRA_CHANNEL} ${CHANNEL_FLAGS}"
fi

rattler-build build --recipe recipe/ \
  ${CHANNEL_FLAGS} \
  --channel-priority=disabled \
  --output-dir "${OUTPUT_DIR}"
