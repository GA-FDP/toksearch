#!/bin/bash
set -euo pipefail

# Plain python3: versioneer needs only git, and a pixi solve would fail on a
# platform whose dev dependencies are not all published yet.
export PKG_VERSION=$(python3 print_version.py)

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
