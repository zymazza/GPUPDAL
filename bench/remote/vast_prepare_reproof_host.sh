#!/usr/bin/env bash
# Prepare a fresh timing host without compiling: install runtime tools,
# unpack the one frozen release archive at its fixed path, and verify it.
#
# Usage: vast_prepare_reproof_host.sh FROZEN_TARBALL FROZEN_SHA FIXTURES_TARBALL
set -euo pipefail

FROZEN_TARBALL=${1:?frozen release tarball}
FROZEN_SHA=${2:?frozen release archive SHA-256}
FIXTURES_TARBALL=${3:?fixtures tarball}
ROOT=/workspace/reproof

if [ "$(sha256sum "$FROZEN_TARBALL" | awk '{print $1}')" != "$FROZEN_SHA" ]; then
  echo "frozen release archive hash mismatch" >&2
  exit 2
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  python3 python3-numpy jq ca-certificates \
  gdal-bin proj-bin \
  libgdal-dev libproj-dev libgeos-dev libgeotiff-dev libxml2-dev zlib1g-dev \
  libzstd-dev liblzma-dev libcurl4-openssl-dev libssl-dev libjsoncpp-dev \
  libtiff-dev libsqlite3-dev >/tmp/pdg-reproof-runtime-apt.log 2>&1

mkdir -p "$ROOT"
tar -xzf "$FROZEN_TARBALL" -C "$ROOT"
mkdir -p "$ROOT/fixtures"
tar -xzf "$FIXTURES_TARBALL" -C "$ROOT/fixtures"
python3 "$ROOT/payload/evidence/scripts/pdg/artifact_manifest.py" check \
  --manifest "$ROOT/release-manifest.json" --root "$ROOT/payload" \
  --report "$ROOT/host-artifact-check.json"
test "$(find "$ROOT/fixtures" -maxdepth 1 -type f | wc -l)" -ge 6
echo "REPROOF-HOST-READY"
