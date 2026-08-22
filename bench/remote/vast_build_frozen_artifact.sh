#!/usr/bin/env bash
# Build the candidate and pinned oracle exactly once, install them at the
# fixed path used on every timing host, then close and archive the payload.
#
# Usage:
#   vast_build_frozen_artifact.sh SOURCE_TARBALL SOURCE_SHA SOURCE_COMMIT RELEASE_ID [JOBS]
set -euo pipefail

SOURCE_TARBALL=${1:?source tarball}
SOURCE_SHA=${2:?source archive SHA-256}
SOURCE_COMMIT=${3:?source commit}
RELEASE_ID=${4:?release id}
JOBS=${5:-2}
ROOT=/workspace/reproof
SOURCE=$ROOT/source
CANDIDATE_BUILD=$SOURCE/build/pdg-cuda-release
ORACLE_SOURCE=$ROOT/oracle-source
ORACLE_BUILD=$ROOT/oracle-build
PAYLOAD=$ROOT/payload
MANIFEST=$ROOT/release-manifest.json

if [ "$(sha256sum "$SOURCE_TARBALL" | awk '{print $1}')" != "$SOURCE_SHA" ]; then
  echo "source archive hash mismatch" >&2
  exit 2
fi
if [ -e "$PAYLOAD" ] || [ -e "$MANIFEST" ]; then
  echo "builder output already exists; use a fresh instance" >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential cmake ninja-build git pkg-config ca-certificates curl \
  python3 python3-numpy jq \
  libgdal-dev libproj-dev libgeos-dev libgeotiff-dev libxml2-dev zlib1g-dev \
  libzstd-dev liblzma-dev libcurl4-openssl-dev libssl-dev libjsoncpp-dev \
  libtiff-dev libsqlite3-dev >/tmp/pdg-reproof-apt.log 2>&1

AVAILABLE_KIB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if [ "${AVAILABLE_KIB:-0}" -lt 8388608 ]; then
  echo "builder has less than 8 GiB MemAvailable" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
free -h

mkdir -p "$SOURCE" "$PAYLOAD/candidate" "$PAYLOAD/oracle"
tar -xzf "$SOURCE_TARBALL" -C "$SOURCE"
ORACLE_COMMIT=$(grep -oP 'PDG_ORACLE_COMMIT "\K[0-9a-f]+' \
  "$SOURCE/cmake/pdg-oracle.cmake")

mkdir -p "$ORACLE_SOURCE"
git -C "$ORACLE_SOURCE" init -q
git -C "$ORACLE_SOURCE" remote add origin https://github.com/PDAL/PDAL.git
for attempt in 1 2 3; do
  if git -C "$ORACLE_SOURCE" fetch -q --depth 1 origin "$ORACLE_COMMIT"; then
    break
  fi
  if [ "$attempt" = 3 ]; then
    echo "unable to fetch pinned PDAL oracle" >&2
    exit 3
  fi
  sleep 10
done
git -C "$ORACLE_SOURCE" checkout -q FETCH_HEAD

cmake -S "$ORACLE_SOURCE" -B "$ORACLE_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PAYLOAD/oracle" \
  -DWITH_TESTS=OFF -DBUILD_PLUGIN_HDF=OFF -DBUILD_PLUGIN_TILEDB=OFF \
  -DBUILD_PLUGIN_ICEBRIDGE=OFF >/tmp/pdg-reproof-oracle-configure.log 2>&1
cmake --build "$ORACLE_BUILD" -j "$JOBS" \
  >/tmp/pdg-reproof-oracle-build.log 2>&1
cmake --install "$ORACLE_BUILD" >/tmp/pdg-reproof-oracle-install.log 2>&1

cmake -S "$SOURCE" -B "$CANDIDATE_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PAYLOAD/candidate" \
  -DWITH_PDG=ON -DGPUPDAL_ENABLE_CUDA=ON -DPDG_WARNINGS_AS_ERRORS=OFF \
  -DPDG_BUILD_TESTS=OFF -DWITH_TESTS=OFF -DWITH_GCS=OFF \
  -DWITH_BACKTRACE=OFF -DBUILD_PLUGIN_ARROW=OFF -DBUILD_PLUGIN_TILEDB=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DPDG_PINNED_ORACLE_EXECUTABLE="$PAYLOAD/oracle/bin/pdal" \
  -DPDG_CUDA_ARCHITECTURES=all \
  -DPDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES=ON \
  >/tmp/pdg-reproof-candidate-configure.log 2>&1
cmake --build "$CANDIDATE_BUILD" -j "$JOBS" \
  >/tmp/pdg-reproof-candidate-build.log 2>&1
cmake --install "$CANDIDATE_BUILD" \
  >/tmp/pdg-reproof-candidate-install.log 2>&1

# The closed-payload format deliberately rejects symlinks. Installed SONAME
# links are replaced with independent regular-file copies before closure.
while IFS= read -r -d '' link; do
  target=$(readlink -f "$link")
  cp --remove-destination "$target" "$link"
done < <(find "$PAYLOAD" -type l -print0)

"$PAYLOAD/oracle/bin/pdal" --version
PDG_ORACLE_PDAL="$PAYLOAD/oracle/bin/pdal" \
  "$PAYLOAD/candidate/bin/pdg" --version
grep -q '^PDG_CUDA_ARCHITECTURES:STRING=all$' "$CANDIDATE_BUILD/CMakeCache.txt"
grep -q '^PDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES:BOOL=ON$' \
  "$CANDIDATE_BUILD/CMakeCache.txt"

cat > "$ROOT/build-metadata.json" <<JSON
{
  "schema": "pdg-frozen-build-metadata-v1",
  "source_archive_sha256": "$SOURCE_SHA",
  "source_commit": "$SOURCE_COMMIT",
  "oracle_commit": "$ORACLE_COMMIT",
  "release_id": "$RELEASE_ID",
  "container_image": "nvidia/cuda:13.3.1-devel-ubuntu24.04",
  "build_type": "Release",
  "cuda_architectures": "all",
  "portable_architecture_guard": true,
  "candidate_install_prefix": "/workspace/reproof/payload/candidate",
  "oracle_install_prefix": "/workspace/reproof/payload/oracle"
}
JSON

PDG_BUILD_DIR="$CANDIDATE_BUILD" \
PDG_SOURCE_COMMIT="$SOURCE_COMMIT" \
PDG_ARTIFACT_METADATA="$ROOT/build-metadata.json" \
  "$SOURCE/bench/remote/freeze_release_artifact.sh" \
    "$PAYLOAD" candidate/bin/pdg oracle/bin/pdal "$MANIFEST" "$RELEASE_ID"

python3 "$PAYLOAD/evidence/scripts/pdg/artifact_manifest.py" check \
  --manifest "$MANIFEST" --root "$PAYLOAD" \
  --report "$ROOT/builder-artifact-check.json"
tar -czf "$ROOT/frozen-release.tar.gz" -C "$ROOT" \
  payload release-manifest.json builder-artifact-check.json build-metadata.json
sha256sum "$ROOT/frozen-release.tar.gz" > "$ROOT/frozen-release.tar.gz.sha256"
echo "FROZEN-BUILD-COMPLETE $ROOT/frozen-release.tar.gz"
