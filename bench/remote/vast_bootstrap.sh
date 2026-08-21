#!/usr/bin/env bash
# Bootstrap a rented Vast.ai/cloud box (Ubuntu 24.04, CUDA 13.3 devel image)
# for the cross-machine reference benchmark (B0275): installs the PDAL build
# dependencies, builds the pinned upstream oracle (PDAL 2.10.0 at the commit
# recorded in cmake/pdg-oracle.cmake) and the fork's CUDA Release tree from a
# source tarball, and leaves both under /workspace.
#
# Usage: vast_bootstrap.sh <fork-source.tar.gz> [jobs] [cuda-jobs]
set -euo pipefail
TARBALL=${1:?fork source tarball}
JOBS=${2:-$(nproc)}
CUDA_JOBS=${3:-4}
ROOT=/workspace
mkdir -p "$ROOT"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential cmake ninja-build git pkg-config ca-certificates curl wget \
  python3 python3-numpy python3-venv unzip \
  libgdal-dev libproj-dev libgeos-dev libgeotiff-dev libxml2-dev zlib1g-dev \
  libzstd-dev liblzma-dev libcurl4-openssl-dev libssl-dev libjsoncpp-dev \
  libtiff-dev libsqlite3-dev gdal-bin proj-bin >/tmp/apt.log 2>&1
echo "apt done"

# Fork source.
mkdir -p "$ROOT/pdal-gpu"
tar -xzf "$TARBALL" -C "$ROOT/pdal-gpu"
ORACLE_COMMIT=$(grep -oP 'PDG_ORACLE_COMMIT "\K[0-9a-f]+' "$ROOT/pdal-gpu/cmake/pdg-oracle.cmake")
echo "oracle commit $ORACLE_COMMIT"

# Pinned upstream oracle.
if [ ! -x "$ROOT/oracle/bin/pdal" ]; then
  rm -rf "$ROOT/PDAL-upstream"; mkdir -p "$ROOT/PDAL-upstream"
  git -C "$ROOT/PDAL-upstream" init -q
  git -C "$ROOT/PDAL-upstream" remote add origin https://github.com/PDAL/PDAL.git
  for attempt in 1 2 3; do
    git -C "$ROOT/PDAL-upstream" fetch -q --depth 1 origin "$ORACLE_COMMIT" && break
    sleep 10
  done
  git -C "$ROOT/PDAL-upstream" checkout -q FETCH_HEAD
  cmake -S "$ROOT/PDAL-upstream" -B "$ROOT/oracle" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DWITH_TESTS=OFF -DBUILD_PLUGIN_HDF=OFF \
    -DBUILD_PLUGIN_TILEDB=OFF -DBUILD_PLUGIN_ICEBRIDGE=OFF >/tmp/oracle-configure.log 2>&1
  cmake --build "$ROOT/oracle" -j "$JOBS" >/tmp/oracle-build.log 2>&1
fi
echo "oracle built: $("$ROOT/oracle/bin/pdal" --version | head -1)"

# Fork CUDA Release tree (same preset flags as build/pdg-cuda-release locally).
cmake -S "$ROOT/pdal-gpu" -B "$ROOT/pdal-gpu/build/pdg-cuda-release" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DWITH_PDG=ON -DPDG_ENABLE_CUDA=ON \
  -DPDG_WARNINGS_AS_ERRORS=OFF -DPDG_BUILD_TESTS=ON -DWITH_TESTS=OFF \
  -DWITH_GCS=OFF -DWITH_BACKTRACE=OFF -DBUILD_PLUGIN_ARROW=OFF \
  -DBUILD_PLUGIN_TILEDB=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DPDG_PINNED_ORACLE_EXECUTABLE="$ROOT/oracle/bin/pdal" \
  -DPDG_CUDA_ARCHITECTURES="${PDG_CUDA_ARCHITECTURES:-89}" \
  -DPDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES=OFF >/tmp/fork-configure.log 2>&1
cmake --build "$ROOT/pdal-gpu/build/pdg-cuda-release" -j "$JOBS" \
  --target pdg_cli pdg_engine pdal pdg_frozen_time >/tmp/fork-build.log 2>&1
echo "fork built: $("$ROOT/pdal-gpu/build/pdg-cuda-release/bin/pdg" --version 2>&1 | head -1)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core'
free -g | head -2
