#!/usr/bin/env bash
# Prepare a bootstrapped rented box (after bench/remote/vast_bootstrap.sh) for the
# independent benchmark and run it.
#   remote_setup.sh <label> <lastools-tarball> <fixtures-dir> [sizes...]
# Expects /workspace/oracle (pinned PDAL) and /workspace/pdal-gpu/build/pdg-cuda-release.
set -euo pipefail
LABEL=${1:?label}
LT_TAR=${2:?lastools tarball}
FIX=${3:?fixtures dir with 1m.laz 4m.laz 16m.laz 1m.copc.laz 1m-ortho-3857.tif}
shift 3
SIZES=${*:-1m 4m 16m 47m}
ROOT=/workspace
REPO=$ROOT/pdal-gpu
export DEBIAN_FRONTEND=noninteractive

apt-get install -y -qq --no-install-recommends python3-gdal curl libzip-dev zipcmp zipmerge ziptool >/tmp/apt2.log 2>&1 || true

# LAStools native Linux binaries (unlicensed). The tarball must contain bin/<tool>64,
# bin/lib (bundled libraries) and bin/serf/geo (EPSG tables; without them the tools warn and exit 1).
mkdir -p $ROOT/lastools
tar -xzf "$LT_TAR" -C $ROOT/lastools
ls $ROOT/lastools/bin | head -3
# Wrappers so the vendor's bundled libraries are found without a global LD_LIBRARY_PATH.
mkdir -p $ROOT/lastools/wrap
for b in $ROOT/lastools/bin/*64; do
  n=$(basename "$b")
  # System PROJ/GeoTIFF/TIFF first (the vendor's bundled libproj segfaults at exit on Ubuntu 24.04),
  # the bundled directory only for libraries the system lacks (libjpeg.so.62).
  printf '#!/bin/sh\nexport LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:%s\nexec %s "$@"\n' "$ROOT/lastools/bin/lib" "$b" > "$ROOT/lastools/wrap/$n"
  chmod +x "$ROOT/lastools/wrap/$n"
done
$ROOT/lastools/wrap/laszip64 -version 2>&1 | head -2

# Install the pinned oracle so pdal_wrench can link against it.
if [ ! -x $ROOT/oracle-install/bin/pdal ]; then
  cmake --install $ROOT/oracle --prefix $ROOT/oracle-install >/tmp/oracle-install.log 2>&1
fi
# pdal_wrench (the engine behind QGIS point-cloud processing).
if [ ! -x $ROOT/wrench/build/pdal_wrench ]; then
  rm -rf $ROOT/wrench
  git clone -q --depth 1 https://github.com/PDAL/wrench.git $ROOT/wrench
  cmake -S $ROOT/wrench -B $ROOT/wrench/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$ROOT/oracle-install >/tmp/wrench-configure.log 2>&1
  cmake --build $ROOT/wrench/build -j "$(nproc)" >/tmp/wrench-build.log 2>&1
fi
$ROOT/wrench/build/pdal_wrench --version

# The 47M-point public AHN4 tile, downloaded on the box.
mkdir -p "$FIX"
[ -f "$FIX/47m.laz" ] || curl -sSL -o "$FIX/47m.laz" "https://geotiles.citg.tudelft.nl/AHN4_T/25GN1_01.LAZ"
ls -la "$FIX"

export IB_WORK=$ROOT/independent-bench
export IB_SOURCE_DIR=$FIX
export IB_PDG=$REPO/build/pdg-cuda-release/bin/pdg
export IB_PDAL_PINNED=$ROOT/oracle/bin/pdal
export IB_LASTOOLS=$ROOT/lastools/wrap
export IB_WRENCH=$ROOT/wrench/build/pdal_wrench
export IB_PDAL_SYS=/nonexistent
export IB_QGIS_PROCESS=/nonexistent

THREADS=$(awk '{if($1=="max")print 0; else printf "%d", $1/$2}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "${THREADS:-0}" -gt 0 ] || THREADS=$(nproc)
echo "effective threads: $THREADS (nproc $(nproc))"
cd $REPO
python3 bench/independent/prepare.py --sizes $SIZES 2>&1 | grep -v FutureWarning | grep -v warnings.warn
TOOLS="pdg_gpu pdg_cpu pdg_gpu_all pdal_pinned lastools wrench"
for size in $SIZES; do
  case $size in
    1m) R=3;; 4m) R=3;; 16m) R=3;; *) R=3;;
  esac
  case $size in
    *) W=1;;
  esac
  python3 bench/independent/harness.py --label "$LABEL" --sizes "$size" --repeats $R --warmup $W \
    --tools $TOOLS --threads "$THREADS" 2>&1 | tee -a $IB_WORK/run-$size.log
done
echo "ALL DONE $LABEL"
