#!/usr/bin/env bash
# Cross-machine reference benchmark on a bootstrapped box (B0275).
#   vast_run.sh <fixtures-dir> <results-dir> [suite-runs] [big-runs]
# Requires vast_bootstrap.sh to have built /workspace/oracle and
# /workspace/pdal-gpu/build/pdg-cuda-release. Every measured pipeline runs
# through scripts/pdg/benchmark_reference.py (alternating oracle/candidate,
# frozen clock, exact comparison); the box's own pinned-oracle build is the
# CPU baseline. Big-cloud inputs are downloaded public AHN4 GeoTiles.
set -uo pipefail
FIX=${1:?fixtures dir}
OUT=${2:?results dir}
SUITE_RUNS=${3:-3}
BIG_RUNS=${4:-2}
ROOT=/workspace
REPO=$ROOT/pdal-gpu
ORACLE=$ROOT/oracle/bin/pdal
PDG=$REPO/build/pdg-cuda-release/bin/pdg
FROZEN=$REPO/build/pdg-cuda-release/lib/libpdg_frozen_time.so
mkdir -p "$OUT"
cd "$REPO"
{
  echo "host: $(hostname)"; date -u
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
  free -g | head -2; nproc
  "$ORACLE" --version | head -1; "$PDG" --version 2>&1 | head -1
  sha256sum "$ORACLE" "$PDG" $REPO/build/pdg-cuda-release/bin/pdg-engine $REPO/build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0
} > "$OUT/machine.txt" 2>&1

suite() {  # name cache-state extra-args...
  local name=$1 cs=$2; shift 2
  python3 scripts/pdg/reference_suite.py run \
    --manifest bench/pipelines/reference/manifest.json --repo-root . \
    --benchmark-runner scripts/pdg/benchmark_reference.py \
    --oracle "$ORACLE" --candidate "$PDG" \
    --work-dir "$OUT/$name-work" --reports "$OUT/$name-reports" \
    --aggregate-output "$OUT/$name.json" \
    --cache-state "$cs" --runs "$SUITE_RUNS" --warmups 1 \
    --frozen-time-library "$FROZEN" --freeze-epoch 1704067200 "$@" \
    > "$OUT/$name.log" 2>&1
  tail -3 "$OUT/$name.log"
}
# The suite validates fixture hashes from the manifest against build/bench-data/reference.
mkdir -p build/bench-data/reference
cp -n "$FIX"/ref-* build/bench-data/reference/ 2>/dev/null || true
suite suite-1m-default warm
suite suite-1m-cuda-experimental warm --candidate-env PDG_EXPERIMENTAL_CUDA_HYBRID=1
suite suite-1m-fast-cuda-experimental warm --contract fast --candidate-arg=--fast --candidate-env PDG_EXPERIMENTAL_CUDA_HYBRID=1

# Big clouds: public AHN4 GeoTiles (EPSG:28992), the same CRS as the reference data.
BIG=$OUT/big
mkdir -p "$BIG"
for tile in 25GN1_01; do
  [ -f "$BIG/$tile.LAZ" ] || curl -sSL -o "$BIG/$tile.LAZ" "https://geotiles.citg.tudelft.nl/AHN4_T/$tile.LAZ"
done
[ -f "$BIG/25GN1_01.info" ] || true
"$ORACLE" info --metadata "$BIG/25GN1_01.LAZ" 2>/dev/null | grep -E '"count"|"minx"|"maxx"|"miny"|"maxy"|dataformat_id|"minor_version"' | head -8 > "$BIG/25GN1_01.info"
# 16M-point subset from the local corpus layout is uploaded as prefix-16m.las when present.
run_big() { # label runs pipeline fixture-laz [extra...]
  local label=$1 runs=$2 pipeline=$3 laz=$4; shift 4
  python3 scripts/pdg/benchmark_reference.py \
    --oracle "$ORACLE" --candidate "$PDG" \
    --pipeline "$pipeline" --fixture-laz "$laz" \
    --label "$label" --work-dir "$OUT/big-work/$label" --report "$OUT/big-$label.json" \
    --runs "$runs" --warmups 0 --cache-state warm \
    --frozen-time-library "$FROZEN" --freeze-epoch 1704067200 "$@" \
    > "$OUT/big-$label.log" 2>&1
  tail -1 "$OUT/big-$label.log"
  # keep the work directory small: only the last artifacts survive anyway
}
BIGLAZ=$BIG/25GN1_01.LAZ
# 47.5M-point tile: pinned PDAL alone needs minutes per run on the heavy
# workloads, so the default configuration measures two runs and the
# experimental-CUDA configuration one run on the CUDA-relevant graphs.
for wl in r14-laz2las r1-translate r10-decimate r7-dsm r3-dtm r2-ground-normalize r4-denoise-thin r6-features; do
  run_big "ahn4-$wl-default" "$BIG_RUNS" bench/pipelines/reference/$wl.json "$BIGLAZ" --fixture input.laz="$BIGLAZ"
done
for wl in r2-ground-normalize r4-denoise-thin r6-features; do
  run_big "ahn4-$wl-cuda-experimental" 1 bench/pipelines/reference/$wl.json "$BIGLAZ" --fixture input.laz="$BIGLAZ" --candidate-env PDG_EXPERIMENTAL_CUDA_HYBRID=1
done
echo RUN-DONE
