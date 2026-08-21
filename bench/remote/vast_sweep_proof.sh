#!/usr/bin/env bash
# Proof phase of the shipped-profile sweep (D0279/B0278): with the shipped
# GPU-class profiles (and the generic fallback) embedded in the build and NO
# local profile, run the fourteen workflows in default mode and, on big cards,
# the AHN4 tile — the "drop-in" measurement: nothing calibrated on this box.
#   vast_sweep_proof.sh <fixtures-dir> <results-dir> <profiles-dir> [exclude-slug] [suite-runs]
# `exclude-slug` removes that shipped profile from the embedded set before the
# build so the box exercises the generic tier instead of its own profile.
set -uo pipefail
FIX=${1:?fixtures dir}
OUT=${2:?results dir}
PROFILES=${3:?profiles dir}
EXCLUDE=${4:-}
SUITE_RUNS=${5:-3}
ROOT=/workspace
REPO=$ROOT/pdal-gpu
ORACLE=$ROOT/oracle/bin/pdal
PDG=$REPO/build/pdg-cuda-release/bin/pdg
FROZEN=$REPO/build/pdg-cuda-release/lib/libpdg_frozen_time.so
PROFILE=${XDG_CONFIG_HOME:-$HOME/.config}/pdg/placement-profile.json
TAG=${EXCLUDE:+generic}
TAG=${TAG:-shipped}
mkdir -p "$OUT"
cd "$REPO"
mkdir -p data/placement-profiles
rm -f data/placement-profiles/*.json
cp "$PROFILES"/*.json data/placement-profiles/ || { echo PROFILES-MISSING; exit 1; }
[ "$(ls data/placement-profiles/*.json 2>/dev/null | wc -l)" -ge 1 ] || { echo PROFILES-MISSING; exit 1; }
[ -n "$EXCLUDE" ] && rm -f "data/placement-profiles/$EXCLUDE.json"
ls data/placement-profiles/ > "$OUT/embedded-profiles-$TAG.txt"
# Source patches uploaded alongside the profiles are applied first.
if [ -d "$PROFILES/src" ]; then (cd "$PROFILES/src" && find . -type f | while read -r f; do cp "$f" "$REPO/$f"; done); fi
cmake build/pdg-cuda-release > "$OUT/proof-$TAG-configure.log" 2>&1
cmake --build build/pdg-cuda-release --target pdg_engine pdg_cli pdal -j 8 > "$OUT/proof-$TAG-rebuild.log" 2>&1 || { echo REBUILD-FAILED; exit 1; }
sha256sum build/pdg-cuda-release/bin/pdg-engine >> "$OUT/machine.txt"
mv "$PROFILE" "$OUT/placement-profile-local-removed.json" 2>/dev/null || true
"$PDG" calibrate --status > "$OUT/status-$TAG.txt" 2>&1
"$PDG" doctor > "$OUT/doctor-$TAG.txt" 2>&1
grep -E 'active_profile' "$OUT/status-$TAG.txt"
grep -q "active_profile_tier: $TAG" "$OUT/status-$TAG.txt" || { echo "TIER-NOT-ACTIVE-$TAG"; exit 1; }
# Keep the previous invalid attempt (profiles were not embedded) out of the way.
for f in "$OUT/suite-1m-default-$TAG.json" "$OUT/suite-1m-default-$TAG.log"; do [ -f "$f" ] && mv "$f" "$f.noprofiles"; done
rm -rf "$OUT/suite-1m-default-$TAG-reports" "$OUT/suite-1m-default-$TAG-work"
for f in "$OUT"/big-ahn4-*-default-$TAG.json "$OUT"/big-ahn4-*-default-$TAG.log; do [ -f "$f" ] && mv "$f" "$f.noprofiles"; done
suite() {  # name extra-args...
  local name=$1; shift 1
  python3 scripts/pdg/reference_suite.py run \
    --manifest bench/pipelines/reference/manifest.json --repo-root . \
    --benchmark-runner scripts/pdg/benchmark_reference.py \
    --oracle "$ORACLE" --candidate "$PDG" \
    --work-dir "$OUT/$name-work" --reports "$OUT/$name-reports" \
    --aggregate-output "$OUT/$name.json" \
    --cache-state warm --runs "$SUITE_RUNS" --warmups 1 \
    --frozen-time-library "$FROZEN" --freeze-epoch 1704067200 "$@" \
    > "$OUT/$name.log" 2>&1
  tail -3 "$OUT/$name.log"
}
suite "suite-1m-default-$TAG"
BIG=$OUT/big
if [ -f "$BIG/25GN1_01.LAZ" ] && [ "${SKIP_AHN4:-0}" != 1 ]; then
  run_big() { # label pipeline laz [extra...]
    local label=$1 pipeline=$2 laz=$3; shift 3
    python3 scripts/pdg/benchmark_reference.py \
      --oracle "$ORACLE" --candidate "$PDG" \
      --pipeline "$pipeline" --fixture-laz "$laz" \
      --label "$label" --work-dir "$OUT/big-work/$label" --report "$OUT/big-$label.json" \
      --runs 1 --warmups 0 --cache-state warm \
      --frozen-time-library "$FROZEN" --freeze-epoch 1704067200 "$@" \
      > "$OUT/big-$label.log" 2>&1
    tail -1 "$OUT/big-$label.log"
    rm -rf "$OUT/big-work/$label"
  }
  run_big "ahn4-features-plain-default-$TAG" "$OUT/features-plain.json" "$BIG/25GN1_01.LAZ" --fixture input.laz="$BIG/25GN1_01.LAZ"
  run_big "ahn4-r6-features-default-$TAG" bench/pipelines/reference/r6-features.json "$BIG/25GN1_01.LAZ" --fixture input.laz="$BIG/25GN1_01.LAZ"
fi
rm -rf "$OUT"/*-work
tar -czf "$ROOT/results-sweep.tar.gz" -C "$ROOT" "$(basename "$OUT")"
echo PROOF-$TAG-DONE
