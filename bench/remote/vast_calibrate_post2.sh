#!/usr/bin/env bash
# B0277 follow-up 2 on a box that already ran vast_calibrate.sh and
# vast_calibrate_post.sh: rebuild the engine with the D0278 change (the
# resident layout follows the file's extra-bytes dimension types for
# dimensions no device stage touches) and re-run the AHN4 default-mode cells,
# which previously fell back to the host in the resident preflight.
#   vast_calibrate_post2.sh <fixtures-dir> <results-dir>
set -uo pipefail
FIX=${1:?fixtures dir}
OUT=${2:?results dir}
ROOT=/workspace
REPO=$ROOT/pdal-gpu
ORACLE=$ROOT/oracle/bin/pdal
PDG=$REPO/build/pdg-cuda-release/bin/pdg
FROZEN=$REPO/build/pdg-cuda-release/lib/libpdg_frozen_time.so
cd "$REPO"
cp /workspace/uploads/Dimension.hpp include/pdg/Dimension.hpp
cp /workspace/uploads/dimension.cpp src/core/dimension.cpp
cp /workspace/uploads/PdgResidentContext.cpp src/pdal/PdgResidentContext.cpp
cp /workspace/uploads/Calibrate.cpp src/cli/Calibrate.cpp
cmake --build build/pdg-cuda-release --target pdg_engine pdg_cli pdal -j 8 > "$OUT/post2-rebuild.log" 2>&1 || { echo REBUILD2-FAILED; exit 1; }
sha256sum build/pdg-cuda-release/bin/pdg-engine build/pdg-cuda-release/bin/pdg >> "$OUT/machine.txt"
"$PDG" calibrate --status > "$OUT/status-post2.txt" 2>&1
BIG=$OUT/big
BIGLAZ=$BIG/25GN1_01.LAZ
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
for f in features-plain lof-plain r6-features; do
  for e in json log; do mv "$OUT/big-ahn4-$f-default.$e" "$OUT/big-ahn4-$f-default-prefix.$e" 2>/dev/null; done
done
run_big ahn4-features-plain-default "$OUT/features-plain.json" "$BIGLAZ" --fixture input.laz="$BIGLAZ"
run_big ahn4-lof-plain-default "$OUT/lof-plain.json" "$BIGLAZ" --fixture input.laz="$BIGLAZ"
run_big ahn4-r6-features-default bench/pipelines/reference/r6-features.json "$BIGLAZ" --fixture input.laz="$BIGLAZ"
rm -rf "$OUT"/*-work
tar -czf "$ROOT/results-b0277.tar.gz" -C "$ROOT" "$(basename "$OUT")"
echo POST2-DONE
