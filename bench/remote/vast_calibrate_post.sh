#!/usr/bin/env bash
# B0277 follow-up on a box that already ran vast_calibrate.sh: after the
# --append merge fix (prior sizes are refitted together with the new ones
# instead of being replaced), rebuild the engine from the patched source,
# extend the profile at 250K/1M for the appended models plus the two
# point-program models, and re-run the calibrated default suite.
#   vast_calibrate_post.sh <fixtures-dir> <results-dir> [suite-runs]
set -uo pipefail
FIX=${1:?fixtures dir}
OUT=${2:?results dir}
SUITE_RUNS=${3:-3}
ROOT=/workspace
REPO=$ROOT/pdal-gpu
ORACLE=$ROOT/oracle/bin/pdal
PDG=$REPO/build/pdg-cuda-release/bin/pdg
FROZEN=$REPO/build/pdg-cuda-release/lib/libpdg_frozen_time.so
PROFILE=${XDG_CONFIG_HOME:-$HOME/.config}/pdg/placement-profile.json
cd "$REPO"
cp /workspace/uploads/Calibrate.cpp src/cli/Calibrate.cpp
cmake --build build/pdg-cuda-release --target pdg_engine -j 8 > "$OUT/post-rebuild.log" 2>&1 || { echo REBUILD-FAILED; exit 1; }
sha256sum build/pdg-cuda-release/bin/pdg-engine >> "$OUT/machine.txt"
cp "$PROFILE" "$OUT/placement-profile-before-post.json"
start=$(date +%s)
"$PDG" calibrate --append --models normal-covariancefeatures-compose,lof,nndistance,fused-point-program,simple-ferry \
  --points 250000,1000000 --repeats 1 --work "$OUT/calibrate-post-work" \
  > "$OUT/calibrate-post.log" 2> "$OUT/calibrate-post.stderr"
echo "Elapsed (wall clock) time: $(( $(date +%s) - start )) s" > "$OUT/calibrate-post.time"
cp "$PROFILE" "$OUT/placement-profile.json"
"$PDG" calibrate --status > "$OUT/status-after.txt" 2>&1
"$PDG" doctor > "$OUT/doctor-after.txt" 2>&1
tail -12 "$OUT/calibrate-post.log"
for f in json log; do mv "$OUT/suite-1m-default-calibrated.$f" "$OUT/suite-1m-default-calibrated-appendbug.$f" 2>/dev/null; done
mv "$OUT/suite-1m-default-calibrated-reports" "$OUT/suite-1m-default-calibrated-appendbug-reports" 2>/dev/null
rm -rf "$OUT/suite-1m-default-calibrated-work"
python3 scripts/pdg/reference_suite.py run \
  --manifest bench/pipelines/reference/manifest.json --repo-root . \
  --benchmark-runner scripts/pdg/benchmark_reference.py \
  --oracle "$ORACLE" --candidate "$PDG" \
  --work-dir "$OUT/suite-1m-default-calibrated-work" --reports "$OUT/suite-1m-default-calibrated-reports" \
  --aggregate-output "$OUT/suite-1m-default-calibrated.json" \
  --cache-state warm --runs "$SUITE_RUNS" --warmups 1 \
  --frozen-time-library "$FROZEN" --freeze-epoch 1704067200 \
  > "$OUT/suite-1m-default-calibrated.log" 2>&1
tail -3 "$OUT/suite-1m-default-calibrated.log"
# The AHN4 tile carries extra-bytes dimensions (Amplitude, Reflectance,
# Deviation) whose declared LAS types differ from PDAL's standard dimension
# types; the resident executor's preflight refuses that layout, so default
# mode stayed on the host for the tile above. The same graphs with the
# extra-bytes VLR ignored are the shape a tile without such EB dims presents.
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
# 1.4 files always apply their EB VLR, so the EB-free variant is materialised
# once with the pinned oracle (its default writer keeps only the standard
# dimensions of the point format).
NOEB=$BIG/25GN1_01-noeb.laz
[ -f "$NOEB" ] || "$ORACLE" translate "$BIGLAZ" "$NOEB" --writers.las.forward=all > "$OUT/noeb-translate.log" 2>&1
cat > "$OUT/features-noeb.json" <<'JSON'
{"pipeline":[{"type":"readers.las","filename":"input.laz"},{"type":"filters.normal","knn":8},{"type":"filters.covariancefeatures","knn":8,"feature_set":"Dimensionality"},{"type":"writers.las","filename":"output.las"}]}
JSON
cat > "$OUT/lof-noeb.json" <<'JSON'
{"pipeline":[{"type":"readers.las","filename":"input.laz"},{"type":"filters.lof","minpts":10},{"type":"writers.las","filename":"output.las"}]}
JSON
if [ -f "$NOEB" ]; then
  "$ORACLE" info --metadata "$NOEB" 2>/dev/null | grep -E '"count"|dataformat_id|"minor_version"' | head -3 > "$BIG/25GN1_01-noeb.info"
  run_big ahn4-features-noeb-default "$OUT/features-noeb.json" "$NOEB" --fixture input.laz="$NOEB"
  run_big ahn4-lof-noeb-default "$OUT/lof-noeb.json" "$NOEB" --fixture input.laz="$NOEB"
  run_big ahn4-features-noeb-cuda-experimental "$OUT/features-noeb.json" "$NOEB" --fixture input.laz="$NOEB" --candidate-env PDG_EXPERIMENTAL_CUDA_HYBRID=1
fi
rm -rf "$OUT"/*-work
tar -czf "$ROOT/results-b0277.tar.gz" -C "$ROOT" "$(basename "$OUT")"
echo POST-DONE
