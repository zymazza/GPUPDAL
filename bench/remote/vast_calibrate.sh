#!/usr/bin/env bash
# Local-calibration protocol on a bootstrapped box (B0277/D0277).
#   vast_calibrate.sh <fixtures-dir> <results-dir> [suite-runs]
# 1. status before (host path expected: no embedded/local profile);
# 2. the fourteen-workflow reference suite in default mode, uncalibrated;
# 3. `pdg calibrate` (default sizes/repeats) into the default profile path;
# 4. status after; the same suite in default mode, now calibrated;
# 5. the same suite with the CUDA hybrid forced (the B0275 upper reference);
# 6. calibration on real data (the 1M reference fixture) into a second file
#    for comparison with the synthetic-cloud profile (not used for the suite);
# 7. AHN4 47M-point r6/r4/lof-class graphs default-mode calibrated (one run).
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
mkdir -p "$OUT"
cd "$REPO"
{
  echo "host: $(hostname)"; date -u
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
  free -g | head -2; nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null
  "$ORACLE" --version | head -1; "$PDG" --version 2>&1 | head -1
  sha256sum "$ORACLE" "$PDG" $REPO/build/pdg-cuda-release/bin/pdg-engine $REPO/build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0
} > "$OUT/machine.txt" 2>&1

suite() {  # name extra-args...
  local name=$1; shift 1
  if [ -s "$OUT/$name.json" ]; then echo "suite $name already recorded"; return; fi
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
mkdir -p build/bench-data/reference
cp -n "$FIX"/ref-* build/bench-data/reference/ 2>/dev/null || true

rm -f "$PROFILE"
"$PDG" doctor > "$OUT/doctor-before.txt" 2>&1
"$PDG" calibrate --status > "$OUT/status-before.txt" 2>&1
suite suite-1m-default-uncalibrated

# Calibrate (synthetic cloud, default sizes and repeats), keep everything.
start=$(date +%s)
"$PDG" calibrate --work "$OUT/calibrate-work" --keep-work \
  > "$OUT/calibrate.log" 2> "$OUT/calibrate.stderr"
echo "Elapsed (wall clock) time: $(( $(date +%s) - start )) s" > "$OUT/calibrate.time"
cp "$PROFILE" "$OUT/placement-profile-standard.json"
tail -25 "$OUT/calibrate.log"
# Extend the three big-cloud models to the AHN4 tile size (one pair each);
# --append keeps the standard models and coefficients.
start=$(date +%s)
"$PDG" calibrate --append --models normal-covariancefeatures-compose,lof,nndistance \
  --points 16000000,48000000 --repeats 1 --work "$OUT/calibrate-big-work" \
  > "$OUT/calibrate-big.log" 2> "$OUT/calibrate-big.stderr"
echo "Elapsed (wall clock) time: $(( $(date +%s) - start )) s" > "$OUT/calibrate-big.time"
cp "$PROFILE" "$OUT/placement-profile.json"
"$PDG" calibrate --status > "$OUT/status-after.txt" 2>&1
"$PDG" doctor > "$OUT/doctor-after.txt" 2>&1
tail -12 "$OUT/calibrate-big.log"

suite suite-1m-default-calibrated
suite suite-1m-cuda-experimental --candidate-env PDG_EXPERIMENTAL_CUDA_HYBRID=1

# Real-data calibration for comparison (does not replace the active profile).
PDG_PROFILE_PATH="$OUT/placement-profile-realdata.json" \
  "$PDG" calibrate --input build/bench-data/reference/ref-1m.laz \
  --points 250000,1000000 --quiet > "$OUT/calibrate-realdata.log" 2>&1
tail -20 "$OUT/calibrate-realdata.log"

# Big cloud, default mode with the calibrated profile active.
BIG=$OUT/big
mkdir -p "$BIG"
[ -f "$BIG/25GN1_01.LAZ" ] || curl -sSL -o "$BIG/25GN1_01.LAZ" "https://geotiles.citg.tudelft.nl/AHN4_T/25GN1_01.LAZ"
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
BIGLAZ=$BIG/25GN1_01.LAZ
# The r6 reference graph writes extra_dims=all, whose automatic admission is
# bounded to the measured 1M layout; the plain-writer feature graph below is
# the shape the calibrated compose model covers, so both are measured.
cat > "$OUT/features-plain.json" <<'JSON'
{"pipeline":[{"type":"readers.las","filename":"input.laz"},{"type":"filters.normal","knn":8},{"type":"filters.covariancefeatures","knn":8,"feature_set":"Dimensionality"},{"type":"writers.las","filename":"output.las"}]}
JSON
cat > "$OUT/lof-plain.json" <<'JSON'
{"pipeline":[{"type":"readers.las","filename":"input.laz"},{"type":"filters.lof","minpts":10},{"type":"writers.las","filename":"output.las"}]}
JSON
run_big ahn4-features-plain-default "$OUT/features-plain.json" "$BIGLAZ" --fixture input.laz="$BIGLAZ"
run_big ahn4-r6-features-default bench/pipelines/reference/r6-features.json "$BIGLAZ" --fixture input.laz="$BIGLAZ"
run_big ahn4-lof-plain-default "$OUT/lof-plain.json" "$BIGLAZ" --fixture input.laz="$BIGLAZ"
run_big ahn4-features-plain-cuda-experimental "$OUT/features-plain.json" "$BIGLAZ" --fixture input.laz="$BIGLAZ" --candidate-env PDG_EXPERIMENTAL_CUDA_HYBRID=1
rm -rf "$OUT"/*-work
tar -czf "$ROOT/results-b0277.tar.gz" -C "$ROOT" "$(basename "$OUT")"
echo CALIBRATE-DONE
