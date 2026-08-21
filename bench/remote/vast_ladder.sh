#!/usr/bin/env bash
# B0276 scale ladder on a big GPU: AHN4 GeoTiles merged to ~47M / ~95M /
# ~190M points; pinned oracle vs pdg (host default for I/O graphs, CUDA
# hybrid forced for the kNN graphs; host default too at 47M for reference).
#   vast_ladder.sh <results-dir> [runs]
set -uo pipefail
OUT=${1:?results dir}; RUNS=${2:-1}
ROOT=/workspace; REPO=$ROOT/pdal-gpu
ORACLE=$ROOT/oracle/bin/pdal; PDG=$REPO/build/pdg-cuda-release/bin/pdg
FROZEN=$REPO/build/pdg-cuda-release/lib/libpdg_frozen_time.so
mkdir -p "$OUT/big"; cd "$REPO"
{
  echo "host: $(hostname)"; date -u
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'; cat /sys/fs/cgroup/cpu.max 2>/dev/null
  free -g | head -2; nproc; "$ORACLE" --version | head -1
  sha256sum "$ORACLE" "$PDG" $REPO/build/pdg-cuda-release/bin/pdg-engine $REPO/build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0
} > "$OUT/machine.txt" 2>&1
BIG=$OUT/big
for t in 25GN1_01 25GN1_02 25GN1_03 25GN1_04; do
  [ -f "$BIG/$t.LAZ" ] || curl -sSL -o "$BIG/$t.LAZ" "https://geotiles.citg.tudelft.nl/AHN4_T/$t.LAZ"
done
# Merged inputs (built once with the box's oracle; a valid LAZ is all the runner needs).
merge() { local out=$1; shift; [ -f "$out" ] && return
  local stages="" ; local i=0
  for f in "$@"; do i=$((i+1)); stages="$stages{\"type\":\"readers.las\",\"filename\":\"$f\",\"tag\":\"in$i\"},"; done
  echo "{\"pipeline\":[$stages{\"type\":\"filters.merge\"},{\"type\":\"writers.las\",\"filename\":\"$out\",\"compression\":true,\"minor_version\":4,\"dataformat_id\":8}]}" > "$OUT/merge.json"
  "$PDG" pipeline "$OUT/merge.json" >/dev/null 2>&1 || "$ORACLE" pipeline "$OUT/merge.json"
}
merge "$BIG/ahn4-2tiles.laz" "$BIG/25GN1_01.LAZ" "$BIG/25GN1_02.LAZ"
merge "$BIG/ahn4-4tiles.laz" "$BIG/25GN1_01.LAZ" "$BIG/25GN1_02.LAZ" "$BIG/25GN1_03.LAZ" "$BIG/25GN1_04.LAZ"
for f in 25GN1_01.LAZ ahn4-2tiles.laz ahn4-4tiles.laz; do
  echo "$f $("$ORACLE" info --metadata "$BIG/$f" 2>/dev/null | grep -E '"count"' | head -1) $(stat -c %s "$BIG/$f")" >> "$OUT/inputs.txt"
done
run() { # label pipeline laz [extra...]
  local label=$1 pipeline=$2 laz=$3; shift 3
  python3 scripts/pdg/benchmark_reference.py --oracle "$ORACLE" --candidate "$PDG" \
    --pipeline "$pipeline" --fixture-laz "$laz" --fixture input.laz="$laz" \
    --label "$label" --work-dir "$OUT/work/$label" --report "$OUT/$label.json" \
    --runs "$RUNS" --warmups 0 --cache-state warm \
    --frozen-time-library "$FROZEN" --freeze-epoch 1704067200 "$@" > "$OUT/$label.log" 2>&1
  tail -1 "$OUT/$label.log"; rm -rf "$OUT/work/$label"
}
P=bench/pipelines/reference
for size in 47m 95m 190m; do
  case $size in 47m) laz=$BIG/25GN1_01.LAZ;; 95m) laz=$BIG/ahn4-2tiles.laz;; 190m) laz=$BIG/ahn4-4tiles.laz;; esac
  run "ladder-$size-r14-laz2las-default" $P/r14-laz2las.json "$laz"
  run "ladder-$size-r1-translate-default" $P/r1-translate.json "$laz"
  run "ladder-$size-r3-dtm-default" $P/r3-dtm.json "$laz"
  run "ladder-$size-r6-features-cuda" $P/r6-features.json "$laz" --candidate-env PDG_EXPERIMENTAL_CUDA_HYBRID=1
  run "ladder-$size-r4-denoise-thin-cuda" $P/r4-denoise-thin.json "$laz" --candidate-env PDG_EXPERIMENTAL_CUDA_HYBRID=1
  run "ladder-$size-r2-ground-normalize-cuda" $P/r2-ground-normalize.json "$laz" --candidate-env PDG_EXPERIMENTAL_CUDA_HYBRID=1
  if [ $size = 47m ]; then
    run "ladder-$size-r6-features-default" $P/r6-features.json "$laz"
    run "ladder-$size-r4-denoise-thin-default" $P/r4-denoise-thin.json "$laz"
  fi
done
cd $ROOT && tar -czf $ROOT/results-b0276.tar.gz -C $ROOT results-ladder/*.json results-ladder/*.log results-ladder/machine.txt results-ladder/inputs.txt 2>/dev/null
echo LADDER-DONE
