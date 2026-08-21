#!/usr/bin/env bash
# Second half of the B0275 remote run: waits for vast_run.sh, fills the
# workloads that failed on missing CLIs (gdalinfo/cs2cs) and aggregates the
# three 1M suites, runs the LAStools comparison, renders the big-tile
# outputs, and packs everything for download.
set -uo pipefail
ROOT=/workspace; REPO=$ROOT/pdal-gpu; OUT=$ROOT/results; FIX=$ROOT/fixtures
ORACLE=$ROOT/oracle/bin/pdal; PDG=$REPO/build/pdg-cuda-release/bin/pdg
FROZEN=$REPO/build/pdg-cuda-release/lib/libpdg_frozen_time.so
until grep -q RUN-DONE $ROOT/run.log; do sleep 30; done
cd $REPO
suite_fill() { local name=$1; shift
  python3 scripts/pdg/reference_suite.py run \
    --manifest bench/pipelines/reference/manifest.json --repo-root . \
    --benchmark-runner scripts/pdg/benchmark_reference.py \
    --oracle "$ORACLE" --candidate "$PDG" \
    --work-dir "$OUT/$name-work" --reports "$OUT/$name-reports" \
    --aggregate-output "$OUT/$name.json" --cache-state warm --runs 3 --warmups 1 \
    --frozen-time-library "$FROZEN" --freeze-epoch 1704067200 \
    --workload r3-dtm --workload r7-dsm --workload r9-polygon-clip "$@" >> "$OUT/$name.log" 2>&1
  python3 scripts/pdg/reference_suite.py aggregate --manifest bench/pipelines/reference/manifest.json \
    --reports "$OUT/$name-reports" --cache-state warm --output "$OUT/$name.json" ${CONTRACT:-} >> "$OUT/$name.log" 2>&1
  tail -1 "$OUT/$name.log"
}
suite_fill suite-1m-default
suite_fill suite-1m-cuda-experimental --candidate-env PDG_EXPERIMENTAL_CUDA_HYBRID=1
CONTRACT="--contract fast" suite_fill suite-1m-fast-cuda-experimental --contract fast --candidate-arg=--fast --candidate-env PDG_EXPERIMENTAL_CUDA_HYBRID=1
echo SUITES-FILLED

# LAStools: open-source tools from source, proprietary tools under wine (-demo).
mkdir -p $ROOT/lastools && cd $ROOT/lastools
[ -d LAStools ] || git clone -q --depth 1 https://github.com/LAStools/LAStools.git
[ -x LAStools/bin64/laszip64 ] || (cd LAStools && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja >/dev/null 2>&1 && ninja -j16 >/dev/null 2>&1)
[ -f LAStools.zip ] || curl -sSL -o LAStools.zip https://downloads.rapidlasso.de/LAStools.zip
[ -d zip ] || (mkdir -p zip && cd zip && unzip -q ../LAStools.zip 'bin/*' >/dev/null 2>&1)
cp -rn zip/bin/serf LAStools/bin64/ 2>/dev/null || true
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq wine64 >/tmp/wine-apt.log 2>&1 || true
WINE=$(command -v wine64 || command -v wine || echo wine)
cd $REPO
python3 bench/lastools/lastools_bench.py --oracle "$ORACLE" --pdg "$PDG" \
  --lastools-bin $ROOT/lastools/LAStools/bin64 --lastools-exe $ROOT/lastools/zip/bin --wine "$WINE" \
  --laz $FIX/ref-1m.laz --las $FIX/ref-1m.las --merge-a $FIX/ref-merge-a.laz --merge-b $FIX/ref-merge-b.laz \
  --work-dir $ROOT/lastools-work --report $OUT/lastools-box.json --runs 3 --frozen-time-library "$FROZEN" \
  > $OUT/lastools-box.log 2>&1
tail -3 $OUT/lastools-box.log
# Big-tile renders from the last big-run candidate artifacts.
python3 -m venv $ROOT/venv >/dev/null 2>&1; $ROOT/venv/bin/pip install -q matplotlib numpy >/dev/null 2>&1
mkdir -p $OUT/renders
R="$ROOT/venv/bin/python bench/report/render_outputs.py --pdal $ORACLE"
A=$OUT/big-work
$R --kind raster --input $A/ahn4-r3-dtm-default/ahn4-r3-dtm-default-candidate-artifacts/output.tif --output $OUT/renders/ahn4-r3-dtm-hillshade.png --title "AHN4 25GN1_01 (47.5M points): r3 DTM 1 m, SMRF ground, hillshade" 2>&1 | tail -1
$R --kind raster --input $A/ahn4-r7-dsm-default/ahn4-r7-dsm-default-candidate-artifacts/output.tif --output $OUT/renders/ahn4-r7-dsm-hillshade.png --title "AHN4 25GN1_01: r7 DSM 1 m, first/only returns max Z, hillshade" 2>&1 | tail -1
$R --kind cloud --input $A/ahn4-r2-ground-normalize-default/ahn4-r2-ground-normalize-default-candidate-artifacts/output.laz --output $OUT/renders/ahn4-r2-hag.png --mode hag --step 40 --title "AHN4 25GN1_01: r2 height above ground (every 40th point)" 2>&1 | tail -1
$R --kind cloud --input $A/ahn4-r10-decimate-default/ahn4-r10-decimate-default-candidate-artifacts/output.laz --output $OUT/renders/ahn4-r10-decimated.png --mode z --step 8 --title "AHN4 25GN1_01: r10 voxel decimation output, Z (every 8th kept point)" 2>&1 | tail -1
$R --kind cloud --input $ROOT/results/big/25GN1_01.LAZ --output $OUT/renders/ahn4-input-rgb.png --mode rgb --step 40 --title "AHN4 25GN1_01 input: RGB (every 40th point)" 2>&1 | tail -1
cd $ROOT && tar -czf $ROOT/results-b0275.tar.gz -C $ROOT results/*.json results/*.log results/*-reports results/renders results/machine.txt results/big/*.info 2>/dev/null
ls -la $ROOT/results-b0275.tar.gz
echo POST-DONE
