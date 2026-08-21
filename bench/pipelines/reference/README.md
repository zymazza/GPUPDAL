# Reference pipelines — the speed target (D0208, D0239)

These fourteen equal-weight end-to-end workflows are the product's release
criterion. Each is measured as a complete process, cold and warm cache,
against the pinned PDAL oracle on identical hardware. Wall-clock time here —
not native CUDA stage count — defines progress. The canonical membership,
fixture hashes, materialization policies, exactness requirements, supporting
variants, and aggregate formulas live in `manifest.json`.

The single-stage pipelines in the parent directory remain useful for
attributing cost inside a workflow, but they are not the goal.

| Id | Shape | Exercises |
| --- | --- | --- |
| `r1-translate` | LAZ -> reproject -> crop -> LAZ | decode/encode dominated; the most common PDAL operation |
| `r2-ground-normalize` | LAZ -> SMRF -> HAG -> LAZ | standard lidar normalization; CUDA currently loses on SMRF |
| `r3-dtm` | LAZ -> SMRF -> ground -> GDAL raster | DEM/DSM production; raster assembly is unaccelerated today |
| `r4-denoise-thin` | LAZ -> outlier -> range -> sample -> LAZ | cleanup/reduction over an already-fast neighborhood path |
| `r5-copc-query` | COPC bounds+resolution query -> stats | cloud-native access, range reads, pushdown |
| `r6-features` | LAZ -> normal + covariance features -> LAZ | ML feature extraction; where the shared kNN gather pays |
| `r7-dsm` | first/only returns -> maximum-Z GeoTIFF | explicit one-metre DSM surface policy and raster publication |
| `r8-colorize` | 28992 points -> 3857 RGB raster sample -> 28992 LAZ | orthophoto colorization, CRS interaction, GDAL cache, and uncovered points |
| `r9-polygon-clip` | EPSG:4326 multipolygon/hole -> reprojected crop -> LAZ | standalone production AOI clipping, holes, boundaries, geometry CRS, and explicit longitude/latitude WKT axis order |
| `r10-decimate` | voxel-centroid nearest original record -> LAZ | spatially uniform 3D reduction while retaining complete source records |
| `r11-classify-refine` | SMRF -> statistical outlier -> neighbor vote -> LAZ | realistic built-in classification/refinement without an external model |
| `r12-tile` | fixed-origin splitter -> seven named LAZ outputs | unique boundary ownership, deterministic naming/order, and publication |
| `r13-merge` | heterogeneous LAS 1.4/format 7 + LAS 1.2/format 3 -> LAZ | multi-input layout/header/order assembly and mosaic compression |
| `r14-convert-compress` | LAS -> LAZ | compression-only throughput; companions cover LAZ/LAS/COPC directions |

`REPLACE_*` bounds placeholders are substituted by the harness from the
fixture's actual extent so the crop and query select a meaningful subset
rather than everything or nothing.

Fixtures are built from the retained uncompressed LAS bench data by
recompressing with the pinned oracle, so no new source data is introduced and
the existing hash-pinned inputs remain authoritative. The local fixture recipe
also creates the deterministic EPSG:3857 RGB raster and heterogeneous merge
pair without modifying the source corpus. It regenerates the input COPC with
the pinned oracle, frozen time, `threads=1`, and `fixed_seed=true`; that makes
the fixture byte-stable, while workflow-produced COPC containers continue to
use canonical semantic comparison:

```sh
python3 scripts/pdg/prepare_reference_fixtures.py \
  --oracle build/pdal-upstream-tests/bin/pdal \
  --source-laz build/bench-data/reference/ref-1m.laz \
  --output-dir build/bench-data/reference \
  --frozen-time-library build/pdg-cuda-release/lib/libpdg_frozen_time.so
```

Validate membership without executing data:

```sh
python3 scripts/pdg/reference_suite.py validate \
  --manifest bench/pipelines/reference/manifest.json --repo-root .
```

`reference_suite.py run` verifies every fixture byte count/hash, invokes the
complete-process benchmark for each headline, and writes an aggregate only if
all fourteen are exact. Use `--cache-state warm` and `--cache-state cold` as
separate invocations. Cold mode uses file-scoped `POSIX_FADV_DONTNEED` before
every process and records that method. `--variants-only` executes the
zero-weight r10/r14 companion matrix without changing headline membership.
Materialization refuses output aliases to registered input fixtures, output
paths outside the role-specific artifact directory, and duplicate artifact
patterns before either measured process starts.

The aggregate reports both the equal-workload geometric mean and the ratio of
summed median walls for one execution of every headline. See
`OPPORTUNITY.md` for the current measured priority ledger.

Current r14 result: B0251/D0250 extend B0250's all-direction matrix to 13
cases/21 executions and select exact two-worker lazperf compression only for
the literal 1M LAS -> LAZ headline layout. The production primitive reproduces
the full sequential payload/table and measures 1.735238x. B0252/D0251 correct
the fail-closed environment boundary and keep the activation hook test-only;
corrected final public results are 1.515827x warm and 1.504457x cold with 9/9
wins and exact bytes, metadata, order, streams, and status. Every
grammar/layout/count/control drift and all six companion directions remain
serial host-selected. Current single-binary suite claims are
1.262824x/1.260608x warm/cold equal-workload geometric mean and
1.641092x/1.634768x total-wall speedup. The next measured slice tests
B0253/D0252 confirm that the reader already performs ordered chunk-parallel
decode, reject 10/12 workers as slower on r7/r10, and reject a Point14 setup
cleanup whose exact 1.014837x primitive gain leaves public r10 unresolved at
1.012374x warm. The product reader remains unchanged. The next measured slice
uses the retained formats-6--8 production-factory test/benchmark to evaluate
one exact lazperf integer-decode hot-loop improvement before any public gate.

B0254/D0253 reject that forced-inline hot-loop experiment at only 1.010245x,
then complete the worker screen. Four workers are retained only for the exact
plain 1M compressed format-7/36-byte r7 DSM headline; final 21-pair public
results are 1.031858x warm and 1.020398x cold with complete TIFF metadata/
bytes, streams, status, and refusals exact. r10 remains default-reader because
its integrated cold interval is unresolved. B0254 single-binary suite claims
are 1.267318x/1.262102x warm/cold equal-workload geometric mean and
1.642633x/1.632910x total-wall speedup. The next slice returns to measured r11
classification/refinement attribution; no external model dependency is added.

B0256/D0255 correct and complete that first r11 reuse slice. Prefix
measurements show the two neighborhood consumers dominate. B0255's initial
cache adoption is superseded because a generic XYZ assignment can leave an
incoming stale product. Statistical outlier now invalidates incoming products,
builds its unchanged fresh exact nanoflann tree through the PointView cache,
and lets the adjacent upstream neighbor classifier reuse it. Corrected final
nine-pair public results are 1.029892x warm and 1.037311x cold with exact bytes,
metadata, order, streams, and status. The route and CUDA/native coverage do not
change. B0256 single-binary suite claims were 1.267705x/1.278088x warm/cold
equal-workload geometric mean and 1.656538x/1.665729x total-wall speedup. The
next measured r11 hypothesis was an allocation-free vote tally bounded to the literal
Classification/k=7 semantics; generic shapes remain unchanged.

B0257 measured that tally at about 0.1% of process wall with a checked-in
attribution harness and rejected it; the same harness attributed about
5.55 s of the 6.8 s process to two serial per-point kNN passes. B0258/D0257
run those passes on exact fixed-chunk host workers over the shared read-only
nanoflann index. Final nine-pair public results are **4.309167x warm and
4.345954x cold** with exact bytes, metadata, order, streams, and status; the
same-final-binary serial control is 1.035408x. Current single-binary suite
claims are **1.401126x/1.410744x** warm/cold equal-workload geometric mean and
**2.407033x/2.408896x** total-wall speedup. Route and CUDA/native coverage do
not change; forcing the CUDA statistical outlier alone on r11 is exact but
slower (4.069126x).

B0259/D0258 then attack the next measured limiter, r6's serial LAZ
compression, with the general mechanism rather than another bounded selector:
`writers.las` now defaults to four exact chunk-compression workers, measured
on every headline through a same-binary engine probe first. Every LAZ-writing
workload gains (r6 6.316372x/6.387622x, r13 1.437716x/1.434382x, r1
1.081819x/1.101200x warm/cold over nine pairs; r9, r12, r8, r2, r11 also
improve), non-LAZ workloads are unchanged, and all outputs stay exact.
B0259 single-binary suite claims were 1.597555x/1.589241x warm/cold
equal-workload geometric mean and 2.792319x/2.772361x total-wall speedup.

B0260/B0261 (D0259/D0260) then found, pinned by a new pinned-oracle
differential lane, that B0256's published outlier tree was byte-inexact
against pinned PDAL after a coordinate mutator, and restored pinned private
semantics on the exact cached-coordinate backing. r11 is **5.207843x warm /
5.267489x cold** over nine pairs. Current single-binary suite claims are
**1.592353x/1.595639x** warm/cold equal-workload geometric mean and
**2.794157x/2.794157x** total-wall speedup, all exact.

B0262/D0262 then cut r6's exact eigen tie repair from 0.525 s to 0.142 s
with a planner-proved terminal-sink marker that lets the repair keep a private
cached-coordinate tree: r6 **8.648309x warm / 8.664582x cold** over nine
pairs. Current single-binary suite claims are **1.633482x/1.624540x**
warm/cold equal-workload geometric mean and **2.924049x/2.894853x** total-wall
speedup, all exact.

B0263/D0263 make the exact cached-coordinate KD3 backing the published
default with a coordinate epoch that refreshes reused snapshots (pinned
stale-tree/live-coordinate semantics preserved bit-for-bit): r11 **7.372457x
warm / 7.286586x cold** over nine pairs. Current single-binary suite claims
are **1.672490x/1.663704x** warm/cold equal-workload geometric mean and
**3.073590x/3.042106x** total-wall speedup, all exact.

B0264/D0264 run the exact host SMRF's diamond morphology, void fills, and
point passes on fixed-chunk workers (0.34 s -> 0.17 s at 1M): r3 **1.352415x
warm / 1.338430x cold** (from parity) and r11 **8.867727x / 8.863616x** over
nine pairs. Current single-binary suite claims are **1.731986x/1.729459x**
warm/cold equal-workload geometric mean and **3.185933x/3.147773x** total-wall
speedup, all exact.

B0265/D0265 pool the fork SMRF port's morphology for r2: **1.624190x warm /
1.604373x cold**. Current single-binary suite claims are
**1.750757x/1.737266x** warm/cold equal-workload geometric mean and
**3.268150x/3.192511x** total-wall speedup, all exact.

B0266/D0266 make `filters.reprojection` exact and parallel in streaming and
standard modes (a `Streamable::processStreamBatch` hook and a fixed-slot pool
with one cloned GDAL transformation per slot; scratch-then-commit with a
serial fallback whenever any row fails so every diagnostic and GDAL's
per-object error accounting stay pinned): r8 **2.104750x warm / 2.093405x
cold** (from 1.10x) and r1 **1.487025x / 1.458954x** (from 1.08x). Current
single-binary suite claims are **1.870298x/1.869920x** warm/cold
equal-workload geometric mean and **3.573691x/3.535527x** total-wall speedup,
all exact.

B0267/D0267 hash the `filters.sample` voxel table and prune its probes: r4
**5.306974x warm / 5.271122x cold** (from 3.80x). Current single-binary suite
claims are **1.902538x/1.894000x** warm/cold equal-workload geometric mean
and **3.697978x/3.657649x** total-wall speedup, all exact.

B0268/D0268 pack `writers.las` point records on a fixed-slot pool in both
streaming and standard modes (slot-ordered `las::Summary` merges; any
deferred per-point warning or exception repeats the run serially so streams
and diagnostics stay pinned): r14 **2.084120x warm / 2.048271x cold** (from
1.52x), r13 **1.949116x / 1.963703x** (from 1.44x), r12 **1.456797x /
1.459860x** (from 1.20x), r6 **9.136681x / 9.095218x** (from 8.65x), r1
**1.573685x / 1.580186x** (from 1.49x). Current single-binary suite claims
are **2.133083x/2.122241x** warm/cold equal-workload geometric mean and
**4.063740x/4.028901x** total-wall speedup, all exact.

B0269/D0269 unpack `readers.las` records on a fixed-slot pool in both
streaming (a `Streamable::readStreamBatch` reader hook) and standard
(per-tile) modes, consuming tiles in the pinned order: r14 **2.656026x warm /
2.614287x cold**, r13 **2.682353x / 2.700326x**, r7 **1.320508x / 1.307258x**
(from parity), r9 **1.901385x / 1.911693x**, r1 **2.114222x / 2.107619x**,
r12 **1.946695x / 1.902648x**, r10 **1.446846x / 1.423698x**, r3 **1.586718x /
1.579299x**, r6 **10.122291x / 9.872648x**. Current single-binary suite
claims are **2.558472x/2.555054x** warm/cold equal-workload geometric mean
and **4.662196x/4.655083x** total-wall speedup, all exact; all twelve
zero-weight variants are above parity.

B0270/D0270 decode COPC tiles on a decode pool under `requests=1` while
keeping one request thread and emitting tiles in pinned fetch order: r5
**1.830419x warm / 1.943686x cold** (from parity; the last headline at
parity). Current single-binary suite claims are **2.613780x/2.624146x**
warm/cold equal-workload geometric mean and **4.559263x/4.574280x**
total-wall speedup (a warm repeat measured 2.674064x/4.727176x), all exact.

B0271/D0271 widen `pdg --fast` to resolve kNN distance ties in device order
(no CPU tie repair; coordinates, count, and order unchanged; only tie rows'
attributes may differ, bounded and reported by the fast comparator). Under
the fast contract r6 is **11.587544x / 11.206291x** (25 records differ) and
r2 **2.043481x / 2.128425x** (125 records); fast aggregates are
**2.711573x/2.712707x** geometric mean and **4.923939x/4.927839x** total
wall, recorded separately from the exact claims above, which are unchanged.

B0272/D0272 retire the B0227 automatic r4 CUDA outlier selector: the exact
host path measures faster at 1M and 4M, so r4 runs the general host path
at **7.422330x warm / 7.440248x cold** (from 5.31x). Current single-binary
suite claims are **2.700250x/2.632576x** warm/cold equal-workload geometric
mean and **4.853772x/4.644633x** total-wall speedup, all exact.

B0273/D0273 accumulate `writers.gdal` rasters on row bands in parallel
(each band replays the full point order and updates only its own rows):
r3 **2.077188x warm / 2.037489x cold** (from 1.59x). Current single-binary
suite claims are **2.772537x/2.727890x** warm/cold equal-workload geometric
mean and **5.002094x/4.851449x** total-wall speedup, all exact.

B0274/D0274 build every KD2/KD3 nanoflann tree concurrently
(structure-identical): r6 **10.689432x / 10.523255x**, r11 **12.892762x /
11.924286x**, r4 **8.377378x / 8.499474x**. Current single-binary suite
claims are **2.900213x/2.888163x** warm/cold equal-workload geometric mean
and **5.404502x/5.360525x** total-wall speedup, all exact.

## Reused upstream and fork coverage

The workload definitions intentionally compose maintained PDAL stages instead
of inventing benchmark-only implementations. The initial inventory reused the
upstream `test/data/pipeline/{colorize,crop-hole,decimate,merge,splitter}.json.in`
pipelines, `doc/pipeline.md`, the rasterization/colorization/clipping/thinning
workshop pipelines, and the COPC/GDAL tindex example. It also reuses the
upstream colorization, polygon-reprojection crop, grid-decimation, neighbor
classifier, splitter, and merge unit fixtures; the fork's exact returns/merge
and divider/splitter process matrices; its COPC reader and LAS writer
differential coverage; and the existing SMRF, statistical-outlier, sample,
voxel, reprojection, and raster-writer stage coverage. The new suite contracts
add end-to-end composition, named-fixture materialization, multi-output
publication, complete raster comparison, cold-cache control, and aggregate
refusal on top of that stage-level inventory. They do not relabel a host
fallback as native or performance-qualified.
