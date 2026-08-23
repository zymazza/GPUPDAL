# Compatibility and performance testing strategy

## Acceptance contract

The primary oracle is upstream PDAL at the commit recorded in
`cmake/pdg-oracle.cmake`. Given identical inputs, pipeline/options,
environment, and dependency versions, default `gpupdal` behavior must match:

- output file bytes;
- output ordering and dimension representation;
- metadata and JSON bytes;
- stdout and stderr bytes;
- exit status and failure boundary.

This is stricter than semantic point equality. If exact acceleration is not
available, compatibility mode runs the pinned host implementation. The
explicit `--fast` lane defined by D0261 keeps point records bit-identical and
ordered and relaxes only diagnostics, error text/status, header/VLR metadata,
and metadata JSON; its results are measured with
`benchmark_reference.py --contract fast` and labeled separately
(`docs/fast-mode.md`).

## Reproducible oracle

Every oracle capture records:

- PDAL commit and `pdal --version` output;
- complete build options and dependency versions;
- compiler, libc, CPU, CUDA toolkit, driver, and GPU identifiers;
- locale, timezone, environment allowlist, working directory, and umask;
- input SHA-256, byte size, and provenance/license class;
- pipeline bytes, argv, stdin bytes, and random seed;
- stdout/stderr, exit status, wall time, and output SHA-256.

Runs use `LC_ALL=C`, `TZ=UTC`, a fixed working directory shape, a frozen
`CLOCK_REALTIME`, fixed seeds, and controlled thread counts unless the test
intentionally varies one. Monotonic clocks remain live for timing. No test
normalizes timestamps, VLRs, metadata, NaNs, or point order: unexplained bytes
are part of the compatibility contract.

Complete-process placement qualifications additionally scrub ambient `LD_*`
and must resolve and hash the library actually loaded by both real ELF
executables. The candidate manifest must include the public launcher, its
adjacent ELF `pdg-engine`, and the loaded fork `libpdalcpp`; an `ldd` failure or
missing component rejects the report. `--qualification-model` records the
physical LAS/LAZ layout in the generated report, and the registered provenance
test binds those fields, timings, hashes, exactness, active profile, and
required automatic-route switch to the calibration case.

An automatic resident command is not successful merely because its retained
PDAL writer returned zero. After publication, parse the output header and
VLR/EVLR extents, require the header count to equal the resident executor's
observed output count, and prove the uncompressed record extent. For LAZ, parse
its chunk table and require its point counts and relative byte sizes to cover
the declared points and compressed payload exactly. This keeps the guard
O(chunks) instead of decoding the cloud twice. The physical selection matrix
truncates a just-published LAZ midway through the compressed payload and also
creates a partial plain LAS with a stale zero count; both require a nonzero
exit. Because output side effects have committed, the command must not retry
the original host pipeline.

## Differential harness

For each case the harness creates isolated oracle and candidate directories,
runs the pinned `pdal` and `gpupdal` commands, and compares all observable
artifacts. On failure it reports:

1. the first differing output, byte offset, and a bounded hex context;
2. the producing pipeline stage, where stage-bisection can determine it;
3. for LAS/LAZ/COPC, the decoded header field, VLR/EVLR record, chunk, point
   index, dimension, raw bytes, scale/offset, and decoded values;
4. the smallest named fixture and pipeline that reproduce the difference;
5. the environment manifest and exact rerun command.

Golden manifests store hashes and small redistributable outputs. Large or
restricted golden artifacts remain in local object storage and are addressed
by content hash. A golden may be regenerated only when the pinned oracle is
intentionally updated.

## Corpus design

### Public committed fixtures

Keep tiny, redistributable cases that cover:

- LAS versions 1.0–1.4 and point formats 0–10;
- zero/one/max record counts and boundary integer coordinates;
- negative/global offsets, unusual scales, and extreme valid bounds;
- every standard field, Extra Bytes types, CRS VLRs, EVLRs, waveform data,
  multiple returns, scan flags, synthetic/key/withheld/overlap bits;
- NaNs, infinities, signed zero, subnormals, ties, duplicate XYZ, and stable
  sort boundaries where the format permits them;
- valid LAZ chunk sizes, COPC hierarchy shapes, and spatial query boundaries;
- deliberately truncated, inconsistent, oversized, and previously failing
  files.

Synthetic fixtures are generated from declarative recipes so the intended raw
bytes and edge condition are reviewable.

### Local large corpus

Discover read-only candidates beneath `/home/zy/dev` and
`/home/zy/Downloads`, especially ADKLR, VEIL, LAS/LAZ/COPC,
PLY/PCD, raster, and vector datasets. A generated, Git-ignored manifest records
only path, size, hash, detected format, key metadata, and provenance class.

Stratify rather than merely sample at random: retain at least one fixture per
format/version/point-format/CRS/scale/size/order/density/compression feature,
plus the largest and smallest members. Restricted or uncertain data is never
copied, published, or uploaded.

Regenerate the ignored local manifest without modifying the source corpus:

```sh
python3 scripts/pdg/discover_corpus.py \
  /home/zy/dev/adklr /home/zy/dev/veil \
  /home/zy/Downloads \
  --output build/local-corpus.json --hash-max-bytes 67108864
```

The manifest probes only the public header for LAS/LAZ/COPC files and records
malformed-header status and strata by version, point format, record width, and
compression. It never records file contents. Exercise a selected uncompressed
LAS through the native SoA transpose/repack test with:

```sh
PDG_LOCAL_LAS_FILE='/absolute/path/from/the/manifest.las' \
  build/pdg-host-debug/bin/pdg_unit_tests \
  --gtest_filter=LocalLasCorpus.OptionalUncompressedFileRoundTripsExactly
```

Use `scripts/pdg/differential.py` for LAS, LAZ, and malformed representatives;
reports remain below the selected build directory and contain hashes and
bounded first-difference context, not input payloads.

The committed generated matrix creates deterministic raw LAS cases at test
time rather than blessing writer-produced fixtures. It covers every currently
native source format, coordinate rounding, flags, all modern scan-angle input
values, quiet-NaN payloads, zero points, enough points to cross the host worker
boundary, unregistered trailing bytes, and a malformed format-6 header that
must select fallback. Each supported format also runs an ordered assign
program with a custom intermediate, arithmetic, conditions, failed casts,
format-specific GPS/color/scan-angle fields, and return-summary changes:

```sh
cmake --build build/pdg-host-release \
  --target pdg_differential_prerequisites
ctest --test-dir build/pdg-host-release \
  -R pdg_native_translate_generated_matrix --output-on-failure
```

`PDG_REQUIRE_NATIVE=1` is an internal test guard: a case fails with status 125
instead of silently succeeding through fallback. It is used only by cases
inside the proven native envelope.

## Test families

### 0. Published upstream PDAL suite

PDAL's published testing procedure builds its configuration-dependent unit,
application, and plugin tests and runs `ctest` from the build directory. Keep
that suite distinct from PDG unit and differential tests: it proves the fork
has not regressed upstream behavior, while the stricter PDG harness proves
candidate/oracle byte identity. The maintained equivalent is:

```sh
cmake --preset pdal-upstream-tests
cmake --build --preset pdal-upstream-tests
ctest --preset pdal-upstream-tests
```

The test preset is intentionally sequential. Several upstream tests share
files below `test/temp` and are not safe under an unrestricted `ctest -j` run.
Remote STAC, EPT, and COPC cases also require access to their published HTTP
and S3 fixtures. On 2026-08-06 the dependency set in this preset registered
142 tests. The current post-slice gate passes all 140 local executables; the
two remote-fixture executables also pass on their network-enabled rerun, for
142/142 overall. See the
[upstream testing guide](https://pdal.io/en/2.9.0/project/testing.html).
After any new stage replacement is registered, rerun this entire sequential
procedure; registration or a prior 142/142 checkpoint is not evidence that the
new replacement preserved upstream behavior.

### 0.1 Full-tool and execution-boundary conformance

The public release lane is named the **GPUPDAL Conformance Suite** and is specified
in `docs/conformance.md`. `scripts/pdg/conformance.py` consumes versioned,
offline manifests and emits a schema-versioned complete-process report. The
checked-in `bounded-recipes-v1.json` expands to exactly 2,048 deterministic
boundary cases with hard case, artifact-count, byte, and timeout caps. This is
a finite reviewable matrix, not an open-ended random-input generator.

`bytes` remains the default and authoritative comparison. The explicitly
selected `copc-canonical-v1` mode is supplemental evidence only for a case that
documents pinned-oracle container nondeterminism. It gates decoded header/VLR
semantics, COPC information and hierarchy invariants, the canonical full point
multiset, and an exact bounded query. Physical hierarchy-node assignment and a
coarse preview remain diagnostic because concurrent pinned-PDAL writes can vary
them for the same points. The comparator never reclassifies the product's
default byte contract. A release-wide conformance claim requires a complete
report with zero unexplained semantic differences and every product process at
its declared exit status; partial diagnostic runs are labelled partial and
cannot satisfy the gate.

Treat the tool as more than its current native-stage count. Every release gate
must validate all three execution outcomes independently:

1. an exact whole-pipeline specialized path;
2. an exact in-process graph containing both unchanged PDAL stages and one or
   more accelerated regions;
3. an exact untouched-PDAL command fallback.

Snapshot `gpupdal --drivers` and compare its configured filter/reader/writer names
to the sibling `pdal --drivers` output. For each application verb, retain a
success case, an option/usage failure, and an I/O failure. Hybrid tests place a
point program before, between, and after non-native stages, and require the
same output bytes, streams, exit status, and failure boundary as the original
graph. A region is not considered covered merely because a fully specialized
LAS pipeline passed.

Release conformance also generates a row for every stage and every supported
optional-plugin build configuration. Each row records functional parity,
option/error coverage, native backend status, residency transitions, device
runtime/sanitizer status, break-even threshold, and representative end-to-end
performance. A host fallback can keep the functional column green during
development, but it cannot close the native-completion column. A stage may be
classified GPU-inapplicable only through an accepted decision showing that its
work is external, irreducibly serial, or loses end-to-end on the supported
size range; adjacent accelerated regions must still avoid gratuitous host
round trips.

The bounded CSF lane currently records a 22-case host matrix and a 14-case
CUDA matrix plus focused lifecycle coverage: eight host tests, 12 physical
CUDA tests, and five deterministic CUDA matrix repeats. The same lane requires
the leak-disabled ASan/UBSan preset and
memcheck/initcheck/racecheck/synccheck over its wrapper and direct-device
controls after kernel ownership changes. Stage behavior and benchmarks are
recorded under B0023 as bounded-envelope evidence and are not classified as a
P3 exit.

The bounded ELM lane records a 15-case host matrix and a 10-case CUDA matrix.
Its named fixtures pin empty and one-point execution, fractional-cell binning,
strict equality and near-equality thresholds, zero/negative thresholds,
custom classes, malformed options, nonfinite host fallback, and source-stable
equal-Z ordering across cells with `-0.0`/`+0.0`. Focused tests cover the
descriptor/compiler/rewrite contract, standalone and empty resident
lifecycles, runtime oversize/nonfinite rejection before publication, and the
CUB-derived scratch estimate at the exact 1M budget and one byte below it.
Run the CUDA matrix five times and all four Compute Sanitizer tools after any
ELM kernel, sort, allocation, or ownership change. B0024 is bounded-envelope
evidence, not a P3 exit.

The bounded skewness-balancing lane records a 20-case host matrix and a
seven-case forced-CUDA matrix. Named fixtures pin the upstream unstable-sort
boundary: unique unsorted Z executes on CUDA, while equal values, signed-zero
equivalents, constant surfaces, nonfinite values, `where`, malformed classes,
and unsupported graphs retain the host oracle. Multi-view and unproven-reader
fixtures prove the rewriter preserves the upstream stage. One/two-point,
true-crossing, near-crossing, custom-class, `only_ground`, verbose-diagnostic,
and complete-record permutation cases preserve the sequential recurrence and
observable outputs. Focused tests require rejected CUDA to leave every point
field untouched, and a process guard requires empty forced-CUDA input to
reject before stage execution. Run the CUDA matrix five times and all four
Compute Sanitizer tools after any ordering, equivalence, permutation, or
wrapper ownership change.
B0025 is controlled unique-Z evidence, not an automatic placement or P3-exit
claim.

Only genuinely linear, single-reader graphs are currently eligible for an
in-process rewrite. Tagged stages, explicit `inputs`, multiple readers, and
multiple writers retain the original graph wholesale. This is an exactness
boundary, not merely a parser limitation: standard-mode branches may share a
`PointView`, so grouping mutations into a replacement stage could change what
a sibling branch observes. Unit coverage must keep the linearity classifier
closed until branch ownership and aliasing receive explicit differentials.

### 0.2 P1.5 resident-plan qualification

The D1 planner gate treats a device region as an allocation lifetime, not just
as a sequence of CUDA-selected stages. Focused tests must assert all of the
following before a resident executor consumes a plan:

- reverse dimension liveness and region-wide first/last use, including a
  branched graph whose later sibling prevents premature retirement;
- exact materialize and release lists, with output columns released only after
  the planned asynchronous spill completes;
- peak live column-plus-index bytes rather than the union of every dimension
  ever touched by the pipeline;
- logical double storage for resident XYZ; packed LAS coordinate integers are
  a reader/writer placement cost, not resident spatial-kernel columns;
- shared-index build, reuse, invalidation, and last-consumer release across a
  coordinate-preserving bridge; and
- explicit upload/spill records on both sides of an unsupported host stage.
  Such a boundary lists known live columns and carries a full-record flag so a
  runtime layout cannot be truncated merely because pipeline JSON omitted a
  dimension.

The allocator-level regression executes the plan's materialize/release hooks
against a bounded `PointBatch` and checks allocated bytes after every stage and
spill. Its CUDA twin uses the stream-ordered allocator on physical hardware and
runs under all four Compute Sanitizer tools. V4 later adds a process
differential and reported boundary counts; V5 adds a working set above the
selected VRAM budget; V6 checks predicted versus reported index rebuilds; and
V7 is the placement model's mandatory small-host negative control. V2's gate
runs a shared-index neighborhood region (approximatecoplanar plus a ferry
bridge) over the full hash-pinned fixture and asserts the
`planner_resident_shared_index` executor, the single-tile
`whole_view_neighborhood` schedule with its scratch-inclusive budget check,
region bracket events, and byte-exact output against the pinned oracle.
V3's gate does the same for filters.lof with an assign bridge reading the
LocalOutlierFactor column; its unit-level exactness rests on wrapper
differentials that pin bit-equality against upstream on the host and CUDA
paths — including a forced distance-tie dataset exercising the closure
repair and upstream's stateful per-call minpts increment — plus
host/device engine comparisons across both index backends.
V8 extends the physical composition gate to the shared radius index:
`filters.radiusassign` selects ReturnNumber-1 rows with a ReturnNumber-2–15
reference strictly inside a three-dimensional radius, evaluates the upstream
ordered `UserData = 9` assignment finale on the host, re-uploads that column,
and keeps it resident for a downstream assign bridge that reads UserData.
The full hash-pinned fixture must select the measured radiusassign model,
report one predicted and observed index build, preserve one resident region,
declare executor/calibration agreement, and match the pinned oracle bytes.
Planner metadata alone does not satisfy those process gates.

V9 adds a non-index whole-view producer. An exact
`filters.label_duplicates(dimensions=Classification)` pass must preserve the
first `Duplicate` byte, publish the remaining adjacent-predecessor labels, and
retain the unsigned-byte output for a downstream assign bridge. Preflight and
the emitted region must declare no spatial index and no additional per-point
scratch. A separate composition proves that label_duplicates may precede a
neighborhood consumer while that later consumer still uses exactly one
planner-owned index. Neither path may create a stage-private index.

The current reader matrix runs the same ordered
assign/expression/range/crop/ferry region through LAS/LAZ, BPF, PLY, PCD, and
text streaming readers. Standard mode adds sorted COPC. Generated pipeline
files use mode-specific directories so sorted and streaming matrices remain
race-free under parallel CTest. Unsupported input ordering or writer
execution-mode changes must select untouched PDAL rather than silently changing
point order.

### 1. Exact unit and serialization tests

Test registry mappings, option defaults, parser errors, numeric conversions,
header layouts, record packing, metadata emission, and CLI diagnostics against
literal expected bytes. Check endian, alignment, overflow, and aliasing cases.

### 2. Stage differential matrices

Generate valid option combinations for every stage, including defaults,
explicit defaults, boundary values, `where`, `where_merge`, custom dimensions,
empty views, multi-view pipelines, and error paths. Compare the full command
artifacts. Bisect multi-stage pipelines to identify the first divergence.

Include valid failures in the matrix. Preparation success alone is
insufficient: some PDAL conversions fail only when a particular point value is
written. The type-widening `ScanAngleRank=>Intensity` ferry regression, for
example, requires the hybrid candidate to preserve PDAL's runtime conversion
failure and empty output artifact. Test both sides of every eligibility edge so
conservative fallback cannot become accidental semantic expansion.

The `filters.label_duplicates` matrix must treat input order as observable:
the filter compares only immediate predecessors and performs no implicit sort.
Cover absent, empty, comma-separated, array, and repeated `dimensions` forms;
first-row preservation; binary64 rounding of wide integers; NaN and signed-zero
comparisons; explicit stable-sort composition; empty and multi-batch inputs;
missing/invalid dimensions; and `where`/`where_merge` fallback. Selecting
`Duplicate` as an input stays on the sequential host path because preceding
writes affect later comparisons.

For multi-view stages, a point-set comparison is insufficient. Capture the
ordered artifact-name set, every file byte, empty outputs, warnings, and
downstream composition. Exercise consecutive partitions and multiple incoming
views so persistent stage state is visible. Divider tests must retain requested
empty views and per-view source order; splitter tests must pin first-encounter
tile ids, default-origin persistence, exact negative grid boundaries, strict
buffer edges, and all four possible memberships. Follow partitions with both a
numbered writer and an audited merge plus an order-sensitive stage.

For the shared spatial engine, compare bulk query outputs independently of any
consumer stage and then repeat through complete stages. Radius fixtures include
self-neighbors, duplicate coordinates, points exactly on `r²`, points on both
sides of cell faces, empty/single-point views, 2D-vs-3D disagreement,
nonfinite coordinates, invalid radii, and cell-frame limits. Planner tests put
two neighborhood stages back-to-back, branch them from a common point set, and
separate them with XYZ mutation, compaction, ordering, split, and merge stages;
the reported index-build count and backend-specific 28-byte grid or
76-byte worst-case adaptive-kNN estimate must match the lifecycle. kNN
properties compare every id and squared-distance row against brute force for
k=1/8/32/64, cover 2D-vs-3D, `k > size`, observable distance ties, and a sparse
frame where the bounded grid must report incomplete instead of guessing while
the Morton BVH must prove an exact result. A small mixed-density cluster plus
distant outliers pins the grid below the measured crossover; a broad dense
cluster plus outliers forces BVH, and a regular 3D lattice pins grid. The probe
uses deterministic hashed source positions so periodic input order cannot
alias it. Selector performance fixtures at 262,144, 524,288, and 1,048,576
points must choose the measured winner, while both backends reproduce ordered
means, `filters.nndistance` values, covariance coefficients, and eigensystems
bit-for-bit. The distance property separately pins the `kth` square root and
the serial `avg` square-root/add/divide sequence; equal-distance ties are
accepted only because both modes observe values rather than point identities.
CUDA properties run radius, kNN, ordered mean, ordered distance, covariance,
and eigen queries through both backends across more than one launch tile. The
composed resident CUDA differentials force the Morton BVH, and all
spatial properties plus forced radius/statistical process differentials run
under memcheck, initcheck, racecheck, and synccheck before automatic selection
is possible.

Radius-bounded out-of-core tests treat tiling as a semantic partition, not a
mere chunk-size variation. Core cells are half-open and assign every source
point exactly one owner; radius halos are closed and may duplicate candidates,
but a ghost can never publish a result. Place points on negative and positive
core faces, exactly on halo faces, exactly at `radius²`, immediately inside and
outside both boundaries, and at duplicate coordinates. Retain source order in
every tile, vary the tile edge independently of the radius, and compare the
whole-view and mosaicked outputs bit for bit. Run both a count-valued client
(`filters.outlier`) and a binary64-scaled client (`filters.radialdensity`) so
owner/scatter correctness cannot hide arithmetic-order errors. A 2D core tile
may feed a 3D query because it only adds candidates; explicitly reject a 3D
tile feeding a 2D query. Exercise empty mosaics, nonfinite coordinates, signed
tile-frame overflow, too-small halos, per-tile capacity overflow, incomplete
owner publication, and forced allocation failure before accepting automatic
capacity selection.

The process gate must force at least two tiles and compare the complete
artifact, streams, diagnostics, and exit status with upstream; a silently
selected whole-view path is a failed tiled test. Repeat with several tile edges
and memory budgets, then run the primitive and complete forced-tiled pipelines
under memcheck, initcheck, racecheck, and synccheck. Current radius execution is
double-buffered when staging is pinned and exposes its lane/reuse counts to the
test contract; pageable staging stays on one reusable lane. The seam property
forces both lanes through repeated reuse, injects a later invalid tile while
another lane remains in flight, and then reuses the caller-owned allocation
pool for an exact recovery run. Continue to perturb completion order and
allocation failure while requiring the same output hash. kNN tiling requires a
separate adaptive-halo or global completion proof and must not be inferred from
the radius result.

Covariance/eigensystem tests pin more than the geometric answer. They compare
the ordered online centroid, float-narrowed demeaned coordinates, all six
sample-covariance coefficients, zero-matrix classification, three ascending
eigenvalues, all nine eigenvector coefficients, and every status bit directly
with the same Eigen solver used by upstream. Current fixtures include duplicate
and equal-distance neighborhoods, insufficient rows, zero covariance, and a
bounded-search failure. Promotion fixtures additionally require collinear and
planar sets, nearly repeated eigenvalues, and ill-conditioned covariance. The
complete `filters.normal` matrix separately covers orientation, curvature, skip
messages, empty views, `knn` boundaries, radius/viewpoint/refinement/where
fallbacks, and the invalid simultaneous `knn`/`radius` error. Unit-length or
orthogonality checks remain useful metamorphic guards, but are never accepted
as substitutes for exact coefficients and complete-process artifacts.

The same primitive is exercised through separate complete-process matrices
for `filters.nndistance`, `filters.eigenvalues`, and
`filters.covariancefeatures`. Nndistance covers defaults, both modes, exact
operation order, case-insensitive parsing, ties/duplicates, insufficient and
empty views, both native boundaries, invalid options, and conditional
fallback. Eigenvalues covers
normalization, stride/radius selection, ignored `min_k`, ties, empty input,
and both k boundaries. Covariance features covers all three eigenvalue modes,
every non-density feature, feature-list parsing/order, zero/duplicate
neighborhoods, and the density/optimized prerequisites that remain on
upstream. Device-column properties initialize outputs with sentinels, cross
launch-block boundaries, exercise raw/sqrt/normalized modes, preserve skipped
rows, and compare every projected binary64 bit and status byte with host
formulas. `Omnivariance` and `Eigenentropy` have separate complete-process
fixtures because CUDA `cbrt` failed the bit gate by one ULP on an observed
input; their exact host bridge copies only three eigenvalues and must reproduce
the upstream artifact before its result is uploaded into the resident batch.
The bridge is reported explicitly and is not counted as a native CUDA formula.

`filters.approximatecoplanar` has a separate 26-case process matrix because its
public `knn` contract is not interchangeable with the surrounding feature
stages. The oracle calls `KD3Index::neighbors(point, knn)` directly, including
the query point; an implementation that silently requests `knn + 1` is wrong.
Cover defaults, explicit thresholds, `knn` 3 and 64, duplicates and distance
ties, both strict-equality boundaries and immediately-inside controls,
`knn` larger than the view, zero covariance, widened `Coplanar` storage,
missing coordinate dimensions, solver failure, empty input, conditional forms,
invalid option types, and counts immediately outside the CUDA envelope. The
output is a physical unsigned-byte `Coplanar` column. Initialize it with a
non-Boolean sentinel before a zero-covariance row and require both the exact
upstream informational diagnostic and preservation of the old byte. Compare
the strict expression
`lambda1 > thresh1 * lambda0 && thresh2 * lambda1 > lambda2` bit for bit; an
equality must remain false.

Eight CUDA process differentials exercise simple LAS, maximum `knn`, forced
deterministic tie repair, solver-failure fallback, the CSV strict-boundary and
zero-covariance/preserved-byte fixtures, experimental small-view fallback, and
resident composition. The fallback cases must remain host-selected rather than
falsely satisfying a require-CUDA assertion. Two CUDA unit/property paths cover
typed-U8 projection and injected solver-failure prefix mutation. The projection
property uses sentinels across a launch-block boundary and checks skipped bytes
as well as computed 0/1 values.

The exactness gate is complete on the physical SM-89 runner. Host Debug passes
266 executions plus one expected opt-in corpus skip in 267 registrations; the
CMake-injected `detect_leaks=0` ASan/UBSan differential preset has the same
result, while leak checking remains in its separate lane. Three initial
failures compared process-specific LeakSanitizer diagnostics and reran clean
under the prescribed mode. CUDA Release passes 384 executions plus
the skip in 385 registrations in 103.66 seconds. Tie repair and resident
composition each repeat exact output five consecutive times. Both CUDA unit
paths are clean under memcheck, initcheck, racecheck, and synccheck. The
standalone, strict-boundary, zero-covariance, resident, max-`knn`, and tie-repair
complete process cases are also clean under all four tools. The serialized CI
loop forces the uniform-grid backend for standalone approximate-coplanar and
the Morton BVH for the other cases; tie repair must satisfy its explicit proof
guard. The published PDAL gate is 142/142, comprising 140 local tests and both
remote-fixture tests on a network-enabled rerun.

The accepted checkpoint still has no qualified automatic selector. D0049
proposes a 262,144-point boundary for default or explicit `knn=8` on the
current `NVIDIA GeForce RTX 4090` SM-89/CUDA-13.3 profile. The candidate
requires a direct three-stage reader/filter/writer pipeline, regular pipeline
JSON and input files, reads only a fixed header from a `.las` or `.laz` input,
excludes COPC, honors a reader count cap, requires known cardinality and an
available non-disabled CUDA runtime, writes uncompressed `.las` with
`extra_dims=all`, and creates a standalone terminal neighborhood region with
reuse disabled.
Production artifacts compile the proposed selector off; a separate clean
qualification artifact enables it without a runtime selection variable. Failed
file, header, cardinality, option, graph, runtime, device-profile, or
recoverable execution gates must delegate unchanged or execute the exact host
wrapper.

The current performance evidence is forced, not option-free. At revision
`e0486c78ea7899b316037ebf1162bcc4e52c925f`, clean two-warmup/ten-sample
reports with frozen time and exact paired hashes measure 2.158750x pinned PDAL
for LAS and 1.974880x for point-format-8 LAS at 262,144 points. The 2.200810x
LAZ report was captured from a dirty tree and is diagnostic only. Before D0049
can be accepted, a clean LAZ baseline and clean normal invocations with no PDG
selection environment must reproduce all three exact results and a retained
profile must prove CUDA execution. A second invocation with
`PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA=1` must prove both the
automatic rewrite and device execution rather than enabling them. Boundary
controls at 262,143 points and file/runtime/device rejection cases must use
`PDG_REQUIRE_APPROXIMATECOPLANAR_HOST_FALLBACK=1` where the exact host wrapper
is expected, and must compare the complete oracle artifact and diagnostics.
All of those selector-specific results remain pending.

SM 89 remains the only physically exercised runtime target. Portable source
and fatbin generation make forced CUDA available as the qualification surface
on other compiled NVIDIA architectures, but each profile still needs physical
bit-exact, sanitizer, option-free selection, and crossover results on Vast
before it may join the automatic allowlist. This proposal does not change the
automatically qualified count or imply catalog-wide CUDA coverage.

The resident-neighborhood composition fixture runs `filters.normal`,
`filters.eigenvalues`, and `filters.covariancefeatures` consecutively with the
same `knn`. Its host lane checks the complete upstream artifact. Its CUDA lane
sets `PDG_REQUIRE_NEIGHBORHOOD_REUSE=1`, so the second and third wrappers fail
unless they inherit the first wrapper's device XYZ batch and spatial index;
the cached eigensystem is reused for the equal request. Run the same fixture
under memcheck, initcheck, racecheck, and synccheck. A byte-identical result
without the forced reuse assertion is not accepted as residency evidence.

A second composition fixture appends an assign/ferry-compatible point program
that reads `Linearity`, `Curvature`, and `Eigenvalue0`. It sets both
`PDG_REQUIRE_NEIGHBORHOOD_REUSE=1` and
`PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE=1`; passing therefore proves that the
program executed in the retained device batch, gathered only missing standard
columns, and did not rebuild or re-upload the feature columns. The complete LAS
artifact, diagnostics, and exit status must match upstream, and the entire
pipeline runs under all four Compute Sanitizer tools. Planner tests separately
prove that a non-consuming predicate closes the region instead of receiving a
false residency annotation.

A third composition fixture runs `filters.normal`, `filters.nndistance(kth)`,
an `NNDistance=>GpsTime` ferry, and `filters.nndistance(avg)`. The first two
stages must share one XYZ batch/index, the ferry must consume the resident
`NNDistance` column directly, and the final distance stage must establish a
new region after that point-program boundary. Both reuse environment gates are
mandatory. The host and CUDA artifacts are compared byte-for-byte, the option
matrix and composition run with ASan/UBSan leak detection, and the complete
CUDA pipeline runs under all four Compute Sanitizer tools.

A fourth composition fixture runs `filters.normal(knn=7)`,
`filters.approximatecoplanar(knn=8)`, a `Coplanar=>UserData` ferry, and
`filters.eigenvalues(knn=7)`. The differing public `knn` values intentionally
resolve to the same eight-row self-inclusive eigensystem. Its CUDA lane sets
`PDG_REQUIRE_NEIGHBORHOOD_REUSE=1`,
`PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE=1`,
`PDG_REQUIRE_NEIGHBORHOOD_COPLANAR_COLUMN_REUSE=1`, and
`PDG_REQUIRE_NEIGHBORHOOD_EIGENSYSTEM_REUSE=8`. Passing proves shared XYZ/index
and eigensystem residency and direct consumption of the physical unsigned-byte
column; byte identity alone without every proof gate is insufficient.

The serialized shared-spatial sanitizer loop includes this fixture and the
standalone, strict-boundary, zero-covariance, max-`knn`, and forced tie-repair
cases with their exact LAS/CSV inputs. Resident composition and tie repair each
pass five consecutive exact runs, and all six complete cases are clean under
every Compute Sanitizer tool.

### 3. Metamorphic tests

These catch shared bugs even when both engines agree. Examples:

- identity transform/reprojection is byte-preserving where upstream is;
- split then merge respects the oracle's exact ordering;
- crop nesting and repeated idempotent filters have expected subset behavior;
- translated local coordinates preserve neighbor relations;
- tile-size/VRAM-budget changes do not change strict output;
- LAS decode/encode and SoA reorder/inverse-reorder preserve raw records;
- normals are unit length and voxel filters obey occupancy bounds.

### 4. Determinism and schedule perturbation

Run strict cases repeatedly while varying CUDA stream interleaving, upload
chunking, tile size, allocator reuse, host thread count, and warm/cold state.
Every strict artifact must retain the same hash. Randomized stages use a pinned
counter-based seed and are compared to the oracle's exact sequence.

### 5. Numeric boundary cases

Independently compare GPU math to integer or high-precision reference code at
condition-number extremes, eigenvalue multiplicities, nearly collinear/planar
neighborhoods, cancellation-heavy reductions, UTM-scale offsets, tile edges,
and radius/kNN tie distances. Exact mode also checks evaluation and reduction
order against the oracle.

### 6. Deterministic reader boundaries and I/O failure handling

Use declarative, structure-aware generators for every native reader. Enumerate
field boundaries and replay valid and previously failing corpus members with
pinned recipes and hashes. Require bounded memory, bounded work, no hangs, and
oracle-consistent acceptance/rejection. Simulate short reads, I/O errors,
allocation failures, invalid chunk tables, invalid hierarchies, and
cancellation at each async boundary. Every failure becomes a small named unit
or differential regression case.

### 7. Memory and concurrency correctness

- ASan/UBSan and leak checks for host code.
- compute-sanitizer memcheck, racecheck, initcheck, and synccheck for CUDA.
- stress stream/event lifetime, pool reuse, zero-sized allocations, OOM plan
  fallback, graph cancellation, and exceptions crossing API boundaries.
- assert halo points are queryable but never emitted and that index rebuild
  counts match planner expectations; require exactly one core owner and a
  complete source-order mosaic under every tested tile edge.

For an exact differential run with an ASan candidate, pass the compiler's
`libasan.so` path through `--candidate-preload`. If the oracle is instrumented
too, also pass it through `--oracle-preload`. This keeps the sanitizer runtime
first while the frozen-time shim follows it. The Linux shim uses direct clock
syscalls for non-realtime clocks so sanitizer initialization cannot recurse
through `dlsym`.

The CTest ASan/UBSan presets inject `libasan` automatically before the frozen
time shim for oracle and candidate processes. Manual differential invocations
must supply the equivalent explicit preload flags; do not rely on ambient
`LD_PRELOAD` ordering.

Exact oracle/candidate process differentials use the prescribed
`detect_leaks=0` sanitizer mode because LeakSanitizer emits process-specific
PID, address, and stack diagnostics that are not application stderr. Their
CMake definitions inject this setting automatically. Keep leak checking in its
separate candidate-focused lane; never normalize sanitizer diagnostics out of
the exact stream comparator. AddressSanitizer and UndefinedBehaviorSanitizer
remain active in the differential lane.

The same CMake injection sets `UBSAN_OPTIONS=suppressions=tests/sanitizers/
ubsan-lazperf.supp` (B0270): vendored lazperf's LAZ integer predictors wrap
int32 arithmetic by design (point14 field decoders, integer decompressor),
pinned PDAL ships the same code, and every reference artifact is
byte-identical, so UBSan's report on that vendored path is not application
stderr and must not fail an exact stream comparison. The suppression is
scoped to `vendor/lazperf`; fork code stays fully instrumented. Manual
sanitizer differential runs must pass the same `UBSAN_OPTIONS`.

CUDA builds add a complete-output comparison against the exact host writer and
a generated matrix for formats 0–3 and 6–8. The matrix crosses staging chunk
boundaries, includes unregistered trailing record bytes and every int16 modern
scan-angle encoding, and compares the entire output vector. On machines
without a driver these device cases are explicitly skipped, while argument,
pinned-allocation, and zero-point/header behavior still run. The self-hosted
GPU lane additionally runs memcheck, initcheck, racecheck, and synccheck; a
compile-only result is never described as runtime validation.
The self-hosted CLI differential sets `PDG_REQUIRE_CUDA_TRANSLATE=1`, which
returns status 124 unless the exact CUDA path actually produced the candidate
artifact; this prevents an automatic host fallback from creating a false
positive GPU result.

The first P1 point-program lane checks the pinned PDAL `numericCast` contract
across all 100 source/destination physical type pairs, ordered dependencies,
empty sources, coordinate encoding, conversion-failure preservation, and exact
serial/parallel bytes. Whole-file process differentials cover LAS formats 0–3
and 6–8 with chained standard-field mappings, coordinate destinations, modern
scan angles, flags, colors, bounds, and return histograms. Named fallback cases
cover custom dimensions and upstream layout-type widening. Legacy scan-angle
sources and noncanonical original XYZ sources are rejected by native selection
because filters observe their pre-writer logical values.

Assign coverage adds the pinned expression grammar, all math/logical
functions, new-dimension zero initialization, ordered conditional writes,
malformed expressions, identical 1-vs-7-worker output, a five-stage
assign/ferry/assign process differential, and a full host-math process
differential. Expression/range coverage adds inclusive/exclusive bounds,
negated groups, NaNs, stable survivor order, all-pass, reject-all, selective,
empty-output, and post-filter header/bounds/return summaries. Crop coverage
adds inclusive 2D/3D boxes, `outside`, NaN coordinates, empty standard-mode
input, same-SRS input, direct and in-process execution, and conservative
fallback for multiple bounds, reprojection, malformed bounds, and geometry
variants. CUDA tests cover all destination types, ordered chains crossing
kernel launches, exact special values, raw unaligned packed fields, stable
compaction across every physical type, and generic/encoded coordinates. A
compact 1,025-point selective case forces four launches so both
stream/allocation lanes drain and are reused; it checks survivor order and
packed field values. On the RTX 4090 all device tests pass in Debug and
Release, and memcheck, initcheck, racecheck, and synccheck report zero errors
or hazards.

Ordinal-selection coverage treats execution mode and chunk boundaries as part
of the observable contract. Unit tests pin PDAL's different fractional
decimation sequences in standard and streaming mode, global head state, tail
inversion, zero/oversized counts, warning text, and unsafe standard offset
domains. `tests/differential/ordinal_matrix.py` runs 23 complete-process cases
covering defaults, integral and fractional steps, offset/limit, empty input,
two decimators, predicate splits, chained head/tail, warnings, and conservative
fallback. It accepts `--candidate-preload` so the same matrix runs against an
ASan/UBSan candidate with `libasan` before the frozen-time shim. A
21,970,934-point local LAS case crosses more than 160 batches while bounding
the derived output to 100 points. CUDA promotion additionally requires the
compiled host/device mask comparison and four-chunk direct-LAS regression to
run on hardware under memcheck, initcheck, racecheck, and synccheck; until
those gates pass, compilation is reported separately from device validation.

Global-reduction coverage treats the selected source index as observable, not
just the extreme value. `tests/differential/locate_matrix.py` runs 15 complete
process cases over maxima/minima, case-insensitive and invalid kinds, integer
and floating dimensions, NaNs, empty input, a custom-dimension point-program
chain, a predicate/reduction chain, and conservative fallback boundaries. The
host and device candidate records explicitly distinguish a comparable value
from PDAL's initial sentinel so all-NaN, infinity-only, and exact-sentinel
inputs still select source point zero. Unit and process cases force chunk and
block boundaries through ties. The matrix accepts independent oracle and
candidate preloads and therefore also runs against ASan/UBSan builds. CUDA
promotion additionally requires the multi-block typed reduction under
memcheck, initcheck, racecheck, and synccheck plus a same-machine break-even
matrix; until then `filters.locate` remains force/require-only.
The local large-corpus case scans a read-only 21,970,934-point LAS file in
131,072-point chunks and matches the one-point
oracle LAS exactly (2,301 bytes, SHA-256
`44e4bf505c7f304ddba014a29f3b5480dce1ea04a2ccf918d2e8a1cd650d3095`).

Coordinate-map coverage treats simultaneous reads, arithmetic order, and
coordinate metadata as observable. `tests/differential/transformation_matrix.py`
runs nine complete-process cases covering identity, mixed affine coefficients,
full homogeneous division, a 198,975-point streaming-boundary input, an
assign/transformation/predicate chain, empty input, and untouched-PDAL fallback
for inversion, SRS override, and matrix files. Unit tests independently pin
matrix length and the original-XYZ dependency; the CUDA unit compares all bits
for a 131,103-point multi-block affine launch. The process matrix runs again
with ASan/UBSan. Both forced affine process differentials, the physical unit,
and memcheck/initcheck/racecheck/synccheck now pass on the RTX 4090. Clean
complete-process measurements reject an automatic standalone selector: CUDA
is 0.357x pinned PDAL at 250,000 points, 0.851x at 4,000,000, and only 1.008x
in a one-shot 21,970,934-point run. The kernel remains a resident-fusion
primitive; D0042 records the negative standalone gate.

Color-interpolation coverage treats GDAL ramp bytes, execution mode, and
mutable per-view bounds as observable. `tests/differential/colorinterp_matrix.py`
runs 39 complete processes over all embedded ramps and an external TIFF;
explicit, partial, auto, stddev, and MAD bounds; negative and explicit `k`;
clamp/invert and existing colors; NaN, one-point, empty, and multi-batch data;
expression, sort, divider, and consecutive-stage composition; fallbacks and
failures. It pins the upstream difference between stream mode (explicit bounds,
where `processOne` does not recompute `k`) and standard mode, plus first-view
auto-range persistence after a partition. Unit tests cover exact bin edges and
a compiled 131,103-point host/device RGB comparison. The 39-case matrix is
green in Debug, Release, and ASan/UBSan builds. A read-only 21,970,934-point
explicit-range streaming case also matches the oracle's 790,955,889-byte LAS
exactly (SHA-256
`43fead56eade3ffa503311d9f47328d2fbd6888b913b5c7d92dfa12c8881dc23`).
The forced explicit-stream, inferred-standard, and divided multi-view CUDA
process tests, physical property, and all four Compute Sanitizer tools now
pass. The same-machine end-to-end gate that includes GDAL load, transfers, and
output writes is negative: explicit-range CUDA reaches only 0.345x PDAL at
250,000 points and 0.787x at 21,970,934; auto-range reaches 0.756x at
4,000,000 and 0.873x at full scale. Default standalone execution therefore
remains host-selected, with resident fusion as the next gate (D0042).

Stats coverage treats ordered reduction state and complete metadata as
observable. Direct units feed identical values, including special binary64
cases, into `pdal::stats::Summary` and the native primitive and compare every
state/derived bit. `tests/differential/stats_matrix.py` runs 22 complete
processes over default/explicit dimensions, advanced moments,
enumerate/count/global modes, NaNs, empty input, forced stream and standard
execution, warnings and invalid options, point-region and multi-view
composition, explicit native replacement, and default upstream selection. It
compares metadata JSON as well as output files, status, stdout, and stderr and
passes in Debug, Release, and ASan/UBSan builds. A compiled CUDA property
crosses tile boundaries with 131,103 points, three concurrent dimensions, and
persistent prefix state. A 21,970,934-point six-dimension run emits the same
790,955,889-byte LAS as the oracle, SHA-256
`ff14463744dbe9ddd2f1d10271d278a7e41478e254d96a0780bda9d7aa1da2fe`.
The host wrapper's 0.909x diagnostic is a rejection gate: option-free stats
remains upstream, while the audited stage boundary permits native regions on
either side. Device promotion requires forced stream/standard process
differentials, memcheck, initcheck, racecheck, synccheck, and an end-to-end RTX
4090 break-even matrix; compilation does not count as runtime validation.

Information metadata coverage treats parser quirks, key ordering, and binary64
representations as observable. `tests/differential/info_matrix.py` runs 20
complete processes over option-free bounds/count, point ranges and ordering,
2D ties, the historical 3D query parse, alternate separators, combined
point/query behavior, empty data, forced stream/standard execution, point-
program and partition composition, fallbacks, and failures.
`tests/differential/expressionstats_matrix.py` runs 19 host/fallback complete processes over
canonical expression ordering, overlapping predicates, floating and NaN
targets, no matches, empty input, stream/standard execution, custom
intermediates, multi-view composition, fallbacks, and errors. Both compare
metadata JSON, output artifacts, status, stdout, and stderr and pass in Debug,
Release, and ASan/UBSan builds.

Its `--require-cuda` lane runs the ten cases wholly inside the exact device
envelope and fails if the GPU implementation is not selected. It covers both
execution modes, overlapping/no-match predicates, custom intermediates,
point-program boundaries, and post-partition multi-view accumulation. The
direct 131,103-point property runs under memcheck, initcheck, racecheck, and
synccheck; the full ten-case process lane also runs under memcheck and
racecheck, with initcheck and synccheck success required before promotion.

The read-only 21,970,934-point corpus also passes both host differentials. Both
produce the oracle's exact 790,955,889-byte LAS; info metadata is 25,937 bytes
with SHA-256
`5c7bd64c2c329f2ac770526bf931c55aba10b98cae7701459d02040ee61a56d4`,
and expressionstats metadata is 15,242 bytes with SHA-256
`795209c4dc3e679dc83a56f28dee0ea46a12a4086bd16e090d8d59608741c84b`.

Primitive units separately pin BOX3D sentinels, NaNs, infinities, signed-zero
bits, first ties, encoded-coordinate decode, chunk-global merging, numeric map
order, duplicate counts, empty selection, and envelope rejection. CUDA
properties cross a 131,103-point reduction boundary. The histogram property
also checks stable predicate selection and the first source bit pattern for
equivalent `+0`/`-0` keys. Expressionstats is no longer compile-only: its
physical property, forced process matrix, sanitizer lane, and dirty-tree
real-corpus break-even diagnostics pass. B0008/D0044 additionally accept a
clean persistent-workspace, option-free work curve: CUDA is 1.126x PDAL at
1,000,000 points for three cheap predicates, 1.212x at 2,000,000 for two, and
1.105x at 4,000,000 for one. Matching below-threshold controls stay on host at
0.957x, 0.973x, and 0.979x. The selector reads only a fixed LAS header and
requires known cardinality; its exact runtime envelope, disable/device
fallbacks, and final automatic-path memcheck were exercised separately. The
clean one-million-point Nsight trace records 1.337595 ms of kernels, 0.563131
ms of device memory operations, and one 0.869249-ms pinned allocation,
confirming stage-local workspace reuse. Its one first-use stream creation still
costs 97.046651 ms, reinforcing the resident execution-context target. Info
remains unpromoted and resident-plan coverage remains open for both stages.

Randomize coverage distinguishes exact serial permutation generation from
composition. `tests/differential/randomize_matrix.py` runs 14 complete
processes over fixed seeds including zero and `UINT32_MAX`, point regions on
both sides, stable ordering, empty and multi-batch inputs, per-view shuffle plus
merge, and invalid/option-rich delegation. The matrix passes in Debug, Release,
and ASan/UBSan. A 21,970,934-point seeded shuffle followed by a native
assignment produces the oracle's exact 790,955,889-byte LAS (SHA-256
`e66521fc478722f27b5d782410b8ec32b3e8774ffe426d4380b8a260eb98b347`).
This validates that the unchanged shuffle is an ordering bridge; it does not
qualify a CUDA randomize implementation.

Robust-statistics coverage preserves the selection algorithm, not merely an
equivalent-looking fence. `tests/differential/robust_matrix.py` runs 16
complete processes over IQR/MAD defaults, explicit/zero/negative multipliers,
integer and floating dimensions, NaNs, custom intermediates, empty views,
fallback options, and missing-dimension failure. Host units pin quartile
indices, upper medians, strict endpoints, deviation division, and encoded
coordinate conversion. Device eligibility explicitly excludes nonfinite and
negative-zero keys; the CUDA regression sorts 131,103 finite values and
compares every threshold bit and mask byte. The matrix is repeated under
ASan/UBSan with `libasan` explicitly preloaded into both oracle and candidate.
Two read-only large-corpus cases run IQR with `k=-1` and MAD with `k=0` over
21,970,934 points; each matches the oracle's empty 2,265-byte LAS with SHA-256
`24fd80a8db0b248e9b86e83560679fce458af8b12a4cd24cf4dcaa1d633b583b`.
Forced IQR/MAD process differentials and all four Compute Sanitizer tools now
pass. The clean complete-process gate rejects standalone automatic CUDA: IQR
is 0.349x PDAL at 250,000 points, 0.799x at 4,000,000, and 0.951x at
21,970,934; MAD is 0.801x at 4,000,000 and 0.949x at 21,970,934. D0042 keeps
the exact device selections for resident plans and the faster host path for
ordinary standalone execution.

Ordering coverage treats the permutation itself as observable. Host tests
reproduce the exact `std::sort` first pass and later stable passes over each
physical PDAL type, including the resulting last-dimension priority.
`tests/differential/ordering_matrix.py` runs 20 complete processes across
normal/stable and ascending/descending forms, duplicate integer keys, NaNs,
dimension aliases/arrays, repeated keys, custom point-program intermediates,
selection after ordering, empty and 198,975-point inputs, fallbacks, and
preparation failures. CUDA tests compare complete 131,103-point permutations
for stable duplicates, unique normal keys, and a multi-key program. A separate
device case requires normal-key ties to reject publication. The process matrix
runs under ASan/UBSan. A read-only large-corpus case stable-sorts
Classification over 21,970,934 points and then retains 100 points; both
processes emit the same 5,865-byte LAS with SHA-256
`f0961728c852e64d6d9e1106525913aa2361ea3348ca87f897d3a92f1f020885`.
The physical CUDA properties, forced stable process differential, and all four
direct Compute Sanitizer tools pass. Complete-process measurement rejects a
standalone selector: at 21,970,934 points CUDA and PDG host are within about
0.7%, and host wins at 4,000,000 points. Retain the device permutation for
resident compositions and rerun the break-even gate when publication no
longer crosses PointView/host boundaries.

Morton coverage treats the upstream traversal—not merely spatial locality—as
observable. Host properties compare the ordinary 62-bit radix key with PDAL's
most-significant-differing-axis comparator and pin the reverse grid,
interleave, bit reversal, and stable duplicate-code order.
`tests/differential/morton_matrix.py` runs 15 complete processes across
ordinary/reverse modes, duplicate and degenerate coordinates, sequential
Morton stages, a following predicate, empty views, 198,975-point input,
fallbacks, and failures. The matrix also runs under ASan/UBSan with libasan
preloaded into both processes. A read-only 21,970,934-point local LAS followed
by a 100-point head emits the same 5,865-byte artifact in both processes
(SHA-256
`61638be4f61d806608c4ac6df4da0e0715915d0347b1dcf68b57afe1bf898c6e`).
The physical CUDA regression compares every key bit and the complete stable
permutation for 131,103 points in both modes. Forced ordinary and reverse
process differentials and all four direct Compute Sanitizer tools pass.
Clean checkpoint B0006 brackets automatic selection with ten-sample,
option-free process runs: 1,000,000 points remain host-selected at 0.983x
PDAL; ordinary CUDA is 1.248x at the 2,000,000-point threshold and 1.528x at
4,000,000; reverse CUDA is 1.487x at 2,000,000. Every warmup and measured LAS
artifact is byte-identical. A clean option-free Nsight Systems trace proves
the selector launches the Morton key kernel and eight CUB radix passes, and an
automatic-path memcheck run reports zero errors. The selector still requires
finite nondegenerate bounds, passes the complete value envelope before
publication, honors the disable switch, and returns to host below threshold or
without a usable device.

Categorical-partition coverage treats output view identity and file numbering
as observable. `filters.groupby` creates views when a key first appears, even
though its internal map is key-sorted, and preserves source order within each
view. `tests/differential/groupby_matrix.py` runs 16 complete processes across
standard/custom/negative keys, single and multiple groups, preceding sort and
splitter stages, consecutive groupings, unsafe downstream guard fallbacks,
multiple input views, empty and 198,975-point inputs, runtime conversion
failure, and preparation errors. It compares the full set of generated files
and repeats under ASan/UBSan. A read-only 21,970,934-
point local case emits four byte-identical LAS files totaling 790,962,684
bytes. The forced stable signed-key partition and all four direct primitive
sanitizer tools pass on the RTX 4090. End-to-end measurement, including
PointView assembly and multi-file writes, rejects standalone CUDA at both
250,000 and 21,970,934 points; future promotion is conditional on resident
multi-view publication removing that boundary.

Return-partition and merge coverage additionally treats fixed view identity,
empty-group warnings, malformed return fields, persistent merge state, and the
point at which downstream single-view rewriting becomes safe as observable.
`tests/differential/returns_merge_matrix.py` runs 29 complete processes over
default/reordered/array/duplicate/empty groups; malformed, zero, and invalid
return values; sort, splitter, and groupby predecessors; consecutive returns;
198,975-point, empty, and missing-dimension inputs; warnings and failures;
single/multiple-view
merge; merge followed by point programs or grouping; SRS and `where`
fallbacks; and unknown options. Every status, stream, filename, missing or
extra artifact, size, and byte is compared. The matrix is clean under
ASan/UBSan. A 21,970,934-point read-only local case executes
`returns(all) -> merge -> head(100)` and matches the oracle's 5,865-byte LAS
with SHA-256
`64432d20259eaabbe5c3c93e6be1243f07f9ec0160de90af38f17e2162b853a1`.
The 131,103-point property, forced process differential, and all four direct
primitive sanitizer tools pass on the RTX 4090. Complete-process measurement
including PointView assembly is exact but does not beat PDAL, so the device
partition remains a resident composition primitive rather than an automatic
standalone path. Divider and splitter follow the same rule; their artifact-set
benchmarks cover 17 and 49 numbered outputs respectively.

### 8. Performance and roofline tests

Benchmark exact and fast modes independently on the same machine and input.
Capture end-to-end time, stage time, overlap, bytes transferred, allocations,
index rebuilds, peak VRAM/RAM, throughput, and output hash. GPU kernel records
include ncu DRAM/SM throughput, occupancy, registers, spills, and roofline
percentage. I/O trials distinguish hot cache, cold cache, and storage limits.

Use robust statistics: warmup, at least 10 samples for short operations,
median plus p5/p95, CPU affinity, fixed clocks where available, and a noisy-run
rejection rule chosen before measurement. Per-PR smoke rejects regressions over
5%; nightly data retains raw samples for trend analysis.

For the default LAS translation slice,
`scripts/pdg/benchmark_translate.py` enforces exact process streams, status,
output size, and output SHA-256 on every warmup and measured invocation. It
alternates oracle/candidate order and records raw samples, executable hashes,
the Git working-tree snapshot hash, CMake flags, input/pipeline hashes, cache
state, CPU/GPU/toolkit/storage data, median, p5/p95, and throughput. A report is
not written if any exactness check fails.

Passing `--pipeline path.json` to the same runner benchmarks a pipeline instead
of `translate`. The runner replaces only the first reader and final writer
filenames for each isolated invocation, hashes the normalized pipeline, and
retains the same exactness precondition and alternating measurement order. A
final writer filename containing one PDAL `#` view marker is isolated as a
numbered artifact set. Every logical filename, byte count, and SHA-256 must
match across every warm-up and measured run; the report records each member,
the total bytes, count, and a canonical manifest SHA-256. Missing, extra,
renumbered, reordered, empty, or byte-different outputs invalidate the run.
`pdg_benchmark_runner_multi_output_contract` pins both legacy single-file
hashing and the numbered manifest behavior.

The shared-index primitive has a separate opt-in benchmark target:

```sh
cmake -S . -B build/pdg-cuda-release -DPDG_BUILD_BENCHMARKS=ON
cmake --build build/pdg-cuda-release --target pdg_spatial_benchmark
build/pdg-cuda-release/bin/pdg_spatial_benchmark \
  --profile clustered-tls --query covariance --points 1048576 \
  --neighbors 16 --warmups 2 --iterations 7 \
  --output build/benchmarks/pdg-spatial-g1-clustered-tls-cov-1m.json
sha256sum build/benchmarks/pdg-spatial-g1-*.json
```

Run both queries for all three deterministic profiles and the clustered
crossover sizes. The executable alternates backend order and refuses to report
unless complete output and status buffers match byte-for-byte. These resident
primitive timings diagnose G1 selection only; acceptance additionally requires
a clean tree, the full statistical protocol, real local ALS/TLS/mixed-density
corpora, and complete-process PDAL baselines with exact output artifacts.

The native CLI maps read-only input and writes directly into a caller-owned,
mapped output that remains unpublished until translation succeeds. Unit tests
initialize that output with nonzero bytes, require exact equality with the
vector-returning API, reject an incorrectly sized output span, and compare
automatic, one-worker, and many-worker results. The generated format-7 case
uses 131,073 points and an arithmetic scan-angle sequence spanning every
16-bit encoding. A 21,970,934-point local format-7 case is also compared in
both Release and host-sanitizer builds before performance results are recorded.

## Pipeline coverage generation

Maintain a stage/option interaction model rather than attempting an exhaustive
Cartesian product. Pairwise combinations cover ordinary interactions; targeted
three-way cases cover `where`/tiling/custom-dimension and
ordering/randomness/writer combinations. Real pipelines include DTM, feature,
denoise, downsample, COPC query, reprojection, and cross-cloud metrics.

Coverage is tracked across stages, options, point formats, dimension types,
execution modes, residency transitions, tile/halo paths, fallbacks, readers,
writers, and error categories. A stage is not complete if only its happy-path
kernel ran.

## CI cadence

- Every change: formatting, static checks, the published upstream PDAL suite,
  host unit tests, tiny differential smoke, and CUDA compilation.
- GPU pull request lane: exact stage replay, deterministic repeats, narrow
  performance smoke, memcheck, and racecheck for changed kernels.
- Nightly: stratified local corpus, all reference pipelines, full performance
  regression, multiple tile budgets, and PDAL baseline replay.
- Weekly: exhaustive generated boundary matrices, out-of-core stress, I/O
  failure simulation, cold-cache I/O, and supported compiler/toolkit matrix.

CI publishes machine-readable manifests and failure bundles but never raw
restricted input data.

## Resident neighborhood tie-repair lane

The force/experimental multi-bridge neighborhood lane differentially exercises
ordered CUDA kNN followed by exact host KD3 repair of only tied or incomplete
eigensystem rows. It includes deterministic ties, increasing-k stale-state
fallback, and real Autzen/RMIT fixtures. RMIT two-bridge repair is required to
pass memcheck, initcheck, synccheck, and racecheck; focused tie and increasing-k
process differentials run under memcheck. These are compatibility gates, not
automatic-selection evidence.

Internal proof switches make the sensitive paths fail closed. The deterministic
tie fixture sets `PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR`, which requires at least
one tied or incomplete row to pass through exact KD3 repair. The varying-k
fixture runs internal neighbor counts 6, 12, then 6 across two resident bridges;
the existing index/column gates plus
`PDG_REQUIRE_NEIGHBORHOOD_EIGENSYSTEM_REUSE=6` prove that the final covariance
client reused the first cache key. A `sqrt` assignment deliberately exceeds the
exact resident device-expression envelope and falls back to the exact host
bridge; `PDG_REQUIRE_NEIGHBORHOOD_BRIDGE_REBUILD` requires the following
neighborhood client to observe no stale resident object and build a new
batch/index. All switches are test-only environment controls recognized by the
thin dispatcher and do not alter public PDAL options or default selection.

B0066 adds a planner-resident statistical-outlier adapter without changing
default selection. The core physical wrapper/context/boundary regression
compares Classification and NNDistance to the pinned two-stage oracle,
requires resident reuse, and checks one index build plus one region begin/end
pair. Focused tests positively observe incomplete-row repair, prove retained
Classification refresh through a later device consumer, and reject an
undersized nonempty view. Rewrite coverage must prove the shared
maximum-neighbor envelope and keep radius outlier outside the kNN adapter.
Runtime preflight admits only statistical `mean_k=0..63`, budgets one binary64
mean plus one status byte per point, and rejects a nonempty view smaller than
the requested row so the original pipeline remains the exact fallback. PDAL's
stage runner skips filters and their resident boundary callbacks on empty
views, preserving the existing no-op without entering a delegated lifecycle.
B0066's dirty complete-process and same-binary timings are only a viability
gate. B0067 must add the clean endpoint profile, empty-process and option-edge
differential expansion, all applicable Compute Sanitizer tools, and an
alternating baseline before any performance qualification or automatic
placement model.

B0067 profiles that exact committed composition through a disposable placement
hook which must be absent from the accepted tree. The retained Nsight Compute
report must preserve the exact output hash and record every launch duration.
Its routing gate compares the maximum removable kernel span, not replay wall,
with B0066's complete-process median. Because eliminating the smaller repeated
gather has only a 3.437863% process ceiling, kernel fusion is rejected. B0068
may prototype the already-existing direct-LAS residency boundary only if it
adds exact Classification storage/overlay, compares complete output bytes, and
reverts unless the same 1M composition improves by roughly 5--10%. This is a
prototype gate, not automatic-selection or performance-qualification evidence.

B0068 retains that boundary only after a two-warmup/seven-sample alternating
same-binary gate clears 5--10% while every artifact remains byte-identical.
The checked-in physical process derives adversarial legacy packed flags/class
12 plus modern format 6, 7, and 8 inputs without modifying the source corpus.
It requires direct source/output, record-summary index configuration, no host
XYZ mirror, one index build, and the actual raw-record hydration event. Because
the direct executor is uncalibrated and its transfer is separate from generic
PointView boundary markers, both executor-calibration and boundary-accounting
match flags must remain false. Source-disabled and outside-envelope invocations
must fail without an artifact. This remains an explicit opt-in prototype; a
pinned-PDAL repeated baseline, clean provenance, sanitizer lane, and placement
calibration are still required before performance qualification or automatic
selection.

B0069 returns to a clean, already-positive catalog endpoint before authorizing
new code. One pinned-PDAL process and one resident/direct-output process over
the fixed 1M LAS must be byte-identical, and the fresh Nsight Compute report
must bind every launch duration to that same output. The routing decision uses
both device time and resident execution events: the full eigensystem/status
download and upload positively prove that ambiguity repair ran. B0070 may
replace that full round trip only with a sparse selected-row transfer that
retains pinned KD3 neighbor order, covariance/eigensolver results, failure
prefix behavior, diagnostics, and resident cache publication. Measure the
complete process; retain the prototype only near or above the 5--10% gate.

B0070 is the required negative-gate example: exact bytes and a 99.377286%
repair-transfer reduction are not sufficient when shell, command, and wrapper
wall improve only 0.491159%, 2.267040%, and 3.357725%. The bounded prototype
and its proof are reverted. B0071 must begin from the checked-in composed
normal/eigenvalues/covariancefeatures differential shape, measure complete
pinned-PDAL and candidate processes, and profile only a positive candidate.
Any later fusion must retain every intermediate error/diagnostic and the
writer-visible extra-dimension contract, not merely the final point-program
values.

B0071 proves the executor before changing placement. The forced public-shaped
composition must require both neighborhood and same-k eigensystem reuse, match
the pinned warning/output bytes, and bind the fresh profile to that output.
The ordinary resident probe must continue to report
`mixed_calibration_models` until a separate composition calibration exists.
B0072's ladder spans the retained 50K, 250K, 1M, 2M, 4M, 8M, and 16M inputs;
every candidate must exercise the forced shared region and remain byte-exact.
Fit and validate a separately named region model—never synthesize a match by
relabeling one per-stage model—and keep unrelated mixed-model cases negative.

B0072's seven raw reports are one-sample calibration rows, not a performance
claim. B0073 must pin each raw report hash in the calibration JSON, reproduce
the fit in `PlacementProfile.cpp`, and extend the model-count/audit tests.
Positive placement covers only the exact same-k three-consumer plus measured
three-assignment shape at 250K through 16M. Negative tests include changed k,
normal orientation, eigenvalue normalization, covariance mode/feature set,
assignment text/order/count, missing consumer, extra consumer, and unrelated
mixed-model regions. A physical explicit-resident process must prove accepted
preflight, one index, eigensystem reuse, warning stderr, and exact LAS bytes
before repeated qualification.

B0073 recognizes that shape from compiled programs and options rather than a
stage-name alias. The focused matrix includes changed k, normal orientation,
eigenvalue normalization, covariance mode/features, assignment operation,
order/count, missing/extra consumers, and an unrelated mixed-model stage. The
optional physical process gate requires a hash-pinned 250K--16M LAS source,
one index, 13-neighbor eigensystem reuse, accepted resident preflight, exact
warning stderr, and byte-identical output with first-byte diagnostics.

The retained clean 1M qualification alternates five pinned-PDAL and explicit
resident samples after one warmup. It qualifies only that exact complete
pipeline. B0074 then replaces B0072's forced rows with a clean resident ladder:
50K must remain an exact host-selected negative control, while every
250K-through-16M row must require the shared-index executor and both reuse
proofs. Refit from only those six selected rows, pin every report hash, rerun
the source/JSON matrix audit, and prove the physical executor-match diagnostic
before setting it true.

B0075 admits only the exact six-stage public JSON candidate, then independently
requires the compiled runtime placement to select the one
`eigen-family-compose` region. The host process gate proves a below-floor
required request returns 124 without output. The physical lane compares
explicit resident, required public, and genuinely option-free public execution
with pinned PDAL, and rejects changed k, assignment order, extra topology, and
injected preflight before output. The performance gate alternates one warmup
and five measured public-command pairs; the proof environment variable fails
closed but does not opt the pipeline into admission. The B0071 profile applies
unchanged because B0075 changes only admission and invokes the same executor.

B0076 uses an ordinary 1M ALS prefix and an observable three-field assignment
finale. The shared forced-hybrid path requires neighborhood reuse. A second
pipeline inserts the exact identity transformation between estimate-rank and
optimal-neighborhood; it must preserve the final LAS hash while splitting the
two neighborhood regions, making the same-process boundary comparison
reviewable. Standalone stage controls and a basic-set Nsight Compute profile
separate resident value from kernel value. These zero-warmup one-sample rows
are directional and do not performance-qualify or select the composition.

B0077 reuses the checked-in exact B0076 shape at seven bounded cardinalities.
Every forced candidate row requires shared-index reuse and matches pinned
status, stdout, stderr, and LAS bytes. The conservative 250K-through-16M rows
fit a distinct `rank-optimal-compose` residual; the positive 50K row is
below-fit validation. Existing estimate-rank and optimal-neighborhood
coefficients remain unchanged.

B0078 must pin every B0077 report hash and reproduce the fit in both source and
calibration JSON. Runtime placement may recognize only the exact compiled
estimate-rank/optimal-neighborhood/three-assignment shape. Positive and
negative unit cases cover k/range/threshold changes, assignment text/order,
missing or extra consumers, coordinate invalidation, and unrelated mixed
models. A physical explicit-resident process must require accepted preflight,
one shared index, neighborhood reuse, exact diagnostics, and byte-identical LAS
before a one-warmup/five-pair 1M qualification. Because the ladder provenance
is forced hybrid, the executor-match diagnostic remains false and public
automatic admission remains separate.

B0078 completes those gates. The physical process proves one selected region,
one predicted/observed index build, accepted preflight, required neighborhood
reuse, exact diagnostics, and exact LAS. The one-warmup/five-pair 1M lane is
8.285332x pinned PDAL. Because the selected executor has only one resident
cardinality while its fit rows remain forced hybrid, the executor-match
diagnostic must stay false.

B0079 replaces the seven forced cases with explicit-resident measurements.
The 50K row runs without an executor requirement and must remain exact and
host-selected. Each 250K-through-16M row requires
`planner_resident_shared_index`, accepted preflight, and neighborhood reuse.
Refit from only those six selected rows, pin every resident report, rerun the
source/JSON matrix audit, and require the physical executor-match diagnostic
to become true before any public selector is considered.

B0079 completes that replacement. The 50K host control and six selected
resident rows are all exact; every selected row proves the executor and reuse.
The source/JSON audit passes 126/126 decisions over 31 models, all 132 raw
report hashes verify, and the physical executor-match diagnostic is now true.

B0080 admits only the exact five-stage public JSON candidate, then independently
requires compiled runtime placement to select one `rank-optimal-compose`
region. The physical lane compares explicit resident, required public, and
genuinely option-free public artifacts with pinned PDAL; changed k/range/
threshold, assignment order, extra topology, and injected preflight must fail
before output. The performance gate alternates one warmup and five public-
command pairs. Its proof environment variable is fail-closed evidence, not an
opt-in. Direct LAS output remains disabled for this composition.

B0080 completes those gates. The clean option-free 1M process remains exact at
11.324354873 seconds pinned PDAL versus 1.388344338 seconds automatic PDG, or
8.156734x. The report requires both automatic rank/optimal admission and
neighborhood reuse. The B0076 profile still describes the identical executor;
its 3.557802% perfect single-gather-removal ceiling rejects further kernel work.
Future work must return to a measured existing-stage composition or residency
boundary before considering another stage port.

B0081 completes that first gate. The clean one-pair 1M process is exact and
2.752316x pinned PDAL with the required shared executor and one index build.
Because radiusassign is the sole neighborhood consumer, the old reuse proof
switch was a no-op and does not establish cross-stage reuse. The basic NCU
profile records 16 launches and 1.899456 milliseconds; the radius query is
only 0.234172% of complete wall. Ordinary stats instead identify 0.390594913
seconds of rewritten manager execution and 0.178538164 seconds of runtime
placement/setup. B0019's dirty ladder remains directional prior evidence, not
a current performance claim.

B0082 changes only the source/output boundary for the exact B0081 shape. Its
focused physical process gate requires direct mapped LAS source, record-summary
index configuration, no full host XYZ mirror, canonical UserData output,
shared-index execution, and exactly one index build. It runs both format-7 and
derived format-3 inputs so the four-bit and legacy three-bit ReturnNumber
encodings positively change UserData, then compares status, diagnostics, and
every output byte with the pinned oracle. Radius/domain drift and a disabled
source must fail before an artifact. The summary-based radius-grid builder is
independently bit-
compared with the PointBatch builder, including invalid and empty summaries.
A non-skippable LAS unit uses raw return bytes 9 and 15 to prove the legacy
three-bit and modern four-bit masks independently of the physical fixture.

Seven alternating same-binary 1M samples reduce exact median wall from
0.641774202 to 0.316225432 seconds (50.726372%). Memcheck, racecheck,
initcheck, and synccheck are clean on the required direct endpoint. Retaining
the prototype does not make it calibrated, performance-qualified, or public:
the process gate must continue to observe false executor-calibration and
boundary-accounting provenance until a direct-executor ladder replaces them.
B0083 owns that pinned-PDAL 50K-through-16M ladder and may fit only a separately
named exact-shape model; public admission remains a later gate.
This is a boundary diagnostic, not a pinned-PDAL speedup or qualification.

B0083 records one clean alternating pinned/direct pair per cardinality. The
50K row must remain exact, host-selected, and outside direct requirements.
Rows from 250K through 16M require the direct shared-index executor, output,
source, record-summary, no-host-XYZ, accepted-preflight, and exact-artifact
proofs. All selected rows are positive, so the clamped OLS fit may advance as
the separately named `radiusassign-direct` proposal. B0084 must add it without
replacing or relabeling the ordinary radiusassign calibration, prove exact-
shape positives and topology/option negatives in both profile
representations, and flip executor provenance only through those direct raw
rows. Repeated qualification and automatic admission are later gates.

B0084 stores the planner residual in both the compiled SM-89 profile and the
checked-in JSON record, while preserving the seven B0083 absolute-wall raw
rows and the independent ordinary radiusassign calibration. A unit reconstructs
the 250K absolute host/device predictions after planner infrastructure terms,
pins the 50K outside-envelope host control, and rejects option, topology, and
record-layout drift. The physical format-7 case must report both executor and
boundary provenance true; the exact format-3 companion reports boundary true
but executor calibration false because no format-3 timing row exists. B0085
owns repeated complete-process qualification and any automatic-admission
decision.

B0085 adds independent benchmark-runner requirements for resident executor-
calibration provenance and executor-declared boundary-accounting provenance.
Each timed direct-radiusassign candidate must report both values true; the
runner persists both observed values rather than inferring them from the
requested executor. One warmup and five alternating 1M pinned-PDAL/direct
samples must also match status, stdout, stderr, artifact count, and all LAS
bytes. A fresh Nsight Compute basic capture uses the same frozen-time/UTC
environment and must reproduce the retained output hash before its kernel
timeline is used for a stopping decision. A separate retained profile manifest
binds that environment and output to the input, pipeline, dispatcher/engine,
profile report, stats, observed executor, and proof values. B0085 does not
exercise or qualify option-free admission.

B0086 is a zero-warmup, one-pair routing measurement over the exact
statistical-outlier-to-kth-NNDistance direct Classification envelope. The
candidate must use one accepted whole-view region, one planner-owned index,
direct LAS source/output, record-summary configuration, and no host XYZ
mirror; status, diagnostics, and all output bytes must match pinned PDAL.
Because the lane remains explicitly uncalibrated, its executor-calibration and
boundary-accounting proof values must remain false. The fresh Nsight capture is
bound to the exact output by a separate hash-pinned manifest, as in B0085.

B0087 admits only a planner-owned ordered max-k neighbor superset. For the
named shape, outlier `mean_k=8` requests nine self-inclusive entries and kth
NNDistance `k=10` requests eleven. Positive proof observes one gather while
preserving the outlier mean's exclusion of self and NNDistance's eleventh
value. Fixed incomplete rows exercise each stage's existing host-repair
semantics; an asymmetric small-view case proves the producer uses its prior
exact query when the wider rowset cannot be materialized. An intervening
bridge, branch, incompatible dimension/region, missing producer, or wider
consumer receives no cache. The proof environment variable must fail closed
without nonzero planner provenance.

Host/device projection tests compare the exact bit patterns for online mean,
kth, and average modes on both uniform-grid and Morton-BVH rowsets. Allocation
size products are checked before launch; schedule tests pin `12*k + 1`
bytes/point and rewrite width; the product is released after the consumer's
projection/status copy. The focused neighborhood matrix, memcheck, and
racecheck cover ordinary reuse, incomplete repair, and the asymmetric fallback.

B0087's five-sample same-binary prototype decides retention but remains a dirty
direction gate. A clean-commit Nsight capture may support the stopping decision
only when a hash-pinned manifest binds the literal command, frozen environment,
input/pipeline/binaries, observed `knn_gather_reuse=true`, exact output hash,
report, and stats.

B0088 qualifies only the explicit named route. One warmup and five alternating
clean pinned-PDAL/direct samples require every B0068 direct boundary proof,
planner-owned max-k reuse, and prediction-matching integer one for predicted
and observed index builds. All processes must match status, diagnostics,
artifact count, and complete LAS bytes. The report must hash the dispatcher's
resolved sibling `pdg-engine`; that hash must equal the engine bound by B0087's
fresh exact profile before timing and profile evidence can compose.

B0089's placement ladder must use clean, alternating exact pairs over bounded
named sizes, retain all B0088 proofs, and fit a separately named composition
model. Qualification does not authorize automatic selection: require a stable
break-even, at least one below-threshold host control, at least one
above-threshold device control, exact option-free process gates if a selector
is proposed, and the updated model-count pin.

B0090 and B0091 treat incomplete consumers independently. A forced one-shell
fixture must make both the statistical-outlier mean and NNDistance kth
consumer incomplete while comparing Classification and NNDistance bits against
the pinned chained oracle. Each bounded device path has separate disable,
require-device, and require-parallel switches. Requirements fail when no row is
incomplete, the width/row/backend envelope declines, the device path is
disabled, or the matching stage is absent from an explicit resident selection;
no failure may create the output artifact.

Stats must expose nested `statistical_outlier` and `nndistance` objects under
both exact host and exact device repair. Their incomplete, repaired, and
parallel-repaired counts are integers rather than booleans. A required device
repair observes a positive equal incomplete/repaired count and zero matching
host repairs; a required parallel repair additionally observes the same
positive count in `parallel_repaired_rows`. The benchmark runner must scrub all
repair overrides from ambient state, set only requested proofs, validate the
observed telemetry, and retain it in the report.

The selective repair boundary matrix includes `k=1`, the maximum admitted
`k=15`, multiple partial partitions, the partition cap, exact large-origin
coordinates, disabled paths, complete rows, average mode, too-wide requests,
more than 16 incomplete rows, and unsupported BVH repair. Host Debug, physical
CUDA Release, process/runner tests, and focused memcheck, racecheck, initcheck,
and synccheck are required after changing default repair selection.

A repair profile informs a stopping decision only when a hash-pinned manifest
binds the literal Nsight command, frozen environment, clean commit, input,
pipeline, dispatcher/engine/oracle/frozen-time binaries, observed per-consumer
repair proofs, exact output hash, report, and stats. B0091's final profile uses
that contract. With all non-gather kernels below a one-percent complete-wall
ceiling, further endpoint tuning stops; the remaining shared gather requires a
new generic measured hypothesis rather than an inferred optimization.

B0092 calibration adds 8M/16M repaired rows only after both consumers prove
parallel device repair and zero host repair. The separately named composition
model must reproduce every retained absolute-wall row, retain a below-floor
host control, and pin its minimum/maximum point counts independently from the
ordinary outlier and NNDistance models. The strict runtime matcher must reject
stage/option/mode/topology/layout drift, a non-36-byte input record, anything
other than one resident region/index/lane, and any query product other than
the exact eleven-neighbor 133-byte/point layout. The placement audit and raw-
report verifier are required after changing either calibration surface.

Automatic qualification runs `gpupdal pipeline`, not diagnostic `gpupdal resident`.
The runner scrubs its require flag and injected-failure hook from ambient state
and sets the require flag only for candidate processes. A below-floor input,
option drift, disabled source, rejected preflight, and injected post-execution
proof failure must produce no artifact. Before automatic publication, runtime
must observe direct source use, record-summary index configuration, no full
host XYZ mirror, planner-owned max-k reuse, and exactly one index build even
when no separate proof environment variable was requested. One warmup plus
five alternating complete-process pairs must preserve status, empty
diagnostics, artifact count, and every LAS byte. B0091's hash-pinned profile
may support the stopping decision because B0092 changes only placement and
public admission around that same execution endpoint; no new kernel claim is
introduced.

B0093 applies the same admission-only discipline to `radiusassign-direct`.
The automatic require flag is candidate-only in `gpupdal pipeline`; both it and
the injected post-execution proof hook are scrubbed from ambient benchmark
state. The physical process lane must use an exactly 1,000,000-point pinned
source so its derived 50K below-floor control, 250K device-floor control, and
1M control cannot collapse to duplicate sizes. It also proves non-36-byte
layout and radius drift, disabled direct source, rejected preflight, and
injected proof failure without an artifact. Before publication the runtime
must observe direct mapped-source use, record-summary index configuration, no
full host XYZ mirror, and exactly one index build even without a separate
proof request.

Qualification uses one warmup and five alternating option-free candidate/
pinned-PDAL pairs under frozen UTC. Status, empty stdout/stderr, artifact
count, and every LAS byte must match, and the retained report must bind the
clean implementation commit, materialized pipeline, input, dispatcher,
resolved engine, oracle, and frozen-time library hashes. B0085's profile may
support the stopping decision only because B0093 changes no kernel, placement
coefficient, or execution boundary. The automatic claim stays within the
250K--16M `radiusassign-direct` model envelope and does not make the exact
host assignment-expression finale GPU-native.

B0094 records only clean repeated routing evidence for the explicit resident
`approximatecoplanar(knn=8) -> ferry(Coplanar=>UserData)` composition. One
warmup and five alternating clean pinned-PDAL/resident pairs must require the
direct-output shared-index executor and preserve status, empty diagnostics,
artifact count, and every LAS byte. Retained stats must report device
placement, one predicted/observed index, direct output, and the current false
executor-calibration/boundary-accounting values. Those false values are an
acceptance condition: the ordinary resident `approximatecoplanar` model may not
be represented as direct-composition provenance. Performance qualification
remains unchanged until the current output-bound profile exists.

A prior profile cannot support a current stopping or automatic-admission
decision unless a hash-bound manifest ties it to the current input,
materialized pipeline, dispatcher/engine/oracle/frozen-time binaries, stats,
and exact output. B0069 does not meet that current binding. B0095 adds
approximate-coplanar-specific repair telemetry for trigger, ambiguous and
incomplete rows, KD3 use, and repair transfer bytes. The repair-triggering
fixture positively observes those fields; the runner's required proof rejects
absent, malformed, non-finite, asymmetric, impossible-row, and contradictory
untriggered telemetry. The focused repair test is clean under memcheck and
racecheck.

B0095's fresh profile must reproduce B0094's output hash and bind the clean
implementation commit, materialized pipeline, dispatcher, resolved engine,
oracle, frozen-time library, profile/stats artifacts, and literal capture
command. The accepted trace observes 2,145 ambiguous rows, zero incomplete
rows, one KD3 use, and 97 MB in both transfer directions. Its 20 kernels total
21.403776 milliseconds, while exact host repair is 0.461688645 seconds of the
0.869924635-second unprofiled command interval. Together with B0070's failed
selective-transfer prototype this closes the roughly 5--10% optimization gate.
B0096 must measure current-binary paired pinned-PDAL/direct rows before either
performance qualification or a separately named direct-composition fit.
Calibration must remain isolated from the ordinary resident model, and
automatic selection still requires a later option/topology/fail-closed public
process gate.

B0096's accepted direct approximate-coplanar calibration uses one warmup and
three alternating current-binary pairs at 50K, 250K, 1M, 2M, 4M, 8M, and 16M.
The 50K row is a required host-selected negative control. Every fitted row must
be byte- and diagnostic-exact, require the
`planner_resident_shared_index_direct_las` executor, direct LAS output,
positive approximate-coplanar host repair, and exactly one predicted/observed
index. Fit only the positive 250K--16M rows after subtracting planner-owned
startup, transfer, packing, index, and synchronization terms. The compiled
model and raw calibration JSON must agree exactly; the placement audit and
`verify_placement_calibration.py --require-all` are merge gates.

Runtime tests for this model must prove ordinary-model isolation and reject
changed `knn`, either threshold, ferry mapping/topology, input record layout,
index-build layout, and counts outside 250K--16M. A selected direct-output
process must report executor-calibration match true while continuing to report
boundary-accounting match false; the latter is an intentional diagnostic of
calibration-default byte accounting, not permission to claim direct-source
execution. B0096 itself adds no automatic selection; B0097 separately proves
the public gate below.

B0097's automatic admission retains process negatives for options, topology,
both thresholds, layout, cardinality, device, compressed source/output,
preflight, and post-execution proof failure without publication. Each
pre-execution negative also runs option-free and byte/diagnostic-compares the
unchanged fallback with pinned PDAL. Its positive gate uses the ordinary
`gpupdal pipeline` command, reproduces B0096's exact output and diagnostics,
observes one index and positive pinned-KD3 repair through a fail-closed runtime
proof, and retains a same-machine pinned-PDAL baseline plus output-bound
profile. Any expansion beyond the measured default shape requires a new ladder
rather than inference from this model.

B0098 first repeats the ordinary calibrated `filters.neighborclassifier(k=7)`
route at 1M, then profiles it before changing execution. The retained ordinary
pair must prove the resident shared-index executor, one index, exact output,
and calibration provenance; its profile binds the same output and names the
device ceiling. The direct-boundary prototype may be retained only after a
clean current-binary pair requires mapped LAS source, record-summary index
configuration, no host XYZ mirror, canonical Classification publication, and
one index. Changed `k` and disabled source must fail closed without an output.
B0098 does not fit or borrow a model and does not enable automatic selection.

B0099 started with telemetry, not a new stage or general query kernel. It
counts and times conservative neighbor-vote tie repair without adding a
synchronization. Its cheap prototype may avoid pinned KD3 construction only
when it proves the
vote invariant across every candidate at the tied boundary on the existing
planner-owned query. Compare the full output and diagnostics with pinned PDAL,
include a positive non-invariant tie that still takes exact repair, and retain
the optimization only if a bounded complete-process gate improves roughly
5--10%. Any wider CUDA tie semantics, calibration, or public selector requires
separate evidence.

B0099's retained production stats must expose neighborclassifier repair time,
tie/incomplete/repaired rows, and KD3 use, with both positive-tie and zero-row
fixtures. Its cheap prototype is accepted as a stopping experiment, not a new
runtime envelope: interior and full-boundary proofs may bypass only rows whose
integer vote is invariant for every tied selection, while any non-invariant
row must still take pinned KD3. The measured 1M fixture leaves one such row and
therefore no 5--10% complete-process gain; prototype code is reverted and no
performance or automatic-selection category changes.

B0231 promotes only B0098's already-exact direct-boundary route. Its placement
calibration must retain the seven-row 50K--16M ladder, use the separately fitted
`neighborclassifier-direct-compose` model, and keep the measured 50K loss
outside the 250K floor. Runtime units must prove exact model coefficients,
25-byte upload, one-byte spill, zero packing, 112-byte shared-index accounting,
one lane, the 16M cap, and isolation from the ordinary `neighborclassifier`
model. Changed stage count/order, `k`, LAS format, record width, compression,
or index facts must remain host-selected.

The automatic process matrix must exercise the public `gpupdal pipeline` command,
not only an explicit resident subcommand. Positive 250K and 1M cases require
the selected placement, executable rewrite, mapped LAS source, direct
Classification output, record-summary index configuration, no host XYZ mirror,
one selected region/index, and exact output/diagnostics/status/order. A 50K
control plus disabled-source and injected-preflight negatives must decline
before commitment with status 124 and no output when the automatic route is
required. Format, `k`, compression, and writer-option drift follow that same
pre-commit refusal. The injected post-execution proof failure is different: it
must exit 1 with the exact proof diagnostic and no publication, never retrying
the already-executed pipeline on the host. Memcheck and racecheck run the 250K
floor; expansion requires another complete layout/cardinality ladder rather
than borrowing the ordinary stage model.

B0100 began with a clean current-binary pinned-PDAL/PDG pair for the existing
`filters.radialdensity(radius=1.01)` 1M complete pipeline and an output-bound
profile reproducing the exact LAS and diagnostics. Only if that profile shows
a material rebuild, transfer, or residency surface may a prototype compose a
compatible radius consumer on the planner-owned index. A dirty historical
speedup may prioritize this measurement but may not qualify or select it.

B0100's accepted record is five alternating exact pairs plus an output-bound
NCU capture under the same frozen UTC environment. It performance-qualifies
only the named 1M format-7/radius-1.01 forced-CUDA pipeline. Because all kernels
sum to 0.449921% of candidate wall, B0101 must target the complete host boundary
rather than the radius primitive. Its first prototype should be a strict
resident `RadialDensity` producer feeding an exact ferry/assignment consumer
and canonical LAS output through the existing planner-owned radius product.
Positive execution must prove one selected region/index, resident column
liveness, no intermediate spill/re-upload, exact output/diagnostics, and atomic
publication; unsupported options/topology/layout must fall back before side
effects. Retain it only after a current-binary 5--10% complete-process win.

B0101 accepts that prototype only for the exact 1M VLR-free format-7
`radialdensity(radius=1.01) -> assign(UserData=1 where RadialDensity>=0.2)`
shape. The qualification record must contain five alternating pinned-PDAL/
resident pairs, a second five-pair same-binary hybrid control, and an
output-bound NCU capture that reproduces the retained output hash under the
same frozen UTC environment. The direct proof requires one selected region and
index, selected stage ids for the producer and point program, record-summary
source configuration, no host XYZ mirror, no intermediate binary64
RadialDensity download, the one-byte `UserData` result download, matching
boundary accounting, and atomic LAS publication. Unsupported radius,
assignment, profile, source, output, topology, or layout must fail closed or
retain the unchanged exact route. This evidence does not authorize a placement
model or automatic selection.

B0102 returns to an unmeasured compiled catalog surface. Start with physical
and sanitizer qualification plus a clean pinned-PDAL baseline/profile for a
bounded direct-LAS `decimation(step=2) -> assign` pipeline. The hypothesis is
that an exact fused ordinal compaction/assignment/publication region avoids a
PointView boundary and materially improves complete wall. Measure before
retaining code; kernel timing alone does not qualify the stage, and no selector
or calibration ladder starts unless the exact complete-process prototype wins
by roughly 5--10%.

B0102's accepted coverage record is explicitly non-performance evidence. The
existing ordinal device/ordered-LAS units and process gates must cover
decimation/head/tail, standard/streaming modes, chunk boundaries, split graphs,
global sequence, and exact output, followed by memcheck, initcheck, racecheck,
and synccheck. One bounded 1M profile must reproduce the directional output and
attribute decode, ordinal mask, stable compaction, assignment, and repack. A
larger direction point may test amortization, but one-shot negative rows are
diagnostics only. If CUDA loses and a reversible chunk-size control does not
help, mark the exact envelope GPU-native force-only and keep performance and
automatic selection false.

B0103 applies the same loop to the already-compiled first-tie
`filters.locate` reduction. Physical exactness and all four sanitizer tools
precede timing. Measure pinned PDAL and profile the complete current candidate;
only prototype a resident terminal consumer if the profile names a material
boundary that can plausibly clear roughly 5--10%. Preserve typed values,
sentinels, NaN/infinity behavior, original source index, and first-tie order.

B0103's accepted record requires the two typed/coordinate device units, three
forced-CUDA process graphs, and all four sanitizer tools. Its directional
composition is `assign(UserData=1) -> locate(Z,min)` so the terminal selected
row positively preserves an upstream point operation. The output-bound profile
must reproduce the one-point artifact and separate assignment, block reduction,
and final reduction from non-kernel wall. Negative one-pair rows may establish
forced native coverage and reject optimization, but never performance
qualification. A direct/resident hypothesis advances only if a cheap prototype
proves a complete-process 5--10% win.

B0104 next remeasures the existing `filters.info` reduction. Physical device
and sanitizer gates precede timing; the current-binary profile must distinguish
reduction work from reader/writer and metadata publication. Prefer reuse of the
direct LAS record summary over a new kernel or private summary. If that path
does not clear the complete-process gate, keep exact CUDA force-only and move
on without a model or selector.

B0104's accepted record requires the 131,103-point device property, one
forced-CUDA process differential that compares metadata as well as artifacts,
and all four sanitizer tools. The 1M/4M directional rows remain diagnostic
unless a repeated current-binary gate wins. The output-bound profile must
reproduce both LAS and metadata hashes. Compare exact PDG-host info with the
same-binary reader/writer-only graph before adding record-summary plumbing;
that measured delta is the maximum reusable-summary value in the ordinary
pipeline. A result below 5--10% closes the hypothesis without code or selector
changes.

B0105 next physically rechecks `filters.hag_nn(count=4)` as the strongest
historical unqualified shared-index lane. Preserve the masked 2D index,
ground/non-ground split, count-four ordering, exact distance-tie and incomplete
repair, original point order, and `HeightAboveGround` bytes. Measure and
profile the complete current pipeline before implementation; dirty historical
speedup is only a hypothesis, never qualification or automatic admission.

B0105's accepted record requires the 12-test count-four/masked-index/resident
lifecycle lane under physical execution and all four Compute Sanitizer tools.
Performance qualification requires five alternating current-binary pairs on
the controlled distinct-fifth-candidate fixture plus an output-bound profile;
it remains specific to that forced fixture and never implies ordinary-data or
automatic-selection coverage. A HAG-shaped trivial-publication control bounds
the remaining boundary/index-lifecycle opportunity.

B0106 may add only an explicit experimental force for the existing count-four
plan through the ordinary planner-resident executor. It must preserve one
planner-owned masked 2D index, exact host repair, the 48-byte-record output,
and unchanged diagnostics. Prototype with one exact pair before broader tests;
retain the path only for a 5--10% complete-process win. A direct mapped source
or new extra-dimension publisher is outside this first prototype, and no model
or option-free selection follows.

B0106's accepted stopping record requires the prototype to be fully reverted
when that pair misses the gate. Preserve the exact output hash and execution
proof, but label the one-pair timings directional and unqualified.

B0107's retained dimension-generic publisher accepts exactly one canonical
unsigned-32 `OffsetTime` Extra Bytes descriptor on LAS 1.4/format 7, preserves
the original 40 record bytes, appends raw binary64 bits, regenerates header
bounds/return counts, and publishes atomically. Unit coverage includes empty,
one-row, descriptor/padding/layout/cardinality/overflow rejection,
caller-buffer parity, signed zero, infinity, and distinct NaN payloads. The
physical process gate compares full pinned-PDAL bytes, streams, and status;
proves one index, zero terminal/fallback spill, experimental-only activation,
unforced fallback, and no partial output; and passes all four Compute
Sanitizer tools. The retained 0.98-second PDAL, 0.64-second same-binary hybrid,
and 0.59-second direct trio proves the 7.8125% retention gate but remains
directional rather than a five-pair qualification. The output-bound profile
must reproduce SHA-256
`6e59b3a5b41cc5e219a5d39743c39ac7b2f09c84aa74b67e9b943e4cf41fe293`
and bind its command, binaries, pipeline, stats, and report.

B0108 optimizes only the measured explicit-force placement/preflight interval
in a disposable same-binary prototype. It preserves the strict envelope,
device-memory feasibility, planner boundary accounting, fail-closed proof, and
ordinary calibrated/automatic behavior, but its exact stats-free pair improves
only from 0.55 to 0.53 seconds (3.636%). The prototype must therefore be fully
reverted and the endpoint stopped. Retain the separate regression proving that
an experimental request with an unsupported Extra Bytes descriptor delegates
to pinned PDAL with identical bytes, streams, and status.

B0109 measures pinned PDAL and the current exact
`readers.las -> filters.nndistance(k=10) -> writers.las(extra_dims=all)` graph
on the canonical B0107 input, then captures a complete-process NCU profile.
The oracle/hybrid artifacts must be byte-identical; the output-shaped control
is a lower-bound shape and intentionally has different values. The retained
2.01/0.66/0.39-second directional rows and 25.767520-millisecond kernel total
clear only the cheap prototype gate, not qualification or automatic admission.

B0110 extends only the explicit Extra Bytes force to NNDistance
`mode=kth,k=10`. Its full process differential proves raw binary64 output
bytes, streams/status, one planner-owned index, terminal-spill elision,
positive values, required rejection, and experimental-only exact fallback for
unsupported mode/k/writer shapes. Existing atomic/no-overwrite and strict LAS
layout/descriptor coverage remains shared with B0107. The clean one-pair
0.71/0.57-second result clears only the retention gate. Focused ASan/UBSan and
all four Compute Sanitizer lanes must remain clean; no automatic admission or
performance qualification follows from the directional pair.

B0111 may enable the existing mapped LAS resident source only for that same
explicit B0110 envelope. Its positive process proof must require observed
direct source use, record-summary-configured shared index, no host XYZ mirror,
one index, direct Extra Bytes output, zero terminal/fallback spill, and the
exact output hash. Add negative source/layout/proof-injection cases with no
artifact. Run a same-binary source-disabled/source-enabled pair first and
retain only for another 5--10% complete-process win; calibrated and automatic
admission remain unchanged.

B0111 retains that bounded source composition after the clean pair improves
from 0.58 to 0.38 seconds and the required process, ASan/UBSan, and all four
Compute Sanitizer proofs pass. Its fresh exact profile totals 26.086656
milliseconds/6.865% of wall, almost all in the existing shared gather; the
B0108 startup/placement negative and the 14.145771-millisecond publication
interval leave no clear reusable 5--10% surface. Treat the endpoint as
sufficiently optimized and do not infer qualification or automatic selection
from the directional pair.

B0112 repeats pinned PDAL and the current exact
`readers.las -> filters.hag_delaunay(count=3) ->
writers.las(extra_dims=all)` complete process on B0034's deterministic strict
one-extra-dimension fixture. Its exact 0.810889333/0.607849511-second pair and
2.342688-millisecond profile replace the dirty-snapshot planning evidence. A
0.40-second same-binary output-shaped control exposes a 34.1942% non-kernel
boundary, so B0113 may prototype existing source/publisher reuse.

B0113 admits only count-three HAG Delaunay on that strict layout to the existing
mapped source and generic one-binary64 publisher. Its positive process test
must require observed mapped-source use, record-summary-configured masked 2D
index, no host XYZ mirror, one predicted/observed index, direct Extra Bytes
output, zero terminal/fallback spill, exact HAG bits, and atomic publication.
Tie/incomplete repair and unsupported source/options/layouts must retain the
prior exact path or fail without an artifact when required. Run a current
same-binary disabled/enabled pair before certification and retain only for a
5--10% complete-process win; do not change automatic selection or a kernel.

B0113 retains that bounded source composition after the clean pair improves
from 0.55 to 0.41 seconds and the required physical process, focused
ASan/UBSan, and all four Compute Sanitizer proofs pass. Its fresh exact profile
totals 2.411420 milliseconds/0.588151% of wall; the output-shaped control
removes the whole spatial query and B0108 already rejects the apparent
placement shortcut below 5%. Treat the endpoint as sufficiently optimized.
This is directional retention evidence, not performance qualification or
automatic selection.

B0114 may enable the same mapped source only for B0107's already-exact strict
HAG-NN count-four direct one-binary64 endpoint. First extend the process proof
to require direct source use, record-summary-configured masked 2D index, no host
XYZ mirror, one index, exact native and repaired HAG bits, zero spill, and
atomic output. Unsupported inputs/options and a disabled required source must
fail without an artifact. Run one current-binary source-disabled/source-enabled
pair before retaining the path and require a 5--10% complete-process win. No
kernel, placement model, calibration, or automatic selector may change in this
prototype.

B0114 retains that bounded composition after the clean pair improves from 0.56
to 0.37 seconds and the required process, focused ASan/UBSan, and all four
Compute Sanitizer proofs pass. Its fresh exact profile totals 3.523550
milliseconds/0.952311% of wall; publication is 12.601495 milliseconds and
B0108 already rejects the apparent placement shortcut below 5%. Treat the
endpoint as sufficiently optimized. B0105's prior named count-four
qualification remains unchanged; this one-pair source result adds no
qualification or automatic selection.

B0115 measures the current exact affine `filters.transformation -> pointwise
assignment` fused pipeline before code. Use a deterministic nonidentity affine
matrix and an assignment whose output is serialized, capture the pinned PDAL
artifact first, and run one warmup plus a same-machine directional pair. The
candidate proof must require the fused CUDA path and preserve simultaneous XYZ
semantics. Profile the exact output and name the dominant complete-process cost.
Only a clear reusable 5--10% fusion or residency surface may justify a
prototype; otherwise record the negative and move on.

B0115's exact current-binary directional result is negative: pinned PDAL takes
0.35 seconds and the required descriptor-fused CUDA hybrid route takes 0.60
seconds on 1,000,002 points. The fresh exact-output profile records eight
tiles, each with separate unpack, transformation, assignment, and repack
launches, but all 32 kernels total only 0.849310 milliseconds/0.141552% of
candidate wall. This bounds perfect device-launch/kernel fusion far below the
5--10% process gate. No implementation or selection change is justified.

B0116 measures the existing exact hybrid/resident candidate
`filters.label_duplicates -> filters.nndistance(k=10) -> pointwise UserData`
pipeline before code. Capture the pinned output first, require the selected
CUDA route, and compare complete reader/stage/writer wall time. The
bounded hypothesis is that adjacent-predecessor labeling can piggyback on the
already valuable shared-index region without a material spill or publication
boundary. Profile only after exact execution is proved; retain work only for a
reusable 5--10% complete-process opportunity.

B0116's current-binary gate proves the forced per-stage hybrid chain is
valuable without a new
kernel. One warmup per executable followed by five alternating exact pairs on
1,000,000 points measures 4.374096348 seconds pinned PDAL and 0.645599567
seconds required hybrid CUDA, or 6.775247x. Every output, stream, and status
matches. The output-bound profile reproduces that hash; its 20 kernels total
28.478630 milliseconds/4.411191% of candidate median, including only 0.010810
milliseconds for duplicate labeling. The hybrid label wrapper copies its byte
column back into the host PointView; these runs do not prove resident reuse.
This qualifies only the named explicit forced-hybrid chain. A
`gpupdal resident --stats` preflight separately proves that actual resident
placement declines with `missing_calibration_model`.

B0117 first uses a disposable strict one-shape placement prototype to run the
actual planner-resident graph at 1M. Require selected-region, one-index,
accepted-preflight, boundary, and exact-output stats and compare it with the
B0116 hybrid control. Only a 5--10% complete-process win justifies the 50K,
250K, 1M, 2M, 4M, 8M, and 16M calibration ladder and strict combined model.
Otherwise revert the prototype and keep the exact forced hybrid/host routes.
No kernel or wider stage envelope changes in B0117.

B0117's disposable prototype is exact but negative. It temporarily anchors
the label stage to the existing NNDistance model, then requires an actual
`gpupdal resident --stats` execution with one selected region, stages 1--3, one
observed index, accepted preflight, and no fallback boundary. After one
correctness warmup, the 0.70-second confirmation is 8.43% slower than B0116's
0.645599567-second hybrid median. Stats also reject the borrowed accounting:
observed H2D/D2H are 35/11 MB versus predicted 26/10 MB. Revert the prototype,
restore the clean engine hash, and skip the ladder/model/selector.

B0118 measures the current exact 2M `filters.mortonorder ->
filters.assign(UserData=17)` complete pipeline with one warmup and five
alternating pairs per executable. The forced CUDA wrappers must prove both the
Morton and point-program device paths; every artifact, stream, and status must
match pinned PDAL. The retained pair/profile gate measures 1.252847x pinned
PDAL, while all 29 kernels total 0.618688 milliseconds/0.068310% of candidate
median. A second exact five-pair Morton-only control bounds the whole adjacent
assignment stage to 0.033078598 seconds/3.790666%, below the 5--10% threshold.
No fusion or placement prototype follows.

B0119 measures the exact 2M `filters.mortonorder ->
filters.head(count=100)` graph with one warmup and five alternating pairs per
executable. The required CUDA wrappers must prove both stages; every artifact,
stream, and status must match pinned PDAL. The retained clean gate is 1.395295x
PDAL, while all 61 kernels total 0.688512 milliseconds/0.119095% of candidate
median. A disposable four-line upper-bound hook truncates the CUDA Morton
permutation to 100 ids before PointView publication. Its separate exact
five-pair median is only 4.327416% below the clean candidate, with overlapping
sample ranges. Revert it, rebuild the clean engine, verify the original hash,
and do not advance to a fused ordinal path.

B0120 runs the first physical forced-CUDA measurement of finite basic
`filters.stats`. The deterministic 2M six-dimension graph must compare full
LAS bytes, metadata, streams, and status with pinned PDAL before timing. The
exact current route measures only 0.428539x PDAL. A bare pinned control leaves
0.057345997 seconds of CPU stats work, while the fresh profile records
498.966048 milliseconds in the ordered CUDA recurrence alone. Six dimensions
produce six blocks/0.01 waves per SM because within-dimension order cannot be
relaxed. Stop the stats endpoint without a fusion prototype or selector.

B0121 uses the existing deterministic count-two HAG-NN fixture for two clean
five-pair gates. First require hybrid CUDA and compare the complete
`extra_dims=all` LAS, streams, and status with pinned PDAL. Then admit only
the same strict LAS 1.4/format-7/40-byte input, canonical one-`OffsetTime`
descriptor, `count=2`, and `extra_dims=all` writer through the mapped source
and one-binary64 publisher. The direct gate must fail closed unless stats
prove its executor and `fused_point_program` schedule, accepted resident
preflight, exactly one planned/observed index, direct source/record summary,
no host XYZ mirror, direct output/Extra Bytes publication, terminal-spill
elision, matching boundary accounting, and zero fallback spill. Count-two tie
and forced-grid incomplete-search fixtures must still reproduce pinned host
repair bytes, streams, status, atomic publication, and cleanup; disabled or
unsupported mapped sources and count one/three must reject before output.

Run focused process and runner contracts plus all four Compute Sanitizer tools.
Bind the clean five-pair report and fresh Nsight capture to the exact profile
output and an unprofiled phase record. B0121's accepted medians are
0.951975379 seconds pinned PDAL and 0.365209389 seconds direct resident
(2.606656x); all kernels total 2.552608 milliseconds/0.698944%. Publication
and every separate host-boundary surface remain below 5%, so stop the endpoint
and keep it explicit. B0122 applies that same protocol to count three. Its
clean forced-hybrid/direct medians are 0.620312799/0.365397468 seconds, every
output is exact, all four Compute Sanitizer tools are clean, and the fresh
profile totals 3.076512 milliseconds/0.841963% of wall. Count-three native,
tie/incomplete repair, mapped-source rejection, direct publication, and
unsupported count-one/count-five failure are required in the focused process.
Stop the explicit endpoint because publication and every host-boundary surface
remain below 5%. B0123 next runs the count-one current-clean forced-hybrid
gate before any direct-envelope edit. B0123 then applies the same protocol:
the strict option-free direct producer admits count one, while count zero,
count five, and an option-bearing count-one producer must reject before output
or temporary publication. Native, boundary-tie, forced-grid incomplete repair,
mapped-source rejection, exact atomic output, one-index proof, and all four
Compute Sanitizer tools pass. The accepted clean forced-hybrid/direct medians
are 0.629955252/0.360501731 seconds, every output is exact, and the fresh
profile totals 2.067488 milliseconds/0.573503% of wall. Publication is
4.553248% and the measured host-boundary sum is 3.797665%, so the explicit
endpoint stops.

B0124 measures the already-native standalone statistical-outlier graph before
the direct-output change. Five alternating exact 1M pairs for
`method=statistical,mean_k=8,multiplier=2,class=7` establish 6.521497x pinned
PDAL in forced hybrid. The pre-change output-bound profile totals only
21.501856 milliseconds/3.506149% of wall, so the retained prototype targets
the mapped-source/Classification-publication boundary, not a new query kernel.
Five clean direct pairs then reach 10.739669x pinned PDAL and reduce wall
39.020273% from hybrid. Require strict stage/options matching, accepted
whole-view-neighborhood preflight, one planned/observed index, mapped record
summary, no host XYZ mirror, the exact nine-byte-per-point mean-distance/status
download required by the host-order finale, exactly one one-byte-per-point
Classification boundary spill, matching boundaries, exact atomic LAS bytes,
and zero fallback. Forced-grid incomplete search must reproduce pinned host
repair; changed `mean_k`, multiplier, class, added options, or disabled mapped
source must reject before output or publisher temporary creation.

Run the focused physical matrix, host ASan/UBSan, and all four Compute
Sanitizer tools. Bind the paired reports and both fresh profiles to the exact
output and unprofiled phase record. B0124's direct profile totals 21.450304
milliseconds/5.735910%; its required gather alone is 21.093440
milliseconds/5.640483%, while all other kernels total 0.095427%. Publication,
hydration, row materialization, and the residual wrapper ceiling after the
gather are each below 5%. Stop the explicit endpoint and add no standalone
selector.

B0125 applies the measure-first protocol to the already-native standalone
`outlier(method=radius,radius=1,min_k=2,class=7)` graph. One warmup and five
alternating exact pairs bind both the pre-change forced-hybrid control and the
clean implementation route to pinned PDAL. The direct proof requires one
planner-owned 3D radius index, mapped record summary, no host XYZ mirror, an
exact four-byte-per-point count result, a one-byte Classification spill,
matching boundary accounting, zero fallback, and atomic canonical output.
Formats 3/6/7/8, positive mutation, the strict distance boundary, source and
option rejection, host ASan/UBSan, and all four Compute Sanitizer tools form
the acceptance lane.

The output-bound direct profile must reproduce the retained output hash. It
records 2.972310 milliseconds across all 18 launches, only 0.777018% of the
0.382528055-second candidate median; the radius query itself is 0.690145%.
The matching unprofiled stats record publication, hydration, row
materialization, and the residual wrapper ceiling separately. Each remains
below 5%, so B0125 stops as sufficiently optimized and adds no selector.

B0126 accepts only the exact 1M same-radius composition
`outlier(radius=1.01,min_k=2,class=7) -> radialdensity(radius=1.01) ->
assign(UserData=1 where RadialDensity>=0.2)` through an explicit strict gate.
The physical proof must compare full LAS bytes, stdout, stderr, and status to
the pinned oracle; positively mutate both Classification and UserData; observe
one whole-view region and one planner-owned 3D index; retain RadialDensity on
device; declare the four-byte radius-count result separately from the ten-byte
logical terminal boundary; observe only count plus final UserData D2H; and
reject shape/source drift before output or publisher temporary creation.
Current B0126 passes that process, host ASan/UBSan, and all four Compute
Sanitizer tools. Five paired rows and an output-bound profile qualify only the
named explicit 1M hybrid/direct routes; automatic selection remains off.

B0127 measures 50K/250K/1M/4M current-binary hybrid/direct rows for this same
exact graph. The model uses only the 250K/1M/4M clean direct rows; 50K remains
an exact direction row because its one small, noisy 5.644% direct advantage is
not a stable break-even proof. Automatic acceptance is therefore limited to
250K--4M, LAS 1.4 format 7 with 36-byte records, the exact five-stage graph and
options, and the pinned calibrated SM89 profile. The option-free process matrix
must prove exact selected outputs at 250K/1M/4M, exact below-threshold and
shape-drift fallback, required-path failure before side effects, one observed
index, mapped source/record summary, no host XYZ mirror, successful resident
assignment, matching boundary accounting, and atomic publication. The frozen
placement audit must remain exact; current B0127 passes 152/152 over 35 models.
Default-mode benchmark reports do not embed resident stats: they must record
the automatic require flag as true and every candidate status as zero, while
the runtime makes that status contingent on the full observed proof. A focused
injected-proof failure must return nonzero with no output artifact.

The fresh automatic profile must run with the same frozen environment and
required selector proof and reproduce the accepted 1M output. It records 20
launches/5.654080 milliseconds, of which the two radius-count kernels consume
5.299264 milliseconds/93.724602%. Because this is the already-stopped B0126
device shape and no new reusable 5--10% surface appears, stop endpoint tuning.

B0128 runs one warmup and five alternating current-clean pairs for the exact
comparator-unique 1M `filters.skewnessbalancing` pipeline. The retained report
must bind full LAS output, stdout, stderr, status, binaries, input, materialized
pipeline, and frozen time. Because the exact route remains positive, its NCU
capture must reproduce the accepted artifact. A separate forced-host control,
bare read/write control, and bounded Release-symbol phase profile distinguish
device ordering from host boundary cost; the controls are directional and
must not be represented as an additive wall-time decomposition. B0128 meets
this gate at 1.192843x pinned PDAL, about 0.258730 milliseconds of kernels,
and a 245.294974-millisecond filter call whose recurrence and PointView
permutation consume only 18.895269 milliseconds combined.

B0129 retains only the strict format-7/36-byte comparator-unique graph through
an explicit required path. It reuses the mapped source and a permutation-aware
canonical LAS publisher while retaining the existing CUDA ordering and
byte-for-byte host recurrence. Differential cases cover the accepted
permutation and Classification bytes plus ties, nonfinite Z, unsupported
options/layouts, source/output aliases, existing/symlink output, disabled
source, injected proof failure, generic-direct-output isolation, preflight
fallback, and atomic cleanup. Host Debug and Release focused tests,
ASan/UBSan, the physical process gate, and all four Compute Sanitizer tools
pass.

The current-clean five-pair result is 1.229485169 seconds pinned PDAL versus
0.425794757 seconds direct, or 2.887507x, and direct is 61.235805% below
B0128's forced-hybrid median. The fresh output-bound profile reproduces the
exact artifact and records 13 launches/0.260192 milliseconds. A separate
unprofiled stats pass binds the observed direct proof and records 0.032016684
seconds of canonical publication. Its 0.166260729-second device/profile plus
initial-placement interval includes first CUDA context/free-memory work;
B0108's already-reverted shortcut proves that moving this attribution does not
clear 5%. With no independent reusable 5--10% surface, stop this explicit
endpoint. Do not fit or automatically select it without a separate 10% gate
and bounded cardinality ladder.

B0130 remeasures the current exact `filters.sort` route before code on a
deterministic nonidentity, finite comparator-unique 1M Z fixture whose full LAS
bytes expose permutation. Five clean pairs measure 1.562019960 seconds pinned
PDAL versus 1.117247334 seconds forced CUDA, or 1.398097x. A separate exact
same-binary host control is faster at 0.903809508 seconds; future retention
must beat that control rather than merely PDAL. The fresh output-bound profile
reproduces the accepted artifact and records 13 launches/0.271040
milliseconds. A bounded Release-symbol profile separates the 0.204735422-
second filter into 0.112223505 CUDA wrapper, 0.007759453 full-view
permutation, and 0.084752464 remaining gather/setup. This selects boundary
reuse, not a sort-kernel change.

B0131 adds only a strict explicit mapped-source/permutation-publisher
composition for canonical LAS 1.4 format 7 with 36-byte records and the exact
one-stage `sort(Z,ASC,NORMAL)` graph on finite comparator-unique physical
binary64 Z. Reuse B0129's source identity and genericize its publisher without
letting the generic direct-output switch activate either route. The process
proof must require the dedicated route, mapped-source use, zero indexes, the
exact 8-byte key upload and 8-byte permutation download, complete nonidentity
record permutation, terminal-spill elision, boundary accounting, and atomic
publication. Cover ties including signed zero, nonfinite Z, dimension/order/
algorithm drift, unsupported layout, source/output aliasing, existing and
symbolic-link outputs, disabled source, injected proof failure, generic-direct
isolation, and preflight fallback. Retain only below 0.858619033 seconds, 5%
under the faster B0130 host median. No model or automatic selector follows.

The retained B0131 implementation passes the focused LAS publisher,
planner/rewrite, exact CUDA order/tie, runner-contract, direct-process, and
skewness-regression lanes. The complete Release unit binary passes 540 tests
with two optional local-corpus skips. Focused ASan/UBSan passes with leak
detection disabled because LeakSanitizer cannot run under the managed ptrace
environment; the required 1M direct process is clean under all four Compute
Sanitizer tools. Its current-clean five-pair gate preserves the
exact 36,000,375-byte artifact and prove the dedicated executor, mapped source,
zero indexes, 8 MB upload/download pair, direct sort publisher, terminal-spill
elision, and matching boundaries. The fresh profile is output-bound under the
same frozen UTC environment before its timeline informs the stopping decision.

B0132 is measurement-only. One warmup plus five alternating pinned-PDAL/direct
pairs for the already-implemented strict `hag_delaunay(count=3)` mapped-source/
one-binary64 direct-output graph must reproduce B0113's exact artifact and
prove the dedicated executor, mapped source, record summary, no host XYZ
mirror, one index, direct output, no spill, and matching boundaries. Its fresh
current-binary NCU capture must reproduce the same output before the result is
called performance-qualified. B0113's physical/fallback/sanitizer coverage
remains the compatibility basis because B0132 changes no product code.

B0133 measures pinned PDAL `hag_nn(count=5)` before implementation on the
smallest deterministic fixture that makes the fifth retained candidate
observable in output. Record a read/write/output-shaped control and a CPU
profile that names the dominant stage cost. Only if the complete process shows
a meaningful compute surface may a cheap explicit prototype extend the
existing mapped-source/planner-owned-2D-index/one-binary64 direct envelope.
No count-five code, kernel, model, selector, or calibration ladder precedes
that evidence.

The retained B0133 gate uses one warmup plus five alternating exact pairs for
both the count-five graph and its output-shaped control. The optimized-oracle
CPU phase capture runs under the same frozen UTC environment and must reproduce
the paired output hash before it informs the path decision. It records the
whole filter, `PointView::build2dIndex()`, and the post-index query/projection
tail independently. The measured 0.721933898-second stage surface and
89.906906%-of-filter post-index path clear only the cheap-prototype gate; they
are not GPU-native or performance-qualification evidence. B0134 must add
count-five native-success, fifth-candidate, exact-cutoff, tie/incomplete repair,
backend, source/publisher rejection, and atomic-output cases before widening
the explicit envelope. Its current-binary complete-process result must beat
the existing exact delegate by at least 5--10% before retention; automatic
selection remains out of scope.

B0134 retains the prototype only after the 92-case host and CUDA matrices,
focused count-five planner/rewrite/lifecycle and direct-publication gates,
host ASan/UBSan, and memcheck/initcheck/racecheck/synccheck pass. The matrix
must positively execute unique fifth-neighbor interpolation on both Grid and
Morton BVH and positively prove fifth/sixth-boundary tie plus incomplete-grid
repair. Count six, unsupported producer options, source/layout drift, and
automatic option-free execution remain exact fallback. The dirty-snapshot
one-pair result may choose the B0135 direction but may not qualify performance.
B0135 requires a clean implementation commit, five alternating exact pairs,
all direct-source/index/publication proof fields, and a fresh output-bound
profile before any performance claim or stopping decision.

B0135's clean gate satisfies that contract: five exact pairs preserve the
48,000,909-byte artifact at SHA-256
`7050b7b8fb73d7303a7986506fcd94286786a5caf4ad9a7977db3b6f39e18f73`
and measure 2.930506x pinned PDAL. The output-bound profile reproduces the same
hash and records 4.391810 milliseconds across 20 kernels, only 1.277263% of
candidate median. Publication is a 4.352942% full-elimination ceiling, while
the larger mixed-manager interval has no independently isolated reusable
5--10% component and the B0108 placement shortcut remains rejected. Count five
is qualified only for this explicit named fixture and remains unselected.
B0136 must start with pinned-PDAL count-six and output-control measurements on
a fixture with a distinct sixth retained ground candidate; no count-six code
or selection change precedes that profile.

B0136 uses the existing deterministic offset-grid fixture, whose sixth/seventh
candidate boundary is distinct, for five alternating exact count-six pairs and
a fresh output-shaped control. The optimized-oracle GDB phase capture must
reproduce the paired output hash and separately time the filter, 2D-index
build, and post-index query/projection tail. The measured
0.723439348-second/70.931468% stock stage surface and 90.010190%-of-filter tail
clear only B0137's cheap prototype gate. B0137 must add count-six native,
seventh-candidate, cutoff, arithmetic, backend, tie/incomplete/insufficient/
nonfinite-Z repair, nonfinite-XY fallback, count-seven fallback, lifecycle,
direct-source/publication,
atomic rejection, sanitizer, and option-free fallback proofs before widening
the explicit envelope. Automatic selection remains out of scope.

B0137 satisfies that proof contract with a 110-case host/CUDA matrix, focused
planner/rewrite/lifecycle/bridge tests, the strict direct process gate, the
complete Release unit binary, focused host ASan/UBSan, and all four Compute
Sanitizer tools. The dirty-snapshot one-pair result is exact at 2.882422x pinned
PDAL and proves the mapped source, record summary, no host XYZ mirror, one
planner-owned index, direct one-binary64 publication, matching boundary
accounting, zero spills, and terminal-spill elision. It may retain the bounded
implementation but may not qualify performance. B0138 requires a clean
implementation commit, five alternating exact pairs, and a fresh output-bound
profile before any count-six performance claim or stopping decision.

B0138 satisfies that qualification contract from clean implementation commit
`c3602239b`. Five alternating exact pairs preserve the 48,000,909-byte oracle
artifact and measure 2.968895x pinned PDAL. The output-bound NCU capture
reproduces the same hash and records 5.324470 milliseconds across 20 kernels,
only 1.543801% of candidate median; the HAG projection is 0.104670
milliseconds. An independent unprofiled stats pass bounds canonical
publication at a 4.296978% full-elimination ceiling and shows that the complete
manager would need a 21.700089% reduction merely to save 5% process wall,
without identifying a separable reusable component at that scale. Count six
is performance-qualified only for the named explicit fixture, remains
unselected, and is sufficiently optimized for now.

B0139 must again measure before implementation: use the deterministic
offset-grid fixture to run five alternating exact pinned-PDAL count-seven
pairs and the existing output-shaped control, then bind an exact-output CPU
phase profile naming the dominant stock cost. This is a hypothesis test for
reuse of the existing mapped source, planner-owned masked query, ordered
projection, repair path, and one-binary64 publisher. No count-seven code,
private index, kernel family, calibration ladder, model, or selector precedes
that evidence; any later prototype must retain only after an exact same-binary
5--10% complete-process gate.

B0139 satisfies that measurement contract from clean commit `a03b5bc19`. The
reused offset-grid fixture keeps every candidate distance distinct — the
enumerated seventh/eighth separation is 0.2 for interior rows and 1.0 at both
X extremes — so no new fixture is generated. Five alternating exact pairs
measure pinned PDAL at 1.050776026 seconds against a correctly delegating
1.075603836-second default, and the count-seven artifact differs from the
count-six artifact, proving the seventh candidate changes the published
interpolation. The fresh control leaves a 0.758441881-second/72.179214% stock
stage surface, and the exact-output phase capture reproduces the paired hash
while attributing 9.682102% of filter time to index construction and
88.392121% to the post-index query/projection tail. This clears only B0140's
cheap prototype gate. B0140 must add count-seven native, eighth-candidate,
cutoff, arithmetic, backend, tie/incomplete/insufficient/nonfinite-Z repair,
nonfinite-XY fallback, count-eight fallback, lifecycle,
direct-source/publication, atomic rejection, sanitizer, and option-free
fallback proofs before widening the explicit envelope. Automatic selection
remains out of scope.

B0140 satisfies that proof contract with 128-case host and 120-case CUDA
matrices, focused planner/rewrite/lifecycle/bridge tests, the strict direct
process gate, the complete 573-pass Release unit binary, focused host
ASan/UBSan, and all four Compute Sanitizer tools. A count-seven fixture must
be proved tie-free in binary64 before it may require the native path: the
exactness guard correctly declined one candidate arithmetic ground whose
squared distance was bit-identical to another ground's, and that fixture was
replaced rather than the guard relaxed. The dirty-snapshot one-pair result is
exact at 2.886489x pinned PDAL and proves the mapped source, record summary,
no host XYZ mirror, one planner-owned index, direct one-binary64 publication,
matching boundary accounting, one 25,000,050-byte upload, and zero
spill/fallback boundaries. It may retain the bounded implementation but may
not qualify performance. B0141 requires a clean implementation commit, five
alternating exact pairs, and a fresh output-bound profile before any
count-seven performance claim or stopping decision.

B0141 satisfies that qualification contract from clean implementation commit
`437b0b060`. Five alternating exact pairs preserve the 48,000,909-byte oracle
artifact and measure 2.959118x pinned PDAL. The output-bound NCU capture
reproduces the same hash and records 6.421888 milliseconds across 20 kernels,
only 1.807004% of candidate median; the HAG projection is 0.120224
milliseconds. An independent unprofiled stats pass bounds canonical
publication at a 5.034629% full-elimination ceiling that cannot actually be
eliminated, and shows the complete manager would need a 17.130% reduction
merely to save 5% process wall. The dominant 50.412811% interval is the
process-level CUDA startup already closed by B0054 and B0108, so it is not a
new opportunity. Count seven is performance-qualified only for the named
explicit fixture, remains unselected, and is sufficiently optimized for now.

## Parameterized option envelopes replace per-value proof ladders (D0204)

B0142 measured that HAG-NN stock cost is strongly sub-linear in `count`
(9.142857x more neighbours for 2.3241x more stock stage time) while the device
projection is already generic over `count`. D0203 therefore closes the range as
one parameterized envelope and D0204 generalizes the rule: a proof ladder over
a numeric option is closed as a single parameterized envelope whenever the
implementation is already generic over that option and the measured cost is not
superlinear in it.

This relaxes no proof obligation. The matrix must still cover the distinct
(k+1) candidate on both index backends, cutoffs, inclusive bounds, arithmetic
edges, tie/incomplete/insufficient/nonfinite repair, lifecycle, bridge, direct
source and publication, and atomic rejection — but it is *generated* over the
option rather than transcribed per value, and representative values spanning
the range are qualified rather than every value. Two obligations are added
rather than removed:

- every generated fixture that requires the native path must be proved
  tie-free in exact binary64 before it is accepted, because a generator can
  silently produce a fixture whose candidate distances collide; and
- the value immediately above the admissible cap must be proved to retain
  pinned behaviour, so the envelope has a tested upper edge.

Both obligations are live experience rather than theory: while writing the
count-seven matrix, one hand-chosen ground produced a squared distance
bit-identical to another ground's, and the exactness guard correctly declined
the native path. The fixture was replaced; the guard was not relaxed.

## NVIDIA architecture qualification

Release configuration generates real code for every architecture supported by
the selected CUDA compiler and PTX for its highest major target. That proves
only that the compiler accepted the source. Exact support for an SM requires a
physical-device run of a compact, fixed-bit differential lane covering:

- LAS field translation, ordered assign/ferry/ordinal conversion, signed zero,
  subnormal, NaN payload, half-integer, and cast-limit boundaries;
- point maps, stable radix ordering, selection, histogram, summary, and robust
  reductions around warp, block, CUB, and chunk boundaries;
- spatial cell/radius boundaries, equal-distance ties, near-repeated
  eigensystems, resident publication, and deterministic tie repair; and
- one integer-only translation/compaction control.

Run the CUDA-12.x lane on each legacy-compatible real target (currently SM
50/52/53, 60/61/62, 70/72, 75, 80/86/87, 89, and 90) and the CUDA-13+ lane on
each current target advertised by that compiler. Exercise embedded PTX with
`CUDA_FORCE_PTX_JIT=1` on compatible physical devices. Record toolkit, CCCL,
driver, device, cubin-versus-PTX path, output hashes, and first differing bits.
The full corpus, performance suite, and sanitizers may use representative
machines, but this compact bit lane may not be skipped for an advertised exact
architecture. SM 89 is the only physically qualified architecture at the
current checkpoint.

A serialized CUDA 13.3 compile in `build/pdg-cuda-portable` sets
`PDG_BUILD_TESTS=OFF`, `PDG_CUDA_ARCHITECTURES=all`, and
`PDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES=ON`; `pdg_core` builds 51/51 at
`--parallel 1`, including `NeighborhoodKernels.cu`. `cuobjdump` reports SASS
in that object for SM 75, 80, 86, 87, 88, 89, 90, 100, 103, 110, 120, and 121,
with PTX for SM 120. This is compile evidence for the targets supported by that
installed compiler plus its newest PTX target. It does not promote any runtime
beyond the physically bit-qualified SM 89 device.

All product CUDA translation units pin `--fmad=false`, `--ftz=false`,
`--prec-div=true`, and `--prec-sqrt=true`. These flags remove known compiler
contraction/denormal variability but do not replace the physical matrix:
libdevice, Eigen device code, and CCCL policies remain toolkit- and
architecture-sensitive exactness surfaces.

## Fourteen-workload reference suite (D0239/B0240)

`bench/pipelines/reference/manifest.json` is the closed headline membership
contract. Validation requires exactly r1-r14 in order, one weight each,
declared fixture provenance, cold/warm cache states, materialization policy,
artifact/oracle/profile/test requirements, and repository-contained pipeline
paths. Supporting r10/r14 variants must declare weight zero. Run the bounded
infrastructure contracts with:

```sh
python3 tests/differential/reference_runner_test.py \
  --runner scripts/pdg/benchmark_reference.py
python3 tests/differential/reference_suite_test.py \
  --runner scripts/pdg/reference_suite.py \
  --manifest bench/pipelines/reference/manifest.json --repo-root .
```

The reference runner scrubs ambient `PDG_*`, rejects externally supplied
internal markers, recursively substitutes named inputs and dynamic parameters,
discovers deterministic single/multi-output artifact sets, and removes them
before each process. A failed process, missing output pattern, differing
artifact name/count/size/hash, nondeterministic repeat, stdout/stderr drift, or
status drift returns nonzero. It never publishes a timing from an inexact run.

Raster artifacts add `gdalinfo -json -checksum` after the timed process. Remove
only role-specific `description` and `files` paths; compare and retain every
other field, including driver, dimensions, complete geotransform, CRS, corner
and WGS84 extents, metadata domains, nodata, band count/types, blocks,
checksums, masks, and overviews. Byte identity remains mandatory. Point-cloud
byte identity covers header/VLR/EVLR bytes, encoded order, and point records;
on a difference, the ordinary LAS diagnostic must still report the first byte
and decoded responsible field.

Warm and cold cache are separate runs and aggregates. Warm cache records the
number of untimed warmups. Cold mode must successfully apply
`POSIX_FADV_DONTNEED` to every used input before every oracle and candidate
process; otherwise it fails instead of labeling an uncontrolled run cold.
This is file-scoped cache eviction, not a claim that every system cache is
globally empty. Reports record complete wall and a 2 ms `/proc` sample of the
summed public-process/descendant resident set, plus used input/output bytes,
points/s, MiB/s, compression ratio, and peak RSS.

`reference_suite.py run` authenticates fixture sizes/hashes, serially invokes
all fourteen headlines, and aggregates only after every report is exact.
`--variants-only` runs the zero-weight companion matrix; `--workload ID`
supports bounded reruns and deliberately suppresses a partial aggregate. For
each cache state report:

- equal-workload geometric mean
  `exp(mean(log(median(PDAL_i) / median(PDG_i))))`;
- total-wall speedup
  `sum(median(PDAL_i)) / sum(median(PDG_i))`; and
- aligned suite-round ratios from the sum of all fourteen oracle/candidate
  process walls at each measured iteration.

Missing, malformed, failed, or inexact inputs make the aggregate incomplete;
never omit or renormalize them. B0240's test-first runner caught nondeterminism
in the initial COPC variants: the pinned writer's default random sampling and
concurrent COPC reader changed bytes between repeats. The retained references
use `fixed_seed=true,threads=1` for `writers.copc` and `requests=1` for
`readers.copc`. Removing those options requires a new deterministic oracle
proof, not a semantic-only blessing.

The checked-in fixture recipe may write only derived files below the selected
build directory. It verifies the retained AHN LAZ hash before generating the
LAS, RGB raster, or heterogeneous merge pair. Corpus redistribution remains
unasserted and no source corpus file may be modified, moved, or committed.

The definition contract also pins the literal r7-r14 stage sequences and
surface policy, RGB band/SRS interaction, multipolygon/hole materialization,
all six decimation companions, the external-model-free classification chain,
tile origin/output pattern, ordered heterogeneous merge inputs, and every
conversion direction. A mutated headline weight or malformed manifest is a
refusal test. Stage-specific boundary and malformed behavior reuses upstream
colorization/crop/grid-decimation/neighborclassifier/splitter/merge tests and
the fork's returns/merge, divider/splitter, COPC-reader, LAS-writer, SMRF,
outlier, sample, voxel, reprojection, and raster-writer matrices; the reference
runner adds repeated complete-process differential and determinism checks.

### B0241 r11 and optional-KD3 regression gates

The first measured follow-up keeps r11 on the public host path. Its warm proof
uses nine alternating pairs to resolve the B0240 loss; cold retains three
file-evicted pairs. Both require the same LAZ bytes, metadata, point order,
stdout, stderr, and status. The fourteen-workload aggregate replaces only the
aligned three-pair r11 report; the nine-pair report is retained separately as
the stronger parity-resolution evidence.

Optional KD3 backing is tested as two construction-time specializations. The
cached and uncached indexes must return identical point IDs and squared
distances; an injected cached build failure must not publish a partial index;
and four concurrent cached readers must reproduce a serial result. Run these
gates in Host Debug, leak-disabled ASan/UBSan, and TSan. The physical 4M LOF
process proof must additionally require both cached backing and parallel exact
repair, then compare the complete public artifact and diagnostics against the
pinned oracle. Because no CUDA translation unit changes in this slice, that
physical regression does not create a new Compute Sanitizer claim.

### B0247 corrected r9 polygon contract and negative selection

The reference-runner contract must materialize EPSG:4326 WKT with longitude as
X and latitude as Y. Its bounded infrastructure regression parses the emitted
vertices and requires the Dutch longitude/latitude domain; a byte-exact empty
crop is a failed workload definition, not a baseline. The corrected headline
must emit the measured 473,825-point artifact and continue to compare complete
LAS bytes/metadata/order, stdout, stderr, and status.

The manifest pins that cardinality explicitly. The benchmark runner decodes
the authoritative LAS fixed-header count for every LAS/LAZ artifact, and the
aggregate rejects any measured oracle or candidate r9 record that is missing
the named artifact or differs from 473,825. It also requires one immutable
oracle hash and one immutable candidate hash across all fourteen reports, so a
replacement-row mixture cannot become a headline aggregate. Each report is
also bound to its baseline schema, workload label, and requested cache state.
Mutation tests reject every one of those drifts, while bounded runner tests
exercise LAS 1.2 legacy counts, LAS 1.4 extended counts, and malformed or
short-header omission.

`pdg_r9_polygon_process_matrix_host_exact` generates a deterministic bounded
LAS below the build directory and runs simple polygon,
multipolygon-with-hole/disjoint-member, boundary/hole, and geometry-SRS
reprojection differentials. Missing geometry SRS, malformed WKT, malformed
SRS, and a repeated determinism case freeze refusal and stability behavior.
It uses the pinned oracle and does not modify or redistribute corpus data.

The literal corrected headline must classify to the engine in both dispatcher
unit and real process-boundary tests. B0247's direct-exec and private
in-process CLI reports remain negative performance evidence: both are exact
but resolve slower than pinned PDAL over 21 warm pairs. A future positive r9
route must repeat the bounded refusal/malformed/configured-child matrix and
clear pinned-oracle plus same-final warm and cold gates before selection.
Corrected all-headline aggregates replace, rather than coexist with, claims
containing the old empty r9 row.

### B0248 r7 DSM direct-host contract and clock isolation

`pdg_r7_dsm_process_matrix_host_exact` builds deterministic bounded LAS and
compressed LAZ fixtures below the build directory. Its selected-route cases
use the complete literal r7 grammar and materialized AHN-domain bounds; its
fallback cases cover legal return/writer policy drift. The eight-case matrix
also covers malformed return groups, malformed bounds, selected and fallback
repeat determinism, exact artifact/stream/status comparison, and complete
normalized `gdalinfo -json -checksum` equality. The selected raster contract
requires GTiff, 1,500 x 77, EPSG:28992, the complete geotransform, one Float64
`max` band, nodata -9999, checksum, and byte identity. It runs in Host Debug,
CUDA Release, and leak-disabled ASan/UBSan; no corpus data is modified.

Dispatcher unit tests pin the complete option types and values. The real
process boundary proves the literal graph reaches the configured oracle,
preserves its status/stdout/stderr, remains direct under the preload-only clock
input, and returns to the engine for CLI modifiers, `PDG_*` controls, or graph
drift. Because both paths execute unchanged PDAL stages, the grammar gate is a
performance envelope rather than a semantic fixture identity gate.

The deterministic preload consumes `PDAL_TEST_FROZEN_EPOCH`, not a product
`PDG_*` control. `benchmark_reference.py` scrubs ambient instances, rejects
explicit `--environment` injection, injects the value only when a frozen-time
library is supplied, and records `freeze_environment` plus `freeze_epoch` in
every report. The preload retains `PDG_FROZEN_EPOCH` only for historical manual
replay; that spelling correctly forces the engine through the launcher's
fail-closed `PDG_*` rule. Environment-contract tests freeze both behaviors.

### B0249 r10 decimation direct-host contract

`pdg_r10_decimation_process_matrix_host_exact` builds deterministic dense,
sparse, and empty compressed-LAZ inputs below the build directory. Its
selected cases use the complete literal r10 grammar: lowercase non-COPC LAZ,
`filters.voxelcentroidnearestneighbor` with numeric `cell:2.5`, and lowercase
LAZ output with the string option `compression:"true"`. The nine cases cover
selected dense/sparse/empty and malformed input, legal `cell:1.0` and boolean
compression drift, invalid string-cell refusal, the pinned oracle's accepted
numeric zero-cell behavior, and selected repeat determinism. Every case
compares artifact bytes, LAS point order/count, stdout, stderr, and exit
status; the repeat case compares both candidate and oracle artifacts across
runs.

Dispatcher unit tests freeze exact option types/values and reject neighboring
root, topology, stage, extension, option, and malformed forms. The real
process boundary proves the literal graph reaches the configured oracle and
preserves its observables, while CLI modifiers and any product `PDG_*` control
return to the engine. Both accepted and refused paths execute unchanged PDAL
stages; this is a performance-envelope grammar gate, not fixture-dependent
semantic substitution.

The final gate requires separate nine-pair public-versus-pinned and
same-final-engine-versus-public reports in warm and file-scoped cold-cache
states. The retained route resolves only against the engine path, at
1.041226x warm and 1.034230x cold with 9/9 wins; pinned-PDAL rows are retained
as unresolved. Host Debug, focused CUDA Release, and leak-disabled Host
ASan/UBSan run the selected/fallback matrix plus dispatcher unit and process
boundaries. No CUDA implementation changed, so this contract adds no Compute
Sanitizer requirement or GPU-native claim.

### B0250 r14 conversion/compression contract

`pdg_r14_conversion_process_matrix_host_exact` creates one deterministic
12-point LAS 1.4 format-7 fixture, then derives compressed LAZ and fixed-seed,
single-thread COPC inputs below the build directory with the frozen pinned
oracle. Its 12 semantic cases execute 19 differentials: LAS -> LAZ, LAZ ->
LAS, LAZ recompression, LAS/LAZ -> COPC, and COPC -> LAS/LAZ each run twice;
truncated LAS, malformed LAZ/COPC, an unsupported writer, and the pinned
writer's accepted odd compression-string behavior complete the bounded
matrix.

Every positive compares full bytes, stdout, stderr, and status, then parses
both artifacts to require the expected point count, LAS compression flag, and
COPC info VLR. Repeated reports compare both oracle and candidate artifact
maps, covering deterministic names, sizes, and hashes. Full byte identity also
freezes header, VLR, EVLR, semantic point order, and compression payload. The
dispatcher unit and real process boundary record that r14 uses the existing
generic two-stage oracle delegation, supported PDAL CLI modifiers remain with
that delegation, and any product `PDG_*` control still fails closed to the
engine. No r14-specific matcher exists.

Reader-worker tuning uses a separate no-product prototype that appends only
`--readers.las.threads=N` after the materialized pipeline. Counts 1/2/4/6/8
receive exact three-pair warm and file-scoped cold screens. Only a setting that
clears both states may advance; one worker receives nine-pair confirmation and
is rejected because its cold paired interval spans parity. Host Debug, focused
CUDA Release, and leak-disabled Host ASan/UBSan run the route and matrix gates.
No CUDA implementation changed, so no Compute Sanitizer claim follows.

### B0251 exact parallel LAZ compression contract

`LazChunkCompression.IndependentFormatSevenChunksMatchSequentialPayload`
compares the complete sequential and independently compressed lazperf payload
and chunk table for deterministic format-7 records at empty, last-point,
50K-boundary, and multi-chunk counts. The maintained
`pdg_laz_chunk_compression_benchmark` invokes the production compressor with
one and two workers in alternating order, requires complete payload-byte
identity outside the timed comparison, refuses report overwrite, and records
all samples plus the exact payload size.

The r14 matrix adds a repeated 1M generated LAS case, bringing the contract to
13 semantic cases/21 full process differentials. Before execution it parses
and requires every automatic-admission fact: file size, point count, LAS
version, offsets, compression bit/format, record width, scales, offsets, and
XYZ extrema. `differential.py --candidate-env` applies the test-only writer
assertion only to the candidate; the public process fails unless its real
`LasWriter` observes two workers, while the forked host oracle remains serial.
Full bytes, metadata/header, point order, stdout, stderr, status, and repeated
determinism remain mandatory.

Dispatcher unit and process tests separately freeze the literal grammar,
point/layout refusals, CLI-modifier refusal, and external-thread-control
refusal. Final qualification requires nine alternating public/pinned pairs in
warm and file-scoped cold states plus a fresh exact fourteen-headline
aggregate. Host Debug, CUDA Release, and leak-disabled Host ASan/UBSan run the
byte proof, activation matrix, and route tests. The implementation is host
only; no Compute Sanitizer result is inferred.

B0252/D0251 add the retained environment-boundary regression. External
product-control refusal is evaluated before automatic command eligibility,
and an external `PDG_LAZ_COMPRESSION_THREADS` value is removed before engine
execution. The activation assertion is source-compiled only when
`PDG_BUILD_TESTS` is enabled. `differential.py` removes ambient assertion and
CTest-only sanitizer-preload transport variables before constructing either
product environment, then applies explicit candidate-only assignments. The
leak-disabled ASan/UBSan matrix is required to exercise this transport path;
the final launcher must then receive a fresh public gate, profile, and full
aggregate rather than inheriting the first B0251 hashes.

B0253/D0252 add a direct production-factory decode contract before changing
the existing reader. Deterministic formats 6, 7, and 8 include four Extra
Bytes, all scanner channels, 1/2/50K point counts, two complete decodes, and a
bounded callback that must reject a one-byte truncated chunk. The maintained
primitive constructs 20 `build_las_decompressor` instances and separately
constructs and decodes twenty 50K format-7 chunks per sample; every decoded
record byte must match. Reader-worker and codec-setup experiments cannot alter
public selection unless this seam passes and r10 resolves against pinned PDAL
in both warm and file-scoped cold states. B0253's wider pools are slower and
its mode-gated setup warm interval spans parity, so the production prototypes
are removed and no cold promotion is claimed.

B0254/D0253 complete the reader-worker screen at 4/6/8/10/12 around the
seven-worker default. Selection requires a fresh 21-pair final-binary r7 gate
in both warm and file-scoped cold states, exact complete TIFF metadata and
bytes, stdout, stderr, status, and route/refusal coverage. The
process boundary must prove that only the admitted r7 command receives
`--readers.las.threads=4`; r10, count/layout/grammar drift, CLI modifiers, and
external `PDG_*` controls must not receive it. A positive warm r10 result does
not qualify because its cold interval spans parity. Focused CUDA Release and
leak-disabled ASan/UBSan run the selector, process boundary, r7/r10 matrices,
and bounded decoder tests; no Compute Sanitizer inference is allowed because
no CUDA source changes.

### B0256 r11 shared-index qualification

The r11 process matrix is a deterministic generated 144-point LAZ contract
with eight cases/nine executions: the literal SMRF/statistical-outlier/
neighbor-classifier chain, legal Classification-domain drift, legal outlier
and neighbor count drift, invalid `k`, a missing domain dimension, an invalid
outlier multiplier, a cached-index/coordinate-mutation boundary, and two
identical determinism executions. The boundary runs `filters.normal` to build
a PointView KD3 product, generic `filters.assign` to set X to zero, then
statistical outlier. Each case uses
the central differential to compare artifact bytes, decoded LAS metadata and
point order, stdout, stderr, and exit status; failure cases must not publish an
output.

The selected implementation changes only index ownership after a fresh build.
Statistical outlier must call `invalidateProducts()` at entry, then build the
existing exact nanoflann tree through the PointView cache. This preserves its
historical behavior even when an upstream coordinate mutator left a stale
product, while the adjacent neighbor classifier observes the newly built
XYZ/order generation. Radius outlier keeps its private local index. Existing
standalone outlier and SMRF matrices are required alongside the composed
matrix.

Performance qualification requires alternating final-public-process runs
against pinned PDAL in both warm and file-scoped cold states. B0255's reports
are superseded because they bind the unsafe library. Corrected nine-pair warm
and cold runs must both resolve with exact artifacts and process observables;
the three-pair aggregate is not a substitute. Focused
Host Debug, CUDA Release, and leak-disabled Host ASan/UBSan cover the composed
and standalone matrices plus planner/index lifecycle tests. No Compute
Sanitizer result is inferred because no CUDA source changes.

### B0258 exact host worker passes

B0257 first measured, with the checked-in `pdg_r11_neighborhood_attribution`
harness, that the r11 vote tally is about 0.1% of process wall and that the
two serial per-point kNN passes are about 5.55 s of 6.8 s; a direction is
rejected or chosen by such an in-process controlled prototype before any
certified lane is spent (D0204). The retained implementation runs the
statistical outlier and neighbor classifier per-point kNN passes over fixed
contiguous row chunks on host workers through
`pdal/private/HostNeighborhoodWorkers.hpp`; the outlier's serial online-moment
reduction, the classifier's ordered-map vote, worker-order merge, serial
application, options, diagnostics, and output order are unchanged.

Correctness coverage lives in the r11 process matrix, now eighteen cases/
nineteen executions. Because its 144-point fixture is far below the
4,096-rows-per-worker threshold, the chunked path is forced through the
test-only `PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS` hook and the observed
count is asserted with `PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS`; both are
compiled only into `PDG_BUILD_TESTS` builds and are candidate-only values, so
every case still compares against the unchanged serial oracle. Required cases:
three and five forced workers on the headline and legal-drift graphs, a forced
count above the row count (must clamp to one row per worker), the candidate-file
branch, the no-domain explicit-dimension form, an unconvertible vote dimension
whose first-row diagnostic and status must match the serial oracle across
workers, the `PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS` control, a small input
that must stay serial, a `PDG_NATIVE_WORKERS` cap that a forced test count
overrides, and a required-count mismatch that must fail closed with the named
diagnostic and no output. The matrix must pass Host Debug, CUDA Release,
leak-disabled Host ASan/UBSan, and TSan; a direct 50K-point TSan run of the
forked `pdal` with the natural worker count must report no ThreadSanitizer
warning. Standalone outlier and SMRF matrices remain required. Both benchmark
harnesses scrub the new controls (`benchmark_reference.py` now scrubs the whole
`PDAL_TEST_` prefix) and their contract tests assert it.

Performance qualification requires nine alternating final-binary warm and cold
pairs against pinned PDAL, both resolved and exact; a same-final-binary
disabled control that attributes the gain to the workers; small-input controls
(50K and 250K) that must not lose; and the fourteen-workload aggregates. A
forced experimental CUDA statistical-outlier probe on the same graph is
recorded as negative evidence. No CUDA translation unit changes, so no
Compute Sanitizer result is inferred.

### B0261 pinned-oracle differential lane and outlier semantics

Every registered process matrix compares the public candidate against the
forked host CLI (`$<TARGET_FILE:pdal>`), and `differential.py` sets the
candidate's `PDG_ORACLE_PDAL` to that same oracle. Both hide fork-side changes
to upstream stage code: the oracle shares the candidate's library, and any
delegated route becomes the oracle. B0260's B0256 defect was invisible to
every such test and was only found by hand against the pinned binary.

Rules from B0261/D0260: (1) `PDG_PINNED_ORACLE_EXECUTABLE` names the pinned
upstream build (default `build/pdal-upstream-tests/bin/pdal` when present);
(2) `differential.py --candidate-oracle` keeps the candidate delegating to the
forked sibling while `--oracle` is pinned; (3) a matrix that covers a fork
change under `filters/`, `io/`, or `pdal/` must include `pinned_oracle` cases
that exercise the changed code and its adjacent producers/consumers,
including a coordinate mutator between them, on a fixture large enough for
nanoflann's 100-point leaves to form a multi-level tree (the 144-point lattice
cannot expose stale pruning); (4) such a case must be shown to fail on the
defective library before the fix is accepted; (5) the lane skips with an
explicit message when no pinned build exists. The r11 matrix now carries the
outlier -> mutator -> nndistance/classifier and normal -> mutator -> outlier ->
nndistance cases plus the pinned headline (23 cases/24 executions).

### B0274 concurrent KD builds

Every KD2/KD3/KDFlex index now builds its nanoflann tree concurrently under
the shared host-worker policy. The proof is structural equality observed
through queries: `Kd3Concurrency.ConcurrentBuildProducesTheSerialTree`
compares kNN ids, squared distances, and radius results between a
one-thread build and 2/3/5/8-thread builds on tie-dense fixtures (a
modular-arithmetic cloud at 12K and 130K points and an integer lattice whose
k=8 cuts through equal-distance groups). Every pinned-oracle matrix that
forces `PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=3` also forces three build
threads, so r11, SMRF, outlier, HAG, and LOF lanes exercise the concurrent
build against pinned PDAL. Any change to the vendored nanoflann build must
keep `divideTree` and `divideTreeConcurrent` split-for-split identical and
be proven with this unit and those matrices.

### B0273 banded raster accumulation

`writers.gdal` is proven only against the pinned oracle
(`pdg_gdal_writer_matrix_pinned_exact`, 19 cases; three forced band
workers on a 30,200-point scatter with a dense edge strip and 200 exact
duplicates): dynamic and fixed grids, `all`/`idw`/`mean,stdev,count`/`max`/
`min`, IDW power and radius, bin mode, window fill, percentiles (serial by
design), a non-Z dimension with float32 output, upstream `filters.range`
skips, a large radius, bounds that clip most points, both execution modes,
the disable control, and the natural policy; raster bytes and metadata,
streams, and status compare byte-for-byte. Test builds honor
`PDAL_TEST_TRACE_GDAL_BANDS=1` (rows, bands, listed points, height per
banded run) so a case proves the path it took: streaming dynamic radius
grids and percentiles must show no banded run. Any partition of an
order-dependent accumulation must assign each accumulator to exactly one
worker, replay the full point order per worker, and reproduce any pinned
interleaving quirk (here: cells created by later expansions never see
earlier points) or decline.

### B0271 `--fast` tie-order contract

The fast contract is proven three ways and never mixed with the exact one:
(1) `FastMode` and `CudaSpatialIndex.RelaxedTieOrderMasksOnlyTheTieBitOnBothBackends`
show that under the marker only `KnnDistanceTie` disappears from host and
device kNN statuses (ids, distances, incomplete and eigen bits unchanged) on
both index backends; (2) `PdgNeighborClassifierFilter` and
`PdgApproximateCoplanarFilter` fast cases show no host tie repair runs and
that the rows that differ from upstream are bounded by the exact run's
ambiguous-row count; (3) the CUDA-lane `pdg_fast_tie_contract_cuda` process
test shows the exact route still repairs (proof gate satisfied), `--fast`
fails closed under the same gate and publishes nothing, `--fast` records
keep count, layout, and coordinates and differ only in attributes, and the
default contract is byte-identical. The launcher's dispatcher process test
covers the flag on every route including the environment-selected engine
route. Reference measurements under the fast contract use
`benchmark_reference.py --contract fast --candidate-arg=--fast` (record-by-
record comparison against the first oracle artifact; count, layout, and
coordinates identical; differing records bounded by
`--fast-max-differing-records-fraction`, default 1%; the report carries the
count) and `reference_suite.py --contract fast`, whose aggregate is labeled
and refuses reports measured under another contract. A fast claim states
the differing-record count beside the speedup and cross-checks it against
the exact run's tie-row count; the exact suite is rerun on the same binaries
so the default contract's byte-identity is re-proven whenever a fast-only
path is added.

### B0270 ordered parallel COPC decode

`readers.copc` under `requests=1` is proven only against the pinned oracle
(`pdg_copc_reader_matrix_pinned_exact`, 19 cases; the fixture is a 1.2M-point,
21-node COPC written by the pinned oracle with `threads=1,fixed_seed=true`
from generated text; the candidate delegates to the forked sibling): full,
bounds, bounds+resolution, resolution, polygon, `count`, an empty selection,
`keep_alive` 2 (backpressure with more nodes than the window), the disable
control, and a two-worker cap, in streaming and `--nostream` modes, through
LAS and 17-digit text sinks. Test builds honor `PDAL_TEST_TRACE_COPC_DECODE=1`
(tiles, ordered flag, worker count) so a case proves the path it took. Any
change that decodes tiles concurrently must emit them in the pinned
single-thread order, keep the request concurrency the user asked for, and
be proven on a fixture with more nodes than the keep-alive window.

### B0269 parallel LAS record unpacking and the reader batch hook

`readers.las` record unpacking is proven only against the pinned oracle
(`pdg_las_reader_unpack_matrix_pinned_exact`, 30 cases; fixtures are
written by the pinned oracle from generated text; the candidate delegates
to the forked sibling and forces three unpack workers): formats
0/1/2/3/6/7/8, LAS and LAZ, LAS 1.4 extra-bytes VLRs, LAS 1.2 named and
`use_eb_vlr` extra bytes, 120,000-point multi-tile fixtures, `count`,
`start` (serial in streaming), downstream `filters.range` skips, a 700-point
file that stays serial, the disable control and the natural policy, in
streaming and `--nostream` modes, through LAS 1.4/format-8 `extra_dims=all`
and 17-digit `writers.text` sinks. Test builds honor
`PDAL_TEST_TRACE_LAS_UNPACK=1` (one stderr line per worker run with rows,
segments, and slots) so a case proves which path it took. The
`LasReaderUnpack` unit runs a 15,000-row stream table over 123,457 points so
a batch spans two 50,000-record tiles (two segments in one run) and compares
parallel, serial-control, and standard-mode bytes. Any reader adopting
`Streamable::readStreamBatch` must (1) consume its input in exactly the
per-point order and report `read < pointLimit` only at end of input, (2)
write only rows [0, read) and leave the rest of the batch to the executor's
accounting, and (3) be proven with a pinned-oracle matrix in both execution
modes plus a non-aligned stream-capacity unit when its input is tiled.

### B0268 parallel LAS record packing

`writers.las` record packing is proven only against the pinned oracle
(`pdg_las_writer_pack_matrix_pinned_exact`, 23 cases; the candidate
delegates to the forked sibling and forces three pack workers on 30,000-row
fixtures so three streaming batches and one ~1 MB standard block cross the
parallel threshold): formats 1/3/6/7/8, LAS and LAZ, `extra_dims=all`,
legacy flag packing, upstream `filters.range` skips (compacted rows), the
`where` and `discard_high_return_numbers` serial declines, auto scale/offset
in both modes, PDRF < 6 per-point warnings that must reach stderr in pinned
order (the deferred serial repeat), the identical unconvertible-scale
diagnostic and status raised from inside a parallel batch/block, the
disable control, and the natural policy. The header summary (bounds, counts,
return histogram) is compared through the artifact bytes; a unit test
(`LasSummaryMerge`) proves the slot-ordered `las::Summary::merge` fold equals
the serial fold at every slot count including signed-zero ties, NaN, and
empty slots. Test builds honor `PDAL_TEST_TRACE_LAS_PACK=1`, which prints
each parallel run's outcome (`committed`/`deferred`/`threw`) so a matrix
case proves which path it took instead of inferring it. Any writer that
packs records on workers must decline (not emulate) every option that can
change record count or order, defer per-record diagnostics to a serial
repeat, and commit nothing before every slot has succeeded.

### B0267 sample filter data-structure change

`filters.sample` is proven only against the pinned oracle
(`pdg_sample_process_matrix_pinned_exact`): radius and cell forms on a dense
0.5 m lattice, a sparse lattice, and a deterministically jittered cloud that
places points near voxel boundaries, the marker `dimension` form, explicit
origins, and both execution modes, with kept points emitted at 17 significant
digits. Any change to a greedy host algorithm's data structures must be
covered by such a boundary-heavy pinned matrix before a reference gate is
run.

### B0266 slot-pooled reprojection and the streaming batch hook

`filters.reprojection` is proven only against the pinned oracle
(`pdg_reprojection_process_matrix_pinned_exact`; the candidate delegates to
the forked sibling): streaming and `--nostream` modes, natural and forced
worker counts, the `where` serial fallback, a large lattice, and a geographic
fixture whose unprojectable points must be dropped in the same positions or,
with `error_on_failure`, raise the identical first-point diagnostic and
status. Outputs go through `writers.text` at 17 significant digits so any
bit difference in a reprojected double fails the case. Any stage adopting
`Streamable::processStreamBatch` must (1) honor `skip()` and mark rejected
rows only after its workers join, (2) decline when a `where` expression is
present unless it evaluates it itself identically, and (3) be proven with a
pinned-oracle matrix in both execution modes. Reference gates (r8, r1) and
the fourteen-headline aggregates qualify performance; Host Debug,
leak-disabled ASan/UBSan, and TSan run the matrix.

### B0264 exact host workers inside SMRF

Any worker pass added to upstream stage code must be proven two ways: the
stage's process matrix repeats every case with
`PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS`/`..._REQUIRE_...` on the candidate
(chunked point/cell passes and any partial-result merge versus the serial
forked oracle on the tiny contract fixture), and pinned-oracle cases on a
fixture large enough to cross every natural threshold (SMRF: 240 x 240
lattices at cell 1 and 0.5 so the pooled diamond passes run with 3--14 tasks)
compare against the pinned binary while the candidate delegates to the
forked sibling. Morphology pools are capped at eight workers by measurement;
the minimum-Z merge is bounded to 256 MiB of partial rasters. Reference gates
(r3, r11) and the fourteen-headline aggregates qualify performance; Host Debug,
leak-disabled ASan/UBSan, and TSan run the matrix.

### B0263 cached KD3 default and the coordinate epoch

A change to `PointView` product semantics must be proven three ways: (1)
`Kd3Refresh` units compare a published cached product reused after `setField`,
view-bound `PointRef` writes, and `PointView::sort` against an uncached tree
built before the mutation and read live afterwards, bit-for-bit; (2) the r11
pinned-oracle matrix carries producer/mutator/consumer chains (`normal ->
assign -> nndistance`, `normal -> sort -> nndistance`, `nndistance -> assign
-> classifier`, `normal -> nndistance -> classifier`) with
`PDAL_TEST_VERIFY_KD3_SNAPSHOT=1` armed on the candidate; (3) the full Host
Debug and physical CUDA aggregates run once with the verifier in the
environment so any lane whose reused product diverged without an epoch
change fails closed. Every new mutation path added to `PointView` or
`PointRef` must move the epoch; the verifier is the guard for the ones that
are missed. `PDG_DISABLE_KD3_COORDINATE_CACHE` is the same-final-binary
control and is scrubbed by both harnesses.

### B0262 terminal-sink repair tree and full-aggregate rule

The resident rewrite's `pdg_region_terminal_sink` marker is unit-tested both
ways: true when the terminal writer alone follows the region, false before a
host consumer (an r11-like tail). Every neighborhood wrapper must accept the
hidden marker; the resident selection matrix is the process-level guard
because a wrapper that rejects an unknown hidden option silently falls back
to the host executor. The r6 nine-pair warm/cold gates and the fourteen-
headline aggregates qualify the change; the private tree's bytes are proven
identical through the unchanged r6 artifact hash. Because the r2 automatic
CUDA case had failed since B0243 without anyone running the CUDA aggregate,
each session must run the complete physical CUDA aggregate at least once
before claiming the tree green (827/827 at B0262).

### B0259 default LAZ chunk-compression workers

The writer default moves from serial to `min(4, hardware threads)` workers;
the internal channel still overrides it. Required evidence before adopting a
general default rather than another bounded selector: a same-binary engine
probe over every reference headline at the candidate counts, all exact, with
no LAZ-writing workload slower and no non-LAZ workload changed beyond noise;
then the complete public warm and cold fourteen-headline aggregates and
nine-pair final gates for the largest, largest-relative, and smallest LAZ
beneficiaries. Correctness relies on B0251's byte-identity unit
(`LazChunkCompression.IndependentFormatSevenChunksMatchSequentialPayload`)
plus process matrices: the r11 matrix requires the real writer to observe the
four-worker default with no launcher arming, and the r14 matrix keeps its
launcher-armed two-worker assertion. Host Debug full aggregate, leak-disabled
ASan/UBSan and TSan runs of the LAZ-writing matrices remain required; no CUDA
source changes.

## B0232 automatic direct-sort qualification

The automatic sort gate is a complete-process proof, not a primitive radix
sort test. Its hash-pinned inputs are uncompressed LAS 1.4 format 7 with
36-byte records and finite comparator-unique logical Z. The accepted graph is
exactly option-free LAS -> `sort(Z,ASC,NORMAL)` -> LAS `extra_dims=all`; the
automatic cardinality envelope is 600K--16M on the pinned SM89 profile.

The process matrix must prove:

- exact public and required-route output, stdout, stderr, status, metadata, and
  order at 600K and 1M, plus required refusal at 550K;
- host fallback before publication for duplicate, signed-zero-equivalent, and
  non-finite keys, and for source/preflight/execution-proof injection;
- fail-closed neighboring dimensions, directions, algorithms, options,
  compression, formats, record widths, and topology;
- oracle-identical default behavior for existing, aliased, and symlink output
  destinations, while the required route returns 124 without mutation; and
- memory acceptance at exactly 64 bytes/point and refusal one byte below it.

The device-memory figure is independently guarded by a tracking
`MemoryResource` around the actual double-key ordering call. It counts the
planned key, caller and alternate permutations, both materialized key buffers,
CUB temporary storage, and the duplicate flag. The measured requested-byte
high-water is 33,769,475 at 600K and 900,275,715 at 16M; the planner and
resident preflight both reserve 64 bytes/point. Persistent-column accounting
alone is not an acceptable substitute for this allocation-lifetime test.

Runtime-placement units freeze the separate direct model, literal graph and
layout facts, 8-byte upload/download, zero packing/index work, one lane, 64N
memory, and the 600K/16M bounds. The compiled placement audit must construct
the same direct facts and match every calibration direction; the raw verifier
must authenticate every referenced report. The physical aggregate, automatic
process matrix, Compute Sanitizer memcheck/racecheck at 600K, Host Debug, and
leak-disabled Host ASan/UBSan are required before the selector is called
qualified. The existing broader sort matrix and all-four primitive sanitizer
evidence remain applicable to the implementation but do not widen automatic
admission.

## B0233 automatic label/NNDistance hybrid qualification

The automatic label-duplicates gate is a complete five-stage composition
proof, not a standalone adjacent-comparison benchmark and not a resident-
placement claim. The accepted graph is exactly option-free LAS ->
`label_duplicates(dimensions=Classification)` -> `nndistance(k=10)` ->
`assign(value="UserData = Duplicate")` -> option-free LAS. Input must be
uncompressed LAS 1.4 format 7 with a 375-byte point offset, 36-byte records,
zero VLRs/EVLRs, no trailing bytes, and 250K--16M points on the pinned SM89
profile.

The calibration ladder must include the measured 50K loss and the exact
positive 250K, 1M, 2M, 4M, 8M, and 16M rows. Fit this complete hybrid with its
own host/device affine curves; do not borrow the ordinary label or NNDistance
resident models. Recheck a disposable resident implementation before choosing
the executor and retain it only if it beats the hybrid end to end. B0233's
resident probe remains 4.3% slower and is reverted.

The public process matrix must prove:

- exact option-free and required-route output, stdout, stderr, status,
  metadata, and order at the 250K floor;
- host fallback at 50K, on format/layout drift, with CUDA disabled or
  unavailable, and after an injected recoverable label-device failure;
- selection refusal for root/stage/option/value/writer grammar drift, with no
  output under the required-route proof;
- actual CUDA use by the label, NNDistance, and assignment bridge rather than
  selection alone; and
- oracle-identical default and required-route status, streams, and filesystem
  state for existing, input/output-aliased, and symlink destinations.

Unit tests freeze the literal matcher, the independent model's decision at the
250K floor and 16M cap, fact-free graph preservation, all three automatic
markers, and experimental-mode isolation. The append-only B0233 record freezes
the fitted coefficients and source-report hashes. Header probes fail closed
unless all measured layout facts agree. The resident automatic selector must
decline this literal grammar before device/profile probing so it neither
preempts nor adds failed-placement startup to the faster hybrid.

Run the full Host Debug, leak-disabled Host ASan/UBSan, and physical CUDA unit
aggregates; the focused host process and benchmark-runner gates; the public
automatic matrix; and Compute Sanitizer memcheck/racecheck on the actual 250K
automatic process. Authenticate all retained reports. This selector is
complete-process calibration outside the ordinary resident placement manifest,
so it must not silently alter resident model counts or inherit a resident
calibration claim.

## B0234 automatic skewness direct-composition qualification

B0234 promotes only the exact literal
`readers.las -> filters.skewnessbalancing -> writers.las(extra_dims=all)`
composition. Admission tests pin the one-key object root, exact stage and
option sets, lowercase `.las` endpoints, uncompressed LAS 1.4 format 7,
36-byte input/output records, mapped-source identity, no source extra
dimensions, finite comparator-unique physical Z, the 450K--16M count envelope,
and the exact SM89 profile. Nearby grammar, options, capitalization, layout,
compression, source, count, device, and profile facts must refuse selection.

The calibration suite contains an exact 50K--16M direct ladder. Rows through
300K are explicit loss controls; 350K does not clear the 10% margin. A separate
nine-pair public 400K control also fails the margin at 1.065118x. Required-route
public positives at 450K and 1M prove exact bytes, metadata, point order,
stdout, stderr, status, route selection, successful rewrite, and actual CUDA
ordering use. A bounded 16M case freezes the calibrated cap without creating
an open-ended large test. A deterministic 16,000,001-point derivative at
SHA-256 `1b40ab4052fe89c17c9ea0b6be33d83f25be42b510f1a805cfc058a433f764bb`
proves ordinary oracle fallback and required-route 124/no-output immediately
above it.

Placement and runtime preflight use the same conservative 65-byte/point peak.
The process matrix accepts exactly `65*N` bytes and rejects `65*N - 1`.
Tie, signed-zero-equivalent, and non-finite fixtures positively prove the
data-dependent fallback. Injected source/preflight/execution-proof failures
prove that a selected but unexecutable route is not reported as success.
Existing-output, input/output-alias, and symlink cases compare status, stdout,
stderr, destination state, and side effects with the pinned oracle; the direct
publisher commits only after successful publication.

Focused placement/model units, benchmark-runner proof mutations, and the
automatic process matrix accompany the existing skewness wrapper, ordering,
publisher, and resident tests. The final compiled placement audit is 190/190
and raw calibration verification is 196/196. Host Debug and leak-disabled
ASan/UBSan each pass 496 tests with four optional skips. The physical CUDA
Release runnable set passes with 17 optional skips after the unrelated outlier
matrix's stale timeout is corrected and its isolated 122.17-second run passes.
The actual required 450K route reports zero memcheck errors and zero racecheck
hazards, errors, or warnings.

## B0235 automatic HAG-NN count-one direct-composition qualification

B0235 promotes only the literal
`readers.las -> filters.hag_nn(count=1) -> writers.las(extra_dims=all)`
composition. Admission tests pin the single-key object root, exact stage and
option sets, lowercase `.las` endpoints, uncompressed LAS 1.4 format 7,
40-byte input records with exactly one unsigned-32 `OffsetTime` Extra Bytes
descriptor, 48-byte output records, mapped-source identity, the
450K--16,000,002 count envelope, one planner-owned 2D kNN index/region/lane,
and the exact RTX 4090/SM89/CUDA 13.3/driver 610.43.03 profile. Nearby grammar,
options, capitalization, layout, compression, source, count, device, and
profile facts must refuse selection.

The calibration manifest contains an exact 50,001--16,000,002 direct ladder.
The 50,001, 100,002, 250,002, and 350,001 rows lose; 300,000 is effectively
parity at 1.006644x. The 400,002 row reaches only 1.104286x, so the selector
retains the safer 450K floor rather than treating the first marginal direction
as sufficient public evidence. Required-route public positives at 450K,
1,000,002, and the measured cap prove exact bytes, metadata, point order,
stdout, stderr, status, route selection, executable rewrite, one selected HAG
region/index/lane, the mapped-source/direct-double publisher, and actual CUDA
execution. A deterministic 16,000,003-point derivative at SHA-256
`e2c7985b4a7c45fc39738e8583b55812ba0133d1f7ad7069fe5a0fed0718d042`
proves ordinary oracle fallback and required-route 124/no-output immediately
above the cap.

Runtime placement and resident preflight use the same conservative
160-byte/point peak; the stage-local count-one executor scratch remains 15
bytes/point. The process matrix accepts exactly `160*N` bytes and rejects
`160*N - 1`. The placement proof separately freezes the 25-byte/point upload,
8-byte/point spill, zero packing, 112-byte/point planner-owned index, and two
synchronizations. Missing source Extra Bytes, `count=2`, stage/writer option
drift, array root, uppercase endpoints, disabled mapped source, injected
preflight failure, and injected execution-proof failure all retain the
unchanged public host path.

Existing-output, input/output-alias, and symlink cases compare status, stdout,
stderr, destination state, and side effects with the pinned oracle. All such
refusals remain before commitment, and the required form exits 124 without
output. The compiled placement audit is 201/203: the only disagreements are
the deliberate conservative 300K/parity and 400,002/marginal-win refusals;
every admitted row and losing control agrees. Host Debug, leak-disabled
ASan/UBSan, and physical CUDA Release focused/aggregate gates pass with only
the two documented optional-corpus skips. The actual required 450K public
route reports zero memcheck errors and zero racecheck hazards, errors, or
warnings.

## Exact `extra_dims=all` LAS sink

D0222 admits only an uncompressed `writers.las` whose only non-routing option
is the literal `extra_dims=all`. Planner units must prove that envelope native
and keep compressed LAZ, named extra dimensions, and every additional writer
option non-native. They must also cover an exact ordering/skewness producer
whose spill has no columns but whose descriptor declares one 64-bit
permutation entry per input point; a fixed-size product, a different width, or
an order-preserving descriptor is invalid for that publication contract.

The complete-process gate compares candidate and pinned oracle artifact bytes,
stdout, stderr, status, and LAS point-record stride. The physical selection
matrix includes `normal(knn=8) -> covariancefeatures(knn=8,
Dimensionality) -> extra_dims=all` through the public `gpupdal pipeline` command.
The measured compressed format-7, 1M-point, 36 -> 100-byte case sets
`PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT=1`, requires an all-native
plan, verifies that placement's output width equals the emitted stride, and
requires the `planner_resident_shared_index` executor. B0285 adds a deterministic
274,625-point generated format-6/30 positive and neighboring format-6/31,
format-8/38, compression, and carried-Extra-Bytes refusals to the automated
matrix. Separately retained complete-process reports qualify only the blind
35.98M-point format-6/30 -> format-7/100 VEIL tuple and the 47.48M-point
format-8/44 -> format-7/120 AHN4 tuple against a clean independently built
upstream oracle. A separate compact format-7/48 -> format-7/112 source with
carried Extra Bytes must remain artifact/stdout/stderr/status exact while
automatic placement fails closed with `mixed_calibration_models`. Width ranges
and their interior cross-products are not evidence. The plain
composition has separate exact-pair controls: only uncompressed format-7/36 or
compressed format-8/44 input may publish uncompressed format-7/36 output;
format 6, compressed format 7, waveform, wider, compressed-output, and changed-
stride variants must refuse automatic placement. A public proof
mutation combines the required-route switch with
`PDG_TEST_AUTOMATIC_RESIDENT_PREFLIGHT_FAILURE=1` and must exit 124 without
creating an output, proving that placement alone cannot satisfy the route
contract after rewrite or resident preflight declines. Post-publication proofs
also require a midpoint-truncated LAZ and a partial plain LAS with a stale zero
header count to fail nonzero. The strict direct-sort
and direct-skewness CUDA processes
remain mandatory because their permutation-only publication exposed the
original zero-column failure. Run the focused planner and process cases under
Host ASan/UBSan. No new Compute Sanitizer run is implied when this host/planner
contract changes without a CUDA translation-unit change.

The explicit tuple gate is not an exact fixture-count gate. Once layout,
compression, graph, and writer shape match, the active profile estimates
interior cardinalities from its calibrated point-count curve, bounded to
250,000--47,478,228 on the embedded SM89 profile. Unit tests cover interior
format-6/30 and format-8/44 counts, while the 274,625-point physical format-6
case proves the public route remains exact. Those controls validate selection
and compatibility only; performance publication remains limited to retained
same-machine oracle pairs such as B0285's named full-size tiles.

B0224/D0223 extend that gate only for exact `.laz + extra_dims=all`. Planner
units must admit implicit compression and boolean/string true with no other
writer option, while refusing `compression=false`, named dimensions, and any
additional option. Runtime-placement units must prove the measured 1M,
compressed format-7, 36 -> 100-byte normal/covariance row selects device and
must use eigen-family and rank/optimal negative controls to prove no older
composition model inherits the compressed sink.

The physical matrix must exercise the public required route separately for
implicit, boolean, and string compression spellings. For each, candidate and
oracle artifact/stdout/stderr/status/order must match; the separately observed
resident stats must report an all-native plan, the actual output stride, and
`planner_resident_shared_index`. The post-placement preflight mutation remains
mandatory and must exit 124 without output. The current matrix has twenty
cases. The checked-in r6 benchmark and its frozen-time full profile are the
only performance evidence for compressed `all`; other counts, layouts, and
compositions retain host selection.

B0225/D0224 is the negative control for named r2 publication. The checked-in
r2 reference benchmark must compare the complete public command, not a writer
microbenchmark: its current five-pair result is exact at 0.988063x median and
0.986717x +/- 0.004381 paired. Explicit resident stats must continue to expose
the named writer plus the two independently uncalibrated SMRF/HAG regions.
Named `HeightAboveGround=float32` admission may advance only together with a
complete exact stage candidate that beats pinned PDAL; writer exactness alone
is insufficient under D0208.

## Exact r1 direct-delegation gate

B0226/D0225 adds a thin-dispatcher gate, not a stage implementation. Unit tests
must positively require the literal r1 root, stage order/options, materialized
bounds, SRS pair, LAZ spelling, and measured 1M input facts. Negative controls
must vary count, compressed layout, bounds, root keys, stage options/order,
SRS, and compression spelling while supplying otherwise-valid facts.

The process-boundary test builds a bounded synthetic public header with the
measured file-size/layout facts and proves direct oracle selection. It must
also change the count and compression bit independently and observe engine
selection, pass a pipeline CLI modifier and an engine behavior override and
observe `pdg-engine`, and point `PDG_ORACLE_PDAL` at a failing wrapper to
compare stdout, stderr, and exit status across the direct exec boundary. The
final complete-process benchmark remains the authority for real data: nine
alternating pairs against pinned PDAL plus a separate paired `pdg-engine`
control, with artifact, diagnostic, and materialized-pipeline hashes recorded.
Expanding the header-calibrated dispatch envelope requires a new measured row.

## Exact r5 direct-delegation gate

B0228/D0227 adds a second thin-dispatcher gate, not a COPC or stats
implementation. Unit tests must require the literal single-key root, exact
COPC/stats/LAS stage order and option sets, materialized bounds, floating
resolution 1.0, integer one request, lowercase `.copc.laz` input, and lowercase
uncompressed `.las` output. Negative controls vary root form, endpoint,
bounds, numeric representation, request count, stats options, and stage order.

The process-boundary test proves the plain public invocation selects the
oracle and that a pipeline CLI modifier or request drift selects the engine.
The final complete-process authority is nine alternating pairs against pinned
PDAL plus a separate paired `pdg-engine` control, with artifacts, diagnostics,
status, order, pipeline hashes, and binary hashes recorded. No input identity
gate is required because either route executes the identical pinned-oracle
pipeline and only fixed engine process work is removed. Expanding beyond this
literal grammar requires a new measured row.

## Exact r3 direct-delegation gate

B0229/D0228 adds a thin-dispatcher gate, not an SMRF, range, or GDAL
implementation. Unit tests must require the literal single-key root, exact
LAS/SMRF/range/GDAL stage order and option sets, lowercase non-COPC `.laz`
input, option-free SMRF, literal `Classification[2:2]`, and lowercase `.tif`
output with floating resolution 1.0 and IDW output type. Negative controls
vary root form, endpoint spelling/type, numeric representation, SMRF/range/
writer options, and stage order. Both lowercase and mixed-case COPC suffixes
must fail closed.

The process-boundary test proves the plain public invocation selects the
oracle and that a pipeline CLI modifier, an engine environment override, or
range drift selects the engine. The final complete-process authority is nine
alternating pairs against pinned
PDAL plus a separate paired `pdg-engine` control, with artifacts, diagnostics,
status, order, pipeline hashes, and binary hashes recorded. No input identity
gate is required because either route executes the identical pinned-oracle
pipeline and only fixed engine process/control-plane work is removed.
Expanding beyond this literal grammar requires a new measured row.

## Exact r2 direct-delegation gate

B0230/D0229 adds a thin-dispatcher gate, not an SMRF, HAG-NN, named-writer, or
LAZ implementation. Unit tests must require the literal single-key root, exact
LAS/SMRF/HAG-NN/LAS stage order and option sets, lowercase non-COPC `.laz`
endpoints, option-free SMRF and HAG-NN, string compression true, and exact
`HeightAboveGround=float32` publication. Negative controls vary root form,
endpoint spelling/type, case-insensitive COPC suffix, stage options/order,
compression type, named layout, and additional writer options.

The process-boundary test proves the plain public invocation selects the
oracle and that a pipeline CLI modifier, an engine environment override, or
named-layout drift selects the engine. The final complete-process authority is
nine alternating pairs against pinned PDAL plus a separate paired
`pdg-engine` control, with artifacts, diagnostics, status, order, pipeline
hashes, and binary hashes recorded. A hashed `/bin/true`-oracle trace must show
structural refusal before point work. No input identity gate is required
because either route executes the identical pinned-oracle pipeline and only
fixed engine process/control-plane work is removed. D0224's native named-writer
prohibition remains unchanged. Expanding beyond this literal grammar requires
a new measured row.

## Exact r4 automatic hybrid-outlier gate

B0227/D0226 adds automatic performance selection around an existing exact CUDA
executor; it does not add a sample implementation. Test-first rewrite units
must require the literal root, five-stage order, outlier/range/sample values,
string compression spelling, lowercase LAZ endpoints, 1M count, measured
header summary, and the full-file FNV-1a-64 fingerprint. Negative controls must
vary root form, endpoint, count, layout, extent, and one payload byte,
each stage's option grammar, compression type, and experimental substitution.
Unselected fact-free and neighboring-fact passes must not rewrite outlier or
the adjacent range.

The public proof sets `PDG_REQUIRE_AUTOMATIC_R4_OUTLIER_CUDA=1`. Success
requires both the final automatic rewrite and actual CUDA use inside the
outlier plugin; recoverable CUDA fallback must fail the proof. Run it only on
the exact calibrated RTX 4090/SM89/CUDA 13.3/driver 610.43.03 profile. The
focused physical lane also runs the dispatcher boundary and exact statistical-
outlier and fused-range CUDA differentials.

The committed `pdg_automatic_r4_outlier_cuda_exact` process lane uses the real
reference LAZ through the public dispatcher for the positive proof and exact
differential. Its refusal controls mutate point count, extent, and one payload
byte independently, use an uppercase endpoint, and inject a recoverable CUDA
failure; each must exit one with no output and the relevant selected/used proof
diagnostic.

Performance acceptance is the materialized complete checked-in r4 pipeline,
not an outlier microbenchmark: warm each executable, run at least nine
alternating public pairs, and compare artifacts, stdout, stderr, status, order,
and materialized-pipeline hashes. Profile exact-repair rows before widening the
count because the 4M hybrid path has a known repair cliff. Any new count,
layout, grammar, device, toolkit, or driver is a distinct calibration row.

## Experimental direct resident LAS publication

D0103/B0040 adds an opt-in output envelope after one exact, linear,
order-preserving, cardinality-preserving resident region. The default LAS
reader/writer must have no options, every serialized write must be `UserData`,
and input and output must be distinct uncompressed LAS paths. Unsupported
graphs retain the ordinary writer; `PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT=1`
turns a refusal into a process failure and positively proves selection.

The canonical-overlay unit starts from the exact LAS 1.4/point-format-7 image,
changes only byte 17 of each 36-byte record, and rejects count or record-format
mismatches. The physical resident-v3 process differential runs both the normal
and required-direct publishers against one pinned oracle artifact, requires
one selected region and one planner-owned index, checks the executor and
calibration flags, and compares complete output bytes. B0041 must add
empty/small, unsupported serialized-write and writer-option cases plus
existing-output/temporary cleanup failure coverage before this path is eligible
for performance qualification or automatic placement.

B0041 supplies those first qualification gates for the named 4M LOF envelope.
The host process test requires exact small and zero-point GeoTIFF-VLR fallback
when the experimental endpoint is requested below placement; the latter also
locks default `pdal pipeline` stream preference for unselected resident graphs.
The physical v3 lane requires normal and direct outputs to match one pinned
oracle artifact, then proves an existing target is untouched and that an
additional serialized Classification write or `extra_dims=all` writer refuses
before creating output. The canonical overlay accepts an empty canonical image.

Performance qualification uses separate clean five-sample resident-control
and required-direct reports, each with one warmup, alternating oracle/candidate
order, frozen time, full artifact hashes, and stats-guarded executor identity.
The surviving-boundary capture records exact-repair rows/time, upload pack,
spill wait/publication, transfer bytes, and index-build count. These tests
qualify only the named explicit-resident envelope; option-free selection remains
B0042 and must add public CLI flag/metadata/fallback coverage.

B0042 adds `PDG_REQUIRE_AUTOMATIC_RESIDENT_LAS_OUTPUT=1` as a positive public
`gpupdal pipeline` proof. A below-placement host case must fail without output; the
physical v3 lane must produce a fourth artifact through the public command and
match the same upstream output as ordinary and explicit-direct resident
execution. The benchmark runner strips ambient endpoint flags, injects this
proof only into candidate processes, and accepts it only with
`--candidate-mode default` plus a pipeline. Its clean repeated report therefore
cannot pass through the explicit `resident` command or silently delegate.

At the B0042 checkpoint this remains a proof-switched path. B0043 therefore
requires separately queryable admission with no output or diagnostics plus
unsupported graph/options, `--stream`/`--nostream`, metadata, error bytes,
device/profile/preflight refusal, and existing-output coverage before
option-free automatic selection.

B0043 completes that boundary at `62016556b`. The public selector calls a
separately queryable admission path only for an option-free `pipeline FILE`
command. Static JSON admission requires the exact explicit
`readers.las -> filters.lof,minpts=10 -> UserData conditional assign ->
writers.las` shape. Input/output translation state is checked before CUDA;
device/profile/runtime placement and a rewritten-pipeline preflight must then
accept before the executor commits. Any refusal returns without output or
diagnostics to the existing public selection chain. Once the execution manager
is committed, errors are terminal and cannot cause a second execution.

The host process matrix compares below-placement fallback, `--stream` failure,
`--nostream` success, complete metadata bytes, missing-input diagnostics, and
an unsupported graph to pinned PDAL. The hash-pinned physical v3 gate compares
ordinary resident, direct resident, default public automatic, and pinned-PDAL
outputs; positive proof guards ensure both existing-output and injected
preflight refusals preserve the target and create no temporary artifact. The
gate passes in 281.94 seconds. Clean B0043 timing uses the actual default public
command with `automatic_resident_las_output_required=false`: one warmup and
five alternating samples are exact at a 3.723975123-second candidate median
versus 35.738760555 seconds pinned PDAL (9.596939x). The automatic claim is
limited to this explicit shape and runtime placement envelope; other resident
publishers remain explicit. B0044 must measure each proposed additional
producer/point-operation shape before admission is widened.

B0044 applies that prototype-first rule without changing selection. The exact
4M direct-resident coplanar/ferry, normal/assign, and nndistance/assign shapes
each run once against pinned PDAL under a required
`planner_resident_shared_index_direct_las` stats proof. All three compare full
status, diagnostics, and LAS bytes. Only the largest-ratio/lowest-wall shape
may advance. Nndistance wins at 5.275101x; coplanar and normal remain diagnostic
hypotheses. The separate nndistance stats capture originally reported zero
repaired rows because NND did not yet publish repair telemetry; B0049/D0112
supersede that inference. It records upload/spill phases, while an 18-launch
basic Nsight Compute report identifies the 138.57-millisecond
`knnGatherKernel` as dominant device work.
B0045 must use the maintained template, repeat the complete-process comparison,
and add public-boundary differentials before any automatic admission change.

B0045 qualifies only the exact explicit
`readers.las -> nndistance(k=10) -> UserData = 1 WHERE NNDistance >= 0.4 ->
writers.las` shape. The clean explicit-direct and option-free lanes each use
one warmup and five alternating frozen-time samples and compare complete
status, stdout, stderr, and LAS bytes against pinned PDAL. Their exact medians
are 5.302170x and 5.228585x pinned PDAL. Automatic selection must remain
side-effect-free through admission: the host process matrix proves exact
below-placement fallback, required-proof refusal without an output, and
unsupported `mode=avg` fallback. Once rewritten execution commits, errors are
terminal and cannot retry through PDAL. The expanded 21.97M physical v2 gate
must additionally prove the option-free path byte-identical to pinned PDAL;
it passes in 236.60 seconds. This changes only the named public envelope and
makes nndistance the ninth automatically selected filter. B0046 must add
stats-only phase attribution before any optimization is proposed, because the
retained device profile places only 138.57 milliseconds in the dominant
kernel.

B0046 adds one additive `execution.pipeline_phase_seconds` object only when a
resident command writes stats. Its five non-overlapping fields cover command
work before stats, validation/placement/preflight, rewritten manager execution,
canonical LAS publication, and residual control. The host process test requires
nonnegative fields, positive validation and manager time, zero direct
publication on an ordinary host pipeline, and an exactly reconciling
partition. The physical direct-resident gate requires positive publication as
well. CUDA Release and Host Debug focused process tests pass.

The clean 4M nndistance and LOF captures must use the standard frozen epoch and
compare complete output bytes to pinned PDAL. Their established exact hashes
remain
`174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`
and `e68b4dded6f5571d8eb45afad9fa5698b4a034d56f7083d39b3ab8002fa80a3a`.
Manager execution contains 92.90% and 93.79% of internal wall, while
publication is only 51–52 milliseconds. B0047 must split reader/row-table
work from resident wrapper/index work before any optimization. A stage/kernel
change, wider selection envelope, or new port cannot be justified by the
B0046 aggregate.

B0047 partitions only the eligible direct single-region shared-index manager
timeline. Manager entry through `terminal->execute` entry is graph construction
and preparation; execute entry through the first accepted upload marker is
reader/row-table materialization plus initial `Stage::execute` dispatch and
finalization; accepted upload through completed spill is the resident
boundary/wrapper/index/filter interval; and completed spill through execute
return is post-spill stage control. The intervals must be contiguous,
non-overlapping, nonnegative, and exactly reconcile to
`pipeline_phase_seconds.rewritten_manager_execution`. Host, multi-region, and
other direct executors must report a null breakdown.

The physical LOF and nndistance direct paths each require positive reader and
resident spans, exact output against pinned PDAL, and one planner-owned index
build. In particular, the v2 gate must keep its option-free automatic
nndistance comparison and also run a stats-enabled required-direct nndistance
comparison against the same oracle; it passes on the 21.97M fixture in 257.58
seconds. Clean 4M attribution captures place only 0.502592322/0.509743845
seconds in nndistance/LOF reader-table work, versus
2.524453274/2.901305423 seconds in their resident intervals. B0048 must split
planner-owned index, neighborhood projection, and adjacent bridge work inside
that interval before any optimization is proposed.

B0048 adds `execution.resident_work_breakdown_seconds` only when the existing
stats-enabled direct, single-region neighborhood manager timer is armed. Its
index-configuration, index-build, neighborhood-query/projection, and adjacent
point-program-bridge fields are mutually exclusive host-call wall spans inside
the outer resident interval. They are not exclusive GPU timings: queued index
work may complete during a later query synchronization. Boundary and repair
counters are nested drill-downs inside the outer interval and must not be
added to the four spans; LOF repair is specifically inside the broad query
span. Host and inapplicable executors must report a null object.

Positive LOF and nndistance tests require every call span to be nonnegative
and the four applicable spans positive. Their sum must fit inside the B0047
resident interval, and the emitted nonnegative residual must reconcile the
outer interval exactly. The 21.97M v2 NND gate passes in 256.54 seconds with
these assertions and exact pinned output. Clean final-binary 4M captures keep
the established NND and LOF hashes. B0049 must split the dominant NND broad
query span without adding a synchronization; any pageable-status-copy
explanation remains a hypothesis until those call boundaries are measured.

B0049 adds `execution.nndistance_query_breakdown_seconds` only under the same
stats-enabled direct single-region manager envelope. Output preparation,
query submission, status allocation, result/status transfer calls, explicit
wait, status scan/repair, and publication are sequential host call-wall spans
nested inside the broad query timer. They do not identify exclusive GPU time:
queued device work may complete in a later call. Their sum plus the emitted
nonnegative residual must reconcile to the broad query total; inapplicable and
host executors report null.

The forced one-shell four-point CUDA regression must match upstream exactly,
positively record incomplete/repaired rows and repair time when stats timing is
armed, and leave all three fields at zero without stats. The final frozen 4M
capture records one incomplete row and 2.061175139 seconds of exact host repair
inside a 2.253729404-second broad query; the status-transfer call is only
0.138355024 seconds. A forced planner-owned Morton-BVH prototype must retain
the same bytes, but its measured 4.179201213-second complete candidate rejects
that direction against the 3.278321613-second grid/repair capture. B0050 may
advance only from a cheap selective device-repair prototype that improves the
complete frozen pipeline without private index state or a selection change.

B0050 separates repair telemetry by execution backend. `exact_host_repair`
retains only host elapsed time and ambiguous/incomplete/repaired counts;
`exact_device_repair` reports its own elapsed time and incomplete/repaired
counts. Resident schedules emit a zero-valued object when no device repair
applies; non-resident executors report the field as null. Stats-disabled
execution must not mutate either accumulator. The benchmark runner's
`--require-nnd-device-repair` switch clears inherited NND repair overrides,
sets the proof variable only for candidate runs, fails when no bounded repair
occurs, and records the requirement in the report.

The production device envelope is kth mode with internal neighbor count 2..16
(public `k=1..15`) and at most 16 incomplete rows. One 64-thread block scans
the planner-owned resident XYZ columns for each row, retains each thread's
local top-k in exact `(distance, point-id)` order, and exactly merges their
union; no index state is allocated. The kth distance is invariant to tied
candidate identity. Average mode, public `k>=16`, more than 16 incomplete rows,
an explicit disable, and every unsupported request retain upstream KD3 repair.
The require switch must fail closed for zero incomplete rows or any ineligible
case.

Focused physical tests compare exact upstream outputs for k=1 and k=15,
awkward large-offset binary64 coordinates, an equal kth-distance boundary, and
the established one-shell fixture. Negative cases positively cover k=16,
average mode, more than 16 incomplete rows, require-plus-disable precedence,
and zero incomplete rows under Morton BVH. The final automatic 4M LAS process
must require both the automatic resident output and device-repair proofs and
compare the full artifact byte-for-byte. Memcheck, initcheck, racecheck, and
synccheck cover the bounded positive and fallback matrices; the 21.97M v2 gate
repeats the full lifecycle and exact artifact comparison.

B0051 is a disposable viability measurement, not a new test or performance
qualification. Its 4M probe must map and validate the established LAS through
the existing parser, reuse the existing device coordinate transpose, expand
with the pinned separate multiply/add order, and compare every decoded
binary64 coordinate with the host decoder after timing. Allocation, mapping,
and the five post-warmup hydration samples are reported separately. B0052 may
retain production code only inside the already-qualified default-LAS NND
envelope. It needs a positive proof that the direct source supplied the
planner-owned columns, negative/fallback tests before any side effect, the
existing full-artifact differential, and an alternating complete-process
comparison with B0050 before automatic selection changes. Functional support,
GPU-native coverage, performance qualification, and automatic selection stay
independent throughout this gate.

B0052 satisfies that gate only for the existing exact default-LAS
`nndistance(k=10) -> UserData assign -> default LAS` envelope. Unit coverage
proves all mapped coordinates match the host decoder even when PointView XYZ
is deliberately wrong. Process coverage proves require-plus-disable fails
closed, the ordinary source-disabled path remains byte-exact, and a threshold-
positive fixture changes UserData before the full artifact is compared with
the pinned oracle. The clean benchmark must require automatic resident output,
bounded NND device repair, and direct LAS source use; the retained report must
record a clean commit and identical status, diagnostics, and artifact bytes.
Memcheck, initcheck, racecheck, and synccheck cover coordinate expansion plus
source hydration. The hash-pinned 21.97M v2 gate repeats automatic, direct,
disabled-source, lifecycle, and exact-output coverage before promotion.

B0053 profiles the final selected pipeline with automatic output, direct
source, and bounded device repair all positively required. The complete
artifact must remain pinned-oracle exact after profiling. A disposable launch
sweep may vary only the shared uniform-grid kNN block size, must rebuild
serially, and must compare the complete artifact after every valid run. Any
invalid launch is a rejected prototype, not a fallback result. Restore and
rebuild the qualified 64-thread source before recording the decision.

B0054 adds an optional nested breakdown beneath
`execution.pipeline_phase_seconds.validation_placement_preflight`. The three
outer spans and the three runtime-placement subspans must be nonnegative and
must reconcile to their parent within one nanosecond without adding a CUDA
synchronization. Host Debug and CUDA Release process tests check both sums.
The exact 4M proof must retain its established output hash while recording the
fresh partition. Device/profile discovery and initial memory-budget placement
are process startup; their measured wall is not an optimization claim.

B0055 partitions planner-owned index configuration only in stats-enabled
execution. The nested configuration/validation spans must fit within and
reconcile to the existing index-configuration interval. Core tests require
the kNN grid, Morton BVH, and adaptive builders to return exact host envelopes
in both two and three dimensions and to reject nonfinite coordinates. The
direct process lane requires zero second-scan time for kNN while radius-index
validation remains unchanged. Forced grid/BVH NND differentials, the 16M
direct-resident process lane, and all four focused CUDA sanitizers must pass.
The clean automatic qualification requires direct source and bounded repair,
compares complete diagnostics and LAS bytes, and alternates five samples with
pinned PDAL.

B0056's disposable finite-extrema reuse must preserve those builder contracts,
the deterministic adaptive probe, and forced-backend semantics. Exact frame
comparison and output are necessary but not sufficient: retain the change only
after material option-free complete-process improvement. Its 0.834/0.825/0.844
second samples fail that gate, so the prototype is reverted without a
certification lane.

B0057 adds stats-only spans around direct-source coordinate hydration and
resident setup without adding a CUDA synchronization. Both must fit inside the
existing wrapper interval and reconcile with its profiled total; the direct
source lane requires positive hydration while ordinary PointView-backed LOF
requires zero. The exact 4M artifact and hash-pinned 16M direct process lane
must pass unchanged.

B0058 nests allocation/materialization, transfer/kernel submission, and
final-wait spans inside hydration without adding synchronization. They must be
nonnegative, fit within hydration, and reconcile with its total; the direct
source requires all three positive while PointView-backed LOF reports zero.
The exact 4M artifact and hash-pinned 16M process lane remain unchanged.

B0059's disposable summary probe must derive extrema and every adaptive sample
from point records, decode with the same coordinate operation, and compare the
resulting grid/backend choice bit-for-bit with the established host-column
builder. Any later lazy-host prototype must also prove that every host-repair
branch reads the exact original mapped coordinates before PDAL KD3 use. LAS
header bounds remain insufficient because the reader envelope does not
validate them against point records.

B0060 productionizes that summary only inside the already-qualified direct
default-LAS NND envelope. Focused process differentials must cover the ordinary
adaptive fixture, a clustered fixture that selects the alternate backend, and
forced Grid and forced Morton-BVH. A positive process proof must observe both
record-summary use and absence of a host XYZ mirror. Force at least one exact
host-repair branch and verify that mapped PointView coordinates, not an absent
resident host column, feed pinned KD3. Run the cheap complete-process gate
before the alternating benchmark; retain the change only for material wall
improvement, then repeat the existing hash-pinned large direct-source lane and
all four focused CUDA sanitizers.

B0060 passes that gate at clean commit `07ea2f4a7`: ordinary adaptive,
clustered adaptive Morton-BVH, forced Grid/BVH, malformed-summary rejection,
source-disabled `pipeline`/`resident` proof failures, and a failed-config
summary-latch regression are covered. The 4M automatic benchmark requires the
direct source, record summary, absent host XYZ mirror, and bounded device
repair on every candidate run; all outputs match pinned PDAL byte-for-byte.
The explicit device-repair-disabled process case positively observes mapped
PointView/KD3 host repair. Memcheck, initcheck, synccheck, and racecheck remain
clean, and the hash-pinned 16M v2 process lane must stay green.

B0061's now-reverted prototype could suppress host NNDistance publication only
when the planner proved an immediate resident assignment bridge was the sole
consumer and the selected direct LAS writer could not encode that intermediate
dimension. Its positive hook had to observe both skipped host publication and
device-column bridge reuse; every other consumer, writer, fallback, and
unsupported shape retained ordinary publication.

B0061 passes the exact/proof gate at clean prototype commit `cf5ff3344`, but
fails the material-wall gate: five automatic candidate samples have a
0.750254638-second median versus B0060's 0.755523193 seconds, a 0.697339%
reduction with overlapping samples. Commit `0b61e8ef2` therefore reverts the
product, proof, and runner changes before any sanitizer, large-corpus, or
qualification lane. This is a negative result, not a performance claim.

B0062 qualifies device-only NNDistance lifetime only in the same exact direct
LAS envelope. Status transfers before repair ownership is decided. Zero or
bounded device-repair rows avoid host NNDistance materialization and the full
result copy; a true host repair allocates/copies the complete column, runs
pinned KD3 repair, restores canonical device values, and publishes normally.
The clean process gate requires observed device-only handoff and assignment
reuse, exact bytes/status/stdout/stderr, one warmup and five alternating
samples. Focused tests require the absent 32,000,000-byte transfers on normal
execution and their exact D2H/H2D restoration under forced host repair. The
hash-pinned 16M process lane and memcheck/initcheck/synccheck/racecheck must
remain clean. B0063 profiles the residual status-call interval before any new
prototype.

B0063 records a profile-only negative gate for pinned status storage. The
accepted 4M endpoint's kNN gather takes 137.153696 milliseconds, which already
accounts for the magnitude of the 0.134898783-second host status-call interval;
the timer is synchronization with queued work, not evidence of a material
pageable-copy bottleneck. The same report measures 77.754496 milliseconds in
the one-row, one-block exact repair. B0064 may prototype deterministic
multi-block repair only within the existing bounded device-repair envelope.
It must preserve exact kth-distance output and host fallback, positively prove
the new path, and pass a complete exact process timing before qualification or
selection can change.

B0064 qualifies that parallel repair only inside B0062's exact direct-LAS,
device-only NND-to-assignment envelope. Focused tests cover a small exact
boundary, more than 64 partitions, a 16,385-point capped-128/grid-stride case,
generic-caller serial retention, disable precedence, and fail-closed required
proofs. The clean process gate requires automatic resident output, direct
mapped source/summary, no host XYZ mirror, device repair, device-only handoff,
and parallel repair on every candidate sample. A same-binary alternating
serial/parallel control isolates the repair. The final promoted Nsight report,
all four Compute Sanitizer tools, and the hash-pinned 16M v2 exact process lane
must remain clean. No generic resident caller becomes parallel by default
without a separate complete-pipeline gate.

B0065 is a deliberately non-certified 1M direction gate for statistical
outlier. It requires exact LAS bytes from stock PDAL and the forced CUDA lane,
separates a read/write-only floor, samples stock hardware cycles, and records a
basic-set CUDA profile. Three rough samples may choose B0066's prototype but
must not change performance qualification, placement calibration, or automatic
selection. B0066 must positively prove planner-owned index reuse, the absence
of the private coordinate upload/index boundary, exact incomplete-row repair
and global host classification, and a material complete-composition win before
any certification lane starts.

D0097 extends the bounded exact `filters.hag_nn` contract to explicit
`count=3`. The combined process matrix has 54 host cases and 46 physical CUDA
cases. Count-three rows cover unique interpolation on forced grid and BVH,
same-XY selection, exact `max_distance` equality, partial and negative
cutoffs, inclusive no-extrapolation bounds, outside-bounds behavior, large
finite coordinates, signed zero, subnormal squared distances, weight overflow,
finite-coordinate squared-distance overflow, nonfinite XY/Z, one/two/no-ground
repair, internal and third/fourth boundary ties, bounded-grid incompleteness,
and explicit count-four host ownership. Every repair row has a positive proof
switch; five consecutive physical CUDA matrices must remain byte-exact.

Direct wrapper tests pin the three-term upstream operation order and compare
NaNs by representation-aware semantics where the oracle produces them. A real
squared-distance overflow fixture separates finite coordinates by at least
`1.4e154` and positively proves tie repair on the UniformGrid backend. The
Morton BVH rejects that binary32-incompatible coordinate span before query, so
it is not an overflow tie proof; an ordinary representable candidate-boundary
fixture proves BVH tie repair instead. Strict
cutoff tests include equality, partial selection, and negative
`max_distance`. Resident tests execute terminal counts one/two/three and a
nonterminal count-three HAG-to-assignment bridge after native,
insufficient-ground, tied-fourth, and incomplete-grid results. Each
repair must republish the pinned host HAG column into the retained 2D product
and close the delegated lifecycle. Preflight budgets 39 bytes per point for
count three.

B0035 remains a positive forced, data-dependent gate. Its deterministic
1,000,002-point fixture measures the exact complete pipeline at 1.482696x
pinned PDAL. The 18-launch profile identifies masked `knnGatherKernel` as the
limiter at 2.68 ms, 74.13% compute throughput, 65.01% memory throughput, 64
registers per thread, and 48.93% achieved occupancy. The 59.55-microsecond HAG
projection is memory-bound. The 436-registration Host ASan/UBSan aggregate and
667-registration physical CUDA aggregate pass with their documented skips;
all four Compute Sanitizer tools are clean across the 12-test focused lane.
The D0096 no-ground case deliberately reaches an upstream assertion after its
diagnostic; sanitizer injection alone omits that row because glibc prefixes
the otherwise-identical assertion with `pdal:` versus `pdg:`. Release CUDA
does not execute that Debug assertion and its matrix passes. Host Debug exposes
the prefix mismatch and therefore fails the full process contract; D0221's
post-session audit records it as a pre-existing defect. No automatic model,
`count >= 4`, wider Delaunay, `filters.hag_dem`, or complete HAG-family claim
follows.

D0098 extends the same bounded contract to explicit `count=4`. The combined
process matrix has 74 host cases and 66 physical CUDA cases. Count-four rows
cover unique interpolation with a distinct fifth ground candidate on forced
UniformGrid and MortonBvh, same-XY selection, equality/partial/negative
`max_distance`, inclusive and outside bounds, large finite values, signed
zero, subnormal distances, weight overflow, nonfinite XY/Z, one/three/no-ground
repair, fourth/fifth boundary ties on both backends, UniformGrid squared-
distance-overflow repair, bounded-grid incompleteness, and explicit count-five
host ownership. Five consecutive physical CUDA matrices must remain
byte-exact.

Direct tests preserve the four-term pinned accumulation order and exercise
terminal counts one through four plus native, insufficient-ground,
fourth/fifth-tie, and incomplete count-four assignment bridges. Preflight
budgets 51 bytes per point: three masks/status bytes, four source ids, and four
binary64 squared distances. B0036 is historical dirty-snapshot diagnostic
evidence at 1.587001x pinned PDAL. B0105/D0166 supplies the clean current-binary
qualification for only that forced, data-dependent 1,000,002-point fixture at
1.562755x; neither record authorizes automatic selection. Host ASan/UBSan
covers 439 registrations and physical CUDA Release covers 678, with their
documented skips. Memcheck,
initcheck, racecheck, and synccheck cover the 12-test count-four, masked-index,
and resident-lifecycle lane.

## P1.5 descriptor-planned fusion lane

Fusion tests must prove both selection and execution. Planner unit tests cover
the declared purity, cardinality, dimension, `where`/`where_merge`, placement,
and deterministic contracts without inspecting stage names. The process test
`pdg_cuda_assign_ferry_fused` sets
`PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM=1`; it fails unless the complete middle
chain is a planner candidate, the bounded single-kernel envelope accepts it,
CUDA executes, and the candidate output is byte-identical to pinned PDAL.

The compact CUDA unit matrix exercises point formats 0, 1, 2, 3, 6, 7, and 8,
non-power-of-two chunks, both lanes, conditional writes, custom intermediates,
canonical field repack, final bounds, and ReturnNumber recount. Run its bounded
four-launch regression under memcheck, initcheck, racecheck, and synccheck.
Static cubin resources are diagnostic only; B1/E7 additionally require measured
occupancy and DRAM traffic for translation alone and for every accepted fused
chain shape.

## P1.5 bounded scheduler lane

The shared scheduler exposes only the implemented tiled classes: LAS
translation, fused point programs, ordered/compacting point programs, and
radius neighborhoods. Do not add a class name merely because a future kernel
could use it; adaptive kNN becomes a scheduler class only when its exact tiled
resident path exists.

For each class, sweep every integer lane count from two through six on one
fixed workload. Record the selected width in the raw report and use a proof
gate that fails if the intended CUDA path is not executed. Radius sweeps also
fail unless the requested lane count is actually active, preventing a tile-
count or VRAM clamp from creating a false wider-lane result. Compare complete
process artifacts to pinned PDAL on every warmup and measured invocation.
Standalone translation remains B3 calibration; fused, ordered, and radius
pipelines are B2. A lane sweep is never B1 roofline evidence.

The compact exact regression separately exercises all five widths for fused
LAS and radius tiling, verifies nonoverlapping sink publication and complete
mosaics, and constrains a six-lane request to a three-lane synthetic VRAM
budget. Run that pair under memcheck, initcheck, racecheck, and synccheck.
Ordinal dependencies and one-tile schedules must remain single-lane. Invalid
overrides outside `[2, 6]`, arithmetic overflow, and a budget unable to hold
one lane fail before launch.

B0010/D0053 fix the current default at two lanes for every implemented class.
Compact forced tiling is only scheduler validation. D0060 supplies the natural
V5/E3 gate for the generic resident point-program executor: the hash-pinned
21,970,934-point working set is 1,757,674,720 bytes, exceeds a lower-only
268,435,456-byte planner budget, and executes in 168 fixed 131,072-point tiles
with a 20,971,520-byte two-lane peak. Stats must report the input cardinality,
tile capacity, full-view bytes, budget, planned/observed peak, and lane reuse.
The complete artifact, streams, diagnostics, and exit status match pinned PDAL
standard-mode whole-view execution exactly. An oversized internal validation
budget must fail before writer execution. This satisfies V5/E3 for the generic
resident load only; neighborhood and adaptive-kNN tiled residency remain part
of V2/V3 and their own stage-family gates.

## P1.5 placement-model lane

The D4 audit compiles the same planner used by execution, supplies the frozen
SM-89 calibration record, and evaluates every decision-cited complete-process
row that existed when D0050 froze breadth. Each row records input and output
cardinality, record layout, its normalized pipeline, measured host/device
medians, and raw-report hashes. The audit must use planner-produced upload,
spill, packing, index-build, synchronization, topology-result, and peak-VRAM
terms; copying expected decisions into stage-name branches is forbidden.

Run both gates:

```sh
python3 scripts/pdg/verify_placement_calibration.py \
  test/data/pdg/placement-calibration-sm89.json --repo-root . --require-all
ctest --test-dir build/pdg-cuda-release \
  -R '^pdg_placement_model_matrix_audit$' --output-on-failure
```

The first gate validates every locally retained raw-report hash. Raw
calibration and benchmark reports must be retained under
`build/benchmarks/`, never `/tmp` or another volatile location: the seven
D0049 approximate-coplanar reports stored in `/tmp` were destroyed by the
2026-08-08 host reboots and survive only as recorded digests under an
explicit `physical_file: lost-to-environment` waiver (D0065). The waiver is
narrow — an unmarked missing report still fails `--require-all`, and a
restored file must match its recorded digest. The second
executes the C++ placement model and requires at least 90% measured-winner
accuracy. A missing calibration, unknown layout, out-of-envelope count, or
VRAM excess must select host. Result bytes that are not point columns—keep
masks, permutations, partition membership, and fixed reductions—must be
declared by descriptors, charged only at a spill, and included in peak memory.

B0011/D0054 record 52/52 correct calls after the initially observed small
approximate-coplanar miss added a fail-closed lower calibration envelope. That
passes E5 for the frozen matrix but does not close D4. Calibration rows remain
B3 and never qualify a stage.

D0055 moves that same calibration into the core and applies one measured
residual exactly once per maximal resident region. The manifest audit now
fails on any embedded profile id, infrastructure coefficient, model count, or
stage-residual mismatch before checking its 52 decisions. Profile matching is
exact across device name, compute capability, driver, and CUDA toolkit; every
other environment is uncalibrated and must select host until its physical lane
is appended.

Region tests separate two device candidates with an unsupported host stage and
verify selected-only upload/spill, full-record packing, topology-result,
synchronization, and peak-VRAM totals. They cover combined cold-start
amortization, a warm context rejected solely by the shared synchronization
toll, one memory-infeasible region beside one selected region, unknown/missing
calibration, and branch rejection. The current executor owns only linear
single-reader graphs, so a shared producer must fail closed even though the D1
lifetime model remains branch-safe. These are model-core tests, not V4 or V7:
closure still requires an executor whose implementation matches its calibrated
model, `--stats` predictions and materialized crossings, and complete-process
artifacts.

D0056 adds the explicit, production-default-off
`gpupdal resident PIPELINE --stats FILE` process observer. On an exact profile it
builds runtime facts, evaluates the core placement model, and either preserves
the host pipeline or forces selected point-program wrappers to execute CUDA.
Unknown profiles, configured reader/writer layouts, unsafe ferry mappings,
unmeasured program shapes, non-linear topology, and unsupported selected
regions fail closed. Stats files may not alias the pipeline, reader, or writer,
and stdout is never used for stats.

The schema reports `planned_boundaries` separately from
`observed_crossings`. An observed crossing marked
`inferred_from_region_transfers` is derived from the D2H total of one wrapper
and H2D total of the next; it is not evidence of a planner-owned full-record
D1 spill/upload. The schema also reports the selected executor and whether the
device calibration matches it. The current selected executor is
`pdal_point_program_wrapper`, for which that flag is false: B0005 calibrated
the direct fused-LAS two-lane executor, whereas the wrapper performs CPU
gather/scatter and one-stream per-tile transfers.

The physical V7 test is complete for the exact SM-89 profile: the small input
selects host, emits no CUDA events, and is byte-identical to the oracle. The
bounded V4-shaped test is hash-pinned, capped at a 1 GiB source, preflights
RAM/disk/VRAM, limits swap growth, and proves exact forced-wrapper execution
around `filters.randomize`. It does not satisfy V4/E2 or extend E5 because it
does not retain planner-owned storage across that host boundary and has no B2
winner measurement.

D0057 adds the separate `direct` physical gate for one terminal fused
assign/ferry chain. It must select the actual B0005 raw-LAS executor, consume
the runtime D1 VRAM budget, use the fixed D3 two-lane default, and report tile,
lane, peak-memory, reuse, and exact transfer totals. The candidate and pinned
PDAL must match output bytes, stdout, stderr, and exit status. The test also
seeds a distinct final path and requires the explicit resident command to fail
without replacing it. Caller lane/chunk overrides are removed for this gate;
the source is hash-pinned, at least 16 million points, and at most 1 GiB, with
the same RAM/disk/VRAM/swap guards as V4. This validates only the terminal
calibrated bridge. The next process differential must materialize and resume a
planner-owned device batch across the V4 unsupported-stage boundary; inferred
wrapper transfers remain insufficient for E2.

D0058 adds that resumable `v4` physical gate. Before executing the rewritten
pipeline, the gate prepares its graph, validates descriptor and physical-layout
semantics, computes the fixed two-lane schedule, allocates both packed lanes,
and probes their planned column high-water sequence. A compact
11-point/four-point-tile regression forces lane reuse around a
seeded host permutation and checks an untouched signed custom dimension. The
D0051 three-assignment regression separately proves that the enforced per-lane
resident ceiling follows the 16-byte/point last-use peak rather than the
19-byte/point region union; aggregate current allocations must also remain
below the planner's multi-lane ceiling. Both regressions run under all four
Compute Sanitizer tools.

The V4 process stats must preserve stable planned boundary IDs and report, per
boundary, point count, logical-column bytes, full runtime-record bytes,
predicted transfer bytes, predicted packing bytes, observed transfer bytes, and
observed bytes processed by explicit pack/repack operations. Packing bytes must
not be copied or inferred from transfer volume. Aggregate directional sums must
reproduce the planner and executor totals. Until
`boundary_accounting_matches_prediction` is true for the selected executor,
the test may establish the V4 validation load but must not close E2 or D4.
D0059 supplies a complete executor-declared fact table; V4 now requires that
flag to be true while separately retaining
`selected_device_calibration_matches_executor=false`. This closes E2 but not
D4. The same gate requires predicted configured/active lanes and peak VRAM to
equal the resident schedule. A budget that holds one lane but not the
calibrated multi-lane width must select host rather than borrow unmeasured
one-lane throughput. V7 additionally proves that a native region unsupported
by the PointView rewrite retains `calibration_default` accounting. The
21,970,934-point exact differential is serialized and retains the RAM, VRAM,
disk, source-size, and swap-growth guards.

D0061 adds the `v1` physical gate for the declared cardinality-changing
expression case. The fused chain, one `filters.expression`, and one
post-predicate assignment must form a single resident region whose rewrite,
runtime-placement admission, and preflight all accept the change from the
declared descriptor contract, never from stage names. The executor evaluates
the predicate into a planner-owned per-tile keep mask, spills the complete
input tile plus one mask byte per point, and appends survivors to the output
view in stable source order; the schedule reports the `ordered_point_program`
class with the fixed two-lane default and the observed output cardinality.
The gate requires exact planned-versus-observed boundary bytes including the
mask, a spill prediction exactly one byte per point above the upload
prediction, an output cardinality equal to the pinned oracle's survivor
count, byte/stream/status identity with pinned PDAL, and a where-bearing
small control that is rejected by declared semantics
(`non_cardinality_preserving_stage`) while delegating exactly. Unit coverage
mirrors each admission rule: rewrite and preflight reject a second predicate,
declared where semantics, and undeclared order loss; runtime placement keeps
uncomposed or repeated predicates on host; and the CUDA context regression
proves survivor ordering across lane reuse, an empty-survivor tile, the
unpublished dead intermediate, and mask-inclusive lane/probe/peak accounting.

Recorded process-gate output hashes are point-in-time evidence only:
`writers.las` embeds the UTC file-creation day-of-year, so candidate and
pinned-oracle bytes drift together across a UTC midnight while the
byte-identity assertion stays valid. `pdg_resident_pipeline_process` accepts
either `profile_not_exact` (no usable CUDA profile) or
`outside_calibration_envelope` (exact profile, unmeasured two-assignment
region) for its fail-closed control, because the reason is machine-dependent
while the decision is not.

D0062 makes the generic executor's host boundary row-backed. A
resident-selected execution runs against a dedicated row-backed
`pdal::PointTable` because the manager's `ColumnPointTable` rewrites
dimension offsets into column ordinals at finalize; `bindLayout` must fail
closed unless the declared offsets exactly partition the record, so an
ordinal-offset layout can never reach a boundary. The resident context CUDA
tests bind row-backed tables for the same reason. Resident stats report
aggregate `host_boundary_phase_seconds` (upload_pack, spill_wait,
spill_publish); these are diagnostic observability, never compared against
oracle output, and they are how the B0012 mixed-class deficit was attributed
to host packing before B0013 re-measured the same byte-identical lanes.
Resident stats also report `exact_host_repair` with elapsed seconds and
ambiguous, incomplete, and repaired row counts. B0050 adds
`exact_device_repair` with separate elapsed seconds and incomplete/repaired
counts; a row belongs to only one backend object. These measure exact repair
after a bounded device proof declines or cannot finish a row; they are
diagnostic only and do not weaken the full-output oracle comparison. Host
execution reports both fields as null. The resident LOF process gate bounds
every host count by the input point count and requires the repair closure to
contain its ambiguous and incomplete seeds.

## Bounded grid-stage qualification

D0221 withdraws D0083's SMRF device qualification: the compiled prototype's
eighth-neighbor cutoff-tie rule is known to differ from the oracle, so it fails
closed and its CUDA matrix is not an exactness gate. The exact KD2Index host
wrapper retains a deterministic process matrix and the `cell=1/2/4/8` ladder.
D0084's PMF lane uses a deliberately small global canvas rather than claiming
scalable grid coverage. Its deterministic process
matrices compare entire pipelines against pinned PDAL for defaults, explicit
cell/window/slope/threshold values, return subsets, custom output classes,
`only_ground`, missing and malformed return metadata, empty selections,
unsupported options, and host fallback. SMRF additionally covers scalar/net
cut values. Its committed sparse fixtures contain holes, a compact elevated
object, a low outlier, return combinations, and preexisting classifications,
but they did not expose the contested eighth-neighbor cutoff tie found on the
larger terrain fixture. That counterexample is required before any device
requalification. Every difference still reports the ordinary first differing
output byte and process diagnostics.

The primitive unit fixes the column-major 5x5 classification result and
`only_ground` preservation. Compiler/rewrite units require a whole-view Grid
descriptor, exact dimension reads/writes, no spatial index request, one
standalone upload/spill region, and explicit rejection of an unimplemented
grid bridge. The physical resident unit now traverses that preflight and
boundary setup only to prove required device execution fails closed before
publication. Run any future repaired grid-kernel or lifetime change under
memcheck with full leak checking, racecheck, initcheck, and synccheck before
requalification.

The compiled CUDA resource envelope still describes finite logical-double
XYZ, unsigned-byte Classification, at least two rows/columns, no more than
4,096 cells, and morphology radius no greater than 64. It is not an admission
envelope: `smrfSupportsExactDevice` returns false for every input under D0221.
Experimental work therefore uses the exact host wrapper, required execution
fails closed, and automatic placement remains disabled. B0021 and its Nsight
profile are historical implementation diagnostics, not an exact device
qualification or performance baseline.

PMF uses the same whole-view descriptor and standalone boundary contract, but
its exactness fixtures pin different upstream behavior: initial raster binning
uses `floor(coordinate - minimum) / cell_size`, final classification lookup
uses `floor((coordinate - minimum) / cell_size)`, voids use deterministic
one-nearest fill, and every height comparison is strict. The fractional-cell
fixture exists specifically to prevent reassociation of the two binning
expressions. CUDA admission requires finite logical-double XYZ, unsigned-byte
Classification, at most 4,096 cells, morphology radius at most 64, and no more
than 64 schedule passes. Equal numeric cell minima must retain the earliest
source bits, and every raster address is checked. Its 16-case host/nine-case
CUDA matrix, actual resident lifecycle, and all four Compute Sanitizer tools
are required after a PMF kernel or ownership change. B0022 is the negative
standalone performance baseline.

D0088 adds a separate required planner-resident tiled correctness lane; it
does not widen the automatically or experimentally selected envelope. Fix the
tile frame in column-major order, clip a one-cell halo at every edge/corner,
run each morphology phase over every tile before swapping global backings, and
mosaic owner cells only. Vary shapes and budgets in whole-grid stencil tests,
prove the exact one-byte cumulative pinned-host budget boundary, compare
fractional and nontrivial >4,096-cell fixtures directly with upstream, repeat
the deterministic lane five times, and prove product release at the matching
spill. A distinct-valued equal-distance void-fill tie must reject before
Classification changes because upstream nanoflann visit order is not encoded
by the grid metric. Default and broad experimental wrapper execution must
delegate before reduced-envelope validation or diagnostics; explicit required
and resident execution fail closed.

Run the ambiguity and both seam/oracle tests under memcheck with leak checking,
initcheck, racecheck, and synccheck. B0026 is the negative tiled-prototype
baseline: its 65x65 four-tile fixture is exact but only 0.026602x PDAL, and the
profile shows four all-points-per-tile raster launches dominating the run.

D0089's build-once contract requires named tests proving that raster-build
count is independent of tile count, an unconsumed generation cannot be
overwritten, equal numeric cell minima retain the first source's signed-zero
bits, identical-bit nearest ties are accepted, and distinct-bit equidistant
sources reject before publication. The wrapper lane must observe exactly one
`GridBuild` and one `RasterBuild`, allocate no storage beyond the two planned
host backings, and reach its matching spill even when a required execution
fails before CUDA publication. Re-run the enlarged-frame and nontrivial-seam
lanes under all four Compute Sanitizer tools after changes to the build,
generation, or resident-completion paths.

B0027 is the controlled positive dense follow-up: the same exact 65x65
four-tile fixture reaches 1.828354x PDAL with raster construction included.
The profile contains no per-tile all-point raster kernel and identifies serial
tile gather, copies, synchronization, and mosaic as the next boundary. The
dense fixture has no void cells, while the allocation-free host nearest fill
remains proportional to void cells times populated cells. Do not call PMF
scalable or the Grid product composable until an exact accelerated sparse/void
fill is bounded and tested, phase storage remains on device, and an exact
cross-stage reuse benchmark is accepted.

D0090's device-phase contract adds a one-byte exact budget boundary for the
complete two-backing frame. A fitting product must allocate nothing before its
sole raster generation is published, reject double publication even after
consumption, reject consumption without publication, and refuse
materialization after consumption. The ambiguous nearest-tie lane must prove
that no `RasterBuild` or `RasterUpload` occurs. Larger frames must still take
D0088's exact host-tiled path; the new path is conditional full-frame
residency, not a universal replacement.

For a fitting frame, record exactly one full-raster H2D and zero raster D2H
transfers, compare sparse and dense fixtures with pinned PDAL, retain the
fractional/return/`only_ground` and nontrivial-seam matrices, and repeat the
fractional lane five times. The allocation-free sparse source list must scan
populated cells in column-major order, fall back to the full column-major scan
when scratch is insufficient, and never change first-bit minimum or
equal-distance tie semantics. Run full-device, host-tiled, and ambiguous-
rejection lanes under memcheck, initcheck, racecheck, and synccheck after any
phase-backing or generation-lifetime change.

B0028 is a controlled positive follow-up: the exact dense 4,225-point 65x65
fixture reaches 9.009503x PDAL and the exact 25-point sparse/void fixture
reaches 4.764988x, each with one 33,800-byte raster upload and zero raster
downloads. Its Nsight Compute report shows 20 launches per candidate and no
tiled phase kernel. The tiny fixtures, excluded product setup, host source
build, remaining worst-case quadratic nearest fill, and absent cross-stage
consumer prohibit an automatic model, a general scalability claim, or a
composable Grid claim.

D0091 replaces the unbounded populated-source scan above 255 sources with an
exact byte occupancy hierarchy stored in the already planned second host
backing. Tests must compare a heterogeneous, greater-than-255-source raster to
the literal all-source result by raw binary64 bits, including large-origin and
fractional-cell arithmetic. They must also prove that nonfinite represented
centers reject admission, identical-bit multiway ties succeed, distinct-bit
multiway ties fail before publication, and the same unpublished product can be
retried successfully after the input ambiguity is removed. Work counters must
include both visited source leaves and all visited hierarchy nodes; leaf counts
alone are not scalability evidence.

The literal/hierarchy selector is a performance contract, not an exactness
branch: record both sides of every threshold change with product setup inside
candidate wall time. B0029 fixes the selector at 255 after a 64/65-source
audit exposed a large cliff and the 255/256 pair measured 31.239504 versus
28.410521 ms of raster construction. Its heterogeneous 257x257 and 513x513
rows read candidate Classification back from CUDA; XYZ are immutable staging
columns and return fields are fixture invariants in this direct primitive
harness. Raw binary64 raster parity belongs to the independent literal-scan
unit. The rows record empirical near-linear hierarchy work for that named
distribution. The branch-and-bound walk has no proved worst-case subquadratic
bound, so documentation must not generalize B0029 to arbitrary sparse layouts.
Re-run the fitting, host-tiled, and ambiguity paths under memcheck, initcheck,
racecheck, and synccheck after changes to the hierarchy, backing lifetime,
selector, or finite-center gate.

D0092 adds a distinct device-proof budget to the planner contract. For PMF it
must equal the complete two-phase-backing width: the product rejects a
mismatched width, allocates nothing one byte below the full-frame boundary, and
retains the exact D0091 host-build/tiled path. At the exact boundary, test that
the allocation remains provisional and is not visible as phase storage, host
publication and early phase materialization fail, duplicate proof
materialization fails, and successful publication promotes the same pointer
without a second allocation. A rejected proof must discard the allocation,
publish no generation, and permit a successful retry on the same product.

Device raster exactness requires an independent raw-binary64 completed-raster
comparison on a greater-than-255-source, large-origin, fractional-cell frame.
The matrix must include same-cell signed-zero minima, identical-bit
equal-distance ties, and a distinct-bit tie that fails before Classification
device materialization/H2D or any mutation. The older bounded direct CUDA path
must use the same tie-proof semantics. Both wrappers must catch only that
named ambiguity, execute pinned upstream on a private copy of the untouched
view, and preserve Classification plus warning bytes. A fitting resident execution records
`deviceNativeRaster`, `deviceNativeSourceBuild`, and `usedDeviceTieProof`, one
`RasterBuild`, zero `RasterUpload`/`RasterDownload` events, and zero raster
H2D/D2H bytes. The one-byte-below wrapper case must remain byte-exact through
the host fallback and use more than one morphology tile.

B0030 includes product setup, XYZ and Classification H2D, device source
construction/proof/promotion, all device phases, and Classification D2H in
candidate wall time. Its dense 65x65/513x513 and sparse
65x65/257x257/513x513 rows are Classification-exact at 3.156x-38.950x pinned
PDAL with zero raster transfers. This is a controlled device-source gate, not
an automatic-placement or scalability gate: the proof scans every compact
source per target. The sparse 257x257 Nsight Compute report must remain the
limiter record; it attributes about 1.25 ms and 50.3% SM throughput to the
exact FP64 proof reduction rather than transfers. Re-run the raw-bit, tie
failure/retry, bounded-direct, and one-byte-fallback lanes under memcheck,
initcheck, racecheck, and synccheck after any proof kernel, workspace, promotion,
or allocation-order change.

D0093 supersedes the one-generation PMF product lifetime only for a contiguous
PMF chain with the same Grid/cell contract and exact original JSON `returns`
value. Rewrite tests must prove first/reuse/final hidden markers, preserve the
standalone shape, and reject absent-versus-explicit return forms, unequal
returns, unequal cell frames, and every non-PMF Grid consumer. The runtime must
create the product only for the first marker, require it for reuse markers,
revalidate the exact frame, keep the delegated region active through
intermediate stages, and reach the matching spill after the final stage or an
exception.

The product contract is sequential generation reuse, not content reuse. A new
host build or device proof is legal only after the previous generation is
consumed; it resets phase orientation and native-build state while preserving a
successful phase allocation. Tests must prove cumulative build/consume counts,
the pending-generation overwrite guard, identical device backing pointers
across two promotions, exactly one underlying allocation, and discard/retry
behavior for a failed provisional proof. Every PMF stage must still rebuild and
consume its own exact raster because morphology overwrites both backings.

The physical D0093 differential executes a real prepared two-stage PMF chain
and must match pinned PDAL exactly while observing one `GridBuild`, two
`RasterBuild` events, zero raster H2D/D2H, and final region completion. Run that
pair under memcheck, initcheck, racecheck, and synccheck after changes to
generation state, reuse markers, backing allocation, or region completion.
B0031 v4 times the actual resident scope/preflight, upload, prepared wrapper
chain, spill, return selection, both exact device raster generations/phases,
and a complete six-column readback against the corresponding prepared PDAL
chain and symmetrically includes each worker's RAII teardown. Exact comparison
runs outside both timers. Its 65x65 pair measures 1.039838x pinned PDAL with one
observed product; one underlying allocation is proved separately by the
product unit. The
earlier 4.590078x direct-primitive comparison is rejected. Three-stage tests
cover a true nonterminal reuse stage, an all-fallback tie chain that stays open
through a nonterminal reuse marker and closes at the final fallback, and
exception completion followed by final rejection and spill. A successful
post-tie generation is outside this fixture: unchanged geometry and the exact
source contract reproduce the same tie at each stage.
This is a PMF-only allocation-reuse gate; it supplies no semantic
raster/surface reuse, cross-kind Grid/cloth, automatic-placement, or general
scalability evidence.

CSF uses the same whole-view contract with a bounded cloth-simulation envelope:
`smooth=false`, finite logical-double XYZ, unsigned-byte Classification, at
least a 2x2 and at most 4,096-cell cloth, `0 <= iterations <= 64`,
`rigidness >= 0`, and supported returns/classes/`only_ground`. Native admission
also requires the pinned serial, non-OpenMP oracle capability. Runtime rejection
applies to `dir`, `debug`, `ignore`, `where`, smoothing, an OpenMP-enabled
build, invalid options, and nonfinite/oversized cloths. A 22-case host matrix
and 14-case CUDA matrix are required, plus eight focused host tests and 12
focused physical CUDA tests. Baseline controls include verbose diagnostics and
an empty lifecycle case; the CUDA matrix is repeated five times for
determinism.

Benchmark B0023 on the 1M reference fixture (`SHA-256 e86d960c...`) reports
PDAL 0.303500380 s versus CUDA 1.049116974 s (ratio 0.289291x) with exact
output `a3c13a...`; the same-pipeline ncu profile reports 218.69 ms frame time,
361.55 ms raster, 32.96 ms simulation, and 58.88 µs classification. The
classifier reaches 79.48% SM throughput, 21.81% DRAM throughput, and 84.28%
occupancy. The lane remains host-selected in production placement and does not
claim P3 exit.

ELM uses the same standalone whole-view boundary but no reusable Grid product
or spatial index. Admission requires finite logical-double XYZ, unsigned-byte
Classification, finite positive `cell`, finite threshold, a valid output
class, and at most 4,096 runtime cells. Its exact sort is stable by Z and then
stable by column-major cell key; signed zeros are canonicalized only in the
sort key so they remain one upstream-equivalent class while source order and
stored bits are preserved. The strict threshold walk is serial within each
cell. Empty delegated regions close without allocation, and required CUDA
fails before Classification publication for nonfinite or oversized input.

Benchmark B0024 on the same 1M fixture reports pinned PDAL 0.481539710 s versus
forced CUDA 0.555778589 s (ratio 0.866424x), with exact output `69f253...`.
The same-pipeline Nsight report records 470.78 microseconds total kernel time,
including 254.51 microseconds in radix sorting. The complete-process deficit
is therefore outside kernel throughput and no automatic model is accepted.

Skewness balancing is a narrower global-ordering hybrid rather than a Grid
resident stage. Device admission requires physical binary64, finite, pairwise
comparator-unique logical Z and no more than `INT_MAX` points. The adjacent-
equivalence flag is a proof gate: any equality, including `-0.0`/`+0.0`,
rejects CUDA before the PointView changes. After CUDA returns the permutation,
all fields are reordered together and the original sequential online moments,
sign-crossing walk, and Classification publication execute on the host.

Benchmark B0025 uses a deterministic 1M unique-ramp fixture (SHA-256
`4851c11f...`) and reports pinned PDAL 0.990846171 s versus forced CUDA
0.861619405 s (ratio 1.149981x), with exact output `bdfaabcd...`. The
same-pipeline Nsight report records about 255.07 microseconds total kernel
time, including 202.11 microseconds in eight radix-sort passes. This is a
positive forced gate for the controlled exact envelope; ordinary tied-Z data,
the host recurrence, full-record publication, and the missing resident product
preclude an automatic model.

D0094 adds an exact count-one `filters.hag_nn` lane on the planner-owned
two-dimensional kNN index. The process matrix must cover the default and both
forced backends, empty/all-ground/one-ground/custom-class/same-XY shapes,
count-one options ignored by the pinned single-neighbor branch, equal-distance
tie repair on both backends, sparse-grid incompleteness, no-ground diagnostics,
nonfinite-XY fallback, unsupported counts/options, and preparation errors. The
host matrix has 20 cases; the physical CUDA subset has 11 and is repeated five
times. Tie and incomplete rows use explicit proof switches so a silent native
success cannot pass as fallback coverage.

Resident tests must execute both HAG→3D-neighborhood and reverse order, compare
both output columns exactly, and observe two physical index builds. They must
also cover a HAG tie and no-ground host repair before the 3D consumer, plus
HAG-column reuse through an adjacent point-program bridge for unique, tie,
no-ground, nonfinite, and empty inputs. Any 2D/3D dimension-marker change must
rerun the established neighborhood host matrices because their wrappers retain
the default 3D hidden-option surface.

B0032 is a positive forced gate only. Its controlled 1,000,002-point fixture
has a unique nearest ground point for every non-ground query and measures the
exact complete pipeline at 1.512959x pinned PDAL. The same-pipeline profile
identifies the masked nearest-neighbor gather as the limiter (1.64 ms, 76.00%
SM throughput, 34.10% memory throughput, 47.74% achieved occupancy). The HAG
projection is 15.94 microseconds and memory-bound. Data-dependent ties and a
failed required-CUDA trial on ordinary corpus data preclude an automatic model;
the broader `hag_dem`/`hag_delaunay` family remains open.

D0095 extends the same exact HAG contract to `count=2`. The process matrix has
32 host cases and a 24-case physical CUDA subset. In addition to D0094's
count-one coverage, it exercises ordered two-ground interpolation,
`allow_extrapolation=false`, strict `max_distance` equality, inclusive BOX2D
edges, same-XY selection, negative and excluding distance values, internal and
second/third-candidate boundary ties on both index backends, a one-shell
incomplete grid search, nonfinite Z, one ground reference, and `count=3` host
selection. Explicit proof switches make the insufficient-ground, nonfinite-Z,
tie, and incomplete repair cases fail if they silently take another path.

Resident tests execute both count-one and count-two terminal regions. Count
two must also publish through an adjacent assignment bridge after both a native
result and the pinned one-ground host repair; the bridge compares the retained
HAG column exactly and must not treat the stage as an XYZ mutation. Preflight
accounts for two ids and two squared distances per point, or 27 bytes per row
including the source, ground, and status bytes.

Cross-product residency is tested separately from same-product reuse. The
planner test for PMF -> HAG2 -> PMF requires three distinct regions and all six
upload/spill boundaries, including their dimensions, byte costs, and release
liveness. Rewrite coverage fixes the exact 11-entry marker/stage sequence. A
physical upstream-versus-resident differential executes every boundary and
requires three region begins/ends, two RasterGrid builds, one masked 2D index
build, and exact Classification/HAG output. Grid and non-Grid sibling DAGs
retain same-kind grouping but must fail closed when the selected boundary has
multiple consumers. SMRF, PMF, CSF, and ELM -> point-program rewrite tests
also require two regions and an explicit spill/upload pair; this is a point
batch transfer contract, not permission to reuse a semantic Grid product.

B0033 is again a positive forced, data-dependent gate. Its deterministic
1,000,002-point count-two fixture measures the exact complete pipeline at
1.493330x pinned PDAL. The matching 18-launch profile identifies the masked
two-neighbor gather as the limiter at 2.15 ms, 74.55% compute throughput and
48.30% achieved occupancy; the 49.15-microsecond HAG projection is
memory-bound. No automatic model, `count >= 3`, or broader HAG-family claim
follows.

D0096 adds a separately bounded `filters.hag_delaunay,count=3` contract. The
host matrix has 22 cases and the physical CUDA subset has 15. It covers
explicit count-three selection, default/count-four host ownership, empty and
all-ground views, custom ground class, same XY, inside and outside local
triangles, disabled extrapolation outside global bounds, collinear and
duplicate points, selected/unselected boundary ties, insufficient/no ground
including the exact error diagnostic, nonfinite Z, one-shell grid
incompleteness, both forced shared-index backends, unsupported options, and
preparation errors. Tie, insufficient-ground,
nonfinite, and incomplete rows have positive proof switches; five consecutive
physical CUDA matrix runs must remain byte-exact.

The direct wrapper comparison additionally fixes the bounded arithmetic
surface against the pinned Delaunator and barycentric code: seed reordering,
the `1e-14` outside-edge threshold and rejection beyond it, signed-zero
coordinates, large finite XY products, positive- and negative-infinity
interpolation produced from finite Z inputs, and a point on the global bounds
but outside its selected local triangle. Positive infinity is the upstream
outside sentinel; negative infinity is a valid interpolation and must not be
collapsed into nearest-ground fallback. Subnormal distance collapse must take
the proved tie-repair path.

Resident tests execute a terminal native region and a nonterminal assignment
bridge after native, insufficient-ground, tied-fourth-candidate, and incomplete
grid results. Every repaired case must publish the pinned host HAG column back
into the retained 2D product, make the bridge reuse proof fire, and close the
delegated lifecycle. Preflight accounts for three ids and three squared
distances plus the source, ground, and status bytes, or 39 bytes per row.

B0034 is a positive forced, data-dependent gate. Its deterministic
1,000,002-point fixture measures the exact complete pipeline at 1.314103x
pinned PDAL. The 18-launch profile identifies masked `knnGatherKernel` as the
limiter at 1.96 ms, 67.82% compute throughput, 52.33% memory throughput, and
45.62% achieved occupancy. The 111.46-microsecond Delaunay projection is
register-limited. The 433-registration Host ASan/UBSan and 653-registration
physical CUDA aggregates pass with their documented skips, and all four
Compute Sanitizer tools are clean across the nine-test focused lane. No
automatic model, HAG Delaunay `count >= 4`, HAG-NN `count >= 4`,
`filters.hag_dem`, or complete HAG-family claim follows.

B0237/D0236 supersedes only B0034's no-automatic-model conclusion for the
strict count-three direct composition and corrects B0236's admitted minimum.
Automatic process coverage uses hash-pinned 1,000,002- and 16,000,002-point
sources plus deterministic 400,002-, 450,000-, 500,000-, and 500,001-point
prefixes. It must positively select the 500,001 floor, 1,000,002 reference
row, and cap; reject 400,002, 450,000, 500,000, and cap-plus-one; and freeze
the 184-byte/point memory boundary at exactly 184*N versus 184*N - 1.

The automatic grammar matrix must reject missing or wider counts, explicit
class or extrapolation options even when semantically equal to the default,
writer-option drift, array roots, uppercase endpoints, missing source Extra
Bytes, same-stride semantic descriptor drift, compression/layout drift,
disabled mapping, preflight injection, execution-proof injection, and injected
device decline after selection. Required refusals exit 124 without output.
Existing, aliased, and symlink destinations must compare status, stdout,
stderr, and final filesystem state with the pinned oracle.

Data-dependent automatic coverage uses a single isolated ambiguous query in
an otherwise calibrated 500,001-point source for tie repair and another
isolated query for forced incomplete-grid repair; do not construct an all-query
repair stress case in the correctness lane. The no-ground case compares the public
diagnostic and the pinned 2.10.0 SIGSEGV status directly and does not impose a
required-route sentinel over oracle behavior. Successful rows must prove one
selected Delaunay stage/region, one planner-owned index build, mapped-source
record-summary use, no complete host XYZ mirror, direct binary64 publication,
matching output cardinality, exactly one active lane, calibrated placement/
executor agreement, and observed successful CUDA execution before commitment.

Performance acceptance requires the separate 50,001--16,000,002 explicit
ladder, a conservative composition model, and final ordinary public commands
with one warmup and nine alternating pairs. A fresh B0237 review run measures
450K at only 1.093356x, below the predeclared 10% margin, so the selector
starts at 500,001. Final public medians are 1.245213x at 500,001 and 2.049354x
at 1,000,002, with both paired lower bounds above the margin. A same-final-
binary proof-bearing stats run and a separate final automatic Nsight profile
are required. The placement audit and raw report verifier remain release gates
for this envelope; the current results are 214/218 and 224/224.

B0238/D0237 retains the exact LOF closure algorithm but parallelizes large
repair lists over the one compatibility KD3 index and optionally materializes
that index's immutable binary64 XYZ backing. Correctness coverage must compare
cached and uncached kNN ids and squared-distance bits, compare both LOF repair
states to pinned upstream bits, prove that a fixed 16,384-point closure crosses
the parallel/cache threshold, and prove that the compact small closure remains
serial. The shared-index concurrency unit runs fixed queries through four
threads and must pass TSan; ASan/UBSan and the full host unit aggregate remain
required. Worker-count perturbation may not change output bits.

The public performance proof uses B0043's exact 4M
`LOF(minpts=10) -> UserData assign -> default LAS` endpoint, one warmup, and
nine alternating pinned-PDAL/candidate pairs. Engine-owned
`PDG_REQUIRE_LOF_PARALLEL_REPAIR` and
`PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE` assert the implementation but may not
alter it; a separate option-free public run proves natural selection. The
benchmark runner must scrub both assertions, the cache-disable control, and
`PDG_NATIVE_WORKERS`, plus the kNN shell/prefilter/backend and neighborhood
row-boundary tuning controls,
from the ambient environment and inject proof variables only into candidate
processes. Terminal public guards must reject either assertion when automatic
selection declines. A same-final-binary disable/require comparison,
repair-phase telemetry, exact artifact hash, and peak-RSS delta are acceptance
evidence. Current results are 1.840449x cached versus uncached, 0.504710 seconds
of natural cached exact repair, 18.300959x pinned PDAL in the nine-pair proof,
18.550858x option-free, and a 103,224-KiB observed RSS increase for the 96 MB
payload. Cached build failure must leave PointView without a published index,
retry uncached for ordinary execution, and fail when the cache proof is set.
This changes neither the B0043 selector nor GPU-native coverage. Because no
CUDA translation unit changes, B0238 adds physical SM89 LOF exactness but no
new Compute Sanitizer claim.

B0239/D0238 qualifies only the fingerprinted 1M r2 endpoint. Its process
matrix must prove the complete public `LAZ -> option-free SMRF -> option-free
HAG-NN -> named HeightAboveGround LAZ` command against pinned PDAL with exact
artifact bytes, metadata, point order, stdout, stderr, and exit status. The
positive row requires both automatic-selection and selective-repair proofs.
Wrong point count, measured header/extent, full-file fingerprint, endpoint
case, grammar, layout, device profile, rewrite, or execution must refuse before
output and retain the direct pinned-oracle path when proofs are absent. The
engine must independently reject an externally injected internal marker on a
neighboring grammar/layout. A test-only post-selection CUDA decline with both
proofs must exit nonzero without an artifact; full host recomputation is not a
valid proof success.

The broader HAG-NN matrix remains the arithmetic authority for equal-distance
ties, incomplete searches, empty/ground-domain cases, unsupported options, and
diagnostics. A selected r2 run repairs only the compact rejected ids through
one planner-owned compatibility ground view/KD2 product, publishes each
repaired binary64 value into the retained resident column, and may reach the
named writer only
after every repair succeeds. Unit coverage must first force a rejected exact
GPU result, then prove a deterministic equal-distance repair against upstream
bits. Phase telemetry may attribute scan, partition, index, submit, wait,
status, and repair costs but may not change behavior.

Performance acceptance is one warmup and nine alternating complete public
pairs on the same machine and frozen epoch, plus a separate oracle/candidate
complete-process profile. Record the exact rejected full-host-recompute probe
and any failed selector attempt as negative evidence. The retained B0239 row
is 1.270989x median with 9/9 wins; the final profile moves the limiter to LAZ
decode and remaining host/I/O/startup work. Focused physical CUDA and host
differentials, leak-disabled ASan/UBSan, and Compute Sanitizer memcheck and
racecheck are required because the changed resident HAG path executes CUDA.
The reference harness must scrub ambient `PDG_*` controls, reject public
`PDG_INTERNAL_*` injection, apply proof overrides to the candidate only, and
persist the effective override map in the raw report.

B0243 adds a launcher-wide environment invariant for all thin direct routes.
The process test must run an otherwise eligible direct r1 graph with both a
current engine-only proof (`PDG_REQUIRE_AUTOMATIC_SKEWNESS_RESIDENT`) and an
unlisted future `PDG_*` name and observe the engine wrapper. Unit coverage must
prove the prefix rule, the sole `PDG_ORACLE_PDAL` exception, and an unrelated
non-PDG environment name. The launcher snapshots the real startup environment;
tests may not validate only a maintained list of known names.

The same matrix freezes r8 as a negative selection after the exact direct
prototype failed its corrected-final performance gate. The literal measured
five-stage graph and matching 1M format-7/36-byte header must still route to
the engine. B0243's 21-pair direct-versus-forced-engine report is required
negative evidence; because its interval spans parity, it cannot support an
automatic route even though artifacts and process observables match.

B0244/B0245 freezes r13 as a negative selection. The attempted literal matcher,
full-file calibration fingerprint, proof variable, and one-reader-worker
override were removed after the corrected-final 41-pair cold interval spanned
parity. The 21-pair warm and 41-pair file-evicted cold reports remain required
negative evidence; a future attempt must reintroduce its bounded positive,
modifier, content-drift, malformed-input, configured-child, and exact
artifact/metadata/order/stream/status process matrix before selection. Until
then, the checked-in reference-suite differential and merge/fallback matrices
prove the unchanged engine/PDAL host path. Warm-only resolution must never be
used to qualify a cache-agnostic route.

B0246 freezes r12 as a negative selection. The literal fixed-origin splitter
graph must classify to the engine at both the dispatcher unit boundary and the
real launcher process boundary. The 21-pair warm and 41-pair cold
direct-sibling, integrated-public, and same-final forced-engine reports remain
negative evidence: complete artifact membership/names/bytes, LAS metadata and
point order, stdout, stderr, status, and determinism are exact, but no route
clears both pinned-oracle and same-final performance gates. The exact
1/2/4/6/8 worker sweep likewise permits no reader override. Future r12 work
must start with a different profiled algorithmic or codec/publication
hypothesis and repeat warm/cold exact gates before selection.

The reference runner must reject a writer that aliases a registered input
fixture and must reject duplicate materialized output artifact patterns before
starting either executable. It may not silently deduplicate a collision: doing
so would make an incomplete multi-output publication appear exact.

This is not the P3 mosaic gate. Before SMRF, PMF, ELM, or another grid stage is called
scalable/resident, add named core/halo tiles that hit every edge and corner,
compare each tile alone and the owner-only mosaic against the global oracle,
vary tile size and VRAM budget, and compare raster metadata plus physical block
layout where output rasters are involved. A planner-owned reusable grid
workspace and an accepted composition benchmark are required before a grid
bridge may remain open.

## NVIDIA architecture qualification

Release configuration generates real code for every architecture supported by
the selected CUDA compiler and PTX for its newest target. That proves only that
the compiler accepted the source. Exact support for an SM requires a physical
device run of a compact, fixed-bit differential lane covering:

- LAS field translation, ordered assign/ferry/ordinal conversion, signed zero,
  subnormal, NaN payload, half-integer, and cast-limit boundaries;
- point maps, stable radix ordering, selection, histogram, summary, and robust
  reductions around warp, block, CUB, and chunk boundaries;
- spatial cell/radius boundaries, equal-distance ties, near-repeated
  eigensystems, resident publication, and deterministic tie repair; and
- one integer-only translation/compaction control.

Run the CUDA-12.x lane on each legacy-compatible real target (currently SM
50/52/53, 60/61/62, 70/72, 75, 80/86/87, 89, and 90) and the CUDA-13+ lane on
each current target advertised by that compiler. Exercise embedded PTX with
`CUDA_FORCE_PTX_JIT=1` on compatible physical devices. Record toolkit, CCCL,
driver, device, cubin-versus-PTX path, output hashes, and first differing bits.
The full corpus, performance suite, and sanitizers may use representative
machines, but this compact bit lane may not be skipped for an advertised exact
architecture. SM 89 is the only physically qualified architecture at the
current checkpoint.

All product CUDA translation units pin `--fmad=false`, `--ftz=false`,
`--prec-div=true`, and `--prec-sqrt=true`. These flags remove known compiler
contraction/denormal variability but do not replace the physical matrix:
libdevice, Eigen device code, and CCCL policies remain toolkit- and
architecture-sensitive exactness surfaces.
