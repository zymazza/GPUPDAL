# GPUPAL implementation plan

The public product and repository name are GPUPAL (GPU Pointcloud Abstraction
Library), and the PDAL-compatible command is `gpupal`. Historical evidence and
internal `pdg` identifiers retain their recorded spelling; see D0288.

This plan refines the phase queue in `spec.md` into reviewable vertical slices.
The default acceptance contract is exact compatibility with the pinned
upstream PDAL oracle; see `docs/testing-strategy.md` and decision D0002.

## Product completion contract (revised by D0208)

The goal is speed on the pipelines people actually run. CUDA is one means to
that end, not the end itself. This contract replaces the catalog-wide
GPU-coverage obligation that D0019 originally set; D0019's exactness and
honesty requirements are unchanged and still binding.

What the product must deliver:

- **Exact compatibility, catalog-wide.** Every application, driver, option,
  graph shape, and configured plugin in an equivalently built upstream PDAL
  stays functional, and default output stays byte-identical to the pinned
  oracle. This is unchanged and non-negotiable.
- **Measured speed on the reference pipelines.** The release criterion is
  wall-clock time on the end-to-end workflows in `bench/pipelines/reference/`
  versus the pinned oracle on identical hardware. A stage matters to the extent
  it appears in those pipelines and costs time there.
- **Speed by any means.** Host threading and algorithms, parallel and streaming
  I/O, decode and encode, removed stage boundaries, device residency, and CUDA
  kernels are all equally legitimate. A host-side win is a win. There is no
  requirement that an optimization involve the GPU, and no credit for one that
  does but does not help.
- **Acceleration only where it is faster.** A CUDA path is built or kept only
  where it is measured faster end-to-end on a reference pipeline or a real
  workload. An exact-but-slower CUDA lane remains in-tree, host-selected, and
  labeled catalog coverage; it is never presented as an acceleration and never
  automatically selected. Existing exact negative lanes — PMF 0.583x, CSF
  0.289x, ELM 0.866x, transformation, IQR, MAD, colorinterp — are retained in
  that state rather than removed, since they may win on other hardware or
  frame sizes. SMRF is different: D0221 withdrew its device qualification
  after proving a byte-inexact void-fill tie rule, so only the exact host
  wrapper/fallback remains selectable.
- **Honest separate reporting.** Functionally supported, GPU-native,
  performance-qualified, and automatically selected remain four independent
  labels, and `docs/stage-coverage.md` additionally marks whether an entry is
  on the measured fast path or exists for catalog coverage (D0204).

What is explicitly **no longer** required:

- catalog-wide native CUDA coverage as a release criterion;
- "zero temporary upstream fallbacks at the full-product milestone" — a host
  path that is exact and fast is a finished answer, not scaffolding;
- a CUDA port for any stage merely because it is data-parallel.

The remaining native count is therefore no longer tracked as a completion
metric. Reference-pipeline wall clock is.

## Reference pipelines (D0208, expanded by D0239)

These are the speed target. They are end-to-end workflows, not single stages,
and each is measured as a complete process against the pinned oracle. The
existing 53 single-stage bench pipelines remain useful for attribution but are
not the goal.

| Id | Workflow | Why it is representative |
| --- | --- | --- |
| `r1-translate` | LAZ -> reproject -> crop -> LAZ | The most common PDAL operation of all; dominated by decode/encode and a cheap transform |
| `r2-ground-normalize` | LAZ -> SMRF -> HAG -> LAZ | The standard lidar normalization workflow; SMRF's CUDA prototype is unqualified and disabled, so this is a host-optimization target |
| `r3-dtm` | LAZ -> SMRF -> ground filter -> `writers.gdal` DTM | DEM/DSM production; exercises raster assembly, which has no acceleration today |
| `r4-denoise-thin` | LAZ -> outlier -> sample -> LAZ | Cleanup and reduction; exercises an already-fast neighborhood path plus I/O |
| `r5-copc-query` | COPC bounds/resolution query -> stats | Modern cloud-native access; exercises range reads and pushdown |
| `r6-features` | LAZ -> normal + eigen/covariance features -> LAZ extra dims | ML feature extraction; the shape where the shared kNN gather already pays |
| `r7-dsm` | first/only returns -> maximum-Z `writers.gdal` DSM | Explicit production surface policy plus raster assembly/publication |
| `r8-colorize` | EPSG:28992 points -> EPSG:3857 RGB sample -> EPSG:28992 LAZ | Orthophoto sampling, CRS interaction, out-of-bounds retention, and GDAL cache behavior |
| `r9-polygon-clip` | EPSG:4326 multipolygon/hole -> reprojected crop -> LAZ | Standalone AOI clipping with holes, boundaries, and geometry reprojection |
| `r10-decimate` | voxel-centroid nearest original point -> LAZ | Spatially uniform reduction while preserving complete selected records |
| `r11-classify-refine` | SMRF -> statistical outlier -> `k=7` neighbor vote -> LAZ | Realistic built-in classification/refinement without an external model |
| `r12-tile` | fixed-origin splitter -> deterministic LAZ set | Tile ownership, naming/order, fan-out, and multi-output publication |
| `r13-merge` | heterogeneous format/layout/header inputs -> LAZ | Multi-file mosaic assembly, order, header synthesis, and compression |
| `r14-convert-compress` | LAS -> LAZ, with zero-weight direction variants | Format-only throughput, compression ratio, peak RAM, and COPC support |

Together they cover I/O, terrain, neighborhood, raster, geometry, reduction,
classification, partitioning, assembly, and format-only work. Each headline
has equal aggregate weight. Supporting r10/r14 variants have weight zero, so
their breadth cannot overweight those workload families.

## Active speed-first work order (D0099, D0204, D0208)

New stage ports are paused after D0098. Catalog-wide native coverage remains
the final scope, but stage count no longer chooses the next slice. Every
candidate must follow this loop:

1. Measure the equivalent stock pinned-PDAL stage or complete pipeline on the
   reference machine.
2. Profile the complete process and name its dominant cost before proposing a
   fix.
3. Choose among a fused point-operation kernel, resident stage using shared
   data/index products, standalone CUDA stage, optimized host bridge, or an
   evidence-backed deferral.
4. Use a cheap prefix/controlled prototype to reject a weak direction before a
   certified lane or calibration ladder is spent on it.
5. Proceed only when the path beats CPU standalone, removes a significant
   transfer, shares expensive GPU work, or supplies required catalog coverage
   that remains force-only while slower.
6. Prove pinned-oracle byte, metadata, order, diagnostic, and status exactness.
7. Measure the complete reader/transfer/stage/writer pipeline.
8. Publish a performance claim only with its same-machine baseline, profile,
   and append-only `BENCHMARKS.md` record.

The measured priority order is (rewritten by D0204 after B0142; the previous
eight-item order is superseded but its evidence is retained in the entries it
cites):

**What the record actually shows.** D0204 re-read the benchmark corpus rather
than assuming, and the split is not "kernels versus everything else". It is
*shared* kernels versus *standalone per-stage* kernels:

- Shared kernel work produced the largest measured wins in this repository.
  D0074 took the 22M kNN gather from 32.2 seconds to 0.49 seconds, and the same
  shared `knnGatherKernel` is still 137.767712 milliseconds and 97.3487% of all
  kernel time on the qualified 4M nndistance endpoint — about 22% of that
  endpoint's 0.623028204-second wall. B0064's parallel repair kernel took a
  further 9.242103% off the same endpoint. Shared kernels are not noise and
  must not be deprioritized.
- Standalone per-stage kernels have never paid. PMF 0.583x, CSF 0.289x, ELM
  0.866x, transformation 0.357x--1.008x, IQR 0.951x, MAD 0.949x, and
  colorinterp 0.345x--0.873x are measured slower than stock and correctly
  host-selected. SMRF's historical 0.599x row no longer qualifies anything:
  D0221 disables its known-inexact device fill.
- Bounded epilogue kernels are genuinely negligible. On the qualified 1M HAG-NN
  endpoints all kernels are 1.277263%--1.807004% of wall and the ordered HAG
  projection alone is 0.033829%.
- Boundary and residency removal paid repeatedly and cheaply: B0052 -35.541759%,
  B0111 -34.483%, B0060 -8.02744%, B0062 -5.693104%.

Fixed process-level CUDA startup is about 0.176 seconds and roughly constant,
so it is ~50% of a 0.355-second HAG endpoint, ~28% of the 0.623-second 4M
nndistance endpoint, and negligible at 22M. It is a small-endpoint artifact,
not a general limiter, and B0054 and B0108 already closed it by measurement.
Do not reopen it without new evidence.

**What actually blocks each reference workflow (B0146--B0156, measured).**
Every workflow was run through `pdg resident --stats` and its refusal read
directly rather than inferred:

| Workflow | Executor | Refusal | Cause |
| --- | --- | --- | --- |
| `r1-translate` | `pdal_standard_host` | `non_cardinality_preserving_stage` | `filters.crop` changes point count |
| `r2-ground-normalize` | `pdal_standard_host` | `invalid_runtime_facts` | writer `extra_dims` |
| `r3-dtm` | `pdal_standard_host` | `invalid_runtime_facts` | sink is `writers.gdal`, not `writers.las` |
| `r4-denoise-thin` | `pdal_standard_host` | `non_cardinality_preserving_stage` | `filters.range` and `filters.sample` change point count |
| `r5-copc-query` | `pdal_standard_host` | `invalid_runtime_facts` | source is `readers.copc` |
| `r6-features` | `pdal_standard_host` | `invalid_runtime_facts` | writer `extra_dims`, plus `mixed_calibration_models` behind it |

Six workflows, two root causes. Four fail on runtime facts and two on
cardinality. Nothing here is a kernel problem, a placement-model problem, or a
correctness problem — every one of these pipelines produces byte-exact output
today, just slowly.

The cardinality result was not on any earlier queue and is the single widest
gap: `crop`, `range`, `sample`, and `decimation` are among the most-used filters
in PDAL, and any pipeline containing one is currently unplaceable in full. The
planner already admits *declared predicates* that change cardinality; these
stages simply are not expressed that way.

Work in this order. The ordering is by measured blocker coverage, not by
category:

1. **Admit the sinks and sources real pipelines use — but B0157 showed this is
   partly a calibration question, not purely an admission one.** Admitting
   writer `extra_dims` to placement facts works mechanically and keeps output
   exact: `r2` and `r6` move from `invalid_runtime_facts` to genuine placement
   decisions. It was reverted because the placement calibration predicts
   device-versus-host timing from record sizes measured on the canonical
   layout, and a wider output record falls outside that envelope — an existing
   process gate asserts exactly this. So `extra_dims` needs a calibration
   measured over the wider record, or a placement decision robust to output
   width, and belongs with item 4 rather than here. `readers.copc` for `r5` and
   `writers.gdal` for `r3` are untested against the same objection and may face
   it too; check before assuming they are pure admission work. LAZ is already done (B0150--B0152), and the pattern it
   established applies directly: header-derived placement facts may accept a
   format the engine cannot memory-map, provided every route that turns those
   facts into raw-record access refuses it explicitly.

   **B0223/D0222 continuation.** The exact uncompressed `.las` subset is now
   closed for the literal `extra_dims=all` option. D0218's prepared-writer
   layout derives the physical record width, and a reorder-only terminal
   region publishes its declared permutation rather than pretending it wrote
   an SoA column. The public 1M normal/covariancefeatures composition is
   byte-exact at 6.267794x median, and the selection matrix covers both the
   measured 100-byte sink and a wider carried-Extra-Bytes refusal. Automatic
   device placement is deliberately restricted to the measured compressed
   format-7, 1M-point, 36 -> 100-byte row; other counts and source layouts
   remain host-selected. This does not admit named extra dimensions,
   additional writer options, or compressed `.laz` output; r2 and r6 therefore
   remain blocked at their sinks. Continue item 1 at LAZ encode/named-layout
   measurement rather than reopening the proved uncompressed `all` envelope.

   **B0224/D0223 continuation.** The literal `extra_dims=all` LAZ subset is now
   closed for r6: omitted compression and boolean/string true are exact native
   spellings, and the full checked-in r6 reference is 4.529005x median pinned
   PDAL. Automatic placement is restricted to the one measured 1M compressed
   format-7, 36 -> 100-byte row; a central guard prevents every other
   composition model from inheriting the compressed sink. The twenty-case
   physical matrix covers public selection/execution for all three spellings
   and a wider carried-layout refusal; runtime units cover eigen/rank
   refusals. Continue item 1 only at r2's named
   `HeightAboveGround=float32` layout; do not generalize LAZ `all` beyond the
   recorded row without a new layout-specific ladder.

   **B0225/D0224 continuation.** The r2 named-layout item is measured and
   rejected as a writer-only slice. The current public reference is exact but
   resolvably slower at 0.988063x median, and runtime stats show SMRF and HAG-NN
   remain two separate uncalibrated regions behind the writer. Existing exact
   substitution evidence loses by 350.7 ms for SMRF+HAG and about 36% for the
   fork's host HAG-NN alone; device SMRF is unqualified and HAG CUDA has no
   automatic model. Close item 1 here. Reopen the named sink only as part of a
   complete measured profitable r2 stage route, not for native coverage alone.

   **B0228/D0227 continuation.** r5's immediate source deficit is closed
   without claiming a native COPC reader. The engine cannot consume
   `readers.copc`, so the literal checked-in COPC -> option-free stats -> LAS
   grammar now execs pinned PDAL directly. The current public route moves from
   resolvably slower at 0.930792x +/- 0.011642 to parity; the paired dispatcher
   versus engine control is 1.047266x +/- 0.016446 with 9/9 wins. Exact root,
   bounds, numeric representations, option sets, stage order, endpoint
   spellings, and the plain CLI invocation are required; neighboring shapes
   remain in-engine. Keep native COPC pushdown as a P4/catalog item unless a
   complete exact implementation beats this new direct-host baseline.

   **B0229/D0228 continuation.** r3's immediate sink-path deficit is closed
   without claiming native SMRF or GDAL. Exact automatic SMRF replacement is
   disabled, unchanged SMRF makes the adjacent range rewrite unstable, and
   `writers.gdal` is outside the resident endpoint, so the literal checked-in
   LAZ -> SMRF -> ground range -> GDAL IDW grammar now execs pinned PDAL
   directly. The current public route moves from resolvably slower at
   0.966556x +/- 0.006223 to parity; the paired dispatcher-versus-engine
   control is 1.027780x +/- 0.005196 with 9/9 wins. Exact root, numeric
   representations, option sets, stage order, endpoint spellings, and the
   plain CLI invocation are required; neighboring shapes remain in-engine.
   Keep native GDAL assembly as a P4/catalog item unless a complete exact
   implementation beats this new direct-host baseline.

   **B0230/D0229 continuation.** r2's immediate complete-process deficit is
   closed without weakening B0225/D0224's named-writer rejection. The literal
   named sink makes plan-structure admission refuse before placement or point
   work, so the checked-in LAZ -> option-free SMRF -> option-free HAG-NN ->
   named LAZ grammar now execs pinned PDAL directly. The fresh public route
   moves from resolvably slower at 0.973432x +/- 0.011569 to parity; the paired
   dispatcher-versus-engine control is 1.014563x +/- 0.005922 with 9/9 wins.
   Exact root, option types/values, stage order, non-COPC lowercase endpoints,
   and the plain CLI invocation are required; neighboring shapes remain
   in-engine. The named writer remains non-native and uncalibrated. Reopen it
   only together with a complete SMRF/HAG route that beats this new direct-host
   baseline.
2. **Implement the stages real pipelines actually contain.** B0163 split the
   catch-all refusal, and the answer changed: `filters.reprojection` and
   `filters.sample` now report `unsupported_stage`, not a cardinality problem.
   `r1` needs a native reprojection and `r4` needs a native sample; neither is
   a predicate-expression question. B0162 already made `crop` and `range`
   placeable. What remains genuinely under the cardinality heading is
   `filters.expression` on coordinates, still planned to Host by B0162's
   deliberately narrow scope.

   **B0226/D0225 continuation.** r1's immediate complete-process deficit is
   closed without pretending reprojection is native. For the exact measured
   1M reference JSON/bounds and compressed format-7/36-byte header facts, the
   thin dispatcher now execs pinned PDAL directly instead of loading an engine
   that can only substitute crop before delegating. The public route moves from
   resolvably slower at 0.938267x +/- 0.007951 to parity; the paired direct
   versus engine control is 1.075614x +/- 0.011503 with 9/9 wins. Neighboring
   counts, layouts, bounds, options, SRS shapes, and CLI modifiers remain
   in-engine. A future native reprojection belongs here only if it beats this
   new direct-host baseline end to end and remains byte/diagnostic exact.

   **B0227/D0226 continuation.** r4's immediate deficit is closed without a
   native sample implementation. The literal measured 1M reference grammar and
   compressed format-7/36-byte input facts automatically request the existing
   exact hybrid statistical-outlier CUDA path only on the calibrated RTX
   4090/SM89/CUDA 13.3/driver 610.43.03 profile. Range uses the existing point
   program; sample and LAZ I/O stay upstream. The complete public route is
   exact at 3.686747x median including full-input fingerprinting (3.715180x +/-
   0.055459 paired, 9/9 faster), and
   the proof environment requires actual CUDA execution. Keep native sample as
   a catalog-coverage item unless a separate complete route beats this new
   baseline; do not widen outlier selection without a new count/layout/device
   calibration. The full-file fingerprint deliberately guards the measured
   reference bytes against accidental drift; it is not a cryptographic
   identity claim.

   *(Superseded framing, retained for the record:)* B0158 first showed the
   refusal was a catch-all that misreports — `crop` and `range` already compile to `PredicateProgram` with
   `native = true` and a device residency preference, and are refused anyway;
   so is `filters.expression`, which `docs/stage-coverage.md` lists as
   automatically selected. And `filters.reprojection`, which cannot change
   point count, reports the same cardinality refusal — because
   `fusion.cardinalityPreserving` defaults to `false`
   (`include/pdg/Plan.hpp:123`), so any stage not explicitly declaring itself
   preserving, including every unsupported fallback stage, trips it. The
   refusal means "not known to preserve point count", not "changes point
   count". The work is therefore to split those two conditions into distinct
   reasons and to find why the declared-predicate exemption does not admit a
   bare predicate — not, as previously written here, to express these stages as
   predicates, which they already are. B0159 answered the second half: every
   clause of the exemption passes except `preferredResidency == Device`,
   because plan-time residency calls
   `exactDeviceExpression(expression, allowCoordinateLoads=false)`
   (`src/stages/Assign.cpp:600`), so any predicate reading X, Y, or Z is
   planned to Host. `crop` is inherently such a predicate and can never be
   placed; the identical `range` filter is refused on `Z` and admitted on
   `Intensity`. The fix is to apply the repository's own declare-then-verify
   pattern — the runtime `predicateSupportsExactDevice(batch, program)`
   overload already checks the real condition — which is a placement-semantics
   change needing its own proofs and baseline measurement. Note also that any pipeline containing
   an unsupported stage is refused whole under this reason, which is a far
   broader constraint than a cardinality rule.
3. **Broaden calibration coverage — B0157 and B0165 both landed here from
   opposite directions.** Placement models are keyed at option level, not
   stage-sequence level: `outlier` finds no model even in a shape close to
   B0088's qualified composition. B0223 closes uncompressed `extra_dims=all`,
   and B0224 closes only r6's measured compressed-literal-`all` row using
   D0218's derived width. Named extra dimensions and all other compressed
   layouts remain outside calibrated publication. This gates `r4`, r2's sink,
   and every route in the automatic-selection candidate list below, all of
   which need "a calibrated model plus a fail-closed admission predicate". It
   is the most reusable remaining item: one mechanism unlocks many routes.
   Measure what a model must cover before building one.

4. **Discharge the selection proof burden, then generalize selection.** B0156
   removed the cost objection — declining is now free, and generalized
   selection no longer regresses anything. What remains is that it extends
   automatic selection past what the exactness matrices prove. That burden
   should be discharged deliberately: broaden the matrices, then flip the gate,
   then measure. B0149 measured 3.62x waiting behind it on uncompressed LAS and
   B0152 measured 3.21x on compressed.
4. **Cross-stage residency and planner-owned product reuse.** Every win of this
   shape has paid (B0052 -35.5%, B0111 -34.5%, B0060 -8.0%, B0062 -5.7%) and it
   is the one category that compounds across stages instead of helping one.
   `mixed_calibration_models` sits here: it blocks `r6` behind `extra_dims` and
   needs a calibration answer rather than a policy relaxation.
5. **Fallback-frequency reduction.** A fallback costs 100% of the benefit.
   Never replace a tie, incomplete result, or unsupported case with an unproved
   answer (D0074, D0099).
6. **Host-side optimization where CUDA is unavailable or loses.** `r2` and
   `r3` both run SMRF. Its CUDA prototype is unqualified under D0221, and the
   exact KD2Index host wrapper/fallback is selected. Under D0208 the right
   answer there is host threading and algorithmic work, which is first-class
   effort rather than scaffolding. B0229/D0228 closes r3's immediate public
   deficit at parity by bypassing discarded engine work; any SMRF or GDAL
   optimization must now beat that direct-host complete-process baseline, not
   merely an engine-internal stage comparison. B0230/D0229 now closes r2's
   immediate public deficit by the same measured principle while preserving
   D0224. Further SMRF/HAG work is worthwhile only when a complete exact route
   beats the direct-host r2 baseline; the existing fork substitution remains
   rejected at +350.7 ms +/- 38.5.
7. **Remaining per-stage kernels — no longer required at all.** D0208 withdrew
   catalog-wide CUDA coverage as a release criterion. These are scheduled only
   when a reference pipeline shows the stage dominating its wall clock and CUDA
   is the measured best answer. Nothing is retired: exact-but-slower lanes stay
   in-tree, host-selected, and labeled catalog coverage.

**Process cost is now largely paid down.** B0154 attributed the constant
overhead, B0155 removed 43% of engine startup by loading NVRTC lazily, and
B0156 returned 167 ms to every unplaceable pipeline by ordering admission
checks by cost. About 13 ms of engine startup remains, attributed to the binary
and its PDAL linkage rather than to CUDA.

**Automatic-selection candidates (priority 4).** The original queue had ten
GPU-native routes carrying positive-sounding qualification language while
remaining unselected, but four of them — `filters.transformation`,
`filters.iqr`, `filters.mad`, and `filters.colorinterp` — are qualified
*negatively* and must never be promoted.

Of the six genuine candidates, B0231/D0230 completes
`filters.neighborclassifier`, B0232/D0231 completes `filters.sort`,
B0233/D0232 completes `filters.label_duplicates` inside its measured hybrid
composition, B0234/D0233 completes `filters.skewnessbalancing` inside its
strict direct composition, and B0235/D0234 completes `filters.hag_nn,count=1`
inside its strict direct composition. B0237/D0236 corrects and completes the
sixth and final candidate, `filters.hag_delaunay,count=3`, inside its separately
calibrated strict direct composition. The original automatic-selection
candidate queue is therefore complete; do not manufacture another promotion
target from a
negative or unmeasured stage row.

That is one reusable mechanism, not six pieces of work, and this repository has
already built it five times — B0080 radiusassign, B0092/B0093 outlier
composition, B0097 approximatecoplanar, D0106 LOF, and D0108 nndistance. Treat
the mechanism as the slice and the routes as its instances. Several of these
routes are additionally gated by data-dependent exact repair, so admission must
stay narrow and fail-closed; a route whose ordinary data trips fallback is not
a promotion candidate however fast its named fixture is.

**B0231/D0230 continuation.** The first positive route is now promoted under
a separate direct-boundary model. The exact `readers.las ->
neighborclassifier(k=7) -> writers.las` route automatically selects only for
uncompressed format-7 36-byte input/output records, 250K--16M points, the
mapped LAS source, direct Classification publication, a 112-byte/point
planner-owned index, and the exact SM89 profile. The measured 50K row is a
0.685x loss and stays host-selected; option, topology, layout, compression,
`k`, index, device/profile, preflight, and post-execution-proof drift all fail
closed. The final 1M public route is byte/diagnostic/status/order exact at
4.033570x median pinned PDAL, while a nine-pair attribution proves the direct
boundary itself is 1.150355x +/- 0.024677 faster than the former resident
boundary. At that checkpoint five candidates remained; do not generalize the
neighborclassifier model to the ordinary stage model or neighboring layouts.

**B0232/D0231 continuation.** The second positive route is promoted under its
own `sort-direct-compose` model without reopening B0130's rejected ordinary
sort boundary. The exact literal `readers.las -> sort(Z,ASC,NORMAL) ->
writers.las(extra_dims=all)` route automatically selects only for
uncompressed format-7 36-byte input/output records, finite comparator-unique
Z, 600K--16M points, 8-byte upload/download facts, zero packing/indexes, one
lane, a measured conservative 64-byte/point reservation, and the exact SM89
profile. Final public results are byte/diagnostic/status/order exact at
1.842475x pinned PDAL at 600K and 3.267173x at 1M. The 500K/550K
direct-versus-public intervals span parity and stay host-selected. Allocation
tracking measures 56.283 bytes/point at the floor and 56.267 at the cap;
placement and resident preflight both reserve 64. Duplicate/non-finite keys,
neighboring grammar/layout/profile/budget facts, and side-effect-free
publication refusals fall back before commitment. At that checkpoint four
candidates remained; do not generalize the direct model to ordinary sort or
neighboring ordering forms.

**B0233/D0232 continuation.** The third positive route promotes the fastest
measured executor rather than the most resident one. The literal
`readers.las -> label_duplicates(Classification) -> nndistance(k=10) ->
assign(UserData = Duplicate) -> writers.las` composition automatically selects
the exact per-stage CUDA hybrid only for uncompressed LAS 1.4 format-7/36-byte,
zero-VLR/EVLR inputs from 250K through 16M on the exact SM89 profile. The 50K
row is a 0.893x loss and stays host-selected. A current resident probe is exact
but remains 4.3% slower than the hybrid and is not restored. Final public runs
are byte/diagnostic/status/order exact at 2.846286x pinned PDAL at 250K and
6.391572x at 1M. Grammar, layout, count, and device/profile drift delegate;
recoverable label CUDA failure executes its exact host operation before the
writer; ordinary writer filesystem behavior is differentially pinned. Continue
with skewness-balancing, HAG-NN, then
HAG-Delaunay; do not generalize either ordinary stage model or revive the
slower resident direction.

**B0234/D0233 continuation.** The fourth positive route promotes the already
exact mapped-source/permutation-publisher skewness composition under its own
`skewness-direct-compose` model. The literal option-free
`readers.las -> filters.skewnessbalancing -> writers.las(extra_dims=all)`
graph automatically selects only for uncompressed LAS 1.4 format-7/36-byte
input/output, finite comparator-unique physical Z, mapped source, 450K--16M
points, a conservative 65-byte/point peak, and the exact SM89 profile. A
public nine-pair 400K control is only 1.065118x and remains host-selected;
final public runs are exact at 1.200177x pinned PDAL at 450K and 3.091007x at
1M. Data-proof, allocation, grammar, layout, profile, and side-effect-free
publication refusals fall back before commitment. Continue with HAG-NN and
then HAG-Delaunay; do not generalize this direct model to the ordinary stage
boundary or neighboring layouts.

**B0235/D0234 continuation.** The fifth positive route promotes only the
already-exact count-one HAG-NN mapped-source/direct-double-output composition.
The literal option-free `readers.las -> hag_nn(count=1) ->
writers.las(extra_dims=all)` graph automatically selects for uncompressed LAS
1.4 format-7 40-byte input with exactly one unsigned-32 `OffsetTime` Extra
Bytes descriptor, 48-byte output, 450K--16,000,002 points, one planner-owned
2D kNN index/region/lane, a conservative 160-byte/point peak, and the exact
SM89 profile. The 400,002 direct row is only 1.104286x and stays outside the
safer floor. Final public results are byte/diagnostic/status/order exact at
1.216823x pinned PDAL at 450K and 2.461791x at 1,000,002 points. Grammar,
layout, count, budget, source, profile, rewrite, execution-proof, and side-
effect-free publication drift fail closed before commitment. Continue with
HAG-Delaunay; do not extend this model to wider HAG-NN counts or neighboring
layouts.

**B0237/D0236 correction.** The sixth and final positive route remains the
already-exact count-three HAG-Delaunay mapped-source/direct-double-output
composition. Review found that B0236 could overwrite calibrated placement and
proved selection without proving successful CUDA use. The corrected literal
`readers.las -> hag_delaunay(count=3) -> writers.las(extra_dims=all)` graph
automatically selects only for uncompressed LAS 1.4 format-7 40-byte input with
exactly one unsigned-32 `OffsetTime` Extra Bytes descriptor, 48-byte output,
500,001--16,000,002 points, one planner-owned 2D kNN index/region/lane, a
conservative 184-byte/point peak, and the exact SM89 profile. A fresh 450K
nine-pair public run reaches only 1.093356x and is now refused. Final public
results are byte/diagnostic/status/order exact at 1.245213x pinned PDAL at
500,001 and 2.049354x at 1,000,002 points, with 9/9 wins at both sizes.
Grammar, semantic layout, count, budget, source, profile, rewrite, preflight,
successful-device proof, repair, and side-effect-free publication drift fail
closed before commitment. Same-final-binary stats prove the calibrated direct
executor and actual device use. Do not extend this model to wider Delaunay
counts or neighboring layouts; resume reference-pipeline attribution or
host-side work under priorities 5/6.

**B0238/D0237 continuation.** The next profile-directed host slice attacks
B0043's dominant exact LOF closure repair without weakening B0039's rejected
device-tie direction. Large repair closures now build the existing single
pinned KD3 index over a contiguous immutable binary64 XYZ backing and execute
deterministic fixed-chunk k-distance, density, and factor passes with explicit
join barriers. Small closures and ordinary KD3 callers remain serial and
uncached. The 4M cache payload is 96 MB; matched peak RSS rises by 103,224 KiB.
Cached construction is exception-atomic and ordinary failure retries uncached;
public proof flags fail closed before oracle delegation. Current-final-binary
attribution is exact at 1.840449x over the disabled control, reducing repair
from 2.183796 to 0.504710 seconds. The final nine-pair public route is exact at
18.300959x pinned PDAL (35.860210 versus 1.959472 seconds); a separate
option-free three-pair confirmation is 18.550858x. Functional support,
GPU-native coverage, placement, and the B0043 automatic envelope do not
change. Resume reference-pipeline attribution or another measured host/I/O
limiter; do not revive the unproved device-repair direction.

**B0239/D0238 continuation.** A new complete-process r2 profile corrects the
old assumption that SMRF was the only useful terrain target. For the immutable
checked-in 1M reference fixture, host SMRF plus exact CUDA HAG-NN with 347-row
selective pinned-host repair is byte/metadata/order/diagnostic/status exact and
measures 1.270989x pinned PDAL (1.536685 versus 1.209047 seconds), with 9/9
wins. The selector is deliberately fingerprinted to the exact r2 fixture,
grammar, named output layout, and SM89 profile; wrong count, extent/header,
payload, grammar, externally injected internal marker, or post-selection CUDA
decline fails closed before writer side effects. Exceptional rows use one
planner-owned compatibility ground KD2 product. SMRF and LAZ I/O remain
host-owned, and the final profile identifies lazperf decode and remaining
host/I/O/startup work as the next r2 costs. Do not generalize this endpoint
without a complete measured ladder.

**Expanded reference-suite prerequisite (D0239/B0240: complete).** Before selecting
another optimization target, promote eight additional production workflows to
first-class references `r7` through `r14`: DSM, raster/orthophoto colorization,
standalone polygon clipping with geometry reprojection, decimation,
PDAL-native classification/refinement, deterministic tiling/chipping,
multi-file merge/mosaic, and LAS/LAZ/COPC conversion/compression. First add
their deterministic pipelines, fixture provenance, harness materialization,
oracle baselines, complete-process profiles, differential/refusal matrices,
and a measured opportunity ledger. The headline suite has fourteen equally
weighted workflows; supporting decimation and conversion variants do not gain
extra aggregate weight. Report both the equal-workload geometric mean and the
paired total wall time for one execution of all fourteen. No new optimization
target may be chosen from intuition while this prerequisite remains open.
B0240 closes it: all fourteen headlines and twelve zero-weight variants have
exact warm/cold baselines, r7-r14 complete-process profiles exist, and the
fail-closed manifest/runner reports 1.211602x warm and 1.206003x cold
equal-workload geometric mean plus 1.563923x warm and 1.554365x cold
total-wall speedup.

**D0240/B0241 r11 vertical slice: complete.** Test-first attribution rejected a
literal direct delegate at an exact 0.901714x with 0/9 wins. The regression was
instead an optional-coordinate-cache branch inside every ordinary nanoflann
KD3 callback. Construction-time cached/uncached specializations restore the
ordinary branch-free host adapter while retaining B0238's explicit contiguous
cache. r11 is exact at unresolved oracle parity: 1.000990x over nine warm pairs
and 1.003846x over three cold pairs. The physical 4M cached-LOF proof remains
exact at 20.295840x. No r11 route, native coverage, stage grammar, or placement
model was widened; the domain-qualified neighbor classifier remains outside
the exact CUDA envelope. Updated suite aggregates are 1.220209x warm and
1.215129x cold equal-workload geometric mean, plus 1.632359x warm and
1.627712x cold total-wall speedup.

**B0242/B0243 r8 vertical slice: complete, prototype rejected.** Attribution
confirmed that the engine eventually delegates all PROJ, GDAL colorization,
LAZ decode, and LAZ encode work to sibling PDAL. A literal direct-delegation
prototype was exact, but the corrected-final 21-pair route A/B remained
unresolved at 0.995822x +/- 0.004522 in report orientation, while the direct
prototype resolved slower cold at 0.988765x pinned PDAL. The matcher and probe
were removed; r8 remains host/engine selected with no native or performance
qualification. Independent review also replaced the drift-prone environment
allowlist with a fail-closed rule: every actual startup `PDG_*` name except
`PDG_ORACLE_PDAL` is engine-owned, including future controls.

**D0243/D0244, B0244/B0245 r13 vertical slice: complete, prototype rejected.**
Complete-process attribution shows sequential heterogeneous lazperf decode and
compressed LAS publication dominate; streaming `filters.merge` is effectively
free. Direct sibling delegation alone was exact but unresolved. One
`readers.las` worker resolved faster warm, while an engine-retained
implementation resolved slower. After independent review narrowed the route to
plain invocation and corrected its portable loader linkage, the final 21-pair
warm gate resolved at 1.009929x but the required 41-pair cold gate remained
unresolved at 1.001186x. The matcher, fingerprint, proof, and override were
removed. r13 remains exact host/engine selected with no generalized merge,
reader, writer, host-native, performance-qualified, or GPU-native coverage.

**D0245/B0246 r12 vertical slice: complete, prototypes rejected.** The retained
complete-process profile is LAZ-decode shaped; filesystem syscall time is not
the primary limiter for the seven exact outputs. Direct sibling execution is
unresolved versus pinned PDAL warm and cold. The literal public prototype
resolves slower than pinned PDAL, and its same-final direct-versus-engine A/B
is unresolved in both cache states. A bounded 1/2/4/6/8 reader-worker sweep
also finds no qualified setting: four workers is unresolved and every other
tested count is slower than the default seven. The matcher and positive tests
were removed, and the literal graph is frozen engine-owned. No splitter,
reader, writer, host-native, GPU-native, performance-qualified, or automatic
selection claim is added.

**B0247/D0246 r9 vertical slice: complete, workload corrected and prototypes
rejected.** The original axis-swapped geometry produced an empty artifact and
is superseded. Corrected r9 emits 473,825 points and has bounded
polygon/multipolygon/hole/boundary/reprojection, malformed, refusal, and
determinism coverage. Profiles identify LAZ decode and public startup rather
than polygon membership as the limiter. Literal direct-exec and private
in-process PDAL CLI prototypes are exact but resolve slower than pinned PDAL
over 21 warm pairs, so both are removed and r9 remains engine/host selected.
Fresh same-binary headline aggregates are 1.205506x/1.206910x warm/cold
geometric mean and 1.617787x/1.615619x total-wall speedup. The suite now
refuses mixed binary hashes and any measured r9 artifact whose fixed LAS
header does not report 473,825 points.

**D0247/B0248 r7 vertical slice: complete, bounded direct-host route retained.**
The selected literal graph removes only an unused engine process before
unchanged sibling-PDAL first/only-return maximum-Z GeoTIFF execution. It is
exact and resolves 1.058781x warm / 1.049022x cold against the same-final
engine route. Its pinned-PDAL comparison remains a reported 0.982017x warm
loss and unresolved 0.990449x cold result, so it is a faster PDG dispatcher
backend rather than a native raster acceleration claim. A generated
compressed-LAZ selected-route differential plus fallback/refusal/malformed and
determinism cases freezes complete GeoTIFF bytes/metadata and process
observables. Product `PDG_*` controls and every neighboring grammar remain
engine-owned. Corrected-clock fresh aggregates are 1.223460x/1.227305x
warm/cold geometric mean and 1.600366x/1.621301x total-wall speedup.

**D0248/B0249 r10 vertical slice: complete, bounded direct-host route
retained.** The headline profile is LAZ-decode shaped: format-7 point decode
accounts for 0.302 of 0.430 sampled CPU seconds, while voxel centroid/nearest
original-record selection accounts for only 0.002 seconds inclusive. The
selected literal lowercase-LAZ, `cell=2.5`, string-compression graph removes
only the otherwise unused engine process before unchanged sibling-PDAL
execution. It resolves 1.041226x warm / 1.034230x cold against the same-final
engine path with 9/9 wins in both states; its pinned-PDAL rows remain
unresolved at 1.002615x warm and 0.993073x cold median speedup.

A nine-case generated-fixture matrix freezes dense, sparse, empty,
malformed-input, legal drift, invalid-option, numeric zero-cell, and selected
repeat-determinism behavior with complete LAZ bytes/order/streams/status.
Every neighboring grammar and all six zero-weight decimation variants stay
engine-owned. This adds one performance-qualified direct-host route but no
host-native or GPU-native voxel/LAZ stage. Fresh same-binary aggregates are
1.235323x/1.237401x warm/cold geometric mean and 1.631522x/1.633568x
total-wall speedup.

**D0249/B0250 r14 vertical slice: complete, reader-worker tuning rejected.**
The headline already uses the older generic direct sibling-PDAL delegation,
so no engine-startup optimization or new selector is available. A
12-case/19-execution generated-fixture matrix now freezes all seven supported
LAS/LAZ/COPC directions, repeated byte determinism, header/compression/COPC
identity, truncated and malformed sources, unsupported writer behavior, and
complete process observables. The complete child profile observes 37,323,822
bytes read and 6,748,024 written in 0.535 seconds; its clock sampler is
explicitly unreliable after a descendant timer reset, so no function-level
CPU share is inferred.

Reader-worker counts 1/2/4/6/8 are all exact. One worker confirms at
1.017714x warm but only 1.004441x cold median, with the cold paired interval
spanning parity (1.001577x +/- 0.014533); every other screen is unresolved or
slower. No tuning is selected. r14 remains unchanged-PDAL host execution with
no new host-native, GPU-native, performance-qualified, or automatic-selection
entry. Because the product binary is unchanged, the current aggregate remains
1.235323x/1.237401x warm/cold geometric mean and 1.631522x/1.633568x
total-wall speedup.

**D0250/B0251 exact parallel LAZ compression: complete and selected.**
Independent 50K lazperf chunks reproduce the complete sequential payload and
chunk table at empty, boundary, and multi-chunk counts. The production pool
keeps one moved raw/result chunk per worker and publishes on the writer thread
in source order. Its maintained 1M format-7 primitive is byte-exact and
1.735238x faster. The generated r14 matrix is now 13 cases/21 executions and
asserts both the entire input-admission envelope and real two-worker writer
activation.

Only the literal 1M LAS -> LAZ headline shape selects two workers; every
layout/count/grammar/modifier/control drift and all six other conversion
directions retain the serial exact path. B0252/D0251 move external product
environment refusal ahead of automatic routing, strip a forged internal worker
value before engine execution, and compile the activation assertion only in
test builds. The corrected final public qualification resolves 1.515827x warm
/ 1.504457x cold with 9/9 wins. Fresh all-headline results are
1.262824x/1.260608x warm/cold geometric mean and
1.641092x/1.634768x total-wall speedup.

**D0252/B0253 LAZ reader scheduling and setup: complete and rejected.** The
production reader already schedules chunks independently and consumes them in
source order. Ten and twelve workers are slower than the seven-worker default
for both r7 and r10 and use more peak RAM. Mode-gating Point14 codec setup
improves factory construction 1.661797x and the complete 1M-record primitive
1.014837x, but public r10 warm is unresolved at 1.012374x median / paired
1.009303x +/- 0.013219. The production patch is removed; no cold promotion or
selection change is made. A maintained formats-6--8/Extra-Bytes exact decoder
unit and production-factory primitive remain.

**D0253/B0254 bounded r7 reader schedule: complete and selected.** A forced-
inline integer-decoder experiment improves the primitive only 1.010245x and is
removed. Completing the reader screen finds four workers positive, six
unresolved, eight slower, and preserves B0253's 10/12-worker rejections. Four
workers resolve on the retained final r7 process at 1.031858x warm and
1.020398x cold, with exact complete TIFF metadata/bytes, streams, status, and
lower median peak RSS. The launcher supplies the existing upstream reader
option only for the plain literal one-million-point compressed format-7/36-
byte r7 DSM shape. Every drift fails closed before that option is appended.

The same integrated four-worker r10 prototype resolves warm but remains
unresolved cold at paired 1.002292x +/- 0.009367, so r10 keeps B0249's direct-
host route with the default reader and no broader LAZ default is inferred.
Fresh exact fourteen-headline aggregates are 1.267318x/1.262102x warm/cold
equal-workload geometric mean and 1.642633x/1.632910x total-wall speedup.

**D0255/B0256 r11 shared host index: complete and selected.** Prefix
attribution assigns about 3.65 seconds to statistical outlier and another 2.53
seconds to the neighbor-classification tail. Independent review found that the
first B0255 implementation could adopt a stale pre-existing product after a
generic XYZ assignment. The corrected statistical stage explicitly invalidates
incoming products, builds its unchanged fresh exact nanoflann tree, and
publishes that tree through the PointView cache; the adjacent upstream neighbor
classifier reuses it instead of rebuilding equivalent XYZ state. Radius mode,
query order, recurrences, options, diagnostics, output order, routing, and CUDA
qualification are unchanged.

The corrected final nine-pair public process is byte/metadata/order/stream/
status exact at 1.029892x warm and 1.037311x cold, with 9/9 wins in both
states. An eight-case/nine-execution matrix adds cached-index construction and
generic X mutation to legal drift, malformed options/domain, and repeat
determinism. Fresh corrected fourteen-headline aggregates are 1.267705x/
1.278088x warm/cold equal-workload geometric mean and 1.656538x/1.665729x
total-wall speedup.

**D0256/B0257 r11 vote-tally slice: rejected by attribution before a
prototype.** The checked-in `pdg_r11_neighborhood_attribution` harness measured
the hypothesis in isolation on the literal r11 prefix: the complete upstream
tally is about 0.043 seconds of the 6.8-second process and the allocation-free
ascending-key variant saves 4--6 milliseconds (about 0.1% of wall) against a
+/- 0.9% paired gate error, so no process gate can resolve it and D0204's
cheap-prototype rule rejects it. The generic ordered-map tally remains the only
implementation. The same harness attributes about 3.37 seconds to the
statistical outlier's per-point kNN pass and about 2.18 seconds to the
classifier's per-point kNN pass, and shows both bit-identical under fixed-chunk
multi-worker execution over the read-only PointView-owned nanoflann index
(about 0.60 seconds combined at eleven workers). It also found that `gprofng`
on this workstation undersamples CPU by roughly 7--10x, which is why the
profile-directed tally hypothesis overstated a 0.6%-of-stage cost.

**D0257/B0258 r11 exact host worker passes: complete and selected.** The
maintainer approved the measured direction. The statistical outlier and
neighbor classifier per-point kNN passes now run over fixed contiguous row
chunks on host workers through `pdal/private/HostNeighborhoodWorkers.hpp`
(D0237's pattern), with the outlier's serial online-moment reduction and the
classifier's vote/merge/application, options, diagnostics, output order, and
every route/envelope unchanged. Final nine-pair public results are exact at
**4.309167x warm / 4.345954x cold** (paired 4.321977x +/- 0.034672 and
4.352975x +/- 0.038573, 9/9 wins); the same-final-binary serial control is
1.035408x and 50K/250K controls are 3.050913x/4.297817x. Fourteen-headline
aggregates are **1.401126x/1.410744x** warm/cold equal-workload geometric mean
and **2.407033x/2.408896x** total-wall speedup. Host Debug is 523/523; the
eighteen-case r11 matrix passes CUDA Release, Host Debug, leak-disabled
ASan/UBSan, and TSan. Forcing the existing CUDA statistical outlier on r11 is
exact but slower (4.069126x) because the domain-qualified classifier is
outside the CUDA wrapper and rebuilds the shared host tree; a device route
pays only as a full SMRF -> outlier -> classifier shared-index composition and
is recorded as a candidate, not started.

**D0258/B0259 default LAZ chunk-compression workers: complete and selected.**
The next measured limiter after B0258 was r6's serial 100-byte-record LAZ
compression. Instead of another bounded per-shape selector, B0251's exact
worker pool was measured over all fourteen headlines through a same-binary
engine probe (every LAZ sink faster at two and four workers, non-LAZ workloads
unchanged, all exact; r6 knee at four), and `writers.las` now defaults to
`min(4, hardware threads)` workers; the internal channel still lets the
launcher keep its measured two-worker r14 selection. Final public aggregates
are **1.597555x/1.589241x** warm/cold equal-workload geometric mean and
**2.792319x/2.772361x** total-wall speedup, all exact; nine-pair r6
6.316372x/6.387622x, r13 1.437716x/1.434382x, r1 1.081819x/1.101200x. r9,
r12, r13, and r8 move from exact losses to wins without routes.

**B0260/D0259 attribution and an open exactness defect (STOP point).** A new
stats diagnostic attributes r6's remaining wall: 0.47 s of its 1.24 s is a
full uncached nanoflann KD3 build to repair 2,343 eigen tie rows; the writer
is 0.21 s and reader 0.15 s. Cached-coordinate KD3 backing builds 4x faster
(0.077 s versus 0.307 s at 1M) and makes worker kNN passes 3--4x faster
(bit-identical). While reasoning about publishing such a tree, B0256's
published statistical-outlier tree was found **byte-inexact versus pinned
PDAL** for `outlier -> filters.assign(X = X * 3) -> neighborclassifier` (and,
symmetrically, `producer -> mutator -> outlier -> consumer`), because pinned
PDAL's outlier neither invalidates nor publishes a product. The r11 graph has
no mutator and stays exact; B0258/B0259 are not implicated.

**D0260/B0261 pinned outlier semantics restored; pinned-oracle lane: complete.**
The statistical outlier again builds a private fresh index and neither reads,
invalidates, nor publishes the PointView product, on the exact
cached-coordinate backing (24 transient bytes/point). A new pinned-oracle
differential lane (`PDG_PINNED_ORACLE_EXECUTABLE`, `differential.py
--candidate-oracle`, `pinned_oracle` r11 cases) was shown to fail on the
defective library and pass on the fix. r11 is **5.207843x/5.267489x**
warm/cold over nine pairs; aggregates **1.592353x/1.595639x** geometric mean
and **2.794157x/2.794157x** total wall, all exact.

**D0261 `--fast` contract: shipped, behavior-neutral.** `pdg --fast` keeps
point records bit-identical and ordered and relaxes only diagnostics, error
text/status, header/VLR metadata, and metadata JSON (maintainer's choice);
`benchmark_reference.py --contract fast` compares ordered record digests.
No route changes yet (see `docs/fast-mode.md`).

**D0262/B0262 terminal-sink private repair tree: complete and selected.**
The resident rewrite marks a neighborhood region followed by the terminal
writer alone; the eigen-family exact tie repair then keeps a private
cached-coordinate tree instead of publishing an uncached one nobody reads
(0.525 s -> 0.142 s at 1M). r6 is **8.648309x/8.664582x** warm/cold;
aggregates **1.633482x/1.624540x** geometric mean and **2.924049x/2.894853x**
total wall, all exact. The same entry fixes launcher r2-marker arming on the
engine route (latent since B0243) and records the session's first full
physical CUDA aggregate, 827/827.

**D0263/B0263 cached-coordinate KD3 as the published default: complete and
selected (D0259 option 2).** `PointView::build3dIndex()` builds the exact
cached-coordinate adapter; a view coordinate epoch moved by every mutation
path (view/PointRef X/Y/Z writes, packed/raw access, additions, appends,
order mutations) refreshes a reused snapshot from live coordinates, so pinned
nanoflann's stale-tree/live-coordinate reuse semantics hold bit-for-bit
(`Kd3Refresh` units, r11 pinned producer/mutator/consumer matrix, full Host
528/528 and CUDA 831/831 aggregates with the snapshot verifier armed). r11 is
**7.372457x/7.286586x** warm/cold; aggregates **1.672490x/1.663704x**
geometric mean and **3.073590x/3.042106x** total wall, all exact.

**D0264/B0264 exact host workers inside the upstream SMRF: complete and
selected.** `PDG_DEBUG_SMRF_PHASES` attributed SMRF's 0.34 s at 1M to the
progressive filter's 189 diamond passes (0.134 s), two void fills, and the
point passes. Pooled morphology (eight workers, identical fold), chunked
void-fill queries, chunked classify-ground, and a merged minimum-Z raster
bring SMRF to 0.17 s: r3 **1.352415x/1.338430x** (from parity), r11
**8.867727x/8.863616x**; aggregates **1.731986x/1.729459x** geometric mean
and **3.185933x/3.147773x** total wall, all exact. Proof: forced-worker
repeats of the whole SMRF matrix and three pinned-oracle 240 x 240 lattice
cases. r2's automatic route uses the fork's SMRF port and is unchanged.

**D0265/B0265 fork SMRF port morphology pooled: complete and selected.** r2
**1.624190x/1.604373x**; aggregates **1.750757x/1.737266x** geometric mean
and **3.268150x/3.192511x** total wall, all exact.

**D0266/B0266 slot-pooled exact reprojection: complete and selected.**
`Streamable::processStreamBatch` (opt-in whole-batch hook) plus a fixed-slot
pool with one lazily cloned GDAL transformation per slot make
`filters.reprojection` parallel in streaming and standard modes; slots
transform into scratch under a quiet GDAL handler and commit only when no
row failed, otherwise the rows are re-run serially so every diagnostic and
GDAL's per-object error accounting stay pinned. r8 **2.104750x/2.093405x**,
r1 **1.487025x/1.458954x**; aggregates **1.870298x/1.869920x** geometric mean
and **3.573691x/3.535527x** total wall, all exact. A 14-case pinned-oracle
reprojection matrix (both modes, forced/natural workers, `where`, dropped and
many-failure fixtures, `error_on_failure`) is registered.

**D0267/B0267 hashed sample voxel table: complete and selected.** r4
**5.306974x/5.271122x**; aggregates **1.902538x/1.894000x** geometric mean and
**3.697978x/3.657649x** total wall, all exact; an 8-case pinned-oracle sample
matrix is registered.

**D0268/B0268 parallel LAS record packing: complete and selected.**
`writers.las` packs records on a fixed-slot pool in the standard block path
and a new `processStreamBatch` override (slot-ordered `las::Summary` merges;
any deferred per-point warning or exception discards the run before commit
and repeats it serially; discard/where/first-point/small runs stay serial).
r14 **2.084120x/2.048271x**, r13 **1.949116x/1.963703x**, r12
**1.456797x/1.459860x**, r6 **9.136681x/9.095218x**, r1
**1.573685x/1.580186x**; aggregates **2.133083x/2.122241x** geometric mean
and **4.063740x/4.028901x** total wall, all exact; a 23-case pinned-oracle
writer matrix with a test-only path trace and `LasSummaryMerge` units are
registered.

**D0269/B0269 parallel LAS record unpacking: complete and selected.**
`Streamable::readStreamBatch` (opt-in reader hook) and per-tile runs in
`LasReader::read()` unpack decoded records on a fixed-slot pool while tile
consumption stays serial and pinned; small runs, streaming `start`, and the
read callback stay serial. r14 **2.656026x/2.614287x**, r13
**2.682353x/2.700326x**, r7 **1.320508x/1.307258x** (from parity), r9
**1.901385x/1.911693x**, r1 **2.114222x/2.107619x**, r12
**1.946695x/1.902648x**, r10 **1.446846x/1.423698x**, r3
**1.586718x/1.579299x**, r6 **10.122291x/9.872648x**; aggregates
**2.558472x/2.555054x** geometric mean and **4.662196x/4.655083x** total
wall, all exact; a 30-case pinned-oracle reader matrix, a test-only path
trace, and the two-segment `LasReaderUnpack` unit are registered.

**D0270/B0270 ordered parallel COPC decode: complete and selected.**
`readers.copc` under `requests=1` keeps one request thread but decompresses
on a decode pool and emits tiles in pinned fetch order: r5
**1.830419x/1.943686x** (from parity); aggregates **2.613780x/2.624146x**
geometric mean and **4.559263x/4.574280x** total wall (warm repeat
2.674064x/4.727176x), all exact; a 19-case pinned-oracle COPC matrix on a
21-node fixture with a test-only path trace is registered, and the
sanitizer differential lane carries a `vendor/lazperf`-scoped UBSan
suppression.

**D0271/B0271 `--fast` tie-order contract: complete.** Under `pdg --fast`
the spatial index publishes no `KnnDistanceTie` (per-build `__constant__`
mask in both gather kernels plus the host-index paths through
`pdg::knnStatusMask()`), so device tie choices stand and no CPU tie repair
runs; the launcher consumes the flag on every route. Fast-contract finals:
r6 **11.587544x/11.206291x** (25 of 1M records differ, attributes only), r2
**2.043481x/2.128425x** (125 records); fast aggregates
**2.711573x/2.712707x** geometric mean and **4.923939x/4.927839x** total
wall; the exact suite on the same binaries is unchanged and byte-identical.
The fast comparator (record-by-record, identical count/layout/coordinates,
bounded differing records) and labeled fast aggregates are in the runners.

**D0272/B0272 automatic r4 CUDA outlier selector retired: complete.** The
exact host path (worker kNN outlier, hashed sample, parallel LAS I/O)
measures faster than B0227's literal 1M CUDA route at 1M (0.58 s vs 0.72 s)
and 4M (2.55 s vs 2.78 s forced), byte-identical, so the selector no longer
selects by default (opt-in `PDG_EXPERIMENTAL_AUTOMATIC_R4_OUTLIER_CUDA` keeps
the route's lane). r4 **7.422330x/7.440248x**; aggregates
**2.700250x/2.632576x** geometric mean and **4.853772x/4.644633x** total
wall, all exact. First recorded case of a host slice overtaking a qualified
CUDA route; the plan's measured-selection rule anticipated it.

**D0273/B0273 banded raster accumulation: complete and selected.**
`writers.gdal` accumulates on row bands (pinned walk per band restricted to
its rows; standard mode, fixed streaming grids, and bin mode; streaming
dynamic radius grids and percentiles stay serial to reproduce pinned
interleaving): r3 **2.077188x/2.037489x** (from 1.59x), r7 unchanged;
aggregates **2.772537x/2.727890x** geometric mean and
**5.002094x/4.851449x** total wall, all exact; a 19-case pinned-oracle GDAL
writer matrix with a test-only band trace is registered.

**D0274/B0274 concurrent KD builds: complete and selected.** Vendored
nanoflann gains a structure-identical concurrent `divideTreeConcurrent`
(allocator mutex, `std::async` left children, at most eight threads under
the shared policy; trees under 256 points serial), the cached KD3 snapshot
fills on fixed chunks: r6 **10.689432x/10.523255x**, r11
**12.892762x/11.924286x**, r4 **8.377378x/8.499474x**; aggregates
**2.900213x/2.888163x** geometric mean and **5.404502x/5.360525x** total
wall, all exact.

**D0275/B0275 cross-machine, big-cloud, and LAStools report: delivered.**
`docs/reports/b0275-cross-machine-benchmark.html` (rented Threadripper PRO
3975WX + RTX 4090: default 2.385987x GM / 3.274039x total; forced CUDA
hybrid 2.268169x / 4.076398x; 47M-point AHN4 tile r6 13.6x with CUDA, r4
10.4x; LAStools timings; renders; upstream-merge assessment). Finding: forcing
CUDA stages outside the measured selector loses on the light workflows,
which validates measured selection; the automatic selector's profile lock
means other GPUs run host-only until qualified (docs/vast-qualification.md).

**D0276/B0276 scale ladder on a rented H200: measured.** AHN4 tiles at
47M / 95M / 190M points, one exact pair per cell, all exact: the kNN-heavy
graphs plateau at ~10x (r6 10.40/10.38/10.42x, r4 9.74/10.17/9.34x), r2
climbs 1.40 → 1.65x, the I/O-bound graphs stay at 1.4–3.6x, and the H200
runs the PDG side no faster than the RTX 4090 box in absolute time. Finding:
at these sizes the device is not the bottleneck — the exact host repair,
single-file LAZ decode/encode and host-only stages are; a larger GPU buys
headroom (no tiling at 190M), not ratio. Directions that would move the
ratio: device-side exact tie/incomplete repair, more work per point, and
parallel LAZ chunk decode of a single input.

**D0277/B0277 `pdg calibrate` and D0278 file-typed resident layouts:
delivered.** Explicit local calibration writes a machine-keyed placement
profile (same fit as the embedded SM-89 record, byte-exact device wins only,
fail-closed envelopes, `--append`, `--input`, `--status`); the resident
layout follows a file's extra-bytes dimension types for dimensions no device
stage touches. Two rented boxes: default mode 2.338x → 2.510x GM (4090,
r6 2.66x → 9.05x) and 1.877x → 2.078x (5090), calibrated default above the
forced hybrid on both; AHN4 47.5M default 10.1x/14.8x features and
22.1x/25.5x LOF, all exact. Reference machine unchanged (B0274 stands).
Open follow-ups: a packing-rate probe (the one inherited coefficient),
calibrating the direct whole-view executors, and the r6 `extra_dims=all`
sink beyond the measured 1M layout.

**D0279/B0278 shipped GPU-class profiles and the generic fallback:
delivered.** Drop-in by default: `pdg calibrate` was run on rented boxes of
ten GPU classes (A10, A100, L40S, RTX 3060/3090/4060 Ti/4080/4090/5090,
RTX A6000), each local profile converted with a shipping margin (3x at
B0278, re-issued at 1.7x under D0280 the same day) into
`data/placement-profiles/*.json` (embedded at build time, keyed by GPU
name / compute capability / toolkit) plus a generic profile (intersection
under the same margin, CC ≥ 8.0, ≥ 12.24 GB); tier order embedded → local →
shipped → generic → host; the r6 `extra_dims=all` sink got its own
calibration case and model. Proof on every box with the profiles embedded
and no local profile: eight of ten classes lifted at 1M (4090 box 2.64x →
2.93x GM / 3.48x → 5.03x total; r6 2.3–2.8x → 7.3–12.0x), the 5090
(extradims 2.984x at 1M) and 3090 (weak host) stay on the host under the
margin, AHN4 47.5M features 9.8x–17.2x and the r6 graph 8.6x–12.5x on the
device, all exact; reference machine unchanged (B0274 stands). Open
follow-ups: measure the generic tier on a big cloud on an unswept GPU; a
profile refresh protocol when cases or the toolkit change; the earlier
list (packing-rate probe, direct whole-view executors).

**Next vertical slice.** Remaining warm candidate walls (B0274 suite): r6
0.97 s (preflight/CUDA init 0.18, LAZ decode/encode, repair 0.14, columns
0.13), r2 0.84 s, r4 0.78 s (CUDA outlier route + range + hashed sample
0.27 s + LAZ), r11 0.68 s, r8 0.63 s, r3 0.41 s, r10 0.28 s, r12 0.26 s, r5
0.25 s. Candidate directions: (a) the remaining serial per-point streaming
host passes (`filters.range`, `filters.assign`, `filters.crop`,
`filters.hag_*` epilogues, `writers.gdal` accumulation, `filters.stats`
dimension-parallel accumulation, `readers.copc` per-point filter/unpack)
through the same batch hooks; (b) further `--fast`-only routes now that the
contract admits device tie choices (measure first); (c) r6's CUDA
startup/preflight fixed costs; (d)
`filters.colorization`'s GDAL sampling; (e) LAZ chunk decode itself (about
0.03 s at 1M with seven workers). Cheap forced/controlled prototype first,
every time; run the full CUDA aggregate once per session.

B0043 is complete at `62016556b`. Automatic admission is now a separately
queryable, no-side-effect path for the exact explicit
`LAS -> LOF(minpts=10) -> UserData assign -> default LAS` shape. Unsupported
graphs and CLI options decline before planner work; unsupported input/output
state, device/profile/placement refusal, and failed resident preflight return
to the unchanged public selector before commitment. After commitment, errors
remain errors rather than retrying a pipeline with possible side effects. The
281.94-second physical v3 gate and host option/metadata/stream/error matrix are
exact. Clean option-free B0043 measures 3.723975123 seconds versus
35.738760555 seconds pinned PDAL, or 9.596939x, with no selection environment
variable.

B0044 closes that cheap direct-endpoint sweep on clean `e53f5ab43`. All three
4M shapes are exact and positive: coplanar is 4.932x pinned PDAL, normal is
4.874x, and nndistance is 5.275x. Per the predeclared rule, only nndistance
advances; coplanar and normal remain measured positive hypotheses rather than
automatic envelopes. Nndistance has the lowest candidate wall at 3.363
seconds. Its profile names the remaining surface: `knnGatherKernel` is only
138.57 milliseconds and the measured host upload/spill phases total 0.125
seconds, so roughly 3.10 seconds remains in host-side reader/control/index
preparation and canonical publication rather than a kernel bottleneck.

B0045 is complete. Clean explicit-direct qualification at `d69e37da0` measures
3.339251347 seconds versus 17.705276847 seconds pinned PDAL, or 5.302170x.
Commit `fba6b16ba` adds only the exact explicit
`LAS -> nndistance(k=10) -> UserData assign -> default LAS` shape to B0043's
no-side-effect selector. Its host fallback/option matrix is exact, the expanded
21.97M v2 physical gate passes in 236.60 seconds, and five clean option-free
samples measure 3.390338813 seconds versus 17.726674770 seconds pinned PDAL,
or 5.228585x. The public handoff adds no material penalty.

B0046 is complete at `455bced46`. Its additive stats-only partition is exact
on clean 4M nndistance and LOF direct runs. Nndistance spends 3.005455534 of
3.235190260 seconds (92.90%) in rewritten manager execution, while validation,
canonical publication, and other control take 0.168271752, 0.051060884, and
0.010402090 seconds. After B0044's 0.123230 seconds of boundary work and
138.57-millisecond dominant device kernel, 2.743655928 seconds remain inside
the reader/table/index/resident-wrapper aggregate. LOF likewise spends
3.503516383 of 3.735417920 seconds in the manager, but 2.287512704 seconds is
the already-known exact host repair. Its publication is only 0.051880115
seconds. Both outputs are byte-identical to pinned PDAL.

B0047 is complete at `6cfedcc29`. A stats-enabled single-region manager
timeline now partitions graph/prepare, reader/row-table materialization, the
accepted upload-through-spill resident wrapper/index/filter interval, and
post-spill control without overlap. Clean exact 4M nndistance measures
0.000559325/0.502592322/2.524453274/0.000001730 seconds respectively. The
resident interval is 77.43% of internal wall and 83.38% of manager wall;
after known boundary phases and B0044's kernel it retains a 2.256203803-second
unknown. LOF measures 0.000547915/0.509743845/2.901305423/0.000002499 seconds;
2.294764896 seconds (79.09% of its resident interval) is already-known exact
host repair. Reader/table work is material but not dominant in either path.

B0048 is complete at `3af936e2e`. Its stats-only host-call spans split the
resident interval without adding a synchronization. Clean exact 4M
nndistance measures 0.121900349 seconds index configuration, 0.008554448
seconds index-build call wall, 2.233622069 seconds neighborhood
query/projection, 0.035821079 seconds adjacent assignment bridge, and
0.106845283 seconds residual inside the 2.506743228-second resident interval.
The broad query span is 89.10% of resident wall. Its original zero-repair
report was an instrumentation omission: NND did not yet publish its incomplete
repair counter or timer. B0049/D0112 supersede that inference. LOF shows the
same small index/bridge costs; 2.279198769 seconds of its
2.597038882-second broad query span is already-required exact repair. The
index, bridge, reader, and publisher are rejected as next targets.

B0049 is complete at `20d9ffabc`. Its nested stats-only call spans show that
the final exact 4M NND broad query spends 2.061544515 of 2.253729404 seconds in
status scan plus repair. One incomplete row triggers a 2.061175139-second
exact host KD3 build/recompute/device refresh. The pageable status-transfer
call is only 0.138355024 seconds, so the proposed pinned-status optimization is
rejected. A cheap forced planner-owned Morton-BVH prototype eliminates repair
but slows complete candidate wall from 3.278321613 to 4.179201213 seconds and
is also rejected. Output remains byte-identical to pinned PDAL. Functional,
GPU-native, performance-qualified, and automatic-selection categories are
unchanged.

B0050 is complete at `abb460846`. A rejected serial full-scan prototype was
exact but slower than B0049; the accepted bounded repair assigns one 64-thread
block to each incomplete row, scans the already-resident coordinates, and
exactly merges the per-thread top-k sets without constructing an index. It is
selected only for kth mode with public `k=1..15` and at most 16 incomplete
rows; every other case retains upstream KD3 repair. Five clean, proof-gated
automatic-path samples measure 1.413364609 seconds versus 17.809985276 seconds
pinned PDAL, or 12.601126x, with the established output bytes. This is a
2.398772x candidate-wall improvement over B0045's 3.390338813-second automatic
median. The 21.97M physical v2 gate and all four focused CUDA sanitizers pass.

B0051 is complete. A disposable exact 4M probe mapped and validated the LAS,
reused the existing coordinate transpose, expanded all 12,000,000 coordinate
values into resident binary64 columns, and measured a 0.00840182-second median
after one warmup. Fresh allocator plus device allocation cost 0.0986952 seconds
separately. Every decoded value matched the host coordinate decoder. This is a
directional prototype, not an end-to-end claim, but it clears the
0.582367542-second reader/table-plus-upload ceiling with substantial margin.

B0052 is complete at `8c8ccb523`. The bounded source reuses the established LAS
mapping, validation, coordinate transpose, planner-owned resident batch/index,
and exact output overlay; its internal execution reader adds no public reader
or stage-coverage envelope.
Five clean proof-gated automatic-path samples measure 0.911029970 seconds
versus 17.815840889 seconds pinned PDAL, or 19.555713x, with identical output.
This removes 35.541759% of B0050 candidate wall. The unchanged PointView path
remains the exact fallback, all four focused CUDA sanitizers are clean, and the
hash-pinned 21.97M physical v2 gate passes in 232.49 seconds. The existing
default-LAS NND envelope therefore selects the direct source automatically;
functional support, GPU-native coverage, performance qualification, and
automatic selection remain scoped independently and no new stage is counted.

B0053 is complete. The final exact Nsight Compute capture attributes
218.520992 milliseconds across 21 launches: 139.479840 milliseconds in the
reusable `knnGatherKernel`, 77.548288 milliseconds in the one-row bounded
repair, and only 1.492864 milliseconds in every other launch combined. A
64/128/256-thread gather sweep stays exact but reduces the dominant kernel by
only 4.946912 milliseconds at best; 512 threads is not launchable. The
complete-process samples show no stable material win, so launch-size tuning is
rejected and the qualified 64-thread build is restored.

B0054 is complete at `8aaac57ff`. Additive host-only spans partition a fresh
0.175984958-second validation/placement/preflight interval into 0.002231646
seconds of plan/original validation, 0.173185798 seconds of runtime placement,
and 0.000567514 seconds of rewrite/resident preflight. Runtime placement is
0.078605269 seconds of CUDA device/profile discovery, 0.094578699 seconds of
the initial memory-budget placement, and 0.000001830 seconds of executor
selection. The apparent validation target is therefore CUDA/process startup,
not redundant PDAL validation. Deferring it would move the same initialization
into the first device allocation, so no production prototype or performance
claim follows. Compatibility, placement, selection, and all four coverage
categories are unchanged.

B0055 is complete at `aca65d75e`. The exact 4M partition attributes
0.051803244 seconds to adaptive kNN configuration and 0.072138563 seconds to a
second full host-envelope scan. The kNN builders already derive and validate
their returned grid/BVH envelope from every coordinate, so the resident and
standalone outlier callers now treat that proof as authoritative; radius
indexes retain their independent check. A clean final capture reduces index
configuration from 0.123943417 to 0.051821472 seconds. Five clean alternating
samples measure 0.821465859 seconds versus 17.895370391 seconds pinned PDAL,
or 21.784679x, with exact output. This is a further 9.831083% complete-process
reduction from B0052. Functional support, GPU-native coverage, and automatic
selection are unchanged; the existing endpoint's performance qualification
improves.

B0056 is complete as a rejected disposable prototype. Reusing the first
finite-extrema scan produced bit-identical grid/BVH frames and reduced the
exact stats capture's configuration interval from 0.051821472 to 0.026006785
seconds, but option-free exact samples were 0.834/0.825/0.844 seconds. They do
not improve materially on B0055's 0.821465859-second qualified median, so the
two-file prototype was fully reverted and the B0055 binaries rebuilt. No
certification lane, performance claim, coverage change, or selection change
follows.

B0057 is complete at `ff3a55044`. Additive stats-only spans reduce the fresh
0.083464418-second wrapper residual to 0.023428953 seconds: direct-LAS
coordinate hydration owns 0.058526544 seconds and resident product setup only
0.000147437 seconds. The exact output and qualified implementation are
unchanged. This confirms a material hydration surface, but does not yet
distinguish fresh column/device allocation from H2D/decode/expand, the 96-MiB
XYZ mirror calls, or their final synchronization.

B0058 is complete at `0a943a359`. On the exact 4M source, the fresh
0.055183624-second hydration span contains 0.042686939 seconds of
validation/materialization/allocation, 0.008537849 seconds of transfer/kernel
submission, and only 0.003948276 seconds in the final stream wait. D2H-only
elision therefore has too small a complete-pipeline ceiling. Compatibility,
qualification, selection, and all coverage categories are unchanged.

B0059 is complete as a successful directional prototype. The disposable
mapped-record summary reproduced the established grid frame and adaptive
backend bit-for-bit from point records at both 4M and 16M points. After one
warmup, five scans take 0.00426759 and 0.0148655 seconds median respectively,
without allocating or mirroring host XYZ. This clears the viability gate
against the measured combined hydration/configuration surface, but is not an
end-to-end performance claim. Compatibility, qualification, selection, and
all four coverage categories are unchanged.

B0060 is complete at `07ea2f4a7`. The bounded direct source now derives the
planner-owned kNN frame/backend from exact mapped-record extrema and the same
8,192 adaptive samples, retains device XYZ, and allocates no resident host XYZ
mirror. The clean exact 4M automatic median is 0.755523193 seconds versus
17.891433883 seconds pinned PDAL (23.680853x), 8.02744% below B0055 candidate
wall. The established output is byte-identical, adaptive uniform/clustered
Morton-BVH and both forced backends pass, mapped PointView/KD3 host repair is
proved, all four CUDA sanitizers are clean, and the hash-pinned 16M process
lane passes. Functional support and GPU-native coverage are unchanged; the
existing endpoint's performance qualification improves and its already-bounded
automatic selection does not widen.

B0061 is rejected and cleanly reverted at `0b61e8ef2`. Commit `cf5ff3344`
proved the exact sole-consumer envelope, skipped the host PointView write, and
positively observed the adjacent assignment consuming device NNDistance. The
clean five-sample automatic median was 0.750254638 seconds versus B0060's
0.755523193 seconds, only 0.005268555 seconds (0.697339%) lower with overlapping
samples. That is not a material complete-process win and removes no transfer,
so no production change, qualification, selection, or coverage change remains.

B0062 is accepted at prototype commit `23cc26827` and promoted by D0125. In the
same exact direct-LAS NND-to-assignment endpoint, NNDistance is now allocated
and retained only on device unless pinned KD3 host repair is actually needed.
The clean five-sample median is 0.712510475 seconds versus 17.777145188 seconds
pinned PDAL (24.950012x), 0.043012718 seconds (5.693104%) below B0060. Normal
execution removes one 32,000,000-byte H2D and 32,000,008 bytes of D2H while the
forced host-repair proof restores the complete 32,000,000-byte D2H/H2D pair,
publishes normally, and produces the same exact LAS hash. All four CUDA
sanitizers and the hash-pinned 16M process lane pass. Functional and GPU-native
coverage are unchanged; the existing endpoint's performance qualification
improves and only its already-bounded automatic envelope changes internally.
The retained validation attestation is
`build/benchmarks/b0062-validation-manifest.json` at SHA-256
`727d5f46bc39ff38a6bfe4fd334802067ad4c7bad349913124d52421fb4e757a`.

B0063 is complete as profile attribution. The accepted B0062 endpoint records
216.430208 milliseconds across 21 kernels: 137.153696 milliseconds in the kNN
gather, 77.754496 milliseconds in the one-row exact repair, and 1.522016
milliseconds in every other launch. The gather already accounts for the full
magnitude of the earlier 0.134898783-second status-call interval, so pinned
status storage is rejected without a prototype. Coverage, qualification, and
selection are unchanged.

B0064 is complete at prototype commit `60f4af821` and promoted at
`1aebd1877`. The deterministic multi-block reduction lowers the measured
one-row repair from 77.754496 to 2.232480 milliseconds while preserving the
same candidate operation/order and exact kth-distance result. Five clean
alternating samples measure 0.623028204 seconds versus 17.950882850 seconds
pinned PDAL (28.812312x), 12.558731% below B0062 candidate wall. A same-binary
serial/parallel control measures 0.652247040 versus 0.591965696 seconds, a
9.242103% reduction. Default selection is confined to B0062's exact
direct-LAS/device-only NND-to-assignment endpoint; generic resident callers
retain the serial repair. All four CUDA sanitizers and the hash-pinned 16M v2
lane pass.

The fresh promoted profile records 141.519872 milliseconds across 22 launches:
137.767712 milliseconds (97.3487%) in the reusable kNN gather, 2.232480
milliseconds (1.5775%) in repair, and 1.519680 milliseconds elsewhere. B0053
already found no stable complete-process gain from bounded gather launch-shape
tuning. With no clear reusable path likely to remove another 5--10% without a
new algorithm, this endpoint is sufficiently optimized for now.

B0065 is complete as a directional 1M prefix gate. Stock PDAL statistical
outlier measures 4.001562112 seconds median, the existing forced-exact CUDA
lane measures 0.620354816 seconds (6.450441x), and both produce the same
36,000,375-byte LAS. The read/write-only median is 0.278837248 seconds, so
93.031790% of stock wall lies above I/O. Hardware-cycle samples put 3.169 of
4.991 exclusive CPU-cycle seconds in nanoflann `searchLevel`; the collector's
interval warning keeps this directional rather than certification evidence.
Nsight Compute measures only 21.557088 milliseconds of CUDA kernel work:
21.257536 milliseconds in the kNN gather and 0.299552 milliseconds in index
construction plus every other launch. The remaining 0.341517568 seconds above
the I/O floor contains CUDA/process startup plus the PointView
gather/configuration, coordinate upload, result/status return, exact host
classification, and bridge setup. This is a measured residency opportunity,
not a new performance qualification; `filters.outlier` remains explicitly
unmeasured in the accepted complete-pipeline category.

B0066's disposable prototype gate is positive. A planner-resident statistical
outlier now consumes the existing shared kNN product, keeps the pinned host
global recurrence/classification and incomplete-row repair, and leaves radius
outlier outside this kNN adapter. The focused physical test requires reuse and
observes one index build across `outlier -> nndistance`. The dirty 1M complete
composition is byte-identical and measures 0.712635520 seconds median versus
8.089364299 seconds pinned PDAL (11.351335x). A same-binary five-pair control
measures 0.666293053 seconds resident versus 0.686652044 seconds through two
private CUDA stages, a 2.964965% complete-process reduction. That is modest but
positive, and it removes one 24 MB XYZ upload plus one index build while
creating the reusable composition path this phase prioritizes. No accepted
performance qualification or automatic selection follows from the dirty
prototype.

B0067's fresh profile closes kernel work on the improved resident endpoint.
Its 18 launches total 51.702304 milliseconds: the outlier and NNDistance kNN
gathers take 22.906240 and 28.500256 milliseconds, while every other kernel
takes 0.295808 milliseconds. All device work is only 7.759694% of B0066's
0.666293053-second reference wall; perfect reuse of the smaller gather has a
3.437863% process ceiling. No fused query or launch sweep meets the requested
5--10% gate.

B0068 clears the remaining reusable-boundary gate. Seven alternating exact 1M
samples measure 0.373325396 seconds with direct LAS source plus canonical
Classification overlay versus 0.646892319 seconds for the same B0066 resident
composition through ordinary LAS boundaries, a 42.289407% wall reduction.
The focused physical process gate proves one index build, record-summary use,
no host XYZ mirror, canonical byte/flag preservation across adversarial legacy
and format 6/7/8 inputs, the separate raw-record hydration transfer, and
fail-closed disabled/outside-envelope behavior. It also requires the forced
executor to report uncalibrated/mismatched boundary accounting rather than
impersonating the generic PointView model. The lane stays explicitly opt-in:
its dirty same-binary evidence is not performance qualification and does not
change automatic placement.

The outlier endpoint is sufficiently optimized for now: B0067 rejects another
kernel or query-fusion slice, while B0068 retains the one material existing
boundary optimization. B0069 then resumes the measured catalog with B0044's
exact `approximatecoplanar(knn=8) -> Coplanar=>UserData` endpoint. Its clean
fresh 1M gate is byte-identical at 4.293 seconds pinned PDAL and 1.018 seconds
resident/direct-output. The 20 device launches total only 21.592970
milliseconds, while the exact ambiguity path moves a 1 MB status mask and all
96 MB of eigensystems in both directions before selected-row KD3 repair. This
rejects another coplanar kernel and defers direct-LAS source work while repair
still needs the host PointView.

B0070 rejects that selective-transfer hypothesis. The exact 1M fixture has
2,145 tied rows; transferring only those rows reduces the repair movement from
194,000,000 to 1,208,065 bytes, yet shell wall improves 0.491159%, command-
before-stats 2.267040%, and the resident wrapper 3.357725%. The remaining
0.449844793-second repair is pinned KD3 construction plus ordered row
recomputation. The prototype is reverted and the approximate-coplanar endpoint
is sufficiently optimized for now.

B0071 confirms a large reusable composition win. The exact public-shaped 1M
pipeline takes 14.591 seconds pinned PDAL and 1.226 seconds when forced through
one shared index/eigensystem region, or 11.901305x. Ordinary resident placement
falls back with `mixed_calibration_models` and takes 14.908 seconds. The fresh
profile records only 41.533930 milliseconds across 22 kernels; all three
projections plus the point program total 0.372130 milliseconds. Kernel fusion
is rejected. The missing value is a truthful composition model that lets the
planner select the already-fast executor.

B0072 completes the exact 50K-through-16M forced-composition ladder. Every row
is positive, from 2.155256x to 14.569879x pinned PDAL, and requires both shared
index and 13-neighbor eigensystem reuse. The separately named fit over
250K-through-16M is `host=14,812.948623 ns/point` versus
`device=185.808308 ms + 1,005.425583 ns/point`; no existing model changes.

B0073 integrates only the measured exact normal/eigenvalues/
covariancefeatures/three-assignment region as `eigen-family-compose`; eleven
option/content/topology negatives retain `mixed_calibration_models`. The
hash-pinned physical gate proves explicit resident selection, accepted
preflight, one index/eigensystem, exact warning diagnostics, and exact LAS
bytes. Five alternating 1M samples measure 14.507773633 seconds pinned PDAL
versus 1.397204802 seconds explicit resident, or **10.383427x**. This exact 1M
pipeline is performance-qualified but not automatically selected.

B0074 replaces B0072's forced-hybrid calibration rows with a clean
explicit-resident ladder. The 50K below-floor control is exact and host-
selected; every 250K-through-16M row selects the shared-index resident
executor, remains exact, and wins by 6.406690x through 11.915007x. The
resident-only fit is `host = 120.004601 ms + 14,659.538987 ns/point` versus
`device = 155.357778 ms + 1,234.745145 ns/point`. Both calibration
representations and all seven raw-report pins now use resident provenance, so
the exact model may report
`selected_device_calibration_matches_executor=true`.

B0075 completes option-free public admission for only the exact B0073/B0074
shape. Five clean alternating 1M samples remain byte- and diagnostic-exact at
14.452746005 seconds pinned PDAL versus 1.409086026 seconds automatic PDG, or
**10.256823x**. Required and genuinely option-free physical paths select the
same resident executor; changed options/programs/topology, below-floor
placement, and injected preflight failure decline before output. Normal,
eigenvalues, and covariancefeatures are automatically selected only inside
this measured composition. Direct LAS output remains separate.

B0076 confirms the exact ordinary-data rank/optimal composition is valuable:
11.303210336 seconds pinned PDAL versus 1.246256090 seconds forced shared
resident, or 9.069733x. An exact identity-coordinate boundary produces the
same LAS bytes but forces two index regions and takes 1.697469160 seconds, so
one compatible region removes 26.581518% of candidate wall. The profile's two
kNN gathers dominate 88.786784 milliseconds of device work, but deleting one
perfectly has only a 3.557802% process ceiling. Kernel work is rejected; the
missing value is truthful compatible-region placement.

B0077 completes the exact forced-composition ladder. Every 50K-through-16M row
requires shared-index reuse, remains byte- and diagnostic-exact, and wins by
1.542997x through 10.755371x. The conservative 250K-through-16M fit is
`host = 117.903408 ms + 11,435.542525 ns/point` versus
`device = 233.890937 ms + 1,050.229585 ns/point`; neither per-stage model
changes. The 50K row is positive below-fit validation, not a lower floor.

B0078 integrates only the exact compiled B0076 shape as
`rank-optimal-compose`. Changed options/program/topology and coordinate
invalidation stay negative; the source/JSON model audit and all 132 raw pins
pass. The physical gate proves one selected region/index, accepted preflight,
reuse, exact diagnostics, and exact LAS. Five alternating 1M samples measure
11.353348769 seconds pinned PDAL versus 1.370295007 seconds explicit resident,
or **8.285332x**. This qualifies the exact explicit composition only; its
executor-provenance flag remains truthfully false.

B0079 replaces the forced cases with a clean explicit-resident ladder. The 50K
control is exact and host-selected; every 250K-through-16M row requires the
shared-index executor/reuse, remains exact, and wins by 4.873850x through
9.566431x. The resident-only fit is
`host = 133.650155 ms + 11,448.442352 ns/point` versus
`device = 240.029160 ms + 1,186.527542 ns/point`. Both representations and all
seven raw pins now use resident provenance, and the physical executor-match
diagnostic is truthfully true.

B0080 completes option-free public admission for only the exact B0078/B0079
shape. Required and genuinely option-free physical paths are byte- and
diagnostic-exact; changed options/assignment order/topology, below-floor
placement, and injected preflight failure decline before output. Five clean
alternating 1M samples measure 11.324354873 seconds pinned PDAL versus
1.388344338 seconds automatic PDG, or **8.156734x**. EstimateRank and
OptimalNeighborhood are automatically selected only inside this measured
composition. B0076's unchanged-executor profile limits perfect removal of one
repeated gather to 3.557802% of complete wall, so this endpoint is sufficiently
optimized for now. Direct LAS output for this rank/optimal composition, broader
rank/optimal shapes, HAG count five, and new stage ports remain deferred while
the catalog resumes through measured fusion/residency/shared-product
candidates.

B0081 revalidates the already-implemented exact radiusassign resident endpoint
on the clean current tree. The zero-warmup 1M pair is byte- and diagnostic-
exact at 1.921081414 seconds pinned PDAL versus 0.697987348 seconds resident,
or 2.752316x, with shared-index execution, accepted preflight, and
executor provenance observed. The single consumer proves one planner-owned
index build, not cross-stage neighborhood reuse; the reuse requirement in the
directional runner was a no-op. The fresh profile records only 1.899456
milliseconds across all kernels; perfect query removal is 0.234172% of process
wall. The dominant named interval is 0.390594913 seconds of rewritten manager
execution, while the separately dispositioned placement/setup interval is
0.178538164 seconds. B0081 remains directional and does not promote B0019's
dirty ladder or alter selection.

B0082 retains that exact boundary prototype after seven alternating same-binary
1M samples. Ordinary resident median is 0.641774202 seconds and direct resident
median is 0.316225432 seconds, a **2.029483x** same-binary boundary ratio and
50.726372% complete-wall reduction. Pinned-oracle physical gates over LAS
formats 7 and 3 prove the
ReturnNumber masks, positive UserData publication, exact bytes/diagnostics, one
shared index, record-summary configuration, no host XYZ mirror, and fail-closed
option/source rejection. All four focused Compute Sanitizer tools are clean.
This is directional prototype evidence only: boundary accounting and executor-
calibration provenance truthfully remain false, so there is no performance-
qualification or automatic-selection change.

B0083 completes the clean direct-executor ladder. The 50K below-floor control
is exact and host-selected at 0.358367x. Every 250K-through-16M row requires
`planner_resident_shared_index_direct_las`, source/summary/no-host-XYZ proofs,
accepted preflight, and exact bytes; all win from 1.523498x through 22.408647x.
The separately named bounded absolute-wall fit is
`host = 0 + 1,985.118056 ns/point` versus
`device = 305.407938 ms + 68.335971 ns/point`. These are one-sample calibration
rows, not repeated performance qualification, and no runtime model changes in
B0083.

B0084 integrates `radiusassign-direct` without changing the ordinary
`radiusassign` model or any of its raw rows. The C++ and JSON residuals subtract
only planner-owned infrastructure terms and reconstruct B0083's absolute wall
curve exactly. Runtime matching additionally requires the strict serialized
shape, explicit direct source/output/summary/no-host-XYZ proof, and the measured
36-byte LAS record. Executor-declared zero-pack boundary facts now match both
format-7 and format-3 physical crossings; only the measured format-7 row claims
executor-calibration provenance. Exact-shape, option/topology/layout negatives,
profile parity, and the physical two-format gate pass. Automatic admission and
performance qualification remain unchanged.

B0085 qualifies the explicit direct-radiusassign endpoint on a clean tree.
Five alternating 1M samples remain byte- and diagnostic-exact at
1.896917103 seconds pinned PDAL versus 0.339432621 seconds direct resident, or
**5.588494x**, while every candidate run proves `radiusassign-direct`, direct
source/output/record-summary use, no host XYZ mirror, accepted preflight, and
matching executor/boundary provenance. The fresh exact profile records only
1.953650 milliseconds across all 18 kernels, or 0.575563% of median candidate
wall; the 1.630000-millisecond radius query has a 0.480213% process ceiling.
The larger placement interval is the necessary CUDA/process startup already
dispositioned by B0054/D0117, and no other named reusable phase reaches the
requested 5--10% bar. The explicit endpoint is therefore performance-qualified
and sufficiently optimized for now. Automatic admission remains off because
B0085 did not benchmark the option-free selector.

B0086 cleanly revalidates the existing exact
`outlier(statistical) -> nndistance` one-index composition through its direct
Classification boundary. The current-tree 1M routing pair is byte- and
diagnostic-exact at 8.011663864 seconds pinned PDAL versus 0.420340609 seconds
direct resident, a directional **19.059933x**. The candidate positively proves
one shared index and every direct source/output/summary/no-host-XYZ condition;
executor calibration and boundary accounting remain false because this is
still the explicit uncalibrated B0068 lane. The fresh exact profile records
52.001290 milliseconds across 20 launches. Its two `knnGatherKernel` calls take
23.260000 and 28.390000 milliseconds; after the direct boundary lowers process
wall, eliminating the smaller repeated gather has a 5.533608% ceiling. The
remaining 0.351290 milliseconds of device work is immaterial.

B0087 is complete at `21836fa4d`. One planner-owned ordered max-k rowset now
serves only an adjacent, single-consumer statistical-outlier -> NNDistance pair
inside an already-selected compatible resident region. The query product has
an explicit width and `12*k + 1` bytes-per-point budget, checked allocation,
fail-closed proof, bridge/branch/dimension invalidation, exact per-consumer
projection, conservative incomplete repair, and immediate release after the
declared consumer. The cheap five-sample same-binary gate lowers median wall
from 0.383653 to 0.340241 seconds (11.315433%) with exact output. This is
direction evidence, not performance qualification.

The clean exact profile records one gather and 29.350784 milliseconds of
kernel work, 43.557585% below B0086's two-gather profile. The remaining gather
is 97.745668% of device work; both projections total 0.311200 milliseconds and
no endpoint-specific reusable operation remains near 5--10%. This endpoint is
sufficiently optimized for now. B0087 does not change automatic admission,
calibration, or any four-category coverage result.

B0088 qualifies the explicit improved composition on clean commit `22f5218da`.
One warmup plus five alternating complete-process pairs are byte- and
diagnostic-exact; pinned PDAL's 7.961676747-second median compares with
0.372223208 seconds direct resident, or 21.389523x. Every candidate proves the
direct source/output/record-summary boundary, no host XYZ mirror, one predicted
and observed index, and planner-owned max-k reuse. The dispatcher and resolved
engine hashes match B0087's exact-output-bound stopping profile. The named
route is now performance-qualified but remains uncalibrated and not
automatically selected.

B0089 measures the clean 10K-through-4M ladder. The route loses at 10K/25K,
wins from 50K onward, and reaches 35.036417x at 2M, but a single incomplete
statistical-outlier mean row creates a 2.754445-second pinned-KD3 host cliff at
4M. B0090 removes that cliff with a bounded exact parallel repair from the
planner-owned resident coordinates and requalifies the 4M route at 44.072341x.
Its fresh profile then finds the older serial NNDistance repair consuming
77.181792 milliseconds, 10.451157% of process wall. B0091 makes the existing
bounded exact NNDistance parallel repair the default and improves the clean 4M
median another 8.882913%, to 48.758796x pinned PDAL.

The B0091 final profile leaves 95.545682% of kernel work in the one shared
gather already explored by B0053. All non-gather kernels combined have only a
0.969583% process-wall ceiling, so the endpoint is sufficiently optimized for
now.

B0092 completes the repaired 8M/16M ladder and fits the separately named
`outlier-nndistance-direct-compose` model without changing either standalone
stage model. The 25K host control, 50K device control, exact stage/options/
topology matcher, one-index/one-lane/133-byte query-product pins, and 50K--16M
envelope fail closed. The audit is 142/142 over 33 models and all 148 raw
reports verify. Five clean option-free 1M pairs are byte- and diagnostic-exact
at 7.971261964 seconds pinned PDAL versus 0.392659506 seconds automatic PDG,
or **20.300698x**. Automatic execution observes direct source/summary,
no-host-XYZ, max-k reuse, and one index before publication. The composition is
therefore automatically selected only inside this exact measured envelope;
standalone outlier and broader NNDistance shapes inherit no claim.

B0093 promotes the already-calibrated `radiusassign-direct` endpoint without
changing its kernel, coefficients, or execution boundary. The public selector
requires the exact B0085 stage/options/topology, a 36-byte LAS source, the
250K--16M calibrated cardinality envelope, one device-selected region, and the
named direct model. Automatic publication additionally observes direct source
use, record-summary index configuration, no full host XYZ mirror, and exactly
one index build. The focused physical gate proves a 50K below-floor control,
250K and 1M device controls, layout/option/source/preflight failures, and a
post-execution proof failure with no artifact. Five clean option-free 1M pairs
remain byte- and diagnostic-exact at 1.892418782 seconds pinned PDAL versus
0.352668079 seconds automatic PDG, or **5.366005x**. B0085's unchanged-
executor profile still closes endpoint tuning: all kernels have only a
0.575563% complete-wall ceiling and the radius query itself only 0.480213%.

B0094 repeats the exact existing
`approximatecoplanar(knn=8) -> ferry(Coplanar=>UserData)` resident/direct-output
composition on the clean B0093 checkpoint. Five alternating 1M pairs are byte-
and diagnostic-exact at 4.108373629 seconds pinned PDAL versus 0.990323169
seconds explicit resident PDG, or **4.148518x**. Every candidate proves device
placement, the direct output executor, and one predicted/observed shared index.
It also honestly reports executor-calibration and boundary-accounting mismatch:
the current `approximatecoplanar` model belongs to the ordinary resident
executor, not this direct-output composition. Fresh event telemetry reproduces
the 194 MB ambiguity/eigensystem round trip, but generic repair counters do not
name approximate-coplanar rows or KD3 use. B0069's older profile has different
engine/pipeline/output hashes and is historical routing evidence only; it
cannot close the current profile or automatic-admission gate.

The revised measured priority remains: (1) promote already-calibrated and
explicitly qualified exact compositions through cheap option-free gates;
(2) profile positive forced/resident lanes before any additional optimizer;
(3) prefer shared products and fused endpoints when the profile names a
material boundary; (4) retain measured-negative terrain/global stages on host;
then (5) resume catalog ports when coverage, rather than assumed speed, is the
explicit value. This order is based on
B0043/B0045/B0075/B0080/B0092/B0093/B0097 automatic wins and the
B0037/B0053/B0061/B0070 rejected hypotheses.

B0095 completes that proof on clean implementation commit `f69fe3204`. The
repair-triggering unit remains exact and now positively observes its trigger,
ambiguous/incomplete/repaired rows, KD3 use, and symmetric transfer bytes. The
real 1M B0094 fixture records 2,145 ambiguous rows, zero incomplete rows, one
KD3 use, and 97 MB in each transfer direction; the output remains byte-
identical at SHA-256 `0c52732a847e4427955a1b22525d58d3b4638d7fa35e2afebde7e437786b3245`.
The fresh output-bound profile records 21.403776 milliseconds across 20
kernels, only 2.460417% of the 0.869924635-second unprofiled command interval.
Exact host KD3 repair is the limiter at 0.461688645 seconds, but B0070 already
removed 99.377286% of its transfer for only a 0.491159% shell-wall reduction;
no exact-proved reusable replacement clears the roughly 5--10% gate. The
endpoint is sufficiently optimized for now.

B0096 completes that current-binary paired ladder. The 50K control is exact and
host-selected at 0.484603x; the direct-output composition wins every selected
250K--16M row from 2.476664x through 5.048595x, including 4.248119x at 1M.
The isolated `approximatecoplanar-direct-compose` residual is fitted only to
those six positive rows, bounded to their measured cardinalities, and admitted
only for the exact default-threshold `knn=8 -> Coplanar=>UserData` topology,
format-7 layout, one region/index, measured planner bytes, and explicitly
required direct output. The ordinary resident model and automatic selector are
unchanged. The current audit is 149/149 over 34 models and all 155 raw reports
verify.

B0097 completes that cheap public gate on clean implementation commit
`e36ac6bbd`. Required and genuinely option-free 1M commands are exact; 50K,
option, topology, both thresholds, format/compression, device, and preflight
controls either fall back byte-for-byte or fail closed without publication.
Five current automatic pairs retain 4.230620x pinned-PDAL value. A fresh
output-bound automatic profile records 21.257320 milliseconds across 20
kernels, only 2.166754% of complete automatic wall, so B0095's stopping
decision remains valid and no endpoint optimization reopens. The measured
250K--16M format-7/default-threshold composition is now automatically selected;
host KD3 repair remains explicitly non-native.

B0098 measures the already-native `filters.neighborclassifier(k=7)` lane on the
current binary before changing its kernel. The ordinary resident route is exact
at 3.400787x pinned PDAL. Its fresh profile puts all 18 kernels at only
16.184340 milliseconds, so kernel work is not the limiter. A bounded reuse of
the existing mapped-LAS/Classification-overlay boundary then clears the cheap
gate, and clean commit `d59d3d151` is exact at 3.959949x pinned PDAL. The
direct route is performance-qualified only for this explicit 1M format-7
shape; it remains deliberately uncalibrated, opt-in, and not automatically
selected.

B0099 adds production attribution for neighborclassifier's exact repair. The
clean 1M route has 1,947 distance-tie rows, zero incomplete rows, one pinned
KD3 use, and 0.594524233 seconds in that host bridge. A fully reverted exact
prototype proved 1,946 rows vote-invariant but left one non-invariant boundary
row; that one row still required the whole KD3 construction and left repair at
0.579699672 seconds. Because complete wall did not clear the roughly 5--10%
gate, the prototype is rejected and the endpoint is sufficiently optimized for
now. B0098's explicit qualification remains unchanged and non-automatic.

B0100 cleanly performance-qualifies the already-native
`filters.radialdensity(radius=1.01)` 1M complete pipeline at 4.720283x pinned
PDAL. Its output-bound profile records only 2.973280 milliseconds across all 16
kernels, 0.449921% of the 0.660844528-second candidate median. Device query,
index-build, or launch work cannot clear the roughly 5--10% gate; the inferred
99.550079% host LAS/PointView/extra-dimension boundary is the only material
surface. The stage remains explicitly forced and not automatically selected.

B0101 retains that reusable boundary on clean commit `29451ec13`. The exact
1M VLR-free format-7 `radialdensity(radius=1.01) -> assign` composition measures
3.097940201 seconds pinned PDAL, 0.611763765 seconds through the prior exact
hybrid path, and 0.345532286 seconds through one resident radius region plus
direct LAS publication. The retained path is 8.965704x pinned PDAL and removes
43.518674% from the same-binary hybrid control. Its output-bound profile records
3.057664 milliseconds across 19 kernels, only 0.884914% of complete resident
wall. The remaining control interval is CUDA driver/context initialization and
actual-free-memory placement input; deferring it moves required one-shot work
to first allocation rather than removing a clear reusable 5--10%. This endpoint
is sufficiently optimized for now. The lane remains explicit, uncalibrated,
and not automatically selected.

B0102 physically qualifies the existing exact ordinal CUDA envelope without
promoting its performance. Eight focused device/process gates and all four
Compute Sanitizer tools pass. Directional exact `decimation(step=2) -> assign`
rows are only 0.449898x pinned PDAL at 1M and 0.762657x at 4M. The 1M profile
records 241.376 milliseconds across 64 decode/mask/CUB-select/assign/repack
launches; a reversible one-chunk control is also slower. The bounded ordinal
programs are therefore GPU-native force-only coverage, not
performance-qualified or automatically selected, and optimization is deferred.

B0103 physically qualifies the existing exact first-tie `filters.locate` CUDA
envelope: five focused device/process gates and all four Compute Sanitizer
tools pass. The measured `assign(UserData=1) -> locate(Z,min)` composition is
only 0.387892x pinned PDAL at 1M and 0.706034x at 4M. Its 1M profile records
54.272 milliseconds across ten assignment/reduction launches inside 0.432344452
seconds candidate wall; the roughly 378-millisecond non-kernel span is the
hybrid/PointView boundary. A B0101-sized direct-boundary removal would bring
the 4M row only near break-even, not prove a 5--10% win, and the required
pushdown/direct executor is not a cheap prototype. Locate is therefore
GPU-native force-only coverage, not performance-qualified or selected; the
resident-terminal hypothesis is deferred until a stronger measured consumer.

B0104 physically qualifies the exact option-free `filters.info` bounds/count
CUDA envelope: its 131,103-point device property, a forced process differential
including metadata, and all four Compute Sanitizer tools pass. Current-binary
directional rows are only 0.525893x pinned PDAL at 1M and 0.744854x at 4M.
The 1M profile records just 0.26863 milliseconds across 16 bounds kernels,
0.047591% of candidate wall. A no-code control measures exact PDG-host info at
0.313470297 seconds and the same-binary bare translation lower bound at
0.310210637 seconds, leaving at most 3.259660 milliseconds/1.03986% for record
summary reuse. No 5--10% opportunity exists, so info is GPU-native force-only,
not performance-qualified or selected, and ordinary host selection remains.

B0105 cleanly performance-qualifies the named forced
`filters.hag_nn(count=4)` distinct-fifth-candidate fixture. The current binary
passes the 12-test physical lane and all four Compute Sanitizer tools; five
exact alternating pairs measure 0.996831814 seconds pinned PDAL versus
0.637868218 seconds PDG, or 1.562755x. The output-bound profile records 3.39736
milliseconds across 18 kernels, only 0.532612% of candidate wall, with the
shared masked-index gather owning 3.05 milliseconds. This qualification is
fixture-specific and forced; it adds no automatic selector or ordinary-data
claim.

B0106 fully reverts an exact ordinary planner-resident prototype after its
cheap gate. On the same 1,000,002-point fixture, one warm-cache directional
pair measures 0.675 seconds for the stats-free resident command versus 0.657
seconds for the same-binary forced-hybrid control: resident is 2.740% slower,
not 5--10% faster. Both and pinned PDAL write the exact 48,000,909-byte B0105
artifact. Proof telemetry records one planner-owned index, but also a
33,000,066-byte fallback-writer spill and 0.377666605 seconds in rewritten
manager execution. No runtime, model, or selector change survives.

B0107 retains a reusable exact direct LAS publisher for one resident-produced
standard binary64 extra dimension. Its first strict envelope is LAS 1.4,
format 7, 40-byte input records, and one canonical `OffsetTime` unsigned-32
Extra Bytes descriptor; it appends the standard double descriptor and raw
IEEE-754 bits, rewrites PDAL's summary/header, widens records to 48 bytes, and
atomically publishes without the terminal spill. The HAG fixture remains exact
at 48,000,909 bytes/SHA-256
`6e59b3a5b41cc5e219a5d39743c39ac7b2f09c84aa74b67e9b943e4cf41fe293`.
One warmup per route plus one directional trio measures 0.98 seconds pinned
PDAL, 0.64 seconds for the same clean binary's forced-hybrid control, and 0.59
seconds direct. The direct boundary is 7.8125% below its same-binary control,
so the reusable boundary survives. This is not a five-pair performance
qualification and adds no model or automatic selector.

B0108 fully reverts the explicit-force placement shortcut after its cheap
same-binary gate. The exact control takes 0.55 seconds and the shortcut 0.53
seconds, only a 3.636% reduction rather than the required 5--10%. Stats show
why: removing the 0.086755750-second initial-placement subphase moves startup
into device/profile work, leaving validation/placement/preflight nearly flat
at 0.166577409 versus 0.163389308 seconds; the stats-enabled shortcut is not
faster overall. No fast path, telemetry, model, or selector survives. B0107's
strict publisher now delegates an unsupported Extra Bytes descriptor before
forcing residency, with a full pinned-oracle process regression. Together with
the B0107 kernel and publication ceilings, this negative gate declares the HAG
direct-output endpoint sufficiently optimized for now.

B0109 measures and profiles the exact 1M
`readers.las -> filters.nndistance(k=10) -> writers.las(extra_dims=all)` graph
on the same canonical format-7/`OffsetTime` input. One warmup plus one
directional sample measures 2.01 seconds for pinned PDAL, 0.66 seconds for the
exact CUDA hybrid, and 0.39 seconds for a same-binary output-shaped control.
The fresh exact-output profile records 25.767520 milliseconds across 36
kernels, including 25.358368 milliseconds in the already shared Morton-BVH kNN
gather. The 0.27-second/40.91% gap above the output control is therefore a
material non-kernel lifecycle/boundary hypothesis, not a kernel target. These
single rows are directional and add no qualification or selection.

B0110 retains the bounded NNDistance extension to B0107's generic one-binary64
publisher. The exact envelope is only `mode=kth,k=10` on the same strict LAS
layout and `extra_dims=all`; unsupported mode, k, writer, or descriptor shapes
fail closed or delegate before output side effects. The physical process gate
proves the full pinned artifact, 32-byte-truncated canonical description, one
planner-owned index, zero terminal/fallback spill, exact experimental-only
fallback, and atomic output. ASan/UBSan and all four Compute Sanitizer tools
pass. On clean commit `1ba0a1ffa`, the stats-free same-binary control takes
0.71 seconds and direct takes 0.57 seconds, a 19.718% reduction. This clears
the retention gate but is one directional pair, not qualification or automatic
selection. The fresh profile reproduces the exact output and records only
25.856256 milliseconds/4.536% of direct wall across all 36 kernels.

B0111 retains mapped-LAS source reuse only for B0110's explicit exact endpoint.
The clean source-disabled/source-enabled pair improves from 0.58 to 0.38
seconds, a 34.483% reduction, while preserving the pinned output, record-summary
configured Morton-BVH, no host XYZ mirror, one planner-owned index, zero spill,
and atomic direct publication. The physical process, focused ASan/UBSan, and all
four Compute Sanitizer gates pass. The fresh profile totals 26.086656
milliseconds/6.865% of wall, almost entirely the existing shared gather;
publication is 14.145771 milliseconds, and B0108 already rejected the apparent
startup/placement surface at only 3.636%. The endpoint is sufficiently optimized
for now. This directional pair adds no performance qualification, model, or
automatic selection.

B0112 repeats and profiles the existing exact
`filters.hag_delaunay(count=3) -> writers.las(extra_dims=all)` pipeline on the
current clean binary. One warmup plus one pair measures 0.810889333 seconds
pinned PDAL and 0.607849511 seconds exact hybrid, or 1.334030x; both and the
profile reproduce the pinned output. Eighteen device launches total only
2.342688 milliseconds/0.385406% of candidate wall. A same-binary
HeightAboveGround output-shaped control takes 0.40 seconds, leaving a
0.207849511-second/34.1942% non-kernel source/PointView/publication boundary
hypothesis. This directional result adds no qualification or selection.

B0113 retains the strict mapped-source and one-binary64 publisher composition
around only the already-native count-three HAG Delaunay envelope. Its clean
source-disabled/source-enabled pair improves from 0.55 to 0.41 seconds
(25.4545%) with exact pinned output, one planner-owned masked 2D index, positive
tie/incomplete repair, zero spill, and atomic publication. The fresh profile
totals only 2.411420 milliseconds/0.588151% of wall; publication is 15.792846
milliseconds and B0108 already rejected the apparent placement shortcut below
5%. The endpoint is sufficiently optimized for now. This directional result
adds no performance qualification, model, or automatic selection.

B0114 retains that same mapped LAS source for B0107's already-exact explicit
HAG-NN count-four direct one-binary64 endpoint. Its clean
source-disabled/source-enabled pair improves from 0.56 to 0.37 seconds
(33.9286%) with exact pinned output, one planner-owned masked 2D index, positive
tie/incomplete repair, zero spill, and atomic publication. The fresh profile
totals only 3.523550 milliseconds/0.952311% of wall; publication is 12.601495
milliseconds and B0108 already rejects the apparent placement shortcut below
5%. The endpoint is sufficiently optimized. B0105's existing named forced
count-four qualification is unchanged; B0114 adds no model or automatic route.

B0115 measures the already-implemented exact affine
`filters.transformation -> pointwise assignment` descriptor-fused hybrid
region on clean commit `f3ffc50c2`. The exact 1,000,002-point directional pair
takes 0.35 seconds in pinned PDAL and 0.60 seconds in forced CUDA, only 0.583x
PDAL throughput. Its fresh profile records eight tiles and 32 launches, but all
unpack, transformation, assignment, and repack kernels together take only
0.849310 milliseconds/0.141552% of candidate wall. Even perfect device-kernel
fusion cannot reach the 5--10% complete-process gate. No code, model,
qualification, or selector changes; this endpoint is sufficiently optimized
for now and remains host-selected.

B0116 measures the current exact forced per-stage hybrid
`filters.label_duplicates(Classification) -> filters.nndistance(k=10) ->
filters.assign(UserData=Duplicate)` composition on clean commit `6b2a6c162`.
Five alternating exact pairs measure 4.374096348 seconds pinned PDAL versus
0.645599567 seconds forced hybrid CUDA, or 6.775247x. The fresh profile records
20 kernels totaling 28.478630 milliseconds/4.411191% of candidate median;
duplicate labeling itself is only 0.010810 milliseconds. The label stage uses
its standalone CUDA branch and restores `Duplicate` to the host PointView;
B0116 does not prove planner-resident reuse. Per-stage compute tuning stops.
Actual resident execution still reports `missing_calibration_model`, so this
qualifies only the explicit 1M forced-hybrid chain and changes no selector.

B0117's fully reverted one-line prototype anchors duplicate labeling to the
existing NNDistance model only long enough to execute the actual resident 1M
graph. It is exact and proves one selected region, one planner-owned index, and
no fallback boundary, but its warmed 0.70-second process is 8.43% slower than
B0116's 0.645599567-second forced-hybrid median. Boundary accounting is also
false: observed H2D/D2H are 35/11 MB versus predicted 26/10 MB. The prototype,
calibration ladder, model, and selector are rejected; the clean engine is
restored and all four coverage categories remain unchanged.

B0118 measures the exact 2M `filters.mortonorder ->
filters.assign(UserData=17)` complete pipeline on the current clean binary.
Five alternating pairs are byte-exact and measure 1.134717691 seconds pinned
PDAL versus 0.905711376 seconds forced hybrid CUDA, or 1.252847x. The fresh
profile totals only 0.618688 milliseconds/0.068310% of candidate median. A
separate exact current-binary Morton-only control takes 0.872632778 seconds,
so the entire adjacent point-program stage adds only 0.033078598 seconds/
3.790666%, below the reusable 5--10% gate. No fusion, resident executor,
model, selector, or product-code change follows; this endpoint is sufficiently
optimized for now.

B0119 measures the exact 2M `filters.mortonorder ->
filters.head(count=100)` complete pipeline on clean commit `c714f3e0b`.
Five alternating pairs are byte-exact and measure 0.806647036 seconds pinned
PDAL versus 0.578119502 seconds forced hybrid CUDA, or 1.395295x. The fresh
profile totals only 0.688512 milliseconds/0.119095% of candidate median. A
fully reverted four-line upper-bound prototype truncates the exact CUDA Morton
permutation to 100 ids before PointView publication but reaches only
0.553101868 seconds, a 4.327416% wall reduction with overlapping ranges. It
misses the 5--10% gate, so no fused ordinal product, model, selector, or code is
retained; the clean engine hash is restored.

B0120 physically measures the already-implemented finite-basic
`filters.stats` path on the exact six-dimension 2M LAS graph. LAS bytes,
metadata, streams, and status match pinned PDAL, but the current forced-CUDA
process takes 1.410544301 seconds versus 0.604473773 seconds pinned PDAL, or
0.428539x. A bare pinned read/write control attributes only 0.057345997 seconds
to CPU stats. The fresh profile records 498.966048 milliseconds in the 16
exact ordered-recurrence kernels alone: 8.700974x the CPU stage increment.
Serial within-dimension arithmetic leaves six blocks/0.01 waves per SM, so no
credible stats kernel, fusion, or residency direction can clear the 5--10%
process gate. Retain pinned-host selection and no product/model change.

B0121 replaces B0033's dirty direction with current-clean evidence for
`filters.hag_nn,count=2`. The existing forced-hybrid lane is exact at
1.578557x pinned PDAL. Extending the already-qualified strict mapped source
and one-binary64 publisher to count two then reduces median wall from
0.605963500 to 0.365209389 seconds (39.730794%) and reaches 2.606656x pinned
PDAL. One region, one planner-owned masked 2D index, direct record summary,
no host XYZ mirror, direct `HeightAboveGround` publication, terminal-spill
elision, and zero fallback boundaries are positively observed. Tie and
incomplete searches still repair through pinned host code before publication,
so the route remains explicit and uncalibrated.

The fresh direct profile totals 2.552608 milliseconds across all 20 kernels,
only 0.698944% of complete-process median; `knnGatherKernel` alone is
2.160576 milliseconds/0.591599%. Unprofiled publication is 13.127501
milliseconds/3.594514%, and every separately measured host-boundary surface
is below the 5--10% retention threshold. Device/profile and initial-placement
time includes the required first CUDA context/device setup. No further bounded
reusable optimization has a credible 5--10% ceiling, so this endpoint is
sufficiently optimized for now and no automatic selector follows.

B0122 current-clean qualifies the already-native `filters.hag_nn,count=3` 1M
lane at 1.549304x pinned PDAL in forced hybrid and 2.655409x through the same
strict mapped source and binary64 publisher. Direct median is 0.365397468
seconds, 41.094643% below the clean 0.620312799-second hybrid median. The
count-three direct path positively proves the same one-region/one-index,
source, record-summary, no-host-XYZ, terminal-spill, boundary-accounting, and
atomic-publication invariants, including tie and incomplete host repair. It
remains explicit and uncalibrated.

The fresh direct profile totals 3.076512 milliseconds/0.841963% of wall;
gather is 2.676992 milliseconds/0.732625%. Publication is 12.946378
milliseconds/3.543095%, and all host-boundary subphases together are
3.768882%. No bounded reusable 5--10% surface remains, so stop this endpoint.

B0123 current-clean qualifies the already-native `filters.hag_nn,count=1` 1M
lane at 1.442969x pinned PDAL in forced hybrid and 2.518270x through the same
strict mapped source and binary64 publisher. Direct median is 0.360501731
seconds, 42.773438% below the clean 0.629955252-second hybrid median. The
count-one route proves one region/index, record summary, no host XYZ mirror,
direct atomic output, terminal-spill elision, exact tie/incomplete repair, and
all four Compute Sanitizer tools. The fresh profile totals 2.067488
milliseconds/0.573503% of wall; publication is 4.553248%, and all measured
boundary subphases together are 3.797665%. No reusable 5--10% surface remains,
so stop the explicit, uncalibrated endpoint.

B0124 current-clean qualifies the already-native standalone
`filters.outlier(method=statistical,mean_k=8,multiplier=2,class=7)` 1M lane at
6.521497x pinned PDAL in forced hybrid and 10.739669x through the strict mapped
source and direct Classification publisher. Direct median is 0.373965127
seconds, 39.020273% below the clean 0.613261395-second hybrid median. The
explicit route proves one region/index, mapped record summary, no host XYZ
mirror, the exact nine-byte-per-point mean-distance/status download for the
host-order finale, a one-byte-per-point Classification boundary spill,
boundary accounting, atomic publication, strict option rejection, and exact
forced-grid incomplete host repair. All four Compute Sanitizer tools and
focused host ASan/UBSan are clean.

The fresh direct profile totals 21.450304 milliseconds/5.735910% of wall;
the required shared kNN gather accounts for 21.093440 milliseconds/5.640483%,
while every other kernel totals 0.095427% of wall. Canonical publication is
3.123725%, coordinate hydration 2.526728%, row materialization 2.638052%, and
the residual resident-wrapper ceiling after subtracting the gather is
3.522585%. The remaining device surface above 5% is the indispensable gather
whose bounded launch tuning B0053 already rejected; changing it now would
require a new algorithm. Stop this explicit, uncalibrated endpoint as
sufficiently optimized.

B0125 current-clean qualifies the already-native standalone
`filters.outlier(method=radius,radius=1,min_k=2,class=7)` 1M lane at
4.946130x pinned PDAL in forced hybrid and 8.107334x through the strict mapped
source and direct Classification publisher. Direct median is 0.382528055
seconds, 38.791349% below the clean 0.624957502-second hybrid median. The
explicit route proves one region/index, mapped record summary, no host XYZ
mirror, the exact four-byte-per-point radius-count download, one-byte-per-point
Classification spill, boundary accounting, strict-radius semantics, atomic
publication, and zero fallback. All four Compute Sanitizer tools and focused
host ASan/UBSan are clean.

The fresh direct profile totals 2.972310 milliseconds/0.777018% of wall;
`radiusCountsKernel` accounts for 2.640000 milliseconds/0.690145%, and every
other kernel totals 0.086872%. Canonical publication is 3.812374%, coordinate
hydration 2.556435%, row materialization 2.514113%, and the residual wrapper
ceiling after the radius query is 4.383453%. No bounded reusable 5--10% surface
remains, so stop this explicit, uncalibrated endpoint as sufficiently
optimized.

B0126 current-clean qualifies the exact 1M same-radius composition
`outlier(radius=1.01,min_k=2,class=7) -> radialdensity(radius=1.01) ->
assign(UserData=1 where RadialDensity>=0.2)` at 9.156744x pinned PDAL in forced
hybrid and 15.896697x through one strict mapped-source/direct-publication
region. Direct median is 0.372796165 seconds, 43.083688% below the clean
0.654990023-second hybrid median. The explicit route proves one planner-owned
3D radius index, device-resident RadialDensity, exact host-order outlier
comparison/finale, direct Classification/UserData publication, a separately
declared four-byte count result, matching boundaries, and zero fallback.

The fresh current-direct profile totals 5.661312 milliseconds/1.518608% of
wall; the two required radius queries account for 1.422899%, and every other
kernel totals 0.095709%. Canonical LAS publication is a required 5.540217%
full-elimination ceiling, combined host-boundary subphases are 5.197451%, and
each reusable constituent remains below 5%. No clear bounded optimization can
credibly deliver another 5--10%, so stop this explicit endpoint as sufficiently
optimized.

B0127 current-clean promotes only the exact format-7/36-byte, 250K--4M
B0126 composition. The fit uses the clean 250K/1M/4M direct rows and excludes
the exact 50K direction row: direct improves 5.644% there, but one noisy small
row does not establish a stable automatic break-even. The resulting
`radius-outlier-radialdensity-direct-compose` model is independently matched
by 152/152 frozen calibration cases over 35 models. Strict graph, option,
layout, device/profile, cardinality, preflight, one-index, mapped-source,
record-summary, no-host-XYZ, assignment-execution, boundary, and atomic-output
proofs all precede publication; every unsupported or below-threshold default
command delegates unchanged.

Fresh clean option-free five-pair gates reach 4.717949x pinned PDAL at 250K,
15.855278x at 1M, and 45.178325x at 4M. The current automatic profile
reproduces the exact 1M artifact and records 20 launches totaling 5.654080
milliseconds; the two required radius-count queries consume 93.724602% of
that device time. This is the same already-stopped B0126 kernel shape, and the
selector exposes no new reusable 5--10% non-kernel surface. The endpoint is
sufficiently optimized for now.

B0128 replaces B0025's dirty hypothesis with a clean current-binary result.
The exact comparator-unique 1M forced hybrid measures 1.310245631 seconds
pinned PDAL and 1.098422804 seconds PDG, or 1.192843x. Its exact-output NCU
capture records only 13 launches/about 0.258730 milliseconds, 0.023555% of
candidate wall. A directional Release-symbol phase profile measures the
filter at 245.294974 milliseconds, but only 6.222473 milliseconds is the
PointView permutation and 12.672796 milliseconds is the pinned recurrence;
92.296920% remains in repeated PointView extraction, staging/setup, and
Classification publication. A separate exact read/write control is
0.300782159 seconds. Those controls establish a reusable host-boundary
surface well above 5--10%, so this endpoint is not stopped.

B0129 retains the strict explicit direct composition at clean implementation
commit `87f14b22f`. It reuses the mapped LAS source, existing exact unique-Z
CUDA ordering, unchanged host recurrence, and a permutation-aware canonical
LAS publisher. The current-clean five-pair median is 0.425794757 seconds
versus 1.229485169 seconds pinned PDAL, or 2.887507x; it is 61.235805% below
B0128's 1.098422804-second forced-hybrid control and preserves the exact
36,000,375-byte artifact. The route proves one whole-view global-order region,
zero indexes, one 8,000,000-byte Z upload, one 8,000,000-byte permutation
download, mapped-source use, no host XYZ mirror or record-summary claim,
terminal-spill elision, matching boundaries, and atomic publication.

The fresh exact-output profile records 13 launches totaling 0.260192
milliseconds, only 0.061107% of candidate median. An ordinary stats pass
records 0.032016684 seconds of canonical publication; removing two thirds of
that entire interval would be necessary merely to reach a 5% process gain.
Device/profile plus initial placement is 0.166260729 seconds, but it contains
the required first CUDA context/free-memory work and B0108 already rejected
the reusable shortcut at 3.636364%. No clear independent 5--10% surface
remains, so the explicit endpoint is sufficiently optimized. No model or
automatic selection follows.

B0130 remeasures the existing exact `filters.sort` path from clean commit
`43e57d7e2` on the deterministic nonidentity comparator-unique 1M Z fixture.
The forced CUDA median is 1.117247334 seconds versus 1.562019960 seconds
pinned PDAL, or 1.398097x, but the fork's exact host wrapper is faster still at
0.903809508 seconds. Therefore this names GPU-native coverage and a measured
boundary hypothesis without justifying automatic selection.

The exact-output NCU capture records 13 launches totaling 0.271040
milliseconds, only 0.024260% of CUDA process wall. A directional Release-
symbol profile measures 0.204735422 seconds in the filter: 0.112223505 in the
CUDA wrapper, 0.007759453 in full-view permutation, and 0.084752464 in the
remaining key gather/setup. The dominant opportunity is the host source and
publication boundary plus CUDA setup around an already-tiny ordering
sequence, not another kernel.

B0131 retains that strict mapped-source/permutation-publisher composition at
clean implementation commit `da2e7323f`. On the same canonical LAS 1.4
format-7/36-byte `sort(Z,ASC,NORMAL)` graph, the current-binary direct median is
0.319400289 seconds versus 1.054076074 seconds pinned PDAL, or 3.300173x. The
same current binary's exact host control is 0.699280617 seconds, so direct is
54.324447% faster and clears the 0.664316586-second retention threshold. All
routes preserve the same 36,000,375-byte nonidentity artifact.

The fresh output-bound profile records 13 launches totaling 0.256640
milliseconds, only 0.080351% of candidate wall. An ordinary stats pass records
0.025040367 seconds of canonical publication and 0.036952070 seconds of the
entire rewritten manager. Publication would need a 63.777078% reduction, or
the complete manager a 43.218186% reduction, merely to save 5% process wall;
the B0130-isolated full-view permutation is only 2.429382% of this endpoint.
B0108 already rejected the apparent reusable placement shortcut below 5%.
No clear independent 5--10% surface remains, so this explicit endpoint is
sufficiently optimized. No model, automatic selector, new sort kernel, or
wider sort envelope follows.

B0132 changes no product code and performance-qualifies that already-
implemented explicit `hag_delaunay(count=3)` mapped-source/direct-output
composition from clean commit `7d0460594`. Five exact current-binary pairs
measure 0.365213991 seconds direct versus 0.805184970 seconds pinned PDAL, or
2.204694x, while preserving the exact 48,000,909-byte artifact and proving the
mapped source, record summary, no host XYZ mirror, one planner-owned index,
direct binary64 publisher, matching boundaries, and terminal-spill elision.

The fresh current-binary profile records 20 launches totaling 2.421952
milliseconds, only 0.663160% of candidate wall. Canonical publication is a
3.818205% full-elimination ceiling. The whole manager is 22.858847%, but no
independent removable fraction above the 5--10% gate is identified; it owns
the exact shared-index query, transfer, and result path. B0108 already rejects
the apparent startup shortcut below 5%. This confirms B0113's stopping
decision: the explicit endpoint is sufficiently optimized, has no model or
automatic selector, and now has a named current performance qualification.

B0133 changes no product code. Five alternating exact pairs measure pinned
PDAL `hag_nn(count=5)` at 1.011225052 seconds on the 1,000,002-point fixture;
the current `pdg` default correctly delegates at 1.028807390 seconds. A
separate output-shaped pinned-PDAL control is 0.289291154 seconds, leaving
0.721933898 seconds, or 71.392011% of stock wall, attributable to the stage.
The exact-output CPU phase capture measures 0.756377590 seconds in the filter:
only 0.063686914 seconds/8.419990% builds the 2D index, while the post-index
query/projection path owns 0.680035688 seconds/89.906906%. The dominant cost
therefore supports a cheap reuse prototype, not a private index, new kernel
family, model, or selector. Count five remains pinned-host delegated and
unmeasured as acceleration until that prototype passes its current-binary
gate.

B0134 writes the count-five proof matrix first and extends the existing bounded
HAG-NN path without adding a kernel family or private index. Counts one through
five now share the planner-owned masked 2D query and generic ordered projection;
tie, incomplete, insufficient-ground, nonfinite, option, and backend drift
retain exact host repair. The strict mapped-source/one-binary64 publisher is
explicit only, count six remains host-owned, and ordinary option-free execution
remains unchanged.

The 92-case host/CUDA process matrices, 27 focused planner/rewrite/lifecycle
tests, strict direct process gate, complete Release unit binary (551 pass/two
optional local-corpus skips), focused ASan/UBSan, and all four Compute
Sanitizer tools pass. A dirty-snapshot one-pair exact prototype measures
0.359718221 seconds direct versus 1.042808916 seconds pinned PDAL, or 2.898961x,
and proves one planner-owned index, mapped source/summary, no host XYZ mirror,
direct binary64 output, zero spills, and exact publication. This clears the
retention direction but is not a performance qualification.

B0135 current-clean qualifies that explicit count-five endpoint on implementation
commit `7ca04e899`. Five alternating exact pairs measure 1.007641329 seconds
pinned PDAL versus 0.343845512 seconds direct, or 2.930506x, while proving the
mapped source, record summary, no host XYZ mirror, one planner-owned index,
zero spills, direct binary64 publication, and exact 48,000,909-byte output.
The fresh output-bound profile records 20 launches totaling 4.391810
milliseconds/1.277263% of candidate wall; the shared kNN gather is 3.960000
milliseconds/1.151680% and the HAG projection only 0.090500 milliseconds.
Canonical publication is a 4.352942% full-elimination ceiling. The complete
manager is 23.751502%, but would need a 21.051301% reduction merely to save 5%
process wall and exposes no independent reusable component at that scale.
B0108 already rejects the apparent placement shortcut below 5%. The explicit
count-five endpoint is therefore performance-qualified only for the named 1M
SM89 fixture and sufficiently optimized; no model or automatic selector follows.

B0136 changes no product code. Five exact current-clean pairs measure pinned
PDAL `hag_nn(count=6)` at 1.019913125 seconds; the current default exact
delegate is 1.041636052 seconds. A fresh pinned output-shaped control is
0.296473777 seconds, leaving 0.723439348 seconds/70.931468% of stock wall in
the stage. The exact-output CPU phase capture measures 0.768510391 seconds in
the filter: only 0.063233061 seconds/8.228003% builds the 2D index, while the
post-index query/projection tail owns 0.691737660 seconds/90.010190%. This
clears only a cheap bounded reuse prototype; count six remains host-delegated,
unqualified, and unselected.

B0137 writes the count-six proof matrix first and extends only the strict
explicit HAG-NN envelope. Counts one through six now reuse the same mapped LAS
source, planner-owned masked 2D query, ordered projection, exact repair
semantics, and one-binary64 publisher; count seven and wider remain host-owned.
The 110-case host/CUDA matrices, focused planner/rewrite/lifecycle/bridge
tests, strict direct process gate, complete Release unit binary (562 pass/two
optional corpus skips), focused host ASan/UBSan, and all four Compute Sanitizer
tools pass. A dirty-snapshot exact viability pair measures 1.025019333 seconds
pinned PDAL versus 0.355610480 seconds direct, or 2.882422x, with one index,
zero spills, and exact 48,000,909-byte publication. This clears the retention
gate but is not performance qualification evidence.

B0138 current-clean qualifies that explicit count-six endpoint on implementation
commit `c3602239b`. Five alternating exact pairs measure 1.023952521 seconds
pinned PDAL versus 0.344893464 seconds direct, or 2.968895x, while proving the
mapped source, record summary, no host XYZ mirror, one planner-owned index,
zero spills, direct binary64 publication, and exact 48,000,909-byte output.
The fresh output-bound profile records 20 launches totaling 5.324470
milliseconds/1.543801% of candidate wall; the shared kNN gather is 4.880000
milliseconds/1.414930% and the HAG projection only 0.104670 milliseconds.
Canonical publication is a 4.296978% full-elimination ceiling. The complete
manager is 23.041380%, but would need a 21.700089% reduction merely to save 5%
process wall and exposes no independent reusable component at that scale.
B0108 already rejects the apparent placement shortcut below 5%. The explicit
count-six endpoint is therefore performance-qualified only for the named 1M
SM89 fixture and sufficiently optimized; no model or automatic selector
follows.

B0139 completes that measurement on clean commit `a03b5bc19` and changes no
product code. The retained offset-grid fixture is reused unchanged because its
enumerated seventh/eighth candidate separation is 0.2 for interior rows and 1.0
at both X extremes. Five alternating exact pairs measure pinned PDAL
`hag_nn(count=7)` at 1.050776026 seconds against a correctly delegating
1.075603836-second default, and the count-seven artifact differs from the
count-six artifact, so the seventh retained candidate genuinely changes the
published interpolation. The fresh pinned output-shaped control is 0.292334145
seconds, leaving 0.758441881 seconds/72.179214% of stock wall in the stage —
above count six's 70.931468%. The exact-output optimized-oracle phase capture
reproduces the paired hash and measures 0.801285042 seconds inside the filter:
9.682102% builds the 2D index and 88.392121% is the post-index query and
ordered projection. The stock surface is therefore material and the dominant
cost is again the query/projection tail, not index construction.

B0140 extends only that strict explicit envelope to count seven, writing the
proof matrix first. The device projection was already count-generic — it
accumulates over `min(groundCount, neighbors)` with no fixed-width state — so
the slice raises seven bounds and adds proofs rather than a kernel family. The
128-case host and 120-case CUDA matrices, full 573-pass Release unit binary
plus two optional corpus skips, strict direct process gate, focused host
ASan/UBSan, and all four Compute Sanitizer tools pass. Every native-required
count-seven fixture was verified tie-free in binary64 before acceptance; one
candidate arithmetic ground was replaced after the exactness guard correctly
declined its bit-identical squared distance. One exact dirty-snapshot pair
measures 1.044514334 seconds pinned PDAL versus 0.361863314 seconds direct, or
2.886489x, proving the mapped source, record summary, no host XYZ mirror, one
planner-owned index, direct binary64 publication, one 25,000,050-byte upload,
and zero spill/fallback boundaries. This clears the retention gate but is not
performance qualification evidence.

B0141 current-clean qualifies that explicit count-seven endpoint on
implementation commit `437b0b060`. Five alternating exact pairs measure
1.051637191 seconds pinned PDAL versus 0.355388678 seconds direct, or
2.959118x and 66.206152% lower wall, while proving the mapped source, record
summary, no host XYZ mirror, one planner-owned index, one 25,000,050-byte
upload, zero spill/fallback, direct binary64 publication, and exact
48,000,909-byte output. The fresh output-bound profile records 20 launches
totaling 6.421888 milliseconds/1.807004% of candidate wall; the shared kNN
gather is 1.677343% and the HAG projection 0.033829%. Canonical publication is
a 5.034629% full-elimination ceiling that cannot actually be eliminated, and
the complete manager would need a 17.130% reduction merely to save 5% wall.
The largest interval is the 50.412811% validation/placement/preflight startup
that B0054 measured as necessary and B0108 prototyped away for only 3.636364%.
That startup is roughly fixed in absolute terms, so it dominates any
sub-half-second endpoint and is a shared structural limiter rather than a
count-seven opportunity. The endpoint is therefore performance-qualified only
for the named 1M SM89 fixture, sufficiently optimized, and unselected.

B0142 settles that scoping question by measuring the range instead of guessing
it, and changes no product code. Five exact pairs at each of counts 8, 12, 16,
24, 32, 48, and 64 put pinned PDAL at 1.075645517 through 2.057952491 seconds;
against a fresh 0.290153739-second output control the stage surface grows from
0.785491778 to 1.767798752 seconds and from 73.0252% to 85.9009% of stock
wall. Cost is strongly sub-linear in count — 9.142857x more neighbours for
2.3241x more stock stage time, with stock per-neighbour cost falling 3.93x —
because the dominant cost is the largely count-independent per-query index
traversal. The acceleration opportunity therefore grows with count. The
retained fixture keeps the k/(k+1) boundary distinct for every k in 1..64, the
device projection is already count-generic, the preflight already budgets
`3 + count * (4 + 8)` bytes per point, and the region already rejects
`maximumNeighbors > 64`.

D0203 therefore closes the remaining range as one parameterized envelope: the
count ladder is a proof ladder, not an implementation or cost ladder, and
per-count slices would spend roughly 57 further three-commit triads re-proving
a structurally identical envelope.

B0143 closes that envelope at 64 in one slice. Counts one through seven keep
their explicit cases and 8/16/32/64 are generated over count, growing the
matrices to 200 host and 192 CUDA cases; the Release unit binary passes 577
tests plus two optional corpus skips, host ASan/UBSan passes 386, and all four
Compute Sanitizer tools are clean. Every native-required generated fixture was
proved tie-free in exact binary64 before acceptance. A wider 79-ground fixture
was added after the direct process gate failed honestly at count 16 — the
shared 7x3 fixture has only seven ground rows, so wide counts correctly fell
back for insufficient ground — so the strict direct source and publisher are
still proved at width. One exact dirty-snapshot count-16 pair measures
1.224073288 seconds pinned PDAL versus 0.366412903 seconds direct, or
3.340694x, the highest HAG-NN result recorded and a direct confirmation of
B0142's prediction that the opportunity grows with count.

B0144 qualifies that envelope at representative counts: 3.080885x at count 16
and 4.374517x at count 64, both byte-exact, with speedup rising with count
exactly as B0142 predicted. Unlike every earlier HAG-NN endpoint it does not
stop. Its fresh profiles put all kernels at 5.297114% of wall at count 16 and
17.937498% at count 64, with the shared `bvhKnnGatherKernel` alone at
17.576763% at count 64 — a reusable component clearing the 5--10% gate by a
wide margin, in the one category D0204 identified as historically paying.
Startup falls to 28.2567% purely because the wall grew, confirming D0204's
fixed-cost claim.

B0145 profiles that gather and names its limiter. Backend choice is rejected as
the lever by measurement: adaptive 0.476555539 s, forced BVH 0.477948684 s
(+0.2923%, noise), forced uniform grid 0.832525226 s (+74.6964%), all
byte-identical, so the planner's adaptive selection is already optimal at wide
count. `bvhKnnGatherKernel` is compute/latency-bound (67.00% compute versus
49.95% memory throughput) and its occupancy is capped by register pressure:
48 registers per thread allow only 20 blocks per SM against SM and warp limits
of 24, giving 72.72% achieved against an 83.33% theoretical ceiling. Shared
memory is unused and is not the constraint.

The single next smallest high-value task is B0146: prototype that occupancy
change — `__launch_bounds__` or launch shape on this kernel — behind the usual
cheap gate, retaining it only for an exact same-binary 5--10% complete-process
improvement on at least one shared-index consumer. B0053's launch-shape
rejection was measured on the uniform-grid kernel and must be re-measured here.
Validate against the whole neighborhood matrix and all four Compute Sanitizer
tools, because the kernel is shared: a change that improves HAG-NN while
perturbing another consumer's repair predicate is a regression even if its own
gate passes. This remains the shared gather only; D0204's admission rule still
forbids new standalone per-stage kernels. After that, the D0204 order begins with a LAZ/COPC decode
*measurement*, since that ranking is still an extrapolation. Do not reopen the
process-level CUDA startup partition without new evidence — B0054 and B0108
already closed it.

## Execution status — 2026-08-09

- Fork topology: complete. `gpu-main` is based on and tracks
  `upstream/master` at `f1e35f5c3e416b8c6a2c39966c8205cfae54afe2` (PDAL
  2.10.0); `origin` is the hosted `zymazza/PDAL` fork, whose `master` points to
  the same commit.
- Build spine: host debug, ASan/UBSan, CUDA debug, CUDA release, published
  upstream PDAL tests, and CI presets exist. Release configuration now requests
  every real architecture supported by the selected CUDA compiler plus its
  newest PTX target. CUDA 12.x is the legacy-compatible SM 50–90 builder;
  CUDA 13+ is the current SM 75+ builder. The local CUDA 13.3 compiler reports
  SM 75/80/86/87/88/89/90/100/103/110/120/121. Only the installed RTX 4090
  (SM 8.9, 25,280,839,680 bytes) is physically exact-qualified today; build
  coverage is not represented as cross-SM runtime proof. Managed workspace
  runs require explicit host-device access because the default sandbox masks
  NVIDIA devices.
- Core slice: the generated 132-dimension catalog, checked coordinate model,
  lazy SoA batch, host allocator, CUDA stream-ordered pool, LAS 1.0–1.4 header
  view, host/CUDA XYZ transpose/repack paths, and pinned CUDA staging allocator
  are implemented. Devices without stream-ordered memory pools select a
  lifetime-safe `cudaMalloc`/`cudaFree` allocator; a forced-classic regression
  passes on the RTX 4090 and `gpupal doctor` reports pool capability.
- Full-PDAL execution shell: `gpupal` exposes every command and all 122 drivers in
  the configured fork (84 filters, 24 readers, and 14 writers). Specialized
  whole-pipeline paths run first; otherwise compatible point-local regions are
  replaced inside the original PDAL `PipelineManager`; every other command or
  graph uses the sibling pinned PDAL implementation with its original argv.
  Functional availability therefore no longer depends on native CUDA stage
  count. `docs/stage-coverage.md` reports functionally supported, GPU-native,
  performance-qualified, and automatically selected envelopes separately, as
  required by D0011/D0099. The current automatically selected set is
  enumerated in `docs/stage-coverage.md`; this dated status block does not
  duplicate a count that can drift as measured envelopes are accepted. Every
  other implemented envelope is reported individually rather than collapsed
  into a misleading qualification count. Neither is catalog completion. The
  public `gpupal` launcher is a dependency-light dispatcher:
  commands and pipelines with no current PDG candidate go directly to sibling
  `pdal`, while candidates and ambiguous inputs enter `pdg-engine`. This
  removes PDAL/GDAL/CUDA startup from the common untouched-fallback path
  without changing the conservative in-engine selector.
- Exactness: the default LAS path maps input/output and converts formats 0–3
  and 6–8 to exact PDAL 2.10 default format-7 bytes. Canonical format-7/8
  records use an exact 36-byte copy-and-summary path. The hybrid executor fuses
  contiguous assign/ferry/expression/range/single-bounds-crop regions and
  exact decimation/head/tail ordinal programs while retaining every
  surrounding PDAL stage in-process. Its packed streaming
  boundary transfers each AoS batch once, transposes only touched fields,
  preserves custom device intermediates, and repacks only writes. The direct
  CUDA LAS path also fuses
  ordered predicates, stable compaction, exact bounds/return recount, and
  final-size truncation. Unproven options, layout widening, reader order, or
  conversion failures delegate before output is emitted. The resident
  neighborhood bridge additionally compares the original and rewritten
  ordered dimension layouts when `writers.las` serializes `extra_dims=all`;
  a changed custom-dimension order delegates before execution.
- Exact global reduction: `filters.locate` now compiles into a reusable typed
  first-tie arg-min/arg-max primitive. A linear, order-proven hybrid pipeline
  may place it between point-program regions; NaNs, infinities, PDAL's initial
  sentinels, invalid kinds, empty input, integer-to-double conversion, and
  original source indices remain exact. Its CUDA implementation uses a
  two-level fixed candidate reduction, but automatic selection remains off
  pending the hardware gate.
- Exact summary metadata: `filters.stats` reproduces the pinned online-moment
  update order, advanced and global statistics, enumeration/count structures,
  warnings, dimension order, and bbox/SRS metadata in streaming and standard
  execution. Its finite basic-summary CUDA primitive runs dimensions in
  parallel while preserving the serial recurrence within each dimension.
  Default replacement is disabled after the exact host wrapper measured below
  pinned PDAL; unchanged stats is an audited order/view-preserving bridge so
  qualified regions on either side can still remain native.
- Exact coordinate and robust-statistics maps: `filters.transformation`
  preserves simultaneous original-XYZ reads and full host projective
  arithmetic, with an affine double-XYZ CUDA envelope. `filters.iqr` and
  `filters.mad` reproduce PDAL's exact order-statistics, strict selection, and
  stable survivor order, with a finite/no-negative-zero CUB device envelope.
  Their physical process and all-four-sanitizer gates pass. Automatic
  standalone CUDA remains off because clean complete-process trials are
  negative or effectively tied; D0042 retains the exact kernels for resident
  fusion and the faster host path for ordinary use.
- Exact global ordering: `filters.sort` now preserves typed normal/stable pass
  semantics, directions, duplicate/NaN behavior, and PDAL's last-dimension
  priority while permuting the original point view. Its stable CUB key/index
  implementation publishes only stable single-key or tie-free final-key
  results; all other device candidates return to the exact host path before
  view mutation. `filters.mortonorder` reuses that stable radix primitive while
  reproducing both the ordinary most-significant-bit comparator and reverse
  grid/interleave/bit-reversal traversal, including stable duplicate codes and
  upstream's distinct empty-view behavior. B0006 qualifies automatic CUDA for
  finite nondegenerate views from 2,000,000 points; smaller, unsupported,
  disabled, device-less, or recoverably failed cases retain the exact host
  path.
- Exact categorical and return partitioning: `filters.groupby` converts keys
  through PDAL's `int64` contract and preserves source-first PointView ids.
  `filters.returns` preserves fixed first/intermediate/last/only view identity,
  malformed records, warnings, and stable source order across multiple input
  views. CUDA uses stable CUB key/index permutations for both. Exact no-option
  `filters.merge` retains its persistent upstream output view and is the
  audited boundary that reopens later single-view native stages.
- Validation: Host Debug registers 267 tests (266 pass and the
  opt-in local LAS corpus case skips). The 23-case ordinal, 15-case
  locate, nine-case transformation, 16-case robust-statistics, 20-case
  ordering, 15-case Morton, 16-case groupby, 29-case returns/merge, 35-case
  divider/splitter, 39-case color-interpolation, 14-case randomize-composition,
  22-case stats, 14-case outlier, 10-case radialdensity, 13-case nndistance,
  12-case normal, 13-case eigenvalues, 18-case covariancefeatures, 26-case
  approximatecoplanar, and 19-case expressionstats process matrices, three
  extended all-feature differentials, two composed resident-region
  differentials, and
  two direct neighborhood-column consumer differentials pass. The
  newer matrices, including divider/splitter and color interpolation, also pass
  with libasan preloaded into both oracle and candidate; stats, outlier,
  radialdensity, normal, eigenvalues, covariancefeatures, and nndistance do as
  well. The complete ASan/UBSan tree also passes 266 executions plus the
  expected corpus skip in 267 registrations. Exact-differential CMake now
  injects `detect_leaks=0` itself; the separate candidate-focused leak lane
  remains. Three initial failures compared out-of-band, process-specific
  LeakSanitizer diagnostics; all three reran clean under the prescribed mode.
  The post-portability SM-89 CUDA Release checkpoint on the
  physical RTX 4090 registers 385 tests and passes 384 with that same skip in
  103.66 seconds. Its compact 22-test architecture bit lane and forced classic
  allocator regression pass separately as well. The final CUDA tree includes
  all current CUDA-labelled cases and forced neighborhood process
  differentials. The CUDA build includes process
  differential entries across LAS/LAZ,
  BPF, PLY, PCD, text, and sorted COPC; ordered range/expression/crop cases;
  ordinal stream/standard/split and locate min/max/chain cases; explicit
  fallback boundaries; and
  generated formats 0–3 and 6–8. The rebuilt network-enabled upstream
  validation passes all 142 test executables in PDAL's published procedure:
  all 140 local executables pass, and the two remote-fixture executables pass
  on their network-enabled rerun.
  The `filters.approximatecoplanar` slice contributes a 26-case host process
  matrix, eight focused CUDA process differentials, and two CUDA unit/property
  tests. The tie-repair and resident compositions each repeat exact output five
  consecutive times. Both unit paths are clean under memcheck, initcheck,
  racecheck, and synccheck. Four complete process cases are also clean under
  all four tools (16 invocations); max-`knn` and tie repair subsequently pass
  racecheck and synccheck as well, making all six process cases clean under all
  four tools. The serialized loop uses the uniform grid for standalone
  approximate-coplanar, the Morton BVH for the other cases, and an explicit
  proof guard for forced tie repair.
  D0049 now records a proposed, not yet accepted, automatic
  approximate-coplanar selector at 262,144 points for the current RTX 4090
  profile. At revision `e0486c78ea7899b316037ebf1162bcc4e52c925f`, clean
  forced two-warmup/ten-sample alternating measurements are exact and reach
  2.158750x pinned PDAL for LAS and 1.974880x for point-format-8 LAS. The
  2.200810x LAZ report is a dirty-tree diagnostic and is not accepted evidence.
  These forced reports do not prove normal option-free selection. The
  candidate requires a direct three-stage reader/filter/writer pipeline,
  regular pipeline JSON and LAS/LAZ input files, a valid count header, a known
  cardinality of at least 262,144 after the reader count cap, default or
  explicit `knn=8`, uncompressed `.las` output with `extra_dims=all`, a
  standalone non-reusing region, an available CUDA runtime, and the candidate
  `NVIDIA GeForce RTX 4090` SM-89 profile built with CUDA 13.3. Every
  failed gate delegates unchanged or uses the exact host wrapper. Production
  artifacts compile the proposal off; a separate clean qualification artifact
  enables it without a runtime selection variable.
  `PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA=1` is the proof
  guard, but the clean option-free process runs and profile are still pending;
  the automatically qualified filter count therefore remains seven. Runtime
  exactness is physically qualified only on SM 89. Separately, a serialized
  CUDA 13.3 portable compile in
  `build/pdg-cuda-portable`, configured with tests off, architectures `all`,
  and the portable-architecture guard on, builds `pdg_core` 51/51 at
  `--parallel 1`, including `NeighborhoodKernels.cu`. `cuobjdump` records SASS
  in that object for SM 75, 80, 86, 87, 88, 89, 90, 100, 103, 110, 120, and
  121 plus PTX for SM 120. That proves coverage across the architectures
  supported by the installed compiler plus newest-target PTX, not bit-exact
  runtime support on any architecture other than SM 89.
  At the last accepted crop checkpoint, memcheck, initcheck, and synccheck
  covered the 24-test broad CUDA matrix plus the new crop predicate; racecheck
  covers its 22-test lane plus
  crop, with zero errors or hazards. The current grid/BVH `CudaSpatialIndex`
  suite, device-column projectors, and complete resident consumer separately
  report zero errors under memcheck, initcheck, and synccheck and zero hazards
  under racecheck. Ordinal, locate, and info now pass their focused physical
  and all-four-sanitizer gates, but negative complete-process measurements keep
  them host-selected and force-only; only stats in that group still awaits its
  physical, sanitizer, and break-even decision. Compilation is not counted as
  runtime validation. Transformation, IQR, MAD, and color
  interpolation pass physical/sanitizer gates but retain host standalone
  selection after the negative clean matrix in D0042. Sort, Morton, groupby,
  returns, divider, and splitter now
  pass their physical properties, forced process differentials, and all four
  direct Compute Sanitizer tools. Complete-process diagnostics select host for
  standalone sort and partition stages. Morton additionally passes the clean
  option-free threshold/profile gate and automatic-path memcheck at B0006.
  Expressionstats passes its physical property, ten-case forced-device matrix,
  all four direct Compute Sanitizer tools, and real LAS/LAZ crossover
  diagnostics. B0008/D0044 accept automatic CUDA at the persistent-workspace
  4M/2M/1M one/two/three-expression work curve; cross-stage resident-column
  reuse remains pending.
  The neighborhood slice has passed its current physical
  exactness and shared-index sanitizer gates. Deterministic tiled radius
  execution is also exact through both outlier and radialdensity, but automatic
  stage selection remains disabled pending complete-process break-even,
  plan-wide tiled residency, real-corpus qualification, and option coverage.
  Locate additionally passes a 21,970,934-point, 168-chunk local exact
  differential. IQR and MAD each scan the same 21,970,934-point read-only file
  and exactly match the oracle's 2,265-byte empty LAS (SHA-256
  `24fd80a8db0b248e9b86e83560679fce458af8b12a4cd24cf4dcaa1d633b583b`).
  Stable Classification sort followed by `filters.head` over all 21,970,934
  points also matches the oracle's 5,865-byte LAS (SHA-256
  `f0961728c852e64d6d9e1106525913aa2361ea3348ca87f897d3a92f1f020885`).
  Ordinary Morton order followed by the same head over all 21,970,934 points
  matches the oracle's 5,865-byte LAS (SHA-256
  `61638be4f61d806608c4ac6df4da0e0715915d0347b1dcf68b57afe1bf898c6e`).
  Grouping all 21,970,934 points by Classification matches four oracle LAS
  files totaling 790,962,684 bytes, including view order and filenames.
  Return partition, exact merge, and a 100-point head over the same input match
  the oracle's 5,865-byte LAS (SHA-256
  `64432d20259eaabbe5c3c93e6be1243f07f9ec0160de90af38f17e2162b853a1`).
  Divider, splitter, exact merge, and head over the same input match the
  oracle's 5,865-byte LAS (SHA-256
  `e13d95da3dcc4a39309b5f38dbe4914adaa02af41208c820e56d586876e213c3`).
  Streaming color interpolation over all 21,970,934 points produces the
  oracle's exact 790,955,889-byte LAS (SHA-256
  `43fead56eade3ffa503311d9f47328d2fbd6888b913b5c7d92dfa12c8881dc23`).
  Six-dimension stats over the same point count produces the oracle's exact
  790,955,889-byte LAS (SHA-256
  `ff14463744dbe9ddd2f1d10271d278a7e41478e254d96a0780bda9d7aa1da2fe`).
- Performance: same-machine tuning trials across 2, 4, 5, 6, 8, 12, and 24
  workers selected five for the canonical copy path. Clean benchmark B0002 at
  commit `90e18d976` records a 0.140319 s median, 5.6368 GB/s, and 156.58
  million points/s on the 21.97-million-point case: 39.485x the pinned PDAL
  median with the same output hash. Clean benchmark B0005 at commit
  `11a90bef9` records the option-free exact fused CUDA pipeline at 0.460080 s,
  1.7192 GB/s, and 47.7546 million points/s: 18.835x pinned PDAL, again with
  identical status, streams, size, and output SHA-256. The current ordered
  filtering checkpoint measures 17.347x for an all-pass range and 11.495x for
  a selective range retaining 3,045,832 points, with exact output hashes. The
  generic packed hybrid path measures 1.439x on the same five-assignment/range
  workload; the next clean benchmark record will promote these exploratory
  reports into `BENCHMARKS.md`. Clean benchmark B0006 at commit `7b27cb1fb`
  records option-free exact Morton medians of 1.248x pinned PDAL for ordinary
  mode and 1.487x for reverse at the 2,000,000-point automatic threshold, and
  1.528x ordinary at 4,000,000 points. A 1,000,000-point run remains
  host-selected at 0.983x. Clean B0008 at commit `d70ae78b5` adds a
  persistent-workspace automatic expressionstats gate: 1.126x at 1M points
  for three expressions, 1.212x at 2M for two, and 1.105x at 4M for one.
  Matching controls below each boundary remain host-selected at 0.957x,
  0.973x, and 0.979x. Clean B0009 measures the dependency-light dispatcher on
  an untouched 5,327-point LAS-to-text fallback: it is 1.283x faster than
  entering the former heavy engine, but still 0.973x direct pinned PDAL because
  parse/exec costs 1.43 ms. A separate dirty-tree stats host diagnostic is
  deliberately negative: 6.363435 s versus 5.785434 s for pinned PDAL, or
  0.909x. It disables automatic stats replacement; a follow-up option-free
  trial measures 5.848450 s versus 5.823799 s (0.996x), confirming delegation
  with only selection overhead. Neither makes a CUDA speed claim; an RTX 4090
  runtime/break-even gate remains required. The new exact spatial benchmark
  records a separate unaccepted dirty-tree G1 diagnostic at `k=16`: grid wins
  uniform ALS by 20.573x and mixed density by 2.027x, while BVH wins the
  million-point clustered TLS profile by 1.390x. Complete result/status bytes
  match between backends and the corrected selector chooses each measured
  winner. Those are resident synthetic primitive timings without a PDAL
  process baseline, so they do not establish a stage or product speedup.
- CUDA translation: the complete format-7 pack kernel, integer bounds/return
  reductions, bounded pinned staging, and multi-chunk assembly compile for the
  production architecture matrix. Its all-format, scan-angle, chunk-boundary,
  host-equivalence, and Compute Sanitizer gates pass on the RTX 4090. Pure
  translation remains behind the D0005 experimental CLI gate because the
  mapped host copy path is faster. The same two-lane pipeline now executes
  ordered expression/range/crop predicates and implemented ordinal selection,
  CUB-stable compaction, post-filter bounds and return histograms, and exact
  final-size output. D0010/D0011
  automatically select CUDA only for a measured, sufficiently large and
  compute-heavy fused point-program envelope.
- P1 planner and point-program slice: pipeline arrays and
  `{pipeline:[...]}` objects
  compile into typed reader/filter/writer DAGs with explicit tags/inputs,
  descriptors, reverse liveness, host/device edge counts, touched-column byte
  estimates, and strict fallback reporting. `filters.assign` now has a pinned
  grammar-compatible postfix compiler and exact ordered host VM; its bounded
  CUDA VM covers the proven arithmetic/logical subset. Ferry mappings lower to
  the same program, so arbitrary adjacent assign/ferry chains execute in one
  logical-value LAS pass. The exact CUDA implementation translates input
  records, transposes only touched canonical columns, executes the ordered
  program, recounts logical returns, repacks records, and streams positioned
  output through two overlapping lanes without returning intermediates to
  host memory. Custom intermediate dimensions are supported when the default
  writer ignores them; layout widening and lossy reader/writer boundaries
  delegate to pinned PDAL. B0004 records the host checkpoint and B0005 records
  the faster automatically selected CUDA checkpoint with the same oracle
  output hash. D0011 extends that slice into the in-process hybrid executor and
  ordered filter/compaction path rather than treating a standalone LAS chain
  as the product architecture.
- Corpus: the ignored `build/local-corpus.json` currently indexes 4,550 files
  (152,501,521,208 bytes), including 217 LAS/LAZ/COPC candidates across eight
  valid header strata and four deliberately or accidentally truncated files.

## F0 — Fork and baseline

Status: fork topology and the 142-test published baseline are complete for the
available dependency set; the full optional-plugin matrix remains.

- Seed `gpu-main` from the current `PDAL/PDAL` default branch and retain the
  official repository as `upstream`.
- Pin the exact oracle commit, compiler/dependency manifest, locale, timezone,
  and command-line environment.
- Create the remote fork when GitHub authentication is available; configure it
  as `origin` without changing `upstream`.
- Add agent guidance, decisions, benchmarks, compatibility policy, and corpus
  provenance rules.
- Capture unmodified upstream build/test results before enabling PDG targets.

Exit: the upstream commit is reproducible, local branch topology is correct,
and a fresh session has enough written guidance to continue safely.

## P0 — Executable scaffold and exact LAS vertical slice

Status: in progress. The exact mapped host writer, canonical modern-record
copy path, conservative CLI selection, device-runtime exactness, CUDA
sanitizers, overlap profiling, and full-PDAL fallback shell are complete on the
reference RTX 4090. Pure translation remains host-selected because it wins its
same-machine gate; compute-heavy assign/ferry/expression/range/crop regions
have a separate automatic CUDA envelope. The mapped-host checkpoint is B0002,
the first fused CUDA checkpoint is B0005, and D0011 defines the broader
execution architecture.

### P0.1 Build spine

- Add opt-in `WITH_PDG` integration without breaking the upstream PDAL build.
- Require CMake 3.28 for PDG targets, C++20, Ninja presets, and CUDA >= 12.4.
  Release fatbins use every real target supported by their selected compiler
  plus newest-target PTX. Maintain CUDA-12.x legacy-compatible and CUDA-13+
  current artifacts; run a compact oracle bit corpus on physical hardware for
  every architecture before calling that SM exact-supported.
- Add `libpdg`, `pdg`, unit tests, differential tests, and benchmark target
  groups. Make host-only validation possible when a GPU driver is unavailable.
- Add formatting, warnings-as-errors for PDG, ASan/UBSan preset, CUDA compile
  checks, and CI jobs for host, CUDA-container compile, and self-hosted GPU.

### P0.2 Core data model

- Implement the canonical dimension registry with PDAL names/types and typed
  custom dimensions. String lookup ends at plan time.
- Implement exact LAS XYZ scale/offset metadata and checked global/local
  coordinate conversion.
- Implement lazy structure-of-arrays `PointBatch` columns, capacity/size/ghost
  invariants, and explicit host/device residency.
- Implement RAII streams/events, pinned buffers, and the stream-ordered device
  pool, with a safe classic allocator for devices that report no memory-pool
  support. Add allocation accounting and a 20 GiB planning ceiling.

### P0.3 LAS oracle and GPU path

- Cover LAS 1.0–1.4 and point formats 0–10, including VLRs, EVLRs, Extra Bytes,
  waveform fields, CRS records, and malformed/truncated cases.
- Capture upstream `pdal translate` output as the oracle. Decode fixed-stride
  records into touched SoA columns, then repack on device.
- Preserve or reproduce every header and record byte in compatibility mode.
  Diagnose any non-semantic writer bytes rather than normalizing them away.
- Pipeline pinned host buffers across upload/compute/download streams.

### P0.4 Differential corpus runner

- Discover local LAS/LAZ/COPC fixtures without modifying or committing them.
- Hash and stratify fixtures by version, point format, record count, CRS/VLR
  traits, scale/offset, bounds, and corruption status.
- Compare files, metadata, diagnostics, and exit status byte-for-byte; emit a
  field-aware first-difference report and a small named regression recipe.

Exit: `pdg translate in.las out.las` matches the pinned upstream PDAL output
byte-for-byte across the P0 matrix, CI is green, and parse/repack benchmarks are
recorded. The performance target remains at least 5 GB/s on the reference rig.

## P1 — Per-point planner, fusion, and exact reductions

Status: in progress. Typed DAG planning, liveness/residency accounting, the
  exact expression VM, assign/ferry lowering, expression/range predicates,
  single-bounds 2D/3D crop, decimation/head/tail ordinal selection, exact
  first-tie locate reduction, simultaneous-XYZ transformation, exact IQR/MAD
  robust selection, typed normal/stable sort, exact ordinary/reverse Morton
  order, categorical groupby and return partitioning across multiple input
  views, exact merge composition, count-based divider and persistent XY
  splitter partitioning, exact stateful GDAL color interpolation, an audited
  exact randomize ordering bridge that preserves later native regions, exact
  ordered stats metadata with an audited default host bridge, stable
  host/CUDA compaction, adjacent one-pass fusion,
native LAS selection, and deterministic generated oracle matrices are
implemented. Compatible regions now execute inside otherwise unchanged PDAL
pipelines through a bounded batch-stream interface, while the direct LAS route
uses two-lane overlap, exact filtered output, and profile-guided selection. The
remaining P1 stage families and 70%-of-roofline exit gate are not complete;
  hardware-counter access was unavailable, so no roofline percentage is
  claimed. Ordinal, locate, transformation, IQR, MAD, and sort host gates are
  green; their ASan matrices are green. Transformation, IQR, and MAD now also
  pass physical process and all-four-sanitizer gates, but clean end-to-end
  measurements keep them host-selected standalone. Sort, Morton, groupby,
  returns, divider, and splitter
  also pass their physical device properties, forced complete-process
  differentials, and memcheck/initcheck/racecheck/synccheck primitive gates.
  Exact artifact-set benchmarking now covers numbered multi-view writers.
  Complete-process diagnostics choose deliberately: standalone sort is
  effectively tied with the faster host path at 21,970,934 points; groupby,
  returns, divider, and splitter remain host winners; ordinary Morton is 1.481x
  pinned PDAL over five 4,000,000-point samples versus 1.022x for PDG host, and
  reaches 1.936x on 21,970,934 ALS points and 2.032x on a 36,772,046-point LAZ
  tile. B0006 supersedes that Morton diagnostic with a clean option-free gate:
  1.248x ordinary and 1.487x reverse at the 2,000,000-point threshold, and
  1.528x ordinary at 4,000,000 points, with exact artifacts and profile proof.
  The 39-case color-interpolation matrix, its physical forced-CUDA lane, and
  all four device sanitizers are also green; a negative clean end-to-end gate
  keeps standalone selection on host. The 14-case randomize composition and
  22-case stats matrices pass in
  Debug, Release, and ASan/UBSan. Locate, robust-statistics, sort, Morton,
  groupby, returns/merge, divider/splitter, color interpolation, randomize
  composition, and stats
  large-corpus exactness is also green. Exact `filters.info` and
  `filters.expressionstats` host stages, a fused XYZ bounds/count reduction,
  and a predicate/select/radix/reduce-by-key histogram are now implemented.
  Their 20-case and 19-case process matrices pass in Debug, Release, and
  ASan/UBSan, both CUDA properties compile in Debug and Release, and both host
  pipelines match the 21,970,934-point local oracle artifacts exactly.
  Expressionstats additionally passes a ten-case forced-CUDA process matrix,
  the physical 131,103-point property, and all four direct Compute Sanitizer
  tools. Exact dirty-tree complete-process diagnostics measure 1.260x pinned
  PDAL at 1,000,000 points over ten samples and 2.346x at 21,970,934 points;
  a 36,772,046-point LAZ trial is exact at 2.375x. Clean B0008 supersedes the
  selector calibration with option-free exact medians of 1.126x at 1M/three,
  1.212x at 2M/two, and 1.105x at 4M/one, while all three below-threshold
  controls remain on host. Standalone forced CUDA stats and info lose to pinned PDAL at
  21,970,934 points (0.468x and 0.859x), so they remain host-selected and are
  plan-residency/fusion targets rather than false GPU promotions.
  P2 has started with a planner-owned compact 2D/3D Morton cell table, exact
  bulk radius counts and scaled outputs, deterministic capacity-driven
  core/ghost tiling for radius-bounded queries, bounded exact kNN and ordered
  mean distances,
  fused covariance/eigensystems, invalidation/reuse accounting, and persistent
  28-byte grid/76-byte worst-case adaptive-kNN index estimation. Both modes of
  `filters.outlier`, `filters.radialdensity`, and `filters.nndistance`, plus the
  current kNN envelopes of `filters.normal`, `filters.eigenvalues`, and
  `filters.covariancefeatures`, consume the common query backend.
  `filters.approximatecoplanar` uses the same shared eigensystem with its
  upstream direct, self-inclusive `3 <= knn <= 64` request and projects the
  result as a typed unsigned-byte `Coplanar` column. Covariance eigensystems now
  project normal, eigenvalue, algebraic covariance-feature, and coplanar
  columns directly into the resident device SoA batch. `Omnivariance` and
  `Eigenentropy` retain an explicit exact host-transcendental bridge because
  CUDA `cbrt`/`log` did not pass the binary64 gate; that bridge copies only
  three eigenvalues (24 bytes/point) rather than the former 96-byte/point full
  eigensystem and uploads its exact result columns. Maximal compatible runs
  attach one whole-view device XYZ batch and spatial index to the PointView,
  reuse equal-k eigensystems, and can pass resident feature columns directly
  into an adjacent assign/ferry program while gathering only missing standard
  dimensions. Current stage boundaries still publish their selected columns
  to the host PointView for transactional PDAL semantics. Radius stages can
  now use deterministic half-open XY core ownership, closed radius ghosts, and
  owner-only source-order mosaics when the conservative live-VRAM budget is
  exceeded. Pinned execution reuses two independent host/device allocation
  lanes and completion events so host gather, transfers, index/query work, and
  owner publication can overlap without sharing mutable storage. Pageable
  staging conservatively reuses one lane. Tiling does not yet preserve a
  plan-wide resident region and does not establish exact kNN tiling.
  Twenty-seven spatial/covariance/eigen/distance/tile contract tests and the
  14-case outlier, 10-case radialdensity, 13-case nndistance, 12-case normal,
  13-case eigenvalues, 18-case covariancefeatures, and 26-case
  approximatecoplanar process matrices pass in Debug and Release. Two physical
  device-column properties, three extended
  all-feature process differentials, and a forced
  neighborhood-column-to-point-program differential are exact. Under the
  CMake-injected, prescribed leak-disabled differential mode, ASan/UBSan
  passes 266 of 267 full-tree registrations and only the opt-in local-corpus
  case is skipped; the separate leak lane remains.
  The current neighborhood process matrices and
  both composed resident-region differentials pass with the sanitizer runtime
  loaded into oracle and candidate. CUDA Debug and Release compile the
  131,103-point radius, 8,323-point tiled-radius seam, and 4,099-point
  kNN/mean/distance/covariance/eigen properties plus forced process
  differentials. CUDA Release passes 384 executions plus the expected corpus
  skip in 385 registrations on the physical RTX 4090, including whole/tiled
  outlier and radialdensity, composed resident reuse, direct-column-consumer,
  and all eight approximate-coplanar process differentials. The two new
  approximate-coplanar CUDA unit paths and four complete process cases are
  clean under all four Compute Sanitizer tools; max-`knn` and forced tie repair
  now pass all four tools as well. Tie repair and resident
  composition each retain exact output across five consecutive runs. The
  broader shared grid/BVH suite, scaled-radius and core/ghost tiling
  primitives, projection kernels, complete resident consumer, and complete
  forced-tiled radius pipelines retain their prior clean sanitizer evidence.
  Exact dirty-tree diagnostic trials now cover ALS and TLS, LAS and LAZ, and
  250,000 through 36,772,046 points. The five-sample 21,970,934-point ADKLR
  forced-tiled run is 8.164x pinned PDAL with identical output on every run;
  independent snow-road-twin and VEIL LAZ trials are exact at 7.335x and
  20.228x. These are not release benchmarks or selector evidence while the
  tree is dirty. Tiled multi-stage residency, adaptive exact kNN tiling,
  complete option coverage, clean-tree density-stratified trials, and accepted
  stage-level break-even gates remain pending in this checkpoint.

  The proposed D0049 default selector is tracked independently from those
  forced/resident gates. Its clean exact 262,144-point forced LAS and
  point-format-8 LAS measurements, plus one dirty-tree LAZ diagnostic,
  establish only a candidate boundary on the RTX 4090. Acceptance still
  requires a clean LAZ forced baseline, clean option-free execution with no
  selection environment, the automatic-path proof guard, a retained runtime
  profile, and exact below-boundary/runtime-gate fallback evidence. Other
  compiler-supported NVIDIA architectures remain force-only qualification
  targets until the corresponding Vast physical exactness and crossover lanes
  pass.

  The current P2 implementation includes exact multi-bridge whole-view
  residency: normal, nndistance, eigenvalue, and approximate-coplanar clients
  may span coordinate-preserving assign/ferry bridges. The typed `Coplanar`
  byte can be consumed directly by a resident point program. Ambiguous
  eigensystem rows are
  repaired by the upstream KD3 order before reuse; this is
  force/experimental-only, and full-system repair transfers must be
  reduced/profiled before any automatic gate.

1. Parse PDAL pipeline JSON into a typed DAG and stage descriptors.
2. Implement dimension liveness, residency transitions, VRAM estimation,
   deterministic radius tiling/ghost ownership, adaptive kNN tiling, overlap,
   and stats reporting.
3. Implement expression parsing and a typed bytecode VM. Match upstream
   parsing, coercion, NaN, error, and `where`/`where_merge` behavior exactly.
4. Fuse adjacent per-point operations while retaining the oracle operation
   order in compatibility mode.
5. Deliver assign/ferry/colorinterp/transformation, bbox crop and compaction,
   decimation/head/tail/locate/iqr/mad, sort/order/mortonorder/randomize,
   partitions, merge, stats, info/expressionstats, and the `pdg bench` verb.
6. Add strict and fast benchmark lanes; never compare one mode against the
   other as if they had identical semantics.

Exit: all P1 artifacts match the oracle; fused pointwise chains reach 70% of
applicable DRAM roofline on the reference GPU or carry a decision record.

## P1.5 — Resident execution

Status: in progress and the exclusive architecture milestone before P2 breadth
resumes. D0050 freezes new stage-local CUDA ports. D0051 accepts the D1
planner/allocator liveness contract, D0052 accepts only the first
cardinality-preserving D2 fusion slice, D0053 fixes two lanes for every
implemented D3 scheduler class, and D0054/D0055 retain 52/52 measured-winner
accuracy for the frozen D4 calibration matrix. D0061 completes both V1 cases
by adding the declared cardinality-changing expression path to the generic
resident executor. None of those checkpoints closes P1.5.

D0056 adds a production-default-off `pdg resident` integration observer. The
exact-profile V7 negative control passes. A bounded V4-shaped process probe
selects two point-program wrappers around an unsupported `filters.randomize`
stage and remains byte-identical to pinned PDAL, but its crossings are inferred
from stage-local wrapper transfers. The wrapper does not consume the D1
allocation schedule or D3 two-lane executor and is not the direct fused-LAS
executor calibrated by B0005. Therefore D4, V4/E2, and executable calibrated
placement remain open; the ordinary `pipeline` command is unchanged.

D0057 connects one planner-selected terminal assign/ferry region to the actual
B0005 direct fused-LAS executor. The runtime D1 budget now bounds the shared D3
schedule, stats report its two-lane/peak/reuse state and exact transfer totals,
and the hash-pinned physical process is byte-identical to PDAL. This is the
first executable placement whose selected calibration matches its executor;
it remains production-default-off and does not make the diagnostic wrapper or
a mixed host boundary calibrated.

Completed terminal bridge:

1. Route one selected terminal `readers.las` → assign/ferry region →
   `writers.las` through the existing B0005 direct fused executor without
   changing normal pipeline selection.
2. Supply the planner's D1 VRAM budget and D0053 fixed two-lane request to that
   executor, return its active-lane/peak/reuse schedule, and report it in
   `--stats`.
3. Fail closed before publication for every other topology, layout, profile,
   program, memory, or output case; retain atomic no-overwrite publication.
4. Prove exact stdout, stderr, exit status, and output bytes against pinned
   PDAL on a hash-registered local LAS above the calibrated threshold.

Immediate next slice, test first:

1. D0058 defines a resumable planner-owned tiled device batch that pauses at an
   unsupported stage, executes per-stage `deviceMaterialize`/`deviceRelease`,
   and retires spill-live columns only after tile completion events.
2. D0058 preflights the exact graph/layout/schedule, allocates both packed
   lanes, and probes their planned column high-water sequence before writer
   execution; rejection runs the untouched host JSON.
3. D0059 gives the generic full-record executor a complete stable-ID fact table
   for transfer, pack/repack, staging bytes, and the D3-fixed lane width without
   changing the frozen direct-LAS formula. Its VRAM estimate uses the shared
   scheduler's active-width rule and the planner's plan-wide liveness peak;
   unsupported rewrite envelopes never receive its facts. V4's predicted and
   observed crossings, lane count, and peak VRAM now agree and E2 is satisfied;
   the generic compute residual remains explicitly uncalibrated, so D4 stays
   open.
4. D0060 forces natural over-budget V5 tiling through the same resumable
   interface. The 21,970,934-point full working set exceeds a lower-only
   256-MiB validation budget while the fixed two-lane tiled peak fits; its
   exact output matches the pinned PDAL standard-mode untiled reference. V5
   and E3 are satisfied without changing production defaults or neighborhood
   coverage.
5. D0061 completes V1's cardinality-changing expression case through the
   resident interface. A single descriptor-declared pure, deterministic-safe,
   order-preserving `PredicateProgram` may terminate a fused resident region;
   the executor evaluates it into a planner-owned per-tile keep mask, spills
   the complete input tile plus one mask byte per point so boundary
   accounting stays exactly predictable, and appends survivors to the output
   view in stable source order. The hash-pinned 21,970,934-point process gate
   keeps 14,671,481 survivors byte-identically to pinned PDAL twice, while a
   where-bearing control is rejected by declared semantics and delegates
   exactly. V7 and the direct terminal bridge remain the controls; the
   predicate compute residual is explicitly uncalibrated, so D4 stays open.

B0012 records the first B2 exact end-to-end acceptance lanes for the
resident interface on the hash-pinned 21,970,934-point fixture: 16.668x for
the direct calibrated fused chain through the resident command, 1.254x for
the generic executor's cardinality-changing expression region, and 0.885x
for the four-crossing mixed randomize class, all byte-identical to pinned
PDAL with per-run stats-based executor proof. The mixed-class deficit was
the CPU-side full-record boundary packing surface that the D4 facts already
predict. D0062 rebuilds that surface on physical rows — resident-selected
executions run against a row-backed table, the packed tile layout is the
row layout with a fail-closed offset-partition guard, and lanes serialize
whole rows in one copy per point — after phase accounting showed 16.1 s of
host packing against 2 ms of device waits. Clean B0013 re-runs the same
lanes byte-identically: the mixed class moves to 2.879x, the expression
class to 1.448x, and the direct control to 19.577x.

D0063 fuses native LAS endpoints: a declared compacting writer-prologue
fusion candidate routes the terminal expression chain onto the validated
ordered direct sink, lifting that B2 class from 1.448x to 18.038x (B0014)
while mixed pipelines keep their host middle stages and the generic
keep-mask executor stays covered by the v1 gate's randomize control. The
ordered executor's compute residual is explicitly uncalibrated.

D0064 calibrates the ordered executor as the measured
`ordered-point-program` model (clean linear fits, audit 58/58, the
calibration-match flag flipped true the proper way); D0065 records the
provenance waiver for the seven reboot-lost D0049 reports and fixes report
storage under `build/benchmarks/`. D0066 completes V2: the planner admits
single-stage shared-index neighborhood regions (approximatecoplanar first)
anchored on their measured model with point-program bridges at zero
incremental cost, and the resident executor runs them as delegated
whole-view regions on the exact D0045 shared-index path — validated
byte-exact over the 21.9M-point fixture by the `v2` process gate
(`planner_resident_shared_index`, `whole_view_neighborhood`, peak lane
5.45 GB within budget).

D0067 completes V3: the engine gains the exact three-pass LOF primitive
over one retained shared-kNN adjacency with closure-exact host tie repair,
`filters.lof` compiles onto the measured `lof` model (250k-floor envelope,
audit 64/64), and the v3 gate validates the 21.9M-point fixture byte-exact
through the same delegated whole-view region machinery as V2, with
calibration lanes measuring 2.8–5.2x over pinned PDAL.

D0068 completes V6/E4 (dirty-index rebuild accounting, predicted ==
observed == 2 in --stats) and B0015 satisfies E6 with resident B2 lanes
(approximatecoplanar 1.809x, LOF 4.576x, byte-exact). D0069 records the
first ncu evidence (B1 rooflines; E7 marginal fused-op cost measured at
zero registers/occupancy/DRAM delta). D0070 stages the fused-chain
kernels to 52% of DRAM peak and attributes the residue to interpreter
issue cost; D0071 closes E1 by NVRTC-specializing each fused program
into a straight-line kernel at 89.4% of DRAM peak, byte-exact. P1.5 is
complete: V1–V7, E1–E7, and B1/B2/B3 all hold with recorded evidence.

Post-P1.5, in order:

1. ~~The resident neighborhood kNN gather kernel.~~ Closed for the
   uniform-grid backend. D0073 rebuilt the dense regime around the
   consumer-Ada FP64 pipe (Morton-ordered queries, register worst-entry
   quick reject, contiguous sorted candidates, certified float
   prefilter); B0016 exposed the 22M outlier tail those 4M figures
   hid; D0074 closed it with a device shell budget feeding the exact
   incomplete-row host repair contracts (eigen family pre-existing,
   NNDistance and statistical outlier per-row, LOF with a
   two-hop closure). Net: the 22M gather kernel 32.2 s -> 0.49 s and
   the B0017 lanes reach 3.962x (coplanar) and 8.459x (LOF),
   byte-exact. The BVH gather keeps its bound-driven traversal.
   Placement recalibration for the now-faster families is deferred to
   item 4's calibration work.
2. Catalog-wide stage ports strictly through the resident interface
   (spec §6), watching the two known fusion failure modes:
   register-pressure occupancy cliffs and dimension-footprint DRAM
   growth. First tranche done (D0075/B0018): normal, eigenvalues,
   covariancefeatures, and nndistance execute under the resident
   planner byte-exactly at 3.96-4.36x with measured models (audit
   88/88). Remaining breadth needs new device primitives
   (estimaterank, optimalneighborhood, neighborclassifier, the
   grid/morphology families), not admission work.
3. ~~Extend fused-program JIT specialization beyond the format-7/8,
   36-byte-stride LAS envelope.~~ Done (D0076): the NVRTC generator
   specializes every host-admitted record format (0-3, 6-8, format 8 at
   its native 38-byte stride for the first time) with a straight-line
   constant-format decode mirroring the interpreter, proven bit-exact
   under a required-JIT format matrix; the identity path keeps its
   minimal source and 17.5x 22M lane. Endpoint-fusion admission
   (which pipelines reach the native executor) stays with the catalog
   work.
4. ~~Per-transfer H2D/D2H observation + shared-index executor
   calibration.~~ Done (D0077): the attach machinery records every
   physical transfer as observation events, all six shared-index models
   are re-measured against the resident executor itself (36 byte-exact
   ladders, audit 87/87, coefficients refreshed for the D0073/74
   kernels), and the calibration-match flag reports true for whole-view
   executions; boundary_batch remains honestly false pending its own
   calibration.

Catalog breadth after the roadmap items (all through the resident
interface, all byte-exact on the 22M fixture): D0075 ported normal,
eigenvalues, covariancefeatures, and nndistance; D0078 added
`filters.estimaterank`; D0079 added `filters.optimalneighborhood`,
completing the spec §7.5 flagship family; D0080 added
`filters.neighborclassifier`; D0081 added exact 2D/3D
`filters.radiusassign` selection on the shared radius index, with the original
ordered assignment-expression finale retained on the host; D0082 added exact
adjacent-predecessor `filters.label_duplicates` as a non-index resident
producer whose standalone CUDA gate is negative. Ten neighborhood stages still
  execute resident with measured, resident-provenance models (audit 126/126 over
   31 models), and label_duplicates can retain its output for downstream bridges
   without claiming a measured selector. D0099 supersedes the former
   catalog-order queue: no additional grid/morphology or HAG stage starts until
   the active performance-first work order above chooses it from a complete
   pipeline profile. Feature-kernel
prologue/epilogue fusion into neighborhood producers (D2's open half)
and a `planner_resident_boundary_batch` calibration remain the two known
engine-side gaps.

No catalog-wide stage port may bypass the resident interface.

## P2 — Shared spatial engine and feature family

Status: in progress. The persistent uniform-grid foundation, exact bulk radius
count and scaled output, deterministic capacity-driven core/ghost radius
tiling, bounded exact kNN, ordered mean-distance query, fused ordered
covariance/eigensystem primitives, planner build/reuse/invalidation accounting,
device-memory estimate, both outlier clients, exact `filters.radialdensity`,
both `filters.nndistance` modes, and bounded kNN `filters.normal`,
`filters.eigenvalues`, `filters.covariancefeatures`, and
`filters.approximatecoplanar` clients are implemented. The shared index also
provides exact source/reference-masked radius-any selection for
`filters.radiusassign`, including 2D vertical caps; its assignment expressions
retain the pinned host evaluator and synchronize written resident columns.
An exact adjacent-predecessor `filters.label_duplicates` pass also executes as
an index-free whole-view resident producer, retains its unsigned-byte output
for later point programs, and can precede the planner-owned index of a later
neighborhood stage. Its standalone placement remains host-selected because the
250K, 1M, and 22M complete-process CUDA rows are all slower than pinned PDAL
(B0020/D0082).
Approximate-coplanar
uses the upstream direct, self-inclusive neighbor count for
`3 <= knn <= 64`, strict threshold comparisons, and a typed unsigned-byte
`Coplanar` result; zero covariance preserves the previous byte and diagnostic.
Compatible feature runs retain one whole-view device XYZ batch, shared index,
equal-k eigensystem, and selected output columns across stage boundaries. An
adjacent exact assign/ferry program consumes those
columns in-place on device and gathers only standard columns that are not
already resident. The
compact table uses 63-bit 3D
Morton keys, stable radix ordering, run-length cell compaction, and exclusive
cell offsets on CUDA. An adaptive Morton-BVH backend now reuses that stable
ordering, builds an implicit binary hierarchy of outward-rounded local
  coordinate bounds, and provides exact radius/kNN traversal across sparse gaps
  and mixed density. A bounded deterministic hashed occupancy probe selects it
  only for broad clustering with an estimated hot-cell population above the
  measured threshold, while the planner budgets its 76-byte/point worst case.
  Both host and CUDA implementations have passed current physical-device and
  shared-index sanitizer qualification. The dirty-tree synthetic G1 diagnostic
  selects the measured winner across the recorded grid/BVH matrix, but a clean
  real-corpus and end-to-end PDAL run is still required before G1 is accepted.
  Algebraic covariance features are device-native. `Omnivariance` and
  `Eigenentropy` still use the documented 24-byte/point exact host bridge, and
  every stage still publishes selected columns at its host PointView boundary.
  Radius tiles use one stable source-order owner, conservative closed ghosts,
  complete owner-only mosaics, and two reusable event-governed pinned/device
  lanes; the current executor remains stage-local. Plan-wide tiled residency,
  adaptive exact kNN tiling, SVD and prefix covariance, the remaining
  neighborhood options and clients, stage-level break-even, and G6 are not
  complete.

1. Build fp32 local frames from exact int32 coordinates, Morton codes, CUB
   radix ordering, compact cell tables, and planner-managed index lifecycle.
2. Implement the uniform grid and LBVH behind the one device query API.
3. Implement stable covariance, eigensolver, SVD, prefix covariance, Philox,
   and fixed-order exact-mode reductions with independent math tests.
4. Deliver normal/eigen/feature filters, outlier/LOF, sampling,
   neighborclassifier/radiusassign, duplicate labeling (done through D0082),
   scan lines, voxel filters, and reprojection/projpipeline.
5. Benchmark uniform ALS, clustered TLS, and mixed density for G1. Run
   continental-scale exact/error fixtures for G6.

Exit: the exact lane passes; `normal` meets the 100 M-point target; G1 and G6
are recorded with raw benchmark and accuracy artifacts.

## P3 — Terrain, grids, clustering, geometry, and raster output

- Deliver SMRF, PMF, CSF, ELM, skewness balancing, DEM/HAG, grid decimation,
  cluster/DBSCAN, polygon crop/overlay/distance, faceraster, hexbin,
  `writers.gdal`, and `ground`/`density`.
- Use deterministic integer or fixed-order accumulation in compatibility mode.
  Fast atomics stay behind `--fast` when their bytes differ.
- Test tiles alone and mosaicked, with boundary-heavy halo/seam cases and exact
  raster metadata/block-layout checks.

Progress: D0084/B0022, D0085/B0023, D0086/B0024, and D0087/B0025 deliver
bounded exact PMF, serial-oracle CSF, ELM, and skewness-balancing vertical
slices. D0083/B0021's SMRF device qualification is withdrawn by D0221: its
exact KD2Index host wrapper remains, while the compiled CUDA prototype fails
closed. The canvas stages include host/CUDA primitives,
descriptor/compiler contracts, conservative hybrid fallback, standalone
planner-selected regions, deterministic process matrices, physical sanitizer
evidence, benchmark pipelines, and negative standalone placement gates. Their
qualified device canvases are limited to 4,096 cells. PMF morphology is
limited to radius 64 and 64 schedule passes, and CSF to `smooth=false`, a serial
non-OpenMP oracle, and 64 iterations. ELM preserves the pinned fractional-cell
expression, stable equal-Z source order, and strict threshold comparison.
Skewness balancing adds a narrower exact hybrid: CUDA may produce the unstable
Z-sort permutation only when binary64 logical Z values are finite and
comparator-unique; the pinned sequential moment recurrence, sign-crossing
classification, full-record permutation, and class publication remain on the
host. B0025 is positive on its controlled unique-Z fixture, but this
data-dependent envelope has no automatic model or resident region. None of
these bounded stage slices closes its catalog item or any P3 exit. D0088/B0026
adds the first provisional planner-owned tiled `RasterGridProduct` with PMF as
its single-region consumer: deterministic one-cell halos, owner-only mosaics,
phase barriers, cumulative device/pinned-host budget checks, and named
edge/corner/seam differentials are exact beyond the old 4,096-cell cap on its
tie-safe fixtures. Equal-distance void-fill candidates with distinct minimum
bits fail closed, adjacent Grid bridges and automatic placement remain off,
and the product is not reusable across stages. B0026 is a deliberately
negative 0.026602x prototype gate because each raster tile scans every point.
D0089/B0027 replace that scan with one exact selected-source-order raster
construction per execution in the product's two already-budgeted host
backings, with first-bit minimum semantics, fail-closed distinct-bit nearest
ties, one producer generation, one `RasterBuild` observation, and
exception-safe resident completion. The exact dense four-tile fixture reaches
1.828354x PDAL. D0090/B0028 then declare two planner-owned device phase
backings and materialize them only after the sole raster generation passes its
tie proof and while the complete pair fits the runtime Grid budget. Fitting
frames upload the completed raster once, keep every morphology/filter phase on
device, and download no phase surface; larger frames retain the exact tiled
host-mosaic path. The already-budgeted tile scratch also becomes an exact
column-major populated-source list when it fits, eliminating empty-slot scans
without changing nearest arithmetic or adding allocation. Exact dense and
sparse 65x65 fixtures reach 9.009503x and 4.764988x PDAL. The result still
neither admits an adjacent Grid bridge, cross-stage reuse, nor an automatic
model. D0091/B0029 reuse the second planned host backing as an exact occupancy
hierarchy for 256 or more sources, retain a bounded literal scan below that
selector, add raw-bit heterogeneous and nonfinite-center admission tests, and
record empirical near-linear work on named 257x257/513x513 sparse
distributions. The hierarchy has no worst-case subquadratic proof, so a
general scalability claim remains open. D0092/B0030 add a separately declared
16-byte-per-cell device proof workspace for fitting frames. CUDA preserves
first-source minimum bits, compacts original source cells, proves every target
against every compact source with exact equal-distance bit checks, and promotes
the same allocation into the two phase backings only after success. The
primitive rejects distinct-bit ambiguity before mutation and the wrapper then
falls back to pinned upstream on an untouched private view. Successful
candidates perform zero raster H2D/D2H transfers and record exact controlled
3.156x-38.950x results, but the proof remains proportional to target cells
times populated cells and admits neither an automatic model nor an adjacent
Grid bridge at that checkpoint. D0093/B0031 then admit one deliberately narrow
bridge: contiguous PMF stages with the same cell/Grid contract and exact JSON
return selection keep one resident `RasterGridProduct` and reuse its one device
phase allocation. They still rebuild and consume one exact raster generation
per stage because morphology overwrites both backings. The exact 4,225-point
65x65 pair records one product, a separately proved reused allocation, two
generations, zero raster transfers, exact output, and 1.039838x pinned PDAL in
the final symmetric-scope benchmark. The earlier 4.590078x primitive/full-
pipeline, 1.056201x asymmetric-teardown, and comparison-inclusive 1.016953x
measurements are superseded. Different return sources, cell frames, non-PMF
Grid consumers, semantic surface reuse, cross-kind cloth reuse, and automatic
placement remain rejected. D0094/B0032 add the first HAG slice: exact
`filters.hag_nn,count=1` on the planner-owned masked 2D index, with both backend
proofs, tie/incomplete host repair, nonfinite fallback, adjacent point-column
composition, and explicit 2D↔3D index rebuilds. Its controlled 1,000,002-point
unique-nearest pipeline is exact at 1.512959x pinned PDAL, but the envelope is
data-dependent and retains host-default placement. D0095/B0033 extend that
exact shared-index lane to `count=2`, including ordered inverse-squared-distance
interpolation, strict distance/bounds semantics, positive proofs for historical
one-ground and nonfinite-Z host repair, candidate-boundary tie/incomplete
repair, and adjacent resident bridges. Its controlled 1,000,002-point pipeline
is exact at 1.493330x pinned PDAL. Cross-kind PMF -> HAG2 -> PMF composition
uses three product regions and six explicit materialization boundaries; it
does not retain a RasterGrid across the HAG region. D0096/B0034 add a separate
exact `filters.hag_delaunay,count=3` lane: the planner-owned masked 2D query,
three-point Delaunator seed order, barycentric edge/overflow semantics, and
native or repaired adjacent-column bridges are exact at 1.314103x pinned PDAL
on the controlled 1,000,002-point fixture. It remains a forced,
data-dependent gate. D0097/B0035 extend the exact HAG-NN interpolation lane to
explicit `count=3`, including the pinned three-term inverse-squared-distance
operation order, strict cutoff/bounds behavior, real binary64 squared-distance
overflow, candidate-boundary tie/incomplete repair, and repaired resident
bridges. The controlled 1,000,002-point pipeline is exact at 1.482696x pinned
PDAL and remains a forced, data-dependent gate. D0098/B0036 extend the same
planner-owned lane to explicit `count=4`, preserve the pinned four-term
operation order, prove a distinct fifth candidate on both index backends, and
repair fourth/fifth boundary ties, grid incompleteness, nonfinite Z, and
insufficient grounds before publication. Its controlled complete pipeline is
   exact at 1.587001x pinned PDAL. The lane is GPU-native and the dirty-snapshot
   complete-process result remains useful diagnostic evidence, but D0099 does
   not treat it as performance-qualified or automatically selected. HAG-NN
   `count >= 7`, HAG
Delaunay `count >= 4` and its default ten, `hag_dem`, the remaining
   raster/overlay catalog, and the rest of this phase remain open. D0099 and
   `docs/stage-coverage.md` complete the implemented-stage audit under separate
   functionally-supported, GPU-native, performance-qualified, and
   automatically-selected classifications. The active queue is now the
   performance-first order above, driven by end-to-end profiles rather than
   stage count.

Exit: `dtm.json` matches the oracle byte-for-byte in strict mode, qualitative
and statistical invariants are also green, and the reference run is <= 30 s.

## P4 — LAZ, COPC, EPT, other native I/O, and bridge

- Build chunk-parallel lazperf decode/encode with ordered assembly in exact mode.
- Implement COPC hierarchy pushdown, halo selection, range reads, and writer.
- Implement EPT, PLY, PCD, text, and the explicit libpdal bridge.
- Decide G2 and G5 from cold-cache, same-machine trials. Preserve every
  third-party license and record lazperf provenance.

Exit: exact I/O corpus passes, decode saturates host cores while uploads
overlap, COPC spatial/resolution queries match, and gates are recorded.

## P5 — Heavy/global algorithms and full catalog

- Deliver registration, global sampling/clustering, M3C2, surface/mesh stages,
  global metrics, time/pose/georeference stages, then close every remaining
  data-parallel driver in the equivalently configured upstream catalog.
- For external-language, command, or service bridges where CUDA cannot replace
  the external operation, keep the bridge exact, optimize its transfer and
  scheduling boundaries, and preserve device residency on both sides whenever
  the graph permits it. Classify such a stage as GPU-inapplicable with evidence,
  never as a silent native success.
- Implement explicit spill/merge algorithms for datasets larger than VRAM and
  test the same dataset under multiple tile/budget configurations.
- Decide G3 and G4 only against the stated criteria.

Exit (revised by D0208): every extant stage in the equivalently configured
upstream build is functionally supported with oracle-identical default output,
optional-plugin configurations are covered, and every stage appearing in a
reference pipeline runs by the fastest correct backend measured for it —
CUDA, optimized host, or bridge. A nonzero upstream-fallback count is no longer
a defect: an exact host path that is fast is a finished answer. GPU-native
coverage of stages absent from the reference pipelines is optional and is
recorded as catalog coverage, not as a release blocker.

## P6 — APIs, packaging, conformance, and release

- Add Python builder and pipeline API, CUDA Array Interface, DLPack, GIL
  release, and explicit host copies.
- Publish per-stage parity/fallback/performance pages and a generated
  conformance report.
- Evidence slice complete: the named, versioned PDG Conformance Suite,
  bounded 2,048-case generator, opt-in canonical COPC comparator, frozen
  release-artifact manifest, packaged `pdg verify` JSON/HTML reporter, exact
  1.7x re-proof protocol, and preregistered 40-project 3DEP analysis are
  implemented. B0286 executes the exact frozen 1.7x artifact on all ten named
  machines at 1M, on seven named machines at 47.5M, and through the hardened
  2,048/2,048 conformance gate; the retained author-run `pdg verify` artifact
  is also exact and valid. The preregistered 3DEP data collection and
  unrelated-user reproductions remain open execution gates and are not
  represented as completed measurements.
- Build reproducible packages, SBOM, notices, deterministic regression corpus,
  benchmark bundle, npm installer, and release CI. GPUPAL resolves the working
  product-name choice; availability/trademark review remains required before
  publication (D0288).

Exit: `v0.1.0` packages install in clean supported environments, the exact
conformance suite and sanitizers pass, and every headline speedup links to a
reproducible benchmark record.

## Cross-cutting test lanes

- `unit`: pure parsing, math, registry, planner, and invariant tests.
- `diff-smoke`: tiny public fixtures and first-byte/field diagnostics per PR.
- `gpu-pr`: exact replay plus compute-sanitizer on a self-hosted GPU.
- `nightly-corpus`: stratified local corpus, option/pipeline combinatorics,
  performance regression, and repeated determinism runs.
- `weekly-stress`: exhaustive generated boundary matrices, large out-of-core
  data, fault injection, cold-cache I/O, and cross-toolkit/compiler matrix.

Implementation proceeds in phase order. Resequencing across phases requires an
append-only `DECISIONS.md` entry.

## Appendix A — Active whole-pipeline acceleration program

D0099 activates this program now rather than after the native CUDA catalog is
complete. New stage ports remain necessary for final coverage, but they no
longer outrank a measured reusable fusion, residency, I/O, or bridge win. This
work remains subject to the exact-output contract: an optimization that changes
bytes, metadata, diagnostics, ordering, or fallback behavior is not accepted in
compatibility mode.

Work in the following order, guided by same-machine profiles rather than by
microbenchmarks alone:

1. **Extend device residency across stage boundaries.** Keep complete point
   batches and live columns on the GPU across compatible stages. Remove
   spill/re-upload boundaries only after the producer, consumer, liveness,
   error, and fallback contracts are exact and covered by mixed-pipeline
   differentials.
2. **Reuse planner-owned products.** Reuse compatible spatial indexes,
   neighborhood results, raster grids, cloth/grid workspaces, and scratch
   allocations instead of rebuilding them per stage. Every reuse key must
   include the dimensions, options, source generation, and invalidation rules
   that affect semantics.
3. **Accelerate input and output.** Optimize LAS/LAZ/COPC/EPT decode, range
   selection, point packing, output assembly, compression, and metadata
   publication. Measure complete cold-cache and warm-cache processes so disk,
   decompression, and encoding costs are not hidden behind kernel-only timing.
4. **Pipeline independent work.** Use bounded batches, pinned queues, CUDA
   streams, and events so reading/decoding batch N+1, uploading and processing
   batch N, and downloading/encoding batch N-1 can overlap without changing
   global order or diagnostics.
5. **Remove allocation and synchronization overhead.** Expand persistent
   device and pinned-host pools, reuse stage workspaces, replace unnecessary
   device-wide waits with event-governed dependencies, and reduce repeated
   context/module/JIT setup. Preserve the allocator and RAII rules.
6. **Fuse region edges and lightweight consumers.** Fold exact packing,
   projection, assignment, ferry, predicate, and output-column work into
   surrounding resident producers or consumers when that removes round trips
   and stage-dispatch overhead. Do not fuse across an observable ordering,
   topology, diagnostic, or failure boundary.
7. **Improve whole-region placement and batching.** Calibrate startup,
   transfer, boundary, allocation, index-build, and I/O costs; select CPU or
   GPU using the complete region cost rather than a kernel time. Coalesce
   nearby profitable stages, retain small or GPU-negative work on the host,
   and tune batch sizes within explicit RAM/VRAM budgets.
8. **Attack fallback frequency without weakening proof.** Improve complete
   index searches, tiling, ambiguity detection, and exact repair scheduling so
   ordinary data stays on the accelerated path more often. Never replace a
   tie, incomplete result, or unsupported case with an unproved answer.

For each optimization, record the profile limiter before the change, an
equivalent complete-process PDAL baseline, exact output and diagnostic hashes,
transfer/allocation/synchronization counters, peak RAM and VRAM, and an
append-only `BENCHMARKS.md` result. Prefer improvements that help several
stages or remove a shared boundary over isolated microsecond reductions.

Exit: representative I/O-bound, neighborhood-heavy, raster/terrain, and mixed
catalog pipelines are oracle-identical; profiles show no avoidable full-device
waits, rebuilds, spill/re-upload pairs, or hot-path allocations; CPU/GPU
placement remains fail-closed; and every published end-to-end speedup has a
same-machine reproducible PDAL baseline.
