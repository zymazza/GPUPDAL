# `filters.nndistance`

Status: exact host replacement and shared-index CUDA implementation for both
distance modes in a bounded option envelope. Automatic CUDA selection is
limited to separately measured complete-pipeline compositions; it is not a
general standalone-stage promise.

The native envelope accepts `1 <= k < 64` and the case-insensitive modes
`kth` and `avg`. As in upstream PDAL, the query point occupies neighbor zero,
so the planner requests `k + 1` points and writes one binary64 `NNDistance`
value without changing point order or view identity. Device execution also
requires at least `k + 1` input points. `k=0`, `k >= 64`, too-small views,
`where`/`where_merge`, invalid option types or modes, unknown options, and
unproven layouts retain the unchanged upstream stage.

Compatibility mode preserves the upstream arithmetic sequence exactly.
`kth` applies one square root to the final squared distance. `avg` starts with
binary64 zero, adds `sqrt(distance[i])` serially for `i=1..k`, then performs
one division by `k`. The CUDA kernel uses explicit round-to-nearest binary64
square root, addition, and division instructions and is compiled without
floating-point contraction. Equal-distance ties are safe for this stage
because only distance values are observed; an incomplete bounded-grid search
is rejected before publication. The adaptive Morton BVH can prove completion
across sparse gaps while retaining the same candidate-distance arithmetic.

`NNDistance` is written directly into the PointView-owned resident device SoA
batch. Consecutive compatible neighborhood filters share the same XYZ upload
and planner-owned spatial index. An immediately following exact assign/ferry
program can consume `NNDistance` without uploading it again; only the
program's written columns return to the host. The current transactional stage
contract still publishes `NNDistance` to the host PointView once, and
kNN residency remains whole-view. The deterministic core/ghost executor added
for radius-bounded queries does not prove kNN completeness; adaptive kNN
tiling remains separate P2 work.

B0233 adds one automatic per-stage hybrid composition after adjacent duplicate
labeling. It requires the literal
`label_duplicates(Classification) -> nndistance(k=10) ->
assign(UserData = Duplicate)` graph between option-free LAS endpoints,
uncompressed LAS 1.4 format-7/36-byte input with no VLRs, EVLRs, or trailing
bytes, 250K--16M points, and the exact RTX 4090/SM89/CUDA 13.3/driver 610.43.03
profile. Final public medians are 2.846x pinned PDAL at 250K and 6.392x at 1M.
The 50K loss and every neighboring graph/layout/profile retain the original
host pipeline; recoverable per-stage CUDA failure uses the exact host
operation. This does not widen `k`, mode, or standalone automatic selection.

## Verification

Host units compare `kth` and `avg` against the literal upstream serial
operation order and cover empty input, null outputs, unsupported `k`, ties,
fixed rows, sparse incomplete grids, and exact Morton-BVH completion. The
4,099-point physical CUDA property runs both modes through both index backends
and compares every value and status byte with the host implementation.

A 13-case process matrix covers defaults, explicit modes and `k`,
case-insensitive parsing, duplicate/equal-distance inputs, insufficient and
empty views, both native-envelope boundaries, conditional fallback, invalid
mode, and string-valued `k`. A composed fixture runs
`normal -> nndistance(kth) -> ferry -> nndistance(avg)` and requires both
shared-index reuse and direct resident-column consumption; its complete LAS
artifact is byte-identical to upstream on the RTX 4090. That pipeline is clean
under memcheck, initcheck, racecheck, and synccheck, and its matrix/composition
are clean under ASan/UBSan with leak detection.

The original focused gates pass their published host, CUDA, sanitizer, and PDAL
matrices. B0233 additionally proves exact automatic selection, below-floor and
individual header/layout/device/recoverable-failure fallback, strict grammar
refusals, and existing/alias/symlink behavior through the public command.
Final maintained aggregates pass 407/409 Host Debug and leak-disabled
ASan/UBSan plus 599/601 physical CUDA Release; only the two documented
optional-corpus cases skip. The actual 250K automatic process is clean under
memcheck and racecheck.
