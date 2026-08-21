# `filters.outlier`

Status: exact native host implementation; shared-index radius and statistical
CUDA paths physically qualified in their bounded envelopes but not yet enabled
automatically because complete-process break-even and option coverage are open.

The host implementation preserves both upstream methods, option defaults,
classification writes, view identity, empty input, no/all-outlier behavior,
warnings, and runtime failures. Default statistical execution retains the
pinned nanoflann kNN search. B0256/D0255 briefly invalidated incoming
PointView products and published the statistical tree for an adjacent
consumer; B0260 proved that inexact against pinned PDAL once a coordinate
mutator sat between the outlier and a later KD3 consumer, and B0261/D0260
restored pinned semantics: the statistical index is private and fresh, and
the view's products are neither read, invalidated, nor published. The private
tree uses the exact cached-coordinate backing (identical tree/traversal/tie
order; 24 transient bytes per point), so the literal r11 process measures
**5.207843x warm / 5.267489x cold** against pinned PDAL with exact bytes,
metadata, order, streams, and status. This is exact host work, not a new
automatic CUDA envelope or native stage. Forced/experimental execution can use the common
compact spatial index for both methods when the exact device envelope holds;
all other cases retain nanoflann.

B0258/D0257 additionally run the statistical method's per-point kNN pass over
fixed contiguous row chunks on host workers. Each row's mean neighbor distance
depends only on that row's query against the read-only fresh KD3 index, so the
per-row `sqrt`/online-mean recurrence is bit-identical at any worker count;
the global online-moment reduction, threshold, inlier/outlier split,
Classification writes, warnings, and radius mode are unchanged and serial.
The worker count is `min(ceil(points / 4096), hardware threads,
PDG_NATIVE_WORKERS)`; `PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS` restores the
serial pass. On the reference workstation the 1M-row `mean_k=8` pass falls
from about 3.37 to about 0.30 seconds, and the literal r11 process moves from
1.029892x to **4.309167x warm / 4.345954x cold** pinned PDAL, exact in bytes,
metadata, order, streams, and status. A three-pair same-final-binary probe
that forces the experimental CUDA statistical outlier on r11 remains exact but
is slower (4.069126x) because it drops the shared host tree the adjacent
classifier reuses and pays CUDA startup; it also doubles peak RSS.

The shared radius index builds 63-bit Morton cell keys, a stable point
permutation, and compact key/offset/count arrays. Its cell edge equals the
largest planned radius, so each query scans at most 27 cells. Candidate
distance is evaluated in binary64 and accepted only when `distance < radius²`;
self-points and duplicate coordinates are included exactly as in PDAL. The
planner shares a valid table between compatible consumers, invalidates it
after coordinate, point-set, or order changes, reports rebuilds, and budgets
28 persistent bytes per point for radius-only plans.

Radius mode also supports deterministic capacity-driven execution beyond the
whole-view device budget. Half-open XY core tiles assign one owner per source
point, closed radius halos provide every possible 3D neighbor, and only core
owners publish Classification in original source order. The planner rejects a
tile that exceeds capacity after ghosts rather than emitting a partial mosaic.
Pinned tiled execution reuses two independent host/device lanes. Completion
events protect owner-only publication and buffer reuse while host gathering,
transfers, index/query work, and the other lane may proceed concurrently.
Pageable staging conservatively uses one lane. The path remains stage-local;
plan-wide tiled residency remains open performance work.

CUDA uses CUB radix sort, run-length encoding, and exclusive scan, followed by
bulk radius counts or an expanding-cell kNN query. The kNN search retains up to
65 candidates, uses a conservative distance-to-unvisited-cells proof, reports
an incomplete status rather than guessing after 4,096 shells, and detects
equal-distance ordering. Statistical outlier accepts equal-distance status
because only the identical distances are observable, then applies the pinned
ordered `sqrt`/online-mean recurrence on device. Its current envelope is
`0 <= mean_k < 64` with at least `mean_k + 1` points. It is available only
through `PDG_EXPERIMENTAL_CUDA_HYBRID` or `PDG_REQUIRE_CUDA_HYBRID`.
For broad, expensive clustered kNN inputs, a deterministic hashed occupancy
probe of at most 8,192 points selects the exact Morton BVH instead. It requires
fewer than one occupied cell per eight probes and an estimated hot cell of at
least 2,048 points; smaller clustered and mixed-density inputs retain the
faster grid. BVH outward-rounded local-coordinate bounds are pruning-only;
candidate distances and order retain the same exact contract. Adaptive kNN
plans reserve a conservative 76 bytes per point.
Nonpositive/nonfinite radii, nonfinite coordinates, incomplete grid queries,
and views outside the compact 32-bit envelope use the exact host path.
Automatic replacement is disabled pending a clean density-stratified,
complete-process PDAL break-even matrix and completion of the remaining option
envelopes.

Current evidence includes 27 shared spatial/covariance/eigen/tile contract units, planner
reuse/invalidation tests, and a 14-case complete-process matrix in Debug,
Release, and ASan/UBSan. The leak-enabled sanitizer PDG registry has 175 executed
passes plus the opt-in local-corpus skip in 176 registrations. CUDA Release on
the physical RTX 4090 has 347 passes plus the same skip in 348 registrations,
including the 131,103-point radius property, the 4,099-point dual-backend
kNN/ordered-mean
property, the 8,323-point tiled seam property, and whole/tiled forced outlier
differentials. The shared grid/BVH and tiling suites and complete forced-tiled
pipeline are clean under memcheck, initcheck, racecheck, and synccheck. The exact synthetic
G1 diagnostic selects the measured backend winner, but it is not an accepted
outlier-stage speed result because it has no end-to-end PDAL baseline.
