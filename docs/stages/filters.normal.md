# `filters.normal`

Status: exact kNN host replacement and shared-index CUDA implementation in a
bounded option envelope; automatic CUDA selection is not yet qualified.

The native replacement currently accepts `2 <= knn < 64`, Boolean
`always_up`, and `refine=false`. As in upstream PDAL, the query point is part of
its own neighborhood, so the shared index requests `knn + 1` points. The stage
writes `NormalX`, `NormalY`, `NormalZ`, and `Curvature` without changing point
order or view identity.

Compatibility mode preserves details that are visible in output bits. The
centroid is updated in source-neighbor order, each demeaned coordinate is
narrowed to `float`, sample covariance divides by `n - 1`, eigenvalues are
ascending, the normal is the first eigenvector, and curvature is the absolute
smallest eigenvalue divided by the eigenvalue sum. The CUDA backend compiles
the same fixed-size Eigen self-adjoint QR solver used by the pinned PDAL
implementation; it does not substitute a faster analytic eigensolver with
different rounding. `always_up` negates a normal only when its Z component is
negative. Zero covariance and eigensolver failures retain PDAL's skip behavior
and messages.

The CUDA query publishes explicit status for equal-distance ties and bounded
grid-search incompleteness. Because either condition can change the selected
point identities and therefore the covariance, the wrapper rejects the entire
device result before mutating the view and uses the exact PDAL-compatible host
path. Nonfinite coordinates, too few points, and compact-index/frame limits
are handled the same way.

`radius`, `viewpoint`, `refine=true`, `where`/`where_merge`, `knn` outside the
current device range, unknown options, and invalid combinations remain on the
unchanged upstream `filters.normal`. They are fully functional through `gpupal`
but are not reported as GPU-native. For a maximal consecutive run of
compatible normal, nearest-neighbor distance, eigenvalue, and covariance-
feature stages, the first wrapper uploads whole-view XYZ and builds the index
once. Later wrappers retain that device batch and index; an identical eigen
request also reuses the already-computed device eigensystem.
The CUDA projection writes `NormalX`, `NormalY`, `NormalZ`, and `Curvature`
directly into resident device SoA and copies those selected columns to the host
PointView for the current stage contract. If the next compatible stage is an
assign/ferry point program, it consumes the retained columns directly and
gathers only other missing standard dimensions. The final consumer releases
the region. Tiling and ghost ownership remain P2 work.

kNN construction chooses between the compact uniform grid and the shared
Morton BVH with a bounded deterministic hashed occupancy probe and a
physically measured clustered-input threshold. The BVH uses
outward-rounded binary32 local bounds only to prune; neighbor distances,
ordering, covariance, and eigensystems retain the exact binary64/PDAL
operation order. The composed CUDA differential forces this backend so it
cannot pass solely through the grid implementation.

## Verification

Five covariance/eigensystem contract tests compare every binary64 result with
the pinned PDAL/Eigen arithmetic, including ties, invalid shapes, and zero
covariance. Planner tests verify the normal descriptor, shared kNN request,
output dimensions, and conservative rewrite boundary. A 12-case process
matrix compares status, stdout, stderr, artifact sets, and every output byte
for defaults, nondefault k, orientation, duplicates/ties, empty input, radius,
viewpoint, refinement, conditional execution, both k boundaries, and the
invalid simultaneous `knn`/`radius` failure. It passes in Debug, Release, and
ASan/UBSan builds.

CUDA Debug and Release compile a 4,099-point host/device property that compares
every covariance coefficient, eigenvalue, eigenvector coefficient, and status
byte after crossing launch boundaries. A second property compares every
projected normal/curvature bit and sentinel-preserving skip. On the physical
RTX 4090, CUDA Release has 330 executed passes plus the opt-in corpus skip in
331 registrations, including the forced normal differential, the three-stage
resident differential, and a direct device-column-to-point-program
differential. The composed LAS artifacts are byte-for-byte identical to
upstream. The shared grid/BVH suite, projection kernels, and complete consumer
pipeline are clean under memcheck, initcheck, racecheck, and synccheck.
Automatic selection and a normal-stage performance claim remain withheld until
a clean real-corpus complete-process break-even matrix passes; the resident
synthetic G1 primitive diagnostic is not a substitute for that gate.
