# `filters.eigenvalues`

Status: exact kNN host replacement and shared-index CUDA implementation in a
bounded option envelope; automatic CUDA selection is not yet qualified.

The native replacement currently accepts `2 <= knn < 64`, Boolean
`normalize`, `stride=1`, and `min_k` (which upstream ignores when no radius is
set). As in upstream PDAL, the shared index requests `knn + 1` points because
the query point belongs to its own neighborhood. It writes `Eigenvalue0`,
`Eigenvalue1`, and `Eigenvalue2` in ascending order without changing point
order or view identity.

Exact mode retains the observable PDAL arithmetic: ordered online centroid,
float-narrowed demeaned coordinates, sample covariance, and the same fixed
3-by-3 Eigen self-adjoint QR solver. Normalization divides each ascending
eigenvalue by the sum using the upstream operation order. Approximate-zero
covariance and solver failure retain the original skip/error messages.

Equal-distance kNN ties can change the point identities selected by
nanoflann. The CUDA result therefore carries tie and incomplete-search status,
and the wrapper rejects the entire device result before mutating the view when
either is observable. Nonfinite coordinates, too few points, and compact-grid
limits also retain the exact host path.

`radius`, `stride` other than one, `where`/`where_merge`, `knn` outside the
current device range, invalid option types, and unknown options remain on
unchanged PDAL. Consecutive normal/eigenvalue/covariance-feature wrappers now
share one whole-view device XYZ batch and one spatial index. Equal `knn`
requests also share the computed eigensystem instead of launching or copying
it again. The three eigenvalue columns, including optional normalization, are
projected into resident device SoA and then published through exact host
`PointView` writes for the current stage boundary. An adjacent assign/ferry
program can read those resident columns without re-upload. Tiling and ghost
ownership remain P2 work.

The shared kNN context adaptively selects the compact grid or Morton BVH using
a deterministic hashed occupancy probe and a physically measured
clustered-input threshold. Both backends feed the identical ordered
covariance/eigensystem primitive; the BVH bounds are conservative pruning data
and cannot alter published neighbors.

## Verification

The common covariance/eigensystem contract tests compare all binary64
coefficients and status bits with pinned PDAL/Eigen arithmetic. Planner tests
verify selected dimensions and the shared `knn + 1` request. A 13-case process
matrix compares status, streams, artifact sets, diagnostics, and every output
byte for defaults, normalization, nondefault k, ignored defaults,
ties/duplicates, empty input, radius/stride/conditional fallbacks, both k
boundaries, and invalid option types. It passes in Debug, Release, and
ASan/UBSan builds.

CUDA Debug and Release compile the shared 4,099-point host/device property, a
bit-exact device-column property, and a forced complete-process differential.
CUDA Release on the physical RTX 4090 has 330 executed passes plus the opt-in
corpus skip in 331 registrations, including that differential, the composed
three-consumer case, and the direct device-column consumer case with forced
reuse assertions. The shared grid/BVH suite, projection kernels, and complete
consumer pipeline are clean under memcheck, initcheck, racecheck, and
synccheck. Automatic selection and a stage performance claim remain withheld
until a clean real-corpus complete-process break-even matrix passes; the
resident synthetic G1 primitive diagnostic is not that gate.
