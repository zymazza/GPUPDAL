# `filters.covariancefeatures`

Status: exact kNN replacement with CUDA-backed neighborhood, covariance,
eigensystem, and resident output columns in a bounded option envelope;
transcendental formulas, tiling, and automatic CUDA selection are not complete.

The replacement accepts `2 <= knn < 64`, `threads=1`, `stride=1`,
`optimized=false`, the `raw`/`sqrt`/`normalized` modes, and every named feature
except `density`. `dimensionality` expands to `Linearity`, `Planarity`,
`Scattering`, and `Verticality`; the additional accepted features are
`Omnivariance`, `Anisotropy`, `Eigenentropy`, `EigenvalueSum`,
`SurfaceVariation`, and `DemantkeVerticality`. String and JSON-array feature
lists preserve upstream case-folding, trimming, and output-dimension order.

The shared device backend performs the exact kNN query, ordered float-demeaned
sample covariance, fixed Eigen eigendecomposition, descending/nonnegative
eigenvalue mapping, and every algebraic feature formula. Selected output
columns remain in resident device SoA while also being published to the host
PointView for transactional stage semantics. A following exact assign/ferry
program consumes those device columns directly, so publication does not cause
a device-to-host-to-device round trip. When this filter follows a compatible
normal or eigenvalue filter, it reuses the retained XYZ, spatial index, and an
equal-`knn` eigensystem.

`Omnivariance` and `Eigenentropy` are explicit exceptions. A binary64 test
found CUDA `cbrt` one ULP away from the pinned host result, so compatibility
mode copies only the three eigenvalues (24 bytes/point), evaluates the original
`std::cbrt`/`std::log` formulas on host, and uploads those exact result columns
into the resident batch. This removes the former 96-byte/point full-eigensystem
copy but is still recorded as a host bridge, not a GPU-native formula.

`radius`, strided search, `threads` other than one, `optimized=true`,
`density`/`all`, `where`/`where_merge`, k outside the current range, and
invalid or unknown options stay on unchanged PDAL. In particular, density's
`OptimalKNN`/`OptimalRadius` prerequisites and optimized per-point
neighborhood sizes are not disguised as CUDA coverage. Tie, incomplete-search,
zero-covariance, and solver status is handled before output fields are
published.

## Verification

Planner tests pin all selected output dimensions, feature-mask lowering, the
shared `knn + 1` request, and every conservative rewrite boundary. An 18-case
process matrix compares complete artifacts for the three modes, the full
non-density feature family, string and array feature lists, explicit defaults,
ties/duplicates, empty input, radius/stride/thread/conditional fallbacks,
optimized and density-prerequisite failures, both k boundaries, and an invalid
mode. It passes in Debug, Release, and ASan/UBSan builds.

CUDA Debug and Release compile the shared 4,099-point covariance/eigensystem
property and a device-column property covering all three modes, every
device-safe feature, invalid status, and sentinel-preserving skips. Three
extended process differentials cover the complete non-density feature family
and the transcendental bridge. A composed fixture forces resident-context reuse
across normal, eigenvalues, and this stage; a second fixture forces an adjacent
point program to consume `Linearity`, `Curvature`, and `Eigenvalue0` in the
same device batch. Both force the adaptive Morton-BVH backend and compare exact
artifacts. CUDA Release on the physical RTX 4090 has 330 executed passes plus
the opt-in corpus skip in 331 registrations. The shared grid/BVH suite,
projection kernels, and complete resident consumer pipeline are clean under
memcheck, initcheck, racecheck, and synccheck. Automatic selection and a stage
speed claim remain withheld until a clean real-corpus complete-process
break-even matrix passes.
