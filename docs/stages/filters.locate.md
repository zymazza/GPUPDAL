# filters.locate

`filters.locate` is implemented as the first reusable exact global-reduction
primitive. It returns the original first point whose selected dimension is the
requested minimum or maximum, preserving every dimension and the input point's
identity.

## Exact compatibility status

The replacement envelope accepts a required string `dimension` and an optional
string `minmax`. `minmax` is case-insensitive; `min` and `max` select a point,
while every other string produces the same empty view as PDAL. Missing or
unknown dimensions, non-string options, `where` variants, tagged/branched
graphs, and unproven input order remain on unchanged PDAL.

The reduction reproduces details that a generic numeric `min`/`max` primitive
usually loses:

- strict comparisons retain the first global point on a tie;
- NaNs never become candidates;
- PDAL's `DBL_MAX`/`-DBL_MAX` sentinels are preserved, including the all-NaN,
  all-infinity, and exact-sentinel fallback to input point zero;
- integer dimensions are converted to double before comparison exactly as
  `PointView::getFieldAs<double>` does;
- source order is a compatibility prerequisite because it determines ties.

The hybrid executor can place the reduction between two fused point-program
regions. This permits, for example, assigning a custom device-visible
dimension, locating its maximum, and then mutating only the selected original
point without returning the surrounding stages to a separate process.

## CUDA qualification status

The CUDA implementation uses a typed, grid-stride first pass and a second
fixed-size reduction. Its associative candidate record carries the value,
global index, input-presence flag, and sentinel/comparability flag, so arbitrary
chunk and block partitions retain PDAL's first-tie result. LAS physical
coordinates use separate round-to-nearest multiply and add operations before
comparison.

CUDA execution is currently force/require-only through
`PDG_EXPERIMENTAL_CUDA_HYBRID` or `PDG_REQUIRE_CUDA_HYBRID`. Automatic selection
remains disabled until the RTX 4090 process differentials, all four Compute
Sanitizer tools, and a same-machine break-even matrix pass. Host exactness and
CUDA compilation do not count as GPU qualification.

## Verification

Host units cover ties across chunks, minima/maxima, NaNs, infinities, exact
sentinels, empty input, invalid kinds, missing materialization, and scaled LAS
coordinates. CUDA host/device tests add large multi-block reductions, 64-bit
integer conversion, sentinel-only inputs, and coordinate decoding. A 15-case
complete-process matrix covers defaults, case-insensitive options, integer and
floating dimensions, NaNs, empty output, a custom-dimension point-program
chain, a predicate before reduction, and conservative fallback edges. Small
process tests force 17-point chunks so the selected point crosses reduction
boundaries.

A read-only local-corpus differential also scans 21,970,934 points from a
large local LAS corpus in 168 chunks. The candidate and pinned oracle emit
the same 2,301-byte one-point LAS with SHA-256
`44e4bf505c7f304ddba014a29f3b5480dce1ea04a2ccf918d2e8a1cd650d3095`.
