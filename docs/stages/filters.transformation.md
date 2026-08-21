# filters.transformation

`filters.transformation` is implemented as a reusable exact coordinate-map
operation inside the fused point-program executor. It reads each point's
original X/Y/Z values, applies the upstream row-major 4x4 homogeneous matrix,
and writes all three results simultaneously.

## Exact compatibility status

The replacement envelope accepts an inline string `matrix` containing exactly
16 classic-locale doubles. `invert` may be absent or Boolean `false`. Host
execution implements the full projective form, including the fourth-row
homogeneous denominator, in the same multiply/add/divide order as PDAL.
Coordinate products are invalidated after the operation, and transformation
can be fused before or after assign, ferry, predicate, crop, and ordinal
operations without changing their order.

Matrix files, `invert:true`, `override_srs`, `where` variants, malformed or
non-string matrices, tagged/branched graphs, and non-double XYZ layouts remain
on unchanged PDAL. This preserves PDAL's Eigen inverse, matrix-file lookup,
spatial-reference warnings and metadata, and option/error behavior until each
has its own exact implementation.

## CUDA qualification status

The first CUDA envelope accepts affine matrices whose last row is exactly
`0 0 0 1` and materialized double XYZ columns. The grid-stride kernel uses
explicit round-to-nearest double multiply, add, and divide operations to avoid
contraction changing strict output. Full projective matrices already run on
the exact host operation but remain outside the device envelope until their
zero-denominator, infinity, NaN payload, and signed-zero behavior is qualified
against the reference compiler and GPU.

CUDA execution is force/require-only through
`PDG_EXPERIMENTAL_CUDA_HYBRID` or `PDG_REQUIRE_CUDA_HYBRID`. A transformation
region remains excluded from automatic standalone selection. Its RTX 4090
process differentials and all four Compute Sanitizer tools now pass, but the
clean complete-process break-even matrix is negative or effectively tied.
The exact affine kernel is retained for a resident/fused coordinate region
that can amortize allocation, transfer, and PointView publication.

## Verification

Unit tests cover exact matrix length, full homogeneous division, simultaneous
XYZ reads, affine/projective device classification, and unsupported physical
layouts. The CUDA unit compares every output bit for 131,103 points across a
multi-block affine launch. A nine-case complete-process matrix covers identity,
mixed affine coefficients, a 198,975-point multi-stream-batch input, full
projective division, an assign/predicate chain, empty input, inverse fallback,
SRS override fallback, and matrix-file fallback. The same matrix is in the
ASan/UBSan workflow. Forced affine process forms and the direct property report
zero errors or hazards under memcheck, initcheck, racecheck, and synccheck.

Clean commit `bd5841356` records exact CUDA at 0.357x pinned PDAL for 250,000
points and 0.851x for 4,000,000. A one-shot 21,970,934-point run reaches only
1.008x, below the automatic-selection margin, while the native host path is
0.883x there. D0042 therefore accepts the implementation and its negative
standalone gate without presenting the near tie as a product acceleration.
