# filters.colorinterp

`filters.colorinterp` is implemented as an exact PDAL-facing range/statistics
stage plus a reusable color-map primitive. It loads the same embedded or
user-supplied GDAL raster bands, derives the same bounds, and maps the selected
dimension to Red, Green, and Blue without changing point order.

## Exact compatibility status

The native envelope accepts the upstream `dimension`, `minimum`, `maximum`,
`clamp`, `ramp`, `invert`, `mad`, `mad_multiplier`, and `k` options with their
original types and defaults. The host implementation preserves upper-median
selection, online sample-standard-deviation order, MAD scaling, ternary clamp
behavior, outside-range RGB preservation, built-in ramp lookup, arbitrary
GDAL-readable ramps, and the original ramp-bin formula.

Execution mode is part of this contract. Explicit bounds retain PDAL's
streamable behavior through a bounded `BatchStreamable` implementation; in
that mode an explicit `k` is ignored exactly as it is by upstream `processOne`.
Inferred bounds use standard mode. Auto-range members persist after the first
input view, while nonzero `k` recomputes bounds for every view. Empty views,
one-point/zero-width ranges, existing colors, skipped stream points, and the
order of numbered multi-view outputs are covered by complete-process tests.

`where`, unknown options, invalid option types/ranges, missing dimensions, and
tagged or branched graphs remain on unchanged PDAL so their original
validation and diagnostics are retained.

## CUDA qualification status

The CUDA kernel accepts finite source values, finite increasing bounds,
equal nonempty RGB bands, and ramps whose flattened size fits `uint32_t`.
Values outside an unclamped interval leave existing colors untouched. The
kernel uses explicit round-to-nearest binary64 subtraction, division, and
multiplication before `floor`, matching the strict host bin calculation.
Range and MAD/stddev calculation currently remain on the host; only the color
map is transferred to the device.

CUDA execution is force/require-only through
`PDG_EXPERIMENTAL_CUDA_HYBRID` or `PDG_REQUIRE_CUDA_HYBRID`. Nonfinite values
and other unproved numeric domains use the exact host implementation. Physical
runtime differentials and all four Compute Sanitizer tools now pass. Automatic
standalone selection remains disabled because the clean end-to-end matrix is
negative: transfers, allocation, GDAL/range work, PointView publication, and
writing outweigh the exact map kernel. The device primitive is retained for a
resident region that can fuse it with adjacent pointwise operations.

## Verification

Host unit tests pin exact bin boundaries, maximum handling, clamp/invert,
outside-color preservation, invalid layouts, and the finite CUDA envelope. A
physical CUDA unit compares every RGB value with the host over 131,103 points,
including range and chunk boundaries. The 39-case process matrix covers all seven
embedded ramps, an external TIFF, auto/partial ranges, stddev/MAD, negative and
explicit `k`, NaN and one-point inputs, stream and standard modes, existing
colors, expression/sort/divider composition, consecutive stages, empty and
large-batch inputs, fallbacks, failures, and multi-view state. Forced CUDA
process tests cover explicit streaming bounds, auto-range standard mode, and
auto-range after a round-robin divider. The complete matrix passes in Debug,
Release, and ASan/UBSan builds. The direct property and forced process forms
report zero errors or hazards under memcheck, initcheck, racecheck, and
synccheck. Final aggregate gates record 246 passes plus one opt-in skip in 247
Host Debug and ASan/UBSan registrations, and 347 plus the skip in 348 physical
CUDA Release registrations.

Clean commit `bd5841356` rejects an automatic standalone threshold. Explicit-
range CUDA is exact but only 0.345x pinned PDAL at 250,000 points and 0.787x at
21,970,934. Automatic-range CUDA is 0.756x at 4,000,000 and 0.873x at
21,970,934. D0042 records this as an accepted negative performance gate, not
as missing implementation work.

A read-only 21,970,934-point LAS differential exercises explicit-bound
streaming over the entire source. Both the pinned PDAL oracle and native host
stage produce the same 790,955,889-byte output, SHA-256
`43fead56eade3ffa503311d9f47328d2fbd6888b913b5c7d92dfa12c8881dc23`.
