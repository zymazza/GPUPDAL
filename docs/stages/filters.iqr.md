# filters.iqr

`filters.iqr` is implemented as an exact global order-statistics selection. It
converts the selected physical dimension to PDAL's logical double values,
chooses entries at `int(size * 0.25)` and `int(size * 0.75)`, and retains
points strictly inside the resulting multiplier-expanded fences in original
order.

## Exact compatibility status

The replacement envelope requires a string `dimension` and accepts an optional
numeric `k`; omission preserves PDAL's `1.5` default. Host execution uses the
same two independent `std::nth_element` copies as upstream, including its NaN,
infinity, tie, strict-boundary, negative-multiplier, and physical-to-double
behavior. Empty views produce an empty view through the normal PDAL pipeline.

String-valued numbers, `where` variants, missing or unknown dimensions,
tagged/branched graphs, and any extra option execute through unchanged PDAL.

## CUDA qualification status

The CUDA path materializes logical doubles, uses CUB radix order statistics,
and applies strict fences without changing survivor order. Its exact envelope
requires at most `INT_MAX` finite keys and excludes negative zero because radix
ordering distinguishes zero encodings while `nth_element` treats them equal.
Inputs outside that value envelope use the exact host implementation.

CUDA remains force/require-only for standalone pipelines. RTX 4090 device
differentials and all four Compute Sanitizer tools now pass, but the clean
same-machine complete-process matrix is negative. The exact radix/selection
primitive is retained for resident compositions that avoid a standalone host
round trip.

## Verification

Shared robust-statistics units cover known quartiles, strict fences, integer
and encoded-coordinate conversion, MAD arithmetic, and device eligibility. A
16-case complete-process matrix covers IQR and MAD defaults/options, NaNs,
integer and custom dimensions, empty input, degenerate multipliers, fallbacks,
and failures. The CUDA test compares threshold bits and every selection byte
for 131,103 points. Forced IQR/MAD processes and the direct property report
zero errors or hazards under memcheck, initcheck, racecheck, and synccheck.
A read-only 21,970,934-point local LAS run with `k=-1`
matches the oracle's empty 2,265-byte output (SHA-256
`24fd80a8db0b248e9b86e83560679fce458af8b12a4cd24cf4dcaa1d633b583b`).
Clean commit `bd5841356` records exact IQR CUDA at 0.349x pinned PDAL for
250,000 points, 0.799x for 4,000,000, and 0.951x for 21,970,934. D0042 accepts
that negative standalone gate and keeps default execution on the host.
