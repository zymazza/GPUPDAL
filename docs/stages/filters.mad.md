# filters.mad

`filters.mad` is implemented on the shared exact robust order-statistics
primitive. It selects the upper median at `size / 2`, replaces each value with
`fabs(value - median)`, selects the upper deviation median, applies
`mad_multiplier`, and retains points using PDAL's exact
`deviation / mad < k` predicate.

## Exact compatibility status

The replacement envelope requires a string `dimension` and accepts numeric
`k` and `mad_multiplier`. Their omitted defaults are `2.0` and `1.4862`.
Host execution mirrors upstream copies, operation order, division behavior,
NaNs, infinities, zero MAD, ties, and stable survivor order.

String-valued numbers, `where` variants, missing or unknown dimensions,
tagged/branched graphs, and additional options stay on unchanged PDAL.

## CUDA qualification status

Finite logical values without negative zero use two CUB radix selections with
an explicit round-to-nearest deviation kernel. The final mask evaluates the
division form rather than an algebraically simplified fence comparison so
rounding, zero MAD, and strict comparison remain observable. Other values use
the exact host path. Hardware exactness and all four sanitizer gates now pass,
but automatic standalone selection is disabled by a negative clean break-even
matrix. The device primitive remains available for resident compositions.

## Verification

The shared 16-case robust process matrix includes default and explicit MAD,
custom intermediates, NaNs, zero `mad_multiplier`, empty views, `where`
fallback, and missing-dimension failure. Host units pin the median/deviation
sequence. A 131,103-point CUDA regression compares every result double and
mask byte to host execution. Forced IQR/MAD processes and the direct property
report zero errors or hazards under memcheck, initcheck, racecheck, and
synccheck. A read-only 21,970,934-point local LAS run with `k=0` matches the
oracle's empty 2,265-byte output (SHA-256
`24fd80a8db0b248e9b86e83560679fce458af8b12a4cd24cf4dcaa1d633b583b`).
Clean commit `bd5841356` records exact MAD CUDA at 0.801x pinned PDAL for
4,000,000 points and 0.949x for 21,970,934. D0042 accepts that negative
standalone gate and keeps default execution on the host.
