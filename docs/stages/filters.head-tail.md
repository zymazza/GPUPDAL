# filters.head and filters.tail

`filters.head` and `filters.tail` select a prefix or suffix by global point
ordinal. Both use the same dimension-independent state and stable compaction
primitive as `filters.decimation`.

## Exact compatibility status

The native envelope accepts a nonnegative integer `count` and a Boolean
`invert`. Defaults match PDAL. `filters.head` supports standard and streaming
execution; its streaming counter persists across every batch. `filters.tail`
requires the total view size and therefore remains standard-mode only.

Oversized counts select the whole available prefix or suffix. Tail also emits
PDAL's exact warning when `count` exceeds the input view size, including an
empty view. Inverted forms drop the selected prefix or suffix while preserving
the order of all survivors. String-valued options, branching graphs, and other
unproven forms remain on unchanged PDAL.

## CUDA qualification status

The CUDA implementation evaluates global ordinals and then uses stable device
compaction. It is integrated into both direct uncompressed-LAS execution and
the in-process packed bridge, with one ordered lane whenever ordinal state is
present.

Automatic CUDA selection is intentionally disabled until exact device process
differentials, all four Compute Sanitizer tools, and a same-machine break-even
matrix pass on the reference RTX 4090. Until then these filters are implemented
but are not included in the public GPU-qualified count.

## Verification

Unit coverage crosses zero, in-range, and oversized counts, inversion,
standard/streaming mode, multiple chunks, and empty inputs. Exact process tests
cover standalone and chained head/tail programs, warning text, a value-filter
split, and standard-mode count propagation after decimation. All 23 ordinal
process cases pass byte-for-byte in Host Debug/Release and under ASan/UBSan;
the CUDA host/device mask and four-chunk direct-LAS regressions compile in both
CUDA configurations and await the device-runtime gate.
