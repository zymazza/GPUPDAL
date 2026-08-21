# filters.stats

`filters.stats` has an exact PDAL-facing implementation and a reusable ordered
summary primitive. It preserves the pinned stage's metadata, diagnostics,
dimension order, stream/standard behavior, and state across input views. The
stage observes points but does not change point bytes, order, or view topology.

## Exact compatibility status

The native host implementation accepts the upstream `dimensions`, `enumerate`,
`count`, `global`, `advanced`, and `commonsrs` options with their original
types and defaults. It reproduces the exact online update order for count,
minimum, maximum, average, sample variance and standard deviation, plus the
advanced skewness and excess-kurtosis recurrences. Enumerated values, value
counts, upper median, MAD, native/transformed bounding boxes, SRS metadata,
missing-dimension warnings, and empty-view behavior retain the upstream JSON
schema and byte representation.

The stage remains streamable and carries one summary state across batches.
Standard execution applies the same state across incoming PointViews. An
unchanged, eligible upstream stats stage is also audited as an order- and
view-preserving host bridge, so native regions on either side can remain in the
same in-process pipeline.

`where`, unknown options, invalid option types, and tagged or branched graphs
stay on unchanged PDAL so their original validation and execution boundaries
remain authoritative.

## CUDA qualification status

The CUDA primitive currently accepts finite materialized `double` columns,
basic summaries, and persistent prefix state. It assigns one block to each
dimension and processes point tiles in strict source order. Explicit
round-to-nearest binary64 operations preserve the pinned recurrence rather than
reassociating it into a faster but byte-different tree reduction. Dimensions
run concurrently; the recurrence within one dimension remains serial because
its intermediate states are observable in exact mode.

Advanced moments and the storage-heavy `enumerate`, `count`, and `global`
forms remain on the exact host implementation. CUDA execution is available
only through the internal experimental/require gates. Automatic replacement is
disabled: the RTX 4090 device-runtime, all four Compute Sanitizer tools, and an
end-to-end break-even matrix have not yet run for this primitive. Option-free
`gpupdal` therefore retains the pinned upstream stats stage while still allowing
qualified adjacent regions to execute natively.

## Verification

Unit tests compare every summary-state bit and derived statistic directly with
the pinned `pdal::stats::Summary`, including advanced moments, split batches,
NaNs, infinities, signed zero, and the CUDA eligibility boundary. The compiled
CUDA property compares persistent three-dimension state after 131,103 points,
crossing both tile and launch boundaries.

The 22-case complete-process matrix compares output files, metadata JSON,
status, stdout, and stderr. It covers defaults, explicit dimension order,
advanced moments, enumerate/count/global modes, empty and NaN inputs, forced
stream and standard execution, missing-dimension warnings, point-program and
multi-view composition, explicit native replacement, default upstream
selection, and failure/fallback cases. It passes in Debug, Release, and
ASan/UBSan builds.

A read-only 21,970,934-point LAS run computes basic statistics for X, Y, Z,
Intensity, Classification, and GpsTime. Both implementations emit the same
790,955,889-byte LAS, SHA-256
`ff14463744dbe9ddd2f1d10271d278a7e41478e254d96a0780bda9d7aa1da2fe`.
The three-sample host diagnostic measured the replacement at 6.363435 s versus
5.785434 s for pinned PDAL, or 0.909x. That negative result is why automatic
host replacement is disabled. A follow-up option-free trial measured 5.848450
s versus 5.823799 s, or 0.996x, confirming delegation with only selection
overhead. Neither result is a CUDA performance measurement.
