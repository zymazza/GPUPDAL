# filters.groupby

`filters.groupby` is implemented as an exact categorical partition over the
selected PDAL dimension. It converts each source value through PDAL's own
`getFieldAs<int64_t>` path, creates one output `PointView` per key in source
first-occurrence order, and preserves source order within every group. View
creation order matters because PDAL's `PointViewSet` and templated writer
filenames use view ids rather than sorted key order.

## Exact compatibility status

The in-process replacement accepts the required string `dimension` option and
supports standard dimensions and custom dimensions produced by an adjacent
point program. It accumulates groups across multiple input views exactly as
upstream does, including input produced by `filters.splitter` or an earlier
`filters.groupby`. Empty input creates no new group. Conversion failures occur
through the same PDAL conversion routine and retain the oracle diagnostic and
output boundary.

`where`/`where_merge`, unknown or non-string options, missing dimensions,
tagged/branched graphs, unproven reader order, and a downstream native stage
whose per-view semantics are not yet proved execute through unchanged PDAL.
The current exact replacement is therefore terminal with respect to native
multi-view processing other than another grouping, though unchanged upstream
stages may precede it. Every unproved unchanged filter is conservatively
treated as potentially multi-view before a later single-view native stage.

## CUDA qualification status

CUDA receives the already canonical signed-64 group keys, creates a stable
CUB key/index permutation, and copies only the permutation back for
`PointView` assembly. Output views are still created by scanning original keys
in first-occurrence order; sorted permutation traversal then appends points in
stable order. This separates key partitioning from PDAL's observable view-id
semantics.

The device path is limited to at most `INT_MAX` points and is physically
qualified for explicit use. Automatic standalone CUDA selection is disabled:
complete multi-file publication remains faster on host at the measured sizes.
The stable device partition remains a resident-plan primitive for consumers
that can avoid rebuilding PointViews on host.

## Verification

Planner and rewrite units pin the split descriptor, custom dimension binding,
terminal multi-view guard, fallback matrix, and unstable-reader rejection. A
16-case complete-process matrix covers Classification and ReturnNumber,
negative custom keys, one group, a preceding stable sort, multiple input views
from `filters.splitter`, consecutive groupings, downstream multi-view guard
fallbacks, 198,975-point input, empty input, conversion failure, and preparation
errors. Every generated output filename and byte is compared, and the matrix
is clean under ASan/UBSan with libasan preloaded into both processes.

A read-only 21,970,934-point Snow Road Twin LAS partitions into four exact
outputs totaling 790,962,684 bytes. Their SHA-256 values are
`f847ce5127bab4c8a55e4425cd4ca31e790d6dee5c397ea9b2c754855e0b518b`,
`04366778b02bdb023c99211bb9e73a0f0da1852b78469b6802279c0e57ec2ad2`,
`9168a5cf25b588832b5f497e128cf32b55a9a14607e9afa63ecc2ff4ef120da0`,
and `07fa9a35015884592e90f21c4f95355db9325a8e0a36f20fe87eb48ff5852b0f`.

The physical shared ordering property and forced groupby process differential
pass on the RTX 4090, with the primitive clean under all four Compute
Sanitizer tools. The artifact-set benchmark hashes each numbered output and
rejects missing or extra views. Dirty-tree end-to-end trials are exact but
measure 0.331x pinned PDAL at 250,000 points and 0.913x at 21,970,934 points;
the stage therefore remains host-selected outside a resident device region.
