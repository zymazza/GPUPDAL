# filters.sort

`filters.sort` is implemented as an exact global permutation over typed PDAL
dimension values. Host execution uses the same pass order and
`std::sort`/`std::stable_sort` choice as the pinned PDAL implementation. This
matters for duplicate values, NaNs, signed zero, and multi-dimension requests:
PDAL sorts the first listed dimension normally and then stable-sorts each
later dimension, so the final listed dimension is the primary key.

## Exact compatibility status

The replacement accepts `dimension` or `dimensions`, `ASC`/`DESC`, and
`NORMAL`/`STABLE`, including string arrays for multiple dimensions. It gathers
keys in their physical PDAL types rather than converting 64-bit integers to
double, preserves the point view and its spatial reference/attachments, and
applies the resulting permutation without copying point records.

`where`/`where_merge`, missing or invalid dimensions/options, tagged or
branched graphs, and unproven reader-order boundaries execute through unchanged
PDAL. Empty views remain empty. The host path intentionally retains the
oracle's behavior for floating NaNs rather than imposing a total order that
PDAL does not use.

## CUDA qualification status

The CUDA implementation performs stable CUB key/index radix passes and keeps
the full permutation on device until completion. A single `STABLE` key is
exact with duplicate values, including the documented stable treatment of
positive and negative zero. `NORMAL` and multi-key requests are published from
CUDA only when the final key is tie-free; otherwise the candidate discards the
device permutation and runs the exact host algorithm before changing the
view. NaN keys also use host execution. This conservative envelope avoids
pretending CUB's deterministic tie order is the same as PDAL's implementation-
defined normal-sort tie order.

The ordinary standalone device boundary remains host-selected because it does
not beat the optimized host path by a safe margin. B0232 separately qualifies
the complete mapped-source/permutation-publisher composition for automatic
selection. That route is intentionally narrow: literal
`readers.las -> filters.sort(Z,ASC,NORMAL) -> writers.las(extra_dims=all)`,
uncompressed LAS 1.4 point format 7 with 36-byte input/output records, exactly
one finite comparator-unique binary64 Z key per point, the exact SM89/driver/
toolkit profile, and 600,000 through 16,000,000 points. It uploads 8 bytes per
point, downloads the 8-byte permutation, builds no index, and atomically
publishes reordered source records without a terminal spill. Placement and
resident preflight reserve 64 CUDA bytes per point: the measured requested-
allocation high-water, including both permutations, both materialized key
buffers, CUB workspace, and the duplicate flag, is 33,769,475 bytes at 600K
and 900,275,715 bytes at 16M.

The uniqueness proof remains data-dependent. The automatic graph has no
writer side effects before that proof, so a duplicate, signed-zero comparator
tie, non-finite key, preflight rejection, or other prepublication failure
discards the candidate and runs the unchanged host pipeline. Existing,
aliased, and symlink output refusals do the same while the atomic publisher
has made no change. The automatic route commits only after successful
publication. Other sort forms, record layouts, compression, cardinalities,
devices, and profiles stay host-selected. `--fast` is not involved and
default semantics do not change.

## Verification

Host units pin stable ties in both directions, PDAL's multi-pass priority,
encoded coordinate keys, and CUDA eligibility. The physical CUDA tests compare complete
131,103-point stable, unique-normal, and multi-key permutations and verify that
normal ties are rejected before publication. A 20-case complete-process matrix
covers defaults, aliases, directions, both algorithms, duplicate/integer/NaN
keys, multi-pass and repeated dimensions, custom intermediates, point-filter
chains, empty and 198,975-point inputs, fallbacks, and failures.
The full process matrix is clean under ASan/UBSan with libasan preloaded into
both processes. A stable Classification sort over a read-only 21,970,934-point
local LAS followed by `filters.head` produces the oracle's exact 5,865-byte
output (SHA-256
`f0961728c852e64d6d9e1106525913aa2361ea3348ca87f897d3a92f1f020885`).

On the RTX 4090, the direct ordering suite is clean under memcheck, initcheck,
racecheck, and synccheck, and the forced stable-sort process differential is
byte-identical. Dirty-tree end-to-end diagnostics reject CUDA below the large
range (0.477x pinned PDAL at 250,000 points and 0.921x at 1,000,000). At
4,000,000 points ten samples measure 1.375x PDAL for CUDA but 1.593x for the
exact host path. At 21,970,934 points five samples measure 1.808x PDAL for
CUDA and 1.796x for host, only about a 0.7% device edge. That is not enough to
justify standalone automatic GPU selection.

B0232 retains that conclusion for the ordinary boundary but remeasures the
already exact direct composition. Its exact direct ladder loses through 250K,
is positive but not publicly attributable at 500K/550K, and wins from the
measured 600K floor through the 16M cap. Final-code public runs measure
1.842475x PDAL at 600K and 3.267173x at 1M (nine alternating pairs each); the
paired means are 1.855269x +/- 0.081240 and 3.281541x +/- 0.101215. The 600K
automatic path is clean under Compute Sanitizer memcheck and racecheck. The
physical process matrix also proves exact 600K/1M positives, a 550K refusal,
exact and one-byte-below VRAM budget boundaries, grammar/layout/source/
preflight/proof refusals, existing/alias/symlink oracle fallback, and host
fallback for duplicate and non-finite Z before publication.
