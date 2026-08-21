# `filters.divider` and `filters.splitter`

## Exact contract

The native `filters.divider` envelope covers count-based `partition` and
`round_robin` modes for counts from 2 through 1000. Partition mode uses the
upstream ceiling-sized sequential groups; round-robin mode assigns source
point `i` to view `i % count`. Both modes create exactly the requested number
of output views for every input view, including empty views, and retain source
order within every output. Those empty views and their creation ids are
observable through templated writer filenames.

Capacity mode and expression-triggered splitting remain with unchanged PDAL.
The pinned implementation mutates its shared size state in capacity mode and
has distinct expression view-creation semantics, so neither form is treated as
an alias for count partitioning.

The native `filters.splitter` host envelope mirrors the complete upstream
length/origin/buffer loop for finite, positive lengths and finite explicit
options. An omitted origin coordinate is taken from point zero of the first
incoming view and then persists. The cell-to-view map also persists across
incoming views, while a cell's first encounter creates its output view and
therefore controls numbered-writer publication order. Upstream intentionally
decrements every negative delta after truncating division: with origin zero
and length ten, coordinate `-10` belongs to cell `-2`, not `-1`.

A positive buffer can add a point to its primary tile, one horizontal tile,
one vertical tile, and one diagonal tile. Membership uses strict bounds and
the pinned left/right, down/up, then diagonal test order. Negative and zero
buffers produce only primary membership. Unsupported options, `where`, unsafe
numeric option domains, tagged/non-linear graphs, and unproven input ordering
remain on unchanged PDAL before point execution begins.

## Implementation

The shared divider primitive returns per-view counts and a stable
output-position-to-source-position permutation. Sequential partitioning uses
the identity permutation. CUDA round-robin partitioning generates a ten-bit
view key and source index, then uses stable CUB radix sort; empty view ranges
remain explicit in the returned counts.

The splitter primitive materializes logical `X` and `Y` as doubles and computes
primary signed cell coordinates. The CUDA kernel uses explicit round-to-nearest
subtraction and division and is admitted only after a host scan proves every
coordinate finite and representable in the upstream `int` cell domain.
Positive-buffer membership stays in the exact host loop for now because its
observable multi-tile append order must include PointView construction and
writer costs in any useful device design.

Both CUDA paths are physically qualified for explicit use. Automatic
standalone selection remains disabled because end-to-end numbered-view
publication is still faster on host. Their key/cell primitives remain
available to resident multi-view plans that avoid host PointView assembly.

## Verification

Unit properties pin uneven and empty divider counts, stable partition and
round-robin permutations, invalid counts, splitter zero/positive/negative
boundaries, reversed axes, nonfinite coordinates, missing columns, and unsafe
programs. Compiled CUDA properties compare complete divider counts and
permutations in both modes and all splitter cell coordinates over 131,103
points.

`tests/differential/divider_splitter_matrix.py` performs 35 complete-process
oracle comparisons. It covers both divider modes, uppercase mode parsing,
uneven and empty views, multi-batch input, sort/splitter predecessors,
consecutive partitions, default/partial/explicit splitter origins, positive
and negative buffers, groupby input, coordinate mutation, merge/downstream
reopening, invalid options and failures, and conservative fallback boundaries.
It compares status, stdout, stderr, every filename, missing/extra artifacts,
sizes, and bytes. The matrix is clean in Debug, Release, and ASan/UBSan builds.

A read-only 21,970,934-point local LAS run executes round-robin divider,
splitter, merge, and `head(100)`. Candidate and oracle emit the same 5,865-byte
LAS with SHA-256
`e13d95da3dcc4a39309b5f38dbe4914adaa02af41208c820e56d586876e213c3`.
CUDA Debug and Release compilation, the physical properties, and both forced
process differentials are green. The shared suite reports zero memcheck,
initcheck, and synccheck errors and zero racecheck hazards. The exact
artifact-set benchmark covers all 17 divider outputs and 49 splitter outputs
on the 21,970,934-point corpus. Dirty-tree device trials measure 0.952x and
0.932x pinned PDAL respectively (0.358x and 0.355x at 250,000 points), so host
remains the correct standalone selector.
