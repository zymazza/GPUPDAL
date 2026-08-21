# `filters.returns` and `filters.merge`

## Exact contract

`filters.returns` accepts the upstream `groups` string-list option and defaults
to `last`. Each input view is classified into four fixed output identities in
this order: `first`, `intermediate`, `last`, and `only`. Requested groups are
published in that creation order even when the option lists them differently;
source order is stable within every output. Empty requested groups emit the
same warning and no output view. In particular, the upstream `only` predicate
is `NumberOfReturns == 1`; it does not additionally require
`ReturnNumber == 1`.

`filters.merge` creates one persistent output view during `ready()`, appends
each incoming view in execution order, preserves the upstream spatial-reference
selection and warning, and returns that same view after every append. Within a
linear hybrid graph it is therefore an audited multi-view barrier: native
single-view stages may resume after merge, but not merely after a sort of the
individual views.

Option-rich forms such as `where`, merge SRS overrides, unknown options,
tagged/non-linear graphs, and any form whose input order is not proven remain
with the unchanged PDAL implementation. Eligibility is decided before point
execution or output publication.

## Implementation

The default exact host path mirrors the pinned PDAL point loop directly. The
CUDA path materializes only `ReturnNumber` and `NumberOfReturns`, classifies
points to a three-bit group key, and uses a stable CUB radix sort of
`(group, source-index)`. Atomic integer counts define the four output ranges;
unselected points sort after them and are not published. All allocations use
the batch memory resource, all CUDA calls are checked, and the stage owns an
NVTX range. CUDA is physically qualified for explicit use, but standalone
automatic selection remains disabled because complete publication is not
faster than host in the measured range.

Merge is host-native view composition rather than a useful CUDA kernel: it
moves no point arithmetic to the CPU and prevents unnecessary delegation at
the boundary where multiple views become one. It also enables a following
point program to use its independently qualified device path.

## Verification

Host properties pin all four predicates, fixed group order, stable source
order, malformed `only` behavior, subsets, zero groups, invalid masks, missing
columns, and null permutations. The compiled CUDA property compares complete
counts and selected permutations with host execution over 131,103 points and
four masks.

`tests/differential/returns_merge_matrix.py` runs 29 complete-process oracle
comparisons covering defaults, reordered/array/duplicate/empty groups,
malformed and zero return fields, sort/splitter/groupby predecessors,
consecutive returns, 198,975-point and empty inputs, missing return dimensions,
warnings and failures,
unsupported-option fallback, single and multiple-view merge, downstream
reopening, and SRS/`where` boundaries. It compares status, stdout, stderr,
filenames, missing/extra artifacts, sizes, and bytes. The matrix is clean in
Debug, Release, and ASan/UBSan builds.

A read-only 21,970,934-point local LAS run executes
`returns(all) -> merge -> head(100)` and matches the oracle's 5,865-byte output
with SHA-256
`64432d20259eaabbe5c3c93e6be1243f07f9ec0160de90af38f17e2162b853a1`.
On the physical RTX 4090 the 131,103-point property and forced process
differential pass, and the shared partition suite is clean under memcheck,
initcheck, racecheck, and synccheck. Dirty-tree complete-process trials are
exact but measure 0.378x pinned PDAL at 250,000 points and 0.982x at
21,970,934 points, so returns/merge remains host-selected as a standalone
pipeline. Its device classification/permutation is retained for future
resident multi-view execution.
