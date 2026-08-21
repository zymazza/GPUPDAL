# filters.mortonorder

`filters.mortonorder` is implemented as an exact global permutation over PDAL
X/Y values. Ordinary mode reproduces the pinned upstream most-significant-bit
coordinate comparator. Reverse mode reproduces its `sqrt(point_count)` grid,
16-bit Morton interleave, full 32-bit reversal, and ascending multimap
traversal. Equal codes retain source insertion order.

## Exact compatibility status

The in-process replacement accepts the Boolean `reverse` option and returns a
new point view in exactly the same cases as upstream, including the different
empty-view behavior of ordinary and reverse modes. It preserves all point
dimensions, spatial reference, and attachments by applying the permutation to
the original view.

Degenerate bounds and nonfinite coordinates intentionally use a copied,
attributed version of the pinned upstream host algorithm. `where`,
`where_merge`, non-Boolean option encodings, unknown options, tagged or
branched graphs, and unproven reader-order boundaries remain on unchanged
PDAL. This keeps option parsing, failures, implementation-defined casts, and
tie behavior inside the oracle implementation when an exact radix key cannot
be proved.

## CUDA qualification status

Inside the exact device envelope, ordinary Morton order is represented by a
62-bit interleaved key whose X bit has the same priority as upstream's
comparator. Reverse order uses the exact 32-bit reversed code. CUDA generates
one key per point, then reuses the stable CUB key/index ordering primitive so
duplicate codes preserve input order.

The device envelope requires finite, nondegenerate bounds, finite logical
double X/Y values inside those bounds, and no more than `INT_MAX` points. It is
physically qualified in both ordinary and reverse modes. Automatic selection
starts at 2,000,000 points after the clean B0006 crossover/profile gate;
smaller, unsupported, explicitly disabled, device-less, or recoverably failed
runs retain host execution.
Host-selected execution now goes directly to the upstream-equivalent
comparator and does not allocate or populate the CUDA staging batch.

## Verification

Host properties compare key order directly with the upstream comparator and
pin the reverse 4-by-4 traversal. A 15-case complete-process matrix covers
ordinary/reverse order, duplicate coordinates, color records, sequential
Morton stages, following predicates, empty views, 198,975-point multi-batch
input, degenerate axes, fallbacks, and failures. The same matrix is clean under
ASan/UBSan with libasan preloaded into both processes.

The physical CUDA regression compares every key bit and the complete stable
permutation for 131,103 points in both modes. Both ordinary and reverse
complete-process differentials are byte-identical, and the shared Morton/
ordering property suite reports zero errors under memcheck, initcheck, and
synccheck and zero racecheck hazards. A read-only 21,970,934-point Snow Road Twin
LAS followed by `filters.head` produces the oracle's exact 5,865-byte output
(SHA-256
`61638be4f61d806608c4ac6df4da0e0715915d0347b1dcf68b57afe1bf898c6e`).

B0006 records the clean option-free selector at commit `7b27cb1fb`. Ten-sample
ordinary medians are 0.983x PDAL at 1,000,000 points (host selected), 1.248x at
the 2,000,000-point CUDA threshold, and 1.528x at 4,000,000 points. Reverse
Morton is 1.487x at 2,000,000 points. Every warmup and measured artifact is
byte-identical to the corresponding oracle. A clean option-free Nsight Systems
trace contains the Morton key kernel and eight CUB radix passes; automatic-path
Compute Sanitizer memcheck reports zero errors. D0041 records the selection
decision and preserved fallback envelope.

Earlier dirty-tree diagnostics independently cover 21,970,934 ADKLR points,
5,114,283 snow-road-twin points, and 36,772,046 VEIL LAZ points at 1.936x,
1.578x, and 2.032x PDAL respectively. They remain corroborating corpus breadth,
not substitutes for B0006.
