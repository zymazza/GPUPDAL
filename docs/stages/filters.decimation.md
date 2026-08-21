# filters.decimation

`filters.decimation` retains a deterministic ordinal sequence without reading
any point dimensions. Its implementation shares the stable compaction path
used by the other exact selection filters and keeps sequence state across
every input chunk.

## Exact compatibility status

The native envelope accepts JSON numbers for finite `step >= 1` and
nonnegative JSON integers for `offset` and `limit`. Omitted options retain the
PDAL defaults. String-valued options and every unsupported form execute
through unchanged PDAL.

PDAL intentionally produces different sequences in its two execution modes.
Streaming mode evaluates each point against
`offset + llround(kept * step)` and can retain a final candidate that standard
mode omits. Standard mode first computes
`llround((min(limit, size) - offset) / step)` candidates and appends exactly
that many source points. The native implementation preserves this distinction
and uses global 64-bit state so chunk boundaries cannot alter either result.

PDAL's standard implementation performs unsigned subtraction when `offset`
exceeds the available range. The initial hybrid envelope delegates nonzero
standard-mode offsets before execution unless the direct LAS path can validate
the exact input count. This preserves upstream behavior without evaluating an
unsafe count expression inside the replacement stage.

## CUDA qualification status

The CUDA kernel marks the exact candidate ordinals and feeds the existing
stable device compactor. Direct LAS execution and the in-process packed bridge
both compile with this implementation. An ordinal program uses one ordered
lane so state cannot race across chunks.

Automatic CUDA selection remains disabled. Device runtime differentials,
memcheck, initcheck, racecheck, synccheck, and a same-machine break-even matrix
must pass on the reference RTX 4090 before this stage is counted as
GPU-qualified.

## Verification

Unit tests pin the fractional standard/streaming difference, the upstream
2.6-step sequence, offset/limit boundaries, invalid domains, and arbitrary
chunk partitions. A 23-case process matrix compares complete output files,
status, stdout, and stderr for default, integral, fractional, offset, limit,
zero-limit, chained, empty, fallback, and predicate-split cases. The matrix is
also run with an ASan/UBSan candidate. A bounded large-corpus differential uses
a 21,970,934-point LAS file and emits only 100 points while crossing more than
160 input batches.
