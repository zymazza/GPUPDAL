# `filters.label_duplicates`

Status: exact host and CUDA implementations for the unconditioned stage.
Standalone and resident automatic selection remain disabled; B0233
automatically selects only one measured label/NNDistance/assignment hybrid
composition.

## Exact compatibility status

The stage does not sort its input. It visits points in their current order and
sets `Duplicate` on each row after the first when every selected dimension,
converted through PDAL's binary64 accessor, equals the same dimension on the
immediate predecessor. Users who need equal values to become adjacent must
place an explicit stable sort before this filter. Point order, point count, and
view identity are unchanged.

The first row is deliberately not written. This preserves an existing
`Duplicate` byte exactly and matches the zero-initialized value when the
dimension was just registered. An absent or empty `dimensions` list is valid:
the comparison is vacuously true, so every row after the first becomes one.
NaNs compare unequal, positive and negative zero compare equal, and integer
values beyond binary64's exact range can compare equal after conversion, all
as in upstream PDAL.

The native descriptor reads every selected dimension and the existing
`Duplicate` byte, writes unsigned-byte `Duplicate`, requests no spatial index,
and declares a global order- and cardinality-preserving pass. Selecting
`Duplicate` itself is intentionally retained on the exact host path because
upstream's in-place loop is sequential and each later comparison can observe a
preceding write. Missing dimensions preserve the upstream diagnostic.
`where`, `where_merge`, unsupported option forms, and unproven graphs retain
the unchanged upstream stage.

## CUDA and resident execution

The exact CUDA operation initializes rows one through the end to one, then
runs an adjacent comparison for each selected physical column and clears rows
that differ. Coordinate columns use their checked logical binary64 decode.
Every launch is stream ordered, CUDA errors are checked, and the stage is
covered by one NVTX range.

Standalone execution remains available only through the experimental/required
hybrid controls. In a selected resident plan, the stage can create a
whole-view device product without uploading XYZ or constructing an index,
publish `Duplicate`, and retain that device column for a following assign or
ferry program. When a later neighborhood stage needs an index, the same region
may build the planner-owned index once; duplicate labeling never creates a
private index.

Complete-process measurements over `Classification` are byte-identical but
negative: forced CUDA reaches 0.260x pinned PDAL at 250,000 points, 0.537x at
1,000,000, and 0.959x at 21,970,934. The representative 1M comparison kernel
itself takes 7.36 microseconds on the RTX 4090, at 65.93% SM throughput, 14.00%
DRAM throughput, and 75.75% achieved occupancy. The complete-pipeline gap is
therefore dominated by stage/runtime and host-device publication boundaries,
so no standalone placement model or performance claim is accepted. B0020 and
D0082 record the raw evidence and decision.

B0233 accepts a separate complete-composition selector for exactly:

```text
readers.las
  -> filters.label_duplicates(dimensions=Classification)
  -> filters.nndistance(k=10)
  -> filters.assign(value="UserData = Duplicate")
  -> writers.las
```

The root, stages, and option sets must be literal; endpoints must be lowercase
`.las`. Input must be uncompressed LAS 1.4 format 7 with a 375-byte point
offset, 36-byte records, zero VLRs/EVLRs, no trailing bytes, and 250K--16M
points. Automatic selection is limited to the qualified RTX 4090/SM89/CUDA
13.3/driver 610.43.03 profile. The label, NNDistance, and assignment bridge
each carry an internal automatic marker, and a proof gate verifies actual CUDA
use rather than selection alone.

The current exact forced-hybrid ladder loses at 50K (0.893x pinned PDAL) and
wins from 250K through 16M (3.261x--12.190x). Final option-free public medians
are 2.846x pinned PDAL at 250K and 6.392x at 1M. A current resident probe is
exact but 4.3% slower than the hybrid at 1M, so it remains rejected. Every
neighboring graph, layout, count, or profile uses the unchanged host pipeline;
a recoverable label CUDA failure uses the exact host label operation before
writer execution.

## Verification

Host units pin adjacent binary64 comparisons, wide-integer rounding, NaNs,
signed zero, empty dimensions, first-row preservation, missing columns, and
self-referential output fallback. The physical CUDA unit covers every PDAL
physical type plus encoded coordinates and special floating-point values.
Wrapper differentials compare the upstream and replacement filters directly,
and the resident gate proves that a no-index `Duplicate` device column feeds a
downstream point program without a rebuild. Empty selected resident regions
also remain exact no-ops through both label-to-assign and label-to-kNN
compositions while the terminal stage closes the region normally.

The original complete process matrix covers 15 host cases and 10 CUDA-admitted cases:
defaults, comma and array StringList forms, repeated and empty dimensions,
adjacency with and without explicit stable sorting, special values, empty and
multi-batch inputs, conditional fallbacks, malformed and missing dimensions,
and self-referential output. Host Debug and leak-disabled ASan/UBSan each pass
their maintained aggregates, as does physical CUDA Release. The focused
primitive, wrapper, and resident tests report zero findings under memcheck,
initcheck, racecheck, and synccheck.

B0233 adds an option-free public matrix at the 250K floor. It proves exact
default and required-route selection; 50K, format-6, header-padding,
VLR/EVLR, trailing-byte, record-stride, uppercase-reader/writer,
disabled/unavailable CUDA, and injected recoverable failure fallback; strict
root/stage/option default differentials and required refusals; and oracle-
identical existing-output, input/output-alias, and symlink behavior.
The final Host Debug and leak-disabled ASan/UBSan aggregates pass 407/409, and
physical CUDA Release passes 599/601; only the two documented optional-corpus
cases skip. Compute Sanitizer memcheck and racecheck are clean on the actual
250K automatic process.
