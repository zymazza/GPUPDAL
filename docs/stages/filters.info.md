# `filters.info`

Status: exact native host implementation; option-free CUDA bounds/count path
physically qualified for explicit forced use, but slower and not automatically
selected.

The host implementation preserves the pinned PDAL metadata schema, dimension
order, selected-point packing, nearest-query ordering and ties, empty input,
stream/standard behavior, diagnostics, and the historical three-coordinate
query parser behavior. The JSON `p` alias remains delegated because the pinned
pipeline parser rejects it even though the argument declaration advertises it.

The CUDA envelope currently covers the option-free form. A fused reduction
computes count and all six XYZ extrema in one device pipeline. It preserves
BOX3D initial sentinels, strict comparisons, NaNs, infinities, signed-zero bit
patterns, first global index on ties, LAS integer coordinate decoding, and
ordered merges across chunks. `point` and `query` continue through the exact
host backend until device point gathering and nearest selection have their own
complete differentials.

Automatic replacement is disabled. Internal `PDG_EXPERIMENTAL_CUDA_HYBRID`
and `PDG_REQUIRE_CUDA_HYBRID` gates retain explicit hardware validation. The
option-free envelope is GPU-native but not performance-qualified; `point` and
`query` remain exact host work and are not described as device-native.

Current evidence:

- 20 exact complete-process cases in Debug, Release, and ASan/UBSan;
- host units for special values, signed zero, coordinate decode, count, and
  chunk merging;
- a 21,970,934-point host differential with exact 790,955,889-byte LAS and
  25,937-byte metadata artifacts;
- physical execution of the 131,103-point host/device property and an exact
  forced process differential including metadata;
- zero memcheck, initcheck, racecheck, and synccheck findings;
- B0104 current-binary 1M/4M forced-CUDA diagnostics at only 0.526x/0.745x
  pinned PDAL; and
- an output-bound 1M profile with 0.26863 milliseconds across 16 kernels,
  only 0.047591% of candidate wall. Exact PDG-host info is within 3.259660
  milliseconds/1.03986% of the same-binary bare translation control, so
  record-summary reuse does not meet the 5--10% implementation gate.

Functional support is complete through CUDA or exact host fallback.
GPU-native coverage is limited to the option-free forced envelope.
Performance-qualified: no. Automatically selected: no.
