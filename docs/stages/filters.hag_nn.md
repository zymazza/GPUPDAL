# `filters.hag_nn`

Status: exact planner-owned masked-2D-index CUDA implementation for bounded
counts one through 64, with exact host repair/fallback. B0235 automatically
selects only one separately calibrated count-one direct-LAS composition.

## Exact compatibility status

The native lane separates ground and non-ground rows using the unsigned-byte
Classification field and queries the planner-owned 2D kNN index. Ground rows
receive positive zero. `count=1` preserves the pinned nearest-ground
subtraction and the upstream behavior that ignores interpolation-only options.
Wider admitted counts preserve the ordered inverse-squared-distance recurrence,
strict maximum-distance comparison, and inclusive no-extrapolation bounds.

Candidate-boundary ties, incomplete bounded-grid searches, unsupported or
nonfinite inputs, missing/insufficient ground references, and explicit CUDA
refusals run the unchanged exact host operation before publication. The stage
never builds a private spatial index. Default compatibility mode preserves
artifact bytes, metadata, point order, stdout, stderr, and exit status.

Counts one through 64 remain available in the documented explicit envelopes.
That GPU-native coverage is not a general automatic-selection or performance
claim.

## Automatic count-one direct composition

B0235 qualifies exactly:

```text
readers.las
  -> filters.hag_nn(count=1)
  -> writers.las(extra_dims=all)
```

The JSON root, stages, and option sets must be literal. Both filenames must
end in lowercase `.las`. The source must be mapped, uncompressed LAS 1.4 point
format 7 with 40-byte records and exactly one unsigned-32 `OffsetTime` Extra
Bytes descriptor; the output record must be 48 bytes after appending binary64
`HeightAboveGround`. Point count must be 450,000 through 16,000,002, and the
device must match the qualified RTX 4090/SM89/CUDA 13.3/driver 610.43.03
profile.

The selected plan contains one HAG region, one lane, and one planner-owned 2D
kNN index. Its measured boundary facts are 25 upload bytes and 8 spill bytes
per point with zero packing, plus the full 112-byte-per-point shared-index
charge. Stage-local device scratch is 15 bytes per point; placement and
resident preflight conservatively reserve the complete 160-byte-per-point
high-water.

All admission and publication refusals remain before commitment. Grammar,
layout, cardinality, budget, source, profile, rewrite, preflight, execution-
proof, existing-output, input/output-alias, and symlink drift discard the
candidate and run the unchanged public pipeline. Required-route refusals exit
124 without output. Counts 2--64, other Extra Bytes layouts, other options,
compression, and neighboring devices/profiles are exact but explicit or host-
selected.

## Performance and verification

The exact direct ladder covers 50,001 through 16,000,002 points. The 400,002
row reaches only 1.104286x pinned PDAL, so automatic selection begins at the
safer 450K floor. Final-code nine-pair public medians are 1.216823x at 450K
(0.411108 versus 0.337854 seconds) and 2.461791x at 1,000,002 points (0.931082
versus 0.378213 seconds), with all candidate samples faster.

The public matrix proves exact floor, main, and cap selection; hash-pinned
cap-plus-one fallback; exact and one-byte-below VRAM boundaries; neighboring
grammar/layout/source/proof refusals; and oracle-identical existing/alias/
symlink behavior. Host Debug, leak-disabled ASan/UBSan, and physical CUDA
Release gates pass with only documented optional-corpus skips. The actual
required 450K public route is clean under Compute Sanitizer memcheck and
racecheck.
