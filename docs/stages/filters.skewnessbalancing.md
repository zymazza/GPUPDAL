# `filters.skewnessbalancing`

Status: exact host implementation with a bounded exact CUDA ordering substep.
B0234 automatically selects only one measured mapped-source/direct-permutation
composition; ordinary stage execution remains host-selected.

## Exact compatibility status

The wrapper preserves the pinned PDAL algorithm: it orders the whole view by
physical binary64 Z, runs the sequential moment and sign-crossing recurrence in
that exact order, updates Classification on the host, and publishes the final
point order. Point count and full point records are preserved.

The CUDA lane accelerates only the global Z permutation. It is admitted when Z
is physical binary64, finite, and comparator-unique. Duplicate values,
positive/negative-zero equivalence, non-finite values, unsupported option
forms, `where`, branched or multi-view graphs, and unproved ordering boundaries
use the exact host implementation. This is an exact-default path; `--fast` is
not involved.

## Automatic direct composition

B0234 qualifies exactly:

```text
readers.las
  -> filters.skewnessbalancing
  -> writers.las(extra_dims=all)
```

The root and stages must be literal and option-free except for the writer's
`extra_dims=all`; both filenames must end in lowercase `.las`. The source must
be mapped, uncompressed LAS 1.4 point format 7 with 36-byte input and output
records and no carried source extra dimensions. Point count must be from
450,000 through 16,000,000, and the device must match the qualified RTX
4090/SM89/CUDA 13.3/driver 610.43.03 profile.

The direct route uploads the 8-byte Z key, downloads the 8-byte full-record
permutation, retains the exact recurrence on the host, and atomically publishes
the permuted mapped source records. Placement and resident preflight reserve
65 CUDA bytes per point: 56 bytes of established ordering scratch plus the
concurrent binary64 Z and byte Classification columns.

All refusals occur before commitment. A tie, signed-zero comparator match,
non-finite key, insufficient budget, source/preflight/rewrite/execution-proof
failure, or existing, aliased, symlink, or otherwise rejected destination
discards the candidate and runs the unchanged public pipeline. The automatic
route commits only after successful publication. Other graphs, layouts,
compression modes, cardinalities, devices, and profiles remain host-selected.

## Performance and verification

The exact direct ladder loses from 50K through 300K. Its 350K result does not
clear the acceptance margin, and a separate public 400K run reaches only
1.065118x pinned PDAL. Final-code nine-pair public medians are 1.200177x at the
450K floor and 3.091007x at 1M, with all candidate samples faster. The accepted
ladder extends through the measured 16M cap; it is not evidence beyond that
cap or for a broader stage shape.

The automatic matrix proves exact 450K, 1M, and 16M selection, a hash-pinned
16,000,001-point default fallback and required refusal, exact and one-byte-
below VRAM boundaries, neighboring grammar/layout/source/profile refusals,
data-dependent tie/non-finite fallback, and oracle-identical existing/alias/
symlink behavior. The retained host/CUDA wrapper and ordering tests cover the
recurrence and permutation semantics. Host Debug,
leak-disabled ASan/UBSan, and the physical CUDA Release runnable set pass with
only documented optional-fixture skips; the actual required 450K route is
clean under Compute Sanitizer memcheck and racecheck.
