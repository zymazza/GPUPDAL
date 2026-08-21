# filters.ferry

`filters.ferry` copies dimensions in declaration order. Every source is read as
`double`, and the destination uses PDAL's numeric conversion rules. A later
mapping can therefore read a value written by an earlier mapping.

## Exact compatibility status

The first native CLI envelope accepts one default
`readers.las -> filters.ferry -> writers.las` pipeline when the LAS input is in
the exact D0004 translation envelope and every mapping:

- uses standard dimensions present in the reader layout;
- leaves PDAL's resolved destination layout type unchanged;
- preserves its source logical value through the canonical writer boundary.

Coordinate destinations are supported and cause exact header bounds to be
recomputed. Writing `ReturnNumber` causes the 15 logical return counters to be
recomputed. Mapping order, half-away-from-zero integer conversion, NaN
handling, and failed-assignment behavior match the pinned oracle.

Custom dimensions, type-widening mappings, scan-angle destinations, legacy
scan-angle sources, noncanonical original coordinate sources, extra stage
options, branching, and other pipeline shapes automatically execute through
the pinned PDAL fallback in compatibility mode.

## CUDA status

The CUDA implementation executes up to 16 ordered mappings per grid-stride
kernel launch and preserves launch order on the allocator stream. Longer
programs are split across ordered launches. Coordinate mappings currently use
the exact host path. Automatic GPU selection remains disabled until the
physical-GPU exactness, compute-sanitizer, and same-machine performance gates
pass.

## Verification

Host tests cover all 100 numeric physical-type pairs against pinned PDAL,
ordered chains, initialization, coordinate encoding, conversion boundaries,
and worker-count determinism. Process differentials compare complete LAS bytes
and diagnostics across formats 0–3 and 6–8. GPU-runner tests compare all
destination types and a chain that crosses the kernel-launch boundary.
