# `filters.neighborclassifier`

Status: exact shared-index CUDA implementation in a bounded option envelope;
automatically selected only for the separately calibrated direct-LAS route.

The native path accepts `1 <= k <= 64` when `domain` and `candidate` are empty
and the vote dimension is `Classification`. It reuses the planner-owned 3D kNN
index, performs the bounded neighborhood query and classification vote on
device, and publishes the canonical one-byte `Classification` field without
changing point order or view identity. Conservative boundary ties and
incomplete queries retain the pinned KD3 repair; this host repair is part of
the exact implementation and is not described as fully device-native. The
automatic envelope below is narrower and accepts only `k=7`.

Default compatibility mode preserves upstream output, metadata, point order,
stdout, stderr, and exit status. Unsupported options, topology, layouts,
cardinalities, devices, or runtime proof facts retain the unchanged exact host
path. The stage never builds a private spatial index.

## Exact host worker passes (B0258/D0257)

The exact host implementation — the path every non-native shape uses,
including r11's domain-qualified `k=7` vote — now runs its per-point kNN
query and vote over fixed contiguous chunks of source point ids on host
workers. Each source point's result depends only on its own query against the
read-only PointView-owned nanoflann KD3 index and the read-only neighbor
classes, so the neighbor lists, votes, and reassignments are identical at any
worker count; per-worker reassignment lists are merged in worker order and
applied serially afterwards, and the first per-row conversion failure in
original order surfaces with the pinned diagnostic and status. Domain,
candidate-file, and vote-dimension semantics are unchanged. The worker count
is `min(ceil(rows / 4096), hardware threads, PDG_NATIVE_WORKERS)`, so small
inputs stay serial; `PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS` forces the
single-threaded pass as the same-final-binary control. On the reference
workstation the classifier's 706,098-query r11 pass falls from about 2.18 to
about 0.22 seconds. This is an exact host improvement, not native CUDA
coverage; the CUDA envelope below is unchanged.

## Automatic envelope

B0231/D0230 automatically selects only a literal three-stage
`readers.las -> filters.neighborclassifier(k=7) -> writers.las` pipeline with
uncompressed format-7 36-byte input and output records, 250,000 through
16,000,000 points, mapped-source input, direct Classification publication, a
112-byte-per-point planner-owned index, one selected region/lane, and the exact
RTX 4090/SM89/CUDA 13.3/driver 610.43.03 placement profile. Every neighboring
shape fails closed before output. In particular, the measured 50,000-point row
is slower than pinned PDAL and remains host-selected.

The accepted 1,000,000-point public route is byte-, diagnostic-, status-, and
order-exact. Pinned PDAL measures 3.619967 seconds median versus 0.897460 for
`pdg`, or 4.033570x. A separate nine-pair attribution finds the direct boundary
1.150355x +/- 0.024677 faster than the former exact resident boundary, with all
nine pairs faster. These numbers qualify only the stated envelope.

## Verification

Runtime-placement tests pin the model coefficients, boundary/index byte facts,
floor, cap, one-lane constraint, and isolation from the ordinary stage model.
The public automatic process matrix proves exact 250K and 1M output plus the
selected executable rewrite and actual mapped-source/direct-output executor.
It also covers the 50K host control and fail-closed source, preflight, execution
proof, format, `k`, compression, and writer-option negatives. The 250K floor is
clean under Compute Sanitizer memcheck and racecheck; the full physical CUDA
unit suite and resident-selection matrix pass with only documented optional
fixture skips.
