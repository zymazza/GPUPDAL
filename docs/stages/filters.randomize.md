# filters.randomize

`filters.randomize` is an audited exact host ordering bridge. The stage remains
the pinned PDAL implementation, so seeded permutations use the same
`std::mt19937` and libstdc++ `std::shuffle`, and an omitted seed continues to
use the same `std::random_device` behavior.

## Why this is a bridge

PDAL randomizes the `PointView` index vector; it does not copy or reorder every
point record. Exact seeded order therefore depends on the standard-library
shuffle and `uniform_int_distribution` implementation used to build the pinned
oracle. Fisher-Yates swaps also have a serial dependency. Moving that operation
to a standalone CUDA kernel would either change the permutation or add a host
permutation, upload, and download around an eight-byte index array.

The hybrid planner now preserves this original stage while recognizing its
audited view contract: without `where` or other option-rich forms, randomize
changes point order but neither creates nor combines views. A subsequent exact
point program, ordering stage, reduction, or merge region can therefore remain
native instead of causing the whole pipeline to delegate. If upstream input
order was not already proven, randomize does not manufacture that proof. A
multi-view input also remains multi-view.

`where`, invalid or string/negative seeds, unknown options, tagged graphs, and
other unaudited forms delegate before point or output side effects. They retain
the original diagnostics and base-filter semantics.

## CUDA status

This row is not counted as a device implementation or GPU-qualified filter.
A useful CUDA design must keep the point data resident, generate the exact
permutation on the host, upload that mapping once, and let downstream kernels
consume it without physically shuffling records. That work belongs in the
resident global-stage scheduler; a transfer-heavy standalone imitation is not
accepted as GPU acceleration.

## Verification

The 14-case complete-process matrix covers seeds zero, one, and `UINT32_MAX`;
point regions on both sides; stable-sort composition; empty, 100-point, and
multi-batch inputs; per-view randomization followed by merge; and `where`,
string/negative seed, and unknown-option fallbacks. It is byte-exact in Debug,
Release, and ASan/UBSan builds.

A read-only 21,970,934-point case runs seeded randomize followed by a native
assignment and produces the same 790,955,889-byte LAS as the oracle, SHA-256
`e66521fc478722f27b5d782410b8ec32b3e8774ffe426d4380b8a260eb98b347`.
An exploratory alternating Release benchmark over the same input recorded a
20.443 s candidate median and 20.767 s oracle median (1.016x, three samples).
The dirty-tree result is treated as noise-level composition evidence, not an
accepted performance or GPU claim.
