# `gpupal --fast` (D0261, D0271)

`gpupal --fast <command> ...` runs any `gpupal` command under the **fast contract**.
The flag must be the first argument; the launcher consumes it, arms the
internal `PDG_INTERNAL_FAST_MODE=1` marker for the engine and sibling PDAL,
and dispatches the remaining command exactly as it would without the flag. An
externally supplied `PDG_INTERNAL_FAST_MODE` is removed before the engine
route, so fast mode cannot be switched on from the environment.

## Contract

Relative to the default exact contract (byte-identical artifacts, metadata,
point order, stdout, stderr, and exit status against the pinned PDAL oracle),
fast mode keeps:

- **point records in PDAL's order, with identical count, layout, and
  coordinates** for every LAS/LAZ artifact (X, Y, Z of every record are
  bit-identical; no record is added, dropped, or moved);
- **every attribute bit-identical except on kNN tie rows** (D0271, below);
- **non-point artifacts byte-identical** (rasters, JSON/CSV/text outputs);
- **failure still fails**: an error path must exit non-zero and publish no
  output.

and may relax:

- **kNN tie order (D0271)**: when a point has several neighbors at exactly
  the same distance, pinned PDAL picks whichever its CPU tree visits first;
  the device visits them in a different order. Under `--fast` the spatial
  index does not report distance ties, so the device (or host-index) choice
  stands and the exact CPU tie repairs (r6's eigen family, HAG-NN's selective
  repair, LOF's closure, the classifier vote, and every other
  identity-observing consumer) do not run. Affected rows get a
  different-but-equally-valid neighbor set, or the same set accumulated in a
  different order at identical distances (last-bit rounding); nothing loses
  precision. Records on those rows may differ from stock in the dimensions
  the stage writes. Incomplete bounded searches are still repaired exactly.
- stdout and stderr text (warnings, log lines, progress, program prefix);
- error message text and the exit status value on failure paths;
- LAS header/VLR/EVLR metadata (generating software, dates, system id, VLR set
  and order, LAZ chunk layout), as long as decoded records satisfy the above;
- pipeline `--metadata` and `pdal info` JSON fields other than the point
  records they summarize.

Reduction order and every other numeric path stay exact. Widening the
contract further (order-insensitive records or numeric tolerances) is a
separate product decision.

## What it changes today

The one relaxation is D0271's tie order. `include/pdg/FastMode.hpp`
(`pdg::fastModeEnabled()`, `pdg::relaxedTieOrder()`, `pdg::knnStatusMask()`)
is the engine's only reader of the marker; the spatial index applies the mask
where kNN statuses are produced (both device gather kernels through a
per-build `__constant__` mask, and the host-index paths), so no consumer
needs to know. Reference impact on this workstation (B0271): r6 and r2 gain
their tie repairs back (25 and 125 of 1,000,000 records differ from stock,
coordinates identical); every other headline is record-identical to its exact
run. Any future fast-only optimization must be gated on
`pdg::fastModeEnabled()` (engine) or `pdg::cli::fastModeEnabled()` (launcher)
and qualified with the fast comparator below.

The launcher consumes the flag on every route, including the
environment-selected engine route, where the injected-marker guard strips
only an ambient marker (D0271 fixed a gap where the flag was left in `argv`
on that route).

## Measuring under the contract

`scripts/pdg/benchmark_reference.py --contract fast --candidate-arg=--fast`
compares LAS/LAZ artifacts record by record against the first oracle
artifact (LAZ decoded through the pinned oracle for both sides): count and
record layout must match, every record's coordinates must be identical, and
the number of records whose other bytes differ must stay within
`--fast-max-differing-records-fraction` (default 1%). Reports carry
`comparison.contract`, `exact_records`, `identical_records`,
`fast_differing_records`, and `fast_compared_records`; every other artifact
compares byte-for-byte; stream hashes are recorded but do not decide
`exact_outputs`. `reference_suite.py run --contract fast --candidate-arg=--fast`
labels its aggregate `contract: fast` and carries the per-workload differing
counts; an exact aggregate never mixes with fast reports. A performance claim
under the fast contract must state the differing-record count next to the
speedup and, for the reference workloads, cross-check it against the tie-row
count the exact run reports.

## Tests

`pdg_dispatcher_process_boundary` proves the flag reaches the oracle, engine,
and environment-selected engine routes with the marker, changes no routing,
and that an injected marker is stripped on every route.
`pdg_reference_runner_environment_contract` proves the comparator accepts
identical records under different headers only under the fast contract,
rejects reordered records and coordinate changes under both, and accepts a
bounded attribute difference only under fast within the allowance.
`FastMode`, `CudaSpatialIndex.RelaxedTieOrderMasksOnlyTheTieBitOnBothBackends`,
`PdgNeighborClassifierFilter.RelaxedTieOrderSkipsTheHostRepairUnderFast`,
`PdgApproximateCoplanarFilter.RelaxedTieOrderSkipsTheEigenRepairUnderFast`,
and the CUDA-lane `pdg_fast_tie_contract_cuda` process test prove that only
the tie bit changes, that no host repair runs under the marker (fail-closed
under `PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR`), and that the default contract
is byte-identical.
