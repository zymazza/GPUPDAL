# GPUPDAL Conformance Suite

The **GPUPDAL Conformance Suite** is the named, versioned release gate for the
claim that `gpupdal` is a drop-in replacement for the pinned PDAL oracle within
GPUPDAL's declared semantic envelope. It compares complete processes, not kernels:
exit status, stdout, stderr, created artifact set, file bytes, and deterministic
repeats.

Byte equality is authoritative. The default comparison mode is `bytes`, and a
native stage that cannot satisfy it stays on an exact host/oracle fallback.
`copc-canonical-v1` is a separately named supplemental comparator for a COPC
artifact whose pinned oracle has first been shown to vary its container bytes.
Selecting it requires a case-specific `oracle_nondeterminism_reason`; it does
not weaken the default product contract or make a byte mismatch disappear.

## Bounded generated matrix

`tests/conformance/bounded-recipes-v1.json` expands to exactly 2,048 stable
cases: eight named input boundary families, sixteen stage/option families,
eight writer/layout families, and two pipeline-root forms. It covers empty and
tiny views, duplicate coordinates, exact-distance kNN ties, signed zero and
NaN, large offsets and fine scales, return boundaries, extra dimensions, and a
single bounded one-million-point case. It is deliberately a finite,
reviewable recipe rather than an open-ended random generator.

Generate and validate the manifest:

```sh
python3 scripts/pdg/conformance.py generate \
  --recipes tests/conformance/bounded-recipes-v1.json \
  --output build/conformance/generated-v1.json
python3 scripts/pdg/conformance.py validate \
  build/conformance/generated-v1.json
```

Run a release candidate against the pinned oracle:

```sh
python3 scripts/pdg/conformance.py run \
  --manifest build/conformance/generated-v1.json \
  --oracle build/pdal-upstream-tests/bin/pdal \
  --candidate build/pdg-cuda-release/bin/gpupdal \
  --differential scripts/pdg/differential.py \
  --frozen-time-library build/pdg-cuda-release/lib/libpdg_frozen_time.so \
  --work-dir build/conformance/work \
  --report build/conformance/report.json
```

A qualifying release report must be complete and have zero unexplained
differences. `--case-limit` is diagnostic only when paired with
`--allow-partial`; a partial run never becomes a release-wide pass.

## Arbitrary offline pipelines

The manifest schema also accepts hand-authored PDAL pipeline cases. Each case
declares the exact pipeline object, inputs, expected output paths, comparator,
resource caps, and input hashes. The runner stages inputs into an isolated
directory and rejects undeclared absolute paths, external/network stages, and
outputs outside the case directory. This makes an arbitrary pipeline
reproducible without granting it arbitrary side effects.

The process environment is assembled from a narrow execution allowlist;
ambient `PDG_*`, `PDAL_TEST_*`, and loader controls are not inherited. Time,
stream, artifact-entry, file-count, and artifact-byte limits are enforced
during execution and traversal, not merely reported after unbounded capture.

LAS failures report the first differing byte and its decoded public-header,
VLR/EVLR, or point-record region when possible. COPC semantic reports gate on
header/VLR payloads, COPC information, structural hierarchy invariants, the
all-point canonical record digest, and a fixed exact-bounds query digest.
Physical hierarchy-node assignment and a coarse-resolution preview remain in
the report as diagnostics but do not decide equality: concurrent pinned-PDAL
writes can vary those two container-layout details for the same full point
multiset.

The infrastructure contracts are registered as
`pdg_conformance_runner_contract` and
`pdg_copc_semantic_comparator_contract`.

## B0286 retained results

The qualifying run used the frozen candidate SHA-256
`3e734d39662a985e8cb7b06d6cc72d464a162ad87504a93b67f95345549043bd`
and frozen oracle SHA-256
`6e1b06cf2227f98d5934a886c57e4a2b559b6d0c893918d7345e882324119a39`
on the retained L40S worker. It passed **2,048/2,048** cases with zero
unexplained differences; every one of the 4,096 candidate/oracle product
executions returned zero. The generated manifest SHA-256 is
`e0d880fcd33413fc4d2daae67b4ada4bb189d964664dfbe771a1637c65468a1f`.
The raw report SHA-256 is
`1453f7afaa2ad8846db4eba83b3c15c73b323e7db21811c68e91a0510745e824`
and its retained gzip SHA-256 is
`6ba382ffbec6eeae9186c861a7331bfdf72d8138d47e4af4823ec24202ef16bf`.

An earlier diagnostic run is rejected as release evidence because the first
runner version treated 92 matching nonzero process results—including 16
matching aborts—as passes. The hardened runner requires every case to declare
its expected process status. The generated matrix was corrected so all 2,048
cases are valid successful executions, then rerun completely both locally and
against the frozen binaries.

The real COPC nondeterminism control writes the same one-million-point input
twice with pinned PDAL. The two COPC byte hashes differ, as do physical
hierarchy-node assignment and the coarse preview, while the canonical full
record digest, semantic metadata, hierarchy invariants and exact center query
match. The accepted semantic report SHA-256 is
`94e348cc24c1dd38a2768df248c1d3b8c566a97d8bd64724b183dccb4a1d465b`.
