# `gpupdal verify`

`gpupdal verify` creates a portable, self-contained record that another machine or
user can publish without giving PDG permission to contact anyone or upload
anything. The command performs one local complete-process differential: the
pinned PDAL oracle and `gpupdal` run the same normal/covariance pipeline with an
untimed warm-up and at least three alternating measured pairs.

```sh
gpupdal verify --output-dir gpupdal-proof
```

By default the pinned oracle creates a deterministic 250,000-point fixture.
Existing data and a pipeline using `input.laz` and `output.laz` placeholders
can be supplied instead:

```sh
gpupdal verify --input tile.laz --pipeline pipeline.json \
  --runs 3 --warmups 1 --output-dir gpupdal-proof
```

The command writes:

- `pdg-verification.json`, using schema `pdg-verification-report-v1`;
- `pdg-verification.html`, a self-contained human-readable rendering;
- the raw benchmark report and isolated work products needed to audit it.

The report records the public candidate, sibling engine, pinned oracle, loaded
PDAL libraries, helper, and frozen-clock hashes; versions; CPU, RAM, GPU,
driver, CUDA compiler, process affinity; input and pipeline hashes; placement
profile/status; every raw timing; output hashes; exactness; and an exact rerun
command. The tracked executable/helper hashes are checked again after timing.

The command returns nonzero if evidence is missing, malformed, inexact, the
benchmark process fails, or a tracked binary changes. A speedup below one is
reported honestly but is not a correctness failure. `gpupdal --fast verify` is
rejected because this command proves the default exact contract.

If `PDG_ORACLE_PDAL` redirects the normal compatibility oracle, `gpupdal verify`
refuses to use it silently. Pass `--accept-configured-oracle` to make that
choice explicit; the source, complete binary hash, expected 2.10.0 version
check, and provenance limitation are then recorded in the report. Frozen
release evidence additionally binds the oracle to its retained build record.

This artifact is **prepared evidence**, not third-party validation. The JSON
explicitly records `external_validation.status = not-performed`. An unrelated
user may independently run and publish the same command, but the project must
not call an author-run artifact “independent validation.”

## Retained frozen-release demonstration

The B0286 author-run demonstration used the exact frozen release candidate on
an RTX 4090 / Intel Core i5-14500 worker. It completed one warm-up and three
measured pairs, produced exact files, streams and status, detected no binary
drift, and returned a valid report. Its honest median speedup on the default
250,000-point verification fixture was **0.785296x**; verification success is
an exactness/provenance result, not a requirement that this small fixture win.

The retained JSON SHA-256 is
`961bcb72f869d6ed186232b1a6ab5205051f487335c595e587d0a71dada7d124`
and the self-contained HTML SHA-256 is
`928ee06296b427783df88c46e82ae147f62d1a7622807f8dc83947fd2c392cee`.
The report says `external_validation.status = not-performed`; no unrelated
person or organization was contacted or represented as a validator.
