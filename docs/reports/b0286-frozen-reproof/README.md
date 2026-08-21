# B0286 frozen 1.7x release re-proof

This directory is the publishable raw-evidence bundle for the author-run B0286
release proof. It is not third-party validation and it is not the unexecuted
40-project 3DEP population study.

## Frozen identities

- release id: `pdg-1.7-reproof-a7391d105b71-evidence-v2`
- source archive SHA-256:
  `a7391d105b710efe63476bdeb23c38527362a806227b400a75c37f0a60203026`
- source commit:
  `cc12b04eb0a02f082270ed8a7d0689c44a31c7fc`
- pinned PDAL commit:
  `f1e35f5c3e416b8c6a2c39966c8205cfae54afe2`
- candidate SHA-256:
  `3e734d39662a985e8cb7b06d6cc72d464a162ad87504a93b67f95345549043bd`
- oracle SHA-256:
  `6e1b06cf2227f98d5934a886c57e4a2b559b6d0c893918d7345e882324119a39`
- release-manifest SHA-256:
  `01443c1245f81843766b7b6250fefe3d22fd0d48b513378ba4b25219608312f6`
- payload-tree SHA-256:
  `ceffa6cd41a6fe7626f86dc4bf080fd0f5b931c03e04ce3e0d4a940bee71b4d7`

The working tree was dirty when frozen. The source archive hash, manifest and
binary hashes—not the commit alone—identify the tested release candidate. A
fixture-only closure amendment corrected the COPC fixture and added the real
47,478,228-point AHN4 input before accepted timing; it did not change either
binary. Two earlier deployments failed preflight before accepted aggregate
timing (wrong COPC fixture, then missing GDAL/PROJ runtime tools).

## Current 1M results

Each machine ran all fourteen reference workflows with one warm-up and three
alternating measured pairs. Every aggregate is complete and exact.

| GPU | Equal-workload GM | Total wall | r6 features |
| --- | ---: | ---: | ---: |
| A10 | 1.146x | 1.377x | 7.410x |
| A100 40 GB | 1.156x | 1.371x | 7.942x |
| L40S | 1.149x | 1.385x | 7.913x |
| RTX 3060 12 GB | 1.146x | 1.365x | 6.363x |
| RTX 3090 | 0.983x | 0.987x | 0.986x |
| RTX 4060 Ti | 1.154x | 1.371x | 8.629x |
| RTX 4080 | 1.176x | 1.397x | 10.110x |
| RTX 4090 | 1.119x | 1.347x | 6.531x |
| RTX 5090 | 1.163x | 1.393x | 9.695x |
| RTX A6000 | 1.164x | 1.430x | 10.230x |

Thus seven of ten r6 cells are 7.4–10.2x, two are about 6.4x, and the
RTX 3090 is the explicit 1M exception at 0.986x.

## Current 47.5M results

Only these seven machines were measured at 47,478,228 points. Each row is the
exact r6 LAZ graph with one warm-up and three pairs; the output SHA-256 is
`8d8b81018219f2183e6de1d59269f5f0bcb26d629d1d59986849c83e3ca45238`.

| GPU | PDAL median | PDG median | Speedup |
| --- | ---: | ---: | ---: |
| A10 | 714.414 s | 83.840 s | 8.521x |
| A100 40 GB | 725.837 s | 73.840 s | 9.830x |
| L40S | 818.073 s | 104.690 s | 7.814x |
| RTX 3090 | 892.964 s | 99.839 s | 8.944x |
| RTX 4090 | 432.764 s | 63.522 s | 6.813x |
| RTX 5090 | 459.573 s | 39.617 s | 11.601x |
| RTX A6000 | 1122.234 s | 113.876 s | 9.855x |

This is a seven-card range of 6.81–11.60x, not a ten-card result.

## Other executed gates

- Hardened frozen-binary conformance: 2,048/2,048 cases, zero unexplained
  differences, and all 4,096 product executions returned zero. The compressed
  raw report is under `conformance/`.
- Real nondeterministic COPC control: differing container bytes but identical
  canonical one-million-point multiset, semantic metadata, hierarchy
  invariants and exact bounded query. Physical hierarchy assignment and the
  coarse preview differ and are retained as diagnostics under `copc/`.
- Packaged `pdg verify`: exact and valid with no binary drift on an RTX 4090;
  its small 250K fixture measured 0.785296x and its report explicitly records
  `external_validation.status = not-performed`.

The Vast account credit moved from `50.7409784967` to `42.4604191779`, a total
cost of approximately **$8.28**, including rejected preflight deployments.

## Layout

- `release/`: frozen build metadata and full artifact manifest.
- `systems/<gpu>/`: machine facts, placement status, pre/post artifact checks,
  fourteen raw per-workflow reports, aggregate, and (for seven GPUs) 47.5M
  raw report.
- `conformance/`: generated manifest, compressed raw report and hashes.
- `pdg-verify/`: portable JSON/HTML report and raw benchmark summary.
- `copc/`: accepted canonical nondeterministic-container report.
- `SHA256SUMS`: digest of every other file in this directory.
