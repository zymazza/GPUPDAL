# PDAL-GPU resident-execution handoff

Date: 2026-08-17

This is an operational handoff, not a specification. If anything here conflicts
with `spec.md` or `DECISIONS.md`, those two files are the sole source of truth.
Read `AGENTS.md`, `spec.md`, `IMPLEMENTATION_PLAN.md`, `DECISIONS.md`,
`docs/testing-strategy.md`, and `docs/stage-coverage.md` before resuming, and
`docs/diagnostics.md` before investigating anything — several conclusions in
`BENCHMARKS.md` were wrong for a slice or more because an instrument existed
and was not used.

## Clean stopping point

### Current state — read this before the chronology below

The bullets after this block are **chronological and partly superseded**. They
record how each conclusion was reached, wrong turns included, which is
deliberate — but do not quote a figure from them without checking it here
first.

#### Final session stop — 2026-08-17 UTC (B0279/D0280)

**This subsection is the authoritative resume point.** After the B0278 stop
(same day) the maintainer read the plain-language report and asked, first,
that RTX 5090s and RTX 3090s use the GPU by default, then whether there was
any reason not to prefer the GPU at a 1.7x measured win instead of 3x. The
answer (D0280: no strong reason; the margin covers only the host-side
transfer, the rented-host spread was ~1.8x, a wrong call at 1.7x is a
bounded loss, and no box's fully admitted local profile hurt the suite) was
adopted:

- **Shipped profiles re-issued at 1.7x (D0280/B0279).** All ten
  `data/placement-profiles/*.json` regenerated from the archived B0278
  calibrate evidence with `make_shipped_profile.py --margin 1.7` (now the
  tool's default for a class profile; `--generic` defaults to 3 and the
  generic profile is unchanged); `shipping_margin` is recorded per profile.
  Effect: single-feature filters from 1M on every class, compose graphs
  from 250K on most, RTX 5090 extradims from 1M, RTX 3090 compose 1M–48M;
  `neighborclassifier` still host everywhere; minimum admitted win per
  class ≥ 1.70x. `src/CMakeLists.txt` now lists the profile files as
  `CMAKE_CONFIGURE_DEPENDS` (content-only edits re-embed; verified both
  trees re-ran configure). No box re-measured: the B0278 3x proof is a
  floor; each box's local-profile runs are the nearest evidence (5090 r6
  8.66x at 1M; 3090 AHN4 8.06x/8.18x/18.77x). The open item is a re-proof
  sweep at 1.7x (~$22) if the record is to carry drop-in numbers for the
  files as built.
- Reports: `docs/reports/b0278-drop-in-defaults.html` regenerated with a
  postscript and margin-aware captions (`--postscript`; sha in the anchors);
  `docs/reports/pdg-drop-in-report.html` (plain-language, private artifact
  https://claude.ai/code/artifact/5f773110-63bd-4e4d-913b-b407cd20ffc0)
  rewritten for the 1.7x rule and republished. Docs: `spec.md` D4,
  `docs/calibration.md`, `data/placement-profiles/README.md`,
  `IMPLEMENTATION_PLAN.md`, `docs/reports/README.md`.
- Validation: CUDA Release **858/858** (527.64 s), Host Debug **550/550**
  (one ENOSPC failure in the aggregate — `/home` was 100% full with 38 GB of
  retained `copc-reader-*` differential scratch, now removed — passed alone
  on rerun), 52 profile/placement units under ASan/UBSan, both trees
  re-ran configure on the profile change. Disk note: prune
  `build/*/tests/differential/copc-reader-*` periodically (~1.8 GB per
  aggregate pair) — see B0279.

Checkpoint: committed locally on `p3-skewness-resident` (not pushed) as
`bb9abfdd6`, subject `Re-issue the shipped placement profiles at a
1.7x margin (D0280/B0279)`; previous checkpoints `c26e60a4b` (plain report),
`6928b98d9`/`47b31cc82` (B0278). `git status --short` must be empty at
handoff. Resume protocol: reread D0279/D0280, B0278/B0279 and
`docs/calibration.md`, then the B0274 resume ctest set plus `ctest -R
'pdg_calibrate_local_profile_cuda_exact|LocalProfile|Placement'`.

Next slice candidates: the 1.7x re-proof sweep (only if the maintainer
wants it on record); an `--explain`-style per-stage placement summary for
users (the plain report lists it as the most likely support question);
measure the generic tier on a big cloud on an unswept GPU; a profile refresh
protocol (profiles are keyed to CUDA 13.3); then the unchanged engineering
list (exact SMRF morphology on the device, `--fast` GPU reprojection, LAZ
encode scale-out, pipelined streaming overlap as an opt-in, a packing-rate
probe for `pdg calibrate`, calibrating the direct whole-view executors).

#### Final session stop — 2026-08-17 UTC (B0278/D0279) — retained; superseded by B0279 above

**This subsection was the authoritative resume point before B0279.** Since the B0277 stop
the maintainer restated the goal as a drop-in replacement — same pipelines,
same bytes, faster on the user's machine with nothing to run first — and
D0279/B0278 deliver it. The driving session lost its connection near the
end; this stop was assembled by a resuming session that found the last
rented box (RTX 3090) still running with both phases complete, pulled its
results, destroyed it, verified every shipped profile regenerates from the
archived evidence, ran the aggregates, and wrote B0278 and this handoff.

- **Shipped GPU-class profiles and a generic fallback (D0279).** Two tiers
  below the embedded reference profile and any local profile:
  `data/placement-profiles/{a10,a100,l40s,rtx3060,rtx3090,rtx4060ti,rtx4080,
  rtx4090,rtx5090,rtxa6000}.json` (tier `shipped`, keyed by exact device
  name + compute capability + compiled CUDA toolkit; measured by `pdg
  calibrate` on rented boxes, `bench/remote/vast_sweep.sh`, converted by
  `bench/report/make_shipped_profile.py` with a 3x shipping margin — a model
  is kept only where the device won by ≥ 3x at every measured size, envelope
  bounded to those sizes) and `generic-cuda.json` (tier `generic`; the
  intersection of the shipped models under the same margin in every
  profile, slowest-device/fastest-host coefficients, `applies` bounds CC ≥
  8.0 and ≥ 12.24 GB). Embedded at build time (`src/CMakeLists.txt` →
  `generated/pdg/ShippedProfilesData.cpp`), looked up by
  `shippedPlacementProfileFor` / `genericPlacementProfileFor` in
  `src/plan/LocalProfile.cpp`, consulted by `placementCalibrationFor` as
  tiers 4 and 5 (`src/plan/PlacementProfile.cpp`); `pdg calibrate --status`
  and `pdg doctor` name the active tier; `PDG_DISABLE_SHIPPED_PROFILES`
  turns both off; test hooks `PDG_TEST_SHIPPED_PROFILE_DIR[_ONLY]`. The r6
  reference graph's `extra_dims=all` LAZ layout gained its own calibration
  case and model (`normal-covariancefeatures-compose-extradims`,
  `src/plan/RuntimePlacement.cpp`): a profile that carries it lifts B0224's
  one-layout bound; the embedded reference profile keeps it. Reference
  machine unchanged: `active_profile_tier: embedded`.
- **B0278 measurements**: ten GPU classes, each box calibrated then rebuilt
  with the shipped set and no local profile, fourteen workflows at 1M in
  default mode: shipped tier (no calibration) vs before — A10 1.889x/2.655x
  → **2.126x/3.605x**, A100 2.285x/3.145x → **2.520x/4.324x**, L40S
  2.031x/2.836x → **2.242x/4.005x**, RTX 3060 2.016x/2.845x →
  **2.443x/4.665x**, RTX 3090 2.108x/2.720x → 2.075x/3.024x (thin-margin
  host; compose only from 4M), RTX 4060 Ti 1.991x/2.825x →
  **2.182x/3.768x**, RTX 4080 2.452x/3.232x → **2.619x/4.374x**, RTX 4090
  2.642x/3.477x → **2.928x/5.033x**, RTX 5090 1.973x/2.764x → 1.839x/2.625x
  (extradims measured 2.984x at 1M, under the margin by 0.016; r6 stays host
  at 1M; the difference is that box's host-side noise), RTX A6000
  2.783x/3.529x → **3.050x/5.150x**; r6 2.3x–2.8x → 7.3x–12.0x on the eight
  lifted boxes. Generic tier (own profile excluded; A10, RTX 3060, RTX 4080)
  inert at 1M within noise — its models begin at 4M except eigen-family /
  rank-optimal / LOF. AHN4 47.5M under the shipped tier: features 9.8x
  (A10), 16.1x (A100), 17.2x (L40S), 15.2x (RTX 4090), 11.9x (A6000); the
  r6 reference graph 8.6x–12.5x on the device for the first time; RTX 3090
  host (2.4x; local profile 8.1x). Every cell exact. Vast $21.52 (23
  instances, 10 productive), credit $80.55 → **$59.04**; no instances remain.
- Report `docs/reports/b0278-drop-in-defaults.html` (`9c24476e9c98e7a1...`,
  `bench/report/build_sweep_report.py`); tables by
  `bench/report/sweep_tables.py`; raw results
  `build/benchmarks/b0278-sweep/<slug>/results-sweep/` (the RTX 3090's
  tarball was re-fetched without the 458 MB AHN4 LAZ as
  `results-sweep-nolaz.tar.gz`, `82593c423d004f99...`).
- Validation (this session, D0279 tree): CUDA Release aggregate **858/858**
  (21 optional skips, 514.00 s), Host Debug **550/550** (4 optional skips,
  222.29 s), 52 profile/placement unit tests under ASan/UBSan, placement
  audit unchanged; every shipped profile regenerated and diffed against
  `data/`.
- Plain-language report for the maintainer's open-source release:
  `docs/reports/pdg-drop-in-report.html` (published as a private artifact
  on 2026-08-17; wins / what did not work / what to expect per tier /
  testing / release checklist — incl. the CUDA-13.3 profile key, fat-binary
  architectures, PDG-version key on local profiles, and the missing
  "explain placement" output as release to-dos).
- Docs: `spec.md` D4 (profile tiers; the runtime still never times or
  tunes), `IMPLEMENTATION_PLAN.md` (delivered paragraph),
  `docs/calibration.md` (tier order, `--status` output, the shipping
  margin), `data/placement-profiles/README.md`, `docs/reports/README.md`.
  Wording corrected in D0279 and the docs: the swept hosts were EPYC/Xeon
  class (no Threadripper among them).

Checkpoint: committed locally on `p3-skewness-resident` (not pushed) as
`47b31cc82`, subject `Checkpoint shipped GPU-class placement profiles
through B0278`; previous checkpoints `ed00492ac`/`12503ef72` (B0277). `git
status --short` must be empty at handoff. Resume protocol: reread
D0279/B0278, `docs/calibration.md` and `data/placement-profiles/README.md`,
then the B0274 resume ctest set plus `ctest -R
'pdg_calibrate_local_profile_cuda_exact|LocalProfile|Placement'`.

Next slice candidates: measure the generic tier on a big cloud on an
unswept GPU (the one unmeasured promise in D0279); a shipped-profile
refresh protocol when the calibration cases or the toolkit change (the
profiles are keyed to CUDA 13.3 — a build against another toolkit gets no
shipped tier); the RTX 5090 extradims 1M miss (2.984x) — either accept, or
re-measure with more pairs before the next refresh; then the unchanged
engineering list (exact SMRF morphology on the device, `--fast` GPU
reprojection, LAZ encode scale-out, pipelined streaming overlap as an
opt-in after a phase-split measurement, a packing-rate probe for `pdg
calibrate`, calibrating the direct whole-view executors).

#### Final session stop — 2026-08-17 UTC (B0277/D0277/D0278) — retained; superseded by B0278 above

**This subsection was the authoritative resume point before B0278.** Since the B0276 stop
the maintainer chose recommendation #1 ("let other machines use the GPU by
default") and asked for a report with test results. Delivered:

- **`pdg calibrate` (D0277).** An explicit command that re-measures the
  placement calibration cases on the machine it runs on with the same
  residual-fit procedure as the embedded SM-89 profile (`pdg_placement_audit
  --suggest-models`, factored into `PlacementCalibration.{hpp,cpp}`), and
  writes `pdg-local-placement-profile-v1` JSON keyed to the exact machine
  (device name, CC, driver, compiled CUDA, CPU model, logical CPUs, `pdg`
  version) at `$PDG_PROFILE_PATH` or `~/.config/pdg/placement-profile.json`.
  `placementCalibrationFor` consults embedded → calibration override
  (explicit `resident` only; the automatic route declines under it) → local
  profile if the machine key matches. Host = sibling `pdal`, device =
  `pdg-engine resident` under `PDG_CALIBRATION_FORCE_DEVICE_PLACEMENT=1`;
  outputs byte-compared; models admitted only for byte-exact device wins,
  envelope = smallest winning size … largest measured; `--append` refits
  over every size in the profile's evidence; `--input` calibrates on the
  user's data; `--status`, `--dry-run`, `--force`; `pdg doctor` reports the
  active profile; benchmark reports record `candidate_placement_profile`.
  Files: `src/cli/Calibrate.cpp`, `src/plan/LocalProfile.cpp`,
  `src/plan/PlacementCalibration.cpp`, `src/core/SyntheticCloud.cpp`,
  `src/core/CalibrationProbes.cu` (+stub), `docs/calibration.md`,
  `bench/remote/vast_calibrate{,_post,_post2}.sh`,
  `bench/report/build_calibration_report.py`, tests
  `tests/unit/local_profile_test.cpp` (8) and
  `tests/differential/calibrate_test.py` (`pdg_calibrate_local_profile_cuda_exact`).
- **Resident layout follows file EB types (D0278).** Found by the AHN4
  measurement: the tile's extra-bytes VLR types (`Deviation` uint16,
  `Amplitude`/`Reflectance` double) made the resident preflight refuse the
  layout, so a calibrated box still ran the tile on the host. `bindLayout`
  now redefines the per-execution registry type to the file's for dimensions
  no planned stage touches (`DimensionRegistry::redefineType`), and keeps a
  specific refusal for device-computed dimensions. Locally the 47.5M-point
  tile's feature graph runs on the resident device executor in 23.1 s (host
  156.9 s), byte-identical to the pinned oracle.
- **B0277 measurements** on two rented boxes (RTX 4090/EPYC 7742 with
  driver 580.95.05; RTX 5090/EPYC 7B12), fourteen workflows at 1M in default
  mode: uncalibrated **2.338x GM / 3.175x total → calibrated 2.510x /
  4.296x** (4090; forced hybrid 2.275x / 4.039x), **1.877x / 2.651x →
  2.078x / 3.551x** (5090; forced 1.874x / 3.333x); r6 2.66x → 9.05x and
  2.49x → 7.78x; every cell exact; the calibrated default beats the forced
  hybrid on both aggregates. AHN4 47.5M tile in default mode after D0278:
  features 10.09x / 14.79x, LOF 22.10x / 25.46x, exact (before D0278: 2.4x /
  1.5x host). Report `docs/reports/b0277-local-calibration.html`
  (`70af7351bac1356c...`, published as a private artifact); raw results
  `build/benchmarks/b0277-{4090,5090}/results-b0277/`.
- Validation: CUDA Release aggregate **858/858** (21 optional skips,
  540.87 s), Host Debug **550/550** (4 optional skips, 233.25 s), unit tests
  under ASan/UBSan, placement audit unchanged (214/218). Two defects fixed
  during the run before checkpoint (point-program device runs refused to
  overwrite outputs; first `--append` replaced envelopes) — see B0277.
- Vast: both instances destroyed; credit **$80.55** of the ~$100
  authorization remains (about $19.5 spent across B0275–B0277); no
  instances remain.
- Reference machine unchanged: embedded profile precedence, `pdg calibrate`
  declines without `--force`; B0274 exact claim of record stands.

Checkpoint: committed locally on `p3-skewness-resident` (not pushed) as
`12503ef72`, subject `Checkpoint pdg calibrate and file-typed resident
layouts through B0277` (plus the hash-recording follow-up commit at `HEAD`);
previous checkpoints `187c3a961`/`ba4f6761f` (B0276). `git
status --short` must be empty at handoff. Resume protocol: reread
D0277/D0278/B0277 and `docs/calibration.md`, then the B0274 resume ctest set
plus `ctest -R pdg_calibrate_local_profile_cuda_exact`.

Next slice candidates (unchanged from the post-B0276 recommendations, minus
#1 which is done): exact SMRF morphology on the device (r2/r3), `--fast`
GPU reprojection (r1/r8/r9), LAZ encode scale-out for wide records (r6/r11/
r4), pipelined streaming overlap only as an opt-in after a phase-split
measurement; a packing-rate probe for `pdg calibrate` (the one inherited
coefficient) and calibrating the direct whole-view executors.

#### Final session stop — 2026-08-16 UTC (B0276/D0276) — retained; superseded by B0277 above

**This subsection was the authoritative resume point before B0277.** Since the B0275 stop:
the maintainer asked whether bigger clouds on a bigger GPU raise the ratio
against stock PDAL, and B0276/D0276 answers it with a measured ladder on a
rented H200 (141 GB; SM 90 build; kNN graphs under
`PDG_EXPERIMENTAL_CUDA_HYBRID=1` because automatic selection stays
profile-locked to the qualified RTX 4090): AHN4 25GN1 tiles at 47M / 95M /
190M points, one exact pair per cell, all 20 cells byte-exact. The kNN-heavy
graphs sit on a flat ~10x plateau (r6 10.40/10.38/10.42x, r4
9.74/10.17/9.34x), r2 climbs 1.40 → 1.53 → 1.65x, the I/O-bound graphs stay
at 1.4–3.6x (r14 3.26 → 3.65x, r1 1.4–1.6x, r3 ~1.6x), and the H200 does not
run the PDG side faster than the RTX 4090 box in absolute time (r6 at 47M:
53.6 s vs 48.3 s; workstation 30.7 s) — the device is not the bottleneck at
these sizes; the exact host repair, single-file LAZ I/O and host-only stages
are. The answer to the maintainer's question is therefore "no, the ratio
does not climb with size; a larger card buys headroom, not ratio". Section
7b of the report (`docs/reports/b0275-cross-machine-benchmark.html`,
`b31f3d9d9ba807c0...`, republished at the same artifact URL) and the plain
summary carry the numbers and caveats (single runs; a 26.9-core cgroup
quota on the box; rebuilt oracle). Results:
`build/benchmarks/b0276-h200/results-ladder/` (+ `results-b0276.tar.gz`).
Vast: two H100 NVL offers were destroyed without results (sshd key-mode
error; went offline); the H200 (contract 47864532) ran about 3.3 h and was
destroyed after download; credit $84.53 remains of the ~$100 authorization;
no instances remain. No product code changed; the B0274 lanes stand. New
tooling: `bench/remote/vast_ladder.sh`; `vast_bootstrap.sh` shallow oracle
fetch with retries; `build_report.py --ladder-results`. No build, test,
benchmark, profiler, rented instance, or delegated review is left running.

Checkpoint: committed locally on `p3-skewness-resident` (not pushed) as
`ba4f6761f`, subject `Checkpoint the H200 scale ladder through B0276`; previous
checkpoint `4c8f03c13` (B0275). `git status --short` must be empty at
handoff. Resume protocol: as for B0275 below, plus reread B0276/D0276.

#### Final session stop — 2026-08-16 UTC (B0275/D0275) — retained; superseded by B0276 above

**This subsection was the authoritative resume point before B0276.** Since the B0274 stop:
the maintainer's benchmark report is delivered (B0275/D0275): the fourteen
headlines on the reference workstation (B0274 exact claim
**2.900213x/2.888163x** geometric mean, **5.404502x/5.360525x** total wall;
B0271 fast contract), the same suite on a rented Threadripper PRO 3975WX +
RTX 4090 Vast.ai box in three configurations (default host-only 2.385987x
GM / 3.274039x total; forced CUDA hybrid 2.268169x / 4.076398x; CUDA hybrid
+ `--fast` 2.325248x / 4.192501x — forcing CUDA where the selector would not
choose it loses on the light workflows), a 47,478,228-point AHN4 tile on
both machines (r6 features 12.9x/13.6x with CUDA, r4 10.4x/10.4x, conversion
and rasters 1.6x--2.9x), LAStools timings of comparable jobs on both
machines, rendered outputs, and the upstream-merge assessment, in
`docs/reports/b0275-cross-machine-benchmark.html` (published as a private
artifact). The Vast instance was destroyed after download (about $1.30
spent). No product code changed; the B0274 lanes stand. No build, test,
benchmark, profiler, rented instance, or delegated review is left running.

##### Session summary (B0257 -> B0279)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x |
| B0265/D0265 | pooled morphology in the fork SMRF port | r2 1.62x |
| B0266/D0266 | slot-pooled exact reprojection (stream + standard) | r8 2.10x, r1 1.49x |
| B0267/D0267 | hashed sample voxel table | r4 5.31x |
| B0268/D0268 | parallel LAS record packing (stream + standard) | r14 2.08x, r13 1.95x; suite 2.13x GM, 4.06x total |
| B0269/D0269 | parallel LAS record unpacking (reader batch hook + per-tile) | r14 2.66x, r13 2.68x, r7 1.32x; suite 2.56x GM, 4.66x total |
| B0270/D0270 | ordered parallel COPC decode under requests=1 | r5 1.83x; suite 2.61x GM, 4.56x total |
| B0271/D0271 | `--fast` tie-order contract (device tie choices, no CPU repair) | fast: r6 11.59x, r2 2.04x; fast suite 2.71x GM, 4.92x total; exact unchanged |
| B0272/D0272 | automatic r4 CUDA outlier selector retired (host path faster) | r4 7.42x; suite 2.70x GM, 4.85x total |
| B0273/D0273 | banded parallel raster accumulation in writers.gdal | r3 2.08x; suite 2.77x GM, 5.00x total |
| B0274/D0274 | concurrent structure-identical KD builds | r6 10.69x, r11 12.89x, r4 8.38x; suite 2.90x GM, 5.40x total |
| B0275/D0275 | cross-machine / big-cloud / LAStools report | box 2.39x GM host-only; AHN4 47M r6 13.6x CUDA; report published |
| B0276/D0276 | 47M/95M/190M ladder on a rented H200 | kNN graphs flat ~10x, r2 1.40 → 1.65x, I/O-bound 1.4–3.6x; all exact |
| B0277/D0277/D0278 | `pdg calibrate` local placement profiles; resident layouts follow file EB types | 4090 box default 2.34x → 2.51x GM (3.18x → 4.30x total), r6 2.7x → 9.1x; 5090 1.88x → 2.08x; AHN4 47M default 10.1x/14.8x features, 22.1x/25.5x LOF; all exact |
| B0278/D0279 | shipped GPU-class profiles (ten-GPU sweep, 3x margin) + generic fallback; extradims compose model; calibrate = tightening | drop-in default (no calibration) vs before: 4090 box 2.64x → 2.93x GM (3.48x → 5.03x total), A6000 2.78x → 3.05x, RTX 3060 2.02x → 2.44x, eight of ten lifted, r6 2.3–2.8x → 7.3–12.0x; 5090/3090 host under the margin; AHN4 shipped 9.8x–17.2x features, r6 graph 8.6x–12.5x on device; all exact |
| B0279/D0280 | shipped profiles re-issued at a 1.7x margin (generic keeps 3x); CMake configure-depends on the profile files | no new box measurement; single-feature filters from 1M on every class, compose from 250K, RTX 5090 r6 from 1M, RTX 3090 compose 1M–48M; expected from local-profile runs (5090 r6 8.66x; 3090 AHN4 8.1x/18.8x LOF) |

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `abd3cbe9c08535a99ca3eb6d928a89872adbb717bd53f8cbf89fbd993882eb7e` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `c7afa7ba2d596562c7d23f04d9a22d13fecee9f5ed8a8e2c3556438338e2fd26` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `ea113c56be1498e81088c8900f6c15bf86a2deec34c8a3dfccebd440597084d7` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| report | `docs/reports/b0275-cross-machine-benchmark.html` | `262410c2c5881467...` |
| B0278 report (regenerated at B0279 with the postscript) | `docs/reports/b0278-drop-in-defaults.html` | `23fe43a2a88f508dcc924eddc9ef0c0b13a5f5b8aba0ab7265ddb8e049d891d3` |
| plain-language report (1.7x rule) | `docs/reports/pdg-drop-in-report.html` | `f8172e8cc3e9bdfcfc23b37fbee6ebe60ae34bb2e264a1e6339fc35d543173b1` |
| B0278 sweep results | `build/benchmarks/b0278-sweep/<slug>/results-sweep/` (ten slugs) | rtx3090 `results-sweep-nolaz.tar.gz` `82593c423d004f99...` |
| shipped profiles | `data/placement-profiles/*.json` (11 files) | regenerate byte-identically (modulo `created_utc`) via `make_shipped_profile.py` |
| box results archive | `build/benchmarks/b0275-box/results-b0275.tar.gz` (extracted alongside) | suites `6481017d...` / `183867ea...` / `845afb77...` |
| local AHN4 / LAStools reports | `build/benchmarks/b0275-local-ahn4-*.json`, `b0275-lastools-local.json` | LAStools `f7dcf2f5...` |

##### Validation completed at this boundary

No product code changed since B0274 (CUDA 849/849, Host Debug 542/542,
focused sanitizers). The report's generator, remote scripts, LAStools bench,
and render script are new tooling only.

##### Workspace checkpoint

Committed locally on `p3-skewness-resident` (not pushed) as `HEAD`, subject
`Checkpoint the cross-machine benchmark report through B0275`. `git status
--short` must be empty at handoff. Earlier checkpoints this session:
`1fda5d85b` (B0274), `0897bf990` (B0273), `632ee0508` (B0272), `ba169bd73`
(B0271), `68cc4e4fe` (B0270), `83da57b1e` (B0269), `ae1e088e2` (B0268),
`159a45aec` (B0267), `101135c5c` (B0266), `942a4e285` (B0265), `1ab13243b`
(B0264), `f41d5404b` (B0263), `71b7fd4fc` (B0262), `3610e61ef` (--fast),
`834f3abdb` (B0261), `d7f37fc63` (B0260), `d01139f0a` (B0259), `a20cf3d29`
(B0258).

##### Next slice

Maintainer decisions pending on the report (which CUDA kernels to propose
upstream; whether to keep the r4 selector retired; whether the profile lock
on automatic CUDA selection should be widened via the Vast qualification
protocol). Engineering candidates remain as listed in
`IMPLEMENTATION_PLAN.md` "Next vertical slice" (r6's CUDA start-up and LAZ
encode of 100-byte records, r2, `filters.stats`, colorization sampling).

##### Exact resume protocol

Reread the governing documents, then B0275/D0275 and this subsection. Verify
the anchors with `sha256sum`, then run the B0274 resume ctest set.

#### Final session stop — 2026-08-15 UTC (B0274/D0274) — retained; superseded by B0275 above

**This subsection was the authoritative resume point before B0275.** Since the B0273 stop:
every KD2/KD3 nanoflann tree builds concurrently with identical structure
(B0274/D0274: vendored `divideTreeConcurrent`, allocator mutex, at most
eight threads under the shared policy, snapshot fill on fixed chunks): r6
**10.689432x / 10.523255x**, r11 **12.892762x / 11.924286x**, r4
**8.377378x / 8.499474x**; aggregates **2.900213x / 2.888163x** geometric
mean and **5.404502x / 5.360525x** total wall, all exact. Full physical
CUDA 849/849, Host Debug 542/542, focused ASan/UBSan and TSan pass. The
plan's documented candidate list is exhausted (see IMPLEMENTATION_PLAN.md
"Next vertical slice"); the next deliverable is the cross-machine/big-cloud
benchmark report (B0275) requested by the maintainer. No build, test,
benchmark, profiler, or delegated review is left running.

##### Session summary (B0257 -> B0274)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x |
| B0265/D0265 | pooled morphology in the fork SMRF port | r2 1.62x |
| B0266/D0266 | slot-pooled exact reprojection (stream + standard) | r8 2.10x, r1 1.49x |
| B0267/D0267 | hashed sample voxel table | r4 5.31x |
| B0268/D0268 | parallel LAS record packing (stream + standard) | r14 2.08x, r13 1.95x; suite 2.13x GM, 4.06x total |
| B0269/D0269 | parallel LAS record unpacking (reader batch hook + per-tile) | r14 2.66x, r13 2.68x, r7 1.32x; suite 2.56x GM, 4.66x total |
| B0270/D0270 | ordered parallel COPC decode under requests=1 | r5 1.83x; suite 2.61x GM, 4.56x total |
| B0271/D0271 | `--fast` tie-order contract (device tie choices, no CPU repair) | fast: r6 11.59x, r2 2.04x; fast suite 2.71x GM, 4.92x total; exact unchanged |
| B0272/D0272 | automatic r4 CUDA outlier selector retired (host path faster) | r4 7.42x; suite 2.70x GM, 4.85x total |
| B0273/D0273 | banded parallel raster accumulation in writers.gdal | r3 2.08x; suite 2.77x GM, 5.00x total |
| B0274/D0274 | concurrent structure-identical KD builds | r6 10.69x, r11 12.89x, r4 8.38x; suite 2.90x GM, 5.40x total |

Suite trajectory this session (exact contract): 1.267705x/1.278088x
geometric mean and 1.656538x/1.665729x total wall (B0256) ->
**2.900213x/2.888163x** and **5.404502x/5.360525x** (B0274), all fourteen
headlines exact throughout.

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `abd3cbe9c08535a99ca3eb6d928a89872adbb717bd53f8cbf89fbd993882eb7e` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `c7afa7ba2d596562c7d23f04d9a22d13fecee9f5ed8a8e2c3556438338e2fd26` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `ea113c56be1498e81088c8900f6c15bf86a2deec34c8a3dfccebd440597084d7` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| r6 / r11 / r4 gates | `build/benchmarks/b0274-{r6-features,r11-classify-refine,r4-denoise-thin}-kdbuild-final3-{warm,cold}.json` | `82ef2f08...`/`bfd844ee...`, `4b2f8fe4...`/`5d1e07ff...`, `1ffe1242...`/`e802433d...` |
| warm / cold aggregate | `build/benchmarks/b0274-suite3-{warm,cold}.json` | `c3ffaf2d...` / `5c9f7b87...` |
| nanoflann / KD adapters | `vendor/nanoflann/nanoflann.hpp` / `pdal/private/KDImpl.hpp` | `fad9880b...` / `d1df91ca...` |

##### Validation completed at this boundary

Full physical CUDA aggregate **849/849** with the KD3 snapshot verifier
armed (21 documented optional skips, 514.72 s); Host Debug 542/542 (four
optional skips); focused leak-disabled ASan/UBSan and TSan runs of the KD3
units and the r11/SMRF/outlier matrices (10/10 each). Note: the differential
case directories and benchmark work directories were purged after the disk
filled (regenerable; reports and JSON evidence retained); every tree was
reconfigured so all tests are registered again.

##### Workspace checkpoint

Committed locally on `p3-skewness-resident` (not pushed) as `HEAD`, subject
`Checkpoint concurrent KD builds through B0274`. `git status --short` must
be empty at handoff. Earlier checkpoints this session: `0897bf990`
(B0273), `632ee0508` (B0272), `ba169bd73` (B0271), `68cc4e4fe` (B0270),
`83da57b1e` (B0269), `ae1e088e2` (B0268), `159a45aec` (B0267), `101135c5c`
(B0266), `942a4e285` (B0265), `1ab13243b` (B0264), `f41d5404b` (B0263),
`71b7fd4fc` (B0262), `3610e61ef` (--fast), `834f3abdb` (B0261), `d7f37fc63`
(B0260), `d01139f0a` (B0259), `a20cf3d29` (B0258).

##### Next slice

B0275: the maintainer's cross-machine and big-cloud benchmark report — a
rented Vast.ai RTX 4090/EPYC box (bench/remote/vast_bootstrap.sh,
vast_run.sh), the fourteen headlines at 1M under default, experimental-CUDA,
and `--fast` configurations, AHN4 GeoTiles as big clouds, LAStools (open
tools built from source; proprietary tools in `-demo` mode under wine)
timings, rendered outputs, and the upstream-merge candidate list.

##### Exact resume protocol

Reread the governing documents, then B0274/D0274 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'FastMode|Kd3|Smrf|HybridPipeline|LasSummaryMerge|LasReaderUnpack|pdg_las_(writer_pack|reader_unpack)|pdg_copc_reader|pdg_gdal_writer|pdg_fast_tie|pdg_automatic_r4|pdg_reprojection|pdg_sample|pdg_(r11_classification|outlier|smrf|r14_conversion|r7_dsm)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process|pdg_reference_runner' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0273/D0273) — retained; superseded by B0274 above

**This subsection was the authoritative resume point before B0274.** Since the B0272 stop:
`writers.gdal` accumulates rasters on row bands in parallel (B0273/D0273:
each band replays the full point order with the pinned walk restricted to
its rows; standard mode, fixed streaming grids, and bin mode; streaming
dynamic radius grids and percentiles stay serial to reproduce pinned
interleaving): r3 **2.077188x / 2.037489x** (from 1.59x), r7 unchanged
within noise; aggregates **2.772537x / 2.727890x** geometric mean and
**5.002094x / 4.851449x** total wall, all exact — the exact suite's first
5x warm total. Host Debug 541/541; focused ASan/UBSan and TSan pass;
CUDA-tree raster lanes 28/28; the full physical CUDA aggregate was 847/847
at B0272 and no CUDA translation unit or executor changed since. No build,
test, benchmark, profiler, or delegated review is left running.

##### Session summary (B0257 -> B0273)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x |
| B0265/D0265 | pooled morphology in the fork SMRF port | r2 1.62x |
| B0266/D0266 | slot-pooled exact reprojection (stream + standard) | r8 2.10x, r1 1.49x |
| B0267/D0267 | hashed sample voxel table | r4 5.31x |
| B0268/D0268 | parallel LAS record packing (stream + standard) | r14 2.08x, r13 1.95x; suite 2.13x GM, 4.06x total |
| B0269/D0269 | parallel LAS record unpacking (reader batch hook + per-tile) | r14 2.66x, r13 2.68x, r7 1.32x; suite 2.56x GM, 4.66x total |
| B0270/D0270 | ordered parallel COPC decode under requests=1 | r5 1.83x; suite 2.61x GM, 4.56x total |
| B0271/D0271 | `--fast` tie-order contract (device tie choices, no CPU repair) | fast: r6 11.59x, r2 2.04x; fast suite 2.71x GM, 4.92x total; exact unchanged |
| B0272/D0272 | automatic r4 CUDA outlier selector retired (host path faster) | r4 7.42x; suite 2.70x GM, 4.85x total |
| B0273/D0273 | banded parallel raster accumulation in writers.gdal | r3 2.08x; suite 2.77x GM, 5.00x total |

Suite trajectory this session (exact contract): 1.267705x/1.278088x
geometric mean and 1.656538x/1.665729x total wall (B0256) ->
**2.772537x/2.727890x** and **5.002094x/4.851449x** (B0273), all fourteen
headlines exact throughout. Fast contract (B0271, before B0272/B0273):
2.711573x/2.712707x and 4.923939x/4.927839x.

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `abd3cbe9c08535a99ca3eb6d928a89872adbb717bd53f8cbf89fbd993882eb7e` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `6de500c5ece48a2eb914f36c241edf7300d47eb1e751ee2570042e9663d40c34` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `5a602f7d5f651d824a52b4c9b73f06aead78b239e0faf0d8a83927ce1bf1e102` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| r3 / r7 gates | `build/benchmarks/b0273-{r3-dtm,r7-dsm}-bands-final2-{warm,cold}.json` | `2989293c...`/`bd28878d...`, `401e20f3...`/`c61e2195...` |
| warm / cold aggregate | `build/benchmarks/b0273-suite2-{warm,cold}.json` | `53bc49a5...` / `d46d0302...` |
| GDAL writer / grid | `io/GDALWriter.cpp` / `io/private/GDALGrid.cpp` | `6aa39a57...` / `5f03747c...` |
| GDAL matrix | `tests/differential/gdal_writer_matrix.py` | `2f0e1f8b...` |

##### Validation completed at this boundary

Host Debug 541/541 (four optional skips) including the new GDAL writer
matrix; focused leak-disabled ASan/UBSan and TSan runs of the matrix pass;
CUDA-tree raster/r7/r3 lanes 28/28; full physical CUDA aggregate 847/847 at
B0272 (no CUDA or executor change since).

##### Workspace checkpoint

Committed locally on `p3-skewness-resident` (not pushed) as `HEAD`, subject
`Checkpoint banded raster accumulation through B0273`. `git status --short`
must be empty at handoff. Earlier checkpoints this session: `632ee0508`
(B0272), `ba169bd73` (B0271), `68cc4e4fe` (B0270), `83da57b1e` (B0269),
`ae1e088e2` (B0268), `159a45aec` (B0267), `101135c5c` (B0266), `942a4e285`
(B0265), `1ab13243b` (B0264), `f41d5404b` (B0263), `71b7fd4fc` (B0262),
`3610e61ef` (--fast), `834f3abdb` (B0261), `d7f37fc63` (B0260), `d01139f0a`
(B0259), `a20cf3d29` (B0258).

##### Next slice

`IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm walls r6
0.95 s (CUDA init 0.18, LAZ encode of 100-byte records, columns, repair),
r2 0.82 s, r11 0.67 s, r4 0.63 s, r8 0.62 s (reprojection knee at eight
slots), r3 0.32 s (SMRF 0.17 s); remaining serial per-point streaming host
passes (`filters.stats` dimension-parallel, `readers.copc` per-point
filter/unpack); further `--fast`-only routes; colorization's GDAL sampling.

##### Exact resume protocol

Reread the governing documents, then B0273/D0273 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'FastMode|Kd3|Smrf|HybridPipeline|LasSummaryMerge|LasReaderUnpack|pdg_las_(writer_pack|reader_unpack)|pdg_copc_reader|pdg_gdal_writer|pdg_fast_tie|pdg_automatic_r4|pdg_reprojection|pdg_sample|pdg_(r11_classification|outlier|smrf|r14_conversion|r7_dsm)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process|pdg_reference_runner' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0272/D0272) — retained; superseded by B0273 above

**This subsection was the authoritative resume point before B0273.** Since the B0271 stop:
the automatic r4 CUDA outlier selector (B0227) is retired from default
selection because the exact host path now measures faster (B0272/D0272:
0.582 s host vs 0.718 s route at 1M, 2.55 s vs 2.78 s forced at 4M,
byte-identical; the route stays behind
`PDG_EXPERIMENTAL_AUTOMATIC_R4_OUTLIER_CUDA` for its lane). r4 **7.422330x /
7.440248x**; aggregates **2.700250x / 2.632576x** geometric mean and
**4.853772x / 4.644633x** total wall, all exact (the cold suite ran under a
heavier desktop load). Full physical CUDA 847/847 (21 optional skips), Host
Debug 540/540. No build, test, benchmark, profiler, or delegated review is
left running.

Note for the maintainer: this is the first recorded case of a host slice
overtaking a qualified CUDA route. It follows the plan's measured-selection
rule (D0239, "choose the faster on the reference device") and is recorded
plainly; if the marketing goal should keep the CUDA route selected despite
the measured loss, D0272 is the entry to revisit.

##### Session summary (B0257 -> B0272)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x |
| B0265/D0265 | pooled morphology in the fork SMRF port | r2 1.62x |
| B0266/D0266 | slot-pooled exact reprojection (stream + standard) | r8 2.10x, r1 1.49x |
| B0267/D0267 | hashed sample voxel table | r4 5.31x |
| B0268/D0268 | parallel LAS record packing (stream + standard) | r14 2.08x, r13 1.95x; suite 2.13x GM, 4.06x total |
| B0269/D0269 | parallel LAS record unpacking (reader batch hook + per-tile) | r14 2.66x, r13 2.68x, r7 1.32x; suite 2.56x GM, 4.66x total |
| B0270/D0270 | ordered parallel COPC decode under requests=1 | r5 1.83x; suite 2.61x GM, 4.56x total |
| B0271/D0271 | `--fast` tie-order contract (device tie choices, no CPU repair) | fast: r6 11.59x, r2 2.04x; fast suite 2.71x GM, 4.92x total; exact unchanged |
| B0272/D0272 | automatic r4 CUDA outlier selector retired (host path faster) | r4 7.42x; suite 2.70x GM, 4.85x total |

Suite trajectory this session (exact contract): 1.267705x/1.278088x
geometric mean and 1.656538x/1.665729x total wall (B0256) ->
**2.700250x/2.632576x** and **4.853772x/4.644633x** (B0272), all fourteen
headlines exact throughout. Fast contract (B0271, before B0272's r4 change):
2.711573x/2.712707x and 4.923939x/4.927839x.

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `abd3cbe9c08535a99ca3eb6d928a89872adbb717bd53f8cbf89fbd993882eb7e` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `0a2db805546e0d79b2ec7568124fbf87eba8901a4e2d99312481b394c8536aaf` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `456adf3f7b0f8c9205f422a6e3d679421c8fd9a1c66a48ca97fe4d9830fcb668` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| r4 warm / cold | `build/benchmarks/b0272-r4-denoise-thin-host-final-{warm,cold}.json` | `c8a05639...` / `ed4974d9...` |
| warm / cold aggregate | `build/benchmarks/b0272-suite-{warm,cold}.json` | `fb2b54d6...` / `2ab49263...` |
| hybrid planner / launcher inventory | `src/plan/Hybrid.cpp` / `src/cli/Dispatch.cpp` | `bd5589f1...` / `7a99a86b...` |
| r4 lane | `tests/differential/automatic_r4_outlier.py` | `f6198627...` |

##### Validation completed at this boundary

Full physical CUDA aggregate **847/847** with the KD3 snapshot verifier
armed (21 documented optional skips, 528.79 s); Host Debug 540/540 (four
optional skips). No CUDA translation unit changed since B0271's Compute
Sanitizer runs.

##### Workspace checkpoint

Committed locally on `p3-skewness-resident` (not pushed) as `HEAD`, subject
`Checkpoint the retired r4 CUDA selector through B0272`. `git status
--short` must be empty at handoff. Earlier checkpoints this session:
`ba169bd73` (B0271), `68cc4e4fe` (B0270), `83da57b1e` (B0269), `ae1e088e2`
(B0268), `159a45aec` (B0267), `101135c5c` (B0266), `942a4e285` (B0265),
`1ab13243b` (B0264), `f41d5404b` (B0263), `71b7fd4fc` (B0262), `3610e61ef`
(--fast), `834f3abdb` (B0261), `d7f37fc63` (B0260), `d01139f0a` (B0259),
`a20cf3d29` (B0258).

##### Next slice

`IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm walls r6
0.97 s, r2 0.84 s, r11 0.68 s, r8 0.63 s (reprojection knee at eight
slots), r4 0.62 s, r3 0.43 s; remaining serial per-point streaming host
passes (`filters.stats` dimension-parallel, `readers.copc` per-point
filter/unpack) through the batch hooks; further `--fast`-only routes;
r6's CUDA startup/preflight fixed costs; colorization's GDAL sampling.

##### Exact resume protocol

Reread the governing documents, then B0272/D0272 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'FastMode|Kd3|Smrf|HybridPipeline|LasSummaryMerge|LasReaderUnpack|pdg_las_(writer_pack|reader_unpack)|pdg_copc_reader|pdg_fast_tie|pdg_automatic_r4|pdg_reprojection|pdg_sample|pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process|pdg_reference_runner' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0271/D0271) — retained; superseded by B0272 above

**This subsection was the authoritative resume point before B0272.** Since the B0270 stop:
the maintainer's `--fast` tie-order request is implemented (B0271/D0271):
under `pdg --fast` the spatial index publishes no `KnnDistanceTie`
(per-build `__constant__` mask in both gather kernels and the host-index
paths via `pdg::knnStatusMask()`, `include/pdg/FastMode.hpp`), so device tie
choices stand and no CPU tie repair runs; the launcher consumes `--fast` on
every route (an environment-route gap was fixed); the reference runners
gained the record-by-record fast comparator and labeled fast aggregates.
Fast-contract finals: r6 **11.587544x / 11.206291x** (25 of 1M records
differ, attributes only) and r2 **2.043481x / 2.128425x** (125 records);
fast aggregates **2.711573x / 2.712707x** geometric mean and **4.923939x /
4.927839x** total wall; the exact suite on the same binaries is
byte-identical (2.660624x / 4.755153x warm, within noise of B0270's claim,
which stays the exact claim of record). Full physical CUDA 847/847 (21
optional skips), Host Debug 540/540, focused ASan/UBSan and TSan 29/29,
Compute Sanitizer clean on the changed kernels. No build, test, benchmark,
profiler, or delegated review is left running.

##### Session summary (B0257 -> B0271)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x |
| B0265/D0265 | pooled morphology in the fork SMRF port | r2 1.62x |
| B0266/D0266 | slot-pooled exact reprojection (stream + standard) | r8 2.10x, r1 1.49x |
| B0267/D0267 | hashed sample voxel table | r4 5.31x |
| B0268/D0268 | parallel LAS record packing (stream + standard) | r14 2.08x, r13 1.95x; suite 2.13x GM, 4.06x total |
| B0269/D0269 | parallel LAS record unpacking (reader batch hook + per-tile) | r14 2.66x, r13 2.68x, r7 1.32x; suite 2.56x GM, 4.66x total |
| B0270/D0270 | ordered parallel COPC decode under requests=1 | r5 1.83x; suite 2.61x GM, 4.56x total (repeat 2.67x / 4.73x) |
| B0271/D0271 | `--fast` tie-order contract (device tie choices, no CPU repair) | fast: r6 11.59x, r2 2.04x; fast suite 2.71x GM, 4.92x total; exact unchanged |

Suite trajectory this session (exact contract): 1.267705x/1.278088x
geometric mean and 1.656538x/1.665729x total wall (B0256) ->
**2.613780x/2.624146x** and **4.559263x/4.574280x** (B0270 claim; B0271's
same-binary rerun 2.660624x/4.755153x warm), all fourteen headlines exact
throughout. Fast contract (B0271): 2.711573x/2.712707x and
4.923939x/4.927839x with two headlines differing on tie rows only.

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `0c7a844b902ad93a4fcabf8fb15eb715ac15bae1d8a019d18f3bd7744298ec13` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `cc0a490d506506fbbfbe1cbf43753155d80c97a0c5498c9f69ae3b7c6fec4cb5` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `456adf3f7b0f8c9205f422a6e3d679421c8fd9a1c66a48ca97fe4d9830fcb668` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| fast r6 / r2 gates | `build/benchmarks/b0271-{r6-features,r2-ground-normalize}-fast-final2-{warm,cold}.json` | `34d6e338...`/`0ce297f9...`, `28447833...`/`0607ade9...` |
| fast warm / cold aggregate; exact warm control | `build/benchmarks/b0271-fast-suite2-{warm,cold}.json`, `b0271-exact-suite2-warm.json` | `c5efd4f9...` / `f63c3256...`; `3553774a...` |
| fast mode core | `include/pdg/FastMode.hpp`, `src/core/FastMode.cpp` | `126e55d7...`, `b14c9dd9...` |
| gather kernels / host index | `src/index/SpatialIndexKernels.cu` / `SpatialIndex.cpp` | `d64b4b80...` / `791ee58d...` |
| launcher main | `src/cli/DispatchMain.cpp` | `daeba3df...` |
| runners | `scripts/pdg/benchmark_reference.py` / `reference_suite.py` | `58983e0e...` / `0ef65261...` |

##### Validation completed at this boundary

Full physical CUDA aggregate **847/847** with the KD3 snapshot verifier
armed (21 documented optional skips, 544.21 s) on the final binaries; Host
Debug 540/540 (four optional skips); focused leak-disabled ASan/UBSan and
TSan runs of `FastMode`, the spatial index units, the dispatcher process
test, and the runner contract test (29/29 each); Compute Sanitizer memcheck,
racecheck, initcheck, and synccheck clean over the changed gather kernels.

##### Workspace checkpoint

Committed locally on `p3-skewness-resident` (not pushed) as `HEAD`, subject
`Checkpoint the --fast tie-order contract through B0271`. `git status
--short` must be empty at handoff. Earlier checkpoints this session:
`68cc4e4fe` (B0270), `83da57b1e` (B0269), `ae1e088e2` (B0268), `159a45aec`
(B0267), `101135c5c` (B0266), `942a4e285` (B0265), `1ab13243b` (B0264),
`f41d5404b` (B0263), `71b7fd4fc` (B0262), `3610e61ef` (--fast), `834f3abdb`
(B0261), `d7f37fc63` (B0260), `d01139f0a` (B0259), `a20cf3d29` (B0258).

##### Next slice

`IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm walls r6
0.98 s, r2 0.86 s, r4 0.81 s, r11 0.68 s, r8 0.62 s, r3 0.41 s; remaining
serial per-point streaming host passes (`filters.stats` dimension-parallel,
`readers.copc` per-point filter/unpack, range/assign/crop) through the batch
hooks; further `--fast`-only routes now that the contract admits device tie
choices (measure first); r6's CUDA startup/preflight fixed costs;
colorization's GDAL sampling.

##### Exact resume protocol

Reread the governing documents, then B0271/D0271 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'FastMode|Kd3|Smrf|LasSummaryMerge|LasReaderUnpack|pdg_las_(writer_pack|reader_unpack)|pdg_copc_reader|pdg_fast_tie|pdg_reprojection|pdg_sample|pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process|pdg_reference_runner' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0270/D0270) — retained; superseded by B0271 above

**This subsection was the authoritative resume point before B0271.** Since the B0269 stop:
`readers.copc` under `requests=1` keeps one request thread but decompresses
tiles on a decode pool and emits them in pinned fetch order (B0270/D0270):
r5 **1.830419x / 1.943686x** (from parity; the last headline at parity);
aggregates **2.613780x / 2.624146x** geometric mean and **4.559263x /
4.574280x** total wall (warm repeat 2.674064x / 4.727176x), all exact; all
twelve zero-weight variants exact and above parity. Host Debug 538/538 (four
optional skips); focused ASan/UBSan (with the new `vendor/lazperf`-scoped
UBSan suppression) and TSan pass; CUDA-tree COPC/conversion/hybrid lanes
4/4; the full physical CUDA aggregate was 840/840 at B0269 and no CUDA
translation unit or executor code changed since. No build, test, benchmark,
profiler, or delegated review is left running.

Open maintainer request (received mid-B0270; implemented as D0271/B0271
above): under `--fast`, allow order-insensitive kNN tie choices — drop the exact CPU tie
repairs (r6 eigen family, HAG-NN selective repair, and any other tie
repair) and let the device result stand for tie points; the default path
stays byte-identical. This revises D0261's `--fast` contract (bit-identical
numerics) and is the next slice (D0271).

##### Session summary (B0257 -> B0270)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x |
| B0265/D0265 | pooled morphology in the fork SMRF port | r2 1.62x |
| B0266/D0266 | slot-pooled exact reprojection (stream + standard) | r8 2.10x, r1 1.49x |
| B0267/D0267 | hashed sample voxel table | r4 5.31x |
| B0268/D0268 | parallel LAS record packing (stream + standard) | r14 2.08x, r13 1.95x; suite 2.13x GM, 4.06x total |
| B0269/D0269 | parallel LAS record unpacking (reader batch hook + per-tile) | r14 2.66x, r13 2.68x, r7 1.32x; suite 2.56x GM, 4.66x total |
| B0270/D0270 | ordered parallel COPC decode under requests=1 | r5 1.83x; suite 2.61x GM, 4.56x total (repeat 2.67x / 4.73x) |

Suite trajectory this session: 1.267705x/1.278088x geometric mean and
1.656538x/1.665729x total wall (B0256) -> **2.613780x/2.624146x** and
**4.559263x/4.574280x** (B0270), all fourteen headlines exact throughout.

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `879fdccc598872e393cd65c345b8291808441834cec260c6474da7cdc95f921b` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `be40417afebcfc6dacd8821fd73cecd0e7e1125d6c3097d11fcfca300c4bbf59` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `456adf3f7b0f8c9205f422a6e3d679421c8fd9a1c66a48ca97fe4d9830fcb668` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| r5 warm / cold | `build/benchmarks/b0270-r5-copc-query-copc-final-{warm,cold}.json` | `a261dd39...` / `4a521ce2...` |
| warm / cold / warm-repeat aggregate | `build/benchmarks/b0270-suite-{warm,cold}.json`, `b0270-suite2-warm.json` | `70a53100...` / `f7652634...` / `24abaeea...` |
| COPC reader | `io/CopcReader.cpp` / `.hpp` | `1f952f69...` / `153907dd...` |
| COPC tile | `io/private/copc/Tile.cpp` / `.hpp` | `beb0ac8e...` / `da42eafa...` |
| COPC matrix | `tests/differential/copc_reader_matrix.py` | `460a1e9b...` |

##### Validation completed at this boundary

Host Debug 538/538 (four optional skips) including the new COPC matrix;
focused leak-disabled ASan/UBSan (lazperf UBSan suppression injected by
CMake) and TSan runs of the COPC, r14 conversion, hybrid reader, and LAS
reader matrices pass; CUDA-tree COPC/conversion/hybrid lanes 4/4; full
physical CUDA aggregate 840/840 at B0269 (no CUDA or executor change since).

##### Workspace checkpoint

Committed locally on `p3-skewness-resident` (not pushed) as `HEAD`, subject
`Checkpoint ordered parallel COPC decode through B0270`. `git status --short`
must be empty at handoff. Earlier checkpoints this session: `83da57b1e`
(B0269), `ae1e088e2` (B0268), `159a45aec` (B0267), `101135c5c` (B0266),
`942a4e285` (B0265), `1ab13243b` (B0264), `f41d5404b` (B0263), `71b7fd4fc`
(B0262), `3610e61ef` (--fast), `834f3abdb` (B0261), `d7f37fc63` (B0260),
`d01139f0a` (B0259), `a20cf3d29` (B0258).

##### Next slice

D0271: the maintainer's `--fast` tie-insensitive contract (above), then
`IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm walls r6
0.98 s, r2 0.86 s, r4 0.81 s, r11 0.68 s, r8 0.62 s, r3 0.41 s; remaining
serial per-point streaming host passes (`filters.stats` dimension-parallel,
`readers.copc` per-point filter/unpack, range/assign/crop) through the batch
hooks; r6's CUDA startup/preflight fixed costs; colorization's GDAL sampling.

##### Exact resume protocol

Reread the governing documents, then B0270/D0270 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'Kd3|Smrf|LasSummaryMerge|LasReaderUnpack|pdg_las_(writer_pack|reader_unpack)|pdg_copc_reader|pdg_reprojection|pdg_sample|pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0269/D0269) — retained; superseded by B0270 above

**This subsection was the authoritative resume point before B0270.** Since the B0268 stop:
`readers.las` unpacks decoded records on a fixed-slot pool in both execution
modes (B0269/D0269: an opt-in `Streamable::readStreamBatch` reader hook in
the streaming executor and per-tile runs in the standard `read()`; tile
consumption stays serial and pinned; small runs, streaming `start`, and the
read callback stay serial): r14 **2.656026x / 2.614287x**, r13 **2.682353x /
2.700326x**, r7 **1.320508x / 1.307258x** (from parity), r9 **1.901385x /
1.911693x**, r1 **2.114222x / 2.107619x**, r12 **1.946695x / 1.902648x**, r10
**1.446846x / 1.423698x**, r3 **1.586718x / 1.579299x**, r6 **10.122291x /
9.872648x**; aggregates **2.558472x / 2.555054x** geometric mean and
**4.662196x / 4.655083x** total wall, all exact; all twelve zero-weight
variants exact and above parity. Host Debug 537/537 (four optional skips);
focused ASan/UBSan and TSan pass (9/9 each); the full physical CUDA aggregate
is 840/840 with the KD3 verifier armed (21 documented optional skips).
No build, test, benchmark, profiler, or delegated review is left running.

##### Session summary (B0257 -> B0269)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x |
| B0265/D0265 | pooled morphology in the fork SMRF port | r2 1.62x |
| B0266/D0266 | slot-pooled exact reprojection (stream + standard) | r8 2.10x, r1 1.49x |
| B0267/D0267 | hashed sample voxel table | r4 5.31x |
| B0268/D0268 | parallel LAS record packing (stream + standard) | r14 2.08x, r13 1.95x; suite 2.13x GM, 4.06x total |
| B0269/D0269 | parallel LAS record unpacking (reader batch hook + per-tile) | r14 2.66x, r13 2.68x, r7 1.32x; suite 2.56x GM, 4.66x total |

Suite trajectory this session: 1.267705x/1.278088x geometric mean and
1.656538x/1.665729x total wall (B0256) -> **2.558472x/2.555054x** and
**4.662196x/4.655083x** (B0269), all fourteen headlines exact throughout.

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `879fdccc598872e393cd65c345b8291808441834cec260c6474da7cdc95f921b` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `69e4e557470f32a32b68ff48d6547fc301beff8b85220661fccfff7e45f6d3c3` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `399f5a805481a455235c1e43a2f0b79158797011e9f3d9670d22516eb001ff46` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| nine-pair gates | `build/benchmarks/b0269-{r14-convert-compress,r13-merge,r7-dsm,r9-polygon-clip,r1-translate,r12-tile,r10-decimate,r3-dtm,r6-features}-unpack-final-{warm,cold}.json` | `278323ba...`/`c66c8a02...`, `8cca39ea...`/`ef577941...`, `4750c229...`/`e0760e73...`, `77c1ce07...`/`3a261dc2...`, `24c4710f...`/`d316d509...`, `3ca7eba3...`/`526dc491...`, `d18a95f0...`/`2d79bd1a...`, `515b9c19...`/`01042e67...`, `76ca8171...`/`4bdbcc33...` |
| warm / cold aggregate | `build/benchmarks/b0269-suite-{warm,cold}.json` | `6a65b84e...` / `fa213064...` |
| LAS reader | `io/LasReader.cpp` / `.hpp` | `95198d4b...` / `342f1d42...` |
| streaming executor | `pdal/Streamable.cpp` / `.hpp` | `546b4602...` / `c884092a...` |
| reader matrix | `tests/differential/las_reader_unpack_matrix.py` | `c7c240fb...` |

##### Validation completed at this boundary

Host Debug 537/537 (four optional skips) including the new reader matrix
and `LasReaderUnpack` unit; focused leak-disabled ASan/UBSan and TSan runs
of the reader and writer matrices, `LasReaderUnpack`, `LasSummaryMerge`, the
r14 conversion and reprojection matrices, and `LazChunkCompression` pass
(9/9 each); full physical CUDA aggregate **840/840** with the KD3 snapshot verifier
armed (21 documented optional skips, 493.36 s).

##### Workspace checkpoint

Committed locally on `p3-skewness-resident` (not pushed) as `HEAD`, subject
`Checkpoint parallel LAS record unpacking through B0269`. `git status --short`
must be empty at handoff. Earlier checkpoints this session: `ae1e088e2`
(B0268), `159a45aec` (B0267), `101135c5c` (B0266), `942a4e285` (B0265),
`1ab13243b` (B0264), `f41d5404b` (B0263), `71b7fd4fc` (B0262), `3610e61ef`
(--fast), `834f3abdb` (B0261), `d7f37fc63` (B0260), `d01139f0a` (B0259),
`a20cf3d29` (B0258).

##### Next slice

See `IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm walls r6
0.97 s, r2 0.84 s, r4 0.78 s, r11 0.68 s, r8 0.63 s, r3 0.41 s, r10 0.28 s,
r12 0.26 s, r5 0.25 s. Candidates: remaining serial per-point streaming host
passes through the batch hooks; `readers.copc` unpacking (r5 is the only
headline at parity); r6's CUDA startup/preflight fixed costs; colorization's
GDAL sampling; LAZ chunk decode. `--fast` stays behavior-neutral until
measured.

##### Exact resume protocol

Reread the governing documents, then B0269/D0269 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'Kd3|Smrf|LasSummaryMerge|LasReaderUnpack|pdg_las_(writer_pack|reader_unpack)|pdg_reprojection|pdg_sample|pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0268/D0268) — retained; superseded by B0269 above

**This subsection was the authoritative resume point before B0269.** Since the B0267 stop:
`writers.las` packs point records on a fixed-slot pool in both execution
modes (B0268/D0268: standard blocks and a `processStreamBatch` override,
slot-ordered `las::Summary` merges, and a serial repeat whenever any slot
would have logged a per-point warning or threw, so streams, diagnostics,
and partial output stay pinned): r14 **2.084120x / 2.048271x**, r13
**1.949116x / 1.963703x**, r12 **1.456797x / 1.459860x**, r6 **9.136681x /
9.095218x**, r1 **1.573685x / 1.580186x**; aggregates **2.133083x /
2.122241x** geometric mean and **4.063740x / 4.028901x** total wall, all
exact, twelve zero-weight variants exact. Host Debug 531/531 (535 registered,
four optional skips); focused ASan/UBSan and TSan pass; the full physical CUDA aggregate
is 838/838 with the KD3 verifier armed (21 documented optional skips).
No build, test, benchmark, profiler, or delegated review is left running.

##### Session summary (B0257 -> B0268)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x |
| B0265/D0265 | pooled morphology in the fork SMRF port | r2 1.62x |
| B0266/D0266 | slot-pooled exact reprojection (stream + standard) | r8 2.10x, r1 1.49x |
| B0267/D0267 | hashed sample voxel table | r4 5.31x |
| B0268/D0268 | parallel LAS record packing (stream + standard) | r14 2.08x, r13 1.95x, r12 1.46x, r6 9.14x, r1 1.57x; suite 2.13x GM, 4.06x total |

Suite trajectory this session: 1.267705x/1.278088x geometric mean and
1.656538x/1.665729x total wall (B0256) -> **2.133083x/2.122241x** and
**4.063740x/4.028901x** (B0268), all fourteen headlines exact throughout.

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `879fdccc598872e393cd65c345b8291808441834cec260c6474da7cdc95f921b` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `14cc478bcbef00e39d9ca15d74cd59cf3bf50464195d715d44f2efd54a1f5fbe` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `cf2f3d9a9f296dc6571b06bc64cf6101ba4ec0385c912ad2d240c69413f7810e` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| r14 / r13 / r12 / r6 / r1 nine-pair gates | `build/benchmarks/b0268-{r14-convert-compress,r13-merge,r12-tile,r6-features,r1-translate}-pack-final-{warm,cold}.json` | `14e14e3b...`/`d7d7e6ff...`, `4909387d...`/`6ebc82c7...`, `7dc72c24...`/`71334add...`, `8e8d2e9e...`/`292aeaa8...`, `f31d05e9...`/`26747348...` |
| warm / cold aggregate | `build/benchmarks/b0268-suite-{warm,cold}.json` | `4a6b2f1c...` / `133ac0e2...` |
| LAS writer | `io/LasWriter.cpp` / `.hpp` | `a70a1418...` / `04e965f0...` |
| writer matrix | `tests/differential/las_writer_pack_matrix.py` | `2fb471c9...` |

##### Validation completed at this boundary

Host Debug 531/531 (535 registered, four optional skips) including the new
writer matrix and `LasSummaryMerge` units; focused leak-disabled ASan/UBSan
and TSan runs of the writer matrix, `LasSummaryMerge`, the r14 conversion
matrix, and `LazChunkCompression` pass; CUDA-tree focused lane 74/74;
full physical CUDA aggregate **838/838** with the KD3 snapshot verifier
armed (21 documented optional skips, 487.58 s).

##### Workspace checkpoint

Committed locally on `p3-skewness-resident` (not pushed) as `HEAD`, subject
`Checkpoint parallel LAS record packing through B0268`. `git status --short`
must be empty at handoff. Earlier checkpoints this session: `159a45aec`
(B0267), `101135c5c` (B0266), `942a4e285` (B0265), `1ab13243b` (B0264),
`f41d5404b` (B0263), `71b7fd4fc` (B0262), `3610e61ef` (--fast), `834f3abdb`
(B0261), `d7f37fc63` (B0260), `d01139f0a` (B0259), `a20cf3d29` (B0258).

##### Next slice

See `IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm walls r6
1.02 s, r2 0.88 s, r4 0.85 s, r11 0.76 s, r8 0.69 s, r3 0.49 s. Candidates:
LAZ decode (present in every LAZ workload); r6's CUDA startup/preflight fixed
costs; colorization's GDAL sampling; remaining serial per-point streaming
host passes through the batch hook. `--fast` stays behavior-neutral until
measured.

##### Exact resume protocol

Reread the governing documents, then B0268/D0268 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'Kd3|Smrf|LasSummaryMerge|pdg_las_writer_pack|pdg_reprojection|pdg_sample|pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0267/D0267) — retained; superseded by B0268 above

**This subsection was the authoritative resume point before B0268.** Since the B0266 stop:
`filters.sample`'s voxel table is hashed, its candidate lists are read in
place, and unreachable neighbor voxels are pruned (B0267/D0267): r4
**5.306974x / 5.271122x**; aggregates **1.902538x / 1.894000x** geometric
mean and **3.697978x / 3.657649x** total wall, all exact. Host Debug 531/531;
focused ASan/UBSan and TSan pass; the last full physical CUDA aggregate
(833/833, verifier armed) was at B0266 and no CUDA translation unit changed
since. No build, test, benchmark, profiler, or delegated review is left
running.

##### Session summary (B0257 -> B0267)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x |
| B0265/D0265 | pooled morphology in the fork SMRF port | r2 1.62x |
| B0266/D0266 | slot-pooled exact reprojection (stream + standard) | r8 2.10x, r1 1.49x |
| B0267/D0267 | hashed sample voxel table | r4 5.31x; suite 1.90x GM, 3.70x total |

Suite trajectory this session: 1.267705x/1.278088x geometric mean and
1.656538x/1.665729x total wall (B0256) -> **1.902538x/1.894000x** and
**3.697978x/3.657649x** (B0267), all fourteen headlines exact throughout.

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `879fdccc598872e393cd65c345b8291808441834cec260c6474da7cdc95f921b` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `3ce12e44fe801e73be911f0a63bec94e4084f1f9841dcb9b8cdded61419a9bb3` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `f38acc3f750e381954b9a59959b337878eb59567fb3f9cdaf2f7e846792bcfd1` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| r4 warm / cold | `build/benchmarks/b0267-r4-sample-hash-final-{warm,cold}.json` | `206d2c23...` / `b80c03d1...` |
| warm / cold aggregate | `build/benchmarks/b0267-suite-{warm,cold}.json` | `0eef0284...` / `562c5a9e...` |
| sample filter | `filters/SampleFilter.cpp` / `.hpp` | `cade4259...` / `191e171d...` |
| sample matrix | `tests/differential/sample_matrix.py` | `12ba1a11...` |

##### Validation completed at this boundary

Host Debug 531/531 (four optional skips); CUDA-tree automatic r4, hybrid
outlier, and sample matrix lanes pass; focused leak-disabled ASan/UBSan and
TSan runs of the sample matrix pass. The full physical CUDA aggregate was
833/833 with the KD3 verifier armed at B0266; no CUDA translation unit changed
this session.

##### Workspace checkpoint

Committed locally on `p3-skewness-resident` (not pushed) as `HEAD`, subject
`Checkpoint hashed sample voxel table through B0267`. `git status --short`
must be empty at handoff. Earlier checkpoints this session: `101135c5c`
(B0266), `942a4e285` (B0265), `1ab13243b` (B0264), `f41d5404b` (B0263),
`71b7fd4fc` (B0262), `3610e61ef` (--fast), `834f3abdb` (B0261), `d7f37fc63`
(B0260), `d01139f0a` (B0259), `a20cf3d29` (B0258).

##### Next slice

See `IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm walls r6
1.08 s, r2 0.95 s, r4 0.89 s, r11 0.84 s, r8 0.81 s, r3 0.49 s. Candidates:
LAZ decode and the LAS writer's per-point host path (present in every LAZ
workload); r6's CUDA startup/preflight fixed costs; colorization's GDAL
sampling. `--fast` stays behavior-neutral until measured.

##### Exact resume protocol

Reread the governing documents, then B0267/D0267 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'Kd3|Smrf|pdg_reprojection|pdg_sample|pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0266/D0266) — retained; superseded by B0267 above

**This subsection was the authoritative resume point before B0267.** Since the B0265 stop:
`filters.reprojection` is exact and parallel in streaming and standard modes
(B0266/D0266: `Streamable::processStreamBatch` hook, fixed-slot pool with
cloned GDAL transformations, scratch-then-commit with a serial fallback on
any failure): r8 **2.104750x / 2.093405x**, r1 **1.487025x / 1.458954x**;
aggregates **1.870298x / 1.869920x** geometric mean and **3.573691x /
3.535527x** total wall, all exact. Full physical CUDA aggregate 833/833 and
Host Debug 530/530 with the KD3 snapshot verifier armed; focused ASan/UBSan
and TSan pass. No build, test, benchmark, profiler, or delegated review is
left running.

##### Session summary (B0257 -> B0266)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x |
| B0265/D0265 | pooled morphology in the fork SMRF port | r2 1.62x |
| B0266/D0266 | slot-pooled exact reprojection (stream + standard) | r8 2.10x, r1 1.49x; suite 1.87x GM, 3.57x total |

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `879fdccc598872e393cd65c345b8291808441834cec260c6474da7cdc95f921b` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `3ce12e44fe801e73be911f0a63bec94e4084f1f9841dcb9b8cdded61419a9bb3` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `be1d78c4015a430ec8c1d47ea40c8940236bd9358ab6146459762ad16ac327e6` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| r8 warm / cold | `build/benchmarks/b0266-r8-colorize-parallel-reprojection-final2-{warm,cold}.json` | `13b2c32e...` / `2c4a9e87...` |
| r1 warm / cold | `build/benchmarks/b0266-r1-translate-parallel-reprojection-final2-{warm,cold}.json` | `e8d91a4a...` / `bf05f4c4...` |
| warm / cold aggregate | `build/benchmarks/b0266-suite2-{warm,cold}.json` | `7738248c...` / `71cd903d...` |
| reprojection filter | `filters/ReprojectionFilter.cpp` | `9eeef567cb2755244418663991b0adba125cf8c81476f128051d28d5e1052835` |
| slot pool | `pdal/private/HostSlotPool.hpp` | `e64f70122ed2b76f9f0a31dbe97a2b5c4da8e31670e7ae703d5f82617b7f91d1` |
| reprojection matrix | `tests/differential/reprojection_matrix.py` | `a1bee0b18d4b84c62c05391479a1b500d731e8c257aa1a7ecc68d1c531211398` |

##### Validation completed at this boundary

Physical CUDA 833/833 and Host Debug 530/530 with `PDAL_TEST_VERIFY_KD3_SNAPSHOT=1`
armed; focused leak-disabled ASan/UBSan (9) and TSan (9) lanes pass; the
14-case pinned-oracle reprojection matrix passes in every tree. No CUDA
translation unit changed this session.

##### Workspace checkpoint

(Historical: at the B0266 stop the checkpoint `101135c5c`, subject
`Checkpoint slot-pooled exact reprojection through B0266`, was `HEAD`.)

##### Next slice

See `IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm walls r4
1.22 s, r6 1.05 s, r2 0.95 s, r11 0.80 s, r8 0.79 s, r3 0.50 s. Candidates:
r4's host `filters.sample`; other per-point streaming filters that fit the
`processStreamBatch` slot-pool pattern; r6's fixed costs. `--fast` stays
behavior-neutral until measured.

##### Exact resume protocol

Reread the governing documents, then B0266/D0266 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'Kd3|Smrf|pdg_reprojection|pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0265/D0265) — retained; superseded by B0266 above

**This subsection was the authoritative resume point before B0266.** Since the B0264 stop:
the fork's SMRF port (r2's automatic route) pools its diamond morphology
through an internal pass pool (B0265/D0265): r2 **1.624190x / 1.604373x**;
aggregates **1.750757x / 1.737266x** geometric mean and **3.268150x /
3.192511x** total wall, all exact. Host Debug 529/529; CUDA-tree SMRF/hybrid/
r2 lanes; focused ASan/UBSan and TSan (including the pooled large-frame unit)
pass. No build, test, benchmark, profiler, or delegated review is left
running.

##### Session summary (B0257 -> B0265)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x |
| B0265/D0265 | pooled morphology in the fork SMRF port | r2 1.62x; suite 1.75x GM, 3.27x total |

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `879fdccc598872e393cd65c345b8291808441834cec260c6474da7cdc95f921b` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `ab1b8cca283b2bca9920486115f50779d078e315d18ad26b35cf0695551d6d6c` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `b5a1c8a17677bd6ef9a819d9d922bf43d58e563f74aa447146aa3c0d8ca50bc5` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| r2 warm / cold | `build/benchmarks/b0265-r2-port-morphology-pool-final-{warm,cold}.json` | `2d2e9fe8...` / `257e94f8...` |
| warm / cold aggregate | `build/benchmarks/b0265-suite-{warm,cold}.json` | `c81279bb...` / `44cbf561...` |
| SMRF port | `src/stages/Smrf.cpp` | `ace8cb30a2388a83fce230307a997ac3dcec06928645d7f205637bc3f7f0e1be` |
| SMRF port unit | `tests/unit/smrf_test.cpp` | `a02c5f4e1472cd324f43eb9fb60b19023d356cf2cddaa2393b2f2be2c3066b34` |

##### Validation completed at this boundary

Host Debug 529/529 (four optional skips); CUDA-tree SMRF units, hybrid SMRF
matrix, and automatic r2 lane pass (the full physical CUDA aggregate was
831/831 at B0263 and no CUDA translation unit changed since); focused
leak-disabled ASan/UBSan (9) and TSan (9, incl. the pooled large-frame unit)
pass.

##### Workspace checkpoint

(Historical: at the B0265 stop the checkpoint `942a4e285`, subject
`Checkpoint pooled fork SMRF port morphology through B0265`, was `HEAD`.)

##### Next slice

See `IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm walls r8
1.50 s, r4 1.22 s, r6 1.05 s, r2 0.95 s, r11 0.79 s, r3 0.49 s. Candidates:
r8 PROJ/GDAL host attribution; r4's host sample; r6's fixed costs. `--fast`
stays behavior-neutral until measured.

##### Exact resume protocol

Reread the governing documents, then B0265/D0265 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'Kd3|Smrf|pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0264/D0264) — retained; superseded by B0265 above

**This subsection was the authoritative resume point before B0265.** Since the B0263 stop:
the exact host SMRF's diamond morphology, void fills, and point passes run on
fixed-chunk workers (B0264/D0264; SMRF 0.34 s -> 0.17 s at 1M): r3
**1.352415x / 1.338430x** (from parity), r11 **8.867727x / 8.863616x**;
aggregates **1.731986x / 1.729459x** geometric mean and **3.185933x /
3.147773x** total wall, all exact. Host Debug 528/528; CUDA-tree SMRF/hybrid/
r11/r2 lanes 99/99; focused ASan/UBSan and TSan pass. No build, test,
benchmark, profiler, or delegated review is left running.

##### Session summary (B0257 -> B0264)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x |
| B0264/D0264 | exact host workers inside upstream SMRF | r3 1.35x, r11 8.87x; suite 1.73x GM, 3.19x total |

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `879fdccc598872e393cd65c345b8291808441834cec260c6474da7cdc95f921b` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `86baa8069b48a56283f22eae3029e0e51d52b1be3efc916a2a93379367633cc0` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `b5a1c8a17677bd6ef9a819d9d922bf43d58e563f74aa447146aa3c0d8ca50bc5` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| r3 warm / cold | `build/benchmarks/b0264-r3-dtm-smrf-workers-final-{warm,cold}.json` | `556ce508...` / `fba7e949...` |
| r11 warm / cold | `build/benchmarks/b0264-r11-classify-refine-smrf-workers-final-{warm,cold}.json` | `cb1e8dc3...` / `f3963df2...` |
| warm / cold aggregate | `build/benchmarks/b0264-suite-{warm,cold}.json` | `b242eb71...` / `8c4b2d63...` |
| SMRF source | `filters/SMRFilter.cpp` | `f059ff07afbad89a10b6aea3123fc818d05ff22d98b9cff7c85551cb01dc01ce` |
| morphology | `pdal/private/MathUtils.cpp` | `9d208126f43aaaaa0caa1fcc287ffd4d698a4a4ad1ccb7b0c1a611acb9eb9ba6` |
| SMRF matrix | `tests/differential/smrf_matrix.py` | `6153aca0ce8617400cd9272b44af08186a0bd59def61ecaf8d671d2ed23ffe2d` |

##### Validation completed at this boundary

Host Debug 528/528 (four optional skips); CUDA-tree SMRF/hybrid/r11/r2 lanes
99/99 (the full physical CUDA aggregate was 831/831 at B0263 and no CUDA
translation unit changed since); focused leak-disabled ASan/UBSan (15) and
TSan (8) pass; the SMRF matrix has 36 cases plus 3 pinned-oracle lattice
cases.

##### Workspace checkpoint

(Historical: at the B0264 stop the checkpoint `1ab13243b`, subject
`Checkpoint exact host workers inside SMRF through B0264`, was `HEAD`.)

##### Next slice

See `IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm walls r8
1.49 s, r4 1.22 s, r2 1.05 s (fork SMRF port still serial), r6 1.04 s, r11
0.79 s, r3 0.49 s. Candidates: the fork's SMRF port (r2) with the same
pooled passes; r8's PROJ/GDAL host work attribution; r4's host sample and
LAZ decode. `--fast` stays behavior-neutral until measured.

##### Exact resume protocol

Reread the governing documents, then B0264/D0264 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'Kd3|pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0263/D0263) — retained; superseded by B0264 above

**This subsection was the authoritative resume point before B0264.** Since the B0262 stop:
D0259 option 2 is done — the exact cached-coordinate KD3 backing is the
published `build3dIndex()` default with a view coordinate epoch that
refreshes reused snapshots (pinned stale-tree/live-coordinate semantics
preserved bit-for-bit). r11 **7.372457x / 7.286586x** warm/cold; aggregates
**1.672490x / 1.663704x** geometric mean and **3.073590x / 3.042106x** total
wall, all exact. Full Host Debug (528/528) and physical CUDA (831/831)
aggregates ran with the snapshot verifier armed. No build, test, benchmark,
profiler, or delegated review is left running.

##### Session summary (B0257 -> B0263)

| Record | Slice | Result |
| --- | --- | --- |
| B0257/D0256 | vote-tally hypothesis rejected by attribution; gprofng undersampling found | 0.1% of wall |
| B0258/D0257 | fixed-chunk host worker kNN passes (outlier, classifier) | r11 1.03x -> 4.31x |
| B0259/D0258 | default four-worker LAZ chunk compression | suite 1.40x -> 1.60x GM; r6 6.32x, r13 1.44x |
| B0260/D0259 | r6 attribution diagnostic; B0256 exactness defect found | alert |
| B0261/D0260 | pinned private-index outlier semantics restored (cached backing); pinned-oracle differential lane | r11 5.21x |
| D0261 | `--fast` record-exact contract, behavior-neutral | — |
| B0262/D0262 | terminal-sink private repair tree; r2 marker arming fixed; first full CUDA aggregate | r6 8.65x |
| B0263/D0263 | cached KD3 published default with mutation epoch | r11 7.37x; suite 1.67x GM, 3.07x total |

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `879fdccc598872e393cd65c345b8291808441834cec260c6474da7cdc95f921b` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `aa226784a98e2c20ebf8d2e58a3f220cab6a6ba2c6c9f14c0658b1a3a064ed28` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `65280bcaa52fa7ee299eee51e1f684e1e98ae59aa0ab8ed59d9f987218bcd4ef` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| r11 warm / cold | `build/benchmarks/b0263-r11-cached-kd3-default-final2-{warm,cold}.json` | `5fb142f3...` / `de489047...` |
| warm / cold aggregate | `build/benchmarks/b0263-suite2-{warm,cold}.json` | `c7ccae23...` / `1104931b...` |
| PointView | `pdal/PointView.hpp` / `pdal/PointView.cpp` | `99d2c26a...` / `e5afb0f6...` |
| PointRef | `pdal/PointRef.cpp` | `e1505601...` |
| KD3 refresh units | `tests/unit/kd3_concurrency_test.cpp` | `167a3fda...` |

##### Validation completed at this boundary

Host Debug 528/528 and physical CUDA 831/831 with `PDAL_TEST_VERIFY_KD3_SNAPSHOT=1`
in the environment; focused leak-disabled ASan/UBSan (35) and TSan (8) lanes
pass; r11 matrix 27 cases/28 executions incl. eight pinned-oracle cases. No
CUDA translation unit changed this session.

##### Workspace checkpoint

(Historical: at the B0263 stop the checkpoint `f41d5404b`, subject
`Checkpoint cached KD3 published default with mutation epoch through B0263`,
was `HEAD`.)

##### Next slice

See `IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm walls r8
1.50 s, r4 1.21 s, r2 1.05 s, r6 1.04 s, r11 0.97 s, r3 0.67 s; SMRF's exact
host implementation (~0.31 s at 1M) now appears in r2, r3, and r11 and is the
next shared host lever to attribute. `--fast` stays behavior-neutral until a
record-exact/stream-inexact optimization is measured.

##### Exact resume protocol

Reread the governing documents, then B0263/D0263 and this subsection. Verify
the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'Kd3|pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0262/D0262, D0261) — retained; superseded by B0263 above

**This subsection was the authoritative resume point before B0263.** Since the B0261 stop:
`--fast` shipped as a behavior-neutral record-exact contract (D0261,
`docs/fast-mode.md`); r6's exact tie repair keeps a private cached tree when
the planner proves the writer alone follows the region (B0262/D0262, r6
**8.648309x / 8.664582x**); the launcher's r2 marker arming was repaired
(latent CUDA test failure since B0243); the first full physical CUDA
aggregate of the session is 827/827 and Host Debug is 524/524. No build,
test, benchmark, profiler, or delegated review is left running.

##### Retained exact performance state

Fourteen-headline aggregates (final binaries, all exact): **1.633482x warm /
1.624540x cold** equal-workload geometric mean; **2.924049x / 2.894853x**
total wall (27.897299 / 9.540641 s warm; 28.153412 / 9.725333 s cold).
Nine-pair final gates: r6 8.648309x/8.664582x (B0262), r11
5.207843x/5.267489x (B0261), r13 1.437716x/1.434382x and r1
1.081819x/1.101200x (B0259).

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `c39183f569fc280f778b457c76b84d29c8626f57324b9ff220429620232dbfd5` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `a795f1a7db003421ea9349260a1d38a154a4e2da50a3e9a59239d0ac329d9bfc` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `d8a9ace152b8118e989fdc37c2ad85e815d9fc6fccf374a3adb76f263e6ef8cd` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| r6 warm / cold | `build/benchmarks/b0262-r6-terminal-sink-repair-final-{warm,cold}.json` | `058796c4...` / `0f141814...` |
| warm / cold aggregate | `build/benchmarks/b0262-suite-{warm,cold}.json` | `97d4ce71...` / `d836e4bc...` |
| neighborhood wrapper source | `src/pdal/PdgNeighborhood.cpp` | `e352e8a4cada733d13b3cead10a565c9852916a7c229eebce83bb0bdf837174f` |
| resident rewrite | `src/plan/ResidentPipeline.cpp` | `979b35a4de59d148484f014181af16601c6bd7daf68b4684a1abbca98379d29d` |
| launcher main | `src/cli/DispatchMain.cpp` | `5112de19f94885f76991f5c910fa3a15a97f0ac85bd218c9a5ae05c826130e0a` |

##### Validation completed at this boundary

Physical CUDA Release aggregate 827/827 (documented optional skips); Host
Debug 524/524 (four optional skips); focused leak-disabled ASan/UBSan (72
tests) passes; resident-rewrite marker units both ways. No CUDA translation
unit changed in any slice this session.

##### Workspace checkpoint

(Historical: at the B0262 stop the checkpoint `71b7fd4fc`, subject
`Checkpoint terminal-sink repair tree, r2 arming fix, through B0262`, was
`HEAD`.)

##### Open direction items and next slice

See `IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm candidate
walls r8 1.52 s, r11 1.37 s, r4 1.23 s, r2 1.06 s, r6 1.05 s. D0259 option 2
(cached KD3 as the published default with a mutation-signalled refresh) is
the largest general host lever and needs a mutation signal plus a
producer/mutator/consumer pinned-oracle matrix. `--fast` is behavior-neutral
until a record-exact/stream-inexact optimization is measured. Run the full
CUDA aggregate once per session.

##### Exact resume protocol

Reread the governing documents, then B0262/D0262, D0261, B0261/D0260, and
this subsection. Verify the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact|ResidentPipeline\.|pdg_dispatcher_process' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0261/D0260) — retained; superseded by B0262 above

**This subsection was the authoritative resume point before B0262.** B0260's alert (below,
retained) was resolved on its exactness-mandatory part: the statistical
outlier again has pinned private-index semantics, on the exact cached
backing, and a pinned-oracle differential lane now pins the defect. The
maintainer's D0259 option-2 choice (cached KD3 as the published default with
refresh-at-reuse) and the requested `--fast` contract are the open direction
items. No build, test, benchmark, profiler, or delegated review is left
running.

##### What changed in B0261

- `filters/OutlierFilter.cpp`: private fresh `KD3Index(view, true)`; no
  `invalidateProducts()`, no publication (`c60a2287...`).
- `CMakeLists.txt`: `PDG_PINNED_ORACLE_EXECUTABLE` (default
  `build/pdal-upstream-tests/bin/pdal` when present);
  `tests/differential/CMakeLists.txt` passes it to the r11 matrix.
- `scripts/pdg/differential.py --candidate-oracle` (`a2c2d104...`): the
  candidate's `PDG_ORACLE_PDAL` can differ from the comparison oracle and is
  recorded in the report.
- `tests/differential/r11_classification_matrix.py` (`ae52856b...`): 23
  cases/24 executions incl. four `pinned_oracle` cases on a 3,600-point
  lattice; the two nndistance cases fail on the B0259 library and pass now.
- Records B0261/D0260; plan, coverage, testing strategy, outlier stage page,
  README, and ledger updated.

##### Retained exact performance state

r11 **5.207843x warm / 5.267489x cold** (paired 5.232794x +/- 0.052059 and
5.220376x +/- 0.076892, 9/9), artifact `16eba4857103d13f...`; aggregates
**1.592353x / 1.595639x** warm/cold equal-workload geometric mean and
**2.794157x / 2.794157x** total wall, all fourteen exact. r6 6.32x/6.39x,
r13 1.44x/1.43x, r1 1.08x/1.10x nine-pair gates from B0259 stand.

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `d8a9ace152b8118e989fdc37c2ad85e815d9fc6fccf374a3adb76f263e6ef8cd` |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `2ee86751ecedc525a80b735f40276b0826c9cb577db4fd66ef29026f1e8532e1` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `43127cf0fdbc39df3a525c5ce9dc0b2fca3b0ee2a1eead76e4161dcd3d8df724` |
| sibling PDAL | `build/pdg-cuda-release/bin/pdal` | `7096f614005b26488871a9db2f60b52499e812e6bca2cccfa95c1c7b90d653b9` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| outlier source | `filters/OutlierFilter.cpp` | `c60a2287c63393ae9853005f999692463a09b5edd2af7ab060be814c81549d84` |
| r11 matrix | `tests/differential/r11_classification_matrix.py` | `ae52856bd97e273a91133a960a8446a9ef2d7a16cb45229e104a2f331201e2c8` |
| differential runner | `scripts/pdg/differential.py` | `a2c2d10476283b24113e96e74ed59f1f27838e653f9eb68995aeb708d8825c09` |
| warm / cold r11 | `build/benchmarks/b0261-r11-private-cached-outlier-final-{warm,cold}.json` | `c8dc8340...` / `7672f657...` |
| warm / cold aggregate | `build/benchmarks/b0261-suite-{warm,cold}.json` | `be6b585e...` / `b0b8aa59...` |

##### Validation completed at this boundary

Host Debug 523/523 (four optional skips); CUDA Release r11/outlier/SMRF/r14
matrices and hybrid/automatic outlier lanes; focused leak-disabled ASan/UBSan
and TSan lanes; no CUDA translation unit changed.

##### Workspace checkpoint

(Historical: at the B0261 stop the checkpoint `834f3abdb`, subject
`Checkpoint pinned outlier semantics and the pinned-oracle lane through
B0261`, was `HEAD`.)

##### Open direction items and next slice

1. `--fast` mode (maintainer request): byte-exact point records, relaxed
   diagnostics/metadata; the contract (which header/VLR fields, streams,
   status, point order, harness comparison) must be fixed before design.
2. D0259 option 2: cached KD3 as the published default with a
   refresh-at-reuse rule; needs a mutation signal (per-point
   `build3dIndex()` callers exist) and a producer/mutator/consumer
   pinned-oracle matrix. Payoff: r6 eigen repair -0.4 s and every host kNN
   consumer 3--4x cheaper.
3. Remaining walls: r8 1.50 s, r6 1.49 s, r4 1.26 s, r2 1.11 s.

##### Exact resume protocol

Reread the governing documents, then B0261/D0260, B0260/D0259, and this
subsection. Verify the anchors with `sha256sum`, then:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'pdg_(r11_classification|outlier|smrf|r14_conversion)_process_matrix_host_exact' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0260/D0259) — retained; the alert below was resolved by B0261

**This subsection was the authoritative resume point before B0261.** After the two accepted
slices below (B0258 host worker passes, B0259 default LAZ compression
workers), attribution work for the next slice uncovered a **P1 exactness
defect in B0256** that contradicts the plan's assumption that r11's shared
host index is semantics-preserving. Per the session directive, work stopped
here for direction; no product code changed after B0259 (only a stats-only
diagnostic, B0260). No build, test, benchmark, profiler, or delegated review
is left running.

##### The alert

`filters.outlier(statistical) -> filters.assign(X = X * 3) ->
filters.neighborclassifier(k=7)` produces different bytes on the fork
(`535921da...`) than on the pinned oracle (`f5c49683...`) for the 50K
fixture; controls without the mutator or with it before the outlier are
exact (B0260). Cause: pinned PDAL's outlier builds a private index and
publishes nothing, so pinned's classifier builds a fresh tree over the
mutated coordinates; B0256's outlier publishes its tree and the fork's
classifier reuses it stale. A symmetric divergence follows from B0256's entry
invalidation in `producer -> mutator -> outlier -> consumer`. The r11 reference
graph has no mutator and stays exact; B0258/B0259 results are unaffected. Two
exact fixes are on the table (D0259): (1) restore pinned outlier semantics
(private fresh index, no invalidation, no publication), or (2) make
cached-coordinate KD3 backing the `build3dIndex()` default with a
refresh-at-reuse rule (stale structure queried with live coordinates, exactly
like pinned nanoflann), which also yields the measured 4x cheaper builds and
3--4x cheaper worker kNN passes everywhere. Option 2 needs a
producer/mutator/consumer differential matrix before adoption.

##### Also found (B0260)

- r6's remaining wall is dominated by a **0.47 s uncached nanoflann build to
  repair 2,343 eigen tie rows** (of 1.24 s uncompressed); the writer is 0.21 s
  and the reader 0.15 s. Cached backing would cut about 0.4 s.
- Cached-coordinate KD3: build 0.077 s versus 0.307 s at 1M; 24-worker
  outlier pass 0.079 s versus 0.284 s; `k=7` pass 0.065 s versus 0.251 s;
  bit-identical. r11 could drop from about 1.47 s to about 0.85 s.
- New diagnostic: `pdg-engine resident PIPELINE --stats FILE` now emits the
  manager breakdown for ordinary upstream-writer neighborhood routes and an
  `eigen_family_breakdown_seconds` section (see `docs/diagnostics.md`).

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| r6 resident stats | `build/benchmarks/b0260-evidence/r6-uncompressed-resident-stats.json` | `f03e9cc2d94d1a6f11aabc0161a07e4e283e70cd77fabd7e0c1dfe35f449bf58` |
| defect pipelines | `build/benchmarks/b0260-evidence/outlier-assign-classifier-{pinned,fork}.json` | `77be3a64...` / `d9cdb491...` |
| engine (diagnostic build) | `build/pdg-cuda-release/bin/pdg-engine` | `43127cf0fdbc39df3a525c5ce9dc0b2fca3b0ee2a1eead76e4161dcd3d8df724` |
| selected Release library (B0259) | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `ab10f9346f807099011d13277b90482ee89c8ae2285f665b6e1a16f315590060` |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `2ee86751ecedc525a80b735f40276b0826c9cb577db4fd66ef29026f1e8532e1` |

Reproduce the defect (frozen clock, any tree):

```sh
LD_PRELOAD=build/pdg-cuda-release/lib/libpdg_frozen_time.so PDAL_TEST_FROZEN_EPOCH=1704067200 \
  build/pdal-upstream-tests/bin/pdal pipeline build/benchmarks/b0260-evidence/outlier-assign-classifier-pinned.json
LD_PRELOAD=build/pdg-cuda-release/lib/libpdg_frozen_time.so PDAL_TEST_FROZEN_EPOCH=1704067200 \
  build/pdg-cuda-release/bin/pdal pipeline build/benchmarks/b0260-evidence/outlier-assign-classifier-fork.json
```
(the pipelines write `out-pinned.las` / `out-fork.las` under the scratch
path recorded inside them; edit the `filename` before running elsewhere).

##### Workspace checkpoint

(Historical: at the B0260 stop the checkpoint `d7f37fc63`, subject
`Checkpoint B0260 attribution diagnostic and the B0256 exactness alert`, was
`HEAD`.) `git status --short`
must be empty at handoff. The B0259 checkpoint is `d01139f0a`; B0258 is
`a20cf3d29`. Do not reset, clean, stash, or discard paths.

##### Next slice

Await the maintainer's D0259 choice, then: fix the defect test-first (add
`outlier -> assign -> classifier` and `normal -> assign -> outlier ->
classifier` cases to the r11 matrix), then take the cached-KD3 slices for r6
(eigen repair) and r11, each with fresh warm/cold gates.

#### Final session stop — 2026-08-15 UTC (B0259/D0258) — retained; superseded by B0260 above

**This subsection was the authoritative resume point before B0260.** The session stops at
a complete, validated vertical-slice boundary after two accepted slices in a
row (B0258 host worker passes, then B0259 default LAZ compression workers).
Every gate named below is green; no build, test, benchmark, profiler, or
delegated review is left running. The B0258, B0257, and B0256 subsections
below are retained history; their binaries and "next" paragraphs are
superseded here.

##### What changed in B0259

- `io/LasWriter.cpp`: `readyCompression()` defaults to `min(4, hardware
  threads)` exact chunk-compression workers when the internal
  `PDG_LAZ_COMPRESSION_THREADS` channel is unset (D0250's serial default is
  superseded by D0258 with fourteen-workload evidence). Compressor, chunking,
  byte-identity proof, launcher routes, and the r14 two-worker selector are
  unchanged.
- `tests/differential/r11_classification_matrix.py`: adds
  `default-four-compression-workers` (nineteen cases/twenty executions).
- Records B0259 / D0258; `IMPLEMENTATION_PLAN.md`, `docs/stage-coverage.md`,
  `docs/testing-strategy.md`, `docs/diagnostics.md`,
  `bench/pipelines/reference/README.md`, and `OPPORTUNITY.md` updated.

##### Retained exact performance state

Fourteen-headline aggregates (final binaries, all exact): **1.597555x warm /
1.589241x cold** equal-workload geometric mean; **2.792319x / 2.772361x**
total-wall speedup (27.503016 / 9.849525 s warm; 27.613136 / 9.960152 s
cold). Nine-pair final gates: r6 6.316372x / 6.387622x, r13 1.437716x /
1.434382x, r1 1.081819x / 1.101200x (warm / cold, 9/9 wins each); r11 stays
at B0258's 4.309167x / 4.345954x nine-pair gates plus the writer gain
(4.773908x / 4.748259x in the three-pair suite).

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `ab10f9346f807099011d13277b90482ee89c8ae2285f665b6e1a16f315590060` |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `2ee86751ecedc525a80b735f40276b0826c9cb577db4fd66ef29026f1e8532e1` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `a21ab08ebac5adaabe0669ccc1a53138e3c3e091a902aafc346f2bfbe9ff4c97` |
| sibling PDAL | `build/pdg-cuda-release/bin/pdal` | `7096f614005b26488871a9db2f60b52499e812e6bca2cccfa95c1c7b90d653b9` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| writer source | `io/LasWriter.cpp` | `444de2f0d8fab825d3a9fad34761fbfd86a3ed0104b896a525813e454f91fd2f` |
| r11 matrix | `tests/differential/r11_classification_matrix.py` | `91cebb7698d9dd57f02624d92581f0a71635900239b810a2aaecd9a2fe6d5eb8` |
| warm aggregate | `build/benchmarks/b0259-laz-default-suite-warm.json` | `a47f541fb8a2e8eb445aed04a7b4aa99d59f7e085328aa5575303acccd5369bc` |
| cold aggregate | `build/benchmarks/b0259-laz-default-suite-cold.json` | `55bdbe98144ff83ea52959dfe6e32959f0c145050def16a1572946151e1e143a` |
| r6 warm / cold | `build/benchmarks/b0259-r6-features-laz-default-final-{warm,cold}.json` | `20d5a266...` / `04d91ae5...` |
| r13 warm / cold | `build/benchmarks/b0259-r13-merge-laz-default-final-{warm,cold}.json` | `31d28d61...` / `6dbe92c9...` |
| r1 warm / cold | `build/benchmarks/b0259-r1-translate-laz-default-final-{warm,cold}.json` | `eb4eadaf...` / `2dfb443c...` |
| engine probes serial/2/4 | `build/benchmarks/b0259-probe-engine-{base,laz2,laz4}-warm.json` | `f61d3740...` / `af0a5849...` / `3421a07c...` |

The B0258 anchors (r11 reports `a248c80f...`/`c209461448...`, worker policy
header, filter sources) remain valid; only the library hash moved.

##### Validation completed at this boundary

Host Debug full aggregate 523/523 (four documented optional skips) after the
writer change; CUDA Release r11/r14/byte-proof/dispatcher-process tests pass;
focused leak-disabled Host ASan/UBSan and TSan lanes over the LAZ-writing
matrices pass. B0258's own validation (523/523, ASan/UBSan, TSan including
the direct 50K thirteen-worker process) stands. No CUDA translation unit
changed in either slice.

##### Workspace checkpoint

(Historical: at the B0259 stop the checkpoint commit `d01139f0a`, subject
`Checkpoint default LAZ compression workers through B0259`, was `HEAD`.) `git status --short` must be
empty at handoff. Corpus inputs remain outside Git; the pinned-oracle build
under `build/pdal-upstream-tests` is an evidence artifact and must not be
rebuilt from this tree. Do not reset, clean, checkout, stash, switch
branches, regenerate goldens, or discard any path to reach an older handoff.

##### Next slice

See `IMPLEMENTATION_PLAN.md` "Next vertical slice": remaining warm candidate
walls are r8 1.48 s, r11 1.47 s, r6 1.42 s, r4 1.21 s, r2 1.02 s, r3 0.66 s.
Recorded candidates: (a) attribute r6's remaining ~1.4 s in-process (kernels
are ~24 ms of it); (b) the r11 shared-device SMRF -> outlier -> classifier
composition; (c) the exact fixed-chunk worker pattern on other serial host
neighborhood passes a reference workload still runs; (d) r8's PROJ/GDAL host
work. Cheap forced/controlled prototype first, every time.

##### Exact resume protocol

Reread the governing documents named at the top of this file, then B0259,
D0258, B0258, D0257, and this subsection. Obey the resource rules (8 GiB
`MemAvailable`, one heavy lane, two host/one CUDA compile jobs, physical GPU
before a GPU claim). Verify the anchors above with `sha256sum`, then run:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'pdg_(r11_classification|r14_conversion|outlier|smrf)_process_matrix_host_exact|LazChunkCompression' \
  --output-on-failure -j1
ctest --test-dir build/pdg-host-debug \
  -R 'pdg_(r11_classification|r14_conversion|outlier|smrf)_process_matrix_host_exact' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0258/D0257) — retained; superseded by B0259 above

**This subsection was the authoritative resume point before B0259.** The session stops at a
complete, validated vertical-slice boundary: the maintainer approved the
direction B0257 measured, the exact host worker passes are implemented,
qualified, recorded, and documented, and every gate named below is green. No
build, test, benchmark, profiler, or delegated review is left running. The
B0257 and B0256 subsections below are retained history; their binaries and
"next hypothesis" paragraphs are superseded here.

##### What changed

- `pdal/private/HostNeighborhoodWorkers.hpp` (new): fixed-chunk worker policy
  and runner (`min(ceil(rows/4096), hardware threads, PDG_NATIVE_WORKERS)`;
  `PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS` forces serial; test-only
  `PDAL_TEST_FORCE_/REQUIRE_HOST_NEIGHBORHOOD_WORKERS`).
- `filters/OutlierFilter.cpp`: statistical per-row kNN pass on workers;
  serial reduction/threshold/split/writes unchanged.
- `filters/NeighborClassifierFilter.cpp/.hpp`: per-source query/vote on
  workers with per-worker reassignment lists merged in worker order; serial
  application unchanged.
- `CMakeLists.txt` compiles the test hooks into both filters under
  `PDG_BUILD_TESTS`; `src/cli/Dispatch.cpp` lists the disable control;
  `scripts/pdg/benchmark_reference.py` scrubs the `PDAL_TEST_` prefix and
  `benchmark_translate.py` scrubs the three new names; both contract tests
  updated; `bench/r11_neighborhood_attribution.cpp` (B0257) and its CMake
  target are retained.
- `tests/differential/r11_classification_matrix.py`: eighteen cases/nineteen
  executions (see B0258).
- Records: B0257/B0258 in `BENCHMARKS.md`, D0256/D0257 in `DECISIONS.md`;
  `IMPLEMENTATION_PLAN.md`, `docs/stage-coverage.md`,
  `docs/testing-strategy.md`, `docs/diagnostics.md`,
  `docs/stages/filters.outlier.md`, `docs/stages/filters.neighborclassifier.md`,
  `bench/pipelines/reference/README.md`, and
  `bench/pipelines/reference/OPPORTUNITY.md` carry the new state.

##### Retained exact performance state

| Cache | Pinned median | PDG median | Median speedup | Paired result | Wins |
| --- | ---: | ---: | ---: | --- | ---: |
| Warm | 6.986365534 s | 1.621279660 s | **4.309167x** | **4.321977x +/- 0.034672** | 9/9 |
| Cold | 7.013202373 s | 1.613731392 s | **4.345954x** | **4.352975x +/- 0.038573** | 9/9 |

Same 6,792,339-byte artifact `16eba4857103d13f...`; serial control 1.035408x;
50K/250K controls 3.050913x/4.297817x; forced CUDA statistical-outlier probe
exact but slower at 4.069126x. Fourteen-headline aggregates: **1.401126x warm /
1.410744x cold** equal-workload geometric mean, **2.407033x / 2.408896x**
total-wall speedup, all exact.

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `45c72495a4eda42a17756d13e70068377a8ce6a2fe458adb9233584a589186b6` |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `2ee86751ecedc525a80b735f40276b0826c9cb577db4fd66ef29026f1e8532e1` |
| engine | `build/pdg-cuda-release/bin/pdg-engine` | `a21ab08ebac5adaabe0669ccc1a53138e3c3e091a902aafc346f2bfbe9ff4c97` |
| sibling PDAL | `build/pdg-cuda-release/bin/pdal` | `7096f614005b26488871a9db2f60b52499e812e6bca2cccfa95c1c7b90d653b9` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| warm r11 report | `build/benchmarks/b0258-r11-host-workers-final-warm.json` | `a248c80f31c497fcc68dc7b16bc2512655c08df3bd71896720087aa48730788d` |
| cold r11 report | `build/benchmarks/b0258-r11-host-workers-final-cold.json` | `c209461448a19c0f2bd92033db3657b94e2041fd0049eea644be7e792b7bd42c` |
| warm aggregate | `build/benchmarks/b0258-r11-host-workers-suite-warm.json` | `ccaaf92e5b1268362b23a3934a9c5c1cc5081dc9c9239bf0c3c3b62cb1a06131` |
| cold aggregate | `build/benchmarks/b0258-r11-host-workers-suite-cold.json` | `da358fad134a1411ea37aa49eb0bbb63de0e6eb2510c5c9978847ca124594a59` |
| outlier source | `filters/OutlierFilter.cpp` | `c95c9d3bf9413224858cb95fa068c8a15b78356f2238dd9b8e28c2ea7a9a41d2` |
| classifier source | `filters/NeighborClassifierFilter.cpp` | `47c861f3018380e75fbe3ee9a4e113e0f0b5f534120ddb1cb66256d2609b1ec3` |
| worker policy | `pdal/private/HostNeighborhoodWorkers.hpp` | `0b5aca9781863d9b65d23c87b7a115b570ce2b588995d3eb2fb4a776bf3c334b` |
| r11 matrix | `tests/differential/r11_classification_matrix.py` | `9531106d1cb9a0c72ef38d0508a338142b6f569977912800b7ab8b7d31542c20` |

##### Validation completed at this boundary

- Full serialized Host Debug aggregate 523/523 (four documented optional
  skips); r11/outlier/SMRF matrices, dispatcher, planner/resident, KD3
  concurrency, and harness contract tests pass on CUDA Release and
  leak-disabled Host ASan/UBSan; the r11 matrix, outlier matrix, and KD3
  concurrency units pass TSan; a direct 50K thirteen-worker TSan process is
  clean. No CUDA translation unit changed.

##### Workspace checkpoint

(Historical: at the B0258 stop the checkpoint commit `a20cf3d29`, subject
`Checkpoint exact host worker passes through B0258`, was `HEAD`.) `git status --short` must be empty at
handoff. Corpus inputs remain outside Git; the pinned-oracle build under
`build/pdal-upstream-tests` is an evidence artifact and must not be rebuilt
from this tree. Do not reset, clean, checkout, stash, switch branches,
regenerate goldens, or discard any path to reach an older handoff.

##### Next slice

See `IMPLEMENTATION_PLAN.md` "Next vertical slice": reread the ledger; the
recorded candidates are the r11 shared-device SMRF -> outlier -> classifier
composition (cheap forced prototype first), the same exact worker pattern on
other serial host neighborhood passes a reference workload still executes,
and r4's host-worker path versus its calibrated CUDA route.

##### Exact resume protocol

Reread the governing documents named at the top of this file, then B0258,
D0257, and this subsection. Obey the resource rules (8 GiB `MemAvailable`,
one heavy lane, two host/one CUDA compile jobs, physical GPU before a GPU
claim). Verify the anchors above with `sha256sum`, then run the focused
gates:

```sh
free -h; nvidia-smi; git status --short
ctest --test-dir build/pdg-cuda-release \
  -R 'pdg_(r11_classification|outlier|smrf)_process_matrix_host_exact' \
  --output-on-failure -j1
ctest --test-dir build/pdg-host-debug \
  -R 'pdg_(r11_classification|outlier|smrf)_process_matrix_host_exact' \
  --output-on-failure -j1
```

#### Final session stop — 2026-08-15 UTC (B0257/D0256) — retained; superseded by B0258 above

**This subsection was the authoritative resume point before B0258.** The session stops at a
deliberate direction boundary, not mid-slice: the plan's named next slice was
measured, rejected, and recorded, and the measured replacement slice is
larger in kind than the one the plan named, so it awaits the maintainer's
decision. No product source changed. No build, test, benchmark, profiler, or
delegated review is left running. The B0256/D0255 subsection below remains
accurate for the selected binaries and current performance state; only its
"sole next optimization hypothesis" paragraph is superseded here.

##### What happened

- The vote-tally hypothesis was measured before any prototype with the new
  checked-in harness `bench/r11_neighborhood_attribution.cpp` (target
  `pdg_r11_neighborhood_attribution`, built into the CUDA Release tree without
  touching the selected library). The complete upstream tally is about
  0.043 s of the 6.8-second process; the allocation-free ascending-key variant
  saves 4--6 ms, about 0.1% of wall, against a +/- 0.9% paired gate error.
  Rejected under D0204's cheap-prototype rule (B0257/D0256). The generic
  ordered-map tally remains the only implementation.
- The same harness attributes about 3.37 s to the statistical outlier's
  per-point kNN pass and about 2.18 s to the classifier's per-point kNN pass
  (together about 5.55 s of 6.8 s), and both are bit-identical under
  fixed-chunk multi-worker execution over the read-only PointView-owned
  nanoflann index: about 0.36 s and 0.24 s at eleven workers. That is harness
  attribution, not a performance claim.
- Surprise recorded in `docs/diagnostics.md`: `gprofng` on this workstation
  captures only about 7--10% of CPU time (0.290 s for a 3.0-second spin loop;
  0.499 s for a 6.9-second `pdal pipeline`). The B0255/B0256 profile
  attributions were undersampled by that factor, which is why the
  profile-directed tally hypothesis overstated a 0.6%-of-stage cost.

##### Evidence anchors (rechecked at this handoff)

| Item | Path | SHA-256 |
| --- | --- | --- |
| attribution harness source | `bench/r11_neighborhood_attribution.cpp` | `afcbbff519766ecca34d9387377c502fd81e0acb94d134013a7d5b4a27297454` |
| attribution report | `build/benchmarks/b0257-r11-neighborhood-attribution.json` | `d07a8910e2167b5d9158c44998f555b915507d1657b5bf73a46eb5ec3d1a70b4` |
| selected Release library (unchanged) | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `3cd84804a8cf8f3581b7e007454530e5718fcd34dee048e9be5611e76665db10` |
| public launcher (unchanged) | `build/pdg-cuda-release/bin/pdg` | `2fa75cd7efb2d30651c2222b52b1a204f7a5cd01b10a12a77e999fbfc38257a7` |

The append-only records are B0257 in `BENCHMARKS.md` and D0256 in
`DECISIONS.md`. `IMPLEMENTATION_PLAN.md`, `docs/diagnostics.md`, and
`bench/pipelines/reference/OPPORTUNITY.md` carry the rejection and the new
attribution. Every B0256 evidence hash in the subsection below still verifies.

##### Pending direction decision

The measured next r11 slice is exact host parallelism of the statistical
outlier and neighbor classifier per-point kNN passes (D0237's fixed-chunk
worker pattern; serial online-moment reduction, vote, final assignment,
options, diagnostics, and output order unchanged; bit-identity at one, two,
and many workers plus TSan required; retain only if both exact warm and cold
complete-process gates resolve). It threads two upstream stage hot loops in
the fork library rather than the bounded tally the plan named, so it was not
started without direction. If approved, begin test-first: fresh pinned-PDAL
warm/cold baseline, then the outlier pass alone, then the classifier pass, each
with its own gate. Do not widen the standalone CUDA neighbor-classifier
envelope or add an external model.

##### Workspace state

(Historical: at the B0257 stop the B0256 checkpoint commit `a8dc7ac4f` was
still `HEAD` and the B0257/D0256 work was an uncommitted working-tree change
awaiting the maintainer's checkpoint decision, comprising:
`BENCHMARKS.md`, `DECISIONS.md`, `IMPLEMENTATION_PLAN.md`, `HANDOFF.md`,
`docs/diagnostics.md`, `bench/pipelines/reference/OPPORTUNITY.md`,
`bench/CMakeLists.txt`, and the new `bench/r11_neighborhood_attribution.cpp`,
plus the untracked report under `build/benchmarks/`. No product source, test,
golden, corpus, or evidence artifact from B0256 was modified.) See the B0258
subsection for the current workspace state.

##### Resume protocol (historical)

Reread the governing documents named at the top of this file, then B0257,
D0256, and this subsection, then the B0256/D0255 subsection for the selected
binaries. Run the resource and hash checks in the B0256 subsection's "Exact
resume protocol" (all still apply), plus:

```sh
sha256sum bench/r11_neighborhood_attribution.cpp \
  build/benchmarks/b0257-r11-neighborhood-attribution.json
```

To reproduce the attribution:

```sh
cmake --build build/pdg-cuda-release --target pdg_r11_neighborhood_attribution -j2
./build/pdg-cuda-release/bin/pdg_r11_neighborhood_attribution \
  --input build/bench-data/reference/ref-1m.laz --output /tmp/r11-attrib.json
```

#### Final session stop — 2026-08-15 UTC (B0256/D0255) — retained; next-hypothesis paragraph superseded above

**This subsection was the authoritative resume point before B0257; its binaries, numbers, and hashes remain current.** The session stops at a
complete, reviewed vertical-slice boundary: the B0256 correction is present,
its exact warm/cold qualification and all-fourteen-workload aggregates are
recorded, the relevant sanitizer and full Host Debug gates are green, and the
independent re-review found no remaining P1/P2 in this slice. No build, test,
benchmark, profiler, or delegated review is intentionally left running. The
older `Active next task` and `Resume recipe` sections much later in this file
are historical only and must not be followed.

##### Why this is the stopping point

B0255 found a real reusable host product but its first lifetime argument was
too broad: an existing PointView KD3 could be stale after
`filters.normal -> filters.assign(X/Y/Z)`. B0256 closes that exactness defect
rather than carrying it into another optimization. Statistical outlier now
invalidates incoming products before calling `PointView::build3dIndex()`, so
it retains pinned PDAL's historical always-fresh tree semantics and only then
publishes the newly built exact nanoflann KD3 for the adjacent neighbor
classifier. The following remain unchanged:

- statistical kNN implementation and traversal order, threshold arithmetic,
  Classification writes, diagnostics, output point order, and options;
- radius outlier's private local index;
- dispatcher grammar, automatic placement, native/GPU-native coverage, and
  the standalone CUDA neighbor-classifier envelope; and
- r11's PDAL-native classification/refinement chain, with no external model
  dependency.

The new deterministic r11 matrix has eight cases/nine executions. Its added
lifetime regression deliberately constructs a KD3 with `filters.normal`,
mutates X through generic `filters.assign`, then runs statistical outlier and
requires the pinned oracle's complete bytes, metadata, point order, stdout,
stderr, and status. This is the smallest defensible boundary after which to
pause.

##### Retained exact performance state

The authoritative final-binary r11 results are the corrected B0256 rows, not
B0255's superseded rows:

| Cache | Pinned median | PDG median | Median speedup | Paired result | Wins | Pinned / PDG peak RSS |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Warm | 7.008508261 s | 6.805089262 s | **1.029892x** | **1.035463x +/- 0.009249** | 9/9 | 197,570,560 / 193,183,744 B |
| Cold | 7.031751580 s | 6.778823839 s | **1.037311x** | **1.039144x +/- 0.008688** | 9/9 | 195,100,672 / 192,032,768 B |

Both lanes are resolvable. They publish the same 6,792,339-byte LAZ at
`16eba4857103d13f209ce674c8aea6ae400831c7fb93db260499e6d70a2f13f7`
with exact LAS metadata, point records/order, stdout, stderr, and exit status.

The complete exact reference-suite state is:

| Cache | Equal-workload geometric mean | Pinned summed medians | PDG summed medians | Total-wall speedup |
| --- | ---: | ---: | ---: | ---: |
| Warm | **1.267705x** | 27.522714 s | 16.614599 s | **1.656538x** |
| Cold | **1.278088x** | 27.701157 s | 16.630051 s | **1.665729x** |

All fourteen headline rows are present, exact, and equally weighted. Warm
aligned-suite ratios are 1.761856x, 1.668072x, and 1.651722x; cold ratios are
1.666568x, 1.664624x, and 1.667721x. The aggregate's shorter three-pair r11
warm interval is noisy; it remains included at full weight and is not the
selection proof. Use the independent nine-pair lanes above.

The corrected final public profile is
`build/benchmarks/r11-index-reuse-corrected-final-profile.er`. It records
6.841 seconds wall and 1.061 seconds sampled CPU. Inclusive samples attribute
0.550 seconds to nanoflann KD3 search, 0.370 to statistical outlier, 0.330 to
the kNN implementation, 0.320 to Point14 decode, 0.230 to neighbor
classification, 0.040 to SMRF, and 0.030 to the one fresh PointView KD3 build.
Its I/O collector emitted the known `COL_ERROR_IOINIT`/`llseek` warnings, so
do not use its I/O latency fields. The accepted sorted profile-manifest hash
recorded by B0256 is
`3ea0b3b34d09b124d03da70cb85b8520e8b5893d2666dad85c846f4e42e94254`.

##### Evidence and content anchors

These hashes were rechecked immediately before this handoff update:

| Item | Path | SHA-256 |
| --- | --- | --- |
| corrected source | `filters/OutlierFilter.cpp` | `c63d321da3b4016e47be13b16e718e2b71b87860107826623081203853c5989c` |
| expanded matrix | `tests/differential/r11_classification_matrix.py` | `120a15a2ec032a0a8c79a3c4346f04f81aeed99fb747053437ca1219c4bf8045` |
| selected Release library | `build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0` | `3cd84804a8cf8f3581b7e007454530e5718fcd34dee048e9be5611e76665db10` |
| public launcher | `build/pdg-cuda-release/bin/pdg` | `2fa75cd7efb2d30651c2222b52b1a204f7a5cd01b10a12a77e999fbfc38257a7` |
| sibling PDAL | `build/pdg-cuda-release/bin/pdal` | `7096f614005b26488871a9db2f60b52499e812e6bca2cccfa95c1c7b90d653b9` |
| pinned oracle | `build/pdal-upstream-tests/bin/pdal` | `ea66f6a9ebb833fef1ba6d3def99ba0b36385632deeefebe8e6a9c7e034af7a2` |
| warm r11 report | `build/benchmarks/r11-index-reuse-corrected-final-warm.json` | `8eabc4306706c7e445470c0cb2fe07bc72ece772ed7bb37eb57476898870e47b` |
| cold r11 report | `build/benchmarks/r11-index-reuse-corrected-final-cold.json` | `eb4df3499c4960d9502c12e3dd1551975d96d15f14691481ff18e5a75549bc8d` |
| warm aggregate | `build/benchmarks/b0256-r11-corrected-suite-warm.json` | `a6df24912563e2720b61511f74f85b74afe2ff210a054a9e23a27c34f90bbd68` |
| cold aggregate | `build/benchmarks/b0256-r11-corrected-suite-cold.json` | `62fb48d21f24da656d52d982bc88d1bf6c32b7bfbc265c0fb7d6d4e91bab1da7` |

The append-only acceptance records are B0256 in `BENCHMARKS.md` and D0255 in
`DECISIONS.md`. `IMPLEMENTATION_PLAN.md`, `docs/stage-coverage.md`,
`docs/testing-strategy.md`, `docs/stages/filters.outlier.md`,
`bench/pipelines/reference/README.md`, and
`bench/pipelines/reference/OPPORTUNITY.md` all carry the corrected scope and
numbers. B0255/D0254 remain intentionally visible as superseded history; do
not delete, rewrite, or cite their library-bound performance as current.

##### Validation completed at this boundary

- The eight-case r11 matrix, standalone outlier and SMRF matrices, and the
  relevant planner/index-lifecycle tests pass Host Debug, CUDA Release, and
  leak-disabled Host ASan/UBSan.
- The final serialized Host Debug aggregate enumerates 523 registrations.
  Every enabled test passes; the same four documented optional local-fixture
  lanes are disabled/skipped.
- The numerical evidence-verification pass and final `git diff --check` are
  clean.
- An independent Terra re-review confirmed the stale-cache P1 is closed and
  found no remaining P1/P2 in the corrected r11 implementation, matrix,
  records, or related docs.
- No CUDA translation unit changed in B0256, so there is deliberately no new
  Compute Sanitizer claim.

##### Workspace checkpoint

The complete accepted workspace is committed locally on branch
`p3-skewness-resident`. The checkpoint commit is the current `HEAD` with
subject `Checkpoint expanded exact reference suite through B0256`; run
`git rev-parse HEAD` to obtain its immutable ID. `git status --short` must be
empty at handoff. The checkpoint preserves together all 74 formerly modified
tracked paths and 49 formerly untracked source, test, pipeline, and
documentation paths; none were discarded or hidden in a stash.

Do not reset, clean, checkout, stash, switch branches, regenerate goldens, or
discard any path in an attempt to reach an older handoff. In particular, do
not follow the historical `git switch p1.5/v6-dirty-index-rebuild` command
later in this file. Corpus inputs remain outside Git and must not be modified.
The pinned-oracle build under `build/pdal-upstream-tests` is an evidence
artifact: do not rebuild it from this changed source tree.

The primary current-slice files are:

- `filters/OutlierFilter.cpp` and the pre-existing PointView product lifetime
  machinery in `pdal/PointView.*`;
- `tests/differential/r11_classification_matrix.py` and its registration in
  `tests/differential/CMakeLists.txt`;
- the B0256/D0255 records and the six current status/coverage/reference docs
  named above; and
- the corrected reports and profile under `build/benchmarks/` listed in the
  evidence table.

Use `git status --short` before touching anything; it should be empty before
new work begins. The broader reference-suite, dispatcher, resident execution,
LAZ, and stage work in the checkpoint predates this stop and is not permission
to rewrite or prune it.

##### Exact resume protocol

On the next session, first reread the governing documents named at the top of
this file, then B0256, D0255, and this subsection. Before a heavy build,
sanitizer, corpus lane, or benchmark, obey the resource rules:

```sh
free -h
nvidia-smi
git status --short
sha256sum filters/OutlierFilter.cpp \
  tests/differential/r11_classification_matrix.py \
  build/pdg-cuda-release/lib/libpdalcpp.so.20.0.0 \
  build/pdg-cuda-release/bin/pdg \
  build/pdg-cuda-release/bin/pdal \
  build/pdal-upstream-tests/bin/pdal
sha256sum \
  build/benchmarks/r11-index-reuse-corrected-final-warm.json \
  build/benchmarks/r11-index-reuse-corrected-final-cold.json \
  build/benchmarks/b0256-r11-corrected-suite-warm.json \
  build/benchmarks/b0256-r11-corrected-suite-cold.json
```

Require at least 8 GiB `MemAvailable`, serialize every heavy lane, use at most
two host compile jobs and one CUDA compile job, and confirm a physical GPU
before a GPU claim. Start with focused tests, not the full corpus. Useful
process gates are:

```sh
ctest --test-dir build/pdg-host-debug \
  -R 'pdg_(r11_classification|outlier|smrf)_process_matrix_host_exact' \
  --output-on-failure -j1
ctest --test-dir build/pdg-cuda-release \
  -R 'pdg_(r11_classification|outlier|smrf)_process_matrix_host_exact' \
  --output-on-failure -j1
```

(Superseded by B0257/D0256 above: measured at about 0.1% of wall and
rejected.) The sole next optimization hypothesis was the smallest
profile-directed r11 slice: test an allocation-free ascending-key vote tally only for the literal
`Dimension::Id::Classification`, `k=7` neighbor-classifier semantics. Begin
test-first with exact tie, mixed-class, domain, malformed, and repeat-order
cases, and preserve the generic ordered-map implementation for every unproved
dimension, `k`, domain, candidate-file, and graph shape. Establish a fresh
pinned-PDAL same-machine warm/cold complete-process baseline before changing
the hot path, then re-profile the public process. Retain the change only if
both warm and cold paired gates resolve faster while artifact bytes, complete
LAS metadata, point order, stdout, stderr, and status stay exact. If either
lane is slower, unresolved, or semantically different, remove the prototype
and append the rejection. Do not widen a dispatcher route or CUDA envelope
and do not add an external model.

**Verified end state, 2026-08-12:** tree clean, **775/775 CUDA and 486/486
sanitizer tests pass**, placement audit at 157 cases over 36 models.

**Post-session audit correction, 2026-08-14 (D0221):** the verified-end-state
sentence above describes the unattended session's own report, not an
independent acceptance of all 49 commits. The review retained the narrow
B0183--B0220 planner/reader/dispatch/refusal changes and B0216's exact host
SMRF repair, but made two containment changes:

- B0222's default filter-only substitution gate was reverted exactly. Its
  predicate-only measurements did not cover the transformation and ordinal
  stages its condition also matched, and it had no selection/process tests.
- The SMRF CUDA prototype is now unqualified and fails closed for every input.
  Its integer/index cutoff-tie rule is known to differ from the oracle's
  KD2Index traversal; the prior compact CUDA matrix did not exercise that
  counterexample. The exact KD2Index host wrapper remains, with a committed
  `cell=1/2/4/8` process ladder.

The reference benchmark's exactness verdict also now includes stdout and
stderr hashes rather than artifact bytes alone. B0191's record-size derivation
is retained, while its comments were corrected to say that general
`extra_dims=all` writer admission was reverted. Read the validation record at
the end of this audit correction before relying on the old aggregate counts.

**Independent validation, 2026-08-14:**

- Host Debug rebuilt cleanly. The maintained `pdg` preset is green across all
  401 registered preset tests (four optional local-corpus/LAZ tests skipped).
  The expanded SMRF host differential passes separately.
- The unfiltered 486-test Host Debug aggregate is **485/486**. The one failure,
  `pdg_hag_delaunay_process_matrix_host_exact`, is independently reproducible:
  its deliberate upstream no-ground assertion differs only in libc's process
  prefix (`pdg:` versus `pdal:`). No B0183--B0222 commit changes HAG-Delaunay
  product or matrix code, so this is a pre-existing compatibility defect, but
  it means the old blanket green-count wording must not be reused.
- Host ASan/UBSan rebuilt with the maintained preset and passes all 401 preset
  tests plus the expanded SMRF host differential. Leak detection is disabled
  exactly as the preset and testing strategy require.
- CUDA Release rebuilt for SM 89 and the physical RTX 4090 aggregate is green
  across all 774 registered tests after removal of the invalid SMRF
  CUDA-exact matrix. Sixteen optional/calibration-identity tests skip rather
  than widening their envelope on this dirty review build. The focused SMRF
  lifecycle test positively proves required device execution fails closed.
- A live one-run reference-harness smoke test reports artifact, stdout, and
  stderr exactness independently and combines all three in `exact_outputs`.

No Compute Sanitizer claim is added by this review: no CUDA translation unit
was changed, and the only device behavioral change is to prevent entry into a
known-inexact compiled prototype.

**Continuation, 2026-08-14 (B0223/D0222):** the previously blocked
uncompressed `writers.las(extra_dims=all)` sink is now admitted under its
exact narrow envelope. The missing contract was not another column: a
reorder-only terminal region publishes its declared
output-position-to-input-position permutation while the mapped source supplies
the complete records. This fixes the zero-column spill that made the B0191
prototype fail after `filters.sort`, without inventing a private layout or
weakening the strict direct-sort/skewness proofs.

- The public 1M `normal(knn=8) -> covariancefeatures(knn=8,
  Dimensionality) -> writers.las(extra_dims=all)` process is byte-, stdout-,
  stderr-, order-, and status-exact and measures **6.267794x median** pinned
  PDAL (8.431024 s versus 1.345134 s), with all five paired runs faster and a
  6.281435x +/- 0.040884 paired interval.
- The emitted 100,001,965-byte LAS has a 100-byte point record on both sides;
  `--stats` reports that same derived width and the
  `planner_resident_shared_index` executor. The physical RTX 4090 selection
  matrix now has seventeen exact cases, including a carried-source-Extra-Bytes
  case that remains exact and host-selected.
- The maintained Host Debug preset passes 403/403 registered tests (four
  optional fixture tests skip). The physical CUDA Release aggregate passes
  776/776 registered tests (sixteen optional/calibration-identity tests skip),
  including direct sort/skewness and the expanded selection matrix. Focused
  Host ASan/UBSan passes with the maintained `detect_leaks=0` setting. No CUDA
  translation unit changed, so this continuation adds no new Compute
  Sanitizer claim.
- Functional sink admission covers only the literal `extra_dims=all` on an
  otherwise option-free uncompressed LAS writer. Automatic device selection
  is narrower still: the measured compressed format-7 source at exactly 1M
  points and 36 -> 100-byte records. Other cardinalities and carried source
  layouts remain host-selected. Compressed LAZ output, named extra dimensions,
  and any additional writer option stay non-native, so neither r2 nor r6 is
  unlocked.
- The exact-shape Nsight Compute report records 20 launches totaling
  24.495008 ms, only 1.821009% of complete candidate wall. `knnGather` consumes
  21.459456 ms/87.607467% of kernel time and is L2/memory limited at 76.72%
  memory throughput, leaving a 23.28-point roofline gap. This closes further
  kernel tuning for this endpoint; the remaining wall-time limiter is outside
  the measured kernels and is not yet attributed.

**Continuation, 2026-08-14 (B0224/D0223):** the actual r6 reference pipeline
is now automatically accelerated through its exact compressed
`writers.las(compression=true,extra_dims=all)` sink.

- Five final alternating public-command pairs are artifact/stdout/stderr/
  status/order exact at **4.529005x median** pinned PDAL (9.165000 s versus
  2.023623 s), with a 4.527212x +/- 0.037728 paired interval and 5/5 wins. Both
  sides write the same 61,789,134-byte LAZ at SHA-256 `af3eee9a...`.
- Functional LAZ admission covers literal `all` with compression omitted or
  true as a boolean/string and no additional writer option. Automatic device
  placement covers only the measured 1M, compressed format-7, 36 -> 100-byte
  normal/covariance row. A central compressed-layout guard prevents older
  eigen-family, rank/optimal, and per-stage models from inheriting the sink.
  r2 remains delegated because its extra dimension is named.
- The physical resident selection matrix passes twenty cases and proves all
  three admitted compression spellings through the public dispatcher, exact
  output, executable rewrite, accepted preflight, and the resident executor.
  Wider carried Extra Bytes decline in that matrix; runtime-placement units
  separately refuse unrelated eigen/rank compositions. The injected
  post-placement refusal exits 124 without output.
- Maintained Host Debug and Host ASan/UBSan presets pass 403/403. The
  unfiltered Host suite remains 487/488 only because of the documented
  unrelated HAG-Delaunay process-prefix defect. The physical CUDA Release
  aggregate exits cleanly across 776 registrations; no CUDA translation unit
  changed and no new Compute Sanitizer claim is made.
- The exact frozen-time Nsight Compute profile has 20 launches totaling
  24.428000 ms/1.207142% of candidate wall. `knnGather` is
  21.379936 ms/87.522253% of kernel time and remains L2/memory limited at
  76.956112% L2 versus 58.255184% SM throughput. The remaining ~1.999195 s is
  outside profiled kernels, so further kernel work does not reopen.

**Continuation, 2026-08-14 (B0225/D0224):** do not implement r2's named
`HeightAboveGround=float32` sink in isolation. The final current-binary r2
reference is exact but resolvably slower at 0.988063x median (1.558436 s pinned
PDAL versus 1.577263 s `pdg`; paired 0.986717x +/- 0.004381, 0/5 wins).
Resident stats show two additional blockers behind the writer: SMRF and HAG-NN
are separate device-preferred regions and neither has a placement model.
B0217's exact SMRF+HAG substitution loses 350.7 ms +/- 38.5, B0218's host
HAG-NN loses about 36%, its fast CUDA route is unreachable without a new
model, and device SMRF is unqualified under D0221. Speed-first policy therefore
keeps upstream delegation. Reopen r2 only with a complete measured profitable
stage route; admit and calibrate the named sink in that same slice.

**Continuation, 2026-08-14 (B0226/D0225):** r1 no longer loads the engine for
the header-calibrated reference route. Reprojection is host-only, so the old
path could only substitute crop and then exec pinned PDAL. The pre-change
public result was resolvably slower at 0.938267x +/- 0.007951; the accepted
direct route is at parity (1.002029x +/- 0.009677), while a paired control
against `pdg-engine` is 1.075614x +/- 0.011503 with 9/9 wins. Artifact/stdout/
stderr/status/order are exact. The gate is literal: exact root, stages/options,
materialized bounds, SRS pair, LAZ spelling, and the 1M compressed
format-7/36-byte reference file-size/header/layout/extent facts. It does not
claim content identity. Drift, pipeline CLI modifiers, and PDG behavior
overrides stay in-engine. This is a host dispatch win, not native reprojection
or GPU coverage.

**Continuation, 2026-08-14 (B0227/D0226):** r4 now automatically selects the
already-exact hybrid statistical-outlier CUDA executor for only the measured
1M reference shape and RTX 4090/SM89/CUDA 13.3/driver 610.43.03 profile. The
complete public route is exact and moves from parity to **3.686747x median**
(3.715180x +/- 0.055459 paired, 9/9 faster), including fingerprint cost. The
matcher requires the literal root/stages/options, compressed
format-7/36-byte selected-header/file/extent facts, the measured full-file
FNV-1a-64 fingerprint, and no CLI modifier; neighboring facts and every
fingerprint mismatch, including the tested payload mutation, fail closed. The
fingerprint is a calibration key, not a cryptographic identity claim. Range
uses the existing point program, while sample and LAZ I/O remain upstream. The
public proof environment
checks both the final selected rewrite and actual CUDA use. Do not describe
this as native sample or generalized outlier coverage; expand it only with a
new same-machine exact complete-process calibration.

**Continuation, 2026-08-14 (B0228/D0227):** r5 no longer loads an engine which
cannot consume its COPC reader. The pre-change current binary was exact but
resolvably slower at 0.930792x +/- 0.011642 paired; the literal direct-oracle
route is now at parity, 1.001303x +/- 0.003425. A paired control against
`pdg-engine` is resolvably 1.047266x +/- 0.016446 with 9/9 wins. The matcher
requires the exact single-key root, three-stage order/options, lowercase COPC
and LAS endpoints, materialized bounds, floating resolution 1.0, integer one
request, and option-free stats; every pipeline CLI modifier and PDG behavior
override stays in-engine. No input identity gate is necessary because both
paths run the unchanged pinned-oracle pipeline and only the fixed engine load
is removed. This is host dispatch, not native COPC, stats, or LAS coverage.

**Continuation, 2026-08-14 (B0229/D0228):** r3 no longer loads an engine whose
admission work is always discarded. The pre-change current binary was exact
but resolvably slower at 0.966556x +/- 0.006223 paired; the literal
direct-oracle route is now at parity, 0.996610x +/- 0.009240. A paired control
against `pdg-engine` is resolvably 1.027780x +/- 0.005196 with 9/9 wins. The
matcher requires the exact single-key root, four-stage order/options,
lowercase non-COPC LAZ and `.tif` endpoints, option-free SMRF, literal ground
range, floating resolution 1.0, and GDAL IDW; every pipeline CLI modifier and
PDG behavior override stays in-engine, and case-insensitive COPC suffixes fail
closed. No input identity gate is necessary because both paths run the
unchanged pinned-oracle pipeline and only fixed
process/control-plane work is removed. This is host dispatch, not native SMRF,
range, or GDAL coverage.

**Continuation, 2026-08-14 (B0230/D0229):** r2 no longer loads an engine which
refuses at its named writer before point work. The fresh pre-change public
binary was exact but resolvably slower at 0.973432x +/- 0.011569 paired; the
literal direct-oracle route is now at parity, 1.008753x +/- 0.010559. A paired
control against `pdg-engine` is resolvably 1.014563x +/- 0.005922 with 9/9
wins. The matcher requires the exact single-key root, four-stage order/options,
lowercase non-COPC LAZ endpoints, option-free SMRF/HAG-NN, string compression
true, and the literal `HeightAboveGround=float32` layout; every pipeline CLI
modifier and PDG behavior override stays in-engine. A hashed `/bin/true`
admission trace records structural refusal before the unchanged oracle exec.
No input identity gate is necessary because both paths run the unchanged
pinned-oracle pipeline and only fixed process/control-plane work is removed.
D0224 remains in force: this is host dispatch, not native SMRF, HAG-NN, named
writer, or LAZ coverage.

**Continuation, 2026-08-14 (B0231/D0230):** the already-exact direct
`filters.neighborclassifier(k=7)` route now has its own measured placement
model and fail-closed automatic selector. The accepted public 1M process is
byte-, stdout-, stderr-, order-, and status-exact at **4.033570x median**
pinned PDAL (3.619967 s versus 0.897460 s). A nine-pair direct-versus-former-
resident attribution is resolvably 1.150355x +/- 0.024677 with 9/9 wins.

- Selection requires the literal LAS/filter/LAS grammar, uncompressed
  format-7 36-byte input/output records, 250K--16M points, `k=7`, mapped
  source, direct Classification publication, a 112-byte/point planner-owned
  index, and the exact RTX 4090/SM89/CUDA 13.3/driver 610.43.03 profile.
- The 50K control is an exact 0.685x loss and stays host-selected. Tests also
  prove pre-commit status-124 refusal for option, format, compression, `k`,
  source, and preflight drift; runtime units reject topology, layout, and index
  drift. The injected post-execution proof failure exits 1 without publication
  or host retry because execution is already committed.
- The calibration audit is 170/170. The full physical CUDA unit binary has
  597 registrations: 595 pass and two optional-corpus cases skip. The
  strengthened resident-selection matrix passes; the new 250K automatic route
  is clean under memcheck and racecheck. Host ASan/UBSan passes 404/406 with
  the same two optional-corpus skips. The only unfiltered Host Debug failure
  remains the unrelated,
  reproducible HAG-Delaunay argv-prefix diagnostic mismatch.
- This completes one of the six genuine automatic-selection candidates. The
  next queue is sort, label-duplicates, skewness-balancing, HAG-NN, and
  HAG-Delaunay; do not widen neighborclassifier beyond its measured shape.

**Continuation, 2026-08-14 (B0232/D0231):** the already-exact mapped-source/
permutation-publisher `filters.sort(Z,ASC,NORMAL)` route now has its own
measured placement model and fail-closed automatic selector. Final ordinary
public commands are byte-, metadata-, stdout-, stderr-, status-, and
order-exact at **1.842475x median** pinned PDAL at 600K (0.479522 s versus
0.260260 s) and **3.267173x** at 1M (0.957273 s versus 0.292997 s), with 9/9
wins at both sizes.

- Selection requires the literal option-free LAS/sort/LAS grammar,
  `extra_dims=all`, uncompressed format-7 36-byte input/output records,
  finite comparator-unique Z, 600K--16M points, one 8-byte upload and one
  8-byte download per point, zero packing/indexes, one lane, and the exact
  RTX 4090/SM89/CUDA 13.3/driver 610.43.03 profile. The 500K and 550K
  direct-versus-public intervals span parity and stay host-selected.
- Actual requested CUDA allocation peaks at 33,769,475 bytes at 600K and
  900,275,715 at 16M (56.283/56.267 bytes per point). Placement and resident
  preflight conservatively reserve 64 bytes/point. The physical process gate
  accepts exactly `64*N` and refuses `64*N - 1`; the allocator tracking unit
  independently freezes the real high-water.
- Data-dependent duplicate/non-finite refusal and source/preflight/proof drift
  fall back before publication. Existing, aliased, and symlink destinations
  also return to the unchanged oracle while the atomic publisher has no side
  effect; commitment now occurs only after successful publication. Required
  refusals return 124 and preserve the destination.
- The raw calibration verifier is 183/183 and the compiled placement audit is
  177/177. Host Debug and leak-disabled Host ASan/UBSan pass 406/408 with two
  optional-corpus skips. Physical CUDA Release passes 598/600 with the same
  skips. The expanded automatic process matrix passes; 600K memcheck has zero
  errors and racecheck has zero hazards/errors/warnings.
- This completes sort without reopening B0130's rejected ordinary boundary.
  The next automatic-selection candidates are label-duplicates,
  skewness-balancing, HAG-NN, and HAG-Delaunay. Do not widen sort beyond its
  measured direct shape.

**Continuation, 2026-08-14 (B0233/D0232):** the fastest exact
`filters.label_duplicates(Classification) -> filters.nndistance(k=10) ->
filters.assign(UserData = Duplicate)` composition now has its own bounded
complete-process selector. Final option-free public commands are byte-,
metadata-, stdout-, stderr-, status-, and order-exact at **2.846286x median**
pinned PDAL at the 250K floor (1.137103 s versus 0.399504 s) and
**6.391572x** at 1M (4.579053 s versus 0.716421 s), with 9/9 wins at both
sizes.

- Selection requires the literal five-stage LAS/label/NNDistance/assign/LAS
  grammar, exact option spellings, uncompressed LAS 1.4 format-7 36-byte input
  with a 375-byte header, zero VLRs/EVLRs and no trailing bytes, 250K--16M
  points, and the exact RTX 4090/SM89/CUDA 13.3/driver 610.43.03 profile.
- The current forced-hybrid ladder is exact and positive from 250K through
  16M at 3.261x--12.190x. The 50K control is a 0.893x loss and stays host-
  selected. The independent affine selector has at most 6.08% residual over
  the accepted ladder and is isolated from ordinary resident placement.
- A disposable current resident probe is exact at 6.767x pinned PDAL but 4.3%
  slower than the forced hybrid at 1M. It was reverted. The automatic resident
  path declines this grammar early so it neither preempts nor taxes the faster
  hybrid composition.
- The public process matrix covers exact selection and actual CUDA use,
  below-floor, format, header-padding, VLR/EVLR, trailing-byte, record-stride,
  uppercase-reader/writer, disabled/unavailable, and recoverable-CUDA fallback;
  default-oracle plus required literal root/stage/option drift; and oracle-
  identical existing/alias/symlink behavior. Host Debug and leak-disabled
  ASan/UBSan pass 407/409; physical
  CUDA Release passes 599/601; both aggregates have only the two documented
  optional-corpus skips. The actual 250K automatic process is clean under
  memcheck and racecheck.
- This completes label-duplicates only inside its measured composition. The
  next automatic-selection candidates are skewness-balancing, HAG-NN, and
  HAG-Delaunay. Do not promote standalone label-duplicates or revive the
  slower resident prototype.

**Continuation, 2026-08-14 (B0234/D0233):** the already-exact mapped-source/
permutation-publisher `filters.skewnessbalancing` route now has its own
measured placement model and fail-closed automatic selector. Final option-free
public commands are byte-, metadata-, stdout-, stderr-, status-, and order-
exact at **1.200177x median** pinned PDAL at the 450K floor (0.426586 s versus
0.355436 s) and **3.091007x** at 1M (1.328755 s versus 0.429878 s), with 9/9
wins at both sizes.

- Selection requires the literal LAS/skewness/LAS grammar, `extra_dims=all`,
  uncompressed LAS 1.4 format-7 36-byte input/output records, mapped source,
  finite comparator-unique physical Z, 450K--16M points, one 8-byte upload and
  one 8-byte permutation download per point, and the exact RTX 4090/SM89/CUDA
  13.3/driver 610.43.03 profile. A final public 400K control is only 1.065118x
  and stays host-selected.
- Placement and resident preflight reserve 65 bytes/point: the established
  56-byte ordering scratch plus concurrently retained binary64 Z and byte
  Classification columns. The physical route accepts exactly `65*N` and
  refuses `65*N - 1`.
- Ties, signed-zero-equivalent keys, non-finite Z, source/preflight/proof drift,
  and existing/alias/symlink publication refusals all fall back before
  commitment. Required-route tests prove actual CUDA use at 450K, 1M, and the
  16M cap, plus default fallback and exit 124/no output at 16,000,001 points.
- The compiled placement audit is 190/190 and raw provenance verification is
  196/196. Host Debug and leak-disabled ASan/UBSan each pass 496 tests with
  four optional skips. Physical CUDA Release has 777 passing runnable tests
  and 17 optional skips after the unrelated outlier process timeout is raised
  from 120 to 300 seconds; the isolated matrix passes in 122.17 seconds. The
  actual 450K automatic route is clean under memcheck and racecheck.
- This completes skewness-balancing only inside its measured direct
  composition. The next automatic-selection candidates are HAG-NN and
  HAG-Delaunay. Do not widen the ordinary skewness boundary or neighboring
  layouts.

**Continuation, 2026-08-14 (B0235/D0234):** the already-exact count-one
`filters.hag_nn` mapped-source/direct-double-output route now has its own
measured placement model and fail-closed automatic selector. Final ordinary
public commands are byte-, metadata-, stdout-, stderr-, status-, and order-
exact at **1.216823x median** pinned PDAL at the 450K floor (0.411108 s versus
0.337854 s) and **2.461791x** at 1,000,002 points (0.931082 s versus
0.378213 s), with 9/9 wins at both sizes.

- Selection requires the literal option-free LAS -> `hag_nn(count=1)` -> LAS
  grammar except for writer `extra_dims=all`; lowercase `.las` endpoints;
  uncompressed LAS 1.4 format-7 input with 40-byte records containing exactly
  one unsigned-32 `OffsetTime` Extra Bytes descriptor; 48-byte output records;
  mapped source; 450K--16,000,002 points; one planner-owned 2D kNN index,
  region, and lane; and the exact RTX 4090/SM89/CUDA 13.3/driver 610.43.03
  profile.
- The separate `hag-nn-count1-direct-compose` model includes a 25-byte/point
  upload, 8-byte/point spill, zero packing, the full 112-byte/point shared
  index, and two synchronizations. A 400,002-point direct result is only
  1.104286x, so the safer 450K floor is retained. The measured 16,000,002 row
  is the hard cap.
- Placement and resident preflight reserve a conservative 160 bytes/point;
  stage-local scratch is 15 bytes/point. The physical process accepts exactly
  `160*N` and refuses `160*N - 1`.
- Source, grammar, layout, count, preflight, proof, and side-effect-free
  publication refusals fall back before commitment. Existing, aliased, and
  symlink destinations retain pinned-oracle status, diagnostics, and state;
  required refusals return 124 without output. The selected route proves an
  executable rewrite, one selected HAG region, one shared index, the direct
  mapped source/extra-double publisher, and actual device execution.
- The compiled placement audit reports 201/203 directions. Its two misses are
  deliberate conservative refusals at 300K (1.006644x) and 400,002
  (1.104286x), not wrongly selected losses; every accepted 450K--16,000,002
  row and every losing control matches. Host Debug and leak-disabled
  ASan/UBSan pass their focused and aggregate gates with only the two
  documented optional-corpus skips; physical CUDA Release does likewise. The
  actual 450K public route is clean under Compute Sanitizer memcheck and
  racecheck.
- This completes HAG-NN automatic selection only for count one inside the
  measured direct composition. Counts 2--64 and every neighboring layout,
  option, cardinality, device, and profile remain exact but explicit or host-
  selected. The next automatic-selection candidate is HAG-Delaunay.

**Continuation, 2026-08-14 (B0236/D0235):** the exact count-three
filters.hag_delaunay mapped-source/direct-double-output route now has its own
measured placement model and fail-closed automatic selector. Final ordinary
public commands are byte-, metadata-, stdout-, stderr-, status-, and
order-exact at **1.121367x median** pinned PDAL at the 450K floor (0.375380 s
versus 0.334752 s) and **2.106454x** at 1,000,002 points (0.835361 s versus
0.396572 s), with 9/9 wins at both sizes.

- Selection requires the literal option-free LAS ->
  hag_delaunay(count=3) -> LAS grammar except for writer extra_dims=all;
  lowercase .las endpoints; uncompressed LAS 1.4 format-7 input with 40-byte
  records containing exactly one unsigned-32 OffsetTime descriptor; 48-byte
  output; mapped source; 450K--16,000,002 points; one planner-owned 2D kNN
  index, region, and lane; and the exact RTX 4090/SM89/CUDA 13.3/driver
  610.43.03 profile.
- The separate hag-delaunay-count3-direct-compose model includes a
  25-byte/point upload, 8-byte/point spill, zero packing, full 112-byte/point
  shared index, and two synchronizations. The 400,002 direct row is only
  1.068195x and stays host-selected.
- Placement and resident preflight reserve 184 bytes/point; local count-three
  scratch is 39 bytes/point. The physical process accepts exactly 184*N and
  refuses 184*N - 1.
- The public matrix proves floor/main/cap selection, cap-plus-one and memory
  refusals, grammar/layout/source/preflight/proof drift, bounded tie and
  incomplete repair, the pinned no-ground diagnostic/status, and
  existing/alias/symlink publication behavior. Commitment follows successful
  execution proof and atomic publication.
- The compiled placement audit matches 215/218 directions, with only the new
  400,002 marginal win deliberately refused; provenance verifies 224/224 raw
  reports. The automatic-selection candidate queue is complete. Default,
  count-four/wider, explicit option variants, neighboring layouts, and other
  devices/profiles remain exact but explicit or host-selected.

**Continuation, 2026-08-14 (B0237/D0236 correction):** independent review
found that B0236 could overwrite the selected calibration with a generic
estimate and proved route selection without proving successful CUDA execution.
It also lacked same-stride semantic Extra Bytes rejection and a final
same-route profile/provenance proof. Those gaps are closed: the calibrated
placement survives rewrite, publication requires observed successful exact
HAG-Delaunay CUDA use, a selected device decline is precommit fallback, and
the diagnostic stats path activates the same literal automatic selector.

- A fresh nine-pair 450K public run is only **1.093356x median** with a
  1.071909 paired lower bound, below the predeclared 10% margin. The automatic
  floor is therefore **500,001**, not 450K; the cap remains 16,000,002.
- Final public commands are exact at **1.245213x** at 500,001 points
  (0.426662 s versus 0.342642 s) and **2.049354x** at 1,000,002 points
  (0.838204 s versus 0.409009 s), with 9/9 wins at both sizes and both paired
  lower bounds above the margin.
- The proof-bearing 1,000,002-point same-final-binary stats run is exact at
  2.173383x and proves device placement, calibrated executor agreement,
  `planner_resident_shared_index_direct_las`, one region/index/lane, mapped
  source, 40 -> 48-byte direct output, no host XYZ mirror, boundary agreement,
  and the exact `184*N` peak.
- The final automatic Nsight report records 20 launches/2.421312 ms;
  `knnGatherKernel` remains the limiter at 1.954048 ms and the Delaunay kernel
  consumes 0.113440 ms. The separate same-binary benchmark report carries the
  exact-output proof because Nsight did not retain its tmpfs artifact.
- The corrected process matrix adds 500,000 refusal, incompatible same-stride
  descriptor refusal, and injected post-selection device decline, while
  preserving cap, budget, repair, no-ground, and publication-collision gates.
  The placement audit is now 214/218 and provenance remains 224/224.
- Final validation is 500/504 Host Debug, 779/800 physical CUDA Release,
  focused ASan/UBSan 15/15, benchmark-runner contracts 27/27, and zero
  memcheck errors or racecheck hazards/errors/warnings on the actual
  500,001-point automatic route.

The original six-candidate automatic-selection queue remains complete. The
current HAG-Delaunay automatic envelope is only the literal
500,001--16,000,002 format-7/40 -> 48-byte SM89 composition; resume
reference-pipeline attribution or host-side work under priorities 5/6.

**Continuation, 2026-08-15 UTC (B0238/D0237):** the measured dominant cost in
B0043's 4M LOF endpoint was the pinned host closure repair, not a CUDA kernel
or publisher. B0039's unproved device-tie direction remains rejected. The
retained host optimization builds the same single nanoflann compatibility
index over a contiguous immutable binary64 XYZ backing only for large LOF
repair closures, then executes deterministic fixed-chunk k-distance, density,
and factor passes with join barriers. Small repairs and ordinary KD3 callers
remain serial/uncached; no private index or automatic-envelope change follows.

- Same-final-binary three-pair attribution is exact at **1.840449x** over the
  cache-disabled control: 3.728858155 s versus 2.026059035 s median. Matched
  resident stats reduce exact repair from 2.183796292 to 0.504709589 seconds,
  retaining one index, 12,716 ambiguous rows, one incomplete row, and 40,982
  factor repairs. The final proof reports 11 workers and 4M cached rows.
- The final public one-warmup/nine-pair report is exact at **18.300959x**
  pinned PDAL: 35.860210317 s versus 1.959471607 s. All nine samples are
  faster; artifact SHA-256 remains
  `e68b4dded6f5571d8eb45afad9fa5698b4a034d56f7083d39b3ab8002fa80a3a`.
  Report SHA-256 is
  `b59d9906da41863e7183f8d6d6d2f8c14c77be76c39aaeb91e9b97f5dc2e42ca`.
  A separate proof-free three-pair run is exact at **18.550858x** and records
  both proof fields false.
- The cache payload is exactly 24 bytes/point, 96 MB at 4M. Matched peak RSS is
  935,892 KiB disabled versus 1,039,116 KiB enabled, a 103,224-KiB observed
  delta including allocator/process variation. Cached build failure is
  exception-atomic and retries uncached during ordinary execution; proof runs
  fail closed before oracle delegation.
- Host Debug and leak-disabled ASan/UBSan each run 506 registrations with 502
  passes and the same four optional-corpus skips. Both KD3 tests are
  TSan-clean. Physical RTX 4090 passes 13/13 focused
  Dispatcher/KD3/LOF/process tests, and benchmark-runner contracts pass 43/43.
  No CUDA translation unit changed,
  so there is no new Compute Sanitizer claim.

Resume with another measured reference-pipeline, host, or I/O limiter. Do not
widen B0043's selector from this result, apply cached backing to ordinary KD3
callers without a new memory/performance proof, or revive device LOF repair
without exact tie-order evidence.

**Continuation, 2026-08-15 UTC (B0239/D0238):** the complete r2 profile showed
that the profitable selective slice was HAG-NN, while option-free SMRF and the
named LAZ writer should remain host-owned. The existing exact CUDA HAG lane
reported 347 equal-distance rows. Recomputing the full column on host was exact
but 0.819971x; repairing only those compact rows through a planner-owned pinned
ground KD2 compatibility product made the complete workflow profitable.

- Corrected final public r2 is exact at **1.270989x**: 1.536685334 s pinned
  PDAL versus 1.209046799 s PDG, with a 1.280322x +/- 0.022397 paired mean and
  9/9 wins.
  Artifact SHA-256 is
  `815799548d7bd1d77933d6fbda34089e2caf6e8b632d52c30628e49111554327`;
  report SHA-256 is
  `4095fa468027b1eb56753e8bdab544ab48e2ec7ebfab2902903430c9bb7bcc50`.
- Automatic admission is intentionally limited to the immutable 1M reference
  fixture, its full payload fingerprint/header/extent, literal option-free
  SMRF/HAG grammar, exact named output layout, and the qualified SM89 profile.
  Wrong count, header, payload, or endpoint case refuses before output.
- A first automatic attempt stayed exact but lost at 0.918077x because the
  generic resident probe preempted the hybrid and the wrapper did not request
  CUDA. Both failures are retained as negative evidence and now covered by
  execution proofs.
- Physical CUDA and host differential gates pass, leak-disabled ASan/UBSan is
  clean, and the public route has zero memcheck errors and zero racecheck
  hazards/errors/warnings. The final profile identifies lazperf decode and
  remaining host/I/O/startup work, not HAG projection, as the next r2 limiter.
- Independent review found and the final code closes external internal-marker
  injection, proof-bearing CUDA decline, private compatibility-index ownership,
  and unrecorded ambient benchmark controls. The engine reparses the literal
  graph, the proof fails before writer execution, the KD2 repair product is
  planner-owned, and the v3 harness records candidate-only proof overrides.

**Current next action:** do not choose another optimization target yet. Expand
the reference suite from six to fourteen first-class workflows by adding
`r7`--`r14` for DSM, orthophoto/raster colorization, polygon clipping with
geometry reprojection, decimation, PDAL-native classification/refinement,
tiling/chipping, multi-file merge/mosaic, and format conversion/compression.
Add a canonical manifest, deterministic pipelines and fixtures, dynamic
materialization, exact raster/vector/multi-output comparison, cold/warm I/O,
pinned-oracle baselines, complete-process profiles, and a prioritized ledger.
Headline reporting must include the equal-workload geometric mean and the
paired total wall time for one run of every one of the fourteen workflows.

**Continuation, 2026-08-15 UTC (B0240/D0239):** the prerequisite above is
complete. `bench/pipelines/reference/manifest.json` now fixes fourteen
equal-weight headlines plus zero-weight r10/r14 companion matrices. The
checked-in fixture recipe authenticates the retained AHN-derived LAZ and
derives the LAS, deterministic EPSG:3857 RGB raster, heterogeneous merge pair,
while the established COPC encoding remains independently hash-registered,
all below the build tree without modifying corpus data. The harness
materializes named inputs, bounds, clip WKT/SRS, tile origins, and output
patterns; compares artifact membership/bytes, diagnostics/status, and complete
normalized raster metadata; samples process-tree peak RSS; and fails closed on
an uncontrolled cold-cache run, a missing/inexact workload, or a partial
aggregate.

- All fourteen headlines and twelve companion variants are exact in three
  warm and three cold alternating pairs. r7 is an exact 1,500 x 77 Float64
  EPSG:28992 maximum-Z GeoTIFF; r12 is an exact seven-artifact tile set.
- Warm equal-workload geometric mean is **1.211602x** and warm total-wall is
  **1.563923x** (27.274096 / 17.439535 s). Cold values are **1.206003x** and
  **1.554365x** (27.275383 / 17.547600 s). No loss is omitted or downweighted.
- Complete-process r7-r14 profiles and the ordered ledger are recorded in
  B0240 and `bench/pipelines/reference/OPPORTUNITY.md`. All new workloads
  remain on exact host paths; none of their first three-pair results clears a
  stable positive selection gate.
- r11 classification/refinement is the next measured vertical slice: pinned
  PDAL takes 6.983998 s warm while PDG takes 7.735668 s. Start with a bounded
  exact composition test/prototype for SMRF -> statistical outlier -> k=7
  neighbor classification. Do not infer automatic admission from the existing
  standalone stages, and do not optimize all eight new workflows together.

The earlier “current next action” paragraph is retained as chronology and is
superseded by this continuation.

**Continuation, 2026-08-15 UTC (B0241/D0240):** the smallest r11 slice is
complete. A test-first public direct-delegate prototype was exact but resolved
a 0.901714x loss with 0/9 wins and was removed. The regression came from
B0238's optional KD3 coordinate cache leaving a cache-state branch in every
ordinary nanoflann point/distance callback. `KD3ImplT<false>` and
`KD3ImplT<true>` now make backing a construction-time choice: ordinary host
queries regain the original branch-free callbacks, while explicit large LOF
repair retains contiguous immutable coordinates.

- r11 is exact at unresolved parity: 6.992162/6.985246 seconds pinned/PDG
  over nine warm pairs (1.000990x; paired mean 0.999961x +/- 0.003881) and
  7.013476/6.986604 seconds over three cold pairs (1.003846x).
- The retained physical 4M LOF cache proof is exact at 20.295840x pinned PDAL,
  confirming that parity restoration did not sacrifice B0238's qualified win.
- The updated fourteen-workload aggregates are **1.220209x** warm and
  **1.215129x** cold equal-workload geometric mean, plus **1.632359x** warm
  and **1.627712x** cold total-wall speedup.
- Host Debug is 508/508 passed with four documented optional corpus skips;
  focused ASan/UBSan and TSan exactness/failure/concurrency gates pass. No CUDA
  translation unit or selection envelope changed.

**Current next action:** r8 colorization is the largest remaining
unaccelerated headline wall. Profile and attribute the two PROJ transforms,
GDAL sampling/cache, reader decode, writer compression, and dispatch/startup;
then take only the smallest complete exact measured slice. r11 remains
host-selected, and its domain-qualified neighbor classifier must not be
misreported as GPU-native.

**Continuation, 2026-08-15 UTC (B0242/B0243, D0241/D0242):** r8 attribution is
complete and the proposed shortcut is rejected. The graph's engine route does
no stage work before launching sibling PDAL, but a corrected-final 21-pair
direct-versus-forced-engine control remains unresolved at 0.995822x +/-
0.004522 in report orientation. Direct public execution is also a resolved
0.988765x cold loss against pinned PDAL. Exactness alone is insufficient under
D0208, so the r8 matcher/probe was removed and the measured graph is explicitly
tested to remain engine-owned.

- The retained r8 three-pair rows are exact at 0.997701x warm and 0.986213x
  cold. Reprojection, GDAL colorization, and LAZ I/O remain unchanged host work.
- Independent review found an omitted engine-only proof variable in the old
  direct-route allowlist. The launcher now snapshots the actual environment;
  every `PDG_*` name except `PDG_ORACLE_PDAL` fails closed to the engine, so
  future proof and diagnostic controls cannot silently bypass it.
- Final aggregates are **1.220823x** warm and **1.214832x** cold equal-workload
  geometric mean, plus **1.633139x** warm and **1.627446x** cold total-wall
  speedup. No loss is omitted.
- Host Debug passes 510/510 with four optional corpus skips; focused
  leak-disabled ASan/UBSan and real process-boundary checks pass. No CUDA
  translation unit changed.

**Current next action:** r13 heterogeneous merge. Begin with complete-process
attribution of two-reader concurrency, layout/header assembly, input/view
order, writer compression, and public dispatch. Keep r8 host-selected and do
not revive its direct shortcut without a new resolved same-final gate.

**Continuation, 2026-08-15 UTC (B0244/B0245, D0243/D0244):** the smallest r13
slice is complete and its automatic prototype is rejected. Profiling shows the
two heterogeneous LAZ readers execute in input order, streaming merge itself
is effectively free, and lazperf decode plus writer compression dominate.
Direct sibling dispatch alone was exact but unresolved; one LAS reader worker
was the only positive tuning hypothesis.

- Independent review caught unmeasured CLI modifiers and non-portable loader
  linkage in the first prototype. After both were fixed, the corrected-final
  21-pair warm gate resolved at **1.009929x**, but the required 41-pair cold
  gate was only **1.001186x** and unresolved.
- A public route cannot know cache state. The matcher, fingerprint, proof, and
  override were removed under the fail-closed rule; r13 remains exact
  engine/host execution with no native or performance-qualified claim.
- Removal restores the exact B0243 launcher/engine binaries. Current
  aggregates therefore remain **1.220823x** warm and **1.214832x** cold
  equal-workload geometric mean, plus **1.633139x** warm and **1.627446x**
  cold total-wall speedup. The pre-correction B0244 aggregates are historical
  rejected-prototype evidence, not current claims.
- Host Debug remains 510/510 with four optional corpus skips; focused
  leak-disabled ASan/UBSan and Release process-boundary checks pass. No CUDA
  translation unit changed.

**Current next action:** r12 deterministic tiling. Attribute exact splitter
ownership, seven-output compression/publication, naming/order, and startup;
take only a complete measured slice that preserves the full artifact set and
fails closed before publication.

**Continuation, 2026-08-15 UTC (B0246/D0245):** r12 attribution is complete
and both bounded tuning directions are rejected. The retained profile shows
launcher -> engine -> sibling PDAL and is dominated by format-7 LAZ decode;
the seven-output write syscall time is only a few milliseconds at this size.

- Direct sibling execution is exact but unresolved against pinned PDAL at
  0.997671x warm over 21 pairs and 0.995241x cold over 41 pairs.
- The literal integrated public prototype resolves slower at 0.959977x warm
  and 0.961579x cold. Its required same-final direct-versus-forced-engine A/B
  spans parity in both cache states, so removing the wrapper is unproved.
- One, two, six, and eight LAS reader workers resolve slower; four workers is
  exact but unresolved. The default seven-worker reader remains selected.
- The matcher and positive route tests were removed. The restored launcher and
  engine hashes exactly match B0243/B0245, and a new unit/process negative
  route freeze keeps the literal r12 graph engine-owned.
- The reference harness now rejects fixture-output aliases and duplicate
  artifact patterns before execution, closing r12's multi-output collision
  contract instead of silently deduplicating it.
- Host Debug passes 511/511 with four optional corpus skips; focused CUDA
  Release and leak-disabled ASan/UBSan dispatcher gates pass. No CUDA source or
  runtime selection changed.
- Current aggregates remain **1.220823x** warm and **1.214832x** cold
  equal-workload geometric mean, plus **1.633139x** warm and **1.627446x**
  cold total-wall speedup.

**Current next action:** r9 standalone polygon clipping. Profile geometry CRS
reprojection, polygon/multipolygon-with-hole membership and boundaries,
decode/encode, and public startup. Preserve exact ownership/order, bytes,
metadata, diagnostics, and status; keep the engine/host route unless a new
same-final warm and cold gate resolves faster.

**Continuation, 2026-08-15 UTC (B0247/D0246):** r9's original headline was
invalid: EPSG:4326 authority-axis output was inserted as traditional WKT X/Y,
swapping longitude/latitude and producing an empty 1,899-byte LAZ. The
materializer now reverses the `cs2cs` output axis order and its regression
requires Dutch longitude/latitude domains. Corrected r9 performs real
multipolygon, hole, disjoint-member, boundary, and geometry-SRS work and emits
473,825 points in an exact 3,205,857-byte LAZ.

- An eight-case pinned-oracle process matrix adds positive polygon and
  multipolygon/hole/boundary/reprojection cases, missing/malformed geometry
  inputs, and repeat determinism.
- Complete-process profiles are lazperf-decode shaped; polygon membership and
  filesystem calls are small. The retained deficit is public startup.
- Direct sibling PDAL is at unresolved pinned parity and resolves faster than
  the same-final forced-engine route. The integrated literal direct-exec
  prototype is nevertheless a resolved 0.957175x pinned-PDAL loss over 21 warm
  pairs. A private in-process PDAL CLI prototype is also a resolved loss at
  0.951883x. Both prototypes and all positive routing code were removed.
- Corrected r9 is frozen engine-owned in dispatcher unit and real process
  tests. It remains exact PDAL host polygon/reprojection execution with no new
  host-native, GPU-native, performance-qualified, or automatic-selection
  claim.
- Host Debug passes 513/513 with four optional local-corpus skips. Focused
  CUDA Release and leak-disabled ASan/UBSan dispatcher, harness, manifest, and
  polygon-matrix gates pass.
- The substantive aligned suite supersedes every aggregate containing the old
  empty r9 row. A fresh execution of all fourteen rows with one retained PDG
  hash and one pinned-oracle hash reports **1.205506x** warm and **1.206910x**
  cold equal-workload geometric mean, plus **1.617787x** warm and **1.615619x**
  cold total-wall speedup. Aggregation rejects binary drift and r9 point-count
  drift; no loss is omitted.

**Current next action:** r7 DSM. Begin with its complete-process decode/raster
profile and test the smallest reusable reader/startup or raster-publication
hypothesis. Preserve the maximum-Z, first/only-return surface policy and full
GeoTIFF byte/metadata contract; keep it host-selected unless both corrected
cache-state gates resolve faster.

**Continuation, 2026-08-15 UTC (B0248/D0247):** r7 DSM now uses the fastest
exact measured PDG backend. The unchanged engine path was about 0.94x pinned
PDAL; removing the unused engine process for only the literal first/only,
maximum-Z, one-metre Float64 headline resolves **1.058781x warm** and
**1.049022x cold** against the same-final engine route with 9/9 wins in each
state. Public PDG is still a reported 0.982017x warm loss and unresolved
0.990449x cold against pinned PDAL, so this is a bounded dispatcher improvement,
not a claim that r7 or GDAL beats PDAL.

- A generated compressed-LAZ selected-route matrix plus bounded fallback
  cases proves full GeoTIFF bytes/metadata, first/only maximum-Z policy,
  malformed groups/bounds, refusals, diagnostics/status, and repeat
  determinism. Root, graph, option type/value, SRS, bounds, extension, CLI, or
  `PDG_*` drift remains engine-owned.
- The benchmark clock moved from product namespace `PDG_FROZEN_EPOCH` to
  harness-owned `PDAL_TEST_FROZEN_EPOCH`. The old name had correctly forced
  every thin route into the engine, so B0247's aggregate is historical. The
  runner now scrubs ambient clock input, rejects explicit injection, records
  the injected name, and process tests prove it does not alter dispatch.
- Final r7 emits the exact 924,954-byte 1,500 x 77 Float64 EPSG:28992 GeoTIFF
  at SHA-256 `55069aa3...`; complete normalized raster metadata hashes to
  `9d7ccf48...`. The final profile remains lazperf-decode shaped (0.303 of
  0.424 sampled CPU seconds inclusive in format-7 point decode).
- Fresh exact all-headline aggregates are **1.223460x warm / 1.227305x cold**
  equal-workload geometric mean and **1.600366x warm / 1.621301x cold**
  total-wall speedup. No loss is omitted.
- Host Debug passes all 515 registrations with four optional fixture skips;
  focused CUDA Release and leak-disabled ASan/UBSan selected-route,
  dispatcher, and harness gates pass. Independent re-review reports no
  remaining P1/P2 finding.
- r7 gains performance-qualified direct-host dispatch only. It adds no native
  returns, GDAL, raster, LAZ, or CUDA stage and no Compute Sanitizer claim.

**Current next action:** r10 decimation. Begin with its decode-shaped profile
and bounded headline/variant matrix. Attribute voxel-centroid construction,
nearest-original-record ownership/ties/order, LAZ decode/encode, and startup;
prefer a reusable decode/I/O improvement, and retain the host path unless a
complete exact warm/cold public gate resolves faster.

**Continuation, 2026-08-15 UTC (B0249/D0248):** r10 decimation now uses the
fastest exact measured PDG backend for one deliberately bounded headline
grammar. A complete-process profile attributes 0.302 of 0.430 sampled CPU
seconds to LAZ format-7 point decode but only 0.002 seconds inclusive to
`VoxelCentroidNearestNeighborFilter::run`; the profitable slice is removal of
the unused engine process, not a speculative voxel implementation.

- The literal three-stage graph is lowercase non-COPC LAZ ->
  `filters.voxelcentroidnearestneighbor(cell=2.5)` -> lowercase LAZ with the
  string option `compression:"true"`. Root, stage, option/type/value,
  extension, CLI modifier, or `PDG_*` drift stays engine-owned.
- A nine-case generated-fixture process matrix freezes selected dense, sparse,
  empty, malformed-input, and repeat-determinism behavior plus legal cell and
  writer-option drift, invalid string-cell refusal, and the pinned oracle's
  accepted numeric zero-cell behavior. It compares complete artifact bytes,
  point order, stdout, stderr, and status.
- The final public route is unresolved against pinned PDAL at 1.002615x warm
  and 0.993073x cold median speedup. Against the same-final forced-engine path
  it resolves **1.041226x warm** and **1.034230x cold**, with 9/9 wins in both
  states. The exact 122,162-point, 1,189,864-byte LAZ hashes to
  `3b3f920f...` on every measured side.
- Fresh exact all-headline aggregates are **1.235323x warm / 1.237401x cold**
  equal-workload geometric mean and **1.631522x warm / 1.633568x cold**
  total-wall speedup. No loss is omitted.
- Host Debug passes all 517 registrations with four optional fixture skips;
  focused CUDA Release and leak-disabled ASan/UBSan r10 matrix, dispatcher
  unit, and process gates pass.
- r10 gains one performance-qualified automatic direct-host dispatch. It
  remains functionally exact unchanged-PDAL host execution and adds no
  host-native voxel/LAZ stage, GPU-native stage, CUDA selection, or Compute
  Sanitizer claim. All six zero-weight r10 companion variants remain
  engine/host selected.

**Current next action:** r14 conversion/compression. Re-profile the complete
LAS -> LAZ headline and its zero-weight direction matrix, attribute LAZ
reader/writer/compression, filesystem, publication, and startup costs, and
retain only an exact complete-process change that clears the warm and cold
same-final gate without losing pinned-oracle compatibility.

**Continuation, 2026-08-15 UTC (B0250/D0249):** the expanded r7--r14
definition/fixture/baseline/profile ledger is now closed through its final
headline. r14 already inherits the older generic two-stage LAS-family direct
delegation, so there is no unused engine process to remove and no new route was
added.

- A 12-case/19-execution generated-fixture matrix covers LAS -> LAZ, LAZ ->
  LAS, LAZ recompression, LAS/LAZ -> COPC, COPC -> LAS/LAZ, repeat byte
  determinism for every supported direction, truncated/malformed LAS/LAZ/COPC,
  unsupported writer refusal, and the pinned writer's accepted odd compression
  spelling. It verifies complete artifacts, LAS counts/compression flags, COPC
  identity VLR, point order through byte equality, stdout, stderr, and status.
- The public profile follows the launcher into sibling PDAL and observes a
  0.535-second child, 37,323,822 bytes read, and 6,748,024 bytes written. Its
  child clock sampler reports a timer-reset warning and only 0.030 sampled CPU
  seconds, so no unsupported function-level percentage is claimed. The source
  path still shows sequential `LasWriter` point packing and lazperf compression.
- A bounded 1/2/4/6/8 reader-worker sweep is exact. Only one worker clears the
  three-pair warm screen; its nine-pair confirmation is **1.017714x warm** but
  only 1.004441x cold median, with a cold paired result of 1.001577x +/-
  0.014533. The required cold gate is unresolved, so the prototype is rejected
  despite reducing median process-tree RSS from about 83.9 MB to 66.1 MB.
- No product binary changed. The current all-headline result therefore remains
  **1.235323x warm / 1.237401x cold** equal-workload geometric mean and
  **1.631522x warm / 1.633568x cold** total-wall speedup.
- Host Debug passes all 519 registrations with four optional fixture skips;
  the r14 matrix and dispatcher route freezes pass CUDA Release and
  leak-disabled ASan/UBSan. No CUDA source changed and no Compute Sanitizer
  claim is added.
- r14 remains functionally exact unchanged-PDAL host execution through an
  existing generic direct-host route. It adds no new performance-qualified
  selector and no host-native or GPU-native LAS/LAZ/COPC stage.

**Current next action:** test an exact parallel LAZ chunk-compression
feasibility slice at the r14 LAS -> LAZ boundary. First prove that independently
compressed lazperf chunks can be assembled into byte-identical PDAL output
with identical chunk table/header/VLR/EVLR bytes and bounded memory. Measure
the primitive and complete process before integrating; if byte identity or
warm/cold speed fails, retain the sequential writer and record the rejection.

**Continuation, 2026-08-15 UTC (B0251/D0250):** the exact parallel LAZ
compression feasibility slice is accepted for one bounded r14 shape.

- Independent format-7 chunks reproduce the complete sequential lazperf
  payload/table at 0, 1, 49,999, 50,000, 50,001, and 100,001 points. The
  production pool retains at most one raw/result chunk per worker and publishes
  futures in source order on the writer thread.
- The maintained one-million-point production primitive is byte-exact over
  6,469,296 bytes and measures **1.735238x** median speedup for two workers.
- The r14 matrix is now 13 cases/21 process differentials. Its generated 1M
  fixture is checked against every dispatcher input fact, and a candidate-only
  assertion proves the real selected `LasWriter` observed two workers.
- Only the literal 1M, 36,000,375-byte, LAS 1.4 format-7/36-byte reference
  layout and plain lowercase LAS -> LAZ grammar select. All neighboring facts,
  CLI modifiers, external `PDG_*` controls, and six companion directions retain
  the exact serial/engine route.
- B0252/D0251 make external environment refusal precede automatic routing,
  strip a forged internal worker value before engine fallback, and compile the
  activation assertion only in test builds.
- Corrected final nine-pair public results are **1.515827x warm** and
  **1.504457x cold**
  pinned PDAL, with 9/9 wins and exact bytes/streams/status in both states.
- Fresh all-headline results are **1.262824x warm / 1.260608x cold** geometric
  mean and **1.641092x warm / 1.634768x cold** total-wall speedup. All fourteen
  rows are exact.
- Final Host Debug is 520/520 with four optional fixture skips. The focused
  byte/activation/dispatcher/r14 gates pass CUDA Release and leak-disabled
  ASan/UBSan. No CUDA translation unit changed, so no Compute Sanitizer claim
  is added.

**Continuation, 2026-08-15 UTC (B0253/D0252):** the first two reusable LAZ
decode hypotheses are rejected without changing the selected product.

- `LasReader` already owns bounded chunk-parallel scheduling and ordered tile
  consumption. Wider 10/12-worker screens are slower for both r7 and r10,
  lose all measured warm pairs, and increase peak RSS; no cold lane or product
  control is promoted.
- A deterministic production-factory unit now proves complete repeated bytes
  for formats 6--8, four Extra Bytes, all scanner channels, 1/2/50K points,
  and bounded truncated-input refusal.
- Mode-gating unused Point14 compressor/decompressor setup improves factory
  construction **1.661797x** and full 1M-record primitive decode **1.014837x**.
  Public r10 warm remains unresolved at **1.012374x** median and paired
  **1.009303x +/- 0.013219**, so the production change is removed and no cold
  or r7 promotion run is claimed.
- The decoder unit and maintained primitive remain; B0252's selected shared
  library, r14 lane, and fourteen-workload aggregate remain current.

**Continuation, 2026-08-15 UTC (B0254/D0253):** the completed worker screen
selects one bounded r7 schedule and rejects the broader LAZ hypothesis.

- A forced-inline integer-decode prototype is exact but only 1.010245x at the
  primitive and is removed. Four workers resolve; six are unresolved, eight
  are slower, and B0253 already rejected ten/twelve.
- The retained final public r7 route is exact at **1.031858x warm** and
  **1.020398x cold** over 21 pairs, paired **1.028621x +/- 0.007070** and
  **1.016676x +/- 0.009502**. Complete TIFF metadata/bytes, streams, status,
  and refusals match pinned PDAL.
- Only the literal plain 1M compressed format-7/36-byte r7 DSM graph receives
  `--readers.las.threads=4`. r10's integrated warm result is positive but cold
  is unresolved, so r10 retains B0249's default-reader route. Every other
  shape remains on its prior exact path.
- Fresh physical aggregates are **1.267318x warm / 1.262102x cold** geometric
  mean and **1.642633x warm / 1.632910x cold** total-wall speedup. All fourteen
  rows remain present and exact.
- Every enabled test in the 522-registration Host Debug suite passes, with the
  same four optional tests disabled/skipped. Focused CUDA Release and leak-
  disabled ASan/UBSan selector, process, differential, and decoder gates pass.
  No CUDA source changed.

**Continuation, 2026-08-15 UTC (B0256/D0255):** the first measured r11 reuse
slice is retained after closing an independent-review exactness finding.

- Prefix attribution assigns about 0.455 seconds to LAZ read/write, 0.312
  incremental seconds to SMRF, 3.646 incremental seconds to statistical
  outlier, and about 2.53 seconds to the neighbor-classification tail.
- B0255's first implementation could adopt a stale KD3 product after
  `filters.normal -> filters.assign(XYZ)`. The corrected statistical outlier
  invalidates all incoming products before building the same exact nanoflann
  tree through `PointView::build3dIndex()`. The adjacent classifier still
  reuses that newly built tree; radius mode, query order, arithmetic,
  diagnostics, order, and routes are unchanged.
- The corrected final nine-pair public gate is exact at **1.029892x warm** and
  **1.037311x cold**, paired **1.035463x +/- 0.009249** and
  **1.039144x +/- 0.008688**, with 9/9 wins in both states.
- An eight-case/nine-execution r11 process matrix adds cached-index creation
  followed by generic X mutation to the headline, legal drift, invalid `k`,
  missing domain dimension, invalid outlier multiplier, and repeat
  determinism. Focused Debug, CUDA Release, and leak-disabled
  ASan/UBSan gates pass; Host Debug enumerates 523 registrations with every
  enabled test passing and the same four documented optional fixture lanes
  disabled/skipped. No CUDA source or native-stage count changes.
- Fresh complete physical-GPU aggregates are **1.267705x warm / 1.278088x
  cold** geometric mean and **1.656538x warm / 1.665729x cold** total-wall
  speedup. All fourteen rows remain present and exact; noisy three-pair rows
  remain fully weighted.

**Session stop:** B0256/D0255 are the completed boundary. Resume only from the
authoritative final-session subsection at the start of this section. Its sole
next hypothesis remains the allocation-free, ascending-key
Classification/k=7 vote tally; nothing after B0256 is accepted or in flight.

**Historical reference baseline (B0201, nine paired runs):** r5 0.9420x
±0.0059, r1 0.9372x ±0.0070, r3 0.9727x ±0.0042, r2 0.9913x ±0.0056 all
measurably slower; r6 0.9976x ±0.0090 and r4 0.9995x ±0.0041 at parity. **None
is faster than pinned PDAL.** B0224 supersedes r6, B0226 supersedes r1, B0227
supersedes r4, and B0228 supersedes r5 for their exact measured reference
routes; B0229 supersedes r3, and B0230 supersedes r2. Use those later records.

**Three method warnings, each learned by getting it wrong:**

1. **Never read `--stats` from `pdg resident` to explain a `pdg pipeline`
   timing.** Same function, `automaticAdmission` flipped (B0185).
2. **Use paired alternating runs, not medians of three.** Single-run spreads
   reach 463 ms; a median-of-3 sweep produced a clean story that was entirely
   noise (B0199--B0201). `benchmark_reference.py` now reports a paired standard
   error and a `resolvable` flag — read the flag before quoting a speedup.
3. **Check the exit path before believing a fast number.** Three separate runs
   this session looked like large speedups and had simply failed: a ferry to
   itself, hidden planner args rejected by the oracle, and a 27x that did no
   work (B0218, B0220).

**Read `docs/diagnostics.md` before investigating anything.** Several
conclusions here were wrong for a slice or more because an instrument existed
and was not used. `PDG_ORACLE_PDAL=/bin/true` and `PDG_DEBUG_ADMISSION_PHASES`
each overturned a confident, source-derived mechanism in one run.

---

**Closed this session.**

- **SMRF byte-exactness — FIXED (B0207--B0216).** pdg's `filters.smrf`
  disagreed with the oracle on 99 of 1,000,000 points in `Classification`,
  because its void fill broke distance ties at the eighth neighbour
  differently. Now **byte-identical at cell 1.0/2.0/4.0/8.0** and **13.5 ms
  faster** with 0/7 pairs slower. The fix injects the oracle's own `KD2Index`
  via a `RasterVoidFill` hook rather than approximating its tie order —
  B0215 showed that order is the tree's traversal and matches no short rule, so
  copying it would have fixed 2 of 3 contested cells and left the defect in
  place. `pdg_core` keeps its PDAL-free layering.
- **`filters.hag_nn` "unclaimed win" — WITHDRAWN (B0217/B0218).** It is 49% of
  r2's wall, but pdg's host implementation is **36% slower** than upstream's
  (1351.0 ms against 992.5 ms, byte-exact), and the CUDA route cannot engage
  from any pipeline: `tryCudaHagNnColumn` needs `region.maximumNeighbors`,
  a hidden planner argument only resident placement supplies, and `hag_nn` has
  no calibration model to earn placement. The qualified 2.5--3.0x routes are
  structurally unreachable.
- **Filter-only substitution — GATED (B0219--B0222).** A point-program region
  of value predicates with no `assign`/`ferry` now declines substitution at or
  above one million points. crop at 4M moves from **+51.9 ms to −8.3 ms**
  against the same binary with substitution off, while 250K keeps its −4.7 ms
  win. The calibrated fused arithmetic shape is **untouched at 9.147x**
  byte-exact (9.470x before), which is the evidence that B0180's failure mode
  was not repeated.
- **Device SMRF is capped at 4,096 raster cells** (`SmrfExactDeviceMaximumRaster
  Cells`) against r2/r3's 115,500 at `cell=1.0`, so it cannot engage on real
  terrain. Item 6's "SMRF measured at 0.599x on CUDA" therefore describes a
  raster of at most 64x64 and should not be cited as general evidence.

**Remaining, unreachable but real:** `fillRasterKernel`
(`src/stages/SmrfKernels.cu`) still implements the integer-metric fill rule.
Nothing measured is affected because the device path never runs, but a
selectable device SMRF would reintroduce the defect, and a device cannot host a
KD-tree — so either run the fill on host for the device path too (the raster is
small) or leave device SMRF unqualified.

### Session change log, 2026-08-12 (B0183--B0222) and where to distrust it

49 commits; 917 insertions across 18 product files. Every behaviour change
below is byte-exact against the pinned oracle and passes 775/775 CUDA and
486/486 sanitizer tests. **The confidence notes are the point of this entry** —
read them before building on any of it.

**Behaviour changes**

| # | Change | Files |
| --- | --- | --- |
| 1 | `normal-covariancefeatures-compose` calibration model + matcher; 1.458x--7.260x over a measured 50K--4M envelope (B0187) | `RuntimePlacement.cpp`, `PlacementProfile.cpp`, calibration JSON, tests |
| 2 | Option-free `.laz` **readers** are native; 4.079x--7.221x on compressed input (B0188, D0217) | `Pipeline.cpp`, `plan_test.cpp` |
| 3 | `outputRecordBytes` derived from the writer's layout instead of two constants; `formatCarriesField` exported (B0191, D0218) | `ResidentPipeline.cpp`, `LasFerry.cpp`, `io/Las.hpp` |
| 4 | A bare LAS reader→writer pipeline routes straight to the oracle; 19.34 ms → 0.96 ms at 10k (B0197, D0219) | `Dispatch.cpp`, `dispatcher_test.cpp` |
| 5 | `NoDeviceCandidate` and model-existence refusals hoisted ahead of CUDA discovery; ~246 ms returned to LAS→LAS SMRF pipelines (B0205/B0206, D0220) | `RuntimePlacement.cpp` |
| 6 | SMRF void fill uses the oracle's own `KD2Index` via a `RasterVoidFill` hook; 99 → **0** differing points and 13.5 ms faster (B0216) | `stages/Smrf.hpp`, `Smrf.cpp`, `PdgSmrfFilter.cpp` |
| 7 | Filter-only point-program regions decline substitution at ≥1M points; +51.9 ms → −8.3 ms at 4M, fused lane untouched at 9.147x (B0222) | `Hybrid.hpp`, `Hybrid.cpp`, `HybridPipeline.cpp` |

**Diagnostics only** (opt-in, change no values — see `docs/diagnostics.md`):
`--stats` gained a `plan` section emitted even when placement is unavailable
(B0185, D0215) and `input/output_record_bytes` (B0191);
`PDG_DEBUG_ADMISSION_PHASES` (B0196), `PDG_DEBUG_SMRF_FILL` (B0212),
`PDG_DEBUG_SMRF_DUMP` (B0213).

**Harness:** `benchmark_reference.py` reports a paired standard error and a
`resolvable` flag (B0200/B0201).

#### Least confident, most first

1. **The filter-only substitution gate (7).** Trust this least. It is a
   **default-path** behaviour change; its threshold sits at 1M where the
   measurement is *neutral* (+4.0 ms ± 9.3), so the boundary itself is inside
   noise — only ≤250K (win) and 4M (loss) are resolvable. The ladder is one
   fixture family, one machine, and `crop` is the only predicate measured at
   more than two point counts. **Unmeasured:** regions mixing a predicate with
   adjacent arithmetic, predicates followed by ordinal stages, and any
   expressionstats pipeline *after* the acceptance-rule change — I argued that
   change is inert when the new flag is zero, but did not measure it. Three
   prior attempts at this gate each failed on an unreachability I had not
   predicted, so my model of that code path has been wrong three times running.
2. **`outputRecordBytes` derivation (3).** The derived value is only consumed
   when a writer is native, and `extra_dims` writers are not. So the 100-byte
   path is **effectively unexercised in the shipped configuration** — I verified
   it once under the sink-admission change that was then reverted. The plain
   36-byte case is covered; the interesting case is not.
3. **The composition model (1).** Five points, one fixture, one machine. The
   50K floor rests on a single forced-device measurement, and I did not re-verify
   the model's predictions after (5) and (7) landed. `knn` is pinned to 8 by
   design, which is honest but makes it narrow.
4. **The coarse guard in (5).** It defers whenever a plan contains
   `filters.outlier`, `filters.nndistance` or `filters.radialdensity` — three
   stage types I chose by reading the two direct-region matchers. If a future
   composition region is built from different stages, the guard silently stops
   covering it. D0220 records this; it is a real fragility, not a hypothetical.
5. **`.laz` reader nativeness (2).** Three independent guards keep mapping off
   compressed records and byte-exactness held across a ladder, but I *reasoned*
   about the guards rather than testing a compressed direct-source path,
   because it is unreachable today. Guard order matters if that changes.
6. **The harness interval.** Two standard errors is a normal approximation, not
   a t-interval, at n=9. It is labelled an approximation and is directionally
   right, but do not treat `resolvable` as a formal test. I also shipped the
   *wrong* statistic first (a range, which grows with sample size) and corrected
   it an hour later — B0200 versus B0201.

Most confident: the SMRF fix (6), because it is exact by construction rather
than by tuning — it asks the oracle's own index instead of reproducing its tie
order — and the diagnostics, which cannot change a value.

**Live threads, each with a measured payoff:**

- **`extra_dims` sink admission** — worth **6.227x byte-exact** on the features
  shape. Blocked on one stage shape: `filters.sort` writes no columns, so a
  native sink after it has nothing to name and the spill is empty. Two local
  fixes are already refuted (B0193); the next move is the direct route's
  publication contract, not a third patch.
- **Dispatcher layering** — worth ~15 ms on every pipeline the engine refuses.
  The "will the engine help?" decision costs 0.2 ms but is taken after a
  17--38 ms image load. Moving it into the dispatcher needs `pdg_core`'s plan
  compilation, which has no CUDA dependency (B0198).
- **Ungated hybrid substitution** — `tryHybridPipeline` substitutes on
  structural grounds with no cost model, the opposite of the resident path.
  Measured to cost **r1** 18.6 ms across 15/15 paired runs; r2, r4 and r6 are
  unresolved, which is itself the argument for gating (B0199).
- **`filters.hag_nn` is 49% of r2** and is accelerated to 2.5--3.0x in this
  fork, but only through explicit routes — automatic selection is prohibited
  by data-dependent repair and an absent model. This is the largest single
  unclaimed win in the reference set (B0203).

**Work-order status:** item 4 closed previously; this session advanced subsets
of item 1 and opened item 6. **Items 2, 3, 5 and 7 are untouched, and P4, P5
and P6 are untouched.** P4 alone (COPC pushdown, EPT) exceeds this session.

- Branch: `p3-skewness-resident`, B0222 filter-only-gate checkpoint. **Read
  D0208 first: the goal is now speed on the reference pipelines in
  `bench/pipelines/reference/`, not catalog-wide CUDA coverage.** Host-side
  optimization is first-class; a CUDA path is built or kept only where it is
  measured faster. Catalog-wide functional coverage and byte-exact output are
  unchanged.
- **B0146 measured that baseline and it is bad: `pdg` is slower than stock PDAL
  on five of six reference workflows (0.903899x--1.004472x) and never
  meaningfully faster, with byte-exact output throughout.** Not one qualified
  route fires; every workflow runs `pdal_standard_host`. D0209 identifies three
  independent admission blockers, all engine policy: any writer `extra_dims`,
  LAZ output, and two neighborhood consumers in one graph. The next slices
  next slices remove those, because a faster kernel that never runs is worth
  nothing.
- **B0147 finds the root cause is one rule, not three blockers.** A
  `readers.las`/`writers.las` stage is native only with no options beyond
  `type`, `filename`, `tag`, `inputs` (`src/plan/Pipeline.cpp:2558`), and
  `lasHeaderFacts` refuses runtime facts unless both endpoints are native, so
  **any** writer option — `a_srs`, `scale_x`, `offset_x`, `forward`,
  `minor_version`, `dataformat_id`, `extra_dims`, `compression` — drops the
  whole pipeline to `pdal_standard_host`. The accelerated envelope is
  option-free uncompressed LAS on both ends. D0210 corrects D0209: writer
  options are policy, but LAZ is a genuine capability gap —
  `src/io/las/Las.cpp:322` throws "compressed LAZ records require the chunk
  decoder" and no codec exists. D0204's I/O-first ranking is therefore now
  measured rather than extrapolated.
- **B0148 supersedes that as the primary cause and it is much better news.** On
  clean unmodified code, `readers.las -> filters.normal(knn=8) -> writers.las`
  over 1M points is 4.372143 s in pinned PDAL, 4.407116 s through the public
  `pdg pipeline`, and **1.181099 s through `pdg resident` — 3.70x — with all
  three outputs byte-identical.** The acceleration is already built, exact and
  sanitizer-clean; the public command simply does not route to it, because
  automatic selection matches a whitelist of exact graph shapes while
  `pdg resident` runs anything placement can prove. An `a_srs` admission
  prototype confirmed B0147's mechanism but produced no user-visible speedup
  (0.9936x/0.9917x/1.0014x at 250K/1M/4M) and was fully reverted.
- **B0149 prototyped generalized selection and reverted it.** Relaxing all
  three automatic-admission gates
  (`src/cli/ResidentPipeline.cpp:2691`, `:2747`, `:3232`) made the public
  `pdg pipeline` **3.62x** pinned PDAL on an option-free uncompressed LAS
  `filters.normal` pipeline (1.218549 s versus 4.410862 s), byte-exact, with
  the full unit binary and both selection process gates passing. But on the six
  reference workflows it moved only one to three percent and still won nothing,
  because every reference workflow is LAZ and B0147 showed LAZ blocks
  `lasHeaderFacts` outright. It was reverted: it does not clear D0208's
  reference-baseline gate, and retaining it would extend automatic selection
  far beyond what the exactness matrices prove. D0098 is still the last new stage
  port; B0050 through B0146 are
  measured optimizations, calibration, composition, admission, or
  stopping work on already-implemented paths. Nothing from this branch has been merged.
  This documentation checkpoint is the intended clean stopping point; verify
  `git status` before resuming.
- No build, test, sanitizer, profiler, or benchmark process is left running.
  The D0088-D0098 RasterGrid/PMF/HAG slices and their records are committed.
- Suites at the B0100 checkpoint (not rerun as an aggregate for B0050 or
  B0101):
  physical SM-89 CUDA Release passes 669 tests plus
  nine documented skips in 678 registrations; Host ASan/UBSan passes 438 tests
  plus one opt-in corpus skip in 439 registrations. The SMRF, PMF, CSF, ELM,
  skewness-balancing, and HAG host/CUDA process matrices are exact. Device-source,
  full-device, host-tiled, bounded, primitive ambiguity-rejection, and adjacent
  PMF allocation-reuse lanes plus the HAG masked-index/resident lanes are clean
  under memcheck, initcheck, racecheck, and synccheck. At that checkpoint the
  frozen placement audit was 112/112 at winner accuracy 1.000 over 29 stage
  models. B0096 advances the current audit to 149/149 over 34 models; all 155
  unique raw calibration-report references verify.
- B0101's focused shared-radius planner unit, exact direct process gate, two
  existing radial-density CUDA differentials, memcheck, and racecheck pass.
- B0102's eight focused ordinal device/process gates and all four Compute
  Sanitizer tools pass; its timing rows are diagnostic-only negatives.
- B0103's five focused locate device/process gates and all four Compute
  Sanitizer tools pass; its 1M/4M timing rows are diagnostic-only negatives.
- B0104's information device property, forced metadata process differential,
  and all four Compute Sanitizer tools pass; its 1M/4M rows are
  diagnostic-only negatives.
- B0105's 12 focused count-four HAG/index/lifecycle tests and all four Compute
  Sanitizer lanes pass; its five-pair named fixture is exact at 1.562755x.
- B0125's focused formats 3/6/7/8 direct radius-outlier process, strict-radius
  boundary, planner/rewrite and resident process gates, host ASan/UBSan, and
  all four Compute Sanitizer tools pass. Its named 1M direct route is exact at
  8.107334x pinned PDAL.
- B0127's focused placement, planner/rewrite, runner-contract, and exact
  physical automatic process gates pass. Host ASan/UBSan passes 121 focused
  tests; memcheck, initcheck, racecheck, and synccheck are clean. The current
  placement audit is 152/152 over 35 models. Exact option-free gates reach
  4.717949x/15.855278x/45.178325x pinned PDAL at 250K/1M/4M.
- B0129's strict mapped-source/global-order direct composition is committed at
  `87f14b22f`. Its focused Release/Host Debug, runner, physical process,
  ASan/UBSan, and all four Compute Sanitizer gates pass. The clean exact 1M
  five-pair median is 0.425794757 seconds versus 1.229485169 seconds pinned
  PDAL, or 2.887507x, and is 61.235805% below B0128's forced hybrid. The fresh
  profile records 13 launches/0.260192 milliseconds. Publication is a
  7.519276% full-elimination ceiling, while B0108 already rejects the apparent
  reusable startup shortcut below 5%; the endpoint is sufficiently optimized,
  explicit, and unselected.
- B0130 changes no product code. On the exact nonidentity comparator-unique 1M
  Z graph, five clean pairs measure 1.562019960 seconds pinned PDAL,
  1.117247334 seconds forced CUDA, and 0.903809508 seconds for the exact
  same-binary host wrapper. The fresh CUDA profile reproduces the artifact but
  its 13 launches total only 0.271040 milliseconds. Directional symbol timing
  locates the opportunity in key gather, CUDA setup/copies, and the ordinary
  source/publication boundary. B0131 may reuse B0129's strict direct boundary;
  it must beat the faster host control by at least 5%.
- B0131 is committed at `da2e7323f`. Its strict explicit mapped-source,
  exact CUDA Z-order, and permutation-aware canonical publisher measure
  0.319400289 seconds versus 1.054076074 seconds pinned PDAL (3.300173x) and
  0.699280617 seconds for the same current binary's exact host control
  (54.324447% lower). The fresh exact-output profile has 13 launches totaling
  0.256640 milliseconds/0.080351% of candidate wall. Publication is only a
  7.839807% full-elimination ceiling and the whole manager only 11.569204%; no
  clear reusable 5--10% surface remains. The endpoint is sufficiently
  optimized, explicit, and unselected.
- B0132 changes no product code. Five current-binary exact pairs qualify the
  existing explicit `hag_delaunay(count=3)` mapped-source/direct-output route
  at 0.365213991 seconds versus 0.805184970 seconds pinned PDAL, or 2.204694x.
  The fresh profile reproduces the artifact and records 20 launches totaling
  2.421952 milliseconds/0.663160% of wall. This confirms B0113's stopping
  decision; the route remains explicit and unselected.
- B0133 changes no product code. Five exact pairs put pinned PDAL
  `hag_nn(count=5)` at 1.011225052 seconds, and an output-shaped control at
  0.289291154 seconds leaves 0.721933898 seconds/71.392011% of stock wall in
  the stage. The exact-output CPU phase capture attributes only 8.419990% of
  filter time to index construction and 89.906906% to the post-index query/
  projection path. B0134 may cheaply reuse the existing strict shared-index
  direct envelope; count five remains host-delegated, unqualified, and
  unselected until that exact current-binary gate wins by at least 5--10%.
- B0134 extends the existing exact shared-index HAG-NN machinery and strict
  explicit mapped-source/direct-binary64 boundary to count five. The 92-case
  host/CUDA matrices, 27 focused tests, direct process gate, host sanitizers,
  and all four Compute Sanitizer tools pass. A dirty-snapshot one-pair exact
  direction is 0.359718221 seconds direct versus 1.042808916 seconds pinned
  PDAL, or 2.898961x, with one index and zero spills. This is not a performance
  qualification; count six stays host-owned and no selector/model is added.
- B0135 current-clean qualifies that explicit count-five route at 0.343845512
  seconds versus 1.007641329 seconds pinned PDAL, or 2.930506x. The exact
  profile records 20 launches/4.391810 milliseconds, only 1.277263% of wall;
  publication is a 4.352942% full-elimination ceiling. B0108 already rejected
  the attributed placement shortcut, and no independent reusable manager
  component clears 5--10%, so the endpoint is sufficiently optimized,
  explicit, and unselected.
- B0136 changes no product code. Five exact pairs put pinned PDAL count six at
  1.019913125 seconds and the output-shaped control at 0.296473777 seconds,
  leaving 0.723439348 seconds/70.931468% in the stage. The exact-output CPU
  phase capture attributes only 8.228003% of filter time to index construction
  and 90.010190% to the post-index query/projection tail. This clears a cheap
  reuse prototype but adds no native coverage or selection.
- B0137 extends only the strict explicit HAG-NN envelope to count six using the
  existing mapped source, planner-owned masked 2D query, ordered projection,
  exact repair, resident bridge, and one-binary64 publisher. The 110-case
  host/CUDA matrices, full 562-pass Release unit binary plus two optional
  corpus skips, strict direct process gate, focused host ASan/UBSan, and all
  four Compute Sanitizer tools pass. One exact dirty-snapshot pair is
  1.025019333 seconds pinned PDAL versus 0.355610480 seconds direct, or
  2.882422x, with one index and zero spills. This retains the implementation
  direction but is not a performance qualification; no selector/model is
  added.
- B0138 current-clean qualifies that named explicit count-six route at
  0.344893464 seconds versus 1.023952521 seconds pinned PDAL, or 2.968895x.
  The exact profile records 20 launches/5.324470 milliseconds, only 1.543801%
  of wall; the HAG projection is 0.104670 milliseconds and publication is a
  4.296978% full-elimination ceiling. The full manager would need a 21.700089%
  reduction merely to save 5% wall, and no independent reusable component at
  that scale is identified. Count six is performance-qualified only for the
  named 1,000,002-point SM89 fixture, sufficiently optimized, explicit, and
  unselected. The paired report is
  `build/benchmarks/b0138-hag-nn-count6-direct-current-clean-1m.json`; the
  complete command/input/binary/output/profile manifest is
  `build/benchmarks/b0138-hag-nn-count6-direct-profile-manifest.json`
  (11,887 bytes/SHA-256
  `1d0ca83cc719ade40f7feb886f729a6f75bc1b3f7471afd0129e92fac4097f27`).
- B0139 changes no product code. Five exact pairs put pinned PDAL count seven
  at 1.050776026 seconds and the fresh output-shaped control at 0.292334145
  seconds, leaving 0.758441881 seconds/72.179214% in the stage. The reused
  offset-grid fixture needs no regeneration: its enumerated seventh/eighth
  candidate separation is 0.2 interior and 1.0 at both X extremes, and the
  count-seven artifact differs from the count-six artifact. The exact-output
  CPU phase capture attributes only 9.682102% of filter time to index
  construction and 88.392121% to the post-index query/projection tail. This
  clears a cheap reuse prototype but adds no native coverage or selection.
  The paired reports are
  `build/benchmarks/b0139-hag-nn-count7-current-clean-1m.json` and
  `build/benchmarks/b0139-hag-nn-output-control-current-clean-1m.json`; the
  complete manifest is
  `build/benchmarks/b0139-hag-nn-count7-measurement-profile-manifest.json`
  (9,513 bytes/SHA-256
  `846a142bafe25921232dfff03766cad82ffdd2572249766b4591a7d70516227d`).
- B0140 extends only the strict explicit HAG-NN envelope to count seven using
  the existing mapped source, planner-owned masked 2D query, ordered
  projection, exact repair, resident bridge, and one-binary64 publisher. The
  device projection was already count-generic, so this raises seven bounds and
  adds proofs rather than a kernel family. The 128-case host/120-case CUDA
  matrices, full 573-pass Release unit binary plus two optional corpus skips,
  strict direct process gate, focused host ASan/UBSan, and all four Compute
  Sanitizer tools pass. Every native-required count-seven fixture was verified
  tie-free in binary64 first; one arithmetic ground was replaced after the
  exactness guard correctly declined its bit-identical squared distance. One
  exact dirty-snapshot pair is 1.044514334 seconds pinned PDAL versus
  0.361863314 seconds direct, or 2.886489x, with one index, one
  25,000,050-byte upload, and zero spill/fallback boundaries. This retains the
  implementation direction but is not a performance qualification; no
  selector/model is added.
- B0141 current-clean qualifies that named explicit count-seven route at
  0.355388678 seconds versus 1.051637191 seconds pinned PDAL, or 2.959118x.
  The exact profile records 20 launches/6.421888 milliseconds, only 1.807004%
  of wall; the HAG projection is 0.120224 milliseconds and publication a
  5.034629% full-elimination ceiling that cannot actually be eliminated. The
  manager would need a 17.130% reduction to save 5% wall. The largest
  interval is the 50.412811% validation/placement/preflight startup that B0054
  measured as necessary and B0108 prototyped away for only 3.636364%; that
  startup is roughly fixed in absolute terms, so it dominates any
  sub-half-second endpoint and is a shared limiter rather than a count-seven
  opportunity. Count seven is performance-qualified only for the named
  1,000,002-point SM89 fixture, sufficiently optimized, explicit, and
  unselected. The paired report is
  `build/benchmarks/b0141-hag-nn-count7-direct-current-clean-1m.json`; the
  complete manifest is
  `build/benchmarks/b0141-hag-nn-count7-direct-profile-manifest.json`
  (10,584 bytes/SHA-256
  `fa99292b19fbac91e287e0a49c58578dd97aec4095d6537915821f2ea9b153cf`).
- B0142 changes no product code. Five exact pairs at each of counts 8, 12, 16,
  24, 32, 48, and 64 put pinned PDAL at 1.075645517 through 2.057952491
  seconds; against a fresh 0.290153739-second control the stage surface rises
  monotonically from 73.0252% to 85.9009% of stock wall. Cost is strongly
  sub-linear: 9.142857x more neighbours costs only 2.3241x more stock stage
  time and stock per-neighbour cost falls 3.93x. The retained fixture keeps the
  k/(k+1) boundary distinct for every k in 1..64 at a minimum separation of
  0.2, the device projection is already count-generic, the preflight already
  budgets `3 + count * (4 + 8)` bytes per point, and the region already rejects
  `maximumNeighbors > 64`. D0203 therefore closes the remaining range as one
  parameterized envelope instead of roughly 57 further per-count triads. The
  manifest is
  `build/benchmarks/b0142-hag-nn-count-range-measurement-manifest.json`.
- D0204 changes no product code and retires no stage or route. It re-reads the
  benchmark corpus and reprioritizes: native LAZ/COPC I/O, then cross-stage
  residency and product reuse, then fallback-frequency reduction, then
  automatic selection of already-qualified routes, then remaining per-stage
  kernels as an explicit catalog-coverage obligation. It adds an admission rule
  — no new standalone per-stage kernel without a same-machine profile showing
  that stage dominating a real end-to-end pipeline — and retires per-value
  proof ladders in favour of parameterized envelopes.
  Two framings were corrected against the record rather than inherited.
  "Kernels are noise" is false as a generalization: shared kernel work produced
  the largest wins here (D0074's 22M gather 32.2 s -> 0.49 s; the shared gather
  is still ~22% of the qualified 4M nndistance endpoint; B0064 took a further
  9.242103% off it). It is *standalone per-stage* kernels that have never paid.
  And the ~50% startup share is a sub-second-endpoint artifact of a roughly
  fixed 0.176-second cost, already closed by B0054/B0108. The conclusions hold;
  the reasons changed.
- B0143 closes the exact HAG-NN envelope at 64, the shared index's own
  `maximumNeighbors` cap, in one parameterized slice instead of 57 per-count
  triads. Counts one through seven keep explicit cases; 8/16/32/64 are
  generated over count. The matrices grow to 200 host/192 CUDA cases, the
  Release unit binary passes 577 tests plus two optional corpus skips, host
  ASan/UBSan passes 386, and all four Compute Sanitizer tools are clean. Every
  native-required generated fixture was proved tie-free in binary64 first (44
  verified tie-free, 12 tie-proof cases verified to genuinely tie). A wider
  79-ground fixture was added after the direct gate failed honestly at count
  16: the shared 7x3 fixture has only seven grounds, so wide counts correctly
  fell back for insufficient ground. One exact dirty-snapshot count-16 pair is
  1.224073288 seconds pinned PDAL versus 0.366412903 seconds direct, or
  3.340694x — the highest HAG-NN result so far, confirming B0142's prediction
  that the opportunity grows with count. Not a performance qualification; no
  selector/model is added.
- B0144 current-clean qualifies representative counts rather than every count.
  Count 16 measures 0.397297088 seconds versus 1.224026575 seconds pinned PDAL
  (3.080885x, 67.541792% lower wall) and count 64 measures 0.470088827 versus
  2.056411590 seconds (4.374517x, 77.140334% lower), both byte-exact and both
  reproducing B0142's pinned artifacts. Speedup rises with count — 2.518270x at
  one, 2.968895x at seven, 3.080885x at 16, 4.374517x at 64 — confirming
  B0142's prediction. **This endpoint does not stop.** The fresh profiles
  record 39 launches; all kernels are 5.297114% of wall at count 16 and
  17.937498% at count 64, with the shared `bvhKnnGatherKernel` alone at
  5.108745% and 17.576763%. Startup falls to 28.2567% purely because the wall
  grew, exactly as D0204 predicted. For the first time on this family a
  reusable component clears the 5--10% gate by a wide margin, and it is the
  *shared* gather every neighborhood consumer uses.
- B0145 changes no product code and names that gather's limiter. Backend choice
  is rejected as the lever: five alternating same-binary count-64 runs measure
  adaptive 0.476555539 s, forced BVH 0.477948684 s (+0.2923%, noise), and
  forced uniform grid 0.832525226 s (+74.6964%), all byte-identical, so the
  adaptive selection is already optimal. `bvhKnnGatherKernel` is 82.63 ms over
  15,626 blocks of 64 threads at 48 registers/thread, with 67.00% compute
  versus 49.95% memory throughput — compute/latency-bound, not
  bandwidth-bound. Achieved occupancy is 72.72% against an 83.33% ceiling and
  the block limits are registers 20, SM 24, warps 24, shared memory 32, so
  **register pressure is the binding constraint**. Shared memory is unused.
- B0106's exact ordinary planner-resident prototype is fully reverted after
  measuring 0.675 seconds versus its 0.657-second same-binary hybrid control.
- B0107's strict one-binary64 direct publisher passes focused CUDA/host tests,
  ASan/UBSan, all four Compute Sanitizer tools, and the physical exact process
  gate. Its directional 0.98-second PDAL, 0.64-second same-binary hybrid, and
  0.59-second direct trio retains the reusable boundary at 7.8125% below the
  control; no model, automatic selector, or new qualification follows.
- B0108's disposable explicit-force placement shortcut is fully reverted. Its
  0.53-second direction against the 0.55-second same-binary control saves only
  3.636%, and stats show startup moving between placement subphases rather than
  disappearing. Only unsupported-descriptor fallback hardening remains; the
  focused physical process gate passes after the revert.
- B0109 changes no code. Its exact 1M NNDistance/Extra-Bytes graph measures
  2.01 seconds pinned PDAL, 0.66 seconds CUDA hybrid, and 0.39 seconds for the
  output-shaped control. The fresh exact-output profile totals 25.767520
  milliseconds of kernels, so B0110 may test the measured resident/publication
  boundary without touching the shared kNN kernel.
- B0110 retains that exact explicit composition. The clean same-binary pair is
  0.71 seconds hybrid versus 0.57 seconds direct (19.718% lower), the physical
  process and host sanitizer gates pass, and all four Compute Sanitizer tools
  are clean. The profile has only 25.856256 milliseconds of kernels; B0111 may
  test the still-unused mapped LAS resident source.
- B0111 retains that mapped source for the same explicit endpoint. The clean
  source-disabled/source-enabled pair is 0.58/0.38 seconds (34.483% lower),
  with exact pinned output, record-summary-configured Morton-BVH, no host XYZ
  mirror, one index, zero spill, physical/host sanitizer proof, and all four
  Compute Sanitizer lanes clean. The fresh profile has only 26.086656
  milliseconds of kernels, and B0108 already rejected the remaining apparent
  startup/placement surface below 5%; endpoint tuning stops.
- B0112 changes no code. Its clean exact HAG Delaunay count-three pair is
  0.810889333 seconds pinned PDAL versus 0.607849511 seconds current hybrid
  (1.334030x). The fresh profile totals only 2.342688 milliseconds/0.385406%
  of wall, while an output-shaped same-binary control takes 0.40 seconds. The
  measured next hypothesis is source/output reuse, not another kernel.
- B0113 retains that strict mapped-source/direct-output composition. The clean
  source-disabled/source-enabled pair is 0.55/0.41 seconds (25.455% lower),
  with exact pinned output, no host XYZ mirror, one masked 2D index, zero spill,
  positive tie/incomplete host repair, atomic publication, focused host/GPU
  tests, and all four Compute Sanitizer lanes clean. The fresh profile totals
  2.411420 milliseconds/0.588151% of wall; publication is 15.792846
  milliseconds and B0108 already rejects the apparent placement shortcut
  below 5%. The endpoint is sufficiently optimized for now.
- B0114 retains the same mapped source for B0107's strict HAG-NN count-four
  direct endpoint. The clean source-disabled/source-enabled pair is 0.56/0.37
  seconds (33.929% lower), with exact pinned output, no host XYZ mirror, one
  masked 2D index, zero spill, positive tie/incomplete repair, atomic
  publication, focused host/GPU tests, and all four Compute Sanitizer lanes
  clean. The fresh profile totals 3.523550 milliseconds/0.952311% of wall;
  publication is 12.601495 milliseconds. Endpoint tuning stops.
- B0115 changes no product code. Its exact affine transformation followed by
  pointwise assignment takes 0.35 seconds in pinned PDAL and 0.60 seconds in
  the forced descriptor-fused CUDA hybrid region, or 0.583x PDAL throughput.
  The fresh exact-output profile records eight tiles/four launches per tile,
  but all 32 kernels total only 0.849310 milliseconds/0.141552% of wall. A
  device-fusion rewrite cannot meet the 5--10% process gate, so this endpoint
  is sufficiently optimized and remains host-selected.
- B0116 changes no product code. Five alternating exact 1M pairs measure
  4.374096348 seconds pinned PDAL versus 0.645599567 seconds forced per-stage
  CUDA, or 6.775247x. The fresh profile totals 28.478630 milliseconds/
  4.411191% of median; duplicate labeling itself is only 0.010810
  milliseconds. The exact hybrid chain is performance-qualified, but its label
  stage restores `Duplicate` to the host PointView; this is not resident reuse.
  Actual resident placement still declines with `missing_calibration_model`;
  automatic selection remains false.
- B0117's disposable one-line model anchor is fully reverted. It proves exact
  actual-resident execution with one selected region/one index, but the warmed
  process takes 0.70 seconds, 8.43% slower than B0116's forced-hybrid median.
  Observed 35/11 MB H2D/D2H also misses predicted 26/10 MB. No ladder, model,
  selector, or code is retained; the original engine hash is restored.
- B0118 through B0120 close Morton/assignment, Morton/head, and exact stats
  opportunities by measurement. B0121 then retains the strict direct count-two
  HAG source/publisher composition on clean commit `0deba2034`: five exact
  pairs reach 2.606656x pinned PDAL and improve 39.730794% over the clean
  forced-hybrid median. The fresh profile totals 2.552608 milliseconds/
  0.698944% of wall; all four Compute Sanitizer tools are clean. Count two is
  performance-qualified only for the named explicit 1M routes and remains
  unselected.
- B0122 extends the same strict route to count three after a current-clean
  measure-first gate. Five exact direct pairs reach 2.655409x pinned PDAL and
  reduce 41.094643% from hybrid median. Native and repaired count-three
  processes plus all four Compute Sanitizer tools are clean. The fresh profile
  totals 3.076512 milliseconds/0.841963% of wall; the endpoint is sufficiently
  optimized, explicit, and unselected.
- B0123 closes the count-one-through-four direct-publication gap. Five exact
  forced-hybrid pairs reach 1.442969x pinned PDAL; five clean direct pairs
  reach 2.518270x and reduce wall 42.773438% from hybrid. Count-one native,
  tie/incomplete repair, source rejection, option-bearing producer rejection,
  atomic output, one index, and all four Compute Sanitizer tools are clean.
  The fresh profile totals 2.067488 milliseconds/0.573503% of wall;
  publication is 4.553248%, so the endpoint is sufficiently optimized,
  explicit, and unselected.
- B0124 qualifies the named exact 1M standalone statistical-outlier lane at
  6.521497x pinned PDAL in forced hybrid and 10.739669x through the strict
  mapped source/direct Classification publisher. Direct wall is 39.020273%
  below hybrid. The focused physical process, host ASan/UBSan, and all four
  Compute Sanitizer lanes pass. The fresh profile leaves only the required
  shared kNN gather above 5% of wall; all other kernels, publication,
  hydration, row materialization, and residual wrapper surfaces are below the
  gate. The endpoint is sufficiently optimized, explicit, and unselected.
- P1.5 closed at D0071; all four recorded post-P1.5 roadmap items closed
  at D0077. Catalog work reached D0098. D0099 completes the requested
  end-to-end performance audit and priority rewrite. Do not begin another
  stage port until the active performance-first queue selects it from a
  complete-pipeline profile.
- B0080 adds exact automatic public admission at `b881fe11b`; the clean
  one-warmup/five-pair 1M option-free qualification is byte-exact at 8.156734x.
  Its focused host, runner-contract, and physical process gates pass. B0076's
  unchanged-executor profile limits perfect removal of one repeated gather to
  3.557802% of complete wall, so the endpoint is sufficiently optimized for
  now. No new aggregate suite count is claimed.
- B0081 cleanly revalidates the exact 1M radiusassign resident endpoint at a
  directional 2.752316x and profiles only 1.899456 milliseconds of device
  work. B0082 retains the exact direct-LAS source/output boundary after a
  50.726372% same-binary reduction and pinned format-7/format-3 physical proof.
  It remains opt-in, uncalibrated, and unqualified; no public selector changes.
- B0083 adds the clean 50K-through-16M direct-executor ladder and B0084
  integrates its separate bounded `radiusassign-direct` model. The physical
  format-7 path now proves executor and boundary provenance; format 3 proves
  boundary provenance without borrowing the format-7 timing row. Repeated
  qualification and automatic admission remain pending.
- B0085 qualifies the exact explicit direct-radiusassign endpoint at 5.588494x
  pinned PDAL and stops tuning it after the final profile finds no reusable
  5--10% surface. B0086 reopens the exact direct outlier/NNDistance composition
  because two gathers account for 52.001290 milliseconds. B0087 retains one
  planner-owned max-k rowset after an exact 11.315433% same-binary direction
  gate; the clean profile records one gather and 29.350784 milliseconds total.
  B0088's clean repeated gate qualifies that explicit outlier composition at
  21.389523x pinned PDAL. B0089 finds a 4M incomplete-mean host-repair cliff;
  B0090 repairs it in parallel from resident coordinates. B0091 then defaults
  the already-tested bounded NNDistance parallel repair and qualifies the exact
  4M route at 48.758796x. Its final profile leaves only the shared gather above
  a one-percent process ceiling, so endpoint tuning stops. B0092 adds stable
  repaired 8M/16M rows, a separate 50K--16M composition model, and exact
  option-free public admission. Five clean 1M pairs reach 20.300698x pinned
  PDAL. Outlier is automatically selected only inside that exact composition;
  standalone and broader shapes remain unchanged.
- B0093 changes no CUDA kernel or placement coefficient. It automatically
  admits only the calibrated exact `radiusassign-direct` shape, proves 50K
  below-floor plus 250K/1M device controls and fail-closed layout/option/
  source/preflight/execution negatives, and passes five clean option-free 1M
  pairs at 5.366005x pinned PDAL. B0085's unchanged-executor profile keeps
  endpoint tuning closed. Radiusassign is the sixteenth automatically selected
  filter, only inside this partial-GPU-native measured envelope.
- B0094 cleanly repeats the existing exact
  `approximatecoplanar(knn=8) -> ferry(Coplanar=>UserData)` composition at
  4.148518x pinned PDAL. This is repeated exact routing evidence, not a new
  performance qualification: it proves direct output plus one index but
  honestly reports calibration and boundary mismatch. Fresh events reproduce
  the 194 MB repair round trip; current stats do not identify coplanar repair
  rows/KD3 use, and B0069's profile hashes do not match the current endpoint.
  No automatic claim follows.
- B0095 adds fail-closed repair telemetry and a fresh exact profile on clean
  commit `f69fe3204`. The 1M route observes 2,145 ambiguous rows, zero
  incomplete rows, one KD3 use, and 97 MB transferred in each direction while
  reproducing B0094's output hash. Its 20 kernels total 21.403776 milliseconds,
  only 2.460417% of the 0.869924635-second unprofiled command interval. Exact
  KD3 repair remains dominant at 0.461688645 seconds, but B0070's 99.377286%
  repair-transfer reduction improved shell wall only 0.491159%. The endpoint
  is sufficiently optimized; no kernel work reopens. B0094's 4.148518x result
  remains older-binary routing evidence, not a current B0095 qualification.
  At the B0095 checkpoint the explicit route remained uncalibrated,
  performance-unqualified on the current binary, and not automatic pending
  B0096's paired ladder.
- B0096 pairs the current binary against pinned PDAL from 50K through 16M.
  The 50K host control is 0.484603x; every selected 250K--16M direct-output
  row is exact and wins from 2.476664x through 5.048595x, including 4.248119x
  at 1M. Commit `04a6803de` adds the isolated bounded
  `approximatecoplanar-direct-compose` model and strict measured-shape matcher;
  `cba52b2ca` additionally pins the measured LAS point format instead of
  accepting record width alone. The ordinary model remains unchanged.
- B0097 completes the cheap public gate on implementation commit `e36ac6bbd`.
  Required and genuinely option-free 1M processes are exact, while option,
  topology, both-threshold, layout/compression, cardinality, device, preflight,
  and execution-proof controls fall back exactly or publish no artifact. Five
  clean automatic pairs reach 4.230620x pinned PDAL. The fresh automatic
  capture records only 21.257320 milliseconds of kernels, 2.166754% of
  complete wall, so the endpoint remains sufficiently optimized. The exact
  B0096 250K--16M composition is now automatic; broader shapes are unchanged.
- ncu profiling runs unprivileged through the R610 capability grant
  (`/dev/nvidia-caps/nvidia-cap4324`; `RmProfilingAdminOnly=1` stays set,
  and sudo is never used for it).

## What changed after P1.5 (D0072-D0113)

Read these as the current operating rules; each has a full DECISIONS
entry.

- **D0072 — attach boundary.** The whole-view attach gather/publish loops
  are row-backed behind D0062's fail-closed offset-partition guard and
  report into `host_boundary_phase_seconds`.
  `PDG_DISABLE_NEIGHBORHOOD_ROW_BOUNDARY` forces the semantic field path;
  `PDG_REQUIRE_NEIGHBORHOOD_ROW_BOUNDARY` proves the direct path engaged.
  The loops were never the bottleneck (0.87 s of a 50.7 s lane) — that
  measurement is the entry's real content.
- **D0073 — kNN gather, dense regime.** Morton-ordered queries, a
  register worst-entry quick reject, contiguous sorted-order candidate
  loads, and a certified float prefilter around the consumer-Ada FP64
  pipe. `PDG_DISABLE_KNN_DISTANCE_PREFILTER` forces the exact path. kNN
  device indexes carry 36 B/point of sorted candidate copies behind the
  `knnCandidateArrays` config flag (planner estimate 112 B/point) and the
  kNN launchers fail closed without them.
- **D0074 — kNN gather, tail regime.** The device walk stops at a shell
  budget (default 32, `PDG_KNN_DEVICE_SHELL_BUDGET`); declared-incomplete
  rows are repaired exactly on the host by every consumer (eigen family
  pre-existing; NNDistance and statistical outlier per-row; LOF via the
  two-hop closure on `knnLofValues`'s `neighborNeighborStatus`). 22M
  gather 32.2 s -> 0.49 s. A one-shell-budget differential routes every
  row through the repairs and must stay bit-identical — keep it passing
  whenever a repair predicate changes.
- **D0075 — first catalog tranche.** normal, eigenvalues,
  covariancefeatures, and nndistance execute resident. The resident CLI's
  PDAL log leader is `pdal pipeline`: stage warnings are part of the
  byte-exact stderr contract, so never rename it back.
- **D0076 — fused JIT envelope.** The NVRTC generator specializes every
  admitted LAS record format (0-3, 6-8; format 8 at its native 38-byte
  stride). Keep the identity source minimal — helpers emit conditionally.
  `PDG_DEBUG_FUSED_JIT` dumps the NVRTC log and generated source on a
  failed compilation.
- **D0077 — executor honesty.** The attach machinery records every
  physical transfer as HostToDevice/DeviceToHost observation events, and
  all six then-existing shared-index models were re-measured against the
  planner-selected resident executor itself.
  `selected_device_calibration_matches_executor` is true for whole-view
  executions whose applied region calibrations are all
  resident-provenance models (`ResidentCalibratedModels` in
  `src/cli/ResidentPipeline.cpp`); `planner_resident_boundary_batch`
  still reports false, honestly, until it has a calibration of its own.
- **D0078-D0080 — new primitives.** estimaterank, optimalneighborhood,
  and neighborclassifier are native and resident. estimaterank and
  optimalneighborhood keep their final decomposition/entropy step on the
  host: Eigen's JacobiSVD carries no device annotations and `std::log`
  differs bitwise between glibc and CUDA, so host arithmetic *is* the
  oracle there. neighborclassifier's integer vote runs entirely on
  device.
- **D0081 — shared radius selection.** radiusassign uses planner-owned
  `radiusAny` over the grid or Morton BVH, then preserves exact upstream
  expression order and physical casts in a host assignment finale. Written
  non-XYZ columns are re-uploaded before a resident consumer. This is a
  measured hybrid stage, not a claim that assignment evaluation is wholly on
  device.
- **D0082 — index-free adjacent labeling.** label_duplicates compares only the
  immediate predecessor in current order; it never sorts. Its exact typed CUDA
  pass preserves binary64 comparison, the first `Duplicate` byte, empty-list
  behavior, and order. A resident region can retain the output without XYZ or
  an index and feed a later point program. Complete standalone CUDA loses at
  250K, 1M, and 22M (B0020), so the descriptor intentionally has no placement
  model and default execution stays on pinned PDAL.
- **D0083 — bounded SMRF grid.** The first P3 lane reproduces upstream's
  column-major minimum raster, ordered eight-nearest fill, progressive diamond
  morphology, net mask, provisional surface, gradient, and classification on
  host and CUDA. CUDA is capped at 4,096 cells and radius 64, runtime-checks
  the actual canvas, and has no automatic model after B0021's 0.599x result.
  Only a standalone upload-stage-spill region is admitted; an adjacent grid
  bridge is rejected until the workspace is planner-owned and composable.
- **D0084 — bounded PMF grid.** PMF now has an exact typed host mirror and a
  precise-FP64 CUDA lane for finite logical-double XYZ, unsigned-byte
  Classification, a global raster of at most 4,096 cells, radius at most 64,
  and at most 64 morphology passes. It deliberately reproduces upstream's
  distinct initial fractional-cell binning, later lookup formula, one-nearest
  empty-cell fill, strict threshold, return filtering, and class writes. B0022
  is exact but only 0.583x PDAL, so PMF also remains host-selected and admits
  only a standalone upload-stage-spill region.
- **D0085 — bounded serial-oracle CSF grid.** CSF now has an exact typed host
  mirror and a precise-FP64 CUDA lane for `smooth=false`, finite logical-double
  XYZ, unsigned-byte Classification, an at-least-2x2 cloth of at most 4,096
  cells, and at most 64 iterations. It preserves the pinned non-OpenMP source
  order for cloth construction, raster/fill, in-place constraints, collision,
  strict interpolation, return filtering, classes, and verbose diagnostics.
  OpenMP-enabled or smoothed builds fail closed to PDAL. B0023 is exact but
  only 0.289x PDAL because serial frame/raster work dominates, so CSF remains
  host-selected and admits only a standalone upload-stage-spill region.
- **D0086 — bounded stable-cell ELM grid.** ELM now has an exact typed host
  mirror and precise-FP64 CUDA lane for finite logical-double XYZ,
  unsigned-byte Classification, valid scalar options, and at most 4,096 cells.
  It preserves upstream's `floor(coordinate - minimum) / cell` binning,
  column-major cells, numeric-Z order, source-stable equal values including
  signed zero, strict threshold comparison, point order, and diagnostics. CUB
  scratch is queried from the active toolkit and preflighted at its exact
  overlapping-allocation peak. B0024 is exact but only 0.866x PDAL, so ELM is
  host-selected and admits only a standalone upload-stage-spill region.
- **D0087 — bounded unique-Z skewness ordering.** Skewness balancing now has
  an exact host recurrence plus a forced hybrid CUDA ordering substep. Native
  admission requires a nonempty whole view with physical binary64, finite,
  comparator-unique Z; equal values including signed-zero equivalents fail
  closed before the PointView changes. CUDA returns the exact permutation,
  then the complete record is reordered and the pinned sequential moment and
  sign-crossing recurrence runs on the host. B0025 is exact and 1.150x PDAL on
  its controlled 1M unique-Z fixture, but the data-dependent gate and host
  publication boundary support neither automatic selection nor resident or
  wholly device-native coverage.
- **D0088 — provisional tiled PMF RasterGrid.** The planner now declares a
  PMF-specific Grid frame and creates a region-owned, memory-driven product
  with deterministic column-major cores, clipped one-cell halos, two complete
  pinned-host backings, owner-only mosaics, and phase barriers. Named tests hit
  every edge/corner, multiple budgets, >4,096-cell fractional and morphology
  seams, region release, and ambiguous nearest-fill rejection before class
  publication. Default and broad experimental PMF execution delegate directly
  to upstream; only explicit required or planner-resident execution enters the
  reduced tiled envelope. B0026 recorded an exact but 0.026602x prototype and
  isolated its repeated all-points-per-tile raster build as the bottleneck.
- **D0089 — build-once exact PMF source raster.** PMF now constructs the source
  raster once per resident execution in selected source order, using only the
  product's two already-budgeted host backings. It preserves first-source
  binary64 minimum bits, proves identical-bit nearest ties, rejects
  distinct-bit ties before publication, publishes one `RasterBuild`
  generation, and closes the resident region even when required execution
  fails. B0027 measures 1.828354x pinned PDAL on the exact dense 65x65
  four-tile fixture. The allocation-free host void fill remains proportional
  to void cells times populated cells, so this does not establish sparse-frame
  scalability, automatic placement, adjacent Grid composition, or cross-stage
  reuse.
- **D0090 — fitting device-resident PMF phases.** PMF's Grid request now
  declares two full-frame device phase backings. The product permits exactly
  one raster publication/consumption and cannot allocate the device pair until
  that generation is pending, so distinct-bit nearest-tie rejection happens
  before allocation. When the complete pair fits, one raster H2D feeds every
  global morphology/filter phase with zero raster D2H; larger frames retain
  the exact tiled host-mosaic path. The existing tile scratch holds an exact
  column-major populated-source list when possible. B0028 records exact
  9.009503x dense and 4.764988x sparse/void controlled results. The host source
  build and worst-case quadratic nearest fill remain; no automatic model,
  adjacent bridge, or cross-stage reuse follows.
- **D0091 — exact controlled sparse PMF fill.** The second planner-owned host
  backing now carries a transient byte occupancy pyramid while the completed
  raster remains in the first backing. A literal scan handles at most 255
  populated cells; 256 or more use greedy descent plus a strict-bound exact
  proof that still observes every equal-distance leaf. Heterogeneous
  large-origin/fractional tests match a literal raw-bit scan, distinct-bit
  four-way ties fail before publication and can be retried on the same
  product, and finite inputs whose represented centers overflow fail closed.
  B0029 records 6.298x/5.701x pinned PDAL at 25 sources and 1.817x/1.802x at
  256 sources on 257x257/513x513 matrices. The observed hierarchy scaling is
  near-linear only for those named distributions; pathological layouts retain
  no proved subquadratic bound. Source construction is still host work and no
  automatic model, adjacent bridge, or cross-stage reuse follows.
- **D0092 — fitting device-native PMF source proof.** PMF now declares a
  16-byte-per-cell provisional device workspace that exactly aliases the two
  full-frame phase backings. CUDA preserves first-source minimum bits, compacts
  original populated cells, scans every compact source for each target with
  explicit binary64 arithmetic, and rejects equal-distance distinct-bit ties
  before device Classification materialization/H2D or mutation. Success
  promotes the same allocation into the phase pair and records zero raster
  H2D/D2H; failure discards it and permits retry. The wrapper catches that
  named proof failure and runs pinned upstream on a private copy of the
  untouched view, preserving Classification and diagnostic bytes. One byte
  below the proof boundary retains the exact D0091 host builder. B0030 records exact
  3.156x-38.950x controlled results and profiles
  the FP64 proof reduction as the limiter. The proof is still
  target-cells-times-sources work; automatic placement, arbitrary sparse-frame
  scalability, adjacent composition, and cross-stage reuse remain open.
- **D0093 — compatible adjacent PMF allocation reuse.** A contiguous PMF pair
  with the same cell/Grid contract and exact original JSON return selection can
  now remain in one resident region. The first wrapper creates the product and
  the final wrapper closes the region; every intermediate wrapper requires the
  existing product and rechecks the runtime frame. The product accepts a new
  raster generation only after the previous one is consumed and reuses the
  same promoted device phase allocation. Each stage still rebuilds its own
  source raster and publishes Classification between stages. Physical two- and
  three-stage differentials observe one `GridBuild`, one exact raster
  generation per PMF, zero raster transfers, and exact final output. Corrected
  symmetric-scope B0031 v4 measures 3.296628 ms pinned PDAL versus 3.170328 ms
  resident PDG (1.039838x) for the exact 65x65 pair; the earlier 4.590078x,
  1.056201x, and candidate-comparison-inclusive 1.016953x measurements are
  superseded. Different returns/cell frames, non-PMF Grid consumers, semantic
  raster reuse, cross-kind Grid/cloth composition, and automatic placement
  remain rejected. Grid-to-non-Grid consumers may compose only through the
  explicit D0095 spill/upload boundary; they never share the PMF product.
- **D0094 — exact count-one HAG.** `filters.hag_nn,count=1` now consumes a
  planner-owned masked 2D kNN index on finite binary64 XY. Both UniformGrid and
  MortonBvh retain an extra candidate to prove unique nearest-ground results;
  ties and bounded-grid incompleteness repair with the pinned host filter before
  publication. Nonfinite XY declines before index allocation, no-ground
  diagnostics remain upstream-owned, and repaired HAG columns can feed an
  adjacent point-program bridge. HAG↔3D-neighborhood execution observes two
  physical index builds. The 20-case host/11-case CUDA matrices, five repeated
  CUDA runs, full aggregate suites, and all four focused Compute Sanitizer tools
  are clean. B0032 is exact at 1.512959x on its controlled unique-nearest 1M
  fixture, but ordinary data can trip the fallback, so no automatic model is
  admitted. `hag_dem`, `hag_delaunay`, and count-greater-than-one native
  interpolation remain open.
- **D0095 — exact count-two HAG.** `filters.hag_nn,count=2` retains the same
  planner-owned masked 2D index while preserving pinned ordered
  inverse-squared-distance arithmetic, strict maximum-distance comparison,
  inclusive no-extrapolation bounds, and same-XY behavior. A third candidate
  proves the selected pair; ties/incomplete searches, nonfinite Z, and the
  historical one-ground NaN path repair through pinned upstream before
  publication. PMF -> HAG2 -> PMF now uses three product regions and six
  explicit spill/upload boundaries, with a physical exact differential, two
  grid builds, and one 2D index build. Same-kind DAG siblings still group;
  branched selected boundaries fail closed. The 32-case host/24-case CUDA
  matrices, five repeated CUDA runs, full aggregates, and all four Compute
  Sanitizer tools are clean. B0033 is exact at 1.493330x on its controlled
  1,000,002-point fixture. At that checkpoint counts three and greater stayed
  upstream-owned; D0097 below closes only the count-three HAG-NN case.
- **D0096 — exact count-three HAG Delaunay.** Explicit
  `filters.hag_delaunay,count=3` now uses the planner-owned masked 2D index and
  reproduces the bounded three-point Delaunator seed/circumradius/winding
  order plus PDAL barycentric arithmetic. Positive infinity remains the
  outside-triangle sentinel while negative-infinity overflow remains a valid
  interpolation. Candidate ties/incompleteness, insufficient grounds,
  nonfinite/runtime-incompatible data, default/count-four-or-wider
  triangulations, and unsupported options stay pinned-host owned. Native and
  repaired HAG columns feed adjacent assignments, including tied-fourth and
  incomplete-grid repairs. The 22-case host/15-case CUDA matrices, five CUDA
  repeats, 433-registration Host ASan/UBSan suite, 653-registration physical
  CUDA suite, and all four focused Compute Sanitizer tools are clean. B0034 is
  exact at 1.314103x on its controlled 1,000,002-point fixture. No automatic
  model, wider-count, full-HAG, or P3 claim follows.
- **D0097 — exact count-three HAG NN.** Explicit
  `filters.hag_nn,count=3` extends the existing planner-owned masked 2D index
  lane without changing its candidate-search or host-repair architecture.
  The exact kernel preserves the pinned three-term inverse-squared-distance
  accumulation, strict cutoff and inclusive bounds, signed-zero/subnormal
  behavior, and finite-coordinate squared-distance overflow. Internal or
  fourth-candidate ties, incomplete search, nonfinite Z, and fewer than three
  grounds repair through the original host filter before publication; native
  and repaired HAG columns feed adjacent assignments. The 54-case host/46-case
  CUDA matrices, five CUDA repeats, 436-registration Host ASan/UBSan suite,
  667-registration physical CUDA suite, and all four 12-test Compute Sanitizer
  lanes are clean with their documented skips. B0035 is exact at 1.482696x on
  its controlled 1,000,002-point fixture. No automatic model, count-four,
  full-HAG, or P3 claim follows.
- **D0098 — exact count-four HAG NN and safe stop.** Explicit
  `filters.hag_nn,count=4` reuses the same planner-owned masked 2D query and
  exact host-repair bridge. Both shared-index backends prove a distinct fifth
  candidate and fourth/fifth boundary repair; true binary64 squared-distance
  overflow is proved only on UniformGrid because MortonBvh rejects that
  coordinate span before query. The 74-case host/66-case CUDA matrices, five
  CUDA repeats, 439-registration Host ASan/UBSan suite, 678-registration
  physical CUDA suite, and all four 12-test Compute Sanitizer lanes are clean
  with their documented skips. B0036 is byte-exact at 1.587001x on its
  controlled forced 1,000,002-point fixture. The lane is functionally
  supported and GPU-native inside its proof envelope. D0099 classifies the
  dirty-snapshot timing as complete-process diagnostic evidence rather than a
  performance qualification; it is not automatically selected. The next task
  was the implemented-stage end-to-end audit and priority rewrite, not count
  five.
- **D0099 — pipeline speed becomes the active work order.**
  `docs/stage-coverage.md` now audits every implemented envelope under four
  independent labels: functionally supported, GPU-native,
  performance-qualified, and automatically selected. “Unmeasured” means no
  accepted end-to-end record; force/require/resident execution is not
  automatic. At that checkpoint the seven automatically selected filters were
  the five fused point-program stages, Morton ordering, and expression
  statistics. The audit
  retains complete-process terrain diagnostics: SMRF 0.599x, bounded PMF
  0.583x, adjacent PMF 1.039838x, and CSF 0.289x; their dirty snapshots are not
  performance qualifications. The plan now measures and profiles first,
  favors reusable fusion/residency/I/O over new per-stage kernels, and resumes
  catalog ports only after higher-value measured surfaces are exhausted or
  when coverage is itself the explicit reason.
- **D0106 — the measured resident LOF endpoint is selected automatically.**
  Commit `62016556b` separates refusal from committed execution and admits
  only the exact explicit `LAS -> LOF(minpts=10) -> UserData assign -> default
  LAS` public shape. Unsupported graphs/options, input/output state,
  device/profile/placement, and preflight failures return to the unchanged
  selector without output or diagnostics; execution errors after commitment
  do not retry. The host CLI matrix and 281.94-second physical v3 gate are
  exact. Clean option-free B0043 measures 3.723975123 seconds versus
  35.738760555 seconds pinned PDAL, or 9.596939x. This makes LOF the eighth
  automatically selected filter without widening any other resident or writer
  envelope.
- **D0107 — B0044 chooses nndistance from the reusable endpoint sweep.**
  Clean one-shot 4M direct-resident probes are exact at 4.932x pinned PDAL for
  coplanar, 4.874x for normal, and 5.275x for nndistance. Only nndistance
  advances. The then-zero repair report was based on absent NND repair
  telemetry and is superseded by D0112. Nsight Compute measures
  `knnGatherKernel` at 138.57 milliseconds, while resident host upload/spill
  phases total 0.125 seconds. The remaining roughly 3.10 seconds is host-side
  process work, so B0045 qualifies/integrates the existing path and does not
  start a kernel optimization.
- **D0108 — the exact nndistance endpoint is automatically selected.** Clean
  explicit-direct B0045 is 5.302170x pinned PDAL; commit `fba6b16ba` adds only
  the explicit `k=10`/exact `UserData` assign/default-LAS shape to the
  no-side-effect selector. The host fallback matrix and 236.60-second expanded
  v2 physical differential are exact. Five clean option-free samples remain
  exact at 3.390338813 seconds versus 17.726674770 seconds pinned PDAL, or
  5.228585x. Nndistance becomes the ninth automatically selected filter; no
  other option or B0044 shape is promoted.
- **D0109 — B0046 localizes the remaining wall to manager execution.** Commit
  `455bced46` adds stats-only command phases. Clean exact 4M nndistance spends
  3.005455534 of 3.235190260 seconds in rewritten manager execution;
  validation is 0.168271752 seconds and canonical publication only
  0.051060884 seconds. After known boundary and device-kernel work,
  2.743655928 seconds remains inside reader/table/index/wrapper execution.
  LOF shows the same manager-level dominance, but its known exact host repair
  accounts for 2.287512704 seconds. The writer is rejected as the next target.
- **D0110 — B0047 rejects reader/table work as the dominant remainder.** Clean
  exact 4M nndistance splits its 3.027606651-second manager wall into 0.000559325
  seconds graph/prepare, 0.502592322 reader/table, 2.524453274 resident
  upload-through-spill, and 0.000001730 post-spill control. Known boundary and
  kernel work leave 2.256203803 seconds inside the resident interval. LOF's
  matching 2.901305423-second resident interval is 79.09% exact repair. B0048
  must split reusable index/filter/bridge work; no reader optimization follows.
- **D0111 — B0048 localizes the remaining NND wall to its broad query call.**
  Clean exact 4M nndistance spends 0.121900349 seconds in index configuration,
  0.008554448 in index-build call wall, 2.233622069 in query/projection,
  0.035821079 in the adjacent assignment bridge, and 0.106845283 unattributed
  inside its 2.506743228-second resident interval. Its zero-repair inference
  is superseded by D0112 because NND did not yet publish repair telemetry. LOF
  has the same small index/bridge scale; exact repair explains
  2.279198769 of its 2.597038882-second query span. B0049 must split NND's
  transfer/wait/status/publication call boundaries before any optimization.
- **D0112 — B0049 identifies one full host-index repair as NND's dominant
  cost.** Commit `20d9ffabc` adds stats-only nested call spans and complete NND
  repair accounting. The final exact 4M broad query is 2.253729404 seconds;
  status scan/repair is 2.061544515 seconds, including 2.061175139 seconds to
  build the compatibility KD3 index, repair one incomplete row, and refresh
  the device column. Status-transfer call wall is only 0.138355024 seconds, so
  pinned status storage is rejected. Forced Morton BVH removes repair but
  slows complete candidate wall from 3.278321613 to 4.179201213 seconds. B0050
  may cheaply prototype selective exact device repair for incomplete NND rows
  using resident planner-owned data; no private index or selection widening.
- **D0113 — B0050 repairs exceptional NND rows from resident data.** The cheap
  serial full-scan prototype was exact but slower; the accepted bounded path
  gives each incomplete row one 64-thread block and exactly merges thread-local
  top-k sets. It applies only to kth mode, public `k=1..15`, and at most 16
  incomplete rows; all other cases retain KD3 repair. Five clean automatic
  samples at `abb460846` measure 1.413364609 seconds versus 17.809985276 seconds
  pinned PDAL (12.601126x), with the established LAS hash. The post-change
  profile makes 0.492926525-second reader/table materialization the largest
  reusable single phase; B0051 tests a direct-LAS resident-source hypothesis.

## Active next task

B0037 and B0038 are complete as diagnostic gates, not speed claims. B0037
rejects cross-kind terrain/feature product work because a 3.77-second sparse
masked HAG traversal dominates its 0.846x candidate. B0038's byte-exact 4M
`LOF(minpts=10) -> assign -> LAS` one-shot reaches 8.241x pinned PDAL, but its
instrumented candidate spends 2.340 of 4.410 seconds building the compatibility
host KD3 tree, recomputing a 40,982-row exact closure from 12,716 ambiguous and
one incomplete row, and refreshing the retained device columns. Upload pack,
spill wait, and publication total 0.232 seconds; all profiled device kernels
total about 0.145 seconds. The endpoint hypothesis is therefore secondary.

B0039 is also complete as a negative semantic gate. With all LOF columns
serialized, the ordinary exact hybrid output matches pinned PDAL, but retaining
the current device values for tie-affected rows first differs at point 40,449's
`LocalOutlierFactor` (0.9805045741137234 upstream versus
0.9805045741137232). Skipping repair gives a 2.061-second ceiling but also
differs. The current `(distance, point-id)` grid cannot replace pinned KD3 tie
order, and a private LOF index is forbidden. Exact host repair remains.

B0041 qualifies the opt-in exact `LOF -> assign` resident LAS endpoint on its
named 4M envelope. Five exact samples reduce the ordinary resident median from
4.197 to 3.700 seconds (1.134x, 11.8%) and reach 9.608x pinned PDAL. The full
physical v3 process gate passes ordinary/direct/upstream output, no-overwrite,
extra serialized-write, and writer-option cases. Empty/small experimental
fallback is exact, including the zero-point GeoTIFF-VLR stream case.

Do not build a device-column sink next: the surviving 100 MB spill measures
only 0.0825 seconds publication plus 0.00995 seconds wait, about 2.5% of direct
wall, while authoritative host repair is still 2.330 seconds.

B0042 proves the public `pdg pipeline` handoff under a positive proof switch.
B0043 completes its default admission boundary at `62016556b`. The expanded
host matrix preserves `--stream`, `--nostream`, metadata, missing-input errors,
below-placement fallback, and unsupported-graph behavior. The 281.94-second
physical v3 gate proves exact automatic output plus no-write existing-output
and injected-preflight refusals. Five clean option-free samples measure a
3.723975123-second median versus 35.738760555 seconds pinned PDAL (9.596939x),
with identical 144,000,375-byte output.

B0044 is complete: coplanar, normal, and nndistance are exact at 4.932x,
4.874x, and 5.275x pinned PDAL on one clean 4M direct-endpoint sample each.
Only nndistance advances. Its device profile measures the dominant GPU kernel
at just 138.57 milliseconds and the host boundary timers total 0.125 seconds;
the remaining roughly 3.10 seconds is host-side process work.

B0045 is complete. Explicit-direct and option-free five-sample medians are
3.339251347 and 3.390338813 seconds versus 17.705276847 and 17.726674770
seconds pinned PDAL (5.302170x and 5.228585x). Host fallback/options and the
expanded full-fixture v2 gate are exact.

B0046 is complete. Clean exact 4M nndistance and LOF captures put 92.90% and
93.79% of internal pre-stats wall inside rewritten manager execution.
Canonical publication is only 51–52 milliseconds. LOF's aggregate contains
2.287512704 seconds of already-required host repair; nndistance retains a
2.743655928-second manager remainder after known boundary and kernel work.

B0047 is complete. Exact 4M nndistance spends 0.502592322 seconds in
reader/row-table materialization and 2.524453274 seconds from accepted upload
through completed spill. LOF spends 0.509743845 and 2.901305423 seconds in the
same phases, with 2.294764896 seconds of its resident interval already
explained by exact host repair. Graph/prepare and post-spill work are
sub-millisecond; both established LAS hashes are unchanged.

B0048 is complete. Exact 4M nndistance's 2.506743228-second resident interval
contains 0.121900349 seconds index configuration, 0.008554448 seconds
index-build call wall, 2.233622069 seconds broad query/projection,
0.035821079 seconds adjacent assignment bridge, and 0.106845283 seconds
residual. LOF's matching broad query span is dominated by known exact repair.
Both established LAS hashes are unchanged.

B0049 is complete at `20d9ffabc`. The final exact 4M query span is dominated
by a 2.061175139-second compatibility KD3 repair for one incomplete row, not
the 0.138355024-second pageable status-transfer call. A forced planner-owned
Morton-BVH prototype removes repair but makes the complete candidate 27.48%
slower. The established output hash is unchanged, and the 21.97M physical v2
gate passes in 258.44 seconds.

B0050 is complete at `abb460846`. Its final five-sample automatic median is
1.413364609 seconds versus 17.809985276 seconds pinned PDAL (12.601126x), and
the output remains byte-identical. The final stats capture reports one
resident-coordinate repair in 0.077210784 seconds, zero host repair, a
0.492926525-second reader/table phase, and a 0.554462427-second complete
resident wrapper.

B0051 clears its cheap gate: the disposable exact 4M probe hydrates all mapped
LAS coordinates into resident binary64 columns at a 0.00840182-second median;
fresh allocator/device allocation is another 0.0986952 seconds. All 12,000,000
values match the host decoder. This is not an end-to-end performance claim.

B0052 is complete at `8c8ccb523`. The direct mapped-LAS source is bounded to
the already-qualified default-LAS NND endpoint and reuses the existing parser,
coordinate transpose, planner-owned batch/index, sparse compatibility table,
and canonical output overlay. Clean exact automatic samples measure
0.911029970 seconds versus 17.815840889 seconds pinned PDAL (19.555713x), a
35.541759% candidate-wall reduction from B0050. All four focused CUDA
sanitizers are clean and the 21.97M v2 gate passes in 232.49 seconds. The
source is now automatically selected only inside that existing envelope;
disable and unsupported inputs retain the unchanged PointView/PDAL paths.

B0053 is complete. The final exact profile records 139.479840 milliseconds in
the reusable kNN gather, 77.548288 milliseconds in the one-row repair, and
1.492864 milliseconds across all other kernels. Valid 128/256-thread launch
prototypes save at most 4.946912 milliseconds of gather time with no stable
material complete-process gain; 512 threads fails launch, and the 64-thread
qualified build is restored.

B0054 is complete at `8aaac57ff`. Its fresh 0.175984958-second startup
partition contains only 0.002231646 seconds of plan/original validation and
0.000567514 seconds of rewrite/preflight. Runtime placement consumes
0.173185798 seconds: 0.078605269 seconds for CUDA device/profile discovery,
0.094578699 seconds for initial memory-budget placement, and 0.000001830
seconds for executor selection. This is necessary process-level CUDA startup,
not removable validation work, so no prototype is retained.

B0055 is complete at `aca65d75e`. It finds 0.051803244 seconds in adaptive
configuration and 0.072138563 seconds in a redundant second host-envelope
scan. Treating the kNN builders' existing complete-coordinate proof as
authoritative reduces final index configuration to 0.051821472 seconds. The
clean exact automatic median is 0.821465859 seconds versus 17.895370391 seconds
pinned PDAL (21.784679x), 9.831083% below B0052 candidate wall.

B0056 is complete as a rejected disposable prototype. It halves configuration
to 0.026006785 seconds, but exact option-free samples of 0.834/0.825/0.844
seconds do not improve materially on B0055's 0.821465859-second qualified
median. The prototype is fully reverted and the qualified binaries rebuilt.

B0057 is complete at `ff3a55044`. Direct-LAS coordinate hydration owns
0.058526544 seconds, product setup 0.000147437 seconds, and the remaining
wrapper residual falls from 0.083464418 to 0.023428953 seconds. Output and the
qualified B0055 implementation are unchanged.

B0058 is complete at `0a943a359`. Hydration contains 0.042686939 seconds of
validation/materialization/allocation, 0.008537849 seconds of transfer/kernel
submission, and only 0.003948276 seconds of final wait. A D2H-only change is
too small.

B0059 is complete as a successful directional prototype. A point-record scan
reproduces the existing grid frame and adaptive backend bit-for-bit at both 4M
and 16M while taking 0.00426759 and 0.0148655 seconds median after warmup,
without allocating or mirroring host XYZ. It clears the production viability
gate but changes no compatibility, qualification, selection, or coverage
category.

B0060 is complete at `07ea2f4a7`. The exact mapped-record summary now
configures the planner-owned index while device XYZ remains resident and the
host XYZ mirror is absent. Five clean alternating samples measure 0.755523193
seconds versus 17.891433883 seconds pinned PDAL (23.680853x), 8.02744% below
B0055. Adaptive/forced backends, exact mapped KD3 repair, four CUDA sanitizers,
and the hash-pinned 16M process lane pass. The existing automatic envelope is
unchanged.

B0061 is rejected and cleanly reverted at `0b61e8ef2`. Its exact, proof-gated
implementation reduced the clean automatic median from 0.755523193 to only
0.750254638 seconds (0.697339%) and removed no transfer. No production,
qualification, selection, or coverage change remains.

B0062 is accepted at clean prototype commit `23cc26827` and promoted by D0125.
Five alternating exact samples measure 0.712510475 seconds versus
17.777145188 seconds pinned PDAL (24.950012x), 5.693104% below B0060. Normal
execution removes one 32,000,000-byte H2D and 32,000,008 bytes of D2H; forced
host repair restores the complete column in both directions and remains exact.
All four CUDA sanitizers and the hash-pinned 16M process lane pass.

B0063 is complete. Its 21-launch profile measures 137.153696 milliseconds in
the kNN gather, which already accounts for the apparent 0.134898783-second
status-call interval, plus 77.754496 milliseconds in the one-row exact repair.
Pinned status storage is rejected without a prototype.

B0064 is complete at promoted commit `1aebd1877`. The exact multi-block repair
reduces the one-row repair to 2.232480 milliseconds. Five clean samples measure
0.623028204 seconds versus 17.950882850 seconds pinned PDAL (28.812312x), and a
same-binary serial/parallel control shows a 9.242103% endpoint reduction. The
fresh promoted profile leaves 97.3487% of kernel time in the reusable gather;
B0053 already rejected its bounded launch-shape variants. There is no clear
reusable optimization likely to add another 5--10%, so this endpoint is
sufficiently optimized for now. B0065 resumes the catalog by measuring and
profiling the already-native, performance-unmeasured statistical
`filters.outlier` lane and its plausible shared-index resident composition
before authorizing any new stage port.

B0065 is complete as a directional 1M gate. Stock PDAL, forced exact CUDA, and
read/write-only medians are 4.001562112, 0.620354816, and 0.278837248 seconds;
the outlier outputs are byte-identical. Hardware-cycle samples name
nanoflann's per-point search as the stock limiter. Nsight Compute finds only
21.557088 milliseconds of device work, including 21.257536 milliseconds in the
kNN gather and 0.299552 milliseconds in index construction plus all other
launches. The remaining candidate wall includes CUDA/process startup and a
material residency/bridge surface, not a new-kernel opportunity.

B0066's disposable prototype is positive and retained. The statistical
outlier wrapper now consumes the planner-owned resident kNN product, preserves
the exact host global recurrence/classification and incomplete-row repair, and
does not admit radius outlier. A focused physical test requires index reuse and
observes one build across adjacent NNDistance. The dirty 1M complete
composition is byte-identical at 0.712635520 seconds median versus
8.089364299 seconds pinned PDAL (11.351335x); a same-binary control measures a
smaller but positive 2.964965% reduction versus two private CUDA stages while
removing one 24 MB XYZ upload and one index build. This is not accepted
performance qualification and changes no automatic selection. B0067 therefore
profiles the improved endpoint before any clean exact qualification gate.

B0067's fresh profile now stops kernel work: 18 launches total 51.702304
milliseconds, including 22.906240 and 28.500256 milliseconds in the two kNN
gathers and only 0.295808 milliseconds elsewhere. Perfectly eliminating the
smaller gather can save only 3.437863% of the 0.666293053-second reference
process wall. The remaining 92.240306% is outside kernels, and B0052 already
proved the reusable direct-LAS source/output boundary can be worth far more
than 5--10% on a comparable resident neighborhood endpoint. B0068 next gets
one cheap exact Classification-overlay/source-table prototype for this
existing boundary. Revert and close the endpoint if its complete 1M gain is
below roughly 5--10%; do not add a kernel or automatic model first.

B0068 clears that gate. Seven alternating exact 1M samples measure
0.373325396 seconds for direct source plus Classification overlay versus
0.646892319 seconds for the same resident region with ordinary LAS boundaries,
a 42.289407% reduction. The new format-3 physical gate proves exact canonical
Classification/flag handling and now also covers formats 6, 7, and 8. It
requires direct source/output, one index build, the raw-record hydration event,
an honest uncalibrated/boundary-mismatch report, and fail-closed rejection.
This is retained opt-in infrastructure, not dirty-tree performance
qualification or automatic selection. The outlier endpoint is sufficiently
optimized for now.

B0069 resumes the measured catalog with B0044's exact
`approximatecoplanar(knn=8) -> Coplanar=>UserData` direct-output pipeline. The
clean fresh 1M processes are byte-identical at 4.293 seconds pinned PDAL and
1.018 seconds resident. Its 20 kernels total only 21.592970 milliseconds. The
dominant exact path instead downloads a 1 MB ambiguity mask and all 96 MB of
eigensystems, repairs selected rows through pinned KD3, and uploads all 97 MB
again. Direct LAS source reuse is deferred because this path still needs the
host PointView.

B0070 is negative and reverted. The exact 1M fixture has 2,145 tied rows;
selective repair cuts transfer from 194,000,000 to 1,208,065 bytes but improves
shell wall only 0.491159%, command-before-stats 2.267040%, and the resident
wrapper 3.357725%. Pinned KD3 construction and ordered recomputation still
take 0.449844793 seconds. The approximate-coplanar endpoint is sufficiently
optimized for now.

B0071 is strongly positive. The exact public-shaped 1M composition takes
14.591 seconds pinned PDAL and 1.226 seconds when forced through one shared
index/eigensystem region, or 11.901305x. Ordinary resident placement correctly
falls back with `mixed_calibration_models` and takes 14.908 seconds. All 22
kernels total 41.533930 milliseconds; the three projections and point program
together are only 0.372130 milliseconds, so kernel fusion is rejected.

B0072 is complete. All seven exact forced rows from 50K through 16M are
positive (2.155256x to 14.569879x) and require one shared index plus the
13-neighbor eigensystem cache. The proposed distinct fit is
`host=14,812.948623 ns/point` and
`device=185.808308 ms + 1,005.425583 ns/point`, with a 250K floor.

B0073 is complete at `32b864711`. Only the measured exact region resolves to
`eigen-family-compose`; all changed-k/option/assignment/topology shapes retain
`mixed_calibration_models`. The physical explicit-resident gate proves one
index and the reused 13-neighbor eigensystem, accepted preflight, exact warning
stderr, and exact LAS bytes. Five alternating 1M samples measure
14.507773633 seconds pinned PDAL versus 1.397204802 seconds resident, or
10.383427x.

B0074 is complete. The exact 50K control selects host; every 250K-through-16M
row selects `planner_resident_shared_index` and wins by 6.406690x through
11.915007x. The replacement resident fit is
`host=120.004601 ms + 14,659.538987 ns/point` and
`device=155.357778 ms + 1,234.745145 ns/point`. All seven raw pins now name
resident reports, and the executor-calibration flag is truthfully true.

B0075 is complete at `b39ef04c6`. Required and genuinely option-free public
execution are byte- and diagnostic-exact; changed k, assignment order, extra
topology, preflight failure, and below-floor placement decline before output.
Five alternating 1M samples measure 14.452746005 seconds pinned PDAL versus
1.409086026 seconds automatic PDG, or 10.256823x. Direct LAS output remains
outside this exact three-field finale.

B0076 is complete as a directional gate. The ordinary 1M rank/optimal pipeline
is exact at 11.303210336 seconds pinned PDAL versus 1.246256090 seconds forced
shared resident, or 9.069733x. An identity coordinate boundary preserves the
output hash, forces two regions, and takes 1.697469160 seconds: shared residency
removes 26.581518% of candidate wall. The two 44-millisecond gathers dominate
the 88.786784-millisecond kernel profile, but even deleting one perfectly has
only a 3.557802% process ceiling. Do not optimize kernels.

B0077 is complete. Every exact forced-composition row from 50K through 16M
requires shared-index reuse and wins by 1.542997x through 10.755371x. The
separate conservative fit is
`host=117.903408 ms + 11,435.542525 ns/point` versus
`device=233.890937 ms + 1,050.229585 ns/point`, bounded to 250K--16M. The
per-stage models are unchanged.

B0078 is complete at `7d0efc2ff`. Only the exact compiled B0076 shape resolves
to `rank-optimal-compose`; changed options, assignments, topology, extra
consumers, and coordinate invalidation do not inherit it. The physical gate
proves preflight, one region/index, reuse, and exact output. Five alternating
1M samples measure 11.353348769 seconds pinned PDAL versus 1.370295007 seconds
resident, or 8.285332x. The executor-provenance diagnostic remains false
because B0077 was forced hybrid.

B0079 is complete. The exact 50K control selects host; every
250K-through-16M row requires the shared-index executor/reuse and wins by
4.873850x through 9.566431x. The replacement resident fit is
`host=133.650155 ms + 11,448.442352 ns/point` versus
`device=240.029160 ms + 1,186.527542 ns/point`. All seven raw pins now name
resident reports, and the executor-calibration flag is truthfully true.

B0080 is complete. Required and genuinely option-free public paths are exact;
changed options/assignment order/topology, below-floor placement, and rejected
preflight decline without output. Five alternating 1M samples measure
11.324354873 seconds pinned PDAL versus 1.388344338 seconds automatic PDG, or
8.156734x. The unchanged B0076 profile leaves only a 3.557802% perfect
single-gather-removal ceiling, so this endpoint is sufficiently optimized for
now. Direct output for this rank/optimal composition, broader rank/optimal
shapes, HAG count five, and new stage ports stay deferred while the next task
is selected from measured existing-stage composition value.

B0081 is complete as a directional gate. The clean zero-warmup 1M pair is exact
at 1.921081414 seconds pinned PDAL versus 0.697987348 seconds resident, or
2.752316x, with the shared-index executor and one index build observed. The
single consumer cannot prove reuse; the runner's reuse switch was a no-op. All
16 kernels total 1.899456 milliseconds and perfect radius-query removal is only
0.234172% of process wall. Ordinary stats put 0.390594913 seconds in rewritten
manager execution and 0.178538164 seconds in the already-dispositioned runtime
placement/setup path. Do not optimize the kernel.

B0082 is complete as an accepted directional prototype. Seven alternating 1M
same-binary samples measure 0.641774202 seconds ordinary resident versus
0.316225432 seconds direct resident, or 2.029483x/50.726372% less complete
wall. The pinned physical gate is byte- and diagnostic-exact for LAS formats 7
and 3, positively exercises ReturnNumber-driven UserData output, observes one
shared index and every source/summary/no-host-XYZ proof, and rejects shape drift
or a disabled source before output. All four Compute Sanitizer tools are clean.
The executor and boundary provenance flags remain truthfully false.

B0083 is complete. The exact 50K host control loses at 0.358367x, while every
required direct row from 250K through 16M wins at 1.523498x through 22.408647x
with all direct source/output/summary/no-host-XYZ proofs. The bounded proposal
is `radiusassign-direct`: host `0 + 1,985.118056 ns/point`, device
`305.407938 ms + 68.335971 ns/point`. These are calibration rows only.

B0084 is complete: the separate model and clean raw pins are integrated without
touching ordinary radiusassign, exact-shape and negative matching pass, and the
two-format physical gate reports truthful direct-executor/boundary provenance.

B0085 is complete. The clean five-pair direct-radiusassign gate is exact at
1.896917103 seconds pinned PDAL versus 0.339432621 seconds direct resident, or
5.588494x, with both calibration and boundary provenance positively required.
The fresh exact profile records 1.953650 milliseconds across 18 kernels; even
perfect kernel removal is only 0.575563% of candidate wall. The material
placement interval is the process-level CUDA startup already closed by
B0054/D0117, and every other named reusable phase is below the 5--10% bar.
Treat the explicit endpoint as performance-qualified and sufficiently
optimized. Automatic selection remains off because no option-free gate ran.

B0086 is complete as a clean routing decision, not a qualification. The exact
1M pair takes 8.011663864 seconds pinned PDAL and 0.420340609 seconds through
the direct one-index composition, or 19.059933x. The fresh exact profile has
20 launches and 52.001290 milliseconds of device work; the two
`knnGatherKernel` calls take 23.260000 and 28.390000 milliseconds. The smaller
repeated gather is now 5.533608% of direct process wall, crossing the rough
prototype threshold only because the reusable LAS boundary removed the larger
host endpoint cost. Calibration and boundary provenance remain false and no
performance qualification or automatic selection follows from one pair.

B0087 is complete at `21836fa4d`. The planner now retains one exact ordered
max-k rowset only for the adjacent statistical-outlier -> NNDistance pair,
budgets its 133 bytes/point on the named width, and declines reuse across a
bridge, branch, incompatible region/dimension, or unmaterializable runtime
shape. The dirty-worktree same-binary direction gate improves exact median wall
11.315433%; the clean profile reduces total GPU work 43.557585% and proves one
gather plus two negligible projections. The remaining gather owns 97.745668%
of device work and has no new endpoint-specific 5--10% target, so stop tuning
this endpoint. Qualification, calibration, and automatic selection are still
unchanged.

B0088 is complete on clean commit `22f5218da`. One warmup and five alternating
pinned-PDAL/direct pairs are byte- and diagnostic-exact. Median wall is
7.961676747 versus 0.372223208 seconds, or 21.389523x, and every candidate
proves the direct source/output/record-summary boundary, no host XYZ mirror,
one predicted/observed index, and planner-owned max-k reuse. Its resolved
engine hash matches B0087's fresh exact profile, so the explicit route is now
performance-qualified. It remains uncalibrated and not automatically selected.

B0089's 10K-through-4M ladder is complete. The 4M host-repair cliff led to
B0090's bounded parallel incomplete-mean repair, and B0091 then removed the
profiled serial NNDistance repair. The clean final 4M route is exact at
48.758796x pinned PDAL; one shared gather owns 95.545682% of kernel work and
all non-gather kernels have a 0.969583% complete-wall ceiling.

B0092 completes stable repaired 8M/16M rows and the separately named direct-
composition model. The 25K/50K controls, strict one-region/index/lane and
133-byte query-product matcher, 142/142 audit, fail-closed observed execution
proofs, and five option-free 1M pairs are exact. Automatic median is
0.392659506 seconds versus 7.971261964 seconds pinned PDAL, or 20.300698x.

B0093 completes that public-admission gate without changing the executor or
model. Five option-free 1M pairs are exact at 5.366005x pinned PDAL; the exact
250K--16M direct-radiusassign envelope is now automatic, while broader shapes
remain explicit/host fallback.

B0097 completes the cheap option-free automatic-admission gate around the
unchanged B0096 `approximatecoplanar(knn=8) -> Coplanar=>UserData` route. Its
five exact 1M pairs retain 4.230620x pinned-PDAL value, and the fresh automatic
profile reconfirms the endpoint has no reusable 5--10% optimization surface.
B0098 then measures the already-native `neighborclassifier(k=7)` lane at
3.400787x through ordinary residency and retains the reusable exact direct-LAS
Classification boundary at 3.959949x. It remains explicit, uncalibrated, and
not automatic. B0099 attributes 1,947 tie rows and 0.594524233 seconds to one
pinned KD3 repair. A fully reverted exact proof discharges 1,946 rows but the
one non-invariant row still needs the same full index, so endpoint tuning is
closed below the 5--10% gate.
B0100 resumes the catalog with the already-native
`radialdensity(radius=1.01)` lane. Five clean exact 1M pairs qualify the
explicit forced-CUDA pipeline at 4.720283x pinned PDAL. Its fresh profile has
only 2.973280 milliseconds across all 16 kernels, 0.449921% of candidate wall,
so the next hypothesis is the resident direct-LAS/point-program/output boundary
rather than another radius kernel.

## Rules that bite

- **Measure before porting or optimizing.** Start with stock pinned PDAL and a
  complete-process profile, prototype cheaply, and proceed only for a measured
  standalone win, significant transfer removal, reusable shared work, or an
  explicit catalog-coverage need. A kernel timing never promotes a stage.
- **Every new stage follows the same slice shape**: primitive (+ host
  mirror + non-CUDA stub) -> `tryCuda*` client with tie/incomplete repair
  -> wrapper filter with a verbatim upstream host fallback -> hybrid
  qualification -> planner program/compile/dispatch -> resident admission
  (rewrite producer/wrapper switch, `planDeclaresNeighborhoodRegion`,
  preflight scratch branch) -> differentials (host, CUDA with forced tie
  repair, one-shell-budget full repair) + region emission test ->
  calibration ladders -> 22M lane -> records. D0078 (`f6eacc179`) is the
  cleanest end-to-end template.
- **Calibration provenance is load-bearing.** A model measured on forced
  hybrid-CUDA lanes may bootstrap the profile, but the stage may not join
  `ResidentCalibratedModels` until a resident-mode ladder replaces those
  rows (`status: clean-resident-shared-index`). Both the JSON case rows
  and the embedded `PlacementProfile.cpp` coefficients must match
  bit-for-bit or the audit refuses to run.
- **`PlacementStageCalibration` array size and the placement pin test**
  (`tests/unit/placement_test.cpp`) must be bumped together with every
  new model.
- **Raw reports live under `build/benchmarks/`**, never `/tmp` (D0065).
  One heavy GPU lane at a time; CUDA builds `-j1`, host `-j2`; keep 8 GiB
  `MemAvailable` free.
- **Never commit, upload, rename, or modify the corpus fixture.**
  Exporting it as `PDG_LOCAL_LAS_FILE` to the whole suite activates the
  optional <=128 MiB corpus smoke test and fails it by design: run the
  corpus test without that export and the GPU gates with it.

## Milestone state

- D1: the planner/allocator liveness contract is implemented and exercised
  across a real spill/resume boundary (D0058).
- D2: cardinality-preserving fusion (D0052), the declared cardinality-changing
  expression case (D0061), and declared compacting writer-prologue endpoint
  fusion (D0063) are implemented. Feature-kernel prologue/epilogue fusion into
  neighborhood producers/consumers remains open.
- D3: implemented for current pipeline classes. Point-program regions use
  fixed, benchmark-selected two-lane defaults; delegated shared-index
  neighborhood regions use a planner-verified single-tile/one-lane
  `whole_view_neighborhood` schedule (D0066). The kNN scheduler class stays
  prohibited until exact tiling exists.
- D4: the frozen 52-row matrix is untouched. `direct_ordered_las`
  (D0064) and `planner_resident_shared_index` (D0077) both report
  `selected_device_calibration_matches_executor` true from their own
  measured provenance; `planner_resident_boundary_batch` remains honestly
  false pending its own calibration — do not synthesize it.
- V1-V8, E1-E7, and B1/B2/B3 all hold with recorded evidence; P1.5 is
  complete (D0071). kNN tiling stays prohibited until exact tiling exists
  (it was never a P1.5 item).
- The diagnostic `pdg resident` path remains explicit. B0043 changes the
  public default only for its exact, measured LOF/default-LAS shape; no other
  post-P1.5 resident envelope inherits automatic selection or public
  GPU-native coverage.

## Resident catalog at the tip

Ten neighborhood/radius stages execute under the resident planner with
measured, resident-provenance models, all byte-exact on the 22M fixture:

| Stage | 22M resident lane | Record |
| --- | ---: | --- |
| `filters.radiusassign` | 5.503x | B0019 / D0081 |
| `filters.lof` | 8.459x | B0017 |
| `filters.optimalneighborhood` | 5.062x | D0079 |
| `filters.nndistance` | 4.356x | B0018 |
| `filters.estimaterank` | 4.155x | D0078 |
| `filters.eigenvalues` | 4.084x | B0018 |
| `filters.normal` | 4.037x | B0018 |
| `filters.approximatecoplanar` | 3.962x | B0017 |
| `filters.covariancefeatures` | 3.955x | B0018 |
| `filters.neighborclassifier` | 3.743x | D0080 |

`filters.label_duplicates` is additionally an exact non-index resident
producer. It is not in this speedup table because B0020 is a negative
standalone gate (0.959x at 22M), and no resident end-to-end performance model
has been accepted.

B0098 separately performance-qualifies the explicit exact 1M
`neighborclassifier(k=7)` direct-LAS boundary at 3.959949x; it is not a 22M
resident-model row and does not alter this table.

P3 has started with the bounded D0083 SMRF, D0084 PMF, D0085 CSF, D0086 ELM,
D0087 skewness-balancing, and D0088-D0093 tiled-PMF foundation slices.
Exact host and physical SM-89 CUDA execution cover each admitted upstream
global-raster/cloth phase order for finite logical-double XYZ and at most 4,096
cells. PMF additionally caps morphology at radius 64 and 64 passes; SMRF
requires at least a 2x2 canvas and caps morphology at radius 64; CSF requires a
serial non-OpenMP oracle, `smooth=false`, an at-least-2x2 cloth, and no more
than 64 iterations; ELM preserves stable per-cell numeric-Z/source order and
strict threshold comparison. The bounded global lanes select only standalone
whole-view regions. D0088-D0093 can tile a required PMF region and can keep a
compatible adjacent PMF chain in that region. Every PMF stage rebuilds and
consumes its own exact source-raster generation using the runtime frame and
planner budget, while a fitting chain reuses one promoted device phase
allocation and performs zero raster transfers. Larger or under-budget frames
use the exact host-build/tiled path. Different return sources or cell frames,
non-PMF consumers, and the other canvas stages still cannot cross this bridge.
B0021, B0022,
B0023, and B0024 are exact negative standalone gates (0.599x, 0.583x, 0.289x,
and 0.866x at 1M), so none has an automatic placement model. Unsupported
options and runtime envelopes retain unchanged PDAL or the exact host wrappers.

The D0088-D0093 PMF product is correctness infrastructure, not the scalable P3
canvas. It owns deterministic one-cell halo tiles, two cumulative-budgeted
pinned-host backings, and a serial phase schedule. Distinct-valued
equal-distance void-fill ties reject before Classification publication.
D0089 removes B0026's tile-dependent point scans; B0027's exact dense
65x65/four-tile lane reaches 1.828354x PDAL. D0090 keeps fitting phases on
device; B0028 reaches 9.009503x dense and 4.764988x sparse/void with one raster
upload and no phase download. Its final profile records 20 launches per
candidate and no tiled phase kernel. D0091/B0029 adds an allocation-free exact
occupancy hierarchy and demonstrates near-linear work on the named 256-source
257x257/513x513 distributions. D0092/B0030 adds separately budgeted
device-native construction and tie proof for fitting frames, promotes the
provisional allocation into the phase pair, and records zero raster transfers.
D0093/B0031 then reuses that one phase allocation across a compatible adjacent
PMF pair while rebuilding two exact raster generations. Its final
symmetric-scope pair is exact at 1.039838x pinned PDAL. The exact device proof
still scans every compact source per target and has no worst-case subquadratic
bound. The accepted bridge is allocation reuse only; no general scalability,
semantic surface reuse, or cross-kind product-reuse claim follows.

The skewness lane is separate from those Grid regions: it uses CUDA only for
the exact permutation of comparator-unique binary64 Z, then returns to the
host for full-record publication and the original recurrence. B0025 is a
dirty-snapshot controlled 1.150x diagnostic at 1M, but ordinary tied-Z data
falls back and there is no performance qualification, resident product, or
automatic model.

## Next work

B0233/D0232 supersedes the older queue discussion below: label-duplicates is
complete inside its strict automatic hybrid composition, while standalone and
resident directions remain rejected. The remaining promotion queue is
`filters.skewnessbalancing`, `filters.hag_nn`, and `filters.hag_delaunay`.
Follow the same test-first sequence: public/direct attribution, an exact
cardinality ladder, a separate composition model, fail-closed admission and
post-execution proof, exact public benchmarks, then sanitizer/profile/
documentation gates. Do not borrow an ordinary stage model or expand layouts,
options, or device profiles beyond measured evidence.

`IMPLEMENTATION_PLAN.md` holds the ordered list. At this checkpoint the four
post-P1.5 roadmap items, radiusassign, label_duplicates, and the bounded first
SMRF, PMF, serial-oracle CSF, ELM, skewness-balancing, count-one-through-six
HAG NN, and count-three HAG Delaunay lanes are closed. The
first PMF core/halo/seam/mosaic product and a controlled exact occupancy fill
are now proved. Fitting PMF frames now also build and prove their complete
raster on device with zero raster transfers, and compatible adjacent PMF stages
reuse one planner product/device allocation while rebuilding exact stage-local
rasters. Cross-kind Grid/neighborhood chains use explicit spill/upload
boundaries and never carry either semantic product across the boundary.

Do not start the next P3 stage port. D0099 completed the audit of every
implemented stage as functionally supported, GPU-native,
performance-qualified, and automatically selected, treating “unmeasured” as a
required explicit result. D0100/B0037 rejected the first cross-kind terrain
sharing hypothesis because its sparse masked HAG traversal, not its boundary,
dominates. D0101/B0038 then names exact LOF host repair as the resident
limiter, and D0102/B0039 proves that the current device tie order cannot replace
pinned KD3 repair. B0040 through B0050 qualify the automatic resident NND
endpoint and remove its exceptional host-repair cliff. B0051 has now cleared
the cheap direct-LAS hydration gate, and B0052's bounded production source
reduces exact complete-process wall another 35.54%. B0053 resolves the device
timeline and rejects launch-size tuning. B0054 finds that the remaining
validation/placement/preflight aggregate is CUDA/process startup rather than
removable validation. B0055 removes the measured redundant kNN envelope scan
and improves the exact automatic endpoint another 9.83%. B0056 rejects a
smaller extrema-reuse scan because it does not improve complete wall. B0057
finds 58.53 milliseconds in direct-source hydration. B0058 shows allocation
dominates while final wait is only 3.95 milliseconds. B0059's exact disposable
record-summary probe clears its directional gate at 4.27 milliseconds for 4M
and 14.87 milliseconds for 16M. B0060 productionizes it inside the existing
endpoint and reduces clean exact wall another 8.03%. B0061 rejects
PointView-only publication elision after a non-material 0.70% gate. B0062
removes the larger conditional 32 MB upload/download pair and improves the
clean exact median another 5.69%. B0063 shows that the residual status-call
interval is queued kNN gather work and rejects pinned storage. B0064 closes
the one-row 77.75-millisecond repair underfill with an exact
2.232480-millisecond parallel reduction and improves the clean endpoint to
0.623028204 seconds (28.812312x pinned PDAL). Its fresh final profile leaves
97.3487% of kernel time in the gather already explored by B0053, so the
endpoint is sufficiently optimized for now. B0065 then measures stock PDAL and
profiles the already-native but performance-unmeasured statistical
`filters.outlier` pipeline. Its directional 1M gate measures 4.001562112
seconds pinned PDAL, 0.620354816 seconds forced CUDA, and only 21.557088
milliseconds of device kernels. B0066 therefore prototypes planner residency
and one-index composition around the existing exact outlier work; it does not
authorize a new standalone kernel or a performance claim. B0067 rejects more
query/kernel work, and B0068 retains the exact opt-in direct-LAS
Classification boundary after a 42.289407% same-binary reduction. B0069 then
profiles the existing positive approximate-coplanar composition: device work
is only 21.592970 milliseconds, while exact ambiguity repair performs a 194 MB
full eigensystem/status round trip. B0070 tests sparse selective repair before
any new stage port or direct-source expansion. It removes 99.377286% of that
transfer but improves complete wall by less than the 5--10% gate, so the
prototype is reverted. B0071 now measures the existing composed normal/
eigenvalue/covariance pipeline. It is exact at 11.901305x pinned PDAL; the
planner's truthful `mixed_calibration_models` rejection, not device work, is
the integration gap. B0072's exact seven-row ladder is positive throughout and
fits a separate composition model. B0073 now owns a narrow, exact 1M
explicit-resident qualification at 10.383427x, while correctly retaining the
model-wide calibration/executor mismatch diagnostic. B0074 now replaces the
forced-hybrid ladder with exact resident rows, refits the model, and promotes
the executor-match diagnostic. B0075 now closes the separate public automatic
gate at 10.256823x. B0076 then measures the existing rank/optimal composition
at 9.069733x and rejects kernel work; B0077 owns its bounded calibration
ladder before any new stage port.
B0086 then profiles the already-native outlier/NNDistance composition and finds
duplicate kNN gathers. B0087 retains planner-proved max-k rowset reuse, B0088
qualifies the exact explicit 1M route, and B0089's size ladder exposes two
bounded host-repair cliffs at 4M. B0090 and B0091 remove those cliffs with
independently proved exact parallel resident repairs. The clean 4M route is now
48.758796x pinned PDAL; its final profile leaves only a 0.969583% ideal
complete-wall ceiling outside the shared gather. Endpoint tuning therefore
stops here. B0092 completes the separate bounded fit and promotes only the
exact measured composition through an option-free 20.300698x public gate.
B0093 promotes the unchanged direct-radiusassign endpoint through an exact
option-free 5.366005x public gate. B0094 revalidates the already-native
approximate-coplanar/ferry composition at 4.148518x but finds its current
profile and repair provenance insufficient for calibration. B0095 supplies
that telemetry and a fresh exact profile and closes further endpoint tuning
below the 5--10% gate. B0096's current-binary 50K--16M ladder then qualifies
the explicit route and fits its separate bounded model. B0097 now promotes
only that exact measured composition through an option-free 4.230620x public
gate and reconfirms the stopping decision with a current output-bound profile.
The clean B0101 implementation is commit `29451ec13`. Five exact resident
pairs measure 3.097940201 seconds pinned PDAL versus 0.345532286 seconds PDG,
or 8.965704x. The matching exact same-binary hybrid control is 0.611763765
seconds, so resident composition removes 43.518674%. The fresh output-bound
profile reproduces the accepted LAS and records 19 kernels totaling 3.057664
milliseconds, only 0.884914% of resident wall. The larger placement/control
span is required CUDA driver/context initialization and actual-free-memory
input; no clear reusable 5--10% removal remains. The endpoint is sufficiently
optimized for now, explicit-only, uncalibrated, and not automatically selected.

B0102 physically qualifies the existing ordinal CUDA envelope: eight focused
device/process gates and all four Compute Sanitizer tools pass. The exact
`decimation(step=2) -> assign` direction is nevertheless only 0.449898x pinned
PDAL at 1M and 0.762657x at 4M. NCU records 241.376 milliseconds across 64
decode/mask/CUB-select/assign/repack launches, and a reversible one-chunk
control is slower. Report this as GPU-native force-only coverage, not a
performance qualification or automatic selector.

B0103 physically qualifies the exact first-tie locate envelope: five focused
CUDA gates and all four Compute Sanitizer tools pass. Exact
`assign(UserData=1) -> locate(Z,min)` is only 0.387892x pinned PDAL at 1M and
0.706034x at 4M. The 1M profile has 54.272 milliseconds of device work inside
0.432344452 seconds candidate wall. Direct-boundary reuse could remove the
roughly constant 250-millisecond deficit but only reaches estimated break-even,
and no cheap 5--10% prototype exists. Report locate as GPU-native force-only,
not performance-qualified or selected.

B0104 physically qualifies the exact option-free info reduction, but current
1M/4M complete-process rows reach only 0.525893x/0.744854x pinned PDAL. The 1M
profile has only 0.26863 milliseconds of kernels inside 0.564460173 seconds
candidate wall. Exact host info is just 3.259660 milliseconds/1.03986% above
the same-binary bare translation control, so record-summary reuse cannot clear
the 5--10% gate. Retain info as GPU-native force-only and host-selected.

B0105 cleanly qualifies only the forced 1,000,002-point count-four HAG fixture:
0.996831814 seconds pinned PDAL versus 0.637868218 seconds PDG, or 1.562755x
over five exact pairs. All 18 profiled kernels total 3.39736 milliseconds,
0.532612% of candidate wall. A trivial same-binary HAG-shaped publication
control takes 0.345457608 seconds, leaving a 45.8419% prototype surface.

The fully reverted B0106 prototype executes the existing count-four plan
through one ordinary shared-index region and remains byte-exact, but takes
0.675 seconds against the same-binary hybrid control's 0.657 seconds. Its
33,000,066-byte fallback-writer spill and 0.377666605-second rewritten manager
interval explain why ordinary residency does not remove the B0105 endpoint.

The retained B0107 publisher appends one standard binary64 Extra Bytes column,
elides the terminal spill, and keeps full HAG output exact. B0108 tested its
last measured placement hypothesis and fully reverted it: 0.53 seconds versus
the 0.55-second same-binary control is only a 3.636% reduction. Stats show the
removed initial-placement interval reappearing in device/profile startup, so
there is no reusable 5--10% surface. The HAG direct-output endpoint is
sufficiently optimized for now. Only exact unsupported-descriptor fallback
hardening remains from B0108.

The B0109 exact 1M `filters.nndistance(k=10) ->
writers.las(extra_dims=all)` measurement is positive: pinned PDAL takes 2.01
seconds, CUDA hybrid 0.66 seconds, and the same-binary output-shaped lower bound
0.39 seconds. Its fresh profile reproduces the output hash and records only
25.767520 milliseconds of kernels, including 25.358368 milliseconds in the
already shared Morton-BVH gather. The 0.27-second/40.91% gap is a material
resident lifecycle/direct-output hypothesis, not a kernel target.

B0110 retains only the explicit NNDistance `mode=kth,k=10` composition with
the generic one-binary64 publisher. The clean stats-free pair improves from
0.71 seconds hybrid to 0.57 seconds direct (19.718%), with the exact pinned
artifact, one planner-owned index, zero terminal/fallback spill, and strict
unsupported-shape fallback. The fresh profile totals 25.856256 milliseconds,
including 25.446208 milliseconds in the existing shared gather, so do not tune
the kernel.

The B0111 mapped-source composition is retained after the exact current-binary
pair improves from 0.58 to 0.38 seconds. Its fresh profile totals only
26.086656 milliseconds/6.865% of wall, almost entirely in the existing shared
gather; publication is below 4%, and B0108 already rejected the attributed
startup/placement shortcut at 3.636%. The endpoint is sufficiently optimized
for now. It remains explicit, directionally measured, unqualified, and not
automatically selected.

The B0112 current-clean measurement reproduces the pinned HAG Delaunay output
at 1.334030x. Its 18 kernels total only 2.342688 milliseconds/0.385406% of the
0.607849511-second hybrid process, while the same-record-shape lower bound is
0.40 seconds. The 34.1942% gap is a material mapped-source/PointView/terminal
publication hypothesis, not a kernel target. It is directional, unqualified,
and does not change selection.

B0113 retains that exact mapped-source/direct-output composition at 0.41
seconds versus its 0.55-second source-disabled control. The fresh profile has
only 2.411420 milliseconds/0.588151% of device work, publication is below 4%,
and B0108 already rejects the apparent placement shortcut. The endpoint is
sufficiently optimized for now; it remains explicit, directionally measured,
unqualified, and not automatically selected.

B0114 retains the same source composition for B0107's strict HAG-NN count-four
endpoint at 0.37 seconds versus its 0.56-second source-disabled control. The
fresh profile has only 3.523550 milliseconds/0.952311% of device work and
publication is below 4%; B0108 already rejects the remaining apparent
placement shortcut. The endpoint is sufficiently optimized and remains
explicit; B0105's named performance qualification is unchanged.

B0118 qualifies the exact forced-hybrid 2M `filters.mortonorder ->
filters.assign(UserData=17)` graph at 1.252847x pinned PDAL. Its fresh profile
totals only 0.618688 milliseconds/0.068310% of candidate median, and the exact
current-binary Morton-only control shows that the complete adjacent assignment
stage costs only 0.033078598 seconds/3.790666%. That is below the 5--10% gate,
so no fusion/resident prototype, model, selector, or product-code change
follows. The endpoint is sufficiently optimized for now; B0006's automatic
Morton envelope remains unchanged.

B0119 qualifies the exact forced-hybrid 2M `filters.mortonorder ->
filters.head(count=100)` graph at 1.395295x pinned PDAL. The fresh profile
totals only 0.688512 milliseconds/0.119095% of candidate median. A fully
reverted four-line upper-bound hook truncates the exact CUDA permutation to
100 ids before PointView publication, but its exact five-pair median improves
only 4.327416% with overlapping ranges. That misses the 5--10% gate. No fused
ordinal product, model, selector, or code is retained, and the clean engine
SHA-256 is restored to `f48f17d8cda533322ed379ec7afe679d43bfceb960eb2baf718e8e3230685b35`.

B0120 physically proves the existing finite-basic `filters.stats` CUDA path
on the six-dimension 2M graph, including exact LAS, metadata, streams, and
status, but rejects it decisively at 0.428539x pinned PDAL. Sixteen exact
ordered-recurrence kernels total 498.966048 milliseconds, 8.700974x the
57.345997-millisecond incremental CPU stats cost. Exact within-dimension order
leaves only six blocks/0.01 waves per SM. No stats fusion, model, selector, or
product change follows; option-free stats remains pinned-host selected.

B0121 now current-clean qualifies the named 1,000,002-point
`filters.hag_nn,count=2` forced-hybrid lane at 1.578557x pinned PDAL and its
strict direct mapped-source/one-binary64-publication composition at 2.606656x.
The direct path is 39.730794% below the clean hybrid median, proves one region
and one planner-owned masked 2D index, and preserves exact tie/incomplete host
repair before atomic publication. All four Compute Sanitizer tools are clean.
The fresh direct profile totals only 2.552608 milliseconds/0.698944% of wall;
publication is 3.594514%, and no remaining bounded surface has credible
reusable 5--10% headroom. Treat this endpoint as sufficiently optimized and
keep it explicit/unselected because ordinary data can require host repair and
there is no calibrated model.

B0122 current-clean qualifies the named 1M `filters.hag_nn,count=3` forced-
hybrid lane at 1.549304x pinned PDAL and the strict mapped-source/direct-
publisher composition at 2.655409x. Direct median is 41.094643% below hybrid;
tie/incomplete host repair, source rejection, atomic output, one index, and all
four Compute Sanitizer tools are clean. The fresh profile totals 3.076512
milliseconds/0.841963%, publication is 3.543095%, and no remaining bounded
surface clears the 5--10% gate. The endpoint is sufficiently optimized and
remains explicit/unselected.

B0125 current-clean qualifies the named exact 1M standalone radius-outlier
lane at 4.946130x pinned PDAL in forced hybrid and 8.107334x through the strict
mapped-source/direct-Classification route. Direct median is 38.791349% below
hybrid and proves one planner-owned 3D radius index, the exact four-byte count
download, one-byte Classification spill, record summary, no host XYZ mirror,
matching boundaries, zero fallback, and atomic output. The fresh profile totals
2.972310 milliseconds/0.777018% of wall; publication, hydration, row
materialization, and the residual wrapper ceiling are each below 5%. The
endpoint is sufficiently optimized and remains explicit/unselected.

B0126 current-clean qualifies the exact 1M same-radius
`outlier(radius=1.01,min_k=2,class=7) -> radialdensity(radius=1.01) ->
assign(UserData=1 where RadialDensity>=0.2)` graph at 9.156744x pinned PDAL in
forced hybrid and 15.896697x through the strict explicit direct endpoint.
Direct median is 0.372796165 seconds, 43.083688% below the 0.654990023-second
hybrid median. It uses one resident region and one planner-owned 3D index;
RadialDensity stays on device for assignment, the exact outlier finale stays
host-owned, and Classification/UserData are published without a PointView
writer round trip. The fresh profile totals 5.661312 milliseconds/1.518608%
of wall. Required publication has only a 5.540217% full-elimination ceiling,
and every reusable constituent is below 5%, so the endpoint is sufficiently
optimized and remains explicit/unselected.

B0127 promotes only the exact measured format-7/36-byte same-radius
composition from 250K through 4M. Clean option-free five-pair gates reach
4.717949x/15.855278x/45.178325x pinned PDAL at 250K/1M/4M; the exact 50K
direction row is excluded from the model because its 5.644% direct advantage
does not prove a stable small-input break-even. The strict selector proves the
graph/options/layout/profile/cardinality, one region/index, mapped record
summary, no host XYZ mirror, assignment execution, exact boundaries, zero
fallback, and atomic publication. The placement audit is 152/152 over 35
models. A fresh automatic 1M profile reproduces the accepted artifact and
records 5.654080 milliseconds across 20 launches, 93.724602% in the same two
already-stopped radius-count kernels. The selector exposes no new reusable
5--10% surface, so this endpoint is sufficiently optimized.

B0128 is complete from clean commit `a3e2f68fd`. The exact comparator-unique
1M forced hybrid measures 1.310245631/1.098422804 seconds pinned PDAL/PDG, or
1.192843x. Its exact-output NCU capture has 13 launches/about 0.258730
milliseconds. A bounded Release-symbol phase profile measures the filter at
245.294974 milliseconds, but the PointView permutation and exact recurrence
consume only 6.222473 and 12.672796 milliseconds; 92.296920% remains in
repeated PointView extraction, staging/setup, and Classification publication.
The separate exact read/write control is 0.300782159 seconds. This is a clear
reusable host-boundary opportunity above 5--10%, not a reason for another
kernel.

B0129 is complete from clean implementation commit `87f14b22f`. Its strict
mapped-source/permutation-publisher route measures 1.229485169 seconds pinned
PDAL versus 0.425794757 seconds direct, or 2.887507x, and is 61.235805% below
B0128's forced hybrid while preserving the exact artifact. The fresh profile
has 13 launches/0.260192 milliseconds. Canonical publication is only a
7.519276% full-elimination ceiling and would need a 66.495762% reduction to
save 5% complete wall; B0108 already rejects the apparent reusable startup
shortcut at 3.636364%. The endpoint is sufficiently optimized, explicit, and
unselected.

The B0133 measurement-only checkpoint is complete. Five exact alternating
pairs put pinned PDAL `hag_nn(count=5)` at 1.011225052 seconds; the current
default exact delegate is 1.028807390 seconds. The pinned output-shaped control
is 0.289291154 seconds, so 0.721933898 seconds/71.392011% of stock wall remains
in the stage. The exact-output CPU phase capture measures 0.063686914 seconds
in index construction and 0.680035688 seconds in the post-index query/
projection path, 8.419990% and 89.906906% of the filter respectively. This
clears a cheap reuse prototype and identifies the dominant cost; it changes no
coverage or selection.

B0139 completed that measurement from clean commit `a03b5bc19`: pinned PDAL
count seven is 1.050776026 seconds, the fresh control 0.292334145 seconds, the
stock stage surface 0.758441881 seconds/72.179214%, and the profiled
post-index query/projection tail 88.392121% of filter time.

B0140 completed that extension and its proof matrix, and B0141 qualified and
stopped the endpoint at 2.959118x.

B0142 answered that scoping question with measurement, and D0203 accepted it.

**B0150--B0152 closed the LAZ gap on both ends.** B0150 added exact decode via
the already-vendored lazperf (all 1,000,000 records of the paired fixture match
byte-for-byte). B0151 admitted LAZ *input* to placement, and B0152 admitted
option-free LAZ *output*. A fully compressed
`ref-1m.laz -> filters.normal(knn=8) -> out.laz` pipeline now runs
**1.436302 s versus 4.609780 s pinned PDAL, or 3.21x, byte-exact**, with
`planner_resident_shared_index` selected. Record access stays uncompressed-only
throughout: `compressedReader`/`compressedWriter` facts make the direct mapped
source and the direct publisher decline compressed endpoints explicitly, and a
configured reader of either extension still fails closed.

**B0153 measured generalized selection against the reference baseline and it
regressed real work.** With the LAZ gap closed, B0149's relaxation of the three
automatic-admission gates was re-landed and measured: `r1-translate` fell from
0.903899x to **0.660230x** and `r4-denoise-thin` from 1.004472x to 0.928523x,
with nothing improving. The mechanism was measured, not inferred: `r1` is never
accelerated either way — its placement is unavailable with
`non_cardinality_preserving_stage` because crop changes point count — but
without the whitelist it first pays 0.174804 s of CUDA device and profile
discovery before declining. On a 0.7 s workflow that is a quarter of the wall.
**The whitelist is load-bearing for cost, not only for proof**, and the
generalization is reverted. `compression:true` on a `.laz` sink is retained as
admitted, proved byte-identical to the option-free form.

**B0154 attributed the constant overhead.** The `pdg` dispatcher is not the
problem — `pdg --version` costs the same as `pdal --version`. The engine binary
costs **+0.022902 s** more to load than upstream's `pdal`, which accounts for
essentially all of the +0.025756 s measured on a trivial pipeline. CUDA
libraries are about 7.7 ms of that: `libcudart` +0.83 ms, `libcuda` +1.50 ms,
and **`libnvrtc` +5.42 ms**. NVRTC (114.5 MB) exists only for D0076's fused JIT
envelope, which most pipelines never enter, yet it is a `DT_NEEDED` dependency
loaded eagerly in every process. The remaining ~15 ms is the engine binary and
its PDAL linkage.

**B0155 made NVRTC lazy.** It is now `dlopen`ed at the first fused-JIT
compilation instead of being a `DT_NEEDED` dependency; seven symbols bind
behind a shim and `CUDA::nvrtc` left the link line. Engine startup overhead
versus `pdal` fell from 0.022902 s to **0.013117 s**, a 9.785 ms recovery and
43% of the overhead — more than the 5.4 ms microbenchmark predicted, because
dropping the dependency also removes its relocation work. Five of six reference
workflows improved by one to two percent, consistent with a fixed saving
against 0.24--9 s workloads; none became a win. The fused JIT still compiles a
specialization for every admitted format (310 ms of real compilation), so it is
exercised through the shim rather than skipped.

**B0156 made declining free.** `planStructureRefusal(plan)` extracts the
refusals decidable from the compiled plan alone — `unsupported_topology` and
`non_cardinality_preserving_stage` — and runs them before CUDA device and
profile discovery; `buildRuntimePlacement` calls the same function so the two
cannot drift. On `r1-translate`, which can never be placed, device discovery
falls from 0.079376 s to **0.000010 s** and the whole validation/placement
phase from 0.174804 s to **0.007508 s** — 167 ms returned, refusal reason
unchanged.

It also confirms D0214's diagnosis: re-enabling generalized selection *on top*
of cost-ordering no longer regresses anything (`r1` 0.660230x -> 0.928829x,
indistinguishable from the whitelisted 0.927326x). Generalization stays out,
but now for one reason instead of two — it is harmless yet still produces no
measured reference gain, while extending selection past what the exactness
matrices prove. Cost was the blocking objection; proof burden is what remains.

**B0157--B0160 diagnosed why real pipelines are unplaceable, and corrected the
plan twice.** B0157 found writer `extra_dims` blocked by *calibration*, not
admission: admitting it works and keeps output exact, but the placement model
predicts timing from canonical-layout record sizes, and a process gate asserts
this deliberately — so it belongs with the calibration work and the prototype
was reverted. B0158 found `non_cardinality_preserving_stage` is a catch-all
that misreports: `fusion.cardinalityPreserving` defaults to false
(`include/pdg/Plan.hpp:123`), so any stage not declaring itself preserving —
including every unsupported fallback stage such as `filters.reprojection` —
trips it. B0159 found the actual cause: every clause of the declared-predicate
exemption passes except `preferredResidency == Device`, because plan-time
residency calls `exactDeviceExpression(expression, allowCoordinateLoads=false)`
(`src/stages/Assign.cpp:600`), so **any predicate reading X, Y or Z is planned
to Host**. The identical `filters.range` is refused on `Z` and admitted on
`Intensity`. B0160 then found the hazard in the obvious fix: the preflight probe
is built with a **zero** coordinate offset
(`src/pdal/PdgResidentContext.cpp:203`) while real LAS data carries large
offsets, so an optimistic plan-time declaration could pass preflight and then
throw inside `evaluatePredicateDevice` after commitment.

**B0161--B0162 made coordinate predicates placeable.** B0161 cleared the probe
hazard: the resident lane builds its batch with the same zero coordinate offset
as the preflight probe (`PdgResidentContext.cpp:1409` and `:204`), and B0162
extended that check to every other device predicate path —
`PdgPointProgramFilter` and the fused LAS translate lane — all zero offset. So
probe and evaluator agree everywhere and an optimistic plan-time declaration
cannot become a post-commitment error. B0162 then added
`predicateMaySupportExactDevice`, which permits coordinate loads at plan time,
and wired it into the `crop` and `range` compile sites. Both now reach
`missing_calibration_model` — a genuine placement decision — instead of the
misleading `non_cardinality_preserving_stage`, with cropped output byte-exact.
`filters.expression` was deliberately not widened; it carries
`placementModel = "point-program"` and needs its own slice.

**The reference baseline did not move**, because `r1` and `r4` each carry a
second blocker: `r1` also has `filters.reprojection`, unsupported and tripping
the same catch-all (B0158), and `r4` also has `filters.sample`, which is not a
predicate at all.

**B0163 split the catch-all refusal** into `unsupported_stage` versus
`non_cardinality_preserving_stage`. Behaviour is unchanged; the reasons are now
true. It immediately corrected the plan: `filters.sample` was never a
cardinality problem, it is simply unsupported, and so is
`filters.reprojection`.

**B0164 then ablated both before scheduling either**, per D0208. In `r4`,
`filters.outlier` is **80.5%** of wall (3.683534 s) and `filters.sample` only
9.8% (0.447334 s) — and outlier **already has a qualified 21.389523x CUDA
path** (B0088). It cannot run because `sample` is unsupported and one
unsupported stage refuses the whole pipeline. The case for implementing
`sample` is therefore what it blocks, not what it costs. In `r1`,
`filters.reprojection` is 45.0% of wall but has no existing fast path and is a
PROJ/GDAL bridge candidate rather than a kernel. Note `crop` *reduces* r1's
wall by 0.66 s, so removing a filter can make a pipeline slower — read these
ablations carefully.

**B0165 disproved prefix placement and B0166 measured the wall behind it.**
The `outlier + range` prefix is only 1.014x — not accelerated on its own — so
placing a prefix could not have unlocked anything. Narrowing showed
`missing_calibration_model` for `outlier` alone on LAZ *and* on uncompressed
LAS, and even composed with `nndistance(kth,k=10)`.

B0166 found why: the calibration file has 35 stage models, and **there is no
standalone `outlier` model at all** — only `outlier-nndistance-direct-compose`.
Placement is not rejecting the shape; it has no model to judge it with. Three
other composition models have no standalone equivalent
(`eigen-family-compose`, `radius-outlier-radialdensity-direct-compose`,
`rank-optimal-compose`); only `approximatecoplanar` has both. Across the six
reference workflows, thirteen distinct stages appear and **three are covered**.
`r6` is explained too: `normal` and `covariancefeatures` each have a model,
which D0077 refuses as `mixed_calibration_models`, while
`eigen-family-compose` is keyed to B0075's three-consumer shape and so matches
neither.

**B0167 measured the standalone `outlier` ladder** with the existing D0067
protocol (`build/diagnostics/profile-work/calibrate-outlier.sh`), forced
hybrid-CUDA versus complete pinned PDAL, every row byte-exact: 0.773x at 50K,
2.870x at 250K, 6.589x at 1M, 8.446x at 2M, and **4.947x at 4M**. Host cost is
near-linear at 3896--4134 ns/point across an 80x range, so a
`host_ns_per_point` term is fittable and the break-even sits between 50K and
250K.

**No model was fitted and no selection envelope admitted** — those are
behavioral changes left for a deliberate decision.

**B0168 narrowed the cliff and corrected a claim.** B0167 said the repair
counters could be read; that was wrong — they are published by the *resident*
executor while the ladder measures the *hybrid* path, and `pdg resident` still
returns `pdal_standard_host` with null repair blocks because placement finds no
model. Extending the ladder to 8M shows device cost per point **steps** from
472.4 ns at 2M to 815.5 at 4M and then holds at 811.7 at 8M: a mode change that
switches on once, not a slope.

Four causes are excluded by measurement: the index backend does not switch
(adaptive 798.4 ns/pt at 4M versus forced grid 801.6 and forced BVH 1063.2, and
the same ordering at 2M); the step reproduces under forced grid alone
(461.2 -> 801.6); tiling is constant since `ResidentTilePoints` is 131,072 so
every size from 250K is multi-tile; and host cost per point is flat at
3896--4046 ns across the whole 160x range, so it is device-only.

**B0169 profiled the step and the leading hypothesis was wrong.** Nsight
Compute at 2M and 4M shows an identical 17 launches, so there is no
work-partition change. The gather kernel *is* superlinear — 30.851 ms to
133.582 ms, 4.33x for 2x the points — but that is not what steps the ladder.
The process grew 2.248678 s while all kernels grew 0.103291 s: **kernels are
4.6% of the regression**, and at 4M all kernels together are 4.2% of wall. The
other 95.4% is host-side work in the hybrid wrapper.

That returns the investigation to exact host repair, where B0167 first pointed.
The family has form: B0089 found a 4M incomplete-mean host-repair cliff and
B0090/B0091 fixed the NNDistance side, leaving the standalone statistical path
unaddressed.

**B0170 attributed part of the step without touching code.**
`PDG_KNN_DEVICE_SHELL_BUDGET` controls how many rows go to exact host repair,
and the repair path builds a full host `KD3Index` per call
(`src/pdal/PdgNeighborhood.cpp:3246`). Raising it from 32 to 256 cuts 4M by
**16.2%** (810.4 to 679.2 ns/pt) but 2M by only 6.1% (472.0 to 443.4) —
repair is real and size-dependent. It is still not the main cause: at budget
256 the 4M cost remains 1.53x the 2M cost per point.

Running total for the step, all measured: device kernels 4.6% (B0169), exact
host repair roughly 16% at 4M, and the remaining majority unattributed and
host-side. Excluded so far: index backend, tiling, input density, launch-count
or work-partition change, and repair as a sufficient cause.

**B0171 found that the hybrid and resident outlier lanes are different
implementations.** Instrumentation added to `tryCudaStatisticalOutlier`
produced no output, because that function has one call site
(`PdgOutlierFilter.cpp:370`) inside the `requireResidentExecutionContext()`
branch. The hybrid path calls `tryStatisticalCuda`, a private member at
`PdgOutlierFilter.cpp:241`. The instrumentation is reverted.

This invalidates an assumption under B0167: that ladder used
`--candidate-mode hybrid-cuda`, so it measured `tryStatisticalCuda`, while a
placement model governs selection of the *resident* executor, which runs
`tryCudaStatisticalOutlier`. **The ladder measures a different implementation
from the one a model built on it would govern**, and whether their cost curves
match was never checked. It also explains B0168's dead end: the repair counters
are resident-only because the repair code carrying them lives in the resident
implementation.

A concrete cause for the remaining step is now visible:
`tryStatisticalCuda` allocates a fresh pinned host resource and a fresh device
resource **per call** (`PdgOutlierFilter.cpp:255-262`), where the resident lane
reuses planner-owned allocations. Page-pinning cost scales non-linearly with
size, which fits a host-side step that switches on and holds flat per point.

**B0172 added the missing telemetry and disproved B0171's hypothesis.**
`PDG_DEBUG_HYBRID_OUTLIER_PHASES` now times `tryStatisticalCuda`, stats-only
and off by default, following `PDG_DEBUG_FUSED_JIT`. The per-call pinned
resource costs 1 microsecond at both 2M and 4M because it allocates nothing
until used; `host_gather` grows 0.034 s and `device_resource` 0.024 s, so the
three phases together are **2.6%** of the 2.248678-second step.

Running attribution, all measured: device kernels 4.6% (B0169), exact host
repair ~16% at 4M (B0170), allocation and host gather 2.6% (B0172). Roughly
three quarters remains unattributed and is downstream of the current marks.

**B0173 closed the attribution: the step is a host KD3 rebuild.** Extending
the telemetry through gather, status scan, repair and finale shows that at 2M
the repair branch never executes — no incomplete rows, so `kd3_build` produces
no mark at all — while at 4M incomplete rows appear and the lane builds a full
host `KD3Index` over four million points at **1.622082 s**. Against a
2.248678-second step that is **72%**; with device kernels at 4.6% the
attribution is closed.

**This is B0089's 4M incomplete-mean repair cliff on the path B0090/B0091 did
not fix.** Those slices gave NNDistance a bounded parallel repair from resident
coordinates and qualified 4M at 48.758796x; the statistical outlier hybrid lane
never received it. It also explains B0170 exactly: a deeper shell budget leaves
fewer incomplete rows, but one remaining row still pays the whole KD3 build,
so the saving was real but partial.

**B0174 found the cliff is one incomplete row.** At 2M there are zero; at 4M
there is exactly **one** point of four million with an incomplete bounded
search, and repairing it costs a full host `KD3Index` build over the whole
view — **1.646093 s**, 72% of the step.

The repository solved this before on another lane: B0049 measured a
2.061175139-second KD3 repair for one incomplete NNDistance row, and B0064's
bounded multi-block kernel cut it to 2.232480 ms. **The resident outlier lane
would already handle this case** — its device repair is gated on
`neighbors <= 16 && incompleteRows <= 16`
(`PdgNeighborhood.cpp:3171-3174`), and here `neighbors` is 9 and
`incompleteRows` is 1. Only the hybrid implementation (a separate code path,
B0171) falls back to the rebuild.

This changes the recommended slice. **Do not fit the B0167 ladder**: above 2M
it measures one unrepaired point, not a scaling curve, so a model fitted across
it would encode a 1.6-second host rebuild as device cost for a data-dependent
condition. And under D0208 the hybrid lane may not deserve a fix at all — it is
reached by `PDG_EXPERIMENTAL_CUDA_HYBRID`, not by the default path a user runs.

**B0175 tried to reach the resident outlier lane without new machinery and
could not.** B0092's composition is documented as automatically selected, so
reconstructing it from the record should have reached it. The reconstruction —
`outlier(statistical, mean_k=8, multiplier=2, class=7)` then
`nndistance(kth, k=10)` then an assign to `UserData`, default LAS — measures
1.002x at 2M and 1.000x at 4M. **A faithful-looking reconstruction of a route
qualified at 20.300698x runs at parity**, so it did not match the envelope.

Either the reconstruction differs in some detail the prose record does not
capture, or the envelope is narrower than described. The evidence does not
separate them, and both point at what B0166 measured: admission is keyed at a
level of detail prose cannot reliably reproduce.

**B0176 audited four "automatically selected" envelopes through the public
command.** On the 1M fixture: `lof` **7.901x** (record 9.597x at 4M) and
`approximatecoplanar` **4.075x** (record 4.231x at 1M) both engage, the latter
reproducing its recorded figure almost exactly. `nndistance` runs at 0.997x and
`mortonorder` at 1.029x — neither engages.

`nndistance` is the informative one: through `pdg resident` it reports
`planner_resident_shared_index` with placement available, so placement accepts
the shape while automatic selection does not. Placement availability and
automatic selection are separate gates and this route passes one and fails the
other. `mortonorder` fails earlier, at placement, with
`missing_calibration_model`.

This does not show the two failing routes are broken — B0045 and the Morton
work were measured on exact shapes whose detail may not survive prose (B0175).
It shows they are not *reachable* by reconstruction, and reachability is what a
user gets.

**B0177 completed the reachability audit: three of seven automatic envelopes
engage.** Through the public command on 1M: `assign`+`ferry` **7.492x**, `lof`
**7.901x**, `approximatecoplanar` **4.075x** (within 4% of its 4.231x record).
Not engaging: `nndistance` 0.997x (passes placement, fails selection),
`estimaterank`+`optimalneighborhood` 0.982x and the eigen family 1.025x (both
`mixed_calibration_models`), and `mortonorder` 1.029x
(`missing_calibration_model`). `docs/stage-coverage.md` now carries a
reachability section recording all seven.

The best news is that the most-used filters in PDAL — `assign` and `ferry` —
are among the reachable ones, at 7.492x with no options or environment
variables.

**B0178 withdrew B0177's recommendation and found something better.**
Generalizing composition models over consumer count would extend an envelope
without measurement — B0157's gate objects to exactly that. Narrowing by
measurement instead: one neighborhood consumer is admitted (`normal`, `normal`
+ assign both reach `planner_resident_shared_index`), two or more never are
(`mixed_calibration_models`), and `eigen-family-compose` matches none of the
reconstructions.

**The audit resolves into two independent gates** — placement decides whether a
shape has a usable calibration, selection decides whether the public command
uses it — and the interesting cell is *placement admits, selection refuses*.
`normal` + assign measures 4.4416 s pinned PDAL, 4.4609 through
`pdg pipeline`, and **1.2273 through `pdg resident`: 3.62x, byte-exact.**
`nndistance` sits in the same cell (B0176).

**B0179 discharged the selection proof burden for the isolated class, and
changed no behaviour.** `tests/differential/resident_selection_matrix.py`,
registered as `pdg_resident_selection_matrix_cuda_exact`, crosses five
neighborhood consumers (`normal`, `eigenvalues`, `covariancefeatures`,
`nndistance`, `approximatecoplanar`) with three assignment targets of differing
physical width. All fifteen cases are byte-identical between pinned PDAL and
`pdg resident` — the executor a generalized selection would choose. The matrix
fails rather than skips when a case is exact but ran `pdal_standard_host`,
which caught its own first fixture sitting below the 250,000-point placement
floor; it now generates a 270,000-point fixture from the pinned oracle and is
self-contained.

**No selection gate was flipped.** That is a user-facing default-behaviour
change and is deliberately left as a separate decision.

**B0180 flipped the three gates, measured, and reverted.** With the cost
objection removed (B0156) and the proof objection discharged (B0179), the
generalization was tried and the full reachability audit re-run through the
public command. It **recovers** `nndistance` (0.997x -> **7.338x**) and
`normal`+assign (0.996x -> **4.001x**), leaves `lof` and `approximatecoplanar`
unchanged, and **regresses `assign`+`ferry` from 7.492x to 1.041x** and
`mortonorder` from 1.029x to 0.742x. Everything stayed byte-exact.

Cause: those pipelines were already served by the **fused point-program lane**,
a different and faster mechanism. Once selection stops requiring a named shape,
the resident executor is chosen for graphs the fused lane handled better and
preempts it. Nothing became incorrect; the planner started choosing a worse
path.

**This objection is different in kind from the previous two** — generalized
selection is not merely unproven on new shapes, it is actively wrong on shapes
that already had a better answer, and no amount of exactness coverage would
have revealed it. It also validates auditing beyond the reference baseline:
**none of the six reference workflows contains a bare `assign`+`ferry`
pipeline**, so the baseline alone would have shown the change as harmless and
it would have shipped.

**B0181 landed the ranked generalization B0180 called for.** Automatic
admission is generalized *and ranked*: a plan declaring neither a neighborhood
query nor a grid product is a pure point program the fused lane serves faster,
so the resident executor declines it. Results, all byte-exact:
`nndistance` + assign **0.997x -> 8.641x**, `normal` + assign
**0.996x -> 3.990x**, `lof` 8.016x and `approximatecoplanar` 4.095x unchanged,
and crucially `assign` + `ferry` holds at **7.353x** against its 7.492x
baseline over seven runs, with `pdg resident` reporting
`outside_calibration_envelope` for it — the fused lane still owns it.

**It does not move the release criterion and is not claimed to.** The six
reference workflows are unchanged within noise because each is blocked by
something else: unsupported stages (B0163), writer options (B0147), or
`mixed_calibration_models` (B0177). It was retained because it costs nothing
measurable, is byte-exact everywhere tested, passes every gate including
B0179's matrix, and recovers two ordinary feature-extraction shapes.

**B0182 read the eigen-family matcher and still could not reach it.**
`measuredEigenFamilyRegion` (`src/plan/RuntimePlacement.cpp:190`) requires six
stages, `normal(knn=12)`, `eigenvalues(knn=12, normalize)`,
`covariancefeatures(knn=12, raw, Dimensionality)`, and **one** assign stage
carrying exactly `Classification = Linearity * 10`,
`Intensity = Curvature * 1000`, `UserData = Eigenvalue0 * 100`. B0177's
reconstruction used `knn=8` and three assign stages, which explains that miss —
but transcribing every literal from the source **still** yields 0.984x and
`mixed_calibration_models`. A compiled-plan dump confirms every externally
visible condition holds: six stages, one region, all four filters native,
device-preferred, region 0, `neighbors == 12`, `mode == Raw`.

**This is stronger than B0175's finding.** A qualified route could not be
reconstructed from its prose record; this one cannot be reconstructed from the
source that defines it. Such an envelope is not a route users reach — it exists
for the exact artifact it was measured on.

That reframes work-order item 3: "broaden calibration coverage" is not mainly
about adding models, since `eigen-family-compose` and `rank-optimal-compose`
already exist and are measured. It is that the **matchers** are keyed so tightly
the models are unreachable.

**B0185 closed reachability** — all six compose envelopes reach; none was ever
unreachable (D0215). Standing rules: copy calibrated pipelines out of
`test/data/pdg/placement-calibration-sm89.json` **verbatim**, and never read
`--stats` from `pdg resident` to explain a `pdg pipeline` timing.

**B0186 baselined D0208's real criterion:** all six reference pipelines
byte-exact and at or below parity. Two causes — a fixed ~17 ms per-pipeline
admission tax, and for r6 an admission *policy*, not a missing kernel.

**B0187 fitted the missing composition model** for `normal` +
`covariancefeatures` (D0216): measured 50K–4M ladder, byte-exact,
1.458x/3.871x/6.351x/7.147x/7.260x, predicting within 3.63%. The mixed-models
rule was not relaxed.

**B0188 made option-free `.laz` readers native** (D0217). Reader nativeness had
required an uncompressed filename on the stated grounds that `native` also
authorizes memory-mapping; that claim was checked and is false — mapping lives
solely in `DirectResidentPointTable` under `directResidentLasSource`, which
carries its own `!compressedReader` guard, and `FileView::pointRecord` throws
on compressed data. Headroom was measured first: pinned `laz -> las` translate
(0.309 s) is *faster* than `las -> las` (0.326 s), so decode is not a cost,
while encode is (+0.112 s) — hence readers only. The composition now reaches
compressed input at **4.079x / 6.341x / 7.221x** byte-exact, within its own fit
error of the uncompressed rows. 774/774 CUDA and 485/485 sanitizer tests pass.

**r6-features is 0.9925x and the diagnostic now names exactly one blocker:**
`writers.las` is `native=false`. Reader, `normal` and `covariancefeatures` are
all native. r6 writes `.laz` with `extra_dims=all`, so two writer properties
remain, and unlike decode both are **real unmodelled costs** — encode measured
at +0.112 s.

**B0189 corrects B0188's own next-task premise.** B0188 recorded
`extra_dims=all` as the cheaper writer blocker because it has no encode cost.
Measuring first shows it has a different unmodelled cost: the output record is
**100 bytes per point, not 36** (measured 1,965 B header + 1M x 100 B), because
every computed normal and eigen feature is serialized.
`runtimeFacts->outputRecordBytes` is assigned from one of two compile-time
constants (`DefaultLasOutputRecordBytes = 36`, `ExtraDoubleLasOutputRecordBytes
= 48`) at both assignment sites in `src/cli/ResidentPipeline.cpp`; nothing
derives it from the writer's layout. Admitting that sink as native today would
understate the terminal spill boundary by 64 B/point, about 64 MB at 1M points.

So **both** r6 writer blockers carry an unmodelled term, not one. No code was
changed for B0189.

**B0191 derived the record size and withdrew B0190's escalation.** B0190
reported the derivation blocked on `las::pdrfDims` being hidden by
`-fvisibility=hidden`, and escalated a choice between patching upstream's build
surface and compiling its private LAS subsystem into the engine. **Both were
unnecessary.** pdg already owns the same table: `formatCarriesField`, the
per-format LAS field predicate `LasFerry` has used privately since it was
written. It was exported. No upstream change, no duplicated list (D0218).

`outputRecordBytes` is now derived from the writer's actual layout — PDAL's
public layout gives names and types, the plan's `DimensionRegistry` resolves
them, and `Dimension::size` is charged for whatever the format does not carry.
`--stats` reports `input_record_bytes` and `output_record_bytes`. Verified
against the measured file: 36/36 for a plain sink, and **100** for an
`extra_dims=all` sink, exactly the 100 B/point B0189 measured. 774/774 CUDA and
485/485 sanitizer tests pass.

**Admitting the sink was written, measured at 6.227x byte-exact, and
reverted.** It fails two process gates: `pdg_resident_pipeline_process` and
`pdg_resident_sort_direct_gpu`, the latter with `direct resident LAS boundary
has no logical transfer`. A native writer declares a fusion anchor whose
pack/summarize machinery changes the plan's residency boundaries, and the
direct routes' boundary facts do not survive it. Isolating the halves confirmed
the derivation is not at fault. Per B0157 the change goes back rather than the
gate being edited.

**B0192 located the mechanism behind B0191's revert.** Restoring the admission
and dumping the failing plan (`filters.sort` then `writers.las(extra_dims=all)`)
shows two boundaries: upload 0->1 at 8 B/point, and spill 1->2 at **0
B/point**. `directResidentLasBoundaryFacts` throws on exactly that row.

Spill bytes are derived from the region's *column lifetimes*
(`src/plan/Pipeline.cpp:2135-2175`), accumulated from each device stage's
`deviceLiveBefore`, `descriptor.writes` and `deviceLiveAfter`. `filters.sort`
writes no columns — it only reorders — so the region contributes no lifetime
and the spill is empty. While the writer was non-native the question never
arose, because the sink took the fallback path that moves the whole record.

**So this is a planner-semantics slice, not an accounting patch.** A native
`extra_dims=all` writer consumes every live dimension and nothing in the plan
says so: writers never populate `descriptor.reads`. The stakes are real — a
wrong statement makes the boundary silently under-transfer instead of failing,
which is worse than today's hard throw. The throw is doing its job and the
revert stands.

**B0193 tested two fixes and refuted both**, which narrows the blocker
usefully and corrects B0191's attribution.

*The failures decompose.* B0191 blamed both gates on the sink admission. With
only the writer-nativeness change in `src/plan/Pipeline.cpp`,
`pdg_resident_pipeline_process` **passes** — it was broken by the other half,
`lasHeaderFacts(plan, true)`. The admission alone leaves exactly one real gate
failing, `pdg_resident_sort_direct_gpu`, plus one label assertion.

*Refuted 1 — the writer's declared reads.* Native writers read `filterWrites`
intersected with *standard* dimensions (`src/plan/Pipeline.cpp:1819-1826`), and
adding custom dimensions looked necessary. It is not: the features shape
already reached 6.227x because `Linearity`, `Curvature` and the rest are
standard.

*Refuted 2 — falling back on an empty boundary.* Marking
`boundary.dimensions.empty()` as `fallback` should have restored the exact
full-record semantics the boundary had while the sink was non-native. Measured:
sort still fails **and** `PlacementModel.UsesInputAndOutputCardinalityAtEach
Boundary` breaks too. Reverted.

**What is left is one stage shape.** The blocker is a device region whose
stages write no columns at all — `filters.sort` reorders and writes nothing —
so the product handed to a native sink is a record *permutation* that the
direct output machinery publishes, not a set of SoA columns. The plan cannot
describe "the same columns, in a new order," so the sink has nothing to name;
and forcing a fallback fails because the direct sort route wants its own
publication path, not a full-record pack.

**B0203 opened work-order item 6 and found the session's largest regression.**
Item 6 frames r2 and r3 as SMRF problems. Decomposing them under pinned PDAL:
SMRF is **45% of r3** (+290 ms) but only **20% of r2**, which is dominated by
**`filters.hag_nn` at 49%** (+765 ms). Item 6's r2 framing needs correcting.

Checking that turned up something bigger. On a LAS-to-LAZ pipeline containing
`filters.smrf`, pdg is **+246.3 ms ± 13.9 slower in 5/5 paired runs**,
byte-exact. Stage isolation puts the entire deficit on SMRF: read+write −26.2,
`hag_nn` only −0.8, `smrf` only **+246.3**, both together +251.5. It survives
every toggle — +221.7 default, +235.6 with `PDG_DISABLE_HYBRID` and
`PDG_DISABLE_NATIVE`, +230.0 with `PDG_ENABLE_CUDA=0` — so it is not hybrid
substitution, not the native lane, and not CUDA.

r2 and r3 escape it only because their writers (`writers.gdal`,
`extra_dims=HeightAboveGround=float32`) leave the engine nothing to place, so
it `execv`s the oracle and upstream's SMRF runs. **Any LAS-to-LAS pipeline with
SMRF that the engine executes itself pays ~230 ms** — a common
ground-classification shape the reference set happens not to contain.

**B0204 refuted that mechanism.** `HybridSmrfStage` is `"filters.pdg_smrf"`
(`include/pdg/Hybrid.hpp:41`), a distinct name, so pdg's filter never shadows
upstream's; the only substituter is the hybrid rewriter
(`src/plan/Hybrid.cpp:1640`), and disabling it does not change the deficit.

Six hypotheses are now eliminated **by measurement**: hybrid substitution, the
native lane, CUDA, the fork's own PDAL build (its `pdal` CLI links the
identical `libpdalcpp` and is at parity, −15.5 ms ± 30.4), stage-name
shadowing, and I/O format (+243.8 laz→laz, +235.0 laz→las, +264.9 las→las).
`pdg-engine` itself is +243.2 ms ± 13.9, slower in 7/7.

**B0205 explained it with two zero-code instruments.**
`PDG_ORACLE_PDAL=/bin/true` measures pre-delegation work: **236.1 ms** on the
SMRF pipeline against 17.3 ms on a plain translate, so the engine does the work
and then hands the pipeline over unchanged. `AdmissionTrace` named the phase:
structural refusal is reached at 2.22 ms and the attempt declines at 174.5 ms,
so **~172 ms is CUDA device/profile discovery plus placement, paid and
discarded**.

The guard misses it because `filters.smrf` *is* native and device-preferred —
it passes every topology and cardinality check — but has **no placement model**,
so `missing_calibration_model` only fires inside `buildRuntimePlacement`, after
discovery. B0156/D0214's hoist covered topology, not model existence.

**Landed:** a `NoDeviceCandidate` refusal in `planStructureRefusal` for plans
with no device-capable stage (775/775 and 486/486 pass). It does not help SMRF.

**Written, measured, reverted:** hoisting the model-existence check as well. It
saves **167 ms** (2.14 ms against 169.5 ms) but breaks four tests including the
process gates `pdg_resident_outlier_direct_gpu` and
`pdg_resident_radialdensity_direct_gpu`, because the direct outlier and
radial-density regions legitimately carry empty models and the plan-only
exception is not equivalent to `buildRuntimePlacement`'s facts-gated one. Per
B0157 the change went back.

The single next task is that hoist done correctly: **167 ms returned to every
pipeline whose device-candidate stages have no model**, which includes any
LAS-to-LAS pipeline containing `filters.smrf`. It needs an exception matching
`buildRuntimePlacement`'s facts-gated behaviour exactly, not the conservative
plan-only approximation tried here.

Under D0208 this outranks the remaining tax work: ~250 ms is an order of
magnitude larger than the ~15 ms engine load, and it is a regression against
the oracle rather than a missed win.

**B0201 corrected B0200's metric and produced the baseline that supersedes
every earlier reference figure.** B0200 shipped the *range* of paired speedups
as the uncertainty; a range grows with sample size, so more evidence made a
difference look less resolvable. It is now the standard error of the paired
ratios with a two-s.e. advisory interval, which shrinks as sqrt(n) (r1: ±0.0246
at 3 pairs, ±0.0086 at 15). A minimum-based estimator was tested and rejected
at 93.5% of the median's invocation range.

**Current baseline, nine pairs each — quote this one:**

| Pipeline | Paired speedup | ±2 s.e. | Verdict |
| --- | ---: | ---: | --- |
| r5 COPC query | 0.9280x | 0.0090 | measurably slower |
| r1 translate | 0.9425x | 0.0109 | measurably slower |
| r3 DTM | 0.9741x | 0.0065 | measurably slower |
| r2 ground normalize | 0.9945x | 0.0119 | parity |
| r6 features | 0.9976x | 0.0090 | parity |
| r4 denoise + thin | 0.9995x | 0.0041 | parity |

**Three of six are measurably slower, three are at parity, none is faster.**
That is D0208's gap at a confidence the harness supports.

This reinstates B0198: r5's deficit is real and the largest of the six, so
B0200's withdrawal was an artefact of the broken metric. **B0202 then closed
the attribution** with 15 alternating pairs: `pdg-engine` run directly carries
+15.3 ms ± 2.5 against the dispatcher's +14.4 ms ± 2.5, slower in 15/15 both
ways, so r5's deficit is entirely the engine path and the dispatcher hop adds
nothing measurable. Disabling hybrid makes r5 *slower* (+17.2 ms), refuting
B0199's noisy hint that hybrid cost r5 — hybrid costs **r1** (15/15 pairs) and
not r5.

**B0200 checked whether D0208's own harness can resolve what it reports, and
mostly it cannot.** Five identical invocations at `--runs 3` gave r1 as
0.9342/0.9279/0.9492/0.9313/0.9248x — a 2.4 point range on a ~7 point deficit,
from a script printing one median. The harness now reports paired speedups,
their spread, pairs won, and a `resolvable` flag (false when the spread exceeds
the distance from parity). Reporting only; exactness and the median are
unchanged.

End-of-session baseline at `--runs 5`: r1 **0.9256x** (spread 0.0267,
*resolvable*), r2 0.9909x (0.0526), r3 0.9760x (0.0385), r4 0.9957x (0.0243),
r5 0.9419x (0.0687), r6 1.0016x (0.0127) — **five of six are not
distinguishable from parity.**

Corrections to the record: B0186's six figures were over-precise, and only the
weaker claim survives — *none of the six is faster*. **B0198's attribution of
r5's deficit to the engine load is withdrawn**, being the same size as r5's own
noise; B0198's r1 attribution stands (35 ms deficit against a 12 ms spread).
B0188 and B0197 declined to claim reference movement at their scale, which was
correct and is now quantified.

**Read the `resolvable` flag before quoting any reference speedup**, and add
runs when it is false rather than trusting the median. The default was left
alone deliberately: the right sample size differs per pipeline (r5 needs far
more than r6), and a larger fixed default would hide that rather than expose
it.

**B0199 found a second mechanism behind the reference deficits.**
`tryHybridPipeline` substitutes pdg filters on **structural grounds alone**
(`src/cli/HybridPipeline.cpp:200-242`: `replacementRegions`, linearity, stable
order, CUDA availability, a point-count probe for some stages). There is no
calibration model and no host-versus-device comparison — the opposite of the
resident path, which refuses when its measured model says host wins (D0077),
and exactly what D0208 was written against.

On r1 it loses: 516.8 ms with hybrid against 498.3 ms without, **+18.6 ms
median paired delta, slower in 15 of 15 alternating pairs**, byte-exact. The
rest of r1's deficit is B0198's image load.

**Method warning, learned the hard way here.** A median-of-3 pass over all six
reference pipelines gave +24.1/−41.2/−1.6/+18.4/+1.9/−43.6 ms and looked like a
clean story. Re-running at seven samples showed single-run spreads of
68.9–463.3 ms: every one of those deltas was inside its own noise, and they are
withdrawn. Only a paired alternating protocol resolved r1. **Reference
pipelines need paired alternating runs, not medians of three** — the spread on
r6 alone is 463 ms.

So r2, r4 and r6 are **unresolved**, and no claim is made about them. That is
itself the argument for the fix: an ungated path's effect is unmeasured by
construction. Recommended is to gate hybrid substitution on a measured model as
placement is — a calibration program, not a patch, since B0171 established the
hybrid and resident executors are different implementations and resident models
do not transfer.

**B0196 refuted B0195 with an instrument**, and B0197/B0198 acted on the truth.
`AdmissionTrace` (opt-in via `PDG_DEBUG_ADMISSION_PHASES`) shows the automatic
admission attempt declines in **0.17 ms**, not the ~19 ms B0195 inferred from
source. The tax is two dynamic-link cycles: the dispatcher routes to the
engine, the engine's paths all decline, and `runOracle` `execv`s pinned PDAL —
so an 11 MB image with 76 shared libraries including `libcudart`/`libcuda`
loads only to hand the work back.

**B0197 fixed it for bare translates** (D0219): a pipeline of exactly two LAS
file stages with no filter now routes straight to the oracle. Tax 19.34 → 0.96
ms at 10k, byte-exact; 775/775 CUDA and 486/486 sanitizer tests pass. It helps
no reference pipeline, and the record says so.

**B0198 measured why r1 and r5 are slow, and it is the same waste.** Their
deficits are *entirely* the engine path: r1 +34.5 ms deficit against +37.9 ms
for `pdg-engine` alone, r5 +17.3 ms against +17.5 ms, with admission declining
in 0.23/0.17 ms. Neither is slower for any algorithmic reason.

**B0197's rule cannot be widened to cover them**, and that is the key
constraint: `filters.crop` (r1) and `filters.stats` (r5) are candidate stages,
which is exactly the signal that the hybrid path might substitute a pdg filter.
The classifier is not wrong; it cannot know the substitution will not pay until
the engine is already loaded.

The architectural next task: the decision "will the engine help?" costs 0.2 ms
but is taken *after* the 17–38 ms load. The dispatcher (`pdg`, 4 shared
libraries, 1.8 MB) could take it instead if it linked `pdg_core`'s plan
compilation, which has no CUDA dependency — moving a 0.2 ms decision ahead of
the load, and generalizing instead of needing a hand-maintained shape list. It
is a build and layering change and deserves its own slice. Expect r1 0.931x and
r5 0.952x to reach roughly parity: this removes a self-inflicted loss, it does
not create a win.

**B0195 identified the tax structurally.** `pdg pipeline` has no path of its
own: `main` calls `tryAutomaticResidentLasPipeline`
(`src/cli/ResidentPipeline.cpp:4005`), which rewrites the arguments into a
`resident` invocation and runs `runResidentPipelineImpl(...,
automaticAdmission=true)` — the same function `pdg resident` runs, one boolean
different. On decline it returns `nullopt` and `main` falls through to read,
prepare and execute the pipeline **again**. The tax is a declined admission
attempt that is then thrown away, which matches every B0194 number: fixed, paid
even by an unplaceable shape, and invisible at 4M only because the translate
path wins by more than the attempt costs.

**Do not "fix" this with a blanket early exit.** Bailing before plan
compilation whenever no named envelope matches would remove the tax from
translates and also forfeit every automatic win that is not a named envelope —
including B0188's LAZ features route at **6.351x**, which reaches the resident
executor through this same attempt. The attempt is load bearing; only its cost
when it declines is waste. The question for telemetry is therefore narrower
than "where does the time go": **at what point is the decline already
determined**, so the attempt can stop there. The two candidate cut points in
the source are before `compilePipeline` (`:2822`) and before the duplicated
`PipelineManager::prepare()` (`:2885-2892`); B0194's external numbers cannot
distinguish them.

**B0194 decomposed the admission tax that B0186 left open** (a separate thread
from the r6 chain, and one that touches every pipeline). Pure translate: the
tax is **fixed**, 19.34 ms at 10k, 20.05 ms at 250k, 31.02 ms at 1M, and
**1.74 ms at 4M** where the fork's translate path wins it back.

Measured, not assumed: it is *not* startup (engine binary +1.44 ms, dispatcher
`execv` hop +1.39 ms on a minimal command), *not* CUDA discovery alone
(4.85 ms), and *not* placement (a structurally unplaceable shape still pays
16.14 ms). **~13 ms is unattributed**, and two of my attributions are refuted
in the record: differencing whole-pipeline runs wrongly assigned ~18 ms to
binary loading, and a duplicated `PipelineManager::prepare()` cannot explain it
because pinned PDAL's entire prepare-plus-execute at 10k is only ~4.8 ms.

External differencing has run out. `--stats` phase telemetry exists only on
`pdg resident`, whose 146 ms on the same translate describes a different
admission route and would mislead if quoted. **Give the public `pipeline` path
the same phase telemetry** — D0215's rule applied to latency rather than
refusals. That is the recommended next task on this thread, and it is
independent of the r6 chain below.

The single next task is that publication contract: decide how a
permutation-only device region reports its product, then admit the sink. Do
**not** append a third local guess to the boundary rules — two have now been
measured and refuted. Reproduce with `(supportedOptions || extraDimensionsAll)`
in `src/plan/Pipeline.cpp`, and note `lasHeaderFacts(plan, true)` is a separate
change with its own failure. Payoff is unchanged and measured: **6.227x
byte-exact** on features-plus-extra-dims, with `output_record_bytes` already
derived correctly as 100.

The remaining HAG
family (`filters.hag_dem`, `filters.hag_delaunay,count>=4`, and
`filters.hag_nn,count>=8`) and the raster/overlay catalog remain goals, but
stage count no longer determines their order.
D0092's exact proof remains proportional to target cells times populated cells;
do not generalize the controlled B0030/B0031 results to pathological frames,
semantic surface reuse, or cross-kind stages.
Feature-kernel prologue/epilogue fusion into neighborhood producers (D2's open
half) and a `boundary_batch` calibration remain the two known engine-side gaps.

## Historical slice records (P1.5, superseded by DECISIONS.md)

The three sections below are kept for the operational detail they carry
(fixture facts, gate shapes, oracle resource needs). Their forward-looking
statements — deferred work, "next" items, per-slice counts — were resolved
by later decisions; trust `DECISIONS.md` and the sections above.

## D0064/D0065 result

D0064 calibrates the fused-endpoint ordered executor as the measured
`ordered-point-program` placement model (host_fixed ≈ 9.73 ms,
host_ns_per_point ≈ 366.6, minimum_device_points 1,000,000) from clean
`ordered-expression-{1m..22m}` rows whose raw reports live under
`build/benchmarks/`. The audit is 58/58 and the calibration-match flag for
`direct_ordered_las` flipped true the proper way. D0065 marks the seven
reboot-destroyed D0049 raw reports
`physical_file: lost-to-environment-2026-08-08-host-reboots`;
`verify_placement_calibration.py --require-all` reports
`verified 57/64 unique raw reports; missing 0; waived-lost 7` and still fails
on any unmarked missing file. Raw reports must be stored under
`build/benchmarks/`, never `/tmp`.

## D0066 result (V2)

The planner rewrites a declared approximatecoplanar region into a
`HybridApproximateCoplanarStage` wrapper plus Ferry/Assign point-program
bridges carrying `pdg_resident_context`/`pdg_execution_region`; runtime
placement anchors the region on the measured `approximatecoplanar` model with
point-program bridge content at zero incremental cost (D0055 rule). The
resident context runs the region delegated: whole-view budget
`estimatedDeviceBytes(N) + N × 146` scratch bytes (96 cached eigensystem +
48 covariance scratch + 2 status), a hand-built single-tile/one-lane
schedule, region bracket events, and logical `boundary.bytesPerPoint`
boundary accounting. The exact D0045 shared-index path is mandatory — no
host fallback.

The v2 process gate (reader → approximatecoplanar(knn=8) →
ferry(Coplanar=>UserData) → writer over the hash-pinned fixture below)
reported executor `planner_resident_shared_index`, pipeline class
`whole_view_neighborhood`, item count 21,970,934, one tile, one lane, peak
lane 5,448,791,632 bytes within the 18,386,190,336-byte budget, and output
byte-exact against the pinned oracle (sha256
`46d9fc9f6df539f6da96b1f8c2d3cf8b298e2974a10256e03429e81be51fc142`,
point-in-time evidence — `writers.las` embeds the UTC creation day-of-year).
Per-transfer H2D/D2H observation at the neighborhood executor's memcpy sites
was deferred here and delivered in D0077.

Fixture:

- Path: `build/bench-data/download-3101-nosrs.las`
- SHA-256: `2ea7a921a6f45e0058ee2e491136e80e6450d7920d080ce94f4924d0a1ecd8f9`
- Size: 790,953,999 bytes
- Points: 21,970,934

## D0067 result (V3)

The engine primitive `knnLofValues` reproduces filters.lof's three passes
over one retained shared-kNN adjacency in upstream's literal binary64
operation order and reports per-row `neighborStatus` so consumers repair
exactly the two-hop tie closure on the compatibility KD3Index.
`PdgLofFilter` replicates upstream bit-for-bit on both paths — including
the stateful per-filter()-call minpts increment — and the planner compiles
`filters.lof` onto the measured `lof` model (`device_fixed_ns 583604713.6`,
`host_ns_per_point 7519.6`, floor 250,000 points; audit 64/64; six clean
forced-hybrid rows `build/benchmarks/lof-cal-*-86f2aeaf2.json`, 2.8–5.2x
over pinned PDAL). The v3 gate (reader → lof(minpts=10) →
assign(`UserData = 1 WHERE LocalOutlierFactor >= 1.2`) → writer over the
fixture below) reported `planner_resident_shared_index`,
`whole_view_neighborhood`, one tile/one lane, peak lane 5,668,500,972
bytes within budget, byte-exact output (sha256
`e21bbbec2aab920a0c05a3e7550d7a30046c525c1489f312ee59256b59839c9f`,
point-in-time). The v3 oracle needs roughly 22 GB host RAM (upstream's
adjacency map at 21.9M points) and a 3600 s CTest budget.

## Validation completed at this tip (a58e94628)

- Physical SM-89 CUDA Release: 487/487 with the corpus and fixture-gated
  skips. Caveat: exporting the 790 MB gate fixture as `PDG_LOCAL_LAS_FILE`
  to the whole suite activates the optional <=128 MiB corpus smoke test
  and fails it by design; run the corpus test without that export (it
  then skips) and the GPU gates with it.
- Host Debug and the leak-disabled ASan/UBSan tree: 335/335 each.
- GPU process gates: v2/v3/v6 re-run byte-exactly after D0077 (119.9 s,
  235.1 s, 371.7 s); v1/v4/v5/v7/direct unchanged since D0071.
- Compute Sanitizer memcheck, initcheck, racecheck, and synccheck: clean
  over the spatial-index, neighborhood-column, LAS point-program, and
  wrapper tests as each slice touched them.
- Frozen placement audit: 105/105 at winner accuracy 1.000 over 28 stage
  models.
- 22M B2 lanes: every resident neighborhood stage byte-exact (table
  above); the JIT-forced fused lane byte-exact at 17.529x (D0076).

## Resume recipe

1. Confirm the branch tip and a clean worktree:

   ```sh
   git switch p1.5/v6-dirty-index-rebuild
   git status --short
   git log -3 --oneline    # expect a58e94628 at the tip
   ```

2. Recheck resources before any heavy lane. A failed `nvidia-smi` after a
   reboot may be real hardware absence — check `lspci -d 10de:` first; the
   RTX 4090 dropped off the PCI bus once on 2026-08-08 and required a
   power cycle.

   ```sh
   free -h
   nvidia-smi
   ```

3. Rebuild and re-run the fast suites before starting new work:

   ```sh
   cmake --build build/pdg-cuda-release -j2      # CUDA sources are -j1
   ctest --test-dir build/pdg-cuda-release -j1 -E corpus
   ./build/pdg-cuda-release/bin/pdg_placement_audit \
     --calibration test/data/pdg/placement-calibration-sm89.json | tail -1
   ```

4. Heavy GPU gates run one at a time with the hash-pinned fixture:

   ```sh
   PDG_LOCAL_LAS_FILE="$PWD/build/bench-data/download-3101-nosrs.las" \
   PDG_LOCAL_LAS_SHA256=2ea7a921a6f45e0058ee2e491136e80e6450d7920d080ce94f4924d0a1ecd8f9 \
   ctest --test-dir build/pdg-cuda-release \
     -R '^pdg_resident_placement_(v2|v3|v6)_gpu$' --output-on-failure
   ```

   The same binary registers `v7`, `v4`, `v5`, `v1`, and `direct`. They are
   serialized, have resource guards, and skip with code 77 when the
   required fixture or profile is unavailable. v2 has a 900 s timeout and
   v3/v6 3600 s (their host oracles run full KD3 kNN over the 21.9M-point
   fixture; the v3 oracle also builds upstream LOF's ~22 GB adjacency map).

5. Start the next catalog slice on a fresh branch from this tip, following
   the slice shape above. `build/diagnostics/profile-work/` holds the
   reusable calibration ladder scripts (`calibrate-*.sh`, `rescal-*.sh`)
   and the resident refit helper (`refit-resident-models.py`); copy and
   adapt rather than re-deriving the protocol.

## Known guardrails and traps

- A declared cardinality change must terminate the selected resident chain;
  the rewrite, runtime placement, and preflight enforce this from descriptor
  flags and payload shape, never stage names.
- One declared predicate per region; dropped rows publish nothing, and dead
  in-region writes are planner-released before the spill and never published.
- The keep mask is planner-owned lane storage: one byte per tile point in the
  lane allocation, the preflight probe, the spill boundary fact, and the
  placement peak. Keep the context and placement sides symmetric.
- Delegated neighborhood regions are whole-view: exactly one selected region,
  no cardinality change, and no context lanes. Every payload declares its own
  per-point scratch in the preflight switch (the eigen family shares
  `NeighborhoodExecutorScratchBytesPerPoint = 146`; LOF, nndistance,
  optimalneighborhood, and neighborclassifier compute theirs from the
  retained adjacency). A new stage must add its branch or the budget check
  is wrong.
- Model anchoring: one measured non-point-program model per region;
  point-program content joins at zero incremental cost; two measured models
  in one region are `mixed_calibration_models` by design.
- `ColumnPointTable::finalize()` rewrites dimension offsets into column
  ordinals; resident-selected executions must keep running against the
  dedicated row-backed `pdal::PointTable` behind the fail-closed
  offset-partition guard.
- `planner_resident_boundary_batch` still reports
  `selected_device_calibration_matches_executor=false`; that is honest, not
  a bug. Flip it only by measuring that executor's own calibration, never
  by synthesizing observation events (the shared-index executor earned its
  true in D0077 the measured way).
- Do not update placement coefficients or bless outputs without the evidence
  required by `spec.md` and `DECISIONS.md`.
- Keep the large fixture local and hash-registered. Never commit, upload,
  rename, or modify it.
- Preserve the one-heavy-lane rule, CUDA `-j1`, ordinary host `-j2` maximum,
  8-GiB `MemAvailable` floor, and bounded temporary outputs. Raw reports
  live under `build/benchmarks/`, never `/tmp` (D0065).
- No Vast.ai budget has been spent for this slice.
