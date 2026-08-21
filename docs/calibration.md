# Local placement calibration (`gpupal calibrate`)

Status: shipped in D0277/B0277. Governing text: `spec.md` D4 and
`DECISIONS.md` D0277; this page is operational.

## Why it exists

Automatic device selection in `gpupal pipeline` is a *performance* promise on
top of the exactness promise: a stage runs on the GPU only where a same-machine
measurement showed the complete process finishes sooner. The shipped build
carries exactly one such measurement — the embedded SM-89 profile
(`sm89-2026-08-17-r6-large-layouts`, RTX 4090 / driver 610.43.03 / CUDA 13.3,
evidence in `test/data/pdg/placement-calibration-sm89.json`). Every other
machine therefore fails closed to the host path, even when it has a capable
GPU (B0275: forcing the hybrid on a rented RTX 4090 with another driver took
r6 from 2.7x to 11.2x, but nothing selected it by default).

`gpupal calibrate` closes that gap without weakening the promise: it re-measures
the same calibration cases on the machine it runs on, with the same fitting
procedure that produced the embedded profile, and writes a profile keyed to
that exact machine. Nothing runs it implicitly; the runtime never times, fits,
or tunes anything.

## What it does

1. Identity. Reads the machine key: CUDA device 0 name, compute capability,
   NVIDIA kernel driver version, the CUDA toolkit the build was compiled with,
   the CPU model name, the logical CPU count, and the GPUPAL version.
2. Reference check. If the embedded profile applies to this machine it says
   so and exits (use `--force` to write a local profile anyway).
3. Fixtures. Writes a deterministic synthetic lidar-like cloud (LAS 1.4,
   record format 7, 36-byte records, scan-ordered strips over a rolling
   terrain with a vegetation fraction, `pdg::SyntheticCloudGenerator`) at each
   requested size, or takes head subsets of `--input FILE` via the sibling
   `pdal translate --readers.las.count`.
4. Planner-owned coefficients. Measures CUDA cold start (fresh process),
   pageable host↔device transfer rates, kernel-launch+synchronize latency, and
   the shared kNN index build cost per persistent byte. The LAS packing rate
   is inherited from the reference profile (no independent probe yet); the
   residual fit absorbs it for every measured shape and the profile records
   the provenance.
5. Cases. For each placement model in the embedded plan (the shared
   neighborhood family — `normal`, `covariancefeatures`,
   `normal-covariancefeatures-compose`, `eigenvalues`, `eigen-family-compose`,
   `lof`, `nndistance`, `estimaterank`, `optimalneighborhood`,
   `rank-optimal-compose`, `neighborclassifier`, `approximatecoplanar`,
   `radiusassign` — plus `fused-point-program` and `simple-ferry`) and each
   size, it runs alternating pairs: the host path (`pdal pipeline`, the fork's
   host build beside the engine) and the device path (`pdg-engine resident`
   under `PDG_CALIBRATION_FORCE_DEVICE_PLACEMENT=1`, which prices every stage
   as a device win so the planner's device executor runs before any local
   profile exists). Complete-process wall clock, medians, and the two outputs
   byte-compared (only the LAS header creation day/year is masked).
6. Fit. For each model, residual = (device − host) seconds − the planner's
   own device terms for that shape (`makePlacementCalibrationRequest` +
   `evaluatePlacement` with the local coefficients), fitted by the audit's
   ordinary least squares and split by sign into host/device fixed and
   per-point terms (`fitPlacementResidualModel`, shared with
   `pdg_placement_audit --suggest-models`). Envelope: minimum device points =
   the smallest measured size where the device won; maximum = the largest
   measured size (fail closed above it). A model whose device path never won,
   or whose outputs ever differed, is not written (host). `--append` reuses
   an existing profile's coefficients, keeps its other models, and refits an
   appended model over every size ever measured for it (the prior
   measurements travel in the profile's evidence), so envelopes only grow.
7. Profile. Writes `pdg-local-placement-profile-v1` JSON with the machine
   key, coefficients, stage models, a per-model summary, and the raw evidence
   (all timings, byte-exactness, probes, cgroup CPU quota if visible).

## Profile tiers (D0279): drop-in first, calibrate second

Since D0279 a machine does not have to be calibrated to use the GPU by
default. `placementCalibrationFor` consults, in order:

1. the **embedded reference profile** (the maintainer's RTX 4090, exact
   device/driver/toolkit key);
2. the calibration override (explicit `resident` command only);
3. a **local profile** written by `gpupal calibrate` on this exact machine;
4. a **shipped GPU-class profile** — measured by `gpupal calibrate` on a rented
   machine with the same GPU model, compute capability and compiled CUDA
   toolkit (`bench/remote/vast_sweep.sh`), converted by
   `bench/report/make_shipped_profile.py`, which keeps only the stage models
   whose device win cleared the shipping margin (1.7x, D0280; 3x when first
   shipped, D0279) at every measured size
   and bounds the envelope to those sizes; embedded from
   `data/placement-profiles/*.json` at build time; the driver and the host are
   recorded but not matched;
5. the **generic fallback** for any other CUDA device inside its `applies`
   bounds (compute capability and device memory of the smallest GPU that
   contributed): the intersection of the shipped profiles' models under the
   3x margin in every profile (its margin also covers an unknown, possibly
   slower GPU), with the slowest device and fastest host
   coefficients seen.

`gpupal calibrate --status` prints `active_profile_tier` (embedded / local /
shipped / generic / none) and, for a compiled shipped/generic profile, its
embedded source filename and SHA-256. Release re-proof compares that hash to
the closed profile payload. `gpupal doctor` shows the tier next to the profile
id. A local profile always takes precedence over the shipped and generic
tiers, so `gpupal calibrate` remains the way to tighten placement to a
particular CPU or dataset; it is no longer a prerequisite.

The promise of tiers 4 and 5 is weaker than tier 3's by construction and is
stated in the profile itself: "measured on this GPU class on an EPYC/Xeon-
class rented host, admitted only where the device won by the stated margin".

## Where the profile lives and how it is used

`$PDG_PROFILE_PATH`, else `${XDG_CONFIG_HOME:-~/.config}/gpupal/placement-profile.json`.
The previous file is kept as `.previous`.

A local profile applies if and only if its machine key equals the current
machine key in every field. A different GPU, driver, CUDA build, CPU, core
count, or GPUPAL version makes the file inert (`gpupal calibrate --status`
reports `machine-mismatch` and why) and the shipped/generic tiers take over.

Automatic admission keeps every structural gate it had: profiles supply
coefficients and envelopes; they do not widen shapes. One gate is now
measurable: `normal`+`covariancefeatures` with `extra_dims=all` (the r6
reference graph) is admitted through the separately calibrated
`normal-covariancefeatures-compose-extradims` model when a profile carries it
(D0279). B0285 refreshes the embedded reference profile with independent
same-machine, clean-upstream rows for the 35,976,465-point VEIL format-6/LAZ
graph and both publications on the 47,478,228-point format-8 AHN4 tile. Its
envelope stops at that observed local ceiling rather than inheriting the
shipped profile's rounded 48M class bound. The extra-dimensions route is also
bounded to explicit physical tuples rather than a width range: B0223/B0224's
exactly-1M format-7/36 compressed input may write format-7/100 LAS or LAZ;
B0285's format-6/30 compressed VEIL tuple may write format-7/100 LAZ; and its
format-8/44 compressed AHN4 tuple may write format-7/120 LAZ. Every other
format, stride, compression, or output-width cross-product fails closed to the
host until it has retained evidence.
The enlarged plain-publication model is narrower still: only the original
uncompressed format-7/36 input and the independently measured compressed
format-8/44 AHN4 input may write its uncompressed format-7/36 output.

Layout admission and cardinality estimation are deliberately separate. Within
one admitted physical tuple, the profile's measured point-count curve may
select device at interior counts from 250,000 through the exact 47,478,228
ceiling; it does not require a fixture-identity or endpoint-count match. The
physical matrix exercises the format-6/30 tuple at 274,625 points and placement
units exercise interior counts. This is a placement prediction, not a claim
that every interior tile achieves B0285's measured speedup. A public speed
claim still requires its own same-machine oracle pair and retained report.

Default-mode outputs remain byte-identical to the pinned oracle whatever the
profile says: the profile only chooses between two exact executors.

## Commands and environment

```
gpupal calibrate --status              # machine key, embedded/local profile status
gpupal calibrate                       # 250K/1M/4M points, 3 pairs each, 15 models
gpupal calibrate --quick               # 250K/1M, one pair each
gpupal calibrate --input tile.laz --points 250000,1000000,4000000
gpupal calibrate --models lof,normal-covariancefeatures-compose --repeats 5
gpupal calibrate --dry-run --force     # print the plan
gpupal doctor                          # shows placement_profile / local_profile
```

| Variable | Effect |
| --- | --- |
| `PDG_PROFILE_PATH` | profile file to read and write (also forces the launcher onto the engine route, as every `PDG_*` name does; the default file location needs no variable) |
| `PDG_DISABLE_LOCAL_PROFILE` | never load a local profile |
| `PDG_CALIBRATION_FORCE_DEVICE_PLACEMENT` | calibration override for explicit `resident` runs; the automatic `pipeline` route declines under it |
| `PDG_TEST_IGNORE_BUILTIN_PLACEMENT_PROFILE` | test hook: hide the embedded profile so the local path can be exercised on the reference machine |
| `PDG_DISABLE_SHIPPED_PROFILES` | never consult the shipped GPU-class or generic tiers ("measured on this machine only") |
| `PDG_TEST_SHIPPED_PROFILE_DIR` (+ `_ONLY`) | test hooks: load extra shipped/generic profiles from a directory in addition to (or instead of) the embedded table |

Benchmark reports from `scripts/pdg/benchmark_reference.py` record the
candidate's `calibrate --status` under `environment.candidate_placement_profile`.
For a placement qualification, pass `--qualification-model MODEL`; the harness
derives the physical LAS/LAZ input and output facts from the measured files and
requires `PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT=1`. It also scrubs
ambient `LD_*` and fails closed unless it can hash the adjacent ELF
`pdg-engine` and the `libpdalcpp` actually resolved for candidate and oracle
under their measured environments. The registered calibration-provenance test
requires those fields rather than accepting a report with omitted components.

Automatic resident execution validates a published LAS/LAZ before it returns
success. The public header and VLR/EVLR extents must parse; uncompressed point
records must fit the advertised count/stride, and the header count must equal
the resident executor's observed output count. For LAZ, the LASzip chunk table
must parse, cover the declared point count, and describe exactly the compressed
point-payload byte extent. The LAZ check is O(chunks), not a second O(points)
decode. This is a post-side-effect failure boundary: an invalid publication
returns nonzero and is never retried through the host pipeline.

## Tests

- `tests/unit/local_profile_test.cpp`: schema round trip and rejection,
  lookup statuses, embedded-key precedence, override scoping, residual fit,
  synthetic cloud determinism and LAS layout.
- `tests/differential/calibrate_test.py`
  (`pdg_calibrate_local_profile_cuda_exact`): status/dry-run write nothing;
  the reference machine declines without `--force`; a quick calibration
  admits byte-exact device wins; the profile enables the automatic
  normal/covariance resident route (proof gate), a missing or mismatched
  profile and the calibration override all fail closed (124); default output
  with the profile equals the host output and the pinned oracle.
- `tests/differential/resident_selection_matrix.py`: physical layout positives
  and explicit unmeasured-tuple refusals, required automatic-route proof,
  preflight refusal, a deterministic compressed-payload truncation, and a
  partial plain LAS with a stale zero count; both publications must return
  nonzero.

## Limits (v1)

- Calibrated models: the shared-neighborhood family and the two point-program
  models above. Direct whole-view executors (sort/skewness/HAG/radius/outlier
  direct compositions) stay uncalibrated on non-reference machines.
- The synthetic cloud is a stand-in for real data; `--input` with the user's
  own tiles is the recommended way to calibrate for a workload.
- Single device (CUDA ordinal 0); a machine profile is one GPU + one host.
- The resident executor's own preflight still applies after placement. Since
  D0278 it follows the file for extra-bytes dimensions whose declared type
  differs from PDAL's standard dimension of the same name (AHN4 tiles carry
  `Deviation` as uint16 and `Amplitude`/`Reflectance` as double where the
  standard table says otherwise), exactly as stock PDAL does, provided no
  planned device stage reads or writes that dimension; a dimension a device
  stage computes on keeps the refusal ("resident PointView dimension type
  differs from PDG for a device-computed dimension: NAME") and the graph runs
  on the host, byte-identical as always.
