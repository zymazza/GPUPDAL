# Reports

`b0275-cross-machine-benchmark.html` is the self-contained B0275 report:
the fourteen reference workflows on the reference workstation (exact and
`--fast` contracts), the same workflows on a rented Threadripper PRO 3975WX +
RTX 4090 box (default, CUDA-hybrid, CUDA-hybrid + `--fast`), a 47.5M-point
AHN4 tile on both machines, LAStools timings of comparable jobs, rendered
outputs, the upstream-merge assessment of the CUDA kernels, and (section 7b,
B0276) a 47M / 95M / 190M-point scale ladder on a rented H200. Images are
embedded as data URIs; the same PNGs are kept in `b0275/`.

Regenerate:

```sh
python3 bench/report/build_report.py \
  --local-suite build/benchmarks/b0274-suite3 \
  --fast-local build/benchmarks/b0271-fast-suite2 \
  --box-results build/benchmarks/b0275-box/results \
  --box-machine build/benchmarks/b0275-box/results/machine.txt \
  --lastools-local build/benchmarks/b0275-lastools-local.json \
  --lastools-box build/benchmarks/b0275-box/results/lastools-box.json \
  --renders docs/reports/b0275 --box-renders build/benchmarks/b0275-box/results/renders \
  --ladder-results build/benchmarks/b0276-h200/results-ladder \
  --out docs/reports/b0275-cross-machine-benchmark.html
```

The remote side is `bench/remote/vast_bootstrap.sh`, `vast_run.sh`,
`vast_post.sh`, and `vast_ladder.sh` (the H200 ladder); the LAStools comparison is `bench/lastools/lastools_bench.py`;
renders come from `bench/report/render_outputs.py`. Every number in the report
is traceable to a JSON report under `build/benchmarks/` and to `BENCHMARKS.md`
B0268–B0276.

`b0277-local-calibration.html` is the B0277 report: the `pdg calibrate`
command (D0277), its tests, and what it did on two rented machines (an RTX
4090 with a non-reference driver on an EPYC 7742 host, and an RTX 5090 on an
EPYC 7B12 host): the fourteen reference workflows in default mode before and
after calibration and with the hybrid forced, the profiles written, a
real-data calibration for comparison, and the AHN4 47.5M-point tile in
default mode with the profile active. Regenerate:

```sh
python3 bench/report/build_calibration_report.py \
  --box "RTX 4090 (driver 580.95.05) + 2x EPYC 7742=build/benchmarks/b0277-4090/results-b0277" \
  --box "RTX 5090 (driver 580.159.03) + 2x EPYC 7B12=build/benchmarks/b0277-5090/results-b0277" \
  --tests-summary "..." \
  --out docs/reports/b0277-local-calibration.html
```

The remote protocol is `bench/remote/vast_calibrate.sh`, followed on the
boxes by `vast_calibrate_post.sh` (after the `--append` merge fix) and
`vast_calibrate_post2.sh` (after D0278).

`b0278-drop-in-defaults.html` is the B0278 report: the shipped GPU-class
profiles from the ten-GPU Vast sweep, the generic fallback, and default-mode
results with no calibration step on every swept GPU (D0279). Regenerate:

```sh
python3 bench/report/build_sweep_report.py --sweep build/benchmarks/b0278-sweep \
  --profiles data/placement-profiles --tests-summary "..." \
  --out docs/reports/b0278-drop-in-defaults.html
```

The remote protocol is `bench/remote/vast_sweep.sh` (calibrate on a box)
followed by `vast_sweep_proof.sh` (rebuild with the embedded profiles, no
local profile, default-mode suite); `bench/report/make_shipped_profile.py`
converts calibrate profiles into shipped ones and builds the generic profile;
`bench/report/sweep_tables.py` prints the BENCHMARKS.md tables.

`pdg-drop-in-report.html` is the plain-language release report for the
maintainer and an open-source audience: where the wins are, where they are
not, what to expect on a given machine (tier table, envelopes, switches), how
to test, and the release checklist. It now leads with B0286's exact frozen
1.7x artifact proof: fourteen 1M workflows on all ten named GPU/host machines
and the 47.5M r6 graph on exactly seven named machines, with three pairs per
cell. The B0278 3x tables remain explicitly historical. The evidence is
author-produced rather than third-party validation, and no 3DEP population
claim is made. The frozen no-rebuild protocol is
`bench/remote/freeze_release_artifact.sh` plus
`bench/remote/vast_release_reproof.sh`; retained publication artifacts are
under `b0286-frozen-reproof/`.

After D0280 (shipped margin 1.7x) the B0278 page is regenerated with
`--postscript "…"` stating the re-issue; its section 5/6 captions read the
margin from the profiles. `pdg-drop-in-report.html` was rewritten for the
1.7x rule (B0279).

`pdg-independent-benchmark.pdf` is retained at its historical filename for
stable links, but its title and description are now **separate comparative
benchmark**. Its bundle covers three machines and is separate only from the
maintained reference-suite implementation; it is not independent third-party
validation. Current harness runs preserve nonzero unlicensed-LAStools results,
hash every directory member, freeze staged-input provenance, and use at least
three measured repeats for large jobs. Historical one/two-repeat rows remain
labelled as such rather than being retroactively upgraded.

The PDF was regenerated on 2026-08-21 to use the GPUPDAL presentation label
and a zero-based linear y-axis for the aggregate 18-job chart. The retained
JSON inputs and benchmark values were not changed; their historical `pdg_*`
tool identifiers remain intact.
