# Separate comparative benchmark: pdg vs PDAL vs LAStools vs QGIS

A repository-run, author-produced timing harness written for the plain-language report
`docs/reports/pdg-independent-benchmark.pdf` (2026-08-18). It is deliberately
separate from the repository's own reference suite (`scripts/pdg/reference_suite.py`,
`BENCHMARKS.md`): its own task catalogue, its own staged inputs, and every
competitor run as an ordinary command-line program.

“Separate” means separate from the maintained reference benchmark suite. This
is not unrelated third-party validation. The historical result bundle covers
three machines, uses one capture per machine, and uses medians of 3, 2, and 1
timed repeats for the 1M/4M, 16M, and 47M sizes respectively.

| File | Purpose |
| --- | --- |
| `common.py` | paths, tool locations (`IB_*` environment overrides), source data |
| `prepare.py` | stages the inputs under `build/independent-bench/in` (LAZ/LAS, ground-classified copy, halves for merge, clip polygons) from the local corpus, or from `IB_SOURCE_DIR` on another machine |
| `tasks.py` | the 24 jobs and how each tool is asked to do them |
| `harness.py` | runs every (size, task, tool) job: warm-up + timed repeats, wall clock, peak RSS, output size/points/sha256; appends to `results/<label>.json` and resumes |
| `render.py` | pictures of inputs and outputs (matplotlib + pinned PDAL text writer + GDAL) |
| `charts.py` | the report's charts |
| `report.py` | HTML report + PDF (Chromium headless) |
| `remote_setup.sh` | after `bench/remote/vast_bootstrap.sh` on a rented box: LAStools wrappers, pdal_wrench build, AHN4 download, prepare + harness |

Tools: `pdg_gpu` (automatic), `pdg_cpu` (`CUDA_VISIBLE_DEVICES=""`), `pdg_gpu_all`
(`PDG_EXPERIMENTAL_CUDA_HYBRID=1`), `pdal_pinned` (the oracle build), `pdal_sys`
(`/usr/bin/pdal`), `lastools` (native Linux binaries, unlicensed), `wrench`
(`/usr/lib/qgis/pdal_wrench`), `qgis` (`qgis_process run pdal:*`).

Notes learned the hard way:

- keep the COPC input out of the input directory: QGIS silently substitutes
  `<name>.copc.laz` for `<name>.laz` when both exist;
- unlicensed LAStools tools may write deliberately distorted output and exit 1
  above 1.5–10 M points; current harness runs preserve that nonzero status and
  mark the artifact nonqualifying. Historical report rows that counted those
  timings remain exploratory and are labelled as such;
- LAStools' bundled `libproj` segfaults at exit on Ubuntu 24.04; put the system
  library directory first in `LD_LIBRARY_PATH` (see `remote_setup.sh`); the tools
  also need `bin/serf/geo/*.csv`;
- pdal_wrench `tile` needs `--input-file-list`, writes LAS, and `to_raster` averages;
- PDAL's COPC writer is not byte-repeatable in these historical runs. A checksum
  mismatch is not a conformance pass; use the explicitly versioned
  `scripts/pdg/copc_semantic.py` comparator as supplemental evidence and keep
  the default exact-byte contract authoritative.
