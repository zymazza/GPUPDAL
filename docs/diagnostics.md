# Diagnostic instruments

Every entry here is **opt-in and changes no output**. They exist because of
D0215: a gate that refuses, or a path that is slow, must be able to say why.
Several findings in `BENCHMARKS.md` were only possible because one of these
existed, and several wrong conclusions survived for a slice or more because one
did not.

The ordering below is roughly "cheapest first". Prefer an instrument over
reading source: B0195, B0203, B0207 and B0214 each recorded a confident
mechanism derived from source that a later measurement overturned.

## Zero-code techniques

These need no build and are often decisive.

| Technique | Answers |
| --- | --- |
| `PDG_ORACLE_PDAL=/bin/true` | How much work does the engine do **before** delegating to the oracle? The measured wall time is the engine's own overhead, because the handoff becomes free. B0196 used this to show a translate's whole deficit was two dynamic-link cycles; B0205 used it to find 236 ms of discarded work on an SMRF pipeline. |
| `PDG_ORACLE_PDAL=<path>` | Which `pdal` does the engine hand off to? Pointing it at the pinned oracle isolates "the engine chose badly" from "the engine's own PDAL build is slower". |
| upstream `filters.smrf` `dir` option | Dumps `zimin`, `zipro`, `gsurfs` and their filled variants as GeoTIFF. **The band is Float32**, so at typical elevations one ulp is ~3e-5; differences below that mean agreement, not a small disagreement (B0213). |
| forcing routes with `PDG_EXPERIMENTAL_*` / `PDG_REQUIRE_*` | Whether a route is *reachable* and whether it is *exact*. The `REQUIRE` variants fail closed with a named error; the `EXPERIMENTAL` variants do not, which is how B0207 found a byte-inexact path exiting 0. |

## Built-in instruments

| Variable | Emits | Answers |
| --- | --- | --- |
| `PDG_DEBUG_ADMISSION_PHASES` | `clog`, one line per checkpoint | Where an automatic admission attempt gave up, and how long each phase took. Declines return before stats are written, so this is the only view of them. B0196 used it to refute a 100x-wrong estimate of the admission cost; B0205 used it to find ~172 ms of CUDA discovery spent on a plan-decidable refusal. |
| `PDG_DEBUG_SMRF_PHASES` | `stderr`, one line per phase | Wall time of each phase of the exact host `filters.smrf` (segmentation, return scans, minimum-Z raster + fill, low/net/object masks, provisional DEM + fill, classify ground). B0264 used it to find that the progressive filter's 189 diamond passes and the two void fills dominated, not the point passes. |
| `PDG_DEBUG_SMRF_FILL` | `clog`, one line per fill site | How many void cells each of the three `fillRaster` calls actually fills. Note there are **three** call sites (minimum, provisional, gradient) — reading only the last is how B0212 nearly recorded a false refutation. |
| `PDG_DEBUG_SMRF_DUMP=<prefix>` | `<prefix><stage>.f64` | Each SMRF intermediate as raw little-endian doubles in the port's cell order (`cell = column * rows + row`). Pairs with upstream's `dir` output to localise a divergence to a stage. B0213 used it to pin pdg's SMRF disagreement to the provisional surface's void fill. |
| `PDG_DEBUG_HYBRID` | `stderr` | Why the hybrid rewrite declined or which probe failed. |
| `PDG_DEBUG_HYBRID_OUTLIER_PHASES` | `stderr` | Phase timings inside the hybrid outlier path. |
| `PDG_DEBUG_HAG_NN_PHASES` | `stderr` | HAG-NN coordinate/ground scans, resident-index setup, query/wait/status timings, and exact/tie/incomplete row counts. |
| `PDG_REQUIRE_HAG_NN_SELECTIVE_REPAIR` | proof gate | Require HAG-NN to retain the CUDA result and rebuild only tie/incomplete rows with the pinned ordered host calculation. |
| `PDG_DEBUG_FUSED_JIT` | `stderr` | Fused point-program JIT specialisation decisions. |
| `pdg-engine resident PIPELINE --stats FILE` → `execution.rewritten_manager_execution_breakdown_seconds`, `resident_work_breakdown_seconds`, `eigen_family_breakdown_seconds` | JSON | Since B0260 the manager breakdown is collected for ordinary upstream-writer single-region neighborhood routes too, and the eigen family (normal/eigenvalue/covariance/coplanar) reports submission, ambiguous repair (status wait, system download, **index build**, rows, upload, row counts), output preparation, projection/copy, status scan, transcendental features, column publication, and host upload. B0260 used it to find that r6 spent 0.47 s building a full nanoflann tree to repair 2,343 tie rows. |
| `PDG_LAZ_COMPRESSION_THREADS=N` (engine/sibling only; the public launcher removes external values) | control | LAZ chunk-compression worker count for `writers.las`; the default is `min(4, hardware threads)` (B0259). Setting 1 through `pdg-engine` or the sibling `pdal` gives the same-binary serial control; test builds honor `PDAL_TEST_REQUIRE_LAZ_COMPRESSION_THREADS=N`. |
| `PDG_DISABLE_KD3_COORDINATE_CACHE` | control | Restores the uncached nanoflann adapter for published `build3dIndex()` products (B0263 default is the exact cached-coordinate backing with a mutation epoch that refreshes reused snapshots). Same-final-binary control: r11 falls from 7.37x to 5.32x with it set. Test builds honor `PDAL_TEST_VERIFY_KD3_SNAPSHOT=1`, which fails closed at any reuse whose snapshot silently diverged from the live view. |
| `PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS` / `PDG_NATIVE_WORKERS=N` | control | Force the exact serial statistical-outlier / neighbor-classifier kNN passes, SMRF grid passes, reprojection batches, and `writers.las` record packing (B0268; test builds also honor `PDAL_TEST_TRACE_LAS_PACK=1`, one stderr line per parallel pack run with its outcome), `readers.las` record unpacking (B0269; `PDAL_TEST_TRACE_LAS_UNPACK=1` prints rows/segments/slots per worker run), and `readers.copc`'s ordered decode pool under `requests=1` (B0270; `PDAL_TEST_TRACE_COPC_DECODE=1` prints tiles/ordered/workers), `writers.gdal`'s banded raster accumulation (B0273; `PDAL_TEST_TRACE_GDAL_BANDS=1` prints rows/bands/listed/height per banded run), and every KD2/KD3 nanoflann build (B0274, concurrent subtree construction), or cap their worker count (B0258). |
| `PDG_EXPERIMENTAL_AUTOMATIC_R4_OUTLIER_CUDA=1` | control | Re-enables the retired B0227 literal 1M r4 CUDA outlier selector (D0272): the exact host path (worker kNN outlier + hashed sample) is faster at 1M and 4M, so the default r4 route is host; the opt-in keeps the route's differential lane and the `PDG_REQUIRE_AUTOMATIC_R4_OUTLIER_CUDA` proof gate exercisable. |
| `gpupal --fast` (internal `PDG_INTERNAL_FAST_MODE=1`, launcher-armed only) | contract | D0271: the spatial index publishes no `KnnDistanceTie`, so `pdg-engine resident --stats` reports zero ambiguous/repaired rows for the eigen family, HAG-NN, LOF, and classifier under `--fast`; compare with the exact run's tie-row counts to bound the differing records the fast comparator reports. The serial control is the same-final-binary attribution: r11 returns to about 1.035x when set. Test-only builds also honor `PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=N` and `PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS=N`. |
| `pdg_r11_neighborhood_attribution --input FILE [--output REPORT.json]` (bench target, needs the local 1M reference LAZ) | JSON report | In-process wall and allocation attribution of the r11 prefix, fresh KD3 build, statistical-outlier and neighbor-classifier kNN passes, the isolated vote tally, and fixed-chunk multi-worker exact executions with bit-identity checks. B0257 used it to reject the vote-tally hypothesis (about 0.1% of wall) and to attribute about 5.55 s of the 6.8-second process to the two serial kNN passes. |

## Reported state

`gpupal resident --stats <file>` carries two sections added for the same reason:

- `plan` — the planner-derived attributes the compose matchers test (`native`,
  `preferred_residency`, `resident_region`, `device_knn_gather_neighbors`,
  `device_query_bytes_per_point`, `device_index_build_bytes_per_point`,
  `device_to_host_bytes_per_input_point`, plus `resident_regions`,
  `index_builds`, `grid_builds` and the residency boundaries). Emitted **even
  when placement is unavailable**, because that is the case it explains.
- `placement.input_record_bytes` / `placement.output_record_bytes` — the
  derived record sizes, so a value that used to be a hardcoded constant is
  readable rather than inferred (D0218).

## Three standing warnings

1. **Never read `--stats` from `gpupal resident` to explain a `gpupal pipeline`
   timing.** They are the same function with `automaticAdmission` flipped, so
   the numbers look comparable and are not. `gpupal resident` reported 146 ms on a
   translate whose public-path admission costs 0.17 ms (B0185, B0196).
2. **Use paired alternating runs, not medians of three.** Reference-pipeline
   single-run spreads reach 463 ms; a median-of-3 sweep produced a clean-looking
   result that was entirely noise. `scripts/pdg/benchmark_reference.py` reports
   a paired standard error and a `resolvable` flag — read the flag before
   quoting a speedup (B0199, B0200, B0201).
3. **`gprofng` clock profiles on this workstation undersample CPU by roughly
   7--10x** (binutils 2.46.1, kernel 7.1.3): 0.290 s recorded for a 3.0-second
   single-threaded spin loop and 0.499 s for a 6.9-second `pdal pipeline`
   process (B0257). The B0255/B0256 profiles were affected. Use them only to
   order large costs; rank small host costs against large ones with in-process
   timers or the attribution harness above.
