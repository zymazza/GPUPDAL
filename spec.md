# GPUPAL — Build Specification for the Coding Agent (v0.2)

> Product and repository: GPUPAL (GPU Pointcloud Abstraction Library). Public
> CLI binary: `gpupal`. The internal C++ namespace remains `pdg`; the Python
> package name is still provisional. See D0288 and §15.

---

## 0. How to read this spec

- **§12 is your work queue.** Execute phases in order; reorder freely *within* a phase.
- **§7 is the initial acceleration queue.** D0019 expands the final product
  boundary to every stage and configured plugin in an equivalent upstream PDAL
  build; the catalog controls delivery order, not permission to omit the long
  tail.
- **§14 gates override enthusiasm.** Risky bets (GPU LAZ, JIT, GPU Delaunay) live behind measurable go/no-go criteria.
- **§13 is how you work.** Definition of done, profiling discipline, and what to do when the spec is ambiguous.
- Prime directive: **maximum throughput on consumer NVIDIA hardware subject to
  exact default compatibility**, with RTX 4090 as the performance reference
  only, never the compatibility floor. When two exact designs conflict,
  benchmark both and choose the faster one on the reference device, recording
  the decision in `DECISIONS.md`. Semantically relaxed modes must be explicit
  and never alter default output.

## 1. Mission

Make the forked PDAL point-cloud processing library as fast as possible on the
pipelines people actually run, while preserving its applications, readers,
filters, writers, plugins, and pipeline model exactly. CUDA is the primary
tool where it wins — moving data-parallel regions to the device, retaining
device-resident columnar data across stages, and sharing one spatial index —
but it is a means, not the goal. Optimized host execution, parallel and
streaming I/O, and removed stage boundaries count equally. Success requires
complete exact functional parity and measured end-to-end speedups over the
pinned PDAL oracle on identical hardware for the reference pipelines in
`bench/pipelines/reference/`. Catalog-wide *CUDA* coverage is explicitly not a
release criterion; see D0208, which supersedes the corresponding parts of
D0019 and the catalog-wide language below.

Principles, in priority order:

1. **Exact by default.** Output bytes, ordering, metadata, diagnostics, and
   failure behavior match the pinned oracle; speed is optimized inside that
   contract.
2. **Measure everything.** No performance claim without a benchmark number committed to `BENCHMARKS.md`. No optimization without a profile.
3. **Bandwidth is the budget.** Most point-cloud stages are memory-bound. The reference device moves ~1 TB/s; per-point stages are designed and judged against that roofline, not FLOPs.
4. **One index, many consumers.** Roughly a third of the catalog is "spatial query + small per-point math." Build the neighborhood engine once, superbly, and make every stage a client of it. Never let a stage build a private index.
5. **Never block the GPU.** I/O, host work, and device compute overlap via streams and tiling. The GPU idling is a bug.
6. **A CPU fallback beats a missing feature.** Stages not yet ported run on host threads with automatic transfers at the boundary — but every fallback is logged, and the long-term direction is always device-side.
7. **Open-source hygiene from day one.** Tests, docs, licenses, reproducible builds. This will be public.

## 2. Non-goals for the first release

- Multi-node distribution. Multi-GPU (single-GPU only, but do not bake single-GPU assumptions into the tile scheduler's interfaces).
- Native Windows or macOS. Linux x86_64 only; WSL2 counts as Linux.
- HIP/SYCL/OpenCL portability layers. CUDA-native for maximum speed; portability is a post-1.0 conversation.
- Approximate arithmetic, reordered points, nondeterministic metadata, or any
  other relaxed behavior in the default compatibility mode. Such behavior is
  permitted only behind an explicit `--fast` request with separately labeled
  artifacts and benchmarks.
- Reimplementing every third-party or proprietary long-tail codec merely to
  execute codec work on CUDA. Formats such as e57, rdb, rxp, i3s, nitf, mbio,
  and slpk remain fully and exactly supported through an optimized bridge when
  the external codec is evidence-backed as GPU-inapplicable; adjacent native
  regions must retain device residency wherever possible.
- Any GUI or viewer.

## 3. Target platform & stack

**Reference device:** NVIDIA RTX 4090 — Ada, SM 8.9, 24 GB GDDR6X, ~1008 GB/s DRAM, 82.6 TFLOPS fp32, fp64 at 1/64 rate (~1.3 TFLOPS), 72 MB L2, PCIe 4.0 x16 (~25 GB/s effective H2D with pinned memory). It is the performance reference, not the compatibility floor. Release builds target every real GPU architecture supported by their selected compiler and carry PTX for the compiler's newest target. The CUDA-12.x legacy-compatible artifact presently covers SM 50/52/53, 60/61/62, 70/72, 75, 80/86/87, 89, and 90; CUDA 13+ covers SM 75 and newer, including post-Hopper targets reported by that compiler. Because CUDA 13 removed pre-SM75 code generation, both artifacts are required for the maintained compatibility range. Architecture generation alone is not an exactness claim: physical per-SM bit differentials gate advertised runtime support, and the current physical qualification is SM 89 only. The fp64 ratio is a first-class design constraint — see §5.1.

**Toolchain & dependencies:**

| Component | Choice | Notes |
|---|---|---|
| Language | C++20 host; device code within nvcc's supported subset | |
| CUDA | ≥ 12.4, CCCL (CUB / Thrust / libcu++) | Prefer CCCL primitives over hand-rolled kernels |
| Build | CMake ≥ 3.28 + presets, Ninja | Single top-level target set; `gpupal`, `libpdg`, tests, bench |
| JSON | simdjson (parse), nlohmann (emit) | Pipeline files are small; simdjson is for point-adjacent metadata too |
| CRS | PROJ ≥ 9 (host) | Plus native fp64 device kernels for common projections, §7 transforms |
| Raster/vector I/O | GDAL ≥ 3.8 (host) | Colorization, DEM inputs, GeoTIFF out, OGR polygons |
| LAZ | lazperf (reference impl) | License check before vendoring; never use LASzip (LGPL) — see §15 |
| Tests | GoogleTest; libFuzzer for readers | |
| Bench | nvbench (kernels), google/benchmark (host paths) | |
| Python | pybind11 | Zero-copy device export, §9 |
| Optional | cuFile/GDS (gate G5), NVRTC (gate G3), libpdal (bridge §6.5) | |
| Logging | spdlog + NVTX ranges on every stage | NVTX is mandatory, not optional |
| Sanitizers | compute-sanitizer (memcheck, racecheck), ASan/UBSan host | CI-enforced |

## 4. Repository layout & conventions

```
pdal-gpu/
  CMakeLists.txt  cmake/  .clang-format  .github/workflows/
  include/pdg/            # public API only — keep minimal, ABI unstable pre-1.0
  src/core/               # PointBatch, dimension registry, allocator, plan, scheduler
  src/index/              # morton, hash grid, LBVH, query device API
  src/math/               # eigen3x3, svd3, welford, reductions, philox rng
  src/expr/               # expression parser → bytecode VM (NVRTC JIT behind gate G3)
  src/stages/             # one directory per family: ground/ features/ cull/ mesh/ ...
  src/io/                 # las, laz, copc, ept, text, ply, pcd, gdalraster, pdalbridge
  src/cli/                # pdg binary
  python/                 # pybind11 package
  tests/unit  tests/golden  tests/fuzz  tests/data
  bench/                  # nvbench targets, reference pipelines, BENCHMARKS.md
  docs/                   # mkdocs; one page per stage with PDAL-parity notes
  third_party/            # vendored deps, each with LICENSE preserved
  AGENTS.md               # points here; quickstart for agent sessions
  DECISIONS.md            # append-only log of spec deviations & gate outcomes
  BENCHMARKS.md           # append-only log of measured numbers per commit
```

Conventions: RAII wrappers for all device resources; no raw `cudaMalloc` outside the allocator; `PDG_CUDA_CHECK` on every runtime call; exceptions at the public API boundary only; grid-stride loops as the default kernel shape; `__launch_bounds__` documented wherever register pressure forced a choice; no dynamic parallelism; no managed/UVM memory in hot paths.

## 5. Core architecture

### 5.1 Data model & precision policy (load-bearing — read twice)

`PointBatch` is a structure-of-arrays: one device array per dimension, materialized lazily — the plan (§5.3) computes which dimensions the pipeline touches and only those are decoded/uploaded. Default batch capacity 2^24 points; actual tile sizing per §5.3.

**Coordinate representation:**

- **Canonical storage:** XYZ as `int32` triplets + dataset-level fp64 `scale`/`offset` — LAS semantics, exact, 12 B/pt. This is what lives in VRAM and what readers decode to.
- **Working frame:** neighborhood, feature, grid, and geometry math runs in **fp32 local coordinates**: `(int32 * scale) − tile_origin`, with `tile_origin` snapped per tile. Error analysis: a 2 km tile at mm resolution needs 21 bits — comfortably inside fp32's 24-bit mantissa. Enforced by gate G6 (fp32-local error audit).
- **fp64 is exceptional**, used only in: CRS transforms, GpsTime, tile-origin bookkeeping, and final global accumulators (or Kahan-compensated fp32). Never in `[knn]` inner loops — the reference device runs fp64 at 1/64 rate and it will show up instantly in nsight.
- **Determinism policy:** compatibility mode is the default and must match the
  pinned PDAL oracle byte-for-byte, including point order and metadata, while
  remaining bitwise-stable across repeated runs. Use fixed-order reductions,
  integer-domain accumulation, deterministic tie-breaking, or the fastest
  exact host implementation as necessary. An explicit `--fast` mode may opt
  into reordered or nondeterministic algorithms, but it is never selected
  implicitly and never satisfies exact product coverage. The `--fast` contract
  actually defined and shipped so far is narrower (D0261, widened once by
  D0271): point records stay in PDAL's order with identical count and
  coordinates; kNN distance ties may be resolved in device order (equally
  valid neighbor sets or accumulation orders at identical distances, so
  attributes on those rows may differ from stock, bounded and reported by the
  fast comparator); and only diagnostics, error text/status, LAS header/VLR
  metadata, and metadata JSON may otherwise differ; any wider relaxation
  needs its own decision entry.

**Dimension registry:** mirrors PDAL's dimension names and types verbatim (`X,Y,Z`, `Intensity` u16, `ReturnNumber` u8, `NumberOfReturns` u8, `Classification` u8, `GpsTime` f64, `Red/Green/Blue` u16, `PointSourceId` u16, `UserData` u8, `ScanAngleRank`, …) plus typed custom dimensions. String→column lookup happens at plan time only; kernels see raw typed pointers.

### 5.2 Memory management

- One stream-ordered pool allocator (CUB `CachingDeviceAllocator` or an RMM-style pool — benchmark once, pick, record). All allocation goes through it.
- Pinned host ring buffers for staging; every H2D/D2H is async on a stream.
- Three-stream execution: upload(i+1) ∥ compute(i) ∥ download/write(i−1).
- Plan-time VRAM estimator: Σ(bytes/pt × touched dims) + index (~16 B/pt) + stage scratch peaks. Working-set ceiling 20 GB on the 24 GB reference device; exceeding it switches the plan to tiled execution automatically.

### 5.3 Execution model

- Pipeline JSON → DAG → **plan**. Every stage ships a descriptor declaring: `reads` (dims), `writes` (dims), `kind ∈ {pp, knn, grid, global, split, cpu}`, `needs_index(radius r | knn k)`, `max_radius`, and whether it mutates XYZ.
- **Fusion:** adjacent `pp` stages fuse into one kernel pass. Built-ins fuse via templates; `filters.expression`/`assign` chains compile to a compact stack-machine bytecode executed inside the fused kernel (single memory pass — these stages are bandwidth-bound, so the VM costs ~nothing). NVRTC JIT-to-PTX replaces the VM only if gate G3 passes.
- **Tiling / out-of-core:** datasets exceeding the VRAM budget are processed as spatial tiles — taken from the COPC/EPT hierarchy when present, otherwise built on the fly by Morton binning during read. Each tile loads with a **ghost halo whose width = max over the pipeline of stage `max_radius`** (computed at plan time). Halo points participate as neighbors but are never emitted — a per-point `ghost` flag rides with the batch. Grid stages that want a global canvas (SMRF surface, DTM raster) either allocate the global raster directly when it fits (0.5 m over 100 km² = 4×10^8 cells × 4 B ≈ 1.6 GB — it usually fits) or run per-tile with a seam-resolution pass; each grid stage documents which.
- **Global stages** (dataset-wide sort, fps, registration, chipper) run in gathered mode; when the dataset exceeds VRAM they spill through host-staged merge passes. Each global stage documents its spill strategy.
- CPU stages run on a host thread pool overlapped with GPU tiles; transfers at the boundary are automatic, counted, and reported in `--stats`.

### 5.4 Spatial index engine (the centerpiece)

PDAL's own docs track which filters "invalidate an existing KD-tree" because its per-view KD3Index rebuilds are a dominant cost. Here the index is a first-class, persistent, shared object:

- **Build:** rebase coordinates → 63-bit Morton codes (21 bits/axis over scaled ints) → CUB radix sort → cell tables. Whether to materialize the permutation (physically reorder columns) is decided at plan time: reorder if ≥ 2 neighborhood stages consume the index, else gather through the permutation. Benchmark this heuristic in P2 and adjust.
- **Primary structure — uniform hash grid:** cell edge tied to the query radius (or a k-derived heuristic); compacted cell-start/count tables from the sorted codes; radius queries scan ≤ 27 cells. Optimal for airborne-lidar-like density.
- **Secondary structure — LBVH** (Karras 2012 build over the same Morton codes) for adaptive-density kNN (TLS scans, clustered data); warp-cooperative traversal.
- **Gate G1** (end of P2) keeps both or kills one, decided by benchmark on three density profiles: uniform ALS, high-density TLS clusters, mixed.
- **Device API** (all `[knn]` stages consume exactly this — no stage-private traversal code):

```cpp
template <class F>
__device__ void for_each_in_radius(uint32_t i, float r, F&& f);      // f(j, dist2)
__device__ int  knn_gather(uint32_t i, int k, uint32_t* nbr,
                           float* dist2);                             // k ≤ 64, register/shared heap
```

- **Lifecycle:** the index is owned by the tile; any stage that writes XYZ marks it dirty; the planner inserts rebuilds and reports the count in `--stats`. Design target: ≤ 1 index build per tile for the standard DTM pipeline (§10).

### 5.5 Device math library

Shared, tested, benchmarked once — used everywhere:

- Analytic symmetric 3×3 eigensolver (hybrid analytic + QL fallback, Kopp 2008). This single routine backs ~10 stages.
- 3×3 SVD (McAdams et al.) for `estimaterank` and registration.
- Numerically stable covariance accumulation at fp32 (two-pass or Welford) — naïve Σx² at fp32 on UTM-scale numbers is exactly the bug the local frame exists to prevent; assert coordinates are local-frame in debug builds.
- The exact default random stages reproduce the pinned PDAL RNG, seed, and
  output ordering. Philox is available only to an explicitly requested
  `--fast` algorithm with its own reproducibility contract.
- Segmented reductions / histograms via CUB; prefix-covariance trick for `optimalneighborhood` (sort neighbors by distance once, sweep k with incremental covariance updates instead of recomputing per k).

## 6. I/O subsystem

I/O is where "as fast as possible" is usually lost. Treat it as a peer of the compute engine.

### 6.1 LAS
Point records are fixed-stride (formats 0–10): read raw bytes → pinned staging → device → one decode kernel scatters to SoA (and the inverse for writes). Target ≥ 5 GB/s parse throughput — this path should be NVMe-bound, not CPU-bound.

### 6.2 LAZ
The chunk table makes chunks independently decodable (default 50k points; COPC uses variable chunks). v1 ships **chunk-parallel CPU decode**: one chunk per host thread via lazperf across all cores, into pinned staging, overlapped with upload — this alone is a large win over PDAL's serial decode. A **GPU decoder is an experiment behind gate G2**: arithmetic decoding is inherently serial per chunk, so the parallel grain is chunks (thread-per-chunk, thousands in flight); go only if it beats 32-thread CPU decode by ≥ 2×. Encode: chunk-parallel CPU in v1. Document a recommendation for users to write smaller chunks (or COPC) for maximum decode parallelism.

### 6.3 COPC
Read: octree pushdown for spatial/resolution queries; nodes are natural tiles and their siblings supply ghost halos; range reads via cuFile/GDS behind gate G5 (keep if ≥ 1.5× vs `pread` + pinned staging on cold cache). Write: Morton codes and octree assignment are already on-device — reuse them; LAZ chunk encode on CPU threads.

### 6.4 Other native formats
text/CSV, PLY, PCD, EPT (read). GDAL handles raster in/out (colorization sources, DEM inputs, GeoTIFF products); OGR handles polygon inputs (`crop`, `overlay`) and vector outputs (`hexbin` boundaries).

### 6.5 The libpdal bridge (transitional coverage)
`readers.pdal` / `writers.pdal` wrap a stage of an installed PDAL as a CPU
stage, streaming PointViews into pinned buffers. This keeps the long tail of
exotic formats usable while their native paths are implemented. Bridge usage
is logged and never counted as completed GPU-native coverage; D0019 governs the
full-product exit.

## 7. Initial acceleration catalog (delivery priority)

PDAL stage names are retained **verbatim** for pipeline compatibility (§8).
Tags: `pp` per-point map · `knn` neighborhood query · `grid` raster/voxel op ·
`sort` sort/partition · `red` reduction · `iter` iterative/solver · `geom`
vector-geometry · `cpu` host-side work. **Ph** = delivery phase (§12). **Val**
names a supplemental algorithmic validation tier (§11): E exact · N numeric ·
S statistical · Q qualitative/invariants. Every default stage path must also
pass the byte-exact compatibility tier; N/S/Q never replace it.

### 7.1 Ground / terrain classification

| Stage | GPU strategy | Ph | Val |
|---|---|---|---|
| `filters.smrf` | min-Z grid → progressive morphological opening (increasing windows, slope threshold) → iterative inpaint → classify by height delta. Pure image ops. | 3 | S |
| `filters.pmf` | same family: grid morphology with growing windows | 3 | S |
| `filters.csf` | cloth = particle grid; position-based dynamics, Jacobi iterations, collision vs inverted cloud. Physics sim — near-perfect GPU fit. | 3 | S |
| `filters.skewnessbalancing` | global sort by Z + iterative skewness sweep (reductions) | 3 | S |
| `filters.sparsesurface` | radius-based ground sparsification via shared index | 5 | S |
| `filters.trajectory` | batch residual/Jacobian evaluation on CUDA; choose the measured fastest exact small-system solve and retain an optimized host bridge only with a GPU-inapplicability record | 5 | N |

### 7.2 Noise & outliers

| Stage | GPU strategy | Ph | Val |
|---|---|---|---|
| `filters.outlier` | SOR: kNN mean-distance + global μ/σ + threshold kernel. ROR: radius count. | 2 | S |
| `filters.elm` | grid min-scan for isolated low points | 3 | S |
| `filters.lof` | two-pass kNN (k-distance, reachability), classic Breunig LOF | 2 | N |
| `filters.iqr` / `filters.mad` | device median/quartiles via sort or histogram → cull kernel | 1 | N |

### 7.3 Reclassification & assignment

| Stage | GPU strategy | Ph | Val |
|---|---|---|---|
| `filters.neighborclassifier` | kNN majority vote (u8 ballot in shared mem) | 2 | S |
| `filters.assign` / `filters.ferry` | fused `pp`; expression bytecode VM | 1 | E |
| `filters.radiusassign` | radius query + conditional assign | 2 | E |
| `filters.overlay` | OGR polygons → device edge grid/BVH → point-in-polygon stamp | 3 | E |
| `filters.colorinterp` | ramp LUT in constant/L2, `pp` | 1 | E |
| `filters.colorization` | GDAL reads raster → device texture → sample per point | 3 | N |
| `filters.gpstimeconvert` | `pp` with scan-carried week-rollover (device scan) | 5 | E |
| `filters.label_duplicates` | lexicographic sort on selected dims → adjacent-compare | 2 | E |

### 7.4 Height above ground

| Stage | GPU strategy | Ph | Val |
|---|---|---|---|
| `filters.hag_nn` | kNN over ground-classified points, IDW of ground Z | 3 | N |
| `filters.hag_delaunay` | CPU ground TIN (delaunator) → device triangle grid → parallel locate + barycentric interp | 3 | N |
| `filters.hag_dem` | raster/texture lookup, `pp` | 3 | N |

### 7.5 Pointwise neighborhood features — the flagship family
All are clients of §5.4 + §5.5; each is "one query + registers of math." Ph 2, Val N unless noted.

| Stage | Notes |
|---|---|
| `filters.normal` | covariance → eigen3x3 → normal + curvature; canonical kernel, tune first |
| `filters.eigenvalues` | same accumulation, emit λ₁..λ₃ |
| `filters.covariancefeatures` | full Weinmann feature set (linearity, planarity, verticality, omnivariance, eigenentropy, …) from one eigen pass |
| `filters.approximatecoplanar` | eigen-ratio threshold |
| `filters.estimaterank` | svd3 rank with tolerance |
| `filters.optimalneighborhood` | distance-sorted neighbors + prefix-covariance sweep over k (§5.5) — biggest single speedup in the library |
| `filters.planefit` | plane fit residual |
| `filters.miniball` | Welzl smallest-enclosing-ball on k neighbors (registers) |
| `filters.nndistance` | kth/avg NN distance |
| `filters.reciprocity` | mutual-kNN percentage (bitset handshake) |
| `filters.radialdensity` | radius count × volume⁻¹ |
| `filters.zsmooth` | neighbor Z median/percentile |
| `filters.m3c2` | two-epoch cylinder search along normals; Ph 5; heavy but embarrassingly parallel |

### 7.6 Clustering & segmentation

| Stage | GPU strategy | Ph | Val |
|---|---|---|---|
| `filters.cluster` | implicit radius graph → GPU connected components (ECL-CC style label propagation/union-find) | 3 | S |
| `filters.dbscan` | core/border classification via radius counts + CC | 3 | S |
| `filters.lloydkmeans` | standard GPU k-means (assign kernel + segmented centroid reduce) | 5 | S |
| `filters.litree` | shared-index seed search plus CUDA frontier region growing; retain serial control on host only where profiling proves it irreducible | 5 | S |
| `filters.supervoxel` | iterative constrained clustering over the grid | 5 | S/Q |

### 7.7 Ordering

| Stage | GPU strategy | Ph | Val |
|---|---|---|---|
| `filters.sort` | CUB radix/merge by dimension | 1 | E |
| `filters.mortonorder` | already the index build path — expose it | 1 | E |
| `filters.randomize` | reproduce the pinned PDAL RNG/seed/order exactly; optional Philox shuffle only in explicit `--fast` mode | 1 | E |

### 7.8 Registration & transforms

| Stage | GPU strategy | Ph | Val |
|---|---|---|---|
| `filters.icp` | per-iteration NN correspondences via grid → 3×3 cross-covariance reduce → Umeyama/svd3 → iterate | 5 | N |
| `filters.cpd` | truncated-Gaussian EM: responsibilities via radius queries, solves via cuBLAS | 5 | N |
| `filters.teaser` | preserve the configured external TEASER++ plugin through an exact optimized bridge; accelerate transferable correspondence work or record GPU-inapplicability | 5 | Q |
| `filters.transformation` | 4×4 in fp64 offset math + fp32 local, `pp` | 1 | N |
| `filters.reprojection` | batched PROJ on host (pinned, overlapped) **plus** native fp64 device kernels for the common set: geographic↔geocentric, UTM/TM, LCC, Web Mercator, 7-param datum. fp64 per-point transforms are fine (~50 GFLOP per 100M pts ≈ 40 ms at 1.3 TFLOPS); it's fp64 in *loops* that's banned. | 2 | N |
| `filters.projpipeline` | batched host PROJ | 2 | N |
| `filters.straighten` | closest-segment param via small polyline BVH, `pp` | 5 | N |
| `filters.georeference` | pose interpolation (binary search on GpsTime) + rigid transform, `pp` | 5 | N |
| `filters.h3` | batch cell math on device when faster; otherwise use an exact optimized h3-library bridge with an accepted GPU-inapplicability record | 5 | E |

### 7.9 Culling

| Stage | GPU strategy | Ph | Val |
|---|---|---|---|
| `filters.crop` | bbox: fused `pp` predicate (Ph 1). Polygon: device edge grid + crossing test (Ph 3). | 1/3 | E |
| `filters.geomdistance` | 2D distance via polygon edge grid | 3 | N |
| `filters.expression` / `filters.range` / `filters.mongo` | predicate → bytecode VM → CUB stream compaction. `range` is implemented as an alias over `expression` (PDAL deprecates it at 3.0 — do not build it twice). `mongo` parses to the same bytecode. | 1/1/5 | E |
| `filters.decimation` / `filters.head` / `filters.tail` | trivial compaction | 1 | E |
| `filters.locate` | argmin/argmax reduce | 1 | E |
| `filters.dem` | raster-tolerance cull, `pp` | 3 | E |
| `filters.sample` | Poisson-disk via voxel occupancy; parallelize with phase-group cell coloring (Wei 2008) — 27 independent batches per sweep | 2 | S + min-dist property (exact) |
| `filters.relaxationdartthrowing` | iterate `sample` with shrinking radius | 5 | S |
| `filters.fps` | k sequential selections, O(N) parallel min-distance update each — fine for k ≤ 10⁵ | 5 | S |
| `filters.voxelcenternearestneighbor` / `voxelcentroidnearestneighbor` / `voxeldownsize` | hash-grid segmented argmin / two-pass centroid / occupancy-first | 2 | E |
| `filters.griddecimation` | 2D cell argmax/argmin via atomics | 3 | E |

### 7.10 Partitioning, splitting, merging

| Stage | GPU strategy | Ph | Val |
|---|---|---|---|
| `filters.splitter` | snap-to-grid binning → partition | 1 | E |
| `filters.divider` / `filters.groupby` / `filters.returns` | partition by count / category / return flags (CUB partition) | 1 | E |
| `filters.separatescanline` | edge-flag scan segmentation | 2 | E |
| `filters.chipper` | recursive median splits via repeated sort — gathered mode | 5 | S |
| `filters.merge` | concatenation | 1 | E |

### 7.11 Metadata & stats

| Stage | GPU strategy | Ph | Val |
|---|---|---|---|
| `filters.stats` | fixed-order reductions, histograms, and all upstream moments with exact metadata emission | 1 | N |
| `filters.expressionstats` | predicate VM + segmented counts | 1 | N |
| `filters.info` | reductions + query | 1 | E |
| `filters.hexbin` | hex-cell occupancy on device; boundary trace on host (cheap) | 3 | S/Q |

### 7.12 Meshing & surfaces

| Stage | GPU strategy | Ph | Val |
|---|---|---|---|
| `filters.delaunay` | exact delaunator bridge while gate G4 evaluates gDel2D; a failed gate requires a measured GPU-inapplicability record | 5 | Q |
| `filters.greedyprojection` | shared-index CUDA neighborhood work with measured host control for irreducibly sequential frontier decisions | 5 | Q |
| `filters.poisson` | exact CUDA multigrid/solver path or an evidence-backed optimized external bridge; neither route is deferred past the full-product milestone | 5 | Q |
| `filters.faceraster` | device triangle rasterization, barycentric Z — the GPU's home turf | 3 | N |

### 7.13 Language & callback hooks

| Stage | Strategy | Ph |
|---|---|---|
| `filters.python` | host callback that can hand **device** arrays to user code via `__cuda_array_interface__` (CuPy/torch operate in place — a capability CPU PDAL cannot offer) | 6 |
| `filters.streamcallback` | C++ host callback at tile granularity | 6 |
| `filters.matlab` / `filters.julia` | preserve configured upstream behavior; optimize bridge/residency and decide GPU applicability with evidence | 6 |

### 7.14 Applications (`gpupal <verb>`)

`translate`, `pipeline`, `info`, `merge`, `sort`, `split`, `tile`, `tindex`, `random` — thin wrappers over stages, delivered with their stages. `ground` wraps the §7.1 defaults (Ph 3). `density` wraps hexbin (Ph 3). `hausdorff`, `chamfer`, `delta` — cross-cloud NN distance metrics on the shared index (Ph 5, N). `eval` — classification confusion metrics via segmented counts (Ph 5, E). `bench` — §10 harness (Ph 1).

### 7.15 Writers with real algorithms

| Writer | Strategy | Ph | Val |
|---|---|---|---|
| `writers.gdal` | DTM/DSM binning: mean/min/max/idw/count/stdev via privatized-tile atomics → global raster → GDAL out | 3 | N |
| `writers.raster` | writes `faceraster` output | 3 | N |
| `writers.las` | device SoA → record pack kernel → disk | 0/1 | E |
| `writers.laz` / `writers.copc` | chunk-parallel CPU encode; COPC octree assignment on device | 4 | E |
| `writers.text` / `ply` / `pcd` / `ogr` / `null` | straightforward | 1–5 | E |
| everything else (`pgpointcloud`, `tiledb`, `nitf`, `e57`, …) | libpdal bridge | 4 | — |

## 8. PDAL compatibility

- **Pipeline JSON and applications:** accept the pinned PDAL surface verbatim:
  stage strings, option names and defaults, tags/inputs, graph behavior,
  application arguments, plugins, and failures. A temporary bridge may carry
  an unported configured stage during development, but it does not close the
  catalog row or the full-product milestone.
- **Default observable contract:** output bytes, point and view order, dimension
  types, metadata, filenames, stdout/stderr, exit status, warnings, and failure
  boundaries match the pinned oracle. Tiling, batching, fusion, and device
  residency are implementation details and may not create a semantic delta.
  `where` and `where_merge` retain their exact PDAL behavior. Any reordered,
  approximate, or best-effort behavior belongs only to explicit `--fast` mode.
- **Conformance suite:** run the complete published upstream test procedure plus
  generated option/error matrices and representative real pipelines through
  both engines. Compare complete process artifacts, not only point-set or
  numeric equality; Autzen, 3DEP, and the registered local corpora provide
  scale and format diversity.

## 9. APIs

- **C++:** a stage is (descriptor + kernels + tests + bench entry + doc page), registered via macro; adding one touches only `src/stages/<family>/` plus a catalog row. The §13.1 checklist is the definition of done.
- **Python (`import pdg`):** `Pipeline.from_json(...)`, a fluent builder (`pdg.read("in.copc").filter("filters.smrf", slope=0.2).raster("dtm.tif", resolution=0.5)`), `execute()` returning dimension arrays that expose `__cuda_array_interface__` and DLPack (zero-copy into CuPy/PyTorch), with `.to_numpy()` for host copies. GIL released during execution.
- **C ABI:** post-1.0.

## 10. Performance targets & benchmark harness

Reference workload: 100 M-point ALS tile, k = 16, fp32 local frame, NVMe ≥ 7 GB/s, RTX 4090. These are **targets, not guarantees** — revise only via a `BENCHMARKS.md` entry with profile evidence, never silently.

| Operation | Target |
|---|---|
| LAS parse → device SoA | ≥ 5 GB/s |
| Morton sort + grid build, 100 M pts | ≤ 250 ms |
| `normal` (k=16), incl. query | ≤ 600 ms |
| `covariancefeatures` full set | ≤ 800 ms |
| `outlier` (SOR) | ≤ 500 ms |
| `smrf` end-to-end | ≤ 2.5 s |
| `csf` (500 iterations) | ≤ 3 s |
| voxel downsample | ≤ 200 ms |
| `sort` by dimension | ≤ 150 ms |
| DTM rasterize 0.5 m (`writers.gdal`) | ≤ 300 ms |
| E2E `dtm.json`: LAZ → assign → outlier → smrf → hag → DTM GeoTIFF | ≤ 30 s (CPU LAZ decode) / ≤ 10 s (if G2 passes) |

Standing rules:

- Every `pp`/`red` kernel ≥ 70% of DRAM roofline (ncu-verified) or a written justification exists.
- Every automatically selected CUDA path must beat the fastest exact host/PDAL
  path end-to-end with a predeclared confidence rule; otherwise the selector
  uses the host path. A 10× gain remains a target for compute-heavy P2/P3
  stages, not permission to launch CUDA when it makes a real pipeline slower.
- Reference pipelines live in `bench/pipelines/` (`dtm.json`, `features.json`, `denoise.json`, `downsample.json`) and run nightly on the dev GPU; smoke subset runs per-PR with a ±5% regression gate.

## 11. Validation & testing policy

- **Tier E (exact default, mandatory for every catalog row):** complete process
  equivalence to the pinned oracle: every output byte and filename, point/view
  order, metadata, stdout/stderr, exit status, warning, and failure boundary.
- **Tier N (supplemental):** numeric error and conditioning properties for
  explicit `--fast` algorithms or internal invariant tests; tolerances never
  authorize a default-mode mismatch.
- **Tier S (supplemental):** statistical quality and algorithm-health metrics on
  fixture tiles for explicit `--fast` modes; the same stage still needs Tier E
  in compatibility mode.
- **Tier Q (supplemental):** geometry/mesh invariants and rendered-artifact
  review in addition to, never instead of, exact default conformance.
- Property tests per family (for example normals are unit length, culls are
  subsets, and voxel filters emit at most one point per voxel) complement exact
  process differentials. Fuzz all readers with libFuzzer. Compute Sanitizer
  memcheck, initcheck, racecheck, and synccheck plus ASan/UBSan are merge gates.
  Default mode has schedule-perturbation and bitwise-repeatability tests.

## 12. Phased delivery plan (exit criteria are the phase)

- **P0 — Scaffold.** Repo, CMake presets, CI (build + host tests in CUDA container; GPU jobs on a self-hosted 4090 runner), allocator, `PointBatch`, dimension registry, LAS read/write with device round-trip. *Exit:* `gpupal translate in.las out.las` preserves payload byte-for-byte; CI green.
- **P1 — Per-point core.** Expression parser + bytecode VM; fused `pp` engine; assign/ferry/colorinterp/transformation; compaction culls (expression, decimation, head/tail, locate, iqr/mad, crop-bbox); sort/mortonorder/randomize; partition family (splitter, divider, groupby, returns, merge); stats/info/expressionstats; `gpupal bench` harness + PDAL baselines. *Exit:* fused `pp` chains ≥ 70% DRAM roofline; bench suite + goldens green.
- **P1.5 — Resident execution.** The following milestone directive is normative and recorded verbatim:

```text
INTERRUPT — STOP CURRENT TASK. NEW MILESTONE DIRECTIVE: P1.5 "RESIDENT EXECUTION"

0. IMMEDIATE ACTIONS
- Halt in-flight work. Commit WIP to a branch; merge nothing that adds a stage port.
- Commit this directive verbatim into the repo as a spec amendment: insert milestone
  P1.5 between P1 and P2 in SPEC §12, and add a DECISIONS.md entry citing the
  benchmark review that triggered it (stage-local round trips losing to host).
- From this commit forward, SPEC + DECISIONS.md are the sole source of truth.
  Prior chat/session context does not override them.
- Reply first with a short plan mapping D1–D4 below to intended PRs. Then execute.

1. SCOPE FREEZE
Pause breadth-first stage ports. New stages may proceed only when they exercise or
extend the resident execution architecture; do not add CUDA paths that require
avoidable host round trips between compatible stages.

2. DELIVERABLES
D1  Planner-owned device residency AND liveness across compatible stages, with
    explicit spill/fallback boundaries. Liveness = the planner tracks which columns
    are still consumed downstream, frees/reuses device buffers mid-plan, and feeds
    this into the VRAM budget estimator.
D2  Point-program fusion into surrounding producer/consumer kernels (prologue/
    epilogue of readers, feature kernels, writers) where declared semantics permit.
    Fusion legality lives in stage descriptors, not planner case analysis: add
    declared flags {pure, cardinality_preserving, dims_read, dims_written,
    fusable_as_prologue, fusable_as_epilogue, deterministic_safe}. `where`/
    `where_merge` and --deterministic interact with fusion and must be declared.
D3  Bounded tiled multi-lane scheduler. Stream count is a swept parameter
    (N ∈ 2..6) fixed as a benchmarked default per pipeline class. No autotuner.
D4  Plan-level host/device placement model accounting for transfer bytes, packing,
    index construction/reuse, synchronization, cardinality, memory limits, and CUDA
    startup. Implementation bound: a linear cost model with coefficients measured
    from the existing standalone round-trip matrix. Revise coefficients only on
    demonstrated wrong placement calls that matter. No general optimizer.
    Coefficients for a machine other than the reference come from measured
    profiles only, never from the runtime: a local profile written by the
    explicit `gpupal calibrate` command (D0277), which re-measures the same
    calibration cases there and is keyed to that exact machine; failing that,
    a shipped GPU-class profile measured by the same command on a rented
    machine of that GPU model / compute capability / toolkit and admitted
    only where the device won by the shipping margin (1.7x, D0280) at every
    measured size (D0279); failing that, a generic fallback that is the
    intersection of the shipped profiles under a 3x margin, inside its measured
    compute-capability and memory bounds (D0279). The runtime never times,
    fits, or tunes anything; a machine no tier covers stays on the host path.

3. VALIDATION LOADS (all must run under the new machinery)
V1  Fused assign/ferry/expression chains.
V2  approximatecoplanar and the shared neighborhood family, via the resident
    shared-index interface.
V3  LOF only through the resident shared-kNN interface.
V4  One mixed pipeline deliberately crossing an unsupported-stage (host) boundary.
V5  One dataset large enough to force tiling (exceeds the VRAM working-set budget).
V6  One pipeline with an XYZ-mutating stage (transformation or reprojection)
    BETWEEN two neighborhood stages — exercises index dirty-tracking and rebuild
    accounting; rebuild counts reported in --stats.
V7  One deliberately small dataset where the placement model must select host
    (negative control).

4. BENCHMARK CLASSES (report separately in BENCHMARKS.md)
B1  Resident kernel throughput with ncu roofline/occupancy evidence.
B2  Exact end-to-end pipeline wall time vs same-machine pinned-version PDAL —
    the acceptance metric.
B3  Standalone round-trip measurements — retained ONLY as calibration data for the
    D4 placement model. Never again cited as a stage verdict.

5. EXIT CRITERIA (all required before P1.5 closes; no partial credit)
E1  Fused pp chains ≥ 70% of DRAM roofline on resident data (restores the original
    P1 exit criterion).
E2  V4 executes with boundary crossings planned and reported; outputs match their
    validation tiers.
E3  V5 tiled output matches the untiled reference within tier tolerances (Tier N:
    rel 1e-4 / abs 1e-6 in the local frame unless a stage documents otherwise).
    Bitwise equality is NOT required outside --deterministic; do not chase it.
E4  V6 rebuild count equals the planner's prediction and appears in --stats.
E5  The D4 model predicts the measured host/device winner on ≥ 90% of the existing
    stage matrix.
E6  approximatecoplanar requalified under resident shared-index execution: passes
    Tier-N differentials and end-to-end break-even vs host. Treat the standing
    2.16x result as positive evidence, not a reject.
E7  Marginal cost of point-program stages inside fused chains is measured, not
    assumed: report occupancy/register delta and added DRAM traffic (from newly
    touched dims) per fused op, logged per chain.

6. AFTER THE MILESTONE
Rerun the full stage matrix under B1/B2/B3. Resume catalog-wide ports strictly
through the resident interface. Watch for the two known fusion failure modes as
ports resume: register-pressure occupancy cliffs, and dimension-footprint growth
adding DRAM traffic to otherwise-free fused ops.
```

- **P2 — Spatial engine + feature family.** Grid + LBVH, query API, §7.5 features, outlier, sample, neighborclassifier, radiusassign, label_duplicates, separatescanline, voxel filters, reprojection. *Exit:* `normal` meets target on 100 M pts; gate **G1** decided; gate **G6** (fp32-local audit) passed.
- **P3 — Terrain & grid.** smrf, pmf, csf, elm, skewnessbalancing, dem, hag family, griddecimation, cluster/dbscan, crop-polygon/overlay/geomdistance, faceraster, hexbin, writers.gdal DTM path, `ground`/`density` verbs. *Exit:* `dtm.json` E2E ≤ 30 s; exact gates and supplemental Tier-S health thresholds met.
- **P4 — I/O acceleration.** Chunk-parallel CPU LAZ, COPC read/write, EPT, PLY/PCD/text, libpdal bridge, cuFile gate **G5**, GPU-LAZ experiment gate **G2**. *Exit:* LAZ decode saturates all cores with overlapped upload; COPC pushdown correct; gates decided.
- **P5 — Heavy, global, and catalog closure.** icp, cpd, fps, lloydkmeans, m3c2, relaxationdartthrowing, sparsesurface, chipper, hausdorff/chamfer/delta/eval, straighten, georeference, gpstimeconvert, h3, mongo, delaunay (+gate **G4**), poisson, greedyprojection, litree, supervoxel, trajectory, teaser plugin, and every remaining configured driver. *Exit (revised by D0208):* every stage is functionally supported with exact default output and optional-plugin configurations covered, and every stage appearing in a reference pipeline runs by its measured-fastest correct backend. A nonzero upstream-fallback count is no longer a defect; CUDA coverage of stages absent from the reference pipelines is optional catalog coverage rather than a release blocker.
- **P6 — Release.** Python package + zero-copy interop, docs site with per-stage parity notes, conformance report vs PDAL, packaging (pip wheels with CUDA constraints; conda later), README with headline benchmark table, `v0.1.0`.

Cross-phase resequencing requires a `DECISIONS.md` entry.

## 13. Agent working agreements

### 13.1 Definition of done — any stage
1. Descriptor with PDAL-compatible name/options; plan metadata (`reads/writes/kind/max_radius`) correct.
2. Kernels follow §4 conventions; NVTX range present.
3. Unit tests + byte-exact complete-process differential are mandatory; the
   catalog's supplemental numeric/statistical/qualitative tier is added where
   useful and any tolerance is documented.
4. nvbench entry; numbers vs PDAL baseline appended to `BENCHMARKS.md`.
5. Doc page with PDAL parity, option coverage, residency, fallback status, and
   any explicitly requested nondefault `--fast` semantics.
6. ASan/UBSan and all four Compute Sanitizer tools clean on the relevant path.

### 13.2 Performance discipline
- Profile before optimizing; optimize the roofline gap, not intuition. Every kernel PR includes ncu captures (SM throughput, DRAM throughput, occupancy, % of roofline) pasted in the PR and logged.
- No kernel merges below 50% of its applicable roofline without a `DECISIONS.md` justification.
- Per-PR smoke bench with ±5% regression gate; full suite nightly.

### 13.3 Code discipline
- Prefer CUB/Thrust/libcu++; hand-rolled kernels require a benchmark proving they beat the primitive.
- Tests first for anything numeric. PR diffs ≤ ~800 LOC where feasible. Third-party code only via `third_party/` with license preserved.

### 13.4 When blocked or ambiguous
Choose the option that benchmarks faster on the reference device, record one paragraph in `DECISIONS.md`, and keep moving. Do not stall on questions this spec can't answer; do not silently deviate from things it does answer.

## 14. Risk register & decision gates

| Gate | Question | Criterion | When |
|---|---|---|---|
| **G1** | hash grid vs LBVH — keep both? | Benchmark 3 density profiles; kill a structure only if the survivor is within 10% on its worst profile | end P2 |
| **G2** | GPU LAZ decode | Go if ≥ 2× over 32-thread CPU decode on 100 M-pt COPC | P4 |
| **G3** | NVRTC JIT vs bytecode VM | Adopt if ≥ 1.3× on expression-heavy suite | P5+ |
| **G4** | GPU Delaunay (gDel2D) | Adopt if robust on all fixture tiles and ≥ 5× CPU | P5 |
| **G5** | cuFile/GDS | Keep if ≥ 1.5× on cold-cache COPC reads | P4 |
| **G6** | fp32-local sufficiency | Max abs error vs fp64 reference ≤ 1e-3 m across continental-scale UTM fixtures; else double-single arithmetic in affected kernels | end P2 |

Known risks: register pressure in fused kNN kernels (mitigate with a `__launch_bounds__` tuning table per arch); atomic contention on dense rasters (privatized shared-memory tiles before global merge); VRAM fragmentation (single pool, stream-ordered); toolkit/driver drift (pinned CI container).

## 15. Licensing & naming

- **License: BSD-3-Clause** (matches PDAL; maximally permissive for an open-source launch).
- Porting provenance rules: PDAL source (BSD) may be referenced/ported with attribution in `NOTICE`; verify lazperf's license before vendoring; **never** port LASzip (LGPL) — the LAZ format spec is public, clean-room from the spec if lazperf can't be used; every algorithm page cites its paper (Appendix A).
- **Naming:** "PDAL" is the upstream project's identity. GPUPAL is the chosen
  distinct product name and must not imply upstream endorsement. Availability
  and trademark review of GPUPAL remain public-launch gates; see D0288.

## Appendix A — Algorithm references

SMRF: Pingel, Clarke & McBride 2013 · PMF: Zhang et al. 2003 · CSF: Zhang et al. 2016 · Skewness balancing: Bartels & Wei 2010 · Tree segmentation: Li et al. 2012 · DBSCAN: Ester et al. 1996 · k-means: Lloyd 1982 · LOF: Breunig et al. 2000 · Optimal neighborhoods: Weinmann et al. 2015 · M3C2: Lague et al. 2013 · Poisson reconstruction: Kazhdan et al. 2006 · ICP: Besl & McKay 1992; Umeyama 1991 · CPD: Myronenko & Song 2010 · TEASER: Yang et al. 2020 · LBVH: Karras 2012 · 3×3 eigensolvers: Kopp 2008 · 3×3 SVD: McAdams et al. 2011 · Parallel Poisson-disk sampling: Wei 2008 · GPU connected components: ECL-CC, Jaiganesh & Burtscher 2018 · GPU Delaunay: gDel2D, Cao et al. 2014.

---
*v0.1 — expect revision after P2 gates. Amendments happen in this file via PR, with a `DECISIONS.md` pointer.*
