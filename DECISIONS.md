# Decisions

This log is append-only. New entries may supersede older ones but must not edit
their historical text.

## D0001 — Upstream-backed implementation branch

- Date: 2026-08-05
- Status: accepted
- Context: The project specification calls for a ground-up CUDA-native engine,
  while the requested starting point is a fork of PDAL.
- Decision: Seed `gpu-main` from `PDAL/PDAL` master and keep the official
  repository as `upstream`. Build `pdg` alongside the pinned upstream code so
  it remains available as the conformance oracle and CPU fallback. New public
  APIs live under the `pdg` namespace and spec-defined directories; they do not
  masquerade as an endorsed upstream PDAL release.
- Consequences: The initial tree is larger than a clean-room repository, but
  exact differential testing, fallback coverage, attribution, and upstream
  synchronization are substantially safer. Public naming remains a release
  gate.

## D0002 — Byte-for-byte compatibility is the default contract

- Date: 2026-08-05
- Status: accepted; supersedes conflicting compatibility language in spec v0.1
- Context: Spec v0.1 permits reordered output, floating atomic
  nondeterminism, and tiered numeric/statistical agreement. The project owner
  subsequently required the final product to be bit-for-bit identical to
  current PDAL outputs.
- Decision: Pin a specific upstream PDAL commit and execution environment as
  the oracle. Default mode must reproduce all observable bytes and failures.
  When an exact CUDA path does not yet exist, use the pinned host fallback.
  Potentially faster reordering, approximate math, or nondeterministic atomics
  are allowed only in an explicit `--fast` mode with separately labeled tests
  and benchmarks.
- Consequences: Some GPU algorithms will need fixed operation order, integer
  accumulation, deterministic tie-breaking, or host execution and may miss the
  original speed targets. Optimization now means maximum throughput under the
  exactness constraint. The oracle is deliberately pinned because the phrase
  "current PDAL" is otherwise non-reproducible as upstream changes.

## D0003 — PDG-only CLI commands use an explicit namespace boundary

- Date: 2026-08-05
- Status: accepted
- Context: The product needs diagnostics and benchmarking commands that have
  no upstream PDAL equivalent, while standard PDAL invocations—including no
  arguments and `--version`—are part of the exact compatibility surface.
- Decision: Standard PDAL commands and flags retain oracle behavior. Product
  extensions are explicit command names, currently `pdg version` and
  `pdg doctor`, with future `pdg bench` and `--fast` behavior documented before
  activation. They are not presented as byte-compatible PDAL invocations.
- Consequences: Scripts can use normal PDAL syntax and receive exact fallback
  behavior during incremental delivery. Product diagnostics remain available
  without silently shadowing a standard flag.

## D0004 — Native execution is gated by a proven compatibility envelope

- Date: 2026-08-05
- Status: accepted
- Context: Upstream LAS behavior includes not only record conversion but also
  warnings for malformed-yet-readable headers, VLR/EVLR handling, SRS rules,
  overwrite semantics, and option-dependent defaults. Accelerating a broader
  input set before reproducing those observables would violate D0002.
- Decision: The first native path handles only the default three-argument
  `translate` form, absent output, uncompressed `.las` input, no VLRs or
  EVLRs, valid format/version combinations, formats 0–3 and 6–8, and the
  required WKT flag for formats 6–8. Everything else executes the pinned PDAL
  oracle. Expansion requires a differential test that proves output,
  diagnostics, and status for the new boundary.
- Consequences: The accelerated surface starts deliberately narrow but is
  safe by construction. Generated malformed cases guard fallback selection,
  while valid generated and real-corpus cases guard the native path. The
  envelope can grow monotonically without weakening default compatibility.

## D0005 — CUDA LAS translation remains experimental until the GPU gate runs

- Date: 2026-08-05
- Status: accepted
- Context: This workstation compiles CUDA 13.3 code but its NVIDIA driver does
  not expose a device. Enabling an unmeasured GPU path by default could make
  the already-fast exact host path slower or ship a device-only correctness
  defect.
- Decision: Compile a complete CUDA record converter for canonical 0.01-scale,
  zero-offset inputs in the D0004 formats. It uses bounded pinned staging,
  stream-ordered device allocations, integer min/max and return-count
  reductions, and full format-7 record emission. The public CUDA API and
  device test matrix are available now; CLI selection requires the internal
  `PDG_EXPERIMENTAL_CUDA_TRANSLATE` environment gate. Automatic selection is
  prohibited until the RTX 4090 lane passes exact differential tests,
  memcheck/racecheck/initcheck/synccheck, and same-machine performance trials.
- Consequences: CUDA development and CI compilation continue without risking
  default behavior. A missing driver or runtime error under the experimental
  CLI gate falls back to the exact native host path. The first implementation
  is deliberately bounded and single-stream; overlap and automatic thresholds
  follow profile evidence rather than assumption.

## D0006 — Reader robustness uses deterministic generated matrices

- Date: 2026-08-06
- Status: accepted
- Context: Reader robustness still needs adversarial boundary coverage, but
  mutation-based campaigns create avoidable workflow and review friction for
  this geospatial performance project.
- Decision: Use declarative structure-aware generators, exhaustive field
  boundary matrices, sanitizers, read-only corpus replay, and named regression
  recipes. Each case has bounded input size and work, a fixed seed or explicit
  byte recipe, and oracle-consistent acceptance/rejection criteria.
- Consequences: Failures remain exactly reproducible and reviewable without a
  mutation engine. Weekly stress expands enumerated combinations and fault
  injection; parser changes cannot land without preserving every prior named
  regression.

## D0007 — Exact host translation maps files and specializes canonical records

- Date: 2026-08-06
- Status: accepted
- Context: The first exact host writer allocated complete input and output
  vectors and decoded every field even when a LAS 1.4 format-7/8 record already
  matched the default 36-byte output layout. Those copies and conversions were
  the measured bottleneck on the 21.97-million-point reference case.
- Decision: Map regular LAS input read-only and translate directly into a
  final-sized adjacent output mapping. Publish the output only after successful
  completion, preserving the oracle's no-overwrite behavior. For format 7 or 8
  with 0.01 XYZ scales and zero offsets, copy the first 36 record bytes exactly
  while reducing raw integer bounds and return counts. Keep the general field
  conversion for all other supported inputs. Embed the pinned oracle commit in
  observable PDAL writer metadata for `WITH_PDG` builds. Same-machine trials of
  2, 4, 5, 6, 8, 12, and 24 workers select five as the canonical-copy default;
  `PDG_NATIVE_WORKERS` remains an internal measurement override.
- Consequences: The native path avoids whole-file input/output vectors and
  reaches the specification's host-throughput target in exploratory testing.
  Automatic, single-worker, and many-worker output hashes are identical, and
  both Release and ASan/UBSan match the pinned PDAL bytes on the 790,953,999-byte
  case. Other machines may retune the internal worker override until a portable
  topology-aware policy is implemented.

## D0008 — The first P1 ferry path uses a conservative logical-value envelope

- Date: 2026-08-06
- Status: accepted
- Context: `filters.ferry` looks like a byte copy, but upstream preparation
  resolves the source and destination layout types to a common type. This can
  widen even an existing standard destination. Filters also observe reader
  logical values before the LAS writer quantizes coordinates and scan angles;
  applying ferry after producing canonical format-7 records is exact only when
  those source values survive canonicalization.
- Decision: Compile pipeline JSON into a typed DAG with ordered ferry programs,
  reverse dimension liveness, residency transitions, transfer counts, and
  byte estimates. The first native CLI pipeline envelope is exactly one
  default `readers.las -> filters.ferry -> writers.las` chain within D0004.
  It accepts only standard dimensions already present in the reader layout,
  mappings that leave upstream's resolved destination type unchanged, modern
  scan-angle sources, and original XYZ sources whose scale is 0.01 and offset
  is zero. Scan-angle destinations, custom dimensions, widening mappings,
  legacy scan-angle sources, rewritten-coordinate sources, and other pipeline
  shapes use the pinned PDAL fallback. Coordinate destinations remain native;
  bounds and logical return histograms are recomputed when affected.
- Consequences: Every accepted mapping can execute over the proven canonical
  writer without changing what the filter observed. The envelope is narrower
  than the ferry kernel's numeric capability, but unsupported cases remain
  exactly correct through fallback. Host execution covers every numeric
  physical type and preserves mapping order. The CUDA ordered program compiles
  for the production architecture matrix and has device equivalence tests, but
  automatic device selection waits for the physical-GPU sanitizer and
  performance gate.

## D0009 — Ordered point stages share one exact expression program

- Date: 2026-08-06
- Status: accepted; extends D0008 and records the physical-GPU correctness
  outcome required by D0005
- Context: Upstream `filters.assign` evaluates statements in declaration order
  through double-valued expressions, preserves an existing destination when a
  numeric cast fails, and exposes parser, NaN, coordinate, and layout details
  that a conventional optimizing expression compiler could change. Adjacent
  assign and ferry stages would also decode and pack the same point repeatedly
  if implemented as independent loops. CUDA transcendental functions and even
  unconstrained NaN arithmetic are not automatically byte-identical to the
  host implementation.
- Decision: Compile assign expressions to a typed postfix program and lower
  ordered ferry mappings into that same representation. The host VM implements
  the complete pinned grammar and math-function set. A bounded CUDA VM accepts
  only the arithmetic, comparison, logical, and classification operations
  proven bit-identical on the reference GPU; coordinates, transcendental
  functions, excessive stack/program sizes, and unproven operations select the
  exact host VM. CUDA double operations use explicit round-to-nearest
  intrinsics, division-by-zero emits the pinned quiet-NaN bits, and unary
  negation toggles the IEEE-754 sign bit explicitly. Linear native LAS
  pipelines lower every adjacent assign/ferry stage into one program and make
  one logical-value batch pass after canonical translation. Inputs whose
  original coordinate or scan-angle semantics cannot survive canonicalization
  still use pinned PDAL.
- Consequences: Ordered dependencies and custom intermediate dimensions fuse
  without changing observable output, while unsupported syntax and pipeline
  shapes retain exact fallback. The native CLI point-program slice currently
  uses the mapped host batch path; the CUDA VM is available for device-resident
  batches and planner selection without forcing a transfer that has not yet
  won a profile gate. On the GeForce RTX 4090 (SM 8.9), Debug and Release CUDA
  suites match host bytes, including signed zero, subnormals, extrema,
  infinities, quiet/signaling NaNs and payloads. Compute Sanitizer memcheck,
  initcheck, racecheck, and synccheck report zero errors or hazards. D0005's
  device-correctness prerequisite is satisfied, but automatic CUDA LAS
  translation remains experimental until same-machine performance trials show
  a win over the mapped host path.

## D0010 — Fuse exact point programs on device and select CUDA conservatively

- Date: 2026-08-06
- Status: accepted; extends D0005 and D0009
- Context: Standalone record translation and a one-instruction ferry do too
  little work to repay CUDA startup, pinned-memory, PCIe, and output costs on
  the reference machine. The five-stage assign/ferry program crosses over near
  12 million points, while the existing host path still wins small inputs and
  simple programs. Returning canonical records to host between translation,
  expression evaluation, and packing would also erase most device benefit.
- Decision: Fuse default LAS translation, canonical AoS-to-touched-SoA
  transpose, the exact ordered assign/ferry VM, logical ReturnNumber recount,
  and canonical repack into one chunk pipeline. Custom intermediates remain in
  device SoA storage. Multi-chunk point programs alternate two independent
  pinned/device allocation pools and streams; a single-chunk job allocates one
  lane. Use 131,072-point chunks and emit completed regions through a
  synchronous sink contract. The CLI copies those regions into a bounded
  four-buffer queue whose worker performs positioned writes while subsequent
  GPU work proceeds; the exact header is emitted last and publication remains
  atomic/no-overwrite.
- Decision: Automatic CUDA selection is limited to exact-supported inputs with
  at least 16,000,000 points, five assignments, 28 VM instructions, no more
  than five written dimensions, and no more than six unique touched
  dimensions. The 16-million threshold preserves margin above the measured
  crossover. Internal force/require/disable environment switches remain for
  differential and tuning gates, but option-free users get host execution for
  simple translation, simple ferry, smaller inputs, unsupported operations,
  or any unproven layout semantics.
- Evidence: B0005 records 0.460080 s and 47.7546 million points/s for the
  automatically selected exact CUDA pipeline, 18.835x pinned PDAL, with all
  warmup and measured output hashes identical. The selector matrix records a
  1.023x CUDA/host result at 12 million points and 1.195x at 16 million; a
  full-size simple ferry reaches only 0.544x. Nsight Systems attributes 43.220
  ms to all kernels and roughly 220 ms to positioned writes in the profiled
  fused range, supporting overlap and the bounded writer rather than further
  VM specialization as the immediate optimization. Nsight Compute hardware
  counters returned `ERR_NVGPUCTRPERM`; system profiling security was not
  weakened, and no DRAM-roofline percentage is claimed.
- Consequences: The default gains a measured GPU speedup only where both
  exactness and performance are established; every other case retains the
  pinned-or-native host fallback. Expanding the selector requires a clean
  same-machine break-even matrix and new exact process differentials. The
  large all-format program test remains in normal, memcheck, initcheck, and
  synccheck lanes. Racecheck uses a compact four-launch regression that forces
  both lanes to drain and be reused, because instrumenting the exhaustive
  matrix took more than nine minutes without increasing concurrency coverage.

## D0011 — Make the PDAL fork the product and accelerate regions within it

- Date: 2026-08-06
- Status: accepted; extends D0010 and supersedes any interpretation of its
  narrow selector as the product boundary
- Context: A fast whole-pipeline LAS special case is valuable, but it is not a
  satisfactory architecture for a GPU-accelerated PDAL tool. Users must be
  able to run the complete configured PDAL catalog now, combine accelerated
  and unaccelerated stages in one graph, and gain GPU coverage monotonically
  without waiting for every long-tail stage to be ported. Sending canonical
  records back to host between every accelerated filter would preserve
  functionality but lose most device benefit. Functional availability and
  GPU-native implementation coverage therefore need separate, explicit
  accounting.
- Decision: The forked PDAL library, pipeline manager, applications, drivers,
  and plugins are the execution shell. The `pdg` entry point exposes every
  configured command: it tries a proven specialized path, then an in-process
  hybrid rewrite, then executes the sibling pinned `pdal` binary with the
  original argv. The hybrid rewriter replaces only compatible contiguous
  point-local regions and leaves every surrounding PDAL stage, option, edge,
  and failure boundary intact. A small `BatchStreamable` extension lets such a
  region consume a bounded `FixedPointTable` batch without forcing standard
  mode. Packed AoS records cross the CUDA boundary once; only touched fields
  are transposed, custom intermediates stay in device SoA storage, and only
  written fields are repacked.
- Decision: The first hybrid-native region contains ordered
  `filters.assign`, `filters.ferry`, `filters.expression`, and `filters.range`.
  The direct LAS implementation uses the same operation order plus stable CUB
  compaction, original-index gathering, post-filter bounds and logical-return
  summaries, two independent stream/allocation lanes, and exact final-file
  truncation. Type-widening ferry, unstable input order, unsupported options,
  or any preparation/layout mismatch delegate before data execution. Stage
  coverage documentation must always report both configured functionality and
  GPU-native coverage; a host fallback is never labeled CUDA-native.
- Evidence: The configured executable exposes the same 122 drivers as PDAL
  (84 filters, 24 readers, and 14 writers). The modified fork passes all
  142 published upstream tests, including remote STAC/EPT/COPC integration.
  Host and CUDA reader matrices match exact LAS output through LAS/LAZ, BPF,
  PLY, PCD, text, and sorted COPC inputs. The packed generic hybrid path reaches
  1.439x pinned PDAL on the large five-assignment/range workload. The direct
  ordered path reaches 17.347x for an all-pass range and 11.495x while stably
  retaining 3,045,832 points; output SHA-256 values match the oracle. All CUDA
  primitive families and a compact selective four-launch/two-lane integration
  case pass memcheck, initcheck, synccheck, and racecheck with zero findings.
- Consequences: PDAL functionality is usable before it is CUDA-native, while
  accelerated regions remain resident and can grow across the catalog. The
  specialized LAS path remains an important fast route, not the definition of
  the product. Subsequent work proceeds in reusable waves: remaining simple
  maps/culls/reductions, ordering, the shared spatial engine, spatial stage
  families, and broader native I/O. Claims such as “fully GPU-native PDAL” are
  prohibited until the coverage matrix actually shows that state.

## D0012 — Keep hybrid substitution out of shared-view DAGs

- Date: 2026-08-06
- Status: accepted; narrows D0011's first in-process compatibility envelope
- Context: PDAL standard-mode graph branches may share an input `PointView`.
  A replacement stage that groups several point mutations or selections can
  therefore change when a sibling branch observes those mutations, even if
  every operation is point-local and byte-exact in a linear pipeline. The
  initial rewriter classified tagged and explicit-input stages as non-linear
  but did not reject the whole rewrite when the original graph was already
  non-streamable.
- Decision: In-process point-program substitution requires a genuinely linear
  graph with exactly one reader and no more than one writer. Any tag, explicit
  `inputs`, implicit multi-reader graph, or multi-writer graph delegates the
  original pipeline before execution. Unknown or unstable reader ordering
  remains a separate conservative rejection. Branch acceleration may be
  reopened only after view ownership, execution order, and failure behavior
  have dedicated process differentials.
- Evidence: The linearity classifier now has tagged-stage and implicit
  multi-reader regressions. All 89 Host Release and 124 CUDA Release tests pass
  after the guard (one opt-in local-corpus test skips in each), and the 60-test
  ASan/UBSan unit lane is clean with that same skip.
- Consequences: The current hybrid path accelerates ordinary linear PDAL
  pipelines without claiming unproved DAG semantics. Non-linear pipelines
  retain complete functionality through unchanged PDAL, but are not yet
  counted as in-process GPU-native coverage.

## D0013 — Lower exact bounding-box crop to stable fused compaction

- Date: 2026-08-06
- Status: accepted; extends D0011's reusable point-selection wave
- Context: Bounding-box crop is a common spatial cull and a prerequisite for
  larger tiled and neighborhood pipelines. PDAL's full crop stage also supports
  multiple output views, assigned spatial references, reprojection, polygons,
  OGR geometry, and center/distance selection. Treating all of those forms as
  one kernel would mix distinct view, geometry, and error semantics before the
  shared spatial engine exists. Returning points to host merely to perform the
  common box test would break fusion with adjacent point programs.
- Decision: Compile exactly one parenthesized 2D or 3D bounds string and an
  optional Boolean `outside` into the existing ordered predicate VM. Preserve
  inclusive faces, BOX2D/BOX3D NaN behavior, input order, and exact post-filter
  LAS summaries. Fuse the predicate with adjacent assign, ferry, expression,
  range, and crop operations in both the direct LAS path and the in-process
  packed bridge. Multiple bounds, `a_srs`, centers, polygons, OGR options,
  tagged/branching graphs, malformed bounds, and all other crop forms execute
  through unchanged PDAL. Do not enable a crop-specific automatic CUDA path
  until a clean same-machine break-even matrix establishes its threshold.
- Evidence: Host tests cover inclusive 2D/3D faces, whitespace, outside
  inversion, NaNs, and unsupported grammar. Direct and hybrid Host/CUDA process
  differentials match complete artifacts with two-point CUDA chunks, selective
  output, same-SRS input, and empty standard-mode input. The generic reader
  matrix now crosses LAS, LAZ, BPF, PLY, PCD, text, and sorted COPC with crop in
  the fused region. Explicit multi-bounds, reprojection, and malformed cases
  match the original fallback. The crop device regression passes memcheck,
  initcheck, synccheck, and racecheck with zero findings, and the fork passes
  all 142 tests in the published upstream suite.
- Consequences: `filters.crop` is the fifth GPU-native filter envelope and the
  first spatial cull, while every extant crop form remains usable immediately.
  The next geometry expansion should reuse the planned shared spatial engine
  and add multi-view/process differentials rather than broadening this parser
  speculatively. Functional coverage and native coverage remain separately
  reported.

## D0014 — Preserve source identity in global selection reductions

- Date: 2026-08-06
- Status: accepted; extends D0011's P1 reduction wave
- Context: `filters.decimation`, `filters.head`, `filters.tail`, and
  `filters.locate` are small stages syntactically, but their observable result
  depends on global source order, execution mode, chunk boundaries, strict tie
  comparisons, and—in locate's case—PDAL's initial double sentinels. Replacing
  them with unordered atomics or independent per-batch filters can return the
  right numeric extreme and still select the wrong source point or output
  sequence. Standalone launches may also lose to PDAL after transfer and
  startup costs.
- Decision: Maintain global 64-bit ordinal/index state across chunks. Lower
  decimation/head/tail to exact ordered masks and stable compaction. Implement
  locate as a reusable typed candidate reduction carrying value, original
  index, input-presence, and comparability state; merge ties by the lowest
  original index. Only rewrite locate when reader order and a linear graph are
  proven. Keep all four CUDA paths force/require-only until their device
  differentials, four Compute Sanitizer tools, and same-machine break-even
  matrices pass. Unsupported options and graph/order semantics remain on
  unchanged PDAL.
- Evidence: The 23-case ordinal and 15-case locate process matrices match
  complete oracle artifacts. Host Debug and Release pass 126 of 127 registered
  tests with only the intentional local-corpus opt-in skip; the 77-test
  ASan/UBSan lane has the same single skip, and the locate matrix passes with
  libasan preloaded for both oracle and candidate. A read-only 21,970,934-point
  locate run crosses 168 chunks and emits the same 2,301-byte LAS (SHA-256
  `44e4bf505c7f304ddba014a29f3b5480dce1ea04a2ccf918d2e8a1cd650d3095`). CUDA
  Debug and Release compile the typed multi-block reductions, but managed
  execution has not supplied device access for the new runtime gates.
- Consequences: Global selections can expand on reusable exact primitives
  without being mislabeled GPU-qualified. Functional users retain every PDAL
  variant immediately, while automatic CUDA thresholds will be evidence-based
  rather than inferred from kernel compilation or theoretical parallelism.

## D0015 — Preserve simultaneous XYZ semantics in fused transformation maps

- Date: 2026-08-06
- Status: accepted; extends D0011's P1 point-map wave
- Context: `filters.transformation` is point-local, but a naive sequence of
  three assignments reads already-mutated coordinates and differs from PDAL.
  The stage also includes projective homogeneous division, matrix-file input,
  Eigen inversion, spatial-reference override behavior, coordinate-product
  invalidation, and floating-point contraction hazards. Treating a translation
  demo as the whole stage would therefore be neither functionally complete nor
  exact.
- Decision: Add a first-class fused transformation operation that snapshots
  original X/Y/Z and computes all outputs in PDAL's row-major arithmetic order.
  Implement the full non-inverted inline 4x4 projective form on host. Initially
  admit only an exact affine last row and double XYZ to CUDA, using explicit
  round-to-nearest double operations; keep automatic device selection off
  pending measurement. Delegate matrix files, inversion, SRS override, and all
  unsupported graph/layout forms to unchanged PDAL before point execution.
- Evidence: A nine-case complete-process matrix matches the pinned oracle for
  identity, affine, projective, fused-chain, empty, 198,975-point multi-batch,
  inverse fallback, SRS fallback, and file-matrix fallback cases. Host units
  cover parser and projective semantics. CUDA Debug compiles the 131,103-point
  bitwise host/device test, but the managed namespace does not expose the
  installed GPU driver, so device runtime and sanitizer evidence remain
  pending.
- Consequences: Every configured transformation form remains usable through
  `pdg`; the common inline form now participates in fusion, while only the
  proven affine subset can be forced onto CUDA. Promotion to automatic CUDA
  requires exact hardware results and a positive end-to-end break-even matrix.

## D0016 — Reproduce robust selection algorithms before accelerating them

- Date: 2026-08-06
- Status: accepted; extends D0014's P1 global-selection primitives
- Context: `filters.iqr` and `filters.mad` appear reducible to percentile
  formulas, but PDAL observes specific upper indices, independent
  `std::nth_element` copies, strict inequalities, deviation division, and
  physical-to-double conversion. NaNs, signed zero, zero MAD, negative
  multipliers, and survivor order make algebraic substitutions or a generic
  unordered percentile kernel incompatible.
- Decision: Implement the upstream host algorithms literally and keep their
  stable output order. Use CUB radix order statistics only for at most
  `INT_MAX` finite logical values without negative zero, with explicit
  deviation arithmetic and the original mask predicates. Route every other
  value/option/graph envelope to the exact host implementation or unchanged
  PDAL. Keep automatic CUDA selection disabled until device and performance
  gates pass.
- Evidence: A 16-case complete-process matrix matches the pinned oracle across
  defaults, edge multipliers, integer/floating/NaN/custom dimensions, empty
  views, fallbacks, and preparation failure. Host units pin quartile, median,
  encoded-coordinate, and device-eligibility behavior. Host Debug and Release
  each pass 139 of 140 registered tests with only the opt-in local-corpus skip;
  the 88-test sanitizer lane has the same skip, and all 16 process cases pass
  with libasan preloaded into both executables. IQR and MAD each scan a
  read-only 21,970,934-point LAS and exactly match the oracle's 2,265-byte
  empty output (SHA-256
  `24fd80a8db0b248e9b86e83560679fce458af8b12a4cd24cf4dcaa1d633b583b`).
  CUDA Debug and Release compile a 131,103-point result-bit and mask-byte
  comparison for both algorithms, and the fork accounts for all 142 published
  PDAL tests. The managed namespace still masks the installed driver, so
  device runtime and Compute Sanitizer results are pending.
- Consequences: IQR and MAD are functionally exact native host stages with a
  conservative device implementation, not generic approximate percentile
  demos. Future sort/order/statistics stages can reuse the typed logical-value
  materialization and CUB ordering path while retaining stage-specific
  compatibility predicates.

## D0017 — Treat PDAL sort tie behavior as part of compatibility

- Date: 2026-08-06
- Status: accepted; extends D0016 into the P1 ordering wave
- Context: `filters.sort` does not define one ordinary lexicographic operation.
  It uses `std::sort` for its first pass and `std::stable_sort` for later
  passes, making the last listed dimension primary. Normal-sort ties and NaNs
  are sensitive to the standard-library implementation and source order. A
  stable GPU radix permutation can therefore be numerically sorted and still
  differ byte-for-byte from PDAL.
- Decision: Reproduce the exact typed host pass sequence. Use stable CUB
  key/index radix passes on CUDA, but publish a single-key `STABLE` result only
  when keys contain no NaNs. For `NORMAL` or multi-key programs, additionally
  require the final key to be tie-free; detect ties on device before applying
  the permutation and fall back to host when found. Keep typed 64-bit integer
  keys, decode canonical int32 coordinates with explicit double arithmetic,
  preserve the original `PointView`, and leave automatic CUDA selection off
  until hardware gates pass.
- Evidence: Host and planner units pin stable ties, direction, multi-pass
  priority, coordinate decoding, eligibility, and fallback boundaries. A
  20-case complete-process matrix matches every oracle artifact across both
  algorithms, duplicate and NaN keys, dimension aliases and arrays, custom
  chains, empty and 198,975-point inputs, unsupported options, and failures.
  The complete matrix is also clean with libasan preloaded into both
  processes. CUDA Debug and Release compile three 131,103-point
  full-permutation comparisons and a data-dependent normal-tie rejection test.
  A stable Classification sort over the 21,970,934-point local corpus followed
  by a 100-point head emits the oracle's exact 5,865 bytes (SHA-256
  `f0961728c852e64d6d9e1106525913aa2361ea3348ca87f897d3a92f1f020885`).
  Device runtime and Compute Sanitizer evidence remain pending in the managed
  namespace.
- Consequences: The new radix-ordering primitive is reusable by Morton order,
  partitioning, and spatial-index construction without weakening strict sort
  compatibility. Device results with unproved tie semantics are never
  published merely because they are deterministically ordered.

## D0018 — Reproduce Morton traversal exactly and reuse it as spatial infrastructure

- Date: 2026-08-06
- Status: accepted; extends D0017 into the shared-index ordering path
- Context: `filters.mortonorder` is not equivalent to any convenient Morton
  encoder. Ordinary mode normalizes X/Y, casts to signed `INT_MAX` coordinates,
  and compares the most-significant differing axis. Reverse mode instead uses a
  `sqrt(point_count)` grid, 16-bit interleave, and full 32-bit reversal. Both
  modes traverse a `std::multimap`, so duplicate-code insertion order and the
  ordinary/reverse empty-view difference are observable. Degenerate bounds and
  nonfinite casts also retain upstream implementation behavior.
- Decision: Reproduce the pinned upstream host algorithms, with attribution in
  `NOTICE`, for every value domain. Inside a finite, nondegenerate logical-double
  envelope, encode ordinary order as an equivalent 62-bit key and reverse order
  as the exact 32-bit reversed code. Reuse the stable CUB key/index primitive so
  equal codes preserve source order. Reject device publication for out-of-bounds
  values, more than `INT_MAX` points, unsupported graph/options, or any domain
  whose exact key equivalence is unproved. Keep automatic CUDA selection off
  until hardware correctness, sanitizer, and break-even gates pass.
- Evidence: Comparator properties and the pinned reverse 4-by-4 permutation
  pass. A 15-case complete-process matrix matches every oracle artifact and is
  clean under ASan/UBSan with libasan loaded into both processes. Host Debug and
  Release account for all 153 registered tests (152 pass plus the intentional
  local-corpus skip); the 99-test sanitizer lane has the same single skip. CUDA
  Debug and Release compile all 219 registered tests and 134 `pdg` tests,
  including a 131,103-point bitwise key/permutation comparison in both modes,
  but managed device runtime remains pending. Ordinary Morton order followed by
  a 100-point head over the read-only 21,970,934-point Snow Road Twin LAS emits
  the oracle's exact 5,865 bytes (SHA-256
  `61638be4f61d806608c4ac6df4da0e0715915d0347b1dcf68b57afe1bf898c6e`).
- Consequences: Morton order is a complete exact host replacement with a
  conservative device implementation awaiting qualification, not a generic
  locality approximation. Its key-generation and stable-permutation pieces now
  feed partitioning and the shared spatial index without weakening the public
  filter's semantics.

## D0019 — Define completion as catalog-wide acceleration, not selected fast paths

- Date: 2026-08-06
- Status: accepted; extends D0011 and supersedes any interpretation of
  `spec.md` §7 or the P5 fallback language as a reduced final product boundary
- Context: Preserving all configured PDAL functionality through the fork solves
  immediate usability, but a tool with five qualified CUDA filters and a broad
  upstream fallback is not a fully implemented GPU-accelerated PDAL. A few
  unusually favorable LAS pipelines and large speedup numbers cannot substitute
  for native coverage of mixed real-world pipelines, optional plugins, global
  algorithms, spatial stages, and I/O. Conversely, forcing a GPU launch for tiny
  jobs or inherently external work would make the product slower rather than
  more accelerated.
- Decision: Treat every application, stage, option surface, graph form, and
  configured plugin present in an equivalently built upstream PDAL as the
  functional product surface. The §7 catalog determines delivery order only.
  Every stage containing meaningful data-parallel work must gain a composable
  native CUDA backend across its supported option envelopes. Inherently
  external, irreducibly serial, or end-to-end GPU-negative operations require a
  separate evidence-backed GPU-inapplicability decision and an optimized host
  or bridge implementation that preserves adjacent device residency. A
  temporary call to unchanged upstream PDAL keeps users working during the
  port, but never closes native coverage.
- Decision: Generate release coverage from the complete configured driver and
  plugin manifests, recording functional parity, option/error tests, backend,
  residency, exactness, device runtime, sanitizer status, and break-even
  evidence per row. Automatic backend selection uses CUDA only where a
  same-machine end-to-end confidence interval proves it faster; otherwise it
  chooses the fastest exact host path. Representative complete-pipeline gates
  reject material regressions. The project may not claim “fully
  GPU-accelerated PDAL” while any temporary fallback remains or any applicable
  row lacks device qualification.
- Evidence: The fork currently exposes all 122 drivers in its configured build
  and accounts for all 142 published upstream tests, so functional scaffolding
  is sound. Native status is nevertheless only five GPU-qualified filters plus
  ten implemented-device filters awaiting qualification, one direct I/O
  family, and a large remaining catalog. That gap is now an explicit workload,
  not something hidden by the successful fallback or the 18.835x benchmark.
- Consequences: Full completion expands beyond the initial v1 delivery catalog
  and includes optional-plugin build matrices. Phase reports must state the
  remaining native count and may celebrate vertical slices without presenting
  them as the product. Small inputs can deliberately run on CPU because the
  product objective is faster exact PDAL end to end, not maximal GPU launch
  count.

## D0020 — Preserve PointView identity while partitioning categorical keys

- Date: 2026-08-06
- Status: accepted; extends D0018 into the P1 partition and multi-view wave
- Context: `filters.groupby` is not just a key sort. PDAL converts every value
  through `getFieldAs<int64_t>`, creates a view when a key first appears,
  retains the map across multiple input views, and appends points in source
  order. Although the map is key-sorted, `PointViewSet` is id-sorted, so
  first-occurrence view creation controls templated writer numbering. A GPU
  implementation that emits sorted-key view order would contain identical
  points but produce differently named files and therefore violate exactness.
- Decision: Gather keys through PDAL's conversion path. The exact host path
  follows the upstream first-encounter append loop. The device path sorts
  canonical signed-64 keys and source indices with the stable CUB ordering
  primitive, but creates views in an independent source-order scan before
  appending the stable permutation. Accumulate those views across every input
  view. Limit CUDA to at most `INT_MAX` points and keep it force/require-only
  until hardware and performance gates pass. After grouping, close native
  downstream rewriting until per-view stage semantics are explicitly proved;
  an ordering barrier does not collapse views and therefore cannot reopen the
  envelope by itself. Treat every unchanged filter as potentially multi-view
  unless its single-view behavior has both a source audit and an exact mixed-
  pipeline regression; `filters.stats` is the sole current exception.
- Evidence: Planner/rewrite units cover descriptor state, custom dimensions,
  unstable readers, option fallbacks, and the downstream multi-view guard. A
  16-case complete-process matrix matches all filenames, bytes, streams,
  failures, and empty artifacts for standard/custom/negative keys, stable-sort
  and splitter predecessors, consecutive groupings, unsafe downstream-stage
  fallback, multiple input views, 198,975 points, and runtime conversion
  failure. The matrix is clean under ASan/UBSan. Host Debug and
  Release account for all 156 tests (155 pass plus the local-corpus skip); the
  101-test sanitizer `pdg` lane has the same skip. CUDA Debug and Release
  compile 223 tests and 136 `pdg` tests; device runtime remains pending because
  this namespace exposes the loaded 610.43.03 driver and RTX 4090 PCI device
  but no `/dev/nvidia*` nodes. A 21,970,934-point Snow Road Twin run emits four
  exact files totaling 790,962,684 bytes.
- Consequences: The hybrid executor now has an exact multi-view partitioning
  stage and a reusable stable categorical-key path. `filters.divider`,
  `filters.returns`, splitter binning, and merge can build on the view-identity
  tests instead of treating partitions as unordered sets. Groupby is not
  GPU-qualified until the forced process and Compute Sanitizer gates run and a
  positive end-to-end break-even includes host PointView assembly and writes.

## D0021 — Make exact catalog parity a single, non-conflicting specification

- Date: 2026-08-06
- Status: accepted; applies D0002 and D0019 directly to `spec.md` v0.2 and
  supersedes the remaining v0.1 compatibility, validation-tier, and P5 exit
  language
- Context: The mission and implementation plan required byte-exact,
  catalog-wide acceleration, but stale clauses still called point order and
  bit parity non-goals, allowed nondeterministic defaults and best-effort
  metadata, treated numeric/statistical agreement as sufficient, and closed
  the catalog with a documented CPU fallback. Those contradictions could
  recreate the narrow demonstration product rejected in D0019.
- Decision: Exact complete-process equivalence is mandatory for the default
  path of every catalog row. Numeric, statistical, and qualitative tiers are
  supplemental invariant or explicit `--fast` tests only. Automatic CUDA
  selection requires a measured end-to-end win over the fastest exact host
  route. The full-product exit permits an optimized host/bridge backend only
  for an evidence-backed GPU-inapplicable external or irreducibly serial
  operation; temporary upstream fallback does not count.
- Evidence: The reconciled specification now names bytes, filenames,
  point/view order, metadata, diagnostics, exit status, warnings, and failure
  boundaries as observable. It requires all four Compute Sanitizer tools,
  ASan/UBSan, complete published upstream testing, option/error matrices, and
  real-pipeline artifacts. The current checkpoint remains 122/122 configured
  drivers functional, five filters GPU-qualified, and ten more device-
  implemented but awaiting qualification, so this decision does not inflate
  present coverage.
- Consequences: No tolerance, set equality, deterministic-but-different order,
  or successful upstream delegation can close a default compatibility row.
  Exact host execution may still win small or serial work, but it must be the
  optimized measured product path rather than unexamined migration scaffolding.

## D0022 — Make return partitioning and merge an exact multi-view composition boundary

- Date: 2026-08-06
- Status: accepted; extends D0019, D0020, and D0021
- Context: A native categorical partition could produce multiple views, but
  subsequent stages remained conservatively closed because sorting views does
  not combine them. `filters.returns` has four observable view identities,
  non-obvious malformed-record behavior, and warning/file-numbering semantics;
  `filters.merge` owns persistent state and is the upstream operation that
  deliberately collapses those views. Treating either as a single-output mask
  would change valid PDAL pipelines.
- Decision: Implement a reusable stable return-partition primitive over the
  exact upstream first/intermediate/last/only predicates. The host path uses a
  two-pass count and stable scatter. The force/require-only CUDA path classifies
  to fixed three-bit keys and performs a stable CUB radix sort of group/source
  pairs, returning four integer counts. The in-process stage creates all four
  views before classification, publishes requested nonempty views in fixed
  identity order, preserves source order, and reproduces warning attribution.
  Implement no-option `filters.merge` as an exact host-native persistent view,
  including SRS choice and warning behavior, and use it as the sole audited
  planner boundary that reopens single-view native stages after arbitrary
  upstream partitions. Keep automatic CUDA selection disabled until runtime
  exactness, all sanitizers, and end-to-end break-even pass.
- Evidence: Unit properties pin every return predicate, malformed `only`
  records, subsets, zero/invalid programs, missing columns, fixed boundaries,
  and stable permutation order. A compiled 131,103-point CUDA/host property
  covers four masks. A 29-case complete-process matrix matches status, stdout,
  stderr, filenames, missing/extra outputs, and bytes in Debug, Release, and
  ASan/UBSan builds. Host Debug and Release account for 162 tests each (161
  pass plus the opt-in local-corpus skip); the sanitizer `pdg` lane accounts
  for 106 (105 pass plus that skip). CUDA Debug and Release compile cleanly;
  the CUDA Release `pdg` lane accounts for all 142 registered tests with
  device cases explicitly skipped because this namespace has no CUDA device
  node. The complete published PDAL suite passes 142/142: 140 local
  executables plus the STAC and remote COPC executables with fixture-network
  access. A read-only 21,970,934-point `returns -> merge -> head` case emits
  the same 5,865-byte LAS as the oracle, SHA-256
  `64432d20259eaabbe5c3c93e6be1243f07f9ec0160de90af38f17e2162b853a1`.
- Consequences: Multi-view output is no longer a permanent native dead end:
  an exact merge can feed later point programs and partitions without
  delegating the whole process. Return partitioning becomes the second
  reusable partition consumer after groupby. Neither stage is called
  GPU-qualified yet; the forced CUDA process, memcheck, initcheck, racecheck,
  synccheck, and same-machine performance matrix remain mandatory on the RTX
  4090 before promotion.

## D0023 — Preserve count and spatial partition identities across input views

- Date: 2026-08-06
- Status: accepted; extends D0020 and D0022
- Context: `filters.divider` and `filters.splitter` expose more than point
  membership. Count division creates requested empty views; round-robin order
  is stable within every view. Splitter origins and its cell map persist across
  incoming views, first encounter determines PointView ids and writer
  filenames, exact negative grid boundaries fall into the next lower cell, and
  a buffered point can be appended to four views in a fixed test order.
  Treating either filter as an unordered partition changes valid PDAL output.
- Decision: Implement count-based divider partition and round-robin modes as a
  stable count/permutation primitive with exact empty ranges. Use identity
  indices for sequential partition and stable CUB radix ordering of
  `(view-key, source-index)` for round robin. Retain capacity and expression
  forms in unchanged PDAL until their mutable multi-input state and split-event
  contracts receive a separate native implementation.
- Decision: Mirror splitter's persistent origin/map and buffered append loop on
  host. Add a force/require-only CUDA primary-cell kernel for finite,
  representable coordinates and nonpositive buffers, using explicit binary64
  operation rounding and the pinned truncate-then-decrement rule. Keep positive
  buffer assembly on the exact host backend until a device implementation
  proves the same multi-view creation order and an end-to-end win. Both stages
  are explicitly multi-view aware, but downstream single-view operations remain
  closed until an audited merge.
- Evidence: Unit properties cover uneven, round-robin, empty, invalid, exact
  negative-boundary, reversed-axis, nonfinite, missing-column, and unsafe-domain
  cases. Compiled CUDA properties compare 131,103 points for both divider modes
  and splitter cells. A 35-case complete-process matrix matches exit status,
  streams, filenames, absent/extra artifacts, and every byte in Debug, Release,
  and ASan/UBSan builds. Host Debug registers 169 tests (168 pass plus the
  opt-in local-corpus skip). CUDA Debug and Release compile 242 registered
  tests; the Release `pdg` lane accounts for 150 tests, and the two new device
  properties skip honestly in this namespace because it exposes no CUDA device
  node. A read-only 21,970,934-point round-robin divider, splitter,
  merge, and head composition produces the oracle's exact 5,865-byte LAS,
  SHA-256
  `e13d95da3dcc4a39309b5f38dbe4914adaa02af41208c820e56d586876e213c3`.
- Consequences: The fork now has native exact count and spatial partition
  stages that compose with groupby, returns, and merge without treating
  PointView sets as unordered. They increase implemented-device filter coverage
  from eleven to thirteen but do not increase the five automatically
  GPU-qualified filters. Forced CUDA process tests and memcheck, initcheck,
  racecheck, and synccheck are registered for the RTX 4090 runner; automatic
  selection still requires those runtime gates and a same-machine break-even
  matrix including PointView assembly and numbered output writes.

## D0024 — Preserve color-ramp bytes, execution mode, and per-view range state

- Date: 2026-08-06
- Status: accepted; extends D0019, D0021, and D0023
- Context: `filters.colorinterp` is not just a pointwise scalar-to-RGB map. Its
  observable behavior includes embedded PNG bytes or arbitrary GDAL raster
  bands, upper-median and ordered sample-deviation calculations, stream versus
  standard execution, an explicit `k` that is ignored only by the streaming
  callback, bounds that may persist or recompute across input views, and
  preservation of existing colors outside an unclamped interval. Even the
  upstream constructor's initial zero bounds affect the pre-prepare
  streamability probe. Replacing only the bin calculation would change valid
  pipelines and empty-output metadata.
- Decision: Implement an internal `BatchStreamable` PDAL stage that mirrors the
  complete option/default and range-state contract, uses the same embedded
  ramp data and GDAL loading route, and delegates every unaudited form before
  execution. Use a reusable exact host map for all supported inputs. Provide a
  force/require-only CUDA map for finite source values, finite increasing
  bounds, equal nonempty RGB bands, and representable ramp sizes, with explicit
  round-to-nearest binary64 subtraction, division, and multiplication before
  `floor`. Keep statistics and unproved numeric domains on the exact host path;
  keep automatic CUDA selection disabled until device correctness, all four
  Compute Sanitizer tools, and end-to-end break-even are established.
- Evidence: Host unit tests pin exact bin edges, clamp/invert, outside-color
  preservation, and rejection boundaries. A compiled CUDA property compares
  131,103 points across chunk and ramp edges. The 39-case complete-process
  matrix matches exit status, streams, files, and every byte in Debug, Release,
  and ASan/UBSan builds. Host Debug and Release account for all 176 registered
  tests (175 pass plus the opt-in local-corpus skip); the sanitizer `pdg` lane
  accounts for all 117 (116 pass plus that skip). CUDA Debug and Release compile
  all 253 registered tests; the Release `pdg` lane accounts for 156, with device
  cases explicitly skipped because this namespace has no CUDA device node. A
  read-only 21,970,934-point explicit-range streaming differential produces
  the oracle's exact 790,955,889-byte LAS, SHA-256
  `43fead56eade3ffa503311d9f47328d2fbd6888b913b5c7d92dfa12c8881dc23`.
- Consequences: Implemented-device filter coverage increases from thirteen to
  fourteen without inflating the five automatically GPU-qualified filters.
  The stage composes across point programs and multi-view partitions with its
  range state intact. Promotion requires the three forced CUDA process
  differentials, memcheck, initcheck, racecheck, synccheck, and a same-machine
  matrix that includes ramp loading, transfer, PointView mutation, and output
  writes. Fusion with adjacent pointwise work is expected to matter more than
  the standalone map kernel.

## D0025 — Keep exact randomization on the host without closing device regions

- Date: 2026-08-06
- Status: accepted; extends D0019, D0020, and D0021
- Context: Pinned `filters.randomize` constructs `std::mt19937` from an
  explicit seed, or obtains a fresh seed from `std::random_device`, then calls
  libstdc++ `std::shuffle` directly on the `PointView` index vector. The seeded
  permutation depends on the library's paired-small-range optimization and
  `uniform_int_distribution`; Fisher-Yates swaps are serially dependent. The
  stage moves eight-byte point ids rather than full records. Treating every
  unchanged filter as a possible view split nevertheless closed all later
  native regions, wasting accelerable work after an otherwise cheap exact host
  barrier.
- Decision: Retain the original PDAL stage for every randomization so seeded,
  unseeded, and diagnostic behavior cannot drift. Audit the no-`where`,
  type-correct option shape as a single-view ordering bridge: it preserves the
  existing proof of input order and the existing single/multi-view state, and
  it permits later native regions only when those properties were already
  established. Do not call this a CUDA implementation. A future resident
  scheduler may generate the exact mapping on the host, upload it once, and
  let downstream device kernels consume indirection without physically
  shuffling records.
- Evidence: A 14-case complete-process matrix matches exit status, streams,
  filenames, and every byte in Debug, Release, and ASan/UBSan builds. It covers
  seeds zero, one, and `UINT32_MAX`, regions before and after the barrier,
  stable ordering, empty/multi-batch inputs, per-view randomize plus merge, and
  invalid/option-rich delegation. Host Debug and Release account for all 178
  registered tests (177 pass plus the opt-in local-corpus skip); the sanitizer
  `pdg` lane accounts for all 118 (117 pass plus that skip). CUDA Debug and
  Release compile all 255 registered tests, and the Release `pdg` lane accounts
  for all 157 with unavailable-device cases skipped explicitly. A read-only
  21,970,934-point randomize plus native-assignment composition produces the
  oracle's exact 790,955,889-byte LAS, SHA-256
  `e66521fc478722f27b5d782410b8ec32b3e8774ffe426d4380b8a260eb98b347`.
  An exploratory three-sample alternating Release benchmark records 20.443 s
  versus 20.767 s (1.016x); its dirty-tree status and small margin make it
  composition evidence only, not an accepted performance claim.
- Consequences: Randomize no longer causes unrelated downstream GPU-capable
  work to delegate, but it does not increase the five GPU-qualified or fourteen
  pending-device filter counts. Closing the P1 randomize row requires a
  resident mapping consumer, GPU-runner exactness, and a same-machine result
  that beats this host bridge end to end. A one-thread CUDA shuffle or a
  round-trip of the index vector is explicitly insufficient.

## D0026 — Preserve ordered summary bits and reject a slower default stats path

- Date: 2026-08-06
- Status: accepted; extends D0014, D0019, and D0025
- Context: Pinned `filters.stats` updates online moments one point at a time and
  exposes their exact binary64 recurrence through metadata. A parallel tree
  reduction changes average, variance, skewness, or kurtosis bits. The stage
  also carries state across batches and views, emits enumeration/count/global
  structures and transformed bounding boxes, and is otherwise transparent to
  point order and view topology. A standalone wrapper must repay column
  gathering, pinned allocation, transfer, and metadata costs before becoming a
  default path.
- Decision: Implement the full exact host stage and an exact CUDA primitive
  that runs dimensions concurrently while applying each dimension's recurrence
  in strict point order with explicit round-to-nearest operations. Limit the
  current device envelope to finite basic summaries; retain advanced moments,
  enumerate, count, and global forms on the exact host backend. Treat eligible
  unchanged upstream stats as an audited order/view-preserving bridge. Enable
  native replacement only for internal differential or CUDA require gates
  until hardware correctness and end-to-end performance qualify it.
- Evidence: Direct units compare state and derived values bit-for-bit with the
  pinned `pdal::stats::Summary`. A compiled 131,103-point, three-dimension CUDA
  property crosses tile boundaries with persistent prefix state. The 22-case
  complete-process matrix matches output files, metadata JSON, status, stdout,
  and stderr in Debug, Release, and ASan/UBSan builds. A 21,970,934-point
  six-dimension run emits the oracle's exact 790,955,889-byte LAS, SHA-256
  `ff14463744dbe9ddd2f1d10271d278a7e41478e254d96a0780bda9d7aa1da2fe`.
  Its dirty-tree three-sample host diagnostic is negative: 6.363435 s versus
  5.785434 s, or 0.909x. After disabling default replacement, an option-free
  follow-up measures 5.848450 s versus 5.823799 s, or 0.996x. CUDA Debug and
  Release compile all 263 registered
  tests; the 225-test non-device Release lane is green with 189 passes and 36
  explicit unavailable-device/local-corpus skips. All 142 published PDAL test
  executables pass in one network-enabled run.
- Consequences: Pending device-filter coverage increases from fourteen to
  fifteen, while automatic GPU-qualified coverage remains five. Option-free
  stats-only pipelines retain the faster pinned implementation, and mixed
  pipelines can accelerate regions on either side without replacing stats.
  Promotion requires RTX 4090 process differentials, memcheck, initcheck,
  racecheck, synccheck, and a same-machine break-even matrix including gather,
  transfers, recurrence, metadata, and surrounding I/O. No CUDA speedup or
  completed-catalog claim is made from compilation or this host diagnostic.

## D0027 — Build metadata acceleration from exact reusable reductions

- Date: 2026-08-06
- Status: accepted; extends D0014, D0019, and D0026
- Context: `filters.info` and `filters.expressionstats` are transparent to
  point records but expose ordered global state through metadata. Info uses
  strict BOX3D growth, sentinel values, selected packed records, and a
  historically quirky nearest-query parser. Expressionstats exposes canonical
  parser output and `std::map<double>` equivalence, including the key bit
  pattern chosen for signed-zero ties. Copying every selected value back to the
  host would be compatible but would not constitute useful GPU acceleration.
- Decision: Implement the complete pinned host behavior first. For option-free
  info, fuse count and all six XYZ extrema into one device reduction whose
  merge carries first global indices. For expressionstats, evaluate the exact
  device predicate, stably select values and original indices, radix-sort on
  device, and reduce equal runs to compact count/first-index bins; restore each
  key from its earliest source value before the small result transfer. Limit
  the current histogram envelope to finite binary64 targets and exact-VM
  predicates. Retain info point/query forms, nonfinite target maps, and
  unsupported expression functions on the exact host implementation. Keep
  automatic replacement disabled pending hardware qualification.
- Evidence: Host units cover NaNs, infinities, sentinels, signed zero, first
  ties, encoded-coordinate decode, chunk merges, numeric map order, duplicate
  counts, empty selection, and device-envelope rejection. CUDA Debug and
  Release compile 131,103-point host/device properties for both primitives.
  Complete-process matrices match the pinned oracle in Debug, Release, and
  ASan/UBSan: 20 info cases and 19 expressionstats cases, including streaming,
  standard mode, multi-view composition, custom dimensions, historical parser
  behavior, nonfinite targets, valid failures, metadata JSON, status, stdout,
  stderr, and every output byte. Runtime CUDA tests skip explicitly because the
  current sandbox has no GPU device node. On the read-only 21,970,934-point
  local LAS corpus, both host pipelines emit the oracle's exact
  790,955,889-byte LAS (SHA-256
  `ff14463744dbe9ddd2f1d10271d278a7e41478e254d96a0780bda9d7aa1da2fe`).
  Info metadata is 25,937 bytes, SHA-256
  `5c7bd64c2c329f2ac770526bf931c55aba10b98cae7701459d02040ee61a56d4`;
  expressionstats metadata is 15,242 bytes, SHA-256
  `795209c4dc3e679dc83a56f28dee0ea46a12a4086bd16e090d8d59608741c84b`.
- Consequences: Pending device-filter coverage increases from fifteen to
  seventeen while automatic GPU-qualified coverage remains five. This is an
  implemented but unqualified slice, not completion of either full option
  surface or the PDAL catalog. Promotion requires RTX 4090 runtime
  differentials, memcheck, initcheck, racecheck, synccheck, a read-only large
  local-corpus run, and a same-machine break-even matrix. Info nearest selection
  and the persistent spatial index remain active implementation work.

## D0028 — Make radius outlier the first client of one persistent spatial index

- Date: 2026-08-06
- Status: accepted; extends D0018, D0019, and D0027
- Context: Neighborhood stages are a large fraction of the PDAL catalog. A
  private outlier kernel would repeat coordinate framing, Morton ordering,
  allocation, and query logic in every later filter and would not satisfy the
  catalog-wide product goal. PDAL radius search also has observable strict
  semantics: it includes the query point and duplicates, accepts only squared
  distances strictly below `r²`, and consumers may depend on exact counts at
  the boundary. Index construction and invalidation must be planned across
  stages rather than hidden inside each filter.
- Decision: Introduce a tile/planner-owned compact uniform-grid index shared by
  all radius consumers. Build 63-bit 3D Morton cell keys, stable point ids,
  unique cell keys, counts, and exclusive offsets; on CUDA use CUB stable radix
  sort, run-length encoding, and scan. Tie the cell edge to the largest planned
  radius so a query visits at most 27 cells, then perform the final strict
  predicate in binary64. Track availability through the DAG, share a built
  index across adjacent and sibling consumers, and invalidate it after XYZ
  mutation, point-set changes, or ordering changes. Account for the persistent
  worst case as 28 bytes per point. Make positive finite radius-mode
  `filters.outlier` the first consumer; preserve the entire statistical mode on
  pinned nanoflann until the common kNN/LBVH API is implemented. Keep CUDA
  selection force/require-only until the hardware gates pass.
- Evidence: Six host units cover 2D/3D, self/duplicate inclusion, exact-radius
  exclusion, cross-cell candidates, empty input, invalid geometry/radii, and
  rebuild accounting. Planner units prove one build across consecutive radius
  stages and a rebuild after XYZ transformation. A 131,103-point host/device
  property compiles in CUDA Debug and Release and is registered in memcheck,
  initcheck, racecheck, and synccheck. The 14-case complete-process outlier
  matrix matches output bytes, status, stdout, stderr, warnings, statistical
  behavior, and fallback behavior in Debug, Release, and ASan/UBSan. Full host
  Debug/Release, the PDG sanitizer lane, and CUDA-configured non-device tests
  remain green. Device execution skips because this sandbox exposes no NVIDIA
  device node, so no runtime or speed claim is made.
- Consequences: Pending device-filter coverage increases from seventeen to
  eighteen while automatic GPU-qualified coverage remains five. This closes
  neither P2 nor `filters.outlier`: the radius device path still needs an RTX
  4090 exact differential, all four Compute Sanitizer results, a density-
  stratified break-even matrix, and default-selection threshold; statistical
  mode needs the shared exact kNN/LBVH path. The next implementation slice is
  that kNN query and its reuse by normal/covariance/outlier consumers, not a
  second stage-private index.

## D0029 — Extend the shared cell table with conservative exact kNN

- Date: 2026-08-06
- Status: accepted; extends D0028
- Context: Statistical outlier, normal/covariance, LOF, classifiers, and many
  feature stages need k-nearest queries. A second private tree would defeat
  shared index residency. PDAL's nanoflann traversal makes equal-distance point
  ids observable for id-consuming filters, while statistical outlier observes
  only the ordered distance values. A finite grid walk also needs a proof that
  no unvisited cell can contain a closer point; silently stopping at a tuning
  limit is incompatible with the exact lane.
- Decision: Add a one-to-64-neighbor query to the same compact Morton cell
  table. Expand Chebyshev cell shells, retain k+1 candidates in exact binary64
  distance/id order, and stop only after visiting every point or proving the
  kth distance is strictly below a conservatively rounded lower bound to all
  unvisited cells. Cap the walk at 4,096 shells and publish an incomplete bit,
  never a guessed exact result. Publish a separate tie bit whenever equal
  distances could expose nanoflann traversal order. Fuse PDAL outlier's
  neighbor-1-through-k-1 `sqrt`/online-mean recurrence after the query; this
  distance-only consumer may accept ties but must reject incomplete searches.
  Keep the current statistical device envelope to `0 <= mean_k < 64` and at
  least `mean_k + 1` points. Preserve pinned nanoflann outside the envelope and
  keep automatic CUDA selection disabled pending hardware qualification.
- Evidence: Thirteen host units cover brute-force equality at k=1/8/32/64,
  exact id and squared-distance order, 2D semantics, duplicates and symmetric
  ties, fixed rows when k exceeds the point count, empty/invalid requests,
  explicit sparse-frame incompleteness, radius behavior, and index lifecycle.
  The ordered means match the reference recurrence bit-for-bit. CUDA Debug and
  Release compile a 4,099-point host/device property comparing every status,
  id, squared distance, and mean, alongside the existing 131,103-point radius
  property. Host Debug and Release each pass 208 of 209 registered tests with
  only the opt-in local-corpus case skipped. All 145 sanitizer unit tests have
  the same single skip, and the 14-case outlier process matrix passes under
  ASan/UBSan. Statistical and radius complete-process CUDA differentials and
  all four Compute Sanitizer integration lanes are registered. This sandbox
  has no NVIDIA device node, so none of those device-runtime gates or
  performance claims is recorded as passed.
- Consequences: Both `filters.outlier` methods now have shared-index CUDA
  implementations, but the filter remains in the pending-qualified count and
  option-free execution remains pinned. The uniform grid is expected to serve
  reasonably uniform ALS data; clustered and mixed-density inputs still need
  the planned LBVH backend behind the same API. Next consumers should reuse
  this query/status contract, starting with covariance/normal, rather than
  owning another index.

## D0030 — Reuse the exact kNN rows for covariance, Eigen, and normals

- Date: 2026-08-06
- Status: accepted; extends D0028 and D0029
- Context: Normal, eigenvalue, covariance-feature, optimal-neighborhood, and
  several classification stages share the expensive operation "find a
  neighborhood, form a 3-by-3 covariance matrix, and decompose it." Upstream
  PDAL exposes more than the geometric answer: its centroid follows neighbor
  order, demeaned coordinates narrow to float, covariance is the sample form,
  and Eigen's fixed-size self-adjoint QR solver determines the final binary64
  coefficients. A stage-private analytic CUDA eigensolver could be faster but
  would violate both exact output and the one-index/many-consumers architecture.
- Decision: Extend `SpatialIndex` with fused covariance and eigensystem outputs
  over its bounded exact kNN query. Preserve upstream's ordered online centroid,
  float demeaning, product order, and sample divisor. Compile the same vendored
  Eigen 3-by-3 self-adjoint solver for the device by adding only Eigen's
  existing `EIGEN_DEVICE_FUNC` annotation to its specialized
  tridiagonalization entry point; disable contraction in this CUDA source so a
  different FMA expression is not introduced. Publish zero-covariance and
  solver-failure status alongside tie and incomplete-search status. Make
  default kNN `filters.normal` with `2 <= knn < 64`, Boolean `always_up`, and
  no refinement the first consumer. Reject an entire device result before
  view mutation when selected point identities are ambiguous or the bounded
  grid cannot prove completeness. Leave radius, viewpoint, refinement,
  conditional execution, and other option envelopes on unchanged PDAL until
  their native algorithms are complete. Keep automatic selection disabled.
- Evidence: Five direct covariance/eigensystem tests compare every coefficient
  and status bit with the pinned PDAL/Eigen arithmetic. Planner tests cover the
  shared query request and conservative rewrite boundary. A 12-case complete-
  process normal matrix matches status, streams, diagnostics, artifact sets,
  and every output byte in Host Debug, Host Release, and ASan/UBSan. CUDA Debug
  and Release compile a 4,099-point property comparing kNN rows, means, all six
  covariance coefficients, eigenvalues, eigenvectors, and status. Both CUDA-
  configured non-device lanes pass 221 of 261 tests and explicitly skip 40
  unavailable-device/local-corpus cases. After rebuilding the modified Eigen
  consumers, all 142 published upstream PDAL test executables pass; the public
  STAC fixture required a retry after transient failures moved between S3
  objects. The managed workspace still masks the RTX 4090 device node, so the
  new kernels have not executed there.
- Consequences: Pending implemented-device filter coverage increases from
  eighteen to nineteen while automatic GPU-qualified coverage remains five.
  This is a reusable feature-family primitive and one incremental consumer,
  not completion of `filters.normal`, P2, or the catalog. Promotion requires
  device exactness, memcheck, initcheck, racecheck, synccheck, adversarial
  conditioning fixtures, deterministic repeats, and a same-machine end-to-end
  break-even matrix. The next consumers should be `filters.eigenvalues` and
  `filters.covariancefeatures`; adaptive LBVH selection, tiled ghost ownership,
  and cross-stage device residency remain parallel P2 architecture work.

## D0031 — Add eigenvalue and covariance-feature consumers without claiming residency

- Date: 2026-08-06
- Status: accepted; extends D0019, D0029, and D0030
- Context: The normal checkpoint proved the common exact kNN, ordered sample
  covariance, and fixed Eigen eigensystem, but one consumer did not establish
  a feature-family architecture. `filters.eigenvalues` observes ascending
  eigenvalues and normalization order. `filters.covariancefeatures` reverses
  and clamps those values, applies raw/square-root/normalized modes, expands
  ordered feature lists, and has density and per-point optimal-neighborhood
  dependencies. Calling either stage complete merely because its eigensolver
  ran on CUDA would conceal a remaining host `PointView` publication boundary.
- Decision: Reuse the D0030 eigensystem without a stage-private index or
  solver. Add an exact `filters.eigenvalues` client for `2 <= knn < 64` and
  unit stride. Add an exact `filters.covariancefeatures` client for the same
  bounded kNN family, one host thread, all three eigenvalue modes, and every
  non-density feature. Preserve feature-list expansion, descending
  nonnegative eigenvalues, formula operation order, zero-covariance messages,
  and solver errors. Keep radius, strided, conditional, density, optimal, and
  parallel forms on unchanged PDAL. Run kNN/covariance/eigen on CUDA only under
  the force/experimental gate, return eigensystems before publishing fields,
  and explicitly leave selected device SoA feature emission and cross-stage
  residency open rather than treating this wrapper boundary as final.
- Evidence: The 13-case eigenvalues and 18-case covariancefeatures process
  matrices match exit status, streams, diagnostics, artifact sets, and every
  output byte in Host Debug, Host Release, and ASan/UBSan. Planner and hybrid
  tests pin the shared `knn + 1` requests, ordered output dimensions, feature
  mask, and conservative fallback edges. Host Debug and Release each pass 222
  of 223 registered tests with the opt-in local corpus skipped. The sanitizer
  lane passes 155 of 156 with the same skip; the managed environment requires
  leak detection to be disabled because LeakSanitizer reports its documented
  ptrace incompatibility, while AddressSanitizer and UndefinedBehaviorSanitizer
  report no code error. CUDA Debug and Release compile both clients and
  register forced process differentials plus all four Compute Sanitizer lanes.
  Their 267-test non-device suites pass 227 tests and explicitly skip 40
  no-device/local-corpus cases. The sandbox still masks the RTX 4090 device
  node, so no device-runtime or performance result is claimed.
- Consequences: CUDA-backed pending filter coverage increases from nineteen to
  twenty-one while automatic GPU-qualified coverage remains five. These are
  additional clients of one reusable primitive, not completion of the feature
  family or P2. The immediate architecture work is selected-feature device SoA
  output, resident reuse across consecutive neighborhood stages, adaptive
  LBVH selection for clustered density, and tiled ghost ownership. Promotion
  still requires exact device differentials, deterministic repeats, memcheck,
  initcheck, racecheck, synccheck, adversarial conditioning, and same-machine
  break-even measurements.

## D0032 — Retain one neighborhood context across compatible PDAL stages

- Date: 2026-08-06
- Status: accepted; extends D0019, D0028, D0029, D0030, and D0031
- Context: The planner counted one spatial-index build across adjacent
  neighborhood consumers, but each PDAL compatibility wrapper still gathered
  XYZ, allocated a device batch, built the uniform grid, launched the same
  eigensystem work, and copied it back independently. That was a planning
  fiction rather than actual device residency, and a normal/eigenvalues/
  covariancefeatures pipeline paid three complete setup costs when `knn` was
  identical. A process-global cache would introduce lifetime, aliasing, and
  concurrent-pipeline hazards; silently retaining a cache across an XYZ or
  point-set mutation would be incorrect.
- Decision: Mark each maximal consecutive run of eligible `filters.normal`,
  `filters.eigenvalues`, and `filters.covariancefeatures` stages with a
  deterministic internal region id, the maximum `knn + 1` envelope, explicit
  reuse markers, and a final-stage marker. Attach one opaque resident product
  to the run's actual `PointView`: pinned XYZ, device XYZ, the shared
  `SpatialIndex`, and device/host eigensystem results keyed by requested k.
  Build the grid using the run-wide maximum request, reuse it for smaller
  requests, and reuse the eigensystem without another launch or copy when k is
  equal. Release the product after the final consumer and clear it whenever
  PDAL invalidates spatial products. Keep output publication on the exact host
  path for now. Add `PDG_REQUIRE_NEIGHBORHOOD_REUSE` as an internal test gate
  that makes any planned reuse miss fatal; do not expose these planner options
  as user-facing stage semantics.
- Evidence: A rewrite unit proves one region, maximum-neighbor propagation,
  two reuse markers, one final marker, and separate region ids across an
  intervening point program. The composed normal/eigenvalues/
  covariancefeatures process differential is byte-identical to upstream on
  the host path. The complete Host Debug registry passes 224 of 225 tests with
  only the opt-in local-corpus skip. The CUDA Debug build compiles the resident
  implementation and its 313-test registry; its 269-test non-device lane
  passes 228 with 41 explicit device/local-corpus skips. The rebuilt published
  PDAL procedure passes 142/142 sequentially, including its network STAC, EPT,
  and COPC tests. Host Release also passes 224 of 225 with the same opt-in
  skip; CUDA Release passes its 269-test non-device lane with the same 41
  explicit skips. ASan/UBSan passes 156 of 157 unit registrations plus all 57
  neighborhood matrix cases and the composed resident differential, with only
  the opt-in local-corpus unit skipped. The forced CUDA resident differential and all four Compute
  Sanitizer tools are registered, but the managed process still reports
  `cudaErrorNoDevice` and cannot reach `nvidia-smi`, so device execution and a
  speedup are not claimed.
- Consequences: Cross-stage XYZ/index/eigensystem ownership is now real for
  this compatible whole-view feature run rather than merely represented in the
  plan. This does not complete P2 or any of the three filters: feature and
  normal output columns still publish through host `PointView`, the context is
  not tiled, outlier and other neighborhood consumers are not yet composed
  into the same ownership region, and clustered/mixed-density data still need
  the adaptive LBVH backend. Automatic selection remains disabled until the
  RTX 4090 exact, sanitizer, determinism, and break-even gates pass.

## D0033 — Add an exact adaptive Morton-BVH behind the shared query API

- Date: 2026-08-06
- Status: accepted; extends D0019, D0028, D0029, D0030, D0031, and D0032
- Context: The compact uniform grid is efficient for ordinary airborne-lidar
  density, but bounded shell expansion must reject sparse frames after 4,096
  shells and a globally sized cell can contain a pathological fraction of a
  clustered TLS cloud. Returning such inputs to a private stage index would
  violate the one-index architecture, while publishing an unproven bounded
  result would violate exact compatibility. A full explicit binary topology
  would also add child/parent arrays and a host-to-device topology transfer.
- Decision: Add `MortonBvh` as a second backend of `SpatialIndex`, reusing the
  same stable 63-bit Morton point order. Use an implicit complete binary tree
  whose leaves are Morton positions; only internal bounds are stored. Each
  bound is a three-axis binary32 min/max in coordinates local to the index
  origin, rounded outward on construction. Bounds select and prune candidates
  only. Final radius predicates, kNN distances, point-id tie order, ordered
  means, covariance, and eigensystems retain their existing exact binary64
  operation order. The tree therefore completes sparse and mixed-density
  queries without a shell cap. Its internal-node allocation is less than two
  24-byte bounds per capacity point, so adaptive kNN plans conservatively
  budget 76 persistent bytes per point including the 28-byte Morton/cell
  table. Select it when an evenly spaced, deterministic probe of at most 8,192
  points finds a cell run larger than `max(32, 2*k)` or fewer than one occupied
  cell per eight probes; retain the grid otherwise. Internal force-grid and
  force-BVH switches exist only for differential and tuning gates. The initial
  topology is balanced and implicit, not the Karras radix topology named in
  the original design; Karras remains a G1 benchmark candidate if its tighter
  hierarchy repays added construction/topology cost.
- Evidence: Three new host contracts cover exact completion across a
  5,001-cell gap, strict radius traversal beyond the grid cell edge,
  two/three-dimensional mixed density at k=1/8/64, and deterministic grid/BVH
  selection. The covariance and eigensystem oracle tests now run through both
  backends and remain coefficient-for-coefficient identical. CUDA Debug and
  Release compile bottom-up hierarchy construction plus radius, kNN, ordered
  mean, covariance, and eigen traversal; the existing 131,103-point radius and
  4,099-point kNN properties now execute both backends when a device is
  available. Host Debug and Release pass 227 of 228 tests with only the opt-in
  local-corpus skip. Both CUDA-configured non-device suites pass 231 of 272
  with 41 explicit skips, and ASan/UBSan passes 159 of 160 with the same opt-in
  skip. The composed resident CUDA differential and the shared spatial
  Compute Sanitizer lane force the BVH. The managed process still reports
  `cudaErrorNoDevice`, so none of those physical-device executions or a G1
  performance result is claimed.
- Consequences: Clustered/sparse exactness no longer requires a different
  stage-private query implementation, and every current or future shared-index
  consumer can inherit the adaptive backend without changing its math. This is
  an implemented CUDA backend, not an automatically qualified product path:
  RTX 4090 exact differentials, determinism repeats, all four Compute
  Sanitizer tools, uniform-ALS/clustered-TLS/mixed-density profiles, and the G1
  memory/build/query break-even matrix remain mandatory. The 76-byte planning
  bound is deliberately conservative and materially above the long-term index
  target; profiling must decide whether Karras topology, compressed bounds, or
  a different selection threshold reduces it safely. Tiled ghost ownership
  and device-resident feature publication remain the next P2 architecture
  boundaries.

## D0034 — Qualify physical adaptive-index correctness and reject premature BVH selection

- Date: 2026-08-07
- Status: accepted; extends D0033
- Context: The first physical RTX 4090 G1 shakedown showed that D0033's evenly
  spaced occupancy sample could choose Morton BVH at 65,536 points even when
  the uniform grid was between 1.5 and 5 times faster. Periodically ordered
  sources could also alias the sample. Exactness alone was therefore not
  enough to make the selector safe, and treating this primitive result as a
  stage or product speedup would overstate the current implementation.
- Decision: Add a first-class, deterministic spatial benchmark that runs both
  backends in alternating order and refuses a result unless their complete
  output and status buffers are byte-identical. Replace evenly spaced probes
  with deterministic hashed point positions. Select BVH only when the sample
  indicates both broad clustering (fewer than one occupied cell per eight
  probes) and an estimated hot-cell population of at least 2,048 points;
  otherwise retain the grid. Keep both force switches internal. Do not enable
  automatic CUDA selection for the neighborhood stages until complete-process
  PDAL break-even, option-envelope, and real-corpus gates pass.
- Evidence: With `k=16`, two warmups, and seven alternating samples at
  1,048,576 points, the exact grid won uniform ALS by 20.573x and mixed density
  by 2.027x for ordered mean distance, while BVH won clustered TLS by 1.390x;
  covariance showed the same choices. On clustered TLS, grid won at 262,144
  points by 1.263x and BVH won at 524,288 by 1.508x. The corrected selector
  chose the measured winner in every row, and all backend result/status bytes
  matched. Host Debug passes 228 of 229 registrations with the one opt-in
  local-corpus skip. CUDA Release on the physical RTX 4090 passes 316 of 317
  with the same skip, including every current forced neighborhood process
  differential. The shared grid/BVH spatial suite reports zero errors under
  Compute Sanitizer memcheck, initcheck, racecheck, and synccheck. ASan/UBSan
  passes 160 of 161 unit registrations plus the four neighborhood process
  matrices and composed resident differential. The current published PDAL
  procedure passes 142/142 executables: 140 locally and both official network
  cases on a network-enabled rerun.
- Consequences: Physical correctness and sanitizer qualification are complete
  for the current exact shared-index envelope, and the adaptive rule now has a
  provisional same-machine crossover basis. The G1 report remains an
  unaccepted dirty-tree synthetic diagnostic with no end-to-end PDAL baseline;
  it is not a catalog, stage, or product speed claim. Selected device feature
  columns, tiled ghost ownership, complete option/client coverage, clean
  real-corpus G1 runs, and the remaining PDAL catalog are still open.

## D0035 — Project exact neighborhood columns on device and extend residency into point programs

- Date: 2026-08-06
- Status: accepted; extends D0032, D0033, and D0034
- Context: Retaining XYZ, the shared index, and eigensystems removed repeated
  neighborhood construction, but D0032 still returned a 96-byte eigensystem
  record per point to host and calculated every published feature there. A
  following GPU point program then gathered those host results into a new
  device batch. That boundary was correct but did not establish a useful
  resident pipeline. Moving formulas blindly was also unsafe: an observed CUDA
  `cbrt` result differed from the pinned host result by one ULP.
- Decision: Project normal/curvature, eigenvalue, and algebraic covariance
  feature columns directly into the existing resident device SoA batch using
  the pinned operation order and no contraction. Keep query/eigensystem status
  immutable and copy it into a stage-local status buffer before projection, so
  a feature error cannot contaminate a later consumer of the same cached
  eigensystem. Preserve `Omnivariance` and `Eigenentropy` with an explicit
  compatibility bridge: copy only the three eigenvalues (24 bytes/point), run
  the original host `std::cbrt`/`std::log` sequence, and upload the exact result
  columns. Continue publishing each stage's selected columns through the host
  PointView because unchanged PDAL consumers and failure semantics require
  that boundary. If the immediately adjacent fused region contains only exact
  assign/ferry operations over standard dimensions, execute it in the retained
  batch, gather only missing columns, copy only writes back, and then close the
  region. Unsupported/custom programs retain the existing exact fallback.
  Keep `PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE` internal as a differential gate.
- Evidence: Two physical CUDA properties compare every projected output bit,
  stage status, invalid-row sentinel, three eigenvalue modes, and launch-block
  boundary with host formulas. Eight forced neighborhood CUDA process
  differentials pass, including three all-feature/transcendental cases, the
  three-stage resident case, and a direct normal/eigen/covariance-feature to
  assign consumer with both reuse gates enabled. The direct consumer's LAS
  artifact, diagnostics, and status are byte-identical to upstream. Projection
  kernels and the complete consumer pipeline report zero memcheck, initcheck,
  and synccheck errors and zero racecheck hazards. Host Debug records 229
  executed passes plus one opt-in corpus skip in 230 registrations; physical
  CUDA Release records 323 passes plus the same skip in 324; ASan/UBSan with
  leak detection records 161 passes plus the skip in 162. The published PDAL
  procedure passes 142/142, including the network STAC/EPT/COPC coverage.
- Consequences: The current feature family now owns real output columns on
  device and proves one direct downstream consumer instead of merely caching
  an index. The transcendental bridge is deliberately visible and does not
  count as a native CUDA implementation. Stage outputs still cross to the host
  once, whole-view residency is not tiled, most neighborhood clients/options
  remain unimplemented, and no end-to-end speed claim follows from this
  correctness slice. Automatic selection remains disabled until clean
  real-corpus break-even and determinism gates pass.

## D0036 — Add exact nearest-neighbor distance to the resident spatial region

- Date: 2026-08-06
- Status: accepted; extends D0033, D0034, and D0035
- Context: `filters.nndistance` was still rebuilding a private host KD-tree and
  forcing a host boundary even when an adjacent neighborhood stage already
  owned exact XYZ and a compatible device index. Its `avg` mode is also not
  interchangeable with the shared ordered-mean primitive: upstream takes the
  square root of each of the first `k` neighbor distances, accumulates them in
  row order, and performs one final division. Replacing that sequence with an
  online mean or averaging squared distances would violate the bit gate.
- Decision: Extend the shared exact query with two stage-ordered distance
  projections: the square root of row `k` for `kth`, and the upstream serial
  square-root/add/divide sequence for `avg`. Compile the exact native envelope
  only for `1 <= k < 64`, a valid case-insensitive mode, enough input points,
  and no unsupported options. Accept tied neighbor identities because both
  projections are value-invariant across equal-distance ties, but reject any
  incomplete search status. Publish `NNDistance` through the existing
  PointView contract and retain its device SoA column for an immediately
  adjacent exact assign/ferry program. Preserve upstream execution everywhere
  outside the envelope, and keep automatic CUDA selection disabled until an
  end-to-end break-even gate is recorded.
- Evidence: Host and physical-device properties compare both modes bit for bit
  across the uniform grid and adaptive Morton BVH on 4,099 points. A 13-case
  process matrix covers defaults, case-insensitive modes, ties, duplicates,
  empty and insufficient views, boundary `k` values, invalid modes, string
  options, and `where` fallback. The composed
  `normal -> nndistance(kth) -> ferry -> nndistance(avg)` differential forces
  index and device-column reuse and matches upstream output, diagnostics, and
  status exactly. That complete composition reports zero errors under Compute
  Sanitizer memcheck, initcheck, racecheck, and synccheck. Host Debug passes
  235 tests plus one opt-in corpus skip in 236 registrations; physical CUDA
  Release passes 330 plus the same skip in 331; ASan/UBSan with leak detection
  passes 165 plus the skip in 166 unit registrations. The published PDAL
  procedure passes 142/142, including its upstream `filters.nndistance` test
  and network STAC/EPT/COPC coverage.
- Consequences: A distance-only stage can now join the same resident spatial
  region as outlier, normal, eigenvalue, and covariance-feature clients, and a
  direct downstream point program no longer requires a host gather/upload
  cycle. This is a correctness and composition qualification, not a speed
  claim: tiling, additional options and spatial clients, clean real-corpus
  break-even, and automatic selection remain open. Twenty-two implemented
  filters still await their required performance qualification, and the
  catalog-wide port remains incomplete.

## D0037 — Make radius-bounded spatial work capacity-safe and add exact radial density

- Date: 2026-08-06
- Status: accepted; extends D0033 through D0036
- Context: The shared radius implementation required the complete PointView to
  fit one device allocation. Fixed-size point chunks are not exact for a
  neighborhood stage because a query near a chunk seam can observe points from
  another chunk. `filters.radialdensity` was also still a private host KD-tree
  client even though it needs the same strict radius count as radius outlier,
  followed by one literal binary64 scaling operation.
- Decision: Partition radius-bounded work into deterministic half-open XY core
  tiles and conservative closed halos equal to the largest query radius. Give
  every source point exactly one core owner, retain source order in every tile,
  carry a device ghost mask, and scatter only owner rows into a complete
  source-order mosaic. Permit 2D tiling for 3D queries because it can add but
  cannot omit candidates; reject the unsafe reverse. Reject nonfinite frames,
  insufficient halos, incomplete mosaics, and tiles that exceed capacity after
  ghosts rather than publishing partial results. Choose whole-view or tiled
  execution from the lesser of 20 GiB, 75% of free VRAM, and 75% of total VRAM
  using a conservative 128-byte-per-point estimate; keep tile-edge and
  require-tiling switches internal. Add exact count-scaled radius output and
  implement `filters.radialdensity` with upstream's `3.14159` factor and
  operation order. Keep automatic CUDA selection disabled. The first executor
  is synchronous per tile; overlap and plan-wide tiled residency require a
  later qualification rather than being implied by this decision.
- Evidence: Three host tile contracts cover negative and exact-face ownership,
  one owner per point, source ordering, ghost transfer, owner-only scatter,
  three tile edges, empty input, invalid frames, nonfinite input, and capacity
  rejection. A physical 8,323-point CUDA property reproduces whole-view counts
  and scaled binary64 values across seams. The ten-case radialdensity process
  matrix, forced whole/tiled radialdensity differentials, and forced whole/tiled
  radius-outlier differentials are byte-identical to upstream. The tiled
  primitive and both complete forced-tiled pipelines report zero errors under
  Compute Sanitizer memcheck, initcheck, racecheck, and synccheck. Host Debug
  passes 242 executions plus one opt-in corpus skip in 243 registrations;
  physical CUDA Release passes 341 plus the same skip in 342; ASan/UBSan with
  leak detection passes 171 plus the skip in 172 unit registrations, and the
  ten-case matrix separately passes with the sanitizer runtime loaded into both
  processes. A fresh network-enabled run of PDAL's published testing
  procedure passes all 142 registered executables, including STAC, EPT, and
  COPC coverage.
- Consequences: Radius outlier and radialdensity now have an exact bounded-VRAM
  execution foundation instead of a whole-view-only demonstration. This does
  not establish an out-of-core kNN proof, overlapped tile performance, shared
  tiled residency across stages, or a stage speedup. Those gates and the
  remaining PDAL catalog stay open, and no benchmark entry or automatic
  selector change follows from this correctness checkpoint.

## D0038 — Reuse and overlap exact radius tiles with two event-governed lanes

- Date: 2026-08-07
- Status: accepted; extends D0037
- Context: D0037 allocated pinned columns, device columns, index storage, and
  output storage independently for every tile, then synchronized the complete
  stream before owner publication. That preserved exactness and bounded VRAM,
  but serialized host gathering, transfers, index/query work, and copy-back;
  allocator/startup cost dominated the seam-heavy property. Adding overlap
  without explicit ownership would risk recycling pinned memory while DMA was
  still in flight or publishing a partially completed owner region.
- Decision: Add a reusable gather contract whose caller-owned host batch keeps
  stable column and ghost-mask allocations across tiles. For pinned staging
  with multiple tiles, execute on two independent pinned/device allocation
  pools and nonblocking streams. Record a completion event after each device
  output copy, scatter only after that event, and drain a lane before reusing
  it. Keep pageable staging on one reusable lane. On exceptional submission,
  synchronize any stream that has accepted work before its pinned allocations
  can be destroyed. Divide the existing conservative tiled VRAM budget across
  both resident lanes; do not broaden the exact stage envelope or enable
  automatic CUDA selection.
- Evidence: The 8,323-point seam property reports two lanes and repeated lane
  reuse, reproduces every whole-view count and scaled binary64 value, covers a
  pageable one-lane execution, injects a later nonfinite tile while the other
  lane is in flight, and then proves exact recovery through the caller-owned
  pool. All four Compute Sanitizer tools report zero errors or hazards for that
  property and for the complete forced-tiled outlier and radialdensity
  pipelines. The two complete process differentials remain byte-identical to
  pinned PDAL. Host Debug passes 242 executions plus one opt-in skip in 243
  registrations; physical CUDA Release passes 341 plus that skip in 342;
  ASan/UBSan passes 171 executions plus the skip in 172 registrations when the
  generator test is included; and PDAL's published procedure passes 142/142,
  including its network STAC/EPT/COPC tests. In a same-binary engineering
  probe, twenty repetitions of the unchanged pinned two-query property fell from a
  91–93 ms steady range before reuse to 17–21 ms after warm-up. This is not a
  complete-process PDAL comparison and is not recorded as a product benchmark.
- Diagnostic real-corpus evidence: The complete forced CUDA
  `filters.radialdensity` pipeline produced byte-identical LAS output on a
  21,970,934-point ADKLR ALS tile for every warm-up and five alternating
  measured runs. Forced 16-tile execution had an 8.164x median speedup over
  pinned PDAL (8.855 s versus 72.292 s); whole-view execution, when it fit,
  took 6.386 s in a one-shot trial. A ten-sample 250,000-point prefix was
  exact at 2.104x (0.387 s versus 0.813 s). Independent one-shot exact trials
  covered a 260,438-point RMIT TLS cloud at 24.134x, a 5,114,283-point
  snow-road-twin RGB/NIR LAS at 7.335x, and a 36,772,046-point VEIL LAZ tile at
  20.228x. These reports are retained under `build/benchmarks`; because they
  were collected from a dirty development tree and three are single-sample
  corpus probes, they are diagnostic evidence only. They do not enter
  `BENCHMARKS.md`, qualify an automatic threshold, or establish a catalog-wide
  claim.
- Consequences: Exact radius tiling now has bounded double-buffered overlap and
  explicit lifetime/cancellation coverage rather than a sequential proof of
  concept. Radius density now also has representative exact ALS/TLS and LAS/LAZ
  diagnostic evidence, but clean-tree acceptance, density-stratified thresholds,
  plan-wide tiled residency, adaptive exact kNN tiling, option/client completion,
  and catalog-wide automatic selection remain open. No release benchmark or
  catalog-completion claim follows from this decision.

## D0039 — Physically qualify exact expression histograms; keep losing reductions on host

- Date: 2026-08-07
- Status: accepted; extends D0029
- Context: `filters.stats`, option-free `filters.info`, and
  `filters.expressionstats` already had exact host replacements and compiled
  CUDA primitives, but compilation did not establish useful device execution.
  Treating the family as one generic “GPU reduction” would hide materially
  different arithmetic and transfer costs: exact stats preserves a serial
  recurrence, info performs little work per transferred point, while
  expressionstats replaces predicate-driven ordered host maps with stable
  selection, radix sort, and reduce-by-key.
- Decision: Add a ten-case process lane that requires the exact
  expressionstats CUDA path across standard/streaming execution, overlapping
  and empty selections, custom intermediates, native regions on both sides,
  and post-partition multi-view accumulation. Keep finite binary64 targets and
  exact-VM predicates as the device envelope. Do not automatically select any
  member of the family from dirty-tree evidence. Retain standalone stats and
  info on their faster exact host/upstream paths; optimize them through
  resident-plan fusion rather than an isolated transfer/launch.
- Evidence: The physical 131,103-point histogram property and all ten forced
  process cases match pinned PDAL exactly. The property reports zero errors
  under memcheck, initcheck, and synccheck and zero racecheck hazards. The full
  process lane is clean under memcheck and racecheck and returns success under
  initcheck and synccheck. On a dirty-tree real ALS prefix, two warm-ups and
  ten alternating measured samples at 1,000,000 points are exact; medians are
  0.797752 s pinned PDAL, 0.779113 s PDG host, and 0.633355 s forced CUDA.
  One-shot exact expressionstats trials reach 2.346x on 21,970,934 ADKLR LAS
  points, 1.911x on 5,114,283 snow-road-twin points, and 2.375x on a
  36,772,046-point VEIL LAZ tile. At 250,000 points CUDA is only 0.702x. In
  contrast, forced CUDA stats and info at 21,970,934 points are 0.468x and
  0.859x pinned PDAL. The unaccepted diagnostic and raw report names are
  recorded in `BENCHMARKS.md`.
- Consequences: Expressionstats is physically qualified for explicit forced
  use and has a measured large-input crossover, but automatic selection still
  needs a clean-tree repeated benchmark and profile. Stats and info remain
  correct, fully usable host stages and explicit CUDA test targets, but are not
  mislabeled as accelerations. A shared resident column/reduction region is the
  next architecture step for all three, and the broader PDAL catalog remains
  open.

## D0040 — Qualify global radix primitives and compare complete artifact sets

- Date: 2026-08-07
- Status: accepted; extends D0018, D0020, D0021, D0022, and D0023
- Context: Sort, Morton order, categorical/return partitions, divider, and
  splitter shared exact CUDA key/permutation primitives, but prior evidence
  stopped at compilation or a single small process differential. The benchmark
  runner also assumed one output file, so it could not measure PDAL stages
  whose numbered writers expose view identity, count, and publication order.
  Timing a radix kernel alone would systematically overstate the value of GPU
  partitioning when PointView assembly and multiple writes dominate.
- Decision: Treat a numbered writer as one exact artifact set. Isolate its
  `#` target per invocation; compare every logical filename, byte size, and
  SHA-256; reject missing, extra, renumbered, or byte-different files; and
  record a canonical manifest hash. Physically qualify the shared ordering,
  Morton, return/divider, and splitter properties plus their forced process
  clients. Select standalone execution by complete-process performance against
  both pinned PDAL and PDG host. Keep sort, groupby, returns, divider, and
  splitter host-selected; preserve their exact device primitives for resident
  plans. Treat ordinary and reverse Morton as the sole positive standalone
  device candidate, but do not enable its automatic threshold until a clean
  repeated release checkpoint and profile are recorded.
- Evidence: Six physical properties and the forced stable-sort, ordinary/
  reverse Morton, groupby, returns, divider, and splitter processes match the
  oracle. Memcheck, initcheck, and synccheck report zero errors for the shared
  suite and racecheck reports zero hazards. The runner contract passes, and
  real artifact manifests cover four groupby, 17 divider, and 49 splitter
  outputs at 21,970,934 points. At that size, dirty-tree forced CUDA results
  versus PDAL are 1.808x sort, 1.936x Morton, 0.913x groupby, 0.982x returns,
  0.952x divider, and 0.932x splitter. Five-sample sort trials show PDG host at
  1.796x PDAL, making CUDA only about 0.7% faster; at 4,000,000 points host
  beats CUDA. Five-sample ordinary Morton trials at 4,000,000 points show
  1.481x PDAL for CUDA and 1.022x for PDG host. Independent exact one-shot
  Morton trials are 1.578x on 5,114,283 snow-road-twin points and 2.032x on a
  36,772,046-point VEIL LAZ tile; reverse Morton is 1.682x PDAL at 4,000,000
  points versus 0.997x for host.
- Consequences: Multi-view performance claims now include the complete
  publication surface rather than a conveniently chosen output or kernel.
  Morton has a physically proven, materially faster device envelope ready for
  clean-tree selector qualification. The other global primitives are not
  failures: they are exact resident building blocks, but an isolated GPU round
  trip is not called acceleration. Plan-owned resident permutations and
  device-side multi-view publication remain the next shared optimization.

## D0041 — Automatically select exact Morton CUDA above its clean crossover

- Date: 2026-08-07
- Status: accepted; extends D0018 and D0040
- Context: D0040 established exact ordinary/reverse Morton kernels and a
  positive dirty-tree signal, but correctly withheld a default selector. A
  force-only implementation would still leave ordinary users on the slower
  host path, while selecting CUDA at every size would regress small inputs and
  CUDA startup is material in a fresh process. The selector also must not turn
  degenerate/nonfinite coordinates, a missing device, or an explicit disable
  request into a failure or unnecessary staging allocation.
- Decision: For the already rewritten exact `filters.mortonorder` envelope,
  automatically request CUDA at 2,000,000 points when XY bounds are finite and
  nondegenerate. Before publication, retain the complete finite/in-bounds
  value scan and stable-permutation exactness checks. Apply the same threshold
  to ordinary and reverse mode. Honor `PDG_DISABLE_CUDA_HYBRID`; retain the
  force/require switch for differential tests; avoid allocating/gathering CUDA
  staging when execution is host-selected or no CUDA device is present; and
  fall back transactionally on recoverable device failure. Keep all smaller or
  unsupported forms on the upstream-equivalent host implementation.
- Evidence: Clean benchmark B0006 at commit `7b27cb1fb` uses normal option-free
  selection with two warmups and ten interleaved samples per executable.
  Ordinary mode is host-selected at 1,000,000 points (0.983x pinned PDAL), then
  reaches 1.248x at the 2,000,000-point CUDA threshold and 1.528x at
  4,000,000. Reverse mode reaches 1.487x at 2,000,000. Every complete LAS
  artifact, status, stdout, and stderr comparison passes. A clean option-free
  Nsight Systems trace proves that automatic selection launches the Morton key
  kernel and eight CUB radix passes; total kernel time is 0.429731 ms, memory
  operations are 2.746905 ms, and the included lazy CUDA stream/context start
  costs 87.156854 ms. The option-free threshold path reports zero memcheck
  errors, while the shared primitives remain clean under all four Compute
  Sanitizer tools. The selector boundary, degenerate/nonfinite program gate,
  ordinary/reverse process differentials, host-disable path, and complete
  stable permutation are covered independently. The post-selector aggregates
  pass 244 executions plus one opt-in skip in 245 Host Debug registrations,
  345 plus the skip in 346 physical CUDA Release registrations, and 244 plus
  the skip in the 245-test ASan/UBSan preset. PDAL's published upstream preset
  remains green at 142/142.
- Consequences: `filters.mortonorder` becomes the sixth automatically
  GPU-qualified filter in the configured catalog. This is an accepted product
  speedup rather than a force-only demonstration, but it does not promote
  standalone sort or multi-view partition stages whose complete-process GPU
  paths did not beat the fastest host choice. It also does not reduce the
  catalog-wide completion contract: 22 implemented CUDA filter envelopes still
  await qualification, and the remaining stage/I/O families still require
  native implementations or accepted GPU-inapplicability decisions.

## D0042 — Qualify exact map/selection kernels but reject losing standalone round trips

- Date: 2026-08-07
- Status: accepted; extends D0015, D0016, and D0024
- Context: Transformation, IQR, MAD, and color interpolation had exact host
  stages and compiled CUDA kernels, but describing them as pending hardware
  work understated their implementation and describing them as accelerated
  would have ignored transfer, allocation, PointView, GDAL, and writer costs.
  Their value must be decided by complete PDAL processes, including automatic
  color-range calculation and output publication, rather than isolated kernel
  time.
- Decision: Physically qualify the existing exact device envelopes and retain
  their force/require gates for differential and resident-plan work. Keep all
  four stages host-selected in standalone default pipelines because clean
  complete-process trials do not establish a material CUDA win. Preserve their
  device implementations as composable operations for a future resident region
  that amortizes allocations and avoids host round trips. Reopen automatic
  selection only after such a region has its own clean break-even matrix.
- Evidence: On the RTX 4090, the 131,103-point color-map, affine-transform,
  and robust-statistics properties match host bits/bytes, and each relevant
  unit plus forced complete-process lane reports zero errors or hazards under
  memcheck, initcheck, racecheck, and synccheck. Clean commit `bd5841356`
  records exact color interpolation at 0.345x pinned PDAL for 250,000 explicit-
  range points and 0.787x for 21,970,934; automatic range is 0.756x at
  4,000,000 and 0.873x at 21,970,934. Affine transformation is 0.357x at
  250,000, 0.851x at 4,000,000, and only 1.008x in a one-shot 21,970,934-point
  trial; its exact native host path is 0.883x there. IQR is 0.349x at 250,000,
  0.799x at 4,000,000, and 0.951x at 21,970,934. MAD is 0.801x at 4,000,000
  and 0.949x at 21,970,934. Every recorded output is byte-identical to its
  paired oracle.
- Consequences: These are implemented, physically validated CUDA building
  blocks, not missing kernels and not product speedups. Default users retain
  the faster exact host path. The immediate optimization target is persistent
  allocation and cross-stage device residency; no threshold is inferred from
  a one-shot near tie.

## D0043 — Automatically select expression statistics by measured work

- Date: 2026-08-07
- Status: accepted; extends D0039 and D0042
- Context: D0039 established exact physical expression histograms and a
  positive large-input signal, but its original one-million-point,
  three-expression diagnostic did not bound cheaper supported programs. A
  point-count-only threshold would select CUDA for a single cheap predicate
  while startup, pinned allocation, transfer, radix, and publication still
  cost more than the host implementation. Conversely, keeping the path
  force-only would withhold repeatable exact wins from ordinary pipelines.
- Decision: For a linear local LAS/LAZ reader, exact LAS writer, known
  cardinality, and cardinality-preserving stages before the histogram, select
  the exact CUDA expressionstats stage at 16,000,000 points for one expression,
  4,000,000 for two, and 2,000,000 for three or more. Read only the fixed LAS
  header to obtain the count, cap it by the reader `count` option, and do not
  infer cardinality across `start`, filters, partitions, or unproved stages.
  Unknown counts, zero expressions, unsupported predicates/targets, explicit
  disable, absent devices, or recoverable CUDA failure retain the host or
  unchanged-PDAL path. Avoid device-program compilation and point-id staging
  when CUDA is not requested.
- Evidence: Clean benchmark B0007 at commit `4c0ecd490` uses option-free
  selection and the cheapest supported predicates. Ten-sample medians are
  1.407x pinned PDAL at 2,000,000 points for three predicates, 1.374x at
  4,000,000 for two, and 1.263x at 16,000,000 for one. Matching controls below
  each boundary stay on host and are 0.978x at 1,000,000/three, 0.986x at
  2,000,000/two, and 0.997x at 8,000,000/one. All 114 process artifacts are
  byte-identical. An option-free trace records 3.688035 ms of kernels and
  3.166497 ms of device memory operations at the two-million-point boundary;
  64 pinned allocations consume 113.558469 ms, identifying allocation reuse
  and resident columns as the next target. The automatic path reports zero
  memcheck errors, while the direct/forced suite remains clean under all four
  Compute Sanitizer tools. Final aggregate gates pass 246 executions plus one
  opt-in skip in 247 Host Debug registrations, 347 plus the skip in 348 CUDA
  Release registrations, and 246 plus the skip under ASan/UBSan. The separate
  leak lane passes 175 plus the skip in 176 PDG-labelled registrations, and
  PDAL's published network-enabled procedure passes 142/142.
- Consequences: `filters.expressionstats` becomes the seventh automatically
  GPU-qualified filter in the configured catalog. Twenty-one implemented CUDA
  filter envelopes still await a positive automatic/resident gate, and the
  unimplemented catalog remains open. The result is a measured product path,
  not permission to call the broader PDAL port complete.

## D0044 — Recalibrate expression statistics after persistent workspace reuse

- Date: 2026-08-07
- Status: accepted; extends D0043
- Context: The B0007 automatic path was exact and beneficial, but its
  option-free trace attributed 113.558469 ms to 64 `cudaHostAlloc` calls and
  17.702245 ms to their frees. A fresh pinned and device `PointBatch` per
  histogram batch made that allocation cost part of every process and forced a
  conservative 16M/4M/2M one/two/three-expression selector. The histogram
  still needs its exact stable selection, radix sort, and ordered reduction,
  so only a fresh, cheapest-work, clean-tree matrix can justify lower gates.
- Decision: Retain pinned and device `PointBatch` workspaces for the lifetime
  of the prepared expressionstats stage, growing them only when a later batch
  requires greater capacity and clearing them before replacing their dimension
  registry. For the unchanged exact runtime envelope and known-cardinality
  linear LAS/LAZ pipelines, automatically select CUDA at 4,000,000 points for
  one expression, 2,000,000 for two, and 1,000,000 for three or more. Preserve
  all D0043 cardinality guards and host fallbacks: zero expressions, unknown or
  transformed cardinality, unsupported predicates/targets, explicit disable,
  absent devices, and recoverable device failures remain host or unchanged
  PDAL execution. D0043's old calibration is historical evidence, not current
  selector policy.
- Evidence: The clean implementation checkpoint at `9cb5da992` repeats the
  two-million-point, three-cheap-predicate case at 1.454731x pinned PDAL
  (1.222759 s versus 0.840540 s), byte-identically. Clean B0008 at
  `d70ae78b5` repeats the full option-free curve: 1.126x at 1,000,000/three
  (0.649885 s versus 0.577011 s), 1.212x at 2,000,000/two (1.009875 s versus
  0.833467 s), and 1.105x at 4,000,000/one (1.512560 s versus 1.368268 s).
  The three matching below-boundary controls remain host-selected at 0.957x
  for 500,000/three, 0.973x for 1,000,000/two, and 0.979x for
  2,000,000/one. All 114 final warmup and measured artifacts and the separate
  implementation checkpoint match their paired oracle byte-for-byte. The
  option-free one-million-point trace records 392 kernels in 1.337595 ms and
  0.563131 ms of GPU memory operations, with one 0.869249-ms pinned allocation
  and one 0.380869-ms free. The paired two-million-point trace records 784
  kernels in 2.681445 ms, 1.125203 ms of memory operations, one 0.810466-ms
  allocation, and one 0.381402-ms free. First-use stream creation still costs
  97.046651 ms at one million points, supporting a shared resident execution
  context as the next architectural target. B0008 records the report and
  profile hashes. After workspace reuse, the complete ten-case forced-CUDA process
  matrix again passes and Compute Sanitizer memcheck reports zero errors across
  it; a forced-stream 198,975-point/two-batch racecheck reports zero hazards.
  The option-free one-million-point automatic path also reports zero memcheck
  errors. The earlier all-four-tool direct primitive evidence remains valid for
  the unchanged histogram kernels and exact device envelope.
- Consequences: `filters.expressionstats` remains the seventh automatically
  GPU-qualified filter, now with a measured 4M/2M/1M work curve. This is only
  stage-local reuse: the current path still gathers host columns and crosses
  the H2D/D2H boundary, so device-resident cross-stage composition remains an
  open optimization and no broader catalog coverage is implied.

## D0045 — Exact resident multi-bridge neighborhood repair remains force-only

- Date: 2026-08-07
- Status: accepted; extends D0030 and D0035
- Context: Whole-view neighborhood stages formerly discarded residency at an
  adjacent assign/ferry boundary. Device kNN tie identity is observable through
  covariance/eigen output, so treating tied rows as device-exact would violate
  the oracle contract.
- Decision: A coordinate-preserving
  `normal -> bridge -> nndistance -> bridge -> eigenvalues` region retains
  resident XYZ, Morton-BVH, and eigensystems.
  For tied or incomplete eigensystem rows, one PDAL KD3 index recomputes the
  ordered upstream `computeEigenSystem`; repaired systems/status are uploaded
  before later device projections. Nndistance ties are value-invariant;
  incomplete nndistance remains conservative fallback. The envelope is
  force/experimental-only: no automatic selection is enabled.
- Evidence: Forced Autzen and RMIT outputs are byte-identical. Five-sample
  diagnostics of 2.42x and 3.44x are not a formal clean checkpoint. The final
  pre-portability CUDA Release aggregate records 368 passes plus one opt-in
  skip in 369 registrations. RMIT repair passes all four Compute Sanitizer
  tools; deterministic-ties and increasing-k fallback pass memcheck.
- Consequences: The repair is exact hybrid host KD3 work, not wholly
  GPU-native neighborhood execution. Full-system repair D2H/H2D transfers
  remain a performance risk; no 22M readiness, automatic gate, or promotion
  follows.

## D0046 — Keep untouched fallback out of the heavy engine and guard layout order

- Date: 2026-08-07
- Status: accepted; extends D0011 and D0045
- Context: The public executable linked PDAL, GDAL, and CUDA before it could
  decide that a command belonged entirely to unchanged PDAL. Separately, an
  exact resident bridge could register custom dimensions in a different order
  from the original graph. Most writers do not expose that order, but
  `writers.las` with `extra_dims=all` serializes it into Extra Bytes layout.
- Decision: Split the executable into a dependency-light `pdg` dispatcher and
  the existing `pdg-engine`. Route candidate stages, plausible direct LAS
  pipelines, internal controls, and every malformed or ambiguous request to
  the engine; execute only parsed noncandidate commands directly through the
  sibling pinned `pdal`. Before a rewritten LAS `extra_dims=all` graph runs,
  compare its ordered `(dimension name, type)` layout with the original and
  delegate if they differ.
- Evidence: Unit and process-boundary tests cover every current candidate,
  malformed and ambiguous pipelines, environment controls, sibling lookup,
  custom install prefixes, and exit/stream behavior. The installed launcher
  has no dynamic PDAL, GDAL, or CUDA dependency. The custom-ferry layout
  differential is byte-identical only through pre-execution delegation, while
  ordinary PLY and LAS writers remain eligible. Host Debug, ASan/UBSan, and
  the final pre-portability CUDA Release aggregate pass at 261+skip/262,
  261+skip/262, and 368+skip/369 registrations respectively. Clean B0009 uses
  50 alternating samples of a 5,327-point untouched LAS-to-text pipeline: the
  thin dispatcher takes 0.053683 s versus 0.068882 s through the former heavy
  engine, or 1.283x faster, with all artifacts exact. Direct pinned PDAL still
  takes 0.052251 s, so the thin wrapper retains a measured 2.7% toll on this
  deliberately tiny job.
- Consequences: Untouched fallback no longer pays heavy-engine startup, but no
  claim is made that a wrapper can beat invoking PDAL directly. The dispatcher
  remains deliberately conservative; adding a candidate stage or control
  requires updating its closed unit matrix. Ordered custom layout is an
  observable exactness boundary only for writers that serialize it.

## D0047 — Portability means compiler coverage plus physical per-SM exactness

- Date: 2026-08-07
- Status: accepted; extends D0002 and D0003
- Context: The RTX 4090 is the available performance and sanitizer machine,
  not the intended product floor. The previous SM 86/89/90 fatbin omitted
  supported NVIDIA devices, stream-ordered memory pools are not universal,
  and a successful cross-compile cannot prove bit identity for floating-point,
  libdevice, Eigen, or CCCL behavior on another architecture. CUDA 13 also
  removed pre-Turing offline compilation, so one current-toolkit artifact
  cannot carry the maintained legacy matrix.
- Decision: Release configuration uses `-arch=all`: real code for every
  architecture supported by the selected compiler and PTX for its highest
  major target. Publish a CUDA-12.x legacy-compatible artifact (currently SM
  50 through SM 90 compiler targets) and a CUDA-13+ current artifact (SM 75+
  and post-Hopper targets known to that compiler). Product CUDA translation
  units disable FMA contraction and FTZ and require precise division and square
  root. At runtime, query memory-pool support and use a synchronized classic
  allocator where stream-ordered pools are unavailable. Architecture
  generation is not runtime qualification: each advertised exact SM must pass
  the compact fixed-bit oracle lane on physical hardware, including PTX-JIT
  coverage where claimed.
- Evidence: CUDA 13.3.73 reports SM 75/80/86/87/88/89/90/100/103/110/120/121.
  A serialized `build/pdg-cuda-portable` compile with `PDG_BUILD_TESTS=OFF`,
  `PDG_CUDA_ARCHITECTURES=all`, and
  `PDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES=ON` builds `pdg_core` 51/51 at
  `--parallel 1`, including `NeighborhoodKernels.cu`. `cuobjdump` lists SASS
  cubins in that object for all twelve reported SM targets and PTX for SM 120.
  This proves compiler-supported real-architecture plus newest-target-PTX
  coverage, not physical exactness. Target-wide compilation retains
  `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`. The forced
  classic allocator allocation/copy regression and compact 22-test
  architecture bit lane pass on the physical RTX 4090, and `pdg doctor`
  reports its memory-pool capability.
  Focused Compute Sanitizer memcheck reports zero errors and zero leaked bytes
  for the classic allocator path.
  The resource-limited SM-89-only Release build passes 369 executions plus one
  opt-in corpus skip in 370 registrations in 97.41 seconds. SM 89 remains the
  only physically qualified target; no other runtime exactness claim is made.
- Consequences: The 4090 continues to decide performance thresholds, while
  compatibility artifacts are not Ada-only. Release CI must acquire or rent
  the physical architecture matrix before labeling those targets exact. A
  developer may explicitly disable the portable-architecture guard for a
  smaller iteration build, but that artifact is not releasable.

## D0048 — Make approximate-coplanar a typed resident eigensystem client

- Date: 2026-08-07
- Status: accepted; exact force/experimental runtime qualified on SM 89;
  automatic performance qualification pending; extends D0030 and D0045
- Context: Upstream `filters.approximatecoplanar` resembles the existing
  covariance-feature clients, but its observable contract differs in three
  ways. It passes `knn` directly to the self-inclusive KD3 query rather than
  adding one, writes the physical unsigned-byte `Coplanar` dimension, and uses
  two strict eigenvalue comparisons. Zero covariance logs the upstream message
  and leaves an existing output byte unchanged. Reusing a Planarity formula,
  a binary64 output column, or the neighboring stages' `knn + 1` convention
  would therefore violate exact compatibility.
- Decision: Provide the exact host wrapper and a force/experimental-only CUDA
  envelope for `3 <= knn <= 64`. Reuse the planner-owned spatial index and
  cached ordered eigensystem, repair ambiguous rows through the existing exact
  KD3 path, evaluate
  `lambda1 > thresh1 * lambda0 && thresh2 * lambda1 > lambda2` in the preserved
  order, and publish a typed unsigned-byte resident `Coplanar` column. Preserve
  the prior byte and exact informational diagnostic for zero covariance and
  preserve the upstream eigen-solver failure. Allow an adjacent point program
  to consume the resident byte. Keep automatic selection disabled; too-small
  views, neighbor counts outside 3..64, conditional or invalid forms, absent
  devices, and recoverable CUDA failures retain the exact host or unchanged
  PDAL path.
- Evidence: The implementation passes a 26-case host process matrix, eight
  focused CUDA process differentials, and two CUDA unit/property tests. The
  resident test requires spatial-index, eigensystem, generic-column, and
  Coplanar-column reuse; it and the forced tie-repair case each produce exact
  output for five consecutive runs. Host Debug passes 266 executions plus one
  expected opt-in corpus skip in 267 registrations. Exact-differential CMake
  injects `detect_leaks=0`; that ASan/UBSan preset passes the same 266 plus one
  skip while the separate candidate-focused leak lane remains. Three initial
  comparisons of process-specific LeakSanitizer diagnostics rerun clean under
  that prescribed mode. On the physical RTX 4090, the
  SM-89-only CUDA Release tree passes 384 executions plus the skip in 385
  registrations in 103.66 seconds. Both new CUDA unit paths report zero errors
  or hazards under memcheck, initcheck, racecheck, and synccheck. Four complete
  process cases are clean under all four tools (16 invocations); max-`knn` and
  tie repair subsequently pass racecheck and synccheck too, so all six complete
  cases are clean under every tool. The serialized loop uses the uniform grid
  for standalone execution, the Morton BVH otherwise, and the explicit
  tie-repair proof guard. PDAL's published procedure passes 142/142: 140 local
  executables and both remote-fixture tests on their network-enabled rerun.
- Consequences: The implemented-but-unqualified filter count increases from 21
  to 22 while the automatically qualified count remains seven. Automatic
  approximate-coplanar selection stays disabled until a clean same-machine
  break-even matrix establishes a useful gate. SM 89 is the only physically
  runtime-qualified architecture. The CUDA 13.3 all-architecture `pdg_core`
  build, including the neighborhood kernels, proves broader compile
  portability only; those targets still need per-SM hardware evidence. This
  completes one exact catalog slice, not the remaining neighborhood family or
  catalog-wide GPU acceleration.

## D0049 — Propose RTX 4090 automatic approximate-coplanar at 262,144 points

- Date: 2026-08-07
- Status: proposed; two clean forced LAS boundary reports are exact; the LAZ
  report is a dirty-tree diagnostic only; option-free qualification and
  profiling are pending; extends D0047 and D0048
- Context: D0048 deliberately left automatic selection disabled after the
  exact force/experimental implementation passed its differential and
  sanitizer gates. New same-machine forced measurements show a substantial
  complete-process margin at 262,144 points, but a forced path cannot prove
  that an ordinary `pdg pipeline` invocation enters CUDA or that every
  ineligible/default-runtime case retains the exact fallback.
- Decision (proposed): In a separately configured clean qualification artifact,
  for the current candidate device profile only, allow a
  direct three-stage `readers.las` → `filters.approximatecoplanar` →
  `writers.las` pipeline to replace the default filter or explicit `knn=8` at
  262,144 points and above. The writer must emit uncompressed `.las` with
  `extra_dims=all`, matching the measured end-to-end envelope. Numeric
  `thresh1` and `thresh2` remain eligible. Require both the
  pipeline document and LAS/LAZ input to be regular files, read only the fixed
  LAS header before pipeline execution, honor the reader `count` cap, exclude
  COPC and other input formats, and require an available CUDA runtime with
  CUDA not explicitly disabled. The candidate profile is the first visible
  device named `NVIDIA GeForce RTX 4090` with compute capability 8.9 and a
  CUDA 13.3 runtime. Production artifacts compile this provisional selector
  off until the complete cross-host acceptance protocol passes. Mark the
  replacement as a standalone, terminal,
  non-reusing neighborhood region; this threshold makes no plan-wide residency
  claim. Unknown or sub-threshold cardinality, unsupported options or graph
  shape, header/probe failure, disabled or unavailable CUDA, an unqualified
  device, and recoverable device execution failure must delegate unchanged or
  run the exact host wrapper without output side effects. The test-only
  `PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA=1` guard must fail unless
  both automatic rewrite selection and CUDA execution occur;
  `PDG_REQUIRE_APPROXIMATECOPLANAR_HOST_FALLBACK=1` separately proves host
  fallback.
- Evidence: At revision `e0486c78ea7899b316037ebf1162bcc4e52c925f`, each
  forced report uses two warmups followed by ten alternating pinned-PDAL and
  PDG samples with frozen time and matching output hashes. The clean
  262,144-point LAS report `/tmp/pdg-approx-bench-262144-e0486c78e.json`
  records 2.158750x, and the clean point-format-8 LAS report
  `/tmp/pdg-approx-bench-pf8-262144-e0486c78e.json` records 1.974880x. The LAZ
  report `/tmp/pdg-approx-bench-laz-262144-e0486c78e.json` records 2.200810x
  but has `repository.dirty=true`, so it is diagnostic and is not accepted
  performance evidence. All three used the forced path. Clean option-free
  exact runs, explicit automatic
  proof-guard results, below-boundary and regular-file/runtime fallback
  results, and an option-free retained profile are pending. No accepted
  benchmark entry or default-selection claim follows yet.
- Consequences: The automatically qualified filter count remains seven and
  the support matrix labels this selector proposed. CUDA 13.3 compilation
  provides forced-path binaries for its supported NVIDIA architectures, but
  the proposed automatic selector remains compiled off in production builds.
  A clean qualification build may exercise only the RTX 4090/SM-89/CUDA-13.3
  candidate profile.
  Additional physical profiles require the same bit-exact, sanitizer,
  option-free selection, and crossover evidence on Vast before they can enter
  the allowlist. Acceptance of this proposal will require the pending evidence
  above; it will not imply completion of the neighborhood family or the PDAL
  catalog.

## D0050 — Insert P1.5 and make resident execution the breadth gate

- Date: 2026-08-07
- Status: accepted; supersedes breadth-first P2 work until every P1.5 exit
  criterion passes; extends D0039, D0042, D0045, and D0049
- Context: Review of the complete-process benchmark matrix showed that exact
  stage-local CUDA kernels frequently lose to the host after allocation,
  packing, transfer, synchronization, `PointView` publication, and writer
  costs are included. D0042 records color interpolation at 0.345x–0.873x,
  affine transformation at 0.357x–1.008x, IQR at 0.349x–0.951x, and MAD at
  0.801x–0.949x across the measured sizes. The global/multi-view review also
  records groupby, returns/merge, divider, and splitter at 0.913x, 0.982x,
  0.952x, and 0.932x respectively on the 21,970,934-point ADKLR tile, while
  repeated sort trials found only a sub-margin device edge or a host win.
  These results validate kernels as resident building blocks but reject
  avoidable stage-local round trips as the product architecture. Conversely,
  the standing clean approximate-coplanar result of 2.158750x at 262,144
  points is positive evidence that the shared neighborhood work is useful and
  must be requalified in a resident end-to-end plan rather than discarded.
- Decision: Insert the verbatim P1.5 Resident Execution directive in
  `spec.md` §12 and freeze breadth-first stage ports. Deliver D1 planner-owned
  residency/liveness and spill accounting, D2 descriptor-declared fusion, D3
  a bounded benchmark-fixed multi-lane tiled scheduler, and D4 a measured
  linear placement model before P2 resumes. Every V1–V7 load and E1–E7 exit is
  mandatory; B1, B2, and B3 are reported separately, and only exact
  end-to-end B2 is the acceptance metric. B3 remains calibration input and is
  never again a stage verdict. New stages may land during P1.5 only when they
  exercise or extend the resident interface; LOF is specifically V3 and may
  proceed only through resident shared kNN. For P1.5 Tier-N tiled validation,
  the E3 tolerance and `--deterministic` rule supersede D0002's unconditional
  bitwise requirement; other validation tiers retain their declared
  compatibility contracts. From this decision onward, `spec.md` and
  `DECISIONS.md` are the sole normative sources; prior chat and subordinate
  planning documents cannot override them.
- Evidence: The trigger is the same-machine round-trip review recorded in
  `BENCHMARKS.md` under the expression-statistics and global-ordering/
  multi-view diagnostics and accepted in D0042. The interrupted LOF slice was
  halted before its CUDA build and parked, unmerged, on
  `wip/lof-pre-p1.5` at `4e6e03631`. No stage port is included in this
  governance decision.
- Consequences: P2 and catalog breadth are blocked until P1.5 closes with no
  partial credit. The immediate implementation sequence is D1, D2, D3, then
  D4, with validation added as each architecture surface becomes executable.
  Device columns and indices must survive compatible stages, die or spill at
  explicit planned boundaries, and contribute to the VRAM estimate. Placement
  and index rebuild predictions become observable `--stats` output. After
  closure, the full stage matrix is rerun under B1/B2/B3 and catalog ports
  resume strictly through the resident interface, watching register-pressure
  occupancy cliffs and dimension-footprint DRAM growth.

## D0051 — Make last use and completed spills own resident allocation lifetime

- Date: 2026-08-07
- Status: accepted; implements the planner/allocator contract of D1 and extends
  D0050; P1.5 remains open pending D2–D4 and V1–V7/E1–E7
- Context: The P1 planner's edge-local liveness and union-of-touched-dimensions
  estimate were not sufficient for resident execution. An in-place branch
  could free a source column after its first consumer while a later sibling
  still needed it, output storage could be returned before an asynchronous D2H
  copy completed, and retaining every touched column overstated the reusable
  working set. The prior four-byte XYZ estimate described packed LAS integers,
  while the shared spatial index, transformation, and neighborhood kernels
  consume logical double XYZ.
- Decision: Partition device stages into maximal resident regions separated by
  host stages. Device siblings consuming one host product share a region so an
  uploaded batch and its planner-owned spatial index may be reused. Compute
  column allocation lifetimes from region-wide first and last use; materialize
  at the first use, return dead temporary storage after the last device stage,
  and defer final output retirement to the completion of its last planned
  spill. Unknown host stages conservatively read and write every known live
  column, and both adjacent boundaries require a full-record pack for runtime
  layout fields that pipeline JSON cannot reveal. Track each spatial-index
  allocation through its last actual neighborhood request, ending reuse on a
  host boundary, XYZ mutation, cardinality change, split/merge, or reorder.
  The VRAM estimator uses the peak simultaneous resident columns plus persistent
  index bytes, with resident XYZ charged as doubles; packed records and index
  construction scratch remain explicit D4 placement terms. Runtime hooks apply
  the planner's column materialization, stage retirement, and post-spill
  retirement lists to `PointBatch` allocations.
- Evidence: Host Debug builds with warnings as errors. The focused planner and
  `PointBatch` lane passes 30/30 tests in Host Debug and in the prescribed
  leak-disabled ASan/UBSan configuration. A three-assignment chain touches 19
  bytes/point but has a measured planned peak of 16 bytes/point; its bounded
  allocator regression falls from 160 to 80 bytes after the middle stage and
  reaches zero only after the final spill retirement. The mixed fallback case
  reports four transitions, two spills, two fallback boundaries, and two
  resident regions. Shared-radius-index coverage predicts two builds around an
  XYZ mutation, releases the first allocation at its final pre-mutation
  consumer, and charges 53 bytes/point at the peak (24 bytes of XYZ, one output
  byte, and 28 index bytes). The same ten-point lifetime sequence passes with
  physical stream-ordered allocations on the RTX 4090 and is clean under
  Compute Sanitizer memcheck, initcheck, racecheck, and synccheck. No
  performance result is claimed.
- Consequences: D2–D4 may consume one planner-owned allocation contract instead
  of reconstructing residency in stage wrappers. Full-record fallback bytes,
  packed record buffers, cardinality-dependent capacity, index build scratch,
  and synchronization cost must be added by D4 rather than hidden in the D1
  column estimate. Branch execution remains conservatively disabled by the
  hybrid rewrite until aliasing and ownership differentials prove it; the
  planner nevertheless retains sibling-live storage safely.

## D0052 — Make fusion legality declarative and require a true fused launch

- Date: 2026-08-07
- Status: accepted; implements the first executable D2 vertical slice and the
  cardinality-preserving V1 assign/ferry/assignment-expression chain; D2 and
  the cardinality-changing V1 expression-filter case remain open
- Context: The pre-P1.5 LAS point-program path kept records on device, but it
  still launched translation, canonical AoS-to-SoA decode, assignment, return
  recount, and canonical repack as separate kernels. Calling that path fused
  obscured its launch and intermediate-traffic costs. Fusion selection also
  depended on stage-name checks in the CLI instead of a planner contract, so a
  new stage could be admitted without declaring conditional or deterministic
  behavior.
- Decision: Every compiled stage carries descriptor-owned `pure`,
  `cardinality_preserving`, `dims_read`, `dims_written`,
  `fusable_as_prologue`, `fusable_as_epilogue`, and `deterministic_safe`
  semantics plus explicit `where`/`where_merge` state. Kernel-host capability
  is a separate descriptor declaration; the planner may form a candidate only
  when both sides declare support, the graph chain is linear, dimensions are
  legal at the selected boundary, and deterministic mode is safe. Native LAS
  translation and packing are the first enabled anchors. Their bounded fused
  kernel translates the source record, evaluates up to eight ordered
  assign/ferry statements over at most eight touched dimensions and 96 VM
  instructions, repacks written canonical fields, and accumulates final bounds
  and ReturnNumber counts in one launch. Programs outside that resource
  envelope retain the exact resident multi-kernel path; unsupported semantics
  retain host fallback. A process-level require switch succeeds only when the
  complete middle chain is a planner-declared fusion candidate and the fused
  kernel envelope accepts it. Neighborhood descriptors do not advertise host
  capability until their feature kernels actually accept the point program.
- Evidence: Host Debug compiles with warnings as errors and all 31 planner
  tests pass. The fused CUDA unit matrix is byte-identical to the host oracle
  for point formats 0, 1, 2, 3, 6, 7, and 8 across non-power-of-two chunks.
  Its four-chunk/two-lane regression and the process-level
  `pdg_cuda_assign_ferry_fused` differential both pass on the RTX 4090; the
  latter requires descriptor-planned fusion rather than generic CUDA
  execution. Compute Sanitizer memcheck, initcheck, racecheck, and synccheck
  report zero findings. Static SM-89 resource inspection records 56 registers
  and 320 bytes of stack per fused thread versus 40 registers and no stack for
  translation alone. This is E7 input, not a performance acceptance claim;
  achieved occupancy and added DRAM traffic remain to be measured.
- Consequences: The cardinality-preserving V1 chain has a real
  producer/consumer fusion path rather than a name for several resident
  launches. The static resource delta makes the known register/stack-pressure
  risk explicit. D2 cannot close until compatible neighborhood feature kernels
  declare and execute their supported prologue or epilogue,
  `--deterministic` reaches the public planner surface, the expression-filter
  chain is handled without an avoidable boundary, and E7 records measured
  occupancy and traffic per chain. D3 may reuse the bounded fused LAS lane as
  its first tiled scheduling load without treating this checkpoint as P1.5
  completion.

## D0053 — Fix bounded scheduler widths from class-specific sweeps

- Date: 2026-08-07
- Status: accepted; implements P1.5 D3; P1.5 remains open pending the
  remaining D2 work, D4, V1–V7, and E1–E7
- Context: The pre-P1.5 fused LAS and radius-tile paths each embedded a
  two-lane assumption. That preserved bounded memory but supplied no common
  liveness/budget contract, no mandatory `N=2..6` evidence, and no way to
  distinguish an actually active wider sweep from a request silently clamped
  by tile count or memory. Adding arbitrary streams without measurement risks
  multiplying pinned/device working sets while output I/O, host gathering, or
  one saturated kernel remains the limiting resource. The kNN path has no
  exact tiled implementation yet and therefore must not be labeled as a
  multi-lane scheduler class.
- Decision: One deterministic scheduler owns the implemented tiled classes:
  standalone LAS translation, fused point programs, ordered/compacting point
  programs, and tiled radius neighborhoods. It computes tile count, configured
  and active lane widths, lane reuse, peak lane bytes, serial dependencies,
  and explicit planner-memory clamping. Zero requests a class-fixed default;
  internal test/benchmark overrides are restricted to integers 2 through 6.
  The scheduler never probes hardware, times a launch, or changes its choice
  at runtime. One tile or an ordinal dependency executes on one lane. Radius
  tiling divides the conservative working-set budget across the configured
  lanes, passes the full planner budget into scheduling, and fails a sweep if
  the requested active width was not achieved. Each active lane owns an
  independent stream and pinned/device allocation pool; reuse is event/drain
  governed. Do not introduce a kNN scheduler class until exact adaptive kNN
  tiling exists through the resident shared-index interface.
- Decision: Fix the default at two lanes for all four current classes. B0010's
  clean same-machine sweeps show that three or four lanes improve an isolated
  median by no more than 2.21%, always with overlapping p5–p95 intervals,
  while consuming 1.5–2x the lane working set. Four through six lanes regress
  in the large LAS classes. The two-lane choice therefore preserves the
  measured performance envelope with the smallest swept working set. A future
  default change requires another complete, clean class-specific `N=2..6`
  sweep; it is not an autotuning input.
- Evidence: The bounded fused LAS and radius unit tests execute every width
  from two through six and compare complete exact output to their host/whole-
  view oracles. A synthetic six-lane radius request constrained to a
  three-lane VRAM budget remains exact and reports three active lanes. Host
  scheduler/configuration tests reject zero tile sizes and lane overrides
  outside the swept range, cover overflow and serial execution, and verify
  reuse counts. The fused/radius pair passes Compute Sanitizer memcheck,
  initcheck, racecheck, and synccheck; racecheck reports zero hazards after
  109.364 seconds. A process differential proves six-lane tiled radius
  execution where the fixture contains enough tiles. B0010 records clean raw
  reports, exact output hashes, pinned-PDAL baselines, and class-separated
  end-to-end medians. The final radius sweep fails closed unless each requested
  width is active.
- Consequences: D4 may provide an explicit plan-level VRAM budget without
  changing stream policy, and V5 can force natural tiling by lowering that
  budget rather than allocating unbounded lanes. Forced tiling on the compact
  radius fixture is validation and scheduler evidence, not V5/E3: the required
  naturally over-budget corpus and untiled Tier-N comparison remain open.
  B0010's standalone translation rows are B3 calibration only. No B1 roofline,
  D2 feature-kernel fusion, D4 placement, kNN tiling, or P1.5 exit claim follows
  from this decision.

## D0054 — Calibrate placement from round trips and expose topology transfers

- Date: 2026-08-07
- Status: accepted diagnostic D4 checkpoint; E5 passes for the frozen
  pre-P1.5 matrix; D4 and P1.5 remain open until calibrated placement owns
  executable resident regions and its decisions are reported at runtime
- Context: D0050 requires a bounded linear placement model rather than more
  stage-local thresholds. The first evaluator accounted for resident columns,
  record packing, startup, synchronization, index construction, cardinality,
  and memory, but its feature dump found a missing physical term: selection,
  ordering, and partition stages can spill no dimension column while still
  returning a keep mask, permutation, or cell-membership stream. Hiding those
  bytes inside a fitted stage coefficient would produce incorrect calls when a
  later resident consumer removes the spill.
- Decision: Stage descriptors declare device-to-host result bytes per input
  point and fixed result bytes. Robust filters declare their byte keep mask;
  sort, Morton, groupby, returns, and divider declare their 64-bit mapping;
  splitter declares its two 32-bit cells; stats and info declare their fixed
  summary result. Charge these bytes only at a planned spill and include their
  capacity in the peak VRAM estimate. Keep the placement calculation a pure,
  fail-closed linear evaluator: incomplete calibration, unknown record layout,
  an exceeded calibration envelope, or insufficient VRAM selects host.
- Decision: Freeze the initial SM-89 coefficients and stage residuals in
  `test/data/pdg/placement-calibration-sm89.json`. CUDA startup is the
  device-minus-host intercept of B0005's six-row round-trip curve; directional
  transfer and packing rates come from the retained B0005 trace; synchronized
  index-build cost comes from the D0034 grid/BVH reports; synchronization uses
  the B0005 CUDA API median. The audit uses one fixed linear residual per
  measured program class and performs no runtime fitting or autotuning. Its
  initial fit missed the measured 37,566-point approximate-coplanar host
  winner, so the demonstrated wrong call adds a fail-closed 131,072-point
  lower calibration envelope without changing the fitted coefficients.
- Evidence: Clean checkpoint `88adde94f` builds the warnings-as-errors Host
  Debug unit target and the CUDA Release audit target. All 36 focused planner
  and placement tests pass. The manifest names 52 complete-process cases over
  19 program classes and 58 unique raw reports; the provenance verifier
  matches every report SHA-256. The direct C++ model audit predicts all 52
  measured host/device winners, including the small-host controls, for 100%
  accuracy against E5's 90% threshold. The registered
  `pdg_placement_model_matrix_audit` test passes. B0011 records the coefficient
  derivation and case classification; these B3 inputs do not become stage
  acceptance evidence.
- Consequences: E5 is satisfied for the matrix that existed when D0050 was
  accepted, but this does not close D4. The calibration remains diagnostic and
  does not mutate the production selector. The next D4 slice must attach these
  costs to executable plan regions, choose host/device placement across mixed
  boundaries, and report predicted terms and actual crossings in `--stats`.
  V4 and the process-level V7 negative control remain mandatory. New measured
  rows append to the manifest/evidence record; coefficients change only after
  a consequential wrong call is reproduced and recorded.

## D0055 — Amortize one device toll across calibrated resident regions

- Date: 2026-08-07
- Status: accepted D4 model-core checkpoint; D4 and P1.5 remain open until the
  resident executor consumes the decision, reports predicted/actual crossings,
  and completes V4 and V7
- Context: D0054 evaluated one all-device plan and kept the SM-89 calibration
  only in the audit manifest. A mixed pipeline needs to decide independently
  which maximal resident regions remain on device while charging CUDA startup
  once for the process plan. Returning the old all-device transfer totals while
  judging a partially selected plan would make diagnostics misleading. Applying
  the SM-89 fit to a different GPU, driver, or toolkit without a physical
  calibration would also violate the fail-closed placement contract.
- Decision: Evaluate each maximal resident region without cold-start or
  plan-wide synchronization cost. Retain every region whose calibrated device
  cost is strictly below its exact host alternative only when their combined
  warm benefit repays one shared CUDA-startup term and the declared plan-wide
  synchronization toll. Unprofitable or memory-infeasible regions remain host;
  selected transfer, packing, index-build, topology-result, synchronization,
  and peak-VRAM terms aggregate only those regions. Report these as
  placement-variable costs because unchanged readers, writers, and host stages
  are common to both choices and cancel from this bounded decision. This is one
  fixed linear rule, with no search, optimizer, timing probe, or autotuner.
- Decision: The current executable rewrite supports only a linear graph with
  exactly one reader, at most one writer, one input per non-reader, and no
  shared producer. Plan placement therefore fails closed for branches even
  though D1 safely models their storage lifetimes. Embed the frozen D0054
  profile in the core and require an exact device name, compute capability,
  driver, and CUDA-toolkit key. Apply one measured residual model exactly once
  at the first device stage of each resident region; subsequent stages in that
  region receive zero incremental residual while planner-owned physical terms
  remain explicit. A missing/duplicate region, unknown model, profile mismatch,
  or manifest/core drift selects host.
- Evidence: Fourteen focused placement/profile tests pass in Host Debug and
  the prescribed leak-disabled ASan/UBSan lane, including two
  profitable regions separated by an unsupported host stage, one selected and
  one rejected region, warm-context synchronization rejection, selected-only
  full-record boundary accounting, an independently memory-infeasible region,
  exact profile matching, unknown-model rejection, and branch rejection. The
  broader planner/placement lane passes 45/45. The audit now consumes the core
  profile and refuses any coefficient or 19-model mismatch with
  `placement-calibration-sm89.json`; it retains 52/52 measured-winner accuracy.
  The independent provenance gate still verifies all 58 unique raw reports.
- Consequences: Calibration is no longer an audit-only duplicate, and D4 has a
  deterministic mixed-region decision object suitable for executor and
  `--stats` integration. Production selection remains unchanged: no stage or
  region is promoted by this decision. The next D4 slice must build the request
  from actual runtime cardinalities/layout/VRAM, make the executor honor the
  selected regions, report predicted terms and observed host/device crossings,
  and run the complete-process V4 and V7 gates. Additional NVIDIA devices gain
  placement profiles only after their physical exactness and calibration lanes
  pass; compiler architecture coverage alone does not qualify a profile.

## D0056 — Keep runtime placement execution diagnostic until its executor matches calibration

- Date: 2026-08-07
- Status: accepted integration/observability checkpoint; production defaults
  are unchanged; D4 and P1.5 remain open
- Context: D0055 produced a calibrated plan decision but no runtime adapter or
  process evidence. The bounded adapter can now obtain exact LAS cardinality,
  record-layout, free-VRAM, profile, and program facts, materialize selected
  point-program regions, execute them, and report planner predictions beside
  execution events. Review found that its generic PDAL point-program wrapper
  is not the executor measured by B0005: B0005 maps raw LAS records and fuses
  translation, program execution, canonical repack, two event-governed lanes,
  and positioned output, while the wrapper gathers and scatters a host
  `PointView`, transfers bound columns, owns one allocation/stream pair, and
  synchronizes each tile. Applying the B0005 winner prediction to that wrapper
  would therefore overstate D4 progress.
- Decision: Add the explicit PDG-only command
  `pdg resident PIPELINE [--stats FILE]` as default-off integration plumbing.
  Ordinary `pdg pipeline` selection is unchanged. Runtime placement is
  available only for an exact calibration-profile match, a linear single-
  reader/single-writer default uncompressed-LAS envelope with no configured
  reader or writer options, exact unchanged cardinality, known layouts and
  VRAM, and a measured point-program envelope. Unknown profiles, branches,
  predicates, configured layouts, blank or type-widening ferries, unmeasured
  programs, and unsupported selected regions fail closed to the untouched host
  pipeline before side effects. Once the planner selects the diagnostic CUDA
  wrapper, failure to execute CUDA is an error rather than an unreported host
  substitution.
- Decision: The stats schema distinguishes planned residency boundaries from
  `observed_crossings`. The latter are explicitly marked
  `inferred_from_region_transfers`; they aggregate adjacent wrapper region
  transfer events and are not claimed to be planner-owned full-record D1
  spill/upload operations. Selected wrapper reports identify
  `executor=pdal_point_program_wrapper` and
  `selected_device_calibration_matches_executor=false`. Stats require a real
  file, reject lexical, normalized, and symlink-equivalent aliases of the
  pipeline definition or reader/writer files, and reject a symbolic-link stats
  target so diagnostics cannot corrupt point data or stdout.
- Evidence: The post-review focused planner/runtime/resident/process lane
  passes 48/48 in Host Debug, 48/48 in CUDA Release, and 48/48 in the
  prescribed leak-disabled ASan/UBSan build. The published PDAL suite passes
  142/142; its two remote STAC/COPC registrations were rerun with official
  network fixtures after the restricted-network run passed the other 140. On
  the exact
  RTX-4090/SM-89 profile, V7's small complete process selects host, emits no
  CUDA events, and matches the pinned oracle at SHA-256
  `e98ac535e119ea14cfe5bd8262af1c1616dcc28af7bda74f7ea97919ae1e41bb`.
  The bounded mixed probe uses the 21,970,934-point, 790,953,999-byte local LAS
  input SHA-256
  `2ea7a921a6f45e0058ee2e491136e80e6450d7920d080ce94f4924d0a1ecd8f9`.
  It selects two point-program wrappers around the untouched
  `filters.randomize`, reports the planned fallback spill/upload topology and
  two inferred crossings, and exactly matches the oracle output SHA-256
  `d1788a838a2518147534fc4d89419b555f0e719e9eebdf73ce66496ab7431870`.
  Its 72.80-second CTest duration is validation timing, not B1/B2 benchmark
  evidence. The compact selected-wrapper CUDA path remains clean under
  Compute Sanitizer memcheck, initcheck, racecheck, and synccheck.
- Consequences: V7 is established for the exact reference profile. The mixed
  probe establishes exact forced-wrapper execution and honest topology/event
  observability, but it does not complete V4/E2, demonstrate D1 storage across
  the host stage, validate B0005 placement predictions for the wrapper, or add
  B1/B2 evidence. E5 remains the frozen 52/52 model-core audit only; it is not
  extended to `pdg resident`. No `BENCHMARKS.md` entry or stage-coverage count
  follows. The next slice must route a planner-selected terminal LAS region
  through the actual B0005 direct fused executor, pass it the D1 VRAM budget
  and D3 fixed two-lane schedule, expose that schedule in stats, and prove an
  exact complete-process differential. Resumable device batches and real
  materialized host-boundary spill/upload remain subsequent D1/D4 work.

## D0057 — Route one calibrated terminal region through the direct resident executor

- Date: 2026-08-07
- Status: accepted terminal-region integration checkpoint; production defaults
  are unchanged; D1, D4, V4/E2, and P1.5 remain open
- Context: D0056 could execute selected point-program regions only through the
  generic one-stream PDAL wrapper, although the D4 residual used for that
  selection was measured on B0005's raw-LAS, fused translation/program/repack,
  two-lane positioned-writer executor. That mismatch prevented an executable
  placement decision from claiming the calibrated implementation. The next
  bounded slice needed to connect those two paths without adding a stage or
  changing ordinary `pdg pipeline` selection.
- Decision: When `pdg resident` has an exact SM/profile match and the planner
  selects exactly one terminal, linear, whole-program-fusable
  `readers.las -> assign/ferry region -> writers.las` plan, execute it through
  the existing B0005 direct fused CUDA sink. Pass the runtime request's D1 VRAM
  budget into the shared D3 scheduler. Return and report tile count,
  configured/active lanes, lane reuse, peak lane bytes, memory clamping, and
  serial-dependency state. The fixed class default remains two lanes; a budget
  that holds one but not two lanes clamps to one, and a smaller budget fails
  before invoking the sink.
- Decision: Count successful direct CUDA copy bytes at their transfer sites,
  report one observed begin/H2D/D2H/end event sequence for the selected region,
  and identify the execution as `executor=direct_fused_las` with
  `selected_device_calibration_matches_executor=true`. Eligibility declines
  before the exact direct envelope is established retain D0056's explicit
  diagnostic wrapper/host behavior and its honest mismatch flag. After the
  envelope is established, invalid scheduler configuration, CUDA failure,
  insufficient VRAM, output collision, writer failure, or publication failure
  is an error rather than a silent wrapper/host substitution.
- Decision: Preserve the direct writer's exclusive temporary creation,
  positioned writes, close-before-publication, atomic hard-link publication,
  and cleanup on failure. A pre-existing final path is not replaced by the
  explicit `resident` command. This no-overwrite diagnostic contract is scoped
  to `pdg resident`; the ordinary compatibility dispatcher remains unchanged.
- Evidence: The focused planner/runtime/resident/placement lane passes 65/65 in
  Host Debug, 65/65 in CUDA Release, and 65/65 in the prescribed leak-disabled
  ASan/UBSan build. All eight CUDA LAS point-program tests pass on the physical
  RTX 4090, including exact two-lane execution, a one-lane planner-budget clamp,
  pre-sink rejection below one lane, and exact transfer-metric accounting. The
  compact new test is clean under Compute Sanitizer memcheck, initcheck,
  racecheck, and synccheck. The complete-process direct gate uses the
  21,970,934-point, 790,953,999-byte LAS source with SHA-256
  `2ea7a921a6f45e0058ee2e491136e80e6450d7920d080ce94f4924d0a1ecd8f9`;
  it selects one calibrated region, reports two active lanes and nonzero exact
  transfer totals, preserves a seeded output artifact, and matches pinned PDAL
  output bytes, stdout, stderr, and exit status. Its 10.48-second CTest duration
  is validation timing, not B1/B2 evidence. The parent D0056 checkpoint already
  passed the published PDAL suite 142/142, and this slice changes no upstream
  PDAL stage or command.
- Consequences: One cardinality-preserving V1 terminal chain now exercises the
  actual calibrated resident executor and consumes the planner-owned D1/D3
  contracts. This closes the executor-mismatch gap only for that terminal
  envelope. It does not provide a resumable device batch, materialize a real
  host-boundary spill/upload, complete V4/E2, validate natural over-budget V5
  tiling, add B1/B2 measurements, or close D1, D4, or P1.5. The next slice is
  the resumable planner-owned device batch and explicit unsupported-stage
  boundary; no catalog stage port may bypass it.

## D0058 — Execute planned liveness across a preflighted host boundary

- Date: 2026-08-07
- Status: accepted resumable-boundary checkpoint; production defaults are
  unchanged; the V4 validation load passes, while D1, D4, E2, and P1.5 remain
  open
- Context: D0057 could execute one terminal point-program region but could not
  suspend a resident plan at an unsupported host stage. The diagnostic wrapper
  also materialized the union of every region column before its first kernel
  while its scheduler charged the planner's smaller last-use peak. The D0051
  three-assignment pattern exposed the defect directly: two four-point lanes
  claimed 280 bytes while eager materialization required at least 304 bytes.
  Selection and CUDA allocation also occurred inside `PipelineManager::execute`,
  too late to retry the untouched host pipeline safely if the exact envelope,
  physical layout, or allocation failed.
- Decision: Materialize every selected upload and spill as a private, stable-ID
  pipeline marker bound to one planner-scoped resident context. Each lane owns
  an independent stream, pinned packed tile, device packed tile, `PointBatch`,
  and completion event. For each tile, apply the planned stages in declaration
  order: unpack only `deviceMaterialize` columns at first use, launch the exact
  operation, apply `deviceRelease` on that lane's stream after last use, repack
  only spill-live written columns, and retire final spill allocations only
  after the D2H completion event has been observed. The runtime records its
  aggregate observed allocation high-water mark and requires it not to exceed
  the planner/D3 schedule.
- Decision: Treat resident execution as a two-phase transaction. Prepare the
  rewritten graph, validate its descriptor envelope and physical layout with a
  one-point host probe, compute the bounded schedule, allocate every packed
  lane, and probe its planned column high-water sequence before executing a
  writer. A preflight rejection destroys that
  scope and runs the original pipeline JSON unchanged; a failure after accepted
  preflight remains an execution error. The direct fused-LAS executor retains
  D0057's independent fail-closed publication contract.
- Decision: Make boundary accounting first-class D4 output. Every selected
  placement boundary reports point count, logical-column bytes, complete
  runtime-record bytes, predicted transfer bytes, and predicted packing bytes;
  execution reports matched transfer bytes, bytes actually processed by its
  explicit pack/repack operations, and whether prediction and observation
  agree. Packing observations are never inferred from PCIe volume. The current
  generic boundary executor copies complete
  physical records, while the frozen calibration includes raw reader/writer
  and logical-column terms, so it continues to report
  `selected_device_calibration_matches_executor=false` and
  `boundary_accounting_matches_prediction=false`. Do not call E2 or D4 closed
  until those terms agree for the selected executor.
- Evidence: The deterministic liveness regression enforces the planner's
  140-byte per-lane and 280-byte aggregate ceilings; eager union materialization
  would require 152 bytes per lane and at least 304 bytes across both lanes.
  Its exact result passes on the physical RTX 4090. The compact mixed-boundary
  test
  preflights two lanes, forces reuse with 11 points in four-point tiles,
  preserves a seeded host permutation and an untouched signed custom column,
  and reports exact full-record transfers. A separate bad-layout case is
  rejected before boundary execution. The focused placement/rewrite/stat lane
  passes 36/36 in Host Debug and 36/36 under leak-disabled ASan/UBSan. All
  three CUDA context tests pass under normal execution and Compute Sanitizer
  memcheck, initcheck, racecheck, and synccheck; racecheck reports zero hazards.
  The guarded V4 process gate uses the hash-pinned 21,970,934-point LAS from
  D0057, executes two resident regions around seeded `filters.randomize`, and
  matches pinned PDAL output bytes, stdout, stderr, and exit status in 77.59
  seconds. That duration is validation timing, not B1/B2 evidence.
- Consequences: V4 now runs under a real resumable planner-owned boundary
  executor rather than inferred wrapper crossings. D1 liveness is executable
  for the selected assign/ferry regions, and allocation failure is fail-closed
  before writer side effects. Catalog breadth remains frozen: shared-index
  neighborhood/kNN residency, natural over-budget V5 tiling, and the remaining
  V/E gates must use this contract. The next bounded slice aligns D4's selected
  boundary terms with the executor and then closes V4/E2 only if prediction,
  observation, and exact process output all agree.

## D0059 — Bind placement boundary bytes to the selected executor

- Date: 2026-08-07
- Status: accepted executor-accounting checkpoint; E2 is satisfied; D4 and
  P1.5 remain open because the generic resident executor's compute residual is
  not yet calibrated
- Context: D0058 made V4's four physical crossings executable and observable,
  but the D0054 calibration-default formula counted logical live columns plus
  raw input/output/fallback record terms. The resumable `PointView` executor
  instead transfers one complete runtime record in both directions, packs the
  complete record before each upload, and repacks only surviving written
  columns before each spill. Copying transfer volume into packing diagnostics
  or changing the frozen default formula would either misstate execution or
  invalidate the 52-case B3 audit.
- Decision: Permit a concrete executor to supply one stable-ID boundary fact
  per planner boundary: transfer bytes/point, explicit pack/repack bytes/point,
  and device staging bytes/point. A nonempty table must cover every boundary
  exactly once; missing, duplicate, out-of-range, zero-transfer, or zero-staging
  facts fail closed. These facts replace, rather than augment, the
  calibration-default transfer and packing formulas, so logical-column bytes
  remain diagnostic and cannot be double-counted. They use the existing frozen
  linear coefficients. The fact table also declares a D3-swept configured lane
  count. Active width comes from the shared scheduler's tile-count rule, and
  peak VRAM is that width times the sum of one full packed tile and the
  planner's plan-wide liveness/index high-water mark. A budget that cannot
  hold the full calibrated width selects host; placement never silently
  inherits an unmeasured one-lane residual. An empty fact table preserves D0054
  byte-for-byte.
- Decision: Make spill repacking planner-owned. Each spill declares
  `repackDimensions` as the intersection of its post-completion release set and
  dimensions written by its resident region, plus the corresponding physical
  bytes/point. Read-only carried columns remain live through the transfer but
  do not become repack work; dead writes released before the boundary are also
  excluded. The admitted linear assign/ferry executor builds its fact table
  from the prepared physical `PointView` stride and these planner dimensions.
  Direct terminal LAS placement is evaluated first with the unchanged B0005
  formula. Only a declined direct path whose every candidate resident region
  passes the side-effect-free point-program rewrite envelope is reevaluated
  with executor-declared `PointView` facts; an unsupported region leaves the
  calibration-default request and stats label intact.
- Decision: Report `boundary_accounting_model=executor_declared` separately
  from `selected_device_calibration_matches_executor`. V4 may therefore report
  exact predicted/observed boundary bytes while continuing to state that its
  generic compute residual was not measured by B0005. Boundary agreement closes
  E2, not D4, E5, B1, or B2.
- Evidence: The focused model/planner lane passes 27/27 in Host Debug, 30/30
  with the three CUDA resident-context tests on the physical RTX 4090, and
  40/40 under leak-disabled ASan/UBSan. All three context tests pass Compute
  Sanitizer memcheck, initcheck, racecheck, and synccheck with zero errors or
  hazards. The default calibration audit remains 52/52 without coefficient or
  manifest changes. V7 still selects host and now also proves that an
  unsupported approximate-coplanar region retains calibration-default
  accounting; the combined gate passes in 0.72 seconds. The direct calibrated
  control remains exact in 10.57
  seconds. The guarded 21,970,934-point V4 process reports four ordered
  boundary IDs whose predicted transfer and packing bytes equal the observed
  executor totals. Its predicted configured/active lane counts and peak VRAM
  also equal the executor schedule, while output bytes, stdout, stderr, and
  exit status match pinned PDAL; its 78.10-second duration is validation
  timing, not B1/B2 evidence.
- Consequences: V4 now executes with explicit planned/reported host boundaries
  and exact output, satisfying E2. The generic executor remains default-off and
  must not be called calibrated or performance-qualified. D4 stays open until
  its residual predicts meaningful measured placement calls, while E5 remains
  the unchanged 52/52 frozen-matrix result. The next bounded load is natural
  over-budget V5 tiling through the same resident interface; no stage port or
  coefficient change follows from this decision.

## D0060 — Qualify natural over-budget resident tiling

- Date: 2026-08-07
- Status: accepted generic-resident V5 checkpoint; V5 and E3 are satisfied;
  neighborhood tiling, D4, and P1.5 remain open
- Context: D0058 executed a large mixed pipeline in fixed 131,072-point tiles,
  but its normal 75%-of-free-VRAM budget could also hold the hypothetical
  whole-view working set. That proved reuse and exact host boundaries, not
  V5's required condition that tiling is necessary under the selected planner
  budget. Running an additional whole-view CUDA lane would require a
  1.76-GB pinned/device allocation merely to duplicate the pinned PDAL
  whole-view oracle, increasing machine risk without strengthening default
  compatibility.
- Decision: Report the scheduler's input cardinality and tile capacity, and
  report `untiled_device_bytes` as one hypothetical whole-view PointView lane:
  point count times the sum of physical record stride and the planner's
  plan-wide resident liveness/index high-water mark. This is diagnostic only;
  selected placement continues to use the bounded active-lane peak.
- Decision: Add the internal
  `PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES` validation control. It is consulted
  only after the calibrated direct executor declines and every candidate
  region passes the generic resident rewrite envelope. The value must be a
  positive integer no greater than the normal free-VRAM-derived budget. It
  changes the generic placement request and the same planner budget passed to
  preflight/execution; it never changes production defaults, coefficients,
  tile size, lane policy, or the direct path. Invalid values fail before writer
  execution.
- Decision: Use pinned PDAL standard-mode execution as V5's untiled reference.
  This chain is deterministic and exact, so the process gate requires complete
  byte, stdout, stderr, and exit-status identity, which is stronger than E3's
  Tier-N tolerance. This closes V5/E3 for the generic resident point-program
  machinery only; it does not qualify shared-index neighborhood or kNN tiling.
- Evidence: The hash-pinned 790,953,999-byte, 21,970,934-point LAS from D0057
  runs two assign/ferry regions around seeded `filters.randomize` with a
  268,435,456-byte planner budget. The reported one-lane untiled working set is
  1,757,674,720 bytes. The selected schedule uses 168 tiles of 131,072 points,
  two configured/active lanes, 166 lane reuses, and a 20,971,520-byte
  predicted and observed peak. All four boundary predictions match execution,
  preflight is accepted, and the output hash
  `d1788a838a2518147534fc4d89419b555f0e719e9eebdf73ce66496ab7431870`
  matches pinned PDAL. Two clean runs take 79.11 and 79.40 seconds; these are
  validation timings, not B1/B2 evidence. An oversized validation budget is
  rejected without replacing a sentinel output, while the direct calibrated
  control ignores that generic-only variable and remains exact in 10.55
  seconds. V7 remains the host negative control in 0.67 seconds. The focused
  placement/scheduler lane passes 29/29 in Host Debug, 32/32 with the three
  resident CUDA context tests on the physical RTX 4090, and 40/40 under the
  leak-disabled ASan/UBSan build. The three context tests are clean under
  Compute Sanitizer memcheck, initcheck, racecheck, and synccheck; the frozen
  calibration audit remains 52/52.
- Consequences: The natural budget inequality is explicit:
  `peak_device_bytes <= budget < untiled_device_bytes`, and E3 is closed with
  an exact untiled oracle. No benchmark, coefficient, public coverage, or
  automatic-selection claim follows. The next bounded slice completes V1's
  cardinality-changing expression case through resident machinery; V2/V3
  still require the resident shared-index interfaces.

## D0061 — Qualify the declared cardinality-changing resident expression case

- Date: 2026-08-07
- Status: accepted V1-completion checkpoint; the cardinality-changing V1
  expression case is satisfied for the generic resident point-program
  machinery; D2's feature-kernel fusion, D4, and P1.5 remain open
- Context: D0052 completed only the cardinality-preserving V1 assign/ferry
  chain. The resident PointView rewrite envelope, runtime placement, executor
  preflight, and point-program filter all rejected any declared cardinality
  change, so a `filters.expression` stage inside an otherwise resident region
  forced the whole pipeline back to host with an avoidable boundary. Exact
  predicate evaluation and stable compaction existed only in the direct
  fused-LAS path, and no gate proved `where`/`where_merge` legality from
  descriptors.
- Decision: Admit a cardinality change into a resident region only when the
  stage descriptor declares a pure, deterministic-safe, order-preserving
  contract without conditional where semantics, the payload is a compiled
  `PredicateProgram`, and it is the only declared change in its region. The
  rewrite envelope, runtime placement, and preflight each verify the declared
  flags and payload shape; none of them inspects stage names or option lists.
  A declared cardinality change must terminate the selected resident chain
  because later tiles address original-view offsets. Where-bearing forms stay
  declared on the descriptor (`hasWhere`, `whereMerge`) and keep the
  fail-closed host path; a device-incapable or undeclared cardinality change
  retains the pre-existing `non_cardinality_preserving_stage` rejection.
- Decision: Execute the predicate into a planner-owned one-byte-per-point
  lane keep mask instead of compacting on device. Assignments after the
  predicate run on all tile rows; the exact VM has no per-point value
  diagnostics, dropped rows publish nothing, and the survivor output view is
  appended in stable tile/row order by ordered lane drains. The spill
  transfers the complete input tile plus the mask, so every predicted and
  observed boundary byte stays input-cardinality-based and exactly equal
  despite the data-dependent output. The spill boundary fact declares record
  stride plus one mask byte; the lane allocation, preflight probe, and
  placement peak all charge the same byte, and the schedule uses the
  `OrderedPointProgram` class with the unchanged D0053 two-lane default.
  Dead in-region writes remain planner-released before the spill and are
  never published, exactly as D0059 declared them.
- Decision: The region's placement request reuses the frozen
  fused-point-program residual when its assign/ferry content passes that
  envelope and its single predicate passes the exact device envelope. The
  predicate's compute residual is not separately measured: the executor
  continues to report `selected_device_calibration_matches_executor=false`
  and D4 stays open. Uncomposed or repeated predicates select host with
  `outside_calibration_envelope`. The calibration-default formulas, frozen
  coefficients, and D0054 manifest are untouched.
- Decision: Repair one latent machine-dependent test expectation.
  `pdg_resident_pipeline_process` asserted `profile_not_exact` for its
  two-assignment control, which only occurs in builds without a usable CUDA
  profile; the exact SM-89 machine reports the same fail-closed decision as
  `outside_calibration_envelope`. The assertion now accepts both. No product
  behavior changed. Separately, recorded process-gate output hashes are
  point-in-time evidence: `writers.las` embeds the UTC file-creation
  day-of-year, so candidate and pinned-oracle bytes drift together across a
  UTC midnight; the byte-identity gate itself is unaffected (the V5 rerun in
  this checkpoint matched pinned PDAL exactly with UTC day 220 while D0060
  recorded day-219 bytes).
- Evidence: Host Debug passes 325/325 registrations with the expected corpus
  skip, including the new rewrite, runtime-placement, placement-model, and
  plan declaration tests. The physical SM-89 CUDA Release tree passes its
  full 457-test registration with only the corpus and fixture-gated process
  skips; the four resident context tests, including the new
  cardinality-changing region regression (11 points in four-point tiles,
  seven survivors, an empty final tile, ordered survivor appends, an
  unpublished dead intermediate, and a full-record-plus-mask spill event),
  are clean under Compute Sanitizer memcheck, initcheck, racecheck, and
  synccheck with zero errors and zero hazards. The frozen placement audit
  remains 52/52 and the calibration manifest verifies 58/58 raw reports. The
  hash-pinned 790,953,999-byte, 21,970,934-point LAS runs the fused chain,
  `Intensity <= 10000`, and one post-predicate assignment as one resident
  region: placement selects the device, preflight accepts, 168 tiles of
  131,072 points run on 2/2 lanes with 166 reuses, the predicted
  21,233,664-byte peak bounds the observed 19,922,944 bytes, every planned
  boundary byte (including the spill's +1 mask byte per point) equals its
  observation, and the reported output cardinality equals the oracle's
  14,671,481 survivors. Output bytes, stdout, stderr, and exit status match
  pinned PDAL exactly in two consecutive runs (16.31 s and 16.28 s, output
  SHA-256
  `6c1d5110402adcd869547ed9071d8f87186e134d21733b44c35d6d932951fc74`); these
  are validation timings, not B1/B2 evidence. The where-bearing small control
  is rejected by declared semantics (`non_cardinality_preserving_stage`) and
  delegates exactly. V7 (0.67 s), the direct calibrated control (10.38 s),
  V4 (74.71 s), and V5 (76.17 s) all remain exact. The complete
  leak-disabled ASan/UBSan tree passes its 318 registrations with only the
  expected corpus skip.
- Consequences: V1 is complete for the generic resident interface, closing
  the last D0052 V1 obligation. Catalog breadth remains frozen: V2/V3
  resident shared-index and shared-kNN interfaces, V6 rebuild accounting,
  E1, E4, E6, E7, and B1/B2 must still complete through this contract, and
  no automatic-selection, production-default, coefficient, or public
  coverage claim follows from this decision.

## D0062 — Pack resident boundaries through physical rows

- Date: 2026-08-08
- Status: accepted boundary-packing checkpoint; the generic resident
  executor's host boundary surface is rebuilt on physical rows and
  re-measured under B2; D4 and P1.5 remain open
- Context: B0012 recorded the first B2 acceptance lanes for the resident
  interface and showed the four-crossing mixed class losing to pinned PDAL
  at 0.885x. New planner-owned host-boundary phase accounting attributed
  16.09–16.53 s of the 41.0 s candidate to per-field `getPackedPoint`
  upload serialization and 2.58–2.61 s to `setField` publication, while
  lane completion waits totaled about 2 ms — the device was never the
  limiting resource. Investigating a whole-row copy exposed a latent
  semantic trap: the execution `PipelineManager` owns a `ColumnPointTable`
  whose `finalize()` rewrites `Dimension::Detail` offsets into
  per-dimension column ordinals, so `dimOffset` was never a byte offset on
  the execution path. The pre-existing packed layout survived only because
  it was defined independently by summation and never consulted those
  offsets; the filter's `dimOffset`-based packed columns were silently
  ignored. The row-backed probe additionally proved that `PointView` rows
  match `dimOffset` byte-for-byte on row tables, including custom
  dimensions.
- Decision: A resident-selected execution runs the rewritten pipeline
  against a dedicated row-backed `pdal::PointTable` via
  `validateStageOptions` and terminal `prepare`/`execute`, while every
  non-resident decision keeps the manager's standard column-table path
  byte-for-byte. Table storage does not change output bytes, point order,
  or diagnostics; the oracle remains pinned PDAL standard mode and every
  gate still requires complete identity.
- Decision: The packed tile layout is the physical row layout. `bindLayout`
  fails closed before any boundary executes unless the declared dimension
  offsets exactly partition the record — sorted, contiguous, and summing to
  the point stride — so a layout whose offsets are not byte offsets can
  never pack garbage. Lanes serialize one complete row per point with a
  single copy and publish written columns by direct row writes during
  drains; a table without row storage (`getPoint()` returning null) keeps
  the semantic field-by-field path at the same physical offsets. The
  resident context CUDA tests now bind row-backed tables to match
  production execution, and resident stats report the aggregate
  `host_boundary_phase_seconds` (upload_pack, spill_wait, spill_publish)
  as permanent boundary observability.
- Evidence: Host Debug and the leak-disabled ASan/UBSan tree each pass
  325/325 registrations with the expected corpus skip. The physical SM-89
  CUDA Release suite passes with only the corpus and fixture-gated skips in
  97.35 seconds; the four resident context tests are clean under Compute
  Sanitizer memcheck, initcheck, racecheck, and synccheck with zero errors
  and zero hazards. All five hash-pinned process gates pass: V7 0.64 s,
  V4 45.30 s, V5 46.40 s, V1 15.19 s, and direct 10.07 s, with V4 falling
  from 74.71 s and V5 from 76.17 s at the prior checkpoint. The diagnostic
  mixed run is byte-identical to pinned PDAL with `upload_pack` at 0.807 s
  (from 16.09–16.53 s) and `spill_publish` at 1.169 s (from 2.58–2.61 s).
  Clean B0013 re-runs the three B0012 lanes with byte-identical artifacts:
  the mixed class moves from 0.885x to 2.879x, the expression class from
  1.254x to 1.448x, and the direct control records 19.577x, with unchanged
  schedules and exact boundary-byte agreement throughout. The host was
  power-cycled between the two records (the RTX 4090 had dropped off the
  PCI bus after a reboot), so cross-entry absolute times carry that caveat
  while within-entry ratios remain same-machine and clean.
- Consequences: The generic resident executor now wins every measured B2
  class, and the remaining candidate-versus-direct gap is dominated by the
  PDAL reader/writer and host-stage segments rather than boundary packing.
  The boundary facts, coefficients, frozen audit, and production defaults
  are untouched; D4 still requires a measured compute residual for the
  generic executor, and V2/V3, V6, E1, E4, E6, E7, and B1 remain open. No
  automatic-selection or public coverage claim follows from this decision.

## D0063 — Fuse native LAS endpoints under declared compacting anchors

- Date: 2026-08-08
- Status: accepted endpoint-fusion checkpoint; the terminal expression class
  runs on the validated ordered direct sink; D4 calibration of that executor
  is the next required step and P1.5 remains open
- Context: After D0062, the generic executor won every measured B2 class,
  but the expression class (1.448x) still paid the PDAL reader, PDAL
  writer, and PointView boundary for a pipeline whose endpoints are native
  LAS files with proven layouts, while the direct fused control demonstrated
  19x on the same hardware. The ordered decode/predicate/pack sink that
  D0011 validated for `pdg pipeline` automatic selection already executes
  declared assign/ferry/predicate chains with stable compaction and
  final-size truncation, but the resident command's direct branch declined
  every predicate payload, and the planner never formed a fusion candidate
  across a cardinality change.
- Decision: Fusion legality stays declarative. `writers.las` declares
  `acceptsCompactingPrologue` because its ordered pack/summarize machinery
  performs declared stable compaction; a declared stable predicate is
  `fusableAsPrologue` and never an epilogue; chain assembly admits a
  declared pure, deterministic-safe, order-preserving cardinality change;
  and `pointFusionLegal` accepts a non-cardinality-preserving chain only as
  the consumer prologue of a compacting anchor. The resulting
  writer-prologue `fusionCandidate` covering the complete middle chain is
  the declared proof the resident direct branch requires before lowering
  the chain into an `OrderedPointProgram` and executing the ordered sink,
  which now consumes the planner budget and reports transfer metrics and
  its D3 schedule exactly as the fused variant does.
- Decision: Honest calibration state is preserved. The ordered executor is
  reported as `direct_ordered_las` with
  `selected_device_calibration_matches_executor=false`: B0005 calibrated
  only the fused executor, and the ordered compute residual becomes a
  calibration term only through the pending D4 measurement. Mixed pipelines
  keep their host middle stages — a host stage between regions blocks
  endpoint fusion by construction — and the v1 gate now proves the
  fused-endpoint route byte-identically while a randomize-prefixed control
  keeps the generic keep-mask executor covered end to end.
- Evidence: Host Debug passes 326/326 registrations with the expected
  corpus skip. The physical SM-89 CUDA Release suite is green in 101.9
  seconds; the four resident context tests remain clean under Compute
  Sanitizer memcheck, initcheck, racecheck, and synccheck. All five
  hash-pinned gates pass (V7 0.58 s, V4 47.85 s, V5 48.32 s, V1 49.12 s
  including its new generic control, direct 10.22 s). Clean B0014 records
  the fused-endpoint expression chain at 18.038x pinned PDAL (0.452977 s
  versus 8.170843 s) with byte-identical artifacts and the required
  `direct_ordered_las` proof on every run, the fused direct control at
  19.231x, and the unchanged generic mixed class at 2.941x.
- Consequences: The expression class reaches the direct executor's
  performance envelope without giving up planner-declared legality or exact
  fallback. Before V2 begins, D4 must calibrate the fused-endpoint executor
  so placement predicts with measured residuals instead of borrowed ones;
  the V2/V3 shared-index scope itself is unchanged (D0050 froze breadth,
  not depth). No automatic-selection, production-default, or public
  coverage claim follows from this decision.

## D0064 — Calibrate the ordered-point-program residual for the fused-endpoint executor

- Date: 2026-08-08
- Status: accepted D4 calibration checkpoint for the fused-endpoint
  `direct_ordered_las` executor; the frozen 52-row matrix is untouched and
  audits with the six appended rows at 58/58; one pre-existing provenance
  loss is recorded below and P1.5 remains open
- Context: D0063 routed the terminal expression chain onto the ordered
  decode/predicate/pack sink while explicitly reporting
  `selected_device_calibration_matches_executor=false`, because no measured
  residual existed for that executor. The accepted sequencing places this
  calibration before V2.
- Decision: Append the measured `ordered-point-program` stage model to the
  frozen SM-89 profile. Six clean forced ordered-CUDA rows at 1M, 2M, 4M,
  8M, 16M, and 21,970,934 points — complete pinned-PDAL process versus the
  same ordered sink through the forced `pdg pipeline` lane, prefix fixtures
  derived from the hash-pinned source, frozen time, byte-identical outputs
  on every run — feed the audit's own `--suggest-models` regression:
  `host_fixed_ns 9733181.294258`, `host_ns_per_point 366.606724`, zero
  device residual (the infrastructure terms model the flat device profile),
  `minimum_device_points 1,000,000` as the fail-closed envelope floor. The
  measured host curve is linear to within sampling noise across the full
  domain and the device side is flat at the startup-dominated cost, so the
  bounded linear model form fits the class without revision pressure.
- Decision: A point-program region containing the single declared predicate
  now resolves to `ordered-point-program` (its assignment content must stay
  inside the bounded fused program envelope the measured class used);
  assign/ferry-only regions keep their existing resolution. The direct
  ordered executor reports `selected_device_calibration_matches_executor=
  true` when its selected region resolved to the measured model. The
  generic PointView executor continues to report false everywhere: its
  keep-mask boundary work is a different executor and remains explicitly
  uncalibrated.
- Recorded provenance loss: the seven D0049 approximate-coplanar raw
  reports were retained under `/tmp` and were destroyed by the two host
  reboots of 2026-08-08 (the GPU PCI dropout and its power-cycle repair).
  Their SHA-256 digests remain recorded in the manifest and their audit
  decisions still verify, but
  `verify_placement_calibration.py --require-all` now fails on those seven
  physical files. This predates this decision's changes; regeneration would
  require rerunning the D0049 qualification lanes at their recorded
  revision. Disposition is deliberately left to a follow-up decision rather
  than silently relaxing the provenance gate.
- Evidence: Host Debug passes 246 registrations with the expected corpus
  skip. The placement audit predicts 58/58 winners (100%), including host
  selection below the envelope and device selection across the six measured
  ordered rows; the manifest carries the six new reports with verified
  digests under `build/benchmarks/ordered-cal-*-e48e67ad6.json`. The full
  physical CUDA suite is green; all five hash-pinned gates pass with the v1
  gate now asserting the ordered executor's calibration match (V7 0.62 s,
  V4 47.61 s, V5 49.45 s, V1 48.18 s, direct 10.24 s). The 21,970,934-point
  fused-endpoint expression run reports `direct_ordered_las` with
  `selected_device_calibration_matches_executor=true` and remains
  byte-identical to pinned PDAL.
- Consequences: D4's executable-placement obligation is satisfied for the
  fused, simple-ferry, and ordered direct classes; the generic PointView
  executor's residual remains the explicitly open D4 surface, and E5 now
  stands at 58/58 for the extended matrix. V2 proceeds next through the
  resident shared-index interface with unchanged scope, subject to the
  provenance-loss disposition above.

## D0065 — Waive lost D0049 report files by explicit status and fix report storage

- Date: 2026-08-08
- Status: accepted disposition of the provenance loss recorded in D0064
- Decision: The seven D0049 approximate-coplanar raw reports destroyed by
  the 2026-08-08 host reboots are marked
  `physical_file: lost-to-environment-2026-08-08-host-reboots` in the
  calibration manifest. `verify_placement_calibration.py` exempts only
  explicitly marked reports from `--require-all`, reports them separately
  as waived, still fails on any unmarked missing file, and still requires a
  restored file to match its recorded digest. The rows' measured values,
  digests, and audit decisions are unchanged; the frozen matrix remains
  frozen and the fitted models are untouched.
- Decision: Raw calibration and benchmark reports must be retained under
  `build/benchmarks/`, never `/tmp` or another volatile location. Every
  report referenced by D0064's ordered-point-program rows already follows
  this rule.
- Evidence: `verified 57/64 unique raw reports; missing 0; waived-lost 7`
  under `--require-all`, and the placement audit remains 58/58.
- Consequences: Provenance verification is again a passing gate with the
  loss explicit rather than silent. Regenerating the seven reports at
  their recorded revision remains possible under a future decision; until
  then their digest-only status is visible in every verification run.

## D0066 — V2: shared-index neighborhood regions execute under the resident planner

- Date: 2026-08-08
- Status: accepted, physically validated on the RTX 4090 environment
- Decision: The resident planner admits single-stage shared-index
  neighborhood regions (filters.approximatecoplanar first) selected by the
  measured `approximatecoplanar` model, and the resident executor runs them
  as a delegated whole-view region: the planner emits a
  `HybridApproximateCoplanarStage` wrapper (plus Ferry/Assign bridge
  point-program stages) carrying `pdg_resident_context` and
  `pdg_execution_region`, and `ResidentExecutionContext` verifies the
  whole-view budget as `estimatedDeviceBytes(N) + N × 146` scratch bytes
  per point (96 cached eigensystem + 48 covariance scratch + 2 status),
  hand-builds a single-tile/one-lane `whole_view_neighborhood` schedule,
  and brackets execution with `DeviceRegionBegin`/`DeviceRegionEnd`
  events. Delegated boundaries account logical `boundary.bytesPerPoint`,
  not the physical row stride. The exact D0045-validated
  `tryCudaApproximateCoplanarColumn` path (with host KD3 ambiguous-eigen
  repair) is mandatory in this mode — no host fallback; failure closes the
  pipeline.
- Decision: Model anchoring in runtime placement follows the D0055
  residual-per-region rule: exactly one measured non-point-program model
  anchors a region; point-program bridge content joins at zero incremental
  cost; two distinct measured models in one region remain
  `mixed_calibration_models`.
- Decision: Per-transfer H2D/D2H observation events at the neighborhood
  executor's memcpy sites are deliberately deferred; the stats report the
  region bracket only and
  `selected_device_calibration_matches_executor` stays `false` for this
  executor class until a dedicated calibration exists (the D0056
  wrapper-era pattern). This keeps the accounting honest rather than
  synthesized.
- Evidence: `pdg_resident_placement_v2_gpu` — reader →
  approximatecoplanar(knn=8) → ferry(Coplanar=>UserData) → writer over the
  hash-pinned 21,970,934-point fixture; executor
  `planner_resident_shared_index`, pipeline class
  `whole_view_neighborhood`, one tile/one lane, peak lane 5,448,791,632
  bytes within the 18,386,190,336-byte budget, output byte-exact against
  the pinned oracle (candidate sha256
  46d9fc9f6df539f6da96b1f8c2d3cf8b298e2974a10256e03429e81be51fc142).
  Host tree 328/328; full CUDA suite green with all six GPU gates.
- Consequences: V2 is complete: the resident planner/executor interface
  now spans point programs, cardinality-changing expressions, fused
  endpoints, and shared-index neighborhood work. V3 (LOF through the same
  shared-kNN interface) and E6 (approximatecoplanar break-even) build
  directly on this slice.

## D0067 — V3: LOF executes through the resident shared-kNN interface

- Date: 2026-08-08
- Status: accepted, physically validated on the RTX 4090 environment
- Decision: The shared spatial engine gains an exact local-outlier-factor
  primitive: `knnLofValues` computes filters.lof's three passes on one
  retained k-nearest adjacency (k-distance, inverse ordered-online-mean
  reachability density, ordered-online-mean factor) with upstream's literal
  binary64 operation order, as three CUDA kernels over the shared
  `knnGather` adjacency and an identical host mirror. Because the passes
  observe neighbor identities one and two hops out, the primitive reports a
  per-row `neighborStatus` — the OR of the row's neighbor statuses — so
  consumers can repair exactly the affected closure: an ambiguous own row
  (distance tie) rebuilds its density and factor on the compatibility
  KD3Index in nanoflann order; a row that merely reaches an ambiguous row
  rebuilds only its factor from the repaired densities. The k-distance is
  distance-valued and tie-invariant. Any incomplete search fails the device
  path closed. `tryCudaLofColumns` publishes the three columns and patches
  the device columns after repair so a retained region's bridges stay
  canonical.
- Decision: `PdgLofFilter` (`filters.pdg_lof`) is the exact compatibility
  wrapper: its host fallback replicates upstream LOFFilter's arithmetic —
  including the deliberately stateful per-filter()-call `minpts` increment,
  which the wrapper reproduces across multiple views — and its CUDA path is
  the shared-kNN client above. The planner compiles `filters.lof` into a
  `LofProgram` payload on the measured `lof` placement model with a
  declared kNN index request of `minpts + 1` (the self-inclusive
  increment), emits it through the same shared-index neighborhood region
  rewrite as V2 (wrapper node plus assign/ferry bridges), and the resident
  context admits it as a delegated whole-view region whose executor scratch
  is the retained adjacency: `neighbors × 12 + 2` bytes per point, taking
  the maximum against the eigensystem family's 146 when composed.
- Decision: The measured `lof` stage model is appended to the SM-89
  profile from six clean forced hybrid-CUDA rows at 250k, 1M, 2M, 4M, 8M,
  and 16M points (complete pinned-PDAL process versus the forced wrapper
  lane, prefix fixtures of the hash-pinned source, frozen time,
  byte-identical outputs on every run; raw reports
  `build/benchmarks/lof-cal-*-86f2aeaf2.json`): `device_fixed_ns
  583604713.605761`, `host_ns_per_point 7519.611843`, envelope floor
  `minimum_device_points 250,000`. A 50k probe measured the device 1.33x
  ahead but sits below the declared floor and outside the frozen matrix;
  its report is retained. The audit stands at 64/64 with the six rows.
- Evidence: `pdg_resident_placement_v3_gpu` — reader → lof(minpts=10) →
  assign(`UserData = 1 WHERE LocalOutlierFactor >= 1.2`) → writer over the
  hash-pinned 21,970,934-point fixture; executor
  `planner_resident_shared_index`, `whole_view_neighborhood`, one tile/one
  lane, peak lane 5,668,500,972 bytes within the 18,360,188,928-byte
  budget, output byte-exact against the pinned oracle (candidate sha256
  e21bbbec2aab920a0c05a3e7550d7a30046c525c1489f312ee59256b59839c9f).
  Wrapper differentials pin bit-equality against upstream on host and CUDA
  paths including a forced tie-repair dataset with duplicate points, and
  the stateful minpts increment across two views. Engine host/device LOF
  values compare bit-identical across both index backends. Host tree
  331/331; ASan/UBSan tree 331/331; full 468-test CUDA suite green;
  memcheck/initcheck/racecheck/synccheck clean over the resident-context,
  LOF wrapper, and shared-kNN engine tests; all seven GPU gates pass in
  one serial run (v7 0.65 s, v4 51.95 s, v5 52.80 s, v3 259.03 s, v2
  146.19 s, v1 48.12 s, direct 10.14 s; 568.9 s total); calibration lanes
  measured 2.8x (250k) to 5.2x (steady state) over pinned PDAL.
- Consequences: V3 is complete; the resident shared-kNN interface now
  serves both declared neighborhood families. LOF's B2 acceptance lane and
  E6-style break-even requalification remain future benchmark work; V6
  (dirty-index rebuild accounting) is the next open load.

## D0068 — V6/E4: dirty-index rebuild accounting under the resident planner

- Date: 2026-08-08
- Status: accepted, physically validated on the RTX 4090 environment
- Decision: The resident executor supports a sequence of delegated
  whole-view neighborhood regions in one execution. Preflight admits a plan
  whose selected regions are all neighborhood regions (a plan mixing
  delegated and tiled point-program regions stays fail-closed); regions run
  in order through their planned upload/spill boundaries, and after a spill
  publishes every device result to host rows the context releases its
  PointView binding — a host stage between regions may legitimately
  republish the same points through a new PointView (upstream filters
  often do), while identity stays enforced within each upload/region/spill
  span.
- Decision: Physical shared-index builds are observed facts: the attach
  machinery records one IndexBuild event per `SpatialIndex::build()` with
  the planner-owned region identifier and the index's allocated bytes.
  `pdg resident --stats` reports `execution.index_builds` with the
  planner's prediction (selected regions charged a build), the observed
  event count, and a match flag. E4's exit holds: the observed rebuild
  count equals the planner's prediction and both appear in --stats.
- Decision: `filters.reprojection` gains a declared fallback contract: it
  stays the original PDAL host implementation, is a declared coordinate
  mutator (spatial-index invalidator), and declares the one-view
  cardinality contract only under `error_on_failure: true` — upstream
  silently drops points whose transform fails, so without that option the
  contract remains undeclared and placement fails closed.
- Evidence: `pdg_resident_placement_v6_gpu` — reader →
  approximatecoplanar(knn=8) → ferry(Coplanar=>UserData) →
  reprojection(EPSG:32615→32616, error_on_failure) → lof(minpts=10) →
  assign(`UserData = 1 WHERE LocalOutlierFactor >= 1.2`) → writer over the
  hash-pinned 21,970,934-point fixture. Placement selects both
  neighborhood regions with `predicted.index_build_count = 2`; execution
  reports two delegated regions, two IndexBuild events
  (`index_builds: {predicted: 2, observed: 2, matches_prediction: true}`),
  and byte-exact output against the pinned oracle (candidate sha256
  6a7cee88c61771901f4550c48f37d9964872d53428417db06b3ed1be25452dc4). The
  planner-half regression pins the two-region rewrite (two wrapper nodes
  around the untouched host reprojection, `summary.indexBuilds == 2`).
  Host tree 332/332; ASan/UBSan tree 332/332; full CUDA suite green; four
  Compute Sanitizer tools clean; all eight GPU gates pass in one serial
  run (v7 0.71 s, v4 47.38 s, v5 47.92 s, v6 408.33 s, v3 255.60 s, v2
  147.33 s, v1 48.63 s, direct included; 966.15 s total).
- Consequences: V6 and E4 are satisfied. B0015 records the resident B2
  acceptance lanes for both neighborhood families (approximatecoplanar
  1.809x, LOF 4.576x, byte-exact under the planner executor), satisfying
  E6's break-even exit. Remaining P1.5 items: E1, E7, and B1, which
  require the ncu roofline/occupancy lanes.

## D0069 — First ncu evidence: B1 and E7 measured; E1 fails as written

- Date: 2026-08-08
- Status: recorded measurements; E1 disposition deliberately escalated
- Context: Nsight Compute became available on the physical host (R610
  capability grant `/dev/nvidia-caps/nvidia-cap4324`;
  `RmProfilingAdminOnly=1` remains set and is overridden by the
  capability). All lanes ran unprivileged with the GPU otherwise idle.
- B1 (resident kernel throughput, 4M-point whole-view lanes, roofline
  set): `eigenSystemsKernel` runs compute-bound at 85.3% SM;
  `lofReachabilityKernel` and `lofFactorKernel` run compute-bound at
  88.2% and 90.1% SM; `lofKDistanceKernel` streams its retained adjacency
  at 93.9% of DRAM peak — the shared engine saturates DRAM when access
  is coalesced. `knnGatherKernel` is latency-bound (26.8–36.7% SM,
  ~0.8–0.9 ns/pt at k=9/11) — fast in absolute terms; its bound is
  traversal latency, not bandwidth. Reports:
  `build/benchmarks/ncu-b1-{coplanar,lof}.ncu-rep`.
- E7 (marginal fused-op cost, direct fused executor, 8.4M-point chunks):
  within the fused envelope an added admissible op costs zero registers
  (56/thread flat), no occupancy loss (59.0% → 59.6% achieved), zero
  added DRAM traffic within noise (the kernel reads and writes the whole
  packed record regardless of which fields an op touches: ~302 MB read /
  ~316 MB written per chunk in every variant), and ~150 µs per chunk of
  added execution time (~0.018 ns/pt/op). An op writing a dim outside
  the fused envelope does not grow the kernel — it routes the chain to a
  different executor by declared legality. Marginal cost is therefore
  measured, not assumed, and is execution-time-only inside the envelope.
  Reports: `build/benchmarks/ncu-e7-*.csv`.
- E1 (fused chains ≥ 70% of DRAM roofline on resident data): measured
  and NOT met. With 302 MB device-resident chunks (four times the RTX
  4090's 72 MB L2), `fusedTranslatePointProgramKernel` achieves 5.4% of
  DRAM peak, 8.2% SM, and 58.8% of the L1 subsystem — it is
  L1-transaction-bound on byte-granular packed 36-byte record accesses
  at ~1.4 ns/pt. Reaching the criterion requires restructuring record
  access (coalesced staging through shared memory or a transpose pass) —
  roughly a 20x kernel-level headroom. End-to-end context: at 22M points
  the direct fused pipeline spends ~30 ms in this kernel inside a ~1 s
  wall dominated by storage and PCIe, so kernel-level DRAM saturation
  would not visibly move B2. Report:
  `build/benchmarks/ncu-e1-fused-chain-8m.ncu-rep`.
- Consequences: B1 and E7 hold with recorded evidence. E1 fails as
  written; the honest options — a measured kernel-optimization slice
  targeting coalesced record staging, or a spec-level revision of the
  criterion against the measured end-to-end reality — change scope and
  are escalated rather than decided here. P1.5 remains open on E1 alone.

## D0070 — Staged fused-chain kernels: E1 measured to its interpreter floor

- Date: 2026-08-08
- Status: accepted kernel-optimization slice; E1 remains open at a
  measured, attributed gap
- Decision: The direct LAS translate kernels stage each block's contiguous
  packed-record span through shared memory with coalesced vectorized
  copies; formats 7 and 8 with a packed 36-byte stride mutate their staged
  input in place (the canonical record is byte-identical) and skip both
  the output staging span and the per-point record move. Point summaries
  aggregate per block in shared memory — integer minimum/maximum and
  counter addition are commutative and associative, so one merge per block
  is exactly the per-point global atomics' value. The fused expression
  interpreter keeps its top two stack elements and all eight dimension
  values in registers (a switch-indexed value file), evaluates conditions
  and values uniformly with a predicated store, and walks its instruction
  stream by pointer. The unstaged kernels remain the fallback for record
  strides whose staging exceeds the shared budget, and
  `__launch_bounds__(256, 5)` with the maximum shared-memory carveout
  fixes the occupancy/spill trade measured best on SM 89.
- Measured progression on the 8.4M-point device-resident chunk
  (`fusedTranslatePointProgramStagedKernel`, reports under
  `build/benchmarks/ncu-e1-*.ncu-rep`): 11.77 ms at 5.4% of DRAM peak
  (L1-transaction-bound baseline) → 6.43 ms at 9.5% (staging; then
  latency-bound on per-point global atomics) → 1.68 ms at 36.8%
  (block-local summaries; then local-memory-bound on interpreter spill:
  5.9 GB of local sectors against 604 MB of data) → 1.49 ms at 39.2%
  (register interpreter state; local traffic eliminated) → 1.13 ms at
  52.4% (occupancy tuning). Net 10.4x kernel improvement, byte-exact
  throughout.
- E1 disposition: the ≥70% DRAM-roofline exit remains unmet. The
  remaining 1.34x is measured, not assumed: the kernel issues ~80 warp
  instructions per point, and at the SM 89 issue ceiling that instruction
  volume alone floors the kernel at ~0.86 ms — exactly the 70% line — so
  every real inefficiency lands below it. The cost is the interpreted
  expression VM's dispatch and bookkeeping; closing E1 requires
  compiling fused programs to specialized straight-line kernels (a JIT or
  ahead-of-time program-specialization slice), which is new scope and is
  escalated rather than begun unilaterally.
- Evidence: host tree 332/332; full CUDA suite green; four Compute
  Sanitizer tools clean over the LAS/fused/translate tests; all eight
  GPU gates pass in one serial run (968.5 s). Re-run B2 lanes over the
  optimized kernels are byte-exact and slightly ahead end-to-end —
  direct fused control 19.721x (B0014: 19.231x), fused-endpoint
  expression 18.166x (B0014: 18.038x) — confirming the pipeline class
  stays storage/PCIe-bound end-to-end:
  `build/benchmarks/staged-assign-ferry-fused-22m-55573582d.json`
  (336078fa7d2a784ba0fc04096cc5389049b81cdd53de1eba9818e508494c01a1),
  `build/benchmarks/staged-resident-expression-22m-55573582d.json`
  (0f0329a9b843128ac1064402e936627b9d43856363c08eebdb6b93ad01acf165).
- Consequences: every P1.5 validation load, benchmark class, and exit
  except E1 is satisfied with recorded evidence. E1's residual is now a
  precise engineering statement with two dispositions — a
  program-specialization slice, or a spec-level revision against the
  measured end-to-end reality — and P1.5 remains open on that single
  decision.

## D0071 — E1 satisfied: fused programs specialize to straight-line kernels

- Date: 2026-08-08
- Status: accepted; E1 closes and with it every P1.5 validation load,
  exit criterion, and benchmark class
- Decision: Each admitted FusedPointProgram is compiled once at run time
  with NVRTC into a straight-line specialized kernel (`pdgJitFusedKernel`)
  with the program, record format, and stride baked in as constants. The
  generator walks the same instruction stream as the interpreter with a
  codegen-time operand stack, emitting the same exactly-rounded intrinsics
  in the same order (immediates and range bounds as hexadecimal float
  literals), the same store semantics per dimension type, and per-dimension
  literal decode/pack statements; conditions evaluate uniformly with a
  predicated store. Compilation is cached by generated source; any
  generation or compile failure falls back to the D0070 staged interpreter
  kernel (`PDG_DISABLE_FUSED_JIT` disables, `PDG_REQUIRE_FUSED_JIT` is the
  test hook). Specialization covers identity record formats 7 and 8 at the
  packed 36-byte stride — the measured E1 class; other shapes keep the
  interpreter. `pdg_core` now links `CUDA::nvrtc` and `CUDA::cuda_driver`.
- E1 evidence: on the 8.4M-point device-resident chunk the specialized
  kernel runs 654.7 µs at **89.4% of DRAM peak** (Memory Throughput equals
  DRAM Throughput — purely streaming-bound; SM 71.7%), against the 70%
  exit bar and the interpreter series' 52.4% ceiling (D0070). Report:
  `build/benchmarks/ncu-e1-jit-8m.ncu-rep`. Net kernel progression
  11.77 ms → 0.655 ms (18.0x) with byte-exact outputs at every step.
- Validation: host tree 332/332; full CUDA suite green (--rerun-failed
  re-ran only the expected corpus skip); memcheck/initcheck/racecheck/
  synccheck clean over the LAS/fused/translate tests; all eight GPU gates
  pass; the forced-JIT B2 lane over the 21,970,934-point fixture is
  byte-exact at 18.568x
  (`build/benchmarks/jit-assign-ferry-fused-22m-6f5fd6e45.json`,
  85b28222e4c0cae0aaec6fb52f8e6e9b1a5896cc82720573fed3459e6982d233).
- Consequences: P1.5 is complete: V1–V7, E1–E7, and B1/B2/B3 all hold
  with recorded physical evidence. Catalog-wide stage ports may resume
  strictly through the resident interface per the milestone's closing
  rule.

## D0072 — Measure the attach boundary honestly; move it onto physical rows

- Date: 2026-08-08
- Status: accepted post-P1.5 checkpoint; the whole-view attach boundary is
  measured, row-backed, and observable; the roadmap's first optimization
  surface is corrected to the kNN gather kernel
- Context: The post-P1.5 roadmap's first item assumed, from B0015's prose,
  that the resident coplanar lane spends most of its 52 s wall in the
  whole-view attach getField/setField loops. That attribution had never
  been measured: D0062's `host_boundary_phase_seconds` accounting covered
  only the boundary_batch executor's tile lanes, and the attach machinery
  in PdgNeighborhood.cpp recorded nothing. Poor-man's stack sampling was
  unavailable (`kernel.yama.ptrace_scope=1`, no perf/nsys installed, sudo
  out of bounds), so the measurement instrument became permanent phase
  accounting extended to the attach path itself.
- Decision: The attach machinery accounts its host boundary work into the
  active resident context's phase counters through a new
  `activeResidentPhaseSeconds()` accessor: whole-view XYZ gather and
  output-column pack loops as `upload_pack`, result-transfer stream waits
  as `spill_wait`, and column publication as `spill_publish`. `--stats`
  now reports one aggregate `host_boundary_phase_seconds` surface for both
  resident executors; a run without an active context records nothing.
- Decision: The attach gather and publish boundaries reuse the D0062
  row-backed fast path: when the PointView layout's declared dimension
  offsets exactly partition the record (the same fail-closed test as
  `bindLayout`, applied as a predicate) and the stored type equals the
  transferred type, columns copy bytes directly through `getPoint()` rows;
  every other table — and every per-point null row — keeps the semantic
  per-field path, which performs the identical conversion.
  `PDG_DISABLE_NEIGHBORHOOD_ROW_BOUNDARY` forces the field path for
  differential runs; `PDG_REQUIRE_NEIGHBORHOOD_ROW_BOUNDARY` is a proof
  hook that fails the invocation if the direct path does not engage, and
  every tryCuda* entry rethrows the proof failure rather than swallowing
  it into a fallback.
- Evidence: On the 21,970,934-point fixture the resident coplanar lane is
  byte-identical across the pre-change binary, the forced field path, and
  the row path (output SHA-256
  98db2d0efe1bc8f6bec0718ba8c735d091d8e2b325a15d54d948cbefeed58e87). The
  measured phases refute the roadmap premise: with the old field loops
  upload_pack is 0.600 s, spill_publish 0.273 s, and spill_wait 0.001 s of
  a 50.7 s wall (~1.7% combined); the row path lowers them to 0.446 s and
  0.175 s with wall unchanged within noise. The dominant measured cost is
  `knnGatherKernel`: the D0069 B1 reports record 3.12 s (coplanar, k=8)
  and 3.61 s (LOF, k=11) on the 4M fixture at 8.6% SM throughput, 0.02%
  DRAM, and 12.2% L1 — a latency-bound traversal, roughly 17–20 s at 22M
  scale. Validation: host Debug and the
  leak-disabled ASan/UBSan trees pass 332/332; the physical SM-89 CUDA
  Release suite is green with only the corpus and fixture-gated skips; the
  v2, v3, and v6 hash-pinned process gates pass byte-exactly in serial
  runs (147.5 s, 271.5 s, 405.3 s); and the new unit differential proves
  the direct path engages under the proof hook and matches the field path
  bit for bit.
- Consequences: The roadmap's first post-P1.5 item is rewritten: the
  attach boundary is closed as an optimization surface (0.62 s remaining
  at 22M) and the kNN gather kernel becomes the measured target, under the
  exact tie/incomplete status contract and the D3 fixed-scheduler rule.
  Placement coefficients, frozen audits, and production defaults are
  untouched; no coverage or selection claim changes.

## D0073 — kNN gather rebuilt around the FP64 pipe: Morton order, quick reject, contiguous candidates, certified float prefilter

- Date: 2026-08-09
- Status: accepted post-P1.5 optimization checkpoint; the uniform-grid kNN
  gather kernel is measured to each bottleneck and rebuilt around it with
  byte-exact outputs at every step
- Context: D0072 corrected the resident-lane attribution to
  `knnGatherKernel` (3.12 s coplanar k=8 / 3.61 s LOF k=11 on the 4M
  fixture at 8.6% SM, 0.02% DRAM). The full ncu series then walked the
  kernel through three distinct regimes: (1) 63% idle lanes and 57.5e9
  local-load sectors (~1.8 TB) from the per-candidate insertion scan of
  the local-memory best arrays, tex-throttle dominant; (2) after the local
  traffic fell 18x, dependent-load latency chains; (3) after contiguous
  candidate loads, SASS sampling put ~75% of stalls on DSETP/DADD/DMUL —
  the FP64 pipe itself, which issues at 1/64 FP32 rate on consumer Ada
  (about 176 SM-cycles per candidate-warp for the ~11-op distance/compare
  chain), the arithmetic that reproduces the measured duration.
- Decision (all output-content-preserving, each measured on the 4M lane):
  1. Queries walk in the index's Morton order (`sortedPointIds[slot]`) so
     warps process spatially adjacent points; pure thread-to-query
     remapping. 3.13 s -> 2.74 s.
  2. A register-cached copy of the worst retained entry rejects candidates
     before they touch the local arrays. The (distance, id) precedence is
     a strict total order, so "worst precedes candidate" is exactly the
     insertion-past-capacity discard. 2.74 s -> 1.85 s; local-load
     sectors 37.7e9 -> 2.1e9.
  3. The index build gathers the batch coordinates into sorted order
     (three doubles per point) so candidate loops read bit-identical
     operands contiguously instead of gathering through the ids.
     1.85 s -> 1.75 s.
  4. A certified float prefilter skips the FP64 chain for candidates whose
     margin-reduced float squared distance provably exceeds the round-up
     float of the current worst. Coordinates are grid-origin-shifted
     floats (extent E, unit u = 2^-24): per-axis difference error is
     bounded by 3Eu, the float sum's true value by
     S(1-4u) - 2*(3Eu)*sqrt(3S) - 3*(3Eu)^2; the shipped constants are
     built from 2^-23 with a further 2x inflation of the absolute terms
     and 64-ulp relative slack, an order of magnitude beyond every
     rounding the chain consumes (including the double path's own).
     Equality can never be certified, so ties always reach the exact
     path. 1.75 s -> 1.40 s (2.24x cumulative); LOF k=11
     3.61 s -> 1.05 s (3.4x). `PDG_DISABLE_KNN_DISTANCE_PREFILTER`
     forces the exact path; a frame too large for float squared
     distances disables the filter on the host side.
- Decision: The sorted candidate copies (24 B/point doubles + 12 B/point
  prefilter floats) are allocated only for device uniform-grid indexes
  whose config carries the new `knnCandidateArrays` intent flag set by the
  kNN config builders; radius-only and BVH indexes never pay for them.
  The kNN launchers fail closed if the arrays are absent,
  `SpatialIndex::allocatedBytes` reports them, and the planner's kNN index
  estimate grows to 112 B/point (28 grid + 48 BVH bounds + 36 sorted
  copies) in both the summary and lifetime formulas.
- Measured dead ends (reverted): `__launch_bounds__(64, 21)` forced 40
  registers and spilled (1.88 s); 128-thread blocks were neutral.
- Evidence: every step byte-identical on the 4M coplanar and LOF lane
  outputs (SHA-256 490955d7... and 19c6f60a...). New differential
  `KnnDistancePrefilterNeverChangesABit` (tie-dense quantized cloud,
  prefilter on/off, bitwise). Full validation: host Debug and the
  leak-disabled ASan/UBSan trees pass 332/332; the physical SM-89 CUDA
  Release suite passes with only the corpus and fixture-gated skips;
  memcheck, initcheck, racecheck, and synccheck are clean over the
  spatial-index, neighborhood-column, and LOF tests; and the v2, v3, and
  v6 hash-pinned 22M gates pass byte-exactly in serial runs (149.6 s,
  268.8 s, 407.1 s).
- Consequences: The uniform-grid kNN primitive that feeds every shared
  neighborhood family more than halves its device cost; roadmap item 1
  is closed for the uniform-grid backend. The BVH gather keeps the
  original id-indirect loads and remains a candidate for the same series
  if a profile ever shows it selected on a hot path. Placement time
  coefficients are unchanged in this entry; resident B2 lanes are
  re-measured in B0016.

## D0074 — Device shell budget: the kNN tail moves to the host repair contracts

- Date: 2026-08-09
- Status: accepted; the outlier-tail regime B0016 identified is closed and
  roadmap item 1 completes for the uniform-grid backend
- Context: B0016 measured the 22M `knnGatherKernel` at 32.20 s with the
  GPU 87% idle: a small population of isolated points walks enormous
  empty shell rings (up to MaximumKnnShell = 4096 shells of ~24*s^2 cells,
  each a binary-search probe) while every other lane drains. The exact
  status contract already had the right shape — a row that cannot prove
  completeness reports `KnnSearchIncomplete`, and the eigen family
  repaired such rows on the host since D0045 — but the LOF, NNDistance,
  and statistical-outlier consumers still treated any incomplete row as a
  whole-stage device failure, and nothing bounded how long the device
  would try before declaring one.
- Decision: The uniform-grid kNN walk stops trying at a device shell
  budget (default 32 shells, `PDG_KNN_DEVICE_SHELL_BUDGET` in [1, 4096]);
  rows beyond it report the existing `KnnSearchIncomplete` and carry
  placeholder values. The budget bounds effort, never correctness:
  MaximumKnnShell remains the frame ceiling, and every consumer repairs
  declared-incomplete rows exactly on the host through the compatibility
  index in upstream's operation order:
  - eigen family: unchanged (D0045 machinery already covers
    Tie | Incomplete rows);
  - NNDistance: per-row recompute for the declared mode (kth
    `sqrt(sq[k-1])`; average `sum of sqrt over 1..k-1 then divide`);
  - statistical outlier: per-row ordered online mean recompute;
  - LOF: the closure extends one hop beyond ties because an incomplete
    row's k-distance itself is unknown. A new `lofClosureKernel` emits
    `neighborNeighborStatus` = OR of (status | neighborStatus) over each
    row's retained neighbors, and `knnLofValues` gains that output. The
    host repairs: own incomplete -> k-distance, density, and factor; a
    row reaching an incomplete row -> density and factor (its
    reachability read the unknown k-distance); a row reaching one of
    those -> factor only. Tie rules are unchanged, k-distances repair
    before densities and densities before factors, and the repaired
    NNDistance column joins the device writeback that keeps a retained
    region's bridges canonical.
- Evidence: On the 21,970,934-point fixture the gather kernel falls
  32.20 s -> 0.491 s (65x) byte-identically; the coplanar resident lane
  wall falls 50.7 s -> 23.9 s outside the harness. Even the 4M lanes
  drop to ~4.1-4.3 s wall — the D0069-D0073 4M kernel figures were
  themselves tail-inflated, which is why the D0073 series measured
  throughput-bound at 4M yet stayed flat at 22M. Unit differentials
  force the extreme: a one-shell budget routes essentially every row
  through the repair contracts and must reproduce upstream bit for bit
  (`PdgLofFilter.DeviceShellBudgetRepairsIncompleteRowsBitForBit`,
  `PdgApproximateCoplanarFilter.DeviceShellBudgetRepairsIncompleteRowsExactly`),
  and the LOF host/device mirrors agree on the new closure status. Full
  validation: the CUDA Release suite passes 474/474 with only corpus and
  fixture-gated skips; memcheck/initcheck/racecheck/synccheck are clean
  over the index, LOF, and coplanar tests; host Debug and leak-disabled
  ASan/UBSan trees pass 332/332; and the v2/v3/v6 22M gates pass
  byte-exactly, each faster than every prior recording (119.5 s,
  245.8 s, 358.5 s from 147.5-149.6 s, 268.8-271.5 s, 405.3-407.1 s).
  B0017 records the re-run resident B2 lanes.
- Consequences: Roadmap item 1 closes for the uniform-grid backend: the
  dense regime is FP64-pipe-shaped (D0073) and the tail regime is
  host-repaired (this entry). The BVH gather has no budget and keeps its
  bound-driven traversal. The placement time coefficients for the lof
  and neighborhood families now overestimate device cost at scale;
  recalibration is deferred to the roadmap's calibration item so this
  entry changes no frozen coefficient.

## D0075 — Catalog port, first tranche: the eigen family and nndistance go resident

- Date: 2026-08-09
- Status: accepted; roadmap item 2 opens with four stages ported strictly
  through the resident interface per the milestone's closing rule
- Context: `filters.normal`, `filters.eigenvalues`,
  `filters.covariancefeatures`, and `filters.nndistance` had exact hybrid
  CUDA envelopes and compiled payloads but no resident-planner admission:
  only the coplanar and LOF programs counted as whole-view neighborhood
  producers, and none of the four carried a measured placement model.
- Decision: The four programs join the delegated-region admission chain —
  the rewrite's producer/wrapper switches, the preflight payload branches
  (eigen family at the shared 146 B/point executor scratch; nndistance at
  9 B/point; the LOF stream count also gains its third status byte), the
  CLI's neighborhood declaration, and the wrapper filters' delegated
  branches mirroring `PdgLofFilter`. Compile-time option envelopes are
  unchanged: radius/refine/stride/optimal forms still fall back to
  upstream.
- Decision: Four measured placement models join the SM-89 profile (now 25
  models) from 24 clean forced hybrid-CUDA rows at 250k..16M points
  (`build/benchmarks/<stage>-cal-<size>-1ac89f53f.json`, every row
  byte-exact): normal `host 87701009 + 4470.24/pt` vs
  `device 190649278 + 886.50/pt`; eigenvalues `host 4447.60/pt` vs
  `device 149771691 + 879.12/pt`; covariancefeatures `host 4622.70/pt`
  vs `device 156004996 + 896.02/pt`; nndistance `host 4367.12/pt` vs
  `device 810.05/pt`; all with 250,000-point floors. The audit stands at
  88/88 with winner accuracy 1.000.
- Decision: The resident CLI's PDAL log leader becomes `pdal pipeline`:
  stage diagnostics are part of the byte-exact stderr contract, and the
  eigenvalues stage's upstream min_k warning exposed that a
  resident-selected execution printed a `pdg resident` leader where the
  oracle prints `pdal pipeline`. The wrapper stages already reproduce the
  warning text and stage name exactly; only the leader differed.
- Evidence: Four 22M resident lanes over the hash-pinned fixture, each
  with stats-proven `planner_resident_shared_index` execution and the
  `whole_view_neighborhood` schedule, byte-exact including stderr:
  normal 4.037x (three-run B2 protocol), eigenvalues 4.084x,
  covariancefeatures 3.955x, nndistance 4.356x (single-run checks);
  B0018 records the lanes. Full CUDA suite 474/474 (the placement pin
  moves to 25 models), host Debug and leak-disabled ASan/UBSan trees
  332/332, and the v2/v3/v6 22M gates re-run green after the log-leader
  change.
- Consequences: Every currently-kerneled shared-kNN family now executes
  under the resident planner. The remaining catalog breadth (spec §7
  stages without kernels: estimaterank, optimalneighborhood,
  neighborclassifier, and the grid/morphology families) needs new device
  primitives, not admission work. The placement models predate D0073/74's
  kernel wins on the hybrid path they were measured on; the deferred
  calibration item owns refreshing them alongside the shared-index
  executor calibration.

## D0076 — Fused-program JIT specializes every admitted LAS record format

- Date: 2026-08-09
- Status: accepted; roadmap item 3 closes for the fused-program JIT
- Context: D0071's NVRTC specialization admitted only identity records —
  format 7/8 at the canonical 36-byte stride — so every other admitted
  input format kept the interpreter for fused chains. The interpreter's
  `translatePointFields` already rebuilds the canonical record from any
  admitted format.
- Decision: `generateFusedKernelSource` takes the input format and stride
  and emits a two-region staged kernel for non-identity inputs: the
  source records stage at their stride ahead of the canonical output
  span, and a straight-line, constant-format specialization of
  `translatePointFields` rebuilds each record — legacy bit repacking
  including the classification-12 overlap move and the exact
  legacy-scan-angle rounding, per-format GPS/color placement, and
  modern-format copies — before the unchanged program body and pack
  statements run. Identity inputs keep the in-place single-span layout.
  Admission widens to every host-admitted format (0-3, 6-8; waveform
  formats stay outside the native envelope) with stride and
  shared-budget guards; format 8 at its native 38-byte stride becomes
  JIT-eligible for the first time. `PDG_DEBUG_FUSED_JIT` dumps the NVRTC
  log and generated source on a failed compilation.
- Evidence: `FusedJitSpecializesEveryAdmittedFormat` runs the fused
  matrix across formats 0-3/6/7/8 under `PDG_REQUIRE_FUSED_JIT` (an
  interpreter fallback is a hard failure) and matches the host oracle
  bit for bit; the pre-existing format matrix and the full CUDA suite
  pass 475/475 with memcheck and racecheck clean over the LAS tests.
  The 22M format-7 JIT-forced B2 lane re-runs byte-exact at
  17.529x (candidate 0.449-0.498 s, identity source kept minimal via conditional helper emission) against the 18.568x D0071 reference
  (`build/benchmarks/jit-assign-ferry-fused-22m-multiformat.json`,
  c10b30d48b4ea46023ca416aa38415ad5b8f52aef12fb8f0319bac48c8f98e96).
- Consequences: Fused chains reaching the direct native executor now
  specialize regardless of input record format. Endpoint-fusion
  admission itself (which pipelines reach the native executor) is
  unchanged; widening that envelope stays with the catalog work. The
  interpreter remains the always-available fallback and the JIT
  differential oracle.

## D0077 — The shared-index executor observes its transfers and matches its calibration

- Date: 2026-08-09
- Status: accepted; roadmap item 4 closes and with it every recorded
  post-P1.5 roadmap item
- Context: The shared-index neighborhood executor reported
  `selected_device_calibration_matches_executor: false` honestly since
  D0066/D0067: its models were measured on forced hybrid-CUDA lanes, not
  on the planner-selected resident executor, and its attach machinery's
  physical transfers were invisible to `--stats` (H2D/D2H totals held
  zero for whole-view runs). The six models also predated the D0073/74
  kernel rebuild and overestimated device cost at scale.
- Decision: The whole-view attach machinery records every physical
  transfer as HostToDevice/DeviceToHost observation events with the
  planner-owned region identifier (IndexBuild convention): the XYZ
  gather, output-column preparation, result and status returns for the
  eigen family, NNDistance, and LOF (all three status streams), the
  tie/incomplete repair writebacks in both directions, and the resident
  assignment bridge's uploads and spills.
- Decision: All six shared-index stage models are re-measured against the
  planner-selected resident executor itself: 36 clean ladders
  (`resident-cal-<family>-{250k..16m}.json`, every lane byte-exact with
  stats-proven `planner_resident_shared_index` execution, case status
  `clean-resident-shared-index`). The refreshed fits also capture the
  D0073/74 kernel wins (device slopes now 990–1087 ns/point across the
  family; LOF's host slope rises to its true 9527 ns/point). The audit
  stands at 87/87 with winner accuracy 1.000. A whole-view execution
  whose every applied region calibration is one of the six
  resident-provenance models now reports
  `selected_device_calibration_matches_executor: true`; the
  boundary_batch executor still has no calibration of its own and keeps
  reporting false honestly.
- Evidence: v2/v3/v6 22M gates pass byte-exactly with the flag asserted
  true for the shared-index executor; the full CUDA suite passes 475/475
  and both host trees 332/332; a 4M whole-view run reports the observed
  transfer totals (100 MB across four uploads, 8 MB across two returns
  for the coplanar family) where every prior recording showed zero.
- Consequences: All four recorded post-P1.5 roadmap items are closed
  (D0072–D0077, B0016–B0018). Remaining catalog breadth needs new device
  primitives (spec §7: estimaterank, optimalneighborhood,
  neighborclassifier, the grid/morphology families). The D4
  calibration-match story is now honest for every executor that has a
  calibration; only boundary_batch remains, deliberately, false.

## D0078 — filters.estimaterank goes native and resident

- Date: 2026-08-09
- Status: accepted; the first new-primitive catalog slice after the
  post-P1.5 roadmap lands end to end
- Context: filters.estimaterank (spec §7.5) had no PDG path at all:
  upstream computes, per point, the rank of the self-inclusive KD3
  neighborhood — an online double centroid, a float-cast demeaned matrix
  held in doubles, the sample covariance, and Eigen's JacobiSVD with the
  float-cast threshold (MathUtils computeRank).
- Decision: A `knnRankValues` primitive computes the covariances on the
  device through the exact shared-kNN machinery
  (`knnRankCovariancesDevice` lands them host-side) and decomposes on the
  host with Eigen's own JacobiSVD across worker threads — rows are
  independent, so the parallel pass is bit-identical, and host Eigen is
  the oracle's solver by identity; JacobiSVD carries no device
  annotations, so a device decomposition could not be oracle-exact.
  `computeRankExact` transcribes the unexported upstream chain verbatim
  and serves the wrapper's host fallback and the tie/incomplete repair.
- Decision: `PdgEstimateRankFilter` (`filters.pdg_estimaterank`) is the
  exact wrapper with the standard hybrid gating (knn 3..64) and the
  delegated resident branch; the hybrid rewrite qualifies
  `{knn, thresh}` option envelopes; the planner compiles an
  `EstimateRankProgram` on the measured `estimaterank` model
  (resident-executor provenance, ladder
  `resident-cal-estimaterank-{250k..16m}`, every lane byte-exact with
  stats-proven `planner_resident_shared_index` execution) and admits it
  through the whole-view region chain with the eigen-family scratch
  declaration. The SM-89 profile grows to 26 models; the audit stands at
  93/93 with winner accuracy 1.000.
- Evidence: Unit differentials pin bit-equality against upstream on the
  host path, the CUDA path with a forced tie-repair dataset, and a
  one-shell-budget run that routes every row through the repair; the
  resident emission test pins the wrapper rewrite; the full CUDA suite
  passes 479/479 and both host trees 333/333. The 22M resident lane is
  byte-exact at 4.155x over pinned PDAL (three-run B2 protocol) with the calibration-match flag true
  (`build/benchmarks/resident-estimaterank-22m.json`, 9ef79212e12981335e90cc17609cb669b8d7a7aac0ef92b59f65f6fd37e5fedd).
- Consequences: The §7.5 flagship family is fully native and resident
  except `filters.optimalneighborhood`, which is the next slice (its
  distance-sorted prefix-covariance sweep needs a new kernel shape).
  The host-side SVD finale is a measured candidate for future device
  work only if a bit-exact device JacobiSVD replica is proven.

## D0079 — filters.optimalneighborhood goes native and resident

- Date: 2026-08-09
- Status: accepted; the §7.5 flagship family is complete
- Context: filters.optimalneighborhood selects, per point, the k in
  [min_k, max_k] minimizing the neighborhood eigenentropy: one
  distance-sorted max_k KD3 query, a Welford incremental covariance
  sweep, an eigen solve per k, and a transcendental entropy comparison,
  writing OptimalKNN and OptimalRadius.
- Decision: `knnOptimalValues` computes the distance-sorted adjacency on
  the device through the shared-kNN machinery
  (`knnAdjacencyHostDevice` lands it host-side) and runs the sweep on
  host worker threads in upstream's exact accumulation order — the
  entropy's `std::log` keeps the transcendental comparison on the host
  per the established features precedent, and rows are independent so
  the worker split is bit-identical. `computeOptimalExact` transcribes
  the upstream per-point selection verbatim for the wrapper's host
  fallback and the tie/incomplete repair. The wrapper
  (`filters.pdg_optimalneighborhood`), hybrid qualification
  (`{min_k, max_k}`, 1 <= min_k <= max_k <= 64), planner
  `OptimalNeighborhoodProgram` on the measured `optimalneighborhood`
  model, and the whole-view resident admission (scratch
  max_k*12 + 17 B/point) land together. The model carries
  resident-executor provenance (`resident-cal-optimalneighborhood-*`
  ladders, every lane byte-exact with stats-proven
  `planner_resident_shared_index` execution) and joins the
  calibration-match set; the SM-89 profile holds 27 models with the
  audit at 99/99, winner accuracy 1.000.
- Evidence: Unit differentials pin bit-equality against upstream on the
  host path, the CUDA path with forced tie repair, and a
  one-shell-budget full-repair run; the region emission test pins the
  rewrite; the full CUDA suite passes 483/483 and both host trees
  334/334. The 22M resident lane is byte-exact at 5.062x over pinned
  PDAL (144.1 s -> 28.5 s, three-run B2 protocol,
  `build/benchmarks/resident-optimalneighborhood-22m.json`,
  d2f33a2cc846c0ee15eded085f0e2ddbb26e8203dc0c529641b3d7f39d7ddaf8).
- Consequences: Every §7.5 stage is native and resident. The remaining
  §7 breadth: neighborclassifier, radiusassign, label_duplicates (Ph 2);
  elm, hag family, overlay/colorization, and the grid/morphology
  ground-classification families (Ph 3). The per-k eigen solve is
  device-annotated Eigen, so a future device sweep with host-side
  entropy-only comparison is a measured optimization candidate.

## D0080 — filters.neighborclassifier goes native and resident

- Date: 2026-08-09
- Status: accepted; the §7.3 kNN vote joins the resident catalog
- Context: filters.neighborclassifier reclassifies each point by a
  majority vote over its k self-inclusive neighbors: an ordered count map
  over the neighbors' integer values, the first maximal element, a strict
  k/2 threshold, and a changed-only two-phase application that reads
  every vote against the original values. Its `domain`, `candidate`, and
  non-Classification `dimension` forms load other point sets or evaluate
  range predicates.
- Decision: The vote is a pure function of the neighbor set and integer
  values, so `knnNeighborVotes` runs it entirely on device: per query the
  kernel counts each candidate value across the row and keeps the
  smallest value among maximal counts, reproducing `std::map`'s ordered
  first-maximal selection without a map. Results stage separately from
  the source column so the two-phase semantics hold, and rows carrying a
  tie or incomplete status are repaired with `computeNeighborVoteExact`,
  the verbatim upstream transcription that the wrapper's host fallback
  also uses.
- Decision: `PdgNeighborClassifierFilter`
  (`filters.pdg_neighborclassifier`) accepts the
  `{k, dimension=Classification}` envelope with 1 <= k <= 64; `domain`,
  `candidate`, and other dimensions keep the pinned upstream stage. The
  planner compiles a `NeighborClassifierProgram` on the measured
  `neighborclassifier` model (resident-executor provenance,
  `resident-cal-neighborclassifier-{250k..16m}`, every lane byte-exact
  with stats-proven `planner_resident_shared_index` execution) and admits
  it through the whole-view region chain; the stage joins the
  calibration-match set. The SM-89 profile holds 28 models with the audit
  at 105/105, winner accuracy 1.000.
- Evidence: Unit differentials pin bit-equality against upstream on the
  host path, the CUDA path with forced tie repair, and a
  one-shell-budget full-repair run; the region emission test pins the
  rewrite; the full CUDA suite passes 487/487 and both host trees
  335/335. The 22M resident lane is byte-exact at 3.743x over pinned
  PDAL (86.2 s -> 23.0 s, three-run B2 protocol,
  `build/benchmarks/resident-neighborclassifier-22m.json`,
  512e871dd9c3298a714cda13d72a7ed369e03640fec713332fa5c226ada90ee6).
- Consequences: Eight neighborhood stages now execute resident with
  measured models. The remaining §7 breadth needs different machinery:
  radiusassign and label_duplicates (radius and sort primitives rather
  than kNN), then the Ph 3 grid/morphology ground-classification and
  raster families.

## D0081 — filters.radiusassign uses the shared radius index with an exact host assignment finale

- Date: 2026-08-09
- Status: accepted; the §7.6 radius-selection slice is native and resident
- Context: `filters.radiusassign` selects source rows having at least one
  independently domain-filtered reference point at strict distance less than
  `radius`. It supports 2D or 3D search, inclusive one-sided 2D Z caps, and an
  ordered list of upstream assignment expressions. The selection is
  data-parallel; the expression evaluator and PDAL physical casts are the
  compatibility oracle and are not CUDA-compatible as a whole.
- Decision: the planner-owned `SpatialIndex` gains `radiusAny`, which consumes
  source/reference masks and writes one match byte per query through either
  the uniform-grid or Morton-BVH backend. It preserves self-inclusion, strict
  radius comparison, 2D non-finite-Z behavior, and inclusive above/below caps;
  the stage never builds a private index. `filters.pdg_radiusassign` is the
  exact wrapper. Its resident path performs selection on CUDA, evaluates the
  original prepared `AssignStatement` objects in source and statement order
  on the host, and uploads every written non-coordinate column before a
  downstream resident bridge. Coordinate targets, unsupported options,
  malformed programs, and unavailable CUDA retain the untouched pinned-host
  path. The CUDA selection path is therefore implemented-device coverage, but
  the stage is not represented as wholly device-native.
- Decision: `RadiusAssignProgram` declares domain/update reads, dynamic writes,
  `IndexKind::Radius`, radius/dimensionality, order preservation, and XYZ
  invalidation. Resident regions reject mixed radius/kNN or mixed 2D/3D index
  requests. The SM-89 model is fitted as a planner residual from seven exact
  resident lanes, bounded to the measured 250,000–21,970,934-point envelope;
  the constrained 170 ms device intercept preserves every observed winner.
  The profile holds 29 models and the frozen placement audit is 112/112 at
  winner accuracy 1.000.
- Evidence: the exact process matrices pass 17 host cases and 13 CUDA-admitted
  cases, covering self-inclusion, duplicates, strict boundaries, 2D/3D,
  both Z caps and equality, open/negated and OR domains, conditionals, empty
  input, and malformed/fallback envelopes. Wrapper and shared-index unit
  differentials pass on both backends. V8 runs the 21,970,934-point resident
  stage followed by a device assign consuming the re-uploaded UserData column;
  it is byte-exact, reports one predicted/observed index build, and proves
  executor/calibration agreement. Host Debug and leak-disabled ASan/UBSan pass
  339/339; the serialized CUDA Release tree passes 497/497 (nine fixture-gated
  skips in the default aggregate). Focused Compute Sanitizer memcheck reports
  zero errors and racecheck zero hazards across all four radius tests. The B2
  lane is byte-exact at 5.503x over pinned PDAL
  (`build/benchmarks/resident-radiusassign-22m.json`,
  `55e6c109652f5910e2c2775082852a38fa8b800a1887f5564537fb125df2814b`;
  full provenance and the calibration ladder are B0019).
- Consequences: ten stages now execute under the measured resident shared-index
  family (nine kNN plus radiusassign). `filters.label_duplicates`, whose exact
  semantics need the stable-sort machinery rather than a neighborhood query,
  is the next Phase 2 catalog slice.

## D0082 — filters.label_duplicates is an exact adjacent CUDA pass and index-free resident producer

- Date: 2026-08-09
- Status: accepted; forced and resident CUDA are qualified, while automatic
  standalone placement remains disabled by the negative B0020 gate
- Context: contrary to the prior roadmap shorthand, upstream
  `filters.label_duplicates` does not sort. It writes `Duplicate` for every row
  after the first by comparing each selected field, converted to binary64, with
  the immediate predecessor in the current order. The first byte is untouched;
  an empty dimension list is vacuously equal; NaNs, signed zero, and wide
  integers follow ordinary binary64 comparison. Selecting `Duplicate` itself
  is sequential because earlier writes feed later comparisons.
- Decision: add an exact typed host/CUDA `LabelDuplicatesProgram` and the
  internal `filters.pdg_label_duplicates` wrapper. The descriptor is global,
  order- and cardinality-preserving, reads selected fields plus the initial
  `Duplicate` byte, writes unsigned-byte `Duplicate`, and requests
  `IndexKind::None`. Unsupported options, missing dimensions, conditional
  execution, self-referential output, unavailable CUDA, and unproven graph
  forms retain the exact host or untouched upstream path. The CUDA operation
  initializes rows one through the end and clears mismatches once per selected
  physical column; coordinate values use the checked logical decode.
- Decision: a selected resident region may create and retain a whole-view
  device product without gathering XYZ or building an index. `Duplicate` may
  feed a later point-program bridge directly. If a following neighborhood
  stage needs an index, the region builds exactly one planner-owned index at
  that later envelope; the label stage never constructs a private index. No
  placement model is added: exact complete-process measurements are only
  0.260x pinned PDAL at 250K, 0.537x at 1M, and 0.959x at 21,970,934 points.
- Evidence: host units cover adjacency, binary64 wide-integer collisions,
  NaNs, signed zero, empty dimensions, first-row preservation, errors, and
  self-output rejection. The physical CUDA property covers all ten PDAL
  physical types and encoded coordinates. Upstream-wrapper tests cover host,
  CUDA, and self-output fallback; resident tests prove the no-index product can
  feed an assignment bridge, can precede a planner-owned kNN index, and keeps
  both empty compositions as exact no-ops with normal terminal closure. The
  process matrix is exact for 15 host and 10 CUDA-admitted cases. Host Debug
  and leak-disabled ASan/UBSan each pass 348/348; physical SM-89 CUDA Release
  passes 513/513 with nine documented opt-in fixture skips. Focused memcheck,
  initcheck, racecheck, and synccheck report zero findings across the primitive,
  wrapper, and resident composition. The 1M Nsight profile records a 7.36 us
  comparison kernel at 65.93% SM throughput, 14.00% DRAM throughput, and
  75.75% achieved occupancy. B0020 contains raw-report hashes and full
  provenance.
- Consequences: the resident catalog gains one exact non-index whole-view
  producer without changing the ten-stage measured shared-spatial family or
  the seven automatically selected filters. Implemented-device filter coverage
  awaiting an automatic or resident-plan performance gate increases by one.
  The next catalog slice is the Phase 3 ground-classification family, starting
  with `filters.smrf`; unresolved P2 engine exits and other catalog stages
  remain open and are not implied complete by this decision.

## D0083 — filters.smrf starts P3 with a bounded exact global-canvas CUDA lane

- Date: 2026-08-09
- Status: accepted for forced and standalone planner-selected execution;
  automatic placement and composable grid residency remain disabled by the
  negative B0021 gate and the missing scalable workspace
- Context: upstream SMRF constructs a column-major minimum-Z raster, fills
  voids from the ordered eight nearest valid cells, applies progressive
  diamond morphology and optional net cutting, inpaints a provisional surface,
  and classifies selected returns from its finite-difference gradient. Its
  canvas size is a runtime property of point bounds and `cell`, not a fact the
  static pipeline compiler can prove. The existing planner spatial index is
  neither the upstream raster nor a substitute for its exact inpainting order.
- Decision: add an exact typed host reference, a precise-FP64 CUDA
  implementation, and the internal `filters.pdg_smrf` wrapper. The first
  device envelope is deliberately bounded to finite logical-double XYZ,
  unsigned-byte Classification, at least two rows and columns, at most 4,096
  raster cells, morphology radius at most 64, and the upstream-supported
  return/class options. Equal cell minima retain the earliest source point's
  exact bits after the parallel numeric minimum. Raster addressing is checked
  before every device write, and an invalid frame fails closed. `dir`,
  `ignore`, `classbits`, `where`, unsupported return forms, nonfinite values,
  degenerate canvases, and larger workspaces retain unchanged PDAL or the exact
  host wrapper.
- Decision: declare SMRF as a whole-view, order/cardinality-preserving grid
  stage that reads XYZ, return fields, and Classification and writes
  Classification without requesting a spatial index. Its descriptor has no
  placement model. Forced/experimental selection rechecks the runtime canvas;
  normal unsupported executions fall back exactly, while a required CUDA
  execution fails rather than silently changing placement. The resident
  rewriter admits only a standalone upload-stage-spill region and rejects an
  adjacent grid bridge explicitly. This is planner-selected bounded execution,
  not a claim that the grid workspace or output is yet reusable across stages.
- Evidence: the deterministic process matrix passes 15 host cases and seven
  forced-CUDA cases, including defaults, return subsets, custom class values,
  `only_ground`, net cutting, missing/all-zero/partially-zero return fields,
  empty selections, malformed options, and conservative fallback. Host,
  compiler, rewrite, primitive, and physical resident-lifecycle units pass.
  Host Debug and leak-disabled ASan/UBSan each record 355 passes plus one
  expected skip in 356 registrations; physical SM-89 CUDA Release records 514
  passes plus nine documented skips in 523 registrations.
  Physical SM-89 memcheck, racecheck, initcheck, and synccheck report zero
  findings. B0021 is byte-identical at 1,000,000 points but measures only
  0.599x pinned PDAL; its Nsight profile records 191 one-block morphology
  launches averaging 2.16 microseconds and a 36.26-microsecond final classifier.
- Consequences: P3 has its first exact device implementation, but no P3 exit is
  closed. Automatic SMRF selection remains off. A scalable planner-owned grid,
  tiled halo/seam proof, composable raster residency, PMF/CSF, and the remaining
  terrain/raster catalog are still required.

## D0084 — filters.pmf gains a bounded exact global-canvas CUDA lane

- Date: 2026-08-09
- Status: accepted for forced and standalone planner-selected execution;
  automatic placement and composable grid residency remain disabled by the
  negative B0022 gate and the missing scalable workspace
- Context: upstream PMF rasterizes selected returns, fills voids from a
  one-nearest two-dimensional cell index, then applies a progressive sequence
  of morphological openings and strict height comparisons. Compatibility has
  two non-obvious constraints: the initial raster computes
  `floor(coordinate - minimum) / cell_size`, while the final point lookup uses
  `floor((coordinate - minimum) / cell_size)`; and equal-distance empty cells
  retain upstream's deterministic first cell. Reassociating either expression
  or replacing the fill with a mathematically equivalent traversal changes
  exact classification for boundary fixtures.
- Decision: add an exact typed host reference, a precise-FP64 CUDA
  implementation, and the internal `filters.pdg_pmf` wrapper. The first device
  envelope admits finite logical-double XYZ, unsigned-byte Classification, at
  most 4,096 raster cells, morphology radius at most 64, at most 64 schedule
  passes, and upstream-supported return/class options. Parallel numeric minima
  are repaired to the earliest equal source before the raster is published;
  void fill is an exhaustive deterministic nearest search, and all initial and
  final raster addresses are validated before use. `ignore`, `where`, invalid
  options, nonfinite values, and larger workspaces retain unchanged PDAL or the
  exact host wrapper.
- Decision: declare PMF as a whole-view, order/cardinality-preserving Grid
  stage that reads XYZ, return fields, and Classification and writes
  Classification without requesting a spatial index. Its descriptor has no
  placement model. Forced/experimental selection rechecks the runtime canvas;
  required CUDA fails closed when the bounded lane cannot execute. The
  resident rewriter admits only a standalone upload-stage-spill region and
  rejects an adjacent grid bridge. This is a planner-selected execution
  boundary, not a reusable planner-owned grid product.
- Evidence: the deterministic process matrix passes 16 host cases and nine
  forced-CUDA cases, including default and explicit return selection, custom
  exponential and linear schedules, fractional-cell binning, custom classes,
  `only_ground`, missing/all-zero/partially-zero return fields, empty input,
  malformed options, and conservative `ignore`/`where` fallback. Host/device
  primitives, compiler/rewrite contracts, and the actual physical standalone
  resident lifecycle pass. Host Debug and leak-disabled ASan/UBSan each record
  363 passes plus one expected skip in 364 registrations; physical SM-89 CUDA
  Release records 524 passes plus nine documented skips in 533 registrations.
  Physical SM-89 memcheck, racecheck, initcheck, and synccheck report zero
  findings. B0022 is byte-identical at 1,000,000 points but measures only
  0.583x pinned PDAL; its Nsight profile records 74 one-block morphology
  launches averaging 2.19 microseconds, while the point-parallel raster and
  classification kernels remain sub-150-microsecond work.
- Consequences: PMF gains a bounded exact device implementation without
  closing the PMF catalog item or any P3 exit. Automatic selection remains off.
  CSF is the next bounded terrain slice; a scalable planner-owned grid, tiled
  halo/seam and mosaic proof, composable raster residency, and the remaining
  terrain/raster catalog are still required.

## D0085 — filters.csf gains a bounded exact serial-oracle CUDA lane

- Date: 2026-08-10
- Status: accepted for forced and standalone planner-selected execution when
  the build proves the pinned oracle's serial schedule; automatic placement,
  smoothing, and composable Grid residency remain disabled by B0023 and the
  missing scalable workspace
- Context: upstream CSF transforms `(X, Y, Z)` to `(X, -Z, Y)`, constructs its
  cloth graph in column/row source order, rasterizes to the first strictly
  nearer sample, fills voids with ordered scans followed by a graph BFS, and
  performs in-place constraint relaxation before collision and strict bilinear
  classification. The constraint loop writes each particle and its neighbors.
  It is serial in the pinned build, but an OpenMP-enabled build has a racy,
  schedule-dependent oracle that this exact lane cannot claim to reproduce.
  The default `smooth=true` path also performs a separate topology-changing
  postprocess and remains upstream work.
- Decision: add an exact typed host reference, a precise-FP64 CUDA
  implementation, and the internal `filters.pdg_csf` wrapper. Native admission
  requires the explicit serial-oracle capability, `smooth=false`, finite
  logical-double XYZ and scalar options, unsigned-byte Classification, positive
  resolution, nonnegative rigidness, zero to 64 iterations, an at-least-2x2
  cloth with at most 4,096 cells, and the upstream-supported return/class
  options. The frame, raster/fill, and simulation kernels deliberately execute
  one thread to preserve source order; final classification is grid-stride.
  Raster rejection is synchronized before simulation, and rejected direct
  device input cannot publish Classification. `ignore`, `where`, `debug`,
  `dir`, smoothing, OpenMP-enabled builds, invalid options, nonfinite values,
  and larger cloths retain unchanged PDAL or the exact host wrapper.
- Decision: declare CSF as a whole-view, order/cardinality-preserving Grid
  stage that reads XYZ, return fields, and Classification and writes
  Classification without requesting or constructing a spatial index. Its
  descriptor has no placement model. The resident rewriter admits only a
  standalone upload-stage-spill region, closes empty delegated regions in
  `prerun`, and rejects an adjacent Grid bridge. This is a bounded execution
  boundary, not a reusable planner-owned cloth or Grid product.
- Evidence: the deterministic process matrix passes 22 host cases and 14
  forced-CUDA cases, including verbose diagnostics, empty execution,
  `returns: []`, partial and missing return dimensions, iteration and
  fractional-resolution boundaries, custom classes, malformed options, and
  conservative fallback. Eight focused Host Debug/ASan tests, 12 focused
  physical CUDA tests, five repeated CUDA matrices, and the actual standalone
  and empty resident lifecycles pass. Host Debug and leak-disabled ASan/UBSan
  each record 371 passes plus one expected skip in 372 registrations; physical
  SM-89 CUDA Release records 536 passes plus nine documented skips in 545
  registrations. Physical SM-89 memcheck reports zero
  bytes leaked and zero errors; initcheck and synccheck report zero errors;
  racecheck reports zero hazards. B0023 is byte-identical at 1,000,000 points
  but measures only 0.289x pinned PDAL. Its profile shows that the exact serial
  frame and raster/fill kernels take 218.69 ms and 361.55 ms, while the
  point-parallel classifier takes 58.88 microseconds.
- Consequences: CSF gains a bounded exact device implementation without
  closing the CSF catalog item or any P3 exit. Automatic selection remains off.
  A scalable/tiled planner-owned Grid/cloth product with halo, seam, and mosaic
  proof is now the next terrain foundation; ELM, skewness balancing, DEM/HAG,
  and the rest of the phase remain open.

## D0086 — filters.elm gains a bounded exact stable-cell CUDA lane

- Date: 2026-08-10
- Status: accepted for forced and standalone planner-selected execution;
  automatic placement and composable Grid residency remain disabled by B0024
  and the missing scalable workspace
- Context: upstream ELM builds a whole-view frame, assigns points with the
  non-reassociated expression `floor(coordinate - minimum) / cell`, and stores
  each cell in column-major order. Within a cell it walks numeric Z order,
  retaining source order for equivalent values, and marks each prefix point
  until the strict test `fabs(current - next) < threshold` succeeds. The
  exact contract therefore includes the unusual fractional-cell formula,
  stable equal-Z and signed-zero behavior, strict threshold boundaries,
  original point order, and the stage's info diagnostic.
- Decision: add an exact typed host reference, a precise-FP64 CUDA
  implementation, and the internal `filters.pdg_elm` wrapper. Native admission
  requires finite logical-double XYZ, unsigned-byte Classification, finite
  positive `cell`, finite threshold, a valid unsigned-byte class, no `where`,
  no more than `INT_MAX` points, and a runtime frame of at most 4,096 cells.
  CUDA performs a stable radix sort by canonicalized Z and then a stable radix
  sort by column-major cell key, preserving source order for equal numeric Z
  while never changing stored Z bits. One thread walks each cell's prefix in
  oracle order. Nonfinite and oversized input are rejected before
  Classification publication; unsupported forms retain unchanged PDAL or the
  exact host wrapper.
- Decision: declare ELM as a whole-view, order/cardinality-preserving Grid
  stage that reads XYZ and Classification and writes Classification without
  requesting or constructing a spatial index. Its descriptor has no placement
  model. The resident rewriter admits only a standalone upload-stage-spill
  region, closes an empty delegated region without allocation, and rejects an
  adjacent Grid bridge. Scratch preflight queries the active toolkit's CUB
  reduction and both radix-sort temporary sizes, accounts for the maximum
  overlapping allocation phase plus allocator slack, and fails at one byte
  below the derived budget. This is a bounded execution boundary, not a
  reusable planner-owned Grid product.
- Evidence: the deterministic process matrix passes 15 host and 10 forced-CUDA
  cases covering defaults and diagnostics, empty/one-point input, a custom
  class, fractional-cell binning, source-order equal-Z ties with signed zeros,
  strict and near-strict threshold boundaries, zero/negative thresholds,
  malformed options, `where` fallback, and nonfinite host fallback. Eleven
  focused Host Debug/ASan tests and 17 focused physical CUDA tests cover the
  primitive, compiler/rewrite contract, standalone and empty lifecycles,
  runtime rejection before publication, direct-device mutation safety, and
  the exact 1M scratch budget. Five consecutive CUDA matrices are byte-exact.
  Host Debug and leak-disabled ASan/UBSan each record 382 passes plus one
  expected skip in 383 registrations; physical SM-89 CUDA Release records 553
  passes plus nine documented skips in 562 registrations. Physical SM-89
  memcheck reports zero bytes leaked and zero errors; initcheck and synccheck
  report zero errors; racecheck reports zero hazards. The frozen placement
  audit remains 112/112 because B0024 adds no model.
- Evidence: B0024 is byte-identical at 1,000,000 points but measures only
  0.866x pinned PDAL. Its Nsight report records 470.78 microseconds total
  kernel time, including 254.51 microseconds in radix sorting, so the roughly
  74-millisecond median complete-process deficit is outside the kernel chain.
- Consequences: ELM gains a bounded exact device implementation without
  closing the ELM catalog item or any P3 exit. Automatic selection remains
  off. A scalable/tiled planner-owned Grid product with halo, seam, and mosaic
  proof remains the terrain foundation; skewness balancing, DEM/HAG, and the
  rest of the phase remain open.

## D0087 — filters.skewnessbalancing gains a bounded exact CUDA ordering lane

- Date: 2026-08-10
- Status: accepted for forced experimental hybrid execution only; automatic
  placement and resident/composable execution remain disabled by the
  data-dependent exactness envelope and the host recurrence/permutation
  boundary
- Context: upstream first applies `PointView::sort(Z)`, an unstable
  `std::sort`, to the complete point record and then evaluates the Bartels/Wei
  online moments and sign crossings in strictly sequential order. Comparator-
  equivalent Z values, including `-0.0` and `+0.0`, do not define a portable
  source permutation, while changing any recurrence order can change a
  crossing. A broadly admitted parallel sort or reduction therefore cannot
  satisfy the pinned-oracle contract.
- Decision: add an exact typed host recurrence and the internal
  `filters.pdg_skewnessbalancing` wrapper. Forced CUDA is admitted only for a
  nonempty whole view with physical binary64 Z, finite logical values, no more
  than `INT_MAX` points, and no comparator-equivalent keys. The existing exact
  device ordering primitive returns a full-record permutation and its adjacent
  equivalence flag. The wrapper mutates no point until that proof succeeds,
  applies the permutation to every field, and performs the original sequential
  moment/sign-crossing recurrence and Classification writes on the host.
  Ties, signed-zero equivalents, nonfinite values, other physical Z types,
  `where`, multi-view or order-unsafe graphs, and unsupported options retain
  unchanged PDAL or the exact host wrapper; required CUDA fails before
  publication.
- Decision: describe skewness balancing as a Global ordering boundary that
  reads Z and Classification, writes Classification, changes point order, and
  requests no spatial index. The conservative fallback compiler treats the
  complete record as touched because upstream reorders every dimension. The
  experimental hybrid rewriter may select the standalone wrapper, but the
  planner emits no device placement model or resident region. This is a CUDA
  ordering substep, not a wholly device-native filter.
- Evidence: the deterministic process matrix passes 20 host cases and seven
  forced-CUDA cases, covering defaults and verbose diagnostics, empty/one/two
  points, unique and tied orderings, signed zeros, true and near crossings,
  constant/nonfinite Z, custom classes, `only_ground`, malformed options, and
  `where` fallback, plus multi-view and unproven-reader rewrite fallback. A
  process guard proves empty required-CUDA execution fails rather than
  bypassing the nonempty admission envelope. Five core, one compiler, one
  planner, and two physical wrapper tests cover the recurrence, fallback
  gates, full-record
  permutation, classification, and rejection-before-mutation contract. Five
  consecutive physical CUDA matrices are byte-exact. Host Debug and
  leak-disabled ASan/UBSan each record 390 passes plus one expected skip in
  391 registrations; physical SM-89 CUDA Release records 564 passes plus nine
  documented skips in 573 registrations.
  Memcheck reports zero bytes leaked and zero errors; initcheck and synccheck
  report zero errors; racecheck reports zero hazards. The frozen placement
  audit remains 112/112 because this slice adds no model.
- Evidence: B0025 is byte-identical at 1,000,000 comparator-unique points and
  measures 1.150x pinned PDAL. Its Nsight report records about 255.07
  microseconds over 13 kernel launches, including 202.11 microseconds in the
  eight radix-sort passes. The positive controlled result does not establish
  an automatic production gate for ordinary point clouds with tied Z values
  or eliminate the host recurrence and PointView publication boundary.
- Consequences: skewness balancing gains a bounded exact device ordering
  implementation without closing the stage catalog item or any P3 exit.
  Automatic selection and resident composition remain off. A scalable/tiled
  planner-owned Grid product with halo, seam, and mosaic proof remains the
  terrain foundation; the HAG family and the rest of the phase remain open.

## D0088 — Add a provisional planner-owned tiled RasterGrid with PMF as its first consumer

- Status: accepted as correctness infrastructure; rejected as a scalable,
  composable, or automatically selected PMF implementation.
- Context: D0084 proved the bounded PMF arithmetic but capped its single
  device canvas at 4,096 cells. P3 needs a planner-owned product that can make
  memory-driven core/halo tiles, preserve phase barriers between morphology
  passes, and mosaic owner cells exactly. PMF also exposes an oracle surface
  that a geometric nearest-cell rule cannot settle generally: upstream's
  nanoflann visit order can choose among equal-distance void-fill cells. A
  distinct-valued tie therefore cannot be accepted merely by selecting the
  lower cell id.
- Decision: add a `GridRequest` to stage descriptors and a region-owned
  `RasterGridProduct` with an explicit PMF frame policy, deterministic
  column-major core ownership, a clipped one-cell halo, owner-only mosaic,
  two complete pinned-host backings, one expanded-tile scratch allocation,
  and a one-lane phase-synchronized schedule derived from the declared device
  budget. Preflight accounts cumulatively for the selected XYZ/Classification
  staging columns, fixed device scratch, both global host backings, and tile
  scratch before product allocation. The product is created only inside the
  selected PMF region and is released at its matching spill.
- Decision: the tiled PMF lane preserves upstream's distinct initial
  `floor(coordinate - minimum) / cell_size` expression and final
  `floor((coordinate - minimum) / cell_size)` lookup, exact cell-minimum bits,
  complete-pass barriers, strict thresholds, return selection, class writes,
  point order, and `only_ground`. Equal-distance void-fill candidates with
  distinct minimum bits reject required/resident execution before
  Classification publication. Non-required execution, including the broad
  experimental toggle, delegates directly to the original `filters.pmf` so
  option errors, warnings, and device/runtime rejection cannot duplicate or
  replace upstream diagnostics. Explicit required and planner-resident paths
  remain fail-closed.
- Decision: keep the PMF placement model empty, reject adjacent Grid bridges,
  and describe this as a single-region tiled correctness lane. It does not
  make the product reusable across stages, qualify arbitrary void-fill ties,
  eliminate full host-sized backings, or establish a scalable raster build.
- Evidence: five RasterGrid units cover deterministic ownership, all
  edges/corners, whole-grid stencil equivalence, invalid budgets, product
  lifetime, and a one-byte cumulative pinned-host budget boundary. Five PMF
  core units cover the enlarged frame and rejection-before-mutation contract.
  Seven physical wrapper tests cover standalone lifecycle, multi-tile seams,
  exhausted budgets, ambiguous void ties, product release between regions,
  an 81x81 fractional/return/`only_ground` oracle comparison repeated five
  times, and a nontrivial 65x65 morphology comparison. The existing 16-case
  host and nine-case forced-CUDA process matrices remain exact.
- Evidence: Host Debug records 397 passes plus one expected corpus skip in 398
  registrations. Leak-disabled ASan/UBSan records the same 397 plus one skip.
  Physical SM-89 CUDA Release completes 586 registrations with no failures,
  including the new lanes and nine documented fixture-gated skips. Focused
  memcheck reports zero leaked bytes and zero errors; initcheck and synccheck
  report zero errors; racecheck reports zero hazards across the ambiguity and
  two seam/oracle tests. An independent integration review found no remaining
  exactness, lifecycle, diagnostics, or cumulative-budget blocker.
- Evidence: B0026 is snapshot-exact but measures only 0.026602x pinned PDAL on
  its deterministic 4,225-point, four-tile comparison. Nsight Compute records
  39 launches; four all-points-per-tile raster launches total 94.09 ms under
  profiling and dominate the work, while their grids use only 1-11 blocks.
- Consequences: P3 gains the first tested tiled Grid product and exact PMF
  halo/seam mosaic, but neither the PMF catalog item nor a P3 exit closes. The
  next Grid foundation is a planner-owned point-to-cell/raster build that does
  not scan every point for every tile, followed by device-resident phase
  backings and measured cross-stage reuse. SMRF/ELM/CSF remain on their bounded
  products until their distinct oracle and workspace contracts are integrated.

## D0089 — Build the exact PMF source raster once before tiled morphology

- Status: accepted as an incremental Grid foundation; rejected as a scalable,
  wholly device-native, composable, or automatically selected PMF lane.
- Context: B0026 showed that D0088's four `filledRasterTileKernel` launches
  dominated its 65x65 prototype because every expanded tile cell rescanned
  every input point and repeated the nearest-fill proof. The product already
  owned two complete, cumulatively budgeted pinned-host `double` backings, so
  a separate source list or validity allocation would have violated its
  memory contract.
- Decision: build sparse cell minima once in selected source order into the
  current product backing, preserving upstream's initial
  `floor(coordinate - minimum) / cell_size` expression and the first exact
  bit pattern among equal numeric minima. Fill the second backing by scanning
  populated column-major cells with the same center arithmetic; accept an
  equal-distance set only when every candidate Z has identical binary64 bits,
  otherwise reject before publishing the raster or allocating CUDA output
  state. The build uses no allocation beyond the product's two planned
  backings, swaps only after every cell is complete, and publishes one
  producer generation for the tiled CUDA consumer to claim. A distinct
  `RasterBuild` observation records the completed backing bytes once per
  resident execution.
- Decision: remove the tiled all-points raster kernel but leave D0088's tiled
  CUDA morphology, owner-only mosaics, phase barriers, and memory schedule
  unchanged. A resident completion guard closes the delegated region on both
  success and exception, so product/budget/input rejection can be followed by
  the planned spill instead of stranding the context in `RegionActive`.
  Default and broad experimental PMF execution still delegate unchanged to
  pinned PDAL; adjacent Grid bridges and automatic placement remain off.
- Evidence: four new host units prove one published build independent of tile
  count, reject a second unconsumed build, retain the first signed-zero
  minimum, accept identical-bit nearest ties, and reject distinct-bit ties
  before publication. The physical resident seam test observes exactly one
  `GridBuild` and one `RasterBuild`; the exhausted-budget test now completes
  its spill after the required failure. The existing two >4,096-cell oracle
  differentials and all seven physical PMF wrapper tests remain exact.
  Host Debug and leak-disabled ASan/UBSan each pass 401 tests with one expected
  corpus skip in 402 registrations. Physical SM-89 CUDA Release passes 581
  tests with nine documented skips in 590 registrations. Memcheck, initcheck,
  racecheck, and synccheck report zero errors/hazards on the enlarged-frame
  and nontrivial-seam lanes.
- Evidence: B0027's exact dense four-tile fixture measures 1.828354x pinned
  PDAL with the raster build included in candidate wall time. Its complete
  Nsight Compute report records 71 launches and about 179.94 microseconds of
  kernel time; the removed all-points raster kernel is absent. The remaining
  56 morphology and 12 ground-filter launches are individually only about
  2.06-4.13 microseconds. End-to-end CUDA time is now dominated by the serial
  per-tile gather/copy/synchronize/mosaic phases rather than source-raster
  construction.
- Consequences: the source raster is constructed once per execution and no
  longer scales with tile count, but its allocation-free host void fill is
  still proportional to void cells times populated cells and B0027's dense
  fixture contains no voids. This is therefore neither a scalable sparse-fill
  result nor complete native PMF coverage. Device-resident phase backings,
  an exact bounded accelerated sparse fill, measured cross-stage Grid reuse,
  the HAG family, and the P3 exit remain open.

## D0090 — Retain fitting PMF morphology phases in planner-owned device backings

- Date: 2026-08-10
- Status: accepted as an incremental Grid foundation for required and
  planner-selected PMF execution; rejected as universal device residency,
  scalable sparse fill, cross-stage composition, or an automatic placement
  gate.
- Context: D0089/B0027 removed repeated point scans but left every morphology
  iteration on a serial host-tile gather, H2D, device stencil, D2H,
  synchronization, and owner mosaic path. The planner's 16 device bytes per
  expanded cell already equal two complete binary64 phase backings when a
  frame fits in one tile. Allocating those backings before the host nearest-tie
  proof, or letting a product publish another generation after consumption,
  would weaken the fail-closed exactness and lifetime contract.
- Decision: declare exactly two device phase backings in PMF's `GridRequest`.
  `RasterGridProduct` offers the resident pair only when the complete frame and
  both backings fit the existing runtime Grid budget; one byte below that
  boundary retains the exact host-tiled schedule. Allocation is delayed until
  one and only one host raster generation is published and pending. The
  product itself rejects materialization before publication or after
  consumption, a second publication even after consumption, and consumption
  without publication. The wrapper therefore completes minimum/nearest-tie
  proof before allocating the phase pair or a CUDA output batch.
- Decision: upload the completed host raster once, run every erosion,
  dilation, and ground-filter pass against the two full-frame device backings,
  and publish only the Classification column. No phase surface returns to the
  host in this lane. Frames that do not fit both complete backings retain
  D0088/D0089's exact halo/gather/mosaic fallback. Transfer facts are
  saturating diagnostics and are also exposed as one aggregate `RasterUpload`
  or `RasterDownload` observation per successful execution.
- Decision: reuse the already budgeted host tile scratch as a column-major
  populated-cell source list during raster construction. When every populated
  cell fits, void fill visits only those cells in the same source order and
  preserves identical arithmetic, first-bit minima, and distinct-bit tie
  rejection. If the list does not fit, the prior complete column-major scan
  remains the exact fallback. This adds no allocation and does not change the
  remaining worst-case `void_cells * populated_cells` distance work.
- Evidence: the exact one-byte device-budget unit proves no allocation below
  the boundary, no allocation before publication, one exact-sized allocation
  after publication, one-shot publication/consumption, and rejection after
  consumption. Host units retain signed-zero and same-bit ties, reject
  distinct-bit ties before publication, and prove the sparse source-list visit
  count. Physical wrapper differentials cover sparse and dense 65x65 frames,
  an 81x81 fractional/return/`only_ground` frame repeated five times, the
  multi-tile fallback, and ambiguous rejection before `RasterBuild` or
  `RasterUpload`; all are byte-exact with pinned PDAL.
- Evidence: Host Debug and leak-disabled ASan/UBSan each execute 402 passing
  tests plus one expected opt-in corpus skip in 403 registrations. Physical
  SM-89 CUDA Release executes 582 passing tests plus nine documented
  fixture/profile-gated skips in 591 registrations. Focused physical memcheck,
  initcheck, and synccheck report zero errors, and racecheck reports zero
  hazards across full-device, host-tiled, and ambiguous-rejection paths.
- Evidence: B0028 measures 9.009503x pinned PDAL on the exact dense 4,225-point
  frame and 4.764988x on the exact 25-point sparse/void frame, including host
  raster construction and delayed device allocation. Both record one
  33,800-byte raster H2D and zero raster D2H transfers. The final Nsight
  Compute report contains 20 launches per candidate: one validation, one
  initialization, 14 full-frame morphology, three full-frame filters, and one
  classifier. No tiled phase kernel appears. `morphKernel` averages 2.22
  microseconds, 1.22% SM throughput, and 15.86% achieved occupancy; the tiny
  controlled frame is launch/underutilization limited. Nsight Systems is not
  installed on the workstation, so transfer elimination is established by
  the exact execution facts and CUDA-event timings rather than a Systems
  trace.
- Consequences: a fitting required PMF region now retains all morphology phase
  surfaces on the device, while larger frames retain exact host tiling. This
  does not make PMF wholly device-native: source raster construction and
  nearest fill remain on the host, the exact sparse fill is still quadratic in
  its worst case, and no Grid bridge or cross-stage product reuse is admitted.
  Automatic placement, catalog closure, the HAG family, and the P3 exit remain
  open.

## D0091 — Add an exact occupancy hierarchy for controlled sparse PMF fill

- Date: 2026-08-10
- Status: accepted as an exact, allocation-free improvement for measured
  sparse distributions; rejected as a worst-case scalability proof,
  device-native source build, composable Grid, or automatic placement gate.
- Context: D0090/B0028 retained fitting morphology surfaces on device, but its
  host nearest fill still compared every void with every populated raster
  cell. A first 64-source hierarchy selector removed leaf comparisons on the
  measured fixture but caused a 7.660352-to-24.807174 ms raster-build cliff
  between 64 and 65 sources. The exactness contract also requires literal
  binary64 center arithmetic and observation of every equal-distance source;
  a conventional approximate nearest structure or an arbitrary cell-id tie
  rule is not admissible.
- Decision: use bytes in the planner-owned second host backing as a transient
  byte occupancy pyramid while filling the first backing in place. Original
  populated cells alone seed the pyramid, so completed voids never become
  sources. Greedy descent supplies an incumbent and an exact depth-first proof
  prunes a node only when its literal represented-center lower bound is
  strictly greater than the incumbent. Every equal-distance leaf remains
  visible: identical source bits are accepted and distinct bits reject before
  `publishRasterBuild()`. The product allocates no new persistent or
  per-raster storage.
- Decision: retain the simpler literal column-major source scan through a
  fixed 255-source cap and enter the hierarchy at 256 sources. B0029 measures
  31.239504 ms of raster construction at 255 sources versus 28.410521 ms at
  256, avoiding the original selector cliff. This cap makes the literal
  branch linear in raster cells, but the hierarchy itself remains
  branch-and-bound and has no proved worst-case subquadratic bound. Claims are
  therefore limited to the named measured distributions.
- Decision: native tiled admission now rejects a frame when either endpoint
  center on either axis is nonfinite, even when the input coordinates and cell
  size themselves are finite. This occurs before product build/publication and
  preserves default upstream fallback. The benchmark reads Classification back
  from the candidate, labels XYZ as immutable staging inputs and return fields
  as fixture invariants, and uses heterogeneous checkerboard elevations whose
  representable equal-distance ties necessarily share bits. Raw binary64
  raster parity is established by the independent literal-scan unit.
- Evidence: 14 PMF core tests include the finite-input/nonfinite-center gate,
  a 256-source heterogeneous large-origin/fractional-cell comparison against a
  literal all-source scan, controlled hierarchy work counters, four-way
  same/distinct-bit ties, and a successful exact reattempt on the same
  unpublished product after ambiguity. The focused host suite passes 14/14 and
  the focused CUDA/differential suite passes 27/27. Full Host Debug passes 407
  tests plus one opt-in corpus skip in 408 registrations; leak-disabled
  ASan/UBSan passes 324 plus one skip in 325. Physical SM-89 CUDA Release
  passes 587 tests plus nine documented skips in 596 registrations.
  Memcheck reports zero leaked bytes and zero errors, initcheck and synccheck
  report zero errors, and racecheck reports zero hazards over the fitting,
  host-tiled, and ambiguity paths.
- Evidence: B0029's classification-exact heterogeneous 257x257 and 513x513
  matrices measure 6.298x/5.701x pinned PDAL with 25 sources and 1.817x/1.802x with 256
  sources. For the 256-source hierarchy, cells grow 3.984x while raster-build
  time grows 3.934x, leaf visits 3.716x, and total hierarchy-node visits
  4.057x. This is empirical near-linear scaling for that controlled
  distribution, not a worst-case result. An optimized `-pg` profile attributes
  99.09% of samples to exact fill/distance work; compiler inlining makes the
  helper call counts unreliable, so no finer source-level attribution is
  claimed.
- Consequences: the PMF source fill is substantially faster on the recorded
  sparse matrices and remains exact within its fail-closed envelope. It still
  runs on the host, may visit the full occupied hierarchy for pathological
  layouts, and does not cross a stage bridge. The next PMF/Grid slice is a
  separately budgeted device-native source construction and provisional proof
  workspace, followed by measured cross-stage Grid/cloth reuse. Automatic
  placement, the HAG family, catalog closure, and the P3 exit remain open.

## D0092 — Build and prove a fitting PMF source raster on device

- Date: 2026-08-10
- Status: accepted as an exact device-native raster-construction slice for
  required and planner-selected fitting PMF regions; rejected as a worst-case
  scalability proof, adjacent-Grid composition, or automatic placement gate.
- Context: D0090 retained fitting morphology surfaces on device, but every
  successful execution still built the completed source raster on the host and
  uploaded 8 bytes per cell. D0091 improved controlled sparse host fill without
  proving its worst case. Moving construction to CUDA cannot replace upstream's
  selected-source order with an arbitrary atomic minimum: equal numeric Z must
  retain the first source's binary64 bits, and every equal-distance nearest
  source must be observed so distinct bits can fail closed before the
  planner-resident device Classification column is materialized or any
  classification is mutated.
- Decision: add a separately declared 16-byte-per-cell device proof workspace
  to PMF's planner contract. `RasterGridProduct` permits it only when the whole
  frame fits and its width exactly equals the two planned phase backings. One
  byte below the complete runtime boundary performs no proof allocation and
  retains D0091's exact host-build/tiled fallback. A successful proof promotes
  the same allocation into the two resident phase backings, publishes exactly
  one device raster generation, and performs no second allocation. A failed
  proof discards the provisional allocation and leaves the unpublished product
  reusable. Host publication and phase materialization reject while proof is
  provisional.
- Decision: use the first 8 bytes per cell for the completed binary64 raster
  and the second 8 bytes per cell for a 32-bit first-source map plus a compact
  32-bit source-cell list. The minimum kernels choose the numeric minimum and
  then the lowest selected point index among equal numeric values before
  materializing that point's exact Z bits. Each target is assigned to an entire
  block, with blocks advancing grid-stride across targets, and is proved against
  every compact source with explicit precise binary64 center, subtraction,
  multiplication, and addition order. Compact-list order is unobservable: a
  unique least distance fixes the result, same-bit ties produce the same bits,
  and distinct-bit ties reject. Completed voids never become sources. In the
  planner-resident wrapper, device Classification storage and its H2D are
  materialized only after proof succeeds; the host staging column exists
  before the proof so a rejected path can preserve upstream-visible state.
- Decision: apply the same compact-source/tie-proof reduction to the older
  bounded direct CUDA PMF lane so it also rejects distinct-bit nearest ties
  before Classification mutation. Both wrapper shapes catch only that named
  proof failure, run the pinned stage on a private copy of the still-untouched
  view, and publish its Classification result. Return diagnostics already
  emitted by the wrapper are not repeated. Keep default and broad experimental
  PMF on pinned PDAL, retain the exact host builder when the proof pair does
  not fit, and admit no adjacent Grid bridge.
- Evidence: planner/product units cover the exact one-byte boundary, mismatched
  proof/phase widths, no allocation below the boundary, provisional lifetime,
  duplicate materialization, one-shot pointer-preserving promotion, discard,
  and retry. Physical differentials compare the completed device raster to an
  independent raw-bit literal result on a 1x766 large-origin fractional frame
  with 256 sources and signed-zero same-cell minima; separate tests reject a
  distinct-bit tie before publication/mutation and then successfully retry.
  Direct-device NaN/Inf X, Y, and Z cases reject before raster indexing,
  discard unpublished storage, preserve Classification, and then retry
  successfully on the same product.
  The nontrivial-seam wrapper proves the one-byte-below host fallback and the
  fitting zero-raster-transfer path. A wrapper regression and the complete
  16-case CUDA process matrix prove exact pristine-view fallback for an
  ambiguous tie, including missing-return and all-zero warning bytes. Host
  Debug and ASan/UBSan each pass 409 tests with one expected corpus skip in 410
  registrations. Physical SM-89 CUDA Release passes all 602 registrations.
  Focused memcheck reports zero
  errors and zero leaked bytes, initcheck and synccheck report zero errors, and
  racecheck reports zero hazards across five literal-build, nonfinite,
  ambiguity, bounded, and fallback lanes.
- Evidence: B0030 includes product setup, XYZ and Classification H2D, device
  construction/proof/promotion, device phases, and Classification D2H in the
  candidate wall. It measures 5.900x/38.950x pinned PDAL on dense 65x65/513x513
  frames, 3.156x on a 25-source 65x65 frame, and 37.154x/33.829x on controlled
  256-source 257x257/513x513 sparse frames. Every row is classification-exact
  and records zero raster H2D and D2H bytes. The sparse 257x257 Nsight Compute
  report identifies `fillNearestProofKernel` as the limiter: about 1.25 ms,
  50.0% SM throughput, 6.7% memory throughput, and only 0.34 full waves for 259
  target blocks. This is exact FP64 distance/reduction work and controlled-grid
  underfill, not a transfer bottleneck.
- Consequences: fitting required PMF regions now construct, prove, and consume
  the complete raster on device without a raster transfer. The exact proof is
  still proportional to target cells times populated cells and therefore makes
  no arbitrary sparse-frame or worst-case scalability claim. Larger or
  under-budget frames retain the exact host-build/tiled path. Measured
  cross-stage Grid/cloth reuse, the HAG family, automatic placement, catalog
  closure, and the P3 exit remain open.

## D0093 — Reuse one PMF raster allocation across compatible adjacent stages

- Date: 2026-08-10
- Status: accepted as exact PMF-only allocation and product-geometry reuse;
  rejected as semantic raster reuse, cross-kind Grid/cloth composition,
  automatic placement, or a general scalability gate.
- Context: D0092 gave a selected fitting PMF stage one planner-owned product
  whose successful device proof allocation became its morphology phase pair,
  but the one-generation lifetime forced an immediate region spill. Adjacent
  PMF stages over the same selected points use the same source frame and need
  the same phase allocation, yet cannot reuse raster contents: each stage's
  morphology overwrites both backings and its return selection may change the
  source raster. Reusing allocation is exact; reusing a completed surface is
  not.
- Decision: the resident rewrite may keep one Grid region open only for a
  contiguous chain of `filters.pmf` stages with the same Grid/cell contract and
  the same original JSON `returns` value. The comparison is deliberately
  conservative: an absent `returns` member and an explicitly equivalent value
  do not compose. The rewrite emits hidden first/reuse and final-stage markers;
  the resident context creates the product only for the first marker, requires
  the existing product for every reuse marker, rechecks the runtime frame, and
  completes the delegated region only after the final stage. A non-PMF Grid
  consumer, a different cell frame, or a different return source fails closed.
- Decision: generalize `RasterGridProduct` to sequential, fully ordered raster
  generations with at most one pending generation. A later build is legal only
  after the prior generation is consumed. Starting a new host build or device
  proof resets observable orientation/native-build state but preserves the
  already allocated phase pair; failed device proof still discards its
  provisional allocation. Build and consume counts are cumulative, and a
  dedicated allocation counter proves that two generations used one underlying
  device backing allocation. Planner and execution telemetry count one
  `GridBuild` product and two semantic `RasterBuild` generations.
- Decision: rebuild and consume the source raster for every PMF stage. XYZ and
  Classification remain stage-local staging transfers and Classification is
  published between stages; this slice does not yet retain all point columns
  across the PMF pair. No completed raster, morphology surface, PMF result, or
  cloth/grid content is shared semantically.
- Evidence: product units prove pointer-preserving second-generation device
  promotion, cumulative build/consume counts, one allocation, and the pending-
  generation overwrite guard. Rewrite units cover standalone markers, the
  accepted pair, return mismatch, cell mismatch, and the existing PMF-to-point
  bridge rejection. A physical wrapper differential executes a real prepared
  two-stage PMF chain, matches pinned PDAL exactly, observes one `GridBuild`,
  two `RasterBuild` events, zero raster transfers, and a final spill. Host
  Debug and ASan/UBSan each pass 413 tests plus one expected opt-in corpus skip
  in 414 registrations. Physical SM-89 CUDA Release passes 598 tests plus nine
  documented fixture/profile-gated skips in 607 registrations. Focused
  memcheck, initcheck, and synccheck report zero errors; racecheck reports zero
  hazards.
- Evidence: B0031 includes shared-product construction, pinned source staging,
  both stages' XYZ/Classification H2D, device raster rebuild/proof, device
  phases, and Classification D2H. The exact 4,225-point 65x65 pair measures
  3.280602 ms pinned PDAL versus 0.714716 ms candidate, or 4.590078x. It records
  one product, one device backing allocation, two raster generations, and zero
  raster H2D/D2H. Nsight Compute identifies each exact nearest-source proof as
  the limiter at about 183.2-183.8 microseconds with about 2.1% SM and memory
  throughput on the 17-block controlled grid.
- Consequences: compatible adjacent PMF stages now amortize product geometry
  and one device phase allocation without weakening compatibility. Every stage
  still rebuilds its exact raster and round-trips its point columns, and the
  device proof remains target-cells-times-sources work. Semantic Grid/cloth
  reuse, cross-kind composition, automatic placement, the HAG family, catalog
  closure, and the P3 exit remain open.

### D0093 acceptance-evidence addendum

- The preceding B0031 speed paragraph is withdrawn as acceptance evidence. An
  independent audit found that its candidate timed direct PMF primitives while
  the oracle timed prepared PDAL stages, so the reported 4.590078x ratio did
  not compare equivalent scopes. The allocation and generation result remains
  established by product/runtime tests, but not by that rejected timing ratio.
- Corrected B0031 v2 times the actual resident execution scope, preflight,
  upload boundary, prepared `filters.pdg_pmf` wrapper chain, spill boundary,
  return selection, and complete six-column snapshot against the corresponding
  prepared two-stage PDAL chain. All nine samples are exact and observe one
  `GridBuild`, two `RasterBuild` generations, zero raster transfers, and one
  region begin/end. The corrected medians are 3.288598 ms PDAL and 3.113610 ms
  PDG, or 1.056201x on this controlled fixture. One underlying phase allocation
  remains separately proved by the pointer/allocation-count product unit.
- Acceptance coverage now includes a successful three-stage chain with a true
  `reuse=true,last=false` intermediate stage, an all-fallback tie chain whose
  nonterminal reuse marker keeps the region open until final fallback closes
  it, and a nonterminal marker exception that closes the region, rejects a
  subsequent final wrapper, and permits the planned spill. A successful
  post-tie raster generation is not claimed: unchanged geometry and the exact
  source contract reproduce the same tie at every PMF in that chain.
  The selected-placement accounting helper is shared with CLI stats and proves
  one predicted physical Grid build while the logical plan retains two PMF
  Grid stages. Public selected `--stats` output remains unavailable for this
  path because D0093 deliberately adds no automatic placement model; no
  test-only model or force-selection behavior is introduced.
- Updated gates: Host Debug and ASan/UBSan each pass 415 tests plus one expected
  opt-in corpus skip in 416 registrations. Physical SM-89 CUDA Release passes
  603 tests plus nine documented skips in 612 registrations. Memcheck,
  initcheck, racecheck, and synccheck are clean on all three new lifecycle
  cases. The corrected 62-launch profile keeps the same limiter conclusion:
  four exact nearest-source proofs take 183.616-183.744 microseconds at about
  2.11% SM/memory throughput on a 17-block, 0.02-wave grid.

### D0093 timing-boundary correction

- A second independent audit found that B0031 v2 stopped the candidate timer
  before its local resident/table/stage RAII teardown, while the externally
  timed PDAL call included the corresponding local teardown. B0031 v2's
  1.056201x ratio is therefore also withdrawn as acceptance evidence.
- B0031 v3 starts and stops both timers in the caller around the complete
  worker call. Both paths include destruction of their local table, stages,
  views, and owned resources before the stop time; both retain the returned
  six-column snapshot for comparison. The final exact medians are 3.244038 ms
  PDAL and 3.189958 ms resident PDG, or 1.016953x. The final profile contains
  the same 62 launches; exact nearest-source proof remains the limiter at
  183.584-183.872 microseconds and about 2.11% SM/memory throughput.

### D0093 comparison-scope refinement

- B0031 v3's symmetric worker boundary was accepted by independent review, but
  its candidate timer still included the byte-comparison pass while the PDAL
  timer did not. That conservative 1.016953x ratio is superseded for strict
  scope equivalence. V4 performs the six-column exact comparison after both
  stop times and records that contract in the artifact.
- Final B0031 v4 medians are 3.296628 ms PDAL and 3.170328 ms resident PDG, or
  1.039838x. All nine samples remain exact with the same lifecycle counters.
  The matching 62-launch profile records nearest-source proof at
  183.616-184.064 microseconds and 2.113036-2.115619% SM/memory throughput.

## D0094 — Add an exact count-one HAG lane on the shared 2D index

- Date: 2026-08-10
- Status: accepted as an exact, data-dependent `filters.hag_nn` count-one
  CUDA/resident slice; rejected as an automatic placement model, complete HAG
  family, or P3 exit.
- Context: upstream `filters.hag_nn` separates ground and non-ground rows and
  searches a two-dimensional ground-only domain. Existing planner-owned kNN
  products were unmasked and implicitly three-dimensional. Retrieving one
  candidate also cannot prove a unique nearest result: an equal-distance
  second ground point may change nanoflann's source-order choice, while a
  bounded uniform-grid search may not have reached every reference. A private
  KD tree or unproved candidate would violate the compatibility and index
  ownership contracts.
- Decision: add spatial dimensionality to the planner's index request and
  resident lifetime key. Add a masked kNN gather that accepts distinct source
  and reference domains, retains one extra candidate internally, and reports
  exact, tied, or incomplete status. Both UniformGrid and MortonBvh use the
  same `(distance, source id)` ordering and the same status contract. The HAG
  projection writes exact positive zero to ground points and uses precise
  binary64 `Z_query - Z_ground` for proved non-ground rows. The stage owns an
  outer `pdg::filters.hag_nn` NVTX range.
- Decision: admit only `count=1`, valid class values, and finite numeric option
  forms. Preserve the pinned count-one behavior that ignores `max_distance`
  and `allow_extrapolation`. Equal-distance or incomplete device results run
  the original host implementation before publication. Nonfinite XY declines
  before index allocation or mutation. A nonterminal host repair republishes
  its exact HAG column into the resident product; for nonfinite coordinates it
  uses an index-free bridge product. No-ground behavior and diagnostics remain
  upstream-owned. Default placement remains host because this envelope is
  data-dependent and no calibrated model can predict tie/incomplete status.
- Decision: emit the hidden dimension marker only for non-default-dimensional
  kNN wrappers. Existing 3D wrappers continue to use their established hidden
  option surface. A 2D HAG product never satisfies a 3D neighborhood request;
  either order rebuilds the planner-owned index.
- Evidence: the 20-case host and 11-case physical CUDA process matrices cover
  both forced backends, unique and equal nearest points, sparse-grid
  incompleteness, empty/all/one/no-ground shapes, custom class, same XY,
  nonfinite fallback, ignored count-one options, unsupported counts/options,
  and preparation errors. Five repeated CUDA matrices are exact. Executed
  resident tests prove HAG→nndistance and nndistance→HAG each observe two index
  builds and match both upstream columns, including tie and no-ground repair
  before the 3D consumer. Adjacent assignment bridges cover unique, tie,
  no-ground, nonfinite, and empty views. Host ASan/UBSan passes 421 tests plus
  one expected corpus skip in 422 registrations. Physical SM-89 CUDA Release
  passes 618 tests plus nine documented skips in 627 registrations. Focused
  memcheck reports zero errors and zero leaked bytes; initcheck and synccheck
  report zero errors; racecheck reports zero hazards across eight HAG/masked-
  index tests.
- Evidence: B0032 is an equivalent complete-process measurement on a
  deterministic 1,000,002-point unique-nearest fixture. Pinned PDAL measures
  0.896806490 s median and forced exact PDG 0.592750133 s, or 1.512959x, with
  byte-identical output. The 18-launch Nsight report identifies masked
  `knnGatherKernel` as the limiter at 1.64 ms, 76.00% SM throughput, 34.10%
  memory throughput, 1.75% DRAM throughput, and 47.74% achieved occupancy.
  The HAG projection takes 15.94 microseconds and is memory-bound.
- Consequences: the first HAG vertical slice composes with planner-owned 2D
  indexing and adjacent resident consumers without weakening exactness.
  `filters.hag_dem`, `filters.hag_delaunay`, count-greater-than-one native
  interpolation, automatic placement, the remaining P3 catalog, and the phase
  exit remain open.

## D0095 — Extend exact shared-index HAG interpolation to count two

- Date: 2026-08-10
- Status: accepted as an exact, data-dependent `filters.hag_nn,count=2`
  CUDA/resident slice; rejected as an automatic placement model,
  `count >= 3` admission, complete HAG family, or P3 exit.
- Context: D0094 proved a masked, planner-owned 2D query for a unique nearest
  ground point. The pinned count-two branch additionally observes ordered
  inverse-squared-distance interpolation, a strict squared-distance cutoff,
  inclusive ground bounds when extrapolation is disabled, and historical
  behavior when fewer ground references exist than requested. Proving only the
  first two candidates is insufficient because an equal-distance third
  candidate can change nanoflann's selected pair.
- Decision: admit exactly counts one and two. The planner requests the public
  count from the shared 2D index and resident preflight budgets two ids and two
  binary64 squared distances for count two, for 27 bytes of per-point scratch
  including the source, ground, and status bytes. The masked gather retains the
  existing extra candidate and rejects both internal and selected/unselected
  boundary ties, as well as incomplete bounded-grid searches, before HAG
  publication. It uses the common `(distance, source id)` device ordering on
  both UniformGrid and MortonBvh.
- Decision: the count-two projection preserves the pinned operation sequence:
  ordered `1 / distance_squared` weights, separate multiply/add accumulation,
  division, and final subtraction, all under the exact CUDA translation-unit
  flags. `max_distance` is squared on the host as upstream does and comparison
  remains strictly greater than; no-extrapolation bounds remain inclusive.
  Same-XY queries take the nearest ground Z. Ground rows receive positive zero.
- Decision: finite XY remains mandatory for both counts and finite Z is
  mandatory for count two. Count two with only one ground reference executes
  the pinned host filter because its requested two-wide KD2Index result
  produces the historical NaN outcome; the compact masked query must not
  replace it with nearest subtraction. Nonfinite Z, insufficient ground, ties,
  and incomplete searches have positive fallback-proof switches. A
  nonterminal host repair retains the count-sized 2D resident product and
  republishes the exact HAG column for an adjacent point-program bridge.
  `count >= 3` remains upstream-owned.
- Evidence: the expanded 32-case host and 24-case physical CUDA process
  matrices cover both counts, forced index backends, interpolation and option
  boundaries, same XY, strict maximum-distance equality, inclusive ground
  bounds, one-ground and nonfinite-Z repair, internal and second/third boundary
  ties, grid incompleteness, `count=3` host selection, malformed inputs, and
  diagnostics. Units cover planner/hybrid/resident metadata, both terminal
  lifecycles, exact upstream values, and native plus repaired count-two HAG to
  assignment bridges. Grid and non-Grid resident products now receive distinct
  regions with explicit spill/upload boundaries: the planner accounts all six
  boundaries in PMF -> HAG2 -> PMF, the rewrite preserves their order, and a
  physical differential executes all three regions with two grid builds and
  one 2D index build. Same-kind DAG siblings continue to share their product,
  while a selected multi-consumer boundary fails closed. SMRF, PMF, CSF, and
  ELM to point-program rewrite tests lock the general cross-product boundary
  rule without claiming semantic product reuse. The 429-registration Host
  ASan/UBSan aggregate passes with one optional-corpus skip; the 643-test
  physical CUDA aggregate records 634 passes and the nine documented
  corpus/placement skips.
  Focused memcheck, initcheck, racecheck, and synccheck are clean across the 12
  count-two, masked-index, bridge, and mixed-PMF tests.
- Evidence: B0033 measures an equivalent complete process on a deterministic
  1,000,002-point two-neighbor fixture. Pinned PDAL measures 0.925865468 s
  median and forced exact PDG 0.620000696 s, or 1.493330x, with byte-identical
  output. The 18-launch Nsight report identifies the masked two-neighbor
  `knnGatherKernel` as the limiter at 2.15 ms, 74.55% compute throughput,
  51.44% memory throughput, and 48.30% achieved occupancy. The exact HAG
  projection takes 49.15 microseconds and is memory-bound.
- Consequences: the bounded HAG-NN interpolation branch now composes through
  count two without weakening the oracle contract. Grid-to-non-Grid resident
  products compose only through explicit materialization boundaries; the
  RasterGrid or neighborhood product itself never crosses that boundary.
  Incompatible Grid-to-Grid chains remain rewrite-time fail-closed unless they
  satisfy the separately proved adjacent-PMF contract from D0093.
  Data-dependent repair still prevents an automatic model. Counts three and
  greater, `filters.hag_dem`, `filters.hag_delaunay`, the remaining P3 catalog,
  and the phase exit remain open.

## D0096 — Add an exact count-three HAG Delaunay lane

- Date: 2026-08-10
- Status: accepted as an exact, data-dependent
  `filters.hag_delaunay,count=3` CUDA/resident slice; rejected as an automatic
  placement model, wider Delaunay admission, complete HAG family, or P3 exit.
- Context: the pinned Delaunay filter queries a ground-only two-dimensional
  index and, for each non-ground row, triangulates the requested local ground
  neighborhood before applying PDAL's barycentric interpolation. Exactly three
  selected references produce at most one triangle, making the upstream
  Delaunator seed order and arithmetic bounded enough to reproduce directly.
  Wider counts require a complete advancing-hull triangulation and remain
  outside this decision. As with HAG NN, retrieving only the requested three
  candidates cannot prove the result when an equal-distance fourth reference
  exists or a bounded grid search is incomplete.
- Decision: admit only an explicit integer `count=3`, boolean
  `allow_extrapolation`, and an unsigned-byte ground class. The default
  `count=10`, counts below three or above three, `where`, malformed options,
  and unavailable CUDA retain the original stage. The planner requests three
  neighbors from its masked two-dimensional index and budgets 39 bytes per
  point for source/ground/status bytes, three source ids, and three binary64
  squared distances. No private spatial index or triangulation product is
  built.
- Decision: for a proved candidate order, the exact kernel reproduces the
  three-point Delaunator seed selection, circumradius validity and winding
  swap, followed by the pinned `math::barycentricInterpolation` operation
  sequence and `1e-14` edge tolerance. Ground rows receive positive zero;
  same-XY queries use the nearest ground Z; disabling extrapolation uses the
  inclusive ground BOX2D; a point outside the local triangle uses the nearest
  ground Z; and degenerate/collinear triples preserve the upstream zero-HAG
  exception path. The exact CUDA translation-unit flags disable contraction
  and FTZ and use precise binary64 division.
- Decision: finite binary64 XYZ, at least three ground references, an exact
  masked search, and an unambiguous third/fourth boundary are runtime
  requirements. Nonfinite input, too few grounds, ties, or incomplete searches
  run the pinned host filter before publication. Positive proof switches cover
  every repair category. A nonterminal repair republishes the exact HAG column
  into the retained two-dimensional resident product so an adjacent point
  program can consume it; empty terminal and empty bridge lifecycles remain
  exact no-ops.
- Evidence: the 22-case host and 15-case physical CUDA matrices cover explicit
  count-three admission, the default and wider host paths, empty/all-ground,
  custom class, same XY, inside/outside local triangles, disabled
  extrapolation, collinearity, duplicate and selected-boundary ties,
  insufficient/no ground including the exact error diagnostic, nonfinite Z,
  incomplete grid search, both forced index backends, unsupported options, and
  preparation errors. Five consecutive
  CUDA matrix runs are byte-exact. Focused units add Delaunator seed ordering,
  the barycentric edge threshold, signed-zero/subnormal and large-coordinate
  geometry, positive/negative interpolation overflow, terminal resident
  execution, and native plus insufficient/tie/incomplete repaired
  HAG-to-assignment bridges. Host ASan/UBSan passes 432 tests plus one expected
  corpus skip in 433 registrations. Physical SM-89 CUDA Release passes 644
  tests plus nine documented corpus/placement skips in 653 registrations.
  Focused memcheck, initcheck, racecheck, and synccheck are clean across nine
  Delaunay and masked-index tests.
- Evidence: B0034 measures the exact complete process on a deterministic
  1,000,002-point fixture whose first three ground candidates and
  third/fourth boundary are unique. Pinned PDAL measures 0.785051840 s median
  and forced exact PDG 0.597404936 s, or 1.314103x, with byte-identical output.
  The 18-launch Nsight report identifies masked `knnGatherKernel` as the
  limiter at 1.96 ms, 67.82% compute throughput, 52.33% memory throughput, and
  45.62% achieved occupancy. The exact Delaunay projection takes 111.46
  microseconds and is register-limited.
- Consequences: the bounded HAG family now includes HAG NN counts one and two
  plus Delaunay count three on the shared 2D index, all with exact host repair
  and adjacent-column composition. Data-dependent proof and the modest
  controlled speedup do not justify automatic placement. HAG NN counts three
  and greater, Delaunay counts four and greater/default ten,
  `filters.hag_dem`, the remaining P3 catalog, and the phase exit remain open.

## D0097 — Extend exact shared-index HAG interpolation to count three

- Date: 2026-08-10
- Status: accepted as an exact, data-dependent `filters.hag_nn,count=3`
  CUDA/resident slice; rejected as an automatic placement model, `count >= 4`
  admission, complete HAG family, or P3 exit.
- Context: the pinned HAG-NN branch for counts greater than one visits the
  selected ground candidates in kNN order and performs one binary64
  inverse-squared-distance weight and weighted-Z accumulation per candidate.
  D0095 already reproduced that loop for two candidates and proved that a
  requested-width-only result is insufficient when the next candidate ties or
  a bounded grid search is incomplete. Count three adds one ordered term but no
  new algorithm or semantic product.
- Decision: admit explicit integer counts one through three, boolean
  `allow_extrapolation`, binary64 `max_distance`, and an unsigned-byte ground
  class. Counts four and greater, `where`, malformed options, or unavailable
  CUDA retain the original stage. Counts two and three require finite XYZ and
  at least the requested number of ground references. The planner-owned masked
  two-dimensional index remains the only spatial index. Count-three preflight
  budgets 39 bytes per point for the three masks/status bytes plus three
  source ids and three binary64 squared distances.
- Decision: the exact projection retains the pinned loop order for each of the
  three candidates: divide one by squared distance, add the weight, add
  weight-times-ground-Z, then divide the final Z accumulator by the weight sum
  and subtract it from query Z. Ground rows receive positive zero; same-XY
  lookup, the strict `distance > max_distance` cutoff (including negative
  distance squaring), and inclusive no-extrapolation BOX2D bounds remain
  unchanged. The exact CUDA translation-unit flags disable FMA contraction and
  FTZ and require precise binary64 division.
- Decision: internal or third/fourth candidate ties, squared-distance overflow
  that collapses distinct finite coordinates to equal infinities, incomplete
  bounded-grid searches, nonfinite Z, or fewer than three ground references
  run the pinned host filter before publication. Positive proof switches cover
  each repair class on both forced shared-index backends where applicable. A
  nonterminal repair republishes the exact HAG column into the retained 2D
  resident product so an adjacent point program can consume it; native,
  insufficient-ground, tied-boundary, and incomplete cases preserve the
  delegated lifecycle.
- Evidence: the expanded 54-case host and 46-case physical CUDA matrices cover
  count-three admission, both forced backends, same-XY and unique interpolation,
  strict equality and partial/negative cutoffs, inclusive bounds, disabled
  extrapolation, large finite coordinates, signed zero, subnormal squared
  distances, weight overflow, true squared-distance overflow, nonfinite XY/Z,
  insufficient/no ground, internal and boundary ties, incomplete grid search,
  exact diagnostics, and count-four host ownership. Five consecutive current-
  code CUDA matrix runs are byte-exact. Focused units add terminal execution and
  native plus repaired HAG-to-assignment bridges. Host ASan/UBSan passes 435
  tests plus one expected corpus skip in 436 registrations. Its D0096
  deliberate assertion-abort row is omitted only when ASan is injected because
  glibc prefixes otherwise-identical stderr with the differing executable
  names; normal host and CUDA matrices retain that byte-exact row. Physical
  SM-89 CUDA Release passes 658 tests plus nine documented corpus/placement
  skips in 667 registrations. Memcheck, initcheck, racecheck, and synccheck are
  clean across the 12-test count-three and masked-index lane.
- Evidence: B0035 measures the exact complete process on a deterministic
  1,000,002-point fixture with a unique first three references and
  third/fourth boundary. Pinned PDAL measures 0.957174561 s median and forced
  exact PDG 0.645563424 s, or 1.482696x, with byte-identical output. The
  18-launch Nsight report identifies masked `knnGatherKernel` as the limiter at
  2.68 ms, 74.13% compute throughput, 65.01% memory throughput, and 48.93%
  achieved occupancy. The exact count-three HAG projection takes 59.55
  microseconds and is memory-bound.
- Consequences: the bounded HAG-NN interpolation branch now composes through
  count three without changing shared-index ownership or weakening the oracle
  contract. Data-dependent proof and the controlled speedup do not justify an
  automatic model. HAG-NN counts four and greater, Delaunay counts four and
  greater/default ten, `filters.hag_dem`, the remaining P3 catalog, and the
  phase exit remain open.

## D0098 — Extend exact shared-index HAG interpolation to count four

- Date: 2026-08-10
- Status: accepted as a GPU-native and forced-fixture performance-qualified
  `filters.hag_nn,count=4` slice; rejected as automatically selected, a claim
  about ordinary-data speedup, `count >= 5` admission, complete HAG-family
  coverage, or P3 exit.
- Context: the pinned wider HAG-NN branch uses the same ordered inverse-
  squared-distance recurrence for any requested count. D0097 stopped at three
  only because its admission, scratch budget, and exact evidence were bounded
  there. Count four adds one id, one binary64 squared distance, one ordered
  arithmetic term, and a fifth-candidate proof; it does not introduce a new
  index or algorithm.
- Decision: admit explicit counts one through four under the existing option,
  finite-coordinate, ground-cardinality, and whole-view envelope. The
  planner-owned masked two-dimensional index remains the only spatial index.
  Count-four preflight budgets 51 bytes per point: source mask, ground mask,
  status, four 32-bit ids, and four binary64 squared distances. Count five and
  wider remain pinned-host owned.
- Decision: preserve the pinned four-term loop order, strict squared
  `max_distance` cutoff including negative inputs, inclusive no-extrapolation
  bounds, same-XY branch, positive-zero ground rows, and exact translation-unit
  arithmetic flags. A distinct fifth reference is queried to prove that the
  selected fourth candidate is unique. Fourth/fifth or internal ties,
  incomplete bounded-grid search, nonfinite Z, or fewer than four grounds run
  the original host filter before publication. Native and repaired HAG columns
  may feed the existing adjacent point-program bridge.
- Decision: correct D0097's over-broad overflow-backend wording. Finite values
  separated by about `1.4e154` prove binary64 squared-distance overflow and tie
  repair only through UniformGrid. MortonBvh rejects that span at its binary32
  coordinate-frame admission before issuing a query, so it is not a BVH tie
  proof. A representable third/fourth or fourth/fifth boundary fixture remains
  the positive BVH tie proof. No correctness behavior changes with this
  evidence correction.
- Evidence: the combined matrix contains 74 host and 66 physical CUDA cases.
  Both forced backends execute a successful count-four query with five
  distinctly ordered ground references and prove fourth/fifth boundary repair;
  UniformGrid separately proves true squared-distance-overflow repair. The
  matrix also covers same XY, cutoff equality/partial/negative cases, inclusive
  bounds, disabled extrapolation, large finite and signed-zero/subnormal/weight-
  overflow arithmetic, nonfinite XY/Z, one/three/no-ground fallback, incomplete
  grid search, exact diagnostics, and count-five host ownership. Five
  consecutive current-code CUDA matrices are byte-exact.
- Evidence: focused units execute terminal counts one through four and native,
  insufficient-ground, tied-fifth, and incomplete count-four assignment
  bridges. Host ASan/UBSan passes 438 tests plus one expected corpus skip in
  439 registrations. Physical SM-89 CUDA Release passes 669 tests plus nine
  documented corpus/placement skips in 678 registrations. Memcheck, initcheck,
  racecheck, and synccheck are clean over the 12-test count-four, masked-index,
  and resident-lifecycle lane.
- Evidence: B0036 measures the exact complete forced pipeline on the controlled
  1,000,002-point fixture at 0.975764572 s pinned PDAL versus 0.614848287 s PDG,
  or 1.587001x. Its 18-launch profile identifies masked `knnGatherKernel` as
  the limiter at 3.06 ms, 72.35% compute throughput, 81.58% memory throughput,
  64 registers per thread, and 50.01% achieved occupancy. The exact four-term
  `hagNnKernel` takes 73.63 microseconds and is memory-bound.
- Consequences: count four is functionally supported, GPU-native inside its
  proof envelope, and performance-qualified only for the named forced fixture.
  It is not automatically selected. Per the post-D0098 reprioritization, no
  count-five or other new stage port begins before the implemented catalog is
  audited under those four independent categories and the next work is chosen
  from an end-to-end pipeline profile.

## D0099 — Reorder implementation around measured whole-pipeline value

- Date: 2026-08-10
- Status: accepted; supersedes the post-D0082 catalog-order queue and activates
  the whole-pipeline program before catalog completion. It changes no exact
  execution envelope, placement model, or automatic selector.
- Context: stage count and isolated kernel speed ceased to predict product
  value. D0072 measured the presumed attach hot loop at only 0.87 seconds of a
  50.7-second lane. D0073 improved the dense kNN kernel by 2.2–3.4x without an
  end-to-end gain because a sparse work-imbalance tail still dominated; D0074's
  exact incomplete-row repair removed that tail and finally moved the complete
  pipelines. Conversely, B0014's endpoint fusion lifted an expression class
  from 1.448x to 18.038x without adding another stage kernel. The implementation
  order must therefore be chosen by complete reader/transfer/filter/writer
  evidence, with fusion and residency preferred when they remove shared costs.
- Decision: report four independent properties for every implemented stage
  envelope: **functionally supported**, **GPU-native**,
  **performance-qualified**, and **automatically selected**. Functional support
  may be exact fallback. GPU-native means the stated work actually executes on
  device; a host finale is labelled partial. Performance qualification requires
  accepted same-machine complete-process evidence; a negative result qualifies
  the decision to remain host-selected, while primitive, dirty, one-shot, or
  decision-only timing is labelled diagnostic rather than promoted. Automatic
  means the normal option-free public path, not force/require flags,
  experimental execution, `pdg resident`, or the production-default-off P1.5
  placement model. Where no end-to-end record exists, say **unmeasured**.
- Decision: pause new stage ports after D0098. For every candidate, first
  measure pinned PDAL, profile and name the dominant complete-process cost,
  choose fusion/residency/standalone CUDA/optimized host bridge/deferral, and
  reject weak directions with a cheap prototype. Proceed only for a standalone
  CPU win, material transfer removal, reusable shared work, or explicit
  catalog-coverage need. Exactness precedes a full-pipeline remeasurement; no
  performance claim is published without its baseline, profile, and append-only
  `BENCHMARKS.md` record.
- Audit: all 122 configured drivers remain functionally supported. The seven
  automatically selected filters remain `assign`, `ferry`, `expression`,
  `range`, `crop`, `mortonorder`, and `expressionstats` in their measured
  composite/standalone selector envelopes; pure LAS translation and a
  crop-only CUDA path remain host-selected. The shared-neighborhood resident
  family has the strongest current
  forced/resident evidence (B0017 approximate-coplanar + ferry 3.962x and LOF +
  assign 8.459x). B0018 performance-qualifies normal's three-run 4.037x lane;
  its eigenvalues, covariance-features, and NN-distance single runs are exact
  acceptance checks, explicitly not timing records. Normal public execution
  remains host-selected. `radiusassign` has a 5.503x dirty-snapshot diagnostic
  and is hybrid because its radius selection is CUDA and assignment finale is
  host. Estimate-rank,
  optimal-neighborhood, and neighbor-classifier have positive decision-entry
  timings but lack the required append-only benchmark/profile entries and are
  not performance-qualified by this audit. Outlier is unmeasured as an accepted
  complete stage pipeline; radial-density retains diagnostic-only evidence.
- Audit: the recent terrain stages have complete-process diagnostics rather
  than inferred kernel claims, but their dirty snapshots are not D0099
  performance qualifications. B0021 records SMRF at 0.599x, B0022 bounded PMF
  at 0.583x, B0023 CSF at 0.289x, and B0024 ELM at 0.866x. B0031's final
  symmetric adjacent-PMF comparison is only 1.039838x after allocation reuse.
  CSF's exact serial host work and SMRF's tiny one-block launches provide no
  current standalone optimization case. PMF's Grid-to-HAG transitions may
  still expose material boundary cost, but that is explicitly a hypothesis
  until measured in a clean mixed pipeline.
- Priority: (1) profile representative mixed pipelines; (2) fuse resident
  endpoints and remove reader/writer/host-attach boundaries already identified
  by B0005/B0017; (3) extend reusable point-operation fusion around resident
  producers/consumers; (4) reuse planner products only where a mixed profile
  shows value; (5) qualify and, where warranted, automatically select already
  positive lanes; (6) retain GPU-negative stages on host unless composition
  changes the complete process; then (7) resume new catalog ports for required
  coverage.
- Next task: run a profile-only B0037 gate for an exact
  `PMF -> HAG-NN(count=4) -> normal -> LAS` pipeline. Record the equivalent
  pinned-PDAL and PDG walls, complete artifact equality, Grid/masked-2D/3D
  index builds, spill/upload and packing bytes, peak RAM/VRAM, phase times, and
  the dominant profile limiter. Only that evidence may choose endpoint fusion,
  product reuse, an index transition, an optimized host bridge, or deferral.
  B0037 changes no implementation or selector merely by being measured.
- Consequences: Appendix A's whole-pipeline program is active now rather than
  after catalog completion. The catalog-wide exact/native goal remains; only
  its order changes. Exact-but-slower CUDA code may stay as force-only coverage
  but cannot be called a speedup or automatically selected.

## D0100 — Defer PMF-to-HAG product work after the B0037 limiter proof

- Date: 2026-08-10
- Status: accepted; closes the first D0099 mixed-pipeline gate without changing
  code, placement, or automatic selection.
- Context: D0099 labelled the Grid-to-masked-2D-to-3D transitions in an exact
  `PMF -> HAG-NN(count=4) -> normal -> LAS` graph as a hypothesis. The initial
  profiling recipe used `cell_size=100`, whose first PMF radius is 149 and is
  outside the bounded exact CUDA radius of 64. It therefore measured host PMF
  and could not answer the stated all-native question. The corrected recipe
  uses a deterministic 64 by 64 raster with 245 points per cell, one PMF ground
  return per cell, sub-cell deterministic XY offsets, and a native-eligible
  `cell_size=1` program. It produces 1,003,520 points and 4,096 ground
  references.
- Evidence: a clean one-shot viability run at `edc419e76` forced exact hybrid
  execution and Morton BVH. Pinned PDAL took 3.967187545 seconds; PDG took
  4.689327584 seconds, or 0.846003x. The 80,283,181-byte outputs are identical
  with SHA-256
  `b76d783e93da505bc0e52b4fdedb2b7b267ac88e0ae57eb221a35881c603ea42`.
  This deliberate one-sample/no-warmup prototype is diagnostic and not a
  performance qualification.
- Profile: the 95-launch Nsight Compute report separates PMF (launches 0-19),
  HAG/index/projection (20-56), and normal/index/eigensystem/projection
  (57-94). HAG's masked `bvhKnnGatherKernel` takes 3.77 seconds, about 80% of
  candidate wall, while normal's all-point gather takes 72.50 milliseconds.
  The HAG kernel is 80.99% compute-throughput active with 63.21% achieved
  occupancy but only 1.85% DRAM throughput: the sparse 4,096-reference mask in
  a million-point all-point tree causes traversal through irrelevant leaves.
  PMF device work, both index-build kernel sequences, HAG projection, and
  normal projection are not the complete-process limiter.
- Boundary accounting: normal public execution does not select a resident
  region for this graph, so it has no planner spill/upload events or shared
  semantic product. The forced wrappers build two independent Morton indexes.
  Auditing their bulk memcpy sites gives 115,404,800 H2D bytes and 44,154,880
  D2H bytes, excluding scalar proof/status traffic; PointView packing reads
  113,397,760 bytes and publication writes 41,144,320 bytes. A 20-ms diagnostic
  sampler observed 464,592 KiB candidate peak RSS versus 193,040 KiB for the
  oracle and 688 MiB peak VRAM above idle. Those costs are real but cannot
  recover a 3.77-second masked traversal.
- Decision: do not implement Grid-to-HAG or HAG-to-normal product sharing from
  this profile. PMF's raster, HAG's ground-masked 2D neighborhood, and normal's
  all-point 3D neighborhood are distinct semantic products. A future
  planner-owned compact ground-only index may be reconsidered for a pipeline
  that reuses it across multiple HAG consumers, but a private one-stage index
  is not admitted here.
- Next task: B0038 is a bounded profile-only gate over the existing positive
  resident `LOF(minpts=10) -> assign -> LAS` lane. It measures the 4M ALS
  prefix and decomposes reader/writer, PointView attach, pack/publish,
  transfer, index-build, and query walls. A reusable resident LAS endpoint is
  prototyped only if that measured non-kernel budget is material.
- Consequences: B0037 adds no speed claim, performance qualification, or
  automatic selector. It demonstrates the D0099 decision loop working as
  intended: correct the envelope, prove exactness cheaply, name the limiter,
  and defer an attractive but unsupported residency idea before spending a
  certification ladder.

## D0101 — Target LOF exact-repair cliffs before resident endpoint fusion

- Date: 2026-08-10
- Status: accepted; closes the B0038 endpoint gate without changing exact
  execution, placement, or automatic selection.
- Context: D0100 chose the existing positive resident
  `LOF(minpts=10) -> assign -> LAS` lane to test whether reader/writer or
  PointView boundaries should be fused next. The maintained 4M ALS prefix is
  large enough to exercise the shared neighborhood index and ordinary exact
  tie/incomplete repair while keeping this a cheap one-shot viability gate.
- Evidence: on the clean `dcdffcea7` checkpoint, pinned PDAL took
  36.566725005 seconds and planner-resident PDG took 4.437390012 seconds. The
  complete 144,000,375-byte outputs are identical with SHA-256
  `e68b4dded6f5571d8eb45afad9fa5698b4a034d56f7083d39b3ab8002fa80a3`.
  This single-sample 8.240593x result reconfirms viability but is not a
  performance qualification.
- Profile: Nsight Compute records 22 launches and about 145 milliseconds of
  total device-kernel time. The kNN gather takes 141.37 milliseconds; the two
  LOF kernels take 1.60 and 1.49 milliseconds. Resident upload packing, spill
  wait, and spill publication measure 0.1369, 0.0093, and 0.0861 seconds, while
  a pinned-PDAL LAS read/write-only control takes about 1.10 seconds. Those
  endpoint costs are material but do not explain the candidate wall.
- Attribution: a diagnostic-only instrumentation snapshot preserves the
  frozen output hash and measures exact host repair at 2.339810726 seconds of
  a 4.410-second run. Twelve thousand seven hundred sixteen ambiguous rows and
  one incomplete row expand to a 40,982-row compatibility closure. The path
  builds the full host KD3 tree, performs fixed-order upstream recomputation,
  and refreshes all three 4M LOF columns on device. Shell-budget sweeps from 16
  through 256 do not remove that fixed host build; the 16-shell run is fastest,
  while a forced full Morton-BVH query is worse at 7.331 seconds.
- Decision: do not build a resident LAS endpoint first. B0039 will cheaply
  test selective exact device repair using planner-owned index data and only
  the declined closure. It must preserve ordered kth-neighbor, tie, and
  incomplete semantics, and it may not introduce a private index. The existing
  host repair remains authoritative unless the prototype materially lowers
  both the 2.340-second repair and complete-process wall.
- Consequences: the new `exact_host_repair` stats object reports diagnostic
  seconds plus ambiguous, incomplete, and repaired row counts for resident
  execution; host execution reports null. B0038 adds no new speed claim,
  calibration, performance qualification, or automatic selector. New stage
  ports remain paused.

## D0102 — Reject the current spatial-index order for exact LOF tie repair

- Date: 2026-08-10
- Status: accepted; closes the B0039 selective-repair prototype and advances
  the measured endpoint surface. It changes no product code or selector.
- Context: B0038 attributes 2.340 seconds of its 4.410-second instrumented
  resident LOF wall to compatibility repair. Before designing another index,
  B0039 asks the cheaper question: can the planner-owned grid's retained
  `(distance, point-id)` order already substitute for pinned PDAL's KD3 order
  on the 12,716 tie rows and one incomplete row?
- Evidence: a temporary, fully reverted diagnostic branch serialized all three
  LOF columns with `extra_dims=all`. Pinned host execution took 36.929 seconds;
  the ordinary exact hybrid took 3.871 seconds and produced the identical
  240,001,005-byte SHA-256
  `4da3a755583ab3bb9565c7fa307ca9c2998b608da97bc2021435edc61ea390f8`.
  Leaving tie rows on their device values while retaining host repair for the
  one incomplete row took 3.764 seconds but produced SHA-256
  `86047ab8de6a69661b878bd5ed4d5e42547c5391cf3f6a415be0fe44f3fa0904`.
  The first differing byte is LAS offset 2,427,997 (zero-based): record 40,449,
  byte 52, the first byte of `LocalOutlierFactor`. Pinned KD3 writes
  0.9805045741137234 (`ff b2 ec 20 4b 60 ef 3f`); device tie order writes
  0.9805045741137232 (`fd b2 ec 20 4b 60 ef 3f`).
- Ceiling: suppressing all repair took 2.061 seconds but produced a third,
  non-exact SHA-256
  `ec60c1f09b5415bf2c4381cb82e649b50448b434706706ab5847e86b203b31f7`.
  A standard-dimension writer happened to hide both differences because the
  downstream assign threshold did not flip; that artifact is explicitly not
  an exactness proof for the LOF stage.
- Decision: the current grid/BVH order is not an exact selective repair
  backend. Retain the upstream KD3 host repair and do not add a private LOF
  index. A device repair may be reconsidered only as a planner-owned,
  cross-family product that reproduces upstream KD3 construction and tie order
  and first passes this all-column fixture.
- Next task: B0040 prototypes a direct resident LAS endpoint around the
  existing exact `LOF -> assign` region. It targets the measured secondary
  PointView/boundary/I/O surface without weakening repair semantics.
- Consequences: the 2.061-second number is a non-exact ceiling, not a speed
  claim. B0039 is a dirty-snapshot, no-warmup semantic rejection and creates no
  performance qualification, calibration, or automatic selection.

## D0103 — Retain the positive resident LAS endpoint as an opt-in prototype

- Date: 2026-08-10
- Status: accepted; closes B0040 and authorizes bounded qualification work,
  without changing the placement model or automatic selector.
- Context: B0038 measured exact LOF repair as the primary limiter, but also
  found a reusable secondary endpoint surface. The ordinary resident executor
  spills a complete PointView and then asks PDAL's LAS writer to decode and
  repack it. A pinned-PDAL identity translation takes about 1.10 seconds on the
  4M fixture, while the existing exact direct translator takes 0.455 seconds.
- Prototype: for one linear, order/cardinality-preserving, all-native resident
  region, default uncompressed LAS endpoints, and `UserData` as the only
  serialized mutation, an opt-in executor removes the terminal PDAL writer.
  It creates the canonical exact default LAS image, overlays the final
  `UserData` byte, and atomically publishes it. Any other writer option,
  serialized dimension write, graph shape, input format, missing final column,
  existing output, or point-count mismatch refuses or fails closed. The
  ordinary executor remains the default.
- Evidence: the one-shot 4M direct candidate takes 3.768 seconds versus 4.315
  seconds for its same-build resident control, a 1.145x ratio and 12.7% wall
  reduction. Both produce the same 144,000,375-byte SHA-256
  `e68b4dded6f5571d8eb45afad9fa5698b4a034d56f7083d39b3ab8002fa80a3a`.
  Both retain one planner-owned index and the same 12,716 ambiguous, one
  incomplete, 40,982-row exact repair. The direct path still reports the
  ordinary 100 MB spill and approximately 0.228 seconds of upload/spill host
  phases; this gain therefore comes from redundant writer removal, not an
  unproved device-resident sink.
- Decision: keep the endpoint behind
  `PDG_EXPERIMENTAL_DIRECT_RESIDENT_LAS_OUTPUT` and the positive proof switch
  `PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT`. Mark its calibration as unmatched.
  It is a diagnostic prototype, not a performance qualification, speed claim,
  or automatic selection.
- Next task: B0041 adds the bounded failure/options/empty process matrix, runs
  the physical v3 exact lane, and records a clean repeated same-machine
  control/candidate benchmark. Profile the surviving spill before deciding
  whether a planner-owned device-column sink is worth implementing.
- Consequences: the result validates reusable endpoint fusion as the current
  measured priority and starts no new stage port. Exact LOF host repair remains
  authoritative, and no private spatial index is introduced.

## D0104 — Qualify writer bypass and defer a deeper resident LAS sink

- Date: 2026-08-10
- Status: accepted for the named explicit-resident LOF/default-LAS envelope;
  option-free integration and automatic selection remain open.
- Context: B0040's one-shot 12.7% gain justified hardening the endpoint. B0041
  must show that it repeats on a clean checkpoint, preserve exact fallback and
  atomic failures, and measure the ordinary spill before any resident/device
  output interface is widened.
- Exactness evidence: the host process matrix covers small and zero-point
  experimental fallback, including a GeoTIFF-VLR input whose default streaming
  writer differs from forced standard mode. Unselected resident graphs now use
  PDAL's default `PreferStream` execution. The physical 21.97M-point v3 lane
  passes in 257.20 seconds and compares ordinary and required-direct outputs to
  the same upstream artifact. It also proves no-overwrite preservation and
  refuses a second serialized dimension or `extra_dims=all` before output.
- Performance evidence: on clean `dd221c647`, five exact 4M samples plus one
  warmup give a 4.197157212-second ordinary-resident median and a
  3.700118176-second required-direct median. The writer bypass is therefore
  1.134331x incremental, saves 0.497039036 seconds, and reduces candidate wall
  by 11.842%. Its separately interleaved pinned-PDAL median is 35.550673574
  seconds, so the complete direct pipeline is 9.607983x upstream. Every output
  is the same 144,000,375-byte SHA-256
  `e68b4dded6f5571d8eb45afad9fa5698b4a034d56f7083d39b3ab8002fa80a3a`.
- Profile: the final required-direct capture retains one planner-owned index
  and the same 12,716 ambiguous, one incomplete, 40,982-row exact repair. Host
  repair takes 2.329585210 seconds. The 100,000,000-byte final spill takes only
  0.009948101 seconds waiting and 0.082519635 seconds publishing, 2.5% of the
  direct median; upload packing is 0.131433039 seconds. A deeper device-column
  sink cannot remove the dominant repair and has too little measured ceiling
  to be next.
- Decision: performance-qualify the opt-in endpoint only for the named 4M
  explicit-resident LOF/default-LAS envelope. Retain the host-overlay design,
  defer a device-column sink, and keep calibration/automatic selection false.
- Next task: B0042 prototypes option-free `pdg pipeline` integration for this
  exact envelope, reusing the existing placement, preflight, shared index, and
  publisher behind a positive proof switch. Public CLI flags, diagnostics,
  metadata, fallback, and atomic failure behavior must remain pinned-exact.
- Consequences: endpoint fusion advances because it moves complete-pipeline
  wall, not because it removes a conceptual boundary. New stage ports remain
  paused and exact LOF host repair remains authoritative.

## D0105 — Carry the qualified endpoint through the public command under proof

- Date: 2026-08-10
- Status: accepted as a proof-switched public-command envelope; default
  automatic selection remains pending a side-effect-free delegation boundary.
- Context: B0041 qualifies the explicit `pdg resident` writer bypass but does
  not show that the public `pdg pipeline` dispatch/handoff preserves exactness
  or its measured gain. B0042 must exercise that real command without silently
  falling back or pretending an explicit diagnostic command is automatic.
- Decision: `PDG_REQUIRE_AUTOMATIC_RESIDENT_LAS_OUTPUT=1` adapts an otherwise
  option-free public `pipeline FILE` invocation to the existing resident
  planner and requires its exact direct LAS output. Any missing selection,
  unsupported CLI shape, or endpoint refusal fails positively. The thin
  dispatcher recognizes the proof variable. No behavior changes without it.
- Exactness evidence: the below-placement host process refuses without output;
  the physical v3 lane now adds the public-command artifact to its ordinary
  resident, required-direct, and pinned-upstream comparisons. All four paths
  are exact on the 21.97M local fixture; the expanded gate passes in 282.55
  seconds. The 4M frozen one-shot also reproduces SHA-256
  `e68b4dded6f5571d8eb45afad9fa5698b4a034d56f7083d39b3ab8002fa80a3a`.
- Performance evidence: on clean `52449afa4`, one warmup and five alternating
  samples give 35.877380135 seconds pinned PDAL and 3.772574784 seconds for the
  proof-switched public command, or 9.510051x. The candidate range of
  3.725892929–3.861949962 seconds overlaps B0041's explicit-resident range, so
  the public handoff adds no material penalty.
- Decision: performance-qualify only the named proof-switched 4M
  LOF/default-LAS public-command envelope. It remains **not automatically
  selected** because the positive environment variable is required.
- Next task: B0043 separates automatic admission from committed execution. A
  refusal must return control before plugin diagnostics, CUDA initialization,
  pipeline output, or errors so the unchanged public CLI can delegate to its
  existing exact paths. Only then add option-free positive selection.
- Consequences: the public integration direction is viable; a default route is
  not accepted merely because the proof route is fast. New stage ports and a
  deeper device-column sink remain deferred.

## D0106 — Automatically select the measured resident LOF/default-LAS envelope

- Date: 2026-08-10
- Status: accepted for the exact explicit public shape and runtime placement
  envelope described below; no other resident publisher is promoted.
- Context: B0042 proves that the public command can reach B0041's qualified
  writer bypass without a material handoff penalty, but its positive proof
  variable turns refusal into an error. Default selection needs a queryable
  admission boundary that can decline before observable side effects and must
  not retry after an execution has committed.
- Decision: an option-free `pdg pipeline FILE` may attempt automatic resident
  execution only when the JSON root contains exactly one four-stage pipeline:
  explicit default `readers.las`, explicit `filters.lof,minpts=10`, the exact
  `UserData = 1 WHERE LocalOutlierFactor >= 1.2` assignment, and an explicit
  default `writers.las`. Input/output default-translation support, a missing
  target, an exact device/profile, the measured runtime placement decision, a
  nonempty resident rewrite, direct-publication support, and resident
  preflight must all accept. A future independent fused executor cannot inherit
  this decision without its own complete-process evidence.
- Failure boundary: static graph or CLI refusal returns before planner/device
  work. Input/output, device/profile/placement, and preflight refusal returns
  control without output or diagnostics to the unchanged public selector. The
  execution manager marks the path committed immediately before executing the
  rewritten graph; any later error is returned and is never retried through a
  second path. The proof variable remains as a positive selection assertion.
- Exactness evidence: the host process matrix is byte-identical for
  below-placement fallback, forced stream failure, forced standard success,
  complete metadata output, missing-input errors, and an unsupported graph.
  The hash-pinned 21.97M physical v3 differential passes in 281.94 seconds and
  compares ordinary resident, direct resident, automatic public, and pinned
  upstream artifacts. It also proves existing-output and injected-preflight
  refusals preserve the target and create no temporary output.
- Performance evidence: on clean `62016556b`, one warmup and five alternating
  option-free samples produce a 3.723975123-second candidate median
  (3.699833391–3.765550917) versus 35.738760555 seconds pinned PDAL
  (35.569560594–36.521455654), or 9.596939x. No `PDG_*` selector is present and
  the report records `automatic_resident_las_output_required=false`. Every
  sample produces the exact 144,000,375-byte SHA-256
  `e68b4dded6f5571d8eb45afad9fa5698b4a034d56f7083d39b3ab8002fa80a3a`.
- Decision: add LOF to the automatically selected count only inside this
  explicit resident/default-LAS shape. It is functionally supported and
  GPU-native in its bounded shared-kNN work, performance-qualified for the
  named 4M complete pipeline, and automatically selected when the existing
  measured runtime placement/preflight envelope accepts. Canonical LAS
  translation/overlay and exact ambiguous-row closure remain host work.
- Next task: B0044 runs one cheap exact 4M direct-publication sample for each
  already-positive resident `approximatecoplanar`, `normal`, and `nndistance`
  producer followed by one `UserData` operation. Choose at most one clean
  qualification target from those measurements; do not generalize B0043's
  string/option envelope or start a new stage port first.
- Consequences: eight filters now have an accepted automatic CUDA envelope.
  This is endpoint integration of already-measured resident work, not a new
  kernel, a general automatic resident planner, or catalog completion. The
  deeper device-column sink remains deferred by B0041's 2.5% spill ceiling.

## D0107 — Advance nndistance from the reusable resident-endpoint sweep

- Date: 2026-08-10
- Status: accepted as a cheap viability decision only; no automatic selector
  or new performance qualification follows yet.
- Context: after B0043 promotes the measured LOF endpoint, the plan calls for
  reusable producer/point-operation fusion rather than another stage port.
  B0017/B0018 show positive resident coplanar, normal, and nndistance compute at
  21.97M points, but do not measure their direct default-LAS publisher on the
  current path. B0044 predeclares one exact 4M sample per shape and at most one
  qualification target.
- Evidence: on clean `e53f5ab43`, direct resident coplanar takes 3.460689536
  seconds versus 17.069061712 pinned PDAL (4.932272x), normal takes
  3.708932102 versus 18.078695047 seconds (4.874367x), and nndistance takes
  3.362747326 versus 17.738831773 seconds (5.275101x). Every candidate reports
  `planner_resident_shared_index_direct_las`, one shared index, exact output,
  and no force/selection claim beyond the explicit resident proof.
- Profile: nndistance has the lowest candidate wall and largest ratio. A
  separate direct-resident stats capture has no ambiguous, incomplete, or
  repaired rows. Upload pack is 0.087469810 seconds; spill wait/publication is
  0.001022491/0.036333192 seconds. The 18-launch basic Nsight Compute report
  measures `knnGatherKernel` at 138.57 milliseconds, 28.80% SM and 0.52% DRAM;
  all other device kernels are sub-millisecond. Thus about 3.10 seconds of the
  3.363-second candidate lies in host-side reader/control/index preparation
  and canonical publication, not a device-kernel bottleneck.
- Decision: advance only the exact explicit
  `nndistance(k=10) -> UserData = 1 WHERE NNDistance >= 0.4 -> default LAS`
  shape to B0045. Retain coplanar and normal as measured positive hypotheses;
  do not infer automatic selection from their one-shot results. The maintained
  B0045 template is `bench/pipelines/resident-nndistance.json`, SHA-256
  `99d46835c21e8426569ed129e9618811a47035f33ec585d1c3be3dc9cf2c956a`.
- Next task: B0045 runs a clean repeated direct-endpoint qualification and then
  adds B0043's fail-closed automatic boundary only if the result stays exact
  and positive. Its tests must cover below-placement, CLI options, errors,
  output state, positive physical selection, and no retry after commitment.
- Consequences: the sweep satisfies the prototype-first rule and rejects a
  kernel optimization as the next action. Automatic count remains eight,
  catalog ports remain paused, and no coplanar/normal coverage label changes.

## D0108 — Select the qualified exact nndistance resident LAS endpoint

- Date: 2026-08-10
- Status: accepted for one explicit public shape and the already-measured
  runtime placement envelope; no other nndistance option or B0044 candidate is
  promoted.
- Context: D0107 selects only
  `nndistance(k=10) -> UserData = 1 WHERE NNDistance >= 0.4 -> default LAS`
  for repeated qualification. Its device profile measures the dominant
  `knnGatherKernel` at only 138.57 milliseconds and the existing upload/spill
  timers at 0.125 seconds, so qualification and public integration precede any
  optimization proposal.
- Qualification evidence: on clean `d69e37da0`, one warmup and five
  alternating frozen-time samples of the maintained explicit-direct template
  take a 3.339251347-second PDG median (3.329857166–3.487751068) versus
  17.705276847 seconds pinned PDAL, or 5.302170x. All samples produce the exact
  144,000,375-byte SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
  The report SHA-256 is
  `9b90ab1b384a1b0d59254465d29e1a2d43f90ddc702cd77cb6491e7bc617e609`.
- Decision: extend D0106's side-effect-free automatic admission with only a
  second exact four-stage JSON shape: explicit default `readers.las`, explicit
  `filters.nndistance,k=10`, the exact conditional assignment above, and an
  explicit default `writers.las`. The same input/output, device/profile,
  measured placement, nonempty rewrite, direct-publication, and resident
  preflight gates must accept before execution commits. Static or preflight
  refusal delegates the untouched command; an error after commit is terminal
  and is never retried.
- Exactness evidence: the focused host suite passes 50 C++ tests plus process
  comparisons for below-placement fallback, required-proof refusal with no
  output, and unsupported `mode=avg` fallback. The expanded hash-pinned 21.97M
  physical v2 differential passes in 236.60 seconds and compares the new
  option-free public path byte-for-byte with pinned PDAL.
- Public performance evidence: on clean `fba6b16ba`, one warmup and five
  alternating option-free samples take 3.390338813 seconds
  (3.323764771–3.517354436) versus 17.726674770 seconds pinned PDAL, or
  5.228585x. No selection proof variable is present. The exact output hash is
  unchanged; the report SHA-256 is
  `f424aa5f7227cc43cba7e054f1fb563b0e1673984b1db9cdb69c273f37661a36`.
- Consequences: nndistance becomes the ninth filter with an accepted automatic
  CUDA envelope. It is functionally supported generally through exact CUDA or
  pinned fallback, GPU-native only in its bounded shared-kNN work,
  performance-qualified only for this named 4M complete pipeline, and
  automatically selected only for the exact public shape above. Canonical LAS
  publication remains host work. Coplanar, normal, other nndistance options,
  and other graph shapes remain explicit or host-selected.
- Next task: B0046 adds stats-only attribution around validation/placement,
  rewritten manager execution, and canonical LAS publication for exact 4M
  nndistance and LOF runs. Proceed only if it identifies a reusable material
  component; do not optimize the 138.57-millisecond kernel or resume stage
  ports first.

## D0109 — Localize resident endpoint wall before optimizing

- Date: 2026-08-10
- Status: accepted as a measurement decision; no optimization, selection, or
  coverage envelope changes.
- Context: B0044 measures only about 0.264 seconds in nndistance's dominant
  kernel plus upload/spill phases, leaving roughly 3.10 seconds aggregated as
  reader/control/index preparation and publication. That aggregate is too
  broad to justify a fix.
- Instrumentation: clean commit `455bced46` adds an additive stats-only
  `pipeline_phase_seconds` partition. It measures command work before stats,
  validation/placement/preflight, rewritten `PipelineManager` execution,
  canonical direct-LAS publication, and residual control. The timer is active
  only when `--stats` is requested and changes neither output nor placement.
  Host Debug and CUDA Release focused process tests pass; a bounded physical
  direct-resident smoke positively observes every applicable phase.
- Nndistance evidence: the clean exact 4M capture at `455bced46` takes
  3.235190260 seconds before stats. Rewritten manager execution is
  3.005455534 seconds (92.90%), validation/placement/preflight is 0.168271752,
  canonical publication is 0.051060884, and other control is 0.010402090
  seconds. B0044's corresponding upload/spill phases total 0.123229606 seconds
  in this capture and the retained NCU kernel is 0.13857 seconds. Subtracting
  both leaves 2.743655928 seconds inside the combined reader, row-table,
  planner-index, and resident-wrapper execution. The output is byte-identical
  to pinned PDAL at SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`;
  stats SHA-256 is
  `2ac15019d4ee190491889cabbcda12f16d76e33652ee898585aa1f9ae0e50aaa`.
- LOF evidence: the matching clean exact capture takes 3.735417920 seconds
  before stats, of which 3.503516383 seconds (93.79%) is manager execution.
  Validation is 0.169039318, publication 0.051880115, and other control
  0.010982104 seconds. Exact host repair is 2.287512704 seconds for 40,982
  rows; upload/spill phases total 0.221092861 seconds. The post-boundary,
  post-repair manager remainder is 0.994910819 seconds before subtracting the
  previously profiled device work. Output SHA-256 is the pinned
  `e68b4dded6f5571d8eb45afad9fa5698b4a034d56f7083d39b3ab8002fa80a3a`;
  stats SHA-256 is
  `def5cfbb7f14230cc4a8db7fe0768924690736705894bd3f7d9e965eea0b5d72`.
- Decision: reject canonical LAS publication as the next target; its 51–52
  milliseconds is only 1.39–1.58% of measured internal wall. Retain exact LOF
  host repair and the nndistance kernel unchanged. The only reusable material
  unknown is inside rewritten manager execution, especially nndistance's
  2.744-second remainder.
- Next task: B0047 splits reader/row-table materialization from resident
  wrapper/index work with non-overlapping stats-only timers, or an equivalent
  bounded CPU profile if such a profiler becomes available. Repeat the same
  exact 4M nndistance and LOF shapes, and proceed only if one shared component
  has a material complete-pipeline ceiling. New stage ports remain paused.

## D0110 — Keep attribution inside the resident interval

- Date: 2026-08-10
- Status: accepted as a measurement decision; selection, compatibility, and
  coverage categories are unchanged.
- Context: D0109 puts more than 92% of both qualified endpoint captures inside
  rewritten manager execution but cannot distinguish reader/row-table work
  from planner-owned resident index/filter/bridge work. Individual filter-body
  timers were rejected before build because they would omit PDAL stage control
  and the assignment/boundary lifecycle.
- Instrumentation: clean commit `6cfedcc29` arms one stats-only timeline only
  for successful direct, single-region shared-index manager execution. The
  manager start and execute start bracket graph construction and preparation;
  the first accepted upload marker ends reader/row-table materialization plus
  initial `Stage::execute` dispatch/finalization; the completed spill ends
  resident wrapper/index/filter work; execute return ends post-spill control.
  The four spans are mutually exclusive and reconcile to the existing manager
  timer. Other host, multi-region, and independent direct executors report a
  null breakdown. CUDA Release and Host Debug focused process tests pass. The
  21.97M physical v2 gate additionally runs required-direct nndistance with
  stats, compares its bytes to pinned PDAL, and positively proves the fields,
  one planner-owned index build, and exact reconciliation; it passes in
  257.58 seconds.
- Nndistance evidence: the clean exact 4M capture splits its
  3.027606651-second manager interval into 0.000559325 seconds graph/prepare,
  0.502592322 reader/row-table materialization, 2.524453274 resident
  upload-through-spill work, and 0.000001730 post-spill control. The resident
  span is 83.38% of manager wall and 77.43% of total internal pre-stats wall.
  Its nested upload/spill phases total 0.129679471 seconds; subtracting those
  and B0044's 0.13857-second dominant kernel leaves 2.256203803 seconds still
  aggregated inside planner-index/filter/bridge work. Output remains the
  pinned SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`;
  stats SHA-256 is
  `1b5e5d5b66ea52fa14c4ca839252f1c5a48fd5159ed99072d76e91ebae5db499`.
- LOF evidence: the matching manager interval is 3.411599682 seconds:
  0.000547915 graph/prepare, 0.509743845 reader/table, 2.901305423 resident,
  and 0.000002499 post-spill seconds. Exact host repair takes 2.294764896
  seconds, or 79.09% of the resident interval, and the nested boundary phases
  total 0.228203354 seconds. Output remains pinned at
  `e68b4dded6f5571d8eb45afad9fa5698b4a034d56f7083d39b3ab8002fa80a3a`;
  stats SHA-256 is
  `1f816a8dd889bc95c6512f54c4bccc6074f220e09f9c8c08a3f78e0e05c31ca9`.
- Decision: reader/row-table materialization is a reusable 0.50–0.51-second
  cost but is not the dominant surface, and no reader optimization follows.
  Retain exact LOF repair. Nndistance's 2.256-second post-boundary/post-kernel
  resident remainder is now the only material unresolved component.
- Next task: B0048 adds nested, stats-only attribution for planner-index
  configuration/build, nndistance query/projection, the adjacent point-program
  bridge, and boundary work, reusing the same surface for LOF where possible.
  Proceed only if one reusable subphase has a material complete-pipeline
  ceiling; ports remain paused.

## D0111 — Treat the NND query wrapper as the only unresolved material surface

- Date: 2026-08-10
- Status: accepted as a measurement decision; selection, compatibility, and
  all four coverage categories remain unchanged.
- Instrumentation: clean commit `3af936e2e` adds four stats-only host-call
  wall spans inside B0047's direct single-region resident interval: index
  configuration, index build, neighborhood query/projection, and adjacent
  point-program bridge. They are mutually exclusive host spans, not exclusive
  CUDA timings. Existing boundary and repair counters remain nested
  drill-downs inside the outer interval and are not added to them. In
  particular, queued index work may complete during the later query sync, and
  LOF repair is part of its broad query span. Other executors report null.
- Nndistance evidence: the final-binary exact 4M capture splits its
  2.506743228-second resident interval into 0.121900349 seconds index
  configuration, 0.008554448 index-build call wall, 2.233622069 broad
  query/projection, 0.035821079 adjacent assignment bridge, and 0.106845283
  residual seconds. Query/projection is 89.10% of resident wall and reports
  zero repair rows. Nested upload/spill phases total 0.123576032 seconds.
  Output stays pinned at
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`;
  stats SHA-256 is
  `6f7f24434d05f31d4d79d488d2fa1a502ef977dc3fbf704f770d14b3d3c9da77`.
- LOF evidence: its 2.882876630-second resident interval contains
  0.120797730 seconds configuration, 0.009001124 build call wall,
  2.597038882 query/projection, 0.037615942 bridge, and 0.118422952 residual
  seconds. Exact host repair is 2.279198769 seconds for the established 12,716
  ambiguous, one incomplete, and 40,982 repaired rows, so it explains 87.76%
  of the broad query span. Output stays pinned at
  `e68b4dded6f5571d8eb45afad9fa5698b4a034d56f7083d39b3ab8002fa80a3a`;
  stats SHA-256 is
  `fa54c74873819c3c0084a3cc9c306cc1f89d43580d6c2defc1d1d2c42beb690c`.
- Decision: reject index construction, the adjacent bridge, reader/table work,
  and canonical publication as next optimization targets. Keep exact LOF
  repair. Nndistance's broad query/projection call is the only unresolved
  material wall surface.
- Next task: B0049 splits NND output materialization/allocation, query
  submission, result/status transfer calls, explicit wait, status scan/repair,
  and PointView publication without introducing a synchronization. The
  pageable `std::vector<uint8_t>` status destination may make the nominally
  asynchronous D2H call absorb queued work, but that is an explicit unproven
  hypothesis until B0049 measures it. Ports remain paused.

## D0112 — Repair only exceptional NND rows; reject pinned status and full BVH

- Date: 2026-08-10
- Status: accepted as a measurement decision; it supersedes D0107/D0111's
  zero-NND-repair inference but changes no compatibility, coverage, placement,
  or automatic-selection boundary.
- Correction: NND performed exact host repair for incomplete rows but did not
  update the shared repair timer/counters. Therefore B0044/B0048's zero-row
  reports were missing telemetry rather than proof that no repair occurred.
  Commit `20d9ffabc07a01f60c5c53b2178637fe5377ba18` adds stats-only NND
  repair accounting plus nested host call-wall spans for output preparation,
  query submission, status allocation, result/status transfer calls, explicit
  wait, status scan/repair, and output publication. These are not exclusive
  CUDA timings; queued device work may complete during a later call.
- Exact evidence: the final frozen-time 4M `nndistance(k=10) -> assign -> LAS`
  capture takes 3.278321613 seconds before stats. Its 2.253729404-second broad
  query contains 0.032393288 seconds output preparation, 0.000110899 query
  submission, 0.000701313 status allocation, 0.000007600 result-transfer call,
  0.138355024 status-transfer call, 0.000732083 explicit wait, 2.061544515
  status scan/repair, 0.019728244 output publication, and 0.000156438 residual.
  One incomplete row causes 2.061175139 seconds of exact compatibility KD3
  build, recomputation, and device-column refresh. Output remains byte-equal
  to pinned PDAL at SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`;
  retained stats SHA-256 is
  `f42abe12ac97aa68ae75b7e652ff685e6b84e83603b10437263a2f87c0a3d4ab`.
- Rejected prototypes: pageable status copy is only 6.14% of the broad query,
  so pinned status storage cannot remove the dominant wall. Forcing the
  existing planner-owned Morton BVH eliminates incomplete repair, but shifts
  queued query work into a 3.060009216-second status-transfer call and slows
  complete candidate wall to 4.179201213 seconds, 27.48% behind the grid plus
  exact repair. Its output remains exact; retained prototype stats SHA-256 is
  `b412f8abd6568b44d62746f88cb54a3da0909b967204fea8666f05585d6790e5`.
- Validation: Host Debug and CUDA Release compile; the focused host process
  test passes in both builds; a physical four-point one-shell CUDA regression
  proves exact positive incomplete repair with stats and zero telemetry
  mutation without stats; the 21.97M physical v2 gate passes in 258.44 seconds;
  and the final frozen 4M output matches pinned PDAL byte-for-byte.
- Decision: retain the qualified grid path and its exact host fallback. B0050
  may cheaply prototype a planner-composable exact device full scan for only
  incomplete NND rows using already-resident coordinates. Proceed beyond the
  prototype only if complete frozen 4M wall materially improves with identical
  bytes. Do not build a private index, widen selection, start calibration, or
  resume stage ports first.

## D0113 — Select bounded exact NND repair from resident coordinates

- Date: 2026-08-10
- Status: accepted for the existing automatic NND endpoint; compatibility,
  functional support, and the public pipeline envelope are unchanged.
- Cheap prototype gate: a one-thread device full scan reproduced the frozen
  4M output but took 4.291492454 seconds complete and 3.083769298 seconds in
  repair, so it was rejected. Giving each incomplete row one 64-thread block
  and exactly merging the union of thread-local top-k sets reduced the same
  one-shot complete wall to about 1.31 seconds with identical bytes, so the
  direction advanced.
- Exact envelope: kth mode only, internal neighbor count 2..16 (public
  `k=1..15`), and at most 16 incomplete rows. Each block scans the existing
  planner-owned resident coordinate columns with the exact binary64 operation
  sequence and retains `(distance, point-id)` pairs. The kth distance is
  independent of tied candidate identity. The path creates no spatial index;
  average mode, larger k, larger repair sets, and an explicit disable retain
  upstream KD3 repair. `PDG_REQUIRE_NND_DEVICE_REPAIR` is a positive proof and
  fails for zero or ineligible repairs.
- Telemetry: `exact_host_repair` now contains only host repair seconds and
  rows. `exact_device_repair` separately reports device seconds and
  incomplete/repaired rows; stats-disabled execution mutates neither. This
  avoids representing a device repair as zero-second host work.
- Same-machine qualification: clean commit
  `abb460846448012f82fffdd852bde11690c0182e`, Ryzen 9 7900, RTX 4090/SM 89,
  CUDA 13.3 Release, pinned upstream
  `f1e35f5c3e416b8c6a2c39966c8205cfae54afe2`, and the established 4M ALS
  prefix. One warmup plus five alternating measured samples of the proof-gated
  option-free automatic pipeline give 17.809985276 seconds pinned PDAL and
  1.413364609 seconds PDG, or 12.601125826x. This is a 2.398771549x candidate
  improvement and 58.311995% wall reduction from B0045's
  3.390338813-second automatic median. Every output is 144,000,375 bytes at
  SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
- Post-change profile: the stats-enabled exact capture reports one device
  repair in 0.077210784 seconds and zero host repair. Its 1.047846107-second
  manager interval is 0.492926525 seconds reader/row-table materialization and
  0.554462427 seconds resident wrapper. The resident wrapper contains
  0.270416565 seconds broad query, 0.126630460 seconds index configuration,
  0.008901063 seconds index-build call wall, 0.037118585 seconds adjacent
  bridge, and 0.111395754 seconds residual. Reader/table work is now the
  largest reusable single phase; D0110's earlier ordering is superseded only
  for this improved endpoint.
- Validation: Host Debug and CUDA Release builds and focused process tests
  pass. Exact positive tests cover k=1/k=15, awkward large offsets, an equal
  kth-distance boundary, the one-shell fixture, and stats-disabled execution.
  Fail-closed tests cover k=16, average mode, more than 16 incomplete rows,
  require-plus-disable, and zero incomplete rows. Compute Sanitizer memcheck,
  initcheck, racecheck, and synccheck are clean. The hash-pinned 21.97M
  physical v2 lifecycle gate passes in 230.17 seconds.
- Decision: select bounded device repair inside the already-qualified
  B0045 NND/default-LAS public shape; retain exact KD3 fallback outside the
  bound. The four coverage categories remain independent: this improves the
  existing performance qualification and automatic execution but adds no
  stage or option envelope. B0051 next cheaply tests a reusable direct-LAS
  resident-source hypothesis against the now-dominant reader/table plus upload
  cost before any production design or stage port.

## D0114 — Advance direct LAS resident hydration past the cheap prototype gate

- Date: 2026-08-10
- Status: accepted as a prototype-direction decision only; there is no new
  public envelope, performance claim, or automatic-selection change.
- Measured hypothesis: B0050 leaves 0.492926525 seconds in PDAL
  reader/row-table materialization and 0.089441017 seconds in resident XYZ
  upload packing. A disposable probe mapped the established 4,000,000-point,
  144,000,000-record-byte LAS image, reused the existing LAS coordinate
  transpose, expanded its three int32 columns to exact binary64 resident
  coordinates, and copied no data back on the timed path.
- Result: after one warmup, five complete map-image-to-device-column
  hydrations were 0.00782511, 0.00798922, 0.00840182, 0.00861958, and
  0.00876669 seconds; median was 0.00840182 seconds. File mapping plus LAS
  validation was 0.000028979 seconds. Fresh device allocator construction and
  all temporary/column allocations took 0.0986952 seconds separately. A
  post-timing comparison proved all 12,000,000 decoded coordinate values equal
  to the host `CoordinateEncoding::decode` result.
- Interpretation: the prototype is not a complete-process benchmark and does
  not imply that B0050's 1.413364609-second median will fall by the full
  0.582367542-second ceiling. It does establish ample directional margin even
  when the one-time allocation cost is charged. The production design must
  retain mapped source bytes for exact output overlay and preserve the
  unchanged PDAL fallback before any input or output side effect.
- Decision: advance one bounded production slice for the already-qualified
  default-LAS resident NND endpoint. Reuse the existing mmap, LAS validation,
  transpose, planner-owned columns, and output overlay; do not introduce a
  second parser, widen reader/stage coverage, resume ports, or alter automatic
  selection until a full alternating byte-exact benchmark proves value.

## D0115 — Select the bounded direct LAS source for the qualified NND endpoint

- Date: 2026-08-10
- Status: accepted for the existing automatic default-LAS NND endpoint; no
  stage, option, or functional-coverage envelope is added.
- Exact design: commit `8c8ccb523ed123f0895b18fe5642b392137adf30`
  maps the already-validated default LAS input through the shared no-follow
  mapping, reuses `FileView`, the existing coordinate transpose, and exact
  separate multiply/add expansion, and hydrates the planner-owned resident
  binary64 XYZ columns and spatial index. An internal reader supplies stable
  point identity only. Its sparse compatibility table retains original
  UserData and NNDistance and decodes XYZ on demand for the unchanged exact
  fallback. The canonical writer still publishes by overlaying final UserData
  on the mapped input. No private index or second LAS parser exists.
- Fail-closed envelope: only the already-qualified exact four-stage
  `readers.las -> nndistance(k=10) -> UserData assign -> writers.las` JSON
  shape may select the source. Mapping, header, count, rewrite, and resident
  consumption are proved before or during the committed execution. Explicit
  disable retains the old PointView path; unsupported inputs decline before
  side effects. On the automatic/default `pdg pipeline` path,
  require-plus-disable and any unmet source require return status 124 instead
  of silently benchmarking fallback; explicit `pdg resident` reports its
  ordinary execution error status.
- Same-machine qualification: clean commit `8c8ccb523`, Ryzen 9 7900, RTX
  4090/SM 89, CUDA 13.3 Release, pinned upstream
  `f1e35f5c3e416b8c6a2c39966c8205cfae54afe2`, and the established 4M ALS
  prefix. One warmup plus five alternating proof-gated default-command samples
  give 17.815840889 seconds pinned PDAL and 0.911029970 seconds PDG, or
  19.555713287x. This is 1.551392002x faster than B0050 and removes
  35.541759% of its candidate wall. All outputs are 144,000,375 bytes at
  SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
- New profile: the exact stats capture reports 0.802835564 seconds before
  stats: 0.169932975 validation/placement/preflight, 0.580441540 rewritten
  manager execution, 0.050543367 canonical publication, and 0.001917682 other
  control. Manager wall contains 0.045625654 seconds sparse reader/table work
  and 0.526276268 resident wrapper. The wrapper contains 0.120758605 index
  configuration, 0.002581176 index-build call wall, 0.276098990 broad query,
  0.052368779 assignment bridge, and 0.074468718 unattributed wall. The broad
  query includes 0.139371638 seconds in the status-transfer call, which may
  absorb queued kernels, plus 0.075863885 status scan/device repair. It is not
  yet an exclusive kernel profile.
- Validation: Host Debug and CUDA Release build; focused process and physical
  exact tests pass; the source changes UserData on a threshold-positive 4M
  fixture while matching the pinned artifact exactly. Compute Sanitizer
  memcheck, initcheck, synccheck, and racecheck report zero errors/hazards over
  expansion and hydration. The hash-pinned 21.97M physical v2 differential and
  lifecycle gate passes in 232.49 seconds, including source-required and
  source-disabled paths.
- Decision: automatically select the direct source only inside the existing
  qualified endpoint. This improves its performance qualification and
  automatic implementation without changing functional support or GPU-native
  stage coverage. B0053 profiles the final path and compares the actionable
  query ceiling with validation/preflight and index configuration before any
  prototype. New stage ports remain paused.

## D0116 — Reject kNN launch tuning after profiling the final selected path

- Date: 2026-08-10
- Status: accepted as a measurement and negative-prototype decision; code,
  compatibility, selection, and all four coverage categories are unchanged.
- Final-path profile: clean commit `4af574307`, the B0052 4M input, and the
  exact automatic endpoint with direct source and bounded repair required.
  Nsight Compute records 21 launches and 218.520992 milliseconds total kernel
  time. `knnGatherKernel` takes 139.479840 milliseconds; the one-row
  `repairIncompleteKthDistanceKernel` takes 77.548288 milliseconds; every
  other launch together takes 1.492864 milliseconds. The output remains
  144,000,375 bytes at the established SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
- Profile interpretation: the gather uses 64 threads, 64 registers per thread,
  59.90% achieved occupancy, 29.06% compute throughput, 45.16% memory
  throughput, and only 0.51% DRAM throughput. The repair is intentionally one
  block for one incomplete row and reports 4.17% achieved occupancy. The broad
  0.276098990-second host query interval is therefore principally real queued
  gather/repair work rather than a pageable-status transfer bottleneck.
- Cheap reusable prototype: 64-thread complete-process samples were
  0.92/0.85/0.84 seconds; 128-thread samples were 0.90/0.82/0.82; and
  256-thread samples were 0.92/0.83/0.83. These are directional warm-cache
  timings, not a qualification lane. One-pass kernel durations were
  137.899776 milliseconds at 128 threads and 134.532928 milliseconds at 256,
  versus 139.479840 at 64. The best valid result saves 4.946912 milliseconds,
  only 3.546686% of the kernel and about 0.543% of B0052 median wall. A
  512-thread launch returns `cudaErrorInvalidValue` before output; 1024 is not
  attempted. Every valid output is exact.
- Decision: reject launch-size tuning as immaterial and restore/rebuild the
  qualified 64-thread kernel. Do not optimize the one-row data-dependent
  exception ahead of larger reusable host surfaces. B0054 partitions the
  measured 0.169932975-second validation/placement/preflight aggregate into
  original validation, runtime placement, and rewritten resident preflight
  before choosing a prototype. Ports remain paused.

## D0117 — Treat resident placement wall as necessary CUDA process startup

- Date: 2026-08-10
- Status: accepted as a measurement and negative-optimization decision;
  compatibility, placement, automatic selection, and all four coverage
  categories are unchanged.
- Evidence: stats-only instrumentation committed at `8aaac57ff` adds no CUDA
  synchronization and exactly reconciles nested host intervals. On the exact
  B0052 4M automatic endpoint, the fresh validation/placement/preflight total
  is 0.175984958 seconds: 0.002231646 seconds plan/original validation,
  0.173185798 seconds runtime placement, and 0.000567514 seconds rewritten
  source/resident preflight. Runtime placement contains 0.078605269 seconds
  device/profile discovery, 0.094578699 seconds initial memory-budget
  placement, and 0.000001830 seconds executor selection. The output remains
  144,000,375 bytes at SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
- Interpretation: original validation and resident preflight together consume
  only 2.799160 milliseconds. The material interval initializes the CUDA
  device/profile and allocator budget needed before runtime placement; moving
  it later would charge the same process startup to the first allocation.
  Therefore B0054 retains the additive observability but no optimization
  prototype and makes no speedup claim.
- Next gate: B0055 partitions the measured 0.120758605-second planner-owned
  index-configuration interval into adaptive configuration and exact-envelope
  validation. A one-pass/precomputed-extrema prototype proceeds only if a
  repeated scan is measured material, without trusting unvalidated LAS header
  bounds or weakening generic/forced-backend fallback.

## D0118 — Make complete-coordinate kNN configuration proof authoritative

- Date: 2026-08-10
- Status: accepted exact reusable optimization; functional support,
  GPU-native coverage, and automatic selection are unchanged. The already
  qualified NND endpoint's performance qualification improves.
- Measured cause: B0055's pre-change exact stats capture splits a
  0.123943417-second planner-owned index-configuration interval into
  0.051803244 seconds selecting the adaptive config, 0.072138563 seconds
  rescanning the complete host envelope, and 0.000001610 seconds timer/control
  residual. Every kNN builder has already scanned every host coordinate for
  finiteness and extrema before returning its grid or Morton-BVH config; the
  second scan proves the same generated envelope again.
- Decision: document the kNN selector's complete-coordinate proof and remove
  the redundant post-selection scan from resident kNN construction and the
  standalone statistical-outlier caller. Radius-index construction still
  performs its independent envelope validation. No LAS-header bound is
  trusted, the deterministic adaptive probe and forced grid/BVH switches are
  unchanged, and malformed/nonfinite inputs still fail at configuration.
- Result: clean commit `aca65d75e` reports 0.051821472 seconds final index
  configuration with zero second-scan time. Five alternating exact automatic
  samples measure 0.821465859 seconds candidate versus 17.895370391 seconds
  pinned PDAL, or 21.784679x. This removes 9.831083% of B0052 candidate wall.
  Complete output remains 144,000,375 bytes at SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
- Proof: grid/BVH/adaptive host-envelope tests, focused physical CUDA NND and
  mixed resident differentials, the hash-pinned 16M direct process lane, and
  Compute Sanitizer memcheck, initcheck, synccheck, and racecheck all pass.
  B0056 may prototype one shared extrema pass inside the remaining measured
  0.051820422-second configuration path; ports remain paused.

## D0119 — Reject kNN extrema reuse without complete-pipeline gain

- Date: 2026-08-10
- Status: accepted negative-prototype decision; the disposable source and test
  edits are fully reverted. Compatibility, performance qualification,
  automatic selection, and all coverage categories remain at B0055.
- Prototype: retain the finite extrema found by `makeKnnGridConfig` and
  `makeMortonBvhConfig` instead of calling the uniform-grid builder for a
  second extrema pass. Focused tests compare the resulting origin and maximum
  cells with a fresh full rescan for both dimensions and all three builders;
  every frame and the complete output remain exact.
- Result: the exact stats capture reduces index configuration from
  0.051821472 to 0.026006785 seconds and pre-stats command wall from
  0.737485416 to 0.711166768 seconds. However, the required cheap option-free
  process gate measures 0.834/0.825/0.844 seconds, which does not materially
  improve on B0055's 0.821465859-second qualified median. No alternating
  certification lane is justified.
- Decision: restore and rebuild the qualified B0055 implementation. B0057
  partitions the remaining 0.083464418-second resident-wrapper residual around
  direct-LAS hydration and setup. Only a material measured transfer surface
  may justify a lazy-host/device-summary prototype; LAS header bounds remain
  untrusted and exact host repair remains mandatory.

## D0120 — Localize the remaining wrapper residual to direct-source hydration

- Date: 2026-08-10
- Status: accepted measurement; compatibility, code selection, performance
  qualification, and all coverage categories remain unchanged.
- Evidence: stats-only commit `ff3a55044` adds no synchronization and splits
  the direct resident wrapper's previously unattributed wall. On the exact 4M
  B0055 endpoint, resident product setup takes 0.000147437 seconds and
  direct-LAS coordinate hydration takes 0.058526544 seconds. The residual
  falls from B0055's 0.083464418 to 0.023428953 seconds within a
  0.462337790-second wrapper. Output remains 144,000,375 bytes at SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
- Interpretation: hydration is material, while creation of the empty resident
  host/device products is not. The current span still combines pinned/device
  column and temporary allocation, mapped-record H2D, decode/expand launch
  calls, three XYZ D2H calls, and the existing final stream wait. B0053's tiny
  non-query kernels are suggestive but do not attribute fresh allocation or
  queued transfer wall, so lazy host hydration is not yet authorized.
- Next gate: B0058 partitions hydration into allocation/materialization,
  transfer/kernel submission, and final wait without a new synchronization.
  Only a material transfer/wait result may justify a record-derived device
  summary plus lazy host coordinates, and exact host repair must remain able to
  read the original mapped points.

## D0121 — Reject D2H-only hydration tuning; test combined record summary

- Date: 2026-08-10
- Status: accepted measurement and negative narrow-path decision;
  compatibility, performance qualification, automatic selection, and all
  coverage categories are unchanged.
- Evidence: nested stats-only commit `0a943a359` adds no synchronization. The
  exact 4M direct-source hydration interval is 0.055183624 seconds:
  0.042686939 seconds validation/materialization/allocation, 0.008537849
  seconds transfer/kernel submission, 0.003948276 seconds final wait, and
  0.000010560 seconds timer/control residual. The output remains 144,000,375
  bytes at SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
- Decision: reject a D2H-call/wait-only optimization; its measured ceiling is
  about 12.49 milliseconds before complete-process noise and cannot justify a
  new summary/fallback design. Allocation dominates hydration, and the same
  path still spends 0.062050020 seconds scanning mirrored host XYZ for config.
- Next gate: B0059 cheaply prototypes a point-record-derived extrema and
  deterministic adaptive-sample summary without allocating the host XYZ
  mirror. Production work proceeds only if the combined allocation/mirror and
  config-scan removal is viable. Header bounds remain untrusted, device XYZ
  stays planner-owned, and exact PointView/KD3 repair remains available.

## D0122 — Advance the exact mapped-record summary to a bounded production gate

- Date: 2026-08-10
- Status: accepted directional prototype; compatibility, performance
  qualification, automatic selection, and all coverage categories are
  unchanged.
- Protocol: mmap each retained default-format LAS input, scan raw point-record
  XYZ integers, decode extrema with the established binary64 coordinate
  operation, and gather the same deterministic 8,192 adaptive samples used by
  the planner-owned kNN selector. Build the candidate frame/backend from that
  summary and compare every configuration field bit-for-bit with the existing
  full host-column builder. LAS header bounds are not used.
- Result: after one warmup, five 4M scans have a 0.00426759-second median and
  five 16M scans have a 0.0148655-second median. Both match the established
  origin, maximum cell, cell-size bits, and adaptive backend exactly, without
  allocating or mirroring host XYZ. The retained source is
  `build/benchmarks/b0059-record-summary-probe.cpp`, 9,060 bytes at SHA-256
  `8474eff9f47ac0abaff77ea4b4cde7ee0afa959c7e6d56702621b96bd947f0bd`.
- Decision: the result plausibly removes the measured host allocation/mirror
  and configuration scan together, so B0060 may productionize it only for the
  already-qualified direct default-LAS NND endpoint. Device XYZ remains
  planner-owned; exact host repair must read mapped PointView coordinates
  before pinned KD3 use. Clustered adaptive and forced Grid/BVH selection plus
  a positive no-host-mirror proof precede the cheap complete-process gate. No
  certification lane runs unless that gate shows material wall improvement.

## D0123 — Qualify direct mapped-record kNN configuration without host XYZ

- Date: 2026-08-10
- Status: accepted for the existing automatic default-LAS NND endpoint.
  Functional support and GPU-native coverage are unchanged; the endpoint's
  performance qualification improves and automatic selection does not widen.
- Design: commit `07ea2f4a7` scans exact decoded point-record coordinates and
  the shared deterministic adaptive sample mapping, then feeds a complete
  summary to the planner-owned Grid/Morton-BVH configuration builders. Device
  XYZ and the one shared index remain unchanged. No resident host XYZ columns
  are allocated or mirrored; the sparse compatibility table still decodes
  original mapped coordinates on demand before pinned PDAL KD3 repair. Header
  bounds are never trusted.
- Exact proof: ordinary adaptive, clustered adaptive Morton-BVH, forced Grid
  and forced Morton-BVH, malformed summaries, failed config/index proof
  latching, source-disabled automatic/explicit resident paths, and a
  device-repair-disabled mapped KD3 row all pass. The established 4M artifact
  remains 144,000,375 bytes at SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
  Compute Sanitizer memcheck, initcheck, synccheck, and racecheck report zero;
  the hash-pinned 16M v2 process lane passes in 178.31 seconds.
- Performance: one warmup plus five clean alternating proof-gated samples on
  the Ryzen 9 7900/RTX 4090 measure 0.755523193 seconds candidate versus
  17.891433883 seconds pinned PDAL, or 23.680853x. This removes 0.065942666
  seconds (8.02744%) from B0055 candidate wall. The clean phase capture reduces
  hydration from B0058's 0.055183624 to 0.027851630 seconds and configuration
  from 0.062050020 to 0.000476395 seconds.
- Next gate: B0061 tests only the measured 0.024526099-second NNDistance host
  publication before its immediate 0.050211615-second resident assignment
  bridge. It may advance only with exact sole-consumer/lifetime proof and a
  material cheap complete-process win.

## D0124 — Reject PointView-only NNDistance publication elision

- Date: 2026-08-10
- Status: rejected after the clean complete-process gate; prototype commit
  `cf5ff3344` is cleanly reverted by `0b61e8ef2`. Functional support,
  GPU-native coverage, performance qualification, and automatic selection are
  unchanged.
- Exact proof: the accepted direct-LAS four-stage envelope was the only caller
  allowed to skip publication. Every timed option-free process required
  automatic resident output, direct mapped source/summary, no host XYZ mirror,
  bounded device repair, skipped NNDistance PointView publication, and
  adjacent assignment reuse of the device column. An untimed resident proof
  captured both observed booleans. All outputs remained 144,000,375 bytes at
  SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
- Performance: five clean candidate samples were 0.750254638, 0.687457987,
  0.759552573, 0.684656764, and 0.754369576 seconds, with a 0.750254638-second
  median. This is only 0.005268555 seconds (0.697339%) below B0060's
  0.755523193-second median and the samples overlap materially. The change
  removes no transfer, so the result does not justify production retention or
  downstream certification.
- Next gate: B0062 tests the larger measured hypothesis—conditionally remove
  the initial 32 MB (32,000,000-byte) NNDistance upload, complete 32 MB result
  download, and eager table storage while keeping the forced exact host-repair
  path intact. It remains a prototype until an exact complete-process gate
  wins materially.

## D0125 — Qualify device-only NNDistance handoff in the direct endpoint

- Date: 2026-08-10
- Status: accepted for the existing automatic default-LAS NND-to-assignment
  endpoint. Functional support and GPU-native stage coverage are unchanged;
  the endpoint's performance qualification improves.
- Design: prototype commit `23cc26827` allocates NNDistance only in the
  planner-owned device batch for the exact four-stage direct-LAS envelope.
  Status transfers first. Zero or bounded exact device-repair rows remain on
  device through the adjacent assignment. A true host-repair requirement
  lazily allocates the full host column, copies all values D2H, runs pinned KD3
  repair, restores all values H2D, and publishes normally. Every other
  consumer/writer/fallback shape retains the ordinary path.
- Exact proof: normal execution positively observes device-only handoff and
  assignment device-column reuse with no full NNDistance transfer. Forced host
  repair positively observes full restoration in both directions before the
  same assignment. Both produce the established 144,000,375-byte LAS at
  SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
  Focused host/CUDA tests, the hash-pinned 16M process lane, memcheck,
  initcheck, synccheck, and racecheck pass. The retained combined attestation
  is `build/benchmarks/b0062-validation-manifest.json` at SHA-256
  `727d5f46bc39ff38a6bfe4fd334802067ad4c7bad349913124d52421fb4e757a`.
- Performance: one warmup plus five clean alternating proof-gated samples on
  the Ryzen 9 7900/RTX 4090 measure 0.712510475 seconds candidate versus
  17.777145188 seconds pinned PDAL, or 24.950012x. This removes 0.043012718
  seconds (5.693104%) from B0060 candidate wall, one 32,000,000-byte H2D, and
  32,000,008 bytes of D2H. The clean report is
  `build/benchmarks/b0062-device-only-nndistance-handoff.json`.
- Decision: promote the handoff automatically only inside the already-bounded
  direct endpoint. Do not generalize device-only output lifetime to other
  consumers or writers without their own liveness and exactness proof. B0063
  profiles the remaining 0.134898783-second status-call interval and separates
  queued kernel work from pageable-transfer blocking before any prototype.

## D0126 — Reject pinned status storage; target selective repair underfill

- Date: 2026-08-10
- Status: accepted profile attribution and negative narrow-path decision;
  compatibility, performance qualification, automatic selection, and all
  coverage categories remain unchanged.
- Evidence: the clean B0062 endpoint's Nsight Compute capture records 21
  launches and 216.430208 milliseconds total kernel time. The kNN gather takes
  137.153696 milliseconds, the one-row exact repair takes 77.754496
  milliseconds, and every other launch together takes 1.522016 milliseconds.
  The retained 1,514,389-byte report has SHA-256
  `6adc7790573b6281d19c978d965e568bf7064771ffc7bcd5be9ca69865960556`.
  Its provenance manifest is 3,680 bytes at SHA-256
  `70b0aa5abfa7eaef3465825a7b5ff77ba35bcb4bd8ff45fe33d83376673c47ff`.
- Decision: do not prototype pinned status storage. The prior
  0.134898783-second host interval includes synchronization with the queued
  gather, whose measured duration already accounts for the interval's
  magnitude. Treating that timer as pageable D2H time would optimize the wrong
  surface.
- Next gate: B0064 may cheaply prototype only the measured one-row repair
  underfill. Preserve the same candidate distances and exact kth-distance
  result, partition resident-coordinate scanning deterministically across
  multiple blocks, and merge fixed-order partial top-k sets on device. Keep
  the existing bounded admission and exact host repair. Do not certify or
  widen selection unless a complete exact process gate wins materially.

## D0127 — Promote bounded parallel repair and stop tuning the NND endpoint

- Date: 2026-08-10
- Status: accepted for B0062's existing exact automatic direct-LAS
  NND-to-assignment endpoint. Functional support, GPU-native stage coverage,
  and the automatic-selection envelope do not widen; this endpoint's
  performance qualification improves.
- Design: prototype commit `60f4af821` partitions each bounded incomplete
  query across at most 128 blocks over the planner-owned resident coordinates.
  Each block performs the same binary64 candidate-distance operations and
  `(distance, point id)` total order as the serial repair, then a fixed-order
  device merge emits the exact kth square root. Promotion commit `1aebd1877`
  makes this path default only when the already-proved device-only handoff is
  active. Generic resident NNDistance callers remain serial; test/debug
  opt-in and disable controls retain fail-closed required-path proofs.
- Exact proof: focused tests cover small boundaries, more than 64 partitions,
  the 128-partition cap plus grid-stride traversal at 16,385 points, generic
  serial retention, disable precedence, and required-path rejection. Every
  clean 4M output remains 144,000,375 bytes at SHA-256
  `174c1cb0a2a2e257dcf8c067f4699835eb1e95ebecea3c5bc0b445bfa4fd0407`.
  Memcheck, initcheck, and synccheck report zero errors, racecheck reports zero
  hazards, and the hash-pinned 16M v2 process gate passes in 182.84 seconds.
- Performance: one warmup plus five alternating exact samples measure
  0.623028204 seconds candidate versus 17.950882850 seconds pinned PDAL, or
  28.812312x. This is 12.558731% below B0062 candidate wall. A same-promoted-
  binary serial/parallel control measures medians of 0.652247040 and
  0.591965696 seconds, isolating a 9.242103% endpoint reduction. The clean
  report is `build/benchmarks/b0064-parallel-repair.json`; the combined
  provenance is `build/benchmarks/b0064-validation-manifest.json` at SHA-256
  `aa7516c2e682f486bcb0290e350a63ad19dad0f05eb38d5373c675400cc92df6`.
- Stop decision: the fresh promoted profile leaves 137.767712 of 141.519872
  kernel milliseconds (97.3487%) in the reusable gather, 2.232480 milliseconds
  in repair, and 1.519680 milliseconds elsewhere. B0053 already rejected its
  bounded launch-shape variants for lack of stable process gain. No clear
  reusable change is likely to add another 5--10% without a new algorithm, so
  this endpoint is sufficiently optimized for now.
- Next gate: B0065 resumes catalog work with stock-PDAL measurement and a
  profile of the already GPU-native but performance-unmeasured statistical
  `filters.outlier` lane. Test shared-index resident composition cheaply before
  authorizing any new port; proceed only for material complete-pipeline value.

## D0128 — Route statistical outlier toward residency, not a new kernel

- Date: 2026-08-10
- Status: accepted directional routing decision. Functional support,
  GPU-native coverage, performance qualification, and automatic selection are
  unchanged; the 1M result is not a published stage speedup.
- Evidence: B0065's rough exact 1M medians are 4.001562112 seconds pinned PDAL,
  0.620354816 seconds forced CUDA, and 0.278837248 seconds for pinned LAS
  read/write alone. Hardware-cycle samples identify nanoflann's per-point
  `searchLevel` as the dominant stock function. The collector interval warning
  makes those proportions directional, while the I/O subtraction independently
  places 93.031790% of stock wall above read/write.
- Device profile: the forced CUDA lane has only 21.557088 milliseconds across
  17 kernels. The kNN gather takes 21.257536 milliseconds; index construction
  plus every other launch takes 0.299552 milliseconds. Device work is only
  3.474961% of candidate process wall, so another standalone kernel is the
  wrong next investment.
- Decision: prototype planner residency around the existing exact statistical
  work. Reuse resident coordinates and the planner-owned shared index, retain
  pinned incomplete-row repair and the host global recurrence/classification,
  and target the private gather/upload/index boundary. Prove composition with
  an adjacent compatible kNN consumer rather than optimizing the already-small
  index kernels.
- Gate: B0066 remains disposable until a complete exact 1M composition removes
  a material boundary. Only then add differential expansion, sanitizers,
  calibration, or accepted performance evidence. The retained directional
  manifest is `build/benchmarks/b0065-outlier-directional-manifest.json` at
  SHA-256
  `db00ddb0db4b821b8ec33d95da0d64e07a1b20f1a20f6c5edc9d83a9f0815725`.

## D0129 — Retain the statistical-outlier resident prototype

- Date: 2026-08-10
- Status: accepted as GPU-native composition infrastructure and authorization
  for a clean qualification gate. Functional support is unchanged; no
  performance qualification or automatic selection follows yet.
- Design: statistical outlier may now execute inside a planner-owned 3D kNN
  region. It uses the existing exact `knnMeanDistances` query, repairs only
  incomplete rows with upstream KD3 ordering, preserves the pinned global
  mean/variance recurrence and warning behavior on host, and keeps a retained
  Classification column canonical for later device consumers. Radius outlier
  remains outside this kNN adapter. Empty views retain PDAL's existing filter
  no-op: its stage runner skips the resident boundary and filter callbacks.
- Exact proof: the focused physical wrapper/context/boundary test compares
  `outlier -> nndistance` against both upstream stages, requires neighborhood
  reuse, and observes one physical index build and one region begin/end pair.
  Additional physical tests positively observe incomplete-row repair and prove
  retained Classification refresh through a later device consumer; preflight
  rejects an undersized nonempty view. Planner tests prove a 13-neighbor shared
  envelope and reject radius-outlier materialization through this adapter. The
  disposable 1M complete-process output is byte-identical at 36,000,375 bytes
  and SHA-256
  `33ea951e4907626f1943b5367b447df003197b9a03f10b6f3e1613e0a7d53bfb`.
- Viability: three alternating dirty-tree samples measure 0.712635520 seconds
  resident versus 8.089364299 seconds pinned PDAL, or 11.351335x. A five-pair
  same-binary control measures 0.666293053 seconds resident versus
  0.686652044 seconds through the two existing private CUDA stages, a
  2.964965% reduction. The smaller control result is positive diagnostic
  evidence because the path removes one 24 MB XYZ upload and one index build
  and enables further compatible kNN composition; it is not a material or
  accepted performance claim.
- Decision: retain the adapter, but do not infer a published speedup or choose
  it automatically from dirty prototype data. B0067 profiles the improved
  endpoint and runs clean differential, sanitizer, and alternating process
  gates. If its resident-control advantage is not stable, the adapter remains
  forced GPU-native coverage and host selection remains correct. The retained
  prototype manifest is 3,128 bytes at SHA-256
  `9f81e0d802c1d6abd0834cb19db88c7ba13bfd2f4a3a68099c7384cb5a9204fe`.

## D0130 — Stop outlier-composition kernel work and test the remaining LAS boundary

- Date: 2026-08-10
- Status: accepted routing decision. Functional support, GPU-native coverage,
  performance qualification, and automatic selection are unchanged.
- Fresh profile: B0067 records 18 launches and 51.702304 milliseconds across
  the complete resident `outlier(mean_k=8) -> nndistance(k=10)` device path.
  Its two kNN gathers take 22.906240 and 28.500256 milliseconds; every other
  kernel takes 0.295808 milliseconds total. Output remains byte-identical.
- Kernel decision: all kernels are 7.759694% of B0066's 0.666293053-second
  process median. Perfectly eliminating the smaller repeated gather has only a
  3.437863% ceiling, below the requested 5--10% gate. Do not build a fused
  neighbor-list kernel or repeat launch-shape tuning for this endpoint.
- Boundary decision: do not yet declare the complete endpoint sufficiently
  optimized. The profile leaves 92.240306% of wall outside kernels, and B0052
  already proved that the reusable direct-LAS source/output architecture can
  remove 35.541759% from a comparable resident neighborhood endpoint. The
  present direct-output contract excludes Classification writes, so B0068 may
  cheaply prototype exact Classification overlay plus source-table storage for
  this existing boundary. Keep it only if the complete 1M composition improves
  by roughly 5--10%; otherwise revert it and close the endpoint.
- Evidence: the retained Nsight Compute report is 1,338,716 bytes at SHA-256
  `9f966d1a9313c5bab2ccdd82dd0cbec7e350f715e8d466a9ff45300fb9506ff8`;
  the routing manifest is
  `build/benchmarks/b0067-resident-outlier-profile.json`, 2,628 bytes at
  SHA-256
  `e890f904aca73cdfd3da14f8372a77f8328b9cbb0202c2fd42d4d05e780c2a09`.
  The disposable force hook is not part of the source tree.

## D0131 — Retain the exact opt-in direct-LAS Classification boundary

- Date: 2026-08-10
- Status: accepted as reusable opt-in residency infrastructure. Functional
  support and bounded GPU-native composition remain exact; performance
  qualification and automatic selection are unchanged.
- Evidence: B0068 alternates seven complete 1M processes after two warmups.
  The ordinary-LAS resident control median is 0.646892319 seconds and the
  direct-source/Classification-overlay median is 0.373325396 seconds, a
  42.289407% wall reduction. Every output has the same 36,000,375-byte SHA-256
  `33ea951e4907626f1943b5367b447df003197b9a03f10b6f3e1613e0a7d53bfb`.
- Exact envelope: only the explicit four-stage statistical-outlier-to-kth-
  NNDistance shape may bypass uncalibrated placement through
  `PDG_EXPERIMENTAL_DIRECT_CLASSIFICATION_OUTPUT`. The source table preserves
  upstream legacy Classification normalization, default translation overlays
  only canonical format-7 byte 16, and packed flags at byte 15 remain intact.
  Existing automatic direct-LAS envelopes do not widen.
- Proof: a checked-in physical format-3 process test requires direct source,
  output, record-summary use, no host XYZ mirror, and one index build. Its
  adversarial legacy packed flags/class 12 plus format 6/7/8 cases are exact.
  The forced lane remains explicitly uncalibrated: stats observe the separate
  raw-record hydration and require both executor-calibration and generic-
  boundary accounting matches to be false. Disabled-source and shape-mismatch
  cases fail before artifacts. The retained dirty-prototype manifest is 3,162
  bytes at
  SHA-256
  `e4dac93591c91c5505150973c9bdfcd698723acaa84c2375b76058885ab2dfa8`.
- Consequence: retain this boundary, but do not calibrate or select it from
  prototype evidence. B0069 returns to the measured catalog: profile B0044's
  exact positive `approximatecoplanar(knn=8) -> Coplanar=>UserData` endpoint
  before testing whether the same direct source removes another material
  reader/residency boundary.

## D0132 — Route approximate-coplanar work to selective exact repair

- Date: 2026-08-10
- Status: accepted routing decision. Functional support, GPU-native coverage,
  performance qualification, and automatic selection are unchanged.
- Fresh evidence: B0069's clean one-shot 1M processes take 4.293 seconds for
  pinned PDAL and 1.018 seconds for the existing exact resident/direct-output
  path. Outputs are byte-identical. Nsight Compute records only 21.592970
  milliseconds across 20 launches: 18.790000 milliseconds in kNN gather,
  2.440000 milliseconds in eigensystems, and 0.362970 milliseconds elsewhere.
- Limiter: the resident wrapper records 0.513740014 unattributed seconds and
  the execution events prove the exact ambiguity path moved a 1 MB status mask
  plus all 96 MB of eigensystems in both directions. The full 194 MB round trip
  occurs only after the status mask contains at least one tied or incomplete
  row, after which pinned KD3 repairs the selected rows.
- Decision: do not build another approximate-coplanar kernel. Also defer a
  direct-LAS source extension: the present exact repair still requires the host
  PointView and KD3 index, so source work would attack the smaller boundary
  before the dominant dependency. B0070 may cheaply prototype sparse selective
  eigensystem repair in the shared eigen-family implementation used by normal,
  eigenvalues, covariancefeatures, and approximatecoplanar. Retain it only if
  exact complete-process wall improves by roughly 5--10%; otherwise close the
  endpoint and resume the measured catalog.
- Evidence: the report is 1,521,564 bytes at SHA-256
  `a6c57d9838c73cdff30d0cd2009dd954e5402fff98cd75f21a6d999945bb29e3`;
  the retained routing manifest is 3,814 bytes at SHA-256
  `5c039498874a90956325290688304f056d1b3f585fe39c61769a7f120ef4b43c`.

## D0133 — Reject sparse transfer as the eigen-repair limiter

- Date: 2026-08-10
- Status: accepted negative prototype. The prototype was reverted; functional
  support, GPU-native coverage, performance qualification, and automatic
  selection are unchanged.
- Evidence: B0070 positively exercises a bounded selective-repair path for
  2,145 tied rows. It preserves pinned KD3 source-order recomputation and exact
  output while reducing the repair transfer from 194,000,000 to 1,208,065
  bytes, or 99.377286%.
- Gate: shell wall improves only 0.491159%, command-before-stats improves
  2.267040%, and the resident wrapper improves 3.357725%. Exact selected-row
  host repair still takes 0.449844793 seconds. No complete-process metric
  reaches the required roughly 5--10% threshold.
- Decision: revert selective transfer and close the approximate-coplanar
  endpoint for now. The remaining limiter is pinned KD3 construction and
  ordered recomputation, and the current device tie order cannot replace it
  under the exact contract. Do not infer permission for a private index or a
  relaxed answer.
- Next route: B0071 measures the existing exact composed
  `normal -> eigenvalues -> covariancefeatures -> point-program` pipeline.
  This is the next shared-index/eigensystem and fusion surface already covered
  by differential tests; name its complete-process limiter before changing
  code.
- Evidence manifest: 3,006 bytes at SHA-256
  `ff921f69c3ff9546f7a1dce657840e76797b358b5e36acfd5d9111d828266c88`.

## D0134 — Advance compatible eigen-family region calibration

- Date: 2026-08-10
- Status: accepted routing decision. Exact forced composition is positive;
  performance qualification and automatic selection remain unchanged.
- Evidence: B0071's public-shaped 1M composition is byte-identical at 14.591
  seconds pinned PDAL and 1.226 seconds forced shared residency, or 11.901305x.
  The proof requires one index and the 13-neighbor eigensystem cache to survive
  normal, eigenvalues, covariancefeatures, and their adjacent point program.
- Profile: all 22 launches total 41.533930 milliseconds. One kNN gather and
  one eigensystem build take 40.870000 milliseconds; the three projections and
  point program together take only 0.372130 milliseconds. Do not build a fused
  projection kernel from this profile.
- Planner gap: the ordinary resident request correctly falls back with
  `mixed_calibration_models` and takes 14.908 seconds. Existing per-stage
  calibration models may not be silently treated as additive or aliased to one
  anchor because they each include different measured residuals.
- Decision: B0072 measures a bounded forced-composition ladder and fits a
  separately named compatible eigen-family region model. Only after that model
  has clean provenance may runtime placement admit the exact same-k option
  shape. Keep all other mixed-model regions fail-closed. No new stage port,
  private index, or kernel is authorized.
- Evidence: the profile report is 1,611,745 bytes at SHA-256
  `812ffb28a4ce8610154f4f96baf530eea20ad32187b4622d26a7118b5955504f`;
  the retained manifest is 3,523 bytes at SHA-256
  `46ec51fac00bec11fe03e5efbde7d6f5970345e6fdf0e493c383e7167a7d3ec7`.

## D0135 — Accept a separate compatible eigen-family calibration fit

- Date: 2026-08-10
- Status: accepted calibration evidence; not yet integrated, performance-
  qualified, or automatically selected.
- Evidence: B0072 records seven exact forced-composition rows from 50K through
  16M. Speedup is positive at every size (2.155256x through 14.569879x), and
  both the shared index and 13-neighbor eigensystem reuse proofs are mandatory.
- Fit: the measured 250K-through-16M envelope yields
  `host_fixed_ns=0`, `host_ns_per_point=14812.948623061053`,
  `device_fixed_ns=185808307.7237236`, and
  `device_ns_per_point=1005.4255825490451`. The proposed selection floor is
  250K; 50K is validation only.
- Decision: retain this as a new `eigen-family-compose` model rather than
  modifying or summing the normal, eigenvalues, and covariancefeatures models.
  B0073 owns implementation: exact shape recognition, calibration/profile
  pinning, positive and negative placement tests, resident-executor proof,
  repeated complete-process qualification, and documentation. Unrelated mixed
  models and different-k/option shapes remain fail-closed. Automatic public
  `pdg pipeline` admission is a separate gate and does not follow merely from
  explicit resident placement.
- Evidence: the aggregate ladder manifest is 5,475 bytes at SHA-256
  `7c5c64ccc4be0ea947592c9e0d60f63367543a2dd98db99db2ac267df17eb9a2`.

## D0136 — Qualify the exact 1M explicit-resident eigen-family composition

- Date: 2026-08-10
- Status: accepted exact 1M performance qualification and explicit-resident
  placement; not automatically selected and not model-wide resident
  calibration provenance.
- Implementation: commit `32b864711` adds the separately named
  `eigen-family-compose` residual to both calibration representations and
  recognizes only the exact compiled six-stage shape. Changed k, normal
  orientation, eigenvalue normalization, covariance mode/features, assignment
  operation/order/count, missing/extra consumers, and unrelated mixed regions
  remain fail-closed.
- Exactness: the hash-pinned physical process gate requires one shared index
  and 13-neighbor eigensystem reuse, observes accepted preflight and one
  predicted/observed index build, matches pinned warning stderr, and produces
  byte-identical LAS output.
- Performance: five clean alternating 1M samples measure 14.507773633 seconds
  pinned PDAL versus 1.397204802 seconds explicit resident, or
  **10.383426690x**. The complete artifact remains exact at 36,000,375 bytes.
- Decision: qualify only this exact 1M explicit-resident pipeline. Keep
  `selected_device_calibration_matches_executor=false`: B0072's seven model
  rows came from forced hybrid composition, and one resident cardinality does
  not replace that provenance. B0074 reruns the same seven-cardinality ladder
  through explicit resident mode. Only an exact positive resident fit may
  promote that flag; automatic `pdg pipeline` and direct-output admission stay
  separate later gates.
- Evidence: the B0073 report is 23,277 bytes at SHA-256
  `4b3319beb557e1e698c4480b2afd78a3dc42d1b5618a4610605a09fb9070b91f`.

## D0137 — Replace forced eigen-family provenance with resident measurements

- Date: 2026-08-10
- Status: accepted resident calibration provenance; explicit selection remains
  bounded and automatic public admission remains off.
- Evidence: B0074 reruns the exact B0072 cardinalities through `pdg resident`.
  The 50K row is byte-exact and host-selected at 0.715669x. Every
  250K-through-16M row requires `planner_resident_shared_index`, one shared
  index, and the 13-neighbor eigensystem cache; all are byte-exact and win by
  6.406690x through 11.915007x.
- Fit: the selected resident rows yield
  `host_fixed_ns=120004600.90127563`,
  `host_ns_per_point=14659.538986578957`,
  `device_fixed_ns=155357777.57434177`, and
  `device_ns_per_point=1234.7451449457265`, bounded to 250K--16M.
- Decision: replace the seven forced-hybrid calibration cases with the seven
  resident reports and update both coefficient representations bit-for-bit.
  The selected executor now has full-envelope provenance, so
  `eigen-family-compose` may join `ResidentCalibratedModels` and the physical
  gate must report `selected_device_calibration_matches_executor=true`.
  Preserve the exact shape matcher and every mixed-model negative.
- Next: B0075 cheaply evaluates automatic `pdg pipeline` admission for only
  this shape. It must preserve early fallback/no-side-effect behavior and
  complete-process value. Direct LAS output and broader options remain
  separate hypotheses.
- Evidence: the B0074 aggregate is 6,198 bytes at SHA-256
  `0e0e788967a05fa58cb2177e98d0d68682a146028512dbc085be1d8cf8781595`.

## D0138 — Automatically admit only the measured eigen-family composition

- Date: 2026-08-11
- Status: accepted exact performance qualification and automatic selection
  for one bounded public pipeline shape.
- Implementation: commit `b39ef04c6` adds a no-side-effect admission probe for
  the exact B0073/B0074 six-stage JSON shape, then requires the independently
  compiled runtime placement to resolve to the selected
  `eigen-family-compose` model. Only an uncompressed `.las` source, default
  `.las` writer, exact normal/eigenvalues/covariance options and assignment
  program, matching SM-89 profile, 250K--16M cardinality, accepted preflight,
  and one selected resident region can commit. Direct LAS output remains off.
- Exactness: the physical gate compares explicit resident, required public,
  and genuinely option-free public artifacts with pinned PDAL. Changed k,
  assignment order, extra topology, injected preflight failure, and a
  below-floor host case fail closed without output. Unsupported CLI/options,
  devices, profiles, and placements return to the unchanged public path before
  side effects.
- Performance: five clean alternating 1M samples measure 14.452746005 seconds
  pinned PDAL versus 1.409086026 seconds automatic PDG, or
  **10.256823032x**. B0071's profile remains authoritative because admission
  invokes the same executor; kernels are only 41.533930 milliseconds and the
  projections plus assignment total 0.372130 milliseconds.
- Decision: mark normal, eigenvalues, and covariancefeatures automatically
  selected only inside this exact measured composition. Do not generalize the
  result to standalone stages, other options/assignments, LAZ, direct output,
  or other devices. The proof variable is fail-closed evidence, not an opt-in.
- Next: two cheap post-gate probes reject inherited direct-output support and
  ordinary-data HAG count-four admission before timing. B0076 therefore tests
  an exact ordinary-data `estimaterank -> optimalneighborhood -> assign`
  composition for shared-index value before widening HAG or adding a stage.
- Evidence: the B0075 report is 16,039 bytes at SHA-256
  `8ddb01a385f8293267629bcc082b20b3940e39254810da35faa0d3327d5c50f6`.

## D0139 — Prefer rank/optimal region calibration over another kernel

- Date: 2026-08-11
- Status: accepted directional composition/profile evidence; no performance
  qualification, model, or automatic selection yet.
- Evidence: B0076 measures an ordinary 1M exact
  `estimaterank(knn=14) -> optimalneighborhood(10..14) -> assign -> LAS`
  pipeline at 11.303210336 seconds pinned PDAL versus 1.246256090 seconds
  forced shared resident, or 9.069733x. An exact identity-coordinate boundary
  forces two index regions and preserves the final LAS hash but takes
  1.697469160 seconds, so one compatible region removes 26.581518% of the
  candidate wall.
- Profile: 19 kernels total 88.786784 milliseconds. The two existing kNN
  gathers take 44.095552 and 44.339328 milliseconds; assignment takes 0.060800
  milliseconds and the index/setup remainder is below 0.3 milliseconds.
  Perfectly removing one gather has only a 3.557802% complete-process ceiling.
- Decision: do not fuse neighbor-list kernels, port a new stage, or widen HAG.
  Treat the current integration gap as the same truthful mixed-model problem
  B0071 exposed: compatible existing stages execute quickly when forced but
  ordinary resident placement cannot borrow either per-stage residual.
- Next: B0077 measures exact forced composition at 50K, 250K, 1M, 2M, 4M, 8M,
  and 16M, proves shared-index reuse, and fits a separately named bounded
  residual only if every selected row remains positive. Shape recognition,
  resident-provenance replacement, and public admission remain later gates.
- Evidence: the B0076 shared report is 7,327 bytes at SHA-256
  `5797eded6271f54aa6114f25a59c32cf8d8858657109478e05cb73b8139fedea`;
  the profile is 1,421,750 bytes at SHA-256
  `3bc9fac6d0d7c863c5e8449b8bb708f227ddd4c678e4b0549d4a1a734ef64786`.

## D0140 — Accept a separate compatible rank/optimal calibration fit

- Date: 2026-08-11
- Status: accepted calibration evidence; not yet integrated, performance-
  qualified, or automatically selected.
- Evidence: B0077 records seven exact forced-composition rows from 50K through
  16M. Every process requires shared-neighborhood reuse and matches pinned
  status, diagnostics, and LAS bytes. Speedup is positive at every size,
  ranging from 1.542997x to 10.755371x, with no ordinary-data repair cliff.
- Fit: the conservative 250K-through-16M envelope yields
  `host_fixed_ns=117903407.89407083`,
  `host_ns_per_point=11435.542524532335`,
  `device_fixed_ns=233890937.082784`, and
  `device_ns_per_point=1050.2295849761058`. The proposed floor remains 250K;
  the positive 50K result is validation only.
- Decision: retain this as a new `rank-optimal-compose` model rather than
  changing, summing, or relabeling the existing estimate-rank and optimal-
  neighborhood models. B0078 owns exact compiled-shape recognition,
  calibration/profile pins, positive and negative placement tests, explicit-
  resident proof, and repeated complete-process qualification. Forced-hybrid
  evidence cannot by itself authorize public automatic admission or truthful
  resident-executor provenance.
- Evidence: the aggregate ladder is 5,598 bytes at SHA-256
  `6a4905a1d7a9fd8044655921e5e6bf8a82614b3826d0c1d32a1b03ae2f3b5d37`.

## D0141 — Qualify the exact 1M explicit-resident rank/optimal composition

- Date: 2026-08-11
- Status: accepted exact 1M performance qualification and explicit-resident
  placement; not automatically selected and not model-wide resident
  calibration provenance.
- Implementation: commit `7d0efc2ff` adds the separately named
  `rank-optimal-compose` residual to both calibration representations and
  recognizes only the exact compiled five-stage shape. Changed k, threshold,
  k range, assignment operation/order/count, stage order, extra consumers,
  coordinate invalidation, and unrelated mixed regions do not inherit it.
- Exactness: the physical process gate requires one shared index and
  neighborhood reuse, observes accepted preflight and one predicted/observed
  index build, matches pinned diagnostics, and produces byte-identical LAS.
  The model matrix and all 132/132 unique raw-report hashes also verify.
- Performance: five alternating 1M samples after one warmup measure
  11.353348769 seconds pinned PDAL versus 1.370295007 seconds explicit
  resident, or **8.285331780x**. B0076 remains the profile record and rejects
  another kernel at a 3.557802% perfect-removal ceiling.
- Decision: qualify only this exact 1M explicit-resident pipeline. Keep
  `selected_device_calibration_matches_executor=false`: B0077's seven model
  rows came from forced hybrid composition, and one resident cardinality does
  not replace that provenance. B0079 reruns the same seven sizes through
  explicit resident mode. Public automatic admission remains a separate gate.
- Evidence: the B0078 report is 23,337 bytes at SHA-256
  `134a958aa9b2e0beee699c97fe6398a6bca240392a92579c6876b41e1c87d728`.

## D0142 — Replace forced rank/optimal provenance with resident measurements

- Date: 2026-08-11
- Status: accepted resident calibration provenance; explicit selection remains
  bounded and automatic public admission remains off.
- Evidence: B0079 reruns the exact B0077 cardinalities through `pdg resident`.
  The 50K row is byte-exact and host-selected at 0.698133x. Every
  250K-through-16M row requires `planner_resident_shared_index`, accepted
  preflight, and neighborhood reuse; all remain byte-exact and win by
  4.873850x through 9.566431x.
- Fit: the selected resident rows yield
  `host_fixed_ns=133650155.04406099`,
  `host_ns_per_point=11448.442351895543`,
  `device_fixed_ns=240029160.187007`, and
  `device_ns_per_point=1186.5275421400943`, bounded to 250K--16M.
- Decision: replace the seven forced-hybrid calibration cases with the seven
  resident reports and update both coefficient representations bit-for-bit.
  The selected executor now has full-envelope provenance, so
  `rank-optimal-compose` joins `ResidentCalibratedModels` and the physical gate
  must report `selected_device_calibration_matches_executor=true`. Preserve
  the exact shape matcher and every mixed-model negative.
- Next: B0080 cheaply evaluates automatic public `pdg pipeline` admission for
  only this shape. It must preserve early fallback/no-side-effect behavior and
  complete-process value. Direct LAS output and broader options remain
  separate hypotheses.
- Evidence: the B0079 aggregate is 6,069 bytes at SHA-256
  `43f4fcd0e95ad8e90eaf40082bf522b77a0d1046a760d1828dcc72c60ed55098`.

## D0143 — Automatically select only the measured rank/optimal composition

- Date: 2026-08-11
- Status: accepted exact public automatic selection for the measured shape.
- Implementation: commit `b881fe11b` admits only the exact five-stage
  uncompressed-LAS pipeline whose compiled runtime placement independently
  selects one `rank-optimal-compose` region. The proof requirement is
  fail-closed and does not opt a pipeline into admission. Changed k, threshold,
  range, assignment order, topology, cardinality envelope, or rejected
  preflight declines before source/output side effects.
- Exactness: explicit resident, required public, and genuinely option-free
  public physical paths match pinned status, diagnostics, artifact count, and
  complete LAS bytes. The retained five-pair public benchmark has the same
  36,000,375-byte output at SHA-256
  `07b88e638b5ef3192c356bc4f7e1c91a825668d70e7ba37d7f181550a4e8e6d2`.
- Performance: the clean 1M public command measures 11.324354873 seconds
  pinned PDAL versus 1.388344338 seconds automatic PDG, or **8.156733573x**.
  B0076's unchanged-executor profile bounds perfect removal of one repeated
  gather to 3.557802% of complete wall time, below the requested 5--10%
  reusable-optimization bar.
- Decision: mark EstimateRank and OptimalNeighborhood automatically selected
  only inside this exact measured composition and treat the endpoint as
  sufficiently optimized for now. Do not infer standalone qualification,
  broader option coverage, direct LAS output, or a general mixed-model rule.
  Resume the catalog through measured existing-stage composition candidates.
- Evidence: B0080 report, 16,137 bytes, SHA-256
  `4d3016724a5f5793b05107ed02b0d12a7153352d5f9c56ef9bf2c463f1a5cea2`.

## D0144 — Route radiusassign toward the reusable LAS boundary

- Date: 2026-08-11
- Status: accepted routing evidence; no qualification or selection change.
- Evidence: B0081 revalidates the exact 1M resident radiusassign path on a
  clean current tree at 1.921081414 seconds pinned PDAL versus 0.697987348
  seconds resident, or 2.752315525x. Required shared-index execution, reuse,
  preflight, calibration provenance, diagnostics, and LAS bytes all pass.
- Profile: all 16 kernels total 1.899456 milliseconds. The 1.634496-millisecond
  radius query is only 0.234172% of complete wall, so no device-kernel
  optimization is authorized. The largest named ordinary interval is
  0.390594913 seconds of rewritten manager execution; runtime placement/setup
  is a separate 0.178538164 seconds already dispositioned by B0054.
- Decision: do not promote B0019's dirty-provenance ladder or change automatic
  selection from this one-shot result. B0082 may cheaply adapt the existing
  exact direct-LAS source, record-summary index configuration, and UserData
  output overlay to this measured radiusassign shape. Retain it only after an
  exact alternating same-binary result removes roughly 5--10% or more; revert
  and move on otherwise. New kernels and new stage ports remain out of scope.
- Evidence pins: one-pair report SHA-256
  `3bd6127dba0d7875bae3e97f0eed005ebca181b9fdbac1387ebded8cf5e1f935`;
  Nsight report SHA-256
  `9c515a021d67bb740efd95a291c564b3997dc3ffa40a7399b06793ae130a157f`;
  stats SHA-256
  `cfac0b97ccc00c506af610eb034f5c6601f75933e4766d9c01aa6c2d95cefb0c`.

## D0145 — Retain the exact direct-LAS radiusassign boundary

- Date: 2026-08-11
- Status: accepted directional prototype and exact opt-in coverage; not
  performance-qualified, calibrated, or automatically selected.
- Evidence: B0082 alternates seven 1M samples of the ordinary and direct
  resident endpoints through the same binary. Exact median wall falls from
  0.641774202 to 0.316225432 seconds, a 2.029483201x same-binary boundary
  ratio and 50.726372139% reduction, while status, diagnostics, artifact count,
  and LAS bytes remain identical. This is not a pinned-PDAL speedup claim.
- Implementation: admit only the exact three-stage B0081 shape when direct
  output/source is explicitly requested. The sparse resident table decodes
  ReturnNumber from mapped LAS records using the format-0--5 three-bit and
  format-6+ four-bit fields. A summary overload constructs the same exact
  radius grid without a host XYZ mirror; canonical publication overlays the
  final UserData column. The planner-owned radius index and existing exact
  host assignment semantics are unchanged.
- Exactness/safety: the physical gate covers LAS formats 7 and 3, positively
  changes UserData, and matches the pinned oracle byte-for-byte. Radius or
  domain drift and source disablement fail before output. The summary builder
  is bit-compared with the PointBatch builder and rejects invalid envelopes.
  A non-skippable unit distinguishes raw return bytes 9 and 15 under legacy
  three-bit and modern four-bit masks. Memcheck, racecheck, initcheck, and
  synccheck are clean.
- Correction: D0144/B0081's reuse requirement was a no-op because the exact
  endpoint has one neighborhood consumer and the rewrite sets reuse false. Its
  valid proof is one planner-owned index build, not cross-stage reuse. B0082
  removes that requirement and claim.
- Decision: retain this boundary because it greatly exceeds the 5--10%
  viability threshold and reuses the direct-LAS architecture. Keep automatic
  admission off and keep both calibration-provenance diagnostics false. B0083
  runs a clean pinned/direct 50K-through-16M ladder and may fit only a
  separately named direct-executor residual if all selected rows remain exact
  and positive. Do not borrow B0019's ordinary-resident model or start another
  stage port first.
- Evidence: B0082 aggregate, 4,921 bytes, SHA-256
  `91c49cb1a9313acbd455db2f5fa9092968446553ca1880715659f49ac6191814`;
  direct stats SHA-256
  `c646b5d044afcd4ee16b7d4db1a28f694d2aad7d01445e994d6199e2d411d333`.

## D0146 — Accept a separate direct radiusassign calibration proposal

- Date: 2026-08-11
- Status: accepted calibration evidence; not integrated, performance-
  qualified, or automatically selected.
- Evidence: B0083 runs the exact B0082 shape from 50K through 16M on clean
  commit `50f4ec9d7`. The 50K control is exact and host-selected at 0.358367x.
  All six required direct rows are exact and positive, from 1.523498x at 250K
  through 22.408647x at 16M, with source/output/summary/no-host-XYZ/preflight/
  index proofs.
- Fit: clamped OLS over selected absolute walls gives
  `host=0 + 1,985.118056 ns/point` and
  `device=305.407938 ms + 68.335971 ns/point`, bounded to 250K--16M.
- Decision: preserve this as a separately named `radiusassign-direct` proposal.
  Do not overwrite, relabel, or borrow the ordinary radiusassign model. B0084
  must integrate both coefficient representations, prove exact-shape and
  negative matching, and make direct executor/boundary provenance truthful
  from these raw rows. Repeated performance qualification and automatic public
  admission remain separate gates.
- Evidence: aggregate 5,371 bytes, SHA-256
  `8057739482f1e449d5f1581288b90c7cebe41f526dc5e98bb0ef895a74269db8`.

## D0147 — Integrate the bounded direct radiusassign model without widening admission

- Date: 2026-08-11
- Status: accepted calibration integration; not repeated performance-
  qualified or automatically selected.
- Model: retain B0083's absolute curves
  `host=0+1,985.1180564323188 ns/point` and
  `device=305,407,938.01419646+68.33597116527429 ns/point`. The compiled and
  JSON planner residual is independently named `radiusassign-direct` and
  subtracts the already-modeled cold start, two boundary synchronizations,
  25-byte upload, one-byte spill, and 28-byte index terms:
  `device=66,189,962.889582455+67.0868094224532 ns/point`. Planner evaluation
  reconstructs the absolute curve exactly. The ordinary `radiusassign` model
  and all seven of its raw rows are unchanged.
- Envelope: selection requires the strict three-stage serialized B0083 shape,
  explicit direct output/source/record-summary/no-host-XYZ proofs, a 36-byte
  LAS input record, 250K--16M points, and the exact compiled radius/update
  program. Option, topology, update, or layout drift fails closed. Automatic
  pipeline admission remains off.
- Boundary truth: mapped input and sparse UserData publication cross the
  planner's exact 25-byte upload and one-byte spill without PointView packing;
  the raw record bounds one intrinsic whole-view lane. Format 7 therefore
  reports both boundary and executor-calibration provenance true. The exact
  format-3 companion reports boundary provenance true but leaves executor
  calibration false because B0083 contains no format-3 timing row.
- Proof: focused host, CUDA, and ASan/UBSan placement/scheduler tests pass;
  all 139 local calibration reports verify by hash; and the executor-aware
  placement audit is 133/133 at winner accuracy 1.000. The physical pinned-
  PDAL format-7/format-3 gate is byte- and diagnostic-exact with the observed
  whole-view one-lane schedule and one predicted/observed shared index. B0085
  owns repeated qualification and any option-free admission change.

## D0148 — Qualify and stop tuning the explicit direct-radiusassign endpoint

- Date: 2026-08-11
- Status: accepted exact performance qualification and negative optimization
  decision; automatic selection and broader radiusassign coverage are
  unchanged.
- Evidence: on clean harness commit `0051b11a3` with the candidate implementation
  from `c21b87411`, one warmup and five alternating 1M samples measure
  1.896917103 seconds pinned PDAL versus 0.339432621 seconds explicit direct
  resident, or **5.588493815x**. Every timed candidate requires the direct
  shared-index executor, source/output/record-summary use, no host XYZ mirror,
  accepted preflight, and both executor-calibration and boundary-accounting
  provenance. Status, empty diagnostics, artifact count, and all 36,000,375
  LAS bytes match at SHA-256
  `28302b0ba60ff587976edec6546d2ff7156c720d0d37450299e6ea8ac47c4c00`.
- Profile: the exact-environment Nsight Compute basic report records 18
  launches and 1.953650 milliseconds total. `radiusAnyKernel` is 1.630000
  milliseconds, 83.433573% of kernel time but only 0.480213% of median
  complete wall; perfect removal of every kernel is only 0.575563%. The fresh
  uninstrumented trace again places 0.165975550 seconds in the necessary
  process-level CUDA/profile/placement startup already dispositioned by
  D0117. Canonical publication is the largest other named reusable phase at
  0.012043519 seconds, or 3.548132% of median wall. No identified reusable
  operation reaches the requested roughly 5--10% threshold.
- Decision: describe only this explicit exact envelope as performance-
  qualified and sufficiently optimized for now. Do not enable automatic
  admission from an explicit-resident benchmark. B0086 resumes the catalog
  with the existing exact one-index statistical-outlier-to-NNDistance
  composition and its direct Classification boundary, whose B0068 prototype
  removed 42.289407% of same-binary wall but remains dirty and uncalibrated.
- Evidence: retained benchmark report, 23,897 bytes at SHA-256
  `345cca4d0291406bb86f31ad43afefb6eb59e9c097d5628e0dc98fe91110e024`;
  profile exactness/provenance manifest, 3,799 bytes at SHA-256
  `c58936d5571c4c81468c95a3342965a24584adedfdd0157c31a26cb0b0f3b59a`;
  Nsight Compute report, 1,364,739 bytes at SHA-256
  `0e4f6f76a293fae7393b1a84e9ecdd7d6ec7015f283c6eb34dc89b8922374268`;
  exact uninstrumented stats, 10,425 bytes at SHA-256
  `36b4c89df9f07f94aec24589e534355723986f7e1713345974f294ecd8722d89`.

## D0149 — Reopen one exact gather-fusion prototype after the direct boundary

- Date: 2026-08-11
- Status: accepted routing/profile evidence; performance qualification,
  calibration, boundary provenance, and automatic selection are unchanged.
- Evidence: B0086 runs one zero-warmup alternating 1M pair on clean commit
  `21a3aeb84`. Pinned PDAL takes 8.011663864 seconds and the existing exact
  direct Classification composition takes 0.420340609 seconds, a directional
  **19.059933x**. Status, empty diagnostics, artifact count, and all 36,000,375
  output bytes match at SHA-256
  `c43ef2dd91c6e2cf147af50163422d8c1cfccc853538effb24f000e7e3844d48`.
  The candidate proves mapped source, direct output, record-summary index
  configuration, no host XYZ mirror, one whole-view region, and one index
  build. It remains explicitly uncalibrated, so executor-calibration and
  boundary-accounting matches correctly report false.
- Profile: the exact-output-bound Nsight Compute report records 20 launches
  and 52.001290 milliseconds total. The outlier and NNDistance
  `knnGatherKernel` calls take 23.260000 and 28.390000 milliseconds; all other
  device work is only 0.351290 milliseconds. B0067 rejected shared-query work
  when the smaller gather was 3.437863% of its ordinary-boundary control. The
  direct boundary now lowers candidate wall enough that the same removable
  class is 5.533608%, crossing the requested rough prototype threshold.
- Decision: do not calibrate an executor with a still-measured redundant
  gather. B0087 may cheaply add one region-owned ordered max-k gather for the
  exact compatible pair. Outlier `mean_k=8` must derive its mean from the first
  nine self-inclusive entries while excluding self; kth NNDistance `k=10`
  consumes the eleventh. Preserve independent status/tie/incomplete repair and
  all stage/publication semantics. Retain only after an exact same-binary
  complete-process improvement around 5%; otherwise revert and calibrate the
  unchanged B0086 executor.
- Evidence: routing report, 9,742 bytes at SHA-256
  `57ac2237e0bd4b699abcaaca207fe8ec82641a8e0a43bcc180924dbbd4c852a6`;
  exact profile manifest, 4,373 bytes at SHA-256
  `443db438292ee360b108f0ce0d6773d1668ad999ef908b0973209d017b7b5c4b`;
  Nsight Compute report, 1,452,921 bytes at SHA-256
  `ad8fc0ef5fa29f9c8ff5eb7e87e65ee3254d2fb6c9bbeb0b164375699c9699c4`;
  uninstrumented stats, 11,058 bytes at SHA-256
  `c58e9d4f260ba1ff232f6b7f1501e215261e204f641d7facb29e2f2c0cc035ed`.

## D0150 — Retain planner-owned max-k reuse and stop endpoint tuning

- Date: 2026-08-11
- Status: accepted exact reusable optimization; performance qualification,
  calibration, and automatic selection remain unchanged.
- Direction gate: B0087's cheap same-binary prototype measures five
  interleaved complete-process samples per route. Median wall falls from
  0.383653 to 0.340241 seconds, a **1.127592x** ratio and **11.315433%**
  reduction. Every process has identical status, empty diagnostics, artifact
  count, and 36,000,375 output bytes at SHA-256
  `c43ef2dd91c6e2cf147af50163422d8c1cfccc853538effb24f000e7e3844d48`.
  This dirty-worktree same-binary comparison answers viability only; it is not
  a pinned-PDAL performance qualification.
- Exact contract: the planner annotates only an adjacent, single-consumer
  statistical-outlier -> NNDistance pair in one compatible 3D kNN region.
  The resident executor gathers the explicit maximum self-inclusive width
  once, preserves the ordered squared distances and status, and projects each
  stage's original binary64 arithmetic. Branches, bridges, incompatible
  dimensions/regions, and wider consumers receive no product. Runtime
  cardinality that cannot materialize the superset uses the prior exact query
  unless the proof switch requires reuse. Incomplete max-k rows conservatively
  take each consumer's pinned exact repair. Allocation products are checked,
  budgeted at `12*k + 1` bytes per point, and released at the declared
  consumer.
- Clean proof: commit `21836fa4d` passes the 24-test neighborhood regression,
  the positive one-gather and bridge/producer negatives, asymmetric-k exact
  fallback, and focused memcheck/racecheck. The final direct process records
  one index, `knn_gather_reuse=true`, no repair, and exact output.
- Profile/stopping decision: the fresh exact capture reduces B0086's profiled
  kernel work from 52.001290 to 29.350784 milliseconds, or 43.557585%. One
  28.689120-millisecond gather now owns 97.745668% of device time; the two
  exact projections total 0.311200 milliseconds. B0053 already rejected the
  available gather launch-shape tuning, and no duplicate endpoint work remains
  near the 5--10% bar. Treat this endpoint as sufficiently optimized for now.
  A different shared-index gather algorithm is a new generic measured
  candidate, not more endpoint fusion.
- Coverage/next: B0087 changes no functional, GPU-native, performance-
  qualified, or automatically-selected category. B0088 resumes measured
  catalog progress with a repeated clean pinned-PDAL/direct qualification of
  the improved outlier composition before any calibration or public selector.
- Evidence: prototype report, 4,313 bytes at SHA-256
  `acec2a57b89e9dfd900bc4b307e7dbc469e118742f67040a01b319399dab25af`;
  final profile manifest, 5,986 bytes at SHA-256
  `d892f13ec63ffd909c80208a4c4565d70670a50d4345ec6876f6a7d34349f1ce`;
  Nsight Compute report, 1,520,260 bytes at SHA-256
  `361354e005b780311ccb84d90515c4cdae39402f0384e16b3dbecb5d57c4c5f8`;
  uninstrumented stats, 11,086 bytes at SHA-256
  `7b34a9961b252fa36e33df3e5b0a2bec5f1dc5746c05f5a01d7f310bbde065eb`.

## D0151 — Qualify the explicit max-k outlier composition

- Date: 2026-08-11
- Status: accepted performance qualification for one explicit exact route;
  calibration and automatic selection remain false.
- Gate: B0088 runs one warmup plus five alternating complete-process pairs on
  clean commit `22f5218da`. Pinned PDAL's 7.961676747-second median compares
  with 0.372223208 seconds for the direct resident route, or **21.389523x**.
  Every process has zero status, empty diagnostics, one artifact, and identical
  36,000,375-byte output at SHA-256
  `c43ef2dd91c6e2cf147af50163422d8c1cfccc853538effb24f000e7e3844d48`.
- Execution proof: every candidate observes the shared-index direct-LAS
  executor, direct output/source/record summary, no host XYZ mirror,
  `knn_gather_reuse=true`, and prediction-matching integer one for both
  predicted and observed index builds. The report explicitly records the
  experimental Classification-output switch; it does not imply public or
  automatic admission.
- Profile binding: B0088's resolved sibling engine SHA-256
  `fa0e924da2633f331205219cc32a91923eda43d37d200118d716f23c430335b8`
  matches the engine in B0087's exact-output-bound fresh Nsight manifest. The
  dispatcher, oracle, frozen-time library, input, pipeline, and output hashes
  match too, so the B0087 stopping profile applies to this qualification.
- Coverage/next: the named composition is functionally supported, bounded
  GPU-native, and now performance-qualified. It remains uncalibrated and not
  automatically selected. B0089 is a separately named bounded calibration
  ladder for this composition; do not borrow another model or resume endpoint
  kernel tuning.
- Evidence: B0088 report, 25,332 bytes at SHA-256
  `6ba364553b327651a04b05b7c1c8ec190b6f6d0baedd6fd8a7e653cd57520f31`;
  B0087 profile manifest, 5,986 bytes at SHA-256
  `d892f13ec63ffd909c80208a4c4565d70670a50d4345ec6876f6a7d34349f1ce`.

## D0152 — Remove both bounded repair cliffs, stop endpoint tuning, and resume calibration

- Date: 2026-08-11
- Status: accepted exact reusable optimizations and updated named-route
  performance qualification; automatic selection remains false.
- Measured trigger: B0089's clean size ladder is positive from 50K through
  4M, but candidate wall jumps from 0.460879474 seconds at 2M to
  3.401012038 seconds at 4M. Its uninstrumented 4M trace leaves
  2.754445394 seconds in the resident wrapper. Static attribution finds one
  incomplete statistical-outlier mean row constructing and using pinned KD3.
  Calibration is paused rather than fitting that exceptional host cliff.
- B0090 contract: for at most 16 incomplete statistical-outlier rows and
  self-inclusive width at most 16, select exact neighbors from planner-owned
  resident coordinates with the existing partial top-k/ordered merge and then
  execute the pinned serial online-mean recurrence. Host KD3 remains exact
  fallback outside the envelope. Consumer-specific telemetry and device/
  parallel proof guards are mandatory. The clean 4M gate is byte-exact at
  44.072341x pinned PDAL.
- Profile-directed continuation: B0090's fresh exact profile puts only
  1.654528 milliseconds in the new outlier repair but 77.181792 milliseconds
  in NNDistance's older serial one-row full scan. That is 10.451157% of
  complete median wall and clears the requested last-optimization threshold.
- B0091 contract: make the already-tested bounded NNDistance partial-select/
  ordered-merge implementation the default inside its existing `k <= 15`,
  at-most-16-row exact device envelope. Keep the serial device path behind its
  explicit disable switch and unchanged host fallback outside the envelope.
  Emit separate NNDistance host/device/parallel counters; evidence runners
  type-check positive integer repair counts and zero host repair. Automatic
  pipeline fallback and explicit `pdg resident` requirements fail before an
  output artifact when the proof cannot be evaluated.
- Final gate: B0091's five alternating clean 4M pairs are exact at
  32.809776349 seconds pinned PDAL versus 0.672899645 seconds resident, or
  **48.758796x**. This improves B0090 median wall by **8.882913%**. Both
  incomplete consumers report one parallel device-repaired row and zero host
  repair; all four focused Compute Sanitizer tools and the full host/CUDA
  aggregates are clean.
- Stopping decision: the final profile reduces kernel work from 222.010592 to
  146.471808 milliseconds. One shared gather owns 139.947488 milliseconds or
  95.545682%; every other kernel combined has only a 0.969583% process-wall
  perfect-removal ceiling. B0053 already rejected bounded gather launch tuning,
  and there is no concrete further reusable 5--10% candidate. Treat this
  endpoint as sufficiently optimized. Resume B0089 with 8M/16M rows and fit a
  separately named composition model only if the repaired curve is stable.
- Coverage: the explicit outlier/NNDistance direct composition remains
  functionally supported, bounded GPU-native, and performance-qualified. It
  is still uncalibrated and not automatically selected. Standalone outlier and
  broader NNDistance shapes do not inherit the route claim.
- Evidence: B0090 report/profile manifest, 27,634/6,262 bytes at SHA-256
  `457f6be54cb32cde47dc1c1621816c4b7d81602f77f708e8cfdf589bfe68da56` /
  `8e41e932771762b45f9e34c99cbd684b012cd7311048f96b3ae0e8435303957f`;
  B0091 report/profile manifest, 29,821/6,687 bytes at SHA-256
  `c943e6d49ea7b42f45643dcc427bbc0d88c8b9eed1fbef23f9204fabc8366c8e` /
  `a2a9d6605c9e2b439cf54a6842b4f56e7c4b127aa789a8715b3943987276ed8d`.

## D0153 — Automatically select only the calibrated direct outlier composition

- Date: 2026-08-11
- Status: accepted exact public automatic selection for one measured
  composition; no standalone or broader-option claim.
- Calibration: B0092 extends the repaired direct-composition ladder through
  8M and 16M, then fits the separately named
  `outlier-nndistance-direct-compose` absolute-wall model. The accepted device
  envelope is 50K through 16M points on the exact SM89 profile; 25K remains a
  host control. The ordinary outlier and NNDistance models are unchanged.
- Admission: commit `67b35be1e` requires the exact linear uncompressed-LAS
  shape `outlier(statistical,mean_k=8,multiplier=2,class=7) ->
  nndistance(kth,k=10) -> LAS`, 36-byte records, one resident region/index/
  lane, and the exact 133-byte/point eleven-neighbor gather product. Option,
  mode, topology, layout, cardinality, profile, source, and preflight drift
  decline before commitment. After execution, automatic publication requires
  observed direct source use, record-summary configuration, no host XYZ mirror,
  max-k gather reuse, and exactly one index build; proof failure cannot publish
  an artifact or retry after side effects.
- Exactness/performance: five alternating option-free 1M pairs produce the same
  36,000,375-byte LAS at SHA-256
  `c43ef2dd91c6e2cf147af50163422d8c1cfccc853538effb24f000e7e3844d48`.
  Pinned PDAL median is 7.971261964 seconds; automatic PDG median is
  0.392659506 seconds, or **20.300698x**. The placement audit is 142/142 over
  33 models and all 148 raw-report references verify.
- Profile/stopping decision: B0091's fresh profile applies to the unchanged
  shared-index/direct-LAS execution endpoint. Its one gather owns 95.545682%
  of device work, and every other kernel together has only a 0.969583%
  complete-wall ceiling. B0053 already rejected the bounded gather launch
  sweep, so B0092 changes placement/admission only and endpoint tuning stays
  closed.
- Coverage/next: `filters.outlier` joins the automatically selected count only
  inside this exact composition; NNDistance was already automatic in its
  separate B0045/B0050 public shape. Functional support, bounded GPU-native
  work, and composition performance qualification remain independently scoped.
  B0093 should apply the same cheap public-admission decision to the already
  calibrated, explicitly qualified, sufficiently optimized direct-radiusassign
  endpoint before considering a lower-evidence catalog candidate.
- Evidence: B0092 qualification report, 16,657 bytes at SHA-256
  `d62192ec634692880c3d598403e3ceda5aec85982d5efd63bb4012fbe9caa731`;
  B0091 profile manifest, 6,687 bytes at SHA-256
  `a2a9d6605c9e2b439cf54a6842b4f56e7c4b127aa789a8715b3943987276ed8d`.

## D0154 — Automatically select the measured direct-radiusassign endpoint

- Date: 2026-08-11
- Status: accepted exact public automatic selection for one calibrated
  envelope; the kernel, placement coefficients, broader options, and partial
  GPU-native classification are unchanged.
- Admission: commit `a473f49f9` recognizes only the exact linear uncompressed-
  LAS shape `radiusassign(radius=2,is3d=true,src=ReturnNumber[1:1],
  reference=ReturnNumber[2:15],UserData=9) -> LAS`. Runtime additionally
  requires a 36-byte source record, the `radiusassign-direct` device choice,
  one selected region, and its 250K--16M calibrated cardinality envelope.
  Format, option, topology, layout, cardinality, source, and preflight drift
  decline before commitment.
- Proof: before automatic publication, execution must observe the mapped
  direct source, record-summary index configuration, no complete host XYZ
  mirror, and exactly one index build. The process matrix proves a 50K
  below-floor rejection, 250K and 1M exact automatic success, non-36-byte and
  changed-radius rejection, disabled-source and injected-preflight rejection,
  and injected post-execution proof failure without an artifact.
- Exactness/performance: one warmup and five alternating option-free 1M pairs
  produce the same 36,000,375-byte LAS at SHA-256
  `28302b0ba60ff587976edec6546d2ff7156c720d0d37450299e6ea8ac47c4c00`.
  Pinned PDAL median is 1.892418782 seconds; automatic PDG median is
  0.352668079 seconds, or **5.366005x**. Status and diagnostics also match.
- Profile/stopping decision: B0085's hash-bound profile applies because B0093
  changes only admission and proof around the same direct executor. All 18
  kernels total 1.953650 milliseconds, only 0.575563% of the B0085 process
  median, and the radius query itself has a 0.480213% ceiling. No endpoint
  tuning reopens.
- Coverage/next: `filters.radiusassign` becomes the sixteenth automatically
  selected filter only in this exact measured envelope. Its radius selection
  is GPU-native; ordered assignment-expression evaluation remains an exact
  host finale. B0094 should first repeat and audit the already-profiled exact
  `approximatecoplanar(knn=8) -> ferry(Coplanar=>UserData)` composition before
  any option-free admission or new stage work.
- Evidence: B0093 qualification report, 16,671 bytes at SHA-256
  `2f29a6e96a82e3737fc93d31b7e2ab041a73ddc90f3d4713d8ef6b1709399b9d`;
  B0085 profile manifest, 3,799 bytes at SHA-256
  `c58936d5571c4c81468c95a3342965a24584adedfdd0157c31a26cb0b0f3b59a`.

## D0155 — Repeat but do not yet qualify the coplanar composition

- Date: 2026-08-11
- Status: accepted clean exact repeated routing evidence; current performance
  qualification, profile/stopping evidence, direct-composition calibration,
  and automatic selection remain open.
- Evidence: B0094's clean one-warmup/five-pair 1M gate measures
  4.108373629 seconds pinned PDAL versus 0.990323169 seconds explicit resident
  PDG, or **4.148518x**. All processes preserve status, empty diagnostics,
  artifact count, and the same 36,000,375 output bytes at SHA-256
  `0c52732a847e4427955a1b22525d58d3b4638d7fa35e2afebde7e437786b3245`.
- Execution honesty: every candidate uses device placement, direct LAS output,
  and one predicted/observed planner-owned index. Direct source and record-
  summary use are false. Executor-calibration and boundary-accounting matches
  are also false because the existing `approximatecoplanar` model was measured
  against the ordinary resident executor, not this direct-output composition.
  Do not reuse that model for public selection.
- Profile/repair boundary: fresh events reproduce the 194 MB ambiguity repair
  round trip, but the current exact-repair counters identify only NNDistance
  and statistical outlier; approximate-coplanar row counts and KD3 use are not
  auditable. B0069's older Nsight report binds different implementation,
  pipeline, engine, and output hashes. It remains historical routing evidence,
  not a current hash-bound stopping or automatic-admission profile.
- Decision/next: B0095 adds exact approximate-coplanar repair telemetry and a
  fresh output-bound profile on the existing repair-triggering shape. Only
  after that profile closes further roughly 5--10% reusable optimization may a
  separate direct-composition calibration ladder be fitted. No kernel, model,
  or selector change is authorized by B0094 alone.
- Evidence artifacts: B0094 benchmark report, 29,763 bytes at SHA-256
  `b0f8c017d70567c6b706a65671192e197ff993b6b951fe4115b3a7cd95f799f4`;
  fresh stats, 12,050 bytes at SHA-256
  `d7ea3b1d83722fe91f08a4447d14016a4f82ab3749c47a5033555576481edf23`.

## D0156 — Close coplanar endpoint tuning and proceed to a separate composition fit

- Date: 2026-08-11
- Status: accepted exact repair observability and current profile/stopping
  evidence; current-binary performance qualification, calibration, and
  automatic selection remain false.
- Telemetry contract: clean commit `f69fe3204` attributes the shared
  eigensystem repair specifically when `filters.approximatecoplanar` triggers
  it. Stats report trigger count, ambiguous/incomplete/repaired row counts,
  KD3 use, symmetric repair-transfer bytes, and nested/aggregate host-repair
  time. The benchmark runner's optional parser rejects malformed or
  inconsistent telemetry, while
  `--require-approximate-coplanar-host-repair` fails closed unless a positive
  repair is observed in explicit resident mode.
- Exact proof: the forced-incomplete unit remains byte-identical and observes
  the new values. The real 1M B0094 shape reports one trigger, 2,145 ambiguous
  rows, zero incomplete rows, 2,145 repaired rows, one KD3 use, and exactly
  97,000,000 bytes in each direction. Profiled output reproduces B0094's
  36,000,375-byte SHA-256
  `0c52732a847e4427955a1b22525d58d3b4638d7fa35e2afebde7e437786b3245`.
- Profile/stopping decision: the clean output-bound profile records 20 kernels
  totaling 21.403776 milliseconds. That is only 2.460417% of the
  0.869924635-second unprofiled command interval. Exact host KD3 repair owns
  0.461688645 seconds (53.072258% of command wall), but B0070 already removed
  99.377286% of its transfer for only 0.491159% shell-wall improvement; the
  remaining pinned ordered repair has no exact-proved device replacement.
  Therefore no reusable optimization clears the roughly 5--10% gate and the
  endpoint is sufficiently optimized for now.
- Coverage/next: B0094's 4.148518x paired result binds its older engine, while
  B0095's profile binds the telemetry-enabled current engine. Do not combine
  them into a current qualification. GPU-native work remains bounded and tied/
  incomplete rows still repair on host. No selector or ordinary-stage model
  changes. B0096 must measure current paired PDAL/direct rows before it may
  establish performance qualification or fit a separately named 50K--16M
  direct-composition model; it may not borrow the ordinary resident model or
  add public admission.
- Evidence: B0095 manifest, 5,184 bytes at SHA-256
  `7afc749d1894830fe83010366b28c1d61fee9a312a609678a25449788aebf5b8`;
  Nsight report, 1,557,989 bytes at SHA-256
  `4897f3d0aec3e775edbbd3550d3579b93589275286b03f601dc3dd448060b37f`;
  profiled stats, 12,426 bytes at SHA-256
  `bd091963f40848e44f3aede0b468f0b83d1a4deca8a5e52607173ee4812aa693`;
  unprofiled stats, 12,405 bytes at SHA-256
  `c6de929058694b0d3500673a960151c6bd7a4679314313f17dc15d49c4b5467e`.

## D0157 — Qualify and separately calibrate the direct coplanar composition

- Date: 2026-08-11
- Status: accepted exact explicit-composition performance qualification and a
  bounded direct-output placement model; automatic selection remains false.
- Measurement: clean current-binary B0096 pairs pinned PDAL against explicit
  resident PDG at 50K, 250K, 1M, 2M, 4M, 8M, and 16M. The 50K candidate is an
  exact host control at 0.484603x. Every 250K--16M candidate is byte- and
  diagnostic-exact, requires the direct-output shared-index executor and
  positive coplanar host repair, observes one index, and wins from 2.476664x
  through 5.048595x. The 1M current-binary result is 4.248119x.
- Model: `approximatecoplanar-direct-compose` fits only the six positive rows'
  device-minus-host residual after subtracting planner-owned startup,
  transfers, packing, shared-index build, and synchronizations. Its host
  residual is 87,387,779.74090123 fixed nanoseconds plus
  3,385.876023673574 nanoseconds per point; the device residual is zero. The
  envelope is 250K--16M. The 50K control is retained for winner validation but
  excluded from the positive-region fit.
- Admission: RuntimePlacement requires exactly one linear uncompressed-LAS
  reader, default-threshold `approximatecoplanar(knn=8)`, one exact
  `Coplanar=>UserData` ferry, one LAS writer, one resident region/index, the
  measured 25/2-byte upload/spill and 112-byte index-build layout, a 36-byte
  input record, and explicitly required direct output. Drift fails closed to
  the prior ordinary model or unavailable placement. The ordinary
  `approximatecoplanar` model is unchanged.
- Execution honesty: the qualified executor uses direct LAS publication but
  intentionally retains a PointView source for pinned KD3 repair. Stats may
  report calibration match true only for that no-direct-source composition;
  boundary-accounting match remains false because the model is fitted to the
  measured executor while planner byte diagnostics retain calibration-default
  accounting. No automatic pipeline envelope or public selector is added.
- Acceptance: focused runtime/profile tests pass; the complete placement audit
  is 149/149 over 34 models and all 155 unique report hashes verify. B0095's
  matching-engine exact profile keeps endpoint tuning closed. B0097 is the
  next smallest task: a cheap option-free automatic-admission and fail-closed
  process gate around this unchanged measured route.
- Evidence: the seven B0096 reports and hashes are recorded append-only in
  `BENCHMARKS.md`; implementation commit `04a6803de` contains the isolated
  model and strict matcher, and `cba52b2ca` closes the reviewed format-7 gate.

## D0158 — Automatically admit the measured direct coplanar composition

- Date: 2026-08-11
- Status: accepted exact bounded automatic admission and current-binary
  performance qualification; no kernel/model change and no broader envelope.
- Admission: clean implementation commit `e36ac6bbd` recognizes only the
  four-stage regular-LAS graph
  `approximatecoplanar(knn=8) -> ferry(Coplanar=>UserData)` with an
  uncompressed LAS output. RuntimePlacement remains authoritative and must
  select the isolated `approximatecoplanar-direct-compose` model, one region,
  and the B0096 format-7/layout/cardinality envelope. Options, topology,
  source/output format, model/device mismatch, and preflight rejection decline
  before publication and execute the unchanged pinned fallback when the proof
  variable is absent.
- Execution proof: automatic success requires direct LAS publication without
  the mapped direct-source route, record-summary claim, or host-XYZ mirror;
  exactly one observed index; and, for the qualification proof, exactly one
  positive pinned-KD3 repair with positive symmetric transfer bytes. Injected
  proof failure occurs before atomic publication. The physical matrix proves
  required and option-free success, exact fallback across the negative
  controls, and no artifact after preflight/proof failure.
- Performance: one warmup and five alternating current-binary 1M pairs are
  byte- and diagnostic-exact at 4.150524909 seconds pinned PDAL versus
  0.981067757 seconds automatic PDG, or 4.230620x. The fresh automatic capture
  reproduces the exact output and records 21.257320 milliseconds across 20
  kernels, only 2.166754% of automatic wall. B0095/B0070 already close the
  dominant pinned host-repair direction, so no reusable optimization reaches
  the 5--10% gate and endpoint tuning remains closed.
- Coverage: functional support remains catalog-wide through exact fallback.
  The bounded neighborhood/eigensystem/projection work is GPU-native, but
  tied/incomplete eigensystems still repair on host. The exact B0096
  composition is performance-qualified and now automatically selected only
  from 250K through 16M on the measured format-7 SM89 profile. None of these
  categories broadens the others.
- Next: B0098 is measurement-only: repeat and profile the already-native,
  calibrated `filters.neighborclassifier` lane on the current binary, starting
  with a bounded 1M complete pipeline. The hypothesis is that a shared-index
  resident/public composition can preserve the older 3.743x diagnostic value;
  do not add a selector or kernel until the current profile names the limiter.
- Evidence: B0097 paired report, 16,790 bytes at SHA-256
  `942843e9d3f1d8adc8189eecef8660d5719befdf99ca3aaff8c57b40432d1f3f`;
  profile report, 1,557,511 bytes at SHA-256
  `b61a5e4a895af66772483b78995662b50fb189cb49884656f57e6f7d56e8ec29`;
  profile manifest, 4,963 bytes at SHA-256
  `3414a5fdf9b593ddd9a92caaf38b2bfe33c08cdb8256520e020c9af798d48070`.

## D0159 — Retain the exact direct neighborclassifier boundary and target tie repair

- Date: 2026-08-11
- Status: accepted explicit bounded performance qualification and next-task
  routing. Functional support is unchanged; GPU-native work remains partial;
  no model or automatic selection is added.
- Measure/profile first: the unchanged ordinary 1M resident route is byte- and
  diagnostic-exact at 3.597672579 seconds pinned PDAL versus 1.057894184
  seconds PDG, or 3.400787x. Its fresh 18-launch profile totals only 16.184340
  milliseconds, so another neighbor query or vote kernel cannot clear the
  roughly 5--10% complete-wall gate.
- Reused boundary: clean commit `d59d3d151` extends B0068's exact mapped-LAS
  Classification storage/overlay only to the strict explicit
  `neighborclassifier(k=7)` three-stage shape. It retains the planner-owned
  index, record-summary configuration, no-host-XYZ proof, canonical atomic
  output, and host fallback outside the envelope. Changed `k` and disabled
  source fail without publication. The lane remains behind
  `PDG_EXPERIMENTAL_DIRECT_CLASSIFICATION_OUTPUT`.
- Performance: five alternating clean exact pairs measure 3.590663835 seconds
  pinned PDAL versus 0.906745056 seconds direct PDG, or 3.959949x. The direct
  boundary removes 14.287736% from the preceding ordinary median. Output is
  36,000,375 bytes at SHA-256
  `0c37dfecf3e3e261b768b055b4dbb3cc756dcc77111a4eee977c8f69b6cc4d78`.
  This exact explicit 1M composition is performance-qualified. It remains
  uncalibrated and is not automatically selected.
- Fresh stopping profile: the direct capture records 20 launches totaling
  16.164030 milliseconds, only 1.782643% of direct process wall. Device work
  is sufficiently optimized for now. A disposable, reverted attribution probe
  instead finds 1,947 conservative tied rows and about 580 milliseconds in the
  full pinned KD3 build/ordered repair. That measured bridge is a material,
  reusable hypothesis and prevents closing the endpoint.
- Next: B0099 adds exact neighborclassifier repair counters/timing, then tests
  whether every candidate at a conservative tied boundary has the same vote
  on the existing planner-owned query. Invariant rows may bypass KD3; a
  positive non-invariant tie must still repair exactly. Retain only an exact
  complete-pipeline improvement of roughly 5--10%. Do not add a general kNN
  kernel, model, broader envelope, or selector first.
- Evidence: B0098 direct report, 36,366 bytes at SHA-256
  `581402fdc200f9d7ce5f19387feb8be2d393a5ec3145bb17a91d71c20481ef6c`;
  profile report, 1,490,313 bytes at SHA-256
  `5cca54155ea6939c9943294d2a405b05b35a4780c99dcf81c3d0a0ec46c9cd5a`;
  profile manifest, 4,933 bytes at SHA-256
  `d4f0776249c59f6970cc7c0c2db99ecefca79fdbf7945cafc1aa770005f3d9e7`.

## D0160 — Retain neighborclassifier repair telemetry and close endpoint tuning

- Date: 2026-08-11
- Status: accepted production observability and measured no-go; the exact
  repair path is unchanged and the endpoint is sufficiently optimized for
  now.
- Telemetry: clean commit `e1ba240c5` attributes conservative
  neighborclassifier repair separately in resident stats. The exact B0098 1M
  route reports 1,947 distance-tie rows, zero incomplete rows, one pinned KD3
  use, and 0.594524233 seconds of exact host repair while reproducing the
  accepted output hash. Focused duplicate and zero-repair fixtures prove both
  positive and absent accounting without adding a synchronization.
- Prototype decision: a fully reverted boundary-status prototype safely
  discharged 1,587 interior ties. A second fully reverted exact full-boundary
  class-count proof discharged 359 of the remaining 360 rows. One
  non-invariant boundary row still required pinned KD3, leaving the dominant
  repair span at 0.579699672 seconds and no meaningful complete-process gain.
  Reproducing nanoflann's data-dependent equal-distance selection would require
  a second private index or an unproved answer, so it is rejected under the
  shared-index and exact-output contracts.
- Coverage: functional support, partial GPU-native status, B0098's explicit 1M
  performance qualification, and non-automatic placement remain independent
  and unchanged. No prototype code, wider tie contract, calibration, or
  selector is retained.
- Next: B0100 measures and profiles the current exact
  `filters.radialdensity(radius=1.01)` complete pipeline on a clean 1M binary.
  The hypothesis is explicit: the old dirty one-shot was exact at 4.661x and
  the radius stage can plausibly share the planner-owned index/resident data
  with another radius consumer. Do not change execution until the current
  complete-process profile names the limiter.
- Evidence: B0099 production stats, 11,396 bytes at SHA-256
  `eea91422a5864991504d9d33a5a28cf79b92c03224ca836479f0ff06a7742a5e`;
  manifest, 4,313 bytes at SHA-256
  `3baecf91211bd544030d0f31f9e6e1dc63c5da18b2f83204e789791b75143e67`.

## D0161 — Qualify radial density and route the host-boundary follow-up

- Date: 2026-08-11
- Status: accepted exact explicit 1M performance qualification; automatic
  selection remains false and the existing functional/native envelope is not
  broadened.
- Measure first: on clean commit `f95297538`, the exact
  `radialdensity(radius=1.01)` format-7 pipeline measures 3.119373264 seconds
  pinned PDAL versus 0.660844528 seconds forced CUDA, or 4.720283x, across five
  alternating pairs. Output bytes and diagnostics are identical in every run.
- Profile: the output-bound capture records 16 kernels and 2.973280
  milliseconds total, only 0.449921% of candidate wall. The radius query itself
  is 2.702432 milliseconds; every build/support kernel together is 0.270848
  milliseconds. Therefore device query/index tuning and shared-index launch
  elimination cannot produce a 5--10% end-to-end improvement on this endpoint.
- Route: the only material hypothesis is the inferred 99.550079% host-side
  LAS/PointView/extra-dimension output boundary. B0101 may prototype a reusable
  resident path that keeps the planner-owned radius product and feeds
  `RadialDensity` directly into an exact point-program consumer plus canonical
  LAS output. A specialized standalone writer or a private spatial index is
  rejected. Retain the prototype only after an exact complete-pipeline 5--10%
  gate; calibration and public selection remain separate work.
- Coverage: functionally supported yes; GPU-native yes for the bounded exact
  radius count/scaling work; performance-qualified only for B0100's explicit
  1M `radius=1.01` forced-CUDA shape; automatically selected no.
- Evidence: paired report, 16,790 bytes at SHA-256
  `7cbe6e413d897c13ef555699cf37e5655209168fe5c5086ba835a481bc3df27c`;
  NCU report, 1,284,289 bytes at SHA-256
  `cd00a78b5af93c9a805fe78ae75a8ccf0ddbbaf7f327f3fa3cb35b511049919c`;
  profile manifest, 4,353 bytes at SHA-256
  `9f6e2effcd812325420f8832185ab5710e7ea687ab97b7b22727a268600b7de7`.

## D0162 — Retain resident radial-density composition and close the endpoint

- Date: 2026-08-11
- Status: accepted exact explicit 1M resident composition and measured stopping
  decision; no placement model, automatic selection, or wider option envelope.
- Implementation: clean commit `29451ec13` makes the existing planner-owned 3D
  radius product a resident `RadialDensity` producer for one adjacent exact
  point program. The bounded public proof is the VLR-free format-7 graph
  `radialdensity(radius=1.01) -> assign(UserData=1 where
  RadialDensity>=0.2) -> default LAS`. The binary64 density stays on device;
  only the one-byte result column returns before atomic canonical publication.
  Changed radius/assignment, incompatible source/output/profile/layout, and
  disabled direct source decline or fail closed before publication. A private
  index and specialized writer remain prohibited.
- Performance: five clean alternating resident pairs measure 3.097940201
  seconds pinned PDAL versus 0.345532286 seconds PDG, or 8.965704x. A separate
  five-pair clean same-binary forced-hybrid control measures 0.611763765
  seconds, so the resident boundary is 1.770497x that path and removes
  43.518674% of complete wall. All routes reproduce the same 36,000,375-byte
  output at SHA-256
  `a298621dc7341285c945d5cb658a078b1a9840130802e36279bbed33802d3a95`.
- Profile and stopping decision: the output-bound NCU capture reproduces that
  hash and records 19 launches totaling 3.057664 milliseconds, only 0.884914%
  of resident median wall. `radiusCountsKernel` owns 2.701376 milliseconds,
  `assignKernel` 0.025792 milliseconds, and all other device work 0.330496
  milliseconds. A matching unprofiled command attributes 0.173310631 seconds
  to CUDA device/profile discovery and initial placement: driver/context
  initialization plus actual-free-memory input that would move to first
  allocation if deferred. No clear reusable 5--10% removal remains, so this
  endpoint is sufficiently optimized for now.
- Coverage: functionally supported yes; GPU-native yes for the bounded radius
  query/scaling and adjacent assignment in this region; performance-qualified
  only for this explicit 1M VLR-free format-7 shape; automatically selected no.
- Next: B0102 physically qualifies, measures, and profiles the existing
  compiled ordinal path in a bounded direct-LAS
  `decimation(step=2) -> assign` pipeline. The explicit hypothesis is that one
  compaction/assignment/publication region can cover an unmeasured catalog
  stage while avoiding PointView/residency boundaries. No selector or model
  precedes the exact complete-process gate.
- Evidence: resident paired report, 32,379 bytes at SHA-256
  `8920adb2ec094e69418a7936a4f4b2bb6a0a2a761a2eecc125cfc6068bea1a3c`;
  hybrid-control report, 16,807 bytes at SHA-256
  `a6ccabd56e75f2bf864c9bb78492d93510fccaffb93456e1318e1a53848ed970`;
  NCU report, 1,483,162 bytes at SHA-256
  `acdb087cb3d3641b5c7b30df33ecab7dbecac71575c43fc1ade0cd8651ddd326`;
  profiled/unprofiled stats, 11,271/11,238 bytes at SHA-256
  `956cdcb222f08337bedb6a448fea8114a89cd75c74c93ba3c07d8096435bff8f` /
  `f80e4c2972fe447e09327efb27c199d1beb3aa108390ba63ab5a9cdf8ad3d730`;
  command/input/binary/output/proof manifest, 6,736 bytes at SHA-256
  `a4c8afbb9643670c36ea488510bad64814d611cc3cdbc6ece27f7ad40c8a1487`.

## D0163 — Physically accept ordinal CUDA coverage but reject performance

- Date: 2026-08-11
- Status: accepted exact forced GPU-native ordinal coverage; performance
  qualification and automatic selection rejected.
- Validation: the existing `filters.decimation`, `filters.head`, and
  `filters.tail` device implementation passes eight focused physical unit and
  process gates across standard/streaming modes, global sequence, chunk
  boundaries, split graphs, and all three ordinal kinds. Memcheck, initcheck,
  racecheck, and synccheck are clean. This closes D0014's physical-runtime gap
  without changing code or widening its exact option envelope.
- Directional measurement: on clean commit `090d5a9aa`, the exact VLR-free
  format-7 `decimation(step=2) -> assign(UserData=1)` pipeline is only
  0.449898x pinned PDAL at 1M (0.218874453 versus 0.486497717 seconds) and
  0.762657x at 4M (0.788491272 versus 1.033874358 seconds). These one-pair rows
  are directional diagnostics, not performance qualification.
- Profile and prototype: the output-bound 1M NCU capture reproduces the exact
  18,000,375-byte artifact and records 64 launches totaling 241.376
  milliseconds. Per-column stable compaction owns 107.200 milliseconds;
  unpack, ordinal mask, assignment, and repack own the rest. Raising the
  existing chunk control to one whole-view chunk is slower at both 1M and 4M,
  so synchronization amortization is rejected without a code change. A deeper
  packed-record compaction/JIT would be a substantial new engine path whose
  ideal 1M ceiling still does not establish a CPU win; defer it pending a
  composition with measured resident value.
- Coverage: functionally supported yes; GPU-native yes for the bounded exact
  ordinal program envelopes; performance-qualified no; automatically selected
  no. Slower exact CUDA remains force-only coverage and is never described as
  a speedup.
- Next: B0103 physically qualifies, measures, and profiles the existing typed
  first-tie `filters.locate` reduction. The measured hypothesis is a resident
  terminal consumer that avoids a boundary, not a standalone reduction kernel
  assumed to win. No model or selector precedes that evidence.
- Evidence: 1M/4M diagnostic reports, 9,842/9,840 bytes at SHA-256
  `8964c0b2b4cfc3d76e4f75f5e4845ff22fe3405fe66ac9c8655bf1f9c911a6c0` /
  `47ff7714c764546c8a7bfdcca9a0c681493ec7f47cd9b8de28b3053fd51cc319`;
  NCU report, 3,504,725 bytes at SHA-256
  `f90620a114d7ebce86d64c0448768bd015768b87bbf83ba0e0535b77041c2784`;
  command/input/binary/output manifest, 5,199 bytes at SHA-256
  `1ea6fb242949d9ff6217f6bb016e44e0879f9fa3a82f9bfdffbc142f343bdb7e`.

## D0164 — Physically accept locate CUDA coverage but reject performance

- Date: 2026-08-11
- Status: accepted exact forced GPU-native locate coverage; performance
  qualification and automatic selection rejected.
- Validation: the existing `filters.locate` device implementation passes five
  focused physical unit/process gates covering typed first ties, PDAL
  sentinels, NaNs/infinities, coordinate decoding, chunk-global indices, and
  chained assignment. Memcheck, initcheck, racecheck, and synccheck are clean.
  This closes the physical-runtime gap without changing code or widening the
  exact option envelope.
- Directional measurement: on clean commit `fe4931ab2`, the exact VLR-free
  format-7 `assign(UserData=1) -> locate(Z,min)` pipeline is only 0.387892x
  pinned PDAL at 1M (0.167703020 versus 0.432344452 seconds) and 0.706034x at
  4M (0.600216677 versus 0.850124517 seconds). These one-warmup/one-pair rows
  are directional diagnostics, not performance qualification.
- Profile and decision: the output-bound 1M NCU capture reproduces the exact
  411-byte one-point artifact and records ten launches totaling 54.272
  milliseconds: assignment 34.304, locate block reduction 15.488, and final
  reduction 4.480 milliseconds. The remaining 378.072452 milliseconds are
  outside kernels. The current candidate loses by roughly 250 milliseconds at
  both 1M and 4M; even a B0101-sized direct-boundary removal only brings the 4M
  row near break-even. Legal point-program pushdown plus a direct mapped source
  would be a substantial new executor, not a cheap prototype proving a 5--10%
  complete-process win, so optimization is deferred until an already-resident
  consumer supplies stronger measured value.
- Coverage: functionally supported yes; GPU-native yes for the bounded exact
  typed first-tie reduction envelope; performance-qualified no; automatically
  selected no. Slower exact CUDA remains force-only coverage and is never
  described as a speedup.
- Next: B0104 physically qualifies, measures, and profiles the existing
  `filters.info` reduction. Its record-summary reuse hypothesis must be tested
  against the current complete process before any implementation, model, or
  selector change.
- Evidence: 1M/4M diagnostic reports, 9,775/9,783 bytes at SHA-256
  `03a7265b49f429e6701715855240538671c98a8300fe612bad3f46e5a9e61f7e` /
  `dca31125f0d4ef1b935d668435e9b433d3df6b67c83021b21a84838d6a68b9aa`;
  NCU report, 566,363 bytes at SHA-256
  `d64a0a9739f023d0e83b9ab033d8d878d31cddb39ebf2a4a655d36007307b637`;
  command/input/binary/output manifest, 5,003 bytes at SHA-256
  `86e7813c721d401a1dc29480aca77512511f15f0a7118d00ec57d9c7667f0427`.

## D0165 — Physically accept info CUDA coverage but keep host selection

- Date: 2026-08-11
- Status: accepted exact forced GPU-native option-free info coverage;
  performance qualification and automatic selection rejected.
- Validation: the existing `filters.info` device implementation passes its
  131,103-point physical property, one forced complete-process differential
  including metadata, and memcheck, initcheck, racecheck, and synccheck. The
  accepted CUDA envelope remains option-free; point/query forms stay on the
  exact host path.
- Directional measurement: on clean commit `c1a0633e7`, the exact VLR-free
  format-7 `readers.las -> filters.info -> writers.las` pipeline is only
  0.525893x pinned PDAL at 1M (0.296845919 versus 0.564460173 seconds) and
  0.744854x at 4M (1.109951163 versus 1.490158422 seconds). These
  one-warmup/one-pair rows are directional diagnostics, not performance
  qualification.
- Profile and prototype: the output-bound 1M NCU capture reproduces the exact
  36,000,375-byte LAS and 4,717-byte metadata artifacts. Eight streamed chunks
  launch eight block and eight final reductions totaling only 0.26863
  milliseconds, 0.047591% of candidate wall. Exact PDG-host info takes
  0.313470297 seconds while the same-binary reader/writer-only control takes
  0.310210637 seconds. Perfect record-summary reuse therefore has at most a
  3.259660-millisecond/1.03986% ceiling in this ordinary pipeline, below the
  required 5--10%; no code, private summary, placement model, or selector is
  added.
- Coverage: functionally supported yes; GPU-native yes for the option-free
  exact bounds/count envelope; performance-qualified no; automatically
  selected no. Ordinary host selection remains the measured winner.
- Next: B0105 physically rechecks and clean-measures
  `filters.hag_nn(count=4)`, the strongest historical unqualified shared-index
  lane, before any implementation or selector work.
- Evidence: forced metadata differential, 2,883 bytes at SHA-256
  `e5796a4e0384f620cabd2ae37022cc7aaa7f2b3d76e00f7fa001b6d0c4f5c24f`;
  1M/4M directional reports, 9,823/9,837 bytes at SHA-256
  `3af02a3e7c8868cfd290f21e3a7357a03ed89fadb348ce315eecaea24dfad4e3` /
  `b54ad3aad1e786107a840cb62ca6c25aea6cc7800602b06d51c97cdb0751d883`;
  host/bare controls, 9,819/9,813 bytes at SHA-256
  `40a4073fe31267473a25cc12bb516037075e25bdc41d0497d9c51192d78435b7` /
  `b3b9763616db84ac91afc81ee20ab2d575e35eb21ac91ee60ef1635d4da07f1e`;
  NCU report, 805,456 bytes at SHA-256
  `356a079b24c267d790e215fe5eee6dc28db417f639cb0b207df120fedf39ddde`;
  command/input/binary/output/metadata manifest, 6,780 bytes at SHA-256
  `f1c6fe7260b90e9fe8f78c4a5582baadf57c5ff1eae8a6a9349480106897e66a`.

## D0166 — Cleanly qualify the forced count-four HAG fixture

- Date: 2026-08-11
- Status: accepted as performance-qualified only for the named forced
  1,000,002-point `filters.hag_nn(count=4)` distinct-fifth-candidate fixture;
  automatic selection and ordinary-data performance claims rejected.
- Validation: the current binary passes the 12-test count-four,
  masked-index, and resident-lifecycle lane physically and under memcheck,
  initcheck, racecheck, and synccheck. The exact envelope and host-repair
  behavior remain D0098's; B0105 changes no runtime code.
- Performance: on clean commit `d7e5dd508`, five exact alternating pairs
  measure pinned PDAL at 0.996831814 seconds median and forced PDG at
  0.637868218 seconds median, or 1.562755x. Every run writes the same
  48,000,909-byte artifact at SHA-256
  `6e59b3a5b41cc5e219a5d39743c39ac7b2f09c84aa74b67e9b943e4cf41fe293`.
- Profile and next hypothesis: the output-bound capture reproduces that hash.
  Eighteen kernels total 3.39736 milliseconds/0.532612% of candidate wall;
  masked `knnGatherKernel` owns 3.05 milliseconds, while the exact HAG
  projection takes 0.07181 milliseconds. A no-code HAG-shaped extra-dimension
  control takes 0.345457608 seconds, leaving a 0.292410610-second/45.8419%
  surface. B0106 may cheaply force the existing plan through the ordinary
  planner-resident executor and retain it only for a 5--10% exact process win;
  it does not begin with a specialized direct publisher.
- Coverage: functionally supported yes; GPU-native yes for the proved counts
  one through four; performance-qualified only for this named forced count-four
  fixture; automatically selected no. Count-one-through-three and ordinary
  data remain unqualified.
- Evidence: five-pair report, 16,780 bytes at SHA-256
  `46c6cdd5b1e3e1bdb92e05ce7172b4b64b074b7cdf277ac9e0bde50611428618`;
  NCU report, 1,431,866 bytes at SHA-256
  `9c3de2bebffb2cde15a0429d7a52152936095d0969278ad9ebfc24faa58887ff`;
  publication control, 9,841 bytes at SHA-256
  `a03c0a850b5fb9d7c9a60325660b21a560ce9548498ca74fc64b47eaf687fb00`;
  command/input/binary/output manifest, 6,585 bytes at SHA-256
  `f5ff41eff9786ae8a82bca2b48b07045552c55cd88fa0892d0ebbe61e0c5e4d6`.

## D0167 — Reject ordinary HAG residency and test a reusable extra-dimension endpoint next

- Date: 2026-08-11
- Status: rejected and fully reverted; B0105's fixture-specific forced
  qualification remains unchanged.
- Prototype: an explicit proof-only path forced the existing
  `filters.hag_nn(count=4)` plan through one ordinary whole-view resident
  region with one planner-owned masked 2D index and the original upstream
  `writers.las(extra_dims=all)`. The bounded small process gate was byte-exact
  and rejected changed shape before publication.
- Directional result: after one warmup on the B0105 1,000,002-point fixture,
  shell elapsed time is 0.675 seconds without stats versus 0.657 seconds for
  the same prototype binary's forced-hybrid control and 1.089 seconds for
  pinned PDAL. All three write the exact 48,000,909-byte artifact at SHA-256
  `6e59b3a5b41cc5e219a5d39743c39ac7b2f09c84aa74b67e9b943e4cf41fe293`.
  The 2.740% resident loss is a cheap one-pair direction, not qualification.
- Cause and decision: proof telemetry records one predicted/observed index and
  exact boundary accounting, but the upstream-owned writer forces a
  33,000,066-byte spill. Validation/placement/preflight takes 0.165814939
  seconds and rewritten manager execution takes 0.377666605 seconds. Ordinary
  residency therefore cannot clear the 5--10% gate, and no runtime force,
  model, test, or selector survives.
- Next hypothesis: B0107 may prototype a reusable exact direct LAS publisher
  for one standard binary64 extra dimension on the format-7 fixture whose
  existing Extra Bytes VLR describes `OffsetTime` as unsigned-32. It must
  first pin PDAL's appended descriptor/header/record bytes and remain
  dimension-generic; retain only for a 5--10% exact win over the same-binary
  hybrid control. No automatic admission follows.
- Coverage: functionally supported yes; GPU-native yes in D0098's envelope;
  performance-qualified only for B0105's named forced fixture; automatically
  selected no. D0167 changes none of these categories.
- Evidence: prototype command/binary/output/stats manifest, 5,052 bytes at
  SHA-256
  `f97b7530645ce9cbc7cf69156ac3fe159d48dd852854c2ea8008bfd83db722fa`.

## D0168 — Retain the strict binary64 Extra Bytes publisher and measure placement next

- Date: 2026-08-11
- Status: accepted as an explicit exact reusable output boundary; no new
  performance qualification, placement model, or automatic admission.
- Envelope: B0107 accepts only uncompressed LAS 1.4 point format 7 with a
  375-byte header, 40-byte records, one canonical 192-byte unsigned-32
  `OffsetTime` Extra Bytes descriptor, no gap/EVLR/waveform/trailing data, and
  terminal `writers.las(extra_dims=all)`. The planner records that exact writer
  option without calling the upstream writer native. The publisher preserves
  the original 40 record bytes, appends one standard binary64 dimension as raw
  IEEE-754 bits, combines both descriptors in one VLR, recomputes bounds and
  return counts, widens records to 48 bytes, and atomically publishes. Every
  unsupported layout or option retains pinned PDAL before output side effects.
- Exactness and safety: focused units cover empty/one/multirow summaries,
  caller-buffer parity, undersized/cardinality rejection, VLR padding and
  descriptor drift, extent overflow, signed zero, infinity, and two NaN
  payloads. The physical process gate matches pinned-PDAL bytes, streams, and
  status; proves one planner-owned index, zero terminal/fallback spill,
  experimental-only activation, unforced fallback, and sentinel preservation.
  ASan/UBSan and memcheck/initcheck/racecheck/synccheck pass.
- Directional result: on clean commit `6cb42de3e`, one warmup per route and one
  warm trio on the B0105 1,000,002-point fixture measure pinned PDAL at 0.98
  seconds, the same binary's forced-hybrid control at 0.64 seconds, and direct
  PDG at 0.59 seconds. Direct is 1.084746x/7.8125% below its retention control
  and 1.661017x/39.7959% below pinned PDAL.
  Both and the fresh profile write the exact 48,000,909-byte artifact at
  SHA-256 `6e59b3a5b41cc5e219a5d39743c39ac7b2f09c84aa74b67e9b943e4cf41fe293`.
  This clears the cheap retention gate but is not the five-pair performance
  qualification protocol.
- Profile and next decision: 18 kernels total 3.43344 milliseconds/0.581939%
  of the 0.59-second directional candidate; `knnGatherKernel` is 3.087392
  milliseconds and publication is only 0.015886878 seconds in the unprofiled
  stats run. That same run spends 0.176617650 seconds/31.5389% of wall in
  validation, placement, and preflight, including a 0.095280021-second initial
  placement result later superseded by the strict force. A clear reusable
  5--10% hypothesis remains, so the endpoint is not declared sufficiently
  optimized. B0108 may prototype an exact explicit-force placement/preflight
  fast path, but must not alter calibrated or automatic admission.
- Coverage: functionally supported yes; HAG count four remains GPU-native in
  D0098's envelope; performance-qualified only for B0105's named forced
  fixture, not for B0107's one-pair endpoint; automatically selected no.
- Evidence: command/input/binary/output/profile manifest, 10,395 bytes at
  SHA-256
  `e42502e078d8c288e011184d658e54b91a4df208f5fb6ee1be181e3d8546e0eb`;
  NCU report, 1,431,054 bytes at SHA-256
  `6d7e8b10b17f3e266b1584121ffafeccb3cd59f3a77daeae5a85294823c80481`;
  direction/profile stats, 10,073/10,067 bytes at SHA-256
  `8067e94921f99cdc9e7b49dcd73243a152dfbeb2973eda60818dc12c7e1b73b1` /
  `75e4b12d5632802165318769e02bd49ddebaf067daa466eca83308bb0c99460c`.

## D0169 — Reject the explicit-force placement shortcut and stop the HAG endpoint

- Date: 2026-08-11
- Status: rejected and fully reverted; the B0107 strict publisher and B0105
  named-fixture qualification remain unchanged.
- Gate: on the same 1,000,002-point B0107 graph and disposable prototype
  binary, one warmup plus one stats-free pair measures 0.55 seconds for the
  existing control and 0.53 seconds for the explicit-force shortcut. Both
  produce the exact 48,000,909-byte pinned-PDAL artifact at SHA-256
  `6e59b3a5b41cc5e219a5d39743c39ac7b2f09c84aa74b67e9b943e4cf41fe293`.
  The 3.636364% reduction misses the required 5--10%; it is directional
  rejection evidence, not a qualification.
- Attribution: stats-enabled diagnostics record
  validation/placement/preflight at 0.166577409 seconds for the control and
  0.163389308 seconds for the shortcut. Removing 0.086755750 seconds labeled
  initial placement causes device/profile work to rise from 0.076631200 to
  0.160171180 seconds, and command-before-stats is slightly slower. The
  measured interval is chiefly CUDA startup attribution rather than a
  reusable independent optimization.
- Retained change: the fast path, disable switch, and telemetry are gone. The
  B0107 experimental request now checks the compiled strict publisher plan
  before forcing residency, so an unsupported descriptor delegates to pinned
  PDAL. The focused physical process gate proves exact bytes, stdout, stderr,
  and status for that fallback.
- Stopping decision: together with B0107's sub-1% kernel and sub-3%
  publication fractions, the rejected shortcut leaves no clear reusable
  5--10% surface. The HAG direct-output endpoint is sufficiently optimized for
  now. Functional support remains yes; GPU-native remains the proved HAG
  count-one-through-four envelopes; performance-qualified remains only
  B0105's named forced count-four fixture; automatically selected remains no.
- Next: B0109 measures and profiles the existing exact
  `filters.nndistance(k=10) -> writers.las(extra_dims=all)` pipeline on the
  same canonical Extra Bytes input. Extending the reusable publisher is only a
  hypothesis and requires a 5--10% current-binary boundary opportunity before
  code.
- Evidence: rejected-prototype manifest, 4,075 bytes at SHA-256
  `68509722ec9e459177b8433d01a8bedbfa8900f0f3b118e9f7b27d66796226c3`;
  shortcut/control stats, 10,077/10,111 bytes at SHA-256
  `efd766d6a9485387d951a70fe9d1e277eec7ad265a7cbccbe81d68f399366271` /
  `bc8e64f3a723884347711a05f23852ee65c3359b56ae09f25b2c94c68481141c`.
  The manifest preserves the prototype's uncommitted/overwritten-binary
  limitation and forbids using this row for a positive claim.

## D0170 — Prototype NNDistance direct binary64 publication next

- Date: 2026-08-11
- Status: accepted as a measurement-backed prototype direction; no code,
  qualification, placement model, or automatic-selection change yet.
- Measurement: on clean commit `e9f360855`, one warmup plus one directional
  sample of the exact 1M `filters.nndistance(k=10) ->
  writers.las(extra_dims=all)` graph measures 2.01 seconds pinned PDAL and
  0.66 seconds exact CUDA hybrid, or 3.045455x. Both write the same
  48,000,909-byte artifact at SHA-256
  `a95bad2a632244e76aaf205a3d8174a87cbaf631b9982c5185b43b2e3576da56`.
  These single rows are directional and unqualified.
- Profile and hypothesis: a same-binary output-shaped assignment control takes
  0.39 seconds, leaving a 0.27-second/40.9091% ceiling above it. The fresh
  exact-output NCU capture totals only 25.767520 milliseconds across 36
  kernels, of which the existing planner-owned Morton-BVH gather owns
  25.358368 milliseconds. The material hypothesis is resident lifecycle and
  direct output, not a new kNN kernel.
- Next: B0110 may explicitly admit only this existing NNDistance `kth,k=10`
  region to B0107's generic one-binary64 publisher. Retain it only for exact
  bytes/diagnostics and a 5--10% stats-free same-binary process win. Preserve
  tie/incomplete repair, one planner-owned index, atomic output, and fallback
  before side effects; add no automatic route or model.
- Coverage: functionally supported yes; GPU-native unchanged for the existing
  proved NNDistance envelopes; performance-qualified unchanged; automatically
  selected unchanged.
- Evidence: command/input/binary/output/profile manifest, 4,950 bytes at
  SHA-256
  `df2d9ec36ca5b91d5a60788aa5fec86e790278552ba7fda486c8ab8e594a8526`;
  NCU report, 2,209,063 bytes at SHA-256
  `d74c559c252802fa181962e46c879aec45702fdb7a3914ef4f5f99d7fb0851a8`.

## D0171 — Retain explicit NNDistance direct publication and test source reuse

- Date: 2026-08-11
- Status: accepted as an explicit exact reusable output composition; no new
  performance qualification, placement model, or automatic admission.
- Envelope: only `readers.las -> filters.nndistance(mode=kth,k=10) ->
  writers.las(extra_dims=all)` joins B0107's strict format-7/`OffsetTime`
  layout. The generic publisher truncates PDAL's canonical NNDistance
  description to the 32-byte LAS field, appends raw binary64 values, preserves
  source records, and publishes atomically. Every other mode/k/writer/layout
  stays on the prior exact path.
- Exactness and safety: the physical process gate matches pinned bytes,
  streams, and status; proves one planner-owned index and zero terminal or
  fallback spill; positively checks the descriptor and values; and covers
  required rejection plus exact experimental-only fallback. Focused
  ASan/UBSan and memcheck/initcheck/racecheck/synccheck lanes pass.
- Directional gate: on clean commit `1ba0a1ffa`, the exact same-binary hybrid
  control takes 0.71 seconds and direct takes 0.57 seconds after one warmup per
  route. The 19.7183% reduction clears the retention gate, and both match the
  pinned 48,000,909-byte output at SHA-256
  `a95bad2a632244e76aaf205a3d8174a87cbaf631b9982c5185b43b2e3576da56`.
  One pair is directional, not qualification.
- Profile and next: 36 fresh exact-output kernels total 25.856256
  milliseconds/4.53619% of direct wall; the existing shared gather owns
  25.446208 milliseconds. Kernel tuning is rejected. Unprofiled manager
  execution remains 0.238845412 seconds and direct-source proof is false, so
  B0111 may compose the existing mapped LAS resident source with this exact
  explicit endpoint. Retain only for another 5--10% same-binary win and do not
  alter calibrated or automatic admission.
- Coverage: functionally supported yes; GPU-native unchanged for the bounded
  NNDistance shared-index envelope; performance-qualified unchanged;
  automatically selected no.
- Evidence: command/input/binary/output/profile manifest, 5,998 bytes at
  SHA-256
  `a4bed268960a6380a1bc2968b78e9b10e5c1f729fe470314a5c471603cd11557`;
  NCU report, 2,207,116 bytes at SHA-256
  `c14136250a42e96d75ddfc0b9f75564eac6b01724e9f8e5ecd444c564e9c7500`;
  direction/profile stats, 10,071/10,068 bytes at SHA-256
  `d663c52c91ab294e53e63f53468ba7708675526d1ff8ef5d9bf6e5e6167ae1e6` /
  `7c1825d6289b8af0d96c4ea4b890f4ce8edbad1b7fe0ae717f597dbdbe0fbff8`.

## D0172 — Retain mapped-source NNDistance composition and stop the endpoint

- Date: 2026-08-11
- Status: accepted as an explicit exact reusable source/output composition;
  endpoint tuning stops, with no new qualification, model, or automatic route.
- Envelope and proof: only B0110's strict
  `readers.las -> filters.nndistance(mode=kth,k=10) ->
  writers.las(extra_dims=all)` force may hydrate the canonical one-extra-double
  layout through the mapped source. The physical process gate proves direct
  source use, record-summary-configured Morton-BVH, no host XYZ mirror, one
  planner-owned index, zero terminal/fallback spill, atomic publication, exact
  bytes/streams/status, and no-output source/layout rejection. Focused
  ASan/UBSan and all four Compute Sanitizer lanes pass.
- Directional gate: on clean commit `43237c2f8`, one warmup per route plus one
  stats-free same-binary pair improves from 0.58 seconds with mapped source
  disabled to 0.38 seconds enabled, or 1.526316x/34.4828%. Both routes and the
  profile match the pinned 48,000,909-byte artifact at SHA-256
  `a95bad2a632244e76aaf205a3d8174a87cbaf631b9982c5185b43b2e3576da56`.
  One pair is a retention gate, not performance qualification.
- Stopping decision: the fresh exact profile totals 26.086656
  milliseconds/6.86491% of wall, with 25.613088 milliseconds in the existing
  shared gather; publication is 14.145771 milliseconds. B0108 already proved
  the apparent startup/placement label is not independently removable and
  saves only 3.636% when bypassed. No clear reusable 5--10% surface remains, so
  the endpoint is sufficiently optimized for now.
- Coverage: functionally supported yes; GPU-native unchanged for the bounded
  NNDistance shared-index envelope; performance-qualified unchanged;
  automatically selected no.
- Next: B0112 repeats and profiles the existing exact
  `filters.hag_delaunay(count=3) -> writers.las(extra_dims=all)` graph on the
  current clean binary before any direct publisher/source extension. B0034's
  prior 1.314x row is a dirty-snapshot diagnostic; code requires a measured
  current-process 5--10% boundary opportunity.
- Evidence: command/input/binary/output/profile manifest, 7,284 bytes at
  SHA-256
  `292ad9867ee4a2825c172625eea94f274ce70a7a39e7edeac4834e6350d5099a`;
  NCU report, 2,322,014 bytes at SHA-256
  `a409b3b9b939062f436b69979c54fd4f162fbc1bf4d65976553583d03444255e`;
  direction/profile stats, 9,822/9,818 bytes at SHA-256
  `8ea2f5e64724e2af308aec67daba9ab5ad756e4e43f7abc56a9b3ac52242305a` /
  `4a2fc4c2a18108d19b536b406c3e0352996bf84c682a1180df4ee4f5bc5eb387`.

## D0173 — Prototype HAG Delaunay source/output reuse, not another kernel

- Date: 2026-08-11
- Status: accepted as a measurement-backed bounded prototype direction; no
  code, qualification, model, or automatic-selection change yet.
- Measurement: on clean commit `a28d209f2`, one warmup plus one alternating
  pair of the exact 1M `filters.hag_delaunay(count=3) ->
  writers.las(extra_dims=all)` graph takes 0.810889333 seconds pinned PDAL and
  0.607849511 seconds current CUDA hybrid, or 1.334030x. Both and the profile
  reproduce the 48,000,909-byte pinned output at SHA-256
  `4f4847ef2f48a18158a5ea1c5921c640f58071d3880292a7f02e4a8e9c8290e3`.
  This current clean direction supersedes B0034 for planning but is not a
  repeated performance qualification.
- Profile and choice: 18 device launches total only 2.342688
  milliseconds/0.385406% of candidate wall; the existing masked gather owns
  1.957472 milliseconds and HAG interpolation 0.111584 milliseconds. A
  same-binary same-record-shape control takes 0.40 seconds, leaving a
  0.207849511-second/34.1942% source/PointView/publication ceiling. The selected
  path is reuse of the existing strict mapped source and generic one-binary64
  publisher, not a standalone kernel change.
- Next: B0113 may explicitly compose those existing boundaries only around
  count-three HAG Delaunay on this strict fixture. Retain only for exact bytes
  and another 5--10% current-binary process win; preserve one planner-owned
  masked index, tie/incomplete repair, atomic publication, and existing
  calibrated/automatic behavior.
- Coverage: functionally supported yes; GPU-native unchanged for the exact
  count-three shared-index envelope; performance-qualified no; automatically
  selected no.
- Evidence: command/input/binary/output/profile manifest, 5,696 bytes at
  SHA-256
  `dec9924f4cf51e082583aad671c2eb8f59527fd7b1c1e4434b8931dd1317fedd`;
  paired report, 9,828 bytes at SHA-256
  `1b36ae397f4af401e2efb561650be9547369e7dea2c962440f038147600266aa`;
  NCU report, 1,431,585 bytes at SHA-256
  `7bb787dc6d45e1a2cf1b3de5aab264b7b6a24d3b134f09275ab6ae282240a162`.

## D0174 — Retain mapped-source HAG Delaunay composition and stop the endpoint

- Date: 2026-08-11
- Status: accepted as an explicit exact reusable source/output composition;
  endpoint tuning stops, with no new qualification, model, or automatic route.
- Envelope and proof: only B0112's strict
  `readers.las -> filters.hag_delaunay(count=3) ->
  writers.las(extra_dims=all)` force may hydrate the canonical one-extra-double
  layout through the mapped source. The physical process gate proves direct
  source use, record-summary configuration, no host XYZ mirror, one
  planner-owned masked 2D index, zero terminal/fallback spill, atomic
  publication, and exact bytes/streams/status. Positive fourth-candidate-tie
  and forced-grid incomplete-search fixtures prove that pinned host repair
  still feeds the direct publisher; required source/layout/option rejections
  leave no artifact. Focused ASan/UBSan and all four Compute Sanitizer lanes
  pass.
- Directional gate: on clean commit `59886d80d`, one warmup per route plus one
  stats-free same-binary pair improves from 0.55 seconds with the mapped source
  disabled to 0.41 seconds enabled, or 1.341463x/25.4545%. Both routes and the
  profile match the pinned 48,000,909-byte artifact at SHA-256
  `4f4847ef2f48a18158a5ea1c5921c640f58071d3880292a7f02e4a8e9c8290e3`.
  One pair is a retention gate, not performance qualification.
- Stopping decision: the fresh exact profile totals 2.411420 milliseconds/
  0.588151% of directional wall, with 1.960000 milliseconds in the existing
  shared gather; publication is 15.792846 milliseconds. The current 0.34-second
  output-shaped control removes the entire spatial query and intentionally
  changes HAG values, while B0108 already rejected the apparent independent
  startup/placement shortcut at 3.636%. No clear reusable independent 5--10%
  surface remains, so the endpoint is sufficiently optimized for now.
- Coverage: functionally supported yes; GPU-native unchanged for the bounded
  count-three shared masked-index envelope; performance-qualified no;
  automatically selected no.
- Next: B0114 cheaply tests mapped-source reuse for B0107's already-exact
  explicit HAG-NN count-four direct binary64 endpoint. Retain only for another
  exact 5--10% source-disabled/source-enabled process win; do not change a
  kernel, placement model, calibration, or automatic selector.
- Evidence: command/input/binary/output/profile manifest, 8,442 bytes at
  SHA-256
  `0cd2446c585d4ee90765affe02ffada6917cb2863c5ed7eb8f894e6386f9ab00`;
  NCU report, 1,548,174 bytes at SHA-256
  `5481fb0bae252423c4ff7e054915820c15639b1159c2f948126363b83d76e6ca`;
  direction/profile stats, 9,815/9,816 bytes at SHA-256
  `97fc8f3af8b2f559218f4dc13995276e39874b600150ace9a966246639e5c554` /
  `ddac5a45a4befcc3e9dac1151cc0fb83a40f1fa89dd4d387e2c2cc99997a8754`.

## D0175 — Retain mapped-source HAG-NN composition and stop the endpoint

- Date: 2026-08-11
- Status: accepted as an explicit exact reusable source/output composition;
  endpoint tuning stops, with no new qualification, model, or automatic route.
- Envelope and proof: only B0107's strict
  `readers.las -> filters.hag_nn(count=4) ->
  writers.las(extra_dims=all)` force may use the mapped source. The physical
  process gate proves record-summary configuration, no host XYZ mirror, one
  planner-owned masked 2D index, zero terminal/fallback spill, atomic output,
  and exact bytes/streams/status. Positive fourth/fifth-distance-tie and
  forced-grid incomplete-search fixtures prove pinned host repair through the
  direct composition; disabled or unsupported required sources leave no
  artifact. Focused ASan/UBSan and all four Compute Sanitizer lanes pass.
- Directional gate: on clean commit `052ff6678`, one warmup per route plus one
  stats-free same-binary pair improves from 0.56 seconds with the mapped source
  disabled to 0.37 seconds enabled, or 1.513514x/33.9286%. Both routes and the
  profile match the pinned 48,000,909-byte artifact at SHA-256
  `6e59b3a5b41cc5e219a5d39743c39ac7b2f09c84aa74b67e9b943e4cf41fe293`.
  One pair is retention evidence, not a new performance qualification.
- Stopping decision: the fresh exact profile totals 3.523550 milliseconds/
  0.952311% of directional wall, with 3.110000 milliseconds in the existing
  shared gather; publication is 12.601495 milliseconds. B0108 already rejects
  the apparent independent startup/placement shortcut at 3.636%. No clear
  reusable independent 5--10% surface remains, so the endpoint is sufficiently
  optimized for now.
- Coverage: functionally supported yes; GPU-native unchanged for the bounded
  count-four shared masked-index envelope; B0105's named forced count-four
  performance qualification is unchanged and B0114 adds none; automatically
  selected no.
- Next: B0115 measures and profiles the existing exact affine
  `filters.transformation -> pointwise assignment` fused pipeline on the
  current binary before selector or fusion changes. This is a reusable fusion
  measurement, not a new stage port.
- Evidence: command/input/binary/output/profile manifest, 7,284 bytes at
  SHA-256
  `cb96aadf5c4ab21b15751567c35ba70b1a0fb5b9030da107629150bbc352c84e`;
  NCU report, 1,543,282 bytes at SHA-256
  `2813e7563bbbd7ddc5caf1bb86e05861b16e6919e86a466ffb2164922ce66f4b`;
  direction/profile stats, 9,817/9,816 bytes at SHA-256
  `a90a4aa5d3a11831d748a54334ae341bc8b9ab31fce94844bc19d0618922100b` /
  `65226d6ca1071e161644c222b585fb3cfe6df0420c40a7d3983ce7b2378f726e`.

## D0176 — Reject affine-plus-assignment kernel fusion as an end-to-end target

- Date: 2026-08-11
- Status: accepted exact measurement no-go; no product, model, qualification,
  or automatic-selection change.
- Measure first: on clean source commit `f3ffc50c2`, the exact 1,000,002-point
  nonidentity affine transformation followed by `UserData=17` takes 0.35
  seconds in pinned PDAL and 0.60 seconds in the required descriptor-fused CUDA
  hybrid route, or 0.583333x PDAL throughput. Both routes and the profile
  reproduce the same 40,000,701-byte output at SHA-256
  `55db0ba39ce832cbfe0c941660184ffa3b56904a0ca07deaffca230ea13e04ff`.
  One pair is a directional rejection, not qualification.
- Profile and decision: eight tiles launch unpack, transformation, assignment,
  and repack separately, but all 32 kernels total only 0.849310 milliseconds/
  0.141552% of candidate wall. Perfect device-pass fusion is therefore bounded
  far below the required 5--10% process gate while the current CUDA process is
  already slower than PDAL. The dominant cost is outside the kernels; write no
  fusion prototype and treat this endpoint as sufficiently optimized for now.
- Coverage: functionally supported yes; GPU-native unchanged for the exact
  affine binary64 XYZ and pointwise assignment envelopes; prior standalone
  negative qualification unchanged and B0115 adds none; automatically selected
  no.
- Next: B0116 measures and profiles the current exact resident
  `filters.label_duplicates -> filters.nndistance(k=10) -> pointwise UserData`
  graph. Its explicit hypothesis is resident piggybacking without a material
  boundary; no code precedes the exact complete-process baseline/profile.
- Evidence: command/input/binary/output/profile manifest, 5,779 bytes at
  SHA-256
  `47ee11a266e318645a8f1831ec0915c07a663385f0923a80164575a0f1925e15`;
  NCU report, 1,575,822 bytes at SHA-256
  `a49974e5250cba61d07eda1f7f1fc2c304af985d696a7b3f1ea9f70ac28b6c51`.

## D0177 — Qualify the label/NNDistance hybrid chain and test residency next

- Date: 2026-08-11
- Status: accepted explicit exact 1M performance qualification; no code,
  placement model, or automatic-selection change.
- Measure first: on clean commit `6b2a6c162`, five alternating exact pairs of
  `label_duplicates(Classification) -> nndistance(k=10) ->
  UserData=Duplicate` measure 4.374096348 seconds pinned PDAL and 0.645599567
  seconds required hybrid CUDA, or 6.775247x. Every output, stream, and status
  matches the 36,000,375-byte artifact at SHA-256
  `38aa55c56418f6e77c6fce22897c3c3982846639d322d15a8775e08b477ae2c3`.
- Profile and stopping decision: the per-stage hybrid path's 20 kernels total
  28.478630 milliseconds/4.411191% of candidate median. The existing shared
  gather owns 28.150000
  milliseconds; duplicate labeling is only 0.010810 milliseconds. No device
  kernel or extra fusion prototype can meet the 5--10% process gate, so compute
  tuning stops. The label wrapper restores `Duplicate` to the host PointView,
  so this evidence does not prove planner-resident composition or reuse.
- Placement: an explicit resident stats preflight reports
  `missing_calibration_model`, selects no region, and executes pinned host.
  Therefore B0116 changes no option-free behavior. B0117 first uses a
  disposable strict one-shape placement prototype to run the actual resident
  graph at 1M with selected-region and one-index stats. Only a 5--10% win over
  B0116 may advance to a 50K--16M ladder and selector work.
- Coverage: functionally supported yes; GPU-native exact per-stage duplicate,
  NNDistance, and assignment computations with no resident-reuse claim;
  performance-qualified only for the named 1M forced-hybrid graph;
  automatically selected no.
- Evidence: five-pair report, 16,816 bytes at SHA-256
  `75ce4c6e2af7e9cfadc52a3d174c6f0223ba1600daf6534b47ea9de29a72b3bb`;
  command/input/output/profile manifest, 6,782 bytes at SHA-256
  `c9ddc600ed207d17bc2f00ed054a20248a351da87d98d6aefb6edf89652c668f`;
  NCU report, 1,542,060 bytes at SHA-256
  `49524c22e299addc77e7d260076a390bbbbcf3e2f840fa7b758cb1be99ecb681`;
  placement-preflight stats, 3,451 bytes at SHA-256
  `b6cf014d0e1542d5d17c752ad7b4d6952ccc42eebf3ef3ecd7d781e1caef6009`.

## D0178 — Reject label/NNDistance resident placement and restore the engine

- Date: 2026-08-11
- Status: accepted disposable no-go; prototype fully reverted, with no ladder,
  calibration model, selector, or coverage change.
- Prototype proof: a one-line temporary model anchor makes the exact B0116 1M
  graph select `planner_resident_shared_index`. Stats prove one selected region
  containing stages 1--3, accepted preflight, one predicted/observed index,
  zero fallback boundaries, and the exact B0116 output hash.
- Gate: after a 0.72-second correctness warmup, the confirmation takes 0.70
  seconds versus B0116's 0.645599567-second forced-hybrid median. Actual
  residency is 8.426343% slower, not 5--10% faster. It also observes 35/11 MB
  H2D/D2H against the borrowed model's 26/10 MB prediction, so boundary
  accounting correctly reports false.
- Decision: skip the calibration ladder and automatic selector. B0116's fresh
  profile already bounds identical device kernels to 4.411191% of its median;
  B0117's changed cost is placement/boundary control and is measured directly
  in resident stats. Restore the clean engine hash and move to the next
  existing positive-stage composition.
- Coverage: functional support unchanged; GPU-native coverage unchanged;
  performance qualification remains B0116's forced-hybrid 1M chain only;
  automatically selected no.
- Next: B0118 measures and profiles exact
  `filters.mortonorder -> pointwise UserData` on the current clean binary
  before any fusion, placement, or selector work.
- Evidence: reverted-prototype manifest, 5,109 bytes at SHA-256
  `c8905fbc80d0a151b1c570afb9c3fdb47b5197fd523c3c40fe0cd058f3d9bc65`;
  resident stats, 11,393 bytes at SHA-256
  `6e12a1fd5015e9babfe733ad47e5d4eb5cb249c86fdc4a54c93dd8e28ba52518`.

## D0179 — Qualify Morton-plus-assignment and reject more composition work

- Date: 2026-08-11
- Status: accepted explicit exact 2M performance qualification; no product,
  placement, model, or automatic-selection change.
- Measure first: on clean commit `d2aaa57a8`, five alternating exact pairs of
  `filters.mortonorder -> filters.assign(UserData=17)` measure 1.134717691
  seconds pinned PDAL and 0.905711376 seconds required hybrid CUDA, or
  1.252847x. Every output, stream, and status matches the 72,000,375-byte
  artifact at SHA-256
  `0bd3f19edcfd1f89272ed96ee11da3f1f0fb1bfd550bc674fdcd98a78f5c81ca`.
- Profile and stopping decision: all 29 device kernels total only 0.618688
  milliseconds/0.068310% of candidate median; the 16 assignment launches are
  0.068480 milliseconds/0.007561%. A second exact five-pair current-binary
  Morton-only control takes 0.872632778 seconds, so the whole adjacent stage
  adds only 0.033078598 seconds/3.790666%. Both perfect kernel elimination and
  perfect boundary elimination remain below the required 5--10% gate. Do not
  write a fusion or resident prototype; treat the endpoint as sufficiently
  optimized for now.
- Coverage: functionally supported yes; GPU-native exact Morton and assignment
  work in separate forced hybrid stages, with no fused/resident claim;
  performance-qualified only for the named 2M forced-hybrid composition;
  B0006's automatic Morton envelope remains authoritative and B0118 adds no
  automatic composition selector.
- Next: B0119 measures the exact 2M `filters.mortonorder ->
  filters.head(count=100)` complete graph. Its explicit hypothesis is that
  ordinal selection can consume the device permutation before full
  reordered-PointView publication. Profile and prototype only after exact
  current-binary measurement, and retain work only for a reusable 5--10% gain.
- Evidence: five-pair composition report, 16,819 bytes at SHA-256
  `b0901c09988ed3cdff2bc7b24fe13c05cf66c701588efa22c68d255ee44401a7`;
  five-pair control report, 16,820 bytes at SHA-256
  `2621d1300927eb5c696062cc6b389aa3764bdbece0204b15538750947b614b00`;
  command/input/output/profile manifest, 8,087 bytes at SHA-256
  `631a7913ff07b3db0a173d470ab9251fe73a5f105f684dc1c4534d7b4f60d14a`;
  NCU report, 3,433,172 bytes at SHA-256
  `5f1fd172b6bb20f57a7032c54396084d93b65b59ecfcbc53e8b42892f02701b5`.

## D0180 — Qualify Morton-plus-head and reject selective publication fusion

- Date: 2026-08-11
- Status: accepted explicit exact 2M performance qualification; prototype
  fully reverted, with no product, model, or automatic-selection change.
- Measure first: on clean commit `c714f3e0b`, five alternating exact pairs of
  `filters.mortonorder -> filters.head(count=100)` measure 0.806647036 seconds
  pinned PDAL and 0.578119502 seconds required hybrid CUDA, or 1.395295x.
  Every output, stream, and status matches the 3,975-byte artifact at SHA-256
  `2676c5bf761b572cea4215bf990c1193f1f56dc2df8b0edde9f050c612c490f2`.
- Profile: all 61 device kernels total only 0.688512 milliseconds/0.119095% of
  candidate median. Morton owns 0.541856 milliseconds; all 16 head tiles own
  0.146656 milliseconds. Kernel work cannot meet the 5--10% process gate.
- Prototype and decision: a four-line one-shape hook truncates the exact CUDA
  Morton permutation to 100 ids before full PointView publication. Its separate
  five-pair median is 0.553101868 seconds, only 4.327416% below the clean
  candidate with overlapping sample ranges. Reject and fully revert it;
  restore the clean engine SHA-256. No fused ordinal product, model, selector,
  or code remains, and the endpoint is sufficiently optimized for now.
- Coverage: functionally supported yes; GPU-native exact Morton and ordinal
  work in separate forced hybrid stages, with no fused-handoff claim;
  performance-qualified only for the named 2M forced-hybrid composition;
  B0119 adds no automatic selector and B0006 remains authoritative for Morton.
- Next: B0120 physically measures and profiles the already-implemented finite
  basic `filters.stats` CUDA envelope on the six-dimension 2M LAS graph. No
  selector or kernel work precedes exact current-binary evidence and a 5--10%
  complete-process opportunity.
- Evidence: clean five-pair report, 16,718 bytes at SHA-256
  `1e938e134162c050b7d4749c04ca5e2d6d41b37b5cad6adc4eca2cdbed6705e5`;
  command/input/output/profile/prototype manifest, 8,205 bytes at SHA-256
  `eb45362985b4738832bf948df3941f8716ec831535ee02b0d0737f3f056f6f36`;
  NCU report, 5,334,497 bytes at SHA-256
  `8a356f8e0feed759ae666f998b67de9d78ec788d3f820fe6ddaa605a398d1daf`;
  prototype report, 16,766 bytes at SHA-256
  `636fa8ff1520d012b4f6c28e13cbf8648f1de485364780d98d2952d4d38ebad7`.

## D0181 — Keep exact ordered stats on the host after the physical gate

- Date: 2026-08-11
- Status: accepted negative directional gate; no product, selector, model, or
  automatic-selection change.
- Exactness: on clean commit `15ae5728a`, a forced-CUDA 2M six-dimension stats
  process matches pinned PDAL's complete LAS, 3,543-byte metadata JSON,
  stdout, stderr, and status. The profile run reproduces the same artifacts.
- Measurement: one warmup and one exact current-binary pair measures
  0.604473773 seconds pinned PDAL and 1.410544301 seconds required CUDA, or
  0.428539x. This is a bounded negative diagnostic, not a performance claim.
  A bare pinned read/write control takes 0.547127776 seconds, leaving only
  0.057345997 seconds of incremental CPU stats cost.
- Profile and decision: 16 exact `summaryKernel` launches total 498.966048
  milliseconds. The recurrence is serial within each dimension, so the
  six-dimension grid has six blocks/0.01 waves per SM and reaches only 1.94%
  compute and 0.02% DRAM throughput. That unavoidable device recurrence is
  8.700974x the measured CPU stats increment before any gathering or transfer.
  Reject standalone optimization and adjacent stats fusion/residency work;
  preserve the exact force-only implementation and pinned option-free path.
- Coverage: functionally supported yes; the named finite-basic route is now
  physically observed on CUDA, without broader sanitizer/runtime promotion;
  performance-qualified no; automatically selected no.
- Next: B0121 runs a current-clean exact complete-process gate and profile for
  the existing `filters.hag_nn,count=2` shared masked-2D-index lane. Its earlier
  dirty-snapshot result is directional history, not current qualification.
- Evidence: directional report, 9,837 bytes at SHA-256
  `a4926cc5ba68b3a6ce0ec7b51bf32c2063af1ed3b7c293398bc3fa5fca64aa9b`;
  bare control, 9,627 bytes at SHA-256
  `ea57e31a0b71bd52709634e9a81b2c3d17d930f5c02fa4590e038f49ef9da90a`;
  exact metadata differential, 2,964 bytes at SHA-256
  `55d11580e7f90d6c5e70996df655813a13b6dee62fe72a2b903d3001a0d332a1`;
  NCU report, 785,635 bytes at SHA-256
  `b5a05fd6b16b60cc53c8c2e656542a04cd121a0bc8d9b7fdc964bb00e2e4b542`;
  profile manifest, 9,619 bytes at SHA-256
  `e816ac960b3d363a95dd0e9471b22adce181b0bd115b449240239e5b6204eb8d`.

## D0182 — Qualify direct count-two HAG-NN and stop the endpoint

- Date: 2026-08-11
- Status: accepted exact explicit count-two direct-source/direct-publication
  composition and named performance qualification; rejected automatic
  selection and further endpoint tuning.
- Measure first: on clean commit `0c3d3f688`, five exact forced-hybrid pairs
  measure 0.956547913 seconds pinned PDAL and 0.605963500 seconds candidate,
  or 1.578557x. This replaces B0033's dirty-snapshot direction with current
  evidence before product changes.
- Decision: extend the existing strict one-extra-binary64 direct envelope from
  HAG-NN count four to counts two or four. Count one and count three remain
  rejected. Count two must preserve the canonical LAS 1.4/format-7/40-byte
  mapped source, one exact `OffsetTime` descriptor, `extra_dims=all`, point
  order/cardinality, planner-owned masked 2D index, and exact
  `HeightAboveGround` descriptor/bits. Tied or incomplete candidate searches
  still run pinned host repair before atomic publication.
- Proof: required runs fail closed unless they observe the direct shared-index
  executor, an allowed direct schedule class, accepted preflight, exactly one
  planned/observed index, direct source and record summary, no host XYZ mirror,
  direct LAS/Extra Bytes publication, terminal-spill elision, matching boundary
  accounting, and no fallback spill. Count-two native, boundary-tie,
  shell-budget incomplete-repair, source-disabled, unsupported-source, and
  atomic-cleanup cases are byte/stream/status exact. Runner telemetry mutations
  fail closed. All four focused Compute Sanitizer tools are clean.
- Performance: on clean implementation commit `0deba2034`, five exact direct
  pairs measure 0.951975379 seconds pinned PDAL and 0.365209389 seconds
  candidate, or 2.606656x. Direct composition is 39.730794% below the clean
  hybrid median. Every route produces 48,000,909 bytes at SHA-256
  `9c8384faebecfd7f60bb097f2c2d4cf7c5da22f965fcf6197147f0b7ecdf4346`.
- Stop: the fresh direct profile's 20 kernels total 2.552608 milliseconds/
  0.698944% of median; gather is 2.160576 milliseconds/0.591599%.
  Unprofiled canonical publication is 13.127501 milliseconds/3.594514%, and
  every separate host-boundary surface is below the 5--10% gate. The larger
  device/profile/initial-placement interval includes required first CUDA
  context/device setup. No bounded reusable change has credible 5--10%
  headroom; declare the endpoint sufficiently optimized for now.
- Coverage: functionally supported yes; GPU-native only inside the existing
  bounded count-two envelope, with rejected rows repaired on host;
  performance-qualified only for the named forced-hybrid/direct routes;
  automatically selected no because repair is data-dependent and no calibrated
  model exists.
- Next: B0122 remeasures the already-native count-three HAG-NN 1M lane before
  any direct-envelope edit. Prototype the same boundary removal only after a
  current-clean exact forced-hybrid result preserves a 5--10% opportunity.
- Evidence: hybrid report 16,773 bytes at SHA-256
  `8d6b9ab1789f65b3d4a09831b2b9df36f2e6870e27e9b942d61799995aaf461c`;
  direct report 32,951 bytes at SHA-256
  `06a7a9ec49a3ab752fd4d69723a136a8b991fb6e8a95e39a16f64f1f0d9231bc`;
  NCU report 1,541,938 bytes at SHA-256
  `8a9a54affa846af3b75867814e533b6cc94f15ac94f44b3cde9344162e9e0198`;
  manifest 12,217 bytes at SHA-256
  `cb73ebd591a19ebee716ad2efd94f2f9673048f2aee27dcdab234a0aee0b31f3`.

## D0183 — Qualify direct count-three HAG-NN and stop the endpoint

- Date: 2026-08-11
- Status: accepted exact explicit count-three mapped-source/direct-publication
  composition and named performance qualification; rejected automatic
  selection and further tuning.
- Measure first: clean commit `340019d37` measures five exact forced-hybrid
  pairs at 0.961052803 seconds pinned PDAL and 0.620312799 seconds candidate,
  or 1.549304x, before changing the direct envelope.
- Decision and proof: admit HAG-NN counts two through four, and no others, in
  the existing strict direct Extra Bytes envelope. Count three preserves the
  canonical mapped LAS source, exact descriptor/output bits, one planner-owned
  masked 2D index, and atomic point order/cardinality. Direct proof requires
  accepted schedule/preflight, exactly one planned/observed index, record
  summary, no host XYZ, terminal-spill elision, and matching boundaries. Tie
  and incomplete searches still run pinned host repair. Native, repair,
  disabled/unsupported source, count-one/count-five rejection, and no-temp
  process cases pass; all four Compute Sanitizer tools are clean.
- Performance: clean implementation commit `add09d90d` measures five direct
  pairs at 0.970279753 seconds pinned PDAL and 0.365397468 seconds candidate,
  or 2.655409x. Direct wall is 41.094643% below clean hybrid. Every output is
  48,000,909 bytes at SHA-256
  `ce66d73f4b723c0f9ef40744f8f762961123a5bafc896c361b4dd59550f0ee51`.
- Stop: 20 fresh-profile kernels total 3.076512 milliseconds/0.841963% of
  median; gather is 2.676992 milliseconds/0.732625%. Publication is
  12.946378 milliseconds/3.543095%, and all host-boundary subphases together
  are 3.768882%. The remaining device/profile/placement interval includes
  required first CUDA context/device setup. No bounded reusable 5--10% surface
  remains, so the endpoint is sufficiently optimized.
- Coverage: functionally supported yes; GPU-native inside the bounded
  count-three envelope with rejected rows repaired on host;
  performance-qualified only for the named explicit hybrid/direct routes;
  automatically selected no because repair is data-dependent and no calibrated
  model exists.
- Next: B0123 measures the already-native count-one 1M forced-hybrid lane
  before any strict source/publisher extension. This is the remaining
  count-one-through-four direct-publication gap, not a new stage port.
- Evidence: hybrid report 16,832 bytes at SHA-256
  `396282e4984d4b9169312597f8a523cdabc46952bf272b03c35422c92dfc65cb`;
  direct report 32,951 bytes at SHA-256
  `1fb506801a1d3f4b2168941c7b5345a335335c6fe7566bb12e320f0b025cf1c8`;
  NCU report 1,544,036 bytes at SHA-256
  `bb464ff1bcabd3269c08b8221ab889130b259170141987cadaf67e157c7d10ad`;
  manifest 12,264 bytes at SHA-256
  `7d960a318d430a7e370d5da4fb2e0d630c32ad14c23c07f6bd93ca4c0ba05c9a`.

## D0184 — Complete direct count-one HAG-NN and stop the endpoint

- Date: 2026-08-11
- Status: accepted exact explicit count-one mapped-source/direct-publication
  composition and named performance qualification; rejected automatic
  selection and further tuning.
- Measure first: clean commit `1527371a7` measures five exact forced-hybrid
  pairs at 0.909005973 seconds pinned PDAL and 0.629955252 seconds candidate,
  or 1.442969x, before changing the direct envelope.
- Decision and proof: admit HAG-NN counts one through four, and no others, in
  the existing strict direct Extra Bytes envelope. Count one preserves its
  pinned subtraction and ignored-option semantics in the native stage; the
  strict direct producer itself remains option-free. Direct proof requires an
  accepted schedule/preflight, exactly one planned/observed masked 2D index,
  mapped record summary, no host XYZ mirror, direct atomic binary64 output,
  terminal-spill elision, and matching boundaries. Boundary ties and bounded-
  grid incompleteness still run pinned host repair. Native, repair,
  disabled/unsupported source, count-zero/count-five rejection, option-bearing
  count-one rejection, and no-temp cases pass; all four Compute Sanitizer tools
  are clean.
- Performance: clean implementation commit `d69dfb419` measures five direct
  pairs at 0.907840598 seconds pinned PDAL and 0.360501731 seconds candidate,
  or 2.518270x. Direct wall is 42.773438% below clean hybrid. Every output is
  48,000,909 bytes at SHA-256
  `ebe321bb4051d39d8f9e5bea8fbcb63d17f597376453c03cf899b86ecc6f5014`.
- Stop: 20 fresh-profile kernels total 2.067488 milliseconds/0.573503% of
  median; gather is 1.711872 milliseconds/0.474858%. Publication is
  16.414537 milliseconds/4.553248%, and all host-boundary subphases together
  are 3.797665%. The remaining device/profile/placement interval includes
  required first CUDA context/device setup. No bounded reusable 5--10% surface
  remains, so the endpoint is sufficiently optimized.
- Coverage: functionally supported yes; GPU-native inside the bounded
  count-one envelope with rejected rows repaired on host; performance-
  qualified only for the named explicit hybrid/direct routes; automatically
  selected no because repair is data-dependent and no calibrated model exists.
- Next: B0124 runs a current-clean exact 1M five-pair standalone statistical-
  outlier gate before any selector or direct-endpoint edit. The stage is
  already native and composition-friendly, but accepted standalone performance
  remains explicitly unmeasured. The prior 6.45x directional result is the
  hypothesis; it is not current qualification and authorizes no new port.
- Evidence: hybrid report 16,860 bytes at SHA-256
  `bc750b3dd740a9bbe8ae4eb1558e550c5cf1e6ca89c0e9f8ced4dfee936a6d6d`;
  direct report 32,993 bytes at SHA-256
  `fa04d08627eb08a3b850b783eaabb7bff85520f33912ce327f972cfb00ea9a07`;
  NCU report 1,543,860 bytes at SHA-256
  `94948a95fcf0597e93e73c129de139656b91b7deeb8ce548bdef2aadd1a3e8de`;
  manifest 12,581 bytes at SHA-256
  `6c10f0c69abae77a7353fbbb4600c5f86c94ea0031e2d0b56e9281e2d7673410`.

## D0185 — Qualify direct standalone statistical outlier and stop the endpoint

- Date: 2026-08-11
- Status: accepted exact explicit standalone statistical-outlier mapped-source/
  direct-Classification composition and named performance qualification;
  rejected automatic selection and further bounded tuning.
- Measure first: clean commit `b4419528f` measures five exact forced-hybrid
  pairs at 3.999382550 seconds pinned PDAL and 0.613261395 seconds candidate,
  or 6.521497x. A fresh output-bound profile totals 21.501856 milliseconds/
  3.506149% of candidate wall and attributes 21.209696 milliseconds to the
  existing shared gather. That names the source/publication boundary, not a
  new query kernel, as the only credible prototype.
- Decision and proof: extend the existing explicit direct Classification
  publisher only to the exact three-stage statistical graph with `mean_k=8`,
  `multiplier=2`, and `class=7`. Require accepted whole-view-neighborhood
  preflight, exactly one planned/observed index, mapped source/record summary,
  no host XYZ mirror, the exact nine-byte-per-point mean-distance/status
  download for the host-order finale, one Classification boundary spill of one
  byte per point, matching boundaries, atomic canonical LAS publication, and
  zero fallback. Changed
  options decline before output; required disabled source fails closed. A
  forced-grid incomplete search retains exact pinned host repair.
- Performance: clean implementation commit `796e8af58` measures five direct
  pairs at 4.016261748 seconds pinned PDAL and 0.373965127 seconds candidate,
  or 10.739669x. Direct wall is 39.020273% below the clean hybrid median. Every
  route produces 36,000,375 bytes at SHA-256
  `c43ef2dd91c6e2cf147af50163422d8c1cfccc853538effb24f000e7e3844d48`.
- Stop: 19 fresh-profile kernels total 21.450304 milliseconds/5.735910% of
  median; the required gather is 21.093440 milliseconds/5.640483%, while all
  other kernels total 0.095427%. Publication, hydration, row materialization,
  and the residual wrapper ceiling after the gather are 3.123725%, 2.526728%,
  2.638052%, and 3.522585%. B0053 already rejected bounded gather launch
  tuning; a different algorithm would be required. The endpoint is
  sufficiently optimized.
- Validation: the focused physical formats 3/6/7/8 matrix, positive mutation,
  incomplete host repair, strict rejection, and atomic cleanup pass. Host
  ASan/UBSan and all four Compute Sanitizer tools are clean.
- Coverage: functionally supported yes; GPU-native inside the bounded
  statistical envelope with exact host repair for incomplete rows;
  performance-qualified only for the named explicit 1M forced-hybrid/direct
  routes; automatically selected no because repair remains data-dependent and
  there is no standalone placement model.
- Next: B0125 measures a current-clean exact standalone radius-outlier gate and
  fresh profile before any reuse of this strict mapped source/Classification
  publisher. The older dirty 250K 2.072x row is only the hypothesis. No new
  stage port or selector is authorized.
- Evidence: hybrid report 16,868 bytes at SHA-256
  `f827f2a2b879ae09d101784711e367fe867b2906bcfebd0f1f7f019546094b77`;
  direct report 33,036 bytes at SHA-256
  `12aa3bb0dcfd78233bcc4e8e13b7f7110d6657d42b1f7af58907372403bea786`;
  hybrid NCU report 1,331,161 bytes at SHA-256
  `251543e1c95768ce310b3edd4c92334bbe5dc746242c11de5e9b300cfefe1fc8`;
  direct NCU report 1,442,362 bytes at SHA-256
  `66dff00477f189db9884a223a6758a82c6d0dd9a1950e8f5cc0e335261d3b8c6`;
  manifest 14,799 bytes at SHA-256
  `c836d6040007673c3c09985985ebe067fb539fad4b444d7bf7b93797698d1337`.

## D0186 — Qualify direct standalone radius outlier and stop the endpoint

- Date: 2026-08-11
- Status: accepted exact explicit standalone radius-outlier mapped-source/
  direct-Classification composition and named performance qualification;
  rejected automatic selection and further bounded endpoint tuning.
- Measure first: clean commit `f1a6decd0` measures five exact forced-hybrid
  pairs at 3.091121142 seconds pinned PDAL and 0.624957502 seconds candidate,
  or 4.946130x. A fresh output-bound profile totals 2.910440 milliseconds/
  0.465704% of candidate wall and attributes 2.640000 milliseconds to the
  existing exact radius-count kernel. A directional pinned-PDAL hardware-cycle
  capture names nanoflann traversal as its dominant CPU cost. That evidence
  chooses source/publication reuse, not a new radius kernel.
- Decision and proof: extend the existing explicit direct Classification
  publisher only to the exact three-stage radius graph with `radius=1`,
  `min_k=2`, and `class=7`. Require accepted whole-view-neighborhood preflight,
  exactly one planned/observed 3D radius index, mapped source/record summary,
  no host XYZ mirror, the exact four-byte-per-point count download for pinned
  `count > min_k` and all-outlier handling, one one-byte-per-point
  Classification boundary spill, matching boundary accounting, atomic
  canonical LAS publication, and zero fallback. Changed options and
  unsupported source/output shapes decline or fail closed before output.
- Performance: clean implementation commit `1ec2f724e` measures five direct
  pairs at 3.101282835 seconds pinned PDAL and 0.382528055 seconds candidate,
  or 8.107334x. Direct wall is 38.791349% below the clean hybrid median. Every
  route produces 36,000,375 bytes at SHA-256
  `bb932237b43270ea8ce2e2ee1152eb6bb4a49be8b24014879933d1529256d384`.
- Stop: 18 fresh-profile kernels total 2.972310 milliseconds/0.777018% of
  median; the radius query is 2.640000 milliseconds/0.690145%, while all other
  kernels total 0.086872%. Publication, hydration, row materialization, and
  the residual wrapper ceiling after the radius query are 3.812374%,
  2.556435%, 2.514113%, and 4.383453%. No bounded reusable surface reaches the
  5--10% gate, so the endpoint is sufficiently optimized.
- Validation: the focused physical formats 3/6/7/8 matrix, positive mutation,
  strict-radius boundary, strict rejection, and atomic cleanup pass. Planner/
  rewrite and resident process gates pass. Host ASan/UBSan and all four
  Compute Sanitizer tools are clean.
- Coverage: functionally supported yes; GPU-native for the bounded
  shared-index radius count, while the exact comparison and Classification
  finale are host; performance-qualified only for the named explicit 1M
  forced-hybrid/direct routes; automatically selected no because there is no
  standalone placement model and the measured envelope is deliberately exact
  and narrow.
- Next: B0126 measures and profiles the exact 1M same-radius composition
  `outlier(radius=1.01,min_k=2,class=7) -> radialdensity(radius=1.01) ->
  assign(UserData=1 where RadialDensity>=0.2)`. The explicit hypothesis is one
  planner-owned 3D radius index and one resident source/publication lifetime
  across two already-native consumers. Prototype a dual-column direct route
  only for reusable 5--10% complete-process value; no selector or new stage
  port is authorized.
- Evidence: hybrid report 16,844 bytes at SHA-256
  `1e6875148897d0daedd4a6d086410345a5bfaeed4babee789b05e1aaab7e4f29`;
  direct report 32,999 bytes at SHA-256
  `1a74ee36f3d1bf9eb861bdb9c8c048b8b7eeb1271cb53140e612c6b3ea58fa6b`;
  hybrid NCU report 1,285,932 bytes at SHA-256
  `bb121a3da5b80c82a2b69e89682bf1fd443ca2bf2ea57dfe57ef651c516e85d6`;
  direct NCU report 1,398,545 bytes at SHA-256
  `11087fce05a381c76e6ced9e29d843612b8fb807cd29735a25deb7f33068282c`;
  manifest 16,354 bytes at SHA-256
  `630337e2d18a5bcfd08fef8494af7fdc4449627047ed0ba456b3dafcb16fc96b`.

## D0187 — Qualify one-index same-radius composition and stop endpoint tuning

- Date: 2026-08-11
- Status: accepted exact explicit radius-outlier/radial-density mapped-source,
  one-index, direct-publication composition and named performance
  qualification; rejected automatic selection and further bounded endpoint
  tuning.
- Measure first: the clean preimplementation route at `b01e35209` is exact at
  5.968314287 seconds pinned PDAL and 0.660095887 seconds forced hybrid. Its
  40-launch profile shows two complete index-build chains and only 6.026080
  milliseconds of total GPU work, so the selected direction is shared
  source/index/publication lifetime rather than a kernel rewrite.
- Decision and proof: commit `34aeb65d1` admits only the exact five-stage
  `radius outlier(1.01,2,7) -> radialdensity(1.01) -> UserData assign` graph
  through an explicit require gate. It uses one whole-view region and one
  planner-owned 3D radius index, mapped record summary, no host XYZ mirror,
  exact host-order outlier comparison/finale, device-resident RadialDensity,
  direct Classification/UserData overlays, matching logical boundary
  accounting with separately observed physical transfers, atomic LAS
  publication, and zero fallback. Automatic admission remains excluded.
- Performance: five current-clean pairs measure 5.997575882 seconds pinned
  PDAL and 0.654990023 seconds hybrid, or 9.156744x. Five direct pairs measure
  5.926227766 seconds pinned PDAL and 0.372796165 seconds candidate, or
  15.896697x. Direct wall is 43.083688% below hybrid. Every route produces
  36,000,375 bytes at SHA-256
  `9b267e086afb28a09f9f7fabf869d16349ad4e314c4a7a96180f0e7788417878`.
- Stop: the current direct profile has 20 launches totaling 5.661312
  milliseconds/1.518608% of median. Its two required radius queries account
  for 5.304512 milliseconds/1.422899%; all other kernels are 0.095709%.
  Required canonical publication has only a 5.540217% full-elimination ceiling,
  combined host-boundary subphases are 5.197451%, and every reusable
  constituent is below 5%. The device/profile/initial-placement interval
  includes required first CUDA context and memory-budget setup. No clear
  bounded optimization can credibly deliver another 5--10%; the endpoint is
  sufficiently optimized.
- Validation: physical exact process, strict rejection/atomic cleanup, 92
  planner/rewrite tests, resident/runner gates, host ASan/UBSan, and all four
  Compute Sanitizer tools pass. Placement declares the intermediate four-byte
  radius result once, separately from the ten-byte logical terminal boundary;
  execution observes 4 MB count plus 1 MB final UserData D2H and no
  RadialDensity download.
- Coverage: functionally supported yes; GPU-native for the bounded queries and
  assignment inside one shared-index region, while the outlier finale and LAS
  publication remain host; performance-qualified only for the named explicit
  1M hybrid/direct routes; automatically selected no because no cardinality
  ladder or placement model exists.
- Next: B0127 measures exact current-binary 50K/250K/1M/4M hybrid/direct rows
  for this already-positive composition. A stable break-even ladder may
  authorize a conservative automatic model; no new stage port or kernel change
  precedes that evidence.
- Evidence: hybrid report 16,953 bytes at SHA-256
  `cee349c5a36c5bd25cb239cf1e4e16a641490091f667d02ae3c2e5898fdc4735`;
  direct report 33,101 bytes at SHA-256
  `9cad1dcc94b91880eaf88a15ae69d307d312a19f2a12c67e6ce072f01263071c`;
  direct NCU report 1,529,604 bytes at SHA-256
  `373c619cb2edbda601bcc7d9dedba639d2eba45e2c30cbd739de9285bcc97f3c`;
  manifest 16,624 bytes at SHA-256
  `7a5f2789e8dcb8bfc9141c57ce0b5edd64eef679cdaf8d2e7ad7d12a2a5dccdb`.

## D0188 — Select the exact same-radius composition and stop endpoint tuning

- Date: 2026-08-11
- Status: accepted bounded automatic placement for the exact measured
  radius-outlier/radial-density composition; rejected the 50K direction row,
  broader admission, another radius kernel, and further endpoint tuning.
- Measure first: clean hybrid/direct rows at 50K/250K/1M/4M preserve the pinned
  output. Direct wall is below hybrid at every point, but the 50K improvement
  is only 5.644% from one noisy direction row. The fit therefore uses only the
  250K/1M/4M direct rows and admits no extrapolation.
- Decision: implementation commit
  `a7c63f571f00d761851f4f261b75edc695d662c1` adds the distinct
  `radius-outlier-radialdensity-direct-compose` model with device fixed cost
  266,587,419.5420545 ns, host slope 6,228.969755310317 ns/point, zero other
  coefficients, and an exact 250,000--4,000,000-point bound. The strict matcher
  requires the exact five-stage graph/options, LAS 1.4 format 7/36-byte source
  and output, calibrated SM89 identity, one region/index, 24-byte upload,
  four-byte radius result, ten-byte logical spill, and intrinsic single-lane
  schedule. Frozen placement audit accuracy remains 152/152 over 35 models.
- Proof: automatic publication requires the direct mapped source/output,
  record-summary-backed index, no host XYZ mirror, one observed index,
  successful resident assignment, matching boundary accounting, and zero
  fallback before the atomic output rename. Below-threshold, shape-drift,
  disabled-source, placement, and injected-proof cases delegate exactly or
  fail closed before an artifact is created.
- Performance: clean option-free one-warmup/five-pair gates measure
  1.516960335/0.321529632 seconds at 250K, 5.937959360/0.374509952 seconds at
  1M, and 25.288767870/0.559754441 seconds at 4M for pinned PDAL/PDG. These
  are 4.717949x, 15.855278x, and 45.178325x. Outputs are exact at
  9,000,375/36,000,375/144,000,375 bytes.
- Stop: the fresh automatic 1M NCU capture reproduces the accepted artifact
  and records 20 launches totaling 5.654080 milliseconds. Two required
  radius-count launches consume 5.299264 milliseconds/93.724602% of device
  time. This is the already-stopped B0126 kernel shape; selection introduces
  no new reusable non-kernel surface with a credible 5--10% complete-process
  ceiling. The endpoint is sufficiently optimized for now.
- Validation: Release and Host Debug builds pass; focused placement is 29/29
  in both trees, planner/rewrite is 92/92, runner contract is 12/12, the exact
  physical automatic process passes, Host ASan/UBSan is 121/121, and
  memcheck/initcheck/racecheck/synccheck are clean.
- Coverage: functionally supported yes; GPU-native for bounded radius counts
  and assignment in one planner-owned 3D-index region, while the exact outlier
  comparison/finale and canonical LAS publication remain host work;
  performance-qualified only for the exact measured format-7/36-byte
  250K--4M composition on the calibrated SM89 profile; automatically selected
  yes only inside that envelope.
- Next: B0128 remeasures the already-implemented comparator-unique 1M
  `filters.skewnessbalancing` pipeline from a clean current binary and profiles
  it only if positive. B0025's dirty 1.150x result is the explicit hypothesis
  that its exact host recurrence/permutation/publication boundary may make an
  adjacent point-program resident composition valuable; no prototype or
  selection change precedes current evidence.
- Evidence: exact automatic reports are 16,997 bytes/SHA-256
  `3f2f78b22e1d44775f4b03d09608f4f19d2a9311ead89d85f2029435c54e13e0`,
  17,010 bytes/`e064131ce8d3d7ae4a78456df69af71bafffd4040e494a6b0b1f51c4c33e83fd`,
  and 17,064 bytes/`32ef10cabb3a3f5382d49ce4b680cd96085650824a488f388f966c6b4e7f1d3b`
  at 250K/1M/4M. The NCU report is 1,531,172 bytes/SHA-256
  `2e8cc9298fbfff408e4cea22f575e388c67421faab2eccf16b38fe58618a7547`;
  the bound manifest is 13,928 bytes/SHA-256
  `60266e8fa1b31da147b2b2ad89f9738191886fe81e83ea168ed72a4ad77f90b1`.

## D0189 — Pursue a direct skewness boundary after clean profile evidence

- Date: 2026-08-11
- Status: accepted current-clean forced-hybrid performance qualification and
  one explicit direct-boundary prototype; rejected a new skewness kernel,
  automatic selection, and any broad resident claim.
- Measure first: the exact comparator-unique 1M format-7 graph measures
  1.310245631 seconds pinned PDAL and 1.098422804 seconds forced CUDA, or
  1.192843x. B0025's dirty 1.150x row is now historical rather than the
  qualification basis.
- Profile: the fresh exact-output NCU capture has 13 launches totaling about
  0.258730 milliseconds from rounded summaries, only 0.023555% of candidate
  wall. A directional Release-symbol phase profile measures the filter call at
  245.294974 milliseconds, `PointView::applyPermutation` at 6.222473
  milliseconds, and the unchanged exact recurrence at 12.672796 milliseconds.
  The remaining 226.399705 milliseconds contains repeated PointView
  extraction, staging/setup, and Classification publication and is the named
  limiter. A separate exact bare read/write control is 0.300782159 seconds;
  its difference from the stage-bearing graph is a direction/ceiling, not an
  additive decomposition.
- Decision: B0129 may implement one strict, explicit-only composition that
  reuses the mapped LAS source, existing exact unique-Z CUDA ordering, and
  unchanged sequential host recurrence, plus a permutation-aware canonical
  LAS publisher designed for reuse by other exact global-order filters. The
  differential/fail-closed contract precedes code. Retain only after a clean
  complete-process result at least 5% below B0128's forced-hybrid median.
  Require a separate 10% gate and cardinality ladder before fitting or
  automatically selecting anything.
- Coverage: functionally supported yes; only the bounded order substep is
  GPU-native; B0128 performance-qualifies the named forced 1M fixture;
  automatic selection remains no. No new stage port begins before B0129's
  retention decision.
- Evidence: paired report 16,965 bytes/SHA-256
  `4f19a050a78fa5badbb2f9b81825b77834d2ffb38a12bfbe52baf8aaa64bfb76`;
  NCU report 2,654,035 bytes/
  `41433148d7c5ae94326e1a9a007e579ddf72453f2a290c525b0eb1bb7dc5b7c3`;
  phase log 3,306 bytes/
  `e494669c4722e835d022baee37706602eb7ab373248609571e67c91e0b925293`;
  manifest 12,287 bytes/
  `11f00b2cdce2f05f53322d8a47c9a13948a85823915518730882668b35f02422`.

## D0190 — Retain direct skewness composition, stop the endpoint, and measure sort next

- Date: 2026-08-11.
- Status: accepted the strict explicit B0129 mapped-source/global-order
  composition and its named-fixture performance qualification; rejected a new
  skewness kernel, placement shortcut, calibration model, automatic selector,
  and further endpoint tuning.
- Retention gate: clean implementation commit
  `87f14b22f86ac3677e11f5e83e18b79093eb61a5` measures 1.229485169 seconds
  pinned PDAL and 0.425794757 seconds direct on the exact comparator-unique 1M
  format-7 graph, or 2.887507x. Direct is 61.235805% below B0128's clean
  1.098422804-second forced-hybrid median and preserves the exact
  36,000,375-byte artifact at SHA-256
  `bdfaabcd16921cf95c66a8000611926a3bf587a810185112526eb25c1f61df80`.
- Exact envelope and proof: the route requires canonical LAS 1.4 format 7 with
  36-byte records, finite comparator-unique physical binary64 Z, the exact
  single skewness stage/options, and explicit activation. It proves one
  whole-view global-order region, zero indexes, mapped-source use, no record
  summary or host XYZ mirror, an 8 MB Z upload and 8 MB permutation download,
  terminal-spill elision, matching boundaries, and atomic canonical output.
  All unsupported or ambiguous cases delegate unchanged or fail without an
  artifact when required.
- Fresh profile and stopping decision: the exact output-bound NCU capture has
  13 launches totaling 0.260192 milliseconds/0.061107% of candidate median.
  Ordinary stats record 0.032016684 seconds of canonical publication; it
  would need a 66.495762% reduction merely to save 5% complete wall. The
  apparent 0.166260729-second device/profile plus initial-placement interval
  contains required first CUDA context/free-memory work, and B0108 already
  rejected the reusable shortcut at a 3.636364% process gain. No independent
  reusable 5--10% surface remains, so the endpoint is sufficiently optimized.
- Coverage: functionally supported yes; GPU-native only for the bounded exact
  unique-Z ordering, with the sequential recurrence and Classification update
  on host; performance-qualified only for this named explicit direct 1M
  fixture; automatically selected no. No broader global-order or input claim
  follows.
- Next: B0130 measures the already-implemented exact `filters.sort` route from
  a clean current binary on a deterministic nonidentity full-record ordering
  fixture. A strict mapped-source/permutation-publisher composition is the
  measured reuse hypothesis. Profile only if positive and retain no prototype
  absent a 5--10% same-binary complete-process win; do not add another sort
  kernel or selector first.
- Evidence: paired report 33,580 bytes/SHA-256
  `bf28a84b0cd44550a172931789992d6f6f3bd888a33912f0dc9c265f3ef1bd05`;
  NCU report 2,654,323 bytes/
  `9737adaaba0b451a008f19962355fc9d433feb84d8473059570405c6c79775a0`;
  profiled/unprofiled stats 9,607/9,606 bytes at
  `5f0a10ffa15376c88a7dd70fe92cd5caa19152283c71a5acb4d4aed20f37032b` /
  `94387ab0843539b09a04052d4b892dca5383388be01b93a3e85a61d8f64fa5bf`;
  complete manifest 10,452 bytes/
  `ad07a5a73ff3c4116fe100ef1fb7ade24656afe4812e8372798e8e3b840e3607`.

## D0191 — Measure sort, keep host selection, and prototype boundary reuse

- Date: 2026-08-11.
- Status: accepted a current-clean named-fixture forced-CUDA qualification and
  one strict direct-boundary prototype; retained host selection and rejected a
  new sort kernel, calibration model, automatic selector, and broader sort
  claim.
- Measure first: on the exact nonidentity comparator-unique 1M Z graph, five
  clean pairs measure 1.562019960 seconds pinned PDAL and 1.117247334 seconds
  forced CUDA, or 1.398097x. A separate exact same-binary host wrapper median
  is 0.903809508 seconds, 1.236154x/23.615355% faster than CUDA. Therefore a
  direct prototype must beat the host route, not merely upstream PDAL.
- Profile: the fresh exact-output NCU capture has 13 launches totaling
  0.271040 milliseconds/0.024260% of CUDA wall. A directional Release-symbol
  profile measures the filter at 0.204735422 seconds: 0.112223505 in the CUDA
  wrapper, 0.007759453 applying the full-view permutation, and 0.084752464 in
  remaining key gather/setup. The named limiter is source/publication and CUDA
  setup around an already-tiny ordering sequence.
- Decision: B0131 may implement only canonical LAS 1.4 format-7/36-byte
  `sort(Z,ASC,NORMAL)` on finite comparator-unique physical binary64 Z through
  an explicit mapped-source/permutation-publisher composition. Reuse B0129's
  source identity, exact CUDA ordering, and canonical publisher contract.
  Retain only below 0.858619033 seconds, 5% under the exact 0.903809508-second
  host median. Differential and fail-closed proof precede code; no model or
  automatic selection follows.
- Coverage: functionally supported yes; GPU-native inside the existing exact
  tie-free ordering envelope; performance-qualified only for the named forced
  1M CUDA fixture at 1.398097x pinned PDAL, even though host remains faster;
  automatically selected no.
- Evidence: CUDA/host reports 17,009/17,029 bytes at SHA-256
  `11b9026576615b6983a79b740fc563c792c05cd1d421647bad29196074e9ecd0` /
  `7572f0aef0c4504fa6051472085ba128cb626dc38d846aa60ed47dff47013e90`;
  NCU report 2,653,104 bytes/
  `e363ab01da75db966d7651bed6a6ab1b9c9615bf7b051cf5f8109da94ee9472c`;
  phase log 3,377 bytes/
  `98d2406579249952269bfd815b4034e7db171dace9018c608597756387430dec`;
  complete manifest 9,582 bytes/
  `6204a3cabe53ad9565abf509fa20d694f8f2c1bf8ce9dceb3260730dd687406c`.

## D0192 — Retain direct sort composition and stop the endpoint

- Date: 2026-08-11.
- Status: accepted the strict explicit direct composition and named-fixture
  performance qualification; retained host/default selection and rejected a
  new sort kernel, calibration model, automatic selector, and broader sort
  envelope.
- Exact route: clean implementation commit
  `da2e7323fccaf1967647b7485b72c1674d650b97` admits only canonical LAS 1.4
  format-7/36-byte `sort(Z,ASC,NORMAL)` with finite comparator-unique physical
  binary64 Z. It reuses the mapped source, existing exact CUDA ordering, and a
  generic exact permutation publisher. Every option/data/layout drift retains
  unchanged host/upstream behavior; required proof failures publish nothing.
- Complete-process gate: five exact alternating pairs measure 1.054076074
  seconds pinned PDAL and 0.319400289 seconds direct, or 3.300173x. The current
  binary's separate exact host control is 0.699280617 seconds; direct is
  2.189355x/54.324447% faster and clears the 0.664316586-second retention
  threshold. Every route emits the same 36,000,375-byte artifact at SHA-256
  `a757dc40991bf624fdcfb53d8accf1bba26a33f3da9dd795ea2422ac44495c03`.
- Profile and stopping decision: the fresh exact-output NCU capture records 13
  launches totaling 0.256640 milliseconds, 0.080351% of candidate median. An
  unprofiled stats pass measures 0.025040367 seconds of canonical publication
  and 0.036952070 seconds for the entire rewritten manager. Publication would
  need a 63.777078% reduction, or the entire manager a 43.218186% reduction,
  merely to save 5% process wall. The B0130-isolated full-view permutation is
  only 2.429382% of this endpoint, and B0108 already rejected the apparent
  reusable placement shortcut below 5%. No clear independent reusable 5--10%
  surface remains, so stop this endpoint.
- Coverage: functionally supported yes; GPU-native only for the bounded exact
  ordering/tie proof while source identity and canonical record publication
  remain host work; performance-qualified only for the named explicit direct
  1M fixture on the recorded SM89 machine; automatically selected no.
- Next: B0132 performs a clean current-binary five-pair exact acceptance gate
  for the already-implemented explicit HAG Delaunay count-three mapped-source/
  direct-output composition. Its B0113 one-pair/profile record is positive but
  unqualified. Measure before any code, model, selector, or new stage port.
- Evidence: direct/host reports 34,230/17,087 bytes at SHA-256
  `b98a654d17f1e8a829e7e41decb598c4650abab8e53650b4208413ed2a8a32eb` /
  `991beb505ce91eacd35c2c810f554674c01d54ff56f88bbf726dbf94f9189012`;
  NCU report 2,654,677 bytes/
  `7cc55f00e0df0ee4030cfc478bb7b920b5c93ea4f46397c81955fcc4a92dc9b4`;
  profiled/unprofiled stats 9,897/9,896 bytes at
  `e7bd2370a74627c689c9f0cfd62471922042dac323c9d484b1eee7b41201a691` /
  `ffac635e29950d389cf32f418ec5f37d7809a5783560cbc0bd74254284d1da74`;
  complete manifest 13,094 bytes/
  `192914ded6ea466c0578d3e9943e007405d6b778920bf0158f19dc0359d97090`.

## D0193 — Qualify explicit HAG Delaunay count three and retain its stop

- Date: 2026-08-11.
- Status: accepted a current-clean named-fixture performance qualification for
  the existing strict explicit direct route; changed no product code and
  retained no model or automatic selector.
- Measure first: five exact alternating pairs from clean commit
  `7d046059401db1c93171cecdd938bffcddbd65fe` measure 0.805184970 seconds
  pinned PDAL and 0.365213991 seconds direct, or 2.204694x. Every process emits
  the same 48,000,909-byte artifact at SHA-256
  `4f4847ef2f48a18158a5ea1c5921c640f58071d3880292a7f02e4a8e9c8290e3`.
- Proof: the candidate requires `planner_resident_shared_index_direct_las`,
  mapped-source use, mapped-record index summary, no complete host XYZ mirror,
  one predicted/observed planner-owned index, direct one-binary64 output,
  matching boundary accounting, no fallback spill, and terminal-spill elision.
- Profile and stop: the fresh current-binary exact-output NCU capture records
  20 launches totaling 2.421952 milliseconds, only 0.663160% of candidate
  median. Canonical publication is 0.013944619 seconds/3.818205% of wall and
  cannot buy 5% even if eliminated. The 0.083483706-second full manager owns
  the exact shared-index query, transfers, and result path; no independently
  removable 5--10% component is identified. B0108 already rejects the
  apparent startup shortcut below 5%. B0113's endpoint-stopping decision is
  confirmed.
- Coverage: functionally supported yes; GPU-native only for the bounded
  count-three masked-index query and Delaunay projection while repair and
  canonical publication remain host work; performance-qualified only for the
  named explicit 1,000,002-point direct fixture on recorded SM89; automatically
  selected no.
- Next: B0133 measures and profiles pinned PDAL `hag_nn(count=5)` before
  implementation, testing reuse of the existing mapped source, planner-owned
  2D index, masked query, and one-binary64 publisher. No code, kernel, model,
  selector, or calibration ladder precedes that baseline/profile.
- Evidence: paired report 34,238 bytes/SHA-256
  `452bb56285a0945e1b1a11a168eb6cf6775813ef6f0c9e469dc7bdc2f070a201`;
  NCU report 1,544,621 bytes/
  `3dbe861166e3f118246568624f7bc5c228d4d532531abb34ca6155ece7a4fa6d`;
  profiled/unprofiled stats 10,290/10,293 bytes at
  `82e099fe2bdb7b696b432c9eb01b5b6df48975de9a87cd2652094ff35ef94a4a` /
  `6a10f1b63f43428f5cfa5542df008ca82fa676900972025b45b711ff2610ebeb`;
  complete manifest 11,606 bytes/
  `57062e777f29071274ad55421190e8a0f5bef9a0e41caa89db713342d098d8aa`.

## D0194 — Measure HAG-NN count five and select shared-index reuse

- Date: 2026-08-11.
- Status: accepted a cheap explicit prototype direction; changed no product
  code, coverage, model, or automatic selection.
- Measure first: from clean commit
  `f80fb5056a3f77edaa1784be70b1915bf047d032`, five alternating exact pairs
  measure pinned PDAL `hag_nn(count=5)` at 1.011225052 seconds. The current
  `pdg` default correctly delegates at 1.028807390 seconds. Both produce the
  same 48,000,909-byte artifact at SHA-256
  `7050b7b8fb73d7303a7986506fcd94286786a5caf4ad9a7977db3b6f39e18f73`.
- Control/profile: the pinned output-shaped read/write control is 0.289291154
  seconds, leaving 0.721933898 seconds/71.392011% of stock wall attributable
  to the stage. The frozen-UTC optimized-oracle phase capture reproduces the
  paired output and measures the filter at 0.756377590 seconds: 2D-index build
  is 0.063686914 seconds/8.419990%, while the post-index query/projection path
  is 0.680035688 seconds/89.906906%.
- Path: reuse the existing mapped source, planner-owned masked 2D index,
  exact tie/incomplete repair semantics, and generic one-binary64 publisher.
  A private index, new kernel family, model, selector, and calibration ladder
  are rejected at this decision point.
- Coverage: count five remains functionally supported through unchanged
  pinned-PDAL fallback; GPU-native no; performance-qualified no; automatically
  selected no. The evidence clears only a cheap-prototype gate.
- Next: B0134 writes the count-five differential and proof matrix first, then
  extends only the strict explicit direct envelope. Retain it only for exact
  current-binary complete-process improvement of at least 5--10%.
- Evidence: count-five/control reports 17,083/17,086 bytes at SHA-256
  `bfc94d55bfc55cf0cee42fdd92591d59d40b6c2d2b9ebd35a10e945deaa542f6` /
  `7befba84c7df20186a924413971b9ba05bb60e15b95765a5405d60d6df98ac0d`;
  phase script/log 2,417/1,706 bytes at SHA-256
  `e9341639a2025dbf91076c51a40e3f6fbd2d04d4339a754b2b1334b8c45ce41f` /
  `55f850bd113a7074810685ffb72b1814cfac5fd3c4ab3bccd38780141ca714eb`;
  complete manifest 8,246 bytes/
  `151351f2ac0abd1618ab6d166ed5013cb54942e9deaf2c82549d75334baf1a66`.

## D0195 — Extend only the explicit exact HAG-NN envelope to count five

- Date: 2026-08-11.
- Status: accepted the tests-first implementation; retained no model or
  automatic selector and made no performance qualification.
- Scope: the existing generic ordered HAG-NN projection, planner-owned masked
  2D index, exact host-repair branches, mapped LAS source, and one-binary64
  publisher now admit count five. No private index or new kernel family is
  introduced. Count six and option/layout/source drift remain host-owned.
- Exactness: the expanded 92-case host and physical-CUDA matrices cover unique
  fifth-neighbor interpolation on Grid and Morton BVH, strict/equal/negative
  distance cutoffs, bounds/extrapolation, large/underflow/signed-zero arithmetic,
  insufficient/nonfinite repair, fifth/sixth-boundary ties, overflow, and
  incomplete search. The direct process proves mapped source/summary, no host
  XYZ mirror, one predicted/observed index, zero spills, terminal-spill elision,
  atomic output, tie/incomplete-query repair, and source/layout rejection. The
  count-five output remains SHA-256
  `7050b7b8fb73d7303a7986506fcd94286786a5caf4ad9a7977db3b6f39e18f73`.
- Validation: 27 focused Release planner/rewrite/lifecycle tests pass; the
  complete Release unit binary passes 551 tests with only two optional local-
  corpus skips. Focused host ASan/UBSan passes with leak detection disabled
  because LeakSanitizer is unavailable under ptrace; memcheck/initcheck/
  synccheck report zero errors and racecheck reports zero hazards on the
  bounded Grid/BVH count-five path.
- Cheap retention direction: one exact dirty-snapshot pair measures
  1.042808916 seconds pinned PDAL and 0.359718221 seconds direct, or 2.898961x,
  while proving the strict direct executor and one-index/zero-spill facts. This
  clears the 5--10% prototype gate but is not retained performance evidence.
- Coverage: functionally supported yes; GPU-native yes only inside the bounded
  explicit count-five query/projection envelope with exact host repair;
  performance-qualified no pending B0135; automatically selected no.
- Next: B0135 commits this implementation cleanly, runs five current-binary
  exact pairs and one fresh exact-output profile, and then applies the endpoint
  stopping rule. No calibration ladder, model, or selector follows.

## D0196 — Qualify and stop the explicit HAG-NN count-five endpoint

- Date: 2026-08-11.
- Status: accepted a current-clean named-fixture performance qualification;
  retained the explicit endpoint without a model or automatic selector.
- Measure: from clean implementation commit
  `7ca04e89992d1d2aceb843c88947455eb96e0c0d`, five exact alternating pairs
  measure 1.007641329 seconds pinned PDAL and 0.343845512 seconds direct, or
  2.930506x. Every route emits the same 48,000,909-byte artifact at SHA-256
  `7050b7b8fb73d7303a7986506fcd94286786a5caf4ad9a7977db3b6f39e18f73`.
- Proof: every candidate observes the strict direct shared-index executor,
  mapped source and record summary, no host XYZ mirror, one predicted/observed
  planner-owned index, zero fallback/spill, direct binary64 output, matching
  boundaries, and terminal-spill elision.
- Profile/stop: the exact-output NCU capture records 20 launches totaling
  4.391810 milliseconds/1.277263% of candidate wall; the shared kNN gather is
  3.960000 milliseconds/1.151680% and the HAG projection 0.090500
  milliseconds/0.026320%. Canonical publication is a 4.352942% full-elimination
  ceiling. The full exact manager is 23.751502%, but needs a 21.051301%
  reduction merely to save 5% process wall and exposes no independent reusable
  component at that scale. B0108 already rejects the apparent placement
  shortcut at 3.636364%. Stop this endpoint.
- Coverage: functionally supported yes; GPU-native only inside the bounded
  count-five query/projection envelope; performance-qualified only for the
  named explicit 1M SM89 fixture; automatically selected no.
- Next: B0136 measures pinned PDAL `hag_nn(count=6)` and an output-shaped
  control on a deterministic sixth-candidate fixture before implementation.
  This tests reuse of the same planner-owned 2D index and direct publisher;
  no private index, new kernel family, model, selector, or calibration ladder
  precedes the evidence.
- Evidence: paired report 34,232 bytes/SHA-256
  `bbb7eed4772022bed8aeddb49d2d1b96fe88630be79623c09ff3e64ee5b65639`;
  NCU report 1,543,885 bytes/
  `c664d0131e78d75253ee7cd19d8098882a86b55d118ab9061b0badf47c78078e`;
  profiled/unprofiled stats 10,295/10,304 bytes at
  `9f14a3b06da7d5b0720ec3fd073f6475c50ed380e23cd323167935cdd096c046` /
  `4e66f701bab5812fb63f82beebd7bcb16cfab59e144c8430b8bcd327d20b6e19`;
  complete manifest 11,857 bytes/
  `c8b9e38c694eafc4987064213e79233ab042f326e745e0041e7ea383abaf9e7b`.

## D0197 — Measure HAG-NN count six before extending the envelope

- Date: 2026-08-11.
- Status: accepted a cheap explicit prototype direction; changed no product
  code, coverage, model, or automatic selection.
- Measure: from clean commit
  `6c5fca157c47a8280d15725307a36cf3ad80f62c`, five alternating exact pairs
  measure pinned PDAL `hag_nn(count=6)` at 1.019913125 seconds. The current
  `pdg` default correctly delegates at 1.041636052 seconds. Both produce the
  same 48,000,909-byte artifact at SHA-256
  `675b88f1902a37a659fe6ba837a4e0f6985f3cf54fd676a80df66c757205272c`.
- Control/profile: the fresh pinned output-shaped control is 0.296473777
  seconds, leaving 0.723439348 seconds/70.931468% of stock wall attributable
  to the stage. The exact-output optimized-oracle phase capture measures the
  filter at 0.768510391 seconds: 2D-index construction is only 0.063233061
  seconds/8.228003%, while post-index query/projection is 0.691737660
  seconds/90.010190%.
- Path: test a bounded extension of the existing mapped source, planner-owned
  masked 2D query, ordered projection, exact repair semantics, and generic
  one-binary64 publisher. A private index, new kernel family, model, selector,
  and calibration ladder are rejected at this decision point.
- Coverage: count six remains functionally supported through unchanged
  pinned-PDAL fallback; GPU-native no; performance-qualified no; automatically
  selected no. The evidence clears only a cheap-prototype gate.
- Next: B0137 writes the count-six proof matrix first, then extends only the
  strict explicit envelope. Retain it only for an exact current-binary
  complete-process improvement of at least 5--10%.
- Evidence: count-six/control reports 17,068/17,081 bytes at SHA-256
  `3b3c5c9b49aa0812a54fa1429a48016fcc1cd7fff4c9b4915011ad396c1b8b22` /
  `1e6500d8e26bd900ca991928fb4a4f3992d170fa01c64ce09b0a9bc6a5fdd3a2`;
  phase script/log 2,417/1,706 bytes at
  `2eb1d6113243bf8b2d221adbe354650804bb21e41766b5189ec90687f647b98d` /
  `cdc41e38409cfa7997adb1d52c190abcda36f7d8273d3699960544c7313862f5`;
  complete manifest 8,670 bytes/
  `78261f5852bb2cfb358ab205214638eca61f88cac578ed57818926d55a45b69a`.

## D0198 — Extend only the explicit exact HAG-NN envelope to count six

- Date: 2026-08-11.
- Status: accepted the bounded count-six implementation after its tests-first
  exactness and cheap complete-process gates; added no model or automatic
  selector.
- Path: the existing generic masked 2D query, ordered projection, exact host
  repair, resident bridge, mapped LAS source, and one-binary64 publisher now
  admit count six. No private index or new kernel family is introduced. Count
  seven and wider, option/layout/source drift, and ordinary unforced execution
  remain host-owned.
- Exactness: the expanded 110-case host and physical-CUDA matrices cover a
  unique sixth retained candidate on Grid and Morton BVH, interpolation,
  strict/equal/negative cutoffs, inclusive bounds, large/underflow/signed-zero
  arithmetic, insufficient/nonfinite-Z repair, nonfinite-XY exact fallback,
  the sixth/seventh tie boundary, Grid overflow/incompleteness, and count-seven
  fallback. Focused planner,
  rewrite, lifecycle, downstream bridge, direct-source/publication, and atomic
  rejection proofs pass. The complete Release unit binary passes 562 tests
  with only two optional local-corpus skips.
- Safety: focused host ASan/UBSan passes with leak detection disabled because
  LeakSanitizer is unavailable under ptrace. Compute Sanitizer memcheck,
  initcheck, and synccheck report zero errors; racecheck reports zero hazards
  across the bounded eight-test count-six lane.
- Cheap retention direction: one exact dirty-snapshot pair measures
  1.025019333 seconds pinned PDAL and 0.355610480 seconds direct, or 2.882422x.
  Both emit the same 48,000,909-byte LAS at SHA-256
  `675b88f1902a37a659fe6ba837a4e0f6985f3cf54fd676a80df66c757205272c`;
  the candidate proves the strict direct executor, mapped source and record
  summary, no host XYZ mirror, one predicted/observed planner-owned index,
  zero spill/fallback boundaries, direct binary64 output, matching boundary
  accounting, and terminal-spill elision. This clears the 5--10% prototype
  gate but is not performance qualification evidence.
- Coverage: functionally supported yes; GPU-native only inside the bounded
  explicit count-six query/projection envelope with exact host repair;
  performance-qualified no pending B0138; automatically selected no.
- Next: B0138 commits this implementation cleanly, runs five alternating exact
  current-binary pairs plus one fresh exact-output profile, and applies the
  endpoint stopping rule. No calibration ladder, model, or selector follows.

## D0199 — Qualify and stop the explicit HAG-NN count-six endpoint

- Date: 2026-08-12.
- Status: accepted the named clean count-six performance qualification and
  declared that endpoint sufficiently optimized; no model or automatic
  selector is added.
- Measure: from clean implementation commit
  `c3602239bf4d329f66dcc7fca19ac769ebdac239`, five alternating exact pairs
  measure 1.023952521 seconds pinned PDAL and 0.344893464 seconds for the
  required explicit direct route, or 2.968895x/66.317436% lower wall. Every
  run, the fresh NCU capture, and the unprofiled stats pass reproduce the same
  48,000,909-byte LAS at SHA-256
  `675b88f1902a37a659fe6ba837a4e0f6985f3cf54fd676a80df66c757205272c`.
- Proof: the candidate reports the strict planner-resident direct executor,
  mapped source and record summary, no host XYZ mirror, exactly one predicted
  and observed planner-owned index, zero spill/fallback boundaries, direct
  one-binary64 publication, matching boundary accounting, and terminal-spill
  elision.
- Profile/stopping rule: 20 launches total 5.324470 milliseconds/1.543801% of
  candidate wall; the shared gather is 4.880000 milliseconds/1.414930% and
  the ordered count-six projection is 0.104670 milliseconds/0.030349%.
  Canonical publication has only a 4.296978% full-elimination ceiling. The
  0.079468215-second complete manager would require a 21.700089% reduction to
  save 5% process wall, but no independent reusable component at that scale is
  identified; B0108 already rejects the apparent placement shortcut at only
  3.636364%. Further endpoint tuning is deferred.
- Coverage: functionally supported yes; GPU-native only in the bounded
  explicit count-six query/projection envelope with exact host repair and
  publication ownership; performance-qualified only for the named 1,000,002-
  point direct SM89 fixture; automatically selected no.
- Next: B0139 measures pinned PDAL `hag_nn(count=7)` and an output-shaped
  control before any implementation. It tests reuse of the same planner-owned
  query/direct-publication architecture; no private index, new kernel family,
  model, selector, or calibration ladder precedes that evidence.
- Evidence: paired report 34,223 bytes/SHA-256
  `406e95f0837596f08bfe96c403ec12326580777f13342abb38d2a585134c5025`;
  NCU report 1,544,971 bytes/
  `c83ea588b729fbc42c37393362cc1006d2b3bcaedda8debc2e3e2e9873e52514`;
  profiled/unprofiled stats 10,293/10,293 bytes at
  `1dad3d9567a4938f0f24a63b593df5816f6e903e7887e5f1ea3d614fcd6d08d8` /
  `cfa4b360aeeb83727d0ed4d8ae62c8b92d1c8f10c6056bfefb20a118eb1bf673`;
  complete manifest 11,887 bytes/
  `1d0ca83cc719ade40f7feb886f729a6f75bc1b3f7471afd0129e92fac4097f27`.

## D0200 — Measure HAG-NN count seven before extending the envelope

- Date: 2026-08-12.
- Status: accepted a cheap explicit prototype direction; changed no product
  code, coverage, model, or automatic selection.
- Measure: from clean commit
  `a03b5bc19c7f8f1588de8a8821c12b4a3c52b6f6`, five alternating exact pairs
  measure pinned PDAL `hag_nn(count=7)` at 1.050776026 seconds. The current
  `pdg` default correctly delegates at 1.075603836 seconds. Both produce the
  same 48,000,909-byte artifact at SHA-256
  `98fb53141a3da0e8d0d3bf94384f7dba21d8468b9448189a881f079b34df3536`, which
  differs from the count-six artifact, so the seventh retained candidate
  genuinely changes the published interpolation.
- Fixture: the retained deterministic offset-grid fixture is reused unchanged.
  Ground rows lie on `Y == 0` at integer X while query rows carry a `Y / 5` X
  offset, so all candidate distances are distinct; the enumerated
  seventh/eighth separation is 0.2 for interior rows and 1.0 at both X
  extremes. No new fixture is generated.
- Control/profile: the fresh pinned output-shaped control is 0.292334145
  seconds, leaving 0.758441881 seconds/72.179214% of stock wall attributable
  to the stage, above count six's 70.931468%. The exact-output
  optimized-oracle phase capture measures the filter at 0.801285042 seconds:
  2D-index construction is only 0.077581237 seconds/9.682102%, post-index
  query/projection is 0.708272845 seconds/88.392121%, and 0.015430960 seconds
  covers the ground/non-ground split, bounds, and return.
- Path: test a bounded extension of the existing mapped source, planner-owned
  masked 2D query, ordered projection, exact repair semantics, and generic
  one-binary64 publisher. A private index, new kernel family, model, selector,
  and calibration ladder are rejected at this decision point.
- Coverage: count seven remains functionally supported through unchanged
  pinned-PDAL fallback; GPU-native no; performance-qualified no; automatically
  selected no. The evidence clears only a cheap-prototype gate.
- Next: B0140 writes the count-seven proof matrix first, then extends only the
  strict explicit envelope. Retain it only for an exact current-binary
  complete-process improvement of at least 5--10%.
- Evidence: count-seven/control reports 17,088/17,112 bytes at SHA-256
  `694a25733b1ea915f79fa56eac8502485d071c8a3629a28a5b6d2bac7579be94` /
  `ad9a2d28808bbfdc64eda6a49eaeb7b0175a3341847cb8dfb28e23fc46ce9fa4`;
  phase script/log 2,562/1,706 bytes at
  `e4ac7c6439c094ff99eeb04c9aaab9d67f4512371d4bacc979b221437d49d655` /
  `0a40b23ca920e76230b386e4ec4dafa55d025de8aea116ba3e9fbf0f07ffe36c`;
  complete manifest 9,513 bytes/
  `846a142bafe25921232dfff03766cad82ffdd2572249766b4591a7d70516227d`.

## D0201 — Extend only the explicit exact HAG-NN envelope to count seven

- Date: 2026-08-12.
- Status: accepted the bounded count-seven implementation after its tests-first
  exactness and cheap complete-process gates; added no model or automatic
  selector.
- Path: the existing generic masked 2D query, ordered projection, exact host
  repair, resident bridge, mapped LAS source, and one-binary64 publisher now
  admit count seven. The device projection was already count-generic — it
  accumulates over `min(groundCount, neighbors)` with no fixed-width state —
  so this raises seven bounds and adds proofs rather than a kernel family.
  No private index is introduced. Count eight and wider, option/layout/source
  drift, and ordinary unforced execution remain host-owned.
- Exactness: the expanded 128-case host and 120-case physical-CUDA matrices
  cover a unique seventh retained candidate on Grid and Morton BVH,
  interpolation, strict/equal/negative cutoffs, inclusive bounds,
  large/underflow/signed-zero arithmetic, insufficient/nonfinite-Z repair,
  nonfinite-XY exact fallback, the seventh/eighth tie boundary, Grid
  overflow/incompleteness, and count-eight fallback. Every native-required
  count-seven fixture was verified tie-free in binary64 before acceptance;
  one candidate arithmetic ground was rejected and replaced because its
  squared distance was bit-identical to another ground's, which the exactness
  guard correctly declined. Focused planner, rewrite, lifecycle, downstream
  bridge, direct-source/publication, and atomic rejection proofs pass. The
  complete Release unit binary passes 573 tests with only two optional
  local-corpus skips.
- Safety: focused host ASan/UBSan passes 386 tests with two optional corpus
  skips and leak detection disabled because LeakSanitizer is unavailable under
  ptrace. Compute Sanitizer memcheck, initcheck, and synccheck report zero
  errors; racecheck reports zero hazards across the bounded nine-test
  count-seven lane.
- Cheap retention direction: one exact dirty-snapshot pair measures
  1.044514334 seconds pinned PDAL and 0.361863314 seconds direct, or
  2.886489x. Both emit the same 48,000,909-byte LAS at SHA-256
  `98fb53141a3da0e8d0d3bf94384f7dba21d8468b9448189a881f079b34df3536`;
  the candidate proves the strict direct executor, mapped source and record
  summary, no host XYZ mirror, one predicted/observed planner-owned index,
  direct binary64 output, matching boundary accounting, and terminal-spill
  elision. A separate stats execution reproduces the same hash with one
  25,000,050-byte upload boundary, zero spill boundaries, and zero fallback
  boundaries. This clears the 5--10% prototype gate but is not performance
  qualification evidence.
- Coverage: functionally supported yes; GPU-native only inside the bounded
  explicit count-seven query/projection envelope with exact host repair;
  performance-qualified no pending B0141; automatically selected no.
- Next: B0141 commits this implementation cleanly, runs five alternating exact
  current-binary pairs plus one fresh exact-output profile, and applies the
  endpoint stopping rule. No calibration ladder, model, or selector follows.

## D0202 — Qualify and stop the explicit HAG-NN count-seven endpoint

- Date: 2026-08-12.
- Status: accepted a fixture-specific performance qualification and applied the
  endpoint stopping rule; added no model, calibration ladder, or automatic
  selector.
- Measure: from clean implementation commit
  `437b0b0606cfd4f1e195511e2ad928e281b944dd`, five alternating exact pairs
  measure 1.051637191 seconds pinned PDAL versus 0.355388678 seconds for the
  required explicit direct count-seven route, or 2.959118x and 66.206152%
  lower wall. Every pair emits the same 48,000,909-byte LAS at SHA-256
  `98fb53141a3da0e8d0d3bf94384f7dba21d8468b9448189a881f079b34df3536`.
- Route proof: `planner_resident_shared_index_direct_las`, mapped source,
  record-summary index configuration, no complete host XYZ mirror, one
  predicted and observed planner-owned index, direct one-binary64 output,
  matching boundary accounting, one 25,000,050-byte upload, zero spill, zero
  fallback, and terminal-spill elision.
- Profile: the fresh output-bound NCU capture reproduces the same hash and
  records 20 launches totaling 6.421888 milliseconds, only 1.807004% of
  candidate median. The shared `knnGatherKernel` is 5.961088
  milliseconds/1.677343% and the count-seven HAG projection only 0.120224
  milliseconds/0.033829%. An unprofiled stats pass reconciles 0.301803236
  seconds of engine command work: 0.179161423 validation/placement/preflight,
  0.103735030 rewritten manager, 0.017892503 canonical publication, and
  0.001014280 other control.
- Stopping rule: no reusable surface survives its own prior evidence.
  Canonical publication is a 5.034629% full-elimination ceiling that cannot
  actually be eliminated, since the artifact must still be written, and
  B0129/B0131 already stopped at comparable 7.5%/7.8% ceilings. The manager is
  29.189177% of wall but owns the exact shared-index query, transfers, and
  result path and would need a 17.130% reduction merely to save 5% complete
  wall, while all its kernels together are 1.807004%. The largest interval,
  validation/placement/preflight at 50.412811%, is the process-level CUDA
  startup B0054 measured as necessary and B0108 prototyped away for only
  3.636364%.
- Structural observation: that startup partition is roughly fixed in absolute
  terms, so it necessarily dominates a sub-half-second endpoint. It is a
  shared limiter of every small direct route rather than anything specific to
  count seven, and it is already closed by measurement. Recording it here
  prevents a future slice from rediscovering it as an apparent count-seven
  opportunity.
- Coverage: functionally supported yes; GPU-native only inside the bounded
  explicit count-seven query/projection envelope with exact host repair;
  performance-qualified only for the named 1,000,002-point SM89 fixture;
  automatically selected no.
- Next: B0142 selects the next slice from the performance-first queue. Count
  eight and wider remain host-owned. Because the lane is already
  count-generic, the queue may consider closing the remaining range as one
  parameterized proof envelope rather than one count per slice; that decision
  needs its own measure-first entry.

## D0203 — Close the remaining HAG-NN count range as one parameterized envelope

- Date: 2026-08-12.
- Status: accepted a scoping decision from measurement; changed no product
  code, coverage, model, or automatic selection.
- Measure: from clean commit `c4c8dc2af`, five alternating exact pairs at each
  of counts 8, 12, 16, 24, 32, 48, and 64 put pinned PDAL at 1.075645517,
  1.147854476, 1.219574480, 1.371048773, 1.508035821, 1.778506026, and
  2.057952491 seconds. Against a fresh 0.290153739-second pinned output
  control the stage surface is 0.785491778 through 1.767798752 seconds, rising
  monotonically from 73.0252% to 85.9009% of stock wall. Every count emits a
  distinct 48,000,909-byte artifact and the current default delegates all of
  them exactly at 0.970x--0.988x.
- Finding: cost is strongly sub-linear in count. Seven to 64 neighbours is
  9.142857x more neighbours for only 2.3241x more stock stage time, and stock
  per-neighbour cost falls 3.93x. The dominant stock cost is the largely
  count-independent per-query index traversal, consistent with D0200's
  88.392121% post-index attribution. The acceleration opportunity therefore
  grows, not shrinks, with count.
- Fixture: enumerating both query lanes at interior and extreme rows shows the
  retained offset-grid fixture keeps the k/(k+1) boundary distinct for every k
  in 1..64 at a minimum separation of 0.2. One fixture serves the whole range.
- Engine readiness: the device projection already accumulates over
  `min(groundCount, neighbors)` with no fixed-width state; the resident
  preflight already budgets `3 + count * (4 + 8)` bytes per point, which is
  771 bytes per point and 771 MB for this fixture at count 64; and the
  neighborhood region already rejects `maximumNeighbors > 64`. Raising the
  admissible bound to 64 introduces no new constant and no unbudgeted
  allocation — 64 is the spatial index's own pre-existing limit.
- Decision: close counts eight through 64 as one parameterized envelope rather
  than one count per slice. The count ladder is a proof ladder, not an
  implementation or cost ladder; per-count slices would spend roughly 57
  further three-commit triads re-proving a structurally identical envelope.
  This does not relax any proof obligation: the matrix must still cover the
  distinct (k+1) candidate, cutoffs, bounds, arithmetic, repair, and fallback
  behaviour, but generated over count instead of transcribed per count.
- Coverage: unchanged by this entry. Counts above seven remain functionally
  supported through pinned fallback, not GPU-native, not
  performance-qualified, and not automatically selected.
- Next: B0143 raises the bound to the shared index cap with a count-generated
  proof matrix, keeps `count > 64` and every option/layout/source drift
  host-owned, and qualifies representative counts rather than every count. No
  calibration ladder, placement model, or automatic selector follows.

## D0204 — Reprioritize toward I/O, residency, and selection; gate new per-stage kernels

- Date: 2026-08-12.
- Status: accepted a scheduling and labeling change from a re-reading of the
  existing benchmark corpus. Changes no product code, retires no stage or
  route, and removes no kernel.
- Form: this is a new entry rather than an edit to D0099 because `AGENTS.md`
  makes `DECISIONS.md` append-only and forbids rewriting prior entries. D0099
  remains valid and is extended, not replaced: its measure-and-profile-first
  loop still governs, and this entry adds an explicit admission rule plus a
  priority order.
- Verification first. The proposal that prompted this entry asserted that
  kernel time is a rounding error in real endpoints. Every number it cited is
  accurate — count-seven 20 launches/6.421888 ms/1.807004% of wall with the HAG
  projection at 0.120224 ms/0.033829%; B0138 1.543801%; B0135 1.277263%;
  B0113 0.588151%; B0095 2.460417%; startup 50.412811% of the count-seven
  endpoint; SMRF 0.599x, PMF 0.583x, CSF 0.289x; B0052 -35.541759%,
  B0111 -34.483%, B0060 -8.02744%, B0062 -5.693104% — but two of its framings
  do not survive the record and are corrected here.
- Correction 1: "kernels are noise" is false as a generalization. The
  distinction is shared versus standalone, not kernel versus non-kernel. D0074
  took the 22M kNN gather from 32.2 seconds to 0.49 seconds, the single largest
  measured win in this repository. The same shared `knnGatherKernel` is still
  137.767712 milliseconds and 97.3487% of kernel time on the qualified 4M
  nndistance endpoint, roughly 22% of its 0.623028204-second wall, and B0064's
  parallel repair kernel took a further 9.242103% off that endpoint. B0124 also
  leaves its shared gather above the 5% gate. Shared kernel work has paid more
  than anything else in the record and is not deprioritized. What has never
  paid is the *standalone per-stage* kernel: SMRF, PMF, CSF, ELM,
  transformation, IQR, MAD, and colorinterp are all measured slower than stock
  and all correctly host-selected. The reprioritization survives with this
  label corrected; the conclusion is the same but the reason is different.
- Correction 2: the ~50% startup share is a small-endpoint artifact, not a
  general limiter. The partition is about 0.176 seconds and roughly constant in
  absolute terms, so it is ~50% of a 0.355-second HAG endpoint, ~28% of the
  0.623-second 4M nndistance endpoint, and negligible at 22M. B0054 measured it
  as necessary process-level CUDA startup and B0108 prototyped bypassing it for
  only 3.636364%. It is already closed and must not be reopened without new
  evidence, and it must not be cited as a reason to reprioritize large-workload
  work.
- No recent slice contradicts the corrected claim. B0139--B0142 are consistent
  with it: the HAG-NN family's kernels are genuinely negligible while its stock
  host cost is 72.3867%--85.9009% of wall, which is precisely the profile that
  makes query/boundary work rather than kernel work the target.
- Priority order (recorded in full in `IMPLEMENTATION_PLAN.md`): native
  LAZ/COPC I/O; cross-stage residency and planner-owned product reuse;
  fallback-frequency reduction; automatic selection of already-qualified
  routes; then remaining per-stage kernels as a catalog-coverage obligation.
  The I/O ranking rests partly on extrapolation and this entry says so: every
  benchmark here reads uncompressed LAS and no LAZ or COPC measurement exists,
  so the first P4 slice must be a measurement rather than an implementation.
- Admission rule for new standalone per-stage kernels: none is scheduled
  without a same-machine profile showing that stage dominating a real
  end-to-end pipeline. Exact-but-slower native implementations remain
  force-only, are labeled catalog coverage, and are never presented as
  speedups. D0019's catalog-wide obligation is unchanged; this governs
  scheduling and labeling only.
- Automatic-selection candidates. Ten routes are GPU-native with a
  positive-sounding qualification but unselected; four —
  `filters.transformation`, `filters.iqr`, `filters.mad`, `filters.colorinterp`
  — are qualified *negatively* and must never be promoted. The six genuine
  candidates are `filters.sort`, `filters.neighborclassifier`,
  `filters.label_duplicates`, `filters.skewnessbalancing`, `filters.hag_nn`
  explicit `count=1..7`, and `filters.hag_delaunay` explicit `count=3`. All six
  are blocked by the same two missing artifacts — a calibrated placement model
  and a fail-closed admission predicate — so this is one reusable mechanism
  already built five times (B0080, B0092/B0093, B0097, D0106, D0108), not six
  separate pieces of work. Several are additionally gated by data-dependent
  exact repair, so admission must stay narrow and fail-closed.
- Per-count proof ladders are retired. B0142 measured that HAG-NN stock cost is
  strongly sub-linear in count (9.142857x more neighbours for 2.3241x more
  stock stage time) while the engine is already count-generic, so D0203's
  parameterized envelope replaces per-count slices. Generalizing: a proof
  ladder over a numeric option is closed as one parameterized envelope whenever
  the implementation is already generic over that option and the measured cost
  is not superlinear in it. No proof obligation is relaxed by this; the matrix
  is generated over the option instead of transcribed per value.
- Coverage: unchanged by this entry. No stage or route is retired, demoted in
  capability, or removed; only scheduling priority and labeling change.
- Next: the next implementable slice is B0143, the already-written D0203
  parameterized HAG-NN envelope, which is complete and gate-green but not yet
  committed. After it, the first P4 slice is a measurement of LAZ/COPC decode
  against pinned PDAL on the reference machine.

## D0205 — Close the exact HAG-NN envelope at the shared index cap of 64

- Date: 2026-08-12.
- Status: accepted the parameterized count envelope authorized by D0203/D0204
  after its tests-first exactness and cheap complete-process gates; added no
  model or automatic selector.
- Path: the bounded exact `filters.hag_nn` lane now admits counts one through
  64, which is the shared spatial index's own pre-existing `maximumNeighbors`
  cap rather than a new constant. The device projection already accumulated
  over `min(groundCount, neighbors)` with no fixed-width state and the resident
  preflight already budgeted `3 + count * (4 + 8)` bytes per point, so this
  slice raises seven bounds and adds proofs rather than a kernel family. Count
  65 and wider, option/layout/source drift, and ordinary unforced execution
  remain host-owned.
- Exactness: the process matrices grow to 200 host and 192 CUDA cases. Counts
  one through seven keep their explicit hand-written cases; counts 8, 16, 32,
  and 64 are generated over count, each shape stating the property it proves —
  distinct (k+1) candidate on both index backends, interpolation,
  equality/partial/negative cutoffs, inclusive bounds, large-finite, underflow,
  signed-zero, insufficient-ground, nonfinite-Z, nonfinite-XY, candidate-tie,
  all-infinite overflow, and bounded-grid incompleteness. Every generated
  fixture that requires the native path was proved tie-free in exact binary64
  before acceptance: 44 native-required cases verified tie-free and 12
  tie-proof cases verified to genuinely tie. The complete Release unit binary
  passes 577 tests with two optional local-corpus skips.
- Direct-route proof at width: the shared 7x3 process fixture has only seven
  ground rows, so counts above seven legitimately fall back for insufficient
  ground. A wider 79-ground/237-point fixture was added so the strict direct
  source and publisher are still proved at counts 16 and 64, including exact
  output, the direct executor, one observed index, and zero spill boundaries.
  This was found by the gate failing honestly rather than by inspection.
- Safety: focused host ASan/UBSan passes 386 tests with two optional corpus
  skips. Compute Sanitizer memcheck, initcheck, and synccheck report zero
  errors and racecheck zero hazards across the generated wide-count lane.
- Cheap retention direction: one exact dirty-snapshot pair at count 16 measures
  1.224073288 seconds pinned PDAL and 0.366412903 seconds direct, or
  3.340694x, reproducing B0142's pinned count-16 artifact at SHA-256
  `09d466acd02a63c9720b5ee6af9342e0857d91b4816183e10260d5c6c754574e` and
  proving the direct executor, mapped source, no host XYZ mirror, one index,
  and terminal-spill elision. This is the highest HAG-NN result recorded so
  far and confirms B0142's prediction that the opportunity grows with count:
  stock cost rises with count while the device path's cost barely moves. It
  clears the retention gate but is not performance qualification evidence.
- Coverage: functionally supported yes; GPU-native inside the bounded explicit
  count-one-through-64 query/projection envelope with exact host repair;
  performance-qualified pending B0144; automatically selected no.
- Next: B0144 commits this cleanly and qualifies representative counts —
  not every count, per D0203 — with five alternating exact pairs and a fresh
  exact-output profile each, then applies the endpoint stopping rule.

## D0206 — Qualify the wide-count HAG-NN envelope and reopen the shared gather

- Date: 2026-08-12.
- Status: accepted a fixture-specific performance qualification at
  representative counts and, unusually, declined to stop. Added no model,
  calibration ladder, or automatic selector.
- Measure: from clean implementation commit `b889df20b`, five alternating
  exact pairs at count 16 measure 1.224026575 seconds pinned PDAL versus
  0.397297088 seconds direct (3.080885x, 67.541792% lower wall), and at count
  64 measure 2.056411590 versus 0.470088827 seconds (4.374517x, 77.140334%
  lower). Both are byte-exact at 48,000,909 bytes and reproduce B0142's pinned
  artifacts for those counts.
- Representative rather than exhaustive: per D0203 this qualifies counts 16 and
  64 — the measured best result and the cap — not every count. The qualification
  is explicitly fixture-specific and count-specific and must not be read as a
  claim about intermediate counts.
- Trend: speedup rises with count (2.518270x at one, 2.968895x at seven,
  3.080885x at 16, 4.374517x at 64) because stock cost grows with count while
  the device path's cost grows far more slowly. This is the direct confirmation
  of B0142's prediction and the reason the parameterized envelope was worth
  closing in one slice.
- Profile: 39 launches at both counts. All kernels are 5.297114% of wall at
  count 16 and 17.937498% at count 64; the shared `bvhKnnGatherKernel` alone is
  5.108745% and 17.576763%. The ordered HAG projection stays negligible at
  0.298016 and 1.242848 milliseconds. Both wide counts select the adaptive
  Morton-BVH gather where counts up to seven selected the uniform-grid gather.
- Startup behaves as predicted: its share falls from 50.412811% at count seven
  to 28.2567% at count 64 purely because the wall grew, confirming D0204's
  claim that the partition is roughly fixed in absolute terms and that its
  apparent dominance is a small-endpoint artifact. It remains closed by B0054
  and B0108.
- Decision not to stop: for the first time on this family, a reusable component
  clears the 5--10% gate by a wide margin. The shared kNN gather at 17.576763%
  of count-64 wall is a genuine target, and it is shared by every neighborhood
  consumer in the catalog — nndistance, LOF, statistical outlier,
  approximatecoplanar, normal, the eigen family, neighborclassifier, and the
  HAG family all query it. Optimizing it is exactly the category D0204
  identified as the one where kernel work has historically paid (D0074's 22M
  gather 32.2 s -> 0.49 s; B0064's 9.242103% endpoint reduction). The stopping
  rule therefore does not fire here.
- Scope guard: this reopens the *shared* gather only. It authorizes no new
  standalone per-stage kernel, and D0204's admission rule still governs those.
  Any gather change must hold every existing exact contract — candidate order,
  tie and incompleteness status reporting, and the host repair predicates that
  depend on them — across all shared-index consumers, not just HAG-NN.
- Coverage: functionally supported yes; GPU-native for counts one through 64
  inside the proved envelope with exact host repair; performance-qualified for
  the named count-16 and count-64 1,000,002-point direct fixtures on the
  recorded SM89 machine; automatically selected no.
- Next: B0145 profiles the shared Morton-BVH kNN gather at wide count against
  its uniform-grid counterpart and names its limiter before proposing any
  change. B0053 already rejected bounded launch-shape variants for the
  uniform-grid gather, so that negative result must be checked against the BVH
  path rather than assumed to transfer.

## D0207 — The shared BVH gather's limiter is register pressure

- Date: 2026-08-12.
- Status: profiling only. Changed no product code, coverage, model, or
  automatic selection; rejected two candidate directions by measurement.
- Backend rejected as the lever: five alternating same-binary count-64 runs
  measure adaptive at 0.476555539 seconds, forced Morton BVH at 0.477948684
  (+0.2923%, noise), and forced uniform grid at 0.832525226 (+74.6964%), all
  byte-identical. The planner's adaptive selection is already optimal at wide
  count, so there is no backend change to make.
- Limiter named: `bvhKnnGatherKernel` runs 82.63 milliseconds over 15,626
  blocks of 64 threads with 48 registers per thread. Compute throughput is
  67.00% against 49.95% memory and DRAM, so it is compute/latency-bound rather
  than bandwidth-bound. Achieved occupancy is 72.72% against an 83.33%
  theoretical ceiling, and the block limits are registers 20, SM 24, warps 24,
  shared memory 32 — register pressure is the binding constraint. Static and
  dynamic shared memory are both zero, so shared-memory staging is unused and
  is not the constraint.
- Next hypothesis: raise occupancy by reducing register pressure on this
  specific kernel — `__launch_bounds__` or a launch-shape change — and measure
  it. B0053's rejection of bounded launch-shape variants was measured on the
  uniform-grid `knnGatherKernel`, which has a different register and occupancy
  profile, so it must be re-measured here rather than assumed to transfer.
- Guard: this kernel is shared. Any change must preserve exact candidate order,
  tie and incompleteness status reporting, and the host repair predicates that
  depend on them, across every shared-index consumer, and must be validated
  against the whole neighborhood family rather than HAG-NN alone. A change that
  improves HAG-NN while perturbing another consumer's repair predicate is a
  regression even if its own gate passes.
- Next: B0146 may prototype the occupancy change behind the usual cheap gate,
  retaining it only for an exact same-binary 5--10% complete-process
  improvement on at least one shared-index consumer, with the full
  neighborhood matrix and all four Compute Sanitizer tools clean.

## D0208 — The goal is speed on real pipelines, not catalog-wide CUDA coverage

- Date: 2026-08-12.
- Status: accepted a goal restatement. Supersedes the catalog-wide *CUDA
  coverage* obligation in D0019 and the corresponding language in `AGENTS.md`
  priority 7 and `spec.md`'s mission. D0019's exactness, honesty, and
  separate-reporting requirements are unchanged and still binding, as is
  D0002's exact-output contract.
- Restated goal: make the fork as fast as possible on the pipelines people
  actually run. CUDA is one means among several, not the objective. Success is
  wall-clock time on the reference pipelines against the pinned oracle on
  identical hardware, not the number of stages carrying a CUDA path.
- What changes:
  - **Host work becomes first-class.** Threading, algorithms, I/O, decode and
    encode, allocation, and boundary removal count as wins even when no GPU is
    involved. A host path that is exact and fast is a finished answer, not
    scaffolding. This is the largest practical change, because several of the
    most common stages are host-bound and CUDA measured *worse* on them.
  - **CUDA only where it wins.** A CUDA path is built or kept only where it is
    measured faster end-to-end. Exact-but-slower lanes stay in-tree,
    host-selected, and labeled catalog coverage.
  - **Catalog-wide CUDA coverage is no longer a release criterion**, and the
    "zero temporary upstream fallbacks at the full-product milestone"
    requirement is withdrawn. The remaining native count is no longer tracked
    as a completion metric.
- What does not change: catalog-wide *functional* coverage, byte-exact default
  output against the pinned oracle, the four independent coverage labels, the
  planner-owned shared-index rule, benchmark evidence standards, and the
  prohibition on presenting a host fallback as GPU-native.
- Retention rather than removal: the measured-negative lanes — SMRF 0.599x,
  PMF 0.583x, CSF 0.289x, ELM 0.866x, transformation, IQR, MAD, colorinterp —
  are kept in-tree, exact and host-selected. They may win on other hardware or
  frame sizes and their proofs are already paid for. No further investment is
  scheduled against them, and they are labeled catalog coverage per D0204.
- Reference pipelines: six end-to-end workflows now define the target and live
  in `bench/pipelines/reference/` — `r1-translate` (LAZ, crop, reproject),
  `r2-ground-normalize` (SMRF, HAG), `r3-dtm` (SMRF, ground, GDAL raster),
  `r4-denoise-thin` (outlier, range, sample), `r5-copc-query` (COPC bounds and
  resolution query, stats), and `r6-features` (normal, covariance features).
  All six were executed against the pinned oracle on a 25,000-point LAZ/COPC
  fixture before being adopted, so the target set is validated rather than
  assumed. Two corrections came out of that validation: `r1` needs an explicit
  `in_srs` because the retained bench data is deliberately SRS-stripped, and
  its crop belongs before the reprojection so the bounds are expressed in the
  source CRS.
- Deliberate composition: the set includes workflows where the current GPU
  answer is worse than the host answer — `r2` and `r3` both run SMRF, measured
  at 0.599x. Under the previous contract that was an embarrassment to be fixed
  by more CUDA. Under this one it is simply a host-optimization target, which
  is the point of the restatement.
- Evidence that the reframe is warranted: the existing bench corpus contains 53
  pipelines, essentially all single-stage, and no end-to-end workflow
  benchmark at all. The product has therefore been steering by stage coverage
  because that is what it measured. D0204 already found kernels at
  1.277263%--2.460417% of wall on qualified endpoints while boundary and I/O
  work delivered -35.541759%, -34.483%, -8.02744%, and -5.693104%.
- Naming: the product's identity moves from "PDAL-GPU" toward "fast PDAL". The
  working CLI and namespace `pdg` are unchanged for now; P6's public naming
  gate is the place to resolve this and should be decided before publication.
- Next: B0146 builds the reference-pipeline harness and fixtures and records
  the first complete baseline of all six against the pinned oracle. That
  baseline replaces stage count as the progress metric, and it also converts
  D0204's LAZ/COPC ranking from extrapolation into measurement — the pinned
  oracle reads and writes both, so the differential is available immediately.

## D0209 — Admission policy, not kernels, is what blocks real pipelines

- Date: 2026-08-12.
- Status: accepted a measured baseline and a re-pointed queue. Changed no stage
  behavior; added the reference harness and fixtures only.
- Baseline: on the six D0208 reference workflows at 1M points, `pdg` measures
  0.903899x, 0.980306x, 0.970380x, 1.004472x, 0.917378x, and 0.999277x against
  pinned PDAL. Output is byte-exact on all six. **The product is slower than
  stock on five of six workflows people actually run and never meaningfully
  faster.** Overhead is roughly constant at +0.02 to +0.05 seconds, so it hurts
  most on the shortest and most common workflow.
- Diagnosis: not one qualified route fires. Every workflow runs
  `pdal_standard_host`. Single-variable probing identifies three independent
  admission blockers, all engine policy rather than algorithmic limits:
  any `extra_dims` on the writer, named or `all`, yields
  `invalid_runtime_facts`; `compression=true` LAZ output yields
  `invalid_runtime_facts`; and two neighborhood consumers in one graph yield
  `mixed_calibration_models`. Each alone is fatal. Writing computed dimensions
  is the entire purpose of a feature filter and LAZ is the standard interchange
  format, so these are ordinary shapes, not exotic ones.
- This reframes the whole optimization queue. D0204 already demoted per-stage
  kernels and D0207 found a genuine 17.576763%-of-wall shared-gather target,
  but none of that reaches a user while admission refuses every real graph. The
  cheapest available speed is not a faster kernel; it is letting the kernels we
  already have run at all.
- Honest correction retained: an intermediate probe concluded LAZ was not a
  blocker by comparing `r6` under LAZ and LAS. That was wrong — `r6` was
  already blocked twice over, masking the LAZ effect. Single-variable probes
  supersede multi-variable comparisons for admission questions.
- Upstream property recorded: the pinned `readers.copc` is multi-threaded and
  its point order varies run to run, so any exact contract over COPC input must
  pin `requests=1` or impose an explicit ordering. This is upstream behavior,
  not a `pdg` defect, and it is now encoded in `r5-copc-query`.
- Next, in order: B0147 admits `extra_dims` writers, since a feature pipeline
  that cannot write its features is useless and this blocks `r6` and most real
  feature work. B0148 admits LAZ output. B0149 addresses multi-consumer
  neighborhood graphs, which is the `mixed_calibration_models` rule from D0077
  and needs a calibration answer rather than a policy relaxation. Each must
  hold the exact-output contract and be measured on the reference baseline
  rather than an isolated endpoint.
- The D0207 gather optimization is not cancelled but is queued behind these,
  because a 17% improvement to a kernel that never runs is worth nothing.

## D0210 — Widen the native I/O option envelope before anything else

- Date: 2026-08-12.
- Status: accepted a root-cause correction to D0209 and a re-ordered queue.
  Changed no product code.
- Root cause of B0146's baseline: a single rule, not three blockers. A
  `readers.las`/`writers.las` stage is native only when it carries no options
  beyond `type`, `filename`, `tag`, and `inputs`
  (`src/plan/Pipeline.cpp:2558`), and `lasHeaderFacts` refuses runtime facts
  unless both endpoints are native, so any writer option at all drops the whole
  pipeline to `pdal_standard_host`. Probes confirm `a_srs`, `scale_x`,
  `offset_x`, `forward`, `minor_version`, `dataformat_id`, `extra_dims`, and
  `compression` each individually disqualify acceleration.
- Correction to D0209: it called all the blockers "engine policy rather than
  algorithmic limits". That holds for writer options, which are header and
  metadata concerns that do not touch the point-processing mathematics, but not
  for LAZ. `src/io/las/Las.cpp:322` throws "compressed LAZ records require the
  chunk decoder" and no such codec exists in the engine, so LAZ is a real
  capability gap rather than a predicate to relax.
- Consequence: D0204 ranked native I/O first on an explicit extrapolation. That
  ranking is now measured. The reference workflows read and write LAZ, and the
  engine physically cannot, so LAZ decode and encode is both the largest
  measured cost and a hard prerequisite for any acceleration reaching a user.
- Queue, in order:
  1. **B0148 — widen the writer option envelope.** Admit header and metadata
     options that provably do not change point mathematics: `a_srs`,
     `forward`, `minor_version`, `dataformat_id`, and `scale`/`offset` where
     the publisher can honour them exactly. This is the cheapest speed in the
     product because the compute path already exists and is proven exact. Each
     admitted option needs an exact differential and must be measured on the
     reference baseline.
  2. **B0149 — `extra_dims` in the resident publisher.** Generalize B0107's
     one-binary64 direct publisher to arbitrary declared extra dimensions, so a
     feature pipeline can write its features.
  3. **B0150 — the LAZ chunk codec (P4).** Decode first, then encode, with
     chunk-parallel decode as the design target. This is real implementation
     work, not a predicate change, and it unblocks every reference workflow.
  4. **B0151 — `mixed_calibration_models`** for multi-consumer neighborhood
     graphs, which needs a calibration answer rather than a policy relaxation.
- The D0207 shared-gather occupancy work stays queued behind all of these.
- Guard: every item above must hold the exact-output contract and be measured
  on the reference baseline rather than an isolated endpoint. Widening
  admission is exactly the kind of change that can trade correctness for
  speed, so each admitted option requires its own differential proving the
  published bytes are unchanged.

## D0211 — Automatic selection, not admission or kernels, is the measured opportunity

- Date: 2026-08-12.
- Status: accepted a measured redirection of the queue; reverted a prototype.
  Changed no product code.
- Measure: on clean unmodified code, `readers.las -> filters.normal(knn=8) ->
  writers.las` over 1M points takes 4.372143 seconds in pinned PDAL, 4.407116
  through the public `pdg pipeline`, and 1.181099 through `pdg resident`. All
  three outputs are byte-identical. The resident executor is 3.70x faster than
  the public command on the same pipeline, today, and the public command does
  not use it.
- Root cause, corrected again: automatic selection matches a small whitelist of
  exact graph shapes — B0043's LOF shape, B0045/B0050's nndistance shape,
  B0075's three-consumer composition, B0080, B0092, B0093, B0097, B0127 — while
  `pdg resident` will execute anything runtime placement can prove. Everything
  outside the whitelist runs stock, which is every reference workflow.
- This supersedes D0210's ordering. D0210 put writer-option admission first on
  the reasoning that admission blocked acceleration. Admission does block the
  *resident* path, and that finding stands, but it is not why the reference
  baseline is slow: an option-free pipeline that placement accepts is still run
  unaccelerated by the public command. Widening admission without widening
  selection changes nothing a user can observe, which the reverted `a_srs`
  prototype demonstrated directly at three sizes.
- Prototype disposition: the `a_srs` admission change was exact and did enable
  `planner_resident_shared_index`, but produced 0.993601x, 0.991722x, and
  1.001350x at 250K, 1M, and 4M through the public command. Fully reverted per
  B0056/B0061/B0108; the measurement is retained as the deliverable.
- Revised queue:
  1. **B0149 — generalize automatic selection.** Admit any pipeline that
     runtime placement already proves, using D0106's existing fail-closed
     machinery — separate refusal from committed execution, decline before side
     effects, never retry after commitment — instead of a shape whitelist. This
     is the single highest-value change available: roughly 3.7x is already
     built, exact, and sanitizer-clean, and is simply unreachable from the
     command users run. It must be gated by the reference baseline and by the
     existing exactness matrices, and it must stay fail-closed: a shape that
     placement cannot prove must continue to run stock.
  2. **B0150 — writer option admission** (formerly B0148), which widens what
     placement can prove and therefore what selection can reach.
  3. **B0151 — `extra_dims` in the resident publisher.**
  4. **B0152 — the LAZ chunk codec (P4)**, still a real capability gap.
  5. **B0153 — `mixed_calibration_models`.**
  The D0207 shared-gather occupancy work remains behind all of these.
- Standing caution recorded three times now in one session: B0147 corrected
  B0146's attribution, and this entry corrects B0147's. Each correction came
  from a single-variable measurement rather than from reasoning about the code.
  Admission and selection are separate mechanisms and must be measured
  separately; a change to one proves nothing about the other.

## D0212 — LAZ before selection: the codec gates everything else

- Date: 2026-08-12.
- Status: accepted a measured reordering; reverted a prototype. Changed no
  product code.
- Measure: relaxing all three automatic-admission gates makes the public
  `pdg pipeline` 3.62x pinned PDAL on an option-free uncompressed LAS
  `filters.normal` pipeline (1.218549 versus 4.410862 seconds), byte-exact,
  with the full unit binary and the selection-relevant process gates passing.
  On the six reference workflows the same prototype moves 0.903899x/0.980306x/
  0.970380x/1.004472x/0.917378x/0.999277x to 0.935046x/0.993491x/0.977403x/
  0.993697x/0.926887x/0.996922x — one to three percent, and still no win.
- Explanation: every reference workflow reads and writes LAZ. B0147 established
  that LAZ prevents `lasHeaderFacts` from producing runtime facts, so placement
  is unavailable and a relaxed selection gate has nothing to select.
  Generalized selection unlocks only the uncompressed option-free LAS subset.
- Reordering, which reverses D0211: the LAZ chunk codec moves to the front. It
  is the only item that unblocks the reference workflows at all, and every
  other queued item — generalized selection, writer-option admission,
  `extra_dims` publication — is worth measurably more once it exists and
  worth little before. D0204 ranked native I/O first on extrapolation, D0210
  made that measured, and this entry makes it the immediate next
  implementation rather than a later phase.
- Revised queue: **B0150 the LAZ chunk decoder**, then encode, with
  chunk-parallel decode as the design target and lazperf as the licensed
  implementation route already named in P4; then B0151 generalized selection,
  re-measuring this prototype's 3.62x on workflows that can reach it; then
  writer-option admission; then `extra_dims` publication; then
  `mixed_calibration_models`. The D0207 gather work remains last.
- Prototype disposition: reverted. It does not clear D0208's reference-baseline
  gate, and retaining it would extend automatic selection across a far larger
  surface of graph shapes than the exactness matrices prove. D0106 restricted
  automatic admission to exact named shapes precisely because broad selection
  requires broad proof; discharging that burden is its own slice and should be
  done when it can be measured against workflows that benefit.
- Session note: four attributions were corrected by measurement in sequence —
  B0146 by B0147, B0147 by B0148, and D0211's ordering by this entry. Every
  correction came from measuring the specific mechanism rather than reasoning
  about the code, and three prototypes were built and reverted to produce those
  measurements. That is the intended cost of the measure-first rule, not a
  failure of it.

## D0213 — Exact LAZ decode lands; wiring and parallelism are separate slices

- Date: 2026-08-12.
- Status: accepted an exact, tested capability increment. No placement,
  selection, coverage, or performance claim changes.
- Change: the engine can decode LAZ point records byte-exactly, via
  `decodeCompressedPointRecords` in `src/io/las/LasChunkDecoder.cpp`, reusing
  the already-vendored lazperf. Proved against a paired 1,000,000-point
  LAZ/LAS fixture record by record, with both build configurations clean
  including ASan/UBSan.
- Deliberately not done in this slice, so the boundary is unambiguous:
  `FileView::pointRecord` still throws for compressed files, `lasHeaderFacts`
  still refuses `.laz`, and therefore no reference workflow is accelerated yet.
  Landing the capability and changing admission are separate changes with
  separate risk, and this session has already recorded four attributions
  corrected by measurement — conflating them would be exactly the mistake that
  pattern warns against.
- Sequencing from here: B0151 wires the decoder into `FileView` and
  `lasHeaderFacts` so LAZ inputs produce valid runtime facts, then re-measures
  the reference baseline; B0152 adds LAZ encode for writers; B0153 re-lands
  B0149's generalized selection and re-measures its 3.62x on workflows that can
  finally reach it; then writer-option admission, `extra_dims` publication, and
  `mixed_calibration_models`. The D0207 gather work remains last.
- Performance note recorded up front so it is not discovered as a surprise: the
  current decode is sequential and copies the mapped view once. It is a
  correctness baseline, not a fast path. P4's chunk-parallel target is reachable
  through lazperf's `reader::chunk_decompressor`, and the decision on whether it
  is worth building must come from a measurement of the wired-in path, not from
  the assumption that parallel decode must win.

## D0214 — The admission whitelist is a cost mechanism; order checks by cost

- Date: 2026-08-12.
- Status: accepted a measured negative result and a design correction; reverted
  a prototype; retained a small admission widening.
- Measure: re-landing B0149's generalized automatic selection, with the LAZ gap
  now closed by B0150--B0152, regresses the reference baseline. `r1-translate`
  falls from 0.903899x to 0.660230x and `r4-denoise-thin` from 1.004472x to
  0.928523x. No workflow improves materially.
- Mechanism: `r1` is not accelerated in either configuration — its placement is
  unavailable with `non_cardinality_preserving_stage` because the crop filter
  changes point count. The whitelist rejected it on cheap JSON structure before
  any device work; without the whitelist it first pays runtime placement's CUDA
  device and profile discovery, 0.174804 seconds measured, and then declines
  anyway. On a 0.7-second workflow that is roughly a quarter of the wall.
- Design correction: D0106's narrow automatic admission has been treated in
  this session as conservatism about proof. It is also a cost mechanism. The
  decline path must stay free, because most pipelines decline. Any future
  generalization must order admission checks by cost — every structural test
  that can reject without touching CUDA runs before device and profile
  discovery — rather than removing the early gate.
- Next: **B0154 hoists the cheap placement preconditions ahead of CUDA device
  and profile discovery.** `non_cardinality_preserving_stage`,
  `unsupported_topology`, and the LAS-family/option-free endpoint facts are all
  decidable from the compiled plan alone. Making them reject before device work
  turns declining into a free operation, which helps every pipeline the engine
  cannot accelerate — currently most of them — and is a prerequisite for any
  later selection generalization rather than an alternative to it.
- Retained: `compression:true` on a `.laz` sink is admitted, proved
  byte-identical to the option-free form in B0152. Other compression
  combinations stay unadmitted.
- Session tally, recorded because the pattern is the lesson: five attributions
  were corrected by measurement (B0146 by B0147, B0147 by B0148, D0211 by
  D0212, and D0212's selection direction by this entry), and four prototypes
  were built and reverted. Every correction came from measuring the specific
  mechanism. The recurring failure mode was reasoning from code structure to
  expected behaviour; the recurring fix was a single-variable measurement.

## D0215 — A refusing gate must be able to say why

Six benchmark slices (B0176, B0177, B0182, B0183, B0184, B0185) investigated
why automatic compose envelopes appeared unreachable. The conclusion is that
**every one of the six envelopes was reachable the whole time**, and each
apparent defect was an error of observation:

- B0177 and B0182 hand-transcribed matcher predicates out of
  `RuntimePlacement.cpp` while `test/data/pdg/placement-calibration-sm89.json`
  stored, under `pipelines`, the exact pipeline each of the 35 models was
  calibrated on.
- B0183 missed `always_up=false`, an option whose default is the opposite of
  what the envelope requires.
- B0184 wrote `value` where `filters.radiusassign` takes `update_expression`,
  and a one-element array where an envelope tests `is_string()`.
- B0184 also read `--stats` from `pdg resident` to explain a `pdg pipeline`
  timing. `runResidentPipelineImpl` takes an `automaticAdmission` parameter
  that is false in the first case and true in the second, and two envelopes are
  gated behind facts set only under automatic admission — so the probe disabled
  the route being measured.

The common cause is that a declining gate reported `missing_calibration_model`,
which is a fallback artifact: when a compose matcher refuses, the per-stage
path reports the first stage without a model (`filters.outlier` has none) and
the predicate that actually failed is never named.

**Decision.** A gate that refuses must expose enough state to identify the
refusal without reading its source. `--stats` therefore carries an
unconditional `plan` section with the planner-derived attributes the compose
matchers test. It is emitted even when placement is unavailable, because that
is the case it exists to explain, and it is diagnostics only — it reads the
compiled plan and changes no placement decision.

**Rationale.** Three slices inferred matcher predicates without instrumentation
and two inferred wrong; one of those wrong inferences (B0182's) became a
recorded next task to relax `sameAssignProgram`, which would have widened an
admission envelope to fix a problem it did not have. The cost of the missing
instrument was six slices and a near-miss on the gate's integrity. This does
not relax any gate, and it does not make envelopes easier to hit — reachability
is a separate obligation (D0208) still owed on the reference pipelines.

**Consequences.** Envelope reconstructions must be copied verbatim from the
calibration file, never retyped, because matching is on exact JSON document
text: key counts, scalar-versus-array shape, and non-default option values.
Stats must be read from the invocation actually being measured. New compose
envelopes should record their calibrated pipeline in the calibration file as
the single authority, and any new refusal reason must name a condition an
operator can act on rather than a fallback artifact.

## D0216 — Composition models are fitted for the shapes users write

B0186 measured all six reference pipelines at or below parity while the six
calibrated compose envelopes reach 2.728x–23.785x. The largest reference
pipeline, r6-features, carries 9.1 s of neighborhood work and gets 1.003x. Its
two stages are already planner-assigned to device; D0077's mixed-models rule
declines them because `normal` and `covariancefeatures` carry separate
per-stage models and no composition covers the pair. The eigen family would,
but demands an `eigenvalues` stage, an `assign` with three exact expressions,
`knn=12`, `always_up=false` and `normalize=true`.

**Decision.** When a measured refusal is a missing composition model rather
than a missing capability, the response is to fit the model for the shape real
pipelines have — not to relax the admission rule, and not to require users to
contort a pipeline into a calibration fixture's shape. B0187 fits
`normal-covariancefeatures-compose` over a 50K–4M ladder, byte-exact at every
row, reaching 1.458x–7.260x.

**Rationale.** D0077 exists because separately-fitted models compose unsoundly,
and B0157 recorded what editing a calibration gate to admit a prototype costs.
Fitting a model is the sound way to widen admission: the envelope grows only
over shapes that were measured, and the audit re-verifies every case.

**Consequences.**

- An envelope's conditions must be justified by the model's own terms. `knn` is
  pinned because the model has no neighbor-count term and kNN cost scales with
  it. `always_up` is not pinned, because it normalizes a normal's sign per
  point after the neighborhood work and cannot move the predicted cost.
  Pinning options that do not change cost is the D0215/B0183 defect.
- Envelope floors are measured, not inherited. The sibling neighborhood models
  stop at 250K; this pair wins by 1.458x at 50K and the model predicts it, so
  copying 250K would have discarded a real win.
- Fitted intercepts are constrained non-negative. Free fits here produced
  negative fixed terms, which understate device cost precisely where the
  decision is closest.
- A model may only admit shapes whose cost it measured. This composition
  requires native LAS I/O, so r6-features stays refused while its LAZ reader,
  LAZ writer and `extra_dims` output are non-native: predicting compression
  cost with a curve fitted on uncompressed I/O would be the unsoundness D0077
  prevents. Making that I/O native is the separate, now-quantified obligation.

## D0217 — Nativeness means the records are obtainable exactly, not mappable

`stage.native` gated reader admission on an uncompressed filename, with the
stated reason that nativeness "additionally authorizes memory-mapped record
access, which compressed records cannot serve." B0188 checked that claim
instead of inheriting it. It is false: the only consumer of mapped records is
`DirectResidentPointTable`, which is constructed solely under
`directResidentLasSource`, and that route carries its own `!compressedReader`
guard (B0151/D0213). `FileView::pointRecord` additionally throws on compressed
data, so the failure mode is closed, not silent.

**Decision.** Reader nativeness means the engine can obtain the stage's records
exactly, by mapping or by decoding. Option-free `.las` and `.laz` readers are
both native. Writers stay uncompressed-only.

**Rationale.** The asymmetry is measured, not assumed. Pinned `laz -> las`
translate is *faster* than `las -> las` (0.309 s versus 0.326 s at 1M) because
the smaller file more than pays for decompression, so admitting compressed
input spends no unmeasured time in a model fitted on uncompressed input.
Encoding is different: `las -> laz` costs +0.112 s, a real cost no current
model accounts for, so writers are excluded until a model measures it.

**Consequences.**

- A `.laz` reader now reaches every composition whose other conditions it
  meets; the `normal` + `covariancefeatures` pair measures 4.079x–7.221x
  byte-exact on compressed input, within its own fit error of the uncompressed
  rows.
- The direct LAS source is unchanged and still refuses compressed input. That
  guard is what makes this decision safe, and it must not be folded into
  `native` — the two questions are genuinely separate, which is the whole
  content of this entry.
- A *configured* reader stays non-native regardless of extension, because an
  option such as `count` makes the header-derived placement facts wrong. That
  rule predates this decision and is untouched.
- Admitting `.laz` writers requires first fitting the encode cost, per D0216.

## D0218 — Derived facts beat hardcoded ones, and pdg's own tables beat upstream's private ones

`runtimeFacts->outputRecordBytes` was assigned from one of two compile-time
constants, 36 or 48. B0189 measured an `extra_dims=all` sink emitting 100 bytes
per point, so any pipeline with such a sink would have been placed against a
64 B/point understatement of its terminal spill boundary — roughly 64 MB at 1M
points.

**Decision.** A placement fact that varies with the pipeline is derived from the
pipeline, not selected from constants. `outputRecordBytes` is computed from the
writer's actual layout, and `--stats` reports it alongside `input_record_bytes`
so the derivation is readable rather than inferred from source (D0215).

**Decision.** When the engine needs knowledge upstream also encodes, prefer
pdg's own table if it already exists, and upstream's symbol if it does not.
Never copy upstream's table into pdg. B0190 found `las::pdrfDims` unlinkable
under `-fvisibility=hidden` and escalated a choice between patching upstream's
build surface and compiling its private LAS subsystem into the engine. Both
were unnecessary: `formatCarriesField`, the per-format LAS field predicate
`LasFerry` had used privately since the ferry was written, answers exactly that
question. It was exported rather than reinvented.

**Rationale.** Duplicating upstream's dimension table would let the two drift,
and drift in a record layout is a byte-exactness failure against a pinned
oracle — the one invariant the fork cannot trade. Patching upstream's
visibility to reach a table pdg already had would have been a real cost paid
for nothing.

**Consequences.**

- Before escalating a blocker that needs a design decision from the user, check
  whether the fork already solves it internally. B0190's escalation was
  premature and is withdrawn in B0191.
- Admitting an `extra_dims=all` sink is a **separate** obligation and is not
  granted by this entry. It was measured at 6.227x byte-exact with
  `output_record_bytes` correctly derived as 100, and reverted because
  `pdg_resident_pipeline_process` and `pdg_resident_sort_direct_gpu` fail with
  `direct resident LAS boundary has no logical transfer`: a native writer
  declares a fusion anchor whose pack/summarize machinery changes the plan's
  residency boundaries, and the direct routes' boundary facts do not survive
  it. Per B0157 the change goes back rather than the gate being edited.
- A derived fact must be verified against a measurement, not an expectation.
  The 100 was confirmed against the file B0189 measured, not against a
  recomputation of the same rule.

## D0219 — Do not load the engine for work the engine will refuse

`classifyPipelineForDispatch` sent any pipeline with a LAS reader and a LAS
writer to the engine. B0196 measured what that costs when the engine cannot
help: the automatic resident attempt declines in 0.17 ms, `tryNativePipeline`
requires three stages, and `main` falls through to `runOracle`, which `execv`s
pinned PDAL — so the process loads an 11 MB image and 76 shared libraries,
including `libcudart` and `libcuda` that PDAL does not link, purely to hand the
work back. A measured 15–19 ms, fixed, on every such pipeline.

**Decision.** A pipeline shape that no engine route can accelerate is routed to
the oracle by the dispatcher, before the engine image is loaded. The first such
shape is a bare LAS reader-to-writer pipeline with no filter between them.

**Rationale.** The cost is not the admission decision, which is free; it is
loading a CUDA-linked binary to reach that decision. The saving is measured at
19.34 ms → 0.96 ms on a 10,000-point translate, byte-exact.

**Consequences.**

- The rule is a *shape* whitelist in reverse and must stay narrow. Anything
  with a filter keeps the engine route, because a filter is what every
  accelerated shape has. B0195 records what a blanket early exit would forfeit:
  B0188's LAZ features route reaches the resident executor at 6.351x through
  the path this decision leaves alone.
- Adding a shape to this list requires measuring that no engine route
  accelerates it. B0196's premise that the engine won the cost back at 4M was
  refuted by re-measuring; a single delta inside run-to-run variance is not
  evidence either way.
- This changes no output. The oracle is pinned PDAL, which is what the engine
  was going to exec regardless, so byte-exactness is unaffected by
  construction.
- It does not help the reference pipelines. None of r1–r6 is a bare translate,
  so D0208's criterion is untouched by this entry.

## D0220 — Hoist a plan-decidable refusal; defer coarsely where facts decide

D0214 established that a pipeline which can never be placed should decline
before CUDA device and profile discovery, and hoisted `planStructureRefusal` to
do it. That hoist judged topology and cardinality only. B0205 measured what it
missed: `filters.smrf` is native and device-preferred, so it passes every
structural check, but carries no placement model — and the resulting refusal
happened inside `buildRuntimePlacement`, after ~170 ms of discovery that the
compiled plan had already made unnecessary.

**Decision.** Any refusal decidable from the compiled plan alone is taken in
`planStructureRefusal`, before device discovery. This now includes a plan with
no device-capable stage (`NoDeviceCandidate`) and a plan whose device stage has
no calibration model (`MissingCalibrationModel`).

**Decision.** Where `buildRuntimePlacement` excepts a case using executor facts
that the plan-only path does not have, the plan-only path **defers on the whole
class** rather than approximating the exception. Concretely: a plan containing
`filters.outlier`, `filters.nndistance` or `filters.radialdensity` skips the
model-existence refusal entirely, because the two direct composition regions
are built from those stages and their admission is facts-gated.

**Rationale.** B0205 tried the approximation — the plan-only region lookups —
and it broke `pdg_resident_outlier_direct_gpu` and
`pdg_resident_radialdensity_direct_gpu`, because a drifted shape that misses
the plan-only matcher then receives a different and less accurate reason than
the facts-gated path would give it. A coarse guard costs a little discovery on
plans containing those three stages and is never wrong; an approximate one is
cheaper and is sometimes wrong, which D0215 already establishes is the worse
trade for a refusal.

**Consequences.**

- Returning ~246 ms to LAS-to-LAS SMRF pipelines, byte-exact, with 775/775 and
  486/486 tests passing.
- No reference pipeline moves. r2 and r3 contain SMRF but delegate to the
  oracle before reaching this path, so D0208's criterion is untouched.
- The refusal reason a caller sees must stay the one `buildRuntimePlacement`
  would have produced. Both hoisted reasons satisfy that; a future hoist must
  demonstrate it rather than assume it.
- When a new stage type joins a facts-gated composition region, it must be
  added to the deferral list, or the hoist will refuse a shape that runtime
  placement would have admitted.

## D0221 — The unattended-session review fails closed and reverts an unproved selector

A post-session audit of B0183--B0222 found two changes that did not satisfy the
default exactness and evidence gates. First, B0214--B0216 proved that the SMRF
device fill selects a different eighth neighbor from the pinned oracle when a
cutoff-distance tie is contested, yet the bounded device lane remained
selectable on smaller rasters. The existing CUDA matrix did not contain that
counterexample. Second, B0222 called a region "filter-only" whenever it had a
value predicate and lacked `assign`/`ferry`; the eligible region grammar also
contains transformations and ordinal stages, so the gate changed unmeasured
predicate-plus-transformation and predicate-plus-ordinal default paths. B0222
added no selection or process regression for those shapes.

**Decision.** The SMRF CUDA implementation remains compiled for repair work,
but `smrfSupportsExactDevice` returns false for every input. Required resident
or CUDA execution therefore fails closed; ordinary execution uses the exact
KD2Index host wrapper or the pinned fallback. The CUDA process matrix is no
longer registered as an exact gate. Requalification requires an
oracle-equivalent device-path void fill and a committed differential that
exercises a contested eighth-neighbor tie.

**Decision.** B0222's threshold and `filterOnlyRegion` selection fact are
reverted exactly. The B0219--B0222 measurements remain append-only historical
evidence, but they grant no default-selection envelope. Any successor must
recognize the complete region grammar, test every admitted stage composition,
and establish a measured threshold outside uncertainty before changing the
default path.

**Decision.** The reference benchmark's `exact_outputs` verdict covers output
artifacts, stdout, and stderr, not artifacts alone. Per-run diagnostic hashes
were already captured; the comparison now uses them. Exit status remains part
of the verdict through the existing failure check.

**Consequences.**

- B0216's host-side KD2Index repair is retained and the committed host process
  matrix includes the measured `cell=1/2/4/8` ladder. This is functional host
  work, not GPU-native coverage.
- B0191's derived `outputRecordBytes` fact is retained, but comments now match
  D0218: a general `extra_dims=all` writer was tried and reverted, and only
  separately proved explicit routes may consume the file fact.
- The remaining B0183--B0220 planner, reader, dispatch, refusal, diagnostic,
  and measurement-harness changes remain subject to their narrow tests and
  claims; this review does not promote them beyond the recorded envelopes.

## D0222 — Publish reorder products explicitly and admit the proved uncompressed `extra_dims=all` sink

D0218 derived `outputRecordBytes` from the prepared writer layout and B0191
measured the 100-byte feature record correctly, but general writer admission
was reverted because a terminal `filters.sort` region appeared to have a
zero-byte spill. B0192/B0193 established why: sort writes no point column. It
publishes an output-position-to-input-position permutation, and the direct LAS
publisher combines that permutation with the mapped source record. Treating
the empty column set as the entire publication contract discarded that
separately declared product.

**Decision.** An uncompressed `writers.las` is native with
`extra_dims=all` only when the literal value is `all` and the object contains
no options beyond `type`, `filename`, `tag`, `inputs`, and `extra_dims`.
Runtime placement may consume D0218's prepared-writer layout for this exact
shape. Compressed LAZ, a named extra-dimension list, and every additional
writer option remain non-native.

**Decision.** Functional native admission is not a performance qualification.
The existing normal/covariancefeatures composition model may select this wider
sink automatically only for the measured compressed format-7 source at exactly
1,000,000 points with a 36-byte input record and 100-byte output record. Other
cardinalities, point formats, compression states, and carried source Extra
Bytes decline the mixed calibration and retain the exact host path. The public
proof switch is `PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT=1`.

**Decision.** When a direct resident LAS terminal boundary follows the exact
ordering or skewness producer, its declared result is the logical publication
even if the column spill is empty. The product must be one 64-bit permutation
entry per point, declare no fixed-size bytes, and mark the producer
non-order-preserving; otherwise the route fails closed. The existing strict
direct-source/layout/publication gates remain authoritative, and this rule does
not authorize raw access to a compressed source or sink.

**Rationale.** This states what the engine already publishes instead of
inventing a synthetic point column or weakening the zero-transfer refusal.
The mapped source retains every untouched standard and extra field, while the
permutation supplies the only missing information: output order. Ordinary
column-producing stages continue to use the planner's declared spill columns.

**Consequences.**

- The public 1M normal/covariancefeatures composition with an uncompressed
  `extra_dims=all` sink is byte-, stdout-, stderr-, order-, and status-exact at
  6.267794x median pinned PDAL, and reports the actual 100-byte output stride.
- The physical selection matrix covers that sink directly, while the strict
  direct sort and skewness process gates prove their permutation-only paths did
  not regress. A carried-source-Extra-Bytes case proves byte-exact host
  selection outside the measured layout. An injected post-placement resident
  preflight refusal makes the required public route exit 124 without output,
  proving the switch cannot be satisfied by placement alone. The maintained
  Host Debug preset is 403/403 and the physical CUDA Release aggregate is
  776/776, with their
  documented optional and calibration-identity skips; focused Host ASan/UBSan
  is clean under the maintained `detect_leaks=0` setting.
- This narrowly supersedes D0221's continued refusal for the now-proved
  uncompressed `all` envelope. It does not unlock the r2 or r6 reference
  writers: both are compressed LAZ, and r2 also uses a named extra dimension.
- No CUDA kernel, allocation, index, or arithmetic contract changes. B0223's
  exact public-shape profile supersedes B0071 as the endpoint evidence: 20
  launches total 24.495008 ms/1.821009% of candidate wall, and the
  L2/memory-limited `knnGather` accounts for 87.607467% of kernel time. B0187's
  broader plain-writer ladder does not qualify other `extra_dims=all` layouts.

## D0223 — Admit only the measured exact LAZ `extra_dims=all` publication row

D0222 proved that the prepared writer layout and explicit publication product
are sufficient for an exact uncompressed `extra_dims=all` sink. B0152 had
already proved `.laz` and redundant `compression=true` spellings equivalent,
while B0188 showed that LAZ encode adds material endpoint cost. Functional
equivalence therefore permits a native writer, but performance selection still
requires evidence for the compressed endpoint and cannot inherit B0187's
plain-writer ladder.

**Decision.** A `.laz` `writers.las` with literal `extra_dims=all` is native
only when its object contains no non-routing writer option except omitted
compression or compression true as either a boolean or case-insensitive
string. `compression=false`, named extra dimensions, and any additional writer
option remain non-native. Raw direct LAS source/output paths remain forbidden
for compressed records.

**Decision.** Runtime facts carry input and output compression independently.
The normal/covariance composition may automatically select compressed
`extra_dims=all` publication only for B0224's measured row: 1,000,000 points,
point format 7, compressed input, 36-byte input records, and 100-byte compressed
output records. Every other compressed `extra_dims=all` graph fails the mixed
calibration closed. In particular, eigen-family, rank/optimal, and ordinary
per-stage models may not inherit the new native writer merely because their
older stage arithmetic is calibrated. D0222's separately proved uncompressed
and direct-output envelopes are unchanged.

**Decision.** `PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT=1` is a public
route proof, not a placement-only assertion. It requires the measured model,
one selected region, an executable rewrite, accepted resident preflight, and
the resident scope; any later rewrite or preflight decline returns to the
dispatcher before output and makes the public command exit 124.

**Rationale.** The full checked-in r6 reference is the relevant product gate.
Five alternating same-machine pairs measure pinned PDAL at 9.165000 seconds
median and `pdg` at 2.023623 seconds, or 4.529005x, while preserving the exact
61,789,134-byte LAZ, stdout, stderr, status, and order. This qualifies one
compressed publication row without pretending that encode cost is independent
of layout, count, or an unrelated composition model.

**Consequences.**

- r6 is now automatically accelerated under the exact measured facts. r2
  remains delegated because `HeightAboveGround=float32` is a named layout.
- The twenty-case physical matrix covers string, boolean, and implicit
  compression spellings through the public dispatcher and separately verifies
  the resident executor and emitted stride. It also proves wider carried Extra
  Bytes remain host-selected and that an injected post-placement preflight
  decline creates no output. Runtime-placement units supply the distinct
  eigen/rank negative controls.
- Maintained Host Debug and ASan/UBSan presets pass 403/403; the physical CUDA
  Release aggregate exits cleanly across 776 registrations. The documented
  unrelated unfiltered Host HAG-Delaunay prefix defect remains 487/488.
- No CUDA translation unit changes. B0224's exact full profile records 20
  launches/24.428000 ms, 1.207142% of candidate wall; the L2/memory-limited
  `knnGather` remains 87.522253% of kernel time. No kernel or Compute Sanitizer
  claim follows from this host/planner/publication slice.

## D0224 — Do not admit r2's named sink without a profitable SMRF/HAG route

B0224 leaves one tempting sink-coverage item: r2 writes compressed LAZ with
`HeightAboveGround=float32`. Treating it as another writer spelling would be
mechanically possible, but D0208 makes wall-clock speed—not native-stage
count—the acceptance criterion.

**Decision.** Retain exact pinned-host delegation for r2's named writer. Do not
make it native, add a writer-only calibration, or present it as an acceleration
until a complete exact SMRF/HAG candidate wins on the checked-in r2 reference.
At that point the named float32 publication layout must be proved and measured
as part of the same vertical slice.

**Rationale.** B0225's final-binary five-pair public benchmark is exact but
resolvably slower: 1.558436 seconds pinned PDAL versus 1.577263 seconds `pdg`,
or 0.988063x median and 0.986717x +/- 0.004381 paired, with 0/5 candidate wins.
Resident stats show the writer is not the only blocker: SMRF and HAG-NN occupy
two separate device-preferred regions and both have empty placement models.
Writer admission would expose, not solve, those uncalibrated regions.

B0217/B0218 already measured the plausible substitution and rejected it. The
fork's exact SMRF+HAG path is 350.7 ms +/- 38.5 slower on r2's shape; its host
HAG-NN alone is about 36% slower than upstream, while its faster CUDA route
requires planner facts no automatic model supplies. D0221 also forbids device
SMRF because its void-fill tie behavior is known inexact. Repeating those
facts behind a newly native writer would widen coverage without increasing
speed.

**Consequences.** r2 remains functionally supported and byte/diagnostic/order/
status exact through the pinned host. The next r2 work is a measured complete
stage-route improvement—host threading/algorithm work or a newly calibrated
exact resident composition—not named-writer admission in isolation. No product
code, test count, CUDA, or sanitizer claim changes in B0225.

## D0225 — Direct-delegate only the header-calibrated r1 reference route

The r1 reference contains one engine candidate, `filters.crop`, followed by
`filters.reprojection`, which the engine cannot execute. B0196 attributed the
fixed refusal cost to loading the CUDA-linked engine image, B0199 measured the
hybrid crop substitution losing on r1, and B0226 reproduces the complete public
deficit at 0.938267x +/- 0.007951. The engine ultimately execs the same pinned
oracle after paying that cost.

**Decision.** The thin dispatcher directly execs the pinned oracle for the
literal measured r1 route. This is not a general reprojection rule. Admission
requires the exact root/stage/option grammar, materialized crop bounds,
`EPSG:28992` -> `EPSG:3857`, lowercase LAZ endpoints, string
`compression="true"`, and fixed-header input facts matching the measured 1M
compressed LAS 1.4 format-7/36-byte reference layout, byte size, coordinate
encoding, and extent. The gate is header-calibrated, not a claim of input
content identity. Missing or drifted facts remain in-engine.

**Decision.** Engine-affecting PDG environment variables and every pipeline
CLI modifier continue to preempt this route. `PDG_ORACLE_PDAL` selects the same
direct oracle and does not force the engine. Direct exec must preserve argv,
stdout, stderr, and exit status exactly.

**Rationale.** The accepted direct route is at parity with pinned PDAL rather
than claimed faster: 1.002029x +/- 0.009677 spans parity. The conservative
paired control against `pdg-engine` is resolvably 1.075614x +/- 0.011503 faster
in 9/9 pairs, proving the removed engine path is the gain. The output artifact
and diagnostics are exact by construction and measurement; the report also
records the materialized pipeline hash consumed by the matcher.

**Consequences.** r1 remains host/oracle functional coverage and gains no
GPU-native stage. The rule does not generalize to other bounds, counts,
layouts, SRS pairs, option spellings, CLI modes, or reprojection pipelines.
Unit and process tests pin those refusals and the configured-oracle diagnostic
boundary. Expanding the rule requires a separate same-machine complete-process
envelope; an exact profitable native reprojection may supersede it later.

## D0226 — Admit only the measured r4 hybrid-outlier reference route

B0164 measured statistical outlier at about 80.5% of r4 wall while sample was
about 9.8%. B0167 already supplied an exact fast hybrid outlier executor, but
no automatic calibration selected it, and the unsupported sample stage caused
the engine to delegate the original pipeline. B0227's forced public probe
shows that leaving sample upstream while replacing outlier is sufficient:
3.777769x +/- 0.127607 paired with exact output in 3/3 pairs.

**Decision.** Automatically request exact CUDA outlier only for the literal
checked-in r4 grammar and measured 1M input facts. Admission requires the exact
root and five-stage order, statistical `mean_k=8`/`multiplier=2.0`, literal
`Classification![7:7]`, sample radius 1, string `compression="true"`, lowercase
LAZ endpoints, and the measured compressed LAS 1.4 format-7/36-byte file size,
selected header summary, coordinate encoding, and extent. A streaming full-file
FNV-1a-64 fingerprint additionally restricts selection to the measured
compressed-byte fingerprint; the tested payload-byte mutation fails closed.
This is a calibration/drift key, not a cryptographic integrity claim. Pipeline
CLI modifiers, experimental substitution, missing facts, and every neighboring
shape also fail closed.

**Decision.** Performance qualification is device-specific: automatic
selection additionally requires RTX 4090/SM89, CUDA toolkit 13.3, and NVIDIA
driver 610.43.03. The hidden automatic marker may select CUDA only on that
profile. `PDG_REQUIRE_AUTOMATIC_R4_OUTLIER_CUDA=1` is a proof control: the
public invocation must have the selected final rewrite and the plugin must
report actual CUDA use, so a declined rewrite or recoverable device fallback
is a failure rather than a false pass.

**Rationale.** The final fingerprinted nine-pair public result is 3.686747x
median and 3.715180x +/- 0.055459 paired, with 9/9 wins and exact artifact,
diagnostics, status, order, and metadata-sensitive bytes. A phase profile has zero
incomplete rows and negligible exact repair; it does not inherit the known 4M
repair cliff. Restricting the rule to its measured shape and machine honors
the speed-first and evidence requirements without waiting for a native sample
implementation that contributes little to this route's wall time.

**Consequences.** `filters.outlier` gains automatic performance-qualified CUDA
selection only inside this r4 envelope. The adjacent range uses the existing
exact point program; `filters.sample` and LAZ I/O remain upstream, so this does
not add native sample coverage or claim generalized outlier placement. New
input bytes, counts, layouts, option forms, devices, toolkits, or drivers
require a separate complete-process calibration and exactness result.

## D0227 — Direct-delegate only the literal measured r5 reference grammar

The r5 reference enters the engine because `filters.stats` is a candidate, but
the engine cannot consume `readers.copc`; it ultimately execs the unchanged
pipeline in pinned PDAL. B0202 attributed the complete deficit to the unused
engine image, and B0228 reproduces it at 0.930792x +/- 0.011642 paired.

**Decision.** The thin dispatcher directly execs pinned PDAL only for the
literal measured r5 grammar: a single-key object root; lowercase `.copc.laz`
reader; exact materialized bounds; floating `resolution=1.0`; integer
`requests=1`; option-free `filters.stats`; and lowercase uncompressed `.las`
writer. Stage order and option sets are exact. Any neighboring grammar remains
in-engine.

**Decision.** Every pipeline CLI modifier and engine-affecting PDG environment
preempts this route. No input header or content identity gate is added. Direct
and engine-delegated execution both run the identical pinned-oracle pipeline;
the removed engine load is fixed process work, not a data-dependent algorithm
whose performance or exactness could change with COPC contents.

**Rationale.** The accepted public route is at parity with pinned PDAL:
1.001303x +/- 0.003425 paired spans parity. A direct-versus-`pdg-engine`
control is resolvably 1.047266x +/- 0.016446 faster in 9/9 pairs, proving the
removed image load is the gain. Every measured artifact, diagnostic, status,
and order is exact.

**Consequences.** r5 remains host/oracle functional coverage; this adds no
native COPC, stats, or LAS stage. Bounds, numeric representation, options,
root, stage order, endpoint spelling, CLI, and environment drift all fail
closed. A future native COPC pushdown must beat this direct-host baseline end
to end and preserve the compatibility contract before it can supersede the
route.

## D0228 — Direct-delegate only the literal measured r3 DTM grammar

The r3 reference enters the engine because `filters.smrf` and `filters.range`
are candidates, but exact automatic SMRF replacement is disabled under D0221.
The unchanged SMRF stage makes the adjacent range rewrite order-unstable, and
the GDAL sink is outside every native/resident LAS endpoint. The engine
therefore discards its admission work and execs the unchanged pipeline in
pinned PDAL. B0229 reproduces the public deficit at 0.966556x +/- 0.006223
paired.

**Decision.** The thin dispatcher directly execs pinned PDAL only for the
literal measured r3 grammar: a single-key object root; lowercase non-COPC
`.laz` reader; option-free SMRF; exact `Classification[2:2]` range; and a
lowercase `.tif` GDAL writer with floating `resolution=1.0` and
`output_type=idw`. Stage order and option sets are exact. Any neighboring
grammar remains in-engine.

**Decision.** Every pipeline CLI modifier and engine-affecting PDG environment
preempts this route. No input header or content identity gate is added. Direct
and engine-delegated execution both run the identical pinned-oracle pipeline;
the removed engine work is fixed process/control-plane work, not a
data-dependent algorithm whose performance or exactness could change with
input contents.

**Rationale.** The accepted public route is at parity with pinned PDAL:
0.996610x +/- 0.009240 paired spans parity. A direct-versus-`pdg-engine`
control is resolvably 1.027780x +/- 0.005196 faster in 9/9 pairs, proving the
removed engine path is the gain. A hashed `/bin/true`-oracle trace records
plan-structure refusal before the unchanged oracle exec. Every measured
artifact, diagnostic, status, and order is exact.

**Consequences.** r3 remains host/oracle functional coverage; this adds no
native SMRF, range, or GDAL stage. Root, numeric representation, options,
stage order, endpoint spelling, CLI, and environment drift all fail closed.
A future native SMRF/GDAL route or host-side stage optimization must beat this
direct-host baseline end to end and preserve the compatibility contract before
it can supersede the route.

## D0229 — Direct-delegate only the literal measured r2 reference grammar

The r2 reference enters the engine because SMRF and HAG-NN are candidates, but
its literal named `HeightAboveGround=float32` writer is outside native
publication. Plan-structure admission refuses before placement or point work,
then the engine execs the unchanged pipeline in pinned PDAL. B0230 reproduces
the public deficit at 0.973432x +/- 0.011569 paired, while direct engine versus
pinned PDAL spans parity.

**Decision.** The thin dispatcher directly execs pinned PDAL only for the
literal measured r2 grammar: a single-key object root; lowercase non-COPC
`.laz` reader; option-free SMRF; option-free HAG-NN; and a lowercase non-COPC
`.laz` writer with string `compression="true"` and exact
`extra_dims="HeightAboveGround=float32"`. Stage order and option sets are
exact. Any neighboring grammar remains in-engine.

**Decision.** Every pipeline CLI modifier and engine-affecting PDG environment
preempts this route. No input header or content identity gate is added. Direct
and engine-delegated execution both run the identical pinned-oracle pipeline;
the removed engine work is fixed process/control-plane work, not a
data-dependent algorithm whose performance or exactness could change with
input contents.

**Rationale.** The accepted public route is at parity with pinned PDAL:
1.008753x +/- 0.010559 paired spans parity. A direct-versus-`pdg-engine`
control is resolvably 1.014563x +/- 0.005922 faster in 9/9 pairs, proving the
removed process path is the gain. A hashed `/bin/true`-oracle trace records
plan-structure refusal before unchanged oracle execution. Every measured
artifact, diagnostic, status, and order is exact.

**Consequences.** D0224 remains in force: this does not admit the named writer,
fit a writer model, or add native SMRF/HAG execution. r2 remains host/oracle
functional coverage. Root, option type/value, stage order, endpoint spelling,
COPC suffix, CLI, and environment drift all fail closed. A future native or
host-optimized SMRF/HAG route must beat this direct-host baseline end to end
and preserve the compatibility contract before it can supersede the route.

## D0230 — Automatically select only the measured direct neighbor-classifier envelope

B0098 proved an exact mapped-LAS-source/direct-Classification
`filters.neighborclassifier(k=7)` executor, but deliberately left it opt-in
without a placement model. B0231 repeats the exact route across 50K--16M
points. It loses at 50K (0.684973x pinned PDAL), wins from 250K through 16M,
and shows that the direct boundary itself is resolvably faster than the former
resident boundary at 1M (1.150355x +/- 0.024677, 9/9 pairs).

**Decision.** Add a separate `neighborclassifier-direct-compose` placement
model rather than widening or borrowing the ordinary `neighborclassifier`
model. Fit the host curve through the origin over the positive 250K--16M rows.
Fit a nonnegative device residual after the planner-owned 239.214 ms startup,
two synchronizations, 25-byte/point upload, one-byte/point spill, zero packing,
and 112-byte/point shared-index terms. Use a hard 250K floor and 16M cap. The
conservative model overpredicts the 250K device row by 15.79% and cannot invert
any measured winner.

**Decision.** Automatic admission requires the literal three-stage
`readers.las -> filters.neighborclassifier(k=7) -> writers.las` grammar,
uncompressed format-7 36-byte input and output records, the mapped LAS source,
direct Classification publication, one selected region and lane, the measured
112-byte/point index build, and the exact SM89 placement profile. Compression,
other layouts, options, topology, `k`, cardinality outside the ladder, index
facts, profile, preflight, or post-execution proof drift fails closed before
publication. A required-route proof checks the final executable rewrite and
actual execution evidence, not merely an earlier placement prediction.

**Rationale.** The final ordinary public 1M process is byte-, stdout-, stderr-,
status-, and order-exact at 4.033570x median pinned PDAL (3.619967 seconds
versus 0.897460). The route reuses the planner-owned spatial index and the
existing exact tie/incomplete repair; no private index or relaxed arithmetic is
introduced. The measured 50K loss and layout-sensitive boundary costs make a
bounded direct-composition model necessary under the speed-first policy.

**Consequences.** `filters.neighborclassifier` becomes automatically selected
only inside this measured direct envelope. This does not generalize the
ordinary stage model, remove exact KD3 repair, qualify compressed or wider
records, or claim automatic support on another device/toolkit/driver. Any
expansion requires a new same-machine complete-process ladder, exact public
selection/execution proof, and updated calibration record.

## D0231 — Automatically select only the measured direct Z-sort envelope

B0130 rejected automatic selection of the ordinary standalone sort boundary:
at 4M the exact host wrapper was faster than CUDA, and at 21.97M the device
edge was only about 0.7%. B0131 nevertheless proved a much faster strict
composition that maps the LAS source, uploads one binary64 Z key, downloads
one source permutation, and atomically republishes complete source records.
B0232 repeats that complete direct route from 50K through 16M and measures its
public-boundary attribution before changing selection.

**Decision.** Add a separate `sort-direct-compose` placement model; do not
widen or borrow the ordinary `sort` model. Fit the accepted host curve through
the origin at 793.782365 ns/point and use the maximum nonnegative measured
device residual, 58.74206627340533 ns/point, after existing startup, transfer,
and synchronization terms. Charge exactly one 8-byte/point upload and one
8-byte/point download, with zero packing and indexes. Use a hard 600K floor and
16M cap. Although the explicit device route is positive at 500K, its 500K and
550K direct-versus-public intervals span parity; 600K is the first resolvable
public-boundary win.

**Decision.** Automatic admission requires the literal three-stage
option-free `readers.las -> filters.sort(Z,ASC,NORMAL) ->
writers.las(extra_dims=all)` grammar, uncompressed LAS 1.4 format-7 36-byte
input and output records, one whole-view region/lane, finite comparator-unique
binary64 Z, the measured cardinality envelope, exact direct boundary facts,
and the pinned SM89 profile. Any topology, option, compression, layout,
cardinality, profile, memory, preflight, source, or execution-proof drift
returns to the unchanged host pipeline before output side effects.

**Decision.** Reserve 64 CUDA bytes per point for this executor. Requested
allocation high-water is 33,769,475 bytes at 600K and 900,275,715 at 16M,
including the planned key, caller and alternate permutations, two materialized
keys, CUB temporary storage, and duplicate flag. The former 24-byte/point
estimate described only persistent/simplified columns and is not a valid peak.
Placement and resident preflight must agree on the 64-byte reservation, and
the required-route process gate freezes acceptance at `64*N` and refusal at
`64*N - 1`.

**Decision.** Data-dependent exactness and side-effect-free publication
refusals remain precommit. Duplicate, signed-zero-equivalent, or non-finite
keys discard the device result and execute the original pipeline. Existing,
aliased, symlink, or otherwise rejected destinations also fall back while the
atomic publisher has made no output change. Commitment occurs only after
successful publication; later failures remain errors and are never retried.

**Rationale.** Final ordinary public runs are byte-, metadata-, order-,
stdout-, stderr-, and status-exact. Nine alternating pairs measure 1.842475x
median pinned PDAL at the 600K floor and 3.267173x at 1M, with all candidate
samples faster. The retained B0131 profile puts all kernels at only 0.080351%
of direct wall, so selection and boundary removal—not a new kernel—are the
measured win.

**Consequences.** `filters.sort` is automatically selected only inside this
strict direct envelope. Ordinary sort, stable/descending/multi-key forms,
ties, other layouts or cardinalities, compressed I/O, and other hardware
profiles remain exact and host-selected. Expansion requires a new exact
complete-process ladder, public attribution, allocation high-water proof, and
updated placement/calibration evidence.

## D0232 — Automatically select the measured label/NNDistance hybrid, not the resident prototype

B0020 showed that standalone `filters.label_duplicates` CUDA loses end to end.
B0116 later found a useful exact composition:
`label_duplicates(Classification) -> nndistance(k=10) ->
assign(UserData = Duplicate)`. A current 50K--16M ladder reproduces the forced
hybrid result: 50K loses at 0.893x, while 250K through 16M win at
3.261x--12.190x. A current disposable resident probe is exact but remains 4.3%
slower than the hybrid at 1M, so residency is not the speed-first answer for
this shape.

**Decision.** Add a separate complete-process hybrid selector rather than
widening either ordinary stage placement model or reviving B0117's rejected
resident model. Fit independent host and device affine curves over the exact
positive 250K, 1M, 2M, 4M, 8M, and 16M rows. Use a hard 250K floor and 16M cap;
the measured 50K loss remains host-selected. The selector chooses only when
the fitted device curve beats the fitted host curve inside that measured
envelope.

**Decision.** Automatic admission requires a single-key object root and the
literal five-stage grammar: option-free lowercase `.las` reader;
`filters.label_duplicates(dimensions=Classification)`;
`filters.nndistance(k=10)`; exact string
`filters.assign(value="UserData = Duplicate")`; and option-free lowercase
`.las` writer. Input facts must be uncompressed LAS 1.4 format 7 with a
375-byte point offset, 36-byte records, zero VLRs and EVLRs, no trailing
bytes, and a count in the measured envelope. The device promise is restricted
to the exact RTX 4090/SM89/CUDA 13.3/driver 610.43.03 profile.

**Decision.** The automatic route marks the label, neighborhood, and point-
program bridge separately and requires actual CUDA use. Selection, input, or
device-profile refusal delegates the unchanged public pipeline. A recoverable
label-device failure executes the exact host label operation before any writer
work; the proof control rejects that fallback instead of reporting a false
CUDA success. Filesystem behavior remains owned by the ordinary PDAL writer;
existing, input/output-aliased, and symlink destinations must match the pinned
oracle in status, streams, and final state. The resident automatic selector
declines this literal graph before device/profile probing so it cannot preempt
or tax the measured winner.

**Rationale.** Final option-free public runs are byte-, metadata-, order-,
stdout-, stderr-, and status-exact. Nine alternating pairs measure 2.846286x
median pinned PDAL at the 250K floor and 6.391572x at 1M, with all candidate
samples faster and paired intervals clear of parity. This promotes the fastest
measured exact executor while preserving the standalone negative and rejected
resident conclusions.

**Consequences.** `filters.label_duplicates` gains automatic selection only as
part of this strict five-stage hybrid composition. This does not qualify the
standalone boundary, generalize NNDistance options, add a resident model, or
admit compressed, wider, VLR-bearing, differently spelled, or differently
ordered pipelines. Any expansion requires another exact complete-process
ladder and public selection/execution proof on the target profile.

## D0233 — Automatically select only the measured direct skewness composition

B0128 established an exact CUDA ordering substep for
`filters.skewnessbalancing`, but its forced hybrid complete process was only
1.192843x pinned PDAL at 1M. B0129 then proved a faster mapped-source/direct-
permutation composition at 2.887507x without changing the pinned sequential
moment/sign-crossing recurrence. D0190 deliberately kept that route explicit:
it had no separate placement model, public automatic measurement, or
cardinality ladder capable of excluding losses.

A current exact direct ladder closes that evidence gap. The route loses from
50K through 300K, reaches only 1.059448x at 350K, and wins from 400K through
16M. A separate ordinary public nine-pair 400K run is only 1.065118x, below
the predeclared 10% acceptance margin. The first accepted public size is 450K
at 1.200177x; 1M reaches 3.091007x. The accepted direct ladder continues
monotonically through the measured 16M cap.

**Decision.** Add a separate `skewness-direct-compose` placement model rather
than widening the ordinary skewness stage model. Its conservative terms are a
zero-intercept 953.993317777778 ns/point host slope and a
290.879214148791 ns/point device slope after the existing direct-boundary cold
start, one 8-byte/point upload, one 8-byte/point download, and two
synchronizations. Evaluate it only inside the hard 450K--16M evidence
envelope. The 400K public loss-margin control and every smaller row remain
host-selected.

**Decision.** Automatic admission requires a single-key object root and the
literal three-stage grammar: an option-free lowercase `.las` reader; an
option-free `filters.skewnessbalancing`; and a lowercase `.las` writer whose
only additional option is `extra_dims=all`. Runtime facts must prove an
uncompressed LAS 1.4 format-7 mapped source, 36-byte input and output records,
no carried source extra dimensions, finite comparator-unique physical Z, and
a point count in the measured envelope. Selection is restricted to the exact
RTX 4090/SM89/CUDA 13.3/driver 610.43.03 profile.

**Decision.** Placement and resident preflight both reserve 65 bytes/point:
56 bytes/point for the established direct double-key ordering high-water plus
the concurrently retained binary64 Z and byte Classification columns. The
required-route process gate freezes acceptance at exactly `65*N` bytes and
refusal at `65*N - 1`; persistent-column size alone is not a valid peak.

**Decision.** All data-dependent and filesystem refusals remain precommit.
Comparator ties, signed-zero-equivalent keys, non-finite Z, preflight or
execution-proof failure, and existing, aliased, symlink, or otherwise rejected
destinations discard the candidate and run the unchanged public pipeline.
Commitment occurs only after successful atomic publication. A selected route
must prove successful rewrite and actual CUDA ordering use; selection alone is
not sufficient evidence.

**Rationale.** Final ordinary public runs are byte-, metadata-, point-order-,
stdout-, stderr-, and status-exact. Nine alternating pairs measure 1.200177x
median pinned PDAL at the 450K floor and 3.091007x at 1M, with all 18
candidate samples faster. The existing B0129 profile records only 0.260192 ms
across 13 launches, so the accepted win comes from the already-proved direct
boundary rather than a new kernel claim.

**Consequences.** `filters.skewnessbalancing` is automatically selected only
inside this strict direct envelope. Its recurrence remains host code; the
ordinary stage boundary, other options, layouts, compression, cardinalities,
data, devices, and profiles remain exact and host-selected. Any expansion
requires a new exact complete-process ladder, public attribution, allocation
proof, and updated calibration evidence.

## D0234 — Automatically select only the measured count-one HAG-NN direct composition

D0094 established an exact data-dependent `filters.hag_nn,count=1` lane on the
planner-owned masked 2D index. B0123 later qualified the named explicit
forced-hybrid and mapped-source/direct-double-output routes at 1.442969x and
2.518270x pinned PDAL, but D0184 kept them explicit because there was no
cardinality ladder, distinct placement model, or automatic public proof.

A current exact direct ladder closes that evidence gap. It loses at 50,001,
100,002, 250,002, and 350,001 points; 300,000 is effectively parity at
1.006644x. The 400,002-point row reaches only 1.104286x. The first deliberately
accepted public size is therefore 450K, whose final nine-pair route reaches
1.216823x; 1,000,002 points reaches 2.461791x. The accepted ladder continues
through the measured 16,000,002-point cap.

**Decision.** Add a separate `hag-nn-count1-direct-compose` placement model
rather than widening the ordinary HAG-NN stage model. Use a zero-intercept
923.5390448076193 ns/point host slope and a zero-intercept
229.06845240043793 ns/point device residual after charging the existing cold
start and two synchronizations plus a 25-byte/point upload, 8-byte/point spill,
zero packing, and full 112-byte/point planner-owned index. Evaluate it only
inside the hard 450,000--16,000,002 evidence envelope. The marginal 400,002
direction and all smaller rows remain host-selected.

**Decision.** Automatic admission requires a single-key object root and the
literal three-stage grammar: an option-free lowercase `.las` reader;
`filters.hag_nn` with unsigned integer `count=1` and no other option; and a
lowercase `.las` writer whose only additional option is `extra_dims=all`.
Runtime facts must prove an uncompressed LAS 1.4 format-7 mapped source with
40-byte records containing exactly one unsigned-32 `OffsetTime` Extra Bytes
descriptor, a 48-byte output record, one 2D kNN index/region/lane, and a point
count in the measured envelope. Selection is restricted to the exact RTX
4090/SM89/CUDA 13.3/driver 610.43.03 profile.

**Decision.** Placement and resident preflight both reserve a conservative
160 bytes/point for the complete executor high-water; the count-one stage's
local scratch remains 15 bytes/point. The required-route process gate freezes
acceptance at exactly `160*N` bytes and refusal at `160*N - 1`. A selected
route must prove the executable rewrite, one selected HAG region and shared
index, mapped-source/direct-extra-double execution, correct output
cardinality/record summary, and actual device use. Selection alone is not
sufficient evidence.

**Decision.** All recoverable and filesystem refusals remain precommit.
Grammar, source, layout, count, budget, profile, rewrite, preflight, execution-
proof, existing-output, alias, or symlink refusal discards the candidate and
runs the unchanged public pipeline. The required form exits 124 without
output. The existing exact HAG repair semantics remain unchanged; this slice
adds selection and proof, not a new kernel or relaxed answer.

**Rationale.** Final ordinary public runs are byte-, metadata-, point-order-,
stdout-, stderr-, and status-exact. Nine alternating pairs measure 1.216823x
median pinned PDAL at the 450K floor and 2.461791x at 1,000,002 points, with
all 18 candidate samples faster and both paired intervals clear of parity. A
201/203 placement audit is intentionally conservative: its only disagreements
are refusals at the 300K parity row and 400,002 marginal-win row; every
admitted row and every measured losing control agrees.

**Consequences.** `filters.hag_nn,count=1` is automatically selected only
inside this strict direct envelope. Counts 2--64 remain exact GPU-native only
inside their explicit envelopes, and all other options, layouts, compression,
cardinalities, devices, and profiles remain exact but explicit or host-
selected. Expansion requires a new exact complete-process ladder, public
selection/execution proof, full allocation accounting, and updated calibration
evidence.

## D0235 — Automatically select only the measured count-three HAG-Delaunay direct composition

D0096 established an exact data-dependent filters.hag_delaunay,count=3 lane on
the planner-owned masked 2D index. B0112 and B0113 proved the forced and
mapped-source/direct-double-output mechanisms, and B0132 qualified the named
1,000,002-point explicit route at 2.204694x pinned PDAL. They did not provide
a cardinality ladder, distinct placement model, or public automatic proof, so
the route correctly remained explicit.

A current exact direct ladder loses from 50,001 through 350,001 points. The
400,002 row is only 1.068195x, below D0190's predeclared 10% acceptance
margin. The first accepted public size is 450K: nine alternating pairs reach
1.121367x median pinned PDAL, with a 1.118491x +/- 0.013647 paired interval.
At 1,000,002 points the public route reaches 2.106454x. The accepted direct
ladder continues through the measured 16,000,002-point cap.

**Decision.** Add a separate hag-delaunay-count3-direct-compose placement
model rather than widening any ordinary HAG stage model. Use a zero-intercept
823.250610498779 ns/point host slope and a zero-intercept
185.91405462266016 ns/point device residual after charging the existing cold
start and two synchronizations plus a 25-byte/point upload, 8-byte/point spill,
zero packing, and full 112-byte/point planner-owned index. Evaluate it only
inside the hard 450,000--16,000,002 evidence envelope.

**Decision.** Automatic admission requires a single-key object root and the
literal three-stage grammar: an option-free lowercase .las reader;
filters.hag_delaunay with unsigned integer count=3 and no other option; and a
lowercase .las writer whose only additional option is extra_dims=all. Runtime
facts must prove an uncompressed LAS 1.4 format-7 mapped source with 40-byte
records containing exactly one unsigned-32 OffsetTime Extra Bytes descriptor,
a 48-byte output record, one 2D kNN index/region/lane, and a point count in the
measured envelope. Selection is restricted to the exact RTX
4090/SM89/CUDA 13.3/driver 610.43.03 profile.

**Decision.** Placement and resident preflight both reserve a conservative
184 bytes/point for the complete executor high-water. Stage-local count-three
scratch remains 39 bytes/point: three status bytes and three ordered
uint32/binary64 neighbor pairs. The required-route process gate freezes
acceptance at exactly 184*N bytes and refusal at 184*N - 1. A selected route
must prove the executable rewrite, only the HAG-Delaunay stage/region, one
shared index, mapped-source/direct-extra-double execution, record summary,
output cardinality, one active lane, and actual device use.

**Decision.** Grammar, source, layout, count, budget, profile, rewrite,
preflight, execution proof, and rejected destinations remain precommit
refusals. Tie and incomplete-search rows retain the existing exact pinned-host
repair before publication. The public no-ground row intentionally preserves
the pinned 2.10.0 diagnostic and SIGSEGV status; a test-only required-route
sentinel may not supersede oracle behavior. This slice changes bounded
selection and proof, not the kernel, interpolation recurrence, repair
semantics, or public error contract.

**Rationale.** Both final public rows are byte-, metadata-, point-order-,
stdout-, stderr-, and status-exact, with all 18 candidate samples faster. The
placement audit matches 215/218 directions; its new disagreement is only the
deliberately refused 400,002 marginal-win row, while every admitted Delaunay
row and every measured loss matches. Raw provenance authenticates 224/224
unique reports.

**Consequences.** filters.hag_delaunay,count=3 is automatically selected only
inside this strict direct envelope. Default/count-four and wider local
triangulations, explicit option variants, other Extra Bytes layouts,
compression, cardinalities, devices, and profiles remain exact but explicit
or host-selected. Expansion requires another exact complete-process ladder,
public selection/execution proof, allocation accounting, and updated
calibration evidence.

## D0236 — Correct the HAG-Delaunay proof and raise its automatic floor

D0235 admitted 450K after a final nine-pair median cleared 1.10, but independent
review found four proof defects: the calibrated estimate could be overwritten
by a generic forced-composition estimate; the required-route sentinel proved
selection without proving successful CUDA execution; the automatic matrix did
not reject an incompatible same-stride Extra Bytes descriptor; and there was
no final same-route profile/provenance proof. A fresh review-time nine-pair
public 450K run then measured only 1.093356x median, with a 1.094401x +/-
0.022493 paired interval. Its lower bound does not clear the predeclared 10%
automatic-selection margin.

**Decision.** Supersede D0235's 450,000-point minimum with 500,001. Retain its
model slopes, cost terms, exact grammar/layout/profile predicates,
184-byte/point reservation, repair semantics, and 16,000,002-point ceiling.
The next measured ladder row is the first admissible row: final current-binary
public runs reach 1.245213x median at 500,001 points and 2.049354x at
1,000,002, with paired lower bounds of 1.221150 and 2.015534 respectively.
The 450K row becomes a deliberate host selection even though its point
estimate remains above parity.

**Decision.** Preserve the selected `hag-delaunay-count3-direct-compose`
calibration through rewrite; the generic forced-composition estimate may not
replace it. Calibration telemetry must recognize the HAG-Delaunay region and
report that the selected calibration matches the executor. The diagnostic
`resident --stats` form may activate this literal automatic selector only
under the engine-owned required-route environment, so it can prove the same
placement and executor without broadening ordinary CLI admission.

**Decision.** Selection and executable rewrite remain necessary but are not
sufficient. Before atomic publication, the selected public route must observe
successful exact HAG-Delaunay CUDA execution from the active mapped-source
scope. A selected route whose device lane declines is recoverable before
publication: the ordinary public invocation runs the unchanged host pipeline,
while the required form exits 124 without output. The required success proof
also freezes one selected HAG-Delaunay stage/region/index/lane, 40 -> 48-byte
record summary, `184*N` peak, boundary agreement, direct publication, and no
complete host XYZ mirror.

**Decision.** The source Extra Bytes predicate is semantic, not merely a
stride check. Automatic selection requires exactly the unsigned-32
`OffsetTime` descriptor calibrated by the ladder. A different name, type,
width, count, or descriptor set retains the unchanged host path even if the
physical point-record stride is also 40 bytes.

**Rationale.** The correction makes the public performance gate reproducible
and fail-closed. A same-final-binary proof report is output-exact and observes
the calibrated direct resident executor; a separate final automatic Nsight
profile records the expected 20 launches and identifies the shared kNN gather
as the limiter. The process matrix covers device decline and semantic
descriptor drift. The placement audit is intentionally 214/218 after adding
the 450K refusal, and raw provenance remains complete at 224/224.

**Consequences.** B0237/D0236 supersedes only D0235's accepted minimum,
final-code performance figures, and incomplete route-proof claims. The exact
kernel, pinned repair/fallback behavior, model coefficients, memory accounting,
and measured direct ladder remain valid. Automatic `count=3` selection is now
limited to 500,001--16,000,002 points in the same literal format-7/40 ->
48-byte SM89 envelope; every neighboring cardinality, layout, option, device,
or profile remains exact but explicit or host-selected.

## D0237 — Cache immutable KD3 coordinates and parallelize large exact LOF repair

B0043's exact automatic 4M LOF pipeline spent 2.329585210 seconds in host
closure repair, and B0046 later confirmed that this dominated its resident
manager wall. B0039 rejected a device substitute because the current GPU
index does not reproduce the pinned KD3 tie order. The remaining compatible
surface is therefore the pinned host repair itself. Profiling showed that most
of its time was not the 40,982 repaired rows: it was constructing and querying
the one full compatibility KD3 index through repeated PointView field
indirection.

**Decision.** Preserve nanoflann's pinned tree, traversal, result order, and
squared-distance arithmetic. Add an opt-in KD3 backing that materializes the
immutable logical binary64 XYZ values into one contiguous array before the
same index is built. The original `KD3Index(PointView)` and
`PointView::build3dIndex()` entry points remain uncached, so ordinary upstream
stages and small repairs keep their prior behavior and memory. LOF requests
the cached backing only when its exact factor-repair closure contains at least
4,096 rows. This remains the single PointView-owned compatibility index; no
private tree or second spatial product is permitted.

**Decision.** Large LOF closure repair uses deterministic compact point-id
lists and fixed contiguous chunks. Worker joins are strict dependency barriers
between k-distance, density, and factor passes. Each worker owns its query
buffers and disjoint output rows; the built KD3 index and input coordinates are
read-only. Small closures remain serial. `PDG_NATIVE_WORKERS`, when present,
is a positive cap rather than a request to oversubscribe useful work. Worker
exceptions are captured, all threads are joined, and the first failure is
re-thrown through the existing prepublication stage boundary.

**Decision.** The cache costs exactly 24 bytes per input point, or 96,000,000
payload bytes on the qualified 4M case. Allocation and every repair pass occur
before publication. PointView builds into a local owner and publishes the
index only after success. An ordinary cache allocation/build failure therefore
retries the same exact KD3 index uncached; a cache-proof invocation fails
closed. Engine-owned assertion variables independently require parallel repair
and the cached backing, while an engine-owned disable variable supplies the
same-final-binary control. Terminal public proof guards also reject any
automatic selector, preflight, or device decline before oracle delegation.

**Rationale.** Three alternating current-final-binary public pairs are exact
and improve from a 3.728858155-second uncached median to 2.026059035 seconds,
or 1.840449x, with every pair between 1.791923x and 1.840449x. Matched stats
reduce exact host repair from 2.183796292 to 0.504709589 seconds while
retaining one planner-owned index, 12,716 ambiguous rows, one incomplete row,
and 40,982 factor repairs. The final nine-pair public proof is exact at
18.300959x pinned PDAL: 35.860210317 seconds versus 1.959471607 seconds. A
separate option-free three-pair run is exact at 18.550858x. Peak RSS rises by
103,224 KiB in the matched diagnostic, consistent with the deliberate 96 MB
payload plus allocator/process variation.

**Consequences.** This is a host-side performance improvement inside the
existing exact LOF route. Functional support, GPU-native stage coverage,
placement coefficients, automatic grammar, and fallback semantics do not
change. Cached and uncached kNN results and LOF outputs at one, two, and four
workers are bit-identical; cached-build failure is exception-atomic; four
concurrent readers pass TSan; the full host and ASan/UBSan aggregates are
clean; and the physical SM89 LOF matrix remains exact. No CUDA translation
unit changed, so this decision adds no new Compute Sanitizer claim.

## D0238 — Qualify selective exact HAG-NN repair for the measured r2 endpoint

B0225/D0224 rejected r2's named LAZ writer as a writer-only optimization, and
B0230/D0229 restored public parity by dispatching the literal reference graph
straight to pinned PDAL. A complete-process profile subsequently showed that
HAG-NN, not SMRF, was the largest removable stage cost. Forcing the existing
exact CUDA HAG implementation still lost because 347 equal-distance rows made
the wrapper discard the entire device column and rerun HAG-NN on all one
million points.

**Decision.** Keep option-free SMRF, LAZ decode/encode, and the named
`HeightAboveGround=float32` writer on their exact host implementations. Execute
only option-free HAG-NN through the planner-owned two-dimensional resident
index, collect every tie/incomplete status, and repair only those compact rows
with the existing pinned ground KD2 compatibility index and upstream HAG
arithmetic. Publish the resident binary64 HAG column to the unchanged named
writer only after all repairs succeed. This compatibility repair index is not
a second device spatial product and may not be reused to broaden GPU-native
coverage.

**Decision.** Automatic admission is a workload-specific endpoint, not a
general HAG or terrain model. It requires the checked-in literal object-root
`r2-ground-normalize` grammar, exactly 1,000,000 compressed format-7 points,
the measured header/extent and complete-file fingerprint, the exact named LAZ
layout, and the RTX 4090/SM89/CUDA 13.3/driver 610.43.03 profile. The dispatcher
arms an engine-owned internal marker only after those facts match. The generic
resident route must decline that marker so the intended hybrid can evaluate;
the wrapper must request the exact CUDA HAG lane. Any neighboring source,
cardinality, case, option, layout, profile, fingerprint, rewrite, or execution
state delegates before output side effects.

**Decision.** Engine-owned proof variables independently require successful
automatic selection and observed selective repair. They assert the naturally
selected behavior and never widen it. An ordinary decline preserves B0230's
direct pinned-oracle path. Required refusals fail without output. The full
HAG-NN output remains subject to pinned byte, metadata, order, stdout, stderr,
and status comparison; no tolerance or relaxed arithmetic is introduced.

**Rationale.** Five-pair attribution improves from an exact 0.819971x when all
HAG rows are recomputed on host to 1.267276x when only 347 rejected rows are
repaired. The final public nine-pair route is exact at 1.278607x pinned PDAL,
with a 1.291345x +/- 0.015572 paired mean and 9/9 wins. The earlier 0.918077x
automatic attempt is retained as negative evidence: it exposed resident-route
preemption and a missing CUDA request before qualification. The final profile
places the next r2 limiter in lazperf decode and remaining host/I/O work, not
the HAG projection kernel.

**Consequences.** B0239/D0238 supersedes B0225's measured r2 deferral and
B0230's direct-host performance selection only for the exact fingerprinted 1M
reference endpoint. It does not qualify SMRF, a general named-layout writer,
another HAG count, another fixture, or another device. Focused host, physical
CUDA, ASan/UBSan, memcheck, and racecheck gates are clean. Expansion requires
a new complete-process fixture ladder and exact/refusal/profile evidence.

**Independent-review correction.** The inherited internal marker is a routing
hint, never authority. The engine must freshly parse the original JSON and
revalidate the literal object root, exact stage order/options, lowercase
non-COPC LAZ endpoints, and named writer before enabling either replacement;
it must then validate the complete source fingerprint. An external marker
paired with any grammar/layout drift fails before execution.

The selective-repair proof is terminal: if the selected wrapper does not
execute CUDA and observe at least one rejected row, it throws before the named
writer rather than running full host HAG. A test-only post-selection decline
freezes this behavior. The exact ground-domain view and its pinned KD2 tree are
now a lazily acquired, exception-atomic field of the planner-scoped resident
neighborhood product and are built at most once per region; the stage wrapper
does not own a private index. This is an explicitly planner-owned compatibility
product beside the shared device index, not a new independently managed stage
tree.

Finally, qualification reports must scrub ambient `PDG_*` variables and inject
recorded candidate-only proof values through the reference harness. The
corrected final row remains exact at 1.270989x median with 9/9 wins. B0239's v3
report and same-final-binary profile, not the pre-review v2 row, are the
acceptance evidence.

## D0239 — Expand the speed target from six to fourteen production workflows

D0208 correctly made measured end-to-end speed the product goal, but its six
references omitted several common production jobs. Optimizing only that set
could improve the reported aggregate while leaving raster colorization,
standalone geometry clipping, decimation, classification, publication tiling,
multi-file assembly, and format-only throughput unmeasured.

**Decision.** The release/reference suite is the ordered fourteen-workload
manifest in `bench/pipelines/reference/manifest.json`: r1-r6 unchanged, then
r7 DSM, r8 orthophoto colorization, r9 polygon clipping with geometry
reprojection, r10 decimation, r11 PDAL-native classification/refinement, r12
tiling, r13 merge/mosaic, and r14 conversion/compression. Every headline has
weight one. Ordinal, radius, grid, voxel-center, density-aware, and additional
LAS/LAZ/COPC direction pipelines are zero-weight supporting variants; they do
not give decimation or conversion extra influence on the headline.

**Decision.** Dynamic inputs, bounds, raster paths, tile origins, and the
EPSG:4326 multipolygon/hole WKT are materialized by the benchmark harness from
hash-pinned fixtures and source metadata. The retained AHN-derived LAS/LAZ/COPC
files remain immutable local corpus. A checked-in recipe derives the
uncompressed input, an EPSG:3857 three-band raster, and two merge inputs that
differ in order, LAS version, point format, layout, and header. No corpus file
is committed, renamed, or modified. COPC conversion variants set a fixed
writer seed/single writer thread and one reader request because the pinned
writer's default random sampling and concurrent reader otherwise make its own
bytes vary between repeats.

**Decision.** Exactness is fail-closed at the complete-process boundary.
Artifact bytes, names/membership, metadata, point order, stdout, stderr, and
exit status must match. Raster reports additionally retain and compare the
complete normalized `gdalinfo -json -checksum` document: driver, dimensions,
geotransform, CRS, nodata, bands/types, blocks, checksums, masks, overviews,
and metadata. A missing artifact or workload, nondeterministic oracle/candidate
run, malformed report, or inexact comparison makes the suite incomplete and
no performance aggregate is emitted. Slower or unproved native paths remain
host-selected.

**Decision.** Warm and cold cache are separate reports. Warm uses explicit
untimed warmups; cold applies file-scoped `POSIX_FADV_DONTNEED` to every used
input before every process and records the method. Reports include complete
wall, sampled descendant-tree peak RSS, input/output throughput, output size,
and compression ratio. Profiles follow the public command and all descendants
and attribute reader, transformation/stage, writer/compression, transfer, I/O,
and startup as available; an instrumentation gap is recorded rather than
filled with an inferred number.

**Decision.** Two aggregates are mandatory for each cache state. For workload
speedups `S_i = median(PDAL_i) / median(PDG_i)`, equal-workload speed is
`exp(mean(log(S_i)))`. Total-wall speed is
`sum(median(PDAL_i)) / sum(median(PDG_i))`. The report also sums the aligned
oracle and candidate walls for each measured suite round. There is no silent
omission, imputation, or renormalization.

**Rationale.** B0240's first exact three-pair baseline measures 1.211602x warm
and 1.206003x cold equal-workload geometric mean, plus 1.563923x warm and
1.554365x cold total-wall speedup. The gain is concentrated in r2/r4/r6; all
new headlines remain at or below parity. At 6.983998 seconds pinned PDAL and
7.735668 seconds PDG warm, r11 is both the largest remaining reference wall
and the largest absolute loss. Its existing exact standalone neighborhood
lanes make a composition attribution/prototype the next smallest measured
slice, but no automatic selection follows from that hypothesis.

**Consequences.** D0208 remains the speed-first policy but its six-workload
membership is superseded by this manifest. No later optimization may claim an
aggregate from r1-r6 alone or choose among the new workflows by intuition.
Functional, GPU-native, performance-qualified, and automatically selected
remain separate categories. The current new-workload paths remain exact host
fallbacks until a complete measured vertical slice proves a faster backend.

## D0240 — Specialize optional KD3 backing at construction and retain r11 host selection

B0240 identified r11 as the largest remaining headline wall and a roughly ten
percent public-process loss. Its literal five-stage graph contains a
domain-qualified neighbor-classifier outside the current exact CUDA envelope,
and no existing composition matcher covers the graph. A test-first direct
delegate to the sibling forked-PDAL CLI preserved exact output but measured
only 0.901714x pinned PDAL with 0/9 wins, disproving process routing as the
cause and preventing that route from shipping.

**Decision.** Preserve B0238's opt-in contiguous immutable KD3 coordinate
backing and pinned nanoflann algorithm. Choose cached or uncached backing once
when the `KD3Index` is constructed, using separate compile-time
specializations. The uncached specialization reads the `PointView` through
the original branch-free point and distance callbacks. The cached
specialization reads the same logical binary64 XYZ values from its contiguous
backing. Both keep the same tree parameters, traversal, result ordering,
squared-distance arithmetic, and tie behavior.

**Decision.** Do not add an r11 dispatcher shortcut, automatic matcher,
resident region, or native-coverage claim. Its exact host graph remains the
fastest measured backend. Cache selection remains explicit and fail-closed:
ordinary `KD3Index` construction is uncached, the qualified large LOF repair
requests the cached specialization, and cached build publication remains
exception-atomic.

**Rationale.** The optional-cache implementation had left a cache-state branch
inside every nanoflann point and distance callback, including ordinary
uncached uses. Construction-time specialization removes that tax. Nine warm
public r11 pairs are exact at 1.000990x median with a paired mean of 0.999961x
+/- 0.003881, and three cold pairs are exact at 1.003846x; both are unresolved
oracle parity rather than an acceleration. The retained physical 4M LOF proof
remains exact at 20.295840x pinned PDAL, so the prior qualified cache win is
not sacrificed.

**Consequences.** B0241 closes the r11 regression without changing functional,
GPU-native, performance-qualified, or automatic selection scope. Cached and
uncached results, failure atomicity, and concurrent readers pass Host Debug,
ASan/UBSan, and TSan; the complete 508-test Host Debug suite passes with only
four documented optional corpus skips. No CUDA translation unit changed. The
updated fourteen-workload aggregates are 1.220209x warm and 1.215129x cold
equal-workload geometric mean, plus 1.632359x warm and 1.627712x cold
total-wall speedup. The next measured slice is r8 complete-process
colorization attribution.

## D0241 — Direct-host the measured r8 endpoint before implementing new stages

B0240/B0241 made r8 the largest remaining unaccelerated headline wall. The
literal graph contains two `filters.reprojection` stages and one
`filters.colorization`; none has an engine implementation. The LAS endpoints
nevertheless caused the thin launcher to start the full engine, which then
started sibling PDAL to execute every stage unchanged. A no-op oracle handoff
measures 19--23 ms of engine-only work against B0240's 15--17 ms r8 deficit.

**Decision.** Send only the measured r8 endpoint directly from the thin
launcher to sibling PDAL. Admission requires the literal object root, exact
five-stage order and option sets, lowercase non-COPC LAZ endpoints, `.tif`
raster suffix, EPSG:28992-to-3857 and reverse SRS pairs, the exact RGB
dimensions string, string-valued `compression=true`, one million points, and
the measured compressed format-7/36-byte layout. Supported CLI modifiers and
every engine-owned environment control retain the engine route. Grammar,
count, or header/layout drift also stays in-engine.

**Decision.** This is a host-dispatch selection, not native reprojection,
colorization, raster, reader, writer, or CUDA coverage. It executes the
original pipeline before any pipeline output side effect and preserves the
configured oracle/sibling process's artifact bytes, metadata, order,
diagnostics, and status. Future native work must beat this lower-overhead
endpoint and add a separately measured matcher; this route is not evidence to
generalize by stage name.

**Rationale.** A test-first direct-sibling control is exact and resolves at
1.005962x pinned PDAL over nine pairs. The final public warm/cold rows are exact
at unresolved 1.005180x/1.004458x pinned-oracle parity. More importantly, a
same-final-binary A/B measures the selected public route at 1.624616676 seconds
and the forced former engine route at 1.640871976 seconds: **1.010006x** by
medians, with the public/engine paired ratio resolving below parity. The final
profile contains one sibling-PDAL descendant instead of engine plus sibling,
and all PROJ/GDAL/lazperf/writer work remains in that unchanged child.

**Consequences.** r8's fastest exact measured backend is now automatically
selected without adding a stage implementation. Unit refusals cover count,
layout, SRS, raster suffix, dimensions, writer option, and extra-option drift;
the process boundary proves selection/refusal and configured-oracle stream and
status preservation. Complete warm/cold artifacts remain exact. Host Debug is
509/509, focused ASan/UBSan is clean, and pinned upstream colorization and
reprojection units pass. The fourteen-workload aggregate is now 1.219925x warm
and 1.216985x cold equal-workload geometric mean plus 1.631658x warm and
1.630957x cold total-wall speedup. r13 heterogeneous merge is next by the
measured opportunity ledger.

## D0242 — Supersede r8 direct selection after corrected-final measurement

Independent review found two proof weaknesses in D0241. First, direct routing
depended on an enumerated environment-control list and could bypass a newly
added engine-only proof variable. Second, the selected route's performance
gate had to survive that correction on the final binary rather than inherit an
earlier positive sample.

**Decision.** Every startup environment name with the `PDG_` prefix requires
engine ownership except `PDG_ORACLE_PDAL`, which only identifies the equivalent
sibling oracle. This is a prefix policy over the actual process environment,
not an allowlist. The r2 internal marker is added only after startup capture,
so internal dispatch still works while external internal/future controls fail
closed. Keep the known-control array only as a reviewable inventory, never as
the routing authority.

**Decision.** Remove the r8 direct-delegation matcher and supersede D0241's
automatic-selection consequence. A corrected-final 21-pair direct-versus-
forced-engine control is exact but unresolved at 0.995822x +/- 0.004522 in
report orientation; direct is only 1.000950x faster by inverse medians. The
same direct prototype is 0.994289x warm and a resolved 0.988765x cold against
pinned PDAL. Earlier positive samples do not establish a stable final-binary
win. Under D0208 and the fail-closed rule, exact but unproved work stays on the
existing host/engine path.

**Consequences.** B0243 records the rejected prototype append-only and freezes
the literal r8 graph as an engine-route refusal. r8 remains functional host
fallback with no GPU-native, performance-qualified, or automatic-selection
claim. The retained three-pair aggregates are 1.220823x warm and 1.214832x cold
equal-workload geometric mean plus 1.633139x warm and 1.627446x cold total-wall
speedup. The environment change strengthens every direct route's diagnostic
contract without broadening any pipeline grammar. r13 merge remains next.

## D0243 — Select one reader worker only for the immutable r13 merge endpoint

B0244 profiles r13 as sequential heterogeneous LAZ decode plus compressed LAS
publication; the streaming `filters.merge` itself is effectively free. A
direct-sibling control was exact but unresolved, while forcing one
`readers.las` worker resolved faster. Keeping the engine in that route erased
the gain and resolved slower, so that implementation was rejected.

**Decision.** Directly exec sibling PDAL with
`--readers.las.threads=1` only for the literal checked-in r13 graph and both
immutable fixture calibration fingerprints. Admission freezes
stage/root/option/tag/input order, lowercase non-COPC LAZ endpoints, full file
bytes through a runtime
CRC32, and exact size/count/version/raw-format/header/offset/record-length
facts. The fixture manifest's SHA-256 remains authoritative provenance; the
CRC is a non-cryptographic runtime calibration key, not a corpus identity. All malformed,
drifted, option-rich, differently ordered, or unmatched inputs retain the
ordinary engine route before output.

**Decision.** This is reader scheduling for one measured complete workflow,
not generalized `readers.las`, `filters.merge`, `writers.las`, host-native, or
GPU-native coverage. The direct route may bypass the engine only with the
startup controls `PDG_ORACLE_PDAL`, `PDG_FROZEN_EPOCH`, and the r13 proof;
every other current or future `PDG_*` name remains engine-owned under D0242.
The proof fails with status 124 when the exact route is not selected.

**Rationale.** The final public route is byte/metadata/order/stream/status
exact and resolves at 1.018691x over 21 warm pairs and 1.004013x over 41 cold
pairs. Its profile contains only launcher and sibling PDAL, with the appended
option visible before the child performs unchanged decode, merge, and writer
work. The rejected engine-retained variant is 0.978435x and remains recorded.

**Consequences.** r13 gains one bounded performance-qualified automatic host
selection but no stage or CUDA implementation. The final all-headline physical
GPU aggregates are 1.206488x warm and 1.207186x cold equal-workload geometric
mean plus 1.626512x warm and 1.628676x cold total-wall speedup. Host Debug is
510/510 with four optional corpus skips and focused leak-disabled ASan/UBSan is
clean. The refreshed complete-workload ledger selects r12 deterministic tiling
as the next smallest vertical slice.

## D0244 — Supersede r13 automatic selection after corrected-final cold refusal

Independent review found that D0243's initial implementation admitted
supported pipeline modifiers outside its documented plain-invocation scope and
omitted portable dynamic-loader linkage. Both defects were corrected and a
registered-fixture positive process boundary was added before remeasurement.

**Decision.** Remove the r13 matcher, fingerprint, proof variable, reader
override, and direct route. The corrected final binary resolves faster over 21
warm pairs, but its 41-pair file-evicted cold interval spans parity. A public
route cannot know cache state, and the expanded workload contract requires
both warm and cold evidence. Warm-only resolution therefore cannot qualify a
cache-agnostic selection under D0208's fail-closed rule.

**Decision.** Retain the reader-worker sweep and both rejected implementations
as negative opportunity evidence. Do not generalize one-worker scheduling by
stage, input count, format, or size, and do not retain a dormant CRC-based
fixture identity surface. Any future attempt needs a new final-binary warm and
cold gate and a complete process test before selection.

**Consequences.** Removing the prototype restores the byte-identical B0243
launcher and engine binaries, so B0243's all-headline aggregates remain the
current claims: 1.220823x/1.214832x warm/cold equal-workload geometric mean and
1.633139x/1.627446x total-wall speedup. r13 remains exact engine/host fallback
with no GPU-native, performance-qualified, or automatic-selection entry. r12
deterministic tiling remains the next measured vertical slice. Host Debug
passes 510/510 with four optional corpus skips, and the negative r13 route
freeze passes focused leak-disabled ASan/UBSan.

## D0245 — Reject r12 dispatcher and reader-scheduling prototypes

B0246 profiles the deterministic seven-output tiling reference as LAZ
decode-shaped host work behind launcher and engine processes. Complete output
publication is exact, but the profile and timing do not identify a qualified
splitter, compressor, or filesystem replacement.

**Decision.** Do not add a literal r12 direct route. The direct sibling alone
is unresolved versus pinned PDAL in both cache states. The integrated public
prototype resolves slower than pinned PDAL warm and cold, while the required
same-final direct-versus-forced-engine A/B spans parity in both states and
changes direction. Remove the matcher, input probe, plain-invocation rule, and
positive route tests; retain a literal unit/process negative-selection freeze.

**Decision.** Keep the default seven LAS reader workers for r12. One, two,
six, and eight workers resolve slower in the bounded exact sweep; four workers
is unresolved. Do not generalize reader scheduling from r13 or retain an r12
override. The existing splitter CUDA path also remains explicit-only because
its prior complete-process qualification is slower than PDAL and B0246 finds
no new evidence for automatic selection.

**Consequences.** r12 remains exact engine/PDAL host execution with no new
host-native, GPU-native, performance-qualified, or automatic-selection claim.
The restored launcher and engine are byte-identical to B0243/B0245, so current
aggregate claims remain 1.220823x/1.214832x warm/cold equal-workload geometric
mean and 1.633139x/1.627446x total-wall speedup. The prioritized ledger moves
to r9 polygon clipping. A future r12 attempt must present a new exact
algorithmic or codec/publication hypothesis plus resolved same-final warm and
cold gates; repeating wrapper removal or reader-count sampling is insufficient.

## D0246 — Correct r9's axis order and reject startup-only selection

B0247 proves that the original r9 reference was not representative. PROJ's
EPSG:4326 authority-axis output was inserted directly into traditional X/Y
WKT, swapping longitude and latitude and producing an empty crop. Determinism
and byte equality did not make that zero-work endpoint acceptable.

**Decision.** Materialize the r9 geometry with longitude as WKT X and latitude
as WKT Y, and test the resulting coordinate domain. The corrected headline
must contain the documented multipolygon, hole, disjoint member, and geometry
CRS reprojection and must emit a substantive deterministic artifact. Treat the
old r9 row and every aggregate containing it as historical and superseded;
never preserve a headline claim merely because an invalid workload was fast.

**Decision.** Do not add an r9 direct or embedded-oracle route. Same-final
controls confirm that unused engine startup is removable, but the complete
public direct-exec prototype resolves at only 0.957175x pinned PDAL and the
private in-process PDAL CLI prototype resolves at only 0.951883x. Both are
exact losses over 21 warm pairs, so the warm gate alone is sufficient to
reject them. Remove the matcher, module, application entry point, build and
install rules, and positive route tests. Retain a literal unit/process
negative-selection freeze and the ordinary engine/PDAL host fallback.

**Consequences.** Corrected r9 emits 473,825 points in a 3,205,857-byte exact
LAZ and gains an eight-case differential/refusal/malformed/determinism matrix.
It remains functionally exact host execution with no new host-native,
GPU-native, performance-qualified, or automatic-selection entry. Corrected
all-headline aggregates are fresh same-binary executions, not replacement-row
splices: 1.205506x/1.206910x warm/cold equal-workload geometric mean and
1.617787x/1.615619x total-wall speedup. Aggregation now fails closed on mixed
oracle/candidate hashes, report schema/label/cache-slot drift, or any measured
r9 artifact other than 473,825 points. The ledger moves to r7 DSM. Any future
r9 attempt must improve measured decode, startup, or a reusable end-to-end host
path and clear both cache-state gates; repeating wrapper removal or embedding
the unchanged CLI is insufficient.

## D0247 — Retain bounded r7 direct-host dispatch and separate benchmark controls

B0248 profiles r7 as unchanged PDAL LAZ decode plus a small maximum-Z GeoTIFF
sink behind an engine process that performs no productive stage work. A direct
sibling control is at pinned-PDAL parity, and the final public direct route is
exact while resolving 1.058781x warm and 1.049022x cold against the same-final
engine path. Against pinned PDAL, the launcher remains a resolved 0.986187x
paired warm loss and unresolved 0.996456x cold result; those rows are retained
explicitly and are not an algorithmic acceleration claim.

**Decision.** Retain direct host delegation only for the complete literal r7
reference grammar: non-COPC lowercase LAZ with `override_srs=EPSG:28992`,
`filters.returns(groups=first,only)`, and the exact one-metre Float64
maximum-Z writer policy, nodata, and materialized full bounds. Require the
plain pipeline invocation. Every root/stage/option/type/extension/SRS/bounds
drift, supported CLI modifier, malformed graph, and startup `PDG_*` product
control remains engine-owned. The accepted and refused implementations both
execute unchanged PDAL stages, so fixture fingerprinting is unnecessary for
semantic identity; the matcher is a performance envelope, not a native-stage
substitution. Do not generalize it to other return groups, surface policies,
rasters, bounds, SRSs, formats, or DSM graphs without new complete-process
evidence.

**Decision.** Reserve `PDG_*` for product behavior and diagnostics. The
deterministic clock preload and maintained benchmark/fixture/differential
harnesses use `PDAL_TEST_FROZEN_EPOCH`; the benchmark runner scrubs ambient
instances, rejects explicit injection, injects it only with the named preload,
and records the actual variable in its report. Keep `PDG_FROZEN_EPOCH` only as
a compatibility input inside the preload library. Because the launcher must
continue failing closed on every present `PDG_*` name except
`PDG_ORACLE_PDAL`, using the historical spelling in a manual command will
still select the engine. Do not weaken the launcher's prefix rule to make
instrumentation convenient.

**Consequences.** B0247's aggregate was collected with the old engine-forcing
clock name and is historical. Fresh single-binary all-headline results are
1.223460x/1.227305x warm/cold equal-workload geometric mean and
1.600366x/1.621301x total-wall speedup. r7 remains functionally exact PDAL
host execution, adds a bounded performance-qualified automatic direct-host
route, and adds no host-native or GPU-native stage. Its deterministic process
matrix freezes maximum-Z/first-or-only behavior, full raster metadata/bytes,
refusals, malformed inputs, and repeat stability. The next measured vertical
slice is r10 decimation; prefer reusable decode/I/O work over an r7-specific
raster kernel because the final r7 profile remains lazperf-decode shaped.

## D0248 — Retain bounded r10 direct-host dispatch; do not port the voxel stage

B0249 profiles the r10 headline as unchanged PDAL LAZ decode plus a very small
voxel-centroid-nearest-original-record selection behind an engine process that
performs no productive stage work. Format-7 point decode accounts for 0.302 of
0.430 sampled CPU seconds, while the filter accounts for only 0.002 seconds
inclusive. Direct sibling PDAL is at unresolved pinned-oracle parity, and the
final literal route is exact while resolving 1.041226x warm and 1.034230x cold
against the same-final engine path with 9/9 wins in both cache states. Against
pinned PDAL, public warm and cold results remain unresolved at 1.002615x and
0.993073x median speedup.

**Decision.** Retain direct host delegation only for the complete literal r10
headline grammar: a three-stage root, lowercase non-COPC `.laz` input and
output, numeric `filters.voxelcentroidnearestneighbor` `cell:2.5`, and writer
string `compression:"true"`. Require the plain pipeline invocation. Every
root, topology, stage, option name/type/value, extension, supported CLI
modifier, malformed graph, and startup `PDG_*` product control remains
engine-owned. The accepted and refused implementations both execute unchanged
PDAL stages, so fixture fingerprinting is unnecessary for semantic identity;
the grammar is a bounded performance envelope, not a native-stage
substitution. Do not generalize it to the ordinal, radius, grid, voxel-center,
density-aware, or other decimation variants without new complete-process
evidence.

**Decision.** Do not implement an r10-specific host-native or CUDA voxel path
for this endpoint. The measured filter contribution is below the useful
end-to-end opportunity, while reusable LAZ decode/I/O dominates. A future
native path must first prove exact centroid accumulation, nearest-original
record tie ownership, input order, layout/metadata, and LAZ publication, then
clear both cache-state public gates. Stage coverage cannot be inferred from
this dispatcher route.

**Consequences.** r10 remains functionally exact unchanged-PDAL host
execution, gains one bounded performance-qualified automatic direct-host
dispatch, and adds no host-native or GPU-native stage. Its nine-case
generated-fixture matrix freezes dense, sparse, empty, malformed, legal and
invalid option drift, numeric zero-cell oracle behavior, exact bytes/order/
streams/status, and repeat determinism. Fresh single-binary all-headline
results are 1.235323x/1.237401x warm/cold equal-workload geometric mean and
1.631522x/1.633568x total-wall speedup. The next measured vertical slice is
r14 conversion/compression; prioritize reusable codec/I/O work and keep it
host-selected unless a complete exact warm/cold gate qualifies a change.

## D0249 — Close r14 without reader tuning; test exact chunk compression next

B0250 confirms that the r14 LAS -> LAZ headline already takes the previously
qualified generic two-stage LAS-family direct delegation. The launcher has no
engine child to remove, and all six zero-weight conversion directions likewise
execute unchanged PDAL. A complete child profile records about 0.535 seconds,
37.324 MB read, and 6.748 MB written, but its descendant clock timer is reset;
the resulting 0.030 sampled CPU seconds are not sufficient for a function-level
percentage claim.

**Decision.** Keep the existing generic direct-host route unchanged. Do not
add an r14-specific matcher, fixture fingerprint, or automatic selector. The
new all-direction matrix is a compatibility and workload proof, not evidence
that LAS, LAZ, COPC, their readers/writers, or compression are host-native or
GPU-native.

**Decision.** Reject automatic reader-worker tuning. Counts 2/4/6/8 are
unresolved or slower in their bounded screens. One worker is exact and
resolves 1.017714x warm over nine pairs, but its cold result is unresolved at
1.001577x +/- 0.014533. Its lower peak RSS is useful diagnostic evidence but
does not satisfy the speed-first two-cache gate. Preserve the pinned reader's
default worker count.

**Decision.** Make exact parallel LAZ chunk compression the next feasibility
slice. Before integrating anything into `LasWriter`, prove that independent
lazperf compression reproduces every chunk payload and the assembled chunk
table, header, VLR/EVLR, point order, timestamp, and diagnostics byte-for-byte.
Bound task queues and peak memory. Require a primitive benchmark and same-
machine warm/cold complete-process r14 gate. A byte mismatch or unresolved/
slower endpoint closes the attempt and preserves the sequential writer.

**Consequences.** r14 gains a 12-case/19-execution generated-fixture matrix for
all seven supported directions, repeated determinism, malformed/truncated
inputs, unsupported writer behavior, and complete artifacts/process
observables. It remains functionally exact unchanged-PDAL host execution with
no new host-native, GPU-native, performance-qualified, or automatic-selection
entry. The product binary and aggregate are unchanged at 1.235323x/1.237401x
warm/cold equal-workload geometric mean and 1.631522x/1.633568x total-wall
speedup. All eight expanded headlines now have a complete first measured
vertical slice.

## D0250 — Select bounded exact two-worker LAZ compression for r14

B0251 satisfies D0249's feasibility gate. Independent fixed-50K lazperf chunks
produce the complete sequential payload and chunk table byte-for-byte across
empty, boundary, and multi-chunk counts. The production worker pool owns moved
raw chunks, retains at most one outstanding result per worker, and publishes
future results on the writer thread in source order. The maintained primitive
is exact and 1.735238x faster at one million format-7 records. Final public
same-machine gates resolve at 1.510422x warm and 1.497653x cold, with exact
artifacts, streams, status, and 9/9 wins in both states.

**Decision.** Retain two-worker compression only for the complete measured r14
LAS -> LAZ envelope: plain pipeline invocation; exact literal lowercase
reader/writer grammar and string compression spelling; one million points;
36,000,375-byte LAS 1.4 input with point offset 375, uncompressed format 7,
36-byte records, and the reference scales, offsets, and XYZ extrema. The gate
is structural, not a fixture hash: a second highly compressible admitted shape
also clears both cache states. Do not infer qualification for other counts,
layouts, formats, bounds, filenames, options, graphs, or conversion directions.

**Decision.** Keep `PDG_LAZ_COMPRESSION_THREADS` internal. The launcher sets it
only after the startup environment snapshot. Any externally supplied current
or future `PDG_*` variable except the oracle locator continues to select the
engine, so users cannot silently widen this compatibility envelope. Invalid
internal values fail explicitly, and the serial writer remains the default for
every unselected path. The candidate-only `PDAL_TEST_REQUIRE_...` assertion is
a differential proof hook, not a performance or public tuning option.

**Consequences.** r14 gains one bounded, performance-qualified exact
host-native compression backend within `writers.las`, but no new standalone or
GPU-native stage. Its six zero-weight companion directions remain unchanged
serial host execution. The all-direction matrix is now 13 cases/21 executions
and proves the real selected writer as well as output identity. Fresh
all-headline aggregates are 1.268754x/1.261640x warm/cold equal-workload
geometric mean and 1.644949x/1.629571x total-wall speedup. Ordered
chunk-parallel LAZ decode is the next reusable measured hypothesis because r7,
r10, and several r14 directions remain decode-shaped; it must preserve exact
reader diagnostics/order and clear warm and cold complete-process gates before
automatic selection.

## D0251 — Preserve r14 selection behind the corrected environment boundary

Independent review after D0250 found two proof-boundary defects in its first
launcher: automatic r14 command eligibility was evaluated before refusal of an
external product-control environment, and the candidate-only writer activation
assertion was present in production builds. B0252 corrects both and binds new
public, profile, aggregate, sanitizer, and full-host evidence to launcher
`366729409ed19872a5992568591d4cc5e139018493af3ed3ab408183e922a6eb`.

**Decision.** Preserve D0250's exact two-worker r14 selection, but make the
startup environment refusal precede every automatic command route. Any
external `PDG_*` value except the oracle locator remains engine-owned. Remove
an external `PDG_LAZ_COMPRESSION_THREADS` before entering that engine so the
internal writer channel cannot be user-controlled through fallback. Arm two
workers only after the accepted launcher's environment snapshot. Compile the
`PDAL_TEST_REQUIRE_LAZ_COMPRESSION_THREADS` observation hook only in
test-enabled builds, and remove harness-only ambient assertion/preload
transport variables before constructing either compared product environment.

**Decision.** Supersede only B0251's first-final-binary measurements and
hashes with B0252. The retained final public medians are 1.515827x warm and
1.504457x cold with 9/9 paired wins and exact artifacts/process observables.
The retained complete-suite claims are 1.262824x/1.260608x warm/cold
equal-workload geometric mean and 1.641092x/1.634768x total-wall speedup.
D0250's worker count, narrow structural envelope, six serial companion
directions, primitive evidence, and next decode hypothesis remain unchanged.

**Consequences.** External environment drift now fails closed before route
activation, the engine cannot consume a forged internal worker value, and the
production writer contains no assertion-only behavior. The four focused gates
pass Host Debug, CUDA Release, and leak-disabled Host ASan/UBSan; all 520 Host
Debug registrations pass with four documented optional skips. Ordered
chunk-parallel LAZ decode remains the next measured target.

Traceability correction: the retained selector and its three test comments
cite B0251/D0250 and B0252/D0251. The provisional `B0254` label had no
corresponding evidence or decision entry and is removed. This changes no
selection rule or executable.

## D0252 — Preserve the existing LAZ reader after two decode rejections

B0253 audits the production path before adding another scheduler. The existing
`LasReader` already queues one lazperf decode task per chunk with seven workers
and consumes completed tiles by source chunk index. A separate local decoder
under `src/io/las` has no production caller and is not an acceptable seam for
a performance claim.

**Decision.** Reject 10- and 12-worker reader schedules. Both counts are
slower for r7 and r10 in exact warm screens, lose every pair, and raise peak
RSS. Do not run cold confirmation or introduce a product control after that
failed first gate. Preserve the upstream default schedule and ordered
publication path.

**Decision.** Reject the Point14 mode-gated setup prototype. Avoiding the
unused opposite-mode integer-model bank improves factory construction
1.661797x, but complete construction plus one million decoded records improves
only 1.014837x. The final public r10 warm result is exact at 1.012374x median
but unresolved at paired 1.009303x +/- 0.013219. The production patch is
removed, and cold/r7 qualification is not promoted after a warm interval that
spans parity.

**Decision.** Retain the deterministic production-factory unit and primitive
benchmark. They cover complete bytes/order for formats 6--8, Extra Bytes, all
scanner channels, repeat determinism, and bounded truncated input without
widening public selection. A later decode optimization must improve the hot
integer-decode loop itself, pass this seam, and then independently clear warm
and cold complete-process gates.

**Consequences.** No reader option, launcher route, automatic envelope,
functional coverage, host-native coverage, or GPU-native coverage changes.
B0252's exact selected binary and fourteen-workload aggregate remain current.
No CUDA source changed, so no Compute Sanitizer result is inferred.

## D0253 — Select four LAZ reader workers only for the measured r7 DSM shape

B0254 completes the reader-worker screen that B0253 began. Four workers are
resolved positive for r7 in both cache states on the final public binary, but
r10's integrated cold interval spans parity. Six workers are unresolved,
eight workers are slower, and B0253 already rejected ten and twelve. A forced-
inline integer decoder improves the primitive only 1.010245x and is removed.
This legitimate new B0254 is distinct from the provisional label misuse that
B0252/D0251 recorded and corrected before this slice existed.

**Decision.** Retain four ordered `readers.las` workers only for the exact
measured r7 headline envelope: plain `pdg pipeline FILE`; literal lowercase
LAS reader, `filters.returns(groups=first,only)`, and maximum-Z GDAL-writer
grammar; one million compressed LAS 1.4 format-7 points with 36-byte records
and the reference input facts. The launcher appends the existing upstream
reader option only after environment and structural admission. Count, layout,
grammar, stage-option, CLI-modifier, and external product-control drift must
fail closed before the option is supplied.

**Decision.** Do not select four workers for r10 or infer a general LAZ-reader
default. Preserve B0249's exact r10 direct-host route with the ordinary reader
because B0254's 21-pair cold interval is unresolved. Preserve the existing
engine/default-reader path for all other workloads. The selector is a bounded
complete-workload tuning, not a new reader implementation or stage.

**Decision.** Accept the final r7 results of 1.031858x warm and 1.020398x cold
median, paired 1.028621x +/- 0.007070 and 1.016676x +/- 0.009502, only with
their exact TIFF metadata/bytes, streams, status, route/refusal matrices,
profile, and pinned-PDAL reports. Retain the decoder unit/benchmark but remove
the forced-inline prototype. The next target is the largest remaining measured
whole-suite loss, r11 classification/refinement, rather than another LAZ
micro-optimization with no resolved public endpoint.

**Consequences.** r7 gains one narrow performance-qualified direct-host
schedule while functional, host-native-stage, and GPU-native-stage coverage do
not change. r10 and every neighboring shape remain on their prior exact paths.
Fresh exact warm/cold equal-workload geometric means are 1.267318x/1.262102x;
one execution of every reference pipeline improves 1.642633x/1.632910x by
summed medians. The focused selector/decode tests pass CUDA Release and leak-
disabled Host ASan/UBSan, and every enabled test in the 522-registration Host
Debug suite passes with the same four optional fixture skips. No CUDA source
changed, so no Compute Sanitizer result is inferred.

## D0254 — Reuse the statistical-outlier host index on the same PointView

B0255's complete-process prefix attribution assigns roughly 3.65 seconds to
statistical outlier and another 2.53 seconds to the downstream neighbor-
classification tail of r11. The statistical stage built and discarded an
upstream nanoflann `KD3Index`; the immediately adjacent upstream neighbor
classifier then requested an equivalent index from the unchanged PointView.
Statistical outlier writes Classification only, so XYZ and point order remain
valid for reuse.

**Decision.** Build statistical outlier's existing exact nanoflann index via
`PointView::build3dIndex()` and let later consumers on that same view reuse it.
Do not change the kNN implementation, query order, distance/threshold
recurrences, classification writes, options, or diagnostics. Preserve the
private local build for radius outlier, which has no measured qualifying
composition in this slice. Rely on the existing PointView product lifetime:
coordinate, point-set, and order mutators invalidate cached products before a
later neighborhood consumer can use them.

**Decision.** Accept the general semantics-preserving host implementation but
make only the literal r11 reference process a performance claim. Its final
nine-pair public results are exact at 1.037596x warm and 1.036983x cold, with
paired 1.035692x +/- 0.004605 and 1.039730x +/- 0.008657 and 9/9 wins in both
states. Preserve the first noisy nine-pair warm report as unresolved; do not
discard it or use its median as selection evidence. A seven-case/eight-
execution r11 differential plus existing outlier/SMRF matrices freezes legal
drift, malformed input/options, determinism, all bytes, metadata, order,
streams, and status.

**Decision.** Do not infer a new automatic route, native stage, CUDA envelope,
or placement model. r11 remains upstream host execution inside the existing
engine/delegate path, without an external model dependency. Its domain-
qualified neighbor classifier remains outside the standalone GPU-native
envelope. No CUDA source changed, so no Compute Sanitizer result is inferred.

**Consequences.** Final exact fourteen-headline geometric means are
1.271378x warm and 1.278556x cold. One execution of every headline improves
1.639635x warm and 1.669959x cold by summed medians. Host Debug enumerates 523
registrations and every enabled test passes; focused CUDA Release and leak-
disabled Host ASan/UBSan gates pass. r11 remains the largest absolute headline wall,
and its final profile still identifies neighborhood search and classification
as the limiting region. The next smallest hypothesis is an allocation-free,
ascending-key vote tally bounded to r11's Classification/k=7 semantics; the
generic map path and every unproved option shape must remain unchanged unless
both warm and cold complete-process gates resolve.

## D0255 — Require a fresh statistical-outlier tree before downstream reuse

Independent review of D0254 found that its PointView lifetime premise was not
enforceable for all valid upstream graphs. `filters.normal` can populate the
cached KD3 product, and generic `filters.assign` can later modify X/Y/Z without
invalidating it. D0254's first implementation could therefore consume stale
topology where pinned PDAL's private statistical-outlier index was fresh.

**Decision.** Correct and supersede D0254's lifetime rule. Statistical outlier
must invalidate incoming PointView products before building its exact
nanoflann KD3 tree. It then publishes that newly built tree for downstream
reuse on the same XYZ/order state. This preserves the former always-fresh
stage semantics without requiring every existing upstream mutator to have
perfect invalidation behavior. Radius outlier remains unchanged.

**Decision.** Supersede B0255's library-bound performance, profile, aggregate,
and matrix claims with B0256. The corrected final r11 process is exact and
resolves 1.029892x warm / 1.037311x cold by medians, paired 1.035463x +/-
0.009249 and 1.039144x +/- 0.008688 with 9/9 wins. The expanded matrix directly
exercises cached-index construction, a generic X mutation, and statistical
outlier before publication, in addition to the r11 option/malformed/
determinism boundary.

**Decision.** Preserve D0254's scope restrictions. This is exact host product
reuse and no automatic route, grammar, native stage, CUDA envelope, placement
model, or external model dependency is added. Performance qualification is
only for the literal r11 reference process; the semantics-preserving fresh
build applies generally.

**Consequences.** Corrected exact headline aggregates are 1.267705x warm and
1.278088x cold geometric mean, and 1.656538x warm / 1.665729x cold summed-wall
speedup. Host Debug enumerates 523 registrations with every enabled test
passing; focused CUDA Release and leak-disabled Host ASan/UBSan pass. The next
measured r11 hypothesis remains an allocation-free ascending-key
Classification/k=7 vote tally, with the generic map path retained for every
unproved shape.

## D0256 — Reject the r11 vote-tally slice by attribution; name exact host-parallel kNN as the measured limiter

D0255 left one r11 hypothesis open: an allocation-free ascending-key
`Classification`/`k=7` vote tally, retained only if exact warm and cold
complete-process gates both resolved. B0257 applied D0204's cheap-prototype
rule first and measured the hypothesis in isolation with a checked-in
attribution harness before any product code was written.

**Decision.** Reject the vote-tally slice without a product prototype, process
baseline, or gate. The complete upstream tally is about 0.043 seconds of the
6.8-second public process and the allocation-free variant saves 4--6
milliseconds of it, about 0.1% of wall, against a measured paired standard
error of about 0.9%. A gate cannot resolve it, and D0204 forbids spending
certified lanes on a direction a controlled prototype has already shown to be
weak. The generic ordered-map tally therefore remains the only implementation
for every dimension, `k`, domain, candidate-file, and graph shape.

**Decision.** Record the measured limiter instead of inferring one from
sampled profiles. B0257 attributes about 3.37 seconds to the statistical
outlier's per-point kNN pass and about 2.18 seconds to the classifier's
per-point kNN pass — together about 5.55 seconds of the 6.8-second process —
and shows both passes bit-identical under fixed-chunk multi-worker execution
over the same read-only PointView-owned nanoflann index (about 0.60 seconds
combined at eleven workers). This is harness attribution, not a performance
claim. The next r11 vertical slice is therefore exact host parallelism of those
two per-point passes following D0237's worker pattern, with the outlier's
serial online-moment reduction, the classifier's vote and final assignment,
options, diagnostics, output order, and every route/envelope unchanged.
Whether to take that slice is a direction decision for the maintainer, since
it threads two upstream stage hot loops rather than the bounded tally D0255
named.

**Decision.** Treat `gprofng` sampled-CPU totals on this workstation as
undersampled by roughly 7--10x until the collector is fixed. They may still
order large costs, but must not be used to rank a small host cost against a
large one; `docs/diagnostics.md` records the measurement and points to the
in-process harness.

**Consequences.** No product source, dispatcher route, grammar, placement
model, automatic envelope, or functional/host-native/GPU-native coverage
changes. B0256's exact selected binaries, r11 qualification, and
fourteen-workload aggregates remain current. The bench tree gains
`pdg_r11_neighborhood_attribution`; it is not a registered test and requires
the local 1M reference fixture. No CUDA translation unit changed, so no
Compute Sanitizer claim is inferred.

## D0257 — Run the exact host outlier and neighbor-classifier kNN passes on fixed-chunk workers

B0257 measured that r11's remaining cost was two serial per-point kNN passes,
not the vote tally, and that both are bit-identical under fixed-chunk
multi-worker execution. D0237 already established the exact worker pattern
for large LOF repair. The maintainer approved taking that direction and asked
whether CUDA kernels for the two stages would also pay.

**Decision.** Parallelize the statistical `filters.outlier` per-row kNN pass
and the `filters.neighborclassifier` per-source-point query/vote on the exact
host path, in the fork's PDAL library, using a shared header-only policy
(`pdal/private/HostNeighborhoodWorkers.hpp`). Each worker owns its query
buffers and disjoint output rows over the read-only PointView-owned nanoflann
index; the outlier's serial online-moment reduction, threshold, split, writes,
warnings, and radius mode, and the classifier's ordered-map vote, worker-order
merge, serial application, domain, candidate-file, and vote-dimension
semantics are unchanged. The first per-row failure in original order surfaces
with the pinned diagnostic and status. This is general stage behavior, not a
route: any pipeline using either stage benefits above 4,096 rows per worker.

**Decision.** The worker count is `min(ceil(rows / 4096), hardware threads,
PDG_NATIVE_WORKERS)`; small inputs stay serial;
`PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS` is the same-final-binary serial
control; test-only builds honor force/require hooks so the tiny r11 matrix
fixture proves the chunked path against the serial oracle. Both benchmark
harnesses scrub these controls; `benchmark_reference.py` scrubs the whole
`PDAL_TEST_` prefix so no compiled-in hook can silently alter a measurement.

**Decision.** Accept the final r11 result of 4.309167x warm / 4.345954x cold
median (paired 4.321977x +/- 0.034672 and 4.352975x +/- 0.038573, 9/9 wins),
exact in bytes, metadata, order, streams, and status, with the serial control
at 1.035408x and 50K/250K controls at 3.050913x/4.297817x. Fourteen-headline
aggregates rise to 1.401126x/1.410744x warm/cold geometric mean and
2.407033x/2.408896x total wall.

**Decision.** Do not force the existing CUDA statistical outlier on r11. The
same-final-binary probe is exact but slower (4.069126x) and doubles peak RSS,
because the domain-qualified classifier is outside the CUDA wrapper's envelope
and must rebuild the shared host tree while the process pays CUDA startup. A
device route for r11 pays only if both stages run on one shared device index
after host SMRF: extend the classifier CUDA wrapper to the `domain` option
with exact tie repair, compose SMRF -> outlier -> classifier residency, and
calibrate a placement model. That composition is the recorded next candidate,
estimated at most about a further 1.3x on r11, and must be preceded by a
cheap forced prototype like this one before any certified lane is spent.

**Consequences.** Functional coverage, GPU-native coverage, dispatcher routes,
grammar, placement models, and automatic CUDA envelopes do not change; the
stage-coverage document records this as an exact host-native performance
qualification of general stage behavior. Host Debug is 523/523; the r11 matrix
(eighteen cases) passes CUDA Release, Host Debug, leak-disabled ASan/UBSan, and
TSan; a direct 50K TSan process is clean. No CUDA translation unit changed, so
no Compute Sanitizer claim is inferred.

## D0258 — Default every LAZ writer to four exact chunk-compression workers

D0250 kept B0251's exact worker pool bounded to the measured r14 shape and
left the serial writer as the default everywhere else, explicitly refusing to
infer qualification for other counts, layouts, or graphs. B0259 supplies that
missing evidence: a same-binary engine probe over all fourteen reference
headlines shows every LAZ-writing workload faster at two and at four workers
(1.04x--1.56x), every non-LAZ workload unchanged within noise, and byte-exact
artifacts throughout; an r6 sweep places the knee at four workers.

**Decision.** Supersede D0250's serial-default rule. `writers.las` compresses
LAZ chunks on `min(4, hardware threads)` workers by default. The exact
mechanism, fixed-50K independent chunks, source-order publication, and byte
identity proof are B0251's; only the default changes. The internal
`PDG_LAZ_COMPRESSION_THREADS` channel still overrides the default, so the
launcher's measured two-worker r14 selection and its tests are unchanged, and
external `PDG_*` values remain engine-owned exactly as D0251 requires. This is
general host-native writer behavior, not a route or an automatic envelope, and
it is measured on the complete reference suite rather than inferred.

**Decision.** Accept the final fourteen-headline aggregates of 1.597555x warm /
1.589241x cold equal-workload geometric mean and 2.792319x / 2.772361x
total-wall speedup, all exact, and the nine-pair final gates r6 6.316372x /
6.387622x, r13 1.437716x / 1.434382x, and r1 1.081819x / 1.101200x
(warm/cold, 9/9 wins each). r13 and r9 move from exact losses to resolved
wins without reviving their rejected route prototypes; r12 and r8 also turn
positive. Candidate peak RSS rises by at most about 26 MiB on LAZ-writing
workloads, which is accepted for these gains.

**Decision.** Do not widen further without measurement: wider pools were
marginal on r6 (six/eight/twelve within 2% of four) and B0251 found them
slower or heavier on r14. A width- or count-aware policy is not adopted
because the two-versus-four differences on narrow-record workloads are within
three-pair noise.

**Consequences.** Functional coverage, GPU-native coverage, dispatcher routes,
grammar, placement models, and automatic CUDA envelopes do not change. The r11
matrix gains a case that requires the real writer to observe the four-worker
default without launcher arming; the r14 matrix's launcher-armed two-worker
assertion still holds. Host Debug is 523/523; focused leak-disabled
ASan/UBSan and TSan lanes over the LAZ-writing matrices pass. No CUDA
translation unit changed, so no Compute Sanitizer claim is inferred.

## D0259 — B0256's published statistical-outlier tree is inexact in mutator chains; fix pending direction

B0260 reproduced a byte divergence from pinned PDAL for
`filters.outlier(statistical) -> filters.assign(X = X * 3) ->
filters.neighborclassifier` and reasoned a symmetric divergence for
`producer -> mutator -> outlier -> consumer`. Both stem from D0255: pinned
PDAL's outlier neither invalidates nor publishes a PointView KD3 product, so
its consumers reuse whatever earlier product exists (stale coordinates read
live) or build fresh; the fork's consumers instead reuse the outlier's fresh
published tree. D0255's independent review covered
`normal -> assign -> outlier`, not a consumer after the outlier.

**Decision.** Record the defect as open P1 against priority 1 (default mode
byte-identical to the pinned oracle) and do not add further reuse on top of
it. The r11 reference graph itself has no coordinate mutator and stays exact;
B0258/B0259 results stand.

**Decision (pending maintainer direction).** Two exact resolutions were
measured or reasoned; the maintainer chooses:

1. Restore pinned semantics exactly: the statistical outlier builds a private
   fresh index and neither invalidates nor publishes. r11 loses the shared
   build (about 0.30 s uncached at 1M, or about 0.08 s if the classifier's
   own build is cached-backed).
2. Make cached-coordinate KD3 backing the `PointView::build3dIndex()` default
   with a refresh-at-reuse rule: when an existing cached-backed product is
   returned to a later consumer, its coordinate snapshot is refreshed from the
   live view before use, so a stale tree structure is queried with live
   coordinates exactly as pinned nanoflann does; the outlier then keeps
   pinned semantics (private, unpublished) while every ordinary KD3 consumer
   gains a roughly 4x cheaper build and 3--4x cheaper queries (B0260) at 24
   bytes per point. This needs a pathological-chain differential matrix
   (producer/mutator/consumer permutations versus pinned) before adoption.

Either way, the r6 eigen repair should stop building a full uncached tree for
a few thousand tie rows: with option 2 it inherits the cached default; with
option 1 it may request the cached backing explicitly because no upstream
consumer follows it inside the automatic r6 envelope.

**Consequences.** No product code changed in B0260 beyond stats-only
diagnostics. `IMPLEMENTATION_PLAN.md`, `HANDOFF.md`, and the ledger mark r11's
B0256 reuse as carrying an open exactness defect until one option is taken.

## D0260 — Statistical outlier keeps pinned private-index semantics; pinned-oracle differential lane

D0259 recorded B0256's published outlier tree as byte-inexact in mutator
chains and offered two resolutions. Priority 1 requires the fix regardless of
which broader direction is later chosen, and both options keep the outlier
private, so B0261 takes the common part now.

**Decision.** Supersede D0255's invalidation and publication. The statistical
outlier builds a private fresh nanoflann index and neither reads,
invalidates, nor publishes the PointView KD3 product, exactly as pinned PDAL
does. Its private tree may use the exact cached-coordinate backing (D0237),
because a private transient tree has no reuse lifetime and therefore no
stale-snapshot risk; B0260/B0261 supply the memory/performance proof D0237
required (24 bytes per point transient, 4x cheaper build, 3--4x cheaper
worker passes, r11 5.207843x/5.267489x warm/cold, +27 MiB peak RSS).

**Decision.** Published (`build3dIndex()`) trees stay uncached until a
refresh-at-reuse rule or an equivalent proof exists (D0259 option 2). Reuse
after a non-invalidating coordinate mutator must keep querying live
coordinates through a stale tree exactly as pinned nanoflann does; a snapshot
would diverge. Callers such as `RadiusAssignFilter` and `MathUtils` call
`build3dIndex()` per point, so any refresh must be conditional on a mutation
signal, which does not exist yet.

**Decision.** Add a pinned-oracle differential lane. Registered process
matrices compare against the forked host CLI, and `differential.py` made the
candidate delegate to that same oracle; fork-side changes to upstream stage
code were therefore never differentially tested against pinned PDAL inside
CTest. `PDG_PINNED_ORACLE_EXECUTABLE`, `differential.py --candidate-oracle`,
and `pinned_oracle` matrix cases (candidate delegating to the forked sibling,
oracle pinned) close that gap for r11; other matrices should adopt the same
pattern whenever a slice modifies code under `filters/`, `io/`, or `pdal/`.
The lane skips, and says so, when no pinned build exists.

**Consequences.** r11's B0256 reuse claim is withdrawn; the exact r11 result
is B0261's. Functional coverage, GPU-native coverage, routes, grammar,
placement models, and automatic envelopes do not change. The r6 eigen-repair
tree remains uncached and published pending D0259's broader choice, because a
cached published tree would carry the stale-snapshot risk in forced chains.

## D0261 — Define and ship the record-exact `--fast` contract

The maintainer asked for a `--fast` flag that keeps byte-for-byte exactness in
the normal path while allowing fast mode to drop diagnostics and metadata as
long as the point clouds themselves match the PDAL oracle. When asked to fix
the ambiguous points, the maintainer chose: point order must match; numeric
values must be bit-identical; stdout/stderr text, error paths (message text
and status; still must fail without output), LAS header/VLR metadata, and
`--metadata`/`info` JSON non-record fields may differ.

**Decision.** `pdg --fast <command> ...` is the public entry. The launcher
consumes the leading flag, arms `PDG_INTERNAL_FAST_MODE=1`, and dispatches the
stripped command unchanged; an externally supplied marker is removed before
the engine route (the same rule as the other internal channels). The engine
and sibling read it through `pdg::cli::fastModeEnabled()`. This narrows
`spec.md`'s broader `--fast` language: reordered points or numeric tolerances
would need a further decision.

**Decision.** Fast-mode optimizations are gated on the marker and qualified
with `benchmark_reference.py --contract fast`, whose LAS/LAZ comparison is an
ordered point-record digest decoded through the pinned oracle; non-point
artifacts remain byte-exact and failure paths must still fail. Nothing in
routing changes today because every current route is exact under the default
contract; the record documents plainly that the flag is behavior-neutral until
a measured record-exact/stream-inexact optimization is admitted under it.

**Consequences.** Dispatcher process tests cover flag consumption on both
routes and marker stripping; the runner contract test covers the comparator.
`docs/fast-mode.md` and `spec.md` carry the contract. No performance claim is
attached.

## D0262 — Private cached tie-repair tree only for terminal-sink eigen regions; arm r2 on the engine route

B0260 found that r6's largest remaining cost was building a full uncached
nanoflann tree to repair a few thousand eigen tie rows, and D0260 established
that a *published* PointView KD3 product must stay uncached so a later
consumer after a non-invalidating coordinate mutator observes pinned
nanoflann's stale-tree/live-coordinate behavior.

**Decision.** The resident rewrite emits `pdg_region_terminal_sink` for a
neighborhood region only when the terminal writer alone follows it; the
planner-proved marker is never inferred at runtime. Under that marker the
eigen-family exact tie repair builds a private cached-coordinate
`KD3Index(view, true)` and publishes nothing, because no later stage can
observe a product. Every other graph keeps building and publishing the
uncached tree. This is exact by construction: the only difference is a
product no stage reads. Peak RSS rises by 24 bytes per point during the
stage.

**Decision.** Accept r6 at 8.648309x warm / 8.664582x cold (9/9, exact) and
the aggregates 1.633482x/1.624540x geometric mean and 2.924049x/2.894853x
total wall.

**Decision.** The launcher arms the automatic r2 hybrid marker on the engine
route as well as the command-classified route. B0243's external-`PDG_*`
fail-closed rule had silently stopped arming it under proof environments, so
`pdg_automatic_r2_ground_normalize_cuda_exact` failed from B0243 until this
entry with no CUDA aggregate having been run in between. Marker arming stays
an engine-route hint that the engine revalidates against the literal grammar,
so external injection still cannot widen the selector.

**Consequences.** Every neighborhood wrapper filter accepts the hidden
marker; a wrapper that did not would make preparation fail and silently
select the host executor, which the resident selection matrix exposes.
Functional coverage, GPU-native coverage, routes, grammar, placement models,
and automatic envelopes do not change. The full physical CUDA aggregate
(827/827) and Host Debug (524/524) pass. A rule follows for future slices:
run the complete CUDA aggregate at least once per session before claiming
the tree green.

## D0263 — Cached-coordinate KD3 is the published default; a coordinate epoch refreshes reused snapshots

D0259 offered, and D0260 deferred, making D0237's exact cached-coordinate
KD3 backing the default for `PointView::build3dIndex()`. The blocker was
exactness under reuse: pinned nanoflann reads coordinates live through the
view, so a product reused after a non-invalidating coordinate or order
mutation queries a stale tree with live values, and some callers request the
product per point.

**Decision.** `build3dIndex()` builds the cached-coordinate adapter by
default. The view carries a coordinate epoch moved by every mutation path a
KD3 product can observe (typed and packed field writes for X/Y/Z on the view
or through a view-bound `PointRef`, raw point access, point additions,
appends, and every order mutation), and a reused cached product whose
snapshot epoch differs refreshes its snapshot from the live view before it
is handed out; the tree is never rebuilt, so results equal the uncached
adapter's bit-for-bit. Explicit `build3dIndex(false)` and private
`KD3Index(view, ...)` constructions are unchanged. The engine-owned
`PDG_DISABLE_KD3_COORDINATE_CACHE` restores the uncached default as the
same-final-binary control; test builds honor
`PDAL_TEST_VERIFY_KD3_SNAPSHOT=1`, which fails closed at any reuse whose
snapshot diverged from the live view without an epoch change.

**Decision.** Accept the final r11 result of 7.372457x warm / 7.286586x cold
(9/9, exact) with the disabled control at 5.320564x, and the aggregates
1.672490x/1.663704x geometric mean and 3.073590x/3.042106x total wall. The
write-path cost (one register compare per field write, one pointer load per
X/Y/Z write) is not resolvable on the LAZ-only workloads within the day's
noise.

**Decision.** Exactness proof for this class of change: `Kd3Refresh` units
comparing a reused cached product against an uncached tree built before the
mutation and read live afterwards, the r11 pinned-oracle producer/mutator/
consumer matrix with the verifier armed, and full Host Debug and CUDA
aggregates run with the verifier in the environment. The B0262 terminal-sink
private repair tree remains (it also avoids publishing a product).

**Consequences.** Every upstream and fork host consumer of the published KD3
product (normal, eigen family, nndistance, radial density, LOF, reciprocity,
optimal neighborhood, estimate rank, plane fit, miniball, neighbor
classifier, greedy projection, supervoxel, ICP, DEM utilities) now builds
about 4x cheaper and queries 3--4x cheaper at 24 bytes per point. Functional
coverage, GPU-native coverage, routes, grammar, placement models, and
automatic envelopes do not change. The launcher inventory and both harness
scrub lists carry the two new controls.

## D0264 — Exact fixed-chunk host workers inside the upstream SMRF

Host `filters.smrf` costs about 0.32 s at 1M points and sits in three
reference workflows (r2, r3, r11). B0264's phase timer shows the cost is
dominated by the progressive filter's diamond morphology passes and the void
fills, with the per-point passes second.

**Decision.** Parallelize only what is exact by construction and prove it
against pinned PDAL: (1) each diamond erosion/dilation pass over disjoint
column ranges with the identical fold, comparison, NaN behavior, and the
serial code's carried-over start values, on a caller-owned pool capped at
eight workers; (2) the void-fill query loop over disjoint column ranges;
(3) the classify-ground point pass on fixed contiguous chunks; (4) the
minimum-Z raster on fixed contiguous chunks with per-worker partial rasters
merged in worker order under the serial rule (bounded to 256 MiB of partials).
Everything else in SMRF — view segmentation, temp view construction, KD2
builds, gradient/threshold rasters, diagnostics — is unchanged.

**Decision.** Accept r3 at 1.352415x/1.338430x and r11 at 8.867727x/8.863616x
(warm/cold, 9/9, exact) plus the fourteen-headline aggregates recorded in
B0264. `PDG_DEBUG_SMRF_PHASES` is a retained opt-in diagnostic. The fork's own
SMRF port used by r2's automatic route keeps serial morphology; giving it the
same pooled passes is a recorded follow-up.

**Consequences.** Functional coverage, GPU-native coverage, routes, grammar,
placement models, and automatic envelopes do not change. The SMRF matrix
repeats every tiny case with forced workers and adds three pinned-oracle
lattice cases; Host Debug, focused ASan/UBSan and TSan, and CUDA-tree SMRF/
hybrid lanes are recorded in B0264. No CUDA translation unit changed.

## D0265 — Pool the fork SMRF port's diamond morphology

B0264 left r2 untouched because its automatic route runs the fork's own SMRF
port rather than the upstream filter. The port's progressive filter is the
same algorithm with the same serial diamond passes.

**Decision.** Give the port an internal persistent pass pool (the engine core
does not link the PDAL library) and evaluate each diamond pass over disjoint
column ranges with the identical fold and carried-over start values, capped
at eight workers, only for frames of at least 32,768 cells, honoring
`PDG_NATIVE_WORKERS` and `PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS`. Everything
else in the port is unchanged. Accept r2 at 1.624190x/1.604373x (9/9, exact)
and the aggregates 1.750757x/1.737266x geometric mean and 3.268150x/3.192511x
total wall.

**Consequences.** A large-frame pooled-versus-serial unit test guards the
pool in every tree including TSan. Functional coverage, GPU-native coverage,
routes, grammar, placement models, and automatic envelopes do not change.

## D0266 — Slot-pooled exact reprojection with a streaming batch hook

B0266 attributed most of r8 to two per-point PROJ reprojections executed by
the streaming executor's per-point loop, and showed that per-thread clones
of the GDAL transformation give identical per-point results.

**Decision.** Add an opt-in whole-batch hook to `Streamable`
(`processStreamBatch`) that a stage may use once per reader fill and that
falls back to the unchanged per-point loop when the stage declines. Give
`filters.reprojection` a fixed-slot worker pool (at most eight slots, lazily
created, released in `done()`) with one lazily cloned transformation per
slot for both the batch hook and standard-mode `run()`. Ascending contiguous
slot ranges, slot-ordered appends, post-join skip marking, and lowest-slot
exception rethrow keep output order, dropped-point positions, and the
`error_on_failure` diagnostic identical to pinned PDAL. `where` expressions,
tiny batches, invalid transformations, and clone failures use the serial
upstream loop.

**Decision.** Accept r8 at 2.093332x/2.140982x and r1 at 1.489033x/1.485207x
(warm/cold, 9/9, exact) plus the fourteen-headline aggregates recorded in
B0266. `pdal/private/HostSlotPool.hpp` is the reusable pattern for any
per-point stage whose worker state must stay bound to one thread.

**Consequences.** Functional coverage, GPU-native coverage, routes, grammar,
placement models, and automatic envelopes do not change. The pinned-oracle
reprojection matrix (11 cases) is registered; Host Debug, focused ASan/UBSan
and TSan, and CUDA-tree lanes are recorded in B0266.

## D0267 — Hashed voxel table and pruned probes in the sample filter

B0267 attributed half of r4's remaining wall to `filters.sample`, whose
pinned implementation keys its voxel table with an ordered map, copies every
candidate list it inspects, and probes all 27 neighbor voxels per point.

**Decision.** Keep the greedy algorithm and every decision it makes; change
only the data structure and the probes: an unordered voxel table (only ever
probed by key), in-place candidate lists, reuse of the center probe for the
insertion, and conservative geometric pruning of neighbor voxels whose
box — widened by a 1e-8 x cell margin against index rounding — lies at least
the radius away. Streaming and standard modes, the marker `dimension`,
explicit origins, and the pinned origin quirk are unchanged.

**Decision.** Accept r4 at 5.306974x/5.271122x (warm/cold, 9/9, exact) and
the fourteen-headline aggregates recorded in B0267. The exactness proof is
the new pinned-oracle sample matrix (8 cases, both modes, dense/sparse/
jittered lattices, marker and origins) plus the automatic r4 CUDA lane.

**Consequences.** Functional coverage, GPU-native coverage, routes, grammar,
placement models, and automatic envelopes do not change.

## D0268 — Parallel LAS record packing with slot-ordered summaries and serial repeats

After B0267 the LAS/LAZ-writing headlines shared one remaining serial cost:
`writers.las` packs every point record on one thread (`fillPointBuf`,
scale/offset conversion, flag packing, extra dimensions, and the per-point
`las::Summary` fold), in the streaming per-point loop and in the standard
~1 MB block loop alike. B0259 parallelized only the compression that follows.

**Decision.** Pack records on a fixed-slot pool (at most eight slots,
`hostworkers::rowWorkerCount` policy, `PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS`
and `PDG_NATIVE_WORKERS` controls) in both modes: the standard block path and
a new `Streamable::processStreamBatch` override (surviving rows compacted in
order first). Each slot packs a contiguous run of fixed-size records at their
fixed offsets into its own `las::Summary`; the summaries are merged in slot
order, which reproduces the serial fold bit-for-bit because `BOX3D::grow` is
a strict-comparison fold (first-seen among equal extremes, NaN never enters)
and the counts are sums.

**Decision.** Commit nothing before every slot succeeds. Per-point warnings
(PDRF < 6 Overlap-with-Classification and Classification > 31) are not
logged from workers: a slot that would warn raises a deferred flag and stops;
any flag or any exception discards the run (no summary merged, no bytes
written or compressed) and the same records are repeated with the serial
loop, which logs and throws exactly where pinned PDAL does. Options that can
change record count or order (`discard_high_return_numbers`, `where`), the
streaming first-point auto-offset setup, and runs below 2,048 rows stay
serial. The pool is released with the file.

**Decision.** Accept the final nine-pair gates recorded in B0268 (r14, r13,
r12, r6, r1 warm and cold, all exact) and the fourteen-headline aggregates.
The exactness proof is the new pinned-oracle writer matrix (23 cases, both
modes, formats 1/3/6/7/8, LAS/LAZ, extra dimensions, skips, declines,
warnings, unconvertible scale, controls) with a test-only path trace, plus
the `LasSummaryMerge` unit tests.

**Consequences.** This is exact host-native writer behavior for every
pipeline that writes LAS/LAZ. Functional coverage, GPU-native coverage,
routes, grammar, placement models, and automatic envelopes do not change.

## D0269 — Parallel LAS record unpacking with a reader batch hook

B0268 left the mirror image of the writer's serial cost in place: after
`readers.las`'s pooled tile decode (LAZ chunks or 50,000-record LAS reads),
the main thread still unpacked every record with `loadPoint` — about
0.11 s per million points, present in twelve of the fourteen reference
workloads and dominant in the LAS/LAZ I/O headlines.

**Decision.** Add a reader-side whole-batch hook to the streaming executor,
`Streamable::readStreamBatch(table, pointLimit, read)`: a reader that fills
rows [0, read) itself returns true and `read < pointLimit` means end of input
exactly as a false `processOne()` return would; every other reader keeps
the per-point loop. `readers.las` implements it, and its standard-mode
`read()` takes the same shape per tile: tiles are consumed serially in the
pinned order (same waits, same `queueNext()` cadence), each tile contributes
a contiguous run of records with the rows they fill, and a fixed-slot pool
(at most eight slots, `hostworkers::rowWorkerCount` policy and controls) runs
`loadPoint` on disjoint rows. Standard mode appends the rows serially before
the workers write fields; tiles stay alive until the workers join. Small
runs, `start` in streaming (pinned PDAL's own eof accounting hangs there,
and the per-point path reproduces it), and the read callback stay serial.

**Decision.** Accept the final nine-pair gates recorded in B0269 and the
fourteen-headline aggregates. The exactness proof is the new pinned-oracle
reader matrix (30 cases: formats 0/1/2/3/6/7/8, LAS and LAZ, 1.4 extra-bytes
VLRs and 1.2 named/`use_eb_vlr` extra bytes, multi-tile fixtures, `count`,
`start`, downstream skips, small files, the disable control and natural
policy, both modes, LAS and 17-digit text sinks) with a test-only path
trace, plus the `LasReaderUnpack` unit that runs a 15,000-row stream table
across tile boundaries so a worker run carries two segments.

**Consequences.** This is exact host-native reader behavior for every
pipeline that reads LAS/LAZ through `readers.las`. Functional coverage,
GPU-native coverage, routes, grammar, placement models, and automatic
envelopes do not change. `readers.copc` (r5) is a separate reader and is
unchanged.

## D0270 — Ordered parallel COPC tile decode under `requests=1`

r5 was the last headline at parity. Its wall is `readers.copc` with
`requests=1` — the reference pins one request thread because pinned PDAL
emits tiles in completion order, which is only deterministic with one
thread — so one thread fetched and decompressed every selected node while
the consumer waited (0.237 s of the 0.245 s process; eight request threads
would read the same query in 0.081 s but in an unpinned order).

**Decision.** Under `requests=1` keep exactly one request thread (the
connector sees the same serialized fetches, in the same order, with the
same keep-alive backpressure counted on fetched-but-unconsumed tiles) but
split `copc::Tile::read()` into `fetch()` and `decompress()`, hand each
fetched tile to a decode pool (at most eight threads,
`hostworkers::rowWorkerCount` policy and controls), and emit decoded tiles
strictly in fetch order through a sequence-keyed map. Consumers (`read()`
and `processOne()`) pop the next tile in that order, so points, order, the
first erroring tile's diagnostic, and status equal pinned PDAL's. `requests`
above one keeps the pinned unordered path unchanged; the disable control
restores the pinned single-thread path.

**Decision.** Accept the r5 nine-pair gates recorded in B0270 and the
fourteen-headline aggregates. The exactness proof is the new pinned-oracle
COPC reader matrix (19 cases on a 1.2M-point, 21-node fixture written by
the pinned oracle: full, bounds, bounds+resolution, resolution, polygon,
`count`, empty selection, `keep_alive` 2, the disable control and a
two-worker cap, in both modes, LAS and 17-digit text sinks) with a
test-only path trace, plus the r14 conversion and hybrid reader matrices.

**Consequences.** Exact host-native reader behavior for every `requests=1`
COPC read. Functional coverage, GPU-native coverage, routes, grammar,
placement models, and automatic envelopes do not change.

## D0271 — `--fast` resolves kNN distance ties in device order

The maintainer asked that `--fast` allow order-insensitivity in kNN ties: when
a point has several neighbors at exactly the same distance, pinned PDAL picks
whichever its CPU tree visits first and the device visits them in a different
order; the result is equally correct (same distances, same math), but the
chosen neighbor set — and therefore a normal, an outlier score, a class vote
on those points — can differ. On the reference data that is about 2,300 tie
rows per million in r6, and the exact contract spends about 0.14 s per job
repairing them on the CPU only to match stock's arbitrary choice. Loosening
this under `--fast` drops the repair and lets the device result stand; the
default path stays byte-identical.

**Decision.** Widen D0261's fast contract by exactly this: under `--fast`
the spatial index does not report `KnnDistanceTie`. `pdg::relaxedTieOrder()`
/ `pdg::knnStatusMask()` (`include/pdg/FastMode.hpp`, the engine's only
reader of the marker) drive a per-build `__constant__` mask in both device
gather kernels and the host-index paths, so every identity-observing
consumer (eigen family, HAG-NN, LOF closure, classifier vote, and the rest)
sees exact status for tie rows and runs no host tie repair; incomplete
bounded searches keep their flag and their exact repair. Coordinates,
record count, layout, and order are unchanged; affected rows may carry a
different-but-equally-valid neighbor set or the same set accumulated in a
different order at identical distances (last-bit rounding). Reduction order
and every other numeric path stay exact. The launcher consumes the flag on
every route (a gap on the environment-selected engine route is fixed: the
injected-marker guard now strips only an ambient marker).

**Decision.** The fast comparator (`benchmark_reference.py --contract fast
--candidate-arg=--fast`) compares LAS/LAZ records in order against the first
oracle artifact: identical count, layout, and coordinates; the number of
records whose other bytes differ is reported and bounded by
`--fast-max-differing-records-fraction` (default 1%); every other artifact
stays byte-exact; `reference_suite.py` labels fast aggregates
`contract: fast` with per-workload differing counts and refuses to mix
contracts. A fast performance claim states the differing-record count next
to the speedup and cross-checks it against the exact run's tie-row count.

**Decision.** Accept the r6 and r2 fast gates and the fast aggregates
recorded in B0271 alongside an unchanged exact aggregate. Proof: `FastMode`
and `CudaSpatialIndex` units (only the tie bit changes, both backends),
`PdgNeighborClassifierFilter`/`PdgApproximateCoplanarFilter` fast cases (no
repair, differing rows bounded by the exact ambiguous count), the CUDA-lane
`pdg_fast_tie_contract_cuda` process test (exact repair proven; `--fast`
fails closed under the repair proof gate; `--fast` records differ in
attributes only; default byte-identical), the dispatcher process test on the
environment route, and the runner contract test for the comparator.

**Consequences.** `--fast` is no longer behavior-neutral: it changes r6 and
r2 outputs on tie rows (25 and 125 of 1,000,000 records on the reference
data) and is faster there. The default contract, functional coverage,
GPU-native coverage, routes, grammar, placement models, and automatic
envelopes do not change. `docs/fast-mode.md` and `spec.md` carry the widened
contract.

## D0272 — Retire the automatic r4 CUDA outlier selector; the exact host path is faster

B0227/D0226 selected the CUDA statistical outlier for the literal 1M r4
layout when it measured 3.69x against a serial host path. Since then the
host path gained fixed-chunk worker kNN passes (B0258), the hashed sample
table (B0267), parallel record packing and unpacking (B0268/B0269), and the
CUDA route did not: on the same graph and data `pdg` (route) now measures
0.718 s against the sibling host path's 0.582 s at 1M, and 2.78 s (forced)
against 2.55 s at 4M, byte-identical. The route's cost is the fixed CUDA
startup plus a serial host coordinate gather (0.13 s at 1M) that the host
path no longer has to beat.

**Decision.** Apply the D0239 speed-first rule literally: the r4 selector
no longer selects the CUDA route by default. `rewriteHybridPipeline` keeps
the grammar candidate but selects it only behind
`PDG_EXPERIMENTAL_AUTOMATIC_R4_OUTLIER_CUDA=1`, so the route's exact
differential lane (selection, header summary, refusals, recoverable
fallback) and its proof gate keep exercising it, plus a control that the
literal 1M layout is not selected without the opt-in. Nothing else about the
route changes; the general host path is what `pdg` runs for r4.

**Decision.** Accept the r4 host-path gates and aggregates recorded in
B0272. Optimizing the route itself (parallel host gather) was measured to
tie the host path at best at 1M and is not pursued; a future selector must
be re-measured against the current host path before it can be selected.

**Consequences.** Automatic CUDA envelopes shrink by one bounded literal
route; functional and GPU-native coverage labels do not change (the hybrid
outlier stays functionally covered). This is the first case in the record
of a host slice overtaking a qualified CUDA route, and it is a consequence
the plan's measured-selection rule anticipates.

## D0273 — Banded parallel raster accumulation in `writers.gdal`

r3's remaining wall after SMRF was `writers.gdal` (IDW, 1 m): 0.13 s of the
0.42 s process at 1M input points, a serial per-point radius walk whose
per-cell folds (count, min/max, Welford mean/variance, IDW sums, percentile
bins) are order-dependent only within a cell.

**Decision.** Split the raster into row bands (one per pool slot, at most
eight); bucket every point, in order, into the list of each band its
radius can touch (`GDALGrid::rowSpan`, a conservative superset); each
band's slot replays its list with `GDALGrid::addPoint(x, y, z, rowBegin,
rowEnd)`, the pinned walk with updates restricted to its own rows. Every
cell therefore receives exactly pinned PDAL's sequence of updates.
Standard mode (`writeView`, grid created once from the view bounds) and
streaming with a fixed grid or in bin mode take the banded path through a
`processStreamBatch` override; a streaming grid that grows point by point
in radius mode keeps the per-point path, because pinned PDAL's walk reaches
only the cells that exist when each point arrives (cells added by a later
expansion never receive earlier points) and only the per-point interleaving
reproduces that. Percentile bins live in a per-cell hash map that grows on
first use and stay serial. Governed by the shared host-worker controls.

**Decision.** Accept the r3 and r7 gates and aggregates recorded in B0273.
The exactness proof is the new pinned-oracle GDAL writer matrix (19 cases:
dynamic and fixed grids, all statistics, IDW power/radius, bin mode, window
fill, percentiles, a non-Z dimension with float32 output, upstream skips,
a large radius, bounds clipping, both modes, controls) with a test-only
band trace.

**Consequences.** Exact host-native writer behavior for every raster
publication; functional coverage, GPU-native coverage, routes, grammar,
placement models, and automatic envelopes do not change.

## D0274 — Concurrent, structure-identical nanoflann builds for every KD2/KD3 index

r6's exact-mode attribution after B0273 put 0.109 s of the 0.155 s tie
repair in building the private cached KD3 tree for 2,343 rows; r11 and r4
build the same 1M-point tree once each, and every other host neighborhood
consumer (HAG-NN's compatibility ground index, LOF, SMRF's KD2 fills) pays
the same serial nanoflann build.

**Decision.** Patch vendored nanoflann 1.3.1 with `divideTreeConcurrent`
(the upstream 1.5 pattern): the same recursion, splits, and bounding boxes
as `divideTree`, node allocation serialized through one mutex, and the left
child of a node built on a `std::async` thread while the thread count stays
below `n_thread_build`. The tree is a deterministic function of the points,
so the concurrent build yields the same structure, leaves, and split values,
and every kNN and radius query — including its tie order — visits nodes in
the same sequence. `KD2Impl`, `KD3ImplT`, and `KDFlexImpl` set the thread
count from the shared host-worker policy (at most eight;
`PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS` restores the serial build), and the
cached KD3 snapshot copy is filled on fixed row chunks. The critical path
(the root's serial partition and each level's) bounds the gain: the r6
repair tree falls from 0.104 s to 0.058 s.

**Decision.** Accept the r6/r11/r4 gates and aggregates recorded in B0274.
Proof: `Kd3Concurrency.ConcurrentBuildProducesTheSerialTree` (tie-dense
arithmetic fixtures at 12K and 130K points and a 12,800-point integer
lattice: identical kNN ids/distances and radius results at 1/2/3/5/8
threads) plus every existing pinned-oracle matrix that forces three
workers, which now also forces three build threads (r11, SMRF, outlier,
HAG, LOF, sample-adjacent lanes).

**Consequences.** Exact host-native behavior for every KD index build.
Functional coverage, GPU-native coverage, routes, grammar, placement
models, and automatic envelopes do not change.

## D0275 — Cross-machine, big-cloud, and LAStools benchmark report

The maintainer asked for a full report after the plan's documented
candidates were exhausted: a benchmark on the reference workstation, a
second machine "like a Threadripper" with big point clouds, LAStools
timings for the same jobs, rendered outputs, a plain-language and a
technical summary, and a list of CUDA kernels worth proposing upstream.

**Decision.** Rent one Vast.ai box (AMD Ryzen Threadripper PRO 3975WX, RTX
4090, driver 590.48.01, CUDA 13.3.1 image, Ubuntu 24.04) under the
existing Vast authorization; build the pinned oracle (upstream PDAL 2.10.0 at
`f1e35f5c`) and the fork there from source (`bench/remote/vast_bootstrap.sh`),
and measure with the same runners as the workstation: the fourteen headlines
at 1M in three configurations (default; `PDG_EXPERIMENTAL_CUDA_HYBRID=1`,
because automatic CUDA selection is profile-locked to the reference GPU and
driver; and CUDA-hybrid + `--fast`), and a public 47,478,228-point AHN4
GeoTile (25GN1_01, LAS 1.4 format 8, EPSG:28992) on eight headline graphs
(`vast_run.sh`), the same tile on the workstation, and LAStools timings of
comparable jobs on both machines (`bench/lastools/lastools_bench.py`: the
open-source tools built from source, the proprietary tools as the vendor's
Windows binaries under wine in `-demo` mode, timings only — never treated as
equivalent outputs). AWS was not used; the Vast box already provides the
Threadripper-class CPU the maintainer asked for.

**Decision.** The report (`docs/reports/b0275-cross-machine-benchmark.html`,
generated by `bench/report/build_report.py`, published as an artifact) states
every configuration and caveat next to its numbers: exact-contract results
are byte-identical to the box's own oracle; `--fast` rows carry their
differing-record counts; the box's oracle links different GDAL/PROJ versions
than the workstation's, so only same-box comparisons are exact; the AHN4
runs are two-run (default) or one-run (CUDA-hybrid) medians because pinned
PDAL needs minutes per run there. The upstream-merge assessment ranks the
shared spatial index and exact kNN gather (with its tie/incomplete status
contract), the covariance/eigen kernels, LOF, and the radius family as
proposable; fused point programs and direct LAS I/O as tied to the planner;
and standalone per-stage kernels as not worth proposing (measured slower
than stock end-to-end). The exact host improvements are listed as the
easiest upstream merges.

**Consequences.** New reproducibility surface: `bench/remote/`,
`bench/lastools/`, `bench/report/`, `docs/reports/`. No product behavior
changes; the exact suite claim of record stays B0274.

## D0276 — Scale ladder on a datacenter GPU

The maintainer asked whether the remaining Vast budget could be used to test
bigger point clouds on bigger GPUs "to compare the theoretical speedups at
the limit", expecting the fork to pull further ahead of stock PDAL on the
largest workflows.

**Decision.** Rent one H200 (141 GB) offer under the existing Vast
authorization, build the pinned oracle and the fork there (SM 90; the
automatic CUDA routes remain profile-locked to the qualified RTX 4090, so
the kNN graphs run with `PDG_EXPERIMENTAL_CUDA_HYBRID=1`), and run a
three-rung ladder of public AHN4 tiles — 47M, 95M (two tiles merged), 190M
(four tiles merged) — through six headline graphs, one exact pair per cell
(`bench/remote/vast_ladder.sh`), the pinned side being too slow at 190M for
three-pair medians. Report the result as measured even though it does not
match the expectation: the kNN-heavy graphs plateau at about 10x across the
whole ladder rather than climbing, r2 climbs modestly (1.40 → 1.65x), the
I/O-bound graphs stay at 1.4–3.6x, and the H200's ratios at 47M are lower
than the RTX 4090 box's because the pinned baseline CPU differs. The report
(section 7b) and B0276 state the reading and the caveats (single runs, a
26.9-core cgroup quota, forced hybrid selection, rebuilt oracle) next to
the numbers.

**Consequences.** No product behavior changes; the exact suite claim of
record stays B0274. The remaining Vast credit is about $84.5. Should a
future slice want the ratio to climb with size, the lever is more work per
point (larger neighbourhoods, more features per query) or a device-side
exact repair path, not more points; and a datacenter GPU is not qualified
for automatic selection until a same-machine profile exists.

## D0277 — `pdg calibrate`: locally measured placement profiles for non-reference machines

The maintainer chose, from the post-B0276 recommendations, to let other
machines use the GPU by default: automatic device selection is a
performance promise measured on exactly one physical profile (the embedded
SM-89 record, D0049/D0054 lineage), so every other machine — including the
rented RTX 4090 in B0275, where the forced hybrid took r6 from 2.7x to
11.2x — runs the host path by default. The maintainer's constraint was
explicit: no silent first-run calibration, no behaviour or UX change in the
default path.

**Decision.** Add an explicit command, `pdg calibrate`, that re-measures the
placement calibration cases on the machine it runs on with the same
procedure that produced the embedded profile, and writes a local profile
keyed to that exact machine. Nothing invokes it implicitly. The runtime keeps
spec D4's fixed linear rule with no timing probe, search, or autotuner: the
new inputs are a file, produced offline, and consulted only when its
seven-field machine key (device name, compute capability, kernel driver,
compiled CUDA toolkit, CPU model, logical CPU count, `pdg` version) equals
the current machine's; any mismatch is inert and reported by
`pdg calibrate --status` and `pdg doctor`.

- Measurement: complete-process wall clock, alternating host/device pairs,
  medians; host = the sibling `pdal` (the fork's exact host build, the
  executor the planner would otherwise choose); device = `pdg-engine
  resident` under a calibration-only override
  (`PDG_CALIBRATION_FORCE_DEVICE_PLACEMENT`) that prices every stage as a
  device win so the resident executor runs before any profile exists — and
  under which the automatic `pipeline` route declines, so the override can
  never widen the promise. Every pair's outputs are byte-compared; a model
  whose outputs ever differ, or whose device path never won, is not written.
- Fit: the audit's residual model (`pdg_placement_audit --suggest-models`),
  factored into `PlacementCalibration.{hpp,cpp}` and shared by both, over
  planner-owned terms measured locally (CUDA cold start, host↔device
  transfer, synchronization, index build per persistent byte; LAS packing
  inherited from the reference profile and marked so). Envelope: minimum =
  the smallest measured size where the device won, maximum = the largest
  measured size — fail closed above it. `--append` extends a profile at
  more sizes or models while reusing its coefficient set.
- Scope of models: the shared-neighborhood family and the two point-program
  models of the embedded plan (fifteen). Direct whole-view executors stay
  uncalibrated on non-reference machines. Structural gates are unchanged —
  the profile supplies coefficients and envelopes, not shapes; the
  `extra_dims=all` normal/covariance sink stays bounded to the measured 1M
  layout as on the reference machine.
- Inputs: a deterministic synthetic lidar-like cloud by default
  (`SyntheticCloudGenerator`, LAS 1.4 format 7), or `--input FILE` head
  subsets of the user's own data, which is the recommended way to calibrate
  for a workload.
- Records: `benchmark_reference.py` reports carry the candidate's
  `calibrate --status` so a default-mode measurement states its own selection
  evidence.

**Rejected alternatives.** Loosening the embedded lock to an SM class or
driver range (an untested performance promise); shipping a Vast-qualified
allowlist (D0049's protocol stays available but qualifies one profile at a
time and does not reach the maintainer's own machines); silent first-run
self-tuning (a multi-minute surprise and a hidden behaviour change); pipelined
stage overlap (#2 of the recommendations) — deferred because its neutrality
would have to be engineered rather than fall out of a fold.

**Consequences.** New: `src/cli/Calibrate.cpp`, `src/plan/LocalProfile.cpp`,
`src/plan/PlacementCalibration.cpp`, `src/core/SyntheticCloud.cpp`,
`src/core/CalibrationProbes.cu` (+ stub), `docs/calibration.md`,
`bench/remote/vast_calibrate.sh`, unit and process tests. `pdg doctor` gains
`placement_profile`/`local_profile` lines; the launcher forwards `calibrate`
to the engine like `doctor`. Default-mode outputs remain byte-identical to
the pinned oracle whatever the profile says; the exact suite claim of record
on the reference machine stays B0274 (the embedded profile takes precedence
there). Evidence: B0277.

## D0278 — The resident layout follows a file's extra-bytes dimension types

While measuring B0277 the AHN4 GeoTiles exposed a resident-executor refusal
that no profile can lift: their extra-bytes VLR declares `Deviation` as
uint16 and `Amplitude`/`Reflectance` as double, while PDAL's standard
dimension table (and therefore PDG's generated registry) types them
otherwise. The resident context's layout binding compared every PointView
dimension's physical type with the registry definition of the same name and
threw "resident PointView dimension type differs from PDG", so a calibrated
machine still ran the tile's feature/LOF graphs on the host (2.4x) although
the forced hybrid on the same tile was 16x and byte-exact. Stock PDAL simply
follows the file.

**Decision.** In `bindLayout` (src/pdal/PdgResidentContext.cpp), a
PointView dimension whose name is registered with another concrete type is
now redefined in the per-execution registry to the file's type
(`DimensionRegistry::redefineType`, id and name unchanged, per registry
instance) when no planned stage reads or writes that dimension
(`Plan::summary().touchedDimensions` by name): such a dimension travels in
the physical row through the packed tile layout and is never interpreted
through the PDG definition — packing, boundaries and publication already use
the PointView layout's types and offsets. A dimension the plan touches keeps
the refusal with a specific reason ("... for a device-computed dimension:
NAME") because device stages compute through the planned type, and it falls
back to the exact host pipeline as before. Genuinely unsupported types still
throw. Nothing outside the resident path changes; the host/oracle route
already handled these files.

**Evidence.** Locally, the 47,478,228-point AHN4 tile's plain feature graph
now runs on the resident device executor (23.1 s vs 156.9 s host; placement
device, executor `planner_resident_shared_index`) and its output is
byte-identical to the pinned oracle's; the process test extends
`pdg_calibrate_local_profile_cuda_exact` with a file that carries
`Deviation=uint16` (device route used, identical to host and oracle) and a
file that carries `Rank=uint16` under `filters.estimaterank` (refused with the
device-computed reason, host fallback identical to the oracle). Box
measurements are in B0277.

## D0279 — Drop-in by default: shipped GPU-class profiles, a generic fallback, and calibrate as the tightening tool

After B0277 the maintainer restated the goal: a drop-in replacement for
PDAL — same pipelines, same bytes, faster on the user's machine with nothing
to run first. `pdg calibrate` (D0277) closed the promise gap correctly but
made GPU use conditional on a step stock PDAL does not have. The
measurements behind B0276/B0277 show why a shipped default is defensible:
the device side of every kNN-family case was nearly identical across the
reference 4090, a rented 4090 on another driver/host, a 5090 and an H200,
because the device is not the bottleneck at these sizes; what varies is the
host side. So a profile keyed to the GPU class transfers where the margin is
large and is a guess where it is thin.

**Decision.** Three tiers below the embedded reference profile, consulted in
order after any local profile: (4) **shipped GPU-class profiles**, measured
by `pdg calibrate` on rented machines of that GPU model
(`bench/remote/vast_sweep.sh`), converted by
`bench/report/make_shipped_profile.py` with a shipping margin of 3x — a stage
model is kept only where the device won by at least that margin at every
measured size, and its envelope is bounded to those sizes — embedded from
`data/placement-profiles/*.json` at build time and keyed by exact device
name, compute capability and compiled CUDA toolkit (driver and host recorded
as evidence, not matched); (5) a **generic fallback** for any other CUDA
device inside `applies` bounds (the smallest compute capability and device
memory among the contributing GPUs): the intersection of the shipped
profiles' models under the same 3x margin in every profile, with the slowest
device and fastest host coefficients seen — a 5x margin was tried first and
left only LOF, because one rented host (the 3090's ten-core Xeon E5 v4)
halves every ratio; the intersection over ten GPU classes with worst-case
coefficients is the conservatism, not a taller margin; (3, unchanged) a **local profile** from `pdg calibrate`
takes precedence over both, so calibrate becomes the optional way to tighten
placement to a CPU or dataset. `pdg calibrate --status` and `pdg doctor`
name the tier. Thin-margin models (point programs, neighborclassifier,
everything at sizes below the margin) stay on the host unless calibrated
locally. This deliberately changes the automatic-selection promise for tiers
4–5 from "measured on this exact machine" to "measured on this GPU class on
an EPYC/Xeon-class rented host and admitted only where the device won by
the stated margin"; the exactness promise is untouched.

Also under this decision, two drop-in gaps that no profile could lift:
`normal`+`covariancefeatures` with `extra_dims=all` (the r6 reference graph)
gains its own calibration case and model
(`normal-covariancefeatures-compose-extradims`); a profile that carries it
lifts B0224's one-layout bound, the embedded reference profile keeps it. The
direct whole-view executors (sort/skewness/HAG/radius/outlier direct
compositions) remain uncalibrated off-reference: they need executor-specific
proof flags and mapped uncompressed inputs that the reference workflows do
not present, and none of the fourteen headline graphs reaches them; they are
listed as an open follow-up rather than shipped on inference.

**Rejected.** Loosening the embedded key to an SM class without new
measurements; shipping thin-margin models; a runtime microbenchmark at first
use (spec D4: no timing probe or autotuner in the runtime).

**Consequences.** New: `data/placement-profiles/` (embedded via
`src/CMakeLists.txt` → `ShippedProfilesData.cpp`), tier lookup in
`LocalProfile.cpp` (`shippedPlacementProfileFor`, `genericPlacementProfileFor`,
`placementProfileTier`), `bench/report/make_shipped_profile.py`,
`bench/remote/vast_sweep.sh` and `vast_sweep_proof.sh`, the extradims case
in `pdg calibrate`, tests in `calibrate_test.py` (tier order, generic bounds).
Evidence: B0278.

## D0280 — Shipped GPU-class profiles admit at a 1.7x margin; the generic fallback keeps 3x

The same day as D0279 the maintainer read the B0278 record and asked why the
GPU should need a 3x measured win before it is used by default — first for
the RTX 5090 (whose r6 case measured 2.984x at 1M and was left on the host)
and the RTX 3090 (whose slow rented host left the 47M tile on the host at
2.4x against 8.1x calibrated), then in general: "prefer the GPU at 1.7x or
above; is there a reason not to?"

The reasons examined: the shipping margin exists only to cover the transfer
of a host-side measurement from a rented EPYC/Xeon host to the user's CPU
(the device side is stable within a GPU class). Across the ten rented hosts
the same GPU-vs-CPU ratio spread by about 1.8x (compose at 4M: 3.6x on the
3090's ten-core Xeon E5 v4 to 6.6x on the EPYC 9124), and the reference
Ryzen 9 7900 desktop with a 4090 landed inside the 4090 box's figures (r6
10.69x local vs 11.11x shipped), so a host 3x faster than the rented ones is
not in evidence while 1.7x is about the real spread. The cost of a wrong call
is bounded (a stage admitted at 1.7x on a host 2x faster than the rented one
runs at about 0.85x — a loss on that stage, not a cliff), and every box's
local profile (margin 1, every byte-exact device win admitted) never made
the fourteen-workflow suite slower than the host path beyond run-to-run
noise (B0278). The embedded reference profile and every local profile
already admit at margin 1. Against: the 16M/48M points are single-pair
measurements (±10–20%), so a 1.74x admission may be ~1.5x in truth; and the
generic tier's margin also has to cover an unknown GPU that may be slower
than the slowest swept class (RTX 3060).

**Decision.** Shipped GPU-class profiles are converted at a **1.7x** margin;
the generic fallback stays at **3x**. All ten profiles in
`data/placement-profiles/` were regenerated from the archived B0278
calibrate evidence (`make_shipped_profile.py`, whose default is now 1.7 for
a class profile and 3 for `--generic`; the margin is recorded per profile as
`shipping_margin`); the generic profile is unchanged. Net effect: every
single-feature filter (normal, eigenvalues, covariancefeatures,
approximatecoplanar) is admitted from 1M instead of 4M on every class; the
compose graphs, rank/optimal and optimalneighborhood from 250K on most; the
RTX 5090 admits the r6 extradims composition from 1M (was 4M) and the RTX
3090 admits the compose models 1M–48M (was 4M only / never); nndistance
widens; `neighborclassifier` remains unadmitted everywhere and the point
programs remain 4M-only on most classes because the measurements, not the
margin, exclude them. Minimum admitted win per class ≥ 1.70x, medians
2.7x–5.2x. The tier-4 promise wording becomes "measured on this GPU class on
an EPYC/Xeon-class rented host and admitted only where the device won by
1.7x at every measured size".

**Rejected.** Margin 1 for shipped profiles (single-pair noise at the big
sizes and the transfer risk argue for keeping some margin); lowering the
generic margin (unknown-GPU risk); per-class margins (the maintainer's rule
is general).

**Consequences.** No new measurement: the B0278 proof was run with the 3x
set, so its shipped-tier numbers are a floor for the 1.7x set, and each
box's local-profile runs (a superset of the 1.7x admissions) are the nearest
evidence for what the wider admissions do (5090 r6 8.66x at 1M; 3090 AHN4
features 8.06x, r6 graph 8.18x, LOF 18.77x). A re-proof sweep at 1.7x is the
open item if the maintainer wants the record to carry it (about $22 and a
few hours on Vast). Also fixed here: `src/CMakeLists.txt` now lists the
profile files as `CMAKE_CONFIGURE_DEPENDS`, so a content-only change to a
profile re-embeds it (before, only adding or removing a file re-ran
configure). Evidence: B0279.

## D0281 — An exact-machine profile may not shadow newer same-class placement evidence

The independent B0280 audit found a precedence bug on the reference machine.
`placementCalibrationFor` correctly chose the exact-machine embedded SM-89
profile before the shipped RTX-4090 profile, but the embedded profile was
older: its `normal-covariancefeatures-compose` envelope ended at 4M points and
it had no `normal-covariancefeatures-compose-extradims` model. The shipped
profile already carried both compositions through 48M. Consequently the
ordinary 35,976,465-point VEIL r6 graph reached the safe
`mixed_calibration_models` refusal and delegated to the host. That fallback
was the correct fail-closed response to the profile it was given, but the
profile state was wrong: a higher-priority record had silently discarded
newer applicable evidence for the same GPU class.

**Decision.** Refresh the embedded profile as
`sm89-2026-08-17-r6-large-layouts`. Retain its exact-machine coefficients and
50K floor for the plain normal/covariance composition and add the separately
named `normal-covariancefeatures-compose-extradims` model at a 250K floor. Both
end at 47,478,228 points: the largest independently rerun exact-machine row,
not the shipped profile's rounded 48M same-class bound. A higher-priority exact-
machine profile must carry every same-device-class model name, but its envelope
must remain bounded by its own device-winning evidence rather than copied from
another host.

The extra-dimensions publication still gets its own model. Its automatic shape
is limited to LAS formats 6, 7, and 8; at least the standard record width;
input records no wider than 63 bytes; and output records from 100 through 127
bytes. Compressed and uncompressed input/output combinations inside that family
are qualified. Legacy formats, waveform formats, malformed narrow records, and
wider records remain host-selected until separately proved. Writer stride,
packing, transfer, memory budget, plan-shape recognition, and resident preflight
remain runtime facts.

The enlarged plain-publication route is narrower: it admits only B0187's
uncompressed format-7/36 input or B0280's compressed format-8/44 AHN4 input,
both writing uncompressed format-7/36 output. Point-count admission alone never
authorizes another physical layout.

Three regression rules make that maintenance requirement executable. The
profile unit requires every shipped RTX-4090 model name in the higher-priority
embedded record. The compiled placement-model audit rejects an embedded model
that lacks a calibration case or whose explicitly declared maximum exceeds its
largest recorded device-winning row. A registered provenance test requires
every case to name a report; it requires retained reports to remain present,
hash-verifies them, and semantically binds qualified complete-process reports
to their case's exactness, timing, point/layout, pipeline/input/output, active-
profile, and required-route fields. The reference benchmark harness now scrubs
ambient loader variables and resolves and hashes `pdg-engine` and the
`libpdalcpp` selected by `ldd` under the measured environment, as well as the
public launcher, so a stable launcher can no longer hide a changed
implementation.

**Rejected.** Leave the 2.55x host fallback because it was safe; reorder the
shipped tier ahead of an exact-machine local record; force every r6 layout to
CUDA independent of profile/envelope/preflight; or add a runtime timing probe.

**Consequences.** Default VEIL r6 now automatically selects the resident
shared-index CUDA executor and remains byte-identical to clean pinned upstream.
The official complete-process pair is 313.166 s upstream versus 21.308 s
candidate, **14.697x**, with the required automatic-route proof switch active.
At the 47,478,228-point ceiling, the plain graph is 364.950 s versus 21.186 s
(**17.226x**) and the `extra_dims=all` graph is 410.961 s versus 32.188 s
(**12.767x**), again exact and guarded against fallback. Deterministic format-
6/8 and carried-Extra-Bytes matrix cases exercise the admitted shape family;
explicit out-of-family controls continue to refuse it. Evidence: B0280.

## D0282 — Automatic acceleration and benchmark provenance fail closed after publication

B0280 repaired the stale placement profile, but a final independent review
found two acceptance gaps. First, benchmark loader resolution returned no
component on `ldd` failure and the retained verifier did not require the new
roles. Second, a storage-pressure reproduction showed that the unchanged PDAL
LAS writer can leave a partial, unfinalized file yet return success. The exact
benchmark comparison rejected that artifact, but the public `pdg pipeline`
command itself had returned zero.

**Decision.** A real ELF reference benchmark must identify the candidate
launcher, adjacent ELF `pdg-engine`, and the `libpdalcpp` actually resolved for
both candidate and oracle under their measured loader environments. Failure to
run or parse `ldd`, a missing engine, a missing library, or aliasing component
paths rejects the benchmark. Ambient `LD_*` is scrubbed; any explicit loader
override is recorded. A qualified placement report records its physical LAS
input/output facts in the original generated JSON, and the registered verifier
requires the loader scrub, exact component roles and hashes, route proof,
profile, timings, and pipeline/fixture/artifact bindings.

After any accepted automatic resident execution publishes a LAS/LAZ file,
`pdg` maps it read-only and validates the public header plus VLR/EVLR extents
before returning zero. For uncompressed LAS, the advertised point count and
record stride must fit completely in the file. A nonempty compressed output
must contain a point payload. This check occurs after output side effects are
committed: failure returns nonzero and never delegates the original pipeline,
which could overwrite or duplicate side effects. The existing direct publisher
remains atomic; this rule covers routes that retain PDAL's writer.

**Rejected.** Treat process exit zero as sufficient publication evidence;
silently omit unresolved loader components; relabel old reports as having
captured evidence they did not capture; fall back after a writer side effect;
or validate by scanning/hashing the complete point payload on every ordinary
command.

**Consequences.** The deterministic publication-truncation process case now
fails nonzero. All three B0282 qualification pairs pass with the hardened
launcher/engine, exact clean-upstream artifacts, required automatic-route
switch, and fail-closed loader evidence: VEIL r6 16.018x, AHN4 plain 17.521x,
and AHN4 r6 13.192x. B0280 is retained unchanged; active calibration provenance
points to B0282. Evidence: B0282.

## D0283 — Validate LAZ publication structurally, not by a second full decode

D0282's requirement that a nonempty LAZ merely contain point-payload bytes did
not prove that a writer had finalized the complete compressed publication. A
read-only review supplied the concrete counterexample: retain the header and a
prefix of compressed points, truncate the rest, and the shallow test could
still return success. The first repair decoded all declared points and closed
the correctness gap, but B0283 measured a VEIL candidate regression from about
20 seconds to 69.849 seconds. That duplicated an O(points) operation on every
successful compressed-output command.

**Decision.** After the public header and VLR/EVLR extents parse, a compressed
publication must also have a finalized LASzip chunk-table pointer, supported
table version, and bounded chunk count. Lazperf must parse its LASzip VLR and
chunk table. Fixed-size tables must contain exactly the number of chunks needed
for the declared points; variable-size tables must sum exactly to the declared
point count. In either form, every chunk must be nonempty and the sum of its
relative compressed byte sizes must equal exactly the interval from the first
chunk through the chunk-table offset. This check is O(chunks) in time and
memory. The exact differential harness continues to hash and compare the whole
artifact; this publication guard is a fail-closed completeness check, not a
replacement for the oracle.

**Rejected.** Header-plus-nonempty-payload validation; accepting a missing or
`-1` chunk-table pointer; decoding/hashing every point a second time on the
ordinary command path; or retrying the host pipeline after publication side
effects.

**Consequences.** The physical process test now truncates a published LAZ
midway through its compressed payload and returns nonzero, while complete LAZ
outputs pass without the B0283 regression. Final route-proof B0284 pairs are
byte-exact and measure VEIL r6 15.706x, AHN4 plain 17.996x, and AHN4 r6 13.441x
against the independently built pinned upstream oracle. Active calibration
provenance points to B0284; B0282 and rejected B0283 remain historical evidence.
Evidence: B0283, B0284.

## D0284 — Placement admits retained physical tuples and publication counts match execution

A final independent review found two remaining fail-open boundaries. D0281's
format/width ranges formed a rectangle: they admitted unmeasured combinations
such as format-6/31, format-8/38, format-8/63 -> 7/127, plain inputs, and
compression-swapped outputs even though retained large-file reports proved
only format-6/30 -> 7/100 and format-8/44 -> 7/120 compressed tuples. Separately,
D0282/D0283 validated a raw LAS extent against its own advertised count. An
unfinalized header advertising zero points plus a partial payload therefore
still passed, which was the original storage-pressure failure shape.

**Decision.** A placement profile supplies cost coefficients and a cardinality
envelope, never a Cartesian product of physical layouts. Automatic
normal/covariance `extra_dims=all` placement admits only these tuples:

- exactly 1,000,000 format-7/36 compressed input to format-7/100 LAS or LAZ
  (B0223/B0224);
- format-6/30 compressed input to format-7/100 LAZ (VEIL, B0285);
- format-8/44 compressed input to format-7/120 LAZ (AHN4 r6, B0285).

The plain composition remains limited to uncompressed format-7/36 or compressed
format-8/44 input writing uncompressed format-7/36. Any other format, stride,
compression, or emitted-width tuple remains host-selected until retained
same-machine evidence and physical tests qualify it.

Every automatic LAS/LAZ publication must advertise exactly the output point
count observed by the resident executor. FileView's raw extent check remains,
and compressed publications retain D0283's exact chunk-table point/byte
coverage. Missing observed cardinality fails nonzero. These checks occur after
side effects, so failure never retries the host writer.

**Rejected.** Width-range interpolation across physical layouts; treating a
unit placement-facts test as physical evidence; assuming trailing bytes prove a
raw LAS header was finalized; or claiming run-to-run noise from a one-pair
benchmark with no spread. B0284's noise wording is corrected by B0285 without
rewriting the historical entry.

**Consequences.** The expanded physical matrix passes 28 cases, including
explicit tuple/layout assertions, neighboring unmeasured refusals, a
mid-payload LAZ truncation, and a partial plain LAS with stale zero count. Final
route-proof B0285 pairs are byte-exact at VEIL r6 15.518x, AHN4 plain 18.243x,
and AHN4 r6 13.372x against the independently built pinned upstream oracle.
Active calibration provenance points to B0285; prior reports remain historical
evidence. Evidence: B0285.

## D0285 — Qualified physical tuples use bounded cardinality models

D0284 closed layout interpolation but left implicit whether its two large
physical tuples were admitted only at the retained corpus counts. An
identity/count gate would make an otherwise identical format-6/30 VEIL-family
tile fall back merely because it contains fewer points, recreating the class of
selection failure that prompted this audit.

**Decision.** Physical-layout admission and point-count placement remain
separate. The format/stride/compression/output tuple must exactly match one of
D0284's qualified tuples. Within that tuple, the active machine profile may
estimate host and device cost across its bounded cardinality curve. The
embedded SM89 extra-dimensions model is bounded to 250,000--47,478,228 points;
below or above it remains host-selected. No interpolation across point format,
record width, compression, output layout, graph, or writer options is allowed.

**Consequences.** Placement units select both format-6/30 and format-8/44
tuples at 274,625 and 4,000,000 points and retain outside-envelope controls.
The public physical matrix proves the 274,625-point format-6/30 route exact and
the final full-size B0285 reports prove the named VEIL and AHN4 pairs. Interior
selection is a profile prediction, not a published speedup: each public
performance claim still requires a same-machine pinned-oracle measurement and
retained report. Evidence: B0285 and the resident-selection matrix.

## D0286 — Public proof is versioned, frozen, population-scoped, and reproducible

The August 2026 reports contained strong exactness and cross-machine evidence,
but four distinct claims were too easy to blur: repository-run comparison,
release-artifact proof, population generalization, and unrelated third-party
validation. The 1.7x shipped profiles had also not been rerun on the ten sweep
machines after the historical 3x proof, and the 47M exploratory benchmark used
one timed observation per cell.

**Decision.** Name and version the complete-process compatibility lane as the
PDG Conformance Suite. Its default comparator is byte equality over files,
streams, status, and deterministic repeats. `copc-canonical-v1` is an explicit
supplemental semantic comparator allowed only with a per-case pinned-oracle
nondeterminism reason; it does not relax the default product contract. Expand
the checked-in bounded recipe to exactly 2,048 stable cases rather than use an
open-ended random generator, and reject undeclared paths, network stages, and
unbounded artifacts.

Close the exact runnable candidate, independently built pinned oracle,
placement profiles, benchmark code, pipeline bytes, and frozen-time helper in
`pdg-frozen-release-artifact-v1` before performance timing. Any added, removed,
or changed payload byte fails both the pre-run and post-run checks. The ten-GPU
1.7x re-proof must use that payload without rebuilding and at least three
measured pairs, including large-cloud rows. The active shipped profile id,
embedded-source filename, and compiled source SHA-256 must match the closed
profile bytes before timing.

Preregister the 3DEP population experiment before selection or timing: 40
unique projects, deterministic performance-blind project/tile ranking, one
observation per project, exact r6 bytes, RTX 4090, one warm-up and three pairs,
and success only at `PDAL/PDG >= 10`. Missing, failed, inexact, and slow
attempts are failures. Acceptance is a one-sided 95% exact Clopper–Pearson
lower bound strictly above 0.90; 29/29 is the minimum all-success result and
40/40 yields about 0.9278. Acceptance regenerates the selection from the
frozen catalog and recomputes medians and speedups from successful, finite raw
run records rather than trusting asserted summaries.

Package `pdg verify` to produce local JSON and self-contained HTML containing
binary/library/helper hashes, versions, device/host/driver facts, input and
pipeline hashes, placement status, raw timings, exactness, output hashes, and
the rerun command. It never uploads or contacts validators, and it states that
an author-run artifact is not unrelated third-party validation.

**Consequences.** The implementation and focused infrastructure contracts are
release-ready evidence machinery, not new performance measurements. The ten-
machine 1.7x rerun, the selected/downloaded 3DEP corpus and 40-project result,
and unrelated-user reproductions remain explicitly open. No `BENCHMARKS.md`
entry or performance claim is created until those gates actually run with
retained raw artifacts. Historical reports keep their original values and are
relabeled at the claim boundary rather than retroactively strengthened.

## D0287 — The frozen 1.7x re-proof replaces calibration-based public expectations

D0286 required the exact 1.7x shipped-profile artifact to be timed without a
worker rebuild. B0286 completes that gate and contradicts two convenient
expectations in the earlier plain-language report: a universal 7--12x 1M r6
band and an RTX 3090 result near 2.5x regardless of placement.

**Decision.** Lead public performance material with the B0286 frozen-artifact
results. At 1M, state that seven of ten r6 cells measure 7.4--10.2x, RTX 3060
and RTX 4090 measure 6.363x and 6.531x, and RTX 3090 is the explicit exception
at 0.986x. State the complete fourteen-workflow total-wall range of
0.987--1.430x; do not promise that every user receives a CPU-side speedup.

Limit the 47,478,228-point r6 claim to the seven actually measured classes:
A10, A100 40 GB, L40S, RTX 3090, RTX 4090, RTX 5090 and RTX A6000. Their exact
three-pair median range is 6.813--11.601x. Do not imply a result for RTX 3060,
RTX 4060 Ti or RTX 4080 at that size, and do not turn seven hardware cells on
one AHN4 tile into a 3DEP-project population claim.

Use the same conformance formulation in every front-page location: PDG
preserves pinned-PDAL semantics; deterministic outputs are byte-identical,
while an inherently nondeterministic container is compared with its explicit
canonical comparator. Physical COPC hierarchy assignment and coarse previews
are diagnostic-only, but full canonical points, semantic metadata, hierarchy
invariants and the exact bounded query remain gating. A release-wide pass also
requires every process to return its declared status; matching crashes do not
pass by accident.

Retain B0278's 3x tables only as labelled historical evidence. Calibration
results remain placement inputs, not substitutes for a shipped-file timing.
The preregistered 40-project 3DEP study and unrelated-user reproduction remain
open; `pdg verify` prepares evidence and must continue to record author-run
validation as `not-performed` externally.

**Consequences.** The most prominent 1M statement is narrower but defensible,
the RTX 3090's 0.986x/8.944x scale dependence is explicit, and 47M is never
described as a ten-card sweep. B0286's hardened conformance result supersedes
the rejected permissive diagnostic that counted 92 matching failures. The
frozen rerun cost about $8.28 including rejected preflight deployments.

## D0288 — GPUPAL is the public product and drop-in command

The PCPU working name and `pdg` public command no longer matched the selected
release identity. The owner selected **GPUPAL — GPU Pointcloud Abstraction
Library** and required `gpupal` itself to be the drop-in command for `pdal`.
Renaming every internal namespace, environment variable, calibration id, and
historical evidence artifact would create an ABI- and proof-wide change with
no user benefit before the first release.

**Decision.** The public product, repository, and executable are GPUPAL,
`GPUPAL`, and `gpupal`. The `gpupal` launcher accepts PDAL commands and
pipelines and retains the existing exact fallback contract. Internal C++
namespace and target names, `PDG_*` controls, build preset names, test data,
helper names, and frozen historical evidence retain `pdg` until a separately
scoped compatibility migration proves value. Historical append-only decisions,
benchmarks, and retained reports keep the spelling under which they ran.

The intended npm entry point is the unscoped `gpupal` package. It may publish
only after release automation supplies an immutable, checksummed native bundle
containing `gpupal`, its internal engine, and the pinned sibling PDAL fallback;
the development manifest fails closed. No npm install may silently compile a
different dependency stack or download an unhashed mutable asset.

**Consequences.** Public docs, diagnostics, configuration defaults, CI paths,
and new package metadata use GPUPAL/`gpupal`. The Python import and public C++
API names remain an explicit release-scope decision. Name/trademark clearance,
third-party licensing, native packaging, release CI, and npm ownership remain
gates recorded in `RELEASE_READINESS.md`; selecting the name does not claim
those gates have passed.

## D0289 — GPUPDAL supersedes GPUPAL as the public release identity

After selecting GPUPAL in D0288, the owner selected **GPUPDAL — GPU Point Data
Abstraction Library** as the clearer public name and reported that Howard
Butler gave permission to use `GPUPDAL`. The project has not yet made a public
release, so changing the public identifiers now avoids a post-release command,
package, and repository migration.

**Decision.** The public product, repository, executable, configuration
directory, and intended unscoped npm package are GPUPDAL, `GPUPDAL`,
`gpupdal`, `~/.config/gpupdal`, and `gpupdal`, respectively. D0289 supersedes
D0288 only for those public identifiers. Internal C++ namespaces and targets,
`PDG_*` controls, build presets, helper paths, raw benchmark tool ids, and
immutable historical evidence retain `pdg`. Append-only decisions and retained
reports keep the names under which they were produced; regenerated presentation
material may label the current product GPUPDAL while disclosing a historical
`pdg` benchmark id.

The reported permission resolves the owner's naming choice. Before public
visibility, retain the permission in a durable project record and confirm its
scope and the non-endorsement wording in `RELEASE_READINESS.md`. The npm
registry returned not-found for the unscoped `gpupdal` name on 2026-08-21; that
availability check does not reserve the name.

**Consequences.** Public source, diagnostics, packaging metadata, CI workflow
names, documentation, and new release assets use GPUPDAL/`gpupdal`. Existing
GPUPAL public identifiers are removed before the first release. This is a
branding and distribution change only; it does not alter compatibility,
exactness, stage coverage, or benchmark measurements.

## D0290 — Releases are local/manual and Linux-first

The owner does not want GitHub Actions or paid hosted CI and expects the
project to own native packaging rather than making release assembly an owner
prerequisite. The long-term platform goal is Linux, Windows, and macOS, but
only the Linux workstation is currently available for validation. npm
provenance cannot be produced by a local manual publish from a private source
repository.

**Decision.** Disable GitHub Actions for the hosted GPUPDAL repository and
remove the GPUPDAL-specific hosted workflow. Preserve inherited upstream
workflow files as inert upstream material. Release gates run locally with
retained logs and exact commit identities; lack of a hosted runner is not a
blocker.

The first native artifact is CLI-only Linux x86-64. It is assembled by the
maintained `gpupdal_linux_bundle` target and contains launchers, `gpupdal`,
`pdg-engine`, sibling `pdal`, discovered runtime libraries and data, notices,
checksums, a dependency map, and an SPDX 2.3 SBOM. The workstation artifact is
a developer candidate until the same recipe uses a declared oldest-supported
distribution and passes clean-machine tests. Windows and macOS support require
real-machine differential and install evidence; current macOS is CPU-only
because supported contemporary CUDA toolkits are unavailable there.

npm publication is interactive from the owner's 2FA-protected npm account.
The package's provenance setting remains off until a supported publishing
environment and public source repository are intentionally adopted. A public
npm package may not point at an asset that anonymous installers cannot read.

**Consequences.** GPUPDAL does not incur hosted CI cost, and release acceptance
still requires the same conformance and sanitizer evidence. The local bundle
recipe exposes the exact distribution-specific shared-library closure for
license review instead of treating compilation as portability. Windows,
macOS, CUDA binary variants, and npm publication remain explicit later gates.

## D0291 — Preserve the Autzen raster values with lossless compression

The inherited `test/data/autzen/autzen-surface.tif.min.tif` fixture occupied
57,053,532 bytes and triggered GitHub's warning for blobs larger than 50 MB.
The programmable HAG fixture depends on its Float32 samples, geotransform, CRS,
extent, and `-9999` no-data value, so substituting an unrelated small raster
would weaken or invalidate the input.

**Decision.** Re-encode that same raster with tiled DEFLATE compression and
floating-point prediction. The raster remains 3250 by 4386 pixels with the
same one-foot geotransform, CRS, Float32 band, no-data value, and GDAL checksum
60319; its tracked size becomes 305,336 bytes. No golden output is regenerated
and no test semantics are changed.

Do not rewrite Git history implicitly. The old 54.41 MB blob remains in the
short private repository history until the owner separately approves a
history rewrite and force-push before public visibility.

**Consequences.** New source archives and commits carry the 305 KB lossless
fixture while preserving raster behavior. Fully removing the historical blob
is a transparent owner decision rather than a side effect of fixture cleanup.

## D0292 — Keep BSD-3-Clause and record owner authority

On 2026-08-21, Zy selected BSD-3-Clause for GPUPDAL-owned work and confirmed
authority to license the existing project-specific code, tests, documentation,
generated assets, and modifications. Upstream and third-party components retain
their own licenses and attribution. The existing inbound=outbound contribution
policy remains appropriate while BSD-3-Clause is the settled policy; no DCO or
CLA is added for the first release.

The public security and conduct contact is `zy@automagics.com`. A response-time
target remains an owner decision before public visibility. If future dual
licensing becomes important, contributor terms must be reconsidered before
accepting outside contributions whose relicensing rights would be needed.

## D0293 — Replace the private hosted history with a clean root snapshot

Zy approved rewriting the short private GPUPDAL history on the condition that a
smaller replacement test fixture exist. D0291's losslessly compressed 305 KB
Autzen raster satisfies that condition and preserves the original samples and
geospatial behavior.

**Decision.** Before public visibility, replace the private repository's
reachable `main` history with one root snapshot of the reviewed current tree.
Use a force-with-lease tied to the known private `main` commit, retain no public
tag or release pointing to the old history, and verify a fresh clone contains
the compressed fixture but not the 57,053,532-byte blob.

**Consequences.** The hosted public-history candidate is compact and contains
all current attribution, decisions, tests, and release preparation. Local
development remotes unrelated to `zymazza/GPUPDAL` are not rewritten.

## D0294 — Make GPUPDAL name non-endorsement explicit

BSD-3-Clause already contains a non-endorsement condition. Zy requested that
the GPUPDAL-specific grant name GPUPDAL, Zy Mazza, and Automagics explicitly so
derived products cannot use those names to imply endorsement or promotion
without specific prior written permission.

**Decision.** Add a separate GPUPDAL-specific BSD 3-Clause grant with the three
names in its third condition, and retain the upstream PDAL/Hobu license text
verbatim as its own grant. This remains BSD-3-Clause: it does not restrict
commercial use, truthful attribution, or factual compatibility statements.
Trademark registration and enforcement, if pursued later, are separate from
this copyright license condition.

## D0295 — Confirm the first binary scope and best-effort contact target

Zy confirmed a CLI-only Linux x86-64 first binary with optional external
plugins disabled, provided it remains a `pdal` drop-in. The configured release
already enforces identical `gpupdal --drivers` and sibling `pdal --drivers`
output, and default mode delegates any unaccelerated configured stage to that
exact sibling implementation. Optional plugin source remains in the repository;
plugins can enter later artifacts only with their dependency, license, platform,
and clean-install gates.

The npm account is `zymazza`; authentication was verified without publishing a
package. npm's August 2026 policy permits direct publication without account
2FA when a narrowly scoped granular write token has **Bypass 2FA** enabled, so
account 2FA is recommended but is not a GPUPDAL release requirement. No
publication credential is retained in the repository. The public
security/conduct contact remains
`zy@automagics.com`, with a five-business-day best-effort acknowledgement goal.
That goal is explicitly not an SLA, support contract, warranty, promise of a
fix, or modification of the BSD **AS IS** disclaimer.

## D0296 — Pin the first Linux release environment and exact oracle build

The workstation-linked developer archive did not establish an oldest-supported
glibc baseline or a reviewable geospatial dependency closure. The release lane
now builds in the pinned official Debian 12 image
`sha256:abd67ffcfa541b485a3dff59865ab629aa048a6c613e639d36e7456b0b229241`,
uses checksummed CMake 3.28.6 and GDAL 3.8.5 sources, and builds GDAL with the
bounded driver set, command-line applications, and GEOS support needed by the
exact differential corpus.

The lane creates a true detached checkout of the commit in
`cmake/pdg-oracle.cmake` and builds that checkout as the oracle. A source archive
or nested checkout of the current tree is insufficient because PDAL embeds its
source revision in observable LAS metadata. GPUPDAL compilation and packaging
run with networking disabled, at no more than two compile jobs. The current
CPU-only archive contains 51 runtime dependency rows, 45 copied system notice
sets, 47 SPDX packages, 290 files, 337 relationships, and no missing-license
marker. Its internal and outer hashes, relative runtime paths, configured driver
parity, private-path scan, and non-root bare-Debian startup pass. The controlled
unit result is 447 passed with two optional local-corpus skips; the exact
differential result is 98/98. The host ASan/UBSan lane passes all 454 applicable
tests with five intentional fixture-dependent skips and no sanitizer finding;
the preset disables LeakSanitizer because the managed workstation tracer cannot
host it, while retaining address and undefined-behavior checks.

**Consequences.** Debian 12/glibc 2.36 is the oldest-supported baseline for this
candidate, and the prior broad workstation GDAL closure is not a release input.
This closes the reproducible CPU bundle and dependency-notice work; it does not
qualify CUDA artifacts, the full sequential upstream suite, or a public version.

## D0297 — Bind neighborhood covariance to the pinned PDAL arithmetic

A clean GCC 12 release build exposed two independent weaknesses: the resident
covariance path's hand-written arithmetic was not bit-identical to the pinned
PDAL `pdal::math::computeCovariance` implementation for every kNN row, and the
test's second hand-written oracle could agree or disagree based on its compiler
context. Compatibility mode now uses a separately attributed covariance helper
with the pinned upstream operation order and Eigen expression shape. The unit
test links the pinned `pdal/private/MathUtils.cpp` implementation itself and
constructs a PDAL `PointView`, so the oracle is no longer another transcription
of the product code.

The same clean build diagnosed GCC 12 `-Wmaybe-uninitialized` false positives
around six direct-placement `std::optional` values. Those values now use the
existing `NoResidentRegion` sentinel after explicit successful validation;
placement selection and calibrated model names are unchanged.

The published upstream-suite build under GCC 16 also rejected `GDALWriter`'s
inline constructor because exception cleanup instantiated a `unique_ptr` whose
private `BandWorkers` pointee was still incomplete. Its constructor and virtual
destructor now live after the private type definition in `GDALWriter.cpp`;
object state and writer behavior are unchanged.

After that portability fix, all 142 tests in the sequential published upstream
PDAL suite pass. The two network-dependent STAC/COPC tests were rerun against
their official public fixtures after the initial network-isolated run passed
the other 140.

**Consequences.** Default neighborhood covariance is byte-gated against the
actual pinned upstream helper. The retained upstream BSD header and existing
NOTICE attribution cover the derived product helper. The GDAL writer ownership
change is source-level compiler portability only. No performance claim or
automatic-selection envelope changes.

## D0298 — Make npm independent of source visibility and keep verification optional

A public npm installer cannot anonymously fetch a release asset from a private
GitHub repository. A second download also adds mutable hosting, lifecycle-script,
and system-`tar` failure modes. The release staging command now extracts the
checksummed Linux bundle into `native/linux-x64` before `npm pack`; npm's
immutable package tarball carries that complete tree. Prepack validates the
release manifest, required executables, and every entry in the bundle's
`SHA256SUMS`. The installed `gpupdal` command therefore needs neither GitHub
access nor a postinstall script.

The core PDAL-compatible CLI has no Python runtime dependency. The optional
`gpupdal verify` evidence command remains a standard-library Python helper and
now reports clearly when `python3` is absent from `PATH`; its process-boundary
test covers that status-127 contract. Bundling a private Python runtime would
materially enlarge and complicate every install for an auxiliary command.

**Consequences.** Private source visibility no longer blocks `npm i gpupdal`.
The checked-in development manifest remains intentionally empty and unpublishable;
final version selection, a clean-commit native build, staged pack/install tests,
and explicit owner authorization still precede any registry publication.

## D0299 — Rehearse the private-source npm delivery path without publishing

Clean commit `984e2d21aae51677da1f0cc8bd22f7f7da4ae301` produced the controlled
Debian 12 archive `gpupdal-0.0.0-test.1-linux-x64.tar.gz` with SHA-256
`d1661a0cf0ce2703af62c126afeb720ccd033c9cc61d4f95fc1918709b6aa812`.
The archive passed a read-only, non-root bare-Debian smoke after the smoke
tmpfs was explicitly mounted executable. Its staged npm package verified all
291 native checksum entries; `npm pack --dry-run` reported 300 total entries,
51,482,737 compressed bytes, and 130,510,405 unpacked bytes. Installing the
actual tarball into an empty temporary prefix with lifecycle scripts disabled
then passed `gpupdal --version` and `gpupdal --drivers`.

**Consequences.** The Linux npm delivery mechanism is qualified independently
of GitHub visibility, post-install downloads, and registry publication. This
test version is intentionally nonpublic and is not a release selection. The
owner's final version/binary choice, the full final conformance report, a
repeat build from that final clean commit, and explicit publish approval remain.
