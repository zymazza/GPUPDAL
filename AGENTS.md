# GPUPDAL agent guide

This repository is an upstream-backed fork of PDAL for a CUDA-native engine
whose public drop-in CLI is `gpupdal` and whose internal C++ namespace remains
`pdg`. Read this file first, then read
`spec.md`, `IMPLEMENTATION_PLAN.md`, `DECISIONS.md`, and
`docs/testing-strategy.md`. Read `docs/stage-coverage.md` before changing stage
selection, fallback, or public coverage claims.

## Non-negotiable priorities

1. The default compatibility mode reproduces the pinned upstream PDAL oracle
   byte-for-byte, including output files, metadata, point order, stdout,
   stderr, and exit status where applicable.
2. Speed is optimized subject to that compatibility contract. A stage that
   cannot yet be exact must use the upstream/host fallback in compatibility
   mode. Relaxed floating-point or ordering semantics belong behind an
   explicit `--fast` mode and must never silently change defaults.
3. No performance claim is accepted without a same-machine PDAL baseline,
   profile evidence, and an append-only entry in `BENCHMARKS.md`.
4. GPU stages share the planner-owned spatial index. A stage must never build
   a private k-d tree, grid, or BVH.
5. Preserve upstream licensing and attribution. Never introduce LASzip or
   other incompatible code.
6. The PDAL fork is the product. Preserve every configured PDAL command and
   stage. Report functional and GPU-native coverage separately; never call a
   host fallback GPU-native.
7. **The goal is speed, not GPU.** CUDA is one means among several. Success is
   measured wall-clock time on the reference pipelines in
   `bench/pipelines/reference/` against the pinned oracle on identical
   hardware — not by how many stages have a CUDA path. Host-side work
   (threading, algorithms, I/O, allocation) is first-class and counts as a win
   even when no GPU is involved. A CUDA path is built or kept only where it is
   measured faster end-to-end; an exact-but-slower CUDA lane stays host-selected
   and is never presented as an acceleration. Functional coverage remains
   catalog-wide and the exact-output contract is unchanged; catalog-wide *CUDA*
   coverage is no longer a release criterion. See D0208, which supersedes the
   corresponding parts of D0019.

The exact-output requirement supersedes the looser order-insensitive and
numeric-tolerance language in `spec.md`. See decision D0002.

## Upstream and branch policy

- `upstream` is `https://github.com/PDAL/PDAL.git`.
- `origin` is the hosted fork at `https://github.com/zymazza/PDAL.git`.
- The compatibility oracle is pinned in `cmake/pdg-oracle.cmake`; update it
  only with a decision entry and regenerated golden manifests.
- GPU work happens on `gpu-main` or a short-lived branch based on it.
- Keep upstream PDAL buildable. Its library and CLI are the differential
  oracle until a separately pinned oracle container replaces them.
- Do not copy an upstream implementation into CUDA without retaining BSD
  attribution in `NOTICE` and citing the relevant source or paper.

## Required work loop

1. Identify the smallest phase item from `IMPLEMENTATION_PLAN.md`.
2. Write the differential or numeric test first. For exact stages, capture the
   upstream bytes and diagnostics before implementing the accelerated path.
3. Implement the smallest complete vertical slice: descriptor, planner data,
   implementation, fallback, tests, benchmark entry, and documentation.
4. Build and run the narrow tests, then the relevant corpus tier.
5. Run sanitizers appropriate to the change. CUDA code must pass
   compute-sanitizer memcheck and racecheck on a GPU runner.
6. Profile before optimizing. Record the limiting resource and roofline gap.
7. Append results or deviations to `BENCHMARKS.md` or `DECISIONS.md`; never
   rewrite prior entries.

## Definition of done for a stage

- PDAL-compatible stage name, options, defaults, errors, and descriptor.
- Correct planner metadata: reads, writes, kind, index request, maximum radius,
  ordering behavior, and XYZ invalidation.
- Default mode is byte-identical to the pinned oracle across its fixture
  matrix, or explicitly selects the host fallback.
- `--fast` differences, if any, are documented and separately tested.
- NVTX ranges cover the stage and every CUDA runtime call uses
  `PDG_CUDA_CHECK`.
- Unit, differential, property, determinism, and malformed-input tests exist
  as applicable.
- An nvbench or google-benchmark case and same-machine upstream baseline exist.
- The public stage page documents parity, fallback behavior, and performance.
- Mixed pipelines retain unchanged PDAL stages in-process when execution mode
  and ordering are proven; otherwise delegate the original pipeline before
  data or output side effects.
- Host sanitizers and compute-sanitizer are clean.

## CUDA and C++ rules

- C++20 host code and the nvcc-supported C++20 subset only.
- CUDA 12.4 or newer. RTX 4090/SM 89 is the performance reference, never the
  compatibility floor. Release builds use every real architecture supported
  by their CUDA compiler plus PTX for its newest target. CUDA 12.x supplies the
  legacy-compatible SM 50–90 artifact; CUDA 13+ supplies the current SM 75+
  artifact, including post-Hopper targets known to that compiler. A generated
  cubin is only build compatibility: claim exact runtime support for an SM only
  after its compact bit-differential lane passes on physical hardware.
- No raw `cudaMalloc`/`cudaFree` outside `src/core/allocator*`.
- No managed/UVM allocation in a hot path.
- Use RAII for streams, events, allocations, modules, and library handles.
- Use stream-ordered allocation and async pinned transfers.
- Use grid-stride loops by default. Explain every `__launch_bounds__` choice.
- Prefer CUB/CCCL/libcu++; a custom primitive needs a benchmark showing why.
- Keep fp64 out of neighborhood inner loops. Exact mode may use integer,
  fixed-order, or host computation when required to match the oracle.
- Product CUDA translation units compile exact mode with FMA contraction and
  FTZ disabled and precise division/square root enabled. A relaxed fast-mode
  kernel belongs in a separately contracted translation unit or target.
- Never synchronize the device merely to simplify ownership. Model lifetime
  with streams/events and test it.

## Data and testing

- Public, redistributable fixtures live in `tests/data` with license and source
  metadata.
- Large or private local fixtures stay outside Git. Register only their path,
  size, hash, format, and provenance status in a generated local manifest.
- Useful local corpora include `/home/zy/dev/adklr`, `/home/zy/dev/veil`,
  `/home/zy/dev/snow-road-twin`, related projects under `/home/zy/dev`, and
  geospatial files under `/home/zy/Downloads`.
- Never upload, commit, rename, or modify corpus files. Tests write derived
  artifacts only below the configured build directory or `/tmp`.
- Use named, deterministic boundary fixtures and fixed arithmetic sequences.
  Do not add open-ended random input generators; turn each important case into
  a small, bounded, reviewable regression recipe.
- A differential failure must report the first differing byte and, for LAS,
  the decoded header/VLR/EVLR/point-record field responsible.
- Never bless new golden output merely because the implementation changed.
  Regeneration requires an intentional oracle update.

## Repository hygiene

- Preserve unrelated user changes and upstream behavior.
- Use `rg`/`rg --files` for discovery.
- Keep public headers minimal and implementation details under `src/`.
- Update `DECISIONS.md` for spec deviations, gate outcomes, oracle changes, or
  compatibility/performance tradeoffs.
- Update `BENCHMARKS.md` only with reproducible measured results; label machines,
  builds, datasets, modes, and commit SHAs.
- Do not claim a phase exit until every stated exit test is green.

## Model routing and delegation

- Delegate small, deterministic, easily reviewed tasks to the project
  `spark_explorer` or `spark_worker` agent. Typical work includes bounded
  searches, inventories, evidence extraction, mechanical documentation checks,
  and narrowly specified edits.
- Use `terra_reviewer` for broader read-heavy audits and medium-complexity
  reviews. Keep architectural choices, exactness-critical CUDA semantics,
  performance acceptance, and final integration decisions with the primary
  agent or an explicitly requested high-capability independent reviewer.
- The project default subagent is `gpt-5.3-codex-spark` at low reasoning.
  Override it explicitly when the task needs Terra or a stronger model; never
  represent a substitute model as Spark.
- Delegated agents may not start a heavy build, sanitizer, benchmark, corpus
  scan, network operation, or secret-reading command unless the primary agent
  assigns that exact lane after checking resources. The primary agent remains
  responsible for reviewing every delegated result before it affects code or
  acceptance claims.

## Local resource discipline

- Check `free -h` before a CUDA build, sanitizer run, large differential, or
  corpus benchmark. Do not start a heavy lane below 8 GiB `MemAvailable`, and
  stop if available memory approaches that floor or swap use grows materially.
- Run only one heavy lane at a time. Never overlap CUDA compilation, host
  sanitizers, Compute Sanitizer, large-corpus hashing, or a benchmark.
- Use one compile job for CUDA and at most two for ordinary host validation on
  the reference workstation unless a fresh resource check justifies more.
- Use a deliberately reduced architecture list with
  `PDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES=OFF` for iterative local builds.
  Reserve `-arch=all` fatbin compilation for the serialized release/CI lane.
- Bound test inputs, pinned queues, and temporary outputs. Check both host RAM
  and VRAM before a large run; prefer focused tests until the final aggregate
  gate is necessary.

## Initial commands

The maintained presets are introduced in P0. The intended workflow is:

```sh
cmake --preset pdg-host-debug
cmake --build --preset pdg-host-debug
ctest --preset pdg-host-debug

cmake --build build/pdg-host-debug \
  --target pdg_differential_prerequisites
ctest --test-dir build/pdg-host-debug -L differential --output-on-failure

cmake --preset pdal-upstream-tests
cmake --build --preset pdal-upstream-tests
ctest --preset pdal-upstream-tests

cmake --preset pdg-cuda-release
cmake --build --preset pdg-cuda-release
ctest --test-dir build/pdg-cuda-release --output-on-failure
```

Run the published upstream PDAL preset sequentially. Some upstream suites share
`test/temp`, and remote STAC/EPT/COPC cases need their official network
fixtures.

Run GPU tests and benchmarks only when `nvidia-smi` confirms a usable device.
Compilation-only CUDA validation is still required when the driver is absent.
In a managed workspace, NVIDIA devices may be masked even when the host GPU is
healthy. A sandbox-local `cudaErrorNoDevice` is not sufficient evidence that
hardware is absent: confirm `nvidia-smi` with approved host-device access, then
run the GPU gates with that same access.
