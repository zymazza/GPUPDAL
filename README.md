# GPUPDAL

GPUPDAL (GPU Point Data Abstraction Library) is an independent,
performance-focused point-cloud processing engine derived from PDAL. Its
public `gpupdal` command is a drop-in command-line replacement for `pdal`: it
accepts the same commands and pipelines, then chooses between exact CPU and
CUDA implementations using measured end-to-end performance.

Default mode prioritizes reproducibility. Deterministic outputs must match the
frozen reference bytes; inherently nondeterministic containers are
canonicalized and compared semantically. An accelerated path that cannot meet
that contract declines to the exact host path instead of weakening the
default. See the [conformance suite](docs/conformance.md),
[stage coverage](docs/stage-coverage.md), and
[testing strategy](docs/testing-strategy.md) for the precise contracts.

## Quick start

```sh
gpupdal pipeline pipeline.json
gpupdal translate input.laz output.laz
gpupdal doctor
```

## Profile this machine

Run calibration to measure exact CPU and GPU paths and write a machine-local
placement profile:

```sh
gpupdal calibrate
```

For placement tuned to representative data, supply a LAS or LAZ file:

```sh
gpupdal calibrate --input /path/to/tile.laz --points 250000,1000000,4000000
```

GPUPDAL then uses the profile automatically, selecting the GPU only where it
measured a complete-process win. Inspect the active profile with
`gpupdal calibrate --status`, or preview the calibration plan with
`gpupdal calibrate --dry-run`. The default profile location is
`~/.config/gpupdal/placement-profile.json`. See the
[calibration guide](docs/calibration.md) for quick runs, selected models,
custom output paths, and invalidation rules.

## Verify and benchmark GPUPDAL

Run the portable verification benchmark to compare the installed `gpupdal`
binary with the configured frozen reference on the same machine. It performs
one warm-up followed by three alternating complete-process timing pairs,
checks exactness, and writes portable JSON and HTML evidence:

```sh
gpupdal verify --output-dir gpupdal-proof
```

To use representative data and a pipeline of your own, use `input.laz` and
`output.laz` as the pipeline's input and output placeholders:

```sh
gpupdal verify --input /path/to/tile.laz --pipeline /path/to/pipeline.json \
  --runs 3 --warmups 1 --output-dir gpupdal-proof
```

Publishing the resulting `gpupdal-proof` directory lets another person audit the
machine, binary and input hashes, placement decision, exactness result, and raw
timings. An author-run report is prepared evidence; it becomes third-party
validation only when an unrelated user runs and publishes it. See the full
[`gpupdal verify` guide](docs/verification.md).

## npm installation status

The repository contains the release-safe scaffold for the intended install
experience:

```sh
npm install gpupdal
npx gpupdal --version
```

Use `npm install --global gpupdal` when you want `gpupdal` directly on your
shell path.

The package is not published yet. GPUPDAL is a native C++/CUDA distribution,
so npm publication is gated on checksummed Linux x86-64 release binaries, a
declared support matrix, and clean-machine installation tests. See
[packages/npm/README.md](packages/npm/README.md) and
[RELEASE_READINESS.md](RELEASE_READINESS.md). Releases are built and checked
locally; the repository does not require GitHub Actions. See the
[manual release guide](docs/releasing.md).

## Development status

GPUPDAL is under active development. Runtime and differential-test coverage are
mature, while packaging, APIs, optional integrations, naming, and release
conformance continue to be tracked explicitly:

- [Stage coverage](docs/stage-coverage.md) separates functional, native,
  performance-qualified, and automatically selected coverage.
- [Implementation plan](IMPLEMENTATION_PLAN.md) records remaining release
  work and acceptance gates.
- [Benchmarks](BENCHMARKS.md) and [decisions](DECISIONS.md) contain the
  append-only engineering evidence.
- [Reports](docs/reports/README.md) index the reproducible public evidence
  packages.

## Build

The maintained developer presets are documented in [AGENTS.md](AGENTS.md). A
typical host build starts with:

```sh
cmake --preset pdg-host-debug
cmake --build --preset pdg-host-debug
ctest --preset pdg-host-debug
```

CUDA builds require CUDA 12.4 or newer. Runtime acceleration is conservative:
when a device, pipeline, layout, or performance profile is not qualified,
GPUPDAL uses the exact host implementation.

## License and contributing

GPUPDAL is distributed under the BSD 3-Clause license except where a bundled
component states different terms. See [LICENSE.txt](LICENSE.txt),
[NOTICE](NOTICE), [ORIGIN.md](ORIGIN.md), and the
[third-party inventory](THIRD_PARTY_LICENSES.md) for retained copyright,
provenance, and third-party notices.

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), which
describes the exactness, testing, and benchmark requirements for changes.
