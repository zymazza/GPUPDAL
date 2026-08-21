# Contributing to GPUPAL

GPUPAL welcomes focused bug fixes, compatibility improvements, performance
work, tests, documentation, and reproducibility tooling.

## Before you start

- Read [AGENTS.md](AGENTS.md), [spec.md](spec.md),
  [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), and
  [DECISIONS.md](DECISIONS.md).
- Read [docs/testing-strategy.md](docs/testing-strategy.md) before changing
  execution semantics, and [docs/stage-coverage.md](docs/stage-coverage.md)
  before changing stage selection, fallback behavior, or coverage claims.
- For a substantial change, open an issue or discussion first so its scope and
  acceptance evidence can be agreed upon.

## Make a change

1. Create a short-lived topic branch from the active development branch.
2. Add the smallest test that demonstrates the required behavior or captures
   the existing failure.
3. Implement one complete, reviewable change without weakening default-mode
   exactness.
4. Update the relevant documentation and append measured performance evidence
   or architectural decisions where required.
5. Keep commits focused and write messages that explain the outcome.

Do not modify or upload private corpus data. Preserve existing copyright and
license notices, including notices in vendored or derived files.

## Validate the change

Run the narrow tests first, then the relevant maintained preset. A typical
host validation is:

```sh
cmake --preset pdg-host-debug
cmake --build --preset pdg-host-debug
ctest --preset pdg-host-debug
```

Changes affecting compatibility should also run the differential lane:

```sh
cmake --build build/pdg-host-debug --target pdg_differential_prerequisites
ctest --test-dir build/pdg-host-debug -L differential --output-on-failure
```

CUDA changes require a supported GPU run, appropriate sanitizer coverage, and
a same-machine baseline. Check available host memory and GPU capacity before
starting a heavy build, sanitizer, corpus, or benchmark lane.

## Performance claims

A performance claim must include:

- the exact GPUPAL commit and build configuration;
- machine, CPU, RAM, GPU, driver, and CUDA details;
- input identity, size, and hash;
- a same-machine frozen-reference baseline;
- complete-process timings and placement evidence; and
- an append-only entry in [BENCHMARKS.md](BENCHMARKS.md).

An exact but slower accelerated path must remain host-selected and must not be
presented as a speedup.

## Submit the change

- Run `git diff --check` and review the complete diff.
- Push the topic branch and open a pull request against the active development
  branch.
- Describe what changed, why it changed, the validation performed, and any
  remaining limitations.
- Link related issues and include evidence artifacts when the change affects
  compatibility, placement, or performance.

By contributing, you agree that your contribution is distributed under the
repository's [BSD 3-Clause license](LICENSE.txt), unless a file explicitly
states different terms.
