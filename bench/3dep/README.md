# Preregistered 3DEP RTX 4090 study

This directory defines, but does not claim to have completed, the population
experiment for:

> On an RTX 4090, frozen-release PDG default mode achieves at least 10× pinned
> PDAL wall-clock performance on r6 for at least 90% of independently sampled
> eligible 3DEP projects, with a one-sided exact 95% confidence bound.

The immutable protocol is `preregistration-v1.json`. It fixes a 40-project
sample, the exact r6 pipeline hash, deterministic project and tile selection,
eligibility and replacement rules, frozen binaries, exact-output admission,
one warm-up plus three measured pairs, the project-level Bernoulli endpoint,
and publication requirements before performance is observed.

```sh
python3 bench/3dep/study.py select \
  --preregistration bench/3dep/preregistration-v1.json \
  --catalog frozen-3dep-catalog.json --output selected-projects.json

python3 bench/3dep/study.py seal-inputs \
  --selection selected-projects.json --mapping local-input-map.json \
  --output sealed-inputs.json

python3 frozen-release/evidence/bench/3dep/study.py run \
  --preregistration frozen-release/evidence/bench/3dep/preregistration-v1.json \
  --selection selected-projects.json --inputs sealed-inputs.json \
  --payload frozen-release --artifact-manifest frozen-release.json \
  --output-dir study-results

python3 frozen-release/evidence/bench/3dep/study.py analyze \
  --preregistration frozen-release/evidence/bench/3dep/preregistration-v1.json \
  --catalog frozen-3dep-catalog.json \
  --selection selected-projects.json --inputs sealed-inputs.json \
  --artifact-manifest frozen-release.json \
  --run-report study-results/study-run.json --output study-analysis.json
```

One project contributes one observation. Extra tiles or repetitions do not
increase `n`; missing, failed, inexact, and `<10×` attempts are failures. The
analysis regenerates the complete selection from the supplied frozen catalog,
checks that the release payload contains the same preregistration bytes, and
recomputes each speedup from the six successful finite measured records rather
than trusting a summary field. It then uses the one-sided 95% Clopper–Pearson
lower bound. `29/29` is barely
sufficient (about 0.9019); the planned `40/40` result is about 0.9278.

No catalog snapshot, selected-project manifest, timing output, or accepted
population result is checked in yet. Until those public artifacts exist, the
3DEP statement remains an unexecuted preregistered claim, not benchmark fact.
