## Outcome

Describe the user-visible result and the smallest implementation slice.

## Compatibility

- [ ] Default-mode output, ordering, metadata, diagnostics, and status remain
      identical to the pinned oracle, or the path delegates before side effects.
- [ ] Stage coverage and fallback claims are updated when selection changed.
- [ ] No private corpus data, credentials, or machine-local artifacts are added.

## Validation

- [ ] Focused unit/differential tests
- [ ] Relevant host preset
- [ ] ASan/UBSan when applicable
- [ ] Physical GPU and Compute Sanitizer evidence when CUDA changed
- [ ] Same-machine pinned-PDAL baseline and `BENCHMARKS.md` entry for any
      performance claim

List exact commands, environment, and skipped gates:

## Licensing and provenance

- [ ] New or copied code retains required copyright/license notices.
- [ ] New dependencies and fixtures include license, source, and provenance.
- [ ] No LASzip or other incompatible code was introduced.
