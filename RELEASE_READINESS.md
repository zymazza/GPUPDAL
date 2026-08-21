# GPUPDAL release readiness

Status date: 2026-08-21
Target: first public pre-release (`v0.1.0`)
Current repository posture: private development

GPUPDAL is not ready for a public binary or npm release yet. The engine has a
large exactness and performance evidence base, but the distribution,
licensing, support, and clean-install gates below remain open. This checklist
is intentionally separate from historical benchmark claims.

## Decisions for the owner

Copy this section into an issue or answer it in a review. Items marked
**required** block a public release.

1. **Copyright owner — required.** What exact person or legal entity and year
   should identify GPUPDAL-specific contributions? Do you have authority to
   license every non-upstream contribution already imported into this root
   history?
2. **Project license — required.** Confirm BSD-3-Clause for GPUPDAL-owned code.
   Should contributions use a Developer Certificate of Origin, a CLA, or the
   current inbound=outbound statement in `CONTRIBUTING.md`?
3. **Third-party distribution — required.** Must every vendored subtree carry
   its canonical license file in addition to inline headers? Approve a policy
   and counsel review for lazperf, Eigen, H3, LEPCC, Kazhdan, GoogleTest, and
   all enabled plugins.
4. **Optional plugins — required.** Which plugins and external codecs are in
   the supported release build? Decide explicitly for NITF, HDF/Icebridge,
   TEASER, trajectory/Ceres, RXP, RDB, CPD, Draco, E57, TileDB, and other
   optional drivers. Proprietary or commercial dependencies need separate
   approval.
5. **Fixtures and reports — required.** Which committed datasets and derived
   artifacts may be redistributed? Require a source URL, license, provenance,
   and hash for each public fixture. Decide whether machine-local paths in
   immutable evidence reports remain as provenance or are replaced by a new,
   privacy-reviewed release evidence bundle.
6. **Name permission record — required.** The owner selected “GPUPDAL — GPU
   Point Data Abstraction Library” and reports that Howard Butler gave
   permission to use `GPUPDAL`. Retain that permission in a durable project
   record, confirm its scope, and approve the non-endorsement language for
   PDAL, Hobu, Flaxen, and upstream contributors before public visibility.
7. **Public identifiers.** The command is `gpupdal`; internal C++ namespace,
   environment variables, test data, and CMake targets remain `pdg` for now.
   Should the eventual Python import and C++ API be `gpupdal`, or are those APIs
   intentionally deferred beyond the CLI-only first release?
8. **Version and compatibility.** Confirm `v0.1.0`, Linux x86-64, and the
   supported CUDA/toolkit/driver/GPU range. Decide whether CUDA 12.x and 13+
   are separate artifacts and which physical GPU classes may be advertised.
9. **npm scope — required for npm publication.** Confirm that `gpupdal` is the
   desired unscoped npm name and who owns the npm organization/account. Decide
   whether npm initially supports only Linux x86-64 and whether installation
   may download a checksummed GitHub release asset during `postinstall`.
10. **Security and conduct — required.** Name a monitored security contact,
    response target, code-of-conduct enforcement contact, and appeal path.
11. **Release evidence.** Should the preregistered 40-project 3DEP study and
    at least one unrelated-user `gpupdal verify` report block `v0.1.0`, or may
    they remain clearly labeled post-release validation? Do not market a
    universal speedup: the frozen evidence includes a 0.986x RTX 3090 result
    at 1M points and a fourteen-workflow total-wall range of 0.987–1.430x.
12. **Distribution channels.** Choose the first supported channels: source
    archive, native tarball, npm, container, Python wheel, and/or conda. Each
    selected channel needs its own clean-install and uninstall test.
13. **Citation — required.** Provide GPUPDAL author/maintainer names,
    affiliations/ORCIDs if desired, preferred citation text, release date, and
    whether a DOI will be minted. The inherited PDAL citation must remain as
    upstream attribution, not be presented as the GPUPDAL citation.

## Technical release blockers

- [ ] Create a GPUPDAL-native package target. The inherited CPack/release flow
      currently names PDAL artifacts and validates the PDAL version; it does
      not prove an installed `gpupdal` command.
- [ ] Make the release build enable GPUPDAL, include `gpupdal`, `pdg-engine`, the
      pinned sibling `pdal`, helpers, runtime libraries, notices, and an SBOM,
      then verify all hashes from a clean environment.
- [ ] Add an installed CMake export/config if a public C++ API is in v0.1.0;
      otherwise state that v0.1.0 is CLI-only. The Python builder, CUDA Array
      Interface, DLPack, and GIL-release API remain unfinished P6 work.
- [ ] Replace or supersede the inherited tag workflow. A `v0.1.0` tag cannot
      pass its current PDAL 2.10.0 comparison and it packages with
      `WITH_PDG=OFF` by default.
- [ ] Require host conformance and sanitizers in release CI. Require physical
      GPU exactness and Compute Sanitizer evidence for each advertised GPU
      artifact; manual-only GPU jobs are not a release gate.
- [ ] Restore GitHub-hosted runner availability. The first private-repository
      run failed every `ubuntu-24.04` job before checkout, with no runner or
      logs assigned, even though Actions is enabled. Review the account's
      Actions billing/quota state or attach an approved hosted runner, then
      rerun the green local gates in CI.
- [ ] Complete or explicitly narrow the optional-plugin configuration matrix.
- [ ] Produce SPDX or CycloneDX SBOMs and a reviewed third-party notice bundle.
- [ ] Reconcile public coverage counts. The dated implementation status says
      16 automatically selected filters while the current coverage audit says
      23; the generated audit must become the single source of truth.
- [ ] Replace inherited PDAL-only citation metadata with an approved GPUPDAL
      citation while retaining upstream attribution.
- [ ] Populate `packages/npm/native-manifest.json` with an immutable release
      URL and SHA-256, set the package version, test the installer without
      pre-existing PDAL/CUDA state, and publish with npm provenance.
- [ ] Review committed evidence for private paths or corpus details before
      changing repository visibility. The new one-root-commit release snapshot
      and all newly added files passed a high-confidence credential, private-key,
      and sensitive-filename scan on 2026-08-21; the legacy development history
      is intentionally not part of the new repository.
- [ ] Decide how to ship the inherited
      `test/data/autzen/autzen-surface.tif.min.tif` fixture. GitHub accepted its
      54.41 MB blob but warns above 50 MB; keep it as an attributed upstream
      test input, migrate it to Git LFS with clean-install coverage, or exclude
      it from distribution without weakening upstream tests.
- [ ] Run clean Linux x86-64 source and binary installs, the frozen conformance
      suite, host sanitizers, physical GPU sanitizer lanes, the fourteen
      reference workflows, and `gpupdal verify`; retain immutable reports.

## Completed preparation

- [x] Distinct public name selected: GPUPDAL; the owner reports permission
      from Howard Butler. Durable permission evidence remains a launch gate.
- [x] Public executable renamed to `gpupdal`; it retains the PDAL-compatible
      command/fallback surface while internal `pdg` identifiers remain stable.
- [x] npm package name checked as unclaimed on 2026-08-21 and a fail-closed,
      checksum-verifying Linux x86-64 installer scaffold added.
- [x] Private-phase security policy, conduct policy, issue template, and pull
      request checklist added.
- [x] Root README, origin, license wrapper, and contributor-facing branding
      updated without rewriting historical benchmark/decision evidence.
- [x] Host build, 459-test default host suite, and 96-test extended exact
      differential matrix pass; five optional local-corpus tests are skipped.
- [x] Private `zymazza/GPUPDAL` repository created with `main` at the clean
      GPUPDAL root snapshot; no public release or npm publication was made.
