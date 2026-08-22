# GPUPDAL release readiness

Status date: 2026-08-22

Target: first stable CUDA release (proposed version `0.1.0`; owner approval
still required before publication)

Current repository posture: private development

The clean `0.1.0` CUDA release candidate is physically qualified on the
declared RTX 4090 / SM 89 support profile, including the final conformance and
npm clean-install rehearsals. GPUPDAL has not been published. The remaining
release gates are the owner's explicit npm-publication and
repository-visibility approvals.

## Owner answers recorded

- Product: **GPUPDAL — GPU Point Data Abstraction Library**; public command and
  intended unscoped npm package: `gpupdal`.
- Citation author: **Zy Mazza**. Affiliation, ORCID, and DOI are optional and
  may be added later.
- License: **BSD-3-Clause** for GPUPDAL-owned work. Commercial use by others is
  permitted; Zy may commercialize support, certified builds, hosted services,
  integrations, warranties, and the GPUPDAL brand. The GPUPDAL-specific third
  clause explicitly bars use of the names GPUPDAL, Zy Mazza, or Automagics to
  endorse or promote a derived product without prior written permission.
- Rights: Zy confirms authority to license the existing GPUPDAL-specific code,
  tests, documentation, generated assets, and modifications. Upstream and
  third-party work retains its existing license and attribution.
- Public security/conduct contact: **zy@automagics.com**. GPUPDAL aims to
  acknowledge reports within five business days on a best-effort basis. This
  is not an SLA, warranty, support promise, or guarantee of a fix.
- npm owner: **zymazza**. Authentication was verified without publishing or
  reserving a package. The owner's npm authentication configuration is private
  operational information rather than project release metadata.
- First binary scope: CLI-only Linux x86-64 with optional external plugins off.
  Drop-in behavior covers the complete configured stage catalog and is gated
  by exact `gpupdal --drivers` parity with the bundled sibling `pdal`; plugin
  source remains available for later artifacts and source builds.
- Stable-product meaning: the first stable package is the CUDA 13 artifact,
  and supported measured envelopes must select real GPU execution. A CPU-only
  archive may be shipped as a compatibility companion, but is not the stable
  GPUPDAL acceleration product.
- Long-term platforms: Linux, Windows, and macOS. The first supported native
  artifact is Linux x86-64. Windows Server 2022 source-build and physical L4
  CUDA qualification is complete, but a redistributable Windows bundle and
  its clean-machine package gates remain future work. Modern macOS is CPU-only
  and still requires real-machine validation.
- Release operation: local/manual. GitHub Actions was disabled for the private
  repository on 2026-08-21; no paid hosted CI is a release requirement.
- Native packaging is project work, not an owner prerequisite. A maintained
  pinned-Debian Linux bundle, SPDX SBOM generator, and non-root bare-container
  smoke exist; the final clean-commit and selected-binary qualification gates
  below are complete.
- Howard Butler's GPUPDAL naming permission is in the owner's email. Zy
  confirmed that it expressly covers the project name; the lowercase
  `gpupdal` executable is the ordinary command form of that same name and does
  not need a separate follow-up absent limiting language in the message. The
  email need not be published or committed. Preserve the original and one
  private backup; public documentation may say “permission on file.”
- History cleanup: Zy approved rewriting the short private GPUPDAL history once
  a smaller equivalent test fixture existed. The same Autzen raster now exists
  as a losslessly compressed 305 KB fixture, satisfying that condition; the
  private `main` history was replaced with a clean root snapshot before public
  visibility.

## Owner review still needed

The initial license, rights, contact, response target, npm owner, platform,
plugin scope, and stable CUDA binary decisions are complete. Before
publication, Zy still needs to approve:

1. **Version/channel:** publish the physically qualified CUDA artifact as
   stable `0.1.0` on npm's `latest` channel (recommended), or choose a different
   version/channel and repeat the version-bound archive/npm staging.
2. **External state changes:** separately authorize the actual npm publication
   and changing `zymazza/GPUPDAL` from private to public. Neither action is
   implied by approving the engineering candidate.

## Dependency and rights audit

The source and CPU dependency set uses open-source components such as GDAL,
PROJ, GeoTIFF, curl, zlib, zstd, libxml2, and vendored code carrying BSD, MIT,
Apache-2.0, Boost-1.0, MPL-2.0, and per-file notices. The stable CUDA artifact
also redistributes NVIDIA's proprietary `libcudart`, `libnvrtc`, and matching
NVRTC builtins under the CUDA Toolkit EULA. Those are accelerator runtime
dependencies, not closed-source PDAL stages. NVIDIA driver libraries remain a
host prerequisite and are never bundled. Optional plugins can introduce
materially different dependencies, including MATLAB and vendor SDK
integrations, so all are excluded from the first bundle. See
`THIRD_PARTY_LICENSES.md`.

The exact shared-library closure is distribution-specific. Each binary bundle
must include its generated `RUNTIME_DEPENDENCIES.tsv`, copied license material,
and SPDX SBOM, followed by review of any missing license entry. The controlled
CPU and final CUDA archives pass that review; future platform artifacts repeat
it. None of this is evidence that the project as a whole depends on
closed-source PDAL components.

## Technical release blockers

- [x] Add a GPUPDAL Linux x86-64 bundle target containing `gpupdal`,
      `pdg-engine`, sibling `pdal`, runtime libraries/data, notices, hashes, and
      a file-level SPDX 2.3 SBOM.
- [x] Build the Linux archive in the pinned Debian 12 oldest-supported
      environment and review its exact dependency/license closure. The current
      controlled archive contains 51 runtime dependency rows, 45 copied system
      notice sets, 47 SPDX packages, 290 inventoried files, 337 SPDX
      relationships, and no missing-license marker. Its relative runtime paths,
      internal/outer hashes, driver parity, private-path scan, and bare Debian
      non-root startup passed.
- [x] Replace the reference workstation's broad Arch GDAL linkage with pinned
      GDAL 3.8.5 built in the controlled Debian image, including the GEOS and
      command-line support required by the release differentials.
- [x] Validate the stable CUDA runtime policy and physical device lane. The
      controlled Debian 12/CUDA 13.3.73 all-architecture build produced the
      `-linux-x64-cuda13` archive. On RTX 4090/SM 89 with driver 610.43.03,
      all 863 registered release tests passed with zero failures; 22 optional
      local-fixture/profile-envelope cases were explicitly skipped. The 107
      CUDA-labeled tests include option-free automatic GPU selection for the
      accepted acceleration envelopes. The checked-in eight-test sanitizer
      matrix passes memcheck, initcheck, and synccheck with zero errors and
      racecheck with zero hazards/errors/warnings. An extracted archive also
      passed the forced fused CUDA/NVRTC exact differential without the host
      CUDA toolkit mounted. A second extracted-archive smoke passed exact
      fallback with no NVIDIA device, driver library, or CUDA toolkit present.
- [x] Review CUDA binary portability and dependency closure. CUDA 13.3 emitted
      cubins for SM 75, 80, 86, 87, 88, 89, 90, 100, 103, 110, 120, and 121,
      plus newest-target SM 120 PTX. Only SM 89 is physically qualified; the
      other images are compile coverage. The final 121,512,265-byte archive
      has SHA-256
      `f241da5888ac8de837449da3a96bc09d042bb3d265a8d51084ddd88520261841`,
      54 runtime dependency rows, 45 copied system notice sets, 48 SPDX
      packages, 295 SPDX files, 343 relationships, 296 internal checksum
      entries, and no missing-license marker or bundled NVIDIA driver library.
      The base image, source archives, CUDA compiler, and CCCL digest are
      verified. Debian apt inputs are version-recorded but not snapshot-pinned,
      so this is an auditable controlled build rather than a bit-reproducible
      rebuild claim.
- [x] Complete the current controlled aggregate release gate locally. The
      final CUDA-enabled aggregate passed 863/863 registrations, including the
      exact host/fallback and CUDA process matrices. Earlier host ASan/UBSan
      passed all 454 applicable registrations with five intentional
      fixture-dependent skips; the sequential published upstream PDAL suite
      passed 142/142, including the official network STAC/COPC cases. GitHub
      Actions is intentionally not part of this gate.
- [x] Run and retain the final 2,048-case conformance report against the clean
      `0.1.0` archive. All 2,048 cases passed, the run is complete rather than
      partial, and it records zero unexplained semantic differences. The raw
      report SHA-256 is
      `b1a46bbd2cd1e1a423e877df6d81aec0100328b9c10252ae760489d83837f8a0`;
      its deterministic retained gzip SHA-256 is
      `ca0866a851e255b5a951daf32ca4b39beae9cf1b06258385e04bfd2b866a118a`.
- [x] Reconcile the configured stage catalog against the selected controlled
      build: `gpupdal --drivers` and sibling `pdal --drivers` are byte-identical
      at 124 entries (84 filters, 25 readers, 15 writers).
- [x] Replace inherited PDAL-only citation metadata with Zy Mazza as the
      GPUPDAL citation author while retaining upstream PDAL attribution.
- [x] Rebuild the proposed `0.1.0` from the clean
      commit, place the immutable archive inside the npm package, populate
      `packages/npm/native-manifest.json`, and run pack plus clean-install
      tests. Commit `7981754d150a96116875be1fdcac525b52ff4afd` produced a
      305-entry, 122,632,557-byte npm tarball with SHA-256
      `c2ba3ad8f1f211bd266f593fcef9df8924c482d04a03e3c66faccfaf918a3bcb`.
      A clean offline install validated all 296 native checksums, ran
      `gpupdal --version` and `gpupdal --drivers`, passed a forced fused CUDA
      exact differential without the host toolkit, and passed byte-exact
      fallback with neither an NVIDIA driver nor a GPU.
      Authenticate with an npm-supported publication method and publish from
      `zymazza` only with explicit final approval; remove temporary credential
      material after verifying the registry package.
- [x] Preserve the private GPUPDAL naming-permission email. Zy confirmed the
      permission expressly covers the GPUPDAL project name; no separate
      command-name confirmation is required absent limiting language. Do not
      publish the email unless Zy and its sender intentionally choose to.
- [ ] Review committed evidence for private paths or corpus details before
      changing repository visibility. A 2026-08-22 credential-pattern scan is
      clean, and the new `0.1.0` report contains only container-local
      `/qualification` paths. Historical agent guidance and benchmark JSON
      intentionally retain `/home/zy` provenance strings and local corpus
      names. Before making the repository public, Zy must approve retaining
      those non-secret strings or authorize a mechanical redaction pass.
- [x] Losslessly compress
      `test/data/autzen/autzen-surface.tif.min.tif` from 54.41 MB to about
      305 KB while preserving raster values and geospatial metadata.
- [x] Rewrite the short private GPUPDAL `main` history as a clean root snapshot
      after the owner approved the force-push and the equivalent compressed
      fixture passed its metadata/value checks. The old 54.41 MB blob is not
      reachable from the new public-history candidate.
- [ ] Post-release validation: retain immutable release logs and invite at
      least one unrelated user to run
      `gpupdal verify` before making broad performance claims. Historical
      evidence includes the 18-job aggregate graph, a 0.986x RTX 3090 1M-point
      result, and a fourteen-workflow total-wall subset ranging 0.987–1.430x,
      so no universal speedup is claimed. This is not a blocker for the narrow
      SM 89 automatic-acceleration support statement.

## Future Windows artifact gates

These items do not block the qualified Linux x86-64 `0.1.0` candidate.

- [x] Compile the native `gpupdal`, `pdg-engine`, and `pdg_unit_tests`
      targets on Windows Server 2022 with Visual Studio 2022/MSVC 19.44 and
      CUDA 13.3. The port preserves the public `gpupdal` command and internal
      `pdg` namespace, uses Windows process/mapping/profile primitives, and
      retains exact host fallback.
- [x] Run the Windows product on physical CUDA hardware. On an AWS G6
      non-fractional NVIDIA L4/SM 89 with Tesla driver 610.88, `gpupdal
      doctor` reported CUDA toolkit/runtime/driver 13.3 and one 23,910,350,848
      byte device. The explicit NVRTC specialization gate passed, the broad
      CUDA unit surface passed 95/95, and the forced fused assign/ferry process
      differential matched the sibling pinned PDAL oracle exactly.
- [x] Run the rebuilt complete Windows unit binary and focused physical
      sanitizer lane. The final binary SHA-256 is
      `3f60e3855d174301e1fd18c49c62299564776d8548ac8a9a6d6c1fccffe74233`;
      it discovered 650 tests, passed 645, explicitly skipped five unavailable
      local/benchmark fixtures, and had zero failures. Its checked-in
      eight-test CUDA matrix passed Compute Sanitizer memcheck, initcheck, and
      synccheck with zero errors and racecheck with zero hazards, errors, or
      warnings.
- [ ] Produce a redistributable Windows archive and npm platform payload with
      an audited DLL/data dependency closure, copied license material,
      `SHA256SUMS`, and a file-level SPDX SBOM. NVIDIA's host driver must remain
      external.
- [ ] Exercise that archive on a separate clean Windows machine: install,
      command discovery, `--version`, exact driver catalog, forced GPU/NVRTC
      differential, driverless exact fallback, uninstall, and absence of
      compiler/Conda/source-tree dependencies.
- [ ] Run the complete Windows CTest process/differential aggregate from the
      packaged candidate. Source-tree units and one exact process lane are not
      a substitute for the complete package-level compatibility matrix.
- [ ] Establish same-machine Windows PDAL baselines before making a Windows
      performance or automatic-selection support claim. The current Windows
      work qualifies correctness and CUDA execution, not a speedup claim.

## Completed preparation

- [x] Private repository created at `zymazza/GPUPDAL`; no public release or npm
      publication has been made.
- [x] `gpupdal` is the PDAL-compatible public command; internal `pdg`
      identifiers remain stable.
- [x] The unscoped npm name returned not-found on 2026-08-21. That check does
      not reserve the name.
- [x] Private-phase security policy, conduct policy, issue template, pull
      request checklist, license wrapper, attribution, and provenance files are
      present.
- [x] The controlled Debian unit suite passed 447 tests with two optional local
      corpus tests skipped; the extended exact differential suite passed 98/98,
      and the npm scaffold tests pass.
- [x] The independent benchmark report labels the current product GPUPDAL and
      presents the aggregate 18-job graph on a zero-based linear scale.
