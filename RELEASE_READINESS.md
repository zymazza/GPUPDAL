# GPUPDAL release readiness

Status date: 2026-08-21

Target: first public pre-release (final version/channel needs owner approval)

Current repository posture: private development

GPUPDAL is not ready for public npm publication yet. The source and exactness
evidence are mature, and this file now separates decisions already made from
the few owner choices and technical gates that remain.

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
- Long-term platforms: Linux, Windows, and macOS. The first supported native
  artifact will be Linux x86-64. Windows and macOS require real-machine
  validation; modern macOS is CPU-only.
- Release operation: local/manual. GitHub Actions was disabled for the private
  repository on 2026-08-21; no paid hosted CI is a release requirement.
- Native packaging is project work, not an owner prerequisite. A maintained
  pinned-Debian Linux bundle, SPDX SBOM generator, and non-root bare-container
  smoke now exist; final clean-commit and selected-binary qualification remain
  engineering gates below.
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

The initial license, rights, contact, response target, npm owner, platform, and
plugin-scope decisions are complete. Before publication, Zy still needs to
review these release-specific choices:

1. **Version/channel:** use `0.1.0-alpha.1` for the first public compatibility
   preview (recommended for a clearly labeled CPU-only first artifact), or wait
   and publish stable `0.1.0` only after the selected release gates close.
2. **First public binary:** approve a clearly labeled CPU-only compatibility
   preview, or wait for physically qualified CUDA artifacts. The controlled
   Debian archive currently has `PDG_ENABLE_CUDA=OFF`; it preserves configured
   PDAL behavior but is not a GPU acceleration binary.
Actual npm publication and any change from private to public visibility are
separate, explicit owner actions after the selected gates pass.

## Dependency and rights audit

The default build uses open-source system dependencies such as GDAL, PROJ,
GeoTIFF, curl, zlib, zstd, and libxml2, plus vendored components carrying BSD,
MIT, Apache-2.0, Boost-1.0, MPL-2.0, and per-file notices. No proprietary
dependency was identified in the default release preset. Optional plugins can
introduce materially different dependencies, including MATLAB and vendor SDK
integrations, so all are excluded from the proposed first bundle. See
`THIRD_PARTY_LICENSES.md`.

The exact shared-library closure is distribution-specific. Each binary bundle
must include its generated `RUNTIME_DEPENDENCIES.tsv`, copied license material,
and SPDX SBOM, followed by review of any missing license entry. The controlled
CPU archive passes that review; future CUDA and platform artifacts repeat it.
None of this is evidence that the project as a whole depends on closed-source
PDAL components.

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
- [ ] Decide and validate the CUDA runtime policy. The current controlled
      artifact is CPU-only. Produce separate
      CUDA-toolkit artifacts only after physical-GPU exactness and Compute
      Sanitizer gates pass; do not bundle NVIDIA driver libraries.
- [ ] Complete the remaining host release gate locally. In the controlled
      Debian image, 447 unit tests passed and two optional local-corpus tests
      skipped; the full exact differential suite passed 98/98. The host
      ASan/UBSan lane passed all 454 applicable tests with five intentional
      fixture-dependent skips and no sanitizer finding. All 142 tests in the
      sequential published upstream PDAL suite pass, including the two remote
      STAC/COPC cases against their official fixtures. Final conformance and
      retained clean-commit logs remain. GitHub Actions is intentionally not
      part of this gate.
- [ ] Complete or explicitly defer the generated stage-coverage reconciliation
      between older dated counts and the current audit.
- [x] Replace inherited PDAL-only citation metadata with Zy Mazza as the
      GPUPDAL citation author while retaining upstream PDAL attribution.
- [ ] After the owner selects a version and binary lane, rebuild from the clean
      commit, place the immutable archive inside the npm package, populate
      `packages/npm/native-manifest.json`, and run pack plus clean-install
      tests. A nonpublic `0.0.0-test.1` rehearsal from clean commit
      `984e2d21a` passed archive validation, read-only/non-root bare-Debian
      smoke, a 300-entry npm dry run, and a clean local install with both
      `gpupdal --version` and `gpupdal --drivers`. The bundled-archive design
      works while GitHub remains private; repeat the same gate for the selected
      final version and binary lane.
      Authenticate with an npm-supported publication method and publish from
      `zymazza` only with explicit final approval; remove temporary credential
      material after verifying the registry package.
- [x] Preserve the private GPUPDAL naming-permission email. Zy confirmed the
      permission expressly covers the GPUPDAL project name; no separate
      command-name confirmation is required absent limiting language. Do not
      publish the email unless Zy and its sender intentionally choose to.
- [ ] Review committed evidence for private paths or corpus details before
      changing repository visibility. The clean bootstrap snapshot passed a
      high-confidence secret scan on 2026-08-21.
- [x] Losslessly compress
      `test/data/autzen/autzen-surface.tif.min.tif` from 54.41 MB to about
      305 KB while preserving raster values and geospatial metadata.
- [x] Rewrite the short private GPUPDAL `main` history as a clean root snapshot
      after the owner approved the force-push and the equivalent compressed
      fixture passed its metadata/value checks. The old 54.41 MB blob is not
      reachable from the new public-history candidate.
- [ ] Retain immutable release logs and run at least one unrelated-user
      `gpupdal verify` before making broad performance claims. Historical
      evidence includes the 18-job aggregate graph, a 0.986x RTX 3090 1M-point
      result, and a fourteen-workflow total-wall subset ranging 0.987–1.430x,
      so no universal speedup is claimed.

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
