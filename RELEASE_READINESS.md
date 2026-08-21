# GPUPDAL release readiness

Status date: 2026-08-21

Target: first public pre-release (`v0.1.0`)

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
  Linux bundle target and SPDX SBOM generator now exist; its portable build and
  clean-machine qualification remain engineering gates below.
- Howard Butler's GPUPDAL naming permission is in the owner's email. It need
  not be published or committed. Preserve the original message and a backup
  privately before launch, including its date, sender, recipients, and exact
  scope; public documentation may say “permission on file.”
- History cleanup: Zy approved rewriting the short private GPUPDAL history once
  a smaller equivalent test fixture existed. The same Autzen raster now exists
  as a losslessly compressed 305 KB fixture, satisfying that condition; the
  private `main` history was replaced with a clean root snapshot before public
  visibility.

## Owner questionnaire status

The initial license, rights, contact, response target, npm owner, platform, and
plugin-scope decisions are complete. Remaining items below are operational
release qualification, not unanswered owner policy questions.

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
and SPDX SBOM, followed by review of any missing license entry. This is the
remaining binary-distribution check; it is not evidence that the project as a
whole depends on closed-source PDAL components.

## Technical release blockers

- [x] Add a GPUPDAL Linux x86-64 bundle target containing `gpupdal`,
      `pdg-engine`, sibling `pdal`, runtime libraries/data, notices, hashes, and
      a file-level SPDX 2.3 SBOM.
- [ ] Build the Linux archive in a declared oldest-supported distribution,
      review its exact dependency/license closure, and pass clean installs on
      every advertised distribution. The current workstation bundle is only a
      developer release candidate.
- [ ] Replace the reference workstation's broad Arch GDAL linkage with a
      controlled minimal release build. The first dependency scan found 15
      packages without locally installed license texts and GPL/LGPL libraries
      in the transitive GDAL closure. These are open-source dependencies, not
      proprietary ones, but the public bundle needs a smaller reviewed closure
      and complete corresponding notices.
- [ ] Decide and validate the CUDA runtime policy. Produce separate
      CUDA-toolkit artifacts only after physical-GPU exactness and Compute
      Sanitizer gates pass; do not bundle NVIDIA driver libraries.
- [ ] Complete the host release gate locally: conformance, differential tests,
      sanitizers, upstream PDAL tests, and retained logs. GitHub Actions is
      intentionally not part of this gate.
- [ ] Complete or explicitly defer the generated stage-coverage reconciliation
      between older dated counts and the current audit.
- [x] Replace inherited PDAL-only citation metadata with Zy Mazza as the
      GPUPDAL citation author while retaining upstream PDAL attribution.
- [ ] Populate `packages/npm/native-manifest.json` only after an immutable
      public asset exists, set the final version, run a clean npm installation,
      authenticate with an npm-supported publication method, and publish from
      the `zymazza` account. Remove temporary credential material after
      verifying the registry package.
- [ ] Preserve the private GPUPDAL naming-permission email and confirm that its
      exact scope covers the project name and command. Do not publish the email
      unless Zy and its sender intentionally choose to.
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
- [x] The default host suite passed 459/459 tests with five optional local
      fixtures skipped; the extended exact differential suite passed 98/98,
      and the npm scaffold tests pass.
- [x] The independent benchmark report labels the current product GPUPDAL and
      presents the aggregate 18-job graph on a zero-based linear scale.
