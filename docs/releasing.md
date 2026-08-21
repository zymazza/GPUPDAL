# Manual release guide

GPUPDAL releases are prepared and verified locally. GitHub Actions is disabled
for the project repository and is not part of the release gate.

## Supported artifacts

The first native package is Linux x86-64. It is a CLI distribution containing
the public `gpupdal` launcher, `pdg-engine`, the pinned sibling `pdal`, runtime
libraries, GDAL/PROJ data, notices, checksums, and an SPDX 2.3 software bill of
materials (SBOM).

Optional external plugins are off in this first artifact to keep its dependency
and license closure controlled. This does not weaken compatibility within the
configured catalog: the release gate requires the complete `gpupdal --drivers`
listing to equal the bundled sibling `pdal --drivers` listing. Both commands
therefore expose the same built-in stages and defaults, and GPUPDAL delegates
unsupported accelerated paths to that sibling implementation.

Windows and macOS remain intended platforms, but they are not supported merely
because the source might compile. Each needs a real-machine build, clean
install, exact differential suite, and uninstall test. Modern macOS will be a
CPU-only compatibility build because current CUDA toolkits do not support
macOS. Windows can gain CUDA support after the same physical-GPU gates as
Linux.

## Build a Linux bundle

Check resources, configure the maintained release preset, and use no more than
two host compile jobs on the reference workstation:

```sh
free -h
cmake --preset pdg-host-release
cmake --build build/pdg-host-release --target gpupdal_linux_bundle --parallel 2
```

The artifact and its outer SHA-256 file are written under `dist/`. The bundle
target builds the required executables, discovers their transitive shared
libraries, excludes the host glibc and GPU driver, copies package-manager
license material, records `RUNTIME_DEPENDENCIES.tsv`, creates
`SBOM.spdx.json`, assigns bundle-relative ELF runtime paths, rejects embedded
local source paths, verifies the archive's internal checksums and npm layout,
compares `gpupdal --drivers` with its sibling `pdal`, and runs an extracted
`gpupdal verify` smoke test in a minimal environment.

This workstation artifact is a developer release candidate until the same
recipe runs in the selected oldest-supported Linux build environment and the
archive passes a clean install on every advertised distribution. Before
publication, inspect `licenses/SYSTEM-LICENSES-MISSING.txt` if present and
review every runtime dependency and notice.

## Required release checks

Run these serially, retaining the logs and exact commit SHA:

```sh
ctest --preset pdg-host-release
cmake --build build/pdg-host-debug --target pdg_differential_prerequisites
ctest --test-dir build/pdg-host-debug -L differential --output-on-failure
npm test --prefix packages/npm
git diff --check
```

CUDA artifacts additionally require the portable architecture build, exactness
on physical supported GPUs, and Compute Sanitizer memcheck/racecheck. A local
`cudaErrorNoDevice` is not acceptance evidence.

## npm publication

The npm package is a small launcher that downloads the complete, checksummed
native archive. Immediately before the first public publish:

1. Publish the immutable native archive at a URL accessible without GitHub
   credentials. A release asset in a private repository cannot serve public
   `npm install` users.
2. Set the same non-development version in `CMakeLists.txt`,
   `packages/npm/package.json`, and `packages/npm/native-manifest.json`.
3. Put the immutable asset URL and SHA-256 in the `linux-x64` manifest entry.
4. Run `node packages/npm/scripts/validate-package.js`, `npm pack --dry-run`,
   and a clean-directory installation test.
5. From the `zymazza` npm account with 2FA enabled, run `npm publish --access
   public` inside `packages/npm` and complete the interactive challenge.

No npm token belongs in the repository. npm provenance is intentionally off
while publishing is local and the source repository is private.

## What the SBOM is

An SBOM is an inventory of the files and components in a shipped artifact. It
helps users and maintainers answer questions such as “does this archive contain
a library affected by a newly disclosed vulnerability?” GPUPDAL's generated
SPDX document inventories every bundled file; `RUNTIME_DEPENDENCIES.tsv` and
the copied package-manager license directories retain the build-machine package
mapping needed for review. Runtime package entries intentionally use
`NOASSERTION` rather than guessing normalized licenses, and currently omit
package URLs; treat this as a file inventory and human-review aid, not as a
complete vulnerability-management component catalog.
