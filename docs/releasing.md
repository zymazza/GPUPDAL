# Manual release guide

GPUPDAL releases are prepared and verified locally. GitHub Actions is disabled
for the project repository and is not part of the release gate.

## Supported artifacts

The first stable native package is Linux x86-64 with CUDA 13. It is a CLI
distribution containing the public `gpupdal` launcher, CUDA-capable
`pdg-engine`, the pinned sibling `pdal`, runtime libraries, GDAL/PROJ data,
notices, checksums, and an SPDX 2.3 software bill of materials (SBOM). The
driver is a host prerequisite and is never bundled.

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

Check resources and run the maintained Debian 12 container build. It pins the
official base-image digest, installs only the required compiler and geospatial
development packages, builds pinned GDAL 3.8.5 with its required command-line
tools and GEOS support, creates a detached checkout and build of the exact
oracle commit from `cmake/pdg-oracle.cmake`, disables networking during
GPUPDAL compilation and packaging, and uses no more than two compile jobs:

```sh
free -h
scripts/release/build_linux_bundle_debian12.sh
```

The resulting binaries require glibc 2.36 or newer. Debian 12 is the declared
oldest-supported build environment for the first Linux artifact; clean-install
qualification on other advertised distributions remains mandatory. The CPU
bundle is a compatibility companion, not the stable GPUPDAL acceleration
artifact. Its archive name ends in `-linux-x64-cpu.tar.gz`.

The artifact and its outer SHA-256 file are written under `dist/`. The bundle
records the pinned base image, compiler, CMake, glibc, and source revision in
`BUILD-ENVIRONMENT.txt`. The target builds the required executables, discovers
their transitive shared
libraries, excludes the host glibc and GPU driver, copies package-manager
license material, records `RUNTIME_DEPENDENCIES.tsv`, creates
`SBOM.spdx.json`, assigns bundle-relative ELF runtime paths, rejects embedded
local source paths, verifies the archive's internal checksums and npm layout,
compares `gpupdal --drivers` with its sibling `pdal`, and runs an extracted
`gpupdal verify` smoke test in a clean environment. The optional verification
command requires Python 3 on `PATH`; normal PDAL-compatible commands do not.

The archive remains a release candidate until it passes a clean install on
every advertised distribution. Before publication, inspect
`licenses/SYSTEM-LICENSES-MISSING.txt` if present and review every runtime
dependency and notice.

Run the dependency-free core smoke in a read-only, non-root Debian 12
container. It verifies internal hashes, startup, driver parity, and the clear
diagnostic for the optional Python-based verifier:

```sh
scripts/release/smoke_linux_bundle_debian12.sh \
  dist/gpupdal-0.1.0-dev-linux-x64-cpu.tar.gz
```

## Build and qualify the stable CUDA artifact

The CUDA lane mounts the locally installed, version-checked CUDA 13.3.73
toolkit read-only into the same pinned Debian 12 builder, compiles serially for
every real architecture supported by that compiler plus newest-target PTX,
and creates `gpupdal-<version>-linux-x64-cuda13.tar.gz`. It requires the host
NVIDIA container runtime because packaging runs a real GPU smoke test:

```sh
free -h
nvidia-smi
scripts/release/build_linux_cuda_bundle_debian12.sh
scripts/release/test_linux_cuda_bundle_debian12.sh
scripts/release/sanitize_linux_cuda_debian12.sh
scripts/release/smoke_linux_cuda_bundle_debian12.sh \
  dist/gpupdal-0.1.0-dev-linux-x64-cuda13.tar.gz
```

The first stable support profile is compute capability 8.9 on an RTX 4090.
Other cubins in the portable archive are compile coverage, not a physical
exactness or acceleration promise. CUDA 13 applications require an NVIDIA
driver from the CUDA 13 family (580 or newer); the release qualification
records the exact tested driver and hardware.

The stable archive carries `libcudart`, `libnvrtc`, and matching NVRTC builtins
under the CUDA Toolkit EULA. It never carries `libcuda` or `libnvidia-*` driver
libraries. The extracted-artifact gate must run the forced fused CUDA
differential without mounting the host CUDA toolkit; this proves JIT
specialization resolves from the archive instead of an undeclared machine
dependency. `smoke_linux_cuda_bundle_debian12.sh` enforces that `/opt/cuda` is
absent in its fresh GPU container.

## Required release checks

Run these serially, retaining the logs and exact commit SHA. The controlled
Debian build directory records `/src`, so execute its tests through the same
container rather than invoking that build directory directly from the host:

```sh
scripts/release/test_linux_bundle_debian12.sh
git diff --check
```

CUDA artifacts additionally require the portable architecture build, a full
physical CTest aggregate, the compact architecture bit lane, a forced-public
CUDA acceleration proof, and Compute Sanitizer memcheck, initcheck, racecheck,
and synccheck on the final candidate. A local `cudaErrorNoDevice` is not
acceptance evidence.

## npm publication

The npm package is a small launcher carrying the complete, checksummed native
tree inside the immutable npm tarball. It needs no post-install download or
lifecycle script and remains independent of the private source repository.
Immediately before the first public publish:

1. Choose one non-development version for the controlled build and npm
   staging. A non-development bundle refuses a dirty source tree:

   ```sh
   GPUPDAL_RELEASE_VERSION=0.1.0 \
     scripts/release/build_linux_cuda_bundle_debian12.sh
   ```
2. Stage the package without committing the binary archive:

   ```sh
   node packages/npm/scripts/prepare-release.js \
     --version 0.1.0 \
     --archive dist/gpupdal-0.1.0-linux-x64-cuda13.tar.gz \
     --output dist/npm/gpupdal
   ```

3. From `dist/npm/gpupdal`, run `node scripts/validate-package.js`,
   `npm pack --dry-run`, and a clean-directory installation test.
4. Authenticate as `zymazza` with an npm-supported publication method. Run
   `npm publish --access public` inside the reviewed `dist/npm/gpupdal` staging
   directory, keeping any temporary credential configuration outside the
   repository with mode 0600. npm currently accepts either interactive account
   2FA or a narrowly scoped granular write token created with **Bypass 2FA**;
   see [npm's publishing-authentication policy](https://docs.npmjs.com/requiring-2fa-for-package-publishing-and-settings-modification/).
5. Verify the published version and a clean install, delete the temporary npm
   authentication material, and retire any release-only credentials.

No npm credential belongs in the repository, shell history, or retained
release logs. The owner's authentication configuration is not release
metadata. npm provenance is intentionally off while publishing is local and
the source repository is private.

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
