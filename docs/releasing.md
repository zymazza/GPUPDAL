# Manual release guide

GPUPDAL releases are prepared and verified locally. GitHub Actions is disabled
for the project repository and is not part of the release gate.

## Supported artifacts

The `0.1.0` npm release target carries CUDA 13 native payloads for Linux x86-64
and Windows x64. Each is a CLI distribution containing the public `gpupdal`
launcher, CUDA-capable `pdg-engine`, the pinned sibling `pdal`, runtime
libraries, GDAL/PROJ data, notices, checksums, and an SPDX 2.3 software bill of
materials (SBOM). The driver is a host prerequisite and is never bundled.

Optional external plugins are off in this first artifact to keep its dependency
and license closure controlled. This does not weaken compatibility within the
configured catalog: the release gate requires the complete `gpupdal --drivers`
listing to equal the bundled sibling `pdal --drivers` listing. Both commands
therefore expose the same built-in stages and defaults, and GPUPDAL delegates
unsupported accelerated paths to that sibling implementation.

Windows is accepted only after its real-machine build, clean GPU and
driverless installs, exact differential, npm local/global install, and
uninstall gates pass. Modern macOS remains an intended CPU-only compatibility
build because current CUDA toolkits do not support macOS; it still needs its
own real-machine qualification before being added to the support matrix.

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

The Debian base-image digest, CMake archive, GDAL source, CUDA compiler version,
and CCCL header-tree digest are verified. Debian apt packages come from the
current signed Bookworm repositories rather than a dated snapshot; their
actual versions and copied notices are recorded in the artifact. The lane is
auditable and controlled, but a later rebuild is not claimed to be bit-for-bit
reproducible unless those apt inputs are snapshot-pinned too.

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
scripts/release/smoke_linux_cuda_bundle_no_driver_debian12.sh \
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

The CUDA engine itself cannot load without NVIDIA's unbundled `libcuda`.
The public launcher therefore checks driver availability before selecting the
engine and delegates directly to the bundled pinned PDAL command on a
driverless machine. The separate no-driver smoke proves that delegation is
byte-exact in a container with neither NVIDIA devices nor a CUDA toolkit.

## Build and qualify the Windows CUDA artifact

Configure an optimized x64 Ninja `Release` or `RelWithDebInfo` build with
Visual Studio 2022, CUDA 13.3,
`GPUPDAL_ENABLE_CUDA=ON`, `PDG_CUDA_ARCHITECTURES=all`,
`PDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES=ON`, a full clean Git revision in
`GPUPDAL_SOURCE_REVISION`, and optional plugins off. Build serially. Then run
the maintained bundle target:

```powershell
cmake --build C:\gpupdal\build-cuda --parallel 1 `
  --target pdg_cli pdg_engine pdal pdg_verify_helpers
cmake --build C:\gpupdal\build-cuda --parallel 1 `
  --target gpupdal_windows_bundle
```

The bundle target rejects the wrong version/revision, a dirty Git source tree,
non-portable architecture settings, enabled optional plugins, unresolved DLLs,
unowned Conda data, missing license files, an incomplete CUDA image set,
embedded personal or noncanonical build paths, unchecksummed archive entries,
and extracted startup or driver-catalog failures. It copies the app-local
Visual C++ runtime, CUDA runtime/NVRTC libraries, GDAL/PROJ/PDAL data, CA
roots, third-party notices, the runtime dependency map, and an SPDX SBOM. It
never copies the NVIDIA host driver.

Run the complete unit/CTest and Compute Sanitizer gates from a separate
same-revision SM 89 qualification build configured for the physical device.
Do not compile every test-only CUDA translation unit for all portable cubins;
the product artifact carries the all-architecture image, while the tests prove
the physically available architecture.

Run `scripts/release/test_windows_bundle.ps1` on two separate clean Windows
hosts: one x64 SM 89 machine with only the NVIDIA driver added and one x64
machine with no NVIDIA driver. The GPU lane requires a forced CUDA fused
pipeline and byte-exact output against the bundled PDAL oracle. The driverless
lane requires byte-exact direct fallback, including a reprojection that proves
the launcher discovers bundle-relative GDAL/PROJ data before dispatch. Both
lanes reject compiler, CMake, Conda, and CUDA-toolkit commands on `PATH` and
exercise local and global npm install, command discovery, exact output, and
uninstall from the final npm tarball.

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

The public `gpupdal` package is a small launcher whose exact optional
dependencies select a native package and CUDA-runtime package for the current
platform. This split keeps each immutable npm tarball below the registry's
upload ceiling. The five-package release set needs no post-install download or
lifecycle script and remains independent of the private source repository.
Users install only `gpupdal`; the four support packages are implementation
details. Immediately before a public publish:

1. Choose one non-development version for the controlled build and npm
   staging. A non-development bundle refuses a dirty source tree:

   ```sh
   GPUPDAL_RELEASE_VERSION=0.1.0 \
     scripts/release/build_linux_cuda_bundle_debian12.sh
   ```
2. Stage the package set without committing the binary archives:

   ```sh
   node packages/npm/scripts/prepare-release.js \
     --version 0.1.0 \
     --linux-archive dist/gpupdal-0.1.0-linux-x64-cuda13.tar.gz \
     --windows-archive dist/gpupdal-0.1.0-win32-x64-cuda13.zip \
     --output dist/npm/gpupdal
   ```

3. Run `node dist/npm/gpupdal/scripts/validate-release-set.js`. Pack all five
   staging directories and install the launcher plus the matching native and
   runtime tarballs into a clean directory with lifecycle scripts disabled.
4. Authenticate as `zymazza` with an npm-supported publication method, keeping
   temporary credential configuration outside the repository with mode 0600.
   npm currently accepts either interactive account 2FA or a narrowly scoped
   granular write token created with **Bypass 2FA**; see
   [npm's publishing-authentication policy](https://docs.npmjs.com/requiring-2fa-for-package-publishing-and-settings-modification/).
5. Publish `@zymazza/gpupdal-linux-x64`,
   `@zymazza/gpupdal-cuda13-linux-x64`, `@zymazza/gpupdal-win32-x64`, and
   `@zymazza/gpupdal-cuda13-win32-x64` at the exact release version before
   publishing `gpupdal`. Use `--access public --tag latest` for every package.
   Publishing the launcher last prevents a visible root version from referring
   to support versions that do not yet exist.
6. Verify registry metadata and a clean `npm install gpupdal@<version>`, test
   the installed command, then delete temporary npm authentication material
   and retire any release-only credentials.

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
