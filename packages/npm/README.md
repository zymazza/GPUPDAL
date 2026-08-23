# GPUPDAL npm installer

This directory is the publication source for the unscoped `gpupdal` package:

```sh
npm install gpupdal
npx gpupdal --version
```

For a shell-wide command, use `npm install --global gpupdal` followed by
`gpupdal --version`.

GPUPDAL is native software. The public `gpupdal` package is a small launcher,
not a JavaScript reimplementation. Exact optional dependencies install the
matching immutable native and CUDA-runtime support packages for the current
operating system. Installation does not depend on access to the source
repository, a second download host, or an npm lifecycle script. Before packing,
the release-set validator reconstructs the original platform archive from its
two support packages and verifies every file against that archive's internal
`SHA256SUMS`. The installed native tree contains the complete sibling bundle
needed by the launcher (`gpupdal`, `pdg-engine`, and the pinned `pdal` oracle).

The checked-in manifest is intentionally empty and the development version is
rejected by `prepack`. The manual release process stages one launcher and four
internal support packages under `dist/npm/`, records both source-archive
digests, validates the combined release set, performs clean-machine installs,
publishes support packages first, and publishes `gpupdal` last. Users still
install only `gpupdal`; npm selects the matching platform packages. Staging
requires `tar` and `unzip`. See `docs/releasing.md` in the source repository
for the exact commands.
The package intentionally disables npm provenance for now because npm does not
support provenance from a local manual publish or a private source repository.

The initial stable support declaration covers Linux x86-64 and Windows x64
with an NVIDIA GPU, driver 580 or newer, and the bundled CUDA 13 runtime.
Physical exactness is qualified on compute capability 8.9 (RTX 4090 on Linux
and NVIDIA L4 on Windows). The portable binaries contain every real
architecture supported by their CUDA 13 compiler plus newest-target PTX, but
an architecture is not advertised as stable until its physical fixed-bit lane
passes. The same package also retains exact PDAL fallback behavior on a
driverless host; that is compatibility, not a GPU-acceleration claim.

The npm installer validates the declared artifact; it does not reject a
machine at install time for lacking the qualified GPU or driver. On such a
machine the command may still run through its exact CPU/PDAL fallback, but no
stable GPU-acceleration claim applies.

Publishing the unscoped package makes its current npm maintainers responsible
for future releases, access changes, deprecation, and maintainer membership.
The first publish therefore belongs in Zy Mazza's `zymazza` npm account. Keep
authentication material outside the repository and retained logs. GPUPDAL does
not need a paid npm organization for a public package.
