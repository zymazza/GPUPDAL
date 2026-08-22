# GPUPDAL npm installer

This directory is the publication source for the unscoped `gpupdal` package:

```sh
npm install gpupdal
npx gpupdal --version
```

For a shell-wide command, use `npm install --global gpupdal` followed by
`gpupdal --version`.

GPUPDAL is native software. The npm package is a small launcher distribution,
not a JavaScript reimplementation. The immutable npm package carries the
declared platform files themselves, so installation does not depend on access
to the source repository, a second download host, or an npm lifecycle script.
Before packing, every file is verified against the release archive's internal
`SHA256SUMS`. The native tree contains the complete sibling bundle needed by
the launcher (`gpupdal`, `pdg-engine`, and the pinned `pdal` oracle).

The checked-in manifest is intentionally empty and the development version is
rejected by `prepack`. The manual release process stages one version and its
matching Linux x86-64 and Windows x64 archives under `dist/npm/gpupdal`,
records both source digests, runs the tests, performs clean-machine installs,
and only then runs an authenticated `npm publish --access public --tag latest`
from the owner's npm account. Staging requires `tar` and `unzip`. See
`docs/releasing.md` in the source repository for the exact commands.
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
