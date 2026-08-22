# GPUPDAL npm installer

This directory is the publication source for the future unscoped `gpupdal`
package:

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
rejected by `prepack`. The manual release process stages one version and the
matching Linux x86-64 archive under `dist/npm/gpupdal`, records its relative
path and digest, runs the tests, performs a clean-machine install, and only
then runs an authenticated `npm publish --access public` from the owner's npm
account. See `docs/releasing.md` in the source repository for the exact
commands.
The package intentionally disables npm provenance for now because npm does not
support provenance from a local manual publish or a private source repository.

The initial support declaration is Linux x86-64. Additional platforms require
their own tested artifacts and manifest entries; they must not silently reuse
the Linux bundle.

Publishing the unscoped package makes its current npm maintainers responsible
for future releases, access changes, deprecation, and maintainer membership.
The first publish therefore belongs in Zy Mazza's `zymazza` npm account. Keep
authentication material outside the repository and retained logs. GPUPDAL does
not need a paid npm organization for a public package.
