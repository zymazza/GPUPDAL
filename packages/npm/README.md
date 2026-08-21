# GPUPDAL npm installer

This directory is the publication source for the future unscoped `gpupdal`
package:

```sh
npm install gpupdal
npx gpupdal --version
```

For a shell-wide command, use `npm install --global gpupdal` followed by
`gpupdal --version`.

GPUPDAL is native software. The npm package is a small launcher and installer,
not a JavaScript reimplementation. On install it selects the declared
platform artifact, downloads it from the matching GitHub release, verifies the
archive's SHA-256 digest, and installs the complete sibling bundle needed by
the launcher (`gpupdal`, `pdg-engine`, and the pinned `pdal` oracle).

The checked-in manifest is intentionally empty and the development version is
rejected by `prepack`. The manual release process must set one version in both
`package.json` and `native-manifest.json`, add the immutable Linux x86-64 asset
URL and digest, run the tests, perform a clean-machine install, and only then
run an interactive `npm publish --access public` from the owner's npm account.
The package intentionally disables npm provenance for now because npm does not
support provenance from a local manual publish or a private source repository.

The initial support declaration is Linux x86-64. Additional platforms require
their own tested artifacts and manifest entries; they must not silently reuse
the Linux bundle.

Publishing the unscoped package makes its current npm maintainers responsible
for future releases, access changes, deprecation, and maintainer membership.
The first publish therefore belongs in Zy Mazza's `zymazza` npm account with
account 2FA and recovery methods enabled; GPUPDAL does not need a paid npm
organization for a public package.
