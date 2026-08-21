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
rejected by `prepack`. Release automation must set one version in both
`package.json` and `native-manifest.json`, add the immutable Linux x86-64 asset
URL and digest, run the tests, perform a clean-machine install, and only then
run `npm publish --provenance`.

The initial support declaration is Linux x86-64. Additional platforms require
their own tested artifacts and manifest entries; they must not silently reuse
the Linux bundle.
