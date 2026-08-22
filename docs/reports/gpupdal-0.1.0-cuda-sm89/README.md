# GPUPDAL 0.1.0 CUDA/SM 89 qualification evidence

This directory retains the complete bounded conformance report for the clean
Linux x86-64 CUDA 13 release candidate built from commit
`7981754d150a96116875be1fdcac525b52ff4afd`.

The candidate archive is
`gpupdal-0.1.0-linux-x64-cuda13.tar.gz`, 121,512,265 bytes, SHA-256
`f241da5888ac8de837449da3a96bc09d042bb3d265a8d51084ddd88520261841`.
It was physically tested on one NVIDIA GeForce RTX 4090 (compute capability
8.9) with driver 610.43.03. The CUDA toolkit was 13.3.73 and the controlled
CCCL 3.4.0 header tree had SHA-256
`b4410252cb1351a8e350976c55eb8ae097a92cb1a76979f52e6a923bfa4c70a7`.

The generated `pdg-bounded-generated-v1` manifest contained 2,048 complete
cases and had SHA-256
`e0d880fcd33413fc4d2daae67b4ada4bb189d964664dfbe771a1637c65468a1f`.
All 2,048 cases passed, `partial` was false, and the report records zero
unexplained semantic differences. The uncompressed report SHA-256 is
`b1a46bbd2cd1e1a423e877df6d81aec0100328b9c10252ae760489d83837f8a0`.
`conformance/report.json.gz` is a deterministic `gzip -n` encoding of those
bytes.

The outer `gpupdal` and `pdal` files are intentionally identical launcher
scripts, so the report records the same launcher-file hash for both roles.
Their basenames select distinct `libexec/gpupdal` and `libexec/pdal`
executables; every case is a complete-process differential between those two
roles.

The same commit also passed 863/863 registered controlled tests with zero
failures (107 CUDA-labelled and 204 differential), the four-tool eight-test
Compute Sanitizer matrix, an extracted-archive forced fused CUDA/NVRTC
differential without a host toolkit, and a driverless exact-fallback smoke.
Twenty-two explicitly optional external-fixture/profile cases self-skipped.

The staged npm tarball was 122,632,557 bytes with 305 entries and SHA-256
`c2ba3ad8f1f211bd266f593fcef9df8924c482d04a03e3c66faccfaf918a3bcb`.
A clean offline install validated all 296 native checksum entries, then passed
both the forced CUDA differential and driverless exact fallback. This is
qualification evidence, not evidence of npm publication.
