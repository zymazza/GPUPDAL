# Third-party licensing inventory

GPUPDAL is derived from PDAL and retains PDAL's BSD 3-Clause license and
attribution. The default release feature set reviewed on 2026-08-21 did not
identify a proprietary dependency. This inventory is a packaging aid, not
legal advice or a substitute for reviewing the exact final binary.

## Source components used by the default build

| Component | License signal retained in this tree |
| --- | --- |
| PDAL-derived source | BSD-3-Clause (`LICENSE.txt`, source headers) |
| Arbiter | MIT (`vendor/arbiter/LICENSE`) |
| Eigen | MPL-2.0 and per-file notices (`vendor/eigen`) |
| H3 | Apache-2.0 source notices (`vendor/h3`) |
| Kazhdan utilities | BSD-style source notices (`vendor/kazhdan`) |
| laz-perf | Apache-2.0 and BSD-style source notices (`vendor/lazperf`) |
| LEPCC | Apache-2.0 source notices (`vendor/lepcc`) |
| nlohmann/json | MIT and embedded per-component notices (`vendor/nlohmann`) |
| JSON schema validator | MIT source notices (`vendor/schema-validator`) |
| utfcpp | Boost-1.0 (`vendor/utfcpp/LICENSE`) |

CUDA builds additionally compile against NVIDIA CCCL (CUB, Thrust, and
libcu++). The controlled CUDA 13 release pins CCCL 3.4.0, records a digest of
the exact header tree, and copies CCCL's complete combined license into the
binary archive. CCCL includes Apache-2.0 with LLVM exceptions, BSD-3-Clause,
Boost-1.0, and other per-file compatible notices; the combined upstream
license is authoritative.

Canonical Apache-2.0 and MPL-2.0 texts are retained under `licenses/` for the
short-form source notices above. Component-specific copyright statements stay
in their source files and in `NOTICE`.

The normal system dependency set includes GDAL, PROJ, GeoTIFF, curl, zlib,
zstd, Threads, and optionally discovered libxml2 support. Their licenses and
transitive dependency set depend on the distribution used to make a binary.
The Linux bundle recipe therefore records every copied shared library in
`RUNTIME_DEPENDENCIES.tsv`, copies package-manager license material when it is
available, and includes a file-level SPDX 2.3 SBOM.

The CUDA artifact redistributes `libcudart.so`, `libnvrtc.so`, and the matching
NVRTC builtins as permitted by NVIDIA's CUDA Toolkit EULA. NVRTC is required
for the measured fused-program specialization lane and is loaded dynamically,
so the release recipe includes it explicitly rather than relying on ELF
dependency discovery. The NVIDIA driver (`libcuda.so` and other
`libnvidia-*` files) is a host prerequisite and is explicitly excluded. The
bundle identifies these redistributables as `NVIDIA CUDA Toolkit`, records the
toolkit version, and includes the complete toolkit EULA under
`licenses/source/cuda/`.

## Excluded from the first supported bundle

All optional plugin switches are off in the maintained release preset. This
excludes CPD, Draco, E57, HDF/Icebridge, MATLAB, MB-System, NITF, OpenSceneGraph,
PostgreSQL pointcloud, RDB/RiVLib, SPZ, TEASER, TileDB, and trajectory/Ceres
integrations from the first bundle. Enabling one changes both the support and
license inventory and requires a new review.

## Binary review rule

Before publication, inspect the generated `RUNTIME_DEPENDENCIES.tsv`, resolve
every entry in `licenses/SYSTEM-LICENSES-MISSING.txt` if that file exists, and
review the notices in the exact archive. GPU artifacts must also review the
CUDA redistributables actually copied; driver libraries are never bundled.
