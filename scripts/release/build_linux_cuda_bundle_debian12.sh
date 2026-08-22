#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
base_image="debian@sha256:abd67ffcfa541b485a3dff59865ab629aa048a6c613e639d36e7456b0b229241"
release_image="${GPUPDAL_RELEASE_IMAGE:-gpupdal-release-debian12:bookworm-20260821}"
cuda_root="${GPUPDAL_CUDA_ROOT:-/opt/cuda}"
cccl_include="${GPUPDAL_CCCL_INCLUDE:-/usr/include/cccl}"
cccl_license="${GPUPDAL_CCCL_LICENSE:-/usr/share/licenses/cccl/LICENSE}"
host_build_dir="$(realpath -m -- \
    "${GPUPDAL_CUDA_BUILD_DIR:-/tmp/gpupdal-debian12-cuda13-release}")"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required for the controlled Debian 12 CUDA release build" >&2
    exit 2
fi
if [[ ! -x "${cuda_root}/bin/nvcc" ]]; then
    echo "CUDA toolkit is missing: ${cuda_root}/bin/nvcc" >&2
    exit 2
fi
if [[ ! -f "${cccl_include}/cub/device/device_select.cuh" ||
        ! -f "${cccl_license}" ]]; then
    echo "CCCL 3.4 headers or license are missing" >&2
    exit 2
fi
cccl_version="$(sed -n \
    's/^#define CUB_VERSION \([0-9][0-9]*\).*/\1/p' \
    "${cccl_include}/cub/version.cuh")"
if [[ "${cccl_version}" != "300400" ]]; then
    echo "expected CCCL/CUB 3.4.0, found encoded version ${cccl_version}" >&2
    exit 2
fi
cccl_tree_sha256="$(find "${cccl_include}" -type f -print0 | sort -z | \
    xargs -0 sha256sum | sha256sum | cut -d ' ' -f 1)"
expected_cccl_tree_sha256="b4410252cb1351a8e350976c55eb8ae097a92cb1a76979f52e6a923bfa4c70a7"
if [[ "${cccl_tree_sha256}" != "${expected_cccl_tree_sha256}" ]]; then
    echo "CCCL 3.4 header tree does not match the approved release input" >&2
    echo "expected ${expected_cccl_tree_sha256}" >&2
    echo "actual   ${cccl_tree_sha256}" >&2
    exit 2
fi

available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
minimum_kib=$((8 * 1024 * 1024))
available_tmp_kib="$(df --output=avail -k /tmp | tail -n 1)"
minimum_tmp_kib=$((18 * 1024 * 1024))
if (( available_kib < minimum_kib )); then
    echo "CUDA release build requires at least 8 GiB MemAvailable" >&2
    exit 2
fi
if (( available_tmp_kib < minimum_tmp_kib )); then
    echo "CUDA release build requires at least 18 GiB free under /tmp" >&2
    exit 2
fi

mkdir -p "${host_build_dir}"

docker build \
    --file "${script_dir}/Dockerfile.debian12" \
    --tag "${release_image}" \
    "${script_dir}"

docker run --rm \
    --gpus all \
    --network none \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --user "$(id -u):$(id -g)" \
    --env GPUPDAL_BUILD_JOBS=1 \
    --env GPUPDAL_RELEASE_BASELINE="Debian GNU/Linux 12 (bookworm); CUDA 13.3" \
    --env GPUPDAL_RELEASE_BASE_IMAGE="${base_image}" \
    --env GPUPDAL_RELEASE_VERSION="${GPUPDAL_RELEASE_VERSION:-}" \
    --env GPUPDAL_RELEASE_CUDA=1 \
    --env GPUPDAL_RELEASE_BUILD_DIR=/src/build/pdg-debian12-cuda13-release \
    --env GPUPDAL_CUDA_ROOT=/opt/cuda \
    --env GPUPDAL_CUDA_VERSION=13.3.73 \
    --env GPUPDAL_CCCL_VERSION=3.4.0 \
    --env GPUPDAL_CCCL_SHA256="${cccl_tree_sha256}" \
    --env GPUPDAL_CCCL_LICENSE=/opt/cccl/LICENSE \
    --env GPUPDAL_CUDA_LICENSE=/opt/cuda/EULA.txt \
    --env GPUPDAL_GDAL_PREFIX=/opt/gpupdal-deps \
    --env LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib:/opt/gpupdal-deps/lib \
    --env GPUPDAL_PRIVATE_SOURCE_ROOT="${source_root}" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --volume "${source_root}:/src" \
    --volume "${host_build_dir}:/src/build/pdg-debian12-cuda13-release" \
    --volume "${cuda_root}:/opt/cuda:ro" \
    --volume "${cccl_include}:/opt/cccl/include:ro" \
    --volume "${cccl_license}:/opt/cccl/LICENSE:ro" \
    --workdir /src \
    "${release_image}" \
    bash -c '
        set -euo pipefail
        /src/scripts/release/verify_cuda_release_device.sh
        exec /src/scripts/release/build_linux_bundle_in_container.sh
    '
