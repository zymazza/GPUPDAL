#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
release_image="${GPUPDAL_RELEASE_IMAGE:-gpupdal-release-debian12:bookworm-20260821}"
cuda_root="${GPUPDAL_CUDA_ROOT:-/opt/cuda}"
cccl_include="${GPUPDAL_CCCL_INCLUDE:-/usr/include/cccl}"
host_build_dir="$(realpath -e -- \
    "${GPUPDAL_CUDA_BUILD_DIR:-/tmp/gpupdal-debian12-cuda13-release}")"

available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
minimum_kib=$((8 * 1024 * 1024))
if (( available_kib < minimum_kib )); then
    echo "CUDA release tests require at least 8 GiB MemAvailable" >&2
    exit 2
fi
if [[ ! -x "${cuda_root}/bin/nvcc" ]]; then
    echo "CUDA toolkit is missing: ${cuda_root}/bin/nvcc" >&2
    exit 2
fi
if [[ ! -f "${host_build_dir}/CMakeCache.txt" ]]; then
    echo "controlled CUDA build is missing; build the CUDA bundle first" >&2
    exit 2
fi

docker run --rm \
    --gpus all \
    --network none \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --user "$(id -u):$(id -g)" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib:/opt/gpupdal-deps/lib \
    --volume "${source_root}:/src" \
    --volume "${host_build_dir}:/src/build/pdg-debian12-cuda13-release" \
    --volume "${cuda_root}:/opt/cuda:ro" \
    --volume "${cccl_include}:/opt/cccl/include:ro" \
    --workdir /src \
    "${release_image}" \
    bash -c '
        set -euo pipefail
        nvidia-smi --query-gpu=name,compute_cap,driver_version \
            --format=csv,noheader
        /opt/cmake/bin/cmake --build \
            /src/build/pdg-debian12-cuda13-release --parallel 1
        PDAL_TEST_VERIFY_KD3_SNAPSHOT=1 /opt/cmake/bin/ctest \
            --test-dir /src/build/pdg-debian12-cuda13-release \
            --output-on-failure --parallel 1
    '
