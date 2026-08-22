#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
release_cuda="${GPUPDAL_RELEASE_CUDA:-0}"
if [[ "${release_cuda}" == "1" ]]; then
    build_dir="${GPUPDAL_RELEASE_BUILD_DIR:-${source_root}/build/pdg-debian12-cuda13-release}"
else
    build_dir="${GPUPDAL_RELEASE_BUILD_DIR:-${source_root}/build/pdg-debian12-release}"
fi
oracle_commit="$(sed -n \
    's/^set(PDG_ORACLE_COMMIT "\([0-9a-f]*\)").*/\1/p' \
    "${source_root}/cmake/pdg-oracle.cmake")"

if [[ ! "${oracle_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "could not read the pinned PDAL oracle commit" >&2
    exit 2
fi

oracle_key="${oracle_commit:0:12}"
oracle_source_dir="${source_root}/build/pdg-pinned-oracle-${oracle_key}-checkout"
oracle_build_dir="${source_root}/build/pdg-debian12-oracle-${oracle_key}-checkout"

if [[ ! -f "${oracle_source_dir}/CMakeLists.txt" ]]; then
    if [[ -e "${oracle_source_dir}" ]]; then
        echo "incomplete pinned-oracle source tree: ${oracle_source_dir}" >&2
        exit 2
    fi
    mkdir -p "${source_root}/build"
    oracle_source_tmp="$(mktemp -d \
        "${source_root}/build/.pdg-pinned-oracle-${oracle_key}.XXXXXX")"
    trap 'rm -rf -- "${oracle_source_tmp}"' EXIT
    git clone --quiet --shared --no-checkout \
        "${source_root}" "${oracle_source_tmp}"
    git -C "${oracle_source_tmp}" checkout --quiet --detach "${oracle_commit}"
    mv "${oracle_source_tmp}" "${oracle_source_dir}"
    trap - EXIT
fi

cmake -S "${oracle_source_dir}" -B "${oracle_build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/gpupdal-deps \
    -DWITH_TESTS=OFF \
    -DWITH_GCS=OFF \
    -DWITH_BACKTRACE=OFF

cmake --build "${oracle_build_dir}" \
    --target pdal \
    --parallel "${GPUPDAL_BUILD_JOBS:-2}"

"${oracle_build_dir}/bin/pdal" --version >/dev/null

gpupdal_version_argument=(-U GPUPDAL_VERSION)
if [[ -n "${GPUPDAL_RELEASE_VERSION:-}" ]]; then
    gpupdal_version_argument=(
        -DGPUPDAL_VERSION="${GPUPDAL_RELEASE_VERSION}"
    )
fi

cuda_arguments=(-DGPUPDAL_ENABLE_CUDA=OFF -DPDG_WARNINGS_AS_ERRORS=ON)
if [[ "${release_cuda}" == "1" ]]; then
    cuda_compiler="${GPUPDAL_CUDA_ROOT:-/opt/cuda}/bin/nvcc"
    if [[ ! -x "${cuda_compiler}" ]]; then
        echo "controlled CUDA release requires ${cuda_compiler}" >&2
        exit 2
    fi
    actual_cuda="$(${cuda_compiler} --version | \
        sed -n 's/.*V\([0-9][0-9.]*\).*/\1/p' | tail -n 1)"
    expected_cuda="${GPUPDAL_CUDA_VERSION:-13.3.73}"
    if [[ "${actual_cuda}" != "${expected_cuda}" ]]; then
        echo "expected CUDA ${expected_cuda}, found ${actual_cuda}" >&2
        exit 2
    fi
    cuda_arguments=(
        -DGPUPDAL_ENABLE_CUDA=ON
        -DCMAKE_CUDA_COMPILER="${cuda_compiler}"
        -DCUDAToolkit_ROOT="${GPUPDAL_CUDA_ROOT:-/opt/cuda}"
        -DPDG_CCCL_INCLUDE_DIR=/opt/cccl/include
        -DPDG_CUDA_ARCHITECTURES=all
        -DPDG_REQUIRE_PORTABLE_CUDA_ARCHITECTURES=ON
        -DPDG_WARNINGS_AS_ERRORS=OFF
    )
fi

cmake -S "${source_root}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/gpupdal-deps \
    -DWITH_PDG=ON \
    -DPDG_BUILD_TESTS=ON \
    -DPDG_BUILD_BENCHMARKS=OFF \
    -DPDG_PINNED_ORACLE_EXECUTABLE="${oracle_build_dir}/bin/pdal" \
    -DWITH_TESTS=OFF \
    -DWITH_GCS=OFF \
    -DWITH_BACKTRACE=OFF \
    "${cuda_arguments[@]}" \
    "${gpupdal_version_argument[@]}"

cmake --build "${build_dir}" \
    --target gpupdal_linux_bundle \
    --parallel "${GPUPDAL_BUILD_JOBS:-2}"
