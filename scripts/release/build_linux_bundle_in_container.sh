#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
build_dir="${source_root}/build/pdg-debian12-release"
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

cmake -S "${source_root}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/gpupdal-deps \
    -DWITH_PDG=ON \
    -DPDG_ENABLE_CUDA=OFF \
    -DPDG_BUILD_TESTS=ON \
    -DPDG_BUILD_BENCHMARKS=OFF \
    -DPDG_WARNINGS_AS_ERRORS=ON \
    -DPDG_PINNED_ORACLE_EXECUTABLE="${oracle_build_dir}/bin/pdal" \
    -DWITH_TESTS=OFF \
    -DWITH_GCS=OFF \
    -DWITH_BACKTRACE=OFF \
    "${gpupdal_version_argument[@]}"

cmake --build "${build_dir}" \
    --target gpupdal_linux_bundle \
    --parallel "${GPUPDAL_BUILD_JOBS:-2}"
