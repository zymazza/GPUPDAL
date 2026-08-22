#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
release_image="${GPUPDAL_RELEASE_IMAGE:-gpupdal-release-debian12:bookworm-20260821}"
cuda_root="${GPUPDAL_CUDA_ROOT:-/opt/cuda}"
cccl_include="${GPUPDAL_CCCL_INCLUDE:-/usr/include/cccl}"
host_build_dir="$(realpath -e -- \
    "${GPUPDAL_CUDA_BUILD_DIR:-/tmp/gpupdal-debian12-cuda13-release}")"
test_filter='CudaLasPointProgram.*SanitizerSmoke:CudaSpatialIndex.CellTableAndRadiusCountsMatchHost:CudaSpatialIndex.KnnGatherMatchesUniqueHostOrder:ResidentExecutionContextCuda.*'

available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
minimum_kib=$((8 * 1024 * 1024))
if (( available_kib < minimum_kib )); then
    echo "CUDA sanitizer qualification requires at least 8 GiB MemAvailable" >&2
    exit 2
fi
if [[ ! -x "${cuda_root}/bin/compute-sanitizer" ]]; then
    echo "Compute Sanitizer is missing: ${cuda_root}/bin/compute-sanitizer" >&2
    exit 2
fi
if [[ ! -x "${host_build_dir}/bin/pdg_unit_tests" ]]; then
    echo "controlled CUDA unit binary is missing; build and test first" >&2
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
    --env GPUPDAL_SANITIZER_FILTER="${test_filter}" \
    --volume "${source_root}:/src" \
    --volume "${host_build_dir}:/src/build/pdg-debian12-cuda13-release" \
    --volume "${cuda_root}:/opt/cuda:ro" \
    --volume "${cccl_include}:/opt/cccl/include:ro" \
    --workdir /src \
    "${release_image}" \
    bash -c '
        set -euo pipefail
        for tool in memcheck initcheck synccheck racecheck; do
            log="$(mktemp "/tmp/gpupdal-${tool}.XXXXXX")"
            set +e
            /opt/cuda/bin/compute-sanitizer \
                --tool "${tool}" \
                --error-exitcode 99 \
                /src/build/pdg-debian12-cuda13-release/bin/pdg_unit_tests \
                "--gtest_filter=${GPUPDAL_SANITIZER_FILTER}" \
                2>&1 | tee "${log}"
            status="${PIPESTATUS[0]}"
            set -e
            if (( status != 0 )); then
                exit "${status}"
            fi
            if [[ "${tool}" == racecheck ]]; then
                grep -Eq "RACECHECK SUMMARY: 0 hazards.*0 errors, 0 warnings" \
                    "${log}"
            else
                grep -Fq "ERROR SUMMARY: 0 errors" "${log}"
            fi
            rm -f -- "${log}"
        done
    '
