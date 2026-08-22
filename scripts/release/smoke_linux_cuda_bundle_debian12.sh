#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
release_image="${GPUPDAL_RELEASE_IMAGE:-gpupdal-release-debian12:bookworm-20260821}"
artifact="$(realpath -e -- \
    "${1:-${source_root}/dist/gpupdal-0.1.0-dev-linux-x64-cuda13.tar.gz}")"

if [[ "$(basename -- "${artifact}")" != gpupdal-*-linux-x64-cuda13.tar.gz ]]; then
    echo "expected a GPUPDAL Linux x86-64 CUDA 13 archive" >&2
    exit 2
fi

docker run --rm \
    --gpus all \
    --network none \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --user "$(id -u):$(id -g)" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --volume "${source_root}:/src:ro" \
    --volume "${artifact}:/artifact/gpupdal.tar.gz:ro" \
    --workdir /tmp \
    "${release_image}" \
    bash -c '
        set -euo pipefail
        /src/scripts/release/verify_cuda_release_device.sh
        if [[ -e /opt/cuda ]]; then
            echo "archive-only CUDA smoke must not mount a host toolkit" >&2
            exit 2
        fi
        artifact_dir="$(mktemp -d /tmp/gpupdal-installed.XXXXXX)"
        work_dir="$(mktemp -d /tmp/gpupdal-fused.XXXXXX)"
        tar -xzf /artifact/gpupdal.tar.gz -C "${artifact_dir}"
        env -i \
            HOME=/tmp \
            PATH=/usr/bin:/bin \
            LC_ALL=C \
            PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM=1 \
            NVIDIA_VISIBLE_DEVICES=all \
            NVIDIA_DRIVER_CAPABILITIES=compute,utility \
            python3 /src/scripts/pdg/differential.py \
                --oracle "${artifact_dir}/pdal" \
                --candidate "${artifact_dir}/gpupdal" \
                --case installed_cuda_assign_ferry_fused \
                --work-dir "${work_dir}" \
                --frozen-time-library \
                    "${artifact_dir}/libexec/libpdg_frozen_time.so" \
                --seed-file input.las=/src/test/data/las/simple.las \
                --seed-file \
                    pipeline.json=/src/test/data/pdg/assign-ferry-fused.json \
                -- pipeline pipeline.json
        "${artifact_dir}/gpupdal" --version
        "${artifact_dir}/gpupdal" --drivers >/dev/null
    '
