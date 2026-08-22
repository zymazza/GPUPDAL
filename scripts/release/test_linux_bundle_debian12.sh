#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
release_image="${GPUPDAL_RELEASE_IMAGE:-gpupdal-release-debian12:bookworm-20260821}"
build_jobs="${GPUPDAL_BUILD_JOBS:-2}"
available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
minimum_kib=$((8 * 1024 * 1024))
available_disk_kib="$(df --output=avail -k "${source_root}" | tail -n 1)"
minimum_disk_kib=$((4 * 1024 * 1024))

if (( available_kib < minimum_kib )); then
    echo "release tests require at least 8 GiB MemAvailable" >&2
    exit 2
fi
if (( available_disk_kib < minimum_disk_kib )); then
    echo "release tests require at least 4 GiB free in the workspace filesystem" >&2
    exit 2
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "npm is required for the release package tests" >&2
    exit 2
fi
if ! docker image inspect "${release_image}" >/dev/null 2>&1; then
    echo "controlled release image is missing; run build_linux_bundle_debian12.sh first" >&2
    exit 2
fi

docker run --rm \
    --network none \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --user "$(id -u):$(id -g)" \
    --env GPUPDAL_BUILD_JOBS="${build_jobs}" \
    --volume "${source_root}:/src" \
    --workdir /src \
    "${release_image}" \
    bash -c '
        set -euo pipefail
        /opt/cmake/bin/cmake -S /src -B /src/build/pdg-debian12-release
        /opt/cmake/bin/cmake --build /src/build/pdg-debian12-release \
            --target pdg_unit_tests pdg_differential_prerequisites \
            --parallel "${GPUPDAL_BUILD_JOBS}"
        /src/build/pdg-debian12-release/bin/pdg_unit_tests \
            --gtest_color=no
        /opt/cmake/bin/ctest --test-dir /src/build/pdg-debian12-release \
            -L differential --output-on-failure \
            --parallel 1
    '

npm test --prefix "${source_root}/packages/npm"
