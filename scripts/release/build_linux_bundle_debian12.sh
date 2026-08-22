#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
base_image="debian@sha256:abd67ffcfa541b485a3dff59865ab629aa048a6c613e639d36e7456b0b229241"
release_image="${GPUPDAL_RELEASE_IMAGE:-gpupdal-release-debian12:bookworm-20260821}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required for the controlled Debian 12 release build" >&2
    exit 2
fi

docker build \
    --file "${script_dir}/Dockerfile.debian12" \
    --tag "${release_image}" \
    "${script_dir}"

docker run --rm \
    --network none \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --user "$(id -u):$(id -g)" \
    --env GPUPDAL_BUILD_JOBS="${GPUPDAL_BUILD_JOBS:-2}" \
    --env GPUPDAL_RELEASE_BASELINE="Debian GNU/Linux 12 (bookworm)" \
    --env GPUPDAL_RELEASE_BASE_IMAGE="${base_image}" \
    --env GPUPDAL_RELEASE_VERSION="${GPUPDAL_RELEASE_VERSION:-}" \
    --env GPUPDAL_GDAL_PREFIX=/opt/gpupdal-deps \
    --env GPUPDAL_PRIVATE_SOURCE_ROOT="${source_root}" \
    --volume "${source_root}:/src" \
    --workdir /src \
    "${release_image}" \
    /src/scripts/release/build_linux_bundle_in_container.sh
