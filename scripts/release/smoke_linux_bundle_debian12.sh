#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
base_image="debian@sha256:abd67ffcfa541b485a3dff59865ab629aa048a6c613e639d36e7456b0b229241"
archive="$(realpath -e -- "${1:-${source_root}/dist/gpupdal-0.1.0-dev-linux-x64.tar.gz}")"

docker run --rm \
    --network none \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --read-only \
    --user 65534:65534 \
    --tmpfs /work:rw,nosuid,nodev,size=1g \
    --env LC_ALL=C \
    --volume "${archive}:/archive.tar.gz:ro" \
    "${base_image}" \
    /bin/sh -ec '
        mkdir -p /work/bundle
        tar --extract --gzip --file /archive.tar.gz --directory /work/bundle
        cd /work/bundle
        sha256sum --check SHA256SUMS >/dev/null
        ./gpupdal --version >/dev/null
        ./gpupdal --drivers >/work/gpupdal-drivers.txt
        ./pdal --drivers >/work/pdal-drivers.txt
        gpupdal_hash="$(sha256sum /work/gpupdal-drivers.txt | cut -d " " -f 1)"
        pdal_hash="$(sha256sum /work/pdal-drivers.txt | cut -d " " -f 1)"
        test "${gpupdal_hash}" = "${pdal_hash}"
        set +e
        ./gpupdal verify >/work/verify.stdout 2>/work/verify.stderr
        verify_status=$?
        set -e
        test "${verify_status}" -eq 127
        grep -F "verification requires Python 3" /work/verify.stderr >/dev/null
    '

echo "Bare Debian non-root smoke passed: ${archive}"
