#!/usr/bin/env bash

set -euo pipefail

expected_name="NVIDIA GeForce RTX 4090"
expected_compute="8.9"
expected_driver="610.43.03"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for physical CUDA release qualification" >&2
    exit 2
fi
mapfile -t devices < <(nvidia-smi \
    --query-gpu=name,compute_cap,driver_version \
    --format=csv,noheader,nounits)
if (( ${#devices[@]} != 1 )); then
    echo "CUDA release qualification requires exactly one visible GPU" >&2
    printf 'visible GPU: %s\n' "${devices[@]}" >&2
    exit 2
fi

IFS=, read -r actual_name actual_compute actual_driver <<<"${devices[0]}"
trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}
actual_name="$(trim "${actual_name}")"
actual_compute="$(trim "${actual_compute}")"
actual_driver="$(trim "${actual_driver}")"

if [[ "${actual_name}" != "${expected_name}" ||
        "${actual_compute}" != "${expected_compute}" ||
        "${actual_driver}" != "${expected_driver}" ]]; then
    echo "CUDA release qualification device does not match the approved profile" >&2
    printf 'expected: %s, compute %s, driver %s\n' \
        "${expected_name}" "${expected_compute}" "${expected_driver}" >&2
    printf 'actual:   %s, compute %s, driver %s\n' \
        "${actual_name}" "${actual_compute}" "${actual_driver}" >&2
    exit 2
fi

printf 'qualified CUDA device: %s, compute %s, driver %s\n' \
    "${actual_name}" "${actual_compute}" "${actual_driver}"
