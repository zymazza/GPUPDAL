#!/usr/bin/env bash

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_root="$(CDPATH= cd -- "${script_dir}/../.." && pwd -P)"
build_dir="$(realpath -m -- "${1:-${source_root}/build/pdg-host-release}")"
output_dir="$(realpath -m -- "${2:-${source_root}/dist}")"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "GPUPDAL release bundles currently require Linux x86-64" >&2
    exit 2
fi

for command in cmake ldd patchelf python3 readelf realpath sha256sum tar; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "missing required command: ${command}" >&2
        exit 2
    fi
done

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    if [[ "${build_dir}" != "${source_root}/build/pdg-host-release" ]]; then
        echo "build directory is not configured: ${build_dir}" >&2
        exit 2
    fi
    cmake --preset pdg-host-release
fi

if ! grep -qx 'WITH_PDG:BOOL=ON' "${build_dir}/CMakeCache.txt"; then
    echo "bundle build must be configured with WITH_PDG=ON" >&2
    exit 2
fi

enabled_plugins="$(sed -n 's/^\(BUILD_PLUGIN_[A-Z0-9_]*\):BOOL=ON$/\1/p' \
    "${build_dir}/CMakeCache.txt" | sort -u)"
if [[ -n "${enabled_plugins}" ]]; then
    echo "release bundle requires every optional plugin to be disabled:" >&2
    printf '%s\n' "${enabled_plugins}" >&2
    exit 2
fi

if [[ "${GPUPDAL_SKIP_BUILD:-0}" != "1" ]]; then
    cmake --build "${build_dir}" --target pdg_cli pdal \
        --parallel "${GPUPDAL_BUILD_JOBS:-2}"
fi

version="$(sed -n 's/^set(GPUPDAL_VERSION "\([^"]*\)").*/\1/p' \
    "${source_root}/CMakeLists.txt")"
if [[ "${version}" == *'${'* ]]; then
    version_major="$(sed -n 's/^set(GPUPDAL_VERSION_MAJOR \([^)]*\)).*/\1/p' \
        "${source_root}/CMakeLists.txt")"
    version_minor="$(sed -n 's/^set(GPUPDAL_VERSION_MINOR \([^)]*\)).*/\1/p' \
        "${source_root}/CMakeLists.txt")"
    version_patch="$(sed -n 's/^set(GPUPDAL_VERSION_PATCH \([^)]*\)).*/\1/p' \
        "${source_root}/CMakeLists.txt")"
    version="${version_major}.${version_minor}.${version_patch}-dev"
fi
if [[ -z "${version}" ]]; then
    echo "unable to determine GPUPDAL_VERSION" >&2
    exit 2
fi

commit="$(git -C "${source_root}" rev-parse HEAD)"
source_revision="${commit}"
if [[ -n "$(git -C "${source_root}" status --porcelain --untracked-files=normal)" ]]; then
    if [[ "${version}" != *-dev ]]; then
        echo "refusing to package a non-development version from a dirty tree" >&2
        exit 2
    fi
    source_revision="${commit}-dirty"
fi
created_epoch="${SOURCE_DATE_EPOCH:-$(git -C "${source_root}" show -s --format=%ct HEAD)}"
artifact_name="gpupdal-${version}-linux-x64"
temporary="$(mktemp -d /tmp/gpupdal-bundle.XXXXXX)"
bundle="${temporary}/${artifact_name}"
trap 'rm -rf -- "${temporary}"' EXIT

mkdir -p "${bundle}/libexec" "${bundle}/lib" "${bundle}/licenses/source" \
    "${bundle}/licenses/system" "${bundle}/share"

for executable in gpupdal pdg-engine pdal; do
    source="${build_dir}/bin/${executable}"
    if [[ ! -x "${source}" ]]; then
        echo "built executable is missing: ${source}" >&2
        exit 2
    fi
    install -m 0755 "${source}" "${bundle}/libexec/${executable}"
    install -m 0755 "${script_dir}/linux-launcher.sh" "${bundle}/${executable}"
done

for helper in pdg-verify.py pdg-benchmark-reference.py; do
    source="${build_dir}/bin/${helper}"
    if [[ ! -f "${source}" ]]; then
        echo "built helper is missing: ${source}" >&2
        exit 2
    fi
    install -m 0755 "${source}" "${bundle}/libexec/${helper}"
done

frozen_time="${build_dir}/lib/libpdg_frozen_time.so"
if [[ ! -f "${frozen_time}" ]]; then
    echo "built verification library is missing: ${frozen_time}" >&2
    exit 2
fi
install -m 0755 "${frozen_time}" "${bundle}/libexec/libpdg_frozen_time.so"

install -m 0644 "${source_root}/LICENSE.txt" "${bundle}/LICENSE.txt"
install -m 0644 "${source_root}/NOTICE" "${bundle}/NOTICE"
install -m 0644 "${source_root}/ORIGIN.md" "${bundle}/ORIGIN.md"
install -m 0644 "${source_root}/THIRD_PARTY_LICENSES.md" \
    "${bundle}/THIRD_PARTY_LICENSES.md"
install -m 0644 "${source_root}/licenses/Apache-2.0.txt" \
    "${bundle}/licenses/source/Apache-2.0.txt"
install -m 0644 "${source_root}/licenses/MPL-2.0.txt" \
    "${bundle}/licenses/source/MPL-2.0.txt"

for license in \
    vendor/arbiter/LICENSE \
    vendor/utfcpp/LICENSE; do
    destination="${bundle}/licenses/source/${license%/LICENSE}"
    mkdir -p "${destination}"
    install -m 0644 "${source_root}/${license}" "${destination}/LICENSE"
done

dependency_map="${bundle}/RUNTIME_DEPENDENCIES.tsv"
printf 'soname\tsource_path\tpackage\tversion\tlicense_declared\n' \
    >"${dependency_map}"
missing_licenses="${bundle}/licenses/SYSTEM-LICENSES-MISSING.txt"
: >"${missing_licenses}"

declare -a scan_queue=(
    "${bundle}/libexec/gpupdal"
    "${bundle}/libexec/pdg-engine"
    "${bundle}/libexec/pdal"
    "${bundle}/libexec/libpdg_frozen_time.so"
)
declare -A copied_sonames=()
declare -A copied_licenses=()

package_for_path() {
    local path="$1"
    if command -v pacman >/dev/null 2>&1; then
        pacman -Qoq "${path}" 2>/dev/null | head -n 1 || true
    elif command -v dpkg-query >/dev/null 2>&1; then
        dpkg-query -S "${path}" 2>/dev/null | head -n 1 | cut -d: -f1 || true
    fi
}

package_version() {
    local package="$1"
    if [[ -z "${package}" ]]; then
        return 0
    fi
    if command -v pacman >/dev/null 2>&1; then
        pacman -Q "${package}" 2>/dev/null | awk '{print $2}' || true
    elif command -v dpkg-query >/dev/null 2>&1; then
        dpkg-query -W -f='${Version}' "${package}" 2>/dev/null || true
    fi
}

package_license() {
    local package="$1"
    if [[ -z "${package}" ]]; then
        return 0
    fi
    if command -v pacman >/dev/null 2>&1; then
        pacman -Qi "${package}" 2>/dev/null | \
            sed -n 's/^Licenses[[:space:]]*: //p' || true
    else
        printf 'unknown'
    fi
}

copy_package_license() {
    local package="$1"
    [[ -n "${package}" ]] || return 0
    [[ -z "${copied_licenses[${package}]:-}" ]] || return 0
    copied_licenses["${package}"]=1

    local destination="${bundle}/licenses/system/${package}"
    if [[ -d "/usr/share/licenses/${package}" ]]; then
        cp -aL "/usr/share/licenses/${package}" "${destination}"
    elif [[ -f "/usr/share/doc/${package}/copyright" ]]; then
        mkdir -p "${destination}"
        install -m 0644 "/usr/share/doc/${package}/copyright" \
            "${destination}/copyright"
    else
        printf '%s\n' "${package}" >>"${missing_licenses}"
    fi
}

copy_dependency() {
    local soname="$1"
    local source_path="$2"

    case "${soname}" in
        */ld-linux*.so.*|linux-vdso.so.*|ld-linux*.so.*|libc.so.*|libm.so.*|libpthread.so.*|\
        libdl.so.*|librt.so.*|libresolv.so.*|libutil.so.*|libanl.so.*|\
        libcuda.so.*|libnvidia-*.so.*)
            return 0
            ;;
    esac
    [[ -z "${copied_sonames[${soname}]:-}" ]] || return 0
    copied_sonames["${soname}"]=1

    local resolved
    resolved="$(readlink -f -- "${source_path}")"
    if [[ ! -f "${resolved}" ]]; then
        echo "runtime dependency is missing: ${soname} (${source_path})" >&2
        exit 2
    fi
    install -m 0755 "${resolved}" "${bundle}/lib/${soname}"
    scan_queue+=("${bundle}/lib/${soname}")

    local package version_value license_value recorded_path
    if [[ "${resolved}" == "${build_dir}"/* ]]; then
        package="GPUPDAL"
        version_value="${version}"
        license_value="BSD-3-Clause"
    else
        package="$(package_for_path "${resolved}")"
        version_value="$(package_version "${package}")"
        license_value="$(package_license "${package}")"
    fi
    if [[ "${resolved}" == "${build_dir}"/* ]]; then
        recorded_path="build-output/${resolved#"${build_dir}"/}"
    else
        recorded_path="${resolved}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "${soname}" "${recorded_path}" \
        "${package:-unmanaged}" "${version_value:-unknown}" \
        "${license_value:-unknown}" >>"${dependency_map}"
    if [[ "${package}" != "GPUPDAL" ]]; then
        copy_package_license "${package}"
    fi
}

scan_index=0
while (( scan_index < ${#scan_queue[@]} )); do
    elf="${scan_queue[${scan_index}]}"
    ((scan_index += 1))
    missing="$(LC_ALL=C ldd "${elf}" 2>/dev/null | \
        awk '/=> not found/ {print $1}' | paste -sd, -)"
    if [[ -n "${missing}" ]]; then
        echo "unresolved runtime dependencies for ${elf}: ${missing}" >&2
        exit 2
    fi
    while IFS=$'\t' read -r soname source_path; do
        [[ -n "${soname}" && -n "${source_path}" ]] || continue
        copy_dependency "${soname}" "${source_path}"
    done < <(LC_ALL=C ldd "${elf}" 2>/dev/null | \
        awk '$2 == "=>" && $3 ~ /^\// {print $1 "\t" $3}')
done

for elf in "${bundle}"/libexec/*; do
    if readelf -h "${elf}" >/dev/null 2>&1; then
        patchelf --set-rpath '$ORIGIN/../lib' "${elf}"
    fi
done
for elf in "${bundle}"/lib/*; do
    if readelf -h "${elf}" >/dev/null 2>&1; then
        patchelf --set-rpath '$ORIGIN' "${elf}"
    fi
done

for data_directory in gdal proj; do
    if [[ -d "/usr/share/${data_directory}" ]]; then
        cp -aL "/usr/share/${data_directory}" \
            "${bundle}/share/${data_directory}"
    fi
done

if [[ ! -s "${missing_licenses}" ]]; then
    rm -f -- "${missing_licenses}"
fi

sorted_dependency_map="${temporary}/RUNTIME_DEPENDENCIES.sorted.tsv"
{
    head -n 1 "${dependency_map}"
    tail -n +2 "${dependency_map}" | sort -u
} >"${sorted_dependency_map}"
install -m 0644 "${sorted_dependency_map}" "${dependency_map}"
python3 "${script_dir}/generate_spdx_sbom.py" \
    --root "${bundle}" \
    --version "${version}" \
    --commit "${source_revision}" \
    --created-epoch "${created_epoch}" \
    --output "${bundle}/SBOM.spdx.json"

if LC_ALL=C grep -r -a -l -F -- "${source_root}" "${bundle}" \
    >"${temporary}/absolute-build-paths.txt"; then
    echo "bundle embeds the local source path in:" >&2
    sed "s#^${bundle}/##" "${temporary}/absolute-build-paths.txt" >&2
    exit 2
fi

(
    cd "${bundle}"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | \
        xargs -0 sha256sum >SHA256SUMS
)

mkdir -p "${temporary}/home"
clean_environment=(env -i HOME="${temporary}/home" PATH=/usr/bin:/bin LC_ALL=C)
"${clean_environment[@]}" "${bundle}/gpupdal" --version >/dev/null
"${clean_environment[@]}" "${bundle}/gpupdal" --drivers \
    >"${temporary}/gpupdal-drivers.txt"
"${clean_environment[@]}" "${bundle}/pdal" --drivers \
    >"${temporary}/pdal-drivers.txt"
cmp "${temporary}/gpupdal-drivers.txt" "${temporary}/pdal-drivers.txt"

mkdir -p "${output_dir}"
artifact="${output_dir}/${artifact_name}.tar.gz"
tar --sort=name --mtime="@${created_epoch}" --owner=0 --group=0 \
    --numeric-owner -C "${bundle}" -czf "${artifact}" .
(
    cd "${output_dir}"
    sha256sum "${artifact_name}.tar.gz" >"${artifact_name}.tar.gz.sha256"
)

extracted="${temporary}/npm-layout"
mkdir -p "${extracted}"
tar --extract --gzip --file "${artifact}" --directory "${extracted}" \
    --no-same-owner --no-same-permissions
(
    cd "${extracted}"
    sha256sum --check SHA256SUMS >/dev/null
)
for executable in gpupdal pdg-engine pdal; do
    if [[ ! -x "${extracted}/${executable}" ]]; then
        echo "archive root is missing npm executable: ${executable}" >&2
        exit 2
    fi
done
"${clean_environment[@]}" "${extracted}/gpupdal" --version >/dev/null
"${clean_environment[@]}" "${extracted}/libexec/gpupdal" --version >/dev/null
"${clean_environment[@]}" "${extracted}/libexec/pdg-engine" --version >/dev/null
"${clean_environment[@]}" "${extracted}/gpupdal" verify \
    --points 16 --runs 3 --warmups 1 \
    --output-dir "${temporary}/extracted-verify" >/dev/null

echo "Created ${artifact}"
echo "Created ${artifact}.sha256"
