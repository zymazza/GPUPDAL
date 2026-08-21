#!/bin/sh

set -eu

bundle_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
program=$(basename -- "$0")

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${bundle_root}/lib:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${bundle_root}/lib"
fi

if [ -d "${bundle_root}/share/gdal" ]; then
    export GDAL_DATA="${GDAL_DATA:-${bundle_root}/share/gdal}"
fi
if [ -d "${bundle_root}/share/proj" ]; then
    export PROJ_DATA="${PROJ_DATA:-${bundle_root}/share/proj}"
fi

exec "${bundle_root}/libexec/${program}" "$@"
