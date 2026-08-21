#!/usr/bin/env python3
"""Create a read-only manifest of local geospatial test candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path


FORMATS = (
    (".copc.laz", "copc"),
    ("ept.json", "ept"),
    (".geojson", "geojson"),
    (".tiff", "geotiff"),
    (".gpkg", "geopackage"),
    (".las", "las"),
    (".laz", "laz"),
    (".ply", "ply"),
    (".pcd", "pcd"),
    (".tif", "geotiff"),
    (".vrt", "gdal-vrt"),
    (".shp", "shapefile"),
)

SKIP_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "node_modules",
    "target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--hash-max-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="hash files no larger than this; use -1 to hash every file",
    )
    parser.add_argument("--max-files", type=int, default=100_000)
    return parser.parse_args()


def detected_format(path: Path) -> str | None:
    name = path.name.lower()
    for suffix, format_name in FORMATS:
        if name.endswith(suffix):
            return format_name
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def las_header(path: Path) -> dict[str, object]:
    """Read only the public LAS header shared by LAS, LAZ, and COPC."""
    with path.open("rb") as stream:
        data = stream.read(375)
    if len(data) < 227:
        return {"status": "invalid", "error": "shorter than 227-byte header"}
    if data[:4] != b"LASF":
        return {"status": "invalid", "error": "missing LASF signature"}

    def value(format_code: str, offset: int):
        return struct.unpack_from("<" + format_code, data, offset)[0]

    version_major = value("B", 24)
    version_minor = value("B", 25)
    if version_major != 1 or version_minor > 4:
        return {
            "status": "unsupported",
            "error": f"LAS version {version_major}.{version_minor}",
        }
    if version_minor >= 4 and len(data) < 375:
        return {"status": "invalid", "error": "truncated LAS 1.4 header"}

    raw_point_format = value("B", 104)
    legacy_count = value("I", 107)
    extended_count = value("Q", 247) if version_minor >= 4 else None
    point_count = extended_count if extended_count is not None else legacy_count
    return {
        "status": "ok",
        "version": f"{version_major}.{version_minor}",
        "header_bytes": value("H", 94),
        "point_data_offset": value("I", 96),
        "vlr_count": value("I", 100),
        "point_format": raw_point_format & 0x3F,
        "point_format_raw": raw_point_format,
        "compressed": bool(raw_point_format & 0xC0),
        "point_record_bytes": value("H", 105),
        "point_count": point_count,
        "legacy_point_count": legacy_count,
        "extended_point_count": extended_count,
        "scale": [value("d", 131), value("d", 139), value("d", 147)],
        "offset": [value("d", 155), value("d", 163), value("d", 171)],
        "bounds": {
            "maxx": value("d", 179),
            "minx": value("d", 187),
            "maxy": value("d", 195),
            "miny": value("d", 203),
            "maxz": value("d", 211),
            "minz": value("d", 219),
        },
    }


def candidates(root: Path):
    if root.is_file():
        if detected_format(root):
            yield root
        return
    for directory, child_directories, filenames in os.walk(
        root, followlinks=False
    ):
        child_directories[:] = sorted(
            name
            for name in child_directories
            if name not in SKIP_DIRECTORIES
            and not name.startswith("build-")
            and not name.startswith("build_")
        )
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if detected_format(path):
                yield path


def inspect(path: Path, hash_max_bytes: int) -> dict[str, object]:
    stat = path.stat()
    format_name = detected_format(path)
    should_hash = hash_max_bytes < 0 or stat.st_size <= hash_max_bytes
    record: dict[str, object] = {
        "path": str(path.resolve()),
        "format": format_name,
        "bytes": stat.st_size,
        "provenance": "unknown-local",
        "redistributable": False,
    }
    if format_name in {"las", "laz", "copc"}:
        record["las_header"] = las_header(path)
    if should_hash:
        record["sha256"] = sha256(path)
        record["hash_status"] = "complete"
    else:
        record["sha256"] = None
        record["hash_status"] = "skipped-size-limit"
    return record


def main() -> int:
    args = parse_args()
    roots = sorted({root.expanduser().resolve() for root in args.roots})
    missing = [root for root in roots if not root.exists()]
    if missing:
        for root in missing:
            print(f"missing corpus root: {root}", file=sys.stderr)
        return 2

    paths = sorted({path.resolve() for root in roots for path in candidates(root)})
    if len(paths) > args.max_files:
        print(
            f"discovered {len(paths)} files, exceeding --max-files={args.max_files}",
            file=sys.stderr,
        )
        return 2

    records = [inspect(path, args.hash_max_bytes) for path in paths]
    by_format: dict[str, int] = {}
    las_strata: dict[str, int] = {}
    las_header_status: dict[str, int] = {}
    total_bytes = 0
    for record in records:
        format_name = str(record["format"])
        by_format[format_name] = by_format.get(format_name, 0) + 1
        total_bytes += int(record["bytes"])
        header = record.get("las_header")
        if isinstance(header, dict):
            status = str(header["status"])
            las_header_status[status] = las_header_status.get(status, 0) + 1
            if status == "ok":
                stratum = (
                    f"{header['version']}/format-{header['point_format']}/"
                    f"record-{header['point_record_bytes']}/"
                    f"compressed-{str(header['compressed']).lower()}"
                )
                las_strata[stratum] = las_strata.get(stratum, 0) + 1

    manifest = {
        "schema": 1,
        "roots": [str(root) for root in roots],
        "file_count": len(records),
        "total_bytes": total_bytes,
        "by_format": dict(sorted(by_format.items())),
        "las_header_status": dict(sorted(las_header_status.items())),
        "las_strata": dict(sorted(las_strata.items())),
        "files": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "file_count",
                    "total_bytes",
                    "by_format",
                    "las_header_status",
                    "las_strata",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
