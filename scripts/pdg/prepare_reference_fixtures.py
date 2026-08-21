#!/usr/bin/env python3
"""Prepare immutable local fixtures for the expanded reference suite.

The retained 1M AHN-derived LAZ is the only source point cloud.  This script
derives an uncompressed LAS, two deliberately heterogeneous merge inputs, and
a small deterministic EPSG:3857 RGB orthophoto.  It never changes the source
fixture and reports content hashes suitable for the checked-in suite manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


SOURCE_SHA256 = "bb21eb8e5efe33a90639022a7ee9d250d1b38233162f2de6e3d24e08247f948f"
FREEZE_EPOCH = 1704067200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--source-laz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", type=Path)
    parser.add_argument("--freeze-epoch", type=int, default=FREEZE_EPOCH)
    return parser.parse_args()


def oracle_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith("PDG_")
    }
    environment.update({"LC_ALL": "C", "TZ": "UTC"})
    if args.frozen_time_library:
        environment["LD_PRELOAD"] = str(args.frozen_time_library.resolve())
        environment["PDAL_TEST_FROZEN_EPOCH"] = str(args.freeze_epoch)
    return environment


def run_pipeline(oracle: Path, document: dict, environment: dict[str, str]) -> None:
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False) as handle:
        pipeline = Path(handle.name)
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        completed = subprocess.run(
            [str(oracle), "pipeline", str(pipeline)],
            capture_output=True, env=environment, check=False)
        if completed.returncode:
            raise SystemExit(
                f"pinned oracle failed with status {completed.returncode}: "
                f"{completed.stderr.decode('utf-8', 'replace')[:800]}")
    finally:
        pipeline.unlink(missing_ok=True)


def make_point_fixtures(args: argparse.Namespace, environment: dict[str, str]) -> list[Path]:
    source = args.source_laz.resolve()
    uncompressed = (args.output_dir / "ref-1m.las").resolve()
    copc = (args.output_dir / "ref-1m.copc.laz").resolve()
    merge_a = (args.output_dir / "ref-merge-a.laz").resolve()
    merge_b = (args.output_dir / "ref-merge-b.laz").resolve()

    common_writer = {
        "type": "writers.las",
        "minor_version": 4,
        "scale_x": 0.01,
        "scale_y": 0.01,
        "scale_z": 0.01,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "offset_z": 0.0,
    }
    run_pipeline(args.oracle, {"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        dict(common_writer, filename=str(uncompressed),
             dataformat_id=7, compression=False),
    ]}, environment)

    # COPC's default parallel writer does not promise stable container bytes.
    # Keep the input fixture independently regenerable with a fixed seed and
    # one writer thread. Output COPC conformance remains canonical.
    run_pipeline(args.oracle, {"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "writers.copc", "filename": str(copc),
         "threads": 1, "fixed_seed": True,
         "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01,
         "offset_x": 0.0, "offset_y": 0.0, "offset_z": 0.0},
    ]}, environment)

    # West and east halves are disjoint.  The east input is reverse-X sorted
    # and format 3/LAS 1.2, so r13 exercises source order, layout, LAS version,
    # point-format, and header differences without introducing another corpus.
    run_pipeline(args.oracle, {"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.crop",
         "bounds": "([184500,185249.99],[494923.21,494999.99])"},
        dict(common_writer, filename=str(merge_a), dataformat_id=7,
             compression=True),
    ]}, environment)
    run_pipeline(args.oracle, {"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.crop",
         "bounds": "([185250,185999.99],[494923.21,494999.99])"},
        {"type": "filters.sort", "dimension": "X", "order": "DESC"},
        {"type": "writers.las", "filename": str(merge_b),
         "minor_version": 2, "dataformat_id": 3, "compression": True,
         "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01,
         "offset_x": 0.0, "offset_y": 0.0, "offset_z": 0.0},
    ]}, environment)
    return [uncompressed, copc, merge_a, merge_b]


def make_orthophoto(output: Path) -> None:
    try:
        import numpy
        from osgeo import gdal, osr
    except ImportError as error:
        raise SystemExit(
            "orthophoto generation requires the GDAL Python bindings and numpy") from error

    gdal.UseExceptions()
    width, height = 512, 64
    # The middle half of the EPSG:28992 point extent transformed to EPSG:3857.
    # Leaving the outer point quarters uncovered makes out-of-bounds behavior
    # part of the headline workload rather than a synthetic side test.
    minx, maxx = 648611.75, 649840.00
    miny, maxy = 6880341.44, 6880396.89
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(output), width, height, 3, gdal.GDT_UInt16,
        options=["COMPRESS=NONE", "INTERLEAVE=PIXEL", "TILED=NO"])
    dataset.SetGeoTransform(
        (minx, (maxx - minx) / width, 0.0,
         maxy, 0.0, -(maxy - miny) / height))
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(3857)
    dataset.SetProjection(spatial_ref.ExportToWkt())
    rows, columns = numpy.indices((height, width), dtype=numpy.uint32)
    bands = (
        (1000 + 31 * columns + 17 * rows) % 65536,
        (2000 + 13 * columns + 29 * rows) % 65536,
        (3000 + 7 * columns + 37 * rows) % 65536,
    )
    for index, values in enumerate(bands, start=1):
        band = dataset.GetRasterBand(index)
        band.WriteArray(values.astype(numpy.uint16))
        band.SetDescription(("Red", "Green", "Blue")[index - 1])
        band.SetNoDataValue(0)
    dataset.FlushCache()
    dataset = None


def main() -> int:
    args = arguments()
    args.oracle = args.oracle.resolve()
    args.source_laz = args.source_laz.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual_source = sha256(args.source_laz)
    if actual_source != SOURCE_SHA256:
        raise SystemExit(
            f"source fixture hash mismatch: expected {SOURCE_SHA256}, got {actual_source}")
    environment = oracle_environment(args)
    paths = make_point_fixtures(args, environment)
    orthophoto = (args.output_dir / "ref-orthophoto-rgb-3857.tif").resolve()
    make_orthophoto(orthophoto)
    paths.append(orthophoto)
    report = {
        "schema": "pdg-reference-derived-fixtures-v1",
        "source": {
            "path": str(args.source_laz),
            "bytes": args.source_laz.stat().st_size,
            "sha256": actual_source,
            "provenance": "retained AHN benchmark data used since B0146",
            "redistribution": "not asserted; keep local",
        },
        "oracle": {"path": str(args.oracle), "sha256": sha256(args.oracle)},
        "freeze_epoch": args.freeze_epoch,
        "fixtures": [
            {"path": str(path), "bytes": path.stat().st_size,
             "sha256": sha256(path)} for path in paths
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
