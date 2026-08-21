#!/usr/bin/env python3
"""Stage the benchmark inputs (derived copies only; sources are never touched).

For every size S in {1m, 4m, 16m, 47m} this creates under build/independent-bench/in:
  S.laz          the point cloud, LAZ, EPSG:28992 assigned (written by pinned PDAL)
  S.las          the same points uncompressed
  S-ground.laz   the same points after ground classification (pinned PDAL filters.smrf)
  S-west.laz / S-east.laz   two halves (for the merge task)
  S-clip.gpkg / S-clip.shp  a clipping multipolygon with a hole (for the clip task)
  S-extent.json  bounds and point count
and for 1m also 1m.copc.laz and 1m-ortho-3857.tif copies of the reference fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import sys

from osgeo import ogr, osr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import IN, SIZES, SOURCES, SRS, TOOLS, dump_json, extent_of, run  # noqa: E402


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: pathlib.Path, root: pathlib.Path | None = None) -> dict:
    return {"path": (path.relative_to(root).as_posix() if root else str(path.resolve())),
            "bytes": path.stat().st_size, "sha256": sha256_of(path)}


def validate_existing_manifest(force: bool) -> None:
    manifest_path = IN / "input-manifest.json"
    if force or not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "pdg-separate-benchmark-inputs-v1":
        raise RuntimeError("unsupported staged-input manifest; use --force intentionally")
    for expected in manifest.get("staged_files", []):
        path = IN / expected["path"]
        if (not path.is_file() or path.stat().st_size != expected["bytes"] or
                sha256_of(path) != expected["sha256"]):
            raise RuntimeError(f"staged input differs from its frozen manifest: {path}")


def write_input_manifest() -> None:
    source_paths = []
    for size in SIZES:
        for role, value in SOURCES[size].items():
            if role != "label" and isinstance(value, pathlib.Path) and value.is_file():
                source_paths.append({"size": size, "role": role,
                                     **file_record(value)})
    staged = [file_record(path, IN) for path in sorted(IN.rglob("*"))
              if path.is_file() and path.name != "input-manifest.json"]
    manifest = {
        "schema": "pdg-separate-benchmark-inputs-v1",
        "source_files": source_paths, "staged_files": staged,
        "rule": "Sources are read-only. Existing staged bytes are checked before reuse; regenerate only with explicit --force.",
    }
    dump_json(manifest, IN / "input-manifest.json")


def stage_size(size: str, pdal: str, force: bool) -> None:
    src = SOURCES[size]
    IN.mkdir(parents=True, exist_ok=True)
    laz = IN / f"{size}.laz"
    las = IN / f"{size}.las"
    ground = IN / f"{size}-ground.laz"
    west = IN / f"{size}-west.laz"
    east = IN / f"{size}-east.laz"
    source = src.get("laz") or src.get("las")

    if force or not laz.exists():
        print(f"[{size}] writing {laz.name} from {source.name}")
        run([pdal, "translate", source, laz,
             f"--writers.las.a_srs={SRS}", "--writers.las.forward=all",
             "--writers.las.compression=true"])
    if force or not las.exists():
        print(f"[{size}] writing {las.name}")
        run([pdal, "translate", laz, las, "--writers.las.forward=all"])
    ext = extent_of(pdal, laz)
    dump_json(ext, IN / f"{size}-extent.json")

    if force or not ground.exists():
        print(f"[{size}] ground classification (SMRF) -> {ground.name}")
        run([pdal, "translate", laz, ground, "smrf",
             "--writers.las.forward=all", "--writers.las.compression=true"])

    midx = (ext["minx"] + ext["maxx"]) / 2.0
    if force or not west.exists() or not east.exists():
        print(f"[{size}] halves for merge")
        run([pdal, "translate", laz, west, "crop",
             f"--filters.crop.bounds=([{ext['minx'] - 1},{midx}],[{ext['miny'] - 1},{ext['maxy'] + 1}])",
             "--writers.las.forward=all", "--writers.las.compression=true"])
        run([pdal, "translate", laz, east, "crop",
             f"--filters.crop.bounds=([{midx},{ext['maxx'] + 1}],[{ext['miny'] - 1},{ext['maxy'] + 1}])",
             "--writers.las.forward=all", "--writers.las.compression=true"])

    write_clip_polygon(size, ext, force)

    if size == "1m":
        for key, name in (("copc", "1m.copc.laz"), ("ortho", "1m-ortho-3857.tif")):
            dst = (IN / "copc" / name) if key == "copc" else (IN / name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if force or not dst.exists():
                shutil.copyfile(src[key], dst)


def clip_rings(ext: dict) -> list[list[tuple[float, float]]]:
    """Same shape as the r9 reference clip: a rectangle with a hole plus a second rectangle."""
    w = ext["maxx"] - ext["minx"]
    h = ext["maxy"] - ext["miny"]

    def p(fx, fy):
        return (ext["minx"] + w * fx, ext["miny"] + h * fy)

    return [
        [p(.10, .10), p(.60, .10), p(.60, .90), p(.10, .90), p(.10, .10)],
        [p(.25, .30), p(.40, .30), p(.40, .60), p(.25, .60), p(.25, .30)],
        [p(.70, .20), p(.90, .20), p(.90, .80), p(.70, .80), p(.70, .20)],
    ]


def clip_wkt(ext: dict) -> str:
    rings = clip_rings(ext)
    r = [",".join(f"{x:.3f} {y:.3f}" for x, y in ring) for ring in rings]
    return f"MULTIPOLYGON((({r[0]}),({r[1]})),(({r[2]})))"


def write_clip_polygon(size: str, ext: dict, force: bool) -> None:
    wkt = clip_wkt(ext)
    (IN / f"{size}-clip.wkt").write_text(wkt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(28992)
    for driver, name in (("GPKG", f"{size}-clip.gpkg"), ("ESRI Shapefile", f"{size}-clip.shp")):
        path = IN / name
        if path.exists() and not force:
            continue
        drv = ogr.GetDriverByName(driver)
        if path.exists():
            drv.DeleteDataSource(str(path))
        ds = drv.CreateDataSource(str(path))
        layer = ds.CreateLayer("clip", srs, ogr.wkbMultiPolygon)
        layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("id", 1)
        feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
        layer.CreateFeature(feat)
        ds = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pdal", default=TOOLS["pdal_pinned"])
    a = ap.parse_args()
    validate_existing_manifest(a.force)
    for size in a.sizes:
        stage_size(size, a.pdal, a.force)
    write_input_manifest()
    print(json.dumps({s: json.load(open(IN / f"{s}-extent.json")) for s in a.sizes
                      if (IN / f"{s}-extent.json").exists()}, indent=1))


if __name__ == "__main__":
    main()
