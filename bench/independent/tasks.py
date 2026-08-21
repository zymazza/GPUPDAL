"""Task catalogue: the same job expressed for every tool that can do it.

Every builder receives a Ctx (input paths, extent, output directory) and returns
a Job: a list of commands run one after another and timed as a whole, plus the
output files to inspect. Returning None means "this tool cannot do this job".

Tools:
  pdg_gpu       this repository's `pdg` (automatic GPU use)
  pdg_cpu       the same binary with no GPU visible (CUDA_VISIBLE_DEVICES="")
  pdg_gpu_all   the same binary with the GPU forced on for every stage that has one
  pdal_sys      PDAL 2.10.1 from the operating system's package
  pdal_pinned   PDAL 2.10.0 built from source (the fork's exactness reference)
  lastools      LAStools (rapidlasso) native Linux 64-bit tools, unlicensed
  wrench        pdal_wrench, the engine QGIS uses for point-cloud processing
  qgis          QGIS 4.2 `qgis_process run pdal:*` (the whole QGIS experience)
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib

from common import TOOLS


@dataclasses.dataclass
class Ctx:
    size: str
    inputs: dict          # name -> path
    extent: dict
    outdir: pathlib.Path  # per tool/size/task
    tool: str
    threads: int = 24


@dataclasses.dataclass
class Job:
    commands: list        # list of argv lists
    outputs: list         # list of output paths (files or directories)
    env: dict = dataclasses.field(default_factory=dict)
    note: str = ""
    cwd: pathlib.Path | None = None
    point_outputs: list = dataclasses.field(default_factory=list)  # LAS/LAZ outputs to count


PDAL_LIKE = ("pdg_gpu", "pdg_cpu", "pdg_gpu_all", "pdal_sys", "pdal_pinned")


def pdal_binary(tool: str) -> str:
    if tool.startswith("pdg"):
        return TOOLS["pdg"]
    if tool == "pdal_sys":
        return TOOLS["pdal_sys"]
    return TOOLS["pdal_pinned"]


def pdal_env(tool: str) -> dict:
    if tool == "pdg_cpu":
        return {"CUDA_VISIBLE_DEVICES": ""}
    if tool == "pdg_gpu_all":
        return {"PDG_EXPERIMENTAL_CUDA_HYBRID": "1"}
    return {}


def pipeline_job(ctx: Ctx, stages: list, outputs: list, point_outputs=None,
                 note: str = "") -> Job:
    """Write a PDAL pipeline JSON into ctx.outdir and run it with the tool's binary."""
    path = ctx.outdir / "pipeline.json"
    path.write_text(json.dumps({"pipeline": stages}, indent=1))
    return Job(commands=[[pdal_binary(ctx.tool), "pipeline", str(path)]],
               outputs=outputs, env=pdal_env(ctx.tool), note=note,
               point_outputs=point_outputs if point_outputs is not None else
               [o for o in outputs if str(o).endswith((".las", ".laz"))])


def lt(name: str) -> str:
    return str(pathlib.Path(TOOLS["lastools"]) / f"{name}64")


def wrench(*args) -> list:
    return [TOOLS["wrench"], *[str(a) for a in args]]


def qgis(alg: str, **params) -> list:
    cmd = [TOOLS["qgis_process"], "run", f"pdal:{alg}"]
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            for item in v:
                cmd.append(f"--{k}={item}")
        else:
            cmd.append(f"--{k}={v}")
    return cmd


def bounds_str(e: dict, fx0=0.0, fx1=1.0, fy0=0.0, fy1=1.0) -> str:
    w = e["maxx"] - e["minx"]
    h = e["maxy"] - e["miny"]
    return (f"([{e['minx'] + w * fx0!r},{e['minx'] + w * fx1!r}],"
            f"[{e['miny'] + h * fy0!r},{e['miny'] + h * fy1!r}])")


# --------------------------------------------------------------------------- tasks

TASKS = []


def task(id, name, plain, family, sizes=("1m", "4m", "16m", "47m"), tools=None):
    def deco(fn):
        TASKS.append({"id": id, "name": name, "plain": plain, "family": family,
                      "sizes": sizes, "tools": tools, "build": fn})
        return fn
    return deco


@task("compress", "Compress LAS to LAZ", "Take an uncompressed LAS file and write it as a compressed LAZ file.", "Files and formats")
def t_compress(ctx: Ctx):
    src, out = ctx.inputs["las"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("laszip"), "-i", str(src), "-o", str(out)]], [out], point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("translate", f"--input={src}", f"--output={out}", f"--threads={ctx.threads}")], [out], point_outputs=[out])
    if ctx.tool == "qgis":
        return Job([qgis("convertformat", INPUT=src, OUTPUT=out, VPC_OUTPUT_FORMAT=0)], [out], point_outputs=[out])


@task("decompress", "Decompress LAZ to LAS", "Take a compressed LAZ file and write it as an uncompressed LAS file.", "Files and formats")
def t_decompress(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.las"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "writers.las", "filename": str(out)}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("laszip"), "-i", str(src), "-o", str(out)]], [out], point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("translate", f"--input={src}", f"--output={out}", f"--threads={ctx.threads}")], [out], point_outputs=[out])
    if ctx.tool == "qgis":
        return Job([qgis("convertformat", INPUT=src, OUTPUT=out, VPC_OUTPUT_FORMAT=0)], [out], point_outputs=[out])


@task("copc", "Make a COPC file", "Turn a LAZ file into a cloud-optimized COPC file (a LAZ file with a built-in spatial index).", "Files and formats")
def t_copc(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.copc.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "writers.copc", "filename": str(out)}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("lascopcindex"), "-i", str(src), "-o", str(out)]], [out], point_outputs=[out])
    if ctx.tool == "wrench":
        untwine = os.environ.get("IB_UNTWINE", "/usr/lib/qgis/untwine")
        if not pathlib.Path(untwine).exists():
            return None
        return Job([[untwine, "-i", str(src), "-o", str(out), "--single_file"]], [out],
                   note="untwine, the indexer QGIS ships", point_outputs=[out])
    if ctx.tool == "qgis":
        d = ctx.outdir / "copc"
        d.mkdir(exist_ok=True)
        return Job([qgis("createcopc", LAYERS=src, OUTPUT=d)], [d / f"{src.stem}.copc.laz"],
                   point_outputs=[d / f"{src.stem}.copc.laz"])


@task("info", "Read the file summary", "Read the file and print its summary: point count, extent, and other header facts.", "Files and formats")
def t_info(ctx: Ctx):
    src = ctx.inputs["laz"]
    out = ctx.outdir / "info.txt"
    if ctx.tool in PDAL_LIKE:
        return Job([[pdal_binary(ctx.tool), "info", "--summary", str(src)]], [], env=pdal_env(ctx.tool))
    if ctx.tool == "lastools":
        return Job([[lt("lasinfo"), "-i", str(src), "-o", str(out)]], [out])
    if ctx.tool == "wrench":
        return Job([wrench("info", f"--input={src}")], [])
    if ctx.tool == "qgis":
        return Job([qgis("info", INPUT=src, OUTPUT=ctx.outdir / "info.html")], [ctx.outdir / "info.html"])


@task("reproject", "Reproject to Web Mercator", "Change the map coordinates of every point from the Dutch national grid (EPSG:28992) to Web Mercator (EPSG:3857) and write a LAZ file.", "Coordinates and clipping")
def t_reproject(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.reprojection", "in_srs": "EPSG:28992", "out_srs": "EPSG:3857"},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("las2las"), "-i", str(src), "-proj_epsg", "28992", "3857", "-o", str(out)]], [out], point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("translate", f"--input={src}", f"--output={out}", "--transform-crs=EPSG:3857", f"--threads={ctx.threads}")], [out], point_outputs=[out])
    if ctx.tool == "qgis":
        return Job([qgis("reproject", INPUT=src, CRS="EPSG:3857", OUTPUT=out, VPC_OUTPUT_FORMAT=0)], [out], point_outputs=[out])


@task("crop_reproject", "Crop to a rectangle, then reproject", "Keep only the points inside a rectangle covering the middle of the area, reproject them to Web Mercator, and write a LAZ file (the most common everyday PDAL job).", "Coordinates and clipping")
def t_crop_reproject(ctx: Ctx):
    src, out, e = ctx.inputs["laz"], ctx.outdir / "out.laz", ctx.extent
    b = bounds_str(e, .25, .75, .25, .75)
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.crop", "bounds": b},
                                  {"type": "filters.reprojection", "in_srs": "EPSG:28992", "out_srs": "EPSG:3857"},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    w, h = e["maxx"] - e["minx"], e["maxy"] - e["miny"]
    x0, x1, y0, y1 = e["minx"] + w * .25, e["minx"] + w * .75, e["miny"] + h * .25, e["miny"] + h * .75
    if ctx.tool == "lastools":
        return Job([[lt("las2las"), "-i", str(src), "-keep_xy", repr(x0), repr(y0), repr(x1), repr(y1),
                     "-proj_epsg", "28992", "3857", "-o", str(out)]], [out], point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("translate", f"--input={src}", f"--output={out}", f"--bounds={b}",
                           "--transform-crs=EPSG:3857", f"--threads={ctx.threads}")], [out], point_outputs=[out])
    return None


@task("clip", "Clip with a polygon", "Keep only the points inside a polygon shape (a rectangle with a hole in it, plus a second rectangle) and write a LAZ file.", "Coordinates and clipping")
def t_clip(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        wkt = ctx.inputs["clip_wkt"].read_text()
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.crop", "polygon": wkt},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("lasclip"), "-i", str(src), "-poly", str(ctx.inputs["clip_shp"]), "-o", str(out)]], [out], point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("clip", f"--input={src}", f"--output={out}", f"--polygon={ctx.inputs['clip_gpkg']}", f"--threads={ctx.threads}")], [out], point_outputs=[out])
    if ctx.tool == "qgis":
        return Job([qgis("clip", INPUT=src, OVERLAY=ctx.inputs["clip_gpkg"], OUTPUT=out, VPC_OUTPUT_FORMAT=0)], [out], point_outputs=[out])


@task("merge", "Merge two files", "Read two LAZ files (the west half and the east half of the area) and write one merged LAZ file.", "Files and formats")
def t_merge(ctx: Ctx):
    a, b, out = ctx.inputs["west"], ctx.inputs["east"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(a), "tag": "left"},
                                  {"type": "readers.las", "filename": str(b), "tag": "right"},
                                  {"type": "filters.merge", "inputs": ["left", "right"]},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("lasmerge"), "-i", str(a), str(b), "-o", str(out)]], [out], point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("merge", f"--output={out}", f"--threads={ctx.threads}", str(a), str(b))], [out], point_outputs=[out])
    if ctx.tool == "qgis":
        return Job([qgis("merge", LAYERS=[a, b], OUTPUT=out, VPC_OUTPUT_FORMAT=0)], [out], point_outputs=[out])


@task("tile", "Split into 256 m tiles", "Cut the point cloud into square 256 m tiles and write one file per tile.", "Files and formats")
def t_tile(ctx: Ctx):
    src, e = ctx.inputs["laz"], ctx.extent
    d = ctx.outdir / "tiles"
    d.mkdir(exist_ok=True)
    ox, oy = math.floor(e["minx"] / 256.0) * 256.0, math.floor(e["miny"] / 256.0) * 256.0
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.splitter", "length": 256.0, "origin_x": ox, "origin_y": oy, "buffer": 0.0},
                                  {"type": "writers.las", "filename": str(d / "tile-#.laz"), "compression": "true"}], [d], point_outputs=[d])
    if ctx.tool == "lastools":
        return Job([[lt("lastile"), "-i", str(src), "-tile_size", "256", "-olaz", "-odir", str(d), "-o", "tile.laz"]], [d], point_outputs=[d])
    if ctx.tool == "wrench":
        lst = ctx.outdir / "inputs.txt"
        lst.write_text(str(src) + "\n")
        return Job([wrench("tile", f"--input-file-list={lst}", f"--output={d}", "--length=256", f"--threads={ctx.threads}")], [d],
                   note="pdal_wrench writes the tiles as uncompressed LAS", point_outputs=[d])
    if ctx.tool == "qgis":
        return Job([qgis("tile", LAYERS=src, LENGTH=256, OUTPUT=d, VPC_OUTPUT_FORMAT=0)], [d],
                   note="QGIS writes the tiles as uncompressed LAS", point_outputs=[d])


@task("thin", "Thin to about one point per metre", "Reduce the point cloud so that no two kept points are closer than 1 m (Poisson sampling), and write a LAZ file.", "Reducing points")
def t_thin(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.sample", "radius": 1.0},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("lasthin"), "-i", str(src), "-step", "1", "-o", str(out)]], [out],
                   note="lasthin keeps one point per 1 m grid cell (a different rule with a similar result)", point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("thin", f"--input={src}", f"--output={out}", "--mode=sample", "--step-sample=1", f"--threads={ctx.threads}")], [out], point_outputs=[out])
    if ctx.tool == "qgis":
        return Job([qgis("thinbyradius", INPUT=src, SAMPLING_RADIUS=1, OUTPUT=out, VPC_OUTPUT_FORMAT=0)], [out], point_outputs=[out])


@task("decimate_grid", "Keep one point per 2.5 m cell", "Reduce the point cloud to one original point per 2.5 m cell (the point nearest the cell's centre) and write a LAZ file.", "Reducing points")
def t_decimate_grid(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.voxelcentroidnearestneighbor", "cell": 2.5},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("lasthin"), "-i", str(src), "-step", "2.5", "-central", "-o", str(out)]], [out],
                   note="lasthin -central: 2-D cells (PDAL/pdg use 3-D voxels)", point_outputs=[out])
    return None


@task("ground", "Find the ground", "Decide which points are bare ground (ground classification, SMRF method) and write a LAZ file with the ground points labelled.", "Ground and heights")
def t_ground(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.smrf"},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("lasground_new"), "-i", str(src), "-o", str(out)]], [out],
                   note="lasground_new uses its own algorithm, not SMRF", point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("classify_ground", f"--input={src}", f"--output={out}", "--cell-size=1", "--scalar=1.25",
                           "--slope=0.15", "--threshold=0.5", "--window-size=18", f"--threads={ctx.threads}")], [out], point_outputs=[out])
    if ctx.tool == "qgis":
        return Job([qgis("classifyground", INPUT=src, OUTPUT=out, VPC_OUTPUT_FORMAT=0)], [out], point_outputs=[out])


@task("hag", "Height above ground", "Starting from a file whose ground points are already labelled, work out how high every point is above the ground (nearest ground point) and store that height with each point.", "Ground and heights")
def t_hag(ctx: Ctx):
    src, out = ctx.inputs["ground"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.hag_nn"},
                                  {"type": "writers.las", "filename": str(out), "compression": "true",
                                   "extra_dims": "HeightAboveGround=float32"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("lasheight"), "-i", str(src), "-store_as_extra_bytes", "-o", str(out)]], [out],
                   note="lasheight uses a TIN of the ground points", point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("height_above_ground", f"--input={src}", f"--output={out}", "--algorithm=nn",
                           "--replace-z=false", "--nn-count=1", "--nn-max-distance=0", f"--threads={ctx.threads}")], [out], point_outputs=[out])
    if ctx.tool == "qgis":
        return Job([qgis("heightabovegroundbynearestneighbor", INPUT=src, REPLACE_Z="false", COUNT=1, MAX_DISTANCE=0,
                         OUTPUT=out, VPC_OUTPUT_FORMAT=0)], [out], point_outputs=[out])


@task("ground_hag", "Find the ground, then heights (one job)", "Do the two steps above as one job: find the ground, then give every point its height above ground, and write one LAZ file. Tools without a pipeline run two commands in a row.", "Ground and heights")
def t_ground_hag(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.laz"
    mid = ctx.outdir / "ground.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.smrf"},
                                  {"type": "filters.hag_nn"},
                                  {"type": "writers.las", "filename": str(out), "compression": "true",
                                   "extra_dims": "HeightAboveGround=float32"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("lasground_new"), "-i", str(src), "-o", str(mid)],
                    [lt("lasheight"), "-i", str(mid), "-store_as_extra_bytes", "-o", str(out)]], [out, mid], point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("classify_ground", f"--input={src}", f"--output={mid}", f"--threads={ctx.threads}"),
                    wrench("height_above_ground", f"--input={mid}", f"--output={out}", "--algorithm=nn",
                           "--replace-z=false", "--nn-count=1", "--nn-max-distance=0", f"--threads={ctx.threads}")], [out, mid], point_outputs=[out])
    if ctx.tool == "qgis":
        return Job([qgis("classifyground", INPUT=src, OUTPUT=mid, VPC_OUTPUT_FORMAT=0),
                    qgis("heightabovegroundbynearestneighbor", INPUT=mid, REPLACE_Z="false", COUNT=1, MAX_DISTANCE=0,
                         OUTPUT=out, VPC_OUTPUT_FORMAT=0)], [out, mid], point_outputs=[out])


@task("dtm", "Bare-earth elevation map (DTM)", "From a file whose ground points are labelled, make a 1 m raster (GeoTIFF) of the ground height.", "Rasters (maps)")
def t_dtm(ctx: Ctx):
    src, out = ctx.inputs["ground"], ctx.outdir / "dtm.tif"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.range", "limits": "Classification[2:2]"},
                                  {"type": "writers.gdal", "filename": str(out), "resolution": 1.0, "output_type": "idw"}], [out],
                            note="inverse-distance weighting of the ground points in each cell")
    if ctx.tool == "lastools":
        return Job([[lt("las2dem"), "-i", str(src), "-keep_class", "2", "-step", "1", "-elevation", "-otif", "-o", str(out)]], [out],
                   note="las2dem builds a triangle mesh (TIN) and rasterizes it")
    if ctx.tool == "wrench":
        return Job([wrench("to_raster", f"--input={src}", f"--output={out}", "--attribute=Z", "--resolution=1",
                           "--tile-size=1000", "--filter=Classification == 2", f"--threads={ctx.threads}")], [out],
                   note="average ground height in each cell")
    if ctx.tool == "qgis":
        return Job([qgis("exportraster", INPUT=src, OUTPUT=out, ATTRIBUTE="Z", RESOLUTION=1, TILE_SIZE=1000,
                         FILTER_EXPRESSION="Classification = 2")], [out], note="average ground height in each cell")


@task("dsm", "Surface elevation map (DSM)", "Make a 1 m raster (GeoTIFF) of the surface height using the first return of every laser pulse (tree tops, roofs, and ground where nothing is above it).", "Rasters (maps)")
def t_dsm(ctx: Ctx):
    src, out, e = ctx.inputs["laz"], ctx.outdir / "dsm.tif", ctx.extent
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src), "override_srs": "EPSG:28992"},
                                  {"type": "filters.returns", "groups": "first,only"},
                                  {"type": "writers.gdal", "filename": str(out), "resolution": 1.0, "output_type": "max",
                                   "dimension": "Z", "binmode": True, "data_type": "float64", "nodata": -9999.0,
                                   "bounds": bounds_str(e)}], [out], note="highest first-return point per cell")
    if ctx.tool == "lastools":
        return Job([[lt("lasgrid"), "-i", str(src), "-first_only", "-highest", "-step", "1", "-otif", "-o", str(out)]], [out],
                   note="highest first-return point per cell")
    if ctx.tool == "wrench":
        return Job([wrench("to_raster", f"--input={src}", f"--output={out}", "--attribute=Z", "--resolution=1",
                           "--tile-size=1000", "--filter=ReturnNumber == 1", f"--threads={ctx.threads}")], [out],
                   note="pdal_wrench can only average the first returns in each cell, not pick the highest")
    if ctx.tool == "qgis":
        return Job([qgis("exportraster", INPUT=src, OUTPUT=out, ATTRIBUTE="Z", RESOLUTION=1, TILE_SIZE=1000,
                         FILTER_EXPRESSION="ReturnNumber = 1")], [out],
                   note="QGIS can only average the first returns in each cell, not pick the highest")


@task("density", "Point density map", "Make a 1 m raster where each cell holds the number of points that fell inside it.", "Rasters (maps)")
def t_density(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "density.tif"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "writers.gdal", "filename": str(out), "resolution": 1.0, "output_type": "count",
                                   "binmode": True, "data_type": "int32"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("lasgrid"), "-i", str(src), "-counter", "-step", "1", "-otif", "-o", str(out)]], [out])
    if ctx.tool == "wrench":
        return Job([wrench("density", f"--input={src}", f"--output={out}", "--resolution=1", "--tile-size=1000", f"--threads={ctx.threads}")], [out])
    if ctx.tool == "qgis":
        return Job([qgis("density", INPUT=src, OUTPUT=out, RESOLUTION=1, TILE_SIZE=1000)], [out])


@task("boundary", "Outline of the covered area", "Work out a polygon that outlines where the points are and write it as a vector file.", "Rasters (maps)")
def t_boundary(ctx: Ctx):
    src = ctx.inputs["laz"]
    if ctx.tool in PDAL_LIKE:
        return Job([[pdal_binary(ctx.tool), "info", "--boundary", str(src)]], [], env=pdal_env(ctx.tool),
                   note="hexagon-bin outline printed as text")
    if ctx.tool == "lastools":
        out = ctx.outdir / "boundary.shp"
        return Job([[lt("lasboundary"), "-i", str(src), "-o", str(out)]], [out], note="concave hull")
    if ctx.tool == "wrench":
        out = ctx.outdir / "boundary.gpkg"
        return Job([wrench("boundary", f"--input={src}", f"--output={out}", f"--threads={ctx.threads}")], [out], note="hexagon-bin outline")
    if ctx.tool == "qgis":
        out = ctx.outdir / "boundary.gpkg"
        return Job([qgis("boundary", INPUT=src, OUTPUT=out)], [out], note="hexagon-bin outline")


@task("outlier", "Find noise points", "Find isolated stray points (statistical outliers, 8 neighbours, 2 standard deviations) and label them as noise; write a LAZ file.", "Cleaning")
def t_outlier(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.outlier", "method": "statistical", "mean_k": 8, "multiplier": 2.0},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("lasnoise"), "-i", str(src), "-o", str(out)]], [out],
                   note="lasnoise looks for isolated points in a 3-D grid (a different rule)", point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("filter_noise", f"--input={src}", f"--output={out}", "--algorithm=statistical",
                           "--remove-noise-points=false", "--statistical-mean-k=8", "--statistical-multiplier=2",
                           f"--threads={ctx.threads}")], [out], point_outputs=[out])
    if ctx.tool == "qgis":
        return Job([qgis("filternoisestatistical", INPUT=src, OUTPUT=out, MEAN_K=8, MULTIPLIER=2,
                         REMOVE_NOISE_POINTS="false", VPC_OUTPUT_FORMAT=0)], [out], point_outputs=[out])


@task("denoise_thin", "Remove noise, then thin (one job)", "Find and drop the noise points, then thin to about one point per metre, and write a LAZ file. Tools without a pipeline run two commands in a row.", "Cleaning")
def t_denoise_thin(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.laz"
    mid = ctx.outdir / "clean.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.outlier", "method": "statistical", "mean_k": 8, "multiplier": 2.0},
                                  {"type": "filters.range", "limits": "Classification![7:7]"},
                                  {"type": "filters.sample", "radius": 1.0},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    if ctx.tool == "lastools":
        return Job([[lt("lasnoise"), "-i", str(src), "-remove_noise", "-o", str(mid)],
                    [lt("lasthin"), "-i", str(mid), "-step", "1", "-o", str(out)]], [out, mid], point_outputs=[out])
    if ctx.tool == "wrench":
        return Job([wrench("filter_noise", f"--input={src}", f"--output={mid}", "--algorithm=statistical",
                           "--remove-noise-points=true", "--statistical-mean-k=8", "--statistical-multiplier=2",
                           f"--threads={ctx.threads}"),
                    wrench("thin", f"--input={mid}", f"--output={out}", "--mode=sample", "--step-sample=1",
                           f"--threads={ctx.threads}")], [out, mid], point_outputs=[out])
    if ctx.tool == "qgis":
        return Job([qgis("filternoisestatistical", INPUT=src, OUTPUT=mid, MEAN_K=8, MULTIPLIER=2,
                         REMOVE_NOISE_POINTS="true", VPC_OUTPUT_FORMAT=0),
                    qgis("thinbyradius", INPUT=mid, SAMPLING_RADIUS=1, OUTPUT=out, VPC_OUTPUT_FORMAT=0)], [out, mid], point_outputs=[out])


@task("classify_refine", "Ground, noise, and neighbour vote (one job)", "Find the ground, label noise points, then let each unclassified point take the majority label of its 7 nearest neighbours; write a LAZ file.", "Cleaning",
      tools=("pdg_gpu", "pdg_cpu", "pdg_gpu_all", "pdal_sys", "pdal_pinned"))
def t_classify_refine(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.smrf"},
                                  {"type": "filters.outlier", "method": "statistical", "mean_k": 8, "multiplier": 2.0},
                                  {"type": "filters.neighborclassifier", "k": 7, "domain": "Classification[1:1]"},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    return None


@task("features", "Surface normals and shape features", "For every point, look at its 8 nearest neighbours and compute the surface normal and shape features (how line-like, flat, or scattered the neighbourhood is); write everything to a LAZ file.", "Neighbourhood analysis",
      tools=("pdg_gpu", "pdg_cpu", "pdg_gpu_all", "pdal_sys", "pdal_pinned"))
def t_features(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.normal", "knn": 8},
                                  {"type": "filters.covariancefeatures", "knn": 8, "feature_set": "Dimensionality"},
                                  {"type": "writers.las", "filename": str(out), "compression": "true", "extra_dims": "all"}], [out])
    return None


@task("colorize", "Colour the points from an aerial image", "Give every point a colour by looking it up in an aerial image (in a different map projection), then write a LAZ file.", "Neighbourhood analysis",
      sizes=("1m",), tools=("pdg_gpu", "pdg_cpu", "pdg_gpu_all", "pdal_sys", "pdal_pinned"))
def t_colorize(ctx: Ctx):
    src, out = ctx.inputs["laz"], ctx.outdir / "out.laz"
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.las", "filename": str(src)},
                                  {"type": "filters.reprojection", "in_srs": "EPSG:28992", "out_srs": "EPSG:3857"},
                                  {"type": "filters.colorization", "raster": str(ctx.inputs["ortho"]),
                                   "dimensions": "Red:1:1.0, Green:2:1.0, Blue:3:1.0"},
                                  {"type": "filters.reprojection", "in_srs": "EPSG:3857", "out_srs": "EPSG:28992"},
                                  {"type": "writers.las", "filename": str(out), "compression": "true"}], [out])
    return None


@task("copc_query", "Read a window from a COPC file", "Open a COPC file, read only the points inside the middle rectangle at 1 m resolution, compute statistics, and write a LAS file.", "Files and formats",
      sizes=("1m",), tools=("pdg_gpu", "pdg_cpu", "pdg_gpu_all", "pdal_sys", "pdal_pinned"))
def t_copc_query(ctx: Ctx):
    src, out, e = ctx.inputs["copc"], ctx.outdir / "out.las", ctx.extent
    if ctx.tool in PDAL_LIKE:
        return pipeline_job(ctx, [{"type": "readers.copc", "filename": str(src), "bounds": bounds_str(e, .25, .75, .25, .75),
                                   "resolution": 1.0, "requests": 1},
                                  {"type": "filters.stats"},
                                  {"type": "writers.las", "filename": str(out)}], [out])
    return None


TASK_BY_ID = {t["id"]: t for t in TASKS}
