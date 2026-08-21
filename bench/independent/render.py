#!/usr/bin/env python3
"""Pictures of the benchmark inputs and outputs (PNG files under renders/).

Point clouds are read through pinned PDAL's text writer (a CSV of the wanted
dimensions), so no extra Python packages are needed. Rasters are read with GDAL.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, ListedColormap  # noqa: E402
from osgeo import gdal, ogr  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import IN, OUT, RENDERS, TOOLS, clean_env  # noqa: E402

gdal.UseExceptions()
PDAL = TOOLS["pdal_pinned"]
DPI = 130
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})
TERRAIN = plt.get_cmap("terrain")
plt.rcParams["axes.formatter.useoffset"] = False
plt.rcParams["axes.formatter.limits"] = (-7, 9)


def strip_segments(ext, n=3):
    """Split a wide extent (x0, x1, y0, y1) into n x-segments for stacked display."""
    x0, x1, y0, y1 = ext
    w = (x1 - x0) / n
    return [(x0 + i * w, x0 + (i + 1) * w, y0, y1) for i in range(n)]


def window_of(ext, length=400.0):
    """A representative x-window in the middle of the extent (for zoomed output pictures)."""
    x0, x1, y0, y1 = ext
    mid = (x0 + x1) / 2
    return (mid - length / 2, mid + length / 2)


def read_points(path: pathlib.Path, dims: list[str], step: int = 1) -> dict:
    """Return {dim: np.array} for the requested dimensions (every `step`-th point)."""
    stages = [{"type": "readers.las", "filename": str(path)}]
    if step > 1:
        stages.append({"type": "filters.decimation", "step": step})
    stages.append({"type": "writers.text", "filename": "STDOUT", "order": ",".join(dims),
                   "keep_unspecified": False, "precision": 3, "quote_header": False})
    r = subprocess.run([PDAL, "pipeline", "--stdin"], input=json.dumps({"pipeline": stages}),
                       capture_output=True, text=True, env=clean_env(), check=True)
    data = np.genfromtxt(io.StringIO(r.stdout), delimiter=",", names=True)
    return {d: np.asarray(data[d.replace(" ", "")]) for d in dims}


def grid_mean(x, y, v, cell: float, extent=None, reducer="mean"):
    """Rasterize a per-point value onto a grid (mean or max per cell)."""
    if extent is None:
        extent = (x.min(), x.max(), y.min(), y.max())
    x0, x1, y0, y1 = extent
    nx = max(1, int(np.ceil((x1 - x0) / cell)))
    ny = max(1, int(np.ceil((y1 - y0) / cell)))
    ix = np.clip(((x - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((y1 - y) / cell).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    if reducer == "max":
        img = np.full(nx * ny, np.nan)
        np.maximum.at(img, flat, np.where(np.isnan(img[flat]), v, v))
        img = np.full(nx * ny, -np.inf)
        np.maximum.at(img, flat, v)
        img[img == -np.inf] = np.nan
    else:
        s = np.bincount(flat, weights=v, minlength=nx * ny)
        c = np.bincount(flat, minlength=nx * ny)
        with np.errstate(invalid="ignore", divide="ignore"):
            img = s / c
        img[c == 0] = np.nan
    return img.reshape(ny, nx), (x0, x1, y0, y1)


def save(fig, name: str):
    RENDERS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RENDERS / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def render_input(size: str, step: int, cell: float, title: str):
    p = read_points(IN / f"{size}.laz", ["X", "Y", "Z"], step)
    img, ext = grid_mean(p["X"], p["Y"], p["Z"], cell)
    aspect = (ext[1] - ext[0]) / max(1.0, ext[3] - ext[2])
    if aspect > 6:
        segs = strip_segments(ext, 3)
        fig, axes = plt.subplots(len(segs), 1, figsize=(9, 1.0 + 1.9 * len(segs)))
        vmin, vmax = np.nanpercentile(img, 1), np.nanpercentile(img, 99)
        for ax, sg in zip(axes, segs):
            im = ax.imshow(img, extent=ext, cmap=TERRAIN, origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_xlim(sg[0], sg[1])
            ax.set_ylabel("y (m)")
        axes[0].set_title(title + " — shown as three pieces, west to east")
        axes[-1].set_xlabel("x (m)")
        fig.colorbar(im, ax=axes.tolist(), label="height (m)", shrink=0.9)
    else:
        fig, ax = plt.subplots(figsize=(9, 3.2 if size != "47m" else 6.5))
        im = ax.imshow(img, extent=ext, cmap=TERRAIN, origin="upper", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, label="height (m)", shrink=0.8)
    save(fig, f"input-{size}-height")


def render_input_3d(size: str = "1m"):
    p = read_points(IN / f"{size}.laz", ["X", "Y", "Z"], 5)
    x, y, z = p["X"], p["Y"], p["Z"]
    xs = x.min() + 700
    m = (x >= xs) & (x < xs + 200)
    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[m] - xs, y[m] - y.min(), z[m], c=z[m], cmap=TERRAIN, s=1.2, linewidths=0)
    ax.set_title("A 200 m long piece of the 1-million-point cloud, seen from the side (every 5th point)")
    ax.set_xlabel("metres east")
    ax.set_ylabel("metres north")
    ax.set_zlabel("height (m)")
    ax.view_init(elev=28, azim=-55)
    ax.set_box_aspect((2.6, 1, 0.7))
    save(fig, f"input-{size}-3d")


def zoom(ax, ext, length=400.0):
    w = window_of(ext, length)
    ax.set_xlim(w[0], w[1])
    ax.set_ylim(ext[2], ext[3])


def render_ground(size="1m", tools=("pdg_gpu", "lastools", "wrench")):
    names = {"pdg_gpu": "pdg (SMRF)", "pdal_pinned": "PDAL (SMRF)", "lastools": "LAStools lasground_new",
             "wrench": "QGIS / pdal_wrench (SMRF)"}
    fig, axes = plt.subplots(len(tools), 1, figsize=(9, 2.6 * len(tools)), sharex=True)
    for ax, tool in zip(np.atleast_1d(axes), tools):
        path = OUT / tool / size / "ground" / "out.laz"
        p = read_points(path, ["X", "Y", "Classification"], 1)
        g = (p["Classification"] == 2).astype(float)
        img, ext = grid_mean(p["X"], p["Y"], g, 1.0)
        ax.imshow(img, extent=ext, cmap="YlGn", origin="upper", vmin=0, vmax=1, interpolation="nearest")
        zoom(ax, ext)
        pct = 100.0 * g.mean()
        ax.set_title(f"{names.get(tool, tool)}: {pct:.1f}% of points labelled ground (dark green = ground); 400 m window")
        ax.set_ylabel("y (m)")
    axes[-1].set_xlabel("x (m)")
    save(fig, f"out-{size}-ground-compare")


def render_hag(size="1m", tool="pdg_gpu"):
    p = read_points(OUT / tool / size / "hag" / "out.laz", ["X", "Y", "HeightAboveGround"], 1)
    img, ext = grid_mean(p["X"], p["Y"], p["HeightAboveGround"], 1.0, reducer="max")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(img, extent=ext, cmap="viridis", origin="upper", vmin=0, vmax=25, interpolation="nearest")
    zoom(ax, ext)
    ax.set_title("Height above ground (highest point in each 1 m cell); 400 m window")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, label="metres above ground", shrink=0.8)
    save(fig, f"out-{size}-hag")


def render_raster(path: pathlib.Path, name: str, title: str, cmap="terrain", hillshade=True, label="height (m)",
                  vmin=None, vmax=None, figsize=(9, 3.2), window=True):
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(float)
    nd = band.GetNoDataValue()
    if nd is not None:
        arr[arr == nd] = np.nan
    gt = ds.GetGeoTransform()
    ext = (gt[0], gt[0] + gt[1] * ds.RasterXSize, gt[3] + gt[5] * ds.RasterYSize, gt[3])
    fig, ax = plt.subplots(figsize=figsize)
    if hillshade:
        ls = LightSource(azdeg=315, altdeg=45)
        filled = np.where(np.isnan(arr), np.nanmin(arr), arr)
        rgb = ls.shade(filled, cmap=plt.get_cmap(cmap), blend_mode="overlay", vert_exag=1.5,
                       vmin=vmin if vmin is not None else np.nanpercentile(arr, 1),
                       vmax=vmax if vmax is not None else np.nanpercentile(arr, 99))
        rgb[np.isnan(arr)] = [1, 1, 1, 1]
        ax.imshow(rgb, extent=ext, origin="upper", interpolation="nearest")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
            vmin if vmin is not None else np.nanpercentile(arr, 1), vmax if vmax is not None else np.nanpercentile(arr, 99)))
        fig.colorbar(sm, ax=ax, label=label, shrink=0.8)
    else:
        im = ax.imshow(arr, extent=ext, cmap=cmap, origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    if window and (ext[1] - ext[0]) / max(1.0, ext[3] - ext[2]) > 6:
        zoom(ax, ext)
        title += "; 400 m window"
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    save(fig, name)


def render_thin(size="1m"):
    src = read_points(IN / f"{size}.laz", ["X", "Y", "Z"], 1)
    x0 = src["X"].min() + 700
    y0 = src["Y"].min() + 10
    win = 40
    panels = [("All points (input)", src),
              ("pdg / PDAL: Poisson sampling, 1 m", read_points(OUT / "pdg_gpu" / size / "thin" / "out.laz", ["X", "Y", "Z"])),
              ("QGIS / pdal_wrench: Poisson sampling, 1 m", read_points(OUT / "wrench" / size / "thin" / "out.laz", ["X", "Y", "Z"])),
              ("LAStools lasthin: one point per 1 m cell", read_points(OUT / "lastools" / size / "thin" / "out.laz", ["X", "Y", "Z"])),
              ("pdg / PDAL: one point per 2.5 m cell", read_points(OUT / "pdg_gpu" / size / "decimate_grid" / "out.laz", ["X", "Y", "Z"]))]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.2))
    for ax, (title, p) in zip(axes, panels):
        m = (p["X"] >= x0) & (p["X"] < x0 + win) & (p["Y"] >= y0) & (p["Y"] < y0 + win)
        ax.scatter(p["X"][m], p["Y"][m], c=p["Z"][m], cmap=TERRAIN, s=3, linewidths=0)
        ax.set_title(f"{title}\n{m.sum():,} points in this 40 m square", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, f"out-{size}-thin-compare")


def render_clip(size="1m"):
    p = read_points(OUT / "pdg_gpu" / size / "clip" / "out.laz", ["X", "Y", "Z"], 1)
    src = read_points(IN / f"{size}.laz", ["X", "Y"], 4)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.scatter(src["X"], src["Y"], s=0.2, c="#d0d0d0", linewidths=0, label="all points")
    ax.scatter(p["X"], p["Y"], s=0.3, c=p["Z"], cmap=TERRAIN, linewidths=0)
    ds = ogr.Open(str(IN / f"{size}-clip.gpkg"))
    for feat in ds.GetLayer():
        geom = feat.GetGeometryRef()
        for gi in range(geom.GetGeometryCount()):
            poly = geom.GetGeometryRef(gi)
            for ri in range(poly.GetGeometryCount()):
                ring = poly.GetGeometryRef(ri)
                pts = ring.GetPoints()
                ax.plot([q[0] for q in pts], [q[1] for q in pts], color="crimson", lw=1.2)
    ax.set_title(f"Clip with a polygon: {len(p['X']):,} of 1,000,000 points kept (red outline = the polygon; grey = dropped)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    save(fig, f"out-{size}-clip")


def render_tiles(size="1m"):
    d = OUT / "pdg_gpu" / size / "tile" / "tiles"
    files = sorted(d.glob("*.laz"))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    cmap = plt.get_cmap("tab10")
    for i, f in enumerate(files):
        p = read_points(f, ["X", "Y"], 3)
        ax.scatter(p["X"], p["Y"], s=0.3, color=cmap(i % 10), linewidths=0)
        ax.text(p["X"].mean(), p["Y"].mean(), f.name, ha="center", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.8))
    ax.set_title(f"Split into 256 m tiles: {len(files)} tiles (one colour each)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    save(fig, f"out-{size}-tiles")


def render_features(size="1m"):
    p = read_points(OUT / "pdg_gpu" / size / "features" / "out.laz", ["X", "Y", "NormalZ", "Verticality", "Planarity"], 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    for ax, dim, cmap, title in ((axes[0], "Verticality", "magma", "Verticality (1 = a vertical surface such as a wall or trunk)"),
                                 (axes[1], "Planarity", "cividis", "Planarity (1 = a flat surface such as a roof or road)")):
        img, ext = grid_mean(p["X"], p["Y"], p[dim], 1.0)
        im = ax.imshow(img, extent=ext, cmap=cmap, origin="upper", vmin=0, vmax=1, interpolation="nearest")
        zoom(ax, ext)
        ax.set_title(title + "; 400 m window")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[-1].set_xlabel("x (m)")
    save(fig, f"out-{size}-features")


def render_colorize(size="1m"):
    p = read_points(OUT / "pdg_gpu" / size / "colorize" / "out.laz", ["X", "Y", "Red", "Green", "Blue"], 1)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    rgb = np.stack([p["Red"], p["Green"], p["Blue"]], axis=1)
    rgb = rgb / max(1.0, rgb.max())
    ax.scatter(p["X"], p["Y"], c=rgb, s=0.3, linewidths=0)
    ax.set_title("Colour the points from an aerial image (here a synthetic colour-gradient image over the middle half)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    save(fig, f"out-{size}-colorize")


def render_outlier(size="1m"):
    names = {"pdg_gpu": "pdg / PDAL: statistical outliers", "lastools": "LAStools lasnoise: isolated points"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    for ax, tool in zip(axes, ("pdg_gpu", "lastools")):
        p = read_points(OUT / tool / size / "outlier" / "out.laz", ["X", "Y", "Z", "Classification"], 1)
        noise = p["Classification"] == 7
        ax.scatter(p["X"][~noise][::3], p["Y"][~noise][::3], s=0.2, c="#c8c8c8", linewidths=0)
        ax.scatter(p["X"][noise], p["Y"][noise], s=6, c="red", linewidths=0)
        ax.set_title(f"{names[tool]}: {noise.sum():,} points labelled noise (red); 400 m window")
        ax.set_aspect("equal")
        zoom(ax, (p["X"].min(), p["X"].max(), p["Y"].min(), p["Y"].max()))
        ax.set_ylabel("y (m)")
    axes[-1].set_xlabel("x (m)")
    save(fig, f"out-{size}-outlier-compare")


def render_boundary(size="1m"):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    src = read_points(IN / f"{size}.laz", ["X", "Y"], 4)
    ax.scatter(src["X"], src["Y"], s=0.2, c="#d0d0d0", linewidths=0)
    for tool, color, label in (("wrench", "royalblue", "QGIS / pdal_wrench (hexagon bins)"), ("lastools", "crimson", "LAStools lasboundary (concave hull)")):
        f = OUT / tool / size / "boundary" / ("boundary.gpkg" if tool == "wrench" else "boundary.shp")
        if not f.exists():
            continue
        ds = ogr.Open(str(f))
        for feat in ds.GetLayer():
            geom = feat.GetGeometryRef()
            polys = [geom] if geom.GetGeometryName() == "POLYGON" else [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())]
            first = True
            for poly in polys:
                for ri in range(poly.GetGeometryCount()):
                    pts = poly.GetGeometryRef(ri).GetPoints()
                    ax.plot([q[0] for q in pts], [q[1] for q in pts], color=color, lw=1, label=label if first else None)
                    first = False
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Outline of the covered area")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    save(fig, f"out-{size}-boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    def want(n):
        return a.only is None or n in a.only

    if want("inputs"):
        render_input("1m", 1, 1.0, "The 1-million-point cloud: a 1.5 km × 77 m strip of a New York State lidar tile, coloured by height (1 m cells)")
        render_input("4m", 2, 1.0, "The 4-million-point cloud: 1.5 km × 280 m of the same tile, coloured by height")
        render_input("16m", 8, 2.0, "The 16-million-point cloud: 1.5 km × 1.5 km of the same tile, coloured by height (2 m cells)")
        render_input("47m", 20, 2.0, "The 47-million-point cloud: a full 1 km × 1.25 km AHN4 GeoTile (25GN1_01), coloured by height (2 m cells)")
        render_input_3d("1m")
    if want("ground"):
        render_ground()
    if want("hag"):
        render_hag()
    if want("rasters"):
        render_raster(OUT / "pdg_gpu/1m/dtm/dtm.tif", "out-1m-dtm", "Bare-earth elevation map (DTM), 1 m cells, hill-shaded (pdg / PDAL output)")
        render_raster(OUT / "pdg_gpu/1m/dsm/dsm.tif", "out-1m-dsm", "Surface elevation map (DSM), 1 m cells, hill-shaded (pdg / PDAL output)")
        render_raster(OUT / "pdg_gpu/1m/density/density.tif", "out-1m-density", "Point density: points per 1 m cell (pdg / PDAL output)",
                      cmap="magma", hillshade=False, label="points per cell", vmin=0, vmax=40)
        render_raster(OUT / "lastools/1m/dtm/dtm.tif", "out-1m-dtm-lastools", "The same DTM made by LAStools las2dem (triangle mesh)")
        render_raster(OUT / "wrench/1m/dtm/dtm.tif", "out-1m-dtm-wrench", "The same DTM made by QGIS / pdal_wrench (average height per cell)")
    if want("thin"):
        render_thin()
    if want("clip"):
        render_clip()
    if want("tiles"):
        render_tiles()
    if want("features"):
        render_features()
    if want("colorize"):
        render_colorize()
    if want("outlier"):
        render_outlier()
    if want("boundary"):
        render_boundary()


if __name__ == "__main__":
    main()
