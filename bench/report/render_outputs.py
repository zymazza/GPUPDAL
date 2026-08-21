#!/usr/bin/env python3
"""Render benchmark outputs to PNG for the B0275 report.

Point clouds are decimated through the pinned oracle (`filters.decimation`)
and drawn top-down with matplotlib, colored by a dimension or RGB; rasters
are hillshaded/scaled through GDAL. Nothing here changes an artifact.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def cloud_to_array(pdal: pathlib.Path, path: pathlib.Path, dims: list[str],
                   step: int, extra_stage: dict | None = None) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="pdg-render-") as temp:
        out = pathlib.Path(temp) / "points.txt"
        stages: list = [{"type": "readers.las", "filename": str(path)}]
        if extra_stage:
            stages.append(extra_stage)
        if step > 1:
            stages.append({"type": "filters.decimation", "step": step})
        stages.append({"type": "writers.text", "filename": str(out),
                       "order": ",".join(dims), "keep_unspecified": False,
                       "write_header": True, "precision": 6})
        pipeline = pathlib.Path(temp) / "pipeline.json"
        pipeline.write_text(json.dumps({"pipeline": stages}))
        subprocess.run([str(pdal), "pipeline", str(pipeline)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return np.loadtxt(out, delimiter=",", skiprows=1, ndmin=2)


def render_cloud(pdal, path, png, mode, step, title, window=None):
    if mode == "rgb":
        data = cloud_to_array(pdal, path, ["X", "Y", "Red", "Green", "Blue"], step)
        colors = np.clip(data[:, 2:5] / 65535.0, 0, 1)
        if colors.max() <= 1.0 / 255.0:
            colors = np.clip(data[:, 2:5] / 255.0, 0, 1)
        c = colors
        cmap = None
        label = "RGB from orthophoto"
    elif mode == "z":
        data = cloud_to_array(pdal, path, ["X", "Y", "Z"], step)
        c = data[:, 2]; cmap = "terrain"; label = "Z (m)"
    elif mode == "hag":
        data = cloud_to_array(pdal, path, ["X", "Y", "HeightAboveGround"], step)
        c = np.clip(data[:, 2], 0, 30); cmap = "viridis"; label = "height above ground (m, clipped 0..30)"
    elif mode == "classification":
        data = cloud_to_array(pdal, path, ["X", "Y", "Classification"], step)
        c = data[:, 2]; cmap = "tab20"; label = "classification"
    elif mode == "planarity":
        data = cloud_to_array(pdal, path, ["X", "Y", "Planarity"], step)
        c = data[:, 2]; cmap = "magma"; label = "planarity (covariance feature)"
    elif mode == "normalz":
        data = cloud_to_array(pdal, path, ["X", "Y", "NormalZ"], step)
        c = np.abs(data[:, 2]); cmap = "cividis"; label = "|NormalZ|"
    elif mode == "density":
        data = cloud_to_array(pdal, path, ["X", "Y"], step)
        c = None; cmap = None; label = "points"
    else:
        raise SystemExit(f"unknown mode {mode}")
    if window:
        # Show an X window of `window` metres centred on the extent so the
        # long thin reference strip stays legible.
        centre = 0.5 * (data[:, 0].min() + data[:, 0].max())
        keep = np.abs(data[:, 0] - centre) <= window / 2.0
        data = data[keep]
        c = c if (c is None or np.ndim(c) == 0) else c[keep]
    fig, ax = plt.subplots(figsize=(11, 5), dpi=110)
    size = 0.4 if window else 0.2
    if c is None:
        ax.scatter(data[:, 0], data[:, 1], s=size, c="#1f77b4", linewidths=0)
    else:
        sc = ax.scatter(data[:, 0], data[:, 1], s=size, c=c, cmap=cmap, linewidths=0)
        if cmap:
            fig.colorbar(sc, ax=ax, label=label, fraction=0.03)
    ax.set_aspect("equal")
    ax.set_facecolor("#202020" if mode == "rgb" else "white")
    ax.set_title(f"{title}\n{path.name}: {data.shape[0]:,} of every {step} points shown", fontsize=10)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


def render_raster(path, png, title, hillshade=True):
    with tempfile.TemporaryDirectory(prefix="pdg-render-") as temp:
        src = str(path)
        if hillshade:
            shade = pathlib.Path(temp) / "shade.tif"
            subprocess.run(["gdaldem", "hillshade", "-q", "-z", "2", src, str(shade)],
                           check=True)
            src = str(shade)
            subprocess.run(["gdal_translate", "-q", "-of", "PNG", "-ot", "Byte", src, str(png)],
                           check=True)
        else:
            subprocess.run(["gdal_translate", "-q", "-of", "PNG", "-ot", "Byte", "-scale", src,
                            str(png)], check=True)
        # Title annotation via matplotlib re-plot for consistency.
        img = plt.imread(str(png))
        fig, ax = plt.subplots(figsize=(9, 6), dpi=110)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{title}\n{path.name}", fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(png)
        plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pdal", required=True, type=pathlib.Path)
    p.add_argument("--kind", choices=("cloud", "raster"), required=True)
    p.add_argument("--input", required=True, type=pathlib.Path)
    p.add_argument("--output", required=True, type=pathlib.Path)
    p.add_argument("--mode", default="z")
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--title", default="")
    p.add_argument("--no-hillshade", action="store_true")
    p.add_argument("--window", type=float, default=None,
                   help="X window in metres centred on the cloud")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.kind == "cloud":
        render_cloud(a.pdal, a.input, a.output, a.mode, a.step, a.title,
                     a.window)
    else:
        render_raster(a.input, a.output, a.title, hillshade=not a.no_hillshade)
    print(a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
