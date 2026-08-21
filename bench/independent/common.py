"""Shared paths, tool locations and helpers for the independent benchmark."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = pathlib.Path(os.environ.get("IB_WORK", ROOT / "build" / "independent-bench"))
IN = WORK / "in"
OUT = WORK / "out"
RESULTS = WORK / "results"
RENDERS = WORK / "renders"

# Tool locations (override with environment variables on another machine).
TOOLS = {
    "pdg": os.environ.get("IB_PDG", str(ROOT / "build/pdg-cuda-release/bin/pdg")),
    "pdal_sys": os.environ.get("IB_PDAL_SYS", "/usr/bin/pdal"),
    "pdal_pinned": os.environ.get("IB_PDAL_PINNED",
                                  str(ROOT / "build/pdal-upstream-tests/bin/pdal")),
    "lastools": os.environ.get("IB_LASTOOLS", "/home/zy/LASTools/lastools/bin"),
    "wrench": os.environ.get("IB_WRENCH", "/usr/lib/qgis/pdal_wrench"),
    "qgis_process": os.environ.get("IB_QGIS_PROCESS", "/usr/bin/qgis_process"),
}

# Source data (local corpus; never modified).
SOURCES = {
    "1m": {"laz": ROOT / "build/bench-data/reference/ref-1m.laz",
           "copc": ROOT / "build/bench-data/reference/ref-1m.copc.laz",
           "ortho": ROOT / "build/bench-data/reference/ref-orthophoto-rgb-3857.tif",
           "label": "1 million points"},
    "4m": {"las": ROOT / "build/bench-data/download-3101-prefix-4m.las",
           "label": "4 million points"},
    "16m": {"las": ROOT / "build/bench-data/download-3101-prefix-16m.las",
            "label": "16 million points"},
    "47m": {"laz": ROOT / "build/bench-data/big/25GN1_01.LAZ",
            "label": "47 million points"},
}
SIZES = ["1m", "4m", "16m", "47m"]

# On another machine, IB_SOURCE_DIR holds <size>.laz for every size plus
# 1m.copc.laz and 1m-ortho-3857.tif.
if os.environ.get("IB_SOURCE_DIR"):
    _src = pathlib.Path(os.environ["IB_SOURCE_DIR"])
    for _size in SIZES:
        SOURCES[_size] = {"laz": _src / f"{_size}.laz", "label": SOURCES[_size]["label"]}
    SOURCES["1m"]["copc"] = _src / "1m.copc.laz"
    SOURCES["1m"]["ortho"] = _src / "1m-ortho-3857.tif"
SRS = "EPSG:28992"


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PDG_", "PDAL_TEST_"))}
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def run(cmd, **kw):
    kw.setdefault("check", True)
    kw.setdefault("text", True)
    kw.setdefault("env", clean_env())
    return subprocess.run([str(c) for c in cmd], **kw)


def info_summary(pdal: str, path: pathlib.Path) -> dict:
    out = run([pdal, "info", "--summary", path], capture_output=True).stdout
    return json.loads(out)["summary"]


def extent_of(pdal: str, path: pathlib.Path) -> dict:
    s = info_summary(pdal, path)
    b = s["bounds"]
    return {"minx": b["minx"], "maxx": b["maxx"], "miny": b["miny"],
            "maxy": b["maxy"], "minz": b["minz"], "maxz": b["maxz"],
            "points": s["num_points"]}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
