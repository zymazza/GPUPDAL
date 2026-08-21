#!/usr/bin/env python3
"""Wall-clock comparison of LAStools against pinned PDAL and pdg (B0275).

LAStools' open-source tools (laszip, las2las, lasmerge; built from
github.com/LAStools/LAStools) run natively; the proprietary tools
(lasground_new, lasheight, lasnoise, lasthin, las2dem, lastile, lasgrid,
lascolor) run as the vendor's 64-bit Windows binaries under wine in
`-demo` mode (unlicensed; the vendor states demo mode may perturb output).
Outputs are NOT compared for equality: LAStools implements different
algorithms (and demo mode adds noise), so this is a timing comparison of
comparable jobs only. Every command's wall time is measured around the
whole process (wine startup included for the .exe tools) and, for the
proprietary tools, the tool's own reported "total time" is captured too.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", required=True, type=pathlib.Path)
    p.add_argument("--pdg", required=True, type=pathlib.Path)
    p.add_argument("--lastools-bin", required=True, type=pathlib.Path,
                   help="directory with the open-source LAStools binaries")
    p.add_argument("--lastools-exe", type=pathlib.Path,
                   help="directory with the proprietary 64-bit .exe tools")
    p.add_argument("--wine", default="wine")
    p.add_argument("--laz", required=True, type=pathlib.Path)
    p.add_argument("--las", required=True, type=pathlib.Path)
    p.add_argument("--merge-a", type=pathlib.Path)
    p.add_argument("--merge-b", type=pathlib.Path)
    p.add_argument("--work-dir", required=True, type=pathlib.Path)
    p.add_argument("--report", required=True, type=pathlib.Path)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--frozen-time-library", type=pathlib.Path)
    return p.parse_args()


def clean_env(frozen: pathlib.Path | None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PDG_", "PDAL_TEST_"))}
    if frozen:
        env["LD_PRELOAD"] = str(frozen)
        env["PDAL_TEST_FROZEN_EPOCH"] = "1704067200"
    return env


def timed(command: list[str], env: dict, cwd: pathlib.Path,
          runs: int, outputs: list[pathlib.Path]) -> dict:
    walls = []
    reported = []
    rc = None
    for _ in range(runs):
        for out in outputs:
            if out.exists():
                out.unlink()
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=cwd, env=env, text=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        walls.append(time.perf_counter() - started)
        rc = completed.returncode
        m = (re.search(r"total time ([0-9.]+) sec", completed.stdout) or
             re.search(r"took ([0-9.]+) sec", completed.stdout))
        if m:
            reported.append(float(m.group(1)))
    return {
        "command": [str(c) for c in command],
        "returncode": rc,
        "wall_seconds": walls,
        "wall_median": statistics.median(walls),
        "tool_reported_seconds": reported,
        "tool_reported_median": statistics.median(reported) if reported else None,
        "outputs": [{"name": o.name, "bytes": o.stat().st_size if o.exists() else None}
                    for o in outputs],
    }


def main() -> int:
    a = parse_args()
    a.work_dir.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(tempfile.mkdtemp(prefix="lastools-", dir=a.work_dir))
    env = clean_env(None)   # LAStools/wine: no frozen clock
    pdal_env = clean_env(a.frozen_time_library)
    wine_env = dict(env)
    wine_env["WINEDEBUG"] = "-all"
    wine_env["WINEPREFIX"] = str(a.work_dir / "wineprefix")
    lt = a.lastools_bin
    exe = a.lastools_exe
    laz, las = a.laz.resolve(), a.las.resolve()
    results: dict = {"jobs": {}, "environment": {
        "lastools_bin": str(lt), "lastools_exe": str(exe) if exe else None,
        "runs": a.runs}}

    def pipeline(name: str, stages: list, out: str) -> dict:
        path = work / f"{name}.json"
        path.write_text(json.dumps({"pipeline": stages}))
        return {"pdal": [str(a.oracle), "pipeline", str(path)],
                "pdg": [str(a.pdg), "pipeline", str(path)], "out": out}

    def job(name: str, description: str, pdal_stages: list, out: str,
            lastools_cmd: list[str] | None, lastools_out: str,
            wine_tool: bool = False, note: str = "") -> None:
        entry: dict = {"description": description, "note": note}
        if pdal_stages:
            spec = pipeline(name, pdal_stages, out)
            entry["pinned_pdal"] = timed(spec["pdal"], pdal_env, work, a.runs,
                                        [work / out])
            entry["pdg"] = timed(spec["pdg"], pdal_env, work, a.runs,
                                 [work / out])
        if lastools_cmd:
            cmd = list(lastools_cmd)
            if wine_tool:
                if exe is None or not shutil.which(a.wine):
                    entry["lastools"] = {"skipped": "no wine or exe dir"}
                    results["jobs"][name] = entry
                    return
                cmd = [a.wine] + cmd
            entry["lastools"] = timed(cmd, wine_env if wine_tool else env, work,
                                      a.runs, [work / lastools_out])
            entry["lastools"]["tool"] = "wine+demo" if wine_tool else "open-source"
        results["jobs"][name] = entry
        print(name, {k: round(v.get("wall_median", 0), 3) for k, v in entry.items()
                     if isinstance(v, dict) and "wall_median" in v})

    # I/O and conversion.
    job("las_to_laz", "LAS -> LAZ compression (r14)",
        [{"type": "readers.las", "filename": str(las)},
         {"type": "writers.las", "filename": "out.laz", "compression": True}],
        "out.laz", [str(lt / "laszip64"), "-i", str(las), "-o", "lt-out.laz"],
        "lt-out.laz")
    job("laz_to_las", "LAZ -> LAS decompression (r14 companion)",
        [{"type": "readers.las", "filename": str(laz)},
         {"type": "writers.las", "filename": "out.las"}],
        "out.las", [str(lt / "laszip64"), "-i", str(laz), "-o", "lt-out.las"],
        "lt-out.las")
    job("reproject", "LAZ -> reproject EPSG:28992 -> EPSG:3857 -> LAZ (r1 without the crop)",
        [{"type": "readers.las", "filename": str(laz)},
         {"type": "filters.reprojection", "in_srs": "EPSG:28992", "out_srs": "EPSG:3857"},
         {"type": "writers.las", "filename": "out.laz", "compression": True}],
        "out.laz", [str(lt / "las2las64"), "-i", str(laz), "-proj_epsg", "28992", "3857",
                    "-o", "lt-reproj.laz"], "lt-reproj.laz",
        note="las2las loads libproj dynamically; PDAL uses GDAL/PROJ")
    if a.merge_a and a.merge_b:
        job("merge", "two LAZ -> merged LAZ (r13)",
            [{"type": "readers.las", "filename": str(a.merge_a.resolve()), "tag": "left"},
             {"type": "readers.las", "filename": str(a.merge_b.resolve()), "tag": "right"},
             {"type": "filters.merge", "inputs": ["left", "right"]},
             {"type": "writers.las", "filename": "out.laz", "compression": True}],
            "out.laz", [str(lt / "lasmerge64"), "-i", str(a.merge_a.resolve()),
                        str(a.merge_b.resolve()), "-o", "lt-merge.laz"], "lt-merge.laz")
    # Proprietary tools under wine, demo mode.
    def wexe(name: str) -> str:
        return str((exe / name).resolve()) if exe else name
    job("ground", "ground classification (r2/r3 first stage): PDAL SMRF vs lasground_new -demo",
        [{"type": "readers.las", "filename": str(laz)},
         {"type": "filters.smrf"},
         {"type": "writers.las", "filename": "out.laz", "compression": True}],
        "out.laz", [wexe("lasground_new64.exe"), "-demo", "-i", str(laz), "-o", "lt-ground.laz"],
        "lt-ground.laz", wine_tool=True)
    job("ground_hag", "ground + height above ground (r2): SMRF+HAG vs lasground_new|lasheight -demo",
        [{"type": "readers.las", "filename": str(laz)},
         {"type": "filters.smrf"}, {"type": "filters.hag_nn"},
         {"type": "writers.las", "filename": "out.laz", "compression": True,
          "extra_dims": "HeightAboveGround=float32"}],
        "out.laz", None, "", note="LAStools: two processes; see ground and hag entries")
    job("hag", "height above ground given classified ground: lasheight -demo (PDAL: see ground_hag)",
        [], "", [wexe("lasheight64.exe"), "-demo", "-i", "lt-ground.laz", "-o", "lt-hag.laz"],
        "lt-hag.laz", wine_tool=True)
    job("dtm", "DTM raster 1 m from ground (r3): PDAL SMRF+range+writers.gdal idw vs las2dem -demo on lasground output",
        [{"type": "readers.las", "filename": str(laz)},
         {"type": "filters.smrf"},
         {"type": "filters.range", "limits": "Classification[2:2]"},
         {"type": "writers.gdal", "filename": "out.tif", "output_type": "idw", "resolution": 1.0}],
        "out.tif", [wexe("las2dem64.exe"), "-demo", "-i", "lt-ground.laz", "-keep_class", "2",
                    "-step", "1", "-elevation", "-otif", "-o", "lt-dtm.tif"], "lt-dtm.tif",
        wine_tool=True, note="LAStools time excludes its lasground step (see ground)")
    job("dsm", "first/only-return maximum-Z DSM 1 m (r7): PDAL returns+writers.gdal max vs lasgrid -demo -highest",
        [{"type": "readers.las", "filename": str(laz)},
         {"type": "filters.returns", "groups": "first,only"},
         {"type": "writers.gdal", "filename": "out.tif", "output_type": "max",
          "resolution": 1.0, "binmode": True, "dimension": "Z", "data_type": "float64"}],
        "out.tif", [wexe("lasgrid64.exe"), "-demo", "-i", str(laz), "-first_only", "-step", "1",
                    "-highest", "-otif", "-o", "lt-dsm.tif"], "lt-dsm.tif", wine_tool=True)
    job("denoise", "statistical outlier removal (r4 first stage): PDAL outlier vs lasnoise -demo",
        [{"type": "readers.las", "filename": str(laz)},
         {"type": "filters.outlier", "method": "statistical", "mean_k": 8, "multiplier": 2.0},
         {"type": "writers.las", "filename": "out.laz", "compression": True}],
        "out.laz", [wexe("lasnoise64.exe"), "-demo", "-i", str(laz), "-o", "lt-noise.laz"],
        "lt-noise.laz", wine_tool=True)
    job("thin", "spatial thinning to ~1 m (r4/r10 style): PDAL filters.sample radius 1 vs lasthin -demo -step 1",
        [{"type": "readers.las", "filename": str(laz)},
         {"type": "filters.sample", "radius": 1.0},
         {"type": "writers.las", "filename": "out.laz", "compression": True}],
        "out.laz", [wexe("lasthin64.exe"), "-demo", "-i", str(laz), "-step", "1", "-o", "lt-thin.laz"],
        "lt-thin.laz", wine_tool=True)
    job("tile", "tiling into fixed tiles (r12): PDAL splitter (7 tiles) vs lastile -demo -tile_size 500",
        [{"type": "readers.las", "filename": str(laz)},
         {"type": "filters.splitter", "length": 500},
         {"type": "writers.las", "filename": "tile#.laz", "compression": True}],
        "tile1.laz", [wexe("lastile64.exe"), "-demo", "-i", str(laz), "-tile_size", "500",
                      "-olaz", "-o", "lt-tile.laz"], "lt-tile.laz", wine_tool=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(results, indent=2) + "\n")
    print("report", a.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
