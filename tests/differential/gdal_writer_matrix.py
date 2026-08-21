#!/usr/bin/env python3
"""Exact pinned-oracle process matrix for writers.gdal banded accumulation (D0273).

The fork accumulates raster cells on a fixed-slot pool: points are bucketed
by the row bands their radius can touch and each band replays its points in
original order updating only its own rows, so every cell sees the pinned
sequence of count/min/max/mean/stdev/idw/percentile updates. Every case
compares the public candidate (delegating to the forked sibling) against
the pinned upstream oracle: raster bytes and metadata, stdout, stderr,
status. The test-only ``PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS`` hook
makes the fixtures large enough for the parallel path on any machine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


FORCED = ("PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=3",)
BOUNDS = "([184500,184680],[494920,495020])"


@dataclass(frozen=True)
class Case:
    name: str
    writer: dict[str, object]
    nostream: bool = False
    filters: tuple[dict[str, object], ...] = ()
    candidate_env: tuple[str, ...] = FORCED


CASES = (
    Case("all-dynamic-stream", {"resolution": 1.0}),
    Case("all-dynamic-nostream", {"resolution": 1.0}, nostream=True),
    Case("idw-dynamic-nostream", {"resolution": 1.0, "output_type": "idw"},
         nostream=True),
    Case("idw-power-radius-stream", {"resolution": 0.5, "output_type": "idw",
         "power": 2.5, "radius": 1.25}),
    Case("mean-stdev-count-stream", {"resolution": 2.0,
         "output_type": "mean,stdev,count"}),
    Case("mean-stdev-count-nostream", {"resolution": 2.0,
         "output_type": "mean,stdev,count"}, nostream=True),
    Case("max-binmode-bounds-stream", {"resolution": 1.0, "output_type": "max",
         "binmode": True, "bounds": BOUNDS}),
    Case("max-binmode-bounds-nostream", {"resolution": 1.0,
         "output_type": "max", "binmode": True, "bounds": BOUNDS},
         nostream=True),
    Case("min-fixed-bounds-stream", {"resolution": 1.5, "output_type": "min",
         "bounds": BOUNDS}),
    Case("all-window-fill-nostream", {"resolution": 3.0, "window_size": 3},
         nostream=True),
    Case("percentiles-binmode-nostream", {"resolution": 2.0,
         "output_type": "p10,p50,p90,count", "binmode": True}, nostream=True),
    Case("intensity-dimension-float32-stream", {"resolution": 1.0,
         "dimension": "Intensity", "data_type": "float32",
         "output_type": "mean,max"}),
    Case("range-skips-stream", {"resolution": 1.0, "output_type": "idw"},
         filters=({"type": "filters.range", "limits": "Intensity[0:40000]"},)),
    Case("range-skips-nostream", {"resolution": 1.0, "output_type": "idw"},
         filters=({"type": "filters.range", "limits": "Intensity[0:40000]"},),
         nostream=True),
    Case("large-radius-stream", {"resolution": 1.0, "radius": 4.0,
         "output_type": "idw,mean"}),
    Case("bounds-clip-outside-points-nostream", {"resolution": 1.0,
         "bounds": "([184550,184600],[494950,494980])"}, nostream=True),
    Case("workers-disabled-control-nostream", {"resolution": 1.0},
         nostream=True, candidate_env=(
             "PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS=1",)),
    Case("default-policy-stream", {"resolution": 1.0}, candidate_env=()),
    Case("default-policy-nostream", {"resolution": 1.0}, nostream=True,
         candidate_env=()),
)


def write_fixture(path: Path) -> None:
    # 30,000 points scattered over 180 x 100 m so radius-1 IDW cells receive
    # many points from several rows, plus a dense strip near the top edge and
    # exact duplicates so per-cell folds and zero-distance IDW hits occur.
    lines = ["X,Y,Z,Intensity"]
    for k in range(30000):
        x = 184500.0 + 180.0 * ((k * 7919) % 10007) / 10007.0
        y = 494920.0 + 100.0 * ((k * 104729) % 10009) / 10009.0
        z = -2.0 + 9.0 * ((k * 1299709) % 1009) / 1009.0
        if k % 977 == 0:
            y = 495019.5
        lines.append(f"{x:.4f},{y:.4f},{z:.4f},{(k * 37) % 65536}")
    for k in range(200):
        lines.append(f"{184550.5:.4f},{494950.5:.4f},{1.0 + 0.01 * k:.4f},"
                     f"{k % 65536}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path,
                        help="the forked sibling pdal (candidate delegate)")
    parser.add_argument("--pinned-oracle", type=Path,
                        help="pinned upstream pdal; skips when absent")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pinned_oracle is None:
        print("gdal writer matrix skipped: no --pinned-oracle configured",
              file=sys.stderr)
        return 0
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdg-gdal-writer-matrix-",
                                     dir=args.work_dir) as temporary:
        generated = Path(temporary)
        fixture = generated / "points.csv"
        write_fixture(fixture)
        for case in CASES:
            writer = dict(case.writer)
            writer["type"] = "writers.gdal"
            writer["filename"] = "output.tif"
            writer.setdefault("gdaldriver", "GTiff")
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(json.dumps({"pipeline": [
                {"type": "readers.text", "filename": "input.csv"},
                *case.filters,
                writer,
            ]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            name = f"gdal-writer-{case.name}"
            command = [
                sys.executable, str(args.differential.resolve()),
                "--oracle", str(args.pinned_oracle.resolve()),
                "--candidate-oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", name,
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--keep-success",
                "--seed-file", f"input.csv={fixture.resolve()}",
                "--seed-file", f"pipeline.json={pipeline.resolve()}",
            ]
            for assignment in case.candidate_env:
                command.extend(["--candidate-env", assignment])
            command.extend(["--", "pipeline", "pipeline.json"])
            if case.nostream:
                command.append("--nostream")
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                return completed.returncode
            report_path = args.work_dir / "reports" / f"{name}.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["oracle_run"]["returncode"] != 0:
                print(f"{name}: oracle failed", file=sys.stderr)
                return 1
            artifact = report["oracle_artifacts"].get("output.tif")
            if artifact is None or artifact["bytes"] <= 0:
                print(f"{name}: no output.tif published", file=sys.stderr)
                return 1
    print(f"exact gdal writer matrix: {len(CASES)} pinned-oracle cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
