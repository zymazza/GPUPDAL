#!/usr/bin/env python3
"""Exact pinned-oracle process matrix for filters.sample (D0267).

The fork replaces the sampler's ordered voxel map with a hash map, reads
candidate lists in place, and prunes neighbor voxels that lie farther than
the radius from the point. Every greedy keep/drop decision must equal pinned
PDAL's, so every case compares the public candidate (delegating to the
forked sibling) against the pinned upstream oracle in streaming and standard
modes on lattices dense enough to exercise the 27-voxel neighborhood, with
the marker-dimension form and explicit origins.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    fixture: str
    stage: dict[str, object]
    nostream: bool = False


CASES = (
    Case("radius-1-stream", "dense", {"type": "filters.sample", "radius": 1.0}),
    Case("radius-1-nostream", "dense", {"type": "filters.sample", "radius": 1.0},
         nostream=True),
    Case("radius-0.35-stream", "dense",
         {"type": "filters.sample", "radius": 0.35}),
    Case("cell-2-nostream", "dense", {"type": "filters.sample", "cell": 2.0},
         nostream=True),
    Case("radius-1-marker-dimension-stream", "dense",
         {"type": "filters.sample", "radius": 1.0, "dimension": "Kept"}),
    Case("radius-1-origins-nostream", "dense",
         {"type": "filters.sample", "radius": 1.0, "origin_x": 184500.5,
          "origin_y": 494900.25, "origin_z": -3.0}, nostream=True),
    Case("radius-1-sparse-stream", "sparse",
         {"type": "filters.sample", "radius": 1.0}),
    Case("radius-1-jittered-nostream", "jittered",
         {"type": "filters.sample", "radius": 1.0}, nostream=True),
)


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


def write_fixture(path: Path, kind: str) -> None:
    lines = ["X,Y,Z,Intensity"]
    if kind == "dense":
        # 0.5 m lattice with slow height variation: about 40,000 points,
        # radius 1.0 rejects most of them through the neighborhood probes.
        for i in range(200):
            for j in range(200):
                x = 184500.0 + 0.5 * i
                y = 494920.0 + 0.5 * j
                z = 1.0 + 0.002 * i + 0.001 * j + (0.6 if (i * 3 + j) % 11 == 0 else 0.0)
                lines.append(f"{x:.6f},{y:.6f},{z:.6f},{(i + j) % 251}")
    elif kind == "sparse":
        for i in range(120):
            for j in range(120):
                x = 184500.0 + 1.7 * i
                y = 494920.0 + 1.9 * j
                z = 2.0 + 0.003 * (i * 120 + j) % 5
                lines.append(f"{x:.6f},{y:.6f},{z:.6f},{(i * 5 + j) % 251}")
    else:
        # Deterministic jitter puts many points near voxel boundaries.
        for k in range(30000):
            i, j = k % 150, k // 150
            x = 184500.0 + 0.7 * i + 0.13 * ((k * 7919) % 97) / 97.0
            y = 494920.0 + 0.7 * j + 0.11 * ((k * 104729) % 89) / 89.0
            z = 1.0 + 0.9 * ((k * 1299709) % 101) / 101.0
            lines.append(f"{x:.6f},{y:.6f},{z:.6f},{k % 251}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.pinned_oracle is None:
        print("sample matrix skipped: no --pinned-oracle configured",
              file=sys.stderr)
        return 0
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdg-sample-matrix-",
                                     dir=args.work_dir) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = generated / f"{case.fixture}.csv"
            if not fixture.exists():
                write_fixture(fixture, case.fixture)
            pipeline = generated / f"{case.name}.json"
            order = "X,Y,Z,Intensity"
            if "dimension" in case.stage:
                order += "," + str(case.stage["dimension"])
            pipeline.write_text(json.dumps({"pipeline": [
                {"type": "readers.text", "filename": "input.csv"},
                case.stage,
                {"type": "writers.text", "filename": "output.txt",
                 "order": order, "keep_unspecified": False, "precision": 17},
            ]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            command = [
                sys.executable, str(args.differential.resolve()),
                "--oracle", str(args.pinned_oracle.resolve()),
                "--candidate-oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", f"sample-matrix-{case.name}",
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--seed-file", f"input.csv={fixture.resolve()}",
                "--seed-file", f"pipeline.json={pipeline.resolve()}",
                "--", "pipeline", "pipeline.json",
            ]
            if case.nostream:
                command.append("--nostream")
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                return completed.returncode
    print(f"exact sample matrix: {len(CASES)} pinned-oracle cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
