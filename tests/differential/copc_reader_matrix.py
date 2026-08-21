#!/usr/bin/env python3
"""Exact pinned-oracle process matrix for readers.copc ordered decode (D0270).

With `requests=1`, pinned PDAL fetches and decompresses every selected COPC
node on one thread and emits tiles in hierarchy order. The fork keeps that
single request thread but decompresses tiles on a decode pool and emits
them strictly in fetch order, so points, order, diagnostics, and status are
unchanged while the decode is parallel. Every case compares the public
candidate (delegating to the forked sibling) against the pinned upstream
oracle on a deterministic COPC fixture written by the pinned oracle
(`writers.copc` with `threads=1,fixed_seed=true`).
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
    reader: dict[str, object]
    nostream: bool = False
    candidate_env: tuple[str, ...] = ()
    expected_returncode: int = 0
    text_output: bool = False


BOUNDS = "([184550,184900],[494930,495200])"

CASES = (
    Case("full-stream", {}),
    Case("full-nostream", {}, nostream=True),
    Case("full-text-stream", {}, text_output=True),
    Case("bounds-stream", {"bounds": BOUNDS}),
    Case("bounds-nostream", {"bounds": BOUNDS}, nostream=True),
    Case("bounds-resolution-stream", {"bounds": BOUNDS, "resolution": 1.0}),
    Case("bounds-resolution-nostream-text", {"bounds": BOUNDS,
         "resolution": 1.0}, nostream=True, text_output=True),
    Case("resolution-stream", {"resolution": 2.5}),
    Case("resolution-nostream", {"resolution": 2.5}, nostream=True),
    Case("polygon-stream", {"polygon": "POLYGON((184520 494925, 184880 "
         "494925, 184880 495300, 184700 495100, 184520 495300, 184520 "
         "494925))"}),
    Case("polygon-nostream", {"polygon": "POLYGON((184520 494925, 184880 "
         "494925, 184880 495300, 184700 495100, 184520 495300, 184520 "
         "494925))"}, nostream=True),
    Case("count-stream", {"count": 33333}),
    Case("count-nostream", {"count": 33333}, nostream=True),
    Case("empty-bounds-stream", {"bounds": "([0,1],[0,1])"}),
    Case("keep-alive-2-stream", {"keep_alive": 2}),
    Case("keep-alive-2-nostream", {"keep_alive": 2}, nostream=True),
    Case("workers-disabled-control-stream", {},
         candidate_env=("PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS=1",)),
    Case("workers-disabled-control-bounds-nostream", {"bounds": BOUNDS},
         candidate_env=("PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS=1",),
         nostream=True),
    Case("native-workers-2-stream", {"bounds": BOUNDS},
         candidate_env=("PDG_NATIVE_WORKERS=2",)),
)


def write_text(path: Path, count: int) -> None:
    lines = ["X,Y,Z,Intensity,ReturnNumber,NumberOfReturns,Classification,"
             "GpsTime"]
    for k in range(count):
        # Deterministic pseudo-random scatter over a 500 x 500 x 100 m
        # box; 1.2M points exceed the writer's 100,000-point node capacity
        # by enough for the root, all eight children, and a third level —
        # more nodes than the default keep-alive window, so ordering and
        # backpressure are exercised.
        x = 184500.0 + 500.0 * ((k * 7919) % 10007) / 10007.0
        y = 494920.0 + 500.0 * ((k * 104729) % 10009) / 10009.0
        z = -2.0 + 100.0 * ((k * 1299709) % 1009) / 1009.0
        lines.append(f"{x:.3f},{y:.3f},{z:.3f},{(k * 37) % 65536},"
                     f"{1 + k % 3},{3},{(k * 7) % 32},"
                     f"{300000.0 + 0.0001 * k:.4f}")
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
        print("copc reader matrix skipped: no --pinned-oracle configured",
              file=sys.stderr)
        return 0
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdg-copc-reader-matrix-",
                                     dir=args.work_dir) as temporary:
        generated = Path(temporary)
        text = generated / "points.csv"
        write_text(text, 1200000)
        fixture = generated / "source.copc.laz"
        recipe = generated / "fixture.json"
        recipe.write_text(json.dumps({"pipeline": [
            {"type": "readers.text", "filename": str(text)},
            # Note for the UBSan lane: lazperf's point14 predictors wrap
            # int32 arithmetic while decoding this fixture (vendored code
            # shared with pinned PDAL, byte-identical output); run that lane
            # with tests/sanitizers/ubsan-lazperf.supp.
            {"type": "writers.copc", "filename": str(fixture), "threads": 1,
             "fixed_seed": True, "scale_x": 0.01, "scale_y": 0.01,
             "scale_z": 0.01}]}), encoding="utf-8")
        subprocess.run([str(args.pinned_oracle.resolve()), "pipeline",
                        str(recipe)], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        for case in CASES:
            reader = dict(case.reader)
            reader["type"] = "readers.copc"
            reader["filename"] = "input.copc.laz"
            reader["requests"] = 1
            if case.text_output:
                sink = {"type": "writers.text", "filename": "output.txt",
                        "precision": 17}
                output = "output.txt"
            else:
                sink = {"type": "writers.las", "filename": "output.las",
                        "minor_version": 4, "dataformat_id": 6}
                output = "output.las"
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(json.dumps({"pipeline": [reader, sink]},
                                indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
            name = f"copc-reader-{case.name}"
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
                "--seed-file", f"input.copc.laz={fixture.resolve()}",
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
            observed = report["oracle_run"]["returncode"]
            if observed != case.expected_returncode:
                print(f"{name}: expected status {case.expected_returncode}, "
                      f"observed {observed}", file=sys.stderr)
                return 1
            artifact = report["oracle_artifacts"].get(output)
            if not case.expected_returncode and (
                    artifact is None or artifact["bytes"] <= 0):
                print(f"{name}: no {output} published", file=sys.stderr)
                return 1
    print(f"exact copc reader matrix: {len(CASES)} pinned-oracle cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
