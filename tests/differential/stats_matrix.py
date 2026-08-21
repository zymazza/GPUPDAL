#!/usr/bin/env python3
"""Exact output-and-metadata process matrix for filters.stats."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    fixture: str
    filters: tuple[dict[str, object], ...]
    mode: str | None = None
    hybrid: bool = True
    output: str = "out.las"


CASES = (
    Case("default-all", "simple.las", (
        {"type": "filters.stats"},
    )),
    Case("default-unqualified", "simple.las", (
        {"type": "filters.stats"},
    ), hybrid=False),
    Case("default-bridge-between-regions", "simple.las", (
        {"type": "filters.assign", "value": "Scratch = Intensity"},
        {"type": "filters.stats", "dimensions": "Scratch,Z"},
        {"type": "filters.assign", "value": "UserData = Classification"},
    ), hybrid=False),
    Case("dimensions-string", "simple.las", (
        {"type": "filters.stats", "dimensions": "X,Z"},
    )),
    Case("dimensions-array", "simple.las", (
        {"type": "filters.stats", "dimensions": ["Intensity", "Z"]},
    )),
    Case("advanced", "simple.las", (
        {"type": "filters.stats", "dimensions": "X,Y,Z",
         "advanced": True},
    )),
    Case("enumerate", "simple.las", (
        {"type": "filters.stats", "dimensions": "Classification",
         "enumerate": "Classification"},
    )),
    Case("count", "simple.las", (
        {"type": "filters.stats", "dimensions": "Classification",
         "count": "Classification"},
    )),
    Case("global", "simple.las", (
        {"type": "filters.stats", "dimensions": "Z", "global": "Z"},
    )),
    Case("combined", "simple.las", (
        {"type": "filters.stats",
         "dimensions": ["Z", "Classification", "ReturnNumber"],
         "enumerate": "ReturnNumber", "count": "Classification",
         "global": "Z", "advanced": True},
    )),
    Case("empty", "no-points.las", (
        {"type": "filters.stats", "dimensions": "Z"},
    )),
    Case("nan", "gps-time-nan.las", (
        {"type": "filters.stats", "dimensions": "GpsTime"},
    )),
    Case("forced-stream", "4_6.las", (
        {"type": "filters.stats", "dimensions": "X,Z,Classification"},
    ), mode="stream"),
    Case("forced-standard", "4_6.las", (
        {"type": "filters.stats", "dimensions": "X,Z,Classification"},
    ), mode="nostream"),
    Case("between-point-regions", "simple.las", (
        {"type": "filters.assign", "value": "Scratch = Intensity"},
        {"type": "filters.stats", "dimensions": "Scratch,Z"},
        {"type": "filters.assign", "value": "UserData = Classification"},
    )),
    Case("fused-point-metadata", "simple.las", (
        {"type": "filters.assign", "value": "Scratch = Intensity"},
        {"type": "filters.assign",
         "value": "UserData = Classification"},
        {"type": "filters.stats", "dimensions": "Scratch,Z"},
    )),
    Case("after-divider-merge", "simple.las", (
        {"type": "filters.divider", "count": 3, "mode": "round_robin"},
        {"type": "filters.stats", "dimensions": "Z,Classification",
         "count": "Classification"},
        {"type": "filters.merge"},
    )),
    Case("missing-dimension-warning", "simple.las", (
        {"type": "filters.stats", "dimensions": "Z,NotHere"},
    )),
    Case("missing-enumerate-warning", "simple.las", (
        {"type": "filters.stats", "dimensions": "Z",
         "enumerate": "Classification"},
    )),
    Case("where-fallback", "simple.las", (
        {"type": "filters.stats", "dimensions": "Z", "where": "Z > 0"},
    ), hybrid=False),
    Case("invalid-advanced", "simple.las", (
        {"type": "filters.stats", "advanced": "true"},
    ), hybrid=False),
    Case("unknown-option", "simple.las", (
        {"type": "filters.stats", "bogus": True},
    ), hybrid=False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--oracle-preload", action="append", default=[],
                        type=Path)
    parser.add_argument("--candidate-preload", action="append", default=[],
                        type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pdg-stats-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / "las" / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing stats fixture: {fixture}", file=sys.stderr)
                return 2
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({
                    "pipeline": [
                        {"type": "readers.las", "filename": "input.las"},
                        *case.filters,
                        {"type": "writers.las", "filename": case.output},
                    ]
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            if case.hybrid:
                environment.update({
                    "PDG_DISABLE_NATIVE": "1",
                    "PDG_REQUIRE_HYBRID": "1",
                    "PDG_DISABLE_CUDA_HYBRID": "1",
                })
            else:
                for name in (
                    "PDG_DISABLE_NATIVE",
                    "PDG_REQUIRE_HYBRID",
                    "PDG_DISABLE_CUDA_HYBRID",
                ):
                    environment.pop(name, None)

            command = [
                sys.executable,
                str(args.differential.resolve()),
                "--oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", f"stats-matrix-{case.name}",
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--seed-file", f"input.las={fixture}",
                "--seed-file", f"pipeline.json={pipeline.resolve()}",
            ]
            for preload in args.oracle_preload:
                command.extend(("--oracle-preload", str(preload.resolve())))
            for preload in args.candidate_preload:
                command.extend(("--candidate-preload", str(preload.resolve())))
            command.extend(("--", "pipeline", "pipeline.json"))
            if case.mode:
                command.append(f"--{case.mode}")
            command.extend(("--metadata", "metadata.json"))
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode:
                return completed.returncode

    print(f"exact stats process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
