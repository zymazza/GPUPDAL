#!/usr/bin/env python3
"""Exact process matrix for filters.iqr and filters.mad."""

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
    hybrid: bool = True


CASES = (
    Case("iqr-default", "simple.las",
         ({"type": "filters.iqr", "dimension": "Z"},)),
    Case("iqr-explicit", "simple.las",
         ({"type": "filters.iqr", "dimension": "Z", "k": 1.5},)),
    Case("iqr-zero-k", "simple.las",
         ({"type": "filters.iqr", "dimension": "Z", "k": 0},)),
    Case("iqr-negative-k", "simple.las",
         ({"type": "filters.iqr", "dimension": "Z", "k": -1},)),
    Case("iqr-integer", "simple.las",
         ({"type": "filters.iqr", "dimension": "Intensity", "k": 2.25},)),
    Case("iqr-nan", "gps-time-nan.las",
         ({"type": "filters.iqr", "dimension": "GpsTime"},)),
    Case("mad-default", "simple.las",
         ({"type": "filters.mad", "dimension": "Z"},)),
    Case("mad-explicit", "simple.las", (
        {"type": "filters.mad", "dimension": "Intensity", "k": 2.5,
         "mad_multiplier": 1.4862},
    )),
    Case("mad-zero-multiplier", "simple.las", (
        {"type": "filters.mad", "dimension": "Z", "k": 2,
         "mad_multiplier": 0},
    )),
    Case("mad-nan", "gps-time-nan.las",
         ({"type": "filters.mad", "dimension": "GpsTime"},)),
    Case("custom-chain", "simple.las", (
        {"type": "filters.assign", "value": "Scratch = Intensity * 2"},
        {"type": "filters.mad", "dimension": "Scratch", "k": 2.5,
         "mad_multiplier": 1.4862},
        {"type": "filters.assign", "value": "Classification = 12"},
    )),
    Case("empty-iqr", "no-points.las",
         ({"type": "filters.iqr", "dimension": "Z"},)),
    Case("empty-mad", "no-points.las",
         ({"type": "filters.mad", "dimension": "Z"},)),
    Case("string-k-fallback", "simple.las", (
        {"type": "filters.iqr", "dimension": "Z", "k": "1.5"},
    ), hybrid=False),
    Case("where-fallback", "simple.las", (
        {"type": "filters.mad", "dimension": "Z",
         "where": "Classification == 2"},
    ), hybrid=False),
    Case("missing-dimension-failure", "simple.las", (
        {"type": "filters.iqr", "dimension": "NotHere"},
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
        prefix="pdg-robust-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / "las" / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing robust fixture: {fixture}", file=sys.stderr)
                return 2
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({
                    "pipeline": [
                        {"type": "readers.las", "filename": "input.las"},
                        *case.filters,
                        {"type": "writers.las", "filename": "out.las"},
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
                "--case", f"robust-matrix-{case.name}",
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
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode:
                return completed.returncode

    print(f"exact robust-statistics process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
