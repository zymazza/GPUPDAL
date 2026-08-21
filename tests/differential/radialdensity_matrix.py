#!/usr/bin/env python3
"""Exact process matrix for filters.radialdensity and radius scaling."""

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
    stage: dict[str, object]
    hybrid: bool = True


CASES = (
    Case("default", "pdg/neighborhood-points.csv",
         {"type": "filters.radialdensity"}),
    Case("strict-boundary", "pdg/neighborhood-points.csv",
         {"type": "filters.radialdensity", "radius": 1}),
    Case("expanded", "pdg/neighborhood-points.csv",
         {"type": "filters.radialdensity", "radius": 1.01}),
    Case("tiny", "pdg/neighborhood-points.csv",
         {"type": "filters.radialdensity", "radius": 0.01}),
    Case("large", "pdg/neighborhood-points.csv",
         {"type": "filters.radialdensity", "radius": 100}),
    Case("negative", "pdg/neighborhood-points.csv",
         {"type": "filters.radialdensity", "radius": -1}),
    Case("zero", "pdg/neighborhood-points.csv",
         {"type": "filters.radialdensity", "radius": 0}),
    Case("empty", "las/no-points.las",
         {"type": "filters.radialdensity", "radius": 2}),
    Case("where-fallback", "las/simple.las",
         {"type": "filters.radialdensity", "where": "Classification == 2"},
         hybrid=False),
    Case("string-radius-fallback", "las/simple.las",
         {"type": "filters.radialdensity", "radius": "1"}, hybrid=False),
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
        prefix="pdg-radialdensity-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing radialdensity fixture: {fixture}",
                      file=sys.stderr)
                return 2
            is_csv = case.fixture.endswith(".csv")
            input_name = "input.csv" if is_csv else "input.las"
            reader = "readers.text" if is_csv else "readers.las"
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({"pipeline": [
                    {"type": reader, "filename": input_name},
                    case.stage,
                    {"type": "writers.las", "filename": "out.las",
                     "extra_dims": "all"},
                ]}, indent=2, sort_keys=True) + "\n",
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
                    "PDG_DISABLE_NATIVE", "PDG_REQUIRE_HYBRID",
                    "PDG_DISABLE_CUDA_HYBRID",
                ):
                    environment.pop(name, None)

            command = [
                sys.executable, str(args.differential.resolve()),
                "--oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", f"radialdensity-matrix-{case.name}",
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--seed-file", f"{input_name}={fixture}",
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

    print(f"exact radialdensity process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
