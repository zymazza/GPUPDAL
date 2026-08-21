#!/usr/bin/env python3
"""Exact output-and-metadata process matrix for filters.expressionstats."""

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
    cuda_exact: bool = False


CASES = (
    Case("single", "las/simple.las", (
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": "Classification == 2"},
    ), cuda_exact=True),
    Case("default-unqualified", "las/simple.las", (
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": "Classification == 2"},
    ), hybrid=False),
    Case("canonical-expression-order", "las/simple.las", (
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": ["Classification >= 2", "Classification == 1",
                         "Intensity < 100"]},
    ), cuda_exact=True),
    Case("overlapping", "las/simple.las", (
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": ["Classification >= 1", "Classification <= 2"]},
    ), cuda_exact=True),
    Case("floating-dimension", "pdg/metadata-points.csv", (
        {"type": "filters.expressionstats", "dimension": "X",
         "expressions": ["Classification >= 2", "Intensity < 50"]},
    ), cuda_exact=True),
    Case("nan-dimension", "las/gps-time-nan.las", (
        {"type": "filters.expressionstats", "dimension": "GpsTime",
         "expressions": "GpsTime != 0"},
    )),
    Case("no-matches", "las/simple.las", (
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": ["Classification > 100", "Classification < -1"]},
    ), cuda_exact=True),
    Case("empty", "las/no-points.las", (
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": ["Classification == 1", "Classification == 2"]},
    )),
    Case("forced-stream", "las/4_6.las", (
        {"type": "filters.expressionstats", "dimension": "ReturnNumber",
         "expressions": ["Classification == 2", "ReturnNumber >= 1"]},
    ), mode="stream", cuda_exact=True),
    Case("forced-standard", "las/4_6.las", (
        {"type": "filters.expressionstats", "dimension": "ReturnNumber",
         "expressions": ["Classification == 2", "ReturnNumber >= 1"]},
    ), mode="nostream", cuda_exact=True),
    Case("between-point-regions", "las/simple.las", (
        {"type": "filters.assign", "value": "Scratch = Intensity"},
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": ["Scratch < 100", "Classification == 2"]},
        {"type": "filters.assign", "value": "UserData = Classification"},
    ), cuda_exact=True),
    Case("custom-target", "las/simple.las", (
        {"type": "filters.assign", "value": "Scratch = Classification"},
        {"type": "filters.expressionstats", "dimension": "Scratch",
         "expressions": ["Classification == 1", "Classification >= 2"]},
    ), cuda_exact=True),
    Case("after-divider", "pdg/metadata-points.csv", (
        {"type": "filters.divider", "count": 2, "mode": "round_robin"},
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": ["Intensity <= 30", "Classification >= 2"]},
        {"type": "filters.merge"},
    ), cuda_exact=True),
    Case("invalid-dimension", "las/simple.las", (
        {"type": "filters.expressionstats", "dimension": "NotHere",
         "expressions": "Classification == 2"},
    ), hybrid=False),
    Case("invalid-expression", "las/simple.las", (
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": "Classification === 2"},
    ), hybrid=False),
    Case("invalid-dimension-type", "las/simple.las", (
        {"type": "filters.expressionstats", "dimension": 7,
         "expressions": "Classification == 2"},
    ), hybrid=False),
    Case("invalid-expressions-type", "las/simple.las", (
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": True},
    ), hybrid=False),
    Case("where-fallback", "las/simple.las", (
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": "Classification == 2", "where": "X > 0"},
    ), hybrid=False),
    Case("unknown-option", "las/simple.las", (
        {"type": "filters.expressionstats", "dimension": "Classification",
         "expressions": "Classification == 2", "bogus": True},
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
    parser.add_argument(
        "--require-cuda", action="store_true",
        help="run only exact CUDA-envelope cases and require device execution",
    )
    return parser.parse_args()


def reader(fixture: str) -> tuple[dict[str, object], str]:
    if fixture.endswith(".csv"):
        return {"type": "readers.text", "filename": "input.csv"}, "input.csv"
    return {"type": "readers.las", "filename": "input.las"}, "input.las"


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pdg-expressionstats-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        executed = 0
        for case in CASES:
            if args.require_cuda and not case.cuda_exact:
                continue
            executed += 1
            fixture = (args.fixture_root / case.fixture).resolve()
            if not fixture.is_file():
                print(
                    f"missing expressionstats fixture: {fixture}",
                    file=sys.stderr,
                )
                return 2
            input_stage, input_name = reader(case.fixture)
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({
                    "pipeline": [
                        input_stage,
                        *case.filters,
                        {"type": "writers.las", "filename": "out.las"},
                    ]
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            if args.require_cuda:
                environment.update({
                    "PDG_DISABLE_NATIVE": "1",
                    "PDG_REQUIRE_HYBRID": "1",
                    "PDG_REQUIRE_CUDA_HYBRID": "1",
                })
                environment.pop("PDG_DISABLE_CUDA_HYBRID", None)
            elif case.hybrid:
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
                "--case", f"expressionstats-matrix-{case.name}",
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
            if case.mode:
                command.append(f"--{case.mode}")
            command.extend(("--metadata", "metadata.json"))
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode:
                return completed.returncode

    lane = "CUDA" if args.require_cuda else "host/fallback"
    print(f"exact expressionstats {lane} process matrix: {executed} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
