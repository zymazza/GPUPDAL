#!/usr/bin/env python3
"""Exact bounded process matrix for filters.elm."""

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
    cuda: bool = True
    verbose: bool = False


CASES = (
    Case("defaults", "default", {"type": "filters.elm"}, verbose=True),
    Case("supported-empty", "empty", {"type": "filters.elm"}),
    Case("custom-class", "default", {"type": "filters.elm", "class": 18}),
    Case("cell-boundary-arithmetic", "cell-boundary",
         {"type": "filters.elm", "cell": 1.25}),
    Case("equal-z-and-source-order-ties", "source-order",
         {"type": "filters.elm", "class": 18, "threshold": -1.0}),
    Case("threshold-equality", "threshold-boundary",
         {"type": "filters.elm", "threshold": 1.0}),
    Case("threshold-near-equality", "threshold-near",
         {"type": "filters.elm", "threshold": 1.0000000001}),
    Case("zero-threshold", "default",
         {"type": "filters.elm", "threshold": 0.0}),
    Case("one-point", "one-point", {"type": "filters.elm"}),
    Case("where-fallback", "default", {
        "type": "filters.elm", "where": "Classification != 7",
    }, hybrid=False, cuda=False),
    Case("invalid-cell-error", "default", {
        "type": "filters.elm", "cell": 0.0,
    }, hybrid=False, cuda=False),
    Case("negative-threshold", "default", {
        "type": "filters.elm", "threshold": -1.0,
    }),
    Case("invalid-class-error", "default", {
        "type": "filters.elm", "class": 256,
    }, hybrid=False, cuda=False),
    Case("nonfinite-threshold-option", "default", {
        "type": "filters.elm", "threshold": "nan",
    }, hybrid=False, cuda=False),
    Case("nonfinite-data-host", "nonfinite", {"type": "filters.elm"},
         cuda=False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--oracle-preload", action="append", default=[],
                        type=Path)
    parser.add_argument("--candidate-preload", action="append", default=[],
                        type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def terrain_rows(kind: str) -> list[tuple[float, float, float, int]]:
    if kind == "empty":
        return []
    if kind == "cell-boundary":
        return [
            (0.00, 0.00, 2.0, 7),
            (1.25, 0.00, 2.0, 7),
            (1.24, 0.50, 0.2, 7),
            (2.49, 0.25, 3.0, 7),
            (0.05, 1.24, 1.8, 7),
            (1.26, 1.25, 4.4, 7),
        ]
    if kind == "source-order":
        return [
            (0.00, 0.00, -0.0, 10),
            (20.00, 0.00, 5.0, 20),
            (0.10, 0.00, 0.0, 11),
            (20.10, 0.00, 5.0, 21),
        ]
    if kind == "threshold-boundary":
        return [
            (0.00, 0.00, 0.0, 7),
            (0.20, 0.00, 1.0, 7),
            (0.40, 0.00, 3.0, 7),
        ]
    if kind == "threshold-near":
        return [
            (0.00, 0.00, 0.0, 7),
            (0.20, 0.00, 1.00000000001, 7),
            (0.40, 0.00, 2.0, 7),
        ]
    if kind == "one-point":
        return [(0.0, 0.0, 1.0, 12)]
    if kind == "nonfinite":
        return [
            (0.00, 0.00, float("nan"), 7),
            (0.10, 0.00, 2.0, 7),
            (0.20, 0.10, 4.0, 7),
        ]
    return [
        (0.00, 0.00, 0.0, 7),
        (1.00, 0.00, 0.2, 7),
        (0.00, 1.00, 0.4, 7),
        (1.00, 1.00, 1.5, 7),
        (2.00, 0.00, 0.1, 7),
    ]


def write_fixture(path: Path, kind: str) -> None:
    lines = ["X,Y,Z,Classification\n"]
    if kind == "empty":
        path.write_text("".join(lines), encoding="utf-8")
        return
    for x, y, z, classification in terrain_rows(kind):
        lines.append(",".join((str(x), str(y), str(z),
                              str(classification))) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdg-elm-matrix-",
                                     dir=args.work_dir) as temporary:
        generated = Path(temporary)
        executed = 0
        for case in CASES:
            if args.require_cuda and not case.cuda:
                continue
            executed += 1
            fixture = generated / f"{case.fixture}.csv"
            write_fixture(fixture, case.fixture)
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps(
                    {
                        "pipeline": [
                            {"type": "readers.text", "filename": "input.csv"},
                            case.stage,
                            {
                                "type": "writers.las",
                                "filename": "out.las",
                                "extra_dims": "all",
                            },
                        ]
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
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
                environment.pop("PDG_REQUIRE_CUDA_HYBRID", None)
            else:
                for name in (
                    "PDG_DISABLE_NATIVE", "PDG_REQUIRE_HYBRID",
                    "PDG_DISABLE_CUDA_HYBRID", "PDG_REQUIRE_CUDA_HYBRID",
                ):
                    environment.pop(name, None)

            command = [
                sys.executable, str(args.differential.resolve()),
                "--oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", f"elm-matrix-{case.name}",
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--seed-file", f"input.csv={fixture.resolve()}",
                "--seed-file", f"pipeline.json={pipeline.resolve()}",
            ]
            for preload in args.oracle_preload:
                command.extend(("--oracle-preload", str(preload.resolve())))
            for preload in args.candidate_preload:
                command.extend(("--candidate-preload", str(preload.resolve())))
            command.extend(("--", "pipeline", "pipeline.json"))
            if case.verbose:
                command.extend(("--verbose", "8"))
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode:
                return completed.returncode

    print(f"exact elm process matrix: {executed} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
