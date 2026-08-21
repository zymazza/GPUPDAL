#!/usr/bin/env python3
"""Exact process matrix for filters.locate reductions and fallback edges."""

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
    Case("max-default-kind", "simple.las",
         ({"type": "filters.locate", "dimension": "Intensity"},)),
    Case("max-explicit", "1.2-with-color.las",
         ({"type": "filters.locate", "dimension": "Z", "minmax": "max"},)),
    Case("min-explicit", "1.2-with-color.las",
         ({"type": "filters.locate", "dimension": "Z", "minmax": "min"},)),
    Case("min-uppercase", "simple.las",
         ({"type": "filters.locate", "dimension": "X", "minmax": "MIN"},)),
    Case("integer-dimension", "simple.las",
         ({"type": "filters.locate", "dimension": "ReturnNumber",
           "minmax": "max"},)),
    Case("nan-maximum", "gps-time-nan.las",
         ({"type": "filters.locate", "dimension": "GpsTime",
           "minmax": "max"},)),
    Case("nan-minimum", "gps-time-nan.las",
         ({"type": "filters.locate", "dimension": "GpsTime",
           "minmax": "min"},)),
    Case("invalid-kind-empty-output", "simple.las",
         ({"type": "filters.locate", "dimension": "Z",
           "minmax": "sideways"},)),
    Case("empty-input", "no-points.las",
         ({"type": "filters.locate", "dimension": "Z"},)),
    Case("point-program-chain", "simple.las", (
        {"type": "filters.assign", "value": "Scratch = Intensity * 2"},
        {"type": "filters.locate", "dimension": "Scratch", "minmax": "max"},
        {"type": "filters.assign", "value": "Classification = 8"},
    )),
    Case("predicate-before-locate", "simple.las", (
        {"type": "filters.expression", "expression": "Intensity >= 100"},
        {"type": "filters.locate", "dimension": "Intensity", "minmax": "min"},
    )),
    Case("missing-dimension-option-fallback", "simple.las",
         ({"type": "filters.locate"},), False),
    Case("unknown-dimension-fallback", "simple.las",
         ({"type": "filters.locate", "dimension": "NotADimension"},), False),
    Case("numeric-minmax-fallback", "simple.las",
         ({"type": "filters.locate", "dimension": "Z", "minmax": 7},), False),
    Case("where-option-fallback", "simple.las",
         ({"type": "filters.locate", "dimension": "Z",
           "where": "Classification == 2"},), False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--oracle-preload", action="append", default=[], type=Path)
    parser.add_argument("--candidate-preload", action="append", default=[], type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pdg-locate-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / "las" / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing locate fixture: {fixture}", file=sys.stderr)
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
                    "PDG_CUDA_CHUNK_POINTS": "17",
                })
            else:
                for name in (
                    "PDG_DISABLE_NATIVE",
                    "PDG_REQUIRE_HYBRID",
                    "PDG_REQUIRE_STREAMING_HYBRID",
                    "PDG_DISABLE_CUDA_HYBRID",
                    "PDG_CUDA_CHUNK_POINTS",
                ):
                    environment.pop(name, None)

            command = [
                sys.executable,
                str(args.differential.resolve()),
                "--oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", f"locate-matrix-{case.name}",
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library", str(args.frozen_time_library.resolve()),
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

    print(f"exact locate process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
