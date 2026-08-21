#!/usr/bin/env python3
"""Exact process matrix for filters.label_duplicates."""

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
    Case("adjacent-single", "simple.las", (
        {"type": "filters.label_duplicates",
         "dimensions": "Classification"},
    )),
    Case("adjacent-multiple", "simple.las", (
        {"type": "filters.label_duplicates",
         "dimensions": ["Classification", "ReturnNumber"]},
    )),
    Case("comma-separated-string", "simple.las", (
        {"type": "filters.label_duplicates",
         "dimensions": "X,Y,Z"},
    )),
    Case("repeated-dimension", "simple.las", (
        {"type": "filters.label_duplicates",
         "dimensions": ["Intensity", "Intensity"]},
    )),
    Case("empty-dimensions", "simple.las", (
        {"type": "filters.label_duplicates"},
    )),
    Case("nan-and-signed-zero", "gps-time-nan.las", (
        {"type": "filters.label_duplicates", "dimensions": "GpsTime"},
    )),
    Case("stable-sort-composition", "simple.las", (
        {"type": "filters.sort", "dimension": "Classification",
         "algorithm": "STABLE"},
        {"type": "filters.label_duplicates",
         "dimensions": "Classification"},
    )),
    Case("sequential-stable-sort-composition", "simple.las", (
        {"type": "filters.sort", "dimension": "ReturnNumber",
         "algorithm": "STABLE"},
        {"type": "filters.sort", "dimension": "Classification",
         "algorithm": "STABLE"},
        {"type": "filters.label_duplicates",
         "dimensions": ["Classification", "ReturnNumber"]},
    )),
    Case("empty-view", "no-points.las", (
        {"type": "filters.label_duplicates", "dimensions": "X"},
    )),
    Case("multibatch", "4_6.las", (
        {"type": "filters.label_duplicates",
         "dimensions": ["X", "Y", "Z"]},
    )),
    Case("where-fallback", "simple.las", (
        {"type": "filters.label_duplicates", "dimensions": "X",
         "where": "Classification == 2"},
    ), hybrid=False),
    Case("where-merge-fallback", "simple.las", (
        {"type": "filters.label_duplicates", "dimensions": "X",
         "where": "Classification == 2", "where_merge": "true"},
    ), hybrid=False),
    Case("invalid-dimensions-type", "simple.las", (
        {"type": "filters.label_duplicates", "dimensions": 7},
    ), hybrid=False),
    Case("missing-dimension", "simple.las", (
        {"type": "filters.label_duplicates", "dimensions": "NotHere"},
    ), hybrid=False),
    Case("self-referential-output", "simple.las", (
        {"type": "filters.label_duplicates", "dimensions": "Duplicate"},
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
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pdg-label-duplicates-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        executed = 0
        for case in CASES:
            if args.require_cuda and not case.hybrid:
                continue
            executed += 1
            fixture = (args.fixture_root / "las" / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing label_duplicates fixture: {fixture}",
                      file=sys.stderr)
                return 2
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({"pipeline": [
                    {"type": "readers.las", "filename": "input.las"},
                    *case.filters,
                    {"type": "writers.las", "filename": "out.las",
                     "extra_dims": "all"},
                ]}, indent=2, sort_keys=True) + "\n",
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
                "--case", f"label-duplicates-matrix-{case.name}",
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

    print(f"exact label_duplicates process matrix: {executed} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
