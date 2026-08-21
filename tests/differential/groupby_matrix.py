#!/usr/bin/env python3
"""Exact complete-process matrix for filters.groupby."""

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
    Case("classification", "simple.las", (
        {"type": "filters.groupby", "dimension": "Classification"},
    )),
    Case("return-number", "simple.las", (
        {"type": "filters.groupby", "dimension": "ReturnNumber"},
    )),
    Case("custom-negative", "simple.las", (
        {"type": "filters.assign",
         "value": "Scratch = Classification - 3"},
        {"type": "filters.groupby", "dimension": "Scratch"},
    )),
    Case("single-group", "simple.las", (
        {"type": "filters.assign", "value": "Scratch = 7"},
        {"type": "filters.groupby", "dimension": "Scratch"},
    )),
    Case("after-stable-sort", "simple.las", (
        {"type": "filters.sort", "dimension": "Intensity",
         "algorithm": "STABLE"},
        {"type": "filters.groupby", "dimension": "Classification"},
    )),
    Case("after-splitter", "simple.las", (
        {"type": "filters.splitter", "length": 1000},
        {"type": "filters.groupby", "dimension": "Classification"},
    )),
    Case("sequential-groups", "simple.las", (
        {"type": "filters.groupby", "dimension": "Classification"},
        {"type": "filters.groupby", "dimension": "ReturnNumber"},
    )),
    Case("multibatch", "4_6.las", (
        {"type": "filters.groupby", "dimension": "Classification"},
    )),
    Case("empty", "no-points.las", (
        {"type": "filters.groupby", "dimension": "Classification"},
    )),
    Case("conversion-failure", "simple.las", (
        {"type": "filters.assign", "value": "Scratch = 1e20"},
        {"type": "filters.groupby", "dimension": "Scratch"},
    )),
    Case("where-fallback", "simple.las", (
        {"type": "filters.groupby", "dimension": "Classification",
         "where": "Classification > 1"},
    ), hybrid=False),
    Case("nonstring-dimension", "simple.las", (
        {"type": "filters.groupby", "dimension": 7},
    ), hybrid=False),
    Case("unknown-option", "simple.las", (
        {"type": "filters.groupby", "dimension": "Classification",
         "not_an_option": 1},
    ), hybrid=False),
    Case("missing-dimension", "simple.las", (
        {"type": "filters.groupby", "dimension": "NotHere"},
    ), hybrid=False),
    Case("downstream-ordinal-fallback", "simple.las", (
        {"type": "filters.groupby", "dimension": "Classification"},
        {"type": "filters.head", "count": 1},
    ), hybrid=False),
    Case("splitter-downstream-ordinal-fallback", "simple.las", (
        {"type": "filters.splitter", "length": 1000},
        {"type": "filters.head", "count": 1},
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
        prefix="pdg-groupby-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / "las" / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing groupby fixture: {fixture}", file=sys.stderr)
                return 2
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({
                    "pipeline": [
                        {"type": "readers.las", "filename": "input.las"},
                        *case.filters,
                        {"type": "writers.las",
                         "filename": "group#.las"},
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
                "--case", f"groupby-matrix-{case.name}",
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

    print(f"exact groupby process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
