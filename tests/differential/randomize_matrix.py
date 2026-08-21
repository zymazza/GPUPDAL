#!/usr/bin/env python3
"""Exact complete-process composition matrix for filters.randomize."""

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
    output: str = "out.las"


ASSIGN = {"type": "filters.assign", "value": "UserData = Classification"}

CASES = (
    Case("seed-zero", "simple.las", (
        {"type": "filters.randomize", "seed": 0}, ASSIGN,
    )),
    Case("seed-one", "simple.las", (
        {"type": "filters.randomize", "seed": 1}, ASSIGN,
    )),
    Case("seed-max", "simple.las", (
        {"type": "filters.randomize", "seed": 4_294_967_295}, ASSIGN,
    )),
    Case("between-point-regions", "simple.las", (
        {"type": "filters.assign", "value": "Intensity = Intensity + 1"},
        {"type": "filters.randomize", "seed": 90210},
        {"type": "filters.expression", "expression": "Classification == 2"},
    )),
    Case("then-stable-sort", "simple.las", (
        {"type": "filters.randomize", "seed": 17},
        {"type": "filters.sort", "dimension": "Classification",
         "algorithm": "STABLE"},
        ASSIGN,
    )),
    Case("after-stable-sort", "simple.las", (
        {"type": "filters.sort", "dimension": "Classification",
         "algorithm": "STABLE"},
        {"type": "filters.randomize", "seed": 17}, ASSIGN,
    )),
    Case("empty", "no-points.las", (
        {"type": "filters.randomize", "seed": 3}, ASSIGN,
    )),
    Case("hundred-points", "100-points.las", (
        {"type": "filters.randomize", "seed": 8675309}, ASSIGN,
    )),
    Case("multibatch", "4_6.las", (
        {"type": "filters.randomize", "seed": 123456789}, ASSIGN,
    )),
    Case("per-view-then-merge", "simple.las", (
        {"type": "filters.divider", "count": 3, "mode": "round_robin"},
        {"type": "filters.randomize", "seed": 29},
        {"type": "filters.merge"}, ASSIGN,
    )),
    Case("where-fallback", "simple.las", (
        {"type": "filters.randomize", "seed": 7,
         "where": "Classification == 2"}, ASSIGN,
    ), hybrid=False),
    Case("string-seed-fallback", "simple.las", (
        {"type": "filters.randomize", "seed": "7"}, ASSIGN,
    ), hybrid=False),
    Case("negative-seed", "simple.las", (
        {"type": "filters.randomize", "seed": -1}, ASSIGN,
    ), hybrid=False),
    Case("unknown-option", "simple.las", (
        {"type": "filters.randomize", "seed": 7, "bogus": True}, ASSIGN,
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
        prefix="pdg-randomize-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / "las" / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing randomize fixture: {fixture}", file=sys.stderr)
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
                "--case", f"randomize-matrix-{case.name}",
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

    print(f"exact randomize composition matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
