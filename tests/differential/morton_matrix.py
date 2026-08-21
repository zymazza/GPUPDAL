#!/usr/bin/env python3
"""Exact complete-process matrix for filters.mortonorder."""

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
    Case("forward-default", "simple.las", (
        {"type": "filters.mortonorder"},
    )),
    Case("reverse", "simple.las", (
        {"type": "filters.mortonorder", "reverse": True},
    )),
    Case("forward-color", "1.2-with-color.las", (
        {"type": "filters.mortonorder"},
    )),
    Case("duplicate-coordinates", "simple.las", (
        {"type": "filters.assign", "value": "X = Classification"},
        {"type": "filters.assign", "value": "Y = ReturnNumber"},
        {"type": "filters.mortonorder"},
    )),
    Case("reverse-duplicates", "simple.las", (
        {"type": "filters.assign", "value": "X = Classification"},
        {"type": "filters.assign", "value": "Y = ReturnNumber"},
        {"type": "filters.mortonorder", "reverse": True},
    )),
    Case("then-predicate", "simple.las", (
        {"type": "filters.mortonorder"},
        {"type": "filters.expression", "expression": "Classification == 2"},
    )),
    Case("sequential", "simple.las", (
        {"type": "filters.mortonorder"},
        {"type": "filters.mortonorder", "reverse": True},
    )),
    Case("empty-forward", "no-points.las", (
        {"type": "filters.mortonorder"},
    )),
    Case("empty-reverse", "no-points.las", (
        {"type": "filters.mortonorder", "reverse": True},
    )),
    Case("multibatch", "4_6.las", (
        {"type": "filters.mortonorder"},
    )),
    Case("degenerate-x", "simple.las", (
        {"type": "filters.assign", "value": "X = 7"},
        {"type": "filters.mortonorder"},
    )),
    Case("degenerate-y-reverse", "simple.las", (
        {"type": "filters.assign", "value": "Y = -2"},
        {"type": "filters.mortonorder", "reverse": True},
    )),
    Case("where-fallback", "simple.las", (
        {"type": "filters.mortonorder", "where": "Classification == 2"},
    ), hybrid=False),
    Case("string-reverse-fallback", "simple.las", (
        {"type": "filters.mortonorder", "reverse": "true"},
    ), hybrid=False),
    Case("unknown-option", "simple.las", (
        {"type": "filters.mortonorder", "not_an_option": 1},
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
        prefix="pdg-morton-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / "las" / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing Morton fixture: {fixture}", file=sys.stderr)
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
                "--case", f"morton-matrix-{case.name}",
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

    print(f"exact Morton process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
