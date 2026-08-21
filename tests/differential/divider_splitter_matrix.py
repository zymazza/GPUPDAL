#!/usr/bin/env python3
"""Exact complete-process matrix for filters.divider and filters.splitter."""

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
    Case("divider-partition-two", "simple.las", (
        {"type": "filters.divider", "count": 2},
    )),
    Case("divider-partition-uneven", "simple.las", (
        {"type": "filters.divider", "count": 7},
    )),
    Case("divider-round-robin", "simple.las", (
        {"type": "filters.divider", "mode": "round_robin", "count": 3},
    )),
    Case("divider-uppercase-mode", "simple.las", (
        {"type": "filters.divider", "mode": "ROUND_ROBIN", "count": 7},
    )),
    Case("divider-after-sort", "simple.las", (
        {"type": "filters.sort", "dimension": "Intensity",
         "algorithm": "STABLE"},
        {"type": "filters.divider", "count": 4},
    )),
    Case("divider-after-splitter", "simple.las", (
        {"type": "filters.splitter", "length": 1000},
        {"type": "filters.divider", "count": 2},
    )),
    Case("divider-consecutive", "simple.las", (
        {"type": "filters.divider", "count": 3},
        {"type": "filters.divider", "mode": "round_robin", "count": 2},
    )),
    Case("divider-merge-head", "simple.las", (
        {"type": "filters.divider", "count": 7},
        {"type": "filters.merge"},
        {"type": "filters.head", "count": 11},
    )),
    Case("divider-empty", "no-points.las", (
        {"type": "filters.divider", "count": 3},
    )),
    Case("divider-multibatch", "4_6.las", (
        {"type": "filters.divider", "count": 3},
    )),
    Case("divider-capacity-fallback", "simple.las", (
        {"type": "filters.divider", "capacity": 25},
    ), hybrid=False),
    Case("divider-expression-fallback", "simple.las", (
        {"type": "filters.divider", "expression": "X > 637000"},
    ), hybrid=False),
    Case("divider-both-size-options", "simple.las", (
        {"type": "filters.divider", "count": 2, "capacity": 25},
    ), hybrid=False),
    Case("divider-missing-size", "simple.las", (
        {"type": "filters.divider"},
    ), hybrid=False),
    Case("divider-invalid-count", "simple.las", (
        {"type": "filters.divider", "count": 1},
    ), hybrid=False),
    Case("divider-invalid-mode", "simple.las", (
        {"type": "filters.divider", "mode": "sideways", "count": 2},
    ), hybrid=False),
    Case("divider-string-count", "simple.las", (
        {"type": "filters.divider", "count": "3"},
    ), hybrid=False),
    Case("divider-where-fallback", "simple.las", (
        {"type": "filters.divider", "count": 2,
         "where": "Classification > 1"},
    ), hybrid=False),
    Case("divider-downstream-ordinal-fallback", "simple.las", (
        {"type": "filters.divider", "count": 2},
        {"type": "filters.head", "count": 1},
    ), hybrid=False),
    Case("splitter-default", "simple.las", (
        {"type": "filters.splitter"},
    )),
    Case("splitter-length", "1.2-with-color.las", (
        {"type": "filters.splitter", "length": 1000},
    )),
    Case("splitter-origin", "simple.las", (
        {"type": "filters.splitter", "length": 250,
         "origin_x": 636000.0, "origin_y": 848000.0},
    )),
    Case("splitter-partial-origin", "simple.las", (
        {"type": "filters.splitter", "length": 250,
         "origin_x": 636000.0},
    )),
    Case("splitter-buffer", "1.2-with-color.las", (
        {"type": "filters.splitter", "length": 1000, "buffer": 20},
    )),
    Case("splitter-negative-buffer", "simple.las", (
        {"type": "filters.splitter", "length": 250, "buffer": -1},
    )),
    Case("splitter-after-sort", "simple.las", (
        {"type": "filters.sort", "dimension": "X", "algorithm": "STABLE"},
        {"type": "filters.splitter", "length": 250},
    )),
    Case("splitter-after-groupby", "simple.las", (
        {"type": "filters.groupby", "dimension": "Classification"},
        {"type": "filters.splitter", "length": 250},
    )),
    Case("splitter-consecutive", "simple.las", (
        {"type": "filters.splitter", "length": 500},
        {"type": "filters.splitter", "length": 250},
    )),
    Case("splitter-merge-head", "1.2-with-color.las", (
        {"type": "filters.splitter", "length": 1000},
        {"type": "filters.merge"},
        {"type": "filters.head", "count": 13},
    )),
    Case("splitter-after-coordinate-assign", "simple.las", (
        {"type": "filters.assign", "value": "X = X - 637000"},
        {"type": "filters.splitter", "length": 25,
         "origin_x": 0.0, "origin_y": 848500.0},
    )),
    Case("splitter-empty", "no-points.las", (
        {"type": "filters.splitter", "length": 1000},
    )),
    Case("splitter-invalid-buffer", "simple.las", (
        {"type": "filters.splitter", "length": 1000, "buffer": 500},
    ), hybrid=False),
    Case("splitter-string-length", "simple.las", (
        {"type": "filters.splitter", "length": "1000"},
    ), hybrid=False),
    Case("splitter-where-fallback", "simple.las", (
        {"type": "filters.splitter", "length": 1000,
         "where": "Classification > 1"},
    ), hybrid=False),
    Case("splitter-unknown-option", "simple.las", (
        {"type": "filters.splitter", "length": 1000,
         "not_an_option": 1},
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
        prefix="pdg-divider-splitter-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / "las" / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing divider/splitter fixture: {fixture}",
                      file=sys.stderr)
                return 2
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({
                    "pipeline": [
                        {"type": "readers.las", "filename": "input.las"},
                        *case.filters,
                        {"type": "writers.las",
                         "filename": "result#.las"},
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
                "--case", f"divider-splitter-matrix-{case.name}",
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

    print(f"exact divider/splitter process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
