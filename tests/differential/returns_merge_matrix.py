#!/usr/bin/env python3
"""Exact complete-process matrix for filters.returns and filters.merge."""

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
    fixture_family: str = "las"
    reader_type: str = "readers.las"


CASES = (
    Case("returns-default", "simple.las", (
        {"type": "filters.returns"},
    )),
    Case("returns-all-reordered", "simple.las", (
        {"type": "filters.returns",
         "groups": "last, first, only, intermediate"},
    )),
    Case("returns-array", "simple.las", (
        {"type": "filters.returns", "groups": ["first", "last"]},
    )),
    Case("returns-duplicates-whitespace", "simple.las", (
        {"type": "filters.returns", "groups": " last, last , first "},
    )),
    Case("returns-empty-groups", "simple.las", (
        {"type": "filters.returns", "groups": []},
    )),
    Case("returns-malformed-only", "simple.las", (
        {"type": "filters.assign",
         "value": ["ReturnNumber = 7", "NumberOfReturns = 1"]},
        {"type": "filters.returns", "groups": "only,last"},
    )),
    Case("returns-zero-warnings", "simple.las", (
        {"type": "filters.assign",
         "value": ["ReturnNumber = 0", "NumberOfReturns = 0"]},
        {"type": "filters.returns",
         "groups": "first,intermediate,last,only"},
    )),
    Case("returns-after-sort", "simple.las", (
        {"type": "filters.sort", "dimension": "Intensity",
         "algorithm": "STABLE"},
        {"type": "filters.returns", "groups": "first,last,only"},
    )),
    Case("returns-after-splitter", "simple.las", (
        {"type": "filters.splitter", "length": 1000},
        {"type": "filters.returns", "groups": "first,last,only"},
    )),
    Case("returns-after-groupby", "simple.las", (
        {"type": "filters.groupby", "dimension": "Classification"},
        {"type": "filters.returns", "groups": "first,last,only"},
    )),
    Case("returns-sequential", "simple.las", (
        {"type": "filters.returns", "groups": "first,last,only"},
        {"type": "filters.returns", "groups": "last"},
    )),
    Case("returns-multibatch", "4_6.las", (
        {"type": "filters.returns",
         "groups": "first,intermediate,last,only"},
    )),
    Case("returns-empty-input", "no-points.las", (
        {"type": "filters.returns",
         "groups": "first,intermediate,last,only"},
    )),
    Case("returns-invalid-group", "simple.las", (
        {"type": "filters.returns", "groups": "last,sideways"},
    )),
    Case("returns-where-fallback", "simple.las", (
        {"type": "filters.returns", "groups": "last",
         "where": "Classification > 1"},
    ), hybrid=False),
    Case("returns-nonstring-fallback", "simple.las", (
        {"type": "filters.returns", "groups": 7},
    ), hybrid=False),
    Case("returns-unknown-option", "simple.las", (
        {"type": "filters.returns", "groups": "last",
         "not_an_option": 1},
    ), hybrid=False),
    Case("returns-downstream-ordinal-fallback", "simple.las", (
        {"type": "filters.returns", "groups": "first,last,only"},
        {"type": "filters.head", "count": 1},
    ), hybrid=False),
    Case("merge-single", "simple.las", (
        {"type": "filters.merge"},
    )),
    Case("groupby-merge", "simple.las", (
        {"type": "filters.groupby", "dimension": "Classification"},
        {"type": "filters.merge"},
    )),
    Case("groupby-merge-head", "simple.las", (
        {"type": "filters.groupby", "dimension": "Classification"},
        {"type": "filters.merge"},
        {"type": "filters.head", "count": 7},
    )),
    Case("splitter-merge-head", "simple.las", (
        {"type": "filters.splitter", "length": 1000},
        {"type": "filters.merge"},
        {"type": "filters.head", "count": 7},
    )),
    Case("returns-merge-head", "simple.las", (
        {"type": "filters.returns", "groups": "first,last,only"},
        {"type": "filters.merge"},
        {"type": "filters.head", "count": 7},
    )),
    Case("groupby-merge-groupby", "simple.las", (
        {"type": "filters.groupby", "dimension": "Classification"},
        {"type": "filters.merge"},
        {"type": "filters.groupby", "dimension": "ReturnNumber"},
    )),
    Case("merge-downstream-point-program", "simple.las", (
        {"type": "filters.groupby", "dimension": "Classification"},
        {"type": "filters.merge"},
        {"type": "filters.assign", "value": "Classification = 9"},
        {"type": "filters.head", "count": 7},
    )),
    Case("merge-spatial-reference-fallback", "simple.las", (
        {"type": "filters.merge", "spatialreference": "EPSG:4326"},
    ), hybrid=False),
    Case("merge-where-fallback", "simple.las", (
        {"type": "filters.merge", "where": "Classification > 1"},
    ), hybrid=False),
    Case("merge-unknown-option", "simple.las", (
        {"type": "filters.merge", "not_an_option": 1},
    ), hybrid=False),
    Case("returns-missing-dimensions", "utm17_2.txt", (
        {"type": "filters.returns", "groups": "last"},
    ), fixture_family="text", reader_type="readers.text"),
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
        prefix="pdg-returns-merge-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (
                args.fixture_root / case.fixture_family / case.fixture
            ).resolve()
            if not fixture.is_file():
                print(f"missing returns/merge fixture: {fixture}",
                      file=sys.stderr)
                return 2
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({
                    "pipeline": [
                        {"type": case.reader_type,
                         "filename": "input.las"},
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
                "--case", f"returns-merge-matrix-{case.name}",
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

    print(f"exact returns/merge process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
