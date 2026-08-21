#!/usr/bin/env python3
"""Exact output-and-metadata process matrix for filters.info."""

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


CASES = (
    Case("default", "las/simple.las", (
        {"type": "filters.info"},
    )),
    Case("default-unqualified", "las/simple.las", (
        {"type": "filters.info"},
    ), hybrid=False),
    Case("point-range", "pdg/metadata-points.csv", (
        {"type": "filters.info", "point": "0-2,4"},
    )),
    Case("point-alias", "pdg/metadata-points.csv", (
        {"type": "filters.info", "p": "1,3"},
    ), hybrid=False),
    Case("point-order-quirk", "pdg/metadata-points.csv", (
        {"type": "filters.info", "point": "2,0,2"},
    )),
    Case("query-2d-ties", "pdg/metadata-points.csv", (
        {"type": "filters.info", "query": "0,5/2"},
    )),
    Case("query-3d-history", "pdg/metadata-points.csv", (
        {"type": "filters.info", "query": "0,10,100/1"},
    )),
    Case("query-pipe-separator", "pdg/metadata-points.csv", (
        {"type": "filters.info", "query": "0|5/2"},
    )),
    Case("query-space-separator", "pdg/metadata-points.csv", (
        {"type": "filters.info", "query": "0 5/2"},
    )),
    Case("point-and-query", "pdg/metadata-points.csv", (
        {"type": "filters.info", "point": "1", "query": "0,5/2"},
    )),
    Case("empty", "las/no-points.las", (
        {"type": "filters.info"},
    )),
    Case("forced-stream", "las/4_6.las", (
        {"type": "filters.info", "point": "0,10,100"},
    ), mode="stream"),
    Case("forced-standard", "las/4_6.las", (
        {"type": "filters.info", "query": "0,0/3"},
    ), mode="nostream"),
    Case("between-point-regions", "las/simple.las", (
        {"type": "filters.assign", "value": "Scratch = Intensity"},
        {"type": "filters.info", "point": "0,3"},
        {"type": "filters.assign", "value": "UserData = Classification"},
    )),
    Case("after-divider", "pdg/metadata-points.csv", (
        {"type": "filters.divider", "count": 2, "mode": "round_robin"},
        {"type": "filters.info", "query": "0,5/3"},
        {"type": "filters.merge"},
    )),
    Case("invalid-point", "las/simple.las", (
        {"type": "filters.info", "point": "1-x"},
    ), hybrid=False),
    Case("reversed-point-range", "las/simple.las", (
        {"type": "filters.info", "point": "9-2"},
    ), hybrid=False),
    Case("invalid-query", "las/simple.las", (
        {"type": "filters.info", "query": "1,2/3/4"},
    ), hybrid=False),
    Case("where-fallback", "las/simple.las", (
        {"type": "filters.info", "where": "Classification == 2"},
    ), hybrid=False),
    Case("unknown-option", "las/simple.las", (
        {"type": "filters.info", "bogus": True},
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


def reader(fixture: str) -> tuple[dict[str, object], str]:
    if fixture.endswith(".csv"):
        return {"type": "readers.text", "filename": "input.csv"}, "input.csv"
    return {"type": "readers.las", "filename": "input.las"}, "input.las"


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pdg-info-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing info fixture: {fixture}", file=sys.stderr)
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
                "--case", f"info-matrix-{case.name}",
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

    print(f"exact info process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
