#!/usr/bin/env python3
"""Exact process matrix for filters.transformation and fallback modes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


IDENTITY = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
AFFINE = "1 0.125 0 10.25 0.0625 1 0 -20.5 0 0 1 3.75 0 0 0 1"
PROJECTIVE = (
    "1 0 0 10 0 1 0 -20 0 0 1 3 "
    "0.0000001 -0.00000005 0 1"
)


@dataclass(frozen=True)
class Case:
    name: str
    fixture: str
    filters: tuple[dict[str, object], ...]
    hybrid: bool = True
    matrix_file: bool = False


CASES = (
    Case("identity", "simple.las",
         ({"type": "filters.transformation", "matrix": IDENTITY},)),
    Case("affine-mixed", "simple.las",
         ({"type": "filters.transformation", "matrix": AFFINE},)),
    Case("affine-multibatch", "4_6.las",
         ({"type": "filters.transformation", "matrix": AFFINE},)),
    Case("projective", "simple.las",
         ({"type": "filters.transformation", "matrix": PROJECTIVE},)),
    Case("operation-chain", "simple.las", (
        {"type": "filters.assign", "value": "X = X + 0.01"},
        {"type": "filters.transformation", "matrix": AFFINE},
        {"type": "filters.expression", "expression": "Z >= 0"},
        {"type": "filters.assign",
         "value": "Classification = 11 WHERE X > 0"},
    )),
    Case("empty", "no-points.las",
         ({"type": "filters.transformation", "matrix": AFFINE},)),
    Case("invert-fallback", "simple.las", (
        {"type": "filters.transformation", "matrix": AFFINE,
         "invert": True},
    ), hybrid=False),
    Case("override-srs-fallback", "interesting.las", (
        {"type": "filters.transformation", "matrix": IDENTITY,
         "override_srs": "EPSG:3857"},
    ), hybrid=False),
    Case("matrix-file-fallback", "simple.las", (
        {"type": "filters.transformation", "matrix": "matrix.txt"},
    ), hybrid=False, matrix_file=True),
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
        prefix="pdg-transformation-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        matrix_path = generated / "matrix.txt"
        matrix_path.write_text(IDENTITY + "\n", encoding="utf-8")
        for case in CASES:
            fixture = (args.fixture_root / "las" / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing transformation fixture: {fixture}",
                      file=sys.stderr)
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
                    "PDG_REQUIRE_STREAMING_HYBRID": "1",
                    "PDG_DISABLE_CUDA_HYBRID": "1",
                })
            else:
                for name in (
                    "PDG_DISABLE_NATIVE",
                    "PDG_REQUIRE_HYBRID",
                    "PDG_REQUIRE_STREAMING_HYBRID",
                    "PDG_DISABLE_CUDA_HYBRID",
                ):
                    environment.pop(name, None)

            command = [
                sys.executable,
                str(args.differential.resolve()),
                "--oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", f"transformation-matrix-{case.name}",
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--seed-file", f"input.las={fixture}",
                "--seed-file", f"pipeline.json={pipeline.resolve()}",
            ]
            if case.matrix_file:
                command.extend(("--seed-file",
                                f"matrix.txt={matrix_path.resolve()}"))
            for preload in args.oracle_preload:
                command.extend(("--oracle-preload", str(preload.resolve())))
            for preload in args.candidate_preload:
                command.extend(("--candidate-preload", str(preload.resolve())))
            command.extend(("--", "pipeline", "pipeline.json"))
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode:
                return completed.returncode

    print(f"exact transformation process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
