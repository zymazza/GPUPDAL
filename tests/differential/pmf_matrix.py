#!/usr/bin/env python3
"""Exact bounded process matrix for filters.pmf."""

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


CASES = (
    Case("default-returns", "returns", {"type": "filters.pmf"}),
    Case("returns-first", "returns", {
        "type": "filters.pmf", "returns": "first",
    }),
    Case("returns-all", "returns", {
        "type": "filters.pmf",
        "returns": ["first", "intermediate", "last", "only"],
    }),
    Case("only-ground-custom-classes", "returns", {
        "type": "filters.pmf", "returns": [],
        "only_ground": True, "ground_class": 9, "other_class": 9,
    }),
    Case("custom-exponential", "returns", {
        "type": "filters.pmf", "returns": [], "cell_size": 1.0,
        "exponential": True, "initial_distance": 0.2, "slope": 0.4,
        "max_distance": 1.25, "max_window_size": 9.0,
    }),
    Case("custom-linear", "returns", {
        "type": "filters.pmf", "returns": [], "cell_size": 1.0,
        "exponential": False, "initial_distance": 0.1, "slope": 0.25,
        "max_distance": 0.3, "max_window_size": 9.0,
    }),
    Case("fractional-cell-binning", "fractional", {
        "type": "filters.pmf", "returns": [], "cell_size": 0.6,
        "max_window_size": 5.0,
    }),
    Case("missing-return-dimensions", "no-returns", {
        "type": "filters.pmf",
    }),
    Case("all-zero-returns", "zero-returns", {
        "type": "filters.pmf",
    }),
    Case("ignore-range-fallback", "returns", {
        "type": "filters.pmf", "returns": [],
        "ignore": "Classification[7:7]",
    }, hybrid=False),
    Case("where-fallback", "returns", {
        "type": "filters.pmf", "returns": [],
        "where": "Classification != 7",
    }, hybrid=False),
    Case("partial-zero-error", "partial-zero", {
        "type": "filters.pmf",
    }, hybrid=False),
    Case("invalid-returns-error", "returns", {
        "type": "filters.pmf", "returns": "bogus",
    }, hybrid=False),
    Case("equal-classes-error", "returns", {
        "type": "filters.pmf", "ground_class": 4, "other_class": 4,
    }, hybrid=False),
    Case("missing-ignore-dimension-error", "returns", {
        "type": "filters.pmf", "ignore": "NotHere[0:1]",
    }, hybrid=False),
    Case("empty-error", "empty", {"type": "filters.pmf"}, hybrid=False),
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


def terrain_rows(kind: str) -> list[tuple[float, float, float, int, int, int]]:
    rows: list[tuple[float, float, float, int, int, int]] = []
    point = 0
    for x_index in range(16):
        for y_index in range(16):
            if ((x_index * 3 + y_index * 5) % 17 == 0 and
                    (x_index, y_index) not in ((0, 0), (15, 15))):
                continue
            scale = 0.75 if kind == "fractional" else 1.0
            x = scale * x_index
            y = scale * y_index
            z = 0.05 * x_index + 0.03 * y_index
            if 6 <= x_index <= 9 and 6 <= y_index <= 9:
                z += 4.0
            if (x_index, y_index) == (2, 2):
                z -= 2.0
            return_kind = point % 4
            if return_kind == 0:
                return_number, number_of_returns = 1, 1
            elif return_kind == 1:
                return_number, number_of_returns = 1, 2
            elif return_kind == 2:
                return_number, number_of_returns = 2, 2
            else:
                return_number, number_of_returns = 2, 3
            classification = 34 if point % 31 == 0 else (7 if point % 19 == 0 else 5)
            rows.append((x, y, z, classification,
                         return_number, number_of_returns))
            point += 1
    return rows


def write_fixture(path: Path, kind: str) -> None:
    rows = terrain_rows(kind)
    if kind in ("no-returns", "empty"):
        header = "X,Y,Z,Classification\n"
    else:
        header = "X,Y,Z,Classification,ReturnNumber,NumberOfReturns\n"
    lines = [header]
    if kind == "empty":
        path.write_text("".join(lines), encoding="utf-8")
        return
    for index, (x, y, z, classification, rn, nr) in enumerate(rows):
        if kind == "zero-returns":
            rn, nr = 0, 0
        elif kind == "partial-zero" and index == 3:
            rn = 0
        values: tuple[object, ...]
        if kind == "no-returns":
            values = (x, y, z, classification)
        else:
            values = (x, y, z, classification, rn, nr)
        lines.append(",".join(str(value) for value in values) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdg-pmf-matrix-",
                                     dir=args.work_dir) as temporary:
        generated = Path(temporary)
        executed = 0
        for case in CASES:
            if args.require_cuda and not case.hybrid:
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
                "--case", f"pmf-matrix-{case.name}",
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
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode:
                return completed.returncode

    print(f"exact pmf process matrix: {executed} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
