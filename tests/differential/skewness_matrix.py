#!/usr/bin/env python3
"""Exact bounded process matrix for filters.skewnessbalancing."""

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
    cuda: bool = True
    verbose: bool = False
    experimental_fallback: bool = False
    reader: dict[str, object] | None = None
    before: tuple[dict[str, object], ...] = ()
    output: str = "out.las"


CASES = (
    Case("defaults", "default", {"type": "filters.skewnessbalancing"}, verbose=True),
    Case("one-point", "one-point", {"type": "filters.skewnessbalancing"}),
    Case("two-point", "two-point", {"type": "filters.skewnessbalancing"}),
    Case("z-sort-order", "z-sort", {"type": "filters.skewnessbalancing"}),
    Case("skew-boundary", "skew-boundary",
         {"type": "filters.skewnessbalancing"}, cuda=False),
    Case("skew-near-boundary", "skew-near",
         {"type": "filters.skewnessbalancing"}, cuda=False),
    Case("signed-zero-tie", "signed-zero",
         {"type": "filters.skewnessbalancing"}, cuda=False),
    Case(
        "multiple-crossing",
        "multiple-crossing",
        {"type": "filters.skewnessbalancing"},
    ),
    Case(
        "custom-class-only-ground",
        "mixed-classes",
        {
            "type": "filters.skewnessbalancing",
            "ground_class": 3,
            "other_class": 9,
            "only_ground": True,
        },
    ),
    Case(
        "custom-class-not-only-ground",
        "mixed-classes",
        {
            "type": "filters.skewnessbalancing",
            "ground_class": 3,
            "other_class": 9,
            "only_ground": False,
        },
    ),
    Case(
        "empty",
        "empty",
        {"type": "filters.skewnessbalancing"},
        hybrid=False,
        cuda=False,
    ),
    Case(
        "constant-z",
        "constant-z",
        {"type": "filters.skewnessbalancing"},
        hybrid=False,
        cuda=False,
    ),
    Case(
        "equal-z-order",
        "equal-z",
        {"type": "filters.skewnessbalancing"},
        hybrid=False,
        cuda=False,
    ),
    Case(
        "nonfinite-data",
        "nonfinite",
        {"type": "filters.skewnessbalancing"},
        hybrid=False,
        cuda=False,
    ),
    Case(
        "infinite-data",
        "infinite",
        {"type": "filters.skewnessbalancing"},
        hybrid=False,
        cuda=False,
    ),
    Case(
        "planner-fallback-where",
        "default",
        {"type": "filters.skewnessbalancing", "where": "Classification != 7"},
        hybrid=False,
        cuda=False,
    ),
    Case(
        "multi-view-rewriter-fallback",
        "default",
        {"type": "filters.skewnessbalancing"},
        hybrid=False,
        cuda=False,
        experimental_fallback=True,
        before=({"type": "filters.divider", "count": 2},),
        output="out#.las",
    ),
    Case(
        "unproven-reader-rewriter-fallback",
        "default",
        {"type": "filters.skewnessbalancing"},
        hybrid=False,
        cuda=False,
        experimental_fallback=True,
        reader={
            "type": "readers.faux",
            "bounds": "([0,4],[0,0],[0,4])",
            "count": 5,
            "mode": "ramp",
        },
        before=({"type": "filters.assign", "value": "Classification = 7"},),
    ),
    Case(
        "invalid-nonfinite-ground-option",
        "default",
        {"type": "filters.skewnessbalancing", "ground_class": "nan"},
        hybrid=False,
        cuda=False,
    ),
    Case(
        "invalid-equal-classes-option",
        "default",
        {
            "type": "filters.skewnessbalancing",
            "ground_class": 4,
            "other_class": 4,
        },
        hybrid=False,
        cuda=False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--oracle-preload", action="append", default=[], type=Path)
    parser.add_argument("--candidate-preload", action="append", default=[], type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def terrain_rows(kind: str) -> list[tuple[float, float, float, int]]:
    if kind == "empty":
        return []
    if kind == "one-point":
        return [(0.0, 0.0, 7.0, 7)]
    if kind == "two-point":
        return [(0.0, 0.0, 3.0, 7), (1.0, 0.0, 3.5, 7)]
    if kind == "default":
        return [
            (0.0, 0.0, 0.0, 7),
            (1.0, 0.0, 1.0, 7),
            (2.0, 0.0, 2.0, 7),
            (3.0, 0.0, 3.0, 7),
            (4.0, 0.0, 4.0, 7),
        ]
    if kind == "constant-z":
        return [
            (0.0, 0.0, 10.0, 7),
            (1.0, 0.0, 10.0, 7),
            (2.0, 0.0, 10.0, 7),
            (3.0, 0.0, 10.0, 7),
        ]
    if kind == "z-sort":
        return [
            (0.0, 0.0, 12.0, 7),
            (1.0, 0.0, 1.0, 7),
            (2.0, 0.0, 7.5, 7),
            (3.0, 0.0, 3.0, 7),
            (4.0, 0.0, 5.0, 7),
        ]
    if kind == "equal-z":
        return [
            (0.0, 0.0, 4.0, 7),
            (1.0, 0.0, 4.0, 7),
            (2.0, 0.0, 4.0, 1),
            (3.0, 0.0, 4.0, 7),
            (4.0, 0.0, 4.0, 1),
        ]
    if kind == "skew-boundary":
        return [
            (0.0, 0.0, 1.0, 7),
            (1.0, 0.0, 2.0, 7),
            (2.0, 0.0, 3.5, 7),
            (3.0, 0.0, 2.0, 7),
            (4.0, 0.0, 0.4, 7),
        ]
    if kind == "skew-near":
        return [
            (0.0, 0.0, 1.0, 7),
            (1.0, 0.0, 2.0, 7),
            (2.0, 0.0, 3.0000001, 7),
            (3.0, 0.0, 2.0, 7),
            (4.0, 0.0, 1.0, 7),
        ]
    if kind == "signed-zero":
        return [
            (0.0, 0.0, 0.0, 7),
            (1.0, 0.0, -0.0, 7),
            (2.0, 0.0, 1.0, 7),
        ]
    if kind == "multiple-crossing":
        return [
            (0.0, 0.0, 4.0, 7),
            (1.0, 0.0, 0.0, 7),
            (2.0, 0.0, 6.0, 7),
            (3.0, 0.0, 1.0, 7),
            (4.0, 0.0, 3.0, 7),
        ]
    if kind == "mixed-classes":
        return [
            (0.0, 0.0, 0.4, 2),
            (1.0, 0.0, 1.2, 3),
            (2.0, 0.0, 0.9, 4),
            (3.0, 0.0, 1.1, 2),
            (4.0, 0.0, 2.5, 4),
        ]
    if kind == "nonfinite":
        return [
            (0.0, 0.0, float("nan"), 7),
            (1.0, 0.0, 1.0, 7),
            (2.0, 0.0, 2.0, 7),
            (3.0, 0.0, 3.0, 7),
        ]
    if kind == "infinite":
        return [
            (0.0, 0.0, float("-inf"), 7),
            (1.0, 0.0, 1.0, 7),
            (2.0, 0.0, 2.0, 7),
            (3.0, 0.0, float("inf"), 7),
        ]
    raise ValueError(f"unknown fixture '{kind}'")


def write_fixture(path: Path, kind: str) -> None:
    lines = ["X,Y,Z,Classification\n"]
    if kind == "empty":
        path.write_text("".join(lines), encoding="utf-8")
        return
    for x, y, z, classification in terrain_rows(kind):
        lines.append(",".join((str(x), str(y), str(z), str(classification))) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def required_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "LC_ALL": "C",
            "PDG_DISABLE_NATIVE": "1",
            "PDG_REQUIRE_HYBRID": "1",
            "PDG_REQUIRE_CUDA_HYBRID": "1",
            "TZ": "UTC",
        }
    )
    environment.pop("PDG_DISABLE_CUDA_HYBRID", None)
    return environment


def verify_required_empty_rejection(
    args: argparse.Namespace, generated: Path
) -> bool:
    fixture = generated / "required-empty.csv"
    write_fixture(fixture, "empty")
    pipeline = generated / "required-empty.json"
    output = generated / "required-empty.las"
    pipeline.write_text(
        json.dumps(
            {
                "pipeline": [
                    {"type": "readers.text", "filename": str(fixture)},
                    {"type": "filters.skewnessbalancing"},
                    {
                        "type": "writers.las",
                        "filename": str(output),
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
    completed = subprocess.run(
        [str(args.candidate.resolve()), "pipeline", str(pipeline)],
        cwd=generated,
        env=required_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    expected = (
        "filters.skewnessbalancing: required exact CUDA hybrid "
        "skewnessbalancing path was not used"
    )
    if completed.returncode == 0 or expected not in completed.stderr or output.exists():
        print("required CUDA empty-input guard failed", file=sys.stderr)
        print(f"status: {completed.returncode}", file=sys.stderr)
        print(completed.stdout, file=sys.stderr)
        print(completed.stderr, file=sys.stderr)
        return False
    print("required CUDA empty input rejected before stage execution")
    return True


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdg-skewness-matrix-", dir=args.work_dir) as temporary:
        generated = Path(temporary)
        if args.require_cuda and not verify_required_empty_rejection(args, generated):
            return 1
        executed = 0
        for case in CASES:
            if args.require_cuda and not case.cuda:
                continue
            executed += 1
            fixture = generated / f"{case.fixture}.csv"
            write_fixture(fixture, case.fixture)
            pipeline = generated / f"{case.name}.json"
            reader = case.reader or {
                "type": "readers.text",
                "filename": "input.csv",
            }
            pipeline.write_text(
                json.dumps(
                    {
                        "pipeline": [
                            reader,
                            *case.before,
                            case.stage,
                            {
                                "type": "writers.las",
                                "filename": case.output,
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
                environment = required_environment()
            elif case.hybrid:
                environment.update(
                    {
                        "PDG_DISABLE_NATIVE": "1",
                        "PDG_REQUIRE_HYBRID": "1",
                        "PDG_DISABLE_CUDA_HYBRID": "1",
                    }
                )
                environment.pop("PDG_REQUIRE_CUDA_HYBRID", None)
                environment.pop("PDG_EXPERIMENTAL_CUDA_HYBRID", None)
            elif case.experimental_fallback:
                environment.update(
                    {
                        "PDG_DISABLE_NATIVE": "1",
                        "PDG_DISABLE_CUDA_HYBRID": "1",
                        "PDG_EXPERIMENTAL_CUDA_HYBRID": "1",
                    }
                )
                environment.pop("PDG_REQUIRE_HYBRID", None)
                environment.pop("PDG_REQUIRE_CUDA_HYBRID", None)
            else:
                for name in (
                    "PDG_DISABLE_NATIVE",
                    "PDG_REQUIRE_HYBRID",
                    "PDG_DISABLE_CUDA_HYBRID",
                    "PDG_REQUIRE_CUDA_HYBRID",
                    "PDG_EXPERIMENTAL_CUDA_HYBRID",
                ):
                    environment.pop(name, None)

            command = [
                sys.executable,
                str(args.differential.resolve()),
                "--oracle",
                str(args.oracle.resolve()),
                "--candidate",
                str(args.candidate.resolve()),
                "--case",
                f"skewnessbalancing-matrix-{case.name}",
                "--work-dir",
                str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--seed-file",
                f"input.csv={fixture.resolve()}",
                "--seed-file",
                f"pipeline.json={pipeline.resolve()}",
            ]
            for preload in args.oracle_preload:
                command.extend(("--oracle-preload", str(preload.resolve())))
            for preload in args.candidate_preload:
                command.extend(("--candidate-preload", str(preload.resolve())))
            command.extend(("--", "pipeline", "pipeline.json"))
            if case.verbose:
                command.append("--verbose=3")
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode:
                return completed.returncode

    print(f"exact skewnessbalancing process matrix: {executed} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
