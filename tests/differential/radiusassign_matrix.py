#!/usr/bin/env python3
"""Exact process matrix for filters.radiusassign and radius membership."""

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
    Case("self-inclusion", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification[1:1]",
        "reference_domain": "Classification[1:1]",
        "radius": 0.01,
        "update_expression": "Intensity = 100",
    }),
    Case("duplicate-reference", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification[1:1]",
        "reference_domain": "Classification[2:2]",
        "radius": 0.01,
        "update_expression": "Intensity = 101",
    }),
    Case("strict-radius-boundary", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification[1:1]",
        "reference_domain": "Classification[3:3]",
        "radius": 1,
        "update_expression": "Intensity = 102",
    }),
    Case("inside-radius-boundary", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification[1:1]",
        "reference_domain": "Classification[3:3]",
        "radius": 1.01,
        "update_expression": "Intensity = 103",
    }),
    Case("three-dimensional", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification[1:1]",
        "reference_domain": "Classification[6:6]",
        "radius": 1.01,
        "is3d": True,
        "update_expression": "Intensity = 104",
    }),
    Case("two-dimensional-cap-equality", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification[1:1]",
        "reference_domain": "Classification[6:6]",
        "radius": 0.01,
        "max2d_above": 1,
        "update_expression": "Intensity = 105",
    }),
    Case("two-dimensional-cap-excluded", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification[1:1]",
        "reference_domain": "Classification[6:6]",
        "radius": 0.01,
        "max2d_above": 0,
        "update_expression": "Intensity = 106",
    }),
    Case("two-dimensional-below-equality", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification[6:6]",
        "reference_domain": "Classification[1:1]",
        "radius": 0.01,
        "max2d_below": 1,
        "update_expression": "Intensity = 111",
    }),
    Case("two-dimensional-below-excluded", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification[6:6]",
        "reference_domain": "Classification[1:1]",
        "radius": 0.01,
        "max2d_below": 0,
        "update_expression": "Intensity = 112",
    }),
    Case("open-negated-domain", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification!(1:8)",
        "reference_domain": "Classification[2:2]",
        "radius": 100,
        "update_expression": "Intensity = 113",
    }),
    Case("source-domain-or", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification[1:1],Intensity[70:70]",
        "reference_domain": "Classification[2:2]",
        "radius": 100,
        "update_expression": "Intensity = 107",
    }),
    Case("conditional-update", "pdg/neighborhood-points.csv", {
        "type": "filters.radiusassign",
        "src_domain": "Classification[1:1]",
        "reference_domain": "Classification[2:2]",
        "radius": 0.01,
        "update_expression": "Intensity = 109 WHERE Classification == 1",
    }),
    Case("empty", "las/no-points.las", {
        "type": "filters.radiusassign",
        "radius": 1,
        "update_expression": "Classification = 2",
    }),
    Case("where-fallback", "las/simple.las", {
        "type": "filters.radiusassign",
        "radius": 1,
        "update_expression": "Classification = 2",
        "where": "Classification == 1",
    }, hybrid=False),
    Case("missing-update-expression", "las/simple.las", {
        "type": "filters.radiusassign",
        "radius": 1,
    }, hybrid=False),
    Case("zero-radius", "las/simple.las", {
        "type": "filters.radiusassign",
        "radius": 0,
        "update_expression": "Classification = 2",
    }, hybrid=False),
    Case("bad-source-dimension", "las/simple.las", {
        "type": "filters.radiusassign",
        "src_domain": "NotADimension[0:1]",
        "radius": 1,
        "update_expression": "Classification = 2",
    }, hybrid=False),
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
        prefix="pdg-radiusassign-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        executed = 0
        for case in CASES:
            if args.require_cuda and not case.hybrid:
                continue
            executed += 1
            fixture = (args.fixture_root / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing radiusassign fixture: {fixture}",
                      file=sys.stderr)
                return 2
            is_csv = case.fixture.endswith(".csv")
            input_name = "input.csv" if is_csv else "input.las"
            reader = "readers.text" if is_csv else "readers.las"
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({"pipeline": [
                    {"type": reader, "filename": input_name},
                    case.stage,
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
                "--case", f"radiusassign-matrix-{case.name}",
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
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode:
                return completed.returncode

    print(f"exact radiusassign process matrix: {executed} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
