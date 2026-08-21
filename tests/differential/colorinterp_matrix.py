#!/usr/bin/env python3
"""Exact complete-process matrix for filters.colorinterp."""

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
    numbered: bool = False


RAMPS = (
    "awesome_green",
    "black_orange",
    "blue_hue",
    "blue_red",
    "heat_map",
    "pestel_shades",
    "blue_orange",
)


CASES = (
    *(Case(f"builtin-{ramp}", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600, "ramp": ramp},
    )) for ramp in RAMPS),
    Case("external-ramp", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600, "ramp": "ramp.tif"},
    )),
    Case("invert", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600, "ramp": "blue_red", "invert": True},
    )),
    Case("clamp", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 450,
         "maximum": 500, "clamp": True},
    )),
    Case("intensity", "simple.las", (
        {"type": "filters.colorinterp", "dimension": "Intensity",
         "minimum": 0, "maximum": 65535, "ramp": "heat_map"},
    )),
    Case("preserve-existing-outside", "1.2-with-color.las", (
        {"type": "filters.colorinterp", "minimum": 0,
         "maximum": 1, "ramp": "awesome_green"},
    )),
    Case("autorange", "simple.las", (
        {"type": "filters.colorinterp", "ramp": "black_orange"},
    )),
    Case("minimum-only", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "ramp": "blue_hue"},
    )),
    Case("maximum-only", "simple.las", (
        {"type": "filters.colorinterp", "maximum": 600,
         "ramp": "blue_hue"},
    )),
    Case("stddev-k", "simple.las", (
        {"type": "filters.colorinterp", "k": 1.25,
         "ramp": "blue_orange"},
    )),
    Case("negative-k", "simple.las", (
        {"type": "filters.colorinterp", "k": -1.0,
         "ramp": "blue_orange"},
    )),
    Case("mad", "simple.las", (
        {"type": "filters.colorinterp", "k": 1.5, "mad": True,
         "ramp": "pestel_shades"},
    )),
    Case("mad-multiplier", "simple.las", (
        {"type": "filters.colorinterp", "k": 2.0, "mad": True,
         "mad_multiplier": 2.5, "ramp": "pestel_shades"},
    )),
    Case("explicit-k-stream", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600, "k": 3.0, "mad": True,
         "ramp": "blue_red"},
    )),
    Case("single-point-autorange", "gps-time-nan.las", (
        {"type": "filters.colorinterp", "dimension": "Z"},
    )),
    Case("nan-source-host-fallback", "gps-time-nan.las", (
        {"type": "filters.colorinterp", "dimension": "GpsTime",
         "minimum": 0, "maximum": 1},
    )),
    Case("after-coordinate-assign", "simple.las", (
        {"type": "filters.assign", "value": "Z = Z - 400"},
        {"type": "filters.colorinterp", "minimum": 0,
         "maximum": 200, "ramp": "heat_map"},
    )),
    Case("after-expression", "simple.las", (
        {"type": "filters.expression", "expression": "Z >= 500"},
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600, "ramp": "heat_map"},
    )),
    Case("after-sort", "simple.las", (
        {"type": "filters.sort", "dimension": "Intensity",
         "algorithm": "STABLE"},
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600},
    )),
    Case("after-divider", "simple.las", (
        {"type": "filters.divider", "count": 7},
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600},
    ), numbered=True),
    Case("autorange-after-divider", "simple.las", (
        {"type": "filters.divider", "mode": "round_robin", "count": 3},
        {"type": "filters.colorinterp", "ramp": "blue_red"},
    ), numbered=True),
    Case("before-divider", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600},
        {"type": "filters.divider", "count": 5},
    ), numbered=True),
    Case("consecutive", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600, "ramp": "awesome_green"},
        {"type": "filters.colorinterp", "dimension": "Intensity",
         "minimum": 0, "maximum": 65535, "ramp": "blue_red"},
    )),
    Case("empty-explicit", "no-points.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600},
    )),
    Case("empty-autorange", "no-points.las", (
        {"type": "filters.colorinterp"},
    )),
    Case("multibatch", "4_6.las", (
        {"type": "filters.colorinterp", "minimum": 7000,
         "maximum": 7200, "ramp": "heat_map"},
    )),
    Case("bad-ramp", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600, "ramp": "ramp-that-does-not-exist"},
    )),
    Case("missing-dimension", "simple.las", (
        {"type": "filters.colorinterp", "dimension": "NotHere",
         "minimum": 0, "maximum": 1},
    ), hybrid=False),
    Case("invalid-range", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 10, "maximum": 10},
    ), hybrid=False),
    Case("string-minimum", "simple.las", (
        {"type": "filters.colorinterp", "minimum": "400",
         "maximum": 600},
    ), hybrid=False),
    Case("string-clamp", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600, "clamp": "true"},
    ), hybrid=False),
    Case("where-fallback", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600, "where": "Classification > 1"},
    ), hybrid=False),
    Case("unknown-option", "simple.las", (
        {"type": "filters.colorinterp", "minimum": 400,
         "maximum": 600, "not_an_option": 1},
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
    ramp = (args.fixture_root / "gdal" /
            "1234_red_0_green_0_blue.tif").resolve()
    if not ramp.is_file():
        print(f"missing color ramp fixture: {ramp}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(
        prefix="pdg-colorinterp-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / "las" / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing colorinterp fixture: {fixture}",
                      file=sys.stderr)
                return 2
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({
                    "pipeline": [
                        {"type": "readers.las", "filename": "input.las"},
                        *case.filters,
                        {"type": "writers.las",
                         "filename": ("result#.las" if case.numbered
                                      else "result.las")},
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
                "--case", f"colorinterp-matrix-{case.name}",
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--seed-file", f"input.las={fixture}",
                "--seed-file", f"ramp.tif={ramp}",
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

    print(f"exact colorinterp process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
