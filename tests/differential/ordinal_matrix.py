#!/usr/bin/env python3
"""Exact process matrix for decimation/head/tail modes and edge options."""

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
    streamable: bool
    hybrid: bool = True


CASES = (
    Case("decimation-default", "simple.las",
         ({"type": "filters.decimation"},), True),
    Case("decimation-integral", "simple.las",
         ({"type": "filters.decimation", "step": 10},), True),
    Case("decimation-fractional", "simple.las",
         ({"type": "filters.decimation", "step": 4.2},), True),
    Case("decimation-offset-limit", "simple.las",
         ({"type": "filters.decimation", "step": 2.6,
           "offset": 10, "limit": 90},), True),
    Case("decimation-offset-after-end", "simple.las",
         ({"type": "filters.decimation", "step": 2,
           "offset": 2000},), True),
    Case("decimation-zero-limit", "simple.las",
         ({"type": "filters.decimation", "limit": 0},), True),
    Case("head-default", "simple.las", ({"type": "filters.head"},), True),
    Case("head-zero", "simple.las",
         ({"type": "filters.head", "count": 0},), True),
    Case("head-zero-invert", "simple.las",
         ({"type": "filters.head", "count": 0, "invert": True},), True),
    Case("head-oversize", "simple.las",
         ({"type": "filters.head", "count": 2000},), True),
    Case("tail-default", "simple.las", ({"type": "filters.tail"},), False),
    Case("tail-zero", "simple.las",
         ({"type": "filters.tail", "count": 0},), False),
    Case("tail-zero-invert", "simple.las",
         ({"type": "filters.tail", "count": 0, "invert": True},), False),
    Case("tail-oversize-warning", "simple.las",
         ({"type": "filters.tail", "count": 2000},), False),
    Case("tail-oversize-invert-warning", "simple.las",
         ({"type": "filters.tail", "count": 2000, "invert": True},), False),
    Case("standard-chain", "simple.las", (
        {"type": "filters.decimation", "step": 4.2},
        {"type": "filters.head", "count": 200},
        {"type": "filters.tail", "count": 50, "invert": True},
    ), False),
    Case("stream-predicate-chain", "simple.las", (
        {"type": "filters.expression", "expression": "Classification <= 5"},
        {"type": "filters.decimation", "step": 2.5},
        {"type": "filters.head", "count": 100, "invert": True},
    ), True),
    Case("stream-two-decimations", "simple.las", (
        {"type": "filters.decimation", "step": 2.5},
        {"type": "filters.decimation", "step": 3.5,
         "offset": 2, "limit": 300},
    ), True),
    Case("empty-head", "no-points.las",
         ({"type": "filters.head", "count": 10},), True),
    Case("empty-tail-warning", "no-points.las",
         ({"type": "filters.tail", "count": 10},), False),
    # A nonzero standard offset has an upstream unsigned-underflow failure
    # domain.  The hybrid classifier therefore delegates before execution.
    Case("standard-offset-guard", "simple.las", (
        {"type": "filters.decimation", "step": 2.5,
         "offset": 10, "limit": 1000},
        {"type": "filters.tail", "count": 20},
    ), False, False),
    Case("string-option-fallback", "simple.las",
         ({"type": "filters.decimation", "step": "2.5"},), True, False),
    Case("invalid-step-fallback", "simple.las",
         ({"type": "filters.decimation", "step": 0.5},), True, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument(
        "--oracle-preload",
        action="append",
        default=[],
        type=Path,
        help=(
            "shared library to preload before the frozen-time shim for the oracle"
        ),
    )
    parser.add_argument(
        "--candidate-preload",
        action="append",
        default=[],
        type=Path,
        help=(
            "shared library to preload before the frozen-time shim for the candidate"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pdg-ordinal-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / "las" / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing ordinal fixture: {fixture}", file=sys.stderr)
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
                if case.streamable:
                    environment["PDG_REQUIRE_STREAMING_HYBRID"] = "1"
                else:
                    environment.pop("PDG_REQUIRE_STREAMING_HYBRID", None)
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
                "--case", f"ordinal-matrix-{case.name}",
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

    print(f"exact ordinal process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
