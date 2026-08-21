#!/usr/bin/env python3
"""Exact process matrix for filters.approximatecoplanar."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ENGINE_ENVIRONMENT_VARIABLES = (
    "PDG_CUDA_CHUNK_POINTS",
    "PDG_DEBUG_HYBRID",
    "PDG_DISABLE_CUDA_HYBRID",
    "PDG_DISABLE_CUDA_POINT_PROGRAM",
    "PDG_DISABLE_HYBRID",
    "PDG_DISABLE_NATIVE",
    "PDG_EXPERIMENTAL_CUDA_HYBRID",
    "PDG_EXPERIMENTAL_CUDA_POINT_PROGRAM",
    "PDG_EXPERIMENTAL_CUDA_TRANSLATE",
    "PDG_NATIVE_WORKERS",
    "PDG_REQUIRE_CUDA_HYBRID",
    "PDG_REQUIRE_CUDA_POINT_PROGRAM",
    "PDG_REQUIRE_CUDA_TRANSLATE",
    "PDG_REQUIRE_HYBRID",
    "PDG_REQUIRE_NATIVE",
    "PDG_REQUIRE_STREAMING_HYBRID",
    "PDG_FORCE_MORTON_BVH",
    "PDG_FORCE_LEGACY_CUDA_ALLOCATOR",
    "PDG_FORCE_UNIFORM_GRID",
    "PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE",
    "PDG_REQUIRE_NEIGHBORHOOD_COPLANAR_COLUMN_REUSE",
    "PDG_REQUIRE_NEIGHBORHOOD_BRIDGE_REBUILD",
    "PDG_REQUIRE_NEIGHBORHOOD_EIGENSYSTEM_REUSE",
    "PDG_REQUIRE_NEIGHBORHOOD_REUSE",
    "PDG_REQUIRE_NONTERMINAL_NEIGHBORHOOD_COLUMN_REUSE",
    "PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR",
    "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA",
    "PDG_REQUIRE_APPROXIMATECOPLANAR_HOST_FALLBACK",
    "PDG_TEST_APPROXIMATECOPLANAR_RECOVERABLE_CUDA_FAILURE",
    "PDG_REQUIRE_SPATIAL_TILING",
    "PDG_SPATIAL_TILE_EDGE",
)


@dataclass(frozen=True)
class Case:
    name: str
    fixture: str
    stage: dict[str, object]
    hybrid: bool = True
    prefix: tuple[dict[str, object], ...] = ()
    verbose: bool = False
    experimental_cuda: bool = False


CASES = (
    Case("default", "las/simple.las",
         {"type": "filters.approximatecoplanar"}),
    Case("default-selection-control", "las/simple.las",
         {"type": "filters.approximatecoplanar"}, False),
    Case("explicit-defaults", "las/simple.las", {
        "type": "filters.approximatecoplanar", "knn": 8,
        "thresh1": 25.0, "thresh2": 6.0,
    }),
    Case("custom-thresholds", "las/simple.las", {
        "type": "filters.approximatecoplanar", "knn": 12,
        "thresh1": 30.5, "thresh2": 4.25,
    }),
    Case("zero-thresholds", "las/simple.las", {
        "type": "filters.approximatecoplanar", "knn": 8,
        "thresh1": 0.0, "thresh2": 0.0,
    }),
    Case("negative-thresholds", "las/simple.las", {
        "type": "filters.approximatecoplanar", "knn": 8,
        "thresh1": -25.0, "thresh2": -6.0,
    }),
    Case("minimum-native-knn", "las/simple.las",
         {"type": "filters.approximatecoplanar", "knn": 3}),
    Case("maximum-native-knn", "las/simple.las",
         {"type": "filters.approximatecoplanar", "knn": 64}),
    Case("ties-and-duplicates", "pdg/neighborhood-points.csv",
         {"type": "filters.approximatecoplanar", "knn": 3}),
    Case("strict-first-equality", "pdg/approximatecoplanar-boundary.csv", {
        "type": "filters.approximatecoplanar", "knn": 6,
        "thresh1": 25.0, "thresh2": 5.0,
    }),
    Case("strict-second-equality", "pdg/approximatecoplanar-boundary.csv", {
        "type": "filters.approximatecoplanar", "knn": 6,
        "thresh1": 24.0, "thresh2": 4.0,
    }),
    Case("strict-inside", "pdg/approximatecoplanar-boundary.csv", {
        "type": "filters.approximatecoplanar", "knn": 6,
        "thresh1": 24.0, "thresh2": 5.0,
    }),
    Case("knn-over-view", "pdg/neighborhood-points.csv",
         {"type": "filters.approximatecoplanar", "knn": 64}),
    Case("zero-preserves-existing", "pdg/approximatecoplanar-boundary.csv",
         {"type": "filters.approximatecoplanar", "knn": 3},
         prefix=(
             {"type": "filters.approximatecoplanar", "knn": 6,
              "thresh1": 24.0, "thresh2": 5.0},
             {"type": "filters.assign", "value": [
                 "Coplanar = 7", "X = 0", "Y = 0", "Z = 0",
             ]},
         ), verbose=True),
    Case("widened-coplanar-fallback",
         "pdg/approximatecoplanar-widened.csv",
         {"type": "filters.approximatecoplanar", "knn": 3},
         experimental_cuda=True, verbose=True),
    Case("missing-x-fallback", "pdg/approximatecoplanar-missing-x.csv",
         {"type": "filters.approximatecoplanar", "knn": 3},
         experimental_cuda=True),
    Case("missing-y-fallback", "pdg/approximatecoplanar-missing-y.csv",
         {"type": "filters.approximatecoplanar", "knn": 3},
         experimental_cuda=True),
    Case("missing-z-fallback", "pdg/approximatecoplanar-missing-z.csv",
         {"type": "filters.approximatecoplanar", "knn": 3},
         experimental_cuda=True),
    Case("solver-failure", "pdg/approximatecoplanar-solver-failure.csv",
         {"type": "filters.approximatecoplanar", "knn": 3}),
    Case("empty", "las/no-points.las",
         {"type": "filters.approximatecoplanar"}),
    Case("small-knn-fallback", "las/simple.las",
         {"type": "filters.approximatecoplanar", "knn": 2}, False),
    Case("large-knn-fallback", "las/simple.las",
         {"type": "filters.approximatecoplanar", "knn": 65}, False),
    Case("negative-knn-fallback", "pdg/neighborhood-points.csv",
         {"type": "filters.approximatecoplanar", "knn": -1}, False),
    Case("where-fallback", "las/simple.las", {
        "type": "filters.approximatecoplanar",
        "where": "Classification == 2", "where_merge": "auto",
    }, False),
    Case("where-merge-without-where-fallback", "las/simple.las", {
        "type": "filters.approximatecoplanar", "where_merge": True,
    }, False),
    Case("string-threshold-fallback", "las/simple.las", {
        "type": "filters.approximatecoplanar", "thresh1": "25",
    }, False),
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
        prefix="pdg-approximatecoplanar-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing approximatecoplanar fixture: {fixture}",
                      file=sys.stderr)
                return 2
            is_csv = case.fixture.endswith(".csv")
            input_name = "input.csv" if is_csv else "input.las"
            reader = "readers.text" if is_csv else "readers.las"
            pipeline = generated / f"{case.name}.json"
            stages: list[dict[str, object]] = [
                {"type": reader, "filename": input_name},
                *case.prefix,
                case.stage,
                {"type": "writers.las", "filename": "out.las",
                 "extra_dims": "all"},
            ]
            pipeline.write_text(
                json.dumps({"pipeline": stages}, indent=2,
                           sort_keys=True) + "\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            for name in ENGINE_ENVIRONMENT_VARIABLES:
                environment.pop(name, None)
            if case.hybrid:
                environment.update({
                    "PDG_DISABLE_NATIVE": "1",
                    "PDG_REQUIRE_HYBRID": "1",
                })
                if case.experimental_cuda:
                    environment["PDG_EXPERIMENTAL_CUDA_HYBRID"] = "1"
                    environment[
                        "PDG_REQUIRE_APPROXIMATECOPLANAR_HOST_FALLBACK"
                    ] = "1"
                else:
                    environment["PDG_DISABLE_CUDA_HYBRID"] = "1"

            command = [
                sys.executable, str(args.differential.resolve()),
                "--oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", f"approximatecoplanar-matrix-{case.name}",
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--seed-file", f"{input_name}={fixture}",
                "--seed-file", f"pipeline.json={pipeline.resolve()}",
            ]
            for preload in args.oracle_preload:
                command.extend(("--oracle-preload",
                                str(preload.resolve())))
            for preload in args.candidate_preload:
                command.extend(("--candidate-preload",
                                str(preload.resolve())))
            command.append("--")
            command.append("pipeline")
            command.append("pipeline.json")
            if case.verbose:
                command.append("--verbose=info")
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode:
                return completed.returncode

    print(f"exact approximatecoplanar process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
