#!/usr/bin/env python3
"""Exact process matrix for filters.covariancefeatures on shared kNN."""

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
    Case("default", "las/simple.las", {"type": "filters.covariancefeatures"}),
    Case("raw", "las/simple.las",
         {"type": "filters.covariancefeatures", "mode": "raw"}),
    Case("normalized", "las/simple.las",
         {"type": "filters.covariancefeatures", "mode": "normalized"}),
    Case("knn-twelve", "las/simple.las",
         {"type": "filters.covariancefeatures", "knn": 12}),
    Case("extended-features", "las/simple.las", {
        "type": "filters.covariancefeatures",
        "feature_set": (
            "Dimensionality,Omnivariance,Anisotropy,Eigenentropy,"
            "EigenvalueSum,SurfaceVariation,DemantkeVerticality"
        ),
    }),
    Case("feature-array", "las/simple.las", {
        "type": "filters.covariancefeatures",
        "feature_set": ["linearity", "planarity", "scattering", "verticality"],
    }),
    Case("explicit-ignored-options", "las/simple.las", {
        "type": "filters.covariancefeatures", "threads": 1, "stride": 1,
        "min_k": 17, "optimized": False,
    }),
    Case("ties-and-duplicates", "pdg/neighborhood-points.csv", {
        "type": "filters.covariancefeatures", "knn": 2,
        "feature_set": "dimensionality,anisotropy",
    }),
    Case("empty", "las/no-points.las",
         {"type": "filters.covariancefeatures"}),
    Case("radius-fallback", "pdg/neighborhood-points.csv", {
        "type": "filters.covariancefeatures", "radius": 1.1, "min_k": 3,
    }, hybrid=False),
    Case("stride-fallback", "las/simple.las", {
        "type": "filters.covariancefeatures", "knn": 8, "stride": 2,
    }, hybrid=False),
    Case("threads-fallback", "las/simple.las", {
        "type": "filters.covariancefeatures", "threads": 2,
    }, hybrid=False),
    Case("optimized-failure", "las/simple.las", {
        "type": "filters.covariancefeatures", "optimized": True,
    }, hybrid=False),
    Case("all-features-failure", "las/simple.las", {
        "type": "filters.covariancefeatures", "feature_set": "all",
    }, hybrid=False),
    Case("where-fallback", "las/simple.las", {
        "type": "filters.covariancefeatures", "where": "Classification == 2",
    }, hybrid=False),
    Case("large-knn-fallback", "las/simple.las", {
        "type": "filters.covariancefeatures", "knn": 64,
    }, hybrid=False),
    Case("small-knn-fallback", "las/simple.las", {
        "type": "filters.covariancefeatures", "knn": 1,
    }, hybrid=False),
    Case("invalid-mode-fallback", "las/simple.las", {
        "type": "filters.covariancefeatures", "mode": "cube",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pdg-covariancefeatures-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = (args.fixture_root / case.fixture).resolve()
            if not fixture.is_file():
                print(f"missing covariancefeatures fixture: {fixture}",
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
            if case.hybrid:
                environment.update({
                    "PDG_DISABLE_NATIVE": "1",
                    "PDG_REQUIRE_HYBRID": "1",
                    "PDG_DISABLE_CUDA_HYBRID": "1",
                })
            else:
                for name in (
                    "PDG_DISABLE_NATIVE", "PDG_REQUIRE_HYBRID",
                    "PDG_DISABLE_CUDA_HYBRID",
                ):
                    environment.pop(name, None)

            command = [
                sys.executable, str(args.differential.resolve()),
                "--oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", f"covariancefeatures-matrix-{case.name}",
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

    print(f"exact covariancefeatures process matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
