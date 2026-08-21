#!/usr/bin/env python3
"""Exact bounded process matrix for filters.smrf."""

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
    forced_workers: int = 5


CASES = (
    Case("default-returns", "returns", {"type": "filters.smrf"}),
    Case("returns-first", "returns", {
        "type": "filters.smrf", "returns": "first",
    }),
    Case("returns-all", "returns", {
        "type": "filters.smrf",
        "returns": ["first", "intermediate", "last", "only"],
    }),
    Case("only-ground-custom-classes", "returns", {
        "type": "filters.smrf", "returns": [],
        "only_ground": True, "ground_class": 9, "other_class": 9,
    }),
    Case("custom-morphology", "returns", {
        "type": "filters.smrf", "returns": [], "cell": 1.0,
        "slope": 0.2, "scalar": 1.2, "threshold": 0.45,
        "window": 7.0, "cut": 5.0,
    }),
    # B0216/D0221: keep the measured raster-cell ladder in the committed host
    # differential. The sparse deterministic terrain exercises void filling;
    # the CUDA lane is intentionally not registered while its eighth-neighbor
    # cutoff-tie rule remains unqualified.
    Case("void-fill-cell-2", "returns", {
        "type": "filters.smrf", "returns": [], "cell": 2.0,
    }),
    Case("void-fill-cell-4", "returns", {
        "type": "filters.smrf", "returns": [], "cell": 4.0,
    }),
    Case("void-fill-cell-8", "returns", {
        "type": "filters.smrf", "returns": [], "cell": 8.0,
    # This fixture's morphology raster has four cells, so the shared worker
    # policy correctly caps the forced count at four for that pass.
    }, forced_workers=4),
    Case("missing-return-dimensions", "no-returns", {
        "type": "filters.smrf",
    }),
    Case("all-zero-returns", "zero-returns", {
        "type": "filters.smrf",
    }),
    Case("ignore-range-fallback", "returns", {
        "type": "filters.smrf", "returns": [],
        "ignore": "Classification[7:7]",
    }, hybrid=False),
    Case("classbits-fallback", "returns", {
        "type": "filters.smrf", "returns": [],
        "classbits": "synthetic",
    }, hybrid=False),
    Case("where-fallback", "returns", {
        "type": "filters.smrf", "returns": [],
        "where": "Classification != 7",
    }, hybrid=False),
    Case("partial-zero-error", "partial-zero", {
        "type": "filters.smrf",
    }, hybrid=False),
    Case("invalid-returns-error", "returns", {
        "type": "filters.smrf", "returns": "bogus",
    }, hybrid=False),
    Case("equal-classes-error", "returns", {
        "type": "filters.smrf", "ground_class": 4, "other_class": 4,
    }, hybrid=False),
    Case("missing-ignore-dimension-error", "returns", {
        "type": "filters.smrf", "ignore": "NotHere[0:1]",
    }, hybrid=False),
    Case("empty-error", "empty", {"type": "filters.smrf"}, hybrid=False),
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
    parser.add_argument(
        "--pinned-oracle", type=Path,
        help="pinned upstream PDAL; runs the large-lattice cases against it "
             "with the candidate delegating to the forked sibling (D0260)",
    )
    return parser.parse_args()


# B0264/D0264: the exact host SMRF now runs its per-point passes, void fill
# queries, and diamond morphology passes on fixed-chunk host workers. The
# 16 x 16 contract fixture never reaches the worker thresholds, so (1) every
# tiny case is repeated with the test-only forced worker count so the
# chunked point/void-fill code and the createZImin merge are compared with the
# serial oracle, and (2) large lattices are compared against the pinned
# oracle so the pooled morphology passes (>= 32,768 cells) and the natural
# worker counts are exercised.
LATTICE_SIDE = 240
PINNED_CASES = (
    Case("lattice-default", "lattice", {"type": "filters.smrf"}),
    Case("lattice-custom-morphology-cut", "lattice", {
        "type": "filters.smrf", "returns": [], "cell": 1.0,
        "slope": 0.2, "scalar": 1.2, "threshold": 0.45,
        "window": 7.0, "cut": 5.0,
    }),
    Case("lattice-cell-half", "lattice", {
        "type": "filters.smrf", "returns": [], "cell": 0.5,
    }),
)


def terrain_rows() -> list[tuple[float, float, float, int, int, int]]:
    rows: list[tuple[float, float, float, int, int, int]] = []
    point = 0
    for x in range(16):
        for y in range(16):
            if (x * 3 + y * 5) % 17 == 0 and (x, y) not in ((0, 0), (15, 15)):
                continue
            z = 0.05 * x + 0.03 * y
            if 6 <= x <= 9 and 6 <= y <= 9:
                z += 4.0
            if (x, y) == (2, 2):
                z -= 2.0
            kind = point % 4
            if kind == 0:
                return_number, number_of_returns = 1, 1
            elif kind == 1:
                return_number, number_of_returns = 1, 2
            elif kind == 2:
                return_number, number_of_returns = 2, 2
            else:
                return_number, number_of_returns = 2, 3
            classification = 34 if point % 31 == 0 else (7 if point % 19 == 0 else 5)
            rows.append((float(x), float(y), z, classification,
                         return_number, number_of_returns))
            point += 1
    return rows


def lattice_rows() -> list[tuple[float, float, float, int, int, int]]:
    rows: list[tuple[float, float, float, int, int, int]] = []
    point = 0
    for x in range(LATTICE_SIDE):
        for y in range(LATTICE_SIDE):
            if (x * 3 + y * 5) % 17 == 0 and (x, y) not in ((0, 0), (239, 239)):
                continue
            z = 0.05 * x + 0.03 * y
            if 60 <= x <= 99 and 60 <= y <= 99:
                z += 4.0
            if 150 <= x <= 155 and 20 <= y <= 200:
                z += 12.0
            if (x, y) == (2, 2) or (x, y) == (200, 30):
                z -= 2.0
            kind = point % 4
            if kind == 0:
                return_number, number_of_returns = 1, 1
            elif kind == 1:
                return_number, number_of_returns = 1, 2
            elif kind == 2:
                return_number, number_of_returns = 2, 2
            else:
                return_number, number_of_returns = 2, 3
            classification = 34 if point % 31 == 0 else (7 if point % 19 == 0 else 5)
            rows.append((float(x), float(y), z, classification,
                         return_number, number_of_returns))
            point += 1
    return rows


def write_fixture(path: Path, kind: str) -> None:
    rows = lattice_rows() if kind == "lattice" else terrain_rows()
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
    with tempfile.TemporaryDirectory(prefix="pdg-smrf-matrix-",
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
                "--case", f"smrf-matrix-{case.name}",
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
            if args.require_cuda:
                continue
            forced = command[:-3] + [
                "--candidate-env",
                "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS="
                f"{case.forced_workers}",
                "--candidate-env",
                "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS="
                f"{case.forced_workers}",
            ] + command[-3:]
            forced[forced.index("--case") + 1] = (
                f"smrf-matrix-{case.name}-forced-workers")
            completed = subprocess.run(forced, env=environment, check=False)
            if completed.returncode:
                return completed.returncode
            executed += 1

        pinned = 0
        if args.pinned_oracle is not None and not args.require_cuda:
            for case in PINNED_CASES:
                fixture = generated / f"{case.fixture}.csv"
                write_fixture(fixture, case.fixture)
                pipeline = generated / f"{case.name}.json"
                pipeline.write_text(
                    json.dumps({"pipeline": [
                        {"type": "readers.text", "filename": "input.csv"},
                        case.stage,
                        {"type": "writers.las", "filename": "out.las",
                         "extra_dims": "all"},
                    ]}, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
                environment = os.environ.copy()
                for name in (
                    "PDG_DISABLE_NATIVE", "PDG_REQUIRE_HYBRID",
                    "PDG_DISABLE_CUDA_HYBRID", "PDG_REQUIRE_CUDA_HYBRID",
                ):
                    environment.pop(name, None)
                command = [
                    sys.executable, str(args.differential.resolve()),
                    "--oracle", str(args.pinned_oracle.resolve()),
                    "--candidate-oracle", str(args.oracle.resolve()),
                    "--candidate", str(args.candidate.resolve()),
                    "--case", f"smrf-matrix-{case.name}-pinned",
                    "--work-dir", str(args.work_dir.resolve()),
                    "--frozen-time-library",
                    str(args.frozen_time_library.resolve()),
                    "--seed-file", f"input.csv={fixture.resolve()}",
                    "--seed-file", f"pipeline.json={pipeline.resolve()}",
                    "--", "pipeline", "pipeline.json",
                ]
                completed = subprocess.run(command, env=environment,
                                           check=False)
                if completed.returncode:
                    return completed.returncode
                pinned += 1

    print(f"exact smrf process matrix: {executed} cases"
          + (f", {pinned} pinned-oracle lattice cases" if pinned else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
