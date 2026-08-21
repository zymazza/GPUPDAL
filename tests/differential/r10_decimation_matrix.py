#!/usr/bin/env python3
"""Bounded complete-process matrix for the r10 decimation contract."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    source: str = "dense"
    cell: object = 2.5
    compression: object = "true"
    expected_returncode: int = 0
    expected_points: int | None = 4
    repeat: bool = False


CASES = (
    Case("selected-dense"),
    Case("selected-sparse", source="sparse"),
    Case("selected-empty", source="empty", expected_points=0),
    Case("selected-malformed-input", source="malformed",
         expected_returncode=1, expected_points=None),
    Case("cell-policy-drift", cell=1.0, expected_points=8),
    # Upstream accepts numeric zero and puts this fixture in one effective
    # bucket. Freeze that unusual oracle behavior rather than inventing a
    # compatibility-mode validation error.
    Case("zero-cell-oracle-behavior", cell=0.0, expected_points=1),
    Case("invalid-cell-refusal", cell="invalid",
         expected_returncode=1, expected_points=None),
    Case("writer-policy-drift", compression=True),
    Case("selected-determinism", repeat=True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--freeze-epoch", type=int, default=1_704_067_200)
    return parser.parse_args()


def oracle_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "LC_ALL": "C",
        "TZ": "UTC",
        "PDAL_TEST_FROZEN_EPOCH": str(args.freeze_epoch),
    })
    preloads: list[str] = []
    if asan_runtime := environment.get("PDG_DIFFERENTIAL_ASAN_PRELOAD"):
        preloads.append(asan_runtime)
    preloads.append(str(args.frozen_time_library.resolve()))
    if inherited := environment.get("LD_PRELOAD"):
        preloads.append(inherited)
    environment["LD_PRELOAD"] = ":".join(preloads)
    return environment


def run_fixture_pipeline(
    root: Path,
    args: argparse.Namespace,
    name: str,
    document: dict[str, object],
    target: Path,
) -> None:
    pipeline = root / f"fixture-{name}.json"
    pipeline.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(args.oracle.resolve()), "pipeline", str(pipeline.resolve())],
        cwd=root,
        env=oracle_environment(args),
        capture_output=True,
        check=False,
    )
    if completed.returncode or not target.is_file():
        print(
            f"unable to build r10 fixture {name}: "
            + completed.stderr.decode("utf-8", errors="replace")[:1024],
            file=sys.stderr,
        )
        raise RuntimeError("fixture generation failed")


def build_fixtures(root: Path, args: argparse.Namespace) -> dict[str, Path]:
    points = {
        "dense": (
            "X,Y,Z,Intensity\n"
            "0,0,0,10\n"
            "1,1,1,11\n"
            "2,2,2,12\n"
            "3,0,0,13\n"
            "4,0,0,14\n"
            "-1,-1,-1,15\n"
            "-2,-2,-2,16\n"
            "10,10,10,17\n"
        ),
        "sparse": (
            "X,Y,Z,Intensity\n"
            "100,100,100,20\n"
            "103,100,100,21\n"
            "100,103,100,22\n"
            "100,100,103,23\n"
        ),
    }
    fixtures: dict[str, Path] = {}
    for name, contents in points.items():
        source = root / f"{name}.csv"
        source.write_text(contents, encoding="utf-8")
        target = root / f"r10-{name}.laz"
        run_fixture_pipeline(root, args, name, {"pipeline": [
            {"type": "readers.text", "filename": source.name},
            {"type": "writers.las", "filename": str(target),
             "minor_version": 4, "dataformat_id": 6,
             "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01,
             "compression": True},
        ]}, target)
        fixtures[name] = target

    empty = root / "r10-empty.laz"
    run_fixture_pipeline(root, args, "empty", {"pipeline": [
        {"type": "readers.faux", "bounds": "([0,1],[0,1],[0,1])",
         "count": 0, "mode": "constant"},
        {"type": "writers.las", "filename": str(empty),
         "minor_version": 4, "dataformat_id": 6,
         "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01,
         "compression": True},
    ]}, empty)
    fixtures["empty"] = empty

    malformed = root / "r10-malformed.laz"
    malformed.write_bytes(b"not a LAS or LAZ file\n")
    fixtures["malformed"] = malformed
    return fixtures


def las_point_count(path: Path) -> int | None:
    try:
        header = path.read_bytes()[:375]
    except OSError:
        return None
    if len(header) < 227 or header[:4] != b"LASF" or header[24] != 1:
        return None
    if header[25] >= 4:
        if len(header) < 255:
            return None
        return struct.unpack_from("<Q", header, 247)[0]
    return struct.unpack_from("<I", header, 107)[0]


def run_case(
    args: argparse.Namespace,
    root: Path,
    case: Case,
    fixtures: dict[str, Path],
    suffix: str,
) -> Path | None:
    name = f"r10-decimation-{case.name}{suffix}"
    pipeline = root / f"{name}.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": "input.laz"},
        {"type": "filters.voxelcentroidnearestneighbor", "cell": case.cell},
        {"type": "writers.las", "filename": "output.laz",
         "compression": case.compression},
    ]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completed = subprocess.run([
        sys.executable,
        str(args.differential.resolve()),
        "--oracle", str(args.oracle.resolve()),
        "--candidate", str(args.candidate.resolve()),
        "--case", name,
        "--work-dir", str(args.work_dir.resolve()),
        "--frozen-time-library", str(args.frozen_time_library.resolve()),
        "--keep-success",
        "--seed-file", f"input.laz={fixtures[case.source]}",
        "--seed-file", f"pipeline.json={pipeline}",
        "--", "pipeline", "pipeline.json",
    ], check=False)
    if completed.returncode:
        return None

    report_path = args.work_dir / "reports" / f"{name}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["oracle_run"]["returncode"] != case.expected_returncode:
        print(
            f"{name}: expected status {case.expected_returncode}, observed "
            f"{report['oracle_run']['returncode']}",
            file=sys.stderr,
        )
        return None
    output = report["oracle_artifacts"].get("output.laz")
    if case.expected_returncode:
        if output is not None:
            print(f"{name}: refusal unexpectedly published output.laz",
                  file=sys.stderr)
            return None
        return report_path
    if output is None or output["bytes"] <= 0:
        print(f"{name}: positive case did not publish output.laz",
              file=sys.stderr)
        return None
    case_root = Path(report["case_root"])
    observed_points = las_point_count(case_root / "oracle" / "output.laz")
    if observed_points != case.expected_points:
        print(
            f"{name}: expected {case.expected_points} points, observed "
            f"{observed_points}",
            file=sys.stderr,
        )
        return None
    if (las_point_count(case_root / "candidate" / "output.laz") !=
            observed_points):
        print(f"{name}: candidate point count differs", file=sys.stderr)
        return None
    return report_path


def main() -> int:
    args = parse_args()
    for path in (args.oracle, args.candidate, args.differential,
                 args.frozen_time_library):
        if not path.is_file():
            print(f"required file missing: {path}", file=sys.stderr)
            return 2
    args.work_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
            prefix="pdg-r10-decimation-matrix-",
            dir=args.work_dir) as temporary:
        root = Path(temporary)
        try:
            fixtures = build_fixtures(root, args)
        except RuntimeError:
            return 1
        for case in CASES:
            if case.repeat:
                first = run_case(args, root, case, fixtures, "-run-1")
                second = run_case(args, root, case, fixtures, "-run-2")
                if first is None or second is None:
                    return 1
                left = json.loads(first.read_text(encoding="utf-8"))
                right = json.loads(second.read_text(encoding="utf-8"))
                if left["candidate_artifacts"] != right["candidate_artifacts"]:
                    print("r10 decimation candidate is not deterministic",
                          file=sys.stderr)
                    return 1
            elif run_case(args, root, case, fixtures, "") is None:
                return 1

    print(f"exact r10 decimation matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
