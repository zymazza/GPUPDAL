#!/usr/bin/env python3
"""Bounded complete-process matrix for r9 polygon-crop semantics.

The fixture is generated in the CTest work directory from deterministic text.
Every case executes through the pinned-oracle differential harness so that
stdout, stderr, exit status, and artifact bytes are compared.
"""

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
    polygon: str
    reader_srs: str | None = None
    geometry_srs: str | None = None
    repeat: bool = False
    expected_returncode: int = 0
    expected_output_bytes: int | None = None


@dataclass(frozen=True)
class Fixture:
    points: tuple[tuple[float, float, float], ...]


CASES = (
    Case(
        "polygon-simple",
        "POLYGON((-4 -4,4 -4,4 4,-4 4,-4 -4))",
        expected_output_bytes=555,
    ),
    Case(
        "polygon-multipolygon-hole-disjoint",
        (
            "MULTIPOLYGON (((-4.4 -4.4,4.4 -4.4,4.4 4.4,-4.4 4.4,-4.4 -4.4),"
            "(-1 -1,1 -1,1 1,-1 1,-1 -1)),"
            "((-8.0 -1.0,-6.0 -1.0,-6.0 1.0,-8.0 1.0,-8.0 -1.0)))"
        ),
        expected_output_bytes=699,
    ),
    Case(
        "polygon-boundary-holes",
        "POLYGON((-3 -3,3 -3,3 3,-3 3,-3 -3))",
        expected_output_bytes=519,
    ),
    Case(
        "polygon-crs-reprojection",
        "POLYGON((-0.00003 -0.00003,-0.00003 0.00003,0.00003 0.00003,"
        "0.00003 -0.00003,-0.00003 -0.00003))",
        reader_srs="EPSG:3857",
        geometry_srs="EPSG:4326",
        expected_output_bytes=1991,
    ),
    Case(
        "polygon-geometry-srs-refusal",
        "POLYGON((-4 -4,4 -4,4 4,-4 4,-4 -4))",
        geometry_srs="EPSG:4326",
        expected_returncode=1,
        # PDAL opens the writer before the transform error and leaves an exact
        # zero-byte artifact. Preserve that observable instead of treating the
        # failure as publication-free.
        expected_output_bytes=0,
    ),
    Case(
        "polygon-malformed-wkt",
        "POLYGON((-4 -4,4 -4,4 4)",
        expected_returncode=1,
    ),
    Case(
        "polygon-malformed-srs",
        "POLYGON((-4 -4,4 -4,4 4,-4 4,-4 -4))",
        geometry_srs="EPSG:not_a_number",
        expected_returncode=1,
    ),
    Case(
        "polygon-determinism",
        "POLYGON((-4 -4,4 -4,4 4,-4 4,-4 -4))",
        repeat=True,
        expected_output_bytes=555,
    ),
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


def fixture_text() -> Fixture:
    return Fixture(points=(
        (-4.4, 0.0, 1.0),
        (-2.2, 0.0, 2.0),
        (0.0, 0.0, 3.0),
        (1.0, 0.0, 4.0),
        (2.2, 2.2, 5.0),
        (4.4, 0.0, 6.0),
        (3.3395847237982075, 0.0, 7.0),
        (-7.0, -1.0, 8.0),
        (-6.5, -1.0, 9.0),
        (4.4, 4.4, 10.0),
        (5.0, 5.0, 11.0),
    ))


def build_fixture(root: Path, oracle: Path, freeze_epoch: int,
                  frozen_time_library: Path | None) -> Path:
    source_text = root / "input.csv"
    rows = ["X,Y,Z\n"]
    for x, y, z in fixture_text().points:
        rows.append(f"{x},{y},{z}\n")
    source_text.write_text("".join(rows), encoding="utf-8")

    target_las = root / "r9-input.las"
    pipeline = root / "r9-input.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.text", "filename": "input.csv"},
        {"type": "writers.las", "filename": str(target_las),
         "minor_version": 4, "scale_x": 0.01, "scale_y": 0.01,
         "scale_z": 0.01},
    ]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    environment = dict(os.environ)
    environment.update({"LC_ALL": "C", "TZ": "UTC",
                        "PDAL_TEST_FROZEN_EPOCH": str(freeze_epoch)})
    preloads: list[str] = []
    if asan_runtime := environment.get("PDG_DIFFERENTIAL_ASAN_PRELOAD"):
        # Fixture generation invokes the sanitizer-built oracle directly,
        # before differential.py can order its preloads. ASan must precede
        # the frozen-time shim here for the same reason as in the harness.
        preloads.append(asan_runtime)
    if frozen_time_library:
        preloads.append(str(frozen_time_library.resolve()))
    if inherited := environment.get("LD_PRELOAD"):
        preloads.append(inherited)
    if preloads:
        environment["LD_PRELOAD"] = ":".join(preloads)

    built = subprocess.run(
        [str(oracle), "pipeline", str(pipeline)],
        env=environment,
        cwd=root,
        capture_output=True,
        check=False,
    )
    if built.returncode:
        print(
            "unable to build r9 polygon fixture: "
            + built.stderr.decode("utf-8", errors="replace")[:1024],
            file=sys.stderr,
        )
        raise RuntimeError("fixture generation failed")

    if not target_las.is_file():
        raise RuntimeError("fixture generation did not emit LAS output")

    return target_las


def compare_candidate_determinism(
    first_path: Path,
    second_path: Path,
    name: str,
    ) -> bool:
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    if first["candidate_artifacts"] != second["candidate_artifacts"]:
        print(
            f"candidate artifact non-determinism for {name}: "
            f"{first['candidate_artifacts']} != {second['candidate_artifacts']}",
            file=sys.stderr,
        )
        return False
    return True


def run_case(
    args: argparse.Namespace,
    root: Path,
    case: Case,
    source_las: Path,
    suffix: str,
) -> Path | None:
    name = f"r9-polygon-{case.name}{suffix}"
    stages = [{"type": "readers.las", "filename": "input.las"}]
    if case.reader_srs is not None:
        stages[0]["override_srs"] = case.reader_srs

    crop = {"type": "filters.crop", "polygon": case.polygon}
    if case.geometry_srs is not None:
        crop["a_srs"] = case.geometry_srs
    stages.append(crop)
    stages.append({"type": "writers.las", "filename": "out.las"})

    pipeline = root / f"{name}.json"
    pipeline.write_text(json.dumps({"pipeline": stages}, indent=2, sort_keys=True) +
                        "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(args.differential.resolve()),
            "--oracle",
            str(args.oracle.resolve()),
            "--candidate",
            str(args.candidate.resolve()),
            "--case",
            name,
            "--work-dir",
            str(args.work_dir.resolve()),
            "--frozen-time-library",
            str(args.frozen_time_library.resolve()),
            "--seed-file",
            f"input.las={source_las}",
            "--seed-file",
            f"pipeline.json={pipeline}",
            "--",
            "pipeline",
            "pipeline.json",
        ],
        check=False,
    )
    if completed.returncode:
        return None

    report_path = (args.work_dir / "reports" / f"{name}.json").resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    observed_returncode = report["oracle_run"]["returncode"]
    if observed_returncode != case.expected_returncode:
        print(
            f"{name}: expected oracle status {case.expected_returncode}, "
            f"observed {observed_returncode}",
            file=sys.stderr,
        )
        return None
    output = report["oracle_artifacts"].get("out.las")
    if case.expected_output_bytes is None:
        if output is not None:
            print(f"{name}: refusal unexpectedly published out.las", file=sys.stderr)
            return None
    elif output is None or output["bytes"] != case.expected_output_bytes:
        observed = None if output is None else output["bytes"]
        print(
            f"{name}: expected {case.expected_output_bytes}-byte positive "
            f"artifact, observed {observed}",
            file=sys.stderr,
        )
        return None

    return report_path


def main() -> int:
    args = parse_args()
    for path in (args.oracle, args.candidate, args.differential):
        if not path.is_file():
            print(f"required executable missing: {path}", file=sys.stderr)
            return 2
    if not args.frozen_time_library.is_file():
        print(f"frozen-time library missing: {args.frozen_time_library}", file=sys.stderr)
        return 2

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdg-r9-polygon-matrix-",
                                     dir=args.work_dir) as temporary:
        work_root = Path(temporary)
        try:
            source_las = build_fixture(
                work_root,
                args.oracle,
                args.freeze_epoch,
                args.frozen_time_library,
            )
        except RuntimeError:
            return 1

        for case in CASES:
            if case.repeat:
                first = run_case(args, work_root, case, source_las, "-run-1")
                second = run_case(args, work_root, case, source_las, "-run-2")
                if first is None or second is None:
                    return 1
                if not compare_candidate_determinism(first, second,
                                                   f"r9-polygon-{case.name}"):
                    return 1
            else:
                if run_case(args, work_root, case, source_las, "") is None:
                    return 1

    print(f"exact r9 polygon matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
