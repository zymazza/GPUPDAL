#!/usr/bin/env python3
"""Bounded complete-process matrix for the r7 DSM raster contract.

The fixture is generated below the CTest build directory from deterministic
text.  Every case runs through the pinned-oracle differential harness, and
successful rasters additionally compare the complete normalized
``gdalinfo -json -checksum`` document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    groups: object = "first,only"
    writer_updates: dict[str, object] = field(default_factory=dict)
    expected_returncode: int = 0
    expected_output_bytes: int | None = None
    repeat: bool = False
    headline_contract: bool = False
    selected_route: bool = False


CASES = (
    Case("first-only-max", expected_output_bytes=692, headline_contract=True),
    Case("selected-first-only-max", expected_output_bytes=924954,
         headline_contract=True, selected_route=True),
    Case("return-policy-drift", groups="last", expected_output_bytes=692),
    Case("writer-policy-drift", writer_updates={"output_type": "min"},
         expected_output_bytes=692),
    Case(
        "malformed-return-group",
        groups="first,sideways",
        expected_returncode=1,
    ),
    Case(
        "malformed-bounds",
        writer_updates={"bounds": "([0,4],[0,])"},
        expected_returncode=1,
    ),
    Case("determinism", expected_output_bytes=692, repeat=True,
         headline_contract=True),
    Case("selected-determinism", expected_output_bytes=924954, repeat=True,
         headline_contract=True, selected_route=True),
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
    pipeline = root / f"{name}.json"
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
            f"unable to build r7 DSM fixture {name}: "
            + completed.stderr.decode("utf-8", errors="replace")[:1024],
            file=sys.stderr,
        )
        raise RuntimeError("fixture generation failed")


def build_fixtures(root: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    source = root / "input.csv"
    source.write_text(
        "X,Y,Z,ReturnNumber,NumberOfReturns\n"
        "0.25,0.25,10,1,1\n"
        "0.25,0.25,20,1,2\n"
        "0.25,0.25,30,2,2\n"
        "1.25,0.25,15,1,3\n"
        "1.25,0.25,40,2,3\n"
        "1.25,0.25,45,3,3\n"
        "2.25,1.25,50,1,1\n"
        "3.25,3.25,-5,1,2\n"
        "3.25,3.25,60,2,2\n",
        encoding="utf-8",
    )
    target = root / "r7-input.las"
    run_fixture_pipeline(root, args, "fixture", {"pipeline": [
        {"type": "readers.text", "filename": "input.csv"},
        {"type": "writers.las", "filename": str(target),
         "minor_version": 4, "dataformat_id": 6,
         "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01},
    ]}, target)

    selected_csv = root / "selected.csv"
    selected_csv.write_text(
        "X,Y,Z,ReturnNumber,NumberOfReturns\n"
        "184500.25,494923.46,10,1,1\n"
        "184500.25,494923.46,20,1,2\n"
        "184500.25,494923.46,30,2,2\n"
        "184501.25,494923.46,15,1,3\n"
        "184501.25,494923.46,40,2,3\n"
        "184501.25,494923.46,45,3,3\n"
        "184502.25,494924.46,50,1,1\n"
        "185998.25,494998.25,-5,1,2\n"
        "185998.25,494998.25,60,2,2\n",
        encoding="utf-8",
    )
    selected_las = root / "r7-selected-source.las"
    run_fixture_pipeline(root, args, "selected-las", {"pipeline": [
        {"type": "readers.text", "filename": "selected.csv"},
        {"type": "writers.las", "filename": str(selected_las),
         "minor_version": 4, "dataformat_id": 6,
         "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01},
    ]}, selected_las)

    selected_laz = root / "r7-selected-input.laz"
    run_fixture_pipeline(root, args, "selected-laz", {"pipeline": [
        {"type": "readers.las", "filename": str(selected_las)},
        {"type": "writers.las", "filename": str(selected_laz),
         "compression": True},
    ]}, selected_laz)
    return target, selected_laz


def normalized_raster_metadata(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        ["gdalinfo", "-json", "-checksum", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"gdalinfo failed for {path}: {completed.stderr[:400]}")
    metadata = json.loads(completed.stdout)
    metadata.pop("description", None)
    metadata.pop("files", None)
    return metadata


def metadata_digest(metadata: dict[str, object]) -> str:
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def check_headline_metadata(
    metadata: dict[str, object], name: str, selected_route: bool,
) -> bool:
    bands = metadata.get("bands")
    coordinate_system = metadata.get("coordinateSystem")
    expected_size = [1500, 77] if selected_route else [5, 5]
    expected_transform = (
        [184500.0, 1.0, 0.0, 495000.21, 0.0, -1.0]
        if selected_route else [0.0, 1.0, 0.0, 5.0, 0.0, -1.0]
    )
    valid = (
        metadata.get("driverShortName") == "GTiff"
        and metadata.get("size") == expected_size
        and metadata.get("geoTransform") == expected_transform
        and isinstance(coordinate_system, dict)
        and 'ID["EPSG",28992]' in str(coordinate_system.get("wkt", ""))
        and isinstance(bands, list)
        and len(bands) == 1
        and bands[0].get("type") == "Float64"
        and bands[0].get("description") == "max"
        and bands[0].get("noDataValue") == -9999.0
        and isinstance(bands[0].get("checksum"), int)
    )
    if not valid:
        print(f"{name}: incomplete or drifted headline raster metadata", file=sys.stderr)
        print(json.dumps(metadata, indent=2, sort_keys=True)[:4000], file=sys.stderr)
    return valid


def run_case(
    args: argparse.Namespace,
    root: Path,
    case: Case,
    source_las: Path,
    source_laz: Path,
    suffix: str,
) -> tuple[Path, str] | None:
    name = f"r7-dsm-{case.name}{suffix}"
    writer: dict[str, object] = {
        "type": "writers.gdal",
        "filename": "out.tif",
        "resolution": 1.0,
        "output_type": "max",
        "dimension": "Z",
        "binmode": True,
        "data_type": "float64",
        "nodata": -9999.0,
        "bounds": (
            "([184500,185999.99],[494923.21,494999.99])"
            if case.selected_route else "([0,4],[0,4])"
        ),
    }
    writer.update(case.writer_updates)
    pipeline = root / f"{name}.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.las",
         "filename": "input.laz" if case.selected_route else "input.las",
         "override_srs": "EPSG:28992"},
        {"type": "filters.returns", "groups": case.groups},
        writer,
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
        "--seed-file", (
            f"input.laz={source_laz}" if case.selected_route
            else f"input.las={source_las}"
        ),
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

    output = report["oracle_artifacts"].get("out.tif")
    if case.expected_returncode:
        if output is not None:
            print(f"{name}: refusal unexpectedly published out.tif", file=sys.stderr)
            return None
        return report_path, ""
    if output is None or output["bytes"] <= 0:
        print(f"{name}: positive case did not publish a raster", file=sys.stderr)
        return None
    if (case.expected_output_bytes is not None and
            output["bytes"] != case.expected_output_bytes):
        print(
            f"{name}: expected {case.expected_output_bytes} bytes, observed "
            f"{output['bytes']}",
            file=sys.stderr,
        )
        return None

    case_root = Path(report["case_root"])
    oracle_metadata = normalized_raster_metadata(case_root / "oracle" / "out.tif")
    candidate_metadata = normalized_raster_metadata(
        case_root / "candidate" / "out.tif")
    if oracle_metadata != candidate_metadata:
        print(f"{name}: normalized raster metadata differs", file=sys.stderr)
        return None
    if case.headline_contract and not check_headline_metadata(
            oracle_metadata, name, case.selected_route):
        return None
    return report_path, metadata_digest(candidate_metadata)


def main() -> int:
    args = parse_args()
    for path in (args.oracle, args.candidate, args.differential,
                 args.frozen_time_library):
        if not path.is_file():
            print(f"required file missing: {path}", file=sys.stderr)
            return 2
    args.work_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
            prefix="pdg-r7-dsm-matrix-", dir=args.work_dir) as temporary:
        root = Path(temporary)
        try:
            source_las, source_laz = build_fixtures(root, args)
        except RuntimeError:
            return 1
        for case in CASES:
            if case.repeat:
                first = run_case(
                    args, root, case, source_las, source_laz, "-run-1")
                second = run_case(
                    args, root, case, source_las, source_laz, "-run-2")
                if first is None or second is None:
                    return 1
                first_report = json.loads(first[0].read_text(encoding="utf-8"))
                second_report = json.loads(second[0].read_text(encoding="utf-8"))
                if (first_report["candidate_artifacts"] !=
                        second_report["candidate_artifacts"] or
                        first[1] != second[1]):
                    print("r7 DSM candidate is not deterministic", file=sys.stderr)
                    return 1
            elif run_case(
                    args, root, case, source_las, source_laz, "") is None:
                return 1

    print(f"exact r7 DSM matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
