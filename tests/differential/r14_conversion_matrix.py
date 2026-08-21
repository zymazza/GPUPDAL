#!/usr/bin/env python3
"""Bounded complete-process matrix for r14 conversion/compression."""

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
    source_name: str
    stages: tuple[dict[str, object], ...]
    output_name: str
    expected_returncode: int = 0
    expected_points: int | None = 12
    expected_compressed: bool | None = None
    expected_copc: bool = False
    repeat: bool = False


def reader(kind: str, filename: str) -> dict[str, object]:
    stage: dict[str, object] = {"type": kind, "filename": filename}
    if kind == "readers.copc":
        stage["requests"] = 1
    return stage


def las_writer(filename: str, compressed: bool) -> dict[str, object]:
    stage: dict[str, object] = {"type": "writers.las", "filename": filename}
    if compressed:
        stage["compression"] = "true"
    return stage


def copc_writer(filename: str) -> dict[str, object]:
    return {
        "type": "writers.copc",
        "filename": filename,
        "threads": 1,
        "fixed_seed": True,
    }


CASES = (
    Case("las-to-laz", "source.las",
         (reader("readers.las", "input.las"),
          las_writer("output.laz", True)),
         "output.laz", expected_compressed=True, repeat=True),
    Case("las-to-laz-automatic-1m", "source-1m.las",
         (reader("readers.las", "input.las"),
          las_writer("output.laz", True)),
         "output.laz", expected_points=1_000_000,
         expected_compressed=True, repeat=True),
    Case("laz-to-las", "source.laz",
         (reader("readers.las", "input.laz"),
          las_writer("output.las", False)),
         "output.las", expected_compressed=False, repeat=True),
    Case("laz-recompress", "source.laz",
         (reader("readers.las", "input.laz"),
          las_writer("output.laz", True)),
         "output.laz", expected_compressed=True, repeat=True),
    Case("las-to-copc", "source.las",
         (reader("readers.las", "input.las"),
          copc_writer("output.copc.laz")),
         "output.copc.laz", expected_compressed=True, expected_copc=True,
         repeat=True),
    Case("laz-to-copc", "source.laz",
         (reader("readers.las", "input.laz"),
          copc_writer("output.copc.laz")),
         "output.copc.laz", expected_compressed=True, expected_copc=True,
         repeat=True),
    Case("copc-to-las", "source.copc.laz",
         (reader("readers.copc", "input.copc.laz"),
          las_writer("output.las", False)),
         "output.las", expected_compressed=False, repeat=True),
    Case("copc-to-laz", "source.copc.laz",
         (reader("readers.copc", "input.copc.laz"),
          las_writer("output.laz", True)),
         "output.laz", expected_compressed=True, repeat=True),
    Case("truncated-las-refusal", "truncated.las",
         (reader("readers.las", "input.las"),
          las_writer("output.laz", True)),
         "output.laz", expected_returncode=1, expected_points=None),
    Case("malformed-laz-refusal", "malformed.laz",
         (reader("readers.las", "input.laz"),
          las_writer("output.las", False)),
         "output.las", expected_returncode=1, expected_points=None),
    Case("malformed-copc-refusal", "malformed.copc.laz",
         (reader("readers.copc", "input.copc.laz"),
          las_writer("output.las", False)),
         "output.las", expected_returncode=1, expected_points=None),
    # The .laz extension forces compression in the pinned writer even when
    # the explicit compression spelling is not a recognized boolean token.
    # Freeze that accepted oracle behavior instead of inventing validation.
    Case("compression-string-oracle-behavior", "source.las",
         (reader("readers.las", "input.las"),
          {"type": "writers.las", "filename": "output.laz",
           "compression": "invalid"}),
         "output.laz", expected_compressed=True),
    Case("unsupported-writer-refusal", "source.las",
         (reader("readers.las", "input.las"),
          {"type": "writers.not_a_driver", "filename": "output.bin"}),
         "output.bin", expected_returncode=1, expected_points=None),
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


def frozen_environment(args: argparse.Namespace) -> dict[str, str]:
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


def run_fixture_pipeline(args: argparse.Namespace, root: Path, name: str,
                         stages: list[dict[str, object]], target: Path) -> None:
    pipeline = root / f"fixture-{name}.json"
    pipeline.write_text(
        json.dumps({"pipeline": stages}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(args.oracle.resolve()), "pipeline", str(pipeline.resolve())],
        cwd=root,
        env=frozen_environment(args),
        capture_output=True,
        check=False,
    )
    if completed.returncode or not target.is_file():
        print(
            f"unable to build r14 fixture {name}: "
            + completed.stderr.decode("utf-8", errors="replace")[:1024],
            file=sys.stderr,
        )
        raise RuntimeError("fixture generation failed")


def build_fixtures(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    source_text = root / "source.csv"
    source_text.write_text(
        "X,Y,Z,Intensity,ReturnNumber,NumberOfReturns,Classification,"
        "UserData,PointSourceId,Red,Green,Blue,GpsTime\n"
        "0,0,0,10,1,1,2,1,100,1000,2000,3000,1.0\n"
        "1,2,3,11,1,2,3,2,101,1001,2001,3001,2.0\n"
        "2,4,6,12,2,2,4,3,102,1002,2002,3002,3.0\n"
        "-1,-2,-3,13,1,1,5,4,103,1003,2003,3003,4.0\n"
        "10,10,10,14,1,3,6,5,104,1004,2004,3004,5.0\n"
        "11,10,9,15,2,3,7,6,105,1005,2005,3005,6.0\n"
        "12,10,8,16,3,3,8,7,106,1006,2006,3006,7.0\n"
        "20,1,5,17,1,1,9,8,107,1007,2007,3007,8.0\n"
        "21,2,6,18,1,1,10,9,108,1008,2008,3008,9.0\n"
        "22,3,7,19,1,1,11,10,109,1009,2009,3009,10.0\n"
        "30,30,30,20,1,1,12,11,110,1010,2010,3010,11.0\n"
        "31,31,31,21,1,1,13,12,111,1011,2011,3011,12.0\n",
        encoding="utf-8",
    )

    source_las = root / "source.las"
    run_fixture_pipeline(args, root, "las", [
        {"type": "readers.text", "filename": str(source_text)},
        {"type": "writers.las", "filename": str(source_las),
         "minor_version": 4, "dataformat_id": 7,
         "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01},
    ], source_las)

    source_laz = root / "source.laz"
    run_fixture_pipeline(args, root, "laz", [
        {"type": "readers.las", "filename": str(source_las)},
        {"type": "writers.las", "filename": str(source_laz),
         "compression": "true"},
    ], source_laz)

    # This bounded generated fixture matches the automatic B0251/D0250
    # admission facts, corrected by B0252/D0251, without depending on the local
    # benchmark corpus. Its full-file
    # differential exercises the public two-worker selector and deterministic
    # ordered chunk publication at more than one chunk boundary.
    source_1m_las = root / "source-1m.las"
    run_fixture_pipeline(args, root, "las-1m", [
        {"type": "readers.faux",
         "bounds": "([184500,185999.99],[494923.21,494999.99],"
                   "[367.44,500.41])",
         "count": 1_000_000, "mode": "ramp"},
        {"type": "writers.las", "filename": str(source_1m_las),
         "minor_version": 4, "dataformat_id": 7,
         "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01,
         "offset_x": 0.0, "offset_y": 0.0, "offset_z": 0.0},
    ], source_1m_las)

    source_copc = root / "source.copc.laz"
    run_fixture_pipeline(args, root, "copc", [
        {"type": "readers.las", "filename": str(source_las)},
        {"type": "writers.copc", "filename": str(source_copc),
         "threads": 1, "fixed_seed": True},
    ], source_copc)

    truncated = root / "truncated.las"
    truncated.write_bytes(source_las.read_bytes()[:100])
    malformed_laz = root / "malformed.laz"
    malformed_laz.write_bytes(b"not a LAZ file\n")
    malformed_copc = root / "malformed.copc.laz"
    malformed_copc.write_bytes(b"not a COPC file\n")
    return {
        "source.las": source_las,
        "source-1m.las": source_1m_las,
        "source.laz": source_laz,
        "source.copc.laz": source_copc,
        "truncated.las": truncated,
        "malformed.laz": malformed_laz,
        "malformed.copc.laz": malformed_copc,
    }


def las_header(path: Path) -> tuple[int, bool, bool] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 227 or data[:4] != b"LASF" or data[24] != 1:
        return None
    minor = data[25]
    if minor >= 4:
        if len(data) < 255:
            return None
        points = struct.unpack_from("<Q", data, 247)[0]
    else:
        points = struct.unpack_from("<I", data, 107)[0]
    compressed = bool(data[104] & 0x80)
    header_size = struct.unpack_from("<H", data, 94)[0]
    vlr_count = struct.unpack_from("<I", data, 100)[0]
    offset = header_size
    copc = False
    for _ in range(vlr_count):
        if offset + 54 > len(data):
            return None
        user_id = data[offset + 2:offset + 18].split(b"\0", 1)[0]
        record_id = struct.unpack_from("<H", data, offset + 18)[0]
        length = struct.unpack_from("<H", data, offset + 20)[0]
        copc = copc or (user_id == b"copc" and record_id == 1)
        offset += 54 + length
    return points, compressed, copc


def require_automatic_r14_admission_facts(path: Path) -> None:
    data = path.read_bytes()
    expected: tuple[tuple[int, str, object], ...] = (
        (24, "version_major", 1),
        (25, "version_minor", 4),
        (94, "header_size", 375),
        (96, "point_offset", 375),
        (104, "point_format", 7),
        (105, "point_width", 36),
        (131, "scale_x", 0.01),
        (139, "scale_y", 0.01),
        (147, "scale_z", 0.01),
        (155, "offset_x", 0.0),
        (163, "offset_y", 0.0),
        (171, "offset_z", 0.0),
        (179, "max_x", 185999.99),
        (187, "min_x", 184500.0),
        (195, "max_y", 494999.99),
        (203, "min_y", 494923.21),
        (211, "max_z", 500.41),
        (219, "min_z", 367.44),
        (247, "point_count", 1_000_000),
    )
    if len(data) != 36_000_375 or data[:4] != b"LASF":
        raise RuntimeError(
            "generated automatic r14 fixture has unexpected size/signature")
    for offset, name, value in expected:
        if isinstance(value, float):
            observed: object = struct.unpack_from("<d", data, offset)[0]
        elif offset == 247:
            observed = struct.unpack_from("<Q", data, offset)[0]
        elif offset in (94, 105):
            observed = struct.unpack_from("<H", data, offset)[0]
        elif offset == 96:
            observed = struct.unpack_from("<I", data, offset)[0]
        else:
            observed = data[offset]
        if observed != value:
            raise RuntimeError(
                f"generated automatic r14 fixture {name}={observed!r}, "
                f"expected {value!r}")


def run_case(args: argparse.Namespace, root: Path, case: Case,
             fixtures: dict[str, Path], suffix: str) -> Path | None:
    name = f"r14-conversion-{case.name}{suffix}"
    pipeline = root / f"{name}.json"
    pipeline.write_text(
        json.dumps({"pipeline": list(case.stages)}, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    input_placeholder = case.stages[0]["filename"]
    command = [
        sys.executable,
        str(args.differential.resolve()),
        "--oracle", str(args.oracle.resolve()),
        "--candidate", str(args.candidate.resolve()),
        "--case", name,
        "--work-dir", str(args.work_dir.resolve()),
        "--frozen-time-library", str(args.frozen_time_library.resolve()),
        "--keep-success",
        "--seed-file", f"{input_placeholder}={fixtures[case.source_name]}",
        "--seed-file", f"pipeline.json={pipeline}",
    ]
    if case.name == "las-to-laz-automatic-1m":
        # The public candidate reaches the real LasWriter and fails unless the
        # launcher armed the measured two-worker compressor. Keep the assertion
        # candidate-only so the forked host oracle exercises its serial default.
        command.extend([
            "--candidate-env", "PDAL_TEST_REQUIRE_LAZ_COMPRESSION_THREADS=2",
        ])
    command.extend(["--", "pipeline", "pipeline.json"])
    completed = subprocess.run(command, check=False)
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
    artifact = report["oracle_artifacts"].get(case.output_name)
    if case.expected_returncode:
        return report_path
    if artifact is None or artifact["bytes"] <= 0:
        print(f"{name}: positive case did not publish {case.output_name}",
              file=sys.stderr)
        return None
    case_root = Path(report["case_root"])
    for side in ("oracle", "candidate"):
        header = las_header(case_root / side / case.output_name)
        expected = (case.expected_points, case.expected_compressed,
                    case.expected_copc)
        if header != expected:
            print(f"{name}: {side} header {header} != {expected}",
                  file=sys.stderr)
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
            prefix="pdg-r14-conversion-matrix-",
            dir=args.work_dir) as temporary:
        root = Path(temporary)
        try:
            fixtures = build_fixtures(args, root)
            require_automatic_r14_admission_facts(fixtures["source-1m.las"])
        except RuntimeError as error:
            print(f"r14 fixture validation failed: {error}", file=sys.stderr)
            return 1
        for case in CASES:
            if case.repeat:
                first = run_case(args, root, case, fixtures, "-run-1")
                second = run_case(args, root, case, fixtures, "-run-2")
                if first is None or second is None:
                    return 1
                left = json.loads(first.read_text(encoding="utf-8"))
                right = json.loads(second.read_text(encoding="utf-8"))
                if (left["candidate_artifacts"] !=
                        right["candidate_artifacts"] or
                        left["oracle_artifacts"] !=
                        right["oracle_artifacts"]):
                    print(f"r14 {case.name} is not deterministic",
                          file=sys.stderr)
                    return 1
            elif run_case(args, root, case, fixtures, "") is None:
                return 1

    print(f"exact r14 conversion matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
