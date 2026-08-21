#!/usr/bin/env python3
"""Generate deterministic LAS boundary cases and compare native PDG with PDAL."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


POINT_BYTES = {0: 20, 1: 28, 2: 26, 3: 34, 6: 30, 7: 36, 8: 38}


@dataclass(frozen=True)
class Case:
    name: str
    point_format: int
    point_count: int
    scale: tuple[float, float, float] = (0.005, 0.01, 0.25)
    offset: tuple[float, float, float] = (0.0, -1234.56, 10000.0)
    extra_bytes: int = 0
    wkt_encoding: bool = True
    requires_native: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    return parser.parse_args()


def coordinate(index: int, axis: int, seed: int) -> int:
    multipliers = (104729, 130363, 155921)
    return ((index * multipliers[axis] + seed * (axis + 3)) % 100003) - 50001


def point_record(case: Case, index: int) -> bytes:
    base_bytes = POINT_BYTES[case.point_format]
    record = bytearray(base_bytes + case.extra_bytes)
    seed = case.point_format * 1000003 + index
    for axis in range(3):
        struct.pack_into("<i", record, axis * 4,
                         coordinate(index, axis, seed))
    struct.pack_into("<H", record, 12,
                     (index * 4051 + case.point_format * 7919) & 0xFFFF)

    if case.point_format <= 3:
        return_number = index % 5 + 1
        number_of_returns = max(return_number, (index * 3) % 5 + 1)
        record[14] = (
            return_number
            | (number_of_returns << 3)
            | ((index >> 1) & 1) << 6
            | ((index >> 2) & 1) << 7
        )
        classification = 12 if index % 97 == 0 else index % 32
        record[15] = (
            classification
            | ((index >> 3) & 1) << 5
            | ((index >> 4) & 1) << 6
            | ((index >> 5) & 1) << 7
        )
        struct.pack_into("<b", record, 16, index % 181 - 90)
        record[17] = (index * 29) & 0xFF
        struct.pack_into("<H", record, 18, (index * 257) & 0xFFFF)
        if case.point_format in (1, 3):
            gps_bits = (
                0x7FF8000000000000 | index
                if index % 127 == 0
                else struct.unpack("<Q", struct.pack("<d", -12345.125 + index / 16))[0]
            )
            struct.pack_into("<Q", record, 20, gps_bits)
        color_offset = 20 if case.point_format == 2 else (
            28 if case.point_format == 3 else None
        )
        if color_offset is not None:
            for channel in range(3):
                struct.pack_into(
                    "<H", record, color_offset + channel * 2,
                    (index * (channel + 11) * 101) & 0xFFFF
                )
    else:
        return_number = index % 15 + 1
        number_of_returns = max(return_number, (index * 7) % 15 + 1)
        record[14] = return_number | (number_of_returns << 4)
        record[15] = (
            ((index >> 1) & 1)
            | (((index >> 2) & 1) << 1)
            | (((index >> 3) & 1) << 2)
            | (((index >> 4) & 1) << 3)
            | (((index >> 5) & 3) << 4)
            | (((index >> 7) & 1) << 6)
            | (((index >> 8) & 1) << 7)
        )
        record[16] = (index * 67) & 0xFF
        record[17] = (index * 29) & 0xFF
        struct.pack_into("<h", record, 18,
                         ((index * 40503) & 0xFFFF) - 32768)
        struct.pack_into("<H", record, 20, (index * 257) & 0xFFFF)
        gps_bits = (
            0x7FF8000000000000 | index
            if index % 127 == 0
            else struct.unpack("<Q", struct.pack("<d", -12345.125 + index / 16))[0]
        )
        struct.pack_into("<Q", record, 22, gps_bits)
        if case.point_format in (7, 8):
            for channel in range(3):
                struct.pack_into(
                    "<H", record, 30 + channel * 2,
                    (index * (channel + 11) * 101) & 0xFFFF
                )
        if case.point_format == 8:
            struct.pack_into("<H", record, 36, (index * 313) & 0xFFFF)

    for extra in range(case.extra_bytes):
        record[base_bytes + extra] = (index * 31 + extra * 17) & 0xFF
    return bytes(record)


def write_las(path: Path, case: Case) -> None:
    extended = case.point_format >= 6
    header_bytes = 375 if extended else 227
    point_bytes = POINT_BYTES[case.point_format] + case.extra_bytes
    header = bytearray(header_bytes)
    header[:4] = b"LASF"
    if extended and case.wkt_encoding:
        struct.pack_into("<H", header, 6, 1 << 4)
    header[24] = 1
    header[25] = 4 if extended else 2
    struct.pack_into("<H", header, 94, header_bytes)
    struct.pack_into("<I", header, 96, header_bytes)
    header[104] = case.point_format
    struct.pack_into("<H", header, 105, point_bytes)
    if extended:
        struct.pack_into("<Q", header, 247, case.point_count)
    else:
        struct.pack_into("<I", header, 107, case.point_count)
    for axis in range(3):
        struct.pack_into("<d", header, 131 + axis * 8, case.scale[axis])
        struct.pack_into("<d", header, 155 + axis * 8, case.offset[axis])
    with path.open("wb") as stream:
        stream.write(header)
        for index in range(case.point_count):
            stream.write(point_record(case, index))


def main() -> int:
    args = parse_args()
    for path in (
        args.oracle,
        args.candidate,
        args.differential,
        args.frozen_time_library,
    ):
        if not path.is_file():
            print(f"required file does not exist: {path}", file=sys.stderr)
            return 2
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        *(Case(f"format-{point_format}", point_format, 1027)
          for point_format in POINT_BYTES),
        Case("empty-format-7", 7, 0),
        Case("threaded-format-7", 7, 131073, (0.01, 0.01, 0.01),
             (0.0, 0.0, 0.0)),
        Case("unregistered-extra-bytes-format-3", 3, 2053,
             extra_bytes=7),
        Case("missing-wkt-encoding-format-6", 6, 17,
             wkt_encoding=False, requires_native=False),
    ]
    with tempfile.TemporaryDirectory(
        prefix="pdg-native-matrix-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        pipeline_cases = 0
        assign_cases = 0
        for case in cases:
            fixture = generated / f"{case.name}.las"
            write_las(fixture, case)
            environment = os.environ.copy()
            if case.requires_native:
                environment["PDG_REQUIRE_NATIVE"] = "1"
            else:
                environment.pop("PDG_REQUIRE_NATIVE", None)
            command = [
                sys.executable,
                str(args.differential.resolve()),
                "--oracle",
                str(args.oracle.resolve()),
                "--candidate",
                str(args.candidate.resolve()),
                "--case",
                f"generated-native-{case.name}",
                "--work-dir",
                str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--",
                "translate",
                str(fixture.resolve()),
                "out.las",
            ]
            completed = subprocess.run(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            sys.stdout.buffer.write(completed.stdout)
            sys.stderr.buffer.write(completed.stderr)
            if completed.returncode:
                failures = args.work_dir / "failures"
                failures.mkdir(exist_ok=True)
                preserved = failures / fixture.name
                shutil.copy2(fixture, preserved)
                print(f"preserved failing generated fixture: {preserved}",
                      file=sys.stderr)
                return completed.returncode

            if case.name == f"format-{case.point_format}":
                mappings = [
                    "Intensity=>PointSourceId",
                    "ReturnNumber=>UserData",
                ]
                if case.point_format in (6, 7, 8):
                    mappings.extend(["ScanAngleRank=>GpsTime", "GpsTime=>Z"])
                elif case.point_format in (1, 3):
                    mappings.extend(["Intensity=>GpsTime", "GpsTime=>Z"])
                else:
                    mappings.append("Intensity=>Z")
                if case.point_format in (2, 3, 7, 8):
                    mappings.append("Red=>Blue")
                if case.point_format in (6, 7, 8):
                    mappings.append("ScanChannel=>Classification")
                pipeline = generated / f"{case.name}-ferry.json"
                pipeline.write_text(
                    json.dumps(
                        {
                            "pipeline": [
                                {
                                    "type": "readers.las",
                                    "filename": str(fixture.resolve()),
                                },
                                {
                                    "type": "filters.ferry",
                                    "dimensions": ",".join(mappings),
                                },
                                {
                                    "type": "writers.las",
                                    "filename": "out.las",
                                },
                            ]
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                pipeline_command = [
                    sys.executable,
                    str(args.differential.resolve()),
                    "--oracle",
                    str(args.oracle.resolve()),
                    "--candidate",
                    str(args.candidate.resolve()),
                    "--case",
                    f"generated-native-ferry-{case.name}",
                    "--work-dir",
                    str(args.work_dir.resolve()),
                    "--frozen-time-library",
                    str(args.frozen_time_library.resolve()),
                    "--",
                    "pipeline",
                    str(pipeline.resolve()),
                ]
                pipeline_completed = subprocess.run(
                    pipeline_command,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                sys.stdout.buffer.write(pipeline_completed.stdout)
                sys.stderr.buffer.write(pipeline_completed.stderr)
                if pipeline_completed.returncode:
                    failures = args.work_dir / "failures"
                    failures.mkdir(exist_ok=True)
                    preserved = failures / fixture.name
                    shutil.copy2(fixture, preserved)
                    print(
                        f"preserved failing generated fixture: {preserved}",
                        file=sys.stderr,
                    )
                    return pipeline_completed.returncode
                pipeline_cases += 1

                assignments = [
                    "Scratch = Intensity * 2 - 1",
                    "Classification = 7 WHERE Scratch >= 0 && ReturnNumber >= 1",
                    "PointSourceId = Intensity / 2",
                    "UserData = Classification + NumberOfReturns",
                ]
                if case.point_format in (1, 3, 6, 7, 8):
                    assignments.append(
                        "GpsTime = GpsTime + 0.5 WHERE !isnan(GpsTime)"
                    )
                if case.point_format in (2, 3, 7, 8):
                    assignments.append("Blue = Red + Green")
                if case.point_format in (6, 7, 8):
                    assignments.extend(
                        [
                            "Intensity = ScanAngleRank * 10",
                            "Classification = ScanChannel + 3",
                        ]
                    )
                assignments.append(
                    "ReturnNumber = Classification WHERE Classification >= 1 && Classification <= 15"
                )
                assign_pipeline = generated / f"{case.name}-assign.json"
                assign_pipeline.write_text(
                    json.dumps(
                        {
                            "pipeline": [
                                {
                                    "type": "readers.las",
                                    "filename": str(fixture.resolve()),
                                },
                                {
                                    "type": "filters.assign",
                                    "value": assignments,
                                },
                                {
                                    "type": "writers.las",
                                    "filename": "out.las",
                                },
                            ]
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                assign_command = [
                    sys.executable,
                    str(args.differential.resolve()),
                    "--oracle",
                    str(args.oracle.resolve()),
                    "--candidate",
                    str(args.candidate.resolve()),
                    "--case",
                    f"generated-native-assign-{case.name}",
                    "--work-dir",
                    str(args.work_dir.resolve()),
                    "--frozen-time-library",
                    str(args.frozen_time_library.resolve()),
                    "--",
                    "pipeline",
                    str(assign_pipeline.resolve()),
                ]
                assign_completed = subprocess.run(
                    assign_command,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                sys.stdout.buffer.write(assign_completed.stdout)
                sys.stderr.buffer.write(assign_completed.stderr)
                if assign_completed.returncode:
                    failures = args.work_dir / "failures"
                    failures.mkdir(exist_ok=True)
                    preserved = failures / fixture.name
                    shutil.copy2(fixture, preserved)
                    print(
                        f"preserved failing generated fixture: {preserved}",
                        file=sys.stderr,
                    )
                    return assign_completed.returncode
                assign_cases += 1
    print(
        f"exact native generated matrix: {len(cases)} translation cases, "
        f"{pipeline_cases} ferry cases, {assign_cases} assign cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
