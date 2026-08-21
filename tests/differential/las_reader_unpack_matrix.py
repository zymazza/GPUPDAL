#!/usr/bin/env python3
"""Exact pinned-oracle process matrix for readers.las record unpacking (D0269).

The fork unpacks decoded LAS/LAZ records into point-table rows on a
fixed-slot pool in both execution modes (a `Streamable::readStreamBatch`
hook for the streaming executor and per-tile unpacking in the standard
`read()`), consuming tiles in the pinned order. Every case compares the
public candidate (delegating to the forked sibling) against the pinned
upstream oracle: bytes, header, stdout, stderr, and status. Fixtures are
written by the pinned oracle from generated text so no corpus file is
touched; the test-only ``PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS`` hook
makes them large enough for the parallel path on any machine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


FORCED = ("PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=3",)

COLUMNS = ("X,Y,Z,Intensity,ReturnNumber,NumberOfReturns,Classification,"
           "ScanAngleRank,UserData,PointSourceId,GpsTime,Red,Green,Blue,"
           "Infrared,Synthetic,KeyPoint,Withheld,Overlap,ScanChannel,"
           "ScanDirectionFlag,EdgeOfFlightLine,Foo,Bar")


@dataclass(frozen=True)
class Fixture:
    name: str
    count: int
    writer: dict[str, object]


@dataclass(frozen=True)
class Case:
    name: str
    fixture: str
    reader: dict[str, object] = None
    nostream: bool = False
    filters: tuple[dict[str, object], ...] = ()
    candidate_env: tuple[str, ...] = FORCED
    text_output: bool = False


FIXTURES = (
    Fixture("f3-las", 30000, {"minor_version": 2, "dataformat_id": 3}),
    Fixture("f1-las", 30000, {"minor_version": 2, "dataformat_id": 1}),
    Fixture("f0-las", 30000, {"minor_version": 2, "dataformat_id": 0}),
    Fixture("f2-laz", 30000, {"minor_version": 2, "dataformat_id": 2,
                              "compression": True}),
    Fixture("f6-laz", 30000, {"minor_version": 4, "dataformat_id": 6,
                              "compression": True}),
    Fixture("f7-eb-las", 30000, {"minor_version": 4, "dataformat_id": 7,
                                 "extra_dims": "all"}),
    Fixture("f8-eb-laz", 30000, {"minor_version": 4, "dataformat_id": 8,
                                 "extra_dims": "Foo=double,Bar=uint16",
                                 "compression": True}),
    Fixture("f3-eb-las", 30000, {"minor_version": 2, "dataformat_id": 3,
                                 "extra_dims": "Foo=double,Bar=uint16"}),
    # Multi-tile: 120,000 points cross two 50,000-record tiles/chunks.
    Fixture("f7-tiles-las", 120000, {"minor_version": 4, "dataformat_id": 7}),
    Fixture("f7-tiles-laz", 120000, {"minor_version": 4, "dataformat_id": 7,
                                     "compression": True}),
    Fixture("f7-small-las", 700, {"minor_version": 4, "dataformat_id": 7}),
)

CASES = (
    Case("f3-las-stream", "f3-las"),
    Case("f3-las-nostream", "f3-las", nostream=True),
    Case("f1-las-stream-text", "f1-las", text_output=True),
    Case("f0-las-nostream", "f0-las", nostream=True),
    Case("f2-laz-stream", "f2-laz"),
    Case("f2-laz-nostream-text", "f2-laz", nostream=True, text_output=True),
    Case("f6-laz-stream", "f6-laz"),
    Case("f6-laz-nostream", "f6-laz", nostream=True),
    # LAS 1.4 extra-bytes VLRs are read automatically; 1.2 extra bytes need
    # the reader's extra_dims spec (or use_eb_vlr).
    Case("f7-eb-vlr-stream", "f7-eb-las"),
    Case("f7-eb-vlr-nostream-text", "f7-eb-las", nostream=True,
         text_output=True),
    Case("f8-eb-vlr-laz-stream", "f8-eb-laz"),
    Case("f8-eb-vlr-laz-nostream", "f8-eb-laz", nostream=True),
    Case("f3-eb-named-stream", "f3-eb-las",
         {"extra_dims": "Foo=double,Bar=uint16"}),
    Case("f3-eb-named-nostream-text", "f3-eb-las",
         {"extra_dims": "Foo=double,Bar=uint16"}, nostream=True,
         text_output=True),
    Case("f3-eb-use-eb-vlr-stream", "f3-eb-las", {"use_eb_vlr": True}),
    Case("f3-eb-ignored-stream", "f3-eb-las"),
    Case("f7-tiles-las-stream", "f7-tiles-las"),
    Case("f7-tiles-las-nostream", "f7-tiles-las", nostream=True),
    Case("f7-tiles-laz-stream", "f7-tiles-laz"),
    Case("f7-tiles-laz-nostream", "f7-tiles-laz", nostream=True),
    # `count` bounds the batches; `start` keeps the pinned per-point path.
    Case("f7-tiles-count-stream", "f7-tiles-las", {"count": 73456}),
    Case("f7-tiles-count-nostream", "f7-tiles-las", {"count": 73456},
         nostream=True),
    Case("f7-tiles-start-nostream", "f7-tiles-las", {"start": 51234},
         nostream=True),
    Case("f7-tiles-start-count-nostream", "f7-tiles-las",
         {"start": 49999, "count": 20000}, nostream=True),
    # Downstream streaming skips ride on the same batches.
    Case("f6-laz-range-stream", "f6-laz",
         filters=({"type": "filters.range", "limits": "Intensity[0:40000]"},)),
    Case("f7-small-stays-serial-stream", "f7-small-las"),
    Case("f7-small-stays-serial-nostream", "f7-small-las", nostream=True),
    Case("workers-disabled-control-stream", "f7-tiles-laz", candidate_env=(
        "PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS=1",)),
    Case("default-policy-nostream", "f7-tiles-laz", candidate_env=(),
         nostream=True),
    Case("default-policy-stream", "f7-tiles-las", candidate_env=()),
)


def write_text(path: Path, count: int) -> None:
    lines = [COLUMNS]
    for k in range(count):
        i, j = k % 150, k // 150
        x = 184500.0 + 0.7 * i + 0.13 * ((k * 7919) % 97) / 97.0
        y = 494920.0 + 0.7 * j + 0.11 * ((k * 104729) % 89) / 89.0
        z = -2.0 + 9.0 * ((k * 1299709) % 101) / 101.0
        returns = 1 + (k * 31) % 5
        rn = 1 + (k * 17) % returns
        lines.append(",".join(str(v) for v in (
            f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", (k * 37) % 65536, rn,
            returns, (k * 7) % 32, (k % 181) - 90, k % 256, k % 4097,
            f"{300000.0 + 0.0001 * k:.6f}", (k * 3) % 65536,
            (k * 5) % 65536, (k * 11) % 65536, (k * 13) % 65536,
            1 if k % 41 == 0 else 0, 1 if k % 43 == 0 else 0,
            1 if k % 47 == 0 else 0, 1 if k % 131 == 0 else 0, k % 4, k % 2,
            1 if k % 53 == 0 else 0, f"{0.001 * k:.6f}", (k * 29) % 65536)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path,
                        help="the forked sibling pdal (candidate delegate)")
    parser.add_argument("--pinned-oracle", type=Path,
                        help="pinned upstream pdal; skips when absent")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pinned_oracle is None:
        print("las reader unpack matrix skipped: no --pinned-oracle configured",
              file=sys.stderr)
        return 0
    args.work_dir.mkdir(parents=True, exist_ok=True)
    fixtures = {f.name: f for f in FIXTURES}
    with tempfile.TemporaryDirectory(prefix="pdg-las-reader-unpack-matrix-",
                                     dir=args.work_dir) as temporary:
        generated = Path(temporary)
        texts: dict[int, Path] = {}
        built: dict[str, Path] = {}

        def fixture_path(name: str) -> Path:
            if name in built:
                return built[name]
            fixture = fixtures[name]
            text = texts.get(fixture.count)
            if text is None:
                text = generated / f"points-{fixture.count}.csv"
                write_text(text, fixture.count)
                texts[fixture.count] = text
            suffix = ".laz" if fixture.writer.get("compression") else ".las"
            output = generated / f"{name}{suffix}"
            writer = dict(fixture.writer)
            writer["type"] = "writers.las"
            writer["filename"] = str(output)
            recipe = generated / f"{name}-fixture.json"
            recipe.write_text(json.dumps({"pipeline": [
                {"type": "readers.text", "filename": str(text)}, writer]}),
                encoding="utf-8")
            subprocess.run([str(args.pinned_oracle.resolve()), "pipeline",
                            str(recipe)], check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            built[name] = output
            return output

        for case in CASES:
            fixture = fixture_path(case.fixture)
            reader = dict(case.reader or {})
            reader["type"] = "readers.las"
            reader["filename"] = "input" + fixture.suffix
            if case.text_output:
                sink = {"type": "writers.text", "filename": "output.txt",
                        "precision": 17}
                output = "output.txt"
            else:
                sink = {"type": "writers.las", "filename": "output.las",
                        "minor_version": 4, "dataformat_id": 8,
                        "extra_dims": "all"}
                output = "output.las"
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(json.dumps({"pipeline": [
                reader, *case.filters, sink]}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            name = f"las-reader-unpack-{case.name}"
            command = [
                sys.executable, str(args.differential.resolve()),
                "--oracle", str(args.pinned_oracle.resolve()),
                "--candidate-oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", name,
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--keep-success",
                "--seed-file", f"input{fixture.suffix}={fixture.resolve()}",
                "--seed-file", f"pipeline.json={pipeline.resolve()}",
            ]
            for assignment in case.candidate_env:
                command.extend(["--candidate-env", assignment])
            command.extend(["--", "pipeline", "pipeline.json"])
            if case.nostream:
                command.append("--nostream")
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                return completed.returncode
            report_path = args.work_dir / "reports" / f"{name}.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["oracle_run"]["returncode"] != 0:
                print(f"{name}: oracle failed", file=sys.stderr)
                return 1
            artifact = report["oracle_artifacts"].get(output)
            if artifact is None or artifact["bytes"] <= 0:
                print(f"{name}: no {output} published", file=sys.stderr)
                return 1
    print(f"exact las reader unpack matrix: {len(CASES)} pinned-oracle cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
