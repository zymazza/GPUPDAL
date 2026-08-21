#!/usr/bin/env python3
"""Exact pinned-oracle process matrix for writers.las record packing (D0268).

The fork packs LAS point records on a fixed-slot pool in both the streaming
batch hook and the standard block path, merging one las::Summary per slot in
slot order, and repeats a run serially whenever a slot would have logged a
per-point warning or thrown. Every case compares the public candidate
(delegating to the forked sibling) against the pinned upstream oracle: bytes,
header summary (bounds, counts, return histograms), stdout, stderr, and
status. The test-only ``PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS`` hook makes
the fixtures large enough for the parallel path on any machine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


FORCED = ("PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=3",)


@dataclass(frozen=True)
class Case:
    name: str
    fixture: str
    writer: dict[str, object]
    nostream: bool = False
    filters: tuple[dict[str, object], ...] = ()
    candidate_env: tuple[str, ...] = FORCED
    expected_returncode: int = 0
    output: str = "output.las"


CASES = (
    Case("format3-las-stream", "plain",
         {"minor_version": 2, "dataformat_id": 3}),
    Case("format3-las-nostream", "plain",
         {"minor_version": 2, "dataformat_id": 3}, nostream=True),
    Case("format6-laz-stream", "plain",
         {"minor_version": 4, "dataformat_id": 6, "compression": True},
         output="output.laz"),
    Case("format8-laz-nostream", "plain",
         {"minor_version": 4, "dataformat_id": 8, "compression": True},
         nostream=True, output="output.laz"),
    Case("format7-extra-dims-all-nostream", "plain",
         {"minor_version": 4, "dataformat_id": 7, "extra_dims": "all"},
         nostream=True),
    Case("format7-extra-dims-all-stream", "plain",
         {"minor_version": 4, "dataformat_id": 7, "extra_dims": "all",
          "compression": True}, output="output.laz"),
    Case("format1-legacy-flags-stream", "plain",
         {"minor_version": 2, "dataformat_id": 1}),
    Case("format6-laz-nostream", "large",
         {"minor_version": 4, "dataformat_id": 6, "compression": True},
         nostream=True, output="output.laz"),
    Case("format3-las-large-stream", "large",
         {"minor_version": 2, "dataformat_id": 3}),
    # Per-point warnings (PDRF < 6 with Classification > 31, and Overlap set
    # alongside a Classification) must reach stderr in pinned order: any slot
    # that would warn defers, and the run repeats serially.
    Case("format3-class-over-31-warnings-stream", "warnings",
         {"minor_version": 2, "dataformat_id": 3}),
    Case("format3-class-over-31-warnings-nostream", "warnings",
         {"minor_version": 2, "dataformat_id": 3}, nostream=True),
    Case("format6-class-over-31-no-warning-stream", "warnings",
         {"minor_version": 4, "dataformat_id": 6}),
    # discard_high_return_numbers can drop records: serial path.
    Case("discard-high-return-numbers-stream", "highreturns",
         {"minor_version": 2, "dataformat_id": 3,
          "discard_high_return_numbers": True}),
    Case("discard-high-return-numbers-nostream", "highreturns",
         {"minor_version": 2, "dataformat_id": 3,
          "discard_high_return_numbers": True}, nostream=True),
    # A where clause on the writer keeps the per-point path; skipped table
    # rows from an upstream streaming filter are compacted before packing.
    Case("writer-where-stream", "plain",
         {"minor_version": 2, "dataformat_id": 3,
          "where": "Classification == 2"}),
    Case("upstream-range-skips-stream", "plain",
         {"minor_version": 4, "dataformat_id": 6, "compression": True},
         filters=({"type": "filters.range", "limits": "Intensity[0:40000]"},),
         output="output.laz"),
    Case("upstream-range-skips-nostream", "plain",
         {"minor_version": 4, "dataformat_id": 6},
         filters=({"type": "filters.range", "limits": "Intensity[0:40000]"},),
         nostream=True),
    # Auto offset in stream mode is decided from the first point serially.
    Case("auto-offset-stream", "plain",
         {"minor_version": 2, "dataformat_id": 3, "offset_x": "auto",
          "offset_y": "auto", "offset_z": "auto"}),
    Case("auto-scale-offset-nostream", "plain",
         {"minor_version": 2, "dataformat_id": 3, "scale_x": "auto",
          "scale_y": "auto", "scale_z": "auto", "offset_x": "auto",
          "offset_y": "auto", "offset_z": "auto"}, nostream=True),
    # An unconvertible scaled value inside a parallel run must fail with the
    # pinned diagnostic and status (serial repeat rethrows at the first
    # offending record).
    Case("unconvertible-scale-stream", "large",
         {"minor_version": 2, "dataformat_id": 3, "scale_x": 0.0001,
          "offset_x": 184500.0}, expected_returncode=1),
    Case("unconvertible-scale-nostream", "large",
         {"minor_version": 2, "dataformat_id": 3, "scale_x": 0.0001,
          "offset_x": 184500.0}, nostream=True, expected_returncode=1),
    Case("workers-disabled-control-nostream", "plain",
         {"minor_version": 4, "dataformat_id": 6},
         nostream=True, candidate_env=(
             "PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS=1",)),
    Case("default-policy-nostream", "large",
         {"minor_version": 4, "dataformat_id": 7, "compression": True},
         nostream=True, candidate_env=(), output="output.laz"),
)


COLUMNS = ("X,Y,Z,Intensity,ReturnNumber,NumberOfReturns,Classification,"
           "ScanAngleRank,UserData,PointSourceId,GpsTime,Red,Green,Blue,"
           "Infrared,Synthetic,KeyPoint,Withheld,Overlap,ScanChannel,"
           "ScanDirectionFlag,EdgeOfFlightLine,Foo")


def write_fixture(path: Path, kind: str) -> None:
    # 30,000 rows: three 10,000-row stream batches (the first is serial
    # because of the writer's first-point setup) and, in standard mode, one
    # ~1 MB parallel block followed by a short serial tail.
    count = 30000
    lines = [COLUMNS]
    for k in range(count):
        i, j = k % 150, k // 150
        x = 184500.0 + 0.7 * i + 0.13 * ((k * 7919) % 97) / 97.0
        y = 494920.0 + 0.7 * j + 0.11 * ((k * 104729) % 89) / 89.0
        z = -2.0 + 9.0 * ((k * 1299709) % 101) / 101.0
        if kind == "large" and k == 15000:
            # Beyond int32 at scale 0.0001/offset 184500 (unconvertible
            # cases only): the record sits inside a parallel batch/block.
            x = 184500.0 + 500000.0
        returns = 1 + (k * 31) % 5
        rn = 1 + (k * 17) % returns
        if kind == "highreturns" and k % 97 == 0:
            returns = 8 + k % 5
            rn = 1 + (k * 13) % returns
        classification = (k * 7) % 32
        if kind == "warnings":
            if k % 211 == 0:
                classification = 32 + (k % 200)
            elif k % 173 == 0:
                classification = 12
        overlap = 1 if (kind == "warnings" and k % 131 == 0) else 0
        lines.append(",".join(str(v) for v in (
            f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", (k * 37) % 65536, rn,
            returns, classification, (k % 181) - 90, k % 256, k % 4097,
            f"{300000.0 + 0.0001 * k:.6f}", (k * 3) % 65536,
            (k * 5) % 65536, (k * 11) % 65536, (k * 13) % 65536,
            1 if k % 41 == 0 else 0, 1 if k % 43 == 0 else 0,
            1 if k % 47 == 0 else 0, overlap, k % 4, k % 2, 1 if k % 53 == 0
            else 0, f"{0.001 * k:.6f}")))
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
        print("las writer pack matrix skipped: no --pinned-oracle configured",
              file=sys.stderr)
        return 0
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdg-las-writer-pack-matrix-",
                                     dir=args.work_dir) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = generated / f"{case.fixture}.csv"
            if not fixture.exists():
                write_fixture(fixture, case.fixture)
            pipeline = generated / f"{case.name}.json"
            writer = dict(case.writer)
            writer["type"] = "writers.las"
            writer["filename"] = case.output
            pipeline.write_text(json.dumps({"pipeline": [
                {"type": "readers.text", "filename": "input.csv"},
                *case.filters,
                writer,
            ]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            name = f"las-writer-pack-{case.name}"
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
                "--seed-file", f"input.csv={fixture.resolve()}",
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
            observed = report["oracle_run"]["returncode"]
            if observed != case.expected_returncode:
                print(f"{name}: expected status {case.expected_returncode}, "
                      f"observed {observed}", file=sys.stderr)
                return 1
            output = report["oracle_artifacts"].get(case.output)
            if not case.expected_returncode and (
                    output is None or output["bytes"] <= 0):
                print(f"{name}: positive case did not publish {case.output}",
                      file=sys.stderr)
                return 1
            if case.fixture == "warnings" and case.writer["dataformat_id"] < 6:
                preview = report["oracle_run"]["stderr_preview"]
                if "can't be written to LAS" not in preview:
                    print(f"{name}: expected per-point warnings on stderr",
                          file=sys.stderr)
                    return 1
    print(f"exact las writer pack matrix: {len(CASES)} pinned-oracle cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
