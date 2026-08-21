#!/usr/bin/env python3
"""Exact process matrix for filters.reprojection under D0266's worker slots.

Every case compares the public candidate against the pinned upstream oracle
(the candidate delegates to the forked sibling, --candidate-oracle) because
the reprojection code changed in the fork's PDAL library. Coordinates are
emitted through writers.text at 17 significant digits so any bit difference
in a reprojected double is visible. The streaming batch (10,000 rows) and
standard mode (--nostream) are both covered, with natural and forced worker
counts, the `where` serial fallback, dropped unprojectable points, and the
error_on_failure diagnostic/status.
"""

from __future__ import annotations

import argparse
import json
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
    nostream: bool = False
    candidate_env: tuple[str, ...] = ()
    expected_returncode: int = 0


FORCED = ("PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=5",
          "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS=5")
RD_TO_MERCATOR = {"type": "filters.reprojection", "in_srs": "EPSG:28992",
                  "out_srs": "EPSG:3857"}
GEO_TO_MERCATOR = {"type": "filters.reprojection", "in_srs": "EPSG:4326",
                   "out_srs": "EPSG:3857"}
CASES = (
    Case("rd-to-mercator-stream", "rd", RD_TO_MERCATOR),
    Case("rd-to-mercator-nostream", "rd", RD_TO_MERCATOR, nostream=True),
    Case("rd-to-mercator-stream-forced", "rd", RD_TO_MERCATOR,
         candidate_env=FORCED),
    Case("rd-to-mercator-nostream-forced", "rd", RD_TO_MERCATOR,
         nostream=True, candidate_env=FORCED),
    Case("rd-to-mercator-where-serial", "rd",
         dict(RD_TO_MERCATOR, where="Intensity > 100"),
         candidate_env=FORCED),
    Case("geo-drop-unprojectable-stream", "geo-bad", GEO_TO_MERCATOR,
         candidate_env=FORCED),
    Case("geo-drop-unprojectable-nostream", "geo-bad", GEO_TO_MERCATOR,
         nostream=True, candidate_env=FORCED),
    Case("geo-error-on-failure-stream", "geo-bad",
         dict(GEO_TO_MERCATOR, error_on_failure=True),
         candidate_env=FORCED, expected_returncode=1),
    Case("geo-error-on-failure-nostream", "geo-bad",
         dict(GEO_TO_MERCATOR, error_on_failure=True), nostream=True,
         candidate_env=FORCED, expected_returncode=1),
    # Every 97th row is unprojectable: each slot drops several rows and GDAL
    # reports each failure; the replayed report order and count must equal
    # the serial oracle's in both modes.
    Case("geo-drop-many-stream-forced", "geo-bad-many", GEO_TO_MERCATOR,
         candidate_env=FORCED),
    Case("geo-drop-many-nostream-forced", "geo-bad-many", GEO_TO_MERCATOR,
         nostream=True, candidate_env=FORCED),
    Case("geo-drop-many-stream-natural", "geo-bad-many", GEO_TO_MERCATOR),
    Case("rd-large-natural-workers-stream", "rd-large", RD_TO_MERCATOR),
    Case("rd-large-natural-workers-nostream", "rd-large", RD_TO_MERCATOR,
         nostream=True),
)


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


def write_fixture(path: Path, kind: str) -> None:
    lines = ["X,Y,Z,Intensity"]
    if kind in ("rd", "rd-large"):
        side = 60 if kind == "rd" else 160
        for i in range(side):
            for j in range(side):
                x = 184500.0 + 25.0 * i + 0.013 * j
                y = 494923.21 + 0.48 * j + 0.007 * i
                z = 1.0 + 0.001 * (i * side + j) - 0.5 * ((i * 7 + j) % 3)
                lines.append(f"{x:.6f},{y:.6f},{z:.6f},{(i * 3 + j) % 251}")
    else:
        # Mostly valid geographic points plus latitudes beyond the Mercator
        # domain, which PROJ rejects; the third row is the first failure.
        count = 12000 if kind == "geo-bad-many" else 3000
        for k in range(count):
            lat = 52.0 + 0.0001 * k
            lon = 4.9 + 0.0002 * k
            bad = (k % 97 == 2) if kind == "geo-bad-many" else \
                k in (2, 700, 2999)
            if bad:
                lat = 95.0 + k
            lines.append(f"{lon:.7f},{lat:.7f},{k % 17},{k % 251}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.pinned_oracle is None:
        print("reprojection matrix skipped: no --pinned-oracle configured",
              file=sys.stderr)
        return 0
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdg-reprojection-matrix-",
                                     dir=args.work_dir) as temporary:
        generated = Path(temporary)
        for case in CASES:
            fixture = generated / f"{case.fixture}.csv"
            if not fixture.exists():
                write_fixture(fixture, case.fixture)
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(json.dumps({"pipeline": [
                {"type": "readers.text", "filename": "input.csv",
                 "override_srs": case.stage["in_srs"]},
                case.stage,
                {"type": "writers.text", "filename": "output.txt",
                 "order": "X,Y,Z,Intensity", "keep_unspecified": False,
                 "precision": 17},
            ]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            command = [
                sys.executable, str(args.differential.resolve()),
                "--oracle", str(args.pinned_oracle.resolve()),
                "--candidate-oracle", str(args.oracle.resolve()),
                "--candidate", str(args.candidate.resolve()),
                "--case", f"reprojection-matrix-{case.name}",
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
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
            report = json.loads(
                (args.work_dir / "reports" /
                 f"reprojection-matrix-{case.name}.json").read_text())
            if report["oracle_run"]["returncode"] != case.expected_returncode:
                print(f"{case.name}: expected status {case.expected_returncode}, "
                      f"observed {report['oracle_run']['returncode']}",
                      file=sys.stderr)
                return 1
    print(f"exact reprojection matrix: {len(CASES)} pinned-oracle cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
