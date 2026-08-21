#!/usr/bin/env python3
"""Bounded complete-process matrix for the r11 classification contract.

B0258/D0257 additions: the statistical outlier and neighbor classifier
per-point kNN passes may run on fixed-chunk host workers. The 144-point
fixture is far below the 4096-rows-per-worker threshold, so cases force the
chunked path through the test-only ``PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS``
hook and prove the observed count with ``PDAL_TEST_REQUIRE_...``; both are
candidate-only environment values, so every case still compares against the
unchanged pinned oracle.
"""

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
    mean_k: object = 8
    multiplier: object = 2.0
    k: object = 7
    domain: object = "Classification[1:1]"
    expected_returncode: int = 0
    repeat: bool = False
    precache_then_mutate_xyz: bool = False
    dimension: object = None
    candidate_file: bool = False
    assign_gpstime: bool = False
    candidate_env: tuple[str, ...] = ()
    candidate_only_failure: bool = False
    mutator: str | None = None
    mutator_after_outlier: bool = False
    pinned_oracle: bool = False
    consumer: str = "classifier"
    side: int = 12
    producer: str | None = None
    chain_classifier: bool = False


CASES = (
    Case("headline"),
    Case("legal-domain-drift", domain="Classification[2:2]"),
    Case("legal-outlier-neighbor-count-drift", mean_k=12, k=9),
    Case("invalid-neighbor-count", k=0, expected_returncode=1),
    Case("invalid-domain-dimension", domain="Missing[1:1]",
         expected_returncode=1),
    Case("invalid-outlier-multiplier", multiplier="invalid",
         expected_returncode=1),
    Case("cached-index-coordinate-mutation", mean_k=2, multiplier=0.0,
         precache_then_mutate_xyz=True),
    Case("determinism", repeat=True),
    # Worker-chunked exact passes (B0258/D0257). Forced counts split the
    # 144-point view into uneven fixed chunks; the required count proves the
    # candidate really executed that many workers in both stages.
    Case("forced-three-workers", candidate_env=(
        "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=3",
        "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS=3")),
    Case("forced-five-workers-legal-drift", domain="Classification[2:2]",
         mean_k=12, k=9, candidate_env=(
             "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=5",
             "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS=5")),
    Case("forced-workers-exceed-rows", candidate_env=(
        "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=1000",
        "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS=144")),
    Case("forced-workers-candidate-file", candidate_file=True,
         candidate_env=(
             "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=3",
             "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS=3")),
    Case("forced-workers-no-domain-explicit-dimension", domain="",
         dimension="Classification", candidate_env=(
             "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=4",
             "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS=4")),
    # A per-row conversion failure inside a worker must surface the same
    # first-row diagnostic and status as the serial oracle: every GpsTime is
    # assigned outside the integer vote range before the classifier.
    Case("forced-workers-unconvertible-vote-dimension", dimension="GpsTime",
         domain="Classification[2:2]", assign_gpstime=True,
         expected_returncode=1, candidate_env=(
             "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=3",)),
    Case("workers-disabled-control", candidate_env=(
        "PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS=1",
        "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS=1")),
    Case("small-input-stays-serial", candidate_env=(
        "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS=1",)),
    Case("worker-cap-below-forced-count-is-forced", candidate_env=(
        "PDG_NATIVE_WORKERS=2",
        "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=6",
        "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS=6")),
    # D0260/B0261: pinned PDAL's statistical outlier neither invalidates nor
    # publishes a PointView KD3 product. A coordinate mutator between the
    # outlier and a later KD3 consumer must therefore leave that consumer a
    # fresh tree, and a mutator between an earlier producer and the outlier
    # must leave the earlier (stale) product for later consumers, exactly as
    # pinned PDAL does. These compare against the pinned oracle because the
    # forked host CLI shares the candidate's library.
    # The consumer is filters.nndistance (kth, k=7) writing NNDistance through
    # extra_dims=all: any neighbor-set difference changes emitted doubles,
    # whereas the 144-point vote can coincide by majority.
    Case("outlier-mutator-nndistance-pinned", mutator="X = X * 3",
         consumer="nndistance", mutator_after_outlier=True,
         pinned_oracle=True, side=60),
    Case("normal-mutator-outlier-nndistance-pinned", mutator="X = X * 3",
         consumer="nndistance", precache_then_mutate_xyz=True,
         pinned_oracle=True, side=60),
    Case("outlier-mutator-classifier-pinned", mutator="X = X * 3",
         domain="Classification[2:2]", mutator_after_outlier=True,
         pinned_oracle=True, side=60),
    # D0263: published cached KD3 products refresh their snapshot at reuse.
    # A producer's tree reused after a coordinate write, after a point-order
    # mutation, and by a second consumer must all match pinned nanoflann's
    # stale-tree/live-coordinate behavior; the test-only snapshot verifier is
    # armed so any mutation path the epoch misses fails closed.
    Case("normal-assign-nndistance-pinned", producer="normal",
         mutator="X = X * 3", consumer="nndistance", pinned_oracle=True,
         side=60, candidate_env=("PDAL_TEST_VERIFY_KD3_SNAPSHOT=1",)),
    Case("normal-sort-nndistance-pinned", producer="normal",
         mutator="sort", consumer="nndistance", pinned_oracle=True,
         side=60, candidate_env=("PDAL_TEST_VERIFY_KD3_SNAPSHOT=1",)),
    Case("nndistance-assign-classifier-pinned", producer="nndistance",
         mutator="X = X * 3", domain="Classification[2:2]",
         pinned_oracle=True, side=60,
         candidate_env=("PDAL_TEST_VERIFY_KD3_SNAPSHOT=1",)),
    Case("normal-nndistance-classifier-reuse-pinned", producer="normal",
         consumer="nndistance", chain_classifier=True,
         domain="Classification[2:2]", pinned_oracle=True, side=60,
         candidate_env=("PDAL_TEST_VERIFY_KD3_SNAPSHOT=1",)),
    Case("headline-pinned", pinned_oracle=True, side=60,
         candidate_env=("PDAL_TEST_VERIFY_KD3_SNAPSHOT=1",)),
    # The default LAZ writer now compresses chunks on four workers (D0258);
    # the tiny single-chunk output must still be exact and the candidate must
    # observe that default without any launcher arming.
    Case("default-four-compression-workers", candidate_env=(
        "PDAL_TEST_REQUIRE_LAZ_COMPRESSION_THREADS=4",)),
    Case("required-count-mismatch-fails-closed", expected_returncode=1,
         candidate_only_failure=True, candidate_env=(
             "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS=3",
             "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS=4")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--freeze-epoch", type=int, default=1_704_067_200)
    parser.add_argument(
        "--pinned-oracle", type=Path,
        help="the pinned upstream PDAL executable; cases marked "
             "pinned_oracle compare against it instead of the forked host "
             "CLI, which shares the candidate's library and cannot expose "
             "fork-side upstream-code divergences",
    )
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


def build_fixture(root: Path, args: argparse.Namespace,
                  side: int = 12) -> Path:
    # A side x side lattice with periodic spikes and ground labels. The
    # 12-point side is the historical 144-point contract; the pinned-oracle
    # mutator cases use a 60-point side (3,600 points) so nanoflann's
    # 100-point leaves form a multi-level tree whose stale pruning is
    # observable after a coordinate mutation.
    source = root / f"r11-input-{side}.csv"
    lines = [
        "X,Y,Z,Intensity,Classification,ReturnNumber,NumberOfReturns"
    ]
    for y in range(side):
        for x in range(side):
            point = y * side + x
            base = 100.0 + 0.02 * x + 0.01 * y
            elevation = base + (4.0 if point % 17 == 0 else 0.0)
            classification = 2 if point % 11 == 0 else 1
            lines.append(
                f"{x},{y},{elevation:.2f},{point},{classification},1,1")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    target = root / f"r11-input-{side}.laz"
    pipeline = root / f"r11-fixture-{side}.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.text", "filename": source.name},
        {"type": "writers.las", "filename": str(target),
         "minor_version": 4, "dataformat_id": 6,
         "scale_x": 0.01, "scale_y": 0.01, "scale_z": 0.01,
         "compression": True},
    ]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [str(args.oracle.resolve()), "pipeline", str(pipeline.resolve())],
        cwd=root, env=oracle_environment(args), capture_output=True,
        check=False,
    )
    if completed.returncode or not target.is_file():
        print(
            "unable to build r11 fixture: " +
            completed.stderr.decode("utf-8", errors="replace")[:1024],
            file=sys.stderr,
        )
        raise RuntimeError("fixture generation failed")
    return target


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


SKIPPED = Path("skipped")


def consumer_stage(case: Case) -> dict[str, object]:
    if case.consumer == "nndistance":
        return {"type": "filters.nndistance", "mode": "kth", "k": case.k}
    stage: dict[str, object] = {"type": "filters.neighborclassifier",
                                "k": case.k}
    if case.domain:
        stage["domain"] = case.domain
    return stage


def writer_stage(case: Case) -> dict[str, object]:
    stage: dict[str, object] = {"type": "writers.las",
                                "filename": "output.laz",
                                "compression": "true"}
    if case.consumer == "nndistance" or case.producer is not None:
        stage["extra_dims"] = "all"
    return stage


def run_case(
    args: argparse.Namespace,
    root: Path,
    case: Case,
    fixtures: dict[int, Path],
    suffix: str,
) -> Path | None:
    fixture = fixtures[case.side]
    expected_points = case.side * case.side
    name = f"r11-classification-{case.name}{suffix}"
    pipeline = root / f"{name}.json"
    stages: list[dict[str, object]] = [
        {"type": "readers.las", "filename": "input.laz"},
    ]
    if case.producer is not None:
        # producer -> optional mutator -> consumer [-> classifier] -> writer;
        # no SMRF/outlier so the producer's published tree is the one reused.
        if case.producer == "normal":
            stages.append({"type": "filters.normal", "knn": 8})
        else:
            stages.append({"type": "filters.nndistance", "mode": "kth",
                           "k": 7})
        if case.mutator == "sort":
            stages.append({"type": "filters.sort", "dimension": "Z"})
        elif case.mutator is not None:
            stages.append({"type": "filters.assign", "value": case.mutator})
        stages.append(consumer_stage(case))
        if case.chain_classifier:
            stages.append({"type": "filters.neighborclassifier",
                           "k": case.k, "domain": case.domain})
        stages.append(writer_stage(case))
    elif case.precache_then_mutate_xyz:
        # Normal publishes a valid PointView KD3 product. Assign is an
        # upstream coordinate mutator that doesn't invalidate that product.
        # Statistical outlier must build its own fresh private tree, and a
        # later classifier must observe exactly what pinned PDAL leaves it.
        stages.extend([
            {"type": "filters.normal", "knn": 8},
            {"type": "filters.assign", "value": case.mutator or "X = 0"},
            {"type": "filters.outlier", "method": "statistical",
             "mean_k": case.mean_k, "multiplier": case.multiplier},
        ])
        if case.mutator is not None:
            stages.append(consumer_stage(case))
        stages.append(writer_stage(case))
    else:
        classifier: dict[str, object] = {
            "type": "filters.neighborclassifier", "k": case.k,
        }
        if case.domain:
            classifier["domain"] = case.domain
        if case.dimension is not None:
            classifier["dimension"] = case.dimension
        if case.candidate_file:
            classifier["candidate"] = "input.laz"
        stages.extend([
            {"type": "filters.smrf"},
            {"type": "filters.outlier", "method": "statistical",
             "mean_k": case.mean_k, "multiplier": case.multiplier},
        ])
        if case.mutator_after_outlier:
            stages.append({"type": "filters.assign", "value": case.mutator})
        if case.assign_gpstime:
            stages.append({"type": "filters.assign",
                           "value": "GpsTime = 3000000000"})
        stages.extend([
            classifier if case.consumer == "classifier" else
            consumer_stage(case),
            writer_stage(case),
        ])
    pipeline.write_text(json.dumps({"pipeline": stages}, indent=2,
                                   sort_keys=True) + "\n", encoding="utf-8")

    oracle = args.oracle
    extra: list[str] = []
    if case.pinned_oracle:
        if args.pinned_oracle is None:
            print(f"{name}: skipped, no --pinned-oracle configured",
                  file=sys.stderr)
            return SKIPPED
        oracle = args.pinned_oracle
        # The candidate must keep delegating to the forked sibling pdal (its
        # production default) while the comparison oracle is pinned PDAL;
        # otherwise a delegated route silently becomes the oracle.
        extra = ["--candidate-oracle", str(args.oracle.resolve())]
    command = [
        sys.executable,
        str(args.differential.resolve()),
        *extra,
        "--oracle", str(oracle.resolve()),
        "--candidate", str(args.candidate.resolve()),
        "--case", name,
        "--work-dir", str(args.work_dir.resolve()),
        "--frozen-time-library", str(args.frozen_time_library.resolve()),
        "--keep-success",
        "--seed-file", f"input.laz={fixture}",
        "--seed-file", f"pipeline.json={pipeline}",
    ]
    for assignment in case.candidate_env:
        command.extend(["--candidate-env", assignment])
    command.extend(["--", "pipeline", "pipeline.json"])
    completed = subprocess.run(command, check=False)
    report_path = args.work_dir / "reports" / f"{name}.json"
    if case.candidate_only_failure:
        # The oracle succeeds; the candidate must refuse with the required
        # status and publish nothing. This is the only case where the
        # differential itself is expected to report a difference.
        if not completed.returncode:
            print(f"{name}: candidate-only failure was not detected",
                  file=sys.stderr)
            return None
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["oracle_run"]["returncode"] != 0:
            print(f"{name}: oracle unexpectedly failed", file=sys.stderr)
            return None
        if report["candidate_run"]["returncode"] != case.expected_returncode:
            print(
                f"{name}: expected candidate status "
                f"{case.expected_returncode}, observed "
                f"{report['candidate_run']['returncode']}",
                file=sys.stderr,
            )
            return None
        if report["candidate_artifacts"].get("output.laz") is not None:
            print(f"{name}: candidate refusal published output.laz",
                  file=sys.stderr)
            return None
        if b"Required host neighborhood worker count" not in \
                report["candidate_run"]["stderr_preview"].encode("utf-8"):
            print(f"{name}: candidate refusal lacks the named diagnostic",
                  file=sys.stderr)
            return None
        return report_path
    if completed.returncode:
        return None

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
    if las_point_count(case_root / "oracle" / "output.laz") != expected_points:
        print(f"{name}: oracle output point count differs", file=sys.stderr)
        return None
    if (las_point_count(case_root / "candidate" / "output.laz") !=
            expected_points):
        print(f"{name}: candidate output point count differs", file=sys.stderr)
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
            prefix="pdg-r11-classification-matrix-",
            dir=args.work_dir) as temporary:
        root = Path(temporary)
        try:
            fixtures = {side: build_fixture(root, args, side)
                        for side in sorted({case.side for case in CASES})}
        except RuntimeError:
            return 1
        skipped = 0
        for case in CASES:
            if case.repeat:
                first = run_case(args, root, case, fixtures, "-run-1")
                second = run_case(args, root, case, fixtures, "-run-2")
                if first is None or second is None:
                    return 1
                left = json.loads(first.read_text(encoding="utf-8"))
                right = json.loads(second.read_text(encoding="utf-8"))
                if (left["candidate_artifacts"] !=
                        right["candidate_artifacts"]):
                    print("r11 classification candidate is not deterministic",
                          file=sys.stderr)
                    return 1
            else:
                result = run_case(args, root, case, fixtures, "")
                if result is None:
                    return 1
                if result is SKIPPED:
                    skipped += 1

    print(f"exact r11 classification matrix: {len(CASES)} cases"
          f" ({skipped} pinned-oracle cases skipped)" if skipped else
          f"exact r11 classification matrix: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
