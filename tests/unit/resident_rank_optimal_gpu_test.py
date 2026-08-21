#!/usr/bin/env python3
"""Physical exact gate for the calibrated rank/optimal composition."""

import argparse
import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile


SKIP = 77
GIB = 1024 * 1024 * 1024


def run(command, environment=None):
    return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=environment)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_identical(left, right):
    offset = 0
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_block = left_stream.read(1024 * 1024)
            right_block = right_stream.read(1024 * 1024)
            if left_block == right_block:
                if not left_block:
                    return
                offset += len(left_block)
                continue
            common = min(len(left_block), len(right_block))
            for index in range(common):
                if left_block[index] != right_block[index]:
                    raise AssertionError(
                        f"first output difference at byte {offset + index}: "
                        f"candidate=0x{left_block[index]:02x}, "
                        f"oracle=0x{right_block[index]:02x}")
            raise AssertionError(
                f"output sizes differ after byte {offset + common}: "
                f"candidate={left.stat().st_size}, oracle={right.stat().st_size}")


def las_point_count(path):
    with path.open("rb") as stream:
        header = stream.read(255)
    if len(header) < 227 or header[:4] != b"LASF":
        raise AssertionError(f"not a LAS file: {path}")
    if header[104] & 0x80:
        raise AssertionError("physical gate requires uncompressed LAS")
    legacy = struct.unpack_from("<I", header, 107)[0]
    extended = struct.unpack_from("<Q", header, 247)[0]
    return extended or legacy


def mem_available_bytes():
    for line in pathlib.Path("/proc/meminfo").read_text(
            encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("MemAvailable missing from /proc/meminfo")


def write_pipeline(path, source, destination, mutate=None):
    stages = [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.estimaterank", "knn": 14, "thresh": 0.01},
        {"type": "filters.optimalneighborhood", "min_k": 10, "max_k": 14},
        {"type": "filters.assign", "value": [
            "Classification = Rank",
            "Intensity = OptimalKNN",
            "PointSourceId = OptimalRadius",
        ]},
        {"type": "writers.las", "filename": str(destination)},
    ]
    if mutate:
        mutate(stages)
    path.write_text(json.dumps({"pipeline": stages}), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdg", type=pathlib.Path)
    parser.add_argument("oracle", type=pathlib.Path)
    args = parser.parse_args()

    configured = os.environ.get("PDG_RANK_OPTIMAL_LAS_FILE")
    expected_hash = os.environ.get("PDG_RANK_OPTIMAL_LAS_SHA256")
    if not configured or not expected_hash:
        print("set PDG_RANK_OPTIMAL_LAS_FILE and "
              "PDG_RANK_OPTIMAL_LAS_SHA256 for the physical gate",
              file=sys.stderr)
        return SKIP
    source = pathlib.Path(configured).resolve()
    if not source.is_file() or source.suffix.lower() != ".las":
        raise AssertionError("PDG_RANK_OPTIMAL_LAS_FILE must name a LAS file")
    if sha256(source) != expected_hash.lower():
        raise AssertionError("physical rank/optimal source hash mismatch")
    points = las_point_count(source)
    if not 250_000 <= points <= 16_000_000:
        raise AssertionError("physical source is outside calibrated cardinality")
    if mem_available_bytes() < 8 * GIB:
        print("rank/optimal gate skipped below 8 GiB MemAvailable",
              file=sys.stderr)
        return SKIP
    gpu = run(["nvidia-smi", "--query-gpu=memory.free",
               "--format=csv,noheader,nounits"])
    if gpu.returncode != 0 or int(gpu.stdout.splitlines()[0]) < 2048:
        print("rank/optimal gate skipped without 2 GiB free VRAM",
              file=sys.stderr)
        return SKIP

    with tempfile.TemporaryDirectory(prefix="pdg-rank-optimal-") as temporary:
        root = pathlib.Path(temporary)
        candidate_output = root / "candidate.las"
        oracle_output = root / "oracle.las"
        stats_path = root / "stats.json"
        candidate_pipeline = write_pipeline(
            root / "candidate.json", source, candidate_output)
        oracle_pipeline = write_pipeline(
            root / "oracle.json", source, oracle_output)
        environment = os.environ.copy()
        environment["PDG_REQUIRE_NEIGHBORHOOD_REUSE"] = "1"
        candidate = run([
            str(args.pdg), "resident", str(candidate_pipeline),
            "--stats", str(stats_path),
        ], environment)
        reference = run([
            str(args.oracle), "pipeline", str(oracle_pipeline),
        ], environment)
        assert candidate.returncode == reference.returncode == 0, (
            candidate, reference)
        assert candidate.stdout == reference.stdout, (candidate, reference)
        assert candidate.stderr == reference.stderr, (candidate, reference)
        assert_identical(candidate_output, oracle_output)

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        placement = stats["placement"]
        assert placement["available"] is True, placement
        assert placement["choice"] == "device", placement
        assert placement["selected_region_count"] == 1, placement
        execution = stats["execution"]
        assert execution["executor"] == "planner_resident_shared_index", (
            execution)
        assert execution["direct_las_output"] is False, execution
        # B0079 replaces the forced rows with this executor's complete
        # resident-mode ladder, so the provenance diagnostic must now match.
        assert execution[
            "selected_device_calibration_matches_executor"] is True, execution
        assert execution["rewrite_executable"] is True, execution
        assert execution["resident_preflight"]["accepted"] is True, execution
        assert execution["selected_regions"] == [0], execution
        assert execution["selected_stage_ids"] == [1, 2, 3], execution
        assert execution["index_builds"] == {
            "predicted": 1, "observed": 1, "matches_prediction": True,
        }, execution["index_builds"]
        kinds = [event["kind"] for event in execution["events"]]
        assert kinds.count("index_build") == 1, kinds
        assert kinds.count("device_region_begin") == 1, kinds
        assert kinds.count("device_region_end") == 1, kinds

        automatic_output = root / "automatic.las"
        automatic_pipeline = write_pipeline(
            root / "automatic.json", source, automatic_output)
        automatic_environment = environment.copy()
        automatic_environment[
            "PDG_REQUIRE_AUTOMATIC_RANK_OPTIMAL_RESIDENT"] = "1"
        automatic = run([
            str(args.pdg), "pipeline", str(automatic_pipeline),
        ], automatic_environment)
        assert automatic.returncode == reference.returncode == 0, (
            automatic, reference)
        assert automatic.stdout == reference.stdout, (automatic, reference)
        assert automatic.stderr == reference.stderr, (automatic, reference)
        assert_identical(automatic_output, oracle_output)

        option_free_output = root / "option-free.las"
        option_free_pipeline = write_pipeline(
            root / "option-free.json", source, option_free_output)
        option_free = run([
            str(args.pdg), "pipeline", str(option_free_pipeline),
        ], os.environ.copy())
        assert option_free.returncode == reference.returncode == 0, (
            option_free, reference)
        assert option_free.stdout == reference.stdout, (option_free, reference)
        assert option_free.stderr == reference.stderr, (option_free, reference)
        assert_identical(option_free_output, oracle_output)

        def assert_rejected(name, mutate=None, inject_preflight=False):
            rejected_output = root / f"{name}.las"
            rejected_pipeline = write_pipeline(
                root / f"{name}.json", source, rejected_output, mutate)
            rejected_environment = automatic_environment.copy()
            if inject_preflight:
                rejected_environment[
                    "PDG_TEST_AUTOMATIC_RESIDENT_PREFLIGHT_FAILURE"] = "1"
            rejected = run([
                str(args.pdg), "pipeline", str(rejected_pipeline),
            ], rejected_environment)
            assert rejected.returncode == 124, rejected
            assert ("required automatic rank/optimal resident path was not "
                    "used") in rejected.stderr, rejected.stderr
            assert not rejected_output.exists(), rejected_output

        assert_rejected(
            "changed-estimate-knn",
            lambda stages: stages[1].__setitem__("knn", 13))
        assert_rejected(
            "changed-threshold",
            lambda stages: stages[1].__setitem__("thresh", 0.02))
        assert_rejected(
            "changed-range",
            lambda stages: stages[2].__setitem__("min_k", 9))
        assert_rejected(
            "reordered-assignment",
            lambda stages: stages[3]["value"].reverse())
        assert_rejected(
            "extra-stage",
            lambda stages: stages.insert(
                4, {"type": "filters.ferry",
                    "dimensions": "Classification=>UserData"}))
        assert_rejected("injected-preflight", inject_preflight=True)

        print(json.dumps({
            "points": points,
            "output_bytes": candidate_output.stat().st_size,
            "output_sha256": sha256(candidate_output),
            "automatic_output_sha256": sha256(automatic_output),
            "option_free_output_sha256": sha256(option_free_output),
            "warning_sha256": hashlib.sha256(
                candidate.stderr.encode("utf-8")).hexdigest(),
            "index_builds": execution["index_builds"],
            "calibration_matches_executor": execution[
                "selected_device_calibration_matches_executor"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
