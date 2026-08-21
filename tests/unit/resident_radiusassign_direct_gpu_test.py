#!/usr/bin/env python3
"""Physical exact gate for the opt-in direct-LAS radiusassign endpoint."""

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
                          stderr=subprocess.PIPE, env=environment,
                          check=False)


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
                f"candidate={left.stat().st_size}, "
                f"oracle={right.stat().st_size}")


def las_layout(path):
    data = path.read_bytes()
    assert data[:4] == b"LASF" and len(data) >= 227, path
    point_offset = struct.unpack_from("<I", data, 96)[0]
    point_format = data[104] & 0x0f
    assert not (data[104] & 0x80), "physical gate requires uncompressed LAS"
    point_stride = struct.unpack_from("<H", data, 105)[0]
    point_count = (struct.unpack_from("<Q", data, 247)[0]
                   if data[25] >= 4 else struct.unpack_from("<I", data, 107)[0])
    assert point_offset + point_count * point_stride <= len(data), path
    return data, point_offset, point_format, point_stride, point_count


def user_data_changed(source, output):
    source_data, source_offset, source_format, source_stride, source_count = (
        las_layout(source))
    output_data, output_offset, output_format, output_stride, output_count = (
        las_layout(output))
    assert source_count == output_count, (
        source_format, output_format, source_count, output_count)
    return any(
        source_data[source_offset + point * source_stride + 17] !=
        output_data[output_offset + point * output_stride + 17]
        for point in range(source_count))


def mem_available_bytes():
    for line in pathlib.Path("/proc/meminfo").read_text(
            encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("MemAvailable missing from /proc/meminfo")


def write_pipeline(path, source, destination, radius=2.0,
                   source_domain="ReturnNumber[1:1]"):
    path.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.radiusassign", "radius": radius, "is3d": True,
         "src_domain": source_domain,
         "reference_domain": "ReturnNumber[2:15]",
         "update_expression": "UserData = 9"},
        {"type": "writers.las", "filename": str(destination)},
    ]}), encoding="utf-8")
    return path


def exact_environment():
    environment = os.environ.copy()
    environment["PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT"] = "1"
    environment["PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
    environment["PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY"] = "1"
    environment["PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ"] = "1"
    return environment


def automatic_environment():
    environment = os.environ.copy()
    environment["PDG_REQUIRE_AUTOMATIC_RADIUSASSIGN_RESIDENT"] = "1"
    return environment


def convert_format(root, oracle, source, point_format):
    converted = root / f"source-format{point_format}.las"
    pipeline = root / f"convert-format{point_format}.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "writers.las", "filename": str(converted),
         "minor_version": 4, "dataformat_id": point_format},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    assert las_layout(converted)[2] == point_format
    return converted


def prefix_format7(root, oracle, source, count):
    prefix = root / f"automatic-prefix-{count}.las"
    pipeline = root / f"automatic-prefix-{count}.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.head", "count": count},
        {"type": "writers.las", "filename": str(prefix),
         "minor_version": 4, "dataformat_id": 7},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    assert las_layout(prefix)[2:] == (7, 36, count), las_layout(prefix)[2:]
    return prefix


def automatic_case(root, name, pdg, oracle, source, exercise_failures=False):
    candidate_output = root / f"automatic-{name}-candidate.las"
    option_free_output = root / f"automatic-{name}-option-free.las"
    oracle_output = root / f"automatic-{name}-oracle.las"
    candidate_pipeline = write_pipeline(
        root / f"automatic-{name}-candidate.json", source, candidate_output)
    option_free_pipeline = write_pipeline(
        root / f"automatic-{name}-option-free.json", source,
        option_free_output)
    oracle_pipeline = write_pipeline(
        root / f"automatic-{name}-oracle.json", source, oracle_output)
    candidate = run([
        str(pdg), "pipeline", str(candidate_pipeline),
    ], automatic_environment())
    option_free = run([
        str(pdg), "pipeline", str(option_free_pipeline),
    ])
    reference = run([
        str(oracle), "pipeline", str(oracle_pipeline),
    ])
    assert candidate.returncode == option_free.returncode == (
        reference.returncode) == 0, (
            candidate.returncode, candidate.stderr,
            option_free.returncode, option_free.stderr,
            reference.returncode, reference.stderr)
    assert candidate.stdout == option_free.stdout == reference.stdout
    assert candidate.stderr == option_free.stderr == reference.stderr
    assert_identical(candidate_output, oracle_output)
    assert_identical(option_free_output, oracle_output)
    assert user_data_changed(source, candidate_output)

    if not exercise_failures:
        return
    for name, variable, expected_status in (
        ("disabled-source", "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE", 124),
        ("preflight-failure", "PDG_TEST_AUTOMATIC_RESIDENT_PREFLIGHT_FAILURE",
         124),
        ("proof-failure", "PDG_TEST_AUTOMATIC_RADIUSASSIGN_PROOF_FAILURE",
         None),
    ):
        output = root / f"automatic-{name}.las"
        pipeline = write_pipeline(
            root / f"automatic-{name}.json", source, output)
        environment = automatic_environment()
        environment[variable] = "1"
        result = run([str(pdg), "pipeline", str(pipeline)], environment)
        if expected_status is not None:
            assert result.returncode == expected_status, (name, result)
        else:
            assert result.returncode != 0, (name, result)
            assert "automatic radiusassign resident execution proof failed" in (
                result.stderr), (name, result.stderr)
        assert not output.exists(), (name, output)


def exact_case(root, name, pdg, oracle, source):
    _, _, _, point_stride, point_count = las_layout(source)
    candidate_output = root / f"{name}-candidate.las"
    oracle_output = root / f"{name}-oracle.las"
    stats_path = root / f"{name}-stats.json"
    candidate_pipeline = write_pipeline(
        root / f"{name}-candidate.json", source, candidate_output)
    oracle_pipeline = write_pipeline(
        root / f"{name}-oracle.json", source, oracle_output)
    candidate = run([
        str(pdg), "resident", str(candidate_pipeline),
        "--stats", str(stats_path),
    ], exact_environment())
    reference = run([str(oracle), "pipeline", str(oracle_pipeline)])
    assert candidate.returncode == reference.returncode == 0, (
        name, candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout, (
        name, candidate.stdout, reference.stdout)
    assert candidate.stderr == reference.stderr, (
        name, candidate.stderr, reference.stderr)
    assert_identical(candidate_output, oracle_output)
    assert user_data_changed(source, candidate_output), (
        f"{name} must positively exercise UserData output")

    report = json.loads(stats_path.read_text(encoding="utf-8"))
    placement = report["placement"]
    assert placement["available"] is True, placement
    assert placement["choice"] == "device", placement
    assert placement["selected_region_count"] == 1, placement
    assert placement["boundary_accounting_model"] == "executor_declared", (
        placement)
    assert placement["predicted"]["host_to_device_bytes"] == (
        25 * point_count), placement
    assert placement["predicted"]["device_to_host_bytes"] == point_count, (
        placement)
    assert placement["predicted"]["packing_bytes"] == 0, placement
    assert placement["predicted"]["configured_device_lane_count"] == 1, (
        placement)
    execution = report["execution"]
    assert execution["executor"] == (
        "planner_resident_shared_index_direct_las"), execution
    assert execution["direct_las_output"] is True, execution
    assert execution["direct_las_resident_source"] is True, execution
    assert execution["direct_las_record_summary"] is True, execution
    assert execution["direct_las_host_xyz_mirror"] is False, execution
    assert execution["selected_regions"] == [0], execution
    assert execution["selected_stage_ids"] == [1], execution
    assert execution["resident_preflight"]["accepted"] is True, execution
    schedule = execution["schedule"]
    assert schedule["item_count"] == point_count, schedule
    assert schedule["tile_item_capacity"] == point_count, schedule
    assert schedule["tile_count"] == 1, schedule
    assert schedule["configured_lane_count"] == 1, schedule
    assert schedule["active_lane_count"] == 1, schedule
    assert execution["index_builds"] == {
        "predicted": 1, "observed": 1, "matches_prediction": True,
    }, execution["index_builds"]
    assert execution["boundary_accounting_matches_prediction"] is True, (
        execution)
    # B0083 calibrated the default 36-byte format-7 record shape. The exact
    # format-3 companion still proves the direct boundary, but truthfully
    # retains the ordinary radiusassign model rather than borrowing that fit.
    assert execution["selected_device_calibration_matches_executor"] is (
        point_stride == 36), execution
    return candidate_output, execution


def rejected_case(root, name, pdg, source, environment, **pipeline_options):
    output = root / f"{name}.las"
    pipeline = write_pipeline(
        root / f"{name}.json", source, output, **pipeline_options)
    result = run([str(pdg), "resident", str(pipeline)], environment)
    assert result.returncode != 0, result
    assert not output.exists(), output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdg", type=pathlib.Path)
    parser.add_argument("oracle", type=pathlib.Path)
    args = parser.parse_args()

    configured = os.environ.get("PDG_RADIUSASSIGN_LAS_FILE")
    expected_hash = os.environ.get("PDG_RADIUSASSIGN_LAS_SHA256")
    if not configured or not expected_hash:
        print("set PDG_RADIUSASSIGN_LAS_FILE and "
              "PDG_RADIUSASSIGN_LAS_SHA256 for the physical gate",
              file=sys.stderr)
        return SKIP
    source = pathlib.Path(configured).resolve()
    if not source.is_file() or source.suffix.lower() != ".las":
        raise AssertionError("PDG_RADIUSASSIGN_LAS_FILE must name a LAS file")
    if sha256(source) != expected_hash.lower():
        raise AssertionError("physical radiusassign source hash mismatch")
    _, _, point_format, _, points = las_layout(source)
    if points != 1_000_000:
        raise AssertionError(
            "physical radiusassign source must contain exactly 1M points")
    if mem_available_bytes() < 8 * GIB:
        print("radiusassign gate skipped below 8 GiB MemAvailable",
              file=sys.stderr)
        return SKIP
    gpu = run(["nvidia-smi", "--query-gpu=memory.free",
               "--format=csv,noheader,nounits"])
    if gpu.returncode != 0 or int(gpu.stdout.splitlines()[0]) < 2048:
        print("radiusassign gate skipped without 2 GiB free VRAM",
              file=sys.stderr)
        return SKIP

    with tempfile.TemporaryDirectory(
            prefix="pdg-radiusassign-direct-") as temporary:
        root = pathlib.Path(temporary)
        formats = [(f"format{point_format}", source)]
        companion_format = 3 if point_format > 5 else 7
        formats.append((f"format{companion_format}", convert_format(
            root, args.oracle, source, companion_format)))
        outputs = []
        input_formats = []
        executions = []
        for name, exact_source in formats:
            input_formats.append(las_layout(exact_source)[2])
            output, execution = exact_case(
                root, name, args.pdg, args.oracle, exact_source)
            outputs.append(output)
            executions.append(execution)

        automatic_source = (
            source if las_layout(source)[3] == 36
            else convert_format(root, args.oracle, source, 7))
        below_source = prefix_format7(root, args.oracle, automatic_source,
                                      50_000)
        below_output = root / "automatic-below.las"
        below_pipeline = write_pipeline(
            root / "automatic-below.json", below_source, below_output)
        below = run([
            str(args.pdg), "pipeline", str(below_pipeline),
        ], automatic_environment())
        assert below.returncode == 124, below
        assert not below_output.exists()
        floor_source = prefix_format7(root, args.oracle, automatic_source,
                                      250_000)
        automatic_case(root, "floor-250k", args.pdg, args.oracle,
                       floor_source)
        automatic_case(root, "main", args.pdg, args.oracle,
                       automatic_source, exercise_failures=True)

        wrong_layout_source = next(
            item for _, item in formats if las_layout(item)[3] != 36)
        wrong_layout_output = root / "automatic-wrong-layout.las"
        wrong_layout_pipeline = write_pipeline(
            root / "automatic-wrong-layout.json", wrong_layout_source,
            wrong_layout_output)
        wrong_layout = run([
            str(args.pdg), "pipeline", str(wrong_layout_pipeline),
        ], automatic_environment())
        assert wrong_layout.returncode == 124, wrong_layout
        assert not wrong_layout_output.exists()

        drift_output = root / "automatic-drift.las"
        drift_pipeline = write_pipeline(
            root / "automatic-drift.json", automatic_source, drift_output,
            radius=2.5)
        drift = run([
            str(args.pdg), "pipeline", str(drift_pipeline),
        ], automatic_environment())
        assert drift.returncode == 124, drift
        assert not drift_output.exists()

        rejected_case(root, "changed-radius", args.pdg, source,
                      exact_environment(), radius=2.5)
        rejected_case(root, "changed-domain", args.pdg, source,
                      exact_environment(),
                      source_domain="ReturnNumber[1:2]")
        disabled_environment = exact_environment()
        disabled_environment["PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
        rejected_case(root, "disabled-source", args.pdg, source,
                      disabled_environment)

        print(json.dumps({
            "points": points,
            "input_formats": input_formats,
            "output_sha256": [sha256(item) for item in outputs],
            "index_builds": [item["index_builds"] for item in executions],
            "calibration_matches_executor": [
                item["selected_device_calibration_matches_executor"]
                for item in executions],
            "boundary_accounting_matches_prediction": [
                item["boundary_accounting_matches_prediction"]
                for item in executions],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
