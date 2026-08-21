#!/usr/bin/env python3
"""Exact process gates for direct-LAS neighborclassifier execution."""

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
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
    )


def write_pipeline(path, source, destination, k=7, writer_options=None):
    writer = {"type": "writers.las", "filename": str(destination)}
    writer.update(writer_options or {})
    path.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.neighborclassifier", "k": k},
        writer,
    ]}), encoding="utf-8")
    return path


def exact_environment():
    environment = os.environ.copy()
    environment["PDG_EXPERIMENTAL_DIRECT_CLASSIFICATION_OUTPUT"] = "1"
    environment["PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT"] = "1"
    environment["PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
    environment["PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY"] = "1"
    environment["PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ"] = "1"
    return environment


def automatic_environment():
    environment = os.environ.copy()
    environment["PDG_REQUIRE_AUTOMATIC_NEIGHBORCLASSIFIER_RESIDENT"] = "1"
    return environment


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
    compressed = bool(data[104] & 0x80)
    stride = struct.unpack_from("<H", data, 105)[0]
    count = (struct.unpack_from("<Q", data, 247)[0]
             if data[25] >= 4 else struct.unpack_from("<I", data, 107)[0])
    if not compressed:
        assert point_offset + count * stride <= len(data), path
    return point_format, stride, count, compressed


def mem_available_bytes():
    for line in pathlib.Path("/proc/meminfo").read_text(
            encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("MemAvailable missing from /proc/meminfo")


def prefix_format7(root, oracle, source, count):
    output = root / f"prefix-{count}.las"
    pipeline = root / f"prefix-{count}.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.head", "count": count},
        {"type": "writers.las", "filename": str(output),
         "minor_version": 4, "dataformat_id": 7},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    assert las_layout(output) == (7, 36, count, False), las_layout(output)
    return output


def convert_format(root, oracle, source, point_format):
    output = root / f"format-{point_format}.las"
    pipeline = root / f"format-{point_format}.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "writers.las", "filename": str(output),
         "minor_version": 4, "dataformat_id": point_format},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    assert las_layout(output)[0] == point_format, las_layout(output)
    return output


def automatic_case(root, name, pdg, oracle, source, exercise_failures=False):
    candidate_output = root / f"{name}-candidate.las"
    option_free_output = root / f"{name}-option-free.las"
    oracle_output = root / f"{name}-oracle.las"
    candidate_pipeline = write_pipeline(
        root / f"{name}-candidate.json", source, candidate_output)
    option_free_pipeline = write_pipeline(
        root / f"{name}-option-free.json", source, option_free_output)
    oracle_pipeline = write_pipeline(
        root / f"{name}-oracle.json", source, oracle_output)
    candidate = run(
        [str(pdg), "pipeline", str(candidate_pipeline)],
        automatic_environment())
    option_free = run([str(pdg), "pipeline", str(option_free_pipeline)])
    reference = run([str(oracle), "pipeline", str(oracle_pipeline)])
    assert candidate.returncode == option_free.returncode == (
        reference.returncode) == 0, (
            candidate.returncode, candidate.stderr,
            option_free.returncode, option_free.stderr,
            reference.returncode, reference.stderr)
    assert candidate.stdout == option_free.stdout == reference.stdout
    assert candidate.stderr == option_free.stderr == reference.stderr
    assert_identical(candidate_output, oracle_output)
    assert_identical(option_free_output, oracle_output)
    assert classifications(candidate_output) != classifications(source), (
        "fixture must positively exercise automatic Classification output")

    if not exercise_failures:
        return
    for case, variable, expected_status in (
        ("disabled-source", "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE", 124),
        ("preflight", "PDG_TEST_AUTOMATIC_RESIDENT_PREFLIGHT_FAILURE", 124),
        ("proof", "PDG_TEST_AUTOMATIC_NEIGHBORCLASSIFIER_PROOF_FAILURE",
         1),
    ):
        output = root / f"{case}.las"
        pipeline = write_pipeline(root / f"{case}.json", source, output)
        environment = automatic_environment()
        environment[variable] = "1"
        result = run([str(pdg), "pipeline", str(pipeline)], environment)
        assert result.returncode == expected_status, (case, result)
        if case == "proof":
            expected_stderr = (
                "PDAL: automatic neighborclassifier resident execution proof "
                "failed\n\n")
            assert result.stderr == expected_stderr, (case, result.stderr)
        assert not output.exists(), (case, output)


def classifications(path):
    data = path.read_bytes()
    assert data[:4] == b"LASF" and len(data) >= 227, path
    point_offset = struct.unpack_from("<I", data, 96)[0]
    point_format = data[104] & 0x0f
    assert not (data[104] & 0x80), "compressed LAS is outside this gate"
    stride = struct.unpack_from("<H", data, 105)[0]
    count = (struct.unpack_from("<Q", data, 247)[0]
             if data[25] >= 4
             else struct.unpack_from("<I", data, 107)[0])
    assert point_offset + count * stride <= len(data), path
    result = []
    for point in range(count):
        record = point_offset + point * stride
        value = data[record + (15 if point_format <= 5 else 16)]
        if point_format <= 5:
            value &= 0x1f
            if value == 12:
                value = 0
        result.append(value)
    return result


def automatic_main(pdg, oracle):
    configured = os.environ.get("PDG_NEIGHBORCLASSIFIER_LAS_FILE")
    expected_hash = os.environ.get("PDG_NEIGHBORCLASSIFIER_LAS_SHA256")
    if not configured or not expected_hash:
        print("set PDG_NEIGHBORCLASSIFIER_LAS_FILE and "
              "PDG_NEIGHBORCLASSIFIER_LAS_SHA256 for the automatic gate",
              file=sys.stderr)
        return SKIP
    source = pathlib.Path(configured).resolve()
    if not source.is_file() or source.suffix.lower() != ".las":
        raise AssertionError(
            "PDG_NEIGHBORCLASSIFIER_LAS_FILE must name a LAS file")
    if sha256(source) != expected_hash.lower():
        raise AssertionError("physical neighborclassifier source hash mismatch")
    if las_layout(source) != (7, 36, 1_000_000, False):
        raise AssertionError(
            "physical neighborclassifier source must be uncompressed "
            "format-7/36-byte LAS with exactly 1M points")
    if mem_available_bytes() < 8 * GIB:
        print("neighborclassifier gate skipped below 8 GiB MemAvailable",
              file=sys.stderr)
        return SKIP
    gpu = run(["nvidia-smi", "--query-gpu=memory.free",
               "--format=csv,noheader,nounits"])
    if gpu.returncode != 0 or int(gpu.stdout.splitlines()[0]) < 2048:
        print("neighborclassifier gate skipped without 2 GiB free VRAM",
              file=sys.stderr)
        return SKIP

    with tempfile.TemporaryDirectory(
            prefix="pdg-neighborclassifier-automatic-") as temporary:
        root = pathlib.Path(temporary)
        below_source = prefix_format7(root, oracle, source, 50_000)
        below_output = root / "below.las"
        below_pipeline = write_pipeline(
            root / "below.json", below_source, below_output)
        below = run([str(pdg), "pipeline", str(below_pipeline)],
                    automatic_environment())
        assert below.returncode == 124, below
        assert not below_output.exists(), below_output

        floor_source = prefix_format7(root, oracle, source, 250_000)
        automatic_case(root, "floor", pdg, oracle, floor_source)
        automatic_case(root, "main", pdg, oracle, source,
                       exercise_failures=True)

        wrong_layout = convert_format(root, oracle, source, 3)
        for case, rejected_source, k, writer_options in (
            ("wrong-layout", wrong_layout, 7, None),
            ("changed-k", source, 8, None),
            ("compressed-writer", source, 7, {"compression": True}),
            ("writer-option", source, 7, {"forward": "all"}),
        ):
            suffix = ".laz" if case == "compressed-writer" else ".las"
            output = root / f"{case}{suffix}"
            pipeline = write_pipeline(root / f"{case}.json", rejected_source,
                                      output, k, writer_options)
            result = run([str(pdg), "pipeline", str(pipeline)],
                         automatic_environment())
            assert result.returncode == 124, (case, result)
            assert not output.exists(), (case, output)
    return 0


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--automatic":
        return automatic_main(pathlib.Path(sys.argv[2]).resolve(),
                              pathlib.Path(sys.argv[3]).resolve())
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: resident_neighborclassifier_direct_gpu_test.py "
            "PDG PDAL INPUT\n"
            "   or: resident_neighborclassifier_direct_gpu_test.py "
            "--automatic PDG PDAL")
    pdg = pathlib.Path(sys.argv[1]).resolve()
    oracle = pathlib.Path(sys.argv[2]).resolve()
    source = pathlib.Path(sys.argv[3]).resolve()

    with tempfile.TemporaryDirectory(
            prefix="pdg-resident-neighborclassifier-direct-") as temporary:
        root = pathlib.Path(temporary)
        probe_output = root / "probe.las"
        probe_stats = root / "probe-stats.json"
        probe_pipeline = write_pipeline(
            root / "probe.json", source, probe_output)
        probe = run([
            str(pdg), "resident", str(probe_pipeline), "--stats",
            str(probe_stats),
        ])
        assert probe.returncode == 0, probe.stderr
        placement = json.loads(
            probe_stats.read_text(encoding="utf-8"))["placement"]
        if (not placement["available"] and
                placement["unavailable_reason"] == "profile_not_exact"):
            print("exact resident placement profile is unavailable",
                  file=sys.stderr)
            return SKIP

        candidate_output = root / "candidate.las"
        oracle_output = root / "oracle.las"
        candidate_stats = root / "candidate-stats.json"
        candidate_pipeline = write_pipeline(
            root / "candidate.json", source, candidate_output)
        oracle_pipeline = write_pipeline(
            root / "oracle.json", source, oracle_output)
        candidate = run([
            str(pdg), "resident", str(candidate_pipeline), "--stats",
            str(candidate_stats),
        ], exact_environment())
        reference = run([
            str(oracle), "pipeline", str(oracle_pipeline),
        ])
        assert candidate.returncode == reference.returncode == 0, (
            candidate.returncode, candidate.stderr,
            reference.returncode, reference.stderr)
        assert candidate.stdout == reference.stdout
        assert candidate.stderr == reference.stderr
        assert candidate_output.read_bytes() == oracle_output.read_bytes()
        assert classifications(candidate_output) != classifications(source), (
            "fixture must positively exercise Classification publication")

        report = json.loads(candidate_stats.read_text(encoding="utf-8"))
        execution = report["execution"]
        assert execution["executor"] == (
            "planner_resident_shared_index_direct_las"), execution
        assert execution["direct_las_output"] is True, execution
        assert execution["direct_las_resident_source"] is True, execution
        assert execution["direct_las_record_summary"] is True, execution
        assert execution["direct_las_host_xyz_mirror"] is False, execution
        assert execution[
            "selected_device_calibration_matches_executor"] is False, (
                execution)
        assert report["placement"]["reason"] == "uncalibrated_stage", report
        assert execution["index_builds"] == {
            "matches_prediction": True,
            "observed": 1,
            "predicted": 1,
        }, execution["index_builds"]
        repair = execution["exact_host_repair"]["neighborclassifier"]
        assert repair == {
            "ambiguous_rows": 0,
            "incomplete_rows": 0,
            "kd3_used": False,
            "kd3_uses": 0,
            "repaired_rows": 0,
            "seconds": 0.0,
        }, repair

        outside_output = root / "outside.las"
        outside_pipeline = write_pipeline(
            root / "outside.json", source, outside_output, k=8)
        outside = run([
            str(pdg), "resident", str(outside_pipeline),
        ], exact_environment())
        assert outside.returncode != 0, outside
        assert not outside_output.exists()

        disabled_output = root / "disabled.las"
        disabled_pipeline = write_pipeline(
            root / "disabled.json", source, disabled_output)
        disabled_environment = exact_environment()
        disabled_environment["PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
        disabled = run([
            str(pdg), "resident", str(disabled_pipeline),
        ], disabled_environment)
        assert disabled.returncode != 0, disabled
        assert "required direct LAS resident source path was not used" in (
            disabled.stderr), disabled.stderr
        assert not disabled_output.exists()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
