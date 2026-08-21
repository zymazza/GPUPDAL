#!/usr/bin/env python3
"""Exact process gate for the opt-in resident radial-density composition."""

import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile


SKIP = 77


def run(command, environment=None):
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
    )


def write_pipeline(path, source, destination, radius=1.01,
                   assignment="UserData = 1 WHERE RadialDensity >= 0.2"):
    path.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.radialdensity", "radius": radius},
        {"type": "filters.assign", "value": assignment},
        {"type": "writers.las", "filename": str(destination)},
    ]}), encoding="utf-8")
    return path


def exact_environment():
    environment = os.environ.copy()
    environment["PDG_EXPERIMENTAL_DIRECT_RESIDENT_LAS_OUTPUT"] = "1"
    environment["PDG_EXPERIMENTAL_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
    environment["PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT"] = "1"
    environment["PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
    environment["PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY"] = "1"
    environment["PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ"] = "1"
    environment["PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE"] = "1"
    return environment


def las_layout(path):
    data = path.read_bytes()
    assert data[:4] == b"LASF" and len(data) >= 227, path
    offset = struct.unpack_from("<I", data, 96)[0]
    assert not (data[104] & 0x80), "compressed LAS is outside this gate"
    stride = struct.unpack_from("<H", data, 105)[0]
    count = (struct.unpack_from("<Q", data, 247)[0]
             if data[25] >= 4 else struct.unpack_from("<I", data, 107)[0])
    assert offset + count * stride <= len(data), path
    return data, offset, stride, count


def user_data(path):
    data, offset, stride, count = las_layout(path)
    return [data[offset + point * stride + 17] for point in range(count)]


def rejected_case(root, name, pdg, source, environment, **options):
    output = root / f"{name}.las"
    pipeline = write_pipeline(root / f"{name}.json", source, output,
                              **options)
    result = run([str(pdg), "resident", str(pipeline)], environment)
    assert result.returncode != 0, (name, result)
    assert not output.exists(), (name, output)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: resident_radialdensity_direct_gpu_test.py PDG PDAL INPUT")
    pdg = pathlib.Path(sys.argv[1]).resolve()
    oracle = pathlib.Path(sys.argv[2]).resolve()
    source = pathlib.Path(sys.argv[3]).resolve()

    if shutil.which("nvidia-smi") is None:
        print("radial-density resident gate skipped without nvidia-smi",
              file=sys.stderr)
        return SKIP
    gpu = run(["nvidia-smi", "--query-gpu=memory.free",
               "--format=csv,noheader,nounits"])
    if gpu.returncode != 0:
        print("radial-density resident gate skipped without a GPU",
              file=sys.stderr)
        return SKIP

    with tempfile.TemporaryDirectory(
            prefix="pdg-resident-radialdensity-direct-") as temporary:
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
        stats_path = root / "candidate-stats.json"
        candidate_pipeline = write_pipeline(
            root / "candidate.json", source, candidate_output)
        oracle_pipeline = write_pipeline(
            root / "oracle.json", source, oracle_output)
        candidate = run([
            str(pdg), "resident", str(candidate_pipeline), "--stats",
            str(stats_path),
        ], exact_environment())
        reference = run([str(oracle), "pipeline", str(oracle_pipeline)])
        assert candidate.returncode == reference.returncode == 0, (
            candidate.returncode, candidate.stderr,
            reference.returncode, reference.stderr)
        assert candidate.stdout == reference.stdout
        assert candidate.stderr == reference.stderr
        assert candidate_output.read_bytes() == oracle_output.read_bytes()
        assert user_data(candidate_output) != user_data(source), (
            "fixture must positively exercise UserData publication")

        _, _, record_bytes, point_count = las_layout(source)
        report = json.loads(stats_path.read_text(encoding="utf-8"))
        execution = report["execution"]
        assert execution["executor"] == (
            "planner_resident_shared_index_direct_las"), execution
        assert execution["direct_las_output"] is True, execution
        assert execution["direct_las_resident_source"] is True, execution
        assert execution["direct_las_record_summary"] is True, execution
        assert execution["direct_las_host_xyz_mirror"] is False, execution
        assert execution["selected_regions"] == [0], execution
        assert execution["selected_stage_ids"] == [1, 2], execution
        assert execution["resident_preflight"]["accepted"] is True, execution
        assert execution["index_builds"] == {
            "predicted": 1, "observed": 1, "matches_prediction": True,
        }, execution["index_builds"]
        assert execution["boundary_accounting_matches_prediction"] is True, (
            execution)
        assert execution["totals"]["host_to_device"]["bytes"] == (
            (record_bytes + 1) * point_count), execution["totals"]
        assert execution["totals"]["device_to_host"]["bytes"] == (
            point_count), execution["totals"]
        assert execution["selected_device_calibration_matches_executor"] is (
            False), execution
        assert report["placement"]["reason"] == "uncalibrated_stage", report

        rejected_case(root, "changed-radius", pdg, source,
                      exact_environment(), radius=1.02)
        rejected_case(
            root, "changed-assignment", pdg, source, exact_environment(),
            assignment="UserData = 1 WHERE RadialDensity >= 1.0")
        disabled = exact_environment()
        disabled["PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
        rejected_case(root, "disabled-source", pdg, source, disabled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
