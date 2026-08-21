#!/usr/bin/env python3
"""Exact process gate for the explicit one-binary64 direct LAS publisher."""

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


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def write_fixture_pipeline(path, destination):
    return write_json(path, {"pipeline": [
        {"type": "readers.faux",
         "bounds": "([0,7],[0,3],[0,1])", "mode": "grid"},
        {"type": "filters.assign", "value": [
            "Classification = 1",
            "Classification = 2 WHERE Y == 0",
        ]},
        {"type": "filters.assign", "value": "X = X + Y / 5"},
        {"type": "filters.assign", "value": "Z = X / 1000 + Y"},
        {"type": "writers.las", "filename": str(destination),
         "minor_version": 4, "dataformat_id": 7, "extra_dims": "all"},
    ]})


def write_wide_fixture_pipeline(path, destination):
    """A fixture with enough ground rows to admit the widest supported count.

    The shared 7x3 fixture only has seven ground rows, so counts above that
    legitimately fall back for insufficient ground. This grid keeps the same
    layout and the same Y / 5 query offset while providing 79 ground rows, so
    count 64 still has a distinct 65th candidate.
    """
    return write_json(path, {"pipeline": [
        {"type": "readers.faux",
         "bounds": "([0,79],[0,3],[0,1])", "mode": "grid"},
        {"type": "filters.assign", "value": [
            "Classification = 1",
            "Classification = 2 WHERE Y == 0",
        ]},
        {"type": "filters.assign", "value": "X = X + Y / 5"},
        {"type": "filters.assign", "value": "Z = X / 1000 + Y"},
        {"type": "writers.las", "filename": str(destination),
         "minor_version": 4, "dataformat_id": 7, "extra_dims": "all"},
    ]})


def write_hag_pipeline(path, source, destination, count=4,
                       extra_dims="all", producer_options=None):
    producer = {"type": "filters.hag_nn", "count": count}
    if producer_options:
        producer.update(producer_options)
    writer = {"type": "writers.las", "filename": str(destination)}
    if extra_dims is not None:
        writer["extra_dims"] = extra_dims
    return write_json(path, {"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        producer,
        writer,
    ]})


def write_hag_delaunay_fixture_pipeline(path, destination):
    return write_json(path, {"pipeline": [
        {"type": "readers.faux",
         "bounds": "([0,7],[0,3],[0,1])", "mode": "grid"},
        {"type": "filters.assign", "value": [
            "Classification = 1",
            "Classification = 2 WHERE Y == 0 || Y == 2 || Y == 3",
        ]},
        {"type": "filters.assign", "value": "X = X + Y * Y / 10"},
        {"type": "filters.assign", "value": "Z = X / 1000 + Y"},
        {"type": "writers.las", "filename": str(destination),
         "minor_version": 4, "dataformat_id": 7, "extra_dims": "all"},
    ]})


def write_hag_delaunay_rows_fixture_pipeline(path, csv, destination, rows):
    lines = ["X,Y,Z,Classification,OffsetTime\n"]
    lines.extend(
        f"{x},{y},{z},{classification},{offset_time}\n"
        for x, y, z, classification, offset_time in rows)
    csv.write_text("".join(lines), encoding="utf-8")
    return write_json(path, {"pipeline": [
        {"type": "readers.text", "filename": str(csv)},
        {"type": "writers.las", "filename": str(destination),
         "minor_version": 4, "dataformat_id": 7,
         "extra_dims": "OffsetTime=float"},
    ]})


def write_hag_delaunay_pipeline(path, source, destination, count=3,
                                allow_extrapolation=None,
                                extra_dims="all"):
    producer = {"type": "filters.hag_delaunay", "count": count}
    if allow_extrapolation is not None:
        producer["allow_extrapolation"] = allow_extrapolation
    writer = {"type": "writers.las", "filename": str(destination)}
    if extra_dims is not None:
        writer["extra_dims"] = extra_dims
    return write_json(path, {"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        producer,
        writer,
    ]})


def write_nndistance_pipeline(path, source, destination, k=10, mode="kth",
                              extra_dims="all"):
    writer = {"type": "writers.las", "filename": str(destination)}
    if extra_dims is not None:
        writer["extra_dims"] = extra_dims
    return write_json(path, {"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.nndistance", "mode": mode, "k": k},
        writer,
    ]})


def exact_environment():
    environment = os.environ.copy()
    environment["PDG_EXPERIMENTAL_DIRECT_EXTRA_DOUBLE_OUTPUT"] = "1"
    environment["PDG_REQUIRE_DIRECT_EXTRA_DOUBLE_OUTPUT"] = "1"
    return environment


def experimental_environment():
    environment = os.environ.copy()
    environment["PDG_EXPERIMENTAL_DIRECT_EXTRA_DOUBLE_OUTPUT"] = "1"
    environment.pop("PDG_REQUIRE_DIRECT_EXTRA_DOUBLE_OUTPUT", None)
    return environment


def nndistance_source_environment():
    environment = exact_environment()
    environment["PDG_EXPERIMENTAL_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
    environment["PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
    environment["PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY"] = "1"
    environment["PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ"] = "1"
    return environment


def las_layout(path):
    data = path.read_bytes()
    assert data[:4] == b"LASF" and len(data) >= 375, path
    assert not (data[104] & 0x80), "compressed LAS is outside this gate"
    offset = struct.unpack_from("<I", data, 96)[0]
    stride = struct.unpack_from("<H", data, 105)[0]
    count = struct.unpack_from("<Q", data, 247)[0]
    assert offset + count * stride == len(data), path
    return data, offset, stride, count


def rejected_case(root, name, pdg, source, **options):
    output = root / f"{name}.las"
    pipeline = write_hag_pipeline(
        root / f"{name}.json", source, output, **options)
    result = run([str(pdg), "resident", str(pipeline)], exact_environment())
    assert result.returncode != 0, (name, result)
    assert not output.exists(), (name, output)
    assert not list(root.glob("*.pdg-resident-output-*")), name


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: resident_extra_double_direct_gpu_test.py PDG PDAL")
    pdg = pathlib.Path(sys.argv[1]).resolve()
    oracle = pathlib.Path(sys.argv[2]).resolve()

    if shutil.which("nvidia-smi") is None:
        print("direct Extra Bytes gate skipped without nvidia-smi",
              file=sys.stderr)
        return SKIP
    gpu = run(["nvidia-smi", "--query-gpu=memory.free",
               "--format=csv,noheader,nounits"])
    if gpu.returncode != 0:
        print("direct Extra Bytes gate skipped without a GPU", file=sys.stderr)
        return SKIP

    with tempfile.TemporaryDirectory(
            prefix="pdg-resident-extra-double-direct-") as temporary:
        root = pathlib.Path(temporary)
        source = root / "source.las"
        fixture = run([
            str(oracle), "pipeline",
            str(write_fixture_pipeline(root / "fixture.json", source)),
        ])
        assert fixture.returncode == 0, fixture.stderr
        source_bytes, source_offset, source_stride, source_count = las_layout(
            source)
        assert (source_offset, source_stride) == (621, 40)
        assert source_count > 4

        candidate_output = root / "candidate.las"
        oracle_output = root / "oracle.las"
        stats_path = root / "candidate-stats.json"
        candidate_pipeline = write_hag_pipeline(
            root / "candidate.json", source, candidate_output)
        oracle_pipeline = write_hag_pipeline(
            root / "oracle.json", source, oracle_output)
        candidate = run([
            str(pdg), "resident", str(candidate_pipeline), "--stats",
            str(stats_path),
        ], nndistance_source_environment())
        reference = run([str(oracle), "pipeline", str(oracle_pipeline)])
        assert candidate.returncode == reference.returncode == 0, (
            candidate.returncode, candidate.stderr,
            reference.returncode, reference.stderr)
        assert candidate.stdout == reference.stdout
        assert candidate.stderr == reference.stderr
        assert candidate_output.read_bytes() == oracle_output.read_bytes()

        output_bytes, output_offset, output_stride, output_count = las_layout(
            candidate_output)
        assert (output_offset, output_stride, output_count) == (
            813, 48, source_count)
        assert struct.unpack_from("<H", output_bytes, 395)[0] == 384
        assert output_bytes[429:621] == source_bytes[429:621]
        descriptor = output_bytes[621:813]
        assert descriptor[2] == 10
        assert descriptor[4:21] == b"HeightAboveGround"
        assert descriptor[160:179] == b"Height Above Ground"
        appended = [struct.unpack_from(
            "<Q", output_bytes, output_offset + point * output_stride + 40)[0]
                    for point in range(output_count)]
        assert any(value not in (0, 1 << 63) for value in appended), (
            "fixture must positively exercise binary64 publication")

        report = json.loads(stats_path.read_text(encoding="utf-8"))
        execution = report["execution"]
        assert execution["executor"] == (
            "planner_resident_shared_index_direct_las"), execution
        assert execution["direct_las_resident_source"] is True, execution
        assert execution["direct_las_record_summary"] is True, execution
        assert execution["direct_las_host_xyz_mirror"] is False, execution
        assert execution["direct_las_output"] is True, execution
        assert execution["direct_extra_double_output"] is True, execution
        assert execution["terminal_spill_elided"] is True, execution
        assert execution["selected_regions"] == [0], execution
        assert execution["selected_stage_ids"] == [1], execution
        assert execution["resident_preflight"]["accepted"] is True, execution
        assert execution["index_builds"] == {
            "predicted": 1, "observed": 1, "matches_prediction": True,
        }, execution["index_builds"]
        assert execution["boundary_accounting_matches_prediction"] is True, (
            execution)
        assert execution["totals"]["boundary_spill"] == {
            "bytes": 0, "count": 0, "packing_bytes": 0,
        }, execution["totals"]
        assert execution["totals"]["fallback_boundary"] == {
            "bytes": 0, "count": 0, "packing_bytes": 0,
        }, execution["totals"]
        assert report["placement"]["reason"] == "uncalibrated_stage", report

        wide_source = root / "wide-source.las"
        wide_fixture = run([
            str(oracle), "pipeline",
            str(write_wide_fixture_pipeline(root / "wide-fixture.json",
                                            wide_source)),
        ])
        assert wide_fixture.returncode == 0, wide_fixture.stderr
        _, wide_offset, wide_stride, wide_count = las_layout(wide_source)
        assert (wide_offset, wide_stride) == (621, 40)
        assert wide_count == 237
        for count in (16, 64):
            wide_candidate_output = root / f"wide-count{count}-candidate.las"
            wide_oracle_output = root / f"wide-count{count}-oracle.las"
            wide_stats = root / f"wide-count{count}-candidate-stats.json"
            wide_candidate = run([
                str(pdg), "resident",
                str(write_hag_pipeline(
                    root / f"wide-count{count}-candidate.json", wide_source,
                    wide_candidate_output, count=count)),
                "--stats", str(wide_stats),
            ], nndistance_source_environment())
            wide_oracle = run([
                str(oracle), "pipeline",
                str(write_hag_pipeline(
                    root / f"wide-count{count}-oracle.json", wide_source,
                    wide_oracle_output, count=count)),
            ])
            assert (
                wide_candidate.returncode == wide_oracle.returncode == 0
            ), (count, wide_candidate.stderr, wide_oracle.stderr)
            assert wide_candidate.stdout == wide_oracle.stdout, count
            assert wide_candidate.stderr == wide_oracle.stderr, count
            assert wide_candidate_output.read_bytes() == (
                wide_oracle_output.read_bytes()), count
            wide_execution = json.loads(
                wide_stats.read_text(encoding="utf-8"))["execution"]
            assert wide_execution["executor"] == (
                "planner_resident_shared_index_direct_las"), count
            assert wide_execution["direct_las_resident_source"] is True, count
            assert wide_execution["index_builds"]["observed"] == 1, count
            assert wide_execution["totals"]["boundary_spill"]["count"] == 0, (
                count)
            assert not list(root.glob("*.pdg-resident-output-*")), count

        for count in (1, 2, 3, 5, 6, 7):
            candidate_output = root / f"count{count}-candidate.las"
            oracle_count_output = root / f"count{count}-oracle.las"
            count_stats = root / f"count{count}-candidate-stats.json"
            candidate_count_pipeline = write_hag_pipeline(
                root / f"count{count}-candidate.json", source,
                candidate_output, count=count)
            oracle_count_pipeline = write_hag_pipeline(
                root / f"count{count}-oracle.json", source,
                oracle_count_output, count=count)
            candidate_count = run([
                str(pdg), "resident", str(candidate_count_pipeline),
                "--stats", str(count_stats),
            ], nndistance_source_environment())
            oracle_count = run([
                str(oracle), "pipeline", str(oracle_count_pipeline),
            ])
            assert (
                candidate_count.returncode == oracle_count.returncode == 0
            ), (count, candidate_count.stderr, oracle_count.stderr)
            assert candidate_count.stdout == oracle_count.stdout, count
            assert candidate_count.stderr == oracle_count.stderr, count
            assert candidate_output.read_bytes() == (
                oracle_count_output.read_bytes()), count
            count_bytes, count_offset, count_stride, count_count = las_layout(
                candidate_output)
            assert (count_offset, count_stride, count_count) == (
                813, 48, source_count), count
            assert count_bytes[429:621] == source_bytes[429:621], count
            count_descriptor = count_bytes[621:813]
            assert count_descriptor[2] == 10, count
            assert count_descriptor[4:21] == b"HeightAboveGround", count
            assert count_descriptor[160:179] == b"Height Above Ground", count
            count_values = [struct.unpack_from(
                "<Q", count_bytes,
                count_offset + point * count_stride + 40)[0]
                            for point in range(count_count)]
            assert any(value not in (0, 1 << 63) for value in count_values), (
                count)
            count_report = json.loads(
                count_stats.read_text(encoding="utf-8"))
            count_execution = count_report["execution"]
            assert count_execution["executor"] == (
                "planner_resident_shared_index_direct_las"), count
            assert count_execution["direct_las_resident_source"] is True
            assert count_execution["direct_las_record_summary"] is True
            assert count_execution["direct_las_host_xyz_mirror"] is False
            assert count_execution["direct_extra_double_output"] is True
            assert count_execution["terminal_spill_elided"] is True
            assert count_execution["index_builds"] == {
                "predicted": 1, "observed": 1, "matches_prediction": True,
            }, count
            assert count_execution["totals"][
                "boundary_spill"]["count"] == 0, count
            assert count_execution["totals"][
                "fallback_boundary"]["count"] == 0, count

        hag_repair_cases = (
            ("hag-count1-boundary-tie-repair", (
                (1.0, 0.0, 10.0, 2, 0.0),
                (-1.0, 0.0, 14.0, 2, 1.0),
                (0.0, 0.0, 30.0, 1, 2.0),
            ), {
                "PDG_REQUIRE_HAG_NN_TIE_FALLBACK": "1",
            }, 1),
            ("hag-count1-incomplete-repair", (
                (0.0, 0.0, 20.0, 1, 0.0),
                (5001.0, 0.0, 10.0, 2, 1.0),
                (5002.0, 0.0, 11.0, 2, 2.0),
            ), {
                "PDG_FORCE_UNIFORM_GRID": "1",
                "PDG_KNN_DEVICE_SHELL_BUDGET": "1",
                "PDG_REQUIRE_HAG_NN_HOST_FALLBACK": "1",
            }, 1),
            ("hag-tie-repair", (
                (1.0, 0.0, 10.0, 2, 0.0),
                (2.0, 0.0, 14.0, 2, 1.0),
                (3.0, 0.0, 18.0, 2, 2.0),
                (4.0, 0.0, 22.0, 2, 3.0),
                (-4.0, 0.0, 26.0, 2, 4.0),
                (0.0, 0.0, 30.0, 1, 5.0),
            ), {
                "PDG_REQUIRE_HAG_NN_TIE_FALLBACK": "1",
            }, 4),
            ("hag-incomplete-repair", (
                (0.0, 0.0, 20.0, 1, 0.0),
                (5001.0, 0.0, 10.0, 2, 1.0),
                (5002.0, 0.0, 11.0, 2, 2.0),
                (5003.0, 0.0, 12.0, 2, 3.0),
                (5004.0, 0.0, 13.0, 2, 4.0),
            ), {
                "PDG_FORCE_UNIFORM_GRID": "1",
                "PDG_KNN_DEVICE_SHELL_BUDGET": "1",
                "PDG_REQUIRE_HAG_NN_HOST_FALLBACK": "1",
            }, 4),
            ("hag-count2-boundary-tie-repair", (
                (1.0, 0.0, 10.0, 2, 0.0),
                (2.0, 0.0, 14.0, 2, 1.0),
                (-2.0, 0.0, 18.0, 2, 2.0),
                (0.0, 0.0, 30.0, 1, 3.0),
            ), {
                "PDG_REQUIRE_HAG_NN_TIE_FALLBACK": "1",
            }, 2),
            ("hag-count2-incomplete-repair", (
                (0.0, 0.0, 20.0, 1, 0.0),
                (5001.0, 0.0, 10.0, 2, 1.0),
                (5002.0, 0.0, 11.0, 2, 2.0),
                (5003.0, 0.0, 12.0, 2, 3.0),
            ), {
                "PDG_FORCE_UNIFORM_GRID": "1",
                "PDG_KNN_DEVICE_SHELL_BUDGET": "1",
                "PDG_REQUIRE_HAG_NN_HOST_FALLBACK": "1",
            }, 2),
            ("hag-count3-boundary-tie-repair", (
                (1.0, 0.0, 10.0, 2, 0.0),
                (2.0, 0.0, 14.0, 2, 1.0),
                (3.0, 0.0, 18.0, 2, 2.0),
                (-3.0, 0.0, 22.0, 2, 3.0),
                (0.0, 0.0, 30.0, 1, 4.0),
            ), {
                "PDG_REQUIRE_HAG_NN_TIE_FALLBACK": "1",
            }, 3),
            ("hag-count3-incomplete-repair", (
                (0.0, 0.0, 20.0, 1, 0.0),
                (5001.0, 0.0, 10.0, 2, 1.0),
                (5002.0, 0.0, 11.0, 2, 2.0),
                (5003.0, 0.0, 12.0, 2, 3.0),
                (5004.0, 0.0, 13.0, 2, 4.0),
            ), {
                "PDG_FORCE_UNIFORM_GRID": "1",
                "PDG_KNN_DEVICE_SHELL_BUDGET": "1",
                "PDG_REQUIRE_HAG_NN_HOST_FALLBACK": "1",
            }, 3),
            ("hag-count5-boundary-tie-repair", (
                (1.0, 0.0, 10.0, 2, 0.0),
                (2.0, 0.0, 14.0, 2, 1.0),
                (3.0, 0.0, 18.0, 2, 2.0),
                (4.0, 0.0, 22.0, 2, 3.0),
                (5.0, 0.0, 26.0, 2, 4.0),
                (-5.0, 0.0, 28.0, 2, 5.0),
                (0.0, 0.0, 30.0, 1, 6.0),
            ), {
                "PDG_REQUIRE_HAG_NN_TIE_FALLBACK": "1",
            }, 5),
            ("hag-count5-incomplete-repair", (
                (0.0, 0.0, 20.0, 1, 0.0),
                (5001.0, 0.0, 10.0, 2, 1.0),
                (5002.0, 0.0, 11.0, 2, 2.0),
                (5003.0, 0.0, 12.0, 2, 3.0),
                (5004.0, 0.0, 13.0, 2, 4.0),
                (5005.0, 0.0, 14.0, 2, 5.0),
            ), {
                "PDG_FORCE_UNIFORM_GRID": "1",
                "PDG_KNN_DEVICE_SHELL_BUDGET": "1",
                "PDG_REQUIRE_HAG_NN_HOST_FALLBACK": "1",
            }, 5),
            ("hag-count6-boundary-tie-repair", (
                (1.0, 0.0, 10.0, 2, 0.0),
                (2.0, 0.0, 14.0, 2, 1.0),
                (3.0, 0.0, 18.0, 2, 2.0),
                (4.0, 0.0, 22.0, 2, 3.0),
                (5.0, 0.0, 26.0, 2, 4.0),
                (6.0, 0.0, 30.0, 2, 5.0),
                (-6.0, 0.0, 34.0, 2, 6.0),
                (0.0, 0.0, 38.0, 1, 7.0),
            ), {
                "PDG_REQUIRE_HAG_NN_TIE_FALLBACK": "1",
            }, 6),
            ("hag-count6-incomplete-repair", (
                (0.0, 0.0, 30.0, 1, 0.0),
                (5001.0, 0.0, 10.0, 2, 1.0),
                (5002.0, 0.0, 11.0, 2, 2.0),
                (5003.0, 0.0, 12.0, 2, 3.0),
                (5004.0, 0.0, 13.0, 2, 4.0),
                (5005.0, 0.0, 14.0, 2, 5.0),
                (5006.0, 0.0, 15.0, 2, 6.0),
            ), {
                "PDG_FORCE_UNIFORM_GRID": "1",
                "PDG_KNN_DEVICE_SHELL_BUDGET": "1",
                "PDG_REQUIRE_HAG_NN_HOST_FALLBACK": "1",
            }, 6),
            ("hag-count7-boundary-tie-repair", (
                (1.0, 0.0, 10.0, 2, 0.0),
                (2.0, 0.0, 14.0, 2, 1.0),
                (3.0, 0.0, 18.0, 2, 2.0),
                (4.0, 0.0, 22.0, 2, 3.0),
                (5.0, 0.0, 26.0, 2, 4.0),
                (6.0, 0.0, 30.0, 2, 5.0),
                (7.0, 0.0, 34.0, 2, 6.0),
                (-7.0, 0.0, 38.0, 2, 7.0),
                (0.0, 0.0, 42.0, 1, 8.0),
            ), {
                "PDG_REQUIRE_HAG_NN_TIE_FALLBACK": "1",
            }, 7),
            ("hag-count7-incomplete-repair", (
                (0.0, 0.0, 30.0, 1, 0.0),
                (5001.0, 0.0, 10.0, 2, 1.0),
                (5002.0, 0.0, 11.0, 2, 2.0),
                (5003.0, 0.0, 12.0, 2, 3.0),
                (5004.0, 0.0, 13.0, 2, 4.0),
                (5005.0, 0.0, 14.0, 2, 5.0),
                (5006.0, 0.0, 15.0, 2, 6.0),
                (5007.0, 0.0, 16.0, 2, 7.0),
            ), {
                "PDG_FORCE_UNIFORM_GRID": "1",
                "PDG_KNN_DEVICE_SHELL_BUDGET": "1",
                "PDG_REQUIRE_HAG_NN_HOST_FALLBACK": "1",
            }, 7),
        )
        for name, rows, proof, count in hag_repair_cases:
            repair_source = root / f"{name}-source.las"
            repair_fixture = run([
                str(oracle), "pipeline",
                str(write_hag_delaunay_rows_fixture_pipeline(
                    root / f"{name}-fixture.json",
                    root / f"{name}-fixture.csv", repair_source, rows)),
            ])
            assert repair_fixture.returncode == 0, (
                name, repair_fixture.stderr)
            repair_source_bytes, repair_offset, repair_stride, repair_count = \
                las_layout(repair_source)
            assert (repair_offset, repair_stride, repair_count) == (
                621, 40, len(rows)), (
                    name, repair_offset, repair_stride, repair_count)
            repair_source_bytes = bytearray(repair_source_bytes)
            repair_source_bytes[429:621] = source_bytes[429:621]
            repair_source.write_bytes(repair_source_bytes)

            repair_candidate_output = root / f"{name}-candidate.las"
            repair_oracle_output = root / f"{name}-oracle.las"
            repair_candidate_pipeline = write_hag_pipeline(
                root / f"{name}-candidate.json", repair_source,
                repair_candidate_output, count=count)
            repair_oracle_pipeline = write_hag_pipeline(
                root / f"{name}-oracle.json", repair_source,
                repair_oracle_output, count=count)
            repair_environment = nndistance_source_environment()
            repair_environment.update(proof)
            repair_candidate = run([
                str(pdg), "resident", str(repair_candidate_pipeline),
            ], repair_environment)
            repair_oracle = run([
                str(oracle), "pipeline", str(repair_oracle_pipeline),
            ])
            assert (
                repair_candidate.returncode == repair_oracle.returncode == 0
            ), (name, repair_candidate.stderr, repair_oracle.stderr)
            assert repair_candidate.stdout == repair_oracle.stdout, name
            assert repair_candidate.stderr == repair_oracle.stderr, name
            assert repair_candidate_output.read_bytes() == (
                repair_oracle_output.read_bytes()), name
            assert not list(root.glob("*.pdg-resident-output-*")), name

        for count in (1, 2, 3, 4, 5, 6, 7):
            source_disabled_output = root / f"source-disabled-{count}.las"
            source_disabled_pipeline = write_hag_pipeline(
                root / f"source-disabled-{count}.json", source,
                source_disabled_output, count=count)
            source_disabled_environment = nndistance_source_environment()
            source_disabled_environment[
                "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
            source_disabled = run([
                str(pdg), "resident", str(source_disabled_pipeline),
            ], source_disabled_environment)
            assert source_disabled.returncode != 0, source_disabled
            assert not source_disabled_output.exists()
            assert not list(root.glob("*.pdg-resident-output-*"))

        experimental_output = root / "experimental-only.las"
        experimental_stats = root / "experimental-only-stats.json"
        experimental_pipeline = write_hag_pipeline(
            root / "experimental-only.json", source, experimental_output)
        experimental = run([
            str(pdg), "resident", str(experimental_pipeline), "--stats",
            str(experimental_stats),
        ], experimental_environment())
        assert experimental.returncode == 0, experimental.stderr
        experimental_bytes = experimental_output.read_bytes()
        oracle_bytes = oracle_output.read_bytes()
        first_difference = next((offset for offset, values in enumerate(
            zip(experimental_bytes, oracle_bytes))
            if values[0] != values[1]),
            min(len(experimental_bytes), len(oracle_bytes)))
        assert experimental_bytes == oracle_bytes, (
            first_difference, len(experimental_bytes), len(oracle_bytes))
        experimental_report = json.loads(
            experimental_stats.read_text(encoding="utf-8"))
        assert experimental_report["execution"][
            "direct_extra_double_output"] is True, experimental_report

        unforced_output = root / "unforced.las"
        unforced_pipeline = write_hag_pipeline(
            root / "unforced.json", source, unforced_output)
        unforced_environment = os.environ.copy()
        unforced_environment.pop(
            "PDG_EXPERIMENTAL_DIRECT_EXTRA_DOUBLE_OUTPUT", None)
        unforced_environment.pop("PDG_REQUIRE_DIRECT_EXTRA_DOUBLE_OUTPUT",
                                 None)
        unforced = run([
            str(pdg), "resident", str(unforced_pipeline),
        ], unforced_environment)
        assert unforced.returncode == 0, unforced.stderr
        assert unforced_output.read_bytes() == oracle_output.read_bytes()

        automatic_count5_output = root / "automatic-count5.las"
        automatic_count5_oracle_output = root / "automatic-count5-oracle.las"
        automatic_count5_pipeline = write_hag_pipeline(
            root / "automatic-count5.json", source,
            automatic_count5_output, count=5)
        automatic_count5_oracle_pipeline = write_hag_pipeline(
            root / "automatic-count5-oracle.json", source,
            automatic_count5_oracle_output, count=5)
        automatic_count5_environment = os.environ.copy()
        automatic_count5_environment["PDG_ORACLE_PDAL"] = str(oracle)
        automatic_count5_environment.pop(
            "PDG_EXPERIMENTAL_DIRECT_EXTRA_DOUBLE_OUTPUT", None)
        automatic_count5_environment.pop(
            "PDG_REQUIRE_DIRECT_EXTRA_DOUBLE_OUTPUT", None)
        automatic_count5 = run([
            str(pdg), "pipeline", str(automatic_count5_pipeline),
        ], automatic_count5_environment)
        automatic_count5_oracle = run([
            str(oracle), "pipeline", str(automatic_count5_oracle_pipeline),
        ])
        assert automatic_count5.returncode == (
            automatic_count5_oracle.returncode) == 0
        assert automatic_count5.stdout == automatic_count5_oracle.stdout
        assert automatic_count5.stderr == automatic_count5_oracle.stderr
        assert automatic_count5_output.read_bytes() == (
            automatic_count5_oracle_output.read_bytes())
        assert not list(root.glob("*.pdg-resident-output-*"))

        unsupported_source = root / "unsupported-descriptor.las"
        unsupported_bytes = bytearray(source_bytes)
        unsupported_bytes[433] = ord("P")
        unsupported_source.write_bytes(unsupported_bytes)
        unsupported_candidate_output = root / "unsupported-candidate.las"
        unsupported_oracle_output = root / "unsupported-oracle.las"
        unsupported_candidate_pipeline = write_hag_pipeline(
            root / "unsupported-candidate.json", unsupported_source,
            unsupported_candidate_output)
        unsupported_oracle_pipeline = write_hag_pipeline(
            root / "unsupported-oracle.json", unsupported_source,
            unsupported_oracle_output)
        unsupported_candidate = run([
            str(pdg), "resident", str(unsupported_candidate_pipeline),
        ], experimental_environment())
        unsupported_oracle = run([
            str(oracle), "pipeline", str(unsupported_oracle_pipeline),
        ])
        assert unsupported_candidate.returncode == (
            unsupported_oracle.returncode) == 0, (
                unsupported_candidate.stderr, unsupported_oracle.stderr)
        assert unsupported_candidate.stdout == unsupported_oracle.stdout
        assert unsupported_candidate.stderr == unsupported_oracle.stderr
        assert unsupported_candidate_output.read_bytes() == (
            unsupported_oracle_output.read_bytes())

        for count in (65,):
            unsupported_required_output = (
                root / f"unsupported-required-{count}.las")
            unsupported_required_pipeline = write_hag_pipeline(
                root / f"unsupported-required-{count}.json",
                unsupported_source, unsupported_required_output, count=count)
            unsupported_required = run([
                str(pdg), "resident", str(unsupported_required_pipeline),
            ], nndistance_source_environment())
            assert unsupported_required.returncode != 0, unsupported_required
            assert not unsupported_required_output.exists()
            assert not list(root.glob("*.pdg-resident-output-*"))

        nnd_candidate_output = root / "nnd-candidate.las"
        nnd_oracle_output = root / "nnd-oracle.las"
        nnd_stats = root / "nnd-candidate-stats.json"
        nnd_candidate_pipeline = write_nndistance_pipeline(
            root / "nnd-candidate.json", source, nnd_candidate_output)
        nnd_oracle_pipeline = write_nndistance_pipeline(
            root / "nnd-oracle.json", source, nnd_oracle_output)
        nnd_candidate = run([
            str(pdg), "resident", str(nnd_candidate_pipeline), "--stats",
            str(nnd_stats),
        ], exact_environment())
        nnd_oracle = run([
            str(oracle), "pipeline", str(nnd_oracle_pipeline),
        ])
        assert nnd_candidate.returncode == nnd_oracle.returncode == 0, (
            nnd_candidate.stderr, nnd_oracle.stderr)
        assert nnd_candidate.stdout == nnd_oracle.stdout
        assert nnd_candidate.stderr == nnd_oracle.stderr
        assert nnd_candidate_output.read_bytes() == (
            nnd_oracle_output.read_bytes())
        nnd_bytes, nnd_offset, nnd_stride, nnd_count = las_layout(
            nnd_candidate_output)
        assert (nnd_offset, nnd_stride, nnd_count) == (813, 48, source_count)
        nnd_descriptor = nnd_bytes[621:813]
        assert nnd_descriptor[2] == 10
        assert nnd_descriptor[4:14] == b"NNDistance"
        assert nnd_descriptor[160:192] == (
            b"Distance metric related to a poi")
        nnd_values = [struct.unpack_from(
            "<Q", nnd_bytes, nnd_offset + point * nnd_stride + 40)[0]
                      for point in range(nnd_count)]
        assert any(value != 0 for value in nnd_values)
        nnd_report = json.loads(nnd_stats.read_text(encoding="utf-8"))
        nnd_execution = nnd_report["execution"]
        assert nnd_execution["executor"] == (
            "planner_resident_shared_index_direct_las"), nnd_execution
        assert nnd_execution["direct_extra_double_output"] is True
        assert nnd_execution["terminal_spill_elided"] is True
        assert nnd_execution["index_builds"] == {
            "predicted": 1, "observed": 1, "matches_prediction": True,
        }
        assert nnd_execution["totals"]["boundary_spill"]["count"] == 0

        nnd_source_output = root / "nnd-source.las"
        nnd_source_stats = root / "nnd-source-stats.json"
        nnd_source_pipeline = write_nndistance_pipeline(
            root / "nnd-source.json", source, nnd_source_output)
        nnd_source = run([
            str(pdg), "resident", str(nnd_source_pipeline), "--stats",
            str(nnd_source_stats),
        ], nndistance_source_environment())
        assert nnd_source.returncode == 0, nnd_source.stderr
        assert nnd_source.stdout == nnd_oracle.stdout
        assert nnd_source.stderr == nnd_oracle.stderr
        assert nnd_source_output.read_bytes() == (
            nnd_oracle_output.read_bytes())
        nnd_source_report = json.loads(
            nnd_source_stats.read_text(encoding="utf-8"))
        nnd_source_execution = nnd_source_report["execution"]
        assert nnd_source_execution["direct_las_resident_source"] is True
        assert nnd_source_execution["direct_las_record_summary"] is True
        assert nnd_source_execution["direct_las_host_xyz_mirror"] is False
        assert nnd_source_execution["direct_extra_double_output"] is True
        assert nnd_source_execution["index_builds"] == {
            "predicted": 1, "observed": 1, "matches_prediction": True,
        }

        nnd_disabled_output = root / "nnd-source-disabled.las"
        nnd_disabled_pipeline = write_nndistance_pipeline(
            root / "nnd-source-disabled.json", source, nnd_disabled_output)
        nnd_disabled_environment = nndistance_source_environment()
        nnd_disabled_environment[
            "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
        nnd_disabled = run([
            str(pdg), "resident", str(nnd_disabled_pipeline),
        ], nnd_disabled_environment)
        assert nnd_disabled.returncode != 0, nnd_disabled
        assert not nnd_disabled_output.exists()

        nnd_unsupported_source_output = root / "nnd-source-unsupported.las"
        nnd_unsupported_source_pipeline = write_nndistance_pipeline(
            root / "nnd-source-unsupported.json", unsupported_source,
            nnd_unsupported_source_output)
        nnd_unsupported_source = run([
            str(pdg), "resident", str(nnd_unsupported_source_pipeline),
        ], nndistance_source_environment())
        assert nnd_unsupported_source.returncode != 0, (
            nnd_unsupported_source)
        assert not nnd_unsupported_source_output.exists()

        for name, options in (
                ("nnd-average", {"mode": "avg"}),
                ("nnd-k9", {"k": 9}),
                ("nnd-no-extra", {"extra_dims": None})):
            output = root / f"{name}.las"
            pipeline = write_nndistance_pipeline(
                root / f"{name}.json", source, output, **options)
            rejected = run([
                str(pdg), "resident", str(pipeline),
            ], exact_environment())
            assert rejected.returncode != 0, (name, rejected)
            assert not output.exists(), (name, output)

        nnd_fallback_output = root / "nnd-fallback.las"
        nnd_fallback_oracle_output = root / "nnd-fallback-oracle.las"
        nnd_fallback_pipeline = write_nndistance_pipeline(
            root / "nnd-fallback.json", source, nnd_fallback_output,
            mode="avg")
        nnd_fallback_oracle_pipeline = write_nndistance_pipeline(
            root / "nnd-fallback-oracle.json", source,
            nnd_fallback_oracle_output, mode="avg")
        nnd_fallback = run([
            str(pdg), "resident", str(nnd_fallback_pipeline),
        ], experimental_environment())
        nnd_fallback_oracle = run([
            str(oracle), "pipeline", str(nnd_fallback_oracle_pipeline),
        ])
        assert nnd_fallback.returncode == nnd_fallback_oracle.returncode == 0
        assert nnd_fallback.stdout == nnd_fallback_oracle.stdout
        assert nnd_fallback.stderr == nnd_fallback_oracle.stderr
        assert nnd_fallback_output.read_bytes() == (
            nnd_fallback_oracle_output.read_bytes())

        delaunay_source = root / "delaunay-source.las"
        delaunay_fixture = run([
            str(oracle), "pipeline",
            str(write_hag_delaunay_fixture_pipeline(
                root / "delaunay-fixture.json", delaunay_source)),
        ])
        assert delaunay_fixture.returncode == 0, delaunay_fixture.stderr
        delaunay_source_bytes, delaunay_source_offset, \
            delaunay_source_stride, delaunay_source_count = las_layout(
                delaunay_source)
        assert (delaunay_source_offset, delaunay_source_stride) == (621, 40)
        assert delaunay_source_count > 3

        delaunay_candidate_output = root / "delaunay-candidate.las"
        delaunay_oracle_output = root / "delaunay-oracle.las"
        delaunay_stats = root / "delaunay-candidate-stats.json"
        delaunay_candidate_pipeline = write_hag_delaunay_pipeline(
            root / "delaunay-candidate.json", delaunay_source,
            delaunay_candidate_output)
        delaunay_oracle_pipeline = write_hag_delaunay_pipeline(
            root / "delaunay-oracle.json", delaunay_source,
            delaunay_oracle_output)
        delaunay_candidate = run([
            str(pdg), "resident", str(delaunay_candidate_pipeline), "--stats",
            str(delaunay_stats),
        ], nndistance_source_environment())
        delaunay_oracle = run([
            str(oracle), "pipeline", str(delaunay_oracle_pipeline),
        ])
        assert (
            delaunay_candidate.returncode ==
            delaunay_oracle.returncode == 0
        ), (delaunay_candidate.stderr, delaunay_oracle.stderr)
        assert delaunay_candidate.stdout == delaunay_oracle.stdout
        assert delaunay_candidate.stderr == delaunay_oracle.stderr
        assert delaunay_candidate_output.read_bytes() == (
            delaunay_oracle_output.read_bytes())

        delaunay_bytes, delaunay_offset, delaunay_stride, delaunay_count = \
            las_layout(delaunay_candidate_output)
        assert (delaunay_offset, delaunay_stride, delaunay_count) == (
            813, 48, delaunay_source_count)
        assert delaunay_bytes[429:621] == delaunay_source_bytes[429:621]
        delaunay_descriptor = delaunay_bytes[621:813]
        assert delaunay_descriptor[2] == 10
        assert delaunay_descriptor[4:21] == b"HeightAboveGround"
        assert delaunay_descriptor[160:179] == b"Height Above Ground"
        delaunay_values = [struct.unpack_from(
            "<Q", delaunay_bytes,
            delaunay_offset + point * delaunay_stride + 40)[0]
                           for point in range(delaunay_count)]
        assert any(value not in (0, 1 << 63) for value in delaunay_values)

        delaunay_report = json.loads(
            delaunay_stats.read_text(encoding="utf-8"))
        delaunay_execution = delaunay_report["execution"]
        assert delaunay_execution["executor"] == (
            "planner_resident_shared_index_direct_las")
        assert delaunay_execution["direct_las_resident_source"] is True
        assert delaunay_execution["direct_las_record_summary"] is True
        assert delaunay_execution["direct_las_host_xyz_mirror"] is False
        assert delaunay_execution["direct_extra_double_output"] is True
        assert delaunay_execution["terminal_spill_elided"] is True
        assert delaunay_execution["index_builds"] == {
            "predicted": 1, "observed": 1, "matches_prediction": True,
        }
        assert delaunay_execution["totals"]["boundary_spill"]["count"] == 0
        assert delaunay_execution["totals"]["fallback_boundary"][
            "count"] == 0

        repair_cases = (
            ("delaunay-tie-repair", (
                (1.0, 0.0, 10.0, 2, 0.0),
                (0.0, 1.0, 14.0, 2, 1.0),
                (-1.0, 0.0, 18.0, 2, 2.0),
                (0.0, -1.0, 22.0, 2, 3.0),
                (0.0, 0.0, 30.0, 1, 4.0),
            ), {
                "PDG_REQUIRE_HAG_DELAUNAY_TIE_FALLBACK": "1",
            }),
            ("delaunay-incomplete-repair", (
                (0.0, 0.0, 10.0, 2, 0.0),
                (5001.0, 0.0, 20.0, 2, 1.0),
                (5002.0, 0.0, 11.0, 2, 2.0),
                (1.0, 1.0, 30.0, 1, 3.0),
            ), {
                "PDG_FORCE_UNIFORM_GRID": "1",
                "PDG_KNN_DEVICE_SHELL_BUDGET": "1",
                "PDG_REQUIRE_HAG_DELAUNAY_HOST_FALLBACK": "1",
            }),
        )
        for name, rows, proof in repair_cases:
            repair_source = root / f"{name}-source.las"
            repair_fixture = run([
                str(oracle), "pipeline",
                str(write_hag_delaunay_rows_fixture_pipeline(
                    root / f"{name}-fixture.json",
                    root / f"{name}-fixture.csv", repair_source, rows)),
            ])
            assert repair_fixture.returncode == 0, (
                name, repair_fixture.stderr)
            repair_source_bytes, repair_offset, repair_stride, repair_count = \
                las_layout(repair_source)
            assert (repair_offset, repair_stride, repair_count) == (
                621, 40, len(rows)), (
                    name, repair_offset, repair_stride, repair_count)
            repair_source_bytes = bytearray(repair_source_bytes)
            repair_source_bytes[429:621] = delaunay_source_bytes[429:621]
            repair_source.write_bytes(repair_source_bytes)

            repair_candidate_output = root / f"{name}-candidate.las"
            repair_oracle_output = root / f"{name}-oracle.las"
            repair_candidate_pipeline = write_hag_delaunay_pipeline(
                root / f"{name}-candidate.json", repair_source,
                repair_candidate_output)
            repair_oracle_pipeline = write_hag_delaunay_pipeline(
                root / f"{name}-oracle.json", repair_source,
                repair_oracle_output)
            repair_environment = nndistance_source_environment()
            repair_environment.update(proof)
            repair_candidate = run([
                str(pdg), "resident", str(repair_candidate_pipeline),
            ], repair_environment)
            repair_oracle = run([
                str(oracle), "pipeline", str(repair_oracle_pipeline),
            ])
            assert (
                repair_candidate.returncode == repair_oracle.returncode == 0
            ), (name, repair_candidate.stderr, repair_oracle.stderr)
            assert repair_candidate.stdout == repair_oracle.stdout, name
            assert repair_candidate.stderr == repair_oracle.stderr, name
            assert repair_candidate_output.read_bytes() == (
                repair_oracle_output.read_bytes()), name
            assert not list(root.glob("*.pdg-resident-output-*")), name

        delaunay_disabled_output = root / "delaunay-source-disabled.las"
        delaunay_disabled_pipeline = write_hag_delaunay_pipeline(
            root / "delaunay-source-disabled.json", delaunay_source,
            delaunay_disabled_output)
        delaunay_disabled_environment = nndistance_source_environment()
        delaunay_disabled_environment[
            "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
        delaunay_disabled = run([
            str(pdg), "resident", str(delaunay_disabled_pipeline),
        ], delaunay_disabled_environment)
        assert delaunay_disabled.returncode != 0, delaunay_disabled
        assert not delaunay_disabled_output.exists()

        delaunay_unsupported_output = root / "delaunay-unsupported.las"
        delaunay_unsupported_pipeline = write_hag_delaunay_pipeline(
            root / "delaunay-unsupported.json", unsupported_source,
            delaunay_unsupported_output)
        delaunay_unsupported = run([
            str(pdg), "resident", str(delaunay_unsupported_pipeline),
        ], nndistance_source_environment())
        assert delaunay_unsupported.returncode != 0, delaunay_unsupported
        assert not delaunay_unsupported_output.exists()

        for name, options in (
                ("delaunay-count-four", {"count": 4}),
                ("delaunay-explicit-default",
                 {"allow_extrapolation": False}),
                ("delaunay-no-extra", {"extra_dims": None})):
            output = root / f"{name}.las"
            pipeline = write_hag_delaunay_pipeline(
                root / f"{name}.json", delaunay_source, output, **options)
            rejected = run([
                str(pdg), "resident", str(pipeline),
            ], exact_environment())
            assert rejected.returncode != 0, (name, rejected)
            assert not output.exists(), (name, output)

        rejected_case(root, "unsupported-count", pdg, source, count=65)
        rejected_case(root, "zero-count", pdg, source, count=0)
        rejected_case(
            root, "count-one-option", pdg, source, count=1,
            producer_options={"max_distance": 25.0})
        rejected_case(root, "missing-extra-dims", pdg, source,
                      extra_dims=None)
        rejected_case(root, "named-extra-dims", pdg, source,
                      extra_dims="HeightAboveGround=double")

        existing_output = root / "existing.las"
        existing_output.write_bytes(b"sentinel")
        existing_pipeline = write_hag_pipeline(
            root / "existing.json", source, existing_output)
        existing = run([
            str(pdg), "resident", str(existing_pipeline),
        ], exact_environment())
        assert existing.returncode != 0, existing
        assert existing_output.read_bytes() == b"sentinel"
        assert not list(root.glob("*.pdg-resident-output-*")), root
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
