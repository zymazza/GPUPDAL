#!/usr/bin/env python3
"""Exact process gate for the opt-in direct-LAS outlier composition."""

import json
import os
import pathlib
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


def write_pipeline(path, source, destination, mean_k=8):
    path.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.outlier", "method": "statistical",
         "mean_k": mean_k, "multiplier": 2.0, "class": 7},
        {"type": "filters.nndistance", "mode": "kth", "k": 10},
        {"type": "writers.las", "filename": str(destination)},
    ]}), encoding="utf-8")
    return path


def write_standalone_pipeline(path, source, destination, mean_k=8,
                              multiplier=2.0, classification=7,
                              extra_options=None):
    outlier = {"type": "filters.outlier", "method": "statistical",
               "mean_k": mean_k, "multiplier": multiplier,
               "class": classification}
    if extra_options:
        outlier.update(extra_options)
    path.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        outlier,
        {"type": "writers.las", "filename": str(destination)},
    ]}), encoding="utf-8")
    return path


def write_radius_standalone_pipeline(path, source, destination, radius=1.0,
                                     min_k=2, extra_options=None):
    outlier = {"type": "filters.outlier", "method": "radius",
               "radius": radius, "min_k": min_k}
    if extra_options:
        outlier.update(extra_options)
    path.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        outlier,
        {"type": "writers.las", "filename": str(destination)},
    ]}), encoding="utf-8")
    return path


def write_radius_composition_pipeline(path, source, destination,
                                      outlier_radius=1.01,
                                      density_radius=1.01):
    path.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.outlier", "method": "radius",
         "radius": outlier_radius, "min_k": 2, "class": 7},
        {"type": "filters.radialdensity", "radius": density_radius},
        {"type": "filters.assign",
         "value": "UserData = 1 WHERE RadialDensity >= 0.2"},
        {"type": "writers.las", "filename": str(destination)},
    ]}), encoding="utf-8")
    return path


def assert_no_direct_output_temp(output):
    assert not list(output.parent.glob(
        output.name + ".pdg-resident-output-*")), output


def exact_environment():
    environment = os.environ.copy()
    environment["PDG_EXPERIMENTAL_DIRECT_CLASSIFICATION_OUTPUT"] = "1"
    environment["PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT"] = "1"
    environment["PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
    environment["PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY"] = "1"
    environment["PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ"] = "1"
    return environment


def radius_composition_environment():
    environment = exact_environment()
    environment[
        "PDG_REQUIRE_RADIUS_OUTLIER_RADIALDENSITY_COMPOSITION"] = "1"
    environment["PDG_REQUIRE_NEIGHBORHOOD_REUSE"] = "1"
    return environment


def las_layout(data, source):
    assert data[:4] == b"LASF" and len(data) >= 227, source
    point_offset = struct.unpack_from("<I", data, 96)[0]
    point_format = data[104] & 0x0f
    assert not (data[104] & 0x80), "compressed input is outside this gate"
    point_stride = struct.unpack_from("<H", data, 105)[0]
    point_count = (struct.unpack_from("<Q", data, 247)[0]
                   if data[25] >= 4
                   else struct.unpack_from("<I", data, 107)[0])
    assert point_offset + point_count * point_stride <= len(data), source
    return point_offset, point_format, point_stride, point_count


def las_classifications(path):
    data = path.read_bytes()
    point_offset, point_format, point_stride, point_count = las_layout(
        data, path)
    values = []
    for point in range(point_count):
        record = point_offset + point * point_stride
        if point_format <= 5:
            value = data[record + 15] & 0x1f
            values.append(0 if value == 12 else value)
        else:
            values.append(data[record + 16])
    return values


def las_user_data(path):
    data = path.read_bytes()
    point_offset, _, point_stride, point_count = las_layout(data, path)
    return [data[point_offset + point * point_stride + 17]
            for point in range(point_count)]


def classification_sources(root, oracle, source):
    legacy = root / "legacy-flags-class12.las"
    legacy_data = bytearray(source.read_bytes())
    offset, point_format, stride, count = las_layout(legacy_data, source)
    assert point_format == 3, point_format
    for point in range(count):
        flags = ((point % 8) << 5) & 0xe0
        legacy_data[offset + point * stride + 15] = flags | 12
    legacy.write_bytes(legacy_data)

    sources = [("legacy-format3", legacy)]
    for point_format in (6, 7, 8):
        converted = root / f"modern-format{point_format}.las"
        conversion = root / f"modern-format{point_format}.json"
        conversion.write_text(json.dumps({"pipeline": [
            {"type": "readers.las", "filename": str(source)},
            {"type": "writers.las", "filename": str(converted),
             "minor_version": 4, "dataformat_id": point_format},
        ]}), encoding="utf-8")
        result = run([str(oracle), "pipeline", str(conversion)])
        assert result.returncode == 0, result.stderr
        modern_data = bytearray(converted.read_bytes())
        offset, actual_format, stride, count = las_layout(
            modern_data, converted)
        assert actual_format == point_format, (actual_format, point_format)
        for point in range(count):
            record = offset + point * stride
            modern_data[record + 15] = (point * 29 + 0xa5) & 0xff
            modern_data[record + 16] = (point * 37 + 12) & 0xff
        converted.write_bytes(modern_data)
        sources.append((f"modern-format{point_format}", converted))
    return sources


def faux_source(root, oracle, count):
    source = root / f"automatic-{count}.las"
    pipeline = root / f"automatic-source-{count}.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.faux", "bounds": "([0,100],[0,100],[0,20])",
         "count": count, "mode": "ramp"},
        {"type": "writers.las", "filename": str(source),
         "minor_version": 4, "dataformat_id": 7},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    _, point_format, point_stride, point_count = las_layout(
        source.read_bytes(), source)
    assert (point_format, point_stride, point_count) == (7, 36, count)
    return source


def radius_mutation_source(root, oracle):
    csv = root / "radius-mutation.csv"
    csv.write_text(
        "X,Y,Z,Classification\n"
        "0,0,0,1\n"
        "0.1,0,0,2\n"
        "0,0.1,0,3\n"
        "10,10,10,4\n",
        encoding="utf-8")
    source = root / "radius-mutation.las"
    pipeline = root / "radius-mutation-source.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.text", "filename": str(csv)},
        {"type": "writers.las", "filename": str(source),
         "minor_version": 4, "dataformat_id": 7},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    _, point_format, point_stride, point_count = las_layout(
        source.read_bytes(), source)
    assert (point_format, point_stride, point_count) == (7, 36, 4)
    return source


def radius_boundary_source(root, oracle):
    csv = root / "radius-boundary.csv"
    csv.write_text(
        "X,Y,Z,Classification\n"
        "0,0,0,1\n"
        "0.5,0,0,2\n"
        "1,0,0,3\n",
        encoding="utf-8")
    source = root / "radius-boundary.las"
    pipeline = root / "radius-boundary-source.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.text", "filename": str(csv)},
        {"type": "writers.las", "filename": str(source),
         "minor_version": 4, "dataformat_id": 7},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    _, point_format, point_stride, point_count = las_layout(
        source.read_bytes(), source)
    assert (point_format, point_stride, point_count) == (7, 36, 3)
    return source


def unsupported_vlr_source(root, source):
    data = bytearray(source.read_bytes())
    point_offset, point_format, point_stride, _ = las_layout(data, source)
    assert (point_format, point_stride) == (7, 36)
    assert struct.unpack_from("<I", data, 100)[0] == 0
    vlr = bytearray(55)
    vlr[2:10] = b"PDG_TEST"
    struct.pack_into("<H", vlr, 18, 1)
    struct.pack_into("<H", vlr, 20, 1)
    vlr[22:33] = b"unsupported"
    shifted = len(vlr)
    widened = data[:point_offset] + vlr + data[point_offset:]
    struct.pack_into("<I", widened, 96, point_offset + shifted)
    struct.pack_into("<I", widened, 100, 1)
    for offset in (227, 235):
        trailing = struct.unpack_from("<Q", widened, offset)[0]
        if trailing:
            struct.pack_into("<Q", widened, offset, trailing + shifted)
    destination = root / "unsupported-vlr.las"
    destination.write_bytes(widened)
    return destination


def automatic_environment():
    environment = os.environ.copy()
    environment[
        "PDG_REQUIRE_AUTOMATIC_OUTLIER_NNDISTANCE_RESIDENT"] = "1"
    return environment


def automatic_radius_composition_environment():
    environment = os.environ.copy()
    environment[
        "PDG_REQUIRE_AUTOMATIC_RADIUS_OUTLIER_RADIALDENSITY_RESIDENT"] = "1"
    return environment


def run_automatic_case(root, pdg, oracle, source):
    candidate_output = root / "automatic-candidate.las"
    oracle_output = root / "automatic-oracle.las"
    candidate_pipeline = write_pipeline(
        root / "automatic-candidate.json", source, candidate_output)
    oracle_pipeline = write_pipeline(
        root / "automatic-oracle.json", source, oracle_output)
    candidate = run([
        str(pdg), "pipeline", str(candidate_pipeline),
    ], automatic_environment())
    reference = run([
        str(oracle), "pipeline", str(oracle_pipeline),
    ])
    assert candidate.returncode == reference.returncode == 0, (
        candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout
    assert candidate.stderr == reference.stderr
    assert candidate_output.read_bytes() == oracle_output.read_bytes()

    disabled_output = root / "automatic-disabled.las"
    disabled_pipeline = write_pipeline(
        root / "automatic-disabled.json", source, disabled_output)
    disabled_environment = automatic_environment()
    disabled_environment["PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
    disabled = run([
        str(pdg), "pipeline", str(disabled_pipeline),
    ], disabled_environment)
    assert disabled.returncode == 124, disabled
    assert "required automatic outlier/NNDistance resident path" in (
        disabled.stderr), disabled.stderr
    assert not disabled_output.exists()

    proof_output = root / "automatic-proof-failure.las"
    proof_pipeline = write_pipeline(
        root / "automatic-proof-failure.json", source, proof_output)
    proof_environment = automatic_environment()
    proof_environment["PDG_TEST_AUTOMATIC_OUTLIER_PROOF_FAILURE"] = "1"
    proof_failure = run([
        str(pdg), "pipeline", str(proof_pipeline),
    ], proof_environment)
    assert proof_failure.returncode != 0, proof_failure
    assert "automatic outlier/NNDistance resident execution proof failed" in (
        proof_failure.stderr), proof_failure.stderr
    assert not proof_output.exists()


def run_automatic_radius_composition_case(root, pdg, oracle, source):
    candidate_output = root / "automatic-radius-composition-candidate.las"
    oracle_output = root / "automatic-radius-composition-oracle.las"
    candidate_pipeline = write_radius_composition_pipeline(
        root / "automatic-radius-composition-candidate.json", source,
        candidate_output)
    oracle_pipeline = write_radius_composition_pipeline(
        root / "automatic-radius-composition-oracle.json", source,
        oracle_output)
    candidate = run([
        str(pdg), "pipeline", str(candidate_pipeline),
    ], automatic_radius_composition_environment())
    reference = run([
        str(oracle), "pipeline", str(oracle_pipeline),
    ])
    assert candidate.returncode == reference.returncode == 0, (
        candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout
    assert candidate.stderr == reference.stderr
    assert candidate_output.read_bytes() == oracle_output.read_bytes()

    disabled_output = root / "automatic-radius-composition-disabled.las"
    disabled_pipeline = write_radius_composition_pipeline(
        root / "automatic-radius-composition-disabled.json", source,
        disabled_output)
    disabled_environment = automatic_radius_composition_environment()
    disabled_environment["PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
    disabled = run([
        str(pdg), "pipeline", str(disabled_pipeline),
    ], disabled_environment)
    assert disabled.returncode == 124, disabled
    assert "required automatic radius-outlier/radial-density resident path" in (
        disabled.stderr), disabled.stderr
    assert not disabled_output.exists()
    assert_no_direct_output_temp(disabled_output)

    proof_output = root / "automatic-radius-composition-proof-failure.las"
    proof_pipeline = write_radius_composition_pipeline(
        root / "automatic-radius-composition-proof-failure.json", source,
        proof_output)
    proof_environment = automatic_radius_composition_environment()
    proof_environment[
        "PDG_TEST_AUTOMATIC_RADIUS_COMPOSITION_PROOF_FAILURE"] = "1"
    proof_failure = run([
        str(pdg), "pipeline", str(proof_pipeline),
    ], proof_environment)
    assert proof_failure.returncode != 0, proof_failure
    assert "automatic radius-outlier/radial-density resident execution " in (
        proof_failure.stderr), proof_failure.stderr
    assert not proof_output.exists()
    assert_no_direct_output_temp(proof_output)


def run_radius_composition_fallback_case(root, name, pdg, oracle, source,
                                         density_radius=1.01):
    candidate_output = root / f"{name}-candidate.las"
    oracle_output = root / f"{name}-oracle.las"
    candidate_pipeline = write_radius_composition_pipeline(
        root / f"{name}-candidate.json", source, candidate_output,
        density_radius=density_radius)
    oracle_pipeline = write_radius_composition_pipeline(
        root / f"{name}-oracle.json", source, oracle_output,
        density_radius=density_radius)
    candidate = run([str(pdg), "pipeline", str(candidate_pipeline)])
    reference = run([str(oracle), "pipeline", str(oracle_pipeline)])
    assert candidate.returncode == reference.returncode == 0, (
        name, candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout, name
    assert candidate.stderr == reference.stderr, name
    assert candidate_output.read_bytes() == oracle_output.read_bytes(), name


def run_exact_case(root, name, pdg, oracle, source,
                   expected_calibrated=False):
    candidate_output = root / f"{name}-candidate.las"
    oracle_output = root / f"{name}-oracle.las"
    candidate_stats = root / f"{name}-candidate-stats.json"
    candidate_pipeline = write_pipeline(
        root / f"{name}-candidate.json", source, candidate_output)
    oracle_pipeline = write_pipeline(
        root / f"{name}-oracle.json", source, oracle_output)
    candidate = run([
        str(pdg), "resident", str(candidate_pipeline), "--stats",
        str(candidate_stats),
    ], exact_environment())
    reference = run([
        str(oracle), "pipeline", str(oracle_pipeline),
    ])
    assert candidate.returncode == reference.returncode == 0, (
        name, candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout, (
        name, candidate.stdout, reference.stdout)
    assert candidate.stderr == reference.stderr, (
        name, candidate.stderr, reference.stderr)
    assert candidate_output.read_bytes() == oracle_output.read_bytes(), name
    assert las_classifications(candidate_output) != las_classifications(
        source), f"{name} must positively exercise Classification output"

    report = json.loads(candidate_stats.read_text(encoding="utf-8"))
    execution = report["execution"]
    assert execution["executor"] == (
        "planner_resident_shared_index_direct_las"), execution
    assert execution["direct_las_output"] is True, execution
    assert execution["direct_las_resident_source"] is True, execution
    assert execution["direct_las_record_summary"] is True, execution
    assert execution["direct_las_host_xyz_mirror"] is False, execution
    assert execution["boundary_accounting_matches_prediction"] is True, (
        execution)
    assert execution[
        "selected_device_calibration_matches_executor"] is (
            expected_calibrated), execution
    assert report["placement"]["reason"] == (
        "device_faster" if expected_calibrated else "uncalibrated_stage"), (
            report)
    assert execution["selected_regions"] == [0], execution
    assert execution["index_builds"] == {
        "matches_prediction": True,
        "observed": 1,
        "predicted": 1,
    }, execution["index_builds"]
    source_data = source.read_bytes()
    _, _, point_stride, point_count = las_layout(source_data, source)
    raw_record_bytes = point_stride * point_count
    assert any(
        event["kind"] == "host_to_device" and
        event["bytes"] == raw_record_bytes
        for event in execution["events"]), (
            raw_record_bytes, execution["events"])
    return report


def run_standalone_exact_case(root, name, pdg, oracle, source):
    candidate_output = root / f"{name}-standalone-candidate.las"
    oracle_output = root / f"{name}-standalone-oracle.las"
    candidate_stats = root / f"{name}-standalone-candidate-stats.json"
    candidate_pipeline = write_standalone_pipeline(
        root / f"{name}-standalone-candidate.json", source,
        candidate_output)
    oracle_pipeline = write_standalone_pipeline(
        root / f"{name}-standalone-oracle.json", source, oracle_output)
    candidate = run([
        str(pdg), "resident", str(candidate_pipeline), "--stats",
        str(candidate_stats),
    ], exact_environment())
    reference = run([
        str(oracle), "pipeline", str(oracle_pipeline),
    ])
    assert candidate.returncode == reference.returncode == 0, (
        name, candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout, (
        name, candidate.stdout, reference.stdout)
    assert candidate.stderr == reference.stderr, (
        name, candidate.stderr, reference.stderr)
    assert candidate_output.read_bytes() == oracle_output.read_bytes(), name
    assert las_classifications(candidate_output) != las_classifications(
        source), f"{name} must positively exercise Classification output"

    report = json.loads(candidate_stats.read_text(encoding="utf-8"))
    execution = report["execution"]
    assert execution["executor"] == (
        "planner_resident_shared_index_direct_las"), execution
    assert execution["schedule"]["pipeline_class"] == (
        "whole_view_neighborhood"), execution
    assert execution["direct_las_output"] is True, execution
    assert execution["direct_las_resident_source"] is True, execution
    assert execution["direct_las_record_summary"] is True, execution
    assert execution["direct_las_host_xyz_mirror"] is False, execution
    assert execution["boundary_accounting_matches_prediction"] is True, (
        execution)
    assert execution["selected_regions"] == [0], execution
    assert execution["selected_stage_ids"] == [1], execution
    assert execution["index_builds"] == {
        "matches_prediction": True,
        "observed": 1,
        "predicted": 1,
    }, execution["index_builds"]
    assert execution["totals"]["fallback_boundary"]["count"] == 0, (
        execution)
    _, _, _, point_count = las_layout(source.read_bytes(), source)
    assert execution["totals"]["boundary_spill"] == {
        "bytes": point_count,
        "count": 1,
        "packing_bytes": 0,
    }, execution
    return report


def run_radius_standalone_exact_case(root, name, pdg, oracle, source,
                                     require_mutation=False):
    candidate_output = root / f"{name}-radius-candidate.las"
    oracle_output = root / f"{name}-radius-oracle.las"
    candidate_stats = root / f"{name}-radius-candidate-stats.json"
    candidate_pipeline = write_radius_standalone_pipeline(
        root / f"{name}-radius-candidate.json", source, candidate_output)
    oracle_pipeline = write_radius_standalone_pipeline(
        root / f"{name}-radius-oracle.json", source, oracle_output)
    candidate = run([
        str(pdg), "resident", str(candidate_pipeline), "--stats",
        str(candidate_stats),
    ], exact_environment())
    reference = run([
        str(oracle), "pipeline", str(oracle_pipeline),
    ])
    assert candidate.returncode == reference.returncode == 0, (
        name, candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout, (
        name, candidate.stdout, reference.stdout)
    assert candidate.stderr == reference.stderr, (
        name, candidate.stderr, reference.stderr)
    assert candidate_output.read_bytes() == oracle_output.read_bytes(), name
    if require_mutation:
        assert las_classifications(candidate_output) != las_classifications(
            source), f"{name} must positively exercise radius Classification"

    report = json.loads(candidate_stats.read_text(encoding="utf-8"))
    execution = report["execution"]
    assert execution["executor"] == (
        "planner_resident_shared_index_direct_las"), execution
    assert execution["schedule"]["pipeline_class"] == (
        "whole_view_neighborhood"), execution
    assert execution["direct_las_output"] is True, execution
    assert execution["direct_las_resident_source"] is True, execution
    assert execution["direct_las_record_summary"] is True, execution
    assert execution["direct_las_host_xyz_mirror"] is False, execution
    assert execution["boundary_accounting_matches_prediction"] is True, (
        execution)
    assert execution["selected_regions"] == [0], execution
    assert execution["selected_stage_ids"] == [1], execution
    assert execution["index_builds"] == {
        "matches_prediction": True,
        "observed": 1,
        "predicted": 1,
    }, execution["index_builds"]
    assert execution["totals"]["fallback_boundary"]["count"] == 0, (
        execution)
    _, _, _, point_count = las_layout(source.read_bytes(), source)
    assert report["placement"]["predicted"]["stage_result_bytes"] == (
        point_count * 4), report["placement"]
    assert execution["totals"]["boundary_spill"] == {
        "bytes": point_count,
        "count": 1,
        "packing_bytes": 0,
    }, execution
    assert any(
        event["kind"] == "device_to_host" and
        event["bytes"] == point_count * 4
        for event in execution["events"]), execution["events"]
    return report


def run_radius_composition_exact_case(root, pdg, oracle, source):
    candidate_output = root / "radius-composition-candidate.las"
    oracle_output = root / "radius-composition-oracle.las"
    candidate_stats = root / "radius-composition-candidate-stats.json"
    candidate_pipeline = write_radius_composition_pipeline(
        root / "radius-composition-candidate.json", source,
        candidate_output)
    oracle_pipeline = write_radius_composition_pipeline(
        root / "radius-composition-oracle.json", source, oracle_output)
    environment = radius_composition_environment()
    candidate = run([
        str(pdg), "resident", str(candidate_pipeline), "--stats",
        str(candidate_stats),
    ], environment)
    reference = run([
        str(oracle), "pipeline", str(oracle_pipeline),
    ])
    assert candidate.returncode == reference.returncode == 0, (
        candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout
    assert candidate.stderr == reference.stderr
    assert candidate_output.read_bytes() == oracle_output.read_bytes()
    assert las_classifications(candidate_output) != las_classifications(
        source), "composition must positively exercise Classification"
    assert las_user_data(candidate_output) != las_user_data(
        source), "composition must positively exercise UserData"

    report = json.loads(candidate_stats.read_text(encoding="utf-8"))
    execution = report["execution"]
    assert execution["executor"] == (
        "planner_resident_shared_index_direct_las"), execution
    assert execution["schedule"]["pipeline_class"] == (
        "whole_view_neighborhood"), execution
    assert execution["direct_las_output"] is True, execution
    assert execution["direct_las_resident_source"] is True, execution
    assert execution["direct_las_record_summary"] is True, execution
    assert execution["direct_las_host_xyz_mirror"] is False, execution
    assert execution["boundary_accounting_matches_prediction"] is True, (
        execution)
    assert execution["selected_regions"] == [0], execution
    assert execution["selected_stage_ids"] == [1, 2, 3], execution
    assert execution["index_builds"] == {
        "matches_prediction": True,
        "observed": 1,
        "predicted": 1,
    }, execution["index_builds"]
    assert execution["totals"]["fallback_boundary"]["count"] == 0, (
        execution)
    _, _, _, point_count = las_layout(source.read_bytes(), source)
    assert report["placement"]["predicted"]["stage_result_bytes"] == (
        point_count * 4), report["placement"]
    assert execution["totals"]["boundary_spill"] == {
        "bytes": point_count * 10,
        "count": 1,
        "packing_bytes": 0,
    }, execution
    device_to_host = [event["bytes"] for event in execution["events"]
                      if event["kind"] == "device_to_host"]
    assert device_to_host == [point_count * 4, point_count], (
        execution["events"])
    assert point_count * 8 not in device_to_host, execution["events"]
    return report


def run_standalone_incomplete_repair_case(root, pdg, oracle, source):
    candidate_output = root / "standalone-incomplete-candidate.las"
    oracle_output = root / "standalone-incomplete-oracle.las"
    candidate_stats = root / "standalone-incomplete-candidate-stats.json"
    candidate_pipeline = write_standalone_pipeline(
        root / "standalone-incomplete-candidate.json", source,
        candidate_output)
    oracle_pipeline = write_standalone_pipeline(
        root / "standalone-incomplete-oracle.json", source, oracle_output)
    environment = exact_environment()
    environment["PDG_FORCE_UNIFORM_GRID"] = "1"
    environment["PDG_KNN_DEVICE_SHELL_BUDGET"] = "1"
    environment["PDG_DISABLE_OUTLIER_DEVICE_REPAIR"] = "1"
    candidate = run([
        str(pdg), "resident", str(candidate_pipeline), "--stats",
        str(candidate_stats),
    ], environment)
    reference = run([
        str(oracle), "pipeline", str(oracle_pipeline),
    ])
    assert candidate.returncode == reference.returncode == 0, (
        candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout
    assert candidate.stderr == reference.stderr
    assert candidate_output.read_bytes() == oracle_output.read_bytes()
    execution = json.loads(
        candidate_stats.read_text(encoding="utf-8"))["execution"]
    repair = execution["exact_host_repair"]["statistical_outlier"]
    assert repair["incomplete_rows"] > 0, repair
    assert repair["repaired_rows"] == repair["incomplete_rows"], repair
    assert repair["seconds"] > 0.0, repair


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: resident_outlier_direct_gpu_test.py PDG PDAL INPUT")
    pdg = pathlib.Path(sys.argv[1]).resolve()
    oracle = pathlib.Path(sys.argv[2]).resolve()
    source = pathlib.Path(sys.argv[3]).resolve()

    with tempfile.TemporaryDirectory(
            prefix="pdg-resident-outlier-direct-") as temporary:
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

        exact_sources = classification_sources(root, oracle, source)
        for name, exact_source in exact_sources:
            run_exact_case(root, name, pdg, oracle, exact_source)
            run_standalone_exact_case(root, name, pdg, oracle, exact_source)
            run_radius_standalone_exact_case(
                root, name, pdg, oracle, exact_source)
        radius_positive_source = radius_mutation_source(root, oracle)
        run_radius_standalone_exact_case(
            root, "radius-positive", pdg, oracle,
            radius_positive_source, require_mutation=True)
        run_radius_composition_exact_case(
            root, pdg, oracle, radius_positive_source)
        run_radius_standalone_exact_case(
            root, "radius-boundary", pdg, oracle,
            radius_boundary_source(root, oracle), require_mutation=True)
        run_standalone_incomplete_repair_case(
            root, pdg, oracle, exact_sources[0][1])

        below_source = faux_source(root, oracle, 25_000)
        below_output = root / "automatic-below.las"
        below_pipeline = write_pipeline(
            root / "automatic-below.json", below_source, below_output)
        below = run([
            str(pdg), "pipeline", str(below_pipeline),
        ], automatic_environment())
        assert below.returncode == 124, below
        assert "required automatic outlier/NNDistance resident path" in (
            below.stderr), below.stderr
        assert not below_output.exists()

        automatic_source = faux_source(root, oracle, 50_000)
        run_automatic_case(root, pdg, oracle, automatic_source)
        run_exact_case(root, "calibrated-50k", pdg, oracle,
                       automatic_source, expected_calibrated=True)

        run_radius_composition_fallback_case(
            root, "automatic-radius-composition-below", pdg, oracle,
            automatic_source)
        radius_below_output = root / "automatic-radius-composition-required-below.las"
        radius_below_pipeline = write_radius_composition_pipeline(
            root / "automatic-radius-composition-required-below.json",
            automatic_source, radius_below_output)
        radius_below = run([
            str(pdg), "pipeline", str(radius_below_pipeline),
        ], automatic_radius_composition_environment())
        assert radius_below.returncode == 124, radius_below
        assert not radius_below_output.exists()

        radius_automatic_source = faux_source(root, oracle, 250_000)
        run_automatic_radius_composition_case(
            root, pdg, oracle, radius_automatic_source)
        run_radius_composition_fallback_case(
            root, "automatic-radius-composition-shape-fallback", pdg,
            oracle, radius_automatic_source, density_radius=1.0)
        radius_mismatch_output = root / "automatic-radius-composition-mismatch.las"
        radius_mismatch_pipeline = write_radius_composition_pipeline(
            root / "automatic-radius-composition-mismatch.json",
            radius_automatic_source, radius_mismatch_output,
            density_radius=1.0)
        radius_mismatch = run([
            str(pdg), "pipeline", str(radius_mismatch_pipeline),
        ], automatic_radius_composition_environment())
        assert radius_mismatch.returncode == 124, radius_mismatch
        assert not radius_mismatch_output.exists()

        automatic_outside_output = root / "automatic-outside.las"
        automatic_outside_pipeline = write_pipeline(
            root / "automatic-outside.json", automatic_source,
            automatic_outside_output, mean_k=9)
        automatic_outside = run([
            str(pdg), "pipeline", str(automatic_outside_pipeline),
        ], automatic_environment())
        assert automatic_outside.returncode == 124, automatic_outside
        assert not automatic_outside_output.exists()

        disabled_output = root / "disabled.las"
        disabled_pipeline = write_pipeline(
            root / "disabled.json", exact_sources[0][1], disabled_output)
        disabled_environment = exact_environment()
        disabled_environment["PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
        disabled = run([
            str(pdg), "resident", str(disabled_pipeline),
        ], disabled_environment)
        assert disabled.returncode != 0, disabled
        assert "required direct LAS resident source path was not used" in (
            disabled.stderr), disabled.stderr
        assert not disabled_output.exists()

        standalone_disabled_output = root / "standalone-disabled.las"
        standalone_disabled_pipeline = write_standalone_pipeline(
            root / "standalone-disabled.json", exact_sources[0][1],
            standalone_disabled_output)
        standalone_disabled = run([
            str(pdg), "resident", str(standalone_disabled_pipeline),
        ], disabled_environment)
        assert standalone_disabled.returncode != 0, standalone_disabled
        assert "required direct LAS resident source path was not used" in (
            standalone_disabled.stderr), standalone_disabled.stderr
        assert not standalone_disabled_output.exists()
        assert_no_direct_output_temp(standalone_disabled_output)

        radius_disabled_output = root / "radius-disabled.las"
        radius_disabled_pipeline = write_radius_standalone_pipeline(
            root / "radius-disabled.json", exact_sources[0][1],
            radius_disabled_output)
        radius_disabled = run([
            str(pdg), "resident", str(radius_disabled_pipeline),
        ], disabled_environment)
        assert radius_disabled.returncode != 0, radius_disabled
        assert "required direct LAS resident source path was not used" in (
            radius_disabled.stderr), radius_disabled.stderr
        assert not radius_disabled_output.exists()
        assert_no_direct_output_temp(radius_disabled_output)

        composition_disabled_output = root / "composition-disabled.las"
        composition_disabled_pipeline = write_radius_composition_pipeline(
            root / "composition-disabled.json", exact_sources[0][1],
            composition_disabled_output)
        composition_disabled = run([
            str(pdg), "resident", str(composition_disabled_pipeline),
        ], {**radius_composition_environment(),
            "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE": "1"})
        assert composition_disabled.returncode != 0, composition_disabled
        assert "required direct LAS resident source path was not used" in (
            composition_disabled.stderr), composition_disabled.stderr
        assert not composition_disabled_output.exists()
        assert_no_direct_output_temp(composition_disabled_output)

        unsupported_radius_output = root / "radius-unsupported-source.las"
        unsupported_radius_pipeline = write_radius_standalone_pipeline(
            root / "radius-unsupported-source.json",
            unsupported_vlr_source(root, exact_sources[2][1]),
            unsupported_radius_output)
        unsupported_radius = run([
            str(pdg), "resident", str(unsupported_radius_pipeline),
        ], exact_environment())
        assert unsupported_radius.returncode != 0, unsupported_radius
        assert "required direct LAS resident source path was not used" in (
            unsupported_radius.stderr), unsupported_radius.stderr
        assert not unsupported_radius_output.exists()
        assert_no_direct_output_temp(unsupported_radius_output)

        outside_output = root / "outside.las"
        outside_pipeline = write_pipeline(
            root / "outside.json", exact_sources[0][1], outside_output,
            mean_k=9)
        outside = run([
            str(pdg), "resident", str(outside_pipeline),
        ], exact_environment())
        assert outside.returncode != 0, outside
        assert not outside_output.exists()

        standalone_outside_output = root / "standalone-outside.las"
        standalone_outside_pipeline = write_standalone_pipeline(
            root / "standalone-outside.json", exact_sources[0][1],
            standalone_outside_output, mean_k=9)
        standalone_outside = run([
            str(pdg), "resident", str(standalone_outside_pipeline),
        ], exact_environment())
        assert standalone_outside.returncode != 0, standalone_outside
        assert not standalone_outside_output.exists()
        assert_no_direct_output_temp(standalone_outside_output)

        for option_name, option_values in (
                ("multiplier", {"multiplier": 2.5}),
                ("class", {"classification": 8}),
                ("extra-option", {
                    "extra_options": {"where": "Classification >= 0"}})):
            rejected_output = root / f"standalone-{option_name}.las"
            rejected_pipeline = write_standalone_pipeline(
                root / f"standalone-{option_name}.json",
                exact_sources[0][1], rejected_output, **option_values)
            rejected = run([
                str(pdg), "resident", str(rejected_pipeline),
            ], exact_environment())
            assert rejected.returncode != 0, (option_name, rejected)
            assert not rejected_output.exists(), option_name
            assert_no_direct_output_temp(rejected_output)

        composition_mismatch_output = root / "composition-mismatch.las"
        composition_mismatch_pipeline = write_radius_composition_pipeline(
            root / "composition-mismatch.json", exact_sources[0][1],
            composition_mismatch_output, density_radius=1.0)
        composition_mismatch = run([
            str(pdg), "resident", str(composition_mismatch_pipeline),
        ], radius_composition_environment())
        assert composition_mismatch.returncode != 0, composition_mismatch
        assert not composition_mismatch_output.exists()
        assert_no_direct_output_temp(composition_mismatch_output)

        for option_name, option_values in (
                ("radius", {"radius": 1.01}),
                ("min-k", {"min_k": 3}),
                ("class", {"extra_options": {"class": 8}}),
                ("extra-option", {
                    "extra_options": {"where": "Classification >= 0"}})):
            rejected_output = root / f"radius-{option_name}.las"
            rejected_pipeline = write_radius_standalone_pipeline(
                root / f"radius-{option_name}.json",
                exact_sources[0][1], rejected_output, **option_values)
            rejected = run([
                str(pdg), "resident", str(rejected_pipeline),
            ], exact_environment())
            assert rejected.returncode != 0, (option_name, rejected)
            assert not rejected_output.exists(), option_name
            assert_no_direct_output_temp(rejected_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
