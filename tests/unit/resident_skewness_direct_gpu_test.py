#!/usr/bin/env python3
"""Exact process gate for the direct-LAS skewness composition endpoint."""

import json
import hashlib
import math
import os
import pathlib
import shutil
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


def assert_no_direct_output_temp(output):
    assert not list(output.parent.glob(
        output.name + ".pdg-resident-output-*")), output


def las_layout(path):
    data = path.read_bytes()
    assert data[:4] == b"LASF" and len(data) >= 227, path
    point_offset = struct.unpack_from("<I", data, 96)[0]
    point_format = data[104] & 0x0f
    assert not (data[104] & 0x80), "compressed input is outside this gate"
    point_stride = struct.unpack_from("<H", data, 105)[0]
    point_count = (struct.unpack_from("<Q", data, 247)[0]
                   if data[25] >= 4 else struct.unpack_from("<I", data, 107)[0])
    assert point_offset + point_count * point_stride <= len(data), path
    return point_offset, point_format, point_stride, point_count


def exact_environment():
    environment = os.environ.copy()
    environment["PDG_REQUIRE_DIRECT_SKEWNESS_COMPOSITION"] = "1"
    return environment


def automatic_environment():
    environment = os.environ.copy()
    environment["PDG_REQUIRE_AUTOMATIC_SKEWNESS_RESIDENT"] = "1"
    return environment


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mem_available_bytes():
    for line in pathlib.Path("/proc/meminfo").read_text(
            encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("MemAvailable missing from /proc/meminfo")


def write_text_source(path, points, target):
    path.write_text("X,Y,Z,Classification\n" + "\n".join(
        f"{x},{y},{z},{classification}" for x, y, z, classification in points
    ) + "\n", encoding="utf-8")
    return target


def write_source_pipeline(path, csv, oracle_output):
    path.write_text(json.dumps({"pipeline": [
        {"type": "readers.text", "filename": str(csv)},
        {"type": "writers.las", "filename": str(oracle_output),
         "minor_version": 4, "dataformat_id": 7},
    ]}), encoding="utf-8")
    return path


def comparator_unique_source(root, oracle):
    points = (
        (0.0, 0.0, 5.0, 7),
        (1.0, 1.0, 0.0, 7),
        (2.0, 2.0, 8.75, 7),
        (3.0, 3.0, 2.5, 7),
        (4.0, 4.0, 7.5, 7),
        (5.0, 5.0, 1.25, 7),
        (6.0, 6.0, 6.25, 7),
        (7.0, 7.0, 3.75, 7),
    )
    z_values = [point[2] for point in points]
    assert len(set(z_values)) == len(z_values), points
    assert z_values != sorted(z_values), points

    source_csv = root / "strict-unique-z.csv"
    write_text_source(source_csv, points, source_csv)
    source = root / "strict-unique-z.las"
    source_pipeline = write_source_pipeline(
        root / "strict-unique-z.json", source_csv, source)
    oracle_source = run([str(oracle), "pipeline", str(source_pipeline)])
    assert oracle_source.returncode == 0, oracle_source.stderr
    assert las_layout(source)[1:] == (7, 36, len(points)), las_layout(source)
    return source


def duplicate_source(root, oracle):
    points = (
        (0.0, 0.0, 0.0, 7),
        (1.0, 1.0, 1.25, 7),
        (2.0, 2.0, 1.25, 7),
        (3.0, 3.0, 2.5, 7),
        (4.0, 4.0, 3.75, 7),
        (5.0, 5.0, 5.0, 7),
    )
    source_csv = root / "duplicate-z.csv"
    write_text_source(source_csv, points, source_csv)
    source = root / "duplicate-z-source.las"
    source_pipeline = write_source_pipeline(
        root / "duplicate-z.json", source_csv, source)
    oracle_source = run([str(oracle), "pipeline", str(source_pipeline)])
    assert oracle_source.returncode == 0, oracle_source.stderr
    assert las_layout(source)[1:] == (7, 36, len(points)), las_layout(source)
    return source


def write_skewness_pipeline(path, source, destination):
    path.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.skewnessbalancing"},
        {"type": "writers.las", "filename": str(destination),
         "extra_dims": "all"},
    ]}), encoding="utf-8")
    return path


def write_skewness_pipeline_with_options(path, source, destination, options):
    skewness = {"type": "filters.skewnessbalancing", **options}
    path.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        skewness,
        {"type": "writers.las", "filename": str(destination),
         "extra_dims": "all"},
    ]}), encoding="utf-8")
    return path


def exact_case(root, pdg, oracle, source):
    candidate_output = root / "candidate.las"
    oracle_output = root / "oracle.las"
    candidate_stats = root / "candidate-stats.json"
    candidate_pipeline = write_skewness_pipeline(
        root / "candidate.json", source, candidate_output)
    oracle_pipeline = write_skewness_pipeline(
        root / "oracle.json", source, oracle_output)
    candidate = run([str(pdg), "resident", str(candidate_pipeline), "--stats",
                     str(candidate_stats)], exact_environment())
    reference = run([str(oracle), "pipeline", str(oracle_pipeline)])
    assert candidate.returncode == reference.returncode == 0, (
        candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout, (
        candidate.stdout, reference.stdout)
    assert candidate.stderr == reference.stderr, (
        candidate.stderr, reference.stderr)
    assert candidate_output.read_bytes() == oracle_output.read_bytes()

    _, point_format, point_stride, point_count = las_layout(candidate_output)
    assert (point_format, point_stride, point_count) == (7, 36, 8), (
        point_format, point_stride, point_count)

    report = json.loads(candidate_stats.read_text(encoding="utf-8"))
    execution = report["execution"]
    assert execution["executor"] == (
        "planner_resident_global_order_direct_las"), execution
    assert execution["direct_las_output"] is True, execution
    assert execution["direct_las_resident_source"] is True, execution
    assert execution["direct_permuted_classification_output"] is True, execution
    assert execution["direct_las_record_summary"] is False, execution
    assert execution["direct_las_host_xyz_mirror"] is False, execution
    assert execution["terminal_spill_elided"] is True, execution
    assert execution["selected_regions"] == [0], execution
    assert execution["selected_stage_ids"] == [1], execution
    assert execution["index_builds"] == {
        "predicted": 0, "observed": 0, "matches_prediction": True,
    }, execution["index_builds"]
    expected = point_count * 8
    host_to_device = [event["bytes"] for event in execution["events"]
                      if event["kind"] == "host_to_device"]
    device_to_host = [event["bytes"] for event in execution["events"]
                      if event["kind"] == "device_to_host"]
    assert host_to_device == [expected], (
        expected, host_to_device)
    assert device_to_host == [expected], (
        expected, device_to_host)


def failing_case(root, name, pdg, oracle, source, environment, options=None):
    output = root / f"{name}.las"
    pipeline = (
        write_skewness_pipeline(root / f"{name}.json", source, output)
        if options is None else
        write_skewness_pipeline_with_options(
            root / f"{name}.json", source, output, options))
    result = run([str(pdg), "resident", str(pipeline)], environment)
    assert result.returncode != 0, (name, result)
    assert not output.exists(), (name, output)
    assert_no_direct_output_temp(output)
    return result


def fallback_case(root, name, pdg, oracle, source, environment):
    candidate_output = root / f"{name}-candidate.las"
    oracle_output = root / f"{name}-oracle.las"
    stats = root / f"{name}-stats.json"
    candidate_pipeline = write_skewness_pipeline(
        root / f"{name}-candidate.json", source, candidate_output)
    oracle_pipeline = write_skewness_pipeline(
        root / f"{name}-oracle.json", source, oracle_output)
    candidate = run([str(pdg), "resident", str(candidate_pipeline),
                     "--stats", str(stats)], environment)
    reference = run([str(oracle), "pipeline", str(oracle_pipeline)])
    assert candidate.returncode == reference.returncode == 0, (
        name, candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout, name
    assert candidate.stderr == reference.stderr, name
    assert candidate_output.read_bytes() == oracle_output.read_bytes(), name
    execution = json.loads(stats.read_text(encoding="utf-8")).get("execution")
    assert (execution is None or
            (execution.get("direct_las_output") is False and
             execution.get(
                 "direct_permuted_classification_output") is not True)), (
                    name, execution)
    assert_no_direct_output_temp(candidate_output)


def existing_output_case(root, name, pdg, source, output):
    before = output.read_bytes() if output.is_file() else None
    pipeline = write_skewness_pipeline(
        root / f"{name}.json", source, output)
    result = run([str(pdg), "resident", str(pipeline)], exact_environment())
    assert result.returncode != 0, (name, result)
    if before is not None:
        assert output.read_bytes() == before, name
    assert_no_direct_output_temp(output)
    return result


def prefix_preserving_layout(root, oracle, source, count):
    header = source.read_bytes()[:227]
    scales = struct.unpack_from("<ddd", header, 131)
    offsets = struct.unpack_from("<ddd", header, 155)
    output = root / f"prefix-{count}.las"
    pipeline = root / f"prefix-{count}.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.head", "count": count},
        {"type": "writers.las", "filename": str(output),
         "minor_version": 4, "dataformat_id": 7,
         "scale_x": scales[0], "scale_y": scales[1],
         "scale_z": scales[2], "offset_x": offsets[0],
         "offset_y": offsets[1], "offset_z": offsets[2]},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    assert las_layout(output)[1:] == (7, 36, count)
    return output


def cap_plus_one_source(root, source):
    """Derive a valid 16,000,001-row LAS without touching the pinned source."""
    point_offset, point_format, stride, count = las_layout(source)
    assert (point_format, stride, count) == (7, 36, 16_000_000)
    output = root / "cap-plus-one-16000001.las"
    shutil.copyfile(source, output)
    with source.open("rb") as stream:
        header = bytearray(stream.read(point_offset))
        stream.seek(point_offset)
        record = bytearray(stream.read(stride))
    assert len(header) == point_offset and len(record) == stride

    scale_z = struct.unpack_from("<d", header, 147)[0]
    offset_z = struct.unpack_from("<d", header, 171)[0]
    max_z = struct.unpack_from("<d", header, 211)[0]
    assert math.isfinite(scale_z) and scale_z > 0.0
    max_raw_z = round((max_z - offset_z) / scale_z)
    assert max_raw_z < (1 << 31) - 1
    struct.pack_into("<i", record, 8, max_raw_z + 1)
    struct.pack_into("<d", header, 211,
                     offset_z + (max_raw_z + 1) * scale_z)
    struct.pack_into("<Q", header, 247, count + 1)

    return_number = record[14] & 0x0f
    assert 1 <= return_number <= 15
    return_offset = 255 + (return_number - 1) * 8
    prior_return_count = struct.unpack_from("<Q", header, return_offset)[0]
    struct.pack_into("<Q", header, return_offset, prior_return_count + 1)

    with output.open("r+b") as stream:
        stream.write(header)
        stream.seek(0, os.SEEK_END)
        stream.write(record)
    assert las_layout(output)[1:] == (7, 36, count + 1)
    assert sha256(output) == (
        "1b40ab4052fe89c17c9ea0b6be33d83f25be42b510f1a805cfc058a433f764bb")
    return output


def automatic_case(root, name, pdg, oracle, source, environment=None):
    candidate_output = root / f"{name}-candidate.las"
    option_free_output = root / f"{name}-option-free.las"
    oracle_output = root / f"{name}-oracle.las"
    candidate_pipeline = write_skewness_pipeline(
        root / f"{name}-candidate.json", source, candidate_output)
    option_free_pipeline = write_skewness_pipeline(
        root / f"{name}-option-free.json", source, option_free_output)
    oracle_pipeline = write_skewness_pipeline(
        root / f"{name}-oracle.json", source, oracle_output)
    candidate = run([str(pdg), "pipeline", str(candidate_pipeline)],
                    environment or automatic_environment())
    option_free = run([str(pdg), "pipeline", str(option_free_pipeline)])
    reference = run([str(oracle), "pipeline", str(oracle_pipeline)])
    assert candidate.returncode == option_free.returncode == (
        reference.returncode) == 0, (
            name, candidate.returncode, candidate.stderr,
            option_free.returncode, option_free.stderr,
            reference.returncode, reference.stderr)
    assert candidate.stdout == option_free.stdout == reference.stdout, name
    assert candidate.stderr == option_free.stderr == reference.stderr, name
    assert candidate_output.read_bytes() == oracle_output.read_bytes(), name
    assert option_free_output.read_bytes() == oracle_output.read_bytes(), name


def automatic_fallback_case(root, name, pdg, oracle, source,
                            default_environment=None, options=None,
                            output_suffix=".las"):
    shared_output = root / f"{name}-shared{output_suffix}"
    shared_pipeline = (
        write_skewness_pipeline(root / f"{name}-shared.json", source,
                                shared_output)
        if options is None else
        write_skewness_pipeline_with_options(
            root / f"{name}-shared.json", source, shared_output, options))
    candidate = run([str(pdg), "pipeline", str(shared_pipeline)],
                    default_environment)
    candidate_exists = shared_output.exists()
    candidate_bytes = shared_output.read_bytes() if candidate_exists else None
    shared_output.unlink(missing_ok=True)
    reference = run([str(oracle), "pipeline", str(shared_pipeline)])
    assert candidate.returncode == reference.returncode, (
        name, candidate, reference)
    assert candidate.stdout == reference.stdout, name
    assert candidate.stderr == reference.stderr, (
        name, candidate.stderr, reference.stderr)
    assert candidate_exists == shared_output.exists(), name
    if candidate_exists:
        assert candidate_bytes == shared_output.read_bytes(), name
    shared_output.unlink(missing_ok=True)

    required_output = root / f"{name}-required{output_suffix}"
    required_pipeline = (
        write_skewness_pipeline(root / f"{name}-required.json", source,
                                required_output)
        if options is None else
        write_skewness_pipeline_with_options(
            root / f"{name}-required.json", source, required_output, options))
    required_environment = automatic_environment()
    if default_environment:
        required_environment.update({
            key: value for key, value in default_environment.items()
            if key.startswith("PDG_")
        })
    required = run([str(pdg), "pipeline", str(required_pipeline)],
                   required_environment)
    assert required.returncode == 124, (name, required)
    assert not required_output.exists(), (name, required_output)
    assert_no_direct_output_temp(required_output)


def path_state(path):
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if not path.exists():
        return ("absent",)
    if path.is_file():
        return ("file", path.read_bytes())
    return ("other", path.stat().st_mode)


def assert_matching_result(name, candidate, reference):
    assert candidate.returncode == reference.returncode, (
        name, candidate, reference)
    assert candidate.stdout == reference.stdout, name
    assert candidate.stderr == reference.stderr, (
        name, candidate.stderr, reference.stderr)


def automatic_main(pdg, oracle):
    configured = os.environ.get("PDG_SKEWNESS_LAS_FILE")
    expected_hash = os.environ.get("PDG_SKEWNESS_LAS_SHA256")
    cap_configured = os.environ.get("PDG_SKEWNESS_CAP_LAS_FILE")
    cap_expected_hash = os.environ.get("PDG_SKEWNESS_CAP_LAS_SHA256")
    if (not configured or not expected_hash or not cap_configured or
            not cap_expected_hash):
        print("set PDG_SKEWNESS_LAS_FILE/PDG_SKEWNESS_LAS_SHA256 and "
              "PDG_SKEWNESS_CAP_LAS_FILE/PDG_SKEWNESS_CAP_LAS_SHA256 for "
              "the automatic skewness gate", file=sys.stderr)
        return SKIP
    source = pathlib.Path(configured).resolve()
    if not source.is_file() or source.suffix != ".las":
        raise AssertionError("PDG_SKEWNESS_LAS_FILE must name a lowercase LAS")
    if sha256(source) != expected_hash.lower():
        raise AssertionError("physical skewness source hash mismatch")
    point_offset, point_format, stride, count = las_layout(source)
    if (point_format, stride, count) != (7, 36, 1_000_000):
        raise AssertionError(
            "physical skewness source must be uncompressed format-7/36-byte "
            "LAS with exactly 1M points")
    source_bytes = source.read_bytes()
    scale_z = struct.unpack_from("<d", source_bytes, 147)[0]
    if not math.isfinite(scale_z) or scale_z == 0.0:
        raise AssertionError("physical skewness source has invalid Z scale")
    raw_z = {
        struct.unpack_from("<i", source_bytes,
                           point_offset + row * stride + 8)[0]
        for row in range(count)
    }
    if len(raw_z) != count:
        raise AssertionError("physical skewness source Z keys are not unique")
    cap_source = pathlib.Path(cap_configured).resolve()
    if not cap_source.is_file() or cap_source.suffix != ".las":
        raise AssertionError(
            "PDG_SKEWNESS_CAP_LAS_FILE must name a lowercase LAS")
    if sha256(cap_source) != cap_expected_hash.lower():
        raise AssertionError("physical skewness cap source hash mismatch")
    if las_layout(cap_source)[1:] != (7, 36, 16_000_000):
        raise AssertionError(
            "physical skewness cap source must be uncompressed "
            "format-7/36-byte LAS with exactly 16M points")
    if mem_available_bytes() < 8 * GIB:
        print("skewness gate skipped below 8 GiB MemAvailable", file=sys.stderr)
        return SKIP
    gpu = run(["nvidia-smi", "--query-gpu=memory.free",
               "--format=csv,noheader,nounits"])
    if gpu.returncode != 0 or int(gpu.stdout.splitlines()[0]) < 2048:
        print("skewness gate skipped without 2 GiB free VRAM", file=sys.stderr)
        return SKIP

    with tempfile.TemporaryDirectory(
            prefix="pdg-skewness-automatic-") as temporary:
        root = pathlib.Path(temporary)
        below = prefix_preserving_layout(root, oracle, source, 400_000)
        floor = prefix_preserving_layout(root, oracle, source, 450_000)
        automatic_fallback_case(root, "below", pdg, oracle, below)
        automatic_case(root, "floor", pdg, oracle, floor)
        automatic_case(root, "main", pdg, oracle, source)
        automatic_case(root, "cap", pdg, oracle, cap_source)
        cap_plus_one = cap_plus_one_source(root, cap_source)
        automatic_fallback_case(
            root, "cap-plus-one", pdg, oracle, cap_plus_one)

        exact_budget = 65 * 450_000
        exact_budget_environment = automatic_environment()
        exact_budget_environment["PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES"] = str(
            exact_budget)
        automatic_case(root, "exact-budget", pdg, oracle, floor,
                       exact_budget_environment)
        below_budget = os.environ.copy()
        below_budget["PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES"] = str(
            exact_budget - 1)
        automatic_fallback_case(root, "below-budget", pdg, oracle, floor,
                                below_budget)

        for name, variable in (
                ("disabled-source", "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"),
                ("preflight", "PDG_TEST_DIRECT_SKEWNESS_PREFLIGHT_FAILURE"),
                ("proof", "PDG_TEST_DIRECT_SKEWNESS_PROOF_FAILURE")):
            environment = os.environ.copy()
            environment[variable] = "1"
            automatic_fallback_case(root, name, pdg, oracle, floor,
                                    environment)

        duplicate = root / "duplicate-z-400k.las"
        duplicate_bytes = bytearray(floor.read_bytes())
        floor_offset, _, floor_stride, _ = las_layout(floor)
        duplicate_bytes[floor_offset + floor_stride + 8:
                        floor_offset + floor_stride + 12] = (
            duplicate_bytes[floor_offset + 8:floor_offset + 12])
        duplicate.write_bytes(duplicate_bytes)
        automatic_fallback_case(root, "duplicate", pdg, oracle, duplicate)

        nonfinite = root / "nonfinite-z-scale-400k.las"
        nonfinite_bytes = bytearray(floor.read_bytes())
        struct.pack_into("<d", nonfinite_bytes, 147, float("nan"))
        nonfinite.write_bytes(nonfinite_bytes)
        automatic_fallback_case(root, "nonfinite", pdg, oracle, nonfinite)

        automatic_fallback_case(
            root, "option", pdg, oracle, floor, options={"ground_class": 3})

        uppercase_source = root / "uppercase-source.LAS"
        uppercase_source.write_bytes(floor.read_bytes())
        automatic_fallback_case(root, "uppercase-reader", pdg, oracle,
                                uppercase_source)
        automatic_fallback_case(root, "uppercase-writer", pdg, oracle, floor,
                                output_suffix=".LAS")

        existing = root / "existing.las"
        existing.write_bytes(b"sentinel")
        existing_pipeline = write_skewness_pipeline(
            root / "existing.json", floor, existing)
        candidate = run([str(pdg), "pipeline", str(existing_pipeline)])
        candidate_state = path_state(existing)
        existing.write_bytes(b"sentinel")
        reference = run([str(oracle), "pipeline", str(existing_pipeline)])
        assert_matching_result("existing", candidate, reference)
        assert candidate_state == path_state(existing)
        existing.write_bytes(b"sentinel")
        required = run([str(pdg), "pipeline", str(existing_pipeline)],
                       automatic_environment())
        assert required.returncode == 124, required
        assert path_state(existing) == ("file", b"sentinel")
        assert_no_direct_output_temp(existing)

        alias = root / "alias.las"
        alias.write_bytes(floor.read_bytes())
        alias_before = alias.read_bytes()
        alias_pipeline = write_skewness_pipeline(
            root / "alias.json", alias, alias)
        candidate = run([str(pdg), "pipeline", str(alias_pipeline)])
        candidate_state = path_state(alias)
        alias.write_bytes(alias_before)
        reference = run([str(oracle), "pipeline", str(alias_pipeline)])
        assert_matching_result("alias", candidate, reference)
        assert candidate_state == path_state(alias)
        alias.write_bytes(alias_before)
        required = run([str(pdg), "pipeline", str(alias_pipeline)],
                       automatic_environment())
        assert required.returncode == 124, required
        assert path_state(alias) == ("file", alias_before)
        assert_no_direct_output_temp(alias)

        symlink_target = root / "symlink-target.las"
        symlink_output = root / "symlink-output.las"

        def reset_symlink():
            symlink_output.unlink(missing_ok=True)
            symlink_target.unlink(missing_ok=True)
            symlink_output.symlink_to(symlink_target)

        reset_symlink()
        symlink_pipeline = write_skewness_pipeline(
            root / "symlink.json", floor, symlink_output)
        candidate = run([str(pdg), "pipeline", str(symlink_pipeline)])
        candidate_state = (path_state(symlink_output),
                           path_state(symlink_target))
        reset_symlink()
        reference = run([str(oracle), "pipeline", str(symlink_pipeline)])
        assert_matching_result("symlink", candidate, reference)
        assert candidate_state == (path_state(symlink_output),
                                   path_state(symlink_target))
        reset_symlink()
        required = run([str(pdg), "pipeline", str(symlink_pipeline)],
                       automatic_environment())
        assert required.returncode == 124, required
        assert path_state(symlink_output) == (
            "symlink", str(symlink_target))
        assert path_state(symlink_target) == ("absent",)
        assert_no_direct_output_temp(symlink_output)
    return 0


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--automatic":
        return automatic_main(pathlib.Path(sys.argv[2]).resolve(),
                              pathlib.Path(sys.argv[3]).resolve())
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: resident_skewness_direct_gpu_test.py PDG PDAL INPUT\n"
            "   or: resident_skewness_direct_gpu_test.py --automatic PDG PDAL")
    pdg = pathlib.Path(sys.argv[1]).resolve()
    oracle = pathlib.Path(sys.argv[2]).resolve()
    source = pathlib.Path(sys.argv[3]).resolve()
    if not source.is_file():
        raise AssertionError("INPUT must be a readable LAS-like source")

    gpu = run(["nvidia-smi", "--query-gpu=memory.free",
               "--format=csv,noheader,nounits"])
    if gpu.returncode != 0 or not gpu.stdout.strip():
        print("skewness direct gate skipped without visible CUDA device",
              file=sys.stderr)
        return SKIP

    with tempfile.TemporaryDirectory(
            prefix="pdg-resident-skewness-direct-") as temporary:
        root = pathlib.Path(temporary)
        exact_source = comparator_unique_source(root, oracle)
        duplicate_z_source = duplicate_source(root, oracle)

        exact_case(root, str(pdg), str(oracle), exact_source)

        disabled = exact_environment()
        disabled["PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
        disabled_result = failing_case(
            root, "disabled", pdg, oracle, exact_source, disabled)
        assert "not used" in disabled_result.stderr, disabled_result.stderr

        proof_environment = exact_environment()
        proof_environment["PDG_TEST_DIRECT_SKEWNESS_PROOF_FAILURE"] = "1"
        proof_failure = failing_case(
            root, "proof-failure", pdg, oracle, exact_source,
            proof_environment)
        assert "direct skewness resident execution proof failed" in (
            proof_failure.stderr), proof_failure.stderr

        duplicate_result = failing_case(
            root, "duplicate-z", pdg, oracle, duplicate_z_source,
            exact_environment())
        assert "required exact CUDA hybrid skewnessbalancing path was not used" in (
            duplicate_result.stderr), duplicate_result.stderr

        unsupported_result = failing_case(
            root, "unsupported-option", pdg, oracle, exact_source,
            exact_environment(), options={"ground_class": 3})
        assert ("required direct skewness composition envelope" in
                unsupported_result.stderr or
                "required direct skewness composition plan" in
                unsupported_result.stderr or
                "required direct skewness composition path" in
                unsupported_result.stderr), unsupported_result.stderr

        _, unsupported_format, unsupported_stride, _ = las_layout(source)
        assert (unsupported_format, unsupported_stride) != (7, 36), (
            unsupported_format, unsupported_stride)
        failing_case(root, "unsupported-layout", pdg, oracle, source,
                     exact_environment())

        nonfinite_source = root / "nonfinite-z-scale.las"
        nonfinite_bytes = bytearray(exact_source.read_bytes())
        struct.pack_into("<d", nonfinite_bytes, 147, float("nan"))
        nonfinite_source.write_bytes(nonfinite_bytes)
        failing_case(root, "nonfinite-z", pdg, oracle, nonfinite_source,
                     exact_environment())

        source_before = exact_source.read_bytes()
        alias_pipeline = write_skewness_pipeline(
            root / "source-output-alias.json", exact_source, exact_source)
        alias = run([str(pdg), "resident", str(alias_pipeline)],
                    exact_environment())
        assert alias.returncode != 0, alias
        assert exact_source.read_bytes() == source_before
        assert_no_direct_output_temp(exact_source)

        existing_output = root / "existing-output.las"
        existing_output.write_bytes(b"sentinel-existing-output")
        existing_output_case(root, "existing-output", pdg, exact_source,
                             existing_output)

        symlink_target = root / "symlink-target.las"
        symlink_target.write_bytes(b"sentinel-symlink-target")
        symlink_output = root / "symlink-output.las"
        symlink_output.symlink_to(symlink_target)
        existing_output_case(root, "symlink-output", pdg, exact_source,
                             symlink_output)
        assert symlink_output.is_symlink()
        assert symlink_target.read_bytes() == b"sentinel-symlink-target"

        generic_direct = os.environ.copy()
        generic_direct["PDG_EXPERIMENTAL_DIRECT_RESIDENT_LAS_OUTPUT"] = "1"
        fallback_case(root, "generic-direct", pdg, oracle, exact_source,
                      generic_direct)

        fallback = os.environ.copy()
        fallback["PDG_EXPERIMENTAL_DIRECT_SKEWNESS_COMPOSITION"] = "1"
        fallback["PDG_TEST_DIRECT_SKEWNESS_PREFLIGHT_FAILURE"] = "1"
        fallback_case(root, "preflight-fallback", pdg, oracle, exact_source,
                      fallback)

        rejected_preflight = exact_environment()
        rejected_preflight[
            "PDG_TEST_DIRECT_SKEWNESS_PREFLIGHT_FAILURE"] = "1"
        preflight_result = failing_case(
            root, "preflight-rejected", pdg, oracle, exact_source,
            rejected_preflight)
        assert "not used" in preflight_result.stderr, preflight_result.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
