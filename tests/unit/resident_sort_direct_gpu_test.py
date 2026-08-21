#!/usr/bin/env python3
"""Exact process gate for the direct-LAS Z-sort composition endpoint."""

import json
import hashlib
import math
import os
import pathlib
import struct
import sys
import tempfile

import resident_skewness_direct_gpu_test as common


SKIP = 77
GIB = 1024 * 1024 * 1024


def exact_environment():
    environment = os.environ.copy()
    environment["PDG_REQUIRE_DIRECT_SORT_COMPOSITION"] = "1"
    return environment


def automatic_environment():
    environment = os.environ.copy()
    environment["PDG_REQUIRE_AUTOMATIC_SORT_RESIDENT"] = "1"
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


def write_pipeline(path, source, destination, **options):
    sort = {
        "type": "filters.sort",
        "dimension": "Z",
        "order": "ASC",
        "algorithm": "NORMAL",
    }
    sort.update(options)
    path.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        sort,
        {"type": "writers.las", "filename": str(destination),
         "extra_dims": "all"},
    ]}), encoding="utf-8")
    return path


def failing_case(root, name, pdg, source, environment, **options):
    output = root / f"{name}.las"
    pipeline = write_pipeline(root / f"{name}.json", source, output,
                              **options)
    result = common.run([str(pdg), "resident", str(pipeline)], environment)
    assert result.returncode != 0, (name, result)
    assert not output.exists(), (name, output)
    common.assert_no_direct_output_temp(output)
    return result


def exact_case(root, pdg, oracle, source, name="required", environment=None):
    candidate_output = root / f"{name}-candidate.las"
    oracle_output = root / f"{name}-oracle.las"
    stats = root / f"{name}-candidate-stats.json"
    candidate_pipeline = write_pipeline(
        root / f"{name}-candidate.json", source, candidate_output)
    oracle_pipeline = write_pipeline(root / f"{name}-oracle.json", source,
                                     oracle_output)
    candidate = common.run(
        [str(pdg), "resident", str(candidate_pipeline), "--stats", str(stats)],
        environment or exact_environment())
    reference = common.run([str(oracle), "pipeline", str(oracle_pipeline)])
    assert candidate.returncode == reference.returncode == 0, (
        candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout
    assert candidate.stderr == reference.stderr
    assert candidate_output.read_bytes() == oracle_output.read_bytes()

    _, point_format, stride, point_count = common.las_layout(candidate_output)
    assert (point_format, stride, point_count) == (7, 36, 8)
    execution = json.loads(stats.read_text(encoding="utf-8"))["execution"]
    assert execution["executor"] == (
        "planner_resident_global_order_direct_las"), execution
    assert execution["direct_las_output"] is True, execution
    assert execution["direct_las_resident_source"] is True, execution
    assert execution["direct_permuted_output"] is True, execution
    assert execution["direct_permuted_sort_output"] is True, execution
    assert execution["direct_permuted_classification_output"] is False, (
        execution)
    assert execution["direct_las_record_summary"] is False, execution
    assert execution["direct_las_host_xyz_mirror"] is False, execution
    assert execution["terminal_spill_elided"] is True, execution
    assert execution["schedule"]["pipeline_class"] == (
        "whole_view_global_order"), execution
    assert execution["selected_regions"] == [0], execution
    assert execution["selected_stage_ids"] == [1], execution
    assert execution["index_builds"] == {
        "predicted": 0, "observed": 0, "matches_prediction": True,
    }, execution["index_builds"]
    expected = point_count * 8
    assert [event["bytes"] for event in execution["events"]
            if event["kind"] == "host_to_device"] == [expected], execution
    assert [event["bytes"] for event in execution["events"]
            if event["kind"] == "device_to_host"] == [expected], execution


def empty_source(root, oracle):
    csv = root / "empty.csv"
    csv.write_text("X,Y,Z,Classification\n", encoding="utf-8")
    source = root / "empty.las"
    pipeline = common.write_source_pipeline(
        root / "empty-source.json", csv, source)
    result = common.run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    assert common.las_layout(source)[1:] == (7, 36, 0)
    return source


def fallback_case(root, name, pdg, oracle, source, environment):
    candidate_output = root / f"{name}-candidate.las"
    oracle_output = root / f"{name}-oracle.las"
    stats = root / f"{name}-stats.json"
    candidate_pipeline = write_pipeline(
        root / f"{name}-candidate.json", source, candidate_output)
    oracle_pipeline = write_pipeline(
        root / f"{name}-oracle.json", source, oracle_output)
    candidate = common.run(
        [str(pdg), "resident", str(candidate_pipeline), "--stats", str(stats)],
        environment)
    reference = common.run([str(oracle), "pipeline", str(oracle_pipeline)])
    assert candidate.returncode == reference.returncode == 0, (
        name, candidate.stderr, reference.stderr)
    assert candidate.stdout == reference.stdout, name
    assert candidate.stderr == reference.stderr, (
        name, candidate.stderr, reference.stderr)
    assert candidate_output.read_bytes() == oracle_output.read_bytes(), name
    execution = json.loads(stats.read_text(encoding="utf-8")).get("execution")
    assert execution is None or execution.get(
        "direct_permuted_sort_output") is not True, (name, execution)
    common.assert_no_direct_output_temp(candidate_output)


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
    result = common.run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    assert common.las_layout(output)[1:] == (7, 36, count)
    output_header = output.read_bytes()[:227]
    assert struct.unpack_from("<ddd", output_header, 131) == scales
    assert struct.unpack_from("<ddd", output_header, 155) == offsets
    return output


def duplicate_z_source(root, source):
    output = root / "duplicate-z-600k.las"
    data = bytearray(source.read_bytes())
    point_offset, _, stride, count = common.las_layout(source)
    assert count == 600_000
    data[point_offset + stride + 8:point_offset + stride + 12] = (
        data[point_offset + 8:point_offset + 12])
    output.write_bytes(data)
    return output


def nonfinite_z_source(root, source):
    output = root / "nonfinite-z-scale-600k.las"
    data = bytearray(source.read_bytes())
    struct.pack_into("<d", data, 147, float("nan"))
    output.write_bytes(data)
    return output


def automatic_case(root, name, pdg, oracle, source, environment=None):
    candidate_output = root / f"{name}-candidate.las"
    option_free_output = root / f"{name}-option-free.las"
    oracle_output = root / f"{name}-oracle.las"
    candidate_pipeline = write_pipeline(
        root / f"{name}-candidate.json", source, candidate_output)
    option_free_pipeline = write_pipeline(
        root / f"{name}-option-free.json", source, option_free_output)
    oracle_pipeline = write_pipeline(
        root / f"{name}-oracle.json", source, oracle_output)
    candidate = common.run(
        [str(pdg), "pipeline", str(candidate_pipeline)],
        environment or automatic_environment())
    option_free = common.run(
        [str(pdg), "pipeline", str(option_free_pipeline)])
    reference = common.run([str(oracle), "pipeline", str(oracle_pipeline)])
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
                            default_environment=None):
    shared_output = root / f"{name}-shared.las"
    shared_pipeline = write_pipeline(
        root / f"{name}-shared.json", source, shared_output)
    candidate = common.run([str(pdg), "pipeline", str(shared_pipeline)],
                           default_environment)
    candidate_exists = shared_output.exists()
    candidate_bytes = shared_output.read_bytes() if candidate_exists else None
    shared_output.unlink(missing_ok=True)
    reference = common.run([str(oracle), "pipeline", str(shared_pipeline)])
    assert candidate.returncode == reference.returncode, (
        name, candidate, reference)
    assert candidate.stdout == reference.stdout, name
    assert candidate.stderr == reference.stderr, (
        name, candidate.stderr, reference.stderr)
    assert candidate_exists == shared_output.exists(), name
    if candidate_exists:
        assert candidate_bytes == shared_output.read_bytes(), name
    shared_output.unlink(missing_ok=True)

    required_output = root / f"{name}-required.las"
    required_pipeline = write_pipeline(
        root / f"{name}-required.json", source, required_output)
    required_environment = automatic_environment()
    if default_environment:
        required_environment.update({
            name: value for name, value in default_environment.items()
            if name.startswith("PDG_TEST_")
        })
    required = common.run(
        [str(pdg), "pipeline", str(required_pipeline)],
        required_environment)
    assert required.returncode == 124, (name, required)
    assert not required_output.exists(), (name, required_output)
    common.assert_no_direct_output_temp(required_output)


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
    configured = os.environ.get("PDG_SORT_LAS_FILE")
    expected_hash = os.environ.get("PDG_SORT_LAS_SHA256")
    if not configured or not expected_hash:
        print("set PDG_SORT_LAS_FILE and PDG_SORT_LAS_SHA256 for the "
              "automatic sort gate", file=sys.stderr)
        return SKIP
    source = pathlib.Path(configured).resolve()
    if not source.is_file() or source.suffix.lower() != ".las":
        raise AssertionError("PDG_SORT_LAS_FILE must name a LAS file")
    if sha256(source) != expected_hash.lower():
        raise AssertionError("physical sort source hash mismatch")
    point_offset, point_format, stride, count = common.las_layout(source)
    if (point_format, stride, count) != (7, 36, 1_000_000):
        raise AssertionError(
            "physical sort source must be uncompressed format-7/36-byte "
            "LAS with exactly 1M points")
    source_bytes = source.read_bytes()
    scale_z = struct.unpack_from("<d", source_bytes, 147)[0]
    if not math.isfinite(scale_z) or scale_z == 0.0:
        raise AssertionError("physical sort source has invalid Z scale")
    raw_z = {
        struct.unpack_from("<i", source_bytes, point_offset + row * stride + 8)[0]
        for row in range(count)
    }
    if len(raw_z) != count:
        raise AssertionError("physical sort source Z keys are not unique")
    if mem_available_bytes() < 8 * GIB:
        print("sort gate skipped below 8 GiB MemAvailable", file=sys.stderr)
        return SKIP
    gpu = common.run(["nvidia-smi", "--query-gpu=memory.free",
                      "--format=csv,noheader,nounits"])
    if gpu.returncode != 0 or int(gpu.stdout.splitlines()[0]) < 2048:
        print("sort gate skipped without 2 GiB free VRAM", file=sys.stderr)
        return SKIP

    with tempfile.TemporaryDirectory(
            prefix="pdg-sort-automatic-") as temporary:
        root = pathlib.Path(temporary)
        below = prefix_preserving_layout(root, oracle, source, 550_000)
        below_output = root / "below-output.las"
        below_pipeline = write_pipeline(root / "below.json", below,
                                        below_output)
        below_result = common.run(
            [str(pdg), "pipeline", str(below_pipeline)],
            automatic_environment())
        assert below_result.returncode == 124, below_result
        assert not below_output.exists(), below_output

        floor = prefix_preserving_layout(root, oracle, source, 600_000)
        automatic_case(root, "floor", pdg, oracle, floor)
        automatic_case(root, "main", pdg, oracle, source)

        exact_budget = 64 * 600_000
        budget_environment = automatic_environment()
        budget_environment["PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES"] = str(
            exact_budget)
        automatic_case(root, "floor-exact-budget", pdg, oracle, floor,
                       budget_environment)
        below_budget_environment = os.environ.copy()
        below_budget_environment[
            "PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES"] = str(exact_budget - 1)
        automatic_fallback_case(
            root, "floor-below-budget", pdg, oracle, floor,
            below_budget_environment)

        for case, variable in (
                ("disabled-source", "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"),
                ("preflight", "PDG_TEST_DIRECT_SORT_PREFLIGHT_FAILURE"),
                ("proof", "PDG_TEST_DIRECT_SORT_PROOF_FAILURE")):
            output = root / f"{case}.las"
            pipeline = write_pipeline(root / f"{case}.json", source, output)
            environment = automatic_environment()
            environment[variable] = "1"
            result = common.run([str(pdg), "pipeline", str(pipeline)],
                                environment)
            assert result.returncode == 124, (case, result)
            assert not output.exists(), (case, output)
            common.assert_no_direct_output_temp(output)

        proof_fallback = os.environ.copy()
        proof_fallback["PDG_TEST_DIRECT_SORT_PROOF_FAILURE"] = "1"
        automatic_fallback_case(
            root, "proof-default-fallback", pdg, oracle, source,
            proof_fallback)

        duplicate = duplicate_z_source(root, floor)
        automatic_fallback_case(root, "duplicate-z", pdg, oracle, duplicate)
        nonfinite = nonfinite_z_source(root, floor)
        automatic_fallback_case(root, "nonfinite-z", pdg, oracle, nonfinite)

        existing = root / "automatic-existing.las"
        existing.write_bytes(b"sentinel")
        existing_pipeline = write_pipeline(
            root / "automatic-existing.json", floor, existing)
        candidate = common.run(
            [str(pdg), "pipeline", str(existing_pipeline)])
        candidate_state = path_state(existing)
        existing.write_bytes(b"sentinel")
        reference = common.run(
            [str(oracle), "pipeline", str(existing_pipeline)])
        assert_matching_result("automatic-existing", candidate, reference)
        assert candidate_state == path_state(existing)
        existing.write_bytes(b"sentinel")
        required = common.run(
            [str(pdg), "pipeline", str(existing_pipeline)],
            automatic_environment())
        assert required.returncode == 124, required
        assert path_state(existing) == ("file", b"sentinel")
        common.assert_no_direct_output_temp(existing)

        alias = root / "automatic-alias.las"
        alias.write_bytes(floor.read_bytes())
        alias_before = alias.read_bytes()
        alias_pipeline = write_pipeline(root / "automatic-alias.json", alias,
                                        alias)
        candidate = common.run([str(pdg), "pipeline", str(alias_pipeline)])
        candidate_state = path_state(alias)
        alias.write_bytes(alias_before)
        reference = common.run(
            [str(oracle), "pipeline", str(alias_pipeline)])
        assert_matching_result("automatic-alias", candidate, reference)
        assert candidate_state == path_state(alias)
        alias.write_bytes(alias_before)
        required = common.run(
            [str(pdg), "pipeline", str(alias_pipeline)],
            automatic_environment())
        assert required.returncode == 124, required
        assert path_state(alias) == ("file", alias_before)
        common.assert_no_direct_output_temp(alias)

        symlink_target = root / "automatic-symlink-target.las"
        symlink_output = root / "automatic-symlink-output.las"

        def reset_symlink():
            symlink_output.unlink(missing_ok=True)
            symlink_target.unlink(missing_ok=True)
            symlink_output.symlink_to(symlink_target)

        reset_symlink()
        symlink_pipeline = write_pipeline(
            root / "automatic-symlink.json", floor, symlink_output)
        candidate = common.run(
            [str(pdg), "pipeline", str(symlink_pipeline)])
        candidate_state = (path_state(symlink_output),
                           path_state(symlink_target))
        reset_symlink()
        reference = common.run(
            [str(oracle), "pipeline", str(symlink_pipeline)])
        assert_matching_result("automatic-symlink", candidate, reference)
        assert candidate_state == (path_state(symlink_output),
                                   path_state(symlink_target))
        reset_symlink()
        required = common.run(
            [str(pdg), "pipeline", str(symlink_pipeline)],
            automatic_environment())
        assert required.returncode == 124, required
        assert path_state(symlink_output) == (
            "symlink", str(symlink_target))
        assert path_state(symlink_target) == ("absent",)
        common.assert_no_direct_output_temp(symlink_output)

        for name, options in (
                ("descending", {"order": "DESC"}),
                ("stable", {"algorithm": "STABLE"}),
                ("x-key", {"dimension": "X"})):
            output = root / f"{name}.las"
            pipeline = write_pipeline(root / f"{name}.json", source, output,
                                      **options)
            result = common.run([str(pdg), "pipeline", str(pipeline)],
                                automatic_environment())
            assert result.returncode == 124, (name, result)
            assert not output.exists(), (name, output)
    return 0


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--automatic":
        return automatic_main(pathlib.Path(sys.argv[2]).resolve(),
                              pathlib.Path(sys.argv[3]).resolve())
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: resident_sort_direct_gpu_test.py PDG PDAL INPUT\n"
            "   or: resident_sort_direct_gpu_test.py --automatic PDG PDAL")
    pdg = pathlib.Path(sys.argv[1]).resolve()
    oracle = pathlib.Path(sys.argv[2]).resolve()
    unsupported_source = pathlib.Path(sys.argv[3]).resolve()
    if not unsupported_source.is_file():
        raise AssertionError("INPUT must be a readable LAS-like source")

    gpu = common.run(["nvidia-smi", "--query-gpu=memory.free",
                      "--format=csv,noheader,nounits"])
    if gpu.returncode != 0 or not gpu.stdout.strip():
        print("sort direct gate skipped without visible CUDA device",
              file=sys.stderr)
        return SKIP

    with tempfile.TemporaryDirectory(
            prefix="pdg-resident-sort-direct-") as temporary:
        root = pathlib.Path(temporary)
        source = common.comparator_unique_source(root, oracle)
        duplicate = common.duplicate_source(root, oracle)
        empty = empty_source(root, oracle)
        exact_case(root, pdg, oracle, source)
        optional = os.environ.copy()
        optional["PDG_EXPERIMENTAL_DIRECT_SORT_COMPOSITION"] = "1"
        exact_case(root, pdg, oracle, source, "optional", optional)

        fallback_case(root, "empty-host-fallback", pdg, oracle, empty,
                      os.environ.copy())
        empty_required = failing_case(
            root, "empty-required", pdg, empty, exact_environment())
        assert "required direct LAS resident source path was not used" in (
            empty_required.stderr), empty_required.stderr

        disabled = exact_environment()
        disabled["PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
        assert "not used" in failing_case(
            root, "disabled", pdg, source, disabled).stderr

        proof = exact_environment()
        proof["PDG_TEST_DIRECT_SORT_PROOF_FAILURE"] = "1"
        proof_result = failing_case(root, "proof", pdg, source, proof)
        assert "direct sort resident execution proof failed" in (
            proof_result.stderr), proof_result.stderr

        budget = exact_environment()
        budget["PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES"] = "1"
        budget_result = failing_case(root, "budget", pdg, source, budget)
        assert "not used" in budget_result.stderr, budget_result.stderr

        duplicate_result = failing_case(
            root, "duplicate-z", pdg, duplicate, exact_environment())
        assert "required exact CUDA hybrid ordering path was not used" in (
            duplicate_result.stderr), duplicate_result.stderr

        for name, options in (
                ("descending", {"order": "DESC"}),
                ("stable", {"algorithm": "STABLE"}),
                ("x-key", {"dimension": "X"})):
            result = failing_case(root, name, pdg, source,
                                  exact_environment(), **options)
            assert "required direct sort composition" in result.stderr, (
                name, result.stderr)

        default_output = root / "default-rejected.las"
        default_pipeline = write_pipeline(
            root / "default-rejected.json", source, default_output,
            order="DESC")
        default_result = common.run(
            [str(pdg), "pipeline", str(default_pipeline)],
            exact_environment())
        assert default_result.returncode == 124, default_result
        assert "required direct sort composition path was not used" in (
            default_result.stderr), default_result.stderr
        assert not default_output.exists()
        common.assert_no_direct_output_temp(default_output)

        _, fmt, stride, _ = common.las_layout(unsupported_source)
        assert (fmt, stride) != (7, 36), (fmt, stride)
        failing_case(root, "unsupported-layout", pdg, unsupported_source,
                     exact_environment())

        nonfinite = root / "nonfinite-z-scale.las"
        nonfinite_bytes = bytearray(source.read_bytes())
        struct.pack_into("<d", nonfinite_bytes, 147, float("nan"))
        nonfinite.write_bytes(nonfinite_bytes)
        failing_case(root, "nonfinite-z", pdg, nonfinite,
                     exact_environment())

        signed_zero = root / "signed-zero-z-scale.las"
        signed_zero_bytes = bytearray(source.read_bytes())
        struct.pack_into("<d", signed_zero_bytes, 147, -0.0)
        signed_zero.write_bytes(signed_zero_bytes)
        failing_case(root, "signed-zero-z", pdg, signed_zero,
                     exact_environment())

        before = source.read_bytes()
        alias = write_pipeline(root / "alias.json", source, source)
        alias_result = common.run([str(pdg), "resident", str(alias)],
                                  exact_environment())
        assert alias_result.returncode != 0, alias_result
        assert source.read_bytes() == before
        common.assert_no_direct_output_temp(source)

        existing = root / "existing.las"
        existing.write_bytes(b"sentinel")
        existing_result = common.run(
            [str(pdg), "resident", str(write_pipeline(
                root / "existing.json", source, existing))],
            exact_environment())
        assert existing_result.returncode != 0, existing_result
        assert existing.read_bytes() == b"sentinel"
        common.assert_no_direct_output_temp(existing)

        dangling_target = root / "dangling-target.las"
        symlink_output = root / "symlink-output.las"
        symlink_output.symlink_to(dangling_target)
        symlink_result = common.run(
            [str(pdg), "resident", str(write_pipeline(
                root / "symlink-output.json", source, symlink_output))],
            exact_environment())
        assert symlink_result.returncode != 0, symlink_result
        assert symlink_output.is_symlink()
        assert not dangling_target.exists()
        common.assert_no_direct_output_temp(symlink_output)

        generic = os.environ.copy()
        generic["PDG_EXPERIMENTAL_DIRECT_RESIDENT_LAS_OUTPUT"] = "1"
        fallback_case(root, "generic", pdg, oracle, source, generic)

        optional_preflight = os.environ.copy()
        optional_preflight[
            "PDG_EXPERIMENTAL_DIRECT_SORT_COMPOSITION"] = "1"
        optional_preflight["PDG_TEST_DIRECT_SORT_PREFLIGHT_FAILURE"] = "1"
        fallback_case(root, "preflight-fallback", pdg, oracle, source,
                      optional_preflight)

        required_preflight = exact_environment()
        required_preflight["PDG_TEST_DIRECT_SORT_PREFLIGHT_FAILURE"] = "1"
        assert "not used" in failing_case(
            root, "preflight-required", pdg, source,
            required_preflight).stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
