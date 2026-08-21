#!/usr/bin/env python3
"""Exact public-process gate for automatic direct HAG-Delaunay count-three."""

import hashlib
import json
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
FLOOR = 500_001
CAP = 16_000_002
PEAK_BYTES_PER_POINT = 184
CAP_PLUS_ONE_SHA256 = (
    "e28429948ceb8e8dccacb959cefeab2d6df4691252a443b356f11511ca597785")


def run(command, environment=None):
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
    )


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
    with path.open("rb") as stream:
        header = stream.read(375)
    assert header[:4] == b"LASF" and len(header) == 375, path
    point_offset = struct.unpack_from("<I", header, 96)[0]
    raw_format = header[104]
    stride = struct.unpack_from("<H", header, 105)[0]
    count = (struct.unpack_from("<Q", header, 247)[0]
             if header[25] >= 4 else struct.unpack_from("<I", header, 107)[0])
    compressed = bool(raw_format & 0x80)
    if not compressed:
        assert point_offset + count * stride == path.stat().st_size, path
    return point_offset, raw_format & 0x3f, stride, count, compressed, header


def mem_available_bytes():
    for line in pathlib.Path("/proc/meminfo").read_text(
            encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("MemAvailable missing from /proc/meminfo")


def automatic_environment():
    environment = os.environ.copy()
    environment["PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT"] = "1"
    return environment


def write_pipeline(path, source, destination, count=3,
                   producer_options=None, writer_options=None,
                   object_root=True):
    producer = {"type": "filters.hag_delaunay", "count": count}
    producer.update(producer_options or {})
    writer = {"type": "writers.las", "filename": str(destination),
              "extra_dims": "all"}
    writer.update(writer_options or {})
    stages = [
        {"type": "readers.las", "filename": str(source)},
        producer,
        writer,
    ]
    document = {"pipeline": stages} if object_root else stages
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def prefix_source(root, oracle, source, count):
    _, point_format, stride, _, compressed, header = las_layout(source)
    assert (point_format, stride, compressed) == (7, 40, False)
    scales = struct.unpack_from("<ddd", header, 131)
    offsets = struct.unpack_from("<ddd", header, 155)
    output = root / f"prefix-{count}.las"
    pipeline = root / f"prefix-{count}.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.head", "count": count},
        {"type": "writers.las", "filename": str(output),
         "minor_version": 4, "dataformat_id": 7, "extra_dims": "all",
         "scale_x": scales[0], "scale_y": scales[1],
         "scale_z": scales[2], "offset_x": offsets[0],
         "offset_y": offsets[1], "offset_z": offsets[2]},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    assert las_layout(output)[1:5] == (7, 40, count, False)
    return output


def format_without_extra(root, oracle, source):
    output = root / "format7-no-extra.las"
    pipeline = root / "format7-no-extra.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "writers.las", "filename": str(output),
         "minor_version": 4, "dataformat_id": 7},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    assert las_layout(output)[1:5] == (7, 36, las_layout(source)[3], False)
    return output


def incompatible_extra_descriptor(root, source):
    point_offset, point_format, stride, count, compressed, _ = (
        las_layout(source))
    assert (point_format, stride, compressed) == (7, 40, False)
    data = bytearray(source.read_bytes())
    descriptor = b"OffsetTime\x00"
    descriptor_offset = data[:point_offset].find(descriptor)
    assert descriptor_offset >= 0
    assert data[:point_offset].find(
        descriptor, descriptor_offset + 1) == -1
    data[descriptor_offset:descriptor_offset + len(descriptor)] = (
        b"OtherTimeX\x00")
    output = root / "incompatible-extra-descriptor.las"
    output.write_bytes(data)
    assert las_layout(output)[1:5] == (7, 40, count, False)
    return output


def pathological_source(root, source, name):
    point_offset, point_format, stride, count, compressed, _ = (
        las_layout(source))
    assert (point_format, stride, count, compressed) == (
        7, 40, FLOOR, False)
    data = bytearray(source.read_bytes())
    if name == "no-ground":
        for point in range(count):
            record = point_offset + point * stride
            data[record + 16] = 1
    elif name == "tie":
        base = -1_000_000_000
        rows = (
            (base, 0, 1),
            (base + 100, 0, 2),
            (base, 100, 2),
            (base - 100, 0, 2),
            (base, -100, 2),
        )
        for point, (x, y, classification) in enumerate(rows):
            record = point_offset + point * stride
            struct.pack_into("<ii", data, record, x, y)
            data[record + 16] = classification
    elif name == "incomplete":
        base = -1_000_000_000
        rows = (
            (base, 0, 1),
            (base + 500_100, 0, 2),
            (base + 500_200, 0, 2),
            (base + 500_300, 0, 2),
        )
        for point, (x, y, classification) in enumerate(rows):
            record = point_offset + point * stride
            struct.pack_into("<ii", data, record, x, y)
            data[record + 16] = classification
    else:
        raise AssertionError(name)
    output = root / f"{name}-source.las"
    output.write_bytes(data)
    assert las_layout(output)[1:5] == (7, 40, FLOOR, False)
    return output


def cap_plus_one_source(root, source):
    point_offset, point_format, stride, count, compressed, _ = (
        las_layout(source))
    assert (point_offset, point_format, stride, count, compressed) == (
        621, 7, 40, CAP, False)
    output = root / "cap-plus-one-16000003.las"
    shutil.copyfile(source, output)
    with source.open("rb") as stream:
        header = bytearray(stream.read(point_offset))
        record = bytearray(stream.read(stride))
    scale_z = struct.unpack_from("<d", header, 147)[0]
    offset_z = struct.unpack_from("<d", header, 171)[0]
    max_z = struct.unpack_from("<d", header, 211)[0]
    assert math.isfinite(scale_z) and scale_z > 0.0
    max_raw_z = round((max_z - offset_z) / scale_z)
    struct.pack_into("<i", record, 8, max_raw_z + 1)
    struct.pack_into("<d", header, 211,
                     offset_z + (max_raw_z + 1) * scale_z)
    struct.pack_into("<Q", header, 247, count + 1)
    return_number = record[14] & 0x0f
    assert 1 <= return_number <= 15
    return_offset = 255 + (return_number - 1) * 8
    return_count = struct.unpack_from("<Q", header, return_offset)[0]
    struct.pack_into("<Q", header, return_offset, return_count + 1)
    with output.open("r+b") as stream:
        stream.write(header)
        stream.seek(0, os.SEEK_END)
        stream.write(record)
    assert las_layout(output)[1:5] == (7, 40, CAP + 1, False)
    return output


def automatic_case(root, name, pdg, oracle, source, environment=None,
                   include_public=True):
    required_output = root / f"{name}-required.las"
    oracle_output = root / f"{name}-oracle.las"
    required_pipeline = write_pipeline(
        root / f"{name}-required.json", source, required_output)
    oracle_pipeline = write_pipeline(
        root / f"{name}-oracle.json", source, oracle_output)
    required = run([str(pdg), "pipeline", str(required_pipeline)],
                   environment or automatic_environment())
    reference = run([str(oracle), "pipeline", str(oracle_pipeline)])
    assert required.returncode == reference.returncode == 0, (
        name, required, reference)
    assert required.stdout == reference.stdout, name
    assert required.stderr == reference.stderr, (name, required.stderr)
    assert_identical(required_output, oracle_output)
    assert las_layout(required_output)[1:4] == (
        7, 48, las_layout(source)[3])

    if include_public:
        public_output = root / f"{name}-public.las"
        public_pipeline = write_pipeline(
            root / f"{name}-public.json", source, public_output)
        public = run([str(pdg), "pipeline", str(public_pipeline)])
        assert public.returncode == reference.returncode, (name, public)
        assert public.stdout == reference.stdout, name
        assert public.stderr == reference.stderr, name
        assert_identical(public_output, oracle_output)


def automatic_stats_case(root, pdg, source):
    output = root / "automatic-stats.las"
    stats = root / "automatic-stats.json"
    pipeline = write_pipeline(
        root / "automatic-stats-pipeline.json", source, output)
    result = run(
        [str(pdg), "resident", str(pipeline), "--stats", str(stats)],
        automatic_environment())
    assert result.returncode == 0, result
    report = json.loads(stats.read_text(encoding="utf-8"))
    point_count = las_layout(source)[3]
    placement = report["placement"]
    assert placement["available"] is True, placement
    assert placement["choice"] == "device", placement
    assert placement["selected_region_count"] == 1, placement
    assert placement["input_record_bytes"] == 40, placement
    assert placement["output_record_bytes"] == 48, placement
    assert placement["predicted"]["peak_device_bytes"] == (
        PEAK_BYTES_PER_POINT * point_count), placement
    assert placement["predicted"]["configured_device_lane_count"] == 1, (
        placement)
    execution = report["execution"]
    assert execution["selected_device_calibration_matches_executor"] is True, (
        execution)
    assert execution["boundary_accounting_matches_prediction"] is True, (
        execution)
    assert execution["selected_regions"] == [0], execution
    assert execution["selected_stage_ids"] == [1], execution
    assert execution["direct_las_output"] is True, execution
    assert execution["direct_extra_double_output"] is True, execution


def fallback_case(root, name, pdg, oracle, source, default_environment=None,
                  count=3, producer_options=None, writer_options=None,
                  object_root=True, output_suffix=".las",
                  include_required=True):
    output = root / f"{name}-shared{output_suffix}"
    pipeline = write_pipeline(
        root / f"{name}-shared.json", source, output, count,
        producer_options, writer_options, object_root)
    candidate = run([str(pdg), "pipeline", str(pipeline)],
                    default_environment)
    candidate_exists = output.exists()
    candidate_copy = root / f"{name}-candidate-copy{output_suffix}"
    if candidate_exists:
        output.replace(candidate_copy)
    reference = run([str(oracle), "pipeline", str(pipeline)])
    assert candidate.returncode == reference.returncode, (
        name, candidate, reference)
    assert candidate.stdout == reference.stdout, name
    assert candidate.stderr == reference.stderr, (name, candidate.stderr)
    assert candidate_exists == output.exists(), name
    if candidate_exists:
        assert_identical(candidate_copy, output)

    if not include_required:
        return

    required_output = root / f"{name}-required{output_suffix}"
    required_pipeline = write_pipeline(
        root / f"{name}-required.json", source, required_output, count,
        producer_options, writer_options, object_root)
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


def path_state(path):
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if not path.exists():
        return ("absent",)
    if path.is_file():
        return ("file", path.read_bytes())
    return ("other", path.stat().st_mode)


def publication_refusals(root, pdg, oracle, source):
    existing = root / "existing.las"
    existing.write_bytes(b"sentinel")
    existing_pipeline = write_pipeline(
        root / "existing.json", source, existing)
    candidate = run([str(pdg), "pipeline", str(existing_pipeline)])
    candidate_state = path_state(existing)
    existing.write_bytes(b"sentinel")
    reference = run([str(oracle), "pipeline", str(existing_pipeline)])
    assert (candidate.returncode, candidate.stdout, candidate.stderr) == (
        reference.returncode, reference.stdout, reference.stderr)
    assert candidate_state == path_state(existing)
    existing.write_bytes(b"sentinel")
    required = run([str(pdg), "pipeline", str(existing_pipeline)],
                   automatic_environment())
    assert required.returncode == 124 and path_state(existing) == (
        "file", b"sentinel")

    alias = root / "alias.las"
    shutil.copyfile(source, alias)
    alias_before = sha256(alias)
    alias_pipeline = write_pipeline(root / "alias.json", alias, alias)
    candidate = run([str(pdg), "pipeline", str(alias_pipeline)])
    candidate_state = path_state(alias)
    shutil.copyfile(source, alias)
    reference = run([str(oracle), "pipeline", str(alias_pipeline)])
    assert (candidate.returncode, candidate.stdout, candidate.stderr) == (
        reference.returncode, reference.stdout, reference.stderr)
    assert candidate_state == path_state(alias)
    shutil.copyfile(source, alias)
    required = run([str(pdg), "pipeline", str(alias_pipeline)],
                   automatic_environment())
    assert required.returncode == 124 and sha256(alias) == alias_before

    target = root / "symlink-target.las"
    output = root / "symlink-output.las"

    def reset_symlink():
        output.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        output.symlink_to(target)

    reset_symlink()
    pipeline = write_pipeline(root / "symlink.json", source, output)
    candidate = run([str(pdg), "pipeline", str(pipeline)])
    candidate_state = (path_state(output), path_state(target))
    reset_symlink()
    reference = run([str(oracle), "pipeline", str(pipeline)])
    assert (candidate.returncode, candidate.stdout, candidate.stderr) == (
        reference.returncode, reference.stdout, reference.stderr)
    assert candidate_state == (path_state(output), path_state(target))
    reset_symlink()
    required = run([str(pdg), "pipeline", str(pipeline)],
                   automatic_environment())
    assert required.returncode == 124
    assert path_state(output) == ("symlink", str(target))
    assert path_state(target) == ("absent",)


def main():
    if len(sys.argv) != 4 or sys.argv[1] != "--automatic":
        raise SystemExit(
            "usage: resident_hag_delaunay_direct_gpu_test.py --automatic PDG PDAL")
    pdg = pathlib.Path(sys.argv[2]).resolve()
    oracle = pathlib.Path(sys.argv[3]).resolve()
    configured = os.environ.get("PDG_HAG_DELAUNAY_COUNT3_LAS_FILE")
    expected_hash = os.environ.get("PDG_HAG_DELAUNAY_COUNT3_LAS_SHA256")
    cap_configured = os.environ.get("PDG_HAG_DELAUNAY_COUNT3_CAP_LAS_FILE")
    cap_expected_hash = os.environ.get("PDG_HAG_DELAUNAY_COUNT3_CAP_LAS_SHA256")
    if not all((configured, expected_hash, cap_configured, cap_expected_hash)):
        print("set the PDG_HAG_DELAUNAY_COUNT3[_CAP]_LAS_FILE/SHA256 variables",
              file=sys.stderr)
        return SKIP
    source = pathlib.Path(configured).resolve()
    cap_source = pathlib.Path(cap_configured).resolve()
    assert source.is_file() and source.suffix == ".las"
    assert cap_source.is_file() and cap_source.suffix == ".las"
    assert sha256(source) == expected_hash.lower()
    assert sha256(cap_source) == cap_expected_hash.lower()
    assert las_layout(source)[1:5] == (7, 40, 1_000_002, False)
    assert las_layout(cap_source)[1:5] == (7, 40, CAP, False)
    if mem_available_bytes() < 8 * GIB:
        print("HAG-Delaunay automatic gate skipped below 8 GiB MemAvailable",
              file=sys.stderr)
        return SKIP
    gpu = run(["nvidia-smi", "--query-gpu=memory.free",
               "--format=csv,noheader,nounits"])
    if gpu.returncode != 0 or int(gpu.stdout.splitlines()[0]) < 4096:
        print("HAG-Delaunay automatic gate skipped without 4 GiB free VRAM",
              file=sys.stderr)
        return SKIP

    with tempfile.TemporaryDirectory(
            prefix="pdg-hag-delaunay-count3-automatic-") as temporary:
        root = pathlib.Path(temporary)
        marginal = prefix_source(root, oracle, source, 400_002)
        below = prefix_source(root, oracle, source, 450_000)
        just_below = prefix_source(root, oracle, source, 500_000)
        floor = prefix_source(root, oracle, source, FLOOR)
        fallback_case(root, "marginal", pdg, oracle, marginal)
        fallback_case(root, "below", pdg, oracle, below)
        fallback_case(root, "just-below", pdg, oracle, just_below)
        automatic_case(root, "floor", pdg, oracle, floor)
        automatic_stats_case(root, pdg, floor)
        automatic_case(root, "main", pdg, oracle, source)
        automatic_case(root, "cap", pdg, oracle, cap_source,
                       include_public=False)
        cap_plus_one = cap_plus_one_source(root, cap_source)
        assert sha256(cap_plus_one) == CAP_PLUS_ONE_SHA256
        fallback_case(root, "cap-plus-one", pdg, oracle, cap_plus_one)

        exact_budget = automatic_environment()
        exact_budget["PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES"] = str(
            PEAK_BYTES_PER_POINT * FLOOR)
        automatic_case(root, "exact-budget", pdg, oracle, floor,
                       exact_budget, include_public=False)
        below_budget = os.environ.copy()
        below_budget["PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES"] = str(
            PEAK_BYTES_PER_POINT * FLOOR - 1)
        fallback_case(root, "below-budget", pdg, oracle, floor,
                      below_budget)

        for name, variable in (
                ("disabled-source", "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"),
                ("preflight", "PDG_TEST_AUTOMATIC_RESIDENT_PREFLIGHT_FAILURE"),
                ("proof", "PDG_TEST_AUTOMATIC_HAG_DELAUNAY_PROOF_FAILURE"),
                ("device-decline",
                 "PDG_TEST_AUTOMATIC_HAG_DELAUNAY_DEVICE_DECLINE")):
            environment = os.environ.copy()
            environment[variable] = "1"
            fallback_case(root, name, pdg, oracle, floor, environment)

        no_extra = format_without_extra(root, oracle, floor)
        fallback_case(root, "no-source-extra", pdg, oracle, no_extra)
        wrong_descriptor = incompatible_extra_descriptor(root, floor)
        fallback_case(root, "wrong-extra-descriptor", pdg, oracle,
                      wrong_descriptor)
        fallback_case(root, "count-four", pdg, oracle, floor, count=4)
        fallback_case(root, "stage-option", pdg, oracle, floor,
                      producer_options={"class": 9})
        fallback_case(root, "explicit-allow-true", pdg, oracle, floor,
                      producer_options={"allow_extrapolation": True})
        fallback_case(root, "explicit-allow-false", pdg, oracle, floor,
                      producer_options={"allow_extrapolation": False})
        fallback_case(root, "writer-option", pdg, oracle, floor,
                      writer_options={"forward": "all"})
        fallback_case(root, "array-root", pdg, oracle, floor,
                      object_root=False)

        uppercase = root / "uppercase-source.LAS"
        shutil.copyfile(floor, uppercase)
        fallback_case(root, "uppercase-reader", pdg, oracle, uppercase)
        fallback_case(root, "uppercase-writer", pdg, oracle, floor,
                      output_suffix=".LAS")

        no_ground = pathological_source(root, floor, "no-ground")
        # The pinned 2.10.0 oracle itself terminates with SIGSEGV after its
        # exact no-ground diagnostic. Public compatibility must preserve that
        # status and stderr; a required-route sentinel cannot supersede it.
        fallback_case(root, "no-ground", pdg, oracle, no_ground,
                      include_required=False)
        tie = pathological_source(root, floor, "tie")
        fallback_case(root, "tie-repair", pdg, oracle, tie)
        incomplete = pathological_source(root, floor, "incomplete")
        incomplete_environment = automatic_environment()
        incomplete_environment["PDG_FORCE_UNIFORM_GRID"] = "1"
        incomplete_environment["PDG_KNN_DEVICE_SHELL_BUDGET"] = "1"
        automatic_case(root, "incomplete-repair", pdg, oracle, incomplete,
                       incomplete_environment, include_public=False)
        publication_refusals(root, pdg, oracle, floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
