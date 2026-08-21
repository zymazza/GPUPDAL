#!/usr/bin/env python3
"""Physical exact gate for automatic direct approximate-coplanar output."""

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
    source_data, source_offset, _, source_stride, source_count = las_layout(
        source)
    output_data, output_offset, _, output_stride, output_count = las_layout(
        output)
    assert source_count == output_count
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


def pipeline_document(source, destination, knn=8, dimensions=None,
                      thresh1=None, thresh2=None, reader_option=False,
                      extra_stage=False, writer_option=False):
    reader = {"type": "readers.las", "filename": str(source)}
    if reader_option:
        reader["count"] = 1_000_000
    approximate = {
        "type": "filters.approximatecoplanar",
        "knn": knn,
    }
    if thresh1 is not None:
        approximate["thresh1"] = thresh1
    if thresh2 is not None:
        approximate["thresh2"] = thresh2
    stages = [reader, approximate]
    if extra_stage:
        stages.append({"type": "filters.head", "count": 1_000_000})
    stages.append({
        "type": "filters.ferry",
        "dimensions": dimensions or "Coplanar=>UserData",
    })
    writer = {"type": "writers.las", "filename": str(destination)}
    if writer_option:
        writer["minor_version"] = 4
    stages.append(writer)
    return {"pipeline": stages}


def write_pipeline(path, source, destination, **options):
    path.write_text(json.dumps(pipeline_document(
        source, destination, **options)), encoding="utf-8")
    return path


def automatic_environment():
    environment = os.environ.copy()
    environment[
        "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_RESIDENT"] = "1"
    return environment


def prefix_format7(root, oracle, source, count):
    prefix = root / f"prefix-{count}.las"
    pipeline = root / f"prefix-{count}.json"
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


def convert_format(root, oracle, source, point_format):
    converted = root / f"format-{point_format}.las"
    pipeline = root / f"format-{point_format}.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "writers.las", "filename": str(converted),
         "minor_version": 4, "dataformat_id": point_format},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    assert las_layout(converted)[2] == point_format
    return converted


def compress_source(root, oracle, source):
    compressed = root / "compressed-source.laz"
    pipeline = root / "compressed-source.json"
    pipeline.write_text(json.dumps({"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "writers.las", "filename": str(compressed),
         "minor_version": 4, "dataformat_id": 7},
    ]}), encoding="utf-8")
    result = run([str(oracle), "pipeline", str(pipeline)])
    assert result.returncode == 0, result.stderr
    assert compressed.is_file()
    return compressed


def rejected_case(root, name, pdg, source, environment=None,
                  destination_suffix=".las", **options):
    output = root / f"{name}{destination_suffix}"
    pipeline = write_pipeline(root / f"{name}.json", source, output,
                              **options)
    result = run([str(pdg), "pipeline", str(pipeline)],
                 environment or automatic_environment())
    assert result.returncode != 0, (name, result)
    assert not output.exists(), (name, output)
    return result


def fallback_case(root, name, pdg, oracle, source, environment=None,
                  destination_suffix=".las", **options):
    candidate_output = root / f"{name}-candidate{destination_suffix}"
    oracle_output = root / f"{name}-oracle{destination_suffix}"
    candidate_pipeline = write_pipeline(
        root / f"{name}-candidate.json", source, candidate_output, **options)
    oracle_pipeline = write_pipeline(
        root / f"{name}-oracle.json", source, oracle_output, **options)
    candidate = run([str(pdg), "pipeline", str(candidate_pipeline)],
                    environment)
    reference = run([str(oracle), "pipeline", str(oracle_pipeline)],
                    environment)
    assert candidate.returncode == reference.returncode, (
        name, candidate.returncode, candidate.stderr,
        reference.returncode, reference.stderr)
    assert candidate.stdout == reference.stdout, name
    assert candidate.stderr == reference.stderr, name
    if candidate.returncode == 0:
        assert_identical(candidate_output, oracle_output)
    else:
        assert not candidate_output.exists(), (name, candidate_output)
        assert not oracle_output.exists(), (name, oracle_output)


def automatic_exact_case(root, pdg, oracle, source):
    candidate_output = root / "candidate-required.las"
    option_free_output = root / "candidate-option-free.las"
    oracle_output = root / "oracle.las"
    candidate_pipeline = write_pipeline(
        root / "candidate-required.json", source, candidate_output)
    option_free_pipeline = write_pipeline(
        root / "candidate-option-free.json", source, option_free_output)
    oracle_pipeline = write_pipeline(
        root / "oracle.json", source, oracle_output)

    candidate = run([str(pdg), "pipeline", str(candidate_pipeline)],
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
    assert user_data_changed(source, candidate_output), (
        "automatic gate must positively exercise Coplanar publication")
    return candidate_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdg", type=pathlib.Path)
    parser.add_argument("oracle", type=pathlib.Path)
    args = parser.parse_args()

    configured = os.environ.get("PDG_COPLANAR_LAS_FILE")
    expected_hash = os.environ.get("PDG_COPLANAR_LAS_SHA256")
    if not configured or not expected_hash:
        print("set PDG_COPLANAR_LAS_FILE and PDG_COPLANAR_LAS_SHA256 for "
              "the physical gate", file=sys.stderr)
        return SKIP
    source = pathlib.Path(configured).resolve()
    if not source.is_file() or source.suffix.lower() != ".las":
        raise AssertionError("PDG_COPLANAR_LAS_FILE must name a LAS file")
    if sha256(source) != expected_hash.lower():
        raise AssertionError("physical coplanar source hash mismatch")
    _, _, point_format, point_stride, points = las_layout(source)
    if (point_format, point_stride, points) != (7, 36, 1_000_000):
        raise AssertionError(
            "physical coplanar source must be 1M format-7/36-byte LAS")
    if mem_available_bytes() < 8 * GIB:
        print("coplanar gate skipped below 8 GiB MemAvailable",
              file=sys.stderr)
        return SKIP
    gpu = run(["nvidia-smi", "--query-gpu=memory.free",
               "--format=csv,noheader,nounits"])
    if gpu.returncode != 0 or int(gpu.stdout.splitlines()[0]) < 2048:
        print("coplanar gate skipped without 2 GiB free VRAM",
              file=sys.stderr)
        return SKIP

    with tempfile.TemporaryDirectory(
            prefix="pdg-coplanar-automatic-") as temporary:
        root = pathlib.Path(temporary)
        exact_output = automatic_exact_case(
            root, args.pdg, args.oracle, source)

        below = prefix_format7(root, args.oracle, source, 50_000)
        below_result = rejected_case(root, "below-floor", args.pdg, below)
        assert below_result.returncode == 124, below_result
        fallback_case(root, "below-floor-fallback", args.pdg, args.oracle,
                      below)

        floor = prefix_format7(root, args.oracle, source, 250_000)
        cases = (
            ("changed-knn", {"knn": 9}),
            ("explicit-default-thresh1", {"thresh1": 25.0}),
            ("changed-thresh1", {"thresh1": 24.0}),
            ("explicit-default-thresh2", {"thresh2": 6.0}),
            ("changed-thresh2", {"thresh2": 5.0}),
            ("wrong-ferry",
             {"dimensions": "Coplanar=>PointSourceId"}),
            ("extra-stage", {"extra_stage": True}),
            ("reader-option", {"reader_option": True}),
            ("writer-option", {"writer_option": True}),
        )
        for name, options in cases:
            rejected_case(root, name, args.pdg, floor, **options)
            fallback_case(root, f"{name}-fallback", args.pdg, args.oracle,
                          floor, **options)

        wrong_format = convert_format(root, args.oracle, floor, 6)
        rejected_case(root, "wrong-format", args.pdg, wrong_format)
        fallback_case(root, "wrong-format-fallback", args.pdg, args.oracle,
                      wrong_format)
        compressed = compress_source(root, args.oracle, floor)
        rejected_case(root, "compressed-source", args.pdg, compressed)
        fallback_case(root, "compressed-source-fallback", args.pdg,
                      args.oracle, compressed)
        rejected_case(root, "compressed-output", args.pdg, floor,
                      destination_suffix=".laz")
        fallback_case(root, "compressed-output-fallback", args.pdg,
                      args.oracle, floor, destination_suffix=".laz")

        masked_environment = automatic_environment()
        masked_environment["CUDA_VISIBLE_DEVICES"] = ""
        masked = rejected_case(root, "masked-device", args.pdg, floor,
                               masked_environment)
        assert masked.returncode == 124, masked
        option_free_masked = os.environ.copy()
        option_free_masked["CUDA_VISIBLE_DEVICES"] = ""
        fallback_case(root, "masked-device-fallback", args.pdg, args.oracle,
                      floor, option_free_masked)

        preflight_environment = automatic_environment()
        preflight_environment[
            "PDG_TEST_AUTOMATIC_RESIDENT_PREFLIGHT_FAILURE"] = "1"
        preflight = rejected_case(root, "preflight-failure", args.pdg,
                                  floor, preflight_environment)
        assert preflight.returncode == 124, preflight
        option_free_preflight = os.environ.copy()
        option_free_preflight[
            "PDG_TEST_AUTOMATIC_RESIDENT_PREFLIGHT_FAILURE"] = "1"
        fallback_case(root, "preflight-fallback", args.pdg, args.oracle,
                      floor, option_free_preflight)

        proof_environment = automatic_environment()
        proof_environment[
            "PDG_TEST_AUTOMATIC_COPLANAR_PROOF_FAILURE"] = "1"
        proof = rejected_case(root, "proof-failure", args.pdg, source,
                              proof_environment)
        assert "automatic approximate-coplanar resident execution proof " \
            "failed" in proof.stderr, proof.stderr

        print(json.dumps({
            "points": points,
            "point_format": point_format,
            "point_stride": point_stride,
            "output_sha256": sha256(exact_output),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
