#!/usr/bin/env python3
"""Versioned semantic comparator for nondeterministic COPC containers.

This comparator is deliberately opt-in.  It does not replace PDG's default
byte-exact compatibility contract.  A conformance manifest may select it only
for a COPC artifact whose pinned oracle is itself byte-nondeterministic.

The comparison uses the pinned PDAL executable as the decoder for both sides,
then checks:

* semantic LAS header fields and non-container VLR/EVLR payloads;
* COPC info values and structural hierarchy invariants;
* a canonical multiset digest of every uncompressed point record; and
* a canonical point digest for a fixed, exact center-window query.

Physical hierarchy-node assignment and the result of a coarse-resolution
preview are retained as diagnostics.  They do not decide semantic equality:
PDAL can vary both when writing the same point multiset concurrently.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import pathlib
import struct
import subprocess
import tempfile
from typing import BinaryIO, Iterable


SCHEMA = "pdg-copc-canonical-v1"
COPC_INFO_USER = "copc"
COPC_INFO_RECORD = 1
COPC_HIERARCHY_RECORD = 1000
LASZIP_USER = "laszip encoded"


def sha256(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", "replace").rstrip()


def _header(path: pathlib.Path) -> tuple[bytes, dict[str, object]]:
    with path.open("rb") as stream:
        header = stream.read(375)
    if len(header) < 227 or header[:4] != b"LASF" or header[24] != 1:
        raise ValueError(f"not a supported LAS-family file: {path}")
    minor = header[25]
    if minor > 4:
        raise ValueError(f"unsupported LAS version 1.{minor}: {path}")
    count = (struct.unpack_from("<Q", header, 247)[0] if minor >= 4
             else struct.unpack_from("<I", header, 107)[0])
    returns_offset = 255 if minor >= 4 else 111
    returns_count = 15 if minor >= 4 else 5
    returns_format = "<" + ("Q" if minor >= 4 else "I") * returns_count
    values: dict[str, object] = {
        "version": f"1.{minor}",
        "file_source_id": struct.unpack_from("<H", header, 4)[0],
        "global_encoding": struct.unpack_from("<H", header, 6)[0],
        "project_id_hex": header[8:24].hex(),
        "system_identifier": _text(header[26:58]),
        "generating_software": _text(header[58:90]),
        "creation_day": struct.unpack_from("<H", header, 90)[0],
        "creation_year": struct.unpack_from("<H", header, 92)[0],
        "point_format": header[104] & 0x3F,
        "record_bytes": struct.unpack_from("<H", header, 105)[0],
        "point_count": count,
        "points_by_return": list(struct.unpack_from(
            returns_format, header, returns_offset)),
        "scale": list(struct.unpack_from("<3d", header, 131)),
        "offset": list(struct.unpack_from("<3d", header, 155)),
        "bounds": {
            "maxx": struct.unpack_from("<d", header, 179)[0],
            "minx": struct.unpack_from("<d", header, 187)[0],
            "maxy": struct.unpack_from("<d", header, 195)[0],
            "miny": struct.unpack_from("<d", header, 203)[0],
            "maxz": struct.unpack_from("<d", header, 211)[0],
            "minz": struct.unpack_from("<d", header, 219)[0],
        },
    }
    return header, values


def _records(path: pathlib.Path) -> list[dict[str, object]]:
    header, _ = _header(path)
    header_bytes = struct.unpack_from("<H", header, 94)[0]
    vlr_count = struct.unpack_from("<I", header, 100)[0]
    evlr_offset = struct.unpack_from("<Q", header, 235)[0] if header[25] >= 4 else 0
    evlr_count = struct.unpack_from("<I", header, 243)[0] if header[25] >= 4 else 0
    result: list[dict[str, object]] = []
    with path.open("rb") as stream:
        stream.seek(header_bytes)
        for _ in range(vlr_count):
            prefix = stream.read(54)
            if len(prefix) != 54:
                raise ValueError(f"truncated VLR header in {path}")
            length = struct.unpack_from("<H", prefix, 20)[0]
            payload = stream.read(length)
            if len(payload) != length:
                raise ValueError(f"truncated VLR payload in {path}")
            result.append({
                "kind": "vlr", "user_id": _text(prefix[2:18]),
                "record_id": struct.unpack_from("<H", prefix, 18)[0],
                "description": _text(prefix[22:54]), "payload": payload,
            })
        if evlr_offset and evlr_count:
            stream.seek(evlr_offset)
            for _ in range(evlr_count):
                prefix = stream.read(60)
                if len(prefix) != 60:
                    raise ValueError(f"truncated EVLR header in {path}")
                length = struct.unpack_from("<Q", prefix, 20)[0]
                payload = stream.read(length)
                if len(payload) != length:
                    raise ValueError(f"truncated EVLR payload in {path}")
                result.append({
                    "kind": "evlr", "user_id": _text(prefix[2:18]),
                    "record_id": struct.unpack_from("<H", prefix, 18)[0],
                    "description": _text(prefix[28:60]), "payload": payload,
                })
    return result


def _copc_info(records: Iterable[dict[str, object]]) -> dict[str, object]:
    for record in records:
        if (record["user_id"] == COPC_INFO_USER and
                record["record_id"] == COPC_INFO_RECORD):
            payload = record["payload"]
            assert isinstance(payload, bytes)
            if len(payload) < 72:
                raise ValueError("COPC info VLR is shorter than 72 bytes")
            return {
                "center": list(struct.unpack_from("<3d", payload, 0)),
                "halfsize": struct.unpack_from("<d", payload, 24)[0],
                "spacing": struct.unpack_from("<d", payload, 32)[0],
                "root_hierarchy_offset": struct.unpack_from("<Q", payload, 40)[0],
                "root_hierarchy_bytes": struct.unpack_from("<Q", payload, 48)[0],
                "gps_time_minimum": struct.unpack_from("<d", payload, 56)[0],
                "gps_time_maximum": struct.unpack_from("<d", payload, 64)[0],
            }
    raise ValueError("missing COPC info VLR")


def _semantic_records(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for record in records:
        user = str(record["user_id"])
        record_id = int(record["record_id"])
        if user == LASZIP_USER or (user == COPC_INFO_USER and
                                  record_id in (COPC_INFO_RECORD,
                                                COPC_HIERARCHY_RECORD)):
            continue
        payload = record["payload"]
        assert isinstance(payload, bytes)
        result.append({
            "kind": record["kind"], "user_id": user,
            "record_id": record_id, "description": record["description"],
            "payload_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        })
    return result


def _hierarchy(path: pathlib.Path, info: dict[str, object]) -> dict[str, object]:
    size = path.stat().st_size
    pending = [(int(info["root_hierarchy_offset"]),
                int(info["root_hierarchy_bytes"]))]
    seen_pages: set[tuple[int, int]] = set()
    nodes: list[tuple[int, int, int, int, int]] = []
    data_ranges: list[tuple[int, int]] = []
    errors: list[str] = []
    with path.open("rb") as stream:
        while pending:
            offset, length = pending.pop()
            page = (offset, length)
            if page in seen_pages:
                continue
            seen_pages.add(page)
            if offset < 0 or length <= 0 or length % 32 or offset + length > size:
                errors.append(f"invalid hierarchy page {offset}+{length}/{size}")
                continue
            stream.seek(offset)
            payload = stream.read(length)
            if len(payload) != length:
                errors.append(f"truncated hierarchy page at {offset}")
                continue
            for position in range(0, length, 32):
                level, x, y, z, entry_offset, entry_bytes, point_count = \
                    struct.unpack_from("<iiiiQii", payload, position)
                if level < 0:
                    continue
                key = (level, x, y, z)
                if point_count == -1:
                    pending.append((entry_offset, entry_bytes))
                elif point_count >= 0:
                    nodes.append((*key, point_count))
                    if entry_bytes:
                        if entry_offset < 0 or entry_bytes < 0 or \
                                entry_offset + entry_bytes > size:
                            errors.append(
                                f"invalid point range {entry_offset}+{entry_bytes}/{size}")
                        else:
                            data_ranges.append((entry_offset, entry_bytes))
                else:
                    errors.append(f"invalid point count {point_count} for {key}")
    duplicate_keys = len(nodes) - len({node[:4] for node in nodes})
    if duplicate_keys:
        errors.append(f"duplicate hierarchy keys: {duplicate_keys}")
    nodes.sort()
    return {
        "valid": not errors,
        "errors": errors,
        "page_count": len(seen_pages),
        "node_count": len(nodes),
        "point_count_sum": sum(node[4] for node in nodes),
        "nodes_sha256": hashlib.sha256(json.dumps(nodes,
            separators=(",", ":")).encode()).hexdigest(),
        "point_ranges": len(data_ranges),
    }


def _run_pdal(pdal: pathlib.Path, pipeline: dict[str, object],
              destination: pathlib.Path, environment: dict[str, str]) -> None:
    pipeline_path = destination.with_suffix(".pipeline.json")
    pipeline_path.write_text(json.dumps(pipeline, sort_keys=True))
    completed = subprocess.run(
        [str(pdal), "pipeline", str(pipeline_path)], env=environment,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=3600)
    if completed.returncode or not destination.is_file():
        raise ValueError(
            f"pinned PDAL could not canonicalize COPC (exit "
            f"{completed.returncode}): " +
            completed.stderr.decode("utf-8", "replace")[:500])


def _decode(pdal: pathlib.Path, source: pathlib.Path,
            destination: pathlib.Path, environment: dict[str, str],
            *, bounds: str | None = None,
            resolution: float | None = None) -> None:
    reader: dict[str, object] = {"type": "readers.copc", "filename": str(source)}
    if bounds is not None:
        reader["bounds"] = bounds
    if resolution is not None:
        reader["resolution"] = resolution
    pipeline = {"pipeline": [reader, {
        "type": "writers.las", "filename": str(destination),
        "compression": "false", "forward": "all", "extra_dims": "all",
    }]}
    _run_pdal(pdal, pipeline, destination, environment)


def _las_layout(path: pathlib.Path) -> tuple[int, int, int]:
    header, semantic = _header(path)
    return (struct.unpack_from("<I", header, 96)[0],
            int(semantic["record_bytes"]), int(semantic["point_count"]))


def _write_sorted_runs(path: pathlib.Path, directory: pathlib.Path,
                       records_per_run: int = 250_000) -> tuple[list[pathlib.Path], int, int]:
    offset, record_bytes, count = _las_layout(path)
    runs: list[pathlib.Path] = []
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = count
        while remaining:
            take = min(remaining, records_per_run)
            block = stream.read(take * record_bytes)
            if len(block) != take * record_bytes:
                raise ValueError(f"truncated canonical LAS point records: {path}")
            records = [block[index:index + record_bytes]
                       for index in range(0, len(block), record_bytes)]
            records.sort()
            run = directory / f"run-{len(runs):06d}.bin"
            with run.open("wb") as output:
                output.writelines(records)
            runs.append(run)
            remaining -= take
    return runs, record_bytes, count


def _read_record(stream: BinaryIO, record_bytes: int) -> bytes | None:
    value = stream.read(record_bytes)
    if not value:
        return None
    if len(value) != record_bytes:
        raise ValueError("truncated sorted point-record run")
    return value


def _multiset_digest(path: pathlib.Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="pdg-copc-sort-") as temp_text:
        temp = pathlib.Path(temp_text)
        runs, record_bytes, count = _write_sorted_runs(path, temp)
        digest = hashlib.sha256()
        digest.update(struct.pack("<HQ", record_bytes, count))
        streams = [run.open("rb") for run in runs]
        try:
            iterators = [iter(lambda stream=stream: _read_record(stream, record_bytes), None)
                         for stream in streams]
            for record in heapq.merge(*iterators):
                digest.update(record)
        finally:
            for stream in streams:
                stream.close()
        return {"record_bytes": record_bytes, "point_count": count,
                "sorted_records_sha256": digest.hexdigest()}


def _query_specs(header: dict[str, object], info: dict[str, object]) -> list[dict[str, object]]:
    bounds = header["bounds"]
    assert isinstance(bounds, dict)
    def inner(low: float, high: float) -> tuple[float, float]:
        return (low + (high - low) * 0.25, low + (high - low) * 0.75)
    x = inner(float(bounds["minx"]), float(bounds["maxx"]))
    y = inner(float(bounds["miny"]), float(bounds["maxy"]))
    z = (float(bounds["minz"]), float(bounds["maxz"]))
    rendered = f"([{x[0]:.17g},{x[1]:.17g}],[{y[0]:.17g},{y[1]:.17g}],[{z[0]:.17g},{z[1]:.17g}])"
    return [
        {"id": "center-window", "bounds": rendered},
        {"id": "coarse-resolution", "resolution": float(info["spacing"]) * 4.0},
    ]


def describe(pdal: pathlib.Path, path: pathlib.Path,
             work: pathlib.Path, environment: dict[str, str],
             max_input_bytes: int, max_decoded_bytes: int) -> dict[str, object]:
    work.mkdir(parents=True, exist_ok=True)
    if path.stat().st_size > max_input_bytes:
        raise ValueError(
            f"COPC input bytes {path.stat().st_size} exceed {max_input_bytes}")
    records = _records(path)
    info = _copc_info(records)
    _, header = _header(path)
    decoded_estimate = (int(header["point_count"]) *
                        int(header["record_bytes"]) + 16_777_216)
    if decoded_estimate > max_decoded_bytes:
        raise ValueError(
            f"canonical decode estimate {decoded_estimate} exceeds "
            f"{max_decoded_bytes}")
    decoded = work / "full.las"
    _decode(pdal, path, decoded, environment)
    if decoded.stat().st_size > max_decoded_bytes:
        raise ValueError("canonical full decode exceeded its byte budget")
    queries: dict[str, object] = {}
    for query in _query_specs(header, info):
        output = work / f"{query['id']}.las"
        _decode(pdal, path, output, environment,
                bounds=query.get("bounds"), resolution=query.get("resolution"))
        if output.stat().st_size > max_decoded_bytes:
            raise ValueError("canonical query decode exceeded its byte budget")
        queries[str(query["id"])] = {
            "request": query, "records": _multiset_digest(output)}
    public_info = {key: value for key, value in info.items()
                   if key not in ("root_hierarchy_offset", "root_hierarchy_bytes")}
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
        "header": header, "semantic_records": _semantic_records(records),
        "copc_info": public_info, "hierarchy": _hierarchy(path, info),
        "canonical_records": _multiset_digest(decoded), "queries": queries,
    }


def compare(pdal: pathlib.Path, expected: pathlib.Path,
            actual: pathlib.Path, environment: dict[str, str] | None = None,
            max_input_bytes: int = 64 << 30,
            max_decoded_bytes: int = 128 << 30) -> dict[str, object]:
    source = os.environ if environment is None else environment
    env = {key: value for key, value in source.items()
           if not key.startswith(("PDG_", "PDAL_TEST_", "LD_"))}
    env.update({"LC_ALL": "C", "TZ": "UTC"})
    with tempfile.TemporaryDirectory(prefix="pdg-copc-canonical-") as temp_text:
        temp = pathlib.Path(temp_text)
        left = describe(pdal, expected, temp / "oracle", env,
                        max_input_bytes, max_decoded_bytes)
        right = describe(pdal, actual, temp / "candidate", env,
                         max_input_bytes, max_decoded_bytes)
    differences: list[dict[str, object]] = []
    for field in ("header", "semantic_records", "copc_info",
                  "canonical_records"):
        if left[field] != right[field]:
            differences.append({"field": field, "expected": left[field],
                                "actual": right[field]})
    for role, value in (("oracle", left), ("candidate", right)):
        hierarchy = value["hierarchy"]
        assert isinstance(hierarchy, dict)
        if not hierarchy["valid"]:
            differences.append({"field": f"{role}.hierarchy.valid",
                                "errors": hierarchy["errors"]})
    left_hierarchy = left["hierarchy"]
    right_hierarchy = right["hierarchy"]
    assert isinstance(left_hierarchy, dict) and isinstance(right_hierarchy, dict)
    for field in ("page_count", "node_count", "point_count_sum",
                  "point_ranges"):
        if left_hierarchy[field] != right_hierarchy[field]:
            differences.append({"field": f"hierarchy.{field}",
                                "expected": left_hierarchy[field],
                                "actual": right_hierarchy[field]})
    left_queries = left["queries"]
    right_queries = right["queries"]
    assert isinstance(left_queries, dict) and isinstance(right_queries, dict)
    if left_queries["center-window"] != right_queries["center-window"]:
        differences.append({
            "field": "queries.center-window",
            "expected": left_queries["center-window"],
            "actual": right_queries["center-window"],
        })

    physical_differences: list[dict[str, object]] = []
    if left_hierarchy["nodes_sha256"] != right_hierarchy["nodes_sha256"]:
        physical_differences.append({
            "field": "hierarchy.nodes_sha256",
            "expected": left_hierarchy["nodes_sha256"],
            "actual": right_hierarchy["nodes_sha256"],
        })
    if left_queries["coarse-resolution"] != right_queries["coarse-resolution"]:
        physical_differences.append({
            "field": "queries.coarse-resolution",
            "expected": left_queries["coarse-resolution"],
            "actual": right_queries["coarse-resolution"],
        })
    return {"schema": SCHEMA, "semantic_equal": not differences,
            "oracle": left, "candidate": right, "differences": differences,
            "physical_differences": physical_differences,
            "policy": {
                "semantic": [
                    "header", "semantic_records", "copc_info",
                    "hierarchy.valid", "hierarchy.page_count",
                    "hierarchy.node_count", "hierarchy.point_count_sum",
                    "hierarchy.point_ranges", "canonical_records",
                    "queries.center-window",
                ],
                "diagnostic_only": [
                    "hierarchy.nodes_sha256", "queries.coarse-resolution",
                ],
            }}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pdal", required=True, type=pathlib.Path)
    parser.add_argument("--expected", required=True, type=pathlib.Path)
    parser.add_argument("--actual", required=True, type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--max-input-bytes", type=int, default=64 << 30)
    parser.add_argument("--max-decoded-bytes", type=int, default=128 << 30)
    args = parser.parse_args()
    for path in (args.pdal, args.expected, args.actual):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    try:
        if args.max_input_bytes <= 0 or args.max_decoded_bytes <= 0:
            raise ValueError("COPC comparator byte limits must be positive")
        report = compare(args.pdal.resolve(), args.expected.resolve(),
                         args.actual.resolve(),
                         max_input_bytes=args.max_input_bytes,
                         max_decoded_bytes=args.max_decoded_bytes)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report = {"schema": SCHEMA, "semantic_equal": False,
                  "differences": [{"field": "comparator_error",
                                   "actual": str(error)}]}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if report.get("semantic_equal") else 1


if __name__ == "__main__":
    raise SystemExit(main())
