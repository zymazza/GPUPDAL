#!/usr/bin/env python3
"""Complete-process baseline for the D0208 reference pipelines.

The reference pipelines in `bench/pipelines/reference/` are end-to-end
workflows rather than single stages, so they need input and output extensions
that vary per workflow (`.laz`, `.copc.laz`, `.tif`, `.las`) and bounds that
are substituted from the fixture's real extent. `benchmark_translate.py`
materializes exactly one `input.las`/`output.las` pair and cannot express that,
so this is a sibling harness rather than a flag on that one.

It runs the pinned oracle and the candidate alternately with untimed warmups,
compares every produced artifact byte-for-byte, and writes an append-only
JSON report. Timing is complete-process wall clock: that is the metric D0208
made the release criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_elf(path: pathlib.Path) -> bool:
    """Return whether *path* is an ELF executable/library."""
    try:
        with path.open("rb") as stream:
            return stream.read(4) == b"\x7fELF"
    except OSError:
        return False


def resolved_pdal_library(executable: pathlib.Path,
                          environment: dict[str, str]) -> pathlib.Path:
    """Resolve libpdalcpp or reject incomplete benchmark provenance."""
    try:
        completed = subprocess.run(
            ["ldd", str(executable)], env=environment, capture_output=True,
            text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"could not resolve the PDAL library loaded by {executable}: "
            f"{error}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"ldd failed for {executable} with exit {completed.returncode}: "
            f"{completed.stderr.strip()[:400]}")
    for line in completed.stdout.splitlines():
        left, separator, right = line.partition("=>")
        if not separator or not left.strip().startswith("libpdalcpp.so."):
            continue
        rendered = right.strip().split(maxsplit=1)[0]
        library = pathlib.Path(rendered)
        if library.is_file():
            return library.resolve()
    raise RuntimeError(
        f"ldd did not resolve libpdalcpp for benchmark executable {executable}")


def binary_manifest(path: pathlib.Path, *, candidate: bool,
                    environment: dict[str, str]) -> dict:
    """Hash the public executable and runtime components that carry behavior."""
    result: dict[str, object] = {"path": str(path), "sha256": sha256(path)}
    components: list[dict[str, str]] = []
    seen: set[pathlib.Path] = set()
    dependency_carrier = path

    if candidate and is_elf(path):
        engine = path.with_name("pdg-engine")
        if not engine.is_file() or not is_elf(engine):
            raise RuntimeError(
                f"ELF candidate {path} has no ELF pdg-engine beside it")
        resolved = engine.resolve()
        seen.add(resolved)
        dependency_carrier = engine
        components.append({"role": "engine", "path": str(engine),
                           "sha256": sha256(engine)})

    # `gpupal` and `pdal` dynamically load the fork/oracle PDAL library. Resolve
    # the dependency with the exact loader environment used for the measured
    # process rather than assuming the adjacent build-tree library won.
    if is_elf(dependency_carrier):
        library = resolved_pdal_library(dependency_carrier, environment)
        if library in seen:
            raise RuntimeError(
                f"resolved PDAL library aliases another runtime component: "
                f"{library}")
        seen.add(library)
        components.append({"role": "loaded_pdal_shared_library",
                           "path": str(library), "sha256": sha256(library)})

    if components:
        result["runtime_components"] = components
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--oracle", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--pipeline", required=True, type=pathlib.Path,
                        help="a reference pipeline JSON")
    parser.add_argument("--fixture-laz", required=True, type=pathlib.Path)
    parser.add_argument("--fixture-copc", type=pathlib.Path)
    parser.add_argument(
        "--fixture", action="append", default=[], metavar="PLACEHOLDER=PATH",
        help="named input placeholder, for example input.orthophoto.tif=FILE",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--report", required=True, type=pathlib.Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--cache-state", choices=("cold", "warm"), default="warm",
        help="cold uses file-scoped POSIX_FADV_DONTNEED before every run",
    )
    parser.add_argument("--frozen-time-library", type=pathlib.Path)
    parser.add_argument("--freeze-epoch", type=int, default=1704067200)
    parser.add_argument(
        "--environment", action="append", default=[], metavar="NAME=VALUE",
        help="explicit environment value applied to both oracle and candidate",
    )
    parser.add_argument(
        "--candidate-env", action="append", default=[], metavar="NAME=VALUE",
        help="explicit candidate-only proof/tuning value (recorded in report)",
    )
    parser.add_argument("--repo-root", type=pathlib.Path,
                        default=pathlib.Path("."))
    parser.add_argument(
        "--contract", choices=("exact", "fast"), default="exact",
        help="exact (default) requires byte-identical artifacts, stdout and "
             "stderr; fast (D0261/D0271) requires identically ordered LAS/LAZ "
             "point records with identical count and coordinates, allows a "
             "bounded number of records to differ in other attributes "
             "(kNN tie order, D0271), requires byte-identical non-point "
             "artifacts, and only reports stream/header differences",
    )
    parser.add_argument(
        "--candidate-arg", action="append", default=[], metavar="ARG",
        help="argument inserted before the candidate's command (for example "
             "--fast); recorded in the report",
    )
    parser.add_argument(
        "--qualification-model",
        help="record physical LAS input/output facts for a placement-model "
             "qualification; requires the automatic normal/covariance route",
    )
    parser.add_argument(
        "--fast-max-differing-records-fraction", type=float, default=0.01,
        help="fast contract only: the largest fraction of LAS/LAZ records "
             "whose non-coordinate bytes may differ from the oracle "
             "(D0271 tie-order relaxation); default 0.01",
    )
    return parser.parse_args()


def fixture_overrides(values: list[str]) -> dict[str, pathlib.Path]:
    fixtures: dict[str, pathlib.Path] = {}
    for value in values:
        placeholder, separator, path_text = value.partition("=")
        if not separator or not placeholder.startswith("input.") or not path_text:
            raise SystemExit(
                f"invalid fixture mapping {value!r}; expected input.NAME=PATH")
        if placeholder in fixtures:
            raise SystemExit(f"duplicate fixture placeholder {placeholder!r}")
        path = pathlib.Path(path_text).resolve()
        if not path.is_file():
            raise SystemExit(f"fixture {placeholder!r} is not a file: {path}")
        fixtures[placeholder] = path
    return fixtures


def registered_fixtures(args: argparse.Namespace) -> dict[str, pathlib.Path]:
    fixtures = fixture_overrides(args.fixture)
    laz = args.fixture_laz.resolve()
    if not laz.is_file():
        raise SystemExit(f"--fixture-laz is not a file: {laz}")
    fixtures.setdefault("input.laz", laz)
    # Backward compatibility for old one-fixture probes.  Conversion runs set
    # input.las explicitly to the hash-pinned uncompressed fixture.
    fixtures.setdefault("input.las", laz)
    if args.fixture_copc:
        copc = args.fixture_copc.resolve()
        if not copc.is_file():
            raise SystemExit(f"--fixture-copc is not a file: {copc}")
        fixtures.setdefault("input.copc.laz", copc)
    return dict(sorted(fixtures.items()))


def environment_overrides(values: list[str], *, candidate_only: bool) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        name, separator, contents = value.partition("=")
        if not separator or not name:
            raise SystemExit(f"invalid environment override {value!r}; expected NAME=VALUE")
        if name == "PDAL_TEST_FROZEN_EPOCH":
            raise SystemExit(
                "PDAL_TEST_FROZEN_EPOCH is harness-owned; use "
                "--frozen-time-library and --freeze-epoch"
            )
        if name.startswith("PDG_INTERNAL_"):
            raise SystemExit(
                f"internal environment variable {name} may not be injected by the harness"
            )
        if candidate_only and not name.startswith("PDG_"):
            raise SystemExit(
                f"candidate-only environment variable {name} must use the PDG_ namespace"
            )
        overrides[name] = contents
    return dict(sorted(overrides.items()))


def candidate_placement_profile(candidate: Path,
                                environment: dict[str, str]) -> dict | None:
    """`gpupal calibrate --status` of the candidate, if it supports it."""
    try:
        completed = subprocess.run(
            [str(candidate), "calibrate", "--status"], env=environment,
            capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode not in (0, 2):
        return None
    result: dict = {}
    for line in completed.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in ("device_name", "compute_capability", "driver", "cuda",
                   "cpu_model", "logical_cpus", "embedded_profile",
                   "local_profile_path", "local_profile", "local_profile_id",
                   "local_profile_created_utc", "local_profile_models",
                   "shipped_profile", "generic_profile", "active_profile",
                   "active_profile_tier"):
            result[key] = value.strip()
    return result or None


def clean_environment(common: dict[str, str]) -> dict[str, str]:
    # Never let an unrecorded proof, selector, worker, backend, or diagnostic
    # switch influence a baseline. Explicit common/candidate overrides below
    # are the only route back in. The harness-owned clock input is injected
    # separately and cannot be supplied through --environment.
    # PDAL_TEST_* names are compiled-in test hooks (for example forced host
    # neighborhood worker counts, B0258); an ambient one must never reach a
    # measured process. The harness-owned clock input is re-added below.
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith("PDG_") and
        not name.startswith("PDAL_TEST_") and
        not name.startswith("LD_")
    }
    environment.update({"LC_ALL": "C", "TZ": "UTC"})
    environment.update(common)
    return environment


def fixture_extent(oracle: pathlib.Path, fixture: pathlib.Path,
                   environment: dict[str, str]) -> dict:
    result = subprocess.run([str(oracle), "info", "--summary", str(fixture)],
                            capture_output=True, text=True, check=True,
                            env=environment)
    summary = json.loads(result.stdout)["summary"]
    bounds = summary["bounds"]
    return {"minx": bounds["minx"], "maxx": bounds["maxx"],
            "miny": bounds["miny"], "maxy": bounds["maxy"],
            "points": summary["num_points"]}


def clip_multipolygon_wkt(extent: dict) -> str:
    """Build a source-relative polygon/hole/multipolygon in EPSG:4326."""
    width = extent["maxx"] - extent["minx"]
    height = extent["maxy"] - extent["miny"]

    def point(x_fraction: float, y_fraction: float) -> tuple[float, float]:
        return (extent["minx"] + width * x_fraction,
                extent["miny"] + height * y_fraction)

    rings = [
        [point(.10, .10), point(.60, .10), point(.60, .90),
         point(.10, .90), point(.10, .10)],
        [point(.25, .30), point(.40, .30), point(.40, .60),
         point(.25, .60), point(.25, .30)],
        [point(.70, .20), point(.90, .20), point(.90, .80),
         point(.70, .80), point(.70, .20)],
    ]
    flattened = [coordinate for ring in rings for coordinate in ring]
    completed = subprocess.run(
        # cs2cs honors EPSG:4326's latitude/longitude axis order by default,
        # while WKT point coordinates are the traditional X/Y
        # longitude/latitude order consumed by filters.crop.  Reverse the
        # formatted output so the materialized geometry covers the source
        # extent instead of producing an empty crop.
        ["cs2cs", "EPSG:28992", "EPSG:4326", "-f", "%.12f", "-s"],
        input="".join(f"{x:.12f} {y:.12f}\n" for x, y in flattened),
        capture_output=True, text=True, check=False)
    if completed.returncode:
        raise SystemExit(
            "cs2cs could not materialize the clipping geometry: " +
            completed.stderr[:400])
    transformed: list[tuple[float, float]] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2:
            raise SystemExit(f"unexpected cs2cs output line: {line!r}")
        transformed.append((float(fields[0]), float(fields[1])))
    if len(transformed) != len(flattened):
        raise SystemExit("cs2cs returned the wrong number of clipping vertices")
    cursor = 0
    rendered: list[str] = []
    for ring in rings:
        count = len(ring)
        coordinates = transformed[cursor:cursor + count]
        cursor += count
        rendered.append(",".join(f"{x:.12f} {y:.12f}" for x, y in coordinates))
    return f"MULTIPOLYGON((({rendered[0]}),({rendered[1]})),(({rendered[2]})))"


def output_placeholder(value: str) -> bool:
    return value.startswith(("output.", "output-", "output_"))


def materialize(pipeline: pathlib.Path, destination: pathlib.Path,
                fixtures: dict[str, pathlib.Path], extent: dict,
                output_directory: pathlib.Path) -> list[dict]:
    """Substitute named fixtures, parameters and output artifact patterns."""
    pipeline_text = pipeline.read_text(encoding="utf-8")
    document = json.loads(pipeline_text)
    width = extent["maxx"] - extent["minx"]
    height = extent["maxy"] - extent["miny"]
    replacements = {
        "REPLACE_MINX": repr(extent["minx"] + width * 0.25),
        "REPLACE_MAXX": repr(extent["minx"] + width * 0.75),
        "REPLACE_MINY": repr(extent["miny"] + height * 0.25),
        "REPLACE_MAXY": repr(extent["miny"] + height * 0.75),
        "REPLACE_FULL_MINX": repr(extent["minx"]),
        "REPLACE_FULL_MAXX": repr(extent["maxx"]),
        "REPLACE_FULL_MINY": repr(extent["miny"]),
        "REPLACE_FULL_MAXY": repr(extent["maxy"]),
        "REPLACE_TILE_ORIGIN_X": repr(math.floor(extent["minx"] / 256.0) * 256.0),
        "REPLACE_TILE_ORIGIN_Y": repr(math.floor(extent["miny"] / 256.0) * 256.0),
    }
    if "REPLACE_CLIP_MULTIPOLYGON_WKT" in pipeline_text:
        replacements["REPLACE_CLIP_MULTIPOLYGON_WKT"] = clip_multipolygon_wkt(extent)
    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict] = []

    def replace(value):
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if not isinstance(value, str):
            return value
        for token, substitute in replacements.items():
            value = value.replace(token, substitute)
        if value in fixtures:
            return str(fixtures[value])
        if value.startswith("input."):
            raise SystemExit(
                f"pipeline input placeholder {value!r} has no --fixture mapping")
        if output_placeholder(value):
            if pathlib.Path(value).name != value:
                raise SystemExit(f"output placeholder must be a filename: {value!r}")
            pattern = output_directory / value
            artifacts.append({"pattern": pattern, "logical_pattern": value})
            return str(pattern)
        return value

    document = replace(document)
    # A benchmark without a declared artifact cannot prove exactness.
    if not artifacts:
        raise SystemExit("reference pipeline declares no output artifact")
    unique: dict[str, dict] = {}
    for artifact in artifacts:
        key = str(artifact["pattern"])
        if key in unique:
            raise SystemExit(
                "duplicate output artifact pattern: " +
                artifact["logical_pattern"])
        unique[key] = artifact
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return list(unique.values())


def pipeline_input_paths(pipeline: pathlib.Path,
                         fixtures: dict[str, pathlib.Path]) -> list[pathlib.Path]:
    document = json.loads(pipeline.read_text(encoding="utf-8"))
    placeholders: set[str] = set()

    def visit(value) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, str) and value.startswith("input."):
            placeholders.add(value)

    visit(document)
    missing = sorted(placeholders - fixtures.keys())
    if missing:
        raise SystemExit("pipeline has unmapped input placeholders: " + ", ".join(missing))
    return sorted(set(fixtures[placeholder] for placeholder in placeholders))


def artifact_matches(specification: dict) -> list[pathlib.Path]:
    pattern = specification["pattern"]
    if "#" in pattern.name:
        return sorted(pattern.parent.glob(pattern.name.replace("#", "*")))
    return [pattern] if pattern.is_file() else []


def remove_artifacts(specifications: list[dict]) -> None:
    for specification in specifications:
        for artifact in artifact_matches(specification):
            artifact.unlink(missing_ok=True)


def evict_file_cache(paths: list[pathlib.Path]) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise SystemExit("cold-cache mode requires POSIX_FADV_DONTNEED")
    for path in paths:
        try:
            with path.open("rb") as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as error:
            raise SystemExit(f"could not evict {path} from the page cache: {error}") from error


def raster_metadata(path: pathlib.Path) -> dict:
    completed = subprocess.run(
        ["gdalinfo", "-json", "-checksum", str(path)],
        capture_output=True, text=True, check=False)
    if completed.returncode:
        raise SystemExit(
            f"gdalinfo could not inspect raster {path}: {completed.stderr[:400]}")
    try:
        metadata = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(f"gdalinfo returned invalid JSON for {path}: {error}") from error
    # These two fields contain the role-specific materialized path.  Every
    # raster property (driver, size, CRS, transform, metadata, nodata, bands,
    # types, blocks, checksums, masks and overviews) remains in the comparison.
    metadata.pop("description", None)
    metadata.pop("files", None)
    return metadata


def las_point_count(path: pathlib.Path) -> int | None:
    """Read the authoritative LAS/LASzip fixed-header point count."""
    try:
        with path.open("rb") as stream:
            header = stream.read(375)
    except OSError:
        return None
    if len(header) < 227 or header[:4] != b"LASF" or header[24] != 1:
        return None
    if header[25] >= 4:
        if len(header) < 255:
            return None
        return struct.unpack_from("<Q", header, 247)[0]
    return struct.unpack_from("<I", header, 107)[0]


def las_physical_facts(path: pathlib.Path) -> dict | None:
    """Read the placement-relevant physical facts from a LAS/LAZ header."""
    try:
        with path.open("rb") as stream:
            header = stream.read(375)
    except OSError:
        return None
    if len(header) < 227 or header[:4] != b"LASF" or header[24] != 1:
        return None
    raw_format = header[104]
    record_bytes = struct.unpack_from("<H", header, 105)[0]
    if raw_format & 0x40 or raw_format & 0x3f > 10 or not record_bytes:
        return None
    return {
        "point_format": raw_format & 0x3f,
        "record_bytes": record_bytes,
        "compressed": bool(raw_format & 0x80),
    }


def las_record_digest(path: pathlib.Path,
                      oracle: pathlib.Path | None,
                      environment: dict[str, str] | None) -> str | None:
    """SHA-256 of the ordered raw point-record block of a LAS/LAZ file.

    Headers, VLRs, EVLRs and the LAZ chunk table are excluded; a LAZ file is
    first decoded by the oracle with all header facts and Extra Bytes
    forwarded so both sides go through the same decoder. This is the D0261
    fast-mode comparator: point records must be bit-identical and in order.
    """
    try:
        with path.open("rb") as stream:
            header = stream.read(375)
    except OSError:
        return None
    if len(header) < 227 or header[:4] != b"LASF" or header[24] != 1:
        return None
    if header[104] & 0x80:
        if oracle is None:
            return None
        with tempfile.TemporaryDirectory(prefix="pdg-fast-decode-") as temp:
            decoded = pathlib.Path(temp) / "decoded.las"
            completed = subprocess.run(
                [str(oracle), "translate", str(path), str(decoded),
                 "--writers.las.forward=all",
                 "--writers.las.extra_dims=all"],
                env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if completed.returncode != 0 or not decoded.is_file():
                return None
            return las_record_digest(decoded, None, None)
    offset = struct.unpack_from("<I", header, 96)[0]
    length = struct.unpack_from("<H", header, 105)[0]
    if header[25] >= 4:
        if len(header) < 255:
            return None
        count = struct.unpack_from("<Q", header, 247)[0]
    else:
        count = struct.unpack_from("<I", header, 107)[0]
    digest = hashlib.sha256()
    digest.update(struct.pack("<HQ", length, count))
    remaining = length * count
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(1 << 20, remaining))
            if not chunk:
                return None
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def las_record_block(path: pathlib.Path,
                     oracle: pathlib.Path | None,
                     environment: dict[str, str] | None,
                     ) -> tuple[int, int, bytes] | None:
    """(record length, count, ordered raw record block) of a LAS/LAZ file.

    LAZ is decoded by the oracle exactly as las_record_digest does.
    """
    try:
        with path.open("rb") as stream:
            header = stream.read(375)
    except OSError:
        return None
    if len(header) < 227 or header[:4] != b"LASF" or header[24] != 1:
        return None
    if header[104] & 0x80:
        if oracle is None:
            return None
        with tempfile.TemporaryDirectory(prefix="pdg-fast-decode-") as temp:
            decoded = pathlib.Path(temp) / "decoded.las"
            completed = subprocess.run(
                [str(oracle), "translate", str(path), str(decoded),
                 "--writers.las.forward=all",
                 "--writers.las.extra_dims=all"],
                env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if completed.returncode != 0 or not decoded.is_file():
                return None
            return las_record_block(decoded, None, None)
    offset = struct.unpack_from("<I", header, 96)[0]
    length = struct.unpack_from("<H", header, 105)[0]
    if header[25] >= 4:
        if len(header) < 255:
            return None
        count = struct.unpack_from("<Q", header, 247)[0]
    else:
        count = struct.unpack_from("<I", header, 107)[0]
    with path.open("rb") as stream:
        stream.seek(offset)
        block = stream.read(length * count)
    if len(block) != length * count:
        return None
    return length, count, block


def las_record_compare(reference: tuple[int, int, bytes],
                       observed: tuple[int, int, bytes]) -> dict:
    """D0271 fast-contract record comparison.

    Records are compared in order. Coordinates (the leading three int32
    fields of every record) must be identical everywhere; the number of
    records whose remaining bytes differ is reported so a tie-order
    relaxation stays bounded and visible.
    """
    length, count, block = reference
    olength, ocount, oblock = observed
    result = {
        "las_records_compared": min(count, ocount),
        "las_records_layout_match": length == olength and count == ocount,
        "las_records_xyz_differing": None,
        "las_records_differing": None,
    }
    if length != olength or count != ocount:
        return result
    if block == oblock:
        result["las_records_xyz_differing"] = 0
        result["las_records_differing"] = 0
        return result
    try:
        import numpy  # type: ignore
    except ImportError:  # pragma: no cover - fallback without numpy
        xyz = 0
        differing = 0
        for index in range(count):
            start = index * length
            left = block[start:start + length]
            right = oblock[start:start + length]
            if left != right:
                differing += 1
                if left[:12] != right[:12]:
                    xyz += 1
        result["las_records_xyz_differing"] = xyz
        result["las_records_differing"] = differing
        return result
    left = numpy.frombuffer(block, dtype=numpy.uint8).reshape(count, length)
    right = numpy.frombuffer(oblock, dtype=numpy.uint8).reshape(count, length)
    rows = (left != right).any(axis=1)
    result["las_records_differing"] = int(rows.sum())
    result["las_records_xyz_differing"] = int(
        (left[:, :12] != right[:, :12]).any(axis=1).sum())
    return result


def observed_artifacts(specifications: list[dict],
                       record_oracle: pathlib.Path | None = None,
                       record_environment: dict[str, str] | None = None,
                       record_reference: dict | None = None,
                       record_role: str | None = None,
                       ) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    missing: list[str] = []
    for specification in specifications:
        matches = artifact_matches(specification)
        if not matches:
            missing.append(specification["logical_pattern"])
            continue
        for artifact in matches:
            record = {
                "name": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": sha256(artifact),
            }
            if artifact.suffix.lower() in (".tif", ".tiff"):
                metadata = raster_metadata(artifact)
                encoded = json.dumps(
                    metadata, sort_keys=True, separators=(",", ":")).encode()
                record["raster_metadata_sha256"] = hashlib.sha256(encoded).hexdigest()
                record["raster_metadata"] = metadata
            if artifact.suffix.lower() in (".las", ".laz"):
                point_count = las_point_count(artifact)
                if point_count is not None:
                    record["las_point_count"] = point_count
                if record_oracle is not None:
                    block = las_record_block(artifact, record_oracle,
                                             record_environment)
                    if block is None:
                        record["las_record_sha256"] = None
                    else:
                        digest = hashlib.sha256()
                        digest.update(struct.pack("<HQ", block[0], block[1]))
                        digest.update(block[2])
                        record["las_record_sha256"] = digest.hexdigest()
                        # D0271: the first oracle artifact of each name is the
                        # per-record reference every candidate artifact is
                        # compared against.
                        if record_reference is not None:
                            if record_role == "oracle":
                                record_reference.setdefault(artifact.name,
                                                            block)
                            elif record_role == "candidate" and \
                                    artifact.name in record_reference:
                                record.update(las_record_compare(
                                    record_reference[artifact.name], block))
            records.append(record)
    records.sort(key=lambda record: record["name"])
    return records, missing


def process_tree_rss_bytes(root_pid: int) -> int:
    total = 0
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            children = pathlib.Path(
                f"/proc/{pid}/task/{pid}/children").read_text().split()
            pending.extend(int(child) for child in children)
            for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return total


def run_once(executable: pathlib.Path, pipeline: pathlib.Path,
             artifacts: list[dict], environment: dict,
             input_paths: list[pathlib.Path], cache_state: str,
             record_oracle: pathlib.Path | None = None,
             record_environment: dict[str, str] | None = None,
             record_reference: dict | None = None,
             record_role: str | None = None,
             leading_args: list[str] | None = None) -> dict:
    remove_artifacts(artifacts)
    if cache_state == "cold":
        evict_file_cache(input_paths)
    started = time.perf_counter()
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            [str(executable), *(leading_args or []), "pipeline",
             str(pipeline)],
            stdout=stdout_file, stderr=stderr_file, env=environment)
        peak_rss_bytes = 0
        while process.poll() is None:
            peak_rss_bytes = max(
                peak_rss_bytes, process_tree_rss_bytes(process.pid))
            time.sleep(0.002)
        peak_rss_bytes = max(peak_rss_bytes, process_tree_rss_bytes(process.pid))
        returncode = process.returncode
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
    seconds = time.perf_counter() - started
    observed, missing = observed_artifacts(artifacts, record_oracle,
                                           record_environment,
                                           record_reference, record_role)
    return {
        "seconds": seconds,
        "returncode": returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_head": stderr.decode("utf-8", "replace")[:400],
        "peak_rss_bytes": peak_rss_bytes,
        "artifacts": observed,
        "missing_artifact_patterns": missing,
    }


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    fixtures = registered_fixtures(args)
    common_overrides = environment_overrides(
        args.environment, candidate_only=False)
    candidate_overrides = environment_overrides(
        args.candidate_env, candidate_only=True)
    environment = clean_environment(common_overrides)
    if args.frozen_time_library:
        environment["LD_PRELOAD"] = str(args.frozen_time_library.resolve())
        environment["PDAL_TEST_FROZEN_EPOCH"] = str(args.freeze_epoch)
    environments = {
        "oracle": dict(environment),
        "candidate": dict(environment, **candidate_overrides),
    }
    # Fast-contract record digests decode LAZ through the pinned oracle so both
    # sides share one decoder; the exact contract never needs them.
    record_oracle = args.oracle if args.contract == "fast" else None
    record_reference: dict | None = {} if args.contract == "fast" else None
    extent = fixture_extent(args.oracle, args.fixture_laz,
                            environments["oracle"])

    plans = {}
    for role in ("oracle", "candidate"):
        destination = args.work_dir / f"{args.label}-{role}.json"
        output_directory = args.work_dir / f"{args.label}-{role}-artifacts"
        plans[role] = (destination,
                       materialize(args.pipeline, destination, fixtures,
                                   extent, output_directory))

    executables = {"oracle": args.oracle, "candidate": args.candidate}
    leading_args = {"oracle": [], "candidate": list(args.candidate_arg)}
    input_paths = pipeline_input_paths(args.pipeline, fixtures)
    runs: list[dict] = []
    for _ in range(args.warmups):
        for role in ("oracle", "candidate"):
            record = run_once(executables[role], *plans[role],
                              environments[role], input_paths,
                              args.cache_state, record_oracle,
                              environments["oracle"], record_reference, role,
                              leading_args[role])
            record.update(label=role, phase="warmup")
            runs.append(record)
    for iteration in range(args.runs):
        order = ("oracle", "candidate") if iteration % 2 == 0 else \
                ("candidate", "oracle")
        for role in order:
            record = run_once(executables[role], *plans[role],
                              environments[role], input_paths,
                              args.cache_state, record_oracle,
                              environments["oracle"], record_reference, role,
                              leading_args[role])
            record.update(label=role, phase="measured", iteration=iteration)
            runs.append(record)

    def measured(role: str) -> list[dict]:
        return [r for r in runs if r["label"] == role and r["phase"] == "measured"]

    failures = [r for r in runs if r["returncode"] != 0]
    summary: dict = {}
    input_bytes = sum(path.stat().st_size for path in input_paths)
    for role in ("oracle", "candidate"):
        role_runs = measured(role)
        seconds = [r["seconds"] for r in role_runs]
        output_bytes = [sum(a["bytes"] for a in r["artifacts"])
                        for r in role_runs]
        peak_rss = [r["peak_rss_bytes"] for r in role_runs]
        median_seconds = statistics.median(seconds) if seconds else None
        median_output_bytes = statistics.median(output_bytes) if output_bytes else None
        summary[role] = {
            "seconds": seconds,
            "median_seconds": median_seconds,
            "minimum_seconds": min(seconds) if seconds else None,
            "maximum_seconds": max(seconds) if seconds else None,
            "peak_rss_bytes": peak_rss,
            "median_peak_rss_bytes": statistics.median(peak_rss) if peak_rss else None,
            "output_bytes": output_bytes,
            "median_output_bytes": median_output_bytes,
            "points_per_second": (extent["points"] / median_seconds
                                  if median_seconds else None),
            "input_mib_per_second": (input_bytes / (1 << 20) / median_seconds
                                     if median_seconds else None),
            "output_mib_per_second": (
                median_output_bytes / (1 << 20) / median_seconds
                if median_seconds and median_output_bytes is not None else None),
            "compression_ratio_input_to_output": (
                input_bytes / median_output_bytes
                if median_output_bytes else None),
        }

    oracle_hashes = [tuple((a["name"], a["bytes"], a["sha256"],
                            a.get("raster_metadata_sha256"),
                            a.get("las_point_count"))
                           for a in r["artifacts"])
                     for r in measured("oracle")]
    candidate_hashes = [tuple((a["name"], a["bytes"], a["sha256"],
                               a.get("raster_metadata_sha256"),
                               a.get("las_point_count"))
                              for a in r["artifacts"])
                        for r in measured("candidate")]
    # D0261 fast contract: LAS/LAZ artifacts compare by ordered point-record
    # digest; every other artifact still compares byte-for-byte.
    def record_view(run: dict) -> tuple:
        return tuple(
            (a["name"], a["las_record_sha256"])
            if a.get("las_record_sha256") is not None else
            (a["name"], a["bytes"], a["sha256"], a.get("raster_metadata_sha256"))
            for a in run["artifacts"])
    oracle_record_views = [record_view(r) for r in measured("oracle")]
    candidate_record_views = [record_view(r) for r in measured("candidate")]
    oracle_stdout_hashes = [r["stdout_sha256"] for r in measured("oracle")]
    candidate_stdout_hashes = [r["stdout_sha256"]
                               for r in measured("candidate")]
    oracle_stderr_hashes = [r["stderr_sha256"] for r in measured("oracle")]
    candidate_stderr_hashes = [r["stderr_sha256"]
                               for r in measured("candidate")]
    # A run that produced no artifact is never "exact": without this guard a
    # pipeline that failed outright compares None-to-None and reads as a pass.
    produced = (bool(oracle_hashes) and bool(candidate_hashes)
                and bool(oracle_hashes[0]) and bool(candidate_hashes[0])
                and all(not r["missing_artifact_patterns"] for r in runs))
    artifacts_exact = (produced
                       and len(set(oracle_hashes)) == 1
                       and len(set(candidate_hashes)) == 1
                       and oracle_hashes[0] == candidate_hashes[0])
    raster_metadata_present = any(
        value[3] is not None for run in oracle_hashes + candidate_hashes
        for value in run)
    raster_metadata_exact = (not raster_metadata_present or artifacts_exact)
    stdout_exact = (bool(oracle_stdout_hashes) and bool(candidate_stdout_hashes)
                    and len(set(oracle_stdout_hashes)) == 1
                    and len(set(candidate_stdout_hashes)) == 1
                    and oracle_stdout_hashes[0] == candidate_stdout_hashes[0])
    stderr_exact = (bool(oracle_stderr_hashes) and bool(candidate_stderr_hashes)
                    and len(set(oracle_stderr_hashes)) == 1
                    and len(set(candidate_stderr_hashes)) == 1
                    and oracle_stderr_hashes[0] == candidate_stderr_hashes[0])
    # D0271 fast contract: LAS/LAZ records are compared in order against the
    # first oracle artifact; count, layout, and every record's coordinates
    # must match, and the number of records whose other bytes differ (kNN
    # tie-order choices) must stay within the configured fraction. Oracle
    # and candidate must each be self-consistent across runs.
    candidate_las = [a for run in measured("candidate")
                     for a in run["artifacts"]
                     if a["name"].lower().endswith((".las", ".laz"))]
    fast_differing = max((a.get("las_records_differing") or 0
                          for a in candidate_las), default=0)
    fast_compared = min((a.get("las_records_compared") or 0
                         for a in candidate_las), default=0)
    fast_records_ok = all(
        a.get("las_records_layout_match") is True
        and a.get("las_records_xyz_differing") == 0
        and a.get("las_records_differing") is not None
        and a["las_records_differing"] <= (
            args.fast_max_differing_records_fraction *
            (a.get("las_records_compared") or 0))
        for a in candidate_las)
    records_exact = (produced
                     and len(set(oracle_record_views)) == 1
                     and len(set(candidate_record_views)) == 1
                     and all(a.get("las_record_sha256") is not None
                             for run in measured("oracle") +
                             measured("candidate")
                             for a in run["artifacts"]
                             if a["name"].lower().endswith((".las", ".laz")))
                     and fast_records_ok
                     # every non-LAS artifact must still be byte-identical
                     and all(o == c for o, c in zip(oracle_record_views[0],
                                                     candidate_record_views[0])
                             if len(o) == 4))
    records_identical = (records_exact
                         and oracle_record_views[0] == candidate_record_views[0])
    if args.contract == "fast":
        exact = not failures and records_exact
    else:
        exact = (not failures and artifacts_exact and stdout_exact and
                 stderr_exact)
    speedup = None
    if summary["oracle"]["median_seconds"] and summary["candidate"]["median_seconds"]:
        speedup = (summary["oracle"]["median_seconds"]
                   / summary["candidate"]["median_seconds"])

    # B0200/B0201: report whether the measured difference is resolvable at all.
    #
    # Five identical invocations at --runs 3 reported r1 anywhere from 0.9248x
    # to 0.9492x -- a 2.4 point range on a pipeline whose whole deficit is ~7
    # points. A bare median hides that, and D0208 makes these numbers the
    # release criterion, so the headline needs its uncertainty beside it.
    #
    # The runs alternate, so each iteration is a matched pair and per-pair
    # ratios cancel machine drift that affects both binaries together.
    #
    # The uncertainty is the standard error of those ratios, NOT their range.
    # B0200 shipped the range and it was wrong in a way worth recording: a
    # range grows with sample size, so collecting more data made a difference
    # look *less* resolvable. Three invocations at --runs 15 stabilised the
    # reported speedup (range 0.0137 against 0.0244 at --runs 3) while their
    # reported "spread" got worse. Standard error shrinks as sqrt(n), which is
    # the behaviour a reader adding runs expects.
    paired = []
    oracle_runs = measured("oracle")
    candidate_runs = measured("candidate")
    for left, right in zip(oracle_runs, candidate_runs):
        if right["seconds"]:
            paired.append(left["seconds"] / right["seconds"])
    resolution = {
        "paired_speedups": paired,
        "paired_mean": statistics.fmean(paired) if paired else None,
        "paired_spread": (max(paired) - min(paired)) if len(paired) > 1 else None,
        "candidate_faster_in": sum(1 for value in paired if value > 1.0),
        "pairs": len(paired),
    }
    if len(paired) > 1:
        error = statistics.stdev(paired) / math.sqrt(len(paired))
        # Two standard errors is a ~95% interval for this many pairs. It is an
        # approximation and is named as one; the point is a bound that shrinks
        # with evidence, not a precise coverage claim.
        resolution["paired_standard_error"] = error
        resolution["interval_half_width"] = 2.0 * error
        # Advisory only: never changes exactness or the reported speedup, just
        # how much weight the number can bear.
        resolution["resolvable"] = (
            abs(resolution["paired_mean"] - 1.0) > 2.0 * error)

    report = {
        "schema": "pdg-reference-pipeline-baseline-v1",
        "label": args.label,
        "pipeline": {
            "path": str(args.pipeline),
            "sha256": sha256(args.pipeline),
            "materialized": {
                role: {"path": str(plans[role][0]),
                       "sha256": sha256(plans[role][0])}
                for role in ("oracle", "candidate")
            },
        },
        "fixture": {
            "laz": {"path": str(args.fixture_laz),
                    "bytes": args.fixture_laz.stat().st_size,
                    "sha256": sha256(args.fixture_laz)},
            "copc": ({"path": str(args.fixture_copc),
                      "bytes": args.fixture_copc.stat().st_size,
                      "sha256": sha256(args.fixture_copc)}
                     if args.fixture_copc and args.fixture_copc.exists() else None),
            "registered": {
                placeholder: {"path": str(path), "bytes": path.stat().st_size,
                              "sha256": sha256(path)}
                for placeholder, path in fixtures.items()
            },
            "used_input_paths": [
                {"path": str(path), "bytes": path.stat().st_size,
                 "sha256": sha256(path)} for path in input_paths
            ],
            "used_input_bytes": input_bytes,
            "extent": extent,
        },
        "binaries": {
            "oracle": binary_manifest(
                args.oracle, candidate=False,
                environment=environments["oracle"]),
            "candidate": binary_manifest(
                args.candidate, candidate=True,
                environment=environments["candidate"]),
        },
        "environment": {
            "platform": platform.platform(),
            "cache_state": args.cache_state,
            "cache_eviction": (
                "POSIX_FADV_DONTNEED on every registered input before every run"
                if args.cache_state == "cold" else
                "%d untimed warmup(s) per executable; no explicit page-cache eviction"
                % args.warmups),
            "scrubbed_prefixes": ["PDG_", "PDAL_TEST_", "LD_"],
            "scrubbed_variables": ["PDAL_TEST_FROZEN_EPOCH"],
            "common_overrides": common_overrides,
            "candidate_overrides": candidate_overrides,
            "candidate_leading_args": list(args.candidate_arg),
            "frozen_time_library": (str(args.frozen_time_library)
                                    if args.frozen_time_library else None),
            "freeze_environment": (
                "PDAL_TEST_FROZEN_EPOCH"
                if args.frozen_time_library else None),
            "freeze_epoch": args.freeze_epoch,
            "peak_ram_measurement": (
                "2 ms /proc sampling; sum of resident bytes across the public "
                "process and its descendants"),
            # D0277: which placement profile the candidate would use on this
            # machine (embedded reference, a local `gpupal calibrate` profile,
            # or none). Recorded from the candidate itself so a default-mode
            # run states its own selection evidence.
            "candidate_placement_profile": candidate_placement_profile(
                args.candidate, environments["candidate"]),
        },
        "protocol": "alternating oracle/candidate, outputs removed before each run",
        "summary": summary,
        "comparison": {
            "contract": args.contract,
            "exact_outputs": exact,
            "exact_artifacts": artifacts_exact,
            "exact_records": (records_exact if args.contract == "fast"
                              else None),
            "identical_records": (records_identical if args.contract == "fast"
                                  else None),
            "fast_differing_records": (fast_differing
                                       if args.contract == "fast" else None),
            "fast_compared_records": (fast_compared
                                      if args.contract == "fast" else None),
            "fast_max_differing_records_fraction": (
                args.fast_max_differing_records_fraction
                if args.contract == "fast" else None),
            "exact_raster_metadata": raster_metadata_exact,
            "exact_stdout": stdout_exact,
            "exact_stderr": stderr_exact,
            "median_speedup": speedup,
            "resolution": resolution,
            "artifact_names": [a["name"] for a in
                               (measured("oracle")[0]["artifacts"] if measured("oracle") else [])],
            "oracle_artifact_sha256": [value[2] for value in oracle_hashes[0]]
                                      if oracle_hashes else [],
            "candidate_artifact_sha256": [value[2] for value in candidate_hashes[0]]
                                         if candidate_hashes else [],
        },
        "failures": [{"label": r["label"], "returncode": r["returncode"],
                      "stderr_head": r["stderr_head"]} for r in failures],
        "runs": runs,
    }
    if args.qualification_model and exact:
        output_paths = artifact_matches(plans["oracle"][1][0]) \
            if len(plans["oracle"][1]) == 1 else []
        if len(input_paths) != 1 or len(output_paths) != 1:
            raise SystemExit(
                "placement qualification requires one LAS input and one output")
        input_facts = las_physical_facts(input_paths[0])
        output_facts = las_physical_facts(output_paths[0])
        if input_facts is None or output_facts is None:
            raise SystemExit(
                "placement qualification could not read physical LAS facts")
        if candidate_overrides.get(
                "PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT") != "1":
            raise SystemExit(
                "placement qualification requires the automatic resident route")
        report["qualification"] = {
            "model": args.qualification_model,
            "input_point_format": input_facts["point_format"],
            "input_record_bytes": input_facts["record_bytes"],
            "input_compressed": input_facts["compressed"],
            "output_point_format": output_facts["point_format"],
            "output_record_bytes": output_facts["record_bytes"],
            "output_compressed": output_facts["compressed"],
            "required_automatic_resident_route": True,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.contract == "fast" and exact:
        status = ("records-identical" if records_identical else
                  f"records-exact ({fast_differing} of {fast_compared} "
                  "records differ within the tie-order allowance)")
    else:
        status = "exact" if exact else ("FAILED" if failures else "NOT EXACT")
    speed = f"{speedup:.6f}x" if speedup else "n/a"
    half = resolution.get("interval_half_width")
    note = ""
    if half is not None:
        note = (f"; paired {resolution['paired_mean']:.4f}x "
                f"+/- {half:.4f} ({resolution['pairs']} pairs)")
        if resolution.get("resolvable") is False:
            note += " (NOT RESOLVABLE: interval spans parity)"
    print(f"{args.label}: {status}; median speedup {speed}{note}; "
          f"report {args.report}")
    if failures:
        print(f"  {len(failures)} failing run(s); first: "
              f"{failures[0]['label']} rc={failures[0]['returncode']} "
              f"{failures[0]['stderr_head'][:160]}", file=sys.stderr)
        return 1
    if not exact:
        print("  exactness contract failed; refusing a performance result",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
