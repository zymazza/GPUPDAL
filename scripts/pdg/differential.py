#!/usr/bin/env python3
"""Compare all observable artifacts from pinned PDAL and a PDG candidate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import selectors
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from copc_semantic import compare as compare_copc_semantic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument(
        "--candidate-oracle", type=Path,
        help="PDAL executable the candidate delegates to (PDG_ORACLE_PDAL); "
             "defaults to --oracle. Pass the forked sibling pdal here when "
             "--oracle is the pinned upstream binary, so delegated fork "
             "execution is compared against pinned PDAL instead of "
             "silently becoming it (D0260)",
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--keep-success", action="store_true")
    parser.add_argument("--allow-empty-command", action="store_true")
    parser.add_argument(
        "--seed-file",
        action="append",
        default=[],
        metavar="RELATIVE=SOURCE",
        help="copy SOURCE to RELATIVE in both isolated working directories",
    )
    parser.add_argument("--frozen-time-library", type=Path)
    parser.add_argument(
        "--oracle-preload",
        action="append",
        default=[],
        type=Path,
        help=(
            "shared library to preload before the frozen-time shim for the oracle"
        ),
    )
    parser.add_argument(
        "--candidate-preload",
        action="append",
        default=[],
        type=Path,
        help=(
            "shared library to preload before the frozen-time shim for the candidate"
        ),
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set one explicit environment value only for the candidate",
    )
    parser.add_argument("--freeze-epoch", type=int, default=1_704_067_200)
    parser.add_argument(
        "--comparison-mode", choices=("bytes", "copc-canonical-v1"),
        default="bytes",
        help="bytes is the default product contract; copc-canonical-v1 is an "
             "explicit supplementary comparator for oracle-nondeterministic "
             "COPC artifacts only",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--max-artifact-files", type=int, default=4096)
    parser.add_argument("--max-artifact-bytes", type=int, default=64 << 30)
    parser.add_argument("--max-stream-bytes", type=int, default=64 << 20)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command and not args.allow_empty_command:
        parser.error("a PDAL command is required after --")
    if (args.timeout_seconds <= 0 or args.max_artifact_files <= 0 or
            args.max_artifact_bytes <= 0 or args.max_stream_bytes <= 0):
        parser.error("timeout and artifact limits must be positive")
    return args


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def environment(oracle: Path, frozen_time_library: Path | None,
        freeze_epoch: int) -> dict[str, str]:
    # Build the product environment from a narrow execution allowlist. In
    # particular, no ambient PDG_*, PDAL_TEST_*, or loader control can alter a
    # supposed default-mode comparison. Sanitizer libraries are converted to
    # explicit preload arguments by main() before this function is called.
    allowed = {
        "PATH", "TMPDIR", "TMP", "TEMP", "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES", "ASAN_OPTIONS",
        "LSAN_OPTIONS", "TSAN_OPTIONS", "UBSAN_OPTIONS",
    }
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result.update(
        {
            "LC_ALL": "C",
            "TZ": "UTC",
            "PDG_ORACLE_PDAL": str(oracle.resolve()),
        }
    )
    if frozen_time_library and os.name != "nt":
        preload = str(frozen_time_library.resolve())
        if result.get("LD_PRELOAD"):
            preload += ":" + result["LD_PRELOAD"]
        result["LD_PRELOAD"] = preload
        result["PDAL_TEST_FROZEN_EPOCH"] = str(freeze_epoch)
    return result


def sanitizer_preloads_from_environment() -> list[Path]:
    """Return CTest-provided sanitizer libraries in required load order.

    Differential tests normally prepend the frozen-time shim.  ASan must load
    before that shim, so sanitizer builds pass the runtime here and main()
    folds it into the existing explicit preload lists before constructing each
    process environment.
    """
    value = os.environ.get("PDG_DIFFERENTIAL_ASAN_PRELOAD", "")
    if not value:
        return []
    result = [Path(item) for item in value.split(":") if item]
    for preload in result:
        if not preload.is_file():
            raise ValueError(f"sanitizer preload does not exist: {preload}")
    return result


def environment_assignments(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, assignment = value.partition("=")
        if not separator or not name or "\0" in name or "\0" in assignment:
            raise ValueError(f"invalid environment assignment: {value!r}")
        result[name] = assignment
    return result


def role_environment(env: dict[str, str], directory: Path) -> dict[str, str]:
    # Keep product configuration hermetic without removing the basic HOME/XDG
    # contract expected by GDAL/PDAL drivers such as writers.copc.  These
    # role-local directories live outside the compared artifact trees, so a
    # cache or configuration file cannot masquerade as pipeline output.
    process_env = dict(env)
    home = directory.parent / f".{directory.name}-home"
    config = home / ".config"
    config.mkdir(parents=True, exist_ok=True)
    process_env["HOME"] = str(home)
    process_env["XDG_CONFIG_HOME"] = str(config)
    if os.name == "nt":
        roaming = home / "AppData" / "Roaming"
        local = home / "AppData" / "Local"
        roaming.mkdir(parents=True, exist_ok=True)
        local.mkdir(parents=True, exist_ok=True)
        home_text = str(home)
        drive = home.drive or directory.drive
        home_path = home_text[len(drive):] if drive else home_text
        process_env.update({
            "USERPROFILE": home_text,
            "HOMEDRIVE": drive,
            "HOMEPATH": home_path,
            "APPDATA": str(roaming),
            "LOCALAPPDATA": str(local),
        })
    return process_env


def run(program: Path, command: list[str], directory: Path,
        env: dict[str, str], timeout_seconds: float,
        max_stream_bytes: int) -> dict[str, object]:
    process_env = role_environment(env, directory)
    process_options: dict[str, object] = {}
    if os.name == "nt":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        process_options["bufsize"] = 0
    else:
        process_options["start_new_session"] = True
    process = subprocess.Popen(
        [str(program.resolve()), *command], cwd=directory, env=process_env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, **process_options)
    assert process.stdout is not None and process.stderr is not None
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    counts = {"stdout": 0, "stderr": 0}
    termination_reason: str | None = None
    deadline = time.monotonic() + timeout_seconds

    def terminate(reason: str) -> None:
        nonlocal termination_reason
        if termination_reason is not None:
            return
        termination_reason = reason
        if process.poll() is None:
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=2.0, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    if os.name == "nt":
        overflow_streams: list[str] = []
        capture_errors: list[str] = []
        capture_event = threading.Event()

        def capture_stream(stream: str, pipe: object) -> None:
            try:
                while True:
                    chunk = pipe.read(1 << 16)
                    if not chunk:
                        return
                    counts[stream] += len(chunk)
                    room = max_stream_bytes - len(captured[stream])
                    if room > 0:
                        captured[stream].extend(chunk[:room])
                    if counts[stream] > max_stream_bytes:
                        overflow_streams.append(stream)
                        capture_event.set()
                        return
            except (OSError, ValueError) as error:
                capture_errors.append(f"{stream}: {error}")
                capture_event.set()

        readers = [
            threading.Thread(
                target=capture_stream, args=("stdout", process.stdout),
                daemon=True),
            threading.Thread(
                target=capture_stream, args=("stderr", process.stderr),
                daemon=True),
        ]
        for reader in readers:
            reader.start()
        while any(reader.is_alive() for reader in readers):
            remaining = deadline - time.monotonic()
            if overflow_streams:
                terminate(f"{overflow_streams[0]}-limit")
                break
            if capture_errors:
                terminate("capture-error")
                break
            if remaining <= 0.0:
                terminate("timeout")
                break
            capture_event.wait(timeout=min(0.1, remaining))
        if termination_reason is not None:
            process.stdout.close()
            process.stderr.close()
        for reader in readers:
            reader.join(timeout=2.0)
    else:
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map() and termination_reason is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                terminate("timeout")
                remaining = 0.1
            events = selector.select(timeout=min(0.1, max(0.0, remaining)))
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 1 << 16)
                stream = key.data
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                counts[stream] += len(chunk)
                room = max_stream_bytes - len(captured[stream])
                if room > 0:
                    captured[stream].extend(chunk[:room])
                if counts[stream] > max_stream_bytes:
                    terminate(f"{stream}-limit")
        # Do not wait for EOF after containment fires. A malicious grandchild
        # can detach into a new session while retaining a pipe descriptor;
        # closing our read ends keeps that escape from defeating the
        # wall-clock/stream bound.
        for key in list(selector.get_map().values()):
            selector.unregister(key.fileobj)
        selector.close()
    process.stdout.close()
    process.stderr.close()
    try:
        returncode = process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            returncode = process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            returncode = 124 if termination_reason == "timeout" else 125
    stdout = bytes(captured["stdout"])
    stderr = bytes(captured["stderr"])
    if termination_reason == "timeout":
        returncode = 124
        notice = f"\npdg differential timeout after {timeout_seconds}s\n".encode()
        stderr = (stderr[:max_stream_bytes - len(notice)] + notice
                  if len(notice) < max_stream_bytes else notice[:max_stream_bytes])
    elif termination_reason is not None:
        returncode = 125
    return {
        "returncode": returncode, "stdout": stdout, "stderr": stderr,
        "stream_bytes": counts,
        "stream_limit_exceeded": {
            "stdout": counts["stdout"] > max_stream_bytes,
            "stderr": counts["stderr"] > max_stream_bytes,
        },
    }


def files(directory: Path, max_files: int | None = None,
          max_bytes: int | None = None) -> dict[str, dict[str, object]]:
    paths: list[tuple[Path, int]] = []
    directories = [directory]
    total_bytes = 0
    entries = 0
    entry_limit = max_files * 4 + 64 if max_files is not None else None
    while directories:
        current = directories.pop()
        with os.scandir(current) as iterator:
            for entry in iterator:
                entries += 1
                if entry_limit is not None and entries > entry_limit:
                    raise ValueError(
                        f"artifact tree entries exceed limit {entry_limit}")
                path = Path(entry.path)
                if entry.is_symlink():
                    raise ValueError(f"artifact tree contains a symlink: {path}")
                if entry.is_dir(follow_symlinks=False):
                    directories.append(path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"artifact tree contains a special file: {path}")
                if max_files is not None and len(paths) >= max_files:
                    raise ValueError(
                        f"artifact count exceeds limit {max_files}")
                size = entry.stat(follow_symlinks=False).st_size
                total_bytes += size
                if max_bytes is not None and total_bytes > max_bytes:
                    raise ValueError(
                        f"artifact bytes exceed limit {max_bytes}")
                paths.append((path, size))
    paths.sort(key=lambda item: item[0].relative_to(directory).as_posix())
    result: dict[str, dict[str, object]] = {}
    for path, size in paths:
        relative = path.relative_to(directory).as_posix()
        result[relative] = {
            "bytes": size,
            "sha256": file_digest(path),
            "path": path,
        }
    return result


def byte_difference(expected: bytes, actual: bytes) -> dict[str, object] | None:
    common = min(len(expected), len(actual))
    offset = next(
        (index for index in range(common) if expected[index] != actual[index]),
        common if len(expected) != len(actual) else None,
    )
    if offset is None:
        return None
    start = max(0, offset - 16)
    end = offset + 17
    return {
        "offset": offset,
        "expected_size": len(expected),
        "actual_size": len(actual),
        "expected_hex": expected[start:end].hex(),
        "actual_hex": actual[start:end].hex(),
    }


def file_difference(expected: Path, actual: Path) -> dict[str, object] | None:
    expected_size = expected.stat().st_size
    actual_size = actual.stat().st_size
    offset = 0
    with expected.open("rb") as expected_stream, actual.open("rb") as actual_stream:
        while True:
            expected_block = expected_stream.read(1024 * 1024)
            actual_block = actual_stream.read(1024 * 1024)
            if expected_block == actual_block:
                if not expected_block:
                    return None
                offset += len(expected_block)
                continue
            common = min(len(expected_block), len(actual_block))
            relative = next(
                (
                    index
                    for index in range(common)
                    if expected_block[index] != actual_block[index]
                ),
                common,
            )
            offset += relative
            break

    start = max(0, offset - 16)
    with expected.open("rb") as stream:
        stream.seek(start)
        expected_context = stream.read(33)
    with actual.open("rb") as stream:
        stream.seek(start)
        actual_context = stream.read(33)
    return {
        "offset": offset,
        "expected_size": expected_size,
        "actual_size": actual_size,
        "expected_hex": expected_context.hex(),
        "actual_hex": actual_context.hex(),
    }


LAS_HEADER_FIELDS = (
    (0, 4, "file_signature"), (4, 6, "file_source_id"),
    (6, 8, "global_encoding"), (8, 24, "project_id"),
    (24, 26, "version"), (26, 58, "system_identifier"),
    (58, 90, "generating_software"), (90, 94, "creation_date"),
    (94, 96, "header_size"), (96, 100, "offset_to_point_data"),
    (100, 104, "vlr_count"), (104, 105, "point_format"),
    (105, 107, "point_record_length"), (107, 111, "legacy_point_count"),
    (111, 131, "legacy_points_by_return"), (131, 155, "scale"),
    (155, 179, "offset"), (179, 227, "bounds"),
    (227, 235, "waveform_data_offset"), (235, 243, "evlr_offset"),
    (243, 247, "evlr_count"), (247, 255, "point_count"),
    (255, 375, "points_by_return"),
)


LEGACY_POINT_FIELDS = (
    (0, 4, "X"), (4, 8, "Y"), (8, 12, "Z"),
    (12, 14, "Intensity"), (14, 15, "return_flags"),
    (15, 16, "Classification"), (16, 17, "ScanAngleRank"),
    (17, 18, "UserData"), (18, 20, "PointSourceId"),
)


MODERN_POINT_FIELDS = (
    (0, 4, "X"), (4, 8, "Y"), (8, 12, "Z"),
    (12, 14, "Intensity"), (14, 15, "return_flags"),
    (15, 16, "classification_flags"), (16, 17, "Classification"),
    (17, 18, "UserData"), (18, 20, "ScanAngle"),
    (20, 22, "PointSourceId"), (22, 30, "GpsTime"),
)


def _point_fields(point_format: int) -> tuple[tuple[int, int, str], ...]:
    if point_format >= 6:
        result = list(MODERN_POINT_FIELDS)
        base = 30
        if point_format in (7, 8, 10):
            result.extend(((30, 32, "Red"), (32, 34, "Green"),
                           (34, 36, "Blue")))
            base = 36
        if point_format in (8, 10):
            result.append((36, 38, "Infrared"))
            base = 38
        if point_format in (9, 10):
            result.append((base, base + 29, "waveform_packet"))
        return tuple(result)
    result = list(LEGACY_POINT_FIELDS)
    base = 20
    if point_format in (1, 3, 4, 5):
        result.append((base, base + 8, "GpsTime"))
        base += 8
    if point_format in (2, 3, 5):
        result.extend(((base, base + 2, "Red"),
                       (base + 2, base + 4, "Green"),
                       (base + 4, base + 6, "Blue")))
        base += 6
    if point_format in (4, 5):
        result.append((base, base + 29, "waveform_packet"))
    return tuple(result)


def las_difference_context(path: Path, offset: int) -> dict[str, object] | None:
    """Decode the first LAS-family byte difference as far as the container allows."""
    try:
        with path.open("rb") as stream:
            header = stream.read(375)
            size = path.stat().st_size
            if len(header) < 227 or header[:4] != b"LASF":
                return None
            header_bytes = int.from_bytes(header[94:96], "little")
            point_offset = int.from_bytes(header[96:100], "little")
            vlr_count = int.from_bytes(header[100:104], "little")
            record_bytes = int.from_bytes(header[105:107], "little")
            compressed = bool(header[104] & 0x80)
            point_format = header[104] & 0x3F
            count = (int.from_bytes(header[247:255], "little")
                     if header[25] >= 4 and len(header) >= 255
                     else int.from_bytes(header[107:111], "little"))
            if offset < header_bytes:
                field = next((name for start, end, name in LAS_HEADER_FIELDS
                              if start <= offset < end), "header_padding")
                return {"container": "las", "region": "header",
                        "field": field, "relative_offset": offset}
            stream.seek(header_bytes)
            for index in range(vlr_count):
                start = stream.tell()
                prefix = stream.read(54)
                if len(prefix) != 54:
                    break
                length = int.from_bytes(prefix[20:22], "little")
                end = start + 54 + length
                if start <= offset < end:
                    return {
                        "container": "las", "region": "vlr",
                        "vlr_index": index,
                        "user_id": prefix[2:18].split(b"\0", 1)[0].decode(
                            "ascii", "replace"),
                        "record_id": int.from_bytes(prefix[18:20], "little"),
                        "relative_offset": offset - start,
                    }
                stream.seek(length, 1)
            point_end = point_offset + record_bytes * count
            if point_offset <= offset < point_end:
                if compressed:
                    return {"container": "laz", "region": "compressed_points",
                            "relative_offset": offset - point_offset,
                            "semantic_decode_required": True}
                relative = offset - point_offset
                point_index, within = divmod(relative, record_bytes)
                fields = _point_fields(point_format)
                field = next((name for start, end, name in fields
                              if start <= within < end), "ExtraBytes")
                return {"container": "las", "region": "point_record",
                        "point_index": point_index, "point_format": point_format,
                        "record_bytes": record_bytes, "field": field,
                        "record_offset": within}
            return {"container": "las", "region": "evlr_or_trailing",
                    "relative_offset": offset, "file_bytes": size}
    except (OSError, ValueError):
        return None


def serial_run(result: dict[str, object]) -> dict[str, object]:
    stdout = result["stdout"]
    stderr = result["stderr"]
    assert isinstance(stdout, bytes) and isinstance(stderr, bytes)
    preview_bytes = 4096
    return {
        "returncode": result["returncode"],
        "stdout_bytes": result.get("stream_bytes", {}).get("stdout", len(stdout)),
        "stdout_sha256": digest(stdout),
        "stdout_preview": stdout[:preview_bytes].decode(
            "utf-8", errors="backslashreplace"
        ),
        "stdout_preview_truncated": len(stdout) > preview_bytes,
        "stderr_bytes": result.get("stream_bytes", {}).get("stderr", len(stderr)),
        "stderr_sha256": digest(stderr),
        "stderr_preview": stderr[:preview_bytes].decode(
            "utf-8", errors="backslashreplace"
        ),
        "stderr_preview_truncated": len(stderr) > preview_bytes,
    }


def compare(oracle_run: dict[str, object], candidate_run: dict[str, object],
            oracle_directory: Path, candidate_directory: Path,
            expected_files: dict[str, dict[str, object]] | None = None,
            actual_files: dict[str, dict[str, object]] | None = None
            ) -> list[dict[str, object]]:
    differences: list[dict[str, object]] = []
    if oracle_run["returncode"] != candidate_run["returncode"]:
        differences.append(
            {
                "artifact": "returncode",
                "expected": oracle_run["returncode"],
                "actual": candidate_run["returncode"],
            }
        )
    for stream in ("stdout", "stderr"):
        expected = oracle_run[stream]
        actual = candidate_run[stream]
        assert isinstance(expected, bytes) and isinstance(actual, bytes)
        difference = byte_difference(expected, actual)
        if difference:
            differences.append({"artifact": stream, **difference})

    if expected_files is None:
        expected_files = files(oracle_directory)
    if actual_files is None:
        actual_files = files(candidate_directory)
    for relative in sorted(set(expected_files) | set(actual_files)):
        expected = expected_files.get(relative)
        actual = actual_files.get(relative)
        if expected is None or actual is None:
            differences.append(
                {
                    "artifact": relative,
                    "expected": "present" if expected else "missing",
                    "actual": "present" if actual else "missing",
                }
            )
            continue
        if expected["sha256"] != actual["sha256"]:
            expected_path = expected["path"]
            actual_path = actual["path"]
            assert isinstance(expected_path, Path) and isinstance(actual_path, Path)
            detail = file_difference(expected_path, actual_path) or {}
            decoded = (las_difference_context(expected_path, detail["offset"])
                       if "offset" in detail and relative.lower().endswith(
                           (".las", ".laz")) else None)
            differences.append(
                {
                    "artifact": relative,
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": actual["sha256"],
                    **detail,
                    **({"decoded_difference": decoded} if decoded else {}),
                }
            )
    return differences


def serial_files(directory: Path,
                 inventory: dict[str, dict[str, object]] | None = None
                 ) -> dict[str, dict[str, object]]:
    return {
        relative: {"bytes": record["bytes"], "sha256": record["sha256"]}
        for relative, record in (inventory if inventory is not None
                                 else files(directory)).items()
    }


def seed_files(specifications: list[str], oracle_directory: Path,
               candidate_directory: Path) -> list[dict[str, object]]:
    seeded: list[dict[str, object]] = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"seed file must use RELATIVE=SOURCE: {specification}"
            )
        relative_text, source_text = specification.split("=", 1)
        relative = Path(relative_text)
        source = Path(source_text).resolve()
        if (not relative_text or relative.is_absolute() or ".." in relative.parts):
            raise ValueError(f"unsafe seed file destination: {relative_text}")
        if not source.is_file():
            raise ValueError(f"seed file source does not exist: {source}")
        for directory in (oracle_directory, candidate_directory):
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        seeded.append(
            {
                "relative": relative.as_posix(),
                "source": str(source),
                "bytes": source.stat().st_size,
                "sha256": file_digest(source),
            }
        )
    return seeded


def main() -> int:
    args = parse_args()
    if not args.oracle.is_file() or not os.access(args.oracle, os.X_OK):
        print(f"oracle is not executable: {args.oracle}", file=sys.stderr)
        return 2
    if not args.candidate.is_file() or not os.access(args.candidate, os.X_OK):
        print(f"candidate is not executable: {args.candidate}", file=sys.stderr)
        return 2
    if args.frozen_time_library and not args.frozen_time_library.is_file():
        print(
            f"frozen-time library does not exist: {args.frozen_time_library}",
            file=sys.stderr,
        )
        return 2
    try:
        sanitizer_preloads = sanitizer_preloads_from_environment()
        candidate_assignments = environment_assignments(args.candidate_env)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    args.oracle_preload = [*sanitizer_preloads, *args.oracle_preload]
    args.candidate_preload = [*sanitizer_preloads, *args.candidate_preload]
    for preload in (*args.oracle_preload, *args.candidate_preload):
        if not preload.is_file():
            print(f"preload does not exist: {preload}", file=sys.stderr)
            return 2

    args.work_dir.mkdir(parents=True, exist_ok=True)
    case_root = Path(tempfile.mkdtemp(prefix=f"{args.case}-", dir=args.work_dir))
    oracle_directory = case_root / "oracle"
    candidate_directory = case_root / "candidate"
    oracle_directory.mkdir()
    candidate_directory.mkdir()

    try:
        seeded = seed_files(
            args.seed_file, oracle_directory, candidate_directory
        )
    except (OSError, ValueError) as error:
        print(f"unable to seed differential case: {error}", file=sys.stderr)
        shutil.rmtree(case_root)
        return 2

    oracle_env = environment(
        args.oracle, args.frozen_time_library, args.freeze_epoch
    )
    if args.oracle_preload and os.name != "nt":
        preload = ":".join(str(path.resolve()) for path in args.oracle_preload)
        if oracle_env.get("LD_PRELOAD"):
            preload += ":" + oracle_env["LD_PRELOAD"]
        oracle_env["LD_PRELOAD"] = preload
    candidate_env = environment(
        args.candidate_oracle or args.oracle, args.frozen_time_library,
        args.freeze_epoch
    )
    candidate_env.update(candidate_assignments)
    if args.candidate_preload and os.name != "nt":
        preload = ":".join(str(path.resolve()) for path in args.candidate_preload)
        if candidate_env.get("LD_PRELOAD"):
            preload += ":" + candidate_env["LD_PRELOAD"]
        candidate_env["LD_PRELOAD"] = preload
    local_day_before = dt.datetime.now().astimezone().date()
    oracle_run = run(args.oracle, args.command, oracle_directory, oracle_env,
                     args.timeout_seconds, args.max_stream_bytes)
    candidate_run = run(
        args.candidate, args.command, candidate_directory, candidate_env,
        args.timeout_seconds, args.max_stream_bytes
    )
    local_day_after = dt.datetime.now().astimezone().date()
    inventories: dict[str, dict[str, dict[str, object]]] = {}
    resource_differences: list[dict[str, object]] = []
    for role, directory in (("oracle", oracle_directory),
                            ("candidate", candidate_directory)):
        try:
            inventories[role] = files(
                directory, args.max_artifact_files, args.max_artifact_bytes)
        except (OSError, ValueError) as error:
            inventories[role] = {}
            resource_differences.append({
                "artifact": f"{role}_resource_limit", "detail": str(error),
                "max_files": args.max_artifact_files,
                "max_bytes": args.max_artifact_bytes,
            })
    differences = compare(
        oracle_run, candidate_run, oracle_directory, candidate_directory,
        inventories["oracle"], inventories["candidate"])
    differences.extend(resource_differences)
    if os.name == "nt" and local_day_after != local_day_before:
        differences.append({
            "artifact": "windows_local_date_boundary",
            "before": local_day_before.isoformat(),
            "after": local_day_after.isoformat(),
            "detail": "rerun the exact comparison within one local date",
        })
    for role, result in (("oracle", oracle_run), ("candidate", candidate_run)):
        exceeded = result.get("stream_limit_exceeded", {})
        for stream in ("stdout", "stderr"):
            if exceeded.get(stream):
                differences.append({
                    "artifact": f"{role}_{stream}_resource_limit",
                    "bytes": result.get("stream_bytes", {}).get(stream),
                    "max_bytes": args.max_stream_bytes,
                })
    semantic_comparisons: list[dict[str, object]] = []
    if args.comparison_mode == "copc-canonical-v1":
        remaining: list[dict[str, object]] = []
        for difference in differences:
            artifact = str(difference.get("artifact", ""))
            expected = oracle_directory / artifact
            actual = candidate_directory / artifact
            if (artifact.lower().endswith(".copc.laz") and expected.is_file()
                    and actual.is_file()):
                try:
                    semantic = compare_copc_semantic(
                        args.oracle.resolve(), expected, actual,
                        role_environment(oracle_env, oracle_directory),
                        max_input_bytes=args.max_artifact_bytes,
                        max_decoded_bytes=args.max_artifact_bytes * 8)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    semantic = {
                        "schema": "pdg-copc-canonical-v1",
                        "semantic_equal": False,
                        "differences": [{"field": "comparator_error",
                                         "actual": str(error)}],
                    }
                semantic_comparisons.append({"artifact": artifact, **semantic})
                if semantic.get("semantic_equal"):
                    continue
            remaining.append(difference)
        differences = remaining
    report = {
        "schema": 1,
        "case": args.case,
        "command": args.command,
        "oracle": str(args.oracle.resolve()),
        "candidate": str(args.candidate.resolve()),
        "candidate_oracle": str((args.candidate_oracle or args.oracle).resolve()),
        "freeze_epoch": (args.freeze_epoch
                         if args.frozen_time_library and os.name != "nt"
                         else None),
        "time_control": ("windows-same-local-date"
                         if os.name == "nt" else
                         "preloaded-frozen-epoch"
                         if args.frozen_time_library else "none"),
        "oracle_preload": [
            str(path.resolve()) for path in args.oracle_preload
        ],
        "candidate_preload": [
            str(path.resolve()) for path in args.candidate_preload
        ],
        "candidate_environment": candidate_assignments,
        "comparison_mode": args.comparison_mode,
        "semantic_comparisons": semantic_comparisons,
        "limits": {
            "timeout_seconds": args.timeout_seconds,
            "max_artifact_files": args.max_artifact_files,
            "max_artifact_bytes": args.max_artifact_bytes,
            "max_stream_bytes": args.max_stream_bytes,
        },
        "seed_files": seeded,
        "oracle_run": serial_run(oracle_run),
        "candidate_run": serial_run(candidate_run),
        "oracle_artifacts": serial_files(oracle_directory,
                                         inventories["oracle"]),
        "candidate_artifacts": serial_files(candidate_directory,
                                            inventories["candidate"]),
        "differences": differences,
        "case_root": str(case_root),
    }
    reports = args.work_dir / "reports"
    reports.mkdir(exist_ok=True)
    report_path = reports / f"{args.case}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if differences:
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(f"exact match: {args.case} ({report_path})")
    if not args.keep_success:
        shutil.rmtree(case_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
