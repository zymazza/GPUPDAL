#!/usr/bin/env python3
"""Create or verify a closed, immutable PDG release-artifact manifest.

The manifest covers every regular file below an artifact payload directory.
Benchmark work directories must live elsewhere: adding, removing, or changing
one payload byte makes ``check`` fail.  This is the admission gate used by the
release re-proof and 3DEP study protocols so neither workflow can rebuild or
silently substitute a binary after timing starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import stat
import subprocess
import sys


SCHEMA = "pdg-frozen-release-artifact-v1"


def sha256(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def relative_files(root: pathlib.Path) -> list[pathlib.Path]:
    root = root.resolve()
    result: list[pathlib.Path] = []
    for path in sorted(root.rglob("*"),
                       key=lambda value: value.relative_to(root).as_posix()):
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"closed payload may not contain symlinks: {path}")
        if stat.S_ISDIR(status.st_mode):
            continue
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"closed payload may contain regular files only: {path}")
        try:
            path.resolve(strict=True).relative_to(root)
        except ValueError as error:
            raise ValueError(f"payload entry escapes root: {path}") from error
        result.append(path)
    return result


def file_records(root: pathlib.Path) -> list[dict[str, object]]:
    return [{"path": path.relative_to(root).as_posix(),
             "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in relative_files(root)]


def tree_digest(records: list[dict[str, object]]) -> str:
    encoded = json.dumps(records, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def role(root: pathlib.Path, relative: str, name: str) -> dict[str, object]:
    pure = pathlib.PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe {name} path: {relative!r}")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} does not exist below payload root: {relative}")
    try:
        path.resolve(strict=True).relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} escapes payload root: {relative}") from error
    return {"path": pure.as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def version(path: pathlib.Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [str(path), "--version"], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=30)
        return {"returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
                "text": (completed.stdout + completed.stderr).decode(
                    "utf-8", "replace")[:500]}
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": str(error)}


def create(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"payload root is not a directory: {root}")
    output = args.output.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("manifest output must be outside the closed payload root")
    records = file_records(root)
    if not records:
        raise ValueError("artifact payload is empty")
    candidate = role(root, args.candidate, "candidate")
    oracle = role(root, args.oracle, "oracle")
    candidate_path = root / str(candidate["path"])
    oracle_path = root / str(oracle["path"])
    result = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "release_id": args.release_id,
        "payload": {"file_count": len(records),
                    "bytes": sum(int(record["bytes"]) for record in records),
                    "tree_sha256": tree_digest(records), "files": records},
        "roles": {"candidate": candidate, "oracle": oracle,
                  "candidate_version": version(candidate_path),
                  "oracle_version": version(oracle_path)},
        "creation_environment": {
            "platform": platform.platform(), "python": platform.python_version(),
        },
        "benchmark_rule": "No build, install, profile conversion, or payload mutation after this manifest is created.",
    }
    if args.metadata:
        metadata_path = args.metadata.resolve()
        metadata = json.loads(metadata_path.read_text())
        result["metadata"] = metadata
        result["metadata_source_sha256"] = sha256(metadata_path)
    return result


def verify(manifest_path: pathlib.Path, root: pathlib.Path) -> tuple[bool, dict[str, object]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported frozen artifact manifest schema")
    expected = manifest.get("payload")
    if not isinstance(expected, dict) or not isinstance(expected.get("files"), list):
        raise ValueError("artifact manifest has no payload file list")
    actual_records = file_records(root.resolve())
    actual = {"file_count": len(actual_records),
              "bytes": sum(int(record["bytes"]) for record in actual_records),
              "tree_sha256": tree_digest(actual_records), "files": actual_records}
    if any(not isinstance(record, dict) or
           not isinstance(record.get("path"), str)
           for record in expected["files"]):
        raise ValueError("artifact manifest contains a malformed file record")
    expected_by_path = {record["path"]: record for record in expected["files"]}
    if len(expected_by_path) != len(expected["files"]):
        raise ValueError("artifact manifest contains duplicate payload paths")
    actual_by_path = {record["path"]: record for record in actual_records}
    differences: list[dict[str, object]] = []
    roles = manifest.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("artifact manifest has no executable roles")
    for name in ("candidate", "oracle"):
        record = roles.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"artifact manifest has no valid {name} role")
        observed_role = role(root.resolve(), record["path"], name)
        if observed_role != record:
            differences.append({"role": name, "expected": record,
                                "actual": observed_role})
        if expected_by_path.get(record["path"]) != record:
            differences.append({"role": name,
                                "detail": "role does not exactly match its payload inventory record"})
    for path in sorted(set(expected_by_path) | set(actual_by_path)):
        left = expected_by_path.get(path)
        right = actual_by_path.get(path)
        if left != right:
            differences.append({"path": path, "expected": left, "actual": right})
    for name in ("file_count", "bytes", "tree_sha256"):
        if expected.get(name) != actual[name]:
            differences.append({"field": name, "expected": expected.get(name),
                                "actual": actual[name]})
    report = {"schema": "pdg-frozen-release-artifact-check-v1",
              "manifest": str(manifest_path.resolve()),
              "manifest_sha256": sha256(manifest_path),
              "payload_root": str(root.resolve()), "valid": not differences,
              "actual": {key: value for key, value in actual.items()
                         if key != "files"}, "differences": differences}
    return not differences, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create", allow_abbrev=False)
    create_parser.add_argument("--root", required=True, type=pathlib.Path)
    create_parser.add_argument("--candidate", required=True)
    create_parser.add_argument("--oracle", required=True)
    create_parser.add_argument("--source-commit", required=True)
    create_parser.add_argument("--release-id", required=True)
    create_parser.add_argument("--metadata", type=pathlib.Path)
    create_parser.add_argument("--output", required=True, type=pathlib.Path)
    check = subparsers.add_parser("check", allow_abbrev=False)
    check.add_argument("--manifest", required=True, type=pathlib.Path)
    check.add_argument("--root", required=True, type=pathlib.Path)
    check.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            result = create(args)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2,
                                              sort_keys=True) + "\n")
            print(f"frozen artifact: {result['payload']['tree_sha256']} "
                  f"({result['payload']['file_count']} files)")
            return 0
        valid, report = verify(args.manifest.resolve(), args.root.resolve())
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered)
        else:
            print(rendered, end="")
        return 0 if valid else 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"gpupdal artifact manifest: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
