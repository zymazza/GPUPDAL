#!/usr/bin/env python3
"""Hash the inputs and retained files for a comparative-report bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def records(root: pathlib.Path, *, exclude: pathlib.Path | None = None) -> list[dict]:
    result = []
    for path in sorted(root.rglob("*"), key=lambda value:
                       value.relative_to(root).as_posix()):
        if not path.is_file() or (exclude and path.resolve() == exclude.resolve()):
            continue
        result.append({"path": path.relative_to(root).as_posix(),
                       "bytes": path.stat().st_size, "sha256": sha256(path)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-dir", required=True, type=pathlib.Path)
    parser.add_argument("--bundle-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if not args.input_dir.is_dir() or not args.bundle_dir.is_dir():
        parser.error("input and bundle directories must exist")
    value = {
        "schema": "pdg-separate-benchmark-package-v1",
        "source_commit": args.source_commit,
        "producer": "project-author automated repository run",
        "external_validation": "not-performed",
        "evidence_scope": "three machines; one capture per machine; historical medians of 3/2/1 timed repeats at 1M-4M/16M/47M",
        "inputs": records(args.input_dir),
        "bundle": records(args.bundle_dir, exclude=args.output),
        "limitations": [
            "Input hashes identify the exact staged bytes but do not redistribute the source corpus.",
            "The bundle does not contain the separate ten-GPU sweep raw data.",
            "Historical unlicensed LAStools and unhashed directory-output rows are exploratory, not qualifying evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: {len(value['inputs'])} inputs, "
          f"{len(value['bundle'])} bundle files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
