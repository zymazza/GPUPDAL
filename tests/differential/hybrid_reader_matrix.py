#!/usr/bin/env python3
"""Exact hybrid-region replay through diverse upstream PDAL readers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    reader: str
    fixture: str
    input_name: str


CASES = (
    Case("laz", "readers.las", "laz/simple.laz", "input.laz"),
    Case(
        "copc",
        "readers.copc",
        "copc/1.2-with-color.copc.laz",
        "input.copc.laz",
    ),
    Case("bpf", "readers.bpf", "bpf/simple-extra.bpf", "input.bpf"),
    Case("ply", "readers.ply", "ply/simple_binary.ply", "input.ply"),
    Case("pcd", "readers.pcd", "pcd/utm17_space.pcd", "input.pcd"),
    Case("text", "readers.text", "text/utm17_1.txt", "input.txt"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--case-prefix", required=True)
    parser.add_argument(
        "--original-streamable",
        action="store_true",
        help=(
            "Omit the ordering filter and exercise the proven "
            "stream-to-standard execution-mode bridge"
        ),
    )
    return parser.parse_args()


def pipeline(case: Case, original_streamable: bool) -> dict[str, object]:
    stages: list[dict[str, object]] = [
        {"type": case.reader, "filename": case.input_name}
    ]
    if not original_streamable:
        stages.append(
            {
                "type": "filters.sort",
                "dimensions": "X",
                "algorithm": "STABLE",
            }
        )
    stages.extend(
        [
            {
                "type": "filters.assign",
                "value": ["Scratch = X + Y", "Classification = 2"],
            },
            {
                "type": "filters.expression",
                "expression": "Scratch == Scratch && X < Y",
            },
            {
                "type": "filters.crop",
                "bounds": "([-1e9,1e9],[-1e9,1e9],[-1e9,1e9])",
            },
            {
                "type": "filters.assign",
                "value": "Classification = 5 WHERE Z >= lowest()",
            },
            {
                "type": "filters.ferry",
                "dimensions": "Classification=>UserData",
            },
            {"type": "writers.las", "filename": "out.las"},
        ]
    )
    return {"pipeline": stages}


def main() -> int:
    args = parse_args()
    mode = "stream" if args.original_streamable else "sorted"
    generated = args.work_dir / f"generated-hybrid-readers-{mode}"
    generated.mkdir(parents=True, exist_ok=True)

    cases = tuple(
        case
        for case in CASES
        if not args.original_streamable or case.reader != "readers.copc"
    )
    for case in cases:
        fixture = (args.fixture_root / case.fixture).resolve()
        if not fixture.is_file():
            print(f"missing hybrid reader fixture: {fixture}", file=sys.stderr)
            return 2
        pipeline_path = generated / f"{case.name}.json"
        pipeline_path.write_text(
            json.dumps(
                pipeline(case, args.original_streamable),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        name = f"{args.case_prefix}-{case.name}"
        command = [
            sys.executable,
            str(args.differential.resolve()),
            "--oracle",
            str(args.oracle.resolve()),
            "--candidate",
            str(args.candidate.resolve()),
            "--case",
            name,
            "--work-dir",
            str(args.work_dir.resolve()),
            "--frozen-time-library",
            str(args.frozen_time_library.resolve()),
            "--seed-file",
            f"{case.input_name}={fixture}",
            "--seed-file",
            f"pipeline.json={pipeline_path.resolve()}",
            "--",
            "pipeline",
            "pipeline.json",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode

    print(f"exact hybrid reader matrix: {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
