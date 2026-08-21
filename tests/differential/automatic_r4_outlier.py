#!/usr/bin/env python3
"""Prove B0227's r4 selector, header summary, and CUDA-use guard.

D0272/B0272 retired the selector from automatic selection (the exact host
path is faster at 1M and 4M); the route stays reachable behind
PDG_EXPERIMENTAL_AUTOMATIC_R4_OUTLIER_CUDA, which this lane sets so the
selection, header, and refusal proofs keep exercising it. A final control
proves that without the opt-in the same 1M layout is not selected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


PROOF = "PDG_REQUIRE_AUTOMATIC_R4_OUTLIER_CUDA"
OPT_IN = "PDG_EXPERIMENTAL_AUTOMATIC_R4_OUTLIER_CUDA"
INJECT_FALLBACK = "PDG_TEST_R4_OUTLIER_RECOVERABLE_CUDA_FAILURE"
SELECTED_DIAGNOSTIC = b"required automatic exact CUDA r4 outlier path was not selected"
USED_DIAGNOSTIC = b"required automatic exact CUDA r4 outlier path was not used"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--cuda-toolkit", required=True)
    return parser.parse_args()


def clean_environment(**extra: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("PDG_") and name != "PDG_DIFFERENTIAL_ASAN_PRELOAD":
            environment.pop(name)
    environment[PROOF] = "1"
    environment[OPT_IN] = "1"
    for name, value in extra.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def exact_device_profile(cuda_toolkit: str) -> bool:
    if cuda_toolkit.split(".")[:2] != ["13", "3"]:
        return False
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    first = completed.stdout.splitlines()[0] if completed.stdout.splitlines() else ""
    fields = [field.strip() for field in first.split(",")]
    return completed.returncode == 0 and fields == [
        "NVIDIA GeForce RTX 4090", "8.9", "610.43.03"
    ]


def write_pipeline(path: Path, input_path: Path, output_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "pipeline": [
                    {"type": "readers.las", "filename": str(input_path)},
                    {
                        "type": "filters.outlier",
                        "method": "statistical",
                        "mean_k": 8,
                        "multiplier": 2.0,
                    },
                    {
                        "type": "filters.range",
                        "limits": "Classification![7:7]",
                    },
                    {"type": "filters.sample", "radius": 1.0},
                    {
                        "type": "writers.las",
                        "filename": str(output_path),
                        "compression": "true",
                    },
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_selected_differential(args: argparse.Namespace, generated: Path) -> int:
    pipeline = generated / "selected.json"
    write_pipeline(pipeline, Path("input.laz"), Path("out.laz"))
    command = [
        sys.executable,
        str(args.differential.resolve()),
        "--oracle", str(args.oracle.resolve()),
        "--candidate", str(args.candidate.resolve()),
        "--case", "pdg_automatic_r4_outlier_selected",
        "--work-dir", str(args.work_dir.resolve()),
        "--frozen-time-library", str(args.frozen_time_library.resolve()),
        "--seed-file", f"input.laz={args.fixture.resolve()}",
        "--seed-file", f"pipeline.json={pipeline.resolve()}",
        "--", "pipeline", "pipeline.json",
    ]
    try:
        completed = subprocess.run(
            command,
            env=clean_environment(),
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("selected r4 exact differential timed out", file=sys.stderr)
        return 1
    return completed.returncode


def mutate_header(path: Path, offset: int, kind: str, value: int | float) -> None:
    with path.open("r+b") as output:
        output.seek(offset)
        output.write(struct.pack(kind, value))


def rejection_control(
    candidate: Path,
    directory: Path,
    input_path: Path,
    expected: bytes,
    **extra_environment: str,
) -> int:
    directory.mkdir()
    output = directory / "unexpected-out.laz"
    pipeline = directory / "pipeline.json"
    write_pipeline(pipeline, input_path, output)
    try:
        completed = subprocess.run(
            [str(candidate.resolve()), "pipeline", str(pipeline)],
            env=clean_environment(**extra_environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        print(f"{directory.name} timed out", file=sys.stderr)
        return 1
    failures: list[str] = []
    if completed.returncode != 1:
        failures.append(f"status {completed.returncode}, expected 1")
    if completed.stdout:
        failures.append(f"stdout was not empty: {completed.stdout!r}")
    if expected not in completed.stderr:
        failures.append(f"proof diagnostic absent: {completed.stderr!r}")
    if output.exists():
        failures.append("proof refusal created its output")
    if failures:
        print(f"{directory.name} failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    for required in (
        args.oracle,
        args.candidate,
        args.differential,
        args.frozen_time_library,
    ):
        if not required.is_file():
            print(f"required file does not exist: {required}", file=sys.stderr)
            return 2
    if not args.fixture.is_file():
        print(f"SKIP: local r4 fixture is absent: {args.fixture}")
        return 77
    if not exact_device_profile(args.cuda_toolkit):
        print("SKIP: B0227 requires RTX 4090/SM89/CUDA 13.3/driver 610.43.03")
        return 77

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pdg-r4-selector-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        result = run_selected_differential(args, generated)
        if result:
            return result

        wrong_count = generated / "wrong-count.laz"
        shutil.copy2(args.fixture, wrong_count)
        mutate_header(wrong_count, 247, "<Q", 999_999)
        if rejection_control(
            args.candidate, generated / "wrong-count", wrong_count,
            SELECTED_DIAGNOSTIC,
        ):
            return 1

        wrong_extent = generated / "wrong-extent.laz"
        shutil.copy2(args.fixture, wrong_extent)
        mutate_header(wrong_extent, 179, "<d", 186_000.0)
        if rejection_control(
            args.candidate, generated / "wrong-extent", wrong_extent,
            SELECTED_DIAGNOSTIC,
        ):
            return 1

        wrong_payload = generated / "wrong-payload.laz"
        shutil.copy2(args.fixture, wrong_payload)
        with wrong_payload.open("r+b") as output:
            output.seek(-1, os.SEEK_END)
            value = output.read(1)
            output.seek(-1, os.SEEK_END)
            output.write(bytes((value[0] ^ 0x01,)))
        if rejection_control(
            args.candidate, generated / "wrong-payload", wrong_payload,
            SELECTED_DIAGNOSTIC,
        ):
            return 1

        uppercase = generated / "uppercase.LAZ"
        shutil.copy2(args.fixture, uppercase)
        # Grammar refusal makes the range-only fact-free rewrite unstable and
        # returns before the inner selection guard.  The outer public proof
        # boundary therefore owns this deliberate "not used" diagnostic.
        if rejection_control(
            args.candidate, generated / "uppercase", uppercase,
            USED_DIAGNOSTIC,
        ):
            return 1

        if rejection_control(
            args.candidate, generated / "recoverable-fallback", args.fixture,
            USED_DIAGNOSTIC,
            **{INJECT_FALLBACK: "1"},
        ):
            return 1

        # D0272: without the experimental opt-in the literal 1M layout is
        # not selected; the exact host path runs instead.
        if rejection_control(
            args.candidate, generated / "retired-default", args.fixture,
            SELECTED_DIAGNOSTIC, **{OPT_IN: None},
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
