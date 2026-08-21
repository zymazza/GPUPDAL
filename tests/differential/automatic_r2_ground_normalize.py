#!/usr/bin/env python3
"""Prove B0239's public r2 selector and selective exact HAG repair."""

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


PROOF = "PDG_REQUIRE_AUTOMATIC_R2_GROUND_NORMALIZE"
REPAIR_PROOF = "PDG_REQUIRE_HAG_NN_SELECTIVE_REPAIR"
NOT_SELECTED = b"required automatic exact r2 ground-normalization hybrid path was not selected"
WRONG_FINGERPRINT = b"required automatic exact r2 ground-normalization input fingerprint did not match"


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


def clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("PDG_") and name != "PDG_DIFFERENTIAL_ASAN_PRELOAD":
            environment.pop(name)
    environment[PROOF] = "1"
    environment[REPAIR_PROOF] = "1"
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


def write_pipeline(path: Path, input_path: Path, output_path: Path,
                   *, extra_dims: str = "HeightAboveGround=float32") -> None:
    path.write_text(
        json.dumps(
            {
                "pipeline": [
                    {"type": "readers.las", "filename": str(input_path)},
                    {"type": "filters.smrf"},
                    {"type": "filters.hag_nn"},
                    {
                        "type": "writers.las",
                        "filename": str(output_path),
                        "compression": "true",
                        "extra_dims": extra_dims,
                    },
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def selected_differential(args: argparse.Namespace, generated: Path) -> int:
    pipeline = generated / "selected.json"
    write_pipeline(pipeline, Path("input.laz"), Path("out.laz"))
    command = [
        sys.executable,
        str(args.differential.resolve()),
        "--oracle", str(args.oracle.resolve()),
        "--candidate", str(args.candidate.resolve()),
        "--case", "pdg_automatic_r2_ground_normalize_selected",
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
        print("selected r2 exact differential timed out", file=sys.stderr)
        return 1
    return completed.returncode


def mutate_header(path: Path, offset: int, kind: str, value: int | float) -> None:
    with path.open("r+b") as output:
        output.seek(offset)
        output.write(struct.pack(kind, value))


def rejection(candidate: Path, directory: Path, input_path: Path,
              expected: bytes, *, extra_dims: str = "HeightAboveGround=float32",
              environment: dict[str, str] | None = None) -> int:
    directory.mkdir()
    output = directory / "unexpected-out.laz"
    pipeline = directory / "pipeline.json"
    write_pipeline(pipeline, input_path, output, extra_dims=extra_dims)
    try:
        completed = subprocess.run(
            [str(candidate.resolve()), "pipeline", str(pipeline)],
            env=environment or clean_environment(),
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
        print(f"SKIP: local r2 fixture is absent: {args.fixture}")
        return 77
    if not exact_device_profile(args.cuda_toolkit):
        print("SKIP: B0239 requires RTX 4090/SM89/CUDA 13.3/driver 610.43.03")
        return 77

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pdg-r2-selector-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        result = selected_differential(args, generated)
        if result:
            return result

        wrong_count = generated / "wrong-count.laz"
        shutil.copy2(args.fixture, wrong_count)
        mutate_header(wrong_count, 247, "<Q", 999_999)
        if rejection(args.candidate, generated / "wrong-count", wrong_count,
                     NOT_SELECTED):
            return 1

        wrong_extent = generated / "wrong-extent.laz"
        shutil.copy2(args.fixture, wrong_extent)
        mutate_header(wrong_extent, 179, "<d", 186_000.0)
        if rejection(args.candidate, generated / "wrong-extent", wrong_extent,
                     NOT_SELECTED):
            return 1

        wrong_payload = generated / "wrong-payload.laz"
        shutil.copy2(args.fixture, wrong_payload)
        with wrong_payload.open("r+b") as output:
            output.seek(-1, os.SEEK_END)
            value = output.read(1)
            output.seek(-1, os.SEEK_END)
            output.write(bytes((value[0] ^ 0x01,)))
        if rejection(args.candidate, generated / "wrong-payload",
                     wrong_payload, WRONG_FINGERPRINT):
            return 1

        uppercase = generated / "uppercase.LAZ"
        shutil.copy2(args.fixture, uppercase)
        if rejection(args.candidate, generated / "uppercase", uppercase,
                     NOT_SELECTED):
            return 1

        external_marker = clean_environment()
        external_marker["PDG_INTERNAL_AUTOMATIC_R2_HYBRID"] = "1"
        if rejection(
            args.candidate, generated / "external-marker-layout", args.fixture,
            b"required automatic exact r2 ground-normalization stages were not selected",
            extra_dims="all", environment=external_marker
        ):
            return 1

        cuda_decline = clean_environment()
        cuda_decline["PDG_TEST_AUTOMATIC_R2_HAG_NN_DEVICE_DECLINE"] = "1"
        if rejection(
            args.candidate, generated / "cuda-decline", args.fixture,
            b"required HAG selective exact repair did not execute",
            environment=cuda_decline
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
