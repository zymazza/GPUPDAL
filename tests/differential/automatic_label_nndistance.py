#!/usr/bin/env python3
"""Prove B0233's exact automatic label/NNDistance hybrid composition."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from native_las_matrix import Case as LasCase
from native_las_matrix import write_las


MINIMUM_POINTS = 250_000
CONTROL_POINTS = 50_000
PROOF = "PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID"
INJECT_FAILURE = "PDG_TEST_LABEL_NNDISTANCE_RECOVERABLE_CUDA_FAILURE"
SELECTED_DIAGNOSTIC = (
    b"required automatic exact CUDA label/NNDistance hybrid path was not selected"
)
USED_DIAGNOSTIC = (
    b"required automatic exact CUDA label/NNDistance hybrid path was not used"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--cuda-toolkit", required=True)
    return parser.parse_args()


def clean_environment(
    *, proof: bool = False, overrides: dict[str, str] | None = None
) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("PDG_") and name != "PDG_DIFFERENTIAL_ASAN_PRELOAD":
            environment.pop(name)
    if proof:
        environment[PROOF] = "1"
    if overrides:
        environment.update(overrides)
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


def measured_pipeline(input_name: str, output_name: str) -> dict[str, object]:
    return {
        "pipeline": [
            {"type": "readers.las", "filename": input_name},
            {
                "type": "filters.label_duplicates",
                "dimensions": "Classification",
            },
            {"type": "filters.nndistance", "k": 10},
            {"type": "filters.assign", "value": "UserData = Duplicate"},
            {"type": "writers.las", "filename": output_name},
        ]
    }


def write_pipeline(path: Path, root: dict[str, object]) -> None:
    path.write_text(
        json.dumps(root, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def exact_differential(
    args: argparse.Namespace,
    generated: Path,
    label: str,
    fixture: Path,
    root: dict[str, object],
    environment: dict[str, str],
) -> int:
    pipeline = generated / f"{label}.json"
    write_pipeline(pipeline, root)
    command = [
        sys.executable,
        str(args.differential.resolve()),
        "--oracle", str(args.oracle.resolve()),
        "--candidate", str(args.candidate.resolve()),
        "--case", f"pdg_automatic_label_nndistance_{label}",
        "--work-dir", str(args.work_dir.resolve()),
        "--frozen-time-library", str(args.frozen_time_library.resolve()),
        "--seed-file", f"input.las={fixture.resolve()}",
        "--seed-file", f"pipeline.json={pipeline.resolve()}",
        "--", "pipeline", "pipeline.json",
    ]
    try:
        completed = subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"{label} exact differential timed out", file=sys.stderr)
        return 1
    return completed.returncode


def rejection_control(
    args: argparse.Namespace,
    directory: Path,
    fixture: Path,
    root: dict[str, object],
    expected: bytes,
    *,
    overrides: dict[str, str] | None = None,
    output_name: str = "out.las",
) -> int:
    directory.mkdir()
    output = directory / output_name
    pipeline = directory / "pipeline.json"
    rewritten = json.loads(json.dumps(root))
    rewritten["pipeline"][0]["filename"] = str(fixture.resolve())
    rewritten["pipeline"][-1]["filename"] = str(output.resolve())
    write_pipeline(pipeline, rewritten)
    try:
        completed = subprocess.run(
            [str(args.candidate.resolve()), "pipeline", str(pipeline.resolve())],
            env=clean_environment(proof=True, overrides=overrides),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"{directory.name} rejection timed out", file=sys.stderr)
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


def path_state(path: Path) -> tuple[object, ...]:
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if not path.exists():
        return ("absent",)
    if path.is_file():
        return ("file", path.read_bytes())
    return ("other", path.stat().st_mode)


def filesystem_differential(
    args: argparse.Namespace,
    generated: Path,
    label: str,
    root: dict[str, object],
    reset: Callable[[], None],
    observed: tuple[Path, ...],
) -> int:
    pipeline = generated / f"{label}.json"
    write_pipeline(pipeline, root)
    results: list[subprocess.CompletedProcess[bytes]] = []
    states: list[tuple[tuple[object, ...], ...]] = []
    for executable, environment in (
        (args.candidate, clean_environment()),
        (args.candidate, clean_environment(proof=True)),
        (args.oracle, clean_environment()),
    ):
        reset()
        try:
            completed = subprocess.run(
                [str(executable.resolve()), "pipeline", str(pipeline.resolve())],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            print(f"{label} filesystem differential timed out", file=sys.stderr)
            return 1
        results.append(completed)
        states.append(tuple(path_state(path) for path in observed))

    reference = results[-1]
    reference_state = states[-1]
    for lane, completed, state in zip(
        ("default", "required"), results[:2], states[:2]
    ):
        failures: list[str] = []
        if completed.returncode != reference.returncode:
            failures.append(
                f"{lane} status {completed.returncode}, oracle {reference.returncode}"
            )
        if completed.stdout != reference.stdout:
            failures.append(f"{lane} stdout differs from oracle")
        if completed.stderr != reference.stderr:
            failures.append(
                f"{lane} stderr differs: {completed.stderr!r} != {reference.stderr!r}"
            )
        if state != reference_state:
            failures.append(f"{lane} state {state!r}, oracle {reference_state!r}")
        if failures:
            print(f"{label} failed:", file=sys.stderr)
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
    if not exact_device_profile(args.cuda_toolkit):
        print("SKIP: B0233 requires RTX 4090/SM89/CUDA 13.3/driver 610.43.03")
        return 77

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pdg-automatic-label-nndistance-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        selected = generated / "selected-format7.las"
        below = generated / "below-format7.las"
        wrong_format = generated / "wrong-format6.las"
        write_las(selected, LasCase("label-nndistance-selected", 7, MINIMUM_POINTS))
        write_las(below, LasCase("label-nndistance-below", 7, CONTROL_POINTS))
        write_las(
            wrong_format,
            LasCase("label-nndistance-wrong-format", 6, MINIMUM_POINTS),
        )

        selected_bytes = selected.read_bytes()
        header_padding = generated / "header-padding-format7.las"
        padded_header = bytearray(selected_bytes[:375])
        struct.pack_into("<I", padded_header, 96, 376)
        header_padding.write_bytes(
            bytes(padded_header) + b"\0" + selected_bytes[375:]
        )

        with_vlr = generated / "one-vlr-format7.las"
        vlr_header = bytearray(54)
        vlr_header[2:10] = b"PDG_TEST"
        struct.pack_into("<H", vlr_header, 18, 1)
        vlr_description = b"B0233 VLR gate\0"
        vlr_header[22 : 22 + len(vlr_description)] = vlr_description
        vlr_las_header = bytearray(selected_bytes[:375])
        struct.pack_into("<I", vlr_las_header, 96, 429)
        struct.pack_into("<I", vlr_las_header, 100, 1)
        with_vlr.write_bytes(
            bytes(vlr_las_header) + bytes(vlr_header) + selected_bytes[375:]
        )

        with_evlr = generated / "one-evlr-format7.las"
        evlr_header = bytearray(60)
        evlr_header[2:10] = b"PDG_TEST"
        struct.pack_into("<H", evlr_header, 18, 1)
        evlr_description = b"B0233 EVLR gate\0"
        evlr_header[28 : 28 + len(evlr_description)] = evlr_description
        evlr_las = bytearray(selected_bytes)
        struct.pack_into("<Q", evlr_las, 235, len(selected_bytes))
        struct.pack_into("<I", evlr_las, 243, 1)
        with_evlr.write_bytes(bytes(evlr_las) + bytes(evlr_header))

        trailing = generated / "trailing-byte-format7.las"
        trailing.write_bytes(selected_bytes + b"\0")

        wider_record = generated / "record37-format7.las"
        write_las(
            wider_record,
            LasCase(
                "label-nndistance-record37",
                7,
                MINIMUM_POINTS,
                extra_bytes=1,
            ),
        )

        uppercase = generated / "uppercase-format7.LAS"
        uppercase.write_bytes(selected_bytes)

        measured = measured_pipeline("input.las", "out.las")
        for label, environment in (
            ("selected-default", clean_environment()),
            ("selected-proof", clean_environment(proof=True)),
        ):
            if exact_differential(
                args, generated, label, selected, measured, environment
            ):
                return 1

        for label, fixture, environment in (
            ("below-default", below, clean_environment()),
            ("wrong-format-default", wrong_format, clean_environment()),
            (
                "disabled-default",
                selected,
                clean_environment(overrides={"PDG_DISABLE_CUDA_HYBRID": "1"}),
            ),
            (
                "unavailable-default",
                selected,
                clean_environment(overrides={"CUDA_VISIBLE_DEVICES": ""}),
            ),
            (
                "recoverable-device-fallback",
                selected,
                clean_environment(overrides={INJECT_FAILURE: "1"}),
            ),
        ):
            if exact_differential(
                args, generated, label, fixture, measured, environment
            ):
                return 1

        for label, fixture, root in (
            ("header-padding-default", header_padding, measured),
            ("vlr-default", with_vlr, measured),
            ("evlr-default", with_evlr, measured),
            ("trailing-byte-default", trailing, measured),
            ("record37-default", wider_record, measured),
            (
                "uppercase-endpoint-default",
                uppercase,
                measured_pipeline(str(uppercase.resolve()), "out.las"),
            ),
            (
                "uppercase-writer-default",
                selected,
                measured_pipeline("input.las", "out.LAS"),
            ),
        ):
            if exact_differential(
                args, generated, label, fixture, root, clean_environment()
            ):
                return 1

        for label, fixture, overrides in (
            ("below-proof", below, None),
            ("wrong-format-proof", wrong_format, None),
            ("disabled-proof", selected, {"PDG_DISABLE_CUDA_HYBRID": "1"}),
            ("unavailable-proof", selected, {"CUDA_VISIBLE_DEVICES": ""}),
        ):
            if rejection_control(
                args,
                generated / label,
                fixture,
                measured,
                SELECTED_DIAGNOSTIC,
                overrides=overrides,
            ):
                return 1

        for label, fixture in (
            ("header-padding-proof", header_padding),
            ("vlr-proof", with_vlr),
            ("evlr-proof", with_evlr),
            ("trailing-byte-proof", trailing),
            ("record37-proof", wider_record),
            ("uppercase-endpoint-proof", uppercase),
        ):
            if rejection_control(
                args,
                generated / label,
                fixture,
                measured,
                SELECTED_DIAGNOSTIC,
            ):
                return 1

        if rejection_control(
            args,
            generated / "uppercase-writer-proof",
            selected,
            measured,
            SELECTED_DIAGNOSTIC,
            output_name="out.LAS",
        ):
            return 1

        if rejection_control(
            args,
            generated / "recoverable-proof",
            selected,
            measured,
            USED_DIAGNOSTIC,
            overrides={INJECT_FAILURE: "1"},
        ):
            return 1

        grammar_controls: list[tuple[str, dict[str, object]]] = []
        root_option = measured_pipeline("input.las", "out.las")
        root_option["metadata"] = "metadata.json"
        grammar_controls.append(("root-option", root_option))
        reader_option = measured_pipeline("input.las", "out.las")
        reader_option["pipeline"][0]["count"] = MINIMUM_POINTS
        grammar_controls.append(("reader-option", reader_option))
        missing_dimensions = measured_pipeline("input.las", "out.las")
        missing_dimensions["pipeline"][1].pop("dimensions")
        grammar_controls.append(("missing-dimensions", missing_dimensions))
        average = measured_pipeline("input.las", "out.las")
        average["pipeline"][2]["mode"] = "avg"
        grammar_controls.append(("average-mode", average))
        assignment_spacing = measured_pipeline("input.las", "out.las")
        assignment_spacing["pipeline"][3]["value"] = "UserData=Duplicate"
        grammar_controls.append(("assignment-spacing", assignment_spacing))
        stage_order = measured_pipeline("input.las", "out.las")
        stage_order["pipeline"][1], stage_order["pipeline"][2] = (
            stage_order["pipeline"][2],
            stage_order["pipeline"][1],
        )
        grammar_controls.append(("stage-order", stage_order))
        writer_option = measured_pipeline("input.las", "out.las")
        writer_option["pipeline"][4]["extra_dims"] = "all"
        grammar_controls.append(("writer-option", writer_option))

        for label, root in grammar_controls:
            if exact_differential(
                args,
                generated,
                f"{label}-default",
                selected,
                root,
                clean_environment(),
            ):
                return 1
            if rejection_control(
                args,
                generated / label,
                selected,
                root,
                SELECTED_DIAGNOSTIC,
            ):
                return 1

        existing = generated / "existing-output.las"
        existing_pipeline = measured_pipeline(
            str(selected.resolve()), str(existing.resolve())
        )

        def reset_existing() -> None:
            existing.write_bytes(b"sentinel")

        if filesystem_differential(
            args,
            generated,
            "existing-output",
            existing_pipeline,
            reset_existing,
            (existing,),
        ):
            return 1

        alias = generated / "alias-input-output.las"
        alias_pipeline = measured_pipeline(
            str(alias.resolve()), str(alias.resolve())
        )

        def reset_alias() -> None:
            alias.write_bytes(selected_bytes)

        if filesystem_differential(
            args,
            generated,
            "aliased-input-output",
            alias_pipeline,
            reset_alias,
            (alias,),
        ):
            return 1

        symlink_target = generated / "symlink-target.las"
        symlink_output = generated / "symlink-output.las"
        symlink_pipeline = measured_pipeline(
            str(selected.resolve()), str(symlink_output.absolute())
        )

        def reset_symlink() -> None:
            symlink_output.unlink(missing_ok=True)
            symlink_target.unlink(missing_ok=True)
            symlink_output.symlink_to(symlink_target)

        if filesystem_differential(
            args,
            generated,
            "symlink-output",
            symlink_pipeline,
            reset_symlink,
            (symlink_output, symlink_target),
        ):
            return 1

    print("automatic label/NNDistance hybrid matrix: exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
