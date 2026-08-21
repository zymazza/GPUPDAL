#!/usr/bin/env python3
"""Prove the default approximatecoplanar selector at its exact count gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from native_las_matrix import Case as LasCase
from native_las_matrix import write_las

POINT_COUNT = 262_144
CONTROL_COUNT = POINT_COUNT - 1
PROOF_ENVIRONMENT = "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_CUDA"
PROOF_DIAGNOSTIC = (
    b"required automatic exact CUDA hybrid approximatecoplanar path "
    b"was not selected"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--oracle-preload", action="append", default=[],
                        type=Path)
    parser.add_argument("--candidate-preload", action="append", default=[],
                        type=Path)
    parser.add_argument(
        "--sanitizer-case", choices=("laz", "format6", "format8")
    )
    return parser.parse_args()


def clean_environment(
    *, proof: bool = False, overrides: dict[str, str] | None = None
) -> dict[str, str]:
    environment = os.environ.copy()
    # The selector regression must exercise option-free/default planning.
    # Preserve only differential sanitizer plumbing from an enclosing CTest
    # lane; differential.py supplies its own oracle and frozen-time variables.
    for name in tuple(environment):
        if name.startswith("PDG_") and name != "PDG_DIFFERENTIAL_ASAN_PRELOAD":
            environment.pop(name)
    if proof:
        environment[PROOF_ENVIRONMENT] = "1"
    if overrides:
        environment.update(overrides)
    return environment


def write_pipeline(path: Path, input_name: str, output_name: str,
                   count: int | None = None) -> None:
    reader: dict[str, object] = {
        "type": "readers.las",
        "filename": input_name,
    }
    if count is not None:
        reader["count"] = count
    path.write_text(
        json.dumps(
            {
                "pipeline": [
                    reader,
                    {
                        "type": "filters.approximatecoplanar",
                        "thresh1": 30.5,
                        "thresh2": 4.25,
                    },
                    {
                        "type": "writers.las",
                        "filename": output_name,
                        "extra_dims": "all",
                    },
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_rejection_control(candidate: Path, fixture: Path, directory: Path,
                          environment: dict[str, str], label: str,
                          count: int | None = None,
                          overrides: dict[str, str] | None = None) -> int:
    directory.mkdir()
    pipeline = directory / "pipeline.json"
    output = directory / "unexpected-out.las"
    metadata = directory / "unexpected-metadata.json"
    write_pipeline(
        pipeline,
        str(fixture.resolve()),
        output.name,
        count=count,
    )
    candidate_environment = environment.copy()
    if overrides:
        candidate_environment.update(overrides)
    try:
        completed = subprocess.run(
            [
                str(candidate.resolve()),
                "pipeline",
                pipeline.name,
                f"--metadata={metadata.name}",
            ],
            cwd=directory,
            env=candidate_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print(f"{label} automatic selector control timed out", file=sys.stderr)
        return 1

    failures: list[str] = []
    if completed.returncode != 1:
        failures.append(f"exit status was {completed.returncode}, expected 1")
    if completed.stdout:
        failures.append(f"stdout was not empty: {completed.stdout!r}")
    if PROOF_DIAGNOSTIC not in completed.stderr:
        failures.append(f"proof diagnostic was absent: {completed.stderr!r}")
    for artifact in (output, metadata):
        if artifact.exists():
            failures.append(f"proof rejection created {artifact.name}")
    if failures:
        print(f"{label} automatic selector control failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


def run_exact_differential(
    args: argparse.Namespace,
    case_name: str,
    pipeline: Path,
    environment: dict[str, str],
    fixture: Path | None = None,
    input_name: str | None = None,
) -> int:
    command = [
        sys.executable,
        str(args.differential.resolve()),
        "--oracle",
        str(args.oracle.resolve()),
        "--candidate",
        str(args.candidate.resolve()),
        "--case",
        f"pdg_automatic_approximatecoplanar_{case_name}",
        "--work-dir",
        str(args.work_dir.resolve()),
        "--frozen-time-library",
        str(args.frozen_time_library.resolve()),
    ]
    if fixture is not None:
        if input_name is None:
            raise ValueError("a seeded fixture requires an input filename")
        command.extend(("--seed-file", f"{input_name}={fixture.resolve()}"))
    command.extend(("--seed-file", f"pipeline.json={pipeline.resolve()}"))
    for preload in args.oracle_preload:
        command.extend(("--oracle-preload", str(preload.resolve())))
    for preload in args.candidate_preload:
        command.extend(("--candidate-preload", str(preload.resolve())))
    command.extend(
        (
            "--",
            "pipeline",
            "pipeline.json",
            "--metadata=metadata.json",
        )
    )
    try:
        completed = subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("automatic selector exact differential timed out",
              file=sys.stderr)
        return 1
    return completed.returncode


def run_pipeline_differential(
    args: argparse.Namespace,
    generated: Path,
    label: str,
    environment: dict[str, str],
    *,
    pipeline_input: str,
    fixture: Path | None = None,
    input_name: str | None = None,
    count: int | None = None,
) -> int:
    pipeline = generated / f"{label}.json"
    write_pipeline(pipeline, pipeline_input, "out.las", count=count)
    return run_exact_differential(
        args,
        label,
        pipeline,
        environment,
        fixture=fixture,
        input_name=input_name,
    )


def run_selected_case(
    args: argparse.Namespace,
    generated: Path,
    label: str,
    fixture: Path,
    input_name: str,
    count: int | None,
) -> int:
    pipeline = generated / f"selected-{label}.json"
    write_pipeline(pipeline, input_name, "out.las", count=count)
    for suffix, environment in (
        ("option-free", clean_environment()),
        ("proof", clean_environment(proof=True)),
    ):
        result = run_exact_differential(
            args,
            f"selected-{label}-{suffix}",
            pipeline,
            environment,
            fixture=fixture,
            input_name=input_name,
        )
        if result:
            return result
    return 0


def run_sanitizer_case(
    candidate: Path,
    generated: Path,
    label: str,
    fixture: Path,
) -> int:
    case_directory = generated / f"sanitizer-{label}"
    case_directory.mkdir()
    pipeline = case_directory / "pipeline.json"
    output = case_directory / "out.las"
    write_pipeline(pipeline, str(fixture.resolve()), output.name)
    try:
        completed = subprocess.run(
            [str(candidate.resolve()), "pipeline", pipeline.name],
            cwd=case_directory,
            env=clean_environment(proof=True),
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"{label} automatic sanitizer case timed out", file=sys.stderr)
        return 1
    if completed.returncode != 0 or not output.is_file():
        print(
            f"{label} automatic sanitizer case failed with status "
            f"{completed.returncode}",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    args = parse_args()
    for required in (
        args.oracle,
        args.candidate,
        args.differential,
        args.frozen_time_library,
        *args.oracle_preload,
        *args.candidate_preload,
    ):
        if not required.is_file():
            print(f"required file does not exist: {required}", file=sys.stderr)
            return 2

    if not args.fixture_root.is_dir():
        print(f"fixture root does not exist: {args.fixture_root}",
              file=sys.stderr)
        return 2

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pdg-automatic-approximatecoplanar-", dir=args.work_dir
    ) as temporary:
        generated = Path(temporary)
        fixture_root = args.fixture_root.resolve()
        laz_fixture = fixture_root / "ept/source/lone-star.laz"
        if not laz_fixture.is_file():
            print(f"required fixture does not exist: {laz_fixture}",
                  file=sys.stderr)
            return 2

        format6_fixture = generated / "format6.las"
        format8_fixture = generated / "format8.las"

        if args.sanitizer_case:
            sanitizer_fixture = laz_fixture
            if args.sanitizer_case == "format6":
                write_las(
                    format6_fixture,
                    LasCase("automatic-format6", 6, POINT_COUNT),
                )
                sanitizer_fixture = format6_fixture
            elif args.sanitizer_case == "format8":
                write_las(
                    format8_fixture,
                    LasCase("automatic-format8", 8, POINT_COUNT),
                )
                sanitizer_fixture = format8_fixture
            return run_sanitizer_case(
                args.candidate,
                generated,
                args.sanitizer_case,
                sanitizer_fixture,
            )

        write_las(
            format6_fixture, LasCase("automatic-format6", 6, POINT_COUNT)
        )
        write_las(
            format8_fixture, LasCase("automatic-format8", 8, POINT_COUNT)
        )

        proof = clean_environment(proof=True)

        if run_rejection_control(
            args.candidate,
            laz_fixture,
            generated / "below-threshold-control",
            proof,
            "count-capped",
            count=CONTROL_COUNT,
        ):
            return 1

        nonregular = generated / "not-regular.las"
        nonregular.mkdir()
        if run_rejection_control(
            args.candidate,
            nonregular,
            generated / "nonregular-control",
            proof,
            "non-regular input",
        ):
            return 1

        if run_rejection_control(
            args.candidate,
            laz_fixture,
            generated / "disabled-control",
            proof,
            "disabled CUDA",
            count=POINT_COUNT,
            overrides={"PDG_DISABLE_CUDA_HYBRID": "1"},
        ):
            return 1

        if run_rejection_control(
            args.candidate,
            laz_fixture,
            generated / "unavailable-control",
            proof,
            "unavailable CUDA",
            count=POINT_COUNT,
            overrides={"CUDA_VISIBLE_DEVICES": ""},
        ):
            return 1

        # The first three controls must delegate before candidate data/output
        # side effects. Only the injected recoverable device failure reaches
        # the internal exact host wrapper, whose proof guard is set below.
        fallback_cases = (
            (
                "below-threshold-fallback",
                clean_environment(),
                "input.laz",
                laz_fixture,
                CONTROL_COUNT,
            ),
            (
                "disabled-fallback",
                clean_environment(
                    overrides={"PDG_DISABLE_CUDA_HYBRID": "1"}
                ),
                "input.laz",
                laz_fixture,
                POINT_COUNT,
            ),
            (
                "unavailable-fallback",
                clean_environment(overrides={"CUDA_VISIBLE_DEVICES": ""}),
                "input.laz",
                laz_fixture,
                POINT_COUNT,
            ),
            (
                "recoverable-device-fallback",
                clean_environment(
                    overrides={
                        "PDG_TEST_APPROXIMATECOPLANAR_RECOVERABLE_CUDA_FAILURE":
                            "1",
                        "PDG_REQUIRE_APPROXIMATECOPLANAR_HOST_FALLBACK": "1",
                    }
                ),
                "input.laz",
                laz_fixture,
                POINT_COUNT,
            ),
        )
        for label, environment, input_name, fixture, count in fallback_cases:
            result = run_pipeline_differential(
                args,
                generated,
                label,
                environment,
                pipeline_input=input_name,
                fixture=fixture,
                input_name=input_name,
                count=count,
            )
            if result:
                return result

        result = run_pipeline_differential(
            args,
            generated,
            "nonregular-fallback",
            clean_environment(),
            pipeline_input=str(nonregular.resolve()),
        )
        if result:
            return result

        selected_cases = (
            ("laz", laz_fixture, "input.laz", POINT_COUNT),
            ("format6-las", format6_fixture, "input.las", None),
            ("format8-las", format8_fixture, "input.las", None),
        )
        for label, fixture, input_name, count in selected_cases:
            result = run_selected_case(
                args, generated, label, fixture, input_name, count
            )
            if result:
                return result

    print(
        "exact automatic approximatecoplanar selector: three selected formats, "
        f"count={CONTROL_COUNT} and file/runtime controls rejected, "
        "recoverable CUDA failure used the exact host wrapper"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
