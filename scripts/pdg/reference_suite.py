#!/usr/bin/env python3
"""Validate, execute, and aggregate the fixed reference workload suite.

The suite is deliberately a closed contract: a partial report must not turn
into a more flattering aggregate. The stdlib-only runner authenticates named
fixtures and serially delegates complete-process measurements to
``benchmark_reference.py``; it never implements a benchmark stage itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_IDS = (
    "r1-translate", "r2-ground-normalize", "r3-dtm", "r4-denoise-thin",
    "r5-copc-query", "r6-features", "r7-dsm", "r8-colorize",
    "r9-polygon-clip", "r10-decimate", "r11-classify-refine", "r12-tile",
    "r13-merge", "r14-convert-compress",
)
MANIFEST_SCHEMA = "pdg-reference-suite-manifest-v1"


class SuiteError(ValueError):
    """A malformed manifest or incomplete aggregate input."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate the suite manifest")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--repo-root", required=True, type=Path)

    aggregate = commands.add_parser("aggregate", help="combine complete reports")
    aggregate.add_argument("--manifest", required=True, type=Path)
    aggregate.add_argument("--reports", required=True, type=Path)
    aggregate.add_argument("--cache-state", required=True, choices=("cold", "warm"))
    aggregate.add_argument("--output", required=True, type=Path)
    aggregate.add_argument("--contract", choices=("exact", "fast"),
                           default="exact",
                           help="contract the reports were measured under; "
                                "recorded in the aggregate")

    run = commands.add_parser(
        "run", help="execute every headline workload then write its aggregate")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--repo-root", required=True, type=Path)
    run.add_argument("--benchmark-runner", required=True, type=Path)
    run.add_argument("--oracle", required=True, type=Path)
    run.add_argument("--candidate", required=True, type=Path)
    run.add_argument("--work-dir", required=True, type=Path)
    run.add_argument("--reports", required=True, type=Path)
    run.add_argument("--aggregate-output", required=True, type=Path)
    run.add_argument("--cache-state", required=True, choices=("cold", "warm"))
    run.add_argument("--runs", type=int, default=3)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--frozen-time-library", type=Path)
    run.add_argument("--freeze-epoch", type=int, default=1704067200)
    run.add_argument("--environment", action="append", default=[])
    run.add_argument("--candidate-env", action="append", default=[])
    run.add_argument("--candidate-arg", action="append", default=[],
                     help="argument inserted before the candidate's command "
                          "(for example --fast)")
    run.add_argument("--contract", choices=("exact", "fast"),
                     default="exact",
                     help="comparison contract forwarded to the runner")
    run.add_argument(
        "--include-variants", action="store_true",
        help="also execute zero-weight supporting variants; aggregate remains headline-only",
    )
    run.add_argument(
        "--variants-only", action="store_true",
        help="execute only zero-weight supporting variants and skip the headline aggregate",
    )
    run.add_argument(
        "--workload", action="append", default=[], metavar="ID",
        help="run only the named headline/variant ID; repeatable, no partial aggregate",
    )
    return parser.parse_args()


def load_json(path: Path, description: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise SuiteError(f"cannot read {description} {path}: {error}") from error


def require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SuiteError(f"{name} must be a non-empty string")
    return value


def relative_repo_file(repo_root: Path, text: str, name: str) -> Path:
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SuiteError(f"{name} must be a repository-relative path")
    resolved_root = repo_root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved_root not in resolved.parents and resolved != resolved_root:
        raise SuiteError(f"{name} escapes --repo-root")
    if not resolved.is_file():
        raise SuiteError(f"{name} does not name a checked-in file: {text}")
    return resolved


def require_nonempty_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise SuiteError(f"{name} must be a non-empty list")
    return value


def validate_fixture_policy(workload: dict[str, Any], label: str,
                            fixtures: dict[str, Any]) -> None:
    # A fixture may be public or locally registered, but every workload must
    # declare the policy, provenance and concrete registry entries it uses.
    # This prevents a suite headline from silently acquiring an untracked local
    # corpus dependency.
    policy = workload.get("fixture_policy")
    if not isinstance(policy, dict):
        raise SuiteError(f"workload {label} fixture_policy must be an object")
    require_string(policy.get("kind"), f"workload {label} fixture_policy.kind")
    require_string(policy.get("provenance"),
                   f"workload {label} fixture_policy.provenance")
    fixture_ids = require_nonempty_list(policy.get("fixture_ids"),
                                        f"workload {label} fixture_policy.fixture_ids")
    for fixture_id in fixture_ids:
        fixture_id = require_string(fixture_id,
                                    f"workload {label} fixture_policy fixture ID")
        if fixture_id not in fixtures:
            raise SuiteError(
                f"workload {label} references missing fixture ID {fixture_id!r}")


def validate_variant(variant: Any, label: str, repo_root: Path | None,
                     fixtures: dict[str, Any]) -> None:
    if not isinstance(variant, dict):
        raise SuiteError(f"workload {label} variant must be an object")
    variant_id = require_string(variant.get("id"), f"workload {label} variant id")
    if variant.get("headline") is not False:
        raise SuiteError(f"workload {label} variant {variant_id} must set headline=false")
    weight = variant.get("headline_weight")
    if (not isinstance(weight, (int, float)) or isinstance(weight, bool) or
            not math.isfinite(float(weight)) or float(weight) != 0.0):
        raise SuiteError(
            f"workload {label} variant {variant_id} must have headline_weight 0")
    pipeline = require_string(variant.get("pipeline"),
                              f"workload {label} variant {variant_id} pipeline")
    if Path(pipeline).is_absolute() or ".." in Path(pipeline).parts:
        raise SuiteError(f"workload {label} variant {variant_id} pipeline must be repository-relative")
    if repo_root is not None:
        relative_repo_file(repo_root, pipeline,
                           f"workload {label} variant {variant_id} pipeline")
    materialization = variant.get("materialization")
    if materialization is not None:
        if not isinstance(materialization, dict):
            raise SuiteError(
                f"workload {label} variant {variant_id} materialization must be an object")
        inputs = materialization.get("inputs")
        if not isinstance(inputs, dict) or not inputs:
            raise SuiteError(
                f"workload {label} variant {variant_id} materialization.inputs must be an object")
        for placeholder, fixture_id in inputs.items():
            require_string(placeholder,
                           f"workload {label} variant {variant_id} input placeholder")
            fixture_id = require_string(
                fixture_id, f"workload {label} variant {variant_id} fixture ID")
            if fixture_id not in fixtures:
                raise SuiteError(
                    f"workload {label} variant {variant_id} references missing fixture {fixture_id!r}")


def load_manifest(path: Path, repo_root: Path | None = None) -> list[dict[str, Any]]:
    manifest = load_json(path, "manifest")
    if not isinstance(manifest, dict):
        raise SuiteError("manifest root must be an object")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SuiteError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    aggregate_policy = manifest.get("aggregate_policy")
    if not isinstance(aggregate_policy, dict) or not aggregate_policy:
        raise SuiteError("manifest aggregate_policy must be a non-empty object")
    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, dict) or not fixtures:
        raise SuiteError("manifest fixtures must be a non-empty object registry")
    for fixture_id, fixture in fixtures.items():
        require_string(fixture_id, "fixture registry ID")
        if not isinstance(fixture, dict) or not fixture:
            raise SuiteError(f"fixture registry entry {fixture_id!r} must be a non-empty object")
    workloads = manifest.get("workloads")
    if not isinstance(workloads, list) or len(workloads) != len(EXPECTED_IDS):
        raise SuiteError("manifest must contain exactly fourteen workloads")

    ids: list[str] = []
    weight = 0.0
    for expected, workload in zip(EXPECTED_IDS, workloads):
        if not isinstance(workload, dict):
            raise SuiteError("each workload must be an object")
        identifier = require_string(workload.get("id"), "workload id")
        ids.append(identifier)
        if identifier != expected:
            raise SuiteError(
                "manifest IDs must be the canonical fourteen in order; "
                f"expected {expected!r}, found {identifier!r}")
        if workload.get("headline") is not True:
            raise SuiteError(f"workload {identifier} must set headline=true")
        headline_weight = workload.get("headline_weight")
        if (not isinstance(headline_weight, (int, float)) or
                isinstance(headline_weight, bool) or
                not math.isfinite(float(headline_weight)) or
                float(headline_weight) != 1.0):
            raise SuiteError(
                f"workload {identifier} must have headline_weight exactly 1")
        weight += float(headline_weight)
        pipeline = require_string(workload.get("pipeline"),
                                  f"workload {identifier} pipeline")
        if Path(pipeline).is_absolute() or ".." in Path(pipeline).parts:
            raise SuiteError(f"workload {identifier} pipeline must be repository-relative")
        if repo_root is not None:
            relative_repo_file(repo_root, pipeline,
                               f"workload {identifier} pipeline")
        validate_fixture_policy(workload, identifier, fixtures)
        cache_states = workload.get("cache_states")
        if cache_states != ["cold", "warm"]:
            raise SuiteError(f"workload {identifier} cache_states must be exactly ['cold', 'warm']")
        require_string(workload.get("representative_reason"),
                       f"workload {identifier} representative_reason")
        bottlenecks = require_nonempty_list(workload.get("expected_bottlenecks"),
                                            f"workload {identifier} expected_bottlenecks")
        if not all(isinstance(item, str) and item for item in bottlenecks):
            raise SuiteError(f"workload {identifier} expected_bottlenecks must contain strings")
        for field in ("materialization", "oracle_contract", "tests"):
            value = workload.get(field)
            if not isinstance(value, dict) or not value:
                raise SuiteError(f"workload {identifier} {field} must be a non-empty object")
        artifacts = require_nonempty_list(workload.get("artifacts"),
                                          f"workload {identifier} artifacts")
        if not all(isinstance(item, (str, dict)) and item for item in artifacts):
            raise SuiteError(f"workload {identifier} artifacts contains an invalid entry")
        for artifact in artifacts:
            if not isinstance(artifact, dict) or "expected_point_count" not in artifact:
                continue
            require_string(artifact.get("pattern"),
                           f"workload {identifier} artifact pattern")
            point_count = artifact["expected_point_count"]
            if (not isinstance(point_count, int) or isinstance(point_count, bool) or
                    point_count <= 0):
                raise SuiteError(
                    f"workload {identifier} artifact expected_point_count "
                    "must be a positive integer")
        profile = require_nonempty_list(workload.get("profile_attribution"),
                                        f"workload {identifier} profile_attribution")
        if not all(isinstance(item, str) and item for item in profile):
            raise SuiteError(f"workload {identifier} profile_attribution must contain strings")
        variants = workload.get("variants", [])
        if not isinstance(variants, list):
            raise SuiteError(f"workload {identifier} variants must be a list")
        for variant in variants:
            validate_variant(variant, identifier, repo_root, fixtures)

    if tuple(ids) != EXPECTED_IDS or weight != float(len(EXPECTED_IDS)):
        raise SuiteError("manifest headline membership or aggregate weight is invalid")
    return workloads


def validate_command(args: argparse.Namespace) -> int:
    workloads = load_manifest(args.manifest, args.repo_root)
    report = {
        "schema": "pdg-reference-suite-validation-v1",
        "headline_workloads": len(workloads),
        "aggregate_weight": sum(float(w["headline_weight"]) for w in workloads),
        "ids": [w["id"] for w in workloads],
    }
    print(json.dumps(report, sort_keys=True))
    return 0


def positive_number(value: Any, name: str) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or
            not math.isfinite(float(value)) or float(value) <= 0.0):
        raise SuiteError(f"{name} must be a finite positive number")
    return float(value)


def measured_pairs(report: dict[str, Any], identifier: str) -> dict[int, float]:
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise SuiteError(f"{identifier}: runs must be a list")
    by_iteration: dict[int, dict[str, float]] = {}
    for record in runs:
        if not isinstance(record, dict) or record.get("phase") != "measured":
            continue
        iteration = record.get("iteration")
        label = record.get("label")
        if (not isinstance(iteration, int) or isinstance(iteration, bool) or
                label not in ("oracle", "candidate")):
            raise SuiteError(f"{identifier}: measured runs need integer iteration and role")
        values = by_iteration.setdefault(iteration, {})
        if label in values:
            raise SuiteError(f"{identifier}: duplicate {label} measurement for iteration {iteration}")
        values[label] = positive_number(record.get("seconds"),
                                        f"{identifier} {label} seconds")
    if not by_iteration:
        raise SuiteError(f"{identifier}: no measured pairs")
    ratios: dict[int, float] = {}
    for iteration, values in by_iteration.items():
        if set(values) != {"oracle", "candidate"}:
            raise SuiteError(f"{identifier}: unpaired measurement at iteration {iteration}")
        ratios[iteration] = values["oracle"] / values["candidate"]
    return ratios


def validate_report_artifact_contract(
        workload: dict[str, Any], report: dict[str, Any], identifier: str) -> None:
    expected = {
        artifact["pattern"]: artifact["expected_point_count"]
        for artifact in workload["artifacts"]
        if isinstance(artifact, dict) and "expected_point_count" in artifact
    }
    if not expected:
        return
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise SuiteError(f"{identifier}: runs must be a list")
    observed_roles: set[str] = set()
    for record in runs:
        if not isinstance(record, dict) or record.get("phase") != "measured":
            continue
        role = record.get("label")
        if role not in ("oracle", "candidate"):
            raise SuiteError(f"{identifier}: measured artifact has invalid role")
        observed_roles.add(role)
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, list):
            raise SuiteError(f"{identifier}: measured run needs artifact records")
        by_name = {
            artifact.get("name"): artifact for artifact in artifacts
            if isinstance(artifact, dict) and isinstance(artifact.get("name"), str)
        }
        for name, point_count in expected.items():
            artifact = by_name.get(name)
            if artifact is None:
                raise SuiteError(f"{identifier}: missing expected artifact {name}")
            if artifact.get("las_point_count") != point_count:
                raise SuiteError(
                    f"{identifier}: {role} {name} point count must be "
                    f"{point_count}, found {artifact.get('las_point_count')!r}")
    if observed_roles != {"oracle", "candidate"}:
        raise SuiteError(f"{identifier}: point-count contract needs both measured roles")


def incomplete_report(cache_state: str, missing: list[str], inexact: list[str],
                      invalid: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": "pdg-reference-suite-aggregate-v1",
        "cache_state": cache_state,
        "complete": False,
        "missing": missing,
        "inexact": inexact,
        "invalid": invalid,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary_name = stream.name
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def aggregate_command(args: argparse.Namespace) -> int:
    try:
        workloads = load_manifest(args.manifest)
    except SuiteError as error:
        result = incomplete_report(args.cache_state, list(EXPECTED_IDS), [],
                                   {"manifest": str(error)})
        atomic_write_json(args.output, result)
        print(f"reference suite aggregate: {error}", file=sys.stderr)
        return 2

    missing: list[str] = []
    inexact: list[str] = []
    invalid: dict[str, str] = {}
    reports: dict[str, dict[str, Any]] = {}
    for workload in workloads:
        identifier = workload["id"]
        path = args.reports / f"{identifier}-{args.cache_state}.json"
        if not path.is_file():
            missing.append(identifier)
            continue
        try:
            report = load_json(path, f"report for {identifier}")
            if not isinstance(report, dict):
                raise SuiteError("report root must be an object")
            if report.get("schema") != "pdg-reference-pipeline-baseline-v1":
                raise SuiteError(f"{identifier}: unsupported report schema")
            if report.get("label") != identifier:
                raise SuiteError(
                    f"{identifier}: report label must be {identifier!r}")
            environment = report.get("environment")
            if not isinstance(environment, dict):
                raise SuiteError(f"{identifier}: environment must be an object")
            if environment.get("cache_state") != args.cache_state:
                raise SuiteError(
                    f"{identifier}: report cache_state must be "
                    f"{args.cache_state!r}")
            comparison = report.get("comparison")
            if not isinstance(comparison, dict) or comparison.get("exact_outputs") is not True:
                inexact.append(identifier)
                continue
            if comparison.get("contract", "exact") != args.contract:
                raise SuiteError(
                    f"{identifier}: report contract "
                    f"{comparison.get('contract', 'exact')!r} does not match "
                    f"the aggregate contract {args.contract!r}")
            validate_report_artifact_contract(workload, report, identifier)
            reports[identifier] = report
        except SuiteError as error:
            invalid[identifier] = str(error)

    if missing or inexact or invalid:
        result = incomplete_report(args.cache_state, missing, inexact, invalid)
        atomic_write_json(args.output, result)
        print("reference suite aggregate is incomplete: " +
              "; ".join(filter(None, [
                  f"missing={','.join(missing)}" if missing else "",
                  f"inexact={','.join(inexact)}" if inexact else "",
                  f"invalid={','.join(sorted(invalid))}" if invalid else "",
              ])), file=sys.stderr)
        return 1

    try:
        rows: list[dict[str, Any]] = []
        binary_hashes: dict[str, set[str]] = {"oracle": set(), "candidate": set()}
        all_iterations: set[int] | None = None
        total_oracle = 0.0
        total_candidate = 0.0
        per_workload_pairs: dict[str, dict[int, float]] = {}
        for workload in workloads:
            identifier = workload["id"]
            report = reports[identifier]
            binaries = report.get("binaries")
            if not isinstance(binaries, dict):
                raise SuiteError(f"{identifier}: binaries must be an object")
            for role in ("oracle", "candidate"):
                binary = binaries.get(role)
                if not isinstance(binary, dict):
                    raise SuiteError(f"{identifier}: missing {role} binary identity")
                sha = require_string(binary.get("sha256"),
                                     f"{identifier} {role} binary sha256")
                if (len(sha) != 64 or
                        any(character not in "0123456789abcdef" for character in sha)):
                    raise SuiteError(
                        f"{identifier}: {role} binary sha256 must be lowercase hexadecimal")
                binary_hashes[role].add(sha)
            summary = report.get("summary")
            if not isinstance(summary, dict):
                raise SuiteError(f"{identifier}: summary must be an object")
            oracle = summary.get("oracle")
            candidate = summary.get("candidate")
            if not isinstance(oracle, dict) or not isinstance(candidate, dict):
                raise SuiteError(f"{identifier}: summary needs oracle and candidate")
            oracle_median = positive_number(oracle.get("median_seconds"),
                                            f"{identifier} oracle median")
            candidate_median = positive_number(candidate.get("median_seconds"),
                                               f"{identifier} candidate median")
            ratios = measured_pairs(report, identifier)
            iterations = set(ratios)
            if all_iterations is None:
                all_iterations = iterations
            elif all_iterations != iterations:
                raise SuiteError(f"{identifier}: measured iterations do not align across suite")
            per_workload_pairs[identifier] = ratios
            total_oracle += oracle_median
            total_candidate += candidate_median
            comparison = report.get("comparison") or {}
            row = {
                "id": identifier,
                "oracle_median_seconds": oracle_median,
                "candidate_median_seconds": candidate_median,
                "median_speedup": oracle_median / candidate_median,
            }
            if comparison.get("contract") == "fast":
                # D0271: the fast contract may leave a bounded number of
                # tie-order records different; the aggregate carries the
                # count so it is never mistaken for a byte-exact claim.
                row["contract"] = "fast"
                row["identical_records"] = comparison.get("identical_records")
                row["fast_differing_records"] = comparison.get(
                    "fast_differing_records")
                row["fast_compared_records"] = comparison.get(
                    "fast_compared_records")
            rows.append(row)
        for role, hashes in binary_hashes.items():
            if len(hashes) != 1:
                raise SuiteError(
                    f"suite must use one {role} binary hash; found {sorted(hashes)}")
        assert all_iterations is not None
        paired_suite_rounds = []
        for iteration in sorted(all_iterations):
            # Reconstruct each suite round from the report seconds, rather than
            # averaging workload ratios: equal-workload and total-wall metrics
            # intentionally answer different questions.
            oracle_seconds = 0.0
            candidate_seconds = 0.0
            for identifier, report in reports.items():
                found: dict[str, float] = {}
                for record in report["runs"]:
                    if (isinstance(record, dict) and record.get("phase") == "measured" and
                            record.get("iteration") == iteration and
                            record.get("label") in ("oracle", "candidate")):
                        found[record["label"]] = positive_number(
                            record.get("seconds"), f"{identifier} paired seconds")
                if set(found) != {"oracle", "candidate"}:
                    raise SuiteError(f"{identifier}: unpaired suite round {iteration}")
                oracle_seconds += found["oracle"]
                candidate_seconds += found["candidate"]
            paired_suite_rounds.append({
                "iteration": iteration,
                "oracle_seconds": oracle_seconds,
                "candidate_seconds": candidate_seconds,
                "speedup": oracle_seconds / candidate_seconds,
            })

        speedups = [row["median_speedup"] for row in rows]
        result = {
            "schema": "pdg-reference-suite-aggregate-v1",
            "cache_state": args.cache_state,
            "contract": args.contract,
            "complete": True,
            "workload_count": len(rows),
            "binaries": {
                role: {"sha256": next(iter(hashes))}
                for role, hashes in binary_hashes.items()
            },
            "ids": [row["id"] for row in rows],
            "workloads": rows,
            "equal_workload_geometric_mean_speedup": math.exp(
                statistics.fmean(math.log(value) for value in speedups)),
            "total_wall_time": {
                "oracle_median_seconds": total_oracle,
                "candidate_median_seconds": total_candidate,
                "speedup": total_oracle / total_candidate,
            },
            "paired_suite_round_speedups": [
                row["speedup"] for row in paired_suite_rounds
            ],
            "paired_suite_rounds": paired_suite_rounds,
            "missing": [],
            "inexact": [],
            "invalid": {},
        }
    except SuiteError as error:
        result = incomplete_report(args.cache_state, [], [], {"aggregate": str(error)})
        atomic_write_json(args.output, result)
        print(f"reference suite aggregate: {error}", file=sys.stderr)
        return 1

    atomic_write_json(args.output, result)
    return 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_fixture_paths(manifest: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    fixtures = manifest["fixtures"]
    resolved: dict[str, Path] = {}
    for identifier, fixture in fixtures.items():
        path_text = require_string(
            fixture.get("path"), f"fixture {identifier} path")
        path = relative_repo_file(repo_root, path_text, f"fixture {identifier} path")
        expected_bytes = fixture.get("bytes")
        if (not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or
                expected_bytes < 0):
            raise SuiteError(f"fixture {identifier} bytes must be a nonnegative integer")
        if path.stat().st_size != expected_bytes:
            raise SuiteError(
                f"fixture {identifier} byte size changed: expected {expected_bytes}, "
                f"found {path.stat().st_size}")
        expected_sha = require_string(
            fixture.get("sha256"), f"fixture {identifier} sha256")
        if (len(expected_sha) != 64 or
                any(character not in "0123456789abcdef" for character in expected_sha)):
            raise SuiteError(f"fixture {identifier} sha256 must be lowercase hexadecimal")
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            raise SuiteError(
                f"fixture {identifier} hash changed: expected {expected_sha}, "
                f"found {actual_sha}")
        resolved[identifier] = path
    return resolved


def checked_program(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SuiteError(f"{description} is not a file: {resolved}")
    return resolved


def run_command(args: argparse.Namespace) -> int:
    if args.runs < 1 or args.warmups < 0:
        raise SuiteError("--runs must be positive and --warmups must be nonnegative")
    if args.include_variants and args.variants_only:
        raise SuiteError("--include-variants and --variants-only are mutually exclusive")
    repo_root = args.repo_root.resolve()
    workloads = load_manifest(args.manifest, repo_root)
    manifest = load_json(args.manifest, "manifest")
    fixture_paths = verified_fixture_paths(manifest, repo_root)
    primary_laz = fixture_paths.get("primary_laz")
    if primary_laz is None:
        raise SuiteError("suite run requires the primary_laz fixture")
    runner = checked_program(args.benchmark_runner, "benchmark runner")
    oracle = checked_program(args.oracle, "oracle")
    candidate = checked_program(args.candidate, "candidate")
    frozen = (checked_program(args.frozen_time_library, "frozen-time library")
              if args.frozen_time_library else None)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if not args.variants_only:
        tasks.extend((workload, workload) for workload in workloads)
    if args.include_variants or args.variants_only:
        for workload in workloads:
            tasks.extend((workload, variant)
                         for variant in workload.get("variants", []))
    if args.workload:
        requested = set(args.workload)
        known = {task["id"] for _, task in tasks}
        unknown = sorted(requested - known)
        if unknown:
            raise SuiteError("unknown or excluded workload IDs: " + ", ".join(unknown))
        tasks = [(workload, task) for workload, task in tasks
                 if task["id"] in requested]

    failures: list[str] = []
    for workload, task in tasks:
        identifier = task["id"]
        materialization = task.get("materialization", workload["materialization"])
        inputs = materialization.get("inputs")
        if not isinstance(inputs, dict) or not inputs:
            raise SuiteError(f"workload {identifier} materialization.inputs must be an object")
        command = [
            sys.executable, str(runner),
            "--oracle", str(oracle), "--candidate", str(candidate),
            "--pipeline", str((repo_root / task["pipeline"]).resolve()),
            "--fixture-laz", str(primary_laz),
            "--label", identifier,
            "--work-dir", str(args.work_dir / identifier / args.cache_state),
            "--report", str(args.reports / f"{identifier}-{args.cache_state}.json"),
            "--runs", str(args.runs), "--warmups", str(args.warmups),
            "--cache-state", args.cache_state,
            "--repo-root", str(repo_root),
        ]
        for placeholder, fixture_id in sorted(inputs.items()):
            require_string(placeholder, f"workload {identifier} input placeholder")
            fixture_id = require_string(
                fixture_id, f"workload {identifier} input fixture ID")
            if fixture_id not in fixture_paths:
                raise SuiteError(
                    f"workload {identifier} input references missing fixture {fixture_id!r}")
            command.extend([
                "--fixture", f"{placeholder}={fixture_paths[fixture_id]}"])
            if placeholder == "input.copc.laz":
                command.extend(["--fixture-copc", str(fixture_paths[fixture_id])])
        for value in args.environment:
            command.extend(["--environment", value])
        for value in args.candidate_env:
            command.extend(["--candidate-env", value])
        for value in args.candidate_arg:
            # `=` form so a leading-dash value such as --fast is not parsed
            # as an option of the runner.
            command.append(f"--candidate-arg={value}")
        if args.contract != "exact":
            command.extend(["--contract", args.contract])
        if frozen:
            command.extend([
                "--frozen-time-library", str(frozen),
                "--freeze-epoch", str(args.freeze_epoch)])
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.returncode:
            failures.append(identifier)
            print(
                f"{identifier}: benchmark runner failed with status "
                f"{completed.returncode}: {completed.stderr[:800]}",
                file=sys.stderr)
        elif completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)

    aggregate_status = 0
    headline_tasks = sum(1 for _, task in tasks if task.get("headline") is True)
    if not args.variants_only and headline_tasks == len(EXPECTED_IDS):
        aggregate_status = aggregate_command(argparse.Namespace(
            manifest=args.manifest, reports=args.reports,
            cache_state=args.cache_state, output=args.aggregate_output,
            contract=args.contract))
    if failures:
        print("reference suite run failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return aggregate_status


def main() -> int:
    args = parse_args()
    try:
        if args.command == "validate":
            return validate_command(args)
        if args.command == "run":
            return run_command(args)
        return aggregate_command(args)
    except SuiteError as error:
        print(f"reference suite: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
