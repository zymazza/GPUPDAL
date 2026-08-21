#!/usr/bin/env python3
"""Verify the raw-report hashes named by a placement calibration record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class VerificationError(RuntimeError):
    """Raised when calibration provenance is malformed or has changed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail when a referenced local raw report is unavailable",
    )
    parser.add_argument(
        "--require-retained",
        action="store_true",
        help="fail when a report referenced below docs/reports is unavailable",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reports(case: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if "report" in case:
        result.append(case["report"])
    result.extend(case.get("reports", []))
    if not result:
        raise VerificationError(f"case has no report provenance: {case['id']}")
    return result


def verify_qualified_reference_report(
    raw: dict[str, Any], case: dict[str, Any], report: dict[str, str],
    calibration_id: str,
) -> None:
    """Bind retained complete-process evidence to its calibration case."""
    if "pipeline_sha256" not in report:
        return
    case_id = case["id"]

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise VerificationError(f"{case_id}: {message}")

    require(raw.get("schema") == "pdg-reference-pipeline-baseline-v1",
            "qualified report has the wrong schema")
    comparison = raw.get("comparison", {})
    require(comparison.get("contract") == "exact",
            "qualified report did not use the exact contract")
    for field in ("exact_outputs", "exact_artifacts", "exact_stdout",
                  "exact_stderr"):
        require(comparison.get(field) is True,
                f"qualified report does not prove {field}")
    require(raw.get("failures") == [], "qualified report contains failures")
    require(raw.get("pipeline", {}).get("sha256") ==
            report["pipeline_sha256"], "pipeline hash differs from the case")

    fixture = raw.get("fixture", {})
    require(fixture.get("extent", {}).get("points") == case["input_points"],
            "fixture point count differs from the case")
    input_hashes = {entry.get("sha256")
                    for entry in fixture.get("used_input_paths", [])}
    require(report.get("input_sha256") in input_hashes,
            "fixture hash differs from the case")

    summary = raw.get("summary", {})
    oracle_seconds = summary.get("oracle", {}).get("median_seconds")
    candidate_seconds = summary.get("candidate", {}).get("median_seconds")
    require(oracle_seconds == case["host_seconds"],
            "oracle time differs from the registered host time")
    require(candidate_seconds == case["device_seconds"],
            "candidate time differs from the registered device time")
    require(candidate_seconds < oracle_seconds,
            "candidate did not beat the oracle")

    qualification = raw.get("qualification", {})
    for field in ("model", "input_point_format", "input_record_bytes",
                  "input_compressed", "output_point_format",
                  "output_record_bytes", "output_compressed"):
        require(qualification.get(field) == case.get(field),
                f"qualification field {field} differs from the case")
    require(qualification.get("required_automatic_resident_route") is True,
            "report lacks the required automatic-route qualification")
    environment = raw.get("environment", {})
    require(environment.get("candidate_overrides", {}).get(
                "PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT") == "1",
            "required automatic-route switch was not active")
    require(environment.get("candidate_placement_profile", {}).get(
                "active_profile") == calibration_id,
            "active placement profile differs from the calibration record")
    require("LD_" in environment.get("scrubbed_prefixes", []),
            "loader environment was not scrubbed before measurement")

    binaries = raw.get("binaries", {})

    for binary in ("oracle", "candidate"):
        manifest = binaries.get(binary, {})
        require(isinstance(manifest.get("path"), str) and
                bool(manifest["path"]), f"{binary} has no executable path")
        digest = manifest.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64 and
                all(character in "0123456789abcdef" for character in digest),
                f"{binary} has no valid executable SHA-256")

    def require_component(binary: str, role: str) -> None:
        manifest = binaries.get(binary, {})
        matches = [component for component in
                   manifest.get("runtime_components", [])
                   if component.get("role") == role]
        require(len(matches) == 1,
                f"{binary} does not identify exactly one {role}")
        component = matches[0]
        require(isinstance(component.get("path"), str) and
                bool(component["path"]),
                f"{binary} {role} has no path")
        digest = component.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64 and
                all(character in "0123456789abcdef" for character in digest),
                f"{binary} {role} has no valid SHA-256")

    require_component("oracle", "loaded_pdal_shared_library")
    require_component("candidate", "engine")
    require_component("candidate", "loaded_pdal_shared_library")

    expected_output = report.get("output_sha256")
    require(expected_output in comparison.get("oracle_artifact_sha256", []),
            "oracle output hash differs from the case")
    require(expected_output in comparison.get("candidate_artifact_sha256", []),
            "candidate output hash differs from the case")
    for run in raw.get("runs", []):
        for artifact in run.get("artifacts", []):
            require(artifact.get("las_point_count") == case["output_points"],
                    "artifact point count differs from the case")


def main() -> int:
    args = parse_args()
    document = json.loads(args.calibration.read_text(encoding="utf-8"))
    if document.get("schema") != "pdg-placement-calibration-v1":
        raise VerificationError("unsupported placement calibration schema")

    expected: dict[Path, str] = {}
    linked_cases: dict[Path, list[tuple[dict[str, Any], dict[str, str]]]] = {}
    waived: set[Path] = set()
    retained: set[Path] = set()
    for case in document["cases"]:
        for report in reports(case):
            path = Path(report["path"])
            is_retained = (not path.is_absolute() and len(path.parts) >= 2 and
                           path.parts[:2] == ("docs", "reports"))
            if not path.is_absolute():
                path = args.repo_root / path
            if is_retained:
                retained.add(path)
            digest = report["sha256"]
            previous = expected.setdefault(path, digest)
            if previous != digest:
                raise VerificationError(f"conflicting hashes for {path}")
            linked_cases.setdefault(path, []).append((case, report))
            # A narrow status-gated waiver: a report explicitly recorded as
            # lost to the environment may be absent under --require-all. Its
            # recorded digest still binds — a restored file must match — and
            # an unmarked missing file still fails.
            if str(report.get("physical_file", "")).startswith(
                "lost-to-environment"
            ):
                waived.add(path)

    verified = 0
    missing: list[Path] = []
    waived_missing: list[Path] = []
    for path, expected_digest in expected.items():
        if not path.is_file():
            (waived_missing if path in waived else missing).append(path)
            continue
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise VerificationError(
                f"raw report hash mismatch: {path}\n"
                f"expected {expected_digest}\nactual   {actual_digest}"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        for case, report in linked_cases[path]:
            verify_qualified_reference_report(
                raw, case, report, document["calibration_id"])
        verified += 1

    required_missing = [path for path in missing if path in retained]
    if missing and args.require_all:
        rendered = "\n".join(str(path) for path in missing)
        raise VerificationError(f"missing raw calibration reports:\n{rendered}")
    if required_missing and args.require_retained:
        rendered = "\n".join(str(path) for path in required_missing)
        raise VerificationError(
            f"missing retained calibration reports:\n{rendered}")
    print(
        f"verified {verified}/{len(expected)} unique raw reports; "
        f"missing {len(missing)}; waived-lost {len(waived_missing)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, TypeError, ValueError, VerificationError) as error:
        raise SystemExit(f"verify_placement_calibration: {error}") from error
