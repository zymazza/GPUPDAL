#!/usr/bin/env python3
"""PDG Conformance Suite: bounded, complete-process PDAL differentials.

The suite accepts versioned manifests containing arbitrary offline PDAL
pipelines, stages their declared inputs into isolated work directories, and
runs the pinned PDAL oracle and PDG candidate through ``differential.py``.
Byte equality is authoritative by default.  ``copc-canonical-v1`` is allowed
only when the case explicitly documents oracle-side COPC nondeterminism.

The ``generate`` command expands the checked-in finite boundary recipe into
2,048 deterministic cases.  It is intentionally not a mutation fuzzer: case
membership, limits, and every arithmetic sequence are reviewable and stable.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable


MANIFEST_SCHEMA = "pdg-conformance-manifest-v1"
REPORT_SCHEMA = "pdg-conformance-report-v1"
RECIPE_SCHEMA = "pdg-bounded-differential-recipes-v1"


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def safe_relative(value: str, *, what: str) -> pathlib.Path:
    path = pathlib.PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\0" in value:
        raise ValueError(f"unsafe {what}: {value!r}")
    return pathlib.Path(*path.parts)


def _csv_rows(kind: str) -> str:
    rows = {
        "empty": [],
        "one-point": ["0,0,0,1,1,1,1,0,10,20,30,7"],
        "duplicate-xyz": [
            "1,2,3,4,1,1,2,1,11,21,31,8",
            "1,2,3,5,2,2,2,2,12,22,32,9",
            "1,2,3,6,2,1,1,3,13,23,33,10",
        ],
        "exact-distance-ties": [
            "0,0,0,1,1,1,1,0,10,20,30,1",
            "1,0,0,2,1,1,1,1,11,21,31,2",
            "-1,0,0,3,1,1,1,2,12,22,32,3",
            "0,1,0,4,1,1,1,3,13,23,33,4",
            "0,-1,0,5,1,1,1,4,14,24,34,5",
            "0,0,1,6,1,1,1,5,15,25,35,6",
            "0,0,-1,7,1,1,1,6,16,26,36,7",
        ],
        "signed-zero-and-nan": [
            "0,-0.0,0,0,0,1,1,nan,0,0,0,0",
            "-0.0,0,1,65535,31,15,15,-0.0,65535,65535,65535,255",
        ],
        "global-offsets": [
            "-179.9999999,-89.9999999,-10000,1,1,1,1,0,1,2,3,4",
            "179.9999999,89.9999999,100000,2,2,1,1,1,4,5,6,7",
        ],
        "return-boundaries": [
            f"{index},{index % 3},{index % 5},{index * 257 % 65536},"
            f"{index % 32},{1 + index % 15},15,{index * 0.5},"
            f"{index},{index + 1},{index + 2},{index % 256}"
            for index in range(16)
        ],
        "tiny-arithmetic-sequence": [
            f"{index * 0.125 - 2},{(index * 17) % 11 - 5},"
            f"{(index * index) % 13},{(index * 7919) % 65536},"
            f"{index % 32},{1 + index % 15},{1 + (index * 3) % 15},"
            f"{index / 7:.17g},{index % 65536},{index * 3 % 65536},"
            f"{index * 5 % 65536},{index % 256}"
            for index in range(33)
        ],
    }
    return "\n".join(rows[kind]) + ("\n" if rows[kind] else "")


def _reader(kind: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    header = ("X,Y,Z,Intensity,Classification,ReturnNumber,NumberOfReturns,"
              "GpsTime,Red,Green,Blue,Extra0")
    reader: dict[str, object] = {
        "type": "readers.text", "filename": "input.csv",
        "header": header, "separator": ",", "skip": 0,
    }
    if kind in ("global-offsets", "signed-zero-and-nan"):
        reader["override_srs"] = "EPSG:4326"
    contents = _csv_rows(kind)
    return reader, [{"path": "input.csv", "content": contents,
                     "sha256": digest_bytes(contents.encode())}]


def _stage(kind: str) -> dict[str, object] | None:
    stages: dict[str, dict[str, object] | None] = {
        "identity": None,
        "assign-extra-dimension": {
            "type": "filters.assign", "value": "Extra0 = Extra0 + 1"},
        "ordered-ferry": {
            "type": "filters.ferry", "dimensions": "Extra0=>Extra1"},
        "inclusive-range": {
            "type": "filters.range", "limits": "Intensity[0:65535]"},
        "stable-sort": {
            "type": "filters.sort", "dimension": "X", "order": "ASC"},
        "decimation": {"type": "filters.decimation", "step": 2, "offset": 0},
        "head": {"type": "filters.head", "count": 7},
        "tail": {"type": "filters.tail", "count": 7},
        "normal-knn": {"type": "filters.normal", "knn": 3,
                       "always_up": False},
        "covariance-features": {
            "type": "filters.covariancefeatures", "knn": 3,
            "feature_set": "Dimensionality"},
        "nndistance-tie": {"type": "filters.nndistance", "k": 2,
                            "mode": "kth"},
        "label-duplicates": {"type": "filters.label_duplicates",
                              "dimensions": "X,Y,Z"},
        "identity-transformation": {
            "type": "filters.transformation",
            "matrix": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"},
        "fixed-seed-randomize": {"type": "filters.randomize", "seed": 17},
        "same-crs-reprojection": {"type": "filters.reprojection",
                                  "in_srs": "EPSG:4326",
                                  "out_srs": "EPSG:4326"},
        "empty-crop": {"type": "filters.crop",
                       "bounds": "([1000000,1000001],[1000000,1000001])"},
    }
    return stages[kind]


def _writer(kind: str) -> dict[str, object]:
    formats = {f"format-{value}": value for value in (0, 1, 2, 3, 6, 7, 8)}
    writer: dict[str, object] = {
        "type": "writers.las", "filename": "output.las",
        "compression": False, "minor_version": 4,
        "extra_dims": "all",
    }
    if kind in formats:
        writer["dataformat_id"] = formats[kind]
    else:
        writer.update({"dataformat_id": 7, "scale_x": 0.0000002,
                       "scale_y": 0.5, "scale_z": 2.0,
                       "offset_x": -180.0, "offset_y": -90.0,
                       "offset_z": -10000.0})
    return writer


def generated_manifest(recipe: dict[str, object], limit: int | None = None) -> dict[str, object]:
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise ValueError("unsupported bounded recipe schema")
    axes = recipe.get("axes")
    if not isinstance(axes, dict):
        raise ValueError("bounded recipe has no axes object")
    names = ("input", "stage", "writer", "graph")
    values: list[list[str]] = []
    for name in names:
        axis = axes.get(name)
        if not isinstance(axis, list) or not axis or not all(
                isinstance(value, str) for value in axis):
            raise ValueError(f"bounded recipe axis {name!r} is invalid")
        values.append(axis)
    combinations = list(itertools.product(*values))
    target = int(recipe.get("target_cases", len(combinations)))
    if len(combinations) != target:
        raise ValueError(
            f"bounded recipe expands to {len(combinations)}, expected {target}")
    if limit is not None:
        if limit <= 0 or limit > target:
            raise ValueError(f"case limit must be in 1..{target}")
        combinations = combinations[:limit]
    cases: list[dict[str, object]] = []
    determinism_repeats = int(recipe.get("determinism_repeats", 2))
    for index, (input_kind, stage_kind, writer_kind, graph_kind) in enumerate(combinations):
        reader, files = _reader(input_kind)
        pipeline: list[dict[str, object]] = [reader]
        stage = _stage(stage_kind)
        # A one-point covariance is undefined and makes the pinned oracle
        # abort.  Keep the boundary case successful while still exercising
        # the stage's standard conditional-skip semantics.
        if input_kind == "one-point" and stage_kind == "covariance-features":
            assert stage is not None
            stage["where"] = "X > 0"
        if stage is not None:
            pipeline.append(stage)
        pipeline.append(_writer(writer_kind))
        root: object = (pipeline if graph_kind.startswith("array-root") else
                        {"pipeline": pipeline})
        case_id = (f"g{index:04d}-{input_kind}-{stage_kind}-"
                   f"{writer_kind}-{graph_kind}")
        cases.append({
            "id": case_id, "pipeline": root, "files": files,
            "comparison_mode": "bytes",
            "determinism_repeats": determinism_repeats,
            "coverage": {"input": input_kind, "stage": stage_kind,
                         "writer": writer_kind, "graph": graph_kind},
        })
    large = recipe.get("named_large_case")
    if isinstance(large, dict) and len(cases) == target:
        replacement = int(large["replacement_index"])
        points = int(large["points"])
        pipeline = {"pipeline": [
            {"type": "readers.faux", "count": points,
             "bounds": "([0,999],[0,999],[0,100])", "mode": "ramp"},
            {"type": "filters.normal", "knn": 8, "always_up": False},
            {"type": "writers.las", "filename": "output.las",
             "compression": False, "minor_version": 4,
             "dataformat_id": 7, "extra_dims": "all"},
        ]}
        cases[replacement] = {
            "id": str(large["id"]), "pipeline": pipeline, "files": [],
            "comparison_mode": "bytes",
            "determinism_repeats": determinism_repeats,
            "coverage": {"input": "giant-cloud", "stage": "normal-knn",
                         "writer": "format-7", "graph": "object-root"},
            "reason": large["reason"],
        }
    limits = dict(recipe.get("limits", {}))
    limits["max_cases"] = target
    return {
        "schema": MANIFEST_SCHEMA,
        "suite_id": recipe["suite_id"],
        "description": recipe.get("description", ""),
        "recipe_sha256": digest_bytes(canonical_json(recipe)),
        "complete_case_count": target,
        "limits": limits,
        "cases": cases,
    }


def _stage_types(pipeline: object) -> Iterable[dict[str, object]]:
    root = pipeline.get("pipeline") if isinstance(pipeline, dict) else pipeline
    if not isinstance(root, list):
        raise ValueError("pipeline root must be an array or {pipeline:[...]} object")
    for stage in root:
        if isinstance(stage, str):
            yield {"type": "readers.las" if stage == root[0] else
                   "writers.las" if stage == root[-1] else "unknown",
                   "filename": stage}
        elif isinstance(stage, dict):
            yield stage
        else:
            raise ValueError("pipeline stage must be a string or object")


def validate_manifest(manifest: dict[str, object]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported conformance manifest schema")
    suite_id = manifest.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError("manifest suite_id must be a nonempty string")
    limits = manifest.get("limits")
    cases = manifest.get("cases")
    if not isinstance(limits, dict) or not isinstance(cases, list):
        raise ValueError("manifest requires limits and cases")
    max_cases = int(limits.get("max_cases", 0))
    if max_cases <= 0 or len(cases) > max_cases:
        raise ValueError(f"manifest has {len(cases)} cases above cap {max_cases}")
    seen: set[str] = set()
    max_inline = int(limits.get("max_inline_input_bytes_per_case", 0))
    max_repeats = int(limits.get("max_determinism_repeats", 0))
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each conformance case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"invalid or duplicate case id: {case_id!r}")
        seen.add(case_id)
        safe_relative(case_id, what="case id")
        mode = case.get("comparison_mode", "bytes")
        if mode not in ("bytes", "copc-canonical-v1"):
            raise ValueError(f"case {case_id}: unknown comparison mode {mode}")
        if mode == "copc-canonical-v1" and not case.get(
                "oracle_nondeterminism_reason"):
            raise ValueError(
                f"case {case_id}: canonical COPC mode requires an explicit "
                "oracle_nondeterminism_reason")
        repeats = int(case.get("determinism_repeats", 0))
        if repeats < 2 or max_repeats < 2 or repeats > max_repeats:
            raise ValueError(
                f"case {case_id}: determinism repeats {repeats} outside "
                f"release range 2..{max_repeats}")
        expected_returncode = case.get("expected_returncode", 0)
        if (not isinstance(expected_returncode, int) or
                isinstance(expected_returncode, bool) or
                expected_returncode < 0 or expected_returncode > 255):
            raise ValueError(
                f"case {case_id}: expected_returncode must be an integer 0..255")
        files = case.get("files", [])
        if not isinstance(files, list):
            raise ValueError(f"case {case_id}: files must be an array")
        declared: set[str] = set()
        inline_bytes = 0
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError(f"case {case_id}: invalid file declaration")
            relative = safe_relative(item["path"], what="staged input path")
            rendered = relative.as_posix()
            if rendered in declared:
                raise ValueError(f"case {case_id}: duplicate staged input {rendered}")
            declared.add(rendered)
            has_content = isinstance(item.get("content"), str)
            has_source = isinstance(item.get("source"), str)
            if has_content == has_source:
                raise ValueError(
                    f"case {case_id}: each file needs exactly one of content/source")
            if has_content:
                encoded = item["content"].encode()
                inline_bytes += len(encoded)
                if item.get("sha256") != digest_bytes(encoded):
                    raise ValueError(f"case {case_id}: inline input hash mismatch")
            elif not isinstance(item.get("sha256"), str):
                raise ValueError(f"case {case_id}: source input requires sha256")
        if max_inline <= 0 or inline_bytes > max_inline:
            raise ValueError(
                f"case {case_id}: inline bytes {inline_bytes} above cap {max_inline}")
        allow_external = bool(case.get("allow_external_io", False))
        for stage in _stage_types(case.get("pipeline")):
            stage_type = str(stage.get("type", ""))
            if not allow_external and any(token in stage_type for token in (
                    "python", "shell", "pgpointcloud", "mongodb", "ept",
                    "stac", "i3s", "rdb", "redis", "kafka", "s3")):
                raise ValueError(
                    f"case {case_id}: external stage {stage_type!r} needs "
                    "allow_external_io=true")
            filename = stage.get("filename")
            if (not allow_external and stage_type.startswith("readers.") and
                    stage_type != "readers.faux" and
                    not isinstance(filename, str)):
                raise ValueError(
                    f"case {case_id}: offline reader {stage_type!r} must "
                    "declare a staged filename")
            if (not allow_external and stage_type.startswith("writers.") and
                    stage_type != "writers.null" and
                    not isinstance(filename, str)):
                raise ValueError(
                    f"case {case_id}: offline writer {stage_type!r} must "
                    "declare an isolated filename")
            if not isinstance(filename, str):
                continue
            relative = safe_relative(filename, what="pipeline filename")
            if stage_type.startswith("readers.") and relative.as_posix() not in declared:
                raise ValueError(
                    f"case {case_id}: reader input {filename!r} is not staged")


def _stage_case(case: dict[str, object], root: pathlib.Path,
                max_inline: int) -> tuple[pathlib.Path, list[str], list[dict[str, object]]]:
    case_id = str(case["id"])
    stage = root / "staged" / case_id
    stage.mkdir(parents=True, exist_ok=False)
    seeds: list[str] = []
    inputs: list[dict[str, object]] = []
    total = 0
    for declaration in case.get("files", []):
        relative = safe_relative(str(declaration["path"]), what="input path")
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if "content" in declaration:
            encoded = str(declaration["content"]).encode()
            destination.write_bytes(encoded)
        else:
            source = pathlib.Path(str(declaration["source"]))
            if not source.is_file() or digest_file(source) != declaration["sha256"]:
                raise ValueError(f"case {case_id}: missing or changed source {source}")
            shutil.copyfile(source, destination)
        total += destination.stat().st_size
        if total > max_inline:
            raise ValueError(f"case {case_id}: staged inputs exceed byte cap")
        seeds.append(f"{relative.as_posix()}={destination}")
        inputs.append({"path": relative.as_posix(),
                       "bytes": destination.stat().st_size,
                       "sha256": digest_file(destination)})
    pipeline = stage / "pipeline.json"
    pipeline.write_bytes(canonical_json(case["pipeline"]) + b"\n")
    seeds.append(f"pipeline.json={pipeline}")
    inputs.append({"path": "pipeline.json", "bytes": pipeline.stat().st_size,
                   "sha256": digest_file(pipeline)})
    return pipeline, seeds, inputs


def run_suite(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    cases = manifest["cases"]
    assert isinstance(cases, list)
    if args.case_limit is not None:
        if args.case_limit <= 0 or args.case_limit > len(cases):
            raise ValueError("--case-limit is outside the manifest")
        cases = cases[:args.case_limit]
    complete_count = int(manifest.get("complete_case_count", len(manifest["cases"])))
    partial = len(cases) != complete_count
    if partial and not args.allow_partial:
        raise ValueError(
            f"selected {len(cases)} of {complete_count} cases; use --allow-partial "
            "for an explicitly non-qualifying smoke run")
    limits = manifest["limits"]
    assert isinstance(limits, dict)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    failed = 0
    for case_value in cases:
        assert isinstance(case_value, dict)
        case = case_value
        _, seeds, inputs = _stage_case(
            case, args.work_dir,
            int(limits["max_inline_input_bytes_per_case"]))
        repeats = int(case.get("determinism_repeats", 1))
        repeat_reports: list[dict[str, object]] = []
        candidate_views: list[object] = []
        for repeat in range(repeats):
            run_id = f"{case['id']}-repeat-{repeat:02d}"
            command = [sys.executable, str(args.differential.resolve()),
                       "--oracle", str(args.oracle.resolve()),
                       "--candidate", str(args.candidate.resolve()),
                       "--case", run_id, "--work-dir", str(args.work_dir / "runs"),
                       "--comparison-mode", str(case.get("comparison_mode", "bytes")),
                       "--timeout-seconds", str(limits["timeout_seconds_per_process"]),
                       "--max-artifact-files", str(limits["max_artifact_files_per_case"]),
                       "--max-artifact-bytes", str(limits["max_artifact_bytes_per_case"]),
                       "--max-stream-bytes", str(limits["max_stream_bytes_per_process"])]
            if args.candidate_oracle:
                command.extend(["--candidate-oracle", str(args.candidate_oracle.resolve())])
            if args.frozen_time_library:
                command.extend(["--frozen-time-library",
                                str(args.frozen_time_library.resolve()),
                                "--freeze-epoch", str(args.freeze_epoch)])
            for seed in seeds:
                command.extend(["--seed-file", seed])
            command.extend(["--", "pipeline", "pipeline.json"])
            completed = subprocess.run(command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, check=False,
                                       text=True)
            report_path = args.work_dir / "runs" / "reports" / f"{run_id}.json"
            report = json.loads(report_path.read_text()) if report_path.is_file() else {
                "differences": [{"artifact": "missing_differential_report"}],
                "runner_stderr": completed.stderr[-2000:],
            }
            repeat_reports.append({"id": run_id, "returncode": completed.returncode,
                                   "report": report})
            if case.get("comparison_mode", "bytes") == "bytes":
                candidate_views.append(report.get("candidate_artifacts"))
            else:
                candidate_views.append(report.get("semantic_comparisons"))
        deterministic = len({digest_bytes(canonical_json(value))
                             for value in candidate_views}) <= 1
        expected_returncode = int(case.get("expected_returncode", 0))
        product_statuses_match = all(
            record["report"].get("oracle_run", {}).get("returncode") ==
            expected_returncode and
            record["report"].get("candidate_run", {}).get("returncode") ==
            expected_returncode
            for record in repeat_reports)
        passed = (all(record["returncode"] == 0 for record in repeat_reports) and
                  product_statuses_match and deterministic)
        failed += not passed
        results.append({"id": case["id"], "passed": passed,
                        "comparison_mode": case.get("comparison_mode", "bytes"),
                        "expected_returncode": expected_returncode,
                        "product_statuses_match": product_statuses_match,
                        "deterministic": deterministic, "inputs": inputs,
                        "coverage": case.get("coverage", {}),
                        "repeats": repeat_reports})
        print(f"{'PASS' if passed else 'FAIL'} {case['id']}")
        if not passed and args.fail_fast:
            break
    report = {
        "schema": REPORT_SCHEMA, "suite_id": manifest["suite_id"],
        "manifest": {"path": str(manifest_path),
                     "sha256": digest_file(manifest_path),
                     "recipe_sha256": manifest.get("recipe_sha256")},
        "oracle": {"path": str(args.oracle.resolve()),
                   "sha256": digest_file(args.oracle)},
        "candidate": {"path": str(args.candidate.resolve()),
                      "sha256": digest_file(args.candidate)},
        "complete_case_count": complete_count, "executed_cases": len(results),
        "partial": partial, "passed_cases": sum(r["passed"] for r in results),
        "failed_cases": sum(not r["passed"] for r in results),
        "zero_unexplained_semantic_differences": not failed and not partial,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"conformance: {report['passed_cases']}/{len(results)} passed; {args.report}")
    return 0 if not failed and (not partial or args.allow_partial) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", allow_abbrev=False)
    generate.add_argument("--recipes", required=True, type=pathlib.Path)
    generate.add_argument("--output", required=True, type=pathlib.Path)
    generate.add_argument("--limit", type=int)
    validate = subparsers.add_parser("validate", allow_abbrev=False)
    validate.add_argument("manifest", type=pathlib.Path)
    run = subparsers.add_parser("run", allow_abbrev=False)
    run.add_argument("--manifest", required=True, type=pathlib.Path)
    run.add_argument("--oracle", required=True, type=pathlib.Path)
    run.add_argument("--candidate", required=True, type=pathlib.Path)
    run.add_argument("--candidate-oracle", type=pathlib.Path)
    run.add_argument("--differential", required=True, type=pathlib.Path)
    run.add_argument("--work-dir", required=True, type=pathlib.Path)
    run.add_argument("--report", required=True, type=pathlib.Path)
    run.add_argument("--frozen-time-library", type=pathlib.Path)
    run.add_argument("--freeze-epoch", type=int, default=1_704_067_200)
    run.add_argument("--case-limit", type=int)
    run.add_argument("--allow-partial", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "generate":
            recipe = json.loads(args.recipes.read_text())
            manifest = generated_manifest(recipe, args.limit)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(manifest, indent=2,
                                              sort_keys=True) + "\n")
            print(f"generated {len(manifest['cases'])} cases: {args.output}")
            return 0
        if args.command == "validate":
            manifest = json.loads(args.manifest.read_text())
            validate_manifest(manifest)
            print(f"valid {MANIFEST_SCHEMA}: {len(manifest['cases'])} cases")
            return 0
        for executable in (args.oracle, args.candidate, args.differential):
            if not executable.is_file():
                raise ValueError(f"required file does not exist: {executable}")
        return run_suite(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"gpupdal conformance: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
