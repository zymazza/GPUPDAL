#!/usr/bin/env python3
"""Select, seal, run, and analyze the preregistered PDG 3DEP study.

Selection is deterministic and performance-blind.  One frozen catalog record
becomes at most one observation, regardless of timing repeats or tile count.
The run command consumes a closed release payload and records every selected
project, including missing, inexact, and failed attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import tempfile
from typing import Any


PREREG_SCHEMA = "pdg-3dep-preregistration-v1"
SELECTION_SCHEMA = "pdg-3dep-selection-v1"
INPUTS_SCHEMA = "pdg-3dep-inputs-v1"
RUN_SCHEMA = "pdg-3dep-run-v1"
ANALYSIS_SCHEMA = "pdg-3dep-analysis-v1"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_schema(value: Any, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{label} must use schema {schema}")
    return value


def selection_score(seed: str, *parts: str) -> str:
    encoded = "\0".join((seed, *parts)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def eligible_tiles(project: dict[str, Any]) -> list[dict[str, Any]]:
    tiles = project.get("tiles")
    if not isinstance(tiles, list):
        return []
    result = []
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        tile_id = tile.get("tile_id")
        url = tile.get("url")
        if (isinstance(tile_id, str) and tile_id and isinstance(url, str) and
                url.startswith("https://") and
                pathlib.PurePosixPath(url.split("?", 1)[0]).suffix.lower()
                in {".laz", ".copc"}):
            result.append(tile)
        elif (isinstance(tile_id, str) and tile_id and isinstance(url, str) and
              url.startswith("https://") and url.lower().split("?", 1)[0]
              .endswith(".copc.laz")):
            result.append(tile)
    return result


def build_selection(prereg: dict[str, Any], catalog: Any,
                    preregistration_sha256: str,
                    catalog_sha256: str) -> dict[str, Any]:
    if not isinstance(catalog, dict) or not isinstance(catalog.get("projects"), list):
        raise ValueError("catalog must be an object with a projects array")
    seed = prereg["selection"]["seed"]
    count = int(prereg["sample_size"])
    seen: set[str] = set()
    seen_canonical: set[str] = set()
    ranked: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    audit: list[dict[str, Any]] = []
    for project in catalog["projects"]:
        if not isinstance(project, dict) or not isinstance(project.get("project_id"), str):
            raise ValueError("every catalog project requires a string project_id")
        project_id = project["project_id"]
        if not project_id or project_id in seen:
            raise ValueError(f"duplicate or empty project_id: {project_id!r}")
        seen.add(project_id)
        canonical_id = project.get("canonical_project_id", project_id)
        if not isinstance(canonical_id, str) or not canonical_id:
            raise ValueError(f"project {project_id}: invalid canonical_project_id")
        if canonical_id in seen_canonical:
            raise ValueError(
                f"catalog aliases are unresolved; duplicate canonical project "
                f"identifier: {canonical_id!r}")
        seen_canonical.add(canonical_id)
        tiles = eligible_tiles(project)
        score = selection_score(seed, project_id)
        if not tiles:
            audit.append({"project_id": project_id, "project_score": score,
                          "status": "ineligible-no-public-las-or-copc-tile"})
            continue
        ranked.append((score, project, tiles))
    ranked.sort(key=lambda item: (item[0], item[1]["project_id"]))
    if len(ranked) < count:
        raise ValueError(f"catalog has only {len(ranked)} eligible projects; need {count}")
    selected = []
    for index, (score, project, tiles) in enumerate(ranked[:count], 1):
        project_id = project["project_id"]
        ranked_tiles = sorted(
            tiles, key=lambda tile: (selection_score(seed, project_id,
                                                     tile["tile_id"]),
                                     tile["tile_id"]))
        tile = ranked_tiles[0]
        selected.append({
            "selection_index": index, "project_id": project_id,
            "canonical_project_id": project.get("canonical_project_id", project_id),
            "project_title": project.get("title"), "project_score": score,
            "tile_id": tile["tile_id"], "tile_url": tile["url"],
            "tile_score": selection_score(seed, project_id, tile["tile_id"]),
        })
    return {
        "schema": SELECTION_SCHEMA,
        "study_id": prereg["study_id"],
        "preregistration_sha256": preregistration_sha256,
        "catalog_sha256": catalog_sha256,
        "selection_algorithm": "sha256-nul-ranking-v1",
        "sample_size": count,
        "selected": selected,
        "preselection_ineligible": sorted(audit,
                                           key=lambda item: item["project_score"]),
        "status": "selected-before-performance-observation",
    }


def select(args: argparse.Namespace) -> int:
    prereg = require_schema(load(args.preregistration), PREREG_SCHEMA,
                             "preregistration")
    catalog = load(args.catalog)
    result = build_selection(prereg, catalog, sha256(args.preregistration),
                             sha256(args.catalog))
    write_json(args.output, result)
    print(f"selected {result['sample_size']} independent projects -> {args.output}")
    return 0


def seal_inputs(args: argparse.Namespace) -> int:
    selection = require_schema(load(args.selection), SELECTION_SCHEMA,
                               "selection")
    mapping = load(args.mapping)
    if not isinstance(mapping, dict) or not isinstance(mapping.get("inputs"), list):
        raise ValueError("mapping must be an object with an inputs array")
    by_project: dict[str, dict[str, Any]] = {}
    for record in mapping["inputs"]:
        if not isinstance(record, dict) or not isinstance(record.get("project_id"), str):
            raise ValueError("each input mapping requires project_id")
        if record["project_id"] in by_project:
            raise ValueError(f"duplicate input mapping: {record['project_id']}")
        by_project[record["project_id"]] = record
    sealed = []
    for chosen in selection["selected"]:
        project_id = chosen["project_id"]
        record = by_project.get(project_id)
        if record is None:
            sealed.append({"project_id": project_id, "selection_index":
                           chosen["selection_index"], "status": "missing",
                           "reason": "no pre-timing input mapping"})
            continue
        if (record.get("tile_id") != chosen["tile_id"] or
                record.get("tile_url") != chosen["tile_url"]):
            raise ValueError(
                f"input mapping for {project_id} does not match the "
                "deterministically selected tile id and URL")
        path = pathlib.Path(record.get("path", "")).resolve()
        if not path.is_file():
            sealed.append({"project_id": project_id, "selection_index":
                           chosen["selection_index"], "status": "missing",
                           "path": str(path), "reason": "input file absent"})
            continue
        sealed.append({"project_id": project_id,
                       "selection_index": chosen["selection_index"],
                       "tile_id": chosen["tile_id"], "tile_url": chosen["tile_url"],
                       "status": "sealed", "path": str(path),
                       "bytes": path.stat().st_size, "sha256": sha256(path)})
    result = {"schema": INPUTS_SCHEMA, "study_id": selection["study_id"],
              "selection_sha256": sha256(args.selection), "inputs": sealed,
              "rule": "Input membership and bytes are sealed before timing; missing records remain analysis failures."}
    write_json(args.output, result)
    print(f"sealed {sum(r['status'] == 'sealed' for r in sealed)}/"
          f"{len(sealed)} inputs -> {args.output}")
    return 0


def executable_role(payload: pathlib.Path, artifact: dict[str, Any], name: str) -> pathlib.Path:
    record = artifact.get("roles", {}).get(name)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(f"artifact manifest has no {name} role")
    path = payload / record["path"]
    if not path.is_file():
        raise ValueError(f"artifact {name} is absent: {path}")
    return path


def artifact_file(artifact: dict[str, Any], relative: str) -> dict[str, Any]:
    records = artifact.get("payload", {}).get("files")
    if not isinstance(records, list):
        raise ValueError("artifact manifest has no payload inventory")
    matches = [record for record in records
               if isinstance(record, dict) and record.get("path") == relative]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise ValueError(f"artifact manifest does not bind {relative}")
    return matches[0]


def run_command(command: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    try:
        with tempfile.TemporaryFile() as stdout_file, \
                tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=stdout_file,
                stderr=stderr_file, start_new_session=True)
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                returncode = 124
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(1 << 20).decode("utf-8", "replace")
            stderr = stderr_file.read(1 << 20).decode("utf-8", "replace")
        if returncode == 124:
            stderr += f"\n3DEP subprocess timeout after {timeout}s\n"
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)
    except OSError as error:
        return subprocess.CompletedProcess(command, 126, "", str(error))


def run(args: argparse.Namespace) -> int:
    prereg = require_schema(load(args.preregistration), PREREG_SCHEMA,
                             "preregistration")
    selection = require_schema(load(args.selection), SELECTION_SCHEMA, "selection")
    inputs = require_schema(load(args.inputs), INPUTS_SCHEMA, "inputs")
    artifact = require_schema(load(args.artifact_manifest),
                              "pdg-frozen-release-artifact-v1", "artifact manifest")
    if selection["preregistration_sha256"] != sha256(args.preregistration):
        raise ValueError("selection does not bind the supplied preregistration")
    if inputs["selection_sha256"] != sha256(args.selection):
        raise ValueError("inputs do not bind the supplied selection")
    if len(selection["selected"]) != int(prereg["sample_size"]):
        raise ValueError("selection sample size differs from preregistration")
    payload = args.payload.resolve()
    tools = payload / "evidence" / "scripts" / "pdg"
    checker = tools / "artifact_manifest.py"
    runner = tools / "benchmark_reference.py"
    pipeline = payload / "evidence" / prereg["pipeline"]["path"]
    frozen_prereg = payload / "evidence" / "bench" / "3dep" / \
        "preregistration-v1.json"
    frozen = payload / "evidence" / "libpdg_frozen_time.so"
    for required in (checker, runner, pipeline, frozen_prereg, frozen):
        if not required.is_file():
            raise ValueError(f"closed payload lacks required evidence file: {required}")
    if (sha256(frozen_prereg) != sha256(args.preregistration) or
            artifact_file(
                artifact, "evidence/bench/3dep/preregistration-v1.json"
            )["sha256"] != sha256(args.preregistration)):
        raise ValueError("supplied preregistration is not the one frozen in the release payload")
    if sha256(pipeline) != prereg["pipeline"]["sha256"]:
        raise ValueError("frozen r6 pipeline hash differs from preregistration")
    candidate = executable_role(payload, artifact, "candidate")
    oracle = executable_role(payload, artifact, "oracle")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be absent or empty so stale "
                         "reports cannot enter the frozen study record")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before_path = args.output_dir / "artifact-before.json"
    checked = run_command([sys.executable, str(checker), "check", "--manifest",
                           str(args.artifact_manifest), "--root", str(payload),
                           "--report", str(before_path)])
    if checked.returncode:
        raise ValueError(f"closed artifact failed pre-run check: {checked.stderr}")
    gpu = run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
                       "power.limit,clocks.current.graphics,clocks.current.memory",
                       "--format=csv,noheader"], timeout=30)
    if gpu.returncode or len([line for line in gpu.stdout.splitlines() if line.strip()]) != 1:
        raise ValueError("study requires exactly one visible NVIDIA GPU")
    required_gpu = prereg["machine"]["required_gpu_name_substring"]
    if required_gpu not in gpu.stdout:
        raise ValueError(f"study requires {required_gpu}; observed {gpu.stdout.strip()!r}")
    sealed_by_project = {record["project_id"]: record for record in inputs["inputs"]}
    attempts = []
    for chosen in selection["selected"]:
        project_id = chosen["project_id"]
        sealed = sealed_by_project.get(project_id)
        base = {"selection_index": chosen["selection_index"],
                "project_id": project_id, "tile_id": chosen["tile_id"],
                "tile_url": chosen["tile_url"]}
        if (not sealed or sealed.get("status") != "sealed" or
                sealed.get("tile_id") != chosen["tile_id"] or
                sealed.get("tile_url") != chosen["tile_url"]):
            attempts.append({**base, "status": "failed-before-run",
                             "reason": (sealed or {}).get(
                                 "reason", "no exactly matching sealed input")})
            continue
        input_path = pathlib.Path(sealed["path"])
        if (not input_path.is_file() or input_path.stat().st_size != sealed["bytes"] or
                sha256(input_path) != sealed["sha256"]):
            attempts.append({**base, "status": "failed-before-run",
                             "reason": "sealed input bytes changed"})
            continue
        report_path = args.output_dir / "attempts" / f"{chosen['selection_index']:02d}.json"
        work = args.output_dir / "work" / f"{chosen['selection_index']:02d}"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(runner), "--oracle", str(oracle),
                   "--candidate", str(candidate), "--pipeline", str(pipeline),
                   "--fixture-laz", str(input_path), "--fixture",
                   f"input.laz={input_path}", "--label", f"3dep-{project_id}",
                   "--work-dir", str(work), "--report", str(report_path),
                   "--cache-state", "warm", "--runs", "3", "--warmups", "1",
                   "--contract", "exact", "--candidate-env",
                   f"PDG_ORACLE_PDAL={oracle}",
                   "--frozen-time-library", str(frozen),
                   "--freeze-epoch", "1704067200"]
        completed = run_command(command, timeout=args.timeout_seconds)
        report = None
        report_error = None
        if report_path.is_file():
            try:
                report = load(report_path)
            except (OSError, json.JSONDecodeError) as error:
                report_error = f"malformed benchmark report: {error}"
        attempts.append({**base, "status": "completed" if completed.returncode == 0
                         and report_error is None else "failed",
                         "returncode": completed.returncode,
                         "input_sha256": sealed["sha256"], "input_bytes": sealed["bytes"],
                         "report_path": str(report_path), "report_sha256":
                         sha256(report_path) if report_path.is_file() else None,
                         "exact_outputs": (report or {}).get("comparison", {}).get("exact_outputs"),
                         "median_speedup": (report or {}).get("comparison", {}).get("median_speedup"),
                         "point_count": (report or {}).get("fixture", {}).get(
                             "extent", {}).get("points"),
                         "report_error": report_error,
                         "stderr_head": completed.stderr[:1000]})
    after_path = args.output_dir / "artifact-after.json"
    post = run_command([sys.executable, str(checker), "check", "--manifest",
                        str(args.artifact_manifest), "--root", str(payload),
                        "--report", str(after_path)])
    result = {"schema": RUN_SCHEMA, "study_id": prereg["study_id"],
              "preregistration_sha256": sha256(args.preregistration),
              "selection_sha256": sha256(args.selection),
              "inputs_sha256": sha256(args.inputs),
              "artifact_manifest_sha256": sha256(args.artifact_manifest),
              "artifact_valid_before": True,
              "artifact_valid_after": post.returncode == 0,
              "artifact_checks": {
                  "before": {"path": str(before_path),
                             "sha256": sha256(before_path)},
                  "after": {"path": str(after_path),
                            "sha256": sha256(after_path)
                            if after_path.is_file() else None},
              },
              "machine": {"nvidia_smi": gpu.stdout.strip(),
                          "cpu_affinity": sorted(os.sched_getaffinity(0)),
                          "lscpu": run_command(["lscpu"]).stdout,
                          "memory": run_command(["free", "-h"]).stdout,
                          "cuda_compiler": run_command(
                              ["nvcc", "--version"]).stdout},
              "attempts": attempts,
              "protocol": {"warmups": 1, "measured_pairs": 3,
                           "cache_state": "warm", "fast_mode": False}}
    output = args.output_dir / "study-run.json"
    write_json(output, result)
    print(f"recorded {len(attempts)} attempts -> {output}")
    return 0 if post.returncode == 0 else 1


def clopper_pearson_lower(successes: int, attempts: int,
                          alpha: float = 0.05) -> float:
    if attempts <= 0 or successes < 0 or successes > attempts:
        raise ValueError("invalid binomial counts")
    if successes == 0:
        return 0.0
    # Solve P_p[X >= successes] = alpha.  This is Beta^-1(alpha;
    # successes, attempts-successes+1), implemented without scipy.
    def survival(probability: float) -> float:
        return sum(math.comb(attempts, index) * probability ** index *
                   (1.0 - probability) ** (attempts - index)
                   for index in range(successes, attempts + 1))
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if survival(middle) < alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def analyze(args: argparse.Namespace) -> int:
    prereg = require_schema(load(args.preregistration), PREREG_SCHEMA,
                             "preregistration")
    selection = require_schema(load(args.selection), SELECTION_SCHEMA, "selection")
    inputs = require_schema(load(args.inputs), INPUTS_SCHEMA, "inputs")
    artifact = require_schema(load(args.artifact_manifest),
                              "pdg-frozen-release-artifact-v1",
                              "artifact manifest")
    run_result = require_schema(load(args.run_report), RUN_SCHEMA, "run report")
    catalog = load(args.catalog)
    bindings = {
        "preregistration_sha256": sha256(args.preregistration),
        "selection_sha256": sha256(args.selection),
        "inputs_sha256": sha256(args.inputs),
        "artifact_manifest_sha256": sha256(args.artifact_manifest),
    }
    if selection.get("preregistration_sha256") != bindings["preregistration_sha256"]:
        raise ValueError("selection does not bind the supplied preregistration")
    regenerated = build_selection(
        prereg, catalog, bindings["preregistration_sha256"],
        sha256(args.catalog))
    if selection != regenerated:
        raise ValueError("selection is not the deterministic sample from the supplied frozen catalog")
    if inputs.get("selection_sha256") != bindings["selection_sha256"]:
        raise ValueError("inputs do not bind the supplied selection")
    if artifact_file(
            artifact, "evidence/bench/3dep/preregistration-v1.json"
    )["sha256"] != bindings["preregistration_sha256"]:
        raise ValueError("analysis preregistration is not frozen in the release artifact")
    for field, expected_hash in bindings.items():
        if run_result.get(field) != expected_hash:
            raise ValueError(f"run report does not bind {field}")
    checks = run_result.get("artifact_checks")
    if not isinstance(checks, dict):
        raise ValueError("run report has no frozen-artifact checks")
    for phase in ("before", "after"):
        check = checks.get(phase)
        if not isinstance(check, dict) or not isinstance(check.get("path"), str):
            raise ValueError(f"run report lacks {phase} artifact check")
        check_path = pathlib.Path(check["path"])
        if (not check_path.is_file() or check.get("sha256") != sha256(check_path)):
            raise ValueError(f"{phase} artifact-check report is missing or changed")
        checked = require_schema(load(check_path),
                                 "pdg-frozen-release-artifact-check-v1",
                                 f"{phase} artifact check")
        if (checked.get("valid") is not True or
                checked.get("manifest_sha256") !=
                bindings["artifact_manifest_sha256"]):
            raise ValueError(f"{phase} frozen-artifact check did not pass")
    roles = artifact.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("artifact manifest has no executable roles")
    candidate_hash = roles.get("candidate", {}).get("sha256")
    oracle_hash = roles.get("oracle", {}).get("sha256")
    if not isinstance(candidate_hash, str) or not isinstance(oracle_hash, str):
        raise ValueError("artifact executable roles have no hashes")
    expected = {record["project_id"] for record in selection["selected"]}
    input_records = inputs.get("inputs")
    if not isinstance(input_records, list):
        raise ValueError("sealed inputs have no inputs array")
    sealed_by_project = {record["project_id"]: record
                         for record in input_records
                         if isinstance(record, dict) and
                         isinstance(record.get("project_id"), str)}
    if (len(sealed_by_project) != len(input_records) or
            set(sealed_by_project) != expected):
        raise ValueError("sealed inputs do not account exactly once for every project")
    attempts = run_result.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("run report has no attempts array")
    by_project: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        project_id = attempt.get("project_id")
        if project_id not in expected or project_id in by_project:
            raise ValueError("run report has an unknown or duplicate project attempt")
        by_project[project_id] = attempt
    outcomes = []
    for chosen in selection["selected"]:
        attempt = by_project.get(chosen["project_id"], {})
        sealed = sealed_by_project.get(chosen["project_id"], {})
        report = None
        failure_reason = attempt.get("reason") or attempt.get("status") or \
            "missing attempt"
        report_path_text = attempt.get("report_path")
        if (attempt.get("project_id") == chosen["project_id"] and
                attempt.get("tile_id") == chosen["tile_id"] and
                attempt.get("tile_url") == chosen["tile_url"] and
                sealed.get("status") == "sealed" and
                sealed.get("tile_id") == chosen["tile_id"] and
                sealed.get("tile_url") == chosen["tile_url"] and
                isinstance(report_path_text, str)):
            report_path = pathlib.Path(report_path_text)
            if (report_path.is_file() and
                    attempt.get("report_sha256") == sha256(report_path)):
                try:
                    report = require_schema(
                        load(report_path), "pdg-reference-pipeline-baseline-v1",
                        f"benchmark report for {chosen['project_id']}")
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    failure_reason = str(error)
            else:
                failure_reason = "raw benchmark report is missing or changed"
        elif attempt:
            failure_reason = "attempt or sealed input does not match selected tile"
        speedup = None
        exact = False
        raw_valid = False
        if report is not None:
            comparison = report.get("comparison", {})
            fixture = report.get("fixture", {}).get("laz", {})
            pipeline = report.get("pipeline", {})
            binaries = report.get("binaries", {})
            runs = report.get("runs", [])
            measured = [record for record in runs
                        if isinstance(record, dict) and
                        record.get("phase") == "measured"]
            warmups = [record for record in runs
                       if isinstance(record, dict) and
                       record.get("phase") == "warmup"]

            def valid_run_record(record: dict[str, Any]) -> bool:
                seconds = record.get("seconds")
                artifacts = record.get("artifacts")
                return bool(
                    record.get("label") in {"oracle", "candidate"} and
                    record.get("returncode") == 0 and
                    isinstance(seconds, (int, float)) and
                    not isinstance(seconds, bool) and
                    math.isfinite(seconds) and seconds > 0.0 and
                    isinstance(artifacts, list) and artifacts and
                    all(isinstance(item, dict) and
                        isinstance(item.get("name"), str) and
                        isinstance(item.get("bytes"), int) and
                        item.get("bytes") >= 0 and
                        isinstance(item.get("sha256"), str) and
                        len(item["sha256"]) == 64 for item in artifacts) and
                    record.get("missing_artifact_patterns") == [] and
                    isinstance(record.get("stdout_sha256"), str) and
                    isinstance(record.get("stderr_sha256"), str))

            by_role = {role: [record for record in measured
                              if record.get("label") == role]
                       for role in ("oracle", "candidate")}
            warmup_counts = {role: sum(record.get("label") == role
                                       for record in warmups)
                             for role in ("oracle", "candidate")}
            timing_records_valid = bool(
                isinstance(runs, list) and len(runs) == 8 and
                len(measured) == 6 and len(warmups) == 2 and
                warmup_counts == {"oracle": 1, "candidate": 1} and
                all(valid_run_record(record) for record in runs) and
                all({record.get("iteration") for record in by_role[role]} ==
                    {0, 1, 2} for role in ("oracle", "candidate")))
            recomputed_speedup = None
            if timing_records_valid:
                oracle_median = statistics.median(
                    record["seconds"] for record in by_role["oracle"])
                candidate_median = statistics.median(
                    record["seconds"] for record in by_role["candidate"])
                recomputed_speedup = oracle_median / candidate_median
            reported_speedup = comparison.get("median_speedup")
            speedup_matches = bool(
                isinstance(reported_speedup, (int, float)) and
                not isinstance(reported_speedup, bool) and
                math.isfinite(reported_speedup) and
                recomputed_speedup is not None and
                math.isclose(reported_speedup, recomputed_speedup,
                             rel_tol=1e-12, abs_tol=1e-12))

            def artifact_signature(record: dict[str, Any]) -> tuple[Any, ...]:
                return tuple((item["name"], item["bytes"], item["sha256"])
                             for item in record["artifacts"])

            measured_outputs_exact = False
            if timing_records_valid:
                oracle_signatures = {
                    artifact_signature(record)
                    for record in by_role["oracle"]}
                candidate_signatures = {
                    artifact_signature(record)
                    for record in by_role["candidate"]}
                measured_outputs_exact = bool(
                    len(oracle_signatures) == 1 and
                    oracle_signatures == candidate_signatures and
                    len({record["stdout_sha256"] for record in
                         by_role["oracle"]}) == 1 and
                    {record["stdout_sha256"] for record in
                     by_role["oracle"]} ==
                    {record["stdout_sha256"] for record in
                     by_role["candidate"]} and
                    len({record["stderr_sha256"] for record in
                         by_role["oracle"]}) == 1 and
                    {record["stderr_sha256"] for record in
                     by_role["oracle"]} ==
                    {record["stderr_sha256"] for record in
                     by_role["candidate"]})
            speedup = recomputed_speedup
            exact = (comparison.get("exact_outputs") is True and
                     measured_outputs_exact)
            raw_valid = bool(
                report.get("label") == f"3dep-{chosen['project_id']}" and
                report.get("failures") == [] and exact and
                comparison.get("contract") == "exact" and
                pipeline.get("sha256") == prereg["pipeline"]["sha256"] and
                fixture.get("sha256") == sealed.get("sha256") and
                binaries.get("candidate", {}).get("sha256") == candidate_hash and
                binaries.get("oracle", {}).get("sha256") == oracle_hash and
                timing_records_valid and speedup_matches)
            if not raw_valid:
                failure_reason = "raw benchmark report failed a frozen protocol binding"
        success = bool(attempt.get("returncode") == 0 and raw_valid and
                       isinstance(speedup, (int, float)) and speedup >= 10.0 and
                       run_result.get("artifact_valid_before") is True and
                       run_result.get("artifact_valid_after") is True)
        outcomes.append({"project_id": chosen["project_id"], "success": success,
                         "median_speedup": speedup,
                         "exact_outputs": exact,
                         "raw_report_valid": raw_valid,
                         "reason": None if success else failure_reason})
    successes = sum(record["success"] for record in outcomes)
    total = int(prereg["sample_size"])
    if len(outcomes) != total:
        raise ValueError("analysis did not account for the preregistered sample")
    lower = clopper_pearson_lower(successes, total)
    accepted = lower > 0.90
    result = {"schema": ANALYSIS_SCHEMA, "study_id": prereg["study_id"],
              "successes": successes, "attempts": total,
              "observed_fraction": successes / total,
              "one_sided_confidence": 0.95,
              "clopper_pearson_lower": lower, "threshold": 0.90,
              "claim_accepted": accepted, "outcomes": outcomes,
              "bindings": bindings,
              "rule": "All preregistered projects are counted; missing, failed, inexact, or <10x attempts are failures."}
    write_json(args.output, result)
    print(f"{successes}/{total}; one-sided 95% lower bound {lower:.6f}; "
          f"claim {'PASS' if accepted else 'FAIL'}")
    return 0 if accepted else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("select", allow_abbrev=False)
    selection.add_argument("--preregistration", required=True, type=pathlib.Path)
    selection.add_argument("--catalog", required=True, type=pathlib.Path)
    selection.add_argument("--output", required=True, type=pathlib.Path)
    sealing = sub.add_parser("seal-inputs", allow_abbrev=False)
    sealing.add_argument("--selection", required=True, type=pathlib.Path)
    sealing.add_argument("--mapping", required=True, type=pathlib.Path)
    sealing.add_argument("--output", required=True, type=pathlib.Path)
    execution = sub.add_parser("run", allow_abbrev=False)
    execution.add_argument("--preregistration", required=True, type=pathlib.Path)
    execution.add_argument("--selection", required=True, type=pathlib.Path)
    execution.add_argument("--inputs", required=True, type=pathlib.Path)
    execution.add_argument("--payload", required=True, type=pathlib.Path)
    execution.add_argument("--artifact-manifest", required=True, type=pathlib.Path)
    execution.add_argument("--output-dir", required=True, type=pathlib.Path)
    execution.add_argument("--timeout-seconds", type=int, default=14400)
    analysis = sub.add_parser("analyze", allow_abbrev=False)
    analysis.add_argument("--preregistration", required=True, type=pathlib.Path)
    analysis.add_argument("--catalog", required=True, type=pathlib.Path)
    analysis.add_argument("--selection", required=True, type=pathlib.Path)
    analysis.add_argument("--inputs", required=True, type=pathlib.Path)
    analysis.add_argument("--artifact-manifest", required=True,
                          type=pathlib.Path)
    analysis.add_argument("--run-report", required=True, type=pathlib.Path)
    analysis.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        return {"select": select, "seal-inputs": seal_inputs,
                "run": run, "analyze": analyze}[args.command](args)
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"3DEP study error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
