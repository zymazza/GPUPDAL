#!/usr/bin/env python3
"""Contract tests for the preregistered project-level 3DEP study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile


def invoke(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=pathlib.Path)
    parser.add_argument("--preregistration", required=True, type=pathlib.Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("three_dep_study", args.tool)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.clopper_pearson_lower(29, 29) > 0.90
    assert module.clopper_pearson_lower(28, 29) < 0.90
    assert 0.927 < module.clopper_pearson_lower(40, 40) < 0.929
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        projects = []
        for index in range(60):
            projects.append({"project_id": f"project-{index:03d}",
                             "title": f"Project {index}",
                             "tiles": [{"tile_id": "b", "url":
                                        f"https://example.invalid/{index}/b.laz"},
                                       {"tile_id": "a", "url":
                                        f"https://example.invalid/{index}/a.copc.laz"}]})
        catalog = root / "catalog.json"
        catalog.write_text(json.dumps({"projects": projects}) + "\n")
        first, second = root / "first.json", root / "second.json"
        for output in (first, second):
            completed = invoke(sys.executable, str(args.tool), "select",
                               "--preregistration", str(args.preregistration),
                               "--catalog", str(catalog), "--output", str(output))
            assert completed.returncode == 0, completed.stderr
        one, two = json.loads(first.read_text()), json.loads(second.read_text())
        assert one == two and len(one["selected"]) == 40
        assert len({record["project_id"] for record in one["selected"]}) == 40
        fixture = root / "fixture.laz"
        fixture.write_bytes(b"frozen selected tile bytes")
        mapping = root / "mapping.json"
        mapping.write_text(json.dumps({"inputs": [
            {"project_id": chosen["project_id"], "tile_id": chosen["tile_id"],
             "tile_url": chosen["tile_url"], "path": str(fixture)}
            for chosen in one["selected"]]}) + "\n")
        wrong = json.loads(mapping.read_text())
        wrong["inputs"][0]["tile_url"] += ".substituted"
        wrong_mapping = root / "wrong-mapping.json"
        wrong_mapping.write_text(json.dumps(wrong) + "\n")
        completed = invoke(sys.executable, str(args.tool), "seal-inputs",
                           "--selection", str(first), "--mapping",
                           str(wrong_mapping), "--output", str(root / "wrong.json"))
        assert completed.returncode == 2
        inputs_path = root / "inputs.json"
        completed = invoke(sys.executable, str(args.tool), "seal-inputs",
                           "--selection", str(first), "--mapping", str(mapping),
                           "--output", str(inputs_path))
        assert completed.returncode == 0, completed.stderr
        inputs = json.loads(inputs_path.read_text())
        artifact = root / "artifact.json"
        candidate_hash, oracle_hash = "c" * 64, "o" * 64
        artifact.write_text(json.dumps({
            "schema": "pdg-frozen-release-artifact-v1",
            "payload": {"files": [{
                "path": "evidence/bench/3dep/preregistration-v1.json",
                "sha256": digest(args.preregistration)}]},
            "roles": {"candidate": {"sha256": candidate_hash},
                      "oracle": {"sha256": oracle_hash}}
        }) + "\n")
        artifact_hash = digest(artifact)
        checks = {}
        for phase in ("before", "after"):
            path = root / f"artifact-{phase}.json"
            path.write_text(json.dumps({
                "schema": "pdg-frozen-release-artifact-check-v1",
                "valid": True, "manifest_sha256": artifact_hash}) + "\n")
            checks[phase] = {"path": str(path), "sha256": digest(path)}
        attempts = []
        for chosen in one["selected"]:
            report_path = root / f"raw-{chosen['selection_index']:02d}.json"
            runs = []
            for role in ("oracle", "candidate"):
                runs.append({"label": role, "phase": "warmup",
                             "seconds": 10.0 if role == "oracle" else 1.0,
                             "returncode": 0, "artifacts": [{"name": "output.laz",
                             "bytes": 123, "sha256": "a" * 64}],
                             "missing_artifact_patterns": [],
                             "stdout_sha256": "b" * 64,
                             "stderr_sha256": "c" * 64})
            for iteration in range(3):
                for role in ("oracle", "candidate"):
                    runs.append({"label": role, "phase": "measured",
                                 "iteration": iteration,
                                 "seconds": 10.0 if role == "oracle" else 1.0,
                                 "returncode": 0, "artifacts": [{
                                     "name": "output.laz", "bytes": 123,
                                     "sha256": "a" * 64}],
                                 "missing_artifact_patterns": [],
                                 "stdout_sha256": "b" * 64,
                                 "stderr_sha256": "c" * 64})
            report_path.write_text(json.dumps({
                "schema": "pdg-reference-pipeline-baseline-v1",
                "label": f"3dep-{chosen['project_id']}", "failures": [],
                "comparison": {"contract": "exact", "exact_outputs": True,
                               "median_speedup": 10.0},
                "pipeline": {"sha256":
                             "843223e15f866ece2eb807982ea75bb23f22e8fd56435d4ba67acc1ace223c70"},
                "fixture": {"laz": {"sha256": digest(fixture)}},
                "binaries": {"candidate": {"sha256": candidate_hash},
                             "oracle": {"sha256": oracle_hash}},
                "runs": runs,
            }) + "\n")
            attempts.append({"project_id": chosen["project_id"],
                             "tile_id": chosen["tile_id"],
                             "tile_url": chosen["tile_url"],
                             "returncode": 0, "status": "completed",
                             "report_path": str(report_path),
                             "report_sha256": digest(report_path)})
        run_report = root / "run.json"
        run_report.write_text(json.dumps({
            "schema": "pdg-3dep-run-v1", "artifact_valid_before": True,
            "artifact_valid_after": True,
            "preregistration_sha256": digest(args.preregistration),
            "selection_sha256": digest(first),
            "inputs_sha256": digest(inputs_path),
            "artifact_manifest_sha256": artifact_hash,
            "artifact_checks": checks, "attempts": attempts,
        }) + "\n")
        analysis = root / "analysis.json"
        completed = invoke(sys.executable, str(args.tool), "analyze",
                           "--preregistration", str(args.preregistration),
                           "--catalog", str(catalog),
                           "--selection", str(first), "--run-report",
                           str(run_report), "--inputs", str(inputs_path),
                           "--artifact-manifest", str(artifact),
                           "--output", str(analysis))
        assert completed.returncode == 0, completed.stderr
        result = json.loads(analysis.read_text())
        assert result["successes"] == 40 and result["claim_accepted"] is True
        forged_selection = json.loads(first.read_text())
        forged_selection["selected"][0]["project_score"] = "0" * 64
        forged_selection_path = root / "forged-selection.json"
        forged_selection_path.write_text(json.dumps(forged_selection) + "\n")
        completed = invoke(sys.executable, str(args.tool), "analyze",
                           "--preregistration", str(args.preregistration),
                           "--catalog", str(catalog), "--selection",
                           str(forged_selection_path), "--run-report",
                           str(run_report), "--inputs", str(inputs_path),
                           "--artifact-manifest", str(artifact),
                           "--output", str(analysis))
        assert completed.returncode == 2
        raw_path = pathlib.Path(attempts[0]["report_path"])
        raw_value = json.loads(raw_path.read_text())
        raw_value["comparison"]["median_speedup"] = 9.999
        for record in raw_value["runs"]:
            if (record["phase"] == "measured" and
                    record["label"] == "candidate"):
                record["seconds"] = 10.0 / 9.999
        raw_path.write_text(json.dumps(raw_value) + "\n")
        run_value = json.loads(run_report.read_text())
        run_value["attempts"][0]["report_sha256"] = digest(raw_path)
        run_value["attempts"][0]["median_speedup"] = 1000.0
        run_value["attempts"][0]["exact_outputs"] = True
        run_report.write_text(json.dumps(run_value) + "\n")
        completed = invoke(sys.executable, str(args.tool), "analyze",
                           "--preregistration", str(args.preregistration),
                           "--catalog", str(catalog),
                           "--selection", str(first), "--run-report",
                           str(run_report), "--inputs", str(inputs_path),
                           "--artifact-manifest", str(artifact),
                           "--output", str(analysis))
        assert completed.returncode == 1
        assert json.loads(analysis.read_text())["successes"] == 39
    print("3DEP preregistration contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
