#!/usr/bin/env python3
"""Create a portable local exactness and performance verification report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import pathlib
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "pdg-verification-report-v1"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: pathlib.Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def run(command: list[str], *, environment: dict[str, str] | None = None,
        timeout: int = 120, max_output_bytes: int = 1 << 20) -> dict[str, Any]:
    try:
        with tempfile.TemporaryFile() as stdout_file, \
                tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command, stdout=stdout_file, stderr=stderr_file,
                stdin=subprocess.DEVNULL, env=environment,
                start_new_session=True)
            timed_out = False
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                returncode = 124
            stdout_bytes, stderr_bytes = stdout_file.tell(), stderr_file.tell()
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(max_output_bytes).decode("utf-8", "replace")
            stderr = stderr_file.read(max_output_bytes).decode("utf-8", "replace")
        if timed_out:
            stderr += f"\npdg verify timeout after {timeout}s\n"
        return {"command": command, "returncode": returncode,
                "stdout": stdout, "stderr": stderr,
                "stdout_bytes": stdout_bytes, "stderr_bytes": stderr_bytes,
                "output_truncated": stdout_bytes > max_output_bytes or
                                    stderr_bytes > max_output_bytes}
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": command, "error": str(error), "returncode": 126,
                "stdout": "", "stderr": str(error)}


def clean_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("PDG_", "PDAL_TEST_", "LD_"))}
    environment.update({"LC_ALL": "C", "TZ": "UTC"})
    return environment


def find_helper(candidate: pathlib.Path, name: str) -> pathlib.Path:
    prefix = candidate.parent.parent
    choices = [candidate.parent / name, prefix / "libexec" / "pdg" / name]
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    raise ValueError(f"verification installation lacks {name}")


def find_frozen_time(candidate: pathlib.Path,
                     supplied: pathlib.Path | None) -> pathlib.Path:
    if supplied:
        choices = [supplied]
    else:
        prefix = candidate.parent.parent
        choices = [candidate.parent / "libpdg_frozen_time.so",
                   prefix / "lib" / "libpdg_frozen_time.so",
                   prefix / "lib64" / "libpdg_frozen_time.so",
                   prefix / "libexec" / "pdg" / "libpdg_frozen_time.so"]
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    raise ValueError("verification requires the packaged frozen-time library")


def generate_input(oracle: pathlib.Path, output: pathlib.Path,
                   environment: dict[str, str], points: int) -> dict[str, Any]:
    pipeline_path = output.with_suffix(".pipeline.json")
    pipeline = {"pipeline": [
        {"type": "readers.faux", "bounds": "([0,1000],[0,1000],[0,100])",
         "count": points, "mode": "ramp"},
        {"type": "writers.las", "filename": str(output),
         "minor_version": 4, "dataformat_id": 7,
         "scale_x": 0.001, "scale_y": 0.001, "scale_z": 0.001},
    ]}
    pipeline_path.write_text(json.dumps(pipeline, indent=2) + "\n")
    result = run([str(oracle), "pipeline", str(pipeline_path)],
                 environment=environment, timeout=300)
    if result["returncode"] != 0 or not output.is_file():
        raise ValueError("pinned oracle could not generate the verification input: "
                         + result["stderr"][:500])
    return {"source": "deterministic readers.faux verification fixture",
            "generator_pipeline": file_record(pipeline_path),
            "artifact": file_record(output)}


def default_pipeline(path: pathlib.Path) -> None:
    value = {"pipeline": [
        {"type": "readers.las", "filename": "input.laz"},
        {"type": "filters.normal", "knn": 8},
        {"type": "filters.covariancefeatures", "knn": 8,
         "feature_set": "Dimensionality"},
        {"type": "writers.las", "filename": "output.laz",
         "compression": "true", "extra_dims": "all"},
    ]}
    path.write_text(json.dumps(value, indent=2) + "\n")


def diagnostic(command: list[str], environment: dict[str, str]) -> dict[str, Any]:
    value = run(command, environment=environment, timeout=60)
    # Preserve portable evidence without allowing an unbounded diagnostic to
    # dominate the report.
    value["stdout"] = value["stdout"][:20000]
    value["stderr"] = value["stderr"][:20000]
    return value


def machine_record(environment: dict[str, str]) -> dict[str, Any]:
    probes = {}
    for name, command in (
        ("cpu", ["lscpu"]), ("memory", ["free", "-h"]),
        ("gpu", ["nvidia-smi", "--query-gpu=name,uuid,driver_version,"
                 "memory.total,power.limit", "--format=csv,noheader"]),
        ("cuda_compiler", ["nvcc", "--version"]),
    ):
        if shutil.which(command[0]):
            probes[name] = diagnostic(command, environment)
        else:
            probes[name] = {"available": False, "command": command}
    return {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None,
            "probes": probes}


def self_contained_html(report: dict[str, Any]) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True)
    result = report["result"]
    status = "PASS" if result["valid"] else "FAIL"
    speed = result.get("median_speedup")
    speed_text = "n/a" if speed is None else f"{speed:.3f}x"
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PDG verification report</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#18212b}h1{margin-bottom:.25rem}.pass{color:#087830}.fail{color:#b42318}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:1rem;border-radius:.5rem}dl{display:grid;grid-template-columns:max-content 1fr;gap:.4rem 1rem}dt{font-weight:700}</style>
</head><body><h1>PDG verification: <span class="%s">%s</span></h1>
<p>Portable, author-run local evidence. This artifact is not third-party validation.</p>
<dl><dt>Exact contract</dt><dd>%s</dd><dt>Median speedup</dt><dd>%s</dd><dt>Schema</dt><dd>%s</dd></dl>
<h2>Complete JSON evidence</h2><pre>%s</pre></body></html>
""" % ("pass" if result["valid"] else "fail", status,
       "PASS" if result["exact"] else "FAIL", speed_text, SCHEMA,
       html.escape(payload))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--candidate", required=True, type=pathlib.Path,
                        help=argparse.SUPPRESS)
    parser.add_argument("--oracle", required=True, type=pathlib.Path,
                        help=argparse.SUPPRESS)
    parser.add_argument("--oracle-source", required=True,
                        choices=("sibling-pdal", "accepted-PDG_ORACLE_PDAL"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--accept-configured-oracle", action="store_true",
                        help="explicitly use and record PDG_ORACLE_PDAL; never accepted silently")
    parser.add_argument("--input", type=pathlib.Path,
                        help="LAS/LAZ input; default is a deterministic 250k-point fixture")
    parser.add_argument("--pipeline", type=pathlib.Path,
                        help="pipeline with input.laz/output.laz placeholders; default is r6")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("pdg-verification"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--points", type=int, default=250000,
                        help="generated-fixture size when --input is omitted")
    parser.add_argument("--cache-state", choices=("warm", "cold"), default="warm")
    parser.add_argument("--frozen-time-library", type=pathlib.Path)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.runs < 3 or args.warmups < 1:
            raise ValueError("qualifying verification requires at least 3 runs and 1 warmup")
        if args.points < 1 or args.points > 10_000_000:
            raise ValueError("--points must be in [1, 10000000]")
        candidate, oracle = args.candidate.resolve(), args.oracle.resolve()
        if not candidate.is_file() or not oracle.is_file():
            raise ValueError("candidate and pinned oracle must both exist")
        runner = find_helper(candidate, "pdg-benchmark-reference.py")
        frozen = find_frozen_time(candidate, args.frozen_time_library)
        output_dir = args.output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("output directory must be absent or empty so stale "
                             "evidence cannot enter a verification report")
        output_dir.mkdir(parents=True, exist_ok=True)
        environment = clean_environment()
        if args.input:
            input_path = args.input.resolve()
            if not input_path.is_file():
                raise ValueError(f"input does not exist: {input_path}")
            input_provenance = {"source": "user-supplied", "artifact":
                                file_record(input_path)}
        else:
            input_path = output_dir / "verification-input.las"
            input_provenance = generate_input(oracle, input_path, environment,
                                              args.points)
        if args.pipeline:
            pipeline = args.pipeline.resolve()
            if not pipeline.is_file():
                raise ValueError(f"pipeline does not exist: {pipeline}")
            pipeline_source = "user-supplied"
        else:
            pipeline = output_dir / "verification-r6.json"
            default_pipeline(pipeline)
            pipeline_source = "built-in r6 normal/covariance workflow"
        tracked_paths = [candidate, oracle, runner, frozen]
        engine = candidate.with_name("pdg-engine")
        if engine.is_file():
            tracked_paths.append(engine)
        before = {str(path): file_record(path) for path in tracked_paths}
        oracle_version = diagnostic([str(oracle), "--version"], environment)
        oracle_version_matches = bool(
            oracle_version.get("returncode") == 0 and
            "2.10.0" in (str(oracle_version.get("stdout", "")) +
                         str(oracle_version.get("stderr", ""))))
        oracle_source_accepted = bool(
            args.oracle_source == "sibling-pdal" or
            (args.oracle_source == "accepted-PDG_ORACLE_PDAL" and
             args.accept_configured_oracle))
        benchmark_report = output_dir / "benchmark.json"
        benchmark_work = output_dir / "benchmark-work"
        command = [sys.executable, str(runner), "--oracle", str(oracle),
                   "--candidate", str(candidate), "--pipeline", str(pipeline),
                   "--fixture-laz", str(input_path), "--fixture",
                   f"input.laz={input_path}", "--label", "pdg-verify-r6",
                   "--work-dir", str(benchmark_work), "--report",
                   str(benchmark_report), "--runs", str(args.runs),
                   "--warmups", str(args.warmups), "--cache-state",
                   args.cache_state, "--contract", "exact",
                   "--candidate-env", f"PDG_ORACLE_PDAL={oracle}",
                   "--frozen-time-library", str(frozen), "--freeze-epoch",
                   "1704067200"]
        completed = run(command, environment=environment,
                        timeout=args.timeout_seconds)
        benchmark = json.loads(benchmark_report.read_text()) \
            if benchmark_report.is_file() else None
        after = {str(path): file_record(path) for path in tracked_paths}
        drift = [path for path in before if before[path] != after[path]]
        exact = bool(benchmark and benchmark.get("comparison", {}).get(
            "exact_outputs") is True)
        speedup = (benchmark or {}).get("comparison", {}).get("median_speedup")
        valid = (completed["returncode"] == 0 and exact and not drift and
                 oracle_version_matches and oracle_source_accepted)
        report = {
            "schema": SCHEMA,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": "one local complete-process differential verification; not a population study or third-party validation",
            "result": {"valid": valid, "exact": exact,
                       "median_speedup": speedup,
                       "binary_drift_detected": bool(drift)},
            "conformance": {"mode": "default exact",
                            "comparator": "byte equality plus streams and exit status",
                            "benchmark_returncode": completed["returncode"],
                            "failures": (benchmark or {}).get("failures")},
            "oracle_provenance": {
                "source": args.oracle_source,
                "configured_override_explicitly_accepted":
                    oracle_source_accepted
                    if args.oracle_source == "accepted-PDG_ORACLE_PDAL"
                    else None,
                "expected_version": "2.10.0",
                "version_matches": oracle_version_matches,
                "binary": file_record(oracle),
                "limitation": "The report binds the complete oracle binary hash; commit provenance requires the release artifact manifest or an independently retained build record.",
            },
            "performance": {"scope": "whole-process read/compute/write",
                            "warmups": args.warmups, "measured_pairs": args.runs,
                            "cache_state": args.cache_state,
                            "median_speedup": speedup,
                            "placement_decision": (benchmark or {}).get(
                                "environment", {}).get("candidate_placement_profile")},
            "input": input_provenance,
            "pipeline": {"source": pipeline_source, **file_record(pipeline)},
            "binaries": {"before": before, "after": after,
                         "changed": drift,
                         "benchmark_manifest": (benchmark or {}).get("binaries")},
            "software": {"candidate_version": diagnostic(
                              [str(candidate), "version"], environment),
                          "oracle_version": oracle_version,
                          "doctor": diagnostic([str(candidate), "doctor"], environment),
                          "placement_status": diagnostic(
                              [str(candidate), "calibrate", "--status"], environment)},
            "machine": machine_record(environment),
            "raw_benchmark": {"path": str(benchmark_report),
                              "sha256": sha256(benchmark_report)
                              if benchmark_report.is_file() else None,
                              "report": benchmark},
            "rerun": [str(candidate), "verify", "--input", str(input_path),
                      "--pipeline", str(pipeline), "--output-dir",
                      str(output_dir / "rerun"), "--runs", str(args.runs), "--warmups",
                      str(args.warmups), "--cache-state", args.cache_state,
                      *(["--accept-configured-oracle"]
                        if args.oracle_source == "accepted-PDG_ORACLE_PDAL"
                        else [])],
            "external_validation": {"status": "not-performed",
                                    "statement": "This command prepares portable evidence; it does not claim an unrelated party ran it."},
        }
        json_path = output_dir / "pdg-verification.json"
        html_path = output_dir / "pdg-verification.html"
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        html_path.write_text(self_contained_html(report))
        print(f"PDG verification {'PASS' if valid else 'FAIL'}")
        print(f"JSON: {json_path}")
        print(f"HTML: {html_path}")
        if speedup is not None:
            print(f"median whole-process speedup: {speedup:.3f}x")
        return 0 if valid else 1
    except (OSError, ValueError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"gpupdal verify: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
