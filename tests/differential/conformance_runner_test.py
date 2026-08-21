#!/usr/bin/env python3
"""Contract tests for the named PDG Conformance Suite runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import time


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False)
    assert completed.returncode == expected, (
        command, completed.returncode, completed.stdout, completed.stderr)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--recipes", required=True, type=pathlib.Path)
    parser.add_argument("--differential", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pdg-conformance-contract-") as text:
        root = pathlib.Path(text)
        full = root / "full.json"
        repeat = root / "repeat.json"
        run([sys.executable, str(args.runner), "generate", "--recipes",
             str(args.recipes), "--output", str(full)])
        run([sys.executable, str(args.runner), "generate", "--recipes",
             str(args.recipes), "--output", str(repeat)])
        assert full.read_bytes() == repeat.read_bytes()
        manifest = json.loads(full.read_text())
        assert manifest["schema"] == "pdg-conformance-manifest-v1"
        assert manifest["complete_case_count"] == 2048
        assert len(manifest["cases"]) == 2048
        assert manifest["cases"][-1]["id"] == "giant-cloud-1000000"
        assert len({case["id"] for case in manifest["cases"]}) == 2048
        assert all(case["determinism_repeats"] == 2
                   for case in manifest["cases"])
        one_point_covariance = next(
            case for case in manifest["cases"]
            if case["coverage"]["input"] == "one-point" and
            case["coverage"]["stage"] == "covariance-features")
        assert one_point_covariance["pipeline"][1]["where"] == "X > 0"
        ordered_ferry = next(
            case for case in manifest["cases"]
            if case["coverage"]["stage"] == "ordered-ferry")
        assert ordered_ferry["pipeline"][1]["dimensions"] == "Extra0=>Extra1"
        weird_scale = next(
            case for case in manifest["cases"]
            if case["coverage"]["writer"] == "weird-scale-offset")
        assert weird_scale["pipeline"][-1]["scale_x"] == 0.0000002
        run([sys.executable, str(args.runner), "validate", str(full)])

        invalid = dict(manifest)
        invalid["cases"] = [dict(manifest["cases"][0],
                                 comparison_mode="copc-canonical-v1")]
        invalid_path = root / "invalid.json"
        invalid_path.write_text(json.dumps(invalid))
        run([sys.executable, str(args.runner), "validate", str(invalid_path)], 2)

        invalid_status = dict(manifest)
        invalid_status["cases"] = [dict(manifest["cases"][0],
                                        expected_returncode=True)]
        invalid_status_path = root / "invalid-status.json"
        invalid_status_path.write_text(json.dumps(invalid_status))
        run([sys.executable, str(args.runner), "validate",
             str(invalid_status_path)], 2)

        smoke = root / "smoke.json"
        run([sys.executable, str(args.runner), "generate", "--recipes",
             str(args.recipes), "--output", str(smoke), "--limit", "2"])
        fake = root / "fake.py"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib,pathlib,sys\n"
            "if len(sys.argv) >= 3 and sys.argv[1] == 'pipeline':\n"
            " p=pathlib.Path(sys.argv[2]).read_bytes()\n"
            " i=pathlib.Path('input.csv').read_bytes() if pathlib.Path('input.csv').exists() else b''\n"
            " pathlib.Path('output.las').write_bytes(b'LASF'+hashlib.sha256(p+i).digest())\n"
            " print('Pipeline executed')\n"
            " sys.exit(0)\n"
            "sys.exit(2)\n")
        fake.chmod(0o755)
        report = root / "report.json"
        run([sys.executable, str(args.runner), "run", "--manifest", str(smoke),
             "--oracle", str(fake), "--candidate", str(fake),
             "--differential", str(args.differential), "--work-dir",
             str(root / "work"), "--report", str(report),
             "--allow-partial"])
        result = json.loads(report.read_text())
        assert result["schema"] == "pdg-conformance-report-v1"
        assert result["partial"] is True
        assert result["executed_cases"] == 2
        assert result["passed_cases"] == 2
        assert result["failed_cases"] == 0
        assert result["zero_unexplained_semantic_differences"] is False
        assert result["oracle"]["sha256"] == hashlib.sha256(
            fake.read_bytes()).hexdigest()
        assert all(case["product_statuses_match"] is True
                   for case in result["results"])

        home_required = root / "home-required.py"
        home_required.write_text(
            "#!/usr/bin/env python3\n"
            "import os,pathlib,sys\n"
            "home=pathlib.Path(os.environ['HOME'])\n"
            "xdg=pathlib.Path(os.environ['XDG_CONFIG_HOME'])\n"
            "assert home.is_dir() and xdg.is_dir() and xdg.parent == home\n"
            "pathlib.Path('output.bin').write_bytes(b'hermetic-home')\n"
            "sys.exit(0)\n")
        home_required.chmod(0o755)
        home_work = root / "home-work"
        run([sys.executable, str(args.differential), "--oracle",
             str(home_required), "--candidate", str(home_required), "--case",
             "hermetic-home", "--work-dir", str(home_work),
             "--allow-empty-command"])
        home_report = json.loads((home_work / "reports" /
                                  "hermetic-home.json").read_text())
        assert home_report["oracle_run"]["returncode"] == 0
        assert home_report["candidate_run"]["returncode"] == 0

        matched_failure = root / "matched-failure.py"
        matched_failure.write_text("#!/bin/sh\nexit 7\n")
        matched_failure.chmod(0o755)
        failure_manifest = json.loads(smoke.read_text())
        failure_manifest["cases"] = [failure_manifest["cases"][0]]
        failure_path = root / "matched-failure.json"
        failure_path.write_text(json.dumps(failure_manifest))
        failure_report = root / "matched-failure-report.json"
        run([sys.executable, str(args.runner), "run", "--manifest",
             str(failure_path), "--oracle", str(matched_failure),
             "--candidate", str(matched_failure), "--differential",
             str(args.differential), "--work-dir", str(root / "failure-work"),
             "--report", str(failure_report), "--allow-partial"], expected=1)
        failure_result = json.loads(failure_report.read_text())
        assert failure_result["passed_cases"] == 0
        assert failure_result["results"][0]["product_statuses_match"] is False

        failure_manifest["cases"][0]["expected_returncode"] = 7
        expected_failure_path = root / "expected-matched-failure.json"
        expected_failure_path.write_text(json.dumps(failure_manifest))
        expected_failure_report = root / "expected-matched-failure-report.json"
        run([sys.executable, str(args.runner), "run", "--manifest",
             str(expected_failure_path), "--oracle", str(matched_failure),
             "--candidate", str(matched_failure), "--differential",
             str(args.differential), "--work-dir",
             str(root / "expected-failure-work"), "--report",
             str(expected_failure_report), "--allow-partial"])
        expected_failure_result = json.loads(expected_failure_report.read_text())
        assert expected_failure_result["passed_cases"] == 1
        assert expected_failure_result["results"][0][
            "product_statuses_match"] is True

        capped_work = root / "capped"
        completed = run([
            sys.executable, str(args.differential), "--oracle", str(fake),
            "--candidate", str(fake), "--case", "artifact-cap",
            "--work-dir", str(capped_work), "--max-artifact-bytes", "8",
            "--seed-file", f"pipeline.json={root / 'work' / 'staged' / manifest['cases'][0]['id'] / 'pipeline.json'}",
            "--seed-file", f"input.csv={root / 'work' / 'staged' / manifest['cases'][0]['id'] / 'input.csv'}",
            "--", "pipeline", "pipeline.json"], expected=1)
        capped = json.loads((capped_work / "reports" /
                             "artifact-cap.json").read_text())
        assert {difference["artifact"] for difference in capped["differences"]} == {
            "oracle_resource_limit", "candidate_resource_limit"}

        noisy = root / "noisy.py"
        noisy.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "sys.stdout.write('x' * 4096)\n")
        noisy.chmod(0o755)
        stream_work = root / "stream-capped"
        run([sys.executable, str(args.differential), "--oracle", str(noisy),
             "--candidate", str(noisy), "--case", "stream-cap",
             "--work-dir", str(stream_work), "--max-stream-bytes", "64",
             "--allow-empty-command"], expected=1)
        stream_report = json.loads((stream_work / "reports" /
                                    "stream-cap.json").read_text())
        assert {difference["artifact"] for difference in
                stream_report["differences"]} >= {
                    "oracle_stdout_resource_limit",
                    "candidate_stdout_resource_limit"}

        detached = root / "detached.py"
        detached.write_text(
            "#!/usr/bin/env python3\n"
            "import os,time\n"
            "if os.fork() == 0:\n"
            " os.setsid()\n"
            " time.sleep(1)\n"
            " os._exit(0)\n"
            "time.sleep(10)\n")
        detached.chmod(0o755)
        started = time.monotonic()
        run([sys.executable, str(args.differential), "--oracle", str(detached),
             "--candidate", str(detached), "--case", "detached-timeout",
             "--work-dir", str(root / "detached-work"),
             "--timeout-seconds", "0.1", "--allow-empty-command"])
        assert time.monotonic() - started < 1.0


if __name__ == "__main__":
    main()
