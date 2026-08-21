#!/usr/bin/env python3
"""Small physical contract for the explicit COPC semantic comparator."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile


def execute(pdal: pathlib.Path, pipeline: pathlib.Path) -> None:
    completed = subprocess.run([str(pdal), "pipeline", str(pipeline)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               check=False, text=True)
    assert completed.returncode == 0, completed.stderr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdal", required=True, type=pathlib.Path)
    parser.add_argument("--comparator", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pdg-copc-semantic-test-") as text:
        root = pathlib.Path(text)
        def make(name: str, count: int) -> pathlib.Path:
            output = root / f"{name}.copc.laz"
            pipeline = root / f"{name}.json"
            pipeline.write_text(json.dumps({"pipeline": [
                {"type": "readers.faux", "count": count,
                 "bounds": "([0,31],[0,31],[0,7])", "mode": "ramp"},
                {"type": "writers.copc", "filename": str(output),
                 "threads": 1, "fixed_seed": True},
            ]}))
            execute(args.pdal, pipeline)
            return output
        left = make("left", 257)
        right = make("right", 257)
        different = make("different", 258)
        passed = root / "passed.json"
        completed = subprocess.run([
            sys.executable, str(args.comparator), "--pdal", str(args.pdal),
            "--expected", str(left), "--actual", str(right),
            "--report", str(passed)], check=False)
        assert completed.returncode == 0
        report = json.loads(passed.read_text())
        assert report["schema"] == "pdg-copc-canonical-v1"
        assert report["semantic_equal"] is True
        assert report["oracle"]["canonical_records"]["point_count"] == 257
        assert report["oracle"]["hierarchy"]["valid"] is True
        assert report["physical_differences"] == []
        assert report["policy"]["diagnostic_only"] == [
            "hierarchy.nodes_sha256", "queries.coarse-resolution"]
        failed = root / "failed.json"
        completed = subprocess.run([
            sys.executable, str(args.comparator), "--pdal", str(args.pdal),
            "--expected", str(left), "--actual", str(different),
            "--report", str(failed)], check=False)
        assert completed.returncode == 1
        report = json.loads(failed.read_text())
        assert report["semantic_equal"] is False
        assert any(item["field"] in ("header", "canonical_records",
                                     "hierarchy.point_count_sum")
                   for item in report["differences"])
        capped = root / "capped.json"
        completed = subprocess.run([
            sys.executable, str(args.comparator), "--pdal", str(args.pdal),
            "--expected", str(left), "--actual", str(right),
            "--max-input-bytes", "1", "--report", str(capped)], check=False)
        assert completed.returncode == 1
        report = json.loads(capped.read_text())
        assert report["differences"][0]["field"] == "comparator_error"


if __name__ == "__main__":
    main()
