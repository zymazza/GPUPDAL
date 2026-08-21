#!/usr/bin/env python3
"""Contract tests for the fourteen-workload reference-suite manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    return parser.parse_args()


ARGS = parse_args()


class ReferenceSuiteTest(unittest.TestCase):
    def manifest(self) -> dict:
        return json.loads(ARGS.manifest.read_text(encoding="utf-8"))

    def pipeline(self, relative: str) -> list[dict]:
        document = json.loads(
            (ARGS.repo_root / relative).read_text(encoding="utf-8"))
        self.assertIsInstance(document, dict)
        self.assertIsInstance(document.get("pipeline"), list)
        return document["pipeline"]

    def test_checked_in_manifest_is_complete(self) -> None:
        completed = subprocess.run(
            [
                sys.executable, str(ARGS.runner), "validate",
                "--manifest", str(ARGS.manifest),
                "--repo-root", str(ARGS.repo_root),
            ],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["headline_workloads"], 14)
        self.assertEqual(report["aggregate_weight"], 14)
        self.assertEqual(report["ids"], [
            "r1-translate", "r2-ground-normalize", "r3-dtm",
            "r4-denoise-thin", "r5-copc-query", "r6-features",
            "r7-dsm", "r8-colorize", "r9-polygon-clip",
            "r10-decimate", "r11-classify-refine", "r12-tile",
            "r13-merge", "r14-convert-compress",
        ])

    def test_new_workload_definitions_pin_required_semantics(self) -> None:
        manifest = self.manifest()
        workloads = {entry["id"]: entry for entry in manifest["workloads"]}

        r7 = self.pipeline(workloads["r7-dsm"]["pipeline"])
        self.assertEqual([stage["type"] for stage in r7], [
            "readers.las", "filters.returns", "writers.gdal"])
        self.assertEqual(r7[1]["groups"], "first,only")
        self.assertEqual(r7[2]["output_type"], "max")
        self.assertEqual(r7[2]["dimension"], "Z")
        self.assertTrue(r7[2]["binmode"])
        self.assertEqual(r7[2]["data_type"], "float64")
        self.assertEqual(r7[2]["nodata"], -9999.0)

        r8 = self.pipeline(workloads["r8-colorize"]["pipeline"])
        self.assertEqual([stage["type"] for stage in r8], [
            "readers.las", "filters.reprojection", "filters.colorization",
            "filters.reprojection", "writers.las"])
        self.assertEqual((r8[1]["in_srs"], r8[1]["out_srs"]),
                         ("EPSG:28992", "EPSG:3857"))
        self.assertEqual((r8[3]["in_srs"], r8[3]["out_srs"]),
                         ("EPSG:3857", "EPSG:28992"))
        self.assertEqual(r8[2]["raster"], "input.orthophoto.tif")
        self.assertEqual(r8[2]["dimensions"],
                         "Red:1:1.0, Green:2:1.0, Blue:3:1.0")

        r9 = self.pipeline(workloads["r9-polygon-clip"]["pipeline"])
        self.assertEqual(r9[1]["type"], "filters.crop")
        self.assertEqual(r9[1]["polygon"],
                         "REPLACE_CLIP_MULTIPOLYGON_WKT")
        self.assertEqual(r9[1]["a_srs"], "EPSG:4326")
        self.assertEqual(
            workloads["r9-polygon-clip"]["artifacts"][0]["expected_point_count"],
            473825)

        r10 = self.pipeline(workloads["r10-decimate"]["pipeline"])
        self.assertEqual(r10[1], {
            "type": "filters.voxelcentroidnearestneighbor", "cell": 2.5})
        decimation_variants = {
            entry["id"]: self.pipeline(entry["pipeline"])
            for entry in workloads["r10-decimate"]["variants"]}
        self.assertEqual(set(decimation_variants), {
            "r10-ordinal", "r10-radius", "r10-grid",
            "r10-voxel-center-nearest", "r10-voxel-center",
            "r10-density-aware"})
        variant_stage_types = {
            identifier: {stage["type"] for stage in pipeline}
            for identifier, pipeline in decimation_variants.items()}
        self.assertIn("filters.decimation", variant_stage_types["r10-ordinal"])
        self.assertIn("filters.sample", variant_stage_types["r10-radius"])
        self.assertIn("filters.gridDecimation", variant_stage_types["r10-grid"])
        self.assertIn("filters.voxelcenternearestneighbor",
                      variant_stage_types["r10-voxel-center-nearest"])
        self.assertIn("filters.voxeldownsize",
                      variant_stage_types["r10-voxel-center"])
        self.assertIn("filters.radialdensity",
                      variant_stage_types["r10-density-aware"])

        r11 = self.pipeline(workloads["r11-classify-refine"]["pipeline"])
        self.assertEqual([stage["type"] for stage in r11], [
            "readers.las", "filters.smrf", "filters.outlier",
            "filters.neighborclassifier", "writers.las"])
        self.assertEqual(r11[2]["method"], "statistical")
        self.assertEqual(r11[3]["k"], 7)
        self.assertEqual(r11[3]["domain"], "Classification[1:1]")

        r12 = self.pipeline(workloads["r12-tile"]["pipeline"])
        self.assertEqual(r12[1]["type"], "filters.splitter")
        self.assertEqual(r12[1]["length"], 256.0)
        self.assertEqual(r12[1]["origin_x"], "REPLACE_TILE_ORIGIN_X")
        self.assertEqual(r12[1]["origin_y"], "REPLACE_TILE_ORIGIN_Y")
        self.assertEqual(r12[2]["filename"], "output-tile-#.laz")

        r13 = self.pipeline(workloads["r13-merge"]["pipeline"])
        self.assertEqual(r13[0]["filename"], "input.merge-a.laz")
        self.assertEqual(r13[1]["filename"], "input.merge-b.laz")
        self.assertEqual(r13[2], {
            "type": "filters.merge", "inputs": ["left", "right"]})

        r14 = self.pipeline(workloads["r14-convert-compress"]["pipeline"])
        self.assertEqual(r14[0]["filename"], "input.las")
        self.assertEqual(r14[1]["filename"], "output.laz")
        self.assertEqual(r14[1]["compression"], "true")
        conversion_variants = {
            entry["id"]: self.pipeline(entry["pipeline"])
            for entry in workloads["r14-convert-compress"]["variants"]}
        self.assertEqual(set(conversion_variants), {
            "r14-laz-to-las", "r14-laz-recompress", "r14-las-to-copc",
            "r14-laz-to-copc", "r14-copc-to-las", "r14-copc-to-laz"})
        for identifier in ("r14-las-to-copc", "r14-laz-to-copc"):
            writer = conversion_variants[identifier][-1]
            self.assertEqual(writer["type"], "writers.copc")
            self.assertEqual(writer["threads"], 1)
            self.assertIs(writer["fixed_seed"], True)
        for identifier in ("r14-copc-to-las", "r14-copc-to-laz"):
            reader = conversion_variants[identifier][0]
            self.assertEqual(reader["type"], "readers.copc")
            self.assertEqual(reader["requests"], 1)

    def test_validator_refuses_mutated_headline_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-reference-suite-bad-") as temporary:
            manifest = self.manifest()
            manifest["workloads"][6]["headline_weight"] = 0
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable, str(ARGS.runner), "validate",
                    "--manifest", str(path),
                    "--repo-root", str(ARGS.repo_root),
                ],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("headline_weight exactly 1", completed.stderr)

    def make_report(self, path: Path, workload: int, speedup: float,
                    *, exact: bool = True) -> None:
        oracle = float(workload + 1)
        candidate = oracle / speedup
        artifacts = ([{"name": "output.laz", "las_point_count": 473825}]
                     if workload == 8 else [])
        runs = []
        for iteration in range(3):
            drift = 1.0 + iteration * 0.01
            runs.extend([
                {"label": "oracle", "phase": "measured",
                 "iteration": iteration, "seconds": oracle * drift,
                 "artifacts": artifacts},
                {"label": "candidate", "phase": "measured",
                 "iteration": iteration, "seconds": candidate * drift,
                 "artifacts": artifacts},
            ])
        path.write_text(json.dumps({
            "schema": "pdg-reference-pipeline-baseline-v1",
            "label": self.manifest()["workloads"][workload]["id"],
            "binaries": {
                "oracle": {"path": "oracle", "sha256": "a" * 64},
                "candidate": {"path": "candidate", "sha256": "b" * 64},
            },
            "environment": {"cache_state": "warm"},
            "summary": {
                "oracle": {"median_seconds": oracle * 1.01},
                "candidate": {"median_seconds": candidate * 1.01},
            },
            "comparison": {"exact_outputs": exact},
            "runs": runs,
        }), encoding="utf-8")

    def test_aggregate_reports_equal_workload_and_total_wall_speedups(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-reference-suite-") as temporary:
            reports = Path(temporary)
            manifest = self.manifest()
            for index, workload in enumerate(manifest["workloads"]):
                self.make_report(
                    reports / f"{workload['id']}-warm.json", index, 2.0)
            output = reports / "aggregate.json"
            completed = subprocess.run(
                [
                    sys.executable, str(ARGS.runner), "aggregate",
                    "--manifest", str(ARGS.manifest),
                    "--reports", str(reports),
                    "--cache-state", "warm", "--output", str(output),
                ],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            aggregate = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(aggregate["complete"])
            self.assertEqual(aggregate["workload_count"], 14)
            self.assertTrue(math.isclose(
                aggregate["equal_workload_geometric_mean_speedup"], 2.0,
                rel_tol=1e-12))
            self.assertTrue(math.isclose(
                aggregate["total_wall_time"]["speedup"], 2.0,
                rel_tol=1e-12))
            self.assertEqual(len(aggregate["paired_suite_round_speedups"]), 3)

    def test_aggregate_fails_closed_on_missing_or_inexact_workload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-reference-suite-") as temporary:
            reports = Path(temporary)
            manifest = self.manifest()
            for index, workload in enumerate(manifest["workloads"][:-1]):
                self.make_report(
                    reports / f"{workload['id']}-warm.json", index, 1.5,
                    exact=index != 4)
            output = reports / "aggregate.json"
            completed = subprocess.run(
                [
                    sys.executable, str(ARGS.runner), "aggregate",
                    "--manifest", str(ARGS.manifest),
                    "--reports", str(reports),
                    "--cache-state", "warm", "--output", str(output),
                ],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            aggregate = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(aggregate["complete"])
            self.assertIn("r14-convert-compress", aggregate["missing"])
            self.assertIn("r5-copc-query", aggregate["inexact"])

    def test_aggregate_refuses_binary_drift_and_r9_cardinality_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-reference-suite-") as temporary:
            reports = Path(temporary)
            manifest = self.manifest()
            for index, workload in enumerate(manifest["workloads"]):
                self.make_report(
                    reports / f"{workload['id']}-warm.json", index, 1.5)

            r7_path = reports / "r7-dsm-warm.json"
            r7 = json.loads(r7_path.read_text(encoding="utf-8"))
            r7["binaries"]["candidate"]["sha256"] = "c" * 64
            r7_path.write_text(json.dumps(r7), encoding="utf-8")
            output = reports / "aggregate-binary-drift.json"
            completed = subprocess.run([
                sys.executable, str(ARGS.runner), "aggregate",
                "--manifest", str(ARGS.manifest), "--reports", str(reports),
                "--cache-state", "warm", "--output", str(output),
            ], capture_output=True, text=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("one candidate binary hash",
                          result["invalid"]["aggregate"])

            r7["binaries"]["candidate"]["sha256"] = "b" * 64
            r7_path.write_text(json.dumps(r7), encoding="utf-8")
            r9_path = reports / "r9-polygon-clip-warm.json"
            r9 = json.loads(r9_path.read_text(encoding="utf-8"))
            r9["runs"][0]["artifacts"][0]["las_point_count"] = 473824
            r9_path.write_text(json.dumps(r9), encoding="utf-8")
            output = reports / "aggregate-cardinality-drift.json"
            completed = subprocess.run([
                sys.executable, str(ARGS.runner), "aggregate",
                "--manifest", str(ARGS.manifest), "--reports", str(reports),
                "--cache-state", "warm", "--output", str(output),
            ], capture_output=True, text=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("point count must be 473825",
                          result["invalid"]["r9-polygon-clip"])

    def test_aggregate_refuses_report_slot_identity_drift(self) -> None:
        mutations = {
            "schema": lambda report: report.update(schema="unknown-v1"),
            "label": lambda report: report.update(label="r8-colorize"),
            "cache": lambda report: report["environment"].update(
                cache_state="cold"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                    prefix="pdg-reference-suite-") as temporary:
                reports = Path(temporary)
                manifest = self.manifest()
                for index, workload in enumerate(manifest["workloads"]):
                    self.make_report(
                        reports / f"{workload['id']}-warm.json", index, 1.5)
                r7_path = reports / "r7-dsm-warm.json"
                r7 = json.loads(r7_path.read_text(encoding="utf-8"))
                mutate(r7)
                r7_path.write_text(json.dumps(r7), encoding="utf-8")
                output = reports / "aggregate.json"
                completed = subprocess.run([
                    sys.executable, str(ARGS.runner), "aggregate",
                    "--manifest", str(ARGS.manifest),
                    "--reports", str(reports), "--cache-state", "warm",
                    "--output", str(output),
                ], capture_output=True, text=True, check=False)
                self.assertNotEqual(completed.returncode, 0)
                result = json.loads(output.read_text(encoding="utf-8"))
                self.assertIn("r7-dsm", result["invalid"])

    def test_run_executes_all_headlines_and_writes_aggregate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-reference-suite-run-") as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            manifest = self.manifest()
            pipeline_paths = {
                workload["pipeline"] for workload in manifest["workloads"]}
            pipeline_paths.update(
                variant["pipeline"] for workload in manifest["workloads"]
                for variant in workload.get("variants", []))
            for relative in pipeline_paths:
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"pipeline": []}\n', encoding="utf-8")
            fixture_dir = repo / "fixtures"
            fixture_dir.mkdir()
            for identifier, fixture in manifest["fixtures"].items():
                path = fixture_dir / f"{identifier}.bin"
                payload = f"fixture:{identifier}".encode()
                path.write_bytes(payload)
                fixture.clear()
                fixture.update({
                    "kind": "test",
                    "path": str(path.relative_to(repo)),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "provenance": "bounded suite-run contract fixture",
                })
            manifest_path = repo / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            fake_runner = root / "fake-benchmark-runner"
            fake_runner.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "def value(name): return sys.argv[sys.argv.index(name) + 1]\n"
                "out = pathlib.Path(value('--report'))\n"
                "out.parent.mkdir(parents=True, exist_ok=True)\n"
                "artifacts = ([{'name':'output.laz','las_point_count':473825}]\n"
                "             if value('--label') == 'r9-polygon-clip' else [])\n"
                "out.write_text(json.dumps({\n"
                " 'schema':'pdg-reference-pipeline-baseline-v1',\n"
                " 'label':value('--label'),\n"
                " 'binaries':{'oracle':{'path':'oracle','sha256':'a'*64},\n"
                "             'candidate':{'path':'candidate','sha256':'b'*64}},\n"
                " 'environment':{'cache_state':value('--cache-state')},\n"
                " 'summary':{'oracle':{'median_seconds':2.0},\n"
                "            'candidate':{'median_seconds':1.0}},\n"
                " 'comparison':{'exact_outputs':True},\n"
                " 'runs':[{'label':'oracle','phase':'measured','iteration':0,'seconds':2.0,'artifacts':artifacts},\n"
                "         {'label':'candidate','phase':'measured','iteration':0,'seconds':1.0,'artifacts':artifacts}]\n"
                "}) + '\\n')\n",
                encoding="utf-8")
            fake_runner.chmod(0o755)
            oracle = root / "oracle"
            candidate = root / "candidate"
            for binary in (oracle, candidate):
                binary.write_bytes(b"#!/bin/sh\nexit 0\n")
                binary.chmod(0o755)
            reports = root / "reports"
            aggregate = reports / "aggregate-warm.json"
            completed = subprocess.run([
                sys.executable, str(ARGS.runner), "run",
                "--manifest", str(manifest_path), "--repo-root", str(repo),
                "--benchmark-runner", str(fake_runner),
                "--oracle", str(oracle), "--candidate", str(candidate),
                "--work-dir", str(root / "work"), "--reports", str(reports),
                "--aggregate-output", str(aggregate),
                "--cache-state", "warm", "--runs", "1", "--warmups", "0",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(len(list(reports.glob("r*-warm.json"))), 14)
            result = json.loads(aggregate.read_text(encoding="utf-8"))
            self.assertTrue(result["complete"])
            self.assertTrue(math.isclose(
                result["equal_workload_geometric_mean_speedup"], 2.0,
                rel_tol=1e-12))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
