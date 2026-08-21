#!/usr/bin/env python3
"""Contracts for exact single- and multi-artifact benchmark outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import tempfile
import sys
import unittest
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runner", required=True, type=Path)
    return parser.parse_known_args()


ARGS, UNITTEST_ARGS = parse_args()
SPEC = importlib.util.spec_from_file_location("benchmark_translate", ARGS.runner)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import benchmark runner: {ARGS.runner}")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def minimal_las_header(point_count: int = 0) -> bytes:
    header = bytearray(375)
    header[0:4] = b"LASF"
    header[24] = 1
    header[25] = 4
    header[105:107] = struct.pack("<H", 28)
    header[104] = 2
    header[107:111] = struct.pack("<I", point_count)
    header[247:255] = struct.pack("<Q", point_count)
    return bytes(header)


def temporary_program(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def make_validation_arguments(
        root: Path, with_pipeline: bool = False) -> argparse.Namespace:
    oracle = root / "oracle"
    candidate = root / "candidate"
    oracle.write_bytes(b"#!/usr/bin/env python3\n")
    candidate.write_bytes(b"#!/usr/bin/env python3\n")
    os.chmod(oracle, 0o755)
    os.chmod(candidate, 0o755)
    input_path = root / "input.las"
    input_path.write_bytes(minimal_las_header())
    frozen = root / "frozen-time.so"
    frozen.write_bytes(b"")
    work_dir = root / "work-dir"
    work_dir.mkdir()
    report = root / "report.json"
    pipeline = root / "pipeline.json"
    if with_pipeline:
        pipeline.write_text(json.dumps({
            "pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {"type": "writers.las", "filename": "output.las"},
            ]
        }) + "\n", encoding="utf-8")
    argv = [
        str(ARGS.runner),
        "--oracle", str(oracle),
        "--candidate", str(candidate),
        "--input", str(input_path),
        "--input-label", "validation",
        "--work-dir", str(work_dir),
        "--report", str(report),
        "--frozen-time-library", str(frozen),
    ]
    if with_pipeline:
        argv.extend(["--pipeline", str(pipeline)])
    previous_argv = sys.argv
    sys.argv = argv
    try:
        return RUNNER.parse_args()
    finally:
        sys.argv = previous_argv


class BenchmarkOutputTest(unittest.TestCase):
    def test_companion_engine_provenance_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            candidate = Path(root) / "pdg"
            candidate.write_bytes(b"dispatcher")
            self.assertEqual(RUNNER.companion_engine_provenance(candidate), {
                "engine_path": None,
                "engine_binary_sha256": None,
            })

            engine = Path(root) / "pdg-engine"
            engine.write_bytes(b"engine")
            self.assertEqual(RUNNER.companion_engine_provenance(candidate), {
                "engine_path": str(engine),
                "engine_binary_sha256": hashlib.sha256(b"engine").hexdigest(),
            })

            links = Path(root) / "links"
            links.mkdir()
            linked_candidate = links / "pdg"
            linked_candidate.symlink_to(candidate)
            self.assertEqual(
                RUNNER.companion_engine_provenance(linked_candidate), {
                    "engine_path": str(engine),
                    "engine_binary_sha256":
                        hashlib.sha256(b"engine").hexdigest(),
                })

    def test_run_once_wires_and_observes_nndistance_handoff(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            program = directory / "proof-candidate"
            program.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

required = (
    "PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE",
    "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY",
    "PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ",
    "PDG_REQUIRE_NND_DEVICE_ONLY_HANDOFF",
    "PDG_REQUIRE_NND_DEVICE_REPAIR",
    "PDG_REQUIRE_NND_PARALLEL_REPAIR",
    "PDG_REQUIRE_OUTLIER_DEVICE_REPAIR",
    "PDG_REQUIRE_OUTLIER_PARALLEL_REPAIR",
    "PDG_REQUIRE_AUTOMATIC_EIGEN_FAMILY_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_RANK_OPTIMAL_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_OUTLIER_NNDISTANCE_RESIDENT",
    "PDG_REQUIRE_AUTOMATIC_RADIUS_OUTLIER_RADIALDENSITY_RESIDENT",
    "PDG_REQUIRE_NEIGHBORHOOD_REUSE",
    "PDG_REQUIRE_KNN_GATHER_REUSE",
    "PDG_EXPERIMENTAL_DIRECT_CLASSIFICATION_OUTPUT",
    "PDG_EXPERIMENTAL_DIRECT_EXTRA_DOUBLE_OUTPUT",
    "PDG_REQUIRE_DIRECT_EXTRA_DOUBLE_OUTPUT",
    "PDG_REQUIRE_RADIUS_OUTLIER_RADIALDENSITY_COMPOSITION",
)
if any(os.environ.get(name) != "1" for name in required):
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
stats = Path(sys.argv[sys.argv.index("--stats") + 1])
stats.write_text(json.dumps({
    "placement": {"choice": "device"},
    "execution": {
        "direct_las_resident_source": True,
        "direct_las_record_summary": True,
        "direct_las_host_xyz_mirror": False,
        "direct_las_output": True,
        "direct_extra_double_output": True,
        "terminal_spill_elided": True,
        "nndistance_device_only_handoff": True,
        "nndistance_host_restore": False,
        "nndistance_assignment_device_column_reuse": True,
        "selected_device_calibration_matches_executor": True,
        "boundary_accounting_matches_prediction": True,
        "knn_gather_reuse": True,
        "rewrite_executable": True,
        "resident_preflight": {"accepted": True},
        "schedule": {"pipeline_class": "fused_point_program"},
        "index_builds": {
            "matches_prediction": True,
            "predicted": 1,
            "observed": 1,
        },
        "exact_host_repair": {
            "nndistance": {
                "seconds": 0.0,
                "incomplete_rows": 0,
                "repaired_rows": 0,
            },
            "statistical_outlier": {
                "seconds": 0.0,
                "incomplete_rows": 0,
                "repaired_rows": 0,
            },
            "approximate_coplanar": {
                "seconds": 0.002,
                "triggered": True,
                "trigger_count": 1,
                "ambiguous_rows": 2,
                "incomplete_rows": 0,
                "repaired_rows": 2,
                "kd3_used": True,
                "kd3_uses": 1,
                "device_to_host_bytes": 194,
                "host_to_device_bytes": 194,
            },
        },
        "exact_device_repair": {
            "nndistance": {
                "seconds": 0.001,
                "incomplete_rows": 1,
                "repaired_rows": 1,
                "parallel_repaired_rows": 1,
            },
            "statistical_outlier": {
                "seconds": 0.001,
                "incomplete_rows": 1,
                "repaired_rows": 1,
                "parallel_repaired_rows": 1,
            },
        },
    },
}), encoding="utf-8")
""",
                encoding="utf-8",
            )
            program.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            output = directory / "output.las"
            pipeline = {
                "pipeline": [
                    {"type": "readers.las", "filename": "input.las"},
                    {"type": "writers.las", "filename": "output.las"},
                ],
            }
            result = RUNNER.run_once(
                "candidate", program, source, output,
                {"PATH": os.environ["PATH"]}, pipeline, "resident", None,
                False, False, False, False, None,
                require_nnd_device_repair=True,
                require_direct_las_resident_source=True,
                require_direct_las_record_summary=True,
                require_no_direct_las_host_xyz=True,
                require_nnd_device_only_handoff=True,
                require_nnd_parallel_repair=True,
                require_outlier_device_repair=True,
                require_outlier_parallel_repair=True,
                require_approximate_coplanar_host_repair=True,
                require_automatic_eigen_family_resident=True,
                require_automatic_rank_optimal_resident=True,
                require_automatic_outlier_nndistance_resident=True,
                require_automatic_radius_outlier_radialdensity_resident=True,
                require_neighborhood_reuse=True,
                require_knn_gather_reuse=True,
                require_resident_calibration_provenance=True,
                require_resident_boundary_accounting=True,
                experimental_direct_classification_output=True,
                require_direct_extra_double_output=True,
                require_radius_outlier_radialdensity_composition=True,
            )
            self.assertTrue(result["resident"][
                "nndistance_device_only_handoff"])
            self.assertTrue(result["resident"][
                "nndistance_assignment_device_column_reuse"])
            self.assertTrue(result["resident"][
                "selected_device_calibration_matches_executor"])
            self.assertTrue(result["resident"][
                "boundary_accounting_matches_prediction"])
            self.assertTrue(result["resident"][
                "direct_extra_double_output"])
            self.assertTrue(result["resident"]["terminal_spill_elided"])
            self.assertTrue(result["resident"]["knn_gather_reuse"])
            self.assertEqual(result["resident"]["index_builds"]["predicted"],
                             1)
            self.assertEqual(result["resident"]["outlier_device_repair"][
                "repaired_rows"], 1)
            self.assertEqual(result["resident"][
                "approximate_coplanar_host_repair"]["trigger_count"], 1)
            self.assertTrue(result["resident"][
                "approximate_coplanar_host_repair"]["kd3_used"])
            self.assertEqual(result["resident"]["nndistance_device_repair"][
                "parallel_repaired_rows"], 1)
            self.assertEqual(result["resident"]["index_builds"]["observed"],
                             1)

    def test_direct_extra_double_proof_fails_closed(self) -> None:
        valid_execution = {
            "executor": "planner_resident_shared_index_direct_las",
            "rewrite_executable": True,
            "resident_preflight": {"accepted": True},
            "schedule": {"pipeline_class": "fused_point_program"},
            "direct_las_output": True,
            "direct_extra_double_output": True,
            "terminal_spill_elided": True,
            "index_builds": {
                "matches_prediction": True,
                "predicted": 1,
                "observed": 1,
            },
        }
        mutations = (
            ("direct_las_output", False),
            ("direct_extra_double_output", False),
            ("terminal_spill_elided", False),
        )
        with tempfile.TemporaryDirectory(
                prefix="pdg-benchmark-unit-") as root:
            stats = Path(root) / "stats.json"
            for field, value in mutations:
                execution = dict(valid_execution)
                execution[field] = value
                stats.write_text(json.dumps({
                    "placement": {"available": True, "choice": "device"},
                    "execution": execution,
                }), encoding="utf-8")
                with self.assertRaisesRegex(
                        RUNNER.BenchmarkError,
                        "required direct Extra Bytes binary64 output"):
                    RUNNER.validate_resident_stats(
                        stats, "planner_resident_shared_index_direct_las",
                        False, False, False, False, False, False,
                        require_direct_extra_double_output=True,
                    )

            for pipeline_class in (None, "unsupported"):
                execution = dict(valid_execution)
                execution["schedule"] = {
                    "pipeline_class": pipeline_class,
                }
                stats.write_text(json.dumps({
                    "placement": {"available": True, "choice": "device"},
                    "execution": execution,
                }), encoding="utf-8")
                with self.assertRaisesRegex(
                        RUNNER.BenchmarkError,
                        "direct LAS resident schedule class"):
                    RUNNER.validate_resident_stats(
                        stats, "planner_resident_shared_index_direct_las",
                        False, False, False, False, False, False,
                        require_direct_extra_double_output=True,
                    )

    def test_run_once_wires_and_observes_direct_skewness(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            program = directory / "skewness-candidate"
            program.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if os.environ.get("PDG_REQUIRE_DIRECT_SKEWNESS_COMPOSITION") != "1":
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
stats = Path(sys.argv[sys.argv.index("--stats") + 1])
stats.write_text(json.dumps({
    "placement": {"available": True, "choice": "device"},
    "execution": {
        "executor": "planner_resident_global_order_direct_las",
        "direct_las_output": True,
        "direct_las_resident_source": True,
        "direct_permuted_classification_output": True,
        "direct_las_record_summary": False,
        "direct_las_host_xyz_mirror": False,
        "terminal_spill_elided": True,
        "boundary_accounting_matches_prediction": True,
        "rewrite_executable": True,
        "resident_preflight": {"accepted": True},
        "selected_regions": [0],
        "selected_stage_ids": [1],
        "schedule": {
            "pipeline_class": "whole_view_global_order",
            "item_count": 3,
            "observed_output_item_count": 3,
        },
        "index_builds": {
            "matches_prediction": True,
            "predicted": 0,
            "observed": 0,
        },
        "events": [
            {"kind": "host_to_device", "bytes": 24},
            {"kind": "device_to_host", "bytes": 24},
        ],
    },
}), encoding="utf-8")
""",
                encoding="utf-8",
            )
            program.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            output = directory / "output.las"
            pipeline = {"pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {"type": "filters.skewnessbalancing"},
                {"type": "writers.las", "filename": "output.las",
                 "extra_dims": "all"},
            ]}
            result = RUNNER.run_once(
                "candidate", program, source, output,
                {"PATH": os.environ["PATH"]}, pipeline, "resident", None,
                False, False, False, False, None,
                require_resident_executor=(
                    "planner_resident_global_order_direct_las"),
                require_direct_skewness_composition=True,
            )
            self.assertTrue(result["resident"][
                "direct_permuted_classification_output"])
            self.assertEqual(result["resident"]["index_builds"]["observed"],
                             0)

    def test_direct_skewness_proof_fails_closed(self) -> None:
        valid = {
            "executor": "planner_resident_global_order_direct_las",
            "direct_las_output": True,
            "direct_las_resident_source": True,
            "direct_permuted_classification_output": True,
            "direct_las_record_summary": False,
            "direct_las_host_xyz_mirror": False,
            "terminal_spill_elided": True,
            "boundary_accounting_matches_prediction": True,
            "rewrite_executable": True,
            "resident_preflight": {"accepted": True},
            "selected_regions": [0],
            "selected_stage_ids": [1],
            "schedule": {
                "pipeline_class": "whole_view_global_order",
                "item_count": 3,
                "observed_output_item_count": 3,
            },
            "index_builds": {
                "matches_prediction": True,
                "predicted": 0,
                "observed": 0,
            },
            "events": [
                {"kind": "host_to_device", "bytes": 24},
                {"kind": "device_to_host", "bytes": 24},
            ],
        }
        mutations = (
            ("direct_las_output", False),
            ("direct_las_resident_source", False),
            ("direct_permuted_classification_output", False),
            ("terminal_spill_elided", False),
            ("selected_regions", []),
            ("schedule", {"pipeline_class": "whole_view_global_order",
                          "item_count": 0,
                          "observed_output_item_count": 0}),
            ("index_builds", {"matches_prediction": True,
                              "predicted": 0, "observed": 1}),
            ("events", [{"kind": "host_to_device", "bytes": 24}]),
        )
        with tempfile.TemporaryDirectory(
                prefix="pdg-benchmark-unit-") as root:
            stats = Path(root) / "stats.json"
            for field, value in mutations:
                execution = dict(valid)
                execution[field] = value
                stats.write_text(json.dumps({
                    "placement": {"available": True, "choice": "device"},
                    "execution": execution,
                }), encoding="utf-8")
                with self.assertRaisesRegex(
                        RUNNER.BenchmarkError,
                        "required direct skewness composition"):
                    RUNNER.validate_resident_stats(
                        stats, "planner_resident_global_order_direct_las",
                        False, False, False, False, False, False,
                        require_direct_skewness_composition=True,
                    )

    def test_run_once_wires_and_observes_direct_sort(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            program = directory / "sort-candidate"
            program.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if os.environ.get("PDG_REQUIRE_DIRECT_SORT_COMPOSITION") != "1":
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
stats = Path(sys.argv[sys.argv.index("--stats") + 1])
stats.write_text(json.dumps({
    "placement": {"available": True, "choice": "device"},
    "execution": {
        "executor": "planner_resident_global_order_direct_las",
        "direct_las_output": True,
        "direct_las_resident_source": True,
        "direct_permuted_output": True,
        "direct_permuted_sort_output": True,
        "direct_permuted_classification_output": False,
        "direct_las_record_summary": False,
        "direct_las_host_xyz_mirror": False,
        "terminal_spill_elided": True,
        "boundary_accounting_matches_prediction": True,
        "rewrite_executable": True,
        "resident_preflight": {"accepted": True},
        "selected_regions": [0],
        "selected_stage_ids": [1],
        "schedule": {"pipeline_class": "whole_view_global_order",
                     "item_count": 3,
                     "observed_output_item_count": 3},
        "index_builds": {"matches_prediction": True,
                         "predicted": 0, "observed": 0},
        "events": [{"kind": "host_to_device", "bytes": 24},
                   {"kind": "device_to_host", "bytes": 24}],
    },
}), encoding="utf-8")
""",
                encoding="utf-8",
            )
            program.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            output = directory / "output.las"
            pipeline = {"pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {"type": "filters.sort", "dimension": "Z",
                 "order": "ASC", "algorithm": "NORMAL"},
                {"type": "writers.las", "filename": "output.las",
                 "extra_dims": "all"},
            ]}
            result = RUNNER.run_once(
                "candidate", program, source, output,
                {"PATH": os.environ["PATH"]}, pipeline, "resident", None,
                False, False, False, False, None,
                require_resident_executor=(
                    "planner_resident_global_order_direct_las"),
                require_direct_sort_composition=True,
            )
            self.assertTrue(
                result["resident"]["direct_permuted_sort_output"])
            self.assertFalse(result["resident"][
                "direct_permuted_classification_output"])

    def test_direct_sort_proof_fails_closed(self) -> None:
        valid = {
            "executor": "planner_resident_global_order_direct_las",
            "direct_las_output": True,
            "direct_las_resident_source": True,
            "direct_permuted_output": True,
            "direct_permuted_sort_output": True,
            "direct_permuted_classification_output": False,
            "direct_las_record_summary": False,
            "direct_las_host_xyz_mirror": False,
            "terminal_spill_elided": True,
            "boundary_accounting_matches_prediction": True,
            "rewrite_executable": True,
            "resident_preflight": {"accepted": True},
            "selected_regions": [0],
            "selected_stage_ids": [1],
            "schedule": {"pipeline_class": "whole_view_global_order",
                         "item_count": 3,
                         "observed_output_item_count": 3},
            "index_builds": {"matches_prediction": True,
                             "predicted": 0, "observed": 0},
            "events": [{"kind": "host_to_device", "bytes": 24},
                       {"kind": "device_to_host", "bytes": 24}],
        }
        mutations = (
            ("direct_permuted_output", False),
            ("direct_permuted_sort_output", False),
            ("direct_permuted_classification_output", True),
            ("direct_las_resident_source", False),
            ("index_builds", {"matches_prediction": True,
                              "predicted": 0, "observed": 1}),
            ("events", [{"kind": "host_to_device", "bytes": 24}]),
        )
        with tempfile.TemporaryDirectory(
                prefix="pdg-benchmark-unit-") as root:
            stats = Path(root) / "stats.json"
            for field, value in mutations:
                execution = dict(valid)
                execution[field] = value
                stats.write_text(json.dumps({
                    "placement": {"available": True, "choice": "device"},
                    "execution": execution,
                }), encoding="utf-8")
                with self.assertRaisesRegex(
                        RUNNER.BenchmarkError,
                        "required direct sort composition"):
                    RUNNER.validate_resident_stats(
                        stats, "planner_resident_global_order_direct_las",
                        False, False, False, False, False, False,
                        require_direct_sort_composition=True,
                    )

    def test_direct_sort_environment_is_scrubbed(self) -> None:
        names = (
            "PDG_EXPERIMENTAL_DIRECT_SORT_COMPOSITION",
            "PDG_REQUIRE_DIRECT_SORT_COMPOSITION",
            "PDG_REQUIRE_AUTOMATIC_SORT_RESIDENT",
            "PDG_TEST_DIRECT_SORT_PROOF_FAILURE",
            "PDG_TEST_DIRECT_SORT_PREFLIGHT_FAILURE",
        )
        previous = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in names:
            self.assertNotIn(name, clean)

    def test_lof_parallel_repair_environment_is_scrubbed(self) -> None:
        names = (
            "PDG_REQUIRE_LOF_PARALLEL_REPAIR",
            "PDG_NATIVE_WORKERS",
        )
        previous = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in names:
            self.assertNotIn(name, clean)

    def test_host_neighborhood_worker_environment_is_scrubbed(self) -> None:
        names = (
            "PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS",
            "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS",
            "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS",
            "PDG_DISABLE_KD3_COORDINATE_CACHE",
            "PDAL_TEST_VERIFY_KD3_SNAPSHOT",
        )
        previous = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in names:
            self.assertNotIn(name, clean)

    def test_lof_kd3_coordinate_cache_environment_is_scrubbed(self) -> None:
        names = (
            "PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE",
            "PDG_DISABLE_LOF_KD3_COORDINATE_CACHE",
            "PDG_TEST_KD3_CACHE_BUILD_FAILURE",
        )
        previous = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in names:
            self.assertNotIn(name, clean)

    def test_lof_knn_tuning_environment_is_scrubbed(self) -> None:
        names = (
            "PDG_KNN_DEVICE_SHELL_BUDGET",
            "PDG_DISABLE_KNN_DISTANCE_PREFILTER",
            "PDG_FORCE_MORTON_BVH",
            "PDG_FORCE_UNIFORM_GRID",
            "PDG_DISABLE_NEIGHBORHOOD_ROW_BOUNDARY",
            "PDG_REQUIRE_NEIGHBORHOOD_ROW_BOUNDARY",
        )
        previous = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in names:
            self.assertNotIn(name, clean)

    def test_automatic_skewness_environment_is_scrubbed(self) -> None:
        name = "PDG_REQUIRE_AUTOMATIC_SKEWNESS_RESIDENT"
        previous = os.environ.get(name)
        os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        self.assertNotIn(name, clean)

    def test_automatic_hag_nn_environment_is_scrubbed(self) -> None:
        proof_names = (
            "PDG_REQUIRE_AUTOMATIC_HAG_NN_RESIDENT",
            "PDG_TEST_AUTOMATIC_HAG_NN_PROOF_FAILURE",
            "PDG_REQUIRE_AUTOMATIC_R2_GROUND_NORMALIZE",
            "PDG_REQUIRE_HAG_NN_SELECTIVE_REPAIR",
            "PDG_INTERNAL_AUTOMATIC_R2_HYBRID",
            "PDG_TEST_AUTOMATIC_R2_HAG_NN_DEVICE_DECLINE",
        )
        previous = {name: os.environ.get(name) for name in proof_names}
        for name in proof_names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in proof_names:
            self.assertNotIn(name, clean)

    def test_automatic_hag_delaunay_environment_is_scrubbed(self) -> None:
        proof_names = (
            "PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT",
            "PDG_TEST_AUTOMATIC_HAG_DELAUNAY_PROOF_FAILURE",
        )
        previous = {name: os.environ.get(name) for name in proof_names}
        for name in proof_names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in proof_names:
            self.assertNotIn(name, clean)

    def test_validate_paths_rejects_lof_parallel_repair_without_pipeline(
            self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            args = make_validation_arguments(Path(root), with_pipeline=False)
            args.require_lof_parallel_repair = True
            with self.assertRaisesRegex(
                RUNNER.BenchmarkError,
                "--require-lof-parallel-repair requires a pipeline and "
                "default or resident candidate mode",
            ):
                RUNNER.validate_paths(args)

    def test_validate_paths_rejects_lof_parallel_repair_for_incompatible_mode(
            self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            args = make_validation_arguments(Path(root), with_pipeline=True)
            args.require_lof_parallel_repair = True
            args.candidate_mode = "hybrid-host"
            with self.assertRaisesRegex(
                RUNNER.BenchmarkError,
                "--require-lof-parallel-repair requires a pipeline and "
                "default or resident candidate mode",
            ):
                RUNNER.validate_paths(args)

    def test_validate_paths_rejects_lof_parallel_repair_with_low_worker_count(
            self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            args = make_validation_arguments(Path(root), with_pipeline=True)
            args.candidate_mode = "default"
            args.require_lof_parallel_repair = True
            args.candidate_native_workers = 1
            with self.assertRaisesRegex(
                RUNNER.BenchmarkError,
                "--require-lof-parallel-repair requires "
                "--candidate-native-workers >= 2",
            ):
                RUNNER.validate_paths(args)

    def test_validate_paths_rejects_lof_kd3_coordinate_cache_without_pipeline(
            self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            args = make_validation_arguments(Path(root), with_pipeline=False)
            args.require_lof_kd3_coordinate_cache = True
            with self.assertRaisesRegex(
                RUNNER.BenchmarkError,
                "--require-lof-kd3-coordinate-cache requires a pipeline and "
                "default or resident candidate mode",
            ):
                RUNNER.validate_paths(args)

    def test_validate_paths_rejects_lof_kd3_coordinate_cache_for_incompatible_mode(
            self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            args = make_validation_arguments(Path(root), with_pipeline=True)
            args.require_lof_kd3_coordinate_cache = True
            args.candidate_mode = "hybrid-host"
            with self.assertRaisesRegex(
                RUNNER.BenchmarkError,
                "--require-lof-kd3-coordinate-cache requires a pipeline and "
                "default or resident candidate mode",
            ):
                RUNNER.validate_paths(args)

    def test_validate_paths_allows_lof_kd3_coordinate_cache_with_default_candidate_mode(
            self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            args = make_validation_arguments(Path(root), with_pipeline=True)
            args.candidate_mode = "default"
            args.require_lof_kd3_coordinate_cache = True
            RUNNER.validate_paths(args)

    def test_validate_paths_allows_lof_kd3_coordinate_cache_with_resident_mode(
            self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            args = make_validation_arguments(Path(root), with_pipeline=True)
            args.candidate_mode = "resident"
            args.require_lof_kd3_coordinate_cache = True
            RUNNER.validate_paths(args)

    def test_validate_paths_allows_lof_parallel_repair_with_default_candidate_mode(
            self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            args = make_validation_arguments(Path(root), with_pipeline=True)
            args.candidate_mode = "default"
            args.require_lof_parallel_repair = True
            RUNNER.validate_paths(args)

    def test_validate_paths_allows_lof_parallel_repair_with_resident_mode_and_workers(
            self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            args = make_validation_arguments(Path(root), with_pipeline=True)
            args.candidate_mode = "resident"
            args.require_lof_parallel_repair = True
            args.candidate_native_workers = 2
            RUNNER.validate_paths(args)

    def test_run_once_wires_automatic_skewness_requirement(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            program = directory / "skewness-default-candidate"
            program.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if os.environ.get("PDG_REQUIRE_AUTOMATIC_SKEWNESS_RESIDENT") != "1":
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
""",
                encoding="utf-8",
            )
            program.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            output = directory / "output.las"
            pipeline = {"pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {"type": "filters.skewnessbalancing"},
                {"type": "writers.las", "filename": "output.las",
                 "extra_dims": "all"},
            ]}
            result = RUNNER.run_once(
                "candidate", program, source, output,
                {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                False, False, False, False, None,
                require_automatic_skewness_resident=True,
            )
            self.assertEqual(result["output_sha256"], RUNNER.sha256_bytes(
                b"exact"))

    def test_run_once_wires_automatic_hag_nn_requirement(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            program = directory / "hag-nn-default-candidate"
            program.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if os.environ.get("PDG_REQUIRE_AUTOMATIC_HAG_NN_RESIDENT") != "1":
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
""",
                encoding="utf-8",
            )
            program.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            output = directory / "output.las"
            pipeline = {"pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {"type": "filters.hag_nn"},
                {"type": "writers.las", "filename": "output.las"},
            ]}
            result = RUNNER.run_once(
                "candidate", program, source, output,
                {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                False, False, False, False, None,
                require_automatic_hag_nn_resident=True,
            )
            self.assertEqual(result["output_sha256"], RUNNER.sha256_bytes(
                b"exact"))


    def test_run_once_wires_automatic_hag_delaunay_requirement(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            program = directory / "hag-delaunay-default-candidate"
            program.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if os.environ.get("PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT") != "1":
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
""",
                encoding="utf-8",
            )
            program.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            output = directory / "output.las"
            pipeline = {
                "pipeline": [
                    {"type": "readers.las", "filename": "input.las"},
                    {"type": "filters.hag_nn"},
                    {"type": "writers.las", "filename": "output.las"},
                ]
            }
            result = RUNNER.run_once(
                "candidate", program, source, output,
                {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                False, False, False, False, None,
                require_automatic_hag_delaunay_resident=True,
            )
            self.assertEqual(result["output_sha256"], RUNNER.sha256_bytes(
                b"exact"))

    def test_automatic_hag_delaunay_proof_is_default_mode_candidate_only(
            self) -> None:
        proof_names = (
            "PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT",
            "PDG_TEST_AUTOMATIC_HAG_DELAUNAY_PROOF_FAILURE",
        )
        previous = {name: os.environ.get(name) for name in proof_names}
        for name in proof_names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in proof_names:
            self.assertNotIn(name, clean)

        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            script = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

candidate = Path(sys.argv[0]).name == "candidate"
if sys.argv[1] != "pipeline":
    raise SystemExit(8)
proof = os.environ.get(
    "PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT")
if proof != ("1" if candidate else None):
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
"""
            candidate = directory / "candidate"
            oracle = directory / "oracle"
            candidate.write_text(script, encoding="utf-8")
            oracle.write_text(script, encoding="utf-8")
            candidate.chmod(0o755)
            oracle.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            pipeline = {
                "pipeline": [
                    {"type": "readers.las", "filename": "input.las"},
                    {"type": "writers.las", "filename": "output.las"},
                ]
            }
            for label, program in (("candidate", candidate),
                                   ("oracle", oracle)):
                result = RUNNER.run_once(
                    label, program, source, directory / f"{label}.las",
                    {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                    False, False, False, False, None,
                    require_automatic_hag_delaunay_resident=True,
                )
                self.assertEqual(result["output_sha256"], hashlib.sha256(
                    b"exact").hexdigest())


    def test_run_once_wires_automatic_sort_requirement(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            program = directory / "sort-default-candidate"
            program.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if os.environ.get("PDG_REQUIRE_AUTOMATIC_SORT_RESIDENT") != "1":
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
""",
                encoding="utf-8",
            )
            program.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            output = directory / "output.las"
            pipeline = {"pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {"type": "filters.sort", "dimension": "Z",
                 "order": "ASC", "algorithm": "NORMAL"},
                {"type": "writers.las", "filename": "output.las",
                 "extra_dims": "all"},
            ]}
            result = RUNNER.run_once(
                "candidate", program, source, output,
                {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                False, False, False, False, None,
                require_automatic_sort_resident=True,
            )
            self.assertEqual(result["output_sha256"], RUNNER.sha256_bytes(
                b"exact"))

    def test_run_once_wires_lof_parallel_repair_and_native_workers_candidate_only(
            self) -> None:
        ambient = {
            "PDG_REQUIRE_LOF_PARALLEL_REPAIR": "ambient",
            "PDG_NATIVE_WORKERS": "ambient",
        }
        previous = {name: os.environ.get(name) for name in ambient}
        for name, value in ambient.items():
            os.environ[name] = value
        try:
            with tempfile.TemporaryDirectory(
                    prefix="pdg-benchmark-unit-") as root:
                directory = Path(root)
                program = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

candidate = Path(sys.argv[0]).name == "candidate"
if candidate:
    if os.environ.get("PDG_REQUIRE_LOF_PARALLEL_REPAIR") != "1":
        raise SystemExit(9)
    if os.environ.get("PDG_NATIVE_WORKERS") != "7":
        raise SystemExit(10)
else:
    if os.environ.get("PDG_REQUIRE_LOF_PARALLEL_REPAIR") is not None:
        raise SystemExit(11)
    if os.environ.get("PDG_NATIVE_WORKERS") is not None:
        raise SystemExit(12)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
"""
                candidate = directory / "candidate"
                oracle = directory / "oracle"
                candidate.write_text(program, encoding="utf-8")
                oracle.write_text(program, encoding="utf-8")
                candidate.chmod(0o755)
                oracle.chmod(0o755)
                source = directory / "input.las"
                source.write_bytes(b"source")
                pipeline = {
                    "pipeline": [
                        {"type": "readers.las", "filename": "input.las"},
                        {"type": "writers.las", "filename": "output.las"},
                    ]
                }
                for label, binary in (("candidate", candidate),
                                     ("oracle", oracle)):
                    result = RUNNER.run_once(
                        label, binary, source, directory / f"{label}.las",
                        {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                        False, False, False, False, None,
                        candidate_native_workers=7,
                        require_lof_parallel_repair=True,
                    )
                    self.assertEqual(result["output_sha256"], RUNNER.sha256_bytes(
                        b"exact"))
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_run_once_wires_lof_kd3_coordinate_cache_candidate_only(
            self) -> None:
        ambient = {
            "PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE": "ambient",
            "PDG_DISABLE_LOF_KD3_COORDINATE_CACHE": "ambient",
        }
        previous = {name: os.environ.get(name) for name in ambient}
        for name, value in ambient.items():
            os.environ[name] = value
        try:
            with tempfile.TemporaryDirectory(
                    prefix="pdg-benchmark-unit-") as root:
                directory = Path(root)
                program = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

candidate = Path(sys.argv[0]).name == "candidate"
if candidate:
    if os.environ.get("PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE") != "1":
        raise SystemExit(9)
else:
    if os.environ.get("PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE") is not None:
        raise SystemExit(10)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding=\"utf-8\"))
Path(pipeline[\"pipeline\"][-1][\"filename\"]).write_bytes(b\"exact\")
"""
                candidate = directory / "candidate"
                oracle = directory / "oracle"
                candidate.write_text(program, encoding="utf-8")
                oracle.write_text(program, encoding="utf-8")
                candidate.chmod(0o755)
                oracle.chmod(0o755)
                source = directory / "input.las"
                source.write_bytes(b"source")
                pipeline = {
                    "pipeline": [
                        {"type": "readers.las", "filename": "input.las"},
                        {"type": "writers.las", "filename": "output.las"},
                    ]
                }
                for label, binary in (("candidate", candidate),
                                     ("oracle", oracle)):
                    result = RUNNER.run_once(
                        label, binary, source, directory / f"{label}.las",
                        {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                        False, False, False, False, None,
                        require_lof_kd3_coordinate_cache=True,
                    )
                    self.assertEqual(result["output_sha256"], RUNNER.sha256_bytes(
                        b"exact"))
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_label_nndistance_environment_is_scrubbed(self) -> None:
        names = (
            "PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID",
            "PDG_TEST_LABEL_NNDISTANCE_RECOVERABLE_CUDA_FAILURE",
        )
        previous = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in names:
            self.assertNotIn(name, clean)

    def test_run_once_wires_automatic_label_nndistance_requirement(
            self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            program = directory / "label-nndistance-default-candidate"
            program.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if os.environ.get("PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID") != "1":
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
""",
                encoding="utf-8",
            )
            program.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            output = directory / "output.las"
            pipeline = {"pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {"type": "filters.label_duplicates",
                 "dimensions": "Classification"},
                {"type": "filters.nndistance", "k": 10},
                {"type": "filters.assign", "value": "UserData = Duplicate"},
                {"type": "writers.las", "filename": "output.las"},
            ]}
            result = RUNNER.run_once(
                "candidate", program, source, output,
                {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                False, False, False, False, None,
                require_automatic_label_nndistance_hybrid=True,
            )
            self.assertEqual(result["output_sha256"], RUNNER.sha256_bytes(
                b"exact"))

    def test_approximate_coplanar_repair_proof_fails_closed(self) -> None:
        valid = {
            "seconds": 0.002,
            "triggered": True,
            "trigger_count": 1,
            "ambiguous_rows": 2,
            "incomplete_rows": 0,
            "repaired_rows": 2,
            "kd3_used": True,
            "kd3_uses": 1,
            "device_to_host_bytes": 194,
            "host_to_device_bytes": 194,
        }
        invalid = [
            None,
            {**valid, "seconds": float("nan")},
            {**valid, "trigger_count": 2},
            {**valid, "ambiguous_rows": 0, "repaired_rows": 1},
            {
                **valid,
                "ambiguous_rows": 0,
                "incomplete_rows": 0,
                "repaired_rows": 0,
            },
            {**valid, "host_to_device_bytes": 193},
            {
                **valid,
                "seconds": 0.0,
                "triggered": False,
                "trigger_count": 0,
                "kd3_used": False,
                "kd3_uses": 0,
            },
        ]
        with tempfile.TemporaryDirectory(
                prefix="pdg-benchmark-unit-") as root:
            stats = Path(root) / "stats.json"
            for repair in invalid:
                execution = {}
                if repair is not None:
                    execution["exact_host_repair"] = {
                        "approximate_coplanar": repair,
                    }
                stats.write_text(json.dumps({
                    "placement": {},
                    "execution": execution,
                }), encoding="utf-8")
                with self.assertRaises(RUNNER.BenchmarkError):
                    RUNNER.validate_resident_stats(
                        stats, None, False, False, False, False, False,
                        False,
                        require_approximate_coplanar_host_repair=True,
                    )

    def test_automatic_outlier_proof_is_default_mode_candidate_only(self) -> None:
        proof_names = (
            "PDG_REQUIRE_AUTOMATIC_OUTLIER_NNDISTANCE_RESIDENT",
            "PDG_TEST_AUTOMATIC_OUTLIER_PROOF_FAILURE",
        )
        previous = {name: os.environ.get(name) for name in proof_names}
        for name in proof_names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in proof_names:
            self.assertNotIn(name, clean)

        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            script = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

candidate = Path(sys.argv[0]).name == "candidate"
if sys.argv[1] != "pipeline":
    raise SystemExit(8)
proof = os.environ.get(
    "PDG_REQUIRE_AUTOMATIC_OUTLIER_NNDISTANCE_RESIDENT")
if proof != ("1" if candidate else None):
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
"""
            candidate = directory / "candidate"
            oracle = directory / "oracle"
            candidate.write_text(script, encoding="utf-8")
            oracle.write_text(script, encoding="utf-8")
            candidate.chmod(0o755)
            oracle.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            pipeline = {"pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {"type": "writers.las", "filename": "output.las"},
            ]}
            for label, program in (("candidate", candidate),
                                   ("oracle", oracle)):
                result = RUNNER.run_once(
                    label, program, source, directory / f"{label}.las",
                    {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                    False, False, False, False, None,
                    require_automatic_outlier_nndistance_resident=True,
                )
                self.assertEqual(result["output_sha256"],
                                 hashlib.sha256(b"exact").hexdigest())

    def test_automatic_radiusassign_proof_is_default_mode_candidate_only(
            self) -> None:
        proof_names = (
            "PDG_REQUIRE_AUTOMATIC_RADIUSASSIGN_RESIDENT",
            "PDG_TEST_AUTOMATIC_RADIUSASSIGN_PROOF_FAILURE",
        )
        previous = {name: os.environ.get(name) for name in proof_names}
        for name in proof_names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in proof_names:
            self.assertNotIn(name, clean)

        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            script = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

candidate = Path(sys.argv[0]).name == "candidate"
if sys.argv[1] != "pipeline":
    raise SystemExit(8)
proof = os.environ.get("PDG_REQUIRE_AUTOMATIC_RADIUSASSIGN_RESIDENT")
if proof != ("1" if candidate else None):
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
"""
            candidate = directory / "candidate"
            oracle = directory / "oracle"
            candidate.write_text(script, encoding="utf-8")
            oracle.write_text(script, encoding="utf-8")
            candidate.chmod(0o755)
            oracle.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            pipeline = {"pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {"type": "writers.las", "filename": "output.las"},
            ]}
            for label, program in (("candidate", candidate),
                                   ("oracle", oracle)):
                result = RUNNER.run_once(
                    label, program, source, directory / f"{label}.las",
                    {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                    False, False, False, False, None,
                    require_automatic_radiusassign_resident=True,
                )
                self.assertEqual(result["output_sha256"],
                                 hashlib.sha256(b"exact").hexdigest())

    def test_automatic_radius_composition_proof_is_default_candidate_only(
            self) -> None:
        proof_names = (
            "PDG_REQUIRE_AUTOMATIC_RADIUS_OUTLIER_RADIALDENSITY_RESIDENT",
            "PDG_TEST_AUTOMATIC_RADIUS_COMPOSITION_PROOF_FAILURE",
        )
        previous = {name: os.environ.get(name) for name in proof_names}
        for name in proof_names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in proof_names:
            self.assertNotIn(name, clean)

        with tempfile.TemporaryDirectory(
                prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            script = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

candidate = Path(sys.argv[0]).name == "candidate"
if sys.argv[1] != "pipeline":
    raise SystemExit(8)
proof = os.environ.get(
    "PDG_REQUIRE_AUTOMATIC_RADIUS_OUTLIER_RADIALDENSITY_RESIDENT")
if proof != ("1" if candidate else None):
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
"""
            candidate = directory / "candidate"
            oracle = directory / "oracle"
            candidate.write_text(script, encoding="utf-8")
            oracle.write_text(script, encoding="utf-8")
            candidate.chmod(0o755)
            oracle.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            pipeline = {"pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {"type": "writers.las", "filename": "output.las"},
            ]}
            for label, program in (("candidate", candidate),
                                   ("oracle", oracle)):
                result = RUNNER.run_once(
                    label, program, source, directory / f"{label}.las",
                    {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                    False, False, False, False, None,
                    require_automatic_radius_outlier_radialdensity_resident=
                        True,
                )
                self.assertEqual(result["output_sha256"],
                                 hashlib.sha256(b"exact").hexdigest())

    def test_automatic_coplanar_proof_is_default_mode_candidate_only(
            self) -> None:
        proof_names = (
            "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_RESIDENT",
            "PDG_TEST_AUTOMATIC_COPLANAR_PROOF_FAILURE",
        )
        previous = {name: os.environ.get(name) for name in proof_names}
        for name in proof_names:
            os.environ[name] = "ambient"
        try:
            clean = RUNNER.base_environment(
                Path("/oracle"), Path("/frozen-time.so"), 1_700_000_000)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        for name in proof_names:
            self.assertNotIn(name, clean)

        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            script = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

candidate = Path(sys.argv[0]).name == "candidate"
if sys.argv[1] != "pipeline":
    raise SystemExit(8)
proof = os.environ.get(
    "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_RESIDENT")
if proof != ("1" if candidate else None):
    raise SystemExit(9)
pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
"""
            candidate = directory / "candidate"
            oracle = directory / "oracle"
            candidate.write_text(script, encoding="utf-8")
            oracle.write_text(script, encoding="utf-8")
            candidate.chmod(0o755)
            oracle.chmod(0o755)
            source = directory / "input.las"
            source.write_bytes(b"source")
            pipeline = {"pipeline": [
                {"type": "readers.las", "filename": "input.las"},
                {"type": "writers.las", "filename": "output.las"},
            ]}
            for label, program in (("candidate", candidate),
                                   ("oracle", oracle)):
                result = RUNNER.run_once(
                    label, program, source, directory / f"{label}.las",
                    {"PATH": os.environ["PATH"]}, pipeline, "default", None,
                    False, False, False, False, None,
                    require_automatic_approximatecoplanar_resident=True,
                )
                self.assertEqual(result["output_sha256"],
                                 hashlib.sha256(b"exact").hexdigest())

    def test_single_output_preserves_legacy_file_hash(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            base = Path(root) / "0001-oracle.las"
            target, numbered = RUNNER.output_target(base, "output.las")
            self.assertEqual(target, base)
            self.assertFalse(numbered)
            target.write_bytes(b"abc")

            artifacts = RUNNER.collect_output_artifacts(target, numbered)
            self.assertEqual(artifacts, [{
                "name": "<output>.las",
                "bytes": 3,
                "sha256": hashlib.sha256(b"abc").hexdigest(),
            }])

    def test_numbered_outputs_form_a_stable_exact_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            base = Path(root) / "0002-candidate.las"
            target, numbered = RUNNER.output_target(base, "output#.las")
            self.assertEqual(target.name, "0002-candidate-#.las")
            self.assertTrue(numbered)
            Path(str(target).replace("#", "2")).write_bytes(b"second")
            Path(str(target).replace("#", "10")).write_bytes(b"tenth")
            (Path(root) / "unrelated.las").write_bytes(b"ignored")

            artifacts = RUNNER.collect_output_artifacts(target, numbered)
            self.assertEqual([item["name"] for item in artifacts],
                             ["10.las", "2.las"])
            self.assertEqual(sum(item["bytes"] for item in artifacts), 11)

            first = {
                "returncode": 0,
                "stdout_sha256": "stdout",
                "stderr_sha256": "stderr",
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "outputs": artifacts,
            }
            second = dict(first)
            second["outputs"] = [dict(item) for item in artifacts]
            self.assertEqual(RUNNER.exact_signature(first),
                             RUNNER.exact_signature(second))
            second["outputs"][1]["sha256"] = "different"
            self.assertNotEqual(RUNNER.exact_signature(first),
                                RUNNER.exact_signature(second))

    def test_missing_numbered_output_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            base = Path(root) / "0003-oracle.las"
            target, numbered = RUNNER.output_target(base, "output#.las")
            with self.assertRaisesRegex(RUNNER.BenchmarkError,
                                        "produced no output artifacts"):
                RUNNER.collect_output_artifacts(target, numbered)

    def test_direct_las_summary_proofs_are_observed_not_inferred(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            stats = Path(root) / "stats.json"
            payload = {
                "placement": {"choice": "device"},
                "execution": {
                    "direct_las_resident_source": True,
                    "direct_las_record_summary": True,
                    "direct_las_host_xyz_mirror": False,
                    "nndistance_device_only_handoff": True,
                    "nndistance_host_restore": False,
                    "nndistance_assignment_device_column_reuse": True,
                    "selected_device_calibration_matches_executor": True,
                    "boundary_accounting_matches_prediction": True,
                    "knn_gather_reuse": True,
                    "index_builds": {
                        "matches_prediction": True,
                        "predicted": 1,
                        "observed": 1,
                    },
                },
            }
            stats.write_text(json.dumps(payload), encoding="utf-8")
            observed = RUNNER.validate_resident_stats(
                stats, None, True, True, True, True, True, True)
            self.assertTrue(observed["direct_las_record_summary"])
            self.assertFalse(observed["direct_las_host_xyz_mirror"])
            self.assertTrue(observed["nndistance_device_only_handoff"])
            self.assertTrue(observed[
                "nndistance_assignment_device_column_reuse"])
            self.assertTrue(observed[
                "selected_device_calibration_matches_executor"])
            self.assertTrue(observed[
                "boundary_accounting_matches_prediction"])
            self.assertTrue(observed["knn_gather_reuse"])
            self.assertEqual(observed["index_builds"]["predicted"], 1)
            self.assertEqual(observed["index_builds"]["observed"], 1)

            payload["execution"]["direct_las_host_xyz_mirror"] = True
            stats.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.BenchmarkError,
                                        "host XYZ mirror was observed"):
                RUNNER.validate_resident_stats(
                    stats, None, True, True, True, True, True, True)

            payload["execution"]["direct_las_host_xyz_mirror"] = False
            payload["execution"][
                "boundary_accounting_matches_prediction"] = True
            payload["execution"]["knn_gather_reuse"] = False
            stats.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                    RUNNER.BenchmarkError,
                    "max-k kNN gather reuse was not observed"):
                RUNNER.validate_resident_stats(
                    stats, None, True, True, True, True, True, True, True)

            payload["execution"]["knn_gather_reuse"] = True
            payload["execution"]["index_builds"]["observed"] = 2
            stats.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                    RUNNER.BenchmarkError,
                    "did not use exactly one shared index"):
                RUNNER.validate_resident_stats(
                    stats, None, True, True, True, True, True, True, True)

            payload["execution"]["index_builds"]["predicted"] = True
            payload["execution"]["index_builds"]["observed"] = True
            stats.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                    RUNNER.BenchmarkError,
                    "did not use exactly one shared index"):
                RUNNER.validate_resident_stats(
                    stats, None, True, True, True, True, True, True, True)

            payload["execution"]["index_builds"]["predicted"] = 1
            payload["execution"]["index_builds"]["observed"] = 1
            payload["execution"]["index_builds"][
                "matches_prediction"] = False
            stats.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                    RUNNER.BenchmarkError,
                    "did not use exactly one shared index"):
                RUNNER.validate_resident_stats(
                    stats, None, True, True, True, True, True, True, True)

            payload["execution"][
                "nndistance_assignment_device_column_reuse"] = False
            stats.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.BenchmarkError,
                                        "device-column reuse was not observed"):
                RUNNER.validate_resident_stats(
                    stats, None, True, True, True, True, True, True)

            payload["execution"][
                "nndistance_assignment_device_column_reuse"] = True
            payload["execution"][
                "selected_device_calibration_matches_executor"] = False
            stats.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.BenchmarkError,
                                        "calibration did not match"):
                RUNNER.validate_resident_stats(
                    stats, None, True, True, True, True, True, True)

            payload["execution"][
                "selected_device_calibration_matches_executor"] = True
            payload["execution"][
                "boundary_accounting_matches_prediction"] = False
            stats.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.BenchmarkError,
                                        "boundary accounting did not match"):
                RUNNER.validate_resident_stats(
                    stats, None, True, True, True, True, True, True)

    def test_main_records_lof_parallel_repair_requirements(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            pipeline = directory / "pipeline.json"
            pipeline.write_text(json.dumps({
                "pipeline": [
                    {"type": "readers.las", "filename": "input.las"},
                    {"type": "writers.las", "filename": "output.las"},
                ]
            }) + "\n", encoding="utf-8")
            oracle = directory / "oracle"
            candidate = directory / "candidate"
            script = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "version":
    print("benchmark-runner-test")
    raise SystemExit(0)

if sys.argv[1] == "translate":
    Path(sys.argv[3]).write_bytes(b"exact")
    raise SystemExit(0)

if sys.argv[1] == "pipeline":
    pipeline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    Path(pipeline["pipeline"][-1]["filename"]).write_bytes(b"exact")
    raise SystemExit(0)

raise SystemExit(1)
"""
            oracle.write_text(script, encoding="utf-8")
            candidate.write_text(script, encoding="utf-8")
            oracle.chmod(0o755)
            candidate.chmod(0o755)
            input_path = directory / "input.las"
            input_path.write_bytes(minimal_las_header())
            frozen = directory / "frozen-time.so"
            frozen.write_bytes(b"")
            work_dir = directory / "work-dir"
            work_dir.mkdir()
            report = directory / "report.json"
            previous_argv = sys.argv
            sys.argv = [
                str(ARGS.runner),
                "--oracle", str(oracle),
                "--candidate", str(candidate),
                "--input", str(input_path),
                "--input-label", "validation",
                "--work-dir", str(work_dir),
                "--report", str(report),
                "--frozen-time-library", str(frozen),
                "--pipeline", str(pipeline),
                "--candidate-mode", "default",
                "--require-lof-parallel-repair",
                "--candidate-native-workers", "4",
                "--runs", "1",
                "--warmups", "0",
            ]
            try:
                self.assertEqual(RUNNER.main(), 0)
            finally:
                sys.argv = previous_argv
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["candidate"]["native_workers"], 4)
            self.assertTrue(result["candidate"][
                "lof_parallel_repair_required"])

    def test_main_records_lof_kd3_coordinate_cache_requirements(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-benchmark-unit-") as root:
            directory = Path(root)
            pipeline = directory / "pipeline.json"
            pipeline.write_text(json.dumps({
                "pipeline": [
                    {"type": "readers.las", "filename": "input.las"},
                    {"type": "writers.las", "filename": "output.las"},
                ]
            }) + "\n", encoding="utf-8")
            oracle = directory / "oracle"
            candidate = directory / "candidate"
            script = """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "version":
    print("benchmark-runner-test")
    raise SystemExit(0)

if sys.argv[1] == "translate":
    Path(sys.argv[3]).write_bytes(b"exact")
    raise SystemExit(0)

if sys.argv[1] == "pipeline":
    pipeline = json.loads(Path(sys.argv[2]).read_text(encoding=\"utf-8\"))
    Path(pipeline[\"pipeline\"][-1][\"filename\"]).write_bytes(b\"exact\")
    raise SystemExit(0)

raise SystemExit(1)
"""
            oracle.write_text(script, encoding="utf-8")
            candidate.write_text(script, encoding="utf-8")
            oracle.chmod(0o755)
            candidate.chmod(0o755)
            input_path = directory / "input.las"
            input_path.write_bytes(minimal_las_header())
            frozen = directory / "frozen-time.so"
            frozen.write_bytes(b"")
            work_dir = directory / "work-dir"
            work_dir.mkdir()
            report = directory / "report.json"
            previous_argv = sys.argv
            sys.argv = [
                str(ARGS.runner),
                "--oracle", str(oracle),
                "--candidate", str(candidate),
                "--input", str(input_path),
                "--input-label", "validation",
                "--work-dir", str(work_dir),
                "--report", str(report),
                "--frozen-time-library", str(frozen),
                "--pipeline", str(pipeline),
                "--candidate-mode", "default",
                "--require-lof-kd3-coordinate-cache",
                "--runs", "1",
                "--warmups", "0",
            ]
            try:
                self.assertEqual(RUNNER.main(), 0)
            finally:
                sys.argv = previous_argv
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(result["candidate"][
                "lof_kd3_coordinate_cache_required"])


if __name__ == "__main__":
    unittest.main(argv=[__file__, *UNITTEST_ARGS])
