#!/usr/bin/env python3
"""Contract tests for the complete reference-workload harness."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    return parser.parse_args()


ARGS = parse_args()


class ReferenceRunnerEnvironmentTest(unittest.TestCase):
    def make_executable(self, path: Path, environment_log: Path) -> None:
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, sys\n"
            f"log = pathlib.Path({str(environment_log)!r})\n"
            "if len(sys.argv) > 1 and sys.argv[1] == 'info':\n"
            "    print(json.dumps({'summary': {\n"
            "        'bounds': {'minx': 0, 'maxx': 10, 'miny': 0, 'maxy': 20},\n"
            "        'num_points': 4}}))\n"
            "    raise SystemExit(0)\n"
            "document = json.loads(pathlib.Path(sys.argv[2]).read_text())\n"
            "output = pathlib.Path(document['pipeline'][-1]['filename'])\n"
            "output.write_bytes(b'exact-reference-artifact')\n"
            "log.write_text(json.dumps({k: v for k, v in os.environ.items()\n"
            "                            if k.startswith('PDG_') or\n"
            "                            k.startswith('PDAL_TEST_')},\n"
            "                           sort_keys=True))\n",
            encoding="utf-8",
        )
        path.chmod(0o755)

    def make_artifact_executable(self, path: Path, payload: bytes) -> None:
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "if len(sys.argv) > 1 and sys.argv[1] == 'info':\n"
            "    print(json.dumps({'summary': {\n"
            "        'bounds': {'minx': 0, 'maxx': 10, 'miny': 0, 'maxy': 20},\n"
            "        'num_points': 4}}))\n"
            "    raise SystemExit(0)\n"
            "document = json.loads(pathlib.Path(sys.argv[2]).read_text())\n"
            "output = pathlib.Path(document['pipeline'][-1]['filename'])\n"
            f"output.write_bytes(bytes.fromhex({payload.hex()!r}))\n",
            encoding="utf-8",
        )
        path.chmod(0o755)

    def common_command(self, root: Path, oracle: Path, candidate: Path) -> list[str]:
        fixture = root / "fixture.laz"
        fixture.write_bytes(b"fixture")
        pipeline = root / "pipeline.json"
        pipeline.write_text(
            json.dumps({"pipeline": [
                {"type": "readers.las", "filename": "input.laz"},
                {"type": "writers.las", "filename": "output.laz"},
            ]}),
            encoding="utf-8",
        )
        return [
            sys.executable, str(ARGS.runner),
            "--oracle", str(oracle),
            "--candidate", str(candidate),
            "--pipeline", str(pipeline),
            "--fixture-laz", str(fixture),
            "--label", "environment-contract",
            "--work-dir", str(root / "work"),
            "--report", str(root / "report.json"),
            "--runs", "1", "--warmups", "0",
        ]

    def test_scrubs_ambient_pdg_and_injects_recorded_candidate_only_values(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-reference-runner-") as temporary:
            root = Path(temporary)
            oracle_log = root / "oracle-env.json"
            candidate_log = root / "candidate-env.json"
            oracle = root / "oracle"
            candidate = root / "candidate"
            self.make_executable(oracle, oracle_log)
            self.make_executable(candidate, candidate_log)
            command = self.common_command(root, oracle, candidate)
            command.extend([
                "--candidate-env",
                "PDG_REQUIRE_AUTOMATIC_R2_GROUND_NORMALIZE=1",
            ])
            environment = os.environ.copy()
            environment["PDG_NATIVE_WORKERS"] = "99"
            environment["PDG_INTERNAL_AUTOMATIC_R2_HYBRID"] = "poison"
            environment["PDAL_TEST_FROZEN_EPOCH"] = "poison"
            environment["PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS"] = "3"
            completed = subprocess.run(
                command, env=environment, capture_output=True, text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(oracle_log.read_text()), {})
            self.assertEqual(
                json.loads(candidate_log.read_text()),
                {"PDG_REQUIRE_AUTOMATIC_R2_GROUND_NORMALIZE": "1"},
            )
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(
                report["environment"]["candidate_overrides"],
                {"PDG_REQUIRE_AUTOMATIC_R2_GROUND_NORMALIZE": "1"},
            )
            self.assertEqual(report["environment"]["scrubbed_prefixes"],
                             ["PDG_", "PDAL_TEST_", "LD_"])
            self.assertEqual(
                report["environment"]["scrubbed_variables"],
                ["PDAL_TEST_FROZEN_EPOCH"],
            )
            self.assertIsNone(report["environment"]["freeze_environment"])

    def test_rejects_public_injection_of_internal_marker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-reference-runner-") as temporary:
            root = Path(temporary)
            oracle = root / "oracle"
            candidate = root / "candidate"
            self.make_executable(oracle, root / "oracle-env.json")
            self.make_executable(candidate, root / "candidate-env.json")
            command = self.common_command(root, oracle, candidate)
            command.extend([
                "--candidate-env", "PDG_INTERNAL_AUTOMATIC_R2_HYBRID=1",
            ])
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("internal environment", completed.stderr)

    def test_rejects_public_injection_of_harness_clock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-reference-runner-") as temporary:
            root = Path(temporary)
            oracle = root / "oracle"
            candidate = root / "candidate"
            self.make_executable(oracle, root / "oracle-env.json")
            self.make_executable(candidate, root / "candidate-env.json")
            command = self.common_command(root, oracle, candidate)
            command.extend([
                "--environment", "PDAL_TEST_FROZEN_EPOCH=1",
            ])
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("harness-owned", completed.stderr)

    def test_materializes_named_fixtures_geometry_and_multi_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-reference-runner-") as temporary:
            root = Path(temporary)
            fixture = root / "primary.laz"
            merge_a = root / "merge-a.laz"
            merge_b = root / "merge-b.laz"
            for path, payload in (
                    (fixture, b"primary"), (merge_a, b"left"),
                    (merge_b, b"right")):
                path.write_bytes(payload)
            pipeline = root / "pipeline.json"
            pipeline.write_text(json.dumps({"pipeline": [
                {"type": "readers.las", "filename": "input.merge-a.laz"},
                {"type": "readers.las", "filename": "input.merge-b.laz"},
                {"type": "filters.crop",
                 "polygon": "REPLACE_CLIP_MULTIPOLYGON_WKT",
                 "a_srs": "EPSG:4326"},
                {"type": "writers.las", "filename": "output-tile-#.laz"},
            ]}) + "\n", encoding="utf-8")

            program = root / "pipeline-program"
            program.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, re, sys\n"
                "if len(sys.argv) > 1 and sys.argv[1] == 'info':\n"
                "    print(json.dumps({'summary': {\n"
                "      'bounds': {'minx': 184500, 'maxx': 186000,\n"
                "                 'miny': 494923, 'maxy': 495000},\n"
                "      'num_points': 1000}}))\n"
                "    raise SystemExit(0)\n"
                "doc = json.loads(pathlib.Path(sys.argv[2]).read_text())\n"
                f"assert doc['pipeline'][0]['filename'] == {str(merge_a)!r}\n"
                f"assert doc['pipeline'][1]['filename'] == {str(merge_b)!r}\n"
                "wkt = doc['pipeline'][2]['polygon']\n"
                "assert wkt.startswith('MULTIPOLYGON(((') and 'REPLACE_' not in wkt\n"
                "coordinates = [float(value) for value in "
                "re.findall(r'-?[0-9]+(?:\\.[0-9]+)?', wkt)]\n"
                "assert all(4.0 < value < 7.0 for value in coordinates[0::2]), wkt\n"
                "assert all(50.0 < value < 54.0 for value in coordinates[1::2]), wkt\n"
                "pattern = doc['pipeline'][-1]['filename']\n"
                "pathlib.Path(pattern.replace('#', '1')).write_bytes(b'tile-one')\n"
                "pathlib.Path(pattern.replace('#', '2')).write_bytes(b'tile-two')\n",
                encoding="utf-8")
            program.chmod(0o755)
            report = root / "report.json"
            completed = subprocess.run([
                sys.executable, str(ARGS.runner),
                "--oracle", str(program), "--candidate", str(program),
                "--pipeline", str(pipeline), "--fixture-laz", str(fixture),
                "--fixture", f"input.merge-a.laz={merge_a}",
                "--fixture", f"input.merge-b.laz={merge_b}",
                "--label", "named-multi-output", "--work-dir", str(root / "work"),
                "--report", str(report), "--runs", "1", "--warmups", "0",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(result["comparison"]["exact_outputs"])
            self.assertEqual(result["comparison"]["artifact_names"], [
                "output-tile-1.laz", "output-tile-2.laz"])

    def test_rejects_fixture_alias_and_duplicate_output_patterns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-reference-runner-") as temporary:
            root = Path(temporary)
            fixture = root / "fixture.laz"
            fixture.write_bytes(b"immutable-fixture")
            oracle = root / "oracle"
            candidate = root / "candidate"
            self.make_executable(oracle, root / "oracle-env.json")
            self.make_executable(candidate, root / "candidate-env.json")

            alias = root / "alias.json"
            alias.write_text(json.dumps({"pipeline": [
                {"type": "readers.las", "filename": "input.laz"},
                {"type": "writers.las", "filename": "input.laz"},
            ]}), encoding="utf-8")
            command = self.common_command(root, oracle, candidate)
            command[command.index(str(root / "pipeline.json"))] = str(alias)
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("declares no output artifact", completed.stderr)
            self.assertEqual(fixture.read_bytes(), b"fixture")

            collision = root / "collision.json"
            collision.write_text(json.dumps({"pipeline": [
                {"type": "readers.las", "filename": "input.laz"},
                {"type": "writers.las", "filename": "output.laz"},
                {"type": "writers.las", "filename": "output.laz"},
            ]}), encoding="utf-8")
            command[command.index(str(alias))] = str(collision)
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate output artifact pattern", completed.stderr)

    def test_cold_cache_mode_is_explicit_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pdg-reference-runner-") as temporary:
            root = Path(temporary)
            oracle = root / "oracle"
            candidate = root / "candidate"
            self.make_executable(oracle, root / "oracle-env.json")
            self.make_executable(candidate, root / "candidate-env.json")
            command = self.common_command(root, oracle, candidate)
            command.extend(["--cache-state", "cold"])
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["environment"]["cache_state"], "cold")
            self.assertEqual(
                report["environment"]["cache_eviction"],
                "POSIX_FADV_DONTNEED on every registered input before every run")

    def test_reports_las_legacy_and_extended_counts_and_omits_invalid_headers(self) -> None:
        las14 = bytearray(375)
        las14[:4] = b"LASF"
        las14[24:26] = bytes((1, 4))
        struct.pack_into("<I", las14, 107, 0)
        struct.pack_into("<Q", las14, 247, 473825)

        las12 = bytearray(227)
        las12[:4] = b"LASF"
        las12[24:26] = bytes((1, 2))
        struct.pack_into("<I", las12, 107, 12345)

        invalid_signature = bytearray(las14)
        invalid_signature[:4] = b"NOPE"
        cases = (
            ("las14", bytes(las14), 473825),
            ("las12", bytes(las12), 12345),
            ("invalid-signature", bytes(invalid_signature), None),
            ("short-header", b"LASF\x00\x00", None),
        )
        for name, payload, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                    prefix="pdg-reference-runner-") as temporary:
                root = Path(temporary)
                program = root / "pipeline-program"
                self.make_artifact_executable(program, payload)
                command = self.common_command(root, program, program)
                pipeline = root / "pipeline.json"
                document = json.loads(pipeline.read_text(encoding="utf-8"))
                document["pipeline"][-1]["filename"] = "output.las"
                pipeline.write_text(json.dumps(document), encoding="utf-8")
                completed = subprocess.run(
                    command, capture_output=True, text=True, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                report = json.loads((root / "report.json").read_text())
                measured = [record for record in report["runs"]
                            if record["phase"] == "measured"]
                self.assertEqual(len(measured), 2)
                for record in measured:
                    artifact = record["artifacts"][0]
                    if expected is None:
                        self.assertNotIn("las_point_count", artifact)
                    else:
                        self.assertEqual(artifact["las_point_count"], expected)

    def test_fast_contract_compares_ordered_point_records_only(self) -> None:
        # D0261: two uncompressed LAS 1.4 files whose point-record blocks are
        # identical but whose header metadata (generating software) differs.
        # The exact contract must reject them; the fast contract must accept
        # them and record the digest; a swapped record order must be rejected
        # by both.
        def las(records: bytes, software: bytes) -> bytes:
            header = bytearray(375)
            header[:4] = b"LASF"
            header[24:26] = bytes((1, 4))
            header[62:94] = software.ljust(32, b"\0")[:32]
            struct.pack_into("<H", header, 94, 375)
            struct.pack_into("<I", header, 96, 375)
            header[104] = 7
            struct.pack_into("<H", header, 105, 36)
            struct.pack_into("<I", header, 107, 0)
            struct.pack_into("<Q", header, 247, len(records) // 36)
            return bytes(header) + records

        record_a = bytes(range(36))
        record_b = bytes(reversed(range(36)))
        oracle_payload = las(record_a + record_b, b"pinned")
        same_records = las(record_a + record_b, b"candidate")
        swapped_records = las(record_b + record_a, b"pinned")
        # D0271: one record whose non-coordinate bytes differ (a tie-order
        # attribute) is accepted under fast within the allowance, reported
        # as a differing record, and rejected by exact; a coordinate change
        # is rejected by both.
        record_b_attr = record_b[:12] + bytes(24)
        attribute_differs = las(record_a + record_b_attr, b"pinned")
        record_b_xyz = bytes(12) + record_b[12:]
        coordinate_differs = las(record_a + record_b_xyz, b"pinned")
        cases = (
            ("same-records-different-header", same_records, "exact", False),
            ("same-records-different-header", same_records, "fast", True),
            ("swapped-records", swapped_records, "exact", False),
            ("swapped-records", swapped_records, "fast", False),
            ("attribute-differs", attribute_differs, "exact", False),
            ("attribute-differs", attribute_differs, "fast", True),
            ("attribute-differs-over-allowance", attribute_differs, "fast",
             False),
            ("coordinate-differs", coordinate_differs, "exact", False),
            ("coordinate-differs", coordinate_differs, "fast", False),
        )
        for name, candidate_payload, contract, expected in cases:
            with self.subTest(name=name, contract=contract), \
                    tempfile.TemporaryDirectory(
                        prefix="pdg-reference-runner-") as temporary:
                root = Path(temporary)
                oracle = root / "oracle-program"
                candidate = root / "candidate-program"
                self.make_artifact_executable(oracle, oracle_payload)
                self.make_artifact_executable(candidate, candidate_payload)
                command = self.common_command(root, oracle, candidate)
                pipeline = root / "pipeline.json"
                document = json.loads(pipeline.read_text(encoding="utf-8"))
                document["pipeline"][-1]["filename"] = "output.las"
                pipeline.write_text(json.dumps(document), encoding="utf-8")
                command.extend(["--contract", contract])
                if name == "attribute-differs-over-allowance":
                    # One of two records differs (50%); a 10% allowance
                    # must refuse it.
                    command.extend(
                        ["--fast-max-differing-records-fraction", "0.1"])
                elif name == "attribute-differs":
                    command.extend(
                        ["--fast-max-differing-records-fraction", "0.5"])
                completed = subprocess.run(
                    command, capture_output=True, text=True, check=False)
                # The runner refuses a performance result (exit 1) whenever the
                # selected contract fails, but still writes the report.
                self.assertEqual(completed.returncode, 0 if expected else 1,
                                 completed.stderr)
                report = json.loads((root / "report.json").read_text())
                comparison = report["comparison"]
                self.assertEqual(comparison["contract"], contract)
                self.assertEqual(comparison["exact_outputs"], expected)
                self.assertFalse(comparison["exact_artifacts"])
                if contract == "fast":
                    self.assertEqual(comparison["exact_records"], expected)
                    digests = {
                        record["artifacts"][0]["las_record_sha256"]
                        for record in report["runs"]
                        if record["phase"] == "measured"}
                    identical = name == "same-records-different-header"
                    self.assertEqual(len(digests), 1 if identical else 2)
                    self.assertEqual(comparison["identical_records"],
                                     identical and expected)
                    if name.startswith("attribute-differs"):
                        self.assertEqual(comparison["fast_differing_records"],
                                         1)
                        self.assertEqual(comparison["fast_compared_records"],
                                         2)
                    if name == "coordinate-differs":
                        candidate_records = [
                            record["artifacts"][0] for record in report["runs"]
                            if record["phase"] == "measured"
                            and record["label"] == "candidate"]
                        self.assertTrue(all(
                            a["las_records_xyz_differing"] == 1
                            for a in candidate_records))
                else:
                    self.assertIsNone(comparison["exact_records"])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
