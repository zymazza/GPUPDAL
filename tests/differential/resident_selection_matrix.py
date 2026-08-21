#!/usr/bin/env python3
"""Exactness matrix for shapes generalized automatic selection would admit.

B0178 isolated a class of pipelines -- one neighborhood consumer followed by an
assign -- where runtime placement already accepts the graph and the resident
executor already produces byte-exact output about 3.6x faster, but the public
command withholds it because automatic selection matches a whitelist of exact
named shapes (B0148, B0149).

B0149 measured that generalization and reverted it on two objections. B0156
removed the first by making a declining pipeline free. The second is the proof
burden: generalized selection would run the resident executor on graphs the
exactness matrices do not currently cover. This matrix discharges that burden
for the class in question, so the burden is paid whether or not the gate is
ever flipped.

Every case runs the same pipeline twice -- once through pinned upstream PDAL,
once through `gpupal resident`, which is the executor generalized selection would
choose -- and requires byte-identical output. A case that cannot reach the
resident executor is reported rather than skipped silently, because a shape
that stops being admitted is exactly the regression this matrix should catch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile


CONSUMERS = (
    ("normal", {"type": "filters.normal", "knn": 8}, "NormalZ"),
    ("eigenvalues", {"type": "filters.eigenvalues", "knn": 8}, "Eigenvalue0"),
    ("covariancefeatures",
     {"type": "filters.covariancefeatures", "knn": 8, "feature_set": "Dimensionality"},
     "Linearity"),
    ("nndistance", {"type": "filters.nndistance", "mode": "kth", "k": 10},
     "NNDistance"),
    ("approximatecoplanar",
     {"type": "filters.approximatecoplanar", "knn": 8}, "Coplanar"),
)

# Assignment targets are chosen to exercise distinct physical widths on the
# writer side: unsigned byte, unsigned short, and the binary64 path.
TARGETS = ("UserData", "Intensity", "PointSourceId")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--input", type=pathlib.Path,
                        help="optional LAS fixture; one is generated when absent")
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--frozen-time-library", type=pathlib.Path)
    parser.add_argument("--freeze-epoch", type=int, default=1704067200)
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def las_layout(path: pathlib.Path) -> tuple[int, int, bool]:
    with path.open("rb") as stream:
        header = stream.read(107)
    if len(header) != 107:
        raise ValueError(f"truncated LAS header: {path}")
    raw_format = header[104]
    return (raw_format & 0x3f,
            struct.unpack_from("<H", header, 105)[0],
            bool(raw_format & 0x80))


def las_stride(path: pathlib.Path) -> int:
    return las_layout(path)[1]


def feature_pipeline(source: pathlib.Path,
                     output: pathlib.Path,
                     compression_spelling: str = "string") -> dict[str, object]:
    writer = {"type": "writers.las", "filename": str(output),
              "extra_dims": "all"}
    if (output.suffix.lower() == ".laz" and
            compression_spelling == "string"):
        writer["compression"] = "true"
    elif (output.suffix.lower() == ".laz" and
          compression_spelling == "boolean"):
        writer["compression"] = True
    elif compression_spelling != "implicit":
        if output.suffix.lower() == ".laz":
            raise ValueError(
                f"unknown compression spelling: {compression_spelling}")
    return {"pipeline": [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.normal", "knn": 8},
        {"type": "filters.covariancefeatures", "knn": 8,
         "feature_set": "Dimensionality"},
        writer,
    ]}


def main() -> int:
    args = parse_args()

    environment = {"LC_ALL": "C", "TZ": "UTC"}
    import os
    environment = dict(os.environ, **environment)
    if args.frozen_time_library:
        environment["LD_PRELOAD"] = str(args.frozen_time_library.resolve())
        environment["PDAL_TEST_FROZEN_EPOCH"] = str(args.freeze_epoch)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    executed = 0
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pdg-resident-selection-",
                                     dir=args.work_dir) as temporary:
        root = pathlib.Path(temporary)
        source_las = args.input
        if source_las is None or not source_las.exists():
            # Self-contained fixture: a deterministic grid large enough for a
            # knn=8 neighborhood, generated by the pinned oracle so the matrix
            # never depends on unversioned bench data.
            source_las = root / "fixture.las"
            recipe = root / "fixture.json"
            recipe.write_text(json.dumps({"pipeline": [
                # 150 x 150 x 12 = 270,000 points, above the placement
                # models' 250,000-point floor. A smaller fixture declines
                # correctly rather than incorrectly, but then exercises the
                # host path instead of the lane this matrix covers.
                {"type": "readers.faux", "bounds": "([0,150],[0,150],[0,12])",
                 "mode": "grid"},
                {"type": "writers.las", "filename": str(source_las)},
            ]}, indent=1), encoding="utf-8")
            built = subprocess.run([str(args.oracle), "pipeline", str(recipe)],
                                   env=environment, capture_output=True)
            if built.returncode != 0 or not source_las.exists():
                print("resident selection matrix could not build its fixture: "
                      + built.stderr.decode("utf-8", "replace")[:200],
                      file=sys.stderr)
                return 1

        # B0223's automatic performance envelope is exactly the measured
        # format-7, 36 -> 100 byte, 1M-point row. Build that source separately
        # from the smaller general selection fixture, then build a compact
        # source carrying preexisting Extra Bytes for the fail-closed layout
        # and preservation control.
        feature_sources = {
            "measured": {
                "path": root / "features-measured-source.laz",
                "bounds": "([0,100],[0,100],[0,100])",
                "dataformat_id": 7,
                "ferries": [],
                "expected_layout": (7, 36, True),
            },
            "plain-measured": {
                "path": root / "features-plain-measured-source.las",
                "bounds": "([0,100],[0,100],[0,100])",
                "dataformat_id": 7,
                "ferries": [],
                "expected_layout": (7, 36, False),
            },
            "carried-extra": {
                "path": root / "features-carried-source.las",
                "bounds": "([0,20],[0,20],[0,20])",
                "dataformat_id": 7,
                "ferries": ["X=>ExistingValue"],
            },
            "carried-extra-large": {
                "path": root / "features-carried-large-source.las",
                "bounds": "([0,65],[0,65],[0,65])",
                "dataformat_id": 7,
                "ferries": ["X=>ExistingValue"],
            },
            "veil-f6-laz": {
                "path": root / "features-format6-source.laz",
                "bounds": "([0,65],[0,65],[0,65])",
                "dataformat_id": 6,
                "ferries": [],
                "expected_layout": (6, 30, True),
            },
            "veil-f6-u8-laz": {
                "path": root / "features-format6-u8-source.laz",
                "bounds": "([0,65],[0,65],[0,65])",
                "dataformat_id": 6,
                "seed_standard_layout": True,
                "ferries": ["UserData=>ExistingU8"],
                "expected_layout": (6, 31, True),
            },
            "modern-f8-min": {
                "path": root / "features-format8-source.laz",
                "bounds": "([0,65],[0,65],[0,65])",
                "dataformat_id": 8,
                "ferries": [],
                "expected_layout": (8, 38, True),
            },
            "modern-f8-carried": {
                "path": root / "features-format8-carried-source.las",
                "bounds": "([0,65],[0,65],[0,65])",
                "dataformat_id": 8,
                "seed_standard_layout": True,
                "ferries": [
                    "Intensity=>ExistingU16",
                    "UserData=>ExistingU8",
                    "X=>ExistingDouble",
                ],
            },
        }
        for fixture_name, fixture in feature_sources.items():
            source = fixture["path"]
            if fixture.get("seed_standard_layout"):
                seed = root / f"{fixture_name}-standard-seed.las"
                seed_recipe = root / f"{fixture_name}-standard-seed.json"
                seed_recipe.write_text(json.dumps({"pipeline": [
                    {"type": "readers.faux", "bounds": fixture["bounds"],
                     "mode": "grid"},
                    {"type": "writers.las", "filename": str(seed),
                     "minor_version": 4,
                     "dataformat_id": fixture["dataformat_id"]},
                ]}, indent=1), encoding="utf-8")
                seeded = subprocess.run(
                    [str(args.oracle), "pipeline", str(seed_recipe)],
                    env=environment, capture_output=True)
                if seeded.returncode != 0 or not seed.exists():
                    print(f"resident selection matrix could not build "
                          f"{fixture_name} standard seed: "
                          + seeded.stderr.decode("utf-8", "replace")[:200],
                          file=sys.stderr)
                    return 1
                stages = [{"type": "readers.las", "filename": str(seed)}]
            else:
                stages = [
                    {"type": "readers.faux", "bounds": fixture["bounds"],
                     "mode": "grid"},
                ]
            for dimensions in fixture["ferries"]:
                stages.append({"type": "filters.ferry",
                               "dimensions": dimensions})
            writer = {"type": "writers.las", "filename": str(source),
                      "minor_version": 4,
                      "dataformat_id": fixture["dataformat_id"]}
            if fixture["ferries"]:
                writer["extra_dims"] = "all"
            stages.append(writer)
            recipe = root / f"{fixture_name}-source.json"
            recipe.write_text(json.dumps({"pipeline": stages}, indent=1),
                              encoding="utf-8")
            built = subprocess.run(
                [str(args.oracle), "pipeline", str(recipe)],
                env=environment, capture_output=True)
            if built.returncode != 0 or not source.exists():
                print(f"resident selection matrix could not build "
                      f"{fixture_name} fixture: "
                      + built.stderr.decode("utf-8", "replace")[:200],
                      file=sys.stderr)
                return 1
            expected_layout = fixture.get("expected_layout")
            if (expected_layout is not None and
                    las_layout(source) != expected_layout):
                print(f"resident selection matrix expected {fixture_name} "
                      f"layout={expected_layout}, got {las_layout(source)}",
                      file=sys.stderr)
                return 1
            if fixture_name in ("carried-extra", "carried-extra-large"):
                stride = las_stride(source)
                if stride != 48:
                    print(f"resident selection matrix expected carried-extra "
                          f"source stride=48, got {stride} for "
                          f"{source}", file=sys.stderr)
                    return 1

        for name, consumer, source in CONSUMERS:
            for target in TARGETS:
                case = f"{name}-to-{target}"
                executed += 1
                outputs = {}
                for role, executable, extra in (
                        ("oracle", args.oracle, []),
                        ("candidate", args.candidate, [])):
                    out = root / f"{case}-{role}.las"
                    pipeline = root / f"{case}-{role}.json"
                    pipeline.write_text(json.dumps({"pipeline": [
                        {"type": "readers.las", "filename": str(source_las)},
                        consumer,
                        {"type": "filters.assign",
                         "value": f"{target} = {source}"},
                        {"type": "writers.las", "filename": str(out)},
                    ]}, indent=1), encoding="utf-8")
                    command = [str(executable)]
                    # The candidate runs through `resident`, the executor a
                    # generalized selection would choose; the oracle runs its
                    # ordinary pipeline command.
                    command += (["resident"] if role == "candidate"
                                else ["pipeline"])
                    command += [str(pipeline)] + extra
                    if role == "candidate":
                        stats = root / f"{case}-stats.json"
                        command += ["--stats", str(stats)]
                    completed = subprocess.run(command, env=environment,
                                               capture_output=True)
                    if completed.returncode != 0 or not out.exists():
                        failures.append(
                            f"{case}: {role} returned "
                            f"{completed.returncode}: "
                            f"{completed.stderr.decode('utf-8', 'replace')[:200]}")
                        break
                    outputs[role] = sha256(out)
                else:
                    if outputs["oracle"] != outputs["candidate"]:
                        failures.append(
                            f"{case}: output differs "
                            f"({outputs['oracle'][:16]} vs "
                            f"{outputs['candidate'][:16]})")
                        continue
                    stats = root / f"{case}-stats.json"
                    executor = None
                    if stats.exists():
                        report = json.loads(stats.read_text(encoding="utf-8"))
                        executor = report.get("execution", {}).get("executor")
                    if executor == "pdal_standard_host":
                        # Exact, but it did not reach the executor this matrix
                        # exists to cover. Report rather than pass quietly.
                        failures.append(
                            f"{case}: exact but ran {executor}, so the "
                            f"resident lane was not exercised")

        # D0218's derived writer width makes the separately measured
        # extra_dims=all compositions admissible. Cover the public command,
        # not only the explicit resident executor: every computed feature must
        # survive in layout order, placement must charge the emitted record
        # stride, and the proof switch must observe automatic selection. B0224
        # adds the exact compressed sink used by r6 without widening the named
        # Extra Bytes or writer-option envelope.
        def prove_feature_case(case: str, suffix: str,
                               compression_spelling: str = "string",
                               source_name: str = "measured",
                               expected_output: tuple[int, int, bool] | None =
                               None) -> None:
            results = {}
            candidate_stats = root / f"{case}-stats.json"
            for role, executable in (("oracle", args.oracle),
                                     ("candidate", args.candidate)):
                out = root / f"{case}-{role}{suffix}"
                pipeline = root / f"{case}-{role}.json"
                pipeline.write_text(json.dumps(feature_pipeline(
                    feature_sources[source_name]["path"], out,
                    compression_spelling), indent=1),
                    encoding="utf-8")
                role_environment = dict(environment)
                if role == "candidate":
                    role_environment[
                        "PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT"] = (
                            "1")
                completed = subprocess.run(
                    [str(executable), "pipeline", str(pipeline)],
                    env=role_environment, capture_output=True)
                if completed.returncode != 0 or not out.exists():
                    failures.append(
                        f"{case}: {role} returned {completed.returncode}: "
                        f"{completed.stderr.decode('utf-8', 'replace')[:200]}")
                    return
                results[role] = {
                    "status": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "sha256": sha256(out),
                    "stride": las_stride(out),
                    "layout": las_layout(out),
                }
            if results["oracle"] != results["candidate"]:
                failures.append(f"{case}: complete process result differs")
                return
            if (expected_output is not None and
                    results["candidate"]["layout"] != expected_output):
                failures.append(
                    f"{case}: expected output layout {expected_output}, got "
                    f"{results['candidate']['layout']}")
                return

            resident_out = root / f"{case}-resident{suffix}"
            resident_pipeline = root / f"{case}-resident.json"
            resident_pipeline.write_text(json.dumps(feature_pipeline(
                feature_sources[source_name]["path"], resident_out,
                compression_spelling), indent=1),
                encoding="utf-8")
            resident = subprocess.run(
                [str(args.candidate), "resident", str(resident_pipeline),
                 "--stats", str(candidate_stats)],
                env=environment, capture_output=True)
            if (resident.returncode != 0 or not resident_out.exists() or
                    sha256(resident_out) != results["candidate"]["sha256"]):
                failures.append(
                    f"{case}: explicit resident proof differs or failed")
                return
            report = json.loads(candidate_stats.read_text(encoding="utf-8"))
            placement = report.get("placement", {})
            execution = report.get("execution", {})
            if placement.get("output_record_bytes") != results[
                    "candidate"]["stride"]:
                failures.append(
                    f"{case}: placement output width does not match file")
            if report.get("plan", {}).get("all_stages_native") is not True:
                failures.append(f"{case}: sink is not planner-native")
            if execution.get("executor") != "planner_resident_shared_index":
                failures.append(
                    f"{case}: exact but ran {execution.get('executor')}")

        for case, suffix, compression_spelling, expected_output in (
                ("normal-covariancefeatures-extra-dims-all", ".las",
                 "string", (7, 100, False)),
                ("normal-covariancefeatures-laz-extra-dims-all", ".laz",
                 "string", (7, 100, True)),
                ("normal-covariancefeatures-laz-extra-dims-all-boolean",
                 ".laz", "boolean", (7, 100, True)),
                ("normal-covariancefeatures-laz-extra-dims-all-implicit",
                 ".laz", "implicit", (7, 100, True))):
            executed += 1
            prove_feature_case(case, suffix, compression_spelling,
                               expected_output=expected_output)

        # D0284: admit retained physical tuples, not a rectangle of layout
        # cross-products. The generated format-6/30 LAZ tuple mirrors VEIL;
        # B0285's final-engine AHN4 report separately proves
        # format-8/44 -> 7/120.
        executed += 1
        prove_feature_case(
            "normal-covariancefeatures-format6-laz-to-laz", ".laz", "string",
            "veil-f6-laz", (7, 100, True))

        def prove_feature_refusal(case: str, suffix: str,
                                  source_name: str) -> None:
            out = root / f"{case}-required{suffix}"
            pipeline = root / f"{case}-required.json"
            pipeline.write_text(json.dumps(feature_pipeline(
                feature_sources[source_name]["path"], out), indent=1),
                encoding="utf-8")
            required_environment = dict(environment)
            required_environment[
                "PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT"] = "1"
            refused = subprocess.run(
                [str(args.candidate), "pipeline", str(pipeline)],
                env=required_environment, capture_output=True)
            if refused.returncode != 124 or out.exists():
                failures.append(
                    f"{case}: unmeasured tuple was not refused "
                    f"(status={refused.returncode}, output={out.exists()})")

        for case, suffix, source_name in (
                ("normal-covariancefeatures-format6-laz-to-las", ".las",
                 "veil-f6-laz"),
                ("normal-covariancefeatures-format6-u8-laz-to-laz", ".laz",
                 "veil-f6-u8-laz"),
                ("normal-covariancefeatures-format8-min-to-laz", ".laz",
                 "modern-f8-min"),
                ("normal-covariancefeatures-format8-carried-to-laz", ".laz",
                 "modern-f8-carried"),
                ("normal-covariancefeatures-carried-extra-large-to-laz",
                 ".laz", "carried-extra-large")):
            executed += 1
            prove_feature_refusal(case, suffix, source_name)

        # The public proof must cover the executable rewrite and resident
        # preflight, not merely the earlier placement estimate.  Inject a
        # post-placement preflight refusal and require the dispatcher to fail
        # closed before the original host pipeline can write an output.
        case = "normal-covariancefeatures-laz-extra-dims-all"
        refusal_out = root / f"{case}-preflight-refusal.laz"
        refusal_pipeline = root / f"{case}-preflight-refusal.json"
        refusal_pipeline.write_text(json.dumps(feature_pipeline(
            feature_sources["measured"]["path"], refusal_out), indent=1),
            encoding="utf-8")
        refusal_environment = dict(environment)
        refusal_environment[
            "PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT"] = "1"
        refusal_environment[
            "PDG_TEST_AUTOMATIC_RESIDENT_PREFLIGHT_FAILURE"] = "1"
        refused = subprocess.run(
            [str(args.candidate), "pipeline", str(refusal_pipeline)],
            env=refusal_environment, capture_output=True)
        if refused.returncode != 124 or refusal_out.exists():
            failures.append(
                f"{case}: required public route did not fail closed after "
                f"preflight refusal (status={refused.returncode}, "
                f"output={refusal_out.exists()})")

        # B0280 follow-up: the unchanged PDAL writer was observed returning
        # success after tmpfs pressure left a partial LAS with an unfinalized
        # header. Model that bounded failure after the automatic route commits;
        # the public command must return nonzero instead of blessing the file.
        executed += 1
        publication_out = root / f"{case}-publication-truncated.laz"
        publication_pipeline = root / f"{case}-publication-truncated.json"
        publication_pipeline.write_text(json.dumps(feature_pipeline(
            feature_sources["measured"]["path"], publication_out), indent=1),
            encoding="utf-8")
        publication_environment = dict(environment)
        publication_environment[
            "PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT"] = "1"
        publication_environment[
            "PDG_TEST_AUTOMATIC_RESIDENT_PUBLICATION_TRUNCATION"] = "1"
        publication_failure = subprocess.run(
            [str(args.candidate), "pipeline", str(publication_pipeline)],
            env=publication_environment, capture_output=True)
        if (publication_failure.returncode == 0 or
                b"automatic resident LAS publication is invalid" not in
                publication_failure.stderr):
            failures.append(
                f"{case}: invalid publication was not rejected "
                f"(status={publication_failure.returncode}, "
                f"stderr={publication_failure.stderr[:200]!r})")

        # The original incident was an uncompressed LAS with a stale zero
        # count and partial raw payload. File extent alone cannot reject that
        # shape, so the command must compare the publication count with the
        # resident executor's observed output cardinality.
        executed += 1
        plain_case = "normal-covariancefeatures-plain-publication-stale-count"
        plain_out = root / f"{plain_case}.las"
        plain_pipeline = root / f"{plain_case}.json"
        plain_pipeline.write_text(json.dumps({"pipeline": [
            {"type": "readers.las",
             "filename": str(feature_sources["plain-measured"]["path"])},
            {"type": "filters.normal", "knn": 8},
            {"type": "filters.covariancefeatures", "knn": 8,
             "feature_set": "Dimensionality"},
            {"type": "writers.las", "filename": str(plain_out)},
        ]}, indent=1), encoding="utf-8")
        plain_failure = subprocess.run(
            [str(args.candidate), "pipeline", str(plain_pipeline)],
            env=publication_environment, capture_output=True)
        if (plain_failure.returncode == 0 or
                b"published LAS point count does not match resident output" not
                in plain_failure.stderr):
            failures.append(
                f"{plain_case}: stale count was not rejected "
                f"(status={plain_failure.returncode}, "
                f"stderr={plain_failure.stderr[:200]!r})")

        # A source with preexisting Extra Bytes exercises the variable-width
        # layout and full-record preservation contract. Its format-7/48 ->
        # 7/112 tuple has no retained performance evidence, so model presence
        # must not make it automatic even though execution remains exact.
        case = "normal-covariancefeatures-carried-extra-dims"
        executed += 1
        results = {}
        candidate_stats = root / f"{case}-stats.json"
        for role, executable in (("oracle", args.oracle),
                                 ("candidate", args.candidate)):
            out = root / f"{case}-{role}.las"
            pipeline = root / f"{case}-{role}.json"
            pipeline.write_text(json.dumps(feature_pipeline(
                feature_sources["carried-extra"]["path"], out), indent=1),
                encoding="utf-8")
            command = [str(executable),
                       "resident" if role == "candidate" else "pipeline",
                       str(pipeline)]
            if role == "candidate":
                command += ["--stats", str(candidate_stats)]
            completed = subprocess.run(command, env=environment,
                                       capture_output=True)
            if completed.returncode != 0 or not out.exists():
                failures.append(
                    f"{case}: {role} returned {completed.returncode}: "
                    f"{completed.stderr.decode('utf-8', 'replace')[:200]}")
                break
            results[role] = {
                "status": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "sha256": sha256(out),
                "stride": las_stride(out),
            }
        else:
            if results["oracle"] != results["candidate"]:
                failures.append(f"{case}: complete process result differs")
            else:
                expected_stride = 112
                report = json.loads(
                    candidate_stats.read_text(encoding="utf-8"))
                placement = report.get("placement", {})
                execution = report.get("execution", {})
                if (results["oracle"]["stride"] != expected_stride or
                        results["candidate"]["stride"] != expected_stride):
                    failures.append(
                        f"{case}: existing Extra Bytes were not retained with "
                        f"oracle stride {results['oracle']['stride']} and "
                        f"candidate stride {results['candidate']['stride']}, "
                        f"expected {expected_stride}")
                if report.get("plan", {}).get("all_stages_native") is not True:
                    failures.append(f"{case}: sink is not planner-native")
                if (placement.get("available") is not False or
                        placement.get("unavailable_reason") !=
                        "mixed_calibration_models"):
                    failures.append(
                        f"{case}: unmeasured layout did not fail placement "
                        f"closed")
                if execution.get("executor") != "pdal_standard_host":
                    failures.append(
                        f"{case}: wider layout ran "
                        f"{execution.get('executor')}")

    print(f"resident selection matrix: {executed} cases")
    for failure in failures:
        print(f"  FAILED {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
