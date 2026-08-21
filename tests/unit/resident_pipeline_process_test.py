#!/usr/bin/env python3
"""Process smoke for the explicit resident command and stats boundary."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile


PDG, ORACLE, INPUT = map(pathlib.Path, sys.argv[1:4])


def pipeline(path: pathlib.Path, output: pathlib.Path,
             reader_options=None, writer_options=None,
             source: pathlib.Path = INPUT) -> pathlib.Path:
    reader = {"type": "readers.las", "filename": str(source)}
    if reader_options:
        reader.update(reader_options)
    writer = {"type": "writers.las", "filename": str(output)}
    if writer_options:
        writer.update(writer_options)
    document = {
        "pipeline": [
            reader,
            {"type": "filters.assign", "value": "Classification = 7"},
            {
                "type": "filters.ferry",
                "dimensions": "Classification=>UserData",
            },
            writer,
        ]
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def lof_pipeline(path: pathlib.Path, output: pathlib.Path,
                 source: pathlib.Path = INPUT) -> pathlib.Path:
    document = {
        "pipeline": [
            {"type": "readers.las", "filename": str(source)},
            {"type": "filters.lof", "minpts": 10},
            {
                "type": "filters.assign",
                "value": ("UserData = 1 WHERE "
                          "LocalOutlierFactor >= 1.2"),
            },
            {"type": "writers.las", "filename": str(output)},
        ]
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def nndistance_pipeline(path: pathlib.Path, output: pathlib.Path,
                        options=None) -> pathlib.Path:
    stage = {"type": "filters.nndistance", "k": 10}
    if options:
        stage.update(options)
    document = {
        "pipeline": [
            {"type": "readers.las", "filename": str(INPUT)},
            stage,
            {
                "type": "filters.assign",
                "value": "UserData = 1 WHERE NNDistance >= 0.4",
            },
            {"type": "writers.las", "filename": str(output)},
        ]
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def eigen_family_pipeline(path: pathlib.Path, output: pathlib.Path,
                          source: pathlib.Path = INPUT) -> pathlib.Path:
    document = {
        "pipeline": [
            {"type": "readers.las", "filename": str(source)},
            {"type": "filters.normal", "knn": 12, "always_up": False},
            {"type": "filters.eigenvalues", "knn": 12, "normalize": True},
            {"type": "filters.covariancefeatures", "knn": 12,
             "mode": "raw", "feature_set": "dimensionality"},
            {"type": "filters.assign", "value": [
                "Classification = Linearity * 10",
                "Intensity = Curvature * 1000",
                "UserData = Eigenvalue0 * 100",
            ]},
            {"type": "writers.las", "filename": str(output)},
        ]
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def rank_optimal_pipeline(path: pathlib.Path, output: pathlib.Path,
                          source: pathlib.Path = INPUT) -> pathlib.Path:
    document = {
        "pipeline": [
            {"type": "readers.las", "filename": str(source)},
            {"type": "filters.estimaterank", "knn": 14, "thresh": 0.01},
            {"type": "filters.optimalneighborhood",
             "min_k": 10, "max_k": 14},
            {"type": "filters.assign", "value": [
                "Classification = Rank",
                "Intensity = OptimalKNN",
                "PointSourceId = OptimalRadius",
            ]},
            {"type": "writers.las", "filename": str(output)},
        ]
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


with tempfile.TemporaryDirectory(prefix="pdg-resident-process-") as temp:
    root = pathlib.Path(temp)
    candidate_dir = root / "candidate"
    oracle_dir = root / "oracle"
    plain_dir = root / "plain"
    candidate_dir.mkdir()
    oracle_dir.mkdir()
    plain_dir.mkdir()

    candidate_output = candidate_dir / "out.las"
    oracle_output = oracle_dir / "out.las"
    plain_output = plain_dir / "out.las"
    stats = candidate_dir / "stats.json"

    candidate = subprocess.run(
        [str(PDG), "resident",
         str(pipeline(candidate_dir / "pipeline.json", candidate_output)),
         "--stats", str(stats)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    oracle = subprocess.run(
        [str(ORACLE), "pipeline",
         str(pipeline(oracle_dir / "pipeline.json", oracle_output))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert candidate.returncode == oracle.returncode == 0, (
        candidate.stderr,
        oracle.stderr,
    )
    assert candidate_output.read_bytes() == oracle_output.read_bytes()

    report = json.loads(stats.read_text(encoding="utf-8"))
    assert report["schema"] == "pdg-resident-stats-v1"
    assert report["placement"]["available"] is False
    # A build without a usable CUDA profile reports profile_not_exact; the
    # exact SM-89 machine instead rejects this two-assignment region as
    # outside the measured fused envelope. Both are the intended fail-closed
    # decision.
    assert report["placement"]["unavailable_reason"] in (
        "profile_not_exact", "outside_calibration_envelope")
    assert report["execution"]["executor"] == "pdal_standard_host"
    assert (
        report["execution"]["selected_device_calibration_matches_executor"]
        is None
    )
    assert report["execution"]["selected_regions"] == []
    assert report["execution"]["observed_crossings"] == []
    assert report["execution"]["events"] == []
    assert report["execution"]["exact_host_repair"] is None
    assert report["execution"]["exact_device_repair"] is None
    assert report["execution"]["direct_las_output"] is False
    phases = report["execution"]["pipeline_phase_seconds"]
    assert set(phases) == {
        "command_before_stats",
        "validation_placement_preflight",
        "rewritten_manager_execution",
        "canonical_las_publication",
        "other_control",
    }, phases
    assert all(value >= 0.0 for value in phases.values()), phases
    assert phases["validation_placement_preflight"] > 0.0, phases
    assert phases["rewritten_manager_execution"] > 0.0, phases
    assert phases["canonical_las_publication"] == 0.0, phases
    attributed = sum(
        phases[name] for name in (
            "validation_placement_preflight",
            "rewritten_manager_execution",
            "canonical_las_publication",
            "other_control",
        )
    )
    assert abs(phases["command_before_stats"] - attributed) < 1e-9, phases
    validation_breakdown = report["execution"][
        "validation_placement_preflight_breakdown_seconds"]
    assert validation_breakdown[
        "matches_validation_placement_preflight"] is True, (
            validation_breakdown)
    assert validation_breakdown["runtime_subphases_match"] is True, (
        validation_breakdown)
    validation_parts = sum(
        validation_breakdown[name] for name in (
            "plan_and_original_validation",
            "runtime_placement",
            "rewrite_and_resident_preflight",
        )
    )
    assert validation_breakdown["plan_and_original_validation"] > 0.0, (
        validation_breakdown)
    runtime_parts = sum(
        validation_breakdown[name] for name in (
            "runtime_device_and_profile",
            "runtime_initial_placement",
            "runtime_executor_selection",
        )
    )
    assert abs(
        validation_breakdown["runtime_placement"] - runtime_parts) < 1e-9, (
            validation_breakdown)
    assert abs(validation_breakdown["total"] - validation_parts) < 1e-9, (
        validation_breakdown)
    assert abs(
        validation_breakdown["total"] -
        phases["validation_placement_preflight"]) < 1e-9, (
            validation_breakdown, phases)
    assert report["execution"][
        "rewritten_manager_execution_breakdown_seconds"] is None
    assert report["execution"]["resident_work_breakdown_seconds"] is None
    assert report["execution"][
        "index_configuration_breakdown_seconds"] is None
    assert report["execution"][
        "direct_las_hydration_breakdown_seconds"] is None
    assert report["execution"][
        "nndistance_query_breakdown_seconds"] is None

    plain = subprocess.run(
        [str(PDG), "resident",
         str(pipeline(plain_dir / "pipeline.json", plain_output))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert plain.returncode == 0, plain.stderr
    assert plain_output.read_bytes() == oracle_output.read_bytes()
    assert plain.stdout == candidate.stdout == ""
    assert plain.stderr == candidate.stderr == oracle.stderr

    # The experimental endpoint never promotes a small or empty graph merely
    # because its output shape is supported. Placement remains authoritative,
    # and the ordinary writer must stay byte-exact.
    experimental_dir = root / "experimental-small"
    experimental_dir.mkdir()
    experimental_output = experimental_dir / "out.las"
    experimental_stats = experimental_dir / "stats.json"
    experimental_environment = os.environ.copy()
    experimental_environment[
        "PDG_EXPERIMENTAL_DIRECT_RESIDENT_LAS_OUTPUT"] = "1"
    experimental = subprocess.run(
        [str(PDG), "resident",
         str(pipeline(experimental_dir / "pipeline.json",
                      experimental_output)),
         "--stats", str(experimental_stats)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=experimental_environment,
    )
    assert experimental.returncode == 0, experimental.stderr
    assert experimental_output.read_bytes() == oracle_output.read_bytes()
    experimental_report = json.loads(
        experimental_stats.read_text(encoding="utf-8"))
    assert experimental_report["execution"]["direct_las_output"] is False

    empty_input = INPUT.parent / "no-points.las"
    assert empty_input.is_file(), empty_input
    empty_candidate = experimental_dir / "empty.las"
    empty_oracle = oracle_dir / "empty.las"
    empty_stats = experimental_dir / "empty-stats.json"
    empty = subprocess.run(
        [str(PDG), "resident",
         str(pipeline(experimental_dir / "empty.json", empty_candidate,
                      source=empty_input)),
         "--stats", str(empty_stats)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=experimental_environment,
    )
    empty_reference = subprocess.run(
        [str(ORACLE), "pipeline",
         str(pipeline(oracle_dir / "empty.json", empty_oracle,
                      source=empty_input))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert empty.returncode == empty_reference.returncode == 0, (
        empty.stderr,
        empty_reference.stderr,
    )
    assert empty.stdout == empty_reference.stdout
    assert empty.stderr == empty_reference.stderr
    empty_candidate_bytes = empty_candidate.read_bytes()
    empty_oracle_bytes = empty_oracle.read_bytes()
    assert empty_candidate_bytes == empty_oracle_bytes, (
        len(empty_candidate_bytes),
        len(empty_oracle_bytes),
        [(index, left, right) for index, (left, right) in enumerate(
         zip(empty_candidate_bytes, empty_oracle_bytes)) if left != right][
             :16],
    )
    empty_report = json.loads(empty_stats.read_text(encoding="utf-8"))
    assert empty_report["execution"]["direct_las_output"] is False

    required_automatic_output = experimental_dir / "required-automatic.las"
    required_automatic_environment = os.environ.copy()
    required_automatic_environment[
        "PDG_REQUIRE_AUTOMATIC_RESIDENT_LAS_OUTPUT"] = "1"
    required_automatic = subprocess.run(
        [str(PDG), "pipeline",
         str(pipeline(experimental_dir / "required-automatic.json",
                      required_automatic_output))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=required_automatic_environment,
    )
    assert required_automatic.returncode != 0, required_automatic
    assert "required automatic resident LAS output path was not used" in (
        required_automatic.stderr), required_automatic.stderr
    assert not required_automatic_output.exists()

    # The eigen-family public selector is placement-authoritative. Requiring
    # it on this below-floor fixture must fail before the ordinary writer can
    # create an artifact; silently delegating to the host would make this
    # proof a false positive.
    required_eigen_output = experimental_dir / "required-eigen-family.las"
    required_eigen_environment = os.environ.copy()
    required_eigen_environment[
        "PDG_REQUIRE_AUTOMATIC_EIGEN_FAMILY_RESIDENT"] = "1"
    required_eigen = subprocess.run(
        [str(PDG), "pipeline",
         str(eigen_family_pipeline(
             experimental_dir / "required-eigen-family.json",
             required_eigen_output))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=required_eigen_environment,
    )
    assert required_eigen.returncode == 124, required_eigen
    assert "required automatic eigen-family resident path was not used" in (
        required_eigen.stderr), required_eigen.stderr
    assert not required_eigen_output.exists()

    required_rank_output = experimental_dir / "required-rank-optimal.las"
    required_rank_environment = os.environ.copy()
    required_rank_environment[
        "PDG_REQUIRE_AUTOMATIC_RANK_OPTIMAL_RESIDENT"] = "1"
    required_rank = subprocess.run(
        [str(PDG), "pipeline",
         str(rank_optimal_pipeline(
             experimental_dir / "required-rank-optimal.json",
             required_rank_output))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=required_rank_environment,
    )
    assert required_rank.returncode == 124, required_rank
    assert "required automatic rank/optimal resident path was not used" in (
        required_rank.stderr), required_rank.stderr
    assert not required_rank_output.exists()

    # The exact automatic JSON shape still delegates when placement declines.
    # Its ordinary and explicit-standard public CLI behavior stays upstream.
    automatic_fallback_output = experimental_dir / "automatic-fallback.las"
    automatic_fallback_oracle = oracle_dir / "automatic-fallback.las"
    automatic_fallback = subprocess.run(
        [str(PDG), "pipeline",
         str(lof_pipeline(experimental_dir / "automatic-fallback.json",
                          automatic_fallback_output))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    automatic_fallback_reference = subprocess.run(
        [str(ORACLE), "pipeline",
         str(lof_pipeline(oracle_dir / "automatic-fallback.json",
                          automatic_fallback_oracle))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert automatic_fallback.returncode == (
        automatic_fallback_reference.returncode) == 0
    assert automatic_fallback.stdout == automatic_fallback_reference.stdout
    assert automatic_fallback.stderr == automatic_fallback_reference.stderr
    assert automatic_fallback_output.read_bytes() == (
        automatic_fallback_oracle.read_bytes())

    # Public performance assertions are terminal route proofs. If automatic
    # placement declines, they must stop before the pinned oracle can create
    # an otherwise valid output and falsely satisfy the benchmark process.
    for proof_name, proof_environment, proof_message in (
        (
            "lof-parallel-repair",
            "PDG_REQUIRE_LOF_PARALLEL_REPAIR",
            "required parallel LOF repair path was not used",
        ),
        (
            "lof-kd3-coordinate-cache",
            "PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE",
            "required LOF KD3 coordinate cache path was not used",
        ),
    ):
        proof_output = experimental_dir / f"required-{proof_name}.las"
        proof_environment_values = os.environ.copy()
        proof_environment_values[proof_environment] = "1"
        proof = subprocess.run(
            [str(PDG), "pipeline",
             str(lof_pipeline(
                 experimental_dir / f"required-{proof_name}.json",
                 proof_output))],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proof_environment_values,
        )
        assert proof.returncode == 124, proof
        assert proof_message in proof.stderr, proof.stderr
        assert not proof_output.exists()

    standard_output = experimental_dir / "automatic-standard.las"
    standard_oracle = oracle_dir / "automatic-standard.las"
    standard = subprocess.run(
        [str(PDG), "pipeline",
         str(lof_pipeline(experimental_dir / "automatic-standard.json",
                          standard_output)), "--nostream"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    standard_reference = subprocess.run(
        [str(ORACLE), "pipeline",
         str(lof_pipeline(oracle_dir / "automatic-standard.json",
                          standard_oracle)), "--nostream"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert standard.returncode == standard_reference.returncode == 0
    assert standard.stdout == standard_reference.stdout
    assert standard.stderr == standard_reference.stderr
    assert standard_output.read_bytes() == standard_oracle.read_bytes()

    stream_output = experimental_dir / "automatic-stream.las"
    stream_oracle = oracle_dir / "automatic-stream.las"
    stream = subprocess.run(
        [str(PDG), "pipeline",
         str(lof_pipeline(experimental_dir / "automatic-stream.json",
                          stream_output)), "--stream"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stream_reference = subprocess.run(
        [str(ORACLE), "pipeline",
         str(lof_pipeline(oracle_dir / "automatic-stream.json",
                          stream_oracle)), "--stream"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert stream.returncode == stream_reference.returncode != 0
    assert stream.stdout == stream_reference.stdout
    assert stream.stderr == stream_reference.stderr
    assert not stream_output.exists()
    assert not stream_oracle.exists()

    # Metadata is an observable public CLI artifact, so an option-bearing
    # command stays wholly on the existing path. Use the same paths in both
    # sequential runs to make the complete metadata bytes comparable.
    metadata_output = root / "automatic-metadata.las"
    metadata_path = root / "automatic-metadata.json"
    metadata_pipeline = lof_pipeline(
        root / "automatic-metadata-pipeline.json", metadata_output)
    metadata = subprocess.run(
        [str(PDG), "pipeline", str(metadata_pipeline), "--metadata",
         str(metadata_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert metadata.returncode == 0, metadata.stderr
    metadata_output_bytes = metadata_output.read_bytes()
    metadata_bytes = metadata_path.read_bytes()
    metadata_output.unlink()
    metadata_path.unlink()
    metadata_reference = subprocess.run(
        [str(ORACLE), "pipeline", str(metadata_pipeline), "--metadata",
         str(metadata_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert metadata.returncode == metadata_reference.returncode == 0
    assert metadata.stdout == metadata_reference.stdout
    assert metadata.stderr == metadata_reference.stderr
    assert metadata_output_bytes == metadata_output.read_bytes()
    assert metadata_bytes == metadata_path.read_bytes()

    # A failure discovered while inspecting an otherwise eligible public
    # shape delegates before commitment and retains upstream diagnostics.
    missing_input = root / "missing-input.las"
    missing_output = root / "missing-output.las"
    missing_pipeline = lof_pipeline(
        root / "missing-input.json", missing_output, missing_input)
    missing = subprocess.run(
        [str(PDG), "pipeline", str(missing_pipeline)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    missing_reference = subprocess.run(
        [str(ORACLE), "pipeline", str(missing_pipeline)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert missing.returncode == missing_reference.returncode != 0
    assert missing.stdout == missing_reference.stdout
    assert missing.stderr == missing_reference.stderr
    assert not missing_output.exists()

    # B0045's second automatic shape still delegates below placement, and an
    # unmeasured stage option never borrows the exact k=10 admission.
    nndistance_output = experimental_dir / "automatic-nndistance.las"
    nndistance_oracle = oracle_dir / "automatic-nndistance.las"
    nndistance = subprocess.run(
        [str(PDG), "pipeline",
         str(nndistance_pipeline(
             experimental_dir / "automatic-nndistance.json",
             nndistance_output))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    nndistance_reference = subprocess.run(
        [str(ORACLE), "pipeline",
         str(nndistance_pipeline(
             oracle_dir / "automatic-nndistance.json",
             nndistance_oracle))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert nndistance.returncode == nndistance_reference.returncode == 0
    assert nndistance.stdout == nndistance_reference.stdout
    assert nndistance.stderr == nndistance_reference.stderr
    assert nndistance_output.read_bytes() == nndistance_oracle.read_bytes()

    nndistance_required_output = (
        experimental_dir / "required-nndistance.las")
    nndistance_required = subprocess.run(
        [str(PDG), "pipeline",
         str(nndistance_pipeline(
             experimental_dir / "required-nndistance.json",
             nndistance_required_output))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=required_automatic_environment,
    )
    assert nndistance_required.returncode != 0, nndistance_required
    assert "required automatic resident LAS output path was not used" in (
        nndistance_required.stderr), nndistance_required.stderr
    assert not nndistance_required_output.exists()

    direct_source_required_output = (
        experimental_dir / "required-direct-source.las")
    direct_source_required_environment = os.environ.copy()
    direct_source_required_environment[
        "PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
    direct_source_required_environment[
        "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
    direct_source_required = subprocess.run(
        [str(PDG), "pipeline",
         str(nndistance_pipeline(
             experimental_dir / "required-direct-source.json",
             direct_source_required_output))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=direct_source_required_environment,
    )
    assert direct_source_required.returncode == 124, direct_source_required
    assert "required direct LAS resident source path was not used" in (
        direct_source_required.stderr), direct_source_required.stderr
    assert not direct_source_required_output.exists()

    for proof_name, proof_environment, proof_value, proof_message in (
        (
            "record-summary",
            "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY",
            "1",
            "required direct LAS record summary path was not used",
        ),
        (
            "no-host-xyz",
            "PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ",
            "1",
            "required direct LAS no-host-XYZ path was not used",
        ),
        (
            "record-summary-backend",
            "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY_BACKEND",
            "uniform_grid",
            "required direct LAS record summary path was not used",
        ),
        (
            "nnd-device-only-handoff",
            "PDG_REQUIRE_NND_DEVICE_ONLY_HANDOFF",
            "1",
            "required NNDistance device-only assignment handoff was not used",
        ),
        (
            "nnd-host-restore",
            "PDG_REQUIRE_NND_HOST_RESTORE",
            "1",
            "required NNDistance host restoration path was not used",
        ),
        (
            "nnd-parallel-repair",
            "PDG_REQUIRE_NND_PARALLEL_REPAIR",
            "1",
            "required parallel selective NNDistance device repair path was "
            "not used",
        ),
    ):
        proof_output = experimental_dir / f"required-{proof_name}.las"
        proof_env = os.environ.copy()
        proof_env[proof_environment] = proof_value
        proof_env["PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
        proof = subprocess.run(
            [str(PDG), "pipeline",
             str(nndistance_pipeline(
                 experimental_dir / f"required-{proof_name}.json",
                 proof_output))],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proof_env,
        )
        assert proof.returncode == 124, proof
        assert proof_message in proof.stderr, proof.stderr
        assert not proof_output.exists()

        resident_proof_output = (
            experimental_dir / f"required-{proof_name}-resident.las")
        resident_proof_pipeline = nndistance_pipeline(
            experimental_dir / f"required-{proof_name}-resident.json",
            resident_proof_output)
        resident_proof = subprocess.run(
            [str(PDG), "resident", str(resident_proof_pipeline)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proof_env,
        )
        assert resident_proof.returncode != 0, resident_proof
        assert "required direct LAS resident source path was not used" in (
            resident_proof.stderr), resident_proof.stderr
        assert not resident_proof_output.exists()

    # Explicit `gpupdal resident` returns before main's automatic-fallback proof
    # guards. The resident core itself must reject a repair proof when no
    # selected matching stage can evaluate it, before the ordinary host writer
    # creates an artifact.
    for proof_name, proof_environment, proof_message in (
        ("nnd-device-repair", "PDG_REQUIRE_NND_DEVICE_REPAIR",
         "required NNDistance device repair path was "),
        ("outlier-device-repair", "PDG_REQUIRE_OUTLIER_DEVICE_REPAIR",
         "required statistical-outlier device repair path was "),
        ("outlier-parallel-repair",
         "PDG_REQUIRE_OUTLIER_PARALLEL_REPAIR",
         "required statistical-outlier device repair path was "),
        ("lof-parallel-repair", "PDG_REQUIRE_LOF_PARALLEL_REPAIR",
         "required LOF repair path was not selected"),
        ("lof-kd3-coordinate-cache",
         "PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE",
         "required LOF repair path was not selected"),
    ):
        proof_output = experimental_dir / f"required-{proof_name}.las"
        proof_env = os.environ.copy()
        proof_env[proof_environment] = "1"
        resident_proof = subprocess.run(
            [str(PDG), "resident",
             str(pipeline(
                 experimental_dir / f"required-{proof_name}.json",
                 proof_output))],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proof_env,
        )
        assert resident_proof.returncode != 0, resident_proof
        assert proof_message in resident_proof.stderr, resident_proof.stderr
        assert not proof_output.exists()

    average_output = experimental_dir / "nndistance-average.las"
    average_oracle = oracle_dir / "nndistance-average.las"
    average = subprocess.run(
        [str(PDG), "pipeline",
         str(nndistance_pipeline(
             experimental_dir / "nndistance-average.json", average_output,
             {"mode": "avg"}))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    average_reference = subprocess.run(
        [str(ORACLE), "pipeline",
         str(nndistance_pipeline(
             oracle_dir / "nndistance-average.json", average_oracle,
             {"mode": "avg"}))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert average.returncode == average_reference.returncode == 0
    assert average.stdout == average_reference.stdout
    assert average.stderr == average_reference.stderr
    assert average_output.read_bytes() == average_oracle.read_bytes()

    # Stats are a separate artifact. Reject both lexical and symlink-equivalent
    # aliases before the pipeline can create or replace its point output.
    alias_output = candidate_dir / "alias.las"
    alias = subprocess.run(
        [str(PDG), "resident",
         str(pipeline(candidate_dir / "alias.json", alias_output)),
         "--stats", f"{candidate_dir}/./alias.las"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert alias.returncode != 0, alias
    assert "stats output aliases pipeline input or output" in alias.stderr
    assert not alias_output.exists()

    alias_parent = root / "candidate-link"
    alias_parent.symlink_to(candidate_dir, target_is_directory=True)
    symlink_output = candidate_dir / "symlink-alias.las"
    symlink_alias = subprocess.run(
        [str(PDG), "resident",
         str(pipeline(candidate_dir / "symlink-alias.json", symlink_output)),
         "--stats", str(alias_parent / "symlink-alias.las")],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert symlink_alias.returncode != 0, symlink_alias
    assert ("stats output aliases pipeline input or output" in
            symlink_alias.stderr)
    assert not symlink_output.exists()

    # Stdout may itself be a PDAL data sink. Keep diagnostics out of that
    # byte stream by requiring a real stats file.
    stdout_output = candidate_dir / "stdout-stats.las"
    stdout_stats = subprocess.run(
        [str(PDG), "resident",
         str(pipeline(candidate_dir / "stdout-stats.json", stdout_output)),
         "--stats", "-"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert stdout_stats.returncode != 0, stdout_stats
    assert "--stats requires a file path" in stdout_stats.stderr
    assert not stdout_output.exists()

    # Reader aliases are equally destructive because stats are written after
    # execution. Exercise the guard on a disposable copy of the fixture.
    alias_input = candidate_dir / "alias-input.las"
    alias_input.write_bytes(INPUT.read_bytes())
    original_input = alias_input.read_bytes()
    input_alias_output = candidate_dir / "input-alias-output.las"
    input_alias = subprocess.run(
        [str(PDG), "resident",
         str(pipeline(candidate_dir / "input-alias.json",
                      input_alias_output, source=alias_input)),
         "--stats", str(alias_input)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert input_alias.returncode != 0, input_alias
    assert ("stats output aliases pipeline input or output" in
            input_alias.stderr)
    assert alias_input.read_bytes() == original_input
    assert not input_alias_output.exists()

    # A configured reader is intentionally outside the header-only runtime
    # cardinality probe. It must fail placement closed before profile lookup,
    # while the untouched host pipeline retains exact reader semantics.
    limited_candidate = candidate_dir / "limited.las"
    limited_oracle = oracle_dir / "limited.las"
    limited_stats = candidate_dir / "limited-stats.json"
    limited = subprocess.run(
        [str(PDG), "resident",
         str(pipeline(candidate_dir / "limited.json", limited_candidate,
                      {"count": 1})),
         "--stats", str(limited_stats)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    limited_reference = subprocess.run(
        [str(ORACLE), "pipeline",
         str(pipeline(oracle_dir / "limited.json", limited_oracle,
                      {"count": 1}))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert limited.returncode == limited_reference.returncode == 0, (
        limited.stderr,
        limited_reference.stderr,
    )
    assert limited_candidate.read_bytes() == limited_oracle.read_bytes()
    limited_report = json.loads(limited_stats.read_text(encoding="utf-8"))
    assert limited_report["placement"]["available"] is False
    assert (limited_report["placement"]["unavailable_reason"] ==
            "invalid_runtime_facts")
    assert limited_report["execution"]["events"] == []

    # The exact extra_dims=all layout is derived from the prepared writer and
    # is planner-visible.  A host-only build can still lack an exact placement
    # profile, while a profiled device build may make a genuine host/device
    # choice; neither case may reject the writer as invalid runtime facts.
    extras_candidate = candidate_dir / "extras.las"
    extras_oracle = oracle_dir / "extras.las"
    extras_stats = candidate_dir / "extras-stats.json"
    extras = subprocess.run(
        [str(PDG), "resident",
         str(pipeline(candidate_dir / "extras.json", extras_candidate,
                      writer_options={"extra_dims": "all"})),
         "--stats", str(extras_stats)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    extras_reference = subprocess.run(
        [str(ORACLE), "pipeline",
         str(pipeline(oracle_dir / "extras.json", extras_oracle,
                      writer_options={"extra_dims": "all"}))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert extras.returncode == extras_reference.returncode == 0, (
        extras.stderr,
        extras_reference.stderr,
    )
    assert extras_candidate.read_bytes() == extras_oracle.read_bytes()
    extras_report = json.loads(extras_stats.read_text(encoding="utf-8"))
    assert extras_report["plan"]["all_stages_native"] is True
    if extras_report["placement"]["available"] is False:
        assert extras_report["placement"]["unavailable_reason"] in {
            "profile_not_exact", "outside_calibration_envelope",
        }, extras_report["placement"]
    assert extras_report["execution"]["events"] == []

    # A .laz extension implies compression, and an explicit true option is
    # byte-equivalent. Both exact extra_dims=all spellings must reach runtime
    # facts; false compression and named layouts remain planner refusals.
    for label, writer_options in (
            ("extras-laz-implicit", {"extra_dims": "all"}),
            ("extras-laz-explicit",
             {"compression": True, "extra_dims": "all"})):
        compressed_candidate = candidate_dir / f"{label}.laz"
        compressed_oracle = oracle_dir / f"{label}.laz"
        compressed_stats = candidate_dir / f"{label}-stats.json"
        compressed = subprocess.run(
            [str(PDG), "resident",
             str(pipeline(candidate_dir / f"{label}.json",
                          compressed_candidate,
                          writer_options=writer_options)),
             "--stats", str(compressed_stats)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        compressed_reference = subprocess.run(
            [str(ORACLE), "pipeline",
             str(pipeline(oracle_dir / f"{label}.json", compressed_oracle,
                          writer_options=writer_options))],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert compressed.returncode == compressed_reference.returncode == 0, (
            compressed.stderr,
            compressed_reference.stderr,
        )
        assert (compressed_candidate.read_bytes() ==
                compressed_oracle.read_bytes())
        compressed_report = json.loads(
            compressed_stats.read_text(encoding="utf-8"))
        assert compressed_report["plan"]["all_stages_native"] is True
        assert (compressed_report["placement"]["unavailable_reason"] !=
                "invalid_runtime_facts")
