#!/usr/bin/env python3
"""Physical SM-89 resident-placement process gates."""

import argparse
import filecmp
import hashlib
import json
import mmap
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile


SKIP = 77
GIB = 1024 * 1024 * 1024
MAX_V4_SOURCE_BYTES = GIB
V5_VALIDATION_BUDGET_BYTES = 256 * 1024 * 1024


def fused_chain():
    return [
        {
            "type": "filters.assign",
            "value": [
                "Scratch = Intensity * 2 - 1",
                ("Classification = 7 WHERE Scratch >= 1000 && "
                 "ReturnNumber >= 1"),
            ],
        },
        {
            "type": "filters.ferry",
            "dimensions": "Classification=>UserData",
        },
        {
            "type": "filters.assign",
            "value": [
                "PointSourceId = Scratch / 2 WHERE Scratch <= 131070",
                ("ReturnNumber = UserData WHERE UserData >= 1 && "
                 "UserData <= 15"),
            ],
        },
    ]


V1_EXPRESSION = "Intensity <= 10000"


def expression_chain():
    """The V1 declared cardinality change inside one fused resident region."""
    return fused_chain() + [
        {"type": "filters.expression", "expression": V1_EXPRESSION},
        {"type": "filters.assign",
         "value": ["UserData = 3 WHERE Classification == 7"]},
    ]


def write_pipeline(path, source, destination, mixed=False, expression=False):
    stages = [{"type": "readers.las", "filename": str(source)}]
    if expression:
        stages.extend(expression_chain())
    else:
        stages.extend(fused_chain())
    if mixed:
        stages.append({"type": "filters.randomize", "seed": 17})
        stages.extend(fused_chain())
    path.write_text(json.dumps(
        {"pipeline": stages +
         [{"type": "writers.las", "filename": str(destination)}]}),
        encoding="utf-8")
    return path


def write_randomize_expression_pipeline(path, source, destination):
    """Keep the generic keep-mask executor covered: the host randomize stage
    blocks endpoint fusion, so the expression region runs through the
    planner-owned PointView boundary executor."""
    stages = [{"type": "readers.las", "filename": str(source)}]
    stages.append({"type": "filters.randomize", "seed": 17})
    stages.extend(expression_chain())
    stages.append({"type": "writers.las", "filename": str(destination)})
    path.write_text(json.dumps({"pipeline": stages}), encoding="utf-8")
    return path


def write_neighborhood_pipeline(path, source, destination):
    """V2: a shared-index neighborhood region whose Coplanar output is
    consumed by a resident ferry so the result is observable through the
    default writer."""
    stages = [{"type": "readers.las", "filename": str(source)},
              {"type": "filters.approximatecoplanar", "knn": 8},
              {"type": "filters.ferry", "dimensions": "Coplanar=>UserData"},
              {"type": "writers.las", "filename": str(destination)}]
    path.write_text(json.dumps({"pipeline": stages}), encoding="utf-8")
    return path


def write_dirty_index_pipeline(path, source, destination):
    """V6: an XYZ-mutating host reprojection between two shared-index
    neighborhood regions forces one physical index build per region."""
    stages = [{"type": "readers.las", "filename": str(source)},
              {"type": "filters.approximatecoplanar", "knn": 8},
              {"type": "filters.ferry", "dimensions": "Coplanar=>UserData"},
              {"type": "filters.reprojection", "in_srs": "EPSG:32615",
               "out_srs": "EPSG:32616", "error_on_failure": True},
              {"type": "filters.lof", "minpts": 10},
              {"type": "filters.assign",
               "value": "UserData = 1 WHERE LocalOutlierFactor >= 1.2"},
              {"type": "writers.las", "filename": str(destination)}]
    path.write_text(json.dumps({"pipeline": stages}), encoding="utf-8")
    return path


def write_lof_pipeline(path, source, destination):
    """V3: a shared-kNN LOF region whose LocalOutlierFactor output is
    consumed by a resident assign bridge so the result is observable through
    the default writer."""
    stages = [{"type": "readers.las", "filename": str(source)},
              {"type": "filters.lof", "minpts": 10},
              {"type": "filters.assign",
               "value": "UserData = 1 WHERE LocalOutlierFactor >= 1.2"},
              {"type": "writers.las", "filename": str(destination)}]
    path.write_text(json.dumps({"pipeline": stages}), encoding="utf-8")
    return path


def write_nndistance_pipeline(path, source, destination):
    """B0045: the exact k=10 distance column feeds one resident UserData
    assignment and the qualified direct default-LAS publisher."""
    stages = [{"type": "readers.las", "filename": str(source)},
              {"type": "filters.nndistance", "k": 10},
              {"type": "filters.assign",
               "value": "UserData = 1 WHERE NNDistance >= 0.4"},
              {"type": "writers.las", "filename": str(destination)}]
    path.write_text(json.dumps({"pipeline": stages}), encoding="utf-8")
    return path


def write_radiusassign_pipeline(path, source, destination):
    """V8: the radius-selection stage performs the exact ordered assignment
    finale on the host, re-uploads UserData, and a resident assign bridge must
    consume that updated device column without rebuilding the shared index."""
    stages = [
        {"type": "readers.las", "filename": str(source)},
        {
            "type": "filters.radiusassign",
            "radius": 2.0,
            "src_domain": "ReturnNumber[1:1]",
            "reference_domain": "ReturnNumber[2:15]",
            "is3d": True,
            "update_expression": "UserData = 9",
        },
        {
            "type": "filters.assign",
            "value": "Classification = 7 WHERE UserData == 9",
        },
        {"type": "writers.las", "filename": str(destination)},
    ]
    path.write_text(json.dumps({"pipeline": stages}), encoding="utf-8")
    return path


def write_where_expression_pipeline(path, source, destination):
    stages = [{"type": "readers.las", "filename": str(source)}]
    stages.extend(fused_chain())
    stages.append({"type": "filters.expression",
                   "expression": V1_EXPRESSION,
                   "where": "ReturnNumber >= 1",
                   "where_merge": "auto"})
    stages.append({"type": "writers.las", "filename": str(destination)})
    path.write_text(json.dumps({"pipeline": stages}), encoding="utf-8")
    return path


def write_unsupported_resident_pipeline(path, source, destination):
    stages = [
        {"type": "readers.las", "filename": str(source)},
        {"type": "filters.approximatecoplanar", "knn": 8},
        {"type": "writers.las", "filename": str(destination)},
    ]
    path.write_text(json.dumps({"pipeline": stages}), encoding="utf-8")
    return path


def run(command, environment=None):
    return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=environment)


def assert_manager_execution_breakdown(execution, expect_nndistance=False):
    phases = execution["pipeline_phase_seconds"]
    assert phases["command_before_stats"] > 0.0, phases
    assert phases["validation_placement_preflight"] > 0.0, phases
    assert phases["rewritten_manager_execution"] > 0.0, phases
    assert phases["canonical_las_publication"] > 0.0, phases
    assert phases["other_control"] >= 0.0, phases
    validation_breakdown = execution[
        "validation_placement_preflight_breakdown_seconds"]
    assert validation_breakdown[
        "matches_validation_placement_preflight"] is True, (
            validation_breakdown)
    assert validation_breakdown["runtime_subphases_match"] is True, (
        validation_breakdown)
    validation_names = (
        "plan_and_original_validation",
        "runtime_placement",
        "rewrite_and_resident_preflight",
    )
    assert all(validation_breakdown[name] >= 0.0
               for name in validation_names), validation_breakdown
    validation_parts = sum(
        validation_breakdown[name] for name in validation_names)
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
    manager_breakdown = execution[
        "rewritten_manager_execution_breakdown_seconds"]
    assert manager_breakdown[
        "matches_rewritten_manager_execution"] is True, manager_breakdown
    for name in (
            "manager_graph_and_prepare",
            "reader_row_table_materialization",
            "resident_wrapper_index_filter",
            "post_spill_stage_control",
            "total"):
        assert manager_breakdown[name] >= 0.0, manager_breakdown
    assert manager_breakdown[
        "reader_row_table_materialization"] > 0.0, manager_breakdown
    assert manager_breakdown[
        "resident_wrapper_index_filter"] > 0.0, manager_breakdown
    manager_parts = sum(
        manager_breakdown[name] for name in (
            "manager_graph_and_prepare",
            "reader_row_table_materialization",
            "resident_wrapper_index_filter",
            "post_spill_stage_control",
        )
    )
    assert abs(manager_breakdown["total"] - manager_parts) < 1e-9, (
        manager_breakdown)
    assert abs(
        manager_breakdown["total"] -
        phases["rewritten_manager_execution"]) < 1e-9, (
            manager_breakdown, phases)

    resident_work = execution["resident_work_breakdown_seconds"]
    assert resident_work[
        "fits_within_resident_wrapper_interval"] is True, resident_work
    for name in (
            "resident_product_setup",
            "direct_las_coordinate_hydration",
            "index_configuration",
            "index_build",
            "neighborhood_query_projection",
            "adjacent_point_program_bridge",
            "profiled_subphases_total",
            "resident_wrapper_unattributed",
            "resident_wrapper_index_filter_total"):
        assert resident_work[name] >= 0.0, resident_work
    for name in (
            "resident_product_setup",
            "index_configuration",
            "index_build",
            "neighborhood_query_projection",
            "adjacent_point_program_bridge"):
        assert resident_work[name] > 0.0, resident_work
    profiled = sum(
        resident_work[name] for name in (
            "resident_product_setup",
            "direct_las_coordinate_hydration",
            "index_configuration",
            "index_build",
            "neighborhood_query_projection",
            "adjacent_point_program_bridge",
        )
    )
    assert abs(
        resident_work["profiled_subphases_total"] - profiled) < 1e-9, (
            resident_work)
    assert abs(
        resident_work["resident_wrapper_index_filter_total"] -
        manager_breakdown["resident_wrapper_index_filter"]) < 1e-9, (
            resident_work, manager_breakdown)
    assert abs(
        resident_work["resident_wrapper_index_filter_total"] -
        resident_work["profiled_subphases_total"] -
        resident_work["resident_wrapper_unattributed"]) < 1e-9, resident_work
    if expect_nndistance:
        assert resident_work["direct_las_coordinate_hydration"] > 0.0, (
            resident_work)
    else:
        assert resident_work["direct_las_coordinate_hydration"] == 0.0, (
            resident_work)

    hydration = execution["direct_las_hydration_breakdown_seconds"]
    assert hydration[
        "fits_within_direct_las_hydration"] is True, hydration
    for name in (
            "validation_and_allocation",
            "transfer_and_kernel_submission",
            "final_stream_wait",
            "profiled_subphases_total",
            "unattributed",
            "total"):
        assert hydration[name] >= 0.0, hydration
    hydration_profiled = sum(
        hydration[name] for name in (
            "validation_and_allocation",
            "transfer_and_kernel_submission",
            "final_stream_wait",
        )
    )
    assert abs(
        hydration["profiled_subphases_total"] - hydration_profiled) < 1e-9, (
            hydration)
    assert abs(
        hydration["total"] -
        resident_work["direct_las_coordinate_hydration"]) < 1e-9, (
            hydration, resident_work)
    assert abs(
        hydration["total"] - hydration["profiled_subphases_total"] -
        hydration["unattributed"]) < 1e-9, hydration
    if expect_nndistance:
        assert hydration["validation_and_allocation"] > 0.0, hydration
        assert hydration["transfer_and_kernel_submission"] > 0.0, hydration
        assert hydration["final_stream_wait"] > 0.0, hydration
    else:
        assert hydration["total"] == 0.0, hydration

    index_config = execution["index_configuration_breakdown_seconds"]
    assert index_config[
        "fits_within_index_configuration"] is True, index_config
    for name in (
            "config_selection",
            "exact_envelope_validation",
            "profiled_subphases_total",
            "unattributed",
            "total"):
        assert index_config[name] >= 0.0, index_config
    index_profiled = sum(
        index_config[name] for name in (
            "config_selection",
            "exact_envelope_validation",
        )
    )
    assert abs(
        index_config["profiled_subphases_total"] - index_profiled) < 1e-9, (
            index_config)
    assert abs(
        index_config["total"] -
        resident_work["index_configuration"]) < 1e-9, (
            index_config, resident_work)
    assert abs(
        index_config["total"] -
        index_config["profiled_subphases_total"] -
        index_config["unattributed"]) < 1e-9, index_config
    assert index_config["config_selection"] > 0.0, index_config
    assert index_config["exact_envelope_validation"] == 0.0, index_config

    nndistance = execution["nndistance_query_breakdown_seconds"]
    if not expect_nndistance:
        assert nndistance is None, nndistance
        return
    assert nndistance["fits_within_broad_query"] is True, nndistance
    detail_names = (
        "output_preparation",
        "query_submission",
        "status_allocation",
        "result_transfer_call",
        "status_transfer_call",
        "explicit_stream_wait",
        "status_scan_and_repair",
        "output_publication",
    )
    for name in detail_names:
        device_only = execution.get(
            "nndistance_device_only_handoff") is True
        host_restore = execution.get("nndistance_host_restore") is True
        if (name == "result_transfer_call" and
                (device_only or host_restore)):
            assert nndistance[name] == 0.0, nndistance
        elif name == "output_publication" and device_only:
            assert nndistance[name] == 0.0, nndistance
        else:
            assert nndistance[name] > 0.0, nndistance
    assert nndistance["profiled_subphases_total"] >= 0.0, nndistance
    assert nndistance["broad_query_unattributed"] >= 0.0, nndistance
    assert nndistance["broad_query_total"] >= 0.0, nndistance
    profiled = sum(nndistance[name] for name in detail_names)
    assert abs(
        nndistance["profiled_subphases_total"] - profiled) < 1e-9, nndistance
    assert abs(
        nndistance["broad_query_total"] -
        resident_work["neighborhood_query_projection"]) < 1e-9, (
            nndistance, resident_work)
    assert abs(
        nndistance["broad_query_total"] -
        nndistance["profiled_subphases_total"] -
        nndistance["broad_query_unattributed"]) < 1e-9, nndistance


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_exact(candidate, oracle, candidate_output, oracle_output):
    assert candidate.returncode == oracle.returncode == 0, (
        candidate.returncode,
        candidate.stderr,
        oracle.returncode,
        oracle.stderr,
    )
    assert candidate.stdout == oracle.stdout, (candidate.stdout, oracle.stdout)
    assert candidate.stderr == oracle.stderr, (candidate.stderr, oracle.stderr)
    assert filecmp.cmp(candidate_output, oracle_output, shallow=False), (
        sha256(candidate_output),
        sha256(oracle_output),
    )


def assert_boundary_accounting(placement, execution):
    """Validate the D4 boundary-accounting contract without fixture math."""
    planned_fields = (
        "point_count",
        "logical_column_bytes",
        "full_record_bytes",
        "predicted_transfer_bytes",
        "predicted_packing_bytes",
        "repack_bytes_per_point",
    )
    observed_fields = ("transfer_bytes", "packing_bytes",
                       "packing_observed")
    planned = placement["planned_boundaries"]
    observed = execution["observed_crossings"]
    assert planned, placement
    assert observed, execution

    planned_by_id = {}
    for boundary in planned:
        for field in planned_fields:
            assert field in boundary, ("planned boundary missing", field,
                                      boundary)
        boundary_id = boundary["id"]
        assert boundary_id not in planned_by_id, planned
        assert isinstance(boundary["point_count"], int)
        assert boundary["point_count"] > 0, boundary
        for field in planned_fields[1:]:
            assert isinstance(boundary[field], int), (field, boundary)
            assert boundary[field] >= 0, (field, boundary)
        assert boundary["full_record_bytes"] > 0, boundary
        assert boundary["full_record_bytes"] % boundary["point_count"] == 0, (
            boundary)
        if boundary["kind"] == "upload":
            assert boundary["predicted_packing_bytes"] == (
                boundary["full_record_bytes"]), boundary
        else:
            assert boundary["predicted_packing_bytes"] == (
                boundary["repack_bytes_per_point"] *
                boundary["point_count"]), boundary
        planned_by_id[boundary_id] = boundary

    observed_by_id = {}
    for boundary in observed:
        for field in observed_fields:
            assert field in boundary, ("observed boundary missing", field,
                                      boundary)
        boundary_id = boundary["boundary_id"]
        assert boundary_id not in observed_by_id, observed
        for field in observed_fields[:2]:
            assert isinstance(boundary[field], int), (field, boundary)
            assert boundary[field] >= 0, (field, boundary)
        assert boundary["packing_observed"] is True, boundary
        observed_by_id[boundary_id] = boundary

    assert list(observed_by_id) == [boundary["id"] for boundary in planned], (
        planned, observed)
    assert set(observed_by_id) == set(planned_by_id), (planned, observed)

    predicted = placement["predicted"]
    planned_h2d = sum(
        boundary["predicted_transfer_bytes"] for boundary in planned
        if boundary["kind"] == "upload")
    planned_d2h = sum(
        boundary["predicted_transfer_bytes"] for boundary in planned
        if boundary["kind"] == "spill")
    planned_packing = sum(
        boundary["predicted_packing_bytes"] for boundary in planned)
    assert predicted["host_to_device_bytes"] == planned_h2d, predicted
    assert predicted["device_to_host_bytes"] == planned_d2h, predicted
    assert predicted["packing_bytes"] == planned_packing, predicted

    totals = execution["totals"]
    observed_h2d = sum(
        boundary["transfer_bytes"] for boundary in observed
        if boundary["kind"] == "upload")
    observed_d2h = sum(
        boundary["transfer_bytes"] for boundary in observed
        if boundary["kind"] == "spill")
    observed_h2d_packing = sum(
        boundary["packing_bytes"] for boundary in observed
        if boundary["kind"] == "upload")
    observed_d2h_packing = sum(
        boundary["packing_bytes"] for boundary in observed
        if boundary["kind"] == "spill")
    assert totals["host_to_device"]["bytes"] == observed_h2d, totals
    assert totals["device_to_host"]["bytes"] == observed_d2h, totals
    assert totals["host_to_device"]["packing_bytes"] == (
        observed_h2d_packing), totals
    assert totals["device_to_host"]["packing_bytes"] == (
        observed_d2h_packing), totals
    for boundary_id, planned_boundary in planned_by_id.items():
        observed_boundary = observed_by_id[boundary_id]
        assert observed_boundary["transfer_bytes"] == (
            planned_boundary["full_record_bytes"]), (
                planned_boundary, observed_boundary)

    predicted_matches_observed = execution[
        "boundary_accounting_matches_prediction"]
    assert isinstance(predicted_matches_observed, bool), execution
    assert predicted_matches_observed is True, execution
    for boundary_id, planned_boundary in planned_by_id.items():
        observed_boundary = observed_by_id[boundary_id]
        assert planned_boundary["predicted_transfer_bytes"] == (
            observed_boundary["transfer_bytes"]), (
                planned_boundary, observed_boundary)
        assert planned_boundary["predicted_packing_bytes"] == (
            observed_boundary["packing_bytes"]), (
                planned_boundary, observed_boundary)


def run_case(pdg, oracle, source, root, mixed, candidate_environment=None,
             expression=False):
    candidate_output = root / "candidate.las"
    oracle_output = root / "oracle.las"
    stats_path = root / "stats.json"
    candidate_pipeline = write_pipeline(
        root / "candidate.json", source, candidate_output, mixed, expression)
    oracle_pipeline = write_pipeline(
        root / "oracle.json", source, oracle_output, mixed, expression)
    candidate = run([str(pdg), "resident", str(candidate_pipeline),
                     "--stats", str(stats_path)], candidate_environment)
    assert candidate.returncode == 0, candidate.stderr
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return (candidate, stats, candidate_output, oracle_output,
            oracle_pipeline)


def exact_profile_or_skip(pdg, oracle, small_input, root):
    candidate, stats, candidate_output, oracle_output, oracle_pipeline = (
        run_case(pdg, oracle, small_input, root, False))
    placement = stats["placement"]
    if not placement["available"]:
        assert placement["unavailable_reason"] == "profile_not_exact", stats
        print("exact resident placement profile is unavailable", file=sys.stderr)
        raise SystemExit(SKIP)
    reference = run([str(oracle), "pipeline", str(oracle_pipeline)])
    assert_exact(candidate, reference, candidate_output, oracle_output)
    assert placement["choice"] == "host", placement
    assert placement["selected_region_count"] == 0, placement
    assert placement["boundary_accounting_model"] == "executor_declared", (
        placement)
    execution = stats["execution"]
    assert execution["executor"] == "pdal_standard_host", execution
    assert execution["selected_device_calibration_matches_executor"] is None
    assert execution["selected_regions"] == [], execution
    assert execution["observed_crossings"] == [], execution
    assert execution["events"] == [], execution


def las_point_count(path):
    with path.open("rb") as stream:
        header = stream.read(375)
    assert len(header) >= 227 and header[:4] == b"LASF", path
    major, minor = header[24], header[25]
    assert major == 1 and minor <= 4, (major, minor)
    if minor >= 4:
        assert len(header) >= 375
        return struct.unpack_from("<Q", header, 247)[0]
    return struct.unpack_from("<I", header, 107)[0]


def las_record_length(path):
    with path.open("rb") as stream:
        header = stream.read(107)
    assert len(header) == 107 and header[:4] == b"LASF", path
    return struct.unpack_from("<H", header, 105)[0]


def las_user_data_differs(left, right):
    def layout(path):
        with path.open("rb") as stream:
            header = stream.read(375)
        assert len(header) >= 227 and header[:4] == b"LASF", path
        count = (struct.unpack_from("<Q", header, 247)[0]
                 if header[25] >= 4
                 else struct.unpack_from("<I", header, 107)[0])
        return (struct.unpack_from("<I", header, 96)[0],
                struct.unpack_from("<H", header, 105)[0], count)

    left_offset, left_stride, left_count = layout(left)
    right_offset, right_stride, right_count = layout(right)
    assert left_count == right_count, (left_count, right_count)
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        with mmap.mmap(left_stream.fileno(), 0, access=mmap.ACCESS_READ) as a:
            with mmap.mmap(right_stream.fileno(), 0,
                           access=mmap.ACCESS_READ) as b:
                return any(
                    a[left_offset + point * left_stride + 17] !=
                    b[right_offset + point * right_stride + 17]
                    for point in range(left_count)
                )


def linux_memory_bytes():
    values = {}
    for line in pathlib.Path("/proc/meminfo").read_text(
            encoding="utf-8").splitlines():
        name, value = line.split(":", 1)
        values[name] = int(value.strip().split()[0]) * 1024
    return values


def preflight_large_case(source, destination_root):
    memory = linux_memory_bytes()
    if memory["MemAvailable"] < 8 * GIB:
        print(f"V4 skipped: MemAvailable={memory['MemAvailable']}",
              file=sys.stderr)
        raise SystemExit(SKIP)
    required_disk = 2 * source.stat().st_size + GIB
    free_disk = shutil.disk_usage(destination_root).free
    if free_disk < required_disk:
        print(f"V4 skipped: free_disk={free_disk}, required={required_disk}",
              file=sys.stderr)
        raise SystemExit(SKIP)
    query = run(["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"])
    if query.returncode != 0:
        print("V4 skipped: NVIDIA memory query failed", file=sys.stderr)
        raise SystemExit(SKIP)
    free_vram_mib = int(query.stdout.splitlines()[0].strip())
    if free_vram_mib < 2048:
        print(f"V4 skipped: free_vram_mib={free_vram_mib}", file=sys.stderr)
        raise SystemExit(SKIP)
    return memory["SwapTotal"] - memory["SwapFree"]


def fixed_direct_scheduler_environment():
    """Keep the D0053 two-lane default independent of caller test knobs."""
    environment = os.environ.copy()
    environment.pop("PDG_CUDA_SCHEDULER_LANES", None)
    environment.pop("PDG_CUDA_CHUNK_POINTS", None)
    return environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",
                        choices=("direct", "v1", "v2", "v3", "v4", "v5",
                                 "v6", "v7", "v8"))
    parser.add_argument("pdg", type=pathlib.Path)
    parser.add_argument("oracle", type=pathlib.Path)
    parser.add_argument("small_input", type=pathlib.Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
            prefix=f"pdg-resident-{args.mode}-") as temporary:
        root = pathlib.Path(temporary)
        probe = root / "profile-probe"
        probe.mkdir()
        exact_profile_or_skip(
            args.pdg, args.oracle, args.small_input.resolve(), probe)
        if args.mode == "v7":
            unsupported = root / "unsupported-executor"
            unsupported.mkdir()
            candidate_output = unsupported / "candidate.las"
            oracle_output = unsupported / "oracle.las"
            stats_path = unsupported / "stats.json"
            candidate_pipeline = write_unsupported_resident_pipeline(
                unsupported / "candidate.json", args.small_input.resolve(),
                candidate_output)
            oracle_pipeline = write_unsupported_resident_pipeline(
                unsupported / "oracle.json", args.small_input.resolve(),
                oracle_output)
            candidate = run([
                str(args.pdg), "resident", str(candidate_pipeline),
                "--stats", str(stats_path),
            ])
            reference = run([
                str(args.oracle), "pipeline", str(oracle_pipeline),
            ])
            assert_exact(candidate, reference, candidate_output, oracle_output)
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            placement = stats["placement"]
            assert placement["available"] is True, placement
            assert placement["choice"] == "host", placement
            assert placement["boundary_accounting_model"] == (
                "calibration_default"), placement
            assert placement["selected_region_count"] == 0, placement
            execution = stats["execution"]
            assert execution["executor"] == "pdal_standard_host", execution
            assert execution["selected_regions"] == [], execution
            return 0

        configured = os.environ.get("PDG_LOCAL_LAS_FILE")
        if not configured:
            print("set PDG_LOCAL_LAS_FILE for the physical resident gate",
                  file=sys.stderr)
            return SKIP
        large_input = pathlib.Path(configured).resolve()
        if not large_input.is_file() or large_input.suffix.lower() != ".las":
            raise AssertionError("PDG_LOCAL_LAS_FILE must name an uncompressed LAS")
        if large_input.stat().st_size > MAX_V4_SOURCE_BYTES:
            print(
                f"{args.mode} skipped: source_bytes={large_input.stat().st_size}, "
                f"maximum={MAX_V4_SOURCE_BYTES}",
                file=sys.stderr,
            )
            return SKIP
        assert las_point_count(large_input) >= 16_000_000, large_input
        expected_hash = os.environ.get("PDG_LOCAL_LAS_SHA256")
        if not expected_hash:
            print("set PDG_LOCAL_LAS_SHA256 for reproducible resident evidence",
                  file=sys.stderr)
            return SKIP
        actual_hash = sha256(large_input)
        assert actual_hash == expected_hash.lower(), (
            large_input, actual_hash, expected_hash)
        swap_used_before = preflight_large_case(large_input, root)

        if args.mode == "v5":
            tiled = root / "tiled"
            tiled.mkdir()
            environment = os.environ.copy()
            environment.pop("PDG_CUDA_SCHEDULER_LANES", None)
            environment.pop("PDG_CUDA_CHUNK_POINTS", None)
            environment["PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES"] = str(
                V5_VALIDATION_BUDGET_BYTES)
            candidate, stats, candidate_output, oracle_output, oracle_pipeline = (
                run_case(args.pdg, args.oracle, large_input, tiled, True,
                         environment))
            placement = stats["placement"]
            assert placement["available"] is True, placement
            assert placement["choice"] == "device", placement
            assert placement["boundary_accounting_model"] == (
                "executor_declared"), placement
            assert placement["validation_budget_override_bytes"] == (
                V5_VALIDATION_BUDGET_BYTES), placement
            assert placement["selected_region_count"] == 2, placement
            predicted = placement["predicted"]
            assert predicted["untiled_device_bytes"] > (
                V5_VALIDATION_BUDGET_BYTES), predicted
            assert predicted["peak_device_bytes"] <= (
                V5_VALIDATION_BUDGET_BYTES), predicted

            execution = stats["execution"]
            assert execution["executor"] == (
                "planner_resident_boundary_batch"), execution
            assert (
                execution["selected_device_calibration_matches_executor"]
                is False
            ), execution
            assert execution["rewrite_executable"] is True, execution
            assert execution["resident_preflight"]["attempted"] is True
            assert execution["resident_preflight"]["accepted"] is True
            assert execution["selected_regions"] == [0, 1], execution
            assert_boundary_accounting(placement, execution)
            schedule = execution["schedule"]
            point_count = las_point_count(large_input)
            expected_tiles = (
                point_count + schedule["tile_item_capacity"] - 1
            ) // schedule["tile_item_capacity"]
            assert schedule["item_count"] == point_count, schedule
            assert schedule["tile_item_capacity"] == 131_072, schedule
            assert schedule["tile_count"] == expected_tiles == 168, schedule
            assert schedule["configured_lane_count"] == 2, schedule
            assert schedule["active_lane_count"] == 2, schedule
            assert schedule["lane_reuse_count"] == 166, schedule
            assert schedule["memory_limited"] is False, schedule
            assert schedule["memory_budget_bytes"] == (
                V5_VALIDATION_BUDGET_BYTES), schedule
            assert 0 < schedule["observed_peak_lane_bytes"] <= schedule[
                "peak_lane_bytes"], schedule
            assert schedule["peak_lane_bytes"] == predicted[
                "peak_device_bytes"], (schedule, predicted)
            assert predicted["configured_device_lane_count"] == schedule[
                "configured_lane_count"], (schedule, predicted)
            assert predicted["active_device_lane_count"] == schedule[
                "active_lane_count"], (schedule, predicted)

            reference = run([
                str(args.oracle), "pipeline", str(oracle_pipeline),
            ])
            assert_exact(candidate, reference, candidate_output, oracle_output)

            protected = tiled / "invalid-budget.las"
            protected.write_bytes(b"pdg v5 invalid-budget sentinel\n")
            protected_digest = sha256(protected)
            invalid_pipeline = write_pipeline(
                tiled / "invalid-budget.json", large_input, protected, True)
            invalid_environment = os.environ.copy()
            invalid_environment.pop("PDG_CUDA_SCHEDULER_LANES", None)
            invalid_environment.pop("PDG_CUDA_CHUNK_POINTS", None)
            invalid_environment["PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES"] = str(
                (1 << 64) - 1)
            rejected = run([
                str(args.pdg), "resident", str(invalid_pipeline),
            ], invalid_environment)
            assert rejected.returncode != 0, rejected
            assert "PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES" in rejected.stderr, (
                rejected)
            assert sha256(protected) == protected_digest

            print(json.dumps({
                "v5": {
                    "point_count": point_count,
                    "tile_count": schedule["tile_count"],
                    "tile_item_capacity": schedule["tile_item_capacity"],
                    "configured_lane_count": schedule[
                        "configured_lane_count"],
                    "active_lane_count": schedule["active_lane_count"],
                    "lane_reuse_count": schedule["lane_reuse_count"],
                    "memory_budget_bytes": schedule["memory_budget_bytes"],
                    "peak_lane_bytes": schedule["peak_lane_bytes"],
                    "observed_peak_lane_bytes": schedule[
                        "observed_peak_lane_bytes"],
                    "untiled_device_bytes": predicted[
                        "untiled_device_bytes"],
                    "output_sha256": sha256(candidate_output),
                }
            }, sort_keys=True))

            memory_after = linux_memory_bytes()
            swap_used_after = (
                memory_after["SwapTotal"] - memory_after["SwapFree"])
            assert swap_used_after <= swap_used_before + 256 * 1024 * 1024, (
                swap_used_before, swap_used_after)
            return 0

        if args.mode == "v6":
            resident = root / "dirty-index"
            resident.mkdir()
            environment = fixed_direct_scheduler_environment()
            candidate_output = resident / "candidate.las"
            oracle_output = resident / "oracle.las"
            stats_path = resident / "stats.json"
            candidate_pipeline = write_dirty_index_pipeline(
                resident / "candidate.json", large_input, candidate_output)
            oracle_pipeline = write_dirty_index_pipeline(
                resident / "oracle.json", large_input, oracle_output)
            candidate = run([str(args.pdg), "resident",
                             str(candidate_pipeline), "--stats",
                             str(stats_path)], environment)
            assert candidate.returncode == 0, candidate.stderr
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            placement = stats["placement"]
            assert placement["available"] is True, placement
            assert placement["choice"] == "device", placement
            assert placement["selected_region_count"] == 2, placement
            assert placement["predicted"]["index_build_count"] == 2, placement

            execution = stats["execution"]
            assert execution["executor"] == (
                "planner_resident_shared_index"), execution
            assert execution["direct_las_output"] is False, execution
            assert execution["rewrite_executable"] is True, execution
            assert execution["resident_preflight"]["accepted"] is True, (
                execution)
            assert execution["selected_regions"] == [0, 1], execution
            schedule = execution["schedule"]
            assert schedule["pipeline_class"] == (
                "whole_view_neighborhood"), schedule
            # E4: the observed physical index builds equal the planner's
            # prediction and both appear in --stats.
            builds = execution["index_builds"]
            assert builds["predicted"] == 2, builds
            assert builds["observed"] == 2, builds
            assert builds["matches_prediction"] is True, builds
            kinds = [event["kind"] for event in execution["events"]]
            assert kinds.count("index_build") == 2, execution["events"]
            assert kinds.count("device_region_begin") == 2, kinds
            assert kinds.count("device_region_end") == 2, kinds

            reference = run([
                str(args.oracle), "pipeline", str(oracle_pipeline),
            ])
            assert_exact(candidate, reference, candidate_output, oracle_output)

            print(json.dumps({
                "v6": {
                    "point_count": las_point_count(large_input),
                    "executor": execution["executor"],
                    "index_builds": builds,
                    "output_sha256": sha256(candidate_output),
                }
            }, sort_keys=True))

            memory_after = linux_memory_bytes()
            swap_used_after = (
                memory_after["SwapTotal"] - memory_after["SwapFree"])
            assert swap_used_after <= swap_used_before + 256 * 1024 * 1024, (
                swap_used_before, swap_used_after)
            return 0

        if args.mode == "v8":
            resident = root / "radiusassign"
            resident.mkdir()
            environment = fixed_direct_scheduler_environment()
            environment["PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE"] = "1"
            candidate_output = resident / "candidate.las"
            oracle_output = resident / "oracle.las"
            stats_path = resident / "stats.json"
            candidate_pipeline = write_radiusassign_pipeline(
                resident / "candidate.json", large_input, candidate_output)
            oracle_pipeline = write_radiusassign_pipeline(
                resident / "oracle.json", large_input, oracle_output)
            candidate = run([
                str(args.pdg), "resident", str(candidate_pipeline),
                "--stats", str(stats_path),
            ], environment)
            assert candidate.returncode == 0, candidate.stderr
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            placement = stats["placement"]
            assert placement["available"] is True, placement
            assert placement["choice"] == "device", placement
            assert placement["selected_region_count"] == 1, placement

            execution = stats["execution"]
            assert execution["executor"] == (
                "planner_resident_shared_index"), execution
            assert execution[
                "selected_device_calibration_matches_executor"] is True, (
                    execution)
            assert execution["rewrite_executable"] is True, execution
            assert execution["resident_preflight"]["accepted"] is True, (
                execution)
            assert execution["selected_regions"] == [0], execution
            builds = execution["index_builds"]
            assert builds["predicted"] == 1, builds
            assert builds["observed"] == 1, builds
            assert builds["matches_prediction"] is True, builds
            kinds = [event["kind"] for event in execution["events"]]
            assert kinds.count("index_build") == 1, kinds
            assert kinds.count("device_region_begin") == 1, kinds
            assert kinds.count("device_region_end") == 1, kinds

            reference = run([
                str(args.oracle), "pipeline", str(oracle_pipeline),
            ])
            assert_exact(candidate, reference, candidate_output, oracle_output)
            print(json.dumps({
                "v8": {
                    "point_count": las_point_count(large_input),
                    "executor": execution["executor"],
                    "index_builds": builds,
                    "output_sha256": sha256(candidate_output),
                }
            }, sort_keys=True))
            return 0

        if args.mode == "v3":
            resident = root / "lof"
            resident.mkdir()
            environment = fixed_direct_scheduler_environment()
            candidate_output = resident / "candidate.las"
            oracle_output = resident / "oracle.las"
            stats_path = resident / "stats.json"
            candidate_pipeline = write_lof_pipeline(
                resident / "candidate.json", large_input, candidate_output)
            oracle_pipeline = write_lof_pipeline(
                resident / "oracle.json", large_input, oracle_output)
            candidate = run([str(args.pdg), "resident",
                             str(candidate_pipeline), "--stats",
                             str(stats_path)], environment)
            assert candidate.returncode == 0, candidate.stderr
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            placement = stats["placement"]
            assert placement["available"] is True, placement
            assert placement["choice"] == "device", placement
            assert placement["boundary_accounting_model"] == (
                "calibration_default"), placement
            assert placement["selected_region_count"] == 1, placement

            execution = stats["execution"]
            assert execution["executor"] == (
                "planner_resident_shared_index"), execution
            # D0077: the shared-index models carry resident-executor
            # provenance, so the calibration-match flag holds true.
            assert (
                execution["selected_device_calibration_matches_executor"]
                is True
            ), execution
            assert execution["rewrite_executable"] is True, execution
            assert execution["resident_preflight"]["accepted"] is True, (
                execution)
            assert execution["selected_regions"] == [0], execution
            point_count = las_point_count(large_input)
            schedule = execution["schedule"]
            assert schedule["pipeline_class"] == (
                "whole_view_neighborhood"), schedule
            assert schedule["item_count"] == point_count, schedule
            assert schedule["tile_count"] == 1, schedule
            assert schedule["configured_lane_count"] == 1, schedule
            kinds = [event["kind"] for event in execution["events"]]
            assert "device_region_begin" in kinds, execution["events"]
            assert "device_region_end" in kinds, execution["events"]
            exact_repair = execution["exact_host_repair"]
            device_repair = execution["exact_device_repair"]
            assert exact_repair["seconds"] >= 0.0, exact_repair
            assert device_repair["seconds"] >= 0.0, device_repair
            assert 0 <= device_repair["incomplete_rows"] <= point_count, (
                device_repair)
            assert 0 <= device_repair["repaired_rows"] <= point_count, (
                device_repair)
            assert 0 <= exact_repair["ambiguous_rows"] <= point_count, (
                exact_repair)
            assert 0 <= exact_repair["incomplete_rows"] <= point_count, (
                exact_repair)
            assert 0 <= exact_repair["repaired_rows"] <= point_count, (
                exact_repair)
            assert exact_repair["repaired_rows"] >= exact_repair[
                "ambiguous_rows"], exact_repair
            assert exact_repair["repaired_rows"] >= exact_repair[
                "incomplete_rows"], exact_repair

            reference = run([
                str(args.oracle), "pipeline", str(oracle_pipeline),
            ])
            assert_exact(candidate, reference, candidate_output, oracle_output)

            direct_output = resident / "candidate-direct.las"
            direct_stats_path = resident / "stats-direct.json"
            direct_pipeline = write_lof_pipeline(
                resident / "candidate-direct.json", large_input,
                direct_output)
            direct_environment = environment.copy()
            direct_environment[
                "PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT"] = "1"
            direct = run([
                str(args.pdg), "resident", str(direct_pipeline), "--stats",
                str(direct_stats_path),
            ], direct_environment)
            assert_exact(direct, reference, direct_output, oracle_output)
            direct_execution = json.loads(
                direct_stats_path.read_text(encoding="utf-8"))["execution"]
            assert direct_execution["executor"] == (
                "planner_resident_shared_index_direct_las"), direct_execution
            assert direct_execution["direct_las_output"] is True, (
                direct_execution)
            assert direct_execution[
                "selected_device_calibration_matches_executor"] is False, (
                    direct_execution)
            assert direct_execution["selected_regions"] == [0], (
                direct_execution)
            assert direct_execution["index_builds"]["observed"] == 1, (
                direct_execution)
            assert_manager_execution_breakdown(direct_execution)

            automatic_output = resident / "candidate-automatic.las"
            automatic_pipeline = write_lof_pipeline(
                resident / "candidate-automatic.json", large_input,
                automatic_output)
            automatic_environment = environment.copy()
            automatic_environment[
                "PDG_REQUIRE_AUTOMATIC_RESIDENT_LAS_OUTPUT"] = "1"
            automatic = run([
                str(args.pdg), "pipeline", str(automatic_pipeline),
            ], automatic_environment)
            assert_exact(automatic, reference, automatic_output,
                         oracle_output)

            automatic_digest = sha256(automatic_output)
            automatic_existing = run([
                str(args.pdg), "pipeline", str(automatic_pipeline),
            ], automatic_environment)
            assert automatic_existing.returncode != 0, automatic_existing
            assert "required automatic resident LAS output path was not used" in (
                automatic_existing.stderr), automatic_existing.stderr
            assert sha256(automatic_output) == automatic_digest
            assert not list(resident.glob(
                "candidate-automatic.las.pdg-resident-output-*")), resident

            preflight_output = resident / "candidate-preflight-refusal.las"
            preflight_pipeline = write_lof_pipeline(
                resident / "candidate-preflight-refusal.json", large_input,
                preflight_output)
            preflight_environment = automatic_environment.copy()
            preflight_environment[
                "PDG_TEST_AUTOMATIC_RESIDENT_PREFLIGHT_FAILURE"] = "1"
            preflight_refusal = run([
                str(args.pdg), "pipeline", str(preflight_pipeline),
            ], preflight_environment)
            assert preflight_refusal.returncode != 0, preflight_refusal
            assert "required automatic resident LAS output path was not used" in (
                preflight_refusal.stderr), preflight_refusal.stderr
            assert not preflight_output.exists()

            # Admission and atomic publication fail before data execution.
            # An existing output remains untouched and no per-process
            # temporary is left behind.
            direct_digest = sha256(direct_output)
            existing = run([
                str(args.pdg), "resident", str(direct_pipeline),
            ], direct_environment)
            assert existing.returncode != 0, existing
            assert "required direct resident LAS output path was not used" in (
                existing.stderr), existing.stderr
            assert sha256(direct_output) == direct_digest
            assert not list(resident.glob(
                "candidate-direct.las.pdg-resident-output-*")), resident

            serialized_output = resident / "candidate-serialized-write.las"
            serialized_pipeline = resident / "candidate-serialized-write.json"
            serialized_pipeline.write_text(json.dumps({"pipeline": [
                {"type": "readers.las", "filename": str(large_input)},
                {"type": "filters.lof", "minpts": 10},
                {"type": "filters.assign", "value": [
                    ("UserData = 1 WHERE LocalOutlierFactor >= 1.2"),
                    "Classification = 7",
                ]},
                {"type": "writers.las", "filename": str(serialized_output)},
            ]}), encoding="utf-8")
            serialized = run([
                str(args.pdg), "resident", str(serialized_pipeline),
            ], direct_environment)
            assert serialized.returncode != 0, serialized
            assert "required direct resident LAS output path was not used" in (
                serialized.stderr), serialized.stderr
            assert not serialized_output.exists()

            option_output = resident / "candidate-writer-option.las"
            option_pipeline = resident / "candidate-writer-option.json"
            option_pipeline.write_text(json.dumps({"pipeline": [
                {"type": "readers.las", "filename": str(large_input)},
                {"type": "filters.lof", "minpts": 10},
                {"type": "filters.assign",
                 "value": ("UserData = 1 WHERE "
                           "LocalOutlierFactor >= 1.2")},
                {"type": "writers.las", "filename": str(option_output),
                 "extra_dims": "all"},
            ]}), encoding="utf-8")
            option = run([
                str(args.pdg), "resident", str(option_pipeline),
            ], direct_environment)
            assert option.returncode != 0, option
            assert "required direct resident LAS output path was not used" in (
                option.stderr), option.stderr
            assert not option_output.exists()

            print(json.dumps({
                "v3": {
                    "point_count": point_count,
                    "executor": execution["executor"],
                    "pipeline_class": schedule["pipeline_class"],
                    "peak_lane_bytes": schedule["peak_lane_bytes"],
                    "memory_budget_bytes": schedule["memory_budget_bytes"],
                    "output_sha256": sha256(candidate_output),
                    "direct_output_sha256": sha256(direct_output),
                    "automatic_output_sha256": sha256(automatic_output),
                }
            }, sort_keys=True))

            memory_after = linux_memory_bytes()
            swap_used_after = (
                memory_after["SwapTotal"] - memory_after["SwapFree"])
            assert swap_used_after <= swap_used_before + 256 * 1024 * 1024, (
                swap_used_before, swap_used_after)
            return 0

        if args.mode == "v2":
            resident = root / "neighborhood"
            resident.mkdir()
            environment = fixed_direct_scheduler_environment()
            candidate_output = resident / "candidate.las"
            oracle_output = resident / "oracle.las"
            stats_path = resident / "stats.json"
            candidate_pipeline = write_neighborhood_pipeline(
                resident / "candidate.json", large_input, candidate_output)
            oracle_pipeline = write_neighborhood_pipeline(
                resident / "oracle.json", large_input, oracle_output)
            candidate = run([str(args.pdg), "resident",
                             str(candidate_pipeline), "--stats",
                             str(stats_path)], environment)
            assert candidate.returncode == 0, candidate.stderr
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            placement = stats["placement"]
            assert placement["available"] is True, placement
            assert placement["choice"] == "device", placement
            # The shared-index executor keeps calibration-default accounting:
            # the PointView row facts describe the tile executor, not the
            # whole-view attach machinery.
            assert placement["boundary_accounting_model"] == (
                "calibration_default"), placement
            assert placement["selected_region_count"] == 1, placement

            execution = stats["execution"]
            assert execution["executor"] == (
                "planner_resident_shared_index"), execution
            # D0077: the shared-index models carry resident-executor
            # provenance, so the calibration-match flag holds true.
            assert (
                execution["selected_device_calibration_matches_executor"]
                is True
            ), execution
            assert execution["rewrite_executable"] is True, execution
            assert execution["resident_preflight"]["accepted"] is True, (
                execution)
            assert execution["selected_regions"] == [0], execution
            point_count = las_point_count(large_input)
            schedule = execution["schedule"]
            assert schedule["pipeline_class"] == (
                "whole_view_neighborhood"), schedule
            assert schedule["item_count"] == point_count, schedule
            assert schedule["tile_count"] == 1, schedule
            assert schedule["tile_item_capacity"] == point_count, schedule
            assert schedule["configured_lane_count"] == 1, schedule
            assert schedule["active_lane_count"] == 1, schedule
            kinds = [event["kind"] for event in execution["events"]]
            assert "device_region_begin" in kinds, execution["events"]
            assert "device_region_end" in kinds, execution["events"]

            reference = run([
                str(args.oracle), "pipeline", str(oracle_pipeline),
            ])
            assert_exact(candidate, reference, candidate_output, oracle_output)

            nndistance_output = resident / "nndistance-automatic.las"
            nndistance_oracle_output = resident / "nndistance-oracle.las"
            nndistance_pipeline = write_nndistance_pipeline(
                resident / "nndistance-automatic.json", large_input,
                nndistance_output)
            nndistance_oracle_pipeline = write_nndistance_pipeline(
                resident / "nndistance-oracle.json", large_input,
                nndistance_oracle_output)
            automatic_environment = environment.copy()
            automatic_environment[
                "PDG_REQUIRE_AUTOMATIC_RESIDENT_LAS_OUTPUT"] = "1"
            automatic_environment[
                "PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
            nndistance = run([
                str(args.pdg), "pipeline", str(nndistance_pipeline),
            ], automatic_environment)
            nndistance_reference = run([
                str(args.oracle), "pipeline",
                str(nndistance_oracle_pipeline),
            ])
            assert_exact(nndistance, nndistance_reference,
                         nndistance_output, nndistance_oracle_output)
            assert las_user_data_differs(large_input, nndistance_output), (
                "nndistance threshold fixture changed no UserData values")

            nndistance_source_disabled_output = resident / (
                "nndistance-source-disabled.las")
            nndistance_source_disabled_pipeline = write_nndistance_pipeline(
                resident / "nndistance-source-disabled.json", large_input,
                nndistance_source_disabled_output)
            source_disabled_environment = environment.copy()
            source_disabled_environment[
                "PDG_REQUIRE_AUTOMATIC_RESIDENT_LAS_OUTPUT"] = "1"
            source_disabled_environment[
                "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
            nndistance_source_disabled = run([
                str(args.pdg), "pipeline",
                str(nndistance_source_disabled_pipeline),
            ], source_disabled_environment)
            assert_exact(nndistance_source_disabled, nndistance_reference,
                         nndistance_source_disabled_output,
                         nndistance_oracle_output)

            nndistance_direct_output = resident / "nndistance-direct.las"
            nndistance_direct_stats_path = resident / (
                "nndistance-direct-stats.json")
            nndistance_direct_pipeline = write_nndistance_pipeline(
                resident / "nndistance-direct.json", large_input,
                nndistance_direct_output)
            direct_environment = environment.copy()
            direct_environment[
                "PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT"] = "1"
            direct_environment[
                "PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
            direct_environment[
                "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY"] = "1"
            direct_environment[
                "PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ"] = "1"
            direct_environment["PDG_REQUIRE_NND_DEVICE_REPAIR"] = "1"
            direct_environment[
                "PDG_REQUIRE_NND_PARALLEL_REPAIR"] = "1"
            direct_environment[
                "PDG_REQUIRE_NND_DEVICE_ONLY_HANDOFF"] = "1"
            nndistance_direct = run([
                str(args.pdg), "resident", str(nndistance_direct_pipeline),
                "--stats", str(nndistance_direct_stats_path),
            ], direct_environment)
            assert_exact(nndistance_direct, nndistance_reference,
                         nndistance_direct_output,
                         nndistance_oracle_output)
            nndistance_direct_execution = json.loads(
                nndistance_direct_stats_path.read_text(
                    encoding="utf-8"))["execution"]
            assert nndistance_direct_execution["executor"] == (
                "planner_resident_shared_index_direct_las"), (
                    nndistance_direct_execution)
            assert nndistance_direct_execution["direct_las_output"] is True, (
                nndistance_direct_execution)
            assert nndistance_direct_execution[
                "direct_las_resident_source"] is True, (
                    nndistance_direct_execution)
            assert nndistance_direct_execution[
                "direct_las_record_summary"] is True, (
                    nndistance_direct_execution)
            assert nndistance_direct_execution[
                "direct_las_host_xyz_mirror"] is False, (
                    nndistance_direct_execution)
            assert nndistance_direct_execution[
                "nndistance_device_only_handoff"] is True, (
                    nndistance_direct_execution)
            assert nndistance_direct_execution[
                "nndistance_host_restore"] is False, (
                    nndistance_direct_execution)
            assert nndistance_direct_execution[
                "nndistance_assignment_device_column_reuse"] is True, (
                    nndistance_direct_execution)
            assert not any(
                event["kind"] in ("host_to_device", "device_to_host") and
                event["bytes"] == point_count * 8
                for event in nndistance_direct_execution["events"]), (
                    nndistance_direct_execution["events"])
            assert nndistance_direct_execution["selected_regions"] == [0], (
                nndistance_direct_execution)
            assert nndistance_direct_execution[
                "index_builds"]["observed"] == 1, (
                    nndistance_direct_execution)
            assert_manager_execution_breakdown(
                nndistance_direct_execution, expect_nndistance=True)
            nndistance_repair = nndistance_direct_execution[
                "exact_host_repair"]
            nndistance_device_repair = nndistance_direct_execution[
                "exact_device_repair"]
            assert nndistance_repair["ambiguous_rows"] == 0, (
                nndistance_repair)
            assert nndistance_repair["incomplete_rows"] == 0, (
                nndistance_repair)
            assert nndistance_repair["repaired_rows"] == 0, (
                nndistance_repair)
            assert nndistance_repair["seconds"] == 0.0, nndistance_repair
            assert nndistance_device_repair["repaired_rows"] == (
                nndistance_device_repair["incomplete_rows"]), (
                    nndistance_device_repair)
            if nndistance_device_repair["repaired_rows"]:
                assert nndistance_repair["seconds"] == 0.0, (
                    nndistance_repair)
                assert nndistance_device_repair["seconds"] > 0.0, (
                    nndistance_device_repair)

            nndistance_host_repair_output = resident / (
                "nndistance-direct-host-repair.las")
            nndistance_host_repair_stats_path = resident / (
                "nndistance-direct-host-repair-stats.json")
            nndistance_host_repair_pipeline = write_nndistance_pipeline(
                resident / "nndistance-direct-host-repair.json",
                large_input, nndistance_host_repair_output)
            host_repair_environment = direct_environment.copy()
            host_repair_environment.pop("PDG_REQUIRE_NND_DEVICE_REPAIR")
            host_repair_environment.pop(
                "PDG_REQUIRE_NND_PARALLEL_REPAIR")
            host_repair_environment.pop(
                "PDG_REQUIRE_NND_DEVICE_ONLY_HANDOFF")
            host_repair_environment["PDG_DISABLE_NND_DEVICE_REPAIR"] = "1"
            host_repair_environment["PDG_REQUIRE_NND_HOST_RESTORE"] = "1"
            nndistance_host_repair = run([
                str(args.pdg), "resident",
                str(nndistance_host_repair_pipeline), "--stats",
                str(nndistance_host_repair_stats_path),
            ], host_repair_environment)
            assert_exact(nndistance_host_repair, nndistance_reference,
                         nndistance_host_repair_output,
                         nndistance_oracle_output)
            host_repair_execution = json.loads(
                nndistance_host_repair_stats_path.read_text(
                    encoding="utf-8"))["execution"]
            assert host_repair_execution[
                "direct_las_record_summary"] is True, host_repair_execution
            assert host_repair_execution[
                "direct_las_host_xyz_mirror"] is False, host_repair_execution
            assert host_repair_execution[
                "nndistance_device_only_handoff"] is False, (
                    host_repair_execution)
            assert host_repair_execution[
                "nndistance_host_restore"] is True, host_repair_execution
            assert host_repair_execution[
                "nndistance_assignment_device_column_reuse"] is True, (
                    host_repair_execution)
            full_nndistance_transfers = [
                event["kind"] for event in host_repair_execution["events"]
                if event["bytes"] == point_count * 8
            ]
            assert full_nndistance_transfers == [
                "device_to_host", "host_to_device"
            ], full_nndistance_transfers
            assert_manager_execution_breakdown(
                host_repair_execution, expect_nndistance=True)
            host_repair = host_repair_execution["exact_host_repair"]
            assert host_repair["incomplete_rows"] > 0, host_repair
            assert host_repair["repaired_rows"] == host_repair[
                "incomplete_rows"], host_repair
            assert host_repair["seconds"] > 0.0, host_repair
            assert host_repair_execution["exact_device_repair"][
                "repaired_rows"] == 0, host_repair_execution

            print(json.dumps({
                "v2": {
                    "point_count": point_count,
                    "executor": execution["executor"],
                    "pipeline_class": schedule["pipeline_class"],
                    "peak_lane_bytes": schedule["peak_lane_bytes"],
                    "memory_budget_bytes": schedule["memory_budget_bytes"],
                    "output_sha256": sha256(candidate_output),
                    "nndistance_automatic_output_sha256": sha256(
                        nndistance_output),
                    "nndistance_source_disabled_output_sha256": sha256(
                        nndistance_source_disabled_output),
                    "nndistance_direct_output_sha256": sha256(
                        nndistance_direct_output),
                    "nndistance_direct_repair_rows": nndistance_repair[
                        "repaired_rows"],
                    "nndistance_direct_host_repair_rows": host_repair[
                        "repaired_rows"],
                }
            }, sort_keys=True))

            memory_after = linux_memory_bytes()
            swap_used_after = (
                memory_after["SwapTotal"] - memory_after["SwapFree"])
            assert swap_used_after <= swap_used_before + 256 * 1024 * 1024, (
                swap_used_before, swap_used_after)
            return 0

        if args.mode == "v1":
            resident = root / "expression"
            resident.mkdir()
            environment = fixed_direct_scheduler_environment()
            candidate, stats, candidate_output, oracle_output, oracle_pipeline = (
                run_case(args.pdg, args.oracle, large_input, resident, False,
                         environment, expression=True))
            placement = stats["placement"]
            assert placement["available"] is True, placement
            assert placement["choice"] == "device", placement
            assert placement["selected_region_count"] == 1, placement

            # The declared writer-prologue fusion candidate routes the
            # terminal expression chain onto the ordered direct executor,
            # whose measured ordered-point-program residual now matches the
            # selected calibration (D0064).
            execution = stats["execution"]
            assert execution["executor"] == "direct_ordered_las", execution
            assert (
                execution["selected_device_calibration_matches_executor"]
                is True
            ), execution
            point_count = las_point_count(large_input)
            schedule = execution["schedule"]
            assert schedule["pipeline_class"] == (
                "ordered_point_program"), schedule
            assert schedule["item_count"] == point_count, schedule
            assert schedule["configured_lane_count"] == 2, schedule
            assert schedule["active_lane_count"] == 2, schedule
            assert schedule["memory_limited"] is False, schedule

            reference = run([
                str(args.oracle), "pipeline", str(oracle_pipeline),
            ])
            assert_exact(candidate, reference, candidate_output, oracle_output)
            survivors = las_point_count(oracle_output)
            assert 0 < survivors < point_count, (survivors, point_count)
            assert schedule["observed_output_item_count"] == survivors, (
                schedule, survivors)

            # The generic keep-mask executor remains covered: a host
            # randomize stage before the expression region blocks endpoint
            # fusion, so the region crosses the planner-owned PointView
            # boundary with its declared one-byte mask.
            generic = root / "generic-control"
            generic.mkdir()
            generic_output = generic / "candidate.las"
            generic_oracle_output = generic / "oracle.las"
            generic_stats_path = generic / "stats.json"
            generic_pipeline = write_randomize_expression_pipeline(
                generic / "candidate.json", large_input, generic_output)
            generic_oracle_pipeline = write_randomize_expression_pipeline(
                generic / "oracle.json", large_input, generic_oracle_output)
            generic_candidate = run(
                [str(args.pdg), "resident", str(generic_pipeline),
                 "--stats", str(generic_stats_path)], environment)
            assert generic_candidate.returncode == 0, generic_candidate.stderr
            generic_stats = json.loads(
                generic_stats_path.read_text(encoding="utf-8"))
            generic_placement = generic_stats["placement"]
            assert generic_placement["available"] is True, generic_placement
            assert generic_placement["choice"] == "device", generic_placement
            assert generic_placement["boundary_accounting_model"] == (
                "executor_declared"), generic_placement
            generic_execution = generic_stats["execution"]
            assert generic_execution["executor"] == (
                "planner_resident_boundary_batch"), generic_execution
            assert generic_execution["rewrite_executable"] is True, (
                generic_execution)
            assert generic_execution["resident_preflight"]["accepted"] is True
            assert_boundary_accounting(generic_placement, generic_execution)
            planned = generic_placement["planned_boundaries"]
            upload = next(boundary for boundary in planned
                          if boundary["kind"] == "upload")
            spill = next(boundary for boundary in planned
                         if boundary["kind"] == "spill")
            assert upload["point_count"] == point_count, upload
            assert spill["point_count"] == point_count, spill
            # The declared cardinality change spills the complete input tile
            # plus one keep-mask byte per input point.
            record_stride = upload["predicted_transfer_bytes"] // point_count
            assert upload["predicted_transfer_bytes"] == (
                record_stride * point_count), upload
            assert spill["predicted_transfer_bytes"] == (
                (record_stride + 1) * point_count), spill
            generic_schedule = generic_execution["schedule"]
            assert generic_schedule["pipeline_class"] == (
                "ordered_point_program"), generic_schedule
            assert generic_schedule["tile_item_capacity"] == 131_072, (
                generic_schedule)
            assert 0 < generic_schedule["observed_peak_lane_bytes"] <= (
                generic_schedule["peak_lane_bytes"]), generic_schedule
            generic_reference = run([
                str(args.oracle), "pipeline", str(generic_oracle_pipeline),
            ])
            assert_exact(generic_candidate, generic_reference, generic_output,
                         generic_oracle_output)
            generic_survivors = las_point_count(generic_oracle_output)
            assert 0 < generic_survivors < point_count, generic_survivors
            assert generic_schedule["observed_output_item_count"] == (
                generic_survivors), (generic_schedule, generic_survivors)

            # Declared where/where_merge semantics keep the resident path
            # closed: the stage contract, not option names, rejects the
            # region, and the untouched host pipeline stays exact.
            control = root / "where-control"
            control.mkdir()
            control_output = control / "candidate.las"
            control_oracle_output = control / "oracle.las"
            control_stats = control / "stats.json"
            control_pipeline = write_where_expression_pipeline(
                control / "candidate.json", args.small_input.resolve(),
                control_output)
            control_oracle_pipeline = write_where_expression_pipeline(
                control / "oracle.json", args.small_input.resolve(),
                control_oracle_output)
            control_candidate = run([
                str(args.pdg), "resident", str(control_pipeline),
                "--stats", str(control_stats),
            ])
            control_reference = run([
                str(args.oracle), "pipeline", str(control_oracle_pipeline),
            ])
            assert_exact(control_candidate, control_reference, control_output,
                         control_oracle_output)
            where_stats = json.loads(
                control_stats.read_text(encoding="utf-8"))
            where_placement = where_stats["placement"]
            assert where_placement["available"] is False, where_placement
            assert where_placement["unavailable_reason"] == (
                "non_cardinality_preserving_stage"), where_placement
            assert where_stats["execution"]["executor"] == (
                "pdal_standard_host"), where_stats

            print(json.dumps({
                "v1": {
                    "point_count": point_count,
                    "survivors": survivors,
                    "executor": execution["executor"],
                    "tile_count": schedule["tile_count"],
                    "tile_item_capacity": schedule["tile_item_capacity"],
                    "configured_lane_count": schedule[
                        "configured_lane_count"],
                    "active_lane_count": schedule["active_lane_count"],
                    "lane_reuse_count": schedule["lane_reuse_count"],
                    "peak_lane_bytes": schedule["peak_lane_bytes"],
                    "generic_control_executor": generic_execution["executor"],
                    "generic_control_survivors": generic_survivors,
                    "output_sha256": sha256(candidate_output),
                }
            }, sort_keys=True))

            memory_after = linux_memory_bytes()
            swap_used_after = (
                memory_after["SwapTotal"] - memory_after["SwapFree"])
            assert swap_used_after <= swap_used_before + 256 * 1024 * 1024, (
                swap_used_before, swap_used_after)
            return 0

        if args.mode == "direct":
            direct = root / "direct"
            direct.mkdir()
            direct_environment = fixed_direct_scheduler_environment()
            direct_environment[
                "PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES"] = str((1 << 64) - 1)
            candidate, stats, candidate_output, oracle_output, oracle_pipeline = (
                run_case(args.pdg, args.oracle, large_input, direct, False,
                         direct_environment))
            placement = stats["placement"]
            assert placement["available"] is True, placement
            assert placement["choice"] == "device", placement
            assert placement["selected_region_count"] == 1, placement
            assert placement["boundary_accounting_model"] == (
                "calibration_default"), placement
            assert placement["validation_budget_override_bytes"] is None, (
                placement)
            assert [region["selected"] for region in placement["regions"]] == [
                True]

            execution = stats["execution"]
            assert execution["executor"] == "direct_fused_las", execution
            assert (
                execution["selected_device_calibration_matches_executor"] is True
            ), execution
            assert execution["selected_regions"] == [0], execution
            schedule = execution["schedule"]
            assert schedule["pipeline_class"] == "fused_point_program", schedule
            assert schedule["configured_lane_count"] == 2, schedule
            assert schedule["active_lane_count"] == 2, schedule
            assert schedule["tile_count"] > 2, schedule
            assert schedule["peak_lane_bytes"] > 0, schedule
            assert schedule["lane_reuse_count"] > 0, schedule
            assert schedule["memory_budget_bytes"] >= schedule[
                "peak_lane_bytes"], schedule
            assert [event["kind"] for event in execution["events"]] == [
                "device_region_begin",
                "host_to_device",
                "device_to_host",
                "device_region_end",
            ], execution
            totals = execution["totals"]
            assert totals["device_region_begin"]["count"] == 1, totals
            assert totals["device_region_end"]["count"] == 1, totals
            assert totals["host_to_device"]["bytes"] > 0, totals
            assert totals["device_to_host"]["bytes"] > 0, totals
            assert totals["fallback_boundary"]["count"] == 0, totals
            input_point_bytes = (
                las_point_count(large_input) * las_record_length(large_input))
            output_point_bytes = las_point_count(large_input) * 36
            input_summary_bytes = (
                totals["host_to_device"]["bytes"] - input_point_bytes)
            output_summary_bytes = (
                totals["device_to_host"]["bytes"] - output_point_bytes)
            assert input_summary_bytes == output_summary_bytes > 0, totals
            assert input_summary_bytes % schedule["tile_count"] == 0, totals

            reference = run([str(args.oracle), "pipeline", str(oracle_pipeline)])
            assert_exact(candidate, reference, candidate_output, oracle_output)

            # The direct writer must not replace a pre-existing artifact. This
            # uses a distinct output so the exact successful differential above
            # remains independent of the no-overwrite assertion.
            protected = direct / "protected.las"
            protected.write_bytes(b"gpupal resident no-overwrite sentinel\n")
            protected_digest = sha256(protected)
            protected_pipeline = write_pipeline(
                direct / "protected.json", large_input, protected, False)
            no_overwrite = run(
                [str(args.pdg), "resident", str(protected_pipeline)],
                fixed_direct_scheduler_environment())
            assert no_overwrite.returncode != 0, no_overwrite
            assert sha256(protected) == protected_digest

            memory_after = linux_memory_bytes()
            swap_used_after = (
                memory_after["SwapTotal"] - memory_after["SwapFree"])
            assert swap_used_after <= swap_used_before + 256 * 1024 * 1024, (
                swap_used_before, swap_used_after)
            return 0

        mixed = root / "mixed"
        mixed.mkdir()
        candidate, stats, candidate_output, oracle_output, oracle_pipeline = (
            run_case(args.pdg, args.oracle, large_input, mixed, True))
        placement = stats["placement"]
        assert placement["available"] is True, placement
        assert placement["choice"] == "device", placement
        assert placement["boundary_accounting_model"] == (
            "executor_declared"), placement
        assert placement["selected_region_count"] == 2, placement
        assert [region["selected"] for region in placement["regions"]] == [
            True, True]
        fallback = [boundary for boundary in placement["planned_boundaries"]
                    if boundary["fallback"]]
        assert [boundary["kind"] for boundary in fallback] == [
            "spill", "upload"]

        execution = stats["execution"]
        assert execution["executor"] == "planner_resident_boundary_batch", execution
        assert (
            execution["selected_device_calibration_matches_executor"] is False
        ), execution
        assert execution["rewrite_executable"] is True, execution
        assert execution["selected_regions"] == [0, 1], execution
        assert execution["resident_preflight"]["attempted"] is True, execution
        assert execution["resident_preflight"]["accepted"] is True, execution
        assert [boundary["kind"] for boundary in
                execution["observed_crossings"]] == [
                    "upload", "spill", "upload", "spill"]
        assert_boundary_accounting(placement, execution)
        assert all(boundary["transfer_bytes"] > 0 for boundary in
                   execution["observed_crossings"])
        assert all(
            boundary["evidence"] == (
                "materialized_full_record_with_observed_pack")
            for boundary in execution["observed_crossings"]
        )
        assert [boundary["boundary_id"] for boundary in
                execution["observed_crossings"]] == [
                    boundary["id"]
                    for boundary in placement["planned_boundaries"]
                ]
        schedule = execution["schedule"]
        assert schedule["pipeline_class"] == "fused_point_program", schedule
        assert schedule["configured_lane_count"] == 2, schedule
        assert schedule["active_lane_count"] == 2, schedule
        assert placement["predicted"]["configured_device_lane_count"] == (
            schedule["configured_lane_count"]), (placement, schedule)
        assert placement["predicted"]["active_device_lane_count"] == (
            schedule["active_lane_count"]), (placement, schedule)
        assert placement["predicted"]["peak_device_bytes"] == schedule[
            "peak_lane_bytes"], (placement, schedule)
        assert schedule["tile_count"] > schedule["active_lane_count"], schedule
        assert schedule["lane_reuse_count"] > 0, schedule
        assert 0 < schedule["observed_peak_lane_bytes"] <= schedule[
            "peak_lane_bytes"], schedule
        totals = execution["totals"]
        assert totals["device_region_begin"]["count"] == 2, totals
        assert totals["device_region_end"]["count"] == 2, totals
        assert totals["boundary_upload"]["count"] == 2, totals
        assert totals["boundary_spill"]["count"] == 2, totals
        assert totals["fallback_boundary"]["count"] == 2, totals

        reference = run([str(args.oracle), "pipeline", str(oracle_pipeline)])
        assert_exact(candidate, reference, candidate_output, oracle_output)
        memory_after = linux_memory_bytes()
        swap_used_after = memory_after["SwapTotal"] - memory_after["SwapFree"]
        assert swap_used_after <= swap_used_before + 256 * 1024 * 1024, (
            swap_used_before, swap_used_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
