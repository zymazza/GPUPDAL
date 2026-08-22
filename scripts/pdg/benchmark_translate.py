#!/usr/bin/env python3
"""Benchmark exact LAS translation or a LAS pipeline against pinned PDAL.

The runner treats byte identity as a precondition rather than a post-hoc
observation.  Every warmup and measured output must have the same SHA-256,
process status, stdout, and stderr before a timing report is published.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


class BenchmarkError(RuntimeError):
    """Raised when the comparison is invalid or exactness is lost."""


def positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--pipeline",
        type=Path,
        help="optional pipeline JSON whose first/last filenames are replaced",
    )
    parser.add_argument("--input-label", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--runs", type=positive, default=15)
    parser.add_argument("--warmups", type=nonnegative, default=2)
    parser.add_argument(
        "--candidate-mode",
        choices=("native", "hybrid-host", "hybrid-cuda", "default",
                 "resident"),
        default="native",
        help=(
            "candidate execution gate; native preserves the historical "
            "direct-LAS benchmark behavior; resident runs the diagnostic "
            "`gpupdal resident` command with stats-based executor proof"
        ),
    )
    parser.add_argument(
        "--require-resident-executor",
        choices=("planner_resident_boundary_batch",
                 "planner_resident_shared_index", "direct_fused_las",
                 "direct_ordered_las",
                 "planner_resident_shared_index_direct_las",
                 "planner_resident_global_order_direct_las"),
        help=(
            "fail unless every resident candidate run reports this executor "
            "with device placement and an accepted preflight"
        ),
    )
    parser.add_argument(
        "--require-automatic-resident-las-output",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the "
            "qualified resident LAS output route"
        ),
    )
    parser.add_argument(
        "--require-automatic-eigen-family-resident",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the exact "
            "calibrated eigen-family resident route"
        ),
    )
    parser.add_argument(
        "--require-automatic-rank-optimal-resident",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the exact "
            "calibrated rank/optimal resident route"
        ),
    )
    parser.add_argument(
        "--require-automatic-outlier-nndistance-resident",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the exact "
            "calibrated direct outlier/NNDistance composition"
        ),
    )
    parser.add_argument(
        "--require-automatic-radius-outlier-radialdensity-resident",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the exact "
            "calibrated direct radius-outlier/radial-density composition"
        ),
    )
    parser.add_argument(
        "--require-automatic-radiusassign-resident",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the exact "
            "calibrated direct radiusassign endpoint"
        ),
    )
    parser.add_argument(
        "--require-automatic-neighborclassifier-resident",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the exact "
            "calibrated direct neighborclassifier composition"
        ),
    )
    parser.add_argument(
        "--require-automatic-skewness-resident",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the exact "
            "calibrated direct skewness composition"
        ),
    )
    parser.add_argument(
        "--require-automatic-sort-resident",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the exact "
            "calibrated direct Z-sort composition"
        ),
    )
    parser.add_argument(
        "--require-automatic-hag-nn-resident",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the exact "
            "calibrated direct HAG-NN composition"
        ),
    )
    parser.add_argument(
        "--require-automatic-hag-delaunay-resident",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the exact "
            "calibrated direct HAG-Delaunay composition"
        ),
    )
    parser.add_argument(
        "--require-automatic-label-nndistance-hybrid",
        action="store_true",
        help=(
            "fail unless an option-free candidate uses the exact calibrated "
            "label_duplicates/NNDistance/assignment hybrid composition"
        ),
    )
    parser.add_argument(
        "--require-automatic-approximatecoplanar-resident",
        action="store_true",
        help=(
            "fail unless an option-free candidate pipeline uses the exact "
            "calibrated direct approximate-coplanar composition and "
            "positively exercises its pinned host repair"
        ),
    )
    parser.add_argument(
        "--require-neighborhood-reuse",
        action="store_true",
        help=(
            "fail unless a candidate pipeline reuses its planner-owned "
            "neighborhood product"
        ),
    )
    parser.add_argument(
        "--require-knn-gather-reuse",
        action="store_true",
        help=(
            "fail unless a resident candidate consumes its planner-owned "
            "ordered max-k kNN gather through exactly one shared index"
        ),
    )
    parser.add_argument(
        "--experimental-direct-classification-output",
        action="store_true",
        help=(
            "enable the exact experimental direct resident Classification "
            "output route; requires its direct-source proof"
        ),
    )
    parser.add_argument(
        "--require-direct-extra-double-output",
        action="store_true",
        help=(
            "enable and fail unless a resident candidate publishes its "
            "exact single binary64 Extra Bytes column directly"
        ),
    )
    parser.add_argument(
        "--require-direct-skewness-composition",
        action="store_true",
        help=(
            "enable and fail unless a resident candidate uses the exact "
            "mapped-source skewness permutation and direct LAS publisher"
        ),
    )
    parser.add_argument(
        "--require-direct-sort-composition",
        action="store_true",
        help=(
            "enable and fail unless a resident candidate uses the exact "
            "mapped-source Z-sort permutation and direct LAS publisher"
        ),
    )
    parser.add_argument(
        "--require-radius-outlier-radialdensity-composition",
        action="store_true",
        help=(
            "enable and fail unless the exact explicit radius-outlier/"
            "radial-density resident composition is used"
        ),
    )
    parser.add_argument(
        "--require-nnd-device-repair",
        action="store_true",
        help=(
            "fail unless an incomplete nndistance row uses the bounded "
            "exact device-repair path"
        ),
    )
    parser.add_argument(
        "--require-nnd-parallel-repair",
        action="store_true",
        help=(
            "enable and fail unless an incomplete nndistance row uses the "
            "parallel exact device-repair path"
        ),
    )
    parser.add_argument(
        "--require-outlier-device-repair",
        action="store_true",
        help=(
            "fail unless an incomplete statistical-outlier mean row uses "
            "the bounded exact device-repair path"
        ),
    )
    parser.add_argument(
        "--require-outlier-parallel-repair",
        action="store_true",
        help=(
            "fail unless the statistical-outlier selective device repair "
            "uses its parallel exact selection path"
        ),
    )
    parser.add_argument(
        "--require-lof-parallel-repair",
        action="store_true",
        help=(
            "fail unless an incomplete local-outlier-factor selective device "
            "repair uses its parallel exact selection path"
        ),
    )
    parser.add_argument(
        "--require-lof-kd3-coordinate-cache",
        action="store_true",
        help=(
            "fail unless an incomplete local-outlier-factor selective "
            "repair uses its KD3 coordinate cache"
        ),
    )
    parser.add_argument(
        "--require-approximate-coplanar-host-repair",
        action="store_true",
        help=(
            "fail unless approximate-coplanar triggers its exact host KD3 "
            "repair with a symmetric eigensystem round trip"
        ),
    )
    parser.add_argument(
        "--require-direct-las-resident-source",
        action="store_true",
        help=(
            "fail unless the candidate hydrates the resident neighborhood "
            "product directly from the mapped LAS input"
        ),
    )
    parser.add_argument(
        "--require-direct-las-record-summary",
        action="store_true",
        help=(
            "fail unless the direct LAS source configures the shared index "
            "from its exact mapped-record summary"
        ),
    )
    parser.add_argument(
        "--require-no-direct-las-host-xyz",
        action="store_true",
        help=(
            "fail unless the direct LAS source avoids a complete resident "
            "host XYZ mirror"
        ),
    )
    parser.add_argument(
        "--require-resident-calibration-provenance",
        action="store_true",
        help=(
            "fail unless resident telemetry proves the selected device "
            "calibration matches the executing endpoint"
        ),
    )
    parser.add_argument(
        "--require-resident-boundary-accounting",
        action="store_true",
        help=(
            "fail unless resident telemetry proves observed boundary "
            "accounting matches the placement prediction"
        ),
    )
    parser.add_argument(
        "--require-nnd-device-only-handoff",
        action="store_true",
        help=(
            "fail unless the exact direct-LAS NNDistance-to-assignment "
            "pipeline retains NNDistance on device"
        ),
    )
    parser.add_argument(
        "--scheduler-lanes",
        type=int,
        choices=range(2, 7),
        help="fixed candidate CUDA scheduler width for a D3 lane sweep",
    )
    parser.add_argument(
        "--candidate-native-workers",
        type=positive,
        help=(
            "set the positive candidate-only PDG_NATIVE_WORKERS value for "
            "benchmark runs"
        ),
    )
    parser.add_argument(
        "--require-fused-cuda",
        action="store_true",
        help="fail unless the candidate uses descriptor-planned fused CUDA",
    )
    parser.add_argument(
        "--require-cuda-translate",
        action="store_true",
        help="fail unless a direct candidate translation uses CUDA",
    )
    parser.add_argument(
        "--require-cuda-point-program",
        action="store_true",
        help="fail unless an ordered candidate point program uses CUDA",
    )
    parser.add_argument(
        "--require-spatial-tiling",
        action="store_true",
        help="fail unless a candidate radius stage uses spatial tiling",
    )
    parser.add_argument(
        "--spatial-tile-edge",
        type=positive_float,
        help="fixed candidate spatial tile edge for a D3 lane sweep",
    )
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--freeze-epoch", type=int, default=1_704_067_200)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def companion_engine_provenance(candidate: Path) -> dict[str, str | None]:
    engine = candidate.resolve(strict=True).with_name("pdg-engine")
    if not engine.is_file():
        return {"engine_path": None, "engine_binary_sha256": None}
    return {
        "engine_path": str(engine),
        "engine_binary_sha256": sha256_file(engine),
    }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="replace").strip()


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def las_metadata(path: Path) -> dict[str, int | str]:
    with path.open("rb") as stream:
        header = stream.read(375)
    if len(header) < 227 or header[:4] != b"LASF":
        raise BenchmarkError(f"input is not a complete LAS header: {path}")
    version_major = header[24]
    version_minor = header[25]
    point_format = header[104] & 0x3F
    record_bytes = struct.unpack_from("<H", header, 105)[0]
    legacy_count = struct.unpack_from("<I", header, 107)[0]
    if version_major == 1 and version_minor >= 4 and len(header) >= 255:
        point_count = struct.unpack_from("<Q", header, 247)[0]
    else:
        point_count = legacy_count
    return {
        "version": f"{version_major}.{version_minor}",
        "point_format": point_format,
        "point_record_bytes": record_bytes,
        "point_count": point_count,
    }


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def sample_summary(seconds: list[float], input_bytes: int,
                   point_count: int) -> dict[str, Any]:
    median = statistics.median(seconds)
    return {
        "seconds": seconds,
        "median_seconds": median,
        "p5_seconds": percentile(seconds, 0.05),
        "p95_seconds": percentile(seconds, 0.95),
        "minimum_seconds": min(seconds),
        "maximum_seconds": max(seconds),
        "median_input_gb_per_second": input_bytes / median / 1_000_000_000,
        "median_million_points_per_second": point_count / median / 1_000_000,
    }


def parse_cache(build_dir: Path) -> dict[str, str]:
    wanted = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_COMPILER_VERSION",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_FLAGS",
        "CMAKE_CUDA_FLAGS_RELEASE",
        "CMAKE_CUDA_ARCHITECTURES",
        "GPUPDAL_ENABLE_CUDA",
        "WITH_PDG",
    }
    result: dict[str, str] = {}
    try:
        lines = (build_dir / "CMakeCache.txt").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return result
    for line in lines:
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if key in wanted:
            result[key] = value
    return result


def git_metadata(repo_root: Path) -> dict[str, Any]:
    head = command_output(["git", "rev-parse", "HEAD"], repo_root)
    upstream = command_output(
        ["git", "rev-parse", "upstream/master"], repo_root
    )
    branch = command_output(
        ["git", "branch", "--show-current"], repo_root
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if diff.returncode != 0:
        detail = diff.stderr.decode("utf-8", errors="replace").strip()
        raise BenchmarkError(
            "unable to capture the working-tree diff" +
            (f": {detail}" if detail else
             f" (git exited {diff.returncode})")
        )
    untracked_output = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if untracked_output.returncode != 0:
        raise BenchmarkError("unable to enumerate untracked source files")
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(diff.stdout)
    untracked: list[str] = []
    for encoded in untracked_output.stdout.split(b"\0"):
        if not encoded:
            continue
        relative = encoded.decode("utf-8", errors="surrogateescape")
        path = repo_root / relative
        if not path.is_file():
            continue
        untracked.append(relative)
        digest.update(b"untracked\0")
        digest.update(encoded)
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    dirty = bool(diff.stdout or untracked)
    return {
        "head": head,
        "upstream_master": upstream,
        "branch": branch,
        "dirty": dirty,
        "working_tree_sha256": digest.hexdigest() if dirty else None,
        "untracked_file_count": len(untracked),
    }


def machine_metadata(work_dir: Path) -> dict[str, Any]:
    cpu_model = None
    cpuinfo = read_optional(Path("/proc/cpuinfo"))
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    mem_total = None
    meminfo = read_optional(Path("/proc/meminfo"))
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                mem_total = line.split(":", 1)[1].strip()
                break
    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    toolkit = command_output(["nvcc", "--version"])
    storage = command_output(
        ["findmnt", "-T", str(work_dir), "-n", "-o", "SOURCE,FSTYPE,OPTIONS"]
    )
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "allowed_cpu_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "memory_total": mem_total,
        "storage": storage,
        "gpu": gpu,
        "cuda_toolkit": toolkit,
        "scaling_governor": read_optional(
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        ),
    }


def base_environment(oracle: Path, frozen_time_library: Path,
                     freeze_epoch: int) -> dict[str, str]:
    result = os.environ.copy()
    result.update(
        {
            "LC_ALL": "C",
            "TZ": "UTC",
            "PDG_ORACLE_PDAL": str(oracle),
            "PDAL_TEST_FROZEN_EPOCH": str(freeze_epoch),
        }
    )
    preload = str(frozen_time_library)
    if result.get("LD_PRELOAD"):
        preload += ":" + result["LD_PRELOAD"]
    result["LD_PRELOAD"] = preload
    for name in (
        "PDG_DISABLE_NATIVE",
        "PDG_REQUIRE_NATIVE",
        "PDG_DISABLE_HYBRID",
        "PDG_REQUIRE_HYBRID",
        "PDG_DISABLE_CUDA_HYBRID",
        "PDG_REQUIRE_CUDA_HYBRID",
        "PDG_EXPERIMENTAL_CUDA_HYBRID",
        "PDG_CUDA_SCHEDULER_LANES",
        "PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM",
        "PDG_REQUIRE_CUDA_POINT_PROGRAM",
        "PDG_REQUIRE_CUDA_TRANSLATE",
        "PDG_REQUIRE_SPATIAL_TILING",
        "PDG_SPATIAL_TILE_EDGE",
        "PDG_TEST_RESIDENT_VRAM_BUDGET_BYTES",
        "PDG_CUDA_CHUNK_POINTS",
        "PDG_EXPERIMENTAL_DIRECT_RESIDENT_LAS_OUTPUT",
        "PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT",
        "PDG_REQUIRE_AUTOMATIC_RESIDENT_LAS_OUTPUT",
        "PDG_REQUIRE_AUTOMATIC_EIGEN_FAMILY_RESIDENT",
        "PDG_REQUIRE_AUTOMATIC_RANK_OPTIMAL_RESIDENT",
        "PDG_REQUIRE_AUTOMATIC_OUTLIER_NNDISTANCE_RESIDENT",
        "PDG_REQUIRE_AUTOMATIC_RADIUS_OUTLIER_RADIALDENSITY_RESIDENT",
        "PDG_REQUIRE_AUTOMATIC_RADIUSASSIGN_RESIDENT",
        "PDG_REQUIRE_AUTOMATIC_NEIGHBORCLASSIFIER_RESIDENT",
        "PDG_REQUIRE_AUTOMATIC_SKEWNESS_RESIDENT",
        "PDG_REQUIRE_AUTOMATIC_SORT_RESIDENT",
        "PDG_REQUIRE_AUTOMATIC_HAG_NN_RESIDENT",
        "PDG_REQUIRE_AUTOMATIC_R2_GROUND_NORMALIZE",
        "PDG_REQUIRE_HAG_NN_SELECTIVE_REPAIR",
        "PDG_INTERNAL_AUTOMATIC_R2_HYBRID",
        "PDG_TEST_AUTOMATIC_R2_HAG_NN_DEVICE_DECLINE",
        "PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT",
        "PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID",
        "PDG_TEST_LABEL_NNDISTANCE_RECOVERABLE_CUDA_FAILURE",
        "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_RESIDENT",
        "PDG_REQUIRE_NEIGHBORHOOD_REUSE",
        "PDG_REQUIRE_KNN_GATHER_REUSE",
        "PDG_KNN_DEVICE_SHELL_BUDGET",
        "PDG_DISABLE_KNN_DISTANCE_PREFILTER",
        "PDG_FORCE_MORTON_BVH",
        "PDG_FORCE_UNIFORM_GRID",
        "PDG_DISABLE_NEIGHBORHOOD_ROW_BOUNDARY",
        "PDG_REQUIRE_NEIGHBORHOOD_ROW_BOUNDARY",
        "PDG_EXPERIMENTAL_DIRECT_CLASSIFICATION_OUTPUT",
        "PDG_EXPERIMENTAL_DIRECT_EXTRA_DOUBLE_OUTPUT",
        "PDG_REQUIRE_DIRECT_EXTRA_DOUBLE_OUTPUT",
        "PDG_EXPERIMENTAL_DIRECT_SKEWNESS_COMPOSITION",
        "PDG_REQUIRE_DIRECT_SKEWNESS_COMPOSITION",
        "PDG_EXPERIMENTAL_DIRECT_SORT_COMPOSITION",
        "PDG_REQUIRE_DIRECT_SORT_COMPOSITION",
        "PDG_REQUIRE_NND_DEVICE_REPAIR",
        "PDG_DISABLE_NND_DEVICE_REPAIR",
        "PDG_EXPERIMENTAL_NND_PARALLEL_REPAIR",
        "PDG_DISABLE_NND_PARALLEL_REPAIR",
        "PDG_REQUIRE_NND_PARALLEL_REPAIR",
        "PDG_REQUIRE_OUTLIER_DEVICE_REPAIR",
        "PDG_DISABLE_OUTLIER_DEVICE_REPAIR",
        "PDG_REQUIRE_OUTLIER_PARALLEL_REPAIR",
        "PDG_DISABLE_OUTLIER_PARALLEL_REPAIR",
        "PDG_REQUIRE_LOF_PARALLEL_REPAIR",
        "PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE",
        "PDG_DISABLE_LOF_KD3_COORDINATE_CACHE",
        "PDG_TEST_KD3_CACHE_BUILD_FAILURE",
        "PDG_REQUIRE_RADIUS_OUTLIER_RADIALDENSITY_COMPOSITION",
        "PDG_NATIVE_WORKERS",
        "PDG_DISABLE_HOST_NEIGHBORHOOD_WORKERS",
        "PDG_DISABLE_KD3_COORDINATE_CACHE",
        "PDAL_TEST_VERIFY_KD3_SNAPSHOT",
        "PDAL_TEST_FORCE_HOST_NEIGHBORHOOD_WORKERS",
        "PDAL_TEST_REQUIRE_HOST_NEIGHBORHOOD_WORKERS",
        "PDG_EXPERIMENTAL_DIRECT_LAS_RESIDENT_SOURCE",
        "PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE",
        "PDG_DISABLE_DIRECT_LAS_RESIDENT_SOURCE",
        "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY",
        "PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ",
        "PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY_BACKEND",
        "PDG_REQUIRE_NND_DEVICE_ONLY_HANDOFF",
        "PDG_REQUIRE_NND_HOST_RESTORE",
        "PDG_TEST_AUTOMATIC_OUTLIER_PROOF_FAILURE",
        "PDG_TEST_AUTOMATIC_RADIUS_COMPOSITION_PROOF_FAILURE",
        "PDG_TEST_AUTOMATIC_RADIUSASSIGN_PROOF_FAILURE",
        "PDG_TEST_AUTOMATIC_HAG_NN_PROOF_FAILURE",
        "PDG_TEST_AUTOMATIC_HAG_DELAUNAY_PROOF_FAILURE",
        "PDG_TEST_AUTOMATIC_COPLANAR_PROOF_FAILURE",
        "PDG_TEST_DIRECT_SKEWNESS_PROOF_FAILURE",
        "PDG_TEST_DIRECT_SKEWNESS_PREFLIGHT_FAILURE",
        "PDG_TEST_DIRECT_SORT_PROOF_FAILURE",
        "PDG_TEST_DIRECT_SORT_PREFLIGHT_FAILURE",
    ):
        result.pop(name, None)
    return result


def pipeline_stages(value: Any) -> list[Any]:
    stages = value.get("pipeline") if isinstance(value, dict) else value
    if not isinstance(stages, list) or len(stages) < 2:
        raise BenchmarkError("pipeline must contain at least two stages")
    if not isinstance(stages[0], dict) or not isinstance(stages[-1], dict):
        raise BenchmarkError("pipeline reader and writer must be objects")
    return stages


def output_target(output_path: Path, writer_filename: object) -> tuple[Path, bool]:
    """Return the isolated writer target and whether it is PDAL-numbered."""
    if not isinstance(writer_filename, str):
        raise BenchmarkError("pipeline writer filename must be a string")
    markers = writer_filename.count("#")
    if markers > 1:
        raise BenchmarkError(
            "benchmark runner supports one numbered-writer marker"
        )
    if not markers:
        return output_path, False
    return output_path.with_name(
        f"{output_path.stem}-#{output_path.suffix}"
    ), True


def output_artifact_paths(target: Path, numbered: bool) -> list[Path]:
    if not numbered:
        return [target] if target.is_file() else []
    prefix, suffix = target.name.split("#", 1)
    result = [
        path for path in target.parent.iterdir()
        if path.is_file()
        and path.name.startswith(prefix)
        and path.name.endswith(suffix)
        and len(path.name) > len(prefix) + len(suffix)
    ]
    return sorted(result, key=lambda path: path.name)


def collect_output_artifacts(target: Path, numbered: bool) -> list[dict[str, Any]]:
    paths = output_artifact_paths(target, numbered)
    if not paths:
        raise BenchmarkError("process produced no output artifacts")
    artifacts: list[dict[str, Any]] = []
    prefix = ""
    suffix = target.suffix
    if numbered:
        prefix, suffix = target.name.split("#", 1)
    for path in paths:
        logical_name = (
            path.name[len(prefix):] if numbered else f"<output>{suffix}"
        )
        artifacts.append({
            "name": logical_name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return artifacts


def output_manifest_sha256(artifacts: list[dict[str, Any]]) -> str:
    return sha256_bytes(json.dumps(
        artifacts, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))


def validate_resident_stats(stats_path: Path,
                            required_executor: str | None,
                            require_direct_las_resident_source: bool,
                            require_direct_las_record_summary: bool,
                            require_no_direct_las_host_xyz: bool,
                            require_nnd_device_only_handoff: bool,
                            require_resident_calibration_provenance: bool,
                            require_resident_boundary_accounting: bool,
                            require_knn_gather_reuse: bool = False,
                            require_nnd_device_repair: bool = False,
                            require_nnd_parallel_repair: bool = False,
                            require_outlier_device_repair: bool = False,
                            require_outlier_parallel_repair: bool = False,
                            require_approximate_coplanar_host_repair: bool = False,
                            require_direct_extra_double_output: bool = False,
                            require_direct_skewness_composition: bool = False,
                            require_direct_sort_composition: bool = False,
                            ) -> dict[str, Any]:
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(
            f"resident candidate produced no readable stats: {error}"
        ) from error
    placement = stats.get("placement", {})
    execution = stats.get("execution", {})
    if required_executor is not None:
        if placement.get("available") is not True:
            raise BenchmarkError(
                "resident placement is unavailable: "
                + str(placement.get("unavailable_reason"))
            )
        if placement.get("choice") != "device":
            raise BenchmarkError(
                "resident placement selected "
                + str(placement.get("choice"))
            )
        if execution.get("executor") != required_executor:
            raise BenchmarkError(
                "resident executor was "
                + str(execution.get("executor"))
                + f", required {required_executor}"
            )
        if required_executor in (
                "planner_resident_boundary_batch",
                "planner_resident_shared_index",
                "planner_resident_shared_index_direct_las",
                "planner_resident_global_order_direct_las"):
            preflight = execution.get("resident_preflight", {})
            if (execution.get("rewrite_executable") is not True
                    or preflight.get("accepted") is not True):
                raise BenchmarkError(
                    "resident preflight was not accepted: "
                    + str(preflight.get("reason"))
                )
        if required_executor == "planner_resident_boundary_batch":
            if execution.get(
                    "boundary_accounting_matches_prediction") is not True:
                raise BenchmarkError(
                    "resident boundary accounting did not match prediction"
                )
        if required_executor == "planner_resident_shared_index":
            schedule = execution.get("schedule", {})
            if schedule.get("pipeline_class") != "whole_view_neighborhood":
                raise BenchmarkError(
                    "resident schedule class was "
                    + str(schedule.get("pipeline_class"))
                )
        if required_executor == "planner_resident_shared_index_direct_las":
            schedule = execution.get("schedule", {})
            if schedule.get("pipeline_class") not in (
                    "whole_view_neighborhood", "fused_point_program"):
                raise BenchmarkError(
                    "direct LAS resident schedule class was "
                    + str(schedule.get("pipeline_class"))
                )
        if required_executor == "planner_resident_global_order_direct_las":
            schedule = execution.get("schedule", {})
            if schedule.get("pipeline_class") != "whole_view_global_order":
                raise BenchmarkError(
                    "direct LAS global-order schedule class was "
                    + str(schedule.get("pipeline_class"))
                )
    if (require_direct_las_resident_source
            and execution.get("direct_las_resident_source") is not True):
        raise BenchmarkError(
            "required direct LAS resident source was not observed"
        )
    if (require_direct_las_record_summary
            and execution.get("direct_las_record_summary") is not True):
        raise BenchmarkError(
            "required direct LAS record summary was not observed"
        )
    if (require_no_direct_las_host_xyz
            and execution.get("direct_las_host_xyz_mirror") is not False):
        raise BenchmarkError(
            "direct LAS resident host XYZ mirror was observed"
        )
    if require_direct_extra_double_output:
        if (execution.get("direct_las_output") is not True
                or execution.get("direct_extra_double_output") is not True
                or execution.get("terminal_spill_elided") is not True):
            raise BenchmarkError(
                "required direct Extra Bytes binary64 output was not "
                "observed"
            )
    if require_direct_skewness_composition:
        schedule = execution.get("schedule")
        index_builds = execution.get("index_builds")
        events = execution.get("events")
        item_count = (
            schedule.get("item_count") if isinstance(schedule, dict)
            else None
        )
        observed_count = (
            schedule.get("observed_output_item_count")
            if isinstance(schedule, dict) else None
        )
        host_to_device = [
            event.get("bytes") for event in events
            if isinstance(event, dict)
            and event.get("kind") == "host_to_device"
        ] if isinstance(events, list) else None
        device_to_host = [
            event.get("bytes") for event in events
            if isinstance(event, dict)
            and event.get("kind") == "device_to_host"
        ] if isinstance(events, list) else None
        expected_transfer = (
            item_count * 8 if type(item_count) is int and item_count > 0
            else None
        )
        if (execution.get("executor") !=
                "planner_resident_global_order_direct_las"
                or execution.get("direct_las_output") is not True
                or execution.get("direct_las_resident_source") is not True
                or execution.get(
                    "direct_permuted_classification_output") is not True
                or execution.get("direct_las_record_summary") is not False
                or execution.get("direct_las_host_xyz_mirror") is not False
                or execution.get("terminal_spill_elided") is not True
                or execution.get(
                    "boundary_accounting_matches_prediction") is not True
                or not isinstance(schedule, dict)
                or schedule.get("pipeline_class") !=
                    "whole_view_global_order"
                or expected_transfer is None
                or type(observed_count) is not int
                or observed_count != item_count
                or execution.get("selected_regions") != [0]
                or execution.get("selected_stage_ids") != [1]
                or not isinstance(index_builds, dict)
                or index_builds.get("matches_prediction") is not True
                or type(index_builds.get("predicted")) is not int
                or index_builds.get("predicted") != 0
                or type(index_builds.get("observed")) is not int
                or index_builds.get("observed") != 0
                or host_to_device != [expected_transfer]
                or device_to_host != [expected_transfer]):
            raise BenchmarkError(
                "required direct skewness composition was not observed"
            )
    if require_direct_sort_composition:
        schedule = execution.get("schedule")
        index_builds = execution.get("index_builds")
        events = execution.get("events")
        item_count = (
            schedule.get("item_count") if isinstance(schedule, dict)
            else None
        )
        observed_count = (
            schedule.get("observed_output_item_count")
            if isinstance(schedule, dict) else None
        )
        host_to_device = [
            event.get("bytes") for event in events
            if isinstance(event, dict)
            and event.get("kind") == "host_to_device"
        ] if isinstance(events, list) else None
        device_to_host = [
            event.get("bytes") for event in events
            if isinstance(event, dict)
            and event.get("kind") == "device_to_host"
        ] if isinstance(events, list) else None
        expected_transfer = (
            item_count * 8 if type(item_count) is int and item_count > 0
            else None
        )
        if (execution.get("executor") !=
                "planner_resident_global_order_direct_las"
                or execution.get("direct_las_output") is not True
                or execution.get("direct_las_resident_source") is not True
                or execution.get("direct_permuted_output") is not True
                or execution.get("direct_permuted_sort_output") is not True
                or execution.get(
                    "direct_permuted_classification_output") is not False
                or execution.get("direct_las_record_summary") is not False
                or execution.get("direct_las_host_xyz_mirror") is not False
                or execution.get("terminal_spill_elided") is not True
                or execution.get(
                    "boundary_accounting_matches_prediction") is not True
                or not isinstance(schedule, dict)
                or schedule.get("pipeline_class") !=
                    "whole_view_global_order"
                or expected_transfer is None
                or type(observed_count) is not int
                or observed_count != item_count
                or execution.get("selected_regions") != [0]
                or execution.get("selected_stage_ids") != [1]
                or not isinstance(index_builds, dict)
                or index_builds.get("matches_prediction") is not True
                or type(index_builds.get("predicted")) is not int
                or index_builds.get("predicted") != 0
                or type(index_builds.get("observed")) is not int
                or index_builds.get("observed") != 0
                or host_to_device != [expected_transfer]
                or device_to_host != [expected_transfer]):
            raise BenchmarkError(
                "required direct sort composition was not observed"
            )
    if require_nnd_device_only_handoff:
        if execution.get("nndistance_device_only_handoff") is not True:
            raise BenchmarkError(
                "required NNDistance device-only handoff was not observed"
            )
        if execution.get(
                "nndistance_assignment_device_column_reuse") is not True:
            raise BenchmarkError(
                "required NNDistance assignment device-column reuse was not "
                "observed"
            )
    if (require_resident_calibration_provenance
            and execution.get(
                "selected_device_calibration_matches_executor") is not True):
        raise BenchmarkError(
            "resident calibration did not match the selected executor"
        )
    if (require_resident_boundary_accounting
            and execution.get(
                "boundary_accounting_matches_prediction") is not True):
        raise BenchmarkError(
            "resident boundary accounting did not match prediction"
        )
    if (require_knn_gather_reuse
            and execution.get("knn_gather_reuse") is not True):
        raise BenchmarkError(
            "required planner-owned max-k kNN gather reuse was not observed"
        )
    index_builds = execution.get("index_builds")
    if (required_executor == "planner_resident_shared_index_direct_las"
            and (not isinstance(index_builds, dict)
                 or index_builds.get("matches_prediction") is not True
                 or type(index_builds.get("predicted")) is not int
                 or index_builds.get("predicted") != 1
                 or type(index_builds.get("observed")) is not int
                 or index_builds.get("observed") != 1)):
        raise BenchmarkError(
            "direct LAS shared-index executor did not use exactly one "
            "planned index"
        )
    if (require_knn_gather_reuse
            and (not isinstance(index_builds, dict)
                 or index_builds.get("matches_prediction") is not True
                 or type(index_builds.get("predicted")) is not int
                 or index_builds.get("predicted") != 1
                 or type(index_builds.get("observed")) is not int
                 or index_builds.get("observed") != 1)):
        raise BenchmarkError(
            "max-k kNN gather reuse did not use exactly one shared index"
        )
    exact_host_repair = execution.get("exact_host_repair")
    exact_device_repair = execution.get("exact_device_repair")
    outlier_host_repair = (
        exact_host_repair.get("statistical_outlier")
        if isinstance(exact_host_repair, dict) else None
    )
    outlier_device_repair = (
        exact_device_repair.get("statistical_outlier")
        if isinstance(exact_device_repair, dict) else None
    )
    nnd_host_repair = (
        exact_host_repair.get("nndistance")
        if isinstance(exact_host_repair, dict) else None
    )
    nnd_device_repair = (
        exact_device_repair.get("nndistance")
        if isinstance(exact_device_repair, dict) else None
    )
    approximate_coplanar_host_repair = (
        exact_host_repair.get("approximate_coplanar")
        if isinstance(exact_host_repair, dict) else None
    )
    if approximate_coplanar_host_repair is not None:
        if not isinstance(approximate_coplanar_host_repair, dict):
            raise BenchmarkError(
                "approximate-coplanar host repair telemetry is malformed"
            )
        repair = approximate_coplanar_host_repair
        seconds = repair.get("seconds")
        integer_fields = (
            "trigger_count", "ambiguous_rows", "incomplete_rows",
            "repaired_rows", "kd3_uses", "device_to_host_bytes",
            "host_to_device_bytes",
        )
        if (type(seconds) not in (int, float)
                or not math.isfinite(seconds) or seconds < 0.0
                or type(repair.get("triggered")) is not bool
                or type(repair.get("kd3_used")) is not bool
                or any(type(repair.get(name)) is not int
                       or repair.get(name) < 0 for name in integer_fields)):
            raise BenchmarkError(
                "approximate-coplanar host repair telemetry is malformed"
            )
        triggered = repair["trigger_count"] > 0
        kd3_used = repair["kd3_uses"] > 0
        repaired_rows = repair["repaired_rows"]
        ambiguous_rows = repair["ambiguous_rows"]
        incomplete_rows = repair["incomplete_rows"]
        repair_bytes = repair["device_to_host_bytes"]
        if (repair["triggered"] != triggered
                or repair["kd3_used"] != kd3_used
                or triggered != kd3_used
                or repair["trigger_count"] != repair["kd3_uses"]
                or repair_bytes != repair["host_to_device_bytes"]
                or (triggered and
                    (repaired_rows <= 0
                     or ambiguous_rows + incomplete_rows <= 0
                     or repaired_rows < max(ambiguous_rows, incomplete_rows)
                     or repaired_rows > ambiguous_rows + incomplete_rows
                     or repair_bytes <= 0))
                or (not triggered and
                    (seconds != 0.0 or ambiguous_rows != 0
                     or incomplete_rows != 0 or repaired_rows != 0
                     or repair_bytes != 0))):
            raise BenchmarkError(
                "approximate-coplanar host repair telemetry is inconsistent"
            )
    if (require_approximate_coplanar_host_repair
            and (not isinstance(approximate_coplanar_host_repair, dict)
                 or approximate_coplanar_host_repair.get("triggered") is not
                    True)):
        raise BenchmarkError(
            "required approximate-coplanar exact host repair was not "
            "observed"
        )
    if require_nnd_device_repair:
        device_incomplete = (
            nnd_device_repair.get("incomplete_rows")
            if isinstance(nnd_device_repair, dict) else None
        )
        device_repaired = (
            nnd_device_repair.get("repaired_rows")
            if isinstance(nnd_device_repair, dict) else None
        )
        host_repaired = (
            nnd_host_repair.get("repaired_rows")
            if isinstance(nnd_host_repair, dict) else None
        )
        if (type(device_incomplete) is not int or device_incomplete <= 0
                or type(device_repaired) is not int
                or device_repaired != device_incomplete
                or type(host_repaired) is not int or host_repaired != 0):
            raise BenchmarkError(
                "required NNDistance device repair was not observed without "
                "host repair"
            )
    if require_nnd_parallel_repair:
        parallel_repaired = (
            nnd_device_repair.get("parallel_repaired_rows")
            if isinstance(nnd_device_repair, dict) else None
        )
        repaired = (
            nnd_device_repair.get("repaired_rows")
            if isinstance(nnd_device_repair, dict) else None
        )
        if (type(parallel_repaired) is not int or parallel_repaired <= 0
                or type(repaired) is not int
                or parallel_repaired != repaired):
            raise BenchmarkError(
                "required parallel NNDistance device repair was not observed"
            )
    if require_outlier_device_repair:
        device_incomplete = (
            outlier_device_repair.get("incomplete_rows")
            if isinstance(outlier_device_repair, dict) else None
        )
        device_repaired = (
            outlier_device_repair.get("repaired_rows")
            if isinstance(outlier_device_repair, dict) else None
        )
        host_repaired = (
            outlier_host_repair.get("repaired_rows")
            if isinstance(outlier_host_repair, dict) else None
        )
        if (type(device_incomplete) is not int or device_incomplete <= 0
                or type(device_repaired) is not int
                or device_repaired != device_incomplete
                or type(host_repaired) is not int or host_repaired != 0):
            raise BenchmarkError(
                "required statistical-outlier device repair was not "
                "observed without host repair"
            )
    if require_outlier_parallel_repair:
        parallel_repaired = (
            outlier_device_repair.get("parallel_repaired_rows")
            if isinstance(outlier_device_repair, dict) else None
        )
        if type(parallel_repaired) is not int or parallel_repaired <= 0:
            raise BenchmarkError(
                "required parallel statistical-outlier device repair was "
                "not observed"
            )
    return {
        "executor": execution.get("executor"),
        "schedule": execution.get("schedule"),
        "direct_las_output": execution.get("direct_las_output"),
        "direct_las_resident_source": execution.get(
            "direct_las_resident_source"),
        "direct_las_record_summary": execution.get(
            "direct_las_record_summary"),
        "direct_las_host_xyz_mirror": execution.get(
            "direct_las_host_xyz_mirror"),
        "direct_extra_double_output": execution.get(
            "direct_extra_double_output"),
        "direct_permuted_classification_output": execution.get(
            "direct_permuted_classification_output"),
        "direct_permuted_output": execution.get("direct_permuted_output"),
        "direct_permuted_sort_output": execution.get(
            "direct_permuted_sort_output"),
        "terminal_spill_elided": execution.get("terminal_spill_elided"),
        "nndistance_device_only_handoff": execution.get(
            "nndistance_device_only_handoff"),
        "nndistance_host_restore": execution.get(
            "nndistance_host_restore"),
        "nndistance_assignment_device_column_reuse": execution.get(
            "nndistance_assignment_device_column_reuse"),
        "selected_device_calibration_matches_executor": execution.get(
            "selected_device_calibration_matches_executor"),
        "boundary_accounting_matches_prediction": execution.get(
            "boundary_accounting_matches_prediction"),
        "knn_gather_reuse": execution.get("knn_gather_reuse"),
        "index_builds": index_builds,
        "nndistance_host_repair": nnd_host_repair,
        "nndistance_device_repair": nnd_device_repair,
        "outlier_host_repair": outlier_host_repair,
        "outlier_device_repair": outlier_device_repair,
        "approximate_coplanar_host_repair":
            approximate_coplanar_host_repair,
        "placement_choice": placement.get("choice"),
        "schedule": execution.get("schedule"),
    }


def run_once(label: str, program: Path, input_path: Path, output_path: Path,
             environment: dict[str, str],
             pipeline_template: Any | None,
             candidate_mode: str, scheduler_lanes: int | None,
             require_fused_cuda: bool, require_cuda_translate: bool,
             require_cuda_point_program: bool,
             require_spatial_tiling: bool,
             spatial_tile_edge: float | None,
             require_resident_executor: str | None = None,
             require_automatic_resident_las_output: bool = False,
             require_automatic_eigen_family_resident: bool = False,
             require_automatic_rank_optimal_resident: bool = False,
             require_automatic_outlier_nndistance_resident: bool = False,
             require_automatic_radius_outlier_radialdensity_resident:
                 bool = False,
             require_automatic_radiusassign_resident: bool = False,
             require_automatic_neighborclassifier_resident: bool = False,
             require_automatic_hag_nn_resident: bool = False,
             require_automatic_hag_delaunay_resident: bool = False,
             require_automatic_skewness_resident: bool = False,
             require_automatic_approximatecoplanar_resident: bool = False,
             require_neighborhood_reuse: bool = False,
             require_knn_gather_reuse: bool = False,
             require_nnd_device_repair: bool = False,
             require_nnd_parallel_repair: bool = False,
             require_outlier_device_repair: bool = False,
             require_outlier_parallel_repair: bool = False,
             require_approximate_coplanar_host_repair: bool = False,
             require_direct_las_resident_source: bool = False,
             require_direct_las_record_summary: bool = False,
             require_no_direct_las_host_xyz: bool = False,
             require_nnd_device_only_handoff: bool = False,
             require_resident_calibration_provenance: bool = False,
             require_resident_boundary_accounting: bool = False,
             experimental_direct_classification_output: bool = False,
             require_direct_extra_double_output: bool = False,
             require_radius_outlier_radialdensity_composition: bool = False,
             require_direct_skewness_composition: bool = False,
             require_direct_sort_composition: bool = False,
             require_automatic_sort_resident: bool = False,
             require_automatic_label_nndistance_hybrid: bool = False,
             candidate_native_workers: int | None = None,
             require_lof_parallel_repair: bool = False,
             require_lof_kd3_coordinate_cache: bool = False,
             ) -> dict[str, Any]:
    env = environment.copy()
    resident_candidate = (
        label == "candidate" and candidate_mode == "resident"
    )
    if label == "candidate":
        if candidate_mode == "native":
            env["PDG_REQUIRE_NATIVE"] = "1"
        elif candidate_mode in ("hybrid-host", "hybrid-cuda"):
            env["PDG_DISABLE_NATIVE"] = "1"
            env["PDG_REQUIRE_HYBRID"] = "1"
            if candidate_mode == "hybrid-host":
                env["PDG_DISABLE_CUDA_HYBRID"] = "1"
            else:
                env["PDG_REQUIRE_CUDA_HYBRID"] = "1"
        if scheduler_lanes is not None:
            env["PDG_CUDA_SCHEDULER_LANES"] = str(scheduler_lanes)
        if require_fused_cuda:
            env["PDG_REQUIRE_FUSED_CUDA_POINT_PROGRAM"] = "1"
        if require_cuda_translate:
            env["PDG_REQUIRE_CUDA_TRANSLATE"] = "1"
        if require_cuda_point_program:
            env["PDG_REQUIRE_CUDA_POINT_PROGRAM"] = "1"
        if require_spatial_tiling:
            env["PDG_REQUIRE_SPATIAL_TILING"] = "1"
        if spatial_tile_edge is not None:
            env["PDG_SPATIAL_TILE_EDGE"] = str(spatial_tile_edge)
        if require_resident_executor == (
                "planner_resident_shared_index_direct_las"):
            env["PDG_REQUIRE_DIRECT_RESIDENT_LAS_OUTPUT"] = "1"
        if require_automatic_resident_las_output:
            env["PDG_REQUIRE_AUTOMATIC_RESIDENT_LAS_OUTPUT"] = "1"
        if require_automatic_eigen_family_resident:
            env["PDG_REQUIRE_AUTOMATIC_EIGEN_FAMILY_RESIDENT"] = "1"
        if require_automatic_rank_optimal_resident:
            env["PDG_REQUIRE_AUTOMATIC_RANK_OPTIMAL_RESIDENT"] = "1"
        if require_automatic_outlier_nndistance_resident:
            env[
                "PDG_REQUIRE_AUTOMATIC_OUTLIER_NNDISTANCE_RESIDENT"
            ] = "1"
        if require_automatic_radius_outlier_radialdensity_resident:
            env[
                "PDG_REQUIRE_AUTOMATIC_RADIUS_OUTLIER_RADIALDENSITY_RESIDENT"
            ] = "1"
        if require_automatic_radiusassign_resident:
            env["PDG_REQUIRE_AUTOMATIC_RADIUSASSIGN_RESIDENT"] = "1"
        if require_automatic_neighborclassifier_resident:
            env[
                "PDG_REQUIRE_AUTOMATIC_NEIGHBORCLASSIFIER_RESIDENT"
            ] = "1"
        if require_automatic_hag_nn_resident:
            env["PDG_REQUIRE_AUTOMATIC_HAG_NN_RESIDENT"] = "1"
        if require_automatic_hag_delaunay_resident:
            env["PDG_REQUIRE_AUTOMATIC_HAG_DELAUNAY_RESIDENT"] = "1"
        if require_automatic_skewness_resident:
            env["PDG_REQUIRE_AUTOMATIC_SKEWNESS_RESIDENT"] = "1"
        if require_automatic_sort_resident:
            env["PDG_REQUIRE_AUTOMATIC_SORT_RESIDENT"] = "1"
        if require_automatic_label_nndistance_hybrid:
            env["PDG_REQUIRE_AUTOMATIC_LABEL_NNDISTANCE_HYBRID"] = "1"
        if require_automatic_approximatecoplanar_resident:
            env[
                "PDG_REQUIRE_AUTOMATIC_APPROXIMATECOPLANAR_RESIDENT"
            ] = "1"
        if require_neighborhood_reuse:
            env["PDG_REQUIRE_NEIGHBORHOOD_REUSE"] = "1"
        if require_knn_gather_reuse:
            env["PDG_REQUIRE_KNN_GATHER_REUSE"] = "1"
        if experimental_direct_classification_output:
            env["PDG_EXPERIMENTAL_DIRECT_CLASSIFICATION_OUTPUT"] = "1"
        if require_direct_extra_double_output:
            env["PDG_EXPERIMENTAL_DIRECT_EXTRA_DOUBLE_OUTPUT"] = "1"
            env["PDG_REQUIRE_DIRECT_EXTRA_DOUBLE_OUTPUT"] = "1"
        if require_radius_outlier_radialdensity_composition:
            env[
                "PDG_REQUIRE_RADIUS_OUTLIER_RADIALDENSITY_COMPOSITION"
            ] = "1"
        if require_direct_skewness_composition:
            env["PDG_REQUIRE_DIRECT_SKEWNESS_COMPOSITION"] = "1"
        if require_direct_sort_composition:
            env["PDG_REQUIRE_DIRECT_SORT_COMPOSITION"] = "1"
        if require_nnd_device_repair:
            env["PDG_REQUIRE_NND_DEVICE_REPAIR"] = "1"
        if require_nnd_parallel_repair:
            env["PDG_REQUIRE_NND_PARALLEL_REPAIR"] = "1"
        if require_outlier_device_repair:
            env["PDG_REQUIRE_OUTLIER_DEVICE_REPAIR"] = "1"
        if require_outlier_parallel_repair:
            env["PDG_REQUIRE_OUTLIER_PARALLEL_REPAIR"] = "1"
        if require_lof_parallel_repair:
            env["PDG_REQUIRE_LOF_PARALLEL_REPAIR"] = "1"
        if require_lof_kd3_coordinate_cache:
            env["PDG_REQUIRE_LOF_KD3_COORDINATE_CACHE"] = "1"
        if require_direct_las_resident_source:
            env["PDG_REQUIRE_DIRECT_LAS_RESIDENT_SOURCE"] = "1"
        if require_direct_las_record_summary:
            env["PDG_REQUIRE_DIRECT_LAS_RECORD_SUMMARY"] = "1"
        if require_no_direct_las_host_xyz:
            env["PDG_REQUIRE_NO_DIRECT_LAS_HOST_XYZ"] = "1"
        if require_nnd_device_only_handoff:
            env["PDG_REQUIRE_NND_DEVICE_ONLY_HANDOFF"] = "1"
        if candidate_native_workers is not None:
            env["PDG_NATIVE_WORKERS"] = str(candidate_native_workers)
    invocation = None
    stats_path = None
    target = output_path
    numbered = False
    if pipeline_template is None:
        command = [str(program), "translate", str(input_path), str(output_path)]
    else:
        pipeline = copy.deepcopy(pipeline_template)
        stages = pipeline_stages(pipeline)
        stages[0]["filename"] = str(input_path)
        target, numbered = output_target(
            output_path, stages[-1].get("filename")
        )
        stages[-1]["filename"] = str(target)
        invocation = output_path.with_suffix(".pipeline.json")
        invocation.write_text(
            json.dumps(pipeline, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if resident_candidate:
            stats_path = output_path.with_suffix(".stats.json")
            command = [str(program), "resident", str(invocation),
                       "--stats", str(stats_path)]
        else:
            command = [str(program), "pipeline", str(invocation)]
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        cwd=output_path.parent,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    ended = time.perf_counter_ns()
    if invocation is not None:
        invocation.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise BenchmarkError(
            f"{label} failed with status {completed.returncode}: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    resident_stats = None
    if stats_path is not None:
        resident_stats = validate_resident_stats(
            stats_path, require_resident_executor,
            require_direct_las_resident_source,
            require_direct_las_record_summary,
            require_no_direct_las_host_xyz,
            require_nnd_device_only_handoff,
            require_resident_calibration_provenance,
            require_resident_boundary_accounting,
            require_knn_gather_reuse,
            require_nnd_device_repair,
            require_nnd_parallel_repair,
            require_outlier_device_repair,
            require_outlier_parallel_repair,
            require_approximate_coplanar_host_repair,
            require_direct_extra_double_output,
            require_direct_skewness_composition,
            require_direct_sort_composition,
        )
        stats_path.unlink(missing_ok=True)
    artifacts = collect_output_artifacts(target, numbered)
    total_output_bytes = sum(int(item["bytes"]) for item in artifacts)
    manifest_hash = output_manifest_sha256(artifacts)
    output_hash = (
        str(artifacts[0]["sha256"])
        if not numbered and len(artifacts) == 1
        else manifest_hash
    )
    result = {
        "label": label,
        "seconds": (ended - started) / 1_000_000_000,
        "returncode": completed.returncode,
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_sha256": sha256_bytes(completed.stderr),
        "stdout_bytes": len(completed.stdout),
        "stderr_bytes": len(completed.stderr),
        "output_artifact_count": len(artifacts),
        "output_bytes": total_output_bytes,
        "output_sha256": output_hash,
        "output_manifest_sha256": manifest_hash,
        "outputs": artifacts,
    }
    if resident_stats is not None:
        result["resident"] = resident_stats
    for artifact in output_artifact_paths(target, numbered):
        artifact.unlink()
    return result


def exact_signature(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        result["returncode"],
        result["stdout_sha256"],
        result["stderr_sha256"],
        result["stdout_bytes"],
        result["stderr_bytes"],
        tuple(
            (item["name"], item["bytes"], item["sha256"])
            for item in result["outputs"]
        ),
    )


def validate_paths(args: argparse.Namespace) -> None:
    for label, path in (
        ("oracle", args.oracle),
        ("candidate", args.candidate),
        ("input", args.input),
        ("frozen-time library", args.frozen_time_library),
    ):
        if not path.is_file():
            raise BenchmarkError(f"{label} does not exist: {path}")
    if args.pipeline is not None and not args.pipeline.is_file():
        raise BenchmarkError(f"pipeline does not exist: {args.pipeline}")
    if args.require_cuda_translate and args.pipeline is not None:
        raise BenchmarkError(
            "--require-cuda-translate is only valid without --pipeline"
        )
    if (args.require_fused_cuda or args.require_cuda_point_program
            or args.require_spatial_tiling
            or args.spatial_tile_edge is not None) and args.pipeline is None:
        raise BenchmarkError(
            "point-program and spatial overrides require --pipeline"
        )
    if args.candidate_mode == "resident" and args.pipeline is None:
        raise BenchmarkError("resident candidate mode requires --pipeline")
    if (args.require_resident_executor is not None
            and args.candidate_mode != "resident"):
        raise BenchmarkError(
            "--require-resident-executor requires --candidate-mode resident"
        )
    if (args.require_automatic_resident_las_output
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-resident-las-output requires a pipeline "
            "and --candidate-mode default"
        )
    if (args.require_automatic_eigen_family_resident
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-eigen-family-resident requires a pipeline "
            "and --candidate-mode default"
        )
    if (args.require_automatic_rank_optimal_resident
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-rank-optimal-resident requires a pipeline "
            "and --candidate-mode default"
        )
    if (args.require_automatic_outlier_nndistance_resident
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-outlier-nndistance-resident requires a "
            "pipeline and --candidate-mode default"
        )
    if (args.require_automatic_radius_outlier_radialdensity_resident
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-radius-outlier-radialdensity-resident "
            "requires a pipeline and --candidate-mode default"
        )
    if (args.require_automatic_radiusassign_resident
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-radiusassign-resident requires a pipeline "
            "and --candidate-mode default"
        )
    if (args.require_automatic_neighborclassifier_resident
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-neighborclassifier-resident requires a "
            "pipeline and --candidate-mode default"
        )
    if (args.require_automatic_skewness_resident
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-skewness-resident requires a pipeline and "
            "--candidate-mode default"
        )
    if (args.require_automatic_hag_nn_resident
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-hag-nn-resident requires a pipeline and "
            "--candidate-mode default"
        )
    if (args.require_automatic_hag_delaunay_resident
            and (args.candidate_mode not in ("default", "resident")
                 or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-hag-delaunay-resident requires a "
            "pipeline and --candidate-mode default or resident"
        )
    if (args.require_automatic_sort_resident
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-sort-resident requires a pipeline and "
            "--candidate-mode default"
        )
    if (args.require_automatic_label_nndistance_hybrid
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-label-nndistance-hybrid requires a pipeline "
            "and --candidate-mode default"
        )
    if (args.require_automatic_approximatecoplanar_resident
            and (args.candidate_mode != "default" or args.pipeline is None)):
        raise BenchmarkError(
            "--require-automatic-approximatecoplanar-resident requires a "
            "pipeline and --candidate-mode default"
        )
    if args.require_neighborhood_reuse and args.pipeline is None:
        raise BenchmarkError("--require-neighborhood-reuse requires a pipeline")
    if (args.require_knn_gather_reuse
            and (args.candidate_mode not in ("default", "resident")
                 or args.pipeline is None)):
        raise BenchmarkError(
            "--require-knn-gather-reuse requires a pipeline and "
            "--candidate-mode default or resident"
        )
    if (args.experimental_direct_classification_output
            and (args.candidate_mode != "resident"
                 or args.pipeline is None
                 or not args.require_direct_las_resident_source)):
        raise BenchmarkError(
            "--experimental-direct-classification-output requires a "
            "resident pipeline and --require-direct-las-resident-source"
        )
    if (args.require_direct_extra_double_output
            and (args.candidate_mode != "resident"
                 or args.pipeline is None)):
        raise BenchmarkError(
            "--require-direct-extra-double-output requires a resident "
            "pipeline"
        )
    if (args.require_direct_skewness_composition
            and (args.candidate_mode != "resident"
                 or args.pipeline is None
                 or args.require_resident_executor !=
                    "planner_resident_global_order_direct_las")):
        raise BenchmarkError(
            "--require-direct-skewness-composition requires a resident "
            "pipeline and --require-resident-executor "
            "planner_resident_global_order_direct_las"
        )
    if (args.require_direct_sort_composition
            and (args.candidate_mode != "resident"
                 or args.pipeline is None
                 or args.require_resident_executor !=
                    "planner_resident_global_order_direct_las")):
        raise BenchmarkError(
            "--require-direct-sort-composition requires a resident "
            "pipeline and --require-resident-executor "
            "planner_resident_global_order_direct_las"
        )
    if (args.require_radius_outlier_radialdensity_composition
            and (args.candidate_mode != "resident"
                 or args.pipeline is None)):
        raise BenchmarkError(
            "--require-radius-outlier-radialdensity-composition requires "
            "a resident pipeline"
        )
    if args.require_nnd_device_repair and args.pipeline is None:
        raise BenchmarkError(
            "--require-nnd-device-repair requires a pipeline"
        )
    if (args.require_nnd_parallel_repair
            and (args.candidate_mode not in ("default", "resident")
                 or args.pipeline is None)):
        raise BenchmarkError(
            "--require-nnd-parallel-repair requires a pipeline and default "
            "or resident candidate mode"
        )
    if (args.require_nnd_parallel_repair
            and (not args.require_nnd_device_repair
                 or not args.require_direct_las_resident_source)):
        raise BenchmarkError(
            "--require-nnd-parallel-repair requires "
            "--require-nnd-device-repair and "
            "--require-direct-las-resident-source"
        )
    if (args.require_outlier_device_repair
            and (args.candidate_mode != "resident"
                 or args.pipeline is None)):
        raise BenchmarkError(
            "--require-outlier-device-repair requires a pipeline and "
            "--candidate-mode resident"
        )
    if (args.require_outlier_parallel_repair
            and (not args.require_outlier_device_repair
                 or not args.require_direct_las_resident_source)):
        raise BenchmarkError(
            "--require-outlier-parallel-repair requires "
            "--require-outlier-device-repair and "
            "--require-direct-las-resident-source"
        )
    if (args.require_lof_parallel_repair
            and (args.candidate_mode not in ("default", "resident")
                 or args.pipeline is None)):
        raise BenchmarkError(
            "--require-lof-parallel-repair requires a pipeline and "
            "default or resident candidate mode"
        )
    if (args.require_lof_parallel_repair
            and (args.candidate_native_workers is not None
                 and args.candidate_native_workers < 2)):
        raise BenchmarkError(
            "--require-lof-parallel-repair requires "
            "--candidate-native-workers >= 2"
        )
    if (args.require_lof_kd3_coordinate_cache
            and (args.candidate_mode not in ("default", "resident")
                 or args.pipeline is None)):
        raise BenchmarkError(
            "--require-lof-kd3-coordinate-cache requires a pipeline and "
            "default or resident candidate mode"
        )
    if (args.require_approximate_coplanar_host_repair
            and (args.candidate_mode != "resident"
                 or args.pipeline is None)):
        raise BenchmarkError(
            "--require-approximate-coplanar-host-repair requires a pipeline "
            "and --candidate-mode resident"
        )
    if (args.require_direct_las_resident_source
            and (args.candidate_mode not in ("default", "resident")
                 or args.pipeline is None)):
        raise BenchmarkError(
            "--require-direct-las-resident-source requires a pipeline and "
            "default or resident candidate mode"
        )
    if ((args.require_direct_las_record_summary
         or args.require_no_direct_las_host_xyz
         or args.require_nnd_device_only_handoff)
            and (args.candidate_mode not in ("default", "resident")
                 or args.pipeline is None)):
        raise BenchmarkError(
            "direct LAS resident proofs require a pipeline and default or "
            "resident candidate mode"
        )
    if ((args.require_direct_las_record_summary
         or args.require_no_direct_las_host_xyz
         or args.require_nnd_device_only_handoff)
            and not args.require_direct_las_resident_source):
        raise BenchmarkError(
            "direct LAS resident proofs require "
            "--require-direct-las-resident-source"
        )
    if ((args.require_resident_calibration_provenance
         or args.require_resident_boundary_accounting)
            and (args.candidate_mode != "resident"
                 or args.require_resident_executor is None)):
        raise BenchmarkError(
            "resident placement provenance proofs require "
            "--candidate-mode resident and --require-resident-executor"
        )
    for label, path in (("oracle", args.oracle), ("candidate", args.candidate)):
        if not os.access(path, os.X_OK):
            raise BenchmarkError(f"{label} is not executable: {path}")
    if args.report.exists():
        raise BenchmarkError(f"refusing to overwrite benchmark report: {args.report}")


def main() -> int:
    args = parse_args()
    try:
        args.repo_root = args.repo_root.resolve()
        args.oracle = args.oracle.resolve()
        args.candidate = args.candidate.resolve()
        args.input = args.input.resolve()
        if args.pipeline is not None:
            args.pipeline = args.pipeline.resolve()
        args.frozen_time_library = args.frozen_time_library.resolve()
        args.work_dir = args.work_dir.resolve()
        args.report = args.report.resolve()
        validate_paths(args)
        args.work_dir.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)

        input_bytes = args.input.stat().st_size
        input_las = las_metadata(args.input)
        input_hash = sha256_file(args.input)
        pipeline_template = None
        pipeline_template_bytes = None
        if args.pipeline is not None:
            pipeline_template_bytes = args.pipeline.read_bytes()
            try:
                pipeline_template = json.loads(pipeline_template_bytes)
            except json.JSONDecodeError as error:
                raise BenchmarkError(f"invalid pipeline JSON: {error}") from error
            pipeline_stages(pipeline_template)
        repository = git_metadata(args.repo_root)
        environment = base_environment(
            args.oracle, args.frozen_time_library, args.freeze_epoch
        )
        load_before = os.getloadavg() if hasattr(os, "getloadavg") else None

        all_runs: list[dict[str, Any]] = []
        expected: tuple[Any, ...] | None = None
        automatic_resident_proof: dict[str, Any] | None = None
        with tempfile.TemporaryDirectory(
            prefix="pdg-benchmark-", dir=args.work_dir
        ) as temporary:
            temporary_path = Path(temporary)
            phases = (("warmup", args.warmups), ("measured", args.runs))
            sequence = 0
            for phase, count in phases:
                for iteration in range(count):
                    order = (
                        ("oracle", args.oracle),
                        ("candidate", args.candidate),
                    )
                    if iteration % 2:
                        order = tuple(reversed(order))
                    for label, program in order:
                        output = temporary_path / f"{sequence:04d}-{label}.las"
                        result = run_once(
                            label, program, args.input, output, environment,
                            pipeline_template, args.candidate_mode,
                            args.scheduler_lanes, args.require_fused_cuda,
                            args.require_cuda_translate,
                            args.require_cuda_point_program,
                            args.require_spatial_tiling,
                            args.spatial_tile_edge,
                            args.require_resident_executor,
                            args.require_automatic_resident_las_output,
                            args.require_automatic_eigen_family_resident,
                            args.require_automatic_rank_optimal_resident,
                            args.require_automatic_outlier_nndistance_resident,
                            args.require_automatic_radius_outlier_radialdensity_resident,
                            args.require_automatic_radiusassign_resident,
                            args.require_automatic_neighborclassifier_resident,
                            args.require_automatic_hag_nn_resident,
                            args.require_automatic_hag_delaunay_resident,
                            args.require_automatic_skewness_resident,
                            args.require_automatic_approximatecoplanar_resident,
                            args.require_neighborhood_reuse,
                            args.require_knn_gather_reuse,
                            args.require_nnd_device_repair,
                            args.require_nnd_parallel_repair,
                            args.require_outlier_device_repair,
                            args.require_outlier_parallel_repair,
                            args.require_approximate_coplanar_host_repair,
                            args.require_direct_las_resident_source,
                            args.require_direct_las_record_summary,
                            args.require_no_direct_las_host_xyz,
                            args.require_nnd_device_only_handoff,
                            args.require_resident_calibration_provenance,
                            args.require_resident_boundary_accounting,
                            args.experimental_direct_classification_output,
                            args.require_direct_extra_double_output,
                            args.require_radius_outlier_radialdensity_composition,
                            args.require_direct_skewness_composition,
                            args.require_direct_sort_composition,
                            args.require_automatic_sort_resident,
                            args.require_automatic_label_nndistance_hybrid,
                            candidate_native_workers=args.candidate_native_workers,
                            require_lof_parallel_repair=
                                args.require_lof_parallel_repair,
                            require_lof_kd3_coordinate_cache=
                                args.require_lof_kd3_coordinate_cache,
                        )
                        result.update(
                            {
                                "phase": phase,
                                "iteration": iteration,
                                "sequence": sequence,
                            }
                        )
                        signature = exact_signature(result)
                        if expected is None:
                            expected = signature
                        elif signature != expected:
                            raise BenchmarkError(
                                f"exactness mismatch during {phase} run "
                                f"{iteration} for {label}"
                            )
                        all_runs.append(result)
                        sequence += 1

            # Default-mode proof flags fail closed inside every timed
            # candidate process. Preserve observed internal telemetry too,
            # without charging this diagnostic resident run to the samples.
            if (args.candidate_mode == "default"
                    and args.require_nnd_device_only_handoff):
                proof_output = temporary_path / "automatic-proof.las"
                proof = run_once(
                    "candidate", args.candidate, args.input, proof_output,
                    environment, pipeline_template, "resident",
                    args.scheduler_lanes, args.require_fused_cuda,
                    args.require_cuda_translate,
                    args.require_cuda_point_program,
                    args.require_spatial_tiling, args.spatial_tile_edge,
                    "planner_resident_shared_index_direct_las",
                    args.require_automatic_resident_las_output,
                    args.require_automatic_eigen_family_resident,
                    args.require_automatic_rank_optimal_resident,
                    args.require_automatic_outlier_nndistance_resident,
                    args.require_automatic_radius_outlier_radialdensity_resident,
                    args.require_automatic_radiusassign_resident,
                    args.require_automatic_neighborclassifier_resident,
                    args.require_automatic_hag_nn_resident,
                    args.require_automatic_hag_delaunay_resident,
                    args.require_automatic_skewness_resident,
                    args.require_automatic_approximatecoplanar_resident,
                    args.require_neighborhood_reuse,
                    args.require_knn_gather_reuse,
                    args.require_nnd_device_repair,
                    args.require_nnd_parallel_repair,
                    args.require_outlier_device_repair,
                    args.require_outlier_parallel_repair,
                    args.require_approximate_coplanar_host_repair,
                    args.require_direct_las_resident_source,
                    args.require_direct_las_record_summary,
                    args.require_no_direct_las_host_xyz,
                    args.require_nnd_device_only_handoff,
                    args.require_resident_calibration_provenance,
                    args.require_resident_boundary_accounting,
                    args.experimental_direct_classification_output,
                    args.require_direct_extra_double_output,
                    args.require_radius_outlier_radialdensity_composition,
                    args.require_direct_skewness_composition,
                    args.require_direct_sort_composition,
                    args.require_automatic_sort_resident,
                    args.require_automatic_label_nndistance_hybrid,
                    candidate_native_workers=args.candidate_native_workers,
                    require_lof_parallel_repair=
                        args.require_lof_parallel_repair,
                    require_lof_kd3_coordinate_cache=
                        args.require_lof_kd3_coordinate_cache,
                )
                if expected is None or exact_signature(proof) != expected:
                    raise BenchmarkError(
                        "exactness mismatch during untimed automatic resident "
                        "proof run"
                    )
                automatic_resident_proof = proof.get("resident")
                if automatic_resident_proof is None:
                    raise BenchmarkError(
                        "untimed automatic resident proof produced no stats"
                    )

        measured = [run for run in all_runs if run["phase"] == "measured"]
        oracle_seconds = [
            float(run["seconds"])
            for run in measured
            if run["label"] == "oracle"
        ]
        candidate_seconds = [
            float(run["seconds"])
            for run in measured
            if run["label"] == "candidate"
        ]
        oracle_summary = sample_summary(
            oracle_seconds, input_bytes, int(input_las["point_count"])
        )
        candidate_summary = sample_summary(
            candidate_seconds, input_bytes, int(input_las["point_count"])
        )
        if pipeline_template is None:
            pipeline = {
                "command": ["translate", "<input>", "<output>"],
                "sha256": sha256_bytes(
                    json.dumps(
                        ["translate", input_hash, "<output>"],
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
            }
        else:
            normalized_pipeline = copy.deepcopy(pipeline_template)
            normalized_stages = pipeline_stages(normalized_pipeline)
            normalized_stages[0]["filename"] = f"sha256:{input_hash}"
            _, numbered = output_target(
                Path("output.las"), normalized_stages[-1].get("filename")
            )
            normalized_stages[-1]["filename"] = (
                "<output>#.las" if numbered else "<output>"
            )
            assert args.pipeline is not None
            assert pipeline_template_bytes is not None
            pipeline = {
                "command": ["pipeline", "<materialized-pipeline>"],
                "template_path": str(args.pipeline),
                "template_sha256": sha256_bytes(pipeline_template_bytes),
                "sha256": sha256_bytes(
                    json.dumps(
                        normalized_pipeline, sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
            }
        output_hash = all_runs[0]["output_sha256"]
        candidate_build_dir = args.candidate.parent.parent
        oracle_build_dir = args.oracle.parent.parent
        candidate_engine = companion_engine_provenance(args.candidate)
        report = {
            "schema": 1,
            "recorded_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "mode": "exact",
            "cache_state": (
                f"warm; {args.warmups} untimed run(s) per executable; "
                "no explicit page-cache eviction"
            ),
            "interleaving": (
                "oracle/candidate order alternates within each iteration"
            ),
            "repository": repository,
            "input": {
                "label": args.input_label,
                "path": str(args.input),
                "bytes": input_bytes,
                "sha256": input_hash,
                **input_las,
            },
            "pipeline": pipeline,
            "oracle": {
                "path": str(args.oracle),
                "binary_sha256": sha256_file(args.oracle),
                "version": command_output([str(args.oracle), "--version"]),
                "build": parse_cache(oracle_build_dir),
                "summary": oracle_summary,
            },
            "candidate": {
                "path": str(args.candidate),
                "binary_sha256": sha256_file(args.candidate),
                **candidate_engine,
                "version": command_output([str(args.candidate), "version"]),
                "build": parse_cache(candidate_build_dir),
                "execution_mode": args.candidate_mode,
                "native_path_required": args.candidate_mode == "native",
                "native_workers": args.candidate_native_workers,
                "scheduler_lanes": args.scheduler_lanes,
                "fused_cuda_required": args.require_fused_cuda,
                "lof_parallel_repair_required":
                    args.require_lof_parallel_repair,
                "lof_kd3_coordinate_cache_required":
                    args.require_lof_kd3_coordinate_cache,
                "cuda_translate_required": args.require_cuda_translate,
                "cuda_point_program_required":
                    args.require_cuda_point_program,
                "spatial_tiling_required": args.require_spatial_tiling,
                "spatial_tile_edge": args.spatial_tile_edge,
                "resident_executor_required": args.require_resident_executor,
                "automatic_resident_las_output_required":
                    args.require_automatic_resident_las_output,
                "automatic_eigen_family_resident_required":
                    args.require_automatic_eigen_family_resident,
                "automatic_rank_optimal_resident_required":
                    args.require_automatic_rank_optimal_resident,
                "automatic_outlier_nndistance_resident_required":
                    args.require_automatic_outlier_nndistance_resident,
                "automatic_radius_outlier_radialdensity_resident_required":
                    args.require_automatic_radius_outlier_radialdensity_resident,
                "automatic_radiusassign_resident_required":
                    args.require_automatic_radiusassign_resident,
                "automatic_neighborclassifier_resident_required":
                    args.require_automatic_neighborclassifier_resident,
                "automatic_hag_nn_resident_required":
                    args.require_automatic_hag_nn_resident,
                "automatic_hag_delaunay_resident_required":
                    args.require_automatic_hag_delaunay_resident,
                "automatic_skewness_resident_required":
                    args.require_automatic_skewness_resident,
                "automatic_sort_resident_required":
                    args.require_automatic_sort_resident,
                "automatic_label_nndistance_hybrid_required":
                    args.require_automatic_label_nndistance_hybrid,
                "automatic_approximatecoplanar_resident_required":
                    args.require_automatic_approximatecoplanar_resident,
                "neighborhood_reuse_required":
                    args.require_neighborhood_reuse,
                "knn_gather_reuse_required":
                    args.require_knn_gather_reuse,
                "experimental_direct_classification_output":
                    args.experimental_direct_classification_output,
                "direct_extra_double_output_required":
                    args.require_direct_extra_double_output,
                "direct_skewness_composition_required":
                    args.require_direct_skewness_composition,
                "direct_sort_composition_required":
                    args.require_direct_sort_composition,
                "radius_outlier_radialdensity_composition_required":
                    args.require_radius_outlier_radialdensity_composition,
                "nnd_device_repair_required":
                    args.require_nnd_device_repair,
                "nnd_parallel_repair_required":
                    args.require_nnd_parallel_repair,
                "outlier_device_repair_required":
                    args.require_outlier_device_repair,
                "outlier_parallel_repair_required":
                    args.require_outlier_parallel_repair,
                "approximate_coplanar_host_repair_required":
                    args.require_approximate_coplanar_host_repair,
                "direct_las_resident_source_required":
                    args.require_direct_las_resident_source,
                "direct_las_record_summary_required":
                    args.require_direct_las_record_summary,
                "no_direct_las_host_xyz_required":
                    args.require_no_direct_las_host_xyz,
                "nnd_device_only_handoff_required":
                    args.require_nnd_device_only_handoff,
                "resident_calibration_provenance_required":
                    args.require_resident_calibration_provenance,
                "resident_boundary_accounting_required":
                    args.require_resident_boundary_accounting,
                "resident": next(
                    (
                        run["resident"]
                        for run in reversed(measured)
                        if run["label"] == "candidate" and "resident" in run
                    ),
                    automatic_resident_proof,
                ),
                "summary": candidate_summary,
            },
            "comparison": {
                "sample_count_per_executable": args.runs,
                "median_speedup": oracle_summary["median_seconds"]
                / candidate_summary["median_seconds"],
                "exact_outputs": True,
                "output_artifact_count": all_runs[0]["output_artifact_count"],
                "output_bytes": all_runs[0]["output_bytes"],
                "output_sha256": output_hash,
                "output_manifest_sha256":
                    all_runs[0]["output_manifest_sha256"],
                "outputs": all_runs[0]["outputs"],
            },
            "environment": {
                "freeze_epoch": args.freeze_epoch,
                "frozen_time_library": str(args.frozen_time_library),
                "frozen_time_library_sha256": sha256_file(
                    args.frozen_time_library
                ),
                "load_average_before": load_before,
                "load_average_after": os.getloadavg()
                if hasattr(os, "getloadavg")
                else None,
                **machine_metadata(args.work_dir),
            },
            "runs": all_runs,
        }
        temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_report, args.report)
        print(
            f"exact outputs; median speedup "
            f"{report['comparison']['median_speedup']:.3f}x; "
            f"report {args.report}"
        )
        return 0
    except (BenchmarkError, OSError, ValueError) as error:
        print(f"benchmark invalid: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
