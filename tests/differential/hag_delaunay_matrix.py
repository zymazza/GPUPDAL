#!/usr/bin/env python3
"""Exact process matrix for the bounded filters.hag_delaunay count-three lane."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    fixture: str
    stage: dict[str, object]
    hybrid: bool = True
    cuda: bool = True
    verbose: bool = False
    backend: str | None = None
    cuda_required: bool = True
    proof: str | None = None


CASES = (
    Case("count-three-host-wrapper", "unique-distance", {
        "type": "filters.hag_delaunay", "count": 3,
    }, cuda=False),
    Case("count-ten-host-fallback", "unique-distance", {
        "type": "filters.hag_delaunay",
    }, hybrid=False, cuda=False),
    Case("count-four-host-fallback", "unique-distance", {
        "type": "filters.hag_delaunay", "count": 4,
    }, hybrid=False, cuda=False),
    Case("supported-empty", "empty", {"type": "filters.hag_delaunay",
         "count": 3}),
    Case("custom-ground-class", "custom-class", {
        "type": "filters.hag_delaunay", "count": 3, "class": 9,
    }),
    Case("same-xy", "same-xy", {"type": "filters.hag_delaunay", "count": 3}),
    Case("inside-triangle", "inside-triangle", {
        "type": "filters.hag_delaunay", "count": 3,
    }),
    Case("outside-triangle-nearest", "outside-triangle", {
        "type": "filters.hag_delaunay", "count": 3,
    }),
    Case("outside-global-no-extrapolation", "outside-global", {
        "type": "filters.hag_delaunay", "count": 3,
        "allow_extrapolation": False,
    }),
    Case("all-ground", "all-ground", {
        "type": "filters.hag_delaunay", "count": 3,
    }),
    Case("collinear", "collinear", {"type": "filters.hag_delaunay", "count": 3}),
    Case("duplicate-tie-host-repair", "duplicate-tie", {
        "type": "filters.hag_delaunay", "count": 3,
    }, proof="tie", cuda_required=False),
    Case("insufficient-ground-host-fallback", "insufficient-ground", {
        "type": "filters.hag_delaunay", "count": 3,
    }, proof="insufficient", cuda_required=False),
    Case("no-ground-diagnostic", "no-ground", {
        "type": "filters.hag_delaunay", "count": 3,
    }, verbose=True, proof="insufficient", cuda_required=False),
    Case("nonfinite-z-fallback", "nonfinite-z", {
        "type": "filters.hag_delaunay", "count": 3,
    }, proof="nonfinite", cuda_required=False),
    Case("incomplete-host-fallback", "incomplete", {
        "type": "filters.hag_delaunay", "count": 3,
    }, proof="incomplete", cuda_required=False),
    Case("where-fallback", "unique-distance", {
        "type": "filters.hag_delaunay", "where": "Classification != 2",
    }, hybrid=False, cuda=False),
    Case("count-two-error", "unique-distance", {
        "type": "filters.hag_delaunay", "count": 2,
    }, hybrid=False, cuda=False),
    Case("invalid-class-error", "unique-distance", {
        "type": "filters.hag_delaunay", "class": 256,
    }, hybrid=False, cuda=False),
    Case("missing-classification-error", "missing-classification",
         {"type": "filters.hag_delaunay", "count": 3}, hybrid=False,
         cuda=False),
    Case("forced-uniform-grid", "unique-distance", {
        "type": "filters.hag_delaunay", "count": 3,
    }, backend="grid"),
    Case("forced-morton-bvh", "unique-distance", {
        "type": "filters.hag_delaunay", "count": 3,
    }, backend="bvh"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--differential", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--frozen-time-library", required=True, type=Path)
    parser.add_argument("--oracle-preload", action="append", default=[],
                        type=Path)
    parser.add_argument("--candidate-preload", action="append", default=[],
                        type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def rows(kind: str) -> list[tuple[float, float, float, int]]:
    if kind == "empty" or kind == "missing-classification":
        return []
    if kind == "custom-class":
        return [(0.0, 0.0, 10.0, 9), (5.0, 0.0, 14.0, 9),
                (0.0, 3.0, 18.0, 9), (1.0, 0.5, 30.0, 1)]
    if kind == "same-xy":
        return [(0.0, 0.0, 10.0, 2), (5.0, 0.0, 14.0, 2),
                (0.0, 3.0, 18.0, 2), (0.0, 0.0, 30.0, 1)]
    if kind == "inside-triangle":
        return [(0.0, 0.0, 10.0, 2), (5.0, 0.0, 14.0, 2),
                (0.0, 3.0, 18.0, 2), (1.0, 0.5, 30.0, 1)]
    if kind == "outside-triangle":
        return [(0.0, 0.0, 10.0, 2), (5.0, 0.0, 14.0, 2),
                (0.0, 3.0, 18.0, 2), (4.0, 2.0, 30.0, 1)]
    if kind == "outside-global":
        return [(0.0, 0.0, 10.0, 2), (5.0, 0.0, 14.0, 2),
                (0.0, 3.0, 18.0, 2), (6.0, 4.0, 30.0, 1)]
    if kind == "collinear":
        return [(0.0, 0.0, 10.0, 2), (2.0, 0.0, 14.0, 2),
                (5.0, 0.0, 18.0, 2), (0.5, 1.0, 30.0, 1)]
    if kind == "duplicate-tie":
        return [(0.0, 0.0, 10.0, 2), (0.0, 0.0, 14.0, 2),
                (4.0, 0.0, 18.0, 2), (1.0, 1.0, 30.0, 1)]
    if kind == "insufficient-ground":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (1.0, 1.0, 30.0, 1)]
    if kind == "no-ground":
        return [(0.0, 0.0, 10.0, 1), (4.0, 0.0, 14.0, 1),
                (1.0, 1.0, 30.0, 1)]
    if kind == "nonfinite-z":
        return [(0.0, 0.0, 10.0, 2), (4.0, 0.0, 14.0, 2),
                (0.0, 4.0, 18.0, 2), (1.0, 1.0, float("nan"), 1)]
    if kind == "incomplete":
        return [(0.0, 0.0, 10.0, 2), (5001.0, 0.0, 20.0, 2),
                (5002.0, 0.0, 11.0, 2), (1.0, 1.0, 30.0, 1)]
    if kind == "unique-distance":
        return [(0.0, 0.0, 10.0, 2), (5.0, 0.0, 14.0, 2),
                (0.0, 3.0, 18.0, 2), (1.0, 0.5, 30.0, 1)]
    if kind == "all-ground":
        return [(0.0, 0.0, 10.0, 2), (5.0, 0.0, 14.0, 2),
                (1.0, 3.0, 18.0, 2)]
    return []


def write_fixture(path: Path, kind: str) -> None:
    if kind == "missing-classification":
        path.write_text("X,Y,Z\n0,0,10\n", encoding="utf-8")
        return
    lines = ["X,Y,Z,Classification\n"]
    for x, y, z, classification in rows(kind):
        lines.append(f"{x},{y},{z},{classification}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pdg-hag-delaunay-matrix-",
                                     dir=args.work_dir) as temporary:
        generated = Path(temporary)
        executed = 0
        for case in CASES:
            if args.require_cuda and not case.cuda:
                continue
            executed += 1
            fixture = generated / f"{case.fixture}.csv"
            write_fixture(fixture, case.fixture)
            pipeline = generated / f"{case.name}.json"
            pipeline.write_text(
                json.dumps({"pipeline": [
                    {"type": "readers.text", "filename": "input.csv"},
                    case.stage,
                    {"type": "writers.las", "filename": "out.las",
                     "extra_dims": "all"},
                ]}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")

            environment = os.environ.copy()
            if args.require_cuda:
                environment.update({"PDG_DISABLE_NATIVE": "1",
                                    "PDG_REQUIRE_HYBRID": "1"})
                environment.pop("PDG_DISABLE_CUDA_HYBRID", None)
                if case.cuda_required:
                    environment["PDG_REQUIRE_CUDA_HYBRID"] = "1"
                    environment.pop("PDG_EXPERIMENTAL_CUDA_HYBRID", None)
                else:
                    environment["PDG_EXPERIMENTAL_CUDA_HYBRID"] = "1"
                    environment.pop("PDG_REQUIRE_CUDA_HYBRID", None)
            elif case.hybrid:
                environment.update({
                    "PDG_DISABLE_NATIVE": "1",
                    "PDG_REQUIRE_HYBRID": "1",
                    "PDG_DISABLE_CUDA_HYBRID": "1",
                })
                environment.pop("PDG_REQUIRE_CUDA_HYBRID", None)
            else:
                for name in (
                    "PDG_DISABLE_NATIVE", "PDG_REQUIRE_HYBRID",
                    "PDG_DISABLE_CUDA_HYBRID", "PDG_REQUIRE_CUDA_HYBRID",
                    "PDG_EXPERIMENTAL_CUDA_HYBRID",
                ):
                    environment.pop(name, None)

            if case.backend == "grid":
                environment["PDG_FORCE_UNIFORM_GRID"] = "1"
                environment.pop("PDG_FORCE_MORTON_BVH", None)
            elif case.backend == "bvh":
                environment["PDG_FORCE_MORTON_BVH"] = "1"
                environment.pop("PDG_FORCE_UNIFORM_GRID", None)
            else:
                environment.pop("PDG_FORCE_UNIFORM_GRID", None)
                environment.pop("PDG_FORCE_MORTON_BVH", None)

            if case.proof == "tie":
                environment["PDG_REQUIRE_HAG_DELAUNAY_TIE_FALLBACK"] = "1"
            else:
                environment.pop("PDG_REQUIRE_HAG_DELAUNAY_TIE_FALLBACK", None)
            if case.proof == "insufficient":
                environment[
                    "PDG_REQUIRE_HAG_DELAUNAY_INSUFFICIENT_GROUND_FALLBACK"] = "1"
            else:
                environment.pop(
                    "PDG_REQUIRE_HAG_DELAUNAY_INSUFFICIENT_GROUND_FALLBACK", None)
            if case.proof == "nonfinite":
                environment["PDG_REQUIRE_HAG_DELAUNAY_NONFINITE_FALLBACK"] = "1"
            else:
                environment.pop("PDG_REQUIRE_HAG_DELAUNAY_NONFINITE_FALLBACK", None)

            if case.proof == "incomplete":
                environment["PDG_KNN_DEVICE_SHELL_BUDGET"] = "1"
                environment["PDG_REQUIRE_HAG_DELAUNAY_HOST_FALLBACK"] = "1"
            else:
                environment.pop("PDG_KNN_DEVICE_SHELL_BUDGET", None)
                environment.pop("PDG_REQUIRE_HAG_DELAUNAY_HOST_FALLBACK", None)

            candidate = args.candidate.resolve()
            if case.name == "no-ground-diagnostic":
                # The pinned stage deliberately trips a PointView assertion
                # after emitting its no-ground diagnostic. libc prefixes the
                # assertion with argv[0], so invoke the same candidate binary
                # through a `pdal` hard link and compare the substantive
                # stderr byte-for-byte without an executable-name artifact.
                # A hard link survives differential.py's canonicalization;
                # copy the companion engine link beside it because the public
                # launcher intentionally resolves that sibling from
                # /proc/self/exe.
                candidate_alias = generated / "candidate-argv0" / "pdal"
                candidate_alias.parent.mkdir(exist_ok=True)
                os.link(candidate, candidate_alias)
                os.link(candidate.with_name("pdg-engine"),
                        candidate_alias.with_name("pdg-engine"))
                candidate = candidate_alias

            command = [
                sys.executable, str(args.differential.resolve()),
                "--oracle", str(args.oracle.resolve()),
                "--candidate", str(candidate),
                "--case", f"hag-delaunay-matrix-{case.name}",
                "--work-dir", str(args.work_dir.resolve()),
                "--frozen-time-library",
                str(args.frozen_time_library.resolve()),
                "--seed-file", f"input.csv={fixture.resolve()}",
                "--seed-file", f"pipeline.json={pipeline.resolve()}",
            ]
            for preload in args.oracle_preload:
                command.extend(("--oracle-preload", str(preload.resolve())))
            for preload in args.candidate_preload:
                command.extend(("--candidate-preload",
                                str(preload.resolve())))
            command.extend(("--", "pipeline", "pipeline.json"))
            if case.verbose:
                command.extend(("--verbose", "8"))
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode:
                return completed.returncode

    print(f"exact hag_delaunay process matrix: {executed} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
