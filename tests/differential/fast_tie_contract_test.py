#!/usr/bin/env python3
"""Process proof of the D0271 `--fast` tie-order contract on the CUDA lane.

On the deterministic-ties hybrid neighborhood pipeline (forced CUDA hybrid
route): (1) the default contract still performs the exact host tie repair
(the repair proof gate succeeds); (2) under `gpupal --fast` the same gate must
fail closed because no tie repair runs, and nothing is published; (3) under
`gpupal --fast` without the gate the run succeeds and its LAS records have the
same count, layout, and coordinates as the exact run, differing at most in
the tie rows' attributes; (4) the default contract without `--fast` is
unchanged (byte-identical to the exact run).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--reference-runner", required=True, type=pathlib.Path,
                        help="scripts/pdg/benchmark_reference.py (record "
                             "comparator)")
    parser.add_argument("--points", required=True, type=pathlib.Path)
    parser.add_argument("--pipeline", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.reference_runner.resolve().parent))
    import benchmark_reference  # noqa: E402

    args.work_dir.mkdir(parents=True, exist_ok=True)
    base_env = {k: v for k, v in os.environ.items()
                if not k.startswith(("PDG_", "PDAL_TEST_"))}
    base_env.update({
        "PDG_DISABLE_NATIVE": "1",
        "PDG_REQUIRE_HYBRID": "1",
        "PDG_REQUIRE_CUDA_HYBRID": "1",
        "PDG_REQUIRE_NEIGHBORHOOD_REUSE": "1",
        "PDG_REQUIRE_NEIGHBORHOOD_COLUMN_REUSE": "1",
        "PDG_FORCE_MORTON_BVH": "1",
    })

    def run(name: str, fast: bool, prove_repair: bool):
        root = pathlib.Path(tempfile.mkdtemp(prefix=f"pdg-fast-tie-{name}-",
                                             dir=args.work_dir))
        shutil.copy(args.points, root / "input.csv")
        shutil.copy(args.pipeline, root / "pipeline.json")
        env = dict(base_env)
        if prove_repair:
            env["PDG_REQUIRE_NEIGHBORHOOD_TIE_REPAIR"] = "1"
        command = [str(args.candidate.resolve())]
        if fast:
            command.append("--fast")
        command += ["pipeline", "pipeline.json"]
        completed = subprocess.run(command, cwd=root, env=env, text=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        return root, completed

    exact_root, exact = run("exact-proof", fast=False, prove_repair=True)
    if exact.returncode != 0:
        print(f"exact repair-proof run failed: {exact.stderr}",
              file=sys.stderr)
        return 1
    exact_block = benchmark_reference.las_record_block(
        exact_root / "out.las", None, None)
    if exact_block is None:
        print("exact run published no readable out.las", file=sys.stderr)
        return 1

    plain_root, plain = run("exact-plain", fast=False, prove_repair=False)
    if plain.returncode != 0 or (plain_root / "out.las").read_bytes() != \
            (exact_root / "out.las").read_bytes():
        print("default contract without the proof gate is not byte-identical",
              file=sys.stderr)
        return 1

    gated_root, gated = run("fast-proof", fast=True, prove_repair=True)
    if gated.returncode == 0:
        print("--fast with the tie-repair proof gate unexpectedly succeeded: "
              "the relaxed contract must not repair ties", file=sys.stderr)
        return 1
    if "required resident neighborhood tie repair did not occur" not in \
            gated.stderr:
        print(f"--fast proof failure lacks the named diagnostic: "
              f"{gated.stderr}", file=sys.stderr)
        return 1
    if (gated_root / "out.las").exists():
        print("--fast proof failure published out.las", file=sys.stderr)
        return 1

    fast_root, fast = run("fast", fast=True, prove_repair=False)
    if fast.returncode != 0:
        print(f"--fast run failed: {fast.stderr}", file=sys.stderr)
        return 1
    fast_block = benchmark_reference.las_record_block(
        fast_root / "out.las", None, None)
    if fast_block is None:
        print("--fast run published no readable out.las", file=sys.stderr)
        return 1
    comparison = benchmark_reference.las_record_compare(exact_block,
                                                        fast_block)
    if not comparison["las_records_layout_match"]:
        print(f"--fast changed the record layout or count: {comparison}",
              file=sys.stderr)
        return 1
    if comparison["las_records_xyz_differing"] != 0:
        print(f"--fast changed coordinates: {comparison}", file=sys.stderr)
        return 1
    print("fast tie contract: exact repair proven, --fast skips it fail-closed "
          f"under the gate, and {comparison['las_records_differing']} of "
          f"{comparison['las_records_compared']} records differ in "
          "attributes only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
