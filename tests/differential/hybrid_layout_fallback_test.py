#!/usr/bin/env python3
"""Prove a layout-changing hybrid rewrite delegates before side effects."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CANDIDATE, INPUT, PIPELINE = map(Path, sys.argv[1:4])


with tempfile.TemporaryDirectory(prefix="pdg-hybrid-layout-fallback-") as temp:
    root = Path(temp)
    shutil.copy2(INPUT, root / "input.las")
    shutil.copy2(PIPELINE, root / "pipeline.json")

    oracle = root / "oracle"
    oracle.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path('oracle.args').write_text('\\n'.join(sys.argv[1:]), "
        "encoding='utf-8')\n"
        "raise SystemExit(73)\n",
        encoding="utf-8",
    )
    oracle.chmod(0o755)

    env = os.environ.copy()
    for name in (
        "PDG_REQUIRE_CUDA_HYBRID",
        "PDG_REQUIRE_CUDA_POINT_PROGRAM",
        "PDG_REQUIRE_CUDA_TRANSLATE",
        "PDG_REQUIRE_HYBRID",
        "PDG_REQUIRE_NATIVE",
        "PDG_REQUIRE_STREAMING_HYBRID",
    ):
        env.pop(name, None)
    env.update(
        {
            "LC_ALL": "C",
            "PDG_DEBUG_HYBRID": "1",
            "PDG_DISABLE_CUDA_HYBRID": "1",
            "PDG_DISABLE_NATIVE": "1",
            "PDG_EXPERIMENTAL_CUDA_HYBRID": "1",
            "PDG_ORACLE_PDAL": str(oracle),
            "TZ": "UTC",
        }
    )

    result = subprocess.run(
        [str(CANDIDATE.resolve()), "pipeline", "pipeline.json"],
        cwd=root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert result.returncode == 73, result.stderr
    assert result.stdout == "", result.stdout
    assert (
        "gpupdal: rewritten hybrid layout differs from the original pipeline\n"
        in result.stderr
    ), result.stderr
    assert (root / "oracle.args").read_text(encoding="utf-8").splitlines() == [
        "pipeline",
        "pipeline.json",
    ]
    assert not (root / "out.las").exists()
