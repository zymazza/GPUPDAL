#!/usr/bin/env python3
"""Fail-closed contract for frozen release artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile


def run(command: list[str], expected: int = 0) -> None:
    completed = subprocess.run(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, check=False)
    assert completed.returncode == expected, (
        completed.returncode, completed.stdout, completed.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pdg-artifact-contract-") as text:
        root = pathlib.Path(text)
        payload = root / "payload"
        (payload / "bin").mkdir(parents=True)
        candidate = payload / "bin" / "pdg"
        oracle = payload / "bin" / "pdal"
        for path, name in ((candidate, "pdg"), (oracle, "pdal")):
            path.write_text(f"#!/bin/sh\necho {name}-test\n")
            path.chmod(0o755)
        (payload / "profiles.json").write_text("{}\n")
        manifest = root / "manifest.json"
        run([sys.executable, str(args.tool), "create", "--root", str(payload),
             "--candidate", "bin/pdg", "--oracle", "bin/pdal",
             "--source-commit", "0123456789abcdef", "--release-id", "test",
             "--output", str(manifest)])
        run([sys.executable, str(args.tool), "check", "--manifest", str(manifest),
             "--root", str(payload)])
        forged = json.loads(manifest.read_text())
        forged["roles"]["candidate"]["path"] = "../outside-pdg"
        forged_manifest = root / "forged-manifest.json"
        forged_manifest.write_text(json.dumps(forged))
        run([sys.executable, str(args.tool), "check", "--manifest",
             str(forged_manifest), "--root", str(payload)], 2)
        candidate.write_text("#!/bin/sh\necho changed\n")
        run([sys.executable, str(args.tool), "check", "--manifest", str(manifest),
             "--root", str(payload)], 1)
        symlink_payload = root / "symlink-payload"
        (symlink_payload / "bin").mkdir(parents=True)
        outside = root / "outside-pdg"
        outside.write_text("#!/bin/sh\necho outside\n")
        outside.chmod(0o755)
        (symlink_payload / "bin" / "pdg").symlink_to(outside)
        (symlink_payload / "bin" / "pdal").write_text(
            "#!/bin/sh\necho pdal\n")
        (symlink_payload / "bin" / "pdal").chmod(0o755)
        run([sys.executable, str(args.tool), "create", "--root",
             str(symlink_payload), "--candidate", "bin/pdg", "--oracle",
             "bin/pdal", "--source-commit", "0123456789abcdef",
             "--release-id", "symlink-test", "--output",
             str(root / "symlink-manifest.json")], 2)
        candidate.write_text("#!/bin/sh\necho pdg-test\n")
        (payload / "unexpected").write_text("new\n")
        run([sys.executable, str(args.tool), "check", "--manifest", str(manifest),
             "--root", str(payload)], 1)


if __name__ == "__main__":
    main()
