#!/usr/bin/env python3
"""Generate a deterministic SPDX 2.3 file inventory for a GPUPDAL bundle."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re


EXCLUDED = {"SBOM.spdx.json", "SHA256SUMS"}


def digest(path: Path, algorithm: str) -> str:
    checksum = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def spdx_file_id(relative: str, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9.-]", "-", relative).strip("-") or "file"
    candidate = f"SPDXRef-File-{stem}"
    if candidate in used:
        suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
        candidate = f"{candidate}-{suffix}"
    used.add(candidate)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--created-epoch", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.resolve() != output
        and path.relative_to(root).as_posix() not in EXCLUDED
    )

    used_ids: set[str] = set()
    files = []
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": "SPDXRef-Package-GPUPDAL",
        }
    ]
    packages = []
    dependency_map = root / "RUNTIME_DEPENDENCIES.tsv"
    if dependency_map.is_file():
        with dependency_map.open(newline="", encoding="utf-8") as source:
            rows = csv.DictReader(source, delimiter="\t")
            dependencies = sorted(
                {
                    (
                        row["package"],
                        row["version"],
                        row["license_declared"],
                    )
                    for row in rows
                    if row["package"] not in {"GPUPDAL", "unmanaged"}
                }
            )
        for package_name, package_version, license_declared in dependencies:
            package_id = spdx_file_id(
                f"RuntimePackage-{package_name}-{package_version}", used_ids
            ).replace("SPDXRef-File-", "SPDXRef-")
            packages.append(
                {
                    "name": package_name,
                    "SPDXID": package_id,
                    "versionInfo": package_version,
                    "downloadLocation": "NOASSERTION",
                    "filesAnalyzed": False,
                    "licenseConcluded": "NOASSERTION",
                    "licenseDeclared": "NOASSERTION",
                    "copyrightText": "NOASSERTION",
                    "summary": (
                        "Build package manager license declaration: "
                        f"{license_declared}"
                    ),
                }
            )
            relationships.append(
                {
                    "spdxElementId": "SPDXRef-Package-GPUPDAL",
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": package_id,
                }
            )
    verification_hashes = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        identifier = spdx_file_id(relative, used_ids)
        sha1 = digest(path, "sha1")
        sha256 = digest(path, "sha256")
        verification_hashes.append(sha1)
        files.append(
            {
                "fileName": f"./{relative}",
                "SPDXID": identifier,
                "checksums": [
                    {"algorithm": "SHA1", "checksumValue": sha1},
                    {"algorithm": "SHA256", "checksumValue": sha256},
                ],
                "licenseConcluded": "NOASSERTION",
                "licenseInfoInFiles": ["NOASSERTION"],
                "copyrightText": "NOASSERTION",
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-Package-GPUPDAL",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": identifier,
            }
        )

    created = dt.datetime.fromtimestamp(
        args.created_epoch, tz=dt.timezone.utc
    ).replace(microsecond=0)
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"gpupdal-{args.version}-linux-x64",
        "documentNamespace": (
            "https://github.com/zymazza/GPUPDAL/"
            f"spdx/{args.commit}/gpupdal-{args.version}-linux-x64"
        ),
        "creationInfo": {
            "created": created.isoformat().replace("+00:00", "Z"),
            "creators": ["Tool: GPUPDAL-generate-spdx-sbom/1"],
        },
        "packages": [
            {
                "name": "GPUPDAL",
                "SPDXID": "SPDXRef-Package-GPUPDAL",
                "versionInfo": args.version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": True,
                "packageVerificationCode": {
                    "packageVerificationCodeValue": hashlib.sha1(
                        "".join(sorted(verification_hashes)).encode("ascii")
                    ).hexdigest()
                },
                "licenseConcluded": "BSD-3-Clause",
                "licenseDeclared": "BSD-3-Clause",
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:github/zymazza/GPUPDAL@{args.commit}",
                    }
                ],
            },
            *packages,
        ],
        "files": files,
        "relationships": relationships,
    }
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
