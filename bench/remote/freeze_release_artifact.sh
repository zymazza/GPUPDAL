#!/usr/bin/env bash
# Close an already assembled, runnable release payload for evidence collection.
# This script never builds or installs.  The packaging lane must place both the
# candidate and the independently built pinned oracle below PAYLOAD first.
#
# Usage:
#   freeze_release_artifact.sh PAYLOAD CANDIDATE_REL ORACLE_REL MANIFEST RELEASE_ID
set -euo pipefail

PAYLOAD=${1:?release payload directory}
CANDIDATE_REL=${2:?candidate path relative to payload}
ORACLE_REL=${3:?oracle path relative to payload}
MANIFEST=${4:?manifest output outside payload}
RELEASE_ID=${5:?release id}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR=${PDG_BUILD_DIR:-$ROOT/build/pdg-cuda-release}
PAYLOAD=$(realpath "$PAYLOAD")
MANIFEST_PARENT=$(dirname "$MANIFEST")
mkdir -p "$MANIFEST_PARENT"

if [ ! -x "$PAYLOAD/$CANDIDATE_REL" ] || [ ! -x "$PAYLOAD/$ORACLE_REL" ]; then
  echo "candidate and oracle must already be runnable below the payload" >&2
  exit 2
fi
if [[ $(realpath -m "$MANIFEST") == "$PAYLOAD"/* ]]; then
  echo "manifest must live outside the closed payload" >&2
  exit 2
fi

# Evidence code is part of the frozen release object.  Timed hosts execute
# these exact bytes and artifact_manifest.py rejects any later substitution.
EVIDENCE=$PAYLOAD/evidence
if [ -e "$EVIDENCE" ]; then
  echo "payload already contains evidence; assemble a fresh payload before freezing" >&2
  exit 2
fi
mkdir -p "$EVIDENCE/scripts/pdg" "$EVIDENCE/bench/pipelines" \
  "$EVIDENCE/bench/3dep" "$EVIDENCE/bench/remote" \
  "$EVIDENCE/data/placement-profiles" "$EVIDENCE/tests/conformance"
cp "$ROOT/scripts/pdg/artifact_manifest.py" \
  "$ROOT/scripts/pdg/benchmark_reference.py" \
  "$ROOT/scripts/pdg/conformance.py" \
  "$ROOT/scripts/pdg/copc_semantic.py" \
  "$ROOT/scripts/pdg/differential.py" \
  "$ROOT/scripts/pdg/prepare_reference_fixtures.py" \
  "$ROOT/scripts/pdg/reference_suite.py" \
  "$ROOT/scripts/pdg/verify.py" \
  "$EVIDENCE/scripts/pdg/"
cp "$ROOT/tests/conformance/bounded-recipes-v1.json" \
  "$EVIDENCE/tests/conformance/"
cp -a "$ROOT/bench/pipelines/reference" "$EVIDENCE/bench/pipelines/"
cp "$ROOT/bench/3dep/study.py" "$ROOT/bench/3dep/preregistration-v1.json" \
  "$EVIDENCE/bench/3dep/"
cp "$ROOT/bench/remote/vast_release_reproof.sh" \
  "$EVIDENCE/bench/remote/"
if [ -f "$ROOT/bench/remote/reproof-evidence-amendment-v1.json" ]; then
  cp "$ROOT/bench/remote/reproof-evidence-amendment-v1.json" \
    "$EVIDENCE/bench/remote/"
fi
cp -a "$ROOT/data/placement-profiles/." "$EVIDENCE/data/placement-profiles/"
if [ ! -f "$BUILD_DIR/lib/libpdg_frozen_time.so" ]; then
  echo "release frozen-time helper is missing; build the complete package before assembly" >&2
  exit 2
fi
cp "$BUILD_DIR/lib/libpdg_frozen_time.so" "$EVIDENCE/"
if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
  cp "$BUILD_DIR/CMakeCache.txt" "$EVIDENCE/"
fi
if [ -n "${PDG_ARTIFACT_METADATA:-}" ]; then
  cp "$PDG_ARTIFACT_METADATA" "$EVIDENCE/build-metadata.json"
fi

SOURCE_COMMIT=${PDG_SOURCE_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}
CREATE_ARGS=(
  --root "$PAYLOAD" --candidate "$CANDIDATE_REL" --oracle "$ORACLE_REL"
  --source-commit "$SOURCE_COMMIT" --release-id "$RELEASE_ID"
  --output "$MANIFEST"
)
if [ -f "$EVIDENCE/build-metadata.json" ]; then
  CREATE_ARGS+=(--metadata "$EVIDENCE/build-metadata.json")
fi
python3 "$EVIDENCE/scripts/pdg/artifact_manifest.py" create \
  "${CREATE_ARGS[@]}"
python3 "$EVIDENCE/scripts/pdg/artifact_manifest.py" check \
  --manifest "$MANIFEST" --root "$PAYLOAD"
echo "FROZEN-RELEASE-ARTIFACT $MANIFEST"
