#!/usr/bin/env bash
# Re-prove the exact frozen 1.7x shipped-profile release artifact.
#
# This timing lane contains no configure, build, install, profile conversion,
# or payload write.  It verifies the closed artifact before and after, runs the
# fourteen reference workflows with three measured pairs, and runs every named
# large-cloud case with at least three measured pairs.
#
# Usage:
#   vast_release_reproof.sh PAYLOAD MANIFEST FIXTURES RESULTS [LARGE_LAZ]
set -euo pipefail

PAYLOAD=${1:?closed artifact payload}
MANIFEST=${2:?frozen artifact manifest}
FIXTURES=${3:?reference fixture directory}
RESULTS=${4:?results directory}
LARGE_LAZ=${5:-}

PAYLOAD=$(realpath "$PAYLOAD")
MANIFEST=$(realpath "$MANIFEST")
FIXTURES=$(realpath "$FIXTURES")
if [ -d "$RESULTS" ] && [ -n "$(find "$RESULTS" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "results directory must be absent or empty" >&2
  exit 2
fi
mkdir -p "$RESULTS"
RESULTS=$(realpath "$RESULTS")
TOOLS=$PAYLOAD/evidence/scripts/pdg
CANDIDATE_REL=$(jq -er '.roles.candidate.path' "$MANIFEST")
ORACLE_REL=$(jq -er '.roles.oracle.path' "$MANIFEST")
CANDIDATE_SHA=$(jq -er '.roles.candidate.sha256' "$MANIFEST")
ORACLE_SHA=$(jq -er '.roles.oracle.sha256' "$MANIFEST")
PDG=$PAYLOAD/$CANDIDATE_REL
ORACLE=$PAYLOAD/$ORACLE_REL
FROZEN=$PAYLOAD/evidence/libpdg_frozen_time.so
CHECK=$TOOLS/artifact_manifest.py
CLEAN_CONFIG=$RESULTS/clean-config
mkdir -p "$CLEAN_CONFIG"
export XDG_CONFIG_HOME=$CLEAN_CONFIG

python3 "$CHECK" check --manifest "$MANIFEST" --root "$PAYLOAD" \
  --report "$RESULTS/artifact-before.json"

env -i PATH="$PATH" LC_ALL=C TZ=UTC XDG_CONFIG_HOME="$CLEAN_CONFIG" \
  "$PDG" calibrate --status > "$RESULTS/placement-status.txt" 2>&1 || true
if ! grep -Eq 'active_profile_tier: shipped' \
    "$RESULTS/placement-status.txt"; then
  echo "the frozen shipped GPU-class profile is not active" >&2
  exit 3
fi
ACTIVE_PROFILE=$(sed -n 's/^active_profile: //p' "$RESULTS/placement-status.txt")
ACTIVE_PROFILE_SOURCE=$(sed -n 's/^active_profile_source: //p' "$RESULTS/placement-status.txt")
ACTIVE_PROFILE_SHA=$(sed -n 's/^active_profile_sha256: //p' "$RESULTS/placement-status.txt")
if [ -z "$ACTIVE_PROFILE" ] || [ -z "$ACTIVE_PROFILE_SOURCE" ] || \
   [ -z "$ACTIVE_PROFILE_SHA" ] || \
   [ "$(basename "$ACTIVE_PROFILE_SOURCE")" != "$ACTIVE_PROFILE_SOURCE" ]; then
  echo "active shipped profile identity is incomplete or unsafe" >&2
  exit 3
fi
ACTIVE_PROFILE_FILE=$PAYLOAD/evidence/data/placement-profiles/$ACTIVE_PROFILE_SOURCE
if [ ! -f "$ACTIVE_PROFILE_FILE" ] || \
   [ "$(sha256sum "$ACTIVE_PROFILE_FILE" | awk '{print $1}')" != "$ACTIVE_PROFILE_SHA" ]; then
  echo "active shipped profile is not bound to the frozen profile bytes" >&2
  exit 3
fi
jq -e --arg id "$ACTIVE_PROFILE" '
  .id == $id and .tier == "shipped" and (.summaries | length > 0) and
  all(.summaries[]; .shipping_margin == 1.7)
' "$ACTIVE_PROFILE_FILE" >/dev/null || {
  echo "active frozen profile is not the declared 1.7x shipped profile" >&2
  exit 3
}

# The status output proves which profile was selected; the frozen profile
# payload proves its admission rule.  Validate every included profile so a
# stale 3x shipped file cannot hide behind a correctly named active tier.
SHIPPED_PROFILES=0
for profile in "$PAYLOAD"/evidence/data/placement-profiles/*.json; do
  tier=$(jq -r '.tier // ""' "$profile")
  case "$tier" in
    shipped)
      jq -e '(.summaries | length > 0) and
        all(.summaries[]; .shipping_margin == 1.7)' "$profile" >/dev/null || {
        echo "shipped profile is not uniformly frozen at 1.7x: $profile" >&2
        exit 3
      }
      SHIPPED_PROFILES=$((SHIPPED_PROFILES + 1))
      ;;
    generic)
      jq -e '(.summaries | length > 0) and
        all(.summaries[]; .shipping_margin == 3.0)' "$profile" >/dev/null || {
        echo "generic safety profile does not retain the 3x admission rule: $profile" >&2
        exit 3
      }
      ;;
    *)
      echo "unrecognized placement-profile tier in $profile" >&2
      exit 3
      ;;
  esac
done
if [ "$SHIPPED_PROFILES" -lt 10 ]; then
  echo "expected at least ten frozen 1.7x shipped GPU-class profiles" >&2
  exit 3
fi

{
  date -u
  hostname
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  lscpu
  free -h
  sha256sum "$PDG" "$ORACLE" "$MANIFEST"
  cat "$RESULTS/placement-status.txt"
} > "$RESULTS/machine.txt" 2>&1

# Build a writable shadow containing only fixture paths and checked-in
# pipelines.  The closed artifact remains read-only and unchanged.
SHADOW=$RESULTS/shadow-repo
mkdir -p "$SHADOW/bench/pipelines" "$SHADOW/build/bench-data/reference"
cp -a "$PAYLOAD/evidence/bench/pipelines/reference" "$SHADOW/bench/pipelines/"
cp -a "$FIXTURES/." "$SHADOW/build/bench-data/reference/"

python3 "$TOOLS/reference_suite.py" run \
  --manifest "$SHADOW/bench/pipelines/reference/manifest.json" \
  --repo-root "$SHADOW" --benchmark-runner "$TOOLS/benchmark_reference.py" \
  --oracle "$ORACLE" --candidate "$PDG" \
  --work-dir "$RESULTS/suite-work" --reports "$RESULTS/suite-reports" \
  --aggregate-output "$RESULTS/suite-1.7x-frozen.json" \
  --cache-state warm --runs 3 --warmups 1 \
  --candidate-env "PDG_ORACLE_PDAL=$ORACLE" \
  --frozen-time-library "$FROZEN" --freeze-epoch 1704067200

jq -e --arg candidate "$CANDIDATE_SHA" --arg oracle "$ORACLE_SHA" '
  .schema == "pdg-reference-suite-aggregate-v1" and
  .complete == true and .contract == "exact" and
  .workload_count == 14 and
  .binaries.candidate.sha256 == $candidate and
  .binaries.oracle.sha256 == $oracle and
  (.missing | length == 0) and (.inexact | length == 0) and
  (.invalid | length == 0) and
  (.workloads | all(.median_speedup > 0))
' "$RESULTS/suite-1.7x-frozen.json" >/dev/null || {
  echo "frozen 1.7x reference-suite evidence is incomplete or unbound" >&2
  exit 4
}

if [ -n "$LARGE_LAZ" ]; then
  LARGE_LAZ=$(realpath "$LARGE_LAZ")
  python3 "$TOOLS/benchmark_reference.py" \
    --oracle "$ORACLE" --candidate "$PDG" \
    --pipeline "$SHADOW/bench/pipelines/reference/r6-features.json" \
    --fixture-laz "$LARGE_LAZ" --fixture "input.laz=$LARGE_LAZ" \
    --label large-r6-1.7x-frozen --work-dir "$RESULTS/large-work" \
    --report "$RESULTS/large-r6-1.7x-frozen.json" \
    --cache-state warm --runs 3 --warmups 1 \
    --candidate-env "PDG_ORACLE_PDAL=$ORACLE" \
    --frozen-time-library "$FROZEN" --freeze-epoch 1704067200
  jq -e --arg candidate "$CANDIDATE_SHA" --arg oracle "$ORACLE_SHA" '
    .schema == "pdg-reference-pipeline-baseline-v1" and
    .comparison.contract == "exact" and
    .comparison.exact_outputs == true and
    .comparison.median_speedup > 0 and
    .binaries.candidate.sha256 == $candidate and
    .binaries.oracle.sha256 == $oracle
  ' "$RESULTS/large-r6-1.7x-frozen.json" >/dev/null || {
    echo "large-r6 result is incomplete, inexact, or unbound" >&2
    exit 4
  }
fi

python3 "$CHECK" check --manifest "$MANIFEST" --root "$PAYLOAD" \
  --report "$RESULTS/artifact-after.json"
echo "RELEASE-REPROOF-COMPLETE $RESULTS"
