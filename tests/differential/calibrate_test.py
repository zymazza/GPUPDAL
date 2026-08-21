#!/usr/bin/env python3
"""Process proof of `gpupal calibrate` and the local placement profile (D0277).

1. `gpupal calibrate --status` and `--dry-run` work and write nothing.
2. On the reference machine, `gpupal calibrate` without --force declines
   ("embedded reference profile applies") and writes nothing.
3. With the embedded profile hidden by the test hook, a quick calibration
   (two models, 100K/250K points, one pair each) writes a profile keyed to
   this machine whose admitted models are byte-exact device wins.
4. That profile makes the automatic `pipeline` route select the resident
   device path for a normal/covariance graph at 250K (proof gate succeeds),
   the output is byte-identical to the pinned oracle (when available) and to
   the host run without the profile; without the profile the same gate fails
   closed (exit 124), and so does a profile whose machine key differs; the
   calibration placement override never widens the automatic route.
5. (D0278/D0281) A file whose extra-bytes VLR declares a standard dimension
   name with another type stays exact but is not automatically admitted by a
   profile calibrated on the standard-width source, and the calibration
   override cannot widen that shape. A graph whose device stage computes on a
   mismatched file type also fails resident preflight to host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=pathlib.Path,
                        help="pdg launcher")
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--pinned-oracle", type=pathlib.Path)
    parser.add_argument("--make-shipped", type=pathlib.Path,
                        help="bench/report/make_shipped_profile.py (D0279 tiers)")
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix="pdg-calibrate-", dir=args.work_dir))
    base_env = {k: v for k, v in os.environ.items()
                if not k.startswith(("PDG_", "PDAL_TEST_"))}
    base_env.update({"LC_ALL": "C", "TZ": "UTC"})
    profile = root / "profile.json"

    def run(argv, env=None, check=True, timeout=1200):
        merged = dict(base_env)
        merged.update(env or {})
        completed = subprocess.run([str(a) for a in argv], env=merged,
                                   capture_output=True, text=True,
                                   timeout=timeout)
        if check and completed.returncode != 0:
            raise SystemExit(
                f"command failed ({completed.returncode}): {argv}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
        return completed

    # CUDA present?  The test is registered on the CUDA lane only, but fail
    # loudly rather than silently if the device is unavailable.
    doctor = run([args.candidate, "doctor"], check=False)
    if doctor.returncode != 0:
        raise SystemExit(f"gpupal doctor failed:\n{doctor.stdout}{doctor.stderr}")

    # 1. status / dry-run write nothing.
    status = run([args.candidate, "calibrate", "--status"],
                 env={"PDG_PROFILE_PATH": str(profile)})
    for key in ("device_name:", "compute_capability:", "driver:", "cuda:",
                "cpu_model:", "logical_cpus:", "local_profile: not-found",
                "active_profile:"):
        if key not in status.stdout:
            raise SystemExit(f"status lacks {key!r}:\n{status.stdout}")
    dry = run([args.candidate, "calibrate", "--dry-run", "--force", "--quick"],
              env={"PDG_PROFILE_PATH": str(profile)})
    if "normal-covariancefeatures-compose" not in dry.stdout or profile.exists():
        raise SystemExit(f"dry-run misbehaved:\n{dry.stdout}")

    # 2. On the reference machine, calibrate declines without --force.
    reference = "sm89-2026-08-17-r6-large-layouts" in status.stdout
    if reference:
        declined = run([args.candidate, "calibrate", "--quick"],
                       env={"PDG_PROFILE_PATH": str(profile)})
        if "nothing to calibrate" not in declined.stdout or profile.exists():
            raise SystemExit(f"calibrate did not decline:\n{declined.stdout}")

    # 3. Quick calibration with the embedded profile hidden.
    # The embedded reference profile and the shipped/generic tiers are hidden
    # for the local-profile steps so the host baseline is clean; the tier
    # steps below use a directory stand-in for the shipped table.
    hide = {"PDG_PROFILE_PATH": str(profile),
            "PDG_TEST_IGNORE_BUILTIN_PLACEMENT_PROFILE": "1",
            "PDG_DISABLE_SHIPPED_PROFILES": "1"}
    work = root / "calibrate-work"
    calibrated = run([args.candidate, "calibrate", "--points", "100000,250000",
                      "--repeats", "1", "--models",
                      "normal-covariancefeatures-compose,lof", "--work", work,
                      "--keep-work", "--quiet"], env=hide, timeout=1800)
    if not profile.exists():
        raise SystemExit(f"no profile written:\n{calibrated.stdout}\n"
                         f"{calibrated.stderr}")
    document = json.loads(profile.read_text())
    if document["schema"] != "pdg-local-placement-profile-v1":
        raise SystemExit("unexpected profile schema")
    models = document["stage_models"]
    for case in document["evidence"]["cases"]:
        if case["device_used"] and not case["byte_exact"]:
            raise SystemExit(f"host/device outputs differed: {case}")
        if not case["device_used"]:
            raise SystemExit(f"device path unavailable for a case: {case}")
    if "normal-covariancefeatures-compose" not in models:
        raise SystemExit(
            f"normal/covariance was not admitted (device did not win?):\n"
            f"{calibrated.stdout}")
    status = run([args.candidate, "calibrate", "--status"], env=hide)
    if "local_profile: applied" not in status.stdout:
        raise SystemExit(f"profile not applied by status:\n{status.stdout}")

    # 4. The automatic route on a normal/covariance graph at 250K.
    fixture = work / "synthetic-250000.las"
    if not fixture.exists():
        raise SystemExit("calibration fixture missing")
    out_dir = root / "runs"
    out_dir.mkdir()

    def pipeline(name: str) -> pathlib.Path:
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps({"pipeline": [
            str(fixture),
            {"type": "filters.normal", "knn": 8},
            {"type": "filters.covariancefeatures", "knn": 8,
             "feature_set": "Dimensionality"},
            str(out_dir / f"{name}.las")]}))
        return path

    require = {"PDG_REQUIRE_AUTOMATIC_NORMAL_COVARIANCE_RESIDENT": "1"}
    with_profile = run([args.candidate, "pipeline", pipeline("with-profile")],
                       env={**hide, **require})
    del with_profile
    without = run([args.candidate, "pipeline", pipeline("without-profile")],
                  env={**hide, **require,
                       "PDG_PROFILE_PATH": str(root / "missing.json")},
                  check=False)
    if without.returncode != 124:
        raise SystemExit(
            f"without a profile the device gate should fail closed (124), "
            f"got {without.returncode}:\n{without.stderr}")
    # A profile keyed to another machine is ignored.
    other = json.loads(profile.read_text())
    other["machine"]["cpu_model"] = other["machine"]["cpu_model"] + " (other)"
    other_path = root / "other-machine.json"
    other_path.write_text(json.dumps(other))
    mismatch = run([args.candidate, "pipeline", pipeline("mismatch")],
                   env={**hide, **require, "PDG_PROFILE_PATH": str(other_path)},
                   check=False)
    if mismatch.returncode != 124:
        raise SystemExit(
            f"a mismatched profile must not enable the device route, got "
            f"{mismatch.returncode}")
    status = run([args.candidate, "calibrate", "--status"],
                 env={**hide, "PDG_PROFILE_PATH": str(other_path)})
    if "machine-mismatch" not in status.stdout:
        raise SystemExit(f"status did not report the mismatch:\n{status.stdout}")
    # The calibration override never widens the automatic route.
    override = run([args.candidate, "pipeline", pipeline("override")],
                   env={**hide, **require,
                        "PDG_CALIBRATION_FORCE_DEVICE_PLACEMENT": "1"},
                   check=False)
    if override.returncode != 124:
        raise SystemExit(
            f"the calibration override widened the automatic route "
            f"({override.returncode})")

    # Bytes: default with profile == host without profile == pinned oracle.
    run([args.candidate, "pipeline", pipeline("default-with-profile")], env=hide)
    run([args.candidate, "pipeline", pipeline("host-without-profile")],
        env={**hide, "PDG_PROFILE_PATH": str(root / "missing.json")})
    device_bytes = sha256(out_dir / "default-with-profile.las")
    host_bytes = sha256(out_dir / "host-without-profile.las")
    if device_bytes != host_bytes:
        raise SystemExit("device and host outputs differ")
    if args.pinned_oracle and args.pinned_oracle.exists():
        run([args.pinned_oracle, "pipeline", pipeline("oracle")])
        if sha256(out_dir / "oracle.las") != device_bytes:
            raise SystemExit("device output differs from the pinned oracle")

    # 5. D0278: a file whose extra-bytes VLR declares a standard dimension
    # name with another type (Deviation as uint16; PDAL's standard is double)
    # is outside the standard-width calibration shape. Automatic placement
    # must stay host-selected even when no device stage touches the dimension,
    # and the calibration override must not widen that automatic shape.
    eb_fixture = out_dir / "eb-typed.las"
    (out_dir / "make-eb.json").write_text(json.dumps({"pipeline": [
        str(fixture),
        {"type": "filters.ferry", "dimensions": "Intensity=>Deviation"},
        {"type": "writers.las", "filename": str(eb_fixture),
         "minor_version": 4, "dataformat_id": 7,
         "extra_dims": "Deviation=uint16"}]}))
    run([args.candidate, "pipeline", out_dir / "make-eb.json"],
        env={**hide, "PDG_PROFILE_PATH": str(root / "missing.json")})

    def eb_pipeline(name: str, assign: bool) -> pathlib.Path:
        path = out_dir / f"{name}.json"
        stages = [str(eb_fixture)]
        if assign:
            stages.append({"type": "filters.assign", "value": "Deviation = 5"})
        stages += [{"type": "filters.normal", "knn": 8},
                   {"type": "filters.covariancefeatures", "knn": 8,
                    "feature_set": "Dimensionality"},
                   str(out_dir / f"{name}.las")]
        path.write_text(json.dumps({"pipeline": stages}))
        return path

    eb_required = run(
        [args.candidate, "pipeline", eb_pipeline("eb-required", False)],
        env={**hide, **require}, check=False)
    if eb_required.returncode != 124 or (out_dir / "eb-required.las").exists():
        raise SystemExit("file-typed dimension widened automatic placement")
    run([args.candidate, "pipeline", eb_pipeline("eb-default", False)], env=hide)
    run([args.candidate, "pipeline", eb_pipeline("eb-host", False)],
        env={**hide, "PDG_PROFILE_PATH": str(root / "missing.json")})
    eb_default = sha256(out_dir / "eb-default.las")
    if eb_default != sha256(out_dir / "eb-host.las"):
        raise SystemExit("file-typed dimension: default and host outputs differ")
    if args.pinned_oracle and args.pinned_oracle.exists():
        run([args.pinned_oracle, "pipeline", eb_pipeline("eb-oracle", False)])
        if sha256(out_dir / "eb-oracle.las") != eb_default:
            raise SystemExit("file-typed dimension: default output differs from "
                             "the pinned oracle")
    # The explicit resident command under the calibration override places
    # every stage on the device; a stage that computes on the file-typed
    # dimension must be refused by the resident preflight with a clear
    # reason and complete on the host, byte-identical to the oracle.
    # A file that already carries "Rank" as uint16 (PDAL/PDG standard: uint8);
    # filters.estimaterank writes Rank on the device, so PDAL's layout keeps
    # the wider file type and the resident preflight must refuse.
    rank_fixture = out_dir / "rank-typed.las"
    (out_dir / "make-rank.json").write_text(json.dumps({"pipeline": [
        str(fixture),
        {"type": "filters.ferry", "dimensions": "Intensity=>Rank"},
        {"type": "writers.las", "filename": str(rank_fixture),
         "minor_version": 4, "dataformat_id": 7,
         "extra_dims": "Rank=uint16"}]}))
    run([args.candidate, "pipeline", out_dir / "make-rank.json"],
        env={**hide, "PDG_PROFILE_PATH": str(root / "missing.json")})

    def assign_pipeline(name: str) -> pathlib.Path:
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps({"pipeline": [
            str(rank_fixture),
            {"type": "filters.estimaterank", "knn": 8},
            str(out_dir / f"{name}.las")]}))
        return path
    stats = out_dir / "eb-touched-stats.json"
    run([args.candidate, "resident", assign_pipeline("eb-touched"),
         "--stats", stats],
        env={**hide, "PDG_CALIBRATION_FORCE_DEVICE_PLACEMENT": "1"})
    document = json.loads(stats.read_text())
    preflight = document["execution"].get("resident_preflight") or {}
    if document["placement"].get("choice") != "device":
        raise SystemExit(f"override did not place the assign stage on the "
                         f"device: {document['placement']}")
    if preflight.get("accepted") or "device-computed dimension: Rank" \
            not in str(preflight.get("reason", "")):
        raise SystemExit(f"touched file-typed dimension was not refused with "
                         f"the expected reason: {preflight}")
    if document["execution"].get("executor") != "pdal_standard_host":
        raise SystemExit(f"touched file-typed dimension did not fall back to "
                         f"the host: {document['execution'].get('executor')}")
    if args.pinned_oracle and args.pinned_oracle.exists():
        run([args.pinned_oracle, "pipeline", assign_pipeline("eb-touched-oracle")])
        if sha256(out_dir / "eb-touched-oracle.las") != \
                sha256(out_dir / "eb-touched.las"):
            raise SystemExit("touched file-typed dimension: host fallback "
                             "output differs from the pinned oracle")
    # The calibration placement override also cannot turn an unmeasured
    # physical layout into an automatic/resident performance claim.
    stats = out_dir / "eb-device-run.stats.json"
    run([args.candidate, "resident", eb_pipeline("eb-device-run", False),
         "--stats", stats],
        env={**hide, "PDG_CALIBRATION_FORCE_DEVICE_PLACEMENT": "1"})
    document = json.loads(stats.read_text())
    if document["placement"].get("available") is not False or \
            document["placement"].get("unavailable_reason") != \
            "mixed_calibration_models" or \
            document["execution"].get("executor") != "pdal_standard_host":
        raise SystemExit(f"file-typed dimension: calibration override widened "
                         f"the layout: {document['placement']} "
                         f"{document['execution'].get('executor')}")
    if sha256(out_dir / "eb-device-run.las") != eb_default:
        raise SystemExit("file-typed dimension: explicit resident output differs")
    # 6. D0279 tiers: a shipped GPU-class profile (embedded table stand-in via
    # PDG_TEST_SHIPPED_PROFILE_DIR) enables the device route with no local
    # profile; a generic profile applies inside its bounds and not outside;
    # a local profile takes precedence over both.
    if args.make_shipped and args.make_shipped.exists():
        shipped_dir = root / "shipped"
        shipped_dir.mkdir()
        run([sys.executable, args.make_shipped, profile, "--slug", "test-gpu",
             "--margin", "1.2", "--out", shipped_dir / "test-gpu.json"])
        shipped_doc = json.loads((shipped_dir / "test-gpu.json").read_text())
        if "normal-covariancefeatures-compose" not in shipped_doc["stage_models"]:
            raise SystemExit("shipped conversion dropped the compose model")
        tiers = {"PDG_TEST_IGNORE_BUILTIN_PLACEMENT_PROFILE": "1",
                 "PDG_TEST_SHIPPED_PROFILE_DIR": str(shipped_dir),
                 "PDG_TEST_SHIPPED_PROFILE_DIR_ONLY": "1",
                 "PDG_PROFILE_PATH": str(root / "missing.json")}
        status = run([args.candidate, "calibrate", "--status"], env=tiers)
        if "active_profile_tier: shipped" not in status.stdout:
            raise SystemExit(f"shipped tier not active:\n{status.stdout}")
        run([args.candidate, "pipeline", pipeline("shipped-device")],
            env={**tiers, **require})
        if sha256(out_dir / "shipped-device.las") != device_bytes:
            raise SystemExit("shipped-tier output differs")
        # A local profile wins over the shipped one.
        status = run([args.candidate, "calibrate", "--status"],
                     env={**tiers, "PDG_PROFILE_PATH": str(profile)})
        if "active_profile_tier: local" not in status.stdout:
            raise SystemExit(f"local tier should take precedence:\n{status.stdout}")
        # Generic tier: derived from a shipped profile keyed to another GPU so
        # the shipped match fails and only the generic applies.
        other_dir = root / "generic"
        other_dir.mkdir()
        other = dict(shipped_doc)
        other["machine"] = dict(shipped_doc["machine"], device_name="Some Other GPU")
        other["id"] = "shipped-other"
        (other_dir / "other.json").write_text(json.dumps(other))
        run([sys.executable, args.make_shipped, "--generic", other_dir / "other.json",
             "--margin", "1.2", "--out", other_dir / "generic.json"])
        generic_doc = json.loads((other_dir / "generic.json").read_text())
        # Bound the generic to this device's capability so it applies here.
        cc = shipped_doc["machine"]["compute_capability"]
        generic_doc["applies"]["minimum_compute_capability"] = cc
        (other_dir / "generic.json").write_text(json.dumps(generic_doc))
        (other_dir / "other.json").unlink()
        gen_env = {**tiers, "PDG_TEST_SHIPPED_PROFILE_DIR": str(other_dir)}
        status = run([args.candidate, "calibrate", "--status"], env=gen_env)
        if "active_profile_tier: generic" not in status.stdout:
            raise SystemExit(f"generic tier not active:\n{status.stdout}")
        run([args.candidate, "pipeline", pipeline("generic-device")],
            env={**gen_env, **require})
        if sha256(out_dir / "generic-device.las") != device_bytes:
            raise SystemExit("generic-tier output differs")
        # Outside its bounds the generic profile is inert (host path).
        generic_doc["applies"]["minimum_compute_capability"] = "99.0"
        (other_dir / "generic.json").write_text(json.dumps(generic_doc))
        outside = run([args.candidate, "pipeline", pipeline("generic-outside")],
                      env={**gen_env, **require}, check=False)
        if outside.returncode != 124:
            raise SystemExit(f"generic profile applied outside its bounds "
                             f"({outside.returncode})")
    print("calibrate process proof passed:", root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
