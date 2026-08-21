#!/usr/bin/env python3
"""Convert `pdg calibrate` profiles into shipped GPU-class profiles (D0279).

    make_shipped_profile.py PROFILE.json --slug rtx4090 --margin 1.7 \
        --out data/placement-profiles/rtx4090.json
    make_shipped_profile.py --generic data/placement-profiles/*.json \
        --margin 3 --memory-bytes ID=BYTES ... --out data/placement-profiles/generic-cuda.json

Margins of record (D0280): 1.7x for a shipped GPU-class profile, 3x for the
generic fallback (an unknown GPU may be slower than the slowest swept class,
so its margin also covers the device side).

A shipped profile keeps a local profile's coefficients and only the stage
models whose measured device win cleared the margin (device_seconds * margin
<= host_seconds) at every size that stays inside the envelope; the envelope
is bounded to the sizes that cleared it (smallest .. largest, contiguous from
the smallest clearing size upward). Prior measurements carried in the
profile's evidence chain (--append) are honoured, so a model measured at
250K/1M/4M/16M/48M is judged at all five sizes.

The generic profile takes the intersection of the shipped profiles' models
that clear the margin in every profile, the slowest device and
fastest host coefficient of each term across them, the envelope
intersection, and applicability bounds from the smallest compute capability
and device memory among the contributing profiles.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import statistics
import sys


def median(values):
    return statistics.median(values) if values else None


def collect_cases(evidence, out):
    """Walk the evidence chain: model -> {points: (host, device, exact)}."""
    if not isinstance(evidence, dict):
        return
    for case in evidence.get("cases", []) or []:
        if not case.get("device_used"):
            continue
        host = median(case.get("host_seconds") or [])
        device = median(case.get("device_seconds") or [])
        if host is None or device is None:
            continue
        model = case["model"]
        points = int(case["points"])
        # Newer measurements (outer evidence) win over older ones for the same
        # size; the outer level is visited first.
        out.setdefault(model, {})
        out[model].setdefault(points, (host, device, bool(case.get("byte_exact"))))
    collect_cases(evidence.get("previous_evidence"), out)


def clearing_envelope(cases: dict, margin: float):
    """Return (min_points, max_points, admitted sizes) or None."""
    sizes = sorted(cases)
    if not sizes:
        return None
    if any(not cases[p][2] for p in sizes):
        return None  # a non-exact pair anywhere disqualifies the model
    admitted = [p for p in sizes if cases[p][1] * margin <= cases[p][0]]
    if not admitted:
        return None
    lo = admitted[0]
    # Contiguous from the smallest clearing size upward: stop at the first
    # measured size above `lo` that does not clear the margin.
    hi = lo
    for p in sizes:
        if p < lo:
            continue
        if p in admitted:
            hi = p
        else:
            break
    return lo, hi, [p for p in admitted if lo <= p <= hi]


def shipped_from_local(doc: dict, slug: str, margin: float) -> dict:
    cases = {}
    collect_cases(doc.get("evidence"), cases)
    models = {}
    summaries = []
    dropped = []
    for name, model in doc["stage_models"].items():
        env = clearing_envelope(cases.get(name, {}), margin)
        if not env:
            dropped.append(name)
            continue
        lo, hi, admitted = env
        entry = dict(model)
        entry["minimum_device_points"] = lo
        entry["maximum_device_points"] = hi
        models[name] = entry
        top = max(admitted)
        summaries.append({
            "model": name, "minimum_device_points": lo, "maximum_device_points": hi,
            "host_seconds_at_maximum": cases[name][top][0],
            "device_seconds_at_maximum": cases[name][top][1],
            "byte_exact": True, "device_wins_somewhere": True,
            "shipping_margin": margin,
            "measured_speedups": {str(p): round(cases[name][p][0] / cases[name][p][1], 3)
                                  for p in sorted(cases[name])}})
    machine = dict(doc["machine"])
    return {
        "schema": doc["schema"],
        "id": f"shipped-{slug}-sm{machine['compute_capability'].replace('.', '')}-cuda{machine['cuda']}",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tier": "shipped",
        "machine": machine,
        "coefficients": doc["coefficients"],
        "stage_models": models,
        "summaries": summaries,
        "evidence": {
            "source_profile_id": doc["id"],
            "source_created_utc": doc.get("created_utc"),
            "shipping_margin": margin,
            "dropped_models": dropped,
            "measured_cases": {m: {str(p): {"host_seconds": v[0], "device_seconds": v[1],
                                            "byte_exact": v[2]}
                                   for p, v in sorted(cs.items())}
                               for m, cs in cases.items()},
            "source_evidence_summary": {
                "engine": (doc.get("evidence") or {}).get("engine"),
                "sizes": (doc.get("evidence") or {}).get("sizes"),
                "cgroup_cpu_max": (doc.get("evidence") or {}).get("cgroup_cpu_max"),
            },
        },
    }


def parse_cc(text: str):
    major, _, minor = str(text).partition(".")
    return int(major), int(minor or 0)


def generic_from_shipped(docs: list[dict], margin: float, memory_bytes: dict) -> dict:
    """Intersection of the shipped profiles under a stricter margin."""
    per_profile = []
    for doc in docs:
        cases = {}
        # Shipped profiles carry measured_cases directly.
        for model, sizes in (doc.get("evidence") or {}).get("measured_cases", {}).items():
            cases[model] = {int(p): (v["host_seconds"], v["device_seconds"], v["byte_exact"])
                            for p, v in sizes.items()}
        per_profile.append((doc, cases))
    names = set.intersection(*(set(d["stage_models"]) for d, _ in per_profile)) if per_profile else set()
    models = {}
    summaries = []
    contributing = []
    for name in sorted(names):
        envs = []
        for doc, cases in per_profile:
            env = clearing_envelope(cases.get(name, {}), margin)
            if not env:
                envs = []
                break
            envs.append((doc, env))
        if not envs:
            continue
        lo = max(e[1][0] for e in envs)
        hi = min(e[1][1] for e in envs)
        if lo > hi:
            continue
        # Conservative coefficients: slowest device, fastest host.
        sm = [d["stage_models"][name] for d, _ in envs]
        entry = {
            "host_fixed_ns": min(m["host_fixed_ns"] for m in sm),
            "host_ns_per_point": min(m["host_ns_per_point"] for m in sm),
            "device_fixed_ns": max(m["device_fixed_ns"] for m in sm),
            "device_ns_per_point": max(m["device_ns_per_point"] for m in sm),
            "minimum_device_points": lo,
            "maximum_device_points": hi,
        }
        models[name] = entry
        contributing.append(name)
        summaries.append({"model": name, "minimum_device_points": lo,
                          "maximum_device_points": hi,
                          "host_seconds_at_maximum": 0.0, "device_seconds_at_maximum": 0.0,
                          "byte_exact": True, "device_wins_somewhere": True,
                          "shipping_margin": margin,
                          "from_profiles": [d["id"] for d, _ in envs]})
    if not models:
        raise SystemExit("no model clears the generic margin in every shipped profile")
    coeff_keys = ["cuda_startup_ns", "host_to_device_ns_per_byte", "device_to_host_ns_per_byte",
                  "packing_ns_per_byte", "index_build_ns_per_byte", "synchronization_ns"]
    coefficients = {k: max(d["coefficients"][k] for d, _ in per_profile) for k in coeff_keys}
    ccs = sorted(parse_cc(d["machine"]["compute_capability"]) for d, _ in per_profile)
    min_cc = f"{ccs[0][0]}.{ccs[0][1]}"
    mems = [memory_bytes.get(d["id"]) for d, _ in per_profile if memory_bytes.get(d["id"])]
    # nvidia-smi's MiB total is a little above what cudaDeviceProp.totalGlobalMem
    # reports (driver-reserved memory), so a device of exactly the smallest
    # contributing class must still qualify: bound at 95% of the smallest.
    min_mem = int(min(mems) * 0.95) if mems else 0
    toolkits = {d["machine"]["cuda"] for d, _ in per_profile}
    if len(toolkits) != 1:
        raise SystemExit(f"shipped profiles disagree on the CUDA toolkit: {toolkits}")
    template = per_profile[0][0]["machine"]
    return {
        "schema": per_profile[0][0]["schema"],
        "id": f"generic-cuda-sm{min_cc.replace('.', '')}plus-cuda{template['cuda']}",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tier": "generic",
        "applies": {"minimum_compute_capability": min_cc,
                    "minimum_device_memory_bytes": min_mem},
        "machine": {"device_name": "any CUDA device", "compute_capability": min_cc,
                    "driver": "any", "cuda": template["cuda"],
                    "cpu_model": "conservative intersection of the shipped profiles",
                    "logical_cpus": 0, "pdg_version": template["pdg_version"]},
        "coefficients": coefficients,
        "stage_models": models,
        "summaries": summaries,
        "evidence": {"shipping_margin": margin,
                     "from_profiles": [d["id"] for d, _ in per_profile],
                     "device_memory_bytes": memory_bytes},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("profiles", nargs="+", type=pathlib.Path)
    p.add_argument("--slug")
    p.add_argument("--margin", type=float, default=None,
                   help="shipping margin; defaults to 1.7 for a class profile and 3 for --generic (D0280)")
    p.add_argument("--generic", action="store_true")
    p.add_argument("--memory-bytes", action="append", default=[],
                   metavar="PROFILE_ID=BYTES",
                   help="device memory of a shipped profile's GPU (for the generic bounds)")
    p.add_argument("--out", type=pathlib.Path, required=True)
    a = p.parse_args()
    if a.margin is None:
        a.margin = 3.0 if a.generic else 1.7
    if a.generic:
        docs = [json.loads(x.read_text()) for x in a.profiles]
        docs = [d for d in docs if d.get("tier") == "shipped"]
        mem = {}
        for item in a.memory_bytes:
            k, _, v = item.partition("=")
            mem[k] = int(v)
        out = generic_from_shipped(docs, a.margin, mem)
    else:
        if len(a.profiles) != 1 or not a.slug:
            raise SystemExit("one PROFILE.json and --slug are required")
        out = shipped_from_local(json.loads(a.profiles[0].read_text()), a.slug, a.margin)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(a.out, out["id"], "models:", len(out["stage_models"]),
          "dropped:", (out.get("evidence") or {}).get("dropped_models"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
