#!/usr/bin/env python3
"""Print the B0278 sweep tables as Markdown for BENCHMARKS.md.

    sweep_tables.py --sweep build/benchmarks/b0278-sweep --profiles data/placement-profiles
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics


def load(p):
    return json.loads(p.read_text()) if p.is_file() else None


def text(p):
    return p.read_text() if p.is_file() else ""


def suite(agg):
    if not agg or not agg.get("complete"):
        return None
    return {"gm": agg["equal_workload_geometric_mean_speedup"], "total": agg["total_wall_time"]["speedup"],
            "rows": {w["id"]: w["median_speedup"] for w in agg["workloads"]}}


def cal_rows(log):
    rows = {}
    for line in log.splitlines():
        m = re.match(r"^(\S+)\s+(\d+)\s+([\d.]+|-)\s+([\d.]+|-)\s+([\d.]+x|-)\s+(yes|NO|-)", line)
        if m and m.group(1) != "model" and m.group(5) != "-":
            rows.setdefault(m.group(1), {})[int(m.group(2))] = float(m.group(5)[:-1])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=pathlib.Path, required=True)
    p.add_argument("--profiles", type=pathlib.Path, required=True)
    a = p.parse_args()
    boxes = []
    for d in sorted(a.sweep.iterdir()):
        r = d / "results-sweep"
        if not r.is_dir():
            continue
        m = text(r / "machine.txt").splitlines()
        gpu = next((l for l in m if "NVIDIA" in l), "?")
        cpu = next((l.split(":", 1)[1].strip() for l in m if l.startswith("Model name")), "?")
        boxes.append({"name": d.name, "gpu": gpu, "cpu": cpu,
                      "before": suite(load(r / "suite-1m-default-uncalibrated.json")),
                      "local": suite(load(r / "suite-1m-default-calibrated.json")),
                      "shipped": suite(load(r / "suite-1m-default-shipped.json")),
                      "generic": suite(load(r / "suite-1m-default-generic.json")),
                      "cal": cal_rows(text(r / "calibrate.log")), "calbig": cal_rows(text(r / "calibrate-big.log")),
                      "big": {f.stem[4:]: load(f) for f in sorted(r.glob("big-*.json"))}})
    print("### Fourteen workflows at 1M, default mode (GM / total)\n")
    print("| GPU (host) | before | local profile | shipped (no calibration) | generic (own profile excluded) | r6 before → shipped |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for b in boxes:
        def c(s):
            return f"{s['gm']:.3f}x / {s['total']:.3f}x" if s else "—"
        r6b = b["before"]["rows"].get("r6-features") if b["before"] else None
        r6s = b["shipped"]["rows"].get("r6-features") if b["shipped"] else None
        print(f"| {b['gpu']} ({b['cpu']}) | {c(b['before'])} | {c(b['local'])} | **{c(b['shipped'])}** | {c(b['generic'])} | "
              f"{r6b:.2f}x → {r6s:.2f}x |" if r6b and r6s else f"| {b['gpu']} ({b['cpu']}) | {c(b['before'])} | {c(b['local'])} | **{c(b['shipped'])}** | {c(b['generic'])} | — |")
    print("\n### Calibrate speedups (250K / 1M / 4M; 16M / 48M)\n")
    models = sorted({m for b in boxes for m in b["cal"]})
    print("| Model | " + " | ".join(b["gpu"].split(",")[0].replace("NVIDIA ", "").replace("GeForce ", "") for b in boxes) + " |")
    print("| --- | " + " | ".join("---:" for _ in boxes) + " |")
    for m in models:
        cells = []
        for b in boxes:
            r = b["cal"].get(m, {}); rb = b["calbig"].get(m, {})
            def sp(pts, src):
                v = src.get(pts)
                return f"{v:.1f}" if v else "—"
            cell = f"{sp(250000, r)}/{sp(1000000, r)}/{sp(4000000, r)}"
            if rb:
                cell += f"; {sp(16000000, rb)}/{sp(48000000, rb)}"
            cells.append(cell)
        print(f"| {m} | " + " | ".join(cells) + " |")
    print("\n### AHN4 47.5M cells\n")
    print("| GPU | run | pinned / PDG s | speedup | exact | tier |")
    print("| --- | --- | ---: | ---: | --- | --- |")
    for b in boxes:
        for k, doc in b["big"].items():
            if not doc:
                continue
            s = doc["summary"]; c = doc["comparison"]
            cpp = (doc.get("environment") or {}).get("candidate_placement_profile") or {}
            tier = cpp.get("active_profile_tier") or next(
                (pfx[:-1] for pfx in ("local-", "shipped-", "generic-")
                 if (cpp.get("active_profile") or "").startswith(pfx)), "?")
            ex = c["exact_outputs"] and c["exact_stdout"] and c["exact_stderr"]
            print(f"| {b['gpu'].split(',')[0]} | {k} | {s['oracle']['median_seconds']:.1f} / {s['candidate']['median_seconds']:.1f} | {c['median_speedup']:.2f}x | {'yes' if ex else 'NO'} | {tier} |")
    profiles = [load(f) for f in sorted(a.profiles.glob("*.json"))]
    shipped = [x for x in profiles if x and x.get("tier") == "shipped"]
    generic = next((x for x in profiles if x and x.get("tier") == "generic"), None)
    if shipped:
        print("\n### Shipped profiles: admitted envelopes\n")
        names = sorted({m for x in shipped for m in x["stage_models"]})
        print("| Model | " + " | ".join(x["machine"]["device_name"].replace("NVIDIA ", "").replace("GeForce ", "") for x in shipped) + (" | generic" if generic else "") + " |")
        print("| --- | " + " | ".join("---:" for _ in shipped) + (" | ---:" if generic else "") + " |")
        for m in names:
            cells = []
            for x in shipped + ([generic] if generic else []):
                sm = x["stage_models"].get(m)
                cells.append(f"{sm['minimum_device_points']//1000}K–{sm['maximum_device_points']//1000000}M" if sm else "host")
            print(f"| {m} | " + " | ".join(cells) + " |")
        if generic:
            print(f"\nGeneric: id `{generic['id']}`, applies CC >= {generic['applies']['minimum_compute_capability']}, "
                  f"memory >= {generic['applies']['minimum_device_memory_bytes']/1e9:.1f} GB, from {generic['evidence']['from_profiles']}")


if __name__ == "__main__":
    main()
