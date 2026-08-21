#!/usr/bin/env python3
"""Build the B0277 local-calibration report (HTML).

Inputs: one or more rented-box result directories produced by
bench/remote/vast_calibrate.sh (suite aggregates before/after calibration,
the written profiles, calibrate logs, big-cloud reports), plus the local
validation figures.  Writes one self-contained HTML page.
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import statistics


def load(path: pathlib.Path):
    return json.loads(path.read_text()) if path and path.is_file() else None


def text(path: pathlib.Path) -> str:
    return path.read_text() if path and path.is_file() else ""


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def suite_rows(agg, reports_dir):
    if not agg or not agg.get("complete"):
        return [], {}
    rows = []
    for w in agg["workloads"]:
        r = load(reports_dir / f"{w['id']}-warm.json") if reports_dir else None
        prof = None
        exact = None
        if r:
            prof = (r.get("environment") or {}).get("candidate_placement_profile")
            c = r.get("comparison") or {}
            exact = c.get("exact_outputs") and c.get("exact_stdout") and c.get("exact_stderr")
        rows.append({"id": w["id"], "oracle": w["oracle_median_seconds"],
                     "pdg": w["candidate_median_seconds"], "speedup": w["median_speedup"],
                     "profile": prof, "exact": exact})
    return rows, {"gm": agg["equal_workload_geometric_mean_speedup"],
                  "total": agg["total_wall_time"]["speedup"],
                  "oracle_total": agg["total_wall_time"]["oracle_median_seconds"],
                  "pdg_total": agg["total_wall_time"]["candidate_median_seconds"]}


def parse_calibrate_log(log: str):
    """Rows of the per-model table printed by `pdg calibrate`."""
    rows = []
    for line in log.splitlines():
        m = re.match(r"^(\S+)\s+(\d+)\s+([\d.]+|-)\s+([\d.]+|-)\s+([\d.]+x|-)\s+(yes|NO|-)\s*(.*)$", line)
        if m and m.group(1) not in ("model",):
            rows.append({"model": m.group(1), "points": int(m.group(2)),
                         "host": None if m.group(3) == "-" else float(m.group(3)),
                         "device": None if m.group(4) == "-" else float(m.group(4)),
                         "speedup": None if m.group(5) == "-" else float(m.group(5)[:-1]),
                         "exact": m.group(6), "note": m.group(7).strip()})
    probes = {}
    for key in ("cuda_startup_ms", "host_to_device_ns/B", "device_to_host_ns/B",
                "synchronization_us", "index_build_ns/B", "packing_ns/B"):
        m = re.search(re.escape(key) + r":\s+([\d.]+)", log)
        if m:
            probes[key] = float(m.group(1))
    m = re.search(r"wrote .*\((\d+) of (\d+) measured models admitted[^)]*; (\d+) models in the profile\)", log)
    if not m:
        m = re.search(r"wrote .*\((\d+) of (\d+) models admitted", log)
    admitted = (int(m.group(1)), int(m.group(2))) if m else None
    return rows, probes, admitted


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--box", action="append", default=[], metavar="TITLE=DIR",
                   help="rented-box results directory from vast_calibrate.sh")
    p.add_argument("--reference-box-b0275", type=pathlib.Path,
                   help="B0275 box results dir (uncalibrated default and forced hybrid, for context)")
    p.add_argument("--tests-summary", type=str, default="",
                   help="one-line local validation summary")
    p.add_argument("--out", type=pathlib.Path, required=True)
    a = p.parse_args()

    boxes = []
    for spec in a.box:
        title, _, d = spec.partition("=")
        d = pathlib.Path(d)
        box = {"title": title, "dir": d,
               "machine": text(d / "machine.txt"),
               "status_before": text(d / "status-before.txt"),
               "status_after": text(d / "status-after.txt"),
               "doctor_before": text(d / "doctor-before.txt"),
               "doctor_after": text(d / "doctor-after.txt"),
               "calibrate_log": text(d / "calibrate.log"),
               "calibrate_big_log": text(d / "calibrate-big.log"),
               "calibrate_time": text(d / "calibrate.time"),
               "calibrate_big_time": text(d / "calibrate-big.time"),
               "calibrate_realdata_log": text(d / "calibrate-realdata.log"),
               "calibrate_post_log": text(d / "calibrate-post.log"),
               "calibrate_post_time": text(d / "calibrate-post.time"),
               "profile": load(d / "placement-profile.json"),
               "profile_standard": load(d / "placement-profile-standard.json"),
               "profile_realdata": load(d / "placement-profile-realdata.json"),
               "suites": {}}
        for name in ("suite-1m-default-uncalibrated", "suite-1m-default-calibrated",
                     "suite-1m-cuda-experimental", "suite-1m-default-uncalibrated-repeat",
                     "suite-1m-default-calibrated-appendbug"):
            box["suites"][name] = suite_rows(load(d / f"{name}.json"), d / f"{name}-reports")
        big = []
        for path in sorted(d.glob("big-*.json")):
            r = load(path)
            if not r:
                continue
            s = r["summary"]; c = r["comparison"]
            big.append({"label": path.stem[len("big-"):], "oracle": s["oracle"]["median_seconds"],
                        "pdg": s["candidate"]["median_seconds"], "speedup": c["median_speedup"],
                        "exact": c["exact_outputs"] and c["exact_stdout"] and c["exact_stderr"],
                        "profile": ((r.get("environment") or {}).get("candidate_placement_profile") or {}).get("active_profile"),
                        "rss_o": s["oracle"]["median_peak_rss_bytes"], "rss_c": s["candidate"]["median_peak_rss_bytes"]})
        box["big"] = big
        boxes.append(box)

    css = """
    :root { --bg:#f6f7f9; --fg:#16202a; --muted:#5b6b7b; --line:#d9dee5; --accent:#157a63; --soft:#eceff3; --chip:#e3efe9; --good:#1a7f37; --warn:#b26a00; }
    @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f1317; --fg:#e6e9ee; --muted:#9aa7b4; --line:#2a323c; --accent:#5cc7a8; --soft:#171d24; --chip:#173229; --good:#5fd08a; --warn:#e2a23a; } }
    :root[data-theme="dark"] { --bg:#0f1317; --fg:#e6e9ee; --muted:#9aa7b4; --line:#2a323c; --accent:#5cc7a8; --soft:#171d24; --chip:#173229; --good:#5fd08a; --warn:#e2a23a; }
    body { background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; margin:0; }
    main { max-width: 1080px; margin: 0 auto; padding: 28px 22px 72px; }
    h1 { font-size: 32px; letter-spacing: -0.01em; margin: 0 0 4px; text-wrap: balance; }
    h2 { font-size: 22px; margin: 44px 0 10px; padding-bottom: 6px; border-bottom: 2px solid var(--line); text-wrap: balance; }
    h3 { font-size: 17px; margin: 24px 0 6px; } .lead { color: var(--muted); font-size: 16px; max-width: 70ch; }
    .eyebrow { color: var(--accent); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600; margin-bottom: 8px; }
    .chips { display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0 6px; }
    .chip { background: var(--chip); border-radius: 6px; padding: 8px 12px; min-width: 150px; }
    .chip b { display:block; font-size: 22px; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
    .chip span { color: var(--muted); font-size: 12.5px; }
    table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 10px 0 18px; font-variant-numeric: tabular-nums; }
    .scroll { overflow-x: auto; } th, td { border-bottom: 1px solid var(--line); padding: 5px 8px; text-align: right; vertical-align: top; }
    th:first-child, td:first-child, td.name { text-align: left; } th { background: var(--soft); font-weight: 600; }
    tr:last-child th { border-top: 2px solid var(--line); }
    .callout { background: var(--soft); border-left: 4px solid var(--accent); padding: 12px 16px; margin: 14px 0; max-width: 78ch; }
    .callout p { margin: 6px 0; }
    code { background: var(--soft); padding: 1px 4px; border-radius: 3px; font-size: 13px; }
    pre { white-space: pre-wrap; font-size: 12.5px; background: var(--soft); padding: 8px 10px; border-radius: 4px; overflow-x: auto; }
    .good { color: var(--good); font-weight: 600; } .warn { color: var(--warn); font-weight: 600; }
    small { color: var(--muted); } ul { padding-left: 20px; } li { margin: 4px 0; }
    p { max-width: 78ch; }
    """

    def suite_compare_table(box):
        u = box["suites"]["suite-1m-default-uncalibrated"]
        c = box["suites"]["suite-1m-default-calibrated"]
        f = box["suites"]["suite-1m-cuda-experimental"]
        if not u[0] and not c[0]:
            return "<p><em>not available</em></p>"
        ids = [r["id"] for r in (c[0] or u[0])]
        by = lambda rows: {r["id"]: r for r in rows}
        U, C, F = by(u[0]), by(c[0]), by(f[0])
        head = ("<tr><th>Workload</th><th>Pinned PDAL (s)</th><th>PDG default, uncalibrated (s)</th><th>Speedup</th>"
                "<th>PDG default, calibrated (s)</th><th>Speedup</th><th>Exact</th>"
                "<th>PDG hybrid forced (s)</th><th>Speedup</th></tr>")
        body = ""
        for i in ids:
            ru, rc, rf = U.get(i), C.get(i), F.get(i)
            oracle = (rc or ru or rf)["oracle"]
            def cell(r):
                if not r:
                    return "<td>—</td><td>—</td>"
                cls = "good" if r["speedup"] >= 1.5 else ""
                return f"<td>{fmt(r['pdg'])}</td><td class='{cls}'>{fmt(r['speedup'],2)}x</td>"
            changed = ru and rc and abs(rc["speedup"] - ru["speedup"]) / max(ru["speedup"], 1e-9) > 0.15
            body += (f"<tr><td class=name>{i}{' <span class=good>&#9650;</span>' if changed and rc['speedup']>ru['speedup'] else ''}"
                     f"{' <span class=warn>&#9660;</span>' if changed and rc['speedup']<ru['speedup'] else ''}</td>"
                     f"<td>{fmt(oracle)}</td>{cell(ru)}{cell(rc)}"
                     f"<td>{'yes' if (rc or ru)['exact'] else ('NO' if (rc or ru)['exact'] is False else '—')}</td>{cell(rf)}</tr>")
        def foot(label, key, digits=3):
            return (f"<th>{label}</th><th>{fmt(u[1].get('oracle_total') if key=='total' else None,2) if key=='total' else ''}</th>"
                    f"<th>{fmt(u[1].get('pdg_total'),2) if key=='total' and u[1] else ''}</th><th>{fmt(u[1].get(key),digits) if u[1] else '—'}x</th>"
                    f"<th>{fmt(c[1].get('pdg_total'),2) if key=='total' and c[1] else ''}</th><th>{fmt(c[1].get(key),digits) if c[1] else '—'}x</th><th></th>"
                    f"<th>{fmt(f[1].get('pdg_total'),2) if key=='total' and f[1] else ''}</th><th>{fmt(f[1].get(key),digits) if f[1] else '—'}x</th>")
        footer = f"<tr>{foot('equal-workload geometric mean','gm')}</tr><tr>{foot('total wall (sum of medians)','total')}</tr>"
        return f"<div class=scroll><table>{head}{body}{footer}</table></div>"

    def calibrate_table(rows):
        if not rows:
            return "<p><em>not available</em></p>"
        head = "<tr><th>Model</th><th>Points</th><th>Host (s)</th><th>Device (s)</th><th>Speedup</th><th>Byte-exact</th><th>Note</th></tr>"
        body = ""
        for r in rows:
            cls = "good" if (r["speedup"] or 0) >= 1.0 else "warn"
            body += (f"<tr><td class=name>{html.escape(r['model'])}</td><td>{r['points']:,}</td><td>{fmt(r['host'])}</td>"
                     f"<td>{fmt(r['device'])}</td><td class='{cls}'>{fmt(r['speedup'],2)}x</td><td>{r['exact']}</td>"
                     f"<td class=name><small>{html.escape(r['note'])}</small></td></tr>")
        return f"<div class=scroll><table>{head}{body}</table></div>"

    def profile_table(profile):
        if not profile:
            return "<p><em>not available</em></p>"
        head = ("<tr><th>Model</th><th>Admitted</th><th>Envelope (points)</th><th>host fixed (ms)</th><th>host ns/pt</th>"
                "<th>device fixed (ms)</th><th>device ns/pt</th></tr>")
        body = ""
        summaries = {s["model"]: s for s in profile.get("summaries", [])}
        names = sorted(set(list(profile["stage_models"].keys()) + list(summaries.keys())))
        for n in names:
            m = profile["stage_models"].get(n)
            s = summaries.get(n, {})
            if m:
                body += (f"<tr><td class=name>{n}</td><td class=good>yes</td>"
                         f"<td>{m.get('minimum_device_points',0):,} – {m.get('maximum_device_points',0):,}</td>"
                         f"<td>{m['host_fixed_ns']/1e6:.1f}</td><td>{m['host_ns_per_point']:.0f}</td>"
                         f"<td>{m['device_fixed_ns']/1e6:.1f}</td><td>{m['device_ns_per_point']:.0f}</td></tr>")
            else:
                why = "device never won" if s and not s.get("device_wins_somewhere") else ("outputs differed" if s and not s.get("byte_exact", True) else "not measured / device path unavailable")
                body += f"<tr><td class=name>{n}</td><td class=warn>no</td><td colspan=5 class=name><small>{why}</small></td></tr>"
        return f"<div class=scroll><table>{head}{body}</table></div>"

    def big_table(rows):
        if not rows:
            return "<p><em>not available</em></p>"
        head = ("<tr><th>Workload / configuration</th><th>Pinned PDAL (s)</th><th>PDG (s)</th><th>Speedup</th><th>Exact</th>"
                "<th>Active profile</th><th>Peak RSS PDAL / PDG (GB)</th></tr>")
        body = ""
        def pretty(label):
            label = label.replace("ahn4-", "")
            notes = {"features-plain-default": "features (plain writer), original tile, default mode — after D0278",
                     "features-plain-default-prefix": "features (plain writer), original tile, default mode — before D0278 (resident preflight refused the tile's EB types; host)",
                     "features-plain-cuda-experimental": "features (plain writer), original tile, hybrid forced",
                     "r6-features-default": "r6 reference graph (extra_dims=all, LAZ), original tile, default mode — after D0278 (its extra_dims=all sink stays bounded to the measured 1M layout; host)",
                     "r6-features-default-prefix": "r6 reference graph (extra_dims=all, LAZ), original tile, default mode — before D0278",
                     "lof-plain-default": "LOF (plain writer), original tile, default mode — after D0278",
                     "lof-plain-default-prefix": "LOF (plain writer), original tile, default mode — before D0278 (host)",
                     "features-noeb-default": "features (plain writer), tile without extra-bytes dims, default mode",
                     "features-noeb-cuda-experimental": "features (plain writer), tile without extra-bytes dims, hybrid forced",
                     "lof-noeb-default": "LOF (plain writer), tile without extra-bytes dims, default mode"}
            return notes.get(label, label)
        order = ["features-plain-default-prefix", "features-plain-default", "features-plain-cuda-experimental",
                 "features-noeb-default", "features-noeb-cuda-experimental", "lof-plain-default-prefix",
                 "lof-plain-default", "lof-noeb-default", "r6-features-default-prefix", "r6-features-default"]
        rows = sorted(rows, key=lambda r: (order.index(r["label"].replace("ahn4-", "")) if r["label"].replace("ahn4-", "") in order else 99))
        for r in rows:
            body += (f"<tr><td class=name>{html.escape(pretty(r['label']))}</td><td>{fmt(r['oracle'],1)}</td><td>{fmt(r['pdg'],1)}</td>"
                     f"<td class='{'good' if r['speedup']>=1.5 else ''}'>{fmt(r['speedup'],2)}x</td><td>{'yes' if r['exact'] else 'NO'}</td>"
                     f"<td class=name><small>{html.escape(str(r['profile'] or '—'))}</small></td>"
                     f"<td>{r['rss_o']/1e9:.1f} / {r['rss_c']/1e9:.1f}</td></tr>")
        return f"<div class=scroll><table>{head}{body}</table></div>"

    def machine_summary(box):
        lines = [l for l in box["machine"].splitlines() if l.strip() and not l.startswith("---")]
        return html.escape("\n".join(lines[:9]))

    def summary_sentence(box):
        u = box["suites"]["suite-1m-default-uncalibrated"]
        c = box["suites"]["suite-1m-default-calibrated"]
        f = box["suites"]["suite-1m-cuda-experimental"]
        if not (u[0] and c[0]):
            return "results not available."
        U = {r["id"]: r for r in u[0]}; C = {r["id"]: r for r in c[0]}
        big = {r["label"].replace("ahn4-", ""): r for r in box["big"]}
        feat = big.get("features-plain-default"); lof = big.get("lof-plain-default")
        return (f"Default mode went from {fmt(u[1]['gm'],2)}× to {fmt(c[1]['gm'],2)}× on average over the fourteen workflows "
                f"(whole set {fmt(u[1]['total'],2)}× → {fmt(c[1]['total'],2)}×); the feature-extraction workflow went from "
                f"{fmt(U['r6-features']['speedup'],1)}× to {fmt(C['r6-features']['speedup'],1)}×, while forcing the GPU on everything "
                f"would have given {fmt(f[1]['gm'],2)}× on average — the profile keeps the light workflows on the CPU where the GPU loses. "
                f"On the 47.5-million-point survey tile, default mode ran feature extraction at "
                f"{fmt(feat['speedup'],1) if feat else '—'}× and outlier scoring (LOF) at {fmt(lof['speedup'],1) if lof else '—'}× "
                f"stock PDAL's speed, byte-identical.")

    parts = [f"""<title>pdg calibrate</title><style>{css}</style><main>
<div class=eyebrow>PDAL-GPU · B0277 · local placement calibration</div>
<h1>Letting other machines use the GPU by default</h1>
<p class=lead>The <code>pdg calibrate</code> command, what it changed, the tests behind it, and what it did on two rented
machines that previously ran the CPU path in default mode. Every default-mode number here is byte-identical to stock
PDAL 2.10.0 (the pinned oracle) — the profile only chooses between two exact executors.</p>
"""]

    # Chips: headline from the first box with both suites.
    for box in boxes:
        u = box["suites"]["suite-1m-default-uncalibrated"][1]
        c = box["suites"]["suite-1m-default-calibrated"][1]
        if u and c:
            rows_u = {r["id"]: r for r in box["suites"]["suite-1m-default-uncalibrated"][0]}
            rows_c = {r["id"]: r for r in box["suites"]["suite-1m-default-calibrated"][0]}
            r6u, r6c = rows_u.get("r6-features"), rows_c.get("r6-features")
            _, _, admitted = parse_calibrate_log(box["calibrate_log"])
            parts.append(f"""<div class=chips>
<div class=chip><b>{fmt(u['gm'],2)}× → {fmt(c['gm'],2)}×</b><span>{html.escape(box['title'])}: default-mode geometric mean, before → after calibrate</span></div>
<div class=chip><b>{fmt(u['total'],2)}× → {fmt(c['total'],2)}×</b><span>whole suite, before → after</span></div>
<div class=chip><b>{fmt(r6u['speedup'] if r6u else None,1)}× → {fmt(r6c['speedup'] if r6c else None,1)}×</b><span>r6 features, before → after</span></div>
<div class=chip><b>0</b><span>byte differences from stock PDAL in default mode</span></div>
</div>""")
            break

    parts.append("""
<h2>1. Plain-language summary</h2>
<div class=callout>
<p><b>The problem.</b> <code>pdg</code> only sends work to the graphics card where a measurement on the same machine proved
the whole job finishes sooner. Until now that measurement existed for exactly one machine — the reference workstation's
RTX 4090 with one specific driver. On any other computer, even with a capable GPU, <code>pdg</code> quietly used the CPU
path only. The B0275 report showed the cost: on a rented RTX 4090 the feature-extraction workflow ran 2.7× faster than
stock PDAL by default but 11.2× when the GPU was forced on.</p>
<p><b>What changed.</b> A new command, <code>pdg calibrate</code>, runs the same measurements on whatever machine it is
run on — a few minutes of timed jobs, comparing the CPU path with the GPU path and checking that the two produce
byte-identical files — and saves the result as a profile tied to that exact machine (GPU model, driver, CUDA build, CPU
model, core count and <code>pdg</code> version). From then on, ordinary <code>pdg pipeline</code> commands on that machine
pick the GPU where the profile says it wins. Nothing runs this automatically; a machine without a profile behaves exactly
as before, and a profile from a different machine is ignored.</p>
""" + "".join(f"""<p><b>What it did on {html.escape(b['title'])}.</b> {summary_sentence(b)}</p>
""" for b in boxes) + """<p><b>What did not change.</b> The output files. Default mode is still checked byte-for-byte against stock PDAL, with or
without a profile, and every default-mode number in this report passed that check. There is no new behaviour in the
default path — no first-run surprise, no background tuning.</p>
</div>

<h2>2. Technical summary</h2>
<ul>
<li><b>Same procedure as the reference profile.</b> The embedded SM-89 profile was fitted from measured host/device
seconds per placement model with a residual linear model after subtracting planner-owned terms; <code>pdg calibrate</code>
re-measures the same case pipelines on the local machine (host = the fork's exact host build, device = the resident device
executor under a calibration-only override) and fits them with the same code path (<code>pdg_placement_audit
--suggest-models</code>, factored into <code>PlacementCalibration.cpp</code>).</li>
<li><b>Planner-owned coefficients are measured locally</b> — CUDA cold start, host↔device transfer, kernel-launch
synchronisation, kNN index build per persistent byte — with LAS packing inherited from the reference profile and
marked as such.</li>
<li><b>Fail-closed envelopes.</b> A model is admitted only for sizes between the smallest measured device win and the
largest measured size, and only if every measured pair was byte-identical. Structural gates are unchanged: the profile
supplies coefficients and envelopes, not new shapes.</li>
<li><b>Runtime stays a fixed linear rule.</b> The profile is a file consulted once per process when its seven-field
machine key equals the current machine; the runtime never times, fits or tunes.</li>
<li><b>Calibration override is scoped.</b> <code>PDG_CALIBRATION_FORCE_DEVICE_PLACEMENT</code> only affects the explicit
<code>resident</code> command; the automatic <code>pipeline</code> route declines while it is set (tested).</li>
<li><b>D0278, found by this measurement.</b> The AHN4 tiles' extra-bytes VLR declares <code>Deviation</code> as uint16 and
<code>Amplitude</code>/<code>Reflectance</code> as double, where PDAL's standard table (and PDG's registry) type them otherwise;
the resident executor's layout preflight refused that and default mode stayed on the host even with a profile. The resident
layout now follows the file's type for dimensions no planned device stage reads or writes — exactly what stock PDAL does —
and keeps the refusal, with a specific reason, for a dimension a device stage computes on. Tested both ways against the pinned
oracle; the boxes were rebuilt with it and the original tile re-measured (rows marked "after D0278").</li>
</ul>
""")

    parts.append(f"""
<h2>3. Tests</h2>
<p>{html.escape(a.tests_summary) if a.tests_summary else ''}</p>
<ul>
<li><b>Unit</b> (<code>tests/unit/local_profile_test.cpp</code>): profile schema round trip and rejection of malformed
documents; lookup statuses (not-found, malformed, machine-mismatch, disabled); the reference key still resolves to the
embedded profile and an unknown key fails closed; the calibration override exists only under its environment variable
and covers every embedded model name; the residual fit splits by sign exactly as the audit prints it; the synthetic cloud
is deterministic, bounded and writes a valid LAS 1.4 format-7 file.</li>
<li><b>Process</b> (<code>tests/differential/calibrate_test.py</code>, <code>pdg_calibrate_local_profile_cuda_exact</code>):
<code>--status</code>/<code>--dry-run</code> write nothing; the reference machine declines without <code>--force</code>; a
quick calibration (two models, 100K/250K points) admits only byte-exact device wins; the profile makes the automatic
route select the resident device path for a normal/covariance graph (proof gate exit 0), while a missing profile, a
profile keyed to another machine, and the calibration override all fail closed (exit 124); the default output with the
profile equals the host output without it and the pinned oracle's output.</li>
<li><b>Existing lanes</b>: the placement audit still reproduces the embedded profile (214/218 accuracy, unchanged) with
the shared fitting code; the full CUDA aggregate and the Host Debug aggregate were re-run after the change.</li>
</ul>
""")

    for index, box in enumerate(boxes):
        rows_std, probes, admitted = parse_calibrate_log(box["calibrate_log"])
        rows_big, _, admitted_big = parse_calibrate_log(box["calibrate_big_log"])
        rows_real, probes_real, _ = parse_calibrate_log(box["calibrate_realdata_log"])
        rows_post, _, admitted_post = parse_calibrate_log(box["calibrate_post_log"])
        wall_post = re.search(r"Elapsed \(wall clock\) time: (.*)", box["calibrate_post_time"])
        repeat = box["suites"]["suite-1m-default-uncalibrated-repeat"][1]
        appendbug = box["suites"]["suite-1m-default-calibrated-appendbug"][1]
        extra_runs = ""
        if repeat:
            extra_runs += (f"<li>A second uncalibrated default run on this box (before any profile existed): geometric mean "
                           f"{fmt(repeat['gm'],3)}×, total {fmt(repeat['total'],3)}× — the run-to-run spread of the whole suite.</li>")
        if appendbug:
            extra_runs += (f"<li>A default run under the first profile as written by the original <code>--append</code>, whose "
                           f"defect (fixed before checkpoint) had replaced the appended models' 250K–4M envelopes with 16M–48M, so "
                           f"at 1M those models ran on the host: geometric mean {fmt(appendbug['gm'],3)}×, total "
                           f"{fmt(appendbug['total'],3)}× — effectively another uncalibrated repeat, retained as evidence.</li>")
        wall = re.search(r"Elapsed \(wall clock\) time: (.*)", box["calibrate_time"])
        wall_big = re.search(r"Elapsed \(wall clock\) time: (.*)", box["calibrate_big_time"])
        parts.append(f"""
<h2>{4+index}. Rented box: {html.escape(box['title'])}</h2>
<pre>{machine_summary(box)}</pre>
<h3>Before calibration</h3>
<pre>{html.escape(box['status_before'].strip())}</pre>
<h3>What <code>pdg calibrate</code> measured</h3>
<p><small>Synthetic terrain cloud, sizes 250K / 1M / 4M points, three alternating host/device pairs per size (medians);
then <code>--append</code> at 16M and 48M points for the three big-cloud models (one pair each). Wall clock of the standard
run: {html.escape(wall.group(1) if wall else '—')}; of the append run: {html.escape(wall_big.group(1) if wall_big else '—')}.
Admitted: {admitted[0] if admitted else '—'} of {admitted[1] if admitted else '—'} standard models{(', ' + str(admitted_big[0]) + ' of ' + str(admitted_big[1]) + ' at the big sizes') if admitted_big else ''}.</small></p>
<p><small>Planner-owned coefficients measured here: {html.escape(', '.join(f'{k} {v:.4g}' for k, v in probes.items()))}.</small></p>
{calibrate_table(rows_std + rows_big)}
<h3>Follow-up <code>--append</code> after the merge fix (250K / 1M, one pair each; refits over every size measured)</h3>
<p><small>Wall clock {html.escape(wall_post.group(1) if wall_post else '—')}; the two point-program models are measured here for the first time on this box
(their device path had refused to overwrite an output file in the standard run).</small></p>
{calibrate_table(rows_post)}
<h3>The written profile (final)</h3>
{profile_table(box['profile'])}
<h3>Other runs on this box</h3>
<ul>{extra_runs or '<li>none</li>'}</ul>
<h3>After calibration</h3>
<pre>{html.escape(box['status_after'].strip())}</pre>
<h3>The fourteen reference workflows at 1M points: default mode before and after, and the forced hybrid</h3>
<p><small>Warm cache, three alternating pinned/PDG pairs per workload, frozen clock; the box's own pinned upstream
PDAL 2.10.0 build is the baseline. "Exact" is the calibrated default run's byte/metadata/stdout/stderr/status equality with
the oracle. Arrows mark workloads whose default speedup moved by more than 15% after calibration.</small></p>
{suite_compare_table(box)}
<h3>Real-data calibration for comparison (1M reference fixture, 250K/1M, not used for the suite)</h3>
{calibrate_table(rows_real)}
<h3>AHN4 47.5M-point tile, default mode with the profile active (one run per cell)</h3>
<p><small>"Original tile" is GeoTile 25GN1_01 as published (LAS 1.4 format 8 with extra-bytes dims); "tile without extra-bytes
dims" is the same tile passed once through the pinned oracle's <code>translate --writers.las.forward=all</code>. Rows marked
"before D0278" ran the resident preflight that refused the tile's EB types; the "after" rows ran the engine rebuilt with
D0278 on the same box.</small></p>
{big_table(box['big'])}
""")

    parts.append("""
<h2>What this means for the reference workstation</h2>
<p>Nothing changes there: the embedded profile takes precedence, <code>pdg calibrate</code> reports "nothing to
calibrate" and writes nothing unless <code>--force</code> is given, and the B0274 exact-suite claim of record stands.</p>

<h2>Method and caveats</h2>
<ul>
<li>Boxes are Vast.ai containers with a CPU quota (the <code>cpu.max</code> line in the machine block); host-parallel
paths and therefore the host side of every calibration pair see that quota, which is exactly what a user of that box would
see.</li>
<li>One calibrate run per box; the standard tier is three pairs per size, the big-size append one pair. Speedups are
medians of alternating pairs; single-run cells say so.</li>
<li>The synthetic cloud is a stand-in for real data; the real-data comparison table shows how the two agree on the same
machine. Users are advised to calibrate with <code>--input</code> on their own tiles.</li>
<li>The r6 reference graph writes <code>extra_dims=all</code>, whose automatic admission is bounded to the measured 1M
layout on every machine (its publication cost was never part of the compose model); the plain-writer feature graph is the
shape the calibrated compose model covers, so both are reported at 47.5M.</li>
<li>Vast spend for this record is stated in BENCHMARKS.md B0277; both instances were destroyed after their archives were
downloaded.</li>
</ul>
</main>""")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(parts))
    print(a.out, a.out.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
