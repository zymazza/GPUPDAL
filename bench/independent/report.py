#!/usr/bin/env python3
"""Build the plain-language HTML report and print it to PDF with Chromium."""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import pathlib
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import charts  # noqa: E402
from charts import (FAMILIES, SIZE_LABEL, SIZE_POINTS, TASK_META, TOOL_COLOR, TOOL_NAME, TOOL_ORDER,  # noqa: E402
                    fmt_seconds, median)
from common import RENDERS, RESULTS, WORK  # noqa: E402
from tasks import TASKS  # noqa: E402

REPORT = WORK / "report"
CH = REPORT / "charts"
IMG = REPORT / "img"

MACHINES = {
    # results label -> (display name, short name, plain description)
    "workstation": ("Workstation (Ryzen 9 7900 + RTX 4090)", "Workstation",
                    "A desktop PC: AMD Ryzen 9 7900 (12 cores, 24 threads), 62 GB of memory, NVIDIA GeForce RTX 4090 graphics card (24 GB), Arch Linux."),
    "a100-epyc": ("Cloud server (EPYC 7532 + A100 40 GB)", "Cloud server (A100)",
                  "A rented data-centre machine: AMD EPYC 7532 (a container allowed to use about 31 of its 64 threads), 503 GB of memory, NVIDIA A100 SXM4 data-centre GPU (40 GB), Ubuntu 24.04."),
    "rtx3060-xeon": ("Budget cloud PC (Xeon E5-2697A v4 + RTX 3060 12 GB)", "Budget PC (RTX 3060)",
                     "A rented older machine: Intel Xeon E5-2697A v4 from 2016 (a container allowed about 15 threads), 125 GB of memory, NVIDIA GeForce RTX 3060 (12 GB), Ubuntu 24.04."),
}
TOOL_ROWS = [
    ("pdg_gpu", "GPUPDAL (this project) — GPU on, automatic", "The program built from this repository. It is a copy of PDAL with a faster engine that can use an NVIDIA graphics card (GPU). In this mode it decides by itself, per job, whether to use the GPU or the CPU. This is the default mode a user gets.", "<code>gpupdal pipeline job.json</code>"),
    ("pdg_cpu", "GPUPDAL — CPU only", "The same program, run as if the computer had no NVIDIA graphics card (<code>CUDA_VISIBLE_DEVICES=\"\"</code>). This shows what a user without a GPU would get.", "same, with the GPU hidden"),
    ("pdg_gpu_all", "GPUPDAL — GPU forced on", "The same program told to use the GPU for every job that has a GPU version, even where it would normally decide not to (<code>PDG_EXPERIMENTAL_CUDA_HYBRID=1</code>).", "same, with the GPU forced"),
    ("pdal_pinned", "PDAL 2.10.0, built from source", "The standard open-source PDAL library and command line, built from the exact upstream commit this project is based on. Its results are the reference: GPUPDAL preserves its semantics, with byte equality for deterministic outputs and canonical comparison for inherently nondeterministic containers.", "<code>pdal pipeline job.json</code>"),
    ("pdal_sys", "PDAL 2.10.1, Linux package", "PDAL as installed from the Arch Linux package repository — what a typical Linux user would install.", "<code>pdal pipeline job.json</code>"),
    ("lastools", "LAStools 250207 (Linux, unlicensed)", "The well-known LAStools programs from rapidlasso, native 64-bit Linux versions. Some tools (ground, height, DTM, DSM, noise, thin, tile, clip) are paid software; they were run without a licence, which is allowed for testing. The maker says the unlicensed versions may add small noise to results and may refuse or slow down on files with more than 3 million points.", "<code>lasground_new64 -i in.laz -o out.laz</code> and similar"),
    ("wrench", "QGIS engine (pdal_wrench 1.5.1)", "The command-line program that QGIS uses behind the scenes for point-cloud processing. It is built on the PDAL library and can split work over many CPU threads. Run directly, without QGIS.", "<code>pdal_wrench classify_ground --input … --output … --threads 24</code>"),
    ("qgis", "QGIS 4.2 (qgis_process)", "The QGIS desktop GIS, run from the command line with <code>qgis_process</code>. This includes QGIS starting up (about 1.2 s) and then calling pdal_wrench. It is what a user gets from the QGIS Processing toolbox.", "<code>qgis_process run pdal:classifyground --INPUT=… --OUTPUT=…</code>"),
]


def load_all() -> dict:
    docs = {}
    order = ["workstation", "a100-epyc", "rtx3060-xeon"]
    files = sorted(RESULTS.glob("*.json"), key=lambda f: (order.index(f.stem) if f.stem in order else 99, f.stem))
    for f in files:
        if f.stem.startswith("smoke"):
            continue
        d = json.load(open(f))
        docs[d["label"]] = d
    return docs


def esc(s) -> str:
    return html.escape(str(s))


def img(path: pathlib.Path, caption: str = "", width="100%") -> str:
    if path is None or not pathlib.Path(path).exists():
        return f'<p class="missing">[picture not available: {esc(path)}]</p>'
    path = pathlib.Path(path)
    if REPORT not in path.parents:
        # copy renders next to the report so the HTML is self-contained
        dst = IMG / path.name
        IMG.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
        path = dst
    rel = path.relative_to(REPORT)
    return f'<figure><img src="{rel}" style="width:{width}"><figcaption>{caption}</figcaption></figure>'


def table(headers, rows, cls="data") -> str:
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def geo_mean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def rec_status(rec) -> str:
    if rec is None:
        return "—"
    if rec.get("rc", 0) != 0:
        tail = rec.get("tail", "").lower()
        if "licens" in tail:
            return "refused (unlicensed)"
        if rec.get("rc") == -9:
            return "timed out"
        return "failed"
    return fmt_seconds(median(rec))


def results_table(idx, size, tools=TOOL_ORDER, with_points=False) -> str:
    headers = ["Job"] + [charts.SHORT[t] for t in tools]
    rows = []
    for t in TASKS:
        if size not in t["sizes"]:
            continue
        recs = [idx.get((size, t["id"], tool)) for tool in tools]
        if not any(recs):
            continue
        best = min([median(r) for r in recs if median(r)] or [None])
        cells = [f'<span class="task">{esc(t["name"])}</span>']
        for r in recs:
            s = rec_status(r)
            if r and "unlicensed" in (r.get("note") or ""):
                s += "*"
            v = median(r)
            cls = ' class="best"' if v is not None and best is not None and abs(v - best) < 1e-9 else ""
            cells.append(f"<td{cls}>{s}</td>" if cls else s)
        # cells may already contain <td>; normalise
        row = "<tr>" + "".join(c if c.startswith("<td") else f"<td>{c}</td>" for c in cells) + "</tr>"
        rows.append(row)
    return (f'<table class="data"><thead><tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
            '<p class="small">Seconds, median of the timed runs; the fastest tool in each row is in bold blue; "—" = the tool cannot do this job; '
            '* = LAStools ran unlicensed and says its output is deliberately distorted.</p>')


def points_table(idx, size, tools=("pdg_gpu", "pdal_pinned", "lastools", "wrench", "qgis")) -> str:
    headers = ["Job"] + [charts.SHORT[t] for t in tools]
    rows = []
    for t in TASKS:
        if size not in t["sizes"]:
            continue
        cells = [esc(t["name"])]
        any_pts = False
        for tool in tools:
            r = idx.get((size, t["id"], tool))
            pts = None
            if r and r.get("rc", 1) == 0:
                for o in r.get("outputs", []):
                    if o.get("points") is not None:
                        pts = o["points"]
                        break
            if pts is not None:
                any_pts = True
                cells.append(f"{pts:,}")
            else:
                cells.append("—" if r is None or r.get("rc", 1) != 0 else "n/a")
        if any_pts:
            rows.append(cells)
    return table(headers, rows)


def same_bytes_table(idx, size) -> str:
    rows = []
    for t in TASKS:
        if size not in t["sizes"]:
            continue
        a, b = idx.get((size, t["id"], "pdg_gpu")), idx.get((size, t["id"], "pdal_pinned"))
        if not a or not b or a.get("rc", 1) or b.get("rc", 1):
            continue
        sa = [o.get("sha256") for o in a.get("outputs", []) if o.get("kind") == "file"]
        sb = [o.get("sha256") for o in b.get("outputs", []) if o.get("kind") == "file"]
        da = [(o.get("files"), o.get("bytes"), o.get("points"), o.get("sha256")) for o in a.get("outputs", []) if o.get("kind") == "directory"]
        db = [(o.get("files"), o.get("bytes"), o.get("points"), o.get("sha256")) for o in b.get("outputs", []) if o.get("kind") == "directory"]
        if not sa and not da:
            verdict = "no file written (text output)"
        elif sa == sb and da == db:
            verdict = ("yes — identical bytes" if sa or all(v[3] for v in da)
                       else "historical check only — same file count, total size and point count; directory members were not hashed")
        elif t["id"] == "copc":
            verdict = "not byte-identical — historical exploratory result; run the explicit canonical COPC comparator for supplemental semantics"
        else:
            verdict = "NO — the bytes differ"
        # also GPU forced / CPU only
        extra = []
        for tool, name in (("pdg_cpu", "CPU only"), ("pdg_gpu_all", "GPU forced")):
            c = idx.get((size, t["id"], tool))
            if c and not c.get("rc", 1):
                sc = [o.get("sha256") for o in c.get("outputs", []) if o.get("kind") == "file"]
                dc = [(o.get("files"), o.get("bytes"), o.get("points"), o.get("sha256")) for o in c.get("outputs", []) if o.get("kind") == "directory"]
                if sa or da:
                    extra.append(f"{name}: {'same' if (sc == sa and dc == da) else 'differs'}")
        rows.append([esc(t["name"]), verdict, ", ".join(extra)])
    return table(["Job", "Same output as PDAL 2.10.0?", "GPUPDAL's other modes vs its automatic mode"], rows)


def build_charts(docs: dict) -> dict:
    """Make every chart; return a dict of paths."""
    out = {}
    idxs = {docs[k]["label"]: charts.index(docs[k]) for k in docs}
    named = {MACHINES.get(k, (k, k, ""))[0]: v for k, v in idxs.items()}
    if "workstation" in idxs:
        idx = idxs["workstation"]
        out["headline"] = charts.chart_headline({"Workstation (Ryzen 9 7900 + RTX 4090)": idx}, CH / "headline.png")
        for size in ("1m", "4m", "16m", "47m"):
            if not any(k[0] == size for k in idx):
                continue
            for fam in FAMILIES:
                p = charts.chart_family_bars(idx, size, fam, CH / f"ws-{size}-{fam.replace(' ', '_').replace('(', '').replace(')', '')}.png")
                if p:
                    out[f"ws-{size}-{fam}"] = p
            out[f"ws-{size}-speedup"] = charts.chart_speedup(idx, size, CH / f"ws-{size}-speedup.png")
            out[f"ws-{size}-memory"] = charts.chart_memory(idx, size, CH / f"ws-{size}-memory.png")
        out["ws-scaling"] = charts.chart_scaling(idx, CH / "ws-scaling.png")
        out["ws-gpu-vs-cpu"] = charts.chart_gpu_vs_cpu(idx, CH / "ws-gpu-vs-cpu.png")
        report_tool_names = dict(charts.TOOL_NAME)
        report_tool_names.update({
            "pdg_gpu": "GPUPDAL — GPU on (automatic)",
            "pdg_cpu": "GPUPDAL — CPU only (no GPU)",
            "pdg_gpu_all": "GPUPDAL — GPU forced on",
        })
        out["ws-totals"], out["ws-totals-tasks"] = charts.chart_totals(
            idx, CH / "ws-totals.png", tool_names=report_tool_names)
    for label, idx in idxs.items():
        if label == "workstation":
            continue
        short = MACHINES.get(label, (label, label, ""))[1]
        tools = [t for t in TOOL_ORDER if any(k[2] == t for k in idx)]
        for size in ("1m", "4m", "16m", "47m"):
            if not any(k[0] == size for k in idx):
                continue
            for fam in FAMILIES:
                p = charts.chart_family_bars(idx, size, fam, CH / f"{label}-{size}-{fam.replace(' ', '_').replace('(', '').replace(')', '')}.png",
                                             tools=tools, title=f"{fam} — {SIZE_LABEL[size]} points — {short}")
                if p:
                    out[f"{label}-{size}-{fam}"] = p
            out[f"{label}-{size}-speedup"] = charts.chart_speedup(idx, size, CH / f"{label}-{size}-speedup.png",
                                                                  others=[t for t in ("pdal_pinned", "lastools", "wrench") if t in tools],
                                                                  title=f"Speed of GPUPDAL compared with the other tools — {SIZE_LABEL[size]} points — {short}")
        out[f"{label}-scaling"] = charts.chart_scaling(idx, CH / f"{label}-scaling.png", tools=tools,
                                                       title=f"How the time grows with the number of points — {short}")
        out[f"{label}-gpu-vs-cpu"] = charts.chart_gpu_vs_cpu(idx, CH / f"{label}-gpu-vs-cpu.png",
                                                             title=f"{short}: time relative to GPUPDAL with the GPU on (automatic) = 1×")
    if len(idxs) > 1:
        key_tasks = ["compress", "reproject", "ground", "hag", "outlier", "features", "classify_refine", "dsm", "thin", "merge", "tile", "denoise_thin"]
        for size in ("1m", "4m", "16m", "47m"):
            if all(any(k[0] == size for k in v) for v in idxs.values()):
                out[f"cross-{size}"] = charts.chart_cross_machine(
                    {MACHINES.get(k, (k, k, ""))[1]: v for k, v in idxs.items()}, size, key_tasks, CH / f"cross-{size}.png")
    return out


CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
body { font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif; font-size: 10.2pt; line-height: 1.45; color: #0b0b0b; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 22pt; margin: 0 0 4pt 0; }
h2 { font-size: 15pt; margin-top: 22pt; border-bottom: 1.5px solid #c9c8c4; padding-bottom: 3pt; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 16pt; page-break-after: avoid; }
p, li { text-align: left; }
.subtitle { color: #52514e; font-size: 11pt; margin-bottom: 14pt; }
figure { margin: 10pt 0 14pt 0; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 236mm; width: auto !important; height: auto; display: block; margin: 0 auto; }
figcaption { font-size: 8.8pt; color: #52514e; margin-top: 4pt; }
table.data { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin: 8pt 0 12pt 0; page-break-inside: auto; }
table.data th, table.data td { border-bottom: 1px solid #e6e5e1; padding: 2.5pt 4pt; text-align: right; vertical-align: top; }
table.data th { background: #f4f3ef; text-align: right; font-weight: 600; }
table.data th:first-child, table.data td:first-child { text-align: left; }
table.data td.best { font-weight: 700; color: #104281; }
table.plain { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 8pt 0 12pt 0; }
table.plain th, table.plain td { border-bottom: 1px solid #e6e5e1; padding: 4pt 5pt; text-align: left; vertical-align: top; }
table.plain th { background: #f4f3ef; font-weight: 600; }
tr { page-break-inside: avoid; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt; background: #f4f3ef; padding: 0 2pt; }
.box { background: #f4f3ef; border-left: 4px solid #2a78d6; padding: 8pt 10pt; margin: 10pt 0; page-break-inside: avoid; }
.warn { border-left-color: #eb6834; }
.small { font-size: 8.5pt; color: #52514e; }
.pagebreak { page-break-before: always; }
.toc li { margin: 1pt 0; }
.missing { color: #d03b3b; font-size: 8.5pt; }
ul { margin-top: 3pt; }
"""


def build_html(docs: dict, ch: dict) -> str:
    idxs = {k: charts.index(v) for k, v in docs.items()}
    ws = idxs.get("workstation")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    parts = []
    A = parts.append

    # ---------- numbers for the summary
    def gm_vs(idx, size, other, base="pdg_gpu"):
        rs = []
        for t in TASKS:
            b, v = median(idx.get((size, t["id"], base))), median(idx.get((size, t["id"], other)))
            if b and v:
                rs.append(v / b)
        return geo_mean(rs), len(rs), (min(rs) if rs else None), (max(rs) if rs else None)

    summary_rows = []
    if ws:
        for size in ("1m", "4m", "16m", "47m"):
            if not any(k[0] == size for k in ws):
                continue
            row = [f"{SIZE_LABEL[size]} points"]
            for o in ("pdal_pinned", "pdal_sys", "lastools", "wrench", "qgis"):
                g, n, lo, hi = gm_vs(ws, size, o)
                row.append(f"{g:.1f}× (from {lo:.1f}× to {hi:.1f}×, {n} jobs)" if g else "—")
            summary_rows.append(row)

    A(f"""
<h1>How fast is GPUPDAL? A separate comparative speed test against PDAL, LAStools and QGIS</h1>
<p class="subtitle">Report date {now}. Author-produced by an automated run on this repository (branch <code>p3-skewness-resident</code>, commit <code>cc12b04eb</code>). It is separate from the maintained reference suite, not independent third-party validation.</p>

<div class="box">
<p><b>What this report is.</b> We took the program in this repository, now called <b>GPUPDAL</b>, and timed it doing everyday lidar point-cloud jobs. The captured raw results use its historical internal executable identifier, <code>pdg</code>; this regenerated report uses the current GPUPDAL product label without changing any benchmark value. We timed the same jobs with three other well-known tools: <b>PDAL</b>, <b>LAStools</b>, and <b>QGIS</b>. We used four point clouds of different sizes (1, 4, 16 and 47 million points) and three different computers (a fast desktop with a big graphics card, a cloud server with a data-centre GPU, and an older budget machine with a small GPU). This report shows the timings as tables and charts, shows what the inputs and outputs look like, and explains where the comparison is fair and where it is not.</p>
<p><b>Evidence boundary.</b> This is one capture per machine with warm-cache medians of 3 timed repeats at 1M/4M, 2 at 16M, and 1 at 47M. The three-machine bundle does not establish the separate ten-GPU claim, a 3DEP population claim, production readiness, or third-party validation.</p>
</div>

<h2>Contents</h2>
<ol class="toc">
<li>The short version</li>
<li>What was tested: the tools, the computers, the point clouds, the jobs, and how we timed them</li>
<li>Results on the workstation (1 million points, then bigger files)</li>
<li>With and without the graphics card</li>
<li>Results on the other computers</li>
<li>What the outputs look like</li>
<li>Are the outputs the same?</li>
<li>Things to keep in mind (caveats)</li>
<li>Appendix: every number, every command, how to repeat this</li>
</ol>

<h2>1. The short version</h2>
""")
    if ws:
        def ratio(size, task, other, base="pdg_gpu"):
            b, v = median(ws.get((size, task, base))), median(ws.get((size, task, other)))
            return (v / b) if (b and v) else None

        def wins(size, other):
            n = w = 0
            for t in TASKS:
                r = ratio(size, t["id"], other)
                if r is not None:
                    n += 1
                    w += r > 1.0
            return w, n

        g_pdal, n_pdal, lo, hi = gm_vs(ws, "1m", "pdal_pinned")
        g_pdal47 = gm_vs(ws, "47m", "pdal_pinned")[0] if any(k[0] == "47m" for k in ws) else None
        w_pdal, _ = wins("1m", "pdal_pinned")
        nb = [ratio("1m", t, "pdal_pinned") for t in ("outlier", "denoise_thin", "classify_refine", "features")]
        nb = [x for x in nb if x]
        fj = [ratio("1m", t, "pdal_pinned") for t in ("compress", "decompress", "reproject", "merge", "tile", "clip", "copc")]
        fj = [x for x in fj if x]
        g_lt, n_lt, lo_lt, hi_lt = gm_vs(ws, "1m", "lastools")
        w_lt, _ = wins("1m", "lastools")
        g_wr, n_wr, lo_wr, hi_wr = gm_vs(ws, "1m", "wrench")
        g_q, n_q, lo_q, hi_q = gm_vs(ws, "1m", "qgis")
        g_cpu, n_cpu, _, _ = gm_vs(ws, "1m", "pdg_cpu")
        feat_cpu = ratio("1m", "features", "pdg_cpu")
        q_start = median(ws.get(("_", "qgis_startup", "qgis"))) or 1.2
        forced = {}
        for size in ("4m", "16m", "47m"):
            r = ratio(size, "features", "pdg_gpu_all")
            if r:
                forced[size] = 1 / r
        big = {t: ratio("47m", t, "pdal_pinned") for t in ("outlier", "classify_refine", "features", "denoise_thin")}
        big = {k: v for k, v in big.items() if v}
        other_bits = []
        for lab in ("a100-epyc", "rtx3060-xeon"):
            if lab not in idxs:
                continue
            oi = idxs[lab]
            g1 = gm_vs(oi, "1m", "pdal_pinned")[0]
            big_size = next((sz for sz in ("47m", "16m", "4m") if any(k[0] == sz for k in oi)), None)
            gb = gm_vs(oi, big_size, "pdal_pinned")[0] if big_size else None
            other_bits.append(f"on the {MACHINES[lab][1]} PDAL took on average {g1:.1f}× longer than GPUPDAL at 1 million points" +
                              (f" and {gb:.1f}× at {SIZE_LABEL[big_size]}" if gb else ""))
        other_summary = ("; ".join(other_bits) + ".") if other_bits else "(no other-machine results were available)."
        s47_pdal = median(ws.get(("47m", "features", "pdal_pinned")))
        s47_pdg = median(ws.get(("47m", "features", "pdg_gpu")))
        s47_pdg_f = median(ws.get(("47m", "features", "pdg_gpu_all")))
        A(f"""
<ul>
<li><b>GPUPDAL was the fastest tool, or tied for fastest, on almost every job at every size we tried.</b> On the workstation with 1 million points, PDAL 2.10.0 (the program GPUPDAL is based on) took on average <b>{g_pdal:.1f} times longer</b> than GPUPDAL over {n_pdal} jobs (from {lo:.1f}× to {hi:.1f}× depending on the job); GPUPDAL was faster on {w_pdal} of the {n_pdal}. The gains are biggest on jobs that look at each point's neighbours — noise removal, surface features, classification clean-up — where PDAL took {min(nb):.0f} to {max(nb):.0f} times longer, and smallest on jobs that are mostly reading and writing files, where PDAL took {min(fj):.1f} to {max(fj):.1f} times longer.{f' On the 47-million-point tile PDAL took on average {g_pdal47:.1f} times longer.' if g_pdal47 else ''}</li>
<li><b>Against LAStools</b> (the fast, partly commercial tools), GPUPDAL was faster on {w_lt} of {n_lt} comparable jobs at 1 million points; averaged over those jobs LAStools took {g_lt:.1f} times longer (from {lo_lt:.1f}× to {hi_lt:.1f}×). Its best case against GPUPDAL was the point-density map. LAStools often uses a different method for a job of the same name, and its paid tools ran unlicensed, so read those rows as "same task, different program", not as a like-for-like race.</li>
<li><b>Against QGIS</b>, GPUPDAL was much faster: the QGIS engine (pdal_wrench) took on average {g_wr:.1f} times longer at 1 million points, and QGIS run through <code>qgis_process</code> took {g_q:.1f} times longer, partly because QGIS itself needs about {q_start:.1f} seconds to start. On big files the QGIS engine's tiling job (which writes uncompressed tiles from many threads) was as fast as GPUPDAL, and its COPC maker (untwine) was the fastest COPC writer of all.</li>
<li><b>The graphics card matters for only a few jobs.</b> On the workstation, GPUPDAL with the GPU hidden took on average {g_cpu:.2f} times as long as with the GPU on — nearly the same. Most of GPUPDAL's speed comes from doing CPU work better (many threads, faster reading and writing, less copying). At 1 million points the GPU made a big difference for surface-normal and shape-feature computation (the CPU-only run took {feat_cpu:.1f} times longer) and nothing for jobs such as compressing or reprojecting.</li>
<li><b>GPUPDAL's automatic mode is cautious — sometimes too cautious.</b> On files of 4 million points and more, GPUPDAL's automatic mode did not use the GPU at all on this workstation, and its times matched the CPU-only run. When we forced the GPU on, the surface-features job ran {" / ".join(f"{v:.1f}×" for v in forced.values())} faster at {" / ".join(SIZE_LABEL[k] for k in forced)} points, and the height-above-ground jobs also sped up. So there is real GPU speed that the shipped decision table leaves unused on big files. Forcing the GPU everywhere is not free either: on some jobs (tiling, DSM) it made things slower.</li>
<li><b>Bigger files, same story.</b> The speed advantage held or grew as the files grew from 1 to 47 million points. On the 47-million-point tile the largest differences were on the neighbourhood jobs, where PDAL took {min(big.values()):.1f} to {max(big.values()):.0f} times longer (for example surface features: PDAL {s47_pdal / 60:.1f} minutes, GPUPDAL {s47_pdg / 60:.1f} minutes automatic, {s47_pdg_f:.0f} seconds with the GPU forced on).</li>
<li><b>Other computers.</b> The same pattern appeared on the two rented machines (section 5): {other_summary} The advantage over PDAL is smaller on the budget machine, which has fewer CPU threads for GPUPDAL to use, and the "features job only uses the GPU automatically at 1 million points" behaviour was the same on all three machines.</li>
<li><b>Outputs.</b> Deterministic file outputs were checked byte-for-byte against PDAL 2.10.0. COPC bytes differed in this historical harness and are reported as non-identical, not waived as a conformance pass; the versioned canonical comparator now supplies separate semantic evidence. LAStools and QGIS produce different results for several jobs because they use different methods.</li>
</ul>
""")
        A(img(ch.get("headline"), "Figure 1. For each file size, how many times longer each tool took than GPUPDAL with the GPU on, averaged (geometric mean) over the jobs the two tools can both do (18 to 24 jobs, see Table 1). 1× would mean the same speed. Workstation."))
        A(table(["File size", "PDAL 2.10.0 (built)", "PDAL 2.10.1 (package)", "LAStools", "QGIS engine (pdal_wrench)", "QGIS (qgis_process)"], summary_rows, cls="plain"))
        A('<p class="small">Table 1. How many times longer each tool took than GPUPDAL (GPU on, automatic), averaged over the jobs both tools can do, with the best and worst single job in brackets. Workstation.</p>')

    # ---------- section 2
    A('<h2 class="pagebreak">2. What was tested</h2><h3>2.1 The tools</h3>')
    A('<p>All eight columns in the charts are listed here. Three of them are the same GPUPDAL program in three modes, and two are PDAL in two builds, so there are really four products: GPUPDAL, PDAL, LAStools and QGIS.</p>')
    A(table(["Column in charts", "What it is", "How it was run"],
            [[f'<span style="display:inline-block;width:10px;height:10px;background:{TOOL_COLOR[k]};margin-right:4px"></span><b>{n}</b>', d, h] for k, n, d, h in TOOL_ROWS], cls="plain"))
    A('<p class="small">Versions in the run: ' + esc(docs["workstation"]["machine"]["versions"]["pdal_sys"].splitlines()[1] if "workstation" in docs else "") + '; ' +
      esc(docs["workstation"]["machine"]["versions"]["wrench"] if "workstation" in docs else "") + '; ' +
      esc(docs["workstation"]["machine"]["versions"]["qgis"].splitlines()[0] if "workstation" in docs else "") + '; ' +
      esc(docs["workstation"]["machine"]["versions"]["lastools"] if "workstation" in docs else "") + '.</p>')

    A('<h3>2.2 The computers</h3>')
    rows = []
    for label, doc in docs.items():
        m = doc["machine"]
        name, short, desc = MACHINES.get(label, (label, label, ""))
        rows.append([f"<b>{name}</b>", desc, esc(m.get("cpu", "")), esc(m.get("gpu", "")), f'{m.get("ram_gb", "?")} GB', esc(m.get("cgroup_cpu_max") or "no limit")])
    A(table(["Computer", "Plain description", "CPU", "GPU", "Memory", "CPU limit (container quota)"], rows, cls="plain"))
    A('<p class="small">The two cloud machines were rented by the hour from Vast.ai for this test and destroyed afterwards. On rented machines the container is only allowed a share of the CPU (the "quota" column: 3072000/100000 means about 30.7 CPU threads), while the programs see all of the machine\'s threads, so multi-threaded tools may over-subscribe there. QGIS itself was only installed on the workstation; on the cloud machines only its engine (pdal_wrench, built from source against the same PDAL 2.10.0) was measured. PDAL from a Linux package was measured only on the workstation.</p>')

    A('<h3>2.3 The point clouds</h3>')
    A("""<p>All four inputs are real airborne laser scans (lidar). The three smaller ones are the first 1, 4 and 16 million points of one tile of a 2018 survey of hilly, forested land in New York State (USA); this repository's own benchmark fixtures are cut from the same tile, and, as in those fixtures, the coordinates are treated as if they were in the Dutch national grid (EPSG:28992) — the numbers happen to fall inside that grid's range. This only matters for the reprojection jobs, where the arithmetic is the same whatever the ground truth. Because the points are stored in the order the scanner produced them, the 1- and 4-million-point files are thin east–west strips of the tile, and the 16-million-point file covers the whole 1.5 km × 1.5 km tile. The largest input is a complete public Dutch AHN4 tile (25GN1_01, a built-up area) with 47 million points, colours and near-infrared, covering about 1 km × 1.3 km. Every input was written by PDAL 2.10.0 (compressed LAZ, LAS 1.4, point format 7 or 8, coordinate system stamped in the header) so that all tools read exactly the same bytes. For jobs that start from a "ground-classified" file, PDAL 2.10.0's SMRF ground filter was run once beforehand and its output was the input for every tool.</p>""")
    rows = []
    for size in ("1m", "4m", "16m", "47m"):
        p = WORK / "in" / f"{size}-extent.json"
        if p.exists():
            e = json.load(open(p))
            w, h = e["maxx"] - e["minx"], e["maxy"] - e["miny"]
            laz = (WORK / "in" / f"{size}.laz")
            las = (WORK / "in" / f"{size}.las")
            rows.append([f"{SIZE_LABEL[size]} points", f"{e['points']:,}", f"{w:,.0f} m × {h:,.0f} m", f"{e['points'] / (w * h):.1f} points/m²",
                         f"{laz.stat().st_size / 1e6:,.0f} MB" if laz.exists() else "?", f"{las.stat().st_size / 1e6:,.0f} MB" if las.exists() else "?"])
    A(table(["Input", "Points", "Area covered", "Density", "LAZ size", "LAS size"], rows, cls="plain"))
    A(img(RENDERS / "input-1m-height.png", "Figure 2. The 1-million-point input, seen from above, coloured by height. It is a thin east–west strip (1.5 km × 77 m), shown here in three pieces."))
    A(img(RENDERS / "input-1m-3d.png", "Figure 3. A 300 m long piece of the same cloud seen from the side: buildings, trees and ground are visible."))
    A(img(RENDERS / "input-16m-height.png", "Figure 4. The 16-million-point input: the full 1.5 km × 1.5 km area."))
    A(img(RENDERS / "input-47m-height.png", "Figure 5. The 47-million-point input: a complete AHN4 tile."))

    A('<h3>2.4 The jobs</h3>')
    A('<p>We chose jobs that lidar users do all the time. Each job is described in plain words below, with the exact way each tool was asked to do it. Where a tool cannot do a job it is left out for that job. Where a tool does the job with a different method (so the result is not the same), that is noted — the timing is still shown because a user would still use that tool for that job, but the comparison is then "same task, different method".</p>')
    rows = []
    for fam in FAMILIES:
        for t in TASKS:
            if t["family"] != fam:
                continue
            can = []
            for tool in ("pdg_gpu", "pdal_pinned", "lastools", "wrench", "qgis"):
                if t["tools"] and tool not in t["tools"]:
                    continue
                r = ws.get(("1m", t["id"], tool)) if ws else None
                if r is not None:
                    can.append(charts.SHORT[tool])
            notes = []
            for tool in ("pdal_pinned", "lastools", "wrench", "qgis"):
                r = ws.get(("1m", t["id"], tool)) if ws else None
                if r and r.get("note"):
                    notes.append(f"{charts.SHORT[tool]}: {r['note']}")
            rows.append([f"<b>{esc(fam)}</b>", f"<b>{esc(t['name'])}</b>", esc(t["plain"]), ", ".join(can), "; ".join(esc(n) for n in notes)])
    A(table(["Family", "Job", "What it means", "Tools that can do it", "Notes on differences"], rows, cls="plain"))

    A('<h3>2.5 How we timed things</h3>')
    A("""<ul>
<li>Every job was run as a complete program from the command line, and we measured <b>wall-clock time</b> from start to finish, including program start-up, reading the input, doing the work, and writing the output. This is what a user experiences.</li>
<li>Each job was first run once as a warm-up (not counted), then <b>3 times</b> for the 1- and 4-million-point files, <b>2 times</b> for 16 million, and <b>once</b> for 47 million (those runs take minutes). We report the <b>median</b> of the timed runs. The input files were already in the computer's memory cache (a "warm" run), which is the common case when you work on a file repeatedly.</li>
<li>For jobs that need two commands with a tool that has no pipeline (for example, "find the ground, then heights" with LAStools or QGIS), the two commands were run one after the other and the total time was measured.</li>
<li>We also recorded the <b>peak memory</b> (RSS) of the whole job and checked every output: file size, number of points, and a checksum. For GPUPDAL versus PDAL the checksums show whether the outputs are byte-for-byte identical.</li>
<li>Nothing else was running on the workstation during the timed runs. On the rented machines the runs happened inside a container with a CPU quota (see 2.2).</li>
<li>All numbers are from one benchmark run per computer. Repeats varied by a few per cent for the short jobs; the raw times of every repeat are in the appendix data file.</li>
</ul>""")

    # ---------- section 3: workstation
    if ws:
        A('<h2 class="pagebreak">3. Results on the workstation</h2>')
        A('<h3>3.1 One million points: every job, every tool</h3>')
        A('<p>The charts below show the time for each job on a logarithmic scale (each grid line is 10× the previous one), grouped by job family. Shorter bars are better. The number at the end of each bar is the median time in seconds. "Cannot do this job" means the tool has no feature for it.</p>')
        for fam in FAMILIES:
            p = ch.get(f"ws-1m-{fam}")
            if p:
                A(img(p, f"{fam}: median time in seconds, 1 million points, workstation."))
        A('<h3>3.2 The same numbers as a table (seconds; the fastest in each row is in bold blue)</h3>')
        A(results_table(ws, "1m"))
        A('<h3>3.3 How many times faster is GPUPDAL?</h3>')
        A('<p>The next chart divides each other tool\'s time by GPUPDAL\'s time (GPU on, automatic). Bars to the right of the 1× line mean the other tool was slower; bars to the left mean it was faster.</p>')
        A(img(ch.get("ws-1m-speedup"), "Figure. Other tool's time ÷ GPUPDAL's time, 1 million points, workstation."))
        A('<h3>3.4 Bigger files: 4, 16 and 47 million points</h3>')
        A('<p>The full charts for the larger files are in the appendix; here are the summary tables and the scaling chart. As files grow, the jobs that look at neighbours (noise, features, classification clean-up) grow fastest for PDAL and QGIS.</p>')
        for size in ("4m", "16m", "47m"):
            if any(k[0] == size for k in ws):
                A(f'<h4>{SIZE_LABEL[size]} points (seconds)</h4>')
                A(results_table(ws, size))
        A(img(ch.get("ws-scaling"), "Figure. Time versus number of points for every job, all tools (both axes logarithmic). A straight line means the time grows in proportion to the number of points; a steeper line means it grows faster than that."))
        if ch.get("ws-totals"):
            count = len(ch.get("ws-totals-tasks", []))
            A(img(ch.get("ws-totals"), f"Figure. Aggregate wall-clock time to run, one after another, the {count} jobs that every tool completed at every size. The y-axis is linear; each bar is the sum of those jobs' median times. The GPUPDAL series retain the historical raw-result identifier pdg; the benchmark values are unchanged."))
        A('<h3>3.5 Memory</h3>')
        A('<p>Peak memory of the whole job. GPUPDAL uses about the same memory as PDAL for most jobs, sometimes about a gigabyte more (it keeps extra copies of the data for its threads), and clearly more when the GPU is forced on for the features job (the GPU path stages the whole cloud in pinned memory). LAStools uses far less memory than everyone else because it streams points through instead of loading them all. QGIS needs about 380 MB just to start.</p>')
        for size in ("47m", "16m", "1m"):
            if ch.get(f"ws-{size}-memory"):
                A(img(ch[f"ws-{size}-memory"], f"Figure. Peak memory (RSS) per job, {SIZE_LABEL[size]} points, workstation."))
                break

        # ---------- section 4: GPU vs CPU
        A('<h2 class="pagebreak">4. With and without the graphics card</h2>')
        A("""<p>GPUPDAL can use an NVIDIA graphics card (GPU), but it does not always do so. In its normal, automatic mode it uses a built-in table of measurements to decide, per job and per file size, whether the GPU would help. To see what the GPU is really worth, we ran GPUPDAL three ways on the same computer: automatic (the default), with the GPU hidden (as if the machine had none), and with the GPU forced on for every job that has a GPU path.</p>
<p>The chart below shows the time of the "CPU only" and "GPU forced" runs relative to the automatic run. A bar at 1× means no difference. Bars to the right mean that mode was slower than automatic; bars to the left mean it was faster (in other words, the automatic choice was not the best one for that job).</p>""")
        A(img(ch.get("ws-gpu-vs-cpu"), "Figure. GPUPDAL CPU-only and GPU-forced time divided by GPUPDAL automatic time, per job and file size, workstation (RTX 4090)."))
        A("""<p><b>How to read this.</b> Most jobs sit at 1×: GPUPDAL does not use the GPU for them at all, so hiding it changes nothing. The jobs where the GPU makes a real difference are the neighbourhood jobs, above all "Surface normals and shape features". Where the "GPU forced" bar is far to the right (for example the DSM and tiling jobs), forcing the GPU on is a bad idea — moving the data to the card costs more than it saves — and the automatic mode was right not to use it. Where a bar is to the left of 1×, the automatic mode left some speed on the table for that job on this machine.</p>
<div class="box warn"><p><b>Two things we checked more closely.</b></p>
<p><b>1. Why did the automatic mode stop using the GPU for the features job above 1 million points?</b> We asked the program for its own decision record (<code>pdg-engine resident … --stats</code>). At 1 million points it reports "device faster" and runs on the GPU; at 4 million it reports <code>mixed_calibration_models</code> and runs on the CPU. The reason is in the source: for this job with the "write all extra values" output, the automatic GPU route is only allowed for a short list of exact input layouts that were measured beforehand — a 1,000,000-point, point-format-7 compressed file; a point-format-6 compressed file; and a point-format-8 compressed file with 44-byte records (the original AHN4 layout). Our 4- and 16-million-point files are point-format-7 with other counts, and our 47-million-point file was re-written by PDAL with 38-byte records, so none of them qualifies, and GPUPDAL deliberately stays on the CPU rather than trust a speed model it did not measure for that layout. This is a cautious design choice, but for a user it means the GPU speed for this job (5–6× at these sizes) is only reached automatically for particular files, or by forcing the GPU on.</p>
<p><b>2. Why is "Find noise points" a little slower with the GPU on?</b> Re-timing it eight times gave 0.55–0.61 s with the GPU visible and 0.35–0.37 s with it hidden at 1 million points; the difference (about 0.2 s, and 0.2–0.5 s at every size) is a fixed start-up cost of the GPU path that this job does not win back. It is a small absolute loss.</p></div>""")

    # ---------- section 5: other machines
    others = [k for k in docs if k != "workstation"]
    A('<h2 class="pagebreak">5. Results on the other computers</h2>')
    if others:
        A('<p>The same harness, the same inputs and the same commands were run on two rented cloud machines. Only the tools that could be built there were run: GPUPDAL (three modes), PDAL 2.10.0 built from source, LAStools, and the QGIS engine pdal_wrench built from source. Times are not directly comparable across machines when the CPUs differ (and the containers have CPU quotas), but the ratios between tools on the same machine are.</p>')
        for label in others:
            name, short, desc = MACHINES.get(label, (label, label, ""))
            idx = idxs[label]
            A(f'<h3>{name}</h3><p>{desc}</p>')
            for size in ("1m", "47m", "16m", "4m"):
                if any(k[0] == size for k in idx):
                    A(f'<h4>{SIZE_LABEL[size]} points (seconds)</h4>')
                    A(results_table(idx, size, tools=[t for t in TOOL_ORDER if any(k[2] == t for k in idx)]))
            if ch.get(f"{label}-1m-speedup"):
                A(img(ch[f"{label}-1m-speedup"], f"Figure. Other tool's time ÷ GPUPDAL's time, 1 million points, {short}."))
            if ch.get(f"{label}-scaling"):
                A(img(ch[f"{label}-scaling"], f"Figure. Time versus number of points, {short}."))
            if ch.get(f"{label}-gpu-vs-cpu"):
                A(img(ch[f"{label}-gpu-vs-cpu"], f"Figure. GPUPDAL CPU-only and GPU-forced time relative to automatic, {short}."))
        A('<h3>Side by side: the same jobs on all three computers</h3>')
        for size in ("1m", "47m", "16m", "4m"):
            if ch.get(f"cross-{size}"):
                A(img(ch[f"cross-{size}"], f"Figure. Selected jobs on all three computers, {SIZE_LABEL[size]} points (seconds, log scale)."))
    else:
        A('<p class="missing">No other-computer results were available when this report was built.</p>')

    # ---------- section 6: outputs
    A('<h2 class="pagebreak">6. What the outputs look like</h2>')
    A('<p>These pictures are made from the actual output files of the 1-million-point runs on the workstation. Where tools use different methods, their outputs are shown next to each other.</p>')
    A(img(RENDERS / "out-1m-ground-compare.png", "Figure. \"Find the ground\": which points each tool labelled as ground (green). GPUPDAL and PDAL give exactly the same answer (SMRF); QGIS also uses SMRF; LAStools uses its own method and finds a different set."))
    A(img(RENDERS / "out-1m-hag.png", "Figure. \"Height above ground\": the tallest thing in each 1 m cell (GPUPDAL / PDAL output)."))
    A(img(RENDERS / "out-1m-dtm.png", "Figure. \"Bare-earth elevation map (DTM)\" from GPUPDAL / PDAL, hill-shaded."))
    A(img(RENDERS / "out-1m-dtm-lastools.png", "Figure. The same DTM job done by LAStools las2dem (triangle mesh) — smoother because it interpolates across gaps."))
    A(img(RENDERS / "out-1m-dtm-wrench.png", "Figure. The same DTM job done by QGIS / pdal_wrench (average ground height per cell)."))
    A(img(RENDERS / "out-1m-dsm.png", "Figure. \"Surface elevation map (DSM)\": highest first return per 1 m cell (GPUPDAL / PDAL output)."))
    A(img(RENDERS / "out-1m-density.png", "Figure. \"Point density map\": points per 1 m cell (GPUPDAL / PDAL output)."))
    A(img(RENDERS / "out-1m-thin-compare.png", "Figure. Thinning: a 40 m square of the input and the outputs of the thinning jobs. Poisson sampling keeps points that are at least 1 m apart in 3-D (so walls and tree crowns keep more points); LAStools' lasthin keeps one point per 1 m ground cell; the 2.5 m job keeps one point per 2.5 m cell."))
    A(img(RENDERS / "out-1m-clip.png", "Figure. \"Clip with a polygon\": the kept points (coloured) and the polygon (red). Note the hole in the left rectangle. LAStools' lasclip kept the points inside the hole too (see the point counts in section 7)."))
    A(img(RENDERS / "out-1m-tiles.png", "Figure. \"Split into 256 m tiles\": one colour per output file."))
    A(img(RENDERS / "out-1m-features.png", "Figure. \"Surface normals and shape features\": two of the computed features, verticality and planarity, averaged per 1 m cell."))
    A(img(RENDERS / "out-1m-colorize.png", "Figure. \"Colour the points from an aerial image\": the test image is a synthetic colour gradient in Web Mercator covering the middle half of the strip, so the points there get colours and the rest stay black."))
    A(img(RENDERS / "out-1m-outlier-compare.png", "Figure. \"Find noise points\": the points labelled noise (red) by GPUPDAL / PDAL (statistical outliers) and by LAStools lasnoise (isolated points). Different methods, different answers."))
    A(img(RENDERS / "out-1m-boundary.png", "Figure. \"Outline of the covered area\" as drawn by the QGIS engine (hexagon bins) and by LAStools (concave hull)."))

    # ---------- section 7: same outputs?
    A('<h2 class="pagebreak">7. Are the outputs the same?</h2>')
    if ws:
        A('<p>GPUPDAL default mode preserves PDAL 2.10.0 semantics. We checksum deterministic outputs byte-for-byte; inherently nondeterministic containers such as COPC require the separately versioned canonical comparison. The table also shows whether GPUPDAL\'s CPU-only and GPU-forced modes gave the same deterministic file as its automatic mode.</p>')
        A(same_bytes_table(ws, "1m"))
        # larger sizes: summarise
        lines = []
        for size in ("4m", "16m", "47m"):
            if not any(k[0] == size for k in ws):
                continue
            diffs, unproved_directories, n = [], [], 0
            for t in TASKS:
                a, b = ws.get((size, t["id"], "pdg_gpu")), ws.get((size, t["id"], "pdal_pinned"))
                if not a or not b or a.get("rc", 1) or b.get("rc", 1):
                    continue
                sa = [o.get("sha256") for o in a.get("outputs", []) if o.get("kind") == "file"]
                sb = [o.get("sha256") for o in b.get("outputs", []) if o.get("kind") == "file"]
                da = [(o.get("files"), o.get("bytes"), o.get("points"), o.get("sha256")) for o in a.get("outputs", []) if o.get("kind") == "directory"]
                db = [(o.get("files"), o.get("bytes"), o.get("points"), o.get("sha256")) for o in b.get("outputs", []) if o.get("kind") == "directory"]
                if not sa and not da:
                    continue
                n += 1
                if da and not all(value[3] for value in da + db):
                    unproved_directories.append(t["name"])
                elif sa != sb or da != db:
                    diffs.append(t["name"])
            proven = n - len(diffs) - len(unproved_directories)
            detail = []
            if diffs:
                detail.append(f"different: {', '.join(diffs)}")
            if unproved_directories:
                detail.append("historical directory runs checked only by file count, total size and point count: "
                              + ", ".join(unproved_directories))
            lines.append(f"{SIZE_LABEL[size]} points: {proven} of {n} file-writing jobs proved byte-identical" +
                         (f" ({'; '.join(detail)})" if detail else ""))
        A("<p>The same historical check on the larger files: " + "; ".join(lines) + ". A COPC byte mismatch remains a mismatch; semantic equivalence now requires the separately versioned canonical comparator.</p>")
        A('<p>The other tools do not promise the same output, and often use a different method for the same job. The number of points in each output shows where they differ:</p>')
        A(points_table(ws, "1m"))
        A("""<p class="small">Notes: "Thin" — Poisson sampling in 3-D keeps more points than one-per-cell thinning; QGIS' engine also uses Poisson sampling but its parallel, tiled implementation keeps a different set. "Clip" — LAStools' lasclip ignored the hole in the polygon. "Find the ground" and "noise" — all tools keep every point but label them differently (see the pictures). "Crop, then reproject" — the same 291,617 points in every tool.</p>""")

    # ---------- section 8: caveats
    A('<h2 class="pagebreak">8. Things to keep in mind</h2>')
    A("""<ul>
<li><b>Same job, different method.</b> GPUPDAL and PDAL run the very same algorithms (GPUPDAL is a modified PDAL). QGIS' engine is also built on the PDAL library, so its methods are the same or close (SMRF ground, statistical outliers, Poisson thinning, hexagon boundary), but it splits work into tiles and threads. LAStools uses its own algorithms for ground, height, noise, thinning and rasters, so its times are for a job of the same name, not the same computation.</li>
<li><b>LAStools ran unlicensed.</b> The paid LAStools programs were used in their free test mode. Above 1.5–10 million points (the limit differs per tool) they print a warning that the output is deliberately distorted ("tiny xyz noise, points permuted, some fields zeroed", or a black diagonal drawn across a raster) and exit with an error code. The historical capture counted those timings and marked them "ran unlicensed"; they are exploratory, nonqualifying observations. The current harness preserves the error status. The free tools (laszip, las2las, lasmerge) are not affected.</li>
<li><b>QGIS start-up.</b> Every <code>qgis_process</code> call pays about 1.2 s to start QGIS and its Python plugins. For a user in the QGIS desktop this cost is paid once, not per job, so for long jobs "QGIS engine (pdal_wrench)" is the fairer number and for short jobs the truth is in between.</li>
<li><b>Threads.</b> PDAL and LAStools are single-threaded for these jobs. GPUPDAL and pdal_wrench use many CPU threads (up to 24 on the workstation). So part of "GPUPDAL is faster" is "GPUPDAL uses the whole CPU"; that is a real advantage for the user, but it also means the gap shrinks on machines with few cores, and grows on machines with many. On the rented machines the container had a CPU quota smaller than the number of visible threads.</li>
<li><b>Warm cache, one machine per type, few repeats for big files.</b> Inputs were in memory cache; the 47-million-point runs were timed once. Times on the two rented machines depend on which physical host we got.</li>
<li><b>PDAL versions.</b> GPUPDAL is based on PDAL 2.10.0. PDAL 2.10.1 from the Linux package was a little slower than the source build in most jobs, most likely because of different compiler settings, not because of the version.</li>
<li><b>Peak memory</b> is the largest resident memory of the whole job's process tree; for QGIS it includes QGIS itself.</li>
<li><b>This is a separate repository-run harness, not independent validation.</b> Its task definitions and inputs differ from the maintained <code>BENCHMARKS.md</code> suite, but the evidence was still produced by the project author against this repository.</li>
</ul>""")

    # ---------- appendix
    A('<h2 class="pagebreak">9. Appendix</h2>')
    A('<h3>9.1 Every chart for the larger files (workstation)</h3>')
    for size in ("4m", "16m", "47m"):
        for fam in FAMILIES:
            p = ch.get(f"ws-{size}-{fam}")
            if p:
                A(img(p, f"{fam}: median time in seconds, {SIZE_LABEL[size]} points, workstation."))
        if ch.get(f"ws-{size}-speedup"):
            A(img(ch[f"ws-{size}-speedup"], f"Other tool's time ÷ GPUPDAL's time, {SIZE_LABEL[size]} points, workstation."))
    for label in others:
        name, short, desc = MACHINES.get(label, (label, label, ""))
        A(f'<h3>9.2 Every chart for {name}</h3>')
        for size in ("1m", "4m", "16m", "47m"):
            for fam in FAMILIES:
                p = ch.get(f"{label}-{size}-{fam}")
                if p:
                    A(img(p, f"{fam}: median time in seconds, {SIZE_LABEL[size]} points, {short}."))
    A('<h3>9.3 The exact commands (1 million points, workstation)</h3>')
    rows = []
    if ws:
        for t in TASKS:
            for tool in TOOL_ORDER:
                r = ws.get(("1m", t["id"], tool))
                if not r:
                    continue
                cmds = "<br>".join(esc(" ".join(c)) for c in r["commands"])
                env = " ".join(f"{k}={v!r}" for k, v in r.get("env", {}).items())
                rows.append([esc(t["name"]), charts.SHORT[tool], f"<code>{(env + ' ') if env else ''}{cmds}</code>"])
    A(f'<table class="plain" style="font-size:7pt"><thead><tr><th>Job</th><th>Tool</th><th>Command(s)</th></tr></thead><tbody>' +
      "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>")
    A('<p class="small">Pipeline JSON files (for GPUPDAL and PDAL) are written by the harness; their contents are in <code>bench/independent/tasks.py</code>. Paths were shortened for readability where they only differ by tool name.</p>')
    A('<h3>9.4 How to repeat this</h3>')
    A("""<pre style="font-size:8pt;background:#f4f3ef;padding:8pt">python3 bench/independent/prepare.py            # stage the inputs (needs the local corpus and pinned PDAL)
python3 bench/independent/harness.py --label workstation --sizes 1m --repeats 3 --keep-outputs
python3 bench/independent/harness.py --label workstation --sizes 4m --repeats 3
python3 bench/independent/harness.py --label workstation --sizes 16m --repeats 3
python3 bench/independent/harness.py --label workstation --sizes 47m --repeats 3
python3 bench/independent/render.py                # pictures of inputs and outputs
python3 bench/independent/report.py                # charts, HTML and PDF
# on a rented box, after bench/remote/vast_bootstrap.sh:
bash bench/independent/remote_setup.sh &lt;label&gt; lastools.tar.gz /workspace/fix 1m 4m 16m 47m</pre>
<p class="small">Raw results (every repeat, every output's size, point count and checksum): <code>build/independent-bench/results/*.json</code>. The published bundle's <code>manifest.json</code> hashes all three result files and every staged input and records that this was author-run evidence.</p>""")
    return "<!doctype html><html><head><meta charset='utf-8'><title>GPUPDAL separate comparative benchmark</title><style>" + CSS + "</style></head><body>" + "\n".join(parts) + "</body></html>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    docs = load_all()
    ch = {} if a.no_charts else build_charts(docs)
    if a.no_charts:
        # reuse existing files
        for p in CH.glob("*.png"):
            ch[p.stem] = p
    html_text = build_html(docs, ch)
    (REPORT / "pdg-independent-benchmark.html").write_text(html_text)
    print("wrote", REPORT / "pdg-independent-benchmark.html")
    if not a.no_pdf:
        pdf = REPORT / "pdg-independent-benchmark.pdf"
        cmd = ["chromium", "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
               f"--print-to-pdf={pdf}", "--run-all-compositor-stages-before-draw", "--virtual-time-budget=20000",
               str(REPORT / "pdg-independent-benchmark.html")]
        subprocess.run(cmd, check=True, capture_output=True)
        print("wrote", pdf, pdf.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
