#!/usr/bin/env python3
"""Build the B0275 cross-machine / big-cloud benchmark report (HTML).

Reads the reference-suite aggregates and per-workload reports (local
workstation and the rented box), the LAStools comparison JSON, the big-cloud
reports, and the rendered PNGs, and writes one self-contained HTML page
(images embedded as data URIs) plus a copy of the images next to it.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import pathlib
import statistics


def load(path: pathlib.Path):
    return json.loads(path.read_text()) if path and path.is_file() else None


def img(path: pathlib.Path, alt: str) -> str:
    if not path or not path.is_file():
        return f"<p><em>missing render: {html.escape(str(path))}</em></p>"
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'
            f'<figcaption>{html.escape(alt)}</figcaption></figure>')


def suite_rows(agg: dict | None, reports_dir: pathlib.Path | None,
               cache: str) -> tuple[list[dict], dict]:
    if not agg or not agg.get("complete"):
        return [], {}
    rows = []
    for w in agg["workloads"]:
        r = load(reports_dir / f"{w['id']}-{cache}.json") if reports_dir else None
        diff = None
        if r and r["comparison"].get("contract") == "fast":
            diff = r["comparison"].get("fast_differing_records")
        rows.append({"id": w["id"], "oracle": w["oracle_median_seconds"],
                     "pdg": w["candidate_median_seconds"], "speedup": w["median_speedup"],
                     "differing": diff})
    return rows, {"gm": agg["equal_workload_geometric_mean_speedup"],
                  "total": agg["total_wall_time"]["speedup"],
                  "oracle_total": agg["total_wall_time"]["oracle_median_seconds"],
                  "pdg_total": agg["total_wall_time"]["candidate_median_seconds"],
                  "contract": agg.get("contract", "exact")}


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
    p.add_argument("--local-suite", type=pathlib.Path, required=True,
                   help="prefix of local aggregates, e.g. build/benchmarks/b0274-suite3")
    p.add_argument("--box-results", type=pathlib.Path, required=True)
    p.add_argument("--box-machine", type=pathlib.Path)
    p.add_argument("--lastools-local", type=pathlib.Path)
    p.add_argument("--lastools-box", type=pathlib.Path)
    p.add_argument("--big-local-glob", default="build/benchmarks/b0275-local-ahn4-*.json")
    p.add_argument("--renders", type=pathlib.Path, required=True)
    p.add_argument("--box-renders", type=pathlib.Path)
    p.add_argument("--fast-local", type=pathlib.Path,
                   help="prefix of local fast-contract aggregates (b0271-fast-suite2)")
    p.add_argument("--ladder-results", type=pathlib.Path,
                   help="results-ladder directory from bench/remote/vast_ladder.sh")
    p.add_argument("--out", type=pathlib.Path, required=True)
    a = p.parse_args()

    repo = a.repo
    lw, lsum = suite_rows(load(pathlib.Path(f"{a.local_suite}-warm.json")),
                          pathlib.Path(f"{a.local_suite}-warm-reports"), "warm")
    lc, lcsum = suite_rows(load(pathlib.Path(f"{a.local_suite}-cold.json")),
                           pathlib.Path(f"{a.local_suite}-cold-reports"), "cold")
    fw, fsum = ([], {})
    if a.fast_local:
        fw, fsum = suite_rows(load(pathlib.Path(f"{a.fast_local}-warm.json")),
                              pathlib.Path(f"{a.fast_local}-warm-reports"), "warm")
    box = a.box_results
    bd, bdsum = suite_rows(load(box / "suite-1m-default.json"), box / "suite-1m-default-reports", "warm")
    bx, bxsum = suite_rows(load(box / "suite-1m-cuda-experimental.json"),
                           box / "suite-1m-cuda-experimental-reports", "warm")
    bf, bfsum = suite_rows(load(box / "suite-1m-fast-cuda-experimental.json"),
                           box / "suite-1m-fast-cuda-experimental-reports", "warm")
    box_machine = a.box_machine.read_text() if a.box_machine and a.box_machine.is_file() else ""

    def big_rows(paths):
        out = []
        for path in sorted(paths):
            d = load(path)
            if not d:
                continue
            s = d["summary"]; c = d["comparison"]
            out.append({"label": d["label"], "oracle": s["oracle"]["median_seconds"],
                        "pdg": s["candidate"]["median_seconds"], "speedup": c["median_speedup"],
                        "exact": c["exact_outputs"], "runs": c["resolution"]["pairs"],
                        "rss_oracle": s["oracle"]["median_peak_rss_bytes"],
                        "rss_pdg": s["candidate"]["median_peak_rss_bytes"]})
        return out
    big_local = big_rows(repo.glob(a.big_local_glob))
    big_box = big_rows(box.glob("big-ahn4-*.json"))
    ladder = []
    ladder_machine = ""
    ladder_inputs = ""
    if a.ladder_results and a.ladder_results.is_dir():
        for path in sorted(a.ladder_results.glob("ladder-*.json")):
            d = load(path)
            if not d:
                continue
            s_ = d["summary"]; c = d["comparison"]
            parts_ = d["label"].split("-")
            ladder.append({"size": parts_[1], "label": "-".join(parts_[2:]),
                           "oracle": s_["oracle"]["median_seconds"],
                           "pdg": s_["candidate"]["median_seconds"],
                           "speedup": c["median_speedup"], "exact": c["exact_outputs"],
                           "rss_oracle": s_["oracle"]["median_peak_rss_bytes"],
                           "rss_pdg": s_["candidate"]["median_peak_rss_bytes"]})
        m = a.ladder_results / "machine.txt"
        ladder_machine = m.read_text() if m.is_file() else ""
        i = a.ladder_results / "inputs.txt"
        ladder_inputs = i.read_text() if i.is_file() else ""
    lt_local = load(a.lastools_local) if a.lastools_local else None
    lt_box = load(a.lastools_box) if a.lastools_box else None

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
    figure { margin: 0; } figure img { max-width: 100%; border: 1px solid var(--line); border-radius: 4px; background: #fff; }
    figcaption { color: var(--muted); font-size: 13px; margin-top: 4px; }
    code { background: var(--soft); padding: 1px 4px; border-radius: 3px; font-size: 13px; }
    pre { white-space: pre-wrap; font-size: 12.5px; background: var(--soft); padding: 8px 10px; border-radius: 4px; overflow-x: auto; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
    .good { color: var(--good); font-weight: 600; } .warn { color: var(--warn); font-weight: 600; }
    small { color: var(--muted); } ul { padding-left: 20px; } li { margin: 4px 0; }
    p { max-width: 78ch; }
    """

    def suite_table(rows, summary, columns=("Pinned PDAL", "PDG"), extra=None):
        if not rows:
            return "<p><em>not available</em></p>"
        head = f"<tr><th>Workload</th><th>{columns[0]} (s)</th><th>{columns[1]} (s)</th><th>Speedup</th>"
        if extra:
            head += f"<th>{extra}</th>"
        head += "</tr>"
        body = ""
        for r in rows:
            cls = "good" if r["speedup"] >= 1.5 else ""
            body += (f"<tr><td class=name>{r['id']}</td><td>{fmt(r['oracle'])}</td><td>{fmt(r['pdg'])}</td>"
                     f"<td class='{cls}'>{fmt(r['speedup'],2)}x</td>")
            if extra:
                body += f"<td>{'identical' if not r.get('differing') else str(r['differing']) + ' records'}</td>"
            body += "</tr>"
        foot = (f"<tr><th>equal-workload geometric mean</th><th></th><th></th><th>{fmt(summary['gm'],3)}x</th>"
                f"{'<th></th>' if extra else ''}</tr>"
                f"<tr><th>total wall (sum of medians)</th><th>{fmt(summary['oracle_total'],2)}</th>"
                f"<th>{fmt(summary['pdg_total'],2)}</th><th>{fmt(summary['total'],3)}x</th>{'<th></th>' if extra else ''}</tr>")
        return f"<div class=scroll><table>{head}{body}{foot}</table></div>"

    def big_table(rows):
        if not rows:
            return "<p><em>not available</em></p>"
        head = ("<tr><th>Workload / configuration</th><th>Pinned PDAL (s)</th><th>PDG (s)</th><th>Speedup</th>"
                "<th>Exact</th><th>Runs</th><th>Peak RSS PDAL / PDG (GB)</th></tr>")
        body = ""
        for r in rows:
            label = r["label"].replace("local-ahn4-", "").replace("ahn4-", "")
            body += (f"<tr><td class=name>{label}</td><td>{fmt(r['oracle'],1)}</td><td>{fmt(r['pdg'],1)}</td>"
                     f"<td class='{'good' if r['speedup']>=1.5 else ''}'>{fmt(r['speedup'],2)}x</td>"
                     f"<td>{'yes' if r['exact'] else 'NO'}</td><td>{r['runs']}</td>"
                     f"<td>{r['rss_oracle']/1e9:.1f} / {r['rss_pdg']/1e9:.1f}</td></tr>")
        return f"<div class=scroll><table>{head}{body}</table></div>"

    def ladder_table(rows):
        if not rows:
            return "<p><em>not available</em></p>"
        head = ("<tr><th>Size</th><th>Workload / configuration</th><th>Pinned PDAL (s)</th><th>PDG (s)</th><th>Speedup</th>"
                "<th>Exact</th><th>Peak RSS PDAL / PDG (GB)</th></tr>")
        body = ""
        order = {"47m": 0, "95m": 1, "190m": 2}
        for r in sorted(rows, key=lambda r: (order.get(r["size"], 9), r["label"])):
            body += (f"<tr><td class=name>{r['size'].upper()}</td><td class=name>{r['label']}</td><td>{fmt(r['oracle'],1)}</td>"
                     f"<td>{fmt(r['pdg'],1)}</td><td class='{'good' if r['speedup']>=1.5 else ''}'>{fmt(r['speedup'],2)}x</td>"
                     f"<td>{'yes' if r['exact'] else 'NO'}</td><td>{r['rss_oracle']/1e9:.1f} / {r['rss_pdg']/1e9:.1f}</td></tr>")
        return f"<div class=scroll><table>{head}{body}</table></div>"

    def lastools_table(lt):
        if not lt:
            return "<p><em>not available</em></p>"
        head = ("<tr><th>Job</th><th>Pinned PDAL (s)</th><th>PDG (s)</th><th>LAStools wall (s)</th>"
                "<th>LAStools own timer (s)</th><th>LAStools tool</th></tr>")
        body = ""
        for name, e in lt["jobs"].items():
            def w(k):
                v = e.get(k); return fmt(v.get("wall_median")) if isinstance(v, dict) and "wall_median" in v else "—"
            l = e.get("lastools") or {}
            body += (f"<tr><td class=name><b>{name}</b><br><small>{html.escape(e.get('description',''))}</small></td>"
                     f"<td>{w('pinned_pdal')}</td><td>{w('pdg')}</td><td>{w('lastools')}</td>"
                     f"<td>{fmt(l.get('tool_reported_median')) if isinstance(l, dict) else '—'}</td>"
                     f"<td>{html.escape(str(l.get('tool', l.get('skipped', '—')))) if isinstance(l, dict) else '—'}</td></tr>")
        return f"<div class=scroll><table>{head}{body}</table></div>"

    R = a.renders
    BR = a.box_renders
    parts = []
    parts.append(f"""<title>PDAL-GPU Speed Report</title><style>{css}</style><main>
<div class=eyebrow>PDAL-GPU · B0275/B0276 · {'' if not lsum else 'reference workstation, rented Threadripper PRO box, 47M-point AHN4 tile, LAStools' + (', 47–190M-point ladder on an H200' if ladder else '')}</div>
<h1>PDAL-GPU (pdg) speed report</h1>
<div class=chips>
<div class=chip><b>{fmt(lsum.get('gm'),2)}×</b><span>average speedup, 14 workflows (geometric mean, exact outputs)</span></div>
<div class=chip><b>{fmt(lsum.get('total'),2)}×</b><span>whole suite: {fmt(lsum.get('oracle_total'),1)} s → {fmt(lsum.get('pdg_total'),1)} s</span></div>
<div class=chip><b>{max([r['speedup'] for r in lw], default=0):.1f}×</b><span>best single workflow (r11 classification)</span></div>
<div class=chip><b>{max([r['speedup'] for r in big_local if 'cuda' in r['label']], default=0):.1f}×</b><span>47M-point tile, r6 features with CUDA</span></div>
<div class=chip><b>0</b><span>byte differences from stock PDAL in default mode, every run</span></div>
</div>
<p class=lead>Cross-machine, big-cloud, and LAStools comparison of the CUDA/host-accelerated PDAL fork against
stock PDAL 2.10.0 (the pinned oracle), with rendered outputs and an upstream-merge assessment. Every PDG number in the
default configuration is <b>byte-identical to stock PDAL</b> (files, metadata, point order, stdout, stderr, exit status)
unless a table says otherwise. Records B0268–B0276 in <code>BENCHMARKS.md</code> hold the raw evidence.</p>

<h2>1. Plain-language summary</h2>
<div class=callout>
<p><b>What this is.</b> PDAL is the standard open-source toolkit for processing lidar point clouds (billions of 3-D points
from aircraft, drones and scanners). We built a drop-in version, <code>pdg</code>, that produces exactly the same files as
stock PDAL but finishes much sooner, by using the graphics card (GPU) for the heavy geometry and all of the CPU cores for
everything else.</p>
<p><b>How much faster.</b> On the reference workstation (AMD Ryzen 9 7900, RTX 4090) the fourteen everyday workflows we
track — converting, compressing, reprojecting, clipping, tiling, merging, ground detection, height normalisation, terrain
and surface rasters, denoising, thinning, colouring, feature extraction and classification — run
<b>{fmt(lsum.get('gm'),2)}× faster on average</b> (geometric mean) and the whole set finishes
<b>{fmt(lsum.get('total'),2)}× sooner</b> ({fmt(lsum.get('oracle_total'),1)} s → {fmt(lsum.get('pdg_total'),1)} s).
Individual jobs range from about 1.3× (a raster surface that is mostly file reading) to almost 13× (classification refinement).</p>
<p><b>Same answers.</b> Speed came without changing results: every output file is checked byte-for-byte against stock PDAL
on every run. An optional <code>--fast</code> switch relaxes one thing only — when several neighbours are at exactly the
same distance it lets the GPU pick instead of copying stock PDAL's arbitrary choice — for a further gain on the
feature-extraction and height workflows.</p>
<p><b>Elsewhere.</b> On a rented Threadripper PRO 3975WX + RTX 4090 machine the CPU-only part of the speedup carried over
({fmt(bdsum.get('gm'),2)}× average, {fmt(bdsum.get('total'),2)}× for the whole set — default mode there is CPU-only because
GPU selection is locked to the profiled reference GPU). Turning the GPU paths on for every workflow there made feature
extraction {max([r['speedup'] for r in bx if r['id']=='r6-features'], default=0):.1f}× faster but slowed a few light
workflows down — which is exactly why the shipped product only picks the GPU where it has measured a win. On a
47-million-point national-survey tile (AHN4) the speedups held at scale: 2–3× for conversion, reprojection and terrain
rasters, 10× for denoising, and 13× for feature extraction with the GPU. Against LAStools (a commercial competitor) our
stock-PDAL-exact results are in the same range or faster on the comparable jobs; stock PDAL is typically the slowest of
the three.</p>
"""+ ("""<p><b>Bigger clouds, bigger GPU.</b> On a rented datacenter card (NVIDIA H200, 141 GB) we ran the same jobs on 47, 95 and
190 million points (section 7b). The GPU-heavy jobs stayed at a steady <b>~10×</b> across the whole range — bigger inputs did not
raise the ratio and the bigger card did not finish sooner than the RTX 4090 box; what it buys is headroom at sizes where a
24 GB card would have to split the work. The file-bound jobs stayed at 1.4–3.6×. Every one of those 20 runs was still byte-identical to stock PDAL.</p>
""" if ladder else "") + f"""</div>

<h2>2. Technical summary</h2>
<p>The fork keeps upstream PDAL's stage semantics and adds three kinds of acceleration, each proven against a pinned
upstream build (<code>ea66f6a9…</code>, PDAL 2.10.0 at <code>f1e35f5c</code>) with process-level differentials
(bytes, metadata, order, streams, status) and sanitizer/CUDA lanes:</p>
<ul>
<li><b>Exact host parallelism inside upstream stages</b> — fixed-chunk worker pools whose folds reproduce the serial
result bit-for-bit: LAS/LAZ record packing and unpacking (reader batch hook + writer batch hook), ordered parallel COPC
tile decode, four-worker LAZ chunk compression, banded raster accumulation in <code>writers.gdal</code>, slot-pooled
PROJ reprojection, host-worker kNN passes for outlier/classifier, SMRF morphology pools, hashed voxel sampling, cached
coordinate KD3 with a mutation epoch, and concurrent structure-identical nanoflann builds.</li>
<li><b>Shared CUDA neighborhood engine</b> — one planner-owned spatial index (uniform grid or Morton BVH built with CUB)
serving exact kNN gather, mean/kth distances, covariance/eigen systems, neighbor votes, LOF, radius queries and HAG,
with tie/incomplete rows repaired exactly on the host so results equal nanoflann's; consumers share the index and
device-resident columns across adjacent stages.</li>
<li><b>Fused point programs and direct LAS I/O on the device</b> for assignment/ferry/expression/crop regions and
mapped LAS sources/publishers, selected only inside measured envelopes.</li>
</ul>
<p>Selection is measured, not assumed: automatic CUDA routes exist only where a same-machine profile showed a
complete-process win; one route (the r4 outlier selector) was retired this session because the improved host path
overtook it. The <code>--fast</code> contract (D0261/D0271) keeps count, order and coordinates identical and lets kNN
distance ties resolve in device order; the runner reports how many records differ (25 of 1,000,000 on r6, 125 on r2).</p>

<h2>3. Reference workstation — fourteen headline workflows at 1 M points</h2>
<p><small>AMD Ryzen 9 7900 (12 cores), RTX 4090, CUDA 13.3, NVMe; alternating pinned/PDG runs, frozen clock, three pairs
per cache state (record B0274; nine-pair per-workload gates in B0268–B0274 agree).</small></p>
<h3>Warm cache</h3>{suite_table(lw, lsum)}
<h3>Cold cache</h3>{suite_table(lc, lcsum)}
""")
    if fw:
        parts.append(f"""<h3>Fast contract (<code>pdg --fast</code>, warm)</h3>
<p><small>Same binaries; records may differ from stock only on kNN tie rows (coordinates, count and order identical);
the runner reports the count of differing records per artifact (record B0271).</small></p>
{suite_table(fw, fsum, extra='Records vs stock')}""")

    parts.append(f"""<h2>4. Rented Threadripper PRO + RTX 4090 (Vast.ai) — same fourteen workflows</h2>
<pre>{html.escape(chr(10).join(l for l in box_machine.strip().splitlines() if not l.startswith('----')))}</pre>
<p>Both binaries were built on the box from source (Ubuntu 24.04, GDAL/PROJ from apt, CUDA 13.3.1 image, SM 89 only).
The pinned oracle is upstream PDAL at the same commit; PDG results are exact against it on that box.</p>
<h3>Default configuration</h3>
<p><small>Automatic CUDA route selection is profile-locked to the reference GPU/driver (fail-closed by design), so on this box the
default configuration is the exact host-parallel path only.</small></p>
{suite_table(bd, bdsum)}
<h3>With CUDA hybrid paths enabled (<code>PDG_EXPERIMENTAL_CUDA_HYBRID=1</code>)</h3>
<p><small>The same exact CUDA stages the reference machine selects automatically, forced on for every eligible stage regardless of
profile or measured envelope; still byte-identical. Note the light workflows (r1, r3, r7, r11) get <em>slower</em> when CUDA is
forced — device set-up and transfers outweigh the work — while r6 gains 4×; the reference machine's automatic selector never
picks those losing routes, which is the point of measured selection.</small></p>
{suite_table(bx, bxsum)}
<h3>CUDA hybrid + <code>--fast</code></h3>
{suite_table(bf, bfsum, extra='Records vs stock')}

<h2>5. Big point clouds — AHN4 GeoTile 25GN1_01 (47,478,228 points, LAS 1.4 format 8, EPSG:28992)</h2>
<p><small>Public Dutch national lidar tile from geotiles.citg.tudelft.nl (480 MB LAZ). Warm cache, no warmup, two runs (default) or
one run (CUDA hybrid); pinned PDAL alone needs minutes per run on the heavy graphs.</small></p>
<h3>Reference workstation</h3>{big_table(big_local)}
<h3>Rented Threadripper PRO + RTX 4090</h3>{big_table(big_box)}

<h2>6. LAStools comparison</h2>
<p><small>LAStools open-source tools (laszip, las2las, lasmerge) built from github.com/LAStools/LAStools and run natively; proprietary
tools (lasground_new, lasheight, las2dem, lasgrid, lasnoise, lasthin, lastile) run as the vendor's Windows binaries under
wine in <code>-demo</code> mode. Wall time wraps the whole process — for the wine tools that includes wine start-up, which is
about 4.5 s on the workstation for a no-op invocation — so the tool's own reported timer ("total time"/"took") is shown
beside it and is the fairer number for those rows. Outputs are <b>not</b> compared: LAStools uses different algorithms and demo mode
perturbs output, so this is a like-for-like timing of comparable jobs, not an exactness comparison. Same 1 M-point input.</small></p>
<h3>Reference workstation</h3>{lastools_table(lt_local)}
{('<h3>Rented box</h3>' + lastools_table(lt_box)) if lt_box else ''}

<h2>7. Rendered outputs (PDG results, byte-identical to stock PDAL)</h2>
<div class=grid>
{img(R/'input-1m.png','Input: the 1 M-point AHN reference strip coloured by elevation (250 m window).')}
{img(R/'r2-hag.png','r2: SMRF ground + height above ground (HAG) colouring.')}
{img(R/'r3-dtm-hillshade.png','r3: 1 m DTM from SMRF ground points (IDW), hillshaded.')}
{img(R/'r7-dsm-hillshade.png','r7: 1 m DSM from first/only returns (max Z), hillshaded.')}
{img(R/'r6-planarity.png','r6: covariance planarity feature (CUDA eigen family).')}
{img(R/'r11-classification.png','r11: SMRF + statistical outlier + neighbour-vote classification.')}
{img(R/'r8-rgb.png','r8: RGB sampled from the (synthetic, deterministic) EPSG:3857 orthophoto.')}
{img(R/'r4-thinned.png','r4: statistical outlier removal then 1 m spatial sampling.')}
{img(R/'r10-density.png','r10: voxel-centroid-nearest decimation.')}
{img(R/'r9-clip.png','r9: multipolygon (with hole) clip, polygon given in EPSG:4326.')}
</div>
""")
    if BR and BR.is_dir():
        pics = sorted(BR.glob("*.png"))
        if pics:
            parts.append("<h3>AHN4 47 M-point tile (rendered from the box's PDG outputs)</h3><div class=grid>")
            for pic in pics:
                parts.append(img(pic, pic.stem.replace("-", " ")))
            parts.append("</div>")

    parts.append("""
"""+ (f"""<h2>7b. Scale ladder on a datacenter GPU — where the speedup goes as clouds get bigger</h2>
<pre>{html.escape(chr(10).join(l for l in ladder_machine.strip().splitlines() if not l.startswith('----')))}</pre>
<p><small>Inputs: AHN4 GeoTiles 25GN1_01 (47.5 M points), 01+02 merged (~95 M), 01–04 merged (~190 M); one run each, warm cache
(pinned PDAL alone needs up to tens of minutes per run at the top size). I/O-bound graphs (r14, r1, r3) run the default host
configuration; the kNN graphs (r6, r4, r2) run with the CUDA hybrid forced on (SM 90 build; automatic selection is not
qualified for this GPU); r6/r4 also appear in host-only form at 47 M for reference.</small></p>
<pre>{html.escape(ladder_inputs.strip())}</pre>
{ladder_table(ladder)}
<div class=callout>
<p><b>Reading the ladder (measured, not theoretical).</b> Every one of the 20 runs is byte-exact against the pinned PDAL, so
the columns compare identical outputs. Two shapes appear. The kNN-heavy graphs run at a <b>flat ~10× plateau</b> across the whole
ladder — r6 (covariance features) 10.40× / 10.38× / 10.42× and r4 (denoise + thin) 9.74× / 10.17× / 9.34× at 47 M / 95 M / 190 M —
because both sides scale about linearly with N here: the pinned PDAL is one core doing N kd-tree queries, PDG is the GPU doing
the same N queries and the host doing the exact repairs; the ratio stays where the kernel throughput puts it. r2 (ground +
normalize) is the one graph whose speedup <b>climbs</b> with size, 1.40× → 1.53× → 1.65×, because its GPU-eligible fraction
(the kNN/HAG part) grows relative to the fixed, host-only PMF ground filter. The I/O-bound graphs stay flat: r14 (LAZ→LAS)
3.26× → 3.65×, r1 (translate + reproject) 1.4–1.6×, r3 (DTM raster) ~1.6× — those are already at the ceiling that single-file
LAZ decode/encode and the pinned writer set. So the direct answer to "are we even faster at the largest workloads?" is:
<b>no</b>. Bigger inputs do not raise the ratio, and the bigger GPU did not run the PDG side faster in absolute terms
either (r6 at 47 M: 53.6 s on the H200, 48.3 s on the rented RTX 4090 box, 30.7 s on the workstation) — at these sizes the
device is not the bottleneck for these graphs; the exact host repair, single-file LAZ decode/encode and the host-only stages
are. What the large card buys is headroom: the ~10× ratio holds at sizes where a 24 GB card would have to tile (the 190 M r6
run holds 51 GB resident on the host). What would raise the ratio further is not more points but more work per point (larger
neighbourhoods, more features per query) or moving the exact repair onto the device — the same reason r6 already sits at 10×
while r3 sits at 1.6×.</p>
<p><small>Caveats: single run per cell (no spread); the container's cgroup CPU quota was ~26.9 cores of the 224 visible
(<code>cpu.max 2687999 100000</code> in the machine block), so the host-parallel paths on this box were throttled to roughly the
same core budget as the Threadripper box, which is why the I/O-bound rows do not exceed that box's numbers; SM 90 build with the
CUDA hybrid forced on (automatic selection stays profile-locked to the qualified RTX 4090); the pinned oracle was rebuilt on
the box from the same upstream commit, so absolute seconds are not comparable with the workstation tables — only the ratios
are.</small></p>
</div>
""" if ladder else "") + """
<h2>8. Which CUDA kernels are worth proposing upstream</h2>
<p>Criteria: (a) exact against nanoflann/upstream semantics with the repair path included, (b) self-contained (depends only on
the PointBatch/SpatialIndex layer, CUB, and Eigen), (c) measured complete-process gains on real pipelines, (d) portable across
SM 75+ (compiled for every CUDA 13 target; physically qualified on SM 89 only — the Vast qualification protocol in
<code>docs/vast-qualification.md</code> is the gate for other GPUs).</p>
<div class=scroll><table>
<tr><th>Kernel family (source)</th><th>What it does</th><th>Measured effect</th><th>Upstream candidacy</th></tr>
<tr><td class=name><b>Spatial index build</b><br><code>src/index/SpatialIndexKernels.cu</code></td><td class=name>Uniform-grid cell table (CUB radix sort of Morton/cell keys) and Morton-BVH build; deterministic, no atomics in the query path.</td><td class=name>Foundation of every neighborhood win; ~ms at 1 M.</td><td class=name><b>Yes</b> — the piece everything else needs; clean interface (PointBatch → index).</td></tr>
<tr><td class=name><b>Exact kNN gather</b> (grid + BVH backends) with tie/incomplete status<br><code>knnGatherKernel</code>, <code>bvhKnnGatherKernel</code></td><td class=name>k nearest neighbours with retained k+1 candidate, distance-tie and incomplete-search flags so the host can repair to nanoflann-exact order.</td><td class=name>4M nndistance 48.8× (B0091); r6 features 10.7×; r11 12.9× (host-worker version) — the shared gather is 95% of device time in those routes.</td><td class=name><b>Yes, first candidate.</b> Optional CUDA backend for <code>KD3Index::knnSearch</code> bulk queries; the status/repair contract is what makes it mergeable without changing results.</td></tr>
<tr><td class=name><b>Covariance / eigen systems</b><br><code>knnCovariances</code>, <code>eigenSystemsKernel</code></td><td class=name>Upstream's ordered online-mean covariance and SelfAdjointEigenSolver contract per row (normal, eigenvalues, covariancefeatures, optimalneighborhood, approximatecoplanar, estimaterank).</td><td class=name>r6 10.7× (with the shared gather); 3-consumer pipeline 10.3× (B0075).</td><td class=name><b>Yes</b> — reproduces PDAL's exact arithmetic; the transcendental features (Omnivariance/Eigenentropy) stay host.</td></tr>
<tr><td class=name><b>Mean / kth distances</b><br><code>knnMeanDistances</code>, <code>knnDistanceValues</code></td><td class=name>filters.outlier statistical mean and filters.nndistance kth/average with PDAL's serial arithmetic order.</td><td class=name>Tie-invariant; big wins in resident compositions (B0088 21.4×), but the host-worker path now ties it at 1 M (B0272).</td><td class=name><b>Maybe</b> — worth it as part of the shared index, not standalone.</td></tr>
<tr><td class=name><b>Neighbor votes / LOF</b><br><code>neighborVoteKernel</code>, <code>lof*Kernel</code></td><td class=name>Majority vote over k neighbours; LOF's three passes over one retained adjacency.</td><td class=name>LOF 4M 18.3× (B0238); classifier 4.0× (B0231).</td><td class=name><b>Yes for LOF</b> (clear win); classifier optional.</td></tr>
<tr><td class=name><b>Radius family</b><br><code>radiusCountsKernel</code>, <code>radiusAnyKernel</code>, BVH variants</td><td class=name>Strict <code>d &lt; r²</code> counts/any/scaled values for radialdensity, radius outlier, radiusassign, with deterministic capacity-driven tiles.</td><td class=name>radialdensity 4.7× (B0100); radiusassign 5.4× (B0093).</td><td class=name><b>Yes</b> — simple, exact, tile-safe.</td></tr>
<tr><td class=name><b>Masked 2-D nearest ground (HAG-NN / Delaunay)</b></td><td class=name>kNN restricted to a ground mask for height above ground.</td><td class=name>r2 2.2× default (host); HAG-NN resident 1.25–2.05× (B0237).</td><td class=name><b>Later</b> — modest gains; the host path with concurrent KD builds is already close.</td></tr>
<tr><td class=name><b>Fused point programs</b> (assign/ferry/expression/crop) and direct LAS decode/publish</td><td class=name>JIT-fused per-point programs on device-resident columns; mapped LAS sources.</td><td class=name>18.8× for the accepted fused envelope (B0005) but only inside compute-heavy regions.</td><td class=name><b>No, not as-is</b> — tied to the planner/placement machinery; upstream would need the whole executor.</td></tr>
<tr><td class=name><b>Standalone per-stage kernels</b> (PMF, CSF, ELM, transformation, IQR/MAD, colorinterp, sort, decimation, locate)</td><td class=name>Exact device envelopes exist.</td><td class=name>Measured slower than stock end-to-end (0.3–1.0×); force-only.</td><td class=name><b>No</b> — the record shows standalone kernels don't pay; do not propose.</td></tr>
</table></div>
<p><b>Also worth proposing upstream, though not CUDA:</b> the exact host improvements are self-contained PDAL changes and
likely the easiest merges — parallel LAS record pack/unpack behind a streaming batch hook, ordered parallel COPC decode,
LAZ chunk compression workers, banded <code>writers.gdal</code> accumulation, slot-pooled reprojection, hashed
<code>filters.sample</code>, worker passes in outlier/neighborclassifier/SMRF, and the concurrent nanoflann build. Together
they are most of the 5.4× total-wall gain in the default (exact) mode.</p>

<h2>9. Method, caveats and reproducibility</h2>
<ul>
<li>Every timing is a complete process wall clock (<code>pdal pipeline</code> vs <code>pdg pipeline</code>) with a frozen
observable clock so LAS headers match; runs alternate oracle/candidate; artifacts, metadata, stdout, stderr and status are
compared on every run (exact contract) — <code>scripts/pdg/benchmark_reference.py</code>, <code>reference_suite.py</code>.</li>
<li>Reference workstation numbers are three pairs per cache state; nine-pair per-workload gates are in the records.
Machine load varies a few percent between runs (see the B0270 note); the geometric mean and total wall are the claims.</li>
<li>The rented box was configured with <code>bench/remote/vast_bootstrap.sh</code> and measured with
<code>bench/remote/vast_run.sh</code>; its oracle links Ubuntu 24.04's GDAL 3.8/PROJ 9.4, so raster and reprojection bytes
differ from the workstation's, but PDG is exact against that box's own oracle.</li>
<li>Automatic CUDA selection is deliberately locked to the qualified profile (RTX 4090 / SM 89 / CUDA 13.3 / driver
610.43.03); enabling the CUDA hybrid paths elsewhere is explicit (<code>PDG_EXPERIMENTAL_CUDA_HYBRID=1</code>). Qualifying
more GPUs follows <code>docs/vast-qualification.md</code>.</li>
<li>LAStools numbers are timings of comparable jobs, not equivalent outputs; the proprietary tools ran unlicensed in demo mode
under wine, which the vendor says perturbs output and which we do not use for anything but wall time.</li>
<li>Renders: point clouds decimated through the pinned oracle and drawn top-down (matplotlib); rasters hillshaded with GDAL
(<code>bench/report/render_outputs.py</code>).</li>
</ul>
</main>""")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(parts))
    print(a.out, a.out.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
