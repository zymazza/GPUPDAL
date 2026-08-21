"""Charts for the independent benchmark report (matplotlib, PNG)."""
from __future__ import annotations

import math
import pathlib
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tasks import TASKS  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9c8c4",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 8.5,
})

TEXT = "#0b0b0b"
MUTED = "#52514e"

# Tool identity: fixed order, one hue family per product, shades for its modes.
TOOLS = [
    ("pdg_gpu", "GPUPDAL — GPU on (automatic)", "#2a78d6", ""),
    ("pdg_cpu", "GPUPDAL — CPU only (no GPU)", "#86b6ef", ""),
    ("pdg_gpu_all", "GPUPDAL — GPU forced on", "#104281", "////"),
    ("pdal_pinned", "PDAL 2.10.0 (built from source)", "#eb6834", ""),
    ("pdal_sys", "PDAL 2.10.1 (Linux package)", "#f7b08e", ""),
    ("lastools", "LAStools (unlicensed)", "#1baf7a", ""),
    ("wrench", "QGIS engine (pdal_wrench)", "#4a3aa7", ""),
    ("qgis", "QGIS (qgis_process)", "#b3aaf0", ""),
]
TOOL_NAME = {t[0]: t[1] for t in TOOLS}
TOOL_COLOR = {t[0]: t[2] for t in TOOLS}
TOOL_HATCH = {t[0]: t[3] for t in TOOLS}
TOOL_ORDER = [t[0] for t in TOOLS]
SHORT = {"pdg_gpu": "GPUPDAL (GPU)", "pdg_cpu": "GPUPDAL (CPU only)", "pdg_gpu_all": "GPUPDAL (GPU forced)",
         "pdal_pinned": "PDAL 2.10.0", "pdal_sys": "PDAL 2.10.1 pkg", "lastools": "LAStools",
         "wrench": "QGIS engine", "qgis": "QGIS"}
SIZE_LABEL = {"1m": "1 million", "4m": "4 million", "16m": "16 million", "47m": "47 million"}
SIZE_POINTS = {"1m": 1_000_000, "4m": 4_000_000, "16m": 16_000_000, "47m": 47_478_228}
TASK_META = {t["id"]: t for t in TASKS}
FAMILIES = []
for t in TASKS:
    if t["family"] not in FAMILIES:
        FAMILIES.append(t["family"])


def index(doc: dict) -> dict:
    """{(size, task, tool): record}"""
    return {(r["size"], r["task"], r["tool"]): r for r in doc["records"]}


def median(rec):
    if rec is None or rec.get("rc", 1) != 0 or not rec.get("walls"):
        return None
    return statistics.median(rec["walls"])


def fmt_seconds(v: float) -> str:
    if v is None:
        return "—"
    if v < 10:
        return f"{v:.2f}"
    if v < 100:
        return f"{v:.1f}"
    return f"{v:.0f}"


def _sec_formatter(x, _pos):
    if x >= 1:
        return f"{x:g} s"
    return f"{x:g} s"


def _save(fig, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def legend_handles(tools, names=None):
    from matplotlib.patches import Patch
    names = names or TOOL_NAME
    return [Patch(facecolor=TOOL_COLOR[t], hatch=TOOL_HATCH[t], edgecolor="white" if not TOOL_HATCH[t] else "#fcfcfb",
                  label=names[t]) for t in tools]


def chart_family_bars(idx: dict, size: str, family: str, out: pathlib.Path, tools=TOOL_ORDER, title=None):
    tasks = [t for t in TASKS if t["family"] == family and size in t["sizes"]]
    tasks = [t for t in tasks if any(median(idx.get((size, t["id"], tool))) for tool in tools)]
    if not tasks:
        return None
    n_tools = len(tools)
    row_h = 0.16 * n_tools + 0.35
    fig, ax = plt.subplots(figsize=(9, 0.6 + row_h * len(tasks)))
    ys = np.arange(len(tasks))
    bar_h = 0.8 / n_tools
    vmax = 0
    for j, tool in enumerate(tools):
        for i, t in enumerate(tasks):
            rec = idx.get((size, t["id"], tool))
            v = median(rec)
            y = ys[i] - 0.4 + bar_h * (j + 0.5)
            if v is None:
                txt = "cannot do this job" if rec is None else ("failed" if rec.get("rc", 0) != 0 else "—")
                if rec is not None and rec.get("rc", 0) != 0 and "unlicensed" in rec.get("tail", "").lower():
                    txt = "refused (unlicensed)"
                ax.text(1.0, y, txt, va="center", ha="left", fontsize=6.5, color=MUTED, transform=ax.get_yaxis_transform())
                continue
            vmax = max(vmax, v)
            ax.barh(y, v, height=bar_h * 0.9, color=TOOL_COLOR[tool], hatch=TOOL_HATCH[tool],
                    edgecolor="white" if not TOOL_HATCH[tool] else "#fcfcfb", linewidth=0.5)
            ax.text(v * 1.08, y, fmt_seconds(v), va="center", ha="left", fontsize=6.8, color=TEXT)
    ax.set_yticks(ys)
    ax.set_yticklabels([t["name"] for t in tasks], fontsize=9, color=TEXT)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(right=vmax * 3.0)
    ax.xaxis.set_major_formatter(FuncFormatter(_sec_formatter))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("time for the whole job, seconds (log scale; shorter is better)")
    ax.grid(axis="x", which="major")
    ax.set_axisbelow(True)
    ax.set_title(title or f"{family} — {SIZE_LABEL[size]} points")
    fig.legend(handles=legend_handles(tools), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8)
    _save(fig, out)
    return out


def chart_speedup(idx: dict, size: str, out: pathlib.Path, base="pdg_gpu",
                  others=("pdal_pinned", "lastools", "wrench", "qgis"), title=None):
    tasks = [t for t in TASKS if size in t["sizes"] and median(idx.get((size, t["id"], base)))]
    tasks = [t for t in tasks if any(median(idx.get((size, t["id"], o))) for o in others)]
    if not tasks:
        return None
    n = len(others)
    fig, ax = plt.subplots(figsize=(9, 0.6 + (0.16 * n + 0.3) * len(tasks)))
    ys = np.arange(len(tasks))
    bar_h = 0.8 / n
    vmax = 1
    for j, o in enumerate(others):
        for i, t in enumerate(tasks):
            b = median(idx.get((size, t["id"], base)))
            v = median(idx.get((size, t["id"], o)))
            y = ys[i] - 0.4 + bar_h * (j + 0.5)
            if v is None or b is None:
                continue
            ratio = v / b
            vmax = max(vmax, ratio)
            ax.barh(y, ratio, height=bar_h * 0.9, color=TOOL_COLOR[o], hatch=TOOL_HATCH[o], edgecolor="white", linewidth=0.5)
            ax.text(ratio * 1.06, y, f"{ratio:.1f}×", va="center", ha="left", fontsize=6.8, color=TEXT)
    ax.axvline(1.0, color="#8a8984", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([t["name"] for t in tasks], fontsize=9, color=TEXT)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.25, vmax * 2.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:g}×"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(f"how many times longer the other tool took than {TOOL_NAME[base]}\n(log scale; 1× = same time; left of 1× = the other tool was faster)")
    ax.grid(axis="x", which="major")
    ax.set_axisbelow(True)
    ax.set_title(title or f"Speed of GPUPDAL compared with the other tools — {SIZE_LABEL[size]} points")
    fig.legend(handles=legend_handles(list(others)), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8)
    _save(fig, out)
    return out


def chart_scaling(idx: dict, out: pathlib.Path, tools=TOOL_ORDER, sizes=("1m", "4m", "16m", "47m"), title=None):
    tasks = [t for t in TASKS if len([s for s in t["sizes"] if s in sizes]) >= 2]
    tasks = [t for t in tasks if any(median(idx.get((s, t["id"], tool))) for s in sizes for tool in tools)]
    ncol = 4
    nrow = math.ceil(len(tasks) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.5 * nrow), sharex=True)
    axes = np.atleast_2d(axes)
    for k, t in enumerate(tasks):
        ax = axes[k // ncol][k % ncol]
        for tool in tools:
            xs, ys = [], []
            for s in sizes:
                v = median(idx.get((s, t["id"], tool)))
                if v is not None:
                    xs.append(SIZE_POINTS[s] / 1e6)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, marker="o", ms=4, lw=1.8, color=TOOL_COLOR[tool],
                        ls="--" if TOOL_HATCH[tool] else "-", markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(t["name"], fontsize=8.5)
        ax.set_xticks([1, 4, 16, 47])
        ax.set_xticklabels(["1M", "4M", "16M", "47M"], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major")
        ax.set_axisbelow(True)
    for k in range(len(tasks), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=TOOL_COLOR[t], lw=2, marker="o", ms=4, ls="--" if TOOL_HATCH[t] else "-", label=TOOL_NAME[t]) for t in tools]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02 - 0.15 / nrow))
    fig.suptitle(title or "How the time grows with the number of points (seconds, both axes log scale; lower is better)", fontsize=11, fontweight="bold")
    fig.supylabel("seconds", fontsize=9, color=MUTED)
    fig.supxlabel("millions of points", fontsize=9, color=MUTED, y=0.0)
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.97))
    _save(fig, out)
    return out


def chart_gpu_vs_cpu(idx: dict, out: pathlib.Path, sizes=("1m", "4m", "16m", "47m"), title=None):
    """Bars: GPUPDAL CPU-only time / GPUPDAL GPU-auto time and GPU-forced / GPU-auto per task, one panel per size."""
    sizes = [s for s in sizes if any(k[0] == s and k[2] == "pdg_gpu" for k in idx)]
    tasks = [t for t in TASKS if any(median(idx.get((s, t["id"], "pdg_gpu"))) for s in sizes)]
    fig, axes = plt.subplots(1, len(sizes), figsize=(2.6 * len(sizes) + 1.5, 0.35 * len(tasks) + 1.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, sizes):
        ys = np.arange(len(tasks))
        for i, t in enumerate(tasks):
            g = median(idx.get((s, t["id"], "pdg_gpu")))
            c = median(idx.get((s, t["id"], "pdg_cpu")))
            f = median(idx.get((s, t["id"], "pdg_gpu_all")))
            if g and c:
                ax.barh(ys[i] - 0.2, c / g, height=0.36, color=TOOL_COLOR["pdg_cpu"], edgecolor="white")
                ax.text(max(c / g, 0.3) * 1.05, ys[i] - 0.2, f"{c / g:.2f}×", va="center", fontsize=6.5, color=TEXT)
            if g and f:
                ax.barh(ys[i] + 0.2, f / g, height=0.36, color=TOOL_COLOR["pdg_gpu_all"], hatch="////", edgecolor="#fcfcfb")
                ax.text(max(f / g, 0.3) * 1.05, ys[i] + 0.2, f"{f / g:.2f}×", va="center", fontsize=6.5, color=TEXT)
        ax.axvline(1, color="#8a8984", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(0.25, 12)
        ax.set_xticks([0.5, 1, 2, 4, 8])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:g}×"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(ys)
        ax.set_yticklabels([t["name"] for t in tasks], fontsize=8, color=TEXT)
        ax.set_title(f"{SIZE_LABEL[s]} points", fontsize=9.5)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.legend(handles=legend_handles(["pdg_cpu", "pdg_gpu_all"]), loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title or "Same program, same computer: time relative to GPUPDAL with the GPU on (automatic) = 1×\n(right of 1× = slower than automatic; left = faster)", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    _save(fig, out)
    return out


def chart_totals(idx: dict, out: pathlib.Path, tools=TOOL_ORDER, sizes=("1m", "4m", "16m", "47m"), only_tasks=None,
                 title=None, tool_names=None):
    """Sum of medians over the jobs every listed tool completed at every listed size."""
    sizes = [s for s in sizes if any(k[0] == s for k in idx)]
    common = []
    for t in TASKS:
        if only_tasks and t["id"] not in only_tasks:
            continue
        ok = all(median(idx.get((s, t["id"], tool))) for s in sizes for tool in tools if s in t["sizes"])
        if ok and all(s in t["sizes"] for s in sizes):
            common.append(t)
    if not common:
        return None, []
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(sizes))
    w = 0.8 / len(tools)
    totals = {
        tool: [sum(median(idx[(s, t["id"], tool)]) for t in common) for s in sizes]
        for tool in tools
    }
    vmax = max(v for vals in totals.values() for v in vals)
    for j, tool in enumerate(tools):
        vals = totals[tool]
        xs = x - 0.4 + w * (j + 0.5)
        ax.bar(xs, vals, width=w * 0.9, color=TOOL_COLOR[tool], hatch=TOOL_HATCH[tool], edgecolor="white", linewidth=0.5)
        for xi, v in zip(xs, vals):
            ax.annotate(fmt_seconds(v), xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, rotation=90, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SIZE_LABEL[s]} points" for s in sizes])
    ax.yaxis.set_major_formatter(FuncFormatter(_sec_formatter))
    ax.set_ylabel("total seconds (linear scale)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylim(0, vmax * 1.18)
    ax.set_title(title or f"Total time to run the {len(common)} jobs that every tool can do (linear scale)")
    fig.legend(handles=legend_handles(tools, tool_names), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8)
    _save(fig, out)
    return out, common


def chart_cross_machine(idxs: dict, size: str, task_ids: list, out: pathlib.Path, tools=("pdg_gpu", "pdg_cpu", "pdal_pinned", "lastools", "wrench"), title=None):
    """One panel per task; machines on y; bars per tool."""
    machines = list(idxs.keys())
    tasks = [TASK_META[t] for t in task_ids if any(median(idxs[m].get((size, t, tool))) for m in machines for tool in tools)]
    ncol = 2
    nrow = math.ceil(len(tasks) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, (0.22 * len(tools) * len(machines) + 0.9) * nrow))
    axes = np.atleast_2d(axes)
    for k, t in enumerate(tasks):
        ax = axes[k // ncol][k % ncol]
        ys = np.arange(len(machines))
        h = 0.8 / len(tools)
        vmax = 0
        for j, tool in enumerate(tools):
            for i, m in enumerate(machines):
                v = median(idxs[m].get((size, t["id"], tool)))
                y = ys[i] - 0.4 + h * (j + 0.5)
                if v is None:
                    rec = idxs[m].get((size, t["id"], tool))
                    if rec is not None and rec.get("rc", 0) != 0:
                        ax.text(1.0, y, "failed", va="center", ha="left", fontsize=6, color=MUTED, transform=ax.get_yaxis_transform())
                    continue
                vmax = max(vmax, v)
                ax.barh(y, v, height=h * 0.9, color=TOOL_COLOR[tool], hatch=TOOL_HATCH[tool], edgecolor="white", linewidth=0.5)
                ax.text(v * 1.07, y, fmt_seconds(v), va="center", fontsize=6.3, color=TEXT)
        ax.set_yticks(ys)
        ax.set_yticklabels(machines, fontsize=8, color=TEXT)
        ax.invert_yaxis()
        ax.set_xscale("log")
        if vmax > 0:
            ax.set_xlim(right=vmax * 3)
        ax.xaxis.set_major_formatter(FuncFormatter(_sec_formatter))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_title(t["name"], fontsize=9)
    for k in range(len(tasks), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.legend(handles=legend_handles(list(tools)), loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01 - 0.1 / nrow))
    fig.suptitle(title or f"The same jobs on three different computers — {SIZE_LABEL[size]} points (seconds, log scale)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, out)
    return out


def chart_memory(idx: dict, size: str, out: pathlib.Path, tools=TOOL_ORDER, title=None):
    tasks = [t for t in TASKS if size in t["sizes"] and any(idx.get((size, t["id"], tool), {}).get("peak_rss_kb") for tool in tools)]
    fig, ax = plt.subplots(figsize=(9, 0.6 + (0.16 * len(tools) + 0.3) * len(tasks)))
    ys = np.arange(len(tasks))
    h = 0.8 / len(tools)
    vmax = 0
    for j, tool in enumerate(tools):
        for i, t in enumerate(tasks):
            rec = idx.get((size, t["id"], tool))
            if not rec or rec.get("rc", 1) != 0 or not rec.get("peak_rss_kb"):
                continue
            v = rec["peak_rss_kb"] / 1024 / 1024
            vmax = max(vmax, v)
            y = ys[i] - 0.4 + h * (j + 0.5)
            ax.barh(y, v, height=h * 0.9, color=TOOL_COLOR[tool], hatch=TOOL_HATCH[tool], edgecolor="white", linewidth=0.5)
            ax.text(v * 1.06, y, f"{v:.1f}", va="center", fontsize=6.5, color=TEXT)
    ax.set_yticks(ys)
    ax.set_yticklabels([t["name"] for t in tasks], fontsize=8.5, color=TEXT)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(right=max(vmax, 1) * 2.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:g} GB"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("peak memory used by the whole job, GB (log scale; less is better)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_title(title or f"Peak memory (RAM) — {SIZE_LABEL[size]} points")
    fig.legend(handles=legend_handles(tools), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8)
    _save(fig, out)
    return out


def chart_headline(idxs: dict, out: pathlib.Path):
    """One picture: geometric-mean speed of GPUPDAL (GPU auto) vs each other tool at every size, workstation."""
    idx = idxs["Workstation (Ryzen 9 7900 + RTX 4090)"] if "Workstation (Ryzen 9 7900 + RTX 4090)" in idxs else list(idxs.values())[0]
    others = ["pdal_pinned", "pdal_sys", "lastools", "wrench", "qgis"]
    sizes = [s for s in ("1m", "4m", "16m", "47m") if any(k[0] == s for k in idx)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(sizes))
    w = 0.8 / len(others)
    for j, o in enumerate(others):
        vals, ns = [], []
        for s in sizes:
            ratios = []
            for t in TASKS:
                b = median(idx.get((s, t["id"], "pdg_gpu")))
                v = median(idx.get((s, t["id"], o)))
                if b and v:
                    ratios.append(v / b)
            vals.append(math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else np.nan)
            ns.append(len(ratios))
        xs = x - 0.4 + w * (j + 0.5)
        ax.bar(xs, vals, width=w * 0.9, color=TOOL_COLOR[o], edgecolor="white", linewidth=0.5)
        for xi, v, n in zip(xs, vals, ns):
            if not np.isnan(v):
                ax.text(xi, v + 0.08, f"{v:.1f}×", ha="center", va="bottom", fontsize=7, color=TEXT)
    ax.axhline(1, color="#8a8984", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SIZE_LABEL[s]} points" for s in sizes])
    ax.set_ylabel("times slower than GPUPDAL (GPU on)\n(geometric mean over the jobs both can do)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("Headline: on the workstation, how much longer each tool took than GPUPDAL, on average")
    fig.legend(handles=legend_handles(others), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8)
    _save(fig, out)
    return out
