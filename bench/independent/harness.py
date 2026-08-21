#!/usr/bin/env python3
"""Run every (size, task, tool) job, time it, and record what came out.

Results are appended to a JSON file after every task so an interrupted run
keeps what it measured. Each record holds: the machine label, size, task, tool,
the exact commands, wall-clock seconds for every timed repeat (after one
untimed warm-up), the median, peak resident memory, exit status, output file
sizes / point counts / sha256, and the last lines of output on failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import IN, OUT, RESULTS, SIZES, TOOLS, clean_env, load_json  # noqa: E402
from tasks import TASKS, TASK_BY_ID, Ctx  # noqa: E402

ALL_TOOLS = ["pdg_gpu", "pdg_cpu", "pdg_gpu_all", "pdal_sys", "pdal_pinned",
             "lastools", "wrench", "qgis"]


def inputs_for(size: str) -> dict:
    d = {"laz": IN / f"{size}.laz", "las": IN / f"{size}.las",
         "ground": IN / f"{size}-ground.laz", "west": IN / f"{size}-west.laz",
         "east": IN / f"{size}-east.laz", "clip_wkt": IN / f"{size}-clip.wkt",
         "clip_gpkg": IN / f"{size}-clip.gpkg", "clip_shp": IN / f"{size}-clip.shp"}
    if size == "1m":
        d["copc"] = IN / "copc" / "1m.copc.laz"
        d["ortho"] = IN / "1m-ortho-3857.tif"
    return d


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def remove(path: pathlib.Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
    elif path.exists():
        path.unlink()


def run_once_rusage(commands, env, cwd, timeout) -> dict:
    """Like run_once but uses os.wait4 to get per-process peak RSS (process tree)."""
    full_env = clean_env()
    full_env.update(env)
    peak_kb = 0
    tails = []
    rc = 0
    unlicensed_warning = False
    started = time.perf_counter()
    for cmd in commands:
        try:
            proc = subprocess.Popen([str(c) for c in cmd], env=full_env, cwd=cwd,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except OSError as exc:
            rc = 127
            tails.append(f"could not start: {exc}")
            break
        # Read output in a thread-free way: read all, then wait4 for rusage.
        chunks = []
        deadline = time.monotonic() + timeout
        try:
            while True:
                data = proc.stdout.read(1 << 16)
                if not data:
                    break
                chunks.append(data)
                if len(chunks) > 4096:
                    chunks = chunks[-2048:]
                if time.monotonic() > deadline:
                    raise TimeoutError
        except TimeoutError:
            proc.kill()
            proc.stdout.read()
            os.waitpid(proc.pid, 0)
            rc = -9
            tails.append("TIMEOUT\n" + "".join(chunks)[-1500:])
            break
        _, status, ru = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        rc = proc.returncode
        peak_kb = max(peak_kb, ru.ru_maxrss)
        text = "".join(chunks)
        tails.append(text[-1500:])
        # Preserve LAStools' exit status.  A file deliberately distorted by an
        # unlicensed tool is useful exploratory output, not a successful or
        # qualifying benchmark observation.
        if "WARNING: unlicensed" in text:
            unlicensed_warning = True
        if rc != 0:
            break
    wall = time.perf_counter() - started
    return {"wall": wall, "rc": rc, "peak_rss_kb": peak_kb, "tail": "\n".join(tails),
            "unlicensed_warning": unlicensed_warning}


def describe_outputs(job, pdal_pinned: str, count_points: bool) -> list:
    described = []
    for out in job.outputs:
        out = pathlib.Path(out)
        if out.is_dir():
            files = sorted((p for p in out.rglob("*") if p.is_file()),
                           key=lambda p: p.relative_to(out).as_posix())
            members = [{"name": p.relative_to(out).as_posix(),
                        "bytes": p.stat().st_size, "sha256": sha256_of(p)}
                       for p in files]
            manifest = json.dumps(members, sort_keys=True,
                                  separators=(",", ":")).encode()
            entry = {"path": str(out), "kind": "directory", "files": len(files),
                     "bytes": sum(p.stat().st_size for p in files),
                     "sha256": hashlib.sha256(manifest).hexdigest(),
                     "members": members}
            if count_points and out in [pathlib.Path(p) for p in job.point_outputs]:
                total = 0
                ok = True
                for p in files:
                    if p.suffix.lower() in (".las", ".laz"):
                        try:
                            total += point_count(pdal_pinned, p)
                        except Exception:
                            ok = False
                entry["points"] = total if ok else None
            described.append(entry)
        elif out.exists():
            entry = {"path": str(out), "kind": "file", "bytes": out.stat().st_size,
                     "sha256": sha256_of(out)}
            if count_points and out in [pathlib.Path(p) for p in job.point_outputs]:
                try:
                    entry["points"] = point_count(pdal_pinned, out)
                except Exception as exc:  # pragma: no cover
                    entry["points"] = None
                    entry["points_error"] = str(exc)[:200]
            described.append(entry)
        else:
            described.append({"path": str(out), "kind": "missing"})
    return described


def point_count(pdal: str, path: pathlib.Path) -> int:
    r = subprocess.run([pdal, "info", "--metadata", str(path)], capture_output=True,
                       text=True, env=clean_env(), check=True)
    return int(json.loads(r.stdout)["metadata"]["count"])


def machine_info() -> dict:
    info = {"hostname": platform.node(), "platform": platform.platform(),
            "cpu": "", "threads": os.cpu_count(), "gpu": "", "ram_gb": None}
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                info["ram_gb"] = round(int(line.split()[1]) / 1048576, 1)
                break
        try:
            info["cgroup_cpu_max"] = open("/sys/fs/cgroup/cpu.max").read().strip()
        except Exception:
            info["cgroup_cpu_max"] = None
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True)
        info["gpu"] = r.stdout.strip() if r.returncode == 0 else "none"
    except Exception:
        pass
    versions = {}
    for key, cmd in (("pdg", [TOOLS["pdg"], "doctor"]), ("pdal_sys", [TOOLS["pdal_sys"], "--version"]),
                     ("pdal_pinned", [TOOLS["pdal_pinned"], "--version"]),
                     ("lastools", [str(pathlib.Path(TOOLS["lastools"]) / "laszip64"), "-version"]),
                     ("wrench", [TOOLS["wrench"], "--version"]),
                     ("qgis", [TOOLS["qgis_process"], "--version"])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=clean_env(), timeout=60)
            versions[key] = (r.stdout + r.stderr).strip()[:600]
        except Exception as exc:
            versions[key] = f"unavailable: {exc}"
    info["versions"] = versions
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="machine label, e.g. workstation-4090")
    ap.add_argument("--sizes", nargs="+", default=["1m"])
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--skip-tasks", nargs="+", default=[])
    ap.add_argument("--tools", nargs="+", default=ALL_TOOLS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--results", default=None)
    ap.add_argument("--no-count", action="store_true", help="skip point counting of outputs")
    ap.add_argument("--keep-outputs", action="store_true")
    a = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    results_path = pathlib.Path(a.results or RESULTS / f"{a.label}.json")
    if results_path.exists():
        doc = load_json(results_path)
    else:
        doc = {"label": a.label, "machine": machine_info(), "records": []}
    doc.setdefault("machine", machine_info())
    seen = {(r["size"], r["task"], r["tool"]) for r in doc["records"]
            if r.get("rc") == 0 and
            (r.get("task") == "qgis_startup" or
             (r.get("outputs") and all(o.get("kind") != "missing"
                                       for o in r["outputs"])))}

    def save():
        tmp = results_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(results_path)

    # QGIS start-up cost, measured once per invocation of the harness.
    if "qgis" in a.tools and ("_", "qgis_startup", "qgis") not in seen:
        walls = []
        for _ in range(a.warmup + a.repeats):
            r = run_once_rusage([[TOOLS["qgis_process"], "help", "pdal:info"]], {}, None, 300)
            walls.append(r["wall"])
        doc["records"].append({"size": "_", "task": "qgis_startup", "tool": "qgis",
                               "walls": walls[a.warmup:], "median": statistics.median(walls[a.warmup:]),
                               "rc": r["rc"], "commands": [["qgis_process", "help", "pdal:info"]],
                               "peak_rss_kb": r["peak_rss_kb"], "outputs": [], "note": "start-up only"})
        save()

    for size in a.sizes:
        inputs = inputs_for(size)
        extent = load_json(IN / f"{size}-extent.json")
        for t in TASKS:
            if a.tasks and t["id"] not in a.tasks:
                continue
            if t["id"] in a.skip_tasks or size not in t["sizes"]:
                continue
            for tool in a.tools:
                if t["tools"] and tool not in t["tools"]:
                    continue
                if (size, t["id"], tool) in seen:
                    continue
                outdir = OUT / tool / size / t["id"]
                outdir.mkdir(parents=True, exist_ok=True)
                ctx = Ctx(size=size, inputs=inputs, extent=extent, outdir=outdir, tool=tool, threads=a.threads)
                job = t["build"](ctx)
                if job is None:
                    continue
                rec = {"size": size, "task": t["id"], "tool": tool, "commands": [[str(c) for c in cmd] for cmd in job.commands],
                       "env": job.env, "note": job.note, "walls": [], "rc": 0}
                print(f"[{a.label}] {size:>4} {t['id']:<16} {tool:<12} ", end="", flush=True)
                for i in range(a.warmup + a.repeats):
                    for out in job.outputs:
                        remove(pathlib.Path(out))
                    r = run_once_rusage(job.commands, job.env, job.cwd, a.timeout)
                    if r["rc"] != 0:
                        rec["rc"] = r["rc"]
                        rec["tail"] = r["tail"]
                        rec["walls"].append(r["wall"])
                        rec["nonqualifying"] = bool(r.get("unlicensed_warning"))
                        print(f"FAILED rc={r['rc']} after {r['wall']:.2f}s")
                        break
                    if i >= a.warmup:
                        rec["walls"].append(r["wall"])
                    rec["peak_rss_kb"] = max(rec.get("peak_rss_kb", 0), r["peak_rss_kb"])
                    if r.get("unlicensed_warning") and "unlicensed" not in rec["note"]:
                        rec["note"] = (rec["note"] + "; " if rec["note"] else "") + "ran unlicensed: LAStools says the output is deliberately distorted (tiny xyz noise, some fields zeroed)"
                    print(f"{r['wall']:.3f} ", end="", flush=True)
                if rec["rc"] == 0:
                    rec["median"] = statistics.median(rec["walls"])
                    rec["outputs"] = describe_outputs(job, TOOLS["pdal_pinned"], not a.no_count)
                    print(f"-> median {rec['median']:.3f}s  peak {rec.get('peak_rss_kb', 0) / 1024:.0f} MB")
                else:
                    # Retain any artifact for diagnosis, but never calculate a
                    # qualifying median from a nonzero process result.
                    rec["outputs"] = describe_outputs(
                        job, TOOLS["pdal_pinned"], not a.no_count)
                    print("   " + rec.get("tail", "")[-300:].replace("\n", " | "))
                doc["records"].append(rec)
                seen.add((size, t["id"], tool))
                save()
                if not a.keep_outputs:
                    for out in job.outputs:
                        remove(pathlib.Path(out))
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
