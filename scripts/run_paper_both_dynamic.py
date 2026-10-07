#!/usr/bin/env python3
"""A and B both slide in one window. 5 seeds.

Default is no outliers (no_seed*). --variant out loads the outlier
file into out_seed*. The binary keeps that point in A for every
query. This does not touch the no-outlier runs.
Ours is native l2 E2LSH. QuadTree, Uniform, and Benchmark run in the same
process. The driver forces one OpenMP thread and one SIMD exact-NN kernel
so the per-update times are comparable. Jobs run one at a time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "paper_both"
LOG = OUT / "batch.log"
SEEDS = [1, 2, 3, 4, 5]

# Windows are the large both-dynamic windows. Query gap matches the unidirectional
# runs, which already yield ~54–57 queries. Text Embedding's stream is only
# |A|+|B|=3058; querying every 1500/5=300 leaves six queries after the window fills.
# GIST is last: its working set is about 9–12 GiB. The smaller jobs checkpoint
# first, so a GIST RSS stop does not discard Fashion or SIFT.
JOBS = [
    {"name": "textemb", "bin": "compare_perf_d300", "data": "TextEmbedding",
     "w": 1500, "t": 100, "q": 20, "timeout": 7200},
    {"name": "fashion", "bin": "compare_perf", "data": "Fashion-MNIST",
     "w": 3500, "t": 100, "q": 1166, "timeout": 21600},
    {"name": "sift", "bin": "compare_perf_sift", "data": "SIFT",
     "w": 50000, "t": 500, "q": 16666, "n_max": 70000, "timeout": 36000},
    {"name": "gist", "bin": "compare_perf_gist", "data": "GIST_1M",
     "w": 50000, "t": 500, "q": 16666, "n_max": 70000, "timeout": 43200},
]


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = msg if msg.endswith("\n") else msg + "\n"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    print(msg, flush=True)


def which_bin(name: str) -> Path:
    exe = ROOT / "build" / (name + ".exe")
    if exe.is_file():
        return exe
    p = ROOT / "build" / name
    if p.is_file():
        return p
    raise FileNotFoundError(name)


# Joint GIST keeps the float map, the grow-only E2LSH coordinate copy, and the
# QuadTree. Stop before the 15.75 GiB machine starts paging.
GIST_RSS_CAP_BYTES = 14 * 1024 ** 3


def _working_set(pid: int) -> int | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    h = kernel32.OpenProcess(0x0410, False, pid)  # QUERY_INFORMATION | VM_READ
    if not h:
        return None
    try:
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return None
        return int(pmc.WorkingSetSize)
    finally:
        kernel32.CloseHandle(h)


def run_cmd(cmd: list[str], log_path: Path | None = None, timeout: int | None = None,
            rss_cap_bytes: int | None = None) -> int:
    log("CMD " + " ".join(str(c) for c in cmd))
    t0 = time.time()
    fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=fh if fh is not None else None,
            stderr=subprocess.STDOUT if fh is not None else None,
        )
        deadline = (t0 + timeout) if timeout else None
        last_hb = t0
        peak = 0
        while proc.poll() is None:
            if rss_cap_bytes is not None:
                ws = _working_set(proc.pid)
                if ws is not None:
                    peak = max(peak, ws)
                    if ws > rss_cap_bytes:
                        log(f"  RSS_CAP pid={proc.pid} ws_gib={ws / 2**30:.3f} "
                            f"cap_gib={rss_cap_bytes / 2**30:.3f}")
                        proc.kill()
                        proc.wait()
                        return 137
            if deadline is not None and time.time() > deadline:
                proc.kill()
                proc.wait()
                log(f"  TIMEOUT after {timeout}s")
                return 124
            time.sleep(2 if rss_cap_bytes is not None else 15)
            if time.time() - last_hb >= 15:
                log(f"  HEARTBEAT {Path(cmd[0]).name} elapsed_s={time.time() - t0:.0f} pid={proc.pid}")
                last_hb = time.time()
        if rss_cap_bytes is not None and peak:
            log(f"  peak_rss_gib={peak / 2**30:.3f}")
    finally:
        if fh is not None:
            fh.close()
    dt = time.time() - t0
    log(f"  rc={proc.returncode} wall_s={dt:.1f}" + (f" log={log_path}" if log_path else ""))
    return int(proc.returncode or 0)


def done_timing(d: Path) -> bool:
    return (d / "timing_summary.json").is_file() and (d / "compare_cost_log.csv").is_file()


def phase_build() -> None:
    log("=== BUILD both-dynamic binaries ===")
    build = ROOT / "build"
    rc = run_cmd(["cmake", "-S", str(ROOT), "-B", str(build)])
    if rc != 0:
        raise SystemExit("cmake configure failed")
    rc = run_cmd(
        ["cmake", "--build", str(build), "--target",
         "compare_perf", "compare_perf_sift", "compare_perf_gist",
         "compare_perf_d300"],
        timeout=1800,
    )
    if rc != 0:
        raise SystemExit("cmake build failed")
    for name in ("compare_perf", "compare_perf_sift", "compare_perf_gist",
                 "compare_perf_d300"):
        which_bin(name)
    log("  binaries ok")


def _check_smoke_log(text: str, max_a: int, max_b: int, prefix: str, max_run: int) -> None:
    if "omp_threads=1" not in text:
        raise SystemExit("smoke: omp_threads is not 1")
    if "NN_MICROBENCH" not in text or re.search(r"NN_MICROBENCH\b.*\bstatus=OK\b", text) is None:
        raise SystemExit("smoke: NN microbench did not report status=OK")
    if "status=FAIL" in text:
        raise SystemExit("smoke: NN microbench status=FAIL")
    m = re.search(
        r"schedule_done nA=(\d+) ai=(\d+) nB=(\d+) bi=(\d+) max_A_run=(\d+) prefix=([AB]+)",
        text,
    )
    if not m:
        raise SystemExit("smoke: missing schedule_done")
    nA, ai, nB, bi, run, got_prefix = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                       int(m.group(4)), int(m.group(5)), m.group(6))
    if (nA, ai, nB, bi, run) != (max_a, max_a, max_b, max_b, max_run):
        raise SystemExit(
            f"smoke: schedule {nA, ai, nB, bi, run} != {(max_a, max_a, max_b, max_b, max_run)}"
        )
    if not got_prefix.startswith(prefix):
        raise SystemExit(f"smoke: prefix {got_prefix} does not start with {prefix}")


def _check_smoke_outputs(out_dir: Path) -> None:
    timing = json.loads((out_dir / "timing_summary.json").read_text(encoding="utf-8"))
    if timing.get("omp_threads") != 1:
        raise SystemExit(f"smoke: json omp_threads={timing.get('omp_threads')}")
    if timing.get("nn_microbench", {}).get("status") != "OK":
        raise SystemExit(f"smoke: json microbench {timing.get('nn_microbench')}")
    per = timing.get("per_update_ms") or {}
    for key in ("E2LSH", "QuadTree", "Uniform", "Benchmark"):
        val = per.get(key)
        if val is None or not math.isfinite(float(val)) or float(val) < 0:
            raise SystemExit(f"smoke: bad per_update_ms[{key}]={val}")
    csv_text = (out_dir / "compare_cost_log.csv").read_text(encoding="utf-8")
    rows = [ln for ln in csv_text.splitlines()[1:] if ln.strip()]
    if not rows:
        raise SystemExit("smoke: no query rows")
    for ln in rows:
        parts = ln.split(",")
        for rel in parts[-3:]:
            x = float(rel)
            if not math.isfinite(x):
                raise SystemExit(f"smoke: non-finite error in {ln}")


def phase_smoke() -> None:
    log("=== SMOKE both-dynamic fairness ===")
    cases = [
        {"tag": "div", "max_a": 840, "max_b": 140, "window": 210, "q": 70, "t": 100,
         "prefix": "AAAAAAB", "max_run": 6},
        {"tag": "rem", "max_a": 50, "max_b": 30, "window": 20, "q": 5, "t": 32,
         "prefix": "ABAAB", "max_run": 2},
    ]
    data = ROOT / "data" / "Fashion-MNIST"
    for case in cases:
        out_dir = OUT / "smoke" / case["tag"]
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"
        cmd = [
            str(which_bin("compare_perf")),
            "--dual-dynamic", "--no-outlier",
            "--data-dir", str(data),
            "--max-A", str(case["max_a"]),
            "--max-B", str(case["max_b"]),
            "--window-size", str(case["window"]),
            "--query-interval", str(case["q"]),
            "--T", str(case["t"]),
            "--seed", "1",
            "--n-max", "70000",
            "--out", str(out_dir / "compare_cost_log.csv"),
            "--timing-out", str(out_dir / "timing_summary.json"),
        ]
        rc = run_cmd(cmd, log_path=log_path, timeout=1800)
        if rc != 0:
            raise SystemExit(f"smoke {case['tag']} rc={rc}")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        _check_smoke_log(text, case["max_a"], case["max_b"], case["prefix"], case["max_run"])
        _check_smoke_outputs(out_dir)
        bench = json.loads((out_dir / "timing_summary.json").read_text(encoding="utf-8"))
        micro = bench["nn_microbench"]
        log(f"  smoke {case['tag']} ok prefix={case['prefix']} "
            f"ratio={micro['ratio_slow_over_fast']:.3f} note={micro['note']} "
            f"reps={micro['reps']}")


def compare_job(job: dict, seed: int, *, outlier: bool) -> None:
    tag = "out" if outlier else "no"
    out_dir = OUT / job["name"] / f"{tag}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if done_timing(out_dir):
        log(f"SKIP {job['name']} {tag} seed={seed}")
        return
    cmd = [str(which_bin(job["bin"])), "--dual-dynamic"]
    if not outlier:
        cmd.append("--no-outlier")
    cmd += [
        "--data-dir", str(ROOT / "data" / job["data"]),
        "--window-size", str(job["w"]),
        "--query-interval", str(job["q"]),
        "--T", str(job["t"]),
        "--seed", str(seed),
        "--n-max", str(job.get("n_max", 70000)),
        "--out", str(out_dir / "compare_cost_log.csv"),
        "--timing-out", str(out_dir / "timing_summary.json"),
    ]
    cap = GIST_RSS_CAP_BYTES if job["name"] == "gist" else None
    rc = run_cmd(cmd, log_path=out_dir / "run.log", timeout=job["timeout"], rss_cap_bytes=cap)
    if rc == 137:
        raise SystemExit(f"GIST RSS cap {GIST_RSS_CAP_BYTES / 2**30:.0f} GiB exceeded at {out_dir}")
    if rc != 0:
        raise SystemExit(f"compare failed {out_dir} rc={rc}")
    timing = json.loads((out_dir / "timing_summary.json").read_text(encoding="utf-8"))
    micro = timing.get("nn_microbench") or {}
    if timing.get("omp_threads") != 1 or micro.get("status") != "OK":
        raise SystemExit(f"unfair timing {out_dir}: omp={timing.get('omp_threads')} micro={micro}")
    if bool(timing.get("no_outlier")) == outlier:
        raise SystemExit(
            f"outlier flag mismatch {out_dir}: no_outlier={timing.get('no_outlier')}"
        )
    log(f"  {job['name']} {tag} seed={seed} queries={timing.get('queries')} "
        f"nA={timing.get('nA')} per_update_ms={timing.get('per_update_ms')}")


def phase_run(only: set[str] | None = None, *, outlier: bool = False) -> None:
    jobs = JOBS
    if only:
        known = {j["name"] for j in JOBS}
        unknown = sorted(only - known)
        if unknown:
            raise SystemExit(f"unknown --only {unknown}; known={sorted(known)}")
        jobs = [j for j in JOBS if j["name"] in only]
    tag = "out" if outlier else "no"
    log("=== RUN both-dynamic " + ",".join(j["name"] for j in jobs) + f" variant={tag} x 5 seeds ===")
    for job in jobs:
        data = ROOT / "data" / job["data"]
        if not (data / "query.f32bin").is_file():
            raise SystemExit(f"missing {data / 'query.f32bin'}")
        if outlier:
            has_a = ((data / "base_with_outlier.f32bin").is_file()
                     or (data / "train_with_outlier.f32bin").is_file())
            if not has_a:
                raise SystemExit(f"missing *_with_outlier.f32bin in {data}")
        else:
            has_a = (data / "base.f32bin").is_file() or (data / "train.f32bin").is_file()
            if not has_a:
                raise SystemExit(f"missing base/train f32bin in {data}")
        for seed in SEEDS:
            compare_job(job, seed, outlier=outlier)


def phase_plot() -> None:
    log("=== PLOT fig6 fig7 ===")
    rc = run_cmd([sys.executable, str(ROOT / "scripts" / "plot_paper_both_dynamic.py")])
    if rc != 0:
        raise SystemExit("plot failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["build", "smoke", "run", "plot", "all"])
    ap.add_argument("--only", default="",
                    help="comma-separated dataset names to run (e.g. gist)")
    ap.add_argument("--variant", default="no", choices=["no", "out"],
                    help="no: --no-outlier into no_seed*. out: outlier file into out_seed*.")
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    outlier = args.variant == "out"
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"START phase={args.phase} variant={args.variant} only={args.only or 'all'} "
        f"pid={os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    order = {
        "build": [phase_build],
        "smoke": [phase_smoke],
        "run": [lambda: phase_run(only, outlier=outlier)],
        "plot": [phase_plot],
        "all": [phase_build, phase_smoke, lambda: phase_run(only, outlier=outlier), phase_plot],
    }[args.phase]
    for fn in order:
        fn()
    log(f"FINISHED phase={args.phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
