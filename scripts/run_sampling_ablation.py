#!/usr/bin/env python3
"""Ablation: first-collision importance sampling vs stratified layer budgets.

Four datasets, no-outlier and with-outlier, seeds 1..5.
Each run is --e2lsh-only --with-is: both estimators see the same snapshot and T.

Phases: build, data, exact, smoke, run, aggregate, all
Skip a finished run when compare_cost_log.csv and timing_summary.json exist.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "sampling_ablation"
LOG = OUT / "batch.log"
SEEDS = [1, 2, 3, 4, 5]

JOBS = [
    {
        "name": "textemb",
        "bin": "compare_perf_d300",
        "data": ROOT / "data" / "TextEmbedding",
        "window": 100,
        "interval": 20,
        "T": 150,
        "n_max": None,
        "storage": False,
        "queries": 54,
        "timeout": 900,
        "cache_no": OUT / "textemb" / "exact_no" / "exact_cache.csv",
        "cache_out": OUT / "textemb" / "exact_out" / "exact_cache.csv",
    },
    {
        "name": "fashion",
        "bin": "compare_perf",
        "data": ROOT / "data" / "Fashion-MNIST",
        "window": 500,
        "interval": 166,
        "T": 200,
        "n_max": None,
        "storage": False,
        "queries": 57,
        "timeout": 1800,
        "cache_no": OUT / "fashion" / "exact_no" / "exact_cache.csv",
        "cache_out": OUT / "fashion" / "exact_out" / "exact_cache.csv",
    },
    {
        "name": "sift",
        "bin": "compare_perf_sift",
        "data": ROOT / "data" / "SIFT",
        "window": 500,
        "interval": 166,
        "T": 300,
        "n_max": 1100000,
        "storage": False,
        "queries": 57,
        "timeout": 3600,
        "cache_no": OUT / "sift" / "exact_no" / "exact_cache.csv",
        "cache_out": OUT / "sift" / "exact_out" / "exact_cache.csv",
    },
    {
        "name": "gist",
        "bin": "compare_perf_gist",
        "data": ROOT / "data" / "GIST_1M",
        "window": 50,
        "interval": 17,
        "T": 300,
        "n_max": None,
        "storage": True,
        "queries": 56,
        "timeout": 7200,
        "cache_no": OUT / "gist" / "exact_no" / "exact_cache.csv",
        "cache_out": OUT / "gist" / "exact_out" / "exact_cache.csv",
    },
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


def run_cmd(cmd: list[str], log_path: Path | None = None, timeout: int | None = None) -> int:
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
        while proc.poll() is None:
            if deadline is not None and time.time() > deadline:
                proc.kill()
                proc.wait()
                log(f"  TIMEOUT after {timeout}s")
                return 124
            time.sleep(1)
            now = time.time()
            if now - last_hb >= 30:
                log(f"  HEARTBEAT elapsed_s={now - t0:.0f} pid={proc.pid}")
                last_hb = now
    finally:
        if fh is not None:
            fh.close()
    dt = time.time() - t0
    log(f"  rc={proc.returncode} wall_s={dt:.1f}" + (f" log={log_path}" if log_path else ""))
    return int(proc.returncode or 0)


def f32_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        raw = f.read(16)
    if len(raw) != 16:
        raise RuntimeError(f"short header {path}")
    n, d = struct.unpack("<QQ", raw)
    return int(n), int(d)


def done_run(d: Path) -> bool:
    csv = d / "compare_cost_log.csv"
    js = d / "timing_summary.json"
    if not csv.is_file() or not js.is_file():
        return False
    first = csv.open(encoding="utf-8").readline().strip()
    return first == "step,chamfer,strat_est,is_est,strat_err,is_err"


def cleanup_store(d: Path) -> None:
    store = d / "store"
    if store.exists():
        shutil.rmtree(store, ignore_errors=True)


def phase_build() -> None:
    log("=== BUILD ===")
    build = ROOT / "build"
    if not (build / "build.ninja").is_file() and not (build / "Makefile").is_file():
        rc = run_cmd(
            ["cmake", "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER=g++",
             "-DCMAKE_C_COMPILER=gcc"],
            timeout=600,
        )
        if rc != 0:
            raise SystemExit("cmake configure failed")
    rc = run_cmd(
        ["cmake", "--build", str(build), "--target",
         "compare_perf", "compare_perf_sift",
         "compare_perf_d300", "compare_perf_gist", "exact_cache_f32"],
        timeout=1800,
    )
    if rc != 0:
        raise SystemExit("cmake build failed")


def update_gist_meta() -> None:
    meta_path = ROOT / "data" / "GIST_1M" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files = meta.setdefault("files", {})
    files["base.f32bin"] = "base.f32bin"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def phase_data() -> None:
    log("=== DATA gist base.f32bin ===")
    src = ROOT / "data" / "GIST_1M" / "base_with_outlier.f32bin"
    dst = ROOT / "data" / "GIST_1M" / "base.f32bin"
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    n, d = f32_header(src)
    if n != 1_000_001 or d != 960:
        raise SystemExit(f"unexpected gist outlier header n={n} d={d}")
    keep = 1_000_000
    row = d * 4
    need = 16 + keep * row
    if dst.is_file() and dst.stat().st_size == need:
        n2, d2 = f32_header(dst)
        if n2 == keep and d2 == d:
            log(f"  ok {dst} bytes={need}")
            update_gist_meta()
            return
    free = shutil.disk_usage(src.parent).free
    if free < need + (1 << 30):
        raise SystemExit(f"need ~{need} bytes free, have {free}")
    log(f"  slicing first {keep} rows -> {dst}")
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.read(16)
        fout.write(struct.pack("<QQ", keep, d))
        remaining = keep * row
        while remaining:
            chunk = fin.read(min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise SystemExit("short read while slicing gist base")
            fout.write(chunk)
            remaining -= len(chunk)
    if dst.stat().st_size != need:
        raise SystemExit("gist base.f32bin size mismatch")
    update_gist_meta()
    log(f"  wrote {dst} bytes={need}")


def phase_exact() -> None:
    log("=== EXACT gist no-outlier ===")
    out_dir = OUT / "gist" / "exact_no"
    cache = out_dir / "exact_cache.csv"
    timing = Path(str(cache) + ".timing.json")
    if cache.is_file() and timing.is_file():
        n, _d = f32_header(ROOT / "data" / "GIST_1M" / "base.f32bin")
        text = cache.read_text(encoding="utf-8")
        if f"|A|={n}" in text and "window_size=50" in text and "query_interval=17" in text:
            log("SKIP exact gist-no")
            return
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = run_cmd(
        [
            str(which_bin("exact_cache_f32")),
            "--data-dir", str(ROOT / "data" / "GIST_1M"),
            "--out", str(cache),
            "--a-name", "base.f32bin",
            "--window", "50",
            "--query-interval", "17",
            "--timing-out", str(timing),
        ],
        log_path=out_dir / "run.log",
        timeout=21600,
    )
    if rc != 0:
        raise SystemExit("gist no-outlier exact cache failed")


def compare_cmd(job: dict, kind: str, seed: int, out_dir: Path) -> list[str]:
    cache = job["cache_no"] if kind == "no" else job["cache_out"]
    cmd = [
        str(which_bin(job["bin"])),
        "--e2lsh-only", "--with-is",
        "--data-dir", str(job["data"]),
        "--window-size", str(job["window"]),
        "--query-interval", str(job["interval"]),
        "--T", str(job["T"]),
        "--seed", str(seed),
        "--exact-cache", str(cache),
        "--out", str(out_dir / "compare_cost_log.csv"),
        "--timing-out", str(out_dir / "timing_summary.json"),
    ]
    if kind == "no":
        cmd.append("--no-outlier")
    if job["n_max"] is not None:
        cmd += ["--n-max", str(job["n_max"])]
    if job["storage"]:
        store = out_dir / "store"
        store.mkdir(parents=True, exist_ok=True)
        cmd += ["--storage-dir", str(store)]
    return cmd


def check_timing(job: dict, kind: str, seed: int, out_dir: Path) -> None:
    t = json.loads((out_dir / "timing_summary.json").read_text(encoding="utf-8"))
    if int(t.get("seed")) != seed:
        raise SystemExit(f"seed mismatch {out_dir}")
    if int(t.get("queries", 0)) != job["queries"]:
        raise SystemExit(f"queries {t.get('queries')} != {job['queries']} in {out_dir}")
    if t.get("with_is") is not True:
        raise SystemExit(f"with_is not set in {out_dir}")
    if "e2lsh_is" not in t:
        raise SystemExit(f"missing e2lsh_is in {out_dir}")
    if job["storage"] and t.get("storage_backend") != "external":
        raise SystemExit(f"gist run was not external: {out_dir}")
    log(f"  seed={seed} kind={kind} queries={t['queries']} "
        f"wall_s={t['wall_ms_total'] / 1000:.2f} "
        f"strat_ms={t['e2lsh_stratified']['query_ms_avg']:.3f} "
        f"is_ms={t['e2lsh_is']['query_ms_avg']:.3f}")


def run_one(job: dict, kind: str, seed: int) -> None:
    out_dir = OUT / job["name"] / f"{kind}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if done_run(out_dir):
        cleanup_store(out_dir)
        log(f"SKIP {out_dir.relative_to(ROOT)}")
        return
    cache = job["cache_no"] if kind == "no" else job["cache_out"]
    if not cache.is_file():
        raise SystemExit(f"missing exact cache {cache}")
    cleanup_store(out_dir)
    cmd = compare_cmd(job, kind, seed, out_dir)
    rc = run_cmd(cmd, log_path=out_dir / "run.log", timeout=job["timeout"])
    if rc != 0:
        raise SystemExit(f"compare failed {out_dir}")
    check_timing(job, kind, seed, out_dir)
    cleanup_store(out_dir)


def strat_column(path: Path) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("step") or line.startswith("#"):
            continue
        rows.append(line.split(",")[2])
    return rows


def phase_smoke() -> None:
    log("=== SMOKE textemb no seed1: stratified column unchanged by --with-is ===")
    job = JOBS[0]
    with_dir = OUT / "textemb" / "no_seed1"
    only_dir = OUT / "smoke" / "textemb_no_seed1_strat_only"
    run_one(job, "no", 1)
    only_dir.mkdir(parents=True, exist_ok=True)
    only_csv = only_dir / "compare_cost_log.csv"
    only_js = only_dir / "timing_summary.json"
    if not (only_csv.is_file() and only_js.is_file()):
        cmd = compare_cmd(job, "no", 1, only_dir)
        cmd = [c for c in cmd if c != "--with-is"]
        rc = run_cmd(cmd, log_path=only_dir / "run.log", timeout=job["timeout"])
        if rc != 0:
            raise SystemExit("smoke strat-only failed")
    a = strat_column(with_dir / "compare_cost_log.csv")
    b = strat_column(only_csv)
    if a != b:
        raise SystemExit(
            f"smoke mismatch: with-is strat column != strat-only "
            f"(n={len(a)} vs {len(b)}, first diff soon)"
        )
    if not a:
        raise SystemExit("smoke produced no query rows")
    log(f"  smoke OK rows={len(a)} stratified estimates match")


def phase_run() -> None:
    log("=== RUN matrix ===")
    for job in JOBS:
        for kind in ("no", "out"):
            for seed in SEEDS:
                run_one(job, kind, seed)


def load_errs(path: Path) -> tuple[list[float], list[float]]:
    strat, is_err = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("step") or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise SystemExit(f"bad csv row in {path}: {line}")
        strat.append(float(parts[4]))
        is_err.append(float(parts[5]))
    return strat, is_err


def summarize(errs: list[float]) -> dict[str, float]:
    n = len(errs)
    if n == 0:
        raise SystemExit("empty error list")
    mean = sum(errs) / n
    mean_abs = sum(abs(e) for e in errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    var = sum((e - mean) ** 2 for e in errs) / (n - 1) if n >= 2 else 0.0
    return {"n": n, "mean_signed": mean, "mean_abs": mean_abs, "rmse": rmse, "var": var}


def phase_aggregate() -> None:
    log("=== AGGREGATE ===")
    cells = []
    for job in JOBS:
        for kind in ("no", "out"):
            per_seed = []
            for seed in SEEDS:
                d = OUT / job["name"] / f"{kind}_seed{seed}"
                if not done_run(d):
                    raise SystemExit(f"missing run {d}")
                strat, is_err = load_errs(d / "compare_cost_log.csv")
                timing = json.loads((d / "timing_summary.json").read_text(encoding="utf-8"))
                per_seed.append({
                    "seed": seed,
                    "strat": summarize(strat),
                    "is": summarize(is_err),
                    "strat_query_ms_avg": timing["e2lsh_stratified"]["query_ms_avg"],
                    "is_query_ms_avg": timing["e2lsh_is"]["query_ms_avg"],
                    "wall_s": timing["wall_ms_total"] / 1000.0,
                    "K": timing.get("K"),
                    "L_final": timing.get("L_final"),
                    "queries": timing.get("queries"),
                })
            def avg(key_est: str, key_stat: str) -> float:
                return sum(s[key_est][key_stat] for s in per_seed) / len(per_seed)

            cell = {
                "dataset": job["name"],
                "outlier": kind,
                "T": job["T"],
                "window": job["window"],
                "query_interval": job["interval"],
                "queries": per_seed[0]["queries"],
                "strat_mean_abs": avg("strat", "mean_abs"),
                "is_mean_abs": avg("is", "mean_abs"),
                "strat_rmse": avg("strat", "rmse"),
                "is_rmse": avg("is", "rmse"),
                "strat_mean_signed": avg("strat", "mean_signed"),
                "is_mean_signed": avg("is", "mean_signed"),
                "strat_var": avg("strat", "var"),
                "is_var": avg("is", "var"),
                "strat_query_ms_avg": sum(s["strat_query_ms_avg"] for s in per_seed) / len(per_seed),
                "is_query_ms_avg": sum(s["is_query_ms_avg"] for s in per_seed) / len(per_seed),
                "seeds": per_seed,
            }
            cells.append(cell)
            log(f"  {job['name']} {kind} strat|e|={cell['strat_mean_abs']:.5f} "
                f"is|e|={cell['is_mean_abs']:.5f}")

    payload = {"cells": cells}
    (OUT / "aggregate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"  wrote {OUT / 'aggregate.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="all",
        choices=["build", "data", "exact", "smoke", "run", "aggregate", "all"],
    )
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"START phase={args.phase} pid={os.getpid()} "
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    phases = {
        "build": phase_build,
        "data": phase_data,
        "exact": phase_exact,
        "smoke": phase_smoke,
        "run": phase_run,
        "aggregate": phase_aggregate,
    }
    order = ["build", "data", "exact", "smoke", "run", "aggregate"]
    if args.phase != "all":
        order = [args.phase]
    for name in order:
        phases[name]()
    log(f"DONE {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
