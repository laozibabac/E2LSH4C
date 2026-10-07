#!/usr/bin/env python3
"""Static A, sliding-window B on the four Section 5 datasets.

One OpenMP thread. QuadTree and Uniform use simd_l2::min_l2_flat.
Brute Force times come from exact_cache_f32, also one thread.
Jobs run one at a time. Results go to output/paper_uni/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "paper_uni"
LOG = OUT / "batch.log"
SEEDS = [1, 2, 3, 4, 5]

TE_W, TE_T, TE_Q = 100, 150, 20
FASHION_W, FASHION_T, FASHION_Q = 500, 200, 166
SIFT_W, SIFT_T, SIFT_Q = 500, 300, 166
# GIST-1M: window |B|/20, query every 26 steps (37 queries on |B|=1000).
GIST_W, GIST_T, GIST_Q = 50, 300, 26
GIST_N_MAX = 1_100_000


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = msg if msg.endswith("\n") else msg + "\n"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)
    print(msg, flush=True)


def write_state(text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runner.state").write_text(text + "\n", encoding="utf-8")


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
        while proc.poll() is None:
            if deadline is not None and time.time() > deadline:
                proc.kill()
                proc.wait()
                log(f"  TIMEOUT after {timeout}s")
                return 124
            time.sleep(15)
            log(f"  HEARTBEAT {Path(cmd[0]).name} elapsed_s={time.time() - t0:.0f} pid={proc.pid}")
    finally:
        if fh is not None:
            fh.close()
    dt = time.time() - t0
    log(f"  rc={proc.returncode} wall_s={dt:.1f}" + (f" log={log_path}" if log_path else ""))
    return int(proc.returncode or 0)


def _timing_ok(path: Path, *, joint: bool) -> bool:
    if not path.is_file():
        return False
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if j.get("omp_threads") != 1:
        return False
    if j.get("nn_kernel") != "simd_l2::min_l2_flat":
        return False
    if joint:
        status = (j.get("nn_microbench") or {}).get("status")
        if status != "OK":
            return False
    return True


def done_timing(d: Path, *, joint: bool) -> bool:
    return _timing_ok(d / "timing_summary.json", joint=joint) and (d / "compare_cost_log.csv").is_file()


def done_exact(d: Path) -> bool:
    timing = d / "exact_cache.csv.timing.json"
    if not (d / "exact_cache.csv").is_file() or not timing.is_file():
        return False
    try:
        j = json.loads(timing.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return j.get("omp_threads") == 1 and "Benchmark" in (j.get("per_update_ms") or {})


def phase_build() -> None:
    log("=== BUILD ===")
    build = ROOT / "build"
    rc = run_cmd(["cmake", "-S", str(ROOT), "-B", str(build)])
    if rc != 0:
        raise SystemExit("cmake configure failed")
    rc = run_cmd(
        ["cmake", "--build", str(build), "--target",
         "compare_perf", "compare_perf_sift", "compare_perf_gist",
         "compare_perf_d300", "exact_cache_f32"],
        timeout=1800,
    )
    if rc != 0:
        raise SystemExit("cmake build failed")
    for name in ("compare_perf", "compare_perf_sift", "compare_perf_gist",
                 "compare_perf_d300", "exact_cache_f32"):
        which_bin(name)
    log("  binaries ok")


def _check_compare_smoke(text: str) -> None:
    if "omp_threads=1" not in text:
        raise SystemExit("smoke: omp_threads is not 1")
    if re.search(r"NN_MICROBENCH\b.*\bstatus=OK\b", text) is None:
        raise SystemExit("smoke: NN microbench did not report status=OK")
    if "status=FAIL" in text:
        raise SystemExit("smoke: NN microbench status=FAIL")


def phase_smoke() -> None:
    log("=== SMOKE ===")
    fashion = OUT / "smoke" / "fashion"
    fashion.mkdir(parents=True, exist_ok=True)
    rc = run_cmd(
        [str(which_bin("compare_perf")),
         "--data-dir", str(ROOT / "data" / "Fashion-MNIST"),
         "--out", str(fashion / "compare_cost_log.csv"),
         "--timing-out", str(fashion / "timing_summary.json"),
         "--window-size", "500", "--query-interval", "166", "--T", "200",
         "--no-outlier", "--seed", "1",
         "--max-A", "2000", "--max-B", "900", "--n-max", "4000"],
        log_path=fashion / "run.log",
        timeout=1800,
    )
    if rc != 0:
        raise SystemExit(f"fashion smoke failed rc={rc}")
    _check_compare_smoke((fashion / "run.log").read_text(encoding="utf-8", errors="replace"))
    if not done_timing(fashion, joint=True):
        raise SystemExit("fashion smoke timing json rejected")

    sift = OUT / "smoke" / "sift"
    sift.mkdir(parents=True, exist_ok=True)
    rc = run_cmd(
        [str(which_bin("compare_perf_sift")),
         "--data-dir", str(ROOT / "data" / "SIFT"),
         "--out", str(sift / "compare_cost_log.csv"),
         "--timing-out", str(sift / "timing_summary.json"),
         "--window-size", "500", "--query-interval", "166", "--T", "300",
         "--no-outlier", "--seed", "1",
         "--max-A", "8000", "--max-B", "900", "--n-max", "16000"],
        log_path=sift / "run.log",
        timeout=1800,
    )
    if rc != 0:
        raise SystemExit(f"sift smoke failed rc={rc}")
    _check_compare_smoke((sift / "run.log").read_text(encoding="utf-8", errors="replace"))
    if not done_timing(sift, joint=True):
        raise SystemExit("sift smoke timing json rejected")

    exact = OUT / "smoke" / "exact"
    exact.mkdir(parents=True, exist_ok=True)
    rc = run_cmd(
        [str(which_bin("exact_cache_f32")),
         "--data-dir", str(ROOT / "data" / "Fashion-MNIST"),
         "--a-name", "train.f32bin",
         "--out", str(exact / "exact_cache.csv"),
         "--timing-out", str(exact / "exact_cache.csv.timing.json"),
         "--window", "500", "--query-interval", "166",
         "--max-A", "1500", "--max-B", "700"],
        log_path=exact / "run.log",
        timeout=600,
    )
    if rc != 0:
        raise SystemExit(f"exact smoke failed rc={rc}")
    text = (exact / "run.log").read_text(encoding="utf-8", errors="replace")
    if "omp_threads=1" not in text:
        raise SystemExit("exact smoke: omp_threads is not 1")
    log("  smoke ok")


def exact_job(label: str, data_dir: Path, a_name: str, out_dir: Path,
              window: int, qint: int, timeout: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if done_exact(out_dir):
        log(f"SKIP exact {label}")
        return
    cmd = [
        str(which_bin("exact_cache_f32")),
        "--data-dir", str(data_dir),
        "--out", str(out_dir / "exact_cache.csv"),
        "--a-name", a_name,
        "--window", str(window),
        "--query-interval", str(qint),
        "--timing-out", str(out_dir / "exact_cache.csv.timing.json"),
    ]
    rc = run_cmd(cmd, log_path=out_dir / "run.log", timeout=timeout)
    if rc != 0:
        raise SystemExit(f"exact {label} failed")
    if not done_exact(out_dir):
        raise SystemExit(f"exact {label} timing rejected")


def compare_job(bin_name: str, data_dir: Path, out_dir: Path, extra: list[str],
                timeout: int, *, joint: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if done_timing(out_dir, joint=joint):
        log(f"SKIP {out_dir.relative_to(OUT)}")
        return
    cmd = [
        str(which_bin(bin_name)),
        "--data-dir", str(data_dir),
        "--out", str(out_dir / "compare_cost_log.csv"),
        "--timing-out", str(out_dir / "timing_summary.json"),
        *extra,
    ]
    rc = run_cmd(cmd, log_path=out_dir / "run.log", timeout=timeout)
    if rc != 0:
        raise SystemExit(f"compare failed {out_dir}")
    if not done_timing(out_dir, joint=joint):
        raise SystemExit(f"compare timing rejected {out_dir}")
    t = json.loads((out_dir / "timing_summary.json").read_text(encoding="utf-8"))
    log(f"  seed={t.get('seed')} queries={t.get('queries')} per={t.get('per_update_ms')}")


def _pair(name: str, data: Path, bin_name: str, window: int, t: int, q: int,
          a_no: str, a_out: str, timeout_exact: int, timeout_cmp: int,
          extra: list[str] | None = None, *, joint: bool) -> None:
    log(f"=== {name} ===")
    extra = extra or []
    common = ["--window-size", str(window), "--query-interval", str(q), "--T", str(t), *extra]
    exact_job(f"{name}-no", data, a_no, OUT / name / "exact_no", window, q, timeout_exact)
    cache_no = OUT / name / "exact_no" / "exact_cache.csv"
    for s in SEEDS:
        compare_job(
            bin_name, data, OUT / name / f"no_seed{s}",
            common + ["--no-outlier", "--seed", str(s), "--exact-cache", str(cache_no)],
            timeout_cmp, joint=joint,
        )
    exact_job(f"{name}-out", data, a_out, OUT / name / "exact_out", window, q, timeout_exact)
    cache_out = OUT / name / "exact_out" / "exact_cache.csv"
    for s in SEEDS:
        compare_job(
            bin_name, data, OUT / name / f"out_seed{s}",
            common + ["--seed", str(s), "--exact-cache", str(cache_out)],
            timeout_cmp, joint=joint,
        )


def phase_textemb() -> None:
    _pair("textemb", ROOT / "data" / "TextEmbedding", "compare_perf_d300",
          TE_W, TE_T, TE_Q, "base.f32bin", "base_with_outlier.f32bin",
          1800, 1800, joint=True)


def phase_fashion() -> None:
    _pair("fashion", ROOT / "data" / "Fashion-MNIST", "compare_perf",
          FASHION_W, FASHION_T, FASHION_Q, "train.f32bin", "train_with_outlier.f32bin",
          7200, 7200, joint=True)


def phase_sift() -> None:
    log("=== SIFT exact ===")
    data = ROOT / "data" / "SIFT"
    exact_job("sift-no", data, "base.f32bin", OUT / "sift" / "exact_no",
              SIFT_W, SIFT_Q, 14400)
    exact_job("sift-out", data, "base_with_outlier.f32bin", OUT / "sift" / "exact_out",
              SIFT_W, SIFT_Q, 14400)
    cache_no = OUT / "sift" / "exact_no" / "exact_cache.csv"
    cache_out = OUT / "sift" / "exact_out" / "exact_cache.csv"
    common = ["--window-size", str(SIFT_W), "--query-interval", str(SIFT_Q),
              "--T", str(SIFT_T), "--n-max", "1100000"]
    log("=== SIFT compare ===")
    for s in SEEDS:
        compare_job(
            "compare_perf_sift", data, OUT / "sift" / f"no_qt_seed{s}",
            common + ["--no-outlier", "--quadtree-only", "--seed", str(s),
                      "--exact-cache", str(cache_no)],
            14400, joint=False,
        )
        compare_job(
            "compare_perf_sift", data, OUT / "sift" / f"no_e2_seed{s}",
            common + ["--no-outlier", "--e2lsh-only", "--seed", str(s),
                      "--exact-cache", str(cache_no)],
            14400, joint=False,
        )
    for s in SEEDS:
        compare_job(
            "compare_perf_sift", data, OUT / "sift" / f"out_qt_seed{s}",
            common + ["--quadtree-only", "--seed", str(s), "--exact-cache", str(cache_out)],
            14400, joint=False,
        )
        compare_job(
            "compare_perf_sift", data, OUT / "sift" / f"out_e2_seed{s}",
            common + ["--e2lsh-only", "--seed", str(s), "--exact-cache", str(cache_out)],
            14400, joint=False,
        )


def _cache_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("step,"):
                break
            if line.startswith("#") and "=" in line:
                key, val = line[1:].split("=", 1)
                meta[key.strip()] = val.strip()
    return meta


def _install_gist_cache(kind: str, src: Path, nA: int) -> Path:
    dest_dir = OUT / "gist" / f"exact_{kind}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "exact_cache.csv"
    if dest.is_file():
        have = _cache_meta(dest)
        if (have.get("|A|") == str(nA) and have.get("window_size") == str(GIST_W)
                and have.get("query_interval") == str(GIST_Q)):
            log(f"keep exact cache {dest}")
            return dest
    if not src.is_file():
        raise SystemExit(f"missing GIST exact cache {src}")
    meta = _cache_meta(src)
    if meta.get("|A|") != str(nA) or meta.get("window_size") != str(GIST_W):
        raise SystemExit(f"GIST cache {src} header {meta} does not match |A|={nA} window={GIST_W}")
    if meta.get("query_interval") != str(GIST_Q):
        raise SystemExit(f"GIST cache {src} query_interval={meta.get('query_interval')}")
    if not dest.is_file() or dest.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dest)
        log(f"copied exact cache {src} -> {dest}")
    return dest


def _require_gist_timing(kind: str) -> None:
    timing = OUT / "gist" / f"exact_{kind}" / "exact_cache.csv.timing.json"
    if not timing.is_file():
        raise SystemExit(
            f"missing {timing}; run exact_cache_f32 on data/GIST_1M "
            f"with window {GIST_W} and query interval {GIST_Q}"
        )


def phase_gist() -> None:
    log("=== GIST ===")
    data = ROOT / "data" / "GIST_1M"
    caches = {
        "no": (OUT / "gist" / "exact_no" / "exact_cache.csv",
               "base.f32bin", 1_000_000),
        "out": (OUT / "gist" / "exact_out" / "exact_cache.csv",
                "base_with_outlier.f32bin", 1_000_001),
    }
    installed = {}
    for kind, (src, a_name, nA) in caches.items():
        installed[kind] = _install_gist_cache(kind, src, nA)
        _require_gist_timing(kind)
    common = ["--window-size", str(GIST_W), "--query-interval", str(GIST_Q),
              "--T", str(GIST_T), "--n-max", str(GIST_N_MAX)]
    log("=== GIST compare ===")
    for kind, flag in (("no", ["--no-outlier"]), ("out", [])):
        cache = installed[kind]
        for s in SEEDS:
            compare_job(
                "compare_perf_gist", data, OUT / "gist" / f"{kind}_qt_seed{s}",
                common + flag + ["--quadtree-only", "--seed", str(s), "--exact-cache", str(cache)],
                21600, joint=False,
            )
            compare_job(
                "compare_perf_gist", data, OUT / "gist" / f"{kind}_e2_seed{s}",
                common + flag + ["--e2lsh-only", "--seed", str(s), "--exact-cache", str(cache)],
                21600, joint=False,
            )


def phase_plot() -> None:
    log("=== PLOT ===")
    rc = run_cmd([sys.executable, str(ROOT / "scripts" / "plot_paper_uni_fair.py")])
    if rc != 0:
        raise SystemExit("plot failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["build", "smoke", "textemb", "gist", "fashion", "sift",
                             "plot", "run", "all"])
    args = ap.parse_args()
    write_state(f"running\npid={os.getpid()}\nphase={args.phase}")
    (OUT / "runner.pid").write_text(str(os.getpid()), encoding="utf-8")
    log(f"START phase={args.phase} pid={os.getpid()} "
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    order = {
        "build": [phase_build],
        "smoke": [phase_smoke],
        "textemb": [phase_textemb],
        "gist": [phase_gist],
        "fashion": [phase_fashion],
        "sift": [phase_sift],
        "plot": [phase_plot],
        "run": [phase_textemb, phase_gist, phase_fashion, phase_sift, phase_plot],
        "all": [phase_build, phase_smoke, phase_textemb, phase_gist,
                phase_fashion, phase_sift, phase_plot],
    }[args.phase]
    try:
        for phase in order:
            phase()
    except SystemExit as exc:
        write_state(f"exit=1\n{exc}")
        log(f"FAILED {exc}")
        raise
    write_state("exit=0")
    log(f"FINISHED phase={args.phase}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        write_state(f"exit=1\n{exc}")
        raise
