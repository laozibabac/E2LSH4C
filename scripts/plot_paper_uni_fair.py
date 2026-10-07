#!/usr/bin/env python3
"""Section 5 static-A figures from output/paper_uni.

Writes results/figures/figure3.png through figure6.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "paper_uni"
FIGS = ROOT / "results" / "figures"
SEEDS = [1, 2, 3, 4, 5]

COLORS = {
    "E2LSH": "#2a9d8f",
    "QuadTree": "#d62728",
    "Uniform": "#f4a261",
    "Benchmark": "#2f6db3",
}
LINESTYLE = {
    "E2LSH": ("-", "o"),
    "QuadTree": ("--", "s"),
    "Uniform": ("--", "^"),
}
LEGEND = {
    "E2LSH": "Ours (E2LSH4C)",
    "QuadTree": "QuadTree",
    "Uniform": "Uniform",
    "Benchmark": "Brute Force",
}
ERROR_METHODS = ("E2LSH", "QuadTree", "Uniform")


def load_json(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(raw.replace("\\", "/"))


def require_timing(path: Path, *, joint: bool | None) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    j = load_json(path)
    if j.get("omp_threads") != 1:
        raise SystemExit(f"{path} omp_threads={j.get('omp_threads')}")
    kernel = j.get("nn_kernel")
    if joint is None:
        if kernel != "squared_l2_f32":
            raise SystemExit(f"{path} nn_kernel={kernel}")
        if "Benchmark" not in (j.get("per_update_ms") or {}):
            raise SystemExit(f"{path} missing per_update_ms.Benchmark")
        return j
    if kernel != "simd_l2::min_l2_flat":
        raise SystemExit(f"{path} nn_kernel={kernel}")
    if joint:
        status = (j.get("nn_microbench") or {}).get("status")
        if status != "OK":
            raise SystemExit(f"{path} nn_microbench status={status}")
    per = j.get("per_update_ms") or {}
    if not per:
        raise SystemExit(f"{path} missing per_update_ms")
    return j


def read_csv_err(path: Path) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(float(row["step"]))
            for name, col in (
                ("QuadTree", "quadtree_err"),
                ("E2LSH", "e2lsh_err"),
                ("E2LSH", "strat_err"),
                ("Uniform", "uniform_err"),
            ):
                if col in row and row[col] not in (None, ""):
                    out.setdefault(name, []).append((step, abs(float(row[col]))))
    return out


def load_joint(folder: str, prefix: str) -> list[dict[str, list[tuple[int, float]]]]:
    rows = []
    for s in SEEDS:
        d = OUT / folder / f"{prefix}_seed{s}"
        require_timing(d / "timing_summary.json", joint=True)
        path = d / "compare_cost_log.csv"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        rows.append(read_csv_err(path))
    return rows


def load_split(folder: str, prefix: str) -> list[dict[str, list[tuple[int, float]]]]:
    rows = []
    for s in SEEDS:
        qt = OUT / folder / f"{prefix}_qt_seed{s}"
        e2 = OUT / folder / f"{prefix}_e2_seed{s}"
        require_timing(qt / "timing_summary.json", joint=False)
        require_timing(e2 / "timing_summary.json", joint=False)
        merged = read_csv_err(qt / "compare_cost_log.csv")
        e2_err = read_csv_err(e2 / "compare_cost_log.csv")
        merged["E2LSH"] = e2_err.get("E2LSH", [])
        rows.append(merged)
    return rows


def band(seed_series, name: str):
    steps = None
    mats = []
    for series in seed_series:
        if name not in series or not series[name]:
            continue
        st, ys = zip(*series[name])
        if steps is None:
            steps = list(st)
        mats.append(list(ys))
    if not mats or steps is None:
        return None
    mlen = min(len(steps), min(len(m) for m in mats))
    steps = steps[:mlen]
    arr = np.array([m[:mlen] for m in mats], dtype=float)
    return steps, arr.mean(0), arr.min(0), arr.max(0)


def _ylim_for_series(series) -> float:
    focus = []
    uni_p95 = None
    for method in ERROR_METHODS:
        b = band(series, method)
        if b is None:
            continue
        p95 = float(np.percentile(b[1], 95))
        if method == "Uniform":
            uni_p95 = p95
        else:
            focus.append(p95)
    ytop = max(0.08, (max(focus) if focus else 0.04) * 2.0)
    if uni_p95 is not None and uni_p95 < 0.5:
        ytop = max(ytop, uni_p95 * 1.35)
    elif uni_p95 is not None:
        ytop = max(ytop, 0.16)
    return ytop


def _draw_error_ax(ax, series, name: str) -> None:
    ax.set_title(name)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Relative Error")
    for method in ERROR_METHODS:
        b = band(series, method)
        if b is None:
            continue
        steps, mu, lo, hi = b
        ls, mk = LINESTYLE[method]
        ax.plot(steps, mu, color=COLORS[method], label=LEGEND[method], lw=1.4,
                ls=ls, marker=mk, markersize=3.5, markevery=max(len(steps) // 16, 1))
        ax.fill_between(steps, lo, hi, color=COLORS[method], alpha=0.22, linewidth=0)
    ax.set_ylim(0.0, _ylim_for_series(series))
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=8, loc="upper right")


def plot_error_grid(panels, title: str, out_path: Path) -> None:
    n = len(panels)
    ncols = 2
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 3.45 * nrows), sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (series, name) in zip(axes_flat, panels):
        _draw_error_ax(ax, series, name)
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(title, y=1.01, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def mean_per(paths: list[Path], method: str, *, joint: bool) -> float:
    vals = []
    for path in paths:
        j = require_timing(path, joint=joint)
        per = j.get("per_update_ms") or {}
        if method not in per:
            raise SystemExit(f"{path} missing per_update_ms.{method}")
        vals.append(float(per[method]))
    if len(vals) != len(paths):
        raise SystemExit(f"incomplete mean for {method}")
    return statistics.mean(vals)


def time_row_joint(folder: str, kind: str) -> dict[str, float]:
    paths = [OUT / folder / f"{kind}_seed{s}" / "timing_summary.json" for s in SEEDS]
    exact = require_timing(OUT / folder / f"exact_{kind}" / "exact_cache.csv.timing.json", joint=None)
    return {
        "E2LSH": mean_per(paths, "E2LSH", joint=True),
        "QuadTree": mean_per(paths, "QuadTree", joint=True),
        "Uniform": mean_per(paths, "Uniform", joint=True),
        "Benchmark": float(exact["per_update_ms"]["Benchmark"]),
    }


def time_row_split(folder: str, kind: str) -> dict[str, float]:
    qt = [OUT / folder / f"{kind}_qt_seed{s}" / "timing_summary.json" for s in SEEDS]
    e2 = [OUT / folder / f"{kind}_e2_seed{s}" / "timing_summary.json" for s in SEEDS]
    exact = require_timing(OUT / folder / f"exact_{kind}" / "exact_cache.csv.timing.json", joint=None)
    return {
        "E2LSH": mean_per(e2, "E2LSH", joint=False),
        "QuadTree": mean_per(qt, "QuadTree", joint=False),
        "Uniform": mean_per(qt, "Uniform", joint=False),
        "Benchmark": float(exact["per_update_ms"]["Benchmark"]),
    }


def collect_time(kind: str) -> dict[str, dict[str, float]]:
    return {
        "Text Embedding": time_row_joint("textemb", kind),
        "GIST": time_row_split("gist", kind),
        "Fashion-MNIST": time_row_joint("fashion", kind),
        "SIFT": time_row_split("sift", kind),
    }


def plot_time_bars(rows: dict[str, dict[str, float]], title: str, out_path: Path) -> None:
    methods = ["Uniform", "Benchmark", "QuadTree", "E2LSH"]
    legend_order = ["E2LSH", "QuadTree", "Benchmark", "Uniform"]
    datasets = [d for d in ("SIFT", "Fashion-MNIST", "GIST", "Text Embedding") if d in rows]
    n_m = len(methods)
    fig_h = 2.2 + 1.05 * len(datasets)
    fig, ax = plt.subplots(figsize=(9.6, fig_h))
    y = np.arange(len(datasets))
    height = 0.16
    handles = {}
    for i, m in enumerate(methods):
        ys = [rows[d][m] for d in datasets]
        offs = (i - (n_m - 1) / 2) * height
        bars = ax.barh(y + offs, ys, height, label=LEGEND[m], color=COLORS[m])
        handles[m] = bars
        for bar, val in zip(bars, ys):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.3f}",
                    va="center", ha="left", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Running Time (ms)")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.legend([handles[m] for m in legend_order], [LEGEND[m] for m in legend_order],
              frameon=True, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(True, axis="x", alpha=0.3, which="both")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def err_summary(series) -> dict:
    out = {}
    for method in ERROR_METHODS:
        means = []
        for s in series:
            ys = [y for _, y in s.get(method, [])]
            if ys:
                means.append(sum(ys) / len(ys))
        if len(means) != len(SEEDS):
            raise SystemExit(f"{method} missing seeds ({len(means)})")
        out[method] = {"mean_abs": statistics.mean(means), "n_seeds": len(means)}
    return out


def write_table_and_results(agg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        ["dataset", "d", "|A|", "|B|", "window", "T"],
        ["Text Embedding", "300", "1880 / 1881", "1178", "100", "150"],
        ["GIST", "960", "1000000 / 1000001", "1000", "50", "300"],
        ["Fashion-MNIST", "784", "60000 / 60001", "10000", "500", "200"],
        ["SIFT", "128", "1000000 / 1000001", "10000", "500", "300"],
    ]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    for row in rows[1:]:
        md.append("| " + " | ".join(row) + " |")
    table = "\n".join(md) + "\n"
    (OUT / "table1.md").write_text(table, encoding="utf-8")
    names = {"E2LSH": "Ours (E2LSH4C)", "QuadTree": "QuadTree", "Uniform": "Uniform"}
    lines = [
        "# Unidirectional results",
        "",
        "Static A, sliding-window B. Ours is native ℓ₂ E2LSH4C. QuadTree is the NeurIPS 2025 baseline.",
        "The exact baseline is Brute Force.",
        "One OpenMP thread. QuadTree and Uniform use `simd_l2::min_l2_flat`.",
        "The four plotted datasets are Text Embedding, GIST, Fashion-MNIST, and SIFT.",
        "",
        "## Protocol",
        "",
        "- Text Embedding: window 100, query every 20, T=150.",
        "- GIST: window 50, query every 26, T=300, `|B|=1000` (37 queries).",
        "- Fashion-MNIST: window 500, query every 166, T=200.",
        "- SIFT: window 500, query every 166, T=300.",
        "- 5 seeds. Outlier is `ã = 0.1|A|(A[0]-mean(A))+mean(A)`.",
        "- Brute Force is exact maintenance, one OpenMP thread.",
        "",
        "## Table 1",
        "",
        table.rstrip(),
        "",
        "## 5-seed mean |relative error|",
        "",
    ]
    for key, title in (
        ("textemb_no", "Text Embedding, no outlier"),
        ("textemb_out", "Text Embedding, with outlier"),
        ("gist_no", "GIST, no outlier"),
        ("gist_out", "GIST, with outlier"),
        ("fashion_no", "Fashion-MNIST, no outlier"),
        ("fashion_out", "Fashion-MNIST, with outlier"),
        ("sift_no", "SIFT, no outlier"),
        ("sift_out", "SIFT, with outlier"),
    ):
        block = agg[key]
        bits = [f"{names[m]} {block[m]['mean_abs']:.4f}" for m in ("E2LSH", "QuadTree", "Uniform")]
        lines.append(f"- **{title}**: " + "; ".join(bits))
    lines += ["", "## Time per update (ms)", ""]
    for key, title in (("time_no", "no outliers"), ("time_out", "with outliers")):
        lines.append(f"### {title}")
        lines.append("")
        for ds in ("Text Embedding", "GIST", "Fashion-MNIST", "SIFT"):
            row = agg[key][ds]
            lines.append(
                f"- **{ds}**: E2LSH4C {row['E2LSH']:.4f}; QuadTree {row['QuadTree']:.4f}; "
                f"Uniform {row['Uniform']:.4f}; Brute Force {row['Benchmark']:.4f}"
            )
        lines.append("")
    lines.append("Figures: `results/figures/figure3.png`–`figure6.png`.")
    lines.append("")
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def render(figs: Path, *, write_results: bool) -> dict:
    te_no = load_joint("textemb", "no")
    te_out = load_joint("textemb", "out")
    g_no = load_split("gist", "no")
    g_out = load_split("gist", "out")
    f_no = load_joint("fashion", "no")
    f_out = load_joint("fashion", "out")
    s_no = load_split("sift", "no")
    s_out = load_split("sift", "out")
    plot_error_grid(
        [(te_no, "Text Embedding"), (g_no, "GIST"),
         (f_no, "Fashion-MNIST"), (s_no, "SIFT")],
        "Relative error, no outliers",
        figs / "figure3.png",
    )
    plot_error_grid(
        [(te_out, "Text Embedding with outlier"), (g_out, "GIST with outlier"),
         (f_out, "Fashion-MNIST with outlier"), (s_out, "SIFT with outlier")],
        "Relative error, with outliers",
        figs / "figure4.png",
    )
    time_no = collect_time("no")
    time_out = collect_time("out")
    plot_time_bars(time_no, "Time / window update, no outliers", figs / "figure5.png")
    plot_time_bars(time_out, "Time / window update, with outliers", figs / "figure6.png")
    agg = {
        "textemb_no": err_summary(te_no),
        "textemb_out": err_summary(te_out),
        "gist_no": err_summary(g_no),
        "gist_out": err_summary(g_out),
        "fashion_no": err_summary(f_no),
        "fashion_out": err_summary(f_out),
        "sift_no": err_summary(s_no),
        "sift_out": err_summary(s_out),
        "time_no": time_no,
        "time_out": time_out,
    }
    if write_results:
        (OUT / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
        write_table_and_results(agg)
    print("wrote", figs / "figure3.png", "through", figs / "figure6.png")
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", type=Path, default=None,
                    help="Figure directory. Default: results/figures")
    ap.add_argument("--skip-results", action="store_true",
                    help="Write figures only. Leave aggregate.json and RESULTS.md in place.")
    args = ap.parse_args()
    figs = args.figs if args.figs is not None else FIGS
    if not figs.is_absolute():
        figs = ROOT / figs
    render(figs, write_results=not args.skip_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
