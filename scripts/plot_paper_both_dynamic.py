#!/usr/bin/env python3
"""Both A and B dynamic, four datasets, no injected outlier.

Writes results/figures/figure7.png and figure8.png.
Ours is native l2 E2LSH4C. QuadTree is the NeurIPS 2025 baseline.
The exact baseline is labeled Brute Force.
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
OUT = ROOT / "output" / "paper_both"
FIGS = ROOT / "results" / "figures"
SEEDS = [1, 2, 3, 4, 5]

DATASETS = [
    ("textemb", "Text Embedding"),
    ("gist", "GIST"),
    ("fashion", "Fashion-MNIST"),
    ("sift", "SIFT"),
]

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
    return json.loads(path.read_text(encoding="utf-8"))


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


def seed_dir(folder: str, seed: int, tag: str) -> Path:
    return OUT / folder / f"{tag}_seed{seed}"


def complete_seeds(folder: str, tag: str) -> list[int]:
    found = []
    for seed in SEEDS:
        d = seed_dir(folder, seed, tag)
        if (d / "compare_cost_log.csv").is_file() and (d / "timing_summary.json").is_file():
            found.append(seed)
    return found


def load_series(folder: str, tag: str) -> list[dict[str, list[tuple[int, float]]]]:
    seeds = complete_seeds(folder, tag)
    if not seeds:
        raise SystemExit(f"missing {tag} logs for {folder}")
    rows = []
    for seed in seeds:
        path = seed_dir(folder, seed, tag) / "compare_cost_log.csv"
        rows.append(read_csv_err(path))
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
    uni_max = None
    for method in ERROR_METHODS:
        b = band(series, method)
        if b is None:
            continue
        mu = b[1]
        p95 = float(np.percentile(mu, 95))
        if method == "Uniform":
            uni_p95 = p95
            uni_max = float(np.max(mu))
        else:
            focus.append(p95)
    ytop = max(0.08, (max(focus) if focus else 0.04) * 2.0)
    if uni_p95 is not None and uni_p95 < 0.5:
        ytop = max(ytop, uni_p95 * 1.35)
    elif uni_max is not None:
        # The pinned outlier makes the Uniform mean itself large. A 0.16 cap
        # draws that mean as a flat line on the frame.
        ytop = max(ytop, uni_max * 1.08)
    return ytop


def _draw_error_ax(ax, series, name: str, ytop: float) -> None:
    ax.set_title(name)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Relative Error")
    if not series:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        return
    for method in ERROR_METHODS:
        b = band(series, method)
        if b is None:
            continue
        steps, mu, lo, hi = b
        ls, mk = LINESTYLE[method]
        ax.plot(steps, mu, color=COLORS[method], label=LEGEND[method], lw=1.4,
                ls=ls, marker=mk, markersize=3.5, markevery=max(len(steps) // 16, 1))
        ax.fill_between(steps, lo, hi, color=COLORS[method], alpha=0.22, linewidth=0)
    ax.set_ylim(0.0, ytop)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=True, fontsize=8, loc="upper right")


def plot_error(panels, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.9), sharey=False)
    # Each panel sizes its axis from its own mean curve. Forcing Text Embedding
    # onto Fashion-MNIST's limit clips the Uniform mean.
    ylims = [_ylim_for_series(series) for series, _name in panels]
    print(out_path.name, "ylims", [(name, round(y, 4)) for (_, name), y in zip(panels, ylims)])
    for ax, (series, name), ytop in zip(axes.ravel(), panels, ylims):
        _draw_error_ax(ax, series, name, ytop)
    fig.suptitle(title, y=1.01, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def mean_per_update(folder: str, tag: str) -> dict[str, float]:
    acc = {m: [] for m in ("E2LSH", "QuadTree", "Uniform", "Benchmark")}
    seeds = complete_seeds(folder, tag)
    if not seeds:
        raise SystemExit(f"missing {tag} timings for {folder}")
    for seed in seeds:
        path = seed_dir(folder, seed, tag) / "timing_summary.json"
        timing = load_json(path)
        if timing.get("omp_threads") != 1:
            raise SystemExit(f"{path} omp_threads={timing.get('omp_threads')}")
        if (timing.get("nn_microbench") or {}).get("status") != "OK":
            raise SystemExit(f"{path} nn microbench not OK")
        if bool(timing.get("no_outlier")) != (tag == "no"):
            raise SystemExit(f"{path} no_outlier={timing.get('no_outlier')} tag={tag}")
        per = timing.get("per_update_ms") or {}
        for method in acc:
            if method not in per:
                raise SystemExit(f"{path} missing per_update_ms.{method}")
            acc[method].append(float(per[method]))
    return {method: statistics.mean(vals) for method, vals in acc.items()}


def plot_time(rows: dict[str, dict[str, float]], title: str, out_path: Path) -> None:
    methods = ["Uniform", "Benchmark", "QuadTree", "E2LSH"]
    legend_order = ["E2LSH", "QuadTree", "Benchmark", "Uniform"]
    datasets = [name for _folder, name in DATASETS if name in rows]
    # barh draws the first dataset at the bottom. Reverse so Text Embedding is on top.
    datasets = list(reversed(datasets))
    fig, ax = plt.subplots(figsize=(9.6, 2.2 + 1.05 * len(datasets)))
    y = np.arange(len(datasets))
    height = 0.16
    n_m = len(methods)
    handles = {}
    for i, method in enumerate(methods):
        ys = [rows[d][method] for d in datasets]
        offs = (i - (n_m - 1) / 2) * height
        bars = ax.barh(y + offs, ys, height, label=LEGEND[method], color=COLORS[method])
        handles[method] = bars
        for bar, val in zip(bars, ys):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.4f}",
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


def err_summary(series) -> dict[str, float]:
    out = {}
    for method in ERROR_METHODS:
        means = []
        for one in series:
            if method not in one or not one[method]:
                continue
            ys = [y for _s, y in one[method]]
            means.append(sum(ys) / len(ys))
        if means:
            out[method] = statistics.mean(means)
    return out


def collect(tag: str) -> tuple[list, dict[str, dict[str, float]], dict]:
    panels = []
    time_rows = {}
    summary = {}
    for folder, name in DATASETS:
        series = load_series(folder, tag)
        panels.append((series, name))
        time_rows[name] = mean_per_update(folder, tag)
        summary[name] = {
            "mean_abs_rel": err_summary(series),
            "per_update_ms": time_rows[name],
        }
    return panels, time_rows, summary


def _result_lines(summary: dict) -> list[str]:
    lines = []
    for name, row in summary.items():
        err = row["mean_abs_rel"]
        tm = row["per_update_ms"]
        lines.append(
            f"- **{name}**: E2LSH4C {err.get('E2LSH', float('nan')):.4f}; "
            f"QuadTree {err.get('QuadTree', float('nan')):.4f}; "
            f"Uniform {err.get('Uniform', float('nan')):.4f}"
        )
        lines.append(
            f"  - ms/update: E2LSH4C {tm['E2LSH']:.4f}; QuadTree {tm['QuadTree']:.4f}; "
            f"Uniform {tm['Uniform']:.4f}; Brute Force {tm['Benchmark']:.4f}"
        )
    return lines


def _print_summary(tag: str, summary: dict) -> None:
    for name, row in summary.items():
        err = row["mean_abs_rel"]
        tm = row["per_update_ms"]
        print(f"{tag} {name} |rel| E2LSH4C={err.get('E2LSH', float('nan')):.4f} "
              f"QuadTree={err.get('QuadTree', float('nan')):.4f} "
              f"Uniform={err.get('Uniform', float('nan')):.4f}")
        print(f"  ms/update E2LSH4C={tm['E2LSH']:.4f} QuadTree={tm['QuadTree']:.4f} "
              f"Uniform={tm['Uniform']:.4f} Brute Force={tm['Benchmark']:.4f}")


def render(figs: Path, *, write_results: bool) -> dict:
    panels, time_rows, summary_no = collect("no")
    plot_error(panels, "Relative error, A and B dynamic", figs / "figure7.png")
    plot_time(time_rows, "Time / window update, A and B dynamic", figs / "figure8.png")
    summary: dict = {"no": summary_no}
    wrote = [figs / "figure7.png", figs / "figure8.png"]
    if write_results:
        (OUT / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# Both-dynamic results",
            "",
            "Both A and B slide in one window.",
            "Ours is native ℓ₂ E2LSH4C. QuadTree is the NeurIPS 2025 baseline.",
            "The exact baseline is Brute Force.",
            "Seeds 1–5, one OpenMP thread, shared `simd_l2::min_l2_flat`.",
            "The four plotted datasets are Text Embedding, GIST, Fashion-MNIST, and SIFT.",
            "",
            "## Protocol",
            "",
            "- Text Embedding: window 1500, T=100, query every 20. Same gap as the unidirectional run.",
            "- GIST: window 50000, T=500, query every 16666, n_max=70000. Same large-window schedule as SIFT.",
            "- Fashion-MNIST: window 3500, T=100, query every 1166.",
            "- SIFT: window 50000, T=500, query every 16666, n_max=70000.",
            "",
            "## No outliers",
            "",
            *_result_lines(summary_no),
            "",
            "Figures: `results/figures/figure7.png`, `figure8.png`.",
            "",
        ]
        (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in wrote:
        print("wrote", path)
    _print_summary("no", summary_no)
    return summary


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
