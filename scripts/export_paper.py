#!/usr/bin/env python3
"""Rebuild Section 5 figures and Tables 3 and 4 from the archived logs."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
UNI = ROOT / "output" / "paper_uni" / "aggregate.json"
BOTH = ROOT / "output" / "paper_both" / "aggregate.json"
ABLATION = ROOT / "output" / "sampling_ablation" / "aggregate.json"

DATASETS = ("Text Embedding", "GIST", "Fashion-MNIST", "SIFT")
ABLATION_ORDER = (
    ("textemb", "Text Embedding"),
    ("fashion", "Fashion-MNIST"),
    ("sift", "SIFT"),
    ("gist", "GIST"),
)
UNI_KEYS = (
    ("textemb_no", "Text Embedding", "no"),
    ("gist_no", "GIST", "no"),
    ("fashion_no", "Fashion-MNIST", "no"),
    ("sift_no", "SIFT", "no"),
    ("textemb_out", "Text Embedding", "out"),
    ("gist_out", "GIST", "out"),
    ("fashion_out", "Fashion-MNIST", "out"),
    ("sift_out", "SIFT", "out"),
)


def run_plot(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, cwd=ROOT)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float, digits: int) -> str:
    return f"{100.0 * x:.{digits}f}"


def sig4(x: float) -> str:
    """Four significant figures, fixed point, as printed in Table 4."""
    if x == 0.0:
        return "0.000"
    decimals = 3 - math.floor(math.log10(abs(x)))
    return f"{x:.{decimals}f}"


def write_table3(uni: dict, both_no: dict) -> None:
    lines = [
        "# Table 3",
        "",
        "Mean absolute relative error (%), averaged over queries and five seeds.",
        "Static A uses a sliding window on B. Both dynamic updates both sets and has no injected outlier.",
        "",
        "| Dataset | Setting | E2LSH4C | QuadTree | Uniform |",
        "|---|---|---:|---:|---:|",
    ]
    by_name = {}
    for key, name, outlier in UNI_KEYS:
        by_name.setdefault(name, {})[outlier] = uni[key]
    for name in DATASETS:
        clean = by_name[name]["no"]
        out = by_name[name]["out"]
        both = both_no[name]["mean_abs_rel"]
        lines.append(
            f"| {name} | Static A, clean | {pct(clean['E2LSH']['mean_abs'], 2)} | "
            f"{pct(clean['QuadTree']['mean_abs'], 2)} | {pct(clean['Uniform']['mean_abs'], 2)} |"
        )
        lines.append(
            f"| {name} | Static A, outlier | {pct(out['E2LSH']['mean_abs'], 2)} | "
            f"{pct(out['QuadTree']['mean_abs'], 2)} | {pct(out['Uniform']['mean_abs'], 2)} |"
        )
        lines.append(
            f"| {name} | Both dynamic | {pct(both['E2LSH'], 2)} | "
            f"{pct(both['QuadTree'], 2)} | {pct(both['Uniform'], 2)} |"
        )
    lines.append("")
    (RESULTS / "table3.md").write_text("\n".join(lines), encoding="utf-8")


def write_table4(cells: list[dict]) -> None:
    by_key = {(c["dataset"], c["outlier"]): c for c in cells}
    lines = [
        "# Table 4",
        "",
        "Sampling ablation on the same snapshots and sample budgets.",
        "Mean absolute relative error is in percent.",
        "Variance is the sample variance of signed relative errors over queries, averaged over five seeds, scaled by 10^-3.",
        "Layer is layer-budget allocation. IS is independent importance sampling.",
        "",
        "| Dataset | Condition | Layer error | IS error | Layer variance | IS variance |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, name in ABLATION_ORDER:
        for outlier, label in (("no", "Clean"), ("out", "Outlier")):
            cell = by_key[(key, outlier)]
            lines.append(
                f"| {name} | {label} | {pct(cell['strat_mean_abs'], 3)} | "
                f"{pct(cell['is_mean_abs'], 3)} | {sig4(cell['strat_var'] * 1e3)} | "
                f"{sig4(cell['is_var'] * 1e3)} |"
            )
    lines.append("")
    (RESULTS / "table4.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    run_plot("plot_paper_uni_fair.py")
    run_plot("plot_paper_both_dynamic.py")
    both = load(BOTH)
    both_no = both["no"] if "no" in both else both
    write_table3(load(UNI), both_no)
    write_table4(load(ABLATION)["cells"])
    print("wrote", RESULTS / "table3.md")
    print("wrote", RESULTS / "table4.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
