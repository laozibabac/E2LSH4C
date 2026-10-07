#!/usr/bin/env python3
"""Prepare Text Embedding (paper Table 1) from the authors' 2023 supplemental npy.

Official source (NeurIPS 2023 supplemental zip, no reconstruction):
  supplementary/federalist/federalist_84.npy  → A, 1880 × 300
  supplementary/federalist/federalist_85.npy  → B, 1178 × 300

Sizes match the 2025 vendor comments (text_embedding.cpp): A_size=1880, B_size=1178.
The 2025 OpenReview supplementary does not ship a separate TE dump; 2025 README
points at the 2023 OpenReview note. 84 vs 85 is the pair whose cardinalities
match Table 1 (|A|~1.9k, |B|~1.2k).

The outlier is injected here (the npy files are raw bags, no outlier):
  a_tilde = mean(A) + 0.1 * |A| * (A[0] - mean(A))

Writes:
  data/TextEmbedding/{base,base_with_outlier,query}.f32bin
  data/TextEmbedding/meta.json
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = ROOT / "downloads"
DIM = 300
N_A, N_B = 1880, 1178
DEFAULT_ZIP = (
    DOWNLOADS
    / "NeurIPS-2023-near-linear-time-algorithm-for-the-chamfer-distance-Supplemental-Conference.zip"
)
MEMBER_A = "supplementary/federalist/federalist_84.npy"
MEMBER_B = "supplementary/federalist/federalist_85.npy"


def write_f32bin(path: Path, X: np.ndarray) -> None:
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", n, d))
        X.tofile(f)


def inject_outlier(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    c = A.mean(axis=0)
    a_star = A[0]
    a_tilde = 0.1 * float(A.shape[0]) * (a_star - c) + c
    return np.vstack([A, a_tilde]).astype(np.float32)


def load_npy_member(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    arr = np.load(io.BytesIO(zf.read(name)))
    return np.asarray(arr, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "TextEmbedding")
    args = ap.parse_args()

    if not args.zip.is_file():
        print(f"ERROR: missing official supplemental zip: {args.zip}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(args.zip) as z:
        names = set(z.namelist())
        if MEMBER_A not in names or MEMBER_B not in names:
            print(
                f"ERROR: zip missing {MEMBER_A} or {MEMBER_B}; members={sorted(names)}",
                file=sys.stderr,
            )
            return 1
        A = load_npy_member(z, MEMBER_A)
        B = load_npy_member(z, MEMBER_B)

    if A.ndim != 2 or B.ndim != 2:
        print(f"ERROR: expected 2-d arrays, got A={A.shape} B={B.shape}", file=sys.stderr)
        return 1
    if A.shape != (N_A, DIM) or B.shape != (N_B, DIM):
        print(
            f"ERROR: expected A={N_A}x{DIM} B={N_B}x{DIM}, got A={A.shape} B={B.shape}",
            file=sys.stderr,
        )
        return 1
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        print("ERROR: non-finite values in official npy", file=sys.stderr)
        return 1

    A_out = inject_outlier(A)
    print(f"source={args.zip.name}")
    print(f"A={A.shape} B={B.shape} A_with_outlier={A_out.shape}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_f32bin(out / "base.f32bin", A)
    write_f32bin(out / "base_with_outlier.f32bin", A_out)
    write_f32bin(out / "query.f32bin", B)
    meta = {
        "dataset": "TextEmbedding-federalist",
        "source_zip": str(args.zip),
        "source_A": MEMBER_A,
        "source_B": MEMBER_B,
        "citation": "Kusner et al. ICML 2015; NeurIPS 2023 supplemental npy; paper Table 1 d=300",
        "dim": DIM,
        "nA_raw": int(A.shape[0]),
        "nA_with_outlier": int(A_out.shape[0]),
        "nB": int(B.shape[0]),
        "dtype": "float32",
        "outlier": "a_tilde = 0.1*|A|*(A[0]-mean(A))+mean(A)",
        "paper": {"window": 100, "sample_size": 150, "d": 300},
        "binary": "build/compare_perf_d300",
        "files": {
            "base.f32bin": "base.f32bin",
            "base_with_outlier.f32bin": "base_with_outlier.f32bin",
            "query.f32bin": "query.f32bin",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for name in ("base.f32bin", "base_with_outlier.f32bin", "query.f32bin"):
        p = out / name
        print(f"wrote {p} ({p.stat().st_size} bytes)")
    print("wrote meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
