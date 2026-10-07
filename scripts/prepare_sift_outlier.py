#!/usr/bin/env python3
"""Prepare ANN_SIFT1M and the outlier ã = mean(A) + 0.1|A|(A[0] - mean(A)).

Reads either:
  - texmex fvecs: sift_base.fvecs (A) + sift_query.fvecs (B)
  - ann-benchmarks HDF5: train/test datasets

Outlier:
  c = mean(A)
  a* = A[0]
  a_tilde = 0.1 * |A| * (a* - c) + c
  append a_tilde to A  → |A| = 1_000_001

Writes under data/SIFT/ (compact binary preferred):
  base_with_outlier.f32bin  (n,d uint64 little-endian + float32 row-major)
  query.f32bin
  base.fvecs / query.fvecs (optional copies)
  meta.json

Also supports --pilot-A N to write a reduced base for pipeline validation.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_fvecs(path: Path, max_n: int | None = None) -> np.ndarray:
    """Read texmex .fvecs (int32 dim + dim float32 per vector)."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise RuntimeError(f"empty fvecs: {path}")
    d = int(raw[0])
    if d <= 0 or d > 4096:
        raise RuntimeError(f"bad dim {d} in {path}")
    # reinterpret as float32 after stripping dims
    float_view = raw.view(np.float32)
    # each record: 1 int32 + d float32 = (1+d) int32 slots
    cols = 1 + d
    n = raw.size // cols
    if max_n is not None:
        n = min(n, max_n)
    mat = float_view.reshape(-1, cols)[:n, 1:].astype(np.float32, copy=True)
    return mat


def write_fvecs(path: Path, X: np.ndarray) -> None:
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    out = np.empty((n, d + 1), dtype=np.float32)
    out[:, 0] = np.float32(np.int32(d).view(np.float32))  # wrong - need int32 header
    # Proper: write int32 dim then floats per row
    with open(path, "wb") as f:
        dim_i = np.int32(d)
        for i in range(n):
            dim_i.tofile(f)
            X[i].tofile(f)


def write_f32bin(path: Path, X: np.ndarray) -> None:
    """n,d as uint64 LE + float32 row-major."""
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", n, d))
        X.tofile(f)


def inject_outlier(A: np.ndarray) -> np.ndarray:
    """Paper: a_tilde = 0.1*|A|*(a*-c)+c ; append."""
    A = np.asarray(A, dtype=np.float64)
    c = A.mean(axis=0)
    a_star = A[0]
    a_tilde = 0.1 * float(A.shape[0]) * (a_star - c) + c
    A_out = np.vstack([A, a_tilde.astype(np.float64)])
    return A_out.astype(np.float32)


def load_from_hdf5(path: Path, pilot_A: int | None) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
    if pilot_A is not None:
        train = train[:pilot_A]
    return train, test


def load_from_fvecs(base: Path, query: Path, pilot_A: int | None) -> tuple[np.ndarray, np.ndarray]:
    A = read_fvecs(base, max_n=pilot_A)
    B = read_fvecs(query)
    return A, B


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=Path, default=None, help="ann-benchmarks sift-128-euclidean.hdf5")
    ap.add_argument("--base-fvecs", type=Path, default=None)
    ap.add_argument("--query-fvecs", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/SIFT"))
    ap.add_argument("--pilot-A", type=int, default=None, help="use first N of base before outlier")
    ap.add_argument("--also-fvecs", action="store_true", help="also write .fvecs copies")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.hdf5 is not None:
        A, B = load_from_hdf5(args.hdf5, args.pilot_A)
        src = f"hdf5:{args.hdf5}"
    elif args.base_fvecs and args.query_fvecs:
        A, B = load_from_fvecs(args.base_fvecs, args.query_fvecs, args.pilot_A)
        src = f"fvecs:{args.base_fvecs}+{args.query_fvecs}"
    else:
        # defaults
        hdf5 = Path(__file__).resolve().parent.parent / "downloads" / "sift-128-euclidean.hdf5"
        tar_base = Path(__file__).resolve().parent.parent / "downloads" / "sift" / "sift_base.fvecs"
        tar_query = Path(__file__).resolve().parent.parent / "downloads" / "sift" / "sift_query.fvecs"
        if hdf5.is_file():
            A, B = load_from_hdf5(hdf5, args.pilot_A)
            src = f"hdf5:{hdf5}"
        elif tar_base.is_file() and tar_query.is_file():
            A, B = load_from_fvecs(tar_base, tar_query, args.pilot_A)
            src = f"fvecs:{tar_base}+{tar_query}"
        else:
            print("ERROR: provide --hdf5 or --base-fvecs/--query-fvecs", file=sys.stderr)
            return 1

    print(f"source={src} |A_raw|={A.shape} |B|={B.shape}")
    if A.shape[1] != 128 or B.shape[1] != 128:
        print(f"ERROR: expected d=128, got A.d={A.shape[1]} B.d={B.shape[1]}", file=sys.stderr)
        return 1

    A_out = inject_outlier(A)
    print(f"after outlier |A|={A_out.shape[0]} (expected {A.shape[0]+1})")

    base_raw = out / "base.f32bin"
    base_bin = out / "base_with_outlier.f32bin"
    query_bin = out / "query.f32bin"
    write_f32bin(base_raw, A)
    write_f32bin(base_bin, A_out)
    write_f32bin(query_bin, B)
    print(f"wrote {base_raw} ({base_raw.stat().st_size} bytes)")
    print(f"wrote {base_bin} ({base_bin.stat().st_size} bytes)")
    print(f"wrote {query_bin} ({query_bin.stat().st_size} bytes)")

    # Also symlink/copy names expected by compare --data-dir auto-detect
    # base.fvecs / query.fvecs optional
    if args.also_fvecs:
        write_fvecs(out / "base_with_outlier.fvecs", A_out)
        write_fvecs(out / "query.fvecs", B)
        print("wrote fvecs copies")

    meta = {
        "source": src,
        "dim": 128,
        "nA_raw": int(A.shape[0]),
        "nA_with_outlier": int(A_out.shape[0]),
        "nB": int(B.shape[0]),
        "pilot_A": args.pilot_A,
        "outlier": "a_tilde = 0.1*|A|*(A[0]-mean(A))+mean(A)",
        "files": {
            "base.f32bin": str(base_raw.name),
            "base_with_outlier.f32bin": str(base_bin.name),
            "query.f32bin": str(query_bin.name),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
