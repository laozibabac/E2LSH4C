#!/usr/bin/env python3
"""Prepare Fashion-MNIST and the outlier ã = mean(A) + 0.1|A|(A[0] - mean(A)).

Reads the official idx3-ubyte image files (no labels):
  downloads/train-images-idx3-ubyte  (60_000 × 28 × 28)
  downloads/t10k-images-idx3-ubyte   (10_000 × 28 × 28)

Pixels are kept as float32 in [0, 255] (784-d), matching the paper CSV convention.

Writes under data/Fashion-MNIST/:
  train.f32bin                 (n,d uint64 LE + float32 row-major)
  train_with_outlier.f32bin
  query.f32bin                 (test set)
  meta.json
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIM = 784


def read_idx3_ubyte(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
    if magic != 2051:
        raise RuntimeError(f"bad idx3 magic {magic} in {path} (expected 2051)")
    if rows * cols != DIM:
        raise RuntimeError(f"expected {DIM}-d images, got {rows}x{cols} in {path}")
    raw = np.fromfile(path, dtype=np.uint8, offset=16)
    if raw.size != n * DIM:
        raise RuntimeError(f"short idx3 body in {path}: {raw.size} bytes, need {n * DIM}")
    return raw.reshape(n, DIM).astype(np.float32, copy=False)


def write_f32bin(path: Path, X: np.ndarray) -> None:
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    with path.open("wb") as f:
        f.write(struct.pack("<QQ", n, d))
        X.tofile(f)


def inject_outlier(A: np.ndarray) -> np.ndarray:
    A64 = np.asarray(A, dtype=np.float64)
    c = A64.mean(axis=0)
    a_star = A64[0]
    a_tilde = 0.1 * float(A64.shape[0]) * (a_star - c) + c
    return np.vstack([A64, a_tilde]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=ROOT / "downloads" / "train-images-idx3-ubyte")
    ap.add_argument("--test", type=Path, default=ROOT / "downloads" / "t10k-images-idx3-ubyte")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "Fashion-MNIST")
    args = ap.parse_args()

    if not args.train.is_file() or not args.test.is_file():
        print(f"ERROR: missing idx3 files\n  train={args.train}\n  test={args.test}")
        return 1

    train = read_idx3_ubyte(args.train)
    test = read_idx3_ubyte(args.test)
    print(f"train={train.shape} test={test.shape} dtype={train.dtype}")
    A_out = inject_outlier(train)
    print(f"after outlier |A|={A_out.shape[0]} (expected {train.shape[0] + 1})")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_f32bin(out / "train.f32bin", train)
    write_f32bin(out / "train_with_outlier.f32bin", A_out)
    write_f32bin(out / "query.f32bin", test)
    meta = {
        "source": f"idx3:{args.train.name}+{args.test.name}",
        "dim": DIM,
        "nA_raw": int(train.shape[0]),
        "nA_with_outlier": int(A_out.shape[0]),
        "nB": int(test.shape[0]),
        "pixel_range": [0.0, 255.0],
        "outlier": "a_tilde = 0.1*|A|*(A[0]-mean(A))+mean(A)",
        "files": {
            "train.f32bin": "train.f32bin",
            "train_with_outlier.f32bin": "train_with_outlier.f32bin",
            "query.f32bin": "query.f32bin",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for name in ("train.f32bin", "train_with_outlier.f32bin", "query.f32bin"):
        p = out / name
        print(f"wrote {p} ({p.stat().st_size} bytes)")
    print("wrote meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
