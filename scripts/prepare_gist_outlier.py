#!/usr/bin/env python3
"""Prepare ANN_GIST1M and the outlier ã = mean(A) + 0.1|A|(A[0] - mean(A)).

Reads either:
  - texmex fvecs: gist_base.fvecs (A) + gist_query.fvecs (B)
  - ann-benchmarks HDF5: train/test (gist-960-euclidean.hdf5)

Outlier:
  c = mean(A)
  a* = A[0]
  a_tilde = 0.1 * |A| * (a* - c) + c
  append a_tilde to A

Writes compact binary:
  base_with_outlier.f32bin  (n,d uint64 little-endian + float32 row-major)
  query.f32bin
  meta.json

--pilot-A N uses the first N of base before outlier (HDF5 hyperslab, not a full 1M load).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

GIST_DIM = 960


def read_fvecs(path: Path, max_n: int | None = None) -> np.ndarray:
    """Read texmex .fvecs (int32 dim + dim float32 per vector)."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise RuntimeError(f"empty fvecs: {path}")
    d = int(raw[0])
    if d <= 0 or d > 4096:
        raise RuntimeError(f"bad dim {d} in {path}")
    float_view = raw.view(np.float32)
    cols = 1 + d
    n = raw.size // cols
    if max_n is not None:
        n = min(n, max_n)
    mat = float_view.reshape(-1, cols)[:n, 1:].astype(np.float32, copy=True)
    return mat


def write_fvecs(path: Path, X: np.ndarray) -> None:
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
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
        ds = f["train"]
        n = int(ds.shape[0]) if pilot_A is None else min(int(pilot_A), int(ds.shape[0]))
        # Hyperslab: do not materialize the full 1M×960 train set.
        train = np.asarray(ds[:n], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
    return train, test


def write_hdf5_streaming(hdf5: Path, out: Path, pilot_A: int | None) -> dict:
    """Two-pass HDF5 prepare: O(d) RAM. Writes f32bin + meta, returns meta dict."""
    import h5py

    out.mkdir(parents=True, exist_ok=True)
    chunk = 8192
    with h5py.File(hdf5, "r") as f:
        ds = f["train"]
        n = int(ds.shape[0]) if pilot_A is None else min(int(pilot_A), int(ds.shape[0]))
        d = int(ds.shape[1])
        if d != GIST_DIM:
            raise RuntimeError(f"expected d={GIST_DIM}, got {d}")
        a_star = np.asarray(ds[0], dtype=np.float64)
        c = np.zeros(d, dtype=np.float64)
        for i in range(0, n, chunk):
            sl = np.asarray(ds[i : min(i + chunk, n)], dtype=np.float64)
            c += sl.sum(axis=0)
        c /= float(n)
        a_tilde = 0.1 * float(n) * (a_star - c) + c
        n_out = n + 1
        base_bin = out / "base_with_outlier.f32bin"
        with open(base_bin, "wb") as fo:
            fo.write(struct.pack("<QQ", n_out, d))
            for i in range(0, n, chunk):
                sl = np.asarray(ds[i : min(i + chunk, n)], dtype=np.float32)
                sl.tofile(fo)
            np.asarray(a_tilde, dtype=np.float32).tofile(fo)
        test = np.asarray(f["test"], dtype=np.float32)
        query_bin = out / "query.f32bin"
        write_f32bin(query_bin, test)
        try:
            fd = os.open(base_bin, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except Exception:
            pass
    meta = {
        "source": f"hdf5:{hdf5}",
        "dim": GIST_DIM,
        "nA_raw": n,
        "nA_with_outlier": n_out,
        "nB": int(test.shape[0]),
        "pilot_A": pilot_A,
        "outlier": "a_tilde = 0.1*|A|*(A[0]-mean(A))+mean(A)",
        "files": {
            "base_with_outlier.f32bin": str(base_bin.name),
            "query.f32bin": str(query_bin.name),
        },
        "prepare": "streaming_hdf5",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"source=hdf5:{hdf5} |A_raw|={n}x{d} |B|={test.shape}")
    print(f"after outlier |A|={n_out} (expected {n + 1})")
    print(f"wrote {base_bin} ({base_bin.stat().st_size} bytes)")
    print(f"wrote {query_bin} ({query_bin.stat().st_size} bytes)")
    print("wrote meta.json")
    return meta


def load_from_fvecs(base: Path, query: Path, pilot_A: int | None) -> tuple[np.ndarray, np.ndarray]:
    A = read_fvecs(base, max_n=pilot_A)
    B = read_fvecs(query)
    return A, B


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=Path, default=None, help="ann-benchmarks gist-960-euclidean.hdf5")
    ap.add_argument("--base-fvecs", type=Path, default=None)
    ap.add_argument("--query-fvecs", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/GIST_1M"))
    ap.add_argument("--pilot-A", type=int, default=None, help="use first N of base before outlier")
    ap.add_argument("--also-fvecs", action="store_true", help="also write .fvecs copies")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.hdf5 is not None:
        if not args.also_fvecs:
            write_hdf5_streaming(args.hdf5, out, args.pilot_A)
            return 0
        A, B = load_from_hdf5(args.hdf5, args.pilot_A)
        src = f"hdf5:{args.hdf5}"
    elif args.base_fvecs and args.query_fvecs:
        A, B = load_from_fvecs(args.base_fvecs, args.query_fvecs, args.pilot_A)
        src = f"fvecs:{args.base_fvecs}+{args.query_fvecs}"
    else:
        hdf5 = Path(__file__).resolve().parent.parent / "downloads" / "gist-960-euclidean.hdf5"
        tar_base = Path(__file__).resolve().parent.parent / "downloads" / "gist" / "gist_base.fvecs"
        tar_query = Path(__file__).resolve().parent.parent / "downloads" / "gist" / "gist_query.fvecs"
        if hdf5.is_file():
            if not args.also_fvecs:
                write_hdf5_streaming(hdf5, out, args.pilot_A)
                return 0
            A, B = load_from_hdf5(hdf5, args.pilot_A)
            src = f"hdf5:{hdf5}"
        elif tar_base.is_file() and tar_query.is_file():
            A, B = load_from_fvecs(tar_base, tar_query, args.pilot_A)
            src = f"fvecs:{tar_base}+{tar_query}"
        else:
            print("ERROR: provide --hdf5 or --base-fvecs/--query-fvecs", file=sys.stderr)
            return 1

    print(f"source={src} |A_raw|={A.shape} |B|={B.shape}")
    if A.shape[1] != GIST_DIM or B.shape[1] != GIST_DIM:
        print(
            f"ERROR: expected d={GIST_DIM}, got A.d={A.shape[1]} B.d={B.shape[1]}",
            file=sys.stderr,
        )
        return 1

    A_out = inject_outlier(A)
    print(f"after outlier |A|={A_out.shape[0]} (expected {A.shape[0]+1})")

    base_bin = out / "base_with_outlier.f32bin"
    query_bin = out / "query.f32bin"
    write_f32bin(base_bin, A_out)
    write_f32bin(query_bin, B)
    print(f"wrote {base_bin} ({base_bin.stat().st_size} bytes)")
    print(f"wrote {query_bin} ({query_bin.stat().st_size} bytes)")

    if args.also_fvecs:
        write_fvecs(out / "base_with_outlier.fvecs", A_out)
        write_fvecs(out / "query.fvecs", B)
        print("wrote fvecs copies")

    meta = {
        "source": src,
        "dim": GIST_DIM,
        "nA_raw": int(A.shape[0]),
        "nA_with_outlier": int(A_out.shape[0]),
        "nB": int(B.shape[0]),
        "pilot_A": args.pilot_A,
        "outlier": "a_tilde = 0.1*|A|*(A[0]-mean(A))+mean(A)",
        "files": {
            "base_with_outlier.f32bin": str(base_bin.name),
            "query.f32bin": str(query_bin.name),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
