# E2LSH4C

Supplemental implementation for the experiments in Section 5 of
*E2LSH4C: A Fully Dynamic Algorithm for ℓ₂ Chamfer Distance*.

E2LSH4C maintains Chamfer-distance sampling weights with nested Euclidean
LSH buckets and allocates each query budget across first-collision layers.
The comparison binaries also run the NeurIPS 2025 QuadTree baseline, uniform
sampling, and exact maintenance (brute force). Reported distances use exact
nearest neighbors. One OpenMP thread. Five seeds, 1 through 5.

The drivers were built with MinGW GCC 13.1.0, `-O3`, on 64-bit Windows
(Intel Core i7-11800H, 16 GB RAM). Leave `E2LSH_FAST_MATH` and
`E2LSH_EARLY_EXIT` off.

## Build

CMake 3.20 or newer, Ninja, and GCC or Clang. MSVC is not supported.
On Windows, force the MinGW compilers if `cl.exe` is also on `PATH`:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
cmake --build build --target compare_perf compare_perf_sift \
  compare_perf_gist compare_perf_d300 exact_cache_f32
```

`make` is a wrapper around the same targets. Do not use GnuWin32 make 3.81.

| Binary | Role |
|---|---|
| `compare_perf` | Fashion-MNIST, compile-time dimension 784 |
| `compare_perf_d300` | Text Embedding, dimension 300 |
| `compare_perf_sift` | SIFT, dimension 128 |
| `compare_perf_gist` | GIST, dimension 960 |
| `exact_cache_f32` | Exact Chamfer cache and brute-force update time |

## Layout

`src/compare_main.cpp` is the sliding-window driver. It runs E2LSH4C,
QuadTree, uniform sampling, and exact maintenance, including the setting
where A and B slide together.

`src/e2lsh_chamfer.hpp` is the in-memory E2LSH4C estimator
(`queryStratified`, and `queryImportance` for the sampling ablation).
`src/e2lsh_chamfer_external.hpp` is the same estimator with block storage.
The GIST rows of the sampling ablation use it; the other reported runs stay
in memory. `src/e2lsh_hash.hpp` is the nested E2LSH family.
`src/weighted_sampler.hpp` draws layer and bucket samples.
`src/exact_nn.hpp` and `src/simd_l2.hpp` evaluate exact nearest-neighbor
distances. `src/fast_dynamic_chamfer.hpp` maintains the exact Chamfer sum.
`src/exact_cache_f32.cpp` writes that sum and its update time ahead of a run.
`src/csv_io.hpp` reads the prepared point files.
`src/os/` is file mapping and allocation. `src/storage/` is the block store
behind the external estimator (`block_io_win32.hpp` on Windows,
`block_io_posix.hpp` on Linux).

`vendor/neurips2025_chamfer/` is the QuadTree baseline from Goranci, Jiang,
Kiss, Szilagyi, and Yang, NeurIPS 2025. The top-level CMake build compiles
it into the comparison binaries.

`scripts/prepare_text_embedding.py`, `scripts/prepare_fashion_mnist.py`,
`scripts/prepare_sift_outlier.py`, and `scripts/prepare_gist_outlier.py`
build the `.f32bin` inputs and append the outlier
`mean(A) + 0.1|A|(A[0] − mean(A))`.

`scripts/run_paper_uni_fair.py` runs static A and a sliding window on B.
Text Embedding uses window 100, query every 20 steps, and `T = 150`.
GIST uses window 50, query every 26 steps, `T = 300`, and `n_max = 1100000`.
Fashion-MNIST uses window 500, query every 166 steps, and `T = 200`.
SIFT uses window 500, query every 166 steps, `T = 300`, and `n_max = 1100000`.
Each dataset is run with and without the injected outlier.

`scripts/run_paper_both_dynamic.py` slides A and B in one window, with no
injected outlier. Text Embedding uses window 1500, query every 20 steps, and
`T = 100`. Fashion-MNIST uses window 3500, query every 1166 steps, and
`T = 100`. GIST and SIFT use window 50000, query every 16666 steps,
`T = 500`, and `n_max = 70000`.

`scripts/run_sampling_ablation.py` compares layer-budget allocation with
independent importance sampling on the same E2LSH hierarchy. Text Embedding,
Fashion-MNIST, and SIFT use the static-A windows above and the in-memory
estimator. GIST uses window 50, query every 17 steps, `T = 300`, and the
external estimator. Both clean and outlier copies are included.

`scripts/plot_paper_uni_fair.py` and `scripts/plot_paper_both_dynamic.py`
draw the Section 5 figures. `scripts/export_paper.py` runs both plotters and
writes Tables 3 and 4.

`output/paper_uni/` holds the static-A logs (`compare_cost_log.csv`,
`timing_summary.json`, `run.log`) and the exact caches.
`output/paper_both/` holds the both-dynamic logs.
`output/sampling_ablation/` holds the ablation logs and exact caches.
`results/figures/` holds Figures 3–8. `results/table3.md` and
`results/table4.md` hold Tables 3 and 4.

Regenerate the figures and tables from these logs:

```bash
python scripts/export_paper.py
```

That step needs Python 3, NumPy, and Matplotlib. It does not rerun the
C++ experiments.

## Data

The point sets are not in this tree. Put each prepared dataset under `data/`.

Text Embedding comes from the NeurIPS 2023 supplemental archive of
*A Near-Linear Time Algorithm for the Chamfer Distance*
(<https://openreview.net/forum?id=Mv96iC6TMX>).
`supplementary/federalist/federalist_84.npy` is A (1880 × 300) and
`federalist_85.npy` is B (1178 × 300).

```bash
python scripts/prepare_text_embedding.py --zip /path/to/supplemental.zip
```

Fashion-MNIST is the official IDX image set
(<https://github.com/zalandoresearch/fashion-mnist>):
`train-images-idx3-ubyte` (60,000 images) and `t10k-images-idx3-ubyte`
(10,000 images). Pixels are stored as float32 values in `[0, 255]`,
one 784-dimensional vector per image.

```bash
python scripts/prepare_fashion_mnist.py \
  --train /path/to/train-images-idx3-ubyte \
  --test /path/to/t10k-images-idx3-ubyte
```

SIFT1M and GIST1M are the TEXMEX sets (<http://corpus-texmex.irisa.fr/>),
`sift_base.fvecs` with `sift_query.fvecs`, and `gist_base.fvecs` with
`gist_query.fvecs`. The same vectors are in the ann-benchmarks HDF5 files
`sift-128-euclidean.hdf5` and `gist-960-euclidean.hdf5`
(<http://ann-benchmarks.com/>). Use the full base. GIST in these experiments
uses all 1,000,000 base vectors and all 1,000 queries. Do not pass
`--pilot-A`. The HDF5 preparers need Python 3, NumPy, and h5py.

```bash
python scripts/prepare_sift_outlier.py \
  --hdf5 /path/to/sift-128-euclidean.hdf5 --out-dir data/SIFT
python scripts/prepare_gist_outlier.py \
  --hdf5 /path/to/gist-960-euclidean.hdf5 --out-dir data/GIST_1M
```

Each preparer writes `base.f32bin` or `train.f32bin`, the outlier copy, and
`query.f32bin`. A file is a little-endian `uint64` count, a `uint64`
dimension, then row-major float32 coordinates.
