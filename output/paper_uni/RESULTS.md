# Unidirectional results

Static A, sliding-window B. Ours is native ℓ₂ E2LSH4C. QuadTree is the NeurIPS 2025 baseline.
The exact baseline is Brute Force.
One OpenMP thread. QuadTree and Uniform use `simd_l2::min_l2_flat`.
The four plotted datasets are Text Embedding, GIST, Fashion-MNIST, and SIFT.

## Protocol

- Text Embedding: window 100, query every 20, T=150.
- GIST: window 50, query every 26, T=300, `|B|=1000` (37 queries).
- Fashion-MNIST: window 500, query every 166, T=200.
- SIFT: window 500, query every 166, T=300.
- 5 seeds. Outlier is `ã = 0.1|A|(A[0]-mean(A))+mean(A)`.
- Brute Force is exact maintenance, one OpenMP thread.

## Table 1

| dataset | d | |A| | |B| | window | T |
|---|---|---|---|---|---|
| Text Embedding | 300 | 1880 / 1881 | 1178 | 100 | 150 |
| GIST | 960 | 1000000 / 1000001 | 1000 | 50 | 300 |
| Fashion-MNIST | 784 | 60000 / 60001 | 10000 | 500 | 200 |
| SIFT | 128 | 1000000 / 1000001 | 10000 | 500 | 300 |

## 5-seed mean |relative error|

- **Text Embedding, no outlier**: Ours (E2LSH4C) 0.0146; QuadTree 0.0265; Uniform 0.0257
- **Text Embedding, with outlier**: Ours (E2LSH4C) 0.0185; QuadTree 0.0456; Uniform 0.2175
- **GIST, no outlier**: Ours (E2LSH4C) 0.0131; QuadTree 0.0205; Uniform 0.0127
- **GIST, with outlier**: Ours (E2LSH4C) 0.0146; QuadTree 0.0466; Uniform 0.1082
- **Fashion-MNIST, no outlier**: Ours (E2LSH4C) 0.0160; QuadTree 0.0234; Uniform 0.0142
- **Fashion-MNIST, with outlier**: Ours (E2LSH4C) 0.0174; QuadTree 0.0348; Uniform 0.3296
- **SIFT, no outlier**: Ours (E2LSH4C) 0.0090; QuadTree 0.0204; Uniform 0.0074
- **SIFT, with outlier**: Ours (E2LSH4C) 0.0097; QuadTree 0.0227; Uniform 0.1085

## Time per update (ms)

### no outliers

- **Text Embedding**: E2LSH4C 0.0384; QuadTree 0.0967; Uniform 0.0236; Brute Force 0.1401
- **GIST**: E2LSH4C 0.3499; QuadTree 0.5261; Uniform 0.0597; Brute Force 330.8713
- **Fashion-MNIST**: E2LSH4C 0.1218; QuadTree 0.3568; Uniform 0.0700; Brute Force 16.0696
- **SIFT**: E2LSH4C 0.1163; QuadTree 0.1780; Uniform 0.0439; Brute Force 43.9616

### with outliers

- **Text Embedding**: E2LSH4C 0.0261; QuadTree 0.1076; Uniform 0.0233; Brute Force 0.1356
- **GIST**: E2LSH4C 0.3554; QuadTree 0.8401; Uniform 0.0598; Brute Force 332.0188
- **Fashion-MNIST**: E2LSH4C 0.0935; QuadTree 0.4114; Uniform 0.0656; Brute Force 15.7975
- **SIFT**: E2LSH4C 0.1090; QuadTree 0.1878; Uniform 0.0594; Brute Force 43.9632

Figures: `results/figures/figure3.png`–`figure6.png`.
