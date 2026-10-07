# Both-dynamic results

Both A and B slide in one window.
Ours is native ℓ₂ E2LSH4C. QuadTree is the NeurIPS 2025 baseline.
The exact baseline is Brute Force.
Seeds 1–5, one OpenMP thread, shared `simd_l2::min_l2_flat`.
The four plotted datasets are Text Embedding, GIST, Fashion-MNIST, and SIFT.

## Protocol

- Text Embedding: window 1500, T=100, query every 20. Same gap as the unidirectional run.
- GIST: window 50000, T=500, query every 16666, n_max=70000. Same large-window schedule as SIFT.
- Fashion-MNIST: window 3500, T=100, query every 1166.
- SIFT: window 50000, T=500, query every 16666, n_max=70000.

## No outliers

- **Text Embedding**: E2LSH4C 0.0174; QuadTree 0.0371; Uniform 0.0649
  - ms/update: E2LSH4C 0.0696; QuadTree 0.1268; Uniform 0.0576; Brute Force 0.0579
- **GIST**: E2LSH4C 0.0098; QuadTree 0.0166; Uniform 0.0097
  - ms/update: E2LSH4C 0.0287; QuadTree 0.4074; Uniform 0.0001; Brute Force 0.0819
- **Fashion-MNIST**: E2LSH4C 0.0209; QuadTree 0.0325; Uniform 0.0206
  - ms/update: E2LSH4C 0.0552; QuadTree 0.2969; Uniform 0.0051; Brute Force 0.2772
- **SIFT**: E2LSH4C 0.0069; QuadTree 0.0167; Uniform 0.0061
  - ms/update: E2LSH4C 0.0200; QuadTree 0.0707; Uniform 0.0002; Brute Force 0.1044

Figures: `results/figures/figure7.png`, `figure8.png`.

