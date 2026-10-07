# Table 3

Mean absolute relative error (%), averaged over queries and five seeds.
Static A uses a sliding window on B. Both dynamic updates both sets and has no injected outlier.

| Dataset | Setting | E2LSH4C | QuadTree | Uniform |
|---|---|---:|---:|---:|
| Text Embedding | Static A, clean | 1.46 | 2.65 | 2.57 |
| Text Embedding | Static A, outlier | 1.85 | 4.56 | 21.75 |
| Text Embedding | Both dynamic | 1.74 | 3.71 | 6.49 |
| GIST | Static A, clean | 1.31 | 2.05 | 1.27 |
| GIST | Static A, outlier | 1.46 | 4.66 | 10.82 |
| GIST | Both dynamic | 0.98 | 1.66 | 0.97 |
| Fashion-MNIST | Static A, clean | 1.60 | 2.34 | 1.42 |
| Fashion-MNIST | Static A, outlier | 1.74 | 3.48 | 32.96 |
| Fashion-MNIST | Both dynamic | 2.09 | 3.25 | 2.06 |
| SIFT | Static A, clean | 0.90 | 2.04 | 0.74 |
| SIFT | Static A, outlier | 0.97 | 2.27 | 10.85 |
| SIFT | Both dynamic | 0.69 | 1.67 | 0.61 |
