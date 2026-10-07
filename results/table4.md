# Table 4

Sampling ablation on the same snapshots and sample budgets.
Mean absolute relative error is in percent.
Variance is the sample variance of signed relative errors over queries, averaged over five seeds, scaled by 10^-3.
Layer is layer-budget allocation. IS is independent importance sampling.

| Dataset | Condition | Layer error | IS error | Layer variance | IS variance |
|---|---|---:|---:|---:|---:|
| Text Embedding | Clean | 1.460 | 4.105 | 0.3245 | 2.714 |
| Text Embedding | Outlier | 1.852 | 6.086 | 0.4879 | 5.404 |
| Fashion-MNIST | Clean | 1.598 | 3.145 | 0.4034 | 1.464 |
| Fashion-MNIST | Outlier | 1.736 | 5.092 | 0.5110 | 4.656 |
| SIFT | Clean | 0.899 | 2.851 | 0.1284 | 1.248 |
| SIFT | Outlier | 0.975 | 3.233 | 0.1388 | 1.840 |
| GIST | Clean | 1.369 | 3.546 | 0.2781 | 2.289 |
| GIST | Outlier | 1.447 | 4.412 | 0.3155 | 3.739 |
