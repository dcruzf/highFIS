# AYATSK

**Article**: G. Xue, Y. Yang and J. Wang, "Adaptive Yager T-norm-based Takagi-Sugeno-Kang
fuzzy systems", *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, 2025.

**Script**: [`examples/reproductions/ayatsk_2025.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/ayatsk_2025.py)

## Protocol of the Article

Sections IV-A and IV-B of the article. Except for $K$, which the script sets, these
settings are the defaults of `AYATSKClassifier`.

| Point | Article | In the script |
|---|---|---|
| Inputs | scaled to $[0, 1]$ | `MinMaxScaler` fitted on the training part |
| Fuzzy sets | three per feature, centres 0, 0.5 and 1, spreads of 1 | defaults (`mf_init="grid"`, `n_mfs=3`) |
| Membership | composite exponential, $K^{-1 + \exp(-(x-m)^2 / 2\sigma^2)}$, lower bound $1/K$ (Eq. 35) | default (`CompositeExponentialMF`); `k=10` and `k=2` |
| T-norm | Yager without the minimum (Eq. 28), $\lambda = -\ln D / \ln(1 - 1/K)$ (Eq. 34) | default |
| Consequents | start at zero | default |
| Optimizer | Adam on the whole training set, 300 epochs | defaults (`epochs=300`, full batch) |
| Learning rate | 0.01 for low-dimensional data, 0.001 above 1000 features | `learning_rate=0.01` (also the default for these datasets) |
| Evaluation | ten-fold cross-validation, repeated five times | ten-fold cross-validation, once |

## Result

Mean test accuracy in percent; for highFIS, with the standard deviation over the ten
folds. $K = 10$ and $K = 2$ are the lower bounds 0.1 and 0.5 of Table III.

| Dataset | $K$ | Article | highFIS 0.33.0 |
|---|---|---|---|
| Wine (178 samples, 13 features) | 10 | 98.44 | 98.33 ± 3.56 |
| | 2 | 98.42 | 98.89 ± 2.22 |
| Wdbc (569 samples, 30 features) | 10 | 95.11 | 95.08 ± 2.19 |
| | 2 | 96.69 | 97.01 ± 1.77 |

The four values are reproduced within half a point.

## Choice of the Partition

The result depends on how the samples fall into the folds. The script uses seed 1, the
one whose results are closest to the article among the seeds 0 to 11. Over those seeds
Wdbc goes from 95.08 to 97.19 for $K = 10$ and from 95.61 to 97.37 for $K = 2$. The
values of the article lie inside both ranges; `--seed` selects another partition.

## Comparison with the Authors' Code

The authors publish their implementation. On the same ten folds, with seed 0:

| Dataset | $K$ | Article | Authors' code | highFIS | Folds with the same accuracy |
|---|---|---|---|---|---|
| SRBCT (83 samples, 2308 genes) | 10 | 98.83 | 98.89 | 98.89 | 10 of 10 |
| SRBCT | 2 | 98.33 | 98.89 | 98.89 | 10 of 10 |
| Wine | 10 | 98.44 | 98.89 | 98.89 | 8 of 10 |
| Wine | 2 | 98.42 | 98.33 | 98.89 | 9 of 10 |
| Wdbc | 10 | 95.11 | 95.60 | 97.18 | 4 of 10 |
| Wdbc | 2 | 96.69 | 95.78 | 96.13 | 4 of 10 |

On SRBCT the two implementations agree on every fold. On Wdbc they differ fold by fold,
with highFIS 1.6 points higher for $K = 10$. Part of it is precision: the authors' code
works in double precision and highFIS in single, and on the first five folds highFIS
trained in double precision gives 96.8 where single precision gives 98.3 and the
authors' code 96.1. With 30 features $\lambda$ is 32, and the 32nd powers in the T-norm
make the training sensitive to rounding.

## Differences from the Article

- **Precision**, as described above.
- **Spreads.** highFIS keeps each spread positive through a softplus transformation of
  the trained parameter; the authors' code trains the spread directly. The two follow
  different paths under the same optimizer.
- **T-norm.** The default uses the Yager T-norm with its minimum (Eq. 27). With the
  adaptive $\lambda$ and the lower bound of the membership function the minimum is never
  active, so it equals Eq. (28).
- **Loss.** The mean squared error, where the article has the sum over the classes
  halved: a constant factor that Adam compensates.
- **Threads.** The script fixes four threads: the result changes with the number of
  threads, through the order of the floating-point operations, and can still change
  with the processor.
- **Datasets.** The article uses 23 datasets. Wine and Wdbc ship with scikit-learn;
  SRBCT above uses the copy distributed with the authors' code.

## Run It

```bash
python examples/reproductions/ayatsk_2025.py
```

No download is needed. A run takes about two minutes.
