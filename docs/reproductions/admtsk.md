# ADMTSK and DombiTSK

**Article**: G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: a high-dimensional
Takagi-Sugeno-Kang fuzzy system based on adaptive Dombi T-norm", *IEEE Transactions on
Fuzzy Systems*, vol. 33, no. 6, 2025.

**Script**: [`examples/reproductions/admtsk_2025.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/admtsk_2025.py)

## Protocol of the Article

Section IV-A of the article. These settings are the defaults of `ADMTSKClassifier`.

| Point | Article | In the script |
|---|---|---|
| Inputs | scaled to $[0, 1]$ | `MinMaxScaler` fitted on the training part |
| Fuzzy sets | three per feature, centres 0, 0.5 and 1, spreads of 1 | defaults (`mf_init="grid"`, `n_mfs=3`) |
| Membership | composite Gaussian, bounded below by $1/e$ (Eq. 31) | default (`GaussianPiMF`) |
| T-norm | Dombi with $\lambda = \ln D / (\ln(K - \varepsilon) - \ln(1 - \varepsilon))$, $K = 10$ | default (`adaptive=True`, `k=10`) |
| Consequents | start at zero | default |
| Optimizer | Adam, 50 epochs, batches of 10% or 20% of the training samples | defaults (`epochs=50`, 10%) |
| Learning rate | 0.01, 0.001 or 0.0001; the best of the six combinations with the batch size is reported | 0.01 |
| Evaluation | ten-fold cross-validation, repeated five times | ten-fold cross-validation, once |

## Result

Mean test accuracy in percent; for highFIS, with the standard deviation over the ten
folds.

| Dataset | Article | highFIS |
|---|---|---|
| Colon (62 samples, 2000 genes) | 86.33 | 85.95 ± 14.18 |
| Leukemia (72 samples, 7129 genes) | 97.29 | 97.14 ± 5.71 |

Both are reproduced within half a point, with seed 0 and four threads. The seed was not
chosen for this family.

## The Six Combinations of the Article

The article reports, for each dataset, the best of six combinations of learning rate and
batch size. With highFIS, ten folds, accuracy for the learning rates 0.01, 0.001 and
0.0001, each with batches of 10% and 20%:

| Dataset | Model | Article | Best | The six values |
|---|---|---|---|---|
| Colon | ADMTSK | 86.33 | 87.38 | 86.0, 79.0, 84.0, 80.7, 87.4, 85.7 |
| SRBCT (83 samples, 2308 genes) | ADMTSK | 98.81 | 98.89 | 98.9, 98.9, 98.9, 98.9, 98.9, 97.8 |
| Leukemia | ADMTSK | 97.29 | 97.14 | 97.1, 97.1, 95.7, 95.7, 95.7, 97.1 |
| Colon | DombiTSK, $\lambda = 1$ | — | 87.38 | 86.0, 79.0, 82.9, 84.0, 87.4, 85.7 |
| SRBCT | DombiTSK, $\lambda = 1$ | — | 98.89 | 98.9, 98.9, 98.9, 98.9, 97.8, 97.8 |

These runs read Colon from a file with the samples in another order than on OpenML, so
their folds are not the folds of the script, and the first value of Colon (86.0) is not
the 85.95 of the table above. On Colon the choice among the six moves the accuracy by eight points, more than the
differences between the models that the article compares; with 62 samples one test
sample is 1.6 points of the cross-validated accuracy. The best of six chosen on the
test folds is an optimistic estimate.

## Differences from the Article

- **Loss.** The article minimizes the squared error on one-hot targets summed over the
  classes and halved (Eq. 9); highFIS uses the mean squared error, which differs by a
  constant factor that Adam compensates.
- **DombiTSK.** In the article DombiTSK is the same system with a fixed $\lambda$ and
  is evaluated with the same partition. `DombiTSKClassifier` defaults to clustering
  with five rules and 100 epochs, a choice for general use; pass
  `mf_init="grid", n_mfs=3, rule_base="coco", epochs=50` for the setting of the article.
- **Datasets.** The article uses sixteen datasets. Colon and Leukemia are the two that
  OpenML provides with the samples and features of the article and without further
  dependencies; SRBCT above uses the copy distributed with the code of the HDFIS
  article.

## Run It

```bash
python examples/reproductions/admtsk_2025.py
python examples/reproductions/admtsk_2025.py --learning-rate 0.0001
```

The data are downloaded from OpenML on the first run. A run takes about nine minutes,
most of them on Leukemia.
