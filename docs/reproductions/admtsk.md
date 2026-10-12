# ADMTSK and DombiTSK

**Article**: G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: a high-dimensional
Takagi-Sugeno-Kang fuzzy system based on adaptive Dombi T-norm", *IEEE Transactions on
Fuzzy Systems*, vol. 33, no. 6, 2025.

**Script**: [`examples/reproductions/admtsk_2025.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/admtsk_2025.py)

## Protocol of the Article

Section IV-A of the article. Except for the learning rate and the batch size, which the
script varies as the article does, these settings are the defaults of `ADMTSKClassifier`.

| Point | Article | In the script |
|---|---|---|
| Inputs | scaled to $[0, 1]$ | `MinMaxScaler` fitted on the training part |
| Fuzzy sets | three per feature, centres 0, 0.5 and 1, spreads of 1 | defaults (`mf_init="grid"`, `n_mfs=3`) |
| Membership | composite Gaussian, bounded below by $1/e$ (Eq. 31) | default (`GaussianPiMF`) |
| T-norm | Dombi with $\lambda = \ln D / (\ln(K - \varepsilon) - \ln(1 - \varepsilon))$, $K = 10$ | default (`adaptive=True`, `k=10`) |
| Consequents | start at zero | default |
| Optimizer | Adam, 50 epochs, batches of 10% or 20% of the training samples | default (`epochs=50`); both batch sizes |
| Learning rate | 0.01, 0.001 or 0.0001; the best of the six combinations with the batch size is reported | the three values; the six combinations are printed, and the best |
| Evaluation | ten-fold cross-validation, repeated five times | the same, stratified folds |

## Result

Output of the script, on four threads; the standard deviation is over the 50 runs:

```text
ADMTSK: mean test accuracy in percent of each combination
dataset   learning rate  batch  highFIS    std
Colon              0.01    10%    85.86  12.68
Colon              0.01    20%    84.86  11.49
Colon             0.001    10%    82.86  11.45
Colon             0.001    20%    82.86   9.89
Colon            0.0001    10%    87.57  10.79
Colon            0.0001    20%    86.24  11.22
Leukemia           0.01    10%    97.43   5.49
Leukemia           0.01    20%    97.43   5.49
Leukemia          0.001    10%    97.14   5.71
Leukemia          0.001    20%    96.86   5.92
Leukemia         0.0001    10%    96.86   5.92
Leukemia         0.0001    20%    97.14   5.71

Table V: the best of the six combinations (mean test accuracy in percent)
dataset    article  highFIS
Colon        86.33    87.57
Leukemia     97.29    97.43
```

The best of the six combinations is within 1.3 points of the article on Colon and within
0.2 point on Leukemia.

## What the Best of Six Means

On Colon the six combinations go from 82.9 to 87.6, a spread of almost five points,
which is as much as the differences between the models that the article compares; on
Leukemia they go from 96.9 to 97.4. With 62 and 72 samples, one test sample is 1.6 and
1.4 points of a cross-validated accuracy.

The article chooses the best combination for each dataset and each model on the test
folds, without a validation set, so the published value is an optimistic estimate, and
so is the last table above. The defaults of `ADMTSKClassifier` on these datasets are the
third combination (a learning rate of 0.001 above 1000 features, batches of 10%), which
gives 82.9 on Colon and 97.1 on Leukemia. When you report a result of your own, choose
the combination on a validation part of the training data.

## Other Measurements

One repetition of the ten folds, for the six combinations in the order of the output
above. These runs read the data from files with the samples in another order than on
OpenML, so their folds are not those of the script.

| Dataset | Model | Article | Best | The six values |
|---|---|---|---|---|
| SRBCT (83 samples, 2308 genes) | ADMTSK | 98.81 | 98.89 | 98.9, 98.9, 98.9, 98.9, 98.9, 97.8 |
| Colon | DombiTSK, $\lambda = 1$ | — | 87.38 | 86.0, 79.0, 82.9, 84.0, 87.4, 85.7 |
| SRBCT | DombiTSK, $\lambda = 1$ | — | 98.89 | 98.9, 98.9, 98.9, 98.9, 97.8, 97.8 |

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

## Reproducibility

The script fixes the seeds, uses deterministic algorithms and four threads, and ends by
printing the versions of the libraries and the processor. Another number of threads,
another processor or another version of the numerical libraries can change the result.

## Run It

```bash
python examples/reproductions/admtsk_2025.py
```

The data are downloaded from OpenML on the first run. A run takes several hours, most of
them on Leukemia; remove it from `DATASETS` in the script for a run of about half an hour.
