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

Summary of the last model fitted
ADMTSKClassifier
  task               classification, 2 classes
  features           7129 seen in fit, 7129 used by the model
  rules              3 (rule base: coco)
  fuzzy sets         3 per feature, GaussianPiMF
  firing strength    RuleLayer, T-norm AdaptiveDombiTNorm
  defuzzifier        SumBasedDefuzzifier
  consequents        ClassificationConsequentLayer
  parameters         85554 trainable (42774 antecedent, 42780 consequent)
Training
  batch size         13
  epochs             50 epochs, learning rate 0.0001, training loss 0.3134 -> 0.0044, training accuracy 1.0000
```

The best of the six combinations is within 1.3 points of the article on Colon and within
0.2 point on Leukemia.

## The Six Combinations

The article reports, for each dataset, the best of the six combinations, and the last
table of the output does the same. The defaults of `ADMTSKClassifier` on these datasets
are the third combination: a learning rate of 0.001 above 1000 features and batches of
10%. For your own data, choose the combination on a validation part of the training
data.

On SRBCT (83 samples, 2308 genes), with ten folds, the best of the six combinations is
98.89, against 98.81 in the article.

## Membership Functions

The article compares its composite Gaussian set with the traditional Gaussian set and
with the extended one, for several lower bounds (its Table III). With `membership` and
`lower_bound`, on Colon, ten folds, the best of the six combinations:

| Membership function | Lower bound | Article | highFIS |
|---|---|---|---|
| Traditional (`membership="traditional"`) | 0 | 76.62 | 89.05 |
| Extended (`membership="extended"`) | 0.1 | 86.10 | 89.05 |
| | 0.3 | 86.05 | 87.38 |
| | $1/e$ | 84.57 | 87.38 |
| | 0.5 | 85.67 | 87.38 |
| Composite (default) | $1/e$ | 86.33 | 87.38 |

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
