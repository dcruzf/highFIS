# ADPTSK

**Article**: "An adaptive double-parameter softmin based Takagi-Sugeno-Kang fuzzy system
for high-dimensional data", *Fuzzy Sets and Systems*, 2025,
doi:10.1016/j.fss.2025.109582.

**Script**: [`examples/reproductions/adptsk_2025.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/adptsk_2025.py)

## Protocol of the Article

Section 4.1 of the article. These settings are the defaults of `ADPTSKClassifier`.

| Point | Article | In the script |
|---|---|---|
| Inputs | scaled to $[0, 1]$ | `MinMaxScaler` fitted on the training part |
| Fuzzy sets | three per feature, centres 0, 0.5 and 1, spreads of 1 | defaults (`mf_init="grid"`, `n_mfs=3`) |
| Membership | Gaussian bounded below by $e^{-K}$ (Eq. 41) | default (`GaussianPiMF`); `k` of 0.6, 1.0 (the default) and 2.0 |
| Firing strength | ADP-softmin, $\kappa = 690$, $\xi = 730$ (Eqs. 33 and 34) | default |
| Consequents | start at zero | default |
| Optimizer | Adam on the whole training set below 500 samples, 200 iterations, learning rate 0.001 | defaults |
| Evaluation | ten-fold cross-validation, repeated three times | the same, stratified folds |

## Result

Output of the script, on four threads:

```text
Table 3: ADPTSK (mean test accuracy in percent)
std: over the runs, and over the means of the repetitions
dataset      K  article  highFIS          std
Colon      0.6    83.97    84.44 10.75 / 2.91
Colon      1.0    82.46    80.63 11.51 / 2.63
Colon      2.0    77.30    75.16 13.21 / 1.89
Leukemia   0.6    97.32    96.67  6.04 / 0.67
Leukemia   1.0    97.20    95.24  6.73 / 0.67
Leukemia   2.0    92.50    86.49 11.95 / 1.31
```

The article studies ten values of $K$ and reports each of them; the script runs three.
On Colon the three values are within about two points of the article, and the fall of
the accuracy as $K$ grows is reproduced: 84.4, 80.6 and 75.2 against 84.0, 82.5 and 77.3.
On Leukemia $K = 0.6$ and $K = 1$ are within two points of the article, and the accuracy
falls for $K = 2$ as well.

With 62 and 72 samples, one test sample is 1.6 and 1.4 points of the cross-validated
accuracy, and the standard deviation between the 30 runs is 6 to 13 points. The second
value of the last column is the deviation between the means of the three repetitions.

On SRBCT (83 samples, 2308 genes), with ten folds and $K = 1$, highFIS gives 98.89
against 98.75 in the article.

The article recommends a $K$ of at most one. With a larger $K$ the training of the
antecedents can lower the test accuracy while the training accuracy stays at 100%, so
check a validation set before training longer.

## Differences from the Article

- **Gradient of the adaptive parameters.** highFIS treats $\hat\eta$ and $\hat q$ as
  constants in the backward pass. Letting the gradient through them gives the same
  accuracy.
- **Precision.** The bounds $\kappa$ and $\xi$ are those of double precision. highFIS
  works in single precision and evaluates the softmin in the logarithmic domain, where
  they do not have to hold.
- **Loss.** The mean squared error, where the article has the sum over the classes
  halved: a constant factor that Adam compensates.
- **Datasets.** The article uses fourteen datasets. Colon and Leukemia are the two that
  OpenML provides with the samples and features of the article and without further
  dependencies; SRBCT above uses the copy distributed with the code of the HDFIS
  article.

## Reproducibility

The script fixes the seeds, uses deterministic algorithms and four threads, and ends by
printing the versions of the libraries and the processor. Another number of threads,
another processor or another version of the numerical libraries can change the result.

## Run It

```bash
python examples/reproductions/adptsk_2025.py
```

The data are downloaded from OpenML on the first run. A run takes about an hour, most of
it on Leukemia.
