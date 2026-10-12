# AdaTSK

**Article**: G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An adaptive neuro-fuzzy
system with integrated feature selection and rule extraction for high-dimensional
classification problems", *IEEE Transactions on Fuzzy Systems*, vol. 31, no. 7,
pp. 2167-2181, 2023.

**Script**: [`examples/reproductions/adatsk_2022.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/adatsk_2022.py)

The same article proposes FSRE-AdaTSK, the version with feature selection and rule
extraction, which has its own [reproduction](fsre-adatsk.md).

## Protocol of the Article

Section IV of the article, for the classifier without feature selection (its Table III).

| Point | Article | In the script |
|---|---|---|
| Fuzzy sets | three per feature, centres evenly placed between the minimum and the maximum of the training part | defaults (`mf_init="grid"`, `n_mfs=3`) |
| Membership | $e^{-(x - m)^2}$: no spread to train (Eq. 3) | default (`ADATSKGaussianMF`, spreads fixed at 1) |
| Firing strength | Ada-softmin, index $\lceil 690 / \ln \min_d \mu \rceil$, not below $-1000$ (Eqs. 15 and 16) | default |
| Consequents | start at zero | default |
| Loss | squared error summed over the classes and halved (Eq. 8) | default |
| Optimizer | gradient descent on the whole training set | default |
| Learning rate and iterations | not stated | `learning_rate=0.05`, `epochs=1000` |
| Normalization of the consequent inputs | none | `consequent_batch_norm=False` |
| Inputs | not stated | scaled to $[0, 1]$ with the range of the training part |
| Evaluation | ten-fold cross-validation, repeated five times | the same, stratified folds |

## Result

Output of the script, on one thread:

```text
Table III: AdaTSK (mean test accuracy in percent)
std: over the runs, and over the means of the repetitions
dataset   article  highFIS          std
Iris         95.5    94.67  5.33 / 0.42
Wine         98.7    96.52  4.18 / 0.67
Wdbc         94.3    94.13  3.11 / 0.14
```

Iris and Wdbc are reproduced within one point. Wine is 2.2 points below, with a learning
rate and a number of iterations that the article does not state: with a learning rate of
0.1 and 3000 iterations highFIS gives 97.33, 98.33 and 95.96 on the three datasets (ten
folds).

## High-Dimensional Datasets

Table III of the article also reports AdaTSK on seven datasets with more than 1000
features, with the centres kept fixed: 71.9 on Colon, 87.5 on SRBCT and 80.0 on
Leukemia. These are not reproduced: highFIS gives 87.4 on Colon with a learning rate of
0.001 and 1000 iterations, and 87.3 with the defaults of the estimator. The learning
rate and the number of iterations of the article are unknown, and plain gradient
descent on 2000 features diverges with the 0.05 used above. Later articles of the same
group report other values for AdaTSK on the same data: 84.3 and 74.4 on Colon, 98.8 on
SRBCT, and 95.0 and 85.6 on Leukemia.

## Differences from the Article

- **Learning rate and iterations.** Not stated in the article; chosen here.
- **Defaults of the estimator.** `ADATSKClassifier` defaults to a learning rate that
  falls with the number of features (`learning_rate="auto"`), 300 epochs and
  normalization of the consequent inputs, a setting that is stable in low and in high
  dimension. With the defaults the three datasets give 94.7, 99.4 and 96.8 with seed 0.
- **Datasets.** The article uses nineteen datasets. Iris, Wine and Wdbc ship with
  scikit-learn.

## Reproducibility

The script fixes the seeds, uses deterministic algorithms and one thread, and ends by
printing the versions of the libraries and the processor. Another processor or another
version of the numerical libraries can change the last digits.

## Run It

```bash
python examples/reproductions/adatsk_2022.py
```

No download is needed. A run takes about five minutes.
