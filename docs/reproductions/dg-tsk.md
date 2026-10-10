# DG-TSK

**Article**: G. Xue, J. Wang, B. Zhang, B. Yuan and C. Dai, "Double groups of gates based
Takagi-Sugeno-Kang (DG-TSK) fuzzy system for simultaneous feature selection and rule
extraction", *Fuzzy Sets and Systems*, vol. 469, 2023, doi:10.1016/j.fss.2023.108627.

**Script**: [`examples/reproductions/dgtsk_2023.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/dgtsk_2023.py)

The script reproduces two tables: the point-based rule base before any training
(Table 2), and DG-TSK after feature selection and rule extraction (Table 3).

## Protocol of the Article

Section 4 of the article. For Table 3 these settings are the defaults of
`DGTSKClassifier`, except the number of iterations of the gate phase.

| Point | Article | In the script |
|---|---|---|
| Inputs | scaled to $[0, 1]$ | `MinMaxScaler` fitted on the training part |
| Rule base | one rule per training sample, at most 300; one spread for every fuzzy set, the mean over the features of the standard deviations (Eq. 23) | Table 2: built with `GaussianMF` and `TSKClassifierModel`. Table 3: default (`rule_base="pfrb"`, `pfrb_spread="std_mean"`) |
| Consequents of the rule base | the label of the sample of each rule | Table 2: set directly. Table 3: default |
| Gates | M-gates on features and rules, parameters starting at 0.1 | default |
| Thresholds | $\zeta_\lambda = 0.5$, $\zeta_\theta = 0.01$, at least one rule per class | defaults |
| Optimizer | gradient descent on the whole training set | default |
| Learning rate | 0.2 in the comparison of gate functions (Section 4.1); not stated for Table 3 | 0.2 (default) |
| Iterations of the gate phase | 300 in Section 4.1; not stated for Table 3 | `dg_epochs=1000` |
| Iterations of the fine-tuning | not stated | 300 (default) |
| Evaluation | ten-fold cross-validation repeated ten times, mean of the 100 runs | the same, stratified folds |

## Result

Output of the script, on one thread:

```text
Table 2: point-based rule base without training (accuracy in percent)
dataset   article  highFIS
Iris        94.00    95.47
Wine        96.63    96.58
Glass       64.00    66.46

Table 3: DG-TSK (accuracy in percent / selected features / extracted rules)
dataset             article            highFIS  std of accuracy
Iris       96.8 / 2.3 / 3.1   95.8 / 2.6 / 3.0             4.87
Wine       98.3 / 8.0 / 3.0   97.2 / 9.5 / 3.0             3.67
Glass      63.1 / 8.0 / 6.0   62.3 / 8.0 / 6.0             8.86
```

- **Table 2.** The rule base of the article, built without training, is reproduced:
  within 0.1 point on Wine, 1.5 points above on Iris and 2.5 points above on Glass. With
  the same construction Wdbc gives 96.45 against 96.38 in the article.
- **Table 3, accuracy.** Within about one point on the three datasets, a fraction of the
  standard deviation between runs.
- **Table 3, structure.** The numbers of rules are those of the article on the three
  datasets, and the eight features of Glass are reproduced. DG-TSK keeps 2.6 features on
  Iris where the article reports 2.3, and 9.5 on Wine where it reports 8.0.

## Length of the Gate Phase

The article trains the gates for 300 iterations when it compares gate functions on Wine
(its Section 4.1), and does not state the number behind Table 3. The reference code of
the authors uses 300, which is the default of `DGTSKClassifier`. The gates keep opening
as the phase goes on, so its length decides how many features pass the threshold. With
300 iterations too few features are kept on most datasets, and the accuracy falls with
them; with 1000 the table of the article is reproduced.

Ten-fold cross-validation, once (repeated twice for Iris, Wine and Wdbc with 300
iterations); accuracy / selected features / extracted rules:

| Dataset | Article | 300 iterations (default) | 1000 iterations |
|---|---|---|---|
| Iris | 96.8 / 2.3 / 3.1 | 95.0 / 2.0 / 5.8 | 96.0 / 2.6 / 3.1 |
| Pima | 75.9 / 4.5 / 2.2 | 73.8 / 1.1 / 2.4 | 76.6 / 4.2 / 2.0 |
| Glass | 63.1 / 8.0 / 6.0 | 45.8 / 1.2 / 6.0 | 62.2 / 8.0 / 6.0 |
| Wine | 98.3 / 8.0 / 3.0 | 97.2 / 5.0 / 3.0 | 98.3 / 9.7 / 3.0 |
| Heart | 85.0 / 4.0 / 3.4 | 83.3 / 3.0 / 2.1 | 84.1 / 9.2 / 2.9 |
| Vehicle | 70.3 / 12.0 / 4.0 | 61.4 / 6.1 / 4.0 | 67.7 / 11.9 / 4.0 |
| Wdbc | 96.2 / 5.0 / 2.2 | 91.2 / 2.0 / 2.5 | 97.0 / 7.1 / 2.6 |
| Sonar | 77.4 / 12.7 / 2.0 | 72.6 / 3.9 / 2.1 | 72.6 / 14.9 / 2.0 |

With 1000 iterations Heart keeps more features than the article reports and Sonar stays
five points below in accuracy; the other six datasets are within three points, with the
structure of the article.

## Comparison with the Authors' Code

The authors publish an implementation. On the same folds, the two implementations agree
with each other for both lengths of the gate phase:

| Dataset | Gate iterations | Article | Authors' code | highFIS |
|---|---|---|---|---|
| Iris | 300 | 96.8 / 2.3 / 3.1 | 94.3 / 2.0 / 5.8 | 95.0 / 2.0 / 5.8 |
| Wine | 300 | 98.3 / 8.0 / 3.0 | 95.5 / 5.0 / 3.0 | 97.2 / 5.0 / 3.0 |
| Wdbc | 300 | 96.2 / 5.0 / 2.2 | 91.6 / 2.1 / 2.1 | 91.2 / 2.0 / 2.5 |
| Pima | 300 | 75.9 / 4.5 / 2.2 | 74.1 / 1.2 / 2.1 | 73.8 / 1.1 / 2.4 |
| Vehicle | 300 | 70.3 / 12.0 / 4.0 | 60.4 / 6.0 / 4.0 | 61.4 / 6.1 / 4.0 |
| Glass | 300 | 63.1 / 8.0 / 6.0 | 46.8 / 1.2 / 6.0 | 45.8 / 1.2 / 6.0 |
| Glass | 1000 | 63.1 / 8.0 / 6.0 | 59.9 / 8.0 / 6.0 | 62.2 / 8.0 / 6.0 |

Iris, Wine and Wdbc use ten folds repeated twice, the other rows ten folds. On Wine the
two implementations select the same features and the same number of rules in every one
of the 20 folds.

## Differences from the Article

- **Iterations of the gate phase.** Not stated for Table 3; chosen here as described
  above. The default of the estimator stays at 300, the value of the reference code,
  which is three times faster.
- **Folds.** The article does not say whether its folds are stratified; the script uses
  stratified folds. Glass has a class of nine samples, fewer than the ten folds, so some
  test parts have no sample of that class.
- **Datasets.** The article uses eighteen datasets. Iris and Wine ship with scikit-learn
  and Glass comes from OpenML (identifier 41); the three run in an hour and a half. The
  larger datasets take hours with 100 runs each.

## Reproducibility

The script fixes the seeds, uses one thread and deterministic algorithms, and ends by
printing the versions of the libraries and the processor. Two runs on the same machine
print the same table. Another processor or another version of the numerical libraries
can change the last digits.

## Run It

```bash
python examples/reproductions/dgtsk_2023.py
```

Glass is downloaded from OpenML on the first run. A run takes about an hour and a half.
