# DG-TSK

**Article**: G. Xue, J. Wang, B. Zhang, B. Yuan and C. Dai, "Double groups of gates based
Takagi-Sugeno-Kang (DG-TSK) fuzzy system for simultaneous feature selection and rule
extraction", *Fuzzy Sets and Systems*, vol. 469, 2023, doi:10.1016/j.fss.2023.108627.

**Script**: [`examples/reproductions/dgtsk_2023.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/dgtsk_2023.py)

The script reproduces two tables: the point-based rule base before any training
(Table 2), and DG-TSK after feature selection and rule extraction (Table 3).

## Protocol of the Article

Section 4 of the article. For Table 3 these settings are the defaults of
`DGTSKClassifier`.

| Point | Article | In the script |
|---|---|---|
| Inputs | scaled to $[0, 1]$ | `MinMaxScaler` fitted on the training part |
| Rule base | one rule per training sample, at most 300; one spread for every fuzzy set, the mean over the features of the standard deviations (Eq. 23) | Table 2: built with `GaussianMF` and `TSKClassifierModel`. Table 3: default (`rule_base="pfrb"`, `pfrb_spread="std_mean"`) |
| Consequents of the rule base | the label of the sample of each rule | Table 2: set directly. Table 3: default |
| Gates | M-gates on features and rules, parameters starting at 0.1 | default |
| Thresholds | $\zeta_\lambda = 0.5$, $\zeta_\theta = 0.01$, at least one rule per class | defaults |
| Optimizer | gradient descent on the whole training set | default |
| Learning rate and iterations | not stated in the article; 0.2 and 300 per phase in the authors' code | defaults |
| Evaluation | ten-fold cross-validation repeated ten times, mean of the 100 runs | the same, stratified folds |

## Result

Output of the script, on one thread:

```text
Table 2: point-based rule base without training (accuracy in percent)
dataset   article  highFIS
Iris        94.00    95.47
Wine        96.63    96.58

Table 3: DG-TSK (accuracy in percent / selected features / extracted rules)
dataset             article            highFIS  std of accuracy
Iris       96.8 / 2.3 / 3.1   95.7 / 2.0 / 6.0             5.29
Wine       98.3 / 8.0 / 3.0   97.7 / 5.1 / 3.0             3.37
```

- **Table 2.** The rule base of the article, built without training, is reproduced:
  within 0.1 point on Wine and 1.5 points above on Iris. With the same construction
  Wdbc gives 96.45 against 96.38 in the article.
- **Table 3, accuracy.** Within about one point on both datasets, a fraction of the
  standard deviation between runs.
- **Table 3, structure.** The three rules of Wine are reproduced. DG-TSK keeps 5.1
  features on Wine where the article reports 8.0, and 6.0 rules on Iris where the article
  reports 3.1.

## Comparison with the Authors' Code

The authors publish an implementation. Run on the same folds as highFIS (ten folds
repeated twice), it gives the structure of highFIS, not the one of the article's table:

| Dataset | Article | Authors' code | highFIS |
|---|---|---|---|
| Iris | 96.8 / 2.3 / 3.1 | 94.3 / 2.0 / 5.8 | 95.0 / 2.0 / 5.8 |
| Wine | 98.3 / 8.0 / 3.0 | 95.5 / 5.0 / 3.0 | 97.2 / 5.0 / 3.0 |
| Wdbc | 96.2 / 5.0 / 2.2 | 91.6 / 2.1 / 2.1 | 91.2 / 2.0 / 2.5 |

On Wine the two implementations select the same features and the same number of rules
in every one of the 20 folds. The distance to the article in the number of features and
rules is therefore between the article and its published code. The article does not
state the learning rate nor the number of iterations of its Table 3, and the number of
features that pass the threshold depends on them: with a gate phase of 60 iterations
instead of 300, highFIS keeps 10.7 features on Sonar, where the article reports 12.7,
and with 300 it keeps 3.9.

## Differences from the Article

- **Learning rate and iterations.** Taken from the authors' code, since the article does
  not state them.
- **Folds.** The article does not say whether its folds are stratified; the script uses
  stratified folds.
- **Wdbc** is left out of the script: neither highFIS nor the authors' code reaches the
  accuracy and the five features of the article there.
- **Datasets.** The article uses eighteen datasets. Iris and Wine ship with scikit-learn.

## Reproducibility

The script fixes the seeds, uses one thread and deterministic algorithms, and ends by
printing the versions of the libraries and the processor. Two runs on the same machine
print the same table. Another processor or another version of the numerical libraries
can change the last digits.

## Run It

```bash
python examples/reproductions/dgtsk_2023.py
```

No download is needed. A run takes about fifteen minutes.
