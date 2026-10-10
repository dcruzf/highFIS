# FSRE-AdaTSK

**Article**: G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An adaptive neuro-fuzzy
system with integrated feature selection and rule extraction for high-dimensional
classification problems", *IEEE Transactions on Fuzzy Systems*, vol. 31, no. 7,
pp. 2167-2181, 2023.

**Script**: [`examples/reproductions/fsre_adatsk_2022.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/fsre_adatsk_2022.py)

The same article proposes AdaTSK, the classifier without feature selection, which has
its own [reproduction](adatsk.md).

## Protocol of the Article

Section IV-C of the article, for its Table IV.

| Point | Article | In the script |
|---|---|---|
| Phases | feature selection on the compact rule base, rule extraction on the enhanced one, fine-tuning | default |
| Fuzzy sets | ten per feature in the feature-selection phase, five in the rule-extraction phase, evenly placed between the minimum and the maximum of the training part | `fs_n_mfs=10`, `re_n_mfs=5` |
| Membership | $e^{-(x - m)^2}$: no spread (Eq. 3) | default |
| Gates | $M(u) = u\sqrt{e^{1 - u^2}}$, parameters starting at 0.01 | default |
| Thresholds | coefficients 0.5 for the features and 0.3 for the rules | defaults |
| Consequents | start at zero | default |
| Optimizer | gradient descent on the whole training set | default |
| Normalization of the consequent inputs | none | `consequent_batch_norm=False` |
| Inputs | not stated | standardized with the mean and the deviation of the training part |
| Learning rate | not stated | 0.05 (default) |
| Iterations | not stated | 1000 for feature selection and for fine-tuning (defaults), `re_epochs=600` for rule extraction |
| Evaluation | ten-fold cross-validation repeated five times, mean of the 50 runs | the same, stratified folds |

## Result

Output of the script, on one thread:

```text
Table IV: FSRE-AdaTSK (accuracy in percent / selected features / extracted rules)
dataset             article              highFIS  std of accuracy
Iris       96.5 / 2.1 / 6.1  96.40 / 2.00 / 6.20             4.46
Wine       97.3 / 6.3 / 6.3  96.18 / 8.22 / 6.80             5.80
Wdbc       95.4 / 6.2 / 4.9  95.26 / 7.12 / 4.36             2.61
```

The three accuracies are within about one point of the article, a fraction of the
standard deviation between runs. The numbers of extracted rules are reproduced: 6.2, 6.8
and 4.4 against 6.1, 6.3 and 4.9. The number of selected features is the article's on
Iris and one or two above it on Wine and Wdbc.

## Scale of the Inputs

The article does not say how the inputs are scaled, and the choice decides the number of
rules. Its membership function has no spread, so the scale of a feature sets how far
apart its fuzzy sets are. With five sets on a feature scaled to $[0, 1]$, the centre of a
set has a degree of 0.94 in the neighbouring set, the rules of the enhanced rule base
fire almost alike, and their gates open together. On standardized inputs the same degree
is typically between 0.1 and 0.4, and the gates of a few rules open well before the others.

The same protocol with the inputs scaled to $[0, 1]$, 300 iterations of rule
extraction and 3000 of fine-tuning:

| Dataset | Article | highFIS, inputs in $[0, 1]$ |
|---|---|---|
| Iris | 96.5 / 2.1 / 6.1 | 96.0 / 2.1 / 5.6 |
| Wine | 97.3 / 6.3 / 6.3 | 95.2 / 5.7 / 20.1 |
| Wdbc | 95.4 / 6.2 / 4.9 | 92.7 / 3.8 / 19.0 |

On that scale no length of the rule-extraction phase brought Wine below about fifteen
rules.

## Length of the Rule-Extraction Phase

The gates of the rules keep opening as the phase goes on, so the number of extracted
rules grows with the number of iterations, which the article does not state. On
standardized inputs, with the full protocol:

| Iterations | Iris | Wine | Wdbc |
|---|---|---|---|
| 600 (the script) | 96.40 / 2.0 / 6.2 | 96.18 / 8.2 / 6.8 | 95.26 / 7.1 / 4.4 |
| 700 | 95.87 / 2.0 / 7.5 | 96.41 / 8.2 / 7.8 | 95.29 / 7.1 / 4.6 |

The value of 600 was chosen as the one whose numbers of rules are closest to the article
on the three datasets. The accuracy changes by half a point or less between the two.

## Differences from the Article

- **Scale of the inputs, learning rate and iterations.** Not stated in the article;
  chosen here as described above.
- **Defaults of the estimator.** `FSREADATSKClassifier` defaults to three fuzzy sets in
  every phase and to normalization of the consequent inputs, a setting meant for inputs
  scaled to $[0, 1]$. With the defaults and that scale the same protocol gives
  96.9 / 3.0 / 7.5 on Iris and 98.9 / 9.1 / 11.7 on Wine.
- **Selection on the magnitude of the gate.** The gate is an odd function; highFIS
  compares the magnitude of each gate with the threshold.
- **Folds.** The article does not say whether its folds are stratified; the script uses
  stratified folds.
- **Datasets.** The article uses nineteen datasets, seven of them with more than 1000
  features, for which it changes the threshold coefficients to 0.4 and 0.5 and keeps the
  centres fixed in the first two phases. Iris, Wine and Wdbc ship with scikit-learn.

## Reproducibility

The script fixes the seeds, uses deterministic algorithms and one thread, and ends by
printing the versions of the libraries and the processor. Another processor or another
version of the numerical libraries can change the last digits.

## Run It

```bash
python examples/reproductions/fsre_adatsk_2022.py
```

No download is needed. A run takes about half an hour, most of it on Wdbc.
