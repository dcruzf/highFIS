# DG-ALETSK

**Article**: G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: a high-dimensional fuzzy
approach with simultaneous feature selection and rule extraction", *IEEE Transactions on
Fuzzy Systems*, vol. 31, no. 11, 2023, doi:10.1109/TFUZZ.2023.3270445.

**Script**: [`examples/reproductions/dgaletsk_2023.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/dgaletsk_2023.py)

## Protocol of the Article

Section IV of the article. These settings are the defaults of `DGALETSKClassifier`, so
the script builds the model without arguments.

| Point | Article | In the script |
|---|---|---|
| Inputs | scaled to $[0, 1]$ | `MinMaxScaler` fitted on the training part |
| Rule base | one rule per training sample, with a cap; spreads of 1 | defaults (`rule_base="pfrb"`, `pfrb_spread=1.0`) |
| Firing strength | ALE-softmin | default |
| Gates | on features and on rules, parameters starting at 0.01 | default |
| Gate phase | Adam, 10 epochs, batches of 10% of the training samples | defaults |
| Thresholds | chosen among candidate values after the gate phase | default |
| Evaluation | ten-fold cross-validation repeated five times, mean of the 50 runs | the same, stratified folds |

## Result

Output of the script, on four threads:

```text
Table V: DG-ALETSK (accuracy in percent / selected features / extracted rules)
dataset               article              highFIS  std of accuracy
Colon     81.95 / 6.80 / 6.42  86.95 / 7.18 / 6.98            10.75
```

On Colon (62 samples, 2000 genes) the numbers of selected genes and of extracted rules
are reproduced: 7.2 genes and 7.0 rules against 6.8 and 6.4. The accuracy is five points
above the article's, half the standard deviation between runs; with 62 samples one test
sample is 1.6 points of a cross-validated accuracy.

## SRBCT

The same protocol on SRBCT (83 samples, 2308 genes), with the copy of the data that the
authors distribute with the code of another of their articles:

| Dataset | Article | highFIS | Standard deviation of the accuracy |
|---|---|---|---|
| SRBCT | 95.00 / 15.62 / 10.10 | 94.92 / 13.1 / 4.1 | 7.72 |

The accuracy is reproduced within 0.1 point and the number of selected genes is close;
highFIS keeps four rules where the article reports ten. SRBCT is not in the script
because no source provides it without a further dependency.

## Differences from the Article

- **Threshold search.** The article fits the consequents by least squares while it
  searches the thresholds and holds out 10% of the training part for it; highFIS
  evaluates the candidates without that refit, which was more stable on these small
  samples. `use_lse=True` gives the refit.
- **Loss.** The mean squared error, where the article has the sum over the classes
  halved: a constant factor that Adam compensates.
- **Folds.** The article does not say whether its folds are stratified; the script uses
  stratified folds.
- **Datasets.** The article uses twelve datasets. Colon is the one that OpenML provides
  with the samples and features of the article, without a further dependency, and that
  runs in minutes.

## Reproducibility

The script fixes the seeds, uses deterministic algorithms and four threads, and ends by
printing the versions of the libraries and the processor. On Colon it prints the same
table on one thread and on four. Another processor or another version of the numerical
libraries can change the last digits.

## Run It

```bash
python examples/reproductions/dgaletsk_2023.py
```

The data are downloaded from OpenML on the first run. A run takes about fifteen minutes.
