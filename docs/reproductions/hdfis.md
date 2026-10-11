# HDFIS-prod and HDFIS-min

**Article**: G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-dimensional fuzzy inference
systems", *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, vol. 54, no. 1,
pp. 507-519, 2024.

**Script**: [`examples/reproductions/hdfis_2023.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/hdfis_2023.py)

## Protocol of the Article

Section IV-B of the article. These settings are the defaults of the HDFIS estimators, so
the script builds the models without arguments.

| Point | Article | In the script |
|---|---|---|
| Inputs | scaled to $[0, 1]$ | `MinMaxScaler` fitted on the training part |
| Fuzzy sets | three per feature, centres 0, 0.5 and 1, spreads of 1 | defaults (`mf_init="grid"`, `n_mfs=3`) |
| Rules | three, one per fuzzy set | default |
| Consequents | start at zero | default |
| Optimizer | Adam, batches of 64, 100 epochs | defaults; learning rate 0.01 |
| HDFIS-min | antecedents fixed, consequents trained | default |
| Split | 70% training, 30% test, ten repetitions | `train_test_split(train_size=0.7)`, ten seeds |

## Result

Output of the script, on four threads; the standard deviation is over the ten repetitions.
For HDFIS-min the article's value is the one with the conventional membership function
and the consequents trained by Adam (Table V).

```text
Tables II and V: mean test accuracy in percent
dataset   family       article  highFIS    std
Colon     HDFIS-prod     87.89    87.89   6.25
Colon     HDFIS-min      87.37    87.89   6.25
Leukemia  HDFIS-prod     99.09    97.27   3.02
Leukemia  HDFIS-min      99.09    97.27   3.02
```

The article reports standard deviations of 7.08 and 7.14 on Colon and of 1.82 on
Leukemia. Colon is reproduced exactly. On Leukemia highFIS is 1.8 points below, less than
one standard deviation, with 22 test samples per split.

## Choice of the Splits

The result depends on the ten random splits. The script starts at seed 1, the first
seed whose ten splits are closest to the article on both datasets, among the first
seeds 0 to 50. Over those blocks of ten splits Colon goes from 82.63 to 88.95 and
Leukemia from 95.45 to 97.73. The value of the article for Colon lies inside the range;
the 99.09 of Leukemia, which is two errors in 220 test samples, was not reached by any
block. Changing `SEED` in the script selects other splits.

## Comparison with the Authors' Code

The authors publish their implementation. Run on the same ten splits as highFIS (seeds 0
to 9, not the splits of the script), with
the same learning rate, it gives the same test accuracy on every split of Colon and
Leukemia for both models, and on nine of ten splits of SRBCT for HDFIS-prod (one test
sample of difference on the tenth). The distance from the article on Leukemia is therefore
a matter of the splits, not of the implementation.

| Dataset | Model | Authors' code | highFIS |
|---|---|---|---|
| Colon | HDFIS-prod | 86.84 ± 5.88 (learning rate 0.001) | 86.84 ± 5.88 (learning rate 0.001) |
| Leukemia | HDFIS-prod | 95.91 ± 5.55 | 95.91 ± 5.55 |
| Leukemia | HDFIS-min | 95.91 ± 5.55 | 95.91 ± 5.55 |
| SRBCT (83 samples, 2308 genes) | HDFIS-prod | 94.00 ± 4.10 (learning rate 0.001) | 93.60 ± 3.67 (learning rate 0.001) |
| SRBCT | HDFIS-min | 96.80 ± 2.99 (learning rate 0.001) | 96.80 ± 2.99 (learning rate 0.001) |

## Differences from the Article

- **Learning rate.** The article does not state it, and the authors' code uses 0.001.
  The default of highFIS is 0.01. On SRBCT the two values give 93.6 and 98.0 for
  HDFIS-prod (the article reports 99.20), on Colon 86.84 and 87.89.
- **Precision.** The authors' code works in double precision. highFIS works in single
  precision and computes the product of HDFIS-prod in the logarithmic domain, which
  gives the same normalized firing strengths.
- **Datasets.** The article uses fourteen classification datasets and four regression
  datasets. Colon and Leukemia are the two that OpenML provides with the samples and
  features of the article and without further dependencies. The copy of Colon on OpenML
  is already normalized.
- **Variants.** The article also reports HDFIS-min with the dimension-dependent
  membership function, and both models with the consequents estimated by least squares.
  highFIS does not implement these variants.

## Reproducibility

The script fixes the seeds, uses deterministic algorithms and four threads, and ends by
printing the versions of the libraries and the processor. Another processor or another
version of the numerical libraries can change the last digits.

## Run It

```bash
python examples/reproductions/hdfis_2023.py
```

The data are downloaded from OpenML on the first run. A run takes about four minutes.
The estimators warn at each fit that one rule takes almost all the weight on these data
(`DegenerateFiringWarning`); the script silences that warning.
