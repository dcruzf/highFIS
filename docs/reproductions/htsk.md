# HTSK, LogTSK and the Vanilla TSK

**Article**: Y. Cui, D. Wu and Y. Xu, "Curse of dimensionality for TSK fuzzy neural
networks: explanation and solutions", *International Joint Conference on Neural Networks
(IJCNN)*, 2021.

**Script**: [`examples/reproductions/htsk_2021.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/htsk_2021.py)

## Protocol of the Article

Section IV-B of the article, as the script applies it:

| Point | Article | In the script |
|---|---|---|
| Split | 70% training, 30% test | `train_test_split(test_size=0.3)`, stratified |
| Early stopping | 10% of the training part as validation set, patience 20, at most 200 epochs, best model kept | `fit(..., x_val=, y_val=)` with `epochs=200, patience=20` |
| Optimizer | Adam, learning rate 0.01 | `learning_rate=0.01` (the families use AdamW with a weight decay of `1e-8`) |
| Batch size | 512, or min(N, 60) for a smaller training set | `batch_size` set the same way |
| Rules | 30, centres from k-means | `n_mfs=30`, `mf_init="kmeans"` (default) |
| Inputs | normalized | `StandardScaler` fitted on the training part |
| Repetitions | 10 | 10 random splits |

## Result

Mean test accuracy in percent; for highFIS, with the standard deviation over the ten
repetitions.

| Dataset | Family | Article | highFIS 0.32.0 |
|---|---|---|---|
| Vowel (990 samples, 10 features, 11 classes) | TSK | 87.91 | 86.16 ± 2.69 |
| | LogTSK | 85.42 | 81.68 ± 4.32 |
| | HTSK | 88.32 | 89.36 ± 2.63 |
| Biodeg (1055 samples, 41 features, 2 classes) | TSK | 85.71 | 84.04 ± 3.60 |
| | LogTSK | 85.87 | 85.08 ± 2.24 |
| | HTSK | 85.99 | 84.70 ± 3.26 |

The six values are within one standard deviation of the article's. The furthest is
LogTSK on Vowel, 3.7 points below; see the first difference below.

## Differences from the Article

- **Initial spreads.** The article draws every spread from $\mathcal{N}(1, 0.2)$ on
  normalized inputs. highFIS centres the draw on the standard deviation of the feature
  inside the cluster (see [Initialization](../guides/initialization.md)). With the
  article's draw, the same protocol gives 87.81, 84.28 and 89.97 on Vowel and 85.49,
  84.73 and 84.38 on Biodeg for TSK, LogTSK and HTSK: closer to the article for TSK and
  LogTSK, the same for HTSK.
- **Number of rules and epochs.** The defaults of the estimators (`n_mfs=3` for HTSK,
  100 epochs, no validation set) are meant for a quick first fit, not for this
  protocol. With the defaults HTSK reaches 72% on Vowel, which has 11 classes and needs
  more than three rules.
- **Numerical floor.** HTSK and LogTSK compute the geometric mean of the membership
  degrees with each degree bounded below by the machine epsilon (about `1.2e-7` in
  single precision), so a feature further than 5.6 spreads from the centre of a set
  stops adding to the exponent. Raising the bound to the smallest representable number
  did not change any of the accuracies above.
- **Datasets.** The article uses fourteen datasets. Vowel and Biodeg are the two that
  OpenML provides with the samples and features of the article. On Colon (62 samples,
  2000 genes) the article reports 94.74 for HTSK. This protocol gives 72 ± 11 with
  highFIS and 80 ± 12 with the article's spreads, with 19 test samples per split and a
  validation set of 4 samples; a logistic regression gives 82 on the same splits. The
  value of the article was not reproduced.

## Run It

```bash
python examples/reproductions/htsk_2021.py
```

The data are downloaded from OpenML on the first run. Ten repetitions take about two
minutes; `--repetitions 3` gives a quicker look.
