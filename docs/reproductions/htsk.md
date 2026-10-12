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
| Spreads | drawn from $\mathcal{N}(1, 0.2)$ | `sigma_init="constant"` |
| Inputs | normalized | `StandardScaler` fitted on the training part |
| Repetitions | 10 | 10 random splits |

## Result

Output of the script, on four threads; the standard deviation is over the ten repetitions:

```text
Table II: mean test accuracy in percent
dataset  family   article  highFIS    std
Vowel    TSK        87.91    87.81   4.67
Vowel    LogTSK     85.42    84.28   3.53
Vowel    HTSK       88.32    89.97   4.68
Biodeg   TSK        85.71    85.46   1.52
Biodeg   LogTSK     85.87    84.73   2.82
Biodeg   HTSK       85.99    84.38   2.93
```

The six values are within 1.7 points of the article's, less than the standard deviation
between repetitions.

## Differences from the Article

- **Initial spreads.** The article draws every spread from $\mathcal{N}(1, 0.2)$ on
  normalized inputs, which the script selects with `sigma_init="constant"`. The default of
  highFIS, `sigma_init="cluster"`, centres the draw on the standard deviation of the
  feature inside the cluster (see [Initialization](../guides/initialization.md)), which
  does not assume standardized inputs. With the default the same protocol gives 86.16,
  81.68 and 89.36 on Vowel and 84.04, 85.08 and 84.70 on Biodeg for TSK, LogTSK and HTSK.
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
  OpenML provides with the samples and features of the article.

## Reproducibility

The script fixes the seeds, uses deterministic algorithms and four threads, and ends by
printing the versions of the libraries and the processor. Another processor or another
version of the numerical libraries can change the last digits.

## Run It

```bash
python examples/reproductions/htsk_2021.py
```

The data are downloaded from OpenML on the first run. A run takes about two minutes.
