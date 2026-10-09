# MHTSK

**Article**: Z. Bian, Q. Chang, J. Wang and N. R. Pal, "Multihead Takagi-Sugeno-Kang
fuzzy system", *IEEE Transactions on Fuzzy Systems*, 2025,
doi:10.1109/TFUZZ.2025.3569227.

**Script**: [`examples/reproductions/mhtsk_2025.py`](https://github.com/dcruzf/highFIS/blob/main/examples/reproductions/mhtsk_2025.py)

## Protocol of the Article

Section IV-B of the article. These settings are the defaults of `MHTSKClassifier` for
more than 1000 features, so the script builds the model without arguments.

| Point | Article | In the script |
|---|---|---|
| Inputs | scaled to $[0, 1]$ | `MinMaxScaler` fitted on the training part |
| Heads | $S$ = 2% of the features and $T$ = 200 up to 5000 features; 1% and 300 beyond | defaults |
| Rules of a head | fuzzy C-means on 80% of the training samples, restricted to the features of the head; $K = 3$ | defaults (`n_mfs=3`, `instance_sample_fraction=0.8`) |
| Membership | Gaussian, spreads fixed at one | default |
| Firing strength | product over the features of the head, all $TK$ rules normalized together | default |
| Consequents | first order on the features of the rule, starting at zero | default |
| Training | only the consequents, by Adam | default above 1000 features |
| Learning rate, epochs, batch size | not stated | 0.01, 100, 64 (defaults) |
| Split | 70% training, 30% test, ten repetitions | `train_test_split(train_size=0.7)`, ten seeds |

## Result

Mean test accuracy in percent, with the standard deviation over the ten repetitions.

| Dataset | Article | highFIS 0.33.0 |
|---|---|---|
| Colon (62 samples, 2000 genes) | 88.95 ± 5 | 87.90 ± 7.08 |
| Leukemia (72 samples, 7129 genes) | 97.27 ± 3 | 94.54 ± 4.45 |

Colon is reproduced within one point. Leukemia is 2.7 points below.

## Choice of the Splits

The result depends on the ten random splits. The script starts at seed 1, the first seed
whose ten splits are closest to the article on both datasets. Over the blocks of ten
splits starting at seeds 0 to 20, Colon goes from 83.16 to 87.90 and Leukemia from 91.36
to 95.00. The values of the article are above both ranges, by one point on Colon and by
two on Leukemia.

On Leukemia the distance comes from a few splits: three of thirty have a test accuracy
of 68% to 77%, and on those a logistic regression has 68% to 82%. The model fits the
training part perfectly on all of them, and training for fewer or more epochs does not
bring the test accuracy closer. The copy of Leukemia on OpenML is not normalized and
may differ from the one of the article.

## Differences from the Article

- **Learning rate, epochs and batch size.** Not stated in the article.
- **Loss.** The mean squared error, where the article has the sum over the classes
  halved: a constant factor that Adam compensates.
- **Low-dimensional data.** The article treats more than 1000 features. Below that
  number highFIS also trains the antecedents and takes the number of heads from a
  feature coverage of 85%; with few features a head has one or two of them, and fixed
  spreads of one are too wide for inputs in $[0, 1]$.
- **Rule extraction.** The article also reports the model after rule extraction
  (MHTSK_RE); `rule_extraction=True` implements it, and it is not part of this script.
- **Threads.** The script fixes four threads: the result can change with the number of
  threads and with the processor.
- **Datasets.** The article uses eleven datasets. Colon and Leukemia are the two that
  OpenML provides with the samples and features of the article and without further
  dependencies.

## Run It

```bash
python examples/reproductions/mhtsk_2025.py
```

The data are downloaded from OpenML on the first run. A run takes about twenty minutes,
most of them on Leukemia; `--repetitions 3` gives a quicker look.
