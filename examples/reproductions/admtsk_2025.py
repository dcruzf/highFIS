"""Reproduce part of Table V of the ADMTSK article with highFIS.

G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: a high-dimensional Takagi-Sugeno-Kang
fuzzy system based on adaptive Dombi T-norm", IEEE Transactions on Fuzzy Systems, vol. 33,
no. 6, 2025.

The article proposes a TSK system whose firing strengths come from the Dombi T-norm, with
a parameter computed from the number of features and a Gaussian membership function
bounded below by 1/e. This script follows its Section IV-A on two gene-expression datasets
that OpenML provides with the samples and features of the article, Colon and Leukemia:

- inputs scaled to [0, 1] with the range of the training part;
- three rules, centres at 0, 0.5 and 1, spreads of 1, consequents starting at zero;
- Adam, 50 epochs;
- learning rates of 0.01, 0.001 and 0.0001, batches of 10% and 20% of the training
  samples: six combinations, of which the article reports the best for each dataset;
- ten-fold cross-validation repeated five times, mean of the 50 runs.

The script runs the six combinations and prints each of them and the best, as the article
does. The best is chosen on the test folds, so it is an optimistic estimate. The
defaults of ``ADMTSKClassifier`` are the third combination on these datasets: batches of
10% and, above 1000 features, a learning rate of 0.001.

Run it with::

    python examples/reproductions/admtsk_2025.py

The data are downloaded on the first run. It takes about four hours on a laptop, most of
them on Leukemia; remove it from ``DATASETS`` for a run of half an hour.
"""

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import highfis
from highfis import ADMTSKClassifier

# --- Settings ---------------------------------------------------------------------------

SEED = 0
REPETITIONS = 5
FOLDS = 10
THREADS = 4

# OpenML identifier of each dataset.
DATASETS = {"Colon": 1432, "Leukemia": 1104}

# The six combinations of the article: learning rate and fraction of the training samples
# in a batch.
COMBINATIONS = [(0.01, 0.1), (0.01, 0.2), (0.001, 0.1), (0.001, 0.2), (0.0001, 0.1), (0.0001, 0.2)]

# Table V of the article: mean test accuracy in percent, the best of the six combinations.
ARTICLE = {"Colon": 86.33, "Leukemia": 97.29}

# --- Reproducibility --------------------------------------------------------------------

torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
highfis.set_mf_cache_enabled(False)

# --- Protocol ---------------------------------------------------------------------------

results = {(name, combination): [] for name in DATASETS for combination in COMBINATIONS}

for name, data_id in DATASETS.items():
    bunch = fetch_openml(data_id=data_id, as_frame=False, parser="liac-arff")
    # OpenML stores some datasets, Colon among them, as sparse matrices.
    data = bunch.data.toarray() if hasattr(bunch.data, "toarray") else bunch.data
    features = np.asarray(data, dtype=np.float32)
    labels = np.unique(bunch.target, return_inverse=True)[1]

    for learning_rate, batch_fraction in COMBINATIONS:
        for repetition in range(REPETITIONS):
            folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + repetition)
            for train, test in folds.split(features, labels):
                scaler = MinMaxScaler().fit(features[train])
                x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])

                model = ADMTSKClassifier(
                    learning_rate=learning_rate,
                    batch_size=max(1, round(batch_fraction * len(train))),
                    random_state=SEED + repetition,
                ).fit(x_train, labels[train])
                results[name, (learning_rate, batch_fraction)].append(100.0 * model.score(x_test, labels[test]))

# --- Result -----------------------------------------------------------------------------

print("ADMTSK: mean test accuracy in percent of each combination")
print(f"{'dataset':9s} {'learning rate':>13s} {'batch':>6s} {'highFIS':>8s} {'std':>6s}")
for (name, (learning_rate, batch_fraction)), scores in results.items():
    print(f"{name:9s} {learning_rate:13g} {batch_fraction:6.0%} {np.mean(scores):8.2f} {np.std(scores):6.2f}")

print()
print("Table V: the best of the six combinations (mean test accuracy in percent)")
print(f"{'dataset':9s} {'article':>8s} {'highFIS':>8s}")
for name in DATASETS:
    best = max(np.mean(results[name, combination]) for combination in COMBINATIONS)
    print(f"{name:9s} {ARTICLE[name]:8.2f} {best:8.2f}")

print()
highfis.show_versions()
