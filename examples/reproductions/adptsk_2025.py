"""Reproduce part of Table 3 of the ADPTSK article with highFIS.

"An adaptive double-parameter softmin based Takagi-Sugeno-Kang fuzzy system for
high-dimensional data", Fuzzy Sets and Systems, 2025, doi:10.1016/j.fss.2025.109582.

The article proposes a TSK system whose firing strengths come from a softmin with two
adaptive parameters, which approximates the minimum of the membership degrees without
overflow or underflow, with a Gaussian membership function bounded below by exp(-K).
This script follows its Section 4.1 on two gene-expression datasets that OpenML provides
with the samples and features of the article, Colon and Leukemia:

- inputs scaled to [0, 1] with the range of the training part;
- three rules, centres at 0, 0.5 and 1, spreads of 1, consequents starting at zero;
- Adam on the whole training set, 200 iterations, learning rate 0.001;
- ten-fold cross-validation repeated three times, mean of the 30 runs.

These are the defaults of ``ADPTSKClassifier``. The article studies ten values of K; the
script runs three of them, 0.6, 1.0 (the default) and 2.0.

Run it with::

    python examples/reproductions/adptsk_2025.py

The data are downloaded on the first run. It takes about an hour on a laptop, most of it
on Leukemia.
"""

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import highfis
from highfis import ADPTSKClassifier

# --- Settings ---------------------------------------------------------------------------

SEED = 0
REPETITIONS = 3
FOLDS = 10
THREADS = 4

# OpenML identifier of each dataset.
DATASETS = {"Colon": 1432, "Leukemia": 1104}

# Table 3 of the article: mean test accuracy in percent, by K.
ARTICLE = {
    "Colon": {0.6: 83.97, 1.0: 82.46, 2.0: 77.30},
    "Leukemia": {0.6: 97.32, 1.0: 97.20, 2.0: 92.50},
}

# --- Reproducibility --------------------------------------------------------------------

torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
highfis.set_mf_cache_enabled(False)

# --- Protocol ---------------------------------------------------------------------------

results = {(name, k): [] for name in DATASETS for k in ARTICLE[name]}

for name, data_id in DATASETS.items():
    bunch = fetch_openml(data_id=data_id, as_frame=False, parser="liac-arff")
    # OpenML stores some datasets, Colon among them, as sparse matrices.
    data = bunch.data.toarray() if hasattr(bunch.data, "toarray") else bunch.data
    features = np.asarray(data, dtype=np.float32)
    labels = np.unique(bunch.target, return_inverse=True)[1]

    for k in ARTICLE[name]:
        for repetition in range(REPETITIONS):
            folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + repetition)
            for train, test in folds.split(features, labels):
                scaler = MinMaxScaler().fit(features[train])
                x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])

                model = ADPTSKClassifier(k=k, random_state=SEED + repetition).fit(x_train, labels[train])
                results[name, k].append(100.0 * model.score(x_test, labels[test]))

# --- Result -----------------------------------------------------------------------------

print("Table 3: ADPTSK (mean test accuracy in percent)")
print("std: over the runs, and over the means of the repetitions")
print(f"{'dataset':9s} {'K':>4s} {'article':>8s} {'highFIS':>8s} {'std':>12s}")
for (name, k), scores in results.items():
    accuracy = np.asarray(scores)
    deviations = f"{accuracy.std():.2f} / {accuracy.reshape(REPETITIONS, FOLDS).mean(axis=1).std():.2f}"
    print(f"{name:9s} {k:4.1f} {ARTICLE[name][k]:8.2f} {accuracy.mean():8.2f} {deviations:>12s}")

print()
print("Summary of the last model fitted")
model.summary()

print()
highfis.show_versions()
