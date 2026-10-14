"""Reproduce part of Table III of the AYATSK article with highFIS.

G. Xue, Y. Yang and J. Wang, "Adaptive Yager T-norm-based Takagi-Sugeno-Kang fuzzy
systems", IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2025.

The article proposes a TSK system whose firing strengths come from the Yager T-norm, with
a parameter computed from the number of features and a membership function bounded below
by 1/K. This script follows its Sections IV-A and IV-B on Wine and Wdbc, which ship with
scikit-learn, for the lower bounds 0.1 (K = 10, the default) and 0.5 (K = 2):

- inputs scaled to [0, 1] with the range of the training part;
- three rules, centres at 0, 0.5 and 1, spreads of 1, consequents starting at zero;
- Adam on the whole training set, 300 epochs;
- a learning rate of 0.01, the value of the article for low-dimensional data (the
  default, 0.001, is its value for more than 1000 features);
- ten-fold cross-validation repeated five times, mean of the 50 runs.

Run it with::

    python examples/reproductions/ayatsk_2025.py

No download is needed. It takes about ten minutes on a laptop.
"""

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import highfis
from highfis import AYATSKClassifier

# --- Settings ---------------------------------------------------------------------------

SEED = 0
REPETITIONS = 5
FOLDS = 10
THREADS = 4

DATASETS = {"Wine": load_wine, "Wdbc": load_breast_cancer}

# Table III of the article: mean test accuracy in percent, by K.
ARTICLE = {
    "Wine": {10: 98.44, 2: 98.42},
    "Wdbc": {10: 95.11, 2: 96.69},
}

# --- Reproducibility --------------------------------------------------------------------

torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
highfis.set_mf_cache_enabled(False)

# --- Protocol ---------------------------------------------------------------------------

results = {(name, k): [] for name in DATASETS for k in ARTICLE[name]}

for name, loader in DATASETS.items():
    features, labels = loader(return_X_y=True)
    features = features.astype(np.float32)

    for k in ARTICLE[name]:
        for repetition in range(REPETITIONS):
            folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + repetition)
            for train, test in folds.split(features, labels):
                scaler = MinMaxScaler().fit(features[train])
                x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])

                model = AYATSKClassifier(k=float(k), learning_rate=0.01, random_state=SEED + repetition)
                model.fit(x_train, labels[train])
                results[name, k].append(100.0 * model.score(x_test, labels[test]))

# --- Result -----------------------------------------------------------------------------

print("Table III: AYATSK (mean test accuracy in percent)")
print("std: over the runs, and over the means of the repetitions")
print(f"{'dataset':8s} {'K':>3s} {'article':>8s} {'highFIS':>8s} {'std':>12s}")
for (name, k), scores in results.items():
    accuracy = np.asarray(scores)
    deviations = f"{accuracy.std():.2f} / {accuracy.reshape(REPETITIONS, FOLDS).mean(axis=1).std():.2f}"
    print(f"{name:8s} {k:3d} {ARTICLE[name][k]:8.2f} {accuracy.mean():8.2f} {deviations:>12s}")

print()
print("Summary of the last model fitted")
model.summary()

print()
highfis.show_versions()
