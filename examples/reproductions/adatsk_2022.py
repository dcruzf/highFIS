"""Reproduce part of Table III of the AdaTSK article with highFIS.

G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An adaptive neuro-fuzzy system with
integrated feature selection and rule extraction for high-dimensional classification
problems", IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, 2023.

The article proposes Ada-softmin, a softmin whose index adapts to the membership degrees
so that it neither underflows nor returns a fake minimum, and the TSK classifier built on
it, AdaTSK. This script follows its Section IV on Iris, Wine and Wdbc, which ship with
scikit-learn:

- three fuzzy sets per feature, centres evenly placed between the minimum and the
  maximum of the training part, membership exp(-(x - m)^2);
- consequents starting at zero, no normalization of the consequent inputs;
- gradient descent on the whole training set;
- ten-fold cross-validation repeated five times, mean of the 50 runs.

The article does not state the learning rate, the number of iterations nor the scale of
the inputs. The script uses 0.05 and 1000, with inputs scaled to [0, 1].

Run it with::

    python examples/reproductions/adatsk_2022.py

No download is needed. It takes about five minutes on a laptop.
"""

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import highfis
from highfis import ADATSKClassifier

# --- Settings ---------------------------------------------------------------------------

SEED = 0
REPETITIONS = 5
FOLDS = 10
THREADS = 1

DATASETS = {"Iris": load_iris, "Wine": load_wine, "Wdbc": load_breast_cancer}

# Table III of the article (Ada-softmin): mean test accuracy in percent.
ARTICLE = {"Iris": 95.5, "Wine": 98.7, "Wdbc": 94.3}

# --- Reproducibility --------------------------------------------------------------------

torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
highfis.set_mf_cache_enabled(False)

# --- Protocol ---------------------------------------------------------------------------

results = {name: [] for name in DATASETS}

for name, loader in DATASETS.items():
    features, labels = loader(return_X_y=True)
    features = features.astype(np.float32)

    for repetition in range(REPETITIONS):
        folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + repetition)
        for train, test in folds.split(features, labels):
            scaler = MinMaxScaler().fit(features[train])
            x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])

            model = ADATSKClassifier(
                learning_rate=0.05,
                epochs=1000,
                consequent_batch_norm=False,
                random_state=SEED + repetition,
            ).fit(x_train, labels[train])
            results[name].append(100.0 * model.score(x_test, labels[test]))

# --- Result -----------------------------------------------------------------------------

print("Table III: AdaTSK (mean test accuracy in percent)")
print("std: over the runs, and over the means of the repetitions")
print(f"{'dataset':8s} {'article':>8s} {'highFIS':>8s} {'std':>12s}")
for name, scores in results.items():
    accuracy = np.asarray(scores)
    deviations = f"{accuracy.std():.2f} / {accuracy.reshape(REPETITIONS, FOLDS).mean(axis=1).std():.2f}"
    print(f"{name:8s} {ARTICLE[name]:8.1f} {accuracy.mean():8.2f} {deviations:>12s}")

print()
print("Summary of the last model fitted")
model.summary()

print()
highfis.show_versions()
