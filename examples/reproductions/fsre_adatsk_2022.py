"""Reproduce part of Table IV of the FSRE-AdaTSK article with highFIS.

G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An adaptive neuro-fuzzy system with
integrated feature selection and rule extraction for high-dimensional classification
problems", IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181, 2023.

The article trains a fuzzy classifier in three phases: gates select the features, then
gates extract the rules from an enhanced rule base built on the selected features, and
the remaining system is fine-tuned. This script follows its Section IV-C on eight of the
twelve low-dimensional datasets of the article:

- ten fuzzy sets per feature in the feature-selection phase and five in the
  rule-extraction phase, evenly placed between the minimum and the maximum of the feature;
- gradient descent on the whole training set, without normalization of the consequent
  inputs;
- ten-fold cross-validation repeated five times; accuracy, number of selected features and
  number of extracted rules averaged over the 50 runs.

The article does not state the scale of the inputs, the learning rate or the number of
iterations. Here the inputs are standardized, the learning rate is the default of the
classifier and the rule-extraction phase runs for 600 iterations.

Run it with::

    python examples/reproductions/fsre_adatsk_2022.py

Five of the datasets are downloaded from OpenML on the first run. It takes about an hour
and a half on a laptop, on one thread.
"""

import warnings

import numpy as np
import torch
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import highfis
from highfis import FSREADATSKClassifier

# --- Settings ---------------------------------------------------------------------------

SEED = 0
REPETITIONS = 5
FOLDS = 10
THREADS = 1

# Datasets that ship with scikit-learn, and OpenML identifiers of the others.
BUILT_IN = {"Iris": load_iris, "Wine": load_wine, "Wdbc": load_breast_cancer}
OPENML = {"Appendicitis": 1456, "Pima": 37, "Glass": 41, "Heart": 53, "Spectfheart": 1600}
DATASETS = ["Iris", "Appendicitis", "Pima", "Glass", "Wine", "Heart", "Wdbc", "Spectfheart"]

# Table IV of the article: accuracy in percent, selected features, extracted rules.
ARTICLE = {
    "Iris": (96.5, 2.1, 6.1),
    "Appendicitis": (86.3, 3.7, 3.1),
    "Pima": (75.7, 3.0, 3.2),
    "Glass": (65.5, 4.1, 6.0),
    "Wine": (97.3, 6.3, 6.3),
    "Heart": (82.5, 7.4, 2.9),
    "Wdbc": (95.4, 6.2, 4.9),
    "Spectfheart": (79.2, 3.4, 5.1),
}

# --- Reproducibility --------------------------------------------------------------------

torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
highfis.set_mf_cache_enabled(False)

# Glass has a class with nine samples, fewer than the folds; scikit-learn warns at each split.
warnings.filterwarnings("ignore", message="The least populated class")

# --- Data -------------------------------------------------------------------------------


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the features and the labels, numbered from zero, of a dataset."""
    if name in BUILT_IN:
        features, labels = BUILT_IN[name](return_X_y=True)
    else:
        bunch = fetch_openml(data_id=OPENML[name], as_frame=False, parser="liac-arff")
        features, labels = bunch.data, bunch.target
        if name == "Appendicitis":  # on OpenML the class is the last column of the data
            features, labels = features[:, :-1], features[:, -1]
    return features.astype(np.float32), np.unique(labels, return_inverse=True)[1]


# --- Protocol ---------------------------------------------------------------------------

results = {name: [] for name in DATASETS}

for name in DATASETS:
    features, labels = load(name)

    for repetition in range(REPETITIONS):
        folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + repetition)
        for train, test in folds.split(features, labels):
            scaler = StandardScaler().fit(features[train])
            x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])

            model = FSREADATSKClassifier(
                fs_n_mfs=10,
                re_n_mfs=5,
                re_epochs=600,
                consequent_batch_norm=False,
                random_state=SEED + repetition,
            ).fit(x_train, labels[train])
            results[name].append(
                (100.0 * model.score(x_test, labels[test]), len(model.selected_features_), model.model_.n_rules)
            )

# --- Result -----------------------------------------------------------------------------

print("Table IV: FSRE-AdaTSK (accuracy in percent / selected features / extracted rules)")
print("std of accuracy: over the runs, and over the means of the repetitions")
print(f"{'dataset':12s} {'article':>18s} {'highFIS':>20s} {'std of accuracy':>16s}")
for name in DATASETS:
    accuracy, n_features, n_rules = np.asarray(results[name]).T
    reported = "{:.1f} / {:.1f} / {:.1f}".format(*ARTICLE[name])
    measured = f"{accuracy.mean():.2f} / {n_features.mean():.2f} / {n_rules.mean():.2f}"
    deviations = f"{accuracy.std():.2f} / {accuracy.reshape(REPETITIONS, FOLDS).mean(axis=1).std():.2f}"
    print(f"{name:12s} {reported:>18s} {measured:>20s} {deviations:>16s}")

print()
print("Summary of the last model fitted")
print(model.summary())

print()
highfis.show_versions()
