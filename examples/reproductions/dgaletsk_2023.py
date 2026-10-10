"""Reproduce part of Table V of the DG-ALETSK article with highFIS.

G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: a high-dimensional fuzzy approach with
simultaneous feature selection and rule extraction", IEEE Transactions on Fuzzy Systems,
vol. 31, no. 11, 2023.

The article builds a rule base with one rule per training sample, computes the firing
strengths with an adaptive softmin (ALE-softmin), and trains two groups of gates that
switch features and rules off. This script follows its Section IV on Colon, a
gene-expression dataset that OpenML provides with the samples and features of the article:

- inputs scaled to [0, 1] with the range of the training part;
- gates trained for 10 epochs with batches of 10% of the training samples, by Adam;
- thresholds of the gates chosen among the candidates of the article;
- ten-fold cross-validation repeated five times; accuracy, number of selected features and
  number of extracted rules averaged over the 50 runs.

These are the defaults of ``DGALETSKClassifier``, so the model is built without arguments.

Run it with::

    python examples/reproductions/dgaletsk_2023.py

The data are downloaded on the first run. It takes about fifteen minutes on a laptop, on
four threads.
"""

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import highfis
from highfis import DGALETSKClassifier

# --- Settings ---------------------------------------------------------------------------

SEED = 0
REPETITIONS = 5
FOLDS = 10
THREADS = 4

# OpenML identifier of each dataset.
DATASETS = {"Colon": 1432}

# Table V of the article: accuracy in percent, selected features, extracted rules.
ARTICLE = {"Colon": (81.95, 6.80, 6.42)}

# --- Reproducibility --------------------------------------------------------------------

torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
highfis.set_mf_cache_enabled(False)

# --- Protocol ---------------------------------------------------------------------------

results = {name: [] for name in DATASETS}

for name, data_id in DATASETS.items():
    bunch = fetch_openml(data_id=data_id, as_frame=False, parser="liac-arff")
    features = bunch.data.toarray().astype(np.float32)  # OpenML stores this dataset as a sparse matrix
    labels = np.unique(bunch.target, return_inverse=True)[1]

    for repetition in range(REPETITIONS):
        folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + repetition)
        for train, test in folds.split(features, labels):
            scaler = MinMaxScaler().fit(features[train])
            x_train, x_test = scaler.transform(features[train]), scaler.transform(features[test])

            model = DGALETSKClassifier(random_state=SEED + repetition).fit(x_train, labels[train])
            results[name].append(
                (100.0 * model.score(x_test, labels[test]), len(model.selected_features_), model.model_.n_rules)
            )

# --- Result -----------------------------------------------------------------------------

print("Table V: DG-ALETSK (accuracy in percent / selected features / extracted rules)")
print(f"{'dataset':8s} {'article':>20s} {'highFIS':>20s} {'std of accuracy':>16s}")
for name in DATASETS:
    accuracy, n_features, n_rules = np.asarray(results[name]).T
    reported = "{:.2f} / {:.2f} / {:.2f}".format(*ARTICLE[name])
    measured = f"{accuracy.mean():.2f} / {n_features.mean():.2f} / {n_rules.mean():.2f}"
    print(f"{name:8s} {reported:>20s} {measured:>20s} {accuracy.std():16.2f}")

print()
highfis.show_versions()
