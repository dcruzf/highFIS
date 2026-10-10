"""Reproduce parts of Tables 2 and 3 of the DG-TSK article with highFIS.

G. Xue, J. Wang, B. Zhang, B. Yuan and C. Dai, "Double groups of gates based
Takagi-Sugeno-Kang (DG-TSK) fuzzy system for simultaneous feature selection and rule
extraction", Fuzzy Sets and Systems, vol. 469, 2023.

The article builds a rule base with one rule per training sample (the point-based rule
base), and then trains two groups of gates that switch features and rules off. This
script reproduces both steps on Iris and Wine, which ship with scikit-learn:

1. Table 2, the point-based rule base before any training;
2. Table 3, DG-TSK after feature selection and rule extraction.

Protocol of the article, Section 4: inputs scaled to [0, 1]; at most 300 rules; ten-fold
cross-validation repeated ten times; accuracy, number of selected features and number of
extracted rules averaged over the 100 runs.

Run it with::

    python examples/reproductions/dgtsk_2023.py

No download is needed. It takes about fifteen minutes on a laptop, on one thread.
"""

import numpy as np
import torch
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import highfis
from highfis import DGTSKClassifier
from highfis.memberships import GaussianMF
from highfis.models import TSKClassifierModel

# --- Settings ---------------------------------------------------------------------------

SEED = 0
REPETITIONS = 10
FOLDS = 10
THREADS = 1

DATASETS = {"Iris": load_iris, "Wine": load_wine}

# Values of the article, in percent: Table 2 (point-based rule base, no training) and
# Table 3 (DG-TSK: accuracy, selected features, extracted rules).
ARTICLE_TABLE_2 = {"Iris": 94.00, "Wine": 96.63}
ARTICLE_TABLE_3 = {"Iris": (96.8, 2.3, 3.1), "Wine": (98.3, 8.0, 3.0)}

# --- Reproducibility --------------------------------------------------------------------

torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
highfis.set_mf_cache_enabled(False)

# --- Protocol ---------------------------------------------------------------------------

untrained = {name: [] for name in DATASETS}
trained = {name: [] for name in DATASETS}

for name, load in DATASETS.items():
    features, labels = load(return_X_y=True)
    n_classes = int(labels.max()) + 1

    for repetition in range(REPETITIONS):
        folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + repetition)
        for train, test in folds.split(features, labels):
            scaler = MinMaxScaler().fit(features[train])
            x_train = scaler.transform(features[train]).astype(np.float32)
            x_test = scaler.transform(features[test]).astype(np.float32)
            y_train, y_test = labels[train], labels[test]

            # Table 2. One rule per training sample: a Gaussian set centred on the sample for
            # every feature, one spread for all of them (Eq. 23), and the label of the
            # sample as the consequent. Nothing is trained.
            spread = float(np.std(x_train, axis=0, ddof=1).mean())
            fuzzy_sets = {
                f"x{d}": [GaussianMF(mean=float(centre), sigma=spread) for centre in x_train[:, d]]
                for d in range(x_train.shape[1])
            }
            rule_base = TSKClassifierModel(fuzzy_sets, n_classes=n_classes, rule_base="coco")
            with torch.no_grad():
                rule_base.consequent_layer.weight.zero_()
                rule_base.consequent_layer.bias.copy_(torch.eye(n_classes)[torch.as_tensor(y_train)])
                predicted = rule_base(torch.as_tensor(x_test)).argmax(dim=1).numpy()
            untrained[name].append(100.0 * np.mean(predicted == y_test))

            # Table 3. DG-TSK with its defaults, which are the settings of the article.
            model = DGTSKClassifier(random_state=SEED + repetition).fit(x_train, y_train)
            trained[name].append(
                (100.0 * model.score(x_test, y_test), len(model.selected_features_), model.model_.n_rules)
            )

# --- Result -----------------------------------------------------------------------------

print("Table 2: point-based rule base without training (accuracy in percent)")
print(f"{'dataset':8s} {'article':>8s} {'highFIS':>8s}")
for name in DATASETS:
    print(f"{name:8s} {ARTICLE_TABLE_2[name]:8.2f} {np.mean(untrained[name]):8.2f}")

print()
print("Table 3: DG-TSK (accuracy in percent / selected features / extracted rules)")
print(f"{'dataset':8s} {'article':>18s} {'highFIS':>18s} {'std of accuracy':>16s}")
for name in DATASETS:
    accuracy, n_features, n_rules = np.asarray(trained[name]).T
    reported = "{:.1f} / {:.1f} / {:.1f}".format(*ARTICLE_TABLE_3[name])
    measured = f"{accuracy.mean():.1f} / {n_features.mean():.1f} / {n_rules.mean():.1f}"
    print(f"{name:8s} {reported:>18s} {measured:>18s} {accuracy.std():16.2f}")

print()
highfis.show_versions()
