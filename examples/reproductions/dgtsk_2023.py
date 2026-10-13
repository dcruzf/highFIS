"""Reproduce parts of Tables 2 and 3 of the DG-TSK article with highFIS.

G. Xue, J. Wang, B. Zhang, B. Yuan and C. Dai, "Double groups of gates based
Takagi-Sugeno-Kang (DG-TSK) fuzzy system for simultaneous feature selection and rule
extraction", Fuzzy Sets and Systems, vol. 469, 2023.

The article builds a rule base with one rule per training sample (the point-based rule
base), and then trains two groups of gates that switch features and rules off. This
script reproduces both steps on Iris, Wine and Glass:

1. Table 2, the point-based rule base before any training;
2. Table 3, DG-TSK after feature selection and rule extraction.

Protocol of the article, Section 4: inputs scaled to [0, 1]; at most 300 rules; ten-fold
cross-validation repeated ten times; accuracy, number of selected features and number of
extracted rules averaged over the 100 runs.

The article does not state for how many iterations the gates are trained before the
features and rules are selected. The script uses 1000; the default of the classifier is
300, the value of the reference code of the authors.

Run it with::

    python examples/reproductions/dgtsk_2023.py

Glass is downloaded from OpenML on the first run. It takes about an hour and a half on a
laptop, on one thread.
"""

import warnings

import numpy as np
import torch
from sklearn.datasets import fetch_openml, load_iris, load_wine
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

GATE_ITERATIONS = 1000

DATASETS = ["Iris", "Wine", "Glass"]

# Values of the article, in percent: Table 2 (point-based rule base, no training) and
# Table 3 (DG-TSK: accuracy, selected features, extracted rules).
ARTICLE_TABLE_2 = {"Iris": 94.00, "Wine": 96.63, "Glass": 64.00}
ARTICLE_TABLE_3 = {"Iris": (96.8, 2.3, 3.1), "Wine": (98.3, 8.0, 3.0), "Glass": (63.1, 8.0, 6.0)}

# --- Reproducibility --------------------------------------------------------------------

torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
highfis.set_mf_cache_enabled(False)

# Glass has a class with nine samples, fewer than the folds; scikit-learn warns at each split.
warnings.filterwarnings("ignore", message="The least populated class")

# --- Data -------------------------------------------------------------------------------


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the features and the labels, numbered from zero, of a dataset."""
    if name == "Glass":
        bunch = fetch_openml(data_id=41, as_frame=False, parser="liac-arff")
        features, labels = bunch.data, bunch.target
    else:
        features, labels = {"Iris": load_iris, "Wine": load_wine}[name](return_X_y=True)
    return np.asarray(features, dtype=np.float64), np.unique(labels, return_inverse=True)[1]


# --- Protocol ---------------------------------------------------------------------------

untrained = {name: [] for name in DATASETS}
trained = {name: [] for name in DATASETS}

for name in DATASETS:
    features, labels = load(name)
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

            # Table 3. DG-TSK with its defaults, which are the settings of the article, and
            # the number of iterations of the gate phase chosen above.
            model = DGTSKClassifier(dg_epochs=GATE_ITERATIONS, random_state=SEED + repetition)
            model.fit(x_train, y_train)
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
print("Summary of the last model fitted")
model.summary()

print()
highfis.show_versions()
