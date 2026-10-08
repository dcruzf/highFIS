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
- ten-fold cross-validation.

Run it with::

    python examples/reproductions/ayatsk_2025.py

No download is needed. It takes about two minutes on a laptop.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from highfis import AYATSKClassifier

# Mean test accuracy in percent reported in Table III of the article, by K.
ARTICLE = {
    "Wine": {10: 98.44, 2: 98.42},
    "Wdbc": {10: 95.11, 2: 96.69},
}
LOADERS = {"Wine": load_wine, "Wdbc": load_breast_cancer}


def fold_accuracies(features: np.ndarray, labels: np.ndarray, k: int, seed: int) -> list[float]:
    """Return the test accuracy of each fold of a ten-fold cross-validation."""
    scores = []
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for train, test in folds.split(features, labels):
        scaler = MinMaxScaler().fit(features[train])
        model = AYATSKClassifier(k=float(k), learning_rate=0.01, random_state=seed)
        model.fit(scaler.transform(features[train]), labels[train])
        scores.append(100.0 * float(model.score(scaler.transform(features[test]), labels[test])))
    return scores


def main() -> None:
    """Run the comparison and print it next to the values of the article."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="seed of the folds and of the model (default: 0)")
    args = parser.parse_args()

    print(f"{'dataset':8s} {'K':>3s} {'article':>8s} {'highFIS':>16s}")
    for name, reported in ARTICLE.items():
        features, labels = LOADERS[name](return_X_y=True)
        for k, value in reported.items():
            scores = fold_accuracies(features.astype(np.float32), labels, k, args.seed)
            print(f"{name:8s} {k:3d} {value:8.2f} {np.mean(scores):9.2f} +- {np.std(scores):4.2f}")


if __name__ == "__main__":
    main()
