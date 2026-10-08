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
- Adam, 50 epochs, batches of 10% of the training samples;
- ten-fold cross-validation.

These are the defaults of ``ADMTSKClassifier``. The article tries three learning rates and
two batch sizes and reports the best of the six; this script uses the default learning
rate of 0.01, or the one passed with ``--learning-rate``. Run it with::

    python examples/reproductions/admtsk_2025.py

The data are downloaded on the first run. It takes about nine minutes on a laptop,
most of them on Leukemia.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from highfis import ADMTSKClassifier

# Mean test accuracy in percent reported in Table V of the article.
ARTICLE = {"Colon": 86.33, "Leukemia": 97.29}
OPENML_ID = {"Colon": 1432, "Leukemia": 1104}


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the features and the integer labels of one dataset."""
    bunch = fetch_openml(data_id=OPENML_ID[name], as_frame=False, parser="liac-arff")
    features = bunch.data.toarray() if hasattr(bunch.data, "toarray") else bunch.data
    return features.astype(np.float32), np.unique(bunch.target, return_inverse=True)[1]


def fold_accuracies(features: np.ndarray, labels: np.ndarray, learning_rate: float, seed: int) -> list[float]:
    """Return the test accuracy of each fold of a ten-fold cross-validation."""
    scores = []
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for train, test in folds.split(features, labels):
        scaler = MinMaxScaler().fit(features[train])
        model = ADMTSKClassifier(learning_rate=learning_rate, random_state=seed)
        model.fit(scaler.transform(features[train]), labels[train])
        scores.append(100.0 * float(model.score(scaler.transform(features[test]), labels[test])))
    return scores


def main() -> None:
    """Run the comparison and print it next to the values of the article."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--learning-rate", type=float, default=0.01, help="learning rate of Adam (default: 0.01)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the folds and of the model (default: 0)")
    args = parser.parse_args()

    print(f"{'dataset':9s} {'article':>8s} {'highFIS':>16s}")
    for name, reported in ARTICLE.items():
        features, labels = load(name)
        scores = fold_accuracies(features, labels, args.learning_rate, args.seed)
        print(f"{name:9s} {reported:8.2f} {np.mean(scores):9.2f} +- {np.std(scores):4.2f}")


if __name__ == "__main__":
    main()
