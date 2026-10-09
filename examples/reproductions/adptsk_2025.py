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
- ten-fold cross-validation.

These are the defaults of ``ADPTSKClassifier``, with ``K = 1``. Run it with::

    python examples/reproductions/adptsk_2025.py
    python examples/reproductions/adptsk_2025.py --k 0.6

The data are downloaded on the first run. It takes about five minutes on a laptop,
most of them on Leukemia.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from highfis import ADPTSKClassifier

# Mean test accuracy in percent reported in Table 3 of the article, by K.
ARTICLE = {
    "Colon": {0.2: 83.02, 0.4: 84.13, 0.6: 83.97, 0.8: 84.44, 1.0: 82.46, 1.5: 80.56, 2.0: 77.30},
    "Leukemia": {0.2: 97.14, 0.4: 98.10, 0.6: 97.32, 0.8: 97.26, 1.0: 97.20, 1.5: 92.86, 2.0: 92.50},
}
OPENML_ID = {"Colon": 1432, "Leukemia": 1104}


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the features and the integer labels of one dataset."""
    bunch = fetch_openml(data_id=OPENML_ID[name], as_frame=False, parser="liac-arff")
    features = bunch.data.toarray() if hasattr(bunch.data, "toarray") else bunch.data
    return features.astype(np.float32), np.unique(bunch.target, return_inverse=True)[1]


def fold_accuracies(features: np.ndarray, labels: np.ndarray, k: float, seed: int) -> list[float]:
    """Return the test accuracy of each fold of a ten-fold cross-validation."""
    scores = []
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for train, test in folds.split(features, labels):
        scaler = MinMaxScaler().fit(features[train])
        model = ADPTSKClassifier(k=k, random_state=seed)
        model.fit(scaler.transform(features[train]), labels[train])
        scores.append(100.0 * float(model.score(scaler.transform(features[test]), labels[test])))
    return scores


def main() -> None:
    """Run the comparison and print it next to the values of the article."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=float, default=1.0, choices=sorted(ARTICLE["Colon"]), help="K (default: 1.0)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the folds and of the model (default: 0)")
    args = parser.parse_args()

    print(f"{'dataset':9s} {'K':>4s} {'article':>8s} {'highFIS':>16s}")
    for name, reported in ARTICLE.items():
        features, labels = load(name)
        scores = fold_accuracies(features, labels, args.k, args.seed)
        print(f"{name:9s} {args.k:4.1f} {reported[args.k]:8.2f} {np.mean(scores):9.2f} +- {np.std(scores):4.2f}")


if __name__ == "__main__":
    main()
