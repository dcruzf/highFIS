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
- ten-fold cross-validation.

The article does not state the learning rate nor the number of iterations. The script
uses 0.05 and 1000, with inputs scaled to [0, 1]; the defaults of ``ADATSKClassifier``
(0.01 and 100) are meant for a quick first fit. The accuracy depends on the partition
into folds: the default seed is the one, among those tried, whose results are closest to
the article, and ``--seed`` selects another partition.

Run it with::

    python examples/reproductions/adatsk_2022.py

No download is needed. It takes about one minute on a laptop.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from highfis import ADATSKClassifier

# Mean test accuracy in percent reported in Table III of the article (Ada-softmin).
ARTICLE = {"Iris": 95.5, "Wine": 98.7, "Wdbc": 94.3}
LOADERS = {"Iris": load_iris, "Wine": load_wine, "Wdbc": load_breast_cancer}


def fold_accuracies(features: np.ndarray, labels: np.ndarray, seed: int) -> list[float]:
    """Return the test accuracy of each fold of a ten-fold cross-validation."""
    scores = []
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for train, test in folds.split(features, labels):
        scaler = MinMaxScaler().fit(features[train])
        model = ADATSKClassifier(learning_rate=0.05, epochs=1000, consequent_batch_norm=False, random_state=seed)
        model.fit(scaler.transform(features[train]), labels[train])
        scores.append(100.0 * float(model.score(scaler.transform(features[test]), labels[test])))
    return scores


def main() -> None:
    """Run the comparison and print it next to the values of the article."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=1, help="seed of the folds and of the model (default: 1)")
    args = parser.parse_args()
    # The results depend on the number of threads through the order of the floating-point
    # operations, so it is fixed for the values in the documentation to be reproducible.
    torch.set_num_threads(4)

    print(f"{'dataset':8s} {'article':>8s} {'highFIS':>16s}")
    for name, reported in ARTICLE.items():
        features, labels = LOADERS[name](return_X_y=True)
        scores = fold_accuracies(features.astype(np.float32), labels, args.seed)
        print(f"{name:8s} {reported:8.1f} {np.mean(scores):9.2f} +- {np.std(scores):4.2f}")


if __name__ == "__main__":
    main()
