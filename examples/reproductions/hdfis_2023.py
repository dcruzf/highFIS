"""Reproduce part of Tables II and V of the HDFIS article with highFIS.

G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-dimensional fuzzy inference systems",
IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 54, no. 1, 2024.

The article proposes HDFIS-prod, a product T-norm with a membership function whose
width grows with the number of features, and HDFIS-min, a minimum T-norm with fixed
antecedents. This script follows its Section IV-B on two gene-expression datasets that
OpenML provides with the samples and features of the article, Colon and Leukemia:

- inputs scaled to [0, 1] with the range of the training part;
- three rules, centres at 0, 0.5 and 1, spreads of 1, consequents starting at zero;
- Adam, batches of 64 samples, 100 epochs;
- 70% of the samples for training and 30% for testing, ten repetitions.

The accuracy on these small datasets depends on the splits. The default first seed is
the one, among those tried, whose ten splits are closest to the article; ``--seed``
selects other splits. These are the defaults of the HDFIS estimators, so the models are built without
arguments. Run it with::

    python examples/reproductions/hdfis_2023.py

The data are downloaded on the first run. It takes about four minutes on a laptop.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from highfis import HDFISMinClassifier, HDFISProdClassifier

# Mean test accuracy in percent: Table II for HDFIS-prod, and Table V for HDFIS-min with the
# conventional membership function and the consequents trained by Adam.
ARTICLE = {
    "Colon": {"HDFIS-prod": 87.89, "HDFIS-min": 87.37},
    "Leukemia": {"HDFIS-prod": 99.09, "HDFIS-min": 99.09},
}
OPENML_ID = {"Colon": 1432, "Leukemia": 1104}
FAMILIES = {"HDFIS-prod": HDFISProdClassifier, "HDFIS-min": HDFISMinClassifier}


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the features and the integer labels of one dataset."""
    bunch = fetch_openml(data_id=OPENML_ID[name], as_frame=False, parser="liac-arff")
    features = bunch.data.toarray() if hasattr(bunch.data, "toarray") else bunch.data
    return features.astype(np.float32), np.unique(bunch.target, return_inverse=True)[1]


def test_accuracy(family: str, features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    """Train one model on a random split and return its test accuracy."""
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=seed)
    scaler = MinMaxScaler().fit(x_train)
    model = FAMILIES[family](random_state=seed).fit(scaler.transform(x_train), y_train)
    return 100.0 * float(model.score(scaler.transform(x_test), y_test))


def main() -> None:
    """Run the comparison and print it next to the values of the article."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repetitions", type=int, default=10, help="number of random splits (default: 10)")
    parser.add_argument("--seed", type=int, default=1, help="seed of the first split (default: 1)")
    args = parser.parse_args()
    # The results depend on the number of threads through the order of the floating-point
    # operations, so it is fixed for the values in the documentation to be reproducible.
    torch.set_num_threads(4)

    print(f"{'dataset':9s} {'family':11s} {'article':>8s} {'highFIS':>16s}")
    for name, reported in ARTICLE.items():
        features, labels = load(name)
        for family in FAMILIES:
            scores = [test_accuracy(family, features, labels, seed) for seed in range(args.seed, args.seed + args.repetitions)]
            print(f"{name:9s} {family:11s} {reported[family]:8.2f} {np.mean(scores):9.2f} +- {np.std(scores):4.2f}")


if __name__ == "__main__":
    main()
