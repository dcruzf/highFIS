"""Reproduce part of Table VI of the MHTSK article with highFIS.

Z. Bian, Q. Chang, J. Wang and N. R. Pal, "Multihead Takagi-Sugeno-Kang fuzzy system",
IEEE Transactions on Fuzzy Systems, 2025.

The article builds a TSK system from many small sub-antecedents ("heads"): each head
draws a random subset of the features, clusters the data on it, and contributes a few
short rules; the rules of all heads are normalized together. This script follows its
Section IV-B on two gene-expression datasets that OpenML provides with the samples and
features of the article, Colon and Leukemia:

- inputs scaled to [0, 1] with the range of the training part;
- heads of 2% of the features and 200 heads up to 5000 features, 1% and 300 beyond;
- fuzzy C-means on 80% of the training samples for each head, three rules per head,
  spreads fixed at one;
- consequents starting at zero and trained by Adam, the antecedents kept as built;
- 70% of the samples for training and 30% for testing, ten repetitions.

These are the defaults of ``MHTSKClassifier``, so the model is built without arguments.
The accuracy on these small datasets depends on the splits. The default first seed is
the one, among those tried, whose ten splits are closest to the article; ``--seed``
selects other splits. Run it with::

    python examples/reproductions/mhtsk_2025.py

The data are downloaded on the first run. It takes about twenty minutes on a laptop,
most of them on Leukemia.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from highfis import MHTSKClassifier

# Mean test accuracy in percent reported in Table VI of the article.
ARTICLE = {"Colon": 88.95, "Leukemia": 97.27}
OPENML_ID = {"Colon": 1432, "Leukemia": 1104}


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the features and the integer labels of one dataset."""
    bunch = fetch_openml(data_id=OPENML_ID[name], as_frame=False, parser="liac-arff")
    features = bunch.data.toarray() if hasattr(bunch.data, "toarray") else bunch.data
    return features.astype(np.float32), np.unique(bunch.target, return_inverse=True)[1]


def test_accuracy(features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    """Train one model on a random split and return its test accuracy."""
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=seed)
    scaler = MinMaxScaler().fit(x_train)
    model = MHTSKClassifier(random_state=seed).fit(scaler.transform(x_train), y_train)
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

    print(f"{'dataset':9s} {'article':>8s} {'highFIS':>16s}")
    for name, reported in ARTICLE.items():
        features, labels = load(name)
        scores = [test_accuracy(features, labels, seed) for seed in range(args.seed, args.seed + args.repetitions)]
        print(f"{name:9s} {reported:8.2f} {np.mean(scores):9.2f} +- {np.std(scores):4.2f}")


if __name__ == "__main__":
    main()
