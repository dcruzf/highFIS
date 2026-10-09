"""Reproduce part of Table II of the HTSK article with highFIS.

Y. Cui, D. Wu and Y. Xu, "Curse of dimensionality for TSK fuzzy neural networks:
explanation and solutions", IJCNN 2021.

The article compares the vanilla TSK system with LogTSK and HTSK on fourteen datasets.
This script follows the protocol of its Section IV-B on the two datasets that are
available from OpenML with the same samples and features, Vowel and Biodeg:

- 70% of the samples for training and 30% for testing;
- 10% of the training part as a validation set for early stopping, with a patience of 20
  epochs, at most 200 epochs, and the best model on the validation set kept;
- Adam with a learning rate of 0.01, batches of 512 samples or, when the training set is
  smaller, of min(N, 60);
- 30 rules, centres from k-means, spreads drawn from N(1, 0.2) on standardized inputs;
- ten repetitions, mean test accuracy.

Run it with::

    python examples/reproductions/htsk_2021.py

The data are downloaded on the first run. It takes about two minutes on a laptop.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from highfis import HTSKClassifier, LogTSKClassifier, TSKClassifier

# Mean test accuracy in percent reported in Table II of the article.
ARTICLE = {
    "Vowel": {"TSK": 87.91, "LogTSK": 85.42, "HTSK": 88.32},
    "Biodeg": {"TSK": 85.71, "LogTSK": 85.87, "HTSK": 85.99},
}
FAMILIES = {"TSK": TSKClassifier, "LogTSK": LogTSKClassifier, "HTSK": HTSKClassifier}


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the features and the integer labels of one dataset."""
    if name == "Vowel":
        bunch = fetch_openml(data_id=307, as_frame=False, parser="liac-arff")
        features = bunch.data[:, 2:]  # the first two columns identify the speaker
    else:
        bunch = fetch_openml(data_id=1494, as_frame=False, parser="liac-arff")
        features = bunch.data
    return features.astype(np.float32), np.unique(bunch.target, return_inverse=True)[1]


def test_accuracy(family: str, features: np.ndarray, labels: np.ndarray, seed: int) -> float:
    """Train one model with the protocol of the article and return its test accuracy."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
    scaler = StandardScaler().fit(x_train)
    n_train = len(x_train)
    model = FAMILIES[family](
        n_mfs=30,
        sigma_init="constant",
        epochs=200,
        patience=20,
        learning_rate=0.01,
        batch_size=512 if n_train >= 512 else min(n_train, 60),
        random_state=seed,
    )
    model.fit(scaler.transform(x_train), y_train, x_val=scaler.transform(x_val), y_val=y_val)
    return 100.0 * float(model.score(scaler.transform(x_test), y_test))


def main() -> None:
    """Run the comparison and print it next to the values of the article."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repetitions", type=int, default=10, help="number of random splits (default: 10)")
    args = parser.parse_args()
    # The results depend on the number of threads through the order of the floating-point
    # operations, so it is fixed for the values in the documentation to be reproducible.
    torch.set_num_threads(4)

    print(f"{'dataset':8s} {'family':7s} {'article':>8s} {'highFIS':>16s}")
    for name, reported in ARTICLE.items():
        features, labels = load(name)
        for family in FAMILIES:
            scores = [test_accuracy(family, features, labels, seed) for seed in range(args.repetitions)]
            print(f"{name:8s} {family:7s} {reported[family]:8.2f} {np.mean(scores):9.2f} +- {np.std(scores):4.2f}")


if __name__ == "__main__":
    main()
