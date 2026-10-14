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

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import highfis
from highfis import HTSKClassifier, LogTSKClassifier, TSKClassifier

# --- Settings ---------------------------------------------------------------------------

SEED = 0
REPETITIONS = 10
THREADS = 4

# OpenML identifier of each dataset.
DATASETS = {"Vowel": 307, "Biodeg": 1494}

FAMILIES = {"TSK": TSKClassifier, "LogTSK": LogTSKClassifier, "HTSK": HTSKClassifier}

# Table II of the article: mean test accuracy in percent.
ARTICLE = {
    "Vowel": {"TSK": 87.91, "LogTSK": 85.42, "HTSK": 88.32},
    "Biodeg": {"TSK": 85.71, "LogTSK": 85.87, "HTSK": 85.99},
}

# --- Reproducibility --------------------------------------------------------------------

torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
highfis.set_mf_cache_enabled(False)

# --- Protocol ---------------------------------------------------------------------------

results = {(name, family): [] for name in DATASETS for family in FAMILIES}

for name, data_id in DATASETS.items():
    bunch = fetch_openml(data_id=data_id, as_frame=False, parser="liac-arff")
    # The first two columns of Vowel identify the speaker and are not features.
    features = (bunch.data[:, 2:] if name == "Vowel" else bunch.data).astype(np.float32)
    labels = np.unique(bunch.target, return_inverse=True)[1]

    for family, estimator in FAMILIES.items():
        for repetition in range(REPETITIONS):
            seed = SEED + repetition
            x_train, x_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=seed, stratify=labels
            )
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
            scaler = StandardScaler().fit(x_train)
            n_train = len(x_train)

            model = estimator(
                n_mfs=30,
                sigma_init="constant",
                epochs=200,
                patience=20,
                learning_rate=0.01,
                batch_size=512 if n_train >= 512 else min(n_train, 60),
                random_state=seed,
            )
            model.fit(scaler.transform(x_train), y_train, x_val=scaler.transform(x_val), y_val=y_val)
            results[name, family].append(100.0 * model.score(scaler.transform(x_test), y_test))

# --- Result -----------------------------------------------------------------------------

print("Table II: mean test accuracy in percent")
print(f"{'dataset':8s} {'family':7s} {'article':>8s} {'highFIS':>8s} {'std':>6s}")
for (name, family), scores in results.items():
    print(f"{name:8s} {family:7s} {ARTICLE[name][family]:8.2f} {np.mean(scores):8.2f} {np.std(scores):6.2f}")

print()
print("Summary of the last model fitted")
model.summary()

print()
highfis.show_versions()
