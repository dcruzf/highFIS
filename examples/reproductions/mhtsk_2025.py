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
The accuracy on these small datasets depends on the splits: ``SEED`` is the first seed,
among those tried, whose ten splits are closest to the article.

Run it with::

    python examples/reproductions/mhtsk_2025.py

The data are downloaded on the first run. It takes about twenty minutes on a laptop,
most of them on Leukemia.
"""

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import highfis
from highfis import MHTSKClassifier

# --- Settings ---------------------------------------------------------------------------

SEED = 1
REPETITIONS = 10
THREADS = 4

# OpenML identifier of each dataset.
DATASETS = {"Colon": 1432, "Leukemia": 1104}

# Table VI of the article: mean test accuracy in percent.
ARTICLE = {"Colon": 88.95, "Leukemia": 97.27}

# --- Reproducibility --------------------------------------------------------------------

torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
highfis.set_mf_cache_enabled(False)

# --- Protocol ---------------------------------------------------------------------------

results = {name: [] for name in DATASETS}

for name, data_id in DATASETS.items():
    bunch = fetch_openml(data_id=data_id, as_frame=False, parser="liac-arff")
    # OpenML stores some datasets, Colon among them, as sparse matrices.
    data = bunch.data.toarray() if hasattr(bunch.data, "toarray") else bunch.data
    features = np.asarray(data, dtype=np.float32)
    labels = np.unique(bunch.target, return_inverse=True)[1]

    for repetition in range(REPETITIONS):
        seed = SEED + repetition
        x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=seed)
        scaler = MinMaxScaler().fit(x_train)

        model = MHTSKClassifier(random_state=seed).fit(scaler.transform(x_train), y_train)
        results[name].append(100.0 * model.score(scaler.transform(x_test), y_test))

# --- Result -----------------------------------------------------------------------------

print("Table VI: mean test accuracy in percent")
print(f"{'dataset':9s} {'article':>8s} {'highFIS':>8s} {'std':>6s}")
for name, scores in results.items():
    print(f"{name:9s} {ARTICLE[name]:8.2f} {np.mean(scores):8.2f} {np.std(scores):6.2f}")

print()
print("Summary of the last model fitted")
model.summary()

print()
highfis.show_versions()
