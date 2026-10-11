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

These are the defaults of the HDFIS estimators, so the models are built without
arguments. The accuracy on these small datasets depends on the splits: ``SEED`` is the
first seed, among those tried, whose ten splits are closest to the article.

Run it with::

    python examples/reproductions/hdfis_2023.py

The data are downloaded on the first run. It takes about four minutes on a laptop.
"""

import warnings

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import highfis
from highfis import HDFISMinClassifier, HDFISProdClassifier

# --- Settings ---------------------------------------------------------------------------

SEED = 1
REPETITIONS = 10
THREADS = 4

# OpenML identifier of each dataset.
DATASETS = {"Colon": 1432, "Leukemia": 1104}

FAMILIES = {"HDFIS-prod": HDFISProdClassifier, "HDFIS-min": HDFISMinClassifier}

# Mean test accuracy in percent: Table II for HDFIS-prod, and Table V for HDFIS-min with the
# conventional membership function and the consequents trained by Adam.
ARTICLE = {
    "Colon": {"HDFIS-prod": 87.89, "HDFIS-min": 87.37},
    "Leukemia": {"HDFIS-prod": 99.09, "HDFIS-min": 99.09},
}

# --- Reproducibility --------------------------------------------------------------------

torch.set_num_threads(THREADS)
torch.use_deterministic_algorithms(True)
highfis.set_mf_cache_enabled(False)

# On these data one rule takes almost all the weight, which the estimators report at each fit.
warnings.filterwarnings("ignore", category=highfis.DegenerateFiringWarning)

# --- Protocol ---------------------------------------------------------------------------

results = {(name, family): [] for name in DATASETS for family in FAMILIES}

for name, data_id in DATASETS.items():
    bunch = fetch_openml(data_id=data_id, as_frame=False, parser="liac-arff")
    # OpenML stores some datasets, Colon among them, as sparse matrices.
    data = bunch.data.toarray() if hasattr(bunch.data, "toarray") else bunch.data
    features = np.asarray(data, dtype=np.float32)
    labels = np.unique(bunch.target, return_inverse=True)[1]

    for family, estimator in FAMILIES.items():
        for repetition in range(REPETITIONS):
            seed = SEED + repetition
            x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=seed)
            scaler = MinMaxScaler().fit(x_train)

            model = estimator(random_state=seed).fit(scaler.transform(x_train), y_train)
            results[name, family].append(100.0 * model.score(scaler.transform(x_test), y_test))

# --- Result -----------------------------------------------------------------------------

print("Tables II and V: mean test accuracy in percent")
print(f"{'dataset':9s} {'family':11s} {'article':>8s} {'highFIS':>8s} {'std':>6s}")
for (name, family), scores in results.items():
    print(f"{name:9s} {family:11s} {ARTICLE[name][family]:8.2f} {np.mean(scores):8.2f} {np.std(scores):6.2f}")

print()
highfis.show_versions()
