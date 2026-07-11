# Choosing a model

highFIS exposes several TSK families behind the same scikit-learn interface, so you
can swap one for another by changing a single class. A few rules of thumb:

- **`TSKClassifier`** — the vanilla TSK baseline (product T-norm, sum normalization).
- **`HTSKClassifier`** — high-dimensional TSK (geometric-mean / log-space) for more
  features.
- **`LogTSKClassifier`** — log-domain inverse-log normalization for stable
  aggregation.
- **`ADPTSKClassifier`** — adaptive double-parameter softmin, tuned for
  high-dimensional data.
- **`DGTSKClassifier`** / **`FSREADATSKClassifier`** — add embedded feature
  selection and rule extraction (see the
  [feature-selection recipe](high-dimensional-feature-selection.md)).

Because they share the estimator API, comparing a few on a low-dimensional dataset
is just a loop:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKClassifier, LogTSKClassifier, TSKClassifier

X, y = load_iris(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

models = {
    "TSK": TSKClassifier(n_mfs=3, mf_init="grid", epochs=40, random_state=0),
    "HTSK": HTSKClassifier(n_mfs=3, mf_init="grid", epochs=40, random_state=0),
    "LogTSK": LogTSKClassifier(n_mfs=3, mf_init="grid", epochs=40, random_state=0),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=3)
    print(f"{name:8s} cv mean = {float(np.mean(scores)):.3f}")
```

The adaptive/gated families (`ADPTSKClassifier`, `DGTSKClassifier`,
`FSREADATSKClassifier`) are designed for **high-dimensional** problems and shine
there rather than on a small dataset like Iris. Every family also has a
`*Regressor` counterpart with the same interface.
