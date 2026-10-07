# Persistence and the MF cache

## Save and load a model

Every estimator can be serialized with `save(path)` and restored with
`load(path)`. The reloaded estimator is equivalent to the original — including the
dtype of `classes_` / `predict()` — so it works with scikit-learn metrics.

```python
import tempfile
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKClassifier

X, y = load_iris(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

clf = HTSKClassifier(n_mfs=3, mf_init="grid", epochs=20, random_state=0)
clf.fit(X, y)

with tempfile.TemporaryDirectory() as tmp:
    path = str(Path(tmp) / "model.pt")
    clf.save(path)
    reloaded = HTSKClassifier.load(path)

print("same dtype:", reloaded.classes_.dtype == clf.classes_.dtype)
print("reloaded score:", round(reloaded.score(X, y), 3))
```

```text
same dtype: True
reloaded score: 0.733
```

## Save a pipeline and use it on new data

`save` stores the estimator alone. A model is usually fitted on preprocessed data, and
the preprocessing has to be applied to new data in exactly the same way. Put both in a
scikit-learn `Pipeline` and save the whole pipeline with `joblib`:

```python
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKClassifier

X, y = load_wine(return_X_y=True)
X_train, X_new, y_train, y_new = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

pipe = Pipeline([("scale", MinMaxScaler()), ("model", HTSKClassifier(n_mfs=3, epochs=60, random_state=0))])
pipe.fit(X_train, y_train)

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "wine-model.joblib"
    joblib.dump(pipe, path)
    reloaded = joblib.load(path)

# The reloaded pipeline takes raw data: it scales, then predicts.
print("same predictions:", np.array_equal(reloaded.predict(X_new), pipe.predict(X_new)))
print("accuracy on new data:", round(reloaded.score(X_new, y_new), 3))
```

```text
same predictions: True
accuracy on new data: 1.0
```

Which one to use:

| You want to keep | Use | Notes |
|---|---|---|
| The estimator alone | `estimator.save(path)` / `Estimator.load(path)` | Versioned checkpoint, loaded without executing code. You must apply the same preprocessing yourself. |
| Preprocessing and estimator together | `joblib.dump(pipeline, path)` / `joblib.load(path)` | The usual scikit-learn way. Reload with the same versions of highFIS, scikit-learn and PyTorch. |

`joblib` uses `pickle`, which can execute arbitrary code when loading. Only load files
that you created or that come from a source you trust.

The fitted model inside a pipeline is reached with `pipe["model"]`, for example
`pipe["model"].plot()` or `pipe["model"].save(path)`.

## Managing the membership-function cache

highFIS caches membership-function initialization so repeated `fit` calls with the
same data and hyperparameters skip the recompute. It is enabled by default; you can
inspect and control it programmatically.

```python
from highfis import (
    clear_mf_cache,
    mf_cache_info,
    set_mf_cache_enabled,
    set_mf_cache_size,
)

clear_mf_cache()
print("enabled:", mf_cache_info().enabled, "| size limit:", mf_cache_info().maxsize)

set_mf_cache_size(256)      # raise the maximum number of entries
set_mf_cache_enabled(False)  # bypass the cache entirely (always rebuild)
print("after disable:", mf_cache_info())

set_mf_cache_enabled(True)   # restore the default
clear_mf_cache()
```

```text
enabled: True | size limit: 128
after disable: MFCacheInfo(hits=0, misses=0, maxsize=256, currsize=0, enabled=False)
```

See the [Membership-function cache guide](../guides/caching.md) for details and the
`HIGHFIS_DISABLE_MF_CACHE` / `HIGHFIS_MF_CACHE_SIZE` environment variables.
