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
