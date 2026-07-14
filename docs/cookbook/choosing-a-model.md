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

Because the estimators are scikit-learn compatible, you can let `GridSearchCV`
**select the model for you** by putting the estimator itself in the search space —
searching over both the model family and its hyperparameters in one pass:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKClassifier, LogTSKClassifier, TSKClassifier

X, y = load_iris(return_X_y=True)

pipe = Pipeline([("scale", MinMaxScaler()), ("clf", TSKClassifier())])

# Each grid entry pins the "clf" step to a model family and searches its n_mfs.
param_grid = [
    {"clf": [TSKClassifier(mf_init="grid", epochs=40, random_state=0)], "clf__n_mfs": [2, 3]},
    {"clf": [HTSKClassifier(mf_init="grid", epochs=40, random_state=0)], "clf__n_mfs": [2, 3]},
    {"clf": [LogTSKClassifier(mf_init="grid", epochs=40, random_state=0)], "clf__n_mfs": [2, 3]},
]

search = GridSearchCV(pipe, param_grid, cv=3)
search.fit(X, y)

print("best model:", type(search.best_params_["clf"]).__name__)
print("best n_mfs:", search.best_params_["clf__n_mfs"])
print("best cv score:", round(float(search.best_score_), 3))
```

```text
best model: LogTSKClassifier
best n_mfs: 3
best cv score: 0.967
```

`GridSearchCV` refits the winning configuration on the full data as
`search.best_estimator_`, ready to `predict`. For larger or continuous search
spaces, use `RandomizedSearchCV` the same way. If you only want a quick side-by-side
comparison without selecting/refitting, a `cross_val_score` loop over the estimators
also works.

The adaptive/gated families (`ADPTSKClassifier`, `DGTSKClassifier`,
`FSREADATSKClassifier`) are designed for **high-dimensional** problems and shine
there rather than on a small dataset like Iris. Every family also has a
`*Regressor` counterpart with the same interface.
