# Quickstart: train, predict, and score

Train a TSK classifier on the Iris dataset and use it like any scikit-learn
estimator.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKClassifier

X, y = load_iris(return_X_y=True)
X = MinMaxScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = HTSKClassifier(n_mfs=3, mf_init="grid", epochs=30, random_state=42)
clf.fit(X_tr, y_tr)

print("accuracy:", round(clf.score(X_te, y_te), 3))
print("classes:", clf.classes_.tolist())
```

```text
accuracy: 0.733
classes: [0, 1, 2]
```

## Inside a scikit-learn pipeline

highFIS estimators plug into `Pipeline`, `cross_val_score`, and `GridSearchCV`.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKClassifier

X, y = load_iris(return_X_y=True)

pipe = Pipeline([
    ("scale", MinMaxScaler()),
    ("tsk", HTSKClassifier(n_mfs=3, mf_init="grid", epochs=20, random_state=0)),
])

scores = cross_val_score(pipe, X, y, cv=3)
print("cv mean:", round(float(np.mean(scores)), 3))

search = GridSearchCV(pipe, {"tsk__n_mfs": [2, 3]}, cv=3)
search.fit(X, y)
print("best n_mfs:", search.best_params_["tsk__n_mfs"])
```

```text
cv mean: 0.74
best n_mfs: 3
```
