# Building Blocks of the Generic TSK

A TSK fuzzy system is assembled from three building blocks:

1. a **membership function** that turns each feature value into a degree between 0 and 1,
2. a **T-norm** that aggregates the degrees of one rule into its firing strength,
3. a **defuzzifier** that normalizes the firing strengths of all rules.

Each named family in highFIS (HTSK, ADMTSK, DG-ALETSK, ...) is a fixed combination of
these three, as published. Changing one of them would give a different model, so the
named estimators do not expose them; see [Model Families](../models/index.md) for what
each family fixes.

`TSKClassifier` and `TSKRegressor` are the generic estimators. They accept the three
building blocks as arguments, which makes them the place to see what each block does.

```python
from highfis import TSKClassifier

clf = TSKClassifier(n_mfs=3)                                    # classical TSK
clf = TSKClassifier(n_mfs=3, t_norm="min")                      # another aggregation
clf = TSKClassifier(n_mfs=3, mf="bell", defuzzifier="softmax_log")
```

The defaults are `mf="gaussian"`, `t_norm="prod"` and `defuzzifier="sum"`: the classical
(vanilla) TSK system.

## Membership function: `mf`

| Value | Shape | Notes |
|---|---|---|
| `"gaussian"` (default) | Gaussian bell | Smooth and positive everywhere. |
| `"gaussian_pi"` | Gaussian with a positive lower bound | Never falls below `exp(-1)`, which keeps a product of many degrees away from zero. |
| `"bell"` | Generalized bell | Flatter top and heavier tails than the Gaussian. |
| `"triangular"` | Triangle | Piecewise linear, zero outside its support. |
| `"trapezoidal"` | Trapezoid | Piecewise linear with a plateau of full membership, zero outside its support. |

The initialization chosen with `mf_init` always produces a centre and a width per fuzzy
set. Bell, triangular and trapezoidal sets are placed on that centre with the same width
at half maximum as the Gaussian, so changing `mf` changes the shape and nothing else.
`"gaussian_pi"` keeps the Gaussian's mean and sigma.

Triangular and trapezoidal sets are exactly zero outside their support. A sample that
lies outside the support of every rule gets the same weight for all rules, so the
prediction stays finite, but the model has no information there.

## T-norm: `t_norm`

| Value | Aggregation | Notes |
|---|---|---|
| `"prod"` (default) | Product of the degrees | The classical choice. With many features the product underflows. |
| `"min"` | Smallest degree | Only the weakest condition of the rule counts. |
| `"gmean"` | Geometric mean | The product rescaled by the number of features; used by HTSK and LogTSK. |
| `"dombi"` | Dombi T-norm | Parametric, with its shape parameter fixed at 1. |
| `"yager"` | Yager T-norm | Parametric, with its shape parameter fixed at 1. |
| `"yager_simple"` | Yager without the outer minimum | |
| `"ale_softmin_yager"` | Yager with a smooth minimum | |

The estimators take the T-norm by name, so the parametric ones use their default shape
parameter. To set it, build the model by hand with the classes in `highfis.t_norms` and
`highfis.models`, or use the family that adapts it: ADMTSK for Dombi, AYATSK for Yager.

## Defuzzifier: `defuzzifier`

| Value | Normalization | Notes |
|---|---|---|
| `"sum"` (default) | `w / sum(w)` | The classical choice. |
| `"softmax_log"` | `softmax(log w)` | The same quantity computed in log space, which is more stable; used by HTSK. |
| `"log_sum"` | `softmax(log(w) / T)` with `T = 1` | |
| `"inv_log"` | Inverse-log normalization | Used by LogTSK. |

## Changing one block at a time

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from highfis import TSKClassifier

X, y = load_wine(return_X_y=True)


def score(**options):
    clf = TSKClassifier(n_mfs=3, epochs=60, random_state=0, **options)
    return cross_val_score(make_pipeline(MinMaxScaler(), clf), X, y, cv=5).mean()


print(f"classical: {score():.3f}")
print(f"mf='bell': {score(mf='bell'):.3f}")
print(f"t_norm='min': {score(t_norm='min'):.3f}")
print(f"t_norm='gmean': {score(t_norm='gmean'):.3f}")
```

```text
classical: 0.967
mf='bell': 0.955
t_norm='min': 0.961
t_norm='gmean': 0.983
```

The exact figures depend on the platform. On a dataset with 13 features the choices are
close to each other. With many features the product underflows, which is the problem the
high-dimensional families address.

Because the options are ordinary constructor arguments, `GridSearchCV` can search over
them like any other hyperparameter:

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from highfis import TSKClassifier

X, y = load_wine(return_X_y=True)

pipe = Pipeline([("scale", MinMaxScaler()), ("clf", TSKClassifier(n_mfs=3, epochs=30, random_state=0))])
grid = {"clf__mf": ["gaussian", "triangular"], "clf__t_norm": ["prod", "gmean"]}

search = GridSearchCV(pipe, grid, cv=3).fit(X, y)
print(sorted(search.best_params_))
```

```text
['clf__mf', 'clf__t_norm']
```

## Where the families come from

Some families are the generic TSK with other building blocks. The geometric mean with
the softmax-in-log defuzzifier is HTSK:

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKClassifier, TSKClassifier

X, y = load_wine(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

tsk = TSKClassifier(n_mfs=3, epochs=20, random_state=0, t_norm="gmean", defuzzifier="softmax_log").fit(X, y)
htsk = HTSKClassifier(n_mfs=3, epochs=20, random_state=0).fit(X, y)

print(np.array_equal(tsk.predict_proba(X), htsk.predict_proba(X)))
```

```text
True
```

In the same way, `t_norm="gmean"` with `defuzzifier="inv_log"` is LogTSK. The other
families cannot be written this way: they use their own rule layer, or they compute a
parameter of one block from another, as described on their pages.

## Inspecting, saving and cloning

The chosen options are constructor parameters, so `get_params`, `clone`, `save` and
`load` carry them. `get_mf_params()` reports the membership functions with their own
parameter names:

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler

from highfis import TSKClassifier

X, y = load_wine(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

clf = TSKClassifier(n_mfs=3, epochs=20, random_state=0, mf="triangular").fit(X, y)
first = clf.get_mf_params()["x1"][0]
print(first["type"], sorted(k for k in first if k != "type"))
```

```text
TriangularMF ['center', 'left', 'right']
```
