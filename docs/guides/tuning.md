# Hyperparameter Tuning and Scikit-Learn Integration

Every model family in **highFIS** provides a high-level estimator class that is fully compatible with the `scikit-learn` API. This compatibility means that highFIS estimators integrate natively with standard model selection, pipeline, and tuning tools such as `Pipeline`, `cross_val_score`, `GridSearchCV`, and `RandomizedSearchCV`.

---

## 1. Using highFIS in a Pipeline

TSK systems are sensitive to input scaling because membership functions are defined over the feature bounds. Preprocessing with a scaler like `MinMaxScaler` or `StandardScaler` is highly recommended.

You can build a scikit-learn `Pipeline` to chain preprocessing and model fitting together:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from highfis import HTSKClassifier

# Generate classification data
X, y = make_classification(n_samples=600, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Build a pipeline
pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("classifier", HTSKClassifier(n_mfs=3, epochs=100, random_state=42))
])

# Fit the entire pipeline
pipeline.fit(X_train, y_train)

# Evaluate on test data
accuracy = pipeline.score(X_test, y_test)
print(f"Pipeline Test Accuracy: {accuracy:.2%}")
```

---

## 2. Cross-Validation

You can perform k-fold cross-validation using `cross_val_score` to verify model stability:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from highfis import HTSKClassifier

# Chain scaler and estimator
model = make_pipeline(
    MinMaxScaler(),
    HTSKClassifier(n_mfs=3, epochs=80, random_state=42)
)

# Run stratified 5-fold cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print("CV Accuracies:", scores)
print("Mean Accuracy:", scores.mean())
```

---

## 3. Hyperparameter Tuning with GridSearchCV

To find the optimal configuration for your neuro-fuzzy system, you can use `GridSearchCV` to test combinations of hyperparameters:
*   `n_mfs` (number of membership functions/rules)
*   `mf_init` (initialization strategy: `"kmeans"`, `"fcm"`, etc.)
*   `learning_rate` (training step size)

When tuning parameters in a pipeline, prefix the parameter names with the pipeline step name followed by a double underscore (`classifier__<parameter>`).

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from highfis import HTSKClassifier

# Setup pipeline
pipe = Pipeline([
    ("scaler", MinMaxScaler()),
    ("classifier", HTSKClassifier(epochs=100, random_state=42, verbose=False))
])

# Define the parameter grid
param_grid = {
    "classifier__n_mfs": [2, 3, 5],
    "classifier__mf_init": ["kmeans", "fcm"],
    "classifier__learning_rate": [0.01, 0.005]
}

# Run grid search
grid = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", n_jobs=1)
grid.fit(X_train, y_train)

# Output results
print("Best parameters found:", grid.best_params_)
print("Best cross-validation accuracy:", grid.best_score_)

# Evaluate best estimator on holdout test set
best_pipeline = grid.best_estimator_
print("Test Score:", best_pipeline.score(X_test, y_test))
```

---

## 4. RandomizedSearchCV for Large Spaces

If you are tuning multiple parameters across a wide search space, `RandomizedSearchCV` is more efficient than exhaustive grid search. This reuses the `pipe` built in the previous section:

```{.python continuation}
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from highfis import HTSKClassifier

# Define parameter distribution
param_dist = {
    "classifier__n_mfs": randint(2, 8),
    "classifier__learning_rate": loguniform(1e-3, 1e-1),
    "classifier__mf_init": ["kmeans", "minibatch_kmeans", "fcm"]
}

# Run randomized search
random_search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    random_state=42,
    n_jobs=1
)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
```

---

## 5. Regression on Strongly Correlated Inputs

Every regressor in highFIS has first-order consequents trained by gradient-based
optimization. When the inputs are strongly correlated, as the channels of a spectrum are,
the consequents form an ill-conditioned problem and the default number of epochs is far
too small. This is a property of gradient training, not of a family: a plain linear model
trained by the same optimizer shows the same behaviour.

On the Tecator data (100 near-infrared channels, fat content as the target, inputs and
target scaled to `[0, 1]`), ridge regression has a test error of about 2.4 to 2.8 percent
fat and predicting the mean 13.0. With default settings the regressors lie between 7 and
12 (the HDFIS regressors above that), and a linear model trained by Adam for 200 epochs
gives 8.5.

Three remedies, in order of effect:

- **Decorrelate the inputs.** With ten whitened principal components in front of the
  estimator, the defaults give 2.7 for `HTSKRegressor`, 3.4 for `FSREADATSKRegressor` and
  3.3 for `TSKRegressor`.
- **Give the optimizer more updates**, with more epochs or with mini-batches.
  `HTSKRegressor(batch_size=32, epochs=300)` gives 5.7 on the raw channels and
  `TSKRegressor(epochs=2000)` gives 3.5.
- **Prefer a family trained by Adam** (see [Optimisers](optimisers.md)). DG-TSK, ADATSK
  and FSRE-ADATSK use plain gradient descent, as their articles do, and need many more
  epochs on such data: `FSREADATSKRegressor` reaches 4.4 with 5000 epochs per phase, and
  `DGTSKRegressor` does not get below 11 with 3000.

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from highfis import HTSKRegressor

model = make_pipeline(
    MinMaxScaler(),
    PCA(n_components=10, whiten=True),
    MinMaxScaler(),
    HTSKRegressor(random_state=0),
)
```

---

## 6. The Defaults Are a Starting Point

No default can be right for every dataset. The learning rate, the number of epochs, the
number of rules and the constants of each family interact with the number of features,
the number of samples and the correlation between the features. The defaults of highFIS
are chosen so that a first fit takes seconds and does not fail; they are not the best
values for your data, and the articles behind the families do not agree on them either.
**Search the hyperparameters before drawing a conclusion from a model**, with
`GridSearchCV` or `RandomizedSearchCV` as in sections 3 and 4.

### What the defaults do

- **Epochs** are kept small on purpose.
- **Learning rate.** Most families default to `0.01` with Adam. Six families default to
  `learning_rate="auto"`, a rule on the number of features, because one fixed value does
  not serve both a handful of features and thousands of them; the value chosen is stored
  in `learning_rate_` after `fit`. A number always replaces the rule.

  | Family | Optimizer | `"auto"` |
  |---|---|---|
  | ADATSK | gradient descent | `min(0.1, 1 / (n_features + 1))`, stable whatever the data |
  | DG-TSK | gradient descent | `min(0.2, 10 / n_features)`; 0.2 is the value of the article |
  | AYATSK, ADPTSK, DombiTSK, ADMTSK | Adam | `0.01` up to 1000 features, `0.001` above |

  For AYATSK the two values are those of its article. For the others the rule was
  measured: with a fixed value these families either did not train on few features or
  diverged on many.
- **The rule is a safe choice, not the best one.** The step of ADATSK never diverges,
  and on weakly correlated features a larger one trains faster.

### What to search

| Hyperparameter | Estimators | Typical grid |
|---|---|---|
| `learning_rate` | all | three values a factor of 3 to 10 apart around the default |
| `epochs` (`dg_epochs`, `finetune_epochs`, `fs_epochs`, ...) | all | the default, 3 times and 10 times the default |
| `n_mfs` | all | 2 to 10 with clustering; 3 or 5 with a grid |
| `batch_size` | families trained by Adam | 16, 64, full batch |
| `sigma_scale` | families initialized by clustering | 0.5, 1, 2, `"auto"` |
| `k` | AYATSK, ADPTSK | AYATSK: 2, 2.5, 3.3, 5, 10; ADPTSK: 0.2 to 2 |
| `zeta_lambda`, `zeta_theta` | DG-TSK, DG-ALETSK, FSRE-ADATSK | around the defaults |
| `dg_epochs` | DG-ALETSK | 10 (default), 30, 100: with few features the default is short (Wine: 93.9% with 10, 98.9% with 100) |
| `n_heads`, `head_size` | MHTSK | see the model page |

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from highfis import ADPTSKClassifier

search = GridSearchCV(
    Pipeline([("scale", MinMaxScaler()), ("model", ADPTSKClassifier(random_state=0))]),
    param_grid={
        "model__learning_rate": [0.001, 0.003, 0.01],
        "model__epochs": [200, 600],
        "model__k": [0.6, 1.0],
    },
    cv=5,
)
```

Keep the scaler inside the pipeline, so that it is fitted on the training part of each
fold, and report the score of a held-out set or of an outer cross-validation, not the
best score of the search. On small datasets the partition alone moves the accuracy by
several points; see [Reproductions](../reproductions/index.md).

### Signs that the budget is too small

- The training history (`history_["train_loss"]`) is still falling at the last epoch.
- A regressor scores below a linear model, or a classifier predicts fewer classes than
  there are.
- `fit` raises a `RuntimeWarning` that training diverged: lower the learning rate.
- `fit` raises a `DegenerateFiringWarning`: see
  [Rule-Firing Diagnostics](introspection.md).

### Regressors that need more epochs

R² on a test split, inputs and target scaled to `[0, 1]`. "Linear" is a target that is a
linear function of 8 features (ridge regression: 0.998); Friedman-1 has 10 features, 5 of
them informative (ridge regression: 0.625).

| Regressor | Setting | Linear | Friedman-1 |
|---|---|---|---|
| `DGTSKRegressor` | default (`dg_epochs=300, finetune_epochs=300`) | 0.553 | 0.635 |
| | `dg_epochs=3000, finetune_epochs=3000` | 0.98 | 0.71 |

With the defaults the other regressors reach the linear model on the first problem.
