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

If you are tuning multiple parameters across a wide search space, `RandomizedSearchCV` is more efficient than exhaustive grid search:

```python
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
