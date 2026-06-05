# Membership Function Initialization

In neuro-fuzzy Takagi-Sugeno-Kang (TSK) systems, **initialization** is the process of defining the starting locations (centers) and spreads (widths/shapes) of the fuzzy membership functions (MFs) in the input space before gradient descent optimization begins.

Because neuro-fuzzy systems train via backpropagation, poor initialization can lead to slow convergence, vanishing gradients, or local minima. **highFIS** provides two main paradigms for initialization: **uniform grid partitioning** and **data-driven clustering**.

---

## The Initialization Paradigms

```
   ┌────────────────────────────────────────────────────────┐
   │                  Initialization Mode                   │
   └───────────────┬────────────────────────┬───────────────┘
                   │                        │
                   ▼                        ▼
        [ Uniform Grid ("grid") ]   [ Clustering ("kmeans", "fcm", ...) ]
                   │                        │
                   ▼                        ▼
        Partition each dimension    Find clusters in joint
        independently.              input space.
                   │                        │
                   ▼                        ▼
        Cartesian Rule Base         CoCo (Co-Occurrence) Rule Base
        (M^D rules)                 (K rules total)
```

---

## 1. Uniform Grid Partitioning (`mf_init="grid"`)

Uniform grid partitioning divides the range of each input feature into equally spaced intervals.

*   **Behavior**: If you set `n_mfs=3` and `mf_init="grid"`, each input dimension will have 3 membership functions distributed evenly across its minimum and maximum values.
*   **Rule Base**: This mode defaults to the `cartesian` rule base, which generates rules representing all possible combinations of the membership functions across all dimensions.
*   **Rule Count**: For $D$ input features and $M$ membership functions per feature, the grid partitioning generates $M^D$ rules.

> **Warning:** Uniform grid partitioning suffers severely from the **Curse of Dimensionality**. If you have 10 features and choose 3 MFs per feature, the model will instantiate $3^{10} = 59,049$ fuzzy rules, leading to out-of-memory errors. Only use `"grid"` for low-dimensional inputs (typically $D \le 4$).

### Example: Grid Initialization for Low-Dimensional Regression

```python
from highfis import HTSKRegressor

# Instantiate a model using grid partition for a 3-dimensional input
reg = HTSKRegressor(
    n_mfs=3,
    mf_init="grid",
    rule_base="cartesian",  # Generates 3^3 = 27 rules
    epochs=50,
    random_state=42
)
```

---

## 2. Clustering-Based Initialization (`"kmeans"`, `"minibatch_kmeans"`, `"fcm"`)

Clustering-based initialization places membership functions on the centroids of clusters identified in the joint input data space.

*   **Behavior**: A clustering algorithm is run on the training matrix $X$ to find $K$ centroids. Each centroid represents a prototypical sample.
*   **Rule Base**: This mode defaults to the `coco` (Co-occurrence) rule base. Each cluster is translated directly into a single fuzzy rule.
*   **Rule Count**: Regardless of the number of input dimensions $D$, the system only instantiates $K$ rules (where $K$ is set via `n_rules` or `n_mfs`). This is the standard initialization strategy for high-dimensional datasets.

### Built-in Clustering Algorithms
*   `"kmeans"`: Standard K-Means clustering (full-batch).
*   `"minibatch_kmeans"`: Mini-Batch K-Means. Significantly faster on large datasets while yielding comparable cluster quality.
*   `"fcm"`: Fuzzy C-Means. Assigns soft, continuous membership values to centroids rather than hard clusters.

### Example: K-Means Initialization for High-Dimensional Classification

```python
from sklearn.datasets import make_classification
from highfis import HTSKClassifier

# Generate high-dimensional data (e.g. 30 features)
X, y = make_classification(n_samples=1000, n_features=30, random_state=42)

# Instantiate HTSK with K-Means initialization
clf = HTSKClassifier(
    n_mfs=5,  # We will find 5 clusters, creating exactly 5 rules
    mf_init="kmeans",
    rule_base="coco",
    epochs=100,
    random_state=42
)
clf.fit(X, y)
```

---

## 3. Customizing Clustering Estimators

Instead of passing string identifiers like `"kmeans"` or `"fcm"`, you can instantiate and configure a clustering class from the `highfis.clustering` module and pass it directly to `mf_init`. This allows you to customize hyperparameters such as maximum iterations, tolerance, or the Fuzzy C-Means fuzziness parameter $m$.

### Example: Customizing Fuzzy C-Means Parameters

```python
from highfis import HTSKClassifier
from highfis.clustering import FuzzyCMeans

# Configure a custom Fuzzy C-Means clusterer
custom_fcm = FuzzyCMeans(
    n_clusters=8,      # Matches n_mfs / number of rules desired
    m=2.5,             # Higher fuzziness coefficient (default is 2.0)
    max_iter=500,      # Increase max iterations for convergence
    tol=1e-5,          # Tighter convergence tolerance
    random_state=42
)

# Pass the custom FCM clusterer object directly
clf = HTSKClassifier(
    n_mfs=8,
    mf_init=custom_fcm,
    epochs=100,
    random_state=42
)
```
