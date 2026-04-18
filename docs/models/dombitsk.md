# DombiTSK

## Reference

> Dombi, J. (1982). "A general class of fuzzy operators." *Fuzzy Sets and
> Systems* 8(2): 149–163.

## Mathematical Formulation

DombiTSK extends TSK fuzzy inference by using a Dombi t-norm aggregation in
antecedent evaluation while keeping first-order linear consequents.

### Antecedent

Each rule-term membership is computed with a Gaussian function:

$$
\mu_{r,d}(x_d)=\exp\left(-\frac{(x_d-c_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

where $c_{r,d}$ is the center and $\sigma_{r,d}>0$ is the spread.

### Dombi aggregation

DombiTSK uses the Dombi t-norm to combine the antecedent membership values:

$$
\phi_r = \frac{1}{1 + \left(\sum_{d=1}^{D}
    \left(\frac{1 - \mu_{r,d}}{\mu_{r,d}}\right)^{\lambda}\right)^{1/\lambda}}
$$

The hyperparameter $\lambda > 0$ controls the aggregation shape:

- small $\lambda$ makes the aggregation softer and closer to product-like behavior,
- large $\lambda$ makes the aggregation sharper and closer to minimum-like behavior.

### Normalization

Rule firing strengths are normalized by sum normalization:

$$
\bar{\phi}_r = \frac{\phi_r}{\sum_{k=1}^{R} \phi_k}
$$

### Consequent

DombiTSK uses a first-order TSK consequent for both classification and
regression.

For classification:

$$
\mathbf{y}_r = W_r \mathbf{x} + \mathbf{b}_r
$$

For regression:

$$
\hat{y}_r = \mathbf{w}_r^\top \mathbf{x} + b_r
$$

### Output aggregation

The final prediction is the normalized weighted sum of rule consequents:

- Classification:

$$
\mathbf{y} = \sum_{r=1}^{R} \bar{\phi}_r \mathbf{y}_r
$$

- Regression:

$$
\hat{y} = \sum_{r=1}^{R} \bar{\phi}_r \hat{y}_r
$$

## Practical Notes

- `DombiTSKClassifier` and `DombiTSKRegressor` use `t_norm="dombi"` with
  `SumBasedDefuzzifier`.
- The `lambda_` hyperparameter controls the Dombi t-norm shape and must be
  positive.
- The default `rule_base` is `"cartesian"`, producing a conventional TSK rule
  grid from input membership functions.
- `consequent_batch_norm=True` can be used to normalize the consequent inputs
  before the linear output head.

## Example

```python
from highfis import DombiTSKClassifierEstimator

clf = DombiTSKClassifierEstimator(
    n_mfs=4,
    mf_init="kmeans",
    epochs=200,
    learning_rate=1e-3,
    random_state=42,
)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

Low-level use:

```python
from highfis import DombiTSKClassifier, GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}
model = DombiTSKClassifier(
    input_mfs,
    n_classes=3,
    lambda_=2.0,
)
history = model.fit(x_train, y_train, epochs=100, learning_rate=1e-3)
```
