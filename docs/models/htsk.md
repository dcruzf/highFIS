# HTSK

## Reference

> Cui, Y., Wu, D. & Xu, Y. (2021). "Curse of Dimensionality for TSK Fuzzy
> Neural Networks: Explanation and Solutions." *IJCNN 2021*.

## Mathematical Formulation

HTSK adapts TSK fuzzy inference for high-dimensional inputs by replacing the
standard product t-norm with a geometric mean and then normalizing rule
strengths in log-space.

### Antecedent

Each rule-term membership uses a Gaussian function:

$$
\mu_{r,d}(x_d)=\exp\left(-\frac{(x_d-m_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

where $m_{r,d}$ is the center and $\sigma_{r,d}>0$ is the spread.

### HTSK aggregation

HTSK computes each rule's antecedent activation as the geometric mean of the
membership values:

$$
w_r = \left(\prod_{d=1}^{D} \mu_{r,d}(x_d)\right)^{1/D}
= \exp\left(\frac{1}{D} \sum_{d=1}^{D} \log \mu_{r,d}(x_d)\right)
$$

This geometric mean reduces the dimensionality bias that makes product-based
firing strengths vanish in large $D$.

### Normalization

The normalized rule weights are computed in log-space for numerical
stability:

$$
\bar{w}_r = \frac{\exp(\log w_r)}{\sum_{i=1}^{R} \exp(\log w_i)}
$$

In code, this is implemented as `SoftmaxLogDefuzzifier` over the log-rule
weights.

### Consequent

HTSK uses a first-order TSK consequent for classification and regression.

For classification:

$$
\mathbf{y}_r = W_r \mathbf{x} + \mathbf{b}_r
$$

For regression:

$$
\hat{y}_r = \mathbf{w}_r^\top \mathbf{x} + b_r
$$

### Output aggregation

The final prediction is a normalized weighted sum of rule consequents:

- Classification:

$$
\mathbf{y} = \sum_{r=1}^{R} \bar{w}_r \mathbf{y}_r
$$

- Regression:

$$
\hat{y} = \sum_{r=1}^{R} \bar{w}_r \hat{y}_r
$$

## Practical Notes

- `HTSKClassifier` and `HTSKRegressor` use `t_norm="gmean"` by default.
- HTSK also uses `SoftmaxLogDefuzzifier` to normalize rule activations in a way
  that is stable for large $D$.
- The implementation is based on the HTSK formulation from Cui, Wu & Xu.
- `consequent_batch_norm=True` can be enabled to normalize feature inputs
  before the consequent layer.

## Example

```python
from highfis import HTSKClassifierEstimator

clf = HTSKClassifierEstimator(
    n_mfs=5,
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
from highfis import HTSKClassifier, GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}
model = HTSKClassifier(input_mfs, n_classes=3)
history = model.fit(x_train, y_train, epochs=100, learning_rate=1e-3)
```
