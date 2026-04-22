# AYATSK

## Reference

> G. Xue, Y. Yang and J. Wang, "Adaptive Yager T-Norm-Based Takagi–Sugeno–Kang Fuzzy Systems," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 55, no. 12, pp. 9802-9815, Dec. 2025, doi: 10.1109/TSMC.2025.3621346.

## Mathematical Formulation

### Antecedent

Each rule evaluates $D$ membership functions, typically Gaussian or
composite exponential functions:

$$
\,\mu_{r,d}(x_d)
= \exp\left(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

or, when a positive lower bound is desired:

$$
\mu_{r,d}(x_d) = k^{-1 + \exp\left(-\tfrac{1}{2}\left(\frac{x_d - c}{\sigma}\right)^2\right)}
$$

The rule firing strength uses the **Yager t-norm**:

$$
w_r = 1 - \left(\sum_{d=1}^{D} (1 - \mu_{r,d}(x_d))^{\lambda} \right)^{1/\lambda}
$$

where $\lambda > 0$ controls the softmin behavior of the aggregation.

### Defuzzification

AYATSK normalizes rule activations with a standard sum-based defuzzifier:

$$
\bar{f}_r = \frac{w_r}{\sum_{i=1}^{R} w_i}
$$

This is implemented by `SumBasedDefuzzifier`, making AYATSK compatible with
regular TSK consequent aggregation.

### Consequent (first-order)

For classification:

$$
\mathbf{y}_r = W_r \mathbf{x} + \mathbf{b}_r
$$

For regression:

$$
\hat{y}_r = \mathbf{w}_r^\top \mathbf{x} + b_r
$$

### Output aggregation

Final outputs are weighted averages of rule consequents:

- Classification:

$$
\mathbf{y} = \sum_{r=1}^{R} \bar{f}_r \, \mathbf{y}_r
$$

- Regression:

$$
\hat{y} = \sum_{r=1}^{R} \bar{f}_r \, \hat{y}_r
$$

## Practical Notes

- `AYATSKClassifier` and `AYATSKRegressor` use `t_norm="yager"` by default.
- `SumBasedDefuzzifier` is the default defuzzifier for AYATSK.
- `CompositeExponentialMF` is a useful membership type for AYATSK because it
  provides a positive lower bound and helps stabilize high-dimensional
  antecedent aggregation.
- To adjust the Yager shape parameter, pass a custom `t_norm_fn` such as
  `YagerTNorm(lambda_=2.0)`.

## Example

```python
from highfis import AYATSKClassifierEstimator

clf = AYATSKClassifierEstimator(
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
from highfis import AYATSKClassifier, CompositeExponentialMF

input_mfs = {
    "x1": [
        CompositeExponentialMF(center=0.0, sigma=1.0, k=10.0),
        CompositeExponentialMF(center=1.0, sigma=1.0, k=10.0),
    ],
    "x2": [
        CompositeExponentialMF(center=0.0, sigma=1.0, k=10.0),
        CompositeExponentialMF(center=1.0, sigma=1.0, k=10.0),
    ],
}
model = AYATSKClassifier(input_mfs, n_classes=3)
history = model.fit(x_train, y_train, epochs=100, learning_rate=1e-3)
```
