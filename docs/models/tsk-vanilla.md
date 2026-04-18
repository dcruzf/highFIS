# Vanilla TSK

## Reference

> Takagi, T. & Sugeno, M. (1985). "Fuzzy identification of systems and
> its applications to modeling and control." *IEEE Trans. Syst., Man,
> Cybern.* SMC-15(1):116–132.
> DOI: [10.1109/TSMC.1985.6313399](https://doi.org/10.1109/TSMC.1985.6313399)

## Mathematical Formulation

### Antecedent

Each rule $r$ evaluates $D$ membership functions and aggregates them via the
**product t-norm**:

$$
\mu_{r,d}(x_d) = \exp\!\left(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}\right)
\tag{1}
$$

$$
w_r = \prod_{d=1}^{D} \mu_{r,d}(x_d)
\tag{2}
$$

### Defuzzification (sum-based)

The firing strengths are normalized by simple division:

$$
\bar{f}_r = \frac{w_r}{\sum_{i=1}^{R} w_i}
\tag{3}
$$

This is implemented by `SumBasedDefuzzifier`.

### Consequent (first-order)

$$
\hat{y} = \sum_{r=1}^{R} \bar{f}_r \left( b_{r,0} + \sum_{d=1}^{D} b_{r,d}\, x_d \right)
\tag{4}
$$

## Code ↔ Paper Correspondence

| Equation | Class / Method | Description |
|----------|---------------|-------------|
| (1) | `GaussianMF.forward()` | Gaussian membership function |
| (2) | `RuleLayer` with `t_norm="prod"` | Product t-norm via `torch.prod` |
| (3) | `SumBasedDefuzzifier.forward()` | `w.clamp(min=eps) / w.clamp(min=eps).sum(dim=1)` |
| (4) | `ClassificationConsequentLayer.forward()` / `RegressionConsequentLayer.forward()` | Weighted linear consequent |

## Differences from the paper

1. **Learnable parameters**: both $m_{r,d}$ and $\sigma_{r,d}$ are learned
   end-to-end via backpropagation. The original paper used least-squares for
   consequent parameters separately.
2. **$\sigma$ reparameterization**: `GaussianMF` stores a raw parameter and
   applies `softplus` to ensure $\sigma > 0$.
3. **Numeric stability**: `SumBasedDefuzzifier` clamps weights to a small
   floor before normalizing, avoiding underflow without biasing the denominator.

## Comparison with PyTSK

In the PyTSK reference implementation (`AntecedentGMF` with `high_dim=False`),
the product t-norm is computed as `torch.sum` of log-membership values, and
normalization is `softmax` over log-firing-strengths.  Mathematically:

$$
\text{softmax}(\log w)_r = \frac{e^{\log w_r}}{\sum_i e^{\log w_i}}
= \frac{w_r}{\sum_i w_i}
$$

This is **exactly** Equation (3) above.  In highFIS the two representations
are explicit — `SumBasedDefuzzifier` and `SoftmaxLogDefuzzifier` — and can be
swapped via the `defuzzifier` parameter.

## Example

```python
from highfis import TSKClassifierEstimator

clf = TSKClassifierEstimator(
    n_mfs=5,
    mf_init="kmeans",
    epochs=200,
    learning_rate=1e-3,
    random_state=42,
)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

Or using the lower-level `nn.Module` API:

```python
from highfis import TSKClassifier, GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}
model = TSKClassifier(input_mfs, n_classes=3)
history = model.fit(x_train, y_train, epochs=200, learning_rate=1e-3)
```
