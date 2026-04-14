# LogTSK

## Reference

> Cui, Y., Wu, D. & Xu, Y. (2021). "Optimize TSK Fuzzy Systems for
> Regression Problems: Mini-Batch Gradient Descent With Regularization,
> DropRule, and AdaBound (MBGD-RDA)." *IEEE Trans. Fuzzy Syst.*
> 29(5):1003–1015.
> DOI: [10.1109/TFUZZ.2020.2979968](https://doi.org/10.1109/TFUZZ.2020.2979968)

## Mathematical Formulation

### Antecedent

Same as vanilla TSK — product t-norm over Gaussian membership values:

$$
w_r = \prod_{d=1}^{D} \mu_{r,d}(x_d)
\tag{1}
$$

### Defuzzification (log-space with temperature)

LogTSK operates entirely in logarithmic space to avoid the underflow that
plagues high-dimensional TSK systems.  After taking the log of firing
strengths, a **temperature** parameter $\tau$ controls the sharpness of
the distribution:

$$
\log \bar{f}_r = \frac{\log w_r}{\tau}
  - \log\!\left(\sum_{i=1}^{R}
    \exp\!\left(\frac{\log w_i}{\tau}\right)\right)
\tag{2}
$$

which is equivalent to:

$$
\bar{f}_r = \text{softmax}\!\left(\frac{\log \mathbf{w}}{\tau}\right)_r
\tag{3}
$$

**Special cases**:

| $\tau$ | Behaviour |
|--------|-----------|
| $\tau = 1$ | Recovers `SoftmaxLogDefuzzifier` ≡ `w / sum(w)` (vanilla TSK normalization) |
| $\tau < 1$ | Sharpens the distribution — amplifies the dominant rule |
| $\tau > 1$ | Smooths the distribution — makes rules more uniform |

This is implemented by `LogSumDefuzzifier(temperature=τ)`.

### Consequent (first-order)

$$
\hat{y} = \sum_{r=1}^{R} \bar{f}_r \left( b_{r,0} + \sum_{d=1}^{D} b_{r,d}\, x_d \right)
\tag{4}
$$

## Code ↔ Paper Correspondence

| Equation | Class / Method | Description |
|----------|---------------|-------------|
| (1) | `RuleLayer` with `t_norm="prod"` | Product t-norm |
| (2)–(3) | `LogSumDefuzzifier.forward()` | `softmax(log(w) / τ)` |
| (4) | `ClassificationConsequentLayer` / `RegressionConsequentLayer` | Weighted consequent |

## Differences from the paper

1. **Temperature parameter**: The original MBGD-RDA paper describes operating
   in log-space but does not introduce an explicit temperature.  We expose $\tau$
   as a hyperparameter (default 1.0) for additional control.
2. **DropRule and AdaBound**: These are described in the same paper but belong to
   Phase 2 of the roadmap.  This implementation focuses solely on the log-space
   defuzzification component.
3. **Learnable parameters**: All MF parameters are learned via backpropagation
   (the paper uses a similar approach with MBGD).

## Comparison with PyTSK

In PyTSK, the antecedent computes log-firing-strengths and passes them through
`torch.softmax(frs, dim=1)`.  With `high_dim=False` (product t-norm), this is:

$$
\text{softmax}(\log w)_r = \frac{w_r}{\sum_i w_i}
$$

PyTSK does **not** expose a temperature parameter — it always uses $\tau = 1$.
The highFIS `LogSumDefuzzifier` generalizes this by dividing the log-values by
$\tau$ before applying softmax, providing finer control over rule dominance.

## Example

```python
from highfis import LogTSKClassifierEstimator

clf = LogTSKClassifierEstimator(
    n_mfs=5,
    mf_init="kmeans",
    epochs=200,
    learning_rate=1e-3,
    random_state=42,
)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

Using the lower-level API with custom temperature:

```python
from highfis import LogTSKClassifier, GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}
model = LogTSKClassifier(input_mfs, n_classes=3, temperature=0.5)
history = model.fit(x_train, y_train, epochs=200, learning_rate=1e-3)
```
