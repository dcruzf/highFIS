# AdaTSK

## Reference

> Xue, Guangdong; Chang, Qin; Wang, Jian; Zhang, Kai; Pal, Nikhil R. (2023).
> "An Adaptive Neuro-Fuzzy System With Integrated Feature Selection and
> Rule Extraction for High-Dimensional Classification Problems." *IEEE
> Transactions on Fuzzy Systems* 31(7):2167–2181.
> DOI: [10.1109/TFUZZ.2022.3220950](https://doi.org/10.1109/TFUZZ.2022.3220950)

## Mathematical Formulation

AdaTSK extends TSK fuzzy inference by using an adaptive Dombi-based
aggregation in the antecedent and a Composite Gaussian membership function
with a nonzero lower bound.

### Antecedent

Each rule-term membership is computed with a Composite Gaussian MF:

$$
\mu_{r,d}(x_d) = \epsilon + (1 - \epsilon)\exp\left(-\frac{(x_d - c_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

where $\epsilon$ is a small constant (e.g. $10^{-6}$) and
$\sigma_{r,d} > 0$ is enforced via a softplus reparameterization.

### Adaptive Dombi aggregation

The rule firing strength uses a per-rule adaptive Dombi t-norm:

$$
\phi_r = \left(1 + \sum_{d=1}^{D} \left[\left(\frac{1 - \mu_{r,d}}{\mu_{r,d}}\right)^{\lambda_r}\right]\right)^{-1/\lambda_r}
$$

Each $\lambda_r > 0$ is learned via softplus, which allows the model to
adapt the softmin behavior of each rule individually.

### Normalization

Rule firing strengths are normalized by simple sum normalization:

$$
\bar{\phi}_r = \frac{\phi_r}{\sum_{k=1}^{R} \phi_k}
$$

### Consequent

AdaTSK uses a first-order TSK consequent for both classification and
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

- `AdaTSKClassifier` and `AdaTSKRegressor` both use `SumBasedDefuzzifier`
  after the antecedent aggregation.
- `lambda_init` controls the initial Dombi shape parameter. It must be
  positive and is learned using softplus.
- The Composite Gaussian MF ensures membership values never fall to zero,
  improving numerical stability in high dimensions.
- `consequent_batch_norm=True` can be enabled to normalize inputs before
  consequent computation.

## Example

```python
from highfis import AdaTSKClassifierEstimator

clf = AdaTSKClassifierEstimator(
    n_mfs=3,
    mf_init="kmeans",
    lambda_init=1.0,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

For low-level use:

```python
from highfis import AdaTSKClassifier, GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}
model = AdaTSKClassifier(input_mfs, n_classes=2, lambda_init=1.0)
history = model.fit(x_train, y_train, epochs=100, learning_rate=1e-3)
```
