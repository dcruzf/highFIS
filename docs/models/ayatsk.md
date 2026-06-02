# AYATSK

AYATSK is a classification-focused TSK model from the 2025 paper by Xue, Yang, and Wang. In highFIS, `AYATSKClassifier(paper_strict=True)` enforces a paper-style classifier configuration with adaptive Yager T-norm, Composite Exponential membership functions (CEMF), and CoCo rule base defaults.

## Reference

> G. Xue, Y. Yang and J. Wang, "Adaptive Yager T-Norm-Based Takagi–Sugeno–Kang Fuzzy Systems," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 55, no. 12, pp. 9802-9815, Dec. 2025, doi: [10.1109/TSMC.2025.3621346](https://doi.org/10.1109/TSMC.2025.3621346).

## Mathematical Formulation

### Antecedent

The paper uses Composite Exponential membership functions (CEMF) for each input feature:

$$
\mu_{r,d}(x_d) = K^{-1 + \exp\left(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}\right)}
$$

The lower bound is $1/K > 0$, which is important because AYATSK derives the adaptive Yager index from a fixed lower bound rather than from a sample-dependent minimum.

### Adaptive Yager T-norm

Rule firing strengths are computed with a parameterized Yager T-norm:

$$
w_r = 1 - \left[\sum_{d=1}^{D} (1 - \mu_{r,d}(x_d))^{\lambda}\right]^{1/\lambda}
$$

In the paper, $\lambda$ is computed once from the input dimensionality $D$ and the membership lower bound $\varepsilon$:

$$
\lambda = -\frac{\ln D}{\ln(1 - \varepsilon)}
$$

highFIS follows that policy by deriving $\lambda$ from the number of input features and the CEMF lower bound in AYATSK models.

### Defuzzification

AYATSK uses a sum-based defuzzifier to normalize rule activations:

$$
\bar{f}_r = \frac{w_r}{\sum_{i=1}^{R} w_i}
$$

This is implemented in highFIS by `SumBasedDefuzzifier`.

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

## Code ↔ Paper Correspondence

| Concept | Class / Method | Notes |
|---|---|---|
| Adaptive Yager T-norm | `AYATSKClassifierModel`, `AYATSKRegressorModel` | Default AYATSK path computes `lambda` from dimension and lower bound |
| Yager exponent $\lambda$ | `YagerTNorm` | Fixed after dataset-level adaptation; not learnable per rule |
| Sum-based defuzzification | `SumBasedDefuzzifier` | Normalizes $\bar{f}_r$ across rules |
| Composite Exponential MF | `CompositeExponentialMF` | Provides a positive lower bound needed by the adaptive strategy |
| Estimator wrapper | `AYATSKClassifier`, `AYATSKRegressor` | Builds input MFs and handles training/hyperparameters |

## Implementation notes

- `AYATSKClassifierModel` uses MSE loss, Adam, and zero-initialized consequents by default.
- `AYATSKClassifier` defaults to `n_mfs=3`, `mf_init="grid"`, `rule_base="coco"`, `epochs=200`, and `learning_rate=0.001`.
- `AYATSKClassifier` and `AYATSKRegressor` expose `k` (CEMF parameter), with required constraint `k > 1`.
- When `batch_size=None`, the AYATSK estimator uses full-batch for `N < 500` and `0.1 * N` otherwise, matching the paper’s training policy.
- In `paper_strict=True`, `AYATSKClassifier` enforces paper defaults and emits a warning when mini-batch behavior may diverge from the paper’s high-dimensional experimental setting.
- `AYATSKRegressor` remains available as a framework extension, but the paper itself evaluates classification only.

## Estimator wrappers

- `AYATSKClassifier` wraps `AYATSKClassifierModel` and supports classification with paper-style adaptive Yager aggregation.
- `paper_strict=True` is classifier-only in the current implementation.
- `AYATSKRegressor` wraps `AYATSKRegressorModel` for regression tasks, but this is outside the paper’s evaluated scope.
- Estimators accept the usual hyperparameters: `n_mfs`, `mf_init`, `sigma_scale`, `random_state`, `epochs`, `learning_rate`, `batch_size`, `shuffle`, `validation_data`, and `patience`.
- `pfrb_max_rules` exists on the shared estimator base but is unused by AYATSK.

## Alignment with the paper

- The paper evaluates AYATSK on classification datasets only.
- highFIS implements the paper-strict classification path with `CompositeExponentialMF`, dataset-level adaptive `YagerTNorm`, MSE loss, Adam, and zero-initialized consequents.
- The regressor remains supported as a framework extension, but it should not be presented as part of the paper’s experimental validation.
