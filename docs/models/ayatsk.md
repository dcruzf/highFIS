# AYATSK

AYATSK extends TSK by using an adaptive Yager T-norm aggregation and optional positive lower-bound membership functions to improve stability and performance in high-dimensional settings.

## Reference

> G. Xue, Y. Yang and J. Wang, "Adaptive Yager T-Norm-Based Takagi–Sugeno–Kang Fuzzy Systems," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 55, no. 12, pp. 9802-9815, Dec. 2025, doi: 10.1109/TSMC.2025.3621346.

## Mathematical Formulation

### Antecedent

AYATSK uses Gaussian or composite exponential membership functions for each input feature:

$$
\mu_{r,d}(x_d) = \exp\left(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

The paper additionally recommends a Composite Exponential MF (CEMF) with a positive lower bound:

$$
\mu_{r,d}(x_d) = K^{-1 + \exp\left(-\frac{(x_d - c)^2}{2\sigma^2}\right)}
$$

This lower bound is important because AYATSK derives the adaptive Yager index from a fixed minimum membership value.

### Adaptive Yager T-norm

Rule firing strengths are computed with a parameterized Yager T-norm:

$$
w_r = \max\left(0,\; 1 - \left[\sum_{d=1}^{D} (1 - \mu_{r,d}(x_d))^{p_r}\right]^{1/p_r}\right)
$$

Each rule has its own learnable exponent $p_r > 0$. In highFIS, this is implemented by `t_norm="yager"` together with `YagerTNorm`.

The paper’s adaptive strategy sets $p_r$ based on input dimensionality $D$ and a lower bound $\varepsilon$ on the membership values. This keeps the Yager aggregation numerically stable in high dimensions while preserving T-norm properties.

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
| Adaptive Yager T-norm | `AYATSKClassifier`, `AYATSKRegressor` | Default `t_norm="yager"` in highFIS |
| Yager exponent $p_r$ | `YagerTNorm` | Learnable per-rule exponent controlling softmin behavior |
| Sum-based defuzzification | `SumBasedDefuzzifier` | Normalizes $\bar{f}_r$ across rules |
| Composite Exponential MF | `CompositeExponentialMF` | Provides a positive lower bound needed by the adaptive strategy |
| Estimator wrapper | `AYATSKClassifierEstimator`, `AYATSKRegressorEstimator` | Builds input MFs and handles training/hyperparameters |

## Implementation notes

- `AYATSKClassifier` and `AYATSKRegressor` default to `t_norm="yager"`.
- The adaptive Yager index is compatible with standard Gaussian MFs, but the paper’s recommended CEMF ensures a positive lower bound and more stable high-dimensional behavior.
- `rule_base` is typically `"coco"` for k-means initialization and `"cartesian"` for grid initialization.
- `AYATSKClassifierEstimator` and `AYATSKRegressorEstimator` follow the standard highFIS estimator pattern and expose the same fitting parameters as other estimators.
- The model is trained end-to-end with backpropagation; `BaseTSK.fit()` uses mini-batch Adam optimization with optional early stopping.

## Estimator wrappers

- `AYATSKClassifierEstimator` wraps `AYATSKClassifier` and supports classification with adaptive Yager aggregation.
- `AYATSKRegressorEstimator` wraps `AYATSKRegressor` for regression tasks.
- Estimators accept the usual hyperparameters: `n_mfs`, `mf_init`, `sigma_scale`, `random_state`, `epochs`, `learning_rate`, `batch_size`, `shuffle`, `validation_data`, and `patience`.
- `pfrb_max_rules` exists on the shared estimator base but is unused by AYATSK.

## Alignment with the paper

- The paper defines AYATSK through an adaptive Yager T-norm that avoids product underflow in high dimensions.
- highFIS implements the same core idea with `YagerTNorm` and first-order TSK consequents.
- `CompositeExponentialMF` is available in highFIS to reflect the paper’s use of a positive lower bound membership function.
- The estimator wrappers preserve the paper’s training regime of gradient-based optimization over antecedent parameters, Yager exponents, and consequent weights.
