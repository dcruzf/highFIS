# HDFIS

HDFIS (High-Dimensional Fuzzy Inference System) is a family of TSK fuzzy
models designed to solve very high-dimensional problems by avoiding numeric
underflow in product-based inference.

## Reference

> G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy Inference Systems," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024, doi: 10.1109/TSMC.2023.3311475.

## Mathematical formulation

### Antecedent

HDFIS-prod uses a dimension-dependent Gaussian membership function (DMF)
that increases the effective spread with input dimension. In highFIS, this
is implemented by `highfis.memberships.DimensionDependentGaussianMF`.

The DMF activation is:

$$
\mu_{r,d}(x_d) = \exp\left(-\frac{(x_d - m_{r,d})^2}{D^{\rho} + \sigma_{r,d}^2}\right)
$$

where:

- $D$ is the number of input features.
- $\rho = 1 - \frac{\ln(\xi)}{\ln(D)}$ by default.
- $\xi$ is a precision constant, fixed to `745.0` in the default estimator.
- $\sigma_{r,d}$ is the learnable spread parameter.

This dimension-dependent scaling keeps the product of membership values from
falling to numerical zero while preserving the interpretability of the rule
antecedents.

### Aggregation

HDFIS-prod retains the standard product T-norm for rule firing strengths:

$$
w_r(\mathbf{x}) = \prod_{d=1}^{D} \mu_{r,d}(x_d)
$$

Because the antecedent MF is dimension-scaled, the product remains usable in
high-dimensional settings.

### Defuzzification (sum-based normalization)

After antecedent aggregation, HDFIS-prod normalizes rule weights with the
standard sum-based defuzzifier:

$$
\bar{w}_r = \frac{w_r}{\sum_{i=1}^{R} w_i}
$$

### Consequent (first-order)

HDFIS-prod uses first-order TSK consequents for both classification and
regression.

For classification:

$$
\mathbf{y} = \sum_{r=1}^{R} \bar{w}_r \mathbf{y}_r,
\qquad
\mathbf{y}_r = W_r \mathbf{x} + \mathbf{b}_r.
$$

For regression:

$$
\hat{y} = \sum_{r=1}^{R} \bar{w}_r \hat{y}_r,
\qquad
\hat{y}_r = \mathbf{w}_r^\top \mathbf{x} + b_r.
$$

## HDFIS-min

HDFIS-min uses the minimum T-norm for antecedent aggregation and only
optimizes the consequent parameters. Because the minimum T-norm is
nondifferentiable with respect to antecedent membership parameters, highFIS
implements HDFIS-min by freezing antecedent MFs and training only the
consequent layers.

In highFIS, HDFIS-min may be built from either standard Gaussian MFs or
dimension-dependent Gaussian MFs. The estimator wrappers currently construct
HDFIS-min with standard `GaussianMF` initialization by default, which is
consistent with the paper's design principle that the antecedents are fixed
and only consequents are optimized.

The model classes are:

- `highfis.models.HDFISMinClassifier`
- `highfis.models.HDFISMinRegressor`

The estimator wrappers are:

- `highfis.estimators.HDFISMinClassifierEstimator`
- `highfis.estimators.HDFISMinRegressorEstimator`

## Code ↔ paper correspondence

| Paper concept | highFIS implementation |
|---|---|
| Dimension-dependent Gaussian MF | `highfis.memberships.DimensionDependentGaussianMF` |
| HDFIS-prod classifier | `highfis.models.HDFISProdClassifier` |
| HDFIS-prod regressor | `highfis.models.HDFISProdRegressor` |
| Estimator wrapper for HDFIS-prod classifier | `highfis.estimators.HDFISProdClassifierEstimator` |
| Estimator wrapper for HDFIS-prod regressor | `highfis.estimators.HDFISProdRegressorEstimator` |
| Product T-norm antecedent | `t_norm="prod"` in `HDFISProd*` models |
| Sum-based normalization | `highfis.defuzzifiers.SumBasedDefuzzifier` |

## Implementation notes

### Model classes

- `HDFISProdClassifier` and `HDFISProdRegressor` are concrete model classes
  built on `BaseTSKClassifier` and `BaseTSKRegressor`.
- The models default to `t_norm="prod"` and use `SumBasedDefuzzifier` for
  output normalization.

### Estimator wrappers

- `HDFISProdClassifierEstimator` and `HDFISProdRegressorEstimator` are
  sklearn-like wrappers around the low-level HDFIS-prod model classes.
- They support the standard highFIS estimator training hyperparameters,
  including `epochs`, `learning_rate`, `batch_size`, `shuffle`, `patience`,
  `restore_best`, `validation_data`, `ur_weight`, and `weight_decay`.
- The estimators convert built-in Gaussian membership functions into
  `DimensionDependentGaussianMF` objects based on the training input dimension.

### Membership functions

- The HDFIS-prod paper uses a dimension-dependent Gaussian MF to avoid the
  numeric underflow caused by the product T-norm in high dimensions.
- highFIS implements this with `DimensionDependentGaussianMF`, which
  preserves the Gaussian shape while scaling the spread by $D^{\rho}$.

### Training in the paper vs. highFIS

- The paper considers end-to-end TSK training with antecedent and consequent
  parameters, but observes that antecedent updates are small for HDFIS-prod.
- highFIS follows a similar gradient-based training paradigm using `BaseTSK.fit()`.
- `BaseTSK.fit()` supports mini-batch optimization, optional early stopping,
  and regularization through `ur_weight` and `weight_decay`.

## Alignment with the paper

- highFIS implements HDFIS-prod by combining a product T-norm antecedent with
  dimension-scaled Gaussian membership spreads.
- `DimensionDependentGaussianMF` directly realizes the paper's DMF concept.
- The model preserves first-order TSK consequents and sum-based weight
  normalization, matching the paper's HDFIS-prod architecture.

## Notes

- highFIS currently provides complete implementations of **HDFIS-prod** and **HDFIS-min**.
- HDFIS-min uses a minimum T-norm antecedent and freezes antecedent membership
  parameters so that only consequent layers are optimized.
