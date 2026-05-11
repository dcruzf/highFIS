# HDFIS

HDFIS (High-Dimensional Fuzzy Inference System) is a family of TSK fuzzy
models designed to solve very high-dimensional problems using two different
high-dimensional inference strategies: HDFIS-prod and HDFIS-min.

## Reference

> G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy Inference Systems," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024, doi: 10.1109/TSMC.2023.3311475.

## Mathematical formulation

### HDFIS-prod

HDFIS-prod avoids numeric underflow in high-dimensional product-based
antecedent aggregation by using a dimension-dependent Gaussian membership
function (DMF).

The DMF activation in highFIS is implemented by
`highfis.memberships.DimensionDependentGaussianMF`:

$$
\mu_{r,d}(x_d) = \exp\left(-\frac{(x_d - m_{r,d})^2}{D^{\rho} + \sigma_{r,d}^2}\right)
$$

where:

- $D$ is the number of input features.
- $\rho = 1 - \frac{\ln(\xi)}{\ln(D)}$ by default.
- $\xi$ is a precision constant, fixed to `745.0` in the default estimator.
- $\sigma_{r,d}$ is the learnable spread parameter.

The membership scale adapts to $D$ so that the product of membership values
remains numerically stable for high-dimensional inputs.

#### Aggregation

HDFIS-prod uses the standard product T-norm:

$$
w_r(\mathbf{x}) = \prod_{d=1}^{D} \mu_{r,d}(x_d)
$$

#### Defuzzification

Rule weights are normalized using sum-based normalization:

$$
\bar{w}_r = \frac{w_r}{\sum_{i=1}^{R} w_i}
$$

#### Consequent (first-order)

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

### HDFIS-min

HDFIS-min uses the minimum T-norm for antecedent aggregation and trains
only consequent parameters. Because the minimum operator is nondifferentiable
with respect to antecedent membership parameters, highFIS freezes the
antecedent MFs and optimizes the consequent layers alone.

#### Antecedent

The minimum T-norm firing strength is:

$$
w_r(\mathbf{x}) = \min_{d=1,\ldots,D} \mu_{r,d}(x_d)
$$

HighFIS-min can be constructed from standard Gaussian MFs or from
`DimensionDependentGaussianMF` objects, but the default estimator uses
standard Gaussian MFs and then freezes the antecedent parameters.

#### Defuzzification

Normalized rule weights are computed with the standard sum-based defuzzifier:

$$
\bar{w}_r = \frac{w_r}{\sum_{i=1}^{R} w_i}
$$

#### Consequent (first-order)

HDFIS-min uses the same first-order TSK consequent form as HDFIS-prod:

$$
\mathbf{y} = \sum_{r=1}^{R} \bar{w}_r \mathbf{y}_r
\quad\text{or}\quad
\hat{y} = \sum_{r=1}^{R} \bar{w}_r \hat{y}_r.
$$

## Code ↔ paper correspondence

| Paper concept | highFIS implementation |
|---|---|
| Dimension-dependent Gaussian MF | `highfis.memberships.DimensionDependentGaussianMF` |
| HDFIS-prod classifier | `highfis.models.HDFISProdClassifier` |
| HDFIS-prod regressor | `highfis.models.HDFISProdRegressor` |
| HDFIS-min classifier | `highfis.models.HDFISMinClassifier` |
| HDFIS-min regressor | `highfis.models.HDFISMinRegressor` |
| Estimator wrapper for HDFIS-prod classifier | `highfis.estimators.HDFISProdClassifierEstimator` |
| Estimator wrapper for HDFIS-prod regressor | `highfis.estimators.HDFISProdRegressorEstimator` |
| Estimator wrapper for HDFIS-min classifier | `highfis.estimators.HDFISMinClassifierEstimator` |
| Estimator wrapper for HDFIS-min regressor | `highfis.estimators.HDFISMinRegressorEstimator` |
| Product T-norm antecedent | `t_norm="prod"` in `HDFISProd*` models |
| Minimum T-norm antecedent | `t_norm="min"` in `HDFISMin*` models |
| Sum-based normalization | `highfis.defuzzifiers.SumBasedDefuzzifier` |

## Implementation notes

### Model classes

- `HDFISProdClassifier` and `HDFISProdRegressor` are concrete model classes
  that use product aggregation and dimension-dependent Gaussian membership
  functions for high-dimensional inference.
- `HDFISMinClassifier` and `HDFISMinRegressor` are concrete model classes
  that use minimum aggregation and freeze antecedent membership parameters.
- All HDFIS classes use first-order TSK consequents and
  `highfis.defuzzifiers.SumBasedDefuzzifier`.

### Estimator wrappers

- `HDFISProdClassifierEstimator` and `HDFISProdRegressorEstimator` are
  sklearn-like wrappers around `HDFISProd*` models.
- `HDFISMinClassifierEstimator` and `HDFISMinRegressorEstimator` are
  sklearn-like wrappers around `HDFISMin*` models.
- The estimators expose standard training hyperparameters such as `epochs`,
  `learning_rate`, `batch_size`, `shuffle`, `patience`, `restore_best`,
  `validation_data`, `ur_weight`, and `weight_decay`.
- HDFIS-prod estimators build dimension-dependent Gaussian MFs from the
  training input dimension.
- HDFIS-min estimators build antecedent MFs with the chosen initialization
  strategy, then freeze them before training consequents.

### Membership functions

- The HDFIS-prod paper uses a dimension-dependent Gaussian MF to avoid
  numeric underflow when the product T-norm is applied to high-dimensional
  inputs.
- highFIS implements this concept as `DimensionDependentGaussianMF`.
- HDFIS-min preserves the minimum T-norm behavior by freezing antecedent
  membership parameters and optimizing only the consequent layers.

### Training in the paper vs. highFIS

- HDFIS-prod is described in the paper as a product-based high-dimensional
  TSK model with adaptive membership spread scaling; highFIS follows this
  design using dimension-dependent Gaussian MFs and end-to-end gradient
  optimization.
- HDFIS-min is described in the paper as a minimum T-norm model where only
  consequents are optimized; highFIS implements this by freezing antecedent
  parameters in `HDFISMin*` classes.
- `BaseTSK.fit()` supports mini-batch optimization, optional early stopping,
  uniform rule regularization, and weight decay across HDFIS estimators.

## Alignment with the paper

- highFIS implements the HDFIS-prod architecture with product antecedent
  aggregation, dimension-dependent Gaussian membership functions, and
  sum-based normalization.
- highFIS implements the HDFIS-min architecture with minimum antecedent
  aggregation and frozen antecedent membership parameters, optimizing only
  first-order consequents.
- Both HDFIS variants preserve the paper's first-order TSK consequent
  structure and sum-based rule normalization.

## Notes

- highFIS currently provides complete implementations of **HDFIS-prod** and
  **HDFIS-min**.
- HDFIS-min in highFIS uses frozen antecedents to avoid nondifferentiability
  and keep training focused on consequent parameters.
