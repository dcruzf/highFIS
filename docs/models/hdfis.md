# HDFIS

HDFIS (High-Dimensional Fuzzy Inference System) is a family of TSK fuzzy
models designed to solve very high-dimensional problems using two different
high-dimensional inference strategies: HDFIS-prod and HDFIS-min.

## Reference

> G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy Inference Systems," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024, doi: [10.1109/TSMC.2023.3311475](https://doi.org/10.1109/TSMC.2023.3311475).

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
| HDFIS-prod classifier | `highfis.models.HDFISProdClassifierModel` |
| HDFIS-prod regressor | `highfis.models.HDFISProdRegressorModel` |
| HDFIS-min classifier | `highfis.models.HDFISMinClassifierModel` |
| HDFIS-min regressor | `highfis.models.HDFISMinRegressorModel` |
| Estimator wrapper for HDFIS-prod classifier | `highfis.estimators.HDFISProdClassifier` |
| Estimator wrapper for HDFIS-prod regressor | `highfis.estimators.HDFISProdRegressor` |
| Estimator wrapper for HDFIS-min classifier | `highfis.estimators.HDFISMinClassifier` |
| Estimator wrapper for HDFIS-min regressor | `highfis.estimators.HDFISMinRegressor` |
| Product T-norm antecedent | `t_norm="prod"` in `HDFISProd*` models |
| Minimum T-norm antecedent | `t_norm="min"` in `HDFISMin*` models |
| Sum-based normalization | `highfis.defuzzifiers.SumBasedDefuzzifier` |

## Implementation notes

### Strict paper mode

highFIS provides an opt-in strict mode on HDFIS estimators:

- `paper_strict=True` in `HDFISProdClassifier`, `HDFISProdRegressor`,
  `HDFISMinClassifier`, and `HDFISMinRegressor`.
- In this mode, the estimator enforces the paper protocol defaults used in
  HDFIS_2023 experiments:
  - `mf_init="grid"`
  - `rule_base="coco"`
  - `n_mfs=3` (three rules)
  - `batch_size=64`
- For HDFIS-prod, strict mode also enables the paper-form DMF equation
  denominator (`D^rho + sigma^2`) and zero consequent initialization.
- For HDFIS-min, strict mode enables zero consequent initialization while
  keeping antecedent freezing.

When `paper_strict=False` (default), highFIS uses library-oriented defaults
for broader usability (for example, clustering-based initialization).

### Model classes

- `HDFISProdClassifierModel` and `HDFISProdRegressorModel` are concrete model classes
  that use product aggregation and dimension-dependent Gaussian membership
  functions for high-dimensional inference.
- `HDFISMinClassifierModel` and `HDFISMinRegressorModel` are concrete model classes
  that use minimum aggregation and freeze antecedent membership parameters.
- All HDFIS classes use first-order TSK consequents and
  `highfis.defuzzifiers.SumBasedDefuzzifier`.

### Estimator wrappers

- `HDFISProdClassifier` and `HDFISProdRegressor` are
  sklearn-like wrappers around `HDFISProd*` models.
- `HDFISMinClassifier` and `HDFISMinRegressor` are
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

Paper protocol is reproduced in highFIS through `paper_strict=True`.

## Alignment with the paper

- Core architecture alignment (always):
  - HDFIS-prod uses product aggregation + dimension-dependent Gaussian MFs.
  - HDFIS-min uses minimum aggregation + frozen antecedents.
  - Both use first-order TSK consequents and sum-based normalization.
- Experimental-protocol alignment (strict):
  - Use `paper_strict=True` to enforce paper defaults and strict DMF equation
    behavior.
- Non-strict defaults are practical library defaults and are not intended to
  be an exact replication of the paper protocol.

## Notes

- highFIS currently provides complete implementations of **HDFIS-prod** and
  **HDFIS-min**.
- HDFIS-min in highFIS uses frozen antecedents to avoid nondifferentiability
  and keep training focused on consequent parameters.
