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

### Fidelity to the source article

The defaults of the four HDFIS estimators are the settings of the
article, and on the same random splits they give the accuracies of the authors' code
(see the [reproduction](../reproductions/hdfis.md)).

| Point | Article | highFIS |
|---|---|---|
| Fuzzy sets | Partition, not clustering: centres at $(r-1)/(R-1)$ on inputs in $[0, 1]$, spreads of 1 | `mf_init="grid"` (default): centres evenly spaced over the range of each feature, spreads of 1 |
| Rules | Three, one per fuzzy set | `n_mfs=3`, compactly combined rule base (defaults) |
| Membership of HDFIS-prod | $\exp\big(-(x-m)^2 / (D^{\tilde\rho} + \sigma^2)\big)$, $\xi = 745$ | `DimensionDependentGaussianMF`, `xi=745.0` |
| Firing strength of HDFIS-prod | Product, in double precision | Product computed in the logarithmic domain, exact in single and double precision |
| HDFIS-min | Minimum T-norm, antecedents fixed, consequents trained | Same |
| Consequents | Start at zero | Same |
| Loss | Squared error on one-hot targets, summed over the classes and halved (Eq. 14) | `highfis.losses.HalfSumSquaredErrorLoss` |
| Optimizer | Adam, batches of 64, 100 epochs | AdamW with a weight decay of `1e-8`, `batch_size="auto"` (64), `epochs=100` |

Remaining differences:

- **Learning rate.** The article does not state it. The authors' code uses 0.001;
  the default here is 0.01, which gives the same or a better accuracy on the datasets
  tried and is needed on small low-dimensional data (with 0.001, 100 epochs are too few
  on Iris).
- **Precision.** The bound $\xi = 745$ of the article is the limit of double precision.
  highFIS trains in single precision by default and obtains the normalized product from
  the geometric mean of the membership degrees, $\operatorname{softmax}(D \log \bar\mu)$,
  which is the same quantity without underflow. The plain product underflows in
  single precision on high-dimensional data, and every rule then receives the same
  weight.
- **Membership of HDFIS-min.** The article reports HDFIS-min with the conventional and
  with the dimension-dependent membership function; highFIS implements the conventional
  one.
- **Consequents by least squares.** The article also reports both models with the
  consequents estimated by least squares; highFIS trains them by gradient.
- **Low-dimensional data.** The article designs HDFIS for more than 1000 features.
  The estimators work on fewer, but a family for low-dimensional data is a better
  choice there. `mf_init="kmeans"` remains available.

### Model classes

- `HDFISProdClassifierModel` and `HDFISProdRegressorModel` are concrete model classes
  that use product aggregation and dimension-dependent Gaussian membership
  functions for high-dimensional inference.
- `HDFISMinClassifierModel` and `HDFISMinRegressorModel` are concrete model classes
  that use minimum aggregation and freeze antecedent membership parameters.
- All HDFIS classes use first-order TSK consequents. HDFIS-min normalizes with
  `highfis.defuzzifiers.SumBasedDefuzzifier`; HDFIS-prod uses the geometric mean and
  `SoftmaxLogDefuzzifier(scale=D)`, which is the normalized product.

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

## Alignment with the paper

- HDFIS-prod uses product aggregation with dimension-dependent Gaussian MFs.
- HDFIS-min uses minimum aggregation with frozen antecedents.
- Both use first-order TSK consequents and normalized firing strengths.
- The defaults are the experimental settings of the article; see "Fidelity to the
  source article" above.

## Notes

- highFIS currently provides complete implementations of **HDFIS-prod** and
  **HDFIS-min**.
- HDFIS-min in highFIS uses frozen antecedents to avoid nondifferentiability
  and keep training focused on consequent parameters.
- **Loss function**: both classifiers default to `HalfSumSquaredErrorLoss` on one-hot
  targets, Eq. (14) of the article; regression uses `MSELoss` on scalar targets.
