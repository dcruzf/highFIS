# ADPTSK

ADPTSK is a high-dimensional Takagi-Sugeno-Kang fuzzy system built on two key ideas from Ma et al. (2025):
- adaptive double-parameter softmin (ADP-softmin) for antecedent aggregation,
- Gaussian PIMF membership functions with a positive infimum to avoid underflow.

## Reference

> Ma, M., Qian, L., Zhang, Y., Fang, Q., & Xue, G. (2025). An adaptive double-parameter softmin based Takagi-Sugeno-Kang fuzzy system for high-dimensional data. Fuzzy Sets and Systems, 521, 109582, doi: [10.1016/j.fss.2025.109582](https://doi.org/10.1016/j.fss.2025.109582).

## Mathematical Formulation

### Antecedent

ADPTSK replaces the standard product or min T-norm with an adaptive
softmin operator. For rule $r$ and feature $d$, membership values are
computed using the Gaussian PIMF:

$$
\mu_{r,d}(x_d) = \exp\left(-K\left[1 - \exp\left(-\frac{(x_d - m_{r,d})^{2}}{2\sigma_{r,d}^{2}}\right)\right]\right)
$$

The infimum of this membership is $\exp(-K) > 0$, which prevents the
antecedent from producing zero values in high-dimensional settings.

For each rule $r$, ADP-softmin computes a firing coefficient using
adaptive parameters $\eta$ and $q$:

$$
f_r = \left(\frac{1}{D} \sum_{d=1}^{D} (\eta\,\mu_{r,d})^{q}\right)^{1/q}
$$

The parameters are chosen to avoid numeric underflow and fake-minimum
behaviour:

- $\eta$ adapts to the current rule's minimum and maximum membership values,
- $q$ is selected as the tightest negative exponent that still remains
  numerically representable.

In practice, highFIS implements the ADP-softmin computation in a numerically
stable log-space fashion, with recommended default values
$\kappa = 690.0$ and $\xi = 730.0$.

### Consequent and defuzzification

The consequent structure remains first-order TSK with a weighted linear output.
The rule strengths are normalized and passed to the standard TSK consequent
layer.

For classification, the model uses a classification consequent head and
cross-entropy loss. For regression, it uses a regression consequent head and
MSE loss.

## Code ↔ Paper Correspondence

| Paper concept | highFIS class / method | Description |
| --- | --- | --- |
| Gaussian PIMF | `highfis.memberships.GaussianPIMF` | Gaussian membership with positive infimum `exp(-K)` |
| ADP-softmin antecedent | `highfis.layers.ADPSoftminRuleLayer` | Adaptive double-parameter softmin rule aggregation |
| ADPTSK classifier | `highfis.models.ADPTSKClassifier` | Classifier model using ADP-softmin antecedents |
| ADPTSK regressor | `highfis.models.ADPTSKRegressor` | Regressor model using ADP-softmin antecedents |
| Estimator wrapper | `highfis.estimators.ADPTSKClassifierEstimator` | sklearn-style wrapper using `GaussianPIMF` |
| Estimator wrapper | `highfis.estimators.ADPTSKRegressorEstimator` | sklearn-style wrapper using `GaussianPIMF` |

## Implementation notes

- The model uses `GaussianPIMF` to ensure antecedent membership values have a
  nonzero positive lower bound.
- ADPTSK still builds on the BaseTSK pipeline, but replaces the rule layer
  with `ADPSoftminRuleLayer` instead of a vanilla product T-norm.
- Default hyperparameters mirror the paper:
  - `n_mfs=3` for compact CoCo rule bases,
  - `mf_init="kmeans"` with `sigma_scale=1.0`,
  - `kappa=690.0`, `xi=730.0`,
  - `K=1.0` for the Gaussian PIMF lower bound.
- The estimator wrapper converts initialized `GaussianMF` objects to
  `GaussianPIMF` before model construction.

## Model classes

### `highfis.models.ADPTSKClassifier`

- Uses `ADPSoftminRuleLayer` for antecedent aggregation.
- Defaults to `rule_base="coco"` when `mf_init="kmeans"`.
- Uses `ClassificationConsequentLayer` and `CrossEntropyLoss`.

### `highfis.models.ADPTSKRegressor`

- Uses the same ADP-softmin antecedent.
- Uses `RegressionConsequentLayer` and `MSELoss`.

## Estimator wrappers

### `highfis.estimators.ADPTSKClassifierEstimator`

This estimator:

- builds Gaussian membership functions via `mf_init` and `input_configs`,
- wraps them as `GaussianPIMF` with the chosen `K` value,
- constructs `ADPTSKClassifier` with `kappa`, `xi`, and `eps`.

### `highfis.estimators.ADPTSKRegressorEstimator`

This estimator is analogous to the classifier wrapper but builds
`ADPTSKRegressor` for regression tasks.

## Membership functions

### `highfis.memberships.GaussianPIMF`

- Implements the PIMF version of Gaussian membership.
- Prevents crash caused by zero membership values in large dimensionality.
- `K` controls the lower bound: membership infimum is `exp(-K)`.

## Training in the paper vs. highFIS

- The paper optimizes ADPTSK end-to-end with gradient-based learning.
- highFIS follows the same paradigm: the estimator builds the model and
  uses Adam-style optimization within `BaseTSK.fit()`.
- `ADPTSKClassifierEstimator` and `ADPTSKRegressorEstimator` expose the
  same training hyperparameters as other highFIS estimators.

## Alignment with the paper

- ADPTSK in highFIS uses adaptive ADP-softmin aggregation, exactly as the
  paper describes.
- The implementation also uses Gaussian PIMF membership functions to ensure
  a positive antecedent infimum, matching the paper's stability motivation.
- Recommended default values `kappa=690.0`, `xi=730.0`, and `K=1.0` are
  derived directly from the paper's suggested settings.
