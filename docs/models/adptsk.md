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
MSE loss (with one-hot encoded targets in training). For regression, it uses
a regression consequent head and MSE loss.

## Code ↔ Paper Correspondence

| Paper concept | highFIS class / method | Description |
| --- | --- | --- |
| Gaussian PIMF | `highfis.memberships.GaussianPiMF` | Gaussian membership with positive infimum `exp(-K)` |
| ADP-softmin antecedent | `highfis.layers.ADPSoftminRuleLayer` | Adaptive double-parameter softmin rule aggregation |
| ADPTSK classifier | `highfis.models.ADPTSKClassifierModel` | Classifier model using ADP-softmin antecedents |
| ADPTSK regressor | `highfis.models.ADPTSKRegressorModel` | Regressor model using ADP-softmin antecedents |
| Estimator wrapper | `highfis.estimators.ADPTSKClassifier` | sklearn-style wrapper using `GaussianPiMF` |
| Estimator wrapper | `highfis.estimators.ADPTSKRegressor` | sklearn-style wrapper using `GaussianPiMF` |

## Implementation notes

- The model uses `GaussianPiMF` to ensure antecedent membership values have a
  nonzero positive lower bound.
- ADPTSK still builds on the BaseTSK pipeline, but replaces the rule layer
  with `ADPSoftminRuleLayer` instead of a vanilla product T-norm.
- Default hyperparameters mirror the paper:
  - three antecedent MFs per feature with centers `[0.0, 0.5, 1.0]`
    and `sigma=1.0`,
  - `rule_base="coco"` yielding a compact 3-rule structure,
  - `epochs=200`, `learning_rate=0.001`,
  - dynamic batch policy: full-batch when `N < 500`, else `0.2 * N`,
  - Adam optimizer,
  - zero initialization of consequent weights and biases,
  - `kappa=690.0`, `xi=730.0`,
  - `K=1.0` for the Gaussian PIMF lower bound.
- The estimator wrapper converts initialized `GaussianMF` objects to
  `GaussianPiMF` before model construction.
- Preprocessing remains external to the estimator by design:
  train/validation split and feature normalization should be handled by
  the user or by an sklearn pipeline.

## Model classes

### `highfis.models.ADPTSKClassifierModel`

- Uses `ADPSoftminRuleLayer` for antecedent aggregation.
- Uses `ClassificationConsequentLayer` and `MSELoss`.
- Uses Adam by default when no external optimizer is provided.
- Zero-initializes consequent parameters by default.

### `highfis.models.ADPTSKRegressorModel`

- Uses the same ADP-softmin antecedent.
- Uses `RegressionConsequentLayer` and `MSELoss`.
- Uses Adam by default when no external optimizer is provided.
- Zero-initializes consequent parameters by default.

## Estimator wrappers

### `highfis.estimators.ADPTSKClassifier`

This estimator:

- uses paper defaults (`mf_init="grid"`, `n_mfs=3`) to initialize
  antecedents as Gaussian MFs with centers `[0.0, 0.5, 1.0]` and `sigma=1.0`,
- wraps them as `GaussianPiMF` with the chosen `K` value,
- defaults to `rule_base="coco"`,
- applies paper-style batch sizing with the default `batch_size="auto"`,
- constructs `ADPTSKClassifierModel` with `kappa`, `xi`, `eps`, and
  zero consequent initialization. No automatic normalization is applied.

### `highfis.estimators.ADPTSKRegressor`

This estimator is analogous to the classifier wrapper but builds
`ADPTSKRegressorModel` for regression tasks.

## Membership functions

### `highfis.memberships.GaussianPiMF`

- Implements the PIMF version of Gaussian membership.
- Prevents crash caused by zero membership values in large dimensionality.
- `K` controls the lower bound: membership infimum is `exp(-K)`.

## Training in the paper vs. highFIS

- The paper optimizes ADPTSK end-to-end with gradient-based learning.
- highFIS follows the same paradigm: the estimator builds the model and
  uses Adam optimization within `BaseTSK.fit()`.
- `ADPTSKClassifier` and `ADPTSKRegressor` expose the
  same training hyperparameters as other highFIS estimators.
- As in the paper protocol, linear normalization to `[0, 1]` is expected;
  in highFIS this step is intentionally external to the model.

## Alignment with the paper

- ADPTSK in highFIS uses adaptive ADP-softmin aggregation, exactly as the
  paper describes.
- The implementation also uses Gaussian PIMF membership functions to ensure
  a positive antecedent infimum, matching the paper's stability motivation.
- Paper default initialization and training configuration are implemented as
  default estimator behavior.
- Recommended default values `kappa=690.0`, `xi=730.0`, and `K=1.0` are
  derived directly from the paper's suggested settings.
