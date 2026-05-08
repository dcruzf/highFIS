# AdaTSK

AdaTSK extends TSK with an adaptive softmin antecedent that stabilizes high-dimensional fuzzy inference while preserving first-order TSK consequents.

## Reference

> G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive Neuro-Fuzzy System With Integrated Feature Selection and Rule Extraction for High-Dimensional Classification Problems," in IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181, July 2023, doi: [10.1109/TFUZZ.2022.3220950](https://doi.org/10.1109/TFUZZ.2022.3220950)

## Mathematical Formulation

AdaTSK extends TSK fuzzy inference by using an adaptive softmin antecedent
(Ada-softmin) together with first-order linear consequents.

### Antecedent

Each rule-term membership is typically computed with a Gaussian function:

$$
\mu_{r,d}(x_d)=\exp\left(-\frac{(x_d-c_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

where $c_{r,d}$ is the center and $\sigma_{r,d}>0$ is the spread.

In highFIS, the default estimator wrappers build standard Gaussian MFs.
The paper's proposed positive lower-bound variant can be instantiated with
`highfis.memberships.CompositeGaussianMF` when desired.

### Adaptive Ada-softmin aggregation

AdaTSK computes rule firing strengths with an adaptive softmin based on the
minimum antecedent membership for each rule:

$$
\hat{q}_r = \left\lceil \frac{690}{\ln (\min_{d}\mu_{r,d}(x_d))} \right\rceil,
\quad \hat{q}_r \in [-1000, -1]
$$

$$
\phi_r = \left( \frac{1}{D} \sum_{d=1}^{D}\mu_{r,d}(x_d)^{\hat{q}_r} \right)^{1/\hat{q}_r}
$$

The exponent $\hat{q}_r$ is recomputed on every forward pass and is clamped
for numerical stability, which avoids the fixed-parameter softmin problems
of underflow and fake minimum.

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

## Code ↔ Paper Correspondence

| Equation | Class / Method | Description |
|----------|----------------|-------------|
| Adaptive softmin | `highfis.layers.AdaSoftminRuleLayer` | Computes per-rule softmin exponents from the minimum membership value |
| Normalization | `highfis.defuzzifiers.SumBasedDefuzzifier` | Standard sum-based rule strength normalization |
| Consequent | `ClassificationConsequentLayer` / `RegressionConsequentLayer` | First-order linear consequents |
| Membership functions | `highfis.memberships.GaussianMF` | Default Gaussian antecedent MFs |
| Optional membership | `highfis.memberships.CompositeGaussianMF` | Optional positive lower-bound MF matching the paper variant |

## Implementation notes

In highFIS, `AdaTSKClassifier` and `AdaTSKRegressor` implement the core
AdaTSK model by replacing the standard product antecedent with the
adaptive softmin operator.

### Model classes

- `AdaTSKClassifier` and `AdaTSKRegressor` use
  `highfis.layers.AdaSoftminRuleLayer` to compute rule strengths.
- The TSK consequent remains first-order linear and is normalized with
  `highfis.defuzzifiers.SumBasedDefuzzifier`.
- `AdaTSKClassifier` and `AdaTSKRegressor` do not expose the feature-
  selection / rule-extraction gates of FSRE-AdaTSK.

### Estimator wrappers

- `AdaTSKClassifierEstimator` and `AdaTSKRegressorEstimator` are
  sklearn-compatible wrappers around the low-level AdaTSK model classes.
- They build Gaussian membership functions from `input_configs`, `n_mfs`,
  `mf_init`, and `sigma_scale`.
- The default `sigma_scale=1.0` is appropriate because the adaptive softmin
  operator handles high-dimensional stability.

### Membership functions

- The primary antecedent MFs are standard `highfis.memberships.GaussianMF`
  objects.
- An optional nonzero lower-bound membership function is available via
  `highfis.memberships.CompositeGaussianMF` for paper-style stability.

### Training in the paper vs. highFIS

- The paper trains AdaTSK end-to-end by optimizing the task loss through
  the adaptive softmin operator.
- highFIS follows the same gradient-based training paradigm in `BaseTSK.fit()`.
- `eps` is used to clamp membership values and stabilize log-space
  computations in `AdaSoftminRuleLayer`.

- FSRE-AdaTSK is documented separately in `docs/models/fsre-adatsk.md`.

## Alignment with the paper

- The paper's key AdaTSK contribution is the adaptive softmin antecedent
  operator to avoid numeric underflow and fake minimum effects.
- highFIS implements this via `AdaSoftminRuleLayer` with a per-rule exponent
  derived from the rule's minimum antecedent membership.
- The TSK consequent remains first-order, matching the paper's model.
- The default estimator wrappers use `GaussianMF`, while the paper's positive
  lower-bound MF can be supplied via `CompositeGaussianMF`.
