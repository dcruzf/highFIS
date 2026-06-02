# ADATSK

ADATSK extends TSK with an adaptive softmin antecedent that stabilizes high-dimensional fuzzy inference while preserving first-order TSK consequents.

## Reference

> G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive Neuro-Fuzzy System With Integrated Feature Selection and Rule Extraction for High-Dimensional Classification Problems," in IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181, July 2023, doi: [10.1109/TFUZZ.2022.3220950](https://doi.org/10.1109/TFUZZ.2022.3220950)

## Mathematical Formulation

ADATSK extends TSK fuzzy inference by using an adaptive softmin antecedent
(Ada-softmin) together with first-order linear consequents.

### Antecedent

Each rule-term membership is typically computed with a Gaussian function:

$$
\mu_{r,d}(x_d)=\exp\left(-\frac{(x_d-c_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

where $c_{r,d}$ is the center and $\sigma_{r,d}>0$ is the spread.

In highFIS, the paper-strict `ADATSKClassifier` path uses a dedicated
`ADATSKGaussianMF` antecedent, with:

$$
\mu(x)=\exp\left(-\frac{(x-c)^2}{\sigma^2}\right)
$$

and applies `sigma=1` before training, yielding the paper's simplified
form:

$$
\mu(x)=\exp\left(-(x-c)^2\right)
$$

### Adaptive Ada-softmin aggregation

ADATSK computes rule firing strengths with an adaptive softmin based on the
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

In the ADATSK paper, the base formulation is presented for classification,
with first-order TSK consequents:

$$
\mathbf{y}_r = W_r \mathbf{x} + \mathbf{b}_r
$$

### Output aggregation

The final prediction is the normalized weighted sum of rule consequents:

For classification:

$$
\mathbf{y} = \sum_{r=1}^{R} \bar{\phi}_r \mathbf{y}_r
$$

## Code ↔ Paper Correspondence

| Equation | Class / Method | Description |
|----------|----------------|-------------|
| Adaptive softmin | `highfis.layers.AdaSoftminRuleLayer` | Computes per-rule softmin exponents from the minimum membership value |
| Normalization | `highfis.defuzzifiers.SumBasedDefuzzifier` | Standard sum-based rule strength normalization |
| Consequent | `ClassificationConsequentLayer` | First-order linear consequents for classification |
| Membership functions | `highfis.memberships.ADATSKGaussianMF` | Paper-style Gaussian antecedent used by `ADATSKClassifier` |

## Implementation notes

In highFIS, `ADATSKClassifierModel` implements the paper-aligned ADATSK core
by replacing the standard product antecedent with the adaptive softmin
operator.

### Model classes

- `ADATSKClassifierModel` uses `highfis.layers.AdaSoftminRuleLayer` to
  compute rule strengths.
- The classifier consequent remains first-order linear and is normalized
  with `highfis.defuzzifiers.SumBasedDefuzzifier`.
- `ADATSKClassifierModel` does not expose the feature-selection/
  rule-extraction gates of FSRE-ADATSK.

### Estimator wrappers

- `ADATSKClassifier` is the paper-strict default wrapper.
- Default settings follow the paper protocol: `n_mfs=3`, `mf_init="grid"`,
  `rule_base="coco"`, full-batch updates (`batch_size=None`), and no
  shuffling.
- For grid initialization, MF centers are placed directly on
  `[V_{\min}, V_{\max}]` with no margin padding.
- In the pre-train hook, Gaussian spreads are set to `sigma=1` and frozen.
- Consequent parameters are initialized to zero in the ADATSK classifier
  paper-strict path.
- For high-dimensional inputs (default threshold `1000` features), antecedent
  parameters are frozen by default to match the paper's experimental protocol.

### Membership functions

- The primary antecedent MFs are `highfis.memberships.ADATSKGaussianMF`
  objects in the paper-strict classifier path.
- Gaussian spreads are fixed at `sigma=1` in the paper-strict ADATSK path,
  reproducing Eq. (3).

### Training in the paper vs. highFIS

- The paper trains ADATSK end-to-end using full-batch gradient descent and
  MSE-style classification error.
- The paper-strict ADATSK default in highFIS uses `nn.MSELoss()` for the
  classifier and SGD-based full-batch optimization.
- `eps` is used to clamp membership values and stabilize log-space
  computations in `AdaSoftminRuleLayer`.

- FSRE-ADATSK is documented separately in `docs/models/fsre-adatsk.md`.

## Framework extensions (outside paper-strict scope)

- `ADATSKRegressorModel` and `ADATSKRegressor` are provided as framework
  extensions for regression workflows.
- `highfis.memberships.CompositeGaussianMF` remains available as an
  engineering alternative for custom experiments.

## Alignment with the paper

- The paper's key ADATSK contribution is the adaptive softmin antecedent
  operator to avoid numeric underflow and fake minimum effects.
- highFIS implements this via `AdaSoftminRuleLayer` with a per-rule exponent
  derived from the rule's minimum antecedent membership.
- The default `ADATSKClassifier` now follows the paper protocol (CoCo rule
  base, Eq. (3)-style Gaussian antecedent with `sigma=1` fixed, full-batch
  GD, MSE-style classification loss, and zero-initialized consequents).
- Regression and alternative MF variants are treated as explicit framework
  extensions, not part of the strict ADATSK paper baseline.
