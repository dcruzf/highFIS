# DombiTSK

DombiTSK replaces product aggregation with an adaptive Dombi t-norm in the antecedent, providing more flexible high-dimensional rule activation behavior.

## Reference

> G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: A High-Dimensional Takagi–Sugeno–Kang Fuzzy System Based on Adaptive Dombi T-Norm," in IEEE Transactions on Fuzzy Systems, vol. 33, no. 6, pp. 1767-1780, June 2025, doi: [10.1109/TFUZZ.2025.3535640](https://doi.org/10.1109/TFUZZ.2025.3535640).

## Mathematical Formulation

### Antecedent

Each rule-term membership is computed with a Gaussian function:

$$
\mu_{r,d}(x_d)=\exp\left(-\frac{(x_d-c_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

where $c_{r,d}$ is the center and $\sigma_{r,d}>0$ is the spread.

### Dombi aggregation

DombiTSK uses the Dombi t-norm to combine the antecedent membership values:

$$
\phi_r = \frac{1}{1 + \left(\sum_{d=1}^{D}
    \left(\frac{1 - \mu_{r,d}}{\mu_{r,d}}\right)^{\lambda}\right)^{1/\lambda}}
$$

The hyperparameter $\lambda > 0$ controls the aggregation shape:

- small $\lambda$ makes the aggregation softer and closer to product-like behavior,
- large $\lambda$ makes the aggregation sharper and closer to minimum-like behavior.

### Normalization

Rule firing strengths are normalized by sum normalization:

$$
\bar{\phi}_r = \frac{\phi_r}{\sum_{k=1}^{R} \phi_k}
$$

### Consequent

DombiTSK uses a first-order TSK consequent for both classification and
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
| Dombi aggregation | `DombiTSKClassifierModel` / `DombiTSKRegressorModel` | Fixed-λ Dombi antecedent aggregation with `t_norm="dombi"` |
| Normalization | `SumBasedDefuzzifier` | Sum-based rule strength normalization |
| Consequent | `ClassificationConsequentLayer` / `RegressionConsequentLayer` | First-order linear consequents |
| Membership functions | `GaussianMF` | Standard Gaussian antecedent MFs |

## Implementation notes

- `DombiTSKClassifierModel` and `DombiTSKRegressorModel` use a fixed Dombi parameter
  `lambda_ > 0` in the antecedent and default to `SumBasedDefuzzifier`.
- `DombiTSKClassifier` and `DombiTSKRegressor` are
  sklearn-compatible wrappers that build the rule base and membership
  functions from `input_configs`, `n_mfs`, `mf_init`, and `sigma_scale`.
- The estimators default to `mf_init="kmeans"` and `sigma_scale=1.0`.
- The default `rule_base` for estimator-built models is `"coco"` with
  `mf_init="kmeans"` and `"cartesian"` with `mf_init="grid"`.
- `CompositeGaussianMF` is available when a positive lower bound on
  antecedent membership values is desired, supporting ADMTSK-style stability.
- `AdaptiveDombiRuleLayer` is implemented in the codebase and provides
  per-rule adaptive Dombi exponents, which are exposed via the dedicated
  `ADMTSKClassifier` wrapper class.

### Strict paper mode

- Use `paper_strict=True` in `DombiTSKClassifier` to enforce the paper protocol defaults at the estimator level:
  `n_mfs=3`, `mf_init="grid"`, `sigma_scale=1.0`, `rule_base="coco"`, `lambda_=1.0`, `lower_bound=1/e` ($\approx 0.3679$), and `zero_consequent_init=True`.
- When `paper_strict=True`, conflicting values for these parameters raise `ValueError`.
- In `paper_strict` mode, Gaussian membership functions are automatically wrapped in `GaussianPiMF` (Composite Gaussian MF) with the paper-defined lower bound of $1/e$.
- In `paper_strict` mode, the consequent linear layer weights and biases are initialized to exactly zero.

## Alignment with the paper

- The paper defines a Dombi TSK baseline with Dombi antecedent aggregation and
  first-order consequent structure.
- highFIS implements this baseline directly through `DombiTSKClassifierModel` and
  `DombiTSKRegressorModel` (using `DombiTSKClassifier`).
- Rule strengths are normalized by sum-based defuzzification, matching the
  paper's TSK output aggregation.
- The package also includes building blocks for the ADMTSK extension, including
  `CompositeGaussianMF` and `AdaptiveDombiRuleLayer`, wrapped under `ADMTSKClassifier`.
