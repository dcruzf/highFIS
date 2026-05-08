# Vanilla TSK

Vanilla TSK is the standard first-order Takagi-Sugeno-Kang fuzzy model with Gaussian antecedents, product aggregation, and sum-based defuzzification.

## Reference

> Takagi, T. & Sugeno, M. (1985). "Fuzzy identification of systems and its
> applications to modeling and control." *IEEE Trans. Syst., Man, Cybern.*
> SMC-15(1):116–132.
> DOI: [10.1109/TSMC.1985.6313399](https://doi.org/10.1109/TSMC.1985.6313399)

## Mathematical Formulation

### Antecedent

Each rule evaluates $D$ Gaussian membership functions and aggregates them with
the **product t-norm**:

$$
\mu_{r,d}(x_d) = \exp\!\left(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}\right)
\tag{1}
$$

$$
w_r = \prod_{d=1}^{D} \mu_{r,d}(x_d)
\tag{2}
$$

### Defuzzification (sum-based)

Vanilla TSK normalizes rule firing strengths by their sum:

$$
\bar{f}_r = \frac{w_r}{\sum_{i=1}^{R} w_i}
\tag{3}
$$

In highFIS this is implemented by
`highfis.defuzzifiers.SumBasedDefuzzifier`.

When the antecedent is rewritten in log-space,
Equation (3) is mathematically equivalent to a softmax over log firing
strengths:

$$
\bar{f}_r = \frac{\exp(\log w_r)}{\sum_{i=1}^{R} \exp(\log w_i)}.
\tag{4}
$$

This exposes the saturation issue identified in
`HTSK_2021.md`: as $D$ grows, the log-domain activations become more extreme,
making the normalized weights dominated by a single rule.

### Consequent (first-order)

The final TSK output is a weighted sum of first-order consequents:

$$
\hat{y} = \sum_{r=1}^{R} \bar{f}_r \left( b_{r,0} + \sum_{d=1}^{D} b_{r,d}\, x_d \right)
\tag{5}
$$

## Code ↔ Paper Correspondence

| Equation | Class / Method | Description |
|----------|----------------|-------------|
| (1) | `GaussianMF.forward()` | Gaussian membership evaluation |
| (2) | `RuleLayer` with `t_norm="prod"` | Product t-norm antecedent |
| (3) | `SumBasedDefuzzifier.forward()` | Sum-based rule normalization |
| (4) | `SoftmaxLogDefuzzifier` | Equivalent log-space softmax form |
| (5) | `ClassificationConsequentLayer.forward()` / `RegressionConsequentLayer.forward()` | Weighted first-order consequent |

## Implementation notes

- `TSKClassifier` and `TSKRegressor` default to `SumBasedDefuzzifier`.
- The highFIS estimator wrappers `TSKClassifierEstimator` and
  `TSKRegressorEstimator` construct the Gaussian antecedent MFs from
  `input_configs`, `n_mfs`, `mf_init`, and `sigma_scale`.
- `GaussianMF` stores a raw parameter that is transformed with `softplus`
  to guarantee $\sigma > 0$.
- `SumBasedDefuzzifier` clamps rule weights to a small floor before
  normalization, improving numeric stability without changing the semantic
  sum-based normalization.
- In high dimensions, vanilla TSK is prone to softmax saturation because the
  normalized weights in Equation (4) are equivalent to a softmax over
  log-domain firing strengths.
