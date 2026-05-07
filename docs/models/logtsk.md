# LogTSK

## Reference

> Y. Cui, D. Wu and Y. Xu, "Curse of Dimensionality for TSK Fuzzy Neural Networks: Explanation and Solutions," 2021 International Joint Conference on Neural Networks (IJCNN), Shenzhen, China, 2021, pp. 1-8, doi: 10.1109/IJCNN52387.2021.9534265.

## Mathematical Formulation

### Antecedent

Same as vanilla TSK — product t-norm over Gaussian membership values:

$$
w_r = \prod_{d=1}^{D} \mu_{r,d}(x_d)
\tag{1}
$$

### Defuzzification (inverse-log normalization)

LogTSK avoids high-dimensional saturation by normalizing rule weights using
an inverse-log transform rather than a softmax over raw firing strengths.
Let

$$
Z_r = \log w_r = \sum_{d=1}^{D} \log \mu_{r,d}(x_d) \le 0.
\tag{2}
$$

Then the LogTSK rule coefficients are:

$$
f_r^{\text{log}} = -\frac{1}{Z_r} = \frac{1}{|Z_r|},
\tag{3}
$$

and the normalized rule strengths are:

$$
\bar{f}_r = \frac{f_r^{\text{log}}}{\sum_{i=1}^{R} f_i^{\text{log}}}
         = \frac{1/|Z_r|}{\sum_{i=1}^{R} 1/|Z_i|}.
\tag{4}
$$

This form is scale-invariant in log-space and avoids the saturation of the
standard softmax normalization when $D$ is large.

### Consequent (first-order)

$$
\hat{y} = \sum_{r=1}^{R} \bar{f}_r \left( b_{r,0} + \sum_{d=1}^{D} b_{r,d}\, x_d \right)
\tag{5}
$$

## Code ↔ Paper Correspondence

| Equation | Class / Method | Description |
|----------|----------------|-------------|
| (1) | `RuleLayer` with `t_norm="prod"` | Product t-norm antecedent |
| (2)–(4) | `InvLogDefuzzifier.forward()` | Inverse-log normalization |
| (5) | `ClassificationConsequentLayer` / `RegressionConsequentLayer` | Weighted consequent |

## Implementation notes

In highFIS, `LogTSKClassifier` and `LogTSKRegressor` default to
`InvLogDefuzzifier`, which implements the inverse-log normalization above.
This is the repository's LogTSK implementation.

### Model classes

- `LogTSKClassifier` and `LogTSKRegressor` use the standard TSK product
  antecedent (`t_norm="prod"`) together with
  `highfis.defuzzifiers.InvLogDefuzzifier`.
- The antecedent membership values are typically Gaussian, matching the
  paper's use of Gaussian MFs for each input dimension.
- Low-level model construction is done by passing an `input_mfs` mapping
  from feature names to a list of `highfis.memberships.GaussianMF` objects.

### Estimator wrappers

- `LogTSKClassifierEstimator` and `LogTSKRegressorEstimator` are sklearn-like
  wrappers around the low-level model classes.
- They build the rule base and Gaussian membership functions from
  `input_configs` or the high-level `n_mfs`, `mf_init`, and `sigma_scale`
  parameters.
- The default `sigma_scale=1.0` is recommended because the log-space
  defuzzifier is scale-invariant.
- The estimators expose training hyperparameters like `epochs`,
  `learning_rate`, `batch_size`, `shuffle`, `validation_data`, and
  `patience` for early stopping.

### Membership functions

The original LogTSK paper assumes standard Gaussian antecedent MFs:

$$
\mu_{r,d}(x_d)=\exp\left(-\frac{(x_d-m_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

In highFIS, the default membership type for LogTSK is
`highfis.memberships.GaussianMF`. Optional alternatives such as
`highfis.memberships.CompositeGaussianMF` are also available when a nonzero
lower bound on membership values is desired.

### Training in the paper vs. highFIS

- The LogTSK paper trains the model by optimizing the task loss over the
  output, using log-space normalized rule weights to maintain numerical
  stability in high-dimensional inputs.
- highFIS follows the same end-to-end gradient-based training paradigm.
- `BaseTSK.fit()` performs mini-batch optimization with a default AdamW
  optimizer, separate weight decay for consequent parameters, and optional
  validation-based early stopping.
- HighFIS also supports optional uniform-rule regularization via
  `ur_weight` and `ur_target` to encourage more evenly distributed rule
  activations during training.

## Alignment with the paper

- The paper defines LogTSK through inverse-log normalisation of log-domain
  firing strengths: $\bar{f}_r \propto 1/|\log w_r|$.
- highFIS implements this directly with `InvLogDefuzzifier`.
- The antecedent remains standard TSK product aggregation, and the
  consequent remains first-order, matching the paper's TSK structure.

Note: This document covers the paper's LogTSK behaviour. A temperature-
scaled log-space softmax is available in highFIS via
`LogSumDefuzzifier`, but it is not the default LogTSK implementation.
