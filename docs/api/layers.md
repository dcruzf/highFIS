# Layers API

## Module

`highfis.layers`

## MembershipLayer

Evaluates membership functions for each input feature.

Input:

- $x$ with shape $(\text{batch}, n_{\text{inputs}})$.

Output:

- dictionary mapping input names to membership tensors with shape $(\text{batch}, n_{\text{mfs per input}})$.

## RuleLayer

Computes rule firing strengths from membership outputs.

Supported rule-base strategies:

- `"cartesian"`: full combinatorial rule base.
- `"coco"`: aligned same-index terms across dimensions.
- `"en"`: enhanced compact rule base.
- `"custom"`: user-specified rule index tuples.

Supports built-in and custom t-norm functions.

## NormalizationLayer

!!! warning "Deprecated"
    `NormalizationLayer` is now a thin subclass of
    [`SoftmaxLogDefuzzifier`](defuzzifiers.md#softmaxlogdefuzzifier).
    Prefer using `SoftmaxLogDefuzzifier` directly for new code.

Normalizes firing strengths along the rule axis using `softmax(log(w))`:

$$
\bar{w}_r = \mathrm{softmax}(\log w_1,\ldots,\log w_R)_r
= \frac{w_r}{\sum_j w_j}
$$

The `softmax` formulation is mathematically equivalent to dividing by the sum
but numerically stable in high dimensions thanks to the internal
max-subtraction trick.

## ClassificationConsequentLayer

Computes per-rule linear consequents and aggregates class logits:

$$
\mathrm{logits}_k=\sum_{r=1}^{R}\bar{w}_r\,f_{r,k}(x)
$$

where $f_{r,k}(x)$ is affine in input features.

## RegressionConsequentLayer

Computes per-rule linear consequents and aggregates a scalar regression output:

$$
\hat{y}=\sum_{r=1}^{R}\bar{w}_r\,f_r(x)
$$

where $f_r(x) = \mathbf{w}_r^\top x + b_r$ is an affine function of the input features.
Output shape is $(\text{batch}, 1)$.
