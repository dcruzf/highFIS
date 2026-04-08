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

Normalizes firing strengths along the rule axis:

$$
\bar{w}_r=\frac{w_r}{\sum_j w_j + \varepsilon}
$$

## ClassificationConsequentLayer

Computes per-rule linear consequents and aggregates class logits:

$$
\mathrm{logits}_k=\sum_{r=1}^{R}\bar{w}_r\,f_{r,k}(x)
$$

where $f_{r,k}(x)$ is affine in input features.
