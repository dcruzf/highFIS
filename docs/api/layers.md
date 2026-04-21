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

## AdaSoftminRuleLayer

An adaptive softmin rule layer based on `α`-scaling in log-space.

- Uses an input-dependent exponent `q` to compute a stable soft-min aggregation.
- Computes rule strengths from membership degrees via a numerically stable log-sum-exp formulation.
- Useful when you need softmin-like behavior with adaptive sharpness and improved stability.

## DGALETSKRuleLayer

A DG-ALETSK antecedent layer that learns feature gates inside the rule aggregation.

- Learns `λ` gates per feature and rule.
- Applies gate activation before aggregation to enable feature selection.
- Uses an adaptive Ln-Exp softmin formulation for rule firing strengths.
## DGTSKRuleLayer

A DG-TSK antecedent layer that learns feature gates and applies them directly to Gaussian memberships.

- Learns `λ` gates per feature and rule.
- Applies an explicit gate function such as `gate1`, `gate2`, `gate3`, `gate4`, or `gate_m`.
- Uses product aggregation to compute rule firing strengths from gated memberships.
## AdaptiveDombiRuleLayer

A per-rule adaptive Dombi aggregation layer.

- Learns positive `λ_r` values for each rule using a softplus reparameterization.
- Computes Dombi firing strengths for each rule with numerically stable operations.
- Interpolates between strict conjunction and soft consensus per rule.

## ClassificationConsequentLayer

Computes per-rule linear consequents and aggregates class logits:

$$
\mathrm{logits}_k=\sum_{r=1}^{R}\bar{w}_r\,f_{r,k}(x)
$$

where $f_{r,k}(x)$ is affine in input features.

## GatedClassificationConsequentLayer

A gated classification consequent layer with both feature-level and rule-level gating.

- Learns `lambda_gates` to scale each input feature per rule.
- Learns `theta_gates` to gate entire rule outputs.
- Aggregates gated first-order consequents into logits with normalized rule strengths.

## GatedClassificationZeroOrderConsequentLayer

A zero-order gated classification consequent layer used during DG-ALETSK training.

- Produces class bias values per rule instead of first-order affine functions.
- Applies rule gating via `theta_gates` only.
- Supports efficient threshold-based rule pruning.

## GatedRegressionZeroOrderConsequentLayer

A zero-order gated regression consequent layer used during DG-ALETSK training.

- Produces scalar biases per rule.
- Applies rule gating via `theta_gates`.
- Outputs a scalar regression prediction by weighted sum over normalized rule strengths.

## GatedRegressionConsequentLayer

A gated regression consequent layer with feature- and rule-level gating.

- Learns `lambda_gates` to scale each input feature per rule.
- Learns `theta_gates` to gate entire rule outputs.
- Computes gated first-order regression outputs and aggregates them with normalized rule strengths.

## RegressionConsequentLayer

Computes per-rule linear consequents and aggregates a scalar regression output:

$$
\hat{y}=\sum_{r=1}^{R}\bar{w}_r\,f_r(x)
$$

where $f_r(x) = \mathbf{w}_r^\top x + b_r$ is an affine function of the input features.
Output shape is $(\text{batch}, 1)$.
