# Layers API

## Module

`highfis.layers`

This module provides antecedent and consequent layer building blocks for the
highFIS fuzzy model variants.

## MembershipLayer

`MembershipLayer` evaluates membership functions for each input feature.

Input:

- tensor `x` with shape $(\text{batch}, n_{\text{inputs}})$.

Output:

- dictionary mapping input names to membership tensors with shape
  $(\text{batch}, n_{\text{mfs per input}})$.

### Notes

- Each input variable may have a different number of membership functions.
- Input mapping is stored in an `nn.ModuleDict`.
- Validation checks ensure `x.shape[1] == n_inputs`.

## RuleLayer

`RuleLayer` computes rule firing strengths from membership outputs.

Supported rule-base strategies:

- `"cartesian"`: full combinatorial rule base.
- `"coco"`: consistent same-index rule base for identical MF counts.
- `"en"`: enhanced fuzzy rule base (En-FRB).
- `"custom"`: user-specified rule index tuples.

### Behavior

- flattens the per-input membership tensors into a single membership vector;
- gathers the MF values for each rule from `rule_indices`;
- applies a t-norm aggregation over the input dimensions.

### Configuration

- `t_norm`: built-in t-norm name (`prod`, `gmean`, `dombi`, etc.).
- `t_norm_fn`: optional custom callable.
- `rules`: optional custom rule index list when `rule_base="custom"`.

### Validation

- All inputs must define at least one membership function.
- For `coco` and `en`, all inputs must have the same MF count.
- Custom rules are validated for length and index bounds.

## Gate functions

`highfis.layers` exposes several built-in gate activations:

- `gate1(u) = sigmoid(u)`
- `gate2(u) = 1 - exp(-u^2)`
- `gate3(u) = exp(-u^2)`
- `gate4(u) = u * sqrt(exp(1 - u^2))`
- `gate_m(u) = u^2 * exp(1 - u^2)`

The helper `resolve_gate_fn(gate_fn)` resolves either a string name or a
callable to the actual gate function.

`_gate_activation(u)` is the default gate function used by DG-ALETSK and
related models (`gate4`).

## AdaSoftminRuleLayer

`AdaSoftminRuleLayer` implements an adaptive softmin-like aggregation layer.

- Uses a numerically stable log-sum-exp formulation.
- Computes an input-dependent exponent `q` from the minimum membership.
- Produces rule strengths that interpolate toward soft minimum behavior.

### Use case

Use this layer when you need softmin-like firing strength computation with
improved numerical stability.

## DGALETSKRuleLayer

`DGALETSKRuleLayer` implements the DG-ALETSK antecedent.

- learns per-rule feature gate parameters `lambda_gates`;
- applies gate activation `M(lambda)` before aggregation;
- uses an adaptive Ln-Exp softmin formulation over gated membership degrees.

This is the antecedent used by `DGALETSKClassifier` and
`DGALETSKRegressor`.

## DGTSKRuleLayer

`DGTSKRuleLayer` implements the DG-TSK antecedent with explicit feature gates.

- learns per-rule feature gates `lambda_gates`;
- applies a gate function such as `gate1`, `gate2`, `gate3`, `gate4`, or
  `gate_m` to `lambda_gates`;
- multiplies gated values with the membership degrees;
- uses standard product aggregation to compute rule firing strengths.

This is the antecedent used by `DGTSKClassifier` and `DGTSKRegressor`.

## AdaptiveDombiRuleLayer

`AdaptiveDombiRuleLayer` implements per-rule adaptive Dombi aggregation.

- learns strict positive parameters `lambda_r` via softplus;
- computes a rule-specific Dombi soft conjunction across input memberships;
- interpolates between hard conjunction and soft consensus.

## ClassificationConsequentLayer

`ClassificationConsequentLayer` computes per-rule linear consequents and
aggregates class logits.

- each rule has a weight tensor of shape `(n_rules, n_classes, n_inputs)`;
- each rule has a bias tensor of shape `(n_rules, n_classes)`;
- the output is the normalized sum of rule-specific logits.

## GatedClassificationConsequentLayer

`GatedClassificationConsequentLayer` extends classification consequents with
feature-level and rule-level gating.

- learns `lambda_gates` for feature gating per rule;
- learns `theta_gates` for rule gating;
- applies gating to consequent weights and outputs before aggregation.

## GatedClassificationZeroOrderConsequentLayer

`GatedClassificationZeroOrderConsequentLayer` implements zero-order
classification consequents for DG-style training.

- each rule produces a class bias vector instead of a first-order affine map;
- applies rule gating via `theta_gates` only;
- is designed for efficient threshold-based pruning.

## GatedRegressionZeroOrderConsequentLayer

`GatedRegressionZeroOrderConsequentLayer` implements zero-order regression
consequents for DG-style training.

- each rule produces a scalar bias;
- applies rule gating via `theta_gates`;
- aggregates a scalar regression prediction via weighted sum.

## RegressionConsequentLayer

`RegressionConsequentLayer` computes per-rule linear regression consequents.

- each rule has a weight vector of shape `(n_rules, n_inputs)`;
- each rule has a bias scalar of shape `(n_rules,)`;
- the final output is a scalar prediction with shape `(\text{batch}, 1)`.

## GatedRegressionConsequentLayer

`GatedRegressionConsequentLayer` extends regression consequents with both
feature-level and rule-level gating.

- learns `lambda_gates` for feature gating per rule;
- learns `theta_gates` for rule gating;
- computes gated first-order regression outputs and aggregates them with
  normalized rule strengths.

## Notes

- The module registers `__all__` for the primary exported layer classes.
- Several layers use `torch.nn.Parameter` and initialize learnable gate/
  consequent weights for gradient-based optimization.
- Rule layers validate input shape and rule syntax before forward execution.
