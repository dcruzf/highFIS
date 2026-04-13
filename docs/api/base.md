# Base TSK API

## Module

`highfis.base`

## BaseTSK

Abstract base class for all TSK fuzzy models. Factors the common
antecedent–defuzzification pipeline and provides a unified training loop with
early stopping and uniform regularization.

### Pipeline

1. **MembershipLayer** — evaluates membership functions per input feature.
2. **RuleLayer** — computes firing strengths via configurable t-norm and rule base.
3. **Defuzzifier** — normalizes firing strengths (pluggable; default `SoftmaxLogDefuzzifier`).
4. **ConsequentLayer** — task-specific output (built by subclass).

### Constructor Parameters

- `input_mfs`: dictionary of input names to membership-function lists.
- `rule_base`: `"cartesian"`, `"coco"`, `"en"`, or `"custom"`.
- `t_norm`: built-in t-norm name (`"prod"`, `"min"`, `"gmean"`).
- `t_norm_fn`: optional custom t-norm callable.
- `rules`: explicit rule tuples (required when `rule_base="custom"`).
- `defuzzifier`: any `nn.Module` for firing-strength normalization. Defaults to `SoftmaxLogDefuzzifier`.
- `consequent_batch_norm`: apply batch normalization to inputs before the consequent layer.

### Abstract Methods (must be implemented by subclasses)

- `_build_consequent_layer() → nn.Module` — return the task-specific consequent layer.
- `_default_criterion() → nn.Module` — return the default loss function.

### Overridable Hooks

- `_compute_loss(criterion, output, target) → Tensor` — compute the main task loss.
- `_evaluate_validation(criterion, x_val, y_val) → dict` — evaluate on validation data and return a dict with at least `"metric"` (higher is better for early stopping).

### Shared Methods

- `forward(x)` — full forward pass through the pipeline.
- `forward_antecedents(x)` — returns normalized rule strengths.
- `fit(...)` — unified training with optional early stopping, mini-batches, uniform regularization, and AdamW with separate weight-decay groups.

### Training Loop

The default optimizer is `AdamW` with two parameter groups:

- Antecedent (membership) parameters: `weight_decay=0`.
- Consequent parameters: configurable `weight_decay` (default `1e-8`).

Early stopping is activated when `x_val` and `y_val` are provided. The best
model weights are restored automatically.

## Example: Custom TSK Model

```python
from torch import nn
from highfis.base import BaseTSK
from highfis.layers import RegressionConsequentLayer

class MyTSK(BaseTSK):
    def _build_consequent_layer(self) -> nn.Module:
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()
```
