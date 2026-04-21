# Base TSK API

## Module

`highfis.base`

This module defines `BaseTSK`, the abstract foundation for all TSK fuzzy
models in highFIS. It factors out the shared antecedent pipeline, defuzzifier
behavior, and training loop so concrete models can focus on task-specific
consequent layers and loss criteria.

## BaseTSK

`BaseTSK` is an abstract PyTorch module with the following responsibilities:

- Build the antecedent pipeline from `input_mfs`.
- Create a rule layer with a configurable fuzzy rule base and t-norm.
- Normalize rule firing strengths through a pluggable defuzzifier.
- Apply a task-specific consequent layer for prediction.
- Train the model with mini-batches, uniform rule regularization, and early
  stopping.

### Forward pipeline

1. **MembershipLayer** — evaluates membership functions for each input.
2. **RuleLayer** — computes rule firing strengths using a rule base and t-norm.
3. **Defuzzifier** — normalizes firing strengths to probability-like weights.
4. **ConsequentLayer** — produces the final output from inputs and normalized
   rule weights.

### Constructor parameters

- `input_mfs`: mapping from input names to sequences of `MembershipFunction`
  objects. Must not be empty.
- `rule_base`: rule base type. Common values include `"cartesian"`,
  `"coco"`, `"en"`, or `"custom"`.
- `t_norm`: built-in t-norm name, such as `"prod"`, `"min"`, or `"gmean"`.
- `t_norm_fn`: optional custom t-norm callable. When provided, `t_norm` is
  ignored and the rule layer uses product-style activation with the custom
  function.
- `rules`: explicit rule index tuples, required when `rule_base="custom"`.
- `defuzzifier`: optional normalization module. Defaults to
  `SoftmaxLogDefuzzifier()`.
- `consequent_batch_norm`: whether to apply `BatchNorm1d` to the input before
  the consequent layer.

### Initialization behavior

- `input_names`, `n_inputs`, and `n_rules` are derived from `input_mfs` and
  the constructed `RuleLayer`.
- The default `defuzzifier` is `SoftmaxLogDefuzzifier`, which normalizes rule
  weights in log-space for numerical stability.
- If `consequent_batch_norm` is enabled, a `BatchNorm1d` layer is inserted
  before consequential evaluation.

### Abstract methods

Concrete subclasses must implement:

- `_build_consequent_layer() -> nn.Module`

  Return the task-specific consequent layer.

- `_default_criterion() -> nn.Module`

  Return the default loss function used when no custom criterion is supplied.

### Overridable hooks

- `_compute_loss(criterion, output, target) -> Tensor`

  Compute the main task loss. The default implementation simply applies
  `criterion(output, target)`.

- `_evaluate_validation(criterion, x_val, y_val) -> dict[str, float]`

  Evaluate the model on validation data. Returns a dictionary containing at
  least `"metric"`, which is used for early stopping. The default returns
  validation loss and its negation as the metric.

### Shared methods

- `forward(x) -> Tensor`

  Full forward pass through the antecedent pipeline and consequent layer.

- `forward_antecedents(x) -> Tensor`

  Compute normalized rule strengths from membership and rule layers without
  passing through the consequent.

- `_forward_train(x) -> (output, norm_w)`

  Compute both the model output and normalized rule weights in a single pass.

### Training

`fit(...)` trains the model using optional early stopping and uniform rule
regularization.

#### Data requirements

- `x` must be a 2D tensor with shape `(batch, n_inputs)`.
- `y` must be a 1D tensor with shape `(batch,)`.
- If provided, `x_val` must also be 2D with shape `(batch, n_inputs)`.
- If provided, `y_val` must be 1D with shape `(batch,)`.

#### Hyperparameters

- `epochs`: number of training epochs.
- `learning_rate`: optimizer learning rate.
- `criterion`: optional loss function. Defaults to `_default_criterion()`.
- `optimizer`: optional optimizer. If not provided, `AdamW` is created.
- `batch_size`: optional mini-batch size. If `None`, the entire dataset is used.
- `shuffle`: whether to shuffle batches each epoch.
- `ur_weight`: non-negative weight for uniform rule regularization.
- `ur_target`: optional target uniform activation; if provided it must satisfy
  `0.0 < ur_target <= 1.0`.
- `verbose`: whether to emit progress messages.
- `x_val`, `y_val`: optional validation data for early stopping.
- `patience`: number of epochs without improvement before stopping.
- `weight_decay`: weight decay applied only to consequent parameters.

#### Optimizer construction

If no optimizer is supplied, the default is `AdamW` with separate parameter
groups:

- antecedent (membership) parameters: `weight_decay=0.0`
- rule-layer parameters: `weight_decay=0.0`
- consequent parameters: `weight_decay=weight_decay`

#### Early stopping and validation

- When validation data is provided, the model evaluates the validation metric
  after each epoch using `_evaluate_validation(...)`.
- The best model state is copied and restored if the validation metric improves.
- Training stops early when the validation metric does not improve for
  `patience` epochs.

#### History output

The `fit` method returns a dictionary containing:

- `train`: per-epoch training loss.
- `ur`: per-epoch uniform regularization loss.
- `val`: per-epoch validation loss when validation data is used.
- any extra validation keys produced by `_evaluate_validation`.
- `stopped_epoch`: number of epochs completed.

### Regularization helper

The base class includes uniform regularization on normalized weights, which
penalizes deviation from a uniform expected rule activation distribution.

### Example: custom TSK model

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
