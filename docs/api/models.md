# Models API

## Module

`highfis.models`

All models inherit from `BaseTSK` (see [Base TSK API](base.md)), which provides
the shared pipeline and unified training loop.

| Variant | Classifier | Regressor | T-norm | Defuzzifier |
|---------|-----------|-----------|--------|-------------|
| **HTSK** | `HTSKClassifier` | `HTSKRegressor` | `gmean` | `SoftmaxLogDefuzzifier` |
| **TSK (vanilla)** | `TSKClassifier` | `TSKRegressor` | `prod` | `SumBasedDefuzzifier` |
| **DombiTSK** | `DombiTSKClassifier` | `DombiTSKRegressor` | `dombi` | `SumBasedDefuzzifier` |
| **LogTSK** | `LogTSKClassifier` | `LogTSKRegressor` | `prod` | `LogSumDefuzzifier` |

For the mathematical details and scientific references, see:

- [HTSK Technical Notes](../htsk-modelo.md)
- [TSK Vanilla](../models/tsk-vanilla.md)
- [LogTSK](../models/logtsk.md)

## HTSKClassifier

`HTSKClassifier` is a `torch.nn.Module` implementing a full TSK classification pipeline:

1. Membership evaluation
2. Rule firing strength computation
3. Firing normalization (pluggable defuzzifier)
4. Consequent aggregation

### Constructor Highlights

- `input_mfs`: dictionary of input names to membership-function lists.
- `n_classes`: number of output classes.
- `rule_base`: `"cartesian"`, `"coco"`, `"en"`, or `"custom"`.
- `t_norm`: built-in t-norm name.
- `t_norm_fn`: optional custom t-norm callable.
- `defuzzifier`: optional custom defuzzifier (default `SoftmaxLogDefuzzifier`).
- `consequent_batch_norm`: optional batch normalization before consequents.

### Main Methods

- `forward(x)`: returns class logits.
- `predict_proba(x)`: returns softmax probabilities.
- `predict(x)`: returns class indices.
- `forward_antecedents(x)`: returns normalized rule strengths.
- `fit(...)`: trains model parameters with gradient descent.

### Training Notes

- Default optimizer: `AdamW` with separate param groups — antecedent
  (`weight_decay=0`) and consequent (`weight_decay=1e-8`).
- Default loss: `nn.CrossEntropyLoss()` — raw logits with class-index targets.
- Early stopping monitored by **validation accuracy** when `x_val`/`y_val` are
  provided. Best model weights are restored automatically.
- Supports mini-batches, shuffling, and optional uniform regularization.
- Custom criterion and custom optimizer can still be passed to `fit()`.

## HTSKRegressor

`HTSKRegressor` is a `torch.nn.Module` implementing a full TSK regression pipeline:

1. Membership evaluation
2. Rule firing strength computation
3. Firing normalization
4. Regression consequent aggregation

### Constructor Highlights

- `input_mfs`: dictionary of input names to membership-function lists.
- `rule_base`: `"cartesian"`, `"coco"`, `"en"`, or `"custom"`.
- `t_norm`: built-in t-norm name.
- `t_norm_fn`: optional custom t-norm callable.
- `defuzzifier`: optional custom defuzzifier (default `SoftmaxLogDefuzzifier`).
- `consequent_batch_norm`: optional batch normalization before consequents.

### Main Methods

- `forward(x)`: returns predictions with shape `(batch, 1)`.
- `predict(x)`: returns predictions as a 1-D tensor.
- `forward_antecedents(x)`: returns normalized rule strengths.
- `fit(...)`: trains model parameters with gradient descent (inherited from `BaseTSK`).

### Training Notes

- Default optimizer: `AdamW` with separate param groups — antecedent
  (`weight_decay=0`) and consequent (`weight_decay=1e-8`).
- Default loss: `nn.MSELoss()` — scalar predictions with continuous targets.
- Early stopping monitored by **validation loss** when `x_val`/`y_val` are
  provided. Best model weights (lowest validation loss) are restored automatically.
- Supports mini-batches, shuffling, and optional uniform regularization.
- Custom criterion and custom optimizer can still be passed to `fit()`.

## TSKClassifier

`TSKClassifier` is a `torch.nn.Module` implementing the original Takagi–Sugeno–Kang
classification pipeline with product t-norm and sum-based defuzzification.

### Constructor Highlights

- `input_mfs`: dictionary of input names to membership-function lists.
- `n_classes`: number of output classes.
- `rule_base`: `"cartesian"`, `"coco"`, `"en"`, or `"custom"`.
- `t_norm`: built-in t-norm name (default `"prod"`).
- `t_norm_fn`: optional custom t-norm callable.
- `defuzzifier`: optional custom defuzzifier (default `SumBasedDefuzzifier`).
- `consequent_batch_norm`: optional batch normalization before consequents.

### Main Methods

- `forward(x)`: returns class logits.
- `predict_proba(x)`: returns softmax probabilities.
- `predict(x)`: returns class indices.
- `forward_antecedents(x)`: returns normalized rule strengths.
- `fit(...)`: trains model parameters with gradient descent.

### Training Notes

Same optimizer and training configuration as `HTSKClassifier`.
Default loss is `nn.CrossEntropyLoss()`. Early stopping is monitored by
**validation accuracy** when `x_val`/`y_val` are provided.

## TSKRegressor

`TSKRegressor` is a `torch.nn.Module` implementing the original Takagi–Sugeno–Kang
regression pipeline with product t-norm and sum-based defuzzification.

### Constructor Highlights

- `input_mfs`: dictionary of input names to membership-function lists.
- `rule_base`: `"cartesian"`, `"coco"`, `"en"`, or `"custom"`.
- `t_norm`: built-in t-norm name (default `"prod"`).
- `t_norm_fn`: optional custom t-norm callable.
- `defuzzifier`: optional custom defuzzifier (default `SumBasedDefuzzifier`).
- `consequent_batch_norm`: optional batch normalization before consequents.

### Main Methods

- `forward(x)`: returns predictions with shape `(batch, 1)`.
- `predict(x)`: returns predictions as a 1-D tensor.
- `forward_antecedents(x)`: returns normalized rule strengths.
- `fit(...)`: trains model parameters with gradient descent.

### Training Notes

Same optimizer and training configuration as `HTSKRegressor`.
Default loss is `nn.MSELoss()`. Early stopping is monitored by
**validation loss** when `x_val`/`y_val` are provided.

## DombiTSKClassifier

`DombiTSKClassifier` is a `torch.nn.Module` implementing a Dombi-T-norm
TSK classification pipeline with sum-based defuzzification.

### Constructor Highlights

- `input_mfs`: dictionary of input names to membership-function lists.
- `n_classes`: number of output classes.
- `rule_base`: `"cartesian"`, `"coco"`, `"en"`, or `"custom"`.
- `t_norm`: built-in t-norm name (default `"dombi"`).
- `lambda_`: shape parameter for Dombi aggregation.
- `t_norm_fn`: optional custom t-norm callable.
- `defuzzifier`: optional custom defuzzifier (default `SumBasedDefuzzifier`).
- `consequent_batch_norm`: optional batch normalization before consequents.

### Main Methods

- `forward(x)`: returns class logits.
- `predict_proba(x)`: returns softmax probabilities.
- `predict(x)`: returns class indices.
- `forward_antecedents(x)`: returns normalized rule strengths.
- `fit(...)`: trains model parameters with gradient descent.

### Training Notes

Same optimizer and training configuration as `HTSKClassifier`.
Default loss is `nn.CrossEntropyLoss()`.

## DombiTSKRegressor

`DombiTSKRegressor` is a `torch.nn.Module` implementing a Dombi-T-norm
TSK regression pipeline with sum-based defuzzification.

### Constructor Highlights

- `input_mfs`: dictionary of input names to membership-function lists.
- `rule_base`: `"cartesian"`, `"coco"`, `"en"`, or `"custom"`.
- `t_norm`: built-in t-norm name (default `"dombi"`).
- `lambda_`: shape parameter for Dombi aggregation.
- `t_norm_fn`: optional custom t-norm callable.
- `defuzzifier`: optional custom defuzzifier (default `SumBasedDefuzzifier`).
- `consequent_batch_norm`: optional batch normalization before consequents.

### Main Methods

- `forward(x)`: returns predictions with shape `(batch, 1)`.
- `predict(x)`: returns predictions as a 1-D tensor.
- `forward_antecedents(x)`: returns normalized rule strengths.
- `fit(...)`: trains model parameters with gradient descent.

### Training Notes

Same optimizer and training configuration as `HTSKRegressor`.
Default loss is `nn.MSELoss()`.

## LogTSKClassifier

`LogTSKClassifier` is a `torch.nn.Module` implementing log-space defuzzification
with a temperature parameter for classification tasks.

### Constructor Highlights

- `input_mfs`: dictionary of input names to membership-function lists.
- `n_classes`: number of output classes.
- `rule_base`: `"cartesian"`, `"coco"`, `"en"`, or `"custom"`.
- `t_norm`: built-in t-norm name (default `"prod"`).
- `t_norm_fn`: optional custom t-norm callable.
- `defuzzifier`: optional custom defuzzifier (default `LogSumDefuzzifier`).
- `consequent_batch_norm`: optional batch normalization before consequents.
- `temperature`: temperature parameter τ for log-space normalization (default `1.0`).

### Main Methods

- `forward(x)`: returns class logits.
- `predict_proba(x)`: returns softmax probabilities.
- `predict(x)`: returns class indices.
- `forward_antecedents(x)`: returns normalized rule strengths.
- `fit(...)`: trains model parameters with gradient descent.

### Training Notes

Same optimizer and training configuration as `HTSKClassifier`.
Default loss is `nn.CrossEntropyLoss()`. The temperature parameter τ
controls the sharpness of the normalized firing-strength distribution
(τ = 1 is equivalent to vanilla TSK normalization; τ < 1 sharpens).

## LogTSKRegressor

`LogTSKRegressor` is a `torch.nn.Module` implementing log-space defuzzification
with a temperature parameter for regression tasks.

### Constructor Highlights

- `input_mfs`: dictionary of input names to membership-function lists.
- `rule_base`: `"cartesian"`, `"coco"`, `"en"`, or `"custom"`.
- `t_norm`: built-in t-norm name (default `"prod"`).
- `t_norm_fn`: optional custom t-norm callable.
- `defuzzifier`: optional custom defuzzifier (default `LogSumDefuzzifier`).
- `consequent_batch_norm`: optional batch normalization before consequents.
- `temperature`: temperature parameter τ for log-space normalization (default `1.0`).

### Main Methods

- `forward(x)`: returns predictions with shape `(batch, 1)`.
- `predict(x)`: returns predictions as a 1-D tensor.
- `forward_antecedents(x)`: returns normalized rule strengths.
- `fit(...)`: trains model parameters with gradient descent.

### Training Notes

Same optimizer and training configuration as `HTSKRegressor`.
Default loss is `nn.MSELoss()`. The temperature parameter τ controls the
sharpness of the normalized firing-strength distribution.
