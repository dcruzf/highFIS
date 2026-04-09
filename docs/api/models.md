# Models API

## Module

`highfis.models`

## HTSKClassifier

`HTSKClassifier` is a `torch.nn.Module` implementing a full TSK classification pipeline:

1. Membership evaluation
2. Rule firing strength computation
3. Firing normalization
4. Consequent aggregation

### Constructor Highlights

- `input_mfs`: dictionary of input names to membership-function lists.
- `n_classes`: number of output classes.
- `rule_base`: `"cartesian"`, `"coco"`, `"en"`, or `"custom"`.
- `t_norm`: built-in t-norm name.
- `t_norm_fn`: optional custom t-norm callable.
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
