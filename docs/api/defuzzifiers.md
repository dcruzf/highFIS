# Defuzzifiers API

## Module

`highfis.defuzzifiers`

This module provides normalized rule-firing-strengh defuzzifiers for highFIS
models. Defuzzifiers take raw rule weights and convert them into valid
probability-like distributions across rules for consequent aggregation.

## SoftmaxLogDefuzzifier

Normalizes firing strengths via `softmax(log(w))`.

### Description

`SoftmaxLogDefuzzifier` computes normalized weights as:

$$
\bar{w}_r = \mathrm{softmax}(\log w)_r
$$

This is mathematically equivalent to `w / sum(w)` but is numerically more
stable in high dimensions because it uses `torch.softmax` with the internal
max-subtraction trick.

### Parameters

- `eps`: optional numeric stability floor for weight clamping before taking
  the logarithm. If `None`, the epsilon is inferred from `w.dtype`.

### Usage

- Accepts `w` tensors of shape `(n_samples, n_rules)`.
- Raises a `ValueError` if `w` is not 2-dimensional.

## SumBasedDefuzzifier

Classic sum-based normalization.

### Description

`SumBasedDefuzzifier` computes normalized weights as:

$$
\bar{w}_r = \frac{\max(w_r, \epsilon)}{\sum_{j} \max(w_j, \epsilon)}
$$

### Parameters

- `eps`: optional numeric stability floor for weight clamping before division.
  If `None`, the epsilon is inferred from `w.dtype`.

### Usage

- Accepts `w` tensors of shape `(n_samples, n_rules)`.
- Raises a `ValueError` if `w` is not 2-dimensional.

## LogSumDefuzzifier

Temperature-scaled log-space softmax normalization.

### Description

`LogSumDefuzzifier` computes normalized weights as:

$$
\bar{w}_r = \mathrm{softmax}\left(\frac{\log w}{T}\right)_r
$$

- `temperature=1.0` recovers the same behavior as
  `SoftmaxLogDefuzzifier`.
- Lower temperatures produce a sharper distribution.
- Higher temperatures produce a flatter distribution.

### Parameters

- `temperature`: positive scaling factor $T > 0$.
- `eps`: optional numeric stability floor for weight clamping before taking
  the logarithm. If `None`, the epsilon is inferred from `w.dtype`.

### Usage

- Accepts `w` tensors of shape `(n_samples, n_rules)`.
- Raises a `ValueError` if `temperature <= 0` or if `w` is not 2-dimensional.

## Example

```python
from highfis import HTSKClassifier
from highfis.defuzzifiers import SumBasedDefuzzifier
from highfis.memberships import GaussianMF

input_mfs = {"x1": [GaussianMF(0, 1), GaussianMF(1, 1)]}
model = HTSKClassifier(
    input_mfs,
    n_classes=3,
    defuzzifier=SumBasedDefuzzifier(),
)
```
