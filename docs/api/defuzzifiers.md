# Defuzzifiers API

## Module

`highfis.defuzzifiers`

Defuzzifiers normalize rule firing strengths so that each sample's weights sum
to one. They can be injected into any `BaseTSK`-derived model via the
`defuzzifier` constructor parameter.

## SoftmaxLogDefuzzifier

Default defuzzifier. Normalizes via `softmax(log(w))`:

$$
\bar{w}_r = \mathrm{softmax}(\log w_1,\ldots,\log w_R)_r
= \frac{w_r}{\sum_j w_j}
$$

Mathematically equivalent to `w / sum(w)` but numerically stable in high
dimensions thanks to the internal max-subtraction trick of `torch.softmax`.

Parameters:

- `eps`: clamping floor before log (default `1e-8`).

## SumBasedDefuzzifier

Classic sum-based normalization:

$$
\bar{w}_r = \frac{w_r}{\sum_{j=1}^{R} w_j + \epsilon}
$$

Parameters:

- `eps`: additive constant for numerical stability (default `1e-8`).

## LogSumDefuzzifier

Temperature-scaled log-space normalization:

$$
\bar{w}_r = \mathrm{softmax}\!\left(\frac{\log w_1}{T},\ldots,\frac{\log w_R}{T}\right)_r
$$

When $T = 1$, this recovers `SoftmaxLogDefuzzifier`. Lower temperatures sharpen
the distribution; higher temperatures flatten it.

Parameters:

- `temperature`: scaling factor $T > 0$ (default `1.0`).
- `eps`: clamping floor before log (default `1e-8`).

## Example

```python
from highfis import HTSKClassifier, SumBasedDefuzzifier
from highfis.memberships import GaussianMF

input_mfs = {"x1": [GaussianMF(0, 1), GaussianMF(1, 1)]}
model = HTSKClassifier(
    input_mfs,
    n_classes=3,
    defuzzifier=SumBasedDefuzzifier(),
)
```
