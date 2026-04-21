# DG-TSK

DG-TSK is a fuzzy rule-based model inspired by the DG-TSK paper, with learned
gates for feature selection and rule extraction.

## Implementation in highFIS

- `DGTSKClassifier` implements the classification variant.
- `DGTSKRegressor` implements the regression variant.
- `DGTSKRuleLayer` learns feature gates and applies them to membership outputs
  before product aggregation.
- The model can use an enhanced fuzzy rule base via `use_en_frb=True`.

## Gate functions

The following gate functions are supported for DG-TSK:

- `gate1(x) = sigmoid(x)`
- `gate2(x) = 1 - exp(-x^2)`
- `gate3(x) = exp(-x^2)`
- `gate4(x) = x * sqrt(exp(1 - x^2))`
- `gate_m(x) = x^2 * exp(1 - x^2)`

## Usage example

```python
from highfis import DGTSKClassifier, GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}

model = DGTSKClassifier(input_mfs, n_classes=2, gate_fea="gate_m", gate_rule="gate_m")
```
