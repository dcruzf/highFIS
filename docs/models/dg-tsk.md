# DG-TSK

## Reference

> Guangdong Xue, Jian Wang, Bingjie Zhang, Bin Yuan, Caili Dai, Double groups of gates based Takagi-Sugeno-Kang (DG-TSK) fuzzy system for simultaneous feature selection and rule extraction, Fuzzy Sets and Systems, Volume 469, 2023, 108627, ISSN 0165-0114, https://doi.org/10.1016/j.fss.2023.108627.

## Overview

DG-TSK is a TSK fuzzy system designed to perform feature selection and rule extraction simultaneously. The method uses two groups of gates:

- feature gates in the antecedent inputs;
- rule gates in the consequent weights.

The architecture combines:

- a DG training phase with zero-order consequents;
- feature selection via pruning of input gates;
- rule extraction via pruning of rule gates;
- subsequent conversion to first-order consequents;
- fine tuning after rule extraction.

This design is intended for high-dimensional classification and regression problems where feature reduction and model simplification are both desirable.

## Implementation in highFIS

- `DGTSKClassifier` implements the classification variant.
- `DGTSKRegressor` implements the regression variant.
- `DGTSKRuleLayer` learns feature gates (`λ`) and applies them to membership outputs before product aggregation.
- `use_en_frb=True` enables an enhanced fuzzy rule base (`en` FRB) instead of a pre-defined rule set.

## Layer definitions in highFIS

### `DGTSKRuleLayer`

This layer represents the DG-TSK antecedent with feature gates.

- receives membership values `μ_{r,d}` for each rule and each feature;
- learns feature gate parameters `λ_d`;
- applies `M(λ_d)` to the membership values before firing strength computation;
- uses product aggregation (`t_norm='prod'`) over the gated inputs.

### Zero-order consequents

During the DG phase, highFIS uses zero-order consequent layers:

- `GatedClassificationZeroOrderConsequentLayer` for classifiers;
- `GatedRegressionZeroOrderConsequentLayer` for regressors.

These layers:

- learn one bias parameter per rule (classification) or one scalar output per rule (regression);
- apply rule gates `θ_r` through `M(θ_r)`;
- aggregate gated rule contributions in a normalized output.

### First-order consequents

After conversion, the model uses first-order consequent layers:

- `GatedClassificationConsequentLayer` for classifiers;
- `GatedRegressionConsequentLayer` for regressors.

These layers:

- preserve learned rule gates `θ_r`;
- introduce first-order weights for each rule and input;
- support fine tuning once the DG structure is stabilized.

## DG-TSK mechanisms

### DG phase

In highFIS, the DG phase is executed by `fit_dg_phase(x, y, **kwargs)`.

- It initializes a `DGTSKRuleLayer` with feature gates and a zero-order consequent layer.
- It trains feature gates `λ`, rule gates `θ`, and consequent parameters together.
- It preserves the configured gate functions `gate_fea` and `gate_rule`.

### Gate activation functions

DG-TSK supports several gate functions for `λ` and `θ`:

- `gate1(x) = sigmoid(x)`
- `gate2(x) = 1 - exp(-x^2)`
- `gate3(x) = exp(-x^2)`
- `gate4(x) = x * sqrt(exp(1 - x^2))`
- `gate_m(x) = x^2 * exp(1 - x^2)`

These functions are configured through the model constructor parameters `gate_fea` and `gate_rule`.

### Threshold computation

The model computes pruning thresholds for features and rules with `compute_thresholds(zeta_lambda, zeta_theta)`.

- `τ_{λ}` is derived from the feature gate activations `M(λ_d)` and coefficient `ζ_λ`.
- `τ_{θ}` is derived from the rule gate activations `M(θ_r)` and coefficient `ζ_θ`.

These thresholds determine which features and rules are retained or pruned.

### Threshold application

`apply_thresholds(tau_lambda, tau_theta)` performs pruning:

- it zeros feature gates `λ_d` whose gated value is less than or equal to `τ_{λ}`;
- it zeros rule gates `θ_r` whose gated value is less than or equal to `τ_{θ}`.

This reduces the model by keeping only relevant features and rules.

### Threshold search and pruning

`search_thresholds(...)` performs a grid search over `(ζ_λ, ζ_θ)` pairs:

1. clone the current model;
2. convert to first-order consequents if needed;
3. compute `τ_λ` and `τ_θ` with `compute_thresholds`;
4. apply thresholds with `apply_thresholds`;
5. optionally refit consequents by least squares (`use_lse=True`);
6. evaluate the candidate on `x_val, y_val` or the training set;
7. select the best pair based on score.

### Conversion to first order

`convert_to_first_order()` converts DG-TSK from zero-order to first-order consequents.

- it preserves rule gates `θ` learned during the DG phase;
- it initializes first-order weights and biases for each rule;
- it prepares the model for final fine tuning.

### Least-squares consequent re-estimation

During threshold search, highFIS can re-estimate consequent coefficients by LSE to stabilize the final model.

- this happens inside `search_thresholds(...)` when `use_lse=True`;
- it keeps the gate structure fixed and refines only the consequent coefficients.

## highFIS API summary

- `fit_dg_phase(x, y, **kwargs)` — train the DG phase with zero-order consequents.
- `convert_to_first_order()` — convert the DG-TSK model to first-order consequents.
- `compute_thresholds(zeta_lambda, zeta_theta)` — compute `τ_λ` and `τ_θ` from current gate values.
- `apply_thresholds(tau_lambda, tau_theta)` — prune features and rules based on thresholds.
- `search_thresholds(...)` — search the best threshold coefficients and optionally update the model.
- `fit_finetune(x, y, **kwargs)` — fine tune the model after first-order conversion.

## Usage example

```python
from highfis import DGTSKClassifier, GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}

model = DGTSKClassifier(
    input_mfs,
    n_classes=2,
    gate_fea="gate_m",
    gate_rule="gate_m",
    use_en_frb=True,
)

history = model.fit_dg_phase(X_train, y_train, epochs=100, learning_rate=1e-3)

result = model.search_thresholds(
    X_train,
    y_train,
    zeta_lambda=[0.0, 0.25, 0.5, 0.75, 1.0],
    zeta_theta=[0.0, 0.25, 0.5, 0.75, 1.0],
    x_val=X_val,
    y_val=y_val,
    use_lse=True,
    inplace=True,
)
print(result)

model.fit_finetune(X_train, y_train, epochs=50, learning_rate=1e-4)
```
