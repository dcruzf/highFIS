# DG-ALETSK

## Reference

> G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional Fuzzy Approach With Simultaneous Feature Selection and Rule Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no. 11, pp. 3866-3880, Nov. 2023, doi: [10.1109/TFUZZ.2023.3270445](https://doi.org/10.1109/TFUZZ.2023.3270445).


## Overview

DG-ALETSK is a high-dimensional fuzzy TSK architecture that combines:

- an adaptive Ln-Exp softmin antecedent operator;
- feature gates embedded in the antecedent;
- rule gates embedded in the consequent;
- a DG phase that jointly learns feature selection and rule extraction;
- conversion to first-order TSK consequents followed by fine tuning.

This design is aimed at high-dimensional classification and regression
problems where both feature reduction and rule simplification are desirable.

## Layer definitions in highFIS

### `DGALETSKRuleLayer`

This layer implements the paper’s adaptive Ln-Exp softmin antecedent.
Key behaviors:

- receives membership values `μ_{r,d}` from all input features;
- learns a positive softness parameter `α` via softplus from `raw_alpha`;
- applies feature gates `λ_d` to antecedent inputs before aggregation;
- computes rule firing strengths using a stable log-sum-exp formulation.

The layer’s core computation is:

1. `log_terms = -α * μ_{r,d}`
2. `log_sum = logsumexp(log_terms, axis=features)`
3. `log_avg = log_sum - log(D)`
4. `log_w = -log_avg / α`
5. `w_r = exp(log_w)`

This produces a differentiable approximation of a T-norm with adaptive
softness, consistent with DG-ALETSK’s antecedent formulation.

### `GatedClassificationZeroOrderConsequentLayer`

Used in the DG phase for classification:

- each rule produces a scalar class bias;
- each rule is gated by a learned rule gate `θ_r`;
- the gate activation `M(θ_r)` scales the rule contribution;
- class outputs are aggregated from gated rule biases.

This corresponds to the paper’s zero-order DG training, where rule weights are
learned and pruned directly.

### `GatedRegressionZeroOrderConsequentLayer`

The regression analogue to the classification zero-order consequent layer.

- each rule has a scalar bias output;
- each rule is gated by `M(θ_r)`;
- the final output is the normalized weighted sum of gated rule outputs.

### `GatedClassificationConsequentLayer`

A first-order consequent used after converting from the DG phase.

- contains first-order weights and biases per rule and class;
- includes feature-level gate coefficients and rule gate coefficients;
- supports fine tuning once antecedents and gates are fixed.

### `GatedRegressionConsequentLayer`

The first-order regression consequent layer.

- supports feature-level gating in weights and rule-level gating in outputs;
- is used for the fine-tuned first-order DG-ALETSK model.

## DG-ALETSK mechanisms

### DG phase

In highFIS, the DG phase is executed by
`DGALETSKClassifier.fit_dg_phase()` or `DGALETSKRegressor.fit_dg_phase()`.

- The model is initialized with a `DGALETSKRuleLayer` antecedent and a
  zero-order gated consequent.
- Feature gates `λ` and rule gates `θ` are both trained jointly with the
  antecedent and consequent parameters.
- This phase implements DG-ALETSK’s simultaneous feature selection and rule
  extraction.

### Gate activation and thresholds

The implementation uses the paper’s gate activation function:

```math
M(u) = u \sqrt{e^{1 - u^{2}}}
```

Thresholds are computed as:

- Feature threshold:

  ```math
  \tau_{\lambda} = \max_{d} M(\lambda_{d}) - \zeta_{\lambda} \left[\max_{d} M(\lambda_{d}) - \min_{d} M(\lambda_{d})\right]
  ```

- Rule threshold:

  ```math
  \tau_{\theta} = \max_{r} M(\theta_{r}) - \zeta_{\theta} \left[\max_{r} M(\theta_{r}) - \min_{r} M(\theta_{r})\right]
  ```

These formulas are implemented in
`DGALETSKClassifier.compute_thresholds()` and
`DGALETSKRegressor.compute_thresholds()`.

### Threshold search and pruning

HighFIS provides `search_thresholds(...)` to grid-search candidate values for
`ζ_λ` and `ζ_θ`.

For each candidate pair:

1. clone the current model;
2. compute `τ_λ` and `τ_θ` from current gate values;
3. apply thresholds with `apply_thresholds(tau_lambda, tau_theta)`;
4. optionally convert to first-order and refit consequents by least squares;
5. evaluate the candidate on training or validation data;
6. keep the best candidate and optionally apply it to the main model.

This matches the paper’s description of post-DG threshold search with
evaluation of thresholded models.

### Conversion to first-order

Once thresholds are selected, DG-ALETSK converts the pruned zero-order model
into a first-order TSK model via `convert_to_first_order()`.

- Preserves learned rule gates `θ`.
- Introduces first-order consequent weights and biases.
- Prepares the model for final fine tuning.

### Least-squares consequent re-estimation

To stabilize the chosen thresholds, highFIS can optionally refit the first-order
consequent parameters using least squares while keeping antecedent outputs fixed.

This is implemented during threshold search when `use_lse=True` and helps
match the paper’s recommendation to estimate consequent coefficients after gate
pruning.

## Comparison with the paper

The highFIS implementation is aligned with the DG-ALETSK paper in these ways:

- `DGALETSKRuleLayer` realizes the adaptive Ln-Exp softmin antecedent.
- Feature gates in the antecedent and rule gates in the consequent are both
  present and jointly learned.
- The model uses zero-order consequents during DG training and converts to
  first-order consequents later.
- Threshold coefficients `ζ_λ` and `ζ_θ` are searched and used to prune gates.
- Least-squares re-estimation of first-order consequents is available after
  thresholding.

Implementation notes:

- Threshold search in highFIS is performed via an explicit grid over candidate
  `ζ_λ` and `ζ_θ` values.
- The code supports both classification and regression versions of DG-ALETSK.

## highFIS API summary

- `fit_dg_phase(x, y, **kwargs)` — train the DG phase with zero-order
  consequents.
- `convert_to_first_order()` — switch to first-order consequents.
- `compute_thresholds(zeta_lambda, zeta_theta)` — compute gate thresholds.
- `apply_thresholds(tau_lambda, tau_theta)` — prune gates below thresholds.
- `search_thresholds(...)` — search threshold coefficients and optionally refit
  consequents by least squares.
- `fit_finetune(x, y, **kwargs)` — fine-tune the resulting first-order model.

## Example

```python
from highfis import DGALETSKClassifier, GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}
model = DGALETSKClassifier(input_mfs, n_classes=2)

model.fit_dg_phase(X_train, y_train, epochs=100, learning_rate=1e-3)

result = model.search_thresholds(
    X_train,
    y_train,
    zeta_lambda=[0.0, 0.25, 0.5, 0.75, 1.0],
    zeta_theta=[0.0, 0.25, 0.5, 0.75, 1.0],
    x_val=X_val,
    y_val=y_val,
)
print(result)

model.fit_finetune(X_train, y_train, epochs=50, learning_rate=1e-4)
```
