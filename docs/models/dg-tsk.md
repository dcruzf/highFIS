# DG-TSK

DG-TSK uses M-shaped feature and rule gates together with a point-based rule base to perform simultaneous feature selection and rule extraction.

## Reference

> Guangdong Xue, Jian Wang, Bingjie Zhang, Bin Yuan, Caili Dai, Double groups of gates based Takagi-Sugeno-Kang (DG-TSK) fuzzy system for simultaneous feature selection and rule extraction, Fuzzy Sets and Systems, Volume 469, 2023, 108627, ISSN 0165-0114, doi: [10.1016/j.fss.2023.](https://doi.org/10.1016/j.fss.2023.108627).

## Mathematical Formulation

### Antecedent

DG-TSK uses standard Gaussian membership functions for each input feature:

$$
\mu_{r,d}(x_d) = \exp\left(-\frac{(x_d - c_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

where $c_{r,d}$ and $\sigma_{r,d} > 0$ are the center and spread for rule $r$ and feature $d$.

### M-gate

The paper introduces a novel M-shaped gate function for both feature selection and rule extraction:

$$
M(\lambda) = \lambda^2 \exp\left(1 - \lambda^2\right)
$$

This function satisfies:

- $M(\lambda) \in [0, 1]$ for real $\lambda$;
- $M(\lambda) = 1$ at $\lambda = \pm 1$;
- large derivatives near the origin for faster early learning.

### Antecedent gating

DG-TSK embeds feature gates in the antecedents. The gate value raises the
base membership to a power, following the general gating mechanism described
in section 2.2 of the paper ($\mu^{M(\lambda)}$):

$$
\tilde{\mu}_{r,d}(x_d) = \mu_{r,d}(x_d)^{\,M(\lambda_d)}
$$

When $M(\lambda_d)=1$ the gate is fully open (membership unchanged);
when $M(\lambda_d)=0$ the gate is closed ($\tilde{\mu}_{r,d}=1$, so
the feature does not suppress the product T-norm).

Rule firing strengths are then computed with a product T-norm:

$$
w_r(\mathbf{x}) = \prod_{d=1}^{D} \tilde{\mu}_{r,d}(x_d)
$$

### Rule base

The paper defines a point-based fuzzy rule base (P-FRB) where each training example initializes one rule. This strategy is justified because a richer candidate rule base helps the DG-TSK gate mechanism perform both rule extraction and feature selection simultaneously.

In highFIS, the model classes do not construct the exact per-sample P-FRB directly; instead, the closest built-in richer option is `rule_base='en'` via `use_en_frb=True`. At the estimator wrapper level, however, DG-TSK estimators support `rule_base='pfrb'` and `pfrb_max_rules` to build a sample-centered FRB from training data. That option creates Gaussian membership functions at selected training points and then uses a CoCo-style rule base over those sample-centered MFs.

The `en` FRB is richer than a CoCo-FRB but is not identical to the paper's per-sample P-FRB.

### Consequent with rule gates

DG-TSK multiplies each rule's consequent by a rule gate:

$$
\hat{y}_r^c(\mathbf{x}) = M(\theta_r) \left(p_{r,0}^c + \sum_{d=1}^{D} p_{r,d}^c \, x_d\right)
$$

For regression, the same gate forms a scalar gated consequent.

### Output aggregation

The normalized rule weights are:

$$
\bar{w}_r(\mathbf{x}) = \frac{w_r(\mathbf{x})}{\sum_{i=1}^{R} w_i(\mathbf{x})}
$$

The final prediction is:

$$
\hat{y}^c(\mathbf{x}) = \sum_{r=1}^{R} \bar{w}_r(\mathbf{x}) \, \hat{y}_r^c(\mathbf{x})
$$

For regression, the same weighted sum applies to scalar rule outputs.

### Training protocol

The paper describes DG-TSK as a single training phase in which feature gates, rule gates, and zero-order consequents are optimized together. After that phase, the learned gate structure is used to convert the model to first-order consequents and fine-tune the reduced model.

## Code ↔ Paper Correspondence

| Concept | highFIS class / method | Notes |
|---|---|---|
| Gaussian membership | `highfis.memberships.GaussianMF` | antecedent MFs |
| M-gate | `highfis.layers.gate_m` | paper's M-shaped gate |
| Antecedent gating | `DGTSKRuleLayer.forward()` | raises membership to power `M(\lambda_d)`: $\mu^{M(\lambda_d)}$ |
| Rule gating | `GatedClassificationZeroOrderConsequentLayer`, `GatedRegressionZeroOrderConsequentLayer` | gated consequents |
| Rule base | `RuleLayer(rule_base='en')` | `en` FRB; approximates richer candidate set |
| DG phase | `fit_dg_phase()` | antecedent parameters are **frozen** during this phase; only gate params (λ, θ) and zero-order consequents are optimised (paper §3.3) |
| First-order conversion | `convert_to_first_order()` | switch to first-order consequents |
| Threshold search | `search_thresholds(...)` | search over `zeta_lambda`, `zeta_theta` |
| Pruning | `compute_thresholds()`, `apply_thresholds()` | gate-based feature/rule pruning |

## Implementation notes

- The paper's P-FRB is not implemented verbatim in highFIS. The `en` FRB is the closest available richer candidate rule base. When using `rule_base='pfrb'` via the estimator, call `model_.init_consequents_from_labels(y_t)` before `fit_dg_phase()` to apply the paper-faithful one-hot bias initialisation (paper eq. 24).
- `gate_m` is the default M-gate in highFIS and matches the paper's M-shaped gate function.
- DG-TSK feature gating is implemented as exponential gating: each membership value is raised to the power of its gate value ($\mu^{M(\lambda_d)}$), matching the general antecedent gating mechanism described in the paper.
- The DG phase freezes antecedent parameters and trains only gate params and zero-order consequents (paper §3.3). The recommended Phase 2 workflow differs by task:
    - **Classification** (`use_lse=False`): `search_thresholds` evaluates the zero-order model on the validation set directly, preserving the quality of the gate-training phase. Call `fit_finetune` afterwards to convert to first-order and retrain consequents with MFs and λ-gates frozen (paper §3.3).
    - **Regression** (`use_lse=True`): `search_thresholds` prunes gates and fits first-order consequents via least squares in a single step. The LSE result is the final model — **do not call `fit_finetune`** afterwards, as it would reset the LSE-fitted weights and retrain from zero.
- `DGTSKClassifier` and `DGTSKRegressor` support both classification and regression in the same DG-TSK style.

## highFIS API summary

- `init_consequents_from_labels(y)` — (classifier only) initialise zero-order consequent biases with one-hot encoded labels (paper eq. 24 / P-FRB). Call before `fit_dg_phase()`.
- `fit_dg_phase(x, y, **kwargs)` — train DG-TSK with zero-order consequents; antecedent MF parameters are frozen (paper §3.3).
- `convert_to_first_order()` — convert the model to first-order consequents while preserving rule gates.
- `compute_thresholds(zeta_lambda, zeta_theta)` — compute pruning thresholds from gate activations.
- `apply_thresholds(tau_lambda, tau_theta)` — prune features and rules by zeroing gates.
- `search_thresholds(x, y, *, zeta_lambda, zeta_theta, x_val, y_val, use_lse, inplace, ...)` — grid-search over `(zeta_lambda, zeta_theta)` pairs and select the best gate thresholds by validation score. When `use_lse=True`, each candidate also fits first-order consequents via least squares before scoring (recommended for **regression**). When `use_lse=False`, the zero-order model is scored directly (recommended for **classification**). With `inplace=True`, the winning thresholds are applied to `self`.
- `fit_finetune(x, y, **kwargs)` — convert the pruned zero-order model to first-order, reset consequent weights to zero, and retrain with MFs and λ-gates frozen (paper §3.3). **Call this only after `search_thresholds(use_lse=False)`** (classification path). Do not call it after `search_thresholds(use_lse=True)`, which already produces final first-order consequents via LSE.

## Usage examples

### Classification

For classification, Phase 2 uses `use_lse=False` so that threshold candidates are
ranked by the zero-order model's accuracy on the validation set.  After selecting
the best thresholds, `fit_finetune` converts the model to first-order and retrains
consequents with MFs and feature gates frozen (paper §3.3).

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

# Optional: P-FRB one-hot initialisation (paper eq. 24).
# Requires that the number of training samples >= n_rules.
model.model_.init_consequents_from_labels(y_train_t)

history = model.fit_dg_phase(X_train, y_train, epochs=30, learning_rate=1e-3)

# Phase 2a: select thresholds by evaluating the zero-order model.
result = model.search_thresholds(
    X_train,
    y_train,
    zeta_lambda=[0.0, 0.25, 0.5, 0.75, 1.0],
    zeta_theta=[0.0, 0.25, 0.5],
    x_val=X_val,
    y_val=y_val,
    use_lse=False,   # evaluate zero-order quality; do not fit first-order here
    inplace=True,
)
print(result)

# Phase 2b: convert to first-order and fine-tune (MFs and λ-gates frozen).
model.fit_finetune(X_train, y_train, epochs=60, learning_rate=1e-3)
```

### Regression

For regression, Phase 2 uses `use_lse=True` so that each threshold candidate is
evaluated after fitting first-order consequents via least squares.  The best
candidate's LSE-fitted model is the final result — **do not call `fit_finetune`**,
which would reset those weights.

```python
from highfis import DGTSKRegressor, GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}

model = DGTSKRegressor(
    input_mfs,
    gate_fea="gate_m",
    gate_rule="gate_m",
    use_en_frb=True,
)

history = model.fit_dg_phase(X_train, y_train, epochs=60, learning_rate=1e-3)

# Phase 2: select thresholds and fit first-order consequents via LSE in one step.
result = model.search_thresholds(
    X_train,
    y_train,
    zeta_lambda=[0.0, 0.25, 0.5, 0.75, 1.0],
    zeta_theta=[0.0, 0.25, 0.5],
    x_val=X_val,
    y_val=y_val,
    use_lse=True,    # fit first-order via LSE; this IS the final model
    inplace=True,
)
print(result)
# No fit_finetune here — LSE consequents are already optimal.
```
