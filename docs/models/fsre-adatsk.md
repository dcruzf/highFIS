# FSRE-AdaTSK

## Reference

> Xue, Guangdong; Chang, Qin; Wang, Jian; Zhang, Kai; Pal, Nikhil R. (2023).
> "An Adaptive Neuro-Fuzzy System With Integrated Feature Selection and
> Rule Extraction for High-Dimensional Classification Problems." *IEEE
> Transactions on Fuzzy Systems* 31(7):2167–2181.
> DOI: [10.1109/TFUZZ.2022.3220950](https://doi.org/10.1109/TFUZZ.2022.3220950)

## Overview

FSRE-AdaTSK extends AdaTSK into a three-phase high-dimensional fuzzy model:

1. **Feature Selection (FS)** — embedded gates on consequent coefficients
   identify and retain important input features.
2. **Rule Extraction (RE)** — an enhanced fuzzy rule base (En-FRB) and rule
   gates discard irrelevant rules without exponential growth.
3. **Fine Tuning** — the reduced model is re-trained to maximize accuracy and
   interpretability.

The model combines adaptive antecedent aggregation with gated consequents so
that feature and rule importance are learned jointly.

## Mathematical Formulation

### Antecedent Membership

Each rule-term membership is evaluated with a smooth Gaussian MF:

$$
\mu_{r,d}(x_d) = \exp\left(-\frac{(x_d - c_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

where $c_{r,d}$ is the center and $\sigma_{r,d} > 0$ is the spread.

### Adaptive Softmin (Ada-softmin)

FSRE-AdaTSK computes rule firing strengths with an adaptive softmin:

$$
\phi_r(\mathbf{x}) = \left( \frac{1}{D} \sum_{d=1}^{D}
      \mu_{r,d}(x_d)^{\hat{q}_r(\mathbf{x})} \right)^{1/\hat{q}_r(\mathbf{x})}
$$

The exponent $\hat{q}_r(\mathbf{x})$ is adapted per rule and per input so that
firing strengths remain numerically stable in very high dimensions and the
aggregation approximates minimum-like behavior.

### Rule Normalization

Normalized rule weights are computed as:

$$
\bar{\phi}_r(\mathbf{x}) = \frac{\phi_r(\mathbf{x})}{\sum_{i=1}^{R} \phi_i(\mathbf{x})}
$$

The normalized weights are then used to aggregate rule consequents.

### Gated Consequents

FSRE-AdaTSK uses gates only in the consequent layer. For each feature or rule
parameter, a gate value $u$ modulates output strength with:

$$
M(u) = u \sqrt{e^{1 - u^2}}
$$

This gate function has larger derivatives near zero than common alternatives,
making the selection process more responsive during training.

For a first-order TSK rule and class $c$:

$$
\hat{y}_r^c(\mathbf{x}) = M(u_r) \left(p_{r,0}^c + \sum_{d=1}^{D} p_{r,d}^c \, x_d\right)
$$

where $M(u_r)$ can represent a rule-level gate and additional gates can be
applied at the feature/consequent coefficient level.

### Output Aggregation

The FSRE-AdaTSK output for class $c$ is:

$$
y^c(\mathbf{x}) = \sum_{r=1}^{R} \bar{\phi}_r(\mathbf{x}) \, \hat{y}_r^c(\mathbf{x})
$$

For regression, the same normalized-weight aggregation applies to scalar rule
outputs.

### Enhanced Fuzzy Rule Base (En-FRB)

The En-FRB is a middle ground between Compactly Combined FRB (CoCo-FRB) and
Fully Combined FRB (FuCo-FRB). It expands the rule set enough to support rule
extraction while avoiding exponential rule growth with feature dimension.

In highFIS, En-FRB can be enabled via `use_en_frb=True`, and the model supports
explicit phase transitions for FS, RE, and fine tuning.

## Learning Phases

FSRE-AdaTSK is trained in three sequential phases:

- **Feature Selection (`fit_fs`)**
  - Train on the current compact rule base.
  - Learn gate parameters that identify important features.
- **Rule Extraction (`fit_re`)**
  - Expand the rule layer to En-FRB.
  - Learn rule gates that prune irrelevant rules.
- **Fine Tuning (`fit_finetune`)**
  - Re-train the reduced model after extraction.

These phases may be executed from the low-level model API or orchestrated by
higher-level estimator wrappers.

## Code Correspondence

| Concept | highFIS class / method |
|---|---|
| Adaptive softmin antecedent | `AdaSoftminRuleLayer` |
| Gated classification consequent | `GatedClassificationConsequentLayer` |
| Gated regression consequent | `GatedRegressionConsequentLayer` |
| FSRE-AdaTSK classifier | `FSREAdaTSKClassifier` |
| FSRE-AdaTSK regressor | `FSREAdaTSKRegressor` |
| Sklearn-style classifier estimator | `FSREAdaTSKClassifierEstimator` |
| Sklearn-style regressor estimator | `FSREAdaTSKRegressorEstimator` |
| En-FRB expansion | `FSREAdaTSKClassifier.expand_to_en_frb()` / `FSREAdaTSKRegressor.expand_to_en_frb()` |
| Phase training helpers | `fit_fs()`, `fit_re()`, `fit_finetune()` |

## Practical Notes

- Use `lambda_init` to control the initial adaptive aggregation parameter.
- Set `use_en_frb=True` to start training with the enhanced fuzzy rule base.
- `rule_base` can still be set to common bases like `"coco"` or a custom rule
  list, but En-FRB is triggered when `use_en_frb=True`.
- `consequent_batch_norm=True` can improve stability when input features vary in
  scale.

## Example

```python
from highfis import FSREAdaTSKClassifierEstimator

clf = FSREAdaTSKClassifierEstimator(
    n_mfs=4,
    mf_init="kmeans",
    lambda_init=1.0,
    epochs=200,
    learning_rate=1e-3,
    random_state=0,
    use_en_frb=True,
)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

Low-level usage with explicit FS/RE phases:

```python
from highfis import FSREAdaTSKClassifier, GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}
model = FSREAdaTSKClassifier(
    input_mfs,
    n_classes=3,
    lambda_init=1.0,
    use_en_frb=False,
)
model.fit_fs(x_train, y_train, epochs=100, learning_rate=1e-3)
model.fit_re(x_train, y_train, epochs=100, learning_rate=1e-3)
model.fit_finetune(x_train, y_train, epochs=50, learning_rate=1e-4)
```
