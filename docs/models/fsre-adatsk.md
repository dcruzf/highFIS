# FSRE-ADATSK

FSRE-ADATSK extends ADATSK with embedded feature-selection and rule-extraction phases, using gated consequents and an enhanced rule base for high-dimensional data.

## Reference

> G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive Neuro-Fuzzy System With Integrated Feature Selection and Rule Extraction for High-Dimensional Classification Problems," in IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181, July 2023, doi: [10.1109/TFUZZ.2022.3220950](https://doi.org/10.1109/TFUZZ.2022.3220950)

## Overview

FSRE-ADATSK extends ADATSK into a three-phase high-dimensional fuzzy model:

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

FSRE-ADATSK computes rule firing strengths with an adaptive softmin:

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

FSRE-ADATSK uses gates only in the consequent layer, not in the antecedent.
Each gate value $u$ modulates consequent strength with the paper's default
activation:

$$
M(u) = u \sqrt{e^{1 - u^2}}
$$

This gate function has larger derivatives near zero than common alternatives,
making the selection process more responsive during training.

For a first-order TSK rule and class $c$:

$$
\hat{y}_r^c(\mathbf{x}) = M(u_r) \left(p_{r,0}^c + \sum_{d=1}^{D} p_{r,d}^c \, x_d\right)
$$

In highFIS, feature-level gates are shared across rules in the FS phase, while
rule-level gates are applied in the RE phase.

### Output Aggregation

The FSRE-ADATSK output for class $c$ is:

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

FSRE-ADATSK is trained in three sequential phases, each with a clear role in
making the model both compact and accurate.

- **Feature Selection (`fit_fs`)**
  - Purpose: identify the smallest set of input features that are useful for
    classification.
  - Implementation: the model keeps the current rule base and activates only
    the shared feature gates `M(λ_d)` in the consequent layer.
  - Effect: feature gates shrink the contribution of unimportant input
    dimensions by attenuating consequent weights; important features remain
    active while noisy or redundant features are suppressed.

- **Rule Extraction (`fit_re`)**
  - Purpose: remove redundant or irrelevant rules after the important input
    features have been identified.
  - Implementation: the model expands its antecedent layer to the Enhanced FRB
    (En-FRB) and then activates only the rule gates `M(θ_r)` in the consequent
    layer.
  - Effect: rule gates suppress weak rules in the larger candidate rule set,
    allowing the model to prune rules without fully enumerating an exponential
    rule base.

- **Fine Tuning (`fit_finetune`)**
  - Purpose: recover predictive performance after feature selection and rule
    extraction.
  - Implementation: all gates are disabled, so the consequent layer behaves as
    a plain first-order TSK consequent.
  - Effect: the reduced model is re-trained end-to-end to adjust remaining
    weights and bias terms for the final task.

In the low-level API, these phases are explicit methods on
`FSREADATSKClassifierModel`/`FSREADATSKRegressorModel`. The current implementation is
correct: `fit_fs()` sets the consequent layer to feature-gate mode without
changing the rule layer, `fit_re()` rebuilds the rule layer as En-FRB and
switches to rule-gate mode, and `fit_finetune()` turns off gating entirely.

Higher-level estimator wrappers can orchestrate the same sequence for
automatic end-to-end training.

For the related DG-ALETSK implementation, see `docs/models/dg-aletsk.md`.

## Comparison with the original IEEE FSRE-ADATSK paper

This document reflects the same FSRE-ADATSK design presented in the paper
"An Adaptive Neuro-Fuzzy System With Integrated Feature Selection and Rule
Extraction for High-Dimensional Classification Problems." The paper's main
claims are:

- ADATSK is the adaptive antecedent aggregation core.
- Gates are embedded only in the consequent layer, not in antecedents.
- Feature selection and rule extraction are performed in two successive
  phases, followed by a final fine-tuning phase.
- An Enhanced Fuzzy Rule Base (En-FRB) is used to keep rule growth
  manageable while still allowing effective extraction.

In highFIS, this is implemented by `AdaSoftminRuleLayer` for the adaptive
antecedent aggregation and by gated consequent layers for both classification
and regression. The gate function used in the paper,
$M(u) = u\\sqrt{e^{1 - u^2}}$, is the same gate activation function used in
`GatedClassificationConsequentLayer` and `GatedRegressionConsequentLayer`.

The highFIS implementation therefore matches the paper's distinction between
ADATSK as the base model and FSRE-ADATSK as the three-phase extension with
feature selection, rule extraction, and En-FRB support.

## Code Correspondence

| Concept | highFIS class / method |
|---|---|
| Adaptive softmin antecedent | `AdaSoftminRuleLayer` |
| Gated classification consequent | `GatedClassificationConsequentLayer` |
| Gated regression consequent | `GatedRegressionConsequentLayer` |
| FSRE-ADATSK classifier | `FSREADATSKClassifierModel` |
| FSRE-ADATSK regressor | `FSREADATSKRegressorModel` |
| Sklearn-style classifier estimator | `FSREADATSKClassifier` |
| Sklearn-style regressor estimator | `FSREADATSKRegressor` |
| En-FRB expansion | `FSREADATSKClassifierModel.expand_to_en_frb()` / `FSREADATSKRegressorModel.expand_to_en_frb()` |
| Phase training helpers | `fit_fs()`, `fit_re()`, `fit_finetune()` |

## Practical Notes

- `AdaSoftminRuleLayer` implements adaptive softmin antecedent aggregation;
  no per-rule exponent parameter is explicitly learned outside this layer.
- `lambda_init` is accepted by the estimator API for compatibility, but the
  core FSRE-ADATSK model computes its adaptive softmin index from current
  membership values rather than using a fixed learnable `lambda`.
- Set `use_en_frb=True` to start from the enhanced fuzzy rule base; otherwise,
  training begins on a compact CoCo-FRB and expands to En-FRB during RE.
- `rule_base` can still be set to `"coco"` or another supported rule base;
  the `use_en_frb` flag controls whether the enhanced rule base is used.
- `consequent_batch_norm=True` is the **default**. It is required for numerical
  stability: the first-order gated consequent spans all features and is trained
  with plain gradient descent, which diverges (consequent weights grow unbounded
  and become `NaN`) on high-dimensional data without normalisation, collapsing
  the classifier to a single class below the majority-class baseline. This
  mirrors the ADATSK default and is consistent with the CoCo-FRB TSK lineage the
  method builds on (Cui et al., 2020). It can be set to `False` for
  low-dimensional problems where divergence does not occur.
- **Loss function**: `FSREADATSKClassifier` defaults to `MSELoss` on one-hot
  targets, matching the ADATSK paper (eq. 8) that FSRE-ADATSK extends -- the same
  objective as `ADATSKClassifier`; regression uses `MSELoss`.

## Example

```python
from highfis import FSREADATSKClassifier

clf = FSREADATSKClassifier(
    n_mfs=4,
    mf_init="kmeans",
    lambda_init=1.0,
    fs_epochs=100,
    re_epochs=100,
    finetune_epochs=50,
    learning_rate=1e-3,
    random_state=0,
    use_en_frb=True,
)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

Low-level usage with explicit FS/RE phases:

```python
from highfis.models import FSREADATSKClassifierModel
from highfis.memberships import GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}
model = FSREADATSKClassifierModel(
    input_mfs,
    n_classes=3,
    lambda_init=1.0,
    use_en_frb=False,
)
model.fit_fs(x_train, y_train, epochs=100, learning_rate=1e-3)
model.fit_re(x_train, y_train, epochs=100, learning_rate=1e-3)
model.fit_finetune(x_train, y_train, epochs=50, learning_rate=1e-4)
```
