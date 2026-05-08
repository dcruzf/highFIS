# DG-ALETSK

DG-ALETSK combines adaptive Ln-Exp softmin aggregation with feature and rule gating, enabling simultaneous feature selection and rule extraction in high-dimensional fuzzy models.

## Reference

> G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional Fuzzy Approach With Simultaneous Feature Selection and Rule Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no. 11, pp. 3866–3880, Nov. 2023, doi: [10.1109/TFUZZ.2023.3270445](https://doi.org/10.1109/TFUZZ.2023.3270445).

## Overview

DG-ALETSK is a high-dimensional TSK fuzzy model that jointly learns:

- feature selection via antecedent gates;
- rule extraction via consequent gates;
- a differentiable adaptive Ln-Exp softmin antecedent;
- a zero-order DG phase followed by first-order fine tuning.

The highFIS implementation supports both classification and regression via
`DGALETSKClassifier` and `DGALETSKRegressor`.

## Mathematical Formulation

### Antecedent membership

Each antecedent uses Gaussian membership functions:

$$
\mu_{r,d}(x_d) = \exp\left(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

with learned antecedent centers $m_{r,d}$ and spreads $\sigma_{r,d}>0$.

### Feature gating

DG-ALETSK embeds one gate per input feature in the antecedent.
HighFIS implements the paper's gate activation as:

$$
M(\lambda) = \lambda \sqrt{e^{1 - \lambda^2}}
$$

The feature gate values are:

$$
\gamma_d = M(\lambda_d)
$$

and the gated membership terms become:

$$
\tilde{\mu}_{r,d}(x_d) = \gamma_d \, \mu_{r,d}(x_d)
$$

### Adaptive Ln-Exp softmin

DG-ALETSK replaces the standard product T-norm with an adaptive Ln-Exp softmin.
In highFIS the firing strength of rule $r$ is computed as:

$$
w_r(\mathbf{x}) = \frac{1}{\alpha} \log \left( \sum_{d=1}^{D} \exp\left(-\alpha \, \tilde{\mu}_{r,d}(x_d)\right) \right)
$$

where $\alpha > 0$ is a learned softness parameter. In the implementation,
$\alpha$ is stored as `raw_alpha` and activated with `softplus` to keep it
strictly positive.

### Rule gates and consequents

DG-ALETSK also embeds one gate per rule in the consequent.
For zero-order classification, each rule $r$ produces gated class logits:

$$
\hat{y}_{r}^{c}(\mathbf{x}) = M(\theta_r) \, p_{r}^{c}
$$

For regression, the same gate multiplies a scalar rule output.

### Output aggregation

Normalized rule strengths are computed as:

$$
\bar{w}_r(\mathbf{x}) = \frac{w_r(\mathbf{x})}{\sum_{i=1}^{R} w_i(\mathbf{x})}
$$

The final model output is the weighted sum of gated rule consequents:

$$
\hat{y}^c(\mathbf{x}) = \sum_{r=1}^{R} \bar{w}_r(\mathbf{x}) \, \hat{y}_{r}^{c}(\mathbf{x})
$$

### Threshold search and pruning

Gate thresholds are computed from the learned gate values and two
coefficients $\zeta_{\lambda}$ and $\zeta_{\theta}$:

$$
\tau_{\lambda} = \max_d M(\lambda_d) - \zeta_{\lambda} \bigl[ \max_d M(\lambda_d) - \min_d M(\lambda_d) \bigr]
$$

$$
\tau_{\theta} = \max_r M(\theta_r) - \zeta_{\theta} \bigl[ \max_r M(\theta_r) - \min_r M(\theta_r) \bigr]
$$

Features and rules with gate values below these thresholds are pruned.

## Code ↔ Paper Correspondence

| Concept | highFIS class / method | Notes |
|---|---|---|
| Adaptive Ln-Exp softmin antecedent | `DGALETSKRuleLayer` | Implements paper's ALE softmin with a stable log-sum-exp form |
| Feature gates | `DGALETSKRuleLayer.lambda_gates` + `gate4` | Gate values are applied multiplicatively to each membership |
| Rule gates | `GatedClassificationZeroOrderConsequentLayer.theta_gates`, `GatedRegressionZeroOrderConsequentLayer.theta_gates` | Gated zero-order consequents during DG training |
| Zero-order DG phase | `fit_dg_phase()` | Jointly optimizes antecedent, feature gates, rule gates, and zero-order consequents |
| First-order conversion | `convert_to_first_order()` | Preserves learned rule gates and switches consequent form |
| Threshold computation | `compute_thresholds(zeta_lambda, zeta_theta)` | Computes pruning thresholds from gate values |
| Threshold pruning | `apply_thresholds(tau_lambda, tau_theta)` | Sets low gate values to zero |
| Threshold search | `search_thresholds(...)` | Grid-searches `\zeta_\lambda` / `\zeta_\theta` and optionally refits consequents |

## Implementation notes

- The highFIS DG-ALETSK implementation uses `rule_base='coco'` by default.
- `use_en_frb=True` starts from an enhanced rule base (`en` FRB), but the
  paper's point-based FRB (P-FRB) is not constructed by default.
- The DG-ALETSK paper justifies P-FRB as a way to initialize an abundant
  candidate rule base from training samples, enabling the gate-based DG phase
  to perform rule extraction and feature selection in tandem.
- Estimator wrappers now support `rule_base='pfrb'`, which builds a
  point-based FRB from training samples and uses a CoCo rule base over the
  resulting sample-centered Gaussian MFs. Use `pfrb_max_rules` to cap the
  number of sample-based rules when the training set is large.
- `DGALETSKClassifier` and `DGALETSKRegressor` train a zero-order model in
  `fit_dg_phase()` and then rely on `convert_to_first_order()` plus
  `fit_finetune()` for first-order refinement.
- The feature gate activation is fixed to the paper's DG-ALETSK gate
  function `gate4` in the implementation.
- Although the gate function matches the paper exactly, highFIS applies it
  multiplicatively to the antecedent membership values (`μ ← μ · M(λ)`) rather
  than using the paper's more symbolic gate embedding notation (for example,
  `μ^{M(λ)}`). This preserves the gate shape while remaining a practical
  implementation choice.
- Threshold search is implemented by deep-copying the current model,
  pruning candidate copies, optionally refitting first-order consequents via
  least squares, and selecting the best validation score.

## highFIS API summary

- `fit_dg_phase(x, y, **kwargs)` — train the DG-ALETSK zero-order model.
- `convert_to_first_order()` — convert the trained zero-order consequent to
  a gated first-order consequent.
- `compute_thresholds(zeta_lambda, zeta_theta)` — compute pruning thresholds
  from current gate activations.
- `apply_thresholds(tau_lambda, tau_theta)` — prune low-value gates.
- `search_thresholds(...)` — search best gate thresholds and optionally apply
  them to the model.
- `fit_finetune(x, y, **kwargs)` — fine-tune the first-order DG-ALETSK model.

## Example

```python
from highfis import DGALETSKClassifier, GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}

model = DGALETSKClassifier(
    input_mfs,
    n_classes=2,
    use_en_frb=False,
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
