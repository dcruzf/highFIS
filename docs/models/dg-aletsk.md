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
The paper's enhanced gate function (eq. 24) with $k=10$ is used:

$$
M(\lambda) = 1 - e^{-10\lambda^2}
$$

The gated membership term is computed by raising the base membership
value to the power of the gate value (paper eq. 12):

$$
\tilde{\mu}_{r,d}(x_d) = \mu_{r,d}(x_d)^{\,M(\lambda_d)}
$$

When $M(\lambda_d)=1$ the gate is fully open (membership unchanged);
when $M(\lambda_d)=0$ the gate is closed ($\tilde{\mu}_{r,d}=1$,
so the feature contributes nothing to the softmin).

### Adaptive Ln-Exp softmin

DG-ALETSK replaces the standard product T-norm with the Adaptive Ln-Exp (ALE)
softmin.  The firing strength of rule $r$ is computed directly from paper eq. 22:

$$
f_r(\mathbf{x}) = \frac{1}{\hat{q}} \log \left( \sum_{d=1}^{D} \exp\!\left(\hat{q}\,\tilde{\mu}_{r,d}(x_d)\right) \right),
\qquad \hat{q} = -\frac{700}{\max_d \tilde{\mu}_{r,d}(\mathbf{x})}
$$

The adaptive exponent $\hat{q}$ is recomputed on every forward pass: it is
not a learned parameter.  Choosing $\xi = 700$ guarantees
$\exp(\hat{q}\cdot\max_d\tilde{\mu}) = e^{-700} > 0$ in IEEE 754 double
precision (underflow boundary $\approx e^{-745}$), preventing numerical
underflow while driving the softmin to closely approximate
$\min_d \tilde{\mu}_{r,d}$.  Because $\hat{q} \le -700$ and $\tilde{\mu} \in (0,1]$,
the output satisfies $f_r \in (0, 1]$ by construction.

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
\bar{f}_r(\mathbf{x}) = \frac{f_r(\mathbf{x})}{\sum_{i=1}^{R} f_i(\mathbf{x})}
$$

The final model output is the weighted sum of gated rule consequents:

$$
\hat{y}^c(\mathbf{x}) = \sum_{r=1}^{R} \bar{f}_r(\mathbf{x}) \, \hat{y}_{r}^{c}(\mathbf{x})
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
| Adaptive Ln-Exp softmin antecedent | `DGALETSKRuleLayer` | Implements ALE softmin (paper eq. 22) with adaptive $\hat{q}=-700/\max_d\tilde{\mu}_{r,d}$ |
| Feature gates | `DGALETSKRuleLayer.lambda_gates` + `ExpGate(k=10)` | $\mu^{M(\lambda_d)}$ exponential gating (paper eqs. 12, 24) |
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
- The feature gate uses `ExpGate(k=10)` ($M(\lambda)=1-e^{-10\lambda^2}$,
  paper eq. 24), the enhanced gate function introduced in DG-ALETSK.
  Gate values are applied to antecedent memberships as $\mu^{M(\lambda_d)}$
  (paper eq. 12), i.e., exponential gating.
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
from highfis.models import DGALETSKClassifierModel
from highfis import GaussianMF

input_mfs = {
    "x1": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
}

model = DGALETSKClassifierModel(
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
