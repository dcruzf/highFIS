---
icon: lucide/function-square
---

# HTSK in highFIS: Technical Summary

This page summarizes the HTSK model implemented in highFIS based on:

Curse of Dimensionality for TSK Fuzzy Neural Networks: Explanation and Solutions (Cui, Wu, Xu, IJCNN).

## 1) Implementation Check Against the Paper

### 1.1 Core HTSK Idea

The paper replaces the sum in $Z_r$ with an average:

$$
Z_r'=-\frac{1}{D}\sum_{d=1}^{D}\frac{(x_d-m_{r,d})^2}{2\sigma_{r,d}^2}
$$

and then uses:

$$
\bar f_r' = \frac{\exp(Z_r')}{\sum_{i=1}^R \exp(Z_i')}
= \frac{f_r(x)^{1/D}}{\sum_{i=1}^R f_i(x)^{1/D}}
$$

In code, this appears as the geometric mean of antecedent memberships:

$$
\mathrm{gmean}(\mu_{r,1},\ldots,\mu_{r,D})
= \exp\left(\frac{1}{D}\sum_{d=1}^D \log \mu_{r,d}\right)
$$

For Gaussian memberships:

$$
\mu_{r,d}(x_d)=\exp\left(-\frac{(x_d-m_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

thus:

$$
\log \mu_{r,d}(x_d)=-\frac{(x_d-m_{r,d})^2}{2\sigma_{r,d}^2}
\Rightarrow
\exp\left(\frac{1}{D}\sum_d \log \mu_{r,d}\right)=\exp(Z_r')
$$

Conclusion: the implemented HTSK core is mathematically consistent with the paper.

### 1.2 Aligned Components

- Gaussian antecedent definitions match the paper formulation.
- HTSK firing via geometric mean matches $Z_r'$ and $f_r^{1/D}$.
- Firing strength normalization matches the TSK defuzzification structure.
- Linear per-rule consequents with weighted aggregation match first-order TSK.

### 1.3 Engineering Adaptations

- **Antecedent initialization:**
  the estimator default follows the paper, using k-means cluster
  centroids for $m_{r,d}$. A grid-based alternative is still available.
- **Sigma initialization:**
  for `mf_init="kmeans"`, the estimator computes per-cluster spreads and scales
  them with `sigma_scale` (paper's $h$ factor) plus stochastic noise
  $\sigma \sim \mathcal{N}(h, 0.2)$.  For `mf_init="grid"`, sigma is derived
  from spacing and overlap.
- **Rule-base defaulting:**
  `rule_base` defaults to `"coco"` in k-means mode (one rule per cluster), and
  to `"cartesian"` in grid mode.
- **Training protocol:**
  the implementation supports early stopping by **validation accuracy**
  (best-model restore), matching the PyTSK reference.
- **Default loss:**
  `nn.CrossEntropyLoss()` on raw logits (class-index targets), aligned with
  PyTSK. A custom criterion can still be passed.
- **Default optimizer:**
  `AdamW` with two parameter groups — antecedent parameters (centres/sigmas)
  receive `weight_decay=0`, while consequent parameters use a configurable
  `weight_decay` (default $10^{-8}$).
- **Normalization:**
  `NormalizationLayer` uses `softmax(log(w))`, which is mathematically
  equivalent to $w_r / \sum_i w_i$ but numerically stable in high dimensions.
- **Uniform regularization:**
  uses `sum` (not `mean`) over the rule-deviation vector, matching PyTSK.

## 2) Main Mathematical Formulas

### 2.1 First-Order TSK Consequent

For rule $r$:

$$
y_r(x)=b_{r,0}+\sum_{d=1}^{D} b_{r,d}x_d
$$

### 2.2 Gaussian Membership

$$
\mu_{r,d}(x_d)=\exp\left(-\frac{(x_d-m_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

### 2.3 Vanilla Product Firing

$$
f_r(x)=\prod_{d=1}^{D}\mu_{r,d}(x_d)
=\exp\left(-\sum_{d=1}^{D}\frac{(x_d-m_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

### 2.4 Vanilla Defuzzification

$$
\bar f_r(x)=\frac{f_r(x)}{\sum_{i=1}^{R}f_i(x)},
\qquad
\hat y(x)=\sum_{r=1}^{R}\bar f_r(x)\,y_r(x)
$$

Softmax equivalent form:

$$
Z_r=-\sum_{d=1}^{D}\frac{(x_d-m_{r,d})^2}{2\sigma_{r,d}^2},
\qquad
\bar f_r(x)=\frac{\exp(Z_r)}{\sum_{i=1}^{R}\exp(Z_i)}
$$

### 2.5 HTSK Formulation

$$
Z_r'=-\frac{1}{D}\sum_{d=1}^{D}\frac{(x_d-m_{r,d})^2}{2\sigma_{r,d}^2}
$$

$$
\bar f_r'(x)=\frac{\exp(Z_r')}{\sum_{i=1}^{R}\exp(Z_i')}
=\frac{f_r(x)^{1/D}}{\sum_{i=1}^{R}f_i(x)^{1/D}}
$$

### 2.6 highFIS HTSK Computation

Per rule in the implementation:

$$
w_r=\exp\left(\frac{1}{D}\sum_{d=1}^{D}\log\mu_{r,d}(x_d)\right)
$$

then normalized via **softmax on the log-space values**:

$$
\bar w_r = \mathrm{softmax}(\log w_1,\ldots,\log w_R)_r
= \frac{w_r}{\sum_{i=1}^{R}w_i}
$$

The `softmax` implementation is numerically stable because it internally
subtracts $\max_r \log w_r$ before exponentiation, preventing underflow in
high dimensions.

### 2.7 Multi-Class Consequent

For class $k$ and rule $r$:

$$
f_{r,k}(x)=w_{r,k}^\top x + b_{r,k}
$$

Final logits:

$$
\mathrm{logit}_k(x)=\sum_{r=1}^{R}\bar w_r\, f_{r,k}(x)
$$

Class probabilities:

$$
p(y=k\mid x)=\mathrm{softmax}(\mathrm{logits})_k
$$

### 2.8 Optional Uniform Regularization

The implementation supports regularization over average rule activation:

$$
\mathcal{L}_{UR}=\sum_{r=1}^{R}\left(\bar w_r^{avg}-\tau\right)^2,
\qquad
\bar w_r^{avg}=\frac{1}{N}\sum_{n=1}^{N}\bar w_r(x_n)
$$

The default target $\tau = 1/R$ (uniform); for classification, $\tau = 1/C$
(number of classes) can be passed via `ur_target`.

Mini-batch loss:

$$
\mathcal{L}=\mathcal{L}_{main}+\lambda\,\mathcal{L}_{UR}
$$

## 3) Model and Estimator Flow

### 3.1 HTSKClassifier Pipeline

1. `MembershipLayer`: computes $\mu_{r,d}(x_d)$ for every rule/input pair.
2. `RuleLayer`: aggregates antecedents with the selected t-norm (default HTSK: `gmean`).
3. `NormalizationLayer`: normalizes rule strengths across rules.
4. `ClassificationConsequentLayer`: computes linear consequents and weighted logits.
5. `softmax` (in prediction): converts logits to class probabilities.

### 3.2 Training in HTSKClassifier.fit

- **Default optimizer:** `AdamW` with separate parameter groups — antecedent
  (`weight_decay=0`) and consequent (`weight_decay=1e-8`).
- **Default loss:** `nn.CrossEntropyLoss()` (raw logits + class indices).
- **Early stopping:** monitors **validation accuracy** when `x_val`/`y_val` are
  provided; restores the best model weights on stop.
- Mini-batch support with optional shuffling.
- Optional custom criterion, custom optimizer, uniform regularization, and
  consequent batch norm.

### 3.3 HTSKClassifierEstimator Flow

1. Input validation with sklearn utilities.
2. Label encoding with `LabelEncoder`.
3. Feature-name/config resolution.
4. MF initialization:
   - default `mf_init="kmeans"`: k-means centroids + per-cluster sigmas
     ($\sigma \sim \mathcal{N}(h,0.2)$),
   - optional `mf_init="grid"`: per-feature grid initialization.
5. `HTSKClassifier` instantiation and training (CrossEntropyLoss + AdamW +
   early stopping by accuracy when validation data is provided).
6. Storage of training history and sklearn metadata.
7. Prediction through `predict_proba` and `predict` with sklearn compatibility.

## 4) Comparison: highFIS vs. Paper vs. PyTSK

The table below compares the main design choices across the IJCNN paper (Cui et al.),
the PyTSK reference implementation, and highFIS.

| Aspect | Paper (Cui et al.) | PyTSK Reference | highFIS |
|---|---|---|---|
| **Loss function** | Cross-entropy (implied) | `nn.CrossEntropyLoss()` | `nn.CrossEntropyLoss()` (default) |
| **Optimizer** | Adam (stated) | `AdamW` with param groups | `AdamW` with param groups (ante: wd=0, cons: wd=1e-8) |
| **Firing-strength normalization** | $\bar f_r = \exp(Z_r')/\sum_i \exp(Z_i')$ (softmax) | `F.softmax(Z, dim=1)` (log-space) | `softmax(log(w))` — equivalent |
| **Early stopping** | Validation-based | By validation **accuracy** | By validation **accuracy** (best-model restore) |
| **Consequent init** | Not specified | `kaiming_normal_` + zeros bias | `kaiming_normal_` + zeros bias |
| **MF center init** | K-means centroids | K-means centroids | K-means centroids (default) |
| **Sigma init** | $\sigma\sim\mathcal{N}(h,0.2)$ | $\sigma\sim\mathcal{N}(h,0.2)$ | $\sigma\sim\mathcal{N}(h,0.2)$ |
| **UR aggregation** | Not detailed | `sum` of squared deviations | `sum` of squared deviations |
| **HTSK t-norm** | Geometric mean ($f_r^{1/D}$) | Geometric mean | Geometric mean (`gmean`) |
| **Rule base** | CoCo / En-FRB | CoCo / En-FRB | `coco`, `en`, `cartesian`, `custom` |
| **Sigma positivity** | Unconstrained | Unconstrained $\sigma^2$ | Softplus reparameterization (more robust) |
| **sklearn API** | — | — | Full (`BaseEstimator`, `ClassifierMixin`) |
