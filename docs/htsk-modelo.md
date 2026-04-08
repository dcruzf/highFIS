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

- Antecedent initialization:
  the estimator default now follows the paper direction, using k-means cluster
  centroids for $m_{r,d}$. A grid-based alternative is still available.
- Sigma initialization:
  for `mf_init="kmeans"`, the estimator computes per-cluster spreads and scales
  them with `sigma_scale` (paper-like $h$ factor). For `mf_init="grid"`, sigma
  is derived from spacing and overlap.
- Rule-base defaulting:
  `rule_base` defaults to `"coco"` in k-means mode (one rule per cluster), and
  to `"cartesian"` in grid mode.
- Training protocol:
  the paper reports validation/early stopping schemes; the current
  implementation trains for fixed epochs.
- Default loss:
  the current implementation uses MSE over one-hot targets by default (with
  optional custom criterion).

These differences do not change the HTSK definition, but they do affect
experimental protocol.

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

then normalized as:

$$
\bar w_r=\frac{w_r}{\sum_{i=1}^{R}w_i+\varepsilon}
$$

where $\varepsilon$ is a numerical-stability constant.

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
\mathcal{L}_{UR}=\frac{1}{R}\sum_{r=1}^{R}\left(\bar w_r^{avg}-\tau\right)^2,
\qquad
\bar w_r^{avg}=\frac{1}{N}\sum_{n=1}^{N}\bar w_r(x_n)
$$

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

- Default optimizer: Adam.
- Mini-batch support with optional shuffling.
- Default loss: MSE on one-hot targets for classification.
- Optional custom criterion, custom optimizer, uniform regularization, and consequent batch norm.

### 3.3 HTSKClassifierEstimator Flow

1. Input validation with sklearn utilities.
2. Label encoding with `LabelEncoder`.
3. Feature-name/config resolution.
4. MF initialization:
   - default `mf_init="kmeans"`: k-means centroids + per-cluster sigmas,
   - optional `mf_init="grid"`: per-feature grid initialization.
5. `HTSKClassifier` instantiation and training.
6. Storage of training history and sklearn metadata.
7. Prediction through `predict_proba` and `predict` with sklearn compatibility.
