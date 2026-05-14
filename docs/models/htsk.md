# HTSK

HTSK modifies standard TSK aggregation by averaging membership values in log-space, reducing saturation and enabling more stable high-dimensional inference.

## Reference

> Y. Cui, D. Wu & Y. Xu, "Curse of Dimensionality for TSK Fuzzy Neural Networks: Explanation and Solutions," 2021 International Joint Conference on Neural Networks (IJCNN), Shenzhen, China, 2021, pp. 1-8, doi: [10.1109/IJCNN52387.2021.9534265](https://doi.org/10.1109/IJCNN52387.2021.9534265).

## Mathematical Formulation

### Antecedent

HTSK shares the same antecedent structure as vanilla TSK: each rule uses
Gaussian membership functions over every input feature.

$$
\mu_{r,d}(x_d)=\exp\left(-\frac{(x_d-m_{r,d})^2}{2\sigma_{r,d}^2}\right)
$$

where $m_{r,d}$ is the rule centre for feature $d$ and $\sigma_{r,d}>0$ is
its spread.

### Aggregation

Instead of the standard product t-norm,
HTSK computes the rule activation as the geometric mean of the membership
values:

$$
w_r = \left(\prod_{d=1}^{D} \mu_{r,d}(x_d)\right)^{1/D}
= \exp\left(\frac{1}{D} \sum_{d=1}^{D} \log \mu_{r,d}(x_d)\right).
$$

This averaging in log-space reduces the dimensionality bias that makes
product-based firing strengths vanish as $D$ grows.

### Normalization

HTSK normalizes rule weights with a softmax over the log-domain activations:

$$
\bar{w}_r = \frac{\exp(\log w_r)}{\sum_{i=1}^{R} \exp(\log w_i)}.
$$

Because $\log w_r$ is already dimensionally scaled by $1/D$, the resulting
normalisation is stable for high-dimensional inputs and avoids softmax
saturation without inflating the Gaussian widths.

### Output

For both classification and regression, HTSK uses standard first-order TSK
consequents and aggregates them with the normalized rule weights.

Classification:

$$
\mathbf{y} = \sum_{r=1}^{R} \bar{w}_r \mathbf{y}_r,
\qquad
\mathbf{y}_r = W_r \mathbf{x} + \mathbf{b}_r.
$$

Regression:

$$
\hat{y} = \sum_{r=1}^{R} \bar{w}_r \hat{y}_r,
\qquad
\hat{y}_r = \mathbf{w}_r^\top \mathbf{x} + b_r.
$$

## Code ↔ Paper Correspondence

| Paper concept | highFIS implementation |
|--------------|------------------------|
| Geometric-mean antecedent | `HTSKClassifier` / `HTSKRegressor` with `t_norm="gmean"` |
| Log-domain aggregation | `HTSKClassifier` / `HTSKRegressor` uses `SoftmaxLogDefuzzifier` |
| Normalized rule weights | `SoftmaxLogDefuzzifier.forward()` |
| First-order consequent | `ClassificationConsequentLayer` / `RegressionConsequentLayer` |

## Implementation notes

- `HTSKClassifier` and `HTSKRegressor` default to `t_norm="gmean"` and
  `SoftmaxLogDefuzzifier`.
- `HTSK` is not the same as `LogTSK`: HTSK averages log-membership values and
  then applies a softmax, while LogTSK uses inverse-log normalisation.
- The core advantage of HTSK is that the exponent in the softmax is scaled by
  $1/D$, which keeps the activation distribution stable as the number of
  input dimensions grows.
- `consequent_batch_norm=True` can be enabled to normalise consequent inputs
  before the last linear layer.
- HighFIS supports custom `defuzzifier` modules, but the default for HTSK is
  `SoftmaxLogDefuzzifier` to match the paper.

### Estimator wrappers

- `HTSKClassifierEstimator` and `HTSKRegressorEstimator` are sklearn-like
  wrappers around the low-level model classes.
- They build Gaussian membership functions from `input_configs` or `n_mfs`,
  `mf_init`, and `sigma_scale`.
- The estimators expose the standard hyperparameters used in the paper,
  including `epochs`, `learning_rate`, `batch_size`, `shuffle`, and
  `validation_data` for early stopping.
- The default `sigma_scale=1.0` is recommended because HTSK's log-space
  normalization already compensates for dimensionality.

### Membership functions

- The paper assumes Gaussian membership functions, and highFIS uses
  `highfis.memberships.GaussianMF` by default.
- For `mf_init="kmeans"`, the estimators derive MF centres from k-means
  cluster centroids and compute sigmas from within-cluster spread.

### Training in the paper vs. highFIS

- The original paper trains HTSK with mini-batch gradient descent and a
  modest learning rate, typically `0.01`.
- highFIS follows the same end-to-end gradient-based training paradigm using
  `BaseTSK.fit()`, which supports mini-batch AdamW, optional early stopping,
  and optional uniform-rule regularization (`ur_weight`, `ur_target`).
- The default HTSK estimator settings mirror the experimental setup of the
  paper: `n_mfs=30`, `mf_init="kmeans"`, `sigma_scale=1.0`, `epochs=200`,
  `learning_rate=1e-2`, `batch_size=512`, and `patience=20`.

## Alignment with the paper

- The paper introduces HTSK as a high-dimensional variant of TSK that
  avoids softmax saturation by averaging log-domain membership strengths.
- highFIS implements this directly with `HTSKClassifier`, `HTSKRegressor`,
  and `SoftmaxLogDefuzzifier`.
- The antecedent remains a Gaussian product structure, but the rule activation
  is computed as a $D$-th root of the product, which is equivalent to the
  geometric mean of the memberships.
- This makes HTSK numerically stable for large $D$ while preserving the
  first-order TSK consequent form.
