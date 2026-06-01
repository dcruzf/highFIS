# MHTSK

Multihead Takagi–Sugeno–Kang (MHTSK) builds multiple sparse subantecedents from random feature subsets and jointly optimizes their rule consequents.

## Reference

> Z. Bian, Q. Chang, J. Wang and N. R. Pal, "Multihead Takagi–Sugeno–Kang Fuzzy System," in IEEE Transactions on Fuzzy Systems, vol. 33, no. 8, pp. 2561-2573, Aug. 2025, doi: [10.1109/TFUZZ.2025.3569227](https://doi.org/10.1109/TFUZZ.2025.3569227).

## Mathematical Formulation

### Subantecedent construction

MHTSK randomly samples `S` features without replacement to create a subantecedent and repeats this process `T` times. Each subantecedent is trained by Fuzzy C-Means clustering on the selected feature subset.

For a dataset with `D` inputs, the selected feature subset for head `t` is:

$$
\tilde{\mathbf{x}}^{(t)} \in \mathbb{R}^{S}
$$

and a sampled instance subset of size `n` is used to fit FCM for that head.

### Rule generation

Each head generates `K` fuzzy rules by applying FCM to the selected features. A rule from head `t` is defined only on the selected features, while all remaining features use a constant "don't care" membership function.

For rule `r` from head `t`, the antecedent is:

$$
\mu^{r}(\mathbf{x}) = \prod_{s \in \mathcal{S}_t} \mu_{A^{r}_{s}}(x_s)
$$

where `\mathcal{S}_t` are the indices of the `S` features chosen by head `t`.

### Normalization

All `T \times K` rules are normalized together using standard TSK normalization:

$$
\bar{\mu}^{r}(\mathbf{x}) = \frac{\mu^{r}(\mathbf{x})}{\sum_{i=1}^{T K} \mu^{i}(\mathbf{x})}
$$

This joint normalization is the same as the paper's final rule aggregation, ensuring the sparse rules compete globally.

### Sparse consequent

Each rule consequent depends only on the features used by that rule, creating a naturally sparse consequent structure:

$$
f^{r}(\mathbf{x}) = p^{r}_0 + \sum_{s \in \mathcal{S}_t} p^{r}_{s} x_s
$$

For classification, the output is:

$$
\hat{y}_c(\mathbf{x}) = \sum_{r=1}^{TK} \bar{\mu}^{r}(\mathbf{x}) f^{r}_{c}(\mathbf{x})
$$

For regression, the result is the scalar weighted sum of sparse consequents.

## Code ↔ Paper Correspondence

| Equation / Concept | Class / Method | Description |
|-------------------|----------------|-------------|
| Subset feature sampling `S`, `T` | `highfis.estimators._build_mhtsk_input_mfs` | Random head construction with feature subsets and FCM on each head |
| Feature coverage rate | `highfis.estimators.feature_coverage_rate` | Computes `1 - (1 - S/D)^T` from the paper |
| Scale parameter defaults | `highfis.estimators._resolve_mhtsk_scale_parameters` | Derives `S` and `T` from `h_value`, `fcr_target`, `xi`, and `sigma` |
| Sparse antecedents | `highfis.models.MHTSKClassifierModel`, `highfis.models.MHTSKRegressorModel` | Build rules with constant don't-care MFs on inactive features |
| Sparse consequent | `highfis.layers.SparseClassificationConsequentLayer`, `highfis.layers.SparseRegressionConsequentLayer` | Applies a mask so each rule only uses active input dimensions |
| Rule extraction | `highfis.estimators._extract_mhtsk_rule_indices` | Combines unsupervised max-firing strength and supervised Mann–Whitney selection |

## Implementation notes

- `highfis` represents each head with a partial rule base created by `FuzzyCMeans` on the sampled feature subset.
- Every input feature has a constant `ConstantMF(1.0)` to support don't-care membership when the feature is not selected by a head.
- The total number of rules is `n_heads * n_mfs`, because each head produces `n_mfs` cluster-based rules.
- The sparse consequent is implemented by masking the weight tensor with `rule_feature_mask` inside `SparseClassificationConsequentLayer` and `SparseRegressionConsequentLayer`.
- The rule masks are derived from the selected feature subsets and the per-head cluster indices.
- `rule_sigma` controls the Gaussian spread used for all selected feature MFs; the paper fixes this value to preserve interpretability and avoid numeric underflow.

## Model classes

- `highfis.models.MHTSKClassifierModel`
- `highfis.models.MHTSKRegressorModel`

These classes extend `BaseTSKClassifierModel` and `BaseTSKRegressorModel`, respectively, and use `rule_base="custom"` with explicit rule definitions and a sparse consequent.

### MHTSKClassifierModel

- Uses `SparseClassificationConsequentLayer`
- Default antecedent t-norm: `prod`
- Default defuzzifier: `SumBasedDefuzzifier`

### MHTSKRegressorModel

- Uses `SparseRegressionConsequentLayer`
- Default antecedent t-norm: `prod`
- Default defuzzifier: `SumBasedDefuzzifier`

## Estimator wrappers

- `highfis.estimators.MHTSKClassifier`
- `highfis.estimators.MHTSKRegressor`

These sklearn-style wrappers build the MHTSK rule base from raw data and expose configurable scale parameter resolution.

### `paper_strict` mode

Both wrappers support `paper_strict=True` to enforce a constrained configuration aligned with the paper defaults used in highFIS:

- Fixed defaults and constraints: `n_mfs=3`, `fcm_m=2.0`, `rule_sigma=1.0`, `xi=743.0`, `instance_sample_fraction=0.8`.
- Dynamic scale policy from input dimension `D`:
	- If `D <= 5000`: `S = max(1, round(0.02 * D))`, `T = 200`.
	- If `D > 5000`: `S = max(1, round(0.01 * D))`, `T = 300`.
	- `S` is additionally bounded by the numeric-underflow limit based on `xi` and `rule_sigma`.
- Rule extraction defaults enabled: `rule_extraction=True`, `crcr_us=0.5` (and `crcr_s=0.5` for classifier), `retrain_after_extraction=True`.
- Consequent-only optimization in strict mode uses an Adam optimizer over consequent parameters.

`paper_strict=True` intentionally rejects incompatible overrides (for example, manual `n_heads`, `head_size`, or disabling extraction).

No internal train/validation holdout split is performed by MHTSK estimators; validation data is only used when explicitly passed via `x_val` and `y_val`.

### Key estimator parameters

- `n_mfs`: Number of FCM clusters per head (`K`). Default: `3`.
- `n_heads`: Number of heads (`T`). When `None`, defaults are resolved from `head_size`, `fcr_target`, and `h_value`.
- `head_size`: Number of features per head (`S`). When `None`, defaults to `max(1, round(D * 0.02))` for `D <= 5000` or `max(1, round(D * 0.01))` otherwise.
- `head_size_ratio`: Alternative way to specify `head_size` as a fraction of `D`.
- `fcr_target`: Target feature coverage rate. Default behavior follows `0.85` when `h_value` is not set.
- `h_value`: Paper-derived scale constant `H`. If provided, it overrides `fcr_target`.
- `xi`: Numeric underflow threshold constant. Default: `743.0`.
- `rule_sigma`: Gaussian sigma used for FCM-derived MFs. Default: `1.0`.
- `fcm_m`: Fuzzy C-means fuzziness parameter. Default: `2.0`.
- `instance_sample_fraction`: Fraction of training samples used per head. Default: `0.8`.
- `rule_extraction`: Enable MHTSK_RE-style rule extraction. Default: `False`.
- `crcr_us`: Unsupervised cumulative rule contribution rate target. Default: `0.5`.
- `crcr_s`: Supervised cumulative rule contribution rate target (classifier only). Default: `0.5`.
- `retrain_after_extraction`: Retrain after extraction. Default: `True`.

## Membership functions

MHTSK uses standard Gaussian membership functions for active features:

$$
\mu_{r,d}(x_d) = \exp\left(-\frac{(x_d - c_{r,d})^2}{2\sigma^2}\right)
$$

- `highfis.memberships.GaussianMF` is used for selected features.
- `highfis.memberships.ConstantMF(1.0)` is used for inactive features, enabling don't-care semantics.

## Training in the paper vs. highFIS

- The paper trains MHTSK end-to-end with gradient-based optimization over joint rule weights and consequents.
- highFIS preserves this approach via `BaseTSK.fit()`, with mini-batch Adam training and optional early stopping.
- The main difference is that highFIS builds the sparse model structure explicitly before training, while the paper describes the same structure in algorithmic form.
- highFIS also supports optional uniform-rule regularization (`ur_weight`, `ur_target`) to encourage balanced rule activations.

## Rule extraction (MHTSK_RE)

The MHTSK variant with extraction uses two complementary selection strategies:

- Unsupervised: select rules with the largest maximum normalized firing strength across training samples.
- Supervised: for classifiers, select rules with the smallest Mann–Whitney pairwise `p`-value across class groups, using `1 - p_{\min}` as a score.

Selected rules are merged and the model is retrained on the reduced rule base.

## Alignment with the paper

- highFIS implements the random head construction and sparse consequent consistent with the MHTSK paper formulation.
- `feature_coverage_rate()` matches the paper's FCR equation:

$$
\text{FCR} = 1 - \left(1 - \frac{S}{D}\right)^{T}
$$

- The default `head_size` and `n_heads` resolution follows the paper-inspired scale parameter strategy using `xi`, `sigma`, `fcr_target`, and `h_value`.
- The sparse consequent layers mirror the paper's per-rule subspace-specific linear consequents.
- The `MHTSKClassifier` and `MHTSKRegressor` provide a user-facing API that maps to paper symbols `S`, `T`, `K`, and rule extraction workflow.
