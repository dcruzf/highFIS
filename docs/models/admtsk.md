# ADMTSK

ADMTSK is an adaptive Dombi TSK fuzzy system designed for high-dimensional inference.
It combines a Dombi T-norm antecedent with a positive lower-bound Composite Gaussian
membership function (GaussianPiMF) and normalized first-order consequents.

## Reference

> G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: A High-Dimensional Takagi–Sugeno–Kang Fuzzy System Based on Adaptive Dombi T-Norm," in IEEE Transactions on Fuzzy Systems, vol. 33, no. 6, pp. 1767-1780, June 2025, doi: [10.1109/TFUZZ.2025.3535640](https://doi.org/10.1109/TFUZZ.2025.3535640).

## Mathematical Formulation

### Antecedent

The ADMTSK antecedent uses GaussianPiMF with a positive lower bound:

$$
\mu_{r,d}(x_d) = \exp\left(-1 + \exp\left(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}\right)\right)
\tag{1}
$$

This GaussianPiMF produces membership values in $(1/e, 1]$ and avoids zero-valued
inputs to the Dombi T-norm.

The rule firing strength is computed by a Dombi T-norm:

$$
\varphi_r(\mathbf{x}) = \frac{1}{1 + \left[ \sum_{d=1}^{D} \left(\frac{1}{\mu_{r,d}(x_d)} - 1\right)^\lambda \right]^{1/\lambda}}
\tag{2}
$$

### Adaptive lambda

ADMTSK chooses the Dombi parameter $\lambda$ adaptively according to the input
feature dimension $D$ and the membership lower bound $\varepsilon$:

$$
\lambda = \frac{\ln D}{\ln(K - \varepsilon) - \ln(1 - \varepsilon)}
\tag{3}
$$

In highFIS, this is implemented with a default $K = 10$ and
$\varepsilon = 1/e$.

### Defuzzification

Normalized rule strengths are computed as in standard first-order TSK:

$$
\bar{\varphi}_r(\mathbf{x}) = \frac{\varphi_r(\mathbf{x})}{\sum_{i=1}^{R} \varphi_i(\mathbf{x})}
\tag{4}
$$

### Consequent

The model output is a weighted sum of first-order consequents:

$$
\hat{y}^c(\mathbf{x}) = \sum_{r=1}^{R} \bar{\varphi}_r(\mathbf{x}) \left(b_{r,0}^c + \sum_{d=1}^{D} b_{r,d}^c x_d\right)
\tag{5}
$$

## Code ↔ Paper Correspondence

| Equation | Class / Method | Description |
|---|---|---|
| (1) | `GaussianPiMF` | Composite Gaussian membership function with positive lower bound |
| (2) | `AdaptiveDombiTNorm` | Adaptive Dombi T-norm aggregation over antecedent degrees |
| (3) | `AdaptiveDombiTNorm.__init__` | Computes scalar $\lambda$ from $D$, $\varepsilon$, and $K$ |
| (4) | `SumBasedDefuzzifier` | Normalizes firing strengths into rule weights |
| (5) | `ClassificationConsequentLayer` / `RegressionConsequentLayer` | First-order consequent computation |

## Implementation notes

In highFIS, ADMTSK is implemented as two model classes:

- `ADMTSKClassifierModel`
- `ADMTSKRegressorModel`

These classes accept an `input_mfs` mapping of feature names to membership
functions. When used through the estimator wrappers, input Gaussian MFs are
converted to `GaussianPiMF` automatically.

The adaptive lambda strategy is enabled by default through the `adaptive`
parameter. When `adaptive=False`, ADMTSK falls back to a fixed Dombi parameter
`lambda_`.

ADMTSK also follows the paper's CoCo-FRB design by default; the model's
rule base is set to ``rule_base='coco'`` unless explicitly overridden.

The default settings are:

- `adaptive=True`
- `lambda_=1.0`
- `lower_bound=1/e`
- `K=10.0`
- `rule_base='coco'`
- `n_mfs=3`, `mf_init='grid'`
- paper-initialized antecedents with centers `[0.0, 0.5, 1.0]` and `sigma=1.0`
- `epochs=50`, `batch_size=None`
- `ADAM` optimizer (paper-style default in ADMTSK path)
- zero-initialized consequent weights and biases

## Model classes

- `ADMTSKClassifierModel` — classifier variant of ADMTSK.
- `ADMTSKRegressorModel` — regressor variant of ADMTSK.

Both use the same antecedent and consequent structure as standard TSK, while
replacing the product antecedent with adaptive Dombi aggregation and using
GaussianPiMF antecedents.

## Estimator wrappers

- `ADMTSKClassifier`
- `ADMTSKRegressor`

These wrappers provide sklearn-compatible `fit`/`predict` APIs and build the
inferential pipeline from high-level settings such as `n_mfs`, `mf_init`,
`sigma_scale`, and adaptive lambda parameters.

When the estimator constructs input membership functions it converts the
initial Gaussian MFs into `GaussianPiMF`, matching the paper's positive lower
bound membership design.

## Membership functions

ADMTSK is designed around the Gaussian Pi MF (GaussianPiMF):

- positive lower bound avoids zero membership values,
- enables stable Dombi computation in high dimensions,
- improves robustness in adaptive lambda selection.

The estimator wrappers default to `GaussianPiMF` for the ADMTSK pipeline.

## Training in the paper vs. highFIS

The ADMTSK paper describes end-to-end gradient-based training with adaptive
Dombi lambda and GaussianPiMF antecedents. In highFIS, the default ADMTSK
path follows the paper-oriented setup: CoCo-FRB with 3 rules, paper
antecedent initialization (`[0, 0.5, 1]`, `sigma=1`), zero-initialized
consequents, MSE objective for classification and Adam-based optimization.

Optional overrides remain available through estimator arguments to support
non-paper experimental variants.

The highFIS implementation preserves the paper's main design:

- adaptive Dombi T-norm antecedent,
- GaussianPiMF antecedent membership functions,
- first-order TSK consequents,
- normalized rule strengths via sum-based defuzzification.

## Alignment with the paper

This implementation mirrors the paper by:

- using `GaussianPiMF` as the positive lower bound membership function,
- computing a scalar adaptive `lambda` from feature dimension and lower bound,
- using a Dombi T-norm aggregation for antecedent rule firing strengths,
- keeping first-order consequents and standard sum normalization.

## Scope note

`ADMTSK` and `ADATSK` are different model families. This page documents only
the ADMTSK protocol from the 2025 paper.
