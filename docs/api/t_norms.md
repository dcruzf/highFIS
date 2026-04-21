# T-Norms API

## Module

`highfis.t_norms`

## Built-in T-Norms

### BaseTNorm

Abstract base class for t-norm strategy modules.

- Inherits from `torch.nn.Module` and `ABC`.
- Defines `forward(terms: Tensor, dim: int = -1) -> Tensor`.
- Can be used with custom `t_norm_fn` callables in TSK models.

### ProductTNorm

Product t-norm module implementing `t_norm_prod` behavior.

### MinimumTNorm

Minimum t-norm module implementing `t_norm_min` behavior.

### GMeanTNorm

Geometric-mean t-norm module implementing `t_norm_gmean` behavior.

### DombiTNorm

Dombi aggregation module implementing `t_norm_dombi` with a learnable or
fixed `lambda_` parameter.

### t_norm_prod

Product t-norm over antecedent terms:

$$
\mathcal{T}_{\mathrm{prod}}(\mu_1,\ldots,\mu_d)=\prod_{i=1}^{d}\mu_i
$$

### t_norm_min

Minimum t-norm:

$$
\mathcal{T}_{\mathrm{min}}(\mu_1,\ldots,\mu_d)=\min_i\mu_i
$$

### t_norm_gmean

Geometric-mean aggregation used by HTSK:

$$
\mathcal{T}_{\mathrm{gmean}}(\mu_1,\ldots,\mu_d)=\left(\prod_{i=1}^{d}\mu_i\right)^{1/d}
$$

Equivalent log form in implementation:

$$
\exp\left(\frac{1}{d}\sum_{i=1}^{d}\log\mu_i\right)
$$

### t_norm_dombi

Dombi aggregation interpolates between strict conjunction and soft consensus:

$$
\mathcal{T}_{\mathrm{dombi}}(\mu_1,\ldots,\mu_d)=\left[1 + \left(\sum_{i=1}^{d}\left(\frac{1}{\mu_i}-1\right)^{\lambda}\right)^{1/\lambda}\right]^{-1}
$$

The implementation clamps antecedent degrees away from zero for numerical stability.

### resolve_t_norm

Resolves built-in t-norm names: `"prod"`, `"min"`, `"gmean"`, `"dombi"`.

## Custom T-Norms

You can provide a custom callable in `RuleLayer` or `HTSKClassifier` through `t_norm_fn`.
The callable must accept a tensor with shape $(\text{batch}, n_{\text{inputs}})$ and return $(\text{batch},)$.
