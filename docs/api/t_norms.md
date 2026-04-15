# T-Norms API

## Module

`highfis.t_norms`

## Built-in T-Norms

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

## Utility

### resolve_t_norm

Resolves built-in names: `"prod"`, `"min"`, `"gmean"`, `"dombi"`.

## Custom T-Norms

You can provide a custom callable in `RuleLayer` or `HTSKClassifier` through `t_norm_fn`.
The callable must accept a tensor with shape $(\text{batch}, n_{\text{inputs}})$ and return $(\text{batch},)$.
