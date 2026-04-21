# T-Norms API

## Module

`highfis.t_norms`

This module defines fuzzy conjunction strategies for antecedent aggregation.
It exposes both module-based T-norm classes and convenience functions for
common aggregations.

## Types

### `TNormFn`

A callable type alias used for T-norm functions.

**Signature:** `Callable[..., Tensor]`

## BaseTNorm

Abstract base class for T-norm strategy modules.

- Inherits from `torch.nn.Module` and `ABC`.
- Defines `forward(terms: Tensor, dim: int = -1) -> Tensor`.
- Concrete subclasses implement specific aggregation behavior.

## ProductTNorm

Product t-norm module.

### Behavior

- Computes the product of antecedent terms over the specified dimension.
- Equivalent to the classic fuzzy product conjunction.

### `forward(terms, dim=-1)`

Returns `torch.prod(terms, dim=dim)`.

## MinimumTNorm

Minimum t-norm module.

### Behavior

- Computes the minimum over antecedent terms along the specified dimension.
- Equivalent to the classic fuzzy minimum conjunction.

### `forward(terms, dim=-1)`

Returns `torch.min(terms, dim=dim).values`.

## GMeanTNorm

Geometric mean t-norm module.

### Constructor parameters

- `eps`: optional numeric stability floor for input clamping.

### Behavior

- Computes the geometric mean of antecedent terms over the specified
  dimension.
- Uses the log-domain implementation for numerical stability.

### `forward(terms, dim=-1)`

Returns `exp(mean(log(clamp(terms, min=eps))))`.

## DombiTNorm

Dombi t-norm module.

### Constructor parameters

- `lambda_`: positive shape parameter (`lambda_ > 0`).
- `eps`: optional numeric stability floor for input clamping.

### Behavior

- Implements the Dombi aggregation strategy.
- Clamps inputs to `(eps, 1.0]` before computing the Dombi formula.

### `forward(terms, dim=-1)`

Computes:

```python
clamped = terms.clamp(min=eps, max=1.0)
inv = (1.0 / clamped) - 1.0
powered = torch.pow(inv, lambda_)
summed = powered.sum(dim=dim)
return 1.0 / (1.0 + torch.pow(summed, 1.0 / lambda_))
```

## Convenience functions

### `t_norm_prod(terms)`

Product t-norm over antecedent terms with shape `(batch, n_inputs)`.

### `t_norm_min(terms)`

Minimum t-norm over antecedent terms with shape `(batch, n_inputs)`.

### `t_norm_gmean(terms, eps=None)`

Geometric mean aggregation for HTSK-like behavior.

### `t_norm_dombi(terms, lambda_=1.0, eps=None)`

Dombi aggregation over antecedent terms.

## resolve_t_norm

### `resolve_t_norm(name)`

Resolve a built-in t-norm by name.

Supported names:

- `"prod"` → `ProductTNorm()`
- `"min"` → `MinimumTNorm()`
- `"gmean"` → `GMeanTNorm()`
- `"dombi"` → `DombiTNorm()`

Raises `ValueError` if the name is not one of the supported strings.

## Exported names

- `BaseTNorm`
- `ProductTNorm`
- `MinimumTNorm`
- `GMeanTNorm`
- `DombiTNorm`
- `TNormFn`
- `resolve_t_norm`
- `t_norm_prod`
- `t_norm_min`
- `t_norm_gmean`
- `t_norm_dombi`

## Example

```python
from highfis.t_norms import resolve_t_norm, t_norm_prod

prod = resolve_t_norm("prod")
result = prod(terms, dim=1)

result2 = t_norm_prod(terms)
```
