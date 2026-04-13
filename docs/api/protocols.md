# Protocols API

## Module

`highfis.protocols`

All protocols are `@runtime_checkable`, enabling `isinstance()` checks at runtime.

## MembershipFn

Structural type for any callable that maps an input tensor to membership degrees.

**Signature:** `(x: Tensor) → Tensor`

Any `nn.Module` with a matching `__call__` satisfies this protocol, including all
built-in membership functions (`GaussianMF`, `TriangularMF`, etc.).

## TNorm

Structural type for any callable that aggregates antecedent term activations per rule.

**Signature:** `(terms: Tensor) → Tensor`

Input shape: $(\text{batch}, n_{\text{inputs}})$. Output shape: $(\text{batch},)$.

The built-in functions `t_norm_prod`, `t_norm_min`, and `t_norm_gmean` all satisfy
this protocol.

## Defuzzifier

Structural type for any callable that normalizes rule firing strengths.

**Signature:** `(w: Tensor) → Tensor`

Input/output shape: $(\text{batch}, n_{\text{rules}})$.

All classes in `highfis.defuzzifiers` satisfy this protocol.

## ConsequentFn

Structural type for any callable that computes consequent output from inputs and
normalized firing strengths.

**Signature:** `(x: Tensor, norm_w: Tensor) → Tensor`

The built-in `ClassificationConsequentLayer` and `RegressionConsequentLayer`
satisfy this protocol.

## Example

```python
from highfis.protocols import MembershipFn
from highfis import GaussianMF

mf = GaussianMF(mean=0.0, sigma=1.0)
assert isinstance(mf, MembershipFn)  # True at runtime
```
