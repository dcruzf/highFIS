# Protocols API

## Module

`highfis.protocols`

This module defines runtime-checkable structural typing protocols for the
highFIS fuzzy inference pipeline.

All protocols are decorated with `@runtime_checkable`, so `isinstance()` can
be used with conforming objects at runtime.

## MembershipFn

Structural type for any callable that maps an input tensor to membership
degrees.

**Signature:** `(x: Tensor) -> Tensor`

### Description

- The protocol defines `__call__(x: Tensor) -> Tensor`.
- Typically satisfied by membership function modules such as
  `GaussianMF`, `TriangularMF`, and others from `highfis.memberships`.

## TNorm

Structural type for any callable that aggregates antecedent term activations.

**Signature:** `(terms: Tensor) -> Tensor`

### Description

- The protocol defines `__call__(terms: Tensor) -> Tensor`.
- Used for aggregating input-term memberships into rule firing strengths.
- Built-in t-norms such as `t_norm_prod`, `t_norm_min`, and `t_norm_gmean`
  conform to this protocol.

## Defuzzifier

Structural type for any callable that normalizes rule firing strengths.

**Signature:** `(w: Tensor) -> Tensor`

### Description

- The protocol defines `__call__(w: Tensor) -> Tensor`.
- Accepts raw rule weights and returns normalized weights.
- All classes in `highfis.defuzzifiers` satisfy this protocol.

## ConsequentFn

Structural type for any callable that computes consequent output from inputs
and normalized rule weights.

**Signature:** `(x: Tensor, norm_w: Tensor) -> Tensor`

### Description

- The protocol defines `__call__(x: Tensor, norm_w: Tensor) -> Tensor`.
- Used by consequent layers in TSK models.
- `ClassificationConsequentLayer` and `RegressionConsequentLayer` satisfy this
  protocol.

## Exported names

The module exports the following protocol names:

- `MembershipFn`
- `TNorm`
- `Defuzzifier`
- `ConsequentFn`

## Example

```python
from highfis.protocols import MembershipFn
from highfis import GaussianMF

mf = GaussianMF(mean=0.0, sigma=1.0)
assert isinstance(mf, MembershipFn)
```
