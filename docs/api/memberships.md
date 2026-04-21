# Membership Functions API

## Module

`highfis.memberships`

This module exposes differentiable membership functions designed for fuzzy
models in highFIS.

## Base class

### MembershipFunction

`MembershipFunction` is the abstract base class for all membership functions.

- Inherits from `torch.nn.Module`.
- Stores numeric stability epsilon `eps`.
- Provides `_as_tensor(x)` to convert scalar inputs into tensors.

## Membership function classes

### GaussianMF

Gaussian membership function:

$$
\mu(x)=\exp\left(-\frac{(x-c)^2}{2\sigma^2}\right)
$$

Parameters:

- `mean`: center $c$.
- `sigma`: spread parameter $\sigma > 0$.
- `eps`: numeric stability constant.

Notes:

- `sigma` is parameterized through `softplus` so it remains positive.
- `mean` and `sigma` are learnable parameters.

### CompositeGaussianMF

Composite Gaussian membership function with a positive lower bound:

$$
\mu(x)=\epsilon + (1 - \epsilon)\exp\left(-\frac{(x-c)^2}{2\sigma^2}\right)
$$

Parameters:

- `mean`: center $c$.
- `sigma`: spread parameter $\sigma > 0$.
- `eps`: numeric stability floor.

Notes:

- Ensures membership values never fall below `eps`.
- Useful for models where a nonzero infimum is needed.

### TriangularMF

Triangular membership function defined by three vertices:

$$
\mu(x) = \max\left(0,\;\min\left(\frac{x - l}{c - l},\;\frac{r - x}{r - c}\right)\right)
$$

Parameters:

- `left`: left vertex $l$.
- `center`: peak vertex $c$.
- `right`: right vertex $r$.

Constraints:

- $l \le c \le r$
- $l \neq r$

### TrapezoidalMF

Trapezoidal membership function with a plateau:

$$
\mu(x) = \max\left(0,\;\min\left(\frac{x - a}{b - a},\;1,\;\frac{d - x}{d - c}\right)\right)
$$

Parameters:

- `a`, `b`, `c`, `d`: vertices with $a \le b \le c \le d$.
- Must satisfy $a \neq d$.

### BellMF

Generalized bell membership function:

$$
\mu(x) = \frac{1}{1 + \left|\frac{x - c}{a}\right|^{2b}}
$$

Parameters:

- `a`: width parameter ($a > 0$, softplus-reparameterized).
- `b`: slope parameter ($b > 0$, softplus-reparameterized).
- `center`: center $c$.

Notes:

- Both `a` and `b` are learned in a way that enforces positivity.

### SigmoidalMF

Sigmoidal membership function:

$$
\mu(x) = \frac{1}{1 + \exp(-a(x - c))}
$$

Parameters:

- `a`: slope parameter.
- `center`: inflection point $c$.

### DiffSigmoidalMF

Difference of two sigmoid functions:

$$
\mu(x) = \sigma(a_1, x-c_1) - \sigma(a_2, x-c_2)
$$

Parameters:

- `a1`, `center1`: slope and center of the first sigmoid.
- `a2`, `center2`: slope and center of the second sigmoid.

### ProdSigmoidalMF

Product of two sigmoid functions:

$$
\mu(x) = \sigma(a_1, x-c_1) \cdot \sigma(a_2, x-c_2)
$$

Parameters:

- `a1`, `center1`: slope and center of the first sigmoid.
- `a2`, `center2`: slope and center of the second sigmoid.

### SShapedMF

Smooth S-shaped membership from 0 to 1 using a cubic smoothstep.

Parameters:

- `a`: start of the transition.
- `b`: end of the transition, with $a < b$.

### LinSShapedMF

Linear S-shaped membership from 0 to 1.

Parameters:

- `a`: start of the transition.
- `b`: end of the transition, with $a < b$.

### ZShapedMF

Smooth Z-shaped membership from 1 to 0 using an inverted smoothstep.

Parameters:

- `a`: start of the transition.
- `b`: end of the transition, with $a < b$.

### LinZShapedMF

Linear Z-shaped membership from 1 to 0.

Parameters:

- `a`: start of the transition.
- `b`: end of the transition, with $a < b$.

### PiMF

Pi-shaped membership function with smooth rising and falling edges.

Parameters:

- `a`, `b`, `c`, `d`: four boundary points satisfying $a < b \le c < d$.

Notes:

- Combines a smooth S-shaped left edge, a flat plateau, and a smooth Z-shaped right edge.

### GaussianPIMF

Gaussian membership with a positive infimum controlled by `K`:

$$
\mu(x)=\exp\left(-K\left(1-\exp\left(-\frac{(x-m)^2}{2\sigma^2}\right)\right)\right)
$$

Parameters:

- `mean`: center $m$.
- `sigma`: spread parameter $\sigma > 0$.
- `K`: positive infimum control parameter with $0 < K \le 745$.
- `eps`: numeric stability constant.

Notes:

- The membership infimum is $\exp(-K) > 0$.
- Designed to prevent numeric underflow in high-dimensional softmin-based aggregation.

## Implementation details

- All classes inherit from `MembershipFunction`.
- Several classes use `softplus` reparameterization to enforce positive parameters.
- `_as_tensor` supports scalar or tensor inputs with the default torch dtype.

## Example

```python
from highfis import GaussianMF, PiMF, GaussianPIMF

gauss = GaussianMF(mean=0.0, sigma=1.0)
pi_mf = PiMF(a=0.0, b=0.5, c=1.0, d=1.5)
gauss_pimf = GaussianPIMF(mean=0.0, sigma=1.0, K=2.0)

values = gauss(x_column)
```
