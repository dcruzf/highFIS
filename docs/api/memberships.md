# Membership Functions API

## Module

`highfis.memberships`

## Classes

### MembershipFunction

Base class for differentiable membership functions in PyTorch.

- Inherits from `torch.nn.Module`.
- Provides numerical-stability epsilon handling.

### GaussianMF

Gaussian membership function:

$$
\mu(x)=\exp\left(-\frac{(x-c)^2}{2\sigma^2}\right)
$$

Parameters:

- `mean`: center $c$.
- `sigma`: spread parameter $\sigma > 0$.
- `eps`: numerical stability constant.

Implementation notes:

- `sigma` is parameterized through softplus to remain positive during training.
- Both center and spread are trainable parameters.

### TriangularMF

Triangular membership function defined by three vertices:

$$
\mu(x) = \max\!\left(0,\;\min\!\left(\frac{x - l}{c - l},\;\frac{r - x}{r - c}\right)\right)
$$

Parameters:

- `left`: left vertex $l$.
- `center`: peak vertex $c$.
- `right`: right vertex $r$.
- Must satisfy $l \le c \le r$ with $l \neq r$.

### TrapezoidalMF

Trapezoidal membership function with a flat-top plateau:

$$
\mu(x) = \max\!\left(0,\;\min\!\left(\frac{x - a}{b - a},\;1,\;\frac{d - x}{d - c}\right)\right)
$$

Parameters:

- `a`, `b`, `c`, `d`: four vertices with $a \le b \le c \le d$ and $a \neq d$.

### BellMF

Generalized bell membership function:

$$
\mu(x) = \frac{1}{1 + \left|\frac{x - c}{a}\right|^{2b}}
$$

Parameters:

- `a`: width parameter $a > 0$ (softplus-reparameterized).
- `b`: slope parameter $b > 0$ (softplus-reparameterized).
- `center`: center $c$.

### SigmoidalMF

Sigmoidal membership function:

$$
\mu(x) = \frac{1}{1 + \exp(-a(x - c))}
$$

Parameters:

- `a`: slope control (positive = left-to-right, negative = right-to-left).
- `center`: inflection point $c$.

## Example

```python
from highfis import GaussianMF, TriangularMF, BellMF

gauss = GaussianMF(mean=0.0, sigma=1.0)
tri = TriangularMF(left=-1.0, center=0.0, right=1.0)
bell = BellMF(a=1.0, b=2.0, center=0.0)

values = gauss(x_column)
```
