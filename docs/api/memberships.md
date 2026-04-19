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

### DiffSigmoidalMF

Difference of two sigmoids leading to a localized activation interval.

Parameters:

- `a1`, `center1`: slope and center of the first sigmoid.
- `a2`, `center2`: slope and center of the second sigmoid.

### ProdSigmoidalMF

Product of two sigmoids, producing a smooth window between thresholds.

Parameters:

- `a1`, `center1`: slope and center of the first sigmoid.
- `a2`, `center2`: slope and center of the second sigmoid.

### SShapedMF

Smooth S-shaped membership from 0 to 1 using a cubic smoothstep.

Parameters:

- `a`: start of the transition.
- `b`: end of the transition with $a < b$.

### LinSShapedMF

Linear S-shaped membership from 0 to 1.

Parameters:

- `a`: start of the transition.
- `b`: end of the transition with $a < b$.

### ZShapedMF

Smooth Z-shaped membership from 1 to 0 using an inverted cubic smoothstep.

Parameters:

- `a`: start of the transition.
- `b`: end of the transition with $a < b$.

### LinZShapedMF

Linear Z-shaped membership from 1 to 0.

Parameters:

- `a`: start of the transition.
- `b`: end of the transition with $a < b$.

### PiMF

Pi-shaped membership with smooth S/Z transitions and a flat top region.

Parameters:

- `a`, `b`, `c`, `d`: four boundary points with $a < b \le c < d$.

### GaussianPIMF

Gaussian membership with a positive infimum controlled by parameter `K`, which can help avoid underflow in high-dimensional softmin-based aggregation.

Parameters:

- `mean`: center $c$.
- `sigma`: spread parameter $\sigma > 0$.
- `K`: positive infimum control parameter, with $	ext{infimum} = e^{-K}$.

## Example

```python
from highfis import GaussianMF, PiMF, GaussianPIMF

gauss = GaussianMF(mean=0.0, sigma=1.0)
pi_mf = PiMF(a=0.0, b=0.5, c=1.0, d=1.5)
gauss_pimf = GaussianPIMF(mean=0.0, sigma=1.0, K=2.0)

values = gauss(x_column)
```
