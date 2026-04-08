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

## Example

```python
from highfis import GaussianMF

mf = GaussianMF(mean=0.0, sigma=1.0)
values = mf(x_column)
```
