r"""Gate activation functions for feature and rule selection in gated TSK models.

Gate functions map unbounded real-valued gate parameters to normalised
activation values in [0, 1] (or near it). They are used by
``highfis.layers.DGALETSKRuleLayer``, ``highfis.layers.DGTSKRuleLayer``,
and the gated consequent layers to soft-select relevant features and rules
during training.

Each gate is an `~torch.nn.Module` subclass of `BaseGate`.
The module-level singletons ``gate1`` ... ``gate4``, ``gate_m`` are
provided for backward compatibility.

Built-in gate classes:
    - `SigmoidGate`: sigmoid, $M(\lambda) = \sigma(\lambda)$.
    - `ExpGate`: squared-exponential, $M(\lambda) = 1 - e^{-k\lambda^2}$
      (``k=1`` standard, ``k=10`` DG-ALETSK enhanced).
    - `InvExpGate`: inverted-exponential, $M(\lambda) = e^{-\lambda^2}$
      (inverted: open at 0, closes as $|\lambda| \to \infty$).
    - `SignedExpGate`: signed exponential,
      $M(\lambda) = \lambda\sqrt{e^{1 - \lambda^2}}$ (odd — **consequent-only**).
    - `MGate`: M-gate, $M(\lambda) = \lambda^2 e^{1 - \lambda^2}$.

Registry and resolver:
    - ``GATE_FNS``: mapping from string name to `BaseGate` singleton.
    - ``resolve_gate_fn``: resolve a name, callable, or ``None`` to a gate;
      ``None`` defaults to `ExpGate` with ``k=10``.

References:
    Xue, G., Wang, J., Yuan, B., and Dai, C. (2023). DG-ALETSK: A
        High-Dimensional Fuzzy Approach With Simultaneous Feature Selection
        and Rule Extraction. *IEEE Transactions on Fuzzy Systems*, 31(11),
        3866-3880. https://doi.org/10.1109/TFUZZ.2023.3270445

    Xue, G., Wang, J., Zhang, B., Yuan, B., and Dai, C. (2023). Double
        groups of gates based Takagi-Sugeno-Kang (DG-TSK) fuzzy system for
        simultaneous feature selection and rule extraction. *Fuzzy Sets and
        Systems*, 469, 108627. https://doi.org/10.1016/j.fss.2023.108627
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class BaseGate(nn.Module):
    """Abstract base for gate activation modules.

    Subclasses must implement :meth:`forward` and may override
    :meth:`init_params_` to provide a paper-recommended initialisation
    strategy for their gate parameters.

    Attributes:
        is_nonneg (bool): ``True`` when ``forward`` is guaranteed to return
            non-negative values. ``False`` for `SignedExpGate`.
    """

    is_nonneg: bool = True

    def forward(self, u: Tensor) -> Tensor:
        """Compute gate activation.

        Args:
            u (Tensor): Input gate parameter tensor.

        Returns:
            Tensor: Gate activation values.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise *param* in-place using the paper-recommended strategy.

        Args:
            param (nn.Parameter): Gate parameter tensor to initialise.
        """
        nn.init.uniform_(param, 0.01, 0.1)


class SigmoidGate(BaseGate):
    r"""Sigmoid gate activation.

    DG-TSK eq (16) / DG-ALETSK eq (15). Initialised near -5
    (Uniform(-5.5, -4.5)), giving M ≈ 0.007 (nearly closed)
    at the start of training.

    Mathematical definition:
        $$M(\lambda) = \sigma(\lambda) = \frac{1}{1 + e^{-\lambda}}$$
    """

    is_nonneg = True

    def forward(self, u: Tensor) -> Tensor:
        """Compute sigmoid gate activation.

        Args:
            u (Tensor): Input gate parameter tensor.

        Returns:
            Tensor: Gate activation values in (0, 1).
        """
        return torch.sigmoid(u)

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise *param* near -5 so gates start nearly closed.

        Args:
            param (nn.Parameter): Gate parameter tensor to initialise.
        """
        nn.init.uniform_(param, -5.5, -4.5)


class ExpGate(BaseGate):
    r"""Squared-exponential gate activation.

    DG-TSK eq (17) / DG-ALETSK eq (16). The *enhanced gate* used in
    DG-ALETSK is ``ExpGate(k=10)``. Initialised near zero
    (Uniform(0.001, 0.01)), giving M ≈ 0 (nearly closed)
    at the start of training.

    Mathematical definition:
        $$M(\lambda) = 1 - e^{-k \lambda^2}$$

    Attributes:
        k (float): Scale parameter.
    """

    is_nonneg = True

    def __init__(self, k: float = 1.0) -> None:
        """Initialise ExpGate.

        Args:
            k (float): Scale parameter (default ``1.0``).
        """
        super().__init__()
        self.k = float(k)

    def forward(self, u: Tensor) -> Tensor:
        """Compute squared-exponential gate activation.

        Args:
            u (Tensor): Input gate parameter tensor.

        Returns:
            Tensor: Gate activation values in [0, 1).
        """
        return 1.0 - torch.exp(-self.k * u.pow(2))

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise *param* near zero so gates start nearly closed.

        Args:
            param (nn.Parameter): Gate parameter tensor to initialise.
        """
        nn.init.uniform_(param, 0.001, 0.01)


class InvExpGate(BaseGate):
    r"""Inverted-exponential gate activation.

    DG-TSK eq (18) / DG-ALETSK eq (17). Initialised at large values
    (Normal(3.0, 0.2)), giving M = e⁻⁹ ≈ 1e-4 (nearly closed)
    at the start of training.

    Mathematical definition:
        $$M(\lambda) = e^{-\lambda^2}$$

    Warning:
        This gate has **inverted semantics** — M=1 at λ=0 (fully open)
        and M → 0 as |λ| → ∞ (closed). Parameters must be initialised
        at *large* values so gates start closed.
    """

    is_nonneg = True

    def forward(self, u: Tensor) -> Tensor:
        """Compute inverted-exponential gate activation.

        Args:
            u (Tensor): Input gate parameter tensor.

        Returns:
            Tensor: Gate activation values in (0, 1].
        """
        return torch.exp(-u.pow(2))

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise *param* near 3.0 so gates start nearly closed.

        Args:
            param (nn.Parameter): Gate parameter tensor to initialise.
        """
        nn.init.normal_(param, mean=3.0, std=0.2)


class SignedExpGate(BaseGate):
    r"""Signed exponential gate activation.

    DG-TSK eq (19) / DG-ALETSK eq (18). Initialised to small positive
    values (Uniform(0.005, 0.015)) to avoid negating rule outputs at
    the start of training.

    Mathematical definition:
        $$M(\lambda) = \lambda \sqrt{e^{1 - \lambda^2}}$$

    Warning:
        This is an **odd function** with range (-1, 1] that can return
        **negative values**, making it **unsuitable for antecedent feature
        selection** ($\mu^{M(\lambda)} > 1$ when $M(\lambda) < 0$,
        violating fuzzy set theory). Use only in consequent layers where
        ±1 both represent an open gate.
    """

    is_nonneg = False

    def forward(self, u: Tensor) -> Tensor:
        """Compute signed exponential gate activation.

        Args:
            u (Tensor): Input gate parameter tensor.

        Returns:
            Tensor: Gate activation values in (-1, 1].
        """
        return u * torch.sqrt(torch.exp(1.0 - u.pow(2)))

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise *param* to small positive values (gates nearly closed).

        Args:
            param (nn.Parameter): Gate parameter tensor to initialise.
        """
        nn.init.uniform_(param, 0.005, 0.015)


class MGate(BaseGate):
    r"""M-gate activation.

    Introduced in Xue et al., *Fuzzy Sets and Systems*, 2023, eq (20).
    It is an even function with range [0, 1] and two maxima at λ = ±1,
    forming an M-shape. Its derivative near zero is larger than those of
    `SigmoidGate`, `ExpGate`, and `InvExpGate`,
    which speeds up early learning. Initialised to small values
    (Uniform(0.01, 0.1)), giving M ∈ [0.0003, 0.027] (nearly closed;
    even function so sign does not matter).

    Mathematical definition:
        $$M(\lambda) = \lambda^2 e^{1 - \lambda^2}$$
    """

    is_nonneg = True

    def forward(self, u: Tensor) -> Tensor:
        """Compute M-gate activation.

        Args:
            u (Tensor): Input gate parameter tensor.

        Returns:
            Tensor: Gate activation values in [0, 1].
        """
        return u.pow(2) * torch.exp(1.0 - u.pow(2))

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise *param* to small values so gates start nearly closed.

        Args:
            param (nn.Parameter): Gate parameter tensor to initialise.
        """
        nn.init.uniform_(param, 0.01, 0.1)


# ---------------------------------------------------------------------------
# Module-level singleton instances — backward-compatible names
# ---------------------------------------------------------------------------

gate1: SigmoidGate = SigmoidGate()
gate2: ExpGate = ExpGate(k=1.0)
gate3: InvExpGate = InvExpGate()
gate4: SignedExpGate = SignedExpGate()
gate_m: MGate = MGate()

GATE_FNS: dict[str, BaseGate] = {
    "gate1": gate1,
    "gate2": gate2,
    "gate3": gate3,
    "gate4": gate4,
    "gate_m": gate_m,
}


def resolve_gate_fn(
    gate_fn: str | Callable[[Tensor], Tensor] | None,
) -> BaseGate | Callable[[Tensor], Tensor]:
    """Resolve a gate name or callable to a gate.

    Args:
        gate_fn (str | Callable[[Tensor], Tensor] | None): A string key
            from ``GATE_FNS``, a callable ``Tensor → Tensor`` (including
            any `BaseGate` instance), or ``None`` (defaults to
            `ExpGate` with ``k=10``).

    Returns:
        BaseGate | Callable[[Tensor], Tensor]: A `BaseGate`
        singleton for known string keys, an `ExpGate` ``(k=10)``
        for ``None``, or the original callable unchanged.

    Raises:
        ValueError: If ``gate_fn`` is an unrecognised string.
    """
    if gate_fn is None:
        return ExpGate(k=10.0)
    if isinstance(gate_fn, str):
        try:
            return GATE_FNS[gate_fn]
        except KeyError as exc:
            raise ValueError(f"unsupported gate function '{gate_fn}'") from exc
    return gate_fn
