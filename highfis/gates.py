r"""Gate activation functions for feature and rule selection in gated TSK models.

Gate functions map unbounded real-valued gate parameters to normalised
activation values in [0, 1] (or near it).  They are used by
``highfis.layers.DGALETSKRuleLayer``, ``highfis.layers.DGTSKRuleLayer``,
and the gated consequent layers to soft-select relevant features and rules
during training.

Each gate is an :class:`~torch.nn.Module` subclass of :class:`BaseGate`.
The module-level singletons ``gate1`` … ``gate4``, ``gate_m`` are
provided for backward compatibility.

Built-in gate classes:
    - :class:`SigmoidGate` — :math:`\sigma(u)`.
    - :class:`ExpGate` — :math:`1 - e^{-k u^2}` (``k=1`` standard, ``k=10`` DG-ALETSK enhanced).
    - :class:`InvExpGate` — :math:`e^{-u^2}` (*inverted*: open at 0, closes as :math:`|u| \to \infty`).
    - :class:`SignedExpGate` — :math:`u \sqrt{e^{1 - u^2}}` (odd function — **consequent-only**).
    - :class:`MGate` — :math:`u^2 e^{1 - u^2}` (M-gate from the DG-TSK paper).

Registry and resolver:
    - ``GATE_FNS`` — mapping from string name to :class:`BaseGate` singleton instance.
    - ``resolve_gate_fn`` — resolve a name, callable, or ``None`` to a gate.
      ``None`` defaults to :class:`ExpGate` with ``k=10``.
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
        is_nonneg: ``True`` when ``forward`` is guaranteed to return
            non-negative values.  ``False`` for :class:`SignedExpGate`.
    """

    is_nonneg: bool = True

    def forward(self, u: Tensor) -> Tensor:
        """Compute gate activation."""
        raise NotImplementedError

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise *param* in-place using the paper-recommended strategy.

        Args:
            param: Gate parameter tensor to initialise.
        """
        nn.init.uniform_(param, 0.01, 0.1)


class SigmoidGate(BaseGate):
    r"""Sigmoid gate: :math:`\sigma(\lambda)`.

    Paper reference: DG-TSK eq (16) / DG-ALETSK eq (15).

    Initialisation: :math:`\lambda \sim \text{Uniform}(-5.5,\,-4.5)`,
    giving :math:`M \approx 0.007` (nearly closed).
    """

    is_nonneg = True

    def forward(self, u: Tensor) -> Tensor:
        """Compute sigmoid gate activation."""
        return torch.sigmoid(u)

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise params near -5 (gates nearly closed)."""
        nn.init.uniform_(param, -5.5, -4.5)


class ExpGate(BaseGate):
    r"""Squared-exponential gate: :math:`M(\lambda) = 1 - e^{-k \lambda^2}`.

    Paper reference: DG-TSK eq (17) / DG-ALETSK eq (16).
    The *enhanced gate* used in DG-ALETSK is ``ExpGate(k=10)``.

    Args:
        k: Scale parameter (default ``1.0``).

    Initialisation: :math:`\lambda \sim \text{Uniform}(0.001,\,0.01)`,
    giving :math:`M \approx 0` (nearly closed).
    """

    is_nonneg = True

    def __init__(self, k: float = 1.0) -> None:
        """Initialise ExpGate with scale parameter *k*."""
        super().__init__()
        self.k = float(k)

    def forward(self, u: Tensor) -> Tensor:
        """Compute squared-exponential gate activation."""
        return 1.0 - torch.exp(-self.k * u.pow(2))

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise params near zero (gates nearly closed)."""
        nn.init.uniform_(param, 0.001, 0.01)


class InvExpGate(BaseGate):
    r"""Inverted-exponential gate: :math:`M(\lambda) = e^{-\lambda^2}`.

    Paper reference: DG-TSK eq (18) / DG-ALETSK eq (17).

    .. warning::
        This gate has **inverted semantics** — :math:`M=1` at
        :math:`\lambda=0` (fully open) and :math:`M \to 0` as
        :math:`|\lambda| \to \infty` (closed).  Parameters must be
        initialised at *large* values so gates start closed.

    Initialisation: :math:`\lambda \sim \mathcal{N}(3.0,\,0.2)`,
    giving :math:`M = e^{-9} \approx 10^{-4}` (nearly closed).
    """

    is_nonneg = True

    def forward(self, u: Tensor) -> Tensor:
        """Compute inverted-exponential gate activation."""
        return torch.exp(-u.pow(2))

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise params near 3.0 (gates nearly closed)."""
        nn.init.normal_(param, mean=3.0, std=0.2)


class SignedExpGate(BaseGate):
    r"""Signed exponential gate: :math:`M(\lambda) = \lambda \sqrt{e^{1 - \lambda^2}}`.

    Paper reference: DG-TSK eq (19) / DG-ALETSK eq (18).

    .. warning::
        This is an **odd function** with range :math:`(-1, 1]`.  It can
        return **negative values**, making it **unsuitable for antecedent
        feature selection** (:math:`\mu^{M(\lambda)} > 1` when
        :math:`M(\lambda) < 0`, violating fuzzy set theory).  Use only in
        consequent layers where :math:`\pm 1` both represent an open gate.

    Initialisation: :math:`\lambda \sim \text{Uniform}(0.005,\,0.015)`
    (positive-only, to avoid negating rule outputs at the start of training).
    """

    is_nonneg = False

    def forward(self, u: Tensor) -> Tensor:
        """Compute signed exponential gate activation."""
        return u * torch.sqrt(torch.exp(1.0 - u.pow(2)))

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise params to small positive values (gates nearly closed, non-negative)."""
        nn.init.uniform_(param, 0.005, 0.015)


class MGate(BaseGate):
    r"""M-gate: :math:`M(\lambda) = \lambda^2 e^{1 - \lambda^2}`.

    The M-gate introduced in the DG-TSK paper (Xue et al., *Fuzzy Sets and
    Systems*, 2023, eq (20)).  It is an even function with range
    :math:`[0, 1]` and two maxima at :math:`\lambda = \pm 1`, forming an
    M-shape.  Its derivative near zero is larger than those of
    :class:`SigmoidGate`, :class:`ExpGate`, and :class:`InvExpGate`,
    which speeds up early learning.

    Initialisation: :math:`\lambda \sim \text{Uniform}(0.01,\,0.1)`,
    giving :math:`M \in [0.0003,\,0.027]` (nearly closed; even function so
    sign does not matter).
    """

    is_nonneg = True

    def forward(self, u: Tensor) -> Tensor:
        """Compute M-gate activation."""
        return u.pow(2) * torch.exp(1.0 - u.pow(2))

    def init_params_(self, param: nn.Parameter) -> None:
        """Initialise params to small values (gates nearly closed)."""
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
        gate_fn: A string key from :data:`GATE_FNS`, a callable
            ``Tensor → Tensor`` (including any :class:`BaseGate` instance),
            or ``None`` (defaults to :class:`ExpGate` with ``k=10``).

    Returns:
        A :class:`BaseGate` singleton for known string keys, an
        :class:`ExpGate` ``(k=10)`` for ``None``, or the original callable
        unchanged for any other callable.

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
