r"""Gate activation functions for feature and rule selection in gated TSK models.

Gate functions map unbounded real-valued gate parameters to normalised
activation values in [0, 1] (or near it).  They are used by
``highfis.layers.DGALETSKRuleLayer``, ``highfis.layers.DGTSKRuleLayer``,
and the gated consequent layers to soft-select relevant features and rules
during training.

Built-in gate functions:
    - ``gate1`` — sigmoid: :math:`\\sigma(u)`.
    - ``gate2`` — :math:`1 - e^{-u^2}`.
    - ``gate3`` — :math:`e^{-u^2}`.
    - ``gate4`` — :math:`u \\sqrt{e^{1 - u^2}}` (default for DG-ALETSK).
    - ``gate_m`` — :math:`u^2 e^{1 - u^2}` (M-gate, default for DG-TSK).

Registry and resolver:
    - ``GATE_FNS`` — mapping from string name to gate callable.
    - ``resolve_gate_fn`` — resolve a name, callable, or ``None`` to a gate.

Notes:
    ``_gate_activation`` is the default gate used internally by
    DG-ALETSK layers (wraps ``gate4``).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def gate1(u: Tensor) -> Tensor:
    """Compute the sigmoid gate activation."""
    return torch.sigmoid(u)


def gate2(u: Tensor) -> Tensor:
    """Compute the gate activation 1 - exp(-x^2)."""
    return 1.0 - torch.exp(-u.pow(2))


def gate3(u: Tensor) -> Tensor:
    """Compute the gate activation exp(-x^2)."""
    return torch.exp(-u.pow(2))


def gate4(u: Tensor) -> Tensor:
    """Compute the gate activation x * sqrt(exp(1 - x^2))."""
    return u * torch.sqrt(torch.exp(1.0 - u.pow(2)))


def gate_m(u: Tensor) -> Tensor:
    """Compute the gate activation x^2 * exp(1 - x^2)."""
    return u.pow(2) * torch.exp(1.0 - u.pow(2))


GATE_FNS: dict[str, Callable[[Tensor], Tensor]] = {
    "gate1": gate1,
    "gate2": gate2,
    "gate3": gate3,
    "gate4": gate4,
    "gate_m": gate_m,
}


def resolve_gate_fn(gate_fn: str | Callable[[Tensor], Tensor] | None) -> Callable[[Tensor], Tensor]:
    """Resolve a gate name or function to a callable gate function.

    Args:
        gate_fn: A string key from :data:`GATE_FNS`, a callable
            ``Tensor → Tensor``, or ``None`` (defaults to ``gate4``).

    Returns:
        A gate callable.

    Raises:
        ValueError: If ``gate_fn`` is an unrecognised string.
    """
    if gate_fn is None:
        return gate4
    if isinstance(gate_fn, str):
        try:
            return GATE_FNS[gate_fn]
        except KeyError as exc:
            raise ValueError(f"unsupported gate function '{gate_fn}'") from exc
    return gate_fn


def _gate_activation(u: Tensor) -> Tensor:
    """Default feature gate activation used by DG-ALETSK and related models."""
    return gate4(u)
