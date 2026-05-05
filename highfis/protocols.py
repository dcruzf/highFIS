"""Runtime-checkable structural typing protocols for the highFIS pipeline.

All protocols are decorated with ``@runtime_checkable`` so they can be used
with :func:`isinstance` at runtime.

Protocols:

- :class:`MembershipFn` — any callable ``(x: Tensor) -> Tensor`` that
  computes membership degrees.  Satisfied by all classes in
  :mod:`highfis.memberships`.
- :class:`TNorm` — any callable ``(terms: Tensor) -> Tensor`` that
  aggregates antecedent activations.  Satisfied by T-norm classes in
  :mod:`highfis.t_norms`.
- :class:`Defuzzifier` — any callable ``(w: Tensor) -> Tensor`` that
  normalizes rule firing strengths.  Satisfied by all classes in
  :mod:`highfis.defuzzifiers`.
- :class:`ConsequentFn` — any callable
  ``(x: Tensor, norm_w: Tensor) -> Tensor`` that computes the model output.
  Satisfied by consequent layers in :mod:`highfis.layers`.

Examples:
    >>> from highfis.protocols import MembershipFn
    >>> from highfis import GaussianMF
    >>> mf = GaussianMF(mean=0.0, sigma=1.0)
    >>> isinstance(mf, MembershipFn)
    True
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class MembershipFn(Protocol):
    """Callable that maps an input tensor to membership degrees."""

    def __call__(self, x: Tensor) -> Tensor:
        """Compute membership degrees for *x*."""
        ...


@runtime_checkable
class TNorm(Protocol):
    """Callable that aggregates antecedent term activations per rule."""

    def __call__(self, terms: Tensor) -> Tensor:
        """Aggregate *terms* into a single firing strength."""
        ...


@runtime_checkable
class Defuzzifier(Protocol):
    """Callable that normalizes rule firing strengths."""

    def __call__(self, w: Tensor) -> Tensor:
        """Normalize firing strengths *w*."""
        ...


@runtime_checkable
class ConsequentFn(Protocol):
    """Callable that computes consequent output from inputs and normalized strengths."""

    def __call__(self, x: Tensor, norm_w: Tensor) -> Tensor:
        """Compute consequent output from *x* and *norm_w*."""
        ...


__all__: list[str] = ["ConsequentFn", "Defuzzifier", "MembershipFn", "TNorm"]
