"""Structural typing protocols for the highFIS fuzzy inference pipeline."""

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
