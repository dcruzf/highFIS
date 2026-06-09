"""Structural typing protocols for the highFIS optimiser layer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from torch import Tensor


class DGModelProtocol(Protocol):
    """Structural protocol for models that support three-phase DG training.

    Satisfied by DGTSKClassifierModel, DGTSKRegressorModel,
    DGALETSKClassifierModel, and DGALETSKRegressorModel.
    """

    def search_thresholds(
        self,
        x: Tensor,
        y: Tensor,
        zeta_lambda: Sequence[float] | None = None,
        zeta_theta: Sequence[float] | None = None,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
        use_lse: bool = True,
        inplace: bool = True,
        verbose: bool = False,
        structural: bool = True,
    ) -> dict[str, Any]:
        """Search pruning thresholds and optionally apply the best candidate."""
        ...  # pragma: no cover


class PFRBModelProtocol(Protocol):
    """Protocol for DG-TSK models that support P-FRB consequent initialisation."""

    def init_consequents_from_labels(self, y: Tensor) -> None:
        """Initialise zero-order consequents from class labels (paper eq. 24)."""
        ...  # pragma: no cover


class FirstOrderModelProtocol(Protocol):
    """Protocol for DG models that can convert to first-order consequents."""

    def convert_to_first_order(self) -> None:
        """Replace zero-order consequent layer with a first-order one."""
        ...  # pragma: no cover


class _HasConsequentMode(Protocol):
    """Protocol for consequent layers that expose a ``mode`` attribute."""

    mode: str


class FSREModelProtocol(Protocol):
    """Structural protocol for models that support three-phase FSRE training.

    Satisfied by FSREADATSKClassifierModel and FSREADATSKRegressorModel.
    """

    consequent_layer: _HasConsequentMode

    def get_feature_gate_values(self) -> Tensor:
        """Return M(λ_d) gate activations of shape ``(n_inputs,)``."""
        ...  # pragma: no cover

    def get_rule_gate_values(self) -> Tensor:
        """Return M(θ_r) gate activations of shape ``(n_rules,)``."""
        ...  # pragma: no cover

    def expand_to_en_frb(self) -> None:
        """Switch to Enhanced Fuzzy Rule Base for the RE phase."""
        ...  # pragma: no cover

    def prune_to_features(self, surviving_features: list[int]) -> None:
        """Prune the model architecture to the given feature indices."""
        ...  # pragma: no cover

    def prune_to_rules(self, surviving_rules: list[int]) -> None:
        """Prune the model architecture to the given rule indices."""
        ...  # pragma: no cover


__all__: list[str] = [
    "DGModelProtocol",
    "FSREModelProtocol",
    "FirstOrderModelProtocol",
    "PFRBModelProtocol",
]
