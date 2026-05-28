"""Structural typing protocols for the highFIS optimiser layer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from torch import Tensor


class DGModelProtocol(Protocol):
    """Structural protocol for models that support three-phase DG training.

    Satisfied by :class:`~highfis.models.DGTSKClassifierModel`,
    :class:`~highfis.models.DGTSKRegressorModel`,
    :class:`~highfis.models.DGALETSKClassifierModel`, and
    :class:`~highfis.models.DGALETSKRegressorModel`.
    """

    def fit_dg_phase(
        self,
        x: Tensor,
        y: Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute phase-1 (data-guided) training."""
        ...

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
        ...

    def fit_finetune(
        self,
        x: Tensor,
        y: Tensor,
        *,
        freeze_antecedents: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute phase-3 (fine-tune) training."""
        ...


class PFRBModelProtocol(Protocol):
    """Protocol for DG-TSK models that support P-FRB consequent initialisation."""

    def init_consequents_from_labels(self, y: Tensor) -> None:
        """Initialise zero-order consequents from class labels (paper eq. 24)."""
        ...


class FirstOrderModelProtocol(Protocol):
    """Protocol for DG models that can convert to first-order consequents."""

    def convert_to_first_order(self) -> None:
        """Replace zero-order consequent layer with a first-order one."""
        ...


__all__: list[str] = [
    "DGModelProtocol",
    "FirstOrderModelProtocol",
    "PFRBModelProtocol",
]
