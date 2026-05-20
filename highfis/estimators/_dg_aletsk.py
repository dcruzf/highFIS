"""Sklearn-compatible estimators for DG-ALETSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseTSK
from ..memberships import MembershipFunction
from ..models import (
    DGALETSKClassifierModel,
    DGALETSKRegressorModel,
)
from ._fsre import (
    FSREADATSKClassifier,
    FSREADATSKRegressor,
)


class DGALETSKClassifier(FSREADATSKClassifier):
    """DG-ALETSK classifier with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-ADATSK by replacing the adaptive softmin with the
    *Adaptive Ln-Exp (ALE)* softmin — a smoother variant with improved
    numerical stability.  It also uses a zero-order consequent in the DG
    (data-guided) training phase and optionally converts to first-order
    after gate-based pruning.

    Reference:
        G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional
        Fuzzy Approach With Simultaneous Feature Selection and Rule
        Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no.
        11, pp. 3866-3880, Nov. 2023, doi: 10.1109/TFUZZ.2023.3270445.

    Example:
        ```python
        from highfis import DGALETSKClassifier

        clf = DGALETSKClassifier(
            n_mfs=30, lambda_init=1.0, use_en_frb=False, random_state=0
        )
        clf.fit(X_train, y_train)
        ```
    """

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create DGALETSKClassifierModel."""
        return DGALETSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            lambda_init=self.lambda_init,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class DGALETSKRegressor(FSREADATSKRegressor):
    """DG-ALETSK regressor with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-ADATSK by replacing the adaptive softmin with the
    *Adaptive Ln-Exp (ALE)* softmin — a smoother variant with improved
    numerical stability.  It also uses a zero-order consequent in the DG
    (data-guided) training phase and optionally converts to first-order
    after gate-based pruning.

    Reference:
        G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional
        Fuzzy Approach With Simultaneous Feature Selection and Rule
        Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no.
        11, pp. 3866-3880, Nov. 2023, doi: 10.1109/TFUZZ.2023.3270445.

    Example:
        ```python
        from highfis import DGALETSKRegressor

        reg = DGALETSKRegressor(
            n_mfs=30, lambda_init=1.0, use_en_frb=False, random_state=0
        )
        reg.fit(X_train, y_train)
        ```
    """

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create DGALETSKRegressorModel."""
        return DGALETSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            lambda_init=self.lambda_init,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )
