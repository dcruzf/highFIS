"""Sklearn-compatible estimators for DG-TSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseTSK
from ..memberships import MembershipFunction
from ..models import (
    DGTSKClassifierModel,
    DGTSKRegressorModel,
)
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
)


class DGTSKClassifier(_BaseClassifierEstimator):
    """DG-TSK classifier with M-gate antecedent and point-based FRB (P-FRB).

    DG-TSK uses a data-guided M-gate function to automatically select
    relevant features and rules.

    Reference:
        Guangdong Xue, Jian Wang, Bingjie Zhang, Bin Yuan, Caili Dai,
        Double groups of gates based Takagi-Sugeno-Kang (DG-TSK)
        fuzzy system for simultaneous feature selection and rule
        extraction, Fuzzy Sets and Systems, Volume 469, 2023, 108627,
        ISSN 0165-0114, https://doi.org/10.1016/j.fss.2023.108627.

    Example:
        ```python
        from highfis import DGTSKClassifier

        clf = DGTSKClassifier(n_mfs=30, use_en_frb=False, random_state=0)
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a DG-TSK classifier.

        Args:
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) for rule
                extraction (P-FRB). Default ``False`` keeps CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum number of point-based FRB rules when
                ``rule_base='pfrb'``. ``None`` uses all training samples.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.
        """
        self.use_en_frb = bool(use_en_frb)
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            weight_decay=weight_decay,
        )

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        """Create DGTSKClassifierModel."""
        return DGTSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class DGTSKRegressor(_BaseRegressorEstimator):
    """DG-TSK regressor with M-gate antecedent and point-based FRB (P-FRB).

    DG-TSK uses a data-guided M-gate function to automatically select
    relevant features and rules.

    Reference:
        Guangdong Xue, Jian Wang, Bingjie Zhang, Bin Yuan, Caili Dai,
        Double groups of gates based Takagi-Sugeno-Kang (DG-TSK)
        fuzzy system for simultaneous feature selection and rule
        extraction, Fuzzy Sets and Systems, Volume 469, 2023, 108627,
        ISSN 0165-0114, https://doi.org/10.1016/j.fss.2023.108627.

    Example:
        ```python
        from highfis import DGTSKRegressor

        reg = DGTSKRegressor(n_mfs=30, use_en_frb=False, random_state=0)
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        use_en_frb: bool = False,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 5,
        mf_init: str = "kmeans",
        sigma_scale: float | str = 1.0,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise a DG-TSK regressor.

        Args:
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) for rule
                extraction (P-FRB). Default ``False`` keeps CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for reproducibility.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum number of point-based FRB rules when
                ``rule_base='pfrb'``. ``None`` uses all training samples.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.
        """
        self.use_en_frb = bool(use_en_frb)
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            weight_decay=weight_decay,
        )

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        """Create DGTSKRegressorModel."""
        return DGTSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )
