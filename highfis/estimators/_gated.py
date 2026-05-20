"""Sklearn-compatible estimators for FSRE-AdaTSK, DG-ALETSK and DG-TSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseTSK
from ..memberships import (
    MembershipFunction,
)
from ..models import (
    DGALETSKClassifier,
    DGALETSKRegressor,
    DGTSKClassifier,
    DGTSKRegressor,
    FSREAdaTSKClassifier,
    FSREAdaTSKRegressor,
)
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
)


class FSREAdaTSKClassifierEstimator(_BaseClassifierEstimator):
    r"""FSRE-AdaTSK classifier with adaptive softmin antecedent and gated consequents.

    FSRE-AdaTSK (Feature Selection and Rule Extraction) extends AdaTSK.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import FSREAdaTSKClassifierEstimator

        clf = FSREAdaTSKClassifierEstimator()
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        lambda_init: float = 1.0,
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
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an FSRE-AdaTSK classifier.

        Args:
            lambda_init: Initial ALE-softmin parameter ``λ > 0`` inherited
                by :class:`DGALETSKClassifierEstimator`; not used by
                FSRE-AdaTSK proper (Ada-softmin computes its index from
                the current membership values). Default ``1.0``.
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) whose
                size grows linearly with the number of features, allowing
                more candidate rules for the RE phase. Xue et al. (2023)
                activate En-FRB after the FS phase; set ``False`` (default)
                to keep the compact CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.

        Raises:
            ValueError: If ``lambda_init <= 0``.
        """
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")
        self.lambda_init = float(lambda_init)
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
        """Create FSREAdaTSKClassifier."""
        return FSREAdaTSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class FSREAdaTSKRegressorEstimator(_BaseRegressorEstimator):
    r"""FSRE-AdaTSK regressor with adaptive softmin antecedent and gated consequents.

    FSRE-AdaTSK (Feature Selection and Rule Extraction) extends AdaTSK.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import FSREAdaTSKRegressorEstimator

        reg = FSREAdaTSKRegressorEstimator()
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        lambda_init: float = 1.0,
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
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
    ) -> None:
        """Initialise an FSRE-AdaTSK regressor.

        Args:
            lambda_init: Initial ALE-softmin parameter ``λ > 0`` inherited
                by :class:`DGALETSKRegressorEstimator`; not used by
                FSRE-AdaTSK proper (Ada-softmin computes its index from
                the current membership values). Default ``1.0``.
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) for rule
                extraction. Default ``False`` keeps CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            epochs: Maximum training epochs (default ``10``).
            learning_rate: Adam learning rate (default ``0.01``).
            verbose: Print per-epoch progress.
            rule_base: ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            patience: Early-stopping patience (default ``20``). Set to ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.

        Raises:
            ValueError: If ``lambda_init <= 0``.
        """
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")
        self.lambda_init = float(lambda_init)
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
        """Create FSREAdaTSKRegressor."""
        return FSREAdaTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class DGALETSKClassifierEstimator(FSREAdaTSKClassifierEstimator):
    """DG-ALETSK classifier with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-AdaTSK by replacing the adaptive softmin with the
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
        from highfis import DGALETSKClassifierEstimator

        clf = DGALETSKClassifierEstimator(
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
        """Create DGALETSKClassifier."""
        return DGALETSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            lambda_init=self.lambda_init,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class DGALETSKRegressorEstimator(FSREAdaTSKRegressorEstimator):
    """DG-ALETSK regressor with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-AdaTSK by replacing the adaptive softmin with the
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
        from highfis import DGALETSKRegressorEstimator

        reg = DGALETSKRegressorEstimator(
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
        """Create DGALETSKRegressor."""
        return DGALETSKRegressor(
            input_mfs,
            rule_base=rule_base,
            lambda_init=self.lambda_init,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class DGTSKClassifierEstimator(_BaseClassifierEstimator):
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
        from highfis import DGTSKClassifierEstimator

        clf = DGTSKClassifierEstimator(n_mfs=30, use_en_frb=False, random_state=0)
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
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
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
        """Create DGTSKClassifier."""
        return DGTSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class DGTSKRegressorEstimator(_BaseRegressorEstimator):
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
        from highfis import DGTSKRegressorEstimator

        reg = DGTSKRegressorEstimator(n_mfs=30, use_en_frb=False, random_state=0)
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
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
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
        """Create DGTSKRegressor."""
        return DGTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


# =====================================================================
# LogTSK Estimators  (Cui, Wu & Xu, IEEE TFS 2021)
# =====================================================================
