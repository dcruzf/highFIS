"""Sklearn-compatible estimators for FSRE-ADATSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseTSK
from ..memberships import MembershipFunction
from ..models import (
    FSREADATSKClassifierModel,
    FSREADATSKRegressorModel,
)
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
)


class FSREADATSKClassifier(_BaseClassifierEstimator):
    r"""FSRE-ADATSK classifier with adaptive softmin antecedent and gated consequents.

    FSRE-ADATSK (Feature Selection and Rule Extraction) extends ADATSK.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import FSREADATSKClassifier

        clf = FSREADATSKClassifier()
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
        """Initialise an FSRE-ADATSK classifier.

        Args:
            lambda_init: Accepted for API compatibility but not used by
                FSRE-ADATSK or DG-ALETSK.  FSRE-ADATSK computes its
                adaptive softmin index directly from membership values;
                DG-ALETSK uses the fixed exponent ``ξ = 700`` per
                paper eq. 22.  Default ``1.0``.
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) whose
                size grows linearly with the number of features, allowing
                more candidate rules for the RE phase. Xue et al. (2023)
                activate En-FRB after the FS phase; set ``False`` (default)
                to keep the compact CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
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
        """Create FSREADATSKClassifierModel."""
        return FSREADATSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )


class FSREADATSKRegressor(_BaseRegressorEstimator):
    r"""FSRE-ADATSK regressor with adaptive softmin antecedent and gated consequents.

    FSRE-ADATSK (Feature Selection and Rule Extraction) extends ADATSK.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import FSREADATSKRegressor

        reg = FSREADATSKRegressor()
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
        """Initialise an FSRE-ADATSK regressor.

        Args:
            lambda_init: Accepted for API compatibility but not used by
                FSRE-ADATSK or DG-ALETSK.  FSRE-ADATSK computes its
                adaptive softmin index directly from membership values;
                DG-ALETSK uses the fixed exponent ``ξ = 700`` per
                paper eq. 22.  Default ``1.0``.
            use_en_frb: If ``True``, use the Enhanced FRB (En-FRB) for rule
                extraction. Default ``False`` keeps CoCo-FRB.
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
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
        """Create FSREADATSKRegressorModel."""
        return FSREADATSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
        )
