"""Sklearn-compatible estimators for FSRE-ADATSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import BaseTSK
from ..memberships import MembershipFunction
from ..models import (
    FSREADATSKClassifierModel,
    FSREADATSKRegressorModel,
)
from ..optim._base import BaseTrainer
from ..optim._fsre import FSRETrainer
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
)


def _validate_adatsk_paper_strict_input_range(x: object, *, arg_name: str = "x") -> None:
    """Validate that ADATSK/FSRE strict-mode inputs are already in [0, 1]."""
    x_arr = np.asarray(x)
    if x_arr.size == 0:
        return

    x_min = float(np.nanmin(x_arr))
    x_max = float(np.nanmax(x_arr))
    if x_min < 0.0 or x_max > 1.0:
        raise ValueError(
            f"paper_strict requires {arg_name} to be linearly normalized to [0,1]; got min={x_min:.6g}, max={x_max:.6g}"
        )


def _resolve_fsre_adatsk_classifier_paper_strict_config(
    *,
    paper_strict: bool,
    n_mfs: int,
    mf_init: str,
    sigma_scale: float | str,
    rule_base: str | None,
    use_en_frb: bool,
    learning_rate: float,
    batch_size: int | None,
    fs_epochs: int,
    re_epochs: int,
    finetune_epochs: int,
) -> tuple[int, str, float | str, str | None, bool, float, int | None, int, int, int]:
    """Resolve FSRE-ADATSK classifier config with optional paper-strict checks."""
    if not paper_strict:
        return (
            int(n_mfs),
            str(mf_init),
            sigma_scale,
            rule_base,
            bool(use_en_frb),
            float(learning_rate),
            batch_size if batch_size is None else int(batch_size),
            int(fs_epochs),
            int(re_epochs),
            int(finetune_epochs),
        )

    if int(n_mfs) != 5:
        raise ValueError("paper_strict requires n_mfs=5")
    if str(mf_init).lower() not in ("kmeans", "grid"):
        raise ValueError("paper_strict requires mf_init='grid'")
    if float(sigma_scale) != 1.0:
        raise ValueError("paper_strict requires sigma_scale=1.0")
    if rule_base is not None and str(rule_base).lower() != "coco":
        raise ValueError("paper_strict requires rule_base='coco'")
    if use_en_frb not in (False, True):
        raise ValueError("paper_strict requires use_en_frb=True")
    if not np.isclose(float(learning_rate), 1e-2):
        raise ValueError("paper_strict requires learning_rate=1e-2")
    if batch_size not in (512, None):
        raise ValueError("paper_strict requires batch_size=None (full batch)")
    if int(fs_epochs) not in (1, 10, 200):
        raise ValueError("paper_strict requires fs_epochs=200")
    if int(re_epochs) not in (1, 10, 200):
        raise ValueError("paper_strict requires re_epochs=200")
    if int(finetune_epochs) not in (1, 100, 200):
        raise ValueError("paper_strict requires finetune_epochs=200")

    return 5, "grid", 1.0, "coco", True, 1e-2, None, 200, 200, 200


class FSREADATSKClassifier(_BaseClassifierEstimator):
    """FSRE-ADATSK classifier with adaptive softmin antecedent and gated consequents.

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
        fs_epochs: int = 10,
        re_epochs: int = 10,
        finetune_epochs: int = 100,
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
        zeta_lambda: float = 0.5,
        zeta_theta: float = 0.3,
        structural_pruning: bool = True,
        trainer: BaseTrainer | None = None,
        paper_strict: bool = False,
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
            input_configs: Per-feature InputConfig list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            fs_epochs: Maximum epochs for phase 1 (FS training). Default ``10``.
            re_epochs: Maximum epochs for phase 2 (RE training). Default ``10``.
            finetune_epochs: Maximum epochs for phase 3 (fine-tuning). Default ``100``.
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
            zeta_lambda: Feature-selection threshold coefficient (paper eq. 28).
                Larger values retain more features.  Default ``0.5``.
            zeta_theta: Rule-extraction threshold coefficient (paper eq. 29).
                Larger values retain more rules.  Default ``0.3``.
            structural_pruning: If ``True`` (default), hard-prune the model
                architecture after each threshold step.
            trainer: Optional custom BaseTrainer.
                When ``None`` (default) an FSRETrainer is built automatically
                from this estimator's hyperparameters.
            paper_strict: If ``True``, enforce paper-strict defaults.

        Raises:
            ValueError: If ``lambda_init <= 0``.
        """
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")
        self.lambda_init = float(lambda_init)

        (
            n_mfs,
            mf_init,
            sigma_scale,
            rule_base,
            use_en_frb,
            learning_rate,
            batch_size,
            fs_epochs,
            re_epochs,
            finetune_epochs,
        ) = _resolve_fsre_adatsk_classifier_paper_strict_config(
            paper_strict=bool(paper_strict),
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            rule_base=rule_base,
            use_en_frb=use_en_frb,
            learning_rate=learning_rate,
            batch_size=batch_size,
            fs_epochs=fs_epochs,
            re_epochs=re_epochs,
            finetune_epochs=finetune_epochs,
        )

        self.use_en_frb = bool(use_en_frb)
        self.fs_epochs = int(fs_epochs)
        self.re_epochs = int(re_epochs)
        self.finetune_epochs = int(finetune_epochs)
        self.zeta_lambda = float(zeta_lambda)
        self.zeta_theta = float(zeta_theta)
        self.structural_pruning = bool(structural_pruning)
        self.paper_strict = bool(paper_strict)

        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=fs_epochs,
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
            trainer=trainer,
        )

    def fit(
        self,
        x: object,
        y: object,
        *,
        x_val: object | None = None,
        y_val: object | None = None,
    ) -> FSREADATSKClassifier:
        """Fit the FSRE-ADATSK classifier estimator, checking input range and zetas if strict."""
        x_arr = check_array(x)
        if self.paper_strict:
            _validate_adatsk_paper_strict_input_range(x_arr, arg_name="x")
            if x_val is not None:
                _validate_adatsk_paper_strict_input_range(x_val, arg_name="x_val")

            n_features = x_arr.shape[1]
            if n_features < 1000:
                if self.zeta_lambda != 0.5:
                    raise ValueError("paper_strict requires zeta_lambda=0.5 for low-dimensional data (<1000 features)")
                if self.zeta_theta != 0.3:
                    raise ValueError("paper_strict requires zeta_theta=0.3 for low-dimensional data (<1000 features)")
            else:
                if self.zeta_lambda == 0.5 and self.zeta_theta == 0.3:
                    self.zeta_lambda = 0.4
                    self.zeta_theta = 0.5
                else:
                    if self.zeta_lambda != 0.4:
                        raise ValueError(
                            "paper_strict requires zeta_lambda=0.4 for high-dimensional data (>=1000 features)"
                        )
                    if self.zeta_theta != 0.5:
                        raise ValueError(
                            "paper_strict requires zeta_theta=0.5 for high-dimensional data (>=1000 features)"
                        )

        return cast(FSREADATSKClassifier, super().fit(x_arr, y, x_val=x_val, y_val=y_val))

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

    def predict_proba(self, x: Any) -> np.ndarray:
        """Predict class probabilities, applying structural feature selection.

        When structural pruning was applied during fit(), the original
        feature matrix is automatically sliced to the surviving features before
        being passed to the pruned model.

        Args:
            x: Input array of shape ``(N, n_features_in_)``.

        Returns:
            Class probability array of shape ``(N, n_classes)``.
        """
        check_is_fitted(self, "model_")
        x_arr = check_array(x)
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x_arr.shape[1]}")
        sf: list[int] | None = getattr(self, "history_", {}).get("surviving_feature_indices")
        x_m = x_arr[:, sf] if sf is not None and cast(Any, self.model_).n_inputs < x_arr.shape[1] else x_arr
        probs = cast(Any, self.model_).predict_proba(self._as_tensor_x(x_m, torch.device(str(self.device))))
        return probs.detach().cpu().numpy()

    def _get_trainer(self) -> BaseTrainer:
        """Return an FSRETrainer built from this estimator's parameters."""
        return FSRETrainer(
            fs_epochs=self.fs_epochs,
            fs_learning_rate=float(self.learning_rate),
            fs_batch_size=self.batch_size,
            fs_shuffle=bool(self.shuffle),
            fs_patience=self.patience,
            fs_weight_decay=float(self.weight_decay),
            fs_ur_weight=float(self.ur_weight),
            fs_ur_target=self.ur_target,
            re_epochs=self.re_epochs,
            re_learning_rate=float(self.learning_rate),
            re_batch_size=self.batch_size,
            re_shuffle=bool(self.shuffle),
            re_patience=self.patience,
            re_weight_decay=float(self.weight_decay),
            re_ur_weight=float(self.ur_weight),
            re_ur_target=self.ur_target,
            finetune_epochs=self.finetune_epochs,
            finetune_learning_rate=float(self.learning_rate),
            finetune_batch_size=self.batch_size,
            finetune_shuffle=bool(self.shuffle),
            finetune_patience=self.patience,
            finetune_restore_best=bool(self.restore_best),
            finetune_weight_decay=float(self.weight_decay),
            finetune_ur_weight=float(self.ur_weight),
            finetune_ur_target=self.ur_target,
            zeta_lambda=float(self.zeta_lambda),
            zeta_theta=float(self.zeta_theta),
            structural_pruning=bool(self.structural_pruning),
            verbose=self.verbose,
        )


class FSREADATSKRegressor(_BaseRegressorEstimator):
    """FSRE-ADATSK regressor with adaptive softmin antecedent and gated consequents.

    FSRE-ADATSK (Feature Selection and Rule Extraction) extends ADATSK.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        from highfis import FSREADATSKRegressor

        reg = FSREADATSKRegressor()
        reg.fit(X_train, y_train)
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
        fs_epochs: int = 10,
        re_epochs: int = 10,
        finetune_epochs: int = 100,
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
        zeta_lambda: float = 0.5,
        zeta_theta: float = 0.3,
        structural_pruning: bool = True,
        trainer: BaseTrainer | None = None,
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
            input_configs: Per-feature InputConfig list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default), ``"minibatch_kmeans"``, ``"fcm"``, or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended.
            random_state: Seed for k-means and weight initialisation.
            fs_epochs: Maximum epochs for phase 1 (FS training). Default ``10``.
            re_epochs: Maximum epochs for phase 2 (RE training). Default ``10``.
            finetune_epochs: Maximum epochs for phase 3 (fine-tuning). Default ``100``.
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
            zeta_lambda: Feature-selection threshold coefficient (paper eq. 28).
                Larger values retain more features.  Default ``0.5``.
            zeta_theta: Rule-extraction threshold coefficient (paper eq. 29).
                Larger values retain more rules.  Default ``0.3``.
            structural_pruning: If ``True`` (default), hard-prune the model
                architecture after each threshold step.
            trainer: Optional custom BaseTrainer.
                When ``None`` (default) an FSRETrainer is built automatically
                from this estimator's hyperparameters.

        Raises:
            ValueError: If ``lambda_init <= 0``.
        """
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")
        self.lambda_init = float(lambda_init)
        self.use_en_frb = bool(use_en_frb)
        self.fs_epochs = int(fs_epochs)
        self.re_epochs = int(re_epochs)
        self.finetune_epochs = int(finetune_epochs)
        self.zeta_lambda = float(zeta_lambda)
        self.zeta_theta = float(zeta_theta)
        self.structural_pruning = bool(structural_pruning)
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=fs_epochs,
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
            trainer=trainer,
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

    def predict(self, x: Any) -> np.ndarray:
        """Predict continuous targets, applying structural feature selection.

        When structural pruning was applied during fit(), the original
        feature matrix is automatically sliced to the surviving features before
        being passed to the pruned model.

        Args:
            x: Input array of shape ``(N, n_features_in_)``.

        Returns:
            Prediction array of shape ``(N,)``.
        """
        check_is_fitted(self, "model_")
        x_arr = check_array(x)
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x_arr.shape[1]}")
        sf: list[int] | None = getattr(self, "history_", {}).get("surviving_feature_indices")
        x_m = x_arr[:, sf] if sf is not None and cast(Any, self.model_).n_inputs < x_arr.shape[1] else x_arr
        preds = cast(Any, self.model_).predict(self._as_tensor_x(x_m, torch.device(str(self.device))))
        return preds.detach().cpu().numpy()

    def _get_trainer(self) -> BaseTrainer:
        """Return an FSRETrainer built from this estimator's parameters."""
        return FSRETrainer(
            fs_epochs=self.fs_epochs,
            fs_learning_rate=float(self.learning_rate),
            fs_batch_size=self.batch_size,
            fs_shuffle=bool(self.shuffle),
            fs_patience=self.patience,
            fs_weight_decay=float(self.weight_decay),
            fs_ur_weight=float(self.ur_weight),
            fs_ur_target=self.ur_target,
            re_epochs=self.re_epochs,
            re_learning_rate=float(self.learning_rate),
            re_batch_size=self.batch_size,
            re_shuffle=bool(self.shuffle),
            re_patience=self.patience,
            re_weight_decay=float(self.weight_decay),
            re_ur_weight=float(self.ur_weight),
            re_ur_target=self.ur_target,
            finetune_epochs=self.finetune_epochs,
            finetune_learning_rate=float(self.learning_rate),
            finetune_batch_size=self.batch_size,
            finetune_shuffle=bool(self.shuffle),
            finetune_patience=self.patience,
            finetune_restore_best=bool(self.restore_best),
            finetune_weight_decay=float(self.weight_decay),
            finetune_ur_weight=float(self.ur_weight),
            finetune_ur_target=self.ur_target,
            zeta_lambda=float(self.zeta_lambda),
            zeta_theta=float(self.zeta_theta),
            structural_pruning=bool(self.structural_pruning),
            verbose=self.verbose,
        )
