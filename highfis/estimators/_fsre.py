"""Sklearn-compatible estimators for FSRE-ADATSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from sklearn.utils.validation import check_is_fitted, validate_data

from ..memberships import MembershipFunction
from ..models import (
    BaseTSK,
    FSREADATSKClassifierModel,
    FSREADATSKRegressorModel,
)
from ..optim._base import BaseTrainer
from ..optim._fsre import FSRETrainer
from ._base import (
    BatchSizeSpec,
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
)


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
        fs_epochs: int = 100,
        re_epochs: int = 100,
        finetune_epochs: int = 100,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: BatchSizeSpec = "auto",
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = True,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        zeta_lambda: float = 0.5,
        zeta_theta: float = 0.3,
        structural_pruning: bool = True,
        trainer: BaseTrainer | None = None,
        device: str = "cpu",
        eval_metrics_every: int = 1,
        scheduler_class: type[Any] | None = None,
        scheduler_params: Mapping[str, Any] | None = None,
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
            consequent_batch_norm: Batch normalisation on consequent inputs.
                Default ``True``. Required for numerical stability: the
                first-order gated consequent spans all features and is optimized
                with plain gradient descent, which diverges (weights grow
                unbounded and become NaN) on high-dimensional data without this
                normalisation, collapsing the model to a single class. It is
                consistent with the CoCo-FRB TSK lineage the method builds on
                (Cui et al., 2020) and mirrors the ADATSK default. Set to
                ``False`` only for low-dimensional data where divergence does
                not occur.
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
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            eval_metrics_every: Evaluate training metrics every ``n`` epochs; ``0``
                skips them. Each evaluation is an extra forward pass over the training
                set and only fills ``history_["train_<metric>"]``; early stopping uses
                validation metrics regardless.
            scheduler_class: Learning-rate scheduler *class* (e.g.
                ``torch.optim.lr_scheduler.StepLR``), not an instance -- the optimiser
                it must bind to is only built inside ``fit``.
            scheduler_params: Keyword arguments for ``scheduler_class``.

        Raises:
            ValueError: If ``lambda_init <= 0``.
        """
        self.lambda_init = lambda_init
        self.use_en_frb = use_en_frb
        self.fs_epochs = fs_epochs
        self.re_epochs = re_epochs
        self.finetune_epochs = finetune_epochs
        self.zeta_lambda = zeta_lambda
        self.zeta_theta = zeta_theta
        self.structural_pruning = structural_pruning

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
            device=device,
            eval_metrics_every=eval_metrics_every,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
            trainer=trainer,
        )

    def fit(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        x_val: npt.ArrayLike | None = None,
        y_val: npt.ArrayLike | None = None,
        metrics: list[str] | None = None,
    ) -> FSREADATSKClassifier:
        """Fit the FSRE-ADATSK classifier estimator, checking lambda_init."""
        if self.lambda_init is not None and self.lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")
        return cast(FSREADATSKClassifier, super().fit(x, y, x_val=x_val, y_val=y_val, metrics=metrics))

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create FSREADATSKClassifierModel."""
        return FSREADATSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            rules=rules,
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
        x_arr = validate_data(self, x, reset=False)
        history = getattr(self, "history_", None) or {}
        sf: list[int] | None = history.get("surviving_feature_indices")
        if sf is None and "threshold" in history:
            sf = history["threshold"].get("surviving_feature_indices")
        x_m = x_arr[:, sf] if sf is not None and cast(Any, self.model_).n_inputs < x_arr.shape[1] else x_arr
        device_str = str(self.device).lower()
        x_tensor = torch.as_tensor(x_m, dtype=torch.float32, device=torch.device(device_str))
        probs = cast(Any, self.model_).predict_proba(x_tensor)
        return probs.detach().cpu().numpy()

    def __sklearn_tags__(self) -> Any:
        """Mark as poor_score: FSRE-ADATSK is designed for high-dimensional feature selection."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags

    def _get_trainer(self) -> BaseTrainer:
        """Return an FSRETrainer built from this estimator's parameters."""
        return FSRETrainer(
            fs_epochs=self.fs_epochs,
            fs_learning_rate=float(self.learning_rate),
            fs_batch_size=self._effective_batch_size,
            fs_shuffle=bool(self.shuffle),
            fs_patience=self.patience,
            fs_weight_decay=float(self.weight_decay),
            fs_ur_weight=float(self.ur_weight),
            fs_ur_target=self.ur_target,
            re_epochs=self.re_epochs,
            re_learning_rate=float(self.learning_rate),
            re_batch_size=self._effective_batch_size,
            re_shuffle=bool(self.shuffle),
            re_patience=self.patience,
            re_weight_decay=float(self.weight_decay),
            re_ur_weight=float(self.ur_weight),
            re_ur_target=self.ur_target,
            finetune_epochs=self.finetune_epochs,
            finetune_learning_rate=float(self.learning_rate),
            finetune_batch_size=self._effective_batch_size,
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
            eval_metrics_every=self.eval_metrics_every,
            scheduler_class=self.scheduler_class,
            scheduler_params=self.scheduler_params,
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
        fs_epochs: int = 100,
        re_epochs: int = 100,
        finetune_epochs: int = 100,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        rule_base: str | None = None,
        batch_size: BatchSizeSpec = "auto",
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = True,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        zeta_lambda: float = 0.5,
        zeta_theta: float = 0.3,
        structural_pruning: bool = True,
        trainer: BaseTrainer | None = None,
        device: str = "cpu",
        eval_metrics_every: int = 1,
        scheduler_class: type[Any] | None = None,
        scheduler_params: Mapping[str, Any] | None = None,
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
            consequent_batch_norm: Batch normalisation on consequent inputs.
                Default ``True``. Required for numerical stability: the
                first-order gated consequent spans all features and is optimized
                with plain gradient descent, which diverges (weights grow
                unbounded and become NaN) on high-dimensional data without this
                normalisation, collapsing the model to a single class. It is
                consistent with the CoCo-FRB TSK lineage the method builds on
                (Cui et al., 2020) and mirrors the ADATSK default. Set to
                ``False`` only for low-dimensional data where divergence does
                not occur.
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
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            eval_metrics_every: Evaluate training metrics every ``n`` epochs; ``0``
                skips them. Each evaluation is an extra forward pass over the training
                set and only fills ``history_["train_<metric>"]``; early stopping uses
                validation metrics regardless.
            scheduler_class: Learning-rate scheduler *class* (e.g.
                ``torch.optim.lr_scheduler.StepLR``), not an instance -- the optimiser
                it must bind to is only built inside ``fit``.
            scheduler_params: Keyword arguments for ``scheduler_class``.

        Raises:
            ValueError: If ``lambda_init <= 0``.
        """
        self.lambda_init = lambda_init
        self.use_en_frb = use_en_frb
        self.fs_epochs = fs_epochs
        self.re_epochs = re_epochs
        self.finetune_epochs = finetune_epochs
        self.zeta_lambda = zeta_lambda
        self.zeta_theta = zeta_theta
        self.structural_pruning = structural_pruning
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
            device=device,
            eval_metrics_every=eval_metrics_every,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
            trainer=trainer,
        )

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create FSREADATSKRegressorModel."""
        return FSREADATSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            rules=rules,
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
        x_arr = validate_data(self, x, reset=False)
        history = getattr(self, "history_", None) or {}
        sf: list[int] | None = history.get("surviving_feature_indices")
        if sf is None and "threshold" in history:
            sf = history["threshold"].get("surviving_feature_indices")
        x_m = x_arr[:, sf] if sf is not None and cast(Any, self.model_).n_inputs < x_arr.shape[1] else x_arr
        device_str = str(self.device).lower()
        x_tensor = torch.as_tensor(x_m, dtype=torch.float32, device=torch.device(device_str))
        preds = cast(Any, self.model_).predict(x_tensor)
        return preds.detach().cpu().numpy()

    def _get_trainer(self) -> BaseTrainer:
        """Return an FSRETrainer built from this estimator's parameters."""
        return FSRETrainer(
            fs_epochs=self.fs_epochs,
            fs_learning_rate=float(self.learning_rate),
            fs_batch_size=self._effective_batch_size,
            fs_shuffle=bool(self.shuffle),
            fs_patience=self.patience,
            fs_weight_decay=float(self.weight_decay),
            fs_ur_weight=float(self.ur_weight),
            fs_ur_target=self.ur_target,
            re_epochs=self.re_epochs,
            re_learning_rate=float(self.learning_rate),
            re_batch_size=self._effective_batch_size,
            re_shuffle=bool(self.shuffle),
            re_patience=self.patience,
            re_weight_decay=float(self.weight_decay),
            re_ur_weight=float(self.ur_weight),
            re_ur_target=self.ur_target,
            finetune_epochs=self.finetune_epochs,
            finetune_learning_rate=float(self.learning_rate),
            finetune_batch_size=self._effective_batch_size,
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
            eval_metrics_every=self.eval_metrics_every,
            scheduler_class=self.scheduler_class,
            scheduler_params=self.scheduler_params,
        )

    def fit(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        x_val: npt.ArrayLike | None = None,
        y_val: npt.ArrayLike | None = None,
        metrics: list[str] | None = None,
    ) -> FSREADATSKRegressor:
        """Fit the FSRE-ADATSK regressor, checking lambda_init."""
        if self.lambda_init is not None and self.lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")
        return cast(FSREADATSKRegressor, super().fit(x, y, x_val=x_val, y_val=y_val, metrics=metrics))
