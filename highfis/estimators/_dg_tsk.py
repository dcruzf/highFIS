"""Sklearn-compatible estimators for DG-TSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from ..layers import GatedClassificationZeroOrderConsequentLayer, GatedRegressionZeroOrderConsequentLayer
from ..memberships import MembershipFunction
from ..models import (
    BaseTSK,
    DGTSKClassifierModel,
    DGTSKRegressorModel,
)
from ..optim._base import BaseTrainer
from ..optim._dg import DGTrainer
from ..optim._protocols import FirstOrderModelProtocol, PFRBModelProtocol
from ..persistence import (
    deserialize_input_mfs,
    load_checkpoint,
    save_checkpoint,
    serialize_input_mfs,
    validate_checkpoint_payload,
)
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
)


class DGTSKClassifier(_BaseClassifierEstimator):
    """DG-TSK classifier with M-gate antecedent and point-based FRB (P-FRB).

    DG-TSK uses a data-guided M-gate function to automatically select
    relevant features and rules.  Training follows the DG + threshold +
    fine-tune pipeline from Xue et al. (2023):

    1. **DG phase** — Train gate parameters and zero-order consequents with
       antecedent MFs frozen (P-FRB).
    2. **Threshold search** — Grid-search for the pruning thresholds
       ``(zeta_lambda, zeta_theta)`` that maximise held-out accuracy.
    3. **Fine-tune phase** — Train first-order consequents on the pruned
       structure.

    Reference:
        Guangdong Xue, Jian Wang, Bingjie Zhang, Bin Yuan, Caili Dai,
        Double groups of gates based Takagi-Sugeno-Kang (DG-TSK)
        fuzzy system for simultaneous feature selection and rule
        extraction, Fuzzy Sets and Systems, Volume 469, 2023, 108627,
        ISSN 0165-0114, https://doi.org/10.1016/j.fss.2023.108627.

    Example:
        ```python
        from highfis import DGTSKClassifier

        clf = DGTSKClassifier(n_mfs=30, rule_base="pfrb", random_state=0)
        clf.fit(X_train, y_train, x_val=X_val, y_val=y_val)
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
        dg_epochs: int = 100,
        finetune_epochs: int = 200,
        learning_rate: float = 0.2,
        verbose: bool | int = False,
        rule_base: str | None = "pfrb",
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = 300,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        zeta_lambda: list[float] | None = None,
        zeta_theta: list[float] | None = None,
        use_lse: bool = False,
        trainer: BaseTrainer | None = None,
        optimizer_type: str = "sgd",
        structural_pruning: bool = True,
        freeze_antecedents_finetune: bool = False,
        device: str = "cpu",
        eval_metrics_every: int = 1,
        scheduler_class: type[Any] | None = None,
        scheduler_params: Mapping[str, Any] | None = None,
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
            dg_epochs: Maximum epochs for phase 1 (DG training).  Default
                ``10`` matches the paper's experimental setting.
            finetune_epochs: Maximum epochs for phase 3 (fine-tune).
                Default ``200``.
            learning_rate: Learning rate for the full-batch SGD (gradient descent)
                used in both phases. Default ``0.2`` matches the paper (Section IV).
                DG-TSK uses SGD, so it needs a larger rate than the Adam-based
                estimators to open the M-gates; too small a value under-converges
                and collapses to a single class in low dimension.
            verbose: Print per-epoch progress.
            rule_base: ``"coco"``, ``"cartesian"``, or ``"pfrb"``.  Defaults
                to ``"pfrb"`` for the paper-strict DG-TSK path, which
                initialises one rule per training sample.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum number of point-based FRB rules when
                ``rule_base='pfrb'``. Defaults to ``300`` to match the
                paper's experimental cap.
            patience: Early-stopping patience (default ``20``). Set to
                ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            zeta_lambda: Grid of λ-pruning threshold candidates.  ``None``
                uses paper default ``[0.5]``.
            zeta_theta: Grid of θ-pruning threshold candidates.  ``None``
                uses paper default ``[0.01]``.
            use_lse: Refit first-order consequents via LSE during threshold
                search.  Defaults to ``False`` for the paper-faithful
                classification path.
            trainer: Optional custom :class:`~highfis.optim.BaseTrainer`.
                When ``None`` (default) a :class:`~highfis.optim.DGTrainer`
                is built from this estimator's hyperparameters.  Pass a
                :class:`~highfis.optim.GradientTrainer` to use single-phase
                training instead.
            optimizer_type: Optimizer type.  ``"sgd"`` (default, paper) or
                ``"adamw"``.
            structural_pruning: If ``True`` (default), apply hard structural
                pruning after threshold search.
            freeze_antecedents_finetune: If ``True``, freeze MF parameters
                and feature gates during fine-tuning. Defaults to ``False``
                to optimize antecedents and consequents in fine-tuning.
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
        """
        self.use_en_frb = use_en_frb
        self.dg_epochs = dg_epochs
        self.finetune_epochs = finetune_epochs
        self.zeta_lambda = zeta_lambda
        self.zeta_theta = zeta_theta
        self.use_lse = use_lse
        self.optimizer_type = optimizer_type
        self.structural_pruning = structural_pruning
        self.freeze_antecedents_finetune = freeze_antecedents_finetune
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=dg_epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base=rule_base,
            batch_size=batch_size,
            shuffle=shuffle,
            weight_decay=weight_decay,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=pfrb_max_rules,
            patience=patience,
            restore_best=restore_best,
            device=device,
            eval_metrics_every=eval_metrics_every,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
            trainer=trainer,
        )

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        x_val: Any | None = None,
        y_val: Any | None = None,
        metrics: list[str] | None = None,
    ) -> DGTSKClassifier:
        """Train the DG-TSK classifier.

        Validation data should be supplied using ``x_val`` and ``y_val``
        when available.
        """
        return cast(DGTSKClassifier, super().fit(x, y, x_val=x_val, y_val=y_val, metrics=metrics))

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> BaseTSK:
        """Create DGTSKClassifierModel."""
        return DGTSKClassifierModel(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            rules=rules,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=self.use_en_frb,
            optimizer_type=self.optimizer_type,
        )

    def _get_trainer(self) -> DGTrainer:
        """Return a :class:`~highfis.optim.DGTrainer` built from this estimator's params."""
        resolved_zeta_lambda = [0.5] if self.zeta_lambda is None else list(self.zeta_lambda)
        resolved_zeta_theta = [0.01] if self.zeta_theta is None else list(self.zeta_theta)
        return DGTrainer(
            dg_epochs=int(self.dg_epochs),
            dg_learning_rate=float(self.learning_rate),
            dg_batch_size=self.batch_size,
            dg_shuffle=bool(self.shuffle),
            dg_patience=self.patience,
            dg_weight_decay=float(self.weight_decay),
            dg_ur_weight=float(self.ur_weight),
            dg_ur_target=self.ur_target,
            zeta_lambda=resolved_zeta_lambda,
            zeta_theta=resolved_zeta_theta,
            use_lse=bool(self.use_lse),
            finetune_epochs=int(self.finetune_epochs),
            finetune_learning_rate=float(self.learning_rate),
            finetune_batch_size=self.batch_size,
            finetune_shuffle=bool(self.shuffle),
            finetune_patience=self.patience,
            finetune_restore_best=bool(self.restore_best),
            finetune_weight_decay=float(self.weight_decay),
            finetune_ur_weight=float(self.ur_weight),
            finetune_ur_target=self.ur_target,
            verbose=self.verbose,
            eval_metrics_every=self.eval_metrics_every,
            scheduler_class=self.scheduler_class,
            scheduler_params=self.scheduler_params,
            optimizer_type=self.optimizer_type,
            structural_pruning=bool(self.structural_pruning),
            finetune_freeze_antecedents=bool(self.freeze_antecedents_finetune),
        )

    def _pre_train_hook(self, model: BaseTSK, x_t: Tensor, y_t: Tensor) -> None:
        """Initialise P-FRB consequents from class labels before training."""
        if self.rule_base == "pfrb" and hasattr(model, "init_consequents_from_labels"):
            cast(PFRBModelProtocol, model).init_consequents_from_labels(self._pfrb_aligned_labels(x_t, y_t))

    def _select_model_features(self, x_arr: np.ndarray) -> np.ndarray:
        """Slice inputs to the surviving features when structural pruning shrank the model."""
        return _select_dgtsking_surviving_features(self, x_arr)

    def save(self, path: str) -> None:
        """Persist estimator including first-order architecture flag."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "model_")
        is_first_order = not isinstance(
            self.model_.consequent_layer,  # type: ignore[attr-defined]
            GatedClassificationZeroOrderConsequentLayer,
        )
        consequent_mode = (
            str(self.model_.consequent_layer.mode)  # type: ignore[attr-defined]
            if is_first_order and hasattr(self.model_.consequent_layer, "mode")  # type: ignore[attr-defined]
            else None
        )
        _fnames: np.ndarray | None = getattr(self, "feature_names_in_", None)
        rules = None
        if hasattr(self.model_, "rule_layer") and hasattr(self.model_.rule_layer, "rules"):
            rules = self.model_.rule_layer.rules
        checkpoint = self._build_checkpoint_base(
            model_init={
                "input_mfs_config": serialize_input_mfs(self.model_.input_mfs),  # type: ignore[attr-defined]
                "n_classes": len(self.classes_),
                "rule_base": self.rule_base_,
                "is_first_order": is_first_order,
                "consequent_mode": consequent_mode,
                "rules": rules,
            },
            fitted_attrs={
                "n_features_in": int(self.n_features_in_),
                "feature_names_in": _fnames.tolist() if _fnames is not None else None,
                "classes": self.classes_.tolist(),
            },
        )
        save_checkpoint(path, checkpoint)

    @classmethod
    def load(cls, path: str) -> DGTSKClassifier:
        """Load a persisted DGTSKClassifier, restoring first-order architecture."""
        checkpoint = load_checkpoint(path)
        validate_checkpoint_payload(checkpoint, expected_estimator_class=cls.__name__)

        params: dict[str, Any] = dict(checkpoint["estimator_params"])
        if params.get("input_configs") is not None:
            params["input_configs"] = [InputConfig(**c) for c in params["input_configs"]]
        estimator = cls(**params)
        model_init = checkpoint["model_init"]
        estimator.rule_base_ = model_init["rule_base"]
        estimator.model_ = estimator._build_model(
            deserialize_input_mfs(model_init["input_mfs_config"]),
            int(model_init["n_classes"]),
            str(model_init["rule_base"]),
            rules=model_init.get("rules"),
        )
        if model_init.get("is_first_order", False):  # pragma: no branch
            cast(FirstOrderModelProtocol, estimator.model_).convert_to_first_order()
            mode = model_init.get("consequent_mode")
            if isinstance(mode, str) and hasattr(estimator.model_.consequent_layer, "mode"):
                estimator.model_.consequent_layer.mode = mode  # type: ignore[attr-defined]
        estimator.model_.load_state_dict(checkpoint["model_state_dict"])
        estimator.model_.to(torch.device(str(estimator.device)))

        fitted = checkpoint["fitted_attrs"]
        estimator.n_features_in_ = int(fitted["n_features_in"])
        if fitted.get("feature_names_in") is not None:
            estimator.feature_names_in_ = np.asarray(fitted["feature_names_in"], dtype=object)
        elif hasattr(estimator, "feature_names_in_"):
            delattr(estimator, "feature_names_in_")
        from sklearn.preprocessing import LabelEncoder

        estimator.classes_ = np.asarray(fitted["classes"], dtype=object)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = estimator.classes_
        estimator._label_encoder_ = label_encoder
        history = checkpoint.get("history")
        estimator.history_ = history if history is not None else {}
        return estimator

    def predict_proba(self, x: Any) -> np.ndarray:
        """Predict class probabilities, applying any surviving-feature pruning from fit()."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "model_")
        from sklearn.utils.validation import validate_data

        x_arr = validate_data(self, x, reset=False)
        x_model = _select_dgtsking_surviving_features(self, x_arr)
        device_str = str(self.device).lower()
        x_tensor = torch.as_tensor(x_model, dtype=torch.float32, device=torch.device(device_str))
        probs = cast(Any, self.model_).predict_proba(x_tensor)
        return probs.detach().cpu().numpy()

    def __sklearn_tags__(self) -> Any:
        """Mark as poor_score: DG-TSK is designed for high-dimensional data."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags


class DGTSKRegressor(_BaseRegressorEstimator):
    """DG-TSK regressor with M-gate antecedent and point-based FRB (P-FRB).

    DG-TSK uses a data-guided M-gate function to automatically select
    relevant features and rules.  Training follows the three-phase protocol
    from Xue et al. (2023).

    Reference:
        Guangdong Xue, Jian Wang, Bingjie Zhang, Bin Yuan, Caili Dai,
        Double groups of gates based Takagi-Sugeno-Kang (DG-TSK)
        fuzzy system for simultaneous feature selection and rule
        extraction, Fuzzy Sets and Systems, Volume 469, 2023, 108627,
        ISSN 0165-0114, https://doi.org/10.1016/j.fss.2023.108627.

    Example:
        ```python
        from highfis import DGTSKRegressor

        reg = DGTSKRegressor(n_mfs=30, rule_base="pfrb", random_state=0)
        reg.fit(X_train, y_train, x_val=X_val, y_val=y_val)
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
        dg_epochs: int = 100,
        finetune_epochs: int = 200,
        learning_rate: float = 0.2,
        verbose: bool | int = False,
        rule_base: str | None = "pfrb",
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        pfrb_max_rules: int | None = None,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        zeta_lambda: list[float] | None = None,
        zeta_theta: list[float] | None = None,
        use_lse: bool = True,
        trainer: BaseTrainer | None = None,
        optimizer_type: str = "sgd",
        structural_pruning: bool = True,
        freeze_antecedents_finetune: bool = True,
        device: str = "cpu",
        eval_metrics_every: int = 1,
        scheduler_class: type[Any] | None = None,
        scheduler_params: Mapping[str, Any] | None = None,
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
            dg_epochs: Maximum epochs for phase 1 (DG training).  Default
                ``10`` matches the paper's experimental setting.
            finetune_epochs: Maximum epochs for phase 3 (fine-tune).
                Default ``200``.
            learning_rate: Learning rate for the full-batch SGD (gradient descent)
                used in both phases. Default ``0.2`` matches the paper (Section IV).
                DG-TSK uses SGD, so it needs a larger rate than the Adam-based
                estimators to open the M-gates; too small a value under-converges
                and collapses to a single class in low dimension.
            verbose: Print per-epoch progress.
            rule_base: ``"coco"``, ``"cartesian"``, or ``"pfrb"``.  Defaults
                to ``"pfrb"`` for the paper-strict DG-TSK regressor path.
            batch_size: Mini-batch size (default ``512``).
            shuffle: Reshuffle each epoch.
            ur_weight: Uncertainty regularisation weight.
            ur_target: Uncertainty regularisation target.
            consequent_batch_norm: Batch normalisation on consequent layers.
            pfrb_max_rules: Maximum number of point-based FRB rules when
                ``rule_base='pfrb'``. ``None`` uses all training samples.
            patience: Early-stopping patience (default ``20``). Set to
                ``None`` to disable early stopping.
            restore_best: If ``True`` (default), restore the best validation
                model weights after training.
            weight_decay: L2 weight decay for consequent parameters.
            zeta_lambda: Grid of λ-pruning threshold candidates.
            zeta_theta: Grid of θ-pruning threshold candidates.
            use_lse: Refit first-order consequents via LSE during threshold
                search.  Recommended (default ``True``).
            trainer: Optional custom :class:`~highfis.optim.BaseTrainer`.
                When ``None`` (default) a :class:`~highfis.optim.DGTrainer`
                is built from this estimator's hyperparameters.
            optimizer_type: Optimizer type.  ``"sgd"`` (default, paper) or
                ``"adamw"``.
            structural_pruning: If ``True`` (default), apply hard structural
                pruning after threshold search.
            freeze_antecedents_finetune: If ``True`` (default), freeze MF
                parameters and feature gates during fine-tuning.
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
        """
        self.use_en_frb = use_en_frb
        self.dg_epochs = dg_epochs
        self.finetune_epochs = finetune_epochs
        self.zeta_lambda = zeta_lambda
        self.zeta_theta = zeta_theta
        self.use_lse = use_lse
        self.optimizer_type = optimizer_type
        self.structural_pruning = structural_pruning
        self.freeze_antecedents_finetune = freeze_antecedents_finetune
        super().__init__(
            input_configs=input_configs,
            n_mfs=n_mfs,
            mf_init=mf_init,
            sigma_scale=sigma_scale,
            random_state=random_state,
            epochs=dg_epochs,
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
        """Create DGTSKRegressorModel."""
        return DGTSKRegressorModel(
            input_mfs,
            rule_base=rule_base,
            rules=rules,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            use_en_frb=bool(self.use_en_frb),
            optimizer_type=self.optimizer_type,
        )

    def _get_trainer(self) -> DGTrainer:
        """Return a :class:`~highfis.optim.DGTrainer` built from this estimator's params."""
        resolved_zeta_lambda = [0.5] if self.zeta_lambda is None else list(self.zeta_lambda)
        resolved_zeta_theta = [0.01] if self.zeta_theta is None else list(self.zeta_theta)
        return DGTrainer(
            dg_epochs=int(self.dg_epochs),
            dg_learning_rate=float(self.learning_rate),
            dg_batch_size=self.batch_size,
            dg_shuffle=bool(self.shuffle),
            dg_patience=self.patience,
            dg_weight_decay=float(self.weight_decay),
            dg_ur_weight=float(self.ur_weight),
            dg_ur_target=self.ur_target,
            zeta_lambda=resolved_zeta_lambda,
            zeta_theta=resolved_zeta_theta,
            use_lse=bool(self.use_lse),
            finetune_epochs=int(self.finetune_epochs),
            finetune_learning_rate=float(self.learning_rate),
            finetune_batch_size=self.batch_size,
            finetune_shuffle=bool(self.shuffle),
            finetune_patience=self.patience,
            finetune_restore_best=bool(self.restore_best),
            finetune_weight_decay=float(self.weight_decay),
            finetune_ur_weight=float(self.ur_weight),
            finetune_ur_target=self.ur_target,
            verbose=self.verbose,
            eval_metrics_every=self.eval_metrics_every,
            scheduler_class=self.scheduler_class,
            scheduler_params=self.scheduler_params,
            optimizer_type=self.optimizer_type,
            structural_pruning=bool(self.structural_pruning),
            finetune_freeze_antecedents=bool(self.freeze_antecedents_finetune),
        )

    def _pre_train_hook(self, model: BaseTSK, x_t: Tensor, y_t: Tensor) -> None:
        """Initialise P-FRB consequents from targets before training."""
        if self.rule_base == "pfrb" and hasattr(model, "init_consequents_from_labels"):
            cast(PFRBModelProtocol, model).init_consequents_from_labels(
                self._pfrb_aligned_labels(x_t, y_t)
            )  # pragma: no cover

    def save(self, path: str) -> None:
        """Persist estimator including first-order architecture flag."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "model_")
        is_first_order = not isinstance(
            self.model_.consequent_layer,  # type: ignore[attr-defined]
            GatedRegressionZeroOrderConsequentLayer,
        )
        consequent_mode = (
            str(self.model_.consequent_layer.mode)  # type: ignore[attr-defined]
            if is_first_order and hasattr(self.model_.consequent_layer, "mode")  # type: ignore[attr-defined]
            else None
        )
        _fnames: np.ndarray | None = getattr(self, "feature_names_in_", None)
        rules = None
        if hasattr(self.model_, "rule_layer") and hasattr(self.model_.rule_layer, "rules"):
            rules = self.model_.rule_layer.rules
        checkpoint = self._build_checkpoint_base(
            model_init={
                "input_mfs_config": serialize_input_mfs(self.model_.input_mfs),  # type: ignore[attr-defined]
                "rule_base": self.rule_base_,
                "is_first_order": is_first_order,
                "consequent_mode": consequent_mode,
                "rules": rules,
            },
            fitted_attrs={
                "n_features_in": int(self.n_features_in_),
                "feature_names_in": _fnames.tolist() if _fnames is not None else None,
            },
        )
        save_checkpoint(path, checkpoint)

    @classmethod
    def load(cls, path: str) -> DGTSKRegressor:
        """Load a persisted DGTSKRegressor, restoring first-order architecture."""
        checkpoint = load_checkpoint(path)
        validate_checkpoint_payload(checkpoint, expected_estimator_class=cls.__name__)

        params: dict[str, Any] = dict(checkpoint["estimator_params"])
        if params.get("input_configs") is not None:
            params["input_configs"] = [InputConfig(**c) for c in params["input_configs"]]
        estimator = cls(**params)
        model_init = checkpoint["model_init"]
        estimator.rule_base_ = model_init["rule_base"]
        estimator.model_ = estimator._build_regressor_model(
            deserialize_input_mfs(model_init["input_mfs_config"]),
            str(model_init["rule_base"]),
            rules=model_init.get("rules"),
        )
        if model_init.get("is_first_order", False):  # pragma: no branch
            cast(FirstOrderModelProtocol, estimator.model_).convert_to_first_order()
            mode = model_init.get("consequent_mode")
            if isinstance(mode, str) and hasattr(estimator.model_.consequent_layer, "mode"):
                estimator.model_.consequent_layer.mode = mode  # type: ignore[attr-defined]
        estimator.model_.load_state_dict(checkpoint["model_state_dict"])
        estimator.model_.to(torch.device(str(estimator.device)))

        fitted = checkpoint["fitted_attrs"]
        estimator.n_features_in_ = int(fitted["n_features_in"])
        if fitted.get("feature_names_in") is not None:
            estimator.feature_names_in_ = np.asarray(fitted["feature_names_in"], dtype=object)
        elif hasattr(estimator, "feature_names_in_"):
            delattr(estimator, "feature_names_in_")
        history = checkpoint.get("history")
        estimator.history_ = history if history is not None else {}
        return estimator

    def predict(self, x: Any) -> np.ndarray:
        """Predict regression values, applying any surviving-feature pruning from fit()."""
        from sklearn.utils.validation import check_is_fitted, validate_data

        check_is_fitted(self, "model_")
        x_arr = validate_data(self, x, reset=False)
        x_model = _select_dgtsking_surviving_features(self, x_arr)
        device_str = str(self.device).lower()
        x_tensor = torch.as_tensor(x_model, dtype=torch.float32, device=torch.device(device_str))
        preds = cast(Any, self.model_).predict(x_tensor)
        return preds.detach().cpu().numpy()

    def __sklearn_tags__(self) -> Any:
        """Mark as poor_score: DG-TSK is designed for high-dimensional data."""
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags


def _select_dgtsking_surviving_features(estimator: Any, x_arr: np.ndarray) -> np.ndarray:
    """Slice to surviving features when DG structural pruning reduced the model input width."""
    history = getattr(estimator, "history_", None) or {}
    surviving_features: list[int] | None = history.get("surviving_feature_indices")
    if surviving_features is None and isinstance(history.get("threshold"), dict):
        surviving_features = cast(list[int] | None, history["threshold"].get("surviving_feature_indices"))
    if surviving_features is not None and getattr(estimator.model_, "n_inputs", x_arr.shape[1]) < x_arr.shape[1]:
        return x_arr[:, surviving_features]
    return x_arr
