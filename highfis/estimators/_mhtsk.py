"""Sklearn-compatible estimators for MHTSK models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np
import torch
from sklearn.utils.validation import check_X_y

from ..base import BaseTSK
from ..memberships import (
    MembershipFunction,
)
from ..models import (
    MHTSKClassifierModel,
    MHTSKRegressorModel,
)
from ..optim._base import BaseTrainer
from ..optim._gradient import GradientTrainer
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
    _build_mhtsk_input_mfs,
    _extract_mhtsk_rule_indices,
    _extract_mhtsk_rule_indices_unsupervised,
    _resolve_mhtsk_scale_parameters,
)


def _resolve_mhtsk_paper_strict_classifier_config(
    *,
    paper_strict: bool,
    n_mfs: int | None,
    n_heads: int | None,
    head_size: int | None,
    head_size_ratio: float | None,
    fcm_m: float | None,
    rule_sigma: float | None,
    fcr_target: float | None,
    h_value: float | None,
    xi: float | None,
    instance_sample_fraction: float | None,
    rule_extraction: bool | None,
    crcr_us: float | None,
    crcr_s: float | None,
    retrain_after_extraction: bool | None,
) -> tuple[int, float, float, float, bool, float, float, bool]:
    if not paper_strict:
        return (
            3 if n_mfs is None else int(n_mfs),
            2.0 if fcm_m is None else float(fcm_m),
            1.0 if rule_sigma is None else float(rule_sigma),
            743.0 if xi is None else float(xi),
            False if rule_extraction is None else bool(rule_extraction),
            0.5 if crcr_us is None else float(crcr_us),
            0.5 if crcr_s is None else float(crcr_s),
            True if retrain_after_extraction is None else bool(retrain_after_extraction),
        )

    if n_mfs is not None and int(n_mfs) != 3:
        raise ValueError("paper_strict requires n_mfs=3")
    if n_heads is not None:
        raise ValueError("paper_strict computes n_heads from input dimension")
    if head_size is not None:
        raise ValueError("paper_strict computes head_size from input dimension")
    if head_size_ratio is not None:
        raise ValueError("paper_strict requires head_size_ratio=None")
    if fcm_m is not None and float(fcm_m) != 2.0:
        raise ValueError("paper_strict requires fcm_m=2.0")
    if rule_sigma is not None and float(rule_sigma) != 1.0:
        raise ValueError("paper_strict requires rule_sigma=1.0")
    if fcr_target is not None:
        raise ValueError("paper_strict requires fcr_target=None")
    if h_value is not None:
        raise ValueError("paper_strict requires h_value=None")
    if xi is not None and float(xi) != 743.0:
        raise ValueError("paper_strict requires xi=743.0")
    if instance_sample_fraction is not None and float(instance_sample_fraction) != 0.8:
        raise ValueError("paper_strict requires instance_sample_fraction=0.8")
    if rule_extraction is not None and not bool(rule_extraction):
        raise ValueError("paper_strict requires rule_extraction=True")
    if crcr_us is not None and float(crcr_us) != 0.5:
        raise ValueError("paper_strict requires crcr_us=0.5")
    if crcr_s is not None and float(crcr_s) != 0.5:
        raise ValueError("paper_strict requires crcr_s=0.5")
    if retrain_after_extraction is not None and not bool(retrain_after_extraction):
        raise ValueError("paper_strict requires retrain_after_extraction=True")

    return 3, 2.0, 1.0, 743.0, True, 0.5, 0.5, True


def _resolve_mhtsk_paper_strict_regressor_config(
    *,
    paper_strict: bool,
    n_mfs: int | None,
    n_heads: int | None,
    head_size: int | None,
    head_size_ratio: float | None,
    fcm_m: float | None,
    rule_sigma: float | None,
    fcr_target: float | None,
    h_value: float | None,
    xi: float | None,
    instance_sample_fraction: float | None,
    rule_extraction: bool | None,
    crcr_us: float | None,
    retrain_after_extraction: bool | None,
) -> tuple[int, float, float, float, bool, float, bool]:
    if not paper_strict:
        return (
            3 if n_mfs is None else int(n_mfs),
            2.0 if fcm_m is None else float(fcm_m),
            1.0 if rule_sigma is None else float(rule_sigma),
            743.0 if xi is None else float(xi),
            False if rule_extraction is None else bool(rule_extraction),
            0.5 if crcr_us is None else float(crcr_us),
            True if retrain_after_extraction is None else bool(retrain_after_extraction),
        )

    if n_mfs is not None and int(n_mfs) != 3:
        raise ValueError("paper_strict requires n_mfs=3")
    if n_heads is not None:
        raise ValueError("paper_strict computes n_heads from input dimension")
    if head_size is not None:
        raise ValueError("paper_strict computes head_size from input dimension")
    if head_size_ratio is not None:
        raise ValueError("paper_strict requires head_size_ratio=None")
    if fcm_m is not None and float(fcm_m) != 2.0:
        raise ValueError("paper_strict requires fcm_m=2.0")
    if rule_sigma is not None and float(rule_sigma) != 1.0:
        raise ValueError("paper_strict requires rule_sigma=1.0")
    if fcr_target is not None:
        raise ValueError("paper_strict requires fcr_target=None")
    if h_value is not None:
        raise ValueError("paper_strict requires h_value=None")
    if xi is not None and float(xi) != 743.0:
        raise ValueError("paper_strict requires xi=743.0")
    if instance_sample_fraction is not None and float(instance_sample_fraction) != 0.8:
        raise ValueError("paper_strict requires instance_sample_fraction=0.8")
    if rule_extraction is not None and not bool(rule_extraction):
        raise ValueError("paper_strict requires rule_extraction=True")
    if crcr_us is not None and float(crcr_us) != 0.5:
        raise ValueError("paper_strict requires crcr_us=0.5")
    if retrain_after_extraction is not None and not bool(retrain_after_extraction):
        raise ValueError("paper_strict requires retrain_after_extraction=True")

    return 3, 2.0, 1.0, 743.0, True, 0.5, True


def _strict_mhtsk_scale_from_dimension(n_features: int, *, sigma: float, xi: float) -> tuple[int, int]:
    base_head_size = max(1, round(n_features * 0.02)) if n_features <= 5000 else max(1, round(n_features * 0.01))
    max_head_size = min(n_features, max(1, math.floor(2.0 * xi * sigma * sigma)))
    head_size = min(base_head_size, max_head_size)
    n_heads = 200 if n_features <= 5000 else 300
    return head_size, n_heads


class _MHTSKPaperStrictTrainer(GradientTrainer):
    """Strict trainer that updates only consequent parameters using Adam."""

    def fit(
        self,
        model: BaseTSK,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        consequent_params = list(model.consequent_layer.parameters())
        if model.consequent_bn is not None:
            consequent_params.extend(model.consequent_bn.parameters())
        optimizer = torch.optim.Adam(consequent_params, lr=float(self.learning_rate))
        return model.fit(
            x,
            y,
            epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            criterion=self.loss,
            optimizer=optimizer,
            batch_size=self.batch_size,
            shuffle=bool(self.shuffle),
            ur_weight=float(self.ur_weight),
            ur_target=self.ur_target,
            verbose=self.verbose,
            x_val=x_val,
            y_val=y_val,
            patience=self.patience,
            restore_best=bool(self.restore_best),
            weight_decay=float(self.weight_decay),
        )


class MHTSKClassifier(_BaseClassifierEstimator):
    """Estimator for the multihead Takagi-Sugeno-Kang fuzzy system.

    This estimator supports paper-derived automatic scale parameter resolution
    for head size and number of heads, and it optionally subsamples instances
    when building each head as described in the MHTSK paper.

    MHTSK builds multiple sparse subantecedents from random feature
    subsets and jointly optimizes their rule consequents.

    Reference:
        Z. Bian, Q. Chang, J. Wang and N. R. Pal, "Multihead
        Takagi-Sugeno-Kang Fuzzy System," in IEEE Transactions
        on Fuzzy Systems, vol. 33, no. 8, pp. 2561-2573, Aug. 2025,
        doi: 10.1109/TFUZZ.2025.3569227.

    Example:
        ```python
        from highfis import MHTSKClassifier

        clf = MHTSKClassifier()
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int | None = None,
        n_heads: int | None = None,
        head_size: int | None = None,
        head_size_ratio: float | None = None,
        fcm_m: float | None = None,
        rule_sigma: float | None = None,
        fcr_target: float | None = None,
        h_value: float | None = None,
        xi: float | None = None,
        instance_sample_fraction: float | None = None,
        rule_extraction: bool | None = None,
        crcr_us: float | None = None,
        crcr_s: float | None = None,
        retrain_after_extraction: bool | None = None,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        device: str = "cpu",
        paper_strict: bool = False,
    ) -> None:
        """Initialize a MHTSK classifier estimator.

        Args:
            input_configs: Optional list of per-feature input configurations.
                Only the feature names are used for FCM-based MHTSK head construction.
            n_mfs: Number of FCM clusters per head (number of rules generated by each head).
            n_heads: Number of random heads. If ``None``, this is resolved from
                ``head_size`` together with ``fcr_target`` or ``h_value``.
            head_size: Number of features sampled per head. If ``None``, defaults to
                ``round(D * 0.02)`` for ``D <= 5000`` and ``round(D * 0.01)`` for larger inputs.
            head_size_ratio: Alternative relative head size, as a fraction of the input dimension.
            fcm_m: Fuzzification exponent for FCM cluster fitting.
            rule_sigma: Gaussian sigma applied to rule antecedent membership functions.
            fcr_target: Target feature coverage rate for randomly sampled heads.
            h_value: Paper-derived scale constant ``H`` used to compute the number of heads.
                When set, this overrides ``fcr_target``.
            xi: Numeric underflow threshold constant used to bound ``head_size``.
            instance_sample_fraction: Fraction of training instances sampled per head for FCM.
            rule_extraction: If ``True``, perform post-fit rule extraction (MHTSK_RE).
            crcr_us: Unsupervised cumulative rule contribution rate target used in extraction.
            crcr_s: Supervised cumulative rule contribution rate target used in extraction.
            retrain_after_extraction: If ``True``, retrain the extracted rule base after extraction.
            random_state: Random seed for reproducible head construction and FCM initialization.
            epochs: Maximum number of training epochs.
            learning_rate: Adam optimizer learning rate.
            verbose: Verbosity level during training.
            batch_size: Mini-batch size for gradient descent.
            shuffle: Whether to shuffle training samples each epoch.
            ur_weight: Weight of the uncertainty regularization term.
            ur_target: Target firing-level for uncertainty regularization.
            consequent_batch_norm: Apply batch normalization to the consequent layer inputs.
            patience: Early-stopping patience for validation.
            restore_best: Whether to restore the best validation model weights after training.
            weight_decay: Weight decay coefficient for the optimizer.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            paper_strict: If ``True``, apply paper-derived defaults for MHTSK
                scale and extraction parameters when omitted.
        """
        (
            resolved_n_mfs,
            resolved_fcm_m,
            resolved_rule_sigma,
            resolved_xi,
            resolved_rule_extraction,
            resolved_crcr_us,
            resolved_crcr_s,
            resolved_retrain_after_extraction,
        ) = _resolve_mhtsk_paper_strict_classifier_config(
            paper_strict=bool(paper_strict),
            n_mfs=n_mfs,
            n_heads=n_heads,
            head_size=head_size,
            head_size_ratio=head_size_ratio,
            fcm_m=fcm_m,
            rule_sigma=rule_sigma,
            fcr_target=fcr_target,
            h_value=h_value,
            xi=xi,
            instance_sample_fraction=instance_sample_fraction,
            rule_extraction=rule_extraction,
            crcr_us=crcr_us,
            crcr_s=crcr_s,
            retrain_after_extraction=retrain_after_extraction,
        )

        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init="fcm",
            sigma_scale=1.0,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base="custom",
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=None,
            patience=patience,
            restore_best=restore_best,
            weight_decay=weight_decay,
            device=device,
        )
        self.n_heads = int(n_heads) if n_heads is not None else None
        self.head_size = int(head_size) if head_size is not None else None
        self.head_size_ratio = float(head_size_ratio) if head_size_ratio is not None else None
        self.fcm_m = float(resolved_fcm_m)
        self.rule_sigma = float(resolved_rule_sigma)
        self.fcr_target = float(fcr_target) if fcr_target is not None else None
        self.h_value = float(h_value) if h_value is not None else None
        self.xi = float(resolved_xi)
        self.instance_sample_fraction = 0.8 if instance_sample_fraction is None else float(instance_sample_fraction)
        self.rule_extraction = bool(resolved_rule_extraction)
        self.crcr_us = float(resolved_crcr_us)
        self.crcr_s = float(resolved_crcr_s)
        self.retrain_after_extraction = bool(resolved_retrain_after_extraction)
        self.paper_strict = bool(paper_strict)
        self._extracted_rule_indices_: list[int] | None = None

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:  # type: ignore[override]
        feature_names = self._resolve_feature_names(x_arr)
        if self.paper_strict:
            head_size, n_heads = _strict_mhtsk_scale_from_dimension(x_arr.shape[1], sigma=self.rule_sigma, xi=self.xi)
        else:
            head_size, n_heads = _resolve_mhtsk_scale_parameters(
                n_features=x_arr.shape[1],
                head_size=self.head_size,
                head_size_ratio=self.head_size_ratio,
                n_heads=self.n_heads,
                fcr_target=self.fcr_target,
                h_value=self.h_value,
                sigma=self.rule_sigma,
                xi=self.xi,
            )
        input_mfs, rules, rule_feature_mask = _build_mhtsk_input_mfs(
            x_arr,
            feature_names=feature_names,
            n_heads=n_heads,
            head_size=head_size,
            n_clusters=int(self.n_mfs),
            fcm_m=self.fcm_m,
            rule_sigma=self.rule_sigma,
            instance_sample_fraction=self.instance_sample_fraction,
            random_state=self.random_state,
        )
        self._mhtsk_rules = rules
        self._mhtsk_rule_feature_mask = torch.as_tensor(rule_feature_mask, dtype=torch.bool)
        return input_mfs, feature_names, "custom"

    def _get_trainer(self) -> BaseTrainer:
        if not self.paper_strict:
            return super()._get_trainer()
        return _MHTSKPaperStrictTrainer(
            epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            batch_size=self.batch_size,
            shuffle=bool(self.shuffle),
            patience=self.patience,
            restore_best=bool(self.restore_best),
            weight_decay=float(self.weight_decay),
            ur_weight=float(self.ur_weight),
            ur_target=self.ur_target,
            verbose=self.verbose,
        )

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        return MHTSKClassifierModel(
            input_mfs,
            self._mhtsk_rule_feature_mask,
            self._mhtsk_rules,
            n_classes=n_classes,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )

    def _build_extracted_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_indices: list[int],
    ) -> None:
        if not rule_indices:
            raise ValueError("At least one rule must be selected for extraction")
        self._mhtsk_rules = [self._mhtsk_rules[i] for i in rule_indices]
        self._mhtsk_rule_feature_mask = self._mhtsk_rule_feature_mask[rule_indices]
        self.model_ = self._build_model(input_mfs, len(self.classes_), self.rule_base_)

    def fit(self, x: Any, y: Any, *, x_val: Any | None = None, y_val: Any | None = None) -> Self:
        """Train the MHTSK classifier and optionally extract rules.

        After the base training step, if ``rule_extraction`` is enabled, the
        firing-strength matrix is used to select a compact rule subset via the
        CRCR criterion.  When ``retrain_after_extraction`` is also set, a
        second training pass is performed on the reduced model.
        """
        x_arr, y_arr = check_X_y(x, y)
        super().fit(x, y, x_val=x_val, y_val=y_val)

        if not bool(self.rule_extraction):
            return self

        x_t = self._as_tensor_x(x_arr)
        self.model_.eval()
        with torch.no_grad():
            norm_w = self.model_.forward_antecedents(x_t)

        y_t = torch.as_tensor(self._label_encoder_.transform(np.asarray(y_arr)), dtype=torch.long)
        selected = _extract_mhtsk_rule_indices(norm_w, y_t, self.crcr_us, self.crcr_s)
        self._extracted_rule_indices_ = selected

        input_mfs = self.model_.input_mfs
        self._build_extracted_model(input_mfs, selected)

        if self.retrain_after_extraction:
            x_val_t: torch.Tensor | None = None
            y_val_t: torch.Tensor | None = None
            if x_val is not None and y_val is not None:
                x_v_arr, y_v_arr = check_X_y(x_val, y_val)
                x_val_t = self._as_tensor_x(x_v_arr)
                y_val_t = torch.as_tensor(
                    self._label_encoder_.transform(np.asarray(y_v_arr)),
                    dtype=torch.long,
                )
            if self.paper_strict:
                trainer = self._get_trainer()
                self.history_ = trainer.fit(self.model_, x_t, y_t, x_val=x_val_t, y_val=y_val_t)
            else:
                self.history_ = self.model_.fit(
                    x_t,
                    y_t,
                    epochs=int(self.epochs),
                    learning_rate=float(self.learning_rate),
                    batch_size=self.batch_size,
                    shuffle=bool(self.shuffle),
                    ur_weight=float(self.ur_weight),
                    ur_target=self.ur_target,
                    verbose=self.verbose,
                    x_val=x_val_t,
                    y_val=y_val_t,
                    patience=self.patience,
                    restore_best=self.restore_best,
                    weight_decay=float(self.weight_decay),
                )
        return self


class MHTSKRegressor(_BaseRegressorEstimator):
    """Estimator for the multihead Takagi-Sugeno-Kang fuzzy system.

    The regressor uses the same MHTSK head construction and scale parameter
    strategy as the classifier. Rule extraction is currently implemented with
    an unsupervised scheme only, since regression does not provide class labels
    for the Mann-Whitney based selection used in classification.

    MHTSK builds multiple sparse subantecedents from random feature
    subsets and jointly optimizes their rule consequents.

    Reference:
        Z. Bian, Q. Chang, J. Wang and N. R. Pal, "Multihead
        Takagi-Sugeno-Kang Fuzzy System," in IEEE Transactions
        on Fuzzy Systems, vol. 33, no. 8, pp. 2561-2573, Aug. 2025,
        doi: 10.1109/TFUZZ.2025.3569227.

    Example:
        ```python
        from highfis import MHTSKRegressor

        reg = MHTSKRegressor()
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int | None = None,
        n_heads: int | None = None,
        head_size: int | None = None,
        head_size_ratio: float | None = None,
        fcm_m: float | None = None,
        rule_sigma: float | None = None,
        fcr_target: float | None = None,
        h_value: float | None = None,
        xi: float | None = None,
        instance_sample_fraction: float | None = None,
        rule_extraction: bool | None = None,
        crcr_us: float | None = None,
        retrain_after_extraction: bool | None = None,
        random_state: int | None = None,
        epochs: int = 10,
        learning_rate: float = 1e-2,
        verbose: bool | int = False,
        batch_size: int | None = 512,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
        patience: int | None = 20,
        restore_best: bool = True,
        weight_decay: float = 1e-8,
        device: str = "cpu",
        paper_strict: bool = False,
    ) -> None:
        """Initialize a MHTSK regressor estimator.

        Args:
            input_configs: Optional list of per-feature input configurations.
                Only the feature names are used for FCM-based MHTSK head construction.
            n_mfs: Number of FCM clusters per head (number of rules generated by each head).
            n_heads: Number of random heads. If ``None``, this is resolved from
                ``head_size`` together with ``fcr_target`` or ``h_value``.
            head_size: Number of features sampled per head. If ``None``, defaults to
                ``round(D * 0.02)`` for ``D <= 5000`` and ``round(D * 0.01)`` for larger inputs.
            head_size_ratio: Alternative relative head size, as a fraction of the input dimension.
            fcm_m: Fuzzification exponent for FCM cluster fitting.
            rule_sigma: Gaussian sigma applied to rule antecedent membership functions.
            fcr_target: Target feature coverage rate for randomly sampled heads.
            h_value: Paper-derived scale constant ``H`` used to compute the number of heads.
                When set, this overrides ``fcr_target``.
            xi: Numeric underflow threshold constant used to bound ``head_size``.
            instance_sample_fraction: Fraction of training instances sampled per head for FCM.
            rule_extraction: If ``True``, perform post-fit rule extraction.
            crcr_us: Unsupervised cumulative rule contribution rate target used in extraction.
            retrain_after_extraction: If ``True``, retrain the extracted rule base after extraction.
            random_state: Random seed for reproducible head construction and FCM initialization.
            epochs: Maximum number of training epochs.
            learning_rate: Adam optimizer learning rate.
            verbose: Verbosity level during training.
            batch_size: Mini-batch size for gradient descent.
            shuffle: Whether to shuffle training samples each epoch.
            ur_weight: Weight of the uncertainty regularization term.
            ur_target: Target firing-level for uncertainty regularization.
            consequent_batch_norm: Apply batch normalization to the consequent layer inputs.
            patience: Early-stopping patience for validation.
            restore_best: Whether to restore the best validation model weights after training.
            weight_decay: Weight decay coefficient for the optimizer.
            device: Target device for training and inference (e.g., ``"cpu"``,
                ``"cuda"``, or ``"mps"``).
            paper_strict: If ``True``, apply paper-derived defaults for MHTSK
                scale and extraction parameters when omitted.

        Notes:
            The regressor supports only unsupervised rule extraction via
            ``crcr_us`` because no label-based Mann-Whitney selection is available.
        """
        (
            resolved_n_mfs,
            resolved_fcm_m,
            resolved_rule_sigma,
            resolved_xi,
            resolved_rule_extraction,
            resolved_crcr_us,
            resolved_retrain_after_extraction,
        ) = _resolve_mhtsk_paper_strict_regressor_config(
            paper_strict=bool(paper_strict),
            n_mfs=n_mfs,
            n_heads=n_heads,
            head_size=head_size,
            head_size_ratio=head_size_ratio,
            fcm_m=fcm_m,
            rule_sigma=rule_sigma,
            fcr_target=fcr_target,
            h_value=h_value,
            xi=xi,
            instance_sample_fraction=instance_sample_fraction,
            rule_extraction=rule_extraction,
            crcr_us=crcr_us,
            retrain_after_extraction=retrain_after_extraction,
        )

        super().__init__(
            input_configs=input_configs,
            n_mfs=resolved_n_mfs,
            mf_init="fcm",
            sigma_scale=1.0,
            random_state=random_state,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            rule_base="custom",
            batch_size=batch_size,
            shuffle=shuffle,
            ur_weight=ur_weight,
            ur_target=ur_target,
            consequent_batch_norm=consequent_batch_norm,
            pfrb_max_rules=None,
            patience=patience,
            restore_best=restore_best,
            weight_decay=weight_decay,
            device=device,
        )
        self.n_heads = int(n_heads) if n_heads is not None else None
        self.head_size = int(head_size) if head_size is not None else None
        self.head_size_ratio = float(head_size_ratio) if head_size_ratio is not None else None
        self.fcm_m = float(resolved_fcm_m)
        self.rule_sigma = float(resolved_rule_sigma)
        self.fcr_target = float(fcr_target) if fcr_target is not None else None
        self.h_value = float(h_value) if h_value is not None else None
        self.xi = float(resolved_xi)
        self.instance_sample_fraction = 0.8 if instance_sample_fraction is None else float(instance_sample_fraction)
        self.rule_extraction = bool(resolved_rule_extraction)
        self.crcr_us = float(resolved_crcr_us)
        self.retrain_after_extraction = bool(resolved_retrain_after_extraction)
        self.paper_strict = bool(paper_strict)
        self._extracted_rule_indices_: list[int] | None = None

    def _build_input_mfs(self, x_arr: np.ndarray) -> tuple[Mapping[str, Sequence[MembershipFunction]], list[str], str]:  # type: ignore[override]
        feature_names = self._resolve_feature_names(x_arr)
        if self.paper_strict:
            head_size, n_heads = _strict_mhtsk_scale_from_dimension(x_arr.shape[1], sigma=self.rule_sigma, xi=self.xi)
        else:
            head_size, n_heads = _resolve_mhtsk_scale_parameters(
                n_features=x_arr.shape[1],
                head_size=self.head_size,
                head_size_ratio=self.head_size_ratio,
                n_heads=self.n_heads,
                fcr_target=self.fcr_target,
                h_value=self.h_value,
                sigma=self.rule_sigma,
                xi=self.xi,
            )
        input_mfs, rules, rule_feature_mask = _build_mhtsk_input_mfs(
            x_arr,
            feature_names=feature_names,
            n_heads=n_heads,
            head_size=head_size,
            n_clusters=int(self.n_mfs),
            fcm_m=self.fcm_m,
            rule_sigma=self.rule_sigma,
            instance_sample_fraction=self.instance_sample_fraction,
            random_state=self.random_state,
        )
        self._mhtsk_rules = rules
        self._mhtsk_rule_feature_mask = torch.as_tensor(rule_feature_mask, dtype=torch.bool)
        return input_mfs, feature_names, "custom"

    def _get_trainer(self) -> BaseTrainer:
        if not self.paper_strict:
            return super()._get_trainer()
        return _MHTSKPaperStrictTrainer(
            epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            batch_size=self.batch_size,
            shuffle=bool(self.shuffle),
            patience=self.patience,
            restore_best=bool(self.restore_best),
            weight_decay=float(self.weight_decay),
            ur_weight=float(self.ur_weight),
            ur_target=self.ur_target,
            verbose=self.verbose,
        )

    def _build_extracted_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_indices: list[int],
    ) -> None:
        if not rule_indices:
            raise ValueError("At least one rule must be selected for extraction")
        self._mhtsk_rules = [self._mhtsk_rules[i] for i in rule_indices]
        self._mhtsk_rule_feature_mask = self._mhtsk_rule_feature_mask[rule_indices]
        self.model_ = self._build_regressor_model(input_mfs, self.rule_base_, None)

    def fit(self, x: Any, y: Any, *, x_val: Any | None = None, y_val: Any | None = None) -> Self:
        """Train the MHTSK regressor and optionally extract rules.

        After the base training step, if ``rule_extraction`` is enabled, rules
        are selected via the unsupervised CRCR criterion on the firing-strength
        matrix.  When ``retrain_after_extraction`` is also set, a second
        training pass is performed on the reduced model.
        """
        x_arr, y_arr = check_X_y(x, y)
        super().fit(x, y, x_val=x_val, y_val=y_val)

        if not bool(self.rule_extraction):
            return self

        x_t = self._as_tensor_x(x_arr)
        self.model_.eval()
        with torch.no_grad():
            norm_w = self.model_.forward_antecedents(x_t)

        selected = _extract_mhtsk_rule_indices_unsupervised(norm_w, self.crcr_us)
        self._extracted_rule_indices_ = selected

        input_mfs = self.model_.input_mfs
        self._build_extracted_model(input_mfs, selected)

        if self.retrain_after_extraction:
            y_t = torch.as_tensor(np.asarray(y_arr), dtype=torch.float32)
            x_val_t: torch.Tensor | None = None
            y_val_t: torch.Tensor | None = None
            if x_val is not None and y_val is not None:
                x_v_arr, y_v_arr = check_X_y(x_val, y_val)
                x_val_t = self._as_tensor_x(x_v_arr)
                y_val_t = torch.as_tensor(np.asarray(y_v_arr, dtype=np.float32), dtype=torch.float32)
            if self.paper_strict:
                trainer = self._get_trainer()
                self.history_ = trainer.fit(self.model_, x_t, y_t, x_val=x_val_t, y_val=y_val_t)
            else:
                self.history_ = self.model_.fit(
                    x_t,
                    y_t,
                    epochs=int(self.epochs),
                    learning_rate=float(self.learning_rate),
                    batch_size=self.batch_size,
                    shuffle=bool(self.shuffle),
                    ur_weight=float(self.ur_weight),
                    ur_target=self.ur_target,
                    verbose=self.verbose,
                    x_val=x_val_t,
                    y_val=y_val_t,
                    patience=self.patience,
                    restore_best=self.restore_best,
                    weight_decay=float(self.weight_decay),
                )
        return self

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        return MHTSKRegressorModel(
            input_mfs,
            self._mhtsk_rule_feature_mask,
            self._mhtsk_rules,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )
