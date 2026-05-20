"""Sklearn-compatible estimators for AdaTSK and ADPTSK models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..base import BaseTSK
from ..memberships import (
    MembershipFunction,
)
from ..models import (
    AdaTSKClassifier,
    AdaTSKRegressor,
    ADPTSKClassifier,
    ADPTSKRegressor,
)
from ._base import (
    InputConfig,
    _BaseClassifierEstimator,
    _BaseRegressorEstimator,
    _wrap_gaussian_pimf_input_mfs,
)


class ADPTSKClassifierEstimator(_BaseClassifierEstimator):
    r"""TSK classifier with ADP-softmin antecedent and Gaussian PIMF.

    The firing strengths of each rule are computed with the ADP-softmin
    operator, and membership functions are wrapped as Gaussian PIMFs to
    preserve a positive infimum during high-dimensional training.

    Reference:
        Ma, M., Qian, L., Zhang, Y., Fang, Q., & Xue, G. (2025). An
        adaptive double-parameter softmin based Takagi-Sugeno-Kang
        fuzzy system for high-dimensional data. Fuzzy Sets and
        Systems, 521, 109582.
        https://doi.org/10.1016/j.fss.2025.109582

    Example:
        ```python
        from highfis import ADPTSKClassifierEstimator

        clf = ADPTSKClassifierEstimator()
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 3,
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
        kappa: float = 690.0,
        xi: float = 730.0,
        k: float = 1.0,
        eps: float | None = None,
    ) -> None:
        """Initialise an ADPTSK classifier estimator.

        Args:
            input_configs: Optional list of :class:`InputConfig` instances,
                one per feature. Only ``name`` is used when
                ``mf_init="kmeans"``.
            n_mfs: Number of membership functions per feature or k-means
                clusters.
            mf_init: Membership-function initialization strategy.
                ``"kmeans"`` or ``"grid"``.
            sigma_scale: Scale factor for Gaussian MF sigma initialization.
            random_state: Seed for k-means and PyTorch weight initialization.
            epochs: Maximum number of training epochs.
            learning_rate: Initial learning rate for the Adam optimizer.
            verbose: Verbosity level for training output.
            rule_base: Rule-base strategy, e.g. ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size. ``None`` uses the full dataset.
            shuffle: Whether to shuffle training samples each epoch.
            ur_weight: Uniform-rule regularization weight.
            ur_target: Target average rule activation for UR.
            consequent_batch_norm: Apply batch normalization to consequent
                linear layers.
            pfrb_max_rules: Maximum rules for point-based FRB when
                ``rule_base="pfrb"``.
            patience: Early-stopping patience. ``None`` disables early stopping.
            restore_best: Restore the best validation model weights after
                training.
                stopping.
            weight_decay: L2 weight decay coefficient for consequent parameters.
            kappa: ADPTSK ``κ`` parameter controlling the double-softmin
                geometry.
            xi: ADPTSK ``ξ`` parameter controlling adaptive softmin sharpness.
            k: Gaussian PIMF scaling constant used when wrapping the input MFs.
            eps: Optional lower bound for Gaussian PIMF values.
        """
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
        self.kappa = float(kappa)
        self.xi = float(xi)
        self.k = float(k)
        self.eps = eps

    def _build_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str,
    ) -> BaseTSK:
        input_mfs = _wrap_gaussian_pimf_input_mfs(input_mfs, k=self.k, eps=self.eps)
        return ADPTSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            kappa=self.kappa,
            xi=self.xi,
            eps=self.eps,
        )


class ADPTSKRegressorEstimator(_BaseRegressorEstimator):
    r"""TSK regressor with ADP-softmin antecedent and Gaussian PIMF.

    The firing strengths of each rule are computed with the ADP-softmin
    operator, and membership functions are wrapped as Gaussian PIMFs to
    preserve a positive infimum during high-dimensional training.

    Reference:
        Ma, M., Qian, L., Zhang, Y., Fang, Q., & Xue, G. (2025). An
        adaptive double-parameter softmin based Takagi-Sugeno-Kang
        fuzzy system for high-dimensional data. Fuzzy Sets and
        Systems, 521, 109582.
        https://doi.org/10.1016/j.fss.2025.109582

    Example:
        ```python
        from highfis import ADPTSKRegressorEstimator

        reg = ADPTSKRegressorEstimator()
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 3,
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
        kappa: float = 690.0,
        xi: float = 730.0,
        k: float = 1.0,
        eps: float | None = None,
    ) -> None:
        """Initialise an ADPTSK regressor estimator.

        Args:
            input_configs: Optional list of :class:`InputConfig` instances,
                one per feature. Only ``name`` is used when
                ``mf_init="kmeans"``.
            n_mfs: Number of membership functions per feature or k-means
                clusters.
            mf_init: Membership-function initialization strategy.
                ``"kmeans"`` or ``"grid"``.
            sigma_scale: Scale factor for Gaussian MF sigma initialization.
            random_state: Seed for k-means and PyTorch weight initialization.
            epochs: Maximum number of training epochs.
            learning_rate: Initial learning rate for the Adam optimizer.
            verbose: Verbosity level for training output.
            rule_base: Rule-base strategy, e.g. ``"coco"`` or ``"cartesian"``.
            batch_size: Mini-batch size. ``None`` uses the full dataset.
            shuffle: Whether to shuffle training samples each epoch.
            ur_weight: Uniform-rule regularization weight.
            ur_target: Target average rule activation for UR.
            consequent_batch_norm: Apply batch normalization to consequent
                linear layers.
            pfrb_max_rules: Maximum rules for point-based FRB when
                ``rule_base="pfrb"``.
            patience: Early-stopping patience. ``None`` disables early stopping.
            restore_best: Restore the best validation model weights after
                training.
                stopping.
            weight_decay: L2 weight decay coefficient for consequent parameters.
            kappa: ADPTSK ``κ`` parameter controlling the double-softmin
                geometry.
            xi: ADPTSK ``ξ`` parameter controlling adaptive softmin sharpness.
            k: Gaussian PIMF scaling constant used when wrapping the input MFs.
            eps: Optional lower bound for Gaussian PIMF values.
        """
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
        self.kappa = float(kappa)
        self.xi = float(xi)
        self.k = float(k)
        self.eps = eps

    def _build_regressor_model(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str,
        n_classes: int | None = None,
    ) -> BaseTSK:
        input_mfs = _wrap_gaussian_pimf_input_mfs(input_mfs, k=self.k, eps=self.eps)
        return ADPTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
            kappa=self.kappa,
            xi=self.xi,
            eps=self.eps,
        )


class AdaTSKClassifierEstimator(_BaseClassifierEstimator):
    r"""TSK classifier with adaptive softmin antecedent (AdaTSK).

    The firing strength of each rule is computed with the Ada-softmin operator.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import AdaTSKClassifierEstimator

        clf = AdaTSKClassifierEstimator(n_mfs=30, random_state=0)
        clf.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
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
        """Initialise an AdaTSK classifier.

        Args:
            input_configs: Per-feature :class:`InputConfig` list. Only
                ``name`` is used when ``mf_init="kmeans"``.
            n_mfs: Number of k-means clusters / grid MFs (default ``5``).
            mf_init: ``"kmeans"`` (default) or ``"grid"``.
            sigma_scale: Sigma scale factor. ``1.0`` recommended; Ada-softmin
                handles high-dimensional stability.
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
        """
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
        """Create AdaTSKClassifier."""
        return AdaTSKClassifier(
            input_mfs,
            n_classes=n_classes,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )


class AdaTSKRegressorEstimator(_BaseRegressorEstimator):
    r"""TSK regressor with adaptive softmin antecedent (AdaTSK).

    The firing strength of each rule is computed with the Ada-softmin operator.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.

    Example:
        ```python
        from highfis import AdaTSKRegressorEstimator

        reg = AdaTSKRegressorEstimator(n_mfs=30, random_state=0)
        reg.fit(X_train, y_train)
        ```
    """

    def __init__(
        self,
        *,
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
        """Initialise an AdaTSK regressor.

        Args:
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
        """
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
        """Create AdaTSKRegressor."""
        return AdaTSKRegressor(
            input_mfs,
            rule_base=rule_base,
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )
