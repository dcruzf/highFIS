"""HTSK and vanilla TSK fuzzy model classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import nn

from ..defuzzifiers import SumBasedDefuzzifier
from ..layers import (
    ClassificationConsequentLayer,
    RegressionConsequentLayer,
)
from ..memberships import MembershipFunction
from ..t_norms import TNormFn
from ._common import (
    BaseTSKClassifier,
    BaseTSKRegressor,
)


class HTSKClassifier(BaseTSKClassifier):
    r"""HTSK classifier for high-dimensional TSK inference.

    HTSK replaces the standard product t-norm with a geometric mean over
    membership values and performs rule normalization in log-space.

    References:
        Y. Cui, D. Wu and Y. Xu, "Curse of Dimensionality for TSK Fuzzy
        Neural Networks: Explanation and Solutions," 2021 International
        Joint Conference on Neural Networks (IJCNN), Shenzhen, China,
        2021, pp. 1-8, doi: 10.1109/IJCNN52387.2021.9534265.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "cartesian",
        t_norm: str = "gmean",
        t_norm_fn: TNormFn | None = None,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the HTSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: Rule-base construction strategy.  ``"cartesian"``
                builds the full Cartesian product; ``"coco"`` uses a
                one-cluster-per-rule scheme.
            t_norm: Antecedent aggregation operator name (default
                ``"gmean"`` for HTSK).
            t_norm_fn: Optional custom t-norm callable; overrides
                ``t_norm`` when provided.
            rules: Explicit rule antecedent indices.  If ``None``, rules
                are inferred from ``rule_base``.
            defuzzifier: Custom defuzzifier module.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
            consequent_batch_norm: Apply batch normalisation to the
                consequent layer inputs.

        Raises:
            ValueError: If ``n_classes < 2``.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = int(n_classes)
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            t_norm_fn=t_norm_fn,
            rules=rules,
            defuzzifier=defuzzifier,
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        """Build classification consequent head."""
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        """Return CrossEntropyLoss as the default classification loss."""
        return nn.CrossEntropyLoss()


class HTSKRegressor(BaseTSKRegressor):
    r"""HTSK regressor for high-dimensional TSK inference.

    HTSK replaces the standard product t-norm with a geometric mean over
    membership values and performs rule normalization in log-space.

    References:
        Y. Cui, D. Wu and Y. Xu, "Curse of Dimensionality for TSK Fuzzy
        Neural Networks: Explanation and Solutions," 2021 International
        Joint Conference on Neural Networks (IJCNN), Shenzhen, China,
        2021, pp. 1-8, doi: 10.1109/IJCNN52387.2021.9534265.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "cartesian",
        t_norm: str = "gmean",
        t_norm_fn: TNormFn | None = None,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the HTSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: Rule-base construction strategy (``"cartesian"`` or
                ``"coco"``).
            t_norm: Antecedent aggregation operator (default ``"gmean"``).
            t_norm_fn: Optional custom t-norm callable.
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
        """
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            t_norm_fn=t_norm_fn,
            rules=rules,
            defuzzifier=defuzzifier,
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        """Build regression consequent head."""
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        """Return MSELoss as the default regression loss."""
        return nn.MSELoss()


class TSKClassifier(BaseTSKClassifier):
    r"""Vanilla TSK classifier with sum-based rule normalization.

    The vanilla Takagi-Sugeno-Kang inference computes rule firing strengths
    with the product t-norm and normalizes them by their total sum.

    References:
        T. Takagi and M. Sugeno, "Fuzzy identification of systems and
        its applications to modeling and control," in IEEE
        Transactions on Systems, Man, and Cybernetics, vol. SMC-15,
        no. 1, pp. 116-132, Jan.-Feb. 1985,
        doi: 10.1109/TSMC.1985.6313399.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "cartesian",
        t_norm: str = "prod",
        t_norm_fn: TNormFn | None = None,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the vanilla TSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: ``"cartesian"`` or ``"coco"`` rule-base strategy.
            t_norm: Antecedent aggregation operator (default ``"prod"``).
            t_norm_fn: Optional custom t-norm callable.
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.

        Raises:
            ValueError: If ``n_classes < 2``.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = int(n_classes)
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            t_norm_fn=t_norm_fn,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        """Build classification consequent head."""
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        """Return CrossEntropyLoss as the default classification loss."""
        return nn.CrossEntropyLoss()


class TSKRegressor(BaseTSKRegressor):
    r"""Vanilla TSK regressor with sum-based rule normalization.

    The vanilla Takagi-Sugeno-Kang inference computes rule firing strengths
    with the product t-norm and normalizes them by their total sum.

    References:
        T. Takagi and M. Sugeno, "Fuzzy identification of systems and
        its applications to modeling and control," in IEEE
        Transactions on Systems, Man, and Cybernetics, vol. SMC-15,
        no. 1, pp. 116-132, Jan.-Feb. 1985,
        doi: 10.1109/TSMC.1985.6313399.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "cartesian",
        t_norm: str = "prod",
        t_norm_fn: TNormFn | None = None,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the vanilla TSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: ``"cartesian"`` or ``"coco"`` rule-base strategy.
            t_norm: Antecedent aggregation operator (default ``"prod"``).
            t_norm_fn: Optional custom t-norm callable.
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
        """
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            t_norm_fn=t_norm_fn,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        """Build regression consequent head."""
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        """Return MSELoss as the default regression loss."""
        return nn.MSELoss()
