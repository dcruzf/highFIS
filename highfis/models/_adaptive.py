"""ADATSK and ADPTSK fuzzy model classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import nn

from ..defuzzifiers import SumBasedDefuzzifier
from ..layers import (
    AdaSoftminRuleLayer,
    ADPSoftminRuleLayer,
    ClassificationConsequentLayer,
    RegressionConsequentLayer,
)
from ..memberships import MembershipFunction
from ._common import (
    BaseTSKClassifierModel,
    BaseTSKRegressorModel,
)


class ADATSKClassifierModel(BaseTSKClassifierModel):
    r"""TSK classifier with adaptive softmin antecedent (ADATSK).

    The firing strength of each rule is computed with the Ada-softmin operator.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "coco",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
    ) -> None:
        """Initialise the ADATSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            eps: Numerical stability epsilon for the Ada-softmin operator.

        Raises:
            ValueError: If ``n_classes < 2``.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")

        self.n_classes = int(n_classes)
        self.eps = eps

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = AdaSoftminRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules,
            rule_base=rule_base,
            eps=self.eps,
        )

    def _build_consequent_layer(self) -> nn.Module:
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class ADATSKRegressorModel(BaseTSKRegressorModel):
    r"""TSK regressor with adaptive softmin antecedent (ADATSK).

    The firing strength of each rule is computed with the Ada-softmin operator.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "coco",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
    ) -> None:
        """Initialise the ADATSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            eps: Numerical stability epsilon for the Ada-softmin operator.
        """
        self.eps = eps

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = AdaSoftminRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules,
            rule_base=rule_base,
            eps=self.eps,
        )

    def _build_consequent_layer(self) -> nn.Module:
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()


class ADPTSKClassifierModel(BaseTSKClassifierModel):
    r"""TSK classifier with adaptive double-parameter softmin antecedent (ADPTSK).

    The firing strengths of each rule are computed with the ADP-softmin
    operator, and membership functions are wrapped as Gaussian PIMFs to
    preserve a positive infimum during high-dimensional training.

    Reference:
        Ma, M., Qian, L., Zhang, Y., Fang, Q., & Xue, G. (2025). An
        adaptive double-parameter softmin based Takagi-Sugeno-Kang
        fuzzy system for high-dimensional data. Fuzzy Sets and
        Systems, 521, 109582.
        https://doi.org/10.1016/j.fss.2025.109582
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "coco",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        kappa: float = 690.0,
        xi: float = 730.0,
        eps: float | None = None,
    ) -> None:
        """Initialise the ADPTSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            kappa: ADP-softmin parameter ``κ > 0`` (default ``690.0``).
            xi: ADP-softmin parameter ``ξ > 0`` (default ``730.0``).
            eps: Numerical stability epsilon.

        Raises:
            ValueError: If ``n_classes < 2``, ``kappa <= 0``, or ``xi <= 0``.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if kappa <= 0.0:
            raise ValueError("kappa must be > 0")
        if xi <= 0.0:
            raise ValueError("xi must be > 0")

        self.n_classes = int(n_classes)
        self.kappa = float(kappa)
        self.xi = float(xi)
        self.eps = eps

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = ADPSoftminRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules,
            rule_base=rule_base,
            kappa=self.kappa,
            xi=self.xi,
            eps=self.eps,
        )

    def _build_consequent_layer(self) -> nn.Module:
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class ADPTSKRegressorModel(BaseTSKRegressorModel):
    r"""TSK regressor with adaptive double-parameter softmin antecedent (ADPTSK).

    The firing strengths of each rule are computed with the ADP-softmin
    operator, and membership functions are wrapped as Gaussian PIMFs to
    preserve a positive infimum during high-dimensional training.

    Reference:
        Ma, M., Qian, L., Zhang, Y., Fang, Q., & Xue, G. (2025). An
        adaptive double-parameter softmin based Takagi-Sugeno-Kang
        fuzzy system for high-dimensional data. Fuzzy Sets and
        Systems, 521, 109582.
        https://doi.org/10.1016/j.fss.2025.109582
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "coco",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        kappa: float = 690.0,
        xi: float = 730.0,
        eps: float | None = None,
    ) -> None:
        """Initialise the ADPTSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            kappa: ADP-softmin parameter ``κ > 0`` (default ``690.0``).
            xi: ADP-softmin parameter ``ξ > 0`` (default ``730.0``).
            eps: Numerical stability epsilon.

        Raises:
            ValueError: If ``kappa <= 0`` or ``xi <= 0``.
        """
        if kappa <= 0.0:
            raise ValueError("kappa must be > 0")
        if xi <= 0.0:
            raise ValueError("xi must be > 0")
        self.kappa = float(kappa)
        self.xi = float(xi)
        self.eps = eps

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = ADPSoftminRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules,
            rule_base=rule_base,
            kappa=self.kappa,
            xi=self.xi,
            eps=self.eps,
        )

    def _build_consequent_layer(self) -> nn.Module:
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()
