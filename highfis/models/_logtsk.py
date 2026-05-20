"""LogTSK fuzzy model classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import nn

from ..defuzzifiers import InvLogDefuzzifier
from ..layers import (
    ClassificationConsequentLayer,
    RegressionConsequentLayer,
)
from ..memberships import MembershipFunction
from ..t_norms import TNormFn
from ._common import (
    BaseTSKClassifierModel,
    BaseTSKRegressorModel,
)


class LogTSKClassifierModel(BaseTSKClassifierModel):
    r"""LogTSK classifier with inverse-log normalization of log-domain rules.

    Firing strengths are normalized using the inverse-log formula, which
    is immune to softmax saturation in high-dimensional input spaces.

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
        t_norm: str | TNormFn = "prod",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the LogTSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: Rule-base construction strategy.  ``"cartesian"``
                (default, all MF combinations), ``"coco"`` (same-index
                compact), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            t_norm: Antecedent aggregation operator (default ``"prod"``).
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.InvLogDefuzzifier`.
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
            rules=rules,
            defuzzifier=defuzzifier or InvLogDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        """Build classification consequent head."""
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        """Return CrossEntropyLoss as the default classification loss."""
        return nn.CrossEntropyLoss()


class LogTSKRegressorModel(BaseTSKRegressorModel):
    r"""LogTSK regressor with inverse-log normalization of log-domain rules.

    Firing strengths are normalized using the inverse-log formula, which
    is immune to softmax saturation in high-dimensional input spaces.

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
        t_norm: str | TNormFn = "prod",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the LogTSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: Rule-base construction strategy.  ``"cartesian"``
                (default, all MF combinations), ``"coco"`` (same-index
                compact), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            t_norm: Antecedent aggregation operator (default ``"prod"``.
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.InvLogDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
        """
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            rules=rules,
            defuzzifier=defuzzifier or InvLogDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        """Build regression consequent head."""
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        """Return MSELoss as the default regression loss."""
        return nn.MSELoss()
