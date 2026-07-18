"""MHTSK (multi-head TSK) fuzzy model classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor, nn

from ..defuzzifiers import SumBasedDefuzzifier
from ..layers import (
    SparseClassificationConsequentLayer,
    SparseRegressionConsequentLayer,
)
from ..memberships import MembershipFunction
from ..t_norms import TNormFn
from ._common import (
    BaseTSKClassifierModel,
    BaseTSKRegressorModel,
)


class MHTSKClassifierModel(BaseTSKClassifierModel):
    """Multihead TSK classifier with sparse rule consequents.

    MHTSK builds multiple sparse subantecedents from random feature
    subsets and jointly optimizes their rule consequents.

    Reference:
        Z. Bian, Q. Chang, J. Wang and N. R. Pal, "Multihead
        Takagi-Sugeno-Kang Fuzzy System," in IEEE Transactions
        on Fuzzy Systems, vol. 33, no. 8, pp. 2561-2573, Aug. 2025,
        doi: 10.1109/TFUZZ.2025.3569227.
    """

    #: MSE on one-hot targets, matching the MHTSK paper (Bian et al. 2025, eq. 12).
    default_criterion = nn.MSELoss

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_feature_mask: Tensor,
        rules: Sequence[Sequence[int]],
        n_classes: int,
        t_norm: str | TNormFn = "prod",
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialize the MHTSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of membership functions.
            rule_feature_mask: Boolean tensor of shape (n_rules, n_inputs) indicating
                which features are active for each rule.
            rules: Explicit per-rule MF index sequences for all inputs.
            n_classes: Number of output classes.
            t_norm: Antecedent aggregation operator (default ``"prod"``).
            defuzzifier: Custom defuzzifier. Defaults to ``SumBasedDefuzzifier``.
            consequent_batch_norm: Batch normalisation on consequent inputs.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = int(n_classes)
        self.rule_feature_mask = rule_feature_mask
        super().__init__(
            input_mfs,
            rule_base="custom",
            t_norm=t_norm,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        """Build sparse classification consequent head."""
        return SparseClassificationConsequentLayer(
            self.n_rules,
            self.n_inputs,
            self.n_classes,
            self.rule_feature_mask,
        )


class MHTSKRegressorModel(BaseTSKRegressorModel):
    """Multihead TSK regressor with sparse rule consequents.

    MHTSK builds multiple sparse subantecedents from random feature
    subsets and jointly optimizes their rule consequents.

    Reference:
        Z. Bian, Q. Chang, J. Wang and N. R. Pal, "Multihead
        Takagi-Sugeno-Kang Fuzzy System," in IEEE Transactions
        on Fuzzy Systems, vol. 33, no. 8, pp. 2561-2573, Aug. 2025,
        doi: 10.1109/TFUZZ.2025.3569227.

    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_feature_mask: Tensor,
        rules: Sequence[Sequence[int]],
        t_norm: str | TNormFn = "prod",
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialize the MHTSK regressor."""
        self.rule_feature_mask = rule_feature_mask
        super().__init__(
            input_mfs,
            rule_base="custom",
            t_norm=t_norm,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        """Build sparse regression consequent head."""
        return SparseRegressionConsequentLayer(
            self.n_rules,
            self.n_inputs,
            self.rule_feature_mask,
        )
