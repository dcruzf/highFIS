"""HDFIS (product and min T-norm) fuzzy model classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from ..defuzzifiers import SumBasedDefuzzifier
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


def _zero_initialize_consequents(consequent_layer: nn.Module) -> None:
    weight = getattr(consequent_layer, "weight", None)
    if isinstance(weight, torch.Tensor):
        nn.init.zeros_(weight)
    bias = getattr(consequent_layer, "bias", None)
    if isinstance(bias, torch.Tensor):
        nn.init.zeros_(bias)


class HDFISProdClassifierModel(BaseTSKClassifierModel):
    r"""HDFIS-prod classifier with dimension-dependent Gaussian MFs.

    HDFIS-prod combines the standard product T-norm with a dimension-dependent
    Gaussian membership function (DMF) to avoid numeric underflow in very
    high-dimensional feature spaces while preserving first-order TSK
    consequents.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.
    """

    #: MSE on one-hot targets, matching the HDFIS paper (Xue et al. 2023, eq. 14).
    default_criterion = nn.MSELoss

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "cartesian",
        t_norm: str | TNormFn = "prod",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        zero_consequent_init: bool = False,
    ) -> None:
        """Initialize the HDFIS-prod classifier."""
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = int(n_classes)
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )
        if zero_consequent_init:
            _zero_initialize_consequents(self.consequent_layer)

    def _build_consequent_layer(self) -> nn.Module:
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)


class HDFISProdRegressorModel(BaseTSKRegressorModel):
    r"""HDFIS-prod regressor with dimension-dependent Gaussian MFs.

    HDFIS-prod combines the standard product T-norm with a dimension-dependent
    Gaussian membership function (DMF) to avoid numeric underflow in very
    high-dimensional feature spaces while preserving first-order TSK
    consequents.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.

    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "cartesian",
        t_norm: str | TNormFn = "prod",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        zero_consequent_init: bool = False,
    ) -> None:
        """Initialize the HDFIS-prod regressor."""
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )
        if zero_consequent_init:
            _zero_initialize_consequents(self.consequent_layer)

    def _build_consequent_layer(self) -> nn.Module:
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)


class HDFISMinClassifierModel(BaseTSKClassifierModel):
    r"""HDFIS-min classifier with frozen antecedents and minimum aggregation.

    HDFIS-min uses the minimum T-norm in the antecedent and only optimizes
    consequent parameters, which avoids the nondifferentiability of the
    minimum operator during training.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.
    """

    #: MSE on one-hot targets, matching the HDFIS paper (Xue et al. 2023, eq. 14).
    default_criterion = nn.MSELoss

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "cartesian",
        t_norm: str | TNormFn = "min",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        zero_consequent_init: bool = False,
    ) -> None:
        """Initialize the HDFIS-min classifier."""
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = int(n_classes)
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )
        if zero_consequent_init:
            _zero_initialize_consequents(self.consequent_layer)
        for param in self.membership_layer.parameters():
            param.requires_grad = False

    def _build_consequent_layer(self) -> nn.Module:
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)


class HDFISMinRegressorModel(BaseTSKRegressorModel):
    r"""HDFIS-min regressor with frozen antecedents and minimum aggregation.

    HDFIS-min uses the minimum T-norm in the antecedent and only optimizes
    consequent parameters, which avoids the nondifferentiability of the
    minimum operator during training.

    References:
        G. Xue, J. Wang, K. Zhang and N. R. Pal, "High-Dimensional Fuzzy
        Inference Systems," in IEEE Transactions on Systems, Man, and
        Cybernetics: Systems, vol. 54, no. 1, pp. 507-519, Jan. 2024,
        doi: 10.1109/TSMC.2023.3311475.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "cartesian",
        t_norm: str | TNormFn = "min",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        zero_consequent_init: bool = False,
    ) -> None:
        """Initialize the HDFIS-min regressor."""
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )
        if zero_consequent_init:
            _zero_initialize_consequents(self.consequent_layer)
        for param in self.membership_layer.parameters():
            param.requires_grad = False

    def _build_consequent_layer(self) -> nn.Module:
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)
