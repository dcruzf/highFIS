"""AYA-TSK (Yager-based) fuzzy model classes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from ..defuzzifiers import SumBasedDefuzzifier
from ..layers import (
    ClassificationConsequentLayer,
    RegressionConsequentLayer,
)
from ..memberships import MembershipFunction
from ..t_norms import TNormFn, YagerTNorm
from ._common import (
    BaseTSKClassifierModel,
    BaseTSKRegressorModel,
)


def _adaptive_yager_lambda(dimension: int, lower_bound: float) -> float:
    if dimension <= 1:
        raise ValueError("dimension must be > 1")
    if not 0.0 < lower_bound < 1.0:
        raise ValueError("lower_bound must be in (0, 1)")
    return -math.log(float(dimension)) / math.log(1.0 - float(lower_bound))


def _infer_lower_bound(input_mfs: Mapping[str, Sequence[MembershipFunction]]) -> float:
    lower_bounds: list[float] = []
    for mfs in input_mfs.values():
        for mf in mfs:
            k = getattr(mf, "k", None)
            if k is not None:
                lower_bounds.append(1.0 / float(k))
    return min(lower_bounds) if lower_bounds else 1.0 / math.e


def _zero_initialize_consequents(consequent_layer: nn.Module) -> None:
    weight = getattr(consequent_layer, "weight", None)
    if isinstance(weight, torch.Tensor):
        nn.init.zeros_(weight)
    bias = getattr(consequent_layer, "bias", None)
    if isinstance(bias, torch.Tensor):
        nn.init.zeros_(bias)


class AYATSKClassifierModel(BaseTSKClassifierModel):
    r"""TSK classifier with an adaptive Yager T-norm in the antecedent.

    AYATSK extends TSK by using an adaptive Yager T-norm aggregation and
    optional positive lower-bound membership functions to improve
    stability and performance in high-dimensional settings.

    Reference:
        G. Xue, Y. Yang and J. Wang, "Adaptive Yager T-Norm-Based
        Takagi-Sugeno-Kang Fuzzy Systems," in IEEE Transactions on
        Systems, Man, and Cybernetics: Systems, vol. 55, no. 12,
        pp. 9802-9815, Dec. 2025, doi: 10.1109/TSMC.2025.3621346.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "coco",
        t_norm: str | TNormFn = "yager",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the AYATSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            t_norm: T-norm name or callable (default ``"yager"``).
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
        self.lower_bound_ = _infer_lower_bound(input_mfs)
        self.lambda_ = _adaptive_yager_lambda(len(input_mfs), self.lower_bound_)
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=YagerTNorm(lambda_=self.lambda_) if t_norm == "yager" else t_norm,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )
        _zero_initialize_consequents(self.consequent_layer)

    def _build_consequent_layer(self) -> nn.Module:
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    default_criterion = nn.MSELoss

    def _get_optimizer_config(
        self,
        learning_rate: float,
        weight_decay: float,
    ) -> tuple[type[torch.optim.Optimizer], list[dict[str, Any]]]:
        ante_params = list(self.membership_layer.parameters())
        rule_params = list(self.rule_layer.parameters())
        cons_params = list(self.consequent_layer.parameters())
        if self.consequent_bn is not None:
            cons_params.extend(self.consequent_bn.parameters())
        return torch.optim.Adam, [
            {"params": ante_params, "weight_decay": weight_decay},
            {"params": rule_params, "weight_decay": weight_decay},
            {"params": cons_params, "weight_decay": weight_decay},
        ]


class AYATSKRegressorModel(BaseTSKRegressorModel):
    r"""TSK regressor with an adaptive Yager T-norm in the antecedent.

    AYATSK extends TSK by using an adaptive Yager T-norm aggregation and
    optional positive lower-bound membership functions to improve
    stability and performance in high-dimensional settings.

    Reference:
        G. Xue, Y. Yang and J. Wang, "Adaptive Yager T-Norm-Based
        Takagi-Sugeno-Kang Fuzzy Systems," in IEEE Transactions on
        Systems, Man, and Cybernetics: Systems, vol. 55, no. 12,
        pp. 9802-9815, Dec. 2025, doi: 10.1109/TSMC.2025.3621346.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "coco",
        t_norm: str | TNormFn = "yager",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the AYATSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            t_norm: T-norm name or callable (default ``"yager"``).
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
        """
        self.lower_bound_ = _infer_lower_bound(input_mfs)
        self.lambda_ = _adaptive_yager_lambda(len(input_mfs), self.lower_bound_)
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=YagerTNorm(lambda_=self.lambda_) if t_norm == "yager" else t_norm,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )
        _zero_initialize_consequents(self.consequent_layer)

    def _build_consequent_layer(self) -> nn.Module:
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    default_criterion = nn.MSELoss

    def _get_optimizer_config(
        self,
        learning_rate: float,
        weight_decay: float,
    ) -> tuple[type[torch.optim.Optimizer], list[dict[str, Any]]]:
        ante_params = list(self.membership_layer.parameters())
        rule_params = list(self.rule_layer.parameters())
        cons_params = list(self.consequent_layer.parameters())
        if self.consequent_bn is not None:
            cons_params.extend(self.consequent_bn.parameters())
        return torch.optim.Adam, [
            {"params": ante_params, "weight_decay": weight_decay},
            {"params": rule_params, "weight_decay": weight_decay},
            {"params": cons_params, "weight_decay": weight_decay},
        ]
