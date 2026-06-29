"""FSRE-ADATSK model classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from torch import Tensor, nn

from ..defuzzifiers import SoftmaxLogDefuzzifier
from ..layers import (
    AdaSoftminRuleLayer,
    GatedClassificationConsequentLayer,
    GatedRegressionConsequentLayer,
    MembershipLayer,
)
from ..memberships import MembershipFunction
from ._common import (
    BaseTSKClassifierModel,
    BaseTSKRegressorModel,
)


class _FSREADATSKMixin:
    """Shared gate-inspection and feature-pruning methods for FSRE-ADATSK models."""

    # Attributes provided by BaseTSK subclasses; declared here for type checkers.
    input_names: list[str]
    input_mfs: Any
    n_inputs: int
    membership_layer: Any
    consequent_batch_norm: bool
    consequent_bn: Any
    consequent_layer: Any

    def get_feature_gate_values(self) -> Tensor:
        """Return M(λ_d) gate activations for all input features.

        Returns:
            Detached tensor of shape ``(n_inputs,)`` with the gate
            activation value for each feature after the FS phase.
        """
        return self.consequent_layer.gate_fn(self.consequent_layer.lambda_gates).detach()

    def get_rule_gate_values(self) -> Tensor:
        """Return M(θ_r) gate activations for all rules.

        Returns:
            Detached tensor of shape ``(n_rules,)`` with the gate
            activation value for each rule after the RE phase.
        """
        return self.consequent_layer.gate_fn(self.consequent_layer.theta_gates).detach()

    def prune_to_features(self, surviving_features: list[int]) -> None:
        """Structurally prune the model to the given feature subset (paper step 2).

        Updates input_names, input_mfs, n_inputs, membership_layer, and
        optionally consequent_bn in-place.
        The rule layer and consequent layer are intentionally left unchanged
        here; they will be rebuilt from the updated feature set when
        fit_re() calls expand_to_en_frb().

        Args:
            surviving_features: Indices of features to retain.

        Raises:
            ValueError: If *surviving_features* is empty.
        """
        if not surviving_features:
            raise ValueError("surviving_features must not be empty")
        surviving_names = [self.input_names[i] for i in surviving_features]
        new_input_mfs = {name: self.input_mfs[name] for name in surviving_names}
        self.input_names = surviving_names
        self.input_mfs = new_input_mfs
        self.n_inputs = len(surviving_features)
        self.membership_layer = MembershipLayer(new_input_mfs)
        if self.consequent_batch_norm:
            self.consequent_bn = nn.BatchNorm1d(self.n_inputs)


class FSREADATSKClassifierModel(_FSREADATSKMixin, BaseTSKClassifierModel):
    """FSRE-ADATSK classifier with adaptive softmin antecedent and gated consequents.

    FSRE-ADATSK (Feature Selection and Rule Extraction) extends ADATSK.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.
    """

    consequent_layer: GatedClassificationConsequentLayer

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "coco",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
        use_en_frb: bool = False,
    ) -> None:
        """Initialise the FSRE-ADATSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                MembershipFunction objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB; see also
                *use_en_frb*), or ``"custom"`` (explicit rules via *rules*).
            rules: Explicit rule antecedent indices; ignored when
                ``use_en_frb=True``.
            defuzzifier: Custom defuzzifier. Defaults to
                SoftmaxLogDefuzzifier.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            eps: Numerical stability epsilon for the Ada-softmin operator.
            use_en_frb: Start directly from the Enhanced FRB (En-FRB)
                instead of CoCo-FRB.

        Raises:
            ValueError: If ``n_classes < 2``.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")

        self.n_classes = int(n_classes)
        self.eps = eps
        self.use_en_frb = bool(use_en_frb)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            rules=rules,
            defuzzifier=defuzzifier or SoftmaxLogDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = AdaSoftminRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules if not self.use_en_frb else None,
            rule_base="en" if self.use_en_frb else rule_base,
            eps=self.eps,
        )
        self.n_rules = self.rule_layer.n_rules
        self.consequent_layer = self._build_consequent_layer()

    def _build_consequent_layer(self) -> GatedClassificationConsequentLayer:
        layer = GatedClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes, shared_lambda=True)
        layer.mode = "fs"
        return layer

    def set_consequent_mode(self, mode: Literal["fs", "re", "finetune", "both"]) -> None:
        """Set training mode for the consequent layer."""
        self.consequent_layer.mode = mode

    def _default_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def expand_to_en_frb(self) -> None:
        """Switch the rule layer to an Enhanced Fuzzy Rule Base for RE phase."""
        self.rule_layer = AdaSoftminRuleLayer(
            self.input_names,
            [len(self.input_mfs[name]) for name in self.input_names],
            rule_base="en",
            eps=self.eps,
        )
        self.n_rules = self.rule_layer.n_rules
        self.consequent_layer = self._build_consequent_layer()

    def prune_to_rules(self, surviving_rules: list[int]) -> None:
        """Structurally prune the model to the given rule subset (paper step 4).

        Rebuilds rule_layer and consequent_layer in-place,
        retaining only the specified rules.  Consequent weights, bias, and
        gate parameters for the surviving rules are copied from the current
        layers.  The new consequent layer is set to ``mode="finetune"``
        ready for phase 3.

        Args:
            surviving_rules: Indices of rules to retain.

        Raises:
            ValueError: If *surviving_rules* is empty.
        """
        if not surviving_rules:
            raise ValueError("surviving_rules must not be empty")
        mf_per_input = [len(self.input_mfs[name]) for name in self.input_names]
        new_rules = [self.rule_layer.rules[r] for r in surviving_rules]
        new_n_rules = len(surviving_rules)

        self.rule_layer = AdaSoftminRuleLayer(
            self.input_names,
            mf_per_input,
            rules=new_rules,
            eps=self.eps,
        )
        self.n_rules = new_n_rules

        old_cons = self.consequent_layer
        new_cons = GatedClassificationConsequentLayer(new_n_rules, self.n_inputs, self.n_classes, shared_lambda=True)
        new_cons.mode = "finetune"
        cast(Tensor, new_cons.theta_gates.data).copy_(old_cons.theta_gates.data[surviving_rules])
        cast(Tensor, new_cons.lambda_gates.data).copy_(old_cons.lambda_gates.data)
        cast(Tensor, new_cons.weight.data).copy_(old_cons.weight.data[surviving_rules])
        cast(Tensor, new_cons.bias.data).copy_(old_cons.bias.data[surviving_rules])
        self.consequent_layer = new_cons


class FSREADATSKRegressorModel(_FSREADATSKMixin, BaseTSKRegressorModel):
    """FSRE-ADATSK regressor with adaptive softmin antecedent and gated consequents.

    FSRE-ADATSK (Feature Selection and Rule Extraction) extends ADATSK.

    Reference:
        G. Xue, Q. Chang, J. Wang, K. Zhang and N. R. Pal, "An Adaptive
        Neuro-Fuzzy System With Integrated Feature Selection and Rule
        Extraction for High-Dimensional Classification Problems," in
        IEEE Transactions on Fuzzy Systems, vol. 31, no. 7, pp. 2167-2181,
        July 2023, doi: 10.1109/TFUZZ.2022.3220950.
    """

    consequent_layer: GatedRegressionConsequentLayer

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "coco",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
        use_en_frb: bool = False,
    ) -> None:
        """Initialise the FSRE-ADATSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                MembershipFunction objects.
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB; see also
                *use_en_frb*), or ``"custom"`` (explicit rules via *rules*).
            rules: Explicit rule antecedent indices; ignored when
                ``use_en_frb=True``.
            defuzzifier: Custom defuzzifier. Defaults to
                SoftmaxLogDefuzzifier.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            eps: Numerical stability epsilon for the Ada-softmin operator.
            use_en_frb: Start directly from the Enhanced FRB (En-FRB).
        """
        self.eps = eps
        self.use_en_frb = bool(use_en_frb)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            rules=rules,
            defuzzifier=defuzzifier or SoftmaxLogDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = AdaSoftminRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules if not self.use_en_frb else None,
            rule_base="en" if self.use_en_frb else rule_base,
            eps=self.eps,
        )
        self.n_rules = self.rule_layer.n_rules
        self.consequent_layer = self._build_consequent_layer()

    def _build_consequent_layer(self) -> GatedRegressionConsequentLayer:
        layer = GatedRegressionConsequentLayer(self.n_rules, self.n_inputs, shared_lambda=True)
        layer.mode = "fs"
        return layer

    def set_consequent_mode(self, mode: Literal["fs", "re", "finetune", "both"]) -> None:
        """Set training mode for the consequent layer."""
        self.consequent_layer.mode = mode

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def expand_to_en_frb(self) -> None:
        """Switch the rule layer to an Enhanced Fuzzy Rule Base for RE phase."""
        self.rule_layer = AdaSoftminRuleLayer(
            self.input_names,
            [len(self.input_mfs[name]) for name in self.input_names],
            rule_base="en",
            eps=self.eps,
        )
        self.n_rules = self.rule_layer.n_rules
        self.consequent_layer = self._build_consequent_layer()

    def prune_to_rules(self, surviving_rules: list[int]) -> None:
        """Structurally prune the model to the given rule subset (paper step 4).

        Rebuilds rule_layer and consequent_layer in-place,
        retaining only the specified rules.  Consequent weights, bias, and
        gate parameters for the surviving rules are copied from the current
        layers.  The new consequent layer is set to ``mode="finetune"``
        ready for phase 3.

        Args:
            surviving_rules: Indices of rules to retain.

        Raises:
            ValueError: If *surviving_rules* is empty.
        """
        if not surviving_rules:
            raise ValueError("surviving_rules must not be empty")
        mf_per_input = [len(self.input_mfs[name]) for name in self.input_names]
        new_rules = [self.rule_layer.rules[r] for r in surviving_rules]
        new_n_rules = len(surviving_rules)

        self.rule_layer = AdaSoftminRuleLayer(
            self.input_names,
            mf_per_input,
            rules=new_rules,
            eps=self.eps,
        )
        self.n_rules = new_n_rules

        old_cons = self.consequent_layer
        new_cons = GatedRegressionConsequentLayer(new_n_rules, self.n_inputs, shared_lambda=True)
        new_cons.mode = "finetune"
        cast(Tensor, new_cons.theta_gates.data).copy_(old_cons.theta_gates.data[surviving_rules])
        cast(Tensor, new_cons.lambda_gates.data).copy_(old_cons.lambda_gates.data)
        cast(Tensor, new_cons.weight.data).copy_(old_cons.weight.data[surviving_rules])
        cast(Tensor, new_cons.bias.data).copy_(old_cons.bias.data[surviving_rules])
        self.consequent_layer = new_cons
