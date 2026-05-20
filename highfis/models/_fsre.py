"""FSRE-AdaTSK model classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from ..defuzzifiers import SoftmaxLogDefuzzifier
from ..layers import (
    AdaSoftminRuleLayer,
    GatedClassificationConsequentLayer,
    GatedRegressionConsequentLayer,
)
from ..memberships import MembershipFunction
from ._common import (
    BaseTSKClassifier,
    BaseTSKRegressor,
)


class FSREAdaTSKClassifier(BaseTSKClassifier):
    r"""FSRE-AdaTSK classifier with adaptive softmin antecedent and gated consequents.

    FSRE-AdaTSK (Feature Selection and Rule Extraction) extends AdaTSK.

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
        """Initialise the FSRE-AdaTSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB; see also
                *use_en_frb*), or ``"custom"`` (explicit rules via *rules*).
            rules: Explicit rule antecedent indices; ignored when
                ``use_en_frb=True``.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
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

    def fit_fs(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Train the FS phase: only feature gates M(λ_d) are active (eq. 21)."""
        self.consequent_layer.mode = "fs"
        return self.fit(x, y, **kwargs)

    def fit_re(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Expand to En-FRB and train the RE phase: only rule gates M(θ_r) active (eq. 22)."""
        self.expand_to_en_frb()
        self.consequent_layer.mode = "re"
        return self.fit(x, y, **kwargs)

    def fit_finetune(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Fine-tune with no gates — plain TSK consequent (eq. 5)."""
        self.consequent_layer.mode = "finetune"
        return self.fit(x, y, **kwargs)


class FSREAdaTSKRegressor(BaseTSKRegressor):
    r"""FSRE-AdaTSK regressor with adaptive softmin antecedent and gated consequents.

    FSRE-AdaTSK (Feature Selection and Rule Extraction) extends AdaTSK.

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
        """Initialise the FSRE-AdaTSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB; see also
                *use_en_frb*), or ``"custom"`` (explicit rules via *rules*).
            rules: Explicit rule antecedent indices; ignored when
                ``use_en_frb=True``.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
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

    def fit_fs(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Train the FS phase: only feature gates M(λ_d) are active (eq. 21)."""
        self.consequent_layer.mode = "fs"
        return self.fit(x, y, **kwargs)

    def fit_re(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Expand to En-FRB and train the RE phase: only rule gates M(θ_r) active (eq. 22)."""
        self.expand_to_en_frb()
        self.consequent_layer.mode = "re"
        return self.fit(x, y, **kwargs)

    def fit_finetune(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Fine-tune with no gates — plain TSK consequent (eq. 5)."""
        self.consequent_layer.mode = "finetune"
        return self.fit(x, y, **kwargs)
