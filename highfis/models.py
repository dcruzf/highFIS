"""Concrete TSK model variants.

Model family overview
---------------------

.. list-table::
   :header-rows: 1

   * - Variant
     - T-norm default
     - Defuzzifier
     - Reference
   * - **HTSK**
     - ``gmean``
     - ``SoftmaxLogDefuzzifier``
     - Cui, Wu & Xu (IJCNN 2021)
   * - **TSK (vanilla)**
     - ``prod``
     - ``SumBasedDefuzzifier``
     - Takagi & Sugeno (IEEE SMC 1985)
   * - **DombiTSK**
     - ``dombi``
     - ``SumBasedDefuzzifier``
     - Dombi (1982)
   * - **LogTSK**
     - ``prod``
     - ``LogSumDefuzzifier``
     - Cui, Wu & Xu (IEEE TFS 2021)

Scientific correspondence
~~~~~~~~~~~~~~~~~~~~~~~~~

In PyTSK (reference implementation) the antecedent computes log-space
firing strengths and applies one of two aggregation operators:

* ``torch.sum``  → product t-norm  (vanilla TSK)
* ``torch.mean`` → geometric mean  (HTSK)

Normalization is always ``softmax`` over rules.  In highFIS each step is
explicit and the defuzzification strategy is pluggable:

* **TSK**:  ``t_norm="prod"`` + ``SumBasedDefuzzifier`` = ``w_r / Σw``
* **HTSK**: ``t_norm="gmean"`` + ``SoftmaxLogDefuzzifier``
            = ``softmax(log(w^{1/D}))``
* **LogTSK**: ``t_norm="prod"`` + ``LogSumDefuzzifier(temperature)``
              = ``softmax(log(w) / τ)``  (Cui et al., IEEE TFS 2021 §III-B)
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from .base import BaseTSK
from .defuzzifiers import LogSumDefuzzifier, SoftmaxLogDefuzzifier, SumBasedDefuzzifier
from .layers import (
    AdaptiveDombiRuleLayer,
    AdaSoftminRuleLayer,
    ClassificationConsequentLayer,
    DGALETSKRuleLayer,
    DGTSKRuleLayer,
    GatedClassificationConsequentLayer,
    GatedClassificationZeroOrderConsequentLayer,
    GatedRegressionConsequentLayer,
    GatedRegressionZeroOrderConsequentLayer,
    RegressionConsequentLayer,
    _gate_activation,
)
from .memberships import MembershipFunction
from .t_norms import DombiTNorm, TNormFn


def _threshold_from_zeta(gate_values: Tensor, zeta: float) -> float:
    """Compute the threshold value from a gate vector and a coefficient."""
    if not 0.0 <= zeta <= 1.0:
        raise ValueError("zeta must be in [0, 1]")
    max_val = float(torch.max(gate_values).item())
    min_val = float(torch.min(gate_values).item())
    return max_val - zeta * (max_val - min_val)


def _build_first_order_design_matrix(
    norm_w: Tensor,
    x: Tensor,
    feature_gates: Tensor,
    rule_gates: Tensor,
) -> Tensor:
    """Build the design matrix for first-order least-squares consequent fitting."""
    batch_size, n_rules = norm_w.shape
    _, n_inputs = x.shape
    if feature_gates.shape != (n_rules, n_inputs):
        raise ValueError("feature_gates must have shape (n_rules, n_inputs)")
    if rule_gates.shape != (n_rules,):
        raise ValueError("rule_gates must have shape (n_rules,)")

    # Weighted bias contributions: (batch, n_rules)
    rule_gates = rule_gates.view(1, n_rules)
    bias_terms = norm_w * rule_gates

    # Weighted feature contributions: (batch, n_rules, n_inputs)
    feature_gates = feature_gates.view(1, n_rules, n_inputs)
    x_expanded = x.unsqueeze(1)  # (batch, 1, n_inputs)
    weighted_terms = norm_w.unsqueeze(-1) * rule_gates.unsqueeze(-1) * feature_gates * x_expanded

    # Concatenate bias and features for each rule
    return torch.cat([bias_terms.unsqueeze(-1), weighted_terms], dim=2).reshape(batch_size, n_rules * (n_inputs + 1))


def _solve_lse(A: Tensor, Y: Tensor) -> Tensor:
    """Solve a least-squares problem A X = Y for X."""
    # torch.pinverse handles both overdetermined and underdetermined systems.
    return torch.pinverse(A) @ Y


# =====================================================================
# Shared task-specific logic
# =====================================================================


class BaseTSKClassifier(BaseTSK):
    """Abstract classifier base that provides task-specific training and inference helpers."""

    def _compute_loss(self, criterion: Callable[[Tensor, Tensor], Tensor], output: Tensor, target: Tensor) -> Tensor:
        """Compute classification loss, handling MSELoss one-hot encoding."""
        if isinstance(criterion, nn.MSELoss):
            one_hot = torch.zeros_like(output)
            one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            return criterion(output, one_hot)
        return criterion(output, target)

    def _evaluate_validation(
        self, criterion: Callable[[Tensor, Tensor], Tensor], x_val: Tensor, y_val: Tensor
    ) -> dict[str, float]:
        """Evaluate validation set using accuracy as the early-stopping metric."""
        with torch.no_grad():
            logits = self.forward(x_val)
            val_loss = float(self._compute_loss(criterion, logits, y_val).item())
            val_acc = float((logits.argmax(dim=1) == y_val).float().mean().item())
        return {"val_loss": val_loss, "val_acc": val_acc, "metric": val_acc}

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return class probabilities computed with softmax."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def predict(self, x: Tensor) -> Tensor:
        """Return predicted class indices."""
        with torch.no_grad():
            return torch.argmax(self.predict_proba(x), dim=1)


class BaseTSKRegressor(BaseTSK):
    """Abstract regressor base that provides task-specific training and inference helpers."""

    def _compute_loss(self, criterion: Callable[[Tensor, Tensor], Tensor], output: Tensor, target: Tensor) -> Tensor:
        """Compute regression loss, squeezing the output to 1-D."""
        return criterion(output.squeeze(1), target)

    def predict(self, x: Tensor) -> Tensor:
        """Return predicted values as a 1-D tensor."""
        with torch.no_grad():
            return self.forward(x).squeeze(1)


# =====================================================================
# HTSK — High-dimensional TSK  (Cui, Wu & Xu, IJCNN 2021)
#
#   w_r = (∏_{d=1}^{D} μ_{r,d}(x_d))^{1/D}      (geometric mean)
#   f̄_r = softmax(log w)_r
# =====================================================================


class HTSKClassifier(BaseTSKClassifier):
    """TSK classifier with HTSK defuzzification for high-dimensional data.

    Uses the geometric-mean t-norm (``gmean``) and ``SoftmaxLogDefuzzifier``
    by default, following Cui, Wu & Xu (IJCNN 2021).
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
        """Initialize HTSK classifier architecture and consequent head."""
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
    """TSK regressor with HTSK defuzzification for high-dimensional data.

    Uses the geometric-mean t-norm (``gmean``) and ``SoftmaxLogDefuzzifier``
    by default, following Cui, Wu & Xu (IJCNN 2021).
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
        """Initialize HTSK regressor architecture and consequent head."""
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


# =====================================================================
# Vanilla TSK  (Takagi & Sugeno, IEEE SMC 1985)
#
#   w_r = ∏_{d=1}^{D} μ_{r,d}(x_d)               (product t-norm)
#   f̄_r = w_r / Σ_{i=1}^{R} w_i                   (sum-based)
# =====================================================================


class TSKClassifier(BaseTSKClassifier):
    r"""Vanilla TSK classifier with sum-based defuzzification.

    Implements the original Takagi-Sugeno-Kang inference:

    .. math::
        w_r = \prod_{d=1}^{D} \mu_{r,d}(x_d), \qquad
        \bar{f}_r = \frac{w_r}{\sum_{i=1}^{R} w_i}

    References:
    ----------
    Takagi, T. & Sugeno, M. (1985). "Fuzzy identification of systems and
    its applications to modeling and control." *IEEE Trans. Syst., Man,
    Cybern.* 15(1):116-132.
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
        """Initialize vanilla TSK classifier."""
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
    r"""Vanilla TSK regressor with sum-based defuzzification.

    Implements the original Takagi-Sugeno-Kang inference:

    .. math::
        w_r = \prod_{d=1}^{D} \mu_{r,d}(x_d), \qquad
        \bar{f}_r = \frac{w_r}{\sum_{i=1}^{R} w_i}

    References:
    ----------
    Takagi, T. & Sugeno, M. (1985). "Fuzzy identification of systems and
    its applications to modeling and control." *IEEE Trans. Syst., Man,
    Cybern.* 15(1):116-132.
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
        """Initialize vanilla TSK regressor."""
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


class DombiTSKClassifier(BaseTSKClassifier):
    """Dombi TSK classifier using Dombi aggregation in the antecedent."""

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "cartesian",
        t_norm: str = "dombi",
        lambda_: float = 1.0,
        t_norm_fn: TNormFn | None = None,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialize Dombi TSK classifier."""
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0")

        self.n_classes = int(n_classes)
        self.lambda_ = float(lambda_)
        if t_norm_fn is None:
            t_norm_fn = DombiTNorm(lambda_=self.lambda_)

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
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class DombiTSKRegressor(BaseTSKRegressor):
    """Dombi TSK regressor using Dombi aggregation in the antecedent."""

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "cartesian",
        t_norm: str = "dombi",
        lambda_: float = 1.0,
        t_norm_fn: TNormFn | None = None,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialize Dombi TSK regressor."""
        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0")

        self.lambda_ = float(lambda_)
        if t_norm_fn is None:
            t_norm_fn = DombiTNorm(lambda_=self.lambda_)

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
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()


class AdaTSKClassifier(BaseTSKClassifier):
    """AdaTSK classifier using adaptive per-rule Dombi aggregation."""

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "coco",
        lambda_init: float = 1.0,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
    ) -> None:
        """Initialize AdaTSK classifier architecture and consequent head."""
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")

        self.n_classes = int(n_classes)
        self.lambda_init = float(lambda_init)
        self.eps = eps

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            t_norm_fn=None,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = AdaptiveDombiRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules,
            rule_base=rule_base,
            lambda_init=self.lambda_init,
            eps=self.eps,
        )

    def _build_consequent_layer(self) -> nn.Module:
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class AdaTSKRegressor(BaseTSKRegressor):
    """AdaTSK regressor using adaptive per-rule Dombi aggregation."""

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "coco",
        lambda_init: float = 1.0,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
    ) -> None:
        """Initialize AdaTSK regressor architecture and consequent head."""
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")

        self.lambda_init = float(lambda_init)
        self.eps = eps

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            t_norm_fn=None,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = AdaptiveDombiRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules,
            rule_base=rule_base,
            lambda_init=self.lambda_init,
            eps=self.eps,
        )

    def _build_consequent_layer(self) -> nn.Module:
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()


class FSREAdaTSKClassifier(BaseTSKClassifier):
    """FSRE-AdaTSK classifier with gate-based consequents and adaptive softmin antecedent."""

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "coco",
        lambda_init: float = 1.0,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
        use_en_frb: bool = False,
    ) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")

        self.n_classes = int(n_classes)
        self.lambda_init = float(lambda_init)
        self.eps = eps
        self.use_en_frb = bool(use_en_frb)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            t_norm_fn=None,
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

    def _build_consequent_layer(self) -> nn.Module:
        return GatedClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

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
        """Train the FS phase on the current rule base."""
        return self.fit(x, y, **kwargs)

    def fit_re(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Expand to En-FRB and train the RE phase."""
        self.expand_to_en_frb()
        return self.fit(x, y, **kwargs)

    def fit_finetune(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Fine-tune the reduced FSRE-AdaTSK model."""
        return self.fit(x, y, **kwargs)


class FSREAdaTSKRegressor(BaseTSKRegressor):
    """FSRE-AdaTSK regressor with gate-based consequents and adaptive softmin antecedent."""

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "coco",
        lambda_init: float = 1.0,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
        use_en_frb: bool = False,
    ) -> None:
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")

        self.lambda_init = float(lambda_init)
        self.eps = eps
        self.use_en_frb = bool(use_en_frb)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            t_norm_fn=None,
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

    def _build_consequent_layer(self) -> nn.Module:
        return GatedRegressionConsequentLayer(self.n_rules, self.n_inputs)

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
        """Train the FS phase on the current rule base."""
        return self.fit(x, y, **kwargs)

    def fit_re(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Expand to En-FRB and train the RE phase."""
        self.expand_to_en_frb()
        return self.fit(x, y, **kwargs)

    def fit_finetune(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Fine-tune the reduced FSRE-AdaTSK model."""
        return self.fit(x, y, **kwargs)


class DGALETSKClassifier(BaseTSKClassifier):
    """DG-ALETSK classifier with Ln-Exp softmin antecedent and double groups of gates."""

    rule_layer: DGALETSKRuleLayer
    consequent_layer: GatedClassificationConsequentLayer | GatedClassificationZeroOrderConsequentLayer

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "coco",
        lambda_init: float = 1.0,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
        use_en_frb: bool = False,
    ) -> None:
        """Initialize the DG-ALETSK classifier with zero-order gated consequents."""
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")

        self.n_classes = int(n_classes)
        self.lambda_init = float(lambda_init)
        self.eps = eps
        self.use_en_frb = bool(use_en_frb)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            t_norm_fn=None,
            rules=rules,
            defuzzifier=defuzzifier or SoftmaxLogDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = DGALETSKRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules if not self.use_en_frb else None,
            rule_base="en" if self.use_en_frb else rule_base,
            alpha_init=self.lambda_init,
            eps=self.eps,
        )
        self.n_rules = self.rule_layer.n_rules
        self.consequent_layer = self._build_zero_order_consequent_layer()

    def _build_zero_order_consequent_layer(self) -> GatedClassificationZeroOrderConsequentLayer:
        return GatedClassificationZeroOrderConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _build_consequent_layer(self) -> GatedClassificationConsequentLayer:
        return GatedClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def convert_to_first_order(self) -> None:
        """Convert the DG phase zero-order consequent to first-order form."""
        previous = self.consequent_layer
        new_consequent = self._build_consequent_layer()
        if isinstance(previous, GatedClassificationZeroOrderConsequentLayer):
            new_consequent.theta_gates.data.copy_(previous.theta_gates.data)
        self.consequent_layer = new_consequent

    def get_feature_gate_values(self) -> Tensor:
        """Return normalized antecedent feature gate values for the DG phase."""
        rule_layer = self.rule_layer
        return _gate_activation(rule_layer.lambda_gates)

    def get_rule_gate_values(self) -> Tensor:
        """Return normalized consequent rule gate values for the DG phase."""
        consequent = self.consequent_layer
        return _gate_activation(consequent.theta_gates)

    def compute_thresholds(self, zeta_lambda: float, zeta_theta: float) -> tuple[float, float]:
        """Compute feature and rule thresholds from gate values and coefficient pairs."""
        tau_lambda = _threshold_from_zeta(self.get_feature_gate_values(), zeta_lambda)
        tau_theta = _threshold_from_zeta(self.get_rule_gate_values(), zeta_theta)
        return tau_lambda, tau_theta

    def apply_thresholds(self, tau_lambda: float, tau_theta: float) -> None:
        """Apply threshold pruning to feature and rule gates."""
        if not torch.isfinite(torch.tensor(tau_lambda)) or not torch.isfinite(torch.tensor(tau_theta)):
            raise ValueError("thresholds must be finite")

        feature_gate_values = self.get_feature_gate_values()
        pruned_features = feature_gate_values <= tau_lambda
        rule_layer = self.rule_layer
        cast(Tensor, rule_layer.lambda_gates.data)[pruned_features] = 0.0

        rule_gate_values = self.get_rule_gate_values()
        pruned_rules = rule_gate_values <= tau_theta
        consequent = self.consequent_layer
        cast(Tensor, consequent.theta_gates.data)[pruned_rules] = 0.0

    def _fit_first_order_consequents_lse(self, x: Tensor, y: Tensor) -> None:
        """Refit first-order consequent parameters with antecedents fixed."""
        if isinstance(self.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
            raise ValueError("convert_to_first_order() must be called before LSE consequent fitting")

        self.eval()
        with torch.no_grad():
            norm_w = self.forward_antecedents(x)
            consequent = cast(GatedClassificationConsequentLayer, self.consequent_layer)
            feature_gates = _gate_activation(consequent.lambda_gates)
            rule_gates = _gate_activation(consequent.theta_gates)
            design = _build_first_order_design_matrix(norm_w, x, feature_gates, rule_gates)

            target = torch.zeros((x.shape[0], self.n_classes), dtype=x.dtype, device=x.device)
            target.scatter_(1, y.unsqueeze(1), 1.0)
            solution = _solve_lse(design, target)

            n_rules = self.n_rules
            n_inputs = self.n_inputs
            effective = solution.reshape(n_rules, n_inputs + 1, self.n_classes)
            effective_bias = effective[:, 0, :]
            effective_weight = effective[:, 1:, :].permute(0, 2, 1)

            rule_gates_unsqueezed = rule_gates.view(n_rules, 1)
            bias = torch.where(
                rule_gates_unsqueezed.abs() > 0,
                effective_bias / rule_gates_unsqueezed,
                torch.zeros_like(effective_bias),
            )

            denom = (rule_gates_unsqueezed.unsqueeze(-1) * feature_gates.unsqueeze(1)).expand_as(effective_weight)
            weight = torch.zeros_like(effective_weight)
            nonzero = denom.abs() > 0
            weight[nonzero] = effective_weight[nonzero] / denom[nonzero]

            consequent = cast(GatedClassificationConsequentLayer, self.consequent_layer)
            cast(Tensor, consequent.bias.data).copy_(bias)
            cast(Tensor, consequent.weight.data).copy_(weight)

    def _evaluate_threshold_score(self, x: Tensor, y: Tensor) -> float:
        """Evaluate model quality for threshold search."""
        with torch.no_grad():
            logits = self.forward(x)
            predicted = torch.argmax(logits, dim=1)
            return float((predicted == y).float().mean().item())

    def search_thresholds(
        self,
        x: Tensor,
        y: Tensor,
        zeta_lambda: Sequence[float] | None = None,
        zeta_theta: Sequence[float] | None = None,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
        use_lse: bool = True,
        inplace: bool = True,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Search threshold coefficients for feature and rule pruning.

        The search follows the DG-ALETSK paper strategy: thresholds are
        computed from gate values, applied to prune gates, and the first-order
        consequent parameters are refit with antecedents fixed.
        """
        if zeta_lambda is None:
            zeta_lambda = [0.0, 0.25, 0.5, 0.75, 1.0]
        if zeta_theta is None:
            zeta_theta = [0.0, 0.25, 0.5, 0.75, 1.0]

        x_eval = x_val if x_val is not None else x
        y_eval = y_val if y_val is not None else y

        best_score = float("-inf")
        best_state: dict[str, Any] | None = None
        best_tau_lambda = 0.0
        best_tau_theta = 0.0
        best_zeta_lambda = 0.0
        best_zeta_theta = 0.0

        for zeta_l in zeta_lambda:
            for zeta_t in zeta_theta:
                candidate = copy.deepcopy(self)
                if isinstance(candidate.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
                    candidate.convert_to_first_order()

                tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                candidate.apply_thresholds(tau_l, tau_t)
                if use_lse:
                    candidate._fit_first_order_consequents_lse(x, y)

                score = candidate._evaluate_threshold_score(x_eval, y_eval)
                if verbose:
                    self._log("zeta_lambda=%s zeta_theta=%s score=%.6f", zeta_l, zeta_t, score, verbose=True)

                if score > best_score:
                    best_score = score
                    best_state = copy.deepcopy(candidate.state_dict())
                    best_tau_lambda = tau_l
                    best_tau_theta = tau_t
                    best_zeta_lambda = zeta_l
                    best_zeta_theta = zeta_t

        if best_state is None:
            raise RuntimeError("threshold search did not yield a valid candidate")

        result = {
            "best_score": best_score,
            "best_zeta_lambda": best_zeta_lambda,
            "best_zeta_theta": best_zeta_theta,
            "tau_lambda": best_tau_lambda,
            "tau_theta": best_tau_theta,
        }

        if inplace:
            if isinstance(self.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
                self.convert_to_first_order()
            self.load_state_dict(best_state)

        return result

    def fit_dg_phase(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Train the DG phase using zero-order TSK and joint FS+RE."""
        return self.fit(x, y, **kwargs)

    def fit_finetune(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Fine-tune the DG-ALETSK model after converting to first-order TSK."""
        return self.fit(x, y, **kwargs)


class DGALETSKRegressor(BaseTSKRegressor):
    """DG-ALETSK regressor with Ln-Exp softmin antecedent and double groups of gates."""

    rule_layer: DGALETSKRuleLayer
    consequent_layer: GatedRegressionConsequentLayer | GatedRegressionZeroOrderConsequentLayer

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "coco",
        lambda_init: float = 1.0,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
        use_en_frb: bool = False,
    ) -> None:
        """Initialize the DG-ALETSK regressor with zero-order gated consequents."""
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")

        self.lambda_init = float(lambda_init)
        self.eps = eps
        self.use_en_frb = bool(use_en_frb)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            t_norm_fn=None,
            rules=rules,
            defuzzifier=defuzzifier or SoftmaxLogDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = DGALETSKRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules if not self.use_en_frb else None,
            rule_base="en" if self.use_en_frb else rule_base,
            alpha_init=self.lambda_init,
            eps=self.eps,
        )
        self.n_rules = self.rule_layer.n_rules
        self.consequent_layer = self._build_zero_order_consequent_layer()

    def _build_zero_order_consequent_layer(self) -> GatedRegressionZeroOrderConsequentLayer:
        return GatedRegressionZeroOrderConsequentLayer(self.n_rules, self.n_inputs)

    def _build_consequent_layer(self) -> GatedRegressionConsequentLayer:
        return GatedRegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def convert_to_first_order(self) -> None:
        """Convert the DG phase zero-order consequent to first-order form."""
        previous = self.consequent_layer
        new_consequent = self._build_consequent_layer()
        if isinstance(previous, GatedRegressionZeroOrderConsequentLayer):
            new_consequent.theta_gates.data.copy_(previous.theta_gates.data)
        self.consequent_layer = new_consequent

    def get_feature_gate_values(self) -> Tensor:
        """Return normalized antecedent feature gate values for the DG phase."""
        rule_layer = self.rule_layer
        return _gate_activation(rule_layer.lambda_gates)

    def get_rule_gate_values(self) -> Tensor:
        """Return normalized consequent rule gate values for the DG phase."""
        consequent = self.consequent_layer
        return _gate_activation(consequent.theta_gates)

    def compute_thresholds(self, zeta_lambda: float, zeta_theta: float) -> tuple[float, float]:
        """Compute feature and rule thresholds from gate values and coefficient pairs."""
        tau_lambda = _threshold_from_zeta(self.get_feature_gate_values(), zeta_lambda)
        tau_theta = _threshold_from_zeta(self.get_rule_gate_values(), zeta_theta)
        return tau_lambda, tau_theta

    def apply_thresholds(self, tau_lambda: float, tau_theta: float) -> None:
        """Apply threshold pruning to feature and rule gates."""
        if not torch.isfinite(torch.tensor(tau_lambda)) or not torch.isfinite(torch.tensor(tau_theta)):
            raise ValueError("thresholds must be finite")

        feature_gate_values = self.get_feature_gate_values()
        pruned_features = feature_gate_values <= tau_lambda
        rule_layer = self.rule_layer
        cast(Tensor, rule_layer.lambda_gates.data)[pruned_features] = 0.0

        rule_gate_values = self.get_rule_gate_values()
        pruned_rules = rule_gate_values <= tau_theta
        consequent = self.consequent_layer
        cast(Tensor, consequent.theta_gates.data)[pruned_rules] = 0.0

    def _fit_first_order_consequents_lse(self, x: Tensor, y: Tensor) -> None:
        """Refit first-order consequent parameters with antecedents fixed."""
        if isinstance(self.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
            raise ValueError("convert_to_first_order() must be called before LSE consequent fitting")

        self.eval()
        with torch.no_grad():
            norm_w = self.forward_antecedents(x)
            consequent = cast(GatedRegressionConsequentLayer, self.consequent_layer)
            feature_gates = _gate_activation(consequent.lambda_gates)
            rule_gates = _gate_activation(consequent.theta_gates)
            design = _build_first_order_design_matrix(norm_w, x, feature_gates, rule_gates)

            target = y.unsqueeze(1)
            solution = _solve_lse(design, target)

            n_rules = self.n_rules
            n_inputs = self.n_inputs
            effective = solution.reshape(n_rules, n_inputs + 1)
            effective_bias = effective[:, 0]
            effective_weight = effective[:, 1:]

            rule_gates_unsqueezed = rule_gates.view(n_rules, 1)
            active_rules = rule_gates_unsqueezed.abs().squeeze(1) > 0
            bias = torch.where(
                active_rules, effective_bias / rule_gates_unsqueezed.squeeze(1), torch.zeros_like(effective_bias)
            )

            denom = (rule_gates_unsqueezed.unsqueeze(-1) * feature_gates.unsqueeze(-1)).squeeze(-1)
            weight = torch.zeros_like(effective_weight)
            nonzero = denom.abs() > 0
            weight[nonzero] = effective_weight[nonzero] / denom[nonzero]

            consequent = cast(GatedRegressionConsequentLayer, self.consequent_layer)
            cast(Tensor, consequent.bias.data).copy_(bias)
            cast(Tensor, consequent.weight.data).copy_(weight)

    def _evaluate_threshold_score(self, x: Tensor, y: Tensor) -> float:
        """Evaluate model quality for threshold search in regression."""
        with torch.no_grad():
            output = self.forward(x).squeeze(1)
            return -float(((output - y) ** 2).mean().item())

    def search_thresholds(
        self,
        x: Tensor,
        y: Tensor,
        zeta_lambda: Sequence[float] | None = None,
        zeta_theta: Sequence[float] | None = None,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
        use_lse: bool = True,
        inplace: bool = True,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Search threshold coefficients for feature and rule pruning."""
        if zeta_lambda is None:
            zeta_lambda = [0.0, 0.25, 0.5, 0.75, 1.0]
        if zeta_theta is None:
            zeta_theta = [0.0, 0.25, 0.5, 0.75, 1.0]

        x_eval = x_val if x_val is not None else x
        y_eval = y_val if y_val is not None else y

        best_score = float("-inf")
        best_state: dict[str, Any] | None = None
        best_tau_lambda = 0.0
        best_tau_theta = 0.0
        best_zeta_lambda = 0.0
        best_zeta_theta = 0.0

        for zeta_l in zeta_lambda:
            for zeta_t in zeta_theta:
                candidate = copy.deepcopy(self)
                if isinstance(candidate.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
                    candidate.convert_to_first_order()

                tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                candidate.apply_thresholds(tau_l, tau_t)
                if use_lse:
                    candidate._fit_first_order_consequents_lse(x, y)

                score = candidate._evaluate_threshold_score(x_eval, y_eval)
                if verbose:
                    self._log("zeta_lambda=%s zeta_theta=%s score=%.6f", zeta_l, zeta_t, score, verbose=True)

                if score > best_score:
                    best_score = score
                    best_state = copy.deepcopy(candidate.state_dict())
                    best_tau_lambda = tau_l
                    best_tau_theta = tau_t
                    best_zeta_lambda = zeta_l
                    best_zeta_theta = zeta_t

        if best_state is None:
            raise RuntimeError("threshold search did not yield a valid candidate")

        result = {
            "best_score": best_score,
            "best_zeta_lambda": best_zeta_lambda,
            "best_zeta_theta": best_zeta_theta,
            "tau_lambda": best_tau_lambda,
            "tau_theta": best_tau_theta,
        }

        if inplace:
            if isinstance(self.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
                self.convert_to_first_order()
            self.load_state_dict(best_state)

        return result

    def fit_dg_phase(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Train the DG phase using zero-order TSK and joint FS+RE."""
        return self.fit(x, y, **kwargs)

    def fit_finetune(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Fine-tune the DG-ALETSK model after converting to first-order TSK."""
        return self.fit(x, y, **kwargs)


class DGTSKClassifier(BaseTSKClassifier):
    """DG-TSK classifier with P-FRB style support and configurable gates."""

    rule_layer: DGTSKRuleLayer
    consequent_layer: GatedClassificationConsequentLayer | GatedClassificationZeroOrderConsequentLayer

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "coco",
        gate_fea: str | Callable[[Tensor], Tensor] | None = "gate_m",
        gate_rule: str | Callable[[Tensor], Tensor] | None = "gate_m",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
        use_en_frb: bool = False,
    ) -> None:
        """Initialize a DGTSK classifier using gate-based feature and rule selection."""
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")

        self.n_classes = int(n_classes)
        self.gate_fea = gate_fea
        self.gate_rule = gate_rule
        self.eps = eps
        self.use_en_frb = bool(use_en_frb)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            t_norm_fn=None,
            rules=rules,
            defuzzifier=defuzzifier or SoftmaxLogDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = DGTSKRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules if not self.use_en_frb else None,
            rule_base="en" if self.use_en_frb else rule_base,
            gate_fea=self.gate_fea,
            eps=self.eps,
        )
        self.n_rules = self.rule_layer.n_rules
        self.consequent_layer = self._build_zero_order_consequent_layer()

    def _build_zero_order_consequent_layer(self) -> GatedClassificationZeroOrderConsequentLayer:
        return GatedClassificationZeroOrderConsequentLayer(
            self.n_rules,
            self.n_inputs,
            self.n_classes,
            gate_fn=self.gate_rule,
        )

    def _build_consequent_layer(self) -> GatedClassificationConsequentLayer:
        return GatedClassificationConsequentLayer(
            self.n_rules,
            self.n_inputs,
            self.n_classes,
            gate_fn=self.gate_rule,
        )

    def _default_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def convert_to_first_order(self) -> None:
        """Convert the DG-TSK model from zero-order to first-order consequent."""
        previous = self.consequent_layer
        new_consequent = self._build_consequent_layer()
        if isinstance(previous, GatedClassificationZeroOrderConsequentLayer):
            new_consequent.theta_gates.data.copy_(previous.theta_gates.data)
        self.consequent_layer = new_consequent

    def get_feature_gate_values(self) -> Tensor:
        """Return normalized DG-TSK feature gate activations from lambda values."""
        return self.rule_layer.gate_fn(self.rule_layer.lambda_gates)

    def get_rule_gate_values(self) -> Tensor:
        """Return normalized DG-TSK rule gate activations from theta values."""
        return self.consequent_layer.gate_fn(self.consequent_layer.theta_gates)

    def compute_thresholds(self, zeta_lambda: float, zeta_theta: float) -> tuple[float, float]:
        """Compute DG-TSK pruning thresholds from gate values and zeta parameters."""
        tau_lambda = _threshold_from_zeta(self.get_feature_gate_values(), zeta_lambda)
        tau_theta = _threshold_from_zeta(self.get_rule_gate_values(), zeta_theta)
        return tau_lambda, tau_theta

    def apply_thresholds(self, tau_lambda: float, tau_theta: float) -> None:
        """Prune DG-TSK feature and rule gates using the computed thresholds."""
        if not torch.isfinite(torch.tensor(tau_lambda)) or not torch.isfinite(torch.tensor(tau_theta)):
            raise ValueError("thresholds must be finite")

        feature_gate_values = self.get_feature_gate_values()
        pruned_features = feature_gate_values <= tau_lambda
        cast(Tensor, self.rule_layer.lambda_gates.data)[pruned_features] = 0.0

        rule_gate_values = self.get_rule_gate_values()
        pruned_rules = rule_gate_values <= tau_theta
        cast(Tensor, self.consequent_layer.theta_gates.data)[pruned_rules] = 0.0

    def _fit_first_order_consequents_lse(self, x: Tensor, y: Tensor) -> None:
        if isinstance(self.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
            raise ValueError("convert_to_first_order() must be called before LSE consequent fitting")

        self.eval()
        with torch.no_grad():
            norm_w = self.forward_antecedents(x)
            consequent = cast(GatedClassificationConsequentLayer, self.consequent_layer)
            feature_gates = consequent.gate_fn(consequent.lambda_gates)
            rule_gates = consequent.gate_fn(consequent.theta_gates)
            design = _build_first_order_design_matrix(norm_w, x, feature_gates, rule_gates)

            target = torch.zeros((x.shape[0], self.n_classes), dtype=x.dtype, device=x.device)
            target.scatter_(1, y.unsqueeze(1), 1.0)
            solution = _solve_lse(design, target)

            n_rules = self.n_rules
            n_inputs = self.n_inputs
            effective = solution.reshape(n_rules, n_inputs + 1, self.n_classes)
            effective_bias = effective[:, 0, :]
            effective_weight = effective[:, 1:, :].permute(0, 2, 1)

            rule_gates_unsqueezed = rule_gates.view(n_rules, 1)
            bias = torch.where(
                rule_gates_unsqueezed.abs() > 0,
                effective_bias / rule_gates_unsqueezed,
                torch.zeros_like(effective_bias),
            )

            denom = (rule_gates_unsqueezed.unsqueeze(-1) * feature_gates.unsqueeze(1)).expand_as(effective_weight)
            weight = torch.zeros_like(effective_weight)
            nonzero = denom.abs() > 0
            weight[nonzero] = effective_weight[nonzero] / denom[nonzero]

            cast(Tensor, consequent.bias.data).copy_(bias)
            cast(Tensor, consequent.weight.data).copy_(weight)

    def _evaluate_threshold_score(self, x: Tensor, y: Tensor) -> float:
        with torch.no_grad():
            logits = self.forward(x)
            predicted = torch.argmax(logits, dim=1)
            return float((predicted == y).float().mean().item())

    def search_thresholds(
        self,
        x: Tensor,
        y: Tensor,
        zeta_lambda: Sequence[float] | None = None,
        zeta_theta: Sequence[float] | None = None,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
        use_lse: bool = True,
        inplace: bool = True,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Search DG-TSK threshold combinations and optionally apply the best candidate."""
        if zeta_lambda is None:
            zeta_lambda = [0.0, 0.25, 0.5, 0.75, 1.0]
        if zeta_theta is None:
            zeta_theta = [0.0, 0.25, 0.5, 0.75, 1.0]

        x_eval = x_val if x_val is not None else x
        y_eval = y_val if y_val is not None else y

        best_score = float("-inf")
        best_state: dict[str, Any] | None = None
        best_tau_lambda = 0.0
        best_tau_theta = 0.0
        best_zeta_lambda = 0.0
        best_zeta_theta = 0.0

        for zeta_l in zeta_lambda:
            for zeta_t in zeta_theta:
                candidate = copy.deepcopy(self)
                if isinstance(candidate.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
                    candidate.convert_to_first_order()

                tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                candidate.apply_thresholds(tau_l, tau_t)
                if use_lse:
                    candidate._fit_first_order_consequents_lse(x, y)

                score = candidate._evaluate_threshold_score(x_eval, y_eval)
                if verbose:
                    self._log("zeta_lambda=%s zeta_theta=%s score=%.6f", zeta_l, zeta_t, score, verbose=True)

                if score > best_score:
                    best_score = score
                    best_state = copy.deepcopy(candidate.state_dict())
                    best_tau_lambda = tau_l
                    best_tau_theta = tau_t
                    best_zeta_lambda = zeta_l
                    best_zeta_theta = zeta_t

        if best_state is None:
            raise RuntimeError("threshold search did not yield a valid candidate")

        result = {
            "best_score": best_score,
            "best_zeta_lambda": best_zeta_lambda,
            "best_zeta_theta": best_zeta_theta,
            "tau_lambda": best_tau_lambda,
            "tau_theta": best_tau_theta,
        }

        if inplace:
            if isinstance(self.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
                self.convert_to_first_order()
            self.load_state_dict(best_state)

        return result

    def fit_dg_phase(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Train the DG-TSK zero-order phase before first-order conversion."""
        return self.fit(x, y, **kwargs)

    def fit_finetune(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Fine-tune the DG-TSK classifier after conversion to first-order consequents."""
        return self.fit(x, y, **kwargs)


class DGTSKRegressor(BaseTSKRegressor):
    """DG-TSK regressor with P-FRB style support and configurable gates."""

    rule_layer: DGTSKRuleLayer
    consequent_layer: GatedRegressionConsequentLayer | GatedRegressionZeroOrderConsequentLayer

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "coco",
        gate_fea: str | Callable[[Tensor], Tensor] | None = "gate_m",
        gate_rule: str | Callable[[Tensor], Tensor] | None = "gate_m",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
        use_en_frb: bool = False,
    ) -> None:
        """Initialize a DGTSK regressor using gate-based antecedents and consequents."""
        self.gate_fea = gate_fea
        self.gate_rule = gate_rule
        self.eps = eps
        self.use_en_frb = bool(use_en_frb)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
            t_norm_fn=None,
            rules=rules,
            defuzzifier=defuzzifier or SoftmaxLogDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

        self.rule_layer = DGTSKRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules if not self.use_en_frb else None,
            rule_base="en" if self.use_en_frb else rule_base,
            gate_fea=self.gate_fea,
            eps=self.eps,
        )
        self.n_rules = self.rule_layer.n_rules
        self.consequent_layer = self._build_zero_order_consequent_layer()

    def _build_zero_order_consequent_layer(self) -> GatedRegressionZeroOrderConsequentLayer:
        return GatedRegressionZeroOrderConsequentLayer(self.n_rules, self.n_inputs, gate_fn=self.gate_rule)

    def _build_consequent_layer(self) -> GatedRegressionConsequentLayer:
        return GatedRegressionConsequentLayer(self.n_rules, self.n_inputs, gate_fn=self.gate_rule)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def convert_to_first_order(self) -> None:
        """Convert the DG-TSK regressor from zero-order to first-order consequent."""
        previous = self.consequent_layer
        new_consequent = self._build_consequent_layer()
        if isinstance(previous, GatedRegressionZeroOrderConsequentLayer):
            new_consequent.theta_gates.data.copy_(previous.theta_gates.data)
        self.consequent_layer = new_consequent

    def get_feature_gate_values(self) -> Tensor:
        """Return normalized DG-TSK feature gate activations from lambda values."""
        return self.rule_layer.gate_fn(self.rule_layer.lambda_gates)

    def get_rule_gate_values(self) -> Tensor:
        """Return normalized DG-TSK rule gate activations from theta values."""
        return self.consequent_layer.gate_fn(self.consequent_layer.theta_gates)

    def compute_thresholds(self, zeta_lambda: float, zeta_theta: float) -> tuple[float, float]:
        """Compute DG-TSK pruning thresholds from gate values and zeta parameters."""
        tau_lambda = _threshold_from_zeta(self.get_feature_gate_values(), zeta_lambda)
        tau_theta = _threshold_from_zeta(self.get_rule_gate_values(), zeta_theta)
        return tau_lambda, tau_theta

    def apply_thresholds(self, tau_lambda: float, tau_theta: float) -> None:
        """Prune DG-TSK feature and rule gates using the computed thresholds."""
        if not torch.isfinite(torch.tensor(tau_lambda)) or not torch.isfinite(torch.tensor(tau_theta)):
            raise ValueError("thresholds must be finite")

        feature_gate_values = self.get_feature_gate_values()
        pruned_features = feature_gate_values <= tau_lambda
        cast(Tensor, self.rule_layer.lambda_gates.data)[pruned_features] = 0.0

        rule_gate_values = self.get_rule_gate_values()
        pruned_rules = rule_gate_values <= tau_theta
        cast(Tensor, self.consequent_layer.theta_gates.data)[pruned_rules] = 0.0

    def _fit_first_order_consequents_lse(self, x: Tensor, y: Tensor) -> None:
        if isinstance(self.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
            raise ValueError("convert_to_first_order() must be called before LSE consequent fitting")

        self.eval()
        with torch.no_grad():
            norm_w = self.forward_antecedents(x)
            consequent = cast(GatedRegressionConsequentLayer, self.consequent_layer)
            feature_gates = consequent.gate_fn(consequent.lambda_gates)
            rule_gates = consequent.gate_fn(consequent.theta_gates)
            design = _build_first_order_design_matrix(norm_w, x, feature_gates, rule_gates)
            target = y.unsqueeze(1)
            solution = _solve_lse(design, target)

            n_rules = self.n_rules
            n_inputs = self.n_inputs
            effective = solution.reshape(n_rules, n_inputs + 1)
            effective_bias = effective[:, 0]
            effective_weight = effective[:, 1:]

            rule_gates_unsqueezed = rule_gates.view(n_rules, 1)
            active_rules = rule_gates_unsqueezed.abs().squeeze(1) > 0
            bias = torch.where(
                active_rules,
                effective_bias / rule_gates_unsqueezed.squeeze(1),
                torch.zeros_like(effective_bias),
            )

            denom = (rule_gates_unsqueezed.unsqueeze(-1) * feature_gates.unsqueeze(-1)).squeeze(-1)
            weight = torch.zeros_like(effective_weight)
            nonzero = denom.abs() > 0
            weight[nonzero] = effective_weight[nonzero] / denom[nonzero]

            cast(Tensor, consequent.bias.data).copy_(bias)
            cast(Tensor, consequent.weight.data).copy_(weight)

    def _evaluate_threshold_score(self, x: Tensor, y: Tensor) -> float:
        with torch.no_grad():
            output = self.forward(x).squeeze(1)
            return -float(((output - y) ** 2).mean().item())

    def search_thresholds(
        self,
        x: Tensor,
        y: Tensor,
        zeta_lambda: Sequence[float] | None = None,
        zeta_theta: Sequence[float] | None = None,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
        use_lse: bool = True,
        inplace: bool = True,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Search DG-TSK regression threshold combinations and optionally apply the best candidate."""
        if zeta_lambda is None:
            zeta_lambda = [0.0, 0.25, 0.5, 0.75, 1.0]
        if zeta_theta is None:
            zeta_theta = [0.0, 0.25, 0.5, 0.75, 1.0]

        x_eval = x_val if x_val is not None else x
        y_eval = y_val if y_val is not None else y

        best_score = float("-inf")
        best_state: dict[str, Any] | None = None
        best_tau_lambda = 0.0
        best_tau_theta = 0.0
        best_zeta_lambda = 0.0
        best_zeta_theta = 0.0

        for zeta_l in zeta_lambda:
            for zeta_t in zeta_theta:
                candidate = copy.deepcopy(self)
                if isinstance(candidate.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
                    candidate.convert_to_first_order()

                tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                candidate.apply_thresholds(tau_l, tau_t)
                if use_lse:
                    candidate._fit_first_order_consequents_lse(x, y)

                score = candidate._evaluate_threshold_score(x_eval, y_eval)
                if verbose:
                    self._log("zeta_lambda=%s zeta_theta=%s score=%.6f", zeta_l, zeta_t, score, verbose=True)

                if score > best_score:
                    best_score = score
                    best_state = copy.deepcopy(candidate.state_dict())
                    best_tau_lambda = tau_l
                    best_tau_theta = tau_t
                    best_zeta_lambda = zeta_l
                    best_zeta_theta = zeta_t

        if best_state is None:
            raise RuntimeError("threshold search did not yield a valid candidate")

        result = {
            "best_score": best_score,
            "best_zeta_lambda": best_zeta_lambda,
            "best_zeta_theta": best_zeta_theta,
            "tau_lambda": best_tau_lambda,
            "tau_theta": best_tau_theta,
        }

        if inplace:
            if isinstance(self.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
                self.convert_to_first_order()
            self.load_state_dict(best_state)

        return result

    def fit_dg_phase(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Train the DG-TSK regression zero-order phase before first-order conversion."""
        return self.fit(x, y, **kwargs)

    def fit_finetune(self, x: Tensor, y: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Fine-tune the DG-TSK regression model after converting to first order."""
        return self.fit(x, y, **kwargs)


# =====================================================================
# LogTSK  (Cui, Wu & Xu, IEEE Trans. Fuzzy Syst. 2021)
#
#   w_r = ∏_{d=1}^{D} μ_{r,d}(x_d)               (product t-norm)
#   f̄_r = softmax(log(w) / τ)                     (log-space, tempered)
#
# The temperature τ controls the sharpness of the distribution.
# τ = 1 recovers softmax(log(w)) ≡ w / Σw; τ < 1 sharpens.
# =====================================================================


class LogTSKClassifier(BaseTSKClassifier):
    r"""LogTSK classifier with log-space defuzzification.

    Normalization operates entirely in log-space to avoid underflow in
    high-dimensional settings:

    .. math::
        \log \bar{f}_r = \frac{\log w_r}{\tau}
        - \log\!\left(\sum_{i=1}^{R}
          \exp\!\left(\frac{\log w_i}{\tau}\right)\right)

    where :math:`\tau` is a temperature parameter (default 1).

    References:
    ----------
    Cui, Y., Wu, D. & Xu, Y. (2021). "Optimize TSK Fuzzy Systems for
    Regression Problems: Mini-Batch Gradient Descent With Regularization,
    DropRule, and AdaBound (MBGD-RDA)." *IEEE Trans. Fuzzy Syst.*
    29(5):1003-1015.
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
        temperature: float = 1.0,
    ) -> None:
        """Initialize LogTSK classifier."""
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = int(n_classes)
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            t_norm_fn=t_norm_fn,
            rules=rules,
            defuzzifier=defuzzifier or LogSumDefuzzifier(temperature=temperature),
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        """Build classification consequent head."""
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        """Return CrossEntropyLoss as the default classification loss."""
        return nn.CrossEntropyLoss()


class LogTSKRegressor(BaseTSKRegressor):
    r"""LogTSK regressor with log-space defuzzification.

    Normalization operates entirely in log-space to avoid underflow in
    high-dimensional settings:

    .. math::
        \log \bar{f}_r = \frac{\log w_r}{\tau}
        - \log\!\left(\sum_{i=1}^{R}
          \exp\!\left(\frac{\log w_i}{\tau}\right)\right)

    where :math:`\tau` is a temperature parameter (default 1).

    References:
    ----------
    Cui, Y., Wu, D. & Xu, Y. (2021). "Optimize TSK Fuzzy Systems for
    Regression Problems: Mini-Batch Gradient Descent With Regularization,
    DropRule, and AdaBound (MBGD-RDA)." *IEEE Trans. Fuzzy Syst.*
    29(5):1003-1015.
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
        temperature: float = 1.0,
    ) -> None:
        """Initialize LogTSK regressor."""
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            t_norm_fn=t_norm_fn,
            rules=rules,
            defuzzifier=defuzzifier or LogSumDefuzzifier(temperature=temperature),
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        """Build regression consequent head."""
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        """Return MSELoss as the default regression loss."""
        return nn.MSELoss()


__all__: list[str] = [
    "AdaTSKClassifier",
    "AdaTSKRegressor",
    "DGALETSKClassifier",
    "DGALETSKRegressor",
    "DGTSKClassifier",
    "DGTSKRegressor",
    "DombiTSKClassifier",
    "DombiTSKRegressor",
    "HTSKClassifier",
    "HTSKRegressor",
    "LogTSKClassifier",
    "LogTSKRegressor",
    "TSKClassifier",
    "TSKRegressor",
]
