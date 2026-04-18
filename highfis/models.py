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

from collections.abc import Callable, Mapping, Sequence
from functools import partial

import torch
from torch import Tensor, nn

from .base import BaseTSK
from .defuzzifiers import LogSumDefuzzifier, SumBasedDefuzzifier
from .layers import AdaptiveDombiRuleLayer, ClassificationConsequentLayer, RegressionConsequentLayer
from .memberships import MembershipFunction
from .t_norms import TNormFn, t_norm_dombi

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
            t_norm_fn = partial(t_norm_dombi, lambda_=self.lambda_)

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
            t_norm_fn = partial(t_norm_dombi, lambda_=self.lambda_)

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
    "DombiTSKClassifier",
    "DombiTSKRegressor",
    "HTSKClassifier",
    "HTSKRegressor",
    "LogTSKClassifier",
    "LogTSKRegressor",
    "TSKClassifier",
    "TSKRegressor",
]
