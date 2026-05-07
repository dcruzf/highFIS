"""Concrete TSK model variants.

Each class in this module is a :class:`~highfis.base.BaseTSK` subclass that
bundles a specific antecedent strategy (t-norm), defuzzification head, and
consequent architecture.  Users typically access these through the sklearn
estimator wrappers in :mod:`highfis.estimators`.

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
     - :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`
     - Cui, Wu & Xu (IJCNN 2021)
   * - **TSK (vanilla)**
     - ``prod``
     - :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`
     - Takagi & Sugeno (IEEE SMC 1985)
   * - **LogTSK**
     - ``prod``
     - :class:`~highfis.defuzzifiers.InvLogDefuzzifier`
     - Cui, Wu & Xu (IEEE TFS 2021)
   * - **DombiTSK**
     - ``dombi``
     - :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`
     - Dombi (1982)
   * - **AYATSK**
     - ``yager``
     - :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`
     - Xue et al. (IEEE TSMC 2025)
   * - **AdaTSK**
     - adaptive softmin (Ada-softmin)
     - :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`
     - Xue et al. (IEEE TFS 2023)
   * - **FSRE-AdaTSK**
     - adaptive softmin
     - :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`
     - Xue et al. (IEEE TFS 2023)
   * - **DG-ALETSK**
     - ALE-softmin
     - :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`
     - Xue et al. (IEEE TFUZZ 2023)
   * - **DG-TSK**
     - product + M-gate
     - :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`
     - Xue et al. (Fuzzy Sets Syst. 2023)

Scientific correspondence
~~~~~~~~~~~~~~~~~~~~~~~~~

Normalization is always computed over rules. In highFIS each model variant
pairs an antecedent aggregation operator with a defuzzifier, and the public
classes defined in this module are grouped as follows:

* **HTSK**: ``t_norm="gmean"`` + ``SoftmaxLogDefuzzifier``
  - ``HTSKClassifier``
  - ``HTSKRegressor``
  - behaviour: ``softmax(log(w^{1/D}))``
* **TSK**: ``t_norm="prod"`` + ``SumBasedDefuzzifier``
  - ``TSKClassifier``
  - ``TSKRegressor``
  - behaviour: ``w_r / Σw``
* **LogTSK**: ``t_norm="prod"`` + ``InvLogDefuzzifier``
  - ``LogTSKClassifier``
  - ``LogTSKRegressor``
  - behaviour: ``softmax(log(w) / τ)``
* **DombiTSK**: ``t_norm="dombi"`` + ``SumBasedDefuzzifier``
  - ``DombiTSKClassifier``
  - ``DombiTSKRegressor``
* **AYATSK**: ``t_norm="yager"`` + ``SumBasedDefuzzifier``
  - ``AYATSKClassifier``
  - ``AYATSKRegressor``
* **AdaTSK**: adaptive softmin (Ada-softmin) + ``SumBasedDefuzzifier``
  - ``AdaTSKClassifier``
  - ``AdaTSKRegressor``
* **FSRE-AdaTSK**: adaptive softmin + ``SoftmaxLogDefuzzifier``
  - ``FSREAdaTSKClassifier``
  - ``FSREAdaTSKRegressor``
* **DG-ALETSK**: ALE-softmin + ``SoftmaxLogDefuzzifier``
  - ``DGALETSKClassifier``
  - ``DGALETSKRegressor``
* **DG-TSK**: product + M-gate + ``SoftmaxLogDefuzzifier``
  - ``DGTSKClassifier``
  - ``DGTSKRegressor``

All of these classes are exported by the module and are intended for use as
concrete TSK classifiers and regressors.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from .base import BaseTSK
from .defuzzifiers import InvLogDefuzzifier, SoftmaxLogDefuzzifier, SumBasedDefuzzifier
from .layers import (
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

    Replaces the standard softmax-based rule normalisation with a
    dimensionality-normalised variant (geometric-mean t-norm +
    :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`) that prevents
    softmax saturation as the number of inputs grows.

    Reference:
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
    """TSK regressor with HTSK defuzzification for high-dimensional data.

    Replaces the standard softmax-based rule normalisation with a
    dimensionality-normalised variant (geometric-mean t-norm +
    :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`) that prevents
    softmax saturation as the number of inputs grows.

    Reference:
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


# =====================================================================
# Vanilla TSK  (Takagi & Sugeno, IEEE SMC 1985)
#
#   w_r = ∏_{d=1}^{D} μ_{r,d}(x_d)               (product t-norm)
#   f̄_r = w_r / Σ_{i=1}^{R} w_i                   (sum-based)
# =====================================================================


class TSKClassifier(BaseTSKClassifier):
    r"""Vanilla TSK classifier with sum-based defuzzification.

    Implements the original Takagi-Sugeno-Kang inference.

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
    r"""Vanilla TSK regressor with sum-based defuzzification.

    Implements the original Takagi-Sugeno-Kang inference.

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


class DombiTSKClassifier(BaseTSKClassifier):
    r"""TSK classifier with a fixed Dombi T-norm in the antecedent.

    The Dombi T-norm with parameter :math:`\lambda > 0` is defined as:

    .. math::
        T_\lambda(a, b) =
        \frac{1}{1 + \left[\left(\frac{1-a}{a}\right)^\lambda
        + \left(\frac{1-b}{b}\right)^\lambda\right]^{1/\lambda}}

    For :math:`\lambda = 1` it recovers the algebraic product T-norm.
    The parameter is fixed at construction time and shared across all rules;
    use :class:`AdaTSKClassifier` for a high-dimensional alternative based
    on Ada-softmin.

    Reference:
        Dombi, J. (1982). A general class of fuzzy operators, the De Morgan
        class of fuzzy operators and fuzziness measures induced by fuzzy
        operators. *Fuzzy Sets and Systems*, 8(2):149-163.
    """

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
        """Initialise the Dombi TSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: ``"cartesian"`` or ``"coco"`` rule-base strategy.
            t_norm: T-norm identifier (default ``"dombi"``).
            lambda_: Dombi parameter ``λ > 0``.  ``λ = 1`` gives the
                algebraic product.
            t_norm_fn: Optional custom t-norm callable; overrides
                ``lambda_`` and ``t_norm`` when provided.
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.

        Raises:
            ValueError: If ``n_classes < 2`` or ``lambda_ <= 0``.
        """
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
    r"""TSK regressor with a fixed Dombi T-norm in the antecedent.

    The Dombi T-norm with parameter :math:`\\lambda > 0` is defined as:

    .. math::
        T_\\lambda(a, b) =
        \frac{1}{1 + \\left[\\left(\frac{1-a}{a}\right)^\\lambda
        + \\left(\frac{1-b}{b}\right)^\\lambda\right]^{1/\\lambda}}

    For :math:`\\lambda = 1` it recovers the algebraic product T-norm.
    The parameter is fixed at construction time and shared across all rules;
    use :class:`AdaTSKRegressor` for a high-dimensional alternative based
    on Ada-softmin.

    Reference:
        Dombi, J. (1982). A general class of fuzzy operators, the De Morgan
        class of fuzzy operators and fuzziness measures induced by fuzzy
        operators. *Fuzzy Sets and Systems*, 8(2):149-163.
    """

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
        """Initialise the Dombi TSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: ``"cartesian"`` or ``"coco"`` rule-base strategy.
            t_norm: T-norm identifier (default ``"dombi"``).
            lambda_: Dombi parameter ``λ > 0``.
            t_norm_fn: Optional custom t-norm callable.
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SumBasedDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.

        Raises:
            ValueError: If ``lambda_ <= 0``.
        """
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


class AYATSKClassifier(BaseTSKClassifier):
    r"""TSK classifier with an adaptive Yager T-norm in the antecedent.

    The Yager T-norm with learnable per-rule exponent $p_r > 0$ is:

    Math:
        $$
        T_{p_r}(a_1, \ldots, a_D) =
        \max\!\left(0,\; 1 - \left[\sum_{d=1}^{D}(1-a_d)^{p_r}\right]^{1/p_r}\right)
        $$

    Each rule maintains its own exponent, which is jointly optimised with
    the consequent parameters via back-propagation.

    Reference:
        G. Xue, Y. Yang and J. Wang, "Adaptive Yager T-Norm-Based Takagi-Sugeno-Kang Fuzzy Systems,"
        in IEEE Transactions on Systems, Man, and Cybernetics: Systems,
        vol. 55, no. 12, pp. 9802-9815, Dec. 2025, doi: 10.1109/TSMC.2025.3621346.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "coco",
        t_norm: str = "yager",
        t_norm_fn: TNormFn | None = None,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the AYATSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: ``"coco"`` (default) or ``"cartesian"``.
            t_norm: T-norm identifier (default ``"yager"``).
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
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class AYATSKRegressor(BaseTSKRegressor):
    r"""TSK regressor with an adaptive Yager T-norm in the antecedent.

    The Yager T-norm with learnable per-rule exponent $p_r > 0$ is:

    Math:
        $$
        T_{p_r}(a_1, \\ldots, a_D) =
        \\max\\!\\left(0,\\; 1 - \\left[\\sum_{d=1}^{D}(1-a_d)^{p_r}\right]^{1/p_r}\right)
        $$

    Each rule maintains its own exponent, which is jointly optimised with
    the consequent parameters via back-propagation.

    Reference:
        G. Xue, Y. Yang and J. Wang, "Adaptive Yager T-Norm-Based Takagi-Sugeno-Kang Fuzzy Systems,"
        in IEEE Transactions on Systems, Man, and Cybernetics: Systems,
        vol. 55, no. 12, pp. 9802-9815, Dec. 2025, doi: 10.1109/TSMC.2025.3621346.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "coco",
        t_norm: str = "yager",
        t_norm_fn: TNormFn | None = None,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the AYATSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: ``"coco"`` (default) or ``"cartesian"``.
            t_norm: T-norm identifier (default ``"yager"``).
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
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()


class AdaTSKClassifier(BaseTSKClassifier):
    r"""TSK classifier with adaptive softmin antecedent (AdaTSK).

    The firing strength of each rule is computed with the Ada-softmin operator
    (Xue et al., 2023, eqs. 15-16):

    .. math::
        f_r(\mathbf{x}) =
        \left(\frac{1}{D}\sum_{d=1}^{D}\mu_{r,d}^{\hat{q}}(\mathbf{x})
        \right)^{1/\hat{q}},\quad
        \hat{q} = \left\lceil\frac{690}{\ln\!\min_d \mu_{r,d}(\mathbf{x})}
        \right\rceil \in [-1000,\,-1].

    The index :math:`\hat{q}` is computed from the current membership values
    on every forward pass and is *not* a learnable parameter.  This adaptive
    scheme prevents both *numeric underflow* and *fake minimum* that plague
    fixed-parameter softmin operators for high-dimensional data.

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
        """Initialise the AdaTSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: ``"coco"`` (default) or ``"cartesian"``.
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
            t_norm_fn=None,
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


class AdaTSKRegressor(BaseTSKRegressor):
    r"""TSK regressor with adaptive softmin antecedent (AdaTSK).

    The firing strength of each rule is computed with the Ada-softmin operator
    (Xue et al., 2023, eqs. 15-16):

    .. math::
        f_r(\mathbf{x}) =
        \left(\frac{1}{D}\sum_{d=1}^{D}\mu_{r,d}^{\hat{q}}(\mathbf{x})
        \right)^{1/\hat{q}},\quad
        \hat{q} = \left\lceil\frac{690}{\ln\!\min_d \mu_{r,d}(\mathbf{x})}
        \right\rceil \in [-1000,\,-1].

    The index :math:`\hat{q}` is computed from the current membership values
    on every forward pass and is *not* a learnable parameter.  This adaptive
    scheme prevents both *numeric underflow* and *fake minimum* that plague
    fixed-parameter softmin operators for high-dimensional data.

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
        """Initialise the AdaTSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: ``"coco"`` (default) or ``"cartesian"``.
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
            t_norm_fn=None,
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


class FSREAdaTSKClassifier(BaseTSKClassifier):
    r"""FSRE-AdaTSK classifier with adaptive softmin antecedent and gated consequents.

    FSRE-AdaTSK (Feature Selection and Rule Extraction) extends AdaTSK with:

    * **Adaptive softmin antecedent** — a differentiable softmin operator
      that produces sparse, interpretable firing strengths.
    * **Double-group gates** — per-feature gates (λ_d) **in the consequent**
      only that perform feature selection, and per-rule gates (θ_r) also
      **in the consequent** that perform rule extraction.  Gates are never
      applied to antecedent membership values.

    Training follows a three-phase protocol:

    1. **FS phase** (:meth:`fit_fs`) — train on the initial CoCo-FRB;
       only feature gates :math:`M(\\lambda_d)` are active (eq. 21).
    2. **RE phase** (:meth:`fit_re`) — expand to En-FRB and retrain;
       only rule gates :math:`M(\\theta_r)` are active (eq. 22).
    3. **Fine-tune** (:meth:`fit_finetune`) — compact model fine-tuning
       with no gates (plain eq. 5).

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
            rule_base: ``"coco"`` (default) or ``"cartesian"``.
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

    FSRE-AdaTSK (Feature Selection and Rule Extraction) extends AdaTSK with:

    * **Adaptive softmin antecedent** — a differentiable softmin operator
      that produces sparse, interpretable firing strengths.
    * **Double-group gates** — per-feature gates (λ_d) **in the consequent**
      only that perform feature selection, and per-rule gates (θ_r) also
      **in the consequent** that perform rule extraction.  Gates are never
      applied to antecedent membership values.

    Training follows a three-phase protocol:

    1. **FS phase** (:meth:`fit_fs`) — train on the initial CoCo-FRB;
       only feature gates :math:`M(\\lambda_d)` are active (eq. 21).
    2. **RE phase** (:meth:`fit_re`) — expand to En-FRB and retrain;
       only rule gates :math:`M(\\theta_r)` are active (eq. 22).
    3. **Fine-tune** (:meth:`fit_finetune`) — compact model fine-tuning
       with no gates (plain eq. 5).

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
            rule_base: ``"coco"`` (default) or ``"cartesian"``.
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


class DGALETSKClassifier(BaseTSKClassifier):
    """DG-ALETSK classifier with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-AdaTSK by replacing the adaptive softmin with the
    *Adaptive Ln-Exp (ALE)* softmin — a smoother variant with improved
    numerical stability.  It also uses a zero-order consequent in the DG
    (data-guided) training phase and optionally converts to first-order
    after gate-based pruning.

    Training follows a three-phase protocol:

    1. **DG phase** (:meth:`fit_dg_phase`) — train with zero-order consequents.
    2. **Threshold search** (:meth:`search_thresholds`) — prune features/rules.
    3. **Fine-tune** (:meth:`fit_finetune`) — retrain first-order consequents.

    Reference:
        G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional
        Fuzzy Approach With Simultaneous Feature Selection and Rule
        Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no.
        11, pp. 3866-3880, Nov. 2023, doi: 10.1109/TFUZZ.2023.3270445.
    """

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
        """Initialise the DG-ALETSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: ``"coco"`` (default) or ``"cartesian"``.
            lambda_init: Initial ALE-softmin parameter ``alpha > 0``
                (default ``1.0``).
            rules: Explicit rule antecedent indices; ignored when
                ``use_en_frb=True``.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            eps: Numerical stability epsilon for the ALE-softmin operator.
            use_en_frb: Start directly from the Enhanced FRB (En-FRB).

        Raises:
            ValueError: If ``n_classes < 2`` or ``lambda_init <= 0``.
        """
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
    """DG-ALETSK regressor with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-AdaTSK by replacing the adaptive softmin with the
    *Adaptive Ln-Exp (ALE)* softmin — a smoother variant with improved
    numerical stability.  It also uses a zero-order consequent in the DG
    (data-guided) training phase and optionally converts to first-order
    after gate-based pruning.

    Training follows a three-phase protocol:

    1. **DG phase** (:meth:`fit_dg_phase`) — train with zero-order consequents.
    2. **Threshold search** (:meth:`search_thresholds`) — prune features/rules.
    3. **Fine-tune** (:meth:`fit_finetune`) — retrain first-order consequents.

    Reference:
        G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional
        Fuzzy Approach With Simultaneous Feature Selection and Rule
        Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no.
        11, pp. 3866-3880, Nov. 2023, doi: 10.1109/TFUZZ.2023.3270445.
    """

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
        """Initialise the DG-ALETSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: ``"coco"`` (default) or ``"cartesian"``.
            lambda_init: Initial ALE-softmin parameter ``alpha > 0``
                (default ``1.0``).
            rules: Explicit rule antecedent indices; ignored when
                ``use_en_frb=True``.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            eps: Numerical stability epsilon for the ALE-softmin operator.
            use_en_frb: Start directly from the Enhanced FRB (En-FRB).

        Raises:
            ValueError: If ``lambda_init <= 0``.
        """
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
    """DG-TSK classifier with M-gate antecedent and point-based FRB (P-FRB).

    DG-TSK (Xue et al., Fuzzy Sets and Systems 2023) uses a data-guided
    M-gate function to automatically select relevant features and rules.
    It supports two rule-base strategies:

    * **CoCo-FRB** — standard one-cluster-per-rule base.
    * **P-FRB** (point-based, via ``use_en_frb=True``) — Enhanced FRB for
      compact rule extraction.

    Like DG-ALETSK, training uses zero-order consequents in the DG phase
    and converts to first-order before fine-tuning.

    Reference:
        Xue, Y., et al. (2023). Data-guided TSK fuzzy systems. *Fuzzy Sets
        and Systems*.
        https://doi.org/10.1016/j.fss.2023.108627
    """

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
        """Initialise the DG-TSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: ``"coco"`` (default) or ``"cartesian"``.
            gate_fea: Gate function for antecedent feature selection.
                ``"gate_m"`` (default) uses the M-gate from the DG-TSK paper.
                Can also be any callable ``Tensor → Tensor``.
            gate_rule: Gate function for consequent rule selection.
                Same options as ``gate_fea``.
            rules: Explicit rule antecedent indices; ignored when
                ``use_en_frb=True``.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            eps: Numerical stability epsilon.
            use_en_frb: Use the Enhanced FRB (P-FRB) rule base.

        Raises:
            ValueError: If ``n_classes < 2``.
        """
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
    """DG-TSK regressor with M-gate antecedent and point-based FRB (P-FRB).

    See :class:`DGTSKClassifier` for a description of the DG-TSK model.

    Reference:
        Xue, Y., et al. (2023). Data-guided TSK fuzzy systems. *Fuzzy Sets
        and Systems*.
        https://doi.org/10.1016/j.fss.2023.108627
    """

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
        """Initialise the DG-TSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: ``"coco"`` (default) or ``"cartesian"``.
            gate_fea: Gate function for antecedent feature selection
                (default ``"gate_m"``).
            gate_rule: Gate function for consequent rule selection
                (default ``"gate_m"``).
            rules: Explicit rule antecedent indices; ignored when
                ``use_en_frb=True``.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            eps: Numerical stability epsilon.
            use_en_frb: Use the Enhanced FRB (P-FRB) rule base.
        """
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
    r"""LogTSK classifier with scale-invariant log-space defuzzification.

    Firing strengths are normalized using the inverse-log formula from
    Du et al. (2020), which is immune to softmax saturation in
    high-dimensional input spaces:

    .. math::
        \bar{f}_r = \frac{1/|Z_r|}{\sum_{i=1}^{R} 1/|Z_i|}

    where :math:`Z_r = \log f_r = \sum_{d=1}^{D} \log \mu_{r,d} \leq 0`.
    Because the normalized weights depend only on the *relative magnitudes*
    of :math:`Z_r`, the output is scale-invariant in log-space.

    References:
    ----------
    Cui, Y., Wu, D. & Xu, Y. (2021). "Optimize TSK Fuzzy Systems for
    Regression Problems: Mini-Batch Gradient Descent With Regularization,
    DropRule, and AdaBound (MBGD-RDA)." *IEEE Trans. Fuzzy Syst.*
    29(5):1003-1015. §III-A.
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
        """Initialise the LogTSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: ``"cartesian"`` or ``"coco"`` rule-base strategy.
            t_norm: Antecedent aggregation operator (default ``"prod"``).
            t_norm_fn: Optional custom t-norm callable.
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
            t_norm_fn=t_norm_fn,
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


class LogTSKRegressor(BaseTSKRegressor):
    r"""LogTSK regressor with scale-invariant log-space defuzzification.

    Firing strengths are normalized using the inverse-log formula from
    Du et al. (2020), which is immune to softmax saturation in
    high-dimensional input spaces:

    .. math::
        \bar{f}_r = \frac{1/|Z_r|}{\sum_{i=1}^{R} 1/|Z_i|}

    where :math:`Z_r = \log f_r = \sum_{d=1}^{D} \log \mu_{r,d} \leq 0`.
    Because the normalized weights depend only on the *relative magnitudes*
    of :math:`Z_r`, the output is scale-invariant in log-space.

    References:
    ----------
    Cui, Y., Wu, D. & Xu, Y. (2021). "Optimize TSK Fuzzy Systems for
    Regression Problems: Mini-Batch Gradient Descent With Regularization,
    DropRule, and AdaBound (MBGD-RDA)." *IEEE Trans. Fuzzy Syst.*
    29(5):1003-1015. §III-A.
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
        """Initialise the LogTSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: ``"cartesian"`` or ``"coco"`` rule-base strategy.
            t_norm: Antecedent aggregation operator (default ``"prod"``).
            t_norm_fn: Optional custom t-norm callable.
            rules: Explicit rule antecedent indices.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.InvLogDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
        """
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            t_norm_fn=t_norm_fn,
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
