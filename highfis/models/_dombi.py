"""Dombi T-norm TSK and ADMTSK fuzzy model classes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from torch import nn

from ..defuzzifiers import SumBasedDefuzzifier
from ..layers import (
    ClassificationConsequentLayer,
    RegressionConsequentLayer,
)
from ..memberships import MembershipFunction
from ..t_norms import AdaptiveDombiTNorm, DombiTNorm, TNormFn
from ._common import (
    BaseTSKClassifier,
    BaseTSKRegressor,
)


class DombiTSKClassifier(BaseTSKClassifier):
    r"""TSK classifier with a fixed Dombi T-norm in the antecedent.

    DombiTSK extends TSK fuzzy inference by using a Dombi t-norm
    aggregation in antecedent evaluation while keeping first-order
    linear consequents.

    Reference:
        G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: A
        High-Dimensional Takagi-Sugeno-Kang Fuzzy System Based on
        Adaptive Dombi T-Norm," in IEEE Transactions on Fuzzy
        Systems, vol. 33, no. 6, pp. 1767-1780, June 2025,
        doi: 10.1109/TFUZZ.2025.3535640.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "cartesian",
        t_norm: str | TNormFn = "dombi",
        lambda_: float = 1.0,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the Dombi TSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: Rule-base construction strategy.  ``"cartesian"``
                (default, all MF combinations), ``"coco"`` (same-index
                compact), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            t_norm: T-norm name or callable (default ``"dombi"``). When a
                string, a ``DombiTNorm`` with *lambda_* is used.
            lambda_: Dombi parameter ``λ > 0``.  ``λ = 1`` gives the
                algebraic product.  Ignored when *t_norm* is callable.
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
        if not callable(t_norm):
            t_norm = DombiTNorm(lambda_=self.lambda_)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
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

    DombiTSK extends TSK fuzzy inference by using a Dombi t-norm
    aggregation in antecedent evaluation while keeping first-order
    linear consequents.

    Reference:
        G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: A
        High-Dimensional Takagi-Sugeno-Kang Fuzzy System Based on
        Adaptive Dombi T-Norm," in IEEE Transactions on Fuzzy
        Systems, vol. 33, no. 6, pp. 1767-1780, June 2025,
        doi: 10.1109/TFUZZ.2025.3535640.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "cartesian",
        t_norm: str | TNormFn = "dombi",
        lambda_: float = 1.0,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialise the Dombi TSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: Rule-base construction strategy.  ``"cartesian"``
                (default, all MF combinations), ``"coco"`` (same-index
                compact), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            t_norm: T-norm name or callable (default ``"dombi"``). When a
                string, a ``DombiTNorm`` with *lambda_* is used.
            lambda_: Dombi parameter ``λ > 0``.  Ignored when *t_norm* is callable.
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
        if not callable(t_norm):
            t_norm = DombiTNorm(lambda_=self.lambda_)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()


class ADMTSKClassifier(BaseTSKClassifier):
    r"""Adaptive Dombi TSK classifier with Composite Gaussian membership functions.

    ADMTSK is an adaptive Dombi TSK fuzzy system designed for high-dimensional inference.
    It combines a Dombi T-norm antecedent with a positive lower-bound Composite Gaussian
    membership function (CGMF) and normalized first-order consequents.

    Reference:
        G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: A High-Dimensional
        Takagi-Sugeno-Kang Fuzzy System Based on Adaptive Dombi T-Norm," in IEEE
        Transactions on Fuzzy Systems, vol. 33, no. 6, pp. 1767-1780, June 2025,
        doi: 10.1109/TFUZZ.2025.3535640.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "coco",
        t_norm: str | TNormFn = "dombi",
        adaptive: bool = True,
        lambda_: float = 1.0,
        lower_bound: float = 1.0 / math.e,
        k: float = 10.0,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialize the ADMTSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                membership functions.
            n_classes: Number of output classes. Must be >= 2.
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            t_norm: T-norm name or callable (default ``"dombi"``). When a
                string, an adaptive or fixed ``DombiTNorm`` is built from
                *adaptive*, *lambda_*, *lower_bound*, and *k*.
            adaptive: If True, compute adaptive lambda using the feature
                dimension and membership lower bound.
            lambda_: Fixed Dombi parameter ``λ > 0`` when adaptive is False.
            lower_bound: The lower bound for Composite GMF values.
            k: Heuristic constant used to compute adaptive lambda.
            rules: Explicit rule antecedent indices for custom rule bases.
            defuzzifier: Optional defuzzifier module.
            consequent_batch_norm: If True, apply batch normalization to
                consequent inputs.

        Raises:
            ValueError: If ``n_classes < 2`` or if ``lambda_`` is invalid
                when adaptive is False.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if not adaptive and lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0")

        self.n_classes = int(n_classes)
        self.adaptive = bool(adaptive)
        self.lambda_ = float(lambda_)
        self.lower_bound = float(lower_bound)
        self.k = float(k)

        if not callable(t_norm):
            if self.adaptive:
                t_norm = AdaptiveDombiTNorm(
                    dimension=len(input_mfs),
                    lower_bound=self.lower_bound,
                    k=self.k,
                )
            else:
                t_norm = DombiTNorm(lambda_=self.lambda_)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class ADMTSKRegressor(BaseTSKRegressor):
    r"""Adaptive Dombi TSK regressor with Composite Gaussian membership functions.

    ADMTSK is an adaptive Dombi TSK fuzzy system designed for high-dimensional inference.
    It combines a Dombi T-norm antecedent with a positive lower-bound Composite Gaussian
    membership function (CGMF) and normalized first-order consequents.

    Reference:
        G. Xue, L. Hu, J. Wang and S. Ablameyko, "ADMTSK: A High-Dimensional
        Takagi-Sugeno-Kang Fuzzy System Based on Adaptive Dombi T-Norm," in IEEE
        Transactions on Fuzzy Systems, vol. 33, no. 6, pp. 1767-1780, June 2025,
        doi: 10.1109/TFUZZ.2025.3535640.
    """

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "coco",
        t_norm: str | TNormFn = "dombi",
        adaptive: bool = True,
        lambda_: float = 1.0,
        lower_bound: float = 1.0 / math.e,
        k: float = 10.0,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialize the ADMTSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                membership functions.
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB), or ``"custom"``
                (explicit rules via *rules*).
            t_norm: T-norm name or callable (default ``"dombi"``). When a
                string, an adaptive or fixed ``DombiTNorm`` is built from
                *adaptive*, *lambda_*, *lower_bound*, and *k*.
            adaptive: If True, compute adaptive lambda using the feature
                dimension and membership lower bound.
            lambda_: Fixed Dombi parameter ``λ > 0`` when adaptive is False.
            lower_bound: The lower bound for Composite GMF values.
            k: Heuristic constant used to compute adaptive lambda.
            rules: Explicit rule antecedent indices for custom rule bases.
            defuzzifier: Optional defuzzifier module.
            consequent_batch_norm: If True, apply batch normalization to
                consequent inputs.

        Raises:
            ValueError: If ``lambda_`` is invalid when adaptive is False.
        """
        if not adaptive and lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0")

        self.adaptive = bool(adaptive)
        self.lambda_ = float(lambda_)
        self.lower_bound = float(lower_bound)
        self.k = float(k)

        if not callable(t_norm):
            if self.adaptive:
                t_norm = AdaptiveDombiTNorm(
                    dimension=len(input_mfs),
                    lower_bound=self.lower_bound,
                    k=self.k,
                )
            else:
                t_norm = DombiTNorm(lambda_=self.lambda_)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            rules=rules,
            defuzzifier=defuzzifier or SumBasedDefuzzifier(),
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()
