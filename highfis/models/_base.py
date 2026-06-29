"""Base TSK model that factors the common antecedent-defuzzification pipeline.

This module defines `BaseTSK`, the abstract foundation for all TSK fuzzy
models in highFIS. It factors out the shared antecedent pipeline,
defuzzifier, and training loop so concrete subclasses can focus on
task-specific consequent layers and loss criteria.

The forward pipeline executes four sequential steps:

1. `highfis.layers.MembershipLayer` — evaluates membership functions for each
   input feature.
2. `highfis.layers.RuleLayer` — computes rule firing strengths via a
   configurable rule base and T-norm.
3. **Defuzzifier** — normalizes firing strengths to probability-like weights
   (default: `highfis.defuzzifiers.SoftmaxLogDefuzzifier`).
4. **ConsequentLayer** — produces the final output from the inputs and the
   normalized rule weights.

Concrete subclasses must implement:

- `BaseTSK._build_consequent_layer` — return the task-specific consequent
  module.
- `BaseTSK._default_criterion` — return the default loss function.

Optional overridable hooks:

- `BaseTSK._compute_loss` — customize target preparation or loss composition.
- `BaseTSK._evaluate_validation` — customize the validation metric used for
  early stopping.
"""

from __future__ import annotations

import logging
import sys
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from ..defuzzifiers import SoftmaxLogDefuzzifier
from ..layers import MembershipLayer, RuleLayer
from ..memberships import MembershipFunction
from ..protocols import Defuzzifier
from ..t_norms import TNormFn

logger: logging.Logger = logging.getLogger(__name__)


class BaseTSK(nn.Module):
    """Abstract base for TSK fuzzy models.

    Subclasses must implement :meth:`_build_consequent_layer`.
    """

    default_criterion: type[nn.Module]
    task_type: str
    _optimizer_type: str = "adamw"

    input_mfs: Mapping[str, Sequence[MembershipFunction]]
    input_names: list[str]
    n_inputs: int
    membership_layer: MembershipLayer
    rule_layer: RuleLayer
    n_rules: int
    defuzzifier: Defuzzifier
    consequent_batch_norm: bool
    consequent_bn: nn.BatchNorm1d | None
    consequent_layer: nn.Module
    logger: logging.Logger

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        *,
        rule_base: str = "cartesian",
        t_norm: str | TNormFn = "gmean",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: Defuzzifier | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialize the TSK pipeline layers.

        Args:
            input_mfs: Mapping from feature names to sequences of
                :class:`~highfis.memberships.MembershipFunction` objects.
                Must not be empty.
            rule_base: Rule-base construction strategy.  Supported values:
                ``"cartesian"`` (all MF combinations), ``"coco"``
                (same-index compact), ``"en"`` (enhanced FRB), or
                ``"custom"`` (explicit rules via *rules*).
            t_norm: T-norm name or callable.  Common string values: ``"prod"``,
                ``"gmean"``, ``"min"``, ``"dombi"``, ``"yager"``.  A
                callable implementing the T-norm interface may be passed
                directly.
            rules: Explicit rule index sequences.  Required when
                *rule_base* is ``"custom"``.
            defuzzifier: Normalization module applied to raw rule firing
                strengths.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
            consequent_batch_norm: If ``True``, insert a
                :class:`~torch.nn.BatchNorm1d` layer on the inputs before
                the consequent computation.

        Raises:
            ValueError: If *input_mfs* is empty.
        """
        super().__init__()
        if not input_mfs:
            raise ValueError("input_mfs must not be empty")

        self.input_mfs = input_mfs
        self.input_names = list(input_mfs.keys())
        self.n_inputs = len(self.input_names)
        mf_per_input = [len(input_mfs[name]) for name in self.input_names]

        self.membership_layer = MembershipLayer(input_mfs)
        self.rule_layer = RuleLayer(
            self.input_names,
            mf_per_input,
            rules=rules,
            rule_base=rule_base,
            t_norm=t_norm,
        )
        self.n_rules = self.rule_layer.n_rules
        self.defuzzifier: Defuzzifier = defuzzifier or SoftmaxLogDefuzzifier()
        self.consequent_batch_norm = bool(consequent_batch_norm)
        self.consequent_bn = nn.BatchNorm1d(self.n_inputs) if self.consequent_batch_norm else None
        self.consequent_layer = self._build_consequent_layer()
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(stream_handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_consequent_layer(self) -> nn.Module:
        """Return the task-specific consequent layer."""

    # ------------------------------------------------------------------
    # Shared forward pipeline
    # ------------------------------------------------------------------

    def forward_antecedents(self, x: Tensor) -> Tensor:
        """Compute normalized rule strengths from model antecedents."""
        mu = self.membership_layer(x)
        w = self.rule_layer(mu)
        return self.defuzzifier(w)

    def get_mf_params(self) -> dict[str, list[dict[str, Any]]]:
        """Return a serializable description of the model's membership functions."""
        return {
            name: [{"type": type(mf).__name__, **mf.inspect_params()} for mf in cast(Sequence[MembershipFunction], mfs)]
            for name, mfs in self.membership_layer.input_mfs.items()
        }

    def get_rule_table(self) -> list[dict[str, Any]]:
        """Return the rule base as a table of feature-to-MF indices."""
        return [
            dict(
                rule_id=rule_index,
                **dict(zip(self.rule_layer.input_names, rule, strict=False)),
            )
            for rule_index, rule in enumerate(self.rule_layer.rules)
        ]

    def get_consequent_weights(self) -> Tensor | None:
        """Return the consequent layer weights or ``None`` when unavailable."""
        weight = getattr(self.consequent_layer, "weight", None)
        if isinstance(weight, Tensor):
            return weight.detach()
        return None

    def _forward_train(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass returning ``(output, norm_w)`` to avoid double computation."""
        mu = self.membership_layer(x)
        w = self.rule_layer(mu)
        norm_w = self.defuzzifier(w)
        x_cons = self.consequent_bn(x) if self.consequent_bn is not None else x
        output = self.consequent_layer(x_cons, norm_w)
        return output, norm_w

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass through the TSK pipeline."""
        output, _ = self._forward_train(x)
        return output

    def _run_inference(self, x: Tensor, fn: Callable[[Tensor], Tensor]) -> Tensor:
        """Run *fn(x)* in eval mode, restoring training state and dtype afterward."""
        was_training = self.training
        use_double = x.dtype == torch.float64
        if use_double:
            self.double()
        try:
            self.eval()
            return fn(x)
        finally:
            if use_double:
                self.float()
            self.train(was_training)


__all__: list[str] = ["BaseTSK"]
