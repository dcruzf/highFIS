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
- `BaseTSK.default_criterion` — class attribute naming the default loss class
  (``CrossEntropyLoss`` for classification, ``MSELoss`` for regression); the
  trainer instantiates it via ``model.default_criterion()``.

Loss composition and validation scoring are hooks on the trainer
(`highfis.optim.GradientTrainer._compute_loss` and
`._evaluate_validation`), not on the model.
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


def set_training_flag(module: nn.Module, mode: bool) -> None:
    """Set ``training`` on *module* and every descendant, bypassing ``nn.Module.train()``.

    Equivalent to ``module.train(mode)``, but ``train()`` walks ``children()`` and routes
    every assignment through ``nn.Module.__setattr__``. A TSK model carries one membership
    submodule per (feature, MF) pair -- 8007 of them at 2000 features -- so a single
    eval/train switch costs thousands of Python calls, which dominated every metric and
    inference pass. For a plain ``bool`` ``nn.Module.__setattr__`` merely falls through to
    ``object.__setattr__``, so this performs the identical assignment without the dispatch.
    """
    object.__setattr__(module, "training", mode)
    for child in module._modules.values():
        if child is not None:
            set_training_flag(child, mode)


#: Largest design matrix, in entries, that ``solve_consequents`` builds (about 1 GiB of doubles).
_MAX_LSE_ELEMENTS = 2**27


def _least_squares(design: Tensor, target: Tensor, ridge: float, rcond: float) -> Tensor:
    """Solve ``min ||design @ B - target||^2 + ridge * ||B||^2`` for ``B``.

    Without a penalty, singular values below ``rcond`` times the largest are treated as zero.
    """
    n_samples, n_columns = design.shape
    if ridge == 0.0:
        # Minimum-norm solution, also for a rank-deficient or underdetermined system. The
        # normalized rule weights add up to one, so the columns are often nearly dependent:
        # the directions below the cut-off are dropped, or they would be fitted with
        # coefficients too large for the precision in which the model stores them.
        return torch.linalg.lstsq(design, target, rcond=rcond, driver="gelsd").solution
    if n_samples >= n_columns:
        gram = design.T @ design
        gram.diagonal().add_(ridge)
        return torch.linalg.solve(gram, design.T @ target)
    # Fewer samples than parameters: the same solution from the smaller system.
    gram = design @ design.T
    gram.diagonal().add_(ridge)
    return design.T @ torch.linalg.solve(gram, target)


def _check_ridge(ridge: float | str) -> None:
    """Reject a penalty that is neither a non-negative number nor ``"auto"``."""
    if isinstance(ridge, str):
        if ridge != "auto":
            raise ValueError(f"ridge must be a non-negative number or 'auto'; got {ridge!r}")
    elif ridge < 0.0:
        raise ValueError("ridge must be non-negative")


#: Candidate penalties of the automatic choice, as fractions of the largest squared singular
#: value of the design matrix.
_RIDGE_GRID = torch.logspace(-6.0, 2.0, 41, dtype=torch.float64)


def _select_ridge(design: Tensor, target: Tensor) -> float:
    """Choose the ridge penalty with the smallest leave-one-out squared error.

    For ridge regression the leave-one-out residuals have a closed form,
    ``e_i / (1 - h_ii)`` with ``h`` the diagonal of the hat matrix, so every candidate is
    evaluated from one singular value decomposition of the design matrix, without refitting.
    """
    u, s, _ = torch.linalg.svd(design, full_matrices=False)
    s2 = s.square()
    projected = u.T @ target  # (rank, outputs)
    u2 = u.square()
    # Part of the targets, and of each sample, outside the column space of the design
    # matrix: exactly zero when there are at least as many columns as samples.
    if u.shape[0] <= u.shape[1]:
        outside, free = torch.zeros_like(target), torch.zeros_like(s2[:1]).expand(u.shape[0])
    else:
        outside, free = target - u @ projected, (1.0 - u2.sum(dim=1)).clamp(min=0.0)
    tiny = torch.finfo(design.dtype).tiny
    best_penalty, best_error = 0.0, float("inf")
    for penalty in (_RIDGE_GRID.to(design.device) * s2.max()).tolist():
        # Written with ``penalty / (s2 + penalty)`` so that small residuals and small
        # denominators keep their precision when the fit is close to an interpolation.
        kept = penalty / (s2 + penalty)
        residual = outside + u @ (kept.unsqueeze(1) * projected)
        denominator = (free + u2 @ kept).clamp(min=tiny)
        error = float((residual / denominator.unsqueeze(1)).square().sum())
        if error < best_error:
            best_penalty, best_error = float(penalty), error
    return best_penalty


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
    lse_ridge_: float

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
        """Return the consequent layer weights or ``None`` when unavailable.

        These are the ``w`` coefficients of the first-order consequent
        ``score_r^c(x) = b_{r,c} + sum_d w_{r,c,d} x_d``; pair them with
        :meth:`get_consequent_bias` to reconstruct a complete rule.
        """
        weight = getattr(self.consequent_layer, "weight", None)
        if isinstance(weight, Tensor):
            return weight.detach()
        return None

    def get_consequent_bias(self) -> Tensor | None:
        """Return the consequent layer bias or ``None`` when unavailable.

        The intercept ``b`` of the first-order consequent, shaped ``(rules, classes)`` for
        classification and ``(rules,)`` for regression. Without it a rule read from
        :meth:`get_consequent_weights` alone is incomplete.
        """
        bias = getattr(self.consequent_layer, "bias", None)
        if isinstance(bias, Tensor):
            return bias.detach()
        return None

    def solve_consequents(self, x: Tensor, y: Tensor, ridge: float | str = 0.0) -> float:
        r"""Replace the consequents by their least-squares solution, with the antecedents as they are.

        With fixed antecedents the output of a TSK system is linear in its consequent
        parameters: :math:`\hat y = \sum_r \bar w_r(x)\,(b_r + \sum_d p_{r,d} x_d)`. The
        parameters that minimize the squared error on ``(x, y)`` are therefore the solution
        of a linear least-squares problem, which is solved here in double precision. For a
        classifier the targets are the one-hot vectors of the classes.

        Args:
            x: Inputs of shape ``(n_samples, n_inputs)``.
            y: Class indices for a classifier, values for a regressor.
            ridge: Penalty on the squared norm of the consequent parameters. ``0.0`` gives
                the least-squares solution of minimum norm, which exists also when there
                are more parameters than samples.
                ``"auto"`` chooses the penalty with the smallest leave-one-out squared error
                on ``(x, y)``, among values spread over eight orders of magnitude; the
                value chosen is kept in ``lse_ridge_``.

        Returns:
            The mean squared error of the solution on ``(x, y)``.

        Raises:
            ValueError: If ``ridge`` is negative or an unknown name, if the gates of a gated consequent layer
                are active, or if the problem is too large to be solved in memory.
        """
        _check_ridge(ridge)
        layer = self.consequent_layer
        if getattr(layer, "mode", "finetune") != "finetune":
            raise ValueError(
                "the consequents can only be solved by least squares once the gates are switched off; "
                "the gated families have their own use_lse option"
            )
        weight = getattr(layer, "weight", None)
        bias = getattr(layer, "bias", None)
        if not isinstance(bias, Tensor):
            raise ValueError("this consequent layer has no parameters to solve for")
        n_rules, n_inputs = int(self.n_rules), int(self.n_inputs)
        n_columns = n_rules * (n_inputs + 1) if isinstance(weight, Tensor) else n_rules
        if int(x.shape[0]) * n_columns > _MAX_LSE_ELEMENTS:
            raise ValueError(
                f"the least-squares problem has {x.shape[0]} samples and {n_columns} consequent parameters, "
                f"more than {_MAX_LSE_ELEMENTS:,} entries; train the consequents by gradient descent instead"
            )

        was_training = self.training
        try:
            set_training_flag(self, False)
            with torch.no_grad():
                norm_w = self.forward_antecedents(x).to(torch.float64)
                x_cons = (self.consequent_bn(x) if self.consequent_bn is not None else x).to(torch.float64)
                if isinstance(weight, Tensor):
                    terms = norm_w.unsqueeze(2) * x_cons.unsqueeze(1)  # (samples, rules, inputs)
                    mask = getattr(layer, "rule_feature_mask", None)
                    if isinstance(mask, Tensor):
                        terms = terms * mask.to(terms.dtype).unsqueeze(0)
                    design = torch.cat([norm_w.unsqueeze(2), terms], dim=2).reshape(x.shape[0], n_columns)
                else:
                    design = norm_w
                if self.task_type == "classification":
                    target = torch.zeros((x.shape[0], int(bias.shape[-1])), dtype=torch.float64, device=x.device)
                    target.scatter_(1, y.reshape(-1, 1).long(), 1.0)
                else:
                    target = y.reshape(-1, 1).to(torch.float64)

                # The solution is stored in the precision of the model, which bounds the
                # conditioning that is worth resolving.
                rcond = 10.0 * float(torch.finfo(bias.dtype).eps)
                penalty = _select_ridge(design, target) if isinstance(ridge, str) else float(ridge)
                self.lse_ridge_ = penalty
                solution = _least_squares(design, target, penalty, rcond)
                error = float(((design @ solution - target) ** 2).mean())

                if isinstance(weight, Tensor):
                    blocks = solution.reshape(n_rules, n_inputs + 1, -1)  # (rules, 1 + inputs, outputs)
                    new_bias, new_weight = blocks[:, 0, :], blocks[:, 1:, :].permute(0, 2, 1)
                    weight.copy_(new_weight.reshape(weight.shape).to(weight.dtype))
                else:
                    new_bias = solution
                bias.copy_(new_bias.reshape(bias.shape).to(bias.dtype))
        finally:
            set_training_flag(self, was_training)
        return error

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
        # A float64 input promotes a single-precision model for the call only. The model's own
        # dtype is restored afterwards, so one fitted in float64 is never downcast to float32.
        original_dtype = next((p.dtype for p in self.parameters() if p.is_floating_point()), None)
        promote = x.dtype == torch.float64 and original_dtype not in (None, torch.float64)
        if promote:
            self.double()
        try:
            set_training_flag(self, False)
            return fn(x)
        finally:
            if promote:
                self.to(original_dtype)
            set_training_flag(self, was_training)


__all__: list[str] = ["BaseTSK", "set_training_flag"]
