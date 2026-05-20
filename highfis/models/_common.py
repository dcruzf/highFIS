"""Shared base classes and utility functions for TSK model families."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from ..base import BaseTSK


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


def build_rule_feature_mask(rules: Sequence[Sequence[int]], dont_care_indices: Sequence[int]) -> Tensor:
    """Build a boolean mask indicating which features are active in each rule."""
    if len(rules) == 0:
        raise ValueError("rules must not be empty")
    n_rules = len(rules)
    n_inputs = len(rules[0])
    if len(dont_care_indices) != n_inputs:
        raise ValueError("dont_care_indices must match the rule input dimension")

    mask = torch.ones((n_rules, n_inputs), dtype=torch.bool)
    for r, rule in enumerate(rules):
        if len(rule) != n_inputs:
            raise ValueError("all rules must have the same length")
        for i, mf_idx in enumerate(rule):
            if mf_idx == dont_care_indices[i]:
                mask[r, i] = False
    return mask


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
