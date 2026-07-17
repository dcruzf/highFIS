"""Shared base classes and utility functions for TSK model families."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ._base import BaseTSK


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


class BaseTSKClassifierModel(BaseTSK):
    """Abstract classifier base that provides task-specific training and inference helpers."""

    task_type = "classification"
    default_criterion = nn.CrossEntropyLoss
    n_classes: int

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return class probabilities computed with softmax."""
        with torch.no_grad():
            return torch.softmax(self._run_inference(x, self.forward), dim=1)

    def predict(self, x: Tensor) -> Tensor:
        """Return predicted class indices."""
        with torch.no_grad():
            return torch.argmax(self.predict_proba(x), dim=1)


class BaseTSKRegressorModel(BaseTSK):
    """Abstract regressor base that provides task-specific training and inference helpers."""

    task_type = "regression"
    default_criterion = nn.MSELoss

    def predict(self, x: Tensor) -> Tensor:
        """Return predicted values as a 1-D tensor."""
        with torch.no_grad():
            return self._run_inference(x, lambda inp: self.forward(inp).squeeze(1))


def _run_threshold_grid_search(
    model: Any,
    x: Tensor,
    y: Tensor,
    x_eval: Tensor,
    y_eval: Tensor,
    zeta_lambda: Sequence[float],
    zeta_theta: Sequence[float],
    use_lse: bool,
    verbose: bool,
) -> tuple[float, dict[str, Any] | None, float, float, float, float]:
    best_score = float("-inf")
    best_state: dict[str, Any] | None = None
    best_tau_lambda = 0.0
    best_tau_theta = 0.0
    best_zeta_lambda = 0.0
    best_zeta_theta = 0.0

    # Callers require ``model`` to come back untouched: ``search_thresholds`` derives the
    # surviving feature/rule indices from its gate values *after* this returns, and a
    # pruned gate would never clear the threshold. Restoring from a state_dict snapshot
    # is enough because ``apply_thresholds`` and the LSE refit only write registered
    # parameters and buffers. Only the zero-order -> first-order conversion rebuilds the
    # module graph, so it is the one case that still needs a private copy.
    needs_conversion = use_lse and "ZeroOrder" in model.consequent_layer.__class__.__name__
    work = copy.deepcopy(model) if needs_conversion else model
    if needs_conversion:
        work.convert_to_first_order()

    snapshot = {k: v.detach().clone() for k, v in work.state_dict().items()}
    was_training = work.training

    try:
        for zeta_l in zeta_lambda:
            for zeta_t in zeta_theta:
                tau_l, tau_t = work.compute_thresholds(zeta_l, zeta_t)
                work.apply_thresholds(tau_l, tau_t)
                if use_lse:
                    work._fit_first_order_consequents_lse(x, y)

                score = work._evaluate_threshold_score(x_eval, y_eval)
                if verbose:
                    model.logger.info("zeta_lambda=%s zeta_theta=%s score=%.6f", zeta_l, zeta_t, score)

                if score > best_score:
                    best_score = score
                    best_state = copy.deepcopy(work.state_dict()) if use_lse else None
                    best_tau_lambda = tau_l
                    best_tau_theta = tau_t
                    best_zeta_lambda = zeta_l
                    best_zeta_theta = zeta_t

                work.load_state_dict(snapshot)
    finally:
        # ``_fit_first_order_consequents_lse`` flips the module to eval mode, and that
        # flag lives outside ``state_dict``.
        work.load_state_dict(snapshot)
        if work.training != was_training:
            work.train(was_training)

    if best_score == float("-inf"):
        raise RuntimeError("threshold search did not yield a valid candidate")

    return best_score, best_state, best_tau_lambda, best_tau_theta, best_zeta_lambda, best_zeta_theta


def _apply_thresholds_and_pruning_inplace(
    model: Any,
    x: Tensor,
    y: Tensor,
    best_tau_lambda: float,
    best_tau_theta: float,
    best_state: dict[str, Any] | None,
    use_lse: bool,
    structural: bool,
    sf: list[int],
    sr: list[int],
    result: dict[str, Any],
) -> None:
    if structural:
        class_name = model.consequent_layer.__class__.__name__
        if use_lse and "ZeroOrder" in class_name:
            model.convert_to_first_order()
        model.apply_thresholds(best_tau_lambda, best_tau_theta)
        model.prune_structure(sf, sr)
        if use_lse:
            model._fit_first_order_consequents_lse(x[:, sf], y)
        result["surviving_feature_indices"] = sf
        result["surviving_rule_indices"] = sr
    else:
        class_name = model.consequent_layer.__class__.__name__
        if use_lse:
            if "ZeroOrder" in class_name:
                model.convert_to_first_order()
            if best_state is None:  # pragma: no cover
                raise RuntimeError("best_state is None despite use_lse=True")
            model.load_state_dict(best_state)
        else:
            model.apply_thresholds(best_tau_lambda, best_tau_theta)
