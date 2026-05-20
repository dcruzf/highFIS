"""Antecedent and consequent layers for highFIS fuzzy models.

This module provides ``torch.nn.Module`` building blocks for the TSK
antecedent and consequent pipeline. The layers here are used by concrete
TSK model variants in ``highfis.models``.

Layer overview:
    **Membership**
        - ``MembershipLayer``

    **Rule aggregation**
        - ``RuleLayer``
        - ``AdaSoftminRuleLayer``
        - ``DGALETSKRuleLayer``
        - ``DGTSKRuleLayer``
        - ``AdaptiveDombiRuleLayer``

    **Consequent heads**
        - ``ClassificationConsequentLayer``
        - ``RegressionConsequentLayer``
        - ``SparseClassificationConsequentLayer``
        - ``SparseRegressionConsequentLayer``
        - ``GatedClassificationConsequentLayer``
        - ``GatedClassificationZeroOrderConsequentLayer``
        - ``GatedRegressionConsequentLayer``
        - ``GatedRegressionZeroOrderConsequentLayer``

Gate activations (see :mod:`highfis.gates`):
    ``gate1``, ``gate2``, ``gate3``, ``gate4``, ``gate_m``

Notes:
    - Sparse consequent layers support per-rule feature masks for MHTSK
      models and other partial-rule consequent architectures.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .gates import (
    GATE_FNS,  # noqa: F401
    _gate_activation,
    gate1,  # noqa: F401
    gate2,  # noqa: F401
    gate3,  # noqa: F401
    gate4,  # noqa: F401
    gate_m,  # noqa: F401
    resolve_gate_fn,
)
from .memberships import MembershipFunction
from .t_norms import TNormFn, resolve_t_norm


def _inv_softplus(value: float, eps: float | None = None) -> float:
    """Map a positive value to the unconstrained space used by softplus."""
    eps = torch.finfo(torch.get_default_dtype()).eps if eps is None else eps
    v = max(value - eps, eps)
    return math.log(math.expm1(v))


def _generate_en_frb(s: int, d: int) -> list[tuple[int, ...]]:
    """Generate Enhanced Fuzzy Rule Base (En-FRB) rules."""
    seen: set[tuple[int, ...]] = set()
    rules: list[tuple[int, ...]] = []

    for i in range(s):
        base = [i] * d
        base_t = tuple(base)
        if base_t not in seen:
            seen.add(base_t)
            rules.append(base_t)

        for j in range(d):
            minus = list(base)
            minus[j] = (i - 1) % s
            minus_t = tuple(minus)
            if minus_t not in seen:
                seen.add(minus_t)
                rules.append(minus_t)

            plus = list(base)
            plus[j] = (i + 1) % s
            plus_t = tuple(plus)
            if plus_t not in seen:
                seen.add(plus_t)
                rules.append(plus_t)

    return rules


class MembershipLayer(nn.Module):
    """Apply membership functions for each input feature.

    Evaluates each input variable against its sequence of
    :class:`~highfis.memberships.MembershipFunction` objects and returns
    a dictionary of per-variable membership tensors.

    Attributes:
        input_names: Ordered list of input feature names.
        n_inputs: Number of input features.
        mf_per_input: Number of membership functions per feature.
        input_mfs: ``nn.ModuleDict`` mapping feature names to their
            ``nn.ModuleList`` of membership functions.
    """

    def __init__(self, input_mfs: Mapping[str, Sequence[MembershipFunction]]) -> None:
        """Initialize membership layer with input-to-membership mapping."""
        super().__init__()
        if not input_mfs:
            raise ValueError("input_mfs must not be empty")

        self.input_names = list(input_mfs.keys())
        self.n_inputs = len(self.input_names)
        self.mf_per_input: list[int] = []

        modules: dict[str, nn.ModuleList] = {}
        for name in self.input_names:
            mfs = input_mfs[name]
            if not mfs:
                raise ValueError(f"input '{name}' must define at least one membership function")
            modules[name] = nn.ModuleList(mfs)
            self.mf_per_input.append(len(mfs))

        self.input_mfs = nn.ModuleDict(modules)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Compute membership outputs for each input variable.

        Args:
            x: Input tensor of shape ``(N, n_inputs)``.

        Returns:
            Dictionary mapping each input name to a membership tensor of
            shape ``(N, n_mfs_for_input)``.

        Raises:
            ValueError: If *x* is not 2-dimensional or has the wrong
                number of columns.
        """
        if x.ndim != 2:
            raise ValueError(f"expected x with 2 dims, got shape {tuple(x.shape)}")
        if x.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs, got {x.shape[1]}")

        outputs: dict[str, Tensor] = {}
        for i, name in enumerate(self.input_names):
            mfs = cast(nn.ModuleList, self.input_mfs[name])
            values = [mf(x[:, i]) for mf in mfs]
            outputs[name] = torch.stack(values, dim=-1)
        return outputs


class RuleLayer(nn.Module):
    """Compute firing strengths from membership degrees.

    Generates a rule base from the specified strategy and aggregates
    per-input membership degrees into scalar firing strengths using a
    configurable T-norm.

    Supported rule-base strategies (``rule_base`` parameter):

    - ``"cartesian"`` / ``"fuco"`` — full combinatorial rule base.
    - ``"coco"`` — same-index rule base; requires identical MF counts
      across all inputs.
    - ``"en"`` — enhanced FRB; requires identical MF counts across all
      inputs.
    - ``"custom"`` — user-supplied rule index sequences.

    Attributes:
        rules: List of rule index tuples, one per rule.
        n_rules: Number of rules in the rule base.
    """

    def __init__(
        self,
        input_names: list[str],
        mf_per_input: list[int],
        rules: Sequence[Sequence[int]] | None = None,
        rule_base: str = "cartesian",
        t_norm: str = "prod",
        t_norm_fn: TNormFn | None = None,
    ) -> None:
        """Initialize rule generation and firing-strength aggregation strategy."""
        super().__init__()
        if not input_names:
            raise ValueError("input_names must not be empty")
        if len(input_names) != len(mf_per_input):
            raise ValueError("input_names and mf_per_input must have the same length")

        rule_base = str(rule_base).lower()
        if rule_base in {"fuco", "fuco-frb"}:
            rule_base = "cartesian"
        if rule_base not in {"cartesian", "custom", "coco", "en"}:
            raise ValueError("rule_base must be 'cartesian', 'custom', 'coco', or 'en'")

        self.input_names = input_names
        self.mf_per_input = mf_per_input
        self.n_inputs = len(input_names)
        self.t_norm_fn = t_norm_fn
        self._resolved_t_norm = resolve_t_norm(t_norm)

        if rules is None and rule_base == "cartesian":
            self.rules = [tuple(r) for r in product(*[range(n) for n in mf_per_input])]
        elif rules is None and rule_base == "coco":
            if len(set(mf_per_input)) != 1:
                raise ValueError(
                    f"CoCo rule base requires all inputs to have the same number of MFs, got {mf_per_input}"
                )
            s = mf_per_input[0]
            self.rules = [tuple(i for _ in range(self.n_inputs)) for i in range(s)]
        elif rules is None and rule_base == "en":
            if len(set(mf_per_input)) != 1:
                raise ValueError(
                    f"En-FRB rule base requires all inputs to have the same number of MFs, got {mf_per_input}"
                )
            self.rules = _generate_en_frb(mf_per_input[0], self.n_inputs)
        else:
            if rules is None:
                raise ValueError("rules must be provided when rule_base='custom'")
            validated: list[tuple[int, ...]] = []
            for i, rule in enumerate(rules):
                if len(rule) != self.n_inputs:
                    raise ValueError(f"rule at index {i} has size {len(rule)} but expected {self.n_inputs}")
                normalized: list[int] = []
                for input_idx, mf_idx in enumerate(rule):
                    max_mf = self.mf_per_input[input_idx]
                    if not 0 <= int(mf_idx) < max_mf:
                        raise ValueError(f"rule index {mf_idx} out of bounds for input {input_idx} with {max_mf} mfs")
                    normalized.append(int(mf_idx))
                validated.append(tuple(normalized))
            if not validated:
                raise ValueError("rules must not be empty")
            self.rules = validated

        self.n_rules = len(self.rules)
        self._register_vectorized_indices()

    def _register_vectorized_indices(self) -> None:
        """Convert per-input rule MF indices into absolute flat indices."""
        offsets: list[int] = []
        current = 0
        for n_mfs in self.mf_per_input:
            offsets.append(current)
            current += n_mfs

        abs_rules = [
            [rule[input_idx] + offsets[input_idx] for input_idx in range(self.n_inputs)] for rule in self.rules
        ]

        self.register_buffer("rule_indices", torch.tensor(abs_rules, dtype=torch.long))

    def _apply_t_norm(self, terms: Tensor) -> Tensor:
        dim = terms.ndim - 1
        if self.t_norm_fn is not None:
            try:
                return self.t_norm_fn(terms, dim=dim)
            except TypeError:
                return self.t_norm_fn(terms)
        return self._resolved_t_norm(terms, dim=dim)

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Compute rule firing strengths from membership outputs.

        Args:
            membership_outputs: Dictionary returned by
                :class:`MembershipLayer`, mapping each input name to a
                membership tensor of shape ``(N, n_mfs)``.

        Returns:
            Firing-strength tensor of shape ``(N, n_rules)``.

        Raises:
            KeyError: If a required input name is missing from
                *membership_outputs*.
        """
        mu_list = []
        for name in self.input_names:
            if name not in membership_outputs:
                raise KeyError(f"missing membership output for input '{name}'")
            mu_list.append(membership_outputs[name])

        mu_flat = torch.cat(mu_list, dim=1)
        batch_size = mu_flat.shape[0]
        rule_indices = cast(Tensor, self.rule_indices)
        indices = rule_indices.unsqueeze(0).expand(batch_size, -1, -1)
        terms = mu_flat.gather(1, indices.reshape(batch_size, -1)).reshape(batch_size, self.n_rules, self.n_inputs)
        return self._apply_t_norm(terms)


class AdaSoftminRuleLayer(RuleLayer):
    """Compute adaptive Ada-softmin firing strengths for each rule."""

    def __init__(
        self,
        input_names: list[str],
        mf_per_input: list[int],
        rules: Sequence[Sequence[int]] | None = None,
        rule_base: str = "cartesian",
        eps: float | None = None,
    ) -> None:
        """Initialize Ada-softmin rule layer."""
        self.eps = torch.finfo(torch.get_default_dtype()).eps if eps is None else float(eps)
        super().__init__(input_names, mf_per_input, rules=rules, rule_base=rule_base, t_norm="prod", t_norm_fn=None)

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Compute Ada-softmin rule strengths from membership outputs."""
        mu_list = []
        for name in self.input_names:
            if name not in membership_outputs:
                raise KeyError(f"missing membership output for input '{name}'")
            mu_list.append(membership_outputs[name])

        mu_flat = torch.cat(mu_list, dim=1)
        batch_size = mu_flat.shape[0]
        rule_indices = cast(Tensor, self.rule_indices)
        indices = rule_indices.unsqueeze(0).expand(batch_size, -1, -1)
        terms = mu_flat.gather(1, indices.reshape(batch_size, -1)).reshape(batch_size, self.n_rules, self.n_inputs)

        mu = terms.clamp(min=self.eps, max=1.0 - self.eps)
        min_mu = mu.min(dim=-1).values
        q = torch.ceil(690.0 / torch.log(min_mu))
        q = q.clamp(min=-1000.0, max=-1.0)

        # Stable computation in log-space
        log_mu = torch.log(mu)
        log_terms = q.unsqueeze(-1) * log_mu
        max_log_terms = log_terms.amax(dim=-1, keepdim=True)
        log_sum = max_log_terms + torch.log(torch.exp(log_terms - max_log_terms).sum(dim=-1, keepdim=True))
        log_avg = log_sum - math.log(self.n_inputs)
        log_w = log_avg.squeeze(-1) / q
        return torch.exp(log_w)


class ADPSoftminRuleLayer(RuleLayer):
    """Compute adaptive ADP-softmin firing strengths for each rule."""

    def __init__(
        self,
        input_names: list[str],
        mf_per_input: list[int],
        rules: Sequence[Sequence[int]] | None = None,
        rule_base: str = "cartesian",
        kappa: float = 690.0,
        xi: float = 730.0,
        eps: float | None = None,
    ) -> None:
        """Initialize ADP-softmin rule layer."""
        self.eps = torch.finfo(torch.get_default_dtype()).eps if eps is None else float(eps)
        self.kappa = float(kappa)
        self.xi = float(xi)
        super().__init__(input_names, mf_per_input, rules=rules, rule_base=rule_base, t_norm="prod", t_norm_fn=None)

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Compute ADP-softmin rule strengths from membership outputs."""
        mu_list = []
        for name in self.input_names:
            if name not in membership_outputs:
                raise KeyError(f"missing membership output for input '{name}'")
            mu_list.append(membership_outputs[name])

        mu_flat = torch.cat(mu_list, dim=1)
        batch_size = mu_flat.shape[0]
        rule_indices = cast(Tensor, self.rule_indices)
        indices = rule_indices.unsqueeze(0).expand(batch_size, -1, -1)
        terms = mu_flat.gather(1, indices.reshape(batch_size, -1)).reshape(batch_size, self.n_rules, self.n_inputs)

        mu = terms.clamp(min=self.eps, max=1.0 - self.eps)
        min_mu = mu.min(dim=-1).values
        max_mu = mu.max(dim=-1).values

        ln_D = math.log(float(self.n_inputs))
        neg_ln_under = -torch.log(min_mu)
        neg_ln_bar = -torch.log(max_mu)

        denom = (self.kappa - ln_D) + self.xi
        log_eta = (self.xi * neg_ln_under + (self.kappa - ln_D) * neg_ln_bar) / denom
        eta = torch.exp(log_eta)

        q1 = (self.kappa - ln_D) / torch.log(eta * min_mu)
        q2 = (-self.xi) / torch.log(eta * max_mu)
        q = torch.maximum(q1, q2)
        q = torch.ceil(q).clamp(min=-1000.0, max=-1.0)

        log_mu = torch.log(mu)
        log_terms = q.unsqueeze(-1) * (torch.log(eta).unsqueeze(-1) + log_mu)
        max_log_terms = log_terms.amax(dim=-1, keepdim=True)
        log_sum = max_log_terms + torch.log(torch.exp(log_terms - max_log_terms).sum(dim=-1, keepdim=True))
        log_avg = log_sum - math.log(self.n_inputs)
        log_w = log_avg.squeeze(-1) / q
        return torch.exp(log_w)


class DGALETSKRuleLayer(RuleLayer):
    """Compute adaptive Ln-Exp softmin firing strengths with antecedent feature gates."""

    def __init__(
        self,
        input_names: list[str],
        mf_per_input: list[int],
        rules: Sequence[Sequence[int]] | None = None,
        rule_base: str = "cartesian",
        alpha_init: float = 1.0,
        eps: float | None = None,
    ) -> None:
        """Initialize DGALETSK rule layer."""
        if alpha_init <= 0.0:
            raise ValueError("alpha_init must be > 0")
        self.eps = torch.finfo(torch.get_default_dtype()).eps if eps is None else float(eps)
        super().__init__(input_names, mf_per_input, rules=rules, rule_base=rule_base, t_norm="prod", t_norm_fn=None)
        self.raw_alpha = nn.Parameter(torch.full((1,), _inv_softplus(alpha_init, self.eps)))
        self.lambda_gates = nn.Parameter(torch.zeros(self.n_inputs))
        nn.init.uniform_(self.lambda_gates, -0.1, 0.1)

    @property
    def alpha(self) -> Tensor:
        """Return positive adaptive alpha parameter."""
        return F.softplus(self.raw_alpha) + self.eps

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Compute adaptive Ln-Exp rule strengths from membership outputs."""
        mu_list = []
        for name in self.input_names:
            if name not in membership_outputs:
                raise KeyError(f"missing membership output for input '{name}'")
            mu_list.append(membership_outputs[name])

        mu_flat = torch.cat(mu_list, dim=1)
        batch_size = mu_flat.shape[0]
        rule_indices = cast(Tensor, self.rule_indices)
        indices = rule_indices.unsqueeze(0).expand(batch_size, -1, -1)
        terms = mu_flat.gather(1, indices.reshape(batch_size, -1)).reshape(batch_size, self.n_rules, self.n_inputs)

        mu = terms.clamp(min=self.eps, max=1.0 - self.eps)
        feature_gates = _gate_activation(self.lambda_gates)  # (n_inputs,) — broadcast over batch and rules
        mu = mu * feature_gates

        alpha = self.alpha.view(1, 1, 1)
        log_terms = -alpha * mu
        max_log_terms = log_terms.amax(dim=-1, keepdim=True)
        log_sum = max_log_terms + torch.log(torch.exp(log_terms - max_log_terms).sum(dim=-1, keepdim=True))
        return (-log_sum / alpha).squeeze(-1)


class DGTSKRuleLayer(RuleLayer):
    """Compute DG-TSK antecedent strengths with learned feature gates."""

    def __init__(
        self,
        input_names: list[str],
        mf_per_input: list[int],
        rules: Sequence[Sequence[int]] | None = None,
        rule_base: str = "coco",
        gate_fea: str | Callable[[Tensor], Tensor] | None = "gate_m",
        eps: float | None = None,
    ) -> None:
        """Initialize DGTSK rule layer."""
        self.eps = torch.finfo(torch.get_default_dtype()).eps if eps is None else float(eps)
        self.gate_fn = resolve_gate_fn(gate_fea)
        super().__init__(input_names, mf_per_input, rules=rules, rule_base=rule_base, t_norm="prod", t_norm_fn=None)
        self.lambda_gates = nn.Parameter(torch.zeros(self.n_inputs))
        nn.init.uniform_(self.lambda_gates, -0.1, 0.1)

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Compute DGTSK rule strengths from membership outputs."""
        mu_list = []
        for name in self.input_names:
            if name not in membership_outputs:
                raise KeyError(f"missing membership output for input '{name}'")
            mu_list.append(membership_outputs[name])

        mu_flat = torch.cat(mu_list, dim=1)
        batch_size = mu_flat.shape[0]
        rule_indices = cast(Tensor, self.rule_indices)
        indices = rule_indices.unsqueeze(0).expand(batch_size, -1, -1)
        terms = mu_flat.gather(1, indices.reshape(batch_size, -1)).reshape(batch_size, self.n_rules, self.n_inputs)

        mu = terms.clamp(min=self.eps, max=1.0 - self.eps)
        feature_gates = self.gate_fn(self.lambda_gates).unsqueeze(0)
        mu = mu * feature_gates

        return self._apply_t_norm(mu)


class AdaptiveDombiRuleLayer(RuleLayer):
    """Compute adaptive Dombi firing strengths with per-rule lambda parameters."""

    def __init__(
        self,
        input_names: list[str],
        mf_per_input: list[int],
        rules: Sequence[Sequence[int]] | None = None,
        rule_base: str = "coco",
        lambda_init: float = 1.0,
        eps: float | None = None,
    ) -> None:
        """Initialize adaptive Dombi rule layer."""
        if lambda_init <= 0.0:
            raise ValueError("lambda_init must be > 0")
        self.eps = torch.finfo(torch.get_default_dtype()).eps if eps is None else float(eps)
        super().__init__(input_names, mf_per_input, rules=rules, rule_base=rule_base, t_norm="prod", t_norm_fn=None)
        self.raw_lambdas = nn.Parameter(torch.full((self.n_rules,), _inv_softplus(lambda_init, self.eps)))

    @property
    def lambdas(self) -> Tensor:
        """Return strictly positive per-rule lambda values."""
        return F.softplus(self.raw_lambdas) + self.eps

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Compute adaptive Dombi firing strengths for each rule."""
        mu_list = []
        for name in self.input_names:
            if name not in membership_outputs:
                raise KeyError(f"missing membership output for input '{name}'")
            mu_list.append(membership_outputs[name])

        mu_flat = torch.cat(mu_list, dim=1)
        batch_size = mu_flat.shape[0]
        rule_indices = cast(Tensor, self.rule_indices)
        indices = rule_indices.unsqueeze(0).expand(batch_size, -1, -1)
        terms_tensor = mu_flat.gather(1, indices.reshape(batch_size, -1)).reshape(
            batch_size, self.n_rules, self.n_inputs
        )
        mu = terms_tensor.clamp(min=self.eps, max=1.0 - self.eps)
        ratio = (1.0 - mu) / mu
        lambdas = self.lambdas.view(1, self.n_rules, 1)
        powered = ratio.pow(lambdas)
        sum_ratio = powered.sum(dim=-1)
        return (1.0 + sum_ratio).pow(-1.0 / lambdas.squeeze(-1))


class ClassificationConsequentLayer(nn.Module):
    """Linear TSK consequent layer for classification logits."""

    def __init__(self, n_rules: int, n_inputs: int, n_classes: int) -> None:
        """Initialize consequent parameters for classification logits."""
        super().__init__()
        if n_rules <= 0 or n_inputs <= 0 or n_classes <= 0:
            raise ValueError("n_rules, n_inputs and n_classes must be positive")
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.weight = nn.Parameter(torch.empty(n_rules, n_classes, n_inputs))
        self.bias = nn.Parameter(torch.empty(n_rules, n_classes))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, norm_w: Tensor) -> Tensor:
        """Compute class logits from inputs and normalized rule strengths."""
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"expected x shape (batch, {self.n_inputs}), got {tuple(x.shape)}")
        if norm_w.ndim != 2 or norm_w.shape[1] != self.n_rules:
            raise ValueError(f"expected norm_w shape (batch, {self.n_rules}), got {tuple(norm_w.shape)}")

        f = torch.einsum("bd,rkd->brk", x, self.weight) + self.bias.unsqueeze(0)
        return torch.einsum("br,brk->bk", norm_w, f)


class SparseClassificationConsequentLayer(nn.Module):
    """Sparse linear TSK consequent layer for classification logits."""

    rule_feature_mask: Tensor

    def __init__(self, n_rules: int, n_inputs: int, n_classes: int, rule_feature_mask: Tensor) -> None:
        """Initialize sparse consequent parameters for classification."""
        super().__init__()
        if n_rules <= 0 or n_inputs <= 0 or n_classes <= 0:
            raise ValueError("n_rules, n_inputs and n_classes must be positive")
        if rule_feature_mask.shape != (n_rules, n_inputs):
            raise ValueError("rule_feature_mask must have shape (n_rules, n_inputs)")

        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.register_buffer("rule_feature_mask", rule_feature_mask.to(torch.bool))
        self.weight = nn.Parameter(torch.empty(n_rules, n_classes, n_inputs))
        self.bias = nn.Parameter(torch.empty(n_rules, n_classes))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, norm_w: Tensor) -> Tensor:
        """Compute class logits from sparse inputs and normalized rule strengths."""
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"expected x shape (batch, {self.n_inputs}), got {tuple(x.shape)}")
        if norm_w.ndim != 2 or norm_w.shape[1] != self.n_rules:
            raise ValueError(f"expected norm_w shape (batch, {self.n_rules}), got {tuple(norm_w.shape)}")

        rule_feature_mask = self.rule_feature_mask
        gated_weight = self.weight * rule_feature_mask.unsqueeze(1)
        f = torch.einsum("bd,rkd->brk", x, gated_weight) + self.bias.unsqueeze(0)
        return torch.einsum("br,brk->bk", norm_w, f)


class SparseRegressionConsequentLayer(nn.Module):
    """Sparse linear TSK consequent layer for scalar regression output."""

    rule_feature_mask: Tensor

    def __init__(self, n_rules: int, n_inputs: int, rule_feature_mask: Tensor) -> None:
        """Initialize sparse consequent parameters for regression."""
        super().__init__()
        if n_rules <= 0 or n_inputs <= 0:
            raise ValueError("n_rules and n_inputs must be positive")
        if rule_feature_mask.shape != (n_rules, n_inputs):
            raise ValueError("rule_feature_mask must have shape (n_rules, n_inputs)")

        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.register_buffer("rule_feature_mask", rule_feature_mask.to(torch.bool))
        self.weight = nn.Parameter(torch.empty(n_rules, n_inputs))
        self.bias = nn.Parameter(torch.empty(n_rules))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, norm_w: Tensor) -> Tensor:
        """Compute scalar regression output from sparse inputs and normalized rule strengths."""
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"expected x shape (batch, {self.n_inputs}), got {tuple(x.shape)}")
        if norm_w.ndim != 2 or norm_w.shape[1] != self.n_rules:
            raise ValueError(f"expected norm_w shape (batch, {self.n_rules}), got {tuple(norm_w.shape)}")

        rule_feature_mask = self.rule_feature_mask
        gated_weight = self.weight * rule_feature_mask
        f = torch.einsum("bd,rd->br", x, gated_weight) + self.bias.unsqueeze(0)
        return torch.einsum("br,br->b", norm_w, f).unsqueeze(1)


class GatedClassificationConsequentLayer(nn.Module):
    r"""Gated TSK consequent layer for classification logits.

    Supports three training modes to match the FSRE-AdaTSK paper protocol:

    * ``"fs"``  — only feature gates :math:`M(\\lambda_d)` are active
      (Phase 1, feature selection, eq. 21).
    * ``"re"``  — only rule gates :math:`M(\\theta_r)` are active
      (Phase 2, rule extraction, eq. 22).
    * ``"finetune"`` — no gates; plain linear TSK consequent (Phase 3,
      eq. 5).
    * ``"both"`` (default) — both gate families applied simultaneously.

    When ``shared_lambda=True`` the feature gate vector has shape
    ``(n_inputs,)`` and is shared across all rules (FSRE-AdaTSK, eq. 21).
    When ``shared_lambda=False`` (default) each rule has its own
    ``(n_inputs,)`` gate vector, stored as ``(n_rules, n_inputs)``
    (DG-ALETSK).
    """

    mode: str

    def __init__(
        self,
        n_rules: int,
        n_inputs: int,
        n_classes: int,
        gate_fn: str | Callable[[Tensor], Tensor] | None = None,
        shared_lambda: bool = False,
    ) -> None:
        """Initialize gated consequent parameters for classification logits."""
        super().__init__()
        if n_rules <= 0 or n_inputs <= 0 or n_classes <= 0:
            raise ValueError("n_rules, n_inputs and n_classes must be positive")
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.gate_fn = resolve_gate_fn(gate_fn)
        self.shared_lambda = shared_lambda
        self.mode = "both"
        self.weight = nn.Parameter(torch.empty(n_rules, n_classes, n_inputs))
        self.bias = nn.Parameter(torch.empty(n_rules, n_classes))
        lambda_shape = (n_inputs,) if shared_lambda else (n_rules, n_inputs)
        self.lambda_gates = nn.Parameter(torch.zeros(lambda_shape))
        self.theta_gates = nn.Parameter(torch.zeros(n_rules))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.bias)
        nn.init.uniform_(self.lambda_gates, -0.1, 0.1)
        nn.init.uniform_(self.theta_gates, -0.1, 0.1)

    def forward(self, x: Tensor, norm_w: Tensor) -> Tensor:
        """Compute gated class logits from inputs and normalized rule strengths."""
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"expected x shape (batch, {self.n_inputs}), got {tuple(x.shape)}")
        if norm_w.ndim != 2 or norm_w.shape[1] != self.n_rules:
            raise ValueError(f"expected norm_w shape (batch, {self.n_rules}), got {tuple(norm_w.shape)}")

        if self.mode == "fs":
            # Phase 1: feature gates only (eq. 21) — M(λ_d) on weights, no rule gate
            feature_gates = self.gate_fn(self.lambda_gates)  # (n_inputs,) or (n_rules, n_inputs)
            if not self.shared_lambda:
                feature_gates = feature_gates.unsqueeze(1)  # (n_rules, 1, n_inputs)
            gated_weight = self.weight * feature_gates
            f = torch.einsum("bd,rkd->brk", x, gated_weight) + self.bias.unsqueeze(0)
            return torch.einsum("br,brk->bk", norm_w, f)
        elif self.mode == "re":
            # Phase 2: rule gates only (eq. 22) — M(θ_r) on full consequent, no feature gate
            rule_gates = self.gate_fn(self.theta_gates).view(1, self.n_rules, 1)
            f = torch.einsum("bd,rkd->brk", x, self.weight) + self.bias.unsqueeze(0)
            f = f * rule_gates
            return torch.einsum("br,brk->bk", norm_w, f)
        elif self.mode == "finetune":
            # Phase 3: no gates (eq. 5) — plain linear consequent
            f = torch.einsum("bd,rkd->brk", x, self.weight) + self.bias.unsqueeze(0)
            return torch.einsum("br,brk->bk", norm_w, f)
        else:
            # "both" (default): both gate families active — DG-ALETSK usage
            if self.shared_lambda:
                feature_gates: Tensor = self.gate_fn(self.lambda_gates)  # (n_inputs,)
            else:
                feature_gates = self.gate_fn(self.lambda_gates).unsqueeze(1)  # (n_rules, 1, n_inputs)
            rule_gates = self.gate_fn(self.theta_gates).view(1, self.n_rules, 1)
            gated_weight = self.weight * feature_gates
            f = torch.einsum("bd,rkd->brk", x, gated_weight) + self.bias.unsqueeze(0)
            f = f * rule_gates
            return torch.einsum("br,brk->bk", norm_w, f)


class GatedClassificationZeroOrderConsequentLayer(nn.Module):
    """Gated zero-order TSK consequent layer for classification logits."""

    def __init__(
        self,
        n_rules: int,
        n_inputs: int,
        n_classes: int,
        gate_fn: str | Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize zero-order gated consequent parameters for classification logits."""
        super().__init__()
        if n_rules <= 0 or n_inputs <= 0 or n_classes <= 0:
            raise ValueError("n_rules, n_inputs and n_classes must be positive")
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.gate_fn = resolve_gate_fn(gate_fn)
        self.bias = nn.Parameter(torch.empty(n_rules, n_classes))
        self.theta_gates = nn.Parameter(torch.zeros(n_rules))
        nn.init.zeros_(self.bias)
        nn.init.uniform_(self.theta_gates, -0.1, 0.1)

    def forward(self, x: Tensor, norm_w: Tensor) -> Tensor:
        """Compute gated class logits from normalized rule strengths."""
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"expected x shape (batch, {self.n_inputs}), got {tuple(x.shape)}")
        if norm_w.ndim != 2 or norm_w.shape[1] != self.n_rules:
            raise ValueError(f"expected norm_w shape (batch, {self.n_rules}), got {tuple(norm_w.shape)}")

        f = self.bias.unsqueeze(0)
        rule_gates = self.gate_fn(self.theta_gates).view(1, self.n_rules, 1)
        f = f * rule_gates
        return torch.einsum("br,brk->bk", norm_w, f)


class GatedRegressionZeroOrderConsequentLayer(nn.Module):
    """Gated zero-order TSK consequent layer for regression."""

    def __init__(self, n_rules: int, n_inputs: int, gate_fn: str | Callable[[Tensor], Tensor] | None = None) -> None:
        """Initialize zero-order gated consequent parameters for regression."""
        super().__init__()
        if n_rules <= 0 or n_inputs <= 0:
            raise ValueError("n_rules and n_inputs must be positive")
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.gate_fn = resolve_gate_fn(gate_fn)
        self.bias = nn.Parameter(torch.empty(n_rules))
        self.theta_gates = nn.Parameter(torch.zeros(n_rules))
        nn.init.zeros_(self.bias)
        nn.init.uniform_(self.theta_gates, -0.1, 0.1)

    def forward(self, x: Tensor, norm_w: Tensor) -> Tensor:
        """Compute gated regression output from normalized rule strengths."""
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"expected x shape (batch, {self.n_inputs}), got {tuple(x.shape)}")
        if norm_w.ndim != 2 or norm_w.shape[1] != self.n_rules:
            raise ValueError(f"expected norm_w shape (batch, {self.n_rules}), got {tuple(norm_w.shape)}")

        f = self.bias.unsqueeze(0)
        rule_gates = self.gate_fn(self.theta_gates).view(1, self.n_rules)
        f = f * rule_gates
        return torch.einsum("br,br->b", norm_w, f).unsqueeze(1)


class RegressionConsequentLayer(nn.Module):
    """Linear TSK consequent layer for scalar regression output."""

    def __init__(self, n_rules: int, n_inputs: int) -> None:
        """Initialize consequent parameters for regression."""
        super().__init__()
        if n_rules <= 0 or n_inputs <= 0:
            raise ValueError("n_rules and n_inputs must be positive")
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.weight = nn.Parameter(torch.empty(n_rules, n_inputs))
        self.bias = nn.Parameter(torch.empty(n_rules))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, norm_w: Tensor) -> Tensor:
        """Compute scalar regression output from inputs and normalized rule strengths."""
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"expected x shape (batch, {self.n_inputs}), got {tuple(x.shape)}")
        if norm_w.ndim != 2 or norm_w.shape[1] != self.n_rules:
            raise ValueError(f"expected norm_w shape (batch, {self.n_rules}), got {tuple(norm_w.shape)}")

        # f_r(x) = x @ W_r^T + b_r  → (batch, n_rules)
        f = torch.einsum("bd,rd->br", x, self.weight) + self.bias.unsqueeze(0)
        # output = sum_r(norm_w_r * f_r)  → (batch, 1)
        return torch.einsum("br,br->b", norm_w, f).unsqueeze(1)


class GatedRegressionConsequentLayer(nn.Module):
    """Gated TSK consequent layer for scalar regression output.

    Supports the same ``mode`` / ``shared_lambda`` protocol as
    :class:`GatedClassificationConsequentLayer`.
    """

    mode: str

    def __init__(
        self,
        n_rules: int,
        n_inputs: int,
        gate_fn: str | Callable[[Tensor], Tensor] | None = None,
        shared_lambda: bool = False,
    ) -> None:
        """Initialize gated consequent parameters for regression."""
        super().__init__()
        if n_rules <= 0 or n_inputs <= 0:
            raise ValueError("n_rules and n_inputs must be positive")
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.gate_fn = resolve_gate_fn(gate_fn)
        self.shared_lambda = shared_lambda
        self.mode = "both"
        self.weight = nn.Parameter(torch.empty(n_rules, n_inputs))
        self.bias = nn.Parameter(torch.empty(n_rules))
        lambda_shape = (n_inputs,) if shared_lambda else (n_rules, n_inputs)
        self.lambda_gates = nn.Parameter(torch.zeros(lambda_shape))
        self.theta_gates = nn.Parameter(torch.zeros(n_rules))
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.bias)
        nn.init.uniform_(self.lambda_gates, -0.1, 0.1)
        nn.init.uniform_(self.theta_gates, -0.1, 0.1)

    def forward(self, x: Tensor, norm_w: Tensor) -> Tensor:
        """Compute gated regression output from inputs and normalized rule strengths."""
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"expected x shape (batch, {self.n_inputs}), got {tuple(x.shape)}")
        if norm_w.ndim != 2 or norm_w.shape[1] != self.n_rules:
            raise ValueError(f"expected norm_w shape (batch, {self.n_rules}), got {tuple(norm_w.shape)}")

        if self.mode == "fs":
            # Phase 1: feature gates only (eq. 21)
            feature_gates = self.gate_fn(self.lambda_gates)  # (n_inputs,) or (n_rules, n_inputs)
            gated_weight = self.weight * feature_gates
            f = torch.einsum("bd,rd->br", x, gated_weight) + self.bias.unsqueeze(0)
            return torch.einsum("br,br->b", norm_w, f).unsqueeze(1)
        elif self.mode == "re":
            # Phase 2: rule gates only (eq. 22)
            rule_gates = self.gate_fn(self.theta_gates).view(1, self.n_rules)
            f = torch.einsum("bd,rd->br", x, self.weight) + self.bias.unsqueeze(0)
            f = f * rule_gates
            return torch.einsum("br,br->b", norm_w, f).unsqueeze(1)
        elif self.mode == "finetune":
            # Phase 3: no gates (eq. 5)
            f = torch.einsum("bd,rd->br", x, self.weight) + self.bias.unsqueeze(0)
            return torch.einsum("br,br->b", norm_w, f).unsqueeze(1)
        else:
            # "both" (default): both gate families active
            feature_gates = self.gate_fn(self.lambda_gates)  # (n_inputs,) or (n_rules, n_inputs)
            rule_gates = self.gate_fn(self.theta_gates).view(1, self.n_rules)
            gated_weight = self.weight * feature_gates
            f = torch.einsum("bd,rd->br", x, gated_weight) + self.bias.unsqueeze(0)
            f = f * rule_gates
            return torch.einsum("br,br->b", norm_w, f).unsqueeze(1)


__all__: list[str] = [
    "AdaSoftminRuleLayer",
    "AdaptiveDombiRuleLayer",
    "ClassificationConsequentLayer",
    "DGALETSKRuleLayer",
    "DGTSKRuleLayer",
    "GatedClassificationConsequentLayer",
    "GatedClassificationZeroOrderConsequentLayer",
    "GatedRegressionConsequentLayer",
    "GatedRegressionZeroOrderConsequentLayer",
    "MembershipLayer",
    "RegressionConsequentLayer",
    "RuleLayer",
]
