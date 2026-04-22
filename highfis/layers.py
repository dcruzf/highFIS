"""Antecedent and consequent layers for highFIS fuzzy models."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

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
    """Apply membership functions for each input feature."""

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
        """Compute membership outputs for each input variable."""
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
    """Compute firing strengths from membership degrees."""

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
        """Compute rule firing strengths from membership outputs."""
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


def gate1(u: Tensor) -> Tensor:
    """Compute the sigmoid gate activation."""
    return torch.sigmoid(u)


def gate2(u: Tensor) -> Tensor:
    """Compute the gate activation 1 - exp(-x^2)."""
    return 1.0 - torch.exp(-u.pow(2))


def gate3(u: Tensor) -> Tensor:
    """Compute the gate activation exp(-x^2)."""
    return torch.exp(-u.pow(2))


def gate4(u: Tensor) -> Tensor:
    """Compute the gate activation x * sqrt(exp(1 - x^2))."""
    return u * torch.sqrt(torch.exp(1.0 - u.pow(2)))


def gate_m(u: Tensor) -> Tensor:
    """Compute the gate activation x^2 * exp(1 - x^2)."""
    return u.pow(2) * torch.exp(1.0 - u.pow(2))


GATE_FNS: dict[str, Callable[[Tensor], Tensor]] = {
    "gate1": gate1,
    "gate2": gate2,
    "gate3": gate3,
    "gate4": gate4,
    "gate_m": gate_m,
}


def resolve_gate_fn(gate_fn: str | Callable[[Tensor], Tensor] | None) -> Callable[[Tensor], Tensor]:
    """Resolve a gate name or function to a callable gate function."""
    if gate_fn is None:
        return gate4
    if isinstance(gate_fn, str):
        try:
            return GATE_FNS[gate_fn]
        except KeyError as exc:
            raise ValueError(f"unsupported gate function '{gate_fn}'") from exc
    return gate_fn


def _gate_activation(u: Tensor) -> Tensor:
    """Default feature gate activation used by DG-ALETSK and related models."""
    return gate4(u)


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
        self.lambda_gates = nn.Parameter(torch.zeros(self.n_rules, self.n_inputs))
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
        feature_gates = _gate_activation(self.lambda_gates).unsqueeze(0)
        mu = mu * feature_gates

        alpha = self.alpha.view(1, 1, 1)
        log_terms = -alpha * mu
        max_log_terms = log_terms.amax(dim=-1, keepdim=True)
        log_sum = max_log_terms + torch.log(torch.exp(log_terms - max_log_terms).sum(dim=-1, keepdim=True))
        log_avg = log_sum - torch.log(torch.tensor(self.n_inputs, dtype=log_sum.dtype, device=log_sum.device))
        log_w = -log_avg.squeeze(-1) / alpha.squeeze(-1)
        return torch.exp(log_w)


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
        self.lambda_gates = nn.Parameter(torch.zeros(self.n_rules, self.n_inputs))
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


class GatedClassificationConsequentLayer(nn.Module):
    """Gated TSK consequent layer for classification logits."""

    def __init__(
        self,
        n_rules: int,
        n_inputs: int,
        n_classes: int,
        gate_fn: str | Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize gated consequent parameters for classification logits."""
        super().__init__()
        if n_rules <= 0 or n_inputs <= 0 or n_classes <= 0:
            raise ValueError("n_rules, n_inputs and n_classes must be positive")
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.gate_fn = resolve_gate_fn(gate_fn)
        self.weight = nn.Parameter(torch.empty(n_rules, n_classes, n_inputs))
        self.bias = nn.Parameter(torch.empty(n_rules, n_classes))
        self.lambda_gates = nn.Parameter(torch.zeros(n_rules, n_inputs))
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

        feature_gates = self.gate_fn(self.lambda_gates).unsqueeze(1)
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
    """Gated TSK consequent layer for scalar regression output."""

    def __init__(self, n_rules: int, n_inputs: int, gate_fn: str | Callable[[Tensor], Tensor] | None = None) -> None:
        """Initialize gated consequent parameters for regression."""
        super().__init__()
        if n_rules <= 0 or n_inputs <= 0:
            raise ValueError("n_rules and n_inputs must be positive")
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.gate_fn = resolve_gate_fn(gate_fn)
        self.weight = nn.Parameter(torch.empty(n_rules, n_inputs))
        self.bias = nn.Parameter(torch.empty(n_rules))
        self.lambda_gates = nn.Parameter(torch.zeros(n_rules, n_inputs))
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

        feature_gates = self.gate_fn(self.lambda_gates)
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
