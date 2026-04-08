from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import cast

import torch
from torch import Tensor, nn

from .memberships import MembershipFunction
from .t_norms import TNormFn, resolve_t_norm


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

    def _apply_t_norm(self, terms: Tensor) -> Tensor:
        if self.t_norm_fn is not None:
            return self.t_norm_fn(terms)
        return self._resolved_t_norm(terms)

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Compute rule firing strengths from membership outputs."""
        mu_list = []
        for name in self.input_names:
            if name not in membership_outputs:
                raise KeyError(f"missing membership output for input '{name}'")
            mu_list.append(membership_outputs[name])

        batch_size = mu_list[0].shape[0]
        outputs: list[Tensor] = []
        for rule in self.rules:
            terms = [mu_list[input_idx][:, mf_idx] for input_idx, mf_idx in enumerate(rule)]
            strength = self._apply_t_norm(torch.stack(terms, dim=1))
            outputs.append(strength)

        return torch.stack(outputs, dim=1).reshape(batch_size, self.n_rules)


class NormalizationLayer(nn.Module):
    """Normalize rule strengths so each sample sums to one."""

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize normalization layer epsilon."""
        super().__init__()
        self.eps = eps

    def forward(self, w: Tensor) -> Tensor:
        """Normalize firing strengths along the rule axis.

        Uses ``softmax(log(w))`` which is mathematically equivalent to
        ``w / sum(w)`` but numerically more stable in high dimensions thanks
        to the internal max-subtraction trick of :func:`torch.softmax`.
        """
        if w.ndim != 2:
            raise ValueError(f"expected w with 2 dims, got shape {tuple(w.shape)}")
        log_w = w.clamp(min=self.eps).log()
        return torch.softmax(log_w, dim=1)


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


__all__ = [
    "MembershipLayer",
    "RuleLayer",
    "NormalizationLayer",
    "ClassificationConsequentLayer",
]
