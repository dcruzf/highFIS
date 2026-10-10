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
    ``gate1`` (:class:`~highfis.gates.SigmoidGate`),
    ``gate2`` (:class:`~highfis.gates.ExpGate` ``k=1``),
    ``gate3`` (:class:`~highfis.gates.InvExpGate`),
    ``gate4`` (:class:`~highfis.gates.SignedExpGate`),
    ``gate_m`` (:class:`~highfis.gates.MGate`)

Notes:
    - Sparse consequent layers support per-rule feature masks for MHTSK
      models and other partial-rule consequent architectures.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .gates import (
    GATE_FNS,  # noqa: F401
    BaseGate,
    gate1,  # noqa: F401
    gate2,  # noqa: F401
    gate3,  # noqa: F401
    gate4,  # noqa: F401
    gate_m,  # noqa: F401
    resolve_gate_fn,
)
from .memberships import (
    ADATSKGaussianMF,
    CompositeExponentialMF,
    ConstantMF,
    DimensionDependentGaussianMF,
    GaussianMF,
    GaussianPiMF,
    MembershipFunction,
    _inv_softplus,
)
from .t_norms import TNormFn, resolve_t_norm


def _kernel_gaussian(x: Tensor, mean: Tensor, raw_sigma: Tensor, consts: Mapping[str, Tensor]) -> Tensor:
    sigma = F.softplus(raw_sigma) + consts["eps"]
    z = (x - mean) / sigma
    return torch.exp(-0.5 * z.square())


def _kernel_adatsk_gaussian(x: Tensor, mean: Tensor, raw_sigma: Tensor, consts: Mapping[str, Tensor]) -> Tensor:
    sigma = F.softplus(raw_sigma) + consts["eps"]
    z = (x - mean) / sigma
    return torch.exp(-z.square())


def _kernel_dimension_dependent_gaussian(
    x: Tensor, mean: Tensor, raw_sigma: Tensor, consts: Mapping[str, Tensor]
) -> Tensor:
    eps = consts["eps"]
    sigma = F.softplus(raw_sigma) + eps
    denom = consts["scale"] + sigma.square() + eps
    return torch.exp(-(x - mean).square() / denom)


def _kernel_gaussian_pi(x: Tensor, mean: Tensor, raw_sigma: Tensor, consts: Mapping[str, Tensor]) -> Tensor:
    sigma = F.softplus(raw_sigma) + consts["eps"]
    z_sq = ((x - mean) / sigma).square()
    inner = 1.0 - torch.exp(-0.5 * z_sq)
    return torch.exp(-consts["k"] * inner)


def _kernel_composite_exponential(x: Tensor, mean: Tensor, raw_sigma: Tensor, consts: Mapping[str, Tensor]) -> Tensor:
    sigma = F.softplus(raw_sigma) + consts["eps"]
    exponent = -0.5 * ((x - mean) / sigma).square()
    return torch.pow(consts["k"], -1.0 + torch.exp(exponent))


def _kernel_constant(x: Tensor, mean: Tensor | None, raw_sigma: Tensor | None, consts: Mapping[str, Tensor]) -> Tensor:
    return consts["value"].expand_as(x).clone()


_MFKernel = Callable[[Tensor, "Tensor | None", "Tensor | None", Mapping[str, Tensor]], Tensor]

#: Vectorized fuzzification kernels, keyed by exact membership class.
#: Each kernel reproduces the corresponding ``forward`` formula elementwise
#: over a pre-gathered ``(N, total_mfs)`` input, with the trainable
#: parameters consolidated into flat ``(total_mfs,)`` tensors owned by the
#: layer. ``consts`` carries the per-MF non-trainable constants cached at
#: layer construction. The Gaussian-family kernels require the flat
#: parameter tensors; only ``ConstantMF`` (parameter-free) receives None.
_VECTORIZED_MF_KERNELS: dict[type, _MFKernel] = cast(
    "dict[type, _MFKernel]",
    {
        GaussianMF: _kernel_gaussian,
        ADATSKGaussianMF: _kernel_adatsk_gaussian,
        DimensionDependentGaussianMF: _kernel_dimension_dependent_gaussian,
        GaussianPiMF: _kernel_gaussian_pi,
        CompositeExponentialMF: _kernel_composite_exponential,
        ConstantMF: _kernel_constant,
    },
)


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


#: Name of the trainable location parameter of the vectorized classes that do not call it ``mean``.
_LOCATION_PARAMETER: dict[type, str] = {CompositeExponentialMF: "center"}


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
        # Flat MF index -> owning feature column, used by the vectorized
        # fast path to gather inputs in one op. Non-persistent so that
        # state_dict keys are unchanged.
        feat_idx = torch.repeat_interleave(torch.arange(self.n_inputs), torch.tensor(self.mf_per_input))
        self.register_buffer("_feat_idx", feat_idx, persistent=False)
        self._build_fast_path()
        self.register_load_state_dict_pre_hook(self._load_legacy_state_dict_hook)

    def _register_constant_sets(self, all_mfs: Sequence[nn.Module]) -> list[int]:
        """Detect one vectorizable class mixed with constant sets and index the two groups.

        A layer mixes them when a rule leaves a feature out (MHTSK). The constant columns
        are then filled in directly and the kernel runs on the others. Returns the
        positions of the sets that are not constant, empty when the layer is not mixed.
        """
        self._mixed = False
        kinds = {type(mf) for mf in all_mfs}
        if len(kinds) != 2 or ConstantMF not in kinds:
            return []
        main_idx = [i for i, mf in enumerate(all_mfs) if type(mf) is not ConstantMF]
        const_idx = [i for i, mf in enumerate(all_mfs) if type(mf) is ConstantMF]
        if type(all_mfs[main_idx[0]]) not in _VECTORIZED_MF_KERNELS:
            return []
        self._mixed = True
        self.register_buffer("_mixed_main_idx", torch.tensor(main_idx), persistent=False)
        self.register_buffer("_mixed_const_idx", torch.tensor(const_idx), persistent=False)
        self.register_buffer(
            "_mixed_const_value",
            torch.tensor([cast(Any, all_mfs[i]).value for i in const_idx]),
            persistent=False,
        )
        return main_idx

    def _build_fast_path(self) -> None:
        """Consolidate parameters for vectorized fuzzification when possible.

        The fast path applies when every membership function in the layer
        is an instance of the exact same class registered in
        ``_VECTORIZED_MF_KERNELS``. In that case the per-MF trainable
        scalars are consolidated into flat ``(total_mfs,)`` parameters
        owned by the layer (``_flat_mean``, ``_flat_raw_sigma``), and each
        module's scalar parameters are replaced by read-only views into
        the flat storage, so introspection (``get_mf_params``,
        ``inspect_params``) keeps working. Per-MF non-trainable constants
        (eps, scale, k, value) are frozen into non-persistent buffers.
        The plan is built once at construction: layers are rebuilt (not
        mutated) by the pruning routines, so it cannot go stale.
        """
        all_mfs = [mf for name in self.input_names for mf in cast(nn.ModuleList, self.input_mfs[name])]
        main_idx = self._register_constant_sets(all_mfs)
        flat_mfs = [all_mfs[i] for i in main_idx] if self._mixed else all_mfs
        mf_type = type(flat_mfs[0])
        kernel = _VECTORIZED_MF_KERNELS.get(mf_type)
        if kernel is None or not all(type(mf) is mf_type for mf in flat_mfs):
            self._flat_mfs: list[MembershipFunction] | None = None
            self._fast_kernel = None
            self._fast_const_names: tuple[str, ...] = ()
            self._flat_mean: nn.Parameter | None = None
            self._flat_raw_sigma: nn.Parameter | None = None
            return

        self._flat_mfs = cast("list[MembershipFunction]", flat_mfs)
        self._fast_kernel = kernel
        consts: dict[str, Tensor] = {"eps": torch.tensor([mf.eps for mf in flat_mfs])}
        if mf_type is DimensionDependentGaussianMF:
            consts["scale"] = torch.tensor([cast(Any, mf).scale for mf in flat_mfs])
        elif mf_type in (GaussianPiMF, CompositeExponentialMF):
            consts["k"] = torch.tensor([cast(Any, mf).k for mf in flat_mfs])
        elif mf_type is ConstantMF:
            consts["value"] = torch.tensor([cast(Any, mf).value for mf in flat_mfs])
        for cname, tensor in consts.items():
            self.register_buffer(f"_fast_const_{cname}", tensor, persistent=False)
        self._fast_const_names = tuple(consts)

        if mf_type is ConstantMF:
            self._flat_mean = None
            self._flat_raw_sigma = None
            return

        with torch.no_grad():
            location = _LOCATION_PARAMETER.get(mf_type, "mean")
            mean = torch.stack([cast(Tensor, getattr(mf, location)).detach() for mf in flat_mfs]).clone()
            raw_sigma = torch.stack([cast(Tensor, mf.raw_sigma).detach() for mf in flat_mfs]).clone()
        self._flat_mean = nn.Parameter(mean)
        self._flat_raw_sigma = nn.Parameter(raw_sigma)
        if not getattr(mf_type, "trainable_spread", True):
            self._flat_raw_sigma.requires_grad_(False)
        for i, mf in enumerate(flat_mfs):
            mf._parameters.pop(location, None)
            mf._parameters.pop("raw_sigma", None)
            mf.__dict__.pop(location, None)
            mf.__dict__.pop("raw_sigma", None)
            # Resolved lazily by MembershipFunction.__getattr__ so the
            # values stay current across optimizer steps and .to() moves.
            mf.__dict__["_vectorized_binding"] = {
                location: (self, "_flat_mean", i),
                "raw_sigma": (self, "_flat_raw_sigma", i),
            }

    def _load_legacy_state_dict_hook(
        self,
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """Map per-module scalar checkpoint keys onto the flat parameters.

        Checkpoints written before parameter consolidation store one
        ``input_mfs.<name>.<i>.mean``/``raw_sigma`` entry per membership
        function. When the layer runs in vectorized mode, gather them into
        ``_flat_mean``/``_flat_raw_sigma`` so old checkpoints keep loading.
        """
        if self._flat_mean is None or f"{prefix}_flat_mean" in state_dict:
            return
        location = _LOCATION_PARAMETER.get(type(cast("list[MembershipFunction]", self._flat_mfs)[0]), "mean")
        means: list[Tensor] = []
        raws: list[Tensor] = []
        owners = [
            (name, i)
            for name in self.input_names
            for i, mf in enumerate(cast(nn.ModuleList, self.input_mfs[name]))
            if not (self._mixed and type(mf) is ConstantMF)
        ]
        for name, i in owners:
            mean_key = f"{prefix}input_mfs.{name}.{i}.{location}"
            raw_key = f"{prefix}input_mfs.{name}.{i}.raw_sigma"
            if mean_key not in state_dict or raw_key not in state_dict:
                return
            means.append(state_dict[mean_key].reshape(()))
            raws.append(state_dict[raw_key].reshape(()))
        for name, i in owners:
            state_dict.pop(f"{prefix}input_mfs.{name}.{i}.{location}")
            state_dict.pop(f"{prefix}input_mfs.{name}.{i}.raw_sigma")
        state_dict[f"{prefix}_flat_mean"] = torch.stack(means)
        state_dict[f"{prefix}_flat_raw_sigma"] = torch.stack(raws)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Compute membership outputs for each input variable.

        When every membership function in the layer is an instance of the
        same vectorizable class (see ``_VECTORIZED_MF_KERNELS``), all
        degrees are computed in a single batched tensor operation instead
        of one module call per (feature, MF) pair. Heterogeneous or custom
        membership functions fall back to the per-module loop.

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

        if self._fast_kernel is not None:
            consts = {name: cast(Tensor, getattr(self, f"_fast_const_{name}")) for name in self._fast_const_names}
            feat_idx = cast(Tensor, self._feat_idx)
            if self._mixed:
                main_idx = cast(Tensor, self._mixed_main_idx)
                mu_main = self._fast_kernel(
                    x.index_select(1, feat_idx[main_idx]), self._flat_mean, self._flat_raw_sigma, consts
                )
                mu_flat = mu_main.new_empty((x.shape[0], feat_idx.shape[0]))
                const_value = cast(Tensor, self._mixed_const_value).to(mu_main.dtype)
                mu_flat[:, cast(Tensor, self._mixed_const_idx)] = const_value
                mu_flat[:, main_idx] = mu_main
            else:
                mu_flat = self._fast_kernel(x.index_select(1, feat_idx), self._flat_mean, self._flat_raw_sigma, consts)
            chunks = torch.split(mu_flat, self.mf_per_input, dim=1)
            return dict(zip(self.input_names, chunks, strict=False))

        outputs: dict[str, Tensor] = {}
        for i, name in enumerate(self.input_names):
            mfs = cast(nn.ModuleList, self.input_mfs[name])
            values = [mf(x[:, i]) for mf in mfs]
            outputs[name] = torch.stack(values, dim=-1)
        return outputs


def _validate_custom_rules(
    rules: Sequence[Sequence[int]],
    mf_per_input: list[int],
    n_inputs: int,
) -> list[tuple[int, ...]]:
    validated: list[tuple[int, ...]] = []
    for i, rule in enumerate(rules):
        if len(rule) != n_inputs:
            raise ValueError(f"rule at index {i} has size {len(rule)} but expected {n_inputs}")
        normalized: list[int] = []
        for input_idx, mf_idx in enumerate(rule):
            max_mf = mf_per_input[input_idx]
            if not 0 <= int(mf_idx) < max_mf:
                raise ValueError(f"rule index {mf_idx} out of bounds for input {input_idx} with {max_mf} mfs")
            normalized.append(int(mf_idx))
        validated.append(tuple(normalized))
    if not validated:
        raise ValueError("rules must not be empty")
    return validated


# A Cartesian rule base has one rule per combination of fuzzy sets, so it grows as
# ``n_mfs ** n_features``. Above this size it is refused instead of exhausting the memory.
MAX_CARTESIAN_RULES = 1_000_000


def _generate_or_validate_rules(
    input_names: list[str],
    mf_per_input: list[int],
    rules: Sequence[Sequence[int]] | None,
    rule_base: str,
) -> list[tuple[int, ...]]:
    n_inputs = len(input_names)
    if rules is None and rule_base == "cartesian":
        n_rules = math.prod(mf_per_input)
        if n_rules > MAX_CARTESIAN_RULES:
            raise ValueError(
                f"a Cartesian rule base over {n_inputs} features with {max(mf_per_input)} fuzzy sets each would have "
                f"{n_rules:,} rules; the limit is {MAX_CARTESIAN_RULES:,}. Pass rule_base='coco' for one rule per "
                "fuzzy set, or reduce the number of features or of sets."
            )
        return [tuple(r) for r in product(*[range(n) for n in mf_per_input])]

    if rules is None and rule_base == "coco":
        if len(set(mf_per_input)) != 1:
            raise ValueError(f"CoCo rule base requires all inputs to have the same number of MFs, got {mf_per_input}")
        s = mf_per_input[0]
        return [tuple(i for _ in range(n_inputs)) for i in range(s)]

    if rules is None and rule_base == "en":
        if len(set(mf_per_input)) != 1:
            raise ValueError(f"En-FRB rule base requires all inputs to have the same number of MFs, got {mf_per_input}")
        return _generate_en_frb(mf_per_input[0], n_inputs)

    if rules is None:
        raise ValueError("rules must be provided when rule_base='custom'")

    return _validate_custom_rules(rules, mf_per_input, n_inputs)


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
        t_norm: str | TNormFn = "prod",
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
        self._resolved_t_norm = resolve_t_norm(t_norm)

        self.rules = _generate_or_validate_rules(input_names, mf_per_input, rules, rule_base)
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
        try:
            return self._resolved_t_norm(terms, dim=dim)
        except TypeError:
            return self._resolved_t_norm(terms)

    def _gather_terms(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Gather per-rule membership terms from membership layer outputs."""
        mu_list = []
        for name in self.input_names:
            if name not in membership_outputs:
                raise KeyError(f"missing membership output for input '{name}'")
            mu_list.append(membership_outputs[name])
        mu_flat = torch.cat(mu_list, dim=1)
        batch_size = mu_flat.shape[0]
        rule_indices = cast(Tensor, self.rule_indices)
        indices = rule_indices.unsqueeze(0).expand(batch_size, -1, -1)
        return mu_flat.gather(1, indices.reshape(batch_size, -1)).reshape(batch_size, self.n_rules, self.n_inputs)

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
        terms = self._gather_terms(membership_outputs)
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
        super().__init__(input_names, mf_per_input, rules=rules, rule_base=rule_base, t_norm="prod")

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Compute Ada-softmin rule strengths from membership outputs."""
        mu = self._gather_terms(membership_outputs).clamp(min=self.eps, max=1.0 - self.eps)
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


def _clamp_log_denom(x: Tensor, eps: float) -> Tensor:
    """Clamp a log-space denominator away from zero, preserving sign.

    Prevents an exact division-by-zero pole (and the resulting NaN through
    the reciprocal's backward pass, since d(1/x)/dx is singular at x=0)
    without altering values that are not already at the pole.
    """
    sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return torch.where(x.abs() < eps, sign * eps, x)


class ADPSoftminRuleLayer(RuleLayer):
    """Compute adaptive double-parameter softmin (ADP-softmin) firing strengths for each rule.

    The firing strength approximates the minimum of the membership degrees of the rule
    (Eqs. (33) and (34) of the ADPTSK article).
    """

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
        if kappa <= 0.0:
            raise ValueError("kappa must be > 0")
        if xi <= 0.0:
            raise ValueError("xi must be > 0")
        self.eps = torch.finfo(torch.get_default_dtype()).eps if eps is None else float(eps)
        self.kappa = float(kappa)
        self.xi = float(xi)
        super().__init__(input_names, mf_per_input, rules=rules, rule_base=rule_base, t_norm="prod")

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        r"""Compute ADP-softmin rule strengths from membership outputs.

        For the membership degrees :math:`v_d` of one rule, with minimum
        :math:`\underline{v}` and maximum :math:`\bar{v}`,

        .. math::

            f = \frac{1}{\hat\eta}\Big[\frac{1}{D}\sum_d (\hat\eta v_d)^{\hat q}\Big]^{1/\hat q},
            \qquad
            \ln\hat\eta = -\frac{\xi\ln\underline{v} + (\kappa - \ln D)\ln\bar{v}}{\kappa - \ln D + \xi},
            \qquad
            \hat q = \frac{-\xi}{\ln(\hat\eta\bar{v})}.

        :math:`\hat\eta` and :math:`\hat q` are computed from the current degrees and
        treated as constants in the backward pass, as the index of Ada-softmin is. The sum
        is evaluated in the logarithmic domain, so the bounds :math:`\kappa` and
        :math:`\xi` of double precision do not have to hold in single precision.
        """
        mu = self._gather_terms(membership_outputs).clamp(min=self.eps, max=1.0 - self.eps)
        log_mu = torch.log(mu)
        with torch.no_grad():
            log_min = log_mu.amin(dim=-1)
            log_max = log_mu.amax(dim=-1)
            headroom = self.kappa - math.log(float(self.n_inputs))
            log_eta = -(self.xi * log_min + headroom * log_max) / (headroom + self.xi)
            q1 = headroom / _clamp_log_denom(log_eta + log_min, self.eps)
            q2 = (-self.xi) / _clamp_log_denom(log_eta + log_max, self.eps)
            # Both are negative and equal at the optimum; when every degree of the rule is
            # the same, the two logarithms vanish and any negative index gives that degree.
            q = torch.maximum(q1, q2).clamp(max=-1.0)

        log_terms = q.unsqueeze(-1) * (log_eta.unsqueeze(-1) + log_mu)
        log_avg = torch.logsumexp(log_terms, dim=-1) - math.log(self.n_inputs)
        return torch.exp(log_avg / q - log_eta)


# Threshold ξ for the adaptive q̂ computation in the ALE-softmin (paper eq. 22).
# 700 is safely below the 64-bit underflow boundary (≈745) so that
# exp(q̂·max(μ̃)) = exp(-700) is always representable as a positive float.
# Reference: Xue et al., IEEE TFS 2023, doi: 10.1109/TFUZZ.2023.3270445.
_ALE_SOFTMIN_XI: float = 700.0


class _FeatureGateSwitch(nn.Module):
    """Switch that lets a gated rule layer run without its feature gates.

    The gated families train their gates, prune with them and then fine-tune a plain TSK
    system: once the features are selected the gates are removed from the model. The switch
    is a persistent buffer, so a saved model reloads in the state it was saved in.
    """

    gates_enabled: Tensor

    def _init_gate_switch(self) -> None:
        self.register_buffer("gates_enabled", torch.tensor(True))

    def disable_gates(self) -> None:
        """Remove the feature gates from the forward pass."""
        self.gates_enabled.fill_(False)

    def _load_from_state_dict(self, state_dict: dict[str, Any], prefix: str, *args: Any, **kwargs: Any) -> None:
        # Checkpoints written before the switch existed always had their gates on.
        state_dict.setdefault(f"{prefix}gates_enabled", torch.tensor(True))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class DGALETSKRuleLayer(_FeatureGateSwitch, RuleLayer):
    """Compute adaptive Ln-Exp softmin firing strengths with antecedent feature gates."""

    def __init__(
        self,
        input_names: list[str],
        mf_per_input: list[int],
        rules: Sequence[Sequence[int]] | None = None,
        rule_base: str = "cartesian",
        eps: float | None = None,
        gate_fea: str | Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize DGALETSK rule layer."""
        self.eps = torch.finfo(torch.get_default_dtype()).eps if eps is None else float(eps)
        _gate_fea = gate_fea  # resolved after super().__init__
        super().__init__(input_names, mf_per_input, rules=rules, rule_base=rule_base, t_norm="prod")
        self.gate_fn = resolve_gate_fn(_gate_fea)
        self.lambda_gates = nn.Parameter(torch.zeros(self.n_inputs))
        self._init_gate_switch()
        if isinstance(self.gate_fn, BaseGate):
            self.gate_fn.init_params_(self.lambda_gates)
        else:
            nn.init.constant_(self.lambda_gates, 0.01)

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Compute adaptive Ln-Exp rule strengths from membership outputs."""
        mu = self._gather_terms(membership_outputs).clamp(min=self.eps, max=1.0 - self.eps)
        if bool(self.gates_enabled):
            feature_gates = self.gate_fn(self.lambda_gates).clamp(0.0, 1.0)  # (n_inputs,)
            mu = mu.pow(feature_gates)  # µ^{M(λ)} — exponential antecedent gating

        # Eq. 22: q̂ = -ξ / max_d{μ̃_{r,d}}, computed per sample and rule
        max_mu = mu.amax(dim=-1, keepdim=True).clamp(min=self.eps)  # (B, R, 1)
        q_hat = -_ALE_SOFTMIN_XI / max_mu  # (B, R, 1), always ≤ -ξ
        log_terms = q_hat * mu  # (B, R, D)
        max_log = log_terms.amax(dim=-1, keepdim=True)
        log_sum = max_log + torch.log(torch.exp(log_terms - max_log).sum(dim=-1, keepdim=True))
        return (log_sum / q_hat).squeeze(-1)  # f_r = (1/q̂) · log(Σ exp(q̂·μ̃))


class DGTSKRuleLayer(_FeatureGateSwitch, RuleLayer):
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
        _gate_fea = gate_fea  # resolved after super().__init__
        super().__init__(input_names, mf_per_input, rules=rules, rule_base=rule_base, t_norm="prod")
        self.gate_fn = resolve_gate_fn(_gate_fea)
        self.lambda_gates = nn.Parameter(torch.zeros(self.n_inputs))
        self._init_gate_switch()
        if isinstance(self.gate_fn, BaseGate):
            self.gate_fn.init_params_(self.lambda_gates)
        else:
            nn.init.constant_(self.lambda_gates, 0.1)

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Compute DGTSK rule strengths from membership outputs."""
        mu = self._gather_terms(membership_outputs).clamp(min=self.eps, max=1.0 - self.eps)
        if bool(self.gates_enabled):
            feature_gates = self.gate_fn(self.lambda_gates).clamp(0.0, 1.0).unsqueeze(0)
            mu = mu.pow(feature_gates)  # µ^{M(λ)} — exponential antecedent gating

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
        super().__init__(input_names, mf_per_input, rules=rules, rule_base=rule_base, t_norm="prod")
        self.raw_lambdas = nn.Parameter(torch.full((self.n_rules,), _inv_softplus(lambda_init, self.eps)))

    @property
    def lambdas(self) -> Tensor:
        """Return strictly positive per-rule lambda values."""
        return F.softplus(self.raw_lambdas) + self.eps

    def forward(self, membership_outputs: dict[str, Tensor]) -> Tensor:
        """Compute adaptive Dombi firing strengths for each rule."""
        mu = self._gather_terms(membership_outputs).clamp(min=self.eps, max=1.0 - self.eps)
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

    Supports three training modes to match the FSRE-ADATSK paper protocol:

    * ``"fs"``  — only feature gates :math:`M(\\lambda_d)` are active
      (Phase 1, feature selection, eq. 21).
    * ``"re"``  — only rule gates :math:`M(\\theta_r)` are active
      (Phase 2, rule extraction, eq. 22).
    * ``"finetune"`` — no gates; plain linear TSK consequent (Phase 3,
      eq. 5).
    * ``"both"`` (default) — both gate families applied simultaneously.

    When ``shared_lambda=True`` the feature gate vector has shape
    ``(n_inputs,)`` and is shared across all rules (FSRE-ADATSK, eq. 21).
    When ``shared_lambda=False`` (default) each rule has its own
    ``(n_inputs,)`` gate vector, stored as ``(n_rules, n_inputs)``
    (DG-ALETSK).
    """

    mode: Literal["fs", "re", "finetune", "both"]

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
        if isinstance(self.gate_fn, BaseGate):
            self.gate_fn.init_params_(self.lambda_gates)
            self.gate_fn.init_params_(self.theta_gates)
        else:
            nn.init.constant_(self.lambda_gates, 0.1)
            nn.init.constant_(self.theta_gates, 0.1)

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
        if isinstance(self.gate_fn, BaseGate):
            self.gate_fn.init_params_(self.theta_gates)
        else:
            nn.init.constant_(self.theta_gates, 0.1)

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
        if isinstance(self.gate_fn, BaseGate):
            self.gate_fn.init_params_(self.theta_gates)
        else:
            nn.init.constant_(self.theta_gates, 0.1)

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

    mode: Literal["fs", "re", "finetune", "both"]

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
        if isinstance(self.gate_fn, BaseGate):
            self.gate_fn.init_params_(self.lambda_gates)
            self.gate_fn.init_params_(self.theta_gates)
        else:
            nn.init.constant_(self.lambda_gates, 0.1)
            nn.init.constant_(self.theta_gates, 0.1)

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
