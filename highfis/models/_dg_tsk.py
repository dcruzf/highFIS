"""DG-TSK model classes."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from ..defuzzifiers import SoftmaxLogDefuzzifier
from ..layers import (
    DGTSKRuleLayer,
    GatedClassificationConsequentLayer,
    GatedClassificationZeroOrderConsequentLayer,
    GatedRegressionConsequentLayer,
    GatedRegressionZeroOrderConsequentLayer,
    MembershipLayer,
)
from ..memberships import MembershipFunction
from ._common import (
    BaseTSKClassifierModel,
    BaseTSKRegressorModel,
    _build_first_order_design_matrix,
    _solve_lse,
    _threshold_from_zeta,
)


class DGTSKClassifierModel(BaseTSKClassifierModel):
    """DG-TSK classifier with M-gate antecedent and point-based FRB (P-FRB).

    DG-TSK uses a data-guided M-gate function to automatically select
    relevant features and rules.

    Reference:
        Guangdong Xue, Jian Wang, Bingjie Zhang, Bin Yuan, Caili Dai,
        Double groups of gates based Takagi-Sugeno-Kang (DG-TSK)
        fuzzy system for simultaneous feature selection and rule
        extraction, Fuzzy Sets and Systems, Volume 469, 2023, 108627,
        ISSN 0165-0114, https://doi.org/10.1016/j.fss.2023.108627.
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
        optimizer_type: str = "sgd",
    ) -> None:
        """Initialise the DG-TSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB; see also
                *use_en_frb*), or ``"custom"`` (explicit rules via *rules*).
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
            optimizer_type: Optimizer for all training phases.  ``"sgd"`` (default,
                paper) or ``"adamw"``.

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
        self._optimizer_type = str(optimizer_type)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
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
        layer = GatedClassificationConsequentLayer(
            self.n_rules,
            self.n_inputs,
            self.n_classes,
            gate_fn=self.gate_rule,
        )
        layer.mode = "re"
        return layer

    default_criterion = nn.MSELoss

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
            feature_gates = torch.ones(self.n_rules, self.n_inputs, dtype=x.dtype, device=x.device)
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

    def prune_structure(
        self,
        surviving_features: list[int],
        surviving_rules: list[int],
    ) -> None:
        """Structurally prune the model to the given surviving feature and rule indices.

        Rebuilds :attr:`membership_layer`, :attr:`rule_layer`, and
        :attr:`consequent_layer` in-place, retaining only the specified
        features and rules.  All associated parameters are copied from
        the current layers.
        """
        if not surviving_features:
            raise ValueError("surviving_features must not be empty")
        if not surviving_rules:
            raise ValueError("surviving_rules must not be empty")

        surviving_names = [self.input_names[i] for i in surviving_features]

        # For each surviving feature collect the MF indices used by the surviving
        # rules and build a remapping to a compressed 0-based index.
        new_input_mfs: dict[str, list[MembershipFunction]] = {}
        mf_remap: dict[int, dict[int, int]] = {}
        for orig_fi in surviving_features:
            name = self.input_names[orig_fi]
            all_mfs = list(cast(Any, self.membership_layer.input_mfs[name]))
            used = sorted({self.rule_layer.rules[r][orig_fi] for r in surviving_rules})
            new_input_mfs[name] = [all_mfs[mf_idx] for mf_idx in used]
            mf_remap[orig_fi] = {old: new for new, old in enumerate(used)}

        new_n_features = len(surviving_features)
        new_n_rules = len(surviving_rules)
        new_mf_per_input = [len(new_input_mfs[self.input_names[fi]]) for fi in surviving_features]
        new_rules = [
            tuple(mf_remap[fi][self.rule_layer.rules[r][fi]] for fi in surviving_features) for r in surviving_rules
        ]

        # Rebuild membership layer.
        self.membership_layer = MembershipLayer(new_input_mfs)

        # Rebuild rule layer, copying lambda_gates for surviving features.
        old_lambda = self.rule_layer.lambda_gates.data[surviving_features].clone()
        new_rule_layer = DGTSKRuleLayer(
            surviving_names,
            new_mf_per_input,
            rules=new_rules,
            gate_fea=self.gate_fea,
            eps=self.eps,
        )
        new_rule_layer.lambda_gates.data.copy_(old_lambda)
        self.rule_layer = new_rule_layer

        # Rebuild consequent layer, copying surviving parameter slices.
        if isinstance(self.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
            old_bias = self.consequent_layer.bias.data[surviving_rules].clone()
            old_theta = self.consequent_layer.theta_gates.data[surviving_rules].clone()
            new_cons: GatedClassificationZeroOrderConsequentLayer | GatedClassificationConsequentLayer
            new_cons = GatedClassificationZeroOrderConsequentLayer(
                new_n_rules, new_n_features, self.n_classes, gate_fn=self.gate_rule
            )
            new_cons.bias.data.copy_(old_bias)
            new_cons.theta_gates.data.copy_(old_theta)
        else:
            old_layer = cast(GatedClassificationConsequentLayer, self.consequent_layer)
            old_mode = old_layer.mode
            old_theta = old_layer.theta_gates.data[surviving_rules].clone()
            old_bias = old_layer.bias.data[surviving_rules].clone()
            old_weight = old_layer.weight.data[surviving_rules][:, :, surviving_features].clone()
            old_lam_cons = old_layer.lambda_gates.data[surviving_rules][:, surviving_features].clone()
            new_cons = GatedClassificationConsequentLayer(
                new_n_rules, new_n_features, self.n_classes, gate_fn=self.gate_rule
            )
            new_cons.mode = old_mode
            new_cons.theta_gates.data.copy_(old_theta)
            new_cons.bias.data.copy_(old_bias)
            new_cons.weight.data.copy_(old_weight)
            new_cons.lambda_gates.data.copy_(old_lam_cons)
        self.consequent_layer = new_cons

        # Update model-level bookkeeping.
        self.input_names = surviving_names
        self.n_inputs = new_n_features
        self.n_rules = new_n_rules
        self.input_mfs = new_input_mfs

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
        structural: bool = True,
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
                if use_lse:
                    # LSE path: convert to first-order, apply thresholds, fit LSE,
                    # then evaluate accuracy on validation set.
                    if isinstance(candidate.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
                        candidate.convert_to_first_order()
                    tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                    candidate.apply_thresholds(tau_l, tau_t)
                    candidate._fit_first_order_consequents_lse(x, y)
                else:
                    # Non-LSE path: evaluate the zero-order model directly after
                    # pruning.  Conversion to first-order happens only in the
                    # inplace step (or is done by the caller).
                    tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                    candidate.apply_thresholds(tau_l, tau_t)

                score = candidate._evaluate_threshold_score(x_eval, y_eval)
                if verbose:
                    self.logger.info("zeta_lambda=%s zeta_theta=%s score=%.6f", zeta_l, zeta_t, score)

                if score > best_score:
                    best_score = score
                    # Store the candidate's raw threshold parameters (not state_dict)
                    # so they can be replayed correctly in both LSE and non-LSE paths.
                    best_state = copy.deepcopy(candidate.state_dict()) if use_lse else None
                    best_tau_lambda = tau_l
                    best_tau_theta = tau_t
                    best_zeta_lambda = zeta_l
                    best_zeta_theta = zeta_t

        if best_score == float("-inf"):
            raise RuntimeError("threshold search did not yield a valid candidate")

        result: dict[str, Any] = {
            "best_score": best_score,
            "best_zeta_lambda": best_zeta_lambda,
            "best_zeta_theta": best_zeta_theta,
            "tau_lambda": best_tau_lambda,
            "tau_theta": best_tau_theta,
        }

        result["surviving_feature_indices"] = list(range(self.n_inputs))
        result["surviving_rule_indices"] = list(range(self.n_rules))

        if inplace:
            if structural:
                # Compute surviving indices before zeroing gate params.
                feature_gate_values = self.get_feature_gate_values()
                rule_gate_values = self.get_rule_gate_values()
                sf = torch.where(feature_gate_values > best_tau_lambda)[0].tolist()
                sr = torch.where(rule_gate_values > best_tau_theta)[0].tolist()
                if not sf:
                    sf = list(range(self.n_inputs))
                if len(sr) < self.n_classes:
                    top_k = min(self.n_classes, self.n_rules)
                    top_rules = torch.topk(rule_gate_values, k=top_k).indices.tolist()
                    sr = sorted(set(sr) | set(top_rules))
                if use_lse and isinstance(self.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
                    self.convert_to_first_order()
                self.apply_thresholds(best_tau_lambda, best_tau_theta)
                self.prune_structure(sf, sr)
                if use_lse:
                    self._fit_first_order_consequents_lse(x[:, sf], y)
                result["surviving_feature_indices"] = sf
                result["surviving_rule_indices"] = sr
            else:
                if use_lse:
                    # best_state holds the already-converted, pruned, LSE-fitted model.
                    if isinstance(self.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
                        self.convert_to_first_order()
                    if best_state is None:  # pragma: no cover
                        raise RuntimeError("best_state is None despite use_lse=True")
                    self.load_state_dict(best_state)
                else:
                    # Non-LSE path: apply the best thresholds to the zero-order model.
                    self.apply_thresholds(best_tau_lambda, best_tau_theta)

        return result

    def init_consequents_from_labels(self, y: Tensor) -> None:
        """Initialise zero-order consequent biases with one-hot encoded labels (P-FRB paper eq. 24).

        Sets ``consequent_layer.bias[r, c] = 1`` when sample *r* belongs to class *c*,
        exactly as in Xue et al. (2023) eq. (24): $p^0_{r,c} = y_{r,c}$.
        Should be called before :meth:`fit_dg_phase` when using P-FRB.

        Args:
            y: Integer class labels of shape ``(N,)`` for the *N* training samples
                used to build the P-FRB. Only the first ``n_rules`` labels are used.

        Raises:
            ValueError: If the model is not in zero-order consequent mode.
        """
        if not isinstance(self.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
            raise ValueError(
                "init_consequents_from_labels() requires a zero-order consequent layer; "
                "call before convert_to_first_order()"
            )
        n = min(len(y), self.n_rules)
        dtype = self.consequent_layer.bias.dtype
        device = self.consequent_layer.bias.device
        one_hot = torch.zeros(self.n_rules, self.n_classes, dtype=dtype, device=device)
        one_hot[:n].scatter_(1, y[:n].to(device=device).unsqueeze(1), 1.0)
        self.consequent_layer.bias.data.copy_(one_hot)


class DGTSKRegressorModel(BaseTSKRegressorModel):
    """DG-TSK regressor with M-gate antecedent and point-based FRB (P-FRB).

    DG-TSK uses a data-guided M-gate function to automatically select
    relevant features and rules.

    Reference:
        Guangdong Xue, Jian Wang, Bingjie Zhang, Bin Yuan, Caili Dai,
        Double groups of gates based Takagi-Sugeno-Kang (DG-TSK)
        fuzzy system for simultaneous feature selection and rule
        extraction, Fuzzy Sets and Systems, Volume 469, 2023, 108627,
        ISSN 0165-0114, https://doi.org/10.1016/j.fss.2023.108627.
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
        optimizer_type: str = "sgd",
    ) -> None:
        """Initialise the DG-TSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB; see also
                *use_en_frb*), or ``"custom"`` (explicit rules via *rules*).
            gate_fea: Gate function for antecedent feature selection
                (default ``"gate_m"``)).
            gate_rule: Gate function for consequent rule selection
                (default ``"gate_m"``).
            rules: Explicit rule antecedent indices; ignored when
                ``use_en_frb=True``.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            eps: Numerical stability epsilon.
            use_en_frb: Use the Enhanced FRB (P-FRB) rule base.
            optimizer_type: Optimizer for all training phases.  ``"sgd"`` (default,
                paper) or ``"adamw"``.
        """
        self.gate_fea = gate_fea
        self.gate_rule = gate_rule
        self.eps = eps
        self.use_en_frb = bool(use_en_frb)
        self._optimizer_type = str(optimizer_type)

        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm="prod",
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
        layer = GatedRegressionConsequentLayer(self.n_rules, self.n_inputs, gate_fn=self.gate_rule)
        layer.mode = "re"
        return layer

    default_criterion = nn.MSELoss

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
            feature_gates = torch.ones(self.n_rules, self.n_inputs, dtype=x.dtype, device=x.device)
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

    def prune_structure(
        self,
        surviving_features: list[int],
        surviving_rules: list[int],
    ) -> None:
        """Structurally prune the model to the given surviving feature and rule indices.

        Rebuilds :attr:`membership_layer`, :attr:`rule_layer`, and
        :attr:`consequent_layer` in-place, retaining only the specified
        features and rules.  All associated parameters are copied from
        the current layers.
        """
        if not surviving_features:
            raise ValueError("surviving_features must not be empty")
        if not surviving_rules:
            raise ValueError("surviving_rules must not be empty")

        surviving_names = [self.input_names[i] for i in surviving_features]

        new_input_mfs: dict[str, list[MembershipFunction]] = {}
        mf_remap: dict[int, dict[int, int]] = {}
        for orig_fi in surviving_features:
            name = self.input_names[orig_fi]
            all_mfs = list(cast(Any, self.membership_layer.input_mfs[name]))
            used = sorted({self.rule_layer.rules[r][orig_fi] for r in surviving_rules})
            new_input_mfs[name] = [all_mfs[mf_idx] for mf_idx in used]
            mf_remap[orig_fi] = {old: new for new, old in enumerate(used)}

        new_n_features = len(surviving_features)
        new_n_rules = len(surviving_rules)
        new_mf_per_input = [len(new_input_mfs[self.input_names[fi]]) for fi in surviving_features]
        new_rules = [
            tuple(mf_remap[fi][self.rule_layer.rules[r][fi]] for fi in surviving_features) for r in surviving_rules
        ]

        self.membership_layer = MembershipLayer(new_input_mfs)

        old_lambda = self.rule_layer.lambda_gates.data[surviving_features].clone()
        new_rule_layer = DGTSKRuleLayer(
            surviving_names,
            new_mf_per_input,
            rules=new_rules,
            gate_fea=self.gate_fea,
            eps=self.eps,
        )
        new_rule_layer.lambda_gates.data.copy_(old_lambda)
        self.rule_layer = new_rule_layer

        if isinstance(self.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
            old_bias = self.consequent_layer.bias.data[surviving_rules].clone()
            old_theta = self.consequent_layer.theta_gates.data[surviving_rules].clone()
            new_cons: GatedRegressionZeroOrderConsequentLayer | GatedRegressionConsequentLayer
            new_cons = GatedRegressionZeroOrderConsequentLayer(new_n_rules, new_n_features, gate_fn=self.gate_rule)
            new_cons.bias.data.copy_(old_bias)
            new_cons.theta_gates.data.copy_(old_theta)
        else:
            old_layer = cast(GatedRegressionConsequentLayer, self.consequent_layer)
            old_mode = old_layer.mode
            old_theta = old_layer.theta_gates.data[surviving_rules].clone()
            old_bias = old_layer.bias.data[surviving_rules].clone()
            old_weight = old_layer.weight.data[surviving_rules][:, surviving_features].clone()
            old_lam_cons = old_layer.lambda_gates.data[surviving_rules][:, surviving_features].clone()
            new_cons = GatedRegressionConsequentLayer(new_n_rules, new_n_features, gate_fn=self.gate_rule)
            new_cons.mode = old_mode
            new_cons.theta_gates.data.copy_(old_theta)
            new_cons.bias.data.copy_(old_bias)
            new_cons.weight.data.copy_(old_weight)
            new_cons.lambda_gates.data.copy_(old_lam_cons)
        self.consequent_layer = new_cons

        self.input_names = surviving_names
        self.n_inputs = new_n_features
        self.n_rules = new_n_rules
        self.input_mfs = new_input_mfs

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
        structural: bool = True,
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
                if use_lse:
                    # LSE path: convert to first-order, apply thresholds, fit LSE,
                    # then evaluate negative MSE on validation set.
                    if isinstance(candidate.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
                        candidate.convert_to_first_order()
                    tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                    candidate.apply_thresholds(tau_l, tau_t)
                    candidate._fit_first_order_consequents_lse(x, y)
                else:
                    # Non-LSE path: evaluate the zero-order model directly after
                    # pruning.  Conversion to first-order happens only in the
                    # inplace step (or is done by the caller).
                    tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                    candidate.apply_thresholds(tau_l, tau_t)

                score = candidate._evaluate_threshold_score(x_eval, y_eval)
                if verbose:
                    self.logger.info("zeta_lambda=%s zeta_theta=%s score=%.6f", zeta_l, zeta_t, score)

                if score > best_score:
                    best_score = score
                    best_state = copy.deepcopy(candidate.state_dict()) if use_lse else None
                    best_tau_lambda = tau_l
                    best_tau_theta = tau_t
                    best_zeta_lambda = zeta_l
                    best_zeta_theta = zeta_t

        if best_score == float("-inf"):
            raise RuntimeError("threshold search did not yield a valid candidate")

        result: dict[str, Any] = {
            "best_score": best_score,
            "best_zeta_lambda": best_zeta_lambda,
            "best_zeta_theta": best_zeta_theta,
            "tau_lambda": best_tau_lambda,
            "tau_theta": best_tau_theta,
        }

        result["surviving_feature_indices"] = list(range(self.n_inputs))
        result["surviving_rule_indices"] = list(range(self.n_rules))

        if inplace:
            if structural:
                feature_gate_values = self.get_feature_gate_values()
                rule_gate_values = self.get_rule_gate_values()
                sf = torch.where(feature_gate_values > best_tau_lambda)[0].tolist()
                sr = torch.where(rule_gate_values > best_tau_theta)[0].tolist()
                if not sf:
                    sf = list(range(self.n_inputs))
                if not sr:
                    sr = list(range(self.n_rules))
                if use_lse and isinstance(self.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
                    self.convert_to_first_order()
                self.apply_thresholds(best_tau_lambda, best_tau_theta)
                self.prune_structure(sf, sr)
                if use_lse:
                    self._fit_first_order_consequents_lse(x[:, sf], y)
                result["surviving_feature_indices"] = sf
                result["surviving_rule_indices"] = sr
            else:
                if use_lse:
                    # best_state holds the already-converted, pruned, LSE-fitted model.
                    if isinstance(self.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
                        self.convert_to_first_order()
                    if best_state is None:  # pragma: no cover
                        raise RuntimeError("best_state is None despite use_lse=True")
                    self.load_state_dict(best_state)
                else:
                    # Non-LSE path: apply the best thresholds to the zero-order model.
                    self.apply_thresholds(best_tau_lambda, best_tau_theta)

        return result
