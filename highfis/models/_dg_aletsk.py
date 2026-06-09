"""DG-ALETSK model classes."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from ..defuzzifiers import SoftmaxLogDefuzzifier
from ..layers import (
    DGALETSKRuleLayer,
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


class DGALETSKClassifierModel(BaseTSKClassifierModel):
    """DG-ALETSK classifier with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-ADATSK by replacing the adaptive softmin with the
    *Adaptive Ln-Exp (ALE)* softmin — a smoother variant with improved
    numerical stability.  It also uses a zero-order consequent in the DG
    (data-guided) training phase and optionally converts to first-order
    after gate-based pruning.

    Reference:
        G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional
        Fuzzy Approach With Simultaneous Feature Selection and Rule
        Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no.
        11, pp. 3866-3880, Nov. 2023, doi: 10.1109/TFUZZ.2023.3270445.
    """

    rule_layer: DGALETSKRuleLayer
    consequent_layer: GatedClassificationConsequentLayer | GatedClassificationZeroOrderConsequentLayer

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "coco",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
        use_en_frb: bool = False,
        optimizer_type: str = "sgd",
    ) -> None:
        """Initialise the DG-ALETSK classifier.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            n_classes: Number of output classes (must be ≥ 2).
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB; see also
                *use_en_frb*), or ``"custom"`` (explicit rules via *rules*).
            rules: Explicit rule antecedent indices; ignored when
                ``use_en_frb=True``.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            eps: Numerical stability epsilon for the ALE-softmin operator.
            use_en_frb: Start directly from the Enhanced FRB (En-FRB).
            optimizer_type: Optimizer for all training phases.  ``"sgd"`` (default,
                paper) or ``"adamw"``.

        Raises:
            ValueError: If ``n_classes < 2``.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")

        self.n_classes = int(n_classes)
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

        self.rule_layer = DGALETSKRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules if not self.use_en_frb else None,
            rule_base="en" if self.use_en_frb else rule_base,
            eps=self.eps,
        )
        self.n_rules = self.rule_layer.n_rules
        self.consequent_layer = self._build_zero_order_consequent_layer()

    def _build_zero_order_consequent_layer(self) -> GatedClassificationZeroOrderConsequentLayer:
        return GatedClassificationZeroOrderConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _build_consequent_layer(self) -> GatedClassificationConsequentLayer:
        layer = GatedClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)
        layer.mode = "re"
        return layer

    default_criterion = nn.MSELoss

    def _get_optimizer_config(
        self,
        learning_rate: float,
        weight_decay: float,
    ) -> tuple[type[torch.optim.Optimizer], list[dict[str, Any]]]:
        if self._optimizer_type == "sgd":
            return torch.optim.SGD, [{"params": list(self.parameters())}]
        return super()._get_optimizer_config(learning_rate, weight_decay)

    def convert_to_first_order(self) -> None:
        """Convert the DG phase zero-order consequent to first-order form."""
        previous = self.consequent_layer
        new_consequent = self._build_consequent_layer()
        if isinstance(previous, GatedClassificationZeroOrderConsequentLayer):
            new_consequent.theta_gates.data.copy_(previous.theta_gates.data)
        self.consequent_layer = new_consequent

    def get_feature_gate_values(self) -> Tensor:
        """Return normalized antecedent feature gate values for the DG phase."""
        rule_layer = self.rule_layer
        return rule_layer.gate_fn(rule_layer.lambda_gates)

    def get_rule_gate_values(self) -> Tensor:
        """Return normalized consequent rule gate values for the DG phase."""
        consequent = self.consequent_layer
        return consequent.gate_fn(consequent.theta_gates)

    def compute_thresholds(self, zeta_lambda: float, zeta_theta: float) -> tuple[float, float]:
        """Compute feature and rule thresholds from gate values and coefficient pairs."""
        tau_lambda = _threshold_from_zeta(self.get_feature_gate_values(), zeta_lambda)
        tau_theta = _threshold_from_zeta(self.get_rule_gate_values(), zeta_theta)
        return tau_lambda, tau_theta

    def apply_thresholds(self, tau_lambda: float, tau_theta: float) -> None:
        """Apply threshold pruning to feature and rule gates."""
        if not torch.isfinite(torch.tensor(tau_lambda)) or not torch.isfinite(torch.tensor(tau_theta)):
            raise ValueError("thresholds must be finite")

        feature_gate_values = self.get_feature_gate_values()
        pruned_features = feature_gate_values <= tau_lambda
        rule_layer = self.rule_layer
        cast(Tensor, rule_layer.lambda_gates.data)[pruned_features] = 0.0

        rule_gate_values = self.get_rule_gate_values()
        pruned_rules = rule_gate_values <= tau_theta
        consequent = self.consequent_layer
        cast(Tensor, consequent.theta_gates.data)[pruned_rules] = 0.0

    def _fit_first_order_consequents_lse(self, x: Tensor, y: Tensor) -> None:
        """Refit first-order consequent parameters with antecedents fixed."""
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

            consequent = cast(GatedClassificationConsequentLayer, self.consequent_layer)
            cast(Tensor, consequent.bias.data).copy_(bias)
            cast(Tensor, consequent.weight.data).copy_(weight)

    def _evaluate_threshold_score(self, x: Tensor, y: Tensor) -> float:
        """Evaluate model quality for threshold search."""
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
        gate_fea = self.rule_layer.gate_fn
        new_rule_layer = DGALETSKRuleLayer(
            surviving_names,
            new_mf_per_input,
            rules=new_rules,
            eps=self.eps,
            gate_fea=gate_fea,
        )
        new_rule_layer.lambda_gates.data.copy_(old_lambda)
        self.rule_layer = new_rule_layer

        existing_gate_fn = self.consequent_layer.gate_fn
        if isinstance(self.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
            old_bias = self.consequent_layer.bias.data[surviving_rules].clone()
            old_theta = self.consequent_layer.theta_gates.data[surviving_rules].clone()
            new_cons: GatedClassificationZeroOrderConsequentLayer | GatedClassificationConsequentLayer
            new_cons = GatedClassificationZeroOrderConsequentLayer(
                new_n_rules, new_n_features, self.n_classes, gate_fn=existing_gate_fn
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
                new_n_rules, new_n_features, self.n_classes, gate_fn=existing_gate_fn
            )
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
        """Search threshold coefficients for feature and rule pruning.

        The search follows the DG-ALETSK paper strategy: thresholds are
        computed from gate values, applied to prune gates, and the first-order
        consequent parameters are refit with antecedents fixed.
        """
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
                    if isinstance(candidate.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
                        candidate.convert_to_first_order()
                    tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                    candidate.apply_thresholds(tau_l, tau_t)
                    candidate._fit_first_order_consequents_lse(x, y)
                else:
                    tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                    candidate.apply_thresholds(tau_l, tau_t)

                score = candidate._evaluate_threshold_score(x_eval, y_eval)
                if verbose:
                    self._log("zeta_lambda=%s zeta_theta=%s score=%.6f", zeta_l, zeta_t, score, verbose=True)

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
                    if isinstance(self.consequent_layer, GatedClassificationZeroOrderConsequentLayer):
                        self.convert_to_first_order()
                    if best_state is None:  # pragma: no cover
                        raise RuntimeError("best_state is None despite use_lse=True")
                    self.load_state_dict(best_state)
                else:
                    self.apply_thresholds(best_tau_lambda, best_tau_theta)

        return result

    def init_consequents_from_labels(self, y: Tensor) -> None:
        """Initialise zero-order consequent biases with one-hot labels for P-FRB.

        Sets ``consequent_layer.bias[r, c] = 1`` when sample *r* belongs to class *c*.
        Should be called before training when using P-FRB.

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


class DGALETSKRegressorModel(BaseTSKRegressorModel):
    """DG-ALETSK regressor with ALE-softmin antecedent and double-group gates.

    DG-ALETSK extends FSRE-ADATSK by replacing the adaptive softmin with the
    *Adaptive Ln-Exp (ALE)* softmin — a smoother variant with improved
    numerical stability.  It also uses a zero-order consequent in the DG
    (data-guided) training phase and optionally converts to first-order
    after gate-based pruning.

    Reference:
        G. Xue, J. Wang, B. Yuan and C. Dai, "DG-ALETSK: A High-Dimensional
        Fuzzy Approach With Simultaneous Feature Selection and Rule
        Extraction," in IEEE Transactions on Fuzzy Systems, vol. 31, no.
        11, pp. 3866-3880, Nov. 2023, doi: 10.1109/TFUZZ.2023.3270445.
    """

    rule_layer: DGALETSKRuleLayer
    consequent_layer: GatedRegressionConsequentLayer | GatedRegressionZeroOrderConsequentLayer

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "coco",
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
        eps: float | None = None,
        use_en_frb: bool = False,
        optimizer_type: str = "sgd",
    ) -> None:
        """Initialise the DG-ALETSK regressor.

        Args:
            input_mfs: Mapping from feature name to a sequence of
                :class:`~highfis.memberships.MembershipFunction` objects.
            rule_base: Rule-base construction strategy.  ``"coco"``
                (default, same-index compact), ``"cartesian"`` (all MF
                combinations), ``"en"`` (enhanced FRB; see also
                *use_en_frb*), or ``"custom"`` (explicit rules via *rules*).
            rules: Explicit rule antecedent indices; ignored when
                ``use_en_frb=True``.
            defuzzifier: Custom defuzzifier.  Defaults to
                :class:`~highfis.defuzzifiers.SoftmaxLogDefuzzifier`.
            consequent_batch_norm: Batch normalisation on consequent inputs.
            eps: Numerical stability epsilon for the ALE-softmin operator.
            use_en_frb: Start directly from the Enhanced FRB (En-FRB).
            optimizer_type: Optimizer for all training phases.  ``"sgd"`` (default,
                paper) or ``"adamw"``.
        """
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

        self.rule_layer = DGALETSKRuleLayer(
            self.input_names,
            [len(input_mfs[name]) for name in self.input_names],
            rules=rules if not self.use_en_frb else None,
            rule_base="en" if self.use_en_frb else rule_base,
            eps=self.eps,
        )
        self.n_rules = self.rule_layer.n_rules
        self.consequent_layer = self._build_zero_order_consequent_layer()

    def _build_zero_order_consequent_layer(self) -> GatedRegressionZeroOrderConsequentLayer:
        return GatedRegressionZeroOrderConsequentLayer(self.n_rules, self.n_inputs)

    def _build_consequent_layer(self) -> GatedRegressionConsequentLayer:
        layer = GatedRegressionConsequentLayer(self.n_rules, self.n_inputs)
        layer.mode = "re"
        return layer

    default_criterion = nn.MSELoss

    def _get_optimizer_config(
        self,
        learning_rate: float,
        weight_decay: float,
    ) -> tuple[type[torch.optim.Optimizer], list[dict[str, Any]]]:
        if self._optimizer_type == "sgd":
            return torch.optim.SGD, [{"params": list(self.parameters())}]
        return super()._get_optimizer_config(learning_rate, weight_decay)

    def convert_to_first_order(self) -> None:
        """Convert the DG phase zero-order consequent to first-order form."""
        previous = self.consequent_layer
        new_consequent = self._build_consequent_layer()
        if isinstance(previous, GatedRegressionZeroOrderConsequentLayer):
            new_consequent.theta_gates.data.copy_(previous.theta_gates.data)
        self.consequent_layer = new_consequent

    def get_feature_gate_values(self) -> Tensor:
        """Return normalized antecedent feature gate values for the DG phase."""
        rule_layer = self.rule_layer
        return rule_layer.gate_fn(rule_layer.lambda_gates)

    def get_rule_gate_values(self) -> Tensor:
        """Return normalized consequent rule gate values for the DG phase."""
        consequent = self.consequent_layer
        return consequent.gate_fn(consequent.theta_gates)

    def compute_thresholds(self, zeta_lambda: float, zeta_theta: float) -> tuple[float, float]:
        """Compute feature and rule thresholds from gate values and coefficient pairs."""
        tau_lambda = _threshold_from_zeta(self.get_feature_gate_values(), zeta_lambda)
        tau_theta = _threshold_from_zeta(self.get_rule_gate_values(), zeta_theta)
        return tau_lambda, tau_theta

    def apply_thresholds(self, tau_lambda: float, tau_theta: float) -> None:
        """Apply threshold pruning to feature and rule gates."""
        if not torch.isfinite(torch.tensor(tau_lambda)) or not torch.isfinite(torch.tensor(tau_theta)):
            raise ValueError("thresholds must be finite")

        feature_gate_values = self.get_feature_gate_values()
        pruned_features = feature_gate_values <= tau_lambda
        rule_layer = self.rule_layer
        cast(Tensor, rule_layer.lambda_gates.data)[pruned_features] = 0.0

        rule_gate_values = self.get_rule_gate_values()
        pruned_rules = rule_gate_values <= tau_theta
        consequent = self.consequent_layer
        cast(Tensor, consequent.theta_gates.data)[pruned_rules] = 0.0

    def _fit_first_order_consequents_lse(self, x: Tensor, y: Tensor) -> None:
        """Refit first-order consequent parameters with antecedents fixed."""
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
                active_rules, effective_bias / rule_gates_unsqueezed.squeeze(1), torch.zeros_like(effective_bias)
            )

            denom = (rule_gates_unsqueezed.unsqueeze(-1) * feature_gates.unsqueeze(-1)).squeeze(-1)
            weight = torch.zeros_like(effective_weight)
            nonzero = denom.abs() > 0
            weight[nonzero] = effective_weight[nonzero] / denom[nonzero]

            consequent = cast(GatedRegressionConsequentLayer, self.consequent_layer)
            cast(Tensor, consequent.bias.data).copy_(bias)
            cast(Tensor, consequent.weight.data).copy_(weight)

    def _evaluate_threshold_score(self, x: Tensor, y: Tensor) -> float:
        """Evaluate model quality for threshold search in regression."""
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
        gate_fea = self.rule_layer.gate_fn
        new_rule_layer = DGALETSKRuleLayer(
            surviving_names,
            new_mf_per_input,
            rules=new_rules,
            eps=self.eps,
            gate_fea=gate_fea,
        )
        new_rule_layer.lambda_gates.data.copy_(old_lambda)
        self.rule_layer = new_rule_layer

        existing_gate_fn = self.consequent_layer.gate_fn
        if isinstance(self.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
            old_bias = self.consequent_layer.bias.data[surviving_rules].clone()
            old_theta = self.consequent_layer.theta_gates.data[surviving_rules].clone()
            new_cons: GatedRegressionZeroOrderConsequentLayer | GatedRegressionConsequentLayer
            new_cons = GatedRegressionZeroOrderConsequentLayer(new_n_rules, new_n_features, gate_fn=existing_gate_fn)
            new_cons.bias.data.copy_(old_bias)
            new_cons.theta_gates.data.copy_(old_theta)
        else:
            old_layer = cast(GatedRegressionConsequentLayer, self.consequent_layer)
            old_mode = old_layer.mode
            old_theta = old_layer.theta_gates.data[surviving_rules].clone()
            old_bias = old_layer.bias.data[surviving_rules].clone()
            old_weight = old_layer.weight.data[surviving_rules][:, surviving_features].clone()
            old_lam_cons = old_layer.lambda_gates.data[surviving_rules][:, surviving_features].clone()
            new_cons = GatedRegressionConsequentLayer(new_n_rules, new_n_features, gate_fn=existing_gate_fn)
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
        """Search threshold coefficients for feature and rule pruning."""
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
                    if isinstance(candidate.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
                        candidate.convert_to_first_order()
                    tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                    candidate.apply_thresholds(tau_l, tau_t)
                    candidate._fit_first_order_consequents_lse(x, y)
                else:
                    tau_l, tau_t = candidate.compute_thresholds(zeta_l, zeta_t)
                    candidate.apply_thresholds(tau_l, tau_t)

                score = candidate._evaluate_threshold_score(x_eval, y_eval)
                if verbose:
                    self._log("zeta_lambda=%s zeta_theta=%s score=%.6f", zeta_l, zeta_t, score, verbose=True)

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
                    if isinstance(self.consequent_layer, GatedRegressionZeroOrderConsequentLayer):
                        self.convert_to_first_order()
                    if best_state is None:  # pragma: no cover
                        raise RuntimeError("best_state is None despite use_lse=True")
                    self.load_state_dict(best_state)
                else:
                    self.apply_thresholds(best_tau_lambda, best_tau_theta)

        return result
