from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .base import BaseTSK
from .layers import ClassificationConsequentLayer, RegressionConsequentLayer
from .memberships import MembershipFunction
from .t_norms import TNormFn


class HTSKClassifier(BaseTSK):
    """TSK classifier with HTSK defuzzification for high-dimensional data."""

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        n_classes: int,
        rule_base: str = "cartesian",
        t_norm: str = "gmean",
        t_norm_fn: TNormFn | None = None,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialize HTSK classifier architecture and consequent head."""
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = int(n_classes)
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            t_norm_fn=t_norm_fn,
            rules=rules,
            defuzzifier=defuzzifier,
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        return ClassificationConsequentLayer(self.n_rules, self.n_inputs, self.n_classes)

    def _default_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _compute_loss(self, criterion: Callable[[Tensor, Tensor], Tensor], output: Tensor, target: Tensor) -> Tensor:
        if isinstance(criterion, nn.MSELoss):
            one_hot = torch.zeros_like(output)
            one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            return criterion(output, one_hot)
        return criterion(output, target)

    def _evaluate_validation(
        self, criterion: Callable[[Tensor, Tensor], Tensor], x_val: Tensor, y_val: Tensor
    ) -> dict[str, float]:
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


class HTSKRegressor(BaseTSK):
    """TSK regressor with HTSK defuzzification for high-dimensional data."""

    def __init__(
        self,
        input_mfs: Mapping[str, Sequence[MembershipFunction]],
        rule_base: str = "cartesian",
        t_norm: str = "gmean",
        t_norm_fn: TNormFn | None = None,
        rules: Sequence[Sequence[int]] | None = None,
        defuzzifier: nn.Module | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Initialize HTSK regressor architecture and consequent head."""
        super().__init__(
            input_mfs,
            rule_base=rule_base,
            t_norm=t_norm,
            t_norm_fn=t_norm_fn,
            rules=rules,
            defuzzifier=defuzzifier,
            consequent_batch_norm=consequent_batch_norm,
        )

    def _build_consequent_layer(self) -> nn.Module:
        return RegressionConsequentLayer(self.n_rules, self.n_inputs)

    def _default_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def _compute_loss(self, criterion: Callable[[Tensor, Tensor], Tensor], output: Tensor, target: Tensor) -> Tensor:
        return criterion(output.squeeze(1), target)

    def predict(self, x: Tensor) -> Tensor:
        """Return predicted values as a 1-D tensor."""
        with torch.no_grad():
            return self.forward(x).squeeze(1)


__all__ = ["HTSKClassifier", "HTSKRegressor"]
