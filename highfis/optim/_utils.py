"""Utility functions for highFIS training and optimization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..models import BaseTSK


def _uniform_regularization_loss(normalized_weights: Tensor, target: float | None = None) -> Tensor:
    """Penalize deviation from a uniform average rule activation distribution."""
    n_rules = normalized_weights.shape[1]
    target_value = (1.0 / float(n_rules)) if target is None else float(target)
    target_tensor = normalized_weights.new_full((n_rules,), target_value)
    avg_activation = normalized_weights.mean(dim=0)
    return torch.sum((avg_activation - target_tensor) ** 2)


def _resolve_verbose(verbose: bool | int = False) -> int:
    """Normalize verbose settings to a numeric verbosity level."""
    if isinstance(verbose, bool):
        return 1 if verbose else 0
    if not isinstance(verbose, int):
        raise TypeError("verbose must be an int in 0..3 or a bool")
    if verbose < 0 or verbose > 3:
        raise ValueError("verbose must be between 0 and 3")
    return verbose


def _log(
    logger: logging.Logger,
    message: str,
    *args: Any,
    level: int = logging.INFO,
    verbose: bool | int = False,
    min_level: int = 2,
    **kwargs: Any,
) -> None:
    """Log a message when verbose mode is enabled."""
    if _resolve_verbose(verbose) < min_level:
        return
    logger.log(level, message, *args, **kwargs)


def _get_optimizer_config(
    model: BaseTSK,
    learning_rate: float,
    weight_decay: float,
) -> tuple[type[torch.optim.Optimizer], list[dict[str, Any]]]:
    """Return the optimizer class and parameter groups for a model."""
    ante_params = list(model.membership_layer.parameters())
    rule_params = list(model.rule_layer.parameters())
    cons_params = list(model.consequent_layer.parameters())
    if model.consequent_bn is not None:
        cons_params.extend(model.consequent_bn.parameters())

    class_name = model.__class__.__name__

    if "ADATSK" in class_name:
        return torch.optim.SGD, [
            {"params": ante_params},
            {"params": rule_params},
            {"params": cons_params},
        ]
    elif "ADPTSK" in class_name or "DombiTSK" in class_name or "ADMTSK" in class_name:
        return torch.optim.Adam, [
            {"params": ante_params},
            {"params": rule_params},
            {"params": cons_params},
        ]
    elif "AYATSK" in class_name:
        return torch.optim.Adam, [
            {"params": ante_params, "weight_decay": weight_decay},
            {"params": rule_params, "weight_decay": weight_decay},
            {"params": cons_params, "weight_decay": weight_decay},
        ]
    elif model._optimizer_type == "sgd":
        return torch.optim.SGD, [{"params": list(model.parameters())}]

    # Default: AdamW
    return torch.optim.AdamW, [
        {"params": ante_params, "weight_decay": 0.0},
        {"params": rule_params, "weight_decay": 0.0},
        {"params": cons_params, "weight_decay": weight_decay},
    ]
