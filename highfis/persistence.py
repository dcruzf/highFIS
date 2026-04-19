from __future__ import annotations

from pathlib import Path
from typing import Any

from .version import __version__

CHECKPOINT_FORMAT = "highfis_estimator"
CHECKPOINT_VERSION = __version__


def save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> None:
    """Save a checkpoint dictionary to disk using torch serialization."""
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint dictionary from disk into CPU memory."""
    import torch

    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)  # nosec
    except TypeError:
        payload = torch.load(source, map_location="cpu")  # nosec

    if not isinstance(payload, dict):
        raise ValueError("invalid checkpoint: expected a dictionary payload")
    return payload


def validate_checkpoint_payload(checkpoint: dict[str, Any], *, expected_estimator_class: str) -> None:
    """Validate basic checkpoint schema and expected estimator class."""
    fmt = checkpoint.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise ValueError(f"invalid checkpoint format '{fmt}', expected '{CHECKPOINT_FORMAT}'")

    version = checkpoint.get("format_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version {version}, expected package version {CHECKPOINT_VERSION}")

    estimator_class = checkpoint.get("estimator_class")
    if estimator_class != expected_estimator_class:
        raise ValueError(f"checkpoint was created for '{estimator_class}', not '{expected_estimator_class}'")

    for key in ("estimator_params", "model_init", "model_state_dict", "fitted_attrs"):
        if key not in checkpoint:
            raise ValueError(f"invalid checkpoint: missing '{key}'")
