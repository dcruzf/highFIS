"""Versioned checkpoint helpers for estimator persistence.

This module provides functions to save and load highFIS estimator checkpoints
using PyTorch serialization.  Checkpoints store the estimator constructor
parameters, the fitted model state dict, and sklearn-compatible fit metadata
(``n_features_in_``, ``feature_names_in_``, ``classes_``, etc.).

The checkpoint schema is versioned: :data:`CHECKPOINT_FORMAT` identifies
the payload type and :data:`CHECKPOINT_VERSION` is tied to the current
package version.  :func:`validate_checkpoint_payload` enforces both so that
incompatible checkpoints are rejected before any state is restored.

Checkpoint schema keys:

- ``format`` — must equal :data:`CHECKPOINT_FORMAT`.
- ``format_version`` — must equal :data:`CHECKPOINT_VERSION`.
- ``estimator_class`` — class name of the estimator that created the
  checkpoint.
- ``estimator_params`` — constructor kwargs used to recreate the estimator.
- ``model_init`` — metadata needed to rebuild the model architecture.
- ``model_state_dict`` — serialized model weights.
- ``fitted_attrs`` — sklearn fit metadata.
- ``history`` *(optional)* — per-epoch training history.

Examples:
    >>> from highfis.persistence import load_checkpoint, validate_checkpoint_payload
    >>> ckpt = load_checkpoint("artifacts/clf.pt")
    >>> validate_checkpoint_payload(
    ...     ckpt, expected_estimator_class="HTSKClassifierEstimator"
    ... )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .version import __version__

CHECKPOINT_FORMAT = "highfis_estimator"
CHECKPOINT_VERSION = __version__


def save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> None:
    """Save a checkpoint dictionary to disk using PyTorch serialization.

    Args:
        path: Target file path.  Parent directories are created
            automatically if they do not exist.
        checkpoint: Dictionary payload containing estimator state.
    """
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint dictionary from disk into CPU memory.

    Args:
        path: Source file path of a checkpoint previously saved with
            :func:`save_checkpoint`.

    Returns:
        The deserialized checkpoint dictionary.

    Raises:
        ValueError: If the loaded payload is not a dictionary.
    """
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
    """Validate a loaded checkpoint payload before estimator reconstruction.

    Args:
        checkpoint: Checkpoint dictionary returned by
            :func:`load_checkpoint`.
        expected_estimator_class: Name of the estimator class that is
            expected to own this checkpoint.

    Raises:
        ValueError: If *format*, *format_version*, or *estimator_class*
            do not match expected values, or if required keys are missing.
    """
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
