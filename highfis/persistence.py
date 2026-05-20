"""Versioned checkpoint helpers for estimator persistence.

This module provides functions to save and load highFIS estimator checkpoints
using PyTorch serialization. Checkpoints store estimator constructor
parameters, the fitted model state dict, and sklearn-compatible fit metadata
(``n_features_in_``, ``feature_names_in_``, ``classes_``, etc.).

The checkpoint schema is versioned. ``CHECKPOINT_FORMAT`` identifies the
payload type and ``CHECKPOINT_FORMAT_VERSION`` is an integer string incremented
only when the checkpoint schema itself changes — independent of the package
version. ``validate_checkpoint_payload`` enforces both so that incompatible
checkpoints are rejected before any state is restored.

Checkpoint schema keys:

- ``format`` — must equal ``CHECKPOINT_FORMAT``.
- ``format_version`` — must equal ``CHECKPOINT_FORMAT_VERSION``.
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


def _get_mf_registry() -> dict[str, type]:
    """Lazily return the supported MF type registry to avoid circular imports."""
    from .memberships import (
        CompositeGMF,
        DimensionDependentGaussianMF,
        GaussianMF,
        GaussianPiMF,
    )

    return {
        "CompositeGMF": CompositeGMF,
        "DimensionDependentGaussianMF": DimensionDependentGaussianMF,
        "GaussianMF": GaussianMF,
        "GaussianPiMF": GaussianPiMF,
        # Legacy key kept for backward-compatible checkpoint loading.
        "GaussianPIMF": GaussianPiMF,
    }


CHECKPOINT_FORMAT = "highfis_estimator"
CHECKPOINT_FORMAT_VERSION = "1"


def serialize_input_mfs(input_mfs: Any) -> dict[str, list[dict[str, Any]]]:
    """Serialize an ``nn.ModuleDict`` of membership functions to a plain dict.

    Converts each membership function to a ``{"type": classname, "params": {...}}``
    entry so the checkpoint contains only primitive Python types and tensors,
    making it compatible with ``torch.load(..., weights_only=True)``.

    Args:
        input_mfs: The ``input_mfs`` attribute of a fitted
            :class:`~highfis.layers.FuzzificationLayer` (an ``nn.ModuleDict``).

    Returns:
        A JSON-serializable dict mapping feature name to a list of MF configs.
    """
    return {
        name: [{"type": type(mf).__name__, "params": mf.inspect_params()} for mf in mf_list]
        for name, mf_list in input_mfs.items()
    }


def deserialize_input_mfs(
    config: dict[str, list[dict[str, Any]]],
) -> dict[str, list[Any]]:
    """Reconstruct ``input_mfs`` from a serialized config dict.

    Args:
        config: A dict as returned by :func:`serialize_input_mfs`.

    Returns:
        A mapping of feature name to a list of
        :class:`~highfis.memberships.MembershipFunction` instances suitable
        for passing to :class:`~highfis.layers.FuzzificationLayer`.

    Raises:
        ValueError: If the config contains an unrecognised MF type name.
    """
    registry = _get_mf_registry()
    result: dict[str, list[Any]] = {}
    for name, mf_configs in config.items():
        mf_list: list[Any] = []
        for mf_cfg in mf_configs:
            mf_type = mf_cfg["type"]
            if mf_type not in registry:
                raise ValueError(f"unknown membership function type '{mf_type}'; known types: {sorted(registry)}")
            mf_list.append(registry[mf_type](**mf_cfg["params"]))
        result[name] = mf_list
    return result


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
    payload = torch.load(source, map_location="cpu", weights_only=True)

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
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint version {version!r}, expected {CHECKPOINT_FORMAT_VERSION!r}")

    estimator_class = checkpoint.get("estimator_class")
    if estimator_class != expected_estimator_class:
        raise ValueError(f"checkpoint was created for '{estimator_class}', not '{expected_estimator_class}'")

    for key in ("estimator_params", "model_init", "model_state_dict", "fitted_attrs"):
        if key not in checkpoint:
            raise ValueError(f"invalid checkpoint: missing '{key}'")
