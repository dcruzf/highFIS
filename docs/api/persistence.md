# Persistence API

## Module

`highfis.persistence`

This module provides versioned checkpoint serialization helpers for estimator
persistence in highFIS.

## Constants

- `CHECKPOINT_FORMAT`: the expected payload format string, `"highfis_estimator"`.
- `CHECKPOINT_VERSION`: the current package version used to validate checkpoint
  compatibility.

## Functions

### `save_checkpoint(path, checkpoint)`

Save a checkpoint dictionary to disk using PyTorch serialization.

- `path`: target file path or `Path`.
- `checkpoint`: dictionary payload containing estimator state.
- Parent directories are created automatically.
- Uses `torch.save` internally.

### `load_checkpoint(path)`

Load a checkpoint dictionary from disk into CPU memory.

- `path`: source file path or `Path`.
- Uses `torch.load(..., map_location="cpu")` internally.
- Supports older PyTorch versions by retrying without the `weights_only` argument.
- Raises `ValueError` if the loaded payload is not a dictionary.

### `validate_checkpoint_payload(checkpoint, *, expected_estimator_class)`

Validate a loaded checkpoint payload before estimator reconstruction.

Checks include:

- `format` must equal `CHECKPOINT_FORMAT`.
- `format_version` must equal `CHECKPOINT_VERSION`.
- `estimator_class` must match `expected_estimator_class`.
- required keys: `estimator_params`, `model_init`, `model_state_dict`, and `fitted_attrs`.

Raises `ValueError` for any schema mismatch.

## Checkpoint schema

Valid checkpoints are dictionaries with at least the following entries:

- `format`: format identifier string.
- `format_version`: package version string.
- `estimator_class`: name of the estimator class used to create the checkpoint.
- `estimator_params`: constructor parameters used to initialize the estimator.
- `model_init`: model initialization metadata required to rebuild the model.
- `model_state_dict`: serialized model weights.
- `fitted_attrs`: sklearn fit metadata such as `n_features_in_`,
  `feature_names_in_`, and `classes_`.

Optional entries may include:

- `history`: training history produced by the estimator.

## Example

```python
from highfis.persistence import load_checkpoint, validate_checkpoint_payload

checkpoint = load_checkpoint("artifacts/tsk_checkpoint.pt")
validate_checkpoint_payload(
    checkpoint,
    expected_estimator_class="TSKClassifierEstimator",
)
```

## Notes

- Checkpoints are loaded into CPU memory regardless of where they were saved.
- The module intentionally validates the format version against the current
  package version to prevent incompatible checkpoint usage.
- The helper functions are designed for use with estimator wrappers that
  implement `save(path)` and `load(path)`.
