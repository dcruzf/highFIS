# Persistence API

## Module

`highfis.persistence`

The persistence module provides versioned checkpoint serialization and
reconstruction for estimator objects without depending on Python object
pickling.

## Overview

Persisted checkpoints are stored with `torch.save` and loaded with
`torch.load` into CPU memory. The checkpoint payload is a structured dictionary
that includes:

- `format`: checkpoint format name (`highfis_estimator`)
- `format_version`: package version used to create the checkpoint
- `estimator_class`: estimator class name used during serialization
- `estimator_params`: constructor parameters for the estimator
- `model_init`: model initialization metadata needed to rebuild `model_`
- `model_state_dict`: learned model weights
- `fitted_attrs`: fitted sklearn metadata such as `n_features_in_`,
  `feature_names_in_`, and `classes_`
- `history`: optional training history, if available

## Functions

### `save_checkpoint(path, checkpoint)`

Save a checkpoint dictionary to disk using PyTorch serialization. Parent
directories are created automatically.

### `load_checkpoint(path)`

Load a checkpoint dictionary from disk into CPU memory. The function raises
`ValueError` if the loaded object is not a dictionary.

### `validate_checkpoint_payload(checkpoint, *, expected_estimator_class)`

Validate a loaded checkpoint payload for:

- the expected `format`
- the expected `format_version`
- the expected `estimator_class`
- required checkpoint keys

This provides a safety net before reconstructing an estimator from a file.

## Estimator persistence

Most estimator classes in `highfis` support convenient persistence methods:

- `estimator.save(path)`
- `EstimatorClass.load(path)`

Example:

```python
from highfis import TSKClassifierEstimator

clf = TSKClassifierEstimator(
    n_mfs=4,
    mf_init="kmeans",
    epochs=150,
    learning_rate=1e-3,
    random_state=42,
)
clf.fit(X_train, y_train)
clf.save("artifacts/tsk_classifier.pt")

restored = TSKClassifierEstimator.load("artifacts/tsk_classifier.pt")
assert restored.score(X_test, y_test) == clf.score(X_test, y_test)
```

## Low-level usage

If you need direct access to checkpoint payloads, use the persistence helpers:

```python
from highfis.persistence import load_checkpoint, validate_checkpoint_payload

checkpoint = load_checkpoint("artifacts/tsk_classifier.pt")
validate_checkpoint_payload(checkpoint, expected_estimator_class="TSKClassifierEstimator")
```
