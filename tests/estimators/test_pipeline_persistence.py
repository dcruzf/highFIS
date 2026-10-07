"""A fitted scikit-learn ``Pipeline`` holding a highFIS estimator survives a save and reload.

This is the path a user follows to apply a model to new data: the preprocessing has to
travel with the estimator. ``estimator.save`` stores the estimator alone, so the pipeline
is persisted with ``joblib``, as scikit-learn recommends.
"""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import highfis

# One plain family, one regressor, and the families that prune features during fit.
ESTIMATORS = ["HTSKClassifier", "TSKRegressor", "DGTSKClassifier", "FSREADATSKRegressor", "MHTSKClassifier"]


def _fitted_pipeline(name: str) -> tuple[Pipeline, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((120, 6)) * 50.0  # far from [0, 1]: predictions are wrong without the scaler
    y: Any = (x[:, 0] + x[:, 1] > 50.0).astype(int) if name.endswith("Classifier") else 2.0 * x[:, 0]
    cls = getattr(highfis, name)
    epochs = {p: 4 for p in inspect.signature(cls.__init__).parameters if p.endswith("epochs")}
    pipe = Pipeline([("scale", MinMaxScaler()), ("model", cls(random_state=0, **epochs))]).fit(x, y)
    return pipe, rng.random((15, 6)) * 50.0


@pytest.mark.parametrize("name", ESTIMATORS)
def test_pipeline_round_trips_through_joblib(name: str, tmp_path: Path) -> None:
    pipe, x_new = _fitted_pipeline(name)
    path = tmp_path / "pipeline.joblib"

    joblib.dump(pipe, path)
    reloaded = joblib.load(path)

    np.testing.assert_array_equal(reloaded.predict(x_new), pipe.predict(x_new))


@pytest.mark.parametrize("name", ESTIMATORS)
def test_pipeline_round_trips_through_pickle(name: str) -> None:
    pipe, x_new = _fitted_pipeline(name)

    reloaded = pickle.loads(pickle.dumps(pipe))

    np.testing.assert_array_equal(reloaded.predict(x_new), pipe.predict(x_new))


def test_reloaded_pipeline_keeps_the_fitted_scaler(tmp_path: Path) -> None:
    """The reloaded pipeline scales new data with the statistics learned in ``fit``."""
    pipe, x_new = _fitted_pipeline("HTSKClassifier")
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipe, path)

    reloaded = joblib.load(path)

    np.testing.assert_array_equal(reloaded.named_steps["scale"].data_max_, pipe.named_steps["scale"].data_max_)
    np.testing.assert_array_equal(reloaded.predict_proba(x_new), pipe.predict_proba(x_new))


def test_estimator_save_alone_does_not_carry_the_preprocessing(tmp_path: Path) -> None:
    """Why the pipeline is saved as a whole: the estimator checkpoint has no scaler."""
    pipe, x_new = _fitted_pipeline("HTSKClassifier")
    path = tmp_path / "model.pt"
    pipe.named_steps["model"].save(str(path))

    model_only = highfis.HTSKClassifier.load(str(path))

    np.testing.assert_array_equal(model_only.predict(pipe.named_steps["scale"].transform(x_new)), pipe.predict(x_new))
    assert not np.array_equal(model_only.predict_proba(x_new), pipe.predict_proba(x_new))
