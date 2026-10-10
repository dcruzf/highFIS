"""Every estimator predicts the same after being saved and loaded."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import highfis

ESTIMATORS = sorted(name for name in highfis.__all__ if name.endswith(("Classifier", "Regressor")))


@pytest.mark.parametrize("name", ESTIMATORS)
def test_predictions_survive_save_and_load(name: str, tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.random((60, 12)).astype(np.float32)
    target = (x[:, 0] > 0.5).astype(int) if name.endswith("Classifier") else x[:, 0]
    estimator_cls = getattr(highfis, name)
    short = {key: 3 for key in estimator_cls().get_params() if key.endswith("epochs")}
    model = estimator_cls(random_state=0, **short).fit(x, target)

    path = str(tmp_path / "model.pt")
    model.save(path)
    loaded = estimator_cls.load(path)

    np.testing.assert_allclose(loaded.predict(x), model.predict(x), rtol=1e-5, atol=1e-5)
    if name.endswith("Classifier"):
        np.testing.assert_allclose(loaded.predict_proba(x), model.predict_proba(x), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(loaded.rule_activation(x), model.rule_activation(x), rtol=1e-5, atol=1e-5)
