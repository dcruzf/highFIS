"""Class probabilities of the classifiers, and the warnings of a fit on a data frame."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
import torch

import highfis


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((90, 4)).astype(np.float32)
    return x, (x[:, 0] > 0.5).astype(int) + (x[:, 1] > 0.5).astype(int)


def _outputs(est: Any, x: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        return est.model_(torch.as_tensor(x))


@pytest.mark.parametrize("name", ["HDFISProdClassifier", "ADMTSKClassifier", "AYATSKClassifier"])
def test_squared_error_classifiers_scale_their_outputs(name: str) -> None:
    """Outputs trained on one-hot targets estimate the probabilities: no softmax on top."""
    x, y = _data()
    est = getattr(highfis, name)(epochs=30, random_state=0).fit(x, y)
    proba = est.predict_proba(x)
    output = _outputs(est, x)

    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert proba.min() >= 0.0
    positive = output.clamp(min=0.0)
    has_positive = (positive.sum(dim=1) > 0).numpy()
    expected = (positive / positive.sum(dim=1, keepdim=True).clamp(min=1e-30)).numpy()
    np.testing.assert_allclose(proba[has_positive], expected[has_positive], atol=1e-6)
    # The class predicted is the one with the largest output, as before.
    np.testing.assert_array_equal(est.predict(x), est.classes_[output.argmax(dim=1).numpy()])


def test_outputs_without_a_positive_score_fall_back_to_the_softmax() -> None:
    x, y = _data()
    est = highfis.HDFISProdClassifier(epochs=2, random_state=0).fit(x, y)
    layer: Any = est.model_.consequent_layer
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.fill_(-1.0)
    np.testing.assert_allclose(est.predict_proba(x), 1.0 / 3.0, atol=1e-6)


def test_cross_entropy_classifiers_keep_the_softmax() -> None:
    x, y = _data()
    est = highfis.HTSKClassifier(epochs=10, random_state=0).fit(x, y)
    expected = torch.softmax(_outputs(est, x), dim=1).numpy()
    np.testing.assert_allclose(est.predict_proba(x), expected, atol=1e-6)


def test_a_fit_on_a_data_frame_does_not_warn_about_feature_names() -> None:
    pandas = pytest.importorskip("pandas")
    x, y = _data()
    frame = pandas.DataFrame(x, columns=["a", "b", "c", "d"])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="X does not have valid feature names")
        est = highfis.HTSKClassifier(epochs=3, random_state=0).fit(frame, y)
        est.explain(frame.iloc[[0]])
        est.predict_proba(frame)
