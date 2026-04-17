from __future__ import annotations

import pytest
import torch

from highfis.memberships import CompositeGaussianMF
from highfis.models import AdaTSKClassifier, AdaTSKRegressor


def _build_adatsk_input_mfs(n_inputs: int = 3, n_rules: int = 2) -> dict[str, list[CompositeGaussianMF]]:
    return {
        f"x{i + 1}": [CompositeGaussianMF(mean=float(j), sigma=1.0, eps=1e-4) for j in range(n_rules)]
        for i in range(n_inputs)
    }


def test_composite_gaussian_mf_lower_bound() -> None:
    mf = CompositeGaussianMF(mean=0.0, sigma=1.0, eps=0.05)
    x = torch.tensor([-5.0, 0.0, 5.0])

    values = mf(x)

    assert torch.all(values >= 0.05)
    assert torch.all(values <= 1.0)


def test_adatsk_classifier_forward_predict_shapes() -> None:
    model = AdaTSKClassifier(_build_adatsk_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_adatsk_classifier_forward_antecedents_row_sum_one() -> None:
    model = AdaTSKClassifier(_build_adatsk_input_mfs(n_inputs=2, n_rules=2), n_classes=2)
    x = torch.randn(6, 2)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_adatsk_regressor_forward_shape() -> None:
    model = AdaTSKRegressor(_build_adatsk_input_mfs(n_inputs=2, n_rules=2))
    x = torch.randn(5, 2)

    output = model.forward(x)

    assert output.shape == (5, 1)


def test_adatsk_classifier_fit_returns_history() -> None:
    torch.manual_seed(0)
    model = AdaTSKClassifier(_build_adatsk_input_mfs(n_inputs=2, n_rules=2), n_classes=2)
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,), dtype=torch.long)

    history = model.fit(x, y, epochs=3, learning_rate=1e-2, batch_size=5)

    assert set(history.keys()) == {"train", "ur", "val", "stopped_epoch"}
    assert len(history["train"]) == 3
    assert len(history["ur"]) == 3
    assert history["stopped_epoch"] == 3


def test_adatsk_classifier_rejects_nonpositive_lambda() -> None:
    with pytest.raises(ValueError, match="lambda_init must be > 0"):
        AdaTSKClassifier(_build_adatsk_input_mfs(), n_classes=2, lambda_init=0.0)
