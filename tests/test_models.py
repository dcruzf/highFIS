from __future__ import annotations

import pytest
import torch
from torch import nn

from highfis.memberships import GaussianMF
from highfis.models import HTSKClassifier


def _build_input_mfs(n_inputs: int = 3, n_mfs: int = 2) -> dict[str, list[GaussianMF]]:
    return {
        f"x{i + 1}": [GaussianMF(mean=float(j), sigma=1.0) for j in range(n_mfs)]
        for i in range(n_inputs)
    }


def test_htsk_classifier_init_validates_arguments() -> None:
    with pytest.raises(ValueError, match="input_mfs must not be empty"):
        HTSKClassifier({}, n_classes=2)

    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        HTSKClassifier(_build_input_mfs(), n_classes=1)


def test_htsk_classifier_forward_predict_shapes() -> None:
    model = HTSKClassifier(_build_input_mfs(), n_classes=3)
    x = torch.randn(8, 3)

    logits = model.forward(x)
    proba = model.predict_proba(x)
    pred = model.predict(x)

    assert logits.shape == (8, 3)
    assert proba.shape == (8, 3)
    assert pred.shape == (8,)
    assert torch.allclose(proba.sum(dim=1), torch.ones(8), atol=1e-6)


def test_htsk_classifier_forward_antecedents_row_sum_one() -> None:
    model = HTSKClassifier(_build_input_mfs(), n_classes=2)
    x = torch.randn(6, 3)

    norm_w = model.forward_antecedents(x)

    assert norm_w.ndim == 2
    assert torch.allclose(norm_w.sum(dim=1), torch.ones(6), atol=1e-6)


def test_htsk_classifier_fit_returns_history() -> None:
    torch.manual_seed(1)
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,), dtype=torch.long)

    history = model.fit(x, y, epochs=4, learning_rate=1e-2, batch_size=5, shuffle=True)

    assert set(history.keys()) == {"train", "ur"}
    assert len(history["train"]) == 4
    assert len(history["ur"]) == 4


def test_htsk_classifier_fit_supports_custom_criterion() -> None:
    torch.manual_seed(1)
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,), dtype=torch.long)

    history = model.fit(x, y, epochs=3, criterion=nn.CrossEntropyLoss())

    assert len(history["train"]) == 3


def test_htsk_classifier_fit_validates_inputs() -> None:
    model = HTSKClassifier(_build_input_mfs(n_inputs=2, n_mfs=2), n_classes=2)
    x = torch.randn(10, 2)
    y = torch.randint(0, 2, (10,), dtype=torch.long)

    with pytest.raises(ValueError, match="expected x shape"):
        model.fit(torch.randn(10, 3), y, epochs=1)

    with pytest.raises(ValueError, match="expected y shape"):
        model.fit(x, y.unsqueeze(1), epochs=1)

    with pytest.raises(ValueError, match="ur_weight must be >= 0"):
        model.fit(x, y, epochs=1, ur_weight=-0.1)

    with pytest.raises(ValueError, match="ur_target must be in"):
        model.fit(x, y, epochs=1, ur_target=0.0)
