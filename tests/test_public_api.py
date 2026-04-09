from __future__ import annotations

import highfis


def test_public_api_exports_expected_symbols() -> None:
    assert hasattr(highfis, "HTSKClassifier")
    assert hasattr(highfis, "HTSKClassifierEstimator")
    assert hasattr(highfis, "GaussianMF")
    assert hasattr(highfis, "InputConfig")
