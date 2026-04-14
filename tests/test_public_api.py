"""Verify that all advertised symbols are importable from the public API."""

from __future__ import annotations

import highfis


def test_version_is_0_4_0() -> None:
    assert highfis.__version__ == "0.4.0"


def test_all_symbols_importable() -> None:
    expected = {
        # Protocols
        "MembershipFn",
        "TNorm",
        "Defuzzifier",
        "ConsequentFn",
        # Base
        "BaseTSK",
        # Membership functions
        "MembershipFunction",
        "GaussianMF",
        "TriangularMF",
        "TrapezoidalMF",
        "BellMF",
        "SigmoidalMF",
        # Defuzzifiers
        "SoftmaxLogDefuzzifier",
        "SumBasedDefuzzifier",
        "LogSumDefuzzifier",
        # Models
        "HTSKClassifier",
        "HTSKRegressor",
        "TSKClassifier",
        "TSKRegressor",
        "LogTSKClassifier",
        "LogTSKRegressor",
        # Estimators
        "InputConfig",
        "HTSKClassifierEstimator",
        "HTSKRegressorEstimator",
        "TSKClassifierEstimator",
        "TSKRegressorEstimator",
        "LogTSKClassifierEstimator",
        "LogTSKRegressorEstimator",
    }
    assert set(highfis.__all__) == expected

    for name in expected:
        assert hasattr(highfis, name), f"missing: {name}"
