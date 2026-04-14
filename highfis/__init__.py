"""highFIS public API."""

from .base import BaseTSK
from .defuzzifiers import LogSumDefuzzifier, SoftmaxLogDefuzzifier, SumBasedDefuzzifier
from .estimators import (
    HTSKClassifierEstimator,
    HTSKRegressorEstimator,
    InputConfig,
    LogTSKClassifierEstimator,
    LogTSKRegressorEstimator,
    TSKClassifierEstimator,
    TSKRegressorEstimator,
)
from .memberships import (
    BellMF,
    GaussianMF,
    MembershipFunction,
    SigmoidalMF,
    TrapezoidalMF,
    TriangularMF,
)
from .models import (
    HTSKClassifier,
    HTSKRegressor,
    LogTSKClassifier,
    LogTSKRegressor,
    TSKClassifier,
    TSKRegressor,
)
from .protocols import ConsequentFn, Defuzzifier, MembershipFn, TNorm

__version__ = "0.4.0"

__all__ = [
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
]
