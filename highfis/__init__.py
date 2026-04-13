"""highFIS public API."""

from .base import BaseTSK
from .defuzzifiers import LogSumDefuzzifier, SoftmaxLogDefuzzifier, SumBasedDefuzzifier
from .estimators import HTSKClassifierEstimator, HTSKRegressorEstimator, InputConfig
from .memberships import (
    BellMF,
    GaussianMF,
    MembershipFunction,
    SigmoidalMF,
    TrapezoidalMF,
    TriangularMF,
)
from .models import HTSKClassifier, HTSKRegressor
from .protocols import ConsequentFn, Defuzzifier, MembershipFn, TNorm

__version__ = "0.3.0"

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
    # Estimators
    "InputConfig",
    "HTSKClassifierEstimator",
    "HTSKRegressorEstimator",
]
