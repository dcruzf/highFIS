"""sklearn-compatible TSK estimator wrappers.

All public estimator classes are re-exported from their respective sub-modules.
"""

from ._adaptive import (
    ADATSKClassifier,
    ADATSKRegressor,
    ADPTSKClassifier,
    ADPTSKRegressor,
)
from ._base import (
    InputConfig,
    feature_coverage_rate,
)
from ._dg_aletsk import (
    DGALETSKClassifier,
    DGALETSKRegressor,
)
from ._dg_tsk import (
    DGTSKClassifier,
    DGTSKRegressor,
)
from ._dombi import (
    ADMTSKClassifier,
    ADMTSKRegressor,
    DombiTSKClassifier,
    DombiTSKRegressor,
)
from ._fsre import (
    FSREADATSKClassifier,
    FSREADATSKRegressor,
)
from ._hdfis import (
    HDFISMinClassifier,
    HDFISMinRegressor,
    HDFISProdClassifier,
    HDFISProdRegressor,
)
from ._htsk import (
    HTSKClassifier,
    HTSKRegressor,
    TSKClassifier,
    TSKRegressor,
)
from ._logtsk import (
    LogTSKClassifier,
    LogTSKRegressor,
)
from ._mhtsk import (
    MHTSKClassifier,
    MHTSKRegressor,
)
from ._yager import (
    AYATSKClassifier,
    AYATSKRegressor,
)

__all__: list[str] = [
    "ADATSKClassifier",
    "ADATSKRegressor",
    "ADMTSKClassifier",
    "ADMTSKRegressor",
    "ADPTSKClassifier",
    "ADPTSKRegressor",
    "AYATSKClassifier",
    "AYATSKRegressor",
    "DGALETSKClassifier",
    "DGALETSKRegressor",
    "DGTSKClassifier",
    "DGTSKRegressor",
    "DombiTSKClassifier",
    "DombiTSKRegressor",
    "FSREADATSKClassifier",
    "FSREADATSKRegressor",
    "HDFISMinClassifier",
    "HDFISMinRegressor",
    "HDFISProdClassifier",
    "HDFISProdRegressor",
    "HTSKClassifier",
    "HTSKRegressor",
    "InputConfig",
    "LogTSKClassifier",
    "LogTSKRegressor",
    "MHTSKClassifier",
    "MHTSKRegressor",
    "TSKClassifier",
    "TSKRegressor",
    "feature_coverage_rate",
]
