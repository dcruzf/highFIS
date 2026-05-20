"""sklearn-compatible TSK estimator wrappers.

All public estimator classes are re-exported from their respective sub-modules.
"""

from ._adaptive import (
    AdaTSKClassifierEstimator,
    AdaTSKRegressorEstimator,
    ADPTSKClassifierEstimator,
    ADPTSKRegressorEstimator,
)
from ._base import (
    InputConfig,
    feature_coverage_rate,
)
from ._dg_aletsk import (
    DGALETSKClassifierEstimator,
    DGALETSKRegressorEstimator,
)
from ._dg_tsk import (
    DGTSKClassifierEstimator,
    DGTSKRegressorEstimator,
)
from ._dombi import (
    ADMTSKClassifierEstimator,
    ADMTSKRegressorEstimator,
    DombiTSKClassifierEstimator,
    DombiTSKRegressorEstimator,
)
from ._fsre import (
    FSREAdaTSKClassifierEstimator,
    FSREAdaTSKRegressorEstimator,
)
from ._hdfis import (
    HDFISMinClassifierEstimator,
    HDFISMinRegressorEstimator,
    HDFISProdClassifierEstimator,
    HDFISProdRegressorEstimator,
)
from ._htsk import (
    HTSKClassifierEstimator,
    HTSKRegressorEstimator,
    TSKClassifierEstimator,
    TSKRegressorEstimator,
)
from ._logtsk import (
    LogTSKClassifierEstimator,
    LogTSKRegressorEstimator,
)
from ._mhtsk import (
    MHTSKClassifierEstimator,
    MHTSKRegressorEstimator,
)
from ._yager import (
    AYATSKClassifierEstimator,
    AYATSKRegressorEstimator,
)

__all__: list[str] = [
    "ADMTSKClassifierEstimator",
    "ADMTSKRegressorEstimator",
    "ADPTSKClassifierEstimator",
    "ADPTSKRegressorEstimator",
    "AYATSKClassifierEstimator",
    "AYATSKRegressorEstimator",
    "AdaTSKClassifierEstimator",
    "AdaTSKRegressorEstimator",
    "DGALETSKClassifierEstimator",
    "DGALETSKRegressorEstimator",
    "DGTSKClassifierEstimator",
    "DGTSKRegressorEstimator",
    "DombiTSKClassifierEstimator",
    "DombiTSKRegressorEstimator",
    "FSREAdaTSKClassifierEstimator",
    "FSREAdaTSKRegressorEstimator",
    "HDFISMinClassifierEstimator",
    "HDFISMinRegressorEstimator",
    "HDFISProdClassifierEstimator",
    "HDFISProdRegressorEstimator",
    "HTSKClassifierEstimator",
    "HTSKRegressorEstimator",
    "InputConfig",
    "LogTSKClassifierEstimator",
    "LogTSKRegressorEstimator",
    "MHTSKClassifierEstimator",
    "MHTSKRegressorEstimator",
    "TSKClassifierEstimator",
    "TSKRegressorEstimator",
    "feature_coverage_rate",
]
