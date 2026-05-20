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
    _build_fuzzy_c_means_input_mfs,  # noqa: F401
    _build_gaussian_input_mfs,  # noqa: F401
    _build_kmeans_input_mfs,  # noqa: F401
    _build_mhtsk_input_mfs,  # noqa: F401
    _build_pfrb_input_mfs,  # noqa: F401
    _extract_mhtsk_rule_indices,  # noqa: F401
    _mann_whitney_p_value,  # noqa: F401
    _normalize_importance,  # noqa: F401
    _rankdata,  # noqa: F401
    _resolve_mhtsk_scale_parameters,  # noqa: F401
    _select_rule_indices,  # noqa: F401
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
