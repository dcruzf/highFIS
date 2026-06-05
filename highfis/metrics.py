"""Evaluation metrics for highFIS estimators.

This module provides a small, sklearn-style evaluation API for both
regression and classification tasks.

Classification Metrics:
    - ``accuracy``: standard accuracy score
    - ``balanced_accuracy``: average recall over classes
    - ``precision_macro``: macro-averaged precision
    - ``recall_macro``: macro-averaged recall
    - ``f1_macro``: macro-averaged F1 score
    - ``precision_micro``: micro-averaged precision
    - ``recall_micro``: micro-averaged recall
    - ``f1_micro``: micro-averaged F1 score
    - ``confusion_matrix``: confusion matrix by class
    - ``classes``: sorted union of true and predicted labels

Regression Metrics:
    - ``mse``: mean squared error
    - ``mae``: mean absolute error
    - ``rmse``: root mean squared error
    - ``r2``: coefficient of determination
    - ``median_absolute_error``: median absolute error
    - ``mean_bias_error``: average prediction bias
    - ``max_error``: maximum absolute error
    - ``std_error``: standard deviation of residuals
    - ``explained_variance``: explained variance score
    - ``mape``: mean absolute percentage error
    - ``smape``: symmetric mean absolute percentage error
    - ``msle``: mean squared logarithmic error
    - ``pearson``: Pearson correlation coefficient

Notes:
    - The module exports ``compute_metrics`` and the helper classes
      ``ClassificationMetrics`` and ``RegressionMetrics``.
    - ``compute_metrics`` validates metric names and returns only the
      requested subset.
    - All metrics accept raw array-like inputs and flatten non-1D arrays.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

ClassificationMetric = Literal[
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "confusion_matrix",
    "classes",
]
RegressionMetric = Literal[
    "mse",
    "mae",
    "rmse",
    "r2",
    "median_absolute_error",
    "mean_bias_error",
    "max_error",
    "std_error",
    "explained_variance",
    "mape",
    "smape",
    "msle",
    "pearson",
]
Task = Literal["classification", "regression"]

DEFAULT_CLASSIFICATION_METRICS: list[ClassificationMetric] = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "confusion_matrix",
    "classes",
]
DEFAULT_REGRESSION_METRICS: list[RegressionMetric] = [
    "mse",
    "mae",
    "rmse",
    "median_absolute_error",
    "mean_bias_error",
    "max_error",
    "std_error",
    "explained_variance",
    "mape",
    "smape",
    "msle",
    "pearson",
    "r2",
]

_CLASSIFICATION_METRIC_NAMES = set(DEFAULT_CLASSIFICATION_METRICS)
_REGRESSION_METRIC_NAMES = set(DEFAULT_REGRESSION_METRICS)


def _flatten_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


class ClassificationMetrics:
    """Standard classification metrics."""

    @staticmethod
    def accuracy(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return classification accuracy."""
        return float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))

    @staticmethod
    def balanced_accuracy(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return the balanced accuracy score."""
        return float(balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight))

    @staticmethod
    def precision_macro(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return macro-averaged precision."""
        return float(precision_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sample_weight))

    @staticmethod
    def recall_macro(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return macro-averaged recall."""
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sample_weight))

    @staticmethod
    def f1_macro(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return macro-averaged F1 score."""
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sample_weight))

    @staticmethod
    def precision_micro(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return micro-averaged precision."""
        return float(precision_score(y_true, y_pred, average="micro", zero_division=0, sample_weight=sample_weight))

    @staticmethod
    def recall_micro(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return micro-averaged recall."""
        return float(recall_score(y_true, y_pred, average="micro", zero_division=0, sample_weight=sample_weight))

    @staticmethod
    def f1_micro(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return micro-averaged F1 score."""
        return float(f1_score(y_true, y_pred, average="micro", zero_division=0, sample_weight=sample_weight))

    @staticmethod
    def confusion_matrix(
        y_true: Any,
        y_pred: Any,
        sample_weight: Any | None = None,
    ) -> np.ndarray:
        """Return the confusion matrix for the predictions."""
        y_true_arr = _flatten_array(y_true)
        y_pred_arr = _flatten_array(y_pred)
        return confusion_matrix(y_true_arr, y_pred_arr, sample_weight=sample_weight)

    @staticmethod
    def classes(
        y_true: Any,
        y_pred: Any,
        sample_weight: Any | None = None,
    ) -> np.ndarray:
        """Return the sorted set of predicted and true classes."""
        y_true_arr = _flatten_array(y_true)
        y_pred_arr = _flatten_array(y_pred)
        return np.unique(np.concatenate([y_true_arr, y_pred_arr]))


class RegressionMetrics:
    """Standard regression metrics."""

    @staticmethod
    def mse(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return mean squared error."""
        return float(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))

    @staticmethod
    def mae(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return mean absolute error."""
        return float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))

    @staticmethod
    def rmse(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return root mean squared error."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)))

    @staticmethod
    def median_absolute_error(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return median absolute error."""
        return float(median_absolute_error(y_true, y_pred, sample_weight=sample_weight))

    @staticmethod
    def mean_bias_error(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return mean bias error (prediction minus truth)."""
        y_true_arr = _flatten_array(y_true)
        y_pred_arr = _flatten_array(y_pred)
        return float(np.mean(y_pred_arr - y_true_arr))

    @staticmethod
    def max_error(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return maximum absolute error."""
        return float(max_error(y_true, y_pred))

    @staticmethod
    def std_error(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return the standard deviation of the errors."""
        y_true_arr = _flatten_array(y_true)
        y_pred_arr = _flatten_array(y_pred)
        return float(np.std(y_pred_arr - y_true_arr))

    @staticmethod
    def explained_variance(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return explained variance."""
        return float(explained_variance_score(y_true, y_pred, sample_weight=sample_weight))

    @staticmethod
    def mape(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return mean absolute percentage error."""
        return float(mean_absolute_percentage_error(y_true, y_pred))

    @staticmethod
    def smape(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return symmetric mean absolute percentage error."""
        y_true_arr = _flatten_array(y_true)
        y_pred_arr = _flatten_array(y_pred)
        numerator = np.abs(y_pred_arr - y_true_arr) * 2.0
        denominator = np.abs(y_true_arr) + np.abs(y_pred_arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denominator == 0.0, 0.0, numerator / denominator)
        return float(np.mean(ratio))

    @staticmethod
    def msle(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return mean squared logarithmic error."""
        try:
            return float(mean_squared_log_error(y_true, y_pred))
        except ValueError:
            return float(np.nan)

    @staticmethod
    def pearson(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return Pearson correlation coefficient."""
        y_true_arr = _flatten_array(y_true)
        y_pred_arr = _flatten_array(y_pred)
        if y_true_arr.size < 2 or np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0:
            return float(np.nan)
        return float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1])

    @staticmethod
    def r2(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return coefficient of determination (R²)."""
        return float(r2_score(y_true, y_pred, sample_weight=sample_weight))


def _validate_classification_metrics(metrics: list[str]) -> list[ClassificationMetric]:
    unknown = set(metrics) - _CLASSIFICATION_METRIC_NAMES
    if unknown:
        raise ValueError(
            f"Unknown classification metrics: {sorted(unknown)}. "
            f"Supported metrics: {sorted(_CLASSIFICATION_METRIC_NAMES)}"
        )
    return [cast(ClassificationMetric, m) for m in metrics if m in _CLASSIFICATION_METRIC_NAMES]


def _validate_regression_metrics(metrics: list[str]) -> list[RegressionMetric]:
    unknown = set(metrics) - _REGRESSION_METRIC_NAMES
    if unknown:
        raise ValueError(
            f"Unknown regression metrics: {sorted(unknown)}. Supported metrics: {sorted(_REGRESSION_METRIC_NAMES)}"
        )
    return [cast(RegressionMetric, m) for m in metrics if m in _REGRESSION_METRIC_NAMES]


def _ensure_classification_inputs(
    y_true: Any,
    y_pred: Any,
    metric_names: list[ClassificationMetric],
) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = _flatten_array(y_true)
    y_pred_arr = _flatten_array(y_pred)
    return y_true_arr, y_pred_arr


def _ensure_regression_inputs(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    return _flatten_array(y_true), _flatten_array(y_pred)


def compute_metrics(
    task: Task,
    y_true: Any,
    y_pred: Any,
    sample_weight: Any | None = None,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Compute a set of named evaluation metrics.

    Args:
        task: ``"classification"`` or ``"regression"``.
        y_true: Ground-truth labels or targets.
        y_pred: Predicted labels or values.
        sample_weight: Optional sample weights.
        metrics: Optional list of metric names to compute.

    Returns:
        Dictionary mapping metric names to scalar float results.
    """
    if task == "classification":
        metric_names = (
            _validate_classification_metrics(metrics) if metrics is not None else DEFAULT_CLASSIFICATION_METRICS
        )
        y_true_arr, y_pred_arr = _ensure_classification_inputs(
            y_true,
            y_pred,
            metric_names,
        )
        results: dict[str, Any] = {}
        for metric in metric_names:
            metric_fn = getattr(ClassificationMetrics, metric)
            results[metric] = metric_fn(y_true_arr, y_pred_arr, sample_weight)
        return results

    if task == "regression":
        metric_names = _validate_regression_metrics(metrics) if metrics is not None else DEFAULT_REGRESSION_METRICS
        y_true_arr, y_pred_arr = _ensure_regression_inputs(y_true, y_pred)
        results = {
            "mse": RegressionMetrics.mse(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "mae": RegressionMetrics.mae(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "rmse": RegressionMetrics.rmse(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "median_absolute_error": RegressionMetrics.median_absolute_error(
                y_true_arr, y_pred_arr, sample_weight=sample_weight
            ),
            "mean_bias_error": RegressionMetrics.mean_bias_error(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "max_error": RegressionMetrics.max_error(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "std_error": RegressionMetrics.std_error(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "explained_variance": RegressionMetrics.explained_variance(
                y_true_arr, y_pred_arr, sample_weight=sample_weight
            ),
            "mape": RegressionMetrics.mape(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "smape": RegressionMetrics.smape(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "msle": RegressionMetrics.msle(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "pearson": RegressionMetrics.pearson(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "r2": RegressionMetrics.r2(y_true_arr, y_pred_arr, sample_weight=sample_weight),
        }
        return {key: results[key] for key in metric_names}

    raise ValueError("task must be 'classification' or 'regression'")
