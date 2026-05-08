"""Evaluation metrics for highFIS estimators.

This module provides a small, sklearn-style evaluation API for both
regression and classification tasks.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
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
    "log_loss",
]
RegressionMetric = Literal["mse", "mae", "rmse", "r2"]
Task = Literal["classification", "regression"]

DEFAULT_CLASSIFICATION_METRICS: list[ClassificationMetric] = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]
DEFAULT_REGRESSION_METRICS: list[RegressionMetric] = ["mse", "mae", "rmse", "r2"]

_CLASSIFICATION_METRIC_NAMES = set(DEFAULT_CLASSIFICATION_METRICS) | {"log_loss"}
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
    def log_loss(y_true: Any, y_prob: Any, sample_weight: Any | None = None) -> float:
        """Return log loss from predicted probabilities."""
        return float(log_loss(y_true, y_prob, sample_weight=sample_weight))


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
    y_prob: Any | None,
    metric_names: list[ClassificationMetric],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    y_true_arr = _flatten_array(y_true)
    y_pred_arr = _flatten_array(y_pred)
    if "log_loss" in metric_names:
        if y_prob is None:
            raise ValueError("y_prob is required for log_loss evaluation")
        y_prob_arr = np.asarray(y_prob)
    else:
        y_prob_arr = None
    return y_true_arr, y_pred_arr, y_prob_arr


def _ensure_regression_inputs(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    return _flatten_array(y_true), _flatten_array(y_pred)


def compute_metrics(
    task: Task,
    y_true: Any,
    y_pred: Any,
    y_prob: Any | None = None,
    sample_weight: Any | None = None,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute a set of named evaluation metrics.

    Args:
        task: ``"classification"`` or ``"regression"``.
        y_true: Ground-truth labels or targets.
        y_pred: Predicted labels or values.
        y_prob: Predicted class probabilities for classification tasks.
        sample_weight: Optional sample weights.
        metrics: Optional list of metric names to compute.

    Returns:
        Dictionary mapping metric names to scalar float results.
    """
    if task == "classification":
        metric_names = (
            _validate_classification_metrics(metrics) if metrics is not None else DEFAULT_CLASSIFICATION_METRICS
        )
        y_true_arr, y_pred_arr, y_prob_arr = _ensure_classification_inputs(
            y_true,
            y_pred,
            y_prob,
            metric_names,
        )
        results: dict[str, float] = {}
        for metric in metric_names:
            if metric == "accuracy":
                results[metric] = ClassificationMetrics.accuracy(
                    y_true_arr,
                    y_pred_arr,
                    sample_weight=sample_weight,
                )
            elif metric == "balanced_accuracy":
                results[metric] = ClassificationMetrics.balanced_accuracy(
                    y_true_arr,
                    y_pred_arr,
                    sample_weight=sample_weight,
                )
            elif metric == "precision_macro":
                results[metric] = ClassificationMetrics.precision_macro(
                    y_true_arr,
                    y_pred_arr,
                    sample_weight=sample_weight,
                )
            elif metric == "recall_macro":
                results[metric] = ClassificationMetrics.recall_macro(
                    y_true_arr,
                    y_pred_arr,
                    sample_weight=sample_weight,
                )
            elif metric == "f1_macro":
                results[metric] = ClassificationMetrics.f1_macro(
                    y_true_arr,
                    y_pred_arr,
                    sample_weight=sample_weight,
                )
            elif metric == "log_loss":
                results[metric] = ClassificationMetrics.log_loss(
                    y_true_arr,
                    y_prob_arr,
                    sample_weight=sample_weight,
                )
        return results

    if task == "regression":
        metric_names = _validate_regression_metrics(metrics) if metrics is not None else DEFAULT_REGRESSION_METRICS
        y_true_arr, y_pred_arr = _ensure_regression_inputs(y_true, y_pred)
        results = {
            "mse": RegressionMetrics.mse(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "mae": RegressionMetrics.mae(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "rmse": RegressionMetrics.rmse(y_true_arr, y_pred_arr, sample_weight=sample_weight),
            "r2": RegressionMetrics.r2(y_true_arr, y_pred_arr, sample_weight=sample_weight),
        }
        return {key: results[key] for key in metric_names}

    raise ValueError("task must be 'classification' or 'regression'")
