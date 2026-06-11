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
import torch
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
        if sample_weight is not None:
            return float(np.average(y_pred_arr - y_true_arr, weights=sample_weight))
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
        return float(mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight))

    @staticmethod
    def smape(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return symmetric mean absolute percentage error."""
        y_true_arr = _flatten_array(y_true)
        y_pred_arr = _flatten_array(y_pred)
        numerator = np.abs(y_pred_arr - y_true_arr) * 2.0
        denominator = np.abs(y_true_arr) + np.abs(y_pred_arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denominator == 0.0, 0.0, numerator / denominator)
        if sample_weight is not None:
            return float(np.average(ratio, weights=sample_weight))
        return float(np.mean(ratio))

    @staticmethod
    def msle(y_true: Any, y_pred: Any, sample_weight: Any | None = None) -> float:
        """Return mean squared logarithmic error."""
        try:
            return float(mean_squared_log_error(y_true, y_pred, sample_weight=sample_weight))
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


class ClassificationMetricsPytorch:
    """Standard classification metrics using PyTorch."""

    @staticmethod
    def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return classification accuracy."""
        if sample_weight is not None:
            return (torch.sum((y_true == y_pred).float() * sample_weight) / torch.sum(sample_weight)).item()
        return torch.mean((y_true == y_pred).float()).item()

    @staticmethod
    def balanced_accuracy(
        y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None
    ) -> float:
        """Return the balanced accuracy score."""
        classes = torch.unique(y_true)
        recalls = []
        for c in classes:
            true_c = y_true == c
            if sample_weight is not None:
                w_c = sample_weight[true_c]
                recalls.append(torch.sum((y_pred[true_c] == c).float() * w_c) / torch.sum(w_c))
            else:
                recalls.append(torch.mean((y_pred[true_c] == c).float()))
        return torch.stack(recalls).mean().item() if recalls else 0.0

    @staticmethod
    def _precision_recall_f1_macro(
        y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        classes = torch.unique(torch.cat([y_true, y_pred]))
        precisions = []
        recalls = []
        f1s = []
        for c in classes:
            true_c = y_true == c
            pred_c = y_pred == c
            if sample_weight is not None:
                tp = torch.sum((true_c & pred_c).float() * sample_weight)
                fp = torch.sum((~true_c & pred_c).float() * sample_weight)
                fn = torch.sum((true_c & ~pred_c).float() * sample_weight)
            else:
                tp = (true_c & pred_c).sum().float()
                fp = (~true_c & pred_c).sum().float()
                fn = (true_c & ~pred_c).sum().float()
            prec = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=y_true.device)
            rec = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=y_true.device)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else torch.tensor(0.0, device=y_true.device)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        return (
            torch.stack(precisions).mean() if precisions else torch.tensor(0.0, device=y_true.device),
            torch.stack(recalls).mean() if recalls else torch.tensor(0.0, device=y_true.device),
            torch.stack(f1s).mean() if f1s else torch.tensor(0.0, device=y_true.device),
        )

    @staticmethod
    def precision_macro(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return macro-averaged precision."""
        prec, _, _ = ClassificationMetricsPytorch._precision_recall_f1_macro(y_true, y_pred, sample_weight)
        return prec.item()

    @staticmethod
    def recall_macro(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return macro-averaged recall."""
        _, rec, _ = ClassificationMetricsPytorch._precision_recall_f1_macro(y_true, y_pred, sample_weight)
        return rec.item()

    @staticmethod
    def f1_macro(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return macro-averaged F1 score."""
        _, _, f1 = ClassificationMetricsPytorch._precision_recall_f1_macro(y_true, y_pred, sample_weight)
        return f1.item()

    @staticmethod
    def _precision_recall_f1_micro(
        y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        classes = torch.unique(torch.cat([y_true, y_pred]))
        tp_sum = torch.tensor(0.0, device=y_true.device)
        fp_sum = torch.tensor(0.0, device=y_true.device)
        fn_sum = torch.tensor(0.0, device=y_true.device)
        for c in classes:
            true_c = y_true == c
            pred_c = y_pred == c
            if sample_weight is not None:
                tp_sum += torch.sum((true_c & pred_c).float() * sample_weight)
                fp_sum += torch.sum((~true_c & pred_c).float() * sample_weight)
                fn_sum += torch.sum((true_c & ~pred_c).float() * sample_weight)
            else:
                tp_sum += (true_c & pred_c).sum().float()
                fp_sum += (~true_c & pred_c).sum().float()
                fn_sum += (true_c & ~pred_c).sum().float()
        prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else torch.tensor(0.0, device=y_true.device)
        rec = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else torch.tensor(0.0, device=y_true.device)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else torch.tensor(0.0, device=y_true.device)
        return prec, rec, f1

    @staticmethod
    def precision_micro(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return micro-averaged precision."""
        prec, _, _ = ClassificationMetricsPytorch._precision_recall_f1_micro(y_true, y_pred, sample_weight)
        return prec.item()

    @staticmethod
    def recall_micro(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return micro-averaged recall."""
        _, rec, _ = ClassificationMetricsPytorch._precision_recall_f1_micro(y_true, y_pred, sample_weight)
        return rec.item()

    @staticmethod
    def f1_micro(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return micro-averaged F1 score."""
        _, _, f1 = ClassificationMetricsPytorch._precision_recall_f1_micro(y_true, y_pred, sample_weight)
        return f1.item()

    @staticmethod
    def confusion_matrix(
        y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None
    ) -> np.ndarray:
        """Return the confusion matrix."""
        classes = torch.unique(torch.cat([y_true, y_pred]))
        classes = torch.sort(classes).values
        n_classes = len(classes)

        y_true_idx = torch.searchsorted(classes, y_true)
        y_pred_idx = torch.searchsorted(classes, y_pred)
        indices = y_true_idx * n_classes + y_pred_idx

        if sample_weight is not None:
            cm = torch.bincount(indices, weights=sample_weight, minlength=n_classes**2)
        else:
            cm = torch.bincount(indices, minlength=n_classes**2)

        return cm.reshape(n_classes, n_classes).cpu().numpy()

    @staticmethod
    def classes(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> np.ndarray:
        """Return the sorted union of true and predicted labels."""
        return torch.unique(torch.cat([y_true, y_pred])).cpu().numpy()


class RegressionMetricsPytorch:
    """Standard regression metrics using PyTorch."""

    @staticmethod
    def mse(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return mean squared error."""
        if sample_weight is not None:
            return (torch.sum(((y_true - y_pred) ** 2) * sample_weight) / torch.sum(sample_weight)).item()
        return torch.mean((y_true - y_pred) ** 2).item()

    @staticmethod
    def mae(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return mean absolute error."""
        if sample_weight is not None:
            return (torch.sum(torch.abs(y_true - y_pred) * sample_weight) / torch.sum(sample_weight)).item()
        return torch.mean(torch.abs(y_true - y_pred)).item()

    @staticmethod
    def rmse(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return root mean squared error."""
        if sample_weight is not None:
            return torch.sqrt(torch.sum(((y_true - y_pred) ** 2) * sample_weight) / torch.sum(sample_weight)).item()
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

    @staticmethod
    def r2(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return coefficient of determination (R²)."""
        if sample_weight is not None:
            mean_y = torch.sum(y_true * sample_weight) / torch.sum(sample_weight)
            ss_res = torch.sum(((y_true - y_pred) ** 2) * sample_weight)
            ss_tot = torch.sum(((y_true - mean_y) ** 2) * sample_weight)
        else:
            mean_y = torch.mean(y_true)
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - mean_y) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return (1.0 - ss_res / ss_tot).item()

    @staticmethod
    def median_absolute_error(
        y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None
    ) -> float:
        """Return median absolute error."""
        errors = torch.abs(y_true - y_pred)
        if sample_weight is not None:
            # Weighted median matching scikit-learn
            sorted_indices = torch.argsort(errors)
            sorted_errors = errors[sorted_indices]
            sorted_weights = sample_weight[sorted_indices]
            cum_weights = torch.cumsum(sorted_weights, dim=0)
            total_weight = cum_weights[-1]
            half_weight = total_weight / 2.0
            idx = torch.searchsorted(cum_weights, half_weight)
            if idx < len(sorted_errors) - 1 and cum_weights[idx] == half_weight:
                return ((sorted_errors[idx] + sorted_errors[idx + 1]) / 2.0).item()
            return sorted_errors[idx].item()
        return torch.quantile(errors.float(), 0.5).item()

    @staticmethod
    def mean_bias_error(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return mean bias error (prediction minus truth)."""
        diff = y_pred - y_true
        if sample_weight is not None:
            return (torch.sum(diff * sample_weight) / torch.sum(sample_weight)).item()
        return torch.mean(diff).item()

    @staticmethod
    def max_error(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return maximum absolute error."""
        return torch.max(torch.abs(y_true - y_pred)).item()

    @staticmethod
    def std_error(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return the standard deviation of the errors."""
        return torch.std(y_pred - y_true, unbiased=False).item()

    @staticmethod
    def explained_variance(
        y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None
    ) -> float:
        """Return explained variance."""
        if sample_weight is not None:
            mean_diff = torch.sum((y_true - y_pred) * sample_weight) / torch.sum(sample_weight)
            var_diff = torch.sum(((y_true - y_pred - mean_diff) ** 2) * sample_weight) / torch.sum(sample_weight)
            mean_y = torch.sum(y_true * sample_weight) / torch.sum(sample_weight)
            var_y = torch.sum(((y_true - mean_y) ** 2) * sample_weight) / torch.sum(sample_weight)
        else:
            var_diff = torch.var(y_true - y_pred, unbiased=False)
            var_y = torch.var(y_true, unbiased=False)
        return (1.0 - var_diff / var_y).item() if var_y > 0 else 0.0

    @staticmethod
    def mape(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return mean absolute percentage error."""
        eps = torch.finfo(y_true.dtype).eps
        ratio = torch.abs(y_true - y_pred) / torch.clamp(torch.abs(y_true), min=eps)
        if sample_weight is not None:
            return (torch.sum(ratio * sample_weight) / torch.sum(sample_weight)).item()
        return torch.mean(ratio).item()

    @staticmethod
    def smape(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return symmetric mean absolute percentage error."""
        numerator = torch.abs(y_pred - y_true) * 2.0
        denominator = torch.abs(y_true) + torch.abs(y_pred)
        ratio = torch.where(denominator == 0.0, 0.0, numerator / denominator)
        if sample_weight is not None:
            return (torch.sum(ratio * sample_weight) / torch.sum(sample_weight)).item()
        return torch.mean(ratio).item()

    @staticmethod
    def msle(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return mean squared logarithmic error."""
        if torch.any(y_true < 0) or torch.any(y_pred < 0):
            return float("nan")
        log_diff = torch.log1p(y_true) - torch.log1p(y_pred)
        if sample_weight is not None:
            return (torch.sum((log_diff**2) * sample_weight) / torch.sum(sample_weight)).item()
        return torch.mean(log_diff**2).item()

    @staticmethod
    def pearson(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: torch.Tensor | None = None) -> float:
        """Return Pearson correlation coefficient."""
        if y_true.numel() < 2:
            return float("nan")
        vx = y_true - torch.mean(y_true)
        vy = y_pred - torch.mean(y_pred)
        std_x = torch.std(y_true)
        std_y = torch.std(y_pred)
        if std_x == 0 or std_y == 0:
            return float("nan")
        return (torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))).item()


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


def compute_metrics_pytorch(
    task: Task,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Compute a set of named evaluation metrics using PyTorch."""
    if y_true.ndim != 1:
        y_true = y_true.ravel()
    if y_pred.ndim != 1:
        y_pred = y_pred.ravel()
    if sample_weight is not None and sample_weight.ndim != 1:
        sample_weight = sample_weight.ravel()

    if task == "classification":
        metric_names = (
            _validate_classification_metrics(metrics) if metrics is not None else DEFAULT_CLASSIFICATION_METRICS
        )
        results: dict[str, Any] = {}
        for m in metric_names:
            metric_fn = getattr(ClassificationMetricsPytorch, m)
            results[m] = metric_fn(y_true, y_pred, sample_weight)
        return results

    if task == "regression":
        metric_names = _validate_regression_metrics(metrics) if metrics is not None else DEFAULT_REGRESSION_METRICS
        results = {}
        for m in metric_names:
            metric_fn = getattr(RegressionMetricsPytorch, m)
            results[m] = metric_fn(y_true, y_pred, sample_weight)
        return results

    raise ValueError("task must be 'classification' or 'regression'")


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
    if isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        if sample_weight is not None and not isinstance(sample_weight, torch.Tensor):
            sample_weight = torch.as_tensor(sample_weight, device=y_true.device, dtype=y_true.dtype)
        return compute_metrics_pytorch(task, y_true, y_pred, sample_weight, metrics)

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
