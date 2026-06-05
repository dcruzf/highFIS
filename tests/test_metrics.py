from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from sklearn.metrics import explained_variance_score

from highfis.metrics import ClassificationMetrics, RegressionMetrics, Task, compute_metrics


def test_compute_metrics_classification_default() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    result = compute_metrics(
        task="classification",
        y_true=y_true,
        y_pred=y_pred,
    )

    assert set(result) == {
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
    }
    assert np.isclose(result["accuracy"], 0.75)


def test_compute_metrics_regression_default() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.9, 2.1, 2.8])

    result = compute_metrics(
        task="regression",
        y_true=y_true,
        y_pred=y_pred,
    )

    assert set(result) == {
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
    }
    assert np.isclose(result["mse"], np.mean((y_true - y_pred) ** 2))
    assert np.isclose(result["rmse"], np.sqrt(result["mse"]))


def test_classification_metrics_helpers() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    assert ClassificationMetrics.accuracy(y_true, y_pred) == 0.75


def test_regression_metrics_helpers() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.9, 2.1, 2.8])

    assert np.isclose(RegressionMetrics.mse(y_true, y_pred), np.mean((y_true - y_pred) ** 2))
    assert np.isclose(RegressionMetrics.rmse(y_true, y_pred), np.sqrt(np.mean((y_true - y_pred) ** 2)))


def test_flatten_array_ravel() -> None:
    from highfis.metrics import _flatten_array

    values = np.array([[1, 2], [3, 4]])
    flat = _flatten_array(values)

    assert flat.shape == (4,)
    assert np.array_equal(flat, np.array([1, 2, 3, 4]))


def test_compute_metrics_classification_extra_metrics() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    result = compute_metrics(
        task="classification",
        y_true=y_true,
        y_pred=y_pred,
        metrics=[
            "precision_micro",
            "recall_micro",
            "f1_micro",
            "confusion_matrix",
            "classes",
        ],
    )

    assert set(result) == {
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "confusion_matrix",
        "classes",
    }
    assert np.isclose(result["precision_micro"], 0.75)
    assert np.isclose(result["recall_micro"], 0.75)
    assert np.isclose(result["f1_micro"], 0.75)
    assert isinstance(result["confusion_matrix"], np.ndarray)
    assert result["confusion_matrix"].shape == (2, 2)
    assert np.array_equal(result["classes"], np.array([0, 1]))


def test_compute_metrics_validation_rejects_unknown_metrics() -> None:
    with pytest.raises(ValueError, match="Unknown classification metrics"):
        compute_metrics(
            task="classification",
            y_true=[0, 1],
            y_pred=[0, 1],
            metrics=["accuracy", "bad_metric"],
        )

    with pytest.raises(ValueError, match="Unknown regression metrics"):
        compute_metrics(task="regression", y_true=[1.0, 2.0], y_pred=[1.0, 2.0], metrics=["mae", "bad_metric"])


def test_compute_metrics_regression_custom_subset() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.9, 2.1, 2.8])

    result = compute_metrics(task="regression", y_true=y_true, y_pred=y_pred, metrics=["mae", "r2"])

    assert set(result) == {"mae", "r2"}
    assert np.isclose(result["mae"], np.mean(np.abs(y_true - y_pred)))


def test_compute_metrics_regression_extra_metrics() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 2.9, 4.1])

    result = compute_metrics(
        task="regression",
        y_true=y_true,
        y_pred=y_pred,
        metrics=[
            "median_absolute_error",
            "mean_bias_error",
            "max_error",
            "std_error",
            "explained_variance",
            "mape",
            "smape",
            "msle",
            "pearson",
        ],
    )

    assert set(result) == {
        "median_absolute_error",
        "mean_bias_error",
        "max_error",
        "std_error",
        "explained_variance",
        "mape",
        "smape",
        "msle",
        "pearson",
    }
    assert np.isclose(result["median_absolute_error"], np.median(np.abs(y_true - y_pred)))
    assert np.isclose(result["mean_bias_error"], np.mean(y_pred - y_true))
    assert np.isclose(result["max_error"], np.max(np.abs(y_true - y_pred)))
    assert np.isclose(result["std_error"], np.std(y_pred - y_true))
    assert np.isclose(result["explained_variance"], explained_variance_score(y_true, y_pred))
    assert np.isclose(result["mape"], np.mean(np.abs((y_pred - y_true) / y_true)))
    assert np.isclose(
        result["smape"],
        np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))),
    )
    assert np.isclose(result["msle"], np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))
    assert np.isclose(result["pearson"], np.corrcoef(y_true, y_pred)[0, 1])


def test_compute_metrics_msle_returns_nan_for_negative_targets() -> None:
    y_true = np.array([-1.0, -2.0, -3.0])
    y_pred = np.array([0.0, 1.0, 2.0])

    result = compute_metrics(task="regression", y_true=y_true, y_pred=y_pred, metrics=["msle"])

    assert np.isnan(result["msle"])


def test_compute_metrics_pearson_returns_nan_when_constant() -> None:
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([2.0, 2.0, 2.0])

    result = compute_metrics(task="regression", y_true=y_true, y_pred=y_pred, metrics=["pearson"])

    assert np.isnan(result["pearson"])


def test_compute_metrics_rejects_invalid_task() -> None:
    task = cast(Task, "invalid")
    with pytest.raises(ValueError, match="task must be 'classification' or 'regression'"):
        compute_metrics(task=task, y_true=[0], y_pred=[0])
