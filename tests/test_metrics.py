from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import median_absolute_error as sk_median_absolute_error

from highfis.metrics import (
    ClassificationMetrics,
    ClassificationMetricsPytorch,
    RegressionMetrics,
    RegressionMetricsPytorch,
    Task,
    compute_metrics,
    compute_metrics_pytorch,
)


def test_compute_metrics_classification_default() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    result = compute_metrics(task="classification", y_true=y_true, y_pred=y_pred)
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
    result = compute_metrics(task="regression", y_true=y_true, y_pred=y_pred)
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
        metrics=["precision_micro", "recall_micro", "f1_micro", "confusion_matrix", "classes"],
    )
    assert set(result) == {"precision_micro", "recall_micro", "f1_micro", "confusion_matrix", "classes"}
    assert np.isclose(result["precision_micro"], 0.75)
    assert np.isclose(result["recall_micro"], 0.75)
    assert np.isclose(result["f1_micro"], 0.75)
    assert isinstance(result["confusion_matrix"], np.ndarray)
    assert result["confusion_matrix"].shape == (2, 2)
    assert np.array_equal(result["classes"], np.array([0, 1]))


def test_compute_metrics_validation_rejects_unknown_metrics() -> None:
    with pytest.raises(ValueError, match="Unknown classification metrics"):
        compute_metrics(task="classification", y_true=[0, 1], y_pred=[0, 1], metrics=["accuracy", "bad_metric"])
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
    assert np.isclose(result["std_error"], np.std(y_pred - y_true, ddof=1))
    assert np.isclose(result["explained_variance"], explained_variance_score(y_true, y_pred))
    assert np.isclose(result["mape"], np.mean(np.abs((y_pred - y_true) / y_true)))
    assert np.isclose(result["smape"], np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))
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


@pytest.mark.filterwarnings("ignore:sample_weight is ignored")
def test_compute_metrics_pytorch_matches_numpy() -> None:
    import torch

    # 1. Classification
    y_true_np = np.array([0, 1, 2, 1, 0, 2, 0, 1])
    y_pred_np = np.array([0, 2, 2, 1, 1, 2, 0, 0])
    weights_np = np.array([1.0, 0.5, 2.0, 1.5, 1.0, 0.8, 1.2, 0.5])

    y_true_torch = torch.tensor(y_true_np)
    y_pred_torch = torch.tensor(y_pred_np)
    weights_torch = torch.tensor(weights_np)

    # Check default metrics without weights
    res_np = compute_metrics(task="classification", y_true=y_true_np, y_pred=y_pred_np)
    res_torch = compute_metrics(task="classification", y_true=y_true_torch, y_pred=y_pred_torch)
    assert set(res_np) == set(res_torch)
    for k in res_np:
        if isinstance(res_np[k], np.ndarray):
            assert np.allclose(res_np[k], res_torch[k])
        else:
            assert np.isclose(res_np[k], res_torch[k], atol=1e-6)

    # Check default metrics with weights
    res_np_w = compute_metrics(task="classification", y_true=y_true_np, y_pred=y_pred_np, sample_weight=weights_np)
    res_torch_w = compute_metrics(
        task="classification", y_true=y_true_torch, y_pred=y_pred_torch, sample_weight=weights_torch
    )
    for k in res_np_w:
        if isinstance(res_np_w[k], np.ndarray):
            assert np.allclose(res_np_w[k], res_torch_w[k])
        else:
            assert np.isclose(res_np_w[k], res_torch_w[k], atol=1e-6)

    # 2. Regression
    y_true_reg_np = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
    y_pred_reg_np = np.array([1.1, 2.5, 3.2, 4.5, 6.0])
    weights_reg_np = np.array([1.0, 1.5, 0.5, 2.0, 1.0])

    y_true_reg_torch = torch.tensor(y_true_reg_np)
    y_pred_reg_torch = torch.tensor(y_pred_reg_np)
    weights_reg_torch = torch.tensor(weights_reg_np)

    # Check default regression metrics without weights
    res_reg_np = compute_metrics(task="regression", y_true=y_true_reg_np, y_pred=y_pred_reg_np)
    res_reg_torch = compute_metrics(task="regression", y_true=y_true_reg_torch, y_pred=y_pred_reg_torch)
    assert set(res_reg_np) == set(res_reg_torch)
    for k in res_reg_np:
        assert np.isclose(res_reg_np[k], res_reg_torch[k], atol=1e-6)

    # Check default regression metrics with weights
    res_reg_np_w = compute_metrics(
        task="regression", y_true=y_true_reg_np, y_pred=y_pred_reg_np, sample_weight=weights_reg_np
    )
    res_reg_torch_w = compute_metrics(
        task="regression", y_true=y_true_reg_torch, y_pred=y_pred_reg_torch, sample_weight=weights_reg_torch
    )
    for k in res_reg_np_w:
        assert np.isclose(res_reg_np_w[k], res_reg_torch_w[k], atol=1e-6)


def test_pytorch_metrics_explicitly_against_sklearn() -> None:
    # Classification
    y_true_cls = torch.tensor([0, 1, 2, 1, 0, 2])
    y_pred_cls = torch.tensor([0, 2, 2, 1, 1, 2])
    w_cls = torch.tensor([1.0, 0.8, 1.2, 0.5, 2.0, 1.1])

    y_true_cls_np = y_true_cls.numpy()
    y_pred_cls_np = y_pred_cls.numpy()
    w_cls_np = w_cls.numpy()

    # Accuracy
    assert np.isclose(
        ClassificationMetricsPytorch.accuracy(y_true_cls, y_pred_cls), accuracy_score(y_true_cls_np, y_pred_cls_np)
    )
    assert np.isclose(
        ClassificationMetricsPytorch.accuracy(y_true_cls, y_pred_cls, w_cls),
        accuracy_score(y_true_cls_np, y_pred_cls_np, sample_weight=w_cls_np),
    )

    # Balanced Accuracy
    assert np.isclose(
        ClassificationMetricsPytorch.balanced_accuracy(y_true_cls, y_pred_cls),
        balanced_accuracy_score(y_true_cls_np, y_pred_cls_np),
    )
    assert np.isclose(
        ClassificationMetricsPytorch.balanced_accuracy(y_true_cls, y_pred_cls, w_cls),
        balanced_accuracy_score(y_true_cls_np, y_pred_cls_np, sample_weight=w_cls_np),
    )

    # Precision/Recall/F1 Macro
    assert np.isclose(
        ClassificationMetricsPytorch.precision_macro(y_true_cls, y_pred_cls),
        precision_score(y_true_cls_np, y_pred_cls_np, average="macro"),
    )
    assert np.isclose(
        ClassificationMetricsPytorch.precision_macro(y_true_cls, y_pred_cls, w_cls),
        precision_score(y_true_cls_np, y_pred_cls_np, sample_weight=w_cls_np, average="macro"),
    )
    assert np.isclose(
        ClassificationMetricsPytorch.recall_macro(y_true_cls, y_pred_cls),
        recall_score(y_true_cls_np, y_pred_cls_np, average="macro"),
    )
    assert np.isclose(
        ClassificationMetricsPytorch.recall_macro(y_true_cls, y_pred_cls, w_cls),
        recall_score(y_true_cls_np, y_pred_cls_np, sample_weight=w_cls_np, average="macro"),
    )
    assert np.isclose(
        ClassificationMetricsPytorch.f1_macro(y_true_cls, y_pred_cls),
        f1_score(y_true_cls_np, y_pred_cls_np, average="macro"),
    )
    assert np.isclose(
        ClassificationMetricsPytorch.f1_macro(y_true_cls, y_pred_cls, w_cls),
        f1_score(y_true_cls_np, y_pred_cls_np, sample_weight=w_cls_np, average="macro"),
    )

    # Precision/Recall/F1 Micro
    assert np.isclose(
        ClassificationMetricsPytorch.precision_micro(y_true_cls, y_pred_cls),
        precision_score(y_true_cls_np, y_pred_cls_np, average="micro"),
    )
    assert np.isclose(
        ClassificationMetricsPytorch.precision_micro(y_true_cls, y_pred_cls, w_cls),
        precision_score(y_true_cls_np, y_pred_cls_np, sample_weight=w_cls_np, average="micro"),
    )
    assert np.isclose(
        ClassificationMetricsPytorch.recall_micro(y_true_cls, y_pred_cls),
        recall_score(y_true_cls_np, y_pred_cls_np, average="micro"),
    )
    assert np.isclose(
        ClassificationMetricsPytorch.recall_micro(y_true_cls, y_pred_cls, w_cls),
        recall_score(y_true_cls_np, y_pred_cls_np, sample_weight=w_cls_np, average="micro"),
    )
    assert np.isclose(
        ClassificationMetricsPytorch.f1_micro(y_true_cls, y_pred_cls),
        f1_score(y_true_cls_np, y_pred_cls_np, average="micro"),
    )
    assert np.isclose(
        ClassificationMetricsPytorch.f1_micro(y_true_cls, y_pred_cls, w_cls),
        f1_score(y_true_cls_np, y_pred_cls_np, sample_weight=w_cls_np, average="micro"),
    )

    # Confusion matrix
    assert np.allclose(
        ClassificationMetricsPytorch.confusion_matrix(y_true_cls, y_pred_cls),
        sk_confusion_matrix(y_true_cls_np, y_pred_cls_np),
    )
    assert np.allclose(
        ClassificationMetricsPytorch.confusion_matrix(y_true_cls, y_pred_cls, w_cls),
        sk_confusion_matrix(y_true_cls_np, y_pred_cls_np, sample_weight=w_cls_np),
    )

    # Regression
    y_true_reg = torch.tensor([1.5, 2.3, 3.8, 4.2])
    y_pred_reg = torch.tensor([1.2, 2.5, 3.5, 4.9])
    w_reg = torch.tensor([1.0, 0.5, 2.0, 1.5])

    y_true_reg_np = y_true_reg.numpy()
    y_pred_reg_np = y_pred_reg.numpy()
    w_reg_np = w_reg.numpy()

    # MSE/MAE/RMSE/R2
    assert np.isclose(
        RegressionMetricsPytorch.mse(y_true_reg, y_pred_reg), mean_squared_error(y_true_reg_np, y_pred_reg_np)
    )
    assert np.isclose(
        RegressionMetricsPytorch.mse(y_true_reg, y_pred_reg, w_reg),
        mean_squared_error(y_true_reg_np, y_pred_reg_np, sample_weight=w_reg_np),
    )
    assert np.isclose(
        RegressionMetricsPytorch.mae(y_true_reg, y_pred_reg), mean_absolute_error(y_true_reg_np, y_pred_reg_np)
    )
    assert np.isclose(
        RegressionMetricsPytorch.mae(y_true_reg, y_pred_reg, w_reg),
        mean_absolute_error(y_true_reg_np, y_pred_reg_np, sample_weight=w_reg_np),
    )
    assert np.isclose(
        RegressionMetricsPytorch.rmse(y_true_reg, y_pred_reg), np.sqrt(mean_squared_error(y_true_reg_np, y_pred_reg_np))
    )
    assert np.isclose(
        RegressionMetricsPytorch.rmse(y_true_reg, y_pred_reg, w_reg),
        np.sqrt(mean_squared_error(y_true_reg_np, y_pred_reg_np, sample_weight=w_reg_np)),
    )
    assert np.isclose(RegressionMetricsPytorch.r2(y_true_reg, y_pred_reg), r2_score(y_true_reg_np, y_pred_reg_np))
    assert np.isclose(
        RegressionMetricsPytorch.r2(y_true_reg, y_pred_reg, w_reg),
        r2_score(y_true_reg_np, y_pred_reg_np, sample_weight=w_reg_np),
    )

    # Median Absolute Error
    assert np.isclose(
        RegressionMetricsPytorch.median_absolute_error(y_true_reg, y_pred_reg),
        sk_median_absolute_error(y_true_reg_np, y_pred_reg_np),
    )
    assert np.isclose(
        RegressionMetricsPytorch.median_absolute_error(y_true_reg, y_pred_reg, w_reg),
        sk_median_absolute_error(y_true_reg_np, y_pred_reg_np, sample_weight=w_reg_np),
    )

    # Explained Variance
    assert np.isclose(
        RegressionMetricsPytorch.explained_variance(y_true_reg, y_pred_reg),
        explained_variance_score(y_true_reg_np, y_pred_reg_np),
    )
    assert np.isclose(
        RegressionMetricsPytorch.explained_variance(y_true_reg, y_pred_reg, w_reg),
        explained_variance_score(y_true_reg_np, y_pred_reg_np, sample_weight=w_reg_np),
    )


def test_pytorch_metrics_edge_cases() -> None:
    import torch

    from highfis.metrics import compute_metrics_pytorch

    # 1. MSLE with negative prediction/target in PyTorch
    y_true_neg = torch.tensor([-1.0, 2.0])
    y_pred_neg = torch.tensor([1.0, 2.0])
    res_neg = compute_metrics(task="regression", y_true=y_true_neg, y_pred=y_pred_neg, metrics=["msle"])
    assert np.isnan(res_neg["msle"])

    # 2. Pearson with < 2 elements in PyTorch
    y_true_small = torch.tensor([1.0])
    y_pred_small = torch.tensor([2.0])
    res_small = compute_metrics(task="regression", y_true=y_true_small, y_pred=y_pred_small, metrics=["pearson"])
    assert np.isnan(res_small["pearson"])

    # 3. Pearson with zero std (constant values) in PyTorch
    y_true_const = torch.tensor([1.0, 1.0])
    y_pred_const = torch.tensor([2.0, 3.0])
    res_const = compute_metrics(task="regression", y_true=y_true_const, y_pred=y_pred_const, metrics=["pearson"])
    assert np.isnan(res_const["pearson"])

    # 4. Multi-dimensional inputs for ravel testing in compute_metrics_pytorch
    y_true_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_pred_2d = torch.tensor([[1.1, 1.9], [3.1, 3.9]])
    weight_2d = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    res_2d = compute_metrics_pytorch(
        task="regression", y_true=y_true_2d, y_pred=y_pred_2d, sample_weight=weight_2d, metrics=["mse"]
    )
    assert "mse" in res_2d

    # 5. Invalid task in compute_metrics_pytorch
    with pytest.raises(ValueError, match="task must be 'classification' or 'regression'"):
        compute_metrics_pytorch(task=cast(Task, "invalid"), y_true=torch.tensor([1.0]), y_pred=torch.tensor([1.0]))

    # 6. compute_metrics with y_true, y_pred as torch.Tensor but sample_weight as a list (non-torch.Tensor)
    y_true_t = torch.tensor([1.0, 2.0])
    y_pred_t = torch.tensor([1.1, 1.9])
    res_weight_list = compute_metrics(
        task="regression", y_true=y_true_t, y_pred=y_pred_t, sample_weight=[1.0, 2.0], metrics=["mse"]
    )
    assert "mse" in res_weight_list


@pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true")
def test_pytorch_metrics_randomized_against_sklearn() -> None:
    from highfis.metrics import ClassificationMetricsPytorch

    rng = np.random.default_rng(12345)

    for _ in range(100):
        n_samples = rng.integers(5, 50)
        n_classes = rng.integers(2, 5)

        y_true = rng.integers(0, n_classes, size=n_samples)
        y_pred = rng.integers(0, n_classes, size=n_samples)

        # Introduce missing classes sometimes
        if rng.random() > 0.5:
            missing_c = rng.integers(0, n_classes)
            y_pred[y_pred == missing_c] = (missing_c + 1) % n_classes

        has_w = rng.random() > 0.5
        w = rng.uniform(0.1, 2.0, size=n_samples) if has_w else None

        y_true_t = torch.tensor(y_true)
        y_pred_t = torch.tensor(y_pred)
        w_t = torch.tensor(w) if w is not None else None

        for avg in ["macro", "micro"]:
            sk_prec = precision_score(y_true, y_pred, average=avg, zero_division=0, sample_weight=w)
            sk_rec = recall_score(y_true, y_pred, average=avg, zero_division=0, sample_weight=w)
            sk_f1 = f1_score(y_true, y_pred, average=avg, zero_division=0, sample_weight=w)

            py_prec_fn = getattr(ClassificationMetricsPytorch, f"precision_{avg}")
            py_rec_fn = getattr(ClassificationMetricsPytorch, f"recall_{avg}")
            py_f1_fn = getattr(ClassificationMetricsPytorch, f"f1_{avg}")

            py_prec = py_prec_fn(y_true_t, y_pred_t, w_t)
            py_rec = py_rec_fn(y_true_t, y_pred_t, w_t)
            py_f1 = py_f1_fn(y_true_t, y_pred_t, w_t)

            assert np.isclose(sk_prec, py_prec, atol=1e-5)
            assert np.isclose(sk_rec, py_rec, atol=1e-5)
            assert np.isclose(sk_f1, py_f1, atol=1e-5)

        sk_bal = balanced_accuracy_score(y_true, y_pred, sample_weight=w)
        py_bal = ClassificationMetricsPytorch.balanced_accuracy(y_true_t, y_pred_t, w_t)
        assert np.isclose(sk_bal, py_bal, atol=1e-5)


def test_metrics_inconsistent_shapes_raises_error() -> None:
    # 1. Inconsistent shapes
    y_true_3 = np.array([1, 2, 3])
    y_pred_2 = np.array([1, 2])
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        compute_metrics(task="classification", y_true=y_true_3, y_pred=y_pred_2)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        compute_metrics(task="regression", y_true=y_true_3, y_pred=y_pred_2)

    y_true_3_t = torch.tensor([1, 2, 3])
    y_pred_2_t = torch.tensor([1, 2])
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        compute_metrics_pytorch(task="classification", y_true=y_true_3_t, y_pred=y_pred_2_t)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        compute_metrics_pytorch(task="regression", y_true=y_true_3_t, y_pred=y_pred_2_t)


def test_numpy_metrics_warnings_on_direct_calls() -> None:
    # 2. Warnings from direct calls (NumPy)
    with pytest.warns(UserWarning, match="ignored by classes"):
        ClassificationMetrics.classes([1, 2], [1, 2], sample_weight=[1.0, 1.0])
    with pytest.warns(UserWarning, match="ignored by max_error"):
        RegressionMetrics.max_error([1, 2], [1, 2], sample_weight=[1.0, 1.0])
    with pytest.warns(UserWarning, match="ignored by std_error"):
        RegressionMetrics.std_error([1, 2], [1, 2], sample_weight=[1.0, 1.0])
    with pytest.warns(UserWarning, match="ignored by pearson"):
        RegressionMetrics.pearson([1, 2], [1, 2], sample_weight=[1.0, 1.0])


def test_pytorch_metrics_warnings_on_direct_calls() -> None:
    # 3. Warnings from direct calls (PyTorch)
    with pytest.warns(UserWarning, match="ignored by classes"):
        ClassificationMetricsPytorch.classes(
            torch.tensor([1, 2]), torch.tensor([1, 2]), sample_weight=torch.tensor([1.0, 1.0])
        )
    with pytest.warns(UserWarning, match="ignored by max_error"):
        RegressionMetricsPytorch.max_error(
            torch.tensor([1, 2]), torch.tensor([1, 2]), sample_weight=torch.tensor([1.0, 1.0])
        )
    with pytest.warns(UserWarning, match="ignored by std_error"):
        RegressionMetricsPytorch.std_error(
            torch.tensor([1, 2]), torch.tensor([1, 2]), sample_weight=torch.tensor([1.0, 1.0])
        )
    with pytest.warns(UserWarning, match="ignored by pearson"):
        RegressionMetricsPytorch.pearson(
            torch.tensor([1, 2]), torch.tensor([1, 2]), sample_weight=torch.tensor([1.0, 1.0])
        )


def test_compute_metrics_warnings_level() -> None:
    # 4. Warnings from compute_metrics / compute_metrics_pytorch level
    with pytest.warns(UserWarning, match="ignored by classes"):
        compute_metrics(
            task="classification", y_true=[1, 2], y_pred=[1, 2], sample_weight=[1.0, 1.0], metrics=["classes"]
        )
    with pytest.warns(UserWarning, match="ignored by the following regression metrics"):
        compute_metrics(
            task="regression", y_true=[1.0, 2.0], y_pred=[1.1, 1.9], sample_weight=[1.0, 1.0], metrics=["max_error"]
        )
    with pytest.warns(UserWarning, match="ignored by classes"):
        compute_metrics_pytorch(
            task="classification",
            y_true=torch.tensor([1, 2]),
            y_pred=torch.tensor([1, 2]),
            sample_weight=torch.tensor([1.0, 1.0]),
            metrics=["classes"],
        )
    with pytest.warns(UserWarning, match="ignored by the following regression metrics"):
        compute_metrics_pytorch(
            task="regression",
            y_true=torch.tensor([1.0, 2.0]),
            y_pred=torch.tensor([1.1, 1.9]),
            sample_weight=torch.tensor([1.0, 1.0]),
            metrics=["max_error"],
        )


def test_regression_std_error_under_minimum_size() -> None:
    # 5. std_error with size < 2
    assert RegressionMetrics.std_error([1.0], [1.0]) == 0.0
    assert RegressionMetricsPytorch.std_error(torch.tensor([1.0]), torch.tensor([1.0])) == 0.0


def test_pytorch_balanced_accuracy_with_zero_weights() -> None:
    # 6. balanced_accuracy with weights summing to 0
    y_true_cls = torch.tensor([0, 1])
    y_pred_cls = torch.tensor([0, 0])
    w_zero = torch.tensor([0.0, 0.0])
    val = ClassificationMetricsPytorch.balanced_accuracy(y_true_cls, y_pred_cls, w_zero)
    assert val == 0.0


def test_pytorch_r2_under_constant_targets() -> None:
    # 7. r2 when ss_tot == 0 (constant target)
    assert RegressionMetricsPytorch.r2(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])) == 1.0
    assert RegressionMetricsPytorch.r2(torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])) == 0.0


def test_regression_metrics_with_sample_weight_no_warnings() -> None:
    import warnings

    # NumPy
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Treat all warnings as errors
        compute_metrics(
            task="regression",
            y_true=[1.0, 2.0],
            y_pred=[1.1, 1.9],
            sample_weight=[1.0, 1.0],
            metrics=["mse", "mae"],
        )
        compute_metrics(
            task="classification",
            y_true=[0, 1],
            y_pred=[0, 1],
            sample_weight=[1.0, 1.0],
            metrics=["accuracy"],
        )

    # PyTorch
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_metrics_pytorch(
            task="regression",
            y_true=torch.tensor([1.0, 2.0]),
            y_pred=torch.tensor([1.1, 1.9]),
            sample_weight=torch.tensor([1.0, 1.0]),
            metrics=["mse", "mae"],
        )
        compute_metrics_pytorch(
            task="classification",
            y_true=torch.tensor([0, 1]),
            y_pred=torch.tensor([0, 1]),
            sample_weight=torch.tensor([1.0, 1.0]),
            metrics=["accuracy"],
        )


def test_compute_metrics_regression_sample_weight_warning() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.9, 2.1, 2.8])
    sample_weight = np.array([1.0, 1.0, 1.0])

    with pytest.warns(UserWarning, match="sample_weight is ignored by the following regression metrics"):
        compute_metrics(
            task="regression",
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            metrics=["mse", "max_error"],
        )


def test_compute_metrics_pytorch_regression_sample_weight_warning() -> None:
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([0.9, 2.1, 2.8])
    sample_weight = torch.tensor([1.0, 1.0, 1.0])

    with pytest.warns(UserWarning, match="sample_weight is ignored by the following regression metrics"):
        compute_metrics_pytorch(
            task="regression",
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            metrics=["mse", "max_error"],
        )
