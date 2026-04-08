from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .memberships import GaussianMF, MembershipFunction
from .models import HTSKClassifier


@dataclass(frozen=True)
class InputConfig:
    """Input configuration for Gaussian MF initialization."""

    name: str
    n_mfs: int = 3
    overlap: float = 0.5
    margin: float = 0.10


def _build_gaussian_input_mfs(
    x: np.ndarray,
    input_configs: list[InputConfig],
) -> dict[str, list[MembershipFunction]]:
    """Build Gaussian MFs per input using grid initialization."""
    input_mfs: dict[str, list[MembershipFunction]] = {}
    for idx, cfg in enumerate(input_configs):
        if cfg.n_mfs < 1:
            raise ValueError(f"n_mfs for '{cfg.name}' must be >= 1")

        x_col = x[:, idx]
        x_min = float(np.min(x_col))
        x_max = float(np.max(x_col))
        pad = (x_max - x_min) * float(cfg.margin)
        rmin = x_min - pad
        rmax = x_max + pad
        if rmax <= rmin:
            rmax = rmin + 1e-3

        centers = np.linspace(rmin, rmax, int(cfg.n_mfs), dtype=np.float64)
        if cfg.n_mfs == 1:
            width = max((rmax - rmin), 1e-3)
        else:
            spacing = (rmax - rmin) / float(cfg.n_mfs - 1)
            width = max(spacing * (1.0 + float(cfg.overlap)), 1e-3)
        sigma = max(width / 2.0, 1e-3)

        input_mfs[cfg.name] = [GaussianMF(mean=float(c), sigma=float(sigma)) for c in centers]

    return input_mfs


class HTSKClassifierEstimator(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """High-level HTSK classifier facade with sklearn-compatible API."""

    def __init__(
        self,
        *,
        input_configs: list[InputConfig] | None = None,
        n_mfs: int = 3,
        random_state: int | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        verbose: bool = False,
        rule_base: str = "cartesian",
        batch_size: int | None = None,
        shuffle: bool = True,
        ur_weight: float = 0.0,
        ur_target: float | None = None,
        consequent_batch_norm: bool = False,
    ) -> None:
        """Configure estimator hyperparameters and training options."""
        self.input_configs = input_configs
        self.n_mfs = n_mfs
        self.random_state = random_state
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.rule_base = rule_base
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ur_weight = ur_weight
        self.ur_target = ur_target
        self.consequent_batch_norm = consequent_batch_norm

    def _resolve_input_configs(self, x: np.ndarray) -> list[InputConfig]:
        if self.input_configs is not None:
            if len(self.input_configs) != x.shape[1]:
                raise ValueError(
                    f"input_configs length ({len(self.input_configs)}) must match number of features ({x.shape[1]})"
                )
            return list(self.input_configs)
        return [InputConfig(name=f"x{i + 1}", n_mfs=int(self.n_mfs)) for i in range(x.shape[1])]

    @staticmethod
    def _as_tensor_x(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32)

    def fit(self, x: Any, y: Any) -> HTSKClassifierEstimator:
        """Train the HTSK classifier on labeled samples."""
        x_arr, y_arr = check_X_y(x, y)

        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        le = LabelEncoder()
        y_idx = le.fit_transform(np.asarray(y_arr))

        input_configs = self._resolve_input_configs(x_arr)
        input_mfs = _build_gaussian_input_mfs(x_arr, input_configs)

        self.n_features_in_ = x_arr.shape[1]
        self.feature_names_in_ = np.asarray([cfg.name for cfg in input_configs], dtype=object)
        self.classes_ = le.classes_
        self._label_encoder_ = le

        self.model_ = HTSKClassifier(
            input_mfs,
            n_classes=len(self.classes_),
            rule_base=str(self.rule_base),
            consequent_batch_norm=bool(self.consequent_batch_norm),
        )

        y_t = torch.as_tensor(y_idx, dtype=torch.long)
        self.history_ = self.model_.fit(
            self._as_tensor_x(x_arr),
            y_t,
            epochs=int(self.epochs),
            learning_rate=float(self.learning_rate),
            batch_size=self.batch_size,
            shuffle=bool(self.shuffle),
            ur_weight=float(self.ur_weight),
            ur_target=self.ur_target,
            verbose=bool(self.verbose),
        )
        return self

    def predict_proba(self, x: Any) -> np.ndarray:
        """Predict class probabilities for input samples."""
        check_is_fitted(self, "model_")
        x_arr = check_array(x)
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x_arr.shape[1]}")
        probs = self.model_.predict_proba(self._as_tensor_x(x_arr))
        return probs.detach().cpu().numpy()

    def predict(self, x: Any) -> np.ndarray:
        """Predict class labels for input samples."""
        proba = self.predict_proba(x)
        y_idx = np.argmax(proba, axis=1)
        return np.asarray(self._label_encoder_.inverse_transform(y_idx))

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return classification accuracy on the provided dataset."""
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))


__all__ = ["InputConfig", "HTSKClassifierEstimator"]
