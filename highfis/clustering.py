"""PyTorch-based clustering utilities used by highFIS.

This module provides lightweight, dependency-free implementations of
K-means and Fuzzy C-Means for use in membership-function initialization
and fuzzy rule construction.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Generator, Tensor


def _as_tensor(x: Any) -> Tensor:
    tensor = torch.as_tensor(x)
    if tensor.ndim != 2:
        raise ValueError(f"expected input array with 2 dims, got {tensor.ndim}")
    return tensor.to(torch.get_default_dtype())


def _build_generator(device: torch.device, random_state: int | None) -> Generator:
    generator = torch.Generator(device=device)
    if random_state is not None:
        generator.manual_seed(int(random_state))
    return generator


def _initialize_centroids(x: Tensor, n_clusters: int, generator: Generator) -> Tensor:
    n_samples = x.shape[0]
    if n_clusters <= 0:
        raise ValueError("n_clusters must be > 0")
    if n_clusters > n_samples:
        raise ValueError("n_clusters must be <= number of samples")
    indices = torch.randperm(n_samples, generator=generator, device=x.device)[:n_clusters]
    return x[indices].clone()


class KMeans:
    """Simple PyTorch K-means implementation."""

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        """Initialize the KMeans estimator."""
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.cluster_centers_: Tensor | None = None
        self.labels_: Tensor | None = None
        self.inertia_: float | None = None

    def fit(self, x: Any) -> KMeans:
        """Fit the KMeans model to the input data."""
        x_t = _as_tensor(x)
        best_inertia = float("inf")
        best_centers: Tensor | None = None
        best_labels: Tensor | None = None

        for init in range(max(1, self.n_init)):
            generator = _build_generator(x_t.device, None if self.random_state is None else self.random_state + init)
            centers = _initialize_centroids(x_t, self.n_clusters, generator)

            for _ in range(self.max_iter):
                distances = torch.cdist(x_t, centers, p=2)
                labels = distances.argmin(dim=1)

                new_centers = []
                for idx in range(self.n_clusters):
                    mask = labels == idx
                    if mask.any():
                        new_centers.append(x_t[mask].mean(dim=0))
                    else:
                        fallback_idx = torch.randint(x_t.shape[0], (), device=x_t.device)
                        new_centers.append(x_t[fallback_idx])
                new_centers = torch.stack(new_centers, dim=0)

                center_shift = torch.norm(centers - new_centers, dim=1).max().item()
                centers = new_centers
                if center_shift <= self.tol:
                    break

            distances = torch.cdist(x_t, centers, p=2)
            labels = distances.argmin(dim=1)
            inertia = float((distances[torch.arange(x_t.shape[0]), labels] ** 2).sum().item())

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.clone()
                best_labels = labels.clone()

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, x: Any) -> Tensor:
        """Predict cluster labels for new samples."""
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans instance is not fitted yet")
        x_t = _as_tensor(x)
        distances = torch.cdist(x_t, self.cluster_centers_, p=2)
        return distances.argmin(dim=1)

    def fit_predict(self, x: Any) -> Tensor:
        """Fit the model and return the predicted labels."""
        self.fit(x)
        if self.labels_ is None:
            raise RuntimeError("KMeans did not produce labels")
        return self.labels_


class FuzzyCMeans:
    """PyTorch implementation of Fuzzy C-Means clustering."""

    def __init__(
        self,
        n_clusters: int = 8,
        m: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the FuzzyCMeans estimator."""
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.eps = float(eps)

        self.cluster_centers_: Tensor | None = None
        self.membership_: Tensor | None = None

    def fit(self, x: Any) -> FuzzyCMeans:
        """Fit the Fuzzy C-Means model to the input data."""
        x_t = _as_tensor(x)
        generator = _build_generator(x_t.device, self.random_state)
        n_samples = x_t.shape[0]

        u = torch.rand((n_samples, self.n_clusters), generator=generator, device=x_t.device)
        u /= u.sum(dim=1, keepdim=True)

        for _ in range(self.max_iter):
            u_pow = u.pow(self.m)
            denominator = u_pow.sum(dim=0).view(self.n_clusters, 1)
            centers = (u_pow.T @ x_t) / denominator.clamp_min(self.eps)

            distances = torch.cdist(x_t, centers, p=2)
            zero_mask = distances == 0.0
            if zero_mask.any():
                u = torch.zeros_like(u)
                chosen = zero_mask.float().argmax(dim=1)
                u[torch.arange(n_samples, device=x_t.device), chosen] = 1.0
            else:
                distances = distances.clamp_min(self.eps)
                power = -2.0 / (self.m - 1.0)
                inv = distances.pow(power)
                u = inv / inv.sum(dim=1, keepdim=True)

            if self.cluster_centers_ is not None:
                center_shift = torch.norm(centers - self.cluster_centers_, dim=1).max().item()
            else:
                center_shift = float("inf")
            self.cluster_centers_ = centers
            self.membership_ = u
            if center_shift <= self.tol:
                break

        self.cluster_centers_ = centers
        self.membership_ = u
        return self

    def predict(self, x: Any) -> Tensor:
        """Predict cluster labels for new samples."""
        if self.membership_ is None:
            raise RuntimeError("FuzzyCMeans instance is not fitted yet")
        x_t = _as_tensor(x)
        distances = torch.cdist(x_t, self.cluster_centers_, p=2).clamp_min(self.eps)
        power = -2.0 / (self.m - 1.0)
        inv = distances.pow(power)
        u = inv / inv.sum(dim=1, keepdim=True)
        return u.argmax(dim=1)

    def fit_predict(self, x: Any) -> Tensor:
        """Fit the model and return the predicted labels."""
        self.fit(x)
        if self.membership_ is None:
            raise RuntimeError("FuzzyCMeans did not produce membership values")
        return self.membership_.argmax(dim=1)
