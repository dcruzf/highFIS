"""PyTorch-based clustering utilities used by highFIS.

This module provides lightweight, dependency-free implementations of
K-means and Fuzzy C-Means clustering. These estimators are used internally
for membership-function initialization and fuzzy rule construction without
requiring external clustering libraries.

The implementations accept array-like inputs and return PyTorch tensors,
allowing smooth integration with the rest of the highFIS model pipeline.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Generator, Tensor


def _as_tensor(x: Any) -> Tensor:
    """Convert input data to a 2-D PyTorch tensor.

    Args:
        x: Array-like or tensor input representing the samples.

    Returns:
        A 2-D tensor containing the input samples in the current default dtype.

    Raises:
        ValueError: If the input tensor is not two-dimensional.
    """
    tensor = torch.as_tensor(x)
    if tensor.ndim != 2:
        raise ValueError(f"expected input array with 2 dims, got {tensor.ndim}")
    return tensor.to(torch.get_default_dtype())


def _build_generator(device: torch.device, random_state: int | None) -> Generator:
    """Create a reproducible PyTorch random generator for clustering.

    Args:
        device: Torch device used to allocate random tensors.
        random_state: Optional integer seed for deterministic behavior.

    Returns:
        A configured PyTorch random number generator.
    """
    generator = torch.Generator(device=device)
    if random_state is not None:
        generator.manual_seed(int(random_state))
    return generator


def _initialize_centroids(x: Tensor, n_clusters: int, generator: Generator) -> Tensor:
    """Initialize cluster centroids by sampling input points without replacement.

    Args:
        x: Input samples tensor of shape (n_samples, n_features).
        n_clusters: Number of clusters to initialize.
        generator: Random generator for sampling indices.

    Returns:
        Initial centroids tensor of shape (n_clusters, n_features).

    Raises:
        ValueError: If n_clusters is not in the valid range [1, n_samples].
    """
    n_samples = x.shape[0]
    if n_clusters <= 0:
        raise ValueError("n_clusters must be > 0")
    if n_clusters > n_samples:
        raise ValueError("n_clusters must be <= number of samples")
    indices = torch.randperm(n_samples, generator=generator, device=x.device)[:n_clusters]
    return x[indices].clone()


class KMeans:
    """PyTorch implementation of K-means clustering.

    This estimator runs multiple random restarts and supports empty-cluster
    handling by reassigning a fallback sample when a cluster becomes empty.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 1,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        """Initialize the KMeans estimator.

        Args:
            n_clusters: Number of clusters to find.
            n_init: Number of random initializations to try.
            max_iter: Maximum number of iterations per run.
            tol: Convergence tolerance for centroid movement.
            random_state: Optional random seed for reproducibility.
        """
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.cluster_centers_: Tensor | None = None
        self.labels_: Tensor | None = None
        self.inertia_: float | None = None

    @torch.no_grad()
    def fit(self, x: Any) -> KMeans:
        """Fit the KMeans model to the input data.

        Args:
            x: Input samples of shape (n_samples, n_features).

        Returns:
            The fitted estimator.
        """
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

    @torch.no_grad()
    def predict(self, x: Any) -> Tensor:
        """Predict cluster labels for new samples.

        Args:
            x: New samples of shape (n_samples, n_features).

        Returns:
            Cluster labels for each sample.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans instance is not fitted yet")
        x_t = _as_tensor(x)
        distances = torch.cdist(x_t, self.cluster_centers_, p=2)
        return distances.argmin(dim=1)

    @torch.no_grad()
    def fit_predict(self, x: Any) -> Tensor:
        """Fit the model and return the predicted labels.

        Args:
            x: Input samples of shape (n_samples, n_features).

        Returns:
            Cluster labels for each sample.
        """
        self.fit(x)
        if self.labels_ is None:
            raise RuntimeError("KMeans did not produce labels")
        return self.labels_


class MiniBatchKMeans:
    """PyTorch implementation of Mini-Batch K-Means clustering.

    Processes random mini-batches at each iteration instead of the full
    dataset, making it significantly faster than standard K-Means for large
    datasets while producing similar cluster quality.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        batch_size: int = 1024,
        max_iter: int = 100,
        tol: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        """Initialise a :class:`MiniBatchKMeans` instance."""
        self.n_clusters = int(n_clusters)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.cluster_centers_: Tensor | None = None
        self.labels_: Tensor | None = None

    @torch.no_grad()
    def fit(self, x: Any) -> MiniBatchKMeans:
        """Fit the Mini-Batch K-Means model to the input data."""
        x_t = _as_tensor(x)
        n_samples = x_t.shape[0]
        generator = _build_generator(x_t.device, self.random_state)
        centers = _initialize_centroids(x_t, self.n_clusters, generator)

        # Running counts for online centroid updates
        counts = torch.zeros(self.n_clusters, device=x_t.device, dtype=x_t.dtype)
        batch_size = min(self.batch_size, n_samples)

        for _ in range(self.max_iter):
            idx = torch.randperm(n_samples, generator=generator, device=x_t.device)[:batch_size]
            batch = x_t[idx]

            dists = torch.cdist(batch, centers, p=2)
            labels = dists.argmin(dim=1)

            for k in range(self.n_clusters):
                mask = labels == k
                if not mask.any():
                    continue
                n_k = mask.sum()
                counts[k] += n_k
                lr = n_k.float() / counts[k]
                centers[k] += lr * (batch[mask].mean(dim=0) - centers[k])

        # Assign all points to final centers
        dists = torch.cdist(x_t, centers, p=2)
        self.labels_ = dists.argmin(dim=1)
        self.cluster_centers_ = centers
        return self

    @torch.no_grad()
    def predict(self, x: Any) -> Tensor:
        """Predict cluster labels for new samples."""
        if self.cluster_centers_ is None:
            raise RuntimeError("MiniBatchKMeans instance is not fitted yet")
        x_t = _as_tensor(x)
        return torch.cdist(x_t, self.cluster_centers_, p=2).argmin(dim=1)


class FuzzyCMeans:
    """PyTorch implementation of Fuzzy C-Means clustering.

    This estimator produces soft cluster memberships and updates centroids
    using the weighted membership matrix generated by FCM.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        m: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the FuzzyCMeans estimator.

        Args:
            n_clusters: Number of clusters.
            m: Fuzziness parameter. Values > 1 produce fuzzier membership.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance on centroid movement.
            random_state: Optional seed for reproducible initialization.
            eps: Small constant to avoid division by zero.
        """
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.eps = float(eps)

        self.cluster_centers_: Tensor | None = None
        self.membership_: Tensor | None = None

    @torch.no_grad()
    def fit(self, x: Any) -> FuzzyCMeans:
        """Fit the Fuzzy C-Means model to the input data.

        Args:
            x: Input samples of shape (n_samples, n_features).

        Returns:
            The fitted estimator.
        """
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

    @torch.no_grad()
    def predict(self, x: Any) -> Tensor:
        """Predict cluster labels for new samples.

        Args:
            x: New samples of shape (n_samples, n_features).

        Returns:
            Hard cluster labels derived from the current membership matrix.
        """
        if self.membership_ is None:
            raise RuntimeError("FuzzyCMeans instance is not fitted yet")
        x_t = _as_tensor(x)
        distances = torch.cdist(x_t, self.cluster_centers_, p=2).clamp_min(self.eps)
        power = -2.0 / (self.m - 1.0)
        inv = distances.pow(power)
        u = inv / inv.sum(dim=1, keepdim=True)
        return u.argmax(dim=1)

    @torch.no_grad()
    def fit_predict(self, x: Any) -> Tensor:
        """Fit the model and return the predicted labels.

        Args:
            x: Input samples of shape (n_samples, n_features).

        Returns:
            Hard cluster labels for each sample.
        """
        self.fit(x)
        if self.membership_ is None:
            raise RuntimeError("FuzzyCMeans did not produce membership values")
        return self.membership_.argmax(dim=1)
