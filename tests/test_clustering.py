from __future__ import annotations

from typing import Any

import pytest
import torch

from highfis.clustering import FuzzyCMeans, KMeans, MiniBatchKMeans
from highfis.estimators._base import _resolve_clusterer


def test_kmeans_class_methods_consistent() -> None:
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]], dtype=torch.float32)
    model = KMeans(n_clusters=2, n_init=3, max_iter=50, random_state=1)
    labels = model.fit_predict(x)
    assert labels.shape == (4,)
    assert set(labels.tolist()) == {0, 1}
    predictions = model.predict(x)
    assert torch.equal(labels, predictions)
    assert model.cluster_centers_ is not None
    assert model.cluster_centers_.shape == (2, 2)
    assert isinstance(model.inertia_, float)


def test_kmeans_fit_predict_raises_when_fit_does_not_set_labels() -> None:
    class BrokenKMeans(KMeans):
        def fit(self, x: Any) -> KMeans:
            return self

    model = BrokenKMeans(n_clusters=2)
    with pytest.raises(RuntimeError, match="KMeans did not produce labels"):
        model.fit_predict(torch.randn(2, 2))


@pytest.mark.parametrize("m", [1.0, 0.5, 0.0])
def test_fcm_rejects_fuzziness_not_greater_than_one(m: float) -> None:
    with pytest.raises(ValueError, match="m must be > 1"):
        FuzzyCMeans(n_clusters=2, m=m)


def test_fcm_class_methods_compute_membership() -> None:
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]], dtype=torch.float32)
    model = FuzzyCMeans(n_clusters=2, m=2.0, max_iter=50, random_state=2)
    labels = model.fit_predict(x)
    assert labels.shape == (4,)
    assert model.membership_ is not None
    assert torch.allclose(model.membership_.sum(dim=1), torch.ones(4), atol=1e-05)
    predictions = model.predict(x)
    assert torch.equal(labels, predictions)


def test_fcm_fit_predict_raises_when_fit_does_not_set_membership() -> None:
    class BrokenFuzzyCMeans(FuzzyCMeans):
        def fit(self, x: Any) -> FuzzyCMeans:
            return self

    model = BrokenFuzzyCMeans(n_clusters=2)
    with pytest.raises(RuntimeError, match="FuzzyCMeans did not produce membership values"):
        model.fit_predict(torch.randn(2, 2))


def test_as_tensor_rejects_non_two_dimensional_input() -> None:
    from highfis.clustering import _as_tensor

    with pytest.raises(ValueError, match="expected input array with 2 dims"):
        _as_tensor(torch.tensor([1.0, 2.0]))


def test_initialize_centroids_invalid_cluster_counts() -> None:
    from highfis.clustering import _build_generator, _initialize_centroids

    x = torch.randn(3, 2)
    generator = _build_generator(x.device, 0)
    with pytest.raises(ValueError, match="n_clusters must be > 0"):
        _initialize_centroids(x, 0, generator)
    with pytest.raises(ValueError, match="n_clusters must be <= number of samples"):
        _initialize_centroids(x, 4, generator)


def test_build_generator_with_random_state_sets_seed() -> None:
    from highfis.clustering import _build_generator

    gen1 = _build_generator(torch.device("cpu"), 42)
    gen2 = _build_generator(torch.device("cpu"), 42)
    a = torch.rand((3,), generator=gen1)
    b = torch.rand((3,), generator=gen2)
    assert torch.allclose(a, b)


def test_build_generator_without_random_state_returns_generator() -> None:
    from highfis.clustering import _build_generator

    generator = _build_generator(torch.device("cpu"), None)
    assert isinstance(generator, torch.Generator)


def test_kmeans_stops_after_max_iter_if_not_converged() -> None:
    x = torch.tensor([[0.0, 0.0], [0.1, 0.0], [100.0, 100.0]], dtype=torch.float32)
    model = KMeans(n_clusters=2, n_init=1, max_iter=1, random_state=4)
    labels = model.fit_predict(x)
    assert labels.shape == (3,)
    assert model.cluster_centers_ is not None
    assert model.cluster_centers_.shape == (2, 2)
    assert model.labels_ is not None


def test_kmeans_handles_empty_cluster_assignments() -> None:
    x = torch.tensor([[0.0, 0.0], [0.0, 0.0], [100.0, 100.0]], dtype=torch.float32)
    model = KMeans(n_clusters=2, n_init=1, max_iter=50, random_state=3)
    labels = model.fit_predict(x)
    assert labels.shape == (3,)
    assert model.cluster_centers_ is not None
    assert model.cluster_centers_.shape == (2, 2)
    assert model.labels_ is not None


def test_kmeans_predict_before_fit_raises() -> None:
    model = KMeans(n_clusters=2)
    with pytest.raises(RuntimeError, match="KMeans instance is not fitted yet"):
        model.predict(torch.randn(2, 2))


def test_minibatch_kmeans_class_methods_consistent_and_empty_cluster_branch() -> None:
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    model = MiniBatchKMeans(n_clusters=3, batch_size=1, max_iter=5, random_state=0)
    model.fit(x)
    assert model.labels_ is not None
    assert model.cluster_centers_ is not None
    assert model.cluster_centers_.shape == (3, 2)
    predictions = model.predict(x)
    assert predictions.shape == (3,)
    assert torch.equal(model.labels_, predictions)


def test_minibatch_kmeans_predict_before_fit_raises() -> None:
    model = MiniBatchKMeans(n_clusters=2)
    with pytest.raises(RuntimeError, match="MiniBatchKMeans instance is not fitted yet"):
        model.predict(torch.randn(2, 2))


def test_resolve_clusterer_supports_minibatch_kmeans_and_fcm() -> None:
    model = _resolve_clusterer("minibatch_kmeans", n_clusters=2, random_state=7)
    assert isinstance(model, MiniBatchKMeans)
    assert model.n_clusters == 2
    assert model.random_state == 7
    model = _resolve_clusterer("fcm", n_clusters=3, random_state=8)
    assert isinstance(model, FuzzyCMeans)
    assert model.n_clusters == 3
    assert model.random_state == 8


def test_resolve_clusterer_overrides_instance_parameters() -> None:
    clusterer = MiniBatchKMeans(n_clusters=1, random_state=123)
    resolved = _resolve_clusterer(clusterer, n_clusters=4, random_state=456)
    assert isinstance(resolved, MiniBatchKMeans)
    assert resolved is not clusterer
    assert resolved.n_clusters == 4
    assert resolved.random_state == 456


def test_resolve_clusterer_rejects_invalid_string() -> None:
    with pytest.raises(ValueError, match="mf_init must be 'kmeans'"):
        _resolve_clusterer("bogus", n_clusters=2, random_state=None)


def test_fcm_predict_before_fit_raises() -> None:
    model = FuzzyCMeans(n_clusters=2, random_state=1)
    with pytest.raises(RuntimeError, match="FuzzyCMeans instance is not fitted yet"):
        model.predict(torch.randn(2, 2))


def test_fcm_zero_distance_branch() -> None:
    x = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    model = FuzzyCMeans(n_clusters=2, random_state=0, max_iter=10)
    labels = model.fit_predict(x)
    assert labels.shape == (3,)
    assert model.cluster_centers_ is not None
    assert model.cluster_centers_.shape == (2, 2)
    assert model.membership_ is not None
    assert torch.allclose(model.membership_.sum(dim=1), torch.ones(3), atol=1e-05)


def test_fcm_runs_complete_iterations_without_early_stop() -> None:
    x = torch.tensor([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=torch.float32)
    model = FuzzyCMeans(n_clusters=2, random_state=0, max_iter=1, tol=0.0)
    labels = model.fit_predict(x)
    assert labels.shape == (3,)
    assert model.cluster_centers_ is not None
    assert model.cluster_centers_.shape == (2, 2)
    assert model.membership_ is not None
    assert torch.allclose(model.membership_.sum(dim=1), torch.ones(3), atol=1e-05)
