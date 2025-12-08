"""Basic tests for DPFederatedKMeans."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from dpfknn import DPFederatedKMeans


def test_basic_clustering():
    """Test basic clustering without privacy."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    kmeans = DPFederatedKMeans(n_clusters=3, n_clients=2, random_state=42)
    kmeans.fit(X)
    
    assert kmeans.cluster_centers_.shape == (3, 2)
    assert kmeans.labels_.shape == (100,)
    assert hasattr(kmeans, 'inertia_')
    assert kmeans.n_iter_ > 0


def test_predict():
    """Test prediction on new data."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    kmeans = DPFederatedKMeans(n_clusters=3, n_clients=2, random_state=42)
    kmeans.fit(X)
    
    X_new = np.array([[0, 0], [10, 10]])
    labels = kmeans.predict(X_new)
    
    assert labels.shape == (2,)
    assert all(0 <= label < 3 for label in labels)


def test_fit_predict():
    """Test fit_predict."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    kmeans = DPFederatedKMeans(n_clusters=3, n_clients=2, random_state=42)
    labels = kmeans.fit_predict(X)
    
    assert labels.shape == (100,)
    assert len(np.unique(labels)) <= 3


def test_transform():
    """Test transform method."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    kmeans = DPFederatedKMeans(n_clusters=3, n_clients=2, random_state=42)
    kmeans.fit(X)
    
    distances = kmeans.transform(X[:10])
    
    assert distances.shape == (10, 3)
    assert np.all(distances >= 0)


def test_score():
    """Test score method."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    kmeans = DPFederatedKMeans(n_clusters=3, n_clients=2, random_state=42)
    kmeans.fit(X)
    
    score = kmeans.score(X)
    
    assert isinstance(score, float)
    assert score <= 0  # Score is negative inertia


def test_with_differential_privacy():
    """Test clustering with differential privacy."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    kmeans = DPFederatedKMeans(
        n_clusters=3,
        n_clients=2,
        epsilon=1.0,
        dp_mechanism='gaussiananalytic',
        random_state=42
    )
    kmeans.fit(X)
    
    assert kmeans.cluster_centers_.shape == (3, 2)
    assert kmeans.labels_.shape == (100,)


def test_with_constraints():
    """Test clustering with constraint methods."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    kmeans = DPFederatedKMeans(
        n_clusters=3,
        n_clients=2,
        epsilon=0.5,
        dp_mechanism='gaussiananalytic',
        constraint_method='diagonal_then_frac',
        post_processing='fold',
        random_state=42
    )
    kmeans.fit(X)
    
    assert kmeans.cluster_centers_.shape == (3, 2)


def test_parameter_validation():
    """Test parameter validation."""
    kmeans = DPFederatedKMeans(n_clusters=-1)
    
    X = np.random.rand(10, 2)
    with pytest.raises(ValueError):
        kmeans.fit(X)


def test_invalid_dp_mechanism():
    """Test invalid DP mechanism."""
    kmeans = DPFederatedKMeans(dp_mechanism='invalid')
    
    X = np.random.rand(10, 2)
    with pytest.raises(ValueError):
        kmeans.fit(X)


def test_reproducibility():
    """Test that results are reproducible with same random state."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    kmeans1 = DPFederatedKMeans(n_clusters=3, n_clients=2, random_state=42)
    kmeans1.fit(X)
    
    kmeans2 = DPFederatedKMeans(n_clusters=3, n_clients=2, random_state=42)
    kmeans2.fit(X)
    
    np.testing.assert_array_almost_equal(
        kmeans1.cluster_centers_,
        kmeans2.cluster_centers_
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
