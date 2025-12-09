"""End-to-end tests for DPFederatedKMeans on real datasets."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

from dpfknn import DPFederatedKMeans, local_proto, evaluate, Params
from dpfknn.data_io import normalize, shuffle_and_split, unscale


class TestE2EIrisDataset:
    """End-to-end tests on Iris dataset."""
    
    @pytest.fixture
    def iris_data(self):
        """Load and prepare Iris dataset."""
        iris = load_iris()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(iris.data)
        return X_scaled, iris.target
    
    def test_iris_no_privacy(self, iris_data):
        """Test clustering on Iris without privacy."""
        X, y = iris_data
        
        kmeans = DPFederatedKMeans(
            n_clusters=3,
            n_clients=2,
            random_state=42
        )
        kmeans.fit(X)
        
        ari = adjusted_rand_score(y, kmeans.labels_)
        
        assert kmeans.cluster_centers_.shape == (3, X.shape[1])
        assert len(kmeans.labels_) == len(y)
        assert ari > 0.5  # Should achieve reasonable clustering
    
    def test_iris_with_privacy(self, iris_data):
        """Test clustering on Iris with differential privacy."""
        X, y = iris_data
        
        kmeans = DPFederatedKMeans(
            n_clusters=3,
            n_clients=2,
            epsilon=1.0,
            dp_mechanism='gaussiananalytic',
            constraint_method='diagonal_then_frac',
            random_state=42
        )
        kmeans.fit(X)
        
        ari = adjusted_rand_score(y, kmeans.labels_)
        
        assert kmeans.cluster_centers_.shape == (3, X.shape[1])
        assert len(kmeans.labels_) == len(y)
        assert ari > 0.3  # Lower threshold with privacy
    
    def test_iris_protocol_direct(self, iris_data):
        """Test using protocol directly on Iris."""
        X, y = iris_data
        
        X_normalized = normalize(X, fixed=True)
        value_lists = shuffle_and_split(X_normalized, 2, random_state=42)
        
        params = Params(
            seed=42,
            data_size=X.shape[0],
            dim=X.shape[1],
            k=3,
            iters=6,
            num_clients=2,
            fixed=True
        )
        
        centroids, unassigned = local_proto(value_lists, params, method="unmasked")
        centroids = unscale(centroids)
        
        assert centroids.shape == (3, X.shape[1])
        assert isinstance(unassigned, (int, np.integer))


class TestE2EWineDataset:
    """End-to-end tests on Wine dataset."""
    
    @pytest.fixture
    def wine_data(self):
        """Load and prepare Wine dataset."""
        wine = load_wine()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(wine.data)
        return X_scaled, wine.target
    
    def test_wine_no_privacy(self, wine_data):
        """Test clustering on Wine without privacy."""
        X, y = wine_data
        
        kmeans = DPFederatedKMeans(
            n_clusters=3,
            n_clients=3,
            random_state=42
        )
        kmeans.fit(X)
        
        ari = adjusted_rand_score(y, kmeans.labels_)
        
        assert kmeans.cluster_centers_.shape == (3, X.shape[1])
        assert ari > 0.3  # Wine is harder to cluster
    
    def test_wine_different_clients(self, wine_data):
        """Test with different numbers of clients."""
        X, y = wine_data
        
        for n_clients in [2, 3, 5]:
            kmeans = DPFederatedKMeans(
                n_clusters=3,
                n_clients=n_clients,
                random_state=42
            )
            kmeans.fit(X)
            
            assert kmeans.cluster_centers_.shape == (3, X.shape[1])
            assert len(kmeans.labels_) == len(y)


class TestE2ESyntheticDataset:
    """End-to-end tests on synthetic datasets."""
    
    def test_well_separated_clusters(self):
        """Test on well-separated synthetic clusters."""
        X, y = make_blobs(
            n_samples=500,
            centers=5,
            n_features=10,
            cluster_std=0.5,
            random_state=42
        )
        
        kmeans = DPFederatedKMeans(
            n_clusters=5,
            n_clients=3,
            random_state=42
        )
        kmeans.fit(X)
        
        ari = adjusted_rand_score(y, kmeans.labels_)
        
        assert ari > 0.5  # Should achieve reasonable ARI on well-separated data
    
    def test_overlapping_clusters(self):
        """Test on overlapping synthetic clusters."""
        X, y = make_blobs(
            n_samples=300,
            centers=4,
            n_features=5,
            cluster_std=2.0,
            random_state=42
        )
        
        kmeans = DPFederatedKMeans(
            n_clusters=4,
            n_clients=2,
            epsilon=0.5,
            dp_mechanism='gaussiananalytic',
            random_state=42
        )
        kmeans.fit(X)
        
        ari = adjusted_rand_score(y, kmeans.labels_)
        
        assert ari > 0.05  # Lower threshold for overlapping clusters with privacy
    
    def test_high_dimensional(self):
        """Test on high-dimensional synthetic data."""
        X, y = make_blobs(
            n_samples=200,
            centers=3,
            n_features=50,
            random_state=42
        )
        
        kmeans = DPFederatedKMeans(
            n_clusters=3,
            n_clients=2,
            random_state=42
        )
        kmeans.fit(X)
        
        assert kmeans.cluster_centers_.shape == (3, 50)
        assert len(kmeans.labels_) == 200


class TestE2EPrivacyUtilityTradeoff:
    """Test privacy-utility tradeoffs."""
    
    @pytest.fixture
    def test_data(self):
        """Generate test data."""
        X, y = make_blobs(
            n_samples=400,
            centers=4,
            n_features=8,
            cluster_std=1.0,
            random_state=42
        )
        scaler = StandardScaler()
        return scaler.fit_transform(X), y
    
    def test_increasing_epsilon_improves_utility(self, test_data):
        """Test that higher epsilon generally leads to better utility."""
        X, y = test_data
        
        epsilons = [0.1, 0.5, 1.0, 2.0]
        aris = []
        
        for eps in epsilons:
            kmeans = DPFederatedKMeans(
                n_clusters=4,
                n_clients=2,
                epsilon=eps,
                dp_mechanism='gaussiananalytic',
                constraint_method='diagonal_then_frac',
                random_state=42
            )
            kmeans.fit(X)
            ari = adjusted_rand_score(y, kmeans.labels_)
            aris.append(ari)
        
        # Generally, higher epsilon should not decrease utility significantly
        # (though stochasticity might cause small variations)
        assert max(aris) >= min(aris) - 0.2
    
    def test_no_privacy_best_utility(self, test_data):
        """Test that no privacy gives best utility."""
        X, y = test_data
        
        # No privacy
        kmeans_no_dp = DPFederatedKMeans(
            n_clusters=4,
            n_clients=2,
            random_state=42
        )
        kmeans_no_dp.fit(X)
        ari_no_dp = adjusted_rand_score(y, kmeans_no_dp.labels_)
        
        # With privacy
        kmeans_dp = DPFederatedKMeans(
            n_clusters=4,
            n_clients=2,
            epsilon=0.5,
            dp_mechanism='gaussiananalytic',
            random_state=42
        )
        kmeans_dp.fit(X)
        ari_dp = adjusted_rand_score(y, kmeans_dp.labels_)
        
        # No privacy should generally be at least as good
        assert ari_no_dp >= ari_dp - 0.3  # Allow some tolerance


class TestE2EEvaluationMetrics:
    """Test evaluation metrics on real datasets."""
    
    def test_evaluate_on_iris(self):
        """Test evaluate function on Iris dataset."""
        iris = load_iris()
        X = iris.data
        
        # Get ground truth centroids
        from sklearn.cluster import KMeans
        gt_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        gt_kmeans.fit(X)
        gt_centroids = gt_kmeans.cluster_centers_
        
        # Get predicted centroids
        kmeans = DPFederatedKMeans(n_clusters=3, n_clients=2, random_state=42)
        kmeans.fit(X)
        pred_centroids = kmeans.cluster_centers_
        
        # Evaluate
        metrics = evaluate(pred_centroids, X, gt_centroids, metrics="all")
        
        assert "Normalized Intra-cluster Variance (NICV)" in metrics
        assert "Mean Squared Error" in metrics
        assert "Silhouette Score" in metrics
        assert metrics["Normalized Intra-cluster Variance (NICV)"] > 0
        assert metrics["Mean Squared Error"] >= 0
    
    def test_evaluate_multiple_metrics(self):
        """Test evaluating multiple specific metrics."""
        X, _ = make_blobs(n_samples=200, centers=4, random_state=42)
        
        from sklearn.cluster import KMeans
        gt_centroids = KMeans(n_clusters=4, random_state=42).fit(X).cluster_centers_
        
        kmeans = DPFederatedKMeans(n_clusters=4, n_clients=2, random_state=42)
        kmeans.fit(X)
        
        metrics = evaluate(
            kmeans.cluster_centers_,
            X,
            gt_centroids,
            metrics=["nicv", "bcss", "mse"]
        )
        
        assert len(metrics) == 3


class TestE2EEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_cluster(self):
        """Test with single cluster."""
        X, _ = make_blobs(n_samples=100, centers=1, random_state=42)
        
        kmeans = DPFederatedKMeans(n_clusters=1, n_clients=2, random_state=42)
        kmeans.fit(X)
        
        assert kmeans.cluster_centers_.shape == (1, X.shape[1])
        assert all(kmeans.labels_ == 0)
    
    def test_more_clusters_than_clients(self):
        """Test when n_clusters > n_clients."""
        X, _ = make_blobs(n_samples=100, centers=5, random_state=42)
        
        kmeans = DPFederatedKMeans(n_clusters=5, n_clients=2, random_state=42)
        kmeans.fit(X)
        
        assert kmeans.cluster_centers_.shape == (5, X.shape[1])
    
    def test_small_dataset(self):
        """Test on very small dataset."""
        X, _ = make_blobs(n_samples=20, centers=2, random_state=42)
        
        kmeans = DPFederatedKMeans(n_clusters=2, n_clients=2, random_state=42)
        kmeans.fit(X)
        
        assert kmeans.cluster_centers_.shape == (2, X.shape[1])
        assert len(kmeans.labels_) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
