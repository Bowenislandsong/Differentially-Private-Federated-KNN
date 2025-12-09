"""Example demonstrating direct use of FastLloyd protocol.

This example shows how to use the local_proto function directly
from the FastLloyd implementation, as well as the evaluation metrics.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from dpfknn import local_proto, evaluate, Params
from dpfknn.data_io import normalize, shuffle_and_split, unscale


def run_fastlloyd_protocol_example():
    """Demonstrates using the FastLloyd protocol directly."""
    print("=" * 60)
    print("FastLloyd Protocol Direct Usage Example")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=500, centers=4, n_features=2, 
                           cluster_std=0.6, random_state=42)
    
    print(f"\nGenerated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"True number of clusters: 4")
    
    # Get ground truth centroids
    gt_centroids = KMeans(n_clusters=4, random_state=42).fit(X).cluster_centers_
    
    # Normalize data for FastLloyd
    X_normalized = normalize(X, fixed=True)
    
    # Split data among clients
    n_clients = 3
    value_lists = shuffle_and_split(X_normalized, n_clients, random_state=42)
    
    print(f"\nData split among {n_clients} clients:")
    for i, values in enumerate(value_lists):
        print(f"  Client {i}: {values.shape[0]} samples")
    
    # Configure parameters
    params = Params(
        seed=42,
        data_size=X.shape[0],
        dim=X.shape[1],
        k=4,
        iters=6,
        num_clients=n_clients,
        eps=1.0,
        dp='gaussiananalytic',
        method='diagonal_then_frac',
        post='fold',
        fixed=True
    )
    
    print(f"\n1. Running without privacy (unmasked)...")
    centroids_no_privacy, unassigned = local_proto(
        value_lists, params, method="unmasked"
    )
    centroids_no_privacy = unscale(centroids_no_privacy)
    
    print(f"   Unassigned points: {unassigned}")
    
    # Evaluate clustering quality
    metrics = evaluate(centroids_no_privacy, X, gt_centroids, metrics="all")
    print(f"   NICV: {metrics['Normalized Intra-cluster Variance (NICV)']:.4f}")
    print(f"   MSE: {metrics['Mean Squared Error']:.4f}")
    
    print(f"\n2. Running with privacy (masked)...")
    centroids_privacy, unassigned = local_proto(
        value_lists, params, method="masked"
    )
    centroids_privacy = unscale(centroids_privacy)
    
    print(f"   Unassigned points: {unassigned}")
    
    # Evaluate clustering quality with privacy
    metrics = evaluate(centroids_privacy, X, gt_centroids, metrics="all")
    print(f"   NICV: {metrics['Normalized Intra-cluster Variance (NICV)']:.4f}")
    print(f"   MSE: {metrics['Mean Squared Error']:.4f}")
    
    print(f"\n3. Comparing different privacy budgets...")
    print(f"{'Epsilon':<10} {'Method':<10} {'NICV':<12} {'MSE':<12} {'Unassigned':<10}")
    print("-" * 60)
    
    for eps in [0.1, 0.5, 1.0, 2.0]:
        params.eps = eps
        params.dp = 'gaussiananalytic'
        params.calculate_iters()  # Adjust iterations based on privacy budget
        
        centroids, unassigned = local_proto(value_lists, params, method="masked")
        centroids = unscale(centroids)
        
        metrics = evaluate(centroids, X, gt_centroids, metrics=["nicv", "mse"])
        nicv = metrics['Normalized Intra-cluster Variance (NICV)']
        mse = metrics['Mean Squared Error']
        
        print(f"{eps:<10.1f} {'Masked':<10} {nicv:<12.4f} {mse:<12.4f} {unassigned:<10}")


def demonstrate_evaluation_metrics():
    """Demonstrates using the evaluation metrics."""
    print("\n" + "=" * 60)
    print("Evaluation Metrics Example")
    print("=" * 60)
    
    # Generate data
    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    
    # Get ground truth centroids
    gt_centroids = KMeans(n_clusters=3, random_state=42).fit(X).cluster_centers_
    
    # Get predicted centroids with some noise
    predicted_centroids = gt_centroids + np.random.randn(*gt_centroids.shape) * 0.3
    
    print(f"\nEvaluating clustering quality with all metrics:")
    metrics = evaluate(predicted_centroids, X, gt_centroids, metrics="all")
    
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")


def demonstrate_different_methods():
    """Compare different constraint methods."""
    print("\n" + "=" * 60)
    print("Comparing Constraint Methods")
    print("=" * 60)
    
    # Generate data
    X, _ = make_blobs(n_samples=400, centers=4, n_features=2, random_state=42)
    gt_centroids = KMeans(n_clusters=4, random_state=42).fit(X).cluster_centers_
    
    # Normalize and split
    X_normalized = normalize(X, fixed=True)
    value_lists = shuffle_and_split(X_normalized, 2, random_state=42)
    
    methods = ['none', 'diagonal_then_frac', 'frac_stay']
    
    print(f"\n{'Method':<20} {'NICV':<12} {'MSE':<12} {'Unassigned':<10}")
    print("-" * 60)
    
    for method in methods:
        params = Params(
            seed=42, data_size=400, dim=2, k=4, iters=6,
            num_clients=2, eps=1.0, dp='gaussiananalytic',
            method=method, post='fold', fixed=True
        )
        
        centroids, unassigned = local_proto(value_lists, params, method="masked")
        centroids = unscale(centroids)
        
        metrics_dict = evaluate(centroids, X, gt_centroids, metrics=["nicv", "mse"])
        nicv = metrics_dict['Normalized Intra-cluster Variance (NICV)']
        mse = metrics_dict['Mean Squared Error']
        
        print(f"{method:<20} {nicv:<12.4f} {mse:<12.4f} {unassigned:<10}")


if __name__ == "__main__":
    run_fastlloyd_protocol_example()
    demonstrate_evaluation_metrics()
    demonstrate_different_methods()
    
    print("\n" + "=" * 60)
    print("All FastLloyd protocol examples completed!")
    print("=" * 60)
