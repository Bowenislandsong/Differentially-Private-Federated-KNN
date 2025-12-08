"""Advanced usage example showing privacy-utility tradeoffs.

This example demonstrates the impact of different privacy budgets and
constraint methods on clustering utility.
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from dpfknn import DPFederatedKMeans


def compare_privacy_budgets():
    """Compare clustering quality at different privacy budgets."""
    print("=" * 60)
    print("Privacy-Utility Tradeoff Analysis")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    X, y_true = make_blobs(
        n_samples=500,
        centers=5,
        n_features=2,
        cluster_std=0.8,
        random_state=42
    )
    
    # Privacy budgets to test
    epsilons = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("\nTesting different privacy budgets...")
    print(f"{'Epsilon':<10} {'ARI':<10} {'Silhouette':<12} {'Inertia':<12} {'Iterations':<10}")
    print("-" * 60)
    
    results = []
    for eps in epsilons:
        if eps == 0.0:
            # No privacy
            kmeans = DPFederatedKMeans(
                n_clusters=5,
                n_clients=3,
                random_state=42
            )
        else:
            # With privacy
            kmeans = DPFederatedKMeans(
                n_clusters=5,
                n_clients=3,
                epsilon=eps,
                dp_mechanism='gaussiananalytic',
                constraint_method='diagonal_then_frac',
                post_processing='fold',
                random_state=42
            )
        
        kmeans.fit(X)
        labels = kmeans.labels_
        
        # Calculate metrics
        ari = adjusted_rand_score(y_true, labels)
        try:
            sil = silhouette_score(X, labels)
        except:
            sil = 0.0
        
        print(f"{eps:<10.1f} {ari:<10.4f} {sil:<12.4f} {kmeans.inertia_:<12.2f} {kmeans.n_iter_:<10}")
        results.append({
            'epsilon': eps,
            'ari': ari,
            'silhouette': sil,
            'inertia': kmeans.inertia_,
            'iterations': kmeans.n_iter_
        })
    
    return results


def compare_constraint_methods():
    """Compare different constraint methods."""
    print("\n" + "=" * 60)
    print("Constraint Method Comparison")
    print("=" * 60)
    
    # Generate data
    X, y_true = make_blobs(
        n_samples=400,
        centers=4,
        n_features=2,
        cluster_std=0.6,
        random_state=42
    )
    
    methods = ['none', 'diagonal_then_frac', 'frac_stay']
    epsilon = 1.0
    
    print(f"\nUsing epsilon = {epsilon}")
    print(f"{'Method':<20} {'ARI':<10} {'Inertia':<12} {'Iterations':<10}")
    print("-" * 60)
    
    for method in methods:
        kmeans = DPFederatedKMeans(
            n_clusters=4,
            n_clients=2,
            epsilon=epsilon,
            dp_mechanism='gaussiananalytic',
            constraint_method=method,
            post_processing='fold',
            random_state=42
        )
        
        kmeans.fit(X)
        labels = kmeans.labels_
        ari = adjusted_rand_score(y_true, labels)
        
        print(f"{method:<20} {ari:<10.4f} {kmeans.inertia_:<12.2f} {kmeans.n_iter_:<10}")


def compare_post_processing():
    """Compare different post-processing methods."""
    print("\n" + "=" * 60)
    print("Post-processing Method Comparison")
    print("=" * 60)
    
    # Generate data
    X, y_true = make_blobs(
        n_samples=400,
        centers=4,
        n_features=2,
        cluster_std=0.6,
        random_state=42
    )
    
    post_methods = ['none', 'fold', 'truncate']
    epsilon = 1.0
    
    print(f"\nUsing epsilon = {epsilon}")
    print(f"{'Post-processing':<15} {'ARI':<10} {'Inertia':<12} {'Iterations':<10}")
    print("-" * 60)
    
    for post in post_methods:
        kmeans = DPFederatedKMeans(
            n_clusters=4,
            n_clients=2,
            epsilon=epsilon,
            dp_mechanism='gaussiananalytic',
            constraint_method='diagonal_then_frac',
            post_processing=post,
            random_state=42
        )
        
        kmeans.fit(X)
        labels = kmeans.labels_
        ari = adjusted_rand_score(y_true, labels)
        
        print(f"{post:<15} {ari:<10.4f} {kmeans.inertia_:<12.2f} {kmeans.n_iter_:<10}")


def compare_dp_mechanisms():
    """Compare Laplace vs Gaussian mechanisms."""
    print("\n" + "=" * 60)
    print("DP Mechanism Comparison")
    print("=" * 60)
    
    # Generate data
    X, y_true = make_blobs(
        n_samples=400,
        centers=4,
        n_features=2,
        cluster_std=0.6,
        random_state=42
    )
    
    mechanisms = ['laplace', 'gaussiananalytic']
    epsilon = 1.0
    
    print(f"\nUsing epsilon = {epsilon}")
    print(f"{'Mechanism':<20} {'ARI':<10} {'Inertia':<12} {'Iterations':<10}")
    print("-" * 60)
    
    for mech in mechanisms:
        kmeans = DPFederatedKMeans(
            n_clusters=4,
            n_clients=2,
            epsilon=epsilon,
            dp_mechanism=mech,
            constraint_method='diagonal_then_frac',
            post_processing='fold',
            random_state=42
        )
        
        kmeans.fit(X)
        labels = kmeans.labels_
        ari = adjusted_rand_score(y_true, labels)
        
        print(f"{mech:<20} {ari:<10.4f} {kmeans.inertia_:<12.2f} {kmeans.n_iter_:<10}")


def scalability_test():
    """Test scalability with different data sizes."""
    print("\n" + "=" * 60)
    print("Scalability Test")
    print("=" * 60)
    
    import time
    
    data_sizes = [100, 500, 1000, 2000]
    
    print(f"\n{'Samples':<10} {'Features':<10} {'Time (s)':<12} {'Inertia':<12}")
    print("-" * 60)
    
    for n_samples in data_sizes:
        X, _ = make_blobs(
            n_samples=n_samples,
            centers=5,
            n_features=10,
            random_state=42
        )
        
        kmeans = DPFederatedKMeans(
            n_clusters=5,
            n_clients=3,
            epsilon=1.0,
            dp_mechanism='gaussiananalytic',
            random_state=42
        )
        
        start_time = time.time()
        kmeans.fit(X)
        elapsed = time.time() - start_time
        
        print(f"{n_samples:<10} {X.shape[1]:<10} {elapsed:<12.4f} {kmeans.inertia_:<12.2f}")


if __name__ == "__main__":
    # Run all comparison analyses
    compare_privacy_budgets()
    compare_constraint_methods()
    compare_post_processing()
    compare_dp_mechanisms()
    scalability_test()
    
    print("\n" + "=" * 60)
    print("Advanced analysis completed!")
    print("=" * 60)
