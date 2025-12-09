"""Basic usage example for DPFederatedKMeans.

This example demonstrates how to use the DPFederatedKMeans estimator
following scikit-learn conventions.
"""

import numpy as np
from sklearn.datasets import make_blobs
from dpfknn import DPFederatedKMeans


def basic_example():
    """Basic example of using DPFederatedKMeans."""
    print("=" * 60)
    print("Basic Example: Clustering with DPFederatedKMeans")
    print("=" * 60)
    
    # Generate synthetic data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                           cluster_std=0.60, random_state=0)
    
    print(f"\nGenerated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"True number of clusters: 4")
    
    # Create and fit the model (no privacy)
    print("\n1. Clustering without differential privacy...")
    kmeans = DPFederatedKMeans(
        n_clusters=4,
        n_clients=2,
        random_state=42
    )
    kmeans.fit(X)
    
    print(f"   Number of iterations: {kmeans.n_iter_}")
    print(f"   Inertia: {kmeans.inertia_:.2f}")
    print(f"   Cluster centers shape: {kmeans.cluster_centers_.shape}")
    
    # Predict on new data
    X_new = np.array([[0, 0], [10, 10]])
    labels_new = kmeans.predict(X_new)
    print(f"\n   Predictions for new points: {labels_new}")
    
    # Clustering with differential privacy
    print("\n2. Clustering with differential privacy (Gaussian)...")
    dp_kmeans = DPFederatedKMeans(
        n_clusters=4,
        n_clients=2,
        epsilon=1.0,
        dp_mechanism='gaussiananalytic',
        random_state=42
    )
    dp_kmeans.fit(X)
    
    print(f"   Privacy budget (epsilon): {dp_kmeans.epsilon}")
    print(f"   Number of iterations: {dp_kmeans.n_iter_}")
    print(f"   Inertia: {dp_kmeans.inertia_:.2f}")
    
    # Using constraint methods for improved utility
    print("\n3. Clustering with DP and constraint methods...")
    dp_kmeans_constrained = DPFederatedKMeans(
        n_clusters=4,
        n_clients=2,
        epsilon=1.0,
        dp_mechanism='gaussiananalytic',
        constraint_method='diagonal_then_frac',
        post_processing='fold',
        random_state=42
    )
    dp_kmeans_constrained.fit(X)
    
    print(f"   Constraint method: {dp_kmeans_constrained.constraint_method}")
    print(f"   Post-processing: {dp_kmeans_constrained.post_processing}")
    print(f"   Number of iterations: {dp_kmeans_constrained.n_iter_}")
    print(f"   Inertia: {dp_kmeans_constrained.inertia_:.2f}")


def sklearn_compatibility_example():
    """Demonstrate scikit-learn compatibility."""
    print("\n" + "=" * 60)
    print("Scikit-learn Compatibility Example")
    print("=" * 60)
    
    # Generate data
    X, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=42)
    
    # Create model
    kmeans = DPFederatedKMeans(n_clusters=3, random_state=42)
    
    # Using fit_predict
    print("\n1. Using fit_predict()...")
    labels = kmeans.fit_predict(X)
    print(f"   Labels shape: {labels.shape}")
    print(f"   Unique labels: {np.unique(labels)}")
    
    # Using transform
    print("\n2. Using transform()...")
    distances = kmeans.transform(X[:5])
    print(f"   Distances shape: {distances.shape}")
    print(f"   Sample distances to clusters:\n{distances}")
    
    # Using score
    print("\n3. Using score()...")
    score = kmeans.score(X)
    print(f"   Score (negative inertia): {score:.2f}")
    
    # Using fit_transform
    print("\n4. Using fit_transform()...")
    kmeans_new = DPFederatedKMeans(n_clusters=3, random_state=42)
    distances = kmeans_new.fit_transform(X)
    print(f"   Transformed data shape: {distances.shape}")


def federated_setup_example():
    """Example showing different federated setups."""
    print("\n" + "=" * 60)
    print("Federated Setup Examples")
    print("=" * 60)
    
    # Generate data
    X, _ = make_blobs(n_samples=400, centers=5, n_features=3, random_state=42)
    
    # Different numbers of clients
    for n_clients in [2, 4, 8]:
        print(f"\n{n_clients} clients:")
        kmeans = DPFederatedKMeans(
            n_clusters=5,
            n_clients=n_clients,
            epsilon=0.5,
            dp_mechanism='gaussiananalytic',
            random_state=42
        )
        kmeans.fit(X)
        print(f"   Inertia: {kmeans.inertia_:.2f}")
        print(f"   Iterations: {kmeans.n_iter_}")


if __name__ == "__main__":
    # Run all examples
    basic_example()
    sklearn_compatibility_example()
    federated_setup_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
