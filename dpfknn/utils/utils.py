"""General utility functions.

Adapted from FastLloyd: https://github.com/D-Diaa/FastLloyd
"""

import numpy as np


def distance_matrix_squared(X, Y):
    """Compute pairwise squared Euclidean distances between two sets of points.
    
    Args:
        X (np.ndarray): First set of points, shape (n_samples, n_features)
        Y (np.ndarray): Second set of points, shape (n_centroids, n_features)
        
    Returns:
        np.ndarray: Matrix of squared distances, shape (n_samples, n_centroids)
    """
    return np.sum((X[:, np.newaxis] - Y) ** 2, axis=2)
