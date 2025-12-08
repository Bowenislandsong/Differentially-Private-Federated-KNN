"""Scikit-learn compatible estimator for DP Federated K-Means.

This module provides a scikit-learn compatible API for differentially private
federated k-means clustering based on the FastLloyd protocol.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .configs import Params
from .data_io import normalize, shuffle_and_split
from .parties import MaskedClient, UnmaskedClient, Server
from .utils import distance_matrix_squared


class DPFederatedKMeans(BaseEstimator, ClusterMixin):
    """Differentially Private Federated K-Means Clustering.
    
    This estimator implements privacy-preserving federated k-means clustering
    following the FastLloyd protocol. It supports:
    - Differential privacy via Laplace or Gaussian mechanisms
    - Secure aggregation via masked computation
    - Radius-constrained updates for improved utility
    - Post-processing methods (truncation and folding)
    
    The algorithm is suitable for horizontally partitioned data across multiple
    clients where privacy must be preserved.
    
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
        
    n_clients : int, default=2
        The number of clients in the federated setup.
        
    epsilon : float, default=0.0
        Privacy budget. If 0, no differential privacy is applied.
        
    dp_mechanism : str, default='none'
        Differential privacy mechanism to use.
        Options: 'none', 'laplace', 'gaussiananalytic'
        
    max_iter : int, default=6
        Maximum number of iterations of the k-means algorithm.
        
    constraint_method : str, default='none'
        Method for constraining centroid updates.
        Options: 'none', 'diagonal_then_frac', 'frac_stay'
        
    post_processing : str, default='none'
        Post-processing method for centroids.
        Options: 'none', 'fold', 'truncate'
        
    alpha : float, default=2.0
        Parameter for controlling update constraints.
        
    rho : float, default=0.225
        Privacy parameter for noise calibration.
        
    use_masking : bool, default=True
        Whether to use secure aggregation via masking.
        
    random_state : int, default=1337
        Random seed for reproducibility.
        
    fixed_point : bool, default=True
        Whether to use fixed-point arithmetic for secure computation.
        
    Attributes
    ----------
    cluster_centers_ : np.ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
        
    labels_ : np.ndarray of shape (n_samples,)
        Labels of each point.
        
    n_iter_ : int
        Number of iterations run.
        
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
        
    Examples
    --------
    >>> from dpfknn import DPFederatedKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = DPFederatedKMeans(n_clusters=2, random_state=0)
    >>> kmeans.fit(X)
    DPFederatedKMeans(n_clusters=2)
    >>> kmeans.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([0, 1])
    >>> kmeans.cluster_centers_
    array([[1., 2.],
           [10., 2.]])
           
    Notes
    -----
    This implementation follows the FastLloyd protocol described in:
    "FastLloyd: Federated, Accurate, Secure, and Tunable k-Means Clustering 
    with Differential Privacy" by Diaa et al., 2024.
    
    References
    ----------
    .. [1] https://github.com/D-Diaa/FastLloyd
    """
    
    def __init__(
        self,
        n_clusters=8,
        n_clients=2,
        epsilon=0.0,
        dp_mechanism='none',
        max_iter=6,
        constraint_method='none',
        post_processing='none',
        alpha=2.0,
        rho=0.225,
        use_masking=True,
        random_state=1337,
        fixed_point=True,
    ):
        self.n_clusters = n_clusters
        self.n_clients = n_clients
        self.epsilon = epsilon
        self.dp_mechanism = dp_mechanism
        self.max_iter = max_iter
        self.constraint_method = constraint_method
        self.post_processing = post_processing
        self.alpha = alpha
        self.rho = rho
        self.use_masking = use_masking
        self.random_state = random_state
        self.fixed_point = fixed_point
        
    def _validate_params(self):
        """Validate the parameters."""
        if self.n_clusters <= 0:
            raise ValueError(f"n_clusters must be > 0, got {self.n_clusters}")
        if self.n_clients <= 0:
            raise ValueError(f"n_clients must be > 0, got {self.n_clients}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")
        if self.dp_mechanism not in ['none', 'laplace', 'gaussiananalytic']:
            raise ValueError(f"dp_mechanism must be one of 'none', 'laplace', 'gaussiananalytic', got {self.dp_mechanism}")
        if self.constraint_method not in ['none', 'diagonal_then_frac', 'frac_stay']:
            raise ValueError(f"constraint_method must be one of 'none', 'diagonal_then_frac', 'frac_stay', got {self.constraint_method}")
        if self.post_processing not in ['none', 'fold', 'truncate']:
            raise ValueError(f"post_processing must be one of 'none', 'fold', 'truncate', got {self.post_processing}")
            
    def fit(self, X, y=None):
        """Compute k-means clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
            
        y : Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        
        # Validate and normalize input
        X = check_array(X, dtype=np.float64, copy=True)
        n_samples, n_features = X.shape
        
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}")
        
        # Normalize data to [-1, 1]
        X_normalized = normalize(X, fixed=self.fixed_point)
        
        # Create parameters
        params = Params(
            seed=self.random_state,
            data_size=n_samples,
            dim=n_features,
            k=self.n_clusters,
            iters=self.max_iter,
            alpha=self.alpha,
            num_clients=self.n_clients,
            rho=self.rho,
            eps=self.epsilon,
            dp=self.dp_mechanism,
            method=self.constraint_method,
            post=self.post_processing,
            fixed=self.fixed_point,
        )
        
        # Calculate optimal iterations if using DP
        if self.epsilon > 0:
            params.calculate_iters()
        
        # Split data among clients
        proportions = np.ones(self.n_clients) / self.n_clients
        value_lists = shuffle_and_split(X_normalized, self.n_clients, proportions)
        
        # Run federated clustering protocol
        centroids, unassigned = self._run_protocol(value_lists, params)
        
        # Store results
        self.cluster_centers_ = centroids
        self.n_iter_ = params.iters
        self._params = params
        
        # Compute labels and inertia on the full dataset
        self.labels_ = self.predict(X)
        self.inertia_ = self._compute_inertia(X, centroids, self.labels_)
        
        return self
        
    def _run_protocol(self, value_lists, params):
        """Run the federated clustering protocol.
        
        Parameters
        ----------
        value_lists : list of np.ndarray
            Data split among clients.
            
        params : Params
            Configuration parameters.
            
        Returns
        -------
        centroids : np.ndarray
            Final cluster centers.
            
        unassigned : int
            Number of unassigned points.
        """
        # Initialize clients and server
        if self.use_masking:
            clients = [MaskedClient(i, values, params) for i, values in enumerate(value_lists)]
        else:
            clients = [UnmaskedClient(i, values, params) for i, values in enumerate(value_lists)]
        
        server = Server(params)
        
        # Run iterations
        total_unassigned = 0
        for iteration in range(params.iters):
            # Update max distance for this iteration
            params.update_maxdist(iteration)
            
            # Client step: compute local statistics
            client_results = [client.step(params) for client in clients]
            sums = [result[0] for result in client_results]
            counts = [result[1] for result in client_results]
            unassigned_counts = [result[2] for result in client_results]
            total_unassigned = sum(unassigned_counts)
            
            # Server step: aggregate and add DP noise
            aggregated_sum, aggregated_count = server.step(sums, counts, params)
            
            # Client update: update centroids
            for client in clients:
                client.update(aggregated_sum, aggregated_count)
        
        # Return centroids from first client (all clients have the same centroids)
        return clients[0].centroids, total_unassigned
        
    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
            
        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, ['cluster_centers_'])
        X = check_array(X, dtype=np.float64)
        
        # Compute distances to cluster centers
        distances = distance_matrix_squared(X, self.cluster_centers_)
        labels = np.argmin(distances, axis=1)
        
        return labels
        
    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.
        
        Convenience method; equivalent to calling fit(X) followed by predict(X).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
            
        y : Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_
        
    def transform(self, X):
        """Transform X to a cluster-distance space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
            
        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_clusters)
            X transformed in the new space (distances to cluster centers).
        """
        check_is_fitted(self, ['cluster_centers_'])
        X = check_array(X, dtype=np.float64)
        
        # Return squared distances to all cluster centers
        return distance_matrix_squared(X, self.cluster_centers_)
        
    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
            
        y : Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        return self.fit(X, y).transform(X)
        
    def score(self, X, y=None):
        """Opposite of the value of X on the K-means objective.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.
            
        y : Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self, ['cluster_centers_'])
        X = check_array(X, dtype=np.float64)
        
        labels = self.predict(X)
        inertia = self._compute_inertia(X, self.cluster_centers_, labels)
        
        return -inertia
        
    def _compute_inertia(self, X, centers, labels):
        """Compute the sum of squared distances to cluster centers.
        
        Parameters
        ----------
        X : np.ndarray
            Data points.
            
        centers : np.ndarray
            Cluster centers.
            
        labels : np.ndarray
            Cluster assignments.
            
        Returns
        -------
        inertia : float
            Sum of squared distances.
        """
        inertia = 0.0
        for i in range(self.n_clusters):
            mask = labels == i
            if np.any(mask):
                cluster_points = X[mask]
                center = centers[i]
                inertia += np.sum((cluster_points - center) ** 2)
        return inertia
