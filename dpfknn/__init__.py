"""DP Federated KNN - Differentially Private Federated K-Means Clustering.

A scikit-learn compatible implementation of privacy-preserving federated k-means clustering.
Based on FastLloyd: https://github.com/D-Diaa/FastLloyd
"""

__version__ = "0.1.0"

from .estimator import DPFederatedKMeans
from .configs import Params

__all__ = ['DPFederatedKMeans', 'Params']
