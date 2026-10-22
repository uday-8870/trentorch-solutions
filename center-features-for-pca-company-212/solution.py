import numpy as np


def solve(X):
    """Implement center-features-for-pca from the mathematical contract in README.md."""
    X = np.array(X, dtype=float)
    return (X - np.mean(X, axis=0)).tolist()
