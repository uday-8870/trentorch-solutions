import numpy as np


def causal_mask_misfire(scores: np.ndarray) -> np.ndarray:
    """
    Row-wise, numerically stable softmax under a causal mask: for query
    row i, only key columns j <= i are visible. Forbidden entries (j > i)
    must be exactly 0.0 in the output.

    scores: shape (n, n).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]

    # Apply the causal mask BEFORE computing row maxima.
    masked_scores = np.where(
        np.tril(np.ones((n, n), dtype=bool)),
        scores,
        -np.inf
    )

    # Numerically stable row-wise softmax.
    row_max = np.max(masked_scores, axis=1, keepdims=True)
    exp_scores = np.exp(masked_scores - row_max)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Explicitly guarantee exact zeros at forbidden future positions.
    probabilities[np.triu_indices(n, k=1)] = 0.0

    return probabilities
