import numpy as np
def valid_token_mean_loss(losses: np.ndarray, mask: np.ndarray) -> float:
    valid_count = mask.sum()
    if valid_count == 0:
        return 0.0

    return float((losses * mask).sum() / valid_count)
