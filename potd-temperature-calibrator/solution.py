import numpy as np


def temperature_calibrator(z: np.ndarray, tau: float) -> np.ndarray:
    """
    z: logits.
    tau: temperature, tau > 0.

    p = softmax(z / tau), using an overflow-safe softmax (subtract the
    max before exponentiating). Divide by tau, never multiply.

    Return p.
    """
    scaled = z / tau
    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)
    p = exp_values / np.sum(exp_values)

    return p
