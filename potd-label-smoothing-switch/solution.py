import numpy as np
def label_smoothing(logits, y, eps):
    C = len(logits)
    q = np.full(C, eps / C)
    q[y] += 1 - eps
    z = logits - logits.max()
    return q, float(-(q * (z - np.log(np.exp(z).sum()))).sum())
    pass
