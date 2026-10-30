import numpy as np


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion; returns [lo, est, hi]."""
    if n == 0:
        return [0.0, 0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(p), float(min(1.0, c + h))]
