"""Random graph utilities."""
from __future__ import annotations

import numpy as np


def simple_ER_graph(n: int, p: float, rng: np.random.Generator, weights: str = "uniform") -> np.ndarray:
    """Generate a symmetric weighted Erdős–Rényi graph adjacency matrix."""
    mask = rng.uniform(size=(n, n))
    upper = np.triu(mask < p, k=1)
    if weights == "uniform":
        w = rng.uniform(0.1, 1.0, size=(n, n))
    else:
        w = rng.normal(loc=1.0, scale=0.2, size=(n, n))
    adj = np.zeros((n, n))
    adj[upper] = w[upper]
    adj = adj + adj.T
    return adj
