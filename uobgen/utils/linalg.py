"""Linear algebra helper utilities for UOBGen."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def haar_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Haar-distributed orthogonal matrix of size ``n``.

    The implementation samples a Gaussian matrix and performs QR decomposition
    with sign correction to ensure uniform distribution over the orthogonal group.
    """
    g = rng.normal(size=(n, n))
    q, r = np.linalg.qr(g)
    d = np.diag(np.sign(np.diag(r)))
    return q @ d


def geometric_spectrum(n: int, sigma_min: float, sigma_max: float) -> np.ndarray:
    """Return a geometric progression of eigenvalues.

    Parameters
    ----------
    n: dimension.
    sigma_min: smallest eigenvalue.
    sigma_max: largest eigenvalue.
    """
    if sigma_min <= 0:
        raise ValueError("sigma_min must be positive")
    if sigma_max < sigma_min:
        raise ValueError("sigma_max must be >= sigma_min")
    if n == 1:
        return np.array([sigma_max], dtype=float)
    ratio = (sigma_max / sigma_min) ** (1 / (n - 1))
    return sigma_min * ratio ** np.arange(n)


def toeplitz_corr(n: int, rho: float) -> np.ndarray:
    """Construct a Toeplitz correlation matrix with parameter ``rho``."""
    rho = float(rho)
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    return rho ** diff


def column_normalize(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Column-normalize matrix ``A``.

    Returns normalized matrix and original column norms.
    """
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0
    return A / norms, norms
