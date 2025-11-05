"""Linear algebra helper routines used by the benchmark generators.

The utilities collected here focus on numerically robust primitives that are
reused across many problem families: drawing Haar orthogonal matrices,
constructing spectra with prescribed condition numbers, Toeplitz covariance
matrices, projections, and diagnostic helpers such as condition numbers and
null-space bases.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def haar_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-distributed orthogonal matrix of size ``n``.

    The routine performs a QR factorisation of a Gaussian random matrix and
    fixes the sign of the diagonal of ``R`` to ensure uniqueness.
    """

    z = rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diag(r))
    d[d == 0] = 1.0
    return q * d


def geometric_spectrum(n: int, sigma_min: float, sigma_max: float) -> np.ndarray:
    """Return ``n`` eigenvalues spanning ``[sigma_min, sigma_max]`` geometrically."""

    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.array([float(sigma_max)], dtype=float)
    return np.geomspace(float(sigma_min), float(sigma_max), num=n)


def toeplitz_corr(n: int, rho: float) -> np.ndarray:
    """Toeplitz covariance matrix with entries ``rho**|i-j|``."""

    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])
    return np.power(rho, diff, dtype=float)


def cond(mat: np.ndarray) -> float:
    """Return the 2-norm condition number of ``mat`` with defensive guards."""

    s = np.linalg.svd(mat, compute_uv=False)
    if s[-1] <= 0:
        return float(np.inf)
    return float(s[0] / s[-1])


def smin(mat: np.ndarray) -> float:
    """Smallest singular value of ``mat``."""

    s = np.linalg.svd(mat, compute_uv=False)
    return float(s[-1])


def nullspace(mat: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Compute a basis for the numerical null-space of ``mat``."""

    u, s, vh = np.linalg.svd(mat)
    tol = rtol * s[0] if s.size else rtol
    rank = int(np.sum(s > tol))
    return vh[rank:].T


def ensure_spd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix to the cone of SPD matrices."""

    sym = 0.5 * (matrix + matrix.T)
    w, v = np.linalg.eigh(sym)
    w = np.maximum(w, eps)
    return (v * w) @ v.T


def project_ball(x: np.ndarray, radius: float) -> np.ndarray:
    """Project a vector onto the Euclidean ball of radius ``radius``."""

    norm = np.linalg.norm(x)
    if norm <= radius:
        return x
    return x * (radius / norm)


def project_box(x: np.ndarray, l: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Project onto a box defined by lower/upper bounds."""

    return np.minimum(np.maximum(x, l), u)


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project onto the probability simplex using the sorting algorithm."""

    if v.ndim != 1:
        raise ValueError("Simplex projection expects a 1D vector")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u + (1 - cssv) / np.arange(1, n + 1) > 0)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def spectral_extrema(matrix: np.ndarray) -> Tuple[float, float]:
    """Return minimum and maximum eigenvalues of a symmetric matrix."""

    w = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    return float(w[0]), float(w[-1])
