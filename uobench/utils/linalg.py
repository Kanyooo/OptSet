"""Linear algebra helper routines used across problems and diagnostics."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def haar_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a Haar-distributed orthogonal matrix via QR."""
    z = rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diag(r))
    q = q * d
    return q


def geometric_spectrum(n: int, sigma_min: float, sigma_max: float) -> np.ndarray:
    """Geometric sequence of eigenvalues achieving desired condition number."""
    if n == 1:
        return np.array([sigma_max])
    ratios = np.geomspace(sigma_min, sigma_max, num=n)
    return ratios


def toeplitz_corr(n: int, rho: float) -> np.ndarray:
    indices = np.arange(n)
    diff = np.abs(indices[:, None] - indices[None, :])
    return rho ** diff


def cond(mat: np.ndarray) -> float:
    s = np.linalg.svd(mat, compute_uv=False)
    return float(s[0] / s[-1])


def smin(mat: np.ndarray) -> float:
    s = np.linalg.svd(mat, compute_uv=False)
    return float(s[-1])


def nullspace(mat: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    u, s, vh = np.linalg.svd(mat)
    rank = np.sum(s > rtol * s[0])
    return vh[rank:].T


def ensure_spd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Symmetrize and project to SPD by shifting eigenvalues."""
    sym = 0.5 * (matrix + matrix.T)
    w, v = np.linalg.eigh(sym)
    w = np.maximum(w, eps)
    return (v * w) @ v.T


def project_ball(x: np.ndarray, radius: float) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm <= radius:
        return x
    return x * (radius / norm)


def project_box(x: np.ndarray, l: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, l), u)


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project onto probability simplex."""
    if np.sum(v) == 1 and np.all(v >= 0):
        return v
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u + (1 - cssv) / np.arange(1, n + 1) > 0)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w
