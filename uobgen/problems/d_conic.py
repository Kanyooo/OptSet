"""Conic-form problems (D1-D3)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..utils import calibrate_sigma_for_snr, column_normalize, simple_ER_graph, toeplitz_corr


def _base_instance(problem_id: str, name: str, seed: int) -> Dict[str, Any]:
    return {"id": problem_id, "name": name, "seed": seed, "meta": {}, "data": {}, "reference": {}}


def _toeplitz_design(rng: np.random.Generator, m: int, n: int, rho: float) -> np.ndarray:
    Sigma = toeplitz_corr(n, rho)
    L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(n))
    return rng.normal(size=(m, n)) @ L.T


def generate_d1_socp(
    seed: int,
    m: int,
    n: int,
    rho: float = 0.0,
    snr: float = 20.0,
    R: float = 5.0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("D1_SOCP", "SOCP Robust Regression", seed)
    A = _toeplitz_design(rng, m, n, rho)
    A, _ = column_normalize(A)
    x_star = rng.normal(size=n)
    x_star *= min(1.0, R / (np.linalg.norm(x_star) + 1e-12))
    y_clean = A @ x_star
    sigma = calibrate_sigma_for_snr(y_clean, m, snr)
    y = y_clean + rng.normal(scale=sigma, size=m)
    t_ref = np.linalg.norm(y_clean - A @ x_star)
    instance["meta"] = {
        "family": "conic",
        "dims": {"m": m, "n": n},
        "knobs": {"rho": float(rho), "snr": float(snr), "R": float(R)},
        "diagnostics": {"sigma": float(sigma)},
        "has_reference": True,
    }
    instance["data"] = {"A": A, "y": y, "R": R}
    instance["reference"] = {"x_star": x_star, "t_star": t_ref}
    return instance


def generate_d2_basis_pursuit(
    seed: int,
    m: int,
    n: int,
    rho: float = 0.0,
    sparsity: float = 0.05,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("D2_BP", "Basis Pursuit", seed)
    A = _toeplitz_design(rng, m, n, rho)
    A, _ = column_normalize(A)
    s = max(1, int(sparsity * n))
    support = rng.choice(n, size=s, replace=False)
    x_star = np.zeros(n)
    x_star[support] = rng.normal(size=s)
    y = A @ x_star
    instance["meta"] = {
        "family": "conic",
        "dims": {"m": m, "n": n},
        "knobs": {"rho": float(rho), "sparsity": float(s / n)},
        "diagnostics": {"rank": int(np.linalg.matrix_rank(A))},
        "has_reference": True,
    }
    instance["data"] = {"A": A, "y": y}
    instance["reference"] = {"x_star": x_star}
    return instance


def generate_d3_sdp(seed: int, n: int, p: float = 0.1, weights: str = "uniform") -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("D3_SDP", "MaxCut SDP Relaxation", seed)
    W = simple_ER_graph(n, p, rng, weights=weights)
    degrees = np.sum(W, axis=1)
    L = np.diag(degrees) - W
    instance["meta"] = {
        "family": "conic",
        "dims": {"n": n},
        "knobs": {"p": float(p), "weights": weights},
        "diagnostics": {"avg_degree": float(np.mean(degrees))},
        "has_reference": False,
    }
    instance["data"] = {"L": L, "W": W}
    return instance


PROBLEMS = {
    "D1_SOCP": generate_d1_socp,
    "D2_BP": generate_d2_basis_pursuit,
    "D3_SDP": generate_d3_sdp,
}
