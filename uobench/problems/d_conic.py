"""Conic programme generators (family D1–D3)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..utils.linalg import haar_orthogonal
from ..utils.stats import calibrate_sigma_for_snr


def _readme(problem: str, knobs: Dict[str, float]) -> str:
    return f"{problem} generated with " + ", ".join(f"{k}={v}" for k, v in sorted(knobs.items()))


def generate_d1_socp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """SOCP robust regression instance."""

    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 200))
    n = int(knobs.get("n", 50))
    R = float(knobs.get("R", 1.0))
    snr = float(knobs.get("snr", 20.0))
    cond_target = float(knobs.get("cond", 1e3))
    if extreme:
        cond_target = max(cond_target, 1e6)
        snr = max(snr, 80.0)
    r = min(m, n)
    U = haar_orthogonal(m, rng)[:, :r]
    V = haar_orthogonal(n, rng)[:r, :]
    singular = np.geomspace(1.0 / cond_target, 1.0, num=r)
    A = U @ (singular[:, None] * V)
    x_true = rng.normal(size=n)
    x_true = x_true / (np.linalg.norm(x_true) + 1e-12) * (0.5 * R)
    y_clean = A @ x_true
    sigma = calibrate_sigma_for_snr(y_clean, m, snr)
    y = y_clean + rng.normal(scale=sigma, size=m)
    data = {"A": A, "y": y, "R": R, "y_clean": y_clean, "x_true": x_true}
    dims = {"m": m, "n": n}
    used_knobs = {"m": m, "n": n, "R": R, "snr": snr, "cond": cond_target}
    witness = {"cert_type": "primal", "x_feas": np.zeros(n).tolist(), "t": float(np.linalg.norm(y))}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": witness,
        "reference": {"x_true": x_true},
        "readme": _readme("D1_SOCP", used_knobs),
    }


def generate_d2_bp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Basis pursuit with sparse ground truth."""

    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 100))
    n = int(knobs.get("n", 200))
    sparsity = max(1, int(knobs.get("sparsity", 0.1 * n)))
    rho = float(knobs.get("rho", 0.0))
    if extreme:
        rho = max(rho, 0.95)
    cov = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    A = rng.multivariate_normal(np.zeros(n), cov, size=m)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A = A / col_norms
    x_true = np.zeros(n)
    idx = rng.choice(n, sparsity, replace=False)
    x_true[idx] = rng.normal(size=sparsity)
    y = A @ x_true
    data = {"A": A, "y": y, "x_true": x_true}
    dims = {"m": m, "n": n}
    used_knobs = {"m": m, "n": n, "sparsity": sparsity, "rho": rho}
    witness = {"cert_type": "primal", "x_feas": x_true.tolist()}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": witness,
        "reference": {"x_true": x_true},
        "readme": _readme("D2_BP", used_knobs),
    }


def generate_d3_sdp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """MaxCut SDP relaxation data."""

    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 40))
    p = float(knobs.get("p", 0.3))
    if extreme:
        p = min(max(p, 0.7), 0.9)
    weights = rng.uniform(size=(n, n))
    mask = rng.uniform(size=(n, n)) < p
    W = np.triu(weights * mask, 1)
    W = W + W.T
    degrees = np.sum(W, axis=1)
    L = np.diag(degrees) - W
    data = {"L": L, "weights": W, "degrees": degrees}
    dims = {"n": n}
    used_knobs = {"n": n, "p": p}
    witness = {"cert_type": "primal", "X": "identity"}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": witness,
        "reference": {},
        "readme": _readme("D3_SDP", used_knobs),
    }
