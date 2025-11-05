"""Smooth problem generators (A1-A6)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..utils import (
    column_normalize,
    geometric_spectrum,
    haar_orthogonal,
    toeplitz_corr,
)


def _base_instance(problem_id: str, name: str, seed: int) -> Dict[str, Any]:
    return {"id": problem_id, "name": name, "seed": seed, "meta": {}, "data": {}, "reference": {}}


def generate_a1_qp(seed: int, n: int, kappa: float, sparsify: bool = False, density: float = 0.2) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("A1_QP", "Strongly Convex Quadratic", seed)
    U = haar_orthogonal(n, rng)
    eigs = geometric_spectrum(n, 1.0, kappa)
    Q = (U * eigs) @ U.T
    if sparsify:
        mask = rng.uniform(size=Q.shape) < density
        Q = Q * mask
        Q = (Q + Q.T) / 2 + np.eye(n) * 1e-6
    b = rng.normal(size=n)
    x_star = np.linalg.solve(Q, b)
    spectrum = np.linalg.eigvalsh(Q)
    instance["meta"] = {
        "family": "smooth-unconstrained",
        "dims": {"n": n},
        "knobs": {"kappa": float(kappa), "sparsify": sparsify},
        "diagnostics": {
            "lambda_max": float(np.max(spectrum)),
            "lambda_min": float(np.min(spectrum)),
            "cond": float(np.max(spectrum) / np.min(spectrum)),
        },
        "has_reference": True,
    }
    instance["data"] = {"Q": Q, "b": b, "eigs": eigs, "U": U}
    instance["reference"] = {"x_star": x_star}
    return instance


def generate_a2_logreg(
    seed: int,
    m: int,
    n: int,
    rho: float,
    snr: float = 20.0,
    lam: float = 1e-3,
    sparsity: float | None = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("A2_LogReg", "L2-regularized Logistic Regression", seed)
    Sigma = toeplitz_corr(n, rho)
    L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(n))
    A = rng.normal(size=(m, n)) @ L.T
    A, norms = column_normalize(A)
    s = max(1, int((sparsity or 0.1) * n))
    support = rng.choice(n, size=s, replace=False)
    x_star = np.zeros(n)
    x_star[support] = rng.normal(scale=1.0, size=s)
    logits = A @ x_star
    var = np.var(logits)
    if var > 0:
        scale = np.sqrt(snr / (var + 1e-9))
        logits *= scale
        x_star *= scale
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs)
    y = np.where(y == 0, -1, 1)
    instance["meta"] = {
        "family": "smooth",
        "dims": {"m": m, "n": n},
        "knobs": {"rho": float(rho), "snr": float(snr), "lambda": float(lam)},
        "diagnostics": {"column_norm_mean": float(np.mean(norms)), "sparsity": float(s / n)},
        "has_reference": True,
    }
    instance["data"] = {"A": A, "y": y, "lambda": lam}
    instance["reference"] = {"x_star": x_star}
    return instance


def generate_a3_rosenbrock(seed: int, n: int, init_sigma: float = 1.0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("A3_Rosenbrock", "Rosenbrock Chain", seed)
    x0 = 1 + init_sigma * rng.normal(size=n)
    instance["meta"] = {
        "family": "smooth-nonconvex",
        "dims": {"n": n},
        "knobs": {"init_sigma": float(init_sigma)},
        "diagnostics": {"x0_norm": float(np.linalg.norm(x0))},
        "has_reference": False,
    }
    instance["data"] = {"x0": x0}
    return instance


def generate_a4_ecqp(seed: int, n: int, p: int, kappa_Q: float = 1e3) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("A4_ECQP", "Equality-constrained QP", seed)
    U = haar_orthogonal(n, rng)
    eigs = geometric_spectrum(n, 1.0, kappa_Q)
    Q = (U * eigs) @ U.T
    b = rng.normal(size=n)
    x_dag = rng.normal(size=n)
    A = rng.normal(size=(p, n))
    U_a, _, Vt = np.linalg.svd(A, full_matrices=False)
    A = U_a @ Vt
    d = A @ x_dag
    KKT = np.block([[Q, A.T], [A, np.zeros((p, p))]])
    rhs = np.concatenate([b, d])
    sol = np.linalg.solve(KKT, rhs)
    x_star = sol[:n]
    instance["meta"] = {
        "family": "smooth-constrained",
        "dims": {"n": n, "p": p},
        "knobs": {"kappa_Q": float(kappa_Q)},
        "diagnostics": {"residual": float(np.linalg.norm(A @ x_star - d))},
        "has_reference": False,
    }
    instance["data"] = {"Q": Q, "b": b, "A": A, "d": d}
    instance["reference"] = {"x_feasible": x_dag, "x_star": x_star}
    return instance


def generate_a5_trs(
    seed: int,
    n: int,
    neg_ratio: float = 0.1,
    theta_deg: float = 75.0,
    delta: float | None = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("A5_TRS", "Trust-region Subproblem", seed)
    neg_count = max(1, int(neg_ratio * n))
    pos_count = n - neg_count
    pos_eigs = np.linspace(1.0, 10.0, pos_count) if pos_count > 0 else np.array([])
    neg_eigs = -np.linspace(0.1, 1.0, neg_count)
    eigs = np.concatenate([pos_eigs, neg_eigs])
    U = haar_orthogonal(n, rng)
    H = (U * eigs) @ U.T
    v_min = U[:, np.argmin(eigs)]
    g_raw = rng.normal(size=n)
    v_min_norm = np.linalg.norm(v_min)
    proj = np.dot(g_raw, v_min) / (np.linalg.norm(g_raw) * (v_min_norm + 1e-12))
    current_angle = np.degrees(np.arccos(np.clip(abs(proj), -1, 1)))
    scale = np.linalg.norm(g_raw)
    target_angle = theta_deg
    if current_angle < target_angle:
        g = g_raw + (np.tan(np.radians(target_angle)) - np.tan(np.radians(current_angle))) * v_min
    else:
        g = g_raw
    Delta = delta or np.sqrt(n)
    instance["meta"] = {
        "family": "smooth-nonconvex",
        "dims": {"n": n},
        "knobs": {"neg_ratio": float(neg_ratio), "theta": float(theta_deg), "delta": float(Delta)},
        "diagnostics": {"negative_fraction": float(neg_count / n)},
        "has_reference": False,
    }
    instance["data"] = {"H": H, "g": g, "Delta": Delta}
    return instance


def generate_a6_boxqp(
    seed: int,
    n: int,
    box_width: float = 1.0,
    negative_strength: float = 0.5,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("A6_BoxQP", "Box-constrained Quadratic", seed)
    U = haar_orthogonal(n, rng)
    pos = np.linspace(0.1, 10.0, n)
    neg = -negative_strength * rng.uniform(0.1, 1.0, size=n)
    eigs = pos + neg
    H = (U * eigs) @ U.T
    c = rng.normal(size=n)
    half_width = box_width / 2
    l = -half_width * np.ones(n)
    u = half_width * np.ones(n)
    instance["meta"] = {
        "family": "smooth-box",
        "dims": {"n": n},
        "knobs": {"box_width": float(box_width), "negative_strength": float(negative_strength)},
        "diagnostics": {"mean_eig": float(np.mean(eigs))},
        "has_reference": False,
    }
    instance["data"] = {"H": H, "c": c, "l": l, "u": u}
    return instance


PROBLEMS = {
    "A1_QP": generate_a1_qp,
    "A2_LogReg": generate_a2_logreg,
    "A3_Rosenbrock": generate_a3_rosenbrock,
    "A4_ECQP": generate_a4_ecqp,
    "A5_TRS": generate_a5_trs,
    "A6_BoxQP": generate_a6_boxqp,
}
