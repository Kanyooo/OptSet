"""Variational inequality, complementarity, and MPCC generators (C1-C3)."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..utils import geometric_spectrum, haar_orthogonal


def _base_instance(problem_id: str, name: str, seed: int) -> Dict[str, Any]:
    return {"id": problem_id, "name": name, "seed": seed, "meta": {}, "data": {}, "reference": {}}


def generate_c1_vi(
    seed: int,
    n: int,
    mu: float = 0.0,
    cond: float = 1e2,
    set_type: str = "box",
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("C1_VI", "Linear Variational Inequality", seed)
    sigma_min = max(mu, 1e-2)
    eigs = geometric_spectrum(n, sigma_min, sigma_min * cond)
    U = haar_orthogonal(n, rng)
    Q = (U * eigs) @ U.T
    c = rng.normal(size=n)
    if mu > 0:
        Q += mu * np.eye(n)
    if set_type == "box":
        l = -np.ones(n)
        u = np.ones(n)
        set_data = {"l": l, "u": u}
    else:
        l = np.zeros(n)
        u = np.ones(n)
        set_data = {"simplex_rhs": 1.0}
    instance["meta"] = {
        "family": "variational-inequality",
        "dims": {"n": n},
        "knobs": {"mu": float(mu), "cond": float(cond), "set_type": set_type},
        "diagnostics": {"lambda_min": float(np.min(np.linalg.eigvalsh(Q)))},
        "has_reference": False,
    }
    data = {"Q": Q, "c": c, "set_type": set_type}
    data.update(set_data)
    instance["data"] = data
    return instance


def generate_c2_lcp(seed: int, n: int, delta: float = 1e-3) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("C2_LCP", "Linear Complementarity Problem", seed)
    S = rng.normal(size=(n, n))
    M = S.T @ S + delta * np.eye(n)
    q = rng.normal(size=n)
    evals = np.linalg.eigvalsh(M)
    instance["meta"] = {
        "family": "complementarity",
        "dims": {"n": n},
        "knobs": {"delta": float(delta)},
        "diagnostics": {"lambda_min": float(np.min(evals)), "lambda_max": float(np.max(evals))},
        "has_reference": False,
    }
    instance["data"] = {"M": M, "q": q}
    return instance


def generate_c3_mpcc(seed: int, n: int, p: int, b_scale: float = 1e-1) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    instance = _base_instance("C3_MPCC", "Simple MPCC", seed)
    A = rng.normal(size=(p, n))
    x = rng.normal(size=n)
    b = b_scale * rng.normal(size=p)
    s = A @ x + b
    y = np.maximum(0, rng.normal(size=p))
    instance["meta"] = {
        "family": "mpcc",
        "dims": {"n": n, "p": p},
        "knobs": {"b_scale": float(b_scale)},
        "diagnostics": {"mean_abs_s": float(np.mean(np.abs(s)))},
        "has_reference": True,
    }
    instance["data"] = {"A": A, "b": b}
    instance["reference"] = {"x": x, "y": y, "s": s}
    return instance


PROBLEMS = {
    "C1_VI": generate_c1_vi,
    "C2_LCP": generate_c2_lcp,
    "C3_MPCC": generate_c3_mpcc,
}
