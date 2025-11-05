"""Smooth problem generators (family A1–A6)."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from ..utils.linalg import ensure_spd, geometric_spectrum, haar_orthogonal


def _readme(problem: str, knobs: Dict[str, float]) -> str:
    knobs_txt = ", ".join(f"{k}={v}" for k, v in sorted(knobs.items()))
    return f"{problem} generated with {knobs_txt}"


def generate_a1_qp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Strongly-convex quadratic programme.

    Parameters
    ----------
    seed:
        RNG seed.
    knobs:
        Must contain ``n`` and ``kappa``.  ``kappa`` is increased to ``1e6`` in
        ``extreme`` mode.
    extreme:
        When set the condition number target is clamped to at least ``1e6``.
    """

    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 128))
    kappa = float(knobs.get("kappa", 1e3))
    if extreme:
        kappa = max(kappa, 1e6)
    eigs = geometric_spectrum(n, 1.0, kappa)
    U = haar_orthogonal(n, rng)
    Q = ensure_spd(U @ np.diag(eigs) @ U.T)
    b = rng.normal(size=n)
    x_star = np.linalg.solve(Q, b)
    data = {"Q": Q, "b": b, "x_star": x_star}
    dims = {"n": n}
    used_knobs = {"n": n, "kappa": kappa}
    reference = {"x_star": x_star}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": {"cert_type": "trivial"},
        "reference": reference,
        "readme": _readme("A1_QP", used_knobs),
    }


def generate_a2_logreg(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """L2-regularised logistic regression synthetic dataset."""

    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 500))
    n = int(knobs.get("n", 50))
    rho = float(knobs.get("rho", 0.3))
    lam = float(knobs.get("lambda", 1e-2))
    snr = float(knobs.get("snr", 10.0))
    sparsity = max(1, int(knobs.get("sparsity", max(1, round(0.1 * n)))))
    if extreme:
        rho = max(min(rho, 0.99), 0.9)
        snr = max(snr, 50.0)
    cov = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    A = rng.multivariate_normal(np.zeros(n), cov, size=m)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A = A / col_norms
    idx = rng.choice(n, sparsity, replace=False)
    x_true = np.zeros(n)
    x_true[idx] = rng.normal(scale=1.0, size=sparsity)
    x_true /= np.linalg.norm(x_true) + 1e-12
    x_true *= math.sqrt(snr)
    logits = A @ x_true
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs) * 2 - 1
    data = {"A": A, "y": y.astype(float), "lambda": lam, "x_true": x_true}
    dims = {"m": m, "n": n}
    used_knobs = {"m": m, "n": n, "rho": rho, "lambda": lam, "snr": snr, "sparsity": sparsity}
    reference = {"x_true": x_true}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": {"cert_type": "trivial"},
        "reference": reference,
        "readme": _readme("A2_LogReg", used_knobs),
    }


def generate_a3_rosenbrock(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Generate parameters for the Rosenbrock chain."""

    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 100))
    init_sigma = float(knobs.get("init_sigma", 1.0))
    if extreme:
        init_sigma = max(init_sigma, 5.0)
    x0 = np.ones(n) + rng.normal(scale=init_sigma, size=n)
    data = {"n": n, "x0": x0, "init_sigma": init_sigma}
    dims = {"n": n}
    used_knobs = {"n": n, "init_sigma": init_sigma}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": {"cert_type": "trivial"},
        "reference": {},
        "readme": _readme("A3_Rosenbrock", used_knobs),
    }


def _build_ecqp_system(rng: np.random.Generator, n: int, p: int, kappa: float, extreme: bool) -> Tuple[np.ndarray, np.ndarray]:
    eigs = geometric_spectrum(n, 1.0, kappa)
    U = haar_orthogonal(n, rng)
    Q = ensure_spd(U @ np.diag(eigs) @ U.T)
    A = rng.normal(size=(p, n))
    if extreme:
        s = np.geomspace(1e-6, 1.0, num=min(p, n))
        u, _ = np.linalg.qr(A.T)
        A = (u[:, :p] * s) @ haar_orthogonal(p, rng).T
    return Q, A


def generate_a4_ecqp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Equality-constrained quadratic programme with feasible witness."""

    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 200))
    p = int(knobs.get("p", 20))
    kappa = float(knobs.get("kappa", 1e4))
    if extreme:
        kappa = max(kappa, 1e7)
    Q, A = _build_ecqp_system(rng, n, p, kappa, extreme)
    x_feas = rng.normal(size=n)
    d = A @ x_feas
    b = rng.normal(size=n)
    data = {"Q": Q, "A": A, "b": b, "d": d}
    dims = {"n": n, "p": p}
    used_knobs = {"n": n, "p": p, "kappa": kappa}
    witness = {"cert_type": "primal", "x_feas": x_feas.tolist()}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": witness,
        "reference": {},
        "readme": _readme("A4_ECQP", used_knobs),
    }


def generate_a5_trs(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Trust-region subproblem with controllable indefiniteness."""

    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 100))
    neg_ratio = float(knobs.get("neg_ratio", 0.2))
    delta = float(knobs.get("delta", 1.0))
    theta = float(knobs.get("theta", 75.0))
    if extreme:
        neg_ratio = min(max(neg_ratio, 0.4), 0.9)
        theta = max(theta, 89.0)
    k = max(1, int(round(neg_ratio * n)))
    eigs = np.linspace(0.1, 10.0, num=n)
    eigs[:k] *= -1
    U = haar_orthogonal(n, rng)
    H = U @ np.diag(eigs) @ U.T
    vmin = U[:, 0]
    g = rng.normal(size=n)
    g -= (g @ vmin) * vmin
    g /= np.linalg.norm(g) + 1e-12
    g = math.cos(math.radians(theta)) * g + math.sin(math.radians(theta)) * vmin
    g *= np.linalg.norm(rng.normal(size=n))
    witness = {"cert_type": "primal", "x_feas": np.zeros(n).tolist(), "radius": delta}
    data = {"H": H, "g": g, "delta": delta}
    dims = {"n": n}
    used_knobs = {"n": n, "neg_ratio": neg_ratio, "delta": delta, "theta": theta}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": witness,
        "reference": {},
        "readme": _readme("A5_TRS", used_knobs),
    }


def generate_a6_boxqp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Box-constrained indefinite quadratic programme."""

    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 200))
    width = float(knobs.get("width", 5.0))
    neg_strength = float(knobs.get("neg_strength", 5.0))
    if extreme:
        neg_strength = max(neg_strength, 20.0)
        width = min(width, 1.0)
    eigs = np.linspace(-neg_strength, neg_strength, num=n)
    U = haar_orthogonal(n, rng)
    H = U @ np.diag(eigs) @ U.T
    c = rng.normal(size=n)
    l = -width * np.ones(n)
    u = width * np.ones(n)
    witness = {"cert_type": "primal", "x_feas": (0.5 * (l + u)).tolist()}
    data = {"H": H, "c": c, "l": l, "u": u}
    dims = {"n": n}
    used_knobs = {"n": n, "width": width, "neg_strength": neg_strength}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": witness,
        "reference": {},
        "readme": _readme("A6_BoxQP", used_knobs),
    }
