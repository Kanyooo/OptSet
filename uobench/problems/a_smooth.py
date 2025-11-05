"""Smooth problem generators (A1–A6)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..utils.linalg import haar_orthogonal, geometric_spectrum, ensure_spd


def generate_a1_qp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 50))
    kappa = float(knobs.get("kappa", 1e3))
    eigs = geometric_spectrum(n, 1.0, kappa)
    U = haar_orthogonal(n, rng)
    Q = U @ np.diag(eigs) @ U.T
    Q = ensure_spd(Q)
    b = rng.normal(size=n)
    x_star = np.linalg.solve(Q, b)
    data = {"Q": Q, "b": b, "x_star": x_star}
    dims = {"n": n}
    return {"data": data, "dims": dims, "witness": {}, "readme": f"A1_QP with n={n}, kappa={kappa}"}


def generate_a2_logreg(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    m = int(knobs.get("m", 200))
    n = int(knobs.get("n", 20))
    rho = float(knobs.get("rho", 0.3))
    lam = float(knobs.get("lambda", 0.1))
    cov = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    A = rng.multivariate_normal(np.zeros(n), cov, size=m)
    col_norms = np.linalg.norm(A, axis=0)
    A = A / (col_norms + 1e-12)
    sparsity = max(1, int(0.1 * n))
    idx = rng.choice(n, sparsity, replace=False)
    x_true = np.zeros(n)
    x_true[idx] = rng.normal(size=sparsity)
    logits = A @ x_true
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs) * 2 - 1
    data = {"A": A, "y": y, "lambda": lam, "x_true": x_true}
    dims = {"m": m, "n": n}
    return {"data": data, "dims": dims, "witness": {}, "readme": f"A2_LogReg m={m} n={n}"}


def generate_a3_rosenbrock(seed: int, knobs: Dict, extreme: bool) -> Dict:
    n = int(knobs.get("n", 20))
    sigma = float(knobs.get("init_sigma", 1.0))
    data = {"n": n, "init_sigma": sigma}
    dims = {"n": n}
    return {"data": data, "dims": dims, "witness": {}, "readme": f"Rosenbrock chain length {n}"}


def generate_a4_ecqp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 40))
    p = int(knobs.get("p", 5))
    kappa = float(knobs.get("kappa", 1e3))
    eigs = geometric_spectrum(n, 1.0, kappa)
    U = haar_orthogonal(n, rng)
    Q = U @ np.diag(eigs) @ U.T
    Q = ensure_spd(Q)
    x_feas = rng.normal(size=n)
    A = rng.normal(size=(p, n))
    d = A @ x_feas
    b = rng.normal(size=n)
    data = {"Q": Q, "A": A, "d": d, "b": b}
    dims = {"n": n, "p": p}
    witness = {"cert_type": "primal", "x_feas": x_feas.tolist()}
    return {"data": data, "dims": dims, "witness": witness, "readme": f"A4_ECQP with n={n}, p={p}"}


def generate_a5_trs(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 30))
    neg_ratio = float(knobs.get("neg_ratio", 0.2))
    delta = float(knobs.get("delta", 1.0))
    theta = float(knobs.get("theta", 75))
    eigs = np.linspace(-1, 1, n)
    neg_count = int(neg_ratio * n)
    eigs[:neg_count] *= -5
    U = haar_orthogonal(n, rng)
    H = U @ np.diag(eigs) @ U.T
    vmin = U[:, 0]
    g = rng.normal(size=n)
    # adjust angle roughly by projecting component
    g = g - (g @ vmin) * vmin
    if np.linalg.norm(g) < 1e-12:
        g = rng.normal(size=n)
    g = g * np.cos(np.deg2rad(theta)) + vmin * np.sin(np.deg2rad(theta))
    witness = {"cert_type": "primal", "x_feas": np.zeros(n).tolist()}
    data = {"H": H, "g": g, "delta": delta, "vmin": vmin}
    dims = {"n": n}
    return {"data": data, "dims": dims, "witness": witness, "readme": f"A5_TRS n={n}"}


def generate_a6_boxqp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 30))
    width = float(knobs.get("width", 2.0))
    eigs = np.linspace(-1, 1, n)
    U = haar_orthogonal(n, rng)
    H = U @ np.diag(eigs) @ U.T
    c = rng.normal(size=n)
    l = -width * np.ones(n)
    u = width * np.ones(n)
    x_center = 0.5 * (l + u)
    witness = {"cert_type": "primal", "x_feas": x_center.tolist()}
    data = {"H": H, "c": c, "l": l, "u": u}
    dims = {"n": n}
    return {"data": data, "dims": dims, "witness": witness, "readme": f"A6_BoxQP n={n}"}
