"""Variational inequality and complementarity generators (C1–C3)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..utils.linalg import haar_orthogonal, geometric_spectrum, ensure_spd


def generate_c1_vi(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 20))
    mu = float(knobs.get("mu", 0.01))
    rho = float(knobs.get("rho", 0.2))
    set_type = knobs.get("set", "box")
    eigs = geometric_spectrum(n, mu, mu + 1)
    Q = haar_orthogonal(n, rng) @ np.diag(eigs) @ haar_orthogonal(n, rng).T
    c = rng.normal(size=n)
    if set_type == "box":
        l = -np.ones(n)
        u = np.ones(n)
        witness = {"cert_type": "primal", "x_feas": np.zeros(n).tolist(), "set": "box"}
        data = {"Q": Q, "c": c, "l": l, "u": u}
    else:
        x = np.ones(n) / n
        witness = {"cert_type": "primal", "x_feas": x.tolist(), "set": "simplex"}
        data = {"Q": Q, "c": c, "simplex": True}
    dims = {"n": n}
    return {"data": data, "dims": dims, "witness": witness, "readme": "C1_VI"}


def generate_c2_lcp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 20))
    delta = float(knobs.get("delta", 0.01))
    S = rng.normal(size=(n, n))
    M = S.T @ S + delta * np.eye(n)
    q = np.abs(rng.normal(size=n))
    witness = {"cert_type": "complementarity", "z": np.zeros(n).tolist(), "q_nonneg": True}
    data = {"M": M, "q": q}
    dims = {"n": n}
    return {"data": data, "dims": dims, "witness": witness, "readme": "C2_LCP"}


def generate_c3_mpcc(seed: int, knobs: Dict, extreme: bool) -> Dict:
    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 20))
    p = int(knobs.get("p", 10))
    b_scale = float(knobs.get("b_scale", 0.1))
    A = rng.normal(size=(p, n))
    b = np.abs(rng.normal(size=p)) * b_scale
    witness = {"cert_type": "primal", "x_feas": np.zeros(n).tolist()}
    data = {"A": A, "b": b}
    dims = {"n": n, "p": p}
    return {"data": data, "dims": dims, "witness": witness, "readme": "C3_MPCC"}
