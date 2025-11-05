"""Semismooth-friendly variational inequality and complementarity generators."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..utils.linalg import ensure_spd, geometric_spectrum, haar_orthogonal


def _readme(problem: str, knobs: Dict[str, float]) -> str:
    knobs_txt = ", ".join(f"{k}={v}" for k, v in sorted(knobs.items()))
    return f"{problem} generated with {knobs_txt}"


def generate_c1_vi(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Linear monotone VI over a box or simplex."""

    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 100))
    mu = float(knobs.get("mu", 1e-3))
    cond_target = float(knobs.get("cond", 1e3))
    geometry = knobs.get("set", "box")
    if extreme:
        mu = min(mu, 1e-6)
        cond_target = max(cond_target, 1e6)
    eigs = geometric_spectrum(n, mu, mu * cond_target)
    U = haar_orthogonal(n, rng)
    Q = ensure_spd(U @ np.diag(eigs) @ U.T)
    c = rng.normal(size=n)
    if geometry == "simplex":
        l = np.zeros(n)
        u = np.ones(n)
    else:
        width = float(knobs.get("width", 1.0))
        l = -width * np.ones(n)
        u = width * np.ones(n)
        geometry = "box"
    witness = {"cert_type": "primal", "x_feas": np.zeros(n).tolist(), "geometry": geometry}
    data = {"Q": Q, "c": c, "l": l, "u": u, "geometry": geometry}
    dims = {"n": n}
    used_knobs = {"n": n, "mu": mu, "cond": cond_target, "geometry": geometry}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": witness,
        "reference": {},
        "readme": _readme("C1_VI", used_knobs),
    }


def generate_c2_lcp(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Linear complementarity problem with tunable δ-gap."""

    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 100))
    delta = float(knobs.get("delta", 1e-2))
    cond_target = float(knobs.get("cond", 1e3))
    if extreme:
        delta = min(delta, 1e-6)
        cond_target = max(cond_target, 1e5)
    W = rng.normal(size=(n, n))
    M = W.T @ W
    eigs = geometric_spectrum(n, delta, delta * cond_target)
    U = haar_orthogonal(n, rng)
    M = U @ np.diag(eigs) @ U.T
    q = np.abs(rng.normal(size=n))
    witness = {"cert_type": "complementarity", "z": np.zeros(n).tolist(), "q_nonneg": True, "delta": delta}
    data = {"M": M, "q": q, "delta": delta}
    dims = {"n": n}
    used_knobs = {"n": n, "delta": delta, "cond": cond_target}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": witness,
        "reference": {},
        "readme": _readme("C2_LCP", used_knobs),
    }


def generate_c3_mpcc(seed: int, knobs: Dict, extreme: bool) -> Dict:
    """Mathematical program with complementarity constraints."""

    rng = np.random.default_rng(seed)
    n = int(knobs.get("n", 200))
    p = int(knobs.get("p", 50))
    b_scale = float(knobs.get("b_scale", 0.1))
    if extreme:
        b_scale = min(b_scale, 1e-3)
    A = rng.normal(size=(p, n))
    b = np.abs(rng.normal(scale=b_scale, size=p))
    witness = {"cert_type": "complementarity", "x": np.zeros(n).tolist(), "y": np.zeros(p).tolist(), "s": b.tolist()}
    data = {"A": A, "b": b}
    dims = {"n": n, "p": p}
    used_knobs = {"n": n, "p": p, "b_scale": b_scale}
    return {
        "data": data,
        "dims": dims,
        "knobs": used_knobs,
        "witness": witness,
        "reference": {},
        "readme": _readme("C3_MPCC", used_knobs),
    }
